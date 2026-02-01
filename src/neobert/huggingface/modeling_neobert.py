"""NeoBERT model implementation for HuggingFace Transformers.

This module provides the NeoBERT architecture compatible with the HuggingFace
Transformers library. It includes the base model, language modeling head, and
sequence classification variants.

Architecture Features:
- SwiGLU activation function for improved training dynamics
- Rotary Position Embeddings (RoPE) for better position encoding
- Pre-RMSNorm for improved training stability
- Flash Attention support for efficient long-context processing

Based on: https://github.com/facebookresearch/llama/blob/main/llama/model.py

NOTE: The training-time implementation lives in ``src/neobert/model/model.py`` and
uses xFormers for flash attention. This HF variant targets export/inference APIs
and may use different attention backends; keep the math consistent when editing.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import scaled_dot_product_attention

try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from transformers import (
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

from .rotary import apply_rotary_emb, precompute_freqs_cis


class DataCollatorWithPacking(DataCollatorForLanguageModeling):
    """Data collator with optional sequence packing for efficient training.

    This collator extends the standard MLM collator with the ability to pack
    multiple sequences into a single batch for more efficient GPU utilization
    when using Flash Attention.

    Args:
        pack_sequences: Whether to pack sequences for Flash Attention.
        **kwargs: Additional arguments passed to DataCollatorForLanguageModeling.
    """

    def __init__(self, pack_sequences: bool = False, **kwargs: Any) -> None:
        """Initialize the data collator.

        Args:
            pack_sequences: Whether to pack multiple sequences together.
            **kwargs: Additional arguments for the parent collator.
        """
        super().__init__(**kwargs)
        self.pack_sequences = pack_sequences

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Process a batch of examples.

        Args:
            batch: List of dictionaries containing tokenized sequences.

        Returns:
            Dictionary containing processed tensors ready for model input.
            If packing is enabled, includes cumulative sequence lengths.
        """
        if self.pack_sequences:
            # Add position_ids if not present
            if "position_ids" not in batch[0]:
                for item in batch:
                    item["position_ids"] = list(range(len(item["input_ids"])))

            # Pack the sequences into a single list
            input_ids_list = [item["input_ids"] for item in batch]
            position_ids_list = [item["position_ids"] for item in batch]
            seqlens = np.array([0] + [len(ids) for ids in input_ids_list])

            packed_batch = {
                "position_ids": np.concatenate(position_ids_list, axis=0),
                "input_ids": np.concatenate(input_ids_list, axis=0),
                "cu_seqlens": np.cumsum(seqlens),
                "max_seqlen": max(seqlens),
            }

            batch = super().__call__([packed_batch])
            batch["cu_seqlens"] = batch["cu_seqlens"].to(torch.int32).squeeze()
        else:
            batch = super().__call__(batch)
            batch["attention_mask"] = batch["attention_mask"].to(torch.bool)

        return batch


class NeoBERTConfig(PretrainedConfig):
    """Configuration class for NeoBERT model.

    This class stores the configuration of a NeoBERT model. It inherits from
    PretrainedConfig for compatibility with the HuggingFace ecosystem.

    Args:
        hidden_size: Dimensionality of encoder layers and pooler layer.
        num_hidden_layers: Number of hidden transformer layers.
        num_attention_heads: Number of attention heads in each attention layer.
        intermediate_size: Dimensionality of the feed-forward layer.
        embedding_init_range: Standard deviation for initializing embeddings.
        decoder_init_range: Standard deviation for initializing decoder weights.
        norm_eps: Epsilon value for layer normalization.
        vocab_size: Size of the vocabulary.
        pad_token_id: Token ID used for padding.
        max_length: Maximum sequence length the model can handle.
        **kwargs: Additional configuration parameters.

    Attributes:
        dim_head: Dimension of each attention head (computed from hidden_size).
    """

    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        norm_eps: float = 1e-06,
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        max_length: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Initialize the NeoBERT configuration.

        :param int hidden_size: Dimensionality of encoder layers.
        :param int num_hidden_layers: Number of transformer layers.
        :param int num_attention_heads: Number of attention heads per layer.
        :param int intermediate_size: Feed-forward hidden size.
        :param float embedding_init_range: Stddev for embedding initialization.
        :param float decoder_init_range: Stddev for decoder initialization.
        :param float norm_eps: Epsilon for normalization layers.
        :param int vocab_size: Vocabulary size.
        :param int pad_token_id: Padding token ID.
        :param int max_length: Maximum sequence length.
        :param Any kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by the number "
                f"of attention heads ({num_attention_heads})."
            )
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.kwargs = kwargs


class NeobertMLP(nn.Module):
    """SwiGLU-based MLP layer for NeoBERT.

    This implements the SwiGLU activation function which combines a gated
    linear unit with the SiLU (Swish) activation. The implementation follows
    the approach from LLaMA where two linear projections are concatenated
    for efficiency.

    Adapted from: transformers.models.llama.modeling_llama.LlamaMLP

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
        bias: Whether to include bias in linear layers.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, bias: bool = False
    ) -> None:
        """Initialize the MLP layer.

        Args:
            hidden_size: Dimension of input and output features.
            intermediate_size: Dimension of the intermediate layer.
            bias: Whether to use bias in linear projections.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Concatenated linear for w1 and w2 for efficiency
        self.w12 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=bias)
        self.w3 = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Split the concatenated projection into gate and value
        w1, w2 = self.w12(x).chunk(2, dim=-1)
        # Apply SwiGLU: SiLU(w1) * w2, then project back
        w3 = self.w3(self.act_fn(w1) * w2)
        return w3


class EncoderBlock(nn.Module):
    """Transformer encoder block with Pre-RMSNorm and SwiGLU activation.

    This block implements a transformer encoder layer with the following features:
    - Multi-head self-attention with RoPE
    - SwiGLU feed-forward network
    - Pre-normalization using RMSNorm
    - Support for Flash Attention and SDPA

    Args:
        config: NeoBERT configuration object.
    """

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the encoder block.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
            bias=False,
        )
        self.wo = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, bias=False
        )

        # Feedforward network - adjust size to be multiple of 8 for efficiency
        multiple_of = 8
        intermediate_size = int(2 * config.intermediate_size / 3)
        intermediate_size = multiple_of * (
            (intermediate_size + multiple_of - 1) // multiple_of
        )
        if XFORMERS_AVAILABLE:
            self.ffn = SwiGLU(
                config.hidden_size, intermediate_size, config.hidden_size, bias=False
            )
        else:
            self.ffn = NeobertMLP(config.hidden_size, intermediate_size, bias=False)

        # Layer norms (Pre-norm architecture)
        self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        output_attentions: bool,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the encoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask of shape
                (batch_size, num_heads, seq_len, seq_len).
            freqs_cis: Precomputed rotary position embeddings.
            output_attentions: Whether to return attention weights.
            max_seqlen: Maximum sequence length for packed sequences.
            cu_seqlens: Cumulative sequence lengths for packed sequences.

        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, hidden_size)
                - Optional attention weights if output_attentions is True
        """
        # Pre-norm attention with residual connection
        attn_output, attn_weights = self._att_block(
            self.attention_norm(x),
            attention_mask,
            freqs_cis,
            output_attentions,
            max_seqlen,
            cu_seqlens,
        )

        # Residual connection
        x = x + attn_output

        # Pre-norm feed-forward with residual connection
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        output_attentions: bool,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head self-attention.

        Args:
            x: Normalized input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            freqs_cis: Rotary position embeddings.
            output_attentions: Whether to return attention weights.
            max_seqlen: Maximum sequence length (for packed sequences).
            cu_seqlens: Cumulative sequence lengths (for packed sequences).

        Returns:
            Tuple of:
                - Attention output of shape (batch_size, seq_len, hidden_size)
                - Optional attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        xq, xk, xv = (
            self.qkv(x)
            .view(
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.dim_head * 3,
            )
            .chunk(3, axis=-1)
        )

        # Apply rotary position embeddings to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Attention computation
        attn_weights = None

        # Flash attention for packed sequences (most efficient)
        if cu_seqlens is not None:
            attn = flash_attn_varlen_func(
                q=xq.squeeze(0),
                k=xk.squeeze(0),
                v=xv.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
            )
        # Eager attention if attention weights are needed
        elif output_attentions:
            attn_weights = (
                xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            )
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))
            attn_weights = attn_weights.softmax(-1)
            attn = attn_weights @ xv.permute(0, 2, 1, 3)
            attn = attn.transpose(1, 2)
        # Scaled dot-product attention (default)
        else:
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attention_mask if attention_mask is not None else None,
                dropout_p=0,
            ).transpose(1, 2)

        # Apply output projection
        return (
            self.wo(
                attn.reshape(
                    batch_size,
                    seq_len,
                    self.config.num_attention_heads * self.config.dim_head,
                )
            ),
            attn_weights,
        )


class NeoBERTPreTrainedModel(PreTrainedModel):
    """Abstract base class for NeoBERT models.

    Handles weights initialization and provides a simple interface for
    downloading and loading pretrained models.
    """

    config_class = NeoBERTConfig
    base_model_prefix = "model"
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of a module.

        Args:
            module: The module to initialize weights for.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(
                -self.config.decoder_init_range, self.config.decoder_init_range
            )
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(
                -self.config.embedding_init_range, self.config.embedding_init_range
            )


class NeoBERT(NeoBERTPreTrainedModel):
    """Base NeoBERT model without any task-specific head.

    This model outputs raw hidden states without any task-specific head on top.
    It can be used as a feature extractor or with custom heads for downstream tasks.

    Args:
        config: Model configuration class with all model parameters.
    """

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the NeoBERT model.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__(config)

        self.config = config

        # Token embeddings
        self.encoder = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # Precompute rotary position embeddings
        # Non-persistent buffers are not saved in the state_dict
        freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, config.max_length
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Stack of transformer encoder blocks
        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        # Final layer normalization
        self.layer_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutput:
        """Forward pass through the NeoBERT model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).
            position_ids: Position indices of shape (batch_size, seq_len).
                If None, positions are assumed to be consecutive starting from 0.
            max_seqlen: Maximum sequence length in the batch (for packed sequences).
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            attention_mask: Attention mask of shape (batch_size, seq_len).
                Values should be 0 for masked positions, 1 for unmasked.
            output_hidden_states: Whether to return hidden states from all layers.
            output_attentions: Whether to return attention weights.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer hidden states
                - hidden_states: Hidden states from all layers (if requested)
                - attentions: Attention weights from all layers (if requested)

        Raises:
            AssertionError: If packed sequences are used incorrectly or if
                Flash Attention is required but not available.
        """
        # Initialize containers for outputs
        hidden_states, attentions = [], []

        # Prepare attention mask for multi-head attention.
        # Shape: (batch, seq_len) -> (batch, heads, seq_len, seq_len)
        # SDPA expects a bool mask where True entries are masked.
        if attention_mask is not None:
            if attention_mask.dtype is not torch.bool:
                # Accept HF-style 1/0 masks as well as additive 0/-inf masks.
                if attention_mask.min() < 0:
                    attention_mask = attention_mask < 0
                else:
                    attention_mask = attention_mask == 0
            attention_mask = (
                attention_mask.unsqueeze(1)
                .unsqueeze(2)
                .expand(
                    -1,
                    self.config.num_attention_heads,
                    attention_mask.size(-1),
                    attention_mask.size(-1),
                )
            )

        # Validate packed sequence configuration
        if cu_seqlens is not None:
            assert FLASH_ATTN_AVAILABLE, (
                "Flash-attention is not available. Please 'pip install flash_attn', "
                "or provide un-packed sequences."
            )
            assert not output_attentions, (
                "Output attentions is not supported when sequences are packed."
            )
            assert max_seqlen is not None, (
                "Missing max_seqlen. It must be provided when cu_seqlens are not None."
            )
            assert input_ids.shape[0] == 1, (
                "Cumulative sequence lengths are provided but input_ids are not packed."
            )
            assert input_ids.is_cuda, (
                "Packing uses an implementation of flash-attention and is only "
                "supported on GPU."
            )

        # Get rotary position embeddings
        freqs_cis = (
            self.freqs_cis[position_ids]
            if position_ids is not None
            else self.freqs_cis[: input_ids.shape[1]].unsqueeze(0)
        )

        # Token embeddings
        x = self.encoder(input_ids)

        # Pass through transformer encoder blocks
        for layer in self.transformer_encoder:
            x, attn = layer(
                x, attention_mask, freqs_cis, output_attentions, max_seqlen, cu_seqlens
            )
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Apply final layer normalization
        x = self.layer_norm(x)

        # Return outputs in HuggingFace format
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
        )


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    """NeoBERT model with a language modeling head for masked language modeling.

    This model is suitable for masked language modeling (MLM) pretraining and
    can be used for tasks like fill-mask predictions.

    Args:
        config: Model configuration class with all model parameters.
    """

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the NeoBERT model with language modeling head.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__(config)

        self.config = config

        # Base NeoBERT model
        self.model = NeoBERT(config)
        # Language modeling head (decoder)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> MaskedLMOutput:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).
            position_ids: Position indices of shape (batch_size, seq_len).
            max_seqlen: Maximum sequence length (for packed sequences).
            cu_seqlens: Cumulative sequence lengths (for packed sequences).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            output_hidden_states: Whether to return hidden states from all layers.
            output_attentions: Whether to return attention weights.
            **kwargs: Additional keyword arguments.

        Returns:
            MaskedLMOutput containing:
                - logits: Prediction scores of shape (batch_size, seq_len, vocab_size)
                - hidden_states: Hidden states from all layers (if requested)
                - attentions: Attention weights from all layers (if requested)
        """
        # Get base model outputs
        output = self.model.forward(
            input_ids,
            position_ids,
            max_seqlen,
            cu_seqlens,
            attention_mask,
            output_hidden_states,
            output_attentions,
        )

        # Apply language modeling head
        logits = self.decoder(output.last_hidden_state)

        return MaskedLMOutput(
            logits=logits,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions if output_attentions else None,
        )


class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):
    """NeoBERT model with a sequence classification head.

    This model is suitable for sequence classification tasks like sentiment
    analysis, text classification, or regression tasks.

    Args:
        config: Model configuration class with all model parameters.
    """

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the NeoBERT model for sequence classification.

        Args:
            config: Configuration object containing model hyperparameters.
                Can include additional attributes:
                - num_labels: Number of labels for classification (default: 2)
                - classifier_dropout: Dropout probability for classifier (default: 0.1)
                - classifier_init_range: Range for classifier weight init (default: 0.02)
        """
        super().__init__(config)

        self.config = config

        # Classification parameters
        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        # Base NeoBERT model
        self.model = NeoBERT(config)

        # Classification head
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for classification layers.

        Args:
            module: The module to initialize weights for.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, tuple]:
        """Forward pass for sequence classification.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).
            position_ids: Position indices of shape (batch_size, seq_len).
            max_seqlen: Maximum sequence length (for packed sequences).
            cu_seqlens: Cumulative sequence lengths (for packed sequences).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            output_hidden_states: Whether to return hidden states from all layers.
            output_attentions: Whether to return attention weights.
            labels: Labels for computing the classification loss.
                Shape depends on the task:
                - (batch_size,) for single-label classification
                - (batch_size, num_labels) for multi-label classification
            return_dict: Whether to return a SequenceClassifierOutput or tuple.

        Returns:
            SequenceClassifierOutput or tuple containing:
                - loss: Classification loss (if labels provided)
                - logits: Classification scores of shape (batch_size, num_labels)
                - hidden_states: Hidden states from all layers (if requested)
                - attentions: Attention weights from all layers (if requested)
        """
        # Get base model outputs
        output = self.model.forward(
            input_ids,
            position_ids,
            max_seqlen,
            cu_seqlens,
            attention_mask,
            output_hidden_states,
            output_attentions,
        )
        hidden_states = output.last_hidden_state

        # Extract CLS token representation (first token)
        x = hidden_states[:, 0, :]

        # Apply classification head with dropout and tanh activation
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        # Get final logits
        logits = self.classifier(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Determine problem type if not specified
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # Apply appropriate loss function
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # Return outputs in requested format
        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions if output_attentions else None,
        )
