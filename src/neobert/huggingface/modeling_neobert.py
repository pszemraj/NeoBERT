"""NeoBERT model implementation for HuggingFace Transformers.

This module provides the NeoBERT architecture compatible with the HuggingFace
Transformers library. It includes the base model, language modeling head, and
sequence classification variants.

Architecture Features:
- SwiGLU activation function for improved training dynamics
- Rotary Position Embeddings (RoPE) for better position encoding
- Pre-RMSNorm for improved training stability
- Scaled dot-product attention for efficient long-context processing

Based on: https://github.com/facebookresearch/llama/blob/main/llama/model.py

NOTE: The training-time implementation lives in ``src/neobert/model/model.py`` and
uses xFormers for flash attention. This HF variant targets export/inference APIs
and may use different attention backends; keep the math consistent when editing.
Packed/varlen sequences are intentionally unsupported in this HF export model.
Module naming follows NeoBERT conventions; loading from unrelated BERT/RoBERTa
checkpoints is not guaranteed to be compatible.
"""

from typing import Any, Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

try:
    from .modeling_utils import scaled_dot_product_attention_compat
    from .modeling_utils import swiglu_intermediate_size
except ImportError:  # pragma: no cover - in-package import path.
    try:
        from ..modeling_utils import scaled_dot_product_attention_compat
        from ..modeling_utils import swiglu_intermediate_size
    except ImportError:  # pragma: no cover - triggered in exported HF repo layout.
        from modeling_utils import swiglu_intermediate_size
        from modeling_utils import scaled_dot_product_attention_compat

from .rotary import apply_rotary_emb, precompute_freqs_cis


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
        ngpt: Whether nGPT-style normalization is enabled (unsupported in HF export).
        base_scale: Base scaling factor for nGPT (retained for config parity).
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
        rms_norm: bool = True,
        rope: bool = True,
        hidden_act: str = "swiglu",
        dropout: float = 0.0,
        flash_attention: bool = False,
        tie_word_embeddings: bool = True,
        ngpt: bool = False,
        base_scale: float = 1.0 / (960.0**0.5),
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
        :param bool rms_norm: Whether to use RMSNorm (otherwise LayerNorm).
        :param bool rope: Whether to use RoPE (otherwise learned positional embeddings).
        :param str hidden_act: Activation name ("swiglu" or "gelu").
        :param float dropout: Dropout probability for residual/MLP blocks.
        :param bool flash_attention: Whether to prefer flash attention backends.
        :param bool tie_word_embeddings: Whether to tie input/output embeddings.
        :param bool ngpt: Whether to enable nGPT-style normalization (unsupported here).
        :param float base_scale: Base scaling factor for nGPT compatibility.
        :param Any kwargs: Additional configuration parameters.
        """
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

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
        self.rms_norm = rms_norm
        # Keep rope=False for ablations and CPU-only tests; HF export supports it.
        self.rope = rope
        normalized_act = str(hidden_act).lower()
        if normalized_act not in {"swiglu", "gelu"}:
            raise ValueError(
                f"Unsupported hidden_act '{hidden_act}'. Supported: swiglu, gelu."
            )
        self.hidden_act = normalized_act
        self.dropout = dropout
        # Retained for config.json compatibility with training configs; silently ignored.
        self.flash_attention = flash_attention
        self.ngpt = ngpt
        self.base_scale = base_scale
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.max_position_embeddings = self.max_length
        self.kwargs = kwargs


class UnpackedSwiGLU(nn.Module):
    """Unpacked SwiGLU MLP (w1, w2, w3) matching training fallback.

    Note: Keeps the w1/w2/w3 layout so a future LigerSwiGLUMLP drop-in can
    replace this class without weight conversion. An adapter will be needed
    to reconcile Liger's activation flag expectations and intermediate_size.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        """Initialize the unpacked SwiGLU block.

        Args:
            in_features: Input feature dimension.
            hidden_features: Hidden feature dimension.
            out_features: Output feature dimension (defaults to in_features).
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation with unpacked weights.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features).

        Returns:
            Output tensor of shape (batch_size, seq_len, out_features).
        """
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


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

        # Feedforward network
        if config.hidden_act == "swiglu":
            # Match training: reduce by 2/3 and round to multiple of 8.
            intermediate_size = swiglu_intermediate_size(config.intermediate_size)
            self.ffn = UnpackedSwiGLU(
                config.hidden_size,
                intermediate_size,
                out_features=config.hidden_size,
                bias=False,
            )
        elif config.hidden_act == "gelu":
            # Keep GELU for CPU-only tests/ablations; SwiGLU is default.
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            )
        else:
            raise ValueError(
                f"Unsupported hidden_act '{config.hidden_act}'. Supported: swiglu, gelu."
            )

        # Layer norms (Pre-norm architecture)
        if config.rms_norm:
            self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
            self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        else:
            self.attention_norm = nn.LayerNorm(config.hidden_size, config.norm_eps)
            self.ffn_norm = nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.resid_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the encoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask of shape
                (batch_size, num_heads, seq_len, seq_len).
            freqs_cis: Precomputed rotary position embeddings.
            output_attentions: Whether to return attention weights.
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
        )

        # Residual connection
        x = x + attn_output

        # Pre-norm feed-forward with residual connection
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        return x, attn_weights

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head self-attention.

        Args:
            x: Normalized input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            freqs_cis: Rotary position embeddings.
            output_attentions: Whether to return attention weights.
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
            .chunk(3, dim=-1)
        )

        # Apply rotary position embeddings to Q and K
        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Attention computation
        attn_weights = None

        # Eager attention if attention weights are needed
        if output_attentions:
            attn_weights = (
                xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            )
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))
            attn_weights = attn_weights.softmax(-1)
            # Apply attention dropout to match SDPA path.
            if self.training and self.config.dropout > 0:
                attn_weights = torch.nn.functional.dropout(
                    attn_weights, p=self.config.dropout, training=True
                )
            attn = attn_weights @ xv.permute(0, 2, 1, 3)
            attn = attn.transpose(1, 2)
        # Scaled dot-product attention (default)
        else:
            attn = scaled_dot_product_attention_compat(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attention_mask if attention_mask is not None else None,
                dropout_p=self.config.dropout if self.training else 0.0,
            ).transpose(1, 2)

        # Apply output projection
        attn_out = self.wo(
            attn.reshape(
                batch_size,
                seq_len,
                self.config.num_attention_heads * self.config.dim_head,
            )
        )
        attn_out = self.resid_dropout(attn_out)
        return attn_out, attn_weights


class NeoBERTPreTrainedModel(PreTrainedModel):
    """Abstract base class for NeoBERT models.

    Handles weights initialization and provides a simple interface for
    downloading and loading pretrained models.
    """

    config_class = NeoBERTConfig
    base_model_prefix = "model"
    _supports_cache_class = False  # Encoder-only; no KV cache support.

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of a module.

        Args:
            module: The module to initialize weights for.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(
                -self.config.decoder_init_range, self.config.decoder_init_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
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
        if getattr(config, "ngpt", False):
            raise ValueError(
                "ngpt/NormNeoBERT is not supported in the HF export path. "
                "Export a non-ngpt checkpoint or use the training model."
            )

        # Token embeddings
        self.encoder = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # Precompute rotary position embeddings (or use learned positional embeddings)
        # Non-persistent buffers are not saved in the state_dict
        if self.config.rope:
            # Lazy RoPE cache; populated on first forward for the active device/length.
            self.register_buffer("freqs_cis", torch.empty(0), persistent=False)
        else:
            # Use a fixed padding index (0) for positional embeddings to decouple
            # position IDs from token padding IDs.
            self.positional_embedding = nn.Embedding(
                # Positions are 1-indexed when using cumsum; reserve 0 for padding.
                config.max_length + 1,
                config.hidden_size,
                padding_idx=0,
            )

        # Stack of transformer encoder blocks
        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        # Final layer normalization
        self.layer_norm = (
            nn.RMSNorm(config.hidden_size, config.norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _normalize_attention_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert HF attention mask (1=keep) to SDPA format (True=masked).

        :param torch.Tensor attention_mask: Input attention mask (batch, seq_len).
        :return torch.Tensor: Boolean mask where True entries are masked.
        """
        if attention_mask.dim() != 2:
            raise ValueError(
                "NeoBERT HF export expects a 2D attention_mask of shape "
                "(batch, seq_len). Pre-expanded masks are not supported."
            )
        if attention_mask.dtype is torch.bool:
            # HF convention: True=keep → SDPA: True=masked
            return ~attention_mask
        # Float/int: HF uses 1=keep, 0=mask; additive uses 0=keep, -inf=mask
        if attention_mask.min() < 0:
            return attention_mask < 0  # Additive mask
        return attention_mask == 0  # Binary 0/1 mask

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
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
            attention_mask: Attention mask of shape (batch_size, seq_len).
                HF convention: 1=keep, 0=mask. Also accepts additive masks (0/-inf).
            output_hidden_states: Whether to return hidden states from all layers.
            output_attentions: Whether to return attention weights.
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final layer hidden states
                - hidden_states: Hidden states from all layers (if requested)
                - attentions: Attention weights from all layers (if requested)

        """
        # Initialize containers for outputs (HF contract, populated only if requested).
        hidden_states, attentions = [], []

        # Prepare attention mask for multi-head attention.
        # Shape: (batch, seq_len) -> (batch, heads, seq_len, seq_len)
        # SDPA expects a bool mask where True entries are masked.
        if attention_mask is not None:
            attention_mask = self._normalize_attention_mask(attention_mask)
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must match input_ids shape for HF export "
                    f"(got {attention_mask.shape} vs {input_ids.shape})."
                )
            # Encoder-only key padding mask, broadcast across query positions.
            # Keep mask in (B, 1, 1, S) form to avoid O(S^2) materialization.
            attention_mask = attention_mask[:, None, None, :]

        # Get rotary position embeddings
        freqs_cis = None
        if self.config.rope:
            seq_len = input_ids.shape[1]
            config_max = getattr(self.config, "max_length", None)
            if config_max is None:
                config_max = getattr(self.config, "max_position_embeddings", None)
            max_pos = max(seq_len, int(config_max)) if config_max else seq_len
            # Reuse cached RoPE frequencies; only grow/reallocate if len/device changes.
            if (
                self.freqs_cis.numel() == 0
                or self.freqs_cis.device != input_ids.device
                or self.freqs_cis.shape[0] < max_pos
            ):
                self.freqs_cis = precompute_freqs_cis(
                    self.config.dim_head, max_pos, device=input_ids.device
                )
            freqs_cis = (
                self.freqs_cis[position_ids]
                if position_ids is not None
                else self.freqs_cis[:seq_len].unsqueeze(0)
            )

        # Token embeddings
        x = self.encoder(input_ids)

        # Add learned positional embeddings if RoPE is disabled (ablations/tests).
        if not self.config.rope:
            if position_ids is not None:
                pos_ids = position_ids
            else:
                # Content-based positions: padding stays at index 0, and left-padding
                # does not shift token positions (padding-invariant positional IDs).
                mask = input_ids.ne(self.config.pad_token_id).int()
                incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
                pos_ids = incremental_indices.long()
            x = x + self.positional_embedding(pos_ids)

        # Pass through transformer encoder blocks
        for layer in self.transformer_encoder:
            x, attn = layer(x, attention_mask, freqs_cis, output_attentions)
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
        if getattr(self.config, "tie_word_embeddings", False):
            self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return input token embeddings for weight tying."""
        return self.model.encoder

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """Set input token embeddings (used by HF APIs)."""
        self.model.encoder = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        """Return output embeddings for weight tying."""
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Set output embeddings (used by HF APIs)."""
        self.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> MaskedLMOutput:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).
            position_ids: Position indices of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
                HF convention: 1=keep, 0=mask. Also accepts additive masks (0/-inf).
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
            attention_mask: Attention mask of shape (batch_size, seq_len).
                HF convention: 1=keep, 0=mask. Also accepts additive masks (0/-inf).
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
