"""NeoBERT encoder backbone and shared modeling utilities."""

# NOTE: HF export/inference uses ``neobert/huggingface/modeling_neobert.py`` with
# different attention backends; keep core math consistent across both.

import logging
import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel

from neobert.kernels.attention import (
    PackedFlashMetadata,
    attention_forward,
    canonicalize_attn_backend,
    prepare_packed_flash_metadata,
)
from neobert.kernels.backend import (
    canonicalize_kernel_backend,
    get_rmsnorm,
    swiglu_forward,
)
from neobert.model.rotary import apply_rotary_emb, precompute_freqs_cis
from neobert.modeling_utils import (
    is_torch_compiling,
    packed_seqlens_to_tensor,
    swiglu_intermediate_size,
)
from neobert.warnings import NeoBERTWarning

if TYPE_CHECKING:
    from neobert.config import ModelConfig

logger = logging.getLogger(__name__)
PackedSeqLens = torch.Tensor | list[list[int]]


def _normalize_pad_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """Normalize additive pad masks to a broadcast-friendly 4D shape.

    :param torch.Tensor pad_mask: Additive mask in 2D or 3D form.
    :return torch.Tensor: Mask shaped (B, 1, 1, S) or (B, 1, S, S).
    """
    # Training API intentionally standardizes on additive masks (0 keep / -inf
    # mask) for the hot path. HF export wrappers normalize 0/1 masks separately.
    if not pad_mask.is_floating_point():
        raise TypeError(
            "pad_mask must be an additive floating mask with 0 for keep and -inf for masked positions."
        )
    if pad_mask.dtype != torch.float32:
        pad_mask = pad_mask.to(torch.float32)

    if pad_mask.dim() == 2:
        # Key padding mask; broadcast across query positions + heads.
        return pad_mask[:, None, None, :]
    if pad_mask.dim() == 3:
        # Full attention mask per sample; broadcast across heads only.
        return pad_mask[:, None, :, :]
    raise ValueError(
        "pad_mask must have shape (batch, seq_len) or (batch, seq_len, seq_len)"
    )


def _infer_single_segment_packed_seqlens_from_pad_mask(
    pad_mask: torch.Tensor, seq_len: int
) -> Optional[torch.Tensor]:
    """Infer packed lengths from an additive pad mask.

    This only works for right-padded masks where tokens are unmasked first and
    masked only at the tail. Returns None if the mask is not a simple padding mask.

    :param torch.Tensor pad_mask: Additive pad mask (0/-inf).
    :param int seq_len: Sequence length for validation.
    :return torch.Tensor | None: Packed lengths tensor shaped ``[B, 1]``.
    """
    if not torch.is_tensor(pad_mask):
        raise TypeError("pad_mask must be a torch.Tensor")

    mask = pad_mask.detach()
    if mask.is_cuda:
        # Avoid an implicit CPU sync in forward(); gracefully fall back to the
        # additive-mask path unless packed_seqlens was prepared by the collator.
        return None

    if mask.dim() == 2:
        key_mask = mask
    elif mask.dim() == 4:
        key_mask = mask[:, 0, 0, :]
    else:
        return None

    if key_mask.shape[-1] != seq_len:
        return None

    keep = torch.isfinite(key_mask) & (key_mask == 0)
    keep_int = keep.to(torch.int)
    if not torch.all(keep_int.cummin(dim=-1).values == keep_int):
        return None

    lengths = keep_int.sum(dim=-1).clamp(max=seq_len).to(torch.int32)
    # Fully padded rows are not valid packed segments; keep additive-mask path.
    if (lengths <= 0).any():
        return None
    return lengths.unsqueeze(1).cpu()


def _normalize_packed_seqlens(
    packed_seqlens: Any,
    *,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Normalize packed sequence lengths to rank-2 int32 tensors.

    :param Any packed_seqlens: Packed segment lengths tensor or list.
    :param int | None seq_len: Optional sequence length for validation.
    :param int | None batch_size: Optional batch size for validation.
    :return torch.Tensor | None: Packed segment lengths tensor of shape ``[B, N]``.
    """
    tensor = packed_seqlens_to_tensor(packed_seqlens)
    if tensor is None:
        return None

    if batch_size is not None and tensor.shape[0] != batch_size:
        raise ValueError(
            "packed_seqlens batch dimension does not match input batch size "
            f"({tensor.shape[0]} != {batch_size})."
        )
    if (tensor < 0).any():
        raise ValueError("packed_seqlens must contain non-negative lengths.")

    if seq_len is not None:
        sums = tensor.sum(dim=1)
        bad = sums > seq_len
        if bad.any():
            bad_idx = int(torch.where(bad)[0][0].item())
            raise ValueError(
                "packed_seqlens sum exceeds seq_len "
                f"(row={bad_idx}, sum={int(sums[bad_idx].item())}, seq_len={seq_len})."
            )

    return tensor


def _build_learned_position_ids(
    src: torch.Tensor,
    *,
    pad_token_id: int,
    packed_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build one-indexed learned position IDs with zero reserved for padding.

    Packed segment lengths define document boundaries, so positions restart at
    one for every segment. Unpacked inputs retain the pad-aware cumulative
    positions used by the original model.

    :param torch.Tensor src: Token IDs shaped ``[batch, sequence]``.
    :param int pad_token_id: Token ID treated as padding for unpacked inputs.
    :param torch.Tensor | None packed_seqlens: Normalized packed segment lengths.
    :return torch.Tensor: Position IDs shaped like ``src``.
    """
    if packed_seqlens is None:
        token_mask = src.ne(pad_token_id).to(torch.long)
        return torch.cumsum(token_mask, dim=1) * token_mask

    batch_size, seq_len = src.shape
    lengths = packed_seqlens.to(device=src.device, dtype=torch.long)
    segment_starts = torch.cumsum(lengths, dim=1) - lengths
    sentinel_starts = torch.full_like(segment_starts, seq_len)
    safe_starts = torch.where(lengths > 0, segment_starts, sentinel_starts)

    start_offsets = torch.zeros(
        (batch_size, seq_len + 1), device=src.device, dtype=torch.long
    )
    start_offsets.scatter_(1, safe_starts, safe_starts)
    latest_start = torch.cummax(start_offsets[:, :seq_len], dim=1).values

    token_offsets = torch.arange(seq_len, device=src.device).unsqueeze(0)
    position_ids = token_offsets - latest_start + 1
    valid_tokens = token_offsets < lengths.sum(dim=1, keepdim=True)
    return position_ids * valid_tokens


class SwiGLU(nn.Module):
    """SwiGLU activation with Liger kernel dispatch (unpacked w1/w2/w3)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        kernel_backend: str = "torch",
    ) -> None:
        """Initialize the SwiGLU block.

        :param int in_features: Input feature dimension.
        :param int | None hidden_features: Hidden feature dimension.
        :param int | None out_features: Output feature dimension.
        :param bool bias: Whether to use bias in linear layers.
        :param str kernel_backend: ``"liger"`` or ``"torch"``.
        """
        super().__init__()
        self.kernel_backend = kernel_backend
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Output tensor.
        """
        return self.w3(swiglu_forward(self.w1(x), self.w2(x), self.kernel_backend))


class NeoBERTConfig(PretrainedConfig):
    """Configuration for the NeoBERT model."""

    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        rope: bool = True,
        norm_eps: float = 1e-06,
        hidden_act: str = "SwiGLU",
        vocab_size: int = 30522,  # Keep in sync with ConfigLoader/HF defaults.
        pad_token_id: int = 0,
        max_length: int = 1024,
        attn_backend: str = "sdpa",
        kernel_backend: str = "auto",
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ):
        """Initialize the NeoBERT configuration.

        :param int hidden_size: Hidden size of the transformer.
        :param int num_hidden_layers: Number of transformer layers.
        :param int num_attention_heads: Number of attention heads.
        :param int intermediate_size: Feed-forward hidden size.
        :param float dropout: Dropout probability.
        :param float embedding_init_range: Embedding init range.
        :param float decoder_init_range: Decoder init range.
        :param bool rms_norm: Whether to use RMSNorm.
        :param bool rope: Whether to use rotary embeddings.
        :param float norm_eps: Normalization epsilon.
        :param str hidden_act: Activation function name.
        :param int vocab_size: Vocabulary size.
        :param int pad_token_id: Padding token ID.
        :param int max_length: Maximum sequence length.
        :param str attn_backend: Attention backend (``"sdpa"`` or ``"flash_attn_varlen"``).
        :param str kernel_backend: Kernel backend (``"auto"``, ``"liger"``, or ``"torch"``).
        :param bool tie_word_embeddings: Whether to tie input/output embeddings.
        :param Any kwargs: Additional configuration parameters.
        """
        removed_fields = {"ngpt", "base_scale"}.intersection(kwargs)
        if removed_fields:
            raise TypeError(
                "Unsupported removed NeoBERT config field(s): "
                + ", ".join(sorted(removed_fields))
            )
        # Legacy: accept flash_attention bool and map to attn_backend
        if "flash_attention" in kwargs:
            fa = kwargs.pop("flash_attention")
            warnings.warn(
                "NeoBERTConfig: 'flash_attention' is deprecated; use "
                '\'attn_backend\' instead ("sdpa" or "flash_attn_varlen").',
                NeoBERTWarning,
                stacklevel=2,
            )
            if isinstance(fa, bool):
                attn_backend = "flash_attn_varlen" if fa else "sdpa"
            else:
                attn_backend = str(fa)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # Core dims
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        self.dim_head = hidden_size // num_attention_heads
        if rope and self.dim_head % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        self.intermediate_size = intermediate_size

        # Dropout: accept legacy 'dropout_prob' as alias
        if "dropout_prob" in kwargs and "dropout" not in kwargs:
            warnings.warn(
                "NeoBERTConfig: 'dropout_prob' is deprecated; use 'dropout' instead.",
                NeoBERTWarning,
                stacklevel=2,
            )
            dropout = kwargs["dropout_prob"]
        self.dropout = dropout

        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.rope = rope
        self.norm_eps = norm_eps
        normalized_act = str(hidden_act).lower()
        if normalized_act not in {"swiglu", "gelu"}:
            raise ValueError(
                f"Unsupported hidden_act '{hidden_act}'. Supported: swiglu, gelu."
            )
        self.hidden_act = normalized_act
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Positional length: accept HF-style 'max_position_embeddings'
        if (
            "max_position_embeddings" in kwargs
            and kwargs["max_position_embeddings"] is not None
        ):
            warnings.warn(
                "NeoBERTConfig: 'max_position_embeddings' is deprecated for the "
                "training model; use 'max_length' when constructing configs.",
                NeoBERTWarning,
                stacklevel=2,
            )
            self.max_length = int(kwargs["max_position_embeddings"])
        else:
            self.max_length = max_length
        # Keep the HF-style attribute for compatibility with downstream tooling.
        self.max_position_embeddings = self.max_length

        self.attn_backend = canonicalize_attn_backend(attn_backend)
        self.kernel_backend = canonicalize_kernel_backend(kernel_backend)

    @classmethod
    def from_model_config(
        cls,
        model_config: "ModelConfig",
        *,
        max_length: int,
        pad_token_id: int,
        attn_backend: str,
        vocab_size: Optional[int] = None,
        num_labels: Optional[int] = None,
    ) -> "NeoBERTConfig":
        """Construct a runtime model config from the typed project model config.

        Sequence length, padding ID, and attention backend are required at each task
        boundary because they depend on tokenizer and execution context.

        :param ModelConfig model_config: Typed project model configuration.
        :param int max_length: Task-specific maximum sequence length.
        :param int pad_token_id: Task tokenizer padding token ID.
        :param str attn_backend: Task-specific attention backend.
        :param int | None vocab_size: Optional task-specific vocabulary override.
        :param int | None num_labels: Optional classification label count.
        :return NeoBERTConfig: Canonical runtime model configuration.
        """
        runtime_kwargs: dict[str, Any] = {
            "classifier_init_range": model_config.classifier_init_range,
        }
        if num_labels is not None:
            runtime_kwargs["num_labels"] = num_labels
        return cls(
            hidden_size=model_config.hidden_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            intermediate_size=model_config.intermediate_size,
            dropout=model_config.dropout_prob,
            embedding_init_range=model_config.embedding_init_range,
            decoder_init_range=model_config.decoder_init_range,
            rms_norm=model_config.rms_norm,
            rope=model_config.rope,
            norm_eps=model_config.norm_eps,
            hidden_act=model_config.hidden_act,
            vocab_size=model_config.vocab_size if vocab_size is None else vocab_size,
            pad_token_id=pad_token_id,
            max_length=max_length,
            attn_backend=attn_backend,
            kernel_backend=model_config.kernel_backend,
            **runtime_kwargs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the encoder block.

        :param NeoBERTConfig config: Model configuration.
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
        self.resid_dropout = nn.Dropout(config.dropout)

        # Kernel backend for Liger/torch dispatch (resolved at forward time)
        self._kb = getattr(config, "kernel_backend", "auto")

        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                intermediate_size = swiglu_intermediate_size(config.intermediate_size)
                self.ffn = SwiGLU(
                    config.hidden_size,
                    intermediate_size,
                    config.hidden_size,
                    bias=False,
                    kernel_backend=self._kb,
                )
            case "gelu":
                self.ffn = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                )
            case _:
                raise ValueError(
                    f"Unsupported hidden_act '{config.hidden_act}'. Supported: swiglu, gelu."
                )

        self.attention_norm = (
            get_rmsnorm(config.hidden_size, config.norm_eps, self._kb)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )
        self.ffn_norm = (
            get_rmsnorm(config.hidden_size, config.norm_eps, self._kb)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        packed_seqlens: Optional[PackedSeqLens] = None,
        packed_flash_meta: Optional[PackedFlashMetadata] = None,
    ) -> torch.Tensor:
        """Run the encoder block forward pass.

        :param torch.Tensor x: Input tensor.
        :param torch.Tensor pad_mask: Additive attention mask.
        :param torch.Tensor freqs_cis: Rotary embedding frequencies.
        :param list[list[int]] | torch.Tensor | None packed_seqlens: Packed segment lengths.
        :param PackedFlashMetadata | None packed_flash_meta: Cached flash varlen metadata.
        :return torch.Tensor: Updated hidden states.
        """
        x = x + self._att_block(
            self.attention_norm(x),
            pad_mask,
            freqs_cis,
            packed_seqlens,
            packed_flash_meta,
        )
        x = x + self._ff_block(self.ffn_norm(x))
        return x

    def _att_block(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        packed_seqlens: Optional[PackedSeqLens] = None,
        packed_flash_meta: Optional[PackedFlashMetadata] = None,
    ) -> torch.Tensor:
        """Apply the attention sub-layer.

        :param torch.Tensor x: Normalized hidden states.
        :param torch.Tensor pad_mask: Additive attention mask.
        :param torch.Tensor freqs_cis: Rotary embedding frequencies.
        :param list[list[int]] | torch.Tensor | None packed_seqlens: Packed segment lengths.
        :param PackedFlashMetadata | None packed_flash_meta: Cached flash varlen metadata.
        :return torch.Tensor: Attention output.
        """
        batch_size, seq_len, _ = x.shape

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

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        attn = attention_forward(
            xq,
            xk,
            xv,
            pad_mask=pad_mask,
            packed_seqlens=packed_seqlens,
            dropout_p=self.config.dropout if self.training else 0.0,
            scale=None,
            attn_backend=self.config.attn_backend,
            packed_flash_metadata=packed_flash_meta,
        )

        return self.resid_dropout(
            self.wo(
                attn.reshape(
                    batch_size,
                    seq_len,
                    self.config.num_attention_heads * self.config.dim_head,
                )
            )
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward sub-layer.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Feed-forward output.
        """
        return self.ffn_dropout(self.ffn(x))


class NeoBERTPreTrainedModel(PreTrainedModel):
    """Base class with NeoBERT weight initialization."""

    config_class = NeoBERTConfig
    _supports_cache_class = True
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for NeoBERT modules.

        :param nn.Module module: Module to initialize.
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
    """NeoBERT encoder model."""

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the NeoBERT encoder.

        :param NeoBERTConfig config: Model configuration.
        """
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        if self.config.rope:
            # Keep a fixed-size RoPE cache to avoid mutating buffers in forward().
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(config.dim_head, config.max_length),
                persistent=False,
            )
        else:
            # Use a fixed padding index (0) for positional embeddings to decouple
            # position IDs from token padding IDs.
            self.positional_embedding = nn.Embedding(
                # Positions are 1-indexed when using cumsum; reserve 0 for padding.
                config.max_length + 1,
                config.hidden_size,
                padding_idx=0,
            )

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        _kb = getattr(config, "kernel_backend", "auto")
        self.layer_norm = (
            get_rmsnorm(config.hidden_size, config.norm_eps, _kb)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()
        self.gradient_checkpointing = False

    def forward(
        self,
        src: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        packed_seqlens: Optional[PackedSeqLens] = None,
    ) -> torch.Tensor:
        """Run the NeoBERT encoder forward pass.

        :param torch.Tensor src: Input token IDs.
        :param torch.Tensor | None pad_mask: Additive attention mask.
        :param torch.Tensor | list[list[int]] | None packed_seqlens: Packed segment lengths.
        :return torch.Tensor: Encoded hidden states.
        """
        seq_len = src.shape[1]
        packed_seqlens = _normalize_packed_seqlens(
            packed_seqlens,
            seq_len=seq_len,
            batch_size=src.shape[0],
        )

        use_packed = self.config.attn_backend != "sdpa" or packed_seqlens is not None
        if use_packed:
            if (
                packed_seqlens is None
                and pad_mask is not None
                and torch.is_tensor(pad_mask)
            ):
                packed_seqlens = _infer_single_segment_packed_seqlens_from_pad_mask(
                    pad_mask, seq_len
                )
                if packed_seqlens is not None:
                    pad_mask = None
            if packed_seqlens is not None and pad_mask is not None:
                logger.warning(
                    "packed_seqlens provided; ignoring pad_mask for packed attention."
                )
                pad_mask = None

        packed_flash_meta: Optional[PackedFlashMetadata] = None
        if (
            packed_seqlens is not None
            and self.config.attn_backend == "flash_attn_varlen"
            and not is_torch_compiling()
        ):
            packed_flash_meta = prepare_packed_flash_metadata(
                packed_seqlens,
                batch_size=src.shape[0],
                seq_len=seq_len,
                device=src.device,
            )

        # Normalize to broadcast-friendly shapes to avoid O(S^2) materialization.
        if pad_mask is not None and torch.is_tensor(pad_mask):
            pad_mask = _normalize_pad_mask(pad_mask)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            seq_len = src.shape[1]
            if seq_len > self.config.max_length:
                warnings.warn(
                    f"Sequence length {seq_len} exceeds max_length {self.config.max_length}; "
                    "using a transient RoPE cache for this forward. Consider truncating inputs.",
                    NeoBERTWarning,
                    stacklevel=2,
                )
                freqs_cis = precompute_freqs_cis(
                    self.config.dim_head, seq_len, device=src.device
                )
            else:
                freqs_cis = self.freqs_cis
                if freqs_cis.device != src.device:
                    freqs_cis = freqs_cis.to(src.device)
                freqs_cis = freqs_cis[:seq_len]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            position_ids = _build_learned_position_ids(
                src,
                pad_token_id=self.config.pad_token_id,
                packed_seqlens=packed_seqlens,
            )
            x += self.positional_embedding(position_ids)

        # Transformer encoder
        for layer in self.transformer_encoder:
            if self.gradient_checkpointing and self.training:

                def custom_forward(
                    hidden_states: torch.Tensor, layer: EncoderBlock = layer
                ) -> torch.Tensor:
                    """Run one encoder block for gradient checkpointing.

                    :param torch.Tensor hidden_states: Input hidden states.
                    :param EncoderBlock layer: Bound layer instance.
                    :return torch.Tensor: Updated hidden states.
                    """
                    return layer(
                        hidden_states,
                        pad_mask,
                        freqs_cis,
                        packed_seqlens,
                        packed_flash_meta,
                    )

                x = checkpoint(
                    custom_forward,
                    x,
                    preserve_rng_state=True,
                    use_reentrant=False,
                )
            else:
                x = layer(x, pad_mask, freqs_cis, packed_seqlens, packed_flash_meta)

        x = self.layer_norm(x)
        return x
