"""NeoBERT model architecture and task heads."""

# NOTE: Hugging Face export/inference uses ``neobert/huggingface/modeling_neobert.py`` with
# different attention backends. Keep core math consistent when updating either path.
# Typing coverage here is incremental; prefer correctness/clarity over exhaustive hints.

# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import logging
import math
import warnings
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ..modeling_utils import (
    scaled_dot_product_attention_compat,
    swiglu_intermediate_size,
)
from .rmsnorm import RMSNorm
from .rotary import apply_rotary_emb, precompute_freqs_cis

XFORMERS_ERROR: Optional[str] = None

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask

    XFORMERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # xformers might be installed but have version conflicts
    XFORMERS_AVAILABLE = False
    XFORMERS_ERROR = str(e)
    memory_efficient_attention = None
    BlockDiagonalMask = None

logger = logging.getLogger(__name__)


def _packed_flash_attention(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    packed_seqlens: list[list[int]],
    dropout_p: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Run xFormers attention with a block-diagonal bias from packed segments.

    :param torch.Tensor xq: Query tensor of shape (batch, seq_len, heads, head_dim).
    :param torch.Tensor xk: Key tensor of shape (batch, seq_len, heads, head_dim).
    :param torch.Tensor xv: Value tensor of shape (batch, seq_len, heads, head_dim).
    :param list[list[int]] packed_seqlens: Per-sample segment lengths.
    :param float dropout_p: Dropout probability (0.0 for eval).
    :param float | None scale: Optional softmax scaling factor.
    :return torch.Tensor: Attention output of shape (batch, seq_len, heads, head_dim).
    """
    if not XFORMERS_AVAILABLE or BlockDiagonalMask is None:
        raise ImportError(
            "Packed flash attention requires xformers. Install with: pip install xformers. "
            f"Import error: {XFORMERS_ERROR}"
        )

    batch_size, seq_len, _, _ = xq.shape
    if len(packed_seqlens) != batch_size:
        raise ValueError(
            "packed_seqlens length must match batch size "
            f"({len(packed_seqlens)} != {batch_size})."
        )

    segments_q: list[torch.Tensor] = []
    segments_k: list[torch.Tensor] = []
    segments_v: list[torch.Tensor] = []
    for batch_idx, seg_lens in enumerate(packed_seqlens):
        start = 0
        for seg_len in seg_lens:
            if seg_len <= 0:
                continue
            end = start + seg_len
            if end > seq_len:
                raise ValueError(
                    "packed_seqlens exceeds sequence length "
                    f"(batch={batch_idx}, end={end}, seq_len={seq_len})."
                )
            segments_q.append(xq[batch_idx : batch_idx + 1, start:end])
            segments_k.append(xk[batch_idx : batch_idx + 1, start:end])
            segments_v.append(xv[batch_idx : batch_idx + 1, start:end])
            start = end

    if not segments_q:
        return torch.zeros_like(xq)

    attn_bias, q_cat = BlockDiagonalMask.from_tensor_list(segments_q)
    k_cat = torch.cat(segments_k, dim=1)
    v_cat = torch.cat(segments_v, dim=1)

    attn_cat = memory_efficient_attention(
        query=q_cat,
        key=k_cat,
        value=v_cat,
        attn_bias=attn_bias,
        p=dropout_p,
        scale=scale,
    )

    segments_out = attn_bias.split(attn_cat)
    attn = torch.zeros_like(xq)
    seg_idx = 0
    for batch_idx, seg_lens in enumerate(packed_seqlens):
        start = 0
        for seg_len in seg_lens:
            if seg_len <= 0:
                continue
            end = start + seg_len
            attn[batch_idx : batch_idx + 1, start:end] = segments_out[seg_idx]
            seg_idx += 1
            start = end

    return attn


def _normalize_pad_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """Normalize additive pad masks to a broadcast-friendly 4D shape.

    :param torch.Tensor pad_mask: Additive mask in 2D or 3D form.
    :param int num_heads: Number of attention heads for validation.
    :return torch.Tensor: Mask shaped (B, 1, 1, S) or (B, 1, S, S).
    """
    if pad_mask.dim() == 2:
        # Key padding mask; broadcast across query positions + heads.
        return pad_mask[:, None, None, :]
    if pad_mask.dim() == 3:
        # Full attention mask per sample; broadcast across heads only.
        return pad_mask[:, None, :, :]
    raise ValueError(
        "pad_mask must have shape (batch, seq_len) or (batch, seq_len, seq_len)"
    )


def _prepare_xformers_bias(
    pad_mask: Optional[torch.Tensor],
    num_heads: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Expand an additive pad mask into xFormers' expected bias shape.

    xFormers currently requires a full (B, H, S, S) bias tensor for key padding
    masks. We expand and materialize the broadcasted mask to avoid shape errors.

    :param torch.Tensor | None pad_mask: Additive pad mask (0/-inf).
    :param int num_heads: Number of attention heads.
    :param int seq_len: Sequence length (S).
    :param torch.dtype dtype: Target dtype for the bias tensor.
    :param torch.device device: Target device for the bias tensor.
    :return torch.Tensor | None: Expanded bias tensor or None.
    """
    if pad_mask is None or not torch.is_tensor(pad_mask):
        return pad_mask

    mask = pad_mask
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    elif mask.dim() != 4:
        raise ValueError(
            "pad_mask must have shape (batch, seq_len), (batch, seq_len, seq_len), "
            f"or (batch, 1, 1, seq_len); got {tuple(mask.shape)}"
        )

    if mask.shape[-1] != seq_len:
        raise ValueError(
            f"pad_mask last dim ({mask.shape[-1]}) must equal seq_len ({seq_len})"
        )

    mask = mask.expand(-1, num_heads, seq_len, seq_len).contiguous()
    return mask.to(device=device, dtype=dtype)


def _infer_single_segment_packed_seqlens_from_pad_mask(
    pad_mask: torch.Tensor, seq_len: int
) -> Optional[list[list[int]]]:
    """Infer packed lengths from an additive pad mask.

    This only works for prefix padding masks where tokens are unmasked first and
    masked only at the tail. Returns None if the mask is not a simple padding mask.

    :param torch.Tensor pad_mask: Additive pad mask (0/-inf).
    :param int seq_len: Sequence length for validation.
    :return list[list[int]] | None: Packed lengths per batch item.
    """
    if not torch.is_tensor(pad_mask):
        raise TypeError("pad_mask must be a torch.Tensor")

    mask = pad_mask.detach()
    if mask.is_cuda:
        # Sync once per batch to avoid per-layer GPU stalls.
        mask = mask.cpu()

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

    lengths = keep_int.sum(dim=-1).tolist()
    return [[int(min(seq_len, length))] for length in lengths]


def _normalize_packed_seqlens(
    packed_seqlens: Any,
    *,
    seq_len: Optional[int] = None,
) -> Optional[list[list[int]]]:
    """Normalize packed sequence lengths to Python lists.

    :param Any packed_seqlens: Packed segment lengths tensor or list.
    :param int | None seq_len: Optional sequence length for validation.
    :return list[list[int]] | None: Packed segment lengths as Python lists.
    """
    if packed_seqlens is None:
        return None

    if torch.is_tensor(packed_seqlens):
        if packed_seqlens.numel() == 0:
            return [[] for _ in range(packed_seqlens.shape[0])]
        tensor = packed_seqlens.detach()
        if tensor.is_cuda:
            logger.warning(
                "packed_seqlens was a CUDA tensor; converting to Python list will sync. "
                "Prefer emitting packed_seqlens as a CPU int32 tensor in the collator."
            )
        tensor = tensor.cpu()

        if tensor.ndim == 1:
            out: list[list[int]] = []
            for value in tensor.tolist():
                length = int(value)
                segs = [length] if length > 0 else []
                if seq_len is not None and sum(segs) > seq_len:
                    raise ValueError(
                        f"packed_seqlens sums to {sum(segs)} > seq_len={seq_len}: {segs}"
                    )
                out.append(segs)
            return out

        if tensor.ndim == 2:
            out = []
            for row in tensor.tolist():
                segs = [int(x) for x in row if int(x) > 0]
                if seq_len is not None and sum(segs) > seq_len:
                    raise ValueError(
                        f"packed_seqlens sums to {sum(segs)} > seq_len={seq_len}: {segs}"
                    )
                out.append(segs)
            return out

        raise ValueError(
            "packed_seqlens tensor must be rank 1 or 2, got "
            f"shape={tuple(tensor.shape)}"
        )

    if isinstance(packed_seqlens, list):
        out: list[list[int]] = []
        for row in packed_seqlens:
            if row is None:
                out.append([])
                continue
            segs = [int(x) for x in row if int(x) > 0]
            if seq_len is not None and sum(segs) > seq_len:
                raise ValueError(
                    f"packed_seqlens sums to {sum(segs)} > seq_len={seq_len}: {segs}"
                )
            out.append(segs)
        return out

    raise TypeError(f"Unsupported packed_seqlens type: {type(packed_seqlens).__name__}")


class SwiGLU(nn.Module):
    """Native SwiGLU implementation (unpacked w1/w2/w3).

    Note: This layout is intentionally compatible with Liger's SwiGLU modules
    (gate/up/down projections). A future Liger swap should preserve the reduced
    2/3 intermediate size and the "swiglu" activation flag used in configs.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        """Initialize the SwiGLU block.

        :param int in_features: Input feature dimension.
        :param int | None hidden_features: Hidden feature dimension.
        :param int | None out_features: Output feature dimension.
        :param bool bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Output tensor.
        """
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))


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
        vocab_size: int = 32064,
        pad_token_id: int = 0,
        max_length: int = 1024,
        flash_attention: bool = True,
        base_scale: float = 1.0 / (960.0**0.5),
        ngpt: bool = False,
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
        :param bool flash_attention: Whether to use xFormers memory-efficient attention.
        :param float base_scale: Base scaling factor for NGPT.
        :param bool ngpt: Whether to enable NGPT mode.
        :param Any kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        # Core dims
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size

        # Dropout: accept legacy 'dropout_prob' as alias
        if "dropout_prob" in kwargs and "dropout" not in kwargs:
            warnings.warn(
                "NeoBERTConfig: 'dropout_prob' is deprecated; use 'dropout' instead.",
                UserWarning,
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

        # Positional length: accept HF-style 'max_position_embeddings'
        if (
            "max_position_embeddings" in kwargs
            and kwargs["max_position_embeddings"] is not None
        ):
            warnings.warn(
                "NeoBERTConfig: 'max_position_embeddings' is deprecated for the "
                "training model; use 'max_length' when constructing configs.",
                UserWarning,
                stacklevel=2,
            )
            self.max_length = int(kwargs["max_position_embeddings"])
        else:
            self.max_length = max_length
        # Keep the HF-style attribute for compatibility with downstream tooling.
        self.max_position_embeddings = self.max_length

        self.flash_attention = flash_attention
        self.base_scale = base_scale
        self.ngpt = ngpt

        # Store any extra kwargs for reference
        self.kwargs = kwargs


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

        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                # Future Liger integration note:
                # Liger expects config.intermediate_size directly and checks hidden_act in
                # {"silu","swish"}. We currently use a 2/3 reduction + "swiglu" flag, so
                # a wrapper/adapter will be needed when swapping implementations.
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8.
                intermediate_size = swiglu_intermediate_size(config.intermediate_size)
                self.ffn = SwiGLU(
                    config.hidden_size,
                    intermediate_size,
                    config.hidden_size,
                    bias=False,
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
            RMSNorm(config.hidden_size, config.norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )
        self.ffn_norm = (
            RMSNorm(config.hidden_size, config.norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        packed_seqlens: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Run the encoder block forward pass.

        :param torch.Tensor x: Input tensor.
        :param torch.Tensor pad_mask: Additive attention mask.
        :param torch.Tensor freqs_cis: Rotary embedding frequencies.
        :param list[list[int]] | torch.Tensor | None packed_seqlens: Packed segment lengths.
        :return torch.Tensor: Updated hidden states.
        """
        x = x + self._att_block(
            self.attention_norm(x), pad_mask, freqs_cis, packed_seqlens
        )
        x = x + self._ff_block(self.ffn_norm(x))
        return x

    def _att_block(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        packed_seqlens: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Apply the attention sub-layer.

        :param torch.Tensor x: Normalized hidden states.
        :param torch.Tensor pad_mask: Additive attention mask.
        :param torch.Tensor freqs_cis: Rotary embedding frequencies.
        :param list[list[int]] | torch.Tensor | None packed_seqlens: Packed segment lengths.
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

        if self.config.flash_attention:
            if not XFORMERS_AVAILABLE:
                raise ImportError(
                    "Flash attention requires xformers. Install with: pip install xformers. "
                    f"Import error: {XFORMERS_ERROR}"
                )
            dropout_p = self.config.dropout if self.training else 0.0
            if packed_seqlens is not None:
                attn = _packed_flash_attention(
                    xq, xk, xv, packed_seqlens, dropout_p=dropout_p
                )
            else:
                if pad_mask is not None:
                    attn_bias = _prepare_xformers_bias(
                        pad_mask,
                        self.config.num_attention_heads,
                        seq_len,
                        xq.dtype,
                        xq.device,
                    )
                    attn = memory_efficient_attention(
                        query=xq,
                        key=xk,
                        value=xv,
                        attn_bias=attn_bias,
                        p=dropout_p,
                    )
                else:
                    attn = memory_efficient_attention(
                        query=xq,
                        key=xk,
                        value=xv,
                        attn_bias=None,
                        p=dropout_p,
                    )
        else:
            if packed_seqlens is not None:
                raise ValueError(
                    "Packed sequences require flash_attention with xFormers."
                )
            # Use explicit scale for consistency with NormEncoderBlock.
            softmax_scale = 1.0 / math.sqrt(self.config.dim_head)
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention_compat(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout if self.training else 0,
                scale=softmax_scale,
            ).transpose(1, 2)

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


class NormEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the normalized encoder block.

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

        self.c_fc = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(
            self.attn_alpha_init_scaling * torch.ones(config.hidden_size)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(
            self.mlp_alpha_init_scaling * torch.ones(config.hidden_size)
        )

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(
            self.sqk_init_scaling * torch.ones(config.hidden_size)
        )

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(2 * config.intermediate_size)
        )

    def justnorm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization across the last dimension.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Normalized tensor.
        """
        return x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        packed_seqlens: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Run the normalized encoder block forward pass.

        :param torch.Tensor x: Input tensor.
        :param torch.Tensor pad_mask: Additive attention mask.
        :param torch.Tensor freqs_cis: Rotary embedding frequencies.
        :param list[list[int]] | None packed_seqlens: Packed segment lengths.
        :return torch.Tensor: Updated hidden states.
        """
        x_attn = self._att_block(x, pad_mask, freqs_cis, packed_seqlens)

        lr = self.attn_alpha * (
            self.attn_alpha_init_value / self.attn_alpha_init_scaling
        )
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_attn)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        x_ff = self._ff_block(x)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_ff)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        return x

    def _att_block(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        packed_seqlens: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Apply the attention sub-layer.

        :param torch.Tensor x: Input tensor.
        :param torch.Tensor pad_mask: Additive attention mask.
        :param torch.Tensor freqs_cis: Rotary embedding frequencies.
        :param list[list[int]] | None packed_seqlens: Packed segment lengths.
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

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1,
            1,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        xq = sqk * self.justnorm(xq)
        xk = sqk * self.justnorm(xk)

        softmax_scale = (
            self.config.hidden_size / self.config.num_attention_heads
        ) ** 0.5

        if self.config.flash_attention:
            if not XFORMERS_AVAILABLE:
                raise ImportError(
                    "Flash attention requires xformers. Install with: pip install xformers. "
                    f"Import error: {XFORMERS_ERROR}"
                )
            dropout_p = self.config.dropout if self.training else 0.0
            if packed_seqlens is not None:
                attn = _packed_flash_attention(
                    xq,
                    xk,
                    xv,
                    packed_seqlens,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )
            else:
                if pad_mask is not None:
                    attn_bias = _prepare_xformers_bias(
                        pad_mask,
                        self.config.num_attention_heads,
                        seq_len,
                        xq.dtype,
                        xq.device,
                    )
                    attn = memory_efficient_attention(
                        query=xq,
                        key=xk,
                        value=xv,
                        attn_bias=attn_bias,
                        p=dropout_p,
                        scale=softmax_scale,
                    )
                else:
                    attn = memory_efficient_attention(
                        query=xq,
                        key=xk,
                        value=xv,
                        attn_bias=None,
                        p=dropout_p,
                        scale=softmax_scale,
                    )
        else:
            if packed_seqlens is not None:
                raise ValueError(
                    "Packed sequences require flash_attention with xFormers."
                )
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention_compat(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout if self.training else 0,
                scale=softmax_scale,
            ).transpose(1, 2)

        return self.resid_dropout(
            self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size))
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward sub-layer.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Feed-forward output.
        """
        uv = self.c_fc(x)
        suv = self.suv * (
            (self.suv_init_value / self.suv_init_scaling)
            * (self.config.hidden_size**0.5)
        )
        uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        x = u * self.silu(v)
        x = self.mlp_c_proj(x)

        return self.ffn_dropout(x)


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

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        self.layer_norm = (
            RMSNorm(config.hidden_size, config.norm_eps)
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
        packed_seqlens: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Run the NeoBERT encoder forward pass.

        :param torch.Tensor src: Input token IDs.
        :param torch.Tensor | None pad_mask: Additive attention mask.
        :param list[list[int]] | None packed_seqlens: Packed segment lengths.
        :return torch.Tensor: Encoded hidden states.
        """
        seq_len = src.shape[1]
        packed_seqlens = _normalize_packed_seqlens(packed_seqlens, seq_len=seq_len)
        if self.config.flash_attention:
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
            if packed_seqlens is not None:
                if not XFORMERS_AVAILABLE:
                    raise ImportError(
                        "Packed sequences require xformers. Install with: pip install xformers. "
                        f"Import error: {XFORMERS_ERROR}"
                    )
                if pad_mask is not None:
                    logger.warning(
                        "packed_seqlens provided; ignoring pad_mask for packed attention."
                    )
                    pad_mask = None
        elif packed_seqlens is not None:
            raise ValueError("Packed sequences require flash_attention with xFormers.")

        # Normalize to broadcast-friendly shapes to avoid O(S^2) materialization.
        if pad_mask is not None and torch.is_tensor(pad_mask):
            # Expect additive masks (0 for keep, -inf for masked positions).
            pad_mask = _normalize_pad_mask(pad_mask)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            seq_len = src.shape[1]
            if seq_len > self.config.max_length:
                warnings.warn(
                    f"Sequence length {seq_len} exceeds max_length {self.config.max_length}; "
                    "RoPE cache will grow to accommodate. Consider truncating inputs.",
                    stacklevel=2,
                )
            target_len = max(self.config.max_length, seq_len)
            if (
                self.freqs_cis.numel() == 0
                or self.freqs_cis.device != src.device
                or self.freqs_cis.shape[0] < target_len
            ):
                self.freqs_cis = precompute_freqs_cis(
                    self.config.dim_head, target_len, device=src.device
                )
            freqs_cis = self.freqs_cis[:seq_len]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            # Content-based positions: padding stays at index 0, and left-padding
            # does not shift token positions (padding-invariant positional IDs).
            # Keep this in sync with the HF create_position_ids_from_input_ids logic.
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
            x += self.positional_embedding(incremental_indices.long())

        # Transformer encoder
        for layer in self.transformer_encoder:
            if self.gradient_checkpointing and self.training:
                # Capture mask + rotary frequencies in closure so checkpoint only sees Tensor inputs.
                def custom_forward(
                    hidden_states: torch.Tensor, layer: EncoderBlock = layer
                ) -> torch.Tensor:
                    """Wrap the encoder layer for checkpointing.

                    :param torch.Tensor hidden_states: Hidden states to process.
                    :param EncoderBlock layer: Encoder layer bound for checkpointing.
                    :return torch.Tensor: Updated hidden states.
                    """
                    return layer(hidden_states, pad_mask, freqs_cis, packed_seqlens)

                x = checkpoint(
                    custom_forward,
                    x,
                    preserve_rng_state=True,
                    use_reentrant=False,
                )
            else:
                x = layer(x, pad_mask, freqs_cis, packed_seqlens)

        # Final normalization layer
        x = self.layer_norm(x)

        # Return the output of the last hidden layer
        return x


class NormNeoBERT(NeoBERTPreTrainedModel):
    """NeoBERT encoder with normalized residuals."""

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the normalized NeoBERT encoder.

        :param NeoBERTConfig config: Model configuration.
        """
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

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

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(NormEncoderBlock(config))

        # Initialize weights and apply final processing
        self.post_init()
        self.gradient_checkpointing = False

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=config.base_scale / math.sqrt(2 * config.num_hidden_layers),
                )

    def forward(
        self,
        src: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        packed_seqlens: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Run the normalized encoder forward pass.

        :param torch.Tensor src: Input token IDs.
        :param torch.Tensor | None pad_mask: Additive attention mask.
        :param list[list[int]] | None packed_seqlens: Packed segment lengths.
        :return torch.Tensor: Encoded hidden states.
        """
        seq_len = src.shape[1]
        packed_seqlens = _normalize_packed_seqlens(packed_seqlens, seq_len=seq_len)
        if self.config.flash_attention:
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
            if packed_seqlens is not None:
                if not XFORMERS_AVAILABLE:
                    raise ImportError(
                        "Packed sequences require xformers. Install with: pip install xformers. "
                        f"Import error: {XFORMERS_ERROR}"
                    )
                if pad_mask is not None:
                    logger.warning(
                        "packed_seqlens provided; ignoring pad_mask for packed attention."
                    )
                    pad_mask = None
        elif packed_seqlens is not None:
            raise ValueError("Packed sequences require flash_attention with xFormers.")

        # Normalize to broadcast-friendly shapes to avoid O(S^2) materialization.
        if pad_mask is not None and torch.is_tensor(pad_mask):
            # Expect additive masks (0 for keep, -inf for masked positions).
            pad_mask = _normalize_pad_mask(pad_mask)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            seq_len = src.shape[1]
            if seq_len > self.config.max_length:
                warnings.warn(
                    f"Sequence length {seq_len} exceeds max_length {self.config.max_length}; "
                    "RoPE cache will grow to accommodate. Consider truncating inputs.",
                    stacklevel=2,
                )
            target_len = max(self.config.max_length, seq_len)
            if (
                self.freqs_cis.numel() == 0
                or self.freqs_cis.device != src.device
                or self.freqs_cis.shape[0] < target_len
            ):
                self.freqs_cis = precompute_freqs_cis(
                    self.config.dim_head, target_len, device=src.device
                )
            freqs_cis = self.freqs_cis[:seq_len]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            # Content-based positions: padding stays at index 0, and left-padding
            # does not shift token positions (padding-invariant positional IDs).
            # Keep this in sync with the HF create_position_ids_from_input_ids logic.
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
            x += self.positional_embedding(incremental_indices.long())

        # Transformer encoder
        for layer in self.transformer_encoder:
            if self.gradient_checkpointing and self.training:

                def custom_forward(
                    hidden_states: torch.Tensor, layer: NormEncoderBlock = layer
                ) -> torch.Tensor:
                    """Wrap the encoder layer for checkpointing.

                    :param torch.Tensor hidden_states: Hidden states to process.
                    :param NormEncoderBlock layer: Encoder layer bound for checkpointing.
                    :return torch.Tensor: Updated hidden states.
                    """
                    return layer(hidden_states, pad_mask, freqs_cis, packed_seqlens)

                x = checkpoint(
                    custom_forward,
                    x,
                    preserve_rng_state=True,
                    use_reentrant=False,
                )
            else:
                x = layer(x, pad_mask, freqs_cis, packed_seqlens)

        # NormNeoBERT applies normalization inside each block; no final norm here.
        # Return the output of the last hidden layer
        return x


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    """NeoBERT with a language modeling head."""

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the language modeling head.

        :param NeoBERTConfig config: Model configuration.
        """
        super().__init__(config)

        self.config = config

        self.model = NormNeoBERT(config) if self.config.ngpt else NeoBERT(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(
        self,
        src: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        packed_seqlens: Optional[list[list[int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the LM head forward pass.

        :param torch.Tensor src: Input token IDs.
        :param torch.Tensor | None pad_mask: Additive attention mask.
        :param list[list[int]] | None packed_seqlens: Packed segment lengths.
        :return dict[str, torch.Tensor]: Hidden states and logits.
        """
        hidden_representation = self.model.forward(src, pad_mask, packed_seqlens)
        logits = self.decoder(hidden_representation)

        return {"hidden_representation": hidden_representation, "logits": logits}


class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):
    """NeoBERT with a classification head."""

    def __init__(
        self,
        config: NeoBERTConfig,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_init_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        """Initialize the sequence classification head.

        :param NeoBERTConfig config: Model configuration.
        :param int num_labels: Number of output labels.
        :param float classifier_dropout: Dropout probability.
        :param float classifier_init_range: Init range for classifier.
        :param Any kwargs: Unused extra arguments for compatibility.
        """
        super().__init__(config)

        self.config = config

        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.classifier_init_range = classifier_init_range

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize classifier weights.

        :param nn.Module module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self, src: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Run the classification head forward pass.

        :param torch.Tensor src: Input token IDs.
        :param torch.Tensor | None pad_mask: Additive attention mask.
        :return dict[str, torch.Tensor]: Hidden states and logits.
        """
        hidden_representation = self.model.forward(src, pad_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        return {"hidden_representation": hidden_representation, "logits": logits}


class NeoBERTHFForSequenceClassification(NeoBERTPreTrainedModel):
    """Hugging Face compatible NeoBERT sequence classifier."""

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the HF-compatible classifier.

        :param NeoBERTConfig config: Model configuration.
        """
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize classifier weights.

        :param nn.Module module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput | tuple:
        """Forward pass for sequence classification.

        :param torch.Tensor | None input_ids: Input token IDs.
        :param torch.Tensor | None attention_mask: Attention mask.
        :param torch.Tensor | None token_type_ids: Token type IDs.
        :param torch.Tensor | None position_ids: Position IDs.
        :param torch.Tensor | None inputs_embeds: Optional input embeddings.
        :param torch.Tensor | None labels: Optional labels for loss.
        :param bool | None output_attentions: Whether to return attentions.
        :param bool | None output_hidden_states: Whether to return hidden states.
        :param bool | None return_dict: Whether to return dict outputs.
        :return SequenceClassifierOutput | tuple: Model outputs.
        """
        # Convert attention masks to additive mask (-inf for masked, 0 for keep).
        if attention_mask is not None:
            if attention_mask.dtype is torch.bool:
                additive_mask = torch.where(attention_mask, float(0.0), float("-inf"))
            elif attention_mask.is_floating_point() and attention_mask.min() < 0:
                additive_mask = attention_mask
            else:
                additive_mask = torch.where(
                    attention_mask == 0, float("-inf"), float(0.0)
                )
            if additive_mask.dtype != torch.float32:
                additive_mask = additive_mask.to(torch.float32)
        else:
            additive_mask = None
        hidden_representation = self.model.forward(input_ids, additive_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )


class NeoBERTForMTEB(NeoBERTPreTrainedModel):
    """NeoBERT wrapper for MTEB-style encoding."""

    config_class = NeoBERTConfig

    def __init__(
        self,
        config: NeoBERTConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        batch_size: int = 8,
        pooling: str = "avg",
        **kwargs: Any,
    ) -> None:
        """Initialize the MTEB encoder wrapper.

        :param NeoBERTConfig config: Model configuration.
        :param PreTrainedTokenizerFast tokenizer: Tokenizer for text inputs.
        :param int max_length: Maximum sequence length.
        :param int batch_size: Encoding batch size.
        :param str pooling: Pooling strategy (avg/cls).
        :param Any kwargs: Unused extra arguments for compatibility.
        """
        super().__init__(config)

        self.config = config
        self.model = NeoBERT(config)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def encode_queries(self, queries: List[str], **kwargs: Any) -> np.ndarray:
        """Encode a list of queries.

        :param list[str] queries: Query strings to encode.
        :param Any kwargs: Additional encoding arguments.
        :return np.ndarray: Encoded query embeddings.
        """
        if "instructions" in kwargs:
            if kwargs["instructions"] is not None:
                queries = [
                    (query + " " + kwargs["instructions"][query]).strip()
                    for query in queries
                ]
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            new_kwargs = kwargs

        return self.encode(
            queries,
            **new_kwargs,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]] | Dict[str, List[str]],
        batch_size: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a corpus of documents.

        :param list[dict[str, str]] | dict[str, list[str]] corpus: Corpus inputs.
        :param int batch_size: Encoding batch size.
        :param Any kwargs: Additional encoding arguments.
        :return np.ndarray: Encoded corpus embeddings.
        """
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            if isinstance(corpus[0], dict):
                sentences = [
                    (doc["title"] + " " + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                    for doc in corpus
                ]
            else:
                sentences = corpus

        if "instructions" in kwargs:  # not used on the doc side
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            new_kwargs = kwargs

        return self.encode(
            sentences,
            **new_kwargs,
        )

    @torch.no_grad()
    def encode(self, sentences: list[str], **kwargs: Any) -> torch.Tensor:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Optional overrides for the DataLoader.
                - num_workers (int): DataLoader worker processes (default: 0).
                - pin_memory (bool | None): Pin CPU memory for CUDA transfer. Defaults
                  to ``True`` on CUDA devices, otherwise ``False``.

        Returns:
            The encoded sentences.
        """

        # Respect the model's current device to avoid CPU/GPU mismatches.
        param = next(self.parameters())
        device = param.device
        # Keep additive masks in float32 for numerical stability (match training).
        mask_dtype = torch.float32
        num_workers = int(kwargs.pop("num_workers", 0))
        pin_memory = kwargs.pop("pin_memory", None)
        if pin_memory is None:
            pin_memory = device.type == "cuda"

        def _transform_func(
            tokenizer: PreTrainedTokenizerFast, x: Dict[str, List]
        ) -> Dict[str, List]:
            """Tokenize a batch of input texts.

            :param PreTrainedTokenizerFast tokenizer: Tokenizer to apply.
            :param dict[str, list] x: Batch with ``input_texts``.
            :return dict[str, list]: Tokenized batch.
            """
            batch_dict = tokenizer(
                x["input_texts"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_token_type_ids=False,
            )

            return batch_dict

        dataset: Dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        )

        encodings = []
        for batch in tqdm(
            dataloader, desc="encoding", mininterval=10, disable=len(sentences) < 128
        ):
            input_ids = batch["input_ids"].to(device)

            pad_mask = batch["attention_mask"].to(device)
            xformers_mask = torch.where(pad_mask == 1, float(0.0), float("-inf")).type(
                mask_dtype
            )

            outputs = self.model(input_ids, xformers_mask)

            if self.pooling == "avg":
                outputs = outputs * pad_mask.unsqueeze(-1).expand(
                    -1, -1, outputs.shape[-1]
                )
                outputs = outputs.sum(dim=1) / pad_mask.to(device).sum(dim=1).unsqueeze(
                    -1
                )
            else:
                outputs = outputs[:, 0, :]

            encodings.append(outputs.cpu().numpy())

        return np.concatenate(encodings, axis=0)
