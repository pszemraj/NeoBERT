"""Attention backend dispatch: PyTorch SDPA + flash_attn_varlen."""

import logging
import math
from typing import Literal, Optional

import torch

from ..modeling_utils import scaled_dot_product_attention_compat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import-time flash-attn availability check
# ---------------------------------------------------------------------------
FLASH_ATTN_AVAILABLE: bool = False
FLASH_ATTN_ERROR: Optional[str] = None

_flash_attn_varlen_func = None

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func  # type: ignore[no-redef]

    FLASH_ATTN_AVAILABLE = True
except (ImportError, RuntimeError) as exc:
    FLASH_ATTN_ERROR = str(exc)

_WARNED_SDPA_PACKED_GPU = False


def resolve_attn_backend(
    requested: str,
) -> Literal["sdpa", "flash_attn_varlen"]:
    """Resolve the attention backend to use.

    :param str requested: One of ``"sdpa"`` or ``"flash_attn_varlen"``.
    :return str: Resolved backend name.
    """
    requested = requested.lower().strip()
    if requested == "sdpa":
        return "sdpa"
    if requested in ("flash_attn_varlen", "flash_attn", "flash"):
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                f"attn_backend='{requested}' requested but flash-attn is not available: "
                f"{FLASH_ATTN_ERROR}"
            )
        return "flash_attn_varlen"
    raise ValueError(
        f"Unknown attn_backend '{requested}'. Expected: 'sdpa' or 'flash_attn_varlen'."
    )


# ---------------------------------------------------------------------------
# flash_attn_varlen helpers
# ---------------------------------------------------------------------------


def packed_seqlens_to_cu_seqlens(
    packed_seqlens: list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Convert nested packed segment lengths to cumulative sequence lengths.

    :param list[list[int]] packed_seqlens: Per-sample segment lengths.
    :param torch.device device: Device for the output tensor.
    :return tuple[torch.Tensor, int]: ``(cu_seqlens_int32, max_seqlen)``.
    """
    all_lens: list[int] = []
    for segs in packed_seqlens:
        all_lens.extend(segs)
    if not all_lens:
        return torch.zeros(1, dtype=torch.int32, device=device), 0
    cu = [0]
    for length in all_lens:
        cu.append(cu[-1] + length)
    max_seqlen = max(all_lens)
    return torch.tensor(cu, dtype=torch.int32, device=device), max_seqlen


def _flash_varlen_attention(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    packed_seqlens: list[list[int]],
    dropout_p: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Run flash_attn_varlen_func on packed sequences.

    Input tensors have shape ``(B, S, H, D)`` where padded positions may exist.
    We flatten valid tokens, run the varlen kernel, and reconstruct.

    :param torch.Tensor xq: Query ``(B, S, H, D)``.
    :param torch.Tensor xk: Key ``(B, S, H, D)``.
    :param torch.Tensor xv: Value ``(B, S, H, D)``.
    :param list[list[int]] packed_seqlens: Segment lengths per batch item.
    :param float dropout_p: Dropout probability.
    :param float | None scale: Softmax scale.
    :return torch.Tensor: Output ``(B, S, H, D)``.
    """
    assert _flash_attn_varlen_func is not None
    batch_size, seq_len, num_heads, head_dim = xq.shape

    # Flatten valid segments into a 1-D token stream
    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for b, segs in enumerate(packed_seqlens):
        start = 0
        for seg_len in segs:
            if seg_len <= 0:
                continue
            end = start + seg_len
            q_parts.append(xq[b, start:end])  # (seg_len, H, D)
            k_parts.append(xk[b, start:end])
            v_parts.append(xv[b, start:end])
            start = end

    if not q_parts:
        return torch.zeros_like(xq)

    q_flat = torch.cat(q_parts, dim=0)  # (total_tokens, H, D)
    k_flat = torch.cat(k_parts, dim=0)
    v_flat = torch.cat(v_parts, dim=0)

    cu_seqlens, max_seqlen = packed_seqlens_to_cu_seqlens(
        packed_seqlens, device=xq.device
    )

    softmax_scale = scale if scale is not None else (1.0 / math.sqrt(head_dim))

    out_flat = _flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
    )

    # Reconstruct back to (B, S, H, D)
    attn = torch.zeros_like(xq)
    seg_idx = 0
    for b, segs in enumerate(packed_seqlens):
        start = 0
        for seg_len in segs:
            if seg_len <= 0:
                continue
            end = start + seg_len
            token_count = seg_len
            attn[b, start:end] = out_flat[seg_idx : seg_idx + token_count]
            seg_idx += token_count
            start = end

    return attn


def _sdpa_packed_fallback(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    packed_seqlens: list[list[int]],
    dropout_p: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Per-segment SDPA loop fallback for packed sequences.

    This is correct but slow; use flash_attn_varlen for production GPU workloads.

    :param torch.Tensor xq: Query ``(B, S, H, D)``.
    :param torch.Tensor xk: Key ``(B, S, H, D)``.
    :param torch.Tensor xv: Value ``(B, S, H, D)``.
    :param list[list[int]] packed_seqlens: Segment lengths per batch item.
    :param float dropout_p: Dropout probability.
    :param float | None scale: Softmax scale.
    :return torch.Tensor: Output ``(B, S, H, D)``.
    """
    global _WARNED_SDPA_PACKED_GPU
    if xq.is_cuda and not _WARNED_SDPA_PACKED_GPU:
        logger.warning(
            "Using per-segment SDPA loop for packed sequences on GPU. "
            "This is slow; install flash-attn for production: pip install flash-attn"
        )
        _WARNED_SDPA_PACKED_GPU = True

    attn = torch.zeros_like(xq)
    for b, segs in enumerate(packed_seqlens):
        start = 0
        for seg_len in segs:
            if seg_len <= 0:
                continue
            end = start + seg_len
            # SDPA expects (B, H, S, D)
            q_seg = xq[b : b + 1, start:end].transpose(1, 2)
            k_seg = xk[b : b + 1, start:end].transpose(1, 2)
            v_seg = xv[b : b + 1, start:end].transpose(1, 2)
            out_seg = scaled_dot_product_attention_compat(
                query=q_seg,
                key=k_seg,
                value=v_seg,
                attn_mask=None,
                dropout_p=dropout_p,
                scale=scale,
            ).transpose(1, 2)
            attn[b, start:end] = out_seg[0]
            start = end
    return attn


# ---------------------------------------------------------------------------
# Unified attention dispatch
# ---------------------------------------------------------------------------


def attention_forward(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    pad_mask: torch.Tensor | None,
    packed_seqlens: list[list[int]] | None,
    dropout_p: float,
    scale: float | None,
    attn_backend: str,
) -> torch.Tensor:
    """Unified attention dispatch for SDPA and flash_attn_varlen.

    :param torch.Tensor xq: Query ``(B, S, H, D)``.
    :param torch.Tensor xk: Key ``(B, S, H, D)``.
    :param torch.Tensor xv: Value ``(B, S, H, D)``.
    :param torch.Tensor | None pad_mask: Additive mask for SDPA (broadcast-friendly).
    :param list[list[int]] | None packed_seqlens: Packed segment lengths.
    :param float dropout_p: Dropout probability.
    :param float | None scale: Softmax scale factor.
    :param str attn_backend: ``"sdpa"`` or ``"flash_attn_varlen"``.
    :return torch.Tensor: Attention output ``(B, S, H, D)``.
    """
    if packed_seqlens is not None:
        # Packed sequences
        if attn_backend == "flash_attn_varlen" and FLASH_ATTN_AVAILABLE:
            return _flash_varlen_attention(xq, xk, xv, packed_seqlens, dropout_p, scale)
        # SDPA per-segment fallback (works on CPU + GPU)
        return _sdpa_packed_fallback(xq, xk, xv, packed_seqlens, dropout_p, scale)

    # Unpacked: always use SDPA
    # SDPA expects (B, H, S, D), our input is (B, S, H, D)
    return scaled_dot_product_attention_compat(
        query=xq.transpose(1, 2),
        key=xk.transpose(1, 2),
        value=xv.transpose(1, 2),
        attn_mask=pad_mask,
        dropout_p=dropout_p,
        scale=scale,
    ).transpose(1, 2)
