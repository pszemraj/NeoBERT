"""Attention backend dispatch: PyTorch SDPA + flash_attn_varlen."""

from dataclasses import dataclass
import logging
import math
from typing import Literal, Optional

import torch

from neobert.modeling_utils import scaled_dot_product_attention_compat

logger = logging.getLogger(__name__)
PackedSeqLens = torch.Tensor | list[list[int]]


@dataclass(frozen=True)
class PackedFlashMetadata:
    """Reusable flash-attn varlen metadata for one packed batch."""

    flat_token_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch_size: int
    seq_len: int


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


try:
    _dynamo_disable = torch.compiler.disable
except (AttributeError, RuntimeError):
    _dynamo_disable = torch._dynamo.disable  # type: ignore[attr-defined]


def _is_torch_compiling() -> bool:
    """Return whether execution is inside a ``torch.compile`` trace.

    :return bool: ``True`` when tracing/compiling is active, else ``False``.
    """
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling):
            return bool(is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling):
            return bool(is_compiling())
    return False


def canonicalize_attn_backend(
    requested: str,
) -> Literal["sdpa", "flash_attn_varlen"]:
    """Canonicalize the attention backend string without environment checks.

    :param str requested: One of ``"sdpa"``, ``"flash_attn_varlen"``, ``"flash_attn"``, or ``"flash"``.
    :return str: Canonical backend name.
    :raises ValueError: If *requested* is unknown.
    """
    normalized = str(requested).lower().strip()
    if normalized == "sdpa":
        return "sdpa"
    if normalized in ("flash_attn_varlen", "flash_attn", "flash"):
        return "flash_attn_varlen"
    raise ValueError(
        f"Unknown attn_backend '{requested}'. Expected: 'sdpa' or 'flash_attn_varlen'."
    )


def resolve_attn_backend(
    requested: str,
) -> Literal["sdpa", "flash_attn_varlen"]:
    """Resolve and validate the attention backend against installed packages.

    :param str requested: One of ``"sdpa"`` or ``"flash_attn_varlen"`` (aliases accepted).
    :return str: Resolved backend name.
    :raises ImportError: If flash-attn backend is requested but unavailable.
    """
    backend = canonicalize_attn_backend(requested)
    if backend == "flash_attn_varlen" and not FLASH_ATTN_AVAILABLE:
        raise ImportError(
            f"attn_backend='{requested}' requested but flash-attn is not available: "
            f"{FLASH_ATTN_ERROR}"
        )
    return backend


def resolve_runtime_attn_backend(
    requested: str,
    *,
    fallback_to_sdpa: bool = False,
) -> Literal["sdpa", "flash_attn_varlen"]:
    """Resolve backend against installed packages and runtime CUDA availability.

    :param str requested: Requested backend (aliases accepted).
    :param bool fallback_to_sdpa: Whether to fallback to ``"sdpa"`` on unsupported flash-attn runtime.
    :return str: Resolved backend.
    :raises ImportError: If flash-attn is requested but unavailable and fallback is disabled.
    :raises RuntimeError: If flash-attn is requested without CUDA and fallback is disabled.
    """
    try:
        backend = resolve_attn_backend(requested)
    except ImportError:
        if fallback_to_sdpa:
            logger.warning(
                f"attn_backend='{requested}' requested but flash-attn is not available. "
                "Falling back to attn_backend='sdpa'."
            )
            return "sdpa"
        raise
    if backend != "flash_attn_varlen":
        return backend

    if torch.cuda.is_available():
        return backend

    message = (
        f"attn_backend='{requested}' requires CUDA for packed flash-attn kernels, "
        "but CUDA is not available."
    )
    if fallback_to_sdpa:
        logger.warning(f"{message} Falling back to attn_backend='sdpa'.")
        return "sdpa"
    raise RuntimeError(message)


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


def _normalize_packed_seqlens_tensor(
    packed_seqlens: PackedSeqLens,
    *,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Normalize packed metadata to a fixed-rank int32 tensor.

    :param torch.Tensor | list[list[int]] packed_seqlens: Packed segment lengths.
    :param int batch_size: Expected batch size.
    :param int seq_len: Expected padded sequence length.
    :return torch.Tensor: ``int32`` tensor of shape ``[B, N]``.
    """
    if torch.is_tensor(packed_seqlens):
        tensor = packed_seqlens.detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 2:
            raise ValueError(
                "packed_seqlens tensor must be rank 1 or 2, got "
                f"shape={tuple(tensor.shape)}"
            )
        tensor = tensor.to(torch.int32)
    else:
        normalized_rows: list[list[int]] = []
        max_segments = 0
        for row in packed_seqlens:
            if row is None:
                segs: list[int] = []
            else:
                segs = [int(x) for x in row if int(x) > 0]
            normalized_rows.append(segs)
            max_segments = max(max_segments, len(segs))
        tensor = torch.zeros((len(normalized_rows), max_segments), dtype=torch.int32)
        for idx, segs in enumerate(normalized_rows):
            if not segs:
                continue
            tensor[idx, : len(segs)] = torch.tensor(segs, dtype=torch.int32)

    if tensor.shape[0] != batch_size:
        raise ValueError(
            "packed_seqlens batch mismatch: "
            f"{tensor.shape[0]} != batch_size={batch_size}"
        )
    sums = tensor.clamp_min(0).sum(dim=1)
    bad = sums > seq_len
    if bad.any():
        bad_idx = int(torch.where(bad)[0][0].item())
        raise ValueError(
            "packed_seqlens sums exceed seq_len "
            f"(row={bad_idx}, sum={int(sums[bad_idx].item())}, seq_len={seq_len})."
        )
    return tensor


@_dynamo_disable
def prepare_packed_flash_metadata(
    packed_seqlens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> PackedFlashMetadata:
    """Build reusable flattening metadata for flash-attn varlen kernels.

    :param torch.Tensor packed_seqlens: Packed lengths ``[B, N]``.
    :param int batch_size: Batch size of the attention input.
    :param int seq_len: Padded sequence length of the attention input.
    :param torch.device device: Device that will run the attention kernel.
    :return PackedFlashMetadata: Precomputed metadata tensors.
    """
    seg_lens_cuda = packed_seqlens.clamp_min(0).to(device=device, dtype=torch.int32)
    if seg_lens_cuda.ndim != 2:
        raise ValueError(
            "packed_seqlens must be rank 2 after normalization, got "
            f"shape={tuple(seg_lens_cuda.shape)}"
        )
    if seg_lens_cuda.shape[0] != batch_size:
        raise ValueError(
            "packed_seqlens batch mismatch: "
            f"{seg_lens_cuda.shape[0]} != batch_size={batch_size}"
        )

    valid_tokens = seg_lens_cuda.sum(dim=1, dtype=torch.int64)
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long), valid_tokens
    )
    if batch_ids.shape[0] == 0:
        return PackedFlashMetadata(
            flat_token_indices=torch.empty(0, dtype=torch.long, device=device),
            cu_seqlens=torch.zeros(1, dtype=torch.int32, device=device),
            max_seqlen=0,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    token_offsets = torch.cumsum(valid_tokens, dim=0) - valid_tokens
    stream_positions = torch.arange(batch_ids.shape[0], device=device, dtype=torch.long)
    positions_in_batch = stream_positions - token_offsets.index_select(0, batch_ids)
    flat_token_indices = batch_ids * seq_len + positions_in_batch

    seg_lens_flat = seg_lens_cuda[seg_lens_cuda > 0]
    if seg_lens_flat.numel() == 0:
        return PackedFlashMetadata(
            flat_token_indices=torch.empty(0, dtype=torch.long, device=device),
            cu_seqlens=torch.zeros(1, dtype=torch.int32, device=device),
            max_seqlen=0,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    seg_tokens_total = seg_lens_flat.sum(dtype=torch.int64)
    flattened_tokens = batch_ids.shape[0]
    seg_tokens_total_int = int(seg_tokens_total.item())
    flattened_tokens_int = int(flattened_tokens)
    if seg_tokens_total_int != flattened_tokens_int:
        raise ValueError(
            "packed_seqlens metadata is inconsistent: sum of positive segment lengths "
            f"({seg_tokens_total_int}) does not match flattened token count "
            f"({flattened_tokens_int})."
        )

    cu_seqlens = torch.empty(
        seg_lens_flat.shape[0] + 1, dtype=torch.int32, device=device
    )
    cu_seqlens[0] = 0
    cu_seqlens[1:] = seg_lens_flat.cumsum(dim=0, dtype=torch.int32)
    if _is_torch_compiling():
        max_seqlen = seq_len
    else:
        max_seqlen = int(seg_lens_flat.max().item())

    return PackedFlashMetadata(
        flat_token_indices=flat_token_indices,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        seq_len=seq_len,
    )


@_dynamo_disable
def _flash_varlen_attention(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    packed_seqlens: torch.Tensor,
    dropout_p: float,
    scale: float | None = None,
    packed_metadata: PackedFlashMetadata | None = None,
) -> torch.Tensor:
    """Run flash_attn_varlen_func on packed sequences.

    Input tensors have shape ``(B, S, H, D)`` where padded positions may exist.
    We flatten valid tokens, run the varlen kernel, and reconstruct.

    :param torch.Tensor xq: Query ``(B, S, H, D)``.
    :param torch.Tensor xk: Key ``(B, S, H, D)``.
    :param torch.Tensor xv: Value ``(B, S, H, D)``.
    :param torch.Tensor packed_seqlens: Packed lengths ``[B, N]``.
    :param float dropout_p: Dropout probability.
    :param float | None scale: Softmax scale.
    :param PackedFlashMetadata | None packed_metadata: Optional reusable metadata.
    :return torch.Tensor: Output ``(B, S, H, D)``.
    """
    assert _flash_attn_varlen_func is not None
    batch_size, seq_len, num_heads, head_dim = xq.shape
    if packed_metadata is None:
        packed_metadata = prepare_packed_flash_metadata(
            packed_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
            device=xq.device,
        )
    elif packed_metadata.batch_size != batch_size or packed_metadata.seq_len != seq_len:
        raise ValueError(
            "packed flash metadata shape mismatch: "
            f"metadata=({packed_metadata.batch_size}, {packed_metadata.seq_len}) "
            f"input=({batch_size}, {seq_len})"
        )

    flat_token_indices = packed_metadata.flat_token_indices
    if flat_token_indices.numel() == 0:
        return torch.zeros_like(xq)
    if flat_token_indices.device != xq.device:
        raise ValueError(
            "packed flash metadata must be on the same device as attention inputs: "
            f"metadata={flat_token_indices.device}, input={xq.device}"
        )

    xq_flat = xq.reshape(batch_size * seq_len, num_heads, head_dim)
    xk_flat = xk.reshape(batch_size * seq_len, num_heads, head_dim)
    xv_flat = xv.reshape(batch_size * seq_len, num_heads, head_dim)
    q_flat = xq_flat.index_select(0, flat_token_indices)
    k_flat = xk_flat.index_select(0, flat_token_indices)
    v_flat = xv_flat.index_select(0, flat_token_indices)

    softmax_scale = scale if scale is not None else (1.0 / math.sqrt(head_dim))

    out_flat = _flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q=packed_metadata.cu_seqlens,
        cu_seqlens_k=packed_metadata.cu_seqlens,
        max_seqlen_q=packed_metadata.max_seqlen,
        max_seqlen_k=packed_metadata.max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
    )

    # Reconstruct back to (B, S, H, D)
    attn_flat = torch.zeros_like(xq_flat)
    attn_flat.index_copy_(0, flat_token_indices, out_flat)
    return attn_flat.view(batch_size, seq_len, num_heads, head_dim)


def _sdpa_packed_fallback(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    packed_seqlens: torch.Tensor,
    dropout_p: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Per-segment SDPA loop fallback for packed sequences.

    This is correct but slow; use flash_attn_varlen for production GPU workloads.

    :param torch.Tensor xq: Query ``(B, S, H, D)``.
    :param torch.Tensor xk: Key ``(B, S, H, D)``.
    :param torch.Tensor xv: Value ``(B, S, H, D)``.
    :param torch.Tensor packed_seqlens: Packed lengths ``[B, N]``.
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
    seg_lens = packed_seqlens.clamp_min(0)
    for b in range(seg_lens.shape[0]):
        start = 0
        for seg_len_tensor in seg_lens[b]:
            seg_len = int(seg_len_tensor.item())
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
    packed_seqlens: PackedSeqLens | None,
    dropout_p: float,
    scale: float | None,
    attn_backend: str,
    packed_flash_metadata: PackedFlashMetadata | None = None,
) -> torch.Tensor:
    """Unified attention dispatch for SDPA and flash_attn_varlen.

    :param torch.Tensor xq: Query ``(B, S, H, D)``.
    :param torch.Tensor xk: Key ``(B, S, H, D)``.
    :param torch.Tensor xv: Value ``(B, S, H, D)``.
    :param torch.Tensor | None pad_mask: Additive mask for SDPA (broadcast-friendly).
    :param torch.Tensor | list[list[int]] | None packed_seqlens: Packed segment lengths.
    :param float dropout_p: Dropout probability.
    :param float | None scale: Softmax scale factor.
    :param str attn_backend: ``"sdpa"`` or ``"flash_attn_varlen"``.
    :param PackedFlashMetadata | None packed_flash_metadata: Reusable flash varlen metadata.
    :return torch.Tensor: Attention output ``(B, S, H, D)``.
    """
    attn_backend = canonicalize_attn_backend(attn_backend)
    if packed_seqlens is not None:
        if packed_flash_metadata is not None and torch.is_tensor(packed_seqlens):
            # Reuse already-normalized tensor path when metadata is precomputed
            # once in the model forward; avoid per-layer revalidation overhead.
            packed_tensor = packed_seqlens.detach()
            if packed_tensor.ndim == 1:
                packed_tensor = packed_tensor.unsqueeze(1)
            if packed_tensor.ndim != 2:
                raise ValueError(
                    "packed_seqlens tensor must be rank 1 or 2, got "
                    f"shape={tuple(packed_tensor.shape)}"
                )
            if packed_tensor.shape[0] != xq.shape[0]:
                raise ValueError(
                    "packed_seqlens batch mismatch: "
                    f"{packed_tensor.shape[0]} != batch_size={xq.shape[0]}"
                )
            packed_tensor = packed_tensor.to(device=xq.device, dtype=torch.int32)
        else:
            packed_tensor = _normalize_packed_seqlens_tensor(
                packed_seqlens,
                batch_size=xq.shape[0],
                seq_len=xq.shape[1],
            )
        # Packed sequences
        if attn_backend == "flash_attn_varlen":
            if not xq.is_cuda:
                raise RuntimeError(
                    "attn_backend='flash_attn_varlen' requires CUDA tensors for packed "
                    f"attention, but input is on {xq.device}."
                )
            if not FLASH_ATTN_AVAILABLE:
                raise ImportError(
                    "attn_backend='flash_attn_varlen' requested but flash-attn is not "
                    f"available: {FLASH_ATTN_ERROR}"
                )
            return _flash_varlen_attention(
                xq,
                xk,
                xv,
                packed_tensor,
                dropout_p,
                scale,
                packed_metadata=packed_flash_metadata,
            )
        # SDPA per-segment fallback (works on CPU + GPU)
        return _sdpa_packed_fallback(xq, xk, xv, packed_tensor, dropout_p, scale)

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
