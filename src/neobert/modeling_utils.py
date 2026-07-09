"""Shared modeling helpers for training and HF implementations."""

from __future__ import annotations

import inspect
import math
from typing import Any, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention


def packed_seqlens_to_tensor(packed_seqlens: Any) -> Optional[torch.Tensor]:
    """Convert packed segment lengths to a rank-2 int32 tensor.

    Tensor inputs retain their device. List inputs are padded with zeros on CPU;
    non-positive segment lengths are omitted.

    :param Any packed_seqlens: Rank-1/rank-2 tensor, nested list, or ``None``.
    :raises TypeError: If the input is neither a tensor nor a nested list.
    :raises ValueError: If a tensor input is not rank 1 or 2.
    :return torch.Tensor | None: Packed lengths shaped ``[batch, segments]``.
    """
    if packed_seqlens is None:
        return None
    if torch.is_tensor(packed_seqlens):
        tensor = packed_seqlens.detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 2:
            raise ValueError(
                "packed_seqlens tensor must be rank 1 or 2, got "
                f"shape={tuple(tensor.shape)}"
            )
        return tensor.to(torch.int32)
    if not isinstance(packed_seqlens, list):
        raise TypeError(
            f"Unsupported packed_seqlens type: {type(packed_seqlens).__name__}"
        )

    rows = [
        [] if row is None else [int(length) for length in row if int(length) > 0]
        for row in packed_seqlens
    ]
    width = max((len(row) for row in rows), default=0)
    tensor = torch.zeros((len(rows), width), dtype=torch.int32)
    for index, row in enumerate(rows):
        if row:
            tensor[index, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return tensor


def swiglu_intermediate_size(intermediate_size: int, multiple_of: int = 8) -> int:
    """Compute the reduced SwiGLU hidden size and round to a multiple.

    The SwiGLU feed-forward uses a 2/3 reduction (per the GLU paper) and rounds
    up to ``multiple_of`` for kernel alignment.

    :param int intermediate_size: Base MLP hidden size from config.
    :param int multiple_of: Alignment multiple (default: 8).
    :return int: Rounded SwiGLU hidden size.
    """
    reduced = int(2 * intermediate_size / 3)
    return multiple_of * ((reduced + multiple_of - 1) // multiple_of)


def is_torch_compiling() -> bool:
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


try:
    _SDPA_SUPPORTS_SCALE = (
        "scale" in inspect.signature(scaled_dot_product_attention).parameters
    )
except (TypeError, ValueError):  # pragma: no cover - defensive fallback.
    _SDPA_SUPPORTS_SCALE = False


def scaled_dot_product_attention_compat(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Dispatch SDPA with manual scaling for torch versions without ``scale=``.

    :param torch.Tensor query: Query tensor of shape (B, H, M, K).
    :param torch.Tensor key: Key tensor of shape (B, H, N, K).
    :param torch.Tensor value: Value tensor of shape (B, H, N, K).
    :param torch.Tensor | None attn_mask: Optional attention mask.
    :param float dropout_p: Dropout probability for attention weights.
    :param float | None scale: Optional scaling factor for QK^T.
    :param bool is_causal: Whether to apply causal masking.
    :return torch.Tensor: Attention output tensor.
    """
    if scale is None or _SDPA_SUPPORTS_SCALE:
        return scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale if _SDPA_SUPPORTS_SCALE else None,
        )

    # Torch < 2.1 does not accept scale; rescale queries to emulate it.
    head_dim = query.size(-1)
    default_scale = 1.0 / math.sqrt(head_dim)
    query = query * (scale / default_scale)
    return scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
