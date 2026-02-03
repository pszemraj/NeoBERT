"""Shared modeling helpers for training and HF implementations."""

from __future__ import annotations

import inspect
import math
from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention


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
