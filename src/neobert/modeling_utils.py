"""Shared modeling helpers for training and HF implementations."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

PackedSeqLens = torch.Tensor | list[list[int]]
REMOVED_MODEL_CONFIG_FIELDS = frozenset({"ngpt", "base_scale"})


def removed_model_config_fields(config: Any) -> list[str]:
    """Return removed NeoBERT model fields present in a config mapping.

    :param Any config: Mapping-like object whose keys are model config fields.
    :return list[str]: Sorted removed fields found in ``config``.
    """
    return sorted(REMOVED_MODEL_CONFIG_FIELDS.intersection(config))


def packed_seqlens_to_tensor(
    packed_seqlens: Any,
    *,
    device: Optional[torch.device] = None,
    validate: bool = True,
) -> Optional[torch.Tensor]:
    """Convert packed segment lengths to a rank-2 int32 tensor.

    Tensor inputs retain their device unless ``device`` is provided. List inputs
    are padded with zeros; zero-length segments are omitted.

    :param Any packed_seqlens: Rank-1/rank-2 tensor, nested list, or ``None``.
    :param torch.device | None device: Optional destination device.
    :param bool validate: Whether to reject negative lengths.
    :raises TypeError: If the input is neither a tensor nor a nested list.
    :raises ValueError: If lengths are negative or a tensor is not rank 1 or 2.
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
        tensor = tensor.to(dtype=torch.int32, device=device)
        if validate and (tensor < 0).any():
            raise ValueError("packed_seqlens must contain non-negative lengths.")
        return tensor
    if not isinstance(packed_seqlens, list):
        raise TypeError(
            f"Unsupported packed_seqlens type: {type(packed_seqlens).__name__}"
        )

    rows = []
    for row in packed_seqlens:
        values = [] if row is None else [int(length) for length in row]
        if validate and any(length < 0 for length in values):
            raise ValueError("packed_seqlens must contain non-negative lengths.")
        rows.append([length for length in values if length > 0])
    width = max((len(row) for row in rows), default=0)
    tensor = torch.zeros((len(rows), width), dtype=torch.int32, device=device)
    for index, row in enumerate(rows):
        if row:
            tensor[index, : len(row)] = torch.tensor(
                row, dtype=torch.int32, device=device
            )
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


def scaled_dot_product_attention_compat(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Dispatch SDPA with the explicit softmax scale.

    :param torch.Tensor query: Query tensor of shape (B, H, M, K).
    :param torch.Tensor key: Key tensor of shape (B, H, N, K).
    :param torch.Tensor value: Value tensor of shape (B, H, N, K).
    :param torch.Tensor | None attn_mask: Optional attention mask.
    :param float dropout_p: Dropout probability for attention weights.
    :param float | None scale: Optional scaling factor for QK^T.
    :param bool is_causal: Whether to apply causal masking.
    :return torch.Tensor: Attention output tensor.
    """
    return scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
