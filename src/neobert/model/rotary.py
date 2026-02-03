"""Rotary positional embedding helpers."""

from typing import Optional, Tuple

import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex rotary frequencies.

    :param int dim: Dimension of the frequency tensor.
    :param int end: End index for precomputing frequencies.
    :param float theta: Scaling factor for frequency computation.
    :param torch.device | None device: Optional device for precomputation.
    :return torch.Tensor: Complex frequency tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcast with a target tensor.

    :param torch.Tensor freqs_cis: Frequency tensor to reshape.
    :param torch.Tensor x: Target tensor for broadcasting compatibility.
    :return torch.Tensor: Reshaped frequency tensor.
    :raises ValueError: If tensor shapes are incompatible.
    """

    ndim = x.ndim
    if ndim < 2:
        raise ValueError(f"Expected x with at least 2 dims, got shape {x.shape}")

    if freqs_cis.dim() == 2:
        if freqs_cis.shape != (x.shape[1], x.shape[-1]):
            raise ValueError(
                f"freqs_cis has shape {freqs_cis.shape}, expected "
                f"({x.shape[1]}, {x.shape[-1]})"
            )
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    if freqs_cis.dim() == 3:
        if (
            freqs_cis.shape[:2] != (x.shape[0], x.shape[1])
            or freqs_cis.shape[-1] != x.shape[-1]
        ):
            raise ValueError(
                f"freqs_cis has shape {freqs_cis.shape}, expected "
                f"({x.shape[0]}, {x.shape[1]}, {x.shape[-1]})"
            )
        shape = [d if i in (0, 1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    raise ValueError(f"freqs_cis must have 2 or 3 dims, got {freqs_cis.dim()}")


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
