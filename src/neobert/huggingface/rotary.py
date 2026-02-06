"""Rotary Position Embeddings (RoPE) helpers for the HF export path.

This module stays shape-compatible with ``neobert/model/rotary.py`` and accepts
both 2D (seq_len, head_dim/2) and 3D (batch, seq_len, head_dim/2) frequency
tensors for broadcasting.

Keep this numerically aligned with ``neobert.model.rotary``; parity is validated
by ``tests/model/test_model_forward.py::test_rotary_training_matches_hf_export``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex rotary frequencies.

    :param int dim: Head dimension (must be even).
    :param int end: Maximum sequence length.
    :param float theta: RoPE base for frequency computation.
    :param torch.device | None device: Optional device for precomputation.
    :return torch.Tensor: Complex frequency tensor of shape (end, dim // 2).
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got dim={dim}")

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis to broadcast against a complex tensor ``x``.

    :param torch.Tensor freqs_cis: Frequency tensor of shape (S, D/2) or (B, S, D/2).
    :param torch.Tensor x: Complex tensor of shape (B, S, H, D/2).
    :return torch.Tensor: Reshaped freqs_cis compatible with ``x``.
    """
    ndim = x.ndim
    if ndim < 3:
        raise ValueError(f"Expected x to have >=3 dims, got shape={tuple(x.shape)}")

    if freqs_cis.dim() == 2:
        expected = (x.shape[1], x.shape[-1])
        if freqs_cis.shape != expected:
            raise ValueError(
                f"freqs_cis has shape {tuple(freqs_cis.shape)}, expected {expected} "
                "for 2D RoPE frequencies."
            )
        shape = [1] * ndim
        shape[1] = x.shape[1]
        shape[-1] = x.shape[-1]
        return freqs_cis.view(*shape)

    if freqs_cis.dim() == 3:
        expected = (x.shape[0], x.shape[1], x.shape[-1])
        if freqs_cis.shape[1:] != expected[1:] or freqs_cis.shape[0] not in {
            1,
            expected[0],
        }:
            raise ValueError(
                f"freqs_cis has shape {tuple(freqs_cis.shape)}, expected {expected} "
                "for 3D RoPE frequencies."
            )
        shape = [1] * ndim
        shape[0] = freqs_cis.shape[0]
        shape[1] = x.shape[1]
        shape[-1] = x.shape[-1]
        return freqs_cis.view(*shape)

    raise ValueError(
        f"freqs_cis must have 2 or 3 dims, got dim={freqs_cis.dim()} "
        f"with shape={tuple(freqs_cis.shape)}"
    )


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    :param torch.Tensor xq: Query tensor of shape (B, S, H, D).
    :param torch.Tensor xk: Key tensor of shape (B, S, H, D).
    :param torch.Tensor freqs_cis: RoPE frequencies of shape (S, D/2) or (B, S, D/2).
    :return tuple[torch.Tensor, torch.Tensor]: Rotated query/key tensors.
    """
    if xq.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim; got head_dim={xq.shape[-1]}")

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
