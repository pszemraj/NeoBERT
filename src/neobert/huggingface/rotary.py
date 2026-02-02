"""Rotary Position Embeddings (RoPE) implementation for NeoBERT.

This module implements Rotary Position Embeddings (RoPE) as described in RoFormer:
Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864).

RoPE encodes absolute positional information with rotation matrix and naturally
incorporates explicit relative position dependency in self-attention formulation.

Based on: https://github.com/facebookresearch/llama/blob/main/llama/model.py
"""

from typing import Optional, Tuple

import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute the frequency tensor for rotary position embeddings.

    This function calculates a frequency tensor with complex exponentials that will
    be used to apply rotary embeddings. The frequencies are computed using the RoPE
    formula with sinusoidal position encoding at different frequency scales.

    Args:
        dim: Dimension of each attention head (must be even).
        end: Maximum sequence length to precompute frequencies for.
        theta: Base value for computing rotation frequencies. Higher values lead to
            lower frequencies, affecting how position information decays with distance.
            Default is 10000.0 as used in the original RoPE paper.
        device: Optional device for precomputation.

    Returns:
        Complex tensor of shape (end, dim // 2) containing precomputed rotation
        frequencies as complex exponentials.

    Examples:
        >>> freqs = precompute_freqs_cis(64, 2048, theta=10000.0)
        >>> freqs.shape
        torch.Size([2048, 32])
        >>> freqs.dtype
        torch.complex64
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with query/key tensors.

    Adjusts the shape of the frequency tensor to enable element-wise multiplication
    with query and key tensors during rotary embedding application.

    Args:
        freqs_cis: Precomputed frequency tensor of shape (seq_len, dim // 2).
        x: Query or key tensor of shape (batch, seq_len, heads, dim) after
            reshaping to complex representation.

    Returns:
        Frequency tensor with shape adjusted for broadcasting.

    Raises:
        AssertionError: If the frequency tensor dimensions don't match the input tensor.
    """
    assert freqs_cis.shape[1:] == (x.shape[1], x.shape[-1]), (
        f"Frequency tensor shape {freqs_cis.shape[1:]} doesn't match "
        f"input tensor shape ({x.shape[1]}, {x.shape[-1]})"
    )
    return freqs_cis.contiguous().unsqueeze(2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Rotary position embeddings encode positional information directly into the
    query and key representations by rotating them in complex space. This allows
    the model to naturally incorporate relative position information in the
    attention mechanism.

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim).
        xk: Key tensor of shape (batch, seq_len, n_heads, head_dim).
        freqs_cis: Precomputed frequency tensor containing complex exponentials.

    Returns:
        Tuple of (rotated_queries, rotated_keys), both with the same shape
        and dtype as the input tensors.

    Notes:
        - The head dimension must be even for complex number reshaping.
        - Input tensors are temporarily converted to float32 for complex operations,
          then cast back to their original dtype.
        - The rotation is applied by element-wise complex multiplication.

    Examples:
        >>> batch, seq_len, n_heads, head_dim = 2, 128, 8, 64
        >>> xq = torch.randn(batch, seq_len, n_heads, head_dim)
        >>> xk = torch.randn(batch, seq_len, n_heads, head_dim)
        >>> freqs = precompute_freqs_cis(head_dim, seq_len)
        >>> xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs)
        >>> assert xq_rot.shape == xq.shape
    """
    # Reshape to complex representation: (..., head_dim) -> (..., head_dim/2, 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Prepare frequency tensor for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Apply rotation via complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # Cast back to original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)
