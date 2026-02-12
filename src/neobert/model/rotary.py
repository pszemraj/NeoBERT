"""Rotary positional embedding helpers.

Uses real-valued cos/sin (no complex tensors) for torch.compile compatibility.

Keep this numerically aligned with ``neobert.huggingface.rotary``; parity is
validated by ``tests/test_model_forward.py::test_rotary_training_matches_hf_export``.
"""

from typing import Optional, Tuple

import torch
from einops import rearrange


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute rotary cos/sin frequencies.

    Returns a ``(end, dim//2, 2)`` tensor where ``[..., 0]`` is cos and
    ``[..., 1]`` is sin.  The name ``freqs_cis`` is kept for backward
    compatibility with callers.

    :param int dim: Dimension of the frequency tensor.
    :param int end: End index for precomputing frequencies.
    :param float theta: Scaling factor for frequency computation.
    :param torch.device | None device: Optional device for precomputation.
    :return torch.Tensor: Stacked cos/sin frequency tensor ``(end, dim//2, 2)``.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=freqs.device)
    angles = torch.outer(t, freqs).float()
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Uses real-valued cos/sin rotation (no complex tensors) for torch.compile
    compatibility.

    :param torch.Tensor xq: Query tensor ``(B, S, H, D)``.
    :param torch.Tensor xk: Key tensor ``(B, S, H, D)``.
    :param torch.Tensor freqs_cis: Stacked cos/sin ``(S, D//2, 2)``.
    :return tuple[torch.Tensor, torch.Tensor]: Rotary-embedded (xq, xk).
    """
    # Split head_dim into pairs: (B, S, H, D) -> (B, S, H, D//2, 2).
    # We upcast to fp32 for numerically stable trig rotation in eager/compile
    # paths; downstream fused attention kernels can still optimize this path.
    xq_pairs = rearrange(xq.float(), "... (d two) -> ... d two", two=2)
    xk_pairs = rearrange(xk.float(), "... (d two) -> ... d two", two=2)

    # Broadcast freqs over batch and heads.
    # freqs_cis is (S, D//2, 2) or (B, S, D//2, 2) for batched positions.
    if freqs_cis.dim() == 3:
        cos_sin = rearrange(freqs_cis, "s d two -> 1 s 1 d two")
    else:
        cos_sin = rearrange(freqs_cis, "b s d two -> b s 1 d two")
    cos, sin = cos_sin[..., 0], cos_sin[..., 1]

    # Apply rotation: (x0 + i*x1) * (cos + i*sin)
    xq_out = torch.stack(
        [
            xq_pairs[..., 0] * cos - xq_pairs[..., 1] * sin,
            xq_pairs[..., 0] * sin + xq_pairs[..., 1] * cos,
        ],
        dim=-1,
    )
    xk_out = torch.stack(
        [
            xk_pairs[..., 0] * cos - xk_pairs[..., 1] * sin,
            xk_pairs[..., 0] * sin + xk_pairs[..., 1] * cos,
        ],
        dim=-1,
    )

    # Merge pairs back: (B, S, H, D//2, 2) -> (B, S, H, D)
    return (
        rearrange(xq_out, "... d two -> ... (d two)").type_as(xq),
        rearrange(xk_out, "... d two -> ... (d two)").type_as(xk),
    )
