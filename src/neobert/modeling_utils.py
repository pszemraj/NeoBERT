"""Shared modeling helpers for training and HF implementations."""

from __future__ import annotations


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
