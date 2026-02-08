"""RMSNorm layer implementation."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization layer."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm parameters.

        :param int dim: Hidden dimension size.
        :param float eps: Numerical stability epsilon, defaults to ``1e-6``.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization and learned scaling.

        :param torch.Tensor x: Input tensor with hidden dimension on last axis.
        :return torch.Tensor: Normalized tensor with the same shape as ``x``.
        """
        # Accumulate in float32 for numerical stability, then cast back.
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight
