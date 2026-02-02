"""RMSNorm layer implementation."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization layer."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the RMSNorm layer.

        :param int dim: Input feature dimension.
        :param float eps: Numerical stability epsilon.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.compile(dynamic=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
