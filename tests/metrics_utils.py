"""Shared test doubles for distributed metric aggregation."""

from typing import Any

import torch


class AcceleratorStub:
    """CPU-only Accelerator double with deterministic summed reductions."""

    def __init__(self, world_size: int = 1) -> None:
        """Initialize captured logging and reduction state.

        :param int world_size: Simulated process count for summed reductions.
        """
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.world_size = int(world_size)
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.reduce_shapes: list[tuple[int, ...]] = []

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Capture a reduction and simulate a summed multi-rank result.

        :param torch.Tensor tensor: Value to reduce.
        :param str reduction: Reduction mode.
        :return torch.Tensor: Simulated reduced tensor.
        """
        self.reduce_shapes.append(tuple(tensor.shape))
        assert reduction == "sum"
        return tensor * self.world_size

    def log(self, values: dict[str, Any], step: int) -> None:
        """Capture tracker payloads for assertions.

        :param dict[str, Any] values: Metrics payload.
        :param int step: Logging step.
        """
        self.logged.append((values, step))
