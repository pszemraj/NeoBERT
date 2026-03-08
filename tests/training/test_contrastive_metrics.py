#!/usr/bin/env python3
"""Regression tests for contrastive metrics logging."""

import unittest
from typing import Any, Dict, List, Tuple

import torch

from neobert.contrastive.metrics import Metrics


class _AcceleratorStub:
    """Minimal accelerator stub for unit-testing contrastive metrics."""

    def __init__(self, world_size: int = 2) -> None:
        """Initialize a CPU-only accelerator stub.

        :param int world_size: Simulated process count for reductions.
        """
        self.device = torch.device("cpu")
        self.is_main_process = True
        self._world_size = world_size
        self.logged: List[Tuple[Dict[str, Any], int]] = []
        self.reduce_shapes: List[tuple[int, ...]] = []

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Capture reduction shapes and simulate a summed multi-rank result.

        :param torch.Tensor tensor: Value to reduce.
        :param str reduction: Reduction mode.
        :return torch.Tensor: Simulated reduced tensor.
        """
        self.reduce_shapes.append(tuple(tensor.shape))
        assert reduction == "sum"
        return tensor * self._world_size

    def log(self, values: Dict[str, Any], step: int) -> None:
        """Capture tracker payloads for assertions.

        :param dict[str, Any] values: Metrics payload.
        :param int step: Logging step.
        """
        self.logged.append((values, step))


class TestContrastiveMetrics(unittest.TestCase):
    """Unit tests for contrastive metric aggregation behavior."""

    def test_log_reduces_only_local_counters(self) -> None:
        """Already-global diagnostics must not participate in scalar reduction."""
        metrics = Metrics()
        metrics["train/epochs"] = 1
        metrics["train/steps"] = 10
        metrics["train/batches"] = 40
        metrics["train/local_samples"] = 3
        metrics["train/local_sum_loss"] = 6.0
        metrics["train/grad_norm"] = 7.5
        metrics["train/weight_norm"] = 8.5
        metrics["train/learning_rate"] = 1e-3

        accelerator = _AcceleratorStub(world_size=2)
        metrics.log(accelerator)

        self.assertEqual(accelerator.reduce_shapes, [(1,), (1,)])
        self.assertEqual(len(accelerator.logged), 1)
        payload, step = accelerator.logged[0]
        self.assertEqual(step, 10)
        self.assertEqual(payload["train/grad_norm"], 7.5)
        self.assertEqual(payload["train/weight_norm"], 8.5)
        self.assertEqual(payload["train/loss"], 2.0)
        self.assertEqual(payload["train/samples"], 6)
        self.assertEqual(metrics["train/local_samples"], 0)
        self.assertEqual(metrics["train/local_sum_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
