#!/usr/bin/env python3
"""Regression tests for pretraining metrics logging."""

import math
import unittest
from typing import Any, Dict, List, Tuple

import torch

from neobert.pretraining.metrics import Metrics


class _AcceleratorStub:
    """Minimal accelerator stub for unit-testing metrics aggregation."""

    def __init__(self) -> None:
        """Initialize a CPU-only accelerator stub."""
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.logged: List[Tuple[Dict[str, Any], int]] = []

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Return local tensors unchanged for single-process tests.

        :param torch.Tensor tensor: Value to reduce.
        :param str reduction: Reduction mode.
        :return torch.Tensor: Unchanged input tensor.
        """
        _ = reduction
        return tensor

    def log(self, values: Dict[str, Any], step: int) -> None:
        """Capture tracker payloads for assertions.

        :param dict[str, Any] values: Metrics payload.
        :param int step: Logging step.
        """
        self.logged.append((values, step))


class TestPretrainingMetrics(unittest.TestCase):
    """Unit tests for pretraining metrics tracker payload behavior."""

    def test_tracker_payload_omits_internal_keys_when_accuracy_disabled(self):
        """Do not emit disabled-accuracy/internal keys to tracker payloads."""
        metrics = Metrics()
        metrics["train/steps"] = 11
        metrics["train/compute_accuracy"] = 0
        metrics["train/local_samples"] = 2
        metrics["train/local_tokens"] = 8
        metrics["train/local_num_pred"] = 4
        metrics["train/local_num_correct"] = 2
        metrics["train/local_sum_loss"] = 6.0

        accelerator = _AcceleratorStub()
        _ = metrics.log(accelerator)

        self.assertEqual(len(accelerator.logged), 1)
        payload, step = accelerator.logged[0]
        self.assertEqual(step, 11)
        self.assertNotIn("train/steps", payload)
        self.assertNotIn("train/compute_accuracy", payload)
        self.assertNotIn("train/local_num_correct", payload)
        self.assertNotIn("train/accuracy", payload)
        self.assertEqual(payload["train/loss"], 1.5)
        self.assertAlmostEqual(payload["train/perplexity"], round(math.exp(1.5), 4))
        self.assertEqual(metrics["train/local_num_correct"], 0)

    def test_tracker_payload_includes_accuracy_when_enabled(self):
        """Emit masked-token accuracy fields when accuracy logging is enabled."""
        metrics = Metrics()
        metrics["train/steps"] = 25
        metrics["train/compute_accuracy"] = 1
        metrics["train/local_samples"] = 4
        metrics["train/local_tokens"] = 16
        metrics["train/local_num_pred"] = 8
        metrics["train/local_num_correct"] = 6
        metrics["train/local_sum_loss"] = 4.0

        accelerator = _AcceleratorStub()
        _ = metrics.log(accelerator)

        payload, step = accelerator.logged[0]
        self.assertEqual(step, 25)
        self.assertNotIn("train/steps", payload)
        self.assertNotIn("train/compute_accuracy", payload)
        self.assertIn("train/local_num_correct", payload)
        self.assertIn("train/accuracy", payload)
        self.assertEqual(payload["train/local_num_correct"], 6)
        self.assertEqual(payload["train/accuracy"], 0.75)

    def test_tracker_payload_strips_loss_path_debug_metrics(self):
        """Never emit masked-loss-path diagnostics to external trackers."""
        metrics = Metrics()
        metrics["train/steps"] = 7
        metrics["train/local_samples"] = 1
        metrics["train/local_tokens"] = 4
        metrics["train/local_num_pred"] = 2
        metrics["train/local_num_correct"] = 1
        metrics["train/local_sum_loss"] = 3.0
        metrics["train/loss_path_steps_liger_flce"] = 10
        metrics["train/loss_path_ratio_liger_flce"] = 1.0

        accelerator = _AcceleratorStub()
        formatted = metrics.log(accelerator)

        payload, _ = accelerator.logged[0]
        self.assertIn("train/loss_path_steps_liger_flce", formatted)
        self.assertIn("train/loss_path_ratio_liger_flce", formatted)
        self.assertNotIn("train/loss_path_steps_liger_flce", payload)
        self.assertNotIn("train/loss_path_ratio_liger_flce", payload)


if __name__ == "__main__":
    unittest.main()
