#!/usr/bin/env python3
"""Regression tests for contrastive metrics logging."""

import unittest

from neobert.contrastive.metrics import Metrics
from tests.metrics_utils import AcceleratorStub


class TestContrastiveMetrics(unittest.TestCase):
    """Unit tests for contrastive metric aggregation behavior."""

    def test_checkpoint_state_is_versioned_and_round_trips(self) -> None:
        """Contrastive resume cursors use the shared topology-bound schema."""
        metrics = Metrics()
        metrics["train/steps"] = 7
        metrics["train/batches_in_epoch"] = 3

        state = metrics.state_dict()
        restored = Metrics()
        restored.load_state_dict(state)

        self.assertEqual(state["metrics_state_version"], 1)
        self.assertEqual(state["world_size"], 1)
        self.assertEqual(restored["train/steps"], 7)
        self.assertEqual(restored["train/batches_in_epoch"], 3)

    def test_checkpoint_state_rejects_unversioned_cursor(self) -> None:
        """Bare dictionaries cannot prove contrastive rank-cursor alignment."""
        with self.assertRaisesRegex(ValueError, "cannot prove"):
            Metrics().load_state_dict({"train/steps": 7})

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

        accelerator = AcceleratorStub(world_size=2)
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
