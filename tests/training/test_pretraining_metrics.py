#!/usr/bin/env python3
"""Regression tests for pretraining metrics logging."""

import math
import unittest

from neobert.pretraining.metrics import Metrics
from tests.metrics_utils import AcceleratorStub


class TestPretrainingMetrics(unittest.TestCase):
    """Unit tests for pretraining metrics tracker payload behavior."""

    def test_checkpoint_state_is_versioned_and_round_trips(self):
        """Loop counters require a versioned, topology-bound checkpoint payload."""
        metrics = Metrics()
        metrics["train/steps"] = 7
        metrics["train/dataloader_batches_in_epoch"] = 3

        state = metrics.state_dict()
        restored = Metrics()
        restored.load_state_dict(state)

        self.assertEqual(state["metrics_state_version"], 1)
        self.assertEqual(state["world_size"], 1)
        self.assertEqual(restored["train/steps"], 7)
        self.assertEqual(restored["train/dataloader_batches_in_epoch"], 3)

    def test_checkpoint_state_rejects_unversioned_cursor(self):
        """Legacy loop state cannot prove distributed packed-cursor alignment."""
        with self.assertRaisesRegex(ValueError, "cannot prove"):
            Metrics().load_state_dict({"train/steps": 7})

    def test_tracker_payload_omits_internal_keys_when_accuracy_disabled(self):
        """Do not emit disabled-accuracy/internal keys to tracker payloads."""
        metrics = Metrics()
        metrics["train/steps"] = 11
        metrics["train/batches"] = 44
        metrics["train/compute_accuracy"] = 0
        metrics["train/local_samples"] = 2
        metrics["train/local_tokens"] = 8
        metrics["train/local_num_pred"] = 4
        metrics["train/local_num_correct"] = 2
        metrics["train/local_sum_loss"] = 6.0

        accelerator = AcceleratorStub()
        _ = metrics.log(accelerator)

        self.assertEqual(len(accelerator.logged), 1)
        payload, step = accelerator.logged[0]
        self.assertEqual(step, 11)
        self.assertNotIn("train/steps", payload)
        self.assertNotIn("train/batches", payload)
        self.assertNotIn("train/samples", payload)
        self.assertNotIn("train/masked_tokens", payload)
        self.assertNotIn("train/compute_accuracy", payload)
        self.assertNotIn("train/local_num_correct", payload)
        self.assertNotIn("train/local_samples", payload)
        self.assertNotIn("train/local_tokens", payload)
        self.assertNotIn("train/local_num_pred", payload)
        self.assertNotIn("train/local_sum_loss", payload)
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

        accelerator = AcceleratorStub()
        _ = metrics.log(accelerator)

        payload, step = accelerator.logged[0]
        self.assertEqual(step, 25)
        self.assertNotIn("train/steps", payload)
        self.assertNotIn("train/compute_accuracy", payload)
        self.assertNotIn("train/local_num_correct", payload)
        self.assertIn("train/accuracy", payload)
        self.assertEqual(payload["train/accuracy"], 0.75)


def test_metrics_accelerate_checkpoint_round_trip(tmp_path) -> None:
    """Accelerate persists and restores the versioned pretraining loop state."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    metrics = Metrics()
    metrics["train/steps"] = 9
    metrics["train/dataloader_batches_in_epoch"] = 4
    accelerator.register_for_checkpointing(metrics)

    accelerator.save_state(str(tmp_path))
    metrics["train/steps"] = 0
    metrics["train/dataloader_batches_in_epoch"] = 0
    accelerator.load_state(str(tmp_path))

    assert metrics["train/steps"] == 9
    assert metrics["train/dataloader_batches_in_epoch"] == 4


if __name__ == "__main__":
    unittest.main()
