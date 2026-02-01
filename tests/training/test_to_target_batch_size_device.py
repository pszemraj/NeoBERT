#!/usr/bin/env python3
"""Tests for device handling in to_target_batch_size."""

import unittest

import torch

from neobert.pretraining.trainer import to_target_batch_size


class TestToTargetBatchSizeDevice(unittest.TestCase):
    """Validate device transfers for buffered batches."""

    def setUp(self) -> None:
        """Skip if CUDA is unavailable."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for device transfer tests.")

    def test_moves_cpu_buffer_to_cuda_for_split(self):
        """Ensure CPU-stored buffers are moved before concatenation."""
        device = torch.device("cuda")
        batch = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, 4), dtype=torch.long, device=device),
            "labels": torch.zeros((1, 4), dtype=torch.long, device=device),
        }
        stored_batch = {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "labels": torch.zeros((2, 4), dtype=torch.long),
        }

        out, stored = to_target_batch_size(batch, stored_batch, target_size=2)

        self.assertEqual(out["input_ids"].shape[0], 2)
        self.assertEqual(out["input_ids"].device.type, "cuda")
        self.assertIsNotNone(stored["input_ids"])
        self.assertEqual(stored["input_ids"].device.type, "cpu")

    def test_moves_cpu_buffer_to_cuda_for_concat(self):
        """Ensure CPU-stored buffers are moved before full concatenation."""
        device = torch.device("cuda")
        batch = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, 4), dtype=torch.long, device=device),
            "labels": torch.zeros((1, 4), dtype=torch.long, device=device),
        }
        stored_batch = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.zeros((1, 4), dtype=torch.long),
        }

        out, stored = to_target_batch_size(batch, stored_batch, target_size=3)

        self.assertEqual(out["input_ids"].shape[0], 2)
        self.assertEqual(out["input_ids"].device.type, "cuda")
        self.assertIsNone(stored["input_ids"])
