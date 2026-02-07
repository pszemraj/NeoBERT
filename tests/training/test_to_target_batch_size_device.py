#!/usr/bin/env python3
"""Tests for device handling in to_target_batch_size."""

import unittest

import torch

from neobert.pretraining.trainer import (
    _append_to_stored_batch,
    _has_stored_batch,
    to_target_batch_size,
)


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

    def test_gpu_batch_appends_to_cpu_buffer(self):
        """Ensure CPU buffers can append GPU leftovers without device errors."""
        device = torch.device("cuda")
        batch = {
            "input_ids": torch.zeros((4, 4), dtype=torch.long, device=device),
            "attention_mask": torch.ones((4, 4), dtype=torch.long, device=device),
            "labels": torch.zeros((4, 4), dtype=torch.long, device=device),
        }
        stored_batch = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.zeros((1, 4), dtype=torch.long),
        }

        out, stored = to_target_batch_size(batch, stored_batch, target_size=2)

        self.assertEqual(out["input_ids"].shape[0], 2)
        self.assertEqual(out["input_ids"].device.type, "cuda")
        self.assertIsNotNone(stored["input_ids"])
        self.assertEqual(stored["input_ids"].device.type, "cpu")
        self.assertEqual(stored["input_ids"].shape[0], 3)


class TestToTargetBatchSizeKeys(unittest.TestCase):
    """Validate buffering with extra batch keys."""

    def test_handles_extra_keys_without_errors(self):
        """Ensure unexpected keys do not raise KeyError when buffering."""
        batch = {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "labels": torch.zeros((2, 4), dtype=torch.long),
            "token_type_ids": torch.zeros((2, 4), dtype=torch.long),
        }
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }

        out, stored = to_target_batch_size(batch, stored_batch, target_size=1)

        self.assertEqual(out["input_ids"].shape[0], 1)
        self.assertIn("token_type_ids", stored)


class TestStoredBatchListConcat(unittest.TestCase):
    """Validate list concatenation (non-tensor path) in to_target_batch_size."""

    def test_stored_batch_list_concat_on_split(self):
        """Ensure non-tensor values are list-concatenated when batch is split."""
        batch = {
            "input_ids": torch.zeros((4, 3), dtype=torch.long),
            "tags": ["a", "b", "c", "d"],
        }
        stored_batch: dict = {"input_ids": None, "tags": None}

        out, stored = to_target_batch_size(batch, stored_batch, target_size=2)

        self.assertEqual(out["input_ids"].shape[0], 2)
        self.assertEqual(out["tags"], ["a", "b"])
        self.assertEqual(stored["tags"], ["c", "d"])

    def test_stored_batch_list_concat_on_undersized(self):
        """Ensure non-tensor values are list-concatenated when batch is undersized."""
        batch = {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "tags": ["x"],
        }
        stored_batch: dict = {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "tags": ["y"],
        }

        out, stored = to_target_batch_size(batch, stored_batch, target_size=4)

        self.assertEqual(out["input_ids"].shape[0], 2)
        # Merge path puts stored_batch first: stored + batch
        self.assertEqual(out["tags"], ["y", "x"])
        self.assertIsNone(stored["tags"])


class TestPackedFragmentBuffering(unittest.TestCase):
    """Regression tests for packed fragment buffering behavior."""

    def test_undersized_packed_fragments_accumulate_to_target(self):
        """Ensure undersized packed fragments are buffered and merged correctly."""
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seqlens": None,
        }
        frag_a = {
            "input_ids": torch.arange(6, dtype=torch.long).view(2, 3),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "labels": torch.zeros((2, 3), dtype=torch.long),
            "packed_seqlens": torch.tensor([[3, 0], [2, 1]], dtype=torch.int32),
        }
        frag_b = {
            "input_ids": torch.arange(9, dtype=torch.long).view(3, 3),
            "attention_mask": torch.ones((3, 3), dtype=torch.long),
            "labels": torch.zeros((3, 3), dtype=torch.long),
            "packed_seqlens": torch.tensor([[1, 1], [2, 0], [3, 0]], dtype=torch.int32),
        }

        _append_to_stored_batch(stored_batch, frag_a)
        self.assertTrue(_has_stored_batch(stored_batch))
        out, stored = to_target_batch_size(frag_b, stored_batch, target_size=4)

        self.assertEqual(out["input_ids"].shape[0], 4)
        self.assertEqual(out["packed_seqlens"].shape[0], 4)
        self.assertIsNotNone(stored["input_ids"])
        self.assertEqual(stored["input_ids"].shape[0], 1)
