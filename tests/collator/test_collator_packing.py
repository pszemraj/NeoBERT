#!/usr/bin/env python3
"""Regression tests for sequence packing collators."""

import unittest

import torch

from neobert.collator import DataCollatorWithPacking


class DummyPadCollator:
    """Pad-only collator for packing tests."""

    def __init__(self, pad_token_id: int = 0) -> None:
        """Initialize the dummy pad-only collator.

        :param int pad_token_id: Token used for padding shorter sequences.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, features, return_tensors=None):
        """Pad input_ids to the longest sequence in the batch.

        :param list[dict[str, list[int]]] features: Packed input features.
        :param Any return_tensors: Unused compatibility argument.
        :return dict[str, torch.Tensor]: Batch with padded input IDs.
        """
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids = [
            f["input_ids"] + [self.pad_token_id] * (max_len - len(f["input_ids"]))
            for f in features
        ]
        return {"input_ids": torch.tensor(batch_ids, dtype=torch.long)}


class TestCollatorPacking(unittest.TestCase):
    """Tests for edge cases in sequence packing."""

    def test_flushes_when_sep_fills_buffer(self):
        """Ensure separator flushes the buffer at max_length."""
        collator = DataCollatorWithPacking(
            sep_token_id=99,
            max_length=4,
            default_data_collator=DummyPadCollator(),
        )

        batch = collator(
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4]},
            ]
        )

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 3, 99])
        self.assertEqual(batch["input_ids"][1].tolist(), [4, 0, 0, 0])
        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["attention_mask"].shape, (2, 4, 4))
