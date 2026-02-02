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
        input_ids = torch.tensor(batch_ids, dtype=torch.long)
        attention_mask = input_ids.ne(self.pad_token_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestCollatorPacking(unittest.TestCase):
    """Tests for edge cases in sequence packing."""

    def test_flushes_before_sep_when_sequence_does_not_fit(self):
        """Ensure sequences are not split across packed buffers."""
        collator = DataCollatorWithPacking(
            sep_token_id=99,
            max_length=4,
            default_data_collator=DummyPadCollator(),
        )

        batch = collator(
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4, 5]},
            ]
        )

        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 3])
        self.assertEqual(batch["input_ids"][1].tolist(), [4, 5, 0])
        self.assertNotIn(99, batch["input_ids"][0].tolist())
        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["attention_mask"].shape, (2, 3))
        self.assertEqual(batch["packed_seqlens"], [[3], [2]])

    def test_separator_used_when_sequence_fits(self):
        """Ensure separator is inserted when the next sequence fits."""
        collator = DataCollatorWithPacking(
            sep_token_id=99,
            max_length=4,
            default_data_collator=DummyPadCollator(),
        )

        batch = collator(
            [
                {"input_ids": [1, 2]},
                {"input_ids": [3]},
            ]
        )

        self.assertEqual(batch["input_ids"].shape, (1, 4))
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 99, 3])
        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["attention_mask"].shape, (1, 4))
        self.assertEqual(batch["packed_seqlens"], [[2, 2]])

    def test_packed_lengths_match_nonpad_tokens(self):
        """Ensure packed lengths sum to non-pad token counts."""
        collator = DataCollatorWithPacking(
            sep_token_id=99,
            max_length=4,
            default_data_collator=DummyPadCollator(),
        )

        batch = collator(
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4, 5]},
            ]
        )

        attention_mask = batch["attention_mask"]
        token_counts = attention_mask.sum(dim=1).tolist()
        packed_counts = [sum(lengths) for lengths in batch["packed_seqlens"]]
        self.assertEqual(token_counts, packed_counts)
