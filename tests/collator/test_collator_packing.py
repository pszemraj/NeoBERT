#!/usr/bin/env python3
"""Regression tests for sequence packing collators."""

import unittest
import warnings

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.collator import CustomCollatorForMLM, DataCollatorWithPacking, get_collator


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
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if all("special_tokens_mask" in f for f in features):
            batch_masks = [
                f["special_tokens_mask"]
                + [1] * (max_len - len(f["special_tokens_mask"]))
                for f in features
            ]
            batch["special_tokens_mask"] = torch.tensor(batch_masks, dtype=torch.long)
        return batch


class TestCollatorPacking(unittest.TestCase):
    """Tests for edge cases in sequence packing."""

    @staticmethod
    def _make_tokenizer(padding_side: str = "right") -> PreTrainedTokenizerFast:
        """Build a minimal tokenizer for collator tests."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "hello": 3, "world": 4}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
            mask_token="[MASK]",
        )
        fast.padding_side = padding_side
        return fast

    @staticmethod
    def _packed_to_list(packed):
        """Convert packed_seqlens tensor to list for assertions."""
        if torch.is_tensor(packed):
            if packed.numel() == 0:
                return [[] for _ in range(packed.shape[0])]
            return [[int(x) for x in row[row > 0].tolist()] for row in packed]
        return packed

    def test_flushes_before_sep_when_sequence_does_not_fit(self):
        """Ensure sequences are not split across packed buffers."""
        collator = DataCollatorWithPacking(
            start_token_id=10,
            end_token_id=11,
            max_length=6,
            default_data_collator=DummyPadCollator(),
        )

        batch = collator(
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4, 5]},
            ]
        )

        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), [10, 1, 2, 3, 11, 0])
        self.assertEqual(batch["input_ids"][1].tolist(), [10, 4, 5, 11, 0, 0])
        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["attention_mask"].shape, (2, 6))
        self.assertEqual(self._packed_to_list(batch["packed_seqlens"]), [[5], [4]])

    def test_boundaries_used_when_sequence_fits(self):
        """Ensure segment boundaries are inserted when packing."""
        collator = DataCollatorWithPacking(
            start_token_id=10,
            end_token_id=11,
            max_length=8,
            default_data_collator=DummyPadCollator(),
        )

        batch = collator(
            [
                {"input_ids": [1, 2]},
                {"input_ids": [3]},
            ]
        )

        self.assertEqual(batch["input_ids"].shape, (1, 8))
        self.assertEqual(batch["input_ids"][0].tolist(), [10, 1, 2, 11, 10, 3, 11, 0])
        self.assertIn("attention_mask", batch)
        self.assertEqual(batch["attention_mask"].shape, (1, 8))
        self.assertEqual(self._packed_to_list(batch["packed_seqlens"]), [[4, 3]])
        self.assertIn("special_tokens_mask", batch)
        self.assertEqual(
            batch["special_tokens_mask"][0].tolist(),
            [1, 0, 0, 1, 1, 0, 1, 1],
        )

    def test_packed_lengths_match_nonpad_tokens(self):
        """Ensure packed lengths sum to non-pad token counts."""
        collator = DataCollatorWithPacking(
            start_token_id=10,
            end_token_id=11,
            max_length=6,
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
        packed_counts = [
            sum(lengths) for lengths in self._packed_to_list(batch["packed_seqlens"])
        ]
        self.assertEqual(token_counts, packed_counts)
        self.assertEqual(batch["input_ids"].shape[1], 6)

    def test_return_packed_seqlens_omits_key_when_left_padded(self):
        """Ensure packed_seqlens is omitted for non-right-padded masks."""
        tokenizer = self._make_tokenizer(padding_side="left")
        collator = get_collator(tokenizer, return_packed_seqlens=True)
        features = [{"input_ids": [2, 3]}, {"input_ids": [2]}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            batch = collator(features)

        self.assertTrue(
            any("Skipping packed_seqlens" in str(w.message) for w in caught)
        )
        self.assertNotIn("packed_seqlens", batch)

    def test_mask_all_numpy_path_masks_without_801010_split(self):
        """Ensure numpy masking path follows 100% mask-all semantics."""
        tokenizer = self._make_tokenizer()
        collator = CustomCollatorForMLM(
            tokenizer=tokenizer,
            mlm_probability=1.0,
        )
        inputs = np.array([[3, 4, 0]], dtype=np.int64)
        special_tokens_mask = np.array([[0, 0, 1]], dtype=np.int64)

        masked_inputs, labels = collator.numpy_mask_tokens(
            inputs.copy(),
            special_tokens_mask=special_tokens_mask,
        )

        self.assertEqual(masked_inputs[0, 0], tokenizer.mask_token_id)
        self.assertEqual(masked_inputs[0, 1], tokenizer.mask_token_id)
        self.assertEqual(masked_inputs[0, 2], 0)
        self.assertEqual(labels[0, 0], 3)
        self.assertEqual(labels[0, 1], 4)
        self.assertEqual(labels[0, 2], -100)

    def test_packing_raises_when_pad_token_id_is_unresolved(self):
        """Ensure packing fails loudly when no pad token can be resolved."""

        class NoPadCollator:
            """Collator stub without tokenizer or pad_token_id metadata."""

            def __call__(self, features, return_tensors=None):
                return {"input_ids": features}

        collator = DataCollatorWithPacking(
            start_token_id=10,
            end_token_id=11,
            max_length=8,
            default_data_collator=NoPadCollator(),
        )

        with self.assertRaisesRegex(ValueError, "Could not resolve pad_token_id"):
            collator([{"input_ids": [1, 2, 3]}])
