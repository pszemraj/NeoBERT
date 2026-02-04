#!/usr/bin/env python3
"""Integration test for Accelerate dispatch batches with packed_seqlens."""

from __future__ import annotations

import unittest

import torch
from accelerate.data_loader import prepare_data_loader
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast

from neobert.collator import get_collator


def _make_tokenizer() -> PreTrainedTokenizerFast:
    """Create a tiny tokenizer for tests.

    :return PreTrainedTokenizerFast: Tokenizer instance.
    """
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
        "[SEP]": 3,
        "hello": 4,
        "world": 5,
        "test": 6,
        "sentence": 7,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        sep_token="[SEP]",
    )


class _StreamingDataset(IterableDataset):
    """Simple iterable dataset for streaming-style tests."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        """Initialize the dataset.

        :param PreTrainedTokenizerFast tokenizer: Tokenizer for inputs.
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._texts = [
            "hello world",
            "test sentence",
            "hello test",
            "world sentence",
            "test world",
            "hello sentence",
        ]

    def __iter__(self):
        """Yield tokenized samples."""
        for text in self._texts:
            tokens = self._tokenizer(text, add_special_tokens=False)
            yield {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }


class TestAccelerateDispatch(unittest.TestCase):
    """Ensure packed_seqlens survives Accelerate dispatch batching."""

    def test_dispatch_batches_accepts_packed_seqlens_tensor(self):
        """Ensure dispatch batching can concatenate packed_seqlens."""
        tokenizer = _make_tokenizer()
        collator = get_collator(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            return_packed_seqlens=True,
        )
        dataset = _StreamingDataset(tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        prepared = prepare_data_loader(
            dataloader,
            device=torch.device("cpu"),
            put_on_device=True,
            dispatch_batches=True,
        )

        batches = 0
        for batch in prepared:
            self.assertIn("packed_seqlens", batch)
            self.assertTrue(torch.is_tensor(batch["packed_seqlens"]))
            batches += 1
            if batches >= 2:
                break

        self.assertGreater(batches, 0)
