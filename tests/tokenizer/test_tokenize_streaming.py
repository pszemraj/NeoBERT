#!/usr/bin/env python3
"""Tests for streaming dataset tokenization helpers."""

import unittest

from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.tokenizer import tokenize


class TestStreamingTokenize(unittest.TestCase):
    """Validate streaming tokenization behavior."""

    def _make_tokenizer(self) -> PreTrainedTokenizerFast:
        """Build a minimal tokenizer for tests.

        :return PreTrainedTokenizerFast: Tokenizer with a tiny word-level vocab.
        """
        vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
        )

    def test_streaming_multicolumn_tokenize(self):
        """Ensure streaming multi-column datasets tokenize without indexing errors."""
        dataset = Dataset.from_dict(
            {"text_a": ["hello world", "hello"], "text_b": ["world", "hello world"]}
        )
        streaming_dataset = dataset.to_iterable_dataset()
        tokenizer = self._make_tokenizer()

        tokenized = tokenize(
            streaming_dataset,
            tokenizer,
            column_name=("text_a", "text_b"),
            max_length=4,
            truncation=True,
            remove_columns=True,
        )

        first = next(iter(tokenized))
        self.assertIn("input_ids_text_a", first)
        self.assertIn("attention_mask_text_a", first)
        self.assertIn("input_ids_text_b", first)
        self.assertIn("attention_mask_text_b", first)
        self.assertLessEqual(len(first["input_ids_text_a"]), 4)
        self.assertLessEqual(len(first["input_ids_text_b"]), 4)
