#!/usr/bin/env python3
"""Tests for streaming dataset tokenization helpers."""

import unittest
from unittest.mock import patch

from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.tokenizer import tokenize
from neobert.tokenizer.tokenizer import get_tokenizer


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
        try:
            streaming_dataset = dataset.to_iterable_dataset()
        except (RuntimeError, PermissionError) as exc:
            msg = str(exc).lower()
            if (
                "shared memory" in msg
                or "share_memory" in msg
                or "permission denied" in msg
            ):
                self.skipTest(f"Shared memory unavailable for streaming dataset: {exc}")
            raise
        tokenizer = self._make_tokenizer()

        tokenized = tokenize(
            streaming_dataset,
            tokenizer,
            column_name=("text_a", "text_b"),
            max_length=4,
            truncation=True,
            remove_columns=True,
            return_special_tokens_mask=True,
        )

        first = next(iter(tokenized))
        self.assertIn("input_ids_text_a", first)
        self.assertIn("attention_mask_text_a", first)
        self.assertIn("special_tokens_mask_text_a", first)
        self.assertIn("input_ids_text_b", first)
        self.assertIn("attention_mask_text_b", first)
        self.assertIn("special_tokens_mask_text_b", first)
        self.assertLessEqual(len(first["input_ids_text_a"]), 4)
        self.assertLessEqual(len(first["input_ids_text_b"]), 4)

    def test_get_tokenizer_pair_template_uses_single_bos(self):
        """Ensure fallback pair template does not inject BOS before sentence B."""
        base = self._make_tokenizer()
        # Trigger the fallback special-token branch.
        base.mask_token = None

        with patch(
            "neobert.tokenizer.tokenizer.AutoTokenizer.from_pretrained",
            return_value=base,
        ):
            tokenizer = get_tokenizer(
                "dummy-tokenizer",
                max_length=32,
                allow_special_token_rewrite=True,
            )

        pair_ids = tokenizer("hello", "world", add_special_tokens=True)["input_ids"]
        bos = tokenizer.bos_token_id
        sep = tokenizer.sep_token_id
        eos = tokenizer.eos_token_id

        self.assertEqual(pair_ids[0], bos)
        self.assertEqual(pair_ids[-1], eos)
        self.assertEqual(sum(token_id == bos for token_id in pair_ids), 1)
        first_sep_idx = pair_ids.index(sep)
        self.assertNotEqual(pair_ids[first_sep_idx + 1], bos)

    def test_get_tokenizer_rejects_implicit_special_token_rewrite(self):
        """Ensure tokenizer fallback rewrite requires explicit opt-in."""
        base = self._make_tokenizer()
        base.mask_token = None

        with patch(
            "neobert.tokenizer.tokenizer.AutoTokenizer.from_pretrained",
            return_value=base,
        ):
            with self.assertRaises(ValueError):
                get_tokenizer("dummy-tokenizer", max_length=32)

    def test_get_tokenizer_allows_missing_mask_when_mlm_enforcement_disabled(self):
        """Ensure non-MLM flows can keep tokenizer special tokens unchanged."""
        base = self._make_tokenizer()
        base.mask_token = None

        with patch(
            "neobert.tokenizer.tokenizer.AutoTokenizer.from_pretrained",
            return_value=base,
        ):
            tokenizer = get_tokenizer(
                "dummy-tokenizer",
                max_length=64,
                enforce_mlm_special_tokens=False,
            )

        self.assertIsNone(tokenizer.mask_token)
        self.assertEqual(tokenizer.model_max_length, 64)
