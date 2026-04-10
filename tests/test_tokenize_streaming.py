#!/usr/bin/env python3
"""Tests for streaming dataset tokenization helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
from datasets import Dataset

from neobert.tokenizer import tokenize
from neobert.tokenizer.tokenizer import get_tokenizer
from tests.tokenizer_utils import build_wordlevel_tokenizer


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    """Construct an HTTPError with a response-like object.

    :param int status_code: HTTP status code to attach.
    :return requests.exceptions.HTTPError: HTTP error instance.
    """
    response = SimpleNamespace(status_code=status_code)
    return requests.exceptions.HTTPError(
        f"{status_code} transient failure",
        response=response,
    )


class _FlakySchemaStreamingDataset:
    """Streaming stub whose first schema read fails transiently."""

    _ex_iterable = object()

    def __init__(self, failures_remaining: int = 1) -> None:
        """Initialize the flaky streaming dataset."""
        self.failures_remaining = failures_remaining
        self.map_kwargs = None

    def state_dict(self) -> dict[str, int]:
        """Return resumable stream state.

        :return dict[str, int]: Empty cursor payload.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Load resumable stream state.

        :param dict[str, int] state_dict: Empty cursor payload.
        """
        _ = state_dict

    def __iter__(self):
        """Yield examples after a one-time transient schema failure.

        :return collections.abc.Iterator[dict[str, str]]: Example iterator.
        """
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise _http_error(503)
        yield {"text": "hello world", "meta": "source"}

    def map(self, **kwargs):
        """Record mapping kwargs and return self.

        :param Any kwargs: Mapping keyword arguments.
        :return _FlakySchemaStreamingDataset: Self.
        """
        self.map_kwargs = kwargs
        return self


class TestStreamingTokenize(unittest.TestCase):
    """Validate streaming tokenization behavior."""

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
        tokenizer = build_wordlevel_tokenizer(
            vocab={"hello": 2, "world": 3},
            include_mask=False,
            include_sep=False,
        )

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

    def test_streaming_tokenize_retries_schema_peek_for_remove_columns(self):
        """Ensure streaming tokenization uses retry-safe schema inspection."""
        dataset = _FlakySchemaStreamingDataset()
        tokenizer = build_wordlevel_tokenizer(
            vocab={"hello": 2, "world": 3},
            include_mask=False,
            include_sep=False,
        )

        tokenized = tokenize(
            dataset,
            tokenizer,
            column_name="text",
            max_length=4,
            truncation=True,
            remove_columns=True,
            streaming_read_retries=1,
            streaming_read_retry_backoff_seconds=0.01,
            streaming_read_retry_max_backoff_seconds=0.01,
        )

        self.assertIs(tokenized, dataset)
        self.assertEqual(dataset.map_kwargs["remove_columns"], ["text", "meta"])

    def test_streaming_tokenize_raises_after_schema_retry_exhaustion(self):
        """Persistent transient schema failures should not fall back silently."""
        dataset = _FlakySchemaStreamingDataset(failures_remaining=2)
        tokenizer = build_wordlevel_tokenizer(
            vocab={"hello": 2, "world": 3},
            include_mask=False,
            include_sep=False,
        )

        with self.assertRaises(requests.exceptions.HTTPError):
            tokenize(
                dataset,
                tokenizer,
                column_name="text",
                max_length=4,
                truncation=True,
                remove_columns=True,
                streaming_read_retries=1,
                streaming_read_retry_backoff_seconds=0.01,
                streaming_read_retry_max_backoff_seconds=0.01,
            )

        self.assertIsNone(dataset.map_kwargs)

    def test_get_tokenizer_pair_template_uses_single_bos(self):
        """Ensure fallback pair template does not inject BOS before sentence B."""
        base = build_wordlevel_tokenizer(
            vocab={"hello": 2, "world": 3},
            include_mask=False,
            include_sep=False,
        )
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
        base = build_wordlevel_tokenizer(
            vocab={"hello": 2, "world": 3},
            include_mask=False,
            include_sep=False,
        )
        base.mask_token = None

        with patch(
            "neobert.tokenizer.tokenizer.AutoTokenizer.from_pretrained",
            return_value=base,
        ):
            with self.assertRaises(ValueError):
                get_tokenizer("dummy-tokenizer", max_length=32)

    def test_get_tokenizer_allows_missing_mask_when_mlm_enforcement_disabled(self):
        """Ensure non-MLM flows can keep tokenizer special tokens unchanged."""
        base = build_wordlevel_tokenizer(
            vocab={"hello": 2, "world": 3},
            include_mask=False,
            include_sep=False,
        )
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
