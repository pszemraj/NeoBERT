#!/usr/bin/env python3
"""Tests for config-driven pretraining dataset preprocessing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from neobert.tokenizer import tokenize_pretraining_dataset
from scripts.pretraining.preprocess import _load_pretraining_dataset, preprocess


def test_wikibook_preprocessing_combines_supported_sources() -> None:
    """The wikibook selector should build the historical text-only union."""
    bookcorpus = Dataset.from_dict({"text": ["book"]})
    wikipedia = Dataset.from_dict({"text": ["wiki"], "title": ["article"]})

    with patch(
        "scripts.pretraining.preprocess.load_dataset",
        side_effect=(bookcorpus, wikipedia),
    ) as load_dataset_mock:
        dataset = _load_pretraining_dataset("wikibook")

    assert load_dataset_mock.call_args_list == [
        (("bookcorpus",), {"split": "train"}),
        (("wikipedia", "20220301.en"), {"split": "train"}),
    ]
    assert dataset.column_names == ["text"]
    assert sorted(dataset["text"]) == ["book", "wiki"]


@pytest.mark.parametrize(
    ("pack_sequences", "expected_max_length", "expected_add_special_tokens"),
    [(False, 12, True), (True, 10, False)],
)
def test_shared_pretraining_tokenization_contract(
    pack_sequences: bool,
    expected_max_length: int,
    expected_add_special_tokens: bool,
) -> None:
    """The shared helper should own packing boundaries and special-token policy."""
    dataset = MagicMock()
    tokenizer = SimpleNamespace(
        cls_token_id=1,
        bos_token_id=None,
        sep_token_id=2,
        eos_token_id=None,
    )
    tokenized_dataset = MagicMock()

    with patch(
        "neobert.tokenizer.tokenizer.tokenize", return_value=tokenized_dataset
    ) as tokenize_mock:
        result = tokenize_pretraining_dataset(
            dataset,
            tokenizer,
            column_name="text",
            max_length=12,
            truncation=True,
            pack_sequences=pack_sequences,
            return_special_tokens_mask=True,
        )

    assert result is tokenized_dataset
    tokenize_mock.assert_called_once_with(
        dataset,
        tokenizer,
        column_name="text",
        max_length=expected_max_length,
        truncation=True,
        add_special_tokens=expected_add_special_tokens,
        remove_columns=True,
        return_special_tokens_mask=True,
    )


def test_packed_preprocessing_emits_raw_segments_with_reserved_boundaries() -> None:
    """Packed preprocessing leaves outer boundaries to the packing collator."""
    cfg = SimpleNamespace(
        tokenizer=SimpleNamespace(
            name="tokenizer",
            max_length=16,
            trust_remote_code=False,
            revision=None,
            allow_special_token_rewrite=False,
            truncation=True,
        ),
        dataset=SimpleNamespace(
            name="dataset",
            path="tokenized",
            text_column=None,
            max_seq_length=16,
        ),
        datacollator=SimpleNamespace(pack_sequences=True, max_length=12),
    )
    tokenizer = SimpleNamespace(
        cls_token_id=1,
        bos_token_id=None,
        sep_token_id=2,
        eos_token_id=None,
    )
    source_dataset = MagicMock()
    tokenized_dataset = MagicMock()

    with (
        patch("scripts.pretraining.preprocess.get_tokenizer", return_value=tokenizer),
        patch(
            "scripts.pretraining.preprocess.load_dataset", return_value=source_dataset
        ),
        patch(
            "scripts.pretraining.preprocess.resolve_text_column", return_value="text"
        ),
        patch(
            "scripts.pretraining.preprocess.tokenize_pretraining_dataset",
            return_value=tokenized_dataset,
        ) as tokenize_mock,
    ):
        preprocess(cfg)

    tokenize_mock.assert_called_once_with(
        source_dataset,
        tokenizer,
        column_name="text",
        max_length=12,
        truncation=True,
        pack_sequences=True,
        return_special_tokens_mask=True,
    )
    tokenized_dataset.save_to_disk.assert_called_once_with(
        "tokenized", max_shard_size="1GB"
    )
