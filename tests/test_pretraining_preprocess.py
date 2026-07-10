#!/usr/bin/env python3
"""Tests for config-driven pretraining dataset preprocessing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts.pretraining.preprocess import preprocess


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
            "scripts.pretraining.preprocess.tokenize", return_value=tokenized_dataset
        ) as tokenize_mock,
    ):
        preprocess(cfg)

    tokenize_mock.assert_called_once_with(
        source_dataset,
        tokenizer,
        column_name="text",
        max_length=10,
        truncation=True,
        add_special_tokens=False,
        remove_columns=True,
        return_special_tokens_mask=True,
    )
    tokenized_dataset.save_to_disk.assert_called_once_with(
        "tokenized", max_shard_size="1GB"
    )
