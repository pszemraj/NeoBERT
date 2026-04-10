"""Tests for tokenized dataset cache identity helpers."""

from __future__ import annotations

import pytest

from neobert.tokenization_cache import (
    build_tokenization_manifest,
    resolve_tokenized_cache_dir,
    validate_tokenized_cache_manifest,
    write_tokenized_cache_manifest,
)
from tests.tokenizer_utils import build_wordlevel_tokenizer


def test_tokenized_cache_manifest_roundtrip(tmp_path):
    """Matching manifests should validate for existing tokenized caches."""
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})
    manifest = build_tokenization_manifest(
        tokenizer,
        dataset_name="unit/dataset",
        dataset_config=None,
        dataset_path=None,
        column_name="text",
        max_length=16,
        truncation=True,
        add_special_tokens=True,
        return_special_tokens_mask=True,
    )

    write_tokenized_cache_manifest(tmp_path, manifest)

    validate_tokenized_cache_manifest(tmp_path, manifest)


def test_tokenized_cache_manifest_rejects_incompatible_contract(tmp_path):
    """Changing max length or tokenizer contract should reject cache reuse."""
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})
    manifest = build_tokenization_manifest(
        tokenizer,
        dataset_name="unit/dataset",
        dataset_config=None,
        dataset_path=None,
        column_name="text",
        max_length=16,
        truncation=True,
        add_special_tokens=True,
        return_special_tokens_mask=False,
    )
    write_tokenized_cache_manifest(tmp_path, manifest)

    changed = dict(manifest)
    changed["max_length"] = 32

    with pytest.raises(RuntimeError, match="different tokenizer/tokenization"):
        validate_tokenized_cache_manifest(tmp_path, changed)


def test_default_tokenized_cache_dir_includes_contract_fingerprint():
    """Default cache paths should change when tokenization inputs change."""
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})
    manifest = build_tokenization_manifest(
        tokenizer,
        dataset_name="unit/dataset",
        dataset_config=None,
        dataset_path=None,
        column_name="text",
        max_length=16,
        truncation=True,
        add_special_tokens=True,
        return_special_tokens_mask=False,
    )
    changed = dict(manifest)
    changed["max_length"] = 32

    first = resolve_tokenized_cache_dir(
        requested_output_dir=None,
        dataset_name="unit/dataset",
        manifest=manifest,
    )
    second = resolve_tokenized_cache_dir(
        requested_output_dir=None,
        dataset_name="unit/dataset",
        manifest=changed,
    )

    assert first != second
    assert first.parent.name == "tokenized_data"
