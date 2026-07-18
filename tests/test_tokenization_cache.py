"""Tests for tokenized dataset cache identity helpers."""

from __future__ import annotations

import pytest
from tokenizers import pre_tokenizers

from neobert.tokenization_cache import (
    build_tokenization_manifest,
    validate_tokenized_cache_manifest,
    write_tokenized_cache_manifest,
)
from tests.tokenizer_utils import build_wordlevel_tokenizer


def _manifest(tokenizer, **overrides):
    """Build a manifest with test defaults, overridable per call."""
    kwargs = dict(
        dataset_name="unit/dataset",
        dataset_config=None,
        dataset_path=None,
        column_name="text",
        max_length=16,
        truncation=True,
        add_special_tokens=True,
        return_special_tokens_mask=True,
    )
    kwargs.update(overrides)
    return build_tokenization_manifest(tokenizer, **kwargs)


def test_tokenized_cache_manifest_roundtrip(tmp_path):
    """Matching manifests should validate for existing tokenized caches."""
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})
    manifest = _manifest(tokenizer)

    write_tokenized_cache_manifest(tmp_path, manifest)

    validate_tokenized_cache_manifest(tmp_path, manifest)


def test_tokenized_cache_manifest_rejects_incompatible_contract(tmp_path):
    """Changing max length or tokenizer contract should reject cache reuse."""
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})
    manifest = _manifest(tokenizer, return_special_tokens_mask=False)
    write_tokenized_cache_manifest(tmp_path, manifest)

    changed = dict(manifest)
    changed["max_length"] = 32

    with pytest.raises(RuntimeError, match="different tokenizer/tokenization"):
        validate_tokenized_cache_manifest(tmp_path, changed)


def test_manifest_is_stable_across_tokenizer_path_change(tmp_path):
    """Cache identity must not depend on where the tokenizer was loaded from.

    On checkpoint resume the tokenizer is reloaded from the checkpoint-local
    ``tokenizer/`` directory, so ``name_or_path`` differs from the original run
    even though the tokenizer is identical (and ``model_max_length`` can be
    force-set at save time). The manifest and its validation must stay stable,
    otherwise a valid cache is spuriously rejected after resume.
    """
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})

    tokenizer.name_or_path = "bert-base-uncased"
    original = _manifest(tokenizer)
    tokenizer.name_or_path = str(tmp_path / "checkpoints" / "1000" / "tokenizer")
    resumed = _manifest(tokenizer)

    assert original == resumed
    # Fields that identify load location or advertised length, not tokenization
    # behavior, are excluded so they cannot spuriously invalidate a cache.
    assert "tokenizer_name_or_path" not in original
    assert "model_max_length" not in original

    write_tokenized_cache_manifest(tmp_path, original)
    validate_tokenized_cache_manifest(tmp_path, resumed)


def test_manifest_records_truncation_side(tmp_path):
    """Truncation side changes which tokens survive long examples.

    Right- vs left-truncation drops opposite ends of an over-length sequence, so
    a cache built under one side must not validate against the other even when
    vocab, special tokens, ``max_length``, and ``truncation=True`` are identical.
    """
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})

    tokenizer.truncation_side = "right"
    right = _manifest(tokenizer)
    write_tokenized_cache_manifest(tmp_path, right)

    tokenizer.truncation_side = "left"
    left = _manifest(tokenizer)

    assert right["truncation_side"] == "right"
    assert left["truncation_side"] == "left"
    with pytest.raises(RuntimeError, match="different tokenizer/tokenization"):
        validate_tokenized_cache_manifest(tmp_path, left)


def test_manifest_records_tokenizer_serialization(tmp_path):
    """A segmentation change with an unchanged vocab table must change the contract.

    Swapping the pre-tokenizer alters how text is split into tokens without
    touching the token-to-id table, so the vocab hash alone cannot tell the two
    tokenizers apart. The full-serialization fingerprint must, otherwise stale
    token IDs from the old segmentation get silently reused.
    """
    tokenizer = build_wordlevel_tokenizer(vocab={"hello": 4, "world": 5})
    before = _manifest(tokenizer)
    write_tokenized_cache_manifest(tmp_path, before)

    tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    after = _manifest(tokenizer)

    assert before["vocab_hash"] == after["vocab_hash"]
    assert (
        before["tokenizer_serialization_hash"] != after["tokenizer_serialization_hash"]
    )
    with pytest.raises(RuntimeError, match="different tokenizer/tokenization"):
        validate_tokenized_cache_manifest(tmp_path, after)
