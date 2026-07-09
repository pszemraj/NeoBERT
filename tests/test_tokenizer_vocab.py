"""Tests for exact tokenizer vocabulary alignment."""

from unittest.mock import patch

import pytest

from neobert.tokenizer import align_tokenizer_vocab
from tests.tokenizer_utils import build_wordlevel_tokenizer


def test_align_tokenizer_vocab_is_noop_at_target() -> None:
    """An already aligned tokenizer should remain unchanged."""
    tokenizer = build_wordlevel_tokenizer()
    original_vocab = tokenizer.get_vocab()

    assert align_tokenizer_vocab(tokenizer, len(tokenizer)) == 0
    assert tokenizer.get_vocab() == original_vocab


def test_align_tokenizer_vocab_adds_deterministic_inert_placeholders() -> None:
    """Alignment should preserve IDs and assign deterministic placeholder IDs."""
    tokenizer = build_wordlevel_tokenizer()
    original_vocab = tokenizer.get_vocab()
    original_size = len(tokenizer)
    target_size = original_size + 4

    added = align_tokenizer_vocab(tokenizer, target_size)

    assert added == 4
    assert len(tokenizer) == target_size
    assert tokenizer.additional_special_tokens == []
    for token, token_id in original_vocab.items():
        assert tokenizer.convert_tokens_to_ids(token) == token_id
    for token_id in range(original_size, target_size):
        token = f"<|neobert_extra_token_{token_id}|>"
        assert tokenizer.convert_tokens_to_ids(token) == token_id


def test_align_tokenizer_vocab_rejects_shrinking() -> None:
    """Exact alignment should reject targets below the existing vocabulary."""
    tokenizer = build_wordlevel_tokenizer()

    with pytest.raises(ValueError, match="smaller target"):
        align_tokenizer_vocab(tokenizer, len(tokenizer) - 1)


def test_align_tokenizer_vocab_rejects_partial_addition() -> None:
    """Alignment should fail if a tokenizer does not reach the exact target."""
    tokenizer = build_wordlevel_tokenizer()
    original_size = len(tokenizer)

    with patch.object(tokenizer, "add_tokens", return_value=1):
        with pytest.raises(
            ValueError,
            match=(
                "Failed to align tokenizer vocabulary: "
                "needed=2, added=1, final_size=6, target=8"
            ),
        ):
            align_tokenizer_vocab(tokenizer, original_size + 2)
