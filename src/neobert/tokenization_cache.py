"""Tokenized dataset cache identity helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

TOKENIZATION_MANIFEST_NAME = "tokenization_manifest.json"


def _jsonable(value: Any) -> Any:
    """Convert common config values to stable JSON-compatible values.

    :param Any value: Raw value.
    :return Any: JSON-compatible value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def tokenizer_vocab_hash(tokenizer: PreTrainedTokenizerBase) -> str:
    """Return a stable hash of tokenizer token-to-id assignments.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer to fingerprint.
    :return str: SHA256 hex digest.
    """
    vocab_items = sorted(
        ((str(token), int(index)) for token, index in tokenizer.get_vocab().items()),
        key=lambda item: (item[1], item[0]),
    )
    payload = json.dumps(vocab_items, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_tokenization_manifest(
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset_name: Any = None,
    dataset_config: Any = None,
    dataset_path: Any = None,
    column_name: Any = None,
    max_length: int,
    truncation: bool,
    add_special_tokens: bool,
    return_special_tokens_mask: bool,
) -> dict[str, Any]:
    """Build the tokenization contract persisted beside cached token IDs.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer used for tokenization.
    :param Any dataset_name: Dataset identifier.
    :param Any dataset_config: Dataset config/subset identifier.
    :param Any dataset_path: Local dataset path when used.
    :param Any column_name: Text column or columns tokenized.
    :param int max_length: Tokenization max length.
    :param bool truncation: Whether truncation is enabled.
    :param bool add_special_tokens: Whether tokenizer special tokens are inserted.
    :param bool return_special_tokens_mask: Whether special-token masks are emitted.
    :return dict[str, Any]: Stable manifest payload.
    """
    vocab_hash = tokenizer_vocab_hash(tokenizer)
    return {
        "schema_version": 1,
        "dataset_name": _jsonable(dataset_name),
        "dataset_config": _jsonable(dataset_config),
        "dataset_path": _jsonable(dataset_path),
        "column_name": _jsonable(column_name),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": _jsonable(getattr(tokenizer, "name_or_path", None)),
        "vocab_hash": vocab_hash,
        "vocab_size": int(len(tokenizer)),
        "special_tokens_map": _jsonable(dict(tokenizer.special_tokens_map)),
        "pad_token_id": _jsonable(tokenizer.pad_token_id),
        "mask_token_id": _jsonable(tokenizer.mask_token_id),
        "model_max_length": _jsonable(getattr(tokenizer, "model_max_length", None)),
        "max_length": int(max_length),
        "truncation": bool(truncation),
        "add_special_tokens": bool(add_special_tokens),
        "return_special_tokens_mask": bool(return_special_tokens_mask),
    }


def tokenization_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Return a stable hash for a complete tokenization manifest.

    :param dict[str, Any] manifest: Tokenization manifest.
    :return str: SHA256 hex digest.
    """
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_tokenized_cache_dir(
    *,
    requested_output_dir: str | Path | None,
    dataset_name: Any,
    manifest: dict[str, Any],
    root: str | Path = "tokenized_data",
) -> Path:
    """Resolve the cache directory for a tokenized dataset.

    Explicit paths are honored and guarded by manifest validation. Default paths
    include tokenizer/tokenization fingerprints to avoid accidental reuse.

    :param str | Path | None requested_output_dir: User-specified output path.
    :param Any dataset_name: Dataset identifier for default path naming.
    :param dict[str, Any] manifest: Tokenization manifest.
    :param str | Path root: Root directory for default caches.
    :return Path: Cache directory.
    """
    if requested_output_dir:
        return Path(requested_output_dir)

    dataset_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name or "dataset"))
    vocab_tag = str(manifest["vocab_hash"])[:12]
    contract_tag = tokenization_manifest_fingerprint(manifest)[:10]
    special_tag = "sp" if manifest["add_special_tokens"] else "nosp"
    length_tag = f"L{manifest['max_length']}"
    return (
        Path(root)
        / f"{dataset_key}__{vocab_tag}__{contract_tag}__{length_tag}__{special_tag}"
    )


def validate_tokenized_cache_manifest(
    cache_dir: str | Path,
    expected_manifest: dict[str, Any],
) -> None:
    """Validate that a tokenized cache matches the current tokenization contract.

    :param str | Path cache_dir: Tokenized dataset cache directory.
    :param dict[str, Any] expected_manifest: Expected manifest payload.
    :raises RuntimeError: If the manifest is missing or incompatible.
    """
    manifest_path = Path(cache_dir) / TOKENIZATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"{manifest_path} is missing; refusing to reuse a tokenized cache whose "
            "tokenizer/tokenization contract is unknown."
        )
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved_manifest != expected_manifest:
        raise RuntimeError(
            "Tokenized cache was built with a different tokenizer/tokenization "
            f"contract. Delete {Path(cache_dir)} or choose a different output path."
        )


def write_tokenized_cache_manifest(
    cache_dir: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """Write a tokenization manifest beside a completed tokenized cache.

    :param str | Path cache_dir: Tokenized dataset cache directory.
    :param dict[str, Any] manifest: Manifest payload.
    :return Path: Written manifest path.
    """
    manifest_path = Path(cache_dir) / TOKENIZATION_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path
