"""Tokenized dataset cache identity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

TOKENIZATION_MANIFEST_NAME = "tokenization_manifest.json"


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Atomically replace a JSON file with a complete serialized payload.

    :param str | Path path: Destination JSON path.
    :param Any payload: JSON-serializable payload.
    :return Path: Destination path after replacement.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


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


def tokenizer_serialization_hash(tokenizer: PreTrainedTokenizerBase) -> str | None:
    """Return a hash of a fast tokenizer's full serialization, or ``None``.

    Fast tokenizers expose ``backend_tokenizer.to_str()``, which serializes the
    complete tokenization behavior - vocabulary, BPE merges, normalizer,
    pre-tokenizer, and post-processor - so hashing it captures segmentation
    differences that a token-to-id vocabulary hash alone misses (for example a
    changed merge table or ``add_prefix_space``). Slow tokenizers have no such
    serialization; callers fall back to the vocab hash and explicit fields.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer to fingerprint.
    :return str | None: SHA256 hex digest, or ``None`` for slow tokenizers.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    return hashlib.sha256(backend.to_str().encode("utf-8")).hexdigest()


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
    # The contract records only what changes the produced token IDs, so a valid
    # cache is never spuriously rejected and a mismatched one is never silently
    # reused. It captures *how* the tokenizer tokenizes, not *where* it was
    # loaded from (``name_or_path``) or its advertised default length
    # (``model_max_length``, irrelevant when ``max_length`` is passed explicitly):
    # those are excluded because they drift across checkpoint resume without
    # affecting output. ``tokenizer_serialization_hash`` fingerprints the fast
    # tokenizer's full serialization (vocab, merges, normalizer, pre-tokenizer),
    # and ``truncation_side`` is recorded separately because it governs which
    # tokens are dropped from long examples yet is a runtime attribute absent
    # from that serialization. Padding side is intentionally omitted: tokenize()
    # runs with ``padding=False`` (padding happens at collation), so it does not
    # affect cached token IDs.
    return {
        "schema_version": 3,
        "dataset_name": _jsonable(dataset_name),
        "dataset_config": _jsonable(dataset_config),
        "dataset_path": _jsonable(dataset_path),
        "column_name": _jsonable(column_name),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_serialization_hash": tokenizer_serialization_hash(tokenizer),
        "vocab_hash": tokenizer_vocab_hash(tokenizer),
        "vocab_size": int(len(tokenizer)),
        "special_tokens_map": _jsonable(dict(tokenizer.special_tokens_map)),
        "pad_token_id": _jsonable(tokenizer.pad_token_id),
        "mask_token_id": _jsonable(tokenizer.mask_token_id),
        "truncation_side": _jsonable(getattr(tokenizer, "truncation_side", None)),
        "max_length": int(max_length),
        "truncation": bool(truncation),
        "add_special_tokens": bool(add_special_tokens),
        "return_special_tokens_mask": bool(return_special_tokens_mask),
    }


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
    return write_json_atomic(manifest_path, manifest)
