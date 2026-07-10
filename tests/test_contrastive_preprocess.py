"""Regression tests for contrastive preprocessing dataset selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, load_from_disk

from neobert.contrastive.datasets import (
    CONTRASTIVE_DATASETS,
    build_contrastive_tokenization_manifest,
    load_cached_contrastive_datasets,
    resolve_contrastive_dataset_name,
)
from neobert.tokenization_cache import (
    write_tokenized_cache_manifest,
)
from tests.tokenizer_utils import build_wordlevel_tokenizer


def _load_contrastive_preprocess_module():
    """Load ``scripts/contrastive/preprocess.py`` for direct helper testing."""
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "contrastive"
        / "preprocess.py"
    )
    spec = importlib.util.spec_from_file_location(
        "neobert_scripts_contrastive_preprocess",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_cfg(
    dataset_name: str | list[str] | None,
    *,
    dataset_path: str,
    load_all_from_disk: bool = False,
):
    """Build a minimal config stub for preprocess helper tests."""
    return SimpleNamespace(
        dataset=SimpleNamespace(
            name=dataset_name,
            path=dataset_path,
            load_all_from_disk=load_all_from_disk,
            force_redownload=False,
        ),
        tokenizer=SimpleNamespace(
            name="bert-base-uncased",
            path=None,
            max_length=128,
            truncation=True,
            trust_remote_code=False,
            revision=None,
            allow_special_token_rewrite=False,
        ),
    )


def _save_cached_split(
    all_dir: Path,
    name: str,
    tokenizer,
    cfg,
) -> None:
    """Save one tokenized contrastive split with its manifest."""
    dataset_dir = all_dir / name
    dataset = Dataset.from_dict(
        {
            "input_ids_query": [[4]],
            "attention_mask_query": [[1]],
            "input_ids_corpus": [[5]],
            "attention_mask_corpus": [[1]],
        }
    )
    dataset.save_to_disk(dataset_dir)
    manifest = build_contrastive_tokenization_manifest(
        tokenizer,
        dataset_name=name,
        column_names=dataset.column_names,
        max_length=cfg.tokenizer.max_length,
        truncation=cfg.tokenizer.truncation,
    )
    write_tokenized_cache_manifest(dataset_dir, manifest)


def test_resolve_dataset_names_accepts_hf_dataset_id_alias() -> None:
    """HF-style dataset IDs should resolve to the canonical contrastive key."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg(
        "sentence-transformers/all-nli",
        dataset_path="/tmp/contrastive",
    )

    assert module._resolve_dataset_names(cfg) == ["ALLNLI"]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("embedding-data/QQP_triplets", "QQP"),
        ("sentence-transformers/trivia-qa-triplet", "TRIVIAQA"),
        ("WhereIsAI/github-issue-similarity", "GITHUBISSUE"),
        ("stanfordnlp/concurrentqa-retrieval", "CONCURRENTQA"),
        ("tomaarsen/gooaq-hard-negatives", "GOOAQ"),
    ],
)
def test_resolve_contrastive_dataset_name_accepts_builtin_hf_ids(
    requested: str, expected: str
) -> None:
    """Built-in wrappers should accept the HF dataset IDs they actually use."""
    assert resolve_contrastive_dataset_name(requested) == expected


@pytest.mark.parametrize("requested", ["my-org/all-nli", "private/QQP_triplets"])
def test_resolve_contrastive_dataset_name_preserves_hf_namespaces(
    requested: str,
) -> None:
    """Unknown namespaced Hub repos must not silently collapse to built-ins."""
    with pytest.raises(ValueError, match="Unknown contrastive dataset name"):
        resolve_contrastive_dataset_name(requested)


def test_resolve_dataset_names_rejects_unknown_selector() -> None:
    """Unknown selectors should fail fast instead of preprocessing everything."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("not-a-real-contrastive-dataset", dataset_path="/tmp/contrastive")

    with pytest.raises(ValueError, match="Unknown contrastive dataset name"):
        module._resolve_dataset_names(cfg)


def test_resolve_dataset_names_treats_shared_default_as_all() -> None:
    """The inherited pretraining dataset default should behave like an unset selector."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("refinedweb", dataset_path="/tmp/contrastive")

    assert module._resolve_dataset_names(cfg) == list(CONTRASTIVE_DATASETS.keys())


def test_pipeline_load_all_from_disk_filters_selected_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cached reloads should honor the selected contrastive datasets."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg(
        "sentence-transformers/all-nli",
        dataset_path=str(tmp_path),
        load_all_from_disk=True,
    )
    all_dir = tmp_path / "all"
    tokenizer = build_wordlevel_tokenizer(vocab={"a": 4, "b": 5})
    _save_cached_split(all_dir, "ALLNLI", tokenizer, cfg)
    _save_cached_split(all_dir, "QQP", tokenizer, cfg)
    monkeypatch.setattr(module, "get_tokenizer", lambda **_: tokenizer)

    dataset = module.pipeline(cfg)

    assert list(dataset.keys()) == ["ALLNLI"]


def test_pipeline_load_all_from_disk_rejects_missing_requested_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cached reloads should error when the requested split is absent."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("ALLNLI", dataset_path=str(tmp_path), load_all_from_disk=True)
    tokenizer = build_wordlevel_tokenizer(vocab={"a": 4, "b": 5})
    _save_cached_split(tmp_path / "all", "QQP", tokenizer, cfg)
    monkeypatch.setattr(module, "get_tokenizer", lambda **_: tokenizer)

    with pytest.raises(ValueError, match="missing requested splits"):
        module.pipeline(cfg)


def test_pipeline_load_all_from_disk_rejects_missing_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The fast disk-only path still validates tokenizer provenance."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("ALLNLI", dataset_path=str(tmp_path), load_all_from_disk=True)
    dataset_dir = tmp_path / "all" / "ALLNLI"
    Dataset.from_dict(
        {
            "input_ids_query": [[4]],
            "attention_mask_query": [[1]],
            "input_ids_corpus": [[5]],
            "attention_mask_corpus": [[1]],
        }
    ).save_to_disk(dataset_dir)
    tokenizer = build_wordlevel_tokenizer(vocab={"a": 4, "b": 5})
    monkeypatch.setattr(module, "get_tokenizer", lambda **_: tokenizer)

    with pytest.raises(RuntimeError, match="tokenization_manifest.json"):
        module.pipeline(cfg)


def test_cached_loader_rejects_same_size_different_vocabulary(tmp_path: Path) -> None:
    """Training cannot consume IDs whose token-to-ID mapping changed."""
    cfg = _make_cfg("ALLNLI", dataset_path=str(tmp_path), load_all_from_disk=True)
    original = build_wordlevel_tokenizer(vocab={"a": 4, "b": 5})
    reordered = build_wordlevel_tokenizer(vocab={"a": 5, "b": 4})
    _save_cached_split(tmp_path / "all", "ALLNLI", original, cfg)

    with pytest.raises(RuntimeError, match="different tokenizer"):
        load_cached_contrastive_datasets(
            tmp_path / "all",
            selected_names=["ALLNLI"],
            tokenizer=reordered,
            max_length=cfg.tokenizer.max_length,
            truncation=cfg.tokenizer.truncation,
        )


def test_cached_loader_rejects_token_ids_without_attention_mask(
    tmp_path: Path,
) -> None:
    """Every cached token-ID view requires a matching attention mask."""
    dataset_dir = tmp_path / "all" / "ALLNLI"
    Dataset.from_dict(
        {
            "input_ids_query": [[4]],
            "attention_mask_query": [[1]],
            "input_ids_corpus": [[5]],
        }
    ).save_to_disk(dataset_dir)
    tokenizer = build_wordlevel_tokenizer(vocab={"a": 4, "b": 5})

    with pytest.raises(ValueError, match="attention_mask_corpus"):
        load_cached_contrastive_datasets(
            tmp_path / "all",
            selected_names=["ALLNLI"],
            tokenizer=tokenizer,
            max_length=128,
            truncation=True,
        )


def test_pipeline_subset_refresh_preserves_other_cached_manifest_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Subset preprocess runs should not drop untouched cached split manifests."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("ALLNLI", dataset_path=str(tmp_path), load_all_from_disk=False)
    all_dir = tmp_path / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_wordlevel_tokenizer(vocab={"a": 4, "b": 5})
    _save_cached_split(all_dir, "ALLNLI", tokenizer, cfg)
    _save_cached_split(all_dir, "QQP", tokenizer, cfg)
    (all_dir / "dataset_dict.json").write_text(
        '{"splits": ["ALLNLI", "QQP"]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "get_tokenizer", lambda **_: tokenizer)

    dataset = module.pipeline(cfg)

    assert list(dataset.keys()) == ["ALLNLI"]
    reloaded = load_from_disk(all_dir)
    assert list(reloaded.keys()) == ["ALLNLI", "QQP"]


def test_pipeline_rejects_cached_split_without_tokenization_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing contrastive caches need a manifest before reuse."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("ALLNLI", dataset_path=str(tmp_path), load_all_from_disk=False)
    all_dir = tmp_path / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(
        {
            "input_ids_query": [[4]],
            "attention_mask_query": [[1]],
            "input_ids_corpus": [[5]],
            "attention_mask_corpus": [[1]],
        }
    ).save_to_disk(all_dir / "ALLNLI")

    tokenizer = build_wordlevel_tokenizer(vocab={"a": 4, "b": 5})
    monkeypatch.setattr(module, "get_tokenizer", lambda **_: tokenizer)

    with pytest.raises(RuntimeError, match="tokenization_manifest.json"):
        module.pipeline(cfg)
