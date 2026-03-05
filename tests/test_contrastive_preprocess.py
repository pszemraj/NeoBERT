"""Regression tests for contrastive preprocessing dataset selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict


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


def test_resolve_dataset_names_accepts_hf_dataset_id_alias() -> None:
    """HF-style dataset IDs should resolve to the canonical contrastive key."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg(
        "sentence-transformers/all-nli",
        dataset_path="/tmp/contrastive",
    )

    assert module._resolve_dataset_names(cfg) == ["ALLNLI"]


def test_resolve_dataset_names_rejects_unknown_selector() -> None:
    """Unknown selectors should fail fast instead of preprocessing everything."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("not-a-real-contrastive-dataset", dataset_path="/tmp/contrastive")

    with pytest.raises(ValueError, match="Unknown contrastive dataset name"):
        module._resolve_dataset_names(cfg)


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
    cached = DatasetDict(
        {
            "ALLNLI": Dataset.from_dict({"query": ["a"], "corpus": ["b"]}),
            "QQP": Dataset.from_dict({"query": ["c"], "corpus": ["d"]}),
        }
    )
    monkeypatch.setattr(module, "load_from_disk", lambda _path: cached)

    dataset = module.pipeline(cfg)

    assert list(dataset.keys()) == ["ALLNLI"]


def test_pipeline_load_all_from_disk_rejects_missing_requested_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cached reloads should error when the requested split is absent."""
    module = _load_contrastive_preprocess_module()
    cfg = _make_cfg("ALLNLI", dataset_path=str(tmp_path), load_all_from_disk=True)
    cached = DatasetDict({"QQP": Dataset.from_dict({"query": ["c"], "corpus": ["d"]})})
    monkeypatch.setattr(module, "load_from_disk", lambda _path: cached)

    with pytest.raises(ValueError, match="missing requested splits"):
        module.pipeline(cfg)
