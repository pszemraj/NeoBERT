"""Regression tests for pretraining preprocessing tokenizer wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import warnings


_PREPROCESS_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "pretraining" / "preprocess.py"
)
_PREPROCESS_SPEC = importlib.util.spec_from_file_location(
    "neobert_preprocess_script",
    _PREPROCESS_PATH,
)
if _PREPROCESS_SPEC is None or _PREPROCESS_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_PREPROCESS_PATH}")
_PREPROCESS_MODULE = importlib.util.module_from_spec(_PREPROCESS_SPEC)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _PREPROCESS_SPEC.loader.exec_module(_PREPROCESS_MODULE)


def test_preprocess_forwards_tokenizer_trust_and_revision(monkeypatch):
    """Ensure preprocess() preserves tokenizer trust/revision config overrides."""
    tokenizer_kwargs: dict[str, object] = {}

    class _DummyDataset:
        column_names = ["text"]

        @staticmethod
        def save_to_disk(_path: str, max_shard_size: str = "1GB") -> None:
            del max_shard_size

    class _DummyTokenizer:
        pass

    def _fake_get_tokenizer(*args, **kwargs):
        del args
        tokenizer_kwargs.update(kwargs)
        return _DummyTokenizer()

    monkeypatch.setattr(_PREPROCESS_MODULE, "get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(
        _PREPROCESS_MODULE, "load_dataset", lambda *args, **kwargs: _DummyDataset()
    )
    monkeypatch.setattr(
        _PREPROCESS_MODULE, "resolve_text_column", lambda *args, **kwargs: "text"
    )
    monkeypatch.setattr(
        _PREPROCESS_MODULE, "tokenize", lambda *args, **kwargs: _DummyDataset()
    )

    cfg = SimpleNamespace(
        tokenizer=SimpleNamespace(
            name="org/model-tokenizer",
            max_length=1024,
            truncation=True,
            trust_remote_code=True,
            revision="pinned-sha",
            allow_special_token_rewrite=False,
        ),
        dataset=SimpleNamespace(
            name="dummy-dataset",
            text_column="text",
            path="/tmp/tokenized-dummy",
        ),
    )

    _PREPROCESS_MODULE.preprocess(cfg)

    assert tokenizer_kwargs["pretrained_model_name_or_path"] == "org/model-tokenizer"
    assert tokenizer_kwargs["trust_remote_code"] is True
    assert tokenizer_kwargs["revision"] == "pinned-sha"
    assert tokenizer_kwargs["allow_special_token_rewrite"] is False
