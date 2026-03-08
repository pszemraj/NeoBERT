"""Regression tests for the pseudo-perplexity evaluation script."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest
import torch


def _load_pseudo_perplexity_module():
    """Load ``scripts/evaluation/pseudo_perplexity.py`` for direct helper tests."""
    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "evaluation"
        / "pseudo_perplexity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "neobert_scripts_evaluation_pseudo_perplexity",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ModelStub:
    """Minimal model stub exposing ``load_state_dict``."""

    def __init__(self) -> None:
        self.loaded_state_dict: dict[str, torch.Tensor] | None = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Capture the loaded state dict for assertions."""
        self.loaded_state_dict = state_dict


def test_pseudo_perplexity_module_imports_without_deepspeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional legacy DeepSpeed dependency must not be imported at module load."""
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deepspeed.utils.zero_to_fp32":
            raise ModuleNotFoundError("simulated missing deepspeed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    module = _load_pseudo_perplexity_module()

    assert hasattr(module, "_load_neobert_checkpoint_weights")


def test_load_neobert_checkpoint_weights_prefers_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable step safetensors should be loaded before legacy conversion."""
    module = _load_pseudo_perplexity_module()
    checkpoint_dir = tmp_path / "123"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / module.MODEL_WEIGHTS_NAME).touch()
    model = _ModelStub()

    expected = {"weight": torch.ones(2, 2)}
    calls = {"safetensors": 0, "deepspeed": 0}

    def _fake_load_model_safetensors(path: Path, *, map_location: str = "cpu"):
        del map_location
        calls["safetensors"] += 1
        assert path == checkpoint_dir
        return expected

    def _fake_load_deepspeed(*args, **kwargs):
        del args, kwargs
        calls["deepspeed"] += 1
        raise AssertionError("Legacy DeepSpeed loader should not run here")

    monkeypatch.setattr(module, "load_model_safetensors", _fake_load_model_safetensors)
    monkeypatch.setattr(module, "load_deepspeed_fp32_state_dict", _fake_load_deepspeed)

    out = module._load_neobert_checkpoint_weights(
        model,
        checkpoint_path=tmp_path,
        checkpoint="123",
    )

    assert out is model
    assert model.loaded_state_dict == expected
    assert calls == {"safetensors": 1, "deepspeed": 0}


def test_load_neobert_checkpoint_weights_resolves_latest_numbered_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`latest` should resolve to the highest loadable numbered checkpoint step."""
    module = _load_pseudo_perplexity_module()
    (tmp_path / "100").mkdir(parents=True, exist_ok=True)
    step_dir = tmp_path / "300"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / module.MODEL_WEIGHTS_NAME).touch()
    (tmp_path / "500").mkdir(parents=True, exist_ok=True)
    model = _ModelStub()
    expected = {"weight": torch.full((2, 2), 3.0)}
    seen_paths: list[Path] = []

    def _fake_load_model_safetensors(path: Path, *, map_location: str = "cpu"):
        del map_location
        seen_paths.append(path)
        return expected

    def _fake_load_deepspeed(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "Legacy DeepSpeed loader should not run for portable latest"
        )

    monkeypatch.setattr(module, "load_model_safetensors", _fake_load_model_safetensors)
    monkeypatch.setattr(module, "load_deepspeed_fp32_state_dict", _fake_load_deepspeed)

    out = module._load_neobert_checkpoint_weights(
        model,
        checkpoint_path=tmp_path,
        checkpoint="latest",
    )

    assert out is model
    assert model.loaded_state_dict == expected
    assert seen_paths == [step_dir]


def test_load_neobert_checkpoint_weights_resolves_latest_file_for_legacy_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`latest` should resolve through legacy DeepSpeed indirection files."""
    module = _load_pseudo_perplexity_module()
    model = _ModelStub()
    expected = {"weight": torch.zeros(2, 2)}
    (tmp_path / "latest").write_text("456\n", encoding="utf-8")
    seen: list[tuple[Path, str]] = []

    def _fake_load_deepspeed(path: Path, *, tag: str | None = None):
        normalized_path = Path(path).resolve()
        normalized_tag = "" if tag is None else str(tag)
        seen.append((normalized_path, normalized_tag))
        return expected

    monkeypatch.setattr(module, "load_deepspeed_fp32_state_dict", _fake_load_deepspeed)

    out = module._load_neobert_checkpoint_weights(
        model,
        checkpoint_path=tmp_path,
        checkpoint="latest",
    )

    assert out is model
    assert model.loaded_state_dict == expected
    assert seen == [(tmp_path.resolve(), "456")]


def test_load_neobert_checkpoint_weights_falls_back_to_legacy_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct step directories should still load through the legacy fallback."""
    module = _load_pseudo_perplexity_module()
    checkpoint_dir = tmp_path / "456"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = _ModelStub()
    expected = {"weight": torch.zeros(2, 2)}
    seen: list[tuple[Path, str]] = []

    def _fake_load_deepspeed(path: Path, *, tag: str | None = None):
        normalized_path = Path(path).resolve()
        normalized_tag = "" if tag is None else str(tag)
        seen.append((normalized_path, normalized_tag))
        if normalized_tag == "456":
            raise FileNotFoundError(
                "explicit root/tag lookup should miss direct step dirs"
            )
        return expected

    monkeypatch.setattr(module, "load_deepspeed_fp32_state_dict", _fake_load_deepspeed)

    out = module._load_neobert_checkpoint_weights(
        model,
        checkpoint_path=checkpoint_dir,
        checkpoint="456",
    )

    assert out is model
    assert model.loaded_state_dict == expected
    assert seen == [(checkpoint_dir.resolve(), "456"), (checkpoint_dir.resolve(), "")]


def test_load_neobert_checkpoint_weights_does_not_ignore_explicit_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing explicit checkpoint tags must not silently fall back to latest."""
    module = _load_pseudo_perplexity_module()
    model = _ModelStub()
    seen: list[tuple[Path, str]] = []

    def _fake_load_deepspeed(path: Path, *, tag: str | None = None):
        normalized_path = Path(path).resolve()
        normalized_tag = "" if tag is None else str(tag)
        seen.append((normalized_path, normalized_tag))
        if tag is None:
            raise AssertionError(
                "direct-path fallback should not run for a missing tag"
            )
        raise FileNotFoundError("requested checkpoint missing")

    monkeypatch.setattr(module, "load_deepspeed_fp32_state_dict", _fake_load_deepspeed)

    with pytest.raises(FileNotFoundError, match="requested checkpoint missing"):
        module._load_neobert_checkpoint_weights(
            model,
            checkpoint_path=tmp_path,
            checkpoint="1000",
        )

    assert seen == [(tmp_path.resolve(), "1000")]


def test_load_neobert_checkpoint_weights_rejects_missing_tag_for_direct_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct portable checkpoint paths must not mask an explicit missing tag."""
    module = _load_pseudo_perplexity_module()
    model = _ModelStub()
    (tmp_path / module.MODEL_WEIGHTS_NAME).touch()

    def _fake_load_model_safetensors(*args, **kwargs):
        del args, kwargs
        raise AssertionError("portable root weights should not load for a missing tag")

    monkeypatch.setattr(module, "load_model_safetensors", _fake_load_model_safetensors)

    with pytest.raises(FileNotFoundError, match="Requested checkpoint 'final'"):
        module._load_neobert_checkpoint_weights(
            model,
            checkpoint_path=tmp_path,
            checkpoint="final",
        )
