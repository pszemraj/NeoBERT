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


def test_load_neobert_checkpoint_weights_falls_back_to_legacy_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy conversion should be called lazily when safetensors is absent."""
    module = _load_pseudo_perplexity_module()
    model = _ModelStub()
    expected = {"weight": torch.zeros(2, 2)}
    seen: list[tuple[Path, str]] = []

    def _fake_load_deepspeed(path: Path, *, tag: str | None = None):
        seen.append((Path(path), "" if tag is None else str(tag)))
        return expected

    monkeypatch.setattr(module, "load_deepspeed_fp32_state_dict", _fake_load_deepspeed)

    out = module._load_neobert_checkpoint_weights(
        model,
        checkpoint_path=tmp_path,
        checkpoint="456",
    )

    assert out is model
    assert model.loaded_state_dict == expected
    assert seen == [(tmp_path, "456")]
