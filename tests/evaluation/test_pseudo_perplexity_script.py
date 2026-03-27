"""Regression tests for the pseudo-perplexity evaluation script."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    assert not hasattr(module, "AutoModelWithLMHead")


def test_load_hub_masked_lm_uses_masked_lm_auto_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub MLM loading should not depend on deprecated auto-class aliases."""
    module = _load_pseudo_perplexity_module()
    config = SimpleNamespace(max_position_embeddings=128)
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=256, max_length=64),
        roberta=SimpleNamespace(embeddings="original"),
    )
    calls: list[tuple[str, bool]] = []

    def _fake_config_from_pretrained(
        model_name: str, *, trust_remote_code: bool = False
    ):
        assert model_name == "roberta-base"
        assert trust_remote_code is True
        return config

    def _fake_model_from_pretrained(
        model_name: str, *, trust_remote_code: bool = False
    ):
        calls.append((model_name, trust_remote_code))
        return model

    monkeypatch.setattr(
        module.AutoConfig,
        "from_pretrained",
        _fake_config_from_pretrained,
    )
    monkeypatch.setattr(
        module.AutoModelForMaskedLM,
        "from_pretrained",
        _fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        module,
        "RobertaEmbeddings",
        lambda cfg: ("roberta-embeddings", cfg.max_position_embeddings),
    )

    out = module._load_hub_masked_lm("roberta-base", max_length=512)

    assert out is model
    assert calls == [("roberta-base", True)]
    assert model.roberta.embeddings == ("roberta-embeddings", 512)
    assert model.config.max_position_embeddings == 512
    assert model.config.max_length == 512


def test_load_neobert_checkpoint_weights_prefers_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The script wrapper should delegate checkpoint loading to the shared helper."""
    module = _load_pseudo_perplexity_module()
    model = _ModelStub()
    expected = {"weight": torch.ones(2, 2)}
    calls: list[tuple[Path, str, str]] = []

    def _fake_load_state_dict(
        path: Path, checkpoint: str, *, map_location: str = "cpu"
    ):
        calls.append((Path(path), checkpoint, map_location))
        return expected

    monkeypatch.setattr(
        module, "load_step_checkpoint_state_dict", _fake_load_state_dict
    )

    out = module._load_neobert_checkpoint_weights(
        model,
        checkpoint_path=tmp_path,
        checkpoint="123",
    )

    assert out is model
    assert model.loaded_state_dict == expected
    assert calls == [(tmp_path, "123", "cpu")]
