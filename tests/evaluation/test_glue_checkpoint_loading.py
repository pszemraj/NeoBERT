"""Regression tests for GLUE pretrained-checkpoint loading paths."""

import logging
from pathlib import Path

import pytest
import torch

from neobert.checkpointing import MODEL_WEIGHTS_NAME
from neobert.glue.train import (
    _normalize_glue_pretrained_checkpoint_root,
    load_pretrained_weights,
)


def test_load_pretrained_weights_prefers_safetensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure portable safetensors payload is preferred when available."""
    checkpoint_dir = tmp_path / "100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / MODEL_WEIGHTS_NAME).touch()

    model = torch.nn.Linear(4, 2)
    expected_weight = torch.full_like(model.weight, 2.0)
    expected_bias = torch.full_like(model.bias, -1.0)

    calls = {"safetensors": 0, "deepspeed": 0}

    def _fake_load_safetensors(*args, **kwargs):
        del args, kwargs
        calls["safetensors"] += 1
        return {
            "weight": expected_weight,
            "bias": expected_bias,
            "decoder.weight": torch.zeros(1),
            "classifier.weight": torch.zeros(1),
        }

    def _fake_load_deepspeed(*args, **kwargs):
        del args, kwargs
        calls["deepspeed"] += 1
        return {"weight": torch.zeros_like(expected_weight)}

    monkeypatch.setattr(
        "neobert.glue.train.load_model_safetensors", _fake_load_safetensors
    )
    monkeypatch.setattr(
        "neobert.glue.train.load_deepspeed_fp32_state_dict",
        _fake_load_deepspeed,
    )

    load_pretrained_weights(
        model,
        checkpoint_dir=str(tmp_path),
        checkpoint_id="100",
        logger=logging.getLogger("test.glue.checkpoint_loading"),
    )

    assert calls["safetensors"] == 1
    assert calls["deepspeed"] == 0
    torch.testing.assert_close(model.weight, expected_weight)
    torch.testing.assert_close(model.bias, expected_bias)


def test_normalize_glue_pretrained_checkpoint_root_preserves_legacy_transfer_root(
    tmp_path: Path,
) -> None:
    """Ensure transfer flow keeps ``model_checkpoints`` roots unchanged."""
    legacy_root = tmp_path / "model_checkpoints"
    legacy_root.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_glue_pretrained_checkpoint_root(legacy_root)

    assert normalized == legacy_root


def test_load_pretrained_weights_filters_only_head_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure filtering drops only ``classifier.``/``decoder.`` head prefixes."""
    checkpoint_dir = tmp_path / "100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / MODEL_WEIGHTS_NAME).touch()

    captured: dict[str, torch.Tensor] = {}

    class _ModelStub(torch.nn.Module):
        def load_state_dict(self, state_dict, strict=False):
            del strict
            captured.update(state_dict)
            return [], []

    def _fake_load_safetensors(*args, **kwargs):
        del args, kwargs
        return {
            "encoder.weight": torch.ones(2, 2),
            "classifier.weight": torch.ones(2, 2),
            "decoder.weight": torch.ones(2, 2),
            "pre_classifier_norm.weight": torch.ones(2),
        }

    monkeypatch.setattr(
        "neobert.glue.train.load_model_safetensors", _fake_load_safetensors
    )

    model = _ModelStub()
    load_pretrained_weights(
        model,
        checkpoint_dir=str(tmp_path),
        checkpoint_id="100",
        logger=logging.getLogger("test.glue.checkpoint_loading"),
    )

    assert "encoder.weight" in captured
    assert "pre_classifier_norm.weight" in captured
    assert "classifier.weight" not in captured
    assert "decoder.weight" not in captured


def test_load_pretrained_weights_falls_back_to_deepspeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure DeepSpeed shard conversion is used when safetensors is missing."""
    checkpoint_dir = tmp_path / "100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = torch.nn.Linear(4, 2)
    expected_weight = torch.full_like(model.weight, 3.0)
    expected_bias = torch.full_like(model.bias, 0.5)

    calls = {"safetensors": 0, "deepspeed": 0}

    def _unexpected_load_safetensors(*args, **kwargs):
        del args, kwargs
        calls["safetensors"] += 1
        raise AssertionError("load_model_safetensors should not be called")

    def _fake_load_deepspeed(*args, **kwargs):
        del args, kwargs
        calls["deepspeed"] += 1
        return {
            "weight": expected_weight,
            "bias": expected_bias,
            "decoder.weight": torch.zeros(1),
        }

    monkeypatch.setattr(
        "neobert.glue.train.load_model_safetensors",
        _unexpected_load_safetensors,
    )
    monkeypatch.setattr(
        "neobert.glue.train.load_deepspeed_fp32_state_dict",
        _fake_load_deepspeed,
    )

    load_pretrained_weights(
        model,
        checkpoint_dir=str(tmp_path),
        checkpoint_id="100",
        logger=logging.getLogger("test.glue.checkpoint_loading"),
    )

    assert calls["safetensors"] == 0
    assert calls["deepspeed"] == 1
    torch.testing.assert_close(model.weight, expected_weight)
    torch.testing.assert_close(model.bias, expected_bias)


def test_load_pretrained_weights_raises_when_no_supported_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure a clear error is raised when neither format is loadable."""
    checkpoint_dir = tmp_path / "100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = torch.nn.Linear(4, 2)

    def _fake_load_deepspeed(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated deepspeed conversion failure")

    monkeypatch.setattr(
        "neobert.glue.train.load_deepspeed_fp32_state_dict",
        _fake_load_deepspeed,
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            r"expected either model\.safetensors or a DeepSpeed "
            r"ZeRO checkpoint layout"
        ),
    ):
        load_pretrained_weights(
            model,
            checkpoint_dir=str(tmp_path),
            checkpoint_id="100",
            logger=logging.getLogger("test.glue.checkpoint_loading"),
        )
