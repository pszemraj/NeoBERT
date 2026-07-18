"""Tests for shared Hugging Face export helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

from neobert.config import Config, ConfigLoader

SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "export-hf"


def _load_script(name: str):
    """Load an export script module by filename.

    :param str name: Script stem within ``scripts/export-hf``.
    :return Any: Loaded module.
    """
    spec = spec_from_file_location(f"neobert_test_{name}", SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hf_config_accepts_serialized_model_config(tmp_path: Path, monkeypatch) -> None:
    """A serialized training config should satisfy the HF export contract."""
    config = Config()
    config.model.dropout_prob = 0.25
    config_path = tmp_path / "config.yaml"
    ConfigLoader.save(config, config_path)

    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    exporter = _load_script("export")

    neobert_config = exporter.load_config(config_path)
    hf_config = exporter.create_hf_config(neobert_config, {"weight": torch.zeros(1)})
    validator = _load_script("validate")

    assert hf_config["dropout"] == 0.25
    assert validator._check_required_config_fields(hf_config) is None


def test_packed_swiglu_detection_is_shared() -> None:
    """Shared export contracts should detect packed SwiGLU weights."""
    helpers = _load_script("export_utils")

    assert helpers.has_packed_swiglu_weights({"model.0.ffn.w12.weight": object()})
    assert not helpers.has_packed_swiglu_weights({"model.0.ffn.w1.weight": object()})


def test_metaspace_cleanup_slices_every_batched_tensor() -> None:
    """Metaspace cleanup should preserve alignment across tokenizer outputs."""
    mlm_predict = _load_script("mlm_predict")

    class _Tokenizer:
        mask_token_id = 7

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            assert token == "▁"
            return 3

        @staticmethod
        def convert_ids_to_tokens(token_id: int) -> str:
            assert token_id == 3
            return "▁"

    inputs = {
        "input_ids": torch.tensor([[1, 3, 7, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "token_type_ids": torch.tensor([[0, 0, 0, 0]]),
    }
    cleaned = mlm_predict.clean_metaspace_before_mask(inputs, _Tokenizer())

    torch.testing.assert_close(cleaned["input_ids"], torch.tensor([[1, 7, 4]]))
    torch.testing.assert_close(cleaned["attention_mask"], torch.tensor([[1, 1, 1]]))
    torch.testing.assert_close(cleaned["token_type_ids"], torch.tensor([[0, 0, 0]]))
