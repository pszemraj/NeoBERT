#!/usr/bin/env python3
"""Tests for HF export helpers."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


def _load_export_module():
    repo_root = Path(__file__).resolve().parents[2]
    export_path = repo_root / "scripts" / "export-hf" / "export.py"
    spec = importlib.util.spec_from_file_location("export_hf", export_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load export module from {export_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExportHF(unittest.TestCase):
    """Regression tests for HF export utilities."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the export module once for the test suite."""
        cls.export = _load_export_module()

    def test_validate_state_dict_accepts_decoder_alias(self):
        """Validate decoder.* keys are accepted via aliasing."""
        export = self.export
        model_config = {
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 6,
            "vocab_size": 8,
            "max_position_embeddings": 16,
            "norm_eps": 1e-5,
            "pad_token_id": 0,
            "hidden_act": "swiglu",
        }
        mlp_hidden = export._swiglu_intermediate_size(model_config["intermediate_size"])
        state_dict = {
            "model.encoder.weight": torch.zeros(
                model_config["vocab_size"], model_config["hidden_size"]
            ),
            "decoder.weight": torch.zeros(
                model_config["vocab_size"], model_config["hidden_size"]
            ),
            "decoder.bias": torch.zeros(model_config["vocab_size"]),
            "model.transformer_encoder.0.qkv.weight": torch.zeros(
                model_config["hidden_size"] * 3, model_config["hidden_size"]
            ),
            "model.transformer_encoder.0.wo.weight": torch.zeros(
                model_config["hidden_size"], model_config["hidden_size"]
            ),
            "model.transformer_encoder.0.ffn.w1.weight": torch.zeros(
                mlp_hidden, model_config["hidden_size"]
            ),
            "model.transformer_encoder.0.ffn.w2.weight": torch.zeros(
                mlp_hidden, model_config["hidden_size"]
            ),
            "model.transformer_encoder.0.ffn.w3.weight": torch.zeros(
                model_config["hidden_size"], mlp_hidden
            ),
        }

        export.validate_state_dict_layout(state_dict, model_config)

        self.assertIn("model.decoder.weight", state_dict)
        self.assertIn("model.decoder.bias", state_dict)

    def test_copy_hf_modeling_files_includes_modeling_utils(self):
        """Ensure modeling_utils.py is copied for exported repos."""
        export = self.export
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            export.copy_hf_modeling_files(target_dir)

            self.assertTrue((target_dir / "model.py").exists())
            self.assertTrue((target_dir / "rotary.py").exists())
            self.assertTrue((target_dir / "modeling_utils.py").exists())
            model_text = (target_dir / "model.py").read_text()
            self.assertIn(
                "from .modeling_utils import swiglu_intermediate_size", model_text
            )
            self.assertIn(
                "from .modeling_utils import scaled_dot_product_attention_compat",
                model_text,
            )
