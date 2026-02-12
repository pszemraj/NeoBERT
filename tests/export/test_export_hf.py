#!/usr/bin/env python3
"""Tests for HF export helpers."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


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
            self.assertIn('["neobert.modeling_utils", "modeling_utils"]', model_text)
            self.assertIn(
                '["neobert.huggingface.rotary", "rotary"]',
                model_text,
            )
            self.assertNotIn("from ..modeling_utils import", model_text)

    def test_get_torch_dtype_from_state_dict_handles_uncommon_dtypes(self):
        """Ensure dtype export path uses generic torch dtype string names."""
        export = self.export
        state_dict = {"model.encoder.weight": torch.zeros(2, 2, dtype=torch.complex64)}
        self.assertEqual(
            export.get_torch_dtype_from_state_dict(state_dict),
            "complex64",
        )

    def test_get_torch_dtype_from_state_dict_rejects_fp16(self):
        """Ensure export fails fast for unsupported fp16 checkpoints."""
        export = self.export
        state_dict = {"model.encoder.weight": torch.zeros(2, 2, dtype=torch.float16)}
        with self.assertRaisesRegex(ValueError, "fp16/float16"):
            export.get_torch_dtype_from_state_dict(state_dict)

    def test_load_state_dict_from_deepspeed_tag_dir_uses_parent_and_tag(self):
        """Ensure DeepSpeed tag-dir checkpoints call zero_to_fp32 with root+tag."""
        export = self.export
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "100000"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "mp_rank_00_model_states.pt").write_text("stub")

            calls: dict[str, object] = {}

            def _fake_zero_to_fp32(path: str, tag: str | None = None):
                calls["path"] = path
                calls["tag"] = tag
                return {"model.encoder.weight": torch.zeros(2, 2)}

            zero_module = types.ModuleType("deepspeed.utils.zero_to_fp32")
            zero_module.get_fp32_state_dict_from_zero_checkpoint = _fake_zero_to_fp32
            utils_module = types.ModuleType("deepspeed.utils")
            deepspeed_module = types.ModuleType("deepspeed")

            with patch.dict(
                sys.modules,
                {
                    "deepspeed": deepspeed_module,
                    "deepspeed.utils": utils_module,
                    "deepspeed.utils.zero_to_fp32": zero_module,
                },
            ):
                state_dict = export.load_state_dict_from_checkpoint(checkpoint_dir)

            self.assertIn("model.encoder.weight", state_dict)
            self.assertEqual(Path(str(calls["path"])), checkpoint_dir.parent)
            self.assertEqual(calls["tag"], checkpoint_dir.name)

    @staticmethod
    def _make_tokenizer() -> PreTrainedTokenizerFast:
        """Build a tiny tokenizer for export helper tests."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
        )

    def test_align_tokenizer_vocab_for_export_adds_placeholder_tokens(self):
        """Ensure export helper pads tokenizer to model vocab size when needed."""
        export = self.export
        tokenizer = self._make_tokenizer()

        added = export._align_tokenizer_vocab_for_export(tokenizer, 8)

        self.assertEqual(added, 5)
        self.assertEqual(len(tokenizer), 8)

    def test_align_tokenizer_vocab_for_export_rejects_oversized_tokenizer(self):
        """Ensure export helper fails when tokenizer is larger than model vocab."""
        export = self.export
        tokenizer = self._make_tokenizer()

        with self.assertRaisesRegex(ValueError, "exceeds model vocab_size"):
            export._align_tokenizer_vocab_for_export(tokenizer, 2)

    def test_map_weights_rejects_legacy_decoder_bias_without_opt_in(self):
        """Fail fast when exporting legacy decoder bias without explicit opt-in."""
        export = self.export
        state_dict = {
            "model.decoder.weight": torch.zeros(8, 4),
            "model.decoder.bias": torch.zeros(8),
            "model.encoder.weight": torch.zeros(8, 4),
        }

        with self.assertRaisesRegex(ValueError, "allow-decoder-bias-drop"):
            export.map_weights(state_dict, model_config={})

    def test_map_weights_allows_legacy_decoder_bias_drop_with_opt_in(self):
        """Allow explicit legacy decoder bias dropping with a warning."""
        export = self.export
        state_dict = {
            "decoder.weight": torch.zeros(8, 4),
            "decoder.bias": torch.zeros(8),
            "model.encoder.weight": torch.zeros(8, 4),
        }

        with self.assertWarnsRegex(UserWarning, "dropping this bias changes logits"):
            mapped = export.map_weights(
                state_dict,
                model_config={},
                allow_decoder_bias_drop=True,
            )

        self.assertIn("decoder.weight", mapped)
        self.assertIn("model.encoder.weight", mapped)
        self.assertNotIn("decoder.bias", mapped)
