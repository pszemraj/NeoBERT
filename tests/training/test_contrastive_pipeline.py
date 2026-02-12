#!/usr/bin/env python3
"""Test contrastive training pipeline functionality."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

from neobert.config import ConfigLoader


class TestContrastivePipeline:
    """Test contrastive training pipeline functionality."""

    def test_contrastive_dataset_classes(self):
        """Test contrastive dataset class functionality."""
        try:
            from neobert.contrastive.datasets import get_bsz

            # Test batch size calculation
            bsz = get_bsz("ALLNLI", target_batch_size=8)
            assert bsz == 4  # 8 // 2 = 4 (ALLNLI has factor 2)

            # Test invalid dataset name
            with pytest.raises(ValueError):
                get_bsz("INVALID_DATASET", target_batch_size=8)

        except ImportError as e:
            pytest.skip(f"Contrastive datasets module not available: {e}")

    def test_contrastive_pretrained_checkpoint_root_rejects_legacy_layout(self):
        """Ensure legacy ``model_checkpoints`` roots fail fast for contrastive init."""
        try:
            from neobert.contrastive.trainer import (
                _normalize_contrastive_pretrained_checkpoint_root,
            )
        except ImportError as e:
            pytest.skip(f"Contrastive trainer module not available: {e}")

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_root = Path(tmpdir) / "model_checkpoints"
            legacy_root.mkdir(parents=True, exist_ok=True)

            with pytest.raises(ValueError, match="model_checkpoints"):
                _normalize_contrastive_pretrained_checkpoint_root(legacy_root)

    def test_contrastive_trainer_saves_under_checkpoints_root(
        self, tiny_contrastive_config_path: Path, temp_output_dir: str
    ):
        """Ensure contrastive saves portable weights under ``checkpoints/<step>/``."""
        config = ConfigLoader.load(str(tiny_contrastive_config_path))
        config.dataset.path = temp_output_dir
        config.trainer.output_dir = temp_output_dir
        config.trainer.max_steps = 1
        config.trainer.save_steps = 1
        config.trainer.save_total_limit = 1
        config.trainer.logging_steps = 1
        config.trainer.save_strategy = "steps"
        config.trainer.save_model = True
        config.trainer.per_device_train_batch_size = 1
        config.trainer.use_cpu = True
        config.trainer.disable_tqdm = True
        config.wandb.mode = "disabled"
        config.wandb.enabled = False
        config.contrastive.pretraining_prob = 0.0

        try:
            from datasets import Dataset, DatasetDict
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast

            from neobert.contrastive.trainer import trainer
        except ImportError as e:
            pytest.skip(f"Contrastive trainer dependencies unavailable: {e}")

        dataset_dict = DatasetDict(
            {
                "ALLNLI": Dataset.from_dict(
                    {
                        "input_ids_query": [[2, 3, 0]],
                        "attention_mask_query": [[1, 1, 0]],
                        "input_ids_corpus": [[2, 4, 0]],
                        "attention_mask_corpus": [[1, 1, 0]],
                    }
                )
            }
        )
        pretraining_dataset = Dataset.from_dict(
            {
                "input_ids": [[2, 5, 0]],
                "attention_mask": [[1, 1, 0]],
            }
        )

        def _fake_load_from_disk(path: str):
            if Path(path).name == "all":
                return dataset_dict
            return pretraining_dataset

        def _make_tokenizer() -> PreTrainedTokenizerFast:
            vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "test": 4, "x": 5}
            tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            return PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                pad_token="[PAD]",
                unk_token="[UNK]",
            )

        with (
            mock.patch(
                "neobert.contrastive.trainer.load_from_disk",
                side_effect=_fake_load_from_disk,
            ),
            mock.patch(
                "neobert.contrastive.trainer.get_tokenizer",
                return_value=_make_tokenizer(),
            ),
        ):
            trainer(config)

        step_dir = Path(temp_output_dir) / "checkpoints" / "1"
        assert step_dir.is_dir()
        assert (step_dir / "model.safetensors").is_file()
        assert not (Path(temp_output_dir) / "model_checkpoints").exists()

    def test_muonclip_trainer_passes_model_config(
        self, tiny_contrastive_config_path: Path, temp_output_dir: str
    ):
        """Ensure MuonClip optimizer receives a model config in trainer."""
        config = ConfigLoader.load(str(tiny_contrastive_config_path))
        config.dataset.path = temp_output_dir
        config.optimizer.name = "muonclip"
        config.trainer.max_steps = 0
        config.wandb.mode = "disabled"

        try:
            from datasets import Dataset, DatasetDict
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast

            from neobert.contrastive.trainer import trainer

            dataset_dict = DatasetDict({"ALLNLI": Dataset.from_dict({"dummy": ["x"]})})
            pretraining_dataset = Dataset.from_dict({"dummy": ["x"]})

            def _fake_load_from_disk(path: str):
                if path.endswith("all"):
                    return dataset_dict
                return pretraining_dataset

            def _make_tokenizer() -> PreTrainedTokenizerFast:
                vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2}
                tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
                tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
                return PreTrainedTokenizerFast(
                    tokenizer_object=tokenizer,
                    pad_token="[PAD]",
                    unk_token="[UNK]",
                )

            captured = {}

            def _fake_get_optimizer(
                model, distributed_type, model_config=None, **kwargs
            ):
                captured["model_config"] = model_config
                return torch.optim.Adam(model.parameters(), lr=1e-3)

            with (
                mock.patch(
                    "neobert.contrastive.trainer.load_from_disk",
                    side_effect=_fake_load_from_disk,
                ),
                mock.patch(
                    "neobert.contrastive.trainer.get_tokenizer",
                    return_value=_make_tokenizer(),
                ),
                mock.patch(
                    "neobert.contrastive.trainer.get_optimizer",
                    side_effect=_fake_get_optimizer,
                ),
            ):
                trainer(config)

            assert captured.get("model_config") is not None

        except ImportError as e:
            pytest.skip(f"Contrastive trainer dependencies unavailable: {e}")


class TestContrastiveLoss:
    """Test contrastive loss implementations."""

    @pytest.mark.parametrize(
        ("temperature", "num_views"),
        [
            (0.1, 1),
            (0.05, 2),
        ],
    )
    def test_contrastive_loss_variants(self, temperature: float, num_views: int):
        """Test supervised/self-supervised SupCon loss variants with one matrix."""
        try:
            from neobert.contrastive.loss import SupConLoss

            loss_fn = SupConLoss(temperature=temperature)
            batch_size = 4
            hidden_size = 16
            features = torch.randn(batch_size, num_views, hidden_size)

            queries = features[:, 0, :]
            corpus = features[:, 0 if num_views == 1 else 1, :]
            loss = loss_fn(queries, corpus)

            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert loss.item() >= 0

        except ImportError:
            pytest.skip("SupConLoss not available")
