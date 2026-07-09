#!/usr/bin/env python3
"""Test contrastive training pipeline functionality."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from datasets import Dataset, DatasetDict

from neobert.config import Config, ConfigLoader
from neobert.contrastive.datasets import get_bsz
from neobert.contrastive.loss import SupConLoss
from neobert.contrastive.trainer import (
    _contrastive_loss_for_backward,
    _load_contrastive_pretrained_backbone_weights,
    _normalize_contrastive_pretrained_checkpoint_root,
    _pool_sequence,
    _resolve_contrastive_dataloader_kwargs,
    _sync_contrastive_runtime_from_pretraining,
    trainer,
)
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.model.wrappers import NeoBERTLMHead
from tests.tokenizer_utils import build_wordlevel_tokenizer


class TestContrastivePipeline:
    """Test contrastive training pipeline functionality."""

    def test_contrastive_dataset_classes(self):
        """Test contrastive dataset class functionality."""
        bsz = get_bsz("ALLNLI", target_batch_size=8)
        assert bsz == 4

        with pytest.raises(ValueError):
            get_bsz("INVALID_DATASET", target_batch_size=8)

    def test_contrastive_pretrained_checkpoint_root_rejects_legacy_layout(self):
        """Ensure legacy ``model_checkpoints`` roots fail fast for contrastive init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_root = Path(tmpdir) / "model_checkpoints"
            legacy_root.mkdir(parents=True, exist_ok=True)

            with pytest.raises(ValueError, match="model_checkpoints"):
                _normalize_contrastive_pretrained_checkpoint_root(legacy_root)

    @pytest.mark.parametrize(
        "max_steps, save_steps",
        # (1, 2): the terminal step is not a save_steps tick, so it exercises the
        # guaranteed final-checkpoint path (nothing would be saved without it).
        [(1, 1), (1, 2)],
    )
    def test_contrastive_trainer_saves_under_checkpoints_root(
        self,
        tiny_contrastive_config_path: Path,
        temp_output_dir: str,
        max_steps: int,
        save_steps: int,
    ):
        """Ensure contrastive saves portable weights under ``checkpoints/<step>/``."""
        config = ConfigLoader.load(str(tiny_contrastive_config_path))
        config.dataset.path = temp_output_dir
        config.trainer.output_dir = temp_output_dir
        config.trainer.max_steps = max_steps
        config.trainer.save_steps = save_steps
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
            return pretraining_dataset

        with (
            mock.patch(
                "neobert.contrastive.trainer.load_from_disk",
                side_effect=_fake_load_from_disk,
            ),
            mock.patch(
                "neobert.contrastive.trainer.load_cached_contrastive_datasets",
                return_value=dataset_dict,
            ) as cached_loader,
            mock.patch(
                "neobert.contrastive.trainer.get_tokenizer",
                return_value=build_wordlevel_tokenizer(
                    vocab={"hello": 2, "world": 3, "test": 4, "x": 5},
                    include_mask=False,
                    include_sep=False,
                ),
            ),
        ):
            trainer(config)

        cached_loader.assert_called_once()
        assert cached_loader.call_args.kwargs["selected_names"] == ["ALLNLI"]
        step_dir = Path(temp_output_dir) / "checkpoints" / str(max_steps)
        assert step_dir.is_dir()
        assert (step_dir / "model.safetensors").is_file()
        assert (step_dir / "config.yaml").is_file()
        assert (step_dir / "optimizer_param_names.json").is_file()
        assert (step_dir / "tokenizer").is_dir()
        assert not (Path(temp_output_dir) / "model_checkpoints").exists()

    def test_contrastive_pooling_modes_respect_attention_mask(self):
        """Configured contrastive pooling should affect pooled embeddings."""
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [5.0, 0.0], [9.0, 9.0]],
                [[3.0, -1.0], [4.0, 6.0], [7.0, 8.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        avg = _pool_sequence(hidden, mask, "avg")
        cls = _pool_sequence(hidden, mask, "cls")
        max_pool = _pool_sequence(hidden, mask, "max")

        assert torch.allclose(avg, torch.tensor([[3.0, 1.0], [3.0, -1.0]]))
        assert torch.allclose(cls, torch.tensor([[1.0, 2.0], [3.0, -1.0]]))
        assert torch.allclose(max_pool, torch.tensor([[5.0, 2.0], [3.0, -1.0]]))

    def test_contrastive_loss_backward_normalizes_summed_loss(self):
        """Summed contrastive CE should be mean-normalized before backward."""
        loss_sum = torch.tensor(6.0, requires_grad=True)
        loss = _contrastive_loss_for_backward(loss_sum, query_count=3)
        loss.backward()

        assert loss.item() == 2.0
        assert torch.allclose(loss_sum.grad, torch.tensor(1.0 / 3.0))

    def test_contrastive_dropout_guard_only_applies_to_simcse_branch(
        self, tiny_contrastive_config_path: Path, temp_output_dir: str
    ):
        """Dropout zero is invalid only when pretraining SimCSE steps can run."""
        config = ConfigLoader.load(str(tiny_contrastive_config_path))
        config.dataset.path = ""
        config.trainer.output_dir = temp_output_dir
        config.model.dropout_prob = 0.0

        config.contrastive.pretraining_prob = 0.1
        with pytest.raises(ValueError, match="pretraining_prob > 0"):
            trainer(config)

        config.contrastive.pretraining_prob = 0.0
        with pytest.raises(ValueError, match="dataset.path"):
            trainer(config)

    def test_muonclip_trainer_passes_model_config(
        self, tiny_contrastive_config_path: Path, temp_output_dir: str
    ):
        """Ensure MuonClip optimizer receives a model config in trainer."""
        config = ConfigLoader.load(str(tiny_contrastive_config_path))
        config.dataset.path = temp_output_dir
        config.optimizer.name = "muonclip"
        config.trainer.max_steps = 0
        config.wandb.mode = "disabled"

        dataset_dict = DatasetDict({"ALLNLI": Dataset.from_dict({"dummy": ["x"]})})
        pretraining_dataset = Dataset.from_dict({"dummy": ["x"]})

        def _fake_load_from_disk(path: str):
            return pretraining_dataset

        captured = {}

        def _fake_get_optimizer(model, distributed_type, model_config=None, **kwargs):
            captured["model_config"] = model_config
            return torch.optim.Adam(model.parameters(), lr=1e-3)

        with (
            mock.patch(
                "neobert.contrastive.trainer.load_from_disk",
                side_effect=_fake_load_from_disk,
            ),
            mock.patch(
                "neobert.contrastive.trainer.load_cached_contrastive_datasets",
                return_value=dataset_dict,
            ),
            mock.patch(
                "neobert.contrastive.trainer.get_tokenizer",
                return_value=build_wordlevel_tokenizer(
                    vocab={"hello": 2},
                    include_mask=False,
                    include_sep=False,
                ),
            ),
            mock.patch(
                "neobert.contrastive.trainer.get_optimizer",
                side_effect=_fake_get_optimizer,
            ),
        ):
            trainer(config)

        assert captured.get("model_config") is not None

    def test_contrastive_loader_kwargs_keep_cuda_pin_memory(self):
        """Contrastive dataloaders should preserve pinned staging on CUDA."""
        cuda_cfg = Config()
        cuda_cfg.dataset.pin_memory = False
        cuda_cfg.trainer.dataloader_num_workers = 4
        dataloader_kwargs, notes = _resolve_contrastive_dataloader_kwargs(
            cuda_cfg,
            device=torch.device("cuda"),
        )
        assert dataloader_kwargs["num_workers"] == 4
        assert dataloader_kwargs["pin_memory"] is True
        assert dataloader_kwargs["shuffle"] is True
        assert len(notes) > 0

        cpu_cfg = Config()
        cpu_cfg.dataset.pin_memory = False
        cpu_cfg.trainer.dataloader_num_workers = 2
        dataloader_kwargs, notes = _resolve_contrastive_dataloader_kwargs(
            cpu_cfg,
            device=torch.device("cpu"),
        )
        assert dataloader_kwargs["num_workers"] == 2
        assert dataloader_kwargs["pin_memory"] is False
        assert dataloader_kwargs["shuffle"] is True
        assert notes == []

    def test_contrastive_pretrained_backbone_loader_strips_lm_head_prefix(self):
        """LM-head checkpoints should load the exact encoder backbone for contrastive."""
        cfg = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=128,
            max_length=16,
        )
        source = NeoBERTLMHead(cfg)
        target = NeoBERT(cfg)

        for param in target.parameters():
            torch.nn.init.zeros_(param)

        _load_contrastive_pretrained_backbone_weights(target, source.state_dict())

        source_state = source.state_dict()
        for key, value in target.state_dict().items():
            assert torch.equal(value, source_state[f"model.{key}"])

    def test_contrastive_pretrained_backbone_loader_rejects_mismatch(self):
        """Partial or shape-mismatched encoder checkpoints must fail loudly."""
        cfg = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=128,
            max_length=16,
        )
        source = NeoBERTLMHead(cfg)
        target = NeoBERT(cfg)
        broken_state = dict(source.state_dict())
        broken_state.pop("model.layer_norm.weight")

        with pytest.raises(ValueError, match="does not match the NeoBERT encoder"):
            _load_contrastive_pretrained_backbone_weights(target, broken_state)

    def test_contrastive_runtime_sync_uses_checkpoint_metadata(self, tmp_path: Path):
        """Checkpoint metadata should drive encoder/tokenizer compatibility for init."""
        cfg = Config()
        cfg.model.hidden_size = 128
        cfg.model.num_hidden_layers = 3
        cfg.model.dropout_prob = 0.2
        cfg.tokenizer.path = None
        cfg.tokenizer.name = "runtime-tokenizer"
        cfg.tokenizer.max_length = 128
        cfg.tokenizer.revision = "runtime-rev"

        pretraining_cfg = Config()
        pretraining_cfg.model.hidden_size = 256
        pretraining_cfg.model.num_hidden_layers = 6
        pretraining_cfg.model.vocab_size = 512
        pretraining_cfg.model.dropout_prob = 0.0
        pretraining_cfg.tokenizer.name = "checkpoint-tokenizer"
        pretraining_cfg.tokenizer.max_length = 384
        pretraining_cfg.tokenizer.revision = "checkpoint-rev"

        checkpoint_step_dir = tmp_path / "123"
        checkpoint_tokenizer_dir = checkpoint_step_dir / "tokenizer"
        checkpoint_tokenizer_dir.mkdir(parents=True, exist_ok=True)

        _sync_contrastive_runtime_from_pretraining(
            cfg,
            pretraining_cfg,
            checkpoint_step_dir=checkpoint_step_dir,
        )

        assert cfg.model.hidden_size == 256
        assert cfg.model.num_hidden_layers == 6
        assert cfg.model.vocab_size == 512
        assert cfg.model.dropout_prob == 0.2
        assert cfg.tokenizer.path == str(checkpoint_tokenizer_dir)
        assert cfg.tokenizer.name == "runtime-tokenizer"
        assert cfg.tokenizer.max_length == 384
        assert cfg.tokenizer.revision == "runtime-rev"

    def test_contrastive_tracker_config_uses_checkpoint_metadata(
        self,
        tmp_path: Path,
    ):
        """Resolved tracker config should include pretrained checkpoint sync."""
        from importlib import import_module

        from neobert.utils import prepare_wandb_config as real_prepare_wandb_config

        trainer_module = import_module("neobert.contrastive.trainer")

        class StopAfterResolvedConfig(Exception):
            pass

        cfg = Config()
        cfg.task = "contrastive"
        cfg.dataset.path = "prepared-contrastive-dataset"
        cfg.contrastive.pretraining_prob = 0.0
        cfg.contrastive.pretrained_checkpoint_dir = str(tmp_path)
        cfg.contrastive.pretrained_checkpoint = "123"
        cfg.model.hidden_size = 128
        cfg.model.num_hidden_layers = 3
        cfg.model.vocab_size = 256
        cfg.tokenizer.path = None
        cfg.tokenizer.max_length = 128
        cfg.wandb.enabled = False
        cfg.wandb.mode = "disabled"

        checkpoint_step_dir = tmp_path / "checkpoints" / "123"
        checkpoint_tokenizer_dir = checkpoint_step_dir / "tokenizer"
        checkpoint_tokenizer_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cfg = Config()
        checkpoint_cfg.model.hidden_size = 384
        checkpoint_cfg.model.num_hidden_layers = 6
        checkpoint_cfg.model.vocab_size = 1024
        checkpoint_cfg.tokenizer.max_length = 512
        ConfigLoader.save(checkpoint_cfg, str(checkpoint_step_dir / "config.yaml"))

        captured: dict[str, object] = {}

        def _capture_prepare(runtime_cfg: Config) -> dict[str, object]:
            payload = real_prepare_wandb_config(runtime_cfg)
            captured["payload"] = payload
            return payload

        def _stop_after_print(message: str) -> None:
            captured["message"] = message
            raise StopAfterResolvedConfig

        accelerator = SimpleNamespace(
            is_main_process=True,
            print=_stop_after_print,
        )
        with (
            mock.patch(
                "neobert.contrastive.trainer.create_accelerator",
                return_value=accelerator,
            ),
            mock.patch(
                "neobert.contrastive.trainer.validate_distributed_runtime_policy",
            ),
            mock.patch(
                "neobert.contrastive.trainer.validate_muon_distributed_compatibility",
            ),
            mock.patch(
                "neobert.contrastive.trainer.prepare_wandb_config",
                side_effect=_capture_prepare,
            ),
            pytest.raises(StopAfterResolvedConfig),
        ):
            trainer_module.trainer(cfg)

        payload = captured["payload"]
        assert isinstance(payload, dict)
        assert payload["model"]["hidden_size"] == 384
        assert payload["model"]["num_hidden_layers"] == 6
        assert payload["model"]["vocab_size"] == 1024
        assert payload["tokenizer"]["path"] == str(checkpoint_tokenizer_dir)
        assert payload["tokenizer"]["max_length"] == 512


class TestContrastiveLoss:
    """Test contrastive loss implementations."""

    def test_contrastive_loss_matches_reference_logits(self):
        """Loss should equal summed cross-entropy over cosine-similarity logits."""
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        corpus = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = SupConLoss(temperature=0.5)(queries, corpus)

        expected_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        expected = torch.nn.functional.cross_entropy(
            expected_logits, torch.arange(2), reduction="sum"
        )
        torch.testing.assert_close(loss, expected)

    def test_contrastive_loss_tensor_and_list_negatives_match(self):
        """Tensor and list negative inputs should construct identical logits."""
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        corpus = queries.clone()
        negative_parts = [
            torch.tensor([[-1.0, 0.0]]),
            torch.tensor([[0.0, -1.0]]),
        ]
        loss_fn = SupConLoss(temperature=0.5)

        tensor_loss = loss_fn(queries, corpus, torch.cat(negative_parts))
        list_loss = loss_fn(queries, corpus, negative_parts)

        torch.testing.assert_close(tensor_loss, list_loss)
        assert tensor_loss > loss_fn(queries, corpus)
