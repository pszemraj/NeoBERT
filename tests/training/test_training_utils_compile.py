"""Unit tests for torch.compile setup helpers."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DataLoaderConfiguration, DistributedType

from neobert.config import Config, ConfigLoader
from neobert.checkpointing import (
    ACCELERATE_STATE_DIR,
    CHECKPOINT_COMPLETE_NAME,
    OPTIMIZER_PARAM_NAMES_MANIFEST,
    checkpoint_resume_errors,
    mark_checkpoint_complete,
)
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import get_optimizer
from neobert.training_utils import (
    _compute_l2_norm_for_logging,
    _maybe_compile_model,
    _reset_accelerate_runtime_state,
    _resolve_resume_checkpoint,
    _update_global_norm_metric_for_logging,
    attach_optimizer_param_names,
    create_accelerator,
    optimizer_state_semantics,
    preserve_sigterm_handler,
    resolve_runtime_mixed_precision_and_attn_backend,
    resolve_wandb_watch_mode,
    save_optimizer_param_name_manifest,
    should_save_step_checkpoint,
    sync_resume_source_of_truth,
    validate_optimizer_param_name_manifest,
    validate_distributed_runtime_policy,
    validate_muon_distributed_compatibility,
    validate_muon_runtime_topology,
)


def test_preserve_sigterm_handler_restores_after_exception() -> None:
    """Training failures must not leak their SIGTERM handler into the process."""
    original_handler = signal.getsignal(signal.SIGTERM)

    def replacement_handler(signum: int, frame: object) -> None:
        del signum, frame

    @preserve_sigterm_handler()
    def fail_after_installing_handler() -> None:
        signal.signal(signal.SIGTERM, replacement_handler)
        raise RuntimeError("training failed")

    try:
        with pytest.raises(RuntimeError, match="training failed"):
            fail_after_installing_handler()
        assert signal.getsignal(signal.SIGTERM) == original_handler
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    from neobert.contrastive.trainer import trainer as contrastive_trainer
    from neobert.pretraining.trainer import trainer as pretraining_trainer

    assert hasattr(contrastive_trainer, "__wrapped__")
    assert hasattr(pretraining_trainer, "__wrapped__")


def _write_complete_checkpoint_shell(
    checkpoint_path: Path,
    *,
    task: str = "pretraining",
) -> None:
    """Write minimal current-format artifacts for resume-selection tests.

    :param Path checkpoint_path: Existing step directory.
    :param str task: Task recorded in config and marker.
    """
    accelerate_dir = checkpoint_path / ACCELERATE_STATE_DIR
    accelerate_dir.mkdir(exist_ok=True)
    for filename in (
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "random_states_0.pkl",
    ):
        (accelerate_dir / filename).write_bytes(b"x")
    (accelerate_dir / "custom_checkpoint_0.pkl").write_bytes(b"x")
    if task == "pretraining":
        (accelerate_dir / "custom_checkpoint_1.pkl").write_bytes(b"x")
    (checkpoint_path / OPTIMIZER_PARAM_NAMES_MANIFEST).write_text(
        '{"schema_version":1,"state_semantics":"adamw-v1","param_name_groups":[]}\n',
        encoding="utf-8",
    )
    (checkpoint_path / "config.yaml").write_text(f"task: {task}\n", encoding="utf-8")
    (checkpoint_path / "model.safetensors").write_bytes(b"x")
    tokenizer_dir = checkpoint_path / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    mark_checkpoint_complete(checkpoint_path, task=task)


def test_numeric_resume_selector_resolves_under_checkpoint_root(tmp_path: Path) -> None:
    """Bare step selectors resolve to the canonical checkpoints directory."""
    output_dir = tmp_path / "run"
    checkpoint_dir = output_dir / "checkpoints"
    expected = checkpoint_dir / "100"
    expected.mkdir(parents=True)
    _write_complete_checkpoint_shell(expected)
    (output_dir / "100").mkdir()

    resume_path, iteration = _resolve_resume_checkpoint(
        "100", str(checkpoint_dir), str(output_dir)
    )

    assert Path(resume_path) == expected
    assert iteration == 101


def test_latest_resume_preserves_zero_padding_and_skips_incomplete(
    tmp_path: Path,
) -> None:
    """Latest returns the exact newest complete directory spelling."""
    output_dir = tmp_path / "run"
    checkpoint_dir = output_dir / "checkpoints"
    complete = checkpoint_dir / "00050"
    incomplete = checkpoint_dir / "60"
    complete.mkdir(parents=True)
    incomplete.mkdir()
    _write_complete_checkpoint_shell(complete)
    (incomplete / ACCELERATE_STATE_DIR).mkdir()

    resume_path, iteration = _resolve_resume_checkpoint(
        "latest", str(checkpoint_dir), str(output_dir)
    )

    assert Path(resume_path) == complete
    assert iteration == 51


@pytest.mark.parametrize(
    "marker_payload",
    [None, []],
    ids=["null", "array"],
)
def test_latest_resume_skips_non_object_completion_marker(
    tmp_path: Path,
    marker_payload: object,
) -> None:
    """Latest skips checkpoints whose completion marker is not a JSON object."""
    output_dir = tmp_path / "run"
    checkpoint_dir = output_dir / "checkpoints"
    complete = checkpoint_dir / "50"
    damaged = checkpoint_dir / "60"
    complete.mkdir(parents=True)
    damaged.mkdir()
    _write_complete_checkpoint_shell(complete)
    _write_complete_checkpoint_shell(damaged)
    (damaged / CHECKPOINT_COMPLETE_NAME).write_text(
        json.dumps(marker_payload), encoding="utf-8"
    )

    assert checkpoint_resume_errors(damaged) == [
        f"invalid {CHECKPOINT_COMPLETE_NAME}: expected a JSON object"
    ]
    resume_path, iteration = _resolve_resume_checkpoint(
        "latest", str(checkpoint_dir), str(output_dir)
    )

    assert Path(resume_path) == complete
    assert iteration == 51


def test_explicit_incomplete_resume_is_rejected(tmp_path: Path) -> None:
    """Explicit selectors cannot bypass checkpoint completion validation."""
    output_dir = tmp_path / "run"
    checkpoint_dir = output_dir / "checkpoints"
    incomplete = checkpoint_dir / "60"
    incomplete.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="not resumable"):
        _resolve_resume_checkpoint("60", str(checkpoint_dir), str(output_dir))

    with pytest.raises(RuntimeError, match="No complete resumable checkpoints"):
        _resolve_resume_checkpoint("latest", str(checkpoint_dir), str(output_dir))


def _make_cfg() -> Config:
    """Build a minimal config for compile helper tests."""
    cfg = Config()
    cfg.trainer.torch_compile = True
    cfg.trainer.torch_compile_dynamic = None
    cfg.trainer.torch_compile_backend = "inductor"
    cfg.optimizer.name = "adamw"
    cfg.datacollator.pack_sequences = False
    cfg.model.attn_backend = "sdpa"
    return cfg


def _make_accelerator() -> SimpleNamespace:
    """Build a minimal accelerator stub."""
    return SimpleNamespace(distributed_type=DistributedType.NO)


class _RuntimeTopologyShard:
    def __init__(self, dim: int) -> None:
        self.dim = dim


class _RuntimeTopologyMesh:
    def __init__(self, ndim: int = 1) -> None:
        self.ndim = ndim


class _RuntimeTopologyDTensorParam:
    def __init__(
        self,
        *,
        mesh_ndim: int = 1,
        shard_dim: int = 0,
        local: torch.Tensor | None = None,
    ) -> None:
        self.device_mesh = _RuntimeTopologyMesh(mesh_ndim)
        self.placements = (_RuntimeTopologyShard(shard_dim),)
        self._local = torch.zeros(1, 1) if local is None else local

    def to_local(self) -> torch.Tensor:
        return self._local


def _runtime_topology_optimizer(
    *, mesh_ndim: int = 1, shard_dim: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        param_groups=[
            {
                "use_muon": True,
                "params": [
                    _RuntimeTopologyDTensorParam(
                        mesh_ndim=mesh_ndim,
                        shard_dim=shard_dim,
                    )
                ],
            }
        ]
    )


class _RuntimeReplicate:
    pass


class _RuntimeLoggingDTensor:
    def __init__(self, local_value: torch.Tensor, placements: tuple[object, ...]):
        self.device_mesh = _RuntimeTopologyMesh()
        self._local_value = local_value
        self.placements = placements

    def to_local(self) -> torch.Tensor:
        return self._local_value


class _RuntimeShardedGradParam(_RuntimeTopologyDTensorParam):
    def __init__(self, grad: torch.Tensor) -> None:
        super().__init__(local=torch.zeros(0))
        self.grad = grad


class _CompileWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the wrapped module forward.

        :param torch.Tensor inputs: Input tensor.
        :return torch.Tensor: Wrapped module output.
        """
        return self._orig_mod(inputs)


def test_maybe_compile_model_allows_muonclip_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure MuonClip clipping does not block torch.compile."""
    cfg = _make_cfg()
    cfg.optimizer.name = "muonclip"
    model = torch.nn.Linear(8, 8)

    called = {"count": 0}

    def _fake_compile(module: torch.nn.Module, **_: object) -> torch.nn.Module:
        called["count"] += 1
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert called["count"] == 1


def test_maybe_compile_model_uses_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure compile backend from config is forwarded to torch.compile."""
    cfg = _make_cfg()
    cfg.trainer.torch_compile_backend = "aot_eager"
    model = torch.nn.Linear(8, 8)

    captured: dict[str, object] = {}

    def _fake_compile(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert captured["backend"] == "aot_eager"
    assert captured["dynamic"] is False


def test_maybe_compile_model_invalid_backend_falls_back_to_inductor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure unsupported backend names fall back to inductor."""
    cfg = _make_cfg()
    cfg.trainer.torch_compile_backend = "bad_backend"
    model = torch.nn.Linear(8, 8)

    captured: dict[str, object] = {}

    def _fake_compile(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert captured["backend"] == "inductor"


def test_maybe_compile_model_defaults_dynamic_false_for_packed_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure packed flash-attn still defaults to static compile."""
    cfg = _make_cfg()
    cfg.datacollator.pack_sequences = True
    cfg.model.attn_backend = "flash_attn_varlen"
    model = torch.nn.Linear(8, 8)

    captured: dict[str, object] = {}

    def _fake_compile(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert captured["dynamic"] is False


def test_resolve_wandb_watch_mode_matrix() -> None:
    """Ensure WANDB watch-mode defaults and env/config override rules stay stable."""
    cases = [
        ("online", "gradients", None, "gradients", False),
        ("offline", "gradients", None, None, False),
        ("online", "parameters", None, "parameters", False),
        ("online", "gradients", "all", "all", False),
        ("online", "gradients", "weights", "parameters", False),
        ("online", "gradients", "off", None, False),
        ("online", "gradients", "bad", None, True),
    ]
    for wandb_mode, config_value, env_value, expected_mode, expect_warning in cases:
        mode, warning = resolve_wandb_watch_mode(
            wandb_mode=wandb_mode,
            config_value=config_value,
            env_value=env_value,
        )
        assert mode == expected_mode
        if expect_warning:
            assert warning is not None
        else:
            assert warning is None


def test_sync_resume_source_of_truth_uses_checkpoint_config(
    tmp_path: Path,
) -> None:
    """Resume should use checkpoint tokenizer/model/objective fields."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)

    checkpoint_cfg = Config()
    checkpoint_cfg.task = "contrastive"
    checkpoint_cfg.model.hidden_size = 128
    checkpoint_cfg.model.dropout_prob = 0.2
    checkpoint_cfg.tokenizer.name = "checkpoint-tokenizer"
    checkpoint_cfg.tokenizer.max_length = 256
    checkpoint_cfg.dataset.name = "checkpoint-dataset"
    checkpoint_cfg.dataset.path = "checkpoint-data"
    checkpoint_cfg.dataset.max_seq_length = 256
    checkpoint_cfg.datacollator.mlm_probability = 0.3
    checkpoint_cfg.datacollator.pack_sequences = True
    checkpoint_cfg.contrastive.pooling = "max"
    checkpoint_cfg.contrastive.pretraining_prob = 0.4
    checkpoint_cfg.contrastive.pretraining_dataset_path = "checkpoint-simcse"
    checkpoint_cfg.trainer.per_device_train_batch_size = 8
    checkpoint_cfg.trainer.gradient_accumulation_steps = 4
    checkpoint_cfg.optimizer.name = "adamw"
    checkpoint_cfg.optimizer.lr = 2e-5
    checkpoint_cfg.scheduler.name = "linear"
    checkpoint_cfg.scheduler.warmup_percent = 10
    checkpoint_cfg.trainer.gradient_checkpointing = True
    checkpoint_cfg.trainer.torch_compile = True
    ConfigLoader.save(checkpoint_cfg, str(checkpoint_dir / "config.yaml"))

    runtime_cfg = Config()
    runtime_cfg.task = "contrastive"
    runtime_cfg.model.hidden_size = 64
    runtime_cfg.model.dropout_prob = 0.0
    runtime_cfg.tokenizer.name = "runtime-tokenizer"
    runtime_cfg.tokenizer.max_length = 128
    runtime_cfg.dataset.name = "runtime-dataset"
    runtime_cfg.dataset.path = "runtime-data"
    runtime_cfg.dataset.max_seq_length = 128
    runtime_cfg.datacollator.mlm_probability = 0.15
    runtime_cfg.datacollator.pack_sequences = False
    runtime_cfg.contrastive.pooling = "avg"
    runtime_cfg.contrastive.pretraining_prob = 0.0
    runtime_cfg.contrastive.pretraining_dataset_path = "runtime-simcse"
    runtime_cfg.trainer.per_device_train_batch_size = 32
    runtime_cfg.trainer.gradient_accumulation_steps = 1
    runtime_cfg.optimizer.name = "adam"
    runtime_cfg.optimizer.lr = 1e-3
    runtime_cfg.scheduler.name = "cosine"
    runtime_cfg.scheduler.warmup_percent = 5
    runtime_cfg.trainer.gradient_checkpointing = False
    runtime_cfg.trainer.torch_compile = False

    drift = sync_resume_source_of_truth(
        runtime_cfg,
        checkpoint_dir,
        task="contrastive",
        log=logging.getLogger("test"),
    )

    # Model shape/semantics, tokenizer identity, masking, and objective are
    # checkpoint-authoritative (forced back).
    assert runtime_cfg.model.hidden_size == 128
    assert runtime_cfg.model.dropout_prob == 0.2
    assert runtime_cfg.tokenizer.path == str(tokenizer_dir)
    assert runtime_cfg.datacollator.mlm_probability == 0.3
    assert runtime_cfg.datacollator.pack_sequences is True
    assert runtime_cfg.contrastive.pooling == "max"
    assert runtime_cfg.contrastive.pretraining_prob == 0.4
    assert runtime_cfg.contrastive.pretraining_dataset_path == "runtime-simcse"
    assert "contrastive.pretraining_dataset_path" in drift
    # Corpus identity is operator-controlled (launch config wins on resume).
    assert runtime_cfg.dataset.name == "runtime-dataset"
    assert runtime_cfg.dataset.path == "runtime-data"
    # Both configs default to RoPE, so context length is operator-controlled too.
    assert runtime_cfg.tokenizer.max_length == 128
    assert runtime_cfg.dataset.max_seq_length == 128
    # Trainer runtime/performance knobs stay launch-controlled on resume.
    assert runtime_cfg.trainer.per_device_train_batch_size == 32
    assert runtime_cfg.trainer.gradient_accumulation_steps == 1
    assert runtime_cfg.trainer.gradient_checkpointing is False
    assert runtime_cfg.trainer.torch_compile is False
    assert runtime_cfg.optimizer.name == "adamw"
    assert runtime_cfg.optimizer.lr == pytest.approx(2e-5)
    assert runtime_cfg.scheduler.name == "linear"
    assert runtime_cfg.scheduler.warmup_percent == 10


def test_sync_resume_forces_sequence_length_for_non_rope(tmp_path: Path) -> None:
    """Non-RoPE context length is checkpoint-authoritative (learned pos table).

    Corpus identity stays operator-controlled regardless of RoPE, but sequence
    length is forced back for non-RoPE checkpoints because changing it would
    break the strict positional-embedding weight load.
    """
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_cfg = Config()
    checkpoint_cfg.model.rope = False
    checkpoint_cfg.model.max_position_embeddings = 256
    checkpoint_cfg.tokenizer.max_length = 256
    checkpoint_cfg.dataset.max_seq_length = 256
    checkpoint_cfg.datacollator.max_length = 256
    checkpoint_cfg.dataset.name = "checkpoint-dataset"
    ConfigLoader.save(checkpoint_cfg, str(checkpoint_dir / "config.yaml"))

    runtime_cfg = Config()
    runtime_cfg.model.rope = False
    runtime_cfg.model.max_position_embeddings = 512
    runtime_cfg.tokenizer.max_length = 512
    runtime_cfg.dataset.max_seq_length = 512
    runtime_cfg.datacollator.max_length = 512
    runtime_cfg.dataset.name = "runtime-dataset"

    sync_resume_source_of_truth(
        runtime_cfg, checkpoint_dir, task="pretraining", log=logging.getLogger("test")
    )

    # Non-RoPE: sequence length forced back to the checkpoint's.
    assert runtime_cfg.model.max_position_embeddings == 256
    assert runtime_cfg.tokenizer.max_length == 256
    assert runtime_cfg.dataset.max_seq_length == 256
    assert runtime_cfg.datacollator.max_length == 256
    # Corpus identity is still operator-controlled.
    assert runtime_cfg.dataset.name == "runtime-dataset"


def test_sync_resume_preserves_rope_packed_context_and_pretraining_batch_size(
    tmp_path: Path,
) -> None:
    """RoPE context may extend, but a batch cursor keeps checkpoint geometry."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_cfg = Config()
    checkpoint_cfg.model.rope = True
    checkpoint_cfg.model.max_position_embeddings = 1024
    checkpoint_cfg.tokenizer.max_length = 1024
    checkpoint_cfg.dataset.max_seq_length = 1024
    checkpoint_cfg.datacollator.max_length = 1024
    checkpoint_cfg.trainer.per_device_train_batch_size = 8
    ConfigLoader.save(checkpoint_cfg, str(checkpoint_dir / "config.yaml"))

    runtime_cfg = Config()
    runtime_cfg.model.rope = True
    runtime_cfg.model.max_position_embeddings = 2048
    runtime_cfg.tokenizer.max_length = 2048
    runtime_cfg.dataset.max_seq_length = 2048
    runtime_cfg.datacollator.max_length = 2048
    runtime_cfg.trainer.per_device_train_batch_size = 32

    drift = sync_resume_source_of_truth(
        runtime_cfg,
        checkpoint_dir,
        task="pretraining",
        log=logging.getLogger("test"),
    )

    assert runtime_cfg.model.max_position_embeddings == 2048
    assert runtime_cfg.tokenizer.max_length == 2048
    assert runtime_cfg.dataset.max_seq_length == 2048
    assert runtime_cfg.datacollator.max_length == 2048
    assert runtime_cfg.trainer.per_device_train_batch_size == 8
    assert {
        "model.max_position_embeddings",
        "tokenizer.max_length",
        "dataset.max_seq_length",
        "datacollator.max_length",
    } <= drift


def test_sync_resume_preserves_new_corpus_split_selection(tmp_path: Path) -> None:
    """A corpus change must not inherit loader/split choices from the old source."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_cfg = Config()
    checkpoint_cfg.dataset.name = "old-corpus"
    checkpoint_cfg.dataset.config = "old-subset"
    checkpoint_cfg.dataset.streaming = False
    checkpoint_cfg.dataset.train_split = "old-train"
    checkpoint_cfg.dataset.eval_split = "old-validation"
    checkpoint_cfg.dataset.validation_split = 0.1
    checkpoint_cfg.dataset.eval_samples = 100
    checkpoint_cfg.dataset.shuffle_buffer_size = 1000
    ConfigLoader.save(checkpoint_cfg, checkpoint_dir / "config.yaml")

    runtime_cfg = Config()
    runtime_cfg.dataset.name = "new-corpus"
    runtime_cfg.dataset.config = "new-subset"
    runtime_cfg.dataset.streaming = True
    runtime_cfg.dataset.train_split = "train"
    runtime_cfg.dataset.eval_split = None
    runtime_cfg.dataset.validation_split = None
    runtime_cfg.dataset.eval_samples = 500
    runtime_cfg.dataset.shuffle_buffer_size = 20000

    drift = sync_resume_source_of_truth(
        runtime_cfg,
        checkpoint_dir,
        task="pretraining",
        log=logging.getLogger("test"),
    )

    assert runtime_cfg.dataset.name == "new-corpus"
    assert runtime_cfg.dataset.config == "new-subset"
    assert runtime_cfg.dataset.streaming is True
    assert runtime_cfg.dataset.train_split == "train"
    assert runtime_cfg.dataset.eval_split is None
    assert runtime_cfg.dataset.validation_split is None
    assert runtime_cfg.dataset.eval_samples == 500
    assert runtime_cfg.dataset.shuffle_buffer_size == 20000
    assert {
        "dataset.name",
        "dataset.config",
        "dataset.streaming",
        "dataset.train_split",
        "dataset.eval_split",
        "dataset.validation_split",
        "dataset.eval_samples",
        "dataset.shuffle_buffer_size",
    } <= drift


def test_sync_resume_restores_same_corpus_split_selection(tmp_path: Path) -> None:
    """An unchanged corpus must retain checkpoint cursor and split semantics."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_cfg = Config()
    checkpoint_cfg.dataset.name = "same-corpus"
    checkpoint_cfg.dataset.config = "same-subset"
    checkpoint_cfg.dataset.streaming = True
    checkpoint_cfg.dataset.train_split = "train"
    checkpoint_cfg.dataset.eval_split = "validation"
    checkpoint_cfg.dataset.eval_samples = 100
    checkpoint_cfg.dataset.shuffle_buffer_size = 1000
    ConfigLoader.save(checkpoint_cfg, checkpoint_dir / "config.yaml")

    runtime_cfg = Config()
    runtime_cfg.dataset.name = "same-corpus"
    runtime_cfg.dataset.config = "same-subset"
    runtime_cfg.dataset.streaming = False
    runtime_cfg.dataset.train_split = "alternate-train"
    runtime_cfg.dataset.eval_split = None
    runtime_cfg.dataset.eval_samples = 500
    runtime_cfg.dataset.shuffle_buffer_size = 20000

    drift = sync_resume_source_of_truth(
        runtime_cfg,
        checkpoint_dir,
        task="pretraining",
        log=logging.getLogger("test"),
    )

    assert runtime_cfg.dataset.streaming is True
    assert runtime_cfg.dataset.train_split == "train"
    assert runtime_cfg.dataset.eval_split == "validation"
    assert runtime_cfg.dataset.eval_samples == 100
    assert runtime_cfg.dataset.shuffle_buffer_size == 1000
    assert (
        not {
            "dataset.streaming",
            "dataset.train_split",
            "dataset.eval_split",
            "dataset.eval_samples",
            "dataset.shuffle_buffer_size",
        }
        & drift
    )


def test_sync_glue_resume_forces_task_and_cursor_geometry(tmp_path: Path) -> None:
    """GLUE continuation should reconstruct task semantics and sample geometry."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    (checkpoint_dir / "tokenizer").mkdir(parents=True)
    checkpoint_cfg = Config()
    checkpoint_cfg.task = "glue"
    checkpoint_cfg.seed = 17
    checkpoint_cfg.model.name = "checkpoint-model"
    checkpoint_cfg.model.from_hub = True
    checkpoint_cfg.model.max_position_embeddings = 256
    checkpoint_cfg.tokenizer.max_length = 192
    checkpoint_cfg.glue.task_name = "stsb"
    checkpoint_cfg.glue.num_labels = 1
    checkpoint_cfg.glue.max_seq_length = 192
    checkpoint_cfg.glue.classifier_dropout = 0.2
    checkpoint_cfg.trainer.per_device_train_batch_size = 8
    checkpoint_cfg.trainer.gradient_accumulation_steps = 4
    checkpoint_cfg.optimizer.name = "adamw"
    checkpoint_cfg.optimizer.lr = 2e-5
    checkpoint_cfg.scheduler.name = "linear"
    checkpoint_cfg.scheduler.warmup_percent = 10
    ConfigLoader.save(checkpoint_cfg, checkpoint_dir / "config.yaml")

    runtime_cfg = Config()
    runtime_cfg.task = "glue"
    runtime_cfg.seed = 99
    runtime_cfg.model.name = "launch-model"
    runtime_cfg.model.from_hub = False
    runtime_cfg.model.max_position_embeddings = 512
    runtime_cfg.tokenizer.max_length = 128
    runtime_cfg.glue.task_name = "sst2"
    runtime_cfg.glue.num_labels = 2
    runtime_cfg.glue.max_seq_length = 128
    runtime_cfg.glue.classifier_dropout = 0.1
    runtime_cfg.trainer.per_device_train_batch_size = 32
    runtime_cfg.trainer.gradient_accumulation_steps = 1
    runtime_cfg.optimizer.name = "adam"
    runtime_cfg.optimizer.lr = 1e-3
    runtime_cfg.scheduler.name = "cosine"
    runtime_cfg.scheduler.warmup_percent = 5

    sync_resume_source_of_truth(
        runtime_cfg,
        checkpoint_dir,
        task="glue",
        log=logging.getLogger("test"),
    )

    assert runtime_cfg.seed == 17
    assert runtime_cfg.model.name == "checkpoint-model"
    assert runtime_cfg.model.from_hub is True
    assert runtime_cfg.model.max_position_embeddings == 256
    assert runtime_cfg.tokenizer.max_length == 192
    assert runtime_cfg.tokenizer.path == str(checkpoint_dir / "tokenizer")
    assert runtime_cfg.glue.task_name == "stsb"
    assert runtime_cfg.glue.num_labels == 1
    assert runtime_cfg.glue.max_seq_length == 192
    assert runtime_cfg.glue.classifier_dropout == pytest.approx(0.2)
    assert runtime_cfg.trainer.per_device_train_batch_size == 8
    assert runtime_cfg.trainer.gradient_accumulation_steps == 4
    assert runtime_cfg.optimizer.name == "adamw"
    assert runtime_cfg.optimizer.lr == pytest.approx(2e-5)
    assert runtime_cfg.scheduler.name == "linear"
    assert runtime_cfg.scheduler.warmup_percent == 10


def test_sync_resume_source_of_truth_rejects_missing_config(tmp_path: Path) -> None:
    """Resume without checkpoint config must fail before runtime state load."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="config.yaml"):
        sync_resume_source_of_truth(
            Config(),
            checkpoint_dir,
            task="pretraining",
            log=logging.getLogger("test"),
        )


def test_optimizer_param_name_manifest_rejects_reordered_groups(
    tmp_path: Path,
) -> None:
    """Same-shaped parameter reordering must not load optimizer state silently."""

    class OrderedPair(torch.nn.Module):
        def __init__(self, reverse: bool = False) -> None:
            super().__init__()
            if reverse:
                self.second = torch.nn.Linear(2, 2, bias=False)
                self.first = torch.nn.Linear(2, 2, bias=False)
            else:
                self.first = torch.nn.Linear(2, 2, bias=False)
                self.second = torch.nn.Linear(2, 2, bias=False)

    model = OrderedPair(reverse=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    attach_optimizer_param_names(model, optimizer)
    save_optimizer_param_name_manifest(optimizer, tmp_path)

    validate_optimizer_param_name_manifest(optimizer, tmp_path)

    reordered_model = OrderedPair(reverse=True)
    reordered_optimizer = torch.optim.AdamW(reordered_model.parameters(), lr=1e-3)
    attach_optimizer_param_names(reordered_model, reordered_optimizer)
    with pytest.raises(RuntimeError, match="parameter order changed"):
        validate_optimizer_param_name_manifest(reordered_optimizer, tmp_path)


def test_optimizer_param_name_manifest_ignores_runtime_wrapper_prefixes(
    tmp_path: Path,
) -> None:
    """Compile/distributed wrapper prefixes must not affect resume manifests."""
    compiled_checkpoint = tmp_path / "compiled"
    compiled_checkpoint.mkdir()
    compiled_model = _CompileWrapper(torch.nn.Linear(2, 2))
    compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-3)
    attach_optimizer_param_names(compiled_model, compiled_optimizer)
    manifest_path = save_optimizer_param_name_manifest(
        compiled_optimizer,
        compiled_checkpoint,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["param_name_groups"] == [["weight", "bias"]]

    uncompiled_model = torch.nn.Linear(2, 2)
    uncompiled_optimizer = torch.optim.AdamW(uncompiled_model.parameters(), lr=1e-3)
    attach_optimizer_param_names(uncompiled_model, uncompiled_optimizer)
    validate_optimizer_param_name_manifest(uncompiled_optimizer, compiled_checkpoint)

    uncompiled_checkpoint = tmp_path / "uncompiled"
    uncompiled_checkpoint.mkdir()
    save_optimizer_param_name_manifest(uncompiled_optimizer, uncompiled_checkpoint)

    compiled_resume_model = _CompileWrapper(torch.nn.Linear(2, 2))
    compiled_resume_optimizer = torch.optim.AdamW(
        compiled_resume_model.parameters(),
        lr=1e-3,
    )
    attach_optimizer_param_names(compiled_resume_model, compiled_resume_optimizer)
    validate_optimizer_param_name_manifest(
        compiled_resume_optimizer,
        uncompiled_checkpoint,
    )


def test_optimizer_state_semantics_tags() -> None:
    """Semantics tags come from STATE_SEMANTICS or a class-name default."""
    from neobert.optimizer import MuonClipOptimizer

    model = torch.nn.Linear(2, 2, bias=False)
    adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert optimizer_state_semantics(adamw) == "adamw-v1"
    assert MuonClipOptimizer.STATE_SEMANTICS == "muonclip-heavyball-v2"


def test_optimizer_state_semantics_honors_instance_qualified_tags() -> None:
    """Config-qualified instance tags must shadow the class-level tag."""
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.STATE_SEMANTICS = "adamw-v1|norm_factor=spectral"

    assert optimizer_state_semantics(optimizer) == "adamw-v1|norm_factor=spectral"


def test_optimizer_param_name_manifest_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """Checkpoints without a manifest must not resume optimizer state silently."""
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    attach_optimizer_param_names(model, optimizer)

    with pytest.raises(RuntimeError, match="predates the optimizer resume manifest"):
        validate_optimizer_param_name_manifest(optimizer, tmp_path)


def test_optimizer_param_name_manifest_rejects_outdated_schema(
    tmp_path: Path,
) -> None:
    """Bare name-list manifests lack state semantics and must be rejected."""
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    attach_optimizer_param_names(model, optimizer)

    manifest_path = tmp_path / "optimizer_param_names.json"
    manifest_path.write_text(json.dumps([["weight"]]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="outdated manifest schema"):
        validate_optimizer_param_name_manifest(optimizer, tmp_path)

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "state_semantics": "adamw-v1",
                "param_name_groups": [["weight"]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="schema_version must be 1"):
        validate_optimizer_param_name_manifest(optimizer, tmp_path)


def test_optimizer_param_name_manifest_rejects_changed_state_semantics(
    tmp_path: Path,
) -> None:
    """State saved under a different update rule must not load silently."""
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    attach_optimizer_param_names(model, optimizer)
    manifest_path = save_optimizer_param_name_manifest(optimizer, tmp_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["state_semantics"] = "adamw-v0"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="state semantics changed"):
        validate_optimizer_param_name_manifest(optimizer, tmp_path)


def test_create_accelerator_recreates_state_for_mixed_precision_reuse() -> None:
    """Sequential trainer runs should honor updated mixed precision settings."""
    _reset_accelerate_runtime_state()

    try:
        first = create_accelerator(
            use_cpu=True,
            log=logging.getLogger("test"),
            mixed_precision="bf16",
        )
        assert first.device.type == "cpu"
        assert first.state.mixed_precision == "bf16"

        second = create_accelerator(
            use_cpu=True,
            log=logging.getLogger("test"),
            mixed_precision="no",
        )
        assert second.device.type == "cpu"
        assert second.state.mixed_precision == "no"
    finally:
        _reset_accelerate_runtime_state()


def test_create_accelerator_resets_on_state_mismatch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State-mismatch errors should reset Accelerate and retry once."""
    import neobert.training_utils as training_utils

    reset_calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        training_utils.AcceleratorState,
        "_reset_state",
        lambda reset_partial_state=False: reset_calls.append(
            ("accelerator", bool(reset_partial_state))
        ),
    )
    monkeypatch.setattr(
        training_utils.GradientState,
        "_reset_state",
        lambda: reset_calls.append(("gradient", None)),
    )

    calls: list[dict[str, object]] = []

    def _fake_factory(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ValueError(
                "AcceleratorState has already been initialized and cannot be "
                "changed, restart your runtime completely and pass `cpu=True` "
                "to `Accelerator()`."
            )
        return SimpleNamespace(**kwargs)

    out = create_accelerator(
        use_cpu=True,
        log=logging.getLogger("test"),
        accelerator_factory=_fake_factory,
        mixed_precision="bf16",
    )

    assert out.cpu is True
    assert out.mixed_precision == "bf16"
    assert calls == [
        {"mixed_precision": "bf16", "cpu": True},
        {"mixed_precision": "bf16", "cpu": True},
    ]
    assert reset_calls == [("gradient", None), ("accelerator", True)]


def test_create_accelerator_resets_when_cpu_request_reuses_cuda_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU requests should not silently reuse stale CUDA accelerator state."""
    import neobert.training_utils as training_utils

    reset_calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        training_utils.AcceleratorState,
        "_reset_state",
        lambda reset_partial_state=False: reset_calls.append(
            ("accelerator", bool(reset_partial_state))
        ),
    )
    monkeypatch.setattr(
        training_utils.GradientState,
        "_reset_state",
        lambda: reset_calls.append(("gradient", None)),
    )

    calls: list[dict[str, object]] = []

    def _fake_factory(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        device = torch.device("cuda" if len(calls) == 1 else "cpu")
        return SimpleNamespace(device=device, **kwargs)

    out = create_accelerator(
        use_cpu=True,
        log=logging.getLogger("test"),
        accelerator_factory=_fake_factory,
        mixed_precision="bf16",
    )

    assert out.device.type == "cpu"
    assert calls == [
        {"mixed_precision": "bf16", "cpu": True},
        {"mixed_precision": "bf16", "cpu": True},
    ]
    assert reset_calls == [("gradient", None), ("accelerator", True)]


def test_create_accelerator_reraises_unrelated_value_errors() -> None:
    """Non-state errors from Accelerator construction must propagate unchanged."""

    def _boom(**_: object) -> SimpleNamespace:
        raise ValueError("different accelerator failure")

    with pytest.raises(ValueError, match="different accelerator failure"):
        create_accelerator(
            use_cpu=False,
            log=logging.getLogger("test"),
            accelerator_factory=_boom,
            mixed_precision="bf16",
        )


def test_create_accelerator_binds_local_cuda_device_before_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA runs should bind LOCAL_RANK before constructing Accelerator."""
    import neobert.training_utils as training_utils

    bound_devices: list[int] = []

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(training_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_utils.torch.cuda,
        "set_device",
        lambda device: bound_devices.append(int(device)),
    )

    out = create_accelerator(
        use_cpu=False,
        log=logging.getLogger("test"),
        accelerator_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        mixed_precision="bf16",
    )

    assert out.mixed_precision == "bf16"
    assert bound_devices == [3]


def test_create_accelerator_preserves_dataloader_config() -> None:
    """Explicit dataloader config should be forwarded unchanged."""
    dataloader_config = DataLoaderConfiguration(even_batches=False)

    out = create_accelerator(
        use_cpu=True,
        log=logging.getLogger("test"),
        accelerator_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        dataloader_config=dataloader_config,
    )

    assert out.cpu is True
    assert out.dataloader_config is dataloader_config
    assert out.dataloader_config.even_batches is False


def test_update_global_norm_metric_for_logging_computes_on_non_main_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FSDP norm logging must run on every rank even if only rank 0 emits."""
    import neobert.training_utils as training_utils

    calls: list[tuple[tuple[object, ...], bool]] = []

    def _fake_compute(
        parameters: object,
        accelerator: object,
        *,
        grad: bool = False,
    ) -> float:
        del accelerator
        calls.append((tuple(parameters), grad))
        return 12.5

    monkeypatch.setattr(
        training_utils,
        "_compute_l2_norm_for_logging",
        _fake_compute,
    )

    metrics = {"train/weight_norm": 99.0}
    accelerator = SimpleNamespace(is_main_process=False)
    params = (object(), object())

    _update_global_norm_metric_for_logging(
        metrics,
        key="train/weight_norm",
        parameters=params,
        accelerator=accelerator,
        enabled=True,
    )

    assert calls == [(params, False)]
    assert "train/weight_norm" not in metrics


def test_update_global_norm_metric_for_logging_sets_main_process_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main rank should publish the already-collected global norm."""
    import neobert.training_utils as training_utils

    monkeypatch.setattr(
        training_utils,
        "_compute_l2_norm_for_logging",
        lambda *args, **kwargs: 8.5,
    )

    metrics: dict[str, float] = {}
    accelerator = SimpleNamespace(is_main_process=True)

    _update_global_norm_metric_for_logging(
        metrics,
        key="train/weight_norm",
        parameters=(object(),),
        accelerator=accelerator,
        enabled=True,
    )

    assert metrics["train/weight_norm"] == 8.5


def test_resolve_runtime_mixed_precision_and_attn_backend_forces_sdpa_on_cpu(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit CPU runs must disable flash-attn even when CUDA is present."""
    with caplog.at_level(logging.WARNING):
        mixed_precision, attn_backend = (
            resolve_runtime_mixed_precision_and_attn_backend(
                mixed_precision="bf16",
                attn_backend="flash_attn_varlen",
                log=logging.getLogger("test"),
                use_cpu=True,
            )
        )

    assert mixed_precision == "bf16"
    assert attn_backend == "sdpa"
    assert "trainer.use_cpu=true" in caplog.text


def test_resolve_runtime_mixed_precision_and_attn_backend_forces_sdpa_on_fp32(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flash attention should be disabled when the run is explicitly fp32."""
    with caplog.at_level(logging.WARNING):
        mixed_precision, attn_backend = (
            resolve_runtime_mixed_precision_and_attn_backend(
                mixed_precision="no",
                attn_backend="flash_attn_varlen",
                log=logging.getLogger("test"),
                use_cpu=False,
            )
        )

    assert mixed_precision == "no"
    assert attn_backend == "sdpa"
    assert "mixed_precision='no'" in caplog.text


def test_validate_muon_distributed_compatibility_rejects_fsdp1() -> None:
    """MuonClip must fail fast when FSDP v1 is active."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=1)),
    )
    with pytest.raises(RuntimeError, match="requires FSDP v2"):
        validate_muon_distributed_compatibility(
            accelerator=accelerator,
            optimizer_name="muonclip",
            context="unit-test",
        )


def test_validate_muon_distributed_compatibility_allows_fsdp2() -> None:
    """MuonClip should allow FSDP2 runtime."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)),
    )
    validate_muon_distributed_compatibility(
        accelerator=accelerator,
        optimizer_name="muonclip",
        context="unit-test",
    )


def test_validate_muon_distributed_compatibility_rejects_fsdp2_tp_mesh() -> None:
    """MuonClip should fail fast when FSDP2 is combined with extra mesh axes."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(fsdp_version=2),
            parallelism_config=SimpleNamespace(tp_enabled=True, cp_enabled=False),
        ),
    )
    with pytest.raises(RuntimeError, match="1D row-sharded device mesh"):
        validate_muon_distributed_compatibility(
            accelerator=accelerator,
            optimizer_name="muonclip",
            context="unit-test",
        )


def test_validate_muon_distributed_compatibility_rejects_unknown_fsdp() -> None:
    """Unknown FSDP version metadata should default to v1-style rejection."""
    accelerator = SimpleNamespace(distributed_type=DistributedType.FSDP)
    with pytest.raises(RuntimeError, match="requires FSDP v2"):
        validate_muon_distributed_compatibility(
            accelerator=accelerator,
            optimizer_name="muonclip",
            context="unit-test",
        )


@pytest.mark.parametrize("zero_stage", [None, 0, 1, 2, 3])
def test_validate_distributed_runtime_policy_rejects_deepspeed(
    zero_stage: int | None,
) -> None:
    """Repo runtime policy should reject DeepSpeed regardless of optimizer."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.DEEPSPEED,
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(zero_stage=zero_stage),
        ),
    )

    match = "unsupported" if zero_stage is None else f"ZeRO stage {zero_stage}"
    with pytest.raises(RuntimeError, match=match):
        validate_distributed_runtime_policy(
            accelerator=accelerator,
            context="unit-test",
        )


def test_validate_muon_runtime_topology_rejects_multidim_mesh() -> None:
    """Prepared MuonClip DTensor params must reject unsupported mesh rank."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
    )
    optimizer = _runtime_topology_optimizer(mesh_ndim=2)

    with pytest.raises(RuntimeError, match="device_mesh.ndim=2"):
        validate_muon_runtime_topology(
            accelerator=accelerator,
            optimizer=optimizer,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )


def test_validate_muon_runtime_topology_accepts_row_shard_layout() -> None:
    """Prepared MuonClip DTensor params should allow 1D Shard(0) layouts."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
    )
    optimizer = _runtime_topology_optimizer()

    validate_muon_runtime_topology(
        accelerator=accelerator,
        optimizer=optimizer,
        optimizer_name="muonclip",
        log=logging.getLogger("test"),
        context="unit-test",
    )


def test_validate_muon_runtime_topology_rejects_missing_dtensor_params() -> None:
    """Prepared multi-rank MuonClip runs must not continue without DTensor params."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
    )
    optimizer = SimpleNamespace(
        param_groups=[
            {"use_muon": True, "params": [torch.nn.Parameter(torch.zeros(1, 1))]}
        ]
    )

    with pytest.raises(RuntimeError, match="expected DTensor Muon parameters"):
        validate_muon_runtime_topology(
            accelerator=accelerator,
            optimizer=optimizer,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )


def test_compute_l2_norm_for_logging_reduces_only_sharded_dtensors() -> None:
    """Global logged norms must reduce shard contributions without double-counting replicas."""
    reduce_calls: list[tuple[float, str]] = []

    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
        reduce=lambda tensor, reduction="sum": (
            reduce_calls.append((float(tensor.item()), str(reduction))) or tensor * 2
        ),
    )
    parameters = [
        _RuntimeLoggingDTensor(torch.tensor([3.0, 4.0]), (_RuntimeTopologyShard(0),)),
        _RuntimeLoggingDTensor(torch.tensor([1.0, 2.0]), (_RuntimeReplicate(),)),
    ]

    norm = _compute_l2_norm_for_logging(parameters, accelerator)

    assert norm is not None
    assert math.isclose(norm, math.sqrt(55.0), rel_tol=0.0, abs_tol=1e-8)
    assert reduce_calls == [(25.0, "sum")]


def test_compute_l2_norm_for_logging_uses_dtensor_owner_for_gradients() -> None:
    """Gradient logging must reduce local grads when the owning param is sharded."""
    reduce_calls: list[tuple[float, str]] = []
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
        reduce=lambda tensor, reduction="sum": (
            reduce_calls.append((float(tensor.item()), str(reduction))) or tensor * 2
        ),
    )

    norm = _compute_l2_norm_for_logging(
        [_RuntimeShardedGradParam(torch.tensor([6.0, 8.0]))],
        accelerator,
        grad=True,
    )

    assert norm is not None
    assert math.isclose(norm, math.sqrt(200.0), rel_tol=0.0, abs_tol=1e-8)
    assert reduce_calls == [(100.0, "sum")]


def test_get_optimizer_rejects_muonclip_clipping_under_fsdp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FSDP MuonClip with clipping must fail fast, not silently downgrade."""
    model_cfg = NeoBERTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_length=32,
        attn_backend="sdpa",
        hidden_act="gelu",
        rope=False,
    )
    model = NeoBERT(model_cfg)

    with pytest.raises(ValueError, match="enable_clipping=false"):
        get_optimizer(
            model,
            DistributedType.FSDP,
            model_config=model_cfg,
            name="muonclip",
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            muon_config={"enable_clipping": True},
        )

    # An explicit Muon-only run is still allowed under FSDP.
    optimizer = get_optimizer(
        model,
        DistributedType.FSDP,
        model_config=model_cfg,
        name="muonclip",
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        muon_config={"enable_clipping": False},
    )
    assert hasattr(optimizer, "config")
    assert not optimizer.config.enable_clipping

    with caplog.at_level(logging.WARNING):
        get_optimizer(
            model,
            DistributedType.FSDP,
            model_config=model_cfg,
            name="muonclip",
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            muon_config={"enable_clipping": False, "param_policy": "all_2d"},
        )
    assert "materially higher communication cost" in caplog.text


def test_get_optimizer_rejects_muonclip_under_deepspeed() -> None:
    """Optimizer factory should fail fast on unsupported DeepSpeed MuonClip."""
    model_cfg = NeoBERTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_length=32,
        attn_backend="sdpa",
        hidden_act="gelu",
        rope=False,
    )
    model = NeoBERT(model_cfg)

    with pytest.raises(RuntimeError, match="FSDP2-only"):
        get_optimizer(
            model,
            DistributedType.DEEPSPEED,
            model_config=model_cfg,
            name="muonclip",
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            muon_config={"enable_clipping": False},
        )


def test_should_save_step_checkpoint_guarantees_terminal_step() -> None:
    """Terminal step must checkpoint even when max_steps is not a save tick."""
    # save_steps tick
    assert should_save_step_checkpoint(
        step=20, max_steps=100, save_steps=20, save_model=True, save_strategy="steps"
    )
    # non-terminal, non-tick -> no save
    assert not should_save_step_checkpoint(
        step=21, max_steps=100, save_steps=20, save_model=True, save_strategy="steps"
    )
    # terminal step that is NOT a save_steps multiple -> must still save
    assert should_save_step_checkpoint(
        step=101, max_steps=101, save_steps=20, save_model=True, save_strategy="steps"
    )
    # any step at/after max_steps saves (guards >= boundary)
    assert should_save_step_checkpoint(
        step=105, max_steps=101, save_steps=20, save_model=True, save_strategy="steps"
    )
    # disabled saving / non-steps strategy never saves, even at the terminal step
    assert not should_save_step_checkpoint(
        step=101, max_steps=101, save_steps=20, save_model=False, save_strategy="steps"
    )
    assert not should_save_step_checkpoint(
        step=101, max_steps=101, save_steps=20, save_model=True, save_strategy="no"
    )
