"""Shared checkpoint-retention policy tests for trainer implementations."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from neobert.config import Config


@pytest.mark.parametrize(
    "module_path",
    [
        "neobert.pretraining.trainer",
        "neobert.contrastive.trainer",
    ],
)
def test_resolve_checkpoint_retention_limit_prefers_save_total_limit(
    module_path: str,
) -> None:
    """save_total_limit should take precedence over legacy max_ckpt."""
    module = importlib.import_module(module_path)

    cfg = Config()
    cfg.trainer.save_total_limit = 1
    cfg.trainer.max_ckpt = 7
    assert module._resolve_checkpoint_retention_limit(cfg) == 1

    cfg.trainer.save_total_limit = None
    cfg.trainer.max_ckpt = 3
    assert module._resolve_checkpoint_retention_limit(cfg) == 3

    cfg.trainer.save_total_limit = None
    cfg.trainer.max_ckpt = None
    assert module._resolve_checkpoint_retention_limit(cfg) == 0


@pytest.mark.parametrize(
    "module_path",
    [
        "neobert.pretraining.trainer",
        "neobert.contrastive.trainer",
    ],
)
def test_prune_step_checkpoints_keeps_latest_numeric_dirs(
    module_path: str,
    tmp_path: Path,
) -> None:
    """Prune should remove old numeric dirs and keep non-numeric entries."""
    module = importlib.import_module(module_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in (10, 20, 30):
        (checkpoint_dir / str(step)).mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "notes").mkdir(parents=True, exist_ok=True)

    module._prune_step_checkpoints(checkpoint_dir, retention_limit=2)

    assert not (checkpoint_dir / "10").exists()
    assert (checkpoint_dir / "20").exists()
    assert (checkpoint_dir / "30").exists()
    assert (checkpoint_dir / "notes").exists()


@pytest.mark.parametrize(
    "module_path",
    [
        "neobert.pretraining.trainer",
        "neobert.contrastive.trainer",
    ],
)
def test_prune_step_checkpoints_limit_one_keeps_latest_only(
    module_path: str,
    tmp_path: Path,
) -> None:
    """retention_limit=1 should keep exactly the newest numeric checkpoint."""
    module = importlib.import_module(module_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in (1, 2):
        (checkpoint_dir / str(step)).mkdir(parents=True, exist_ok=True)

    module._prune_step_checkpoints(checkpoint_dir, retention_limit=1)

    assert not (checkpoint_dir / "1").exists()
    assert (checkpoint_dir / "2").exists()
