"""Shared checkpoint-retention policy tests for trainer implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from neobert.checkpointing import (
    prune_step_checkpoints,
    resolve_checkpoint_retention_limit,
)
from neobert.config import Config


def test_resolve_checkpoint_retention_limit_uses_save_total_limit() -> None:
    """Retention should use the canonical save_total_limit field."""
    cfg = Config()
    cfg.trainer.save_total_limit = 1
    assert resolve_checkpoint_retention_limit(cfg) == 1

    cfg.trainer.save_total_limit = None
    assert resolve_checkpoint_retention_limit(cfg) == 0


@pytest.mark.parametrize(
    ("steps", "retention_limit", "expected_kept"),
    [
        ((10, 20, 30), 2, {20, 30}),
        ((1, 2), 1, {2}),
    ],
)
def test_prune_step_checkpoints_keeps_latest_numeric_dirs(
    tmp_path: Path,
    steps: tuple[int, ...],
    retention_limit: int,
    expected_kept: set[int],
) -> None:
    """Prune should retain the newest numeric dirs and ignore other entries."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in steps:
        (checkpoint_dir / str(step)).mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "notes").mkdir(parents=True, exist_ok=True)

    prune_step_checkpoints(checkpoint_dir, retention_limit=retention_limit)

    for step in steps:
        assert (checkpoint_dir / str(step)).exists() is (step in expected_kept)
    assert (checkpoint_dir / "notes").exists()


def test_prune_step_checkpoints_breaks_numeric_ties_by_name(tmp_path: Path) -> None:
    """Retention must use the same deterministic alias tie-break as resume."""
    checkpoint_dir = tmp_path / "checkpoints"
    for tag in ("50", "050"):
        (checkpoint_dir / tag).mkdir(parents=True, exist_ok=True)

    prune_step_checkpoints(checkpoint_dir, retention_limit=1)

    assert (checkpoint_dir / "50").is_dir()
    assert not (checkpoint_dir / "050").exists()
