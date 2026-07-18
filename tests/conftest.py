"""Shared pytest fixtures for the NeoBERT test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def test_configs_dir() -> Path:
    """Return the root directory containing test YAML configs."""
    return Path(__file__).resolve().parent / "configs"


@pytest.fixture
def tiny_pretrain_config_path(test_configs_dir: Path) -> Path:
    """Return the tiny pretraining config path."""
    return test_configs_dir / "pretraining" / "test_tiny_pretrain.yaml"


@pytest.fixture
def tiny_glue_config_path(test_configs_dir: Path) -> Path:
    """Return the tiny GLUE config path."""
    return test_configs_dir / "evaluation" / "test_tiny_glue.yaml"


@pytest.fixture
def tiny_contrastive_config_path(test_configs_dir: Path) -> Path:
    """Return the tiny contrastive config path."""
    return test_configs_dir / "contrastive" / "test_tiny_contrastive.yaml"
