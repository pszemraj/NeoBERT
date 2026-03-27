"""Shared pytest fixtures for the NeoBERT test suite."""

from pathlib import Path

import pytest

from tests.tokenizer_utils import build_wordlevel_tokenizer


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


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> str:
    """Return a writable temporary output directory as a string path."""
    return str(tmp_path)


@pytest.fixture
def make_wordlevel_tokenizer():
    """Build a tiny word-level fast tokenizer for unit tests."""
    return build_wordlevel_tokenizer
