"""Shared pytest fixtures for the NeoBERT test suite."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


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

    def _make(
        vocab: dict[str, int] | None = None,
        *,
        padding_side: str = "right",
        include_mask: bool = True,
        include_sep: bool = True,
        include_cls: bool = False,
    ) -> PreTrainedTokenizerFast:
        merged = {"[PAD]": 0, "[UNK]": 1}
        if include_mask:
            merged["[MASK]"] = len(merged)
        if include_sep:
            merged["[SEP]"] = len(merged)
        if include_cls:
            merged["[CLS]"] = len(merged)
        merged.update(vocab or {"hello": len(merged), "world": len(merged) + 1})

        tokenizer = Tokenizer(models.WordLevel(merged, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        kwargs = {
            "tokenizer_object": tokenizer,
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
        }
        if include_mask:
            kwargs["mask_token"] = "[MASK]"
        if include_sep:
            kwargs["sep_token"] = "[SEP]"
        if include_cls:
            kwargs["cls_token"] = "[CLS]"

        fast = PreTrainedTokenizerFast(**kwargs)
        fast.padding_side = padding_side
        return fast

    return _make
