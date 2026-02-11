"""Tests for safetensors checkpoint utilities."""

import tempfile
from pathlib import Path

import torch
from torch import nn

from neobert.checkpointing import (
    MODEL_WEIGHTS_NAME,
    load_model_safetensors,
    model_state_dict_for_safetensors,
    resolve_deepspeed_checkpoint_root_and_tag,
    save_model_safetensors,
    save_state_dict_safetensors,
)
from neobert.model import NeoBERTConfig, NeoBERTLMHead


class _CompiledLikeWrapper(nn.Module):
    """Minimal wrapper that mimics torch.compile's ``_orig_mod`` behavior."""

    def __init__(self, module: nn.Module) -> None:
        """Store wrapped module on ``_orig_mod``.

        :param nn.Module module: Model to wrap.
        """
        super().__init__()
        self._orig_mod = module


def _make_small_lm() -> NeoBERTLMHead:
    """Construct a tiny LM head model for checkpoint tests.

    :return NeoBERTLMHead: Tiny language-model head.
    """
    config = NeoBERTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_length=16,
        attn_backend="sdpa",
        hidden_act="gelu",
        rope=False,
        rms_norm=True,
    )
    return NeoBERTLMHead(config)


def test_model_state_dict_for_safetensors_strips_compile_prefixes() -> None:
    """Ensure state dict keys are canonicalized for compiled models."""
    model = _make_small_lm()
    wrapped = _CompiledLikeWrapper(model)
    payload = model_state_dict_for_safetensors(wrapped)

    assert "model.encoder.weight" in payload
    assert "decoder.weight" in payload
    assert all(not key.startswith("_orig_mod.") for key in payload)


def test_save_and_load_model_safetensors_roundtrip() -> None:
    """Ensure safetensors checkpoints load back into the same model class."""
    model = _make_small_lm()
    reference_state = model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        path = save_model_safetensors(model, checkpoint_dir)
        assert path.name == MODEL_WEIGHTS_NAME
        assert path.exists()

        loaded_state = load_model_safetensors(checkpoint_dir, map_location="cpu")
        restored = _make_small_lm()
        missing, unexpected = restored.load_state_dict(loaded_state, strict=True)

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        reference_state["model.encoder.weight"],
        restored.state_dict()["model.encoder.weight"],
    )


def test_save_state_dict_safetensors_roundtrip() -> None:
    """Ensure raw state_dict payloads are serializable via safetensors helper."""
    model = _make_small_lm()
    raw_state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        path = save_state_dict_safetensors(raw_state, checkpoint_dir)
        assert path.name == MODEL_WEIGHTS_NAME
        assert path.exists()

        loaded_state = load_model_safetensors(checkpoint_dir, map_location="cpu")

    assert "model.encoder.weight" in loaded_state
    assert all(not key.startswith("_orig_mod.") for key in loaded_state)


def test_resolve_deepspeed_checkpoint_root_and_tag_for_direct_tag_dir() -> None:
    """Ensure direct DeepSpeed tag directories resolve to (parent, tag)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tag_dir = root / "1234"
        tag_dir.mkdir(parents=True, exist_ok=True)
        (tag_dir / "mp_rank_00_model_states.pt").touch()

        resolved_root, resolved_tag = resolve_deepspeed_checkpoint_root_and_tag(tag_dir)

    assert resolved_root == root
    assert resolved_tag == "1234"


def test_resolve_deepspeed_checkpoint_root_and_tag_for_nested_accelerate_layout() -> (
    None
):
    """Ensure nested ``<step>/pytorch_model`` layouts resolve correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoints_root = Path(tmpdir) / "checkpoints"
        nested_tag_dir = checkpoints_root / "1000" / "pytorch_model"
        nested_tag_dir.mkdir(parents=True, exist_ok=True)
        (nested_tag_dir / "mp_rank_00_model_states.pt").touch()

        resolved_root, resolved_tag = resolve_deepspeed_checkpoint_root_and_tag(
            checkpoints_root,
            tag="1000",
        )

    assert resolved_root == checkpoints_root / "1000"
    assert resolved_tag == "pytorch_model"
