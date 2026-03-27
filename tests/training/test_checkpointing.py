"""Tests for safetensors checkpoint utilities."""

import builtins
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from neobert.checkpointing import (
    MODEL_WEIGHTS_NAME,
    load_deepspeed_fp32_state_dict,
    load_model_safetensors,
    load_step_checkpoint_state_dict,
    model_state_dict_for_safetensors,
    resolve_deepspeed_checkpoint_root_and_tag,
    resolve_step_checkpoint_dir,
    resolve_step_checkpoint_selector,
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


def test_load_model_safetensors_strips_runtime_prefixes_on_read() -> None:
    """Loading should canonicalize wrapper-prefixed keys from generic save paths."""
    weight = torch.arange(6, dtype=torch.float32).view(3, 2)
    bias = torch.arange(3, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        save_file(
            {
                "_orig_mod.weight": weight,
                "module.bias": bias,
            },
            str(checkpoint_dir / MODEL_WEIGHTS_NAME),
            metadata={"format": "pt"},
        )
        loaded_state = load_model_safetensors(checkpoint_dir, map_location="cpu")

    assert set(loaded_state) == {"weight", "bias"}
    torch.testing.assert_close(loaded_state["weight"], weight)
    torch.testing.assert_close(loaded_state["bias"], bias)


def test_load_model_safetensors_strips_stacked_runtime_prefixes_on_read() -> None:
    """Loading should strip stacked runtime prefixes until keys are stable."""
    weight = torch.arange(6, dtype=torch.float32).view(3, 2)
    bias = torch.arange(3, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        save_file(
            {
                "module._orig_mod.weight": weight,
                "_orig_mod.module.bias": bias,
            },
            str(checkpoint_dir / MODEL_WEIGHTS_NAME),
            metadata={"format": "pt"},
        )
        loaded_state = load_model_safetensors(checkpoint_dir, map_location="cpu")

    assert set(loaded_state) == {"weight", "bias"}
    torch.testing.assert_close(loaded_state["weight"], weight)
    torch.testing.assert_close(loaded_state["bias"], bias)


def test_load_model_safetensors_rejects_normalized_key_collisions() -> None:
    """Loading should fail fast when multiple keys collapse to one parameter name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        save_file(
            {
                "weight": torch.ones(2, 2),
                "_orig_mod.weight": torch.zeros(2, 2),
            },
            str(checkpoint_dir / MODEL_WEIGHTS_NAME),
            metadata={"format": "pt"},
        )

        with pytest.raises(ValueError, match="normalize to 'weight'"):
            load_model_safetensors(checkpoint_dir, map_location="cpu")


def test_save_state_dict_safetensors_rejects_normalized_key_collisions() -> None:
    """Saving should fail fast when canonicalization would overwrite a key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        with pytest.raises(ValueError, match="normalize to 'weight'"):
            save_state_dict_safetensors(
                {
                    "weight": torch.ones(2, 2),
                    "module._orig_mod.weight": torch.zeros(2, 2),
                },
                checkpoint_dir,
            )


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


def test_resolve_step_checkpoint_selector_prefers_latest_file() -> None:
    """DeepSpeed ``latest`` metadata should stay authoritative when present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_root = Path(tmpdir)
        (checkpoint_root / "latest").write_text("456\n", encoding="utf-8")
        portable_step = checkpoint_root / "999"
        portable_step.mkdir(parents=True, exist_ok=True)
        (portable_step / MODEL_WEIGHTS_NAME).touch()

        resolved = resolve_step_checkpoint_selector(checkpoint_root, "latest")

    assert resolved == "456"


def test_resolve_step_checkpoint_selector_picks_highest_loadable_numbered_step() -> (
    None
):
    """Portable numbered steps should back ``latest`` when metadata is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_root = Path(tmpdir)
        (checkpoint_root / "100").mkdir(parents=True, exist_ok=True)
        step_dir = checkpoint_root / "300"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / MODEL_WEIGHTS_NAME).touch()
        (checkpoint_root / "500").mkdir(parents=True, exist_ok=True)

        resolved = resolve_step_checkpoint_selector(checkpoint_root, "latest")

    assert resolved == "300"


def test_resolve_step_checkpoint_dir_rejects_mismatched_direct_portable_weights() -> (
    None
):
    """Direct step roots must not ignore an explicit mismatched checkpoint tag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "123"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / MODEL_WEIGHTS_NAME).touch()

        with pytest.raises(FileNotFoundError, match="Requested checkpoint '456'"):
            resolve_step_checkpoint_dir(checkpoint_dir, "456")


def test_load_step_checkpoint_state_dict_prefers_portable_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable safetensors should be loaded before any legacy fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "123"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / MODEL_WEIGHTS_NAME).touch()
        expected = {"weight": torch.ones(2, 2)}
        calls = {"portable": 0, "legacy": 0}

        def _fake_load_model_safetensors(path: Path, *, map_location: str = "cpu"):
            del map_location
            calls["portable"] += 1
            assert path == checkpoint_dir
            return expected

        def _fake_load_deepspeed(*args, **kwargs):
            del args, kwargs
            calls["legacy"] += 1
            raise AssertionError("DeepSpeed fallback should not run")

        monkeypatch.setattr(
            "neobert.checkpointing.load_model_safetensors",
            _fake_load_model_safetensors,
        )
        monkeypatch.setattr(
            "neobert.checkpointing.load_deepspeed_fp32_state_dict",
            _fake_load_deepspeed,
        )

        state_dict = load_step_checkpoint_state_dict(tmpdir, "123", map_location="cpu")

    assert state_dict == expected
    assert calls == {"portable": 1, "legacy": 0}


def test_load_step_checkpoint_state_dict_falls_back_for_direct_step_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct step directories should still use the tag-less DeepSpeed fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "456"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        expected = {"weight": torch.zeros(2, 2)}
        seen: list[tuple[Path, str]] = []

        def _fake_load_deepspeed(path: Path, *, tag: str | None = None):
            normalized_path = Path(path).resolve()
            normalized_tag = "" if tag is None else str(tag)
            seen.append((normalized_path, normalized_tag))
            if normalized_tag == "456":
                raise FileNotFoundError(
                    "explicit root/tag lookup should miss direct step dirs"
                )
            return expected

        monkeypatch.setattr(
            "neobert.checkpointing.load_deepspeed_fp32_state_dict",
            _fake_load_deepspeed,
        )

        state_dict = load_step_checkpoint_state_dict(checkpoint_dir, "456")

    assert state_dict == expected
    assert seen == [(checkpoint_dir.resolve(), "456"), (checkpoint_dir.resolve(), "")]


def test_load_step_checkpoint_state_dict_does_not_ignore_explicit_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing explicit checkpoint tags must not silently fall back to latest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_root = Path(tmpdir)
        seen: list[tuple[Path, str]] = []

        def _fake_load_deepspeed(path: Path, *, tag: str | None = None):
            normalized_path = Path(path).resolve()
            normalized_tag = "" if tag is None else str(tag)
            seen.append((normalized_path, normalized_tag))
            if tag is None:
                raise AssertionError(
                    "tag-less fallback should not run for missing tags"
                )
            raise FileNotFoundError("requested checkpoint missing")

        monkeypatch.setattr(
            "neobert.checkpointing.load_deepspeed_fp32_state_dict",
            _fake_load_deepspeed,
        )

        with pytest.raises(FileNotFoundError, match="requested checkpoint missing"):
            load_step_checkpoint_state_dict(checkpoint_root, "1000")

    assert seen == [(checkpoint_root.resolve(), "1000")]


def test_load_deepspeed_fp32_state_dict_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional DeepSpeed dependency should produce a clear install hint."""
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deepspeed.utils.zero_to_fp32":
            raise ModuleNotFoundError("simulated missing deepspeed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tag_dir = root / "123"
        tag_dir.mkdir(parents=True, exist_ok=True)
        (tag_dir / "mp_rank_00_model_states.pt").touch()

        with pytest.raises(ModuleNotFoundError, match="legacy-checkpoints"):
            load_deepspeed_fp32_state_dict(root, tag="123")
