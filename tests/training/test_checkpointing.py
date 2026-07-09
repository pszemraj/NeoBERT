"""Tests for safetensors checkpoint utilities."""

import builtins
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from neobert.checkpointing import (
    ACCELERATE_STATE_DIR,
    MODEL_WEIGHTS_NAME,
    load_deepspeed_fp32_state_dict,
    load_model_safetensors,
    load_step_checkpoint_state_dict,
    model_state_dict_for_safetensors,
    resolve_accelerate_state_dir,
    resolve_deepspeed_checkpoint_root_and_tag,
    resolve_step_checkpoint_dir,
    resolve_step_checkpoint_selector,
    save_accelerate_state,
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


def test_accelerate_resume_state_coexists_with_portable_weights(
    tmp_path: Path,
) -> None:
    """Tied-weight checkpoints must keep resume state loadable beside exports.

    The portable ``model.safetensors`` duplicates tied tensors for export/eval
    consumers, which safetensors' strict ``load_model`` rejects; regression
    coverage for the layout that separates Accelerate resume state from it.
    """
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
    from neobert.checkpointing import save_portable_checkpoint_weights

    AcceleratorState._reset_state(True)
    try:
        accelerator = Accelerator(cpu=True)
        model = accelerator.prepare(_make_small_lm())
        assert model.decoder.weight is model.model.encoder.weight

        step_dir = tmp_path / "100"
        state_dir = save_accelerate_state(accelerator, step_dir)
        assert state_dir == step_dir / ACCELERATE_STATE_DIR
        assert save_portable_checkpoint_weights(model, accelerator, step_dir)

        portable_keys = load_model_safetensors(step_dir, map_location="cpu").keys()
        assert "model.encoder.weight" in portable_keys
        assert "decoder.weight" in portable_keys

        # Must not raise: resume state and portable payload live side by side.
        accelerator.load_state(str(resolve_accelerate_state_dir(step_dir)))
    finally:
        AcceleratorState._reset_state(True)


def test_resolve_accelerate_state_dir_requires_canonical_layout(
    tmp_path: Path,
) -> None:
    """Resume state must use the canonical accelerate/ subdirectory."""
    with pytest.raises(FileNotFoundError, match="no Accelerate state directory"):
        resolve_accelerate_state_dir(tmp_path)
    (tmp_path / ACCELERATE_STATE_DIR).mkdir()
    assert resolve_accelerate_state_dir(tmp_path) == tmp_path / ACCELERATE_STATE_DIR


def test_save_state_dict_safetensors_roundtrip(tmp_path: Path) -> None:
    """Ensure raw state_dict payloads are serializable via safetensors helper."""
    model = _make_small_lm()
    raw_state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}

    path = save_state_dict_safetensors(raw_state, tmp_path)
    assert path.name == MODEL_WEIGHTS_NAME
    assert path.exists()

    loaded_state = load_model_safetensors(tmp_path, map_location="cpu")

    assert "model.encoder.weight" in loaded_state
    assert all(not key.startswith("_orig_mod.") for key in loaded_state)


@pytest.mark.parametrize(
    ("weight_key", "bias_key"),
    [
        ("_orig_mod.weight", "module.bias"),
        ("module._orig_mod.weight", "_orig_mod.module.bias"),
    ],
    ids=["single-prefix", "stacked-prefixes"],
)
def test_load_model_safetensors_strips_runtime_prefixes_on_read(
    tmp_path: Path,
    weight_key: str,
    bias_key: str,
) -> None:
    """Loading should repeatedly canonicalize wrapper-prefixed keys."""
    weight = torch.arange(6, dtype=torch.float32).view(3, 2)
    bias = torch.arange(3, dtype=torch.float32)

    save_file(
        {weight_key: weight, bias_key: bias},
        str(tmp_path / MODEL_WEIGHTS_NAME),
        metadata={"format": "pt"},
    )
    loaded_state = load_model_safetensors(tmp_path, map_location="cpu")

    assert set(loaded_state) == {"weight", "bias"}
    torch.testing.assert_close(loaded_state["weight"], weight)
    torch.testing.assert_close(loaded_state["bias"], bias)


def test_load_model_safetensors_rejects_normalized_key_collisions(
    tmp_path: Path,
) -> None:
    """Loading should fail fast when multiple keys collapse to one parameter name."""
    save_file(
        {
            "weight": torch.ones(2, 2),
            "_orig_mod.weight": torch.zeros(2, 2),
        },
        str(tmp_path / MODEL_WEIGHTS_NAME),
        metadata={"format": "pt"},
    )

    with pytest.raises(ValueError, match="normalize to 'weight'"):
        load_model_safetensors(tmp_path, map_location="cpu")


def test_save_state_dict_safetensors_rejects_normalized_key_collisions(
    tmp_path: Path,
) -> None:
    """Saving should fail fast when canonicalization would overwrite a key."""
    with pytest.raises(ValueError, match="normalize to 'weight'"):
        save_state_dict_safetensors(
            {
                "weight": torch.ones(2, 2),
                "module._orig_mod.weight": torch.zeros(2, 2),
            },
            tmp_path,
        )


def test_resolve_deepspeed_checkpoint_root_and_tag_for_direct_tag_dir(
    tmp_path: Path,
) -> None:
    """Ensure direct DeepSpeed tag directories resolve to (parent, tag)."""
    tag_dir = tmp_path / "1234"
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "mp_rank_00_model_states.pt").touch()

    resolved_root, resolved_tag = resolve_deepspeed_checkpoint_root_and_tag(tag_dir)

    assert resolved_root == tmp_path
    assert resolved_tag == "1234"


def test_resolve_deepspeed_checkpoint_root_and_tag_for_nested_accelerate_layout(
    tmp_path: Path,
) -> None:
    """Ensure nested ``<step>/pytorch_model`` layouts resolve correctly."""
    checkpoints_root = tmp_path / "checkpoints"
    nested_tag_dir = checkpoints_root / "1000" / "pytorch_model"
    nested_tag_dir.mkdir(parents=True, exist_ok=True)
    (nested_tag_dir / "mp_rank_00_model_states.pt").touch()

    resolved_root, resolved_tag = resolve_deepspeed_checkpoint_root_and_tag(
        checkpoints_root,
        tag="1000",
    )

    assert resolved_root == checkpoints_root / "1000"
    assert resolved_tag == "pytorch_model"


def test_resolve_step_checkpoint_selector_prefers_latest_file(tmp_path: Path) -> None:
    """``latest`` metadata should beat root and numbered checkpoint payloads."""
    (tmp_path / "latest").write_text("456\n", encoding="utf-8")
    (tmp_path / MODEL_WEIGHTS_NAME).touch()
    portable_step = tmp_path / "999"
    portable_step.mkdir(parents=True, exist_ok=True)
    (portable_step / MODEL_WEIGHTS_NAME).touch()

    resolved = resolve_step_checkpoint_selector(tmp_path, "latest")

    assert resolved == "456"


def test_resolve_step_checkpoint_selector_picks_highest_loadable_numbered_step(
    tmp_path: Path,
) -> None:
    """Portable numbered steps should back ``latest`` when metadata is absent."""
    (tmp_path / "100").mkdir(parents=True, exist_ok=True)
    step_dir = tmp_path / "300"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / MODEL_WEIGHTS_NAME).touch()
    (tmp_path / "500").mkdir(parents=True, exist_ok=True)

    resolved = resolve_step_checkpoint_selector(tmp_path, "latest")

    assert resolved == "300"


def test_resolve_step_checkpoint_selector_keeps_zero_padded_step_names(
    tmp_path: Path,
) -> None:
    """Zero-padded step directories must resolve verbatim, not via int round-trip."""
    step_dir = tmp_path / "00050"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / MODEL_WEIGHTS_NAME).touch()

    resolved = resolve_step_checkpoint_selector(tmp_path, "latest")

    assert resolved == "00050"


def test_resolve_step_checkpoint_selector_accepts_direct_step_dir_for_latest(
    tmp_path: Path,
) -> None:
    """Direct step paths should resolve ``latest`` to their own step tag."""
    checkpoint_dir = tmp_path / "123"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / MODEL_WEIGHTS_NAME).touch()

    resolved = resolve_step_checkpoint_selector(checkpoint_dir, "latest")

    assert resolved == "123"


def test_resolve_step_checkpoint_selector_rejects_missing_latest(
    tmp_path: Path,
) -> None:
    """A missing latest checkpoint should fail during selection, not loading."""
    (tmp_path / "100").mkdir()

    with pytest.raises(FileNotFoundError, match="No loadable numbered checkpoints"):
        resolve_step_checkpoint_selector(tmp_path, "latest")


def test_resolve_step_checkpoint_dir_rejects_mismatched_direct_portable_weights(
    tmp_path: Path,
) -> None:
    """Direct step roots must not ignore an explicit mismatched checkpoint tag."""
    checkpoint_dir = tmp_path / "123"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / MODEL_WEIGHTS_NAME).touch()

    with pytest.raises(FileNotFoundError, match="Requested checkpoint '456'"):
        resolve_step_checkpoint_dir(checkpoint_dir, "456")


def test_load_step_checkpoint_state_dict_prefers_portable_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable safetensors should be loaded before any legacy fallback."""
    checkpoint_dir = tmp_path / "123"
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

    state_dict = load_step_checkpoint_state_dict(tmp_path, "123", map_location="cpu")

    assert state_dict == expected
    assert calls == {"portable": 1, "legacy": 0}


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_load_step_checkpoint_state_dict_falls_back_for_direct_step_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
) -> None:
    """Direct DeepSpeed step layouts should use the tag-less fallback."""
    checkpoint_dir = tmp_path / "456"
    if nested:
        checkpoint_dir /= "pytorch_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if nested:
        (checkpoint_dir / "mp_rank_00_model_states.pt").touch()
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


def test_load_step_checkpoint_state_dict_accepts_latest_for_direct_step_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct step paths should load cleanly when callers request ``latest``."""
    checkpoint_dir = tmp_path / "789"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / MODEL_WEIGHTS_NAME).touch()
    expected = {"weight": torch.full((2, 2), 7.0)}

    def _fake_load_model_safetensors(path: Path, *, map_location: str = "cpu"):
        del map_location
        assert path == checkpoint_dir
        return expected

    monkeypatch.setattr(
        "neobert.checkpointing.load_model_safetensors",
        _fake_load_model_safetensors,
    )

    state_dict = load_step_checkpoint_state_dict(
        checkpoint_dir,
        "latest",
        map_location="cpu",
    )

    assert state_dict == expected


@pytest.mark.parametrize(
    ("root_parts", "checkpoint"),
    [((), "1000"), (("123", "checkpoints"), "123")],
    ids=["ordinary-root", "numeric-parent"],
)
def test_load_step_checkpoint_state_dict_does_not_ignore_explicit_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_parts: tuple[str, ...],
    checkpoint: str,
) -> None:
    """Missing explicit tags must never trigger the tag-less fallback."""
    checkpoint_root = tmp_path.joinpath(*root_parts)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    seen: list[tuple[Path, str]] = []

    def _fake_load_deepspeed(path: Path, *, tag: str | None = None):
        normalized_path = Path(path).resolve()
        normalized_tag = "" if tag is None else str(tag)
        seen.append((normalized_path, normalized_tag))
        if tag is None:
            raise AssertionError("tag-less fallback should not run for missing tags")
        raise FileNotFoundError("requested checkpoint missing")

    monkeypatch.setattr(
        "neobert.checkpointing.load_deepspeed_fp32_state_dict",
        _fake_load_deepspeed,
    )

    with pytest.raises(FileNotFoundError, match="requested checkpoint missing"):
        load_step_checkpoint_state_dict(checkpoint_root, checkpoint)

    assert seen == [(checkpoint_root.resolve(), checkpoint)]


def test_load_deepspeed_fp32_state_dict_requires_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional DeepSpeed dependency should produce a clear install hint."""
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deepspeed.utils.zero_to_fp32":
            raise ModuleNotFoundError("simulated missing deepspeed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    tag_dir = tmp_path / "123"
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "mp_rank_00_model_states.pt").touch()

    with pytest.raises(ModuleNotFoundError, match="legacy-checkpoints"):
        load_deepspeed_fp32_state_dict(tmp_path, tag="123")
