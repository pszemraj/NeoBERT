"""Unit tests for MTEB task selection resolution."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import warnings

import pytest
import torch

_RUN_MTEB_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "run_mteb.py"
)
_RUN_MTEB_SPEC = importlib.util.spec_from_file_location(
    "neobert_run_mteb_script",
    _RUN_MTEB_PATH,
)
if _RUN_MTEB_SPEC is None or _RUN_MTEB_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_RUN_MTEB_PATH}")
_RUN_MTEB_MODULE = importlib.util.module_from_spec(_RUN_MTEB_SPEC)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _RUN_MTEB_SPEC.loader.exec_module(_RUN_MTEB_MODULE)

TASK_LIST = _RUN_MTEB_MODULE.TASK_LIST
TASK_TYPE = _RUN_MTEB_MODULE.TASK_TYPE
_resolve_mteb_tasks = _RUN_MTEB_MODULE._resolve_mteb_tasks


def _make_mteb_cfg(tmp_path: Path, *, use_deepspeed: bool) -> SimpleNamespace:
    """Build a minimal config object for ``evaluate_mteb`` tests."""
    return SimpleNamespace(
        mteb_batch_size=2,
        mteb_pooling="mean",
        mteb_overwrite_results=False,
        pretrained_checkpoint="100",
        trainer=SimpleNamespace(output_dir=str(tmp_path)),
        tokenizer=SimpleNamespace(name="bert-base-uncased", max_length=16),
        model=SimpleNamespace(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
            vocab_size=32,
            rope=True,
            rms_norm=True,
            hidden_act="gelu",
            dropout_prob=0.0,
            norm_eps=1e-5,
            embedding_init_range=0.02,
            decoder_init_range=0.02,
            classifier_init_range=0.02,
        ),
        use_deepspeed=use_deepspeed,
        task_types=["sts"],
    )


def test_resolve_mteb_tasks_uses_config_task_type_by_default():
    """Ensure config mteb_task_type is used when --task_types is unset."""
    cfg = SimpleNamespace(mteb_task_type="sts", task_types=None)

    selected = _resolve_mteb_tasks(cfg)

    assert selected == TASK_TYPE["sts"]


def test_resolve_mteb_tasks_accepts_category_overrides():
    """Ensure --task_types categories expand and preserve order."""
    cfg = SimpleNamespace(task_types=["classification", "sts"])

    selected = _resolve_mteb_tasks(cfg)

    assert selected[: len(TASK_TYPE["classification"])] == TASK_TYPE["classification"]
    assert all(task in selected for task in TASK_TYPE["sts"])


def test_resolve_mteb_tasks_accepts_explicit_task_names():
    """Ensure explicit task names can be mixed with category tokens."""
    cfg = SimpleNamespace(task_types=["MSMARCO", "sts"])

    selected = _resolve_mteb_tasks(cfg)

    assert "MSMARCO" in selected
    assert all(task in selected for task in TASK_TYPE["sts"])


def test_resolve_mteb_tasks_supports_all_token():
    """Ensure --task_types=all expands to the full task list."""
    cfg = SimpleNamespace(task_types=["all"])

    selected = _resolve_mteb_tasks(cfg)

    assert selected == TASK_LIST


def test_resolve_mteb_tasks_rejects_unknown_tokens():
    """Ensure unknown task/category names fail fast."""
    cfg = SimpleNamespace(task_types=["classification", "not_a_real_task"])

    with pytest.raises(ValueError):
        _resolve_mteb_tasks(cfg)


@pytest.mark.parametrize("task_types", ["", ",", ["", "   "]])
def test_resolve_mteb_tasks_rejects_empty_selection(task_types):
    """Ensure blank task selections fail fast instead of silently doing nothing."""
    cfg = SimpleNamespace(task_types=task_types)

    with pytest.raises(ValueError, match="No MTEB tasks selected"):
        _resolve_mteb_tasks(cfg)


def test_evaluate_mteb_falls_back_to_deepspeed_when_safetensors_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Ensure shard conversion is used when portable weights are absent."""
    (tmp_path / "checkpoints" / "100").mkdir(parents=True, exist_ok=True)
    cfg = _make_mteb_cfg(tmp_path, use_deepspeed=False)

    calls = {"safetensors": 0, "deepspeed": 0}
    strict_flags: list[bool] = []

    class _DummyModel:
        def load_state_dict(self, _state_dict, strict: bool = True):
            strict_flags.append(strict)
            return None

        def to(self, _device: str):
            return self

        def eval(self):
            return self

    class _DummyMTEB:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def run(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "get_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "NeoBERTForMTEB",
        lambda *args, **kwargs: _DummyModel(),
    )
    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "MTEB",
        _DummyMTEB,
    )
    monkeypatch.setattr(
        _RUN_MTEB_MODULE.torch,
        "autocast",
        lambda *args, **kwargs: nullcontext(),
    )

    def _unexpected_safetensors(*args, **kwargs):
        del args, kwargs
        calls["safetensors"] += 1
        raise AssertionError("load_model_safetensors should not be called in fallback")

    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "load_model_safetensors",
        _unexpected_safetensors,
    )

    def _fake_load_deepspeed(*args, **kwargs):
        del args, kwargs
        calls["deepspeed"] += 1
        return {"model.embeddings.weight": torch.zeros(1)}

    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "load_deepspeed_fp32_state_dict",
        _fake_load_deepspeed,
    )

    _RUN_MTEB_MODULE.evaluate_mteb(cfg)

    assert calls["safetensors"] == 0
    assert calls["deepspeed"] == 1
    assert strict_flags == [False]


def test_evaluate_mteb_prefers_safetensors_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Ensure portable weights remain the preferred loading path."""
    checkpoint_dir = tmp_path / "checkpoints" / "100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / _RUN_MTEB_MODULE.MODEL_WEIGHTS_NAME).touch()
    cfg = _make_mteb_cfg(tmp_path, use_deepspeed=False)

    calls = {"safetensors": 0, "deepspeed": 0}
    strict_flags: list[bool] = []

    class _DummyModel:
        def load_state_dict(self, _state_dict, strict: bool = True):
            strict_flags.append(strict)
            return None

        def to(self, _device: str):
            return self

        def eval(self):
            return self

    class _DummyMTEB:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def run(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "get_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "NeoBERTForMTEB",
        lambda *args, **kwargs: _DummyModel(),
    )
    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "MTEB",
        _DummyMTEB,
    )
    monkeypatch.setattr(
        _RUN_MTEB_MODULE.torch,
        "autocast",
        lambda *args, **kwargs: nullcontext(),
    )

    def _fake_load_safetensors(*args, **kwargs):
        del args, kwargs
        calls["safetensors"] += 1
        return {
            "model.embeddings.weight": torch.zeros(1),
            "decoder.weight": torch.zeros(1),
        }

    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "load_model_safetensors",
        _fake_load_safetensors,
    )

    def _fake_load_deepspeed(*args, **kwargs):
        del args, kwargs
        calls["deepspeed"] += 1
        return {"model.embeddings.weight": torch.zeros(1)}

    monkeypatch.setattr(
        _RUN_MTEB_MODULE,
        "load_deepspeed_fp32_state_dict",
        _fake_load_deepspeed,
    )

    _RUN_MTEB_MODULE.evaluate_mteb(cfg)

    assert calls["safetensors"] == 1
    assert calls["deepspeed"] == 0
    assert strict_flags == [False]


def test_load_mteb_encoder_weights_raises_on_non_head_mismatch():
    """Ensure non-head missing/unexpected keys fail fast for MTEB loads."""

    class _DummyModel:
        @staticmethod
        def load_state_dict(_state_dict, strict: bool = True):
            assert strict is False
            return SimpleNamespace(
                unexpected_keys=["decoder.weight", "encoder.bad_prefix.weight"],
                missing_keys=["model.encoder.weight"],
            )

    with pytest.raises(ValueError, match="checkpoint/model mismatch"):
        _RUN_MTEB_MODULE._load_mteb_encoder_weights(
            _DummyModel(),  # type: ignore[arg-type]
            {"model.embeddings.weight": torch.zeros(1)},
            source="unit-test",
        )
