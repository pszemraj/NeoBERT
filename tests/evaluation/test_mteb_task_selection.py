"""Unit tests for MTEB task selection resolution."""

import importlib.util
from types import SimpleNamespace
from pathlib import Path
import warnings

import pytest

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
