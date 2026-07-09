"""Tests for the shared MTEB task registry and aggregation semantics."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

from neobert.mteb_tasks import (
    CQADUPSTACK_TASKS,
    MTEB_ALL_EXECUTION_TASKS,
    MTEB_TASK_GROUPS_BY_KEY,
    expand_mteb_task_name,
)

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "evaluation"


def _load_script(name: str):
    """Load an evaluation script as an importable module.

    :param str name: Script filename without the Python suffix.
    :return Any: Loaded module.
    """
    spec = spec_from_file_location(f"neobert_test_{name}", SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(score: float) -> dict:
    """Build a minimal MTEB result payload.

    :param float score: Main score.
    :return dict: MTEB-shaped payload.
    """
    return {"scores": {"test": [{"main_score": score}]}}


def test_registry_separates_execution_names_and_reporting_categories() -> None:
    """CQADupstack expands for execution while SummEval remains summarization."""
    retrieval = MTEB_TASK_GROUPS_BY_KEY["retrieval"]
    cqa = next(
        task
        for task in retrieval.tasks
        if task.aggregation_name == "CQADupstackRetrieval"
    )

    assert cqa.execution_names == CQADUPSTACK_TASKS
    assert "CQADupstackRetrieval" not in MTEB_ALL_EXECUTION_TASKS
    assert set(CQADUPSTACK_TASKS).issubset(MTEB_ALL_EXECUTION_TASKS)
    assert "SummEval" not in MTEB_TASK_GROUPS_BY_KEY["sts"].execution_names
    assert MTEB_TASK_GROUPS_BY_KEY["summarization"].execution_names == ("SummEval",)


def test_task_selection_expands_cqadupstack_alias() -> None:
    """The runner should accept the reporting alias and execute every subset."""
    assert expand_mteb_task_name("CQADupstackRetrieval") == CQADUPSTACK_TASKS


def test_aggregation_weights_cqadupstack_as_one_task() -> None:
    """Expanded CQADupstack subsets should contribute one averaged task score."""
    avg_mteb = _load_script("avg_mteb")
    results = {
        name: _result(index / 100)
        for index, name in enumerate(CQADUPSTACK_TASKS, start=1)
    }
    results["SummEval"] = _result(0.8)

    scores = avg_mteb._average_categories(results)

    expected_cqa = sum(range(1, len(CQADUPSTACK_TASKS) + 1)) / (
        100 * len(CQADUPSTACK_TASKS)
    )
    assert scores["Retr."] == pytest.approx(100 * expected_cqa)
    assert scores["Summ."] == 80.0
    assert scores["STS"] == 0
    assert scores["Avg."] == pytest.approx(100 * (expected_cqa + 0.8) / 2, abs=0.01)
