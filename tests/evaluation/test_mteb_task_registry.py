"""Tests for the shared MTEB task registry and aggregation semantics."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
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


def _result(score: float, *, split: str = "test") -> dict:
    """Build a minimal MTEB result payload.

    :param float score: Main score.
    :param str split: Evaluation split containing the score.
    :return dict: MTEB-shaped payload.
    """
    return {"scores": {split: [{"main_score": score}]}}


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
    msmarco = next(
        task
        for task in MTEB_TASK_GROUPS_BY_KEY["retrieval"].tasks
        if task.aggregation_name == "MSMARCO"
    )
    assert msmarco.evaluation_split == "dev"


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
    assert scores["STS"] is None
    assert scores["Avg."] == pytest.approx(100 * (expected_cqa + 0.8) / 2, abs=0.01)


def test_aggregation_uses_msmarco_dev_split() -> None:
    """MSMARCO contributes its configured development-set score."""
    avg_mteb = _load_script("avg_mteb")
    results = {"MSMARCO": _result(0.42, split="dev")}

    scores = avg_mteb._average_categories(results)

    assert scores["Retr."] == 42.0
    assert scores["Avg."] == 42.0


def test_incomplete_cqadupstack_is_not_averaged_as_a_complete_task() -> None:
    """A partial expanded task is reported missing rather than inflated."""
    avg_mteb = _load_script("avg_mteb")
    results = {
        name: _result(index / 100)
        for index, name in enumerate(CQADUPSTACK_TASKS[:-1], start=1)
    }
    results["SummEval"] = _result(0.8)

    scores = avg_mteb._average_categories(results)
    coverage = avg_mteb._result_coverage(results)

    assert scores["Retr."] is None
    assert scores["Avg."] == 80.0
    assert coverage["complete"] is False
    assert f"{CQADUPSTACK_TASKS[-1]}:test" in coverage["missing_results"]


def test_explicit_all_overrides_narrower_config_selection() -> None:
    """CLI omission and an explicit ``all`` selector remain distinguishable."""
    run_mteb = _load_script("run_mteb")
    config = SimpleNamespace(mteb_task_type="sts")

    config.task_types = run_mteb._parse_task_type_override(None)
    assert len(run_mteb._resolve_mteb_tasks(config)) == len(
        MTEB_TASK_GROUPS_BY_KEY["sts"].execution_names
    )

    config.task_types = run_mteb._parse_task_type_override("all")
    assert run_mteb._resolve_mteb_tasks(config) == list(MTEB_ALL_EXECUTION_TASKS)
