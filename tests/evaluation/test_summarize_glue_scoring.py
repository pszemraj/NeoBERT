from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_summarizer():
    summarizer_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "glue"
        / "summarize_glue.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_glue", summarizer_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.mark.parametrize(
    ("task", "files", "expected"),
    [
        (
            "mrpc",
            [
                ("all_results_step_10.json", {"eval_accuracy": 0.50, "eval_f1": 0.90}),
                ("all_results_step_20.json", {"eval_accuracy": 0.80, "eval_f1": 0.80}),
            ],
            0.80,
        ),
        (
            "qqp",
            [
                ("all_results_step_10.json", {"eval_accuracy": 0.40, "eval_f1": 0.95}),
                ("all_results_step_20.json", {"eval_accuracy": 0.85, "eval_f1": 0.85}),
            ],
            0.85,
        ),
        (
            "stsb",
            [
                (
                    "all_results_step_10.json",
                    {"eval_pearson": 0.90, "eval_spearmanr": 0.70},
                ),
                (
                    "all_results_step_20.json",
                    {"eval_pearson": 0.81, "eval_spearmanr": 0.81},
                ),
            ],
            0.81,
        ),
        (
            "mnli",
            [
                (
                    "all_results_step_10.json",
                    {"eval_accuracy": 0.90, "eval_accuracy_mm": 0.70},
                ),
                (
                    "all_results_step_20.json",
                    {"eval_accuracy": 0.82, "eval_accuracy_mm": 0.82},
                ),
            ],
            0.82,
        ),
    ],
)
def test_task_scoring_uses_official_aggregation(tmp_path: Path, task, files, expected):
    module = _load_summarizer()
    task_dir = tmp_path / task
    for filename, payload in files:
        _write_json(task_dir / filename, payload)

    score = module.get_task_results(task_dir, task)
    assert score == pytest.approx(expected)
