"""Aggregate MTEB results into averaged score tables."""

import argparse
import json
from pathlib import Path

from neobert.mteb_tasks import (
    MTEB_ALL_EXECUTION_TASKS,
    MTEB_TASK_GROUPS,
    MTEB_TASK_SPECS_BY_EXECUTION_NAME,
    MTEBTaskSpec,
)


def _execution_score(
    model_results: dict,
    result_name: str,
    split: str,
) -> float | None:
    """Return one concrete task score for its configured evaluation split.

    :param dict model_results: Result payloads keyed by concrete task name.
    :param str result_name: Concrete MTEB task name.
    :param str split: Evaluation split containing the official score.
    :return float | None: Main score, or ``None`` when missing.
    """
    try:
        score = model_results[result_name]["scores"][split][0]["main_score"]
    except (KeyError, TypeError, IndexError):
        return None
    return float(score) if score is not None else None


def _task_score(model_results: dict, task: MTEBTaskSpec) -> float | None:
    """Return one task score, averaging expanded dataset variants when needed.

    :param dict model_results: Result payloads keyed by concrete MTEB task name.
    :param MTEBTaskSpec task: Task specification to aggregate.
    :return float | None: Mean main score, or ``None`` when no result is present.
    """
    scores: list[float] = []
    for result_name in task.execution_names:
        score = _execution_score(
            model_results,
            result_name,
            task.evaluation_split,
        )
        if score is None:
            return None
        scores.append(score)
    return sum(scores) / len(scores)


def _average_categories(model_results: dict) -> dict[str, float | None]:
    """Compute category and overall averages for one model.

    :param dict model_results: Result payloads keyed by concrete MTEB task name.
    :return dict[str, float | None]: Percentage scores keyed by reporting label.
    """
    category_scores: dict[str, float | None] = {}
    all_task_scores = []
    for group in MTEB_TASK_GROUPS:
        scores = [
            score
            for task in group.tasks
            if (score := _task_score(model_results, task)) is not None
        ]
        all_task_scores.extend(scores)
        category_scores[group.label] = (
            round(100 * sum(scores) / len(scores), 2) if scores else None
        )
    category_scores["Avg."] = (
        round(100 * sum(all_task_scores) / len(all_task_scores), 2)
        if all_task_scores
        else None
    )
    return category_scores


def _result_coverage(model_results: dict) -> dict[str, object]:
    """Summarize concrete and logical MTEB result coverage.

    :param dict model_results: Result payloads keyed by concrete task name.
    :return dict[str, object]: Coverage counts, completeness, and missing results.
    """
    missing_results = [
        f"{name}:{MTEB_TASK_SPECS_BY_EXECUTION_NAME[name].evaluation_split}"
        for name in MTEB_ALL_EXECUTION_TASKS
        if _execution_score(
            model_results,
            name,
            MTEB_TASK_SPECS_BY_EXECUTION_NAME[name].evaluation_split,
        )
        is None
    ]
    logical_tasks = [task for group in MTEB_TASK_GROUPS for task in group.tasks]
    logical_present = sum(
        _task_score(model_results, task) is not None for task in logical_tasks
    )
    return {
        "complete": not missing_results,
        "concrete_present": len(MTEB_ALL_EXECUTION_TASKS) - len(missing_results),
        "concrete_expected": len(MTEB_ALL_EXECUTION_TASKS),
        "logical_present": logical_present,
        "logical_expected": len(logical_tasks),
        "missing_results": missing_results,
    }


def compute_table() -> None:
    """Compute and write average MTEB score tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", required=True, type=Path)
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write explicitly marked partial aggregates instead of failing",
    )
    args = parser.parse_args()

    all_results = {}

    result_dir = args.result_folder
    result_file = result_dir / f"{args.model_name}_avg_table.json"

    if result_file.exists():
        result_file.unlink()

    def explore(path: Path) -> list[Path]:
        """Collect leaf directories containing result JSON files.

        :param Path path: Root path to search.
        :return list[Path]: Leaf directories with JSON files.
        """
        paths = []
        file_level = False
        files = list(path.iterdir())
        for file in files:
            if file.is_dir():
                paths.extend(explore(file))
            else:
                file_level = True
        if file_level:
            paths.append(path)
        return paths

    for checkpoint in result_dir.iterdir():
        paths = explore(checkpoint)
        checkpoint_name = checkpoint.name

        for path in paths:
            path_str = path.as_posix()
            i = path_str.find(checkpoint_name) + len(checkpoint_name)
            j = path_str.find("no_model_name_available")
            model_name = f"{args.model_name}_{checkpoint_name}_{path_str[i + 1 : j - 1] if j != -1 else path_str[i + 1 :]}"

            all_results.setdefault(model_name, {})

            for file_path in path.iterdir():
                if not file_path.name.endswith(".json"):
                    print(f"Skipping non-json {file_path.name}")
                    continue
                with file_path.open("r", encoding="utf-8") as f:
                    results = json.load(f)
                    all_results[model_name][file_path.stem] = results

    avg_results = {}
    for model, model_results in all_results.items():
        coverage = _result_coverage(model_results)
        if not coverage["complete"] and not args.allow_partial:
            missing = ", ".join(coverage["missing_results"])
            raise RuntimeError(
                f"MTEB results for {model} are incomplete: {missing}. "
                "Rerun missing tasks or pass --allow-partial."
            )
        avg_results[model] = {
            "scores": _average_categories(model_results),
            "coverage": coverage,
        }

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(avg_results, f, indent=2)


if __name__ == "__main__":
    compute_table()
