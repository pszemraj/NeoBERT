"""Aggregate MTEB results into averaged score tables."""

import argparse
import json
from pathlib import Path

from neobert.mteb_tasks import MTEB_TASK_GROUPS, MTEBTaskSpec


def _task_score(model_results: dict, task: MTEBTaskSpec) -> float | None:
    """Return one task score, averaging expanded dataset variants when needed.

    :param dict model_results: Result payloads keyed by concrete MTEB task name.
    :param MTEBTaskSpec task: Task specification to aggregate.
    :return float | None: Mean main score, or ``None`` when no result is present.
    """
    scores = []
    for result_name in task.execution_names:
        try:
            score = model_results[result_name]["scores"]["test"][0]["main_score"]
        except (KeyError, TypeError, IndexError):
            continue
        if score is not None:
            scores.append(float(score))
    return sum(scores) / len(scores) if scores else None


def _average_categories(model_results: dict) -> dict[str, float]:
    """Compute category and overall averages for one model.

    :param dict model_results: Result payloads keyed by concrete MTEB task name.
    :return dict[str, float]: Percentage scores keyed by reporting label.
    """
    category_scores: dict[str, float] = {}
    all_task_scores = []
    for group in MTEB_TASK_GROUPS:
        scores = [
            score
            for task in group.tasks
            if (score := _task_score(model_results, task)) is not None
        ]
        all_task_scores.extend(scores)
        category_scores[group.label] = (
            round(100 * sum(scores) / len(scores), 2) if scores else 0
        )
    category_scores["Avg."] = (
        round(100 * sum(all_task_scores) / len(all_task_scores), 2)
        if all_task_scores
        else 0
    )
    return category_scores


def compute_table() -> None:
    """Compute and write average MTEB score tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", dest="result_folder", type=str)
    parser.add_argument("--model_name", dest="model_name", type=str)
    args = parser.parse_args()

    all_results = {}

    result_dir = Path(args.result_folder)
    result_file = result_dir / f"{args.model_name}_avg_table.json"

    if result_file.exists():
        UserWarning("Overwriting existing result file.")
        result_file.unlink()

    def explore(path: Path) -> list[Path]:
        """Collect leaf directories containing result JSON files.

        :param Path path: Root path to search.
        :return list[Path]: Leaf directories with JSON files.
        """
        paths = []
        file_level = False
        files = list(path.iterdir())
        if not files:
            UserWarning(f"Empty folder path: {path}.")
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
                else:
                    with file_path.open("r", encoding="utf-8") as f:
                        results = json.load(f)
                        all_results[model_name] = {
                            **all_results[model_name],
                            **{file_path.stem: results},
                        }

    avg_results = {
        model: _average_categories(model_results)
        for model, model_results in all_results.items()
    }

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(avg_results, f, indent=2)


if __name__ == "__main__":
    compute_table()
