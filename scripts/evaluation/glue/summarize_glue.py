#!/usr/bin/env python3
"""Summarize GLUE results from output directories.

Usage:
    python scripts/evaluation/glue/summarize_glue.py outputs/glue/neobert-100m
    python scripts/evaluation/glue/summarize_glue.py ./outputs/experiment_xyz/glue_results
    python scripts/evaluation/glue/summarize_glue.py outputs/glue/neobert-100m --baseline roberta-base
    python scripts/evaluation/glue/summarize_glue.py  # Default: outputs/glue/neobert-100m
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Official GLUE task scoring definitions.
# For tasks with multiple metrics, GLUE uses the unweighted average.
TASK_SCORES = {
    "cola": {
        "metrics": ("matthews_correlation",),
        "label": "Matthews Corr",
        "scale": 100,
    },
    "sst2": {"metrics": ("accuracy",), "label": "Accuracy", "scale": 100},
    "mrpc": {"metrics": ("accuracy", "f1"), "label": "Acc/F1 (avg)", "scale": 100},
    "stsb": {
        "metrics": ("pearson", "spearmanr"),
        "label": "Pearson/Spearman (avg)",
        "scale": 100,
    },
    "qqp": {"metrics": ("accuracy", "f1"), "label": "Acc/F1 (avg)", "scale": 100},
    "mnli": {
        "metrics": ("accuracy", "accuracy_mm"),
        "label": "MNLI-m/mm (avg)",
        "scale": 100,
    },
    "qnli": {"metrics": ("accuracy",), "label": "Accuracy", "scale": 100},
    "rte": {"metrics": ("accuracy",), "label": "Accuracy", "scale": 100},
    "wnli": {"metrics": ("accuracy",), "label": "Accuracy", "scale": 100},
}

# Baseline scores for reference (BERT-base scores from literature)
BERT_BASE_SCORES = {
    "cola": 52.1,
    "sst2": 93.5,
    "mrpc": 86.9,  # Avg(F1=88.9, Acc=84.8)
    "stsb": 85.4,  # Avg(Pearson=85.8, Spearman=84.9)
    "qqp": 80.2,  # Avg(F1=71.2, Acc=89.2)
    "mnli": 84.0,  # Avg(m=84.6, mm=83.4)
    "qnli": 90.5,
    "rte": 66.4,
    "wnli": 65.1,
}


def _normalize_metrics(data: dict) -> dict[str, float]:
    """Normalize metric keys for GLUE reporting.

    :param dict data: Raw metric mapping.
    :return dict[str, float]: Normalized metric mapping.
    """
    normalized: dict[str, float] = {}
    for key, value in (data or {}).items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            continue
        metric_key = key[len("eval_") :] if key.startswith("eval_") else key
        normalized[metric_key] = float(value)
    return normalized


def _compute_task_score(task: str, metrics: dict[str, float]) -> float | None:
    """Compute the official GLUE score for a task.

    :param str task: GLUE task name.
    :param dict[str, float] metrics: Metric mapping for the task.
    :return float | None: Task score or None if unavailable.
    """
    spec = TASK_SCORES.get(task)
    if spec is None:
        return None

    required = list(spec["metrics"])
    values: list[float] = []

    for metric in required:
        if metric in metrics:
            values.append(metrics[metric])
            continue
        if task == "mnli" and metric == "accuracy_mm":
            for alias in ("accuracy_mismatched", "mnli_mm", "accuracy-mm"):
                if alias in metrics:
                    values.append(metrics[alias])
                    break

    if not values:
        return None

    if len(values) != len(required):
        combined = metrics.get("combined_score")
        if combined is not None and isinstance(combined, (int, float)):
            return float(combined)

    return sum(values) / len(values)


def get_task_results(task_dir: Path, task: str) -> float | None:
    """Extract the best (max) official GLUE score from a task directory.

    :param Path task_dir: Directory containing evaluation results.
    :param str task: GLUE task name.
    :return float | None: Best score or None if missing.
    """
    results_files = list(task_dir.glob("all_results_step_*.json"))
    if not results_files:
        results_file = task_dir / "all_results.json"
        if results_file.exists():
            results_files = [results_file]
        else:
            return None

    best_score: float | None = None
    for f in results_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        if not isinstance(data, dict):
            continue

        metrics = _normalize_metrics(data)
        score = _compute_task_score(task, metrics)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score

    return best_score


def main() -> None:
    """Run the GLUE summary CLI."""
    parser = argparse.ArgumentParser(
        description="Parse and summarize GLUE evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        help="Path to GLUE results directory (e.g., outputs/glue/neobert-100m or ./outputs/experiment_xyz)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="bert-base",
        choices=["bert-base", "bert-large", "roberta-base", "roberta-large", "none"],
        help="Baseline model to compare against",
    )
    args = parser.parse_args()

    # Determine the base directory
    if args.path:
        base_dir = Path(args.path)
    else:
        # Default to outputs/glue/neobert-100m for backward compatibility
        base_dir = Path("outputs/glue/neobert-100m")

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist!")

        # Try to be helpful by showing nearby directories
        parent = base_dir.parent
        if parent.exists():
            print(f"\nAvailable directories in {parent}:")
            for subdir in sorted(parent.iterdir()):
                if subdir.is_dir():
                    print(f"  - {subdir.name}")
        return

    # Select baseline scores based on argument
    baseline_scores = BERT_BASE_SCORES  # Default
    baseline_name = "BERT-base"

    if args.baseline == "bert-large":
        baseline_scores = {
            "cola": 60.5,
            "sst2": 94.9,
            "mrpc": 87.4,  # Avg(F1=89.3, Acc=85.4)
            "stsb": 86.2,  # Avg(Pearson=86.5, Spearman=85.9)
            "qqp": 80.7,  # Avg(F1=72.1, Acc=89.3)
            "mnli": 86.3,  # Avg(m=86.7, mm=85.9)
            "qnli": 92.7,
            "rte": 70.1,
            "wnli": 65.1,
        }
        baseline_name = "BERT-large"
    elif args.baseline == "roberta-base":
        baseline_scores = {
            "cola": 63.6,
            "sst2": 94.8,
            "mrpc": 88.4,  # Avg(F1=90.2, Acc=86.7)
            "stsb": 91.0,  # Avg(Pearson=91.2, Spearman=90.9)
            "qqp": 90.2,  # Avg(F1=88.4, Acc=91.9)
            "mnli": 87.6,  # Avg(m=87.6, mm=87.5)
            "qnli": 92.8,
            "rte": 78.7,
            "wnli": 65.1,
        }
        baseline_name = "RoBERTa-base"
    elif args.baseline == "roberta-large":
        baseline_scores = {}
        baseline_name = "RoBERTa-large"
    elif args.baseline == "none":
        baseline_scores = {}
        baseline_name = None

    results = []
    for task, spec in TASK_SCORES.items():
        metric_name = spec["label"]
        scale = spec["scale"]
        task_dir = base_dir / task
        if not task_dir.exists():
            result_dict = {
                "Task": task.upper(),
                "Metric": metric_name,
                "Score": "Not run",
                "Status": "❌",
            }
            if baseline_name and baseline_scores:
                result_dict[baseline_name] = (
                    f"{baseline_scores.get(task, 'N/A'):.1f}"
                    if baseline_scores.get(task)
                    else "N/A"
                )
            results.append(result_dict)
            continue

        score = get_task_results(task_dir, task)
        if score is None:
            result_dict = {
                "Task": task.upper(),
                "Metric": metric_name,
                "Score": "In progress",
                "Status": "⏳",
            }
            if baseline_name and baseline_scores:
                result_dict[baseline_name] = (
                    f"{baseline_scores.get(task, 'N/A'):.1f}"
                    if baseline_scores.get(task)
                    else "N/A"
                )
            results.append(result_dict)
        else:
            score_pct = score * scale
            bert_score = baseline_scores.get(task, 0) if baseline_scores else 0
            diff = score_pct - bert_score if baseline_scores else None

            result_dict = {
                "Task": task.upper(),
                "Metric": metric_name,
                "Score": f"{score_pct:.1f}",
                "Status": "✅",
            }
            if baseline_name and baseline_scores:
                result_dict[baseline_name] = (
                    f"{bert_score:.1f}" if bert_score else "N/A"
                )
                if diff is not None:
                    result_dict["Diff"] = f"{diff:+.1f}" if bert_score else "N/A"
            results.append(result_dict)

    # Create DataFrame and print
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    # Extract a meaningful name from the path for the title
    model_name = base_dir.name if base_dir.name != "glue" else base_dir.parent.name
    print(f"GLUE Results: {model_name}")
    print("=" * 60)
    print(df.to_string(index=False))

    # Calculate average if we have all results
    completed = [r for r in results if r["Status"] == "✅"]
    if len(completed) == 9:
        scores = [float(r["Score"]) for r in completed if r["Score"] != "Not run"]
        avg_score = sum(scores) / len(scores)
        print("\n" + "-" * 60)
        print(f"Average GLUE Score: {avg_score:.1f}")
        if baseline_name and baseline_scores:
            baseline_avg = sum(baseline_scores.values()) / len(baseline_scores)
            print(f"{baseline_name} Average: {baseline_avg:.1f}")
            print(f"Difference: {avg_score - baseline_avg:+.1f}")
    else:
        print(f"\nCompleted {len(completed)}/9 tasks")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
