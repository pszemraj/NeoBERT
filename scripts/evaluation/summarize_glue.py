#!/usr/bin/env python3
"""Summarize GLUE results from output directories.

Usage:
    python scripts/evaluation/summarize_glue.py outputs/glue/neobert-100m
    python scripts/evaluation/summarize_glue.py ./outputs/experiment_xyz/glue_results
    python scripts/evaluation/summarize_glue.py outputs/glue/neobert-100m --baseline roberta-base
    python scripts/evaluation/summarize_glue.py  # Default: outputs/glue/neobert-100m
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Expected metrics for each task
TASK_METRICS = {
    "cola": ("matthews_correlation", 100),
    "sst2": ("accuracy", 100),
    "mrpc": ("f1", 100),
    "stsb": ("pearson", 100),
    "qqp": ("f1", 100),
    "mnli": ("accuracy", 100),
    "qnli": ("accuracy", 100),
    "rte": ("accuracy", 100),
    "wnli": ("accuracy", 100),
}

# Baseline scores for reference (BERT-base scores from literature)
BERT_BASE_SCORES = {
    "cola": 52.1,
    "sst2": 93.5,
    "mrpc": 88.9,  # F1 score
    "stsb": 85.8,  # Pearson
    "qqp": 71.2,  # F1
    "mnli": 84.6,
    "qnli": 90.5,
    "rte": 66.4,
    "wnli": 65.1,
}


def get_task_results(task_dir):
    """Extract best results from a task directory."""
    results_files = list(task_dir.glob("all_results_step_*.json"))
    if not results_files:
        results_file = task_dir / "all_results.json"
        if results_file.exists():
            results_files = [results_file]
        else:
            return None

    best_score = -1
    for f in results_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                if data:  # Check if dict is not empty
                    # Find the relevant metric
                    for key, value in data.items():
                        if any(
                            metric in key
                            for metric in ["matthews", "accuracy", "f1", "pearson"]
                        ):
                            if value > best_score:
                                best_score = value
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    return best_score if best_score > -1 else None


def main():
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
            "mrpc": 89.3,
            "stsb": 86.5,
            "qqp": 72.1,
            "mnli": 86.7,
            "qnli": 92.7,
            "rte": 70.1,
            "wnli": 65.1,
        }
        baseline_name = "BERT-large"
    elif args.baseline == "roberta-base":
        baseline_scores = {
            "cola": 63.6,
            "sst2": 94.8,
            "mrpc": 90.2,
            "stsb": 91.2,
            "qqp": 91.9,
            "mnli": 87.6,
            "qnli": 92.8,
            "rte": 78.7,
            "wnli": 65.1,
        }
        baseline_name = "RoBERTa-base"
    elif args.baseline == "none":
        baseline_scores = {}
        baseline_name = None

    results = []
    for task, (metric_name, scale) in TASK_METRICS.items():
        task_dir = base_dir / task
        if not task_dir.exists():
            result_dict = {
                "Task": task.upper(),
                "Metric": metric_name.replace("_", " ").title(),
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

        score = get_task_results(task_dir)
        if score is None:
            result_dict = {
                "Task": task.upper(),
                "Metric": metric_name.replace("_", " ").title(),
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
                "Metric": metric_name.replace("_", " ").title(),
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
