#!/usr/bin/env python3
"""Summarize GLUE results from output directories."""

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
    base_dir = Path("outputs/glue/neobert-100m")

    results = []
    for task, (metric_name, scale) in TASK_METRICS.items():
        task_dir = base_dir / task
        if not task_dir.exists():
            results.append(
                {
                    "Task": task.upper(),
                    "Metric": metric_name.replace("_", " ").title(),
                    "Score": "Not run",
                    "BERT-base": f"{BERT_BASE_SCORES.get(task, 'N/A'):.1f}",
                    "Status": "❌",
                }
            )
            continue

        score = get_task_results(task_dir)
        if score is None:
            results.append(
                {
                    "Task": task.upper(),
                    "Metric": metric_name.replace("_", " ").title(),
                    "Score": "In progress",
                    "BERT-base": f"{BERT_BASE_SCORES.get(task, 'N/A'):.1f}",
                    "Status": "⏳",
                }
            )
        else:
            score_pct = score * scale
            bert_score = BERT_BASE_SCORES.get(task, 0)
            diff = score_pct - bert_score

            results.append(
                {
                    "Task": task.upper(),
                    "Metric": metric_name.replace("_", " ").title(),
                    "Score": f"{score_pct:.1f}",
                    "BERT-base": f"{bert_score:.1f}",
                    "Diff": f"{diff:+.1f}" if bert_score else "N/A",
                    "Status": "✅",
                }
            )

    # Create DataFrame and print
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("NeoBERT-100m GLUE Results (100k checkpoint)")
    print("=" * 60)
    print(df.to_string(index=False))

    # Calculate average if we have all results
    completed = [r for r in results if r["Status"] == "✅"]
    if len(completed) == 9:
        scores = [float(r["Score"]) for r in completed if r["Score"] != "Not run"]
        avg_score = sum(scores) / len(scores)
        print("\n" + "-" * 60)
        print(f"Average GLUE Score: {avg_score:.1f}")
        print("BERT-base Average: ~79.6")
        print(f"Difference: {avg_score - 79.6:+.1f}")
    else:
        print(f"\nCompleted {len(completed)}/9 tasks")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
