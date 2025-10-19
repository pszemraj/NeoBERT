# Evaluation Scripts

Scripts for evaluating NeoBERT models on benchmarks.

> [!NOTE]
> See [/docs/evaluation.md](/docs/evaluation.md) for comprehensive evaluation guide and benchmark documentation.

## Scripts Overview

- **`run_glue.py`** – GLUE benchmark evaluation
- **`run_mteb.py`** – MTEB benchmark evaluation
- **`run_quick_glue.sh`** – Helper launcher for fast per-task evaluation
- **`run_all_glue.sh`** – Bash wrapper that runs the full GLUE suite
- **`summarize_glue.py`** – Aggregate GLUE metrics across runs
- **`pseudo_perplexity.py`** – Calculate pseudo-perplexity for masked language models
- **`avg_mteb.py`** – Average MTEB results across tasks
- **`validate_glue_config.py`** – Sanity-check GLUE configs before launching jobs
- **`wrappers.py`** – Shared helpers for evaluation scripts

## Quick Start

```bash
# Run GLUE evaluation
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml

# Run MTEB evaluation
python scripts/evaluation/run_mteb.py --config configs/evaluate_neobert.yaml

# Run all GLUE tasks
bash scripts/evaluation/run_all_glue.sh
```

## Implementation Notes

### Flash Attention Compatibility

⚠️ GLUE evaluation always runs with eager attention. See [Flash Attention issues during GLUE evaluation](/docs/troubleshooting.md#flash-attention-issues-during-glue-evaluation) for background and mitigation steps.

### Model Checkpoint Saving

By default, GLUE evaluations **do not save model checkpoints** to conserve disk space. Only evaluation metrics (JSON files) are saved. To enable checkpoint saving:

```yaml
trainer:
  save_model: true  # Enable checkpoint saving
  save_total_limit: 3  # Keep only the 3 best checkpoints
```

### Pretrained Model Requirements

GLUE evaluation **requires** a pretrained model. The trainer includes safety checks to prevent accidentally running with random weights:

- Validates `pretrained_config_path` exists
- Checks checkpoint directory and files
- Verifies weights loaded properly

### Script-Specific Details

**`run_glue.py`:**

- Automatically handles task-specific metrics (Matthews correlation for CoLA, F1 for MRPC, etc.)
- Supports epoch-based or step-based evaluation strategies
- Saves results as `all_results_step_*.json`

**`run_mteb.py`:**

- Evaluates across 56+ tasks in 8 categories
- Requires sentence-transformers package
- Outputs results in MTEB leaderboard format

**`pseudo_perplexity.py`:**

- Calculates MLM perplexity using dynamic masking
- Useful for comparing pretraining quality
- Supports both validation and test sets

**`avg_mteb.py`:**

- Aggregates MTEB results across multiple runs
- Calculates mean and std deviation
- Outputs LaTeX-formatted tables
