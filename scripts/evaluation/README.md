# Evaluation Scripts

Scripts for evaluating NeoBERT models on benchmarks.

📚 **For comprehensive documentation, see [docs/evaluation.md](../../docs/evaluation.md)**

## Scripts Overview

- **`run_glue.py`** - Evaluate models on GLUE benchmark tasks
- **`run_mteb.py`** - Evaluate models on MTEB (Massive Text Embedding Benchmark)
- **`eval_checkpoint.py`** - General checkpoint evaluation utility
- **`pseudo_perplexity.py`** - Calculate pseudo-perplexity for masked language models
- **`avg_mteb.py`** - Average MTEB results across tasks
- **`wrappers.py`** - Utility wrappers for evaluation

## Quick Start

```bash
# Run GLUE evaluation
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml

# Run MTEB evaluation
python scripts/evaluation/run_mteb.py --config configs/evaluate_neobert.yaml

# Run all GLUE tasks
bash scripts/run_all_glue.sh
```

## Implementation Notes

### Flash Attention Compatibility

⚠️ **Flash attention is NOT supported for GLUE evaluation** due to memory alignment issues with variable-length sequences. The GLUE trainer automatically disables flash attention and uses eager attention instead. This is expected behavior and does not affect accuracy.

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