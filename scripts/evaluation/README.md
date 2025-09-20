# Evaluation Scripts

This directory contains scripts for evaluating NeoBERT models on various benchmarks.

## Scripts Overview

- **`run_glue.py`** - Evaluate models on GLUE benchmark tasks
- **`run_mteb.py`** - Evaluate models on MTEB (Massive Text Embedding Benchmark)
- **`eval_checkpoint.py`** - General checkpoint evaluation utility
- **`pseudo_perplexity.py`** - Calculate pseudo-perplexity for masked language models
- **`avg_mteb.py`** - Average MTEB results across tasks
- **`wrappers.py`** - Utility wrappers for evaluation

## GLUE Evaluation

### Quick Start

Evaluate a pretrained checkpoint on GLUE tasks:

```bash
# SST-2 (sentiment analysis)
python scripts/evaluation/run_glue.py --config configs/eval/glue_sst2.yaml

# CoLA (grammatical acceptability)
python scripts/evaluation/run_glue.py --config configs/eval/glue_cola.yaml

# MRPC (paraphrase detection)
python scripts/evaluation/run_glue.py --config configs/eval/glue_mrpc.yaml
```

### Available GLUE Tasks

All 9 GLUE tasks are supported with pre-configured YAML files:

| Task | Description | Config File | Metric |
|------|-------------|-------------|--------|
| CoLA | Grammar acceptability | `configs/eval/glue_cola.yaml` | Matthews correlation |
| SST-2 | Sentiment analysis | `configs/eval/glue_sst2.yaml` | Accuracy |
| MRPC | Paraphrase detection | `configs/eval/glue_mrpc.yaml` | F1/Accuracy |
| STS-B | Semantic similarity | `configs/eval/glue_stsb.yaml` | Pearson/Spearman |
| QQP | Question pair similarity | `configs/eval/glue_qqp.yaml` | F1/Accuracy |
| MNLI | Natural language inference | `configs/eval/glue_mnli.yaml` | Accuracy |
| QNLI | Question NLI | `configs/eval/glue_qnli.yaml` | Accuracy |
| RTE | Textual entailment | `configs/eval/glue_rte.yaml` | Accuracy |
| WNLI | Winograd NLI | `configs/eval/glue_wnli.yaml` | Accuracy |

### Configuration Structure

GLUE evaluation configs require specifying the pretrained model:

```yaml
task: sst2
meta_task: glue

model:
  pretrained_config_path: configs/pretrain/pretrain_neobert100m_smollm2data.yaml
  pretrained_checkpoint_dir: outputs/smollm2_custom_tokenizer
  pretrained_checkpoint: 100000  # Step number or "latest"
  num_labels: 2  # Task-specific
  
trainer:
  output_dir: glue_outputs/sst2
  eval_strategy: "epoch"  # or "steps"
  save_strategy: "epoch"
  per_device_train_batch_size: 32
  learning_rate: 2e-5
  save_model: false  # Disable checkpoint saving (default: false)
  save_total_limit: 0  # Set to 0 to disable checkpoints
  # ... other training arguments
```

### Model Checkpoint Saving

By default, GLUE evaluations **do not save model checkpoints** to conserve disk space. Only evaluation metrics (JSON files) are saved. This is usually sufficient since you're evaluating a pretrained model.

If you need to save the fine-tuned model checkpoints:

```yaml
trainer:
  save_model: true  # Enable checkpoint saving
  save_total_limit: 3  # Keep only the 3 best checkpoints
  max_ckpt: 3  # Alternative limit setting
```

### Running All GLUE Tasks

Use the provided job script to run all tasks:

```bash
bash jobs/run_all_glue.sh
```

### Important Notes

#### Flash Attention Compatibility

⚠️ **Flash attention is NOT supported for GLUE evaluation** due to memory alignment issues with variable-length sequences. The GLUE trainer automatically disables flash attention and uses eager attention instead. If your config specifies `flash_attention: true`, you'll see a warning:

```
Flash attention is not supported for GLUE evaluation due to memory alignment issues with variable-length sequences. Using eager attention instead.
```

This is expected behavior and does not affect model accuracy - only evaluation speed.

#### Pretrained Model Requirements

GLUE evaluation **requires** a pretrained model. The trainer will error if:
- No `pretrained_config_path` is specified
- The checkpoint directory doesn't exist
- The checkpoint files are missing

To prevent accidentally running with random weights, the trainer includes safety checks that verify weights were loaded properly.

#### Epoch-Based Evaluation

The configs support epoch-based evaluation natively:

```yaml
trainer:
  eval_strategy: "epoch"  # Evaluate at the end of each epoch
  save_strategy: "epoch"  # Save checkpoints at epoch boundaries
```

## MTEB Evaluation

Evaluate models on the Massive Text Embedding Benchmark:

```bash
python scripts/evaluation/run_mteb.py --config configs/eval/mteb_neobert.yaml
```

MTEB evaluates embedding quality across multiple tasks including:
- Classification
- Clustering  
- Pair classification
- Reranking
- Retrieval
- STS (Semantic Textual Similarity)
- Summarization

## Checkpoint Evaluation

For general checkpoint evaluation with custom metrics:

```bash
python scripts/evaluation/eval_checkpoint.py \
  --checkpoint_dir outputs/model_checkpoints \
  --checkpoint_step 100000 \
  --config configs/pretrain/pretrain_neobert.yaml
```

## Troubleshooting

### Low Performance on GLUE

1. **Check if weights loaded**: Look for messages confirming checkpoint loading
2. **Verify checkpoint path**: Ensure the path points to the correct checkpoint
3. **Check Matthews correlation**: For CoLA, 0.0 often indicates random weights
4. **Run sanity check**: Use the post-pretraining sanity check to verify basic capabilities

### Memory Issues

- Reduce `per_device_eval_batch_size` if running out of memory
- For large models, use gradient accumulation to simulate larger batches

### Slow Evaluation

- GLUE evaluation uses eager attention (not flash attention) for correctness
- This is slower but ensures accurate results with variable-length sequences
- See `docs/known_issues.md` for technical details

## Logging

All evaluations log to:
- **Console**: Progress bars and key metrics
- **WandB**: Detailed metrics and training curves (project: `neobert-evals`)
- **JSON files**: Results saved as `all_results_step_*.json` in output directory

## See Also

- [`configs/glue/`](../../configs/glue/) - Pre-configured GLUE evaluation configs
- [`src/neobert/glue/`](../../src/neobert/glue/) - GLUE trainer implementation