# Evaluation Guide

This guide covers evaluating NeoBERT models on various benchmarks.

## GLUE Benchmark

### Prerequisites

GLUE evaluation requires a pretrained NeoBERT model. You can:
1. Use an existing checkpoint from pretraining
2. Train a new model (see [Training Guide](training.md))
3. Test with random weights using `--glue.allow_random_weights true`

### Running Single GLUE Task

```bash
# Run a specific GLUE task (e.g., CoLA)
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml

# Override checkpoint path
python scripts/evaluation/run_glue.py \
    --config configs/glue/cola.yaml \
    --glue.pretrained_checkpoint_dir ./outputs/your_checkpoint \
    --glue.pretrained_checkpoint 50000
```

### Running All GLUE Tasks

```bash
# Run full GLUE evaluation suite
bash scripts/run_full_glue.sh

# Tasks run in order from smallest to largest for quick feedback:
# WNLI, RTE, MRPC, STS-B, CoLA, SST-2, QNLI, QQP, MNLI
```

### GLUE Task Details

| Task | Type | Metrics | Train Size | Description |
|------|------|---------|------------|-------------|
| CoLA | Classification | Matthews Corr | 8.5k | Linguistic acceptability |
| SST-2 | Classification | Accuracy | 67k | Sentiment analysis |
| MRPC | Classification | F1/Accuracy | 3.7k | Paraphrase detection |
| STS-B | Regression | Pearson/Spearman | 5.7k | Semantic textual similarity |
| QQP | Classification | F1/Accuracy | 364k | Question pair similarity |
| MNLI | Classification | Accuracy | 393k | Natural language inference |
| QNLI | Classification | Accuracy | 105k | Question answering NLI |
| RTE | Classification | Accuracy | 2.5k | Recognizing textual entailment |
| WNLI | Classification | Accuracy | 600 | Coreference resolution |

### Configuration Structure

All GLUE configs are in `configs/glue/` with standardized structure:

```yaml
task: glue

model:
  name_or_path: neobert-100m
  pretrained_checkpoint_dir: ./outputs/neobert_100m_100k
  pretrained_checkpoint: 100000  # Step number or "latest"
  pretrained_config_path: ./outputs/neobert_100m_100k/model_checkpoints/100000/config.yaml
  # Model architecture params...

glue:
  task_name: cola  # Task identifier
  num_labels: 2    # Automatically set based on task
  max_seq_length: 128  # Most tasks use 128, RTE uses 256
  
trainer:
  output_dir: ./outputs/glue/neobert-100m/cola
  num_train_epochs: 3
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  eval_strategy: steps  # Evaluate periodically
  eval_steps: 50       # Task-dependent (smaller for small datasets)
  save_strategy: steps
  save_steps: 50
  save_total_limit: 3  # Keep only best 3 checkpoints
  early_stopping: 5    # Stop if no improvement for 5 evals
  metric_for_best_model: eval_matthews_correlation  # Task-specific
  mixed_precision: bf16
  tf32: true
  
optimizer:
  name: adamw
  lr: 2e-5  # Standard GLUE learning rate
  weight_decay: 0.01
  
scheduler:
  name: linear
  warmup_percent: 10  # 10% warmup is standard for GLUE

wandb:
  project: neobert-glue
  name: neobert-100m-{task}-{checkpoint}
```

### Summarizing Results

After running GLUE evaluation, use the summary script:

```bash
# Summarize results from a specific path
python scripts/summarize_glue.py outputs/glue/neobert-100m

# Compare against different baselines
python scripts/summarize_glue.py outputs/glue/neobert-100m --baseline roberta-base
python scripts/summarize_glue.py outputs/glue/neobert-100m --baseline bert-large
python scripts/summarize_glue.py outputs/glue/neobert-100m --baseline none

# Works with any output directory structure
python scripts/summarize_glue.py ./experiments/test_123/glue_results
```

### Output Structure

GLUE results are organized as:
```
outputs/
└── glue/
    └── {model_name}/
        ├── cola/
        │   ├── checkpoint-{step}/
        │   ├── all_results.json
        │   └── all_results_step_{step}.json
        ├── sst2/
        ├── mrpc/
        └── ...
```

## MTEB Benchmark

### Running MTEB Evaluation

```bash
# Run full MTEB evaluation
python scripts/evaluation/run_mteb.py \
    --config configs/evaluate_neobert.yaml

# Run specific task types
python scripts/evaluation/run_mteb.py \
    --config configs/evaluate_neobert.yaml \
    --task_types retrieval,sts

# Run specific tasks
python scripts/evaluation/run_mteb.py \
    --config configs/evaluate_neobert.yaml \
    --tasks "MSMARCO,NQ,HotpotQA"
```

### MTEB Task Types

- **Retrieval**: Information retrieval (MSMARCO, NQ, HotpotQA, etc.)
- **STS**: Semantic textual similarity
- **Clustering**: Text clustering
- **PairClassification**: Pair classification tasks
- **Reranking**: Passage reranking
- **Classification**: Text classification
- **Summarization**: Summarization evaluation

### MTEB Configuration

```yaml
model:
  pooling_strategy: mean  # or cls, max
  normalize_embeddings: true

mteb:
  task_types: ["retrieval", "sts", "clustering"]
  batch_size: 32
  corpus_chunk_size: 50000  # For retrieval tasks
```

## Loading Pretrained Checkpoints

NeoBERT uses DeepSpeed checkpoints from pretraining. The GLUE evaluation system handles this automatically:

```python
# In configs, specify:
glue:
  pretrained_checkpoint_dir: ./outputs/neobert_100m_100k
  pretrained_checkpoint: 100000  # or "latest"
  
# The system will:
# 1. Load the DeepSpeed checkpoint
# 2. Extract model weights
# 3. Initialize the GLUE model with these weights
# 4. Add task-specific classification head
```

## Best Practices

### 1. Hyperparameter Selection

Standard GLUE hyperparameters that work well:
- Learning rate: 2e-5 (occasionally 1e-5 or 3e-5)
- Batch size: 32 (16 for large models)
- Epochs: 3 (sometimes 5 for small datasets)
- Warmup: 10% of training steps
- Max sequence length: 128 (256 for RTE)

### 2. Early Stopping

Configure early stopping to prevent overfitting:
```yaml
trainer:
  early_stopping: 5  # Patience in evaluation steps
  load_best_model_at_end: true
  metric_for_best_model: eval_{metric}
  greater_is_better: true  # false for loss
```

### 3. Evaluation Frequency

Balance between compute cost and monitoring:
- Small datasets (WNLI, RTE, MRPC): eval_steps=20-50
- Medium datasets (CoLA, STS-B): eval_steps=50-100  
- Large datasets (SST-2, QNLI): eval_steps=500
- Very large (QQP, MNLI): eval_steps=1000-2000

### 4. Mixed Precision

Always use bf16 for modern GPUs:
```yaml
trainer:
  mixed_precision: bf16
  tf32: true  # Additional speedup on Ampere+
```

## Troubleshooting

### Flash Attention Issues

If you encounter Flash Attention errors with GLUE:
```
Flash attention is not supported for GLUE evaluation due to memory alignment issues
```
This is expected - GLUE uses variable-length sequences that can cause issues with Flash Attention. The system automatically falls back to standard attention.

### Out of Memory

```bash
# Reduce batch size
--trainer.per_device_train_batch_size 16
--trainer.gradient_accumulation_steps 2  # Maintain effective batch size

# Enable gradient checkpointing
--model.gradient_checkpointing true
```

### Slow Training

```bash
# Ensure mixed precision is enabled
--trainer.mixed_precision bf16
--trainer.tf32 true

# Increase dataloader workers
--trainer.dataloader_num_workers 4
```

### Poor Results

If getting random or near-random results:
1. Check the pretrained checkpoint loaded correctly (check logs)
2. Verify learning rate (2e-5 is standard)
3. Ensure sufficient training (3 epochs minimum)
4. Check early stopping isn't too aggressive

## Advanced Usage

### Custom Metrics

Add custom metrics to GLUE evaluation:

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Standard metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Add custom metrics
    custom_metric = your_custom_function(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "custom": custom_metric
    }
```

### Multi-Run Evaluation

For statistical significance:

```bash
# Run with different seeds
for seed in 42 1337 2023; do
    python scripts/evaluation/run_glue.py \
        --config configs/glue/cola.yaml \
        --trainer.seed $seed \
        --trainer.output_dir outputs/glue/seed_$seed
done

# Aggregate results
python scripts/aggregate_results.py outputs/glue/seed_*
```

## WandB Integration

All GLUE runs are tracked in WandB:
- Project: `neobert-glue`
- Run names: `{model}-{task}-{checkpoint}`
- Logged metrics: loss, task metrics, learning rate
- Logged configs: full configuration

View results at: https://wandb.ai/your-username/neobert-glue

## Next Steps

- Review [Training Guide](training.md) for pretraining details
- Check [Configuration Guide](configuration.md) for config system
- See [Testing Guide](testing.md) for running tests