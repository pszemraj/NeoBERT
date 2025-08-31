# Evaluation Guide

This guide covers evaluating NeoBERT models on various benchmarks.

## GLUE Benchmark

### Prerequisites

GLUE evaluation requires a pretrained NeoBERT model. First train a model:

```bash
# Train a small model for testing
python scripts/pretraining/pretrain.py \
    --config configs/test_tiny_pretrain.yaml \
    --trainer.max_steps 100
```

### Running Single Task

```bash
# Run GLUE evaluation (uses model path from config)
python scripts/evaluation/run_glue.py --config configs/eval/glue_cola.yaml

# Or override the model path from command line
python scripts/evaluation/run_glue.py \
    --config configs/eval/glue_cola.yaml \
    --model_name_or_path outputs/different_model
```

### Running All GLUE Tasks

```bash
# Run all 9 GLUE tasks (uses model paths from configs)
bash jobs/run_all_glue.sh

# Or override model path for all tasks
bash jobs/run_all_glue.sh outputs/different_model
```

### GLUE Task Details

| Task | Type | Metrics | Description |
|------|------|---------|-------------|
| CoLA | Classification | Matthews Corr | Linguistic acceptability |
| MNLI | Classification | Accuracy | Natural language inference |
| MRPC | Classification | F1/Accuracy | Paraphrase detection |
| QNLI | Classification | Accuracy | Question answering NLI |
| QQP | Classification | F1/Accuracy | Question pair similarity |
| RTE | Classification | Accuracy | Recognizing textual entailment |
| SST-2 | Classification | Accuracy | Sentiment analysis |
| STS-B | Regression | Pearson/Spearman | Semantic textual similarity |
| WNLI | Classification | Accuracy | Coreference resolution |

### Configuration for GLUE

```yaml
task: glue

glue:
  task_name: cola
  num_labels: 2  # Automatically set based on task
  max_seq_length: 512
  
trainer:
  num_train_epochs: 3
  train_batch_size: 32
  gradient_accumulation_steps: 4  # Effective batch size = 128
  eval_strategy: "epoch"  # Evaluate at end of each epoch
  save_strategy: "epoch"  # Save at end of each epoch
  metric_for_best_model: eval_matthews_correlation  # Task-specific
  greater_is_better: true
  
scheduler:
  name: linear
  warmup_steps: null  # IMPORTANT: Override dataclass default
  decay_steps: null   # IMPORTANT: Override dataclass default
  warmup_percent: 5  # 5% of total steps for warmup
```

### Evaluation Strategies

The evaluation system supports multiple strategies:

- **`eval_strategy: "epoch"`** - Evaluate at the end of each epoch (recommended)
- **`eval_strategy: "steps"`** - Evaluate every N steps (legacy)
- **`eval_strategy: "no"`** - No evaluation during training

### Save Strategies

Control when checkpoints are saved:

- **`save_strategy: "epoch"`** - Save at the end of each epoch
- **`save_strategy: "steps"`** - Save every N steps
- **`save_strategy: "best"`** - Save only when metric improves
- **`save_strategy: "no"`** - Don't save checkpoints

### Task-Specific Tips

**CoLA (Corpus of Linguistic Acceptability)**:
```bash
python scripts/evaluation/run_glue.py \
    --config configs/eval/glue_cola.yaml \
    --optimizer.lr 5e-5 \
    --trainer.num_train_epochs 3
```

**MNLI (Multi-Genre NLI)**:
```bash
python scripts/evaluation/run_glue.py \
    --config configs/eval/glue_mnli.yaml \
    --glue.max_seq_length 128 \
    --trainer.train_batch_size 32 \
    --optimizer.lr 3e-5
```

**STS-B (Semantic Textual Similarity)**:
```bash
# Note: This is a regression task
--glue.task_name stsb \
--trainer.metric_for_best_model eval_combined_score
```

## MTEB Benchmark

### Running MTEB Evaluation

```bash
python scripts/evaluation/run_mteb.py \
    --config configs/evaluate_neobert.yaml \
    --model_name_or_path outputs/pretrained_model \
    --task_types all
```

### MTEB Task Types

- `retrieval`: Information retrieval tasks
- `sts`: Semantic textual similarity
- `clustering`: Text clustering
- `pair_classification`: Pair classification
- `reranking`: Passage reranking
- `classification`: Text classification
- `summarization`: Summarization evaluation

### Specific MTEB Tasks

```bash
# Run specific task type
python scripts/evaluation/run_mteb.py \
    --config configs/evaluate_neobert.yaml \
    --task_types retrieval,sts

# Run specific tasks
python scripts/evaluation/run_mteb.py \
    --config configs/evaluate_neobert.yaml \
    --tasks "MSMARCO,NQ,HotpotQA"
```

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

## Custom Evaluation

### Perplexity Evaluation

```bash
python scripts/evaluation/pseudo_perplexity.py \
    --model_name_or_path outputs/pretrained_model \
    --dataset_name wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --split test
```

### Zero-Shot Evaluation

```python
from neobert.model import NeoBERT
from transformers import pipeline

# Load model
model = NeoBERT.from_pretrained("outputs/pretrained_model")
tokenizer = AutoTokenizer.from_pretrained("outputs/pretrained_model")

# Create pipeline
classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer
)

# Evaluate
result = classifier(
    "This movie is fantastic!",
    candidate_labels=["positive", "negative", "neutral"]
)
```

### Domain-Specific Evaluation

```python
# Custom metric example
def evaluate_domain_specific(model, dataset):
    predictions = []
    references = []
    
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.tolist())
            references.extend(batch["labels"].tolist())
    
    # Calculate domain-specific metrics
    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average='weighted')
    
    return {"accuracy": accuracy, "f1": f1}
```

## Advanced Evaluation Strategies

### 1. Few-Shot Learning

```python
# Configure for few-shot
trainer_args = TrainingArguments(
    num_train_epochs=10,  # More epochs for small data
    learning_rate=5e-5,   # Higher LR
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

# Sample few examples per class
train_dataset = train_dataset.select(range(16))  # 16 examples
```

### 2. Cross-Validation

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # Train on fold
    fold_model = train_on_fold(train_idx, val_idx)
    
    # Evaluate
    score = evaluate(fold_model, val_idx)
    cv_scores.append(score)

print(f"CV Score: {np.mean(cv_scores)} ± {np.std(cv_scores)}")
```

### 3. Multi-Task Evaluation

```bash
# Train on multiple GLUE tasks jointly
python scripts/evaluation/run_glue.py \
    --config configs/evaluate_neobert.yaml \
    --task_name cola,mrpc,rte \
    --multi_task true
```

## Metrics and Interpretation

### Classification Metrics

```python
from sklearn.metrics import classification_report

# Get detailed metrics
report = classification_report(
    y_true=labels,
    y_pred=predictions,
    target_names=class_names,
    output_dict=True
)

# Key metrics
accuracy = report['accuracy']
macro_f1 = report['macro avg']['f1-score']
weighted_f1 = report['weighted avg']['f1-score']
```

### Regression Metrics

```python
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# For STS-B
pearson_corr, _ = pearsonr(predictions, labels)
spearman_corr, _ = spearmanr(predictions, labels)
mse = mean_squared_error(labels, predictions)
```

### Embedding Quality Metrics

```python
# Intrinsic evaluation
from sklearn.metrics import silhouette_score

# Clustering quality
silhouette = silhouette_score(embeddings, cluster_labels)

# Semantic similarity
cos_sim = cosine_similarity(embeddings)
```

## Visualization

### Training Curves

```python
import matplotlib.pyplot as plt

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(eval_losses, label='Eval Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curves.png')
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
```

### Embedding Visualization

```python
from sklearn.manifold import TSNE

# Reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
plt.colorbar()
plt.savefig('embeddings_tsne.png')
```

## Best Practices

### 1. Configuration Best Practices

```yaml
# When creating evaluation configs, explicitly override dataclass defaults
scheduler:
  name: linear
  warmup_steps: null  # Override default of 10000
  decay_steps: null   # Override default of 50000
  warmup_ratio: 0.05  # Use ratio-based warmup

trainer:
  eval_strategy: "epoch"  # Modern approach
  save_strategy: "epoch"  # Aligned with eval
```

### 2. Reproducibility

```bash
# Set seeds
--seed 42 \
--trainer.seed 42 \
--trainer.data_seed 42
```

### 3. Statistical Significance

```python
# Multiple runs with different seeds
seeds = [42, 1337, 2023, 3407, 5555]
scores = []

for seed in seeds:
    score = evaluate_with_seed(seed)
    scores.append(score)

mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"Score: {mean_score:.3f} ± {std_score:.3f}")
```

### 4. Compute Efficiency

```bash
# Batch evaluation
--trainer.per_device_eval_batch_size 128 \
--trainer.dataloader_num_workers 4 \
--trainer.bf16 true \
--trainer.mixed_precision "bf16"
```

## Troubleshooting

### Out of Memory During Evaluation

```bash
# Reduce batch size
--trainer.per_device_eval_batch_size 16

# Use gradient checkpointing
--trainer.gradient_checkpointing true

# Clear cache between evaluations
torch.cuda.empty_cache()
```

### Slow Evaluation

```bash
# Enable faster inference
--model.use_cache true \
--trainer.bf16 true \
--trainer.dataloader_pin_memory true
```

### Metric Computation Issues

```python
# Handle edge cases
def safe_metric_computation(preds, labels):
    if len(np.unique(labels)) == 1:
        # Single class - return 0 for undefined metrics
        return {"f1": 0.0, "precision": 0.0, "recall": 1.0}
    
    return {
        "f1": f1_score(labels, preds, average='macro'),
        "precision": precision_score(labels, preds, average='macro'),
        "recall": recall_score(labels, preds, average='macro')
    }
```

## Configuration Files

Evaluation configurations are in `configs/eval/`:

- **`glue_cola.yaml`** - CoLA evaluation
- **`glue_sst2.yaml`** - SST-2 sentiment analysis
- **`glue_mrpc.yaml`** - MRPC paraphrase detection
- **`glue_stsb.yaml`** - STS-B semantic similarity (regression)
- **`glue_qqp.yaml`** - QQP question pair similarity
- **`glue_mnli.yaml`** - MNLI natural language inference
- **`glue_qnli.yaml`** - QNLI question answering NLI
- **`glue_rte.yaml`** - RTE textual entailment
- **`glue_wnli.yaml`** - WNLI coreference resolution

These configs follow HuggingFace Transformers patterns:
- Evaluate at the end of each epoch (`eval_strategy: "epoch"`)
- Use warmup percentage instead of fixed steps
- Support gradient accumulation for larger effective batch sizes

## Next Steps

- Review [Training Guide](training.md) for fine-tuning tips
- Check [Model Architecture](architecture.md) for model variants
- See [Configuration README](../configs/README.md) for detailed config documentation