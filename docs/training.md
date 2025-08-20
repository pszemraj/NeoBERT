# Training Guide

This guide covers pretraining and fine-tuning NeoBERT models.

## Pretraining

### Basic Pretraining

```bash
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
```

### Key Configuration Options

```yaml
# Model architecture
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  vocab_size: 30522
  max_position_embeddings: 512
  
# Training settings
trainer:
  max_steps: 1000000
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  learning_rate: 1e-4
  warmup_steps: 10000
  
# Optimizer
optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
```

### Multi-GPU Training

Using Accelerate (recommended):
```bash
# Configure accelerate
accelerate config

# Run training
accelerate launch scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
```

Using DeepSpeed:
```bash
# With DeepSpeed config
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --accelerate_config_file configs/accelerate_deepspeed_zero2.yaml
```

### Phase 2 Pretraining (Longer Sequences)

After initial pretraining with 512 tokens:
```bash
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert_phase2.yaml \
    --model.max_position_embeddings 4096 \
    --dataset.max_seq_length 4096
```

### Monitoring Training

With Weights & Biases:
```yaml
wandb:
  project: neobert-pretraining
  entity: your-username
  mode: online
```

View metrics:
```bash
# Training loss
wandb.log({"train/loss": loss})

# Learning rate
wandb.log({"train/learning_rate": lr})

# GPU memory
wandb.log({"train/gpu_memory": torch.cuda.max_memory_allocated()})
```

## Dataset Preparation

### Using HuggingFace Datasets

```python
# In your config
dataset:
  name: "wikipedia"
  train_split: "train[:10%]"  # Use 10% for testing
```

### Using Custom Data

1. Prepare your data in text format
2. Tokenize it:
```bash
python scripts/pretraining/tokenize_dataset.py \
    --dataset "path/to/your/data" \
    --tokenizer "bert-base-uncased" \
    --output "./tokenized_data/custom"
```

3. Update config:
```yaml
dataset:
  path: "./tokenized_data/custom"
```

### Data Formats

Supported formats:
- Raw text files (.txt)
- JSON lines (.jsonl) with "text" field
- HuggingFace datasets
- Pre-tokenized datasets (arrow format)
- Streaming datasets from HuggingFace Hub

### Streaming Datasets

For large datasets that don't fit in memory, use streaming mode:

```yaml
dataset:
  name: "common-pile/comma_v0.1_training_dataset"
  streaming: true
  shuffle_buffer_size: 10000  # Buffer size for shuffling
  num_workers: 0  # Must be 0 for streaming datasets
```

Benefits of streaming:
- No need to download entire dataset
- Tokenization happens on-the-fly
- Memory efficient for large datasets
- Automatic shuffling with buffer

Example command:
```bash
python scripts/pretraining/pretrain.py \
    --config configs/streaming_pretrain.yaml
```

## Fine-Tuning

### GLUE Tasks

```bash
# Single task
python scripts/evaluation/run_glue.py \
    --config configs/evaluate_neobert.yaml \
    --task_name cola \
    --model_name_or_path outputs/pretrained_model

# All GLUE tasks
for task in cola mnli mrpc qnli qqp rte sst2 stsb wnli; do
    python scripts/evaluation/run_glue.py \
        --config configs/evaluate_neobert.yaml \
        --task_name $task
done
```

### Contrastive Learning (SimCSE)

```bash
python scripts/contrastive/finetune.py \
    --config configs/contrastive_neobert.yaml \
    --model_name_or_path outputs/pretrained_model
```

### Custom Tasks

Create a custom trainer:
```python
from neobert.model import NeoBERTForSequenceClassification
from transformers import Trainer, TrainingArguments

model = NeoBERTForSequenceClassification.from_pretrained(
    "outputs/pretrained_model",
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Training Tips

### Memory Optimization

1. **Gradient Accumulation**:
```bash
--trainer.gradient_accumulation_steps 4 \
--trainer.per_device_train_batch_size 8
```

2. **Gradient Checkpointing**:
```bash
--trainer.gradient_checkpointing true
```

3. **Mixed Precision**:
```bash
--trainer.bf16 true  # Uses bfloat16 (recommended for modern GPUs)
--trainer.mixed_precision "bf16"
```

4. **DeepSpeed ZeRO**:
```bash
--accelerate_config_file configs/accelerate_deepspeed_zero3.yaml
```

### Performance Optimization

1. **Flash Attention**:
```bash
--model.flash_attention true
```

2. **Compiled Model** (PyTorch 2.0+):
```python
model = torch.compile(model)
```

3. **Efficient Data Loading**:
```bash
--dataset.num_workers 4 \
--trainer.dataloader_pin_memory true
```

### Learning Rate Scheduling

```yaml
scheduler:
  name: cosine
  warmup_steps: 10000
  total_steps: 1000000
```

Other options:
- `linear`: Linear decay
- `constant`: Constant LR after warmup
- `polynomial`: Polynomial decay

## Debugging

### Enable Debug Mode

```bash
--debug true
```

This enables:
- Verbose logging
- Gradient norm tracking
- Memory profiling

### Common Issues

1. **NaN Loss**:
   - Reduce learning rate
   - Check for gradient explosion
   - Enable gradient clipping: `--trainer.max_grad_norm 1.0`

2. **Slow Training**:
   - Enable flash attention
   - Use larger batches with accumulation
   - Check data loading bottlenecks

3. **Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use DeepSpeed ZeRO

### Profiling

```python
# PyTorch profiler
with torch.profiler.profile() as prof:
    model(batch)
print(prof.key_averages())
```

## Checkpointing and Resume

### Save Checkpoints

```yaml
trainer:
  save_steps: 10000
  save_total_limit: 3  # Keep only 3 checkpoints
```

### Resume Training

```bash
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.resume_from_checkpoint outputs/checkpoint-50000
```

## Next Steps

- Learn about [Evaluation](evaluation.md)
- Explore [Custom Tokenizers](custom_tokenizers.md)
- Read about [Model Architecture](architecture.md)