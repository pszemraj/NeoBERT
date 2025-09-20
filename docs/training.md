# Training Guide

This guide covers pretraining and fine-tuning NeoBERT models.

## Pretraining

### Basic Pretraining

```bash
# Run pretraining with default config
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml

# Override specific settings via CLI
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 64 \
    --optimizer.lr 2e-4 \
    --trainer.max_steps 100000
```

### Key Configuration Options

```yaml
# Model architecture
model:
  name_or_path: neobert-100m
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  vocab_size: 30522
  max_position_embeddings: 512
  hidden_act: swiglu        # Modern activation
  normalization_type: rmsnorm  # RMSNorm instead of LayerNorm
  use_rope: true           # RoPE positional encoding
  rope_theta: 10000
  flash_attention: true    # Enable Flash Attention 2

# Training settings
trainer:
  output_dir: ./outputs/neobert_100m
  max_steps: 1000000
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  eval_strategy: steps
  eval_steps: 10000
  save_strategy: steps
  save_steps: 10000
  logging_steps: 100
  mixed_precision: bf16    # Always use bf16
  tf32: true              # Enable TensorFloat32
  dataloader_num_workers: 4

# Optimizer
optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8

# Scheduler
scheduler:
  name: cosine
  warmup_steps: 10000
  min_lr_ratio: 0.1
```

### Multi-GPU Training with DeepSpeed

The pretraining pipeline uses DeepSpeed for efficient multi-GPU training:

```bash
# DeepSpeed is automatically configured based on your system
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml

# The script auto-detects GPUs and configures:
# - ZeRO Stage 2 optimization
# - FP16/BF16 mixed precision
# - Gradient accumulation
# - CPU offloading if needed
```

DeepSpeed configuration is handled automatically in `src/neobert/training/pretraining.py`:

- Detects available GPUs
- Configures optimal ZeRO stage
- Enables mixed precision based on hardware
- Sets up gradient accumulation

### Streaming Datasets

For large datasets that don't fit in memory:

```yaml
dataset:
  name: "common-pile/comma_v0.1_training_dataset"
  streaming: true
  max_seq_length: 512
  mlm_probability: 0.15
  shuffle_buffer_size: 10000
  num_workers: 0  # Must be 0 for streaming
```

Benefits:

- No need to download entire dataset upfront
- Tokenization happens on-the-fly
- Memory efficient for multi-TB datasets
- Automatic shuffling with buffer

Example streaming config:

```bash
python scripts/pretraining/pretrain.py \
    --config configs/streaming_pretrain.yaml
```

### Monitoring Training

With Weights & Biases (automatic):

```yaml
wandb:
  project: neobert-pretraining
  name: neobert-100m-run1
  mode: online  # or offline, disabled
```

Key metrics tracked:

- `train/loss`: MLM loss
- `train/perplexity`: exp(loss)
- `train/learning_rate`: Current LR
- `train/global_step`: Training step
- `eval/loss`: Validation loss (if eval enabled)
- GPU memory usage and throughput

### Checkpointing

Checkpoints are saved in DeepSpeed format:

```
outputs/neobert_100m/
├── model_checkpoints/
│   ├── 10000/
│   │   ├── global_step10000/
│   │   │   ├── mp_rank_00_model_states.pt
│   │   │   └── zero_pp_rank_*_*.pt
│   │   ├── config.yaml
│   │   └── tokenizer/
│   ├── 20000/
│   └── latest -> 20000/
├── logs/
└── wandb/
```

Resume from checkpoint:

```bash
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --resume_from outputs/neobert_100m/model_checkpoints/50000
```

## Fine-Tuning

### GLUE Fine-Tuning

GLUE fine-tuning loads pretrained DeepSpeed checkpoints:

```bash
# Fine-tune on specific GLUE task
python scripts/evaluation/run_glue.py \
    --config configs/glue/cola.yaml

# The config specifies the pretrained checkpoint:
glue:
  pretrained_checkpoint_dir: ./outputs/neobert_100m_100k
  pretrained_checkpoint: 100000  # or "latest"
```

See [Evaluation Guide](evaluation.md) for detailed GLUE instructions.

### Contrastive Learning (SimCSE)

```bash
python scripts/training/contrastive_learning.py \
    --config configs/contrastive_neobert.yaml \
    --model.pretrained_checkpoint_dir outputs/neobert_100m \
    --model.pretrained_checkpoint 100000
```

### Custom Fine-Tuning

Load a pretrained checkpoint for custom tasks:

```python
from src.neobert.model.neobert import NeoBERTForSequenceClassification
from src.neobert.training.utils import load_pretrained_weights

# Initialize model architecture
model = NeoBERTForSequenceClassification(config)

# Load pretrained weights from DeepSpeed checkpoint
checkpoint_dir = "outputs/neobert_100m"
checkpoint_id = 100000  # or "latest"
model = load_pretrained_weights(model, checkpoint_dir, checkpoint_id)

# Now fine-tune on your task
```

## Training Tips

### Memory Optimization

1. **Gradient Accumulation**:

```yaml
trainer:
  gradient_accumulation_steps: 4
  per_device_train_batch_size: 16
  # Effective batch size = 16 * 4 = 64
```

2. **Gradient Checkpointing**:

```yaml
model:
  gradient_checkpointing: true
```

3. **Mixed Precision (Always use BF16)**:

```yaml
trainer:
  mixed_precision: bf16
  tf32: true  # Additional optimization for Ampere+
```

4. **DeepSpeed ZeRO Stages**:

- Stage 1: Optimizer state sharding
- Stage 2: Optimizer + gradient sharding (default)
- Stage 3: Full model sharding (for very large models)

### Performance Optimization

1. **Flash Attention 2**:

```yaml
model:
  flash_attention: true  # Automatic fallback if not available
```

2. **SwiGLU with xFormers**:

```yaml
model:
  hidden_act: swiglu  # Optimized via xformers if available
```

3. **Efficient Data Loading**:

```yaml
trainer:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
```

### Learning Rate Scheduling

Available schedulers:

```yaml
scheduler:
  name: cosine  # or linear, constant, polynomial
  warmup_steps: 10000
  min_lr_ratio: 0.1  # For cosine
```

## Debugging

### Enable Debug Logging

```bash
export NEOBERT_DEBUG=1
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
```

### Common Issues

1. **NaN Loss**:
   - Reduce learning rate: `--optimizer.lr 5e-5`
   - Enable gradient clipping: `--trainer.max_grad_norm 1.0`
   - Check for data issues with debug mode

2. **Slow Training**:
   - Ensure Flash Attention is enabled (check logs)
   - Verify mixed precision is active
   - Check dataloader workers: `--trainer.dataloader_num_workers 4`

3. **Out of Memory**:
   - Reduce batch size
   - Increase gradient accumulation
   - Enable gradient checkpointing
   - Use DeepSpeed ZeRO-3 for model sharding

### Profiling

```python
# Enable PyTorch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = model(**batch)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Testing Your Setup

Before full training, test with a tiny model:

```bash
# Quick test run (5 minutes)
python scripts/pretraining/pretrain.py \
    --config tests/configs/pretraining/test_tiny_pretrain.yaml

# This uses:
# - Tiny model (2 layers, 128 hidden size)
# - Small dataset sample
# - 100 training steps
```

## Dataset Preparation

### Using HuggingFace Datasets

```yaml
dataset:
  name: wikipedia
  config: 20220301.en
  split: train
  streaming: false
```

### Using Custom Text Data

1. Prepare text files:

```bash
data/
├── train.txt
├── validation.txt
└── test.txt
```

2. Configure dataset:

```yaml
dataset:
  name: text
  data_files:
    train: data/train.txt
    validation: data/validation.txt
```

### Tokenization Settings

```yaml
tokenizer:
  name: bert-base-uncased
  max_length: 512
  padding: max_length
  truncation: true
  return_special_tokens_mask: true
```

## Production Training

### Recommended Settings for 100M Model

```yaml
# 100M parameter model
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072

trainer:
  max_steps: 1000000
  per_device_train_batch_size: 64
  gradient_accumulation_steps: 2
  # Effective batch size: 64 * 2 * num_gpus

optimizer:
  lr: 1e-4
  weight_decay: 0.01

scheduler:
  warmup_steps: 10000
```

### Multi-Node Training

For multi-node setups:

```bash
# On each node
torchrun --nproc_per_node=8 \
         --nnodes=2 \
         --node_rank=$RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
```

## Next Steps

- Learn about [Evaluation](evaluation.md) for GLUE and MTEB
- Read [Architecture Guide](architecture.md) for model details
- See [Configuration Guide](configuration.md) for config system
