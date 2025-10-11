# Configuration System

NeoBERT uses a hierarchical configuration system based on dataclasses, YAML files, and command-line overrides.

> [!NOTE]
> See [/configs/README.md](/configs/README.md) for available configuration files and [/scripts/README.md](/scripts/README.md) for script-specific usage.

## Overview

The configuration system provides:

- **Type-safe dataclasses** for all configuration options
- **YAML files** for base configurations
- **Command-line overrides** using dot notation
- **Automatic validation** and type checking
- **Easy extensibility** for new tasks

## Basic Usage

### Using Configuration Files

```bash
# Use a base configuration file
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml

# GLUE evaluation with task-specific config
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml
```

### Command-Line Overrides

Override any configuration value using dot notation:

```bash
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --model.hidden_size 1024 \
    --trainer.per_device_train_batch_size 64 \
    --optimizer.lr 2e-4 \
    --wandb.project my-experiment
```

### Listing Available Options

See all available configuration options:

```bash
python scripts/pretraining/pretrain.py --help
```

## Configuration Structure

### Main Configuration Classes

Located in `src/neobert/config/`:

```python
@dataclass
class Config:
    """Main configuration container"""
    task: str = "pretrain"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    datacollator: DataCollatorConfig = field(default_factory=DataCollatorConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    glue: GLUEConfig = field(default_factory=GLUEConfig)  # For GLUE tasks
```

### Model Configuration

```yaml
model:
  name_or_path: neobert-100m
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  hidden_act: swiglu          # Modern activation
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
  max_position_embeddings: 512
  vocab_size: 30522
  normalization_type: rmsnorm  # RMSNorm instead of LayerNorm
  use_rope: true              # RoPE positional encoding
  rope_theta: 10000
  flash_attention: true       # Enable Flash Attention 2
  gradient_checkpointing: false
```

### Training Configuration

```yaml
trainer:
  output_dir: ./outputs/neobert
  max_steps: 1000000
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 1
  eval_strategy: steps        # or "epoch", "no"
  eval_steps: 10000
  save_strategy: steps        # or "epoch", "best", "no"
  save_steps: 10000
  save_total_limit: 3         # Keep only N best checkpoints
  logging_steps: 100
  mixed_precision: bf16       # Always use bf16
  tf32: true                  # Enable TensorFloat32
  dataloader_num_workers: 4
  seed: 42
  report_to: ["wandb"]        # or ["tensorboard", "none"]

  # GLUE-specific
  num_train_epochs: 3         # For fine-tuning
  early_stopping: 5           # Patience for early stopping
  metric_for_best_model: eval_loss
  greater_is_better: false
  load_best_model_at_end: true
```

### Dataset Configuration

```yaml
dataset:
  # HuggingFace dataset
  name: wikipedia
  config: 20220301.en
  split: train

  # Or custom files
  data_files:
    train: data/train.txt
    validation: data/val.txt

  # Streaming for large datasets
  streaming: true
  shuffle_buffer_size: 10000

  # Processing options
  max_seq_length: 512
  mlm_probability: 0.15
  preprocessing_num_workers: 8
  overwrite_cache: false
```

### Optimizer Configuration

```yaml
optimizer:
  name: adamw               # or "adam", "sgd", "adafactor"
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8

  # Optional gradient clipping
  max_grad_norm: 1.0
```

### Scheduler Configuration

```yaml
scheduler:
  name: cosine              # or "linear", "constant", "polynomial"
  warmup_steps: 10000       # Or use warmup_percent
  warmup_percent: null      # Alternative to warmup_steps
  min_lr_ratio: 0.1         # For cosine scheduler
  num_cycles: 0.5           # For cosine with restarts
```

### GLUE Configuration

For GLUE tasks, additional configuration:

```yaml
glue:
  task_name: cola           # GLUE task identifier
  num_labels: 2             # Set automatically based on task
  max_seq_length: 128

  # Loading pretrained checkpoints
  pretrained_checkpoint_dir: ./outputs/neobert_100m_100k
  pretrained_checkpoint: 100000  # or "latest"
  pretrained_config_path: ./outputs/neobert_100m_100k/model_checkpoints/100000/config.yaml

  # Testing options
  allow_random_weights: false  # For testing without pretrained model
```

## Configuration Files

### Directory Structure

```
configs/
├── pretrain_neobert.yaml       # Main pretraining config
├── streaming_pretrain.yaml     # Streaming dataset example
├── evaluate_neobert.yaml       # MTEB evaluation config
├── glue/                        # GLUE task configs
│   ├── cola.yaml
│   ├── sst2.yaml
│   ├── mrpc.yaml
│   ├── stsb.yaml
│   ├── qqp.yaml
│   ├── mnli.yaml
│   ├── qnli.yaml
│   ├── rte.yaml
│   └── wnli.yaml
└── tests/                       # Test configurations
    └── test_tiny_pretrain.yaml  # Tiny model for testing
```

### Example: Pretraining Config

```yaml
# configs/pretrain_neobert.yaml
task: pretrain

model:
  name_or_path: neobert-100m
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072

trainer:
  output_dir: ./outputs/neobert_100m
  max_steps: 1000000
  per_device_train_batch_size: 32
  mixed_precision: bf16

dataset:
  name: wikipedia
  streaming: true
  max_seq_length: 512

optimizer:
  lr: 1e-4
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_steps: 10000
```

### MuonClip Optimizer Options

MuonClip extends the standard optimizer block with an optional `muon_config`
section. If the section is omitted the defaults below are applied; if it is
present while `optimizer.name` is not `muonclip`, NeoBERT will warn and ignore
it. See `configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml`
for a full example.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-4` | Shared learning rate for Muon + Adam sub-optimizers |
| `muon_beta` | `0.95` | Momentum coefficient for 2D weight matrices |
| `betas` | `[0.9, 0.98]` | Adam β values for non-matrix params (β₂=0.98 aligns with encoder baselines) |
| `ns_steps` | `5` | Newton–Schulz iterations used to orthogonalize Muon updates |
| `muon_decay` / `weight_decay` | `0.0` / `0.01` | Independent weight decay for Muon vs. Adam parameter sets |
| `enable_clipping` | `true` | Toggle QK-clipping (attention logit rescaling) |
| `clipping_threshold` | `50.0` | Maximum allowed attention logit before rescaling |
| `clipping_alpha` | `0.5` | Balance of scaling between Q (α) and K (1-α) |
| `clipping_warmup_steps` | `0` | Delay clipping for the first *n* steps |
| `clipping_layers_mapping` | `{}` | Optional mapping for models with separate `q_proj`/`k_proj` names |
| `monitor_attention_entropy` | `false` | Collect attention entropy statistics each step |
| `log_max_logits` | `false` | Track maximum attention logits for logging |
| `offload_hooks_to_cpu` | `false` | Move collected attention stats off GPU between steps |
| `enable_profiling` | `false` | Emit per-step profiling diagnostics |
| `log_interval` | `100` | Metric sampling interval during warmup-only logging |
| `log_dir` | `null` | If set, metrics are appended to `<log_dir>/muonclip_metrics.jsonl` |

> Note: Earlier MuonClip experiments used `betas=(0.9, 0.95)`, which can be too
> aggressive for encoder pretraining. We default to β₂=0.98 and recommend
> adjusting only if you have calibration data to justify it.

### Example: GLUE Config

```yaml
# configs/glue/cola.yaml
task: glue

model:
  name_or_path: neobert-100m
  pretrained_checkpoint_dir: ./outputs/neobert_100m_100k
  pretrained_checkpoint: 100000

glue:
  task_name: cola
  num_labels: 2
  max_seq_length: 128

trainer:
  output_dir: ./outputs/glue/neobert-100m/cola
  num_train_epochs: 3
  per_device_train_batch_size: 32
  eval_strategy: steps
  eval_steps: 50
  early_stopping: 5
  metric_for_best_model: eval_matthews_correlation

optimizer:
  lr: 2e-5

scheduler:
  name: linear
  warmup_percent: 10
```

## Advanced Features

### Dynamic Configuration Loading

Load configurations programmatically:

```python
from src.neobert.config import load_config

# Load from YAML
config = load_config("configs/pretrain_neobert.yaml")

# Override values
config.model.hidden_size = 1024
config.trainer.max_steps = 500000

# Access as attributes
print(config.model.hidden_size)  # 1024
print(config.optimizer.lr)       # 1e-4
```

### Custom Configuration Classes

Extend the configuration system:

```python
from dataclasses import dataclass, field
from src.neobert.config import Config

@dataclass
class CustomTaskConfig:
    """Custom task configuration"""
    custom_param: str = "default"
    custom_value: int = 42

@dataclass
class ExtendedConfig(Config):
    """Extended configuration with custom task"""
    custom_task: CustomTaskConfig = field(default_factory=CustomTaskConfig)
```

### Configuration Validation

The system automatically validates:

- Type correctness (int, float, str, bool, list)
- Required fields
- Value ranges (where specified)
- Mutual exclusivity (e.g., warmup_steps vs warmup_percent)

### Environment Variables

Override configurations via environment variables:

```bash
export NEOBERT_DEBUG=1
export WANDB_PROJECT=my-project
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
```

## Best Practices

### 1. Use YAML for Base Configurations

Keep stable, reusable configurations in YAML files:

- Model architectures
- Standard hyperparameters
- Dataset specifications

### 2. Use CLI for Experiments

Override via command line for experimentation:

- Learning rates
- Batch sizes
- Output directories
- Random seeds

### 3. Organize by Task

Structure configs by task type:

```
configs/
├── pretraining/
├── glue/
├── mteb/
└── custom_tasks/
```

### 4. Version Control Configs

Track configuration files in git to ensure reproducibility:

```bash
git add configs/experiment_v1.yaml
git commit -m "Add experiment v1 configuration"
```

### 5. Document Custom Configs

Add comments in YAML files:

```yaml
model:
  hidden_size: 768  # Standard BERT-base size
  num_hidden_layers: 12
  num_attention_heads: 12
  # Using modern improvements
  hidden_act: swiglu  # Better than GELU
  normalization_type: rmsnorm  # More efficient than LayerNorm
```

## Troubleshooting

### Common Issues

1. **"Unknown configuration key"**
   - Check spelling of configuration keys
   - Ensure using correct nesting level

2. **"Type error in configuration"**
   - YAML interprets some values incorrectly
   - Use quotes for strings that look like numbers: `"1e-4"`
   - Use explicit lists: `[0.9, 0.999]` not `0.9, 0.999`

3. **"Configuration conflict"**
   - Some options are mutually exclusive
   - E.g., can't use both `warmup_steps` and `warmup_percent`

### Debugging Configuration

Print loaded configuration:

```python
# In your script
print(config)  # Shows full configuration

# Or specific sections
print(config.model)
print(config.trainer)
```

Check configuration in logs:

```bash
# Logs show configuration at start
grep "Configuration:" logs/training.log
```

## Next Steps

- See [Training Guide](training.md) for pretraining configurations
- See [Evaluation Guide](evaluation.md) for GLUE/MTEB configurations
- Check example configs in `configs/` directory
