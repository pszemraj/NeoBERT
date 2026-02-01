# Configuration System

NeoBERT uses dataclasses backed by YAML files plus dot-notation CLI overrides.

> [!TIP]
> For available configuration files see [configs/README.md](../configs/README.md). Use [scripts/README.md](../scripts/README.md) to locate the right CLI entry point.

## Basic Usage

```bash
# Pretraining
python scripts/pretraining/pretrain.py --config configs/pretraining/pretrain_neobert.yaml

# GLUE evaluation
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml
```

Override values with dot notation:

```bash
python scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert.yaml \
  --model.hidden_size 1024 \
  --trainer.per_device_train_batch_size 64 \
  --optimizer.lr 2e-4
```

Show available CLI overrides:

```bash
python scripts/pretraining/pretrain.py --help
```

## Source of Truth

Configuration dataclasses live in `src/neobert/config.py`. Only **known** keys are applied; unknown keys in YAML files are ignored. Use this file to check the current schema.

## Top-Level Fields

```yaml
task: pretraining  # pretraining, glue, mteb, contrastive
mixed_precision: bf16  # used by pretraining
accelerate_config_file: null
pretrained_checkpoint: latest
use_deepspeed: true
mteb_task_type: all
mteb_batch_size: 32
mteb_pooling: mean
mteb_overwrite_results: false
seed: 0
debug: false
```

> [!NOTE]
> Pretraining reads the **top-level** `mixed_precision`. Other scripts read `trainer.mixed_precision`. To keep behavior consistent, set both.

## Section Reference

### Model (`model`)

```yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 4096
  vocab_size: 30522
  rope: true
  rms_norm: true
  hidden_act: swiglu
  dropout_prob: 0.0
  norm_eps: 1e-5
  flash_attention: true
  ngpt: false
  pad_token_id: 0
```

### Dataset (`dataset`)

```yaml
dataset:
  name: wikipedia
  path: null
  streaming: true
  cache_dir: null
  max_seq_length: 512
  train_split: "train"
  eval_split: "validation"
  validation_split: null
  num_workers: 4
  num_proc: 4
  shuffle_buffer_size: 10000
  pre_tokenize: false
  pre_tokenize_output: null
```

### Tokenizer (`tokenizer`)

```yaml
tokenizer:
  name: bert-base-uncased
  path: null
  max_length: 512
  padding: max_length
  truncation: true
  vocab_size: null
```

### Trainer (`trainer`)

```yaml
trainer:
  output_dir: ./outputs/neobert
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 1
  max_steps: 1000000
  num_train_epochs: 3
  eval_strategy: steps
  eval_steps: 10000
  save_strategy: steps
  save_steps: 10000
  save_total_limit: 3
  logging_steps: 100
  gradient_checkpointing: false
  gradient_clipping: null
  mixed_precision: bf16
  resume_from_checkpoint: null
  tf32: true
  dataloader_num_workers: 0
  report_to: []
```

### Optimizer (`optimizer`)

```yaml
optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
  muon_config: null
```

### Scheduler (`scheduler`)

```yaml
scheduler:
  name: cosine
  warmup_steps: 10000
  total_steps: 1000000
  num_cycles: 0.5
  decay_steps: 50000
  warmup_percent: null
  decay_percent: null
```

### Data Collator (`datacollator`)

```yaml
datacollator:
  mlm_probability: 0.15
  pad_to_multiple_of: 8
```

### Weights & Biases (`wandb`)

```yaml
wandb:
  project: neobert
  entity: null
  name: null
  tags: []
  mode: online  # or offline, disabled
  log_interval: 100
  resume: never
  dir: logs/wandb
```

### GLUE (`glue`)

```yaml
glue:
  task_name: cola
  num_labels: 2
  max_seq_length: 128
  pretrained_checkpoint_dir: ./outputs/neobert_pretrain
  pretrained_checkpoint: latest
  allow_random_weights: false
```

GLUE configs often include extra fields under `model` (e.g., `pretrained_config_path`, `pretrained_checkpoint_dir`) that are read via the raw model block in `src/neobert/glue/train.py`.

### Contrastive (`contrastive`)

```yaml
contrastive:
  temperature: 0.05
  pooling: avg  # avg, cls, max
  loss_type: simcse  # simcse, supcon
  hard_negative_weight: 0.0
```

## MuonClip Optimizer Options

MuonClip extends the optimizer block with an optional `muon_config` section. If the section is omitted the defaults below are applied; if it is present while `optimizer.name` is not `muonclip`, NeoBERT will warn and ignore it. See `configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml` for a full example.

| Parameter | Default | Description |
| --- | --- | --- |
| `muon_beta` | `0.95` | Momentum coefficient for 2D weight matrices |
| `muon_decay` | `0.0` | Weight decay for Muon parameters |
| `ns_steps` | `5` | Newton-Schulz iterations for orthogonalization |
| `orthogonalization` | `polar_express` | Orthogonalization algorithm |
| `enable_clipping` | `true` | Toggle QK-clipping |
| `clipping_threshold` | `50.0` | Max attention logit before rescaling |
| `clipping_alpha` | `0.5` | Balance scaling between Q and K |
| `clipping_warmup_steps` | `0` | Delay clipping for first *n* steps |
| `clipping_layers_mapping` | `{}` | Mapping for separate q/k proj names |
| `detect_anomalies` | `false` | Enable PyTorch anomaly detection |

## Config Directory Layout

```
configs/
├── pretraining/
├── glue/
├── contrastive/
└── README.md
```

Tiny smoke-test configs live in `tests/configs/`.

## Programmatic Loading

```python
from neobert.config import ConfigLoader

config = ConfigLoader.load("configs/pretraining/pretrain_neobert.yaml")
print(config.model.hidden_size)
```

## Next Steps

- Training workflows: [docs/training.md](training.md)
- Evaluation recipes: [docs/evaluation.md](evaluation.md)
- Export to HF: [docs/export.md](export.md)
