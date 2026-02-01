# Training Guide

This guide covers pretraining and fine-tuning NeoBERT models. It focuses on **how to launch runs**; configuration details live in [docs/configuration.md](configuration.md).

> [!TIP]
> Use [scripts/README.md](../scripts/README.md) as the directory map for CLI entry points.

## Script Entry Points

| Script                              | Purpose                               | Reference                                     |
| ----------------------------------- | ------------------------------------- | --------------------------------------------- |
| `scripts/pretraining/pretrain.py`   | Pretraining (MLM)                     | This guide (Pretraining, Checkpointing, Tips) |
| `scripts/pretraining/preprocess.py` | Tokenize raw corpora into shards      | Dataset Preparation                           |
| `scripts/pretraining/longer_seq.py` | Continue training with longer context | Pretraining Tips                              |
| `scripts/contrastive/download.py`   | Fetch contrastive datasets            | Contrastive Learning                          |
| `scripts/contrastive/preprocess.py` | Normalize/tokenize contrastive data   | Contrastive Learning                          |
| `scripts/contrastive/finetune.py`   | Contrastive fine-tuning               | Contrastive Learning                          |
| `scripts/evaluation/run_glue.py`    | GLUE evaluation                       | [Evaluation Guide](evaluation.md)             |
| `scripts/evaluation/run_mteb.py`    | MTEB evaluation                       | [Evaluation Guide](evaluation.md)             |
| `scripts/export-hf/export.py`       | Export to Hugging Face                | [Export Guide](export.md)                     |

## Pretraining

### Basic Pretraining

```bash
# Standard pretraining config
python scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert.yaml

# Override a few settings
python scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert.yaml \
  --trainer.per_device_train_batch_size 64 \
  --optimizer.lr 2e-4 \
  --trainer.max_steps 100000
```

### Configuration Highlights

See [docs/configuration.md](configuration.md) for the full schema. Key knobs used during pretraining:

- `model.*`: architecture (e.g., `hidden_size`, `num_hidden_layers`, `rope`, `rms_norm`, `flash_attention`)
- `dataset.*`: dataset source, streaming, splits, and tokenization settings
- `tokenizer.*`: tokenizer name/path and max length
- `datacollator.*`: MLM masking and padding (`mlm_probability`, `pad_to_multiple_of`)
- `trainer.*`: batch size, steps, logging, checkpointing
- `optimizer.*` / `scheduler.*`: learning rate and scheduling
- `wandb.*`: tracking controls

> [!NOTE]
> Pretraining uses the **top-level** `mixed_precision` field. Other training scripts read `trainer.mixed_precision`. To keep behavior consistent across scripts, set both.

Example (minimal):

```yaml
mixed_precision: bf16
trainer:
  mixed_precision: bf16
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  gradient_checkpointing: false
  gradient_clipping: null
```

### Streaming Datasets

Enable streaming in the dataset block:

```yaml
dataset:
  name: "wikipedia"
  streaming: true
  max_seq_length: 512
  shuffle_buffer_size: 10000
  num_workers: 0
```

### Checkpointing

Pretraining checkpoints are written under `trainer.output_dir/model_checkpoints/`:

```
outputs/neobert_pretrain/
└── model_checkpoints/
    ├── 10000/
    │   ├── state_dict.pt
    │   └── config.yaml
    ├── 20000/
    └── latest
```

Resume from a checkpoint:

```bash
python scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert.yaml \
  --trainer.resume_from_checkpoint outputs/neobert_pretrain/model_checkpoints/20000
```

## Contrastive Learning

```bash
python scripts/contrastive/finetune.py \
  --config configs/contrastive/contrastive_neobert.yaml
```

For dataset setup, see the configs under `configs/contrastive/` and the scripts in `scripts/contrastive/`.

## Dataset Preparation

### Pre-tokenize data

```bash
python scripts/pretraining/preprocess.py \
  --config configs/pretraining/pretrain_neobert.yaml
```

Tokenizer settings are taken from `tokenizer.*`; the output path comes from `dataset.path`.

## Training Tips

- **Gradient checkpointing**: `trainer.gradient_checkpointing: true`
- **Gradient clipping**: `trainer.gradient_clipping: 1.0`
- **Mixed precision**: set `mixed_precision: bf16` (and `trainer.mixed_precision: bf16`)
- **Flash attention**: `model.flash_attention: true` requires `xformers`

## Multi-GPU Launches

The training scripts use `accelerate` under the hood. For multi-GPU runs, use `accelerate launch`:

```bash
accelerate launch scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert.yaml \
  --trainer.per_device_train_batch_size 16 \
  --trainer.gradient_accumulation_steps 4
```

## Next Steps

- Evaluation recipes: [docs/evaluation.md](evaluation.md)
- Config reference: [docs/configuration.md](configuration.md)
- Export to HF: [docs/export.md](export.md)
