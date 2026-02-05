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
  configs/pretraining/pretrain_neobert.yaml

# Override a few settings
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert.yaml \
  --trainer.per_device_train_batch_size 64 \
  --optimizer.lr 2e-4 \
  --trainer.max_steps 100000
```

### CLI-only Smoke Run (No Custom YAML)

```bash
# Short smoke test without writing a new config file
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_smoke.yaml \
  --dataset.streaming false \
  --datacollator.pack_sequences false \
  --trainer.max_steps 50 \
  --wandb.mode offline
```

### Configuration Highlights

See [docs/configuration.md](configuration.md) for the full schema. Key knobs used during pretraining:

- `model.*`: architecture (e.g., `hidden_size`, `num_hidden_layers`, `rope`, `rms_norm`, `attn_backend`, `kernel_backend`)
- `dataset.*`: dataset source, streaming, splits, and tokenization settings
- `tokenizer.*`: tokenizer name/path and max length
- `datacollator.*`: MLM masking, padding, and packing (`mlm_probability`, `mask_all`, `pad_to_multiple_of`, `pack_sequences`)
- `trainer.*`: batch size, steps, logging, checkpointing
- `optimizer.*` / `scheduler.*`: learning rate and scheduling
- `wandb.*`: tracking controls

Note: `datacollator.pack_sequences` requires `model.attn_backend: flash_attn_varlen` and
`flash-attn` installed. It is experimental. See `docs/dev.md`.

> [!NOTE]
> All training entrypoints read `trainer.mixed_precision`. Use that field for consistency.

Example (minimal):

```yaml
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

> [!IMPORTANT]
> Streaming datasets cannot resume from checkpoints because data position is not preserved.
> For resumable runs, pre-tokenize the dataset and disable streaming (see "Pre-tokenize data" below).

### Checkpointing

Pretraining **model checkpoints** are written under `trainer.output_dir/model_checkpoints/`:

```
outputs/neobert_pretrain/
└── model_checkpoints/
    ├── 10000/
    │   ├── state_dict.pt
    │   └── config.yaml
    ├── 20000/
    └── latest
```

Accelerator **training state** (for resumable runs) is written under `trainer.output_dir/checkpoints/`:

```
outputs/neobert_pretrain/
└── checkpoints/
    ├── 10000/
    ├── 20000/
    └── 30000/
```

Resume from the latest accelerator checkpoint:

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert.yaml \
  --trainer.resume_from_checkpoint true
```

Notes:
- `trainer.resume_from_checkpoint` may be `true`/`"latest"`/`"auto"` to resume the newest
  checkpoint under `output_dir/checkpoints`, or a string path to load directly.
- The trainer resumes from the **latest numeric directory** under `output_dir/checkpoints/`.
- `trainer.output_dir` must point at the original run directory so checkpoints can be found.
- To resume from a specific step, remove newer checkpoint directories first.

## Contrastive Learning

```bash
python scripts/contrastive/finetune.py \
  configs/contrastive/contrastive_neobert.yaml
```

For dataset setup, see the configs under `configs/contrastive/` and the scripts in `scripts/contrastive/`. Contrastive training requires `dataset.path` to point to the preprocessed dataset produced by `scripts/contrastive/preprocess.py`.

## Dataset Preparation

### Pre-tokenize data

```bash
python scripts/pretraining/preprocess.py \
  configs/pretraining/pretrain_neobert.yaml
```

Tokenizer settings are taken from `tokenizer.*`; the output path comes from `dataset.path`.

Alternatively, set `dataset.pre_tokenize: true` in your pretraining config to have the trainer invoke
`scripts/pretraining/tokenize_dataset.py` automatically. It will save to `dataset.pre_tokenize_output` if set,
otherwise to `tokenized_data/<dataset-name>/`.

## Training Tips

- **Gradient checkpointing**: `trainer.gradient_checkpointing: true`
- **Gradient clipping**: `trainer.gradient_clipping: 1.0`
- **Mixed precision**: set `trainer.mixed_precision: bf16`
- **torch.compile**: set `trainer.torch_compile: true` (skips automatically with DeepSpeed or if `torch.compile` is unavailable)
- **Packed attention**: `model.attn_backend: flash_attn_varlen` requires `flash-attn` and right-padded (or packed) inputs

## Multi-GPU Launches

The training scripts use `accelerate` under the hood. For multi-GPU runs, use `accelerate launch`:

```bash
accelerate launch scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert.yaml \
  --trainer.per_device_train_batch_size 16 \
  --trainer.gradient_accumulation_steps 4
```

## Next Steps

- Evaluation recipes: [docs/evaluation.md](evaluation.md)
- Config reference: [docs/configuration.md](configuration.md)
- Export to HF: [docs/export.md](export.md)
