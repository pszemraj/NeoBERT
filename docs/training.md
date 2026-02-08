# Training Guide

This guide covers pretraining and contrastive workflows. Full field-level schema
is in [configuration.md](configuration.md).

## Entry Points

| Script                                    | Purpose                           |
| ----------------------------------------- | --------------------------------- |
| `scripts/pretraining/pretrain.py`         | MLM pretraining                   |
| `scripts/pretraining/preprocess.py`       | tokenize and save dataset to disk |
| `scripts/pretraining/tokenize_dataset.py` | standalone tokenization helper    |
| `scripts/pretraining/longer_seq.py`       | continue run at longer context    |
| `scripts/contrastive/finetune.py`         | contrastive fine-tuning           |
| `scripts/contrastive/preprocess.py`       | contrastive dataset preprocessing |

## Pretraining

### Basic launch

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert.yaml
```

### Override selected knobs

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert.yaml \
  --trainer.per_device_train_batch_size 16 \
  --trainer.gradient_accumulation_steps 4 \
  --trainer.max_steps 100000
```

## Packed Training

Enable packing via `datacollator.pack_sequences: true`.

Recommended for throughput:

- `model.attn_backend: flash_attn_varlen`
- install flash-attn (`pip install -e .[flash]`)

Supported but slower:

- `pack_sequences: true` with `attn_backend: sdpa` uses segmented fallback.

Useful control:

- `trainer.enforce_full_packed_batches: true` keeps full microbatches by
  buffering undersized packed outputs (better token throughput stability,
  typically lower step/s).

## Dataloader Throughput Knobs

Primary knobs are in `dataset.*`:

- `num_workers`
- `pin_memory`
- `persistent_workers`
- `prefetch_factor`

When running on CUDA, trainer may warn and apply throughput-friendly defaults
if these are unset/suboptimal.

## Streaming Eval Strategy

For streaming datasets, prefer:

- `dataset.eval_split: null`
- `dataset.eval_samples: <small integer>`

Runtime behavior:

- if `dataset.eval_split` is unset, trainer tries to auto-detect a validation-style
  split (`validation`, `eval`, `test`, `dev`);
- if none exists and `dataset.eval_samples` is set, trainer reserves the first
  `eval_samples` from train for eval and skips them from training to avoid
  leakage;
- when `trainer.eval_max_batches` is unset, trainer derives a practical default
  for streaming eval and still allows explicit override.

## Mixed Precision and Compile

- `trainer.mixed_precision`: `no | fp32 | bf16` (`fp16` unsupported in pretraining)
- `trainer.torch_compile`: enable `torch.compile`
- `trainer.torch_compile_backend`: `inductor | aot_eager | eager`
- `trainer.torch_compile_dynamic`: optional override for dynamic-shape compile;
  default behavior prefers static-shape compile for stability.
- `trainer.masked_logits_only_loss`: `true | false`

## MLM Loss Path Selection

Use exactly one pretraining loss path per run:

- `trainer.masked_logits_only_loss: true`
  Uses masked-logits-only MLM loss (default and recommended). This avoids full
  `(B,S,V)` logits materialization in the hot pretraining path.
- `trainer.masked_logits_only_loss: false`
  Uses the original NeoBERT full-logits CE path (legacy ablation/debug path).

There is no mixed/cross objective mode in trainer config; this flag picks one
path for the run.

Current project default is `true`; new pretraining runs should keep
`masked_logits_only_loss: true` unless you are intentionally running an
ablation against the legacy baseline.

## Checkpointing and Resume

Model checkpoints (export/inference assets):

```text
<output_dir>/model_checkpoints/<step>/
  model.safetensors
  config.yaml
  tokenizer_info.json
  tokenizer/
```

Accelerator state checkpoints (resume source of truth):

```text
<output_dir>/checkpoints/<step>/
```

Resume examples:

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert.yaml \
  --trainer.resume_from_checkpoint latest
```

Notes:

- resume operates from `<output_dir>/checkpoints/`.
- for streaming datasets, exact data position is not preserved, so resume is not
  fully reproducible for data order.

## Pre-tokenized Datasets

Two common paths:

1. Offline preprocess via config:

```bash
python scripts/pretraining/preprocess.py \
  configs/pretraining/pretrain_neobert.yaml
```

1. Standalone tokenizer helper:

```bash
python scripts/pretraining/tokenize_dataset.py \
  --dataset EleutherAI/SmolLM2-1.7B-stage-4-100B \
  --output tokenized_data/smollm2 \
  --tokenizer BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp \
  --max-length 1024
```

## Contrastive

```bash
python scripts/contrastive/finetune.py \
  configs/contrastive/contrastive_neobert.yaml
```

Ensure `dataset.path` points to output from `scripts/contrastive/preprocess.py`.

## Practical Tips

- Use `gradient_checkpointing` for memory headroom on long contexts.
- Use `gradient_clipping` for stability on deep/long runs.
- For paper-style NeoBERT masking strategy, set `datacollator.mask_all: true`.
  Default `false` uses standard 80/10/10 MLM corruption.
- For packed + compile runs, measure `tokens/sec` rather than `steps/sec`.

## Related Docs

- [Configuration](configuration.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
