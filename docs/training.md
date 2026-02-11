# Training Guide

This guide covers pretraining and contrastive workflows.
It is the canonical source for training runtime behavior. Full field-level
schema/defaults are in [configuration.md](configuration.md).

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
- if `trainer.eval_max_batches` is unset, trainer derives the eval budget from
  `dataset.eval_samples` and `trainer.per_device_eval_batch_size`;
- if neither `trainer.eval_max_batches` nor `dataset.eval_samples` is set,
  trainer raises an error (explicit eval budget required for streaming eval).
- if no eval dataset can be resolved, eval is skipped.

## Mixed Precision and Compile

- `trainer.mixed_precision`: `no | fp32 | bf16` (`bf16` recommended default)
- runtime normalization: `fp32 -> no`, `true -> bf16`, `false -> no`
- `fp16` is unsupported in NeoBERT training paths
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

Unified pretraining checkpoints (resume + export assets):

```text
<output_dir>/checkpoints/<step>/
  model.safetensors
  optimizer.bin / scheduler.bin / random_states_*.pkl
  custom_checkpoint_*.pkl
  config.yaml
  tokenizer_info.json
  tokenizer/
```

Resume examples:

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert.yaml \
  --trainer.resume_from_checkpoint latest
```

### Crash Recovery Playbook (step 69,420 example)

Scenario:

- you launched `configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml`
- run crashed around step `69420`
- target is to continue to `trainer.max_steps: 100000`

1. Find the last saved checkpoint step:

```bash
find outputs/neobert-100m-wordpc_msp_32k_tok-muonclip/checkpoints \
  -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort -n | tail -n 1
```

2. Resume from latest checkpoint:

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml \
  --trainer.resume_from_checkpoint latest
```

3. Confirm resume in logs:

- startup prints checkpoint loading from
  `outputs/.../checkpoints/<step>/`
- first train progress resumes from prior global step (not step 0)
- run continues until `trainer.max_steps` (100000 in this config)

Important for this exact config:

- `pretrain_neobert100m_smollm2data_muonclip.yaml` sets
  `dataset.streaming: true`
- streaming resume is supported as best-effort recovery:
  trainer restores checkpoint state and advances the stream by consumed batches
  before continuing

Streaming resume caveat:

- stream advancement can take significant wall time for late checkpoints
- with shuffled streams, exact sample continuity is not guaranteed

For strict deterministic continuation, switch to a non-streaming tokenized
dataset and resume there:

```bash
# one-time tokenize-to-disk (example path)
python scripts/pretraining/tokenize_dataset.py \
  --dataset EleutherAI/SmolLM2-1.7B-stage-4-100B \
  --output tokenized_data/smollm2_32k \
  --tokenizer BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp \
  --max-length 1024

# resume with streaming disabled and local dataset path
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml \
  --dataset.streaming false \
  --dataset.path tokenized_data/smollm2_32k \
  --trainer.resume_from_checkpoint latest
```

This resumes model/optimizer/scheduler states, but data order will not exactly
match the interrupted streaming run.

Notes:

- resume and export both operate from `<output_dir>/checkpoints/`.
- pretraining resume with `dataset.streaming: true` uses best-effort stream
  advancement based on saved batch counters.
- for exact deterministic continuation, prefer pre-tokenized
  `dataset.streaming: false` runs.

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
  Default `false` uses sampled-token 80/10/10 corruption.
  For `p = datacollator.mlm_probability`, global token mix is:
  `(1 - p)` untouched, `0.8p` `[MASK]`, `0.1p` random-token, `0.1p` original-token.
- For packed + compile runs, measure `tokens/sec` rather than `steps/sec`.

## Related Docs

- [Configuration](configuration.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
