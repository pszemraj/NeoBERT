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
- when `trainer.eval_max_batches` is unset, trainer derives a practical default
  for streaming eval and still allows explicit override.
- if no eval dataset can be resolved, eval is skipped.

## Mixed Precision and Compile

- `trainer.mixed_precision`: `no | fp32 | bf16` (`fp16` unsupported in pretraining)
- `trainer.torch_compile`: enable `torch.compile`
- `trainer.torch_compile_backend`: `inductor | aot_eager | eager`
- `trainer.torch_compile_dynamic`: optional override for dynamic-shape compile;
  default behavior prefers static-shape compile for stability.
- `trainer.masked_logits_only_loss`: `true | false`

## TorchAO Low-Precision Pretraining

NeoBERT can apply TorchAO conversion before `torch.compile`/`accelerator.prepare`.
Configuration lives under `torchao.*`.

Example (float8 rowwise):

```yaml
trainer:
  torch_compile: true

torchao:
  enable: true
  recipe: "float8_rowwise"
  skip_first_last_linear: true
  auto_filter_small_kn: false
  filter_fqns: ["decoder"]
```

Example (tensorwise float8 + FSDP precompute hook):

```yaml
trainer:
  torch_compile: true

torchao:
  enable: true
  recipe: "float8_tensorwise"
  enable_fsdp_float8_all_gather: true
  precompute_float8_dynamic_scale_for_fsdp: true
```

Notes:

- Supported recipe families: float8 (`tensorwise`, `rowwise`, `rowwise_with_gw_hp`),
  MXFP8/MXFP4 (`mxfp8_emulated`, `mxfp8_cublas`, `mxfp8_cublas_rceil`,
  `mxfp4_cutlass`, `mxfp4_emulated`), and prototype FP4 QAT recipes
  (`nvfp4_qat`, `mxfp4_qat`).
- The runtime prefers the existing `accelerate.utils.ao.convert_model_to_fp8_ao`
  helper for float8 when available and falls back to direct TorchAO conversion.
- For narrower hidden sizes (for example 768), `float8_rowwise` with
  `auto_filter_small_kn: true` can filter out all linear layers. Set
  `auto_filter_small_kn: false` to force conversion when needed.
- Keep `trainer.torch_compile: true` for best performance/stability. The default
  `torchao.require_compile: true` enforces this.
- Current guardrails: pretraining-only path; DeepSpeed + TorchAO is blocked.

## Transformer Engine Low-Precision Pretraining

NeoBERT can also apply Transformer Engine conversion before
`torch.compile`/`accelerator.prepare`. Configuration lives under
`transformer_engine.*`.

Example (NVFP4 for FP4-capable path):

```yaml
trainer:
  torch_compile: true

transformer_engine:
  enable: true
  recipe: "nvfp4"
  filter_fqns: ["decoder"]
  skip_first_last_linear: true
  require_compile: true
```

Example (MXFP8):

```yaml
trainer:
  torch_compile: true

transformer_engine:
  enable: true
  recipe: "mxfp8"
  filter_fqns: ["decoder"]
  skip_first_last_linear: true
  require_compile: true
  fp8_format: "E4M3"
```

Notes:

- Supported recipe families: `fp8_delayed`, `fp8_current`, `mxfp8`, and `nvfp4`.
- Backend selection is explicit: enable only one of `torchao.enable` or
  `transformer_engine.enable` (or `quartet2.enable`).
- The runtime wraps model forward in TE autocast and prefers Accelerate's
  helper when available.
- Current guardrails: pretraining-only path; DeepSpeed + Transformer Engine is
  blocked.
- For `recipe: "nvfp4"` on packed-sequence pretraining, prefer conservative
  settings first:
  `transformer_engine.disable_2d_quantization: true`,
  `transformer_engine.disable_rht: true`,
  `transformer_engine.disable_stochastic_rounding: true`.
  If kernel launch errors persist, set `trainer.torch_compile: false` and
  `transformer_engine.require_compile: false`.

## Quartet-II Low-Precision Pretraining

NeoBERT can also apply Quartet-II NVFP4 linears before
`torch.compile`/`accelerator.prepare`. Configuration lives under `quartet2.*`.

Example:

```yaml
trainer:
  torch_compile: true
  mixed_precision: "bf16"

quartet2:
  enable: true
  recipe: "quartet_ii"
  filter_fqns: ["decoder"]
  skip_first_last_linear: true
  required_dim_multiple: 128
  four_over_six: true
  require_compile: true
```

Notes:

- Quartet-II uses the `quartet2.linear.Quartet_II_linear` drop-in linear.
- Runtime backend selection is explicit: enable only one of `torchao.enable`,
  `transformer_engine.enable`, or `quartet2.enable`.
- Current guardrails: pretraining-only path; DeepSpeed + Quartet-II is blocked.
- Hardware/runtime requirements follow upstream kernels: CUDA, Blackwell-class
  GPUs (`sm120a`), and BF16 mixed precision.
- Install dependencies with: `pip install --no-build-isolation -e .[quant_quartet]`.
- `qutlass` is a separate dependency; install from GitHub (not PyPI):
  `pip install --no-build-isolation "qutlass @ git+https://github.com/IST-DASLab/qutlass.git"`.

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
- pretraining resume with `dataset.streaming: true` is rejected by trainer.
  Use a pre-tokenized non-streaming dataset for resumable runs.

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
