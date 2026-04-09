# Training Guide

This guide covers pretraining and contrastive workflows.
Full field-level schema/defaults are in [configuration.md](configuration.md).

## Entry Points

| Script                                    | Purpose                           |
| ----------------------------------------- | --------------------------------- |
| `scripts/pretraining/pretrain.py`         | MLM pretraining                   |
| `scripts/pretraining/preprocess.py`       | tokenize and save dataset to disk |
| `scripts/pretraining/tokenize_dataset.py` | standalone tokenization helper    |
| `scripts/pretraining/longer_seq.py`       | continue run at longer context    |
| `scripts/contrastive/finetune.py`         | contrastive fine-tuning           |
| `scripts/contrastive/preprocess.py`       | contrastive dataset preprocessing |

For contrastive preprocessing, `dataset.name` may be omitted, `ALL`, a
canonical key such as `ALLNLI`, or an HF dataset ID alias such as
`sentence-transformers/all-nli`, `embedding-data/QQP_triplets`, or
`WhereIsAI/github-issue-similarity`. Both preprocessing and contrastive
training load only the requested cached splits from `all/`; other cached split
directories may remain on disk for later reuse.

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

### Distributed topology

| Topology | Launch shape | Model layout | Typical use |
| -------- | ------------ | ------------ | ----------- |
| Single process | `python ...` | one full model on one device | local debugging and smoke tests |
| Replicated multi-GPU | `accelerate launch --num_processes N ...` | one full model replica per rank | Adam/AdamW scale-out or launcher sanity checks |
| Sharded multi-GPU | `accelerate launch --use_fsdp --fsdp_version 2 ...` | model and optimizer state sharded across ranks | primary distributed pretraining path |

The maintained multi-rank MuonClip path is the sharded one: Accelerate FSDP2
with a 1D row-sharded DTensor mesh. Do not combine MuonClip with tensor
parallelism, context parallelism, or other multi-axis DTensor layouts.

### Distributed validation

Use the commands in [`tests/manual/README.md`](../tests/manual/README.md)
before long multi-rank MuonClip runs. The two distributed smokes cover the raw
FSDP2 owner-compute path and the shipped Accelerate `save_state/load_state`
resume path.

### 2-GPU FSDP2 launch (MuonClip)

Use Accelerate with FSDP v2 and transformer-based wrapping:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
conda run -s --name neobert accelerate launch \
  --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --use_fsdp --fsdp_version 2 \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap EncoderBlock \
  scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml \
  --wandb.enabled false
```

MuonClip's FSDP2 path currently supports only a 1D row-sharded device mesh.
Do not combine it with tensor/context parallelism or other multi-axis DTensor meshes.
DeepSpeed is no longer a supported runtime backend in this repo; use Accelerate
FSDP v2 for distributed training. Legacy DeepSpeed ZeRO checkpoint conversion
remains available via the optional `neobert[legacy-checkpoints]` extra.

For the explicit no-clipping variant, keep the same launch flags and replace
the config path with
`configs/pretraining/pretrain_neobert100m_smollm2data_muonclip_noclip.yaml`.

Distributed launch policy for this repo:

- Accelerate launch flags control process topology and FSDP plugin selection
  (`--num_processes`, `--use_fsdp`, `--fsdp_version`, wrap policy).
- NeoBERT config controls the actual training precision through
  `trainer.mixed_precision`. Pass a matching `accelerate launch --mixed_precision`
  value only to keep launcher output quiet; if you omit it, Accelerate may warn
  about its CLI default even though the trainer still constructs
  `Accelerator(mixed_precision=...)` from config.
- NeoBERT owns `torch.compile` through `trainer.torch_compile` and
  `trainer.torch_compile_backend`. Leave Accelerate dynamo disabled
  (`--dynamo_backend no`, or omit it and accept the warning) rather than trying
  to compile the model through the launcher as well.
- Use `--wandb.name <run-name>` for the W&B run name override; `--wandb.run`
  is not a NeoBERT config key.

## Gradient Accumulation and Norm Logging

- `trainer.gradient_accumulation_steps` counts microbatches per optimizer update.
- For GLUE and contrastive training, the effective sample batch per optimizer
  step is:
  `per_device_train_batch_size * world_size * gradient_accumulation_steps`.
- Pretraining uses the same sample-count formula when microbatches are full, but
  packed batches can vary token counts per update. For packed runs, compare
  `train/tokens_per_sec` and token counts rather than only `steps/sec`.

Pretraining applies one extra scaling step after accumulation: gradients are
rescaled by the global masked-token count for the just-finished update. That
keeps the effective loss normalization aligned with a full masked-token mean
even when masking density or packing makes per-rank token counts uneven.

Current logging semantics:

- `train/grad_norm` is the global L2 gradient norm after accumulation and, in
  pretraining, after masked-token rescaling.
- `train/grad_norm` is measured before `trainer.gradient_clipping`.
- `trainer.gradient_clipping` clips the final accumulated gradient; in
  pretraining that means after token-based rescaling.
- `train/weight_norm` is the global L2 parameter norm on logging steps after
  the optimizer update.

`trainer.gradient_clipping` clips gradients. MuonClip's
`optimizer.muon_config.enable_clipping` toggles its separate QK activation
clipping path, which is auto-disabled for sharded FSDP2 Muon runs.
Muon ships with `norm_factor=original` and `param_policy=hidden_2d`.
Use `all_2d` explicitly when you want exact v0.1.3-style Muon scope for
compatibility benchmarking.
This default routing follows the original Muon guidance and PyTorch's Muon
documentation: hidden transformer matrices use Muon, while embeddings, output
layers, biases, and norm parameters stay on Adam-style fallback groups.

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
- `streaming_read_retries`
- `streaming_read_retry_backoff_seconds`
- `streaming_read_retry_max_backoff_seconds`

When running on CUDA, trainer may warn and apply throughput-friendly defaults
if these are unset/suboptimal. Current PyTorch builds route
`DataLoader(pin_memory=True)` through a deprecated CUDA-specific path, so
NeoBERT keeps loader-side pinning off and pins final CPU batches explicitly
before non-blocking device transfers in the paths that move batches manually.
For hub-backed streaming datasets, NeoBERT also retries transient read failures
when inspecting stream schemas and while iterating long-running train/eval
dataloaders. Recovery resumes from the last yielded example when the underlying
HF iterable dataset supports `state_dict()/load_state_dict()`. Shuffled streams
can still perturb in-buffer order after a retry because HF refill semantics do
not preserve the exact old shuffle buffer contents.

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
- if bf16 is unstable on a specific host, prefer a known-good PyTorch build for
  that machine rather than repo-local BLAS workarounds
- explicit CPU runs (`trainer.use_cpu: true`) force
  `attn_backend: sdpa`
- when mixed precision resolves to `no`, `attn_backend: flash_attn_varlen` is
  auto-switched to `sdpa`
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

### Crash recovery

Resume from the newest saved checkpoint:

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml \
  --trainer.resume_from_checkpoint latest
```

Confirm in logs that startup loads `outputs/.../checkpoints/<step>/` and that
training resumes from the saved global step instead of step 0.

For the streaming SmolLM2 configs in this repo, resume is best-effort:
trainer restores checkpoint state and advances the stream by the consumed
batches before continuing. Late checkpoints can take noticeable time to
replay, and shuffled streams do not guarantee exact sample continuity.

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
- `train/grad_norm` is logged as the global pre-clip norm after accumulation
  and any token-based scaling, so clipping does not hide overshoot in tracker
  plots.
- For paper-style NeoBERT masking strategy, set `datacollator.mask_all: true`.
  Default `false` uses sampled-token 80/10/10 corruption.
  For `p = datacollator.mlm_probability`, global token mix is:
  `(1 - p)` untouched, `0.8p` `[MASK]`, `0.1p` random-token, `0.1p` original-token.
- For packed + compile runs, measure `tokens/sec` rather than `steps/sec`.

## Related Docs

- [Configuration](configuration.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
