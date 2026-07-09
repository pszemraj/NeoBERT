# Training Guide

This guide covers pretraining and contrastive workflows. Full field-level schema/defaults are in [Configuration Reference](../reference/configuration.md). Optimizer policy, Muon defaults, throughput tuning, and gradient/logging semantics are in [Training Optimization](training-optimization.md).

## Entry Points

| Script                                    | Purpose                           |
| ----------------------------------------- | --------------------------------- |
| `scripts/pretraining/pretrain.py`         | MLM pretraining                   |
| `scripts/pretraining/preprocess.py`       | tokenize and save dataset to disk |
| `scripts/pretraining/tokenize_dataset.py` | standalone tokenization helper    |
| `scripts/pretraining/longer_seq.py`       | continue run at longer context    |
| `scripts/contrastive/finetune.py`         | contrastive fine-tuning           |
| `scripts/contrastive/preprocess.py`       | contrastive dataset preprocessing |
| `scripts/contrastive/download.py`         | pre-download contrastive datasets |

For contrastive preprocessing, `dataset.name` may be omitted, `ALL`, a canonical registry key, or a supported HF dataset ID alias; accepted values and cached-split loading rules are in the [Data Source reference](../reference/configuration.md#data-source). Cached split reuse is guarded by `tokenization_manifest.json`, so changing tokenizer/max-length/tokenization settings requires `dataset.force_redownload: true` or a fresh cache.

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

The maintained multi-rank MuonClip path is the sharded one: Accelerate FSDP2 with a 1D row-sharded DTensor mesh. Mesh constraints are listed in [Distributed Muon](training-optimization.md#distributed-muon).

### Distributed validation

Use the commands in [`tests/manual/README.md`](../../tests/manual/README.md) before long multi-rank MuonClip runs. The two distributed smokes cover the raw FSDP2 owner-compute path and the shipped Accelerate `save_state/load_state` resume path.

### 2-GPU FSDP2 launch (MuonClip)

Use Accelerate with FSDP v2 and transformer-based wrapping:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
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

DeepSpeed is no longer a supported runtime backend in this repo; use Accelerate FSDP v2 for distributed training. Legacy DeepSpeed ZeRO checkpoint conversion remains available via the optional `neobert[legacy-checkpoints]` extra.

For the explicit no-clipping variant, keep the same launch flags and replace the config path with `configs/pretraining/pretrain_neobert100m_smollm2data_muonclip_noclip.yaml`.

### Distributed launch policy

- Accelerate launch flags control process topology and FSDP plugin selection (`--num_processes`, `--use_fsdp`, `--fsdp_version`, wrap policy).
- NeoBERT config controls the actual training precision through `trainer.mixed_precision`. Pass a matching `accelerate launch --mixed_precision` value only to keep launcher output quiet; if you omit it, Accelerate may warn about its CLI default even though the trainer still constructs `Accelerator(mixed_precision=...)` from config.
- NeoBERT owns `torch.compile` through `trainer.torch_compile` and `trainer.torch_compile_backend`. Leave Accelerate dynamo disabled (`--dynamo_backend no`, or omit it and accept the warning) rather than trying to compile the model through the launcher as well.
- Use `--wandb.name <run-name>` for the W&B run name override; `--wandb.run` is not a NeoBERT config key.

## Optimization

Use [Training Optimization](training-optimization.md) for:

- MuonClip defaults and recommended modes,
- gradient accumulation and norm logging semantics,
- packed-training and dataloader throughput knobs,
- QK clipping vs standard gradient clipping,
- distributed Muon validation and performance tradeoffs.

## Streaming Eval Strategy

For streaming datasets, prefer:

- `dataset.eval_split: null`
- `dataset.eval_samples: <small integer>`

Runtime behavior:

- if `dataset.eval_split` is unset, trainer tries to auto-detect a validation-style split (`validation`, `eval`, `test`, `dev`);
- if none exists and `dataset.eval_samples` is set, trainer reserves the first `eval_samples` from train for eval and skips them from training to avoid leakage;
- if `trainer.eval_max_batches` is unset, trainer derives the eval budget from `dataset.eval_samples` and `trainer.per_device_eval_batch_size`;
- if neither `trainer.eval_max_batches` nor `dataset.eval_samples` is set, trainer raises an error (explicit eval budget required for streaming eval).
- if no eval dataset can be resolved, eval is skipped.

## Mixed Precision and Compile

- `trainer.mixed_precision`: `no | fp32 | bf16` (`bf16` recommended default)
- runtime normalization: `fp32 -> no`, `true -> bf16`, `false -> no`
- `fp16` is unsupported in NeoBERT training paths
- if bf16 is unstable on a specific host, prefer a known-good PyTorch build for that machine rather than repo-local BLAS workarounds
- explicit CPU runs (`trainer.use_cpu: true`) force `attn_backend: sdpa`
- when mixed precision resolves to `no`, `attn_backend: flash_attn_varlen` is auto-switched to `sdpa`
- `trainer.torch_compile`: enable `torch.compile`
- `trainer.torch_compile_backend`: `inductor | aot_eager | eager`
- `trainer.torch_compile_dynamic`: optional override for dynamic-shape compile; default behavior prefers static-shape compile for stability.
- `trainer.masked_logits_only_loss`: `true | false`

## MLM Loss Path Selection

Use exactly one pretraining loss path per run:

- `trainer.masked_logits_only_loss: true` Uses masked-logits-only MLM loss (default and recommended). This avoids full `(B,S,V)` logits materialization in the hot pretraining path.
- `trainer.masked_logits_only_loss: false` Uses the original NeoBERT full-logits CE path (legacy ablation/debug path).

There is no mixed/cross objective mode in trainer config; this flag picks one path for the run.

Current project default is `true`; new pretraining runs should keep `masked_logits_only_loss: true` unless you are intentionally running an ablation against the legacy baseline.

## Checkpointing and Resume

Step checkpoints (resume + export assets) share one layout across pretraining, contrastive, and GLUE:

```text
<output_dir>/checkpoints/<step>/
  model.safetensors
  optimizer_param_names.json
  config.yaml
  tokenizer_info.json
  tokenizer/
  accelerate/
    model.safetensors
    optimizer.bin / scheduler.bin / random_states_*.pkl
    custom_checkpoint_*.pkl
```

The top-level `model.safetensors` is the portable export/eval payload and intentionally materializes tied tensors under every expected key. Accelerate's resume state lives under `accelerate/` so its strictly-loaded model file cannot collide with the portable payload; resume falls back to the step root for checkpoints written before this layout. Contrastive step checkpoints carry the same metadata files; GLUE step checkpoints omit the config/tokenizer metadata but do write the optimizer parameter-name manifest, so GLUE resume also fails fast on optimizer parameter-order drift.

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

Confirm in logs that startup loads `outputs/.../checkpoints/<step>/` and that training resumes from the saved global step instead of step 0.

Resume treats checkpoint metadata as authoritative for model/tokenizer/data objective fields. If the current launch config disagrees with the checkpoint's `config.yaml`, the checkpoint values win before tokenizer/model/dataloader construction, and `tokenizer/` inside the checkpoint is used when present. All `trainer.*` runtime/performance knobs (batch size, gradient accumulation, precision, compile flags, loss path, step budget) stay controlled by the current launch config, so overrides such as `--trainer.per_device_train_batch_size` are honored on resume.

Optimizer state is guarded by `optimizer_param_names.json`. Resume fails fast if the current optimizer parameter-group order differs from the order saved with the checkpoint (PyTorch optimizer buffers are positional inside groups), or if the optimizer's recorded state semantics differ from the current implementation (for example a momentum-rule change that reinterprets saved buffers). For MuonClip, the recorded semantics also include the configured update-rule selectors (`norm_factor`, `param_policy`, `orthogonalization`, `nesterov`), so resuming under a drifted selector - including a changed repo default - fails fast instead of silently rescaling updates mid-run; tunables such as learning rate and betas stay freely resumable. Runtime wrapper prefixes such as `_orig_mod.` and `module.` are stripped before manifest comparison, so toggling `trainer.torch_compile` or distributed wrapper surfaces does not count as parameter-order drift. Checkpoints written before the manifest existed are rejected by design: their optimizer state is unverifiable, and this repo does not carry checkpoint back-compat before a stable release - start a new run or continue from model weights only.

Streaming resume is approximate (skip-based), not exact. On resume the trainer restores model/optimizer/scheduler/metrics state and re-advances the stream via `accelerator.skip_first_batches` by the number of raw dataloader batches pulled in the current epoch. That count is tracked separately from trained microbatches because packed collation buffers undersized batches (a raw pull that trains no batch), so skipping by the trained-batch count would under-skip and replay data. It does not checkpoint the dataset cursor, because the dataset is consumed through an Accelerate-prepared `DataLoader` whose adapter iterates one batch ahead (and `DataLoaderDispatcher` can prefetch `num_processes` batches) before yielding the batch the trainer optimizes - so the raw dataset cursor at checkpoint time is ahead of the last trained batch and is not a valid resume boundary. Late checkpoints can take noticeable time to replay, and shuffled streams do not guarantee exact sample continuity. Exact streaming resume needs a stateful-dataloader boundary (torchdata `StatefulDataLoader` via Accelerate `use_stateful_dataloader`) and is tracked as follow-up work in [Deferred Work](../TODO.md).

For strict deterministic continuation, switch to a non-streaming tokenized dataset and resume there:

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

This resumes model/optimizer/scheduler states, but data order will not exactly match the interrupted streaming run.

Notes:

- resume and export both operate from `<output_dir>/checkpoints/`.
- pretraining resume with `dataset.streaming: true` uses best-effort stream advancement based on saved batch counters (the raw dataset cursor is not checkpointed; see the resume section above).
- for exact deterministic continuation, prefer non-streaming (`dataset.streaming: false`) runs, whose map-style datasets support exact index-based resume.
- deferred: exact streaming resume via a stateful-dataloader boundary, and a name-keyed optimizer-state transplant for resume across intentional parameter-registration refactors, are tracked in [Deferred Work](../TODO.md).

## Pre-tokenized Datasets

Non-streaming datasets are tokenized through `Dataset.map`, whose results HuggingFace caches automatically (keyed on a fingerprint of the tokenize function and tokenizer state), so repeated runs reuse tokenized data without a bespoke cache. To materialize a tokenized dataset on disk ahead of time — for a shared location, a faster device, or to inspect it — run one of the helpers below and point `dataset.path` at the output.

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
- `train/grad_norm` is logged as the global pre-clip norm after accumulation and any token-based scaling, so clipping does not hide overshoot in tracker plots.
- For paper-style NeoBERT masking strategy, set `datacollator.mask_all: true`. Default `false` uses sampled-token 80/10/10 corruption; the exact token-mix math is in [Data Collator](../reference/configuration.md#data-collator).
- For packed + compile runs, measure `tokens/sec` rather than `steps/sec`.

## Related Docs

- [Configuration](../reference/configuration.md)
- [Training optimization](training-optimization.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
