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

Contrastive dataset selection and cache-reuse rules are in the [Data Source reference](../reference/configuration.md#data-source).

## Pretraining

### Long-running template

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
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip_noclip.yaml \
  --wandb.enabled false
```

DeepSpeed is no longer a supported runtime backend in this repo; use Accelerate FSDP v2 for distributed training. Legacy DeepSpeed ZeRO checkpoint conversion remains available via the optional `neobert[legacy-checkpoints]` extra.

The `_noclip` recipe explicitly disables QK clipping, which is not supported by the FSDP2 sharded Muon update path. Use `configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml` only for unsharded training where QK clipping remains available.

### Distributed launch policy

- Accelerate launch flags control process topology and FSDP plugin selection (`--num_processes`, `--use_fsdp`, `--fsdp_version`, wrap policy).
- NeoBERT config controls the actual training precision through `trainer.mixed_precision`. Pass a matching `accelerate launch --mixed_precision` value only to keep launcher output quiet; if you omit it, Accelerate may warn about its CLI default even though the trainer still constructs `Accelerator(mixed_precision=...)` from config.
- NeoBERT owns `torch.compile` through `trainer.torch_compile` and `trainer.torch_compile_backend`. Leave Accelerate dynamo disabled (`--dynamo_backend no`, or omit it and accept the warning) rather than trying to compile the model through the launcher as well.
- Use `--wandb.name <run-name>` for the W&B run name override; `--wandb.run` is not a NeoBERT config key.

## Optimization

See [Training Optimization](training-optimization.md) for MuonClip policy, gradient and metric semantics, packing, dataloader throughput, and distributed constraints.

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

Supported precision values, compile fields, defaults, and runtime adjustments are listed in [Configuration](../reference/configuration.md#training-loop). Use `bf16` on a compatible CUDA stack, `no` for FP32 execution, and static-shape compilation unless the workload requires dynamic shapes.

## MLM Loss Path Selection

`trainer.masked_logits_only_loss` selects one loss path for the entire run. Keep the default masked-logits-only path for normal training; set it to `false` only for a full-logits ablation.

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

The top-level `model.safetensors` is the portable export/eval payload and intentionally materializes tied tensors under every expected key. Accelerate's resume state lives under `accelerate/` so its strictly-loaded model file cannot collide with the portable payload; checkpoints without that state directory cannot be resumed as full training runs. Contrastive step checkpoints carry the same metadata files; GLUE step checkpoints omit the config/tokenizer metadata but do write the optimizer parameter-name manifest, so GLUE resume also fails fast on optimizer parameter-order drift.

A bare step selector such as `--trainer.resume_from_checkpoint 100` resolves to `<output_dir>/checkpoints/100`; `latest` scans the same checkpoint root, while absolute paths and existing output-relative paths remain available for explicit selection.

Pretraining and contrastive runs save on `trainer.save_steps` ticks and always save the terminal step when `trainer.save_strategy: steps` and `trainer.save_model: true`. `trainer.save_total_limit: 0` or `null` disables pruning. GLUE applies its own `steps`, `epoch`, `best`, or `no` save strategy.

### Crash recovery

Resume from the newest saved checkpoint:

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml \
  --trainer.resume_from_checkpoint latest
```

Confirm in logs that startup loads `outputs/.../checkpoints/<step>/` and that training resumes from the saved global step instead of step 0. A resumable step contains `accelerate/`, `optimizer_param_names.json`, and `checkpoint_complete.json`. The completion marker is written last; `latest` ignores newer numeric directories left incomplete by an interrupted save, preserves zero-padded step names, and explicit selection of an incomplete step fails with the missing artifacts listed.

Resume treats checkpoint metadata as authoritative for model shape/semantics, tokenizer identity, masking, objective fields, and the pretraining per-device batch size. If the current launch config disagrees with the checkpoint's `config.yaml`, the checkpoint values win before tokenizer/model/dataloader construction, and `tokenizer/` inside the checkpoint is used when present. The batch size cannot change because the mid-epoch data cursor counts batches; reinterpreting it under a different batch size would skip or replay samples. Other `trainer.*` runtime/performance knobs such as gradient accumulation, precision, compile flags, loss path, and step budget stay controlled by the current launch config.

Two continuation knobs are also launch-controlled so continued pretraining is not silently undone: the training corpus identity (`dataset.name`/`config`/`path`/`cache_dir`/`text_column`), which never affects checkpointed model or optimizer state, and - for RoPE models only - the context window (`model.max_position_embeddings`, `tokenizer.max_length`, `dataset.max_seq_length`, `datacollator.max_length`), which RoPE makes weight-compatible. When corpus identity changes, model/optimizer progress is retained but the data epoch and cursor restart at zero and any packed fragments from the old corpus are discarded. A cache-directory-only change does not reset data position. Non-RoPE sequence length stays checkpoint-authoritative because it sizes a learned positional table and a change would break the strict weight load.

Optimizer state is guarded by `optimizer_param_names.json`. Resume fails fast if the current optimizer parameter-group order differs from the order saved with the checkpoint (PyTorch optimizer buffers are positional inside groups), or if the optimizer's recorded state semantics differ from the current implementation (for example a momentum-rule change that reinterprets saved buffers). For MuonClip, the recorded semantics also include the configured update-rule selectors (`norm_factor`, `param_policy`, `orthogonalization`, `nesterov`), so resuming under a drifted selector - including a changed repo default - fails fast instead of silently rescaling updates mid-run; tunables such as learning rate and betas stay freely resumable. Runtime wrapper prefixes such as `_orig_mod.` and `module.` are stripped before manifest comparison, so toggling `trainer.torch_compile` or distributed wrapper surfaces does not count as parameter-order drift. Checkpoints written before the manifest existed are rejected by design: their optimizer state is unverifiable, and this repo does not carry checkpoint back-compat before a stable release - start a new run or continue from model weights only.

Streaming resume is approximate (skip-based), not exact. On resume the trainer restores model/optimizer/scheduler/metrics state and re-advances the stream via `accelerator.skip_first_batches` by the number of raw dataloader batches pulled in the current epoch. That count is tracked separately from trained microbatches because packed collation buffers undersized batches (a raw pull that trains no batch), so skipping by the trained-batch count would under-skip and replay data. The rank-local packed-fragment buffer is checkpointed with the raw-pull cursor so skipping consumed pulls does not discard buffered, untrained samples. The trainer does not checkpoint the dataset cursor, because the dataset is consumed through an Accelerate-prepared `DataLoader` whose adapter iterates one batch ahead (and `DataLoaderDispatcher` can prefetch `num_processes` batches) before yielding the batch the trainer optimizes - so the raw dataset cursor at checkpoint time is ahead of the last trained batch and is not a valid resume boundary. Late checkpoints can take noticeable time to replay, and shuffled streams do not guarantee exact sample continuity. Exact streaming resume needs a stateful-dataloader boundary (torchdata `StatefulDataLoader` via Accelerate `use_stateful_dataloader`) and is tracked as follow-up work in [Deferred Work](../TODO.md).

For reproducible sample ordering, materialize the data as described in [Pre-tokenized Datasets](#pre-tokenized-datasets), set `dataset.streaming: false`, and keep the corpus, seed, per-device batch size, and distributed world size unchanged across resume. Map-style training uses an epoch-seeded sampler and restores its epoch before skipping consumed raw batches, including checkpoints written on the final pull of an epoch. Dynamically generated MLM masks are not bitwise replayed because worker and collator RNG/prefetch state is not checkpointed, so resumed weights can still diverge from an uninterrupted run even when sample IDs match. Switching an interrupted streaming run to a map-style dataset preserves model and optimizer state but cannot recreate the interrupted stream's data order. Exact streaming resume and name-keyed optimizer-state transplant are tracked in [Deferred Work](../TODO.md).

## Pre-tokenized Datasets

Non-streaming datasets are tokenized through `Dataset.map`, whose results HuggingFace caches automatically (keyed on a fingerprint of the tokenize function and tokenizer state), so repeated runs reuse tokenized data without a bespoke cache. To materialize a tokenized dataset on disk ahead of time — for a shared location, a faster device, or to inspect it — run one of the helpers below and point `dataset.path` at the output.

Two common paths:

1. Offline preprocess via config:

```bash
python scripts/pretraining/preprocess.py \
  configs/pretraining/pretrain_neobert.yaml
```

2. Standalone tokenizer helper:

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

Set `dataset.path` to output from `scripts/contrastive/preprocess.py` and configure one initialization source: `contrastive.pretrained_checkpoint_dir` for normal training, a self-contained contrastive `trainer.resume_from_checkpoint`, or `contrastive.allow_random_weights: true` for an intentional random-weight experiment. Random initialization is rejected by default. The trainer validates every selected cache's tokenizer vocabulary/serialization, special tokens, length, truncation policy, and token-ID/attention-mask field pairs before model construction; `dataset.load_all_from_disk` skips downloads and tokenization but never skips this validation. The optional SimCSE anti-forgetting branch is disabled by default; enabling it requires both `contrastive.pretraining_prob > 0` and a separate tokenized dataset at `contrastive.pretraining_dataset_path`. Every active supervised and SimCSE dataloader is prepared and sharded by Accelerate. On SIGTERM, the signal handler records intent only; all ranks finish the current optimizer update, save one synchronized complete checkpoint, restore the previous handler, and exit with status `128 + SIGTERM` so schedulers can requeue the job. Partial accumulated gradients are never checkpointed as if they were a complete update.

## Related Docs

- [Configuration](../reference/configuration.md)
- [Training optimization](training-optimization.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
