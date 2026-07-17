# Training Guide

The full field-level schema, defaults, and interactions are in the [YAML configuration reference](../reference/config_reference.yaml). Optimizer policy, Muon defaults, throughput tuning, and gradient/logging semantics are in [Training Optimization](training-optimization.md).

## Entry Points

| Script                                    | Purpose                           |
| ----------------------------------------- | --------------------------------- |
| `scripts/pretraining/pretrain.py`         | MLM pretraining                   |
| `scripts/pretraining/preprocess.py`       | tokenize and save dataset to disk |
| `scripts/pretraining/tokenize_dataset.py` | standalone tokenization helper    |
| `scripts/pretraining/longer_seq.py`       | build long-sequence dataset views |
| `scripts/contrastive/finetune.py`         | contrastive fine-tuning           |
| `scripts/contrastive/preprocess.py`       | contrastive dataset preprocessing |
| `scripts/contrastive/download.py`         | pre-download contrastive datasets |

Contrastive dataset selection and cache-reuse rules are documented inline on the relevant keys in the [YAML configuration reference](../reference/config_reference.yaml).

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

Supported Muon topologies and clipping combinations are listed in [Distributed Muon](training-optimization.md#distributed-muon) and [Clipping](training-optimization.md#clipping).

### Distributed validation

Run the commands in [`tests/manual/README.md`](../../tests/manual/README.md) before long multi-rank FSDP2 Muon runs.

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

Use the `_noclip` recipe for FSDP2; supported alternatives are listed in [Clipping](training-optimization.md#clipping).

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

Supported precision values, compile fields, defaults, and runtime adjustments are listed in the [YAML configuration reference](../reference/config_reference.yaml). Use `bf16` on a compatible CUDA stack, `no` for FP32 execution, and static-shape compilation unless the workload requires dynamic shapes.

## MLM Loss Path Selection

`trainer.masked_logits_only_loss` selects one loss path for the entire run. Keep the default masked-logits-only path for normal training; set it to `false` only for a full-logits ablation.

## Checkpointing and Resume

Step checkpoints (resume + export assets) share one layout across pretraining, contrastive, and GLUE:

```text
<output_dir>/checkpoints/<step>/
  model.safetensors
  optimizer_param_names.json
  config.yaml
  tokenizer/
  checkpoint_complete.json
  model_config/                 # GLUE Hub-model checkpoints
  tokenizer_info.json           # pretraining checkpoints
  accelerate/
    model.safetensors
    optimizer.bin / scheduler.bin / random_states_*.pkl
    custom_checkpoint_*.pkl
```

The top-level `model.safetensors` is the portable export/eval payload and intentionally materializes tied tensors under every expected key. Accelerate's resume state lives under `accelerate/` so its strictly-loaded model file cannot collide with the portable payload; checkpoints without that state directory cannot be resumed as full training runs. Every current trainer bundles its resolved config and tokenizer. GLUE also saves the instantiated Hugging Face model configuration when applicable.

A bare step selector such as `--trainer.resume_from_checkpoint 100` resolves to `<output_dir>/checkpoints/100`; `latest` scans the same checkpoint root, while absolute paths and existing output-relative paths remain available for explicit selection.

Pretraining and contrastive runs save on `trainer.save_steps` ticks and always save the terminal step when `trainer.save_strategy: steps` and `trainer.save_model: true`. `trainer.save_total_limit: 0` or `null` disables pruning. GLUE applies its own `steps`, `epoch`, `best`, or `no` save strategy.

On SIGTERM, pretraining and contrastive trainers record the request, let every rank finish the current optimizer update, save one synchronized complete checkpoint, restore the previous handler, and exit with status `128 + SIGTERM`. Partial accumulated gradients are never checkpointed as a complete update.

### Crash recovery

Resume from the newest saved checkpoint:

```bash
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml \
  --trainer.resume_from_checkpoint latest
```

Confirm in logs that startup loads `outputs/.../checkpoints/<step>/` and that training resumes from the saved global step instead of step 0. A resumable step contains portable model/config/tokenizer artifacts, the optimizer parameter-name manifest, task-specific loop state, and Accelerate model/optimizer/scheduler/RNG state. The completion marker is written last with a versioned path/size inventory; resume validates the task, required state roles, exact custom-state count, and every inventoried file. Main-only metadata failures are synchronized before portable state collection so every rank exits the checkpoint save instead of entering an unmatched collective. `latest` ignores newer numeric directories left incomplete or damaged after an interrupted save, preserves zero-padded step names, and explicit selection of an incomplete step fails with the invalid artifacts listed.

Resume treats checkpoint metadata as authoritative for model shape/semantics, tokenizer identity, masking, objective fields, and cursor-sensitive data geometry. If the current launch config disagrees with the checkpoint's `config.yaml`, checkpoint values win before tokenizer/model/dataloader construction, and `tokenizer/` inside the checkpoint is used when present. Pretraining forces its per-device batch size. Reinterpreting a mid-epoch cursor under different geometry would skip or replay samples, while constructing a different optimizer or scheduler would make the loaded state ambiguous. Runtime knobs that do not define the saved cursor or model state, including precision, compile flags, logging cadence, and the final step budget, stay launch-controlled. GLUE-specific resume behavior is described in [Important GLUE behavior](evaluation.md#important-glue-behavior).

Data-source locations remain launch-controlled: `dataset.name`/`config`/`path`/`cache_dir`/`text_column` for every task and `contrastive.pretraining_dataset_path` for the optional SimCSE source. Resume warns when these fields drift instead of replacing launch paths with checkpoint paths. When a pretraining corpus changes in `name`, `config`, `path`, or `text_column`, model/optimizer progress is retained but the data epoch and cursor restart at zero and packed fragments from the old corpus are discarded; a cache-directory-only change does not reset data position. For RoPE models, the context window (`model.max_position_embeddings`, `tokenizer.max_length`, `dataset.max_seq_length`, `datacollator.max_length`) is also launch-controlled because changing it does not resize weights. Non-RoPE sequence length stays checkpoint-authoritative because it sizes a learned positional table.

Optimizer state is guarded by `optimizer_param_names.json`. Resume fails fast if the current optimizer parameter-group order differs from the order saved with the checkpoint (PyTorch optimizer buffers are positional inside groups), or if the optimizer's recorded state semantics differ from the current implementation (for example a momentum-rule change that reinterprets saved buffers). Optimizer and scheduler construction comes from the checkpoint config because loading state into a different optimizer class or schedule recipe is not a faithful continuation. For MuonClip, the recorded semantics also include the configured update-rule selectors (`norm_factor`, `param_policy`, `orthogonalization`, `nesterov`, and `clipping`) plus the clipping reduction contract, so a repo implementation change that reinterprets saved buffers fails fast instead of silently rescaling updates mid-run. Runtime wrapper prefixes such as `_orig_mod.` and `module.` are stripped before manifest comparison, so toggling `trainer.torch_compile` or distributed wrapper surfaces does not count as parameter-order drift. Checkpoints written before the manifest or current completion inventory existed are rejected by design: their optimizer state or artifact set is unverifiable, and this repo does not carry checkpoint back-compat before a stable release - start a new run or continue from model weights only.

Streaming resume is approximate (skip-based), not exact. On resume the trainer restores model/optimizer/scheduler/metrics state and re-advances the stream via `accelerator.skip_first_batches` by the number of raw dataloader batches pulled in the current epoch. That count is tracked separately from trained microbatches and paired with the rank-local packed-fragment buffer, which preserves rows retained when an oversized packed result is split or when single-process full-batch buffering is enabled. Distributed packed training never skips a model call to fill an undersized local batch; it uses variable local microbatch sizes, and checkpoint creation verifies that all ranks share the same optimizer step, epoch, trained-microbatch count, and raw-pull cursor before persisting the shared loop state. Unversioned loop-state checkpoints are rejected because they cannot prove this invariant. The trainer does not checkpoint the dataset cursor, because the dataset is consumed through an Accelerate-prepared `DataLoader` whose adapter iterates one batch ahead (and `DataLoaderDispatcher` can prefetch `num_processes` batches) before yielding the batch the trainer optimizes - so the raw dataset cursor at checkpoint time is ahead of the last trained batch and is not a valid resume boundary. Late checkpoints can take noticeable time to replay, and shuffled streams do not guarantee exact sample continuity.

For reproducible sample ordering, materialize the data as described in [Pre-tokenized Datasets](#pre-tokenized-datasets), set `dataset.streaming: false`, and keep the corpus, seed, per-device batch size, and distributed world size unchanged across resume. Map-style training uses an epoch-seeded sampler and restores its epoch before skipping consumed raw batches, including checkpoints written on the final pull of an epoch. Dynamically generated MLM masks are not bitwise replayed because worker and collator RNG/prefetch state is not checkpointed, so resumed weights can still diverge from an uninterrupted run even when sample IDs match. Switching an interrupted streaming run to a map-style dataset preserves model and optimizer state but cannot recreate the interrupted stream's data order. Exact streaming resume and name-keyed optimizer-state transplant are tracked in [Deferred Work](../TODO.md).

## Pre-tokenized Datasets

Non-streaming datasets are tokenized through `Dataset.map`, whose results Hugging Face caches automatically (keyed on a fingerprint of the tokenize function and tokenizer state), so repeated runs reuse tokenized data without a bespoke cache. To materialize a tokenized dataset on disk ahead of time for a shared location, a faster device, or inspection, run one of the helpers below and point `dataset.path` at the output.

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
  --max-length 1024 \
  --for-packing
```

When `datacollator.pack_sequences` is enabled, each stored example must contain raw content tokens without outer CLS/SEP or BOS/EOS tokens. The packing collator inserts one boundary pair per segment and rejects pre-specialized inputs rather than silently duplicating boundaries. The config-driven preprocessor applies this contract automatically; the standalone helper requires `--for-packing`, where `--max-length` is the final packed length and the helper reserves space for the boundary tokens.

To create filtered long-sequence views, pass the threshold explicitly; the utility writes `<dataset.path>+<min-length>` and a second view at twice the threshold:

```bash
python scripts/pretraining/longer_seq.py \
  configs/pretraining/pretrain_neobert.yaml \
  --min-length 512
```

## Contrastive

```bash
python scripts/contrastive/finetune.py \
  configs/contrastive/contrastive_neobert.yaml
```

Set `dataset.path` to output from `scripts/contrastive/preprocess.py` and configure one initialization source: `contrastive.pretrained_checkpoint_dir` for normal training, a self-contained contrastive `trainer.resume_from_checkpoint`, or `contrastive.allow_random_weights: true` for an intentional random-weight experiment. Random initialization is rejected by default. The standalone preprocess pipeline serializes cache access with a dataset-root file lock and atomically publishes manifests, so concurrent launches targeting the same path wait instead of racing. The trainer validates every selected cache's tokenizer vocabulary/serialization, special tokens, length, truncation policy, and token-ID/attention-mask field pairs before model construction; `dataset.load_all_from_disk` skips downloads and tokenization but never skips this validation. The optional SimCSE anti-forgetting branch is disabled by default; enabling it requires both `contrastive.pretraining_prob > 0` and a separate tokenized dataset at `contrastive.pretraining_dataset_path`. Every active supervised and SimCSE dataloader is prepared and sharded by Accelerate.
