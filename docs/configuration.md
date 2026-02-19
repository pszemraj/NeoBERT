# Configuration Reference

> [!TIP]
> Example configs are in [configs/](../configs/) (for production) and
> [tests/configs/](../tests/configs/) (for tiny smoke/regression runs).

This page is the primary source of truth for NeoBERT's YAML config schema
(`src/neobert/config.py`) and defaults.

---

- [How To Use This Page](#how-to-use-this-page)
- [Variables and Dot Overrides](#variables-and-dot-overrides)
  - [YAML variables](#yaml-variables)
  - [Dot-path overrides in Python](#dot-path-overrides-in-python)
- [High-Impact Settings](#high-impact-settings)
- [Model Architecture](#model-architecture)
  - [Core](#core)
  - [Advanced](#advanced)
- [Positional Encoding](#positional-encoding)
- [Tokenizer](#tokenizer)
- [Data Source](#data-source)
  - [Core](#core-1)
  - [Performance and Preprocessing](#performance-and-preprocessing)
  - [Contrastive-Only Data Fields](#contrastive-only-data-fields)
- [Training Loop](#training-loop)
  - [Core](#core-2)
  - [Stability and Performance](#stability-and-performance)
  - [Control and Legacy Compatibility](#control-and-legacy-compatibility)
- [LR Schedule](#lr-schedule)
- [Optimizer](#optimizer)
  - [Base Optimizer](#base-optimizer)
  - [MuonClip (`optimizer.muon_config`)](#muonclip-optimizermuon_config)
- [Data Collator](#data-collator)
- [Checkpointing and Resume](#checkpointing-and-resume)
- [Logging and Tracking](#logging-and-tracking)
  - [Weights and Biases](#weights-and-biases)
  - [Top-Level Runtime Metadata](#top-level-runtime-metadata)
- [Task-Specific Sections](#task-specific-sections)
  - [GLUE (`glue`)](#glue-glue)
  - [Contrastive (`contrastive`)](#contrastive-contrastive)
  - [MTEB Top-Level Keys](#mteb-top-level-keys)
- [Constraints, Requirements, and Gotchas](#constraints-requirements-and-gotchas)
- [Legacy Key Mapping (Still Normalized)](#legacy-key-mapping-still-normalized)
- [Practical YAML Presets](#practical-yaml-presets)
  - [1) Base Pretraining (Balanced)](#1-base-pretraining-balanced)
  - [2) Memory-Constrained Single GPU](#2-memory-constrained-single-gpu)
  - [3) Resumable Local-Data Run](#3-resumable-local-data-run)
  - [4) Full Logging + Frequent Eval](#4-full-logging--frequent-eval)
- [Related Docs](#related-docs)

---

## How To Use This Page

- Treat this as a **config-file reference**, not a CLI-first reference.
- Start with **High-Impact Settings**, then fill in grouped sections.
- Defaults shown here are dataclass defaults unless noted.
- High-impact tables are summaries; field semantics are defined in section tables.
- Unknown keys fail fast during config loading.

## Variables and Dot Overrides

`ConfigLoader.load(...)` supports a small YAML variable system and post-load dot
overrides for sweep-style runs.

### YAML variables

- Define top-level `variables:` in YAML.
- Use exact replacement for type-preserving values:
  - `dataset.max_seq_length: $variables.seq_len`
- Use inline interpolation for strings:
  - `wandb.name: "run-{$variables.tag}"`
  - `wandb.name: "run-${variables.tag}"` (alternate form)
- Nested variable references are supported.
- Circular variable references fail fast with an explicit error.
- Unresolved `$variables.*` tokens in strings emit warnings with field location.

Example: one `seq_len` driving multiple runtime fields (without coupling model
context length yet):

```yaml
variables:
  seq_len: 1024
  run_tag: pretrain-1024

dataset:
  max_seq_length: $variables.seq_len

tokenizer:
  max_length: $variables.seq_len

datacollator:
  max_length: $variables.seq_len

wandb:
  name: "neobert-{$variables.run_tag}"
```

Use this pattern for shared run-time sequence settings. Keep
`model.max_position_embeddings` as an explicit architecture decision.

### Dot-path overrides in Python

When calling `ConfigLoader.load(path, overrides=...)`, overrides can be either:

- a nested mapping (existing behavior), or
- a list of dot-path strings, for example:

```python
cfg = ConfigLoader.load(
    "configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml",
    overrides=[
        "trainer.max_steps=2000",
        "optimizer.lr=2e-4",
        "dataset.streaming=false",
    ],
)
```

Accepted list token forms:

- `section.key=value`
- `--section.key=value`
- `--section.key value`

Unknown paths and invalid value types fail fast with path-specific errors.
Overrides are validated with the same semantic checks as base YAML configs.

## High-Impact Settings

| Key                                   | Type          | Default         | Description                                                          |
| ------------------------------------- | ------------- | --------------- | -------------------------------------------------------------------- |
| `task`                                | `str`         | `"pretraining"` | Run mode: `pretraining`, `glue`, `mteb`, `contrastive`.              |
| `model.hidden_size`                   | `int`         | `768`           | Main width of the transformer.                                       |
| `model.num_hidden_layers`             | `int`         | `12`            | Depth of the encoder stack.                                          |
| `model.num_attention_heads`           | `int`         | `12`            | Attention heads per layer.                                           |
| `model.max_position_embeddings`       | `int`         | `512`           | Maximum sequence length the model is built for.                      |
| `dataset.name`                        | `str`         | `"refinedweb"`  | HF dataset name when loading from hub.                               |
| `dataset.path`                        | `str`         | `""`            | Local dataset path (preferred when available).                       |
| `dataset.streaming`                   | `bool`        | `true`          | Stream from dataset source instead of materializing all data.        |
| `trainer.per_device_train_batch_size` | `int`         | `16`            | Per-device train microbatch size.                                    |
| `trainer.gradient_accumulation_steps` | `int`         | `1`             | Number of microbatches per optimizer step.                           |
| `trainer.max_steps`                   | `int`         | `1000000`       | Total training steps.                                                |
| `optimizer.name`                      | `str`         | `"adamw"`       | Optimizer family (`adamw`, `adam`, `muonclip`).                      |
| `optimizer.lr`                        | `float`       | `1e-4`          | Base learning rate.                                                  |
| `scheduler.name`                      | `str`         | `"cosine"`      | LR schedule type.                                                    |
| `datacollator.mask_all`               | `bool`        | `false`         | `false` uses sampled-token BERT-style 80/10/10 masking.              |
| `datacollator.pack_sequences`         | `bool`        | `false`         | Enable packed-sequence collation.                                    |
| `trainer.resume_from_checkpoint`      | `str \| None` | `null`          | Checkpoint to resume from.                                           |
| `use_deepspeed`                       | `bool`        | `false`         | Legacy hint for loading DeepSpeed-formatted contrastive checkpoints. |

> [!NOTE]
> Runtime preprocessing synchronizes tokenizer-derived values (including `model.pad_token_id`) and aligns vocab size for model/tokenizer consistency.

---

## Model Architecture

### Core

| Key                             | Type          | Default    | Description                                                  |
| ------------------------------- | ------------- | ---------- | ------------------------------------------------------------ |
| `model.name`                    | `str \| None` | `null`     | Optional model identifier/path metadata.                     |
| `model.hidden_size`             | `int`         | `768`      | Hidden width for embeddings, attention, and MLP projections. |
| `model.num_hidden_layers`       | `int`         | `12`       | Number of encoder blocks.                                    |
| `model.num_attention_heads`     | `int`         | `12`       | Attention heads per block.                                   |
| `model.intermediate_size`       | `int`         | `3072`     | FFN/MLP hidden size before activation projection.            |
| `model.max_position_embeddings` | `int`         | `512`      | Maximum supported sequence length.                           |
| `model.vocab_size`              | `int`         | `30522`    | Runtime-synchronized model vocab size.                       |
| `model.hidden_act`              | `str`         | `"swiglu"` | `swiglu` or `gelu`.                                          |
| `model.dropout_prob`            | `float`       | `0.0`      | Dropout probability in model blocks.                         |
| `model.norm_eps`                | `float`       | `1e-5`     | Epsilon for normalization stability.                         |

### Advanced

| Key                           | Type    | Default              | Description                                       |
| ----------------------------- | ------- | -------------------- | ------------------------------------------------- |
| `model.rms_norm`              | `bool`  | `true`               | Use RMSNorm (otherwise LayerNorm).                |
| `model.attn_backend`          | `str`   | `"sdpa"`             | Attention backend: `sdpa` or `flash_attn_varlen`. |
| `model.kernel_backend`        | `str`   | `"auto"`             | Kernel backend: `auto`, `liger`, or `torch`.      |
| `model.rope`                  | `bool`  | `true`               | Enable rotary positional encoding.                |
| `model.ngpt`                  | `bool`  | `false`              | Enable nGPT variant.                              |
| `model.base_scale`            | `float` | `1.0 / (960.0**0.5)` | nGPT scale constant.                              |
| `model.pad_token_id`          | `int`   | `0`                  | Runtime-synced from tokenizer.                    |
| `model.embedding_init_range`  | `float` | `0.02`               | Embedding init stddev.                            |
| `model.decoder_init_range`    | `float` | `0.02`               | Decoder init stddev.                              |
| `model.classifier_init_range` | `float` | `0.02`               | Classifier head init stddev.                      |
| `model.from_hub`              | `bool`  | `false`              | Metadata flag for loading behavior.               |

---

## Positional Encoding

| Key                             | Type    | Default | Description                                        |
| ------------------------------- | ------- | ------- | -------------------------------------------------- |
| `model.rope`                    | `bool`  | `true`  | Rotary positional encoding toggle.                 |
| `model.max_position_embeddings` | `int`   | `512`   | Positional context length.                         |
| `model.norm_eps`                | `float` | `1e-5`  | Stability epsilon used around attention/FFN norms. |

> [!IMPORTANT]
> If you raise context length, verify memory headroom and backend compatibility (`flash_attn_varlen` strongly recommended for packed long-context training).

---

## Tokenizer

| Key                                     | Type          | Default               | Description                                                                           |
| --------------------------------------- | ------------- | --------------------- | ------------------------------------------------------------------------------------- |
| `tokenizer.name`                        | `str`         | `"bert-base-uncased"` | Tokenizer name from HF hub.                                                           |
| `tokenizer.path`                        | `str \| None` | `null`                | Local tokenizer path (takes precedence when provided).                                |
| `tokenizer.max_length`                  | `int`         | `512`                 | Tokenizer max length used during preprocessing.                                       |
| `tokenizer.padding`                     | `str`         | `"max_length"`        | Padding behavior passed to tokenization pipeline.                                     |
| `tokenizer.truncation`                  | `bool`        | `true`                | Truncate to max length during tokenization.                                           |
| `tokenizer.vocab_size`                  | `int \| None` | `null`                | Runtime-synchronized to effective model vocab size.                                   |
| `tokenizer.trust_remote_code`           | `bool`        | `false`               | Allow tokenizer remote code execution.                                                |
| `tokenizer.revision`                    | `str \| None` | `null`                | Optional tokenizer revision/commit pin for reproducibility.                           |
| `tokenizer.allow_special_token_rewrite` | `bool`        | `false`               | Explicit opt-in for fallback special-token rewrite when tokenizer lacks `mask_token`. |

> [!NOTE]
> Trainer now pads tokenizer length with inert placeholder tokens to keep `len(tokenizer) == model.vocab_size`.
> If a tokenizer lacks `mask_token`, NeoBERT now requires explicit
> `tokenizer.allow_special_token_rewrite: true` before mutating special tokens.

---

## Data Source

### Core

| Key                        | Type            | Default        | Description                                                                                                        |
| -------------------------- | --------------- | -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `dataset.name`             | `str`           | `"refinedweb"` | Dataset name for `load_dataset`.                                                                                   |
| `dataset.config`           | `str \| None`   | `null`         | Dataset config/split variant name.                                                                                 |
| `dataset.path`             | `str`           | `""`           | Local path loaded with `load_from_disk` when present.                                                              |
| `dataset.streaming`        | `bool`          | `true`         | Streaming mode for large datasets.                                                                                 |
| `dataset.max_seq_length`   | `int`           | `512`          | Target max sequence length for preprocessing/collation.                                                            |
| `dataset.text_column`      | `str \| None`   | `null`         | Text field override for tokenization.                                                                              |
| `dataset.train_split`      | `str \| None`   | `null`         | Train split (supports slice syntax).                                                                               |
| `dataset.eval_split`       | `str \| None`   | `null`         | Eval split override.                                                                                               |
| `dataset.eval_samples`     | `int \| None`   | `null`         | Eval sample cap. If no eval split is configured, trainer can reserve the first `eval_samples` from train for eval. |
| `dataset.validation_split` | `float \| None` | `null`         | Fraction for random eval split (non-streaming only).                                                               |

> [!NOTE]
> Streaming pretraining defaults to `dataset.eval_split: null`. When unset, trainer
> attempts to auto-detect a validation-style split (`validation`, `eval`, `test`, `dev`).
> If none exists and `dataset.eval_samples` is set, it builds eval from the first
> `eval_samples` training examples and skips those from the training stream to avoid
> leakage.

### Performance and Preprocessing

| Key                           | Type          | Default | Description                                                 |
| ----------------------------- | ------------- | ------- | ----------------------------------------------------------- |
| `dataset.num_workers`         | `int`         | `16`    | DataLoader worker count.                                    |
| `dataset.pin_memory`          | `bool`        | `false` | Explicit pin-memory preference (may be overridden on CUDA). |
| `dataset.persistent_workers`  | `bool`        | `true`  | Keep DataLoader workers alive across epochs.                |
| `dataset.prefetch_factor`     | `int \| None` | `null`  | Worker prefetch depth when workers > 0.                     |
| `dataset.num_proc`            | `int`         | `4`     | Multiprocessing workers for tokenization map.               |
| `dataset.shuffle_buffer_size` | `int`         | `10000` | Streaming shuffle buffer.                                   |
| `dataset.pre_tokenize`        | `bool`        | `false` | Pre-tokenize non-streaming datasets and persist results.    |
| `dataset.pre_tokenize_output` | `str \| None` | `null`  | Output path for pre-tokenized datasets.                     |
| `dataset.cache_dir`           | `str \| None` | `null`  | HF datasets cache directory.                                |
| `dataset.trust_remote_code`   | `bool`        | `false` | Allow remote dataset code execution.                        |

### Contrastive-Only Data Fields

| Key                          | Type    | Default | Description                                                           |
| ---------------------------- | ------- | ------- | --------------------------------------------------------------------- |
| `dataset.load_all_from_disk` | `bool`  | `false` | Load full dataset into memory.                                        |
| `dataset.force_redownload`   | `bool`  | `false` | Force dataset redownload.                                             |
| `dataset.min_length`         | `int`   | `5`     | Short-text-friendly default for optional length filtering helpers.    |
| `dataset.alpha`              | `float` | `1.0`   | Contrastive dataset sampling exponent (`1.0` = proportional by size). |

> [!NOTE]
> `dataset.pretraining_prob` is deprecated and normalized to
> `contrastive.pretraining_prob`.

---

## Training Loop

### Core

| Key                                   | Type  | Default      | Description                                   |
| ------------------------------------- | ----- | ------------ | --------------------------------------------- |
| `trainer.per_device_train_batch_size` | `int` | `16`         | Train microbatch size per device.             |
| `trainer.per_device_eval_batch_size`  | `int` | `32`         | Eval microbatch size per device.              |
| `trainer.gradient_accumulation_steps` | `int` | `1`          | Accumulation steps per optimizer update.      |
| `trainer.max_steps`                   | `int` | `1000000`    | Max optimizer steps.                          |
| `trainer.save_steps`                  | `int` | `10000`      | Save interval in steps.                       |
| `trainer.eval_steps`                  | `int` | `10000`      | Eval interval in steps.                       |
| `trainer.logging_steps`               | `int` | `100`        | Logging interval in steps.                    |
| `trainer.output_dir`                  | `str` | `"./output"` | Output root for checkpoints and artifacts.    |
| `trainer.mixed_precision`             | `str` | `"bf16"`     | `no`, `fp32`, `bf16`, or `fp8` (`fp8` is pretraining-only; `fp16` unsupported). |

### Stability and Performance

| Key                                   | Type            | Default      | Description                                                                                                                                                 |
| ------------------------------------- | --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trainer.gradient_checkpointing`      | `bool`          | `false`      | Activation checkpointing for lower memory usage.                                                                                                            |
| `trainer.gradient_clipping`           | `float \| None` | `null`       | Clip gradient norm when set.                                                                                                                                |
| `trainer.torch_compile`               | `bool`          | `false`      | Enable `torch.compile`.                                                                                                                                     |
| `trainer.torch_compile_dynamic`       | `bool \| None`  | `null`       | Dynamic-shape compile toggle when supported.                                                                                                                |
| `trainer.torch_compile_backend`       | `str`           | `"inductor"` | Compile backend name.                                                                                                                                       |
| `trainer.enforce_full_packed_batches` | `bool`          | `true`       | Buffer packed fragments to emit full-sized microbatches.                                                                                                    |
| `trainer.eval_max_batches`            | `int \| None`   | `null`       | Optional eval cap; required for streaming eval when `dataset.eval_samples` is unset.                                                                        |
| `trainer.log_train_accuracy`          | `bool`          | `false`      | Log MLM masked-token train accuracy (enable only for focused diagnostics; disabling improves throughput).                                                   |
| `trainer.log_grad_norm`               | `bool`          | `true`       | Log grad norm each logging interval.                                                                                                                        |
| `trainer.log_weight_norms`            | `bool`          | `true`       | Log parameter norms (main-process overhead).                                                                                                                |
| `trainer.tf32`                        | `bool`          | `true`       | Enable TF32 on supported CUDA GPUs.                                                                                                                         |
| `trainer.masked_logits_only_loss`     | `bool`          | `true`       | Pretraining MLM loss path selector: `true` = masked-logits-only path (default/recommended), `false` = original full-logits CE path (legacy ablation/debug). |

> [!IMPORTANT]
> `trainer.masked_logits_only_loss` is a run-level path selector, not a
> multi-objective mixing interface. Choose one path for the run.
> The project default is `true`; use `false` only when intentionally running a
> legacy full-logits baseline.

### FP8 (Pretraining Only)

| Key                                             | Type            | Default        | Description                                                                                          |
| ----------------------------------------------- | --------------- | -------------- | ---------------------------------------------------------------------------------------------------- |
| `trainer.fp8.recipe`                            | `str`           | `"tensorwise"` | FP8 recipe: `tensorwise`, `rowwise`, `rowwise_with_gw_hp`.                                          |
| `trainer.fp8.filter_fqns`                       | `list[str]`     | `[]`           | Module-name substrings to exclude from FP8 linear conversion.                                       |
| `trainer.fp8.auto_filter_small_kn`              | `bool`          | `false`        | Enable torchao auto-filter heuristics for small GEMMs.                                               |
| `trainer.fp8.enable_fsdp_float8_all_gather`     | `bool \| None`  | `null`         | Tensorwise-only FSDP2 float8 all-gather toggle (`null` resolves to `true` in tensorwise mode).      |
| `trainer.fp8.pad_inner_dim`                     | `bool \| None`  | `null`         | Tensorwise-only inner-dim padding for fp8 `_scaled_mm` (`null` resolves to `true` in tensorwise).   |
| `trainer.fp8.use_regional_compilation`          | `bool`          | `true`         | Enable Accelerate TorchDynamo regional compilation in FP8 path.                                     |

> [!IMPORTANT]
> `trainer.mixed_precision: fp8` currently requires pretraining + FSDP2, and
> rowwise recipes cannot enable `trainer.fp8.enable_fsdp_float8_all_gather`.

### Control and Legacy Compatibility

| Key                              | Type          | Default   | Description                                               |
| -------------------------------- | ------------- | --------- | --------------------------------------------------------- |
| `trainer.resume_from_checkpoint` | `str \| None` | `null`    | Resume checkpoint selector/path.                          |
| `trainer.overwrite_output_dir`   | `bool`        | `true`    | Overwrite output directory behavior.                      |
| `trainer.num_train_epochs`       | `int`         | `3`       | Epoch count fallback when steps are not the only limiter. |
| `trainer.eval_strategy`          | `str`         | `"steps"` | `steps` or `epoch`.                                       |
| `trainer.save_strategy`          | `str`         | `"steps"` | `steps`, `epoch`, `best`, or `no`.                        |
| `trainer.save_total_limit`       | `int \| None` | `3`       | Keep at most this many `checkpoints/<step>` directories.  |
| `trainer.max_ckpt`               | `int \| None` | `null`    | Deprecated alias for `trainer.save_total_limit`.          |
| `trainer.disable_tqdm`           | `bool`        | `false`   | Disable progress bars.                                    |
| `trainer.dataloader_num_workers` | `int`         | `0`       | Contrastive-only dataloader worker override.              |
| `trainer.use_cpu`                | `bool`        | `false`   | Force CPU execution.                                      |
| `trainer.report_to`              | `list[str]`   | `[]`      | Deprecated and ignored. Use `wandb.enabled` explicitly.   |
| `trainer.train_batch_size`       | `int \| None` | `null`    | Legacy batch-size alias.                                  |
| `trainer.eval_batch_size`        | `int \| None` | `null`    | Legacy batch-size alias.                                  |
| `trainer.early_stopping`         | `int`         | `0`       | Reserved in pretraining path.                             |
| `trainer.metric_for_best_model`  | `str \| None` | `null`    | Reserved in pretraining path.                             |
| `trainer.greater_is_better`      | `bool`        | `true`    | Reserved in pretraining path.                             |
| `trainer.load_best_model_at_end` | `bool`        | `false`   | Reserved in pretraining path.                             |
| `trainer.save_model`             | `bool`        | `true`    | Reserved/compatibility field.                             |

---

## LR Schedule

| Key                        | Type            | Default    | Description                              |
| -------------------------- | --------------- | ---------- | ---------------------------------------- |
| `scheduler.name`           | `str`           | `"cosine"` | LR schedule family.                      |
| `scheduler.warmup_steps`   | `int`           | `10000`    | Absolute warmup steps.                   |
| `scheduler.total_steps`    | `int \| None`   | `null`     | Optional explicit total schedule length. |
| `scheduler.decay_steps`    | `int \| None`   | `null`     | Absolute decay end step.                 |
| `scheduler.warmup_percent` | `float \| None` | `null`     | Percentage override for warmup.          |
| `scheduler.decay_percent`  | `float \| None` | `null`     | Percentage override for decay.           |
| `scheduler.final_lr_ratio` | `float`         | `0.1`      | Final LR floor ratio.                    |

> [!IMPORTANT]
> `warmup_percent` overrides `warmup_steps`; `decay_percent` overrides `decay_steps`.

---

## Optimizer

### Base Optimizer

| Key                      | Type                 | Default        | Description                                              |
| ------------------------ | -------------------- | -------------- | -------------------------------------------------------- |
| `optimizer.name`         | `str`                | `"adamw"`      | `adamw`, `adam`, or `muonclip`.                          |
| `optimizer.lr`           | `float`              | `1e-4`         | Base learning rate.                                      |
| `optimizer.weight_decay` | `float`              | `0.01`         | Weight decay.                                            |
| `optimizer.betas`        | `list[float]`        | `[0.9, 0.999]` | Adam-family beta coefficients.                           |
| `optimizer.eps`          | `float`              | `1e-8`         | Adam-family epsilon.                                     |
| `optimizer.muon_config`  | `MuonConfig \| None` | `null`         | MuonClip settings (used when `optimizer.name=muonclip`). |

### MuonClip (`optimizer.muon_config`)

| Key                            | Type             | Default           | Description                                                  |
| ------------------------------ | ---------------- | ----------------- | ------------------------------------------------------------ |
| `muon_beta`                    | `float`          | `0.95`            | Muon momentum coefficient.                                   |
| `muon_decay`                   | `float`          | `0.0`             | Muon weight decay.                                           |
| `ns_steps`                     | `int`            | `5`               | Newton-Schulz/Polar iterations.                              |
| `enable_clipping`              | `bool`           | `true`            | Enable QK clipping path.                                     |
| `clipping_threshold`           | `float`          | `50.0`            | QK clipping threshold.                                       |
| `clipping_alpha`               | `float`          | `0.5`             | Q/K scaling balance parameter.                               |
| `clipping_warmup_steps`        | `int`            | `0`               | Disable clipping before this many steps.                     |
| `clipping_interval`            | `int`            | `10`              | Apply clipping every N update steps.                         |
| `clipping_qk_chunk_size`       | `int`            | `1024`            | Chunk size for logit-max computation.                        |
| `capture_last_microbatch_only` | `bool`           | `true`            | Capture activations only for final microbatch in GA window.  |
| `detect_anomalies`             | `bool`           | `false`           | Enable anomaly checks in optimizer step.                     |
| `orthogonalization`            | `str`            | `"polar_express"` | Orthogonalization algorithm selector.                        |
| `algorithm`                    | `str \| None`    | `null`            | Deprecated alias of `orthogonalization`.                     |
| `polar_express`                | `bool \| None`   | `null`            | Deprecated legacy toggle.                                    |
| `clipping_layers_mapping`      | `dict[str, str]` | `{}`              | Projection-name overrides for non-standard attention blocks. |

---

## Data Collator

| Key                               | Type          | Default | Description                                                    |
| --------------------------------- | ------------- | ------- | -------------------------------------------------------------- |
| `datacollator.mlm_probability`    | `float`       | `0.15`  | Probability of selecting tokens for MLM corruption.            |
| `datacollator.mask_all`           | `bool`        | `false` | `false`: standard 80/10/10; `true`: 100% `[MASK]` replacement. |
| `datacollator.pack_sequences`     | `bool`        | `false` | Enable sequence packing.                                       |
| `datacollator.max_length`         | `int \| None` | `null`  | Packed target length override.                                 |
| `datacollator.pad_to_multiple_of` | `int \| None` | `null`  | Pad to multiple for kernel efficiency in non-packed mode.      |

For `p = datacollator.mlm_probability`:

- `mask_all: false` global token mix is `(1 - p)` untouched, `0.8p` `[MASK]`,
  `0.1p` random-token, `0.1p` original-token.
- `mask_all: true` global token mix is `(1 - p)` untouched, `p` `[MASK]`.

---

## Checkpointing and Resume

Save cadence/retention knobs live under [Training Loop](#training-loop):
`trainer.save_steps`, `trainer.save_total_limit`, and
`trainer.resume_from_checkpoint`.

| Key                     | Type  | Default    | Description                               |
| ----------------------- | ----- | ---------- | ----------------------------------------- |
| `pretrained_checkpoint` | `str` | `"latest"` | Checkpoint selector for downstream tasks. |

> [!NOTE]
> Pretraining and GLUE resumable state checkpoints are written under
> `output_dir/checkpoints/<step>/`.
> GLUE transfer/loading helpers still accept legacy
> `output_dir/model_checkpoints/<step>/` layouts for older runs.
> Resume path resolution uses numeric step directories and picks the highest
> available step for `resume_from_checkpoint: latest`.
> DeepSpeed `latest` indirection files are optional legacy metadata and are only
> consulted by DeepSpeed conversion/loading helpers when present.

---

## Logging and Tracking

### Weights and Biases

| Key                  | Type          | Default        | Description                                                                                |
| -------------------- | ------------- | -------------- | ------------------------------------------------------------------------------------------ |
| `wandb.enabled`      | `bool`        | `false`        | Enable W&B logging.                                                                        |
| `wandb.project`      | `str`         | `"neo-bert"`   | W&B project name.                                                                          |
| `wandb.entity`       | `str \| None` | `null`         | W&B entity/team.                                                                           |
| `wandb.name`         | `str \| None` | `null`         | Run name override.                                                                         |
| `wandb.tags`         | `list[str]`   | `[]`           | Run tags.                                                                                  |
| `wandb.mode`         | `str`         | `"online"`     | `online`, `offline`, or `disabled`.                                                        |
| `wandb.watch`        | `str`         | `"gradients"`  | Model-watch mode: `gradients`, `parameters`, `all`, or disabled (`off`/`none`/`disabled`). |
| `wandb.log_interval` | `int`         | `100`          | Legacy field; trainer logging uses `trainer.logging_steps`.                                |
| `wandb.resume`       | `str`         | `"never"`      | W&B resume policy.                                                                         |
| `wandb.dir`          | `str`         | `"logs/wandb"` | Artifact/run directory.                                                                    |

> [!NOTE]
> Runtime logging prints a task-scoped resolved config before training and sends
> the same task-scoped payload to W&B (irrelevant task sections are excluded).
> W&B is not auto-enabled by presence of a `wandb` section; set
> `wandb.enabled: true` explicitly.
> For pretraining/contrastive, watch-mode precedence is:
> `WANDB_WATCH` env var > `wandb.watch` config > default (`gradients` for
> `wandb.mode: online`).

### Top-Level Runtime Metadata

| Key                      | Type             | Default | Description                                                              |
| ------------------------ | ---------------- | ------- | ------------------------------------------------------------------------ |
| `seed`                   | `int`            | `0`     | Global random seed.                                                      |
| `debug`                  | `bool`           | `false` | Extra debug logging/prints.                                              |
| `use_deepspeed`          | `bool`           | `false` | Legacy hint for DeepSpeed-formatted contrastive checkpoint loading only. |
| `accelerate_config_file` | `str \| None`    | `null`  | Accelerate launch config path.                                           |
| `pretraining_metadata`   | `dict[str, Any]` | `{}`    | Metadata passed to downstream evaluations.                               |
| `config_path`            | `str \| None`    | `null`  | Source config path metadata.                                             |

---

## Task-Specific Sections

### GLUE (`glue`)

| Key                              | Type                 | Default  | Description                                              |
| -------------------------------- | -------------------- | -------- | -------------------------------------------------------- |
| `glue.task_name`                 | `str`                | `"cola"` | GLUE task identifier.                                    |
| `glue.num_labels`                | `int`                | `2`      | Number of target labels.                                 |
| `glue.max_seq_length`            | `int`                | `128`    | Token length for GLUE preprocessing.                     |
| `glue.pretrained_model_path`     | `str \| None`        | `null`   | Path to pretrained model config.                         |
| `glue.pretrained_checkpoint_dir` | `str \| None`        | `null`   | Directory containing checkpoints.                        |
| `glue.pretrained_checkpoint`     | `str \| int \| None` | `null`   | Specific checkpoint selector.                            |
| `glue.allow_random_weights`      | `bool`               | `false`  | Allow evaluation/fine-tuning without pretrained weights. |
| `glue.classifier_dropout`        | `float`              | `0.1`    | Classifier dropout.                                      |
| `glue.classifier_init_range`     | `float`              | `0.02`   | Classifier init stddev.                                  |
| `glue.transfer_from_task`        | `bool`               | `false`  | Transfer weights from another GLUE task head.            |
| `glue.num_workers`               | `int`                | `4`      | Data worker count for GLUE pipeline.                     |
| `glue.preprocessing_num_proc`    | `int`                | `4`      | Multiprocessing workers for GLUE preprocessing.          |

> [!NOTE]
> Worker-count knobs are task-scoped in the current runtime:
> pretraining uses `dataset.num_workers`, GLUE uses `glue.num_workers`, and
> contrastive uses `trainer.dataloader_num_workers`.

### Contrastive (`contrastive`)

| Key                                     | Type                 | Default    | Description                                                                 |
| --------------------------------------- | -------------------- | ---------- | --------------------------------------------------------------------------- |
| `contrastive.temperature`               | `float`              | `0.05`     | Contrastive temperature.                                                    |
| `contrastive.pooling`                   | `str`                | `"avg"`    | Pooling mode: `avg`, `cls`, `max`.                                          |
| `contrastive.loss_type`                 | `str`                | `"simcse"` | Loss variant: `simcse`, `supcon`.                                           |
| `contrastive.hard_negative_weight`      | `float`              | `0.0`      | Additional hard-negative weighting.                                         |
| `contrastive.pretraining_prob`          | `float`              | `0.3`      | Fraction of steps that draw the pretraining branch in contrastive training. |
| `contrastive.pretrained_checkpoint_dir` | `str \| None`        | `null`     | Optional pretraining checkpoint root used to initialize contrastive runs.   |
| `contrastive.pretrained_checkpoint`     | `str \| int \| None` | `null`     | Optional checkpoint tag/step selector for contrastive initialization.       |
| `contrastive.allow_random_weights`      | `bool`               | `false`    | Allow random initialization when no pretrained checkpoint is configured.    |

### MTEB Top-Level Keys

| Key                      | Type   | Default  | Description                           |
| ------------------------ | ------ | -------- | ------------------------------------- |
| `mteb_task_type`         | `str`  | `"all"`  | MTEB subset selector.                 |
| `mteb_batch_size`        | `int`  | `32`     | MTEB inference batch size.            |
| `mteb_pooling`           | `str`  | `"mean"` | Pooling for embedding extraction.     |
| `mteb_overwrite_results` | `bool` | `false`  | Overwrite existing MTEB output files. |

---

## Constraints, Requirements, and Gotchas

| Rule                                                                              | Type               | Details                                                                                                             |
| --------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `trainer.resume_from_checkpoint` with `dataset.streaming=true`                    | **BEST-EFFORT**    | Streaming resume restores state and advances stream by consumed batches; exact sample continuity is not guaranteed. |
| Streaming eval with neither `trainer.eval_max_batches` nor `dataset.eval_samples` | **ERROR**          | Set an explicit eval budget for reproducible streaming metrics.                                                     |
| `dataset.validation_split` with `dataset.streaming=true`                          | **WARNING / SKIP** | Validation split creation is skipped for streaming datasets.                                                        |
| `scheduler.warmup_percent` and `scheduler.warmup_steps`                           | **PRECEDENCE**     | `warmup_percent` overrides absolute warmup steps.                                                                   |
| `scheduler.decay_percent` and `scheduler.decay_steps`                             | **PRECEDENCE**     | `decay_percent` overrides absolute decay steps.                                                                     |
| `optimizer.name=muonclip` with DeepSpeed ZeRO stage >= 2                          | **ERROR**          | MuonClip is incompatible with sharded grads/params at ZeRO stage >= 2.                                              |
| `datacollator.pack_sequences=true` with `model.attn_backend=sdpa`                 | **WARNING**        | Works, but slower than `flash_attn_varlen`; SDPA uses fallback path.                                                |
| `dataset.path` and `dataset.name` both set                                        | **PRECEDENCE**     | Existing local `dataset.path` is used first; hub dataset acts as fallback.                                          |
| Tokenizer/model vocab sizes                                                       | **IMPORTANT**      | Runtime now pads tokenizer with inert tokens so tokenizer length matches model vocab size.                          |
| `model.pad_token_id`                                                              | **IMPORTANT**      | Runtime syncs this from tokenizer before model init/checkpoint save.                                                |

---

## Legacy Key Mapping (Still Normalized)

| Legacy Key                         | Canonical Key                  | Behavior                                     |
| ---------------------------------- | ------------------------------ | -------------------------------------------- |
| top-level `mixed_precision`        | `trainer.mixed_precision`      | Deprecated alias; normalized with warning.   |
| `trainer.bf16`                     | `trainer.mixed_precision`      | Deprecated alias; normalized with warning.   |
| `trainer.seed`                     | top-level `seed`               | Deprecated alias; normalized with warning.   |
| `trainer.run_name`                 | `wandb.name`                   | Deprecated alias; normalized with warning.   |
| `trainer.learning_rate`            | `optimizer.lr`                 | Deprecated alias; normalized with warning.   |
| `trainer.warmup_steps`             | `scheduler.warmup_steps`       | Deprecated alias; normalized with warning.   |
| `trainer.max_grad_norm`            | `trainer.gradient_clipping`    | Deprecated alias; normalized with warning.   |
| `trainer.dir`                      | `trainer.output_dir`           | Deprecated alias; normalized with warning.   |
| `dataset.tokenizer_name`           | `tokenizer.name`               | Deprecated alias; normalized with warning.   |
| `dataset.column`                   | `dataset.text_column`          | Deprecated alias; normalized with warning.   |
| `dataset.path_to_disk`             | `dataset.path`                 | Deprecated alias; normalized with warning.   |
| `dataset.pretraining_prob`         | `contrastive.pretraining_prob` | Deprecated alias; normalized with warning.   |
| `tokenizer.tokenizer_name_or_path` | `tokenizer.name`               | Deprecated alias; normalized with warning.   |
| `optimizer.hparams.*`              | `optimizer.*`                  | Deprecated block; flattened with warning.    |
| legacy attention booleans          | `model.attn_backend`           | Deprecated aliases; normalized with warning. |

---

## Practical YAML Presets

### 1) Base Pretraining (Balanced)

```yaml
task: pretraining
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
  attn_backend: flash_attn_varlen
  kernel_backend: auto

dataset:
  name: refinedweb
  streaming: true
  max_seq_length: 512
  num_workers: 16

tokenizer:
  name: bert-base-uncased
  max_length: 512

trainer:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 1
  max_steps: 1000000
  save_steps: 10000
  eval_steps: 10000
  mixed_precision: bf16
  masked_logits_only_loss: true

optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_steps: 10000

datacollator:
  mlm_probability: 0.15
  mask_all: false
  pack_sequences: false
```

### 2) Memory-Constrained Single GPU

```yaml
task: pretraining
model:
  hidden_size: 512
  num_hidden_layers: 8
  num_attention_heads: 8
  intermediate_size: 2048
  max_position_embeddings: 512
  attn_backend: sdpa

trainer:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  mixed_precision: bf16
  masked_logits_only_loss: true
  torch_compile: false
  max_steps: 300000

optimizer:
  name: adamw
  lr: 8e-5

scheduler:
  name: cosine
  warmup_percent: 1.0

dataset:
  streaming: true
  num_workers: 4
```

### 3) Resumable Local-Data Run

```yaml
task: pretraining
dataset:
  path: /data/neobert_tokenized
  streaming: false
  max_seq_length: 512

trainer:
  output_dir: outputs/neobert_run
  resume_from_checkpoint: latest
  save_steps: 5000
  save_total_limit: 5
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
```

### 4) Full Logging + Frequent Eval

```yaml
task: pretraining
wandb:
  enabled: true
  project: neo-bert
  mode: online
  name: neobert-full-logging

trainer:
  logging_steps: 20
  eval_steps: 1000
  eval_max_batches: 200
  log_train_accuracy: false
  log_grad_norm: true
  log_weight_norms: true

scheduler:
  name: cosine
  warmup_steps: 5000
```

---

## Related Docs

- [Training](training.md)
- [Testing](testing.md)
- [Troubleshooting](troubleshooting.md)
