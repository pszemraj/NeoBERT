# Configuration Reference

> [!TIP]
> Example configs are in [configs/README.md](../../configs/README.md) (for production) and [tests/configs/README.md](../../tests/configs/README.md) (for tiny smoke/regression runs).

NeoBERT YAML config schema (`src/neobert/config.py`) and defaults.

## Variables and Dot Overrides

`ConfigLoader.load(...)` supports a small YAML variable system and post-load dot overrides for sweep-style runs.

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

Example: one `seq_len` driving multiple runtime fields (without coupling model context length yet):

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

Use this pattern for shared run-time sequence settings. Keep `model.max_position_embeddings` as an explicit architecture decision.

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

Unknown paths and invalid value types fail fast with path-specific errors. Overrides are validated with the same semantic checks as base YAML configs.

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
| `model.hidden_act`              | `str`         | `"swiglu"` | See [Feed-Forward](architecture.md#feed-forward).             |
| `model.dropout_prob`            | `float`       | `0.0`      | Dropout probability in model blocks.                         |
| `model.norm_eps`                | `float`       | `1e-5`     | Epsilon for normalization stability.                         |

### Advanced

| Key                           | Type    | Default              | Description                                       |
| ----------------------------- | ------- | -------------------- | ------------------------------------------------- |
| `model.rms_norm`              | `bool`  | `true`               | Use RMSNorm (otherwise LayerNorm).                |
| `model.attn_backend`          | `str`   | `"sdpa"`             | Attention backend: `sdpa` or `flash_attn_varlen`. |
| `model.kernel_backend`        | `str`   | `"auto"`             | Kernel backend: `auto`, `liger`, or `torch`.      |
| `model.rope`                  | `bool`  | `true`               | Enable rotary positional encoding.                |
| `model.pad_token_id`          | `int`   | `0`                  | Runtime-synced from tokenizer.                    |
| `model.embedding_init_range`  | `float` | `0.02`               | Embedding init stddev.                            |
| `model.decoder_init_range`    | `float` | `0.02`               | Decoder init stddev.                              |
| `model.classifier_init_range` | `float` | `0.02`               | Classifier head init stddev.                      |
| `model.from_hub`              | `bool`  | `false`              | Load a Hub sequence classifier in the GLUE path.  |

---

## Tokenizer

| Key                                     | Type          | Default               | Description                                                                           |
| --------------------------------------- | ------------- | --------------------- | ------------------------------------------------------------------------------------- |
| `tokenizer.name`                        | `str`         | `"bert-base-uncased"` | Tokenizer name from HF hub.                                                           |
| `tokenizer.path`                        | `str \| None` | `null`                | Local tokenizer path (takes precedence when provided).                                |
| `tokenizer.max_length`                  | `int`         | `512`                 | Tokenizer max length used during preprocessing.                                       |
| `tokenizer.padding`                     | `str`         | `"max_length"`        | Stored compatibility field; current tokenization is unpadded and collators add padding. |
| `tokenizer.truncation`                  | `bool`        | `true`                | Truncate to max length during tokenization.                                           |
| `tokenizer.vocab_size`                  | `int \| None` | `null`                | Runtime-synchronized to effective model vocab size.                                   |
| `tokenizer.trust_remote_code`           | `bool`        | `false`               | Allow tokenizer remote code execution.                                                |
| `tokenizer.revision`                    | `str \| None` | `null`                | Optional tokenizer revision/commit pin for reproducibility.                           |
| `tokenizer.allow_special_token_rewrite` | `bool`        | `false`               | Explicit opt-in for fallback special-token rewrite when tokenizer lacks `mask_token`. |

> [!NOTE]
> Pretraining rounds tokenizer length up to a multiple of 128 and adds deterministic `<|neobert_extra_token_{id}|>` placeholders so `len(tokenizer) == model.vocab_size`. Export aligns the tokenizer to the checkpoint's explicit model vocabulary and rejects shrinking or partial insertion. If a tokenizer lacks `mask_token`, set `tokenizer.allow_special_token_rewrite: true` before NeoBERT mutates special tokens.

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
> Streaming eval-split resolution (auto-detection, `eval_samples` reservation, required budgets) is described in [Streaming Eval Strategy](../guides/training.md#streaming-eval-strategy).

### Performance and Preprocessing

| Key                           | Type          | Default | Description                                                 |
| ----------------------------- | ------------- | ------- | ----------------------------------------------------------- |
| `dataset.num_workers`         | `int`         | `16`    | DataLoader worker count.                                    |
| `dataset.pin_memory`          | `bool`        | `false` | Enable pinned CPU staging for non-blocking H2D copies; CUDA runs may force it on. Pinning-path behavior: [Training Optimization](../guides/training-optimization.md#dataloader-and-streaming-throughput). |
| `dataset.persistent_workers`  | `bool`        | `true`  | Keep DataLoader workers alive across epochs.                |
| `dataset.prefetch_factor`     | `int \| None` | `null`  | Worker prefetch depth when workers > 0.                     |
| `dataset.streaming_read_retries` | `int`      | `4`     | Outer retry count for transient streaming read failures after the underlying HF client exhausts its own per-request retries. Recovery semantics: [Training Optimization](../guides/training-optimization.md#dataloader-and-streaming-throughput). |
| `dataset.streaming_read_retry_backoff_seconds` | `float` | `5.0` | Initial exponential-backoff delay for transient streaming read retries. |
| `dataset.streaming_read_retry_max_backoff_seconds` | `float` | `60.0` | Maximum capped backoff delay for transient streaming read retries. |
| `dataset.num_proc`            | `int`         | `4`     | Multiprocessing workers for tokenization map.               |
| `dataset.shuffle_buffer_size` | `int`         | `10000` | Streaming shuffle buffer.                                   |
| `dataset.cache_dir`           | `str \| None` | `null`  | HF datasets cache directory.                                |
| `dataset.trust_remote_code`   | `bool`        | `false` | Allow remote dataset code execution.                        |

### Contrastive-Only Data Fields

| Key                          | Type    | Default | Description                                                           |
| ---------------------------- | ------- | ------- | --------------------------------------------------------------------- |
| `dataset.load_all_from_disk` | `bool`  | `false` | Reuse selected cached contrastive splits from `<dataset.path>/all/`.  |
| `dataset.force_redownload`   | `bool`  | `false` | Force dataset redownload.                                             |
| `dataset.min_length`         | `int`   | `5`     | Short-text-friendly default for optional length filtering helpers.    |
| `dataset.alpha`              | `float` | `1.0`   | Contrastive dataset sampling exponent (`1.0` = proportional by size). |

> [!NOTE]
> `dataset.pretraining_prob` is deprecated and normalized to `contrastive.pretraining_prob`.
>
> Contrastive preprocessing accepts an omitted `dataset.name`, `dataset.name: ALL`, canonical registry keys such as `ALLNLI`, or common HF dataset IDs from the built-in wrapper registry (for example `sentence-transformers/all-nli`, `embedding-data/QQP_triplets`, or `WhereIsAI/github-issue-similarity`). Cached split directories under `all/` are loaded only for the requested selection, missing splits fail fast, and tokenization manifests are validated in both preprocessing and training, including when `dataset.load_all_from_disk=true`. Subset preprocess refreshes preserve other cached split entries already present under `all/`.

---

## Training Loop

### Core

| Key                                   | Type  | Default      | Description                                   |
| ------------------------------------- | ----- | ------------ | --------------------------------------------- |
| `trainer.per_device_train_batch_size` | `int` | `16`         | Train microbatch size per device.             |
| `trainer.per_device_eval_batch_size`  | `int` | `32`         | Eval microbatch size per device.              |
| `trainer.gradient_accumulation_steps` | `int` | `1`          | Microbatches per optimizer update.            |
| `trainer.max_steps`                   | `int` | `1000000`    | Max optimizer steps.                          |
| `trainer.save_steps`                  | `int` | `10000`      | Save interval in steps.                       |
| `trainer.eval_steps`                  | `int` | `10000`      | Eval interval in steps.                       |
| `trainer.logging_steps`               | `int` | `100`        | Logging interval in steps.                    |
| `trainer.output_dir`                  | `str` | `"./output"` | Output root for checkpoints and artifacts.    |
| `trainer.mixed_precision`             | `str` | `"bf16"`     | `no`, `fp32`, or `bf16` (`fp16` unsupported; `fp32` normalizes to `no`). |

### Stability and Performance

| Key                                   | Type            | Default      | Description                                                                                                                                                 |
| ------------------------------------- | --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trainer.gradient_checkpointing`      | `bool`          | `false`      | Activation checkpointing for lower memory usage.                                                                                                            |
| `trainer.gradient_clipping`           | `float \| None` | `null`       | Clip the global gradient norm after accumulation; pretraining applies it after masked-token rescaling.                                                      |
| `trainer.torch_compile`               | `bool`          | `false`      | Enable `torch.compile`.                                                                                                                                     |
| `trainer.torch_compile_dynamic`       | `bool \| None`  | `null`       | Dynamic-shape compile toggle when supported.                                                                                                                |
| `trainer.torch_compile_backend`       | `str`           | `"inductor"` | Compile backend name.                                                                                                                                       |
| `trainer.enforce_full_packed_batches` | `bool`          | `false`      | Single-process-only buffering of undersized packed rows into full microbatches; distributed runs reject `true`.                                             |
| `trainer.eval_max_batches`            | `int \| None`   | `null`       | Optional eval cap; required for streaming eval when `dataset.eval_samples` is unset.                                                                        |
| `trainer.log_train_accuracy`          | `bool`          | `false`      | Log MLM masked-token train accuracy (enable only for focused diagnostics; disabling improves throughput).                                                   |
| `trainer.log_grad_norm`               | `bool`          | `true`       | Log the global pre-clip gradient norm at each logging interval.                                                                                             |
| `trainer.log_weight_norms`            | `bool`          | `true`       | Log the global parameter L2 norm on logging steps after the optimizer update.                                                                               |
| `trainer.tf32`                        | `bool`          | `true`       | Enable TF32 on supported CUDA GPUs.                                                                                                                         |
| `trainer.masked_logits_only_loss`     | `bool`          | `true`       | Pretraining MLM loss path selector: `true` = masked-logits-only path (default/recommended), `false` = original full-logits CE path (legacy ablation/debug). |

> [!IMPORTANT]
> `trainer.masked_logits_only_loss` is a run-level path selector, not a multi-objective mixing interface. Choose one path for the run. The project default is `true`; use `false` only when intentionally running a legacy full-logits baseline.
>
> Gradient-accumulation, effective-batch, and norm-logging behavior is detailed in [Training Optimization](../guides/training-optimization.md#gradient-accumulation-and-logged-norms).

### Training Control

| Key                              | Type          | Default   | Description                                               |
| -------------------------------- | ------------- | --------- | --------------------------------------------------------- |
| `trainer.resume_from_checkpoint` | `str \| None` | `null`    | Resume checkpoint selector/path.                          |
| `trainer.overwrite_output_dir`   | `bool`        | `true`    | Stored compatibility field; current trainers do not consume it.      |
| `trainer.num_train_epochs`       | `int`         | `3`       | GLUE epoch count when `trainer.max_steps <= 0`; other trainers ignore it. |
| `trainer.eval_strategy`          | `str`         | `"steps"` | `steps` or `epoch` for pretraining/GLUE; contrastive has no eval loop. |
| `trainer.save_strategy`          | `str`         | `"steps"` | Pretraining/contrastive: `steps` or `no`; GLUE also accepts `epoch` and `best`. |
| `trainer.save_total_limit`       | `int \| None` | `3`       | Retained step checkpoints; `0` or `null` disables pruning.            |
| `trainer.disable_tqdm`           | `bool`        | `false`   | Disable progress bars.                                    |
| `trainer.dataloader_num_workers` | `int`         | `0`       | Contrastive-only dataloader worker override.              |
| `trainer.use_cpu`                | `bool`        | `false`   | Force CPU execution.                                      |
| `trainer.early_stopping`         | `int`         | `0`       | GLUE evaluation cycles without improvement before stopping; other trainers ignore it. |
| `trainer.save_model`             | `bool`        | `true`    | Enable checkpoint writes in all trainers.                 |

---

## LR Schedule

| Key                        | Type            | Default    | Description                              |
| -------------------------- | --------------- | ---------- | ---------------------------------------- |
| `scheduler.name`           | `str`           | `"cosine"` | LR schedule family.                      |
| `scheduler.warmup_steps`   | `int`           | `10000`    | Absolute warmup steps.                   |
| `scheduler.total_steps`    | `int \| None`   | `null`     | Schedule phase length; does not cap training. |
| `scheduler.decay_steps`    | `int \| None`   | `null`     | Absolute decay end step.                 |
| `scheduler.warmup_percent` | `float \| None` | `null`     | Percentage override for warmup.          |
| `scheduler.decay_percent`  | `float \| None` | `null`     | Percentage override for decay.           |
| `scheduler.final_lr_ratio` | `float`         | `0.1`      | Final LR floor ratio.                    |

> [!IMPORTANT]
> `trainer.max_steps` controls run duration. `scheduler.total_steps` controls percentage-based schedule phases; after decay ends, learning rate remains at `scheduler.final_lr_ratio`. `warmup_percent` overrides `warmup_steps`; `decay_percent` overrides `decay_steps`.

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
| `nesterov`                     | `bool`           | `true`            | Use standard Muon Nesterov momentum (`g + beta * buffer`).   |
| `muon_decay`                   | `float`          | `0.0`             | Muon weight decay.                                           |
| `ns_steps`                     | `int`            | `5`               | Newton-Schulz/Polar iterations.                              |
| `enable_clipping`              | `bool`           | `true`            | Enable MuonClip's QK clipping path (separate from `trainer.gradient_clipping`). |
| `clipping_threshold`           | `float`          | `50.0`            | QK clipping threshold.                                       |
| `clipping_alpha`               | `float`          | `0.5`             | Q/K scaling balance parameter.                               |
| `clipping_warmup_steps`        | `int`            | `0`               | Disable clipping before this many steps.                     |
| `clipping_interval`            | `int`            | `10`              | Apply clipping every N update steps.                         |
| `clipping_qk_chunk_size`       | `int`            | `1024`            | Chunk size for logit-max computation.                        |
| `detect_anomalies`             | `bool`           | `false`           | Enable anomaly checks in optimizer step.                     |
| `orthogonalization`            | `str`            | `"polar_express"` | Orthogonalization algorithm selector.                        |
| `norm_factor`                  | `str`            | `"neobert"` | Post-orthogonalization normalization (`neobert`, `muon_reference`, `spectral`, `match_rms_adamw`, `none`). |
| `param_policy`                 | `str`            | `"hidden_2d"` | Muon routing policy (`hidden_2d` is the shipped default and applies Muon only to hidden transformer matrices; `all_2d` remains available for explicit v0.1.3-scope compatibility tests). |
| `clipping_layers_mapping`      | `dict[str, str]` | `{}`              | Projection-name overrides for non-standard attention blocks. |

Muon routing, normalization, orthogonalization, clipping, and distributed constraints are described in [Training Optimization](../guides/training-optimization.md).

---

## Data Collator

| Key                               | Type          | Default | Description                                                    |
| --------------------------------- | ------------- | ------- | -------------------------------------------------------------- |
| `datacollator.mlm_probability`    | `float`       | `0.15`  | Probability of selecting tokens for MLM corruption.            |
| `datacollator.mask_all`           | `bool`        | `false` | `false`: standard 80/10/10; `true`: 100% `[MASK]` replacement. |
| `datacollator.pack_sequences`     | `bool`        | `false` | Enable sequence packing; inputs must omit outer special tokens. |
| `datacollator.max_length`         | `int \| None` | `null`  | Packed target length override.                                 |
| `datacollator.pad_to_multiple_of` | `int \| None` | `null`  | Pad to multiple for kernel efficiency in non-packed mode.      |

For `p = datacollator.mlm_probability`:

- `mask_all: false` global token mix is `(1 - p)` untouched, `0.8p` `[MASK]`, `0.1p` random-token, `0.1p` original-token.
- `mask_all: true` global token mix is `(1 - p)` untouched, `p` `[MASK]`.

---

## Checkpointing and Resume

Save cadence/retention knobs live under [Training Loop](#training-loop): `trainer.save_steps`, `trainer.save_total_limit`, and `trainer.resume_from_checkpoint`.

| Key                     | Type  | Default    | Description                               |
| ----------------------- | ----- | ---------- | ----------------------------------------- |
| `pretrained_checkpoint` | `str` | `"latest"` | Top-level checkpoint selector used by MTEB. |

Checkpoint layout, selectors, and resume behavior are described in [Training](../guides/training.md#checkpointing-and-resume). Downstream loading behavior is described in [Evaluation](../guides/evaluation.md) and [Export](../guides/export.md).

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
| `wandb.resume`       | `str`         | `"never"`      | W&B resume policy.                                                                         |
| `wandb.dir`          | `str`         | `"logs/wandb"` | Artifact/run directory.                                                                    |

> [!NOTE]
> Runtime logging prints a task-scoped resolved config before training and sends the same task-scoped payload to W&B (irrelevant task sections are excluded). W&B is not auto-enabled by presence of a `wandb` section; set `wandb.enabled: true` explicitly. For pretraining/contrastive, watch-mode precedence is: `WANDB_WATCH` env var > `wandb.watch` config > default (`gradients` for `wandb.mode: online`).

### Top-Level Runtime Metadata

| Key                      | Type             | Default | Description                                                              |
| ------------------------ | ---------------- | ------- | ------------------------------------------------------------------------ |
| `seed`                   | `int`            | `0`     | Global random seed.                                                      |
| `debug`                  | `bool`           | `false` | Extra debug logging/prints.                                              |
| `use_deepspeed`          | `bool`           | `false` | Legacy hint for DeepSpeed-formatted contrastive checkpoint loading only; requires the optional `legacy-checkpoints` extra when conversion is needed. |
| `accelerate_config_file` | `str \| None`    | `null`  | Logged metadata; launcher/runtime does not consume this field.            |
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
> Worker-count knobs are task-scoped in the current runtime: pretraining uses `dataset.num_workers`, GLUE uses `glue.num_workers`, and contrastive uses `trainer.dataloader_num_workers`.

### Contrastive (`contrastive`)

| Key                                     | Type                 | Default    | Description                                                                 |
| --------------------------------------- | -------------------- | ---------- | --------------------------------------------------------------------------- |
| `contrastive.temperature`               | `float`              | `0.05`     | Contrastive temperature.                                                    |
| `contrastive.pooling`                   | `str`                | `"avg"`    | Pooling mode used by contrastive training: `avg`, `cls`, `max`.             |
| `contrastive.pretraining_prob`          | `float`              | `0.0`      | Fraction of steps that draw the optional SimCSE pretraining branch.          |
| `contrastive.pretraining_dataset_path` | `str \| None`        | `null`     | Tokenized dataset used by the SimCSE branch; required when its probability is positive. |
| `contrastive.pretrained_checkpoint_dir` | `str \| None`        | `null`     | Optional pretraining checkpoint root used to initialize contrastive runs.   |
| `contrastive.pretrained_checkpoint`     | `str \| int \| None` | `null`     | Optional checkpoint tag/step selector for contrastive initialization.       |
| `contrastive.allow_random_weights`      | `bool`               | `false`    | Allow random initialization when no pretrained checkpoint is configured.    |

### MTEB Top-Level Keys

| Key                      | Type   | Default  | Description                           |
| ------------------------ | ------ | -------- | ------------------------------------- |
| `mteb_task_type`         | `str`  | `"all"`  | MTEB subset selector.                 |
| `mteb_batch_size`        | `int`  | `32`     | MTEB inference batch size.            |
| `mteb_pooling`           | `str`  | `"avg"`  | Pooling for embedding extraction (`avg`/`mean` or `cls`). |
| `mteb_overwrite_results` | `bool` | `false`  | Overwrite existing MTEB output files. |

---

## Constraints, Requirements, and Gotchas

| Rule                                                                              | Type               | Details                                                                                                             |
| --------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `trainer.resume_from_checkpoint` with `dataset.streaming=true`                    | **BEST-EFFORT**    | Streaming resume restores state and advances stream by consumed batches; exact sample continuity is not guaranteed. |
| `trainer.resume_from_checkpoint` with `dataset.streaming=false`                   | **ORDERED**        | Map-style resume reconstructs the saved epoch's sample order and raw-batch cursor when corpus, seed, per-device batch size, and world size are unchanged. Dynamic MLM masks are not bitwise replayed. |
| `trainer.resume_from_checkpoint` with checkpoint `config.yaml` drift              | **CHECKPOINT WINS** | Resume forces checkpoint model shape/semantics, tokenizer identity, masking, objective fields, and pretraining per-device batch size before constructing runtime objects; missing checkpoint config fails fast. Corpus identity stays launch-controlled but resets the data cursor and packed buffer when changed. For RoPE, model/tokenizer/dataset/collator context length stays launch-controlled. |
| `trainer.resume_from_checkpoint` with optimizer parameter-order drift             | **ERROR**          | Resume validates `optimizer_param_names.json` before loading optimizer state to avoid positional momentum/buffer corruption. |
| GLUE `trainer.resume_from_checkpoint`                                             | **SELF-CONTAINED** | The selected step supplies model/tokenizer artifacts, task/head semantics, seed, batch/accumulation geometry, optimizer/scheduler construction, loop cursor, cumulative loss, best selection score, early-stopping state, and recent metric snapshots. The original pretraining source is not reopened; distributed world-size drift is rejected. |
| Reusing cached contrastive splits                                                | **MANIFEST CHECK** | Cached contrastive splits without a matching `tokenization_manifest.json` are rejected; regenerate the cache with `scripts/contrastive/preprocess.py`. |
| Streaming eval with neither `trainer.eval_max_batches` nor `dataset.eval_samples` | **ERROR**          | Set an explicit eval budget for reproducible streaming metrics.                                                     |
| `dataset.validation_split` with `dataset.streaming=true`                          | **WARNING / SKIP** | Validation split creation is skipped for streaming datasets.                                                        |
| `scheduler.warmup_percent` and `scheduler.warmup_steps`                           | **PRECEDENCE**     | `warmup_percent` overrides absolute warmup steps.                                                                   |
| `scheduler.decay_percent` and `scheduler.decay_steps`                             | **PRECEDENCE**     | `decay_percent` overrides absolute decay steps.                                                                     |
| `trainer.mixed_precision=bf16` on a broken CUDA/PyTorch runtime                    | **ENVIRONMENT**    | NeoBERT does not override BLAS selection; use a known-good PyTorch build for the host or set `mixed_precision='no'`. |
| `trainer.use_cpu=true` on a CUDA host                                              | **CPU TARGET**     | Runtime preserves the requested precision policy and forces `model.attn_backend='sdpa'` for explicit CPU runs.      |
| GLUE task with `model.attn_backend=flash_attn_varlen`                              | **AUTO-ADJUST**    | GLUE classifier wrappers force `model.attn_backend='sdpa'`; packed flash attention is not part of the supported GLUE path. |
| `optimizer.name=muonclip` with FSDP v1                                             | **ERROR**          | MuonClip distributed mode requires FSDP2 (`fsdp_version=2`).                                                       |
| Any DeepSpeed runtime                                                              | **ERROR**          | DeepSpeed execution is unsupported in this repo; use Accelerate FSDP v2 for distributed runs. Legacy DeepSpeed checkpoint conversion remains available separately. |
| `trainer.mixed_precision='no'` with `model.attn_backend=flash_attn_varlen`         | **AUTO-ADJUST**    | Runtime switches attention backend to `sdpa` with a warning.                                                       |
| `contrastive.pretraining_prob > 0` with `model.dropout_prob <= 0`                 | **ERROR**          | SimCSE-style anti-forgetting steps require dropout-created views.                                                   |
| Contrastive training without resume, pretrained checkpoint, or random opt-in      | **ERROR**          | Set `contrastive.pretrained_checkpoint_dir`, resume a self-contained contrastive checkpoint, or explicitly set `contrastive.allow_random_weights=true`. |
| `datacollator.pack_sequences=true` with `model.attn_backend=sdpa`                 | **WARNING**        | Works, but slower than `flash_attn_varlen`; SDPA uses fallback path.                                                |
| `dataset.path` and `dataset.name` both set                                        | **PRECEDENCE**     | Existing local `dataset.path` is used first; hub dataset acts as fallback.                                          |
| Tokenizer/model vocab sizes                                                       | **IMPORTANT**      | Runtime pads the tokenizer with inert tokens so tokenizer length matches model vocabulary size.                    |
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

## Related Docs

- [Training](../guides/training.md)
- [Training optimization](../guides/training-optimization.md)
- [Testing](../guides/testing.md)
- [Troubleshooting](../guides/troubleshooting.md)
