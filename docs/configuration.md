# Configuration Reference

---

- [Overview](#overview)
- [Important Behavior](#important-behavior)
- [Top-Level (`Config`)](#top-level-config)
- [Model (`model`)](#model-model)
- [Dataset (`dataset`)](#dataset-dataset)
- [Tokenizer (`tokenizer`)](#tokenizer-tokenizer)
- [Data Collator (`datacollator`)](#data-collator-datacollator)
- [Trainer (`trainer`)](#trainer-trainer)
- [Optimizer (`optimizer`)](#optimizer-optimizer)
  - [Muon config (`optimizer.muon_config`)](#muon-config-optimizermuon_config)
- [Scheduler (`scheduler`)](#scheduler-scheduler)
- [W\&B (`wandb`)](#wb-wandb)
- [GLUE (`glue`)](#glue-glue)
- [Contrastive (`contrastive`)](#contrastive-contrastive)
- [CLI Notes](#cli-notes)
- [Related Docs](#related-docs)

---

## Overview

NeoBERT configs are YAML files mapped to dataclasses in `src/neobert/config.py`.
CLI overrides use dot notation, for example:

```bash
python scripts/pretraining/pretrain.py configs/pretraining/pretrain_neobert.yaml \
  --trainer.max_steps 1000 \
  --optimizer.lr 2e-4
```

## Important Behavior

- Unknown config keys raise errors at load time.
- Legacy keys are normalized with warnings where supported.
- Pretraining rounds vocab size to a multiple of 128 at startup for efficiency.
- `torch.compile` is skipped automatically when unavailable or when DeepSpeed is
  active.
- Packed training works with `attn_backend: flash_attn_varlen` (recommended) and
  has a slower SDPA fallback path.

## Top-Level (`Config`)

```text
task: str = "pretraining"  # pretraining | glue | mteb | contrastive
accelerate_config_file: str | None = None
pretrained_checkpoint: str = "latest"
use_deepspeed: bool = True
mteb_task_type: str = "all"
mteb_batch_size: int = 32
mteb_pooling: str = "mean"  # NeoBERTForMTEB expects avg|cls (other values fall back to cls)
mteb_overwrite_results: bool = False
pretraining_metadata: dict[str, Any] = {}
seed: int = 0
debug: bool = False
config_path: str | None = None
```

## Model (`model`)

```text
model.name: str | None = None
model.hidden_size: int = 768
model.num_hidden_layers: int = 12
model.num_attention_heads: int = 12
model.intermediate_size: int = 3072
model.max_position_embeddings: int = 512
model.vocab_size: int = 30522
model.rope: bool = True
model.rms_norm: bool = True
model.hidden_act: str = "swiglu"  # swiglu | gelu
model.dropout_prob: float = 0.0
model.norm_eps: float = 1e-5
model.embedding_init_range: float = 0.02
model.decoder_init_range: float = 0.02
model.classifier_init_range: float = 0.02
model.attn_backend: str = "sdpa"  # sdpa | flash_attn_varlen
model.kernel_backend: str = "auto"  # auto | liger | torch
model.ngpt: bool = False
model.base_scale: float = 1/(960**0.5)
model.pad_token_id: int = 0
model.from_hub: bool = False
```

## Dataset (`dataset`)

```text
dataset.name: str = "refinedweb"
dataset.config: str | None = None
dataset.path: str = ""
dataset.num_workers: int = 16
dataset.pin_memory: bool = False
dataset.persistent_workers: bool = True
dataset.prefetch_factor: int | None = None
dataset.streaming: bool = True
dataset.cache_dir: str | None = None
dataset.trust_remote_code: bool = False
dataset.max_seq_length: int = 512
dataset.text_column: str | None = None
dataset.validation_split: float | None = None
dataset.train_split: str | None = None
dataset.eval_split: str | None = None
dataset.num_proc: int = 4
dataset.shuffle_buffer_size: int = 10000
dataset.pre_tokenize: bool = False
dataset.pre_tokenize_output: str | None = None

# contrastive-specific
dataset.load_all_from_disk: bool = False
dataset.force_redownload: bool = False
dataset.pretraining_prob: float = 0.3
dataset.min_length: int = 512
```

## Tokenizer (`tokenizer`)

```text
tokenizer.name: str = "bert-base-uncased"
tokenizer.path: str | None = None
tokenizer.max_length: int = 512
tokenizer.padding: str = "max_length"
tokenizer.truncation: bool = True
tokenizer.vocab_size: int | None = None
```

## Data Collator (`datacollator`)

```text
datacollator.mlm_probability: float = 0.15
datacollator.pad_to_multiple_of: int | None = None
datacollator.mask_all: bool = False
datacollator.pack_sequences: bool = False
datacollator.max_length: int | None = None
```

Notes:

- `mask_all: false` uses standard 80/10/10 MLM corruption.
- `mask_all: true` uses 100% `[MASK]` replacement for selected tokens.
- `pack_sequences: true` is fastest with `model.attn_backend: flash_attn_varlen`
  and flash-attn installed.

## Trainer (`trainer`)

```text
trainer.per_device_train_batch_size: int = 16
trainer.per_device_eval_batch_size: int = 32
trainer.gradient_accumulation_steps: int = 1
trainer.max_steps: int = 1000000
trainer.save_steps: int = 10000
trainer.eval_steps: int = 10000
trainer.eval_max_batches: int | None = None
trainer.logging_steps: int = 100
trainer.enforce_full_packed_batches: bool = True
trainer.log_train_accuracy: bool = False
trainer.log_grad_norm: bool = False
trainer.output_dir: str = "./output"
trainer.overwrite_output_dir: bool = True
trainer.gradient_checkpointing: bool = False
trainer.gradient_clipping: float | None = None
trainer.mixed_precision: str = "bf16"  # no | fp16 | bf16
trainer.torch_compile: bool = False
trainer.torch_compile_dynamic: bool | None = None
trainer.torch_compile_backend: str = "inductor"
trainer.resume_from_checkpoint: str | None = None

# schedule/control
trainer.num_train_epochs: int = 3
trainer.eval_strategy: str = "steps"
trainer.save_strategy: str = "steps"
trainer.save_total_limit: int | None = 3
trainer.early_stopping: int = 0
trainer.metric_for_best_model: str | None = None
trainer.greater_is_better: bool = True
trainer.load_best_model_at_end: bool = False
trainer.save_model: bool = True

# legacy/compat
trainer.disable_tqdm: bool = False
trainer.dataloader_num_workers: int = 0
trainer.use_cpu: bool = False
trainer.report_to: list[str] = []
trainer.tf32: bool = True
trainer.max_ckpt: int = 3
trainer.log_weight_norms: bool = False
trainer.train_batch_size: int | None = None
trainer.eval_batch_size: int | None = None
```

Notes:

- `enforce_full_packed_batches` is useful for stable packed throughput; measure
  `tokens/sec` instead of `steps/sec` when comparing settings.
- `log_train_accuracy` and `log_grad_norm` are optional and can add overhead.

## Optimizer (`optimizer`)

```text
optimizer.name: str = "adamw"  # adamw | adam | muonclip
optimizer.lr: float = 1e-4
optimizer.weight_decay: float = 0.01
optimizer.betas: list[float] = [0.9, 0.999]
optimizer.eps: float = 1e-8
optimizer.muon_config: MuonConfig | None = None
```

### Muon config (`optimizer.muon_config`)

```text
muon_config.muon_beta: float = 0.95
muon_config.muon_decay: float = 0.0
muon_config.ns_steps: int = 5
muon_config.enable_clipping: bool = True
muon_config.clipping_threshold: float = 50.0
muon_config.clipping_alpha: float = 0.5
muon_config.clipping_warmup_steps: int = 0
muon_config.clipping_interval: int = 10
muon_config.clipping_qk_chunk_size: int = 1024
muon_config.capture_last_microbatch_only: bool = True
muon_config.detect_anomalies: bool = False
muon_config.orthogonalization: str = "polar_express"
muon_config.algorithm: str | None = None  # legacy alias
muon_config.polar_express: bool | None = None  # legacy alias
muon_config.clipping_layers_mapping: dict[str, str] = {}
```

## Scheduler (`scheduler`)

```text
scheduler.name: str = "cosine"  # cosine | linear
scheduler.warmup_steps: int = 10000
scheduler.total_steps: int | None = None
scheduler.num_cycles: float = 0.5
scheduler.decay_steps: int | None = None
scheduler.final_lr_ratio: float = 0.1
scheduler.warmup_percent: float | None = None
scheduler.decay_percent: float | None = None
```

Precedence:

- `warmup_percent` overrides `warmup_steps`
- `decay_percent` overrides `decay_steps`

## W&B (`wandb`)

```text
wandb.enabled: bool = False
wandb.project: str = "neo-bert"
wandb.entity: str | None = None
wandb.name: str | None = None
wandb.tags: list[str] = []
wandb.mode: str = "online"  # online | offline | disabled
wandb.log_interval: int = 100
wandb.resume: str = "never"
wandb.dir: str = "logs/wandb"
```

Compatibility behavior:

- if a config includes `wandb:` but omits `wandb.enabled`, logging is auto-enabled.

## GLUE (`glue`)

```text
glue.task_name: str = "cola"
glue.num_labels: int = 2
glue.max_seq_length: int = 128
glue.pretrained_model_path: str | None = None
glue.pretrained_checkpoint_dir: str | None = None
glue.pretrained_checkpoint: str | int | None = None
glue.allow_random_weights: bool = False
glue.classifier_dropout: float = 0.1
glue.classifier_init_range: float = 0.02
glue.transfer_from_task: bool = False
glue.num_workers: int = 4
glue.preprocessing_num_proc: int = 4
```

## Contrastive (`contrastive`)

```text
contrastive.temperature: float = 0.05
contrastive.pooling: str = "avg"  # avg | cls | max
contrastive.loss_type: str = "simcse"  # simcse | supcon
contrastive.hard_negative_weight: float = 0.0
```

## CLI Notes

- Pretraining and contrastive entry points support dot-notation overrides through
  `load_config_from_args`.
- Evaluation scripts have their own CLI and only expose selected arguments.

## Related Docs

- [Training](training.md)
- [Evaluation](evaluation.md)
- [Troubleshooting](troubleshooting.md)
