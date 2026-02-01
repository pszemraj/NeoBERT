# Configuration Reference (Axolotl-style)

This page is intended to be a **single-stop config reference**. Every field is shown as:

```
field: type = default
```

Use dot-notation overrides on the CLI (e.g., `--trainer.max_steps 1000`).

> [!TIP]
> Example configs live in [configs/README.md](../configs/README.md). Training/eval entry points are listed in [scripts/README.md](../scripts/README.md).

## Quick Start

```bash
# Pretraining
python scripts/pretraining/pretrain.py --config configs/pretraining/pretrain_neobert.yaml

# GLUE
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml
```

## Task Requirements and Gotchas

- **Pretraining uses top-level `mixed_precision`**, while GLUE/contrastive use `trainer.mixed_precision`. To keep behavior consistent, set both.
- **GLUE requires pretrained weights** unless `allow_random_weights: true` is set (either in `glue` or the raw `model` block). Provide `pretrained_checkpoint_dir`, `pretrained_checkpoint`, and `pretrained_config_path`.
- **GLUE from Hub**: set `model.from_hub: true` and `model.name: <hf_id>` to load a hosted model instead of local checkpoints.
- **MTEB pooling**: set `mteb_pooling` to `"avg"` or `"cls"`; any other value falls back to CLS pooling.
- **Warmup/decay precedence**:
  - `scheduler.warmup_percent` overrides `scheduler.warmup_steps`.
  - `scheduler.decay_percent` overrides `scheduler.decay_steps`.
- **Model shape sanity**: `model.hidden_size` must be divisible by `model.num_attention_heads`.
- **Dropout bounds**: `model.dropout_prob` must be between 0 and 1.
- **Vocab rounding**: during config preprocessing, vocab size is rounded to a multiple of 128 for GPU efficiency unless `trainer.use_cpu: true`.
- **Unknown keys are ignored** by the config loader, except that the **GLUE trainer consumes extra keys from the raw `model` block** (see below).

## Field Reference

### Top-level (Config)

```
task: Literal["pretraining", "glue", "mteb", "contrastive"] = "pretraining"
mixed_precision: Literal["no", "bf16", "fp32"] = "bf16"  # used by pretraining
accelerate_config_file: str | None = None  # reserved; currently unused
pretrained_checkpoint: str | int = "latest"  # used by MTEB
use_deepspeed: bool = True  # used by MTEB
mteb_task_type: Literal["all", "classification", "clustering", "pair_classification", "reranking", "retrieval", "sts"] = "all"
mteb_batch_size: int = 32
mteb_pooling: str = "mean"  # NeoBERTForMTEB expects "avg" or "cls"
mteb_overwrite_results: bool = False
pretraining_metadata: dict[str, Any] = {}
seed: int = 0
debug: bool = False
config_path: str | None = None  # internal; set by CLI loader
```

### Model (`model`)

```
model.hidden_size: int = 768
model.num_hidden_layers: int = 12
model.num_attention_heads: int = 12
model.intermediate_size: int = 3072
model.max_position_embeddings: int = 512
model.vocab_size: int = 30522
model.rope: bool = True
model.rms_norm: bool = True
model.hidden_act: Literal["swiglu", "gelu"] = "swiglu"
model.dropout_prob: float = 0.0
model.norm_eps: float = 1e-5
model.embedding_init_range: float = 0.02
model.decoder_init_range: float = 0.02
model.classifier_init_range: float = 0.02
model.flash_attention: bool = True  # uses xformers during training
model.ngpt: bool = False
model.base_scale: float = 1 / (960 ** 0.5)
model.pad_token_id: int = 0
```

### Dataset (`dataset`)

```
dataset.name: str = "refinedweb"
dataset.path: str = ""
dataset.num_workers: int = 16

dataset.streaming: bool = True
dataset.cache_dir: str | None = None

dataset.max_seq_length: int = 512
dataset.validation_split: float | None = None
dataset.train_split: str | None = None
dataset.eval_split: str | None = None

dataset.num_proc: int = 4
dataset.shuffle_buffer_size: int = 10000

dataset.pre_tokenize: bool = False
dataset.pre_tokenize_output: str | None = None

# Contrastive-specific

dataset.load_all_from_disk: bool = False
dataset.force_redownload: bool = False
dataset.pretraining_prob: float = 0.3
dataset.min_length: int = 512
```

Note: when `dataset.pre_tokenize: true`, the pretokenized dataset is written to `dataset.pre_tokenize_output` if provided, otherwise to `dataset.path`.

### Tokenizer (`tokenizer`)

```
tokenizer.name: str = "bert-base-uncased"
tokenizer.path: str | None = None
tokenizer.max_length: int = 512
tokenizer.padding: str = "max_length"
tokenizer.truncation: bool = True
tokenizer.vocab_size: int | None = None
```

### Data collator (`datacollator`)

```
datacollator.mlm_probability: float = 0.15
datacollator.pad_to_multiple_of: int | None = None
```

### Trainer (`trainer`)

```
trainer.output_dir: str = "./output"
trainer.overwrite_output_dir: bool = True  # currently unused

trainer.per_device_train_batch_size: int = 16
trainer.per_device_eval_batch_size: int = 32
trainer.gradient_accumulation_steps: int = 1
trainer.max_steps: int = 1000000
trainer.num_train_epochs: int = 3

trainer.eval_strategy: Literal["steps", "epoch"] = "steps"
trainer.eval_steps: int = 10000
trainer.save_strategy: Literal["steps", "epoch", "best", "no"] = "steps"
trainer.save_steps: int = 10000
trainer.save_total_limit: int | None = 3
trainer.max_ckpt: int = 3  # legacy checkpoint limit (fallback if save_total_limit is unset)
trainer.save_model: bool = True

trainer.logging_steps: int = 100
trainer.disable_tqdm: bool = False
trainer.dataloader_num_workers: int = 0
trainer.use_cpu: bool = False  # disables vocab rounding

trainer.gradient_checkpointing: bool = False
trainer.gradient_clipping: float | None = None
trainer.mixed_precision: Literal["no", "bf16", "fp32"] = "bf16"
trainer.bf16: bool = True  # legacy; not used by trainers
trainer.tf32: bool = True

trainer.seed: int = 42
trainer.resume_from_checkpoint: str | None = None

trainer.early_stopping: int = 0
trainer.metric_for_best_model: str | None = None
trainer.greater_is_better: bool = True
trainer.load_best_model_at_end: bool = False

# Legacy fallbacks (GLUE only)
trainer.train_batch_size: int | None = None
trainer.eval_batch_size: int | None = None

# Unused legacy field
trainer.report_to: list[str] = []
```

### Optimizer (`optimizer`)

```
optimizer.name: Literal["adamw", "adam", "muonclip"] = "adamw"
optimizer.lr: float = 1e-4
optimizer.weight_decay: float = 0.01
optimizer.betas: list[float] = [0.9, 0.999]
optimizer.eps: float = 1e-8
optimizer.muon_config: MuonConfig | None = None
```

Note: `optimizer.muon_config` is only used when `optimizer.name` is `muonclip`; otherwise it is ignored with a warning.

### MuonConfig (`optimizer.muon_config`)

```
muon_config.muon_beta: float = 0.95
muon_config.muon_decay: float = 0.0
muon_config.ns_steps: int = 5
muon_config.enable_clipping: bool = True
muon_config.clipping_threshold: float = 50.0
muon_config.clipping_alpha: float = 0.5
muon_config.clipping_warmup_steps: int = 0
muon_config.detect_anomalies: bool = False
muon_config.orthogonalization: str = "polar_express"
muon_config.algorithm: str | None = None  # alias; optional
muon_config.polar_express: bool | None = None  # legacy toggle
muon_config.clipping_layers_mapping: dict[str, str] = {}
```

### Scheduler (`scheduler`)

```
scheduler.name: Literal["cosine", "linear"] = "cosine"
scheduler.warmup_steps: int = 10000
scheduler.decay_steps: int = 50000
scheduler.total_steps: int | None = None  # currently unused
scheduler.num_cycles: float = 0.5  # currently unused
scheduler.warmup_percent: float | None = None
scheduler.decay_percent: float | None = None
```

### Weights and Biases (`wandb`)

```
wandb.project: str = "neo-bert"
wandb.entity: str | None = None
wandb.name: str | None = None
wandb.tags: list[str] = []
wandb.mode: Literal["online", "offline", "disabled"] = "online"
wandb.log_interval: int = 100
wandb.resume: str = "never"
wandb.dir: str = "logs/wandb"
```

### GLUE (`glue`)

```
glue.task_name: str = "cola"
glue.num_labels: int = 2
glue.max_seq_length: int = 128

glue.pretrained_model_path: str | None = None  # unused

glue.pretrained_checkpoint_dir: str | None = None
glue.pretrained_checkpoint: str | int | None = None  # step number or "latest"

glue.allow_random_weights: bool = False

glue.classifier_dropout: float = 0.1
glue.classifier_init_range: float = 0.02

glue.transfer_from_task: bool = False

glue.num_workers: int = 4
glue.preprocessing_num_proc: int = 4
```

Note: the GLUE trainer currently reads **classifier settings and transfer flags from the raw `model` block**, not from `glue.*`. If you need these to take effect, set them under `model:` in your GLUE config.

### Contrastive (`contrastive`)

```
contrastive.temperature: float = 0.05
contrastive.pooling: Literal["avg", "cls", "max"] = "avg"
contrastive.loss_type: Literal["simcse", "supcon"] = "simcse"
contrastive.hard_negative_weight: float = 0.0
```

## GLUE Raw-Model Extras (not in dataclasses)

The GLUE trainer inspects the raw `model:` mapping before it is converted to dataclasses. These keys are supported there even though they are not in `ModelConfig`:

```
model.pretrained_checkpoint_dir: str
model.pretrained_checkpoint: str | int
model.pretrained_config_path: str
model.allow_random_weights: bool
model.from_hub: bool
model.name: str  # HF model id when from_hub is true

# GLUE classifier settings (preferred location today)
model.classifier_dropout: float
model.classifier_init_range: float
model.transfer_from_task: bool
```

## Mutual Exclusivity and Precedence

- `scheduler.warmup_percent` overrides `scheduler.warmup_steps`.
- `scheduler.decay_percent` overrides `scheduler.decay_steps`.
- `trainer.eval_strategy: "epoch"` ignores `trainer.eval_steps`.
- `trainer.save_strategy: "epoch"` ignores `trainer.save_steps`.

## Example: Minimal Pretraining Config

```yaml
task: pretraining
mixed_precision: bf16

dataset:
  name: wikipedia
  max_seq_length: 512

tokenizer:
  name: bert-base-uncased
  max_length: 512

model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 4096
  vocab_size: 30522

trainer:
  output_dir: ./outputs/neobert_pretrain
  per_device_train_batch_size: 32
  max_steps: 1000000

optimizer:
  name: adamw
  lr: 1e-4

scheduler:
  name: cosine
  warmup_steps: 10000
```

## Next Steps

- Training workflows: [docs/training.md](training.md)
- Evaluation recipes: [docs/evaluation.md](evaluation.md)
- Export guide: [docs/export.md](export.md)
