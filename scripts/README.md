# Scripts Overview

Entry points for NeoBERT training, evaluation, contrastive learning, and export workflows live here. Think of this directory as the launch pad-step-by-step instructions now live in the documentation set.

> [!TIP]
> Need detailed walkthroughs? The links below point to the topic-specific guides in `docs/`.

## Directory Map

- `pretraining/` - Pretraining pipeline, data preprocessing, and long-context continuation. See [docs/training.md](../docs/training.md).
- `evaluation/` - GLUE, MTEB, and analysis utilities. See [docs/evaluation.md](../docs/evaluation.md) plus [scripts/evaluation/README.md](evaluation/README.md).
- `contrastive/` - Contrastive fine-tuning datasets and trainers. See the contrastive section in [docs/training.md](../docs/training.md).
- `export-hf/` - Convert checkpoints to Hugging Face format and validate exports. See [docs/export.md](../docs/export.md) and [scripts/export-hf/README.md](export-hf/README.md).

## Shared Conventions

- Training/evaluation entry points take a positional config path plus dot-notation overrides. Export utilities typically take a checkpoint path instead. The config hierarchy is documented in [docs/configuration.md](../docs/configuration.md).
- `--debug` enables verbose logging across scripts.
- Environment variables are the preferred way to pass secrets (e.g., `WANDB_API_KEY`, `HF_TOKEN`). See [docs/training.md](../docs/training.md) for examples.
- Helper shell wrappers live under `jobs/` for batch systems, and tiny smoke-test configs live in `tests/configs/`.

## Related References

- High-level concepts and setup: [README.md](../README.md), [docs/README.md](../docs/README.md)
- Training workflows and best practices: [docs/training.md](../docs/training.md)
- Evaluation recipes and metrics: [docs/evaluation.md](../docs/evaluation.md)
- Export pipeline: [docs/export.md](../docs/export.md)
- Troubleshooting and debugging tips: [docs/troubleshooting.md](../docs/troubleshooting.md)
