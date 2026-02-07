# Scripts Overview

CLI entry points for NeoBERT training, evaluation, and export.

## Directory Map

- `pretraining/` - MLM training and dataset tokenization helpers
- `contrastive/` - contrastive preprocessing + finetuning scripts
- `evaluation/` - GLUE/MTEB runners and analysis helpers
- `export-hf/` - checkpoint export and validation

See [docs/training.md](../docs/training.md), [docs/evaluation.md](../docs/evaluation.md), and [docs/export.md](../docs/export.md) for end-to-end workflows.

## Conventions

- Pretraining and contrastive scripts load config via `load_config_from_args`
  and support dot-notation overrides.
- Evaluation scripts expose their own argparse flags and do not support full
  dot-notation override surface.
- Export script takes a checkpoint path (not a config path).

## Common Commands

```bash
# Pretraining
python scripts/pretraining/pretrain.py configs/pretraining/pretrain_neobert.yaml

# GLUE
python scripts/evaluation/run_glue.py configs/glue/cola.yaml

# MTEB
python scripts/evaluation/run_mteb.py configs/pretraining/pretrain_neobert.yaml --model_name_or_path outputs/<run>

# HF export
python scripts/export-hf/export.py outputs/<run>/model_checkpoints/<step>
```

## Related Docs

- [Configuration](../docs/configuration.md)
- [Troubleshooting](../docs/troubleshooting.md)
- [Jobs launcher examples](../jobs/README.md)
