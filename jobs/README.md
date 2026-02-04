# Job Scripts (`jobs/`)

Lightweight, copy-pasteable shell snippets for launching common NeoBERT workloads.

This folder is intentionally small: the "real" entry points live under `scripts/`
and are driven by YAML configs in `configs/`.

**Contents**
```
jobs/
  README.md
  example_pretrain.sh
  example_evaluate.sh
```

## Environment

Repo convention is to run Python via the `neobert` conda env.

```bash
# Option A: prefix commands (works even if your shell env isn't activated)
conda run --name neobert python -c "import torch; print(torch.__version__)"

# Option B: activate once in your shell, then run python normally
conda activate neobert
python -c "import torch; print(torch.__version__)"
```

## Pretraining

Primary entry point:
- `scripts/pretraining/pretrain.py` (loads a YAML config via `--config`)

Examples:
```bash
# Small smoke test (CPU-friendly)
conda run --name neobert python scripts/pretraining/pretrain.py \
  --config tests/configs/pretraining/test_tiny_pretrain.yaml \
  --wandb.mode disabled

# Real streaming pretraining config
conda run --name neobert python scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert100m_smollm2data.yaml

# MuonClip variant
conda run --name neobert python scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml
```

### Multi-GPU (Accelerate)

Pretraining uses `accelerate`. For multi-GPU:
```bash
accelerate launch scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert100m_smollm2data.yaml
```

If you have an accelerate config file:
```bash
accelerate launch --config_file path/to/accelerate.yaml \
  scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert100m_smollm2data.yaml
```

### Resume

Pretraining checkpoints are stored under:
- `${trainer.output_dir}/checkpoints/` (accelerate state)
- `${trainer.output_dir}/model_checkpoints/` (model weights)

`trainer.resume_from_checkpoint` can be:
- `true` / `latest` / `auto` (resume from latest under `output_dir/checkpoints/`)
- a path to a specific checkpoint directory

```bash
conda run --name neobert python scripts/pretraining/pretrain.py \
  --config configs/pretraining/pretrain_neobert100m_smollm2data.yaml \
  --trainer.resume_from_checkpoint latest
```

## Evaluation (GLUE / MTEB)

Primary entry points:
- `scripts/evaluation/run_glue.py <config.yaml>` (positional config path)
- `scripts/evaluation/run_mteb.py <config.yaml>`
- `scripts/evaluation/glue/run_quick_glue.sh [configs/glue]`
- `scripts/evaluation/glue/run_all_glue.sh [configs/glue]`

Examples:
```bash
# Single GLUE task
conda run --name neobert python scripts/evaluation/run_glue.py configs/glue/cola.yaml

# Quick GLUE subset (fast sanity check)
bash scripts/evaluation/glue/run_quick_glue.sh configs/glue

# Full GLUE suite
bash scripts/evaluation/glue/run_all_glue.sh configs/glue
```

## Preprocessing

Common dataset tokenization entry point:
- `scripts/pretraining/preprocess.py` (tokenize + `save_to_disk`)

```bash
conda run --name neobert python scripts/pretraining/preprocess.py \
  --config configs/pretraining/pretrain_neobert.yaml
```

## Using The Example Scripts

```bash
chmod +x jobs/example_pretrain.sh jobs/example_evaluate.sh
./jobs/example_pretrain.sh
./jobs/example_evaluate.sh
```

`jobs/example_pretrain.sh` runs a tiny smoke test by default. To launch longer
pretraining configs, run it with `RUN_FULL=1`.

```bash
RUN_FULL=1 ./jobs/example_pretrain.sh
```

The scripts are examples; copy them and edit for your setup:
```bash
cp jobs/example_pretrain.sh jobs/my_pretrain.sh
chmod +x jobs/my_pretrain.sh
./jobs/my_pretrain.sh
```
