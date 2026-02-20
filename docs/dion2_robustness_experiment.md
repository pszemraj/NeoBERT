# Dion2 Robustness Experiment Plan

This runbook validates that `optimizer.name: dion2` is correct, stable, and competitive versus existing optimizer options (`adamw`, `muonclip`).

## Scope

- Task scope: pretraining only.
- Optimizers under test:
  - `adamw` (baseline)
  - `muonclip` (baseline)
  - `dion2` (no QK clipping)
  - `dion2` + MuonClip QK clipping (`optimizer.dion2_config.enable_clipping: true`)
- Distributed scope:
  - single GPU (all optimizers)
  - 2-GPU DDP (all optimizers)
  - 2-GPU FSDP2 1D mesh (`adamw`, `dion2` variants only)

## Environment Prerequisites

```bash
# Base dev/test tools
pip install -e ".[dev]"

# Dion2 optional dependency
pip install -e ".[dion]"

# Optional packed attention backend
pip install -e ".[flash]" --no-build-isolation
```

Notes:

- Dion2 requires the upstream `dion` package (installed from GitHub via `.[dion]`).
- Dion2 with `use_triton: true` requires a Triton-compatible runtime.
- Dion2 is unsupported with DeepSpeed in this integration.
- MuonClip is incompatible with FSDP sharded parameters.

## Fixed Controls (for fair comparison)

Keep these constant across all optimizer runs:

- Model: `configs/pretraining/pretrain_neobert100m_smollm2data*.yaml` family.
- Dataset + tokenizer.
- Batch math: per-device batch, gradient accumulation, sequence length.
- Scheduler and LR (except optimizer-specific defaults required by optimizer).
- Precision (`bf16`) and checkpointing settings.
- Seed list for comparative runs: `[42, 1337, 3407]`.

## Experiment Matrix

| Stage | Purpose | Optimizers | Runtime | Steps | Seeds |
| --- | --- | --- | --- | --- | --- |
| A | smoke correctness | all 4 variants | 1 GPU | 200 | 1 |
| B | distributed startup/step | all 4 variants | 2-GPU DDP | 200 | 1 |
| C | FSDP2 mesh compatibility | `adamw`, `dion2`, `dion2+qk` | 2-GPU FSDP2 1D | 200 | 1 |
| D | convergence + stability | all 4 variants | 1 GPU | 10,000 | 3 |
| E | resume integrity | `dion2`, `dion2+qk` | 1 GPU and FSDP2 | 2,000 + resume to 4,000 | 1 |

## Commands

Use `jobs/experiment_dion2_robustness.sh` for an end-to-end W&B run matrix.

One-command W&B matrix run (2 GPUs, DDP by default):

```bash
WANDB_PROJECT=neobert-dion2-robustness \
WANDB_ENTITY=<your_wandb_entity> \
./jobs/experiment_dion2_robustness.sh
```

Include FSDP2 in same script run:

```bash
WANDB_PROJECT=neobert-dion2-robustness \
WANDB_ENTITY=<your_wandb_entity> \
FSDP2_CONFIG=<accelerate_fsdp2_config.yaml> \
RUN_FSDP2=1 \
./jobs/experiment_dion2_robustness.sh
```

Minimal smoke examples:

```bash
# AdamW baseline
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data.yaml \
  --trainer.max_steps 200 \
  --trainer.output_dir ./outputs/exp/dion2_robustness/adamw_smoke \
  --wandb.mode disabled \
  --seed 42

# MuonClip baseline
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml \
  --trainer.max_steps 200 \
  --trainer.output_dir ./outputs/exp/dion2_robustness/muonclip_smoke \
  --wandb.mode disabled \
  --seed 42

# Dion2 (no QK clipping)
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml \
  --trainer.max_steps 200 \
  --optimizer.dion2_config.enable_clipping false \
  --trainer.output_dir ./outputs/exp/dion2_robustness/dion2_smoke \
  --wandb.mode disabled \
  --seed 42

# Dion2 + MuonClip QK clipping
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml \
  --trainer.max_steps 200 \
  --optimizer.dion2_config.enable_clipping true \
  --optimizer.dion2_config.clipping_threshold 50.0 \
  --optimizer.dion2_config.clipping_interval 10 \
  --trainer.output_dir ./outputs/exp/dion2_robustness/dion2_qk_smoke \
  --wandb.mode disabled \
  --seed 42
```

Distributed examples:

```bash
# 2-GPU DDP (replace with your launcher defaults)
accelerate launch --num_processes 2 \
  scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml \
  --trainer.max_steps 200

# 2-GPU FSDP2 1D mesh (requires your fsdp2 accelerate config)
accelerate launch --config_file <accelerate_fsdp2_1d.yaml> \
  scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml \
  --trainer.max_steps 200
```

Resume integrity example:

```bash
# phase 1
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml \
  --trainer.max_steps 2000 \
  --trainer.save_steps 1000 \
  --trainer.output_dir ./outputs/exp/dion2_robustness/dion2_resume \
  --wandb.mode disabled

# phase 2 (resume)
python scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml \
  --trainer.max_steps 4000 \
  --trainer.resume_from_checkpoint latest \
  --trainer.output_dir ./outputs/exp/dion2_robustness/dion2_resume \
  --wandb.mode disabled
```

## Metrics to Track

- Correctness/stability:
  - no runtime errors
  - no NaN/Inf loss
  - no NaN/Inf grad norm
- Optimization quality:
  - train loss trajectory
  - eval loss (fixed eval budget)
  - masked-token accuracy
- Robustness:
  - successful checkpoint save/resume
  - distributed startup success rate (DDP/FSDP2)
- Performance:
  - tokens/sec
  - steps/sec
  - peak GPU memory
- QK clipping observability (`dion2+qk`):
  - `train/max_attention_logit`

## Pass/Fail Gates

- Stage A/B/C: 100% runs complete 200 steps without fatal errors.
- Stage D: all seeds complete 10k steps; no NaN/Inf events.
- Stage E: resume run restores optimizer + scheduler and continues cleanly.
- Relative quality at 10k steps (same seed average):
  - `dion2` and `dion2+qk` final eval loss should be no worse than `adamw` by >1%.
- Relative throughput (informational guardrail):
  - `dion2` >= 85% of `adamw` tokens/sec.
  - `dion2+qk` >= 70% of `adamw` tokens/sec.

## Reporting Template

For each variant, record:

- config path + overrides
- seed
- runtime mode (1 GPU / DDP / FSDP2)
- completed steps
- final train loss
- final eval loss
- average tokens/sec (steady-state window)
- peak GPU memory
- resume check status (if applicable)
- notes on failures/anomalies

A run is considered production-ready when all pass/fail gates are met and no unexplained instability remains.
