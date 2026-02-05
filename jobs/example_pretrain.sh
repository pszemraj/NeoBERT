#!/usr/bin/env bash
# Example pretraining commands.
#
# Run from the repository root.

set -euo pipefail

PYTHON=(conda run --name neobert python)

# ----------------------------
# 1) Small smoke test (safe)
# ----------------------------
"${PYTHON[@]}" scripts/pretraining/pretrain.py \
  tests/configs/pretraining/test_tiny_pretrain.yaml \
  --wandb.mode disabled

# ----------------------------
# 2) Real runs (opt-in)
# ----------------------------
# Set RUN_FULL=1 to actually launch a longer pretraining job.
if [[ "${RUN_FULL:-0}" == "1" ]]; then
  # Basic pretraining with config file
  "${PYTHON[@]}" scripts/pretraining/pretrain.py \
    configs/pretraining/pretrain_neobert100m_smollm2data.yaml

  # MuonClip variant
  "${PYTHON[@]}" scripts/pretraining/pretrain.py \
    configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml
fi
