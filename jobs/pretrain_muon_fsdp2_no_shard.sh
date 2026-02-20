#!/usr/bin/env bash
# Launch NeoBERT pretraining with MuonClip + FSDP2 in non-sharded mode.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/fsdp2_no_shard_1gpu.yaml}"
RUN_TAG="${RUN_TAG:-muon-fsdp2-no-shard-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "Missing TRAIN_CONFIG: ${TRAIN_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Missing ACCELERATE_CONFIG: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util

missing = [
    package
    for package in ("accelerate",)
    if importlib.util.find_spec(package) is None
]
if missing:
    raise SystemExit(
        "Missing required Python package(s): "
        + ", ".join(missing)
        + ". Install them in your training environment before launch."
    )
PY

echo "Launching MuonClip + FSDP2 NO_SHARD pretraining"
echo "  train config:      ${TRAIN_CONFIG}"
echo "  accelerate config: ${ACCELERATE_CONFIG}"
echo "  output dir:        ${OUTPUT_DIR}"
echo "  wandb mode:        ${WANDB_MODE}"

exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  scripts/pretraining/pretrain.py \
  "${TRAIN_CONFIG}" \
  --optimizer.name muonclip \
  --trainer.output_dir "${OUTPUT_DIR}" \
  --wandb.name "${RUN_TAG}" \
  --wandb.mode "${WANDB_MODE}" \
  "$@"
