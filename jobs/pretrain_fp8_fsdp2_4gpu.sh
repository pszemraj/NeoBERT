#!/usr/bin/env bash
# Launch NeoBERT pretraining with torchao FP8 + Accelerate FSDP2 on 4 GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/pretraining/pretrain_neobert_fp8_4gpu.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/fsdp2_fp8_4gpu.yaml}"
RUN_TAG="${RUN_TAG:-fp8-fsdp2-4gpu-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"

# Default to first 4 GPUs when the user does not pre-set visibility.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#CUDA_DEVICE_LIST[@]}" -lt 4 ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose at least 4 GPUs, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

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
    for package in ("accelerate", "torchao")
    if importlib.util.find_spec(package) is None
]
if missing:
    raise SystemExit(
        "Missing required Python package(s): "
        + ", ".join(missing)
        + ". Install them in your training environment before launch."
    )
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${GPU_COUNT}" -lt 4 ]]; then
    echo "Expected >=4 visible NVIDIA GPUs, found ${GPU_COUNT}." >&2
    exit 1
  fi
fi

echo "Launching FP8 FSDP2 pretraining on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  train config:      ${TRAIN_CONFIG}"
echo "  accelerate config: ${ACCELERATE_CONFIG}"
echo "  output dir:        ${OUTPUT_DIR}"
echo "  wandb mode:        ${WANDB_MODE}"

exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  scripts/pretraining/pretrain.py \
  "${TRAIN_CONFIG}" \
  --trainer.output_dir "${OUTPUT_DIR}" \
  --wandb.name "${RUN_TAG}" \
  --wandb.mode "${WANDB_MODE}" \
  "$@"
