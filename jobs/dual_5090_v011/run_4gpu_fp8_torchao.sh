#!/usr/bin/env bash
# Launch a single TorchAO FP8 pretraining run across 4 GPUs (DDP via torch.distributed.run).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT_DIR}/jobs/dual_5090_v011"
PRETRAIN_SCRIPT="${ROOT_DIR}/scripts/pretraining/pretrain.py"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/pretrain_fp8_4gpu.yaml}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29517}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/fp8_4gpu_v011}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${PRETRAIN_SCRIPT}" ]]; then
  echo "Missing pretrain entrypoint: ${PRETRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing FP8 4-GPU config: ${CONFIG_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
AUTO_NPROC="${#GPU_ARRAY[@]}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${AUTO_NPROC}}"

echo "Root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_PATH}"
echo "GPUs: ${GPU_LIST}"
echo "Processes per node: ${NPROC_PER_NODE}"
echo "Master port: ${MASTER_PORT}"
echo "Log: ${LOG_DIR}/fp8_4gpu.log"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${PRETRAIN_SCRIPT}" "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_DIR}/fp8_4gpu.log"
