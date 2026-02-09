#!/usr/bin/env bash
# Launch a single Quartet-II pretraining run across 4 GPUs with Accelerate FSDP2.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT_DIR}/jobs/dual_5090_v011"
PRETRAIN_SCRIPT="${ROOT_DIR}/scripts/pretraining/pretrain.py"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/pretrain_quartet2.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${RUN_DIR}/accelerate_fsdp2_4gpu.yaml}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29527}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/quartet2_4gpu_v011}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${PRETRAIN_SCRIPT}" ]]; then
  echo "Missing pretrain entrypoint: ${PRETRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing Quartet-II config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Missing Accelerate FSDP2 config: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi
if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
  echo "Missing accelerate binary: ${ACCELERATE_BIN}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
AUTO_NPROC="${#GPU_ARRAY[@]}"
NUM_PROCESSES="${NUM_PROCESSES:-${AUTO_NPROC}}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"

echo "Root: ${ROOT_DIR}"
echo "Accelerate: ${ACCELERATE_BIN}"
echo "Config: ${CONFIG_PATH}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "GPUs: ${GPU_LIST}"
echo "Processes: ${NUM_PROCESSES}"
echo "Machines: ${NUM_MACHINES} (rank ${MACHINE_RANK})"
echo "Main process port: ${MAIN_PROCESS_PORT}"
echo "Log: ${LOG_DIR}/quartet2_4gpu.log"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  "${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${PRETRAIN_SCRIPT}" "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_DIR}/quartet2_4gpu.log"
