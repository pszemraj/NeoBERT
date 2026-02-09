#!/usr/bin/env bash
# Launch two pretraining runs in parallel on a 2-GPU host:
# - GPU 0: BF16 baseline
# - GPU 1: FP8 (TorchAO float8 rowwise)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT_DIR}/jobs/dual_5090_v011"
PRETRAIN_SCRIPT="${ROOT_DIR}/scripts/pretraining/pretrain.py"

BF16_CONFIG="${RUN_DIR}/pretrain_bf16.yaml"
FP8_CONFIG="${RUN_DIR}/pretrain_fp8.yaml"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_BF16="${GPU_BF16:-0}"
GPU_FP8="${GPU_FP8:-1}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/dual_5090_v011}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${PRETRAIN_SCRIPT}" ]]; then
  echo "Missing pretrain entrypoint: ${PRETRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${BF16_CONFIG}" ]]; then
  echo "Missing BF16 config: ${BF16_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${FP8_CONFIG}" ]]; then
  echo "Missing FP8 config: ${FP8_CONFIG}" >&2
  exit 1
fi

echo "Root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Logs: ${LOG_DIR}"
echo "BF16 -> GPU ${GPU_BF16}, config ${BF16_CONFIG}"
echo "FP8  -> GPU ${GPU_FP8}, config ${FP8_CONFIG}"

cleanup() {
  local pids=("$@")
  for pid in "${pids[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

CUDA_VISIBLE_DEVICES="${GPU_BF16}" \
  "${PYTHON_BIN}" "${PRETRAIN_SCRIPT}" "${BF16_CONFIG}" \
  >"${LOG_DIR}/bf16.log" 2>&1 &
PID_BF16=$!

CUDA_VISIBLE_DEVICES="${GPU_FP8}" \
  "${PYTHON_BIN}" "${PRETRAIN_SCRIPT}" "${FP8_CONFIG}" \
  >"${LOG_DIR}/fp8.log" 2>&1 &
PID_FP8=$!

trap 'cleanup "${PID_BF16}" "${PID_FP8}"' INT TERM

echo "Started BF16 PID=${PID_BF16}"
echo "Started FP8  PID=${PID_FP8}"

set +e
wait "${PID_BF16}"
STATUS_BF16=$?
wait "${PID_FP8}"
STATUS_FP8=$?
set -e

echo "BF16 exit code: ${STATUS_BF16}"
echo "FP8  exit code: ${STATUS_FP8}"

if [[ ${STATUS_BF16} -ne 0 || ${STATUS_FP8} -ne 0 ]]; then
  echo "One or more runs failed. Check logs in ${LOG_DIR}" >&2
  exit 1
fi

echo "Both runs completed successfully."
