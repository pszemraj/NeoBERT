#!/usr/bin/env bash
# Launch four pretraining runs in parallel on a 4-GPU host:
# - GPU 0: BF16 baseline
# - GPU 1: FP8 (TorchAO float8 rowwise)
# - GPU 2: MXFP8 (TorchAO mxfp8_emulated)
# - GPU 3: NVFP4 (TorchAO nvfp4_qat)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT_DIR}/jobs/dual_5090_v011"
PRETRAIN_SCRIPT="${ROOT_DIR}/scripts/pretraining/pretrain.py"

BF16_CONFIG="${RUN_DIR}/pretrain_bf16.yaml"
FP8_CONFIG="${RUN_DIR}/pretrain_fp8.yaml"
MXFP8_CONFIG="${RUN_DIR}/pretrain_mxfp8.yaml"
NVFP4_CONFIG="${RUN_DIR}/pretrain_nvfp4.yaml"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_BF16="${GPU_BF16:-0}"
GPU_FP8="${GPU_FP8:-1}"
GPU_MXFP8="${GPU_MXFP8:-2}"
GPU_NVFP4="${GPU_NVFP4:-3}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/quad_5090_v011}"

mkdir -p "${LOG_DIR}"

for required in \
  "${PRETRAIN_SCRIPT}" \
  "${BF16_CONFIG}" \
  "${FP8_CONFIG}" \
  "${MXFP8_CONFIG}" \
  "${NVFP4_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

echo "Root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Logs: ${LOG_DIR}"
echo "BF16  -> GPU ${GPU_BF16},  config ${BF16_CONFIG}"
echo "FP8   -> GPU ${GPU_FP8},   config ${FP8_CONFIG}"
echo "MXFP8 -> GPU ${GPU_MXFP8}, config ${MXFP8_CONFIG}"
echo "NVFP4 -> GPU ${GPU_NVFP4}, config ${NVFP4_CONFIG}"

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

CUDA_VISIBLE_DEVICES="${GPU_MXFP8}" \
  "${PYTHON_BIN}" "${PRETRAIN_SCRIPT}" "${MXFP8_CONFIG}" \
  >"${LOG_DIR}/mxfp8.log" 2>&1 &
PID_MXFP8=$!

CUDA_VISIBLE_DEVICES="${GPU_NVFP4}" \
  "${PYTHON_BIN}" "${PRETRAIN_SCRIPT}" "${NVFP4_CONFIG}" \
  >"${LOG_DIR}/nvfp4.log" 2>&1 &
PID_NVFP4=$!

trap 'cleanup "${PID_BF16}" "${PID_FP8}" "${PID_MXFP8}" "${PID_NVFP4}"' INT TERM

echo "Started BF16  PID=${PID_BF16}"
echo "Started FP8   PID=${PID_FP8}"
echo "Started MXFP8 PID=${PID_MXFP8}"
echo "Started NVFP4 PID=${PID_NVFP4}"

set +e
wait "${PID_BF16}"
STATUS_BF16=$?
wait "${PID_FP8}"
STATUS_FP8=$?
wait "${PID_MXFP8}"
STATUS_MXFP8=$?
wait "${PID_NVFP4}"
STATUS_NVFP4=$?
set -e

echo "BF16  exit code: ${STATUS_BF16}"
echo "FP8   exit code: ${STATUS_FP8}"
echo "MXFP8 exit code: ${STATUS_MXFP8}"
echo "NVFP4 exit code: ${STATUS_NVFP4}"

if [[ ${STATUS_BF16} -ne 0 || ${STATUS_FP8} -ne 0 || ${STATUS_MXFP8} -ne 0 || ${STATUS_NVFP4} -ne 0 ]]; then
  echo "One or more runs failed. Check logs in ${LOG_DIR}" >&2
  exit 1
fi

echo "All four runs completed successfully."
