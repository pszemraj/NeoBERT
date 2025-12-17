#!/usr/bin/env bash

# Generate GLUE configs for every checkpoint directory in a sweep.
# Usage: bash scripts/evaluation/glue/build_configs.sh <checkpoint_root> <wandb_project> [options]
# Options:
#   --config-output-dir <path>   (default: configs/glue/generated)
#   --results-root <path>        (default: outputs/glue)
#   --tasks <space-or-comma-list> (restrict tasks to generate)
# Environment overrides still respected: CONFIG_OUTPUT_DIR, RESULTS_ROOT, PYTHON_EXEC

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_root> <wandb_project> [options]" >&2
  exit 1
fi

CHECKPOINT_ROOT=$1
WANDB_PROJECT=$2
shift 2

CONFIG_OUTPUT_DIR=${CONFIG_OUTPUT_DIR:-configs/glue/generated}
RESULTS_ROOT=${RESULTS_ROOT:-outputs/glue}
PYTHON_EXEC=${PYTHON_EXEC:-python}
TASK_LIST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-output-dir)
      shift
      [[ $# -gt 0 ]] || { echo "--config-output-dir requires a value" >&2; exit 1; }
      CONFIG_OUTPUT_DIR=$1
      ;;
    --results-root)
      shift
      [[ $# -gt 0 ]] || { echo "--results-root requires a value" >&2; exit 1; }
      RESULTS_ROOT=$1
      ;;
    --tasks)
      shift
      [[ $# -gt 0 ]] || { echo "--tasks requires a value" >&2; exit 1; }
      IFS=', ' read -r -a TASK_LIST <<< "$1"
      ;;
    *)
      echo "Unrecognized option: $1" >&2
      exit 1
      ;;
  esac
  shift || true
done

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint root '${CHECKPOINT_ROOT}' not found." >&2
  exit 1
fi

CHECKPOINT_ROOT=$(realpath "${CHECKPOINT_ROOT}")
CONFIG_OUTPUT_DIR=$(realpath -m "${CONFIG_OUTPUT_DIR}")
RESULTS_ROOT=$(realpath -m "${RESULTS_ROOT}")

mkdir -p "${CONFIG_OUTPUT_DIR}"

echo "Generating GLUE configs from checkpoints in ${CHECKPOINT_ROOT}" >&2
echo "Writing configs under ${CONFIG_OUTPUT_DIR}" >&2
echo "Fine-tune outputs will default to ${RESULTS_ROOT}" >&2

for run_dir in "${CHECKPOINT_ROOT}"/*; do
  [[ -d "${run_dir}" ]] || continue

  if [[ ! -d "${run_dir}/model_checkpoints" ]]; then
    echo "Skipping $(basename "${run_dir}") (no model_checkpoints directory)" >&2
    continue
  fi

  echo "---" >&2
  echo "Processing $(basename "${run_dir}")" >&2

  cmd=(
    "${PYTHON_EXEC}" "${SCRIPT_DIR}/build_glue_configs.py"
    --checkpoint-dir "${run_dir}"
    --wandb-project "${WANDB_PROJECT}"
    --results-root "${RESULTS_ROOT}"
    --output-dir "${CONFIG_OUTPUT_DIR}"
  )

  if [[ ${#TASK_LIST[@]} -gt 0 ]]; then
    cmd+=(--tasks)
    for task in "${TASK_LIST[@]}"; do
      cmd+=("${task}")
    done
  fi

  "${cmd[@]}"
done

echo "---" >&2
echo "Finished generating GLUE configs." >&2
