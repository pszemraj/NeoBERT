#!/usr/bin/env bash
# End-to-end Dion2 robustness experiment runner (W&B-enabled).
#
# Designed for 2x GPU machines (e.g., 2x RTX 5090).
# Runs optimizer variants for stability checks over a configurable step budget.
# Default behavior is a multi-run DDP matrix + resume checks.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
NUM_GPUS="${NUM_GPUS:-2}"
ACCELERATE_NUM_MACHINES="${ACCELERATE_NUM_MACHINES:-1}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
ACCELERATE_DYNAMO_BACKEND="${ACCELERATE_DYNAMO_BACKEND:-no}"

# Stage toggles
RUN_SINGLE="${RUN_SINGLE:-0}"   # 0/1: optional single-GPU runs
RUN_DDP="${RUN_DDP:-1}"         # 0/1: 2-GPU DDP runs
RUN_FSDP2="${RUN_FSDP2:-auto}"  # auto/0/1: 2-GPU FSDP2 runs
RUN_RESUME="${RUN_RESUME:-1}"   # 0/1: resume integrity checks

# FSDP2 launch config (required when RUN_FSDP2=1)
FSDP2_CONFIG="${FSDP2_CONFIG:-}"

# Step budgets
STABILITY_STEPS="${STABILITY_STEPS:-800}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-250}"
EVAL_STEPS="${EVAL_STEPS:-250}"
SEEDS="${SEEDS:-42 1337}"

# Resume budget
RESUME_PHASE1_STEPS="${RESUME_PHASE1_STEPS:-400}"
RESUME_PHASE2_STEPS="${RESUME_PHASE2_STEPS:-800}"

# Dion2+QK clipping knobs
QK_THRESHOLD="${QK_THRESHOLD:-50.0}"
QK_INTERVAL="${QK_INTERVAL:-10}"
QK_WARMUP="${QK_WARMUP:-0}"
QK_ALPHA="${QK_ALPHA:-0.5}"

# W&B configuration
WANDB_MODE="${WANDB_MODE:-online}"  # online/offline/disabled
WANDB_PROJECT="${WANDB_PROJECT:-neobert-dion2-robustness}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-[dion2,robustness,stability]}"
WANDB_GROUP="${WANDB_GROUP:-dion2-robustness-$(date +%Y%m%d-%H%M%S)}"

ROOT_OUT="${ROOT_OUT:-./outputs/exp/${WANDB_GROUP}}"

CFG_ADAMW="configs/pretraining/pretrain_neobert100m_smollm2data.yaml"
CFG_MUON="configs/pretraining/pretrain_neobert100m_smollm2data_muonclip.yaml"
CFG_DION2="configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}" >&2
    exit 1
  fi
}

require_cmd "${PYTHON_BIN}"
if [[ "${RUN_DDP}" == "1" || "${RUN_FSDP2}" == "1" ]]; then
  require_cmd "${ACCELERATE_BIN}"
fi

# Auto-resolve FSDP2 toggle.
if [[ "${RUN_FSDP2}" == "auto" ]]; then
  if [[ -n "${FSDP2_CONFIG}" && -f "${FSDP2_CONFIG}" ]]; then
    RUN_FSDP2="1"
    echo "INFO: FSDP2 enabled (auto): ${FSDP2_CONFIG}"
  else
    RUN_FSDP2="0"
    echo "INFO: FSDP2 disabled (auto): set FSDP2_CONFIG to enable."
  fi
fi

if [[ "${RUN_FSDP2}" == "1" ]]; then
  if [[ -z "${FSDP2_CONFIG}" ]]; then
    echo "ERROR: RUN_FSDP2=1 but FSDP2_CONFIG is not set." >&2
    echo "Set FSDP2_CONFIG to your Accelerate fsdp_version=2 config file." >&2
    echo "Example: FSDP2_CONFIG=configs/accelerate/fsdp2_2x5090.yaml" >&2
    exit 1
  fi
  if [[ ! -f "${FSDP2_CONFIG}" ]]; then
    echo "ERROR: FSDP2_CONFIG file not found: ${FSDP2_CONFIG}" >&2
    exit 1
  fi
fi

if [[ "${RUN_FSDP2}" != "0" && "${RUN_FSDP2}" != "1" ]]; then
  echo "ERROR: RUN_FSDP2 must be one of: auto, 0, 1 (got '${RUN_FSDP2}')." >&2
  exit 1
fi

mkdir -p "${ROOT_OUT}"

common_wandb_args=(
  --wandb.enabled true
  --wandb.mode "${WANDB_MODE}"
  --wandb.project "${WANDB_PROJECT}"
  --wandb.tags "${WANDB_TAGS}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  common_wandb_args+=(--wandb.entity "${WANDB_ENTITY}")
fi

common_train_args=(
  --trainer.logging_steps "${LOGGING_STEPS}"
  --trainer.save_steps "${SAVE_STEPS}"
  --trainer.eval_steps "${EVAL_STEPS}"
)

launch_pretrain() {
  local runtime="$1"      # single|ddp|fsdp2
  local variant="$2"      # adamw|muonclip|dion2|dion2_qk
  local seed="$3"
  local steps="$4"
  local cfg="$5"
  local out_dir="$6"
  shift 6

  local run_name="${WANDB_GROUP}-${runtime}-${variant}-seed${seed}-steps${steps}"

  local launch_prefix=()
  case "${runtime}" in
    single)
      launch_prefix=("${PYTHON_BIN}")
      ;;
    ddp)
      launch_prefix=(
        "${ACCELERATE_BIN}" launch
        --num_processes "${NUM_GPUS}"
        --num_machines "${ACCELERATE_NUM_MACHINES}"
        --mixed_precision "${ACCELERATE_MIXED_PRECISION}"
        --dynamo_backend "${ACCELERATE_DYNAMO_BACKEND}"
      )
      ;;
    fsdp2)
      launch_prefix=("${ACCELERATE_BIN}" launch --config_file "${FSDP2_CONFIG}")
      ;;
    *)
      echo "ERROR: unsupported runtime '${runtime}'" >&2
      exit 1
      ;;
  esac

  printf "\n=== RUN %s ===\n" "${run_name}"
  WANDB_RUN_GROUP="${WANDB_GROUP}" WANDB_JOB_TYPE="${runtime}" \
    "${launch_prefix[@]}" scripts/pretraining/pretrain.py \
      "${cfg}" \
      --seed "${seed}" \
      --trainer.max_steps "${steps}" \
      --trainer.output_dir "${out_dir}" \
      --wandb.name "${run_name}" \
      "${common_wandb_args[@]}" \
      "${common_train_args[@]}" \
      "$@"
}

run_matrix_for_runtime() {
  local runtime="$1"
  local seed="$2"

  local base_dir="${ROOT_OUT}/${runtime}/seed${seed}"
  mkdir -p "${base_dir}"

  launch_pretrain "${runtime}" adamw "${seed}" "${STABILITY_STEPS}" "${CFG_ADAMW}" \
    "${base_dir}/adamw"

  if [[ "${runtime}" != "fsdp2" ]]; then
    # MuonClip is incompatible with FSDP sharded parameters.
    launch_pretrain "${runtime}" muonclip "${seed}" "${STABILITY_STEPS}" "${CFG_MUON}" \
      "${base_dir}/muonclip"
  fi

  launch_pretrain "${runtime}" dion2 "${seed}" "${STABILITY_STEPS}" "${CFG_DION2}" \
    "${base_dir}/dion2" \
    --optimizer.dion2_config.enable_clipping false

  launch_pretrain "${runtime}" dion2_qk "${seed}" "${STABILITY_STEPS}" "${CFG_DION2}" \
    "${base_dir}/dion2_qk" \
    --optimizer.dion2_config.enable_clipping true \
    --optimizer.dion2_config.clipping_threshold "${QK_THRESHOLD}" \
    --optimizer.dion2_config.clipping_interval "${QK_INTERVAL}" \
    --optimizer.dion2_config.clipping_warmup_steps "${QK_WARMUP}" \
    --optimizer.dion2_config.clipping_alpha "${QK_ALPHA}"
}

run_resume_check() {
  local runtime="$1"
  local seed="$2"

  local base_dir="${ROOT_OUT}/${runtime}/seed${seed}/dion2_qk_resume"
  mkdir -p "${base_dir}"

  launch_pretrain "${runtime}" dion2_qk_resume_phase1 "${seed}" "${RESUME_PHASE1_STEPS}" "${CFG_DION2}" \
    "${base_dir}" \
    --optimizer.dion2_config.enable_clipping true \
    --optimizer.dion2_config.clipping_threshold "${QK_THRESHOLD}" \
    --optimizer.dion2_config.clipping_interval "${QK_INTERVAL}" \
    --optimizer.dion2_config.clipping_warmup_steps "${QK_WARMUP}" \
    --optimizer.dion2_config.clipping_alpha "${QK_ALPHA}"

  launch_pretrain "${runtime}" dion2_qk_resume_phase2 "${seed}" "${RESUME_PHASE2_STEPS}" "${CFG_DION2}" \
    "${base_dir}" \
    --trainer.resume_from_checkpoint latest \
    --optimizer.dion2_config.enable_clipping true \
    --optimizer.dion2_config.clipping_threshold "${QK_THRESHOLD}" \
    --optimizer.dion2_config.clipping_interval "${QK_INTERVAL}" \
    --optimizer.dion2_config.clipping_warmup_steps "${QK_WARMUP}" \
    --optimizer.dion2_config.clipping_alpha "${QK_ALPHA}"
}

print_plan() {
  echo "============================================================"
  echo "Dion2 Robustness W&B Experiment"
  echo "group:      ${WANDB_GROUP}"
  echo "project:    ${WANDB_PROJECT}"
  echo "entity:     ${WANDB_ENTITY:-<unset>}"
  echo "wandb_mode: ${WANDB_MODE}"
  echo "output:     ${ROOT_OUT}"
  echo "seeds:      ${SEEDS}"
  echo "steps:      ${STABILITY_STEPS}"
  echo "run_single: ${RUN_SINGLE}"
  echo "run_ddp:    ${RUN_DDP}"
  echo "run_fsdp2:  ${RUN_FSDP2}"
  echo "run_resume: ${RUN_RESUME}"
  if [[ "${RUN_FSDP2}" == "1" ]]; then
    echo "fsdp2_cfg:  ${FSDP2_CONFIG}"
  fi
  echo "============================================================"
}

print_plan

for seed in ${SEEDS}; do
  if [[ "${RUN_SINGLE}" == "1" ]]; then
    run_matrix_for_runtime single "${seed}"
  fi
  if [[ "${RUN_DDP}" == "1" ]]; then
    run_matrix_for_runtime ddp "${seed}"
  fi
  if [[ "${RUN_FSDP2}" == "1" ]]; then
    run_matrix_for_runtime fsdp2 "${seed}"
  fi

  if [[ "${RUN_RESUME}" == "1" ]]; then
    if [[ "${RUN_DDP}" == "1" ]]; then
      run_resume_check ddp "${seed}"
    fi
    if [[ "${RUN_FSDP2}" == "1" ]]; then
      run_resume_check fsdp2 "${seed}"
    fi
  fi
done

printf "\nAll scheduled runs finished.\n"
echo "W&B group: ${WANDB_GROUP}"
