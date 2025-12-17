#!/usr/bin/env bash
# Example evaluation commands.
#
# Run from the repository root.

set -euo pipefail

# GLUE evaluation - single task example
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml

# GLUE evaluation - quick smoke test (small tasks only)
bash scripts/evaluation/glue/run_quick_glue.sh configs/glue

# GLUE evaluation - full suite
bash scripts/evaluation/glue/run_all_glue.sh configs/glue

# GLUE config generation - from a sweep directory of pretrained runs
# CHECKPOINT_ROOT="outputs/my-sweep"
# WANDB_PROJECT="neobert/glue"
# bash scripts/evaluation/glue/build_configs.sh "${CHECKPOINT_ROOT}" "${WANDB_PROJECT}" \
#   --config-output-dir configs/glue/generated \
#   --results-root outputs/glue \
#   --tasks cola,qnli
#
# Then run:
# bash scripts/evaluation/glue/run_all_glue.sh configs/glue/generated/<run>-ckpt<step>
