#!/usr/bin/env bash
# Example evaluation commands.
#
# Activate your Python environment of choice, then run from the repository root.

set -euo pipefail

# GLUE evaluation - single task example
python scripts/evaluation/run_glue.py configs/glue/cola.yaml

# GLUE evaluation - quick smoke test (small tasks, fail-fast)
python scripts/evaluation/glue/run_glue_suite.py configs/glue --suite quick

# GLUE evaluation - full suite (continues through task failures)
python scripts/evaluation/glue/run_glue_suite.py configs/glue --suite all

# GLUE config generation - from a sweep directory of pretrained runs
# CHECKPOINT_ROOT="outputs/my-sweep"
# WANDB_PROJECT="neobert/glue"
# scripts/evaluation/glue/build_configs.sh "${CHECKPOINT_ROOT}" "${WANDB_PROJECT}" \
#   --config-output-dir configs/glue/generated \
#   --results-root outputs/glue \
#   --tasks cola,qnli
#
# Then run:
# python scripts/evaluation/glue/run_glue_suite.py configs/glue/generated/<run>-ckpt<step> --suite all
