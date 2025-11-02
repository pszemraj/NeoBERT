# GLUE Helpers

Utilities for generating configs, launching sweeps, and summarizing results for GLUE fine-tuning runs.

## Available scripts

- `build_configs.sh` - Iterate over a sweep directory and call `build_glue_configs.py` for each checkpoint bundle.
- `build_glue_configs.py` - Generate task-specific configs under `configs/glue/generated`.
- `run_quick_glue.sh` - Launch the smaller GLUE tasks for smoke-testing checkpoints.
- `run_all_glue.sh` - Execute the full GLUE suite and capture per-task logs.
- `summarize_glue.py` - Aggregate GLUE metrics across runs into a table.
- `validate_glue_config.py` - Sanity-check generated configs before launching jobs.

All scripts assume they are invoked from the repository root. They rely on the shared evaluation code in `scripts/evaluation/run_glue.py`.
