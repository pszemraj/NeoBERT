# GLUE Evaluation Helpers

Automation helpers for generating, running, and summarizing GLUE configs.

## Scripts

- `build_configs.sh` - iterate sweep outputs and generate GLUE configs
- `build_glue_configs.py` - Python config generator
- `run_glue_suite.py` - parameterized quick/full suite launcher
- `summarize_glue.py` - aggregate metrics table
- `validate_glue_config.py` - config sanity checks

## Usage Notes

- Invoke from repository root.
- Generated configs default to `configs/glue/generated/` unless `build_configs.sh` overrides `--config-output-dir`.
- `run_glue_suite.py --suite quick` fails on the first task error; `--suite all` runs every task before returning a failure status.
- GLUE workflow details are in [docs/guides/evaluation.md](../../../docs/guides/evaluation.md).
