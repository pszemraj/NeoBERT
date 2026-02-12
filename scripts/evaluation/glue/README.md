# GLUE Evaluation Helpers

Automation helpers for generating, running, and summarizing GLUE configs.
Behavior semantics for GLUE evaluation are documented in
[docs/evaluation.md](../../../docs/evaluation.md).

## Scripts

- `build_configs.sh` - iterate sweep outputs and generate GLUE configs
- `build_glue_configs.py` - Python config generator
- `run_quick_glue.sh` - smoke subset launcher
- `run_all_glue.sh` - full GLUE launcher
- `summarize_glue.py` - aggregate metrics table
- `validate_glue_config.py` - config sanity checks

## Usage Notes

- Invoke from repository root.
- Generated configs typically land under `configs/glue/generated/`.
- Execution uses `scripts/evaluation/run_glue.py`.
- Keep GLUE behavior semantics in [docs/evaluation.md](../../../docs/evaluation.md).

## Examples

```bash
bash scripts/evaluation/glue/run_quick_glue.sh configs/glue
bash scripts/evaluation/glue/run_all_glue.sh configs/glue
python scripts/evaluation/glue/summarize_glue.py outputs/glue/<run>
```
