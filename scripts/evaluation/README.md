# Evaluation Scripts

Utilities for GLUE/MTEB evaluation and result analysis.

## Files

- `run_glue.py` - run GLUE task from config
- `run_mteb.py` - run MTEB benchmark against a pretraining run directory
- `pseudo_perplexity.py` - pseudo-perplexity helper for MLM checkpoints
- `avg_mteb.py` - aggregate MTEB results
- `wrappers.py` - shared script helpers
- `glue/` - GLUE automation scripts

## GLUE Helpers

[GLUE helpers](glue/README.md) cover config generation, suite launchers, and
result summaries.

## Related Docs

- [Evaluation](../../docs/guides/evaluation.md)
- [Troubleshooting](../../docs/guides/troubleshooting.md)
