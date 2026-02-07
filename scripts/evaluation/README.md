# Evaluation Scripts

Utilities for GLUE/MTEB evaluation and result analysis.

## Files

- `run_glue.py` - run GLUE task from config
- `run_mteb.py` - run MTEB benchmark against a pretraining run directory
- `pseudo_perplexity.py` - pseudo-perplexity helper for MLM checkpoints
- `avg_mteb.py` - aggregate MTEB results
- `wrappers.py` - shared script helpers
- `glue/` - GLUE automation scripts

## Common Commands

```bash
python scripts/evaluation/run_glue.py configs/glue/cola.yaml
python scripts/evaluation/run_mteb.py configs/pretraining/pretrain_neobert.yaml --model_name_or_path outputs/<run>
```

## Important Notes

- GLUE path uses SDPA-oriented classifier wrappers.
- MTEB runner reads `mteb_task_type` from config; `--task_types` is currently
  parsed but not wired into task selection.
- MTEB output path is derived from run dir + checkpoint + tokenizer max length.

## GLUE Helpers

See `scripts/evaluation/glue/README.md` for:
- config generation from sweeps,
- quick/full suite launchers,
- result summarization.

## Related Docs

- [docs/evaluation.md](../../docs/evaluation.md)
- [docs/troubleshooting.md](../../docs/troubleshooting.md)
