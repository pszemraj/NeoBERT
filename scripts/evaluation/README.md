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

- Evaluation behavior/caveats are documented in
  [docs/evaluation.md](../../docs/evaluation.md).
- Keep this README focused on script entry points and helpers.

## GLUE Helpers

See `scripts/evaluation/glue/README.md` for:
- config generation from sweeps,
- quick/full suite launchers,
- result summarization.

## Related Docs

- [docs/README.md](../../docs/README.md)
- [docs/evaluation.md](../../docs/evaluation.md)
- [docs/troubleshooting.md](../../docs/troubleshooting.md)
