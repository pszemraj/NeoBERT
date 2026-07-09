# Test Config Files

Small, fast configs used by unit/integration tests and smoke runs.

These configs are computationally small, but direct entrypoint runs may still require Hub access or a populated cache. Pipeline tests replace external datasets and output paths with local fixtures.

## Layout

- `pretraining/` - pretraining smoke and targeted runtime configs.
- `evaluation/` - tiny GLUE configs.
- `contrastive/` - tiny contrastive config.

## Notes

- These are intentionally tiny and not representative of production throughput.
- For production experiments, use configs under `configs/`.
- Field semantics and defaults are in [docs/reference/configuration.md](../../docs/reference/configuration.md).
