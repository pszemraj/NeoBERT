# Test Config Files

Small, fast configs used by unit/integration tests and smoke runs.

## Layout

- `pretraining/` - pretraining smoke and targeted runtime configs.
- `evaluation/` - tiny GLUE configs.
- `contrastive/` - tiny contrastive config.

## Example Usage

```bash
conda run --name neobert python scripts/pretraining/pretrain.py tests/configs/pretraining/test_tiny_pretrain.yaml
conda run --name neobert python scripts/evaluation/run_glue.py tests/configs/evaluation/test_tiny_glue.yaml
conda run --name neobert python scripts/contrastive/finetune.py tests/configs/contrastive/test_tiny_contrastive.yaml
```

## Notes

- These are intentionally tiny and not representative of production throughput.
- For production experiments, use configs under `configs/`.
- Field semantics and defaults are canonical in
  [docs/configuration.md](../../docs/configuration.md).
