# Testing Guide

This guide covers running and extending the NeoBERT test suite.

> [!NOTE]
> Test layout is documented in [tests/README.md](../tests/README.md). Tiny configs live in [tests/configs/README.md](../tests/configs/README.md).

## Running Tests

### Run the full suite

```bash
pytest -q
```

Or use the repo helper:

```bash
python tests/run_tests.py
```

By default the helper uses pytest; pass `--no-pytest` to force unittest discovery.

### Run a subset

```bash
# One file
pytest tests/training/test_pretrain_pipeline.py -q

# A directory
pytest tests/model -q
```

### Verbose debugging

```bash
pytest tests/model/test_model_forward.py -vv --showlocals
```

## Writing Tests

- Prefer **small configs** from `tests/configs/` to keep runs fast.
- Avoid external network calls when possible (or mark as slow/integration).
- Disable W&B in tests that launch trainers: set `wandb.mode: "disabled"`.
- Keep GPU-only logic guarded with `torch.cuda.is_available()`.

## Common Issues

- **Import errors**: ensure the package is installed (`pip install -e .[dev]`).
- **CUDA/CPU mismatch**: move tensors/model to the same device before asserting shapes.
- **Slow tests**: reduce `max_steps`, batch size, or dataset size in the config.

## Next Steps

- Config reference: [docs/configuration.md](configuration.md)
- Training workflows: [docs/training.md](training.md)
