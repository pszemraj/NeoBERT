# Testing Guide

This guide covers running and extending the NeoBERT regression suite.
It is the canonical reference for test process and authoring conventions.

## Run Tests

### Full suite

```bash
pytest -q
```

Or via helper:

```bash
python tests/run_tests.py
```

### Subsets

```bash
# One file
pytest tests/kernels/test_attention.py -q

# One directory
pytest tests/model -q

# Verbose investigation
pytest tests/model/test_model_forward.py -vv --showlocals
```

### Helper flags

```bash
python tests/run_tests.py --test-dir training
python tests/run_tests.py --pattern "test_*compile*.py"
python tests/run_tests.py --no-pytest
```

## Test Authoring Guidelines

- Prefer tiny configs in `tests/configs/`.
- Keep tests deterministic and local (avoid network where possible).
- Disable external logging for training tests (`wandb.mode: disabled`).
- Guard GPU-only assertions with `torch.cuda.is_available()`.
- For performance-sensitive paths (packing/compile), include regression tests for
  both correctness and expected control-flow behavior.

## Common Failures

1. Import errors

- install editable package (`pip install -e .[dev]`).

1. Device mismatches

- ensure tensors and models are on the same device in assertions.

1. Slow tests

- lower steps/batch sizes and use tiny configs.

## Related Docs

- [tests/README.md](../tests/README.md)
- [tests/configs/README.md](../tests/configs/README.md)
- [Configuration](configuration.md)
