# Testing Guide

Use the regression suite for model, training, configuration, and evaluation changes.

## Run Tests

### Full suite

```bash
pytest -q
```

### Subsets

```bash
# One file
pytest tests/kernels/test_attention.py -q

# One directory
pytest tests/training -q

# Verbose investigation
pytest tests/test_model_forward.py -vv --showlocals

# Match test names
pytest -k compile -q
```

## Manual Validation Scripts

Run the opt-in distributed checks from [tests/manual/README.md](../../tests/manual/README.md); they are excluded from default `pytest -q` discovery.

## Lint and Format

```bash
ruff check --fix .
ruff format .
```

Ruff enforces import sorting with its normal lint pass.

## Test Authoring Guidelines

- Prefer tiny configs in `tests/configs/`.
- Keep tests deterministic and local (avoid network where possible).
- Disable external logging for training tests with `wandb.enabled: false`.
- Guard GPU-only assertions with `torch.cuda.is_available()`.
- Emit actionable project warnings as `NeoBERTWarning`; pytest treats that category as an error while leaving dependency warnings nonfatal.
- For performance-sensitive paths (packing/compile), include regression tests for both correctness and expected control-flow behavior.

## Common Failures

### Import errors

- install editable package (`pip install -e .[dev]`).

### Device mismatches

- ensure tensors and models are on the same device in assertions.

### Slow tests

- lower steps/batch sizes and use tiny configs.

## Related Docs

- [tests/README.md](../../tests/README.md)
- [tests/configs/README.md](../../tests/configs/README.md)
- [YAML configuration reference](../reference/config_reference.yaml)
