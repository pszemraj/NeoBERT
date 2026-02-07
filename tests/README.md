# NeoBERT Test Suite

Automated regression tests for model, training, config, and evaluation code.

## Entry Points

```bash
# Preferred
pytest -q

# Helper wrapper
python tests/run_tests.py
```

Useful helper flags:

```bash
python tests/run_tests.py --test-dir training
python tests/run_tests.py --pattern "test_*attention*.py"
python tests/run_tests.py --no-pytest
```

## Notes

- Tiny smoke configs live in `tests/configs/`.
- For full workflows and test authoring conventions, see
  [docs/testing.md](../docs/testing.md).
