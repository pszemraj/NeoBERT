# NeoBERT Test Suite

Automated regression tests for model, training, config, and evaluation code.
This README is intentionally lightweight; canonical testing guidance is in
[docs/testing.md](../docs/testing.md).

## Entry Points

```bash
# Preferred
conda run --name neobert pytest -q

# Helper wrapper
conda run --name neobert python tests/run_tests.py
```

Useful helper flags:

```bash
conda run --name neobert python tests/run_tests.py --test-dir training
conda run --name neobert python tests/run_tests.py --pattern "test_*attention*.py"
conda run --name neobert python tests/run_tests.py --no-pytest
```

## Notes

- Tiny smoke configs live in `tests/configs/`.
- Manual validation/benchmark scripts live in `tests/manual/` and are excluded
  from default `pytest -q` discovery.
- Run manual scripts explicitly when needed:
  - `conda run --name neobert pytest -q tests/manual/test_muonclip_integration.py -s`
  - `conda run --name neobert python tests/manual/test_muonclip_training.py`
  - `conda run --name neobert python tests/manual/validate_muonclip.py`
- For full workflows and test authoring conventions, see
  [docs/testing.md](../docs/testing.md).
