# Manual Validation Scripts

These scripts are intentionally excluded from default `pytest -q` runs.

Use them for opt-in manual validation and benchmarking:

```bash
conda run --name neobert pytest -q tests/manual/test_muonclip_integration.py -s
conda run --name neobert python tests/manual/test_muonclip_training.py
conda run --name neobert python tests/manual/validate_muonclip.py
```

Notes:

- `test_muonclip_training.py` can run for multiple minutes and may download datasets.
- These scripts are for exploratory/perf validation, not fast CI regression checks.
