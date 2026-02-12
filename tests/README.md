# NeoBERT Test Suite

Automated regression tests for model, training, config, and evaluation code.
This README is intentionally lightweight; canonical testing guidance is in
[docs/testing.md](../docs/testing.md).

## Entry Points

```bash
# Preferred default suite
conda run --name neobert pytest -q

# Optional wrapper
conda run --name neobert python tests/run_tests.py
```

## Suite Layout

- Core regression files live directly under `tests/` (flat layout).
- Multi-file domains stay grouped under:
  `tests/training/`, `tests/evaluation/`, and `tests/kernels/`.
- `tests/configs/` - tiny smoke-test configs used by tests.
- `tests/manual/` - opt-in manual validation/benchmark scripts, excluded from
  default discovery.

## Canonical References

- Process and authoring conventions: [docs/testing.md](../docs/testing.md)
- Tiny test config catalog: [tests/configs/README.md](configs/README.md)
- Manual-script commands: [tests/manual/README.md](manual/README.md)
