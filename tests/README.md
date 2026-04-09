# NeoBERT Test Suite

Automated regression tests for model, training, config, and evaluation code. Commands and authoring guidance are in [docs/guides/testing.md](../docs/guides/testing.md).

## Layout

- Core regression files live directly under `tests/` (flat layout).
- Multi-file domains stay grouped under: `tests/training/`, `tests/evaluation/`, and `tests/kernels/`.
- `tests/configs/` - tiny smoke-test configs used by tests.
- `tests/manual/` - opt-in manual validation/benchmark scripts, excluded from default discovery.

## Related References

- [Testing guide](../docs/guides/testing.md)
- [Test configs](configs/README.md)
- [Manual validation scripts](manual/README.md)
