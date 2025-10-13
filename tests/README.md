# NeoBERT Test Suite

This directory houses the automated regression suite for NeoBERT.

> [!NOTE]
> For the complete testing guide—including workflow details, writing new tests, and troubleshooting—visit [/docs/testing.md](/docs/testing.md). Configuration-specific documentation lives in [/tests/configs/README.md](/tests/configs/README.md).

## Layout

```
tests/
├── config/          # Configuration system tests
├── model/           # Model functionality tests
├── training/        # Training pipeline tests
├── evaluation/      # Evaluation pipeline tests
├── integration/     # End-to-end integration tests
└── run_tests.py     # Unified test runner
```

## Quick Commands

```bash
# Run the full suite (CPU-friendly defaults)
python tests/run_tests.py

# Target a specific area
python tests/run_tests.py --test-dir config

# Increase or decrease verbosity
python tests/run_tests.py --verbose
python tests/run_tests.py --quiet
```

## Notes

- Tiny configs in `tests/configs/` keep runtimes short and avoid GPU requirements.
- Some scenarios skip automatically when optional deps (e.g., `xformers`) are absent.
