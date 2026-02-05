# HuggingFace Export Scripts

Scripts for exporting NeoBERT checkpoints to HuggingFace format.

> [!NOTE]
> See [docs/export.md](../../docs/export.md) for usage, constraints, and validation details.

## Scripts in this Directory

- **`export.py`** - Main export script that converts checkpoints to HuggingFace format
- **`validate.py`** - Validation script to test exported models

## Notes

- Metaspace tokenizer handling and MLM `[MASK]` quirks: see [docs/troubleshooting.md](../../docs/troubleshooting.md#mlm-always-predicts-same-token).
- Export constraints, validation checks, and config/weight mapping details are documented in [docs/export.md](../../docs/export.md).
