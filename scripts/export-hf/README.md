# HuggingFace Export Scripts

Scripts for exporting NeoBERT checkpoints to HuggingFace format.

> [!NOTE]
> See [docs/export.md](../../docs/export.md) for the full export guide.

## Quick Start

```bash
# Export a checkpoint
python scripts/export-hf/export.py outputs/neobert_100m_100k/model_checkpoints/100000

# Validate exported model
python scripts/export-hf/validate.py outputs/neobert_100m_100k/hf/neobert_100m_100k_100000
```

## Scripts in this Directory

- **`export.py`** - Main export script that converts checkpoints to HuggingFace format
- **`validate.py`** - Validation script to test exported models

## Implementation Notes

### Metaspace Tokenizer Handling

Some of the default configs for pretraining use [a Metaspace tokenizer](https://huggingface.co/BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp) with `prepend_scheme=\"always\"`. The export script emits guidance for cleaning up the extra space token (▁, ID 454) that appears before `[MASK]`. Refer to [MLM always predicts the same token](../../docs/troubleshooting.md#mlm-always-predicts-same-token) for the canonical workaround.

### Validation Test Details

The validation script (`validate.py`) performs these checks:

- File presence validation
- Model loading without initialization warnings
- Tokenizer functionality
- MaskedLM variant compatibility
- End-to-end pipeline test
- Cosine similarity sanity check (similar pairs >= 0.5, dissimilar <= 0.95)

### Technical Implementation Details

**Weight Mapping Logic:**

- Training checkpoints with `model.*` prefix are preserved for HuggingFace compatibility
- LM head weights (`model.decoder.*`) are mapped to top-level `decoder.*`
- SwiGLU uses unpacked `w1`/`w2`/`w3` weights; export preserves them without conversion

**Config Field Mappings:**

- `max_position_embeddings` → `max_length`
- `norm_eps` preserved in HF config

### Known Issues & Solutions

**MLM Always Predicting Same Token:**

- Follow the mitigation steps in [docs/troubleshooting.md](../../docs/troubleshooting.md#mlm-always-predicts-same-token) (accelerator unwrap, Metaspace cleanup, attention-mask guard)

**Initialization Warnings:**
Check missing weights with validation script, verify architecture match between training and export.

**Tokenizer Loading Failures:**
Ensure `model_max_length` is integer (not float) in tokenizer_config.json.
