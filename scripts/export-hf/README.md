# HuggingFace Export Scripts

Scripts for exporting NeoBERT checkpoints to HuggingFace format.

📚 **For comprehensive documentation, see [docs/export.md](../../docs/export.md)**

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

NeoBERT uses a Metaspace tokenizer with `prepend_scheme="always"`. When using `[MASK]` directly in text (vs. masking existing tokens during training), an extra space token (▁, ID 454) is inserted that needs special handling:

```python
# The export script automatically generates code to handle this in the README
# See the exported model's README for the complete implementation
```

### Validation Test Details

The validation script (`validate.py`) performs these checks:
- File presence validation
- Model loading without initialization warnings
- Tokenizer functionality
- MaskedLM variant compatibility
- End-to-end pipeline test
- Cosine similarity sanity check (similar sentences > 0.7, different < 0.5)

### Technical Implementation Details

**Weight Mapping Logic:**
- Training checkpoints with `model.*` prefix are preserved for HuggingFace compatibility
- LM head weights (`model.decoder.*`) are mapped to top-level `decoder.*`
- SwiGLU concatenated `w12` weights are preserved without splitting

**Config Field Mappings:**
- `max_position_embeddings` → `max_length`
- `layer_norm_eps` → `norm_eps`
- Training metadata preserved in `training_info` field

### Known Issues & Solutions

**MLM Always Predicting Same Token:**
- Ensure model unwrapped from accelerator before saving: `accelerator.unwrap_model(model).state_dict()`
- Handle Metaspace tokenizer space tokens (ID 454) before [MASK]
- Verify attention_mask None checking in HF model forward pass

**Initialization Warnings:**
Check missing weights with validation script, verify architecture match between training and export.

**Tokenizer Loading Failures:**
Ensure `model_max_length` is integer (not float) in tokenizer_config.json.