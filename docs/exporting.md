# Exporting NeoBERT Models to HuggingFace Format

This guide covers how to export trained NeoBERT checkpoints to HuggingFace format for easy deployment and sharing.

## Overview

NeoBERT training saves checkpoints in a specific format with `state_dict.pt` and `config.yaml` files. To use these models with HuggingFace Transformers or share them on the HuggingFace Hub, you need to convert them to HuggingFace format.

## Export Script

The export script is located at `scripts/export_to_huggingface.py`.

### Basic Usage

```bash
# Export a checkpoint
python scripts/export_to_huggingface.py outputs/neobert_100m_100k/model_checkpoints/100000

# Export to a specific directory
python scripts/export_to_huggingface.py outputs/neobert_100m_100k/model_checkpoints/100000 --output my_model
```

### What Gets Exported

The script creates a directory with:
- `config.json` - HuggingFace configuration
- `model.safetensors` and `pytorch_model.bin` - Model weights in both formats
- `model.py` and `rotary.py` - Model implementation files
- `tokenizer_config.json`, `tokenizer.json`, `vocab.txt` - Tokenizer files
- `README.md` - Auto-generated documentation

## Using Exported Models

Once exported, the model can be loaded with standard HuggingFace code:

```python
from transformers import AutoModel, AutoTokenizer

# Load from local directory
model_path = "outputs/neobert_100m_100k/hf/neobert_100m_100k_100000"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# Use the model
text = "NeoBERT is the most efficient model of its kind!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
```

### Loading as Masked Language Model

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
```

## Validation and Testing

Use the validation script to ensure your exported model works correctly:

```bash
python tests/test_huggingface_export.py outputs/neobert_100m_100k/hf/neobert_100m_100k_100000
```

The test script checks:
- ✅ All required files are present
- ✅ Model loads without initialization warnings
- ✅ Tokenizer works correctly
- ✅ MaskedLM variant loads properly
- ✅ End-to-end pipeline functions
- ✅ Cosine similarity produces meaningful results

### Example Test Output

```
Validating exported model: outputs/neobert_100m_100k/hf/neobert_100m_100k_100000
============================================================

Checking required files...
  ✅ All required files present

Testing model loading...
  Testing AutoModel.from_pretrained...
  Testing model forward pass...
  PASS: Model loaded successfully

Testing cosine similarity...
  Testing cosine similarity sanity check...
    Similar pair 1 (cat/feline): 0.835
    Similar pair 2 (programming): 0.924
    Different pair (cat/quantum): 0.488
  PASS: Cosine similarity sanity check passed

============================================================
SUMMARY
============================================================
✅ PASS     File validation     
✅ PASS     Model loading       
✅ PASS     Tokenizer loading   
✅ PASS     MaskedLM loading    
✅ PASS     Pipeline test       
✅ PASS     Cosine similarity   
============================================================
✅ All tests passed! Model is ready for use.
```

## Technical Details

### Weight Mapping

The export script handles the following transformations:
1. **Prefix handling**: Training checkpoints have `model.*` prefix which is preserved for HuggingFace compatibility
2. **Decoder weights**: LM head weights (`model.decoder.*`) are mapped to top-level `decoder.*`
3. **SwiGLU weights**: The concatenated `w12` weights are preserved (no splitting needed)

### Configuration Mapping

Training config fields are mapped to HuggingFace config:
- `max_position_embeddings` → `max_length`
- `layer_norm_eps` → `norm_eps`
- Training metadata is preserved in `training_info`

## Troubleshooting

### Initialization Warnings

If you see warnings about randomly initialized weights:
- Check that the checkpoint contains all expected weights
- Verify the model architecture matches between training and export
- Run the validation script to identify missing weights

### Tokenizer Issues

If tokenizer fails to load:
- Ensure tokenizer files exist in the checkpoint directory
- Check that `model_max_length` is an integer (not float)

### Performance Validation

The cosine similarity test ensures the model produces meaningful embeddings:
- Similar sentences should have similarity > 0.7
- Different sentences should have lower similarity
- If all similarities are near 1.0, the model may not be properly trained

## Next Steps

- Upload to HuggingFace Hub: Use `huggingface-cli upload` or the web interface
- Fine-tune for downstream tasks: See [evaluation.md](evaluation.md) for GLUE fine-tuning
- Deploy for inference: Compatible with any HuggingFace Transformers deployment