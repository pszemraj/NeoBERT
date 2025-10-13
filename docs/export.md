# HuggingFace Export Guide

This guide covers exporting NeoBERT checkpoints to HuggingFace format for easy distribution and deployment.

> [!NOTE]
> See [/scripts/export-hf/README.md](/scripts/export-hf/README.md) for implementation details and script-specific documentation.

## Overview

The export process converts NeoBERT training checkpoints into a format compatible with the HuggingFace Transformers library. This enables:

- Easy model sharing on HuggingFace Hub
- Standard transformers API usage
- Integration with existing pipelines
- Deployment with HuggingFace Inference Endpoints

## Prerequisites

Before exporting, ensure you have:

1. A trained NeoBERT checkpoint with:
   - `state_dict.pt` - Model weights
   - `config.yaml` - Training configuration
   - `tokenizer/` - Tokenizer files

2. Required dependencies:
   ```bash
   pip install transformers safetensors pyyaml
   ```

## Export Process

### Basic Export

Export a checkpoint to HuggingFace format:

```bash
# Export specific checkpoint
python scripts/export-hf/export.py outputs/neobert_100m_100k/model_checkpoints/100000

# Export with custom output directory
python scripts/export-hf/export.py \
    outputs/neobert_100m_100k/model_checkpoints/100000 \
    --output my_exported_model
```

### What Gets Exported

The export script creates:

```
exported_model/
├── config.json              # HuggingFace model configuration
├── model.safetensors       # Model weights (SafeTensors format)
├── pytorch_model.bin       # Model weights (PyTorch format)
├── model.py               # Custom model implementation
├── rotary.py             # Rotary embeddings implementation
├── tokenizer_config.json  # Tokenizer configuration
├── tokenizer.json        # Fast tokenizer
├── special_tokens_map.json # Special token mappings
├── vocab.txt            # Vocabulary file
└── README.md           # Auto-generated usage instructions
```

### Configuration Mapping

The export process maps NeoBERT configuration to HuggingFace format:

| NeoBERT Config | HuggingFace Config | Description |
|----------------|-------------------|-------------|
| `hidden_size` | `hidden_size` | Model dimension |
| `num_hidden_layers` | `num_hidden_layers` | Number of layers |
| `num_attention_heads` | `num_attention_heads` | Attention heads |
| `intermediate_size` | `intermediate_size` | FFN dimension |
| `max_position_embeddings` | `max_length` | Maximum sequence length |
| `norm_eps` | `norm_eps` | Layer norm epsilon |
| `vocab_size` | `vocab_size` | Vocabulary size |
| `pad_token_id` | `pad_token_id` | Padding token ID |

### Weight Mapping

The export handles weight conversion:

- **Encoder weights**: Direct mapping with `model.` prefix
- **Decoder weights**: Mapped for MLM head compatibility
- **SwiGLU weights**: Properly formatted for xformers/native implementations
- **RoPE embeddings**: Precomputed and stored as buffers

## Validation

### Quick Validation

Verify the exported model works:

```bash
python scripts/export-hf/validate.py my_exported_model

# Expected output:
# ✓ Model loaded successfully
# ✓ Tokenizer loaded successfully
# ✓ Forward pass successful
# ✓ Output shape correct: torch.Size([1, 128, 768])
# ✓ Masked language modeling working
# ✓ All validations passed!
```

### Manual Testing

Test the exported model manually:

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "my_exported_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)

# Test masked language modeling
text = "NeoBERT is a [MASK] model for NLP tasks."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits[0, 4].topk(5)  # Get top 5 predictions for [MASK]

for idx in predictions.indices:
    print(f"Predicted: {tokenizer.decode([idx])}")
```

## Publishing to HuggingFace Hub

### Prepare for Upload

1. **Create model card** (optional):
   ```python
   model_card = """
   ---
   language: en
   license: mit
   tags:
   - neobert
   - masked-language-modeling
   datasets:
   - your-dataset
   ---

   # Your Model Name

   Description of your fine-tuned model...
   """

   with open("my_exported_model/README.md", "w") as f:
       f.write(model_card)
   ```

2. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```

### Upload Model

Upload using the CLI:

```bash
# Create repository and upload
huggingface-cli repo create your-model-name --type model
huggingface-cli upload your-username/your-model-name my_exported_model/
```

Or using Python:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="my_exported_model",
    repo_id="your-username/your-model-name",
    repo_type="model",
)
```

## Using Exported Models

### Local Usage

```python
from transformers import AutoModel, AutoTokenizer

# Load from local path
model = AutoModel.from_pretrained("my_exported_model", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("my_exported_model", trust_remote_code=True)
```

### HuggingFace Hub Usage

```python
from transformers import AutoModel, AutoTokenizer

# Load from Hub
model = AutoModel.from_pretrained("your-username/your-model-name", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model-name", trust_remote_code=True)
```

### Integration Examples

**Text Classification Pipeline**:
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="your-username/your-model-name",
    trust_remote_code=True
)

result = classifier("This movie is fantastic!")
```

**Feature Extraction**:
```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("your-username/your-model-name", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model-name", trust_remote_code=True)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

embeddings = get_embeddings(["Hello world", "NeoBERT rocks!"])
```

## Troubleshooting

### Common Issues

**1. Missing tokenizer files**:
```
FileNotFoundError: tokenizer/special_tokens_map.json not found
```
Solution: Ensure the checkpoint includes a complete tokenizer directory.

**2. Config validation errors**:
```
ValueError: Missing required config fields: ['hidden_size', ...]
```
Solution: Check that your training config.yaml contains all required fields.

**3. Weight shape mismatches**:
```
RuntimeError: Error(s) in loading state_dict
```
Solution: Ensure you're exporting from a valid NeoBERT checkpoint with matching architecture.

**4. Trust remote code warning**:
```
The repository contains custom code which must be executed to load the model
```
Solution: Always use `trust_remote_code=True` when loading NeoBERT models.

### Debug Mode

Enable verbose output for debugging:

```bash
# Set debug environment variable
export NEOBERT_DEBUG=1
python scripts/export-hf/export.py checkpoint_path

# Or use Python logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Options

### Custom Export Settings

Modify export behavior in the script:

```python
# In export.py, customize the configuration mapping
hf_config = {
    "architectures": ["NeoBERTLMHead"],  # Change default architecture
    "model_type": "neobert",
    "torch_dtype": "bfloat16",  # Force specific dtype
    # Add custom fields...
}
```

### Batch Export

Export multiple checkpoints:

```bash
#!/bin/bash
for checkpoint in outputs/*/model_checkpoints/*; do
    if [ -f "$checkpoint/state_dict.pt" ]; then
        echo "Exporting $checkpoint..."
        python scripts/export-hf/export.py "$checkpoint"
    fi
done
```

### Model Quantization

Apply quantization during export (requires additional libraries):

```python
from transformers import AutoModelForMaskedLM
import torch

# Load exported model
model = AutoModelForMaskedLM.from_pretrained("my_exported_model", trust_remote_code=True)

# Quantize to INT8
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
model_int8.save_pretrained("my_exported_model_int8")
```

## Best Practices

1. **Always validate** exported models before publishing
2. **Include model cards** with training details and intended use
3. **Version your models** using tags or version numbers
4. **Test compatibility** with target deployment environments
5. **Document special requirements** (e.g., flash-attn for long contexts)
6. **Preserve training configs** for reproducibility

## Next Steps

- [Evaluation Guide](/docs/evaluation.md) - Run benchmarks on exported models
- [Training Guide](/docs/training.md) - Train custom NeoBERT models
- [Configuration Guide](/docs/configuration.md) - Understand configuration options
