# NeoBERT

> [!IMPORTANT]
> This is a fork of the [original chandar-lab/NeoBERT](https://github.com/chandar-lab/NeoBERT), refactored to support experimentation. ⚠️ WIP/active development⚠️

---

- [NeoBERT](#neobert)
  - [Description](#description)
  - [Get started](#get-started)
  - [How to use](#how-to-use)
    - [For Text Embeddings](#for-text-embeddings)
    - [For Masked Language Modeling](#for-masked-language-modeling)
  - [Documentation](#documentation)
  - [Features](#features)
  - [License](#license)
  - [Citation](#citation)
  - [Training and Development](#training-and-development)
    - [Configuration System](#configuration-system)
    - [Repository Structure](#repository-structure)
    - [Quick Start for Training](#quick-start-for-training)
    - [Testing](#testing)
    - [Exporting Models to HuggingFace](#exporting-models-to-huggingface)

---

## Description

NeoBERT is a **next-generation encoder** model for English text representation, pre-trained from scratch on the RefinedWeb dataset. NeoBERT integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an **optimal depth-to-width ratio**, and leverages an extended context length of **4,096 tokens**. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves **state-of-the-art results** on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions.

- Paper: [paper](https://arxiv.org/abs/2502.19587)
- Model: [huggingface](https://huggingface.co/chandar-lab/NeoBERT)
- **Documentation: [docs/](docs/README.md)**

## Get started

Ensure you have the following dependencies installed:

```bash
pip install transformers torch  # Core dependencies
# For GPU optimization (build from source for latest GPUs):
# pip install flash-attn --no-build-isolation
# pip install -v --no-build-isolation git+https://github.com/facebookresearch/xformers.git@main
```

If you would like to use sequence packing (un-padding), you will need to also install flash-attention:

```bash
pip install transformers torch  # Core dependencies
# For GPU optimization (build from source for latest GPUs):
# pip install flash-attn --no-build-isolation
# pip install -v --no-build-isolation git+https://github.com/facebookresearch/xformers.git@main flash_attn
```

It is **much safer** to first install as stated above. Then, you can clone this repo and install it in editable mode:

```bash
git clone https://github.com/pszemraj/NeoBERT.git
cd NeoBERT
pip install -e .
```

This will install the `neobert` package and all remaining dependencies[^1].

[^1]: Technically, this command installs everything, but package order/resolution is not guaranteed, so it is better to install the core dependencies first.

> [!NOTE]
> If you want to install the development dependencies (for testing, linting, etc.), use `pip install -e .[dev]`.

## How to use

Load [the official model](https://huggingface.co/chandar-lab/NeoBERT) using Hugging Face Transformers:

### For Text Embeddings

```python
from transformers import AutoModel, AutoTokenizer

model_name = "chandar-lab/NeoBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Tokenize input text
text = "NeoBERT is the most efficient model of its kind!"
inputs = tokenizer(text, return_tensors="pt")

# Generate embeddings
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
print(embedding.shape)
```

### For Masked Language Modeling

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "chandar-lab/NeoBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

# Fill in masked tokens
text = "The quick brown [MASK] jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
outputs = model(**inputs)
mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))
```

## Documentation

For detailed guides and documentation, see the **[Documentation](docs/README.md)**:

- [Quick Start Guide](docs/quickstart.md) - Get up and running in 5 minutes
- [Training Guide](docs/training.md) - Pretraining and fine-tuning
- [Evaluation Guide](docs/evaluation.md) - GLUE and MTEB benchmarks
- [Configuration System](docs/configuration.md) - Understanding configs
- [Architecture Details](docs/architecture.md) - Technical model details

## Features

| **Feature**             | **NeoBERT**    |
| ----------------------- | -------------- |
| `Depth-to-width`        | 28 × 768       |
| `Parameter count`       | 250M           |
| `Activation`            | SwiGLU         |
| `Positional embeddings` | RoPE           |
| `Normalization`         | Pre-RMSNorm    |
| `Data Source`           | RefinedWeb     |
| `Data Size`             | 2.8 TB         |
| `Tokenizer`             | google/bert    |
| `Context length`        | 4,096          |
| `MLM Masking Rate`      | 20%            |
| `Optimizer`             | AdamW          |
| `Scheduler`             | CosineDecay    |
| `Training Tokens`       | 2.1 T          |
| `Efficiency`            | FlashAttention |

## License

Model weights and code repository are licensed under the permissive MIT license.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{breton2025neobertnextgenerationbert,
      title={NeoBERT: A Next-Generation BERT},
      author={Lola Le Breton and Quentin Fournier and Mariam El Mezouar and Sarath Chandar},
      year={2025},
      eprint={2502.19587},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19587},
}
```

## Training and Development

This repository includes the complete training and evaluation codebase for NeoBERT, featuring:

### Configuration System

- **Hierarchical YAML configs** with command-line overrides
- **Task-specific configurations** for pretraining, GLUE, contrastive learning, and MTEB evaluation
- **CPU-friendly test configs** for development and validation

```bash
# Basic training with config file
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml

# Override specific parameters
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 32 \
    --optimizer.lr 2e-4
```

### Repository Structure

- **`configs/`** - YAML configuration files ([README](configs/README.md))
- **`scripts/`** - Training and evaluation scripts ([README](scripts/README.md))
- **`jobs/`** - Shell scripts for running experiments ([README](jobs/README.md))
- **`tests/`** - Comprehensive test suite ([README](tests/README.md))
- **`src/neobert/`** - Core model and training code

### Quick Start for Training

1. **Install dependencies:**

   ```bash
   pip install -e .
   ```

2. **Run tests to validate setup:**

   ```bash
   python tests/run_tests.py
   ```

3. **Start with a small test run:**

   ```bash
   python scripts/pretraining/pretrain.py --config tests/configs/pretraining/test_tiny_pretrain.yaml
   ```

4. **Scale up to full training:**

   ```bash
   python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
   ```

### Testing

The repository includes a comprehensive test suite that verifies:

- Configuration system functionality
- Model architecture and forward passes
- Training pipeline integration
- CPU-only compatibility (no GPU required for tests)

### Exporting Models to HuggingFace

After training, export your model for use with HuggingFace Transformers:

```bash
# Export a checkpoint to HuggingFace format
python scripts/export-hf/export.py outputs/neobert_100m_100k/model_checkpoints/100000

# Validate the exported model
python scripts/export-hf/validate.py outputs/neobert_100m_100k/hf/neobert_100m_100k_100000
```

The exported model can then be used with standard HuggingFace code:

```python
from transformers import AutoModel, AutoTokenizer

model_path = "outputs/neobert_100m_100k/hf/neobert_100m_100k_100000"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
```

See the [Evaluation Guide](docs/evaluation.md#exporting-to-huggingface-format) for detailed export instructions.

---
