# NeoBERT

## Description

NeoBERT is a **next-generation encoder** model for English text representation, pre-trained from scratch on the RefinedWeb dataset. NeoBERT integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an **optimal depth-to-width ratio**, and leverages an extended context length of **4,096 tokens**. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves **state-of-the-art results** on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions. 

- Paper: [paper](https://arxiv.org/abs/2502.19587)
- Model: [huggingface](https://huggingface.co/chandar-lab/NeoBERT)
- Documentation: [docs/](docs/README.md)

## Get started

Ensure you have the following dependencies installed:

```bash
pip install transformers torch xformers==0.0.28.post3
```

If you would like to use sequence packing (un-padding), you will need to also install flash-attention:

```bash
pip install transformers torch xformers==0.0.28.post3 flash_attn
```

## How to use

Load the model using Hugging Face Transformers:

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
embedding = outputs.last_hidden_state[:, 0, :]
print(embedding.shape)
```

## Documentation

For detailed guides and documentation, see the [docs/](docs/) directory:

- [Quick Start Guide](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Training Guide](docs/training.md)
- [Custom Tokenizers](docs/custom_tokenizers.md)
- [Evaluation Guide](docs/evaluation.md)
- [Architecture Details](docs/architecture.md)

## Features
| **Feature**       | **NeoBERT**                             |
|---------------------------|-----------------------------|
| `Depth-to-width`        | 28 × 768  |
| `Parameter count`           | 250M                        |
| `Activation`               | SwiGLU                      |
| `Positional embeddings`     | RoPE                        |
| `Normalization`            | Pre-RMSNorm                 |
| `Data Source`              | RefinedWeb                  |
| `Data Size`                | 2.8 TB                       |
| `Tokenizer`                | google/bert                 |
| `Context length`    | 4,096                       |
| `MLM Masking Rate`             | 20%                         |
| `Optimizer`                | AdamW                       |
| `Scheduler`                | CosineDecay                 |
| `Training Tokens`          | 2.1 T                        |
| `Efficiency`               | FlashAttention              |

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

### 🔧 **Configuration System**
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

### 📁 **Repository Structure**
- **`configs/`** - YAML configuration files ([README](configs/README.md))
- **`scripts/`** - Training and evaluation scripts ([README](scripts/README.md))
- **`jobs/`** - Shell scripts for running experiments ([README](jobs/README.md))
- **`tests/`** - Comprehensive test suite ([README](tests/README.md))
- **`src/neobert/`** - Core model and training code

### 🚀 **Quick Start for Training**

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
   python scripts/pretraining/pretrain.py --config configs/test_tiny_pretrain.yaml
   ```

4. **Scale up to full training:**
   ```bash
   python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
   ```

### 🧪 **Testing**
The repository includes a comprehensive test suite that verifies:
- Configuration system functionality
- Model architecture and forward passes  
- Training pipeline integration
- CPU-only compatibility (no GPU required for tests)

### 📖 **Documentation**
Each directory contains detailed README files with usage examples, best practices, and troubleshooting guides.

## Contact

For questions, do not hesitate to reach out and open an issue on here or on our **[GitHub](https://github.com/chandar-lab/NeoBERT)**.

---