# Quick Start Guide

This guide will help you get started with NeoBERT quickly.

## Installation

```bash
# Clone the repository
git clone https://github.com/pszemraj/NeoBERT.git
cd NeoBERT

# Install in development mode
pip install -e .

# For GPU training with flash attention (optional)
pip install flash-attn --no-build-isolation
```

## Basic Usage

### 1. Pretraining a Small Model

```bash
# Pretrain a small model for testing
# The trainer will automatically tokenize raw text datasets
python scripts/pretraining/pretrain.py --config tests/configs/pretraining/test_tiny_pretrain.yaml

# What happens automatically:
# 1. Loads pszemraj/simple_wikipedia_LM dataset (raw text)
# 2. Detects the 'text' column needs tokenization
# 3. Tokenizes using bert-base-uncased tokenizer
# 4. Starts training with MLM objective

# Pretrain with custom settings
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.max_steps 1000 \
    --trainer.per_device_train_batch_size 8
```

### 2. Using a Custom Tokenizer

```bash
# Option 1: Let the trainer tokenize automatically
python scripts/pretraining/pretrain.py \
    --config tests/configs/pretraining/test_tiny_pretrain.yaml \
    --tokenizer.name "your-tokenizer-name" \
    --model.vocab_size 32000  # Match your tokenizer's vocab

# Option 2: Pre-tokenize for faster startup (optional)
python scripts/pretraining/tokenize_dataset.py \
    --tokenizer "your-tokenizer-name" \
    --dataset "your-dataset" \
    --output "./tokenized_data/custom"

# Then train with pre-tokenized data
python scripts/pretraining/pretrain.py \
    --config configs/train_small_custom_tokenizer.yaml
```

### 3. Evaluating on GLUE

```bash
# Run GLUE evaluation
python scripts/evaluation/run_glue.py \
    --config configs/evaluate_neobert.yaml \
    --task_name cola

# Evaluate on all GLUE tasks
python scripts/evaluation/run_glue.py \
    --config configs/evaluate_neobert.yaml \
    --task_name all
```

### 4. Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test suite
python tests/run_tests.py --test-dir model
python tests/run_tests.py --test-dir training
```

## Key Configuration Options

### Model Configuration
- `model.hidden_size`: Hidden dimension size (e.g., 768)
- `model.num_hidden_layers`: Number of transformer layers (e.g., 12)
- `model.num_attention_heads`: Number of attention heads (e.g., 12)
- `model.hidden_act`: Activation function ("gelu" for CPU, "swiglu" for GPU with xformers)

### Training Configuration
- `trainer.max_steps`: Maximum training steps
- `trainer.per_device_train_batch_size`: Batch size per device
- `trainer.learning_rate`: Learning rate (via optimizer.lr)
- `trainer.output_dir`: Where to save checkpoints

### Data Configuration
- `dataset.name`: HuggingFace dataset name or path to local data
- `dataset.streaming`: Enable streaming for large datasets (memory efficient)
- `tokenizer.name`: Tokenizer identifier
- `dataset.max_seq_length`: Maximum sequence length

## Common Issues

### XFormers/Flash Attention
If you encounter xformers errors, use GELU activation instead:
```bash
--model.hidden_act gelu --model.flash_attention false
```

### Memory Issues
Reduce batch size or use gradient accumulation:
```bash
--trainer.per_device_train_batch_size 4 \
--trainer.gradient_accumulation_steps 4
```

### Dataset Not Tokenized
For raw text datasets, tokenize first using the tokenize_dataset.py script.

## Next Steps

- Read the [Configuration Guide](configuration.md) for detailed configuration options
- Check out [Training Guide](training.md) for advanced training techniques
- See [Custom Tokenizers](custom_tokenizers.md) for using your own tokenizer