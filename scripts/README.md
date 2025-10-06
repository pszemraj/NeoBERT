# Scripts

This directory contains Python scripts for training, evaluation, and data processing. All scripts use the new hierarchical YAML configuration system with command-line override support.

> [!NOTE]
> See [/docs/training.md](/docs/training.md) for the training guide, [/docs/evaluation.md](/docs/evaluation.md) for evaluation guide, and [/docs/export.md](/docs/export.md) for export documentation.

## Directory Structure

```
scripts/
├── README.md                    # This file
├── pretraining/                # Pretraining scripts
│   ├── pretrain.py             # Main pretraining script
│   ├── preprocess.py           # Data preprocessing
│   └── longer_seq.py           # Extended sequence length training
├── evaluation/                 # Evaluation scripts
│   ├── run_glue.py            # GLUE benchmark evaluation
│   ├── run_mteb.py            # MTEB benchmark evaluation
│   ├── pseudo_perplexity.py   # Perplexity evaluation
│   ├── avg_mteb.py            # MTEB result aggregation
│   └── wrappers.py            # Evaluation utilities
└── contrastive/               # Contrastive learning scripts
    ├── download.py            # Download contrastive datasets
    ├── preprocess.py          # Preprocess contrastive data
    └── finetune.py            # Contrastive fine-tuning
```

## Core Scripts

### Pretraining Scripts

#### `pretraining/pretrain.py`
Main script for pretraining NeoBERT models.

**Usage:**
```bash
# Basic usage with config file
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml

# With command-line overrides
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 32 \
    --optimizer.lr 2e-4 \
    --wandb.project my-project
```

**Key Features:**
- Supports both masked language modeling and next sentence prediction
- Automatic mixed precision training with BF16 (recommended for modern GPUs: RTX 30xx+, A100, H100)
- Gradient checkpointing for memory efficiency
- Integration with Weights & Biases for experiment tracking
- Multi-GPU training with Accelerate
- Streaming dataset support for memory-efficient training on large datasets

#### `pretraining/preprocess.py`
Preprocesses raw text data for pretraining.

**Usage:**
```bash
python scripts/pretraining/preprocess.py \
    --config configs/pretrain_neobert.yaml \
    --dataset.input_dir /path/to/raw/text \
    --dataset.output_dir /path/to/tokenized/data
```

#### `pretraining/longer_seq.py`
Extends sequence length for continued pretraining.

**Usage:**
```bash
python scripts/pretraining/longer_seq.py \
    --config configs/pretrain_neobert.yaml \
    --model.max_position_embeddings 1024 \
    --trainer.resume_from_checkpoint /path/to/checkpoint
```

### Evaluation Scripts

#### `evaluation/run_glue.py`
Evaluates models on the GLUE benchmark.

**Usage:**
```bash
# Evaluate on all GLUE tasks
python scripts/evaluation/run_glue.py \
    --config configs/evaluate_neobert.yaml \
    --model.checkpoint_path /path/to/checkpoint

# Evaluate on specific tasks
python scripts/evaluation/run_glue.py \
    --config configs/evaluate_neobert.yaml \
    --evaluation.tasks "cola,sst2,mrpc" \
    --model.checkpoint_path /path/to/checkpoint
```

**Supported Tasks:**
- CoLA (Corpus of Linguistic Acceptability)
- SST-2 (Stanford Sentiment Treebank)
- MRPC (Microsoft Research Paraphrase Corpus)
- QQP (Quora Question Pairs)
- MNLI (Multi-Genre Natural Language Inference)
- QNLI (Question Natural Language Inference)
- RTE (Recognizing Textual Entailment)
- WNLI (Winograd Natural Language Inference)

#### `evaluation/run_mteb.py`
Evaluates models on the MTEB (Massive Text Embedding Benchmark).

**Usage:**
```bash
python scripts/evaluation/run_mteb.py \
    --config configs/evaluate_neobert.yaml \
    --model.checkpoint_path /path/to/checkpoint \
    --evaluation.mteb_tasks "STS12,STS13,STS14,STS15,STS16"
```

#### `evaluation/pseudo_perplexity.py`
Computes pseudo-perplexity for model evaluation.

**Usage:**
```bash
python scripts/evaluation/pseudo_perplexity.py \
    --model_name neobert \
    --config_path configs/pretrain_neobert.yaml \
    --checkpoint_path /path/to/checkpoint \
    --data_name wikipedia \
    --output_path ./perplexity_results
```

### Contrastive Learning Scripts

#### `contrastive/download.py`
Downloads and prepares contrastive learning datasets.

**Usage:**
```bash
python scripts/contrastive/download.py \
    --datasets "ALLNLI,MSMARCO,QQP" \
    --output_dir /path/to/contrastive/data
```

**Supported Datasets:**
- ALLNLI, AMAZONQA, CONCURRENTQA, FEVER
- GITHUBISSUE, GOOAQ, MSMARCO, PAQ
- PUBMEDQA, QQP, SENTENCECOMP
- STACKEXCHANGE, STACKOVERFLOW, STS12
- STSBENCHMARK, TRIVIAQA, WIKIHOW

#### `contrastive/preprocess.py`
Preprocesses contrastive learning datasets.

**Usage:**
```bash
python scripts/contrastive/preprocess.py \
    --config configs/contrastive_neobert.yaml \
    --datasets.path /path/to/raw/data \
    --tokenizer.max_length 512
```

#### `contrastive/finetune.py`
Fine-tunes models using contrastive learning.

**Usage:**
```bash
python scripts/contrastive/finetune.py \
    --config configs/contrastive_neobert.yaml \
    --model.checkpoint_path /path/to/pretrained/model \
    --trainer.output_dir ./output/contrastive
```

## Common Usage Patterns

### 1. Configuration-First Approach

All scripts use YAML configuration files as the primary method for specifying parameters:

```bash
# Create/modify a config file first
cp configs/pretrain_neobert.yaml configs/my_config.yaml
# Edit my_config.yaml

# Run script with config
python scripts/pretraining/pretrain.py --config configs/my_config.yaml
```

### 2. Command-Line Overrides

Override specific parameters without modifying config files:

```bash
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 64 \
    --optimizer.lr 1e-3 \
    --trainer.max_steps 100000
```

### 3. Environment Variables

Use environment variables for sensitive or environment-specific values:

```bash
export WANDB_PROJECT=my-neobert-project
export HF_DATASETS_CACHE=/fast/cache/datasets

python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
```

### 4. Debug Mode

All scripts support debug mode for development and troubleshooting:

```bash
python scripts/pretraining/pretrain.py \
    --config configs/test_tiny_pretrain.yaml \
    --debug
```

## Script Arguments

### Universal Arguments

All scripts support these common arguments:

- `--config`: Path to YAML configuration file (required)
- `--debug`: Enable debug mode with verbose logging
- `--help`: Show help message and available overrides

### Configuration Overrides

Any configuration parameter can be overridden using dot notation:

```bash
# Model parameters
--model.hidden_size 512
--model.num_hidden_layers 8

# Training parameters  
--trainer.per_device_train_batch_size 32
--trainer.learning_rate 2e-4

# Dataset parameters
--dataset.max_seq_length 1024
--dataset.num_workers 8

# Optimizer parameters
--optimizer.name adamw
--optimizer.weight_decay 0.01
```

## Integration with Other Tools

### Weights & Biases

Scripts integrate with W&B for experiment tracking:

```yaml
# In config file
wandb:
  project: my-project
  entity: my-team
  tags: [neobert, experiment]
  mode: online
```

### Accelerate (Multi-GPU)

Use Accelerate for distributed training:

```bash
# Create accelerate config
accelerate config

# Run with accelerate
accelerate launch scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml
```

### HuggingFace Hub

Models can be pushed to the Hub after training:

```python
# In your training script
model.push_to_hub("my-username/my-neobert-model")
tokenizer.push_to_hub("my-username/my-neobert-model")
```

## Development Guidelines

### Adding New Scripts

1. **Follow the config pattern**: Use the configuration system
2. **Add argument parsing**: Support config file + overrides
3. **Include debug mode**: Add verbose logging when `--debug` is used
4. **Document usage**: Add docstrings and usage examples
5. **Handle errors gracefully**: Provide helpful error messages

### Script Template

```python
#!/usr/bin/env python3
"""
Script description.
"""

import argparse
import logging
from pathlib import Path

from neobert.config import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Parse known args to allow config overrides
    args, unknown = parser.parse_known_args()
    
    # Load config with overrides
    config = ConfigLoader.load_with_overrides(args.config, unknown)
    
    # Set up logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    
    # Your script logic here
    pass

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Config validation errors**: Check YAML syntax and required fields
2. **Import errors**: Ensure environment is properly set up
3. **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
4. **Slow data loading**: Increase `num_workers` or use faster storage

### Debug Tips

```bash
# Enable debug mode
python script.py --config config.yaml --debug

# Check config loading
python -c "from neobert.config import ConfigLoader; print(ConfigLoader.load('config.yaml'))"

# Validate environment
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```