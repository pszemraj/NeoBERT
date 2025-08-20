# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeoBERT is a next-generation 250M parameter encoder model for English text representation, designed as a modern alternative to BERT with architectural improvements including RoPE, RMSNorm, SwiGLU activation, and Flash Attention. It supports 4,096 token context length and achieves state-of-the-art results on MTEB benchmark.

## Essential Commands

### Testing
```bash
# Run all tests (CPU-compatible, no GPU required)
python tests/run_tests.py

# Test specific category
python tests/run_tests.py --test-dir config      # Configuration system
python tests/run_tests.py --test-dir model       # Model architecture
python tests/run_tests.py --test-dir training    # Training pipelines
python tests/run_tests.py --test-dir evaluation  # Evaluation systems
python tests/run_tests.py --test-dir integration # End-to-end tests

# Verbose output
python tests/run_tests.py --verbose
```

### Training
```bash
# Basic pretraining
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml

# With configuration overrides (dot notation)
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 32 \
    --optimizer.lr 2e-4

# Quick test run with tiny model
python scripts/pretraining/pretrain.py --config tests/configs/pretraining/test_tiny_pretrain.yaml
```

### Evaluation
```bash
# GLUE evaluation
python scripts/evaluation/run_glue.py --config configs/evaluate_neobert.yaml

# MTEB evaluation
python scripts/evaluation/run_mteb.py --config configs/evaluate_neobert.yaml
```

## Architecture & Key Design Patterns

### Core Module Structure
- **src/neobert/model/**: Core model implementation with RoPE, RMSNorm, SwiGLU
  - `neobert.py`: Main model class
  - `attention.py`: Multi-head attention with Flash Attention support
  - `embeddings.py`: Token and position embeddings
  - `layers.py`: Transformer layers with pre-norm architecture

- **src/neobert/config/**: Hierarchical configuration system using dataclasses
  - Type-safe configuration with YAML support
  - Supports CLI overrides with dot notation
  - Separate configs for model, training, optimizer, scheduler

- **src/neobert/training/**: Task-specific trainers
  - `pretraining.py`: MLM pretraining trainer
  - `glue.py`: GLUE fine-tuning trainer
  - `contrastive.py`: Contrastive learning trainer
  - All trainers support streaming datasets and mixed precision

### Configuration System
The project uses a hierarchical YAML configuration system with type-safe dataclasses:
1. Load base config from YAML file
2. Override with CLI arguments using dot notation
3. Validate types and constraints
4. Access via dataclass attributes

Example: `--model.hidden_size 1024 --optimizer.lr 2e-4`

### Key Architectural Decisions
- **No token type embeddings**: Simplified design compared to BERT
- **RoPE instead of absolute position embeddings**: Better extrapolation
- **Pre-norm architecture**: Improved gradient flow
- **Modular activation/normalization**: Easy to swap components
- **CPU fallback**: All components work without GPU/xformers

### Testing Philosophy
- All tests must run on CPU (use tiny model configs)
- Test files mirror source structure
- Integration tests for end-to-end workflows
- Custom test runner with category filtering

## Dependencies & Compatibility

### Core Requirements
- PyTorch >= 2.0
- Transformers (HuggingFace integration)
- xformers==0.0.28.post3 (SwiGLU optimization, optional)
- flash_attn (Flash Attention, optional but recommended)

### GPU Optimization
The model automatically detects and uses:
1. Flash Attention if available
2. xformers SwiGLU if available
3. Falls back to vanilla PyTorch implementations

## Development Workflow

### Adding New Features
1. Implement in appropriate module under `src/neobert/`
2. Add corresponding test in `tests/` mirroring the source structure
3. Update relevant configuration dataclasses if needed
4. Ensure CPU compatibility for testing

### Modifying Model Architecture
1. Check `src/neobert/model/neobert.py` for main model class
2. Update `src/neobert/config/model.py` for configuration changes
3. Ensure backward compatibility or increment version
4. Test with both CPU and GPU configurations

### Working with Configurations
- YAML configs are in `configs/` directory
- Test configs with tiny models are in `tests/configs/`
- Use dot notation for CLI overrides
- Configuration dataclasses are in `src/neobert/config/`

### Streaming Dataset Support
The codebase supports streaming datasets for efficient handling of large data:
- Use `IterableDataset` for streaming
- Implement proper worker initialization
- Handle distributed training with proper sharding

## Important Notes

- The model uses modern architectural improvements over BERT - don't assume BERT-like behavior
- Flash Attention and xformers are optional but provide significant speedups
- All test configurations use tiny models to ensure CPU compatibility
- The configuration system is hierarchical - changes cascade through nested configs
- Streaming dataset support is critical for large-scale training