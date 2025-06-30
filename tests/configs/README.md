# Test Configuration Files

This directory contains test configuration files organized by training pipeline type. These configs are optimized for quick testing and development without requiring GPUs.

## Directory Structure

- `pretraining/` - Test configs for pretraining pipeline
  - `test_tiny_pretrain.yaml` - Minimal CPU-friendly pretraining config
  - `test_streaming*.yaml` - Configs for testing streaming dataset support
  - `test_gpu*.yaml` - GPU-optimized test configs
  - `test_pretokenized.yaml` - Config for pre-tokenized datasets

- `evaluation/` - Test configs for evaluation pipelines
  - `test_tiny_glue.yaml` - Minimal GLUE evaluation config

- `contrastive/` - Test configs for contrastive training
  - `test_tiny_contrastive.yaml` - Minimal contrastive training config

## Usage

These test configs are designed for:
1. Quick functionality testing during development
2. CI/CD pipeline testing
3. Debugging without requiring large compute resources

Example:
```bash
# Test pretraining pipeline
python scripts/pretraining/pretrain.py --config tests/configs/pretraining/test_tiny_pretrain.yaml

# Test GLUE evaluation
python scripts/evaluation/run_glue.py --config tests/configs/evaluation/test_tiny_glue.yaml

# Test contrastive training
python scripts/contrastive/finetune.py --config tests/configs/contrastive/test_tiny_contrastive.yaml
```

## Key Features

- **CPU-friendly**: Small model sizes and batch sizes
- **Fast execution**: Limited training steps for quick testing
- **No external dependencies**: Use small synthetic or cached datasets
- **Comprehensive coverage**: Test all major code paths