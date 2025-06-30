# NeoBERT Test Suite

This comprehensive test suite validates all NeoBERT functionality, ensuring robust behavior across configuration, model, training, and evaluation components.

## Overview

The test suite verifies the configuration system and all NeoBERT functionality. Tests are designed to run on CPU-only machines using tiny model configurations.

## Test Structure

```
tests/
├── config/          # Configuration system tests
├── model/           # Model functionality tests
├── training/        # Training pipeline tests  
├── evaluation/      # Evaluation pipeline tests
├── integration/     # End-to-end integration tests
└── run_tests.py     # Test runner
```

## Key Features Tested

### 1. Configuration System (`config/`)
- ✅ YAML config loading and parsing
- ✅ CLI override system with dot notation
- ✅ Nested configuration merging
- ✅ All task-specific configs (pretraining, GLUE, contrastive)
- ✅ Type-safe dataclass validation

### 2. Model Functionality (`model/`)
- ✅ Config-to-model parameter conversion
- ✅ Model architecture validation
- ✅ Parameter compatibility across tasks
- ✅ PyTorch tensor operations

### 3. Training Pipelines (`training/`)
- ✅ Pretraining pipeline setup
- ✅ GLUE evaluation pipeline
- ✅ Contrastive training pipeline
- ✅ Optimizer and scheduler creation
- ✅ Data collator functionality

### 4. Integration Tests (`integration/`)
- ✅ End-to-end config → model workflow
- ✅ Cross-task compatibility
- ✅ Error handling robustness
- ✅ CLI override system integration

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Categories
```bash
# Config system only
python tests/run_tests.py --test-dir config

# Model functionality only  
python tests/run_tests.py --test-dir model --pattern="test_config_model_integration.py"

# Integration tests only
python tests/run_tests.py --test-dir integration
```

### Run with Different Verbosity
```bash
# Verbose output
python tests/run_tests.py --verbose

# Quiet output
python tests/run_tests.py --quiet
```

## Test Configurations

The test suite uses three tiny model configurations optimized for CPU testing:

### `test_tiny_pretrain.yaml`
- **Purpose**: Pretraining pipeline testing
- **Model**: 64 hidden, 2 layers, 2 heads, vocab_size=1000
- **Features**: MLM collator, cosine scheduler, GELU activation
- **Dataset**: Small wikibook subset
- **Optimizations**: No flash attention, no xformers dependencies

### `test_tiny_glue.yaml`
- **Purpose**: GLUE evaluation testing
- **Model**: Same tiny architecture as pretrain
- **Features**: Classification head, linear scheduler
- **Dataset**: CoLA task (minimal subset)
- **Optimizations**: CPU-only, small batch sizes

### `test_tiny_contrastive.yaml`
- **Purpose**: Contrastive training testing
- **Model**: Same tiny architecture as pretrain
- **Features**: Contrastive loss, ALLNLI dataset
- **Dataset**: ALLNLI subset
- **Optimizations**: Minimal epochs, no GPU dependencies

## CPU-Optimized Settings

All test configs use CPU-friendly settings:
- Small batch sizes (≤4)
- No data workers (`num_workers: 0`)
- Disabled flash attention
- No wandb reporting
- Minimal epochs/steps

## Skipped Tests

Some tests may be skipped on CPU-only systems due to:
- Missing `xformers` dependency
- Network/dataset access limitations
- HuggingFace API restrictions

This is expected behavior and doesn't indicate test failures.

## Bug Fixes Applied

During test development, critical bugs were identified and fixed:

1. **Model dropout references**: Fixed `dropout_prob` vs `dropout` attribute mismatch
2. **Dataset filtering**: Fixed unassigned filter operations  
3. **Duplicate assignments**: Removed redundant variable assignments

## Validation Results

✅ **Configuration System**: Fully functional with CLI overrides
✅ **Model Compatibility**: All task configs work with model creation
✅ **Training Integration**: Optimizer/scheduler creation works
✅ **Error Handling**: Robust to invalid configurations
✅ **Cross-Task Compatibility**: Configs work across different tasks

## Usage Notes

- Tests are designed for continuous integration
- All dependencies are gracefully handled with skipTest()
- Test configs maintain model architecture consistency
- Full test coverage for all major components