# NeoBERT Configuration System

NeoBERT uses a simple, intuitive configuration system based on YAML files and command-line arguments.

## Configuration Overview

The new system uses:
1. **Dataclasses** for type-safe configuration structures
2. **YAML files** for base configurations  
3. **Command-line arguments** for overrides
4. **Hierarchical structure** similar to Axolotl

## Usage Examples

### Basic Usage

```bash
# Use a config file
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml

# Override specific values
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 32 \
    --optimizer.lr 2e-4
```

### Configuration Files

Two reference configurations are provided:
- `configs/pretrain_neobert.yaml` - For pretraining
- `configs/evaluate_neobert.yaml` - For evaluation (GLUE/MTEB)

### Command-Line Override Syntax

Use dot notation to override nested values:
```bash
--model.hidden_size 1024
--trainer.max_steps 500000
--optimizer.lr 1e-4
--wandb.project my-project
```

### Available Scripts

- `scripts/pretraining/pretrain.py` - Pretraining script
- `scripts/evaluation/run_glue.py` - GLUE evaluation
- `scripts/evaluation/run_mteb_new.py` - MTEB evaluation

### Example Job Scripts

See `jobs/example_pretrain.sh` and `jobs/example_evaluate.sh` for complete examples.

## Configuration Structure

The main configuration classes:
- `ModelConfig` - Model architecture settings
- `DatasetConfig` - Dataset and data loading
- `TokenizerConfig` - Tokenizer settings
- `OptimizerConfig` - Optimizer parameters
- `SchedulerConfig` - Learning rate scheduler
- `TrainerConfig` - Training loop settings
- `DataCollatorConfig` - Data collation (MLM)
- `WandbConfig` - Weights & Biases logging

## Implementation Details

The configuration system provides:
1. Type-safe dataclass configurations
2. Single hierarchical YAML files for each task
3. Simple argparse-based CLI overrides
4. Direct access to configuration as dataclass attributes

## Testing

Run the configuration tests:
```bash
python tests/test_config.py
```