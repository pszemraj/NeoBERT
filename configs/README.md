# Configuration Files

This directory contains production YAML configuration files for the NeoBERT training and evaluation pipeline. Test configurations are located in `tests/configs/`.

## Production Configurations

### Pretraining
- **`pretrain_neobert.yaml`** - Standard pretraining configuration (768 hidden, 12 layers)
- **`pretrain_streaming.yaml`** - Streaming dataset configuration for large-scale training
- **`pretrain_gpu_small.yaml`** - Small model config for GPU testing with SwiGLU activation
- **`pretrain_smollm2_custom_tokenizer.yaml`** - Full 250M model on SmolLM2 with 32k tokenizer, 1024 context

### Fine-tuning & Evaluation
- **`evaluate_neobert.yaml`** - GLUE evaluation configuration
- **`contrastive_neobert.yaml`** - Contrastive learning configuration (SimCSE-style)

### Custom Tokenizer
- **`train_small_custom_tokenizer.yaml`** - Example config for using custom tokenizers
- **`pretrain_smollm2_custom_tokenizer.yaml`** - Production config with BEE-spoke 32k tokenizer

## Test Configurations

Test configurations with tiny models are in `tests/configs/`:
- `tests/configs/pretraining/test_tiny_pretrain.yaml` - Tiny model for CPU testing
- `tests/configs/evaluation/test_tiny_glue.yaml` - Tiny model for GLUE testing
- `tests/configs/contrastive/test_tiny_contrastive.yaml` - Tiny model for contrastive learning

## Configuration Structure

Each configuration file follows this hierarchical structure:

```yaml
task: pretraining  # Task type: pretraining, glue, contrastive, evaluation

model:
  # Model architecture parameters
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  # ... other model parameters

dataset:
  # Dataset configuration
  name: refinedweb
  path: ""
  # ... dataset parameters

tokenizer:
  # Tokenizer configuration
  name: bert-base-uncased
  max_length: 512
  # ... tokenizer parameters

optimizer:
  # Optimizer configuration
  name: adamw
  lr: 1.0e-04
  # ... optimizer parameters

scheduler:
  # Learning rate scheduler
  name: cosine
  warmup_steps: 10000
  # ... scheduler parameters

trainer:
  # Training configuration
  per_device_train_batch_size: 16
  max_steps: 1000000
  # ... trainer parameters

# Additional sections for specific tasks
datacollator:  # For pretraining
wandb:         # For experiment tracking
```

## Usage

### Basic Usage

```bash
# Use a configuration file directly
python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml
```

### Command-Line Overrides

You can override any configuration parameter using dot notation:

```bash
# Override specific parameters
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 32 \
    --optimizer.lr 2e-4 \
    --wandb.project my-project
```

### Configuration Validation

The configuration system includes validation for:
- Required fields for each task type
- Type checking (int, float, bool, str)
- Value constraints (e.g., positive integers, valid choices)
- Model architecture constraints (e.g., hidden_size divisible by num_attention_heads)

## Creating Custom Configurations

1. **Start with a base config**: Copy one of the existing configurations
2. **Modify parameters**: Update the values for your specific use case
3. **Validate**: Run with `--debug` flag to check configuration validity
4. **Test**: Use test configs for initial validation before full training

### Example: Custom Pretraining Config

```yaml
# custom_pretrain.yaml
task: pretraining

model:
  hidden_size: 512      # Smaller model
  num_hidden_layers: 8
  num_attention_heads: 8
  
trainer:
  per_device_train_batch_size: 64  # Larger batch size
  max_steps: 500000                # Fewer steps

wandb:
  project: my-custom-bert
  name: small-bert-experiment
```

## Configuration Tips

1. **Start Small**: Use test configs to validate your setup before full training
2. **Use Overrides**: Command-line overrides are great for experimentation
3. **Version Control**: Keep configurations in git to track experiments
4. **Documentation**: Add comments to your custom configs for clarity
5. **Validation**: Always check configs with validation before long training runs

## Environment Variables

Some configurations can use environment variables:

```yaml
wandb:
  entity: ${WANDB_ENTITY}  # Will use env var WANDB_ENTITY
  
dataset:
  cache_dir: ${HF_DATASETS_CACHE}  # Use HuggingFace cache dir
```

## Common Patterns

### CPU Testing
```yaml
# Use tiny models for CPU testing
model:
  hidden_size: 64
  num_hidden_layers: 2
  hidden_act: GELU  # Instead of SwiGLU to avoid xformers
  flash_attention: false

trainer:
  per_device_train_batch_size: 2
  max_steps: 10
```

### Multi-GPU Training
```yaml
trainer:
  per_device_train_batch_size: 16  # Per device
  gradient_accumulation_steps: 4   # Effective batch = 16 * 4 * num_gpus

mixed_precision: bf16
accelerate_config_file: configs/accelerate_ddp.yaml
```

### Memory Optimization
```yaml
trainer:
  gradient_checkpointing: true  # Trade compute for memory
  per_device_train_batch_size: 8  # Smaller batches
  gradient_accumulation_steps: 8  # Maintain effective batch size

model:
  flash_attention: true  # Reduce memory usage
```