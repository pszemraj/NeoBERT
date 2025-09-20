# GLUE Module

This module implements GLUE (General Language Understanding Evaluation) benchmark training and evaluation for NeoBERT models.

## Module Structure

- **`train.py`** - Main GLUE trainer implementation
- **`process.py`** - Data processing utilities for GLUE tasks
- **`__init__.py`** - Module initialization

## Key Features

### Automatic Pretrained Model Loading

The GLUE trainer supports loading pretrained checkpoints from:
- DeepSpeed checkpoints (with `zero_to_fp32` conversion)
- Standard PyTorch checkpoints
- HuggingFace Hub models

```python
# Example: Loading from local checkpoint
if not from_hub and pretrained_checkpoint is not None:
    checkpoint_path = os.path.join(
        cfg.model.pretrained_checkpoint_dir, 
        str(pretrained_checkpoint)
    )
    # Handles both DeepSpeed and standard checkpoints
```

### Safety Checks

The trainer includes safety mechanisms to prevent training with random weights:

1. **Required pretrained config**: GLUE evaluation requires `pretrained_config_path`
2. **Weight verification**: Checks that loaded weights differ from initialization
3. **Explicit random weights flag**: Future support for `allow_random_weights` flag

### Flash Attention Handling

⚠️ **Important**: Flash attention is automatically disabled for GLUE tasks due to memory alignment requirements.

**Technical Details**:
- xformers' `memory_efficient_attention` requires sequence lengths aligned to multiples of 8
- GLUE tasks use variable-length sequences with dynamic batching
- Misaligned sequences cause incorrect attention computation (~50% accuracy)

**Solution**:
```python
# Always use eager attention for GLUE
if hasattr(cfg.model, 'flash_attention') and cfg.model.flash_attention:
    logger.warning(
        "Flash attention is not supported for GLUE evaluation "
        "due to memory alignment issues with variable-length sequences. "
        "Using eager attention instead."
    )
flash_attention = False
```

### Configuration Preservation

GLUE tasks require special fields not in the base model config:
- `pretrained_checkpoint`
- `pretrained_checkpoint_dir`
- `pretrained_config_path`

These are preserved via `_raw_model_dict` in the config system.

## Usage

### From Python

```python
from neobert.glue.train import trainer
from neobert.config import ConfigLoader

# Load configuration
config = ConfigLoader.load("configs/eval/glue_sst2.yaml")

# Run training
trainer(config)
```

### From Command Line

```bash
python scripts/evaluation/run_glue.py --config configs/eval/glue_sst2.yaml
```

## Supported Tasks

All 9 GLUE tasks are supported:

| Task | Type | Classes | Metric |
|------|------|---------|--------|
| CoLA | Acceptability | 2 | Matthews Correlation |
| SST-2 | Sentiment | 2 | Accuracy |
| MRPC | Paraphrase | 2 | F1/Accuracy |
| STS-B | Similarity | Regression | Pearson/Spearman |
| QQP | Question Pairs | 2 | F1/Accuracy |
| MNLI | NLI | 3 | Accuracy |
| QNLI | Question NLI | 2 | Accuracy |
| RTE | Entailment | 2 | Accuracy |
| WNLI | Winograd NLI | 2 | Accuracy |

## Training Details

### Hyperparameters

Default settings (can be overridden in configs):
- Learning rate: 2e-5
- Batch size: 32
- Gradient accumulation: 4 (effective batch size: 128)
- Warmup: 10% of training steps
- Weight decay: 0.01
- Max gradient norm: 1.0

### Evaluation Strategy

Supports both:
- **Epoch-based**: Evaluate at epoch boundaries
- **Step-based**: Evaluate every N steps

```yaml
trainer:
  eval_strategy: "epoch"  # or "steps"
  eval_steps: 500  # if using "steps"
```

### Early Stopping

Configurable early stopping based on evaluation metric:

```yaml
trainer:
  early_stopping_patience: 10
  early_stopping_threshold: 0.001
  metric_for_best_model: "accuracy"
  greater_is_better: true
```

## Output Structure

Results are saved in the specified `output_dir`:

```
glue_outputs/
├── sst2/
│   ├── checkpoint-1054/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── training_args.bin
│   ├── all_results_step_527.json
│   ├── all_results_step_1054.json
│   └── all_results_step_1581.json
├── cola/
│   └── ...
```

## Logging

- **Console**: Training progress with tqdm
- **WandB**: Detailed metrics (project: `neobert-evals`)
- **JSON**: Evaluation results at each checkpoint

## Known Issues

- **Flash Attention Compatibility**: Flash attention is disabled for GLUE evaluation due to memory alignment issues with head_dim=64
- **Workaround**: The trainer automatically sets `flash_attention=False` when loading models
- **Future Plans**: Upgrade to Flash Attention 3 which supports head_dim=64

## Development

When modifying the GLUE trainer:

1. Ensure backward compatibility with existing configs
2. Test with at least 3 different GLUE tasks
3. Verify checkpoint loading works correctly
4. Check that metrics are computed properly
5. Run for at least 60 seconds to ensure stability

## Related

- [`scripts/evaluation/`](../../../scripts/evaluation/) - Evaluation scripts
- [`configs/glue/`](../../../configs/glue/) - GLUE task configurations
- [`jobs/run_all_glue.sh`](../../../jobs/run_all_glue.sh) - Batch evaluation script