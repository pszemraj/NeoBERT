# Job Scripts

This directory contains shell scripts for running NeoBERT training and evaluation jobs. These scripts provide examples and templates for different types of workloads.

## Directory Structure

```
jobs/
├── README.md                 # This file
├── example_pretrain.sh      # Example pretraining job
├── example_evaluate.sh      # Example evaluation job
├── contrastive/            # Contrastive learning jobs
├── evaluation/             # Model evaluation jobs
├── glue/                   # GLUE benchmark jobs
├── mteb/                   # MTEB benchmark jobs
└── preprocess/            # Data preprocessing jobs
```

## Example Scripts

### `example_pretrain.sh`
Basic pretraining script showing different usage patterns:
- Simple config-based training
- Training with command-line overrides
- CPU testing setup

### `example_evaluate.sh`
Evaluation script for trained models:
- GLUE evaluation
- MTEB evaluation
- Custom evaluation setups

## Usage

### Running a Job Script

```bash
# Make script executable
chmod +x jobs/example_pretrain.sh

# Run the script
./jobs/example_pretrain.sh
```

### Customizing Job Scripts

1. **Copy an example script**:
   ```bash
   cp jobs/example_pretrain.sh jobs/my_pretrain.sh
   ```

2. **Edit the script** to match your requirements:
   ```bash
   nano jobs/my_pretrain.sh
   ```

3. **Make it executable and run**:
   ```bash
   chmod +x jobs/my_pretrain.sh
   ./jobs/my_pretrain.sh
   ```

## Job Script Patterns

### Basic Pretraining Job

```bash
#!/bin/bash
# Basic pretraining with config file

python scripts/pretraining/pretrain.py \
    --config configs/pretraining/pretrain_neobert.yaml \
    --trainer.output_dir ./output/my_model \
    --wandb.project my-project \
    --wandb.name my-run-name
```

### Multi-GPU Training Job

```bash
#!/bin/bash
# Multi-GPU training with accelerate

accelerate launch \
    # Use an accelerate config generated via: accelerate config
    --config_file path/to/accelerate.yaml \
    scripts/pretraining/pretrain.py \
    --config configs/pretraining/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 16 \
    --trainer.gradient_accumulation_steps 4
```

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=neobert-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

# Load modules/environment
module load python/3.9
source venv/bin/activate

# Run training
accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    scripts/pretraining/pretrain.py \
    --config configs/pretraining/pretrain_neobert.yaml \
    --trainer.output_dir $SLURM_TMPDIR/output
```

### Evaluation Job

```bash
#!/bin/bash
# Evaluate a trained model on GLUE

CONFIG_DIR="configs/glue"

# Run the full GLUE suite
bash scripts/evaluation/glue/run_all_glue.sh "$CONFIG_DIR"

# Or run a single task
python scripts/evaluation/run_glue.py --config "$CONFIG_DIR/cola.yaml"
```

## Directory-Specific Jobs

### `contrastive/`
Scripts for contrastive learning and sentence embedding training:
- Data preprocessing for contrastive datasets
- Contrastive model training
- Embedding evaluation

### `evaluation/`
Model evaluation scripts:
- Perplexity evaluation
- Downstream task evaluation
- Benchmark comparisons

### `glue/`
GLUE benchmark evaluation:
- Individual task evaluation
- Full GLUE suite
- Task-specific fine-tuning

### `mteb/`
MTEB (Massive Text Embedding Benchmark) evaluation:
- Sentence embedding evaluation
- Retrieval tasks
- Classification tasks

### `preprocess/`
Data preprocessing jobs:
- Dataset tokenization
- Data filtering and cleaning
- Format conversion

## Best Practices

### 1. Environment Setup
```bash
# Always set up environment at the beginning
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=my-project
```

### 2. Error Handling
```bash
# Exit on any error
set -e

# Capture exit status
trap 'echo "Job failed with exit code $?"' ERR
```

### 3. Logging
```bash
# Create log directory
mkdir -p logs

# Redirect output
python script.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

### 4. Resource Management
```bash
# Check GPU availability
nvidia-smi

# Monitor GPU usage during training
python script.py &
PID=$!
while kill -0 $PID 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    sleep 30
done
```

### 5. Checkpoint Management
```bash
# Set checkpoint directory
CHECKPOINT_DIR="./checkpoints/$(date +%Y%m%d_%H%M%S)"
mkdir -p $CHECKPOINT_DIR

# Save config alongside checkpoints
cp configs/pretraining/pretrain_neobert.yaml $CHECKPOINT_DIR/config.yaml
```

## Common Use Cases

### Quick CPU Test
```bash
#!/bin/bash
# Quick test on CPU with tiny model

python scripts/pretraining/pretrain.py \
    --config tests/configs/pretraining/test_tiny_pretrain.yaml \
    --trainer.max_steps 10 \
    --trainer.save_steps 5 \
    --wandb.mode disabled \
    --debug
```

### Resume Training
```bash
#!/bin/bash
# Resume training from checkpoint

OUTPUT_DIR="./output/pretrain"

python scripts/pretraining/pretrain.py \
    --config configs/pretraining/pretrain_neobert.yaml \
    --trainer.output_dir $OUTPUT_DIR \
    --trainer.resume_from_checkpoint true
```

Notes:
- `trainer.resume_from_checkpoint` is treated as a flag and resumes from the latest `output_dir/checkpoints/`.
- To resume a specific step, remove newer checkpoint directories under `output_dir/checkpoints/`.

### Hyperparameter Sweep
```bash
#!/bin/bash
# Simple hyperparameter sweep

for lr in 1e-4 2e-4 5e-4; do
    for batch_size in 16 32 64; do
        python scripts/pretraining/pretrain.py \
            --config configs/pretraining/pretrain_neobert.yaml \
            --optimizer.lr $lr \
            --trainer.per_device_train_batch_size $batch_size \
            --trainer.output_dir ./output/sweep_lr${lr}_bs${batch_size} \
            --wandb.name "lr${lr}_bs${batch_size}"
    done
done
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **CUDA Errors**: Check GPU availability and driver compatibility
3. **Config Errors**: Validate config with `--debug` flag
4. **Permission Errors**: Ensure script is executable (`chmod +x`)

### Debug Mode
```bash
# Run with debug for verbose output
python script.py --config config.yaml --debug
```

### Environment Check
```bash
# Check environment before running
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```
