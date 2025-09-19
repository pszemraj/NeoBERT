# Quick Start Guide

Get NeoBERT running in 5 minutes.

## Installation

```bash
# Clone and install
git clone https://github.com/pszemraj/NeoBERT.git
cd NeoBERT
pip install -e .

# Optional: Install Flash Attention for faster training (requires CUDA)
pip install flash-attn --no-build-isolation
```

## Test Your Setup

```bash
# Quick test with tiny model (5 minutes, CPU-friendly)
python scripts/pretraining/pretrain.py \
    --config tests/configs/pretraining/test_tiny_pretrain.yaml

# Run tests
python tests/run_tests.py
```

## Basic Usage

### Pretrain a Model

```bash
# Full pretraining (requires GPU)
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml

# Override settings via CLI
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --trainer.max_steps 10000 \
    --trainer.per_device_train_batch_size 16 \
    --optimizer.lr 2e-4
```

### Evaluate on GLUE

```bash
# Single GLUE task
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml

# All GLUE tasks
bash scripts/run_full_glue.sh

# Summarize results
python scripts/summarize_glue.py outputs/glue/neobert-100m
```

### Use Custom Tokenizer

```bash
# Just specify the tokenizer - dataset tokenization is automatic
python scripts/pretraining/pretrain.py \
    --config configs/pretrain_neobert.yaml \
    --tokenizer.name "your-tokenizer" \
    --model.vocab_size 32000  # Match your tokenizer
```

## Key Commands

| Task | Command |
|------|---------|
| Pretrain | `python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml` |
| GLUE eval | `python scripts/evaluation/run_glue.py --config configs/glue/{task}.yaml` |
| Run tests | `python tests/run_tests.py` |
| Summarize GLUE | `python scripts/summarize_glue.py {results_path}` |

## Configuration

All settings use YAML configs with CLI overrides:

```bash
# Base config from YAML
--config configs/pretrain_neobert.yaml

# Override with dot notation
--model.hidden_size 1024
--trainer.batch_size 64
--optimizer.lr 1e-4
```

See [Configuration Guide](configuration.md) for details.

## Common Settings

**Model sizes:**
- Tiny (test): 2 layers, 128 hidden
- Small: 6 layers, 512 hidden  
- Base: 12 layers, 768 hidden
- Large: 24 layers, 1024 hidden

**Training tips:**
- Use `bf16` mixed precision (not fp16)
- Enable Flash Attention if available
- Start with batch size 32, lr 1e-4 for pretraining
- Use batch size 32, lr 2e-5 for GLUE

## Next Steps

- [Training Guide](training.md) - Detailed pretraining instructions
- [Evaluation Guide](evaluation.md) - GLUE and MTEB benchmarks
- [Configuration](configuration.md) - Config system details