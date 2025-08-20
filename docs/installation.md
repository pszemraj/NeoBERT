# Installation Guide

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- GPU with BF16 support (NVIDIA Ampere architecture or newer):
  - RTX 30xx series (3060, 3070, 3080, 3090)
  - RTX 40xx series (4060, 4070, 4080, 4090)
  - A100, A6000, H100
  - Or any GPU with compute capability ≥ 8.0

## Basic Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/pszemraj/NeoBERT.git
cd NeoBERT

# Install in development mode
pip install -e .
```

### From PyPI (When Available)

```bash
pip install neobert
```

## Optional Dependencies

### Flash Attention (Recommended for GPU)

Flash Attention significantly speeds up training and reduces memory usage:

```bash
# Install flash-attn (requires CUDA)
pip install flash-attn==2.7.3 --no-build-isolation
```

### XFormers (Recommended for SwiGLU)

Provides optimized SwiGLU activation (default in NeoBERT):

```bash
# Compatible version
pip install xformers==0.0.28.post3
```

Note: NeoBERT will use a native PyTorch fallback if xformers is not available, but performance will be reduced.

### DeepSpeed (For Large-Scale Training)

```bash
pip install deepspeed
```

### Development Dependencies

```bash
# Linting and formatting
pip install ruff isort black

# Testing
pip install pytest pytest-cov

# Documentation
pip install sphinx sphinx-rtd-theme
```

## Verification

Verify your installation:

```bash
# Run tests
python tests/run_tests.py --test-dir config

# Check if GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test model creation (requires xformers for SwiGLU)
python -c "
from neobert.model.model import NeoBERT, NeoBERTConfig
try:
    # Default config uses SwiGLU activation (requires xformers)
    config = NeoBERTConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=4)
    model = NeoBERT(config)
    print(f'Model created with SwiGLU: {sum(p.numel() for p in model.parameters())} parameters')
except ImportError:
    # Fallback to GELU if xformers not available
    config = NeoBERTConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=4, hidden_act='gelu')
    model = NeoBERT(config)
    print(f'Model created with GELU (xformers not available): {sum(p.numel() for p in model.parameters())} parameters')
"
```

## Platform-Specific Notes

### Linux (Recommended)
- All features fully supported
- Best performance with CUDA 11.8 or 12.1

### macOS
- CPU training only (no CUDA)
- Use `model.hidden_act="gelu"` (no xformers)
- Set `trainer.use_cpu=true`

### Windows
- Use WSL2 for best compatibility
- Native Windows may have path issues

### Google Colab
```python
# Install in Colab
!git clone https://github.com/pszemraj/NeoBERT.git
%cd NeoBERT
!pip install -e .
!pip install flash-attn --no-build-isolation
```

## Troubleshooting

### ImportError: xformers
```bash
# Use GELU activation instead of SwiGLU
--model.hidden_act gelu
```

### CUDA Out of Memory
```bash
# Reduce batch size
--trainer.per_device_train_batch_size 4

# Enable gradient checkpointing
--trainer.gradient_checkpointing true

# Use smaller model
--model.hidden_size 512 --model.num_hidden_layers 6
```

### Tokenizer Issues
```bash
# Ensure tokenizers is installed
pip install tokenizers transformers>=4.30.0
```

### Dataset Loading Errors
```bash
# Install datasets library
pip install datasets>=2.14.0
```

## Environment Setup

### Using Conda

```bash
# Create new environment
conda create -n neobert python=3.10
conda activate neobert

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install NeoBERT
cd NeoBERT
pip install -e .
```

### Using venv

```bash
# Create virtual environment
python -m venv neobert_env
source neobert_env/bin/activate  # On Windows: neobert_env\Scripts\activate

# Install dependencies
pip install -e .
```

## Next Steps

- Follow the [Quick Start Guide](quickstart.md)
- Configure your first model with the [Configuration Guide](configuration.md)
- Start training with the [Training Guide](training.md)