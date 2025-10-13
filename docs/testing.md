# Testing Guide

This guide covers running and writing tests for NeoBERT.

> [!NOTE]
> See [/tests/README.md](/tests/README.md) for test suite structure and [/tests/configs/README.md](/tests/configs/README.md) for test configurations.

## Running Tests

### Run All Tests

```bash
# Run the complete test suite
python tests/run_tests.py

# With verbose output
python tests/run_tests.py --verbose

# Quiet mode (only failures)
python tests/run_tests.py --quiet
```

### Run Specific Test Suites

```bash
# Configuration tests
python tests/run_tests.py --test-dir config

# Model tests
python tests/run_tests.py --test-dir model

# Evaluation tests
python tests/run_tests.py --test-dir evaluation

# Integration tests
python tests/run_tests.py --test-dir integration

# Training tests
python tests/run_tests.py --test-dir training
```

### Run Specific Test Files

```bash
# Using pytest directly
pytest tests/model/test_model_forward.py -v

# Run specific test
pytest tests/test_config.py::TestConfig::test_load_from_yaml -v

# Run with pattern matching
python tests/run_tests.py --pattern "test_config*.py"
```

## Test Organization

```
tests/
├── __init__.py
├── run_tests.py              # Test runner script
├── test_actual_functionality.py  # End-to-end tests
├── config/
│   └── test_config_system.py # Configuration tests
├── model/
│   ├── test_model_forward.py # Model forward pass tests
│   └── test_config_model_integration.py
├── evaluation/
│   └── test_glue_pipeline.py # GLUE evaluation tests
├── integration/
│   └── test_end_to_end.py    # Integration tests
└── training/
    ├── test_pretrain_pipeline.py
    └── test_contrastive_pipeline.py
```

## Writing Tests

### Basic Test Structure

```python
import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neobert.config import ConfigLoader

class TestConfig(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_config_path = Path(__file__).parent / "test_config.yaml"

    def test_config_loading(self):
        """Test configuration loading."""
        config = ConfigLoader.load(str(self.test_config_path))
        self.assertIsNotNone(config)
        self.assertEqual(config.model.hidden_size, 768)

    def tearDown(self):
        """Clean up after tests."""
        # Cleanup code here
        pass

if __name__ == "__main__":
    unittest.main()
```

### Testing Model Components

```python
class TestModelComponents(unittest.TestCase):
    def test_attention_mechanism(self):
        """Test attention computation."""
        from neobert.model import MultiheadAttention

        # Create attention layer
        attn = MultiheadAttention(
            embed_dim=64,
            num_heads=4
        )

        # Test input
        x = torch.randn(2, 10, 64)  # [batch, seq_len, dim]

        # Forward pass
        output = attn(x)

        # Assertions
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
```

### Testing Training Pipeline

```python
class TestTrainingPipeline(unittest.TestCase):
    def test_loss_computation(self):
        """Test MLM loss computation."""
        from neobert.model import NeoBERTLMHead
        from neobert.config import ModelConfig

        # Small model for testing
        config = ModelConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            vocab_size=100
        )

        model = NeoBERTLMHead(config)

        # Mock data
        input_ids = torch.randint(0, 100, (2, 10))
        labels = input_ids.clone()
        labels[labels == 0] = -100  # Ignore padding

        # Compute loss
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Assertions
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)
```

## Test Fixtures and Utilities

### Creating Test Configurations

```python
# tests/fixtures/configs.py
def create_test_config():
    """Create a minimal test configuration."""
    return {
        "model": {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 1000,
            "hidden_act": "gelu",  # CPU-friendly
            "flash_attention": False
        },
        "trainer": {
            "max_steps": 10,
            "per_device_train_batch_size": 2,
            "use_cpu": True
        }
    }
```

### Mock Data Generation

```python
# tests/fixtures/data.py
def create_mock_dataset(size=100, seq_length=128, vocab_size=1000):
    """Create mock tokenized dataset."""
    return {
        "input_ids": torch.randint(0, vocab_size, (size, seq_length)),
        "attention_mask": torch.ones(size, seq_length),
        "labels": torch.randint(0, vocab_size, (size, seq_length))
    }
```

### Test Decorators

```python
import pytest
import torch

def requires_gpu(test_func):
    """Skip test if GPU is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Test requires GPU"
    )(test_func)

def requires_xformers(test_func):
    """Skip test if xformers is not installed."""
    try:
        import xformers
        return test_func
    except ImportError:
        return pytest.mark.skip(reason="xformers not installed")(test_func)
```

## Testing Best Practices

### 1. Test Isolation

```python
class TestModelCreation(unittest.TestCase):
    def setUp(self):
        """Create fresh model for each test."""
        self.config = create_test_config()
        self.model = None

    def tearDown(self):
        """Clean up model and free memory."""
        if self.model:
            del self.model
        torch.cuda.empty_cache()
```

### 2. Parameterized Tests

```python
import pytest

class TestActivationFunctions:
    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
    def test_activation_forward(self, activation):
        """Test different activation functions."""
        config = ModelConfig(hidden_act=activation)
        model = NeoBERT(config)

        # Test forward pass
        x = torch.randn(2, 10, config.hidden_size)
        output = model(x)

        assert output.shape == x.shape
```

### 3. Testing Edge Cases

```python
def test_empty_input():
    """Test model with empty input."""
    model = create_test_model()

    # Empty batch
    input_ids = torch.tensor([], dtype=torch.long).reshape(0, 10)

    # Should not crash
    with torch.no_grad():
        output = model(input_ids)

    assert output.shape[0] == 0

def test_single_token():
    """Test model with single token sequences."""
    model = create_test_model()
    input_ids = torch.randint(0, 100, (2, 1))

    output = model(input_ids)
    assert output.shape == (2, 1, model.config.hidden_size)
```

### 4. Performance Tests

```python
import time

def test_inference_speed():
    """Test model inference speed."""
    model = create_test_model()
    model.eval()

    # Warm up
    dummy_input = torch.randint(0, 100, (8, 128))
    with torch.no_grad():
        _ = model(dummy_input)

    # Time inference
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    elapsed = time.time() - start
    throughput = (100 * 8) / elapsed  # samples/second

    assert throughput > 100  # Minimum expected throughput
```

## Debugging Failed Tests

### 1. Verbose Output

```bash
# Run with detailed output
pytest tests/model/test_model_forward.py -vvs

# Show local variables on failure
pytest tests/model/test_model_forward.py --showlocals
```

### 2. Interactive Debugging

```python
def test_complex_scenario():
    """Test with debugging."""
    import pdb

    model = create_model()
    data = create_test_data()

    # Set breakpoint
    pdb.set_trace()

    output = model(data)
    assert output is not None
```

### 3. Test-Specific Logging

```python
import logging

def test_with_logging():
    """Test with detailed logging."""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    model = create_model()
    logger.debug(f"Model config: {model.config}")

    data = create_test_data()
    logger.debug(f"Input shape: {data.shape}")

    output = model(data)
    logger.debug(f"Output shape: {output.shape}")
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        python tests/run_tests.py --verbose

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: tests
        entry: python tests/run_tests.py --quiet
        language: system
        pass_filenames: false
        always_run: true
```

## Test Coverage

### Generate Coverage Report

```bash
# Run with coverage
pytest --cov=neobert tests/

# Generate HTML report
pytest --cov=neobert --cov-report=html tests/

# View report
open htmlcov/index.html
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = neobert
omit =
    */tests/*
    */migrations/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Common Issues

### Import Errors

```python
# Ensure src is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### CUDA/CPU Compatibility

```python
def get_device():
    """Get appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# In tests
device = get_device()
model = model.to(device)
data = data.to(device)
```

### Reproducibility

```python
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

## Next Steps

- Review the [Configuration Guide](/docs/configuration.md) for advanced settings
- See the [Training Guide](/docs/training.md) for model training
- Check the [Architecture Documentation](/docs/architecture.md) for technical details
