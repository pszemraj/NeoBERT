# Scripts Tests

These tests exercise functionality in the `scripts/` directory and are run separately from the main test suite since they don't fit the standard categories (config, model, training, evaluation, integration).

To run these tests:

```bash
# Run all script tests
python -m unittest discover -s tests/scripts -p "test_*.py" -v

# Run specific test
python -m unittest tests/scripts/test_smollm2_streaming.py -v
python -m unittest tests/scripts/test_wandb_step_logging.py -v
```

Note: Some tests may require specific dependencies or environment setup to pass.