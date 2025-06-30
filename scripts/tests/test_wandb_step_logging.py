#!/usr/bin/env python3
"""Test script to verify wandb step logging works correctly."""

import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.neobert.pretraining.metrics import Metrics


def test_metrics_log_with_step():
    """Test that metrics.log passes the step parameter to accelerator.log."""

    # Create a mock accelerator
    mock_accelerator = MagicMock()
    mock_accelerator.device = "cpu"
    mock_accelerator.reduce = MagicMock(side_effect=lambda x, reduction: x)

    # Create metrics instance and populate it
    metrics = Metrics()
    metrics["train/steps"] = 10
    metrics["train/samples"] = 0
    metrics["train/tokens"] = 0
    metrics["train/masked_tokens"] = 0
    metrics["train/local_samples"] = 32
    metrics["train/local_tokens"] = 512 * 32
    metrics["train/local_num_pred"] = 100
    metrics["train/local_sum_loss"] = 250.0
    metrics["train/local_num_correct"] = 85
    metrics["train/grad_norm"] = 1.5
    metrics["train/weight_norm"] = 10.2
    metrics["train/learning_rate"] = 1e-4

    # Call log method
    metrics.log(mock_accelerator)

    # Check that accelerator.log was called with step parameter
    mock_accelerator.log.assert_called_once()
    call_args = mock_accelerator.log.call_args

    # Verify step parameter was passed
    assert "step" in call_args.kwargs, "Step parameter not passed to accelerator.log"
    assert call_args.kwargs["step"] == 10, (
        f"Expected step=10, got step={call_args.kwargs['step']}"
    )

    # Verify metrics were logged
    logged_metrics = call_args.args[0]
    assert "train/loss" in logged_metrics
    assert "train/perplexity" in logged_metrics
    assert "train/accuracy" in logged_metrics
    assert "train/steps" in logged_metrics

    print(
        "✓ Test passed: metrics.log correctly passes step parameter to accelerator.log"
    )
    print(f"  - Step value: {call_args.kwargs['step']}")
    print(f"  - Logged metrics: {list(logged_metrics.keys())}")


if __name__ == "__main__":
    test_metrics_log_with_step()
