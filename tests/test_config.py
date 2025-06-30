#!/usr/bin/env python3
"""Test the new configuration system"""

import sys
import tempfile

from neobert.config import (
    Config,
    ConfigLoader,
    create_argument_parser,
    parse_args_to_dict,
)


def test_config_creation():
    """Test creating a default config"""
    print("Test 1: Creating default config...")
    config = Config()
    assert config.model.hidden_size == 768
    assert config.task == "pretraining"
    assert config.trainer.per_device_train_batch_size == 16
    print("✓ Default config created successfully")


def test_yaml_loading():
    """Test loading config from YAML"""
    print("\nTest 2: Loading config from YAML...")
    config = ConfigLoader.load("configs/pretrain_neobert.yaml")
    assert config.model.rope
    assert config.optimizer.name == "adamw"
    assert config.trainer.max_steps == 1000000
    print("✓ YAML config loaded successfully")


def test_yaml_override():
    """Test overriding YAML config"""
    print("\nTest 3: Testing config overrides...")
    overrides = {
        "model": {"hidden_size": 1024},
        "trainer": {"max_steps": 500},
        "optimizer": {"lr": 1e-3},
    }
    config = ConfigLoader.load("configs/pretrain_neobert.yaml", overrides)
    assert config.model.hidden_size == 1024
    assert config.trainer.max_steps == 500
    assert config.optimizer.lr == 1e-3
    print("✓ Config overrides work correctly")


def test_save_load():
    """Test saving and loading config"""
    print("\nTest 4: Testing save/load roundtrip...")
    config = Config()
    config.model.hidden_size = 512
    config.task = "glue"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        ConfigLoader.save(config, f.name)
        loaded_config = ConfigLoader.load(f.name)

    assert loaded_config.model.hidden_size == 512
    assert loaded_config.task == "glue"
    print("✓ Save/load roundtrip successful")


def test_cli_parsing():
    """Test CLI argument parsing"""
    print("\nTest 5: Testing CLI argument parsing...")
    parser = create_argument_parser()

    # Simulate command line args
    test_args = [
        "--config",
        "configs/pretrain_neobert.yaml",
        "--model.hidden_size",
        "256",
        "--trainer.per_device_train_batch_size",
        "8",
        "--optimizer.lr",
        "5e-5",
        "--debug",
    ]

    args = parser.parse_args(test_args)
    arg_dict = parse_args_to_dict(args)

    assert arg_dict["model"]["hidden_size"] == 256
    assert arg_dict["trainer"]["per_device_train_batch_size"] == 8
    assert arg_dict["optimizer"]["lr"] == 5e-5
    assert arg_dict["debug"]
    print("✓ CLI parsing works correctly")


def test_nested_merge():
    """Test nested config merging"""
    print("\nTest 6: Testing nested config merging...")
    base = {
        "model": {"hidden_size": 768, "num_layers": 12},
        "trainer": {"max_steps": 1000},
    }
    override = {
        "model": {"hidden_size": 1024},  # Override one field
        "trainer": {"save_steps": 100},  # Add new field
    }

    merged = ConfigLoader.merge_configs(base, override)
    assert merged["model"]["hidden_size"] == 1024
    assert merged["model"]["num_layers"] == 12  # Preserved
    assert merged["trainer"]["max_steps"] == 1000  # Preserved
    assert merged["trainer"]["save_steps"] == 100  # Added
    print("✓ Nested merging works correctly")


def main():
    print("Running configuration system tests...\n")

    try:
        test_config_creation()
        test_yaml_loading()
        test_yaml_override()
        test_save_load()
        test_cli_parsing()
        test_nested_merge()

        print("\n✅ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
