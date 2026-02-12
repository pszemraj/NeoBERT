#!/usr/bin/env python3
"""Test the new configuration system functionality."""

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.config import (
    Config,
    ConfigLoader,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainerConfig,
    load_config_from_args,
    round_up_to_multiple,
)


class TestConfigSystem(unittest.TestCase):
    """Test configuration system functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config_dir = Path(__file__).parent / "configs"

    def test_default_config_creation(self):
        """Test creating default config objects."""
        config = Config()

        # Check that all sub-configs are properly initialized
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.dataset, DatasetConfig)
        self.assertIsInstance(config.tokenizer, TokenizerConfig)
        self.assertIsInstance(config.trainer, TrainerConfig)
        self.assertIsInstance(config.optimizer, OptimizerConfig)
        self.assertIsInstance(config.scheduler, SchedulerConfig)

        # Check some default values
        self.assertEqual(config.model.hidden_size, 768)
        self.assertEqual(config.model.num_hidden_layers, 12)
        self.assertEqual(config.trainer.per_device_train_batch_size, 16)
        self.assertFalse(config.trainer.log_train_accuracy)
        self.assertFalse(config.trainer.torch_compile)
        self.assertEqual(config.trainer.torch_compile_backend, "inductor")
        self.assertTrue(config.trainer.enforce_full_packed_batches)
        self.assertIsNone(config.trainer.max_ckpt)
        self.assertEqual(config.dataset.min_length, 5)
        self.assertEqual(config.contrastive.pretraining_prob, 0.3)

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        self.assertTrue(config_path.exists(), f"Test config not found: {config_path}")

        config = ConfigLoader.load(str(config_path))

        # Check that tiny model config was loaded correctly
        self.assertEqual(config.model.hidden_size, 64)
        self.assertEqual(config.model.num_hidden_layers, 2)
        self.assertEqual(config.model.num_attention_heads, 2)
        self.assertEqual(config.trainer.per_device_train_batch_size, 2)
        self.assertEqual(config.dataset.max_seq_length, 128)

    def test_cli_override_system(self):
        """Test CLI override functionality."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"

        # Simulate command line args
        test_args = [
            "script.py",
            str(config_path),
            "--model.hidden_size",
            "128",
            "--optimizer.lr",
            "5e-4",
            "--trainer.per_device_train_batch_size",
            "4",
            "--trainer.logging_steps",
            "17",
            "--trainer.save_total_limit",
            "1",
            "--dataset.streaming",
            "false",
            "--datacollator.pack_sequences",
            "true",
            "--trainer.torch_compile",
            "true",
            "--trainer.torch_compile_backend",
            "aot_eager",
            "--trainer.enforce_full_packed_batches",
            "false",
            "--tokenizer.trust_remote_code",
            "true",
            "--tokenizer.allow_special_token_rewrite",
            "false",
        ]

        # Mock sys.argv
        original_argv = sys.argv
        sys.argv = test_args

        try:
            config = load_config_from_args()

            # Check that overrides were applied
            self.assertEqual(config.model.hidden_size, 128)  # Overridden from 64
            self.assertEqual(config.optimizer.lr, 5e-4)  # Overridden from 1e-4
            self.assertEqual(
                config.trainer.per_device_train_batch_size, 4
            )  # Overridden from 2
            self.assertEqual(config.trainer.save_total_limit, 1)
            self.assertFalse(config.dataset.streaming)
            self.assertTrue(config.datacollator.pack_sequences)
            self.assertEqual(config.trainer.logging_steps, 17)
            self.assertTrue(config.trainer.torch_compile)
            self.assertEqual(config.trainer.torch_compile_backend, "aot_eager")
            self.assertFalse(config.trainer.enforce_full_packed_batches)
            self.assertTrue(config.tokenizer.trust_remote_code)
            self.assertFalse(config.tokenizer.allow_special_token_rewrite)

            # Check that non-overridden values remain the same
            self.assertEqual(config.model.num_hidden_layers, 2)

        finally:
            sys.argv = original_argv

    def test_cli_masked_logits_only_loss_tokens(self):
        """Ensure CLI boolean parsing for loss-path selection is strict."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        cases = [
            ("off", False, None),
            ("ture", None, SystemExit),
        ]

        for token, expected, expected_exc in cases:
            with self.subTest(token=token):
                test_args = [
                    "script.py",
                    str(config_path),
                    "--trainer.masked_logits_only_loss",
                    token,
                ]
                original_argv = sys.argv
                sys.argv = test_args
                try:
                    if expected_exc is not None:
                        with self.assertRaises(expected_exc):
                            load_config_from_args()
                    else:
                        config = load_config_from_args()
                        self.assertEqual(
                            config.trainer.masked_logits_only_loss, expected
                        )
                finally:
                    sys.argv = original_argv

    def test_cli_store_true_defaults_do_not_override_yaml_truthy_values(self):
        """Ensure absent store_true flags do not stomp true values from YAML."""
        config_data = {
            "task": "pretraining",
            "debug": True,
            "mteb_overwrite_results": True,
            "dataset": {
                "load_all_from_disk": True,
                "force_redownload": True,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
            yaml.safe_dump(config_data, f)

        test_args = ["script.py", path]
        original_argv = sys.argv
        sys.argv = test_args
        try:
            cfg = load_config_from_args()
            self.assertTrue(cfg.debug)
            self.assertTrue(cfg.mteb_overwrite_results)
            self.assertTrue(cfg.dataset.load_all_from_disk)
            self.assertTrue(cfg.dataset.force_redownload)
        finally:
            sys.argv = original_argv
            os.unlink(path)

    def test_cli_dataset_and_tokenizer_scalar_overrides(self):
        """Ensure CLI parsing applies common dataset/tokenizer scalar overrides."""
        cases = [
            (
                self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml",
                ["--dataset.eval_samples", "2048"],
                lambda cfg: self.assertEqual(cfg.dataset.eval_samples, 2048),
            ),
            (
                self.test_config_dir / "contrastive" / "test_tiny_contrastive.yaml",
                ["--dataset.alpha", "0.75", "--tokenizer.truncation", "false"],
                lambda cfg: (
                    self.assertEqual(cfg.dataset.alpha, 0.75),
                    self.assertFalse(cfg.tokenizer.truncation),
                ),
            ),
        ]
        for config_path, cli_tokens, checker in cases:
            with self.subTest(config=str(config_path), cli_tokens=cli_tokens):
                original_argv = sys.argv
                sys.argv = ["script.py", str(config_path), *cli_tokens]
                try:
                    checker(load_config_from_args())
                finally:
                    sys.argv = original_argv

    def test_nested_config_override(self):
        """Test deeply nested configuration overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
model:
  hidden_size: 64
  num_attention_heads: 2
trainer:
  output_dir: "./test"
  use_cpu: true
optimizer:
  name: "adamw"
  lr: 1e-4
""")
            temp_config_path = f.name

        try:
            test_args = [
                "script.py",
                temp_config_path,
                "--model.hidden_size",
                "256",
                "--trainer.output_dir",
                "./new_test",
                "--optimizer.lr",
                "2e-4",
            ]

            original_argv = sys.argv
            sys.argv = test_args

            try:
                config = load_config_from_args()

                self.assertEqual(config.model.hidden_size, 256)
                self.assertEqual(config.trainer.output_dir, "./new_test")
                self.assertEqual(config.optimizer.lr, 2e-4)
                self.assertEqual(config.model.num_attention_heads, 2)  # Unchanged

            finally:
                sys.argv = original_argv

        finally:
            os.unlink(temp_config_path)

    def test_yaml_variables_exact_and_inline_resolution(self):
        """Ensure top-level variables resolve for exact and inline forms."""
        config_data = {
            "variables": {
                "seq": 384,
                "tag": "exp-a",
                "nested": {"lr": 2e-4},
            },
            "dataset": {"max_seq_length": "$variables.seq"},
            "tokenizer": {"max_length": "$variables.seq"},
            "wandb": {"name": "run-{$variables.tag}"},
            "optimizer": {"lr": "$variables.nested.lr"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
            yaml.safe_dump(config_data, f)

        try:
            cfg = ConfigLoader.load(path)
            self.assertEqual(cfg.dataset.max_seq_length, 384)
            self.assertEqual(cfg.tokenizer.max_length, 384)
            self.assertEqual(cfg.wandb.name, "run-exp-a")
            self.assertEqual(cfg.optimizer.lr, 2e-4)
        finally:
            os.unlink(path)

    def test_yaml_variables_circular_references_raise(self):
        """Ensure direct and nested variable cycles fail fast."""
        cases = [
            {
                "variables": {
                    "a": "$variables.b",
                    "b": "$variables.c",
                    "c": "$variables.a",
                },
                "dataset": {"max_seq_length": "$variables.a"},
            },
            {
                "variables": {
                    "a": {"nested": "$variables.b"},
                    "b": {"nested": "$variables.a"},
                },
                "dataset": {"max_seq_length": "$variables.a.nested.nested"},
            },
        ]

        for config_data in cases:
            with self.subTest(case=config_data["variables"]):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as f:
                    path = f.name
                    yaml.safe_dump(config_data, f)

                try:
                    with self.assertRaises(ValueError):
                        ConfigLoader.load(path)
                finally:
                    os.unlink(path)

    def test_yaml_unresolved_variable_tokens_warn(self):
        """Ensure unresolved inline variable tokens warn with location."""
        config_data = {
            "variables": {"tag": "abc"},
            "wandb": {"name": "run-${variables.missing}-{$variables.tag}"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
            yaml.safe_dump(config_data, f)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg = ConfigLoader.load(path)
            self.assertEqual(cfg.wandb.name, "run-${variables.missing}-abc")
            self.assertTrue(
                any("Unresolved variable token(s)" in str(w.message) for w in caught)
            )
        finally:
            os.unlink(path)

    def test_load_dot_overrides_list_forms(self):
        """Ensure ``ConfigLoader.load`` supports both dot-override token forms."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        cases = [
            (
                [
                    "trainer.max_steps=77",
                    "dataset.streaming=false",
                    "optimizer.lr=3e-4",
                ],
                {
                    "trainer.max_steps": 77,
                    "dataset.streaming": False,
                    "optimizer.lr": 3e-4,
                },
            ),
            (
                [
                    "--trainer.max_steps",
                    "42",
                    "--wandb.name=test-run",
                ],
                {"trainer.max_steps": 42, "wandb.name": "test-run"},
            ),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                cfg = ConfigLoader.load(str(config_path), overrides=overrides)
                if "trainer.max_steps" in expected:
                    self.assertEqual(
                        cfg.trainer.max_steps, expected["trainer.max_steps"]
                    )
                if "dataset.streaming" in expected:
                    self.assertEqual(
                        cfg.dataset.streaming, expected["dataset.streaming"]
                    )
                if "optimizer.lr" in expected:
                    self.assertEqual(cfg.optimizer.lr, expected["optimizer.lr"])
                if "wandb.name" in expected:
                    self.assertEqual(cfg.wandb.name, expected["wandb.name"])

    def test_load_dot_overrides_validation_errors(self):
        """Ensure unknown paths and invalid bool coercions fail for dot overrides."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        with self.assertRaises(ValueError):
            ConfigLoader.load(str(config_path), overrides=["trainer.not_real=1"])
        with self.assertRaises(ValueError):
            ConfigLoader.load(
                str(config_path),
                overrides=["dataset.streaming=truthy-but-invalid"],
            )

    def test_all_test_configs_load(self):
        """Test that all tiny task configs load with expected task/dataset metadata."""
        test_configs = [
            ("pretraining/test_tiny_pretrain.yaml", "pretraining", None),
            ("evaluation/test_tiny_glue.yaml", "glue", "cola"),
            ("contrastive/test_tiny_contrastive.yaml", "contrastive", "ALLNLI"),
        ]

        for config_name, expected_task, expected_dataset in test_configs:
            config_path = self.test_config_dir / config_name
            with self.subTest(config=config_name):
                self.assertTrue(
                    config_path.exists(), f"Config not found: {config_path}"
                )

                config = ConfigLoader.load(str(config_path))
                self.assertIsInstance(config, Config)
                self.assertEqual(config.task, expected_task)
                if expected_dataset is not None:
                    self.assertEqual(config.dataset.name, expected_dataset)
                self.assertIsNotNone(config.model.hidden_size)
                self.assertIsNotNone(config.trainer.output_dir)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from dataclasses import asdict

        config = Config()
        config.model.hidden_size = 128

        config_dict = asdict(config)

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["model"]["hidden_size"], 128)

    def test_config_save_includes_task_sections(self):
        """Ensure saved configs include glue and contrastive sections."""
        config = Config()
        config.glue.task_name = "sst2"
        config.contrastive.temperature = 0.1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            ConfigLoader.save(config, path)
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)

            self.assertIn("glue", data)
            self.assertIn("contrastive", data)
            self.assertEqual(data["glue"]["task_name"], "sst2")
            self.assertEqual(data["contrastive"]["temperature"], 0.1)
        finally:
            os.unlink(path)

    def test_preprocess_uses_tokenizer_path(self):
        """Ensure tokenizer path is preferred when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "hello": 3}
            raw_tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw_tokenizer,
                pad_token="[PAD]",
                unk_token="[UNK]",
                mask_token="[MASK]",
            )
            tokenizer.save_pretrained(tmpdir)

            config = Config()
            config.trainer.use_cpu = False
            config.tokenizer.name = "non-existent-tokenizer"
            config.tokenizer.path = tmpdir
            config.tokenizer.vocab_size = len(tokenizer)
            config.model.vocab_size = len(tokenizer)

            processed = ConfigLoader.preprocess_config(config, resolve_vocab_size=True)

            expected_vocab_size = round_up_to_multiple(len(tokenizer), 128)
            self.assertEqual(processed.model.vocab_size, expected_vocab_size)

    def test_config_loading_error_paths(self):
        """Ensure missing files and unknown keys fail with clear exceptions."""
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load("nonexistent_config.yaml")

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            config_data = {
                "model": {"hidden_size": 32, "unknown_key": 123},
                "trainer": {"output_dir": tmp_output_dir},
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                path = f.name
                yaml.safe_dump(config_data, f)

            try:
                with self.assertRaises(ValueError):
                    ConfigLoader.load(path)
            finally:
                os.unlink(path)

    def test_legacy_contrastive_key_migrations(self):
        """Ensure legacy contrastive fields map cleanly and reject conflicts."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = ConfigLoader.dict_to_config(
                {
                    "task": "contrastive",
                    "dataset": {"pretraining_prob": 0.45},
                }
            )
        self.assertEqual(cfg.contrastive.pretraining_prob, 0.45)
        self.assertTrue(
            any(
                "dataset.pretraining_prob" in str(w.message)
                and "contrastive.pretraining_prob" in str(w.message)
                for w in caught
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                ConfigLoader.dict_to_config(
                    {
                        "task": "contrastive",
                        "dataset": {"pretraining_prob": 0.4},
                        "contrastive": {"pretraining_prob": 0.6},
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            pre_ckpt_dir = str(Path(tmpdir) / "pre_ckpts")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg = ConfigLoader.dict_to_config(
                    {
                        "task": "contrastive",
                        "model": {
                            "pretrained_checkpoint_dir": pre_ckpt_dir,
                            "pretrained_checkpoint": "1234",
                            "allow_random_weights": True,
                        },
                    }
                )
            self.assertEqual(cfg.contrastive.pretrained_checkpoint_dir, pre_ckpt_dir)
            self.assertEqual(cfg.contrastive.pretrained_checkpoint, "1234")
            self.assertTrue(cfg.contrastive.allow_random_weights)
            self.assertTrue(
                any(
                    "model.pretrained_checkpoint_dir" in str(w.message)
                    and "contrastive.pretrained_checkpoint_dir" in str(w.message)
                    for w in caught
                )
            )

            with self.assertRaises(ValueError):
                ConfigLoader.dict_to_config(
                    {
                        "task": "contrastive",
                        "model": {
                            "pretrained_checkpoint_dir": str(
                                Path(tmpdir) / "model_ckpts"
                            )
                        },
                        "contrastive": {
                            "pretrained_checkpoint_dir": str(
                                Path(tmpdir) / "contrastive_ckpts"
                            )
                        },
                    }
                )

    def test_wandb_defaults_and_watch_validation(self):
        """Ensure W&B enablement and watch-mode validation semantics remain stable."""
        cfg = ConfigLoader.dict_to_config({"wandb": {"project": "unit-test-project"}})
        self.assertFalse(cfg.wandb.enabled)

        cfg = ConfigLoader.dict_to_config({})
        self.assertEqual(cfg.wandb.watch, "gradients")

        cfg = ConfigLoader.dict_to_config({"wandb": {"watch": "weights"}})
        self.assertEqual(cfg.wandb.watch, "parameters")

        with self.assertRaises(ValueError):
            ConfigLoader.dict_to_config({"wandb": {"watch": "mystery_mode"}})

    def test_save_step_and_retention_validation_matrix(self):
        """Ensure save/eval schedule validation behaves consistently across tasks."""
        with self.assertRaises(ValueError):
            ConfigLoader.dict_to_config({"trainer": {"save_steps": 0}})

        with self.assertRaises(ValueError):
            ConfigLoader.dict_to_config(
                {
                    "task": "pretraining",
                    "trainer": {
                        "save_strategy": "steps",
                        "save_steps": 0,
                    },
                }
            )

        valid_save_cases = [
            (
                "glue",
                {
                    "task": "glue",
                    "tokenizer": {"max_length": 128},
                    "trainer": {"save_strategy": "no", "save_steps": 0},
                },
            ),
            (
                "pretraining",
                {
                    "task": "pretraining",
                    "trainer": {"save_strategy": "no", "save_steps": 0, "max_steps": 1},
                },
            ),
            (
                "contrastive",
                {
                    "task": "contrastive",
                    "trainer": {"save_strategy": "no", "save_steps": 0, "max_steps": 1},
                },
            ),
        ]
        for task_name, config_dict in valid_save_cases:
            with self.subTest(task=task_name):
                cfg = ConfigLoader.dict_to_config(config_dict)
                self.assertEqual(cfg.task, task_name)
                self.assertEqual(cfg.trainer.save_strategy, "no")
                self.assertEqual(cfg.trainer.save_steps, 0)

        glue_epoch_cfg = ConfigLoader.dict_to_config(
            {
                "task": "glue",
                "tokenizer": {"max_length": 128},
                "trainer": {
                    "eval_strategy": "epoch",
                    "eval_steps": 0,
                },
            }
        )
        self.assertEqual(glue_epoch_cfg.task, "glue")
        self.assertEqual(glue_epoch_cfg.trainer.eval_strategy, "epoch")
        self.assertEqual(glue_epoch_cfg.trainer.eval_steps, 0)

        retention_cfg = ConfigLoader.dict_to_config(
            {"trainer": {"save_total_limit": 0}}
        )
        self.assertEqual(retention_cfg.trainer.save_total_limit, 0)

    def test_pretraining_tokenizer_max_length_sync_behavior(self):
        """Ensure tokenizer max-length sync warns only when tokenizer is shorter."""
        cases = [
            (512, 256, 512, True),
            (256, 4096, 4096, False),
        ]
        for dataset_max_len, tokenizer_max_len, expected_len, expect_warning in cases:
            with self.subTest(
                dataset_max_len=dataset_max_len, tokenizer_max_len=tokenizer_max_len
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    cfg = ConfigLoader.dict_to_config(
                        {
                            "task": "pretraining",
                            "dataset": {"max_seq_length": dataset_max_len},
                            "tokenizer": {"max_length": tokenizer_max_len},
                        }
                    )
                self.assertEqual(cfg.tokenizer.max_length, expected_len)
                found = any(
                    "tokenizer.max_length is smaller than dataset.max_seq_length"
                    in str(w.message)
                    for w in caught
                )
                self.assertEqual(found, expect_warning)

    def test_legacy_fields_are_ignored_with_deprecation_warnings(self):
        """Ensure deprecated fields are ignored while emitting clear warnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg_report = ConfigLoader.dict_to_config(
                {"trainer": {"report_to": ["wandb"]}, "wandb": {"enabled": False}}
            )
            cfg_scheduler = ConfigLoader.dict_to_config(
                {"scheduler": {"name": "cosine", "num_cycles": 1.5}}
            )
        self.assertEqual(cfg_report.trainer.report_to, [])
        self.assertEqual(cfg_scheduler.scheduler.name, "cosine")
        self.assertTrue(
            any("trainer.report_to" in str(w.message) for w in caught),
            "Expected deprecation warning for trainer.report_to",
        )
        self.assertTrue(
            any("scheduler.num_cycles" in str(w.message) for w in caught),
            "Expected deprecation warning for scheduler.num_cycles",
        )

    def test_config_value_validation_errors(self):
        """Ensure invalid scalar config values fail for known validation paths."""
        with self.assertRaises(ValueError):
            ConfigLoader.dict_to_config(
                {"task": "contrastive", "dataset": {"alpha": 0.0}}
            )
        with self.assertRaises(ValueError):
            ConfigLoader.dict_to_config(
                {
                    "task": "glue",
                    "tokenizer": {"max_length": 128},
                    "glue": {"preprocessing_num_proc": -1},
                }
            )

    def test_legacy_trainer_aliases_map_to_canonical_fields(self):
        """Ensure legacy trainer aliases map to canonical fields."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = ConfigLoader.dict_to_config(
                {
                    "trainer": {
                        "train_batch_size": 8,
                        "eval_batch_size": 12,
                        "max_ckpt": 5,
                    }
                }
            )
        self.assertEqual(cfg.trainer.per_device_train_batch_size, 8)
        self.assertEqual(cfg.trainer.per_device_eval_batch_size, 12)
        self.assertEqual(cfg.trainer.save_total_limit, 5)
        self.assertIsNone(cfg.trainer.max_ckpt)
        self.assertTrue(
            any("trainer.train_batch_size" in str(w.message) for w in caught)
        )
        self.assertTrue(
            any("trainer.eval_batch_size" in str(w.message) for w in caught)
        )
        self.assertTrue(any("trainer.max_ckpt" in str(w.message) for w in caught))

    def test_glue_seq_length_syncs_tokenizer_max_length(self):
        """Ensure tokenizer.max_length is synced to glue.max_seq_length for GLUE."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = ConfigLoader.dict_to_config(
                {
                    "task": "glue",
                    "glue": {"max_seq_length": 192},
                    "tokenizer": {"max_length": 256},
                }
            )
        self.assertEqual(cfg.tokenizer.max_length, 192)
        self.assertTrue(
            any(
                "tokenizer.max_length does not match glue.max_seq_length"
                in str(w.message)
                for w in caught
            )
        )


if __name__ == "__main__":
    unittest.main()
