import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    vocab_size: int = 30522
    rope: bool = True
    rms_norm: bool = True
    hidden_act: str = "swiglu"
    dropout_prob: float = 0.0
    norm_eps: float = 1e-5
    embedding_init_range: float = 0.02
    decoder_init_range: float = 0.02
    classifier_init_range: float = 0.02
    flash_attention: bool = True
    ngpt: bool = False
    base_scale: float = 1.0 / (960.0**0.5)
    pad_token_id: int = 0


@dataclass
class DatasetConfig:
    name: str = "refinedweb"
    path: str = ""
    num_workers: int = 16
    streaming: bool = True
    cache_dir: Optional[str] = None
    max_seq_length: int = 512
    validation_split: Optional[float] = None

    # Contrastive-specific
    load_all_from_disk: bool = False
    force_redownload: bool = False
    pretraining_prob: float = 0.3
    min_length: int = 512


@dataclass
class TokenizerConfig:
    name: str = "bert-base-uncased"
    path: Optional[str] = None
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 10000
    total_steps: Optional[int] = None
    num_cycles: float = 0.5
    decay_steps: int = 50000  # For contrastive training


@dataclass
class TrainerConfig:
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000000
    save_steps: int = 10000
    eval_steps: int = 10000
    logging_steps: int = 100
    output_dir: str = "./output"
    overwrite_output_dir: bool = True
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    gradient_clipping: Optional[float] = None
    mixed_precision: str = "bf16"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    # For backwards compatibility with old configs
    disable_tqdm: bool = False


@dataclass
class DataCollatorConfig:
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None


@dataclass
class WandbConfig:
    project: str = "neo-bert"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "online"
    log_interval: int = 100
    resume: str = "never"
    dir: str = "logs/wandb"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    datacollator: DataCollatorConfig = field(default_factory=DataCollatorConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Task-specific
    task: str = "pretraining"  # pretraining, glue, mteb, contrastive

    # Accelerate config
    accelerate_config_file: Optional[str] = None
    mixed_precision: str = "bf16"

    # MTEB-specific
    mteb_task_type: str = "all"  # all, classification, clustering, etc.
    mteb_batch_size: int = 32
    mteb_pooling: str = "mean"  # mean, cls
    mteb_overwrite_results: bool = False

    # Model loading
    pretrained_checkpoint: str = "latest"
    use_deepspeed: bool = True

    # Misc
    seed: int = 0
    debug: bool = False


class ConfigLoader:
    """Load and merge configuration from YAML files and command line arguments"""

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override config into base config"""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def dict_to_config(cfg_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config dataclass"""
        config = Config()

        # Update model config
        if "model" in cfg_dict:
            for k, v in cfg_dict["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)

        # Update dataset config
        if "dataset" in cfg_dict:
            for k, v in cfg_dict["dataset"].items():
                if hasattr(config.dataset, k):
                    setattr(config.dataset, k, v)

        # Update tokenizer config
        if "tokenizer" in cfg_dict:
            for k, v in cfg_dict["tokenizer"].items():
                if hasattr(config.tokenizer, k):
                    setattr(config.tokenizer, k, v)

        # Update optimizer config
        if "optimizer" in cfg_dict:
            for k, v in cfg_dict["optimizer"].items():
                if hasattr(config.optimizer, k):
                    setattr(config.optimizer, k, v)

        # Update scheduler config
        if "scheduler" in cfg_dict:
            for k, v in cfg_dict["scheduler"].items():
                if hasattr(config.scheduler, k):
                    setattr(config.scheduler, k, v)

        # Update trainer config
        if "trainer" in cfg_dict:
            for k, v in cfg_dict["trainer"].items():
                if hasattr(config.trainer, k):
                    setattr(config.trainer, k, v)

        # Update datacollator config
        if "datacollator" in cfg_dict:
            for k, v in cfg_dict["datacollator"].items():
                if hasattr(config.datacollator, k):
                    setattr(config.datacollator, k, v)

        # Update wandb config
        if "wandb" in cfg_dict:
            for k, v in cfg_dict["wandb"].items():
                if hasattr(config.wandb, k):
                    setattr(config.wandb, k, v)

        # Update top-level config
        for k, v in cfg_dict.items():
            if hasattr(config, k) and k not in [
                "model",
                "dataset",
                "tokenizer",
                "optimizer",
                "scheduler",
                "trainer",
                "datacollator",
                "wandb",
            ]:
                setattr(config, k, v)

        return config

    @staticmethod
    def load(
        config_file: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Config:
        """Load configuration from file and apply overrides"""
        config_dict = {}

        # Load from file if provided
        if config_file:
            config_dict = ConfigLoader.load_yaml(config_file)

        # Apply overrides
        if overrides:
            config_dict = ConfigLoader.merge_configs(config_dict, overrides)

        return ConfigLoader.dict_to_config(config_dict)

    @staticmethod
    def save(config: Config, path: Union[str, Path]):
        """Save configuration to YAML file"""
        # Convert dataclasses to dict
        config_dict = {
            "model": asdict(config.model),
            "dataset": asdict(config.dataset),
            "tokenizer": asdict(config.tokenizer),
            "optimizer": asdict(config.optimizer),
            "scheduler": asdict(config.scheduler),
            "trainer": asdict(config.trainer),
            "datacollator": asdict(config.datacollator),
            "wandb": asdict(config.wandb),
            "task": config.task,
            "accelerate_config_file": config.accelerate_config_file,
            "mixed_precision": config.mixed_precision,
            "mteb_task_type": config.mteb_task_type,
            "mteb_batch_size": config.mteb_batch_size,
            "mteb_pooling": config.mteb_pooling,
            "mteb_overwrite_results": config.mteb_overwrite_results,
            "pretrained_checkpoint": config.pretrained_checkpoint,
            "use_deepspeed": config.use_deepspeed,
            "seed": config.seed,
            "debug": config.debug,
        }

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for command line overrides"""
    parser = argparse.ArgumentParser(description="NeoBERT Configuration")

    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file"
    )

    # Model arguments
    parser.add_argument("--model.hidden_size", type=int, help="Hidden size")
    parser.add_argument(
        "--model.num_hidden_layers", type=int, help="Number of hidden layers"
    )
    parser.add_argument(
        "--model.num_attention_heads", type=int, help="Number of attention heads"
    )
    parser.add_argument("--model.intermediate_size", type=int, help="Intermediate size")
    parser.add_argument(
        "--model.max_position_embeddings", type=int, help="Max position embeddings"
    )
    parser.add_argument("--model.vocab_size", type=int, help="Vocabulary size")
    parser.add_argument(
        "--model.rope", type=lambda x: x.lower() == "true", help="Use RoPE"
    )
    parser.add_argument(
        "--model.rms_norm", type=lambda x: x.lower() == "true", help="Use RMS norm"
    )
    parser.add_argument(
        "--model.hidden_act", type=str, help="Hidden activation function"
    )
    parser.add_argument("--model.dropout_prob", type=float, help="Dropout probability")

    # Dataset arguments
    parser.add_argument("--dataset.name", type=str, help="Dataset name")
    parser.add_argument("--dataset.path", type=str, help="Dataset path")
    parser.add_argument(
        "--dataset.num_workers", type=int, help="Number of data workers"
    )
    parser.add_argument(
        "--dataset.max_seq_length", type=int, help="Maximum sequence length"
    )
    parser.add_argument(
        "--dataset.load_all_from_disk", action="store_true", help="Load all from disk"
    )
    parser.add_argument(
        "--dataset.force_redownload", action="store_true", help="Force redownload"
    )
    parser.add_argument(
        "--dataset.pretraining_prob", type=float, help="Pretraining probability"
    )
    parser.add_argument(
        "--dataset.min_length", type=int, help="Minimum sequence length"
    )

    # Tokenizer arguments
    parser.add_argument("--tokenizer.name", type=str, help="Tokenizer name")
    parser.add_argument("--tokenizer.path", type=str, help="Tokenizer path")

    # Optimizer arguments
    parser.add_argument("--optimizer.name", type=str, help="Optimizer name")
    parser.add_argument("--optimizer.lr", type=float, help="Learning rate")
    parser.add_argument("--optimizer.weight_decay", type=float, help="Weight decay")

    # Scheduler arguments
    parser.add_argument("--scheduler.name", type=str, help="Scheduler name")
    parser.add_argument("--scheduler.warmup_steps", type=int, help="Warmup steps")
    parser.add_argument("--scheduler.decay_steps", type=int, help="Decay steps")

    # Trainer arguments
    parser.add_argument(
        "--trainer.per_device_train_batch_size", type=int, help="Train batch size"
    )
    parser.add_argument(
        "--trainer.per_device_eval_batch_size", type=int, help="Eval batch size"
    )
    parser.add_argument(
        "--trainer.gradient_accumulation_steps",
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--trainer.max_steps", type=int, help="Maximum training steps")
    parser.add_argument(
        "--trainer.save_steps", type=int, help="Save checkpoint every N steps"
    )
    parser.add_argument("--trainer.eval_steps", type=int, help="Evaluate every N steps")
    parser.add_argument("--trainer.output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--trainer.fp16", type=lambda x: x.lower() == "true", help="Use FP16"
    )
    parser.add_argument(
        "--trainer.bf16", type=lambda x: x.lower() == "true", help="Use BF16"
    )
    parser.add_argument("--trainer.seed", type=int, help="Random seed")
    parser.add_argument(
        "--trainer.gradient_clipping", type=float, help="Gradient clipping"
    )
    parser.add_argument("--trainer.mixed_precision", type=str, help="Mixed precision")

    # Data collator arguments
    parser.add_argument(
        "--datacollator.mlm_probability", type=float, help="MLM probability"
    )

    # WandB arguments
    parser.add_argument("--wandb.project", type=str, help="WandB project name")
    parser.add_argument("--wandb.entity", type=str, help="WandB entity")
    parser.add_argument("--wandb.name", type=str, help="WandB run name")
    parser.add_argument(
        "--wandb.mode", type=str, help="WandB mode (online/offline/disabled)"
    )

    # Top-level arguments
    parser.add_argument(
        "--task", type=str, help="Task (pretraining/glue/mteb/contrastive)"
    )
    parser.add_argument("--seed", type=int, help="Global random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # MTEB-specific arguments
    parser.add_argument("--mteb_task_type", type=str, help="MTEB task type")
    parser.add_argument("--mteb_batch_size", type=int, help="MTEB batch size")
    parser.add_argument("--mteb_pooling", type=str, help="MTEB pooling method")
    parser.add_argument(
        "--mteb_overwrite_results", action="store_true", help="Overwrite MTEB results"
    )

    # Model loading arguments
    parser.add_argument(
        "--pretrained_checkpoint", type=str, help="Pretrained checkpoint"
    )
    parser.add_argument(
        "--use_deepspeed", type=lambda x: x.lower() == "true", help="Use DeepSpeed"
    )

    return parser


def parse_args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to nested dictionary"""
    config_dict = {}

    for key, value in vars(args).items():
        if value is not None and key != "config":
            # Handle nested keys like 'model.hidden_size'
            parts = key.split(".")
            current = config_dict

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

    return config_dict


def load_config_from_args() -> Config:
    """Load configuration from command line arguments"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load base config from file if provided
    config_dict = {}
    if args.config:
        config_dict = ConfigLoader.load_yaml(args.config)

    # Apply command line overrides
    overrides = parse_args_to_dict(args)
    if overrides:
        config_dict = ConfigLoader.merge_configs(config_dict, overrides)

    return ConfigLoader.dict_to_config(config_dict)
