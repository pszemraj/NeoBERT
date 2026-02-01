"""Configuration dataclasses and helpers for NeoBERT runs."""

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def round_up_to_multiple(x: int, N: int = 128) -> int:
    """Round an integer up to the nearest multiple of ``N``.

    :param int x: Value to round up.
    :param int N: Multiple to round to (default: 128).
    :return int: Rounded value.
    """
    return ((x + N - 1) // N) * N


@dataclass
class ModelConfig:
    """Model architecture and initialization settings."""

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
    """Dataset loading and preprocessing configuration."""

    name: str = "refinedweb"
    path: str = ""
    num_workers: int = 16
    streaming: bool = True
    cache_dir: Optional[str] = None
    max_seq_length: int = 512
    validation_split: Optional[float] = None
    train_split: Optional[str] = None
    eval_split: Optional[str] = None
    num_proc: int = 4  # Number of processes for tokenization
    shuffle_buffer_size: int = 10000  # Buffer size for streaming dataset shuffling
    pre_tokenize: bool = False  # Whether to pre-tokenize non-streaming datasets
    pre_tokenize_output: Optional[str] = None  # Where to save pre-tokenized datasets

    # Contrastive-specific
    load_all_from_disk: bool = False
    force_redownload: bool = False
    pretraining_prob: float = 0.3
    min_length: int = 512


@dataclass
class TokenizerConfig:
    """Tokenizer setup for training and evaluation."""

    name: str = "bert-base-uncased"
    path: Optional[str] = None
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    vocab_size: Optional[int] = None  # For compatibility with tests


@dataclass
class MuonConfig:
    """Muon optimizer-specific configuration."""

    muon_beta: float = 0.95
    muon_decay: float = 0.0
    ns_steps: int = 5
    enable_clipping: bool = True
    clipping_threshold: float = 50.0
    clipping_alpha: float = 0.5
    clipping_warmup_steps: int = 0
    detect_anomalies: bool = False
    orthogonalization: str = "polar_express"
    algorithm: Optional[str] = None  # Alias for orthogonalization
    polar_express: Optional[bool] = None  # Legacy toggle
    clipping_layers_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters for training."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    muon_config: Optional[MuonConfig] = None


@dataclass
class SchedulerConfig:
    """Learning-rate scheduler configuration."""

    name: str = "cosine"
    warmup_steps: int = 10000
    total_steps: Optional[int] = None
    num_cycles: float = 0.5
    decay_steps: int = 50000  # For contrastive training
    warmup_percent: Optional[float] = None
    decay_percent: Optional[float] = None


@dataclass
class TrainerConfig:
    """Training loop and runtime configuration."""

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000000
    save_steps: int = 10000
    eval_steps: int = 10000
    logging_steps: int = 100
    output_dir: str = "./output"
    overwrite_output_dir: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = False
    gradient_clipping: Optional[float] = None
    mixed_precision: str = "bf16"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    # Training control
    num_train_epochs: int = 3
    eval_strategy: str = "steps"  # "steps" or "epoch"
    save_strategy: str = "steps"  # "steps", "epoch", "best", or "no"
    save_total_limit: Optional[int] = 3
    early_stopping: int = 0
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = True
    load_best_model_at_end: bool = False
    save_model: bool = True

    # For backwards compatibility with old configs
    disable_tqdm: bool = False
    dataloader_num_workers: int = 0
    use_cpu: bool = False
    report_to: List[str] = field(default_factory=list)
    tf32: bool = True
    max_ckpt: int = 3
    # Legacy batch size fields (use per_device versions instead)
    train_batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None


@dataclass
class DataCollatorConfig:
    """Masking and padding configuration for data collators."""

    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    mask_all: bool = False
    pack_sequences: bool = False
    max_length: Optional[int] = None


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    project: str = "neo-bert"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "online"
    log_interval: int = 100
    resume: str = "never"
    dir: str = "logs/wandb"


@dataclass
class GLUEConfig:
    """GLUE task configuration for fine-tuning and evaluation."""

    # Task configuration
    task_name: str = "cola"
    num_labels: int = 2
    max_seq_length: int = 128

    # Model loading
    pretrained_model_path: Optional[str] = None  # Path to pretrained model config.yaml
    pretrained_checkpoint_dir: Optional[str] = None  # Directory containing checkpoints
    pretrained_checkpoint: Optional[Union[str, int]] = (
        None  # Specific checkpoint to load
    )
    allow_random_weights: bool = False  # Allow testing with random weights

    # Fine-tuning specific
    classifier_dropout: float = 0.1
    classifier_init_range: float = 0.02
    transfer_from_task: bool = False  # Whether to transfer from another GLUE task

    # Data configuration (override dataset defaults)
    num_workers: int = 4
    preprocessing_num_proc: int = 4


@dataclass
class ContrastiveConfig:
    """Contrastive training configuration."""

    temperature: float = 0.05
    pooling: str = "avg"  # avg, cls, max
    loss_type: str = "simcse"  # simcse, supcon
    hard_negative_weight: float = 0.0


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    datacollator: DataCollatorConfig = field(default_factory=DataCollatorConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    glue: GLUEConfig = field(default_factory=GLUEConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)

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

    # Metadata for downstream evaluations (e.g., GLUE linkage)
    pretraining_metadata: Dict[str, Any] = field(default_factory=dict)

    # Misc
    seed: int = 0
    debug: bool = False
    config_path: Optional[str] = None


class ConfigLoader:
    """Load and merge configuration from YAML files and command line arguments."""

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML configuration file.

        :param Union[str, Path] path: Path to a YAML file.
        :return dict[str, Any]: Parsed configuration mapping.
        """
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override config into base config.

        :param dict[str, Any] base: Base configuration mapping.
        :param dict[str, Any] override: Override configuration mapping.
        :return dict[str, Any]: Merged configuration mapping.
        """
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
        """Convert dictionary to a ``Config`` dataclass.

        :param dict[str, Any] cfg_dict: Nested configuration mapping.
        :return Config: Fully-populated configuration instance.
        """
        config = Config()

        # Store raw model dict for GLUE compatibility
        if "model" in cfg_dict:
            config._raw_model_dict = cfg_dict["model"]
        else:
            config._raw_model_dict = None

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
            optimizer_dict = dict(cfg_dict["optimizer"])
            muon_cfg_dict = optimizer_dict.pop("muon_config", None)

            for k, v in optimizer_dict.items():
                if hasattr(config.optimizer, k):
                    if k in ["lr", "eps"] and isinstance(v, str):
                        v = float(v)
                    setattr(config.optimizer, k, v)

            if muon_cfg_dict is not None:
                if isinstance(muon_cfg_dict, MuonConfig):
                    config.optimizer.muon_config = muon_cfg_dict
                elif isinstance(muon_cfg_dict, dict):
                    muon_cfg = MuonConfig()
                    for mk, mv in muon_cfg_dict.items():
                        if hasattr(muon_cfg, mk):
                            setattr(muon_cfg, mk, mv)
                    config.optimizer.muon_config = muon_cfg
                else:
                    raise TypeError(
                        "optimizer.muon_config must be a mapping or MuonConfig instance"
                    )

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

        # Update glue config
        if "glue" in cfg_dict:
            for k, v in cfg_dict["glue"].items():
                if hasattr(config.glue, k):
                    setattr(config.glue, k, v)

        # Update contrastive config
        if "contrastive" in cfg_dict:
            for k, v in cfg_dict["contrastive"].items():
                if hasattr(config.contrastive, k):
                    setattr(config.contrastive, k, v)

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
                "glue",
                "contrastive",
            ]:
                setattr(config, k, v)

        return config

    @staticmethod
    def preprocess_config(config: Config) -> Config:
        """Preprocess and validate config, resolving any dynamic values.

        This should be called after config loading but before any downstream consumers.

        :param Config config: Configuration to preprocess.
        :return Config: Preprocessed configuration.
        """
        # Resolve vocab_size for GPU efficiency (skip CPU-only runs/tests).
        use_cpu = getattr(config.trainer, "use_cpu", False)
        if not use_cpu and hasattr(config.tokenizer, "name") and config.tokenizer.name:
            # Import tokenizer here to avoid circular imports
            from .tokenizer import get_tokenizer

            # Create tokenizer to determine actual vocab size
            tokenizer_source = config.tokenizer.path or config.tokenizer.name
            tokenizer = get_tokenizer(
                pretrained_model_name_or_path=tokenizer_source,
                max_length=config.tokenizer.max_length,
                vocab_size=config.tokenizer.vocab_size or config.model.vocab_size,
            )

            actual_vocab_size = len(tokenizer)
            rounded_vocab_size = round_up_to_multiple(actual_vocab_size, 128)

            # Update all vocab_size references consistently
            original_model_vocab_size = config.model.vocab_size

            config.model.vocab_size = rounded_vocab_size
            if hasattr(config.tokenizer, "vocab_size"):
                config.tokenizer.vocab_size = rounded_vocab_size

            # Log the change if significant
            if actual_vocab_size != rounded_vocab_size:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Config preprocessing: vocab_size {actual_vocab_size} rounded to "
                    f"{rounded_vocab_size} for GPU efficiency (original config: {original_model_vocab_size})"
                )

        return config

    @staticmethod
    def load(
        config_file: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        preprocess: bool = True,
    ) -> Config:
        """Load configuration from file and apply overrides.

        :param str | Path | None config_file: Optional YAML configuration path.
        :param dict[str, Any] | None overrides: Optional override mapping.
        :param bool preprocess: Whether to run ``preprocess_config``.
        :return Config: Loaded configuration.
        """
        config_dict = {}

        # Load from file if provided
        if config_file:
            config_dict = ConfigLoader.load_yaml(config_file)

        # Apply overrides
        if overrides:
            config_dict = ConfigLoader.merge_configs(config_dict, overrides)

        config = ConfigLoader.dict_to_config(config_dict)

        # Preprocess config to resolve dynamic values
        if preprocess:
            config = ConfigLoader.preprocess_config(config)

        return config

    @staticmethod
    def save(config: Config, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        :param Config config: Configuration to serialize.
        :param str | Path path: Destination path for the YAML file.
        """
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
            "glue": asdict(config.glue),
            "contrastive": asdict(config.contrastive),
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
            "pretraining_metadata": config.pretraining_metadata,
        }

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for command line overrides.

    :return argparse.ArgumentParser: Configured argument parser.
    """
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
    parser.add_argument(
        "--model.flash_attention",
        type=lambda x: x.lower() == "true",
        help="Use flash attention",
    )

    # Dataset arguments
    parser.add_argument("--dataset.name", type=str, help="Dataset name")
    parser.add_argument("--dataset.path", type=str, help="Dataset path")
    parser.add_argument(
        "--dataset.num_workers", type=int, help="Number of data workers"
    )
    parser.add_argument(
        "--dataset.streaming",
        type=lambda x: x.lower() == "true",
        help="Stream dataset from hub",
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
        "--trainer.bf16",
        type=lambda x: x.lower() == "true",
        help="Use BF16 (default: True)",
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
    parser.add_argument(
        "--datacollator.pack_sequences",
        type=lambda x: x.lower() == "true",
        help="Pack sequences into fixed-length chunks",
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
    """Convert an argparse namespace to a nested dictionary.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return dict[str, Any]: Nested configuration mapping.
    """
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
    """Load configuration from command line arguments.

    :return Config: Loaded configuration with CLI overrides applied.
    """
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

    config = ConfigLoader.dict_to_config(config_dict)
    config.config_path = args.config

    # Preprocess config to resolve dynamic values
    config = ConfigLoader.preprocess_config(config)

    return config
