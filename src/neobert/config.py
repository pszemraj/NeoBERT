"""Configuration dataclasses and helpers for NeoBERT runs."""

import argparse
import re
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

import yaml

from neobert.warnings import NeoBERTWarning


def round_up_to_multiple(x: int, N: int = 128) -> int:
    """Round an integer up to the nearest multiple of ``N``.

    :param int x: Value to round up.
    :param int N: Multiple to round to (default: 128).
    :return int: Rounded value.
    """
    return ((x + N - 1) // N) * N


def _parse_cli_bool(value: str) -> bool:
    """Parse strict boolean CLI override values.

    :param str value: Raw CLI token.
    :raises argparse.ArgumentTypeError: If the token is not boolean-like.
    :return bool: Parsed boolean value.
    """
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value (true/false, 1/0, yes/no, on/off)."
    )


def resolve_mixed_precision(value: Any, *, task: str) -> str:
    """Normalize and validate mixed-precision policy for a task.

    Accepted user values:
    - booleans: ``True`` -> ``"bf16"``, ``False`` -> ``"no"``
    - strings: ``"bf16"``, ``"fp32"``, ``"no"``

    Runtime policy:
    - ``fp32`` is normalized to ``no``
    - ``fp16`` is unsupported and rejected for all tasks

    :param Any value: Raw mixed-precision value from config/CLI.
    :param str task: Active task name.
    :raises ValueError: If the value is unsupported for the given task.
    :return str: Normalized precision mode.
    """
    del task  # Mixed-precision policy is task-independent.
    if isinstance(value, bool):
        normalized = "bf16" if value else "no"
    else:
        normalized = str(value).strip().lower()
    if normalized == "fp32":
        normalized = "no"

    if normalized == "fp16":
        raise ValueError(
            "trainer.mixed_precision='fp16' is not supported. "
            "Use 'bf16' (recommended) or 'no'/'fp32'."
        )

    valid_values = {"no", "bf16"}
    if normalized not in valid_values:
        raise ValueError(
            "trainer.mixed_precision must be one of {'no','bf16','fp32'}, "
            f"got {value!r}."
        )
    return normalized


# Note: mutable defaults in dataclasses below use default_factory to avoid shared state.


@dataclass
class ModelConfig:
    """Model architecture and initialization settings."""

    name: Optional[str] = None
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
    attn_backend: str = "sdpa"  # "sdpa" or "flash_attn_varlen"
    kernel_backend: str = "auto"  # "auto", "liger", or "torch"
    pad_token_id: int = 0
    from_hub: bool = False


@dataclass
class DatasetConfig:
    """Dataset loading and preprocessing configuration."""

    name: str = "refinedweb"
    config: Optional[str] = None
    path: str = ""
    num_workers: int = 16
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = None
    streaming: bool = True
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    max_seq_length: int = 512
    text_column: Optional[str] = None
    validation_split: Optional[float] = None
    train_split: Optional[str] = None
    eval_split: Optional[str] = None
    eval_samples: Optional[int] = None
    streaming_read_retries: int = 4
    streaming_read_retry_backoff_seconds: float = 5.0
    streaming_read_retry_max_backoff_seconds: float = 60.0
    num_proc: int = 4  # Number of processes for tokenization
    shuffle_buffer_size: int = 10000  # Buffer size for streaming dataset shuffling

    # Contrastive-specific
    load_all_from_disk: bool = False
    force_redownload: bool = False
    min_length: int = 5
    alpha: float = 1.0


@dataclass
class TokenizerConfig:
    """Tokenizer setup for training and evaluation."""

    name: str = "bert-base-uncased"
    path: Optional[str] = None
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    # Persisted tokenizer-derived vocabulary size used to validate checkpoint resume.
    vocab_size: Optional[int] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    allow_special_token_rewrite: bool = False


# Single source of truth for Muon enum spellings; the optimizer-side
# MuonClipConfig validates through these same helpers.
_MUON_NORM_FACTORS = frozenset(
    {"neobert", "muon_reference", "spectral", "match_rms_adamw", "none"}
)
_MUON_ORTHOGONALIZATIONS = frozenset({"newton_schulz", "polar_express"})
_MUON_PARAM_POLICIES = frozenset({"all_2d", "hidden_2d"})


def _normalize_muon_choice(
    value: Any,
    *,
    valid: frozenset,
    field_name: str,
) -> str:
    """Normalize a Muon enum-like config value and validate it.

    :param Any value: Raw configured value.
    :param frozenset valid: Canonical accepted values.
    :param str field_name: Field name used in error messages.
    :raises ValueError: If the normalized value is not a valid option.
    :return str: Canonical normalized value.
    """
    normalized = str(value).strip().replace("-", "_").lower()
    if normalized not in valid:
        raise ValueError(
            f"Unsupported {field_name} '{value}'. "
            f"Valid options: {', '.join(sorted(valid))}"
        )
    return normalized


def normalize_muon_norm_factor(value: Any) -> str:
    """Normalize a Muon ``norm_factor`` spelling to its canonical value.

    :param Any value: Raw configured value.
    :raises ValueError: If the value is not a supported norm factor.
    :return str: Canonical norm-factor name.
    """
    return _normalize_muon_choice(
        value,
        valid=_MUON_NORM_FACTORS,
        field_name="norm_factor",
    )


def normalize_muon_orthogonalization(value: Any) -> str:
    """Normalize a Muon ``orthogonalization`` value.

    :param Any value: Raw configured value.
    :raises ValueError: If the value is not a supported algorithm.
    :return str: Canonical orthogonalization name.
    """
    return _normalize_muon_choice(
        value,
        valid=_MUON_ORTHOGONALIZATIONS,
        field_name="orthogonalization",
    )


def normalize_muon_param_policy(value: Any) -> str:
    """Normalize a Muon ``param_policy`` spelling to its canonical value.

    :param Any value: Raw configured value.
    :raises ValueError: If the value is not a supported param policy.
    :return str: Canonical param-policy name.
    """
    return _normalize_muon_choice(
        value,
        valid=_MUON_PARAM_POLICIES,
        field_name="param_policy",
    )


@dataclass
class MuonConfig:
    """Muon optimizer-specific configuration."""

    muon_beta: float = 0.95
    nesterov: bool = True
    muon_decay: float = 0.0
    ns_steps: int = 5
    enable_clipping: bool = True
    clipping_threshold: float = 50.0
    clipping_alpha: float = 0.5
    clipping_warmup_steps: int = 0
    clipping_interval: int = 10
    clipping_qk_chunk_size: int = 1024
    detect_anomalies: bool = False
    orthogonalization: str = "polar_express"
    norm_factor: str = "neobert"
    param_policy: str = "hidden_2d"
    clipping_layers_mapping: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate numeric fields and normalize enum-like fields."""
        if self.clipping_warmup_steps < 0:
            raise ValueError(
                f"clipping_warmup_steps must be >= 0, got {self.clipping_warmup_steps}"
            )
        self.orthogonalization = normalize_muon_orthogonalization(
            self.orthogonalization
        )
        self.norm_factor = normalize_muon_norm_factor(self.norm_factor)
        self.param_policy = normalize_muon_param_policy(self.param_policy)


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
    decay_steps: Optional[int] = None  # Optional absolute decay end step
    final_lr_ratio: float = 0.1
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
    eval_max_batches: Optional[int] = None
    logging_steps: int = 100
    enforce_full_packed_batches: bool = False
    log_train_accuracy: bool = False
    log_grad_norm: bool = True
    output_dir: str = "./output"
    overwrite_output_dir: bool = True
    gradient_checkpointing: bool = False
    gradient_clipping: Optional[float] = None
    mixed_precision: str = "bf16"
    masked_logits_only_loss: bool = True
    torch_compile: bool = False
    torch_compile_dynamic: Optional[bool] = None
    torch_compile_backend: str = "inductor"
    resume_from_checkpoint: Optional[str] = None

    # Training control
    num_train_epochs: int = 3
    eval_strategy: str = "steps"  # "steps" or "epoch"
    save_strategy: str = "steps"  # "steps", "epoch", "best", or "no"
    save_total_limit: Optional[int] = 3
    early_stopping: int = 0
    save_model: bool = True

    disable_tqdm: bool = False
    dataloader_num_workers: int = 0
    use_cpu: bool = False
    tf32: bool = True
    log_weight_norms: bool = True


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

    enabled: bool = False
    project: str = "neo-bert"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "online"
    watch: str = "gradients"
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
    pretraining_prob: float = 0.0
    pretraining_dataset_path: Optional[str] = None
    pretrained_checkpoint_dir: Optional[str] = None
    pretrained_checkpoint: Optional[Union[str, int]] = None
    allow_random_weights: bool = False


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

    # MTEB-specific
    mteb_task_type: str = "all"  # all, classification, clustering, etc.
    mteb_batch_size: int = 32
    mteb_pooling: str = "avg"  # avg (or mean), cls
    mteb_overwrite_results: bool = False

    # Model loading
    pretrained_checkpoint: str = "latest"
    use_deepspeed: bool = False

    # Metadata for downstream evaluations (e.g., GLUE linkage)
    pretraining_metadata: Dict[str, Any] = field(default_factory=dict)

    # Misc
    seed: int = 0
    debug: bool = False
    config_path: Optional[str] = None


class ConfigLoader:
    """Load and merge configuration from YAML files and command line arguments."""

    _VARIABLE_EXACT_PATTERN = re.compile(r"^\$variables\.([A-Za-z0-9_.-]+)$")
    _VARIABLE_INLINE_PATTERN = re.compile(
        r"\{\$variables\.([A-Za-z0-9_.-]+)\}|\$\{variables\.([A-Za-z0-9_.-]+)\}"
    )
    _VARIABLE_TOKEN_PATTERN = re.compile(r"\$variables\.([A-Za-z0-9_.-]+)")

    @staticmethod
    def _resolve_yaml_variables(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve top-level ``variables`` references in a YAML mapping.

        Supported forms:
        - ``$variables.foo`` for exact, type-preserving replacement.
        - ``{$variables.foo}`` and ``${variables.foo}`` for inline interpolation.

        :param dict[str, Any] cfg_dict: Raw YAML mapping.
        :raises ValueError: If variables section is malformed or references are cyclic.
        :return dict[str, Any]: Mapping with variables resolved.
        """
        if not isinstance(cfg_dict, dict):
            return cfg_dict

        source = deepcopy(cfg_dict)
        variables = source.pop("variables", None)
        if variables is None:
            return source
        if not isinstance(variables, dict):
            raise ValueError("Top-level 'variables' must be a mapping when provided.")

        resolved_variables: Dict[str, Any] = {}
        resolving_stack: List[str] = []
        unresolved_refs: Dict[str, set[str]] = {}

        def _lookup_variable(path: str) -> Any:
            """Lookup a variable value by dot path.

            :param str path: Dot path relative to ``variables``.
            :raises KeyError: If the variable path does not exist.
            :return Any: Raw variable value.
            """
            current: Any = variables
            for segment in path.split("."):
                if not isinstance(current, dict) or segment not in current:
                    raise KeyError(path)
                current = current[segment]
            return current

        def _resolve_variable(path: str) -> Any:
            """Resolve a single variable path with cycle detection.

            :param str path: Dot path inside ``variables``.
            :raises ValueError: If a circular reference is detected.
            :raises KeyError: If the path is unknown.
            :return Any: Resolved variable value.
            """
            if path in resolved_variables:
                return deepcopy(resolved_variables[path])
            if path in resolving_stack:
                cycle = " -> ".join(resolving_stack + [path])
                raise ValueError(
                    f"Circular variable reference detected in config variables: {cycle}"
                )

            raw_value = _lookup_variable(path)
            resolving_stack.append(path)
            try:
                # Nested dict/list values still route variable tokens through
                # ``_resolve_variable(...)``, so transitive cycles are caught by
                # ``resolving_stack`` even when references appear deep in objects.
                resolved_value = _resolve_node(raw_value, f"variables.{path}")
            finally:
                resolving_stack.pop()

            resolved_variables[path] = resolved_value
            return deepcopy(resolved_value)

        def _resolve_string(value: str, location: str) -> Any:
            """Resolve variable tokens in a string.

            :param str value: String value to resolve.
            :param str location: Dot path used for diagnostics.
            :raises ValueError: If an exact variable token points to an unknown path.
            :return Any: Resolved value.
            """
            exact_match = ConfigLoader._VARIABLE_EXACT_PATTERN.fullmatch(value)
            if exact_match:
                var_path = exact_match.group(1)
                try:
                    return _resolve_variable(var_path)
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown variable reference at '{location}': "
                        f"$variables.{var_path}"
                    ) from exc

            def _replace_inline(match: re.Match[str]) -> str:
                """Replace inline variable token with resolved string value.

                :param re.Match[str] match: Inline variable regex match.
                :return str: Replacement string for interpolation.
                """
                var_path = match.group(1) or match.group(2)
                if var_path is None:
                    return match.group(0)
                try:
                    replacement = _resolve_variable(var_path)
                except KeyError:
                    unresolved_refs.setdefault(location, set()).add(
                        f"$variables.{var_path}"
                    )
                    return match.group(0)
                return str(replacement)

            resolved = ConfigLoader._VARIABLE_INLINE_PATTERN.sub(
                _replace_inline,
                value,
            )

            for unresolved in ConfigLoader._VARIABLE_TOKEN_PATTERN.findall(resolved):
                unresolved_refs.setdefault(location, set()).add(
                    f"$variables.{unresolved}"
                )
            return resolved

        def _resolve_node(node: Any, location: str) -> Any:
            """Resolve variables recursively for nested objects.

            :param Any node: Nested mapping/list/scalar node.
            :param str location: Dot path used for diagnostics.
            :return Any: Resolved node.
            """
            if isinstance(node, dict):
                return {
                    key: _resolve_node(value, f"{location}.{key}")
                    for key, value in node.items()
                }
            if isinstance(node, list):
                return [
                    _resolve_node(item, f"{location}[{idx}]")
                    for idx, item in enumerate(node)
                ]
            if isinstance(node, str):
                return _resolve_string(node, location)
            return node

        resolved_config = _resolve_node(source, "config")

        for location in sorted(unresolved_refs):
            tokens = ", ".join(sorted(unresolved_refs[location]))
            warnings.warn(
                f"Unresolved variable token(s) at '{location}': {tokens}",
                NeoBERTWarning,
                stacklevel=3,
            )

        return resolved_config

    @staticmethod
    def _normalize_dot_overrides(overrides: List[str]) -> List[tuple[str, str]]:
        """Normalize dot-path overrides from ``key=value`` or ``--key value`` forms.

        :param list[str] overrides: Raw override tokens.
        :raises ValueError: If override syntax is malformed.
        :return list[tuple[str, str]]: Normalized ``(path, raw_value)`` tuples.
        """
        parsed: List[tuple[str, str]] = []
        idx = 0
        while idx < len(overrides):
            token = str(overrides[idx]).strip()
            if not token:
                idx += 1
                continue

            if token.startswith("--"):
                stripped = token[2:].strip()
                if not stripped:
                    raise ValueError(
                        "Invalid override token '--'. Expected '--section.key=value'."
                    )
                if "=" in stripped:
                    key, value = stripped.split("=", 1)
                    if not key:
                        raise ValueError(
                            f"Invalid override '{token}'. Missing override key."
                        )
                    parsed.append((key, value))
                    idx += 1
                    continue
                if idx + 1 >= len(overrides):
                    raise ValueError(
                        f"Invalid override '{token}': missing value token."
                    )
                value = str(overrides[idx + 1])
                if value.startswith("--"):
                    raise ValueError(
                        f"Invalid override '{token}': expected value after key token."
                    )
                parsed.append((stripped, value))
                idx += 2
                continue

            if "=" not in token:
                raise ValueError(
                    f"Invalid override '{token}'. Expected 'section.key=value'."
                )
            key, value = token.split("=", 1)
            if not key:
                raise ValueError(f"Invalid override '{token}'. Missing override key.")
            parsed.append((key.strip(), value))
            idx += 1

        return parsed

    @staticmethod
    def _coerce_dot_override_value(
        path: str,
        raw_value: str,
        current_value: Any,
        expected_type: Any = None,
    ) -> Any:
        """Coerce a dot override token into the existing field type.

        :param str path: Dot-path field identifier.
        :param str raw_value: Override value as provided by CLI/list.
        :param Any current_value: Existing in-memory field value.
        :param Any expected_type: Optional dataclass annotation for target field.
        :raises ValueError: If coercion fails.
        :return Any: Coerced value.
        """

        def _coerce_from_annotation(annotation: Any) -> Any:
            """Coerce override value using a type annotation.

            :param Any annotation: Field annotation.
            :raises ValueError: If coercion fails for the annotation.
            :return Any: Coerced value.
            """
            if annotation is Any:
                return yaml.safe_load(raw_value)

            origin = get_origin(annotation)
            if origin is Union:
                args = list(get_args(annotation))
                allow_none = type(None) in args
                non_none_args = [arg for arg in args if arg is not type(None)]
                if allow_none and str(raw_value).strip().lower() in {
                    "none",
                    "null",
                    "~",
                }:
                    return None
                if len(non_none_args) == 1:
                    return _coerce_from_annotation(non_none_args[0])
                parsed_union = yaml.safe_load(raw_value)
                if any(
                    (
                        candidate is Any
                        or (candidate is type(None) and parsed_union is None)
                        or (
                            candidate in {int, float, bool, str, list, dict}
                            and isinstance(parsed_union, candidate)
                        )
                    )
                    for candidate in non_none_args
                ):
                    return parsed_union
                raise ValueError(
                    f"Invalid override for '{path}': {raw_value!r} does not match "
                    f"allowed union types {non_none_args}."
                )

            if annotation is bool:
                try:
                    return _parse_cli_bool(raw_value)
                except argparse.ArgumentTypeError as exc:
                    raise ValueError(
                        f"Invalid boolean override for '{path}': {raw_value!r}. "
                        "Expected true/false, 1/0, yes/no, or on/off."
                    ) from exc
            if annotation is int:
                try:
                    return int(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid integer override for '{path}': {raw_value!r}."
                    ) from exc
            if annotation is float:
                try:
                    return float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid float override for '{path}': {raw_value!r}."
                    ) from exc
            if annotation is str:
                return raw_value

            if annotation in {list, dict} or origin in {list, dict}:
                parsed_collection = yaml.safe_load(raw_value)
                collection_type = (
                    list if (annotation is list or origin is list) else dict
                )
                if not isinstance(parsed_collection, collection_type):
                    raise ValueError(
                        f"Invalid override for '{path}': expected {collection_type.__name__}, "
                        f"got {type(parsed_collection).__name__}."
                    )
                return parsed_collection

            return yaml.safe_load(raw_value)

        if isinstance(current_value, bool):
            try:
                return _parse_cli_bool(raw_value)
            except argparse.ArgumentTypeError as exc:
                raise ValueError(
                    f"Invalid boolean override for '{path}': {raw_value!r}. "
                    "Expected true/false, 1/0, yes/no, or on/off."
                ) from exc

        if isinstance(current_value, int) and not isinstance(current_value, bool):
            try:
                return int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer override for '{path}': {raw_value!r}."
                ) from exc

        if isinstance(current_value, float):
            try:
                return float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid float override for '{path}': {raw_value!r}."
                ) from exc

        if isinstance(current_value, str):
            return raw_value

        if current_value is None and expected_type is not None:
            return _coerce_from_annotation(expected_type)

        parsed = yaml.safe_load(raw_value)
        if current_value is None:
            return parsed

        if isinstance(current_value, list):
            if not isinstance(parsed, list):
                raise ValueError(
                    f"Invalid list override for '{path}': {raw_value!r}. "
                    'Use YAML list syntax, e.g. "[a, b]".'
                )
            return parsed

        if isinstance(current_value, dict):
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Invalid mapping override for '{path}': {raw_value!r}. "
                    'Use YAML mapping syntax, e.g. "{k: v}".'
                )
            return parsed

        return parsed

    @staticmethod
    def _apply_dot_override(config: Config, path: str, raw_value: str) -> None:
        """Apply one dot-path override directly onto a config object.

        :param Config config: Configuration object to mutate.
        :param str path: Dot-path override key.
        :param str raw_value: Raw override value string.
        :raises ValueError: If path is unknown or value cannot be coerced.
        """
        path = str(path).strip()
        if not path or "." in {path[0], path[-1]}:
            raise ValueError(
                f"Invalid override path '{path}'. Expected dotted key path."
            )

        parts = path.split(".")
        current: Any = config
        traversed: List[str] = []

        for part in parts[:-1]:
            traversed.append(part)
            if isinstance(current, dict):
                if part not in current:
                    available = ", ".join(sorted(current.keys())) or "<empty>"
                    raise ValueError(
                        f"Unknown override path '{path}': key '{part}' not found under "
                        f"'{'.'.join(traversed[:-1]) or '<root>'}'. Available keys: "
                        f"{available}."
                    )
                current = current[part]
                continue

            if not hasattr(current, part):
                available_fields = (
                    ", ".join(sorted(f.name for f in fields(type(current))))
                    if hasattr(current, "__dataclass_fields__")
                    else "<none>"
                )
                raise ValueError(
                    f"Unknown override path '{path}': segment '{part}' does not exist "
                    f"on '{type(current).__name__}'. Available fields: {available_fields}."
                )
            current = getattr(current, part)

        leaf = parts[-1]
        if isinstance(current, dict):
            if leaf not in current:
                available = ", ".join(sorted(current.keys())) or "<empty>"
                raise ValueError(
                    f"Unknown override path '{path}': key '{leaf}' not found. "
                    f"Available keys: {available}."
                )
            current_value = current[leaf]
            current[leaf] = ConfigLoader._coerce_dot_override_value(
                path,
                raw_value,
                current_value,
            )
            return

        if not hasattr(current, leaf):
            available_fields = (
                ", ".join(sorted(f.name for f in fields(type(current))))
                if hasattr(current, "__dataclass_fields__")
                else "<none>"
            )
            raise ValueError(
                f"Unknown override path '{path}': field '{leaf}' does not exist on "
                f"'{type(current).__name__}'. Available fields: {available_fields}."
            )

        current_value = getattr(current, leaf)
        expected_type = None
        if hasattr(current, "__dataclass_fields__"):
            field_map = {f.name: f.type for f in fields(type(current))}
            expected_type = field_map.get(leaf)
        coerced_value = ConfigLoader._coerce_dot_override_value(
            path,
            raw_value,
            current_value,
            expected_type=expected_type,
        )
        setattr(current, leaf, coerced_value)

    @staticmethod
    def _apply_dot_overrides(config: Config, overrides: List[str]) -> None:
        """Apply dot-path overrides to an instantiated config.

        :param Config config: Configuration object to mutate.
        :param list[str] overrides: Dot-path overrides.
        :raises ValueError: If any override token/path/value is invalid.
        """
        for path, raw_value in ConfigLoader._normalize_dot_overrides(overrides):
            ConfigLoader._apply_dot_override(config, path, raw_value)
        ConfigLoader._validate_config_values(config)

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML configuration file.

        :param Union[str, Path] path: Path to a YAML file.
        :return dict[str, Any]: Parsed configuration mapping.
        """
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f) or {}
        if not isinstance(raw_cfg, dict):
            raise ValueError(
                f"Config file '{path}' must define a top-level YAML mapping."
            )
        return ConfigLoader._resolve_yaml_variables(raw_cfg)

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
    def _validate_config_keys(cfg_dict: Dict[str, Any]) -> None:
        """Validate config keys against dataclass fields.

        :param dict[str, Any] cfg_dict: Normalized configuration mapping.
        :raises ValueError: When unknown keys are found.
        """
        if cfg_dict is None:
            return

        unknown_keys: list[str] = []

        config_fields = {f.name for f in fields(Config)}
        for key in cfg_dict:
            if key not in config_fields:
                unknown_keys.append(key)

        def _check_section(section: str, cls: type) -> None:
            """Validate keys for a named config subsection.

            :param str section: Section key to validate.
            :param type cls: Dataclass type describing valid keys.
            """
            if section not in cfg_dict:
                return
            mapping = cfg_dict.get(section)
            if not isinstance(mapping, dict):
                unknown_keys.append(section)
                return
            valid = {f.name for f in fields(cls)}
            for key in mapping:
                if key not in valid:
                    unknown_keys.append(f"{section}.{key}")

        _check_section("model", ModelConfig)
        _check_section("dataset", DatasetConfig)
        _check_section("tokenizer", TokenizerConfig)
        _check_section("optimizer", OptimizerConfig)
        _check_section("scheduler", SchedulerConfig)
        _check_section("trainer", TrainerConfig)
        _check_section("datacollator", DataCollatorConfig)
        _check_section("wandb", WandbConfig)
        _check_section("glue", GLUEConfig)
        _check_section("contrastive", ContrastiveConfig)

        # Validate nested muon_config keys if provided.
        optimizer_cfg = cfg_dict.get("optimizer", {})
        if isinstance(optimizer_cfg, dict) and "muon_config" in optimizer_cfg:
            muon_cfg = optimizer_cfg["muon_config"]
            if isinstance(muon_cfg, dict):
                valid_muon = {f.name for f in fields(MuonConfig)}
                for key in muon_cfg:
                    if key not in valid_muon:
                        unknown_keys.append(f"optimizer.muon_config.{key}")

        if unknown_keys:
            unknown_keys.sort()
            raise ValueError(
                "Unknown configuration keys detected: "
                + ", ".join(unknown_keys)
                + ". Update your config or remove unused fields. This repository "
                "does not support pre-stable config-schema compatibility; recreate "
                "older checkpoint configs against the current reference or use the "
                "matching historical code revision."
            )

    @staticmethod
    def _validate_config_values(config: Config) -> None:
        """Validate semantic config values after dataclass hydration.

        :param Config config: Configuration instance to validate.
        :raises ValueError: If semantic constraints are violated.
        """
        errors: list[str] = []
        task = str(config.task).strip().lower()
        valid_tasks = {"pretraining", "glue", "mteb", "contrastive"}
        if task not in valid_tasks:
            errors.append(
                f"task must be one of {sorted(valid_tasks)}, got {config.task!r}"
            )
        else:
            config.task = task

        if config.dataset.max_seq_length <= 0:
            errors.append(
                "dataset.max_seq_length must be > 0, "
                f"got {config.dataset.max_seq_length}."
            )
        if config.dataset.min_length <= 0:
            errors.append(
                f"dataset.min_length must be > 0, got {config.dataset.min_length}."
            )
        if config.dataset.min_length > config.dataset.max_seq_length:
            errors.append(
                "dataset.min_length must be <= dataset.max_seq_length, got "
                f"{config.dataset.min_length} > {config.dataset.max_seq_length}."
            )
        if config.dataset.alpha <= 0.0:
            errors.append(f"dataset.alpha must be > 0, got {config.dataset.alpha}.")
        if config.contrastive.temperature <= 0.0:
            errors.append(
                "contrastive.temperature must be > 0, got "
                f"{config.contrastive.temperature}."
            )
        if config.dataset.num_workers < 0:
            errors.append(
                f"dataset.num_workers must be >= 0, got {config.dataset.num_workers}."
            )
        if config.dataset.num_proc < 0:
            errors.append(
                f"dataset.num_proc must be >= 0, got {config.dataset.num_proc}."
            )
        if (
            config.dataset.eval_samples is not None
            and int(config.dataset.eval_samples) <= 0
        ):
            errors.append(
                "dataset.eval_samples must be > 0 when set, got "
                f"{config.dataset.eval_samples}."
            )
        if config.dataset.streaming_read_retries < 0:
            errors.append(
                "dataset.streaming_read_retries must be >= 0, got "
                f"{config.dataset.streaming_read_retries}."
            )
        if config.dataset.streaming_read_retry_backoff_seconds <= 0:
            errors.append(
                "dataset.streaming_read_retry_backoff_seconds must be > 0, got "
                f"{config.dataset.streaming_read_retry_backoff_seconds}."
            )
        if config.dataset.streaming_read_retry_max_backoff_seconds <= 0:
            errors.append(
                "dataset.streaming_read_retry_max_backoff_seconds must be > 0, got "
                f"{config.dataset.streaming_read_retry_max_backoff_seconds}."
            )
        if (
            config.dataset.streaming_read_retry_max_backoff_seconds
            < config.dataset.streaming_read_retry_backoff_seconds
        ):
            errors.append(
                "dataset.streaming_read_retry_max_backoff_seconds must be >= "
                "dataset.streaming_read_retry_backoff_seconds, got "
                f"{config.dataset.streaming_read_retry_max_backoff_seconds} < "
                f"{config.dataset.streaming_read_retry_backoff_seconds}."
            )

        if task in {"pretraining", "contrastive"}:
            if config.tokenizer.max_length < config.dataset.max_seq_length:
                warnings.warn(
                    "tokenizer.max_length is smaller than dataset.max_seq_length for "
                    f"{task}; syncing tokenizer.max_length from "
                    f"{config.tokenizer.max_length} to {config.dataset.max_seq_length}.",
                    NeoBERTWarning,
                    stacklevel=2,
                )
                config.tokenizer.max_length = config.dataset.max_seq_length

            if config.dataset.max_seq_length > config.model.max_position_embeddings:
                errors.append(
                    "dataset.max_seq_length must be <= model.max_position_embeddings "
                    f"for {task}, got {config.dataset.max_seq_length} > "
                    f"{config.model.max_position_embeddings}."
                )
        if task == "glue" and config.tokenizer.max_length != config.glue.max_seq_length:
            warnings.warn(
                "tokenizer.max_length does not match glue.max_seq_length; syncing "
                f"tokenizer.max_length from {config.tokenizer.max_length} to "
                f"{config.glue.max_seq_length}.",
                NeoBERTWarning,
                stacklevel=2,
            )
            config.tokenizer.max_length = config.glue.max_seq_length
        if task == "glue":
            if config.glue.max_seq_length <= 0:
                errors.append(
                    "glue.max_seq_length must be > 0, got "
                    f"{config.glue.max_seq_length}."
                )
            if config.glue.num_workers < 0:
                errors.append(
                    f"glue.num_workers must be >= 0, got {config.glue.num_workers}."
                )
            if config.glue.preprocessing_num_proc < 0:
                errors.append(
                    "glue.preprocessing_num_proc must be >= 0, got "
                    f"{config.glue.preprocessing_num_proc}."
                )

        if not 0.0 <= float(config.datacollator.mlm_probability) <= 1.0:
            errors.append(
                "datacollator.mlm_probability must be in [0, 1], got "
                f"{config.datacollator.mlm_probability}."
            )
        if (
            config.datacollator.max_length is not None
            and config.datacollator.max_length <= 0
        ):
            errors.append(
                "datacollator.max_length must be > 0 when set, got "
                f"{config.datacollator.max_length}."
            )

        if config.trainer.per_device_train_batch_size <= 0:
            errors.append(
                "trainer.per_device_train_batch_size must be > 0, got "
                f"{config.trainer.per_device_train_batch_size}."
            )
        if config.trainer.per_device_eval_batch_size <= 0:
            errors.append(
                "trainer.per_device_eval_batch_size must be > 0, got "
                f"{config.trainer.per_device_eval_batch_size}."
            )
        if config.trainer.gradient_accumulation_steps <= 0:
            errors.append(
                "trainer.gradient_accumulation_steps must be > 0, got "
                f"{config.trainer.gradient_accumulation_steps}."
            )
        valid_eval_strategies = {"steps", "epoch"}
        eval_strategy = str(config.trainer.eval_strategy).strip().lower()
        if eval_strategy not in valid_eval_strategies:
            errors.append(
                "trainer.eval_strategy must be one of "
                f"{sorted(valid_eval_strategies)}, got "
                f"{config.trainer.eval_strategy!r}."
            )
        else:
            config.trainer.eval_strategy = eval_strategy

        valid_save_strategies = {"steps", "epoch", "best", "no"}
        save_strategy = str(config.trainer.save_strategy).strip().lower()
        if save_strategy not in valid_save_strategies:
            errors.append(
                "trainer.save_strategy must be one of "
                f"{sorted(valid_save_strategies)}, got "
                f"{config.trainer.save_strategy!r}."
            )
        else:
            config.trainer.save_strategy = save_strategy
        if task in {"pretraining", "contrastive"} and save_strategy not in {
            "steps",
            "no",
        }:
            errors.append(
                "trainer.save_strategy supports only {'steps','no'} for "
                f"{task}, got {config.trainer.save_strategy!r}."
            )

        if config.trainer.logging_steps <= 0:
            errors.append(
                f"trainer.logging_steps must be > 0, got {config.trainer.logging_steps}."
            )
        if config.trainer.save_steps < 0:
            errors.append(
                f"trainer.save_steps must be >= 0, got {config.trainer.save_steps}."
            )
        elif save_strategy == "steps" and config.trainer.save_steps == 0:
            errors.append(
                "trainer.save_steps must be > 0 when "
                f"trainer.save_strategy='steps', got {config.trainer.save_steps}."
            )
        if config.trainer.eval_steps < 0:
            errors.append(
                f"trainer.eval_steps must be >= 0, got {config.trainer.eval_steps}."
            )
        elif eval_strategy == "steps" and config.trainer.eval_steps == 0:
            errors.append(
                "trainer.eval_steps must be > 0 when "
                f"trainer.eval_strategy='steps', got {config.trainer.eval_steps}."
            )
        if task in {"pretraining", "contrastive"} and config.trainer.max_steps <= 0:
            errors.append(
                f"trainer.max_steps must be > 0 for {task}, got {config.trainer.max_steps}."
            )
        if (
            config.trainer.save_total_limit is not None
            and config.trainer.save_total_limit < 0
        ):
            errors.append(
                "trainer.save_total_limit must be >= 0 when set, got "
                f"{config.trainer.save_total_limit}."
            )
        if (
            config.trainer.eval_max_batches is not None
            and config.trainer.eval_max_batches <= 0
        ):
            errors.append(
                "trainer.eval_max_batches must be > 0 when set, got "
                f"{config.trainer.eval_max_batches}."
            )
        if config.trainer.dataloader_num_workers < 0:
            errors.append(
                "trainer.dataloader_num_workers must be >= 0, got "
                f"{config.trainer.dataloader_num_workers}."
            )
        if config.scheduler.warmup_steps < 0:
            errors.append(
                f"scheduler.warmup_steps must be >= 0, got {config.scheduler.warmup_steps}."
            )
        if (
            config.scheduler.total_steps is not None
            and config.scheduler.total_steps <= 0
        ):
            errors.append(
                f"scheduler.total_steps must be > 0 when set, got {config.scheduler.total_steps}."
            )
        if (
            config.scheduler.decay_steps is not None
            and config.scheduler.decay_steps <= 0
        ):
            errors.append(
                f"scheduler.decay_steps must be > 0 when set, got {config.scheduler.decay_steps}."
            )
        if config.scheduler.warmup_percent is not None and not (
            0.0 <= config.scheduler.warmup_percent <= 100.0
        ):
            errors.append(
                "scheduler.warmup_percent must be in [0, 100], got "
                f"{config.scheduler.warmup_percent}."
            )
        if config.scheduler.decay_percent is not None and not (
            0.0 <= config.scheduler.decay_percent <= 100.0
        ):
            errors.append(
                "scheduler.decay_percent must be in [0, 100], got "
                f"{config.scheduler.decay_percent}."
            )
        if config.scheduler.final_lr_ratio < 0.0:
            errors.append(
                "scheduler.final_lr_ratio must be >= 0, got "
                f"{config.scheduler.final_lr_ratio}."
            )

        if config.optimizer.lr <= 0:
            errors.append(f"optimizer.lr must be > 0, got {config.optimizer.lr}.")
        if config.optimizer.weight_decay < 0:
            errors.append(
                f"optimizer.weight_decay must be >= 0, got {config.optimizer.weight_decay}."
            )
        if config.optimizer.eps <= 0:
            errors.append(f"optimizer.eps must be > 0, got {config.optimizer.eps}.")
        if len(config.optimizer.betas) != 2:
            errors.append(
                "optimizer.betas must contain exactly 2 values, got "
                f"{config.optimizer.betas}."
            )
        else:
            beta1, beta2 = config.optimizer.betas
            if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
                errors.append(
                    "optimizer.betas values must be in [0, 1), got "
                    f"{config.optimizer.betas}."
                )

        try:
            config.trainer.mixed_precision = resolve_mixed_precision(
                config.trainer.mixed_precision,
                task=task,
            )
        except ValueError as exc:
            errors.append(str(exc))

        valid_wandb_modes = {"online", "offline", "disabled"}
        wandb_mode = str(config.wandb.mode).strip().lower()
        if wandb_mode not in valid_wandb_modes:
            errors.append(
                f"wandb.mode must be one of {sorted(valid_wandb_modes)}, got {config.wandb.mode!r}."
            )
        else:
            config.wandb.mode = wandb_mode
        wandb_watch = str(getattr(config.wandb, "watch", "gradients")).strip().lower()
        valid_wandb_watch = {
            "gradients",
            "parameters",
            "weights",
            "all",
            "off",
            "none",
            "disabled",
            "false",
            "0",
        }
        if wandb_watch not in valid_wandb_watch:
            errors.append(
                "wandb.watch must be one of "
                f"{sorted(valid_wandb_watch)}, got {config.wandb.watch!r}."
            )
        else:
            # Keep backwards compatibility with older configs that used the
            # pre-W&B API alias ``weights``.
            config.wandb.watch = (
                "parameters" if wandb_watch == "weights" else wandb_watch
            )

        if task != "contrastive" and config.use_deepspeed:
            warnings.warn(
                "Top-level 'use_deepspeed' only affects contrastive checkpoint loading. "
                "Runtime backend selection is controlled by Accelerate launch config. "
                "Install the optional legacy-checkpoints extra when DeepSpeed ZeRO "
                "conversion is actually needed.",
                NeoBERTWarning,
                stacklevel=2,
            )
        if errors:
            raise ValueError("Invalid configuration values:\n- " + "\n- ".join(errors))

    @staticmethod
    def dict_to_config(cfg_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to a ``Config`` dataclass.

        :param dict[str, Any] cfg_dict: Nested configuration mapping.
        :return Config: Fully-populated configuration instance.
        """
        cfg_dict = cfg_dict or {}
        ConfigLoader._validate_config_keys(cfg_dict)

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
                    config.optimizer.muon_config = MuonConfig(**muon_cfg_dict)
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
            wandb_dict = cfg_dict["wandb"]
            for k, v in wandb_dict.items():
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

        ConfigLoader._validate_config_values(config)

        return config

    @staticmethod
    def load(
        config_file: Optional[Union[str, Path]] = None,
        overrides: Optional[Union[Dict[str, Any], List[str]]] = None,
    ) -> Config:
        """Load configuration from file and apply overrides.

        :param str | Path | None config_file: Optional YAML configuration path.
        :param dict[str, Any] | list[str] | None overrides: Optional overrides.
            - mapping form is merged into YAML before dataclass hydration.
            - list form accepts ``section.key=value`` and ``--section.key value``.
        :return Config: Loaded configuration.
        """
        config_dict = {}

        # Load from file if provided
        if config_file:
            config_dict = ConfigLoader.load_yaml(config_file)

        # Apply overrides
        if isinstance(overrides, dict) and overrides:
            config_dict = ConfigLoader.merge_configs(config_dict, overrides)
        elif overrides is not None and not isinstance(overrides, list):
            raise TypeError(
                "ConfigLoader.load(..., overrides=...) expects either a mapping "
                "or a list of dot-path override strings."
            )

        config = ConfigLoader.dict_to_config(config_dict)
        if isinstance(overrides, list) and overrides:
            ConfigLoader._apply_dot_overrides(config, overrides)

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


def load_config_from_args(
    argv: Optional[List[str]] = None, require_config: bool = False
) -> Config:
    """Load configuration from command line arguments.

    :param list[str] | None argv: Raw argv including script name (defaults to sys.argv).
    :param bool require_config: Whether a config path must be provided.
    :return Config: Loaded configuration with CLI overrides applied.
    """
    parser = argparse.ArgumentParser(description="NeoBERT Configuration")
    parser.add_argument(
        "config",
        nargs=None if require_config else "?",
        default=None,
        help="Path to configuration YAML file",
    )
    args, overrides = parser.parse_known_args(argv[1:] if argv is not None else None)
    config = ConfigLoader.load(args.config, overrides=overrides)
    config.config_path = args.config

    return config
