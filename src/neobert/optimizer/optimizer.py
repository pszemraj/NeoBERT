"""Factory helpers for constructing optimizers used in NeoBERT training."""

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate.utils import DistributedType
from torch.optim import Adam, AdamW

from neobert.optimizer.muon_clip import MuonClipConfig, MuonClipOptimizer

# from .soap.soap import SOAP  # TODO: Add SOAP optimizer implementation

logger = logging.getLogger(__name__)


def _build_adamw_param_groups(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups for AdamW.

    :param torch.nn.Module model: Model whose parameters will be grouped.
    :param float weight_decay: Weight decay value for decayed parameters.
    :return list[dict[str, Any]]: Parameter groups for AdamW.
    """
    decay_params = []
    no_decay_params = []
    embedding_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for param in module.parameters(recurse=False)
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if (
            param.ndim < 2
            or name_lower.endswith(".bias")
            or "norm" in name_lower
            or id(param) in embedding_param_ids
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_optimizer(
    model: torch.nn.Module,
    distributed_type: DistributedType,
    model_config: Optional[Any] = None,
    muon_config: Optional[Any] = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Construct an optimizer configured for the current training run.

    :param torch.nn.Module model: Model whose parameters will be optimized.
    :param DistributedType distributed_type: Distributed execution mode.
    :param Any | None model_config: Optional model config (required for MuonClip).
    :param Any | None muon_config: MuonClip overrides or dataclass.
    :param Any kwargs: Additional optimizer-specific keyword arguments.
    :return torch.optim.Optimizer: Initialized optimizer instance.
    :raises ValueError: If MuonClip is requested without ``model_config``.
    :raises ValueError: If an unsupported optimizer name is provided.
    """
    optimizer_name = kwargs.pop("name").lower()

    logger.info(f"Initializing optimizer: {optimizer_name}")

    match optimizer_name:
        case "adamw":
            if muon_config is not None:
                logger.warning(
                    "optimizer.muon_config provided but optimizer is adamw; ignoring"
                )
            weight_decay = kwargs.pop("weight_decay", 0.0)
            param_groups = _build_adamw_param_groups(model, weight_decay)
            optimizer = AdamW(param_groups, **kwargs)
            logger.info(f"AdamW initialized with lr={kwargs.get('lr', 'default')}")
            return optimizer

        case "adam":
            if muon_config is not None:
                logger.warning(
                    "optimizer.muon_config provided but optimizer is adam; ignoring"
                )
            optimizer = Adam(model.parameters(), **kwargs)
            logger.info(f"Adam initialized with lr={kwargs.get('lr', 'default')}")
            return optimizer

        case "muonclip" | "muon-clip" | "muon_clip":
            if model_config is None:
                raise ValueError(
                    "MuonClip requires model_config to be passed. "
                    "Update get_optimizer() call in trainer to include model_config argument."
                )

            # Build MuonClipConfig from kwargs
            lr = kwargs.pop("lr", 1e-4)
            weight_decay = kwargs.pop("weight_decay", 0.0)
            betas = kwargs.pop("betas", (0.9, 0.98))
            eps = kwargs.pop("eps", 1e-10)

            extra_args = {k: v for k, v in kwargs.items()}
            if extra_args:
                logger.warning(
                    "Ignoring unused optimizer kwargs for MuonClip: %s",
                    ", ".join(sorted(extra_args)),
                )

            muon_kwargs: Dict[str, Any] = {}
            if muon_config is not None:
                if is_dataclass(muon_config):
                    muon_kwargs = asdict(muon_config)
                elif isinstance(muon_config, dict):
                    muon_kwargs = dict(muon_config)
                else:
                    raise TypeError(
                        "optimizer.muon_config must be a mapping or dataclass"
                    )

            muon_clip_cfg = MuonClipConfig(
                lr=lr,
                adam_betas=betas,
                adam_decay=weight_decay,
                adam_eps=eps,
                **muon_kwargs,
            )

            logger.info(
                f"MuonClip configuration:\n"
                f"  - Learning rate: {muon_clip_cfg.lr}\n"
                f"  - Clipping enabled: {muon_clip_cfg.enable_clipping}\n"
                f"  - Clipping threshold: {muon_clip_cfg.clipping_threshold}\n"
                f"  - Clipping interval: {muon_clip_cfg.clipping_interval}\n"
                f"  - QK chunk size: {muon_clip_cfg.clipping_qk_chunk_size}\n"
                f"  - Capture last microbatch only: {muon_clip_cfg.capture_last_microbatch_only}\n"
                f"  - Newton-Schulz steps: {muon_clip_cfg.ns_steps}\n"
                f"  - Alpha (Q/K balance): {muon_clip_cfg.clipping_alpha}\n"
                f"  - Orthogonalization: {muon_clip_cfg.orthogonalization}"
            )

            return MuonClipOptimizer(model, model_config, muon_clip_cfg)

        # case "SOAP":
        #     assert distributed_type is not DistributedType.DEEPSPEED, (
        #         "SOAP does not support DeepSpeed"
        #     )
        #     return SOAP(model.parameters(), **kwargs)

        case _:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw, muonclip"
            )
