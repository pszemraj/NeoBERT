import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch
from accelerate.utils import DistributedType
from torch.optim import Adam, AdamW

from .muon_clip import MuonClipConfig, MuonClipOptimizer

# from .soap.soap import SOAP  # TODO: Add SOAP optimizer implementation

logger = logging.getLogger(__name__)


def get_optimizer(
    model: torch.nn.Module,
    distributed_type: DistributedType,
    model_config=None,
    muon_config: Optional[Any] = None,
    **kwargs,
) -> torch.optim.Optimizer:
    """Optimizer.

    Args:
        model (torch.nn.Module): Model.
        distributed_type (DistributedType): Type of distributed training.
        model_config: Model configuration (REQUIRED for MuonClip).
        **kwargs: Optimizer-specific arguments.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.

    Raises:
        ValueError: If model_config not provided for MuonClip.
        ValueError: If unknown optimizer name.
    """
    optimizer_name = kwargs.pop("name").lower()

    logger.info(f"Initializing optimizer: {optimizer_name}")

    match optimizer_name:
        case "adamw":
            if muon_config is not None:
                logger.warning(
                    "optimizer.muon_config provided but optimizer is adamw; ignoring"
                )
            optimizer = AdamW(model.parameters(), **kwargs)
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
                f"  - Newton-Schulz steps: {muon_clip_cfg.ns_steps}\n"
                f"  - Alpha (Q/K balance): {muon_clip_cfg.clipping_alpha}"
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
