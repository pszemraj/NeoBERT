import logging
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
            optimizer = AdamW(model.parameters(), **kwargs)
            logger.info(f"AdamW initialized with lr={kwargs.get('lr', 'default')}")
            return optimizer

        case "adam":
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
            muon_config = MuonClipConfig(
                lr=kwargs.pop("lr", 1e-4),
                muon_beta=kwargs.pop("muon_beta", 0.95),
                muon_decay=kwargs.pop("muon_decay", 0.0),
                ns_steps=kwargs.pop("ns_steps", 5),
                adam_betas=kwargs.pop("betas", (0.9, 0.95)),
                adam_decay=kwargs.pop("weight_decay", 0.0),
                adam_eps=kwargs.pop("eps", 1e-10),
                enable_clipping=kwargs.pop("enable_clipping", True),
                clipping_threshold=kwargs.pop("clipping_threshold", 50.0),
                clipping_alpha=kwargs.pop("clipping_alpha", 0.5),
                clipping_warmup_steps=kwargs.pop("clipping_warmup_steps", 0),
                monitor_attention_entropy=kwargs.pop("monitor_attention_entropy", True),
                detect_anomalies=kwargs.pop("detect_anomalies", False),
                log_max_logits=kwargs.pop("log_max_logits", True),
                log_interval=kwargs.pop("log_interval", 100),
                offload_hooks_to_cpu=kwargs.pop("offload_hooks_to_cpu", True),
                enable_profiling=kwargs.pop("enable_profiling", False),
            )

            logger.info(
                f"MuonClip configuration:\n"
                f"  - Learning rate: {muon_config.lr}\n"
                f"  - Clipping enabled: {muon_config.enable_clipping}\n"
                f"  - Clipping threshold: {muon_config.clipping_threshold}\n"
                f"  - Newton-Schulz steps: {muon_config.ns_steps}\n"
                f"  - Alpha (Q/K balance): {muon_config.clipping_alpha}"
            )

            return MuonClipOptimizer(model, model_config, muon_config)

        # case "SOAP":
        #     assert distributed_type is not DistributedType.DEEPSPEED, (
        #         "SOAP does not support DeepSpeed"
        #     )
        #     return SOAP(model.parameters(), **kwargs)

        case _:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw, muonclip"
            )
