import torch
from accelerate.utils import DistributedType
from torch.optim import Adam, AdamW

# from .soap.soap import SOAP  # TODO: Add SOAP optimizer implementation


def get_optimizer(
    model: torch.nn.Module, distributed_type: DistributedType, **kwargs
) -> torch.optim.Optimizer:
    """Optimizer.

    Args:
        model (torch.nn.Module): Model.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    optimizer_name = kwargs.pop("name").lower()
    match optimizer_name:
        case "adamw":
            return AdamW(model.parameters(), **kwargs)
        case "adam":
            return Adam(model.parameters(), **kwargs)
        # case "SOAP":
        #     assert distributed_type is not DistributedType.DEEPSPEED, (
        #         "SOAP does not support DeepSpeed"
        #     )
        #     return SOAP(model.parameters(), **kwargs)
        case _:
            raise ValueError(
                "Unrecognized optimizer name. Options are: Adam, AdamW."  # SOAP not yet implemented
            )
