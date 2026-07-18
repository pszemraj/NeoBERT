"""Shared parameter-classification helpers for optimizer groups."""

import torch
from torch import nn


def embedding_parameter_ids(model: nn.Module) -> set[int]:
    """Return parameter identities owned directly by embedding modules.

    :param nn.Module model: Model whose embeddings should be inspected.
    :return set[int]: Object identities for embedding parameters.
    """
    return {
        id(param)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for param in module.parameters(recurse=False)
    }


def uses_weight_decay(
    name: str,
    param: torch.nn.Parameter,
    embedding_param_ids: set[int],
) -> bool:
    """Return whether a parameter follows the repository's AdamW decay policy.

    :param str name: Fully qualified parameter name.
    :param torch.nn.Parameter param: Parameter to classify.
    :param set[int] embedding_param_ids: Identities belonging to embeddings.
    :return bool: Whether decoupled weight decay should apply.
    """
    name_lower = name.lower()
    return not (
        param.ndim < 2
        or name_lower.endswith(".bias")
        or "norm" in name_lower
        or id(param) in embedding_param_ids
    )
