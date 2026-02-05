"""Contrastive loss implementations."""

from typing import List, Optional, Union

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss wrapper."""

    def __init__(
        self,
        temperature: float = 0.07,
        similarity_fn: nn.Module = nn.CrossEntropyLoss(reduction="sum"),
    ) -> None:
        """Initialize the loss module.

        :param float temperature: Temperature scaling for cosine similarity.
        :param nn.Module similarity_fn: Loss function applied to logits.
        """
        super().__init__()

        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = similarity_fn

    def forward(
        self,
        queries: torch.Tensor,
        corpus: torch.Tensor,
        negatives: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        :param torch.Tensor queries: Query embeddings of shape ``[batch, hidden]``.
        :param torch.Tensor corpus: Positive embeddings of shape ``[batch, hidden]``.
        :param torch.Tensor | list[torch.Tensor] | None negatives: Optional hard negatives.
        :return torch.Tensor: Scalar loss value.
        """

        labels = torch.arange(queries.size(0)).long().to(queries.device)

        if negatives is not None:
            if isinstance(negatives, list):
                negatives = torch.cat(negatives, dim=0)
            corpus = torch.cat((corpus, negatives), dim=0)

        cos_sim = self.sim(queries.unsqueeze(1), corpus.unsqueeze(0)) / self.temperature

        loss = self.loss_fct(cos_sim, labels)

        return loss
