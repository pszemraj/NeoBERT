import torch
import torch.nn as nn

from typing import Optional, Union, List


class SupConLoss(nn.Module):

    def __init__(self, temperature: float = 0.07, similarity_fn=nn.CrossEntropyLoss(reduction="sum")):
        super().__init__()

        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = similarity_fn

    def forward(self, queries: torch.Tensor, corpus: torch.Tensor, negatives: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None):
        """Compute loss for model.
        Args:
            queries: vector of shape [batch_size, hidden_size].
            corpus: vector of shape [batch_size, hidden_size].
            negatives: vector of shape [..., hidden_size]. Optional additional hard negatives for the queries.
        Returns:
            A loss scalar.
        """

        labels = torch.arange(queries.size(0)).long().to(queries.device)

        if negatives is not None:
            if isinstance(negatives, list):
                negatives = torch.cat(negatives, dim=0)
            corpus = torch.cat((corpus, negatives), dim=0)

        cos_sim = self.sim(queries.unsqueeze(1), corpus.unsqueeze(0)) / self.temperature

        loss = self.loss_fct(cos_sim, labels)

        return loss
