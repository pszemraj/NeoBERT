"""Sequence-classification heads for NeoBERT."""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from torch import nn

from .model import NeoBERT, NeoBERTConfig, NeoBERTPreTrainedModel, NormNeoBERT

logger = logging.getLogger(__name__)


def _resolve_classifier_config(
    config: NeoBERTConfig,
    *,
    owner_name: str,
) -> NeoBERTConfig:
    """Clone a classifier config and normalize unsupported attention backends.

    :param NeoBERTConfig config: Incoming model configuration.
    :param str owner_name: Class name used for warning text.
    :return NeoBERTConfig: Local config copy for the classifier instance.
    """
    local_config = deepcopy(config)
    if local_config.attn_backend != "sdpa":
        logger.warning(
            "%s does not support packed attention; forcing attn_backend='sdpa' "
            "for this instance.",
            owner_name,
        )
        local_config.attn_backend = "sdpa"
    return local_config


class _BaseSequenceClassifier(NeoBERTPreTrainedModel):
    """Shared sequence-classification backbone and head wiring."""

    def _init_classifier_head(
        self,
        config: NeoBERTConfig,
        *,
        num_labels: int,
        classifier_dropout: float,
        classifier_init_range: float,
    ) -> None:
        """Initialize the shared classifier backbone and head layers.

        :param NeoBERTConfig config: Classifier-local config.
        :param int num_labels: Number of output labels.
        :param float classifier_dropout: Dropout probability.
        :param float classifier_init_range: Init range for head layers.
        """
        super().__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.classifier_init_range = classifier_init_range
        self.model = NormNeoBERT(config) if config.ngpt else NeoBERT(config)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.post_init()
        for layer in (self.dense, self.classifier):
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=self.classifier_init_range,
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _classifier_logits(self, hidden_representation: torch.Tensor) -> torch.Tensor:
        """Pool the summary token and produce classifier logits.

        :param torch.Tensor hidden_representation: Encoder hidden states.
        :return torch.Tensor: Sequence-classification logits.
        """
        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.classifier(x)


class NeoBERTForSequenceClassification(_BaseSequenceClassifier):
    """NeoBERT with a training-oriented classification head."""

    def __init__(
        self,
        config: NeoBERTConfig,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_init_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        """Initialize the sequence classification head.

        :param NeoBERTConfig config: Model configuration.
        :param int num_labels: Number of output labels.
        :param float classifier_dropout: Dropout probability.
        :param float classifier_init_range: Init range for classifier.
        :param Any kwargs: Unused extra arguments for compatibility.
        """
        del kwargs
        local_config = _resolve_classifier_config(
            config,
            owner_name=type(self).__name__,
        )
        self._init_classifier_head(
            local_config,
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            classifier_init_range=classifier_init_range,
        )

    def forward(
        self,
        src: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the classification head forward pass.

        :param torch.Tensor src: Input token IDs.
        :param torch.Tensor | None pad_mask: Additive attention mask.
        :return dict[str, torch.Tensor]: Hidden states and logits.
        """
        hidden_representation = self.model.forward(src, pad_mask)
        logits = self._classifier_logits(hidden_representation)
        return {"hidden_representation": hidden_representation, "logits": logits}
