"""Sequence-classification heads for NeoBERT."""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput

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
        nn.init.normal_(self.dense.weight, mean=0.0, std=self.classifier_init_range)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        nn.init.normal_(
            self.classifier.weight,
            mean=0.0,
            std=self.classifier_init_range,
        )
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

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


class NeoBERTHFForSequenceClassification(_BaseSequenceClassifier):
    """Hugging Face compatible NeoBERT sequence classifier."""

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the HF-compatible classifier.

        :param NeoBERTConfig config: Model configuration.
        """
        local_config = _resolve_classifier_config(
            config,
            owner_name=type(self).__name__,
        )
        self._init_classifier_head(
            local_config,
            num_labels=getattr(local_config, "num_labels", 2),
            classifier_dropout=getattr(local_config, "classifier_dropout", 0.1),
            classifier_init_range=getattr(
                local_config,
                "classifier_init_range",
                0.02,
            ),
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput | tuple:
        """Forward pass for sequence classification.

        :param torch.Tensor | None input_ids: Input token IDs.
        :param torch.Tensor | None attention_mask: Attention mask.
        :param torch.Tensor | None token_type_ids: Token type IDs.
        :param torch.Tensor | None position_ids: Position IDs.
        :param torch.Tensor | None inputs_embeds: Optional input embeddings.
        :param torch.Tensor | None labels: Optional labels for loss.
        :param bool | None output_attentions: Whether to return attentions.
        :param bool | None output_hidden_states: Whether to return hidden states.
        :param bool | None return_dict: Whether to return dict outputs.
        :return SequenceClassifierOutput | tuple: Model outputs.
        """
        del token_type_ids, position_ids, inputs_embeds, output_attentions
        del output_hidden_states

        if attention_mask is not None:
            if attention_mask.dtype is torch.bool:
                additive_mask = torch.where(attention_mask, float(0.0), float("-inf"))
            elif attention_mask.is_floating_point() and attention_mask.min() < 0:
                additive_mask = attention_mask
            else:
                additive_mask = torch.where(
                    attention_mask == 0,
                    float("-inf"),
                    float(0.0),
                )
            if additive_mask.dtype != torch.float32:
                additive_mask = additive_mask.to(torch.float32)
        else:
            additive_mask = None

        hidden_representation = self.model.forward(input_ids, additive_mask)
        logits = self._classifier_logits(hidden_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )
