"""Hugging Face-compatible adapters for training-time NeoBERT modules."""

from typing import Optional

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput

from neobert.model.classification import (
    _BaseSequenceClassifier,
    _resolve_classifier_config,
)
from neobert.model.model import NeoBERTConfig


class NeoBERTHFForSequenceClassification(_BaseSequenceClassifier):
    """Hugging Face-compatible wrapper around the training-time classifier."""

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
            classifier_init_range=getattr(local_config, "classifier_init_range", 0.02),
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
