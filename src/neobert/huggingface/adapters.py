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
from neobert.utils import additive_attention_mask


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
        :raises NotImplementedError: If unsupported embeddings, positions,
            segment IDs, attentions, or hidden states are requested.
        :raises ValueError: If the attention mask is ambiguous or contains NaN.
        :return SequenceClassifierOutput | tuple: Model outputs.
        """
        if inputs_embeds is not None:
            raise NotImplementedError(
                "NeoBERTHFForSequenceClassification does not support inputs_embeds. "
                "Pass input_ids, or route through the export HF model."
            )
        if output_attentions:
            raise NotImplementedError(
                "NeoBERTHFForSequenceClassification does not expose attentions. "
                "Use the export HF model for output_attentions=True."
            )
        if output_hidden_states:
            raise NotImplementedError(
                "NeoBERTHFForSequenceClassification does not expose all-layer hidden "
                "states. Use the export HF model for output_hidden_states=True."
            )
        if position_ids is not None:
            raise NotImplementedError(
                "NeoBERTHFForSequenceClassification does not support custom "
                "position_ids. Use the export HF model for position remapping."
            )
        if token_type_ids is not None and torch.any(token_type_ids != 0):
            raise NotImplementedError(
                "NeoBERTHFForSequenceClassification has no token-type embeddings; "
                "only omitted or all-zero token_type_ids are supported."
            )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if attention_mask is not None:
            if attention_mask.is_floating_point() and torch.isnan(attention_mask).any():
                raise ValueError("attention_mask must not contain NaN values.")
            if attention_mask.is_floating_point() and torch.any(attention_mask < 0):
                if torch.any(attention_mask > 0):
                    raise ValueError(
                        "A floating attention_mask containing negative additive "
                        "values must not also contain positive binary values."
                    )
                # Explicit additive masks are nonpositive (normally 0/-inf).
                additive_mask = attention_mask.to(torch.float32)
            else:
                # HF convention: 1/True keeps and 0/False masks. In particular,
                # an all-zero float mask means mask everything, not keep all.
                additive_mask = additive_attention_mask(attention_mask)
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

        # hidden_states must be None or a tuple of per-layer tensors per the HF
        # contract; the training-time backbone returns only the final tensor, so
        # expose nothing here rather than mislabel it (see output_hidden_states
        # guard above).
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
