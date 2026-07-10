"""Shared structural contracts for NeoBERT Hugging Face export tools."""

from collections.abc import Iterable

REQUIRED_HF_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "norm_eps",
    "pad_token_id",
    "rms_norm",
    "rope",
    "hidden_act",
    "dropout",
)


def has_packed_swiglu_weights(keys: Iterable[str]) -> bool:
    """Return whether state-dict keys use the unsupported packed SwiGLU layout.

    :param Iterable[str] keys: State-dict key names.
    :return bool: Whether any key belongs to a packed ``ffn.w12`` projection.
    """
    return any(".ffn.w12." in key for key in keys)
