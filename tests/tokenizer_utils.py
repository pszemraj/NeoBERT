"""Shared tokenizer builders for tests."""

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def build_wordlevel_tokenizer(
    vocab: dict[str, int] | None = None,
    *,
    padding_side: str = "right",
    include_mask: bool = True,
    include_sep: bool = True,
    include_cls: bool = False,
) -> PreTrainedTokenizerFast:
    """Build a tiny word-level tokenizer for tests.

    :param dict[str, int] | None vocab: Optional extra vocab entries.
    :param str padding_side: Tokenizer padding side.
    :param bool include_mask: Include ``[MASK]`` special token.
    :param bool include_sep: Include ``[SEP]`` special token.
    :param bool include_cls: Include ``[CLS]`` special token.
    :return PreTrainedTokenizerFast: Tokenizer instance.
    """
    merged = {"[PAD]": 0, "[UNK]": 1}
    if include_mask:
        merged["[MASK]"] = len(merged)
    if include_sep:
        merged["[SEP]"] = len(merged)
    if include_cls:
        merged["[CLS]"] = len(merged)
    merged.update(vocab or {"hello": len(merged), "world": len(merged) + 1})

    tokenizer = Tokenizer(models.WordLevel(merged, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    kwargs = {
        "tokenizer_object": tokenizer,
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }
    if include_mask:
        kwargs["mask_token"] = "[MASK]"
    if include_sep:
        kwargs["sep_token"] = "[SEP]"
    if include_cls:
        kwargs["cls_token"] = "[CLS]"

    fast = PreTrainedTokenizerFast(**kwargs)
    fast.padding_side = padding_side
    return fast
