#!/usr/bin/env python3
"""Fill mask tokens with predictions from a Hugging Face masked language model."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict masked tokens with a Hugging Face Masked LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_name_or_path",
        nargs="?",
        default="chandar-lab/NeoBERT",
        help="Model name or path on the Hugging Face Hub (or local directory).",
    )
    parser.add_argument(
        "--text",
        default="NeoBERT is the most [MASK] model of its kind!",
        help="Input sentence; you may use literal [MASK] which will be mapped to the tokenizer's mask token.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates to show per mask token.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Where to run the model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom modeling code from the Hub (only enable for trusted repos).",
    )
    return parser


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _clean_metaspace_before_mask(inputs: dict, tokenizer) -> dict:
    """Remove SentencePiece metaspace tokens that appear directly before a mask token."""

    metaspace = "▁"
    metaspace_id = tokenizer.convert_tokens_to_ids(metaspace)
    if tokenizer.convert_ids_to_tokens(metaspace_id) != metaspace:
        return inputs

    input_ids = inputs["input_ids"][0].tolist()
    keep_indices = []
    for idx, token_id in enumerate(input_ids):
        if (
            token_id == metaspace_id
            and idx < len(input_ids) - 1
            and input_ids[idx + 1] == tokenizer.mask_token_id
        ):
            continue
        keep_indices.append(idx)

    if len(keep_indices) == len(input_ids):
        return inputs

    keep = torch.tensor(keep_indices, dtype=torch.long)
    cleaned = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != 1:
            cleaned[key] = value
            continue
        cleaned[key] = value[:, keep]
    return cleaned


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)

    print(f"Loading model and tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    # Validate mask token
    mask_token = tokenizer.mask_token
    if not mask_token:
        raise ValueError("This tokenizer/model does not define a mask token.")

    # Normalize: map literal "[MASK]" to the tokenizer's actual mask token (e.g., <mask>)
    text_norm = args.text.replace("[MASK]", mask_token)

    if mask_token not in text_norm:
        raise ValueError(
            f"Input text must include at least one mask token. Use literal [MASK] "
            f"or {mask_token} for this tokenizer."
        )

    # Tokenize & forward
    inputs = tokenizer(text_norm, return_tensors="pt")
    inputs = _clean_metaspace_before_mask(inputs, tokenizer)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    # Find mask positions
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=False
    )
    if mask_positions.numel() == 0:
        raise ValueError("No mask token found after tokenization (unexpected).")

    # Predict each mask independently (single forward pass)
    predicted_tokens = []
    top_k = max(1, args.top_k)

    for mask_num, (b_idx, t_idx) in enumerate(mask_positions.tolist(), 1):
        logits = outputs.logits[b_idx, t_idx]

        # Get top-k predictions
        scores, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        probs = torch.softmax(scores, dim=0)  # Convert to probabilities

        # Store top prediction for filling
        pred_tok = tokenizer.decode([int(indices[0])]).strip()
        predicted_tokens.append(pred_tok)

        # Display top-k for this mask
        print(f"\nTop {top_k} predictions for [MASK] #{mask_num}:")
        for rank, (idx, prob) in enumerate(zip(indices.tolist(), probs.tolist()), 1):
            token = tokenizer.decode([idx]).strip()
            print(f"  {rank}. {token:20s} (id {idx:6d}): {prob:.4f}")

    # Build filled sentence by replacing masks with top predictions
    filled = text_norm
    for pred in predicted_tokens:
        filled = filled.replace(mask_token, pred, 1)

    # Output summary
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Input: {args.text}")
    if args.text != text_norm:
        print(f"Normalized: {text_norm}")
    print(f"Filled: {filled}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
