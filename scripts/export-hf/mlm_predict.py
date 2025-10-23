#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_parser() -> argparse.ArgumentParser:
    """Get command-line parser."""

    parser = argparse.ArgumentParser(
        description="Predict masked tokens with a Hugging Face Masked LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_name_or_path",
        nargs="?",
        default="chandar-lab/NeoBERT",
        help="Model model_name_or_path on Hugging Face Hub.",
    )
    parser.add_argument(
        "--text",
        default="NeoBERT is the most [MASK] model of its kind!",
        help="Input sentence; you may use literal [MASK] which will be mapped to the tokenizer's mask token.",
    )
    return parser


def main():
    """Main entry point."""

    args = get_parser().parse_args()

    print(f"Loading model and tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="cpu",  # for simplicity; change as needed
        trust_remote_code=True,
    )

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

    # Handle Metaspace tokenizer issue: remove extra space tokens before [MASK]
    # Get the space token ID dynamically instead of hardcoding 454
    space_token_id = tokenizer.convert_tokens_to_ids("▁")

    input_ids = inputs["input_ids"][0].tolist()
    cleaned_ids = []
    for i, token_id in enumerate(input_ids):
        # Skip space token if it's immediately before [MASK]
        if (
            token_id == space_token_id
            and i < len(input_ids) - 1
            and input_ids[i + 1] == tokenizer.mask_token_id
        ):
            continue
        cleaned_ids.append(token_id)

    # Update inputs if we removed any tokens
    if len(cleaned_ids) != len(input_ids):
        inputs["input_ids"] = torch.tensor([cleaned_ids])
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    with torch.no_grad():
        outputs = model(**inputs)

    # Find mask positions
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=False
    )
    if mask_positions.numel() == 0:
        raise ValueError("No mask token found after tokenization (unexpected).")

    # Predict each mask independently (single forward pass)
    predicted_tokens = []
    top_k = 5

    for mask_num, (b_idx, t_idx) in enumerate(mask_positions.tolist(), 1):
        logits = outputs.logits[b_idx, t_idx]

        # Get top-k predictions
        scores, indices = torch.topk(logits, k=top_k)
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
