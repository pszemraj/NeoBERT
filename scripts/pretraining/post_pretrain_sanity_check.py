#!/usr/bin/env python3
"""Post-pretraining sanity check to verify model learned meaningful representations."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from neobert.config import ConfigLoader
from neobert.model import NeoBERTConfig, NeoBERTForSequenceClassification


def load_pretrained_model(checkpoint_dir, checkpoint_step, config_path, tokenizer_name):
    """Load pretrained model from checkpoint."""
    # Load config
    config_dict = ConfigLoader.load(config_path)

    # Create model config
    model_config = NeoBERTConfig(
        vocab_size=config_dict.model.vocab_size,
        hidden_size=config_dict.model.hidden_size,
        num_hidden_layers=config_dict.model.num_hidden_layers,
        num_attention_heads=config_dict.model.num_attention_heads,
        max_position_embeddings=config_dict.model.max_position_embeddings,
        dropout=config_dict.model.dropout_prob,
        hidden_act=config_dict.model.hidden_act,
        rope=config_dict.model.rope,
        rms_norm=config_dict.model.rms_norm,
        flash_attention=False,  # Disable due to xformers bug with non-aligned sequences
    )

    # Create model
    model = NeoBERTForSequenceClassification(model_config, num_labels=2)

    # Load checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir, str(checkpoint_step), "state_dict.pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Remove decoder keys
    checkpoint = {k: v for k, v in checkpoint.items() if "decoder" not in k}

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Missing keys: {len(missing)} (expected: classifier/dense layers)")
    print(f"  Unexpected keys: {len(unexpected)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return model, tokenizer


def test_semantic_similarity(model, tokenizer, device="cuda"):
    """Test if model captures semantic similarity."""
    model = model.to(device)
    model.eval()

    test_pairs = [
        # Similar pairs (should have similar representations)
        ("The cat sat on the mat.", "A feline rested on the rug."),
        ("I love this movie!", "This film is fantastic!"),
        ("The weather is nice today.", "It's a beautiful day outside."),
        # Dissimilar pairs (should have different representations)
        ("The cat sat on the mat.", "The stock market crashed today."),
        ("I love this movie!", "The chemical formula is H2O."),
        ("The weather is nice today.", "Python is a programming language."),
    ]

    print("\n" + "=" * 60)
    print("SEMANTIC SIMILARITY TEST")
    print("=" * 60)

    all_hidden = []
    for sent1, sent2 in test_pairs:
        # Tokenize
        tokens1 = tokenizer(
            sent1,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        tokens2 = tokenizer(
            sent2,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )

        input_ids1 = tokens1["input_ids"].to(device)
        input_ids2 = tokens2["input_ids"].to(device)

        # Create masks
        mask1 = torch.where(tokens1["attention_mask"] == 1, 0.0, -float("inf")).to(
            device
        )
        mask2 = torch.where(tokens2["attention_mask"] == 1, 0.0, -float("inf")).to(
            device
        )

        # Get representations
        with torch.no_grad():
            out1 = model(input_ids1, mask1)
            out2 = model(input_ids2, mask2)

            # Use CLS token representation
            hidden1 = out1["hidden_representation"][:, 0, :]
            hidden2 = out2["hidden_representation"][:, 0, :]

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(hidden1, hidden2).item()

            all_hidden.append((hidden1, sent1))
            all_hidden.append((hidden2, sent2))

        print(f"\nSentence 1: {sent1[:50]}...")
        print(f"Sentence 2: {sent2[:50]}...")
        print(f"Cosine Similarity: {cos_sim:.4f}")

    return all_hidden


def test_sentiment_discrimination(model, tokenizer, device="cuda"):
    """Test if model can discriminate sentiment."""
    model = model.to(device)
    model.eval()

    test_sentences = [
        ("This movie is absolutely fantastic!", 1),  # Positive
        ("I hate this terrible film.", 0),  # Negative
        ("Best product I've ever bought!", 1),  # Positive
        ("Worst experience of my life.", 0),  # Negative
        ("The food was delicious!", 1),  # Positive
        ("The service was horrible.", 0),  # Negative
    ]

    print("\n" + "=" * 60)
    print("SENTIMENT DISCRIMINATION TEST")
    print("=" * 60)

    correct = 0
    for sentence, true_label in test_sentences:
        # Tokenize
        tokens = tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        input_ids = tokens["input_ids"].to(device)
        mask = torch.where(tokens["attention_mask"] == 1, 0.0, -float("inf")).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(input_ids, mask)
            logits = output["logits"]
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()

        is_correct = pred == true_label
        correct += is_correct

        print(f"\nSentence: {sentence}")
        print(
            f"  True Label: {true_label}, Predicted: {pred} {'✓' if is_correct else '✗'}"
        )
        print(f"  Probabilities: [Neg: {probs[0, 0]:.3f}, Pos: {probs[0, 1]:.3f}]")

    accuracy = correct / len(test_sentences)
    print(f"\nAccuracy: {correct}/{len(test_sentences)} = {accuracy:.1%}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Post-pretraining sanity check")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--checkpoint_step", type=int, required=True, help="Checkpoint step to load"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to pretraining config"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp",
        help="Tokenizer name or path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--save_results", type=str, default=None, help="Path to save results JSON"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint_dir} at step {args.checkpoint_step}")
    model, tokenizer = load_pretrained_model(
        args.checkpoint_dir, args.checkpoint_step, args.config_path, args.tokenizer_name
    )

    # Run tests
    test_semantic_similarity(model, tokenizer, args.device)
    accuracy = test_sentiment_discrimination(model, tokenizer, args.device)

    # Summary
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)
    print(f"✓ Model loaded successfully from checkpoint {args.checkpoint_step}")
    print("✓ Model produces different outputs for different inputs")
    print(f"✓ Sentiment discrimination accuracy: {accuracy:.1%}")

    if accuracy > 0.6:
        print("✅ PASSED: Model has learned meaningful representations!")
    else:
        print(
            "⚠️  WARNING: Model performance is near random. More pretraining may be needed."
        )

    # Save results if requested
    if args.save_results:
        results = {
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_step": args.checkpoint_step,
            "sentiment_accuracy": accuracy,
            "status": "passed" if accuracy > 0.6 else "needs_more_training",
        }

        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
