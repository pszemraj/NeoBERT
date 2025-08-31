#!/usr/bin/env python3
"""Evaluate a checkpoint on GLUE and MTEB tasks."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from neobert.config import ConfigLoader
from neobert.model import NeoBERT, NeoBERTConfig


def load_checkpoint(checkpoint_path, config):
    """Load a checkpoint and return the model."""
    from neobert.model import NeoBERTLMHead

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Check if this is a LMHead model (has decoder)
    has_decoder = any("decoder" in k for k in state_dict.keys())

    if has_decoder:
        # This is a NeoBERTLMHead checkpoint
        model = NeoBERTLMHead(config)
        # The state dict should match directly
        model.load_state_dict(state_dict)
        # Return just the base model for evaluation
        return model.model
    else:
        # This is a base NeoBERT checkpoint
        model = NeoBERT(config)

        # Remove 'model.' prefix if present
        if any(k.startswith("model.") for k in state_dict.keys()):
            # Remove the 'model.' prefix from all keys
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }

        model.load_state_dict(state_dict)
        return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on GLUE/MTEB")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint state_dict.pt"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config YAML"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["glue", "mteb", "both"],
        default="glue",
        help="Which evaluation to run",
    )
    parser.add_argument(
        "--glue-task",
        type=str,
        default="cola",
        help="GLUE task name (cola, sst2, mrpc, etc.)",
    )
    parser.add_argument(
        "--mteb-task",
        type=str,
        default=None,
        help="Specific MTEB task (if None, runs all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp",
        help="Tokenizer to use",
    )

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load(args.config)

    # Create model config from loaded config
    model_config = NeoBERTConfig(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        num_hidden_layers=config.model.num_hidden_layers,
        num_attention_heads=config.model.num_attention_heads,
        intermediate_size=config.model.intermediate_size,
        hidden_act=config.model.hidden_act,
        dropout=config.model.dropout_prob,
        max_length=config.model.max_position_embeddings,
        rope=config.model.rope,
        rms_norm=config.model.rms_norm,
        flash_attention=config.model.flash_attention,
        ngpt=getattr(config.model, "ngpt", False),
    )

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, model_config)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded: {args.tokenizer}")

    # Save model in HuggingFace format for easier evaluation
    save_dir = Path(args.output_dir) / "model"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model and config
    model.save_pretrained(save_dir)
    model_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Model saved to {save_dir}")

    if args.task in ["glue", "both"]:
        print(f"\nRunning GLUE evaluation on {args.glue_task}...")
        # Create GLUE evaluation command
        glue_cmd = f"""
        python scripts/evaluation/run_glue.py \\
            --config configs/evaluate_neobert.yaml \\
            --task_name {args.glue_task} \\
            --model_name_or_path {save_dir} \\
            --trainer.output_dir {args.output_dir}/glue_{args.glue_task}
        """
        print(f"Run this command:\n{glue_cmd}")

    if args.task in ["mteb", "both"]:
        print("\nRunning MTEB evaluation...")
        # For MTEB, we need a different approach
        print("Setting up MTEB evaluation...")

        from neobert.model import NeoBERTForMTEB

        # Create MTEB wrapper
        mteb_model = NeoBERTForMTEB(
            config=model_config,
            tokenizer=tokenizer,
            max_length=512,
            batch_size=32,
            pooling="avg",
        )

        # Load the state dict into the MTEB model
        mteb_model.model.load_state_dict(model.state_dict())
        mteb_model.model.to(device)

        # Save for MTEB script
        mteb_save_dir = Path(args.output_dir) / "mteb_model"
        mteb_save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": mteb_model.model.state_dict(),
                "config": model_config,
                "tokenizer_name": args.tokenizer,
            },
            mteb_save_dir / "mteb_checkpoint.pt",
        )

        print(f"MTEB model prepared at {mteb_save_dir}")

        # Run MTEB evaluation
        from mteb import MTEB

        if args.mteb_task:
            tasks = [args.mteb_task]
        else:
            # Run a subset of tasks for testing
            tasks = ["Banking77Classification"]  # Start with one task

        evaluation = MTEB(tasks=tasks)
        results = evaluation.run(
            mteb_model,
            output_folder=str(Path(args.output_dir) / "mteb_results"),
            eval_splits=["test"],
        )

        print("MTEB Results:", results)


if __name__ == "__main__":
    main()
