#!/usr/bin/env python3
"""
Test SmolLM2 streaming dataset pretraining for 200 steps.
"""

import os

# Add to path
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent))

from src.neobert.collator import get_collator
from src.neobert.data import PretrainingDataModule
from src.neobert.model import NeoBERTConfig, NeoBERTLMHead
from src.neobert.scheduler import get_scheduler


def test_smollm2_pretraining():
    print("Testing SmolLM2 pretraining with streaming dataset...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set seed
    set_seed(42)

    # Configuration
    output_dir = "outputs/smollm2_streaming_test"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=output_dir,
        automatic_checkpoint_naming=True,
        total_limit=2,
    )

    accelerator = Accelerator(
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=4,
        log_with=None,  # Disable logging for test
        project_config=project_config,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Create model config
    config = NeoBERTConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="swiglu",
        max_position_embeddings=1024,
        rope=True,
        flash_attention=torch.cuda.is_available(),
        pad_token_id=tokenizer.pad_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    model = NeoBERTLMHead(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        lr=5e-5,
        warmup_steps=20,
        decay="cosine",
        decay_steps=180,  # 200 - 20 warmup
        final_lr_ratio=0.1,
    )

    # Create data module with SmolLM2 streaming dataset
    data_module = PretrainingDataModule(
        dataset_name="EleutherAI/SmolLM2-1.7B-stage-4-100B",
        dataset_split="train",
        tokenizer=tokenizer,
        max_length=1024,
        batch_size=4,
        num_workers=2,
        streaming=True,
        text_column="text",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Get collator
    collate_fn = get_collator(
        tokenizer,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        mlm_probability=0.15,
        max_length=1024,
    )

    # Get dataloader
    dataloader = data_module.get_dataloader(collate_fn)
    print("DataLoader created with SmolLM2 streaming dataset")

    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Loss function
    loss_fn = CrossEntropyLoss()

    # Training loop
    print("\nStarting 200-step pretraining...")
    model.train()

    pbar = tqdm(total=200, desc="Training", disable=not accelerator.is_main_process)
    step = 0

    for batch in dataloader:
        if step >= 200:
            break

        # Only update on gradient accumulation boundaries
        with accelerator.accumulate(model):
            # Forward pass
            with accelerator.autocast():
                outputs = model(
                    src=batch["input_ids"],
                    pad_mask=batch["attention_mask"],
                )

                # Calculate loss
                logits = outputs["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
                )

                # Scale loss by accumulation steps
                loss = loss / accelerator.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            # Only step optimizer on accumulation boundaries
            if accelerator.sync_gradients:
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update progress
                step += 1
                pbar.update(1)

                # Log metrics
                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * accelerator.gradient_accumulation_steps:.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

                # Save checkpoint
                if step % 50 == 0:
                    accelerator.save_state()
                    if accelerator.is_main_process:
                        print(f"\nCheckpoint saved at step {step}")

    pbar.close()

    # Final save
    accelerator.save_state()

    print("\n✓ SmolLM2 streaming pretraining completed successfully!")
    print(f"  - Trained for {step} steps")
    print(
        f"  - Final loss: {loss.item() * accelerator.gradient_accumulation_steps:.4f}"
    )

    # Test memory usage
    if torch.cuda.is_available():
        print(
            f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    test_smollm2_pretraining()
