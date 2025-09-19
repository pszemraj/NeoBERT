#!/usr/bin/env python3
"""
Test SmolLM2 streaming dataset pretraining for 200 steps.
Relocated from scripts/tests/ to tests/scripts/ to keep tests under tests/.
"""

import os

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from neobert.collator import get_collator
from neobert.data import PretrainingDataModule
from neobert.model import NeoBERTConfig, NeoBERTLMHead
from neobert.scheduler import get_scheduler


def test_smollm2_pretraining():
    print("Testing SmolLM2 pretraining with streaming dataset...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(42)

    output_dir = "outputs/smollm2_streaming_test"
    os.makedirs(output_dir, exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=output_dir,
        automatic_checkpoint_naming=True,
        total_limit=2,
    )

    accelerator = Accelerator(
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=4,
        log_with=None,
        project_config=project_config,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")

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

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    scheduler = get_scheduler(
        optimizer=optimizer,
        lr=5e-5,
        warmup_steps=20,
        decay="cosine",
        decay_steps=180,
        final_lr_ratio=0.1,
    )

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

    collate_fn = get_collator(
        tokenizer,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        mlm_probability=0.15,
        max_length=1024,
    )

    dataloader = data_module.get_dataloader(collate_fn)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    loss_fn = CrossEntropyLoss()

    print("\nStarting 200-step pretraining...")
    model.train()

    pbar = tqdm(total=200, desc="Training", disable=not accelerator.is_main_process)
    step = 0

    for batch in dataloader:
        if step >= 200:
            break

        with accelerator.accumulate(model):
            with accelerator.autocast():
                outputs = model(
                    src=batch["input_ids"],
                    pad_mask=batch["attention_mask"],
                )

                logits = outputs["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
                )

                loss = loss / accelerator.gradient_accumulation_steps

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                pbar.update(1)

    pbar.close()

    accelerator.save_state()
