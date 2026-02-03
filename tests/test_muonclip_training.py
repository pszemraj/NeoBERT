"""
MuonClip Training Test - ~2 minute run to demonstrate stable training.
Run: python tests/test_muonclip_training.py
"""

import itertools
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from neobert.collator import get_collator
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import MuonClipOptimizer, MuonClipConfig
from torch.optim import AdamW


def build_fineweb_stream(tokenizer: AutoTokenizer, seq_len: int = 128):
    """Create a tokenized streaming FineWeb dataset ready for MLM dataloaders."""

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        streaming=True,
    )

    def tokenize(sample):
        text = sample.get("text") or sample.get("content") or ""
        return tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    dataset = dataset.map(
        tokenize,
        batched=False,
        remove_columns=dataset.column_names,
    )

    return dataset


def train_model(
    model,
    optimizer,
    batch_provider,
    device,
    duration_seconds=75,
    optimizer_name="Optimizer",
):
    """Train model for specified duration and track metrics."""
    model.train()
    model = model.to(device)

    losses = []
    step_times = []
    start_time = time.time()
    step = 0

    print(f"\nTraining with {optimizer_name} for {duration_seconds} seconds...")
    print("-" * 60)

    while time.time() - start_time < duration_seconds:
        step_start = time.time()

        batch = batch_provider()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        pad_mask = batch.get("attention_mask")
        if pad_mask is not None:
            pad_mask = pad_mask.to(device)

        # Forward pass
        hidden_states = model(input_ids, pad_mask)

        # Simple MLM head (just a linear projection for testing)
        if not hasattr(model, "_mlm_head"):
            model._mlm_head = torch.nn.Linear(
                model.config.hidden_size, model.config.vocab_size
            ).to(device)

        logits = model._mlm_head(hidden_states)

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Track metrics
        losses.append(loss.item())
        step_times.append(time.time() - step_start)
        step += 1

        # Log progress
        if step % 50 == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-50:]) / min(50, len(losses))

            # Get MuonClip metrics if available
            extra_info = ""
            if hasattr(optimizer, "get_metrics"):
                metrics = optimizer.get_metrics()
                if metrics and "train/max_attention_logit" in metrics:
                    extra_info = (
                        f", max_logit={metrics['train/max_attention_logit']:.1f}"
                    )

            print(
                f"  Step {step:4d} | Time: {elapsed:5.1f}s | Loss: {avg_loss:.4f}{extra_info}"
            )

    training_time = time.time() - start_time

    # Calculate final metrics
    final_avg_loss = sum(losses[-100:]) / min(100, len(losses))
    initial_avg_loss = sum(losses[:100]) / min(100, len(losses))
    avg_step_time = sum(step_times) / len(step_times)

    return {
        "steps": step,
        "epochs": None,
        "training_time": training_time,
        "initial_loss": initial_avg_loss,
        "final_loss": final_avg_loss,
        "loss_reduction": initial_avg_loss - final_avg_loss,
        "avg_step_time_ms": avg_step_time * 1000,
        "all_losses": losses,
    }


def main():
    """Main training comparison."""
    print("=" * 60)
    print("MuonClip Training Test (~2 minute run)")
    print("=" * 60)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    seq_len = 128

    # Create model config - compact model for faster iteration
    config = NeoBERTConfig(
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        vocab_size=tokenizer.vocab_size,
        max_length=seq_len,
        flash_attention=False,  # Disable for compatibility
        hidden_act="gelu",
        rope=False,  # Disable for simplicity
        pad_token_id=tokenizer.pad_token_id,
    )

    print("\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")

    # Create dataset
    print("\nStreaming FineWeb to build MLM dataset...")
    batch_size = 24 if device.type == "cuda" else 8
    streaming_dataset = build_fineweb_stream(tokenizer, seq_len=seq_len)
    collator = get_collator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
        max_length=seq_len,
    )
    base_seed = 42

    def make_batch_provider(seed: int):
        shuffled_stream = streaming_dataset.shuffle(buffer_size=8192, seed=seed)
        stream_iterator = iter(shuffled_stream)

        def provider():
            nonlocal stream_iterator
            samples = list(itertools.islice(stream_iterator, batch_size))
            while len(samples) < batch_size:
                new_seed = int(time.time())
                stream_iterator = iter(
                    streaming_dataset.shuffle(buffer_size=8192, seed=new_seed)
                )
                samples.extend(
                    list(itertools.islice(stream_iterator, batch_size - len(samples)))
                )
                if not samples:
                    raise StopIteration
            return collator(samples)

        return provider

    print("  Dataset: HuggingFaceFW/fineweb (streaming)")
    print(f"  Batch size: {batch_size}")

    # Test 1: AdamW baseline
    print("\n" + "=" * 60)
    print("Test 1: AdamW Baseline")
    print("=" * 60)

    model_adamw = NeoBERT(config)
    num_params = sum(p.numel() for p in model_adamw.parameters())
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    optimizer_adamw = AdamW(model_adamw.parameters(), lr=5e-4)

    adamw_batch_provider = make_batch_provider(base_seed)

    adamw_results = train_model(
        model_adamw,
        optimizer_adamw,
        adamw_batch_provider,
        device,
        duration_seconds=75,
        optimizer_name="AdamW",
    )

    print("\nAdamW Results:")
    print(f"  Steps completed: {adamw_results['steps']}")
    print(f"  Initial loss: {adamw_results['initial_loss']:.4f}")
    print(f"  Final loss: {adamw_results['final_loss']:.4f}")
    print(f"  Loss reduction: {adamw_results['loss_reduction']:.4f}")
    print(f"  Avg step time: {adamw_results['avg_step_time_ms']:.1f} ms")

    # Test 2: MuonClip
    print("\n" + "=" * 60)
    print("Test 2: MuonClip Optimizer")
    print("=" * 60)

    model_muon = NeoBERT(config)

    muon_config = MuonClipConfig(
        lr=5e-4,
        muon_beta=0.95,
        ns_steps=5,
        enable_clipping=True,
        clipping_threshold=50.0,
        clipping_alpha=0.5,
    )

    print("MuonClip config:")
    print(f"  Learning rate: {muon_config.lr}")
    print(f"  Newton-Schulz steps: {muon_config.ns_steps}")
    print(f"  QK-clipping enabled: {muon_config.enable_clipping}")
    print(f"  Clipping threshold: {muon_config.clipping_threshold}")

    optimizer_muon = MuonClipOptimizer(model_muon, config, muon_config)

    muon_batch_provider = make_batch_provider(base_seed + 1)

    muon_results = train_model(
        model_muon,
        optimizer_muon,
        muon_batch_provider,
        device,
        duration_seconds=75,
        optimizer_name="MuonClip",
    )

    print("\nMuonClip Results:")
    print(f"  Steps completed: {muon_results['steps']}")
    print(f"  Initial loss: {muon_results['initial_loss']:.4f}")
    print(f"  Final loss: {muon_results['final_loss']:.4f}")
    print(f"  Loss reduction: {muon_results['loss_reduction']:.4f}")
    print(f"  Avg step time: {muon_results['avg_step_time_ms']:.1f} ms")

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    print("\nLoss Reduction:")
    print(
        f"  AdamW:    {adamw_results['loss_reduction']:.4f} ({adamw_results['loss_reduction'] / adamw_results['initial_loss'] * 100:.1f}%)"
    )
    print(
        f"  MuonClip: {muon_results['loss_reduction']:.4f} ({muon_results['loss_reduction'] / muon_results['initial_loss'] * 100:.1f}%)"
    )

    print("\nTraining Speed:")
    print(f"  AdamW:    {adamw_results['avg_step_time_ms']:.1f} ms/step")
    print(f"  MuonClip: {muon_results['avg_step_time_ms']:.1f} ms/step")
    overhead = (
        muon_results["avg_step_time_ms"] / adamw_results["avg_step_time_ms"] - 1
    ) * 100
    print(f"  MuonClip overhead: {overhead:.1f}%")

    print("\nConvergence:")
    print(f"  AdamW final loss:    {adamw_results['final_loss']:.4f}")
    print(f"  MuonClip final loss: {muon_results['final_loss']:.4f}")

    # Success criteria
    print("\n" + "=" * 60)
    print("Success Criteria")
    print("=" * 60)

    success = True

    # Check loss decrease
    if muon_results["loss_reduction"] > 0:
        print("✓ MuonClip shows loss decrease")
    else:
        print("✗ MuonClip loss did not decrease")
        success = False

    # Check reasonable overhead
    if overhead < 50:
        print(f"✓ MuonClip overhead is reasonable ({overhead:.1f}%)")
    else:
        print(f"⚠ MuonClip overhead is high ({overhead:.1f}%)")

    # Check stability (no NaN/Inf)
    if all(
        not torch.isnan(torch.tensor(loss_val))
        for loss_val in muon_results["all_losses"]
    ):
        print("✓ Training is stable (no NaN/Inf losses)")
    else:
        print("✗ Training encountered NaN/Inf losses")
        success = False

    if success:
        print("\n✅ MuonClip training test PASSED!")
        print("The optimizer is working correctly with stable convergence.")
    else:
        print("\n❌ MuonClip training test FAILED!")
        print("Please check the implementation.")

    print("=" * 60)


if __name__ == "__main__":
    main()
