from datasets import load_dataset

tasks = ["cola", "sst2", "mrpc"]
for task in tasks:
    dataset = load_dataset("glue", task)
    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    print(f"\n{task.upper()}:")
    print(f"  Train: {train_size:,} samples")
    print(f"  Val: {val_size:,} samples")

    # Calculate steps per epoch with batch_size=16 (from config)
    batch_size = 16
    steps_per_epoch = train_size // batch_size
    print(f"  Steps per epoch (batch_size={batch_size}): {steps_per_epoch}")
    print(f"  For 3 epochs: {steps_per_epoch * 3} total steps")
