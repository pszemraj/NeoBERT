# Calculate expected values for CoLA
train_size = 8551
batch_size = 32
gradient_accumulation = 4
num_epochs = 3

effective_batch_size = batch_size * gradient_accumulation
num_update_steps_per_epoch = train_size // effective_batch_size
total_steps = num_update_steps_per_epoch * num_epochs

warmup_percent = 5  # 5%
warmup_steps = int(total_steps * warmup_percent / 100)

print(f"Train size: {train_size}")
print(f"Effective batch size: {effective_batch_size}")
print(f"Steps per epoch: {num_update_steps_per_epoch}")
print(f"Total steps: {total_steps}")
print(f"Warmup steps (5%): {warmup_steps}")
print(f"Decay steps (should be total): {total_steps}")
