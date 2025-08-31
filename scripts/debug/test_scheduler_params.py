from neobert.config import ConfigLoader

config = ConfigLoader.load("configs/eval/evaluate_checkpoint.yaml")
print("Scheduler config:")
print(config.scheduler.__dict__)
print("\nTrainer config relevant fields:")
print(f"max_steps: {getattr(config.trainer, 'max_steps', None)}")
print(f"num_train_epochs: {getattr(config.trainer, 'num_train_epochs', None)}")
