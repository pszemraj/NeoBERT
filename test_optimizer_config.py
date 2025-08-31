from neobert.config import ConfigLoader

config = ConfigLoader.load("configs/eval/evaluate_checkpoint.yaml")
print("Optimizer config:")
print(config.optimizer.__dict__)
