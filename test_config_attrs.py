from neobert.config import ConfigLoader

config = ConfigLoader.load("configs/eval_checkpoint_100k.yaml")
print("Model config attributes:")
print(config.model.__dict__)
print("\nTokenizer config attributes:")
print(config.tokenizer.__dict__)
