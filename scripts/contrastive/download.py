"""Download and cache contrastive datasets from the Hugging Face Hub."""

from neobert.contrastive.datasets import CONTRASTIVE_DATASETS

for name, dataset_cls in CONTRASTIVE_DATASETS.items():
    print(f"Downloading {name}...")
    _ = dataset_cls().dataset
