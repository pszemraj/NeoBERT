import hydra
from omegaconf import DictConfig

from datasets import load_dataset, concatenate_datasets

from neobert.tokenizer import get_tokenizer, tokenize


@hydra.main(version_base=None, config_path="../../conf", config_name="pretraining")
def preprocess(cfg: DictConfig):
    # Tokenizer
    tokenizer = get_tokenizer(**cfg.tokenizer)
    print(tokenizer)

    # Load and tokenize the dataset
    print("Loading dataset")
    if cfg.dataset.name == "wikibook":
        bookcorpus = load_dataset("bookcorpus", split="train")
        wiki = load_dataset("wikipedia", "20220301.en", split="train")
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

        assert bookcorpus.features.type == wiki.features.type
        dataset = concatenate_datasets([bookcorpus, wiki])
        dataset = dataset.shuffle(seed=0)
    else:
        dataset = load_dataset(**cfg.dataset.train)

    print("Tokenizing dataset")
    dataset = tokenize(dataset, tokenizer, column_name=cfg.dataset.column, **cfg.tokenizer)

    # Save the tokenized dataset to disk
    print("Saving tokenized dataset")
    dataset.save_to_disk(cfg.dataset.path_to_disk, max_shard_size="1GB")


if __name__ == "__main__":
    preprocess()
