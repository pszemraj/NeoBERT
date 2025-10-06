from datasets import concatenate_datasets, load_dataset

from neobert.config import load_config_from_args
from neobert.tokenizer import get_tokenizer, tokenize


def preprocess(cfg):
    # Tokenizer
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
        vocab_size=cfg.tokenizer.vocab_size or cfg.model.vocab_size,
    )
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
        dataset = load_dataset(cfg.dataset.name, split="train")

    print("Tokenizing dataset")
    dataset = tokenize(
        dataset,
        tokenizer,
        column_name="text",  # Default column name
        max_length=cfg.tokenizer.max_length,
        padding=cfg.tokenizer.padding,
        truncation=cfg.tokenizer.truncation,
    )

    # Save the tokenized dataset to disk
    print("Saving tokenized dataset")
    dataset.save_to_disk(cfg.dataset.path, max_shard_size="1GB")


def main():
    # Load configuration from command line arguments
    config = load_config_from_args()

    # Run preprocessing
    preprocess(config)


if __name__ == "__main__":
    main()
