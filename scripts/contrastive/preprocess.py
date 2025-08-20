import json
import os
import shutil

from datasets import DatasetDict, load_from_disk

from neobert.config import load_config_from_args
from neobert.contrastive.datasets import (
    ALLNLI,
    AMAZONQA,
    CONCURRENTQA,
    FEVER,
    GITHUBISSUE,
    GOOAQ,
    MSMARCO,
    PAQ,
    PUBMEDQA,
    QQP,
    SENTENCECOMP,
    STACKEXCHANGE,
    STACKOVERFLOW,
    STS12,
    STSBENCHMARK,
    TRIVIAQA,
    WIKIHOW,
)
from neobert.tokenizer import get_tokenizer, tokenize

DATASETS = {
    "ALLNLI": ALLNLI,
    "AMAZONQA": AMAZONQA,
    "CONCURRENTQA": CONCURRENTQA,
    "FEVER": FEVER,
    "GITHUBISSUE": GITHUBISSUE,
    "GOOAQ": GOOAQ,
    "MSMARCO": MSMARCO,
    "PAQ": PAQ,
    "PUBMEDQA": PUBMEDQA,
    "QQP": QQP,
    "SENTENCECOMP": SENTENCECOMP,
    "STACKEXCHANGE": STACKEXCHANGE,
    "STACKOVERFLOW": STACKOVERFLOW,
    "STS12": STS12,
    "STSBENCHMARK": STSBENCHMARK,
    "TRIVIAQA": TRIVIAQA,
    "WIKIHOW": WIKIHOW,
}


def pipeline(cfg):
    if cfg.datasets.load_all_from_disk:
        dataset = load_from_disk(os.path.join(cfg.datasets.path, "all"))

    else:
        # Tokenizer
        tokenizer = get_tokenizer(**cfg.tokenizer)

        # Load and tokenize subdatasets if necessary
        dataset_dict = {}
        for name in DATASETS.keys():
            if (
                os.path.isdir(os.path.join(cfg.datasets.path, "all", name))
                and not cfg.datasets.force_redownload
            ):
                print(f"Loading tokenized {name} from disk...")
                subdataset = (
                    DATASETS[name]
                    .from_disk(os.path.join(cfg.datasets.path, "all", name))
                    .dataset
                )
            else:
                if os.path.isdir(os.path.join(cfg.datasets.path, "all", name)):
                    shutil.rmtree(os.path.join(cfg.datasets.path, "all", name))
                print(f"Loading {name} from huggingface and preprocessing...")
                subdataset = DATASETS[name]().dataset
                print(f"Tokenizing {name}...")
                subdataset = tokenize(
                    subdataset,
                    tokenizer,
                    column_name=subdataset.column_names,
                    **cfg.tokenizer,
                )
                subdataset.save_to_disk(os.path.join(cfg.datasets.path, "all", name))

            dataset_dict[name] = subdataset

        # Aggregate datasets
        dataset = DatasetDict(dataset_dict)
        print(
            f"Global dataset is ready ! It contains subdatasets {list(DATASETS.keys())}."
        )

        with open(
            os.path.join(cfg.datasets.path, "all", "dataset_dict.json"), mode="w"
        ) as f:
            json.dump({"splits": list(dataset.keys())}, f)

    return dataset


def main():
    # Load configuration from command line arguments
    config = load_config_from_args()

    # Run contrastive preprocessing
    pipeline(config)


if __name__ == "__main__":
    main()
