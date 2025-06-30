from neobert.config import load_config_from_args
from neobert.contrastive import trainer


def main():
    # Load configuration from command line arguments
    config = load_config_from_args()

    # Run contrastive training
    trainer(config)


if __name__ == "__main__":
    main()
