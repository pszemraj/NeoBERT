from neobert.config import load_config_from_args
from neobert.pretraining import trainer


def main():
    # Load configuration from command line arguments
    config = load_config_from_args()

    # Run the trainer
    trainer(config)


if __name__ == "__main__":
    main()
