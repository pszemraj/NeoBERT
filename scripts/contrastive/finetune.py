"""Entry point for contrastive finetuning."""

from neobert.config import load_config_from_args
from neobert.contrastive import trainer


def main() -> None:
    """Run contrastive finetuning from a CLI config."""
    # Load configuration from command line arguments
    config = load_config_from_args(require_config=True)

    # Run contrastive training
    trainer(config)


if __name__ == "__main__":
    main()
