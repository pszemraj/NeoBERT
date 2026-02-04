"""Entry point for pretraining."""

from __future__ import annotations

import sys
from typing import List

from neobert.config import load_config_from_args
from neobert.pretraining import trainer


def _ensure_config_arg(argv: List[str]) -> List[str]:
    """Normalize CLI arguments to require a positional config path.

    Accepts either:
      - `pretrain.py <config.yaml> [--overrides]`
      - `pretrain.py --config <config.yaml> [--overrides]`

    :param list[str] argv: Raw argv list including script name.
    :return list[str]: argv with `--config` injected when positional is used.
    :raises SystemExit: If no config path is provided.
    """
    if any(arg in {"-h", "--help"} for arg in argv):
        return argv
    if "--config" in argv:
        return argv

    for idx, arg in enumerate(argv[1:], start=1):
        if not arg.startswith("-"):
            return argv[:idx] + ["--config", arg] + argv[idx + 1 :]

    raise SystemExit(
        "Config path is required. Usage: pretrain.py <config.yaml> [--overrides]"
    )


def main() -> None:
    """Run pretraining from a CLI config."""
    sys.argv = _ensure_config_arg(sys.argv)

    # Load configuration from command line arguments
    config = load_config_from_args()

    # Run the trainer
    trainer(config)


if __name__ == "__main__":
    main()
