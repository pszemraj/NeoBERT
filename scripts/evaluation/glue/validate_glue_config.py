#!/usr/bin/env python3
"""Validate a GLUE config through the production config pipeline."""

import argparse
import sys
import warnings
from pathlib import Path

from neobert.config import ConfigLoader
from neobert.glue.validation import GlueValidationError, validate_glue_config


def validate_config_file(config_path: Path) -> tuple[str, ...]:
    """Load and validate one GLUE configuration without runtime side effects.

    :param Path config_path: YAML configuration to validate.
    :raises OSError: If the config cannot be read.
    :raises TypeError: If the config has invalid types.
    :raises ValueError: If generic configuration validation fails.
    :raises GlueValidationError: If GLUE-specific validation fails.
    :return tuple[str, ...]: Validation warning messages.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cfg = ConfigLoader.load(config_path)
    glue_warnings = validate_glue_config(cfg)
    return tuple(str(item.message) for item in captured) + glue_warnings


def main() -> None:
    """Run the GLUE config validation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a GLUE YAML config")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    args = parser.parse_args()

    try:
        validation_warnings = validate_config_file(args.config)
    except (OSError, TypeError, ValueError, GlueValidationError) as exc:
        print(f"Invalid GLUE config {args.config}:\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for message in validation_warnings:
        print(f"Warning: {message}", file=sys.stderr)
    if args.strict and validation_warnings:
        raise SystemExit(1)
    print(f"Valid GLUE config: {args.config}")


if __name__ == "__main__":
    main()
