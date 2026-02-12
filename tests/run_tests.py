#!/usr/bin/env python3
"""Test runner for NeoBERT test suite."""

import argparse
import fnmatch
import sys
import unittest
from pathlib import Path

LEGACY_TEST_TARGETS = {
    "config": "test_config_system.py",
    "model": "test_model_forward.py",
    "integration": "test_task_smoke.py",
    "tokenizer": "test_tokenize_streaming.py",
    "collator": "test_collator_packing.py",
}


def _resolve_test_target(test_dir: str | None, test_root: Path) -> Path:
    """Resolve user-provided test target, including legacy aliases."""
    if test_dir is None:
        return test_root

    candidate = LEGACY_TEST_TARGETS.get(test_dir, test_dir)
    return test_root / candidate


def _matches_pattern(path: Path, pattern: str) -> bool:
    """Return whether a file path matches a test glob pattern."""
    return fnmatch.fnmatch(path.name, pattern)


def _run_unittest_discovery(test_dir: str | None, pattern: str, verbosity: int) -> bool:
    """Run unittest discovery for legacy tests."""
    test_root = Path(__file__).parent

    test_path = _resolve_test_target(test_dir, test_root)
    if not test_path.exists():
        print(f"Test target not found: {test_path}")
        return False

    loader = unittest.TestLoader()
    if test_path.is_file():
        if not _matches_pattern(test_path, pattern):
            print(f"No tests matched pattern: {pattern}")
            return False
        suite = loader.discover(
            str(test_path.parent),
            pattern=test_path.name,
            top_level_dir=str(test_root),
        )
    else:
        suite = loader.discover(str(test_path), pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def _pytest_args(
    test_dir: str | None, pattern: str, quiet: bool, verbose: bool
) -> tuple[list[str], bool]:
    """Build pytest CLI arguments."""
    args: list[str] = []
    if quiet:
        args.append("-q")
    elif verbose:
        args.append("-vv")

    root = _resolve_test_target(test_dir, Path(__file__).parent)
    if not root.exists():
        print(f"Test target not found: {root}")
        return args, False

    if pattern != "test_*.py":
        if root.is_file():
            if not _matches_pattern(root, pattern):
                print(f"No tests matched pattern: {pattern}")
                return args, False
            args.append(str(root))
        else:
            matched = sorted(root.rglob(pattern))
            if not matched:
                print(f"No tests matched pattern: {pattern}")
                return args, False
            args.extend(str(path) for path in matched)
    elif test_dir is not None:
        args.append(str(root))

    return args, True


def discover_and_run_tests(
    test_dir=None,
    pattern="test_*.py",
    verbosity=2,
    use_pytest=True,
    quiet=False,
    verbose=False,
):
    """Discover and run tests."""
    if use_pytest:
        try:
            import pytest  # type: ignore

            args, matched = _pytest_args(
                test_dir, pattern, quiet=quiet, verbose=verbose
            )
            if not matched:
                return False
            return pytest.main(args) == 0
        except Exception:
            # Fall back to unittest discovery if pytest is unavailable.
            pass

    return _run_unittest_discovery(test_dir, pattern, verbosity)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run NeoBERT tests")
    parser.add_argument(
        "--test-dir",
        help=(
            "Run tests from a specific tests/ subpath (directory or file). "
            "Legacy aliases remain supported: config, model, integration, "
            "tokenizer, collator."
        ),
    )
    parser.add_argument(
        "--pattern", default="test_*.py", help="Test file pattern (default: test_*.py)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    parser.add_argument(
        "--no-pytest",
        action="store_true",
        help="Force unittest discovery instead of pytest.",
    )

    args = parser.parse_args()

    verbosity = 2
    if args.verbose:
        verbosity = 3
    elif args.quiet:
        verbosity = 1

    print("=" * 60)
    print("NeoBERT Test Suite")
    print("=" * 60)

    if args.test_dir:
        print(f"Running tests from: {args.test_dir}")
    else:
        print("Running all tests")

    print(f"Test pattern: {args.pattern}")
    print("=" * 60)

    success = discover_and_run_tests(
        test_dir=args.test_dir,
        pattern=args.pattern,
        verbosity=verbosity,
        use_pytest=not args.no_pytest,
        quiet=args.quiet,
        verbose=args.verbose,
    )

    print("=" * 60)
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
