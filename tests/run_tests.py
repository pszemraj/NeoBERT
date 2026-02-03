#!/usr/bin/env python3
"""Test runner for NeoBERT test suite."""

import argparse
import sys
import unittest
from pathlib import Path


def _run_unittest_discovery(test_dir: str | None, pattern: str, verbosity: int) -> bool:
    """Run unittest discovery for legacy tests."""
    test_root = Path(__file__).parent

    if test_dir:
        test_path = test_root / test_dir
        if not test_path.exists():
            print(f"Test directory not found: {test_path}")
            return False
    else:
        test_path = test_root

    loader = unittest.TestLoader()
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

    root = Path(__file__).parent
    if test_dir:
        root = root / test_dir

    if pattern != "test_*.py":
        matched = sorted(root.rglob(pattern))
        if not matched:
            print(f"No tests matched pattern: {pattern}")
            return args, False
        args.extend(str(path) for path in matched)
    elif test_dir:
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
        choices=["config", "model", "training", "evaluation", "integration"],
        help="Run tests from specific directory",
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
        print(f"Running tests from: {args.test_dir}/")
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
