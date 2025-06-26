#!/usr/bin/env python3
"""Test runner for NeoBERT test suite."""

import argparse
import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def discover_and_run_tests(test_dir=None, pattern="test_*.py", verbosity=2):
    """Discover and run tests."""
    test_root = Path(__file__).parent

    if test_dir:
        test_path = test_root / test_dir
        if not test_path.exists():
            print(f"Test directory not found: {test_path}")
            return False
    else:
        test_path = test_root

    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_path), pattern=pattern)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


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
        test_dir=args.test_dir, pattern=args.pattern, verbosity=verbosity
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
