"""Regression tests for tests/run_tests.py target resolution behavior."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_run_tests_module():
    """Load tests/run_tests.py as a module for direct helper testing."""
    script_path = Path(__file__).resolve().parent / "run_tests.py"
    spec = importlib.util.spec_from_file_location(
        "neobert_tests_run_tests", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_args_file_target_honors_custom_pattern_match() -> None:
    """Ensure file targets still work with explicit non-default pattern filters."""
    run_tests = _load_run_tests_module()

    args, matched = run_tests._pytest_args(
        test_dir="model",
        pattern="test_model*.py",
        quiet=False,
        verbose=False,
    )

    assert matched
    assert len(args) == 1
    assert args[0].endswith("tests/test_model_forward.py")


def test_pytest_args_file_target_reports_non_matching_pattern() -> None:
    """Ensure file targets fail fast when the explicit pattern does not match."""
    run_tests = _load_run_tests_module()

    args, matched = run_tests._pytest_args(
        test_dir="model",
        pattern="test_does_not_match*.py",
        quiet=False,
        verbose=False,
    )

    assert not matched
    assert args == []


def test_unittest_discovery_file_target_uses_parent_directory(monkeypatch) -> None:
    """Ensure unittest fallback handles file targets by discovering from parent dir."""
    run_tests = _load_run_tests_module()
    test_root = Path(__file__).resolve().parent
    expected_file = test_root / "test_model_forward.py"
    captured: dict[str, str | None] = {}

    def fake_discover(self, start_dir, pattern, top_level_dir=None):
        captured["start_dir"] = start_dir
        captured["pattern"] = pattern
        captured["top_level_dir"] = top_level_dir
        return unittest.TestSuite()

    class _Result:
        def wasSuccessful(self) -> bool:
            return True

    class _Runner:
        def __init__(self, verbosity: int) -> None:
            self.verbosity = verbosity

        def run(self, _suite: unittest.TestSuite) -> _Result:
            return _Result()

    monkeypatch.setattr(run_tests.unittest.TestLoader, "discover", fake_discover)
    monkeypatch.setattr(run_tests.unittest, "TextTestRunner", _Runner)

    success = run_tests._run_unittest_discovery(
        test_dir="model",
        pattern="test_*.py",
        verbosity=2,
    )

    assert success
    assert captured["start_dir"] == str(expected_file.parent)
    assert captured["pattern"] == expected_file.name
    assert captured["top_level_dir"] == str(test_root)
