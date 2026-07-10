"""Regression tests for the repository pytest warning policy."""

import warnings

import pytest

from neobert.config import ConfigLoader
from neobert.warnings import NeoBERTWarning


def test_warning_policy_scopes_warnings_as_errors(pytestconfig) -> None:
    """Warnings-as-errors should target the explicit NeoBERT category."""
    filters = pytestconfig.getini("filterwarnings")

    assert "error" not in filters
    assert "error::neobert.warnings.NeoBERTWarning" in filters


def test_project_warning_is_an_error_independent_of_stacklevel() -> None:
    """Project warnings remain strict when attributed to an external caller."""
    with pytest.raises(NeoBERTWarning, match="legacy setting"):
        ConfigLoader._warn_legacy("legacy setting")


def test_external_runtime_warning_is_not_promoted_to_error() -> None:
    """External runtime warnings should not fail the suite globally."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("Can't initialize NVML", UserWarning, stacklevel=1)

    assert len(caught) == 1
