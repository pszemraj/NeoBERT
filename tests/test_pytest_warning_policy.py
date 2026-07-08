"""Regression tests for the repository pytest warning policy."""

import warnings


def test_warning_policy_scopes_warnings_as_errors(pytestconfig) -> None:
    """Warnings-as-errors should stay scoped to NeoBERT modules."""
    filters = pytestconfig.getini("filterwarnings")

    assert "error" not in filters
    assert any(entry.startswith("error:::neobert") for entry in filters)


def test_external_runtime_warning_is_not_promoted_to_error() -> None:
    """External runtime warnings should not fail the suite globally."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("Can't initialize NVML", UserWarning, stacklevel=1)

    assert len(caught) == 1
