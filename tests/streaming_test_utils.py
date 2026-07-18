"""Shared helpers for streaming retry tests."""

from types import SimpleNamespace

import requests


def http_error(status_code: int) -> requests.exceptions.HTTPError:
    """Construct an HTTPError with a response-like object.

    :param int status_code: HTTP status code to attach.
    :return requests.exceptions.HTTPError: HTTP error instance.
    """
    response = SimpleNamespace(status_code=status_code)
    return requests.exceptions.HTTPError(
        f"{status_code} transient failure",
        response=response,
    )
