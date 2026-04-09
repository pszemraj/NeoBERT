"""Streaming dataset helpers for transient read resilience."""

import errno
import logging
import time
from typing import Any, Callable, Iterator, TypeVar

import requests
import torch

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_OS_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    }
)
_TRANSIENT_MESSAGE_FRAGMENTS = (
    "service unavailable",
    "temporarily unavailable",
    "too many requests",
    "connection reset",
    "connection aborted",
    "connection refused",
    "timed out",
    "timeout",
    "temporary failure in name resolution",
    "name or service not known",
    "remote end closed connection",
    "bad gateway",
    "gateway timeout",
)

T = TypeVar("T")


def is_streaming_dataset(dataset: object) -> bool:
    """Detect iterable datasets used for streaming-style iteration.

    :param object dataset: Dataset-like object to inspect.
    :return bool: ``True`` when the object should be treated as streaming.
    """
    return bool(
        isinstance(dataset, torch.utils.data.IterableDataset)
        or hasattr(dataset, "_iter")
    )


def supports_streaming_iteration_resume(dataset: object) -> bool:
    """Return whether a dataset exposes resumable iterator state hooks.

    :param object dataset: Dataset-like object to inspect.
    :return bool: ``True`` when ``state_dict`` and ``load_state_dict`` are callable.
    """
    return bool(
        callable(getattr(dataset, "state_dict", None))
        and callable(getattr(dataset, "load_state_dict", None))
    )


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception plus its causal chain.

    :param BaseException exc: Root exception.
    :return collections.abc.Iterator[BaseException]: Exception chain iterator.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_streaming_error(exc: BaseException) -> bool:
    """Return whether an exception looks like a transient remote-read failure.

    :param BaseException exc: Exception to classify.
    :return bool: ``True`` for retryable network/service failures.
    """
    for candidate in _iter_exception_chain(exc):
        if isinstance(
            candidate,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ),
        ):
            return True
        if isinstance(candidate, requests.exceptions.HTTPError):
            response = getattr(candidate, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code in _TRANSIENT_HTTP_STATUS_CODES:
                return True
        if isinstance(candidate, OSError) and getattr(candidate, "errno", None) in (
            _TRANSIENT_OS_ERRNOS
        ):
            return True
        message = str(candidate).strip().lower()
        if message and any(
            fragment in message for fragment in _TRANSIENT_MESSAGE_FRAGMENTS
        ):
            return True
    return False


def compute_retry_backoff_seconds(
    attempt: int,
    *,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
) -> float:
    """Compute capped exponential backoff for retry attempts.

    :param int attempt: 1-based retry attempt counter.
    :param float base_backoff_seconds: Initial wait duration.
    :param float max_backoff_seconds: Maximum capped wait duration.
    :return float: Sleep duration for the given attempt.
    """
    if attempt <= 0:
        return 0.0
    return min(max_backoff_seconds, base_backoff_seconds * (2 ** (attempt - 1)))


def retry_streaming_operation(
    operation: Callable[[], T],
    *,
    context: str,
    max_retries: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a transient streaming operation with exponential backoff.

    :param Callable[[], T] operation: Zero-argument operation to execute.
    :param str context: Human-readable context for logs/errors.
    :param int max_retries: Maximum retry count after the initial failure.
    :param float base_backoff_seconds: Initial wait duration between retries.
    :param float max_backoff_seconds: Maximum capped wait duration.
    :param Callable[[float], None] sleep_fn: Sleep function for backoff delays.
    :raises Exception: Re-raises the last non-transient or exhausted transient error.
    :return T: Operation result.
    """
    attempts = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if not is_transient_streaming_error(exc) or attempts >= max_retries:
                raise
            attempts += 1
            delay = compute_retry_backoff_seconds(
                attempts,
                base_backoff_seconds=base_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
            logger.warning(
                "Transient streaming read failure during %s (retry %s/%s in %.1fs): %s",
                context,
                attempts,
                max_retries,
                delay,
                exc,
            )
            sleep_fn(delay)


def peek_streaming_example(
    dataset: object,
    *,
    context: str,
    max_retries: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Fetch the next example from a streaming dataset with retry handling.

    :param object dataset: Streaming dataset to read from.
    :param str context: Human-readable context for logs/errors.
    :param int max_retries: Maximum retry count after the initial failure.
    :param float base_backoff_seconds: Initial wait duration between retries.
    :param float max_backoff_seconds: Maximum capped wait duration.
    :param Callable[[float], None] sleep_fn: Sleep function for backoff delays.
    :return Any: First yielded example.
    """
    return retry_streaming_operation(
        lambda: next(iter(dataset)),
        context=context,
        max_retries=max_retries,
        base_backoff_seconds=base_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
        sleep_fn=sleep_fn,
    )


class RetryingStreamingDataset(torch.utils.data.IterableDataset):
    """Wrap a streaming dataset and restart iteration after transient failures."""

    def __init__(
        self,
        dataset: object,
        *,
        label: str,
        max_retries: int,
        base_backoff_seconds: float,
        max_backoff_seconds: float,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize the retrying dataset wrapper.

        :param object dataset: Underlying streaming dataset.
        :param str label: Human-readable dataset label for logs/errors.
        :param int max_retries: Maximum retry count after the initial failure.
        :param float base_backoff_seconds: Initial wait duration between retries.
        :param float max_backoff_seconds: Maximum capped wait duration.
        :param Callable[[float], None] sleep_fn: Sleep function for backoff delays.
        :raises TypeError: If the dataset lacks streaming or stateful iteration support.
        """
        if not is_streaming_dataset(dataset):
            raise TypeError("RetryingStreamingDataset requires an iterable dataset.")
        if not supports_streaming_iteration_resume(dataset):
            raise TypeError(
                "RetryingStreamingDataset requires dataset.state_dict/load_state_dict."
            )
        super().__init__()
        self.dataset = dataset
        self.label = str(label)
        self.max_retries = int(max_retries)
        self.base_backoff_seconds = float(base_backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.sleep_fn = sleep_fn
        self._epoch = int(getattr(dataset, "epoch", 0))

    def set_epoch(self, epoch: int) -> None:
        """Set the current dataset epoch and propagate to the wrapped dataset.

        :param int epoch: Epoch index to use for subsequent iterations.
        """
        self._epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self._epoch)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the wrapped dataset with transient read recovery.

        The retry budget counts consecutive failures at a given resume point.
        Each successful yield resets the counter so isolated transient blips
        hours apart do not accumulate toward the budget.

        :raises RuntimeError: If transient failures persist beyond the retry budget.
        :return collections.abc.Iterator[Any]: Example iterator.
        """
        retries = 0
        resume_state: dict[str, Any] | None = None
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self._epoch)

        while True:
            if resume_state is not None:
                self.dataset.load_state_dict(resume_state)
                if hasattr(self.dataset, "set_epoch"):
                    self.dataset.set_epoch(self._epoch)
            try:
                iterator = iter(self.dataset)
                for example in iterator:
                    retries = 0
                    yield example
                return
            except Exception as exc:
                if not is_transient_streaming_error(exc):
                    raise
                if retries >= self.max_retries:
                    raise RuntimeError(
                        f"Streaming dataset '{self.label}' exhausted "
                        f"{self.max_retries} retry attempt(s) after transient "
                        "read failures."
                    ) from exc
                resume_state = self.dataset.state_dict()
                retries += 1
                delay = compute_retry_backoff_seconds(
                    retries,
                    base_backoff_seconds=self.base_backoff_seconds,
                    max_backoff_seconds=self.max_backoff_seconds,
                )
                logger.warning(
                    "Transient streaming read failure in %s (retry %s/%s in %.1fs, epoch=%s): %s",
                    self.label,
                    retries,
                    self.max_retries,
                    delay,
                    self._epoch,
                    exc,
                )
                self.sleep_fn(delay)
