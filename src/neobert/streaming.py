"""Streaming dataset helpers for transient read resilience."""

import errno
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from typing import Any, TypeVar

import requests
import torch
from datasets import IterableDataset as HuggingFaceIterableDataset

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
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

_WRAPPER_STATE_VERSION_KEY = "_retrying_streaming_wrapper_version"
_WRAPPER_STATE_EPOCH_KEY = "_retrying_streaming_wrapper_epoch"
_WRAPPER_DATASET_STATE_KEY = "_retrying_streaming_wrapper_dataset_state"
_WRAPPER_STATE_VERSION = 1
_STREAMING_ITERATOR_MARKERS = ("_iter", "_ex_iterable")


def _has_streaming_iterator_marker(dataset: object) -> bool:
    """Return whether a dataset exposes known streaming iterator internals.

    :param object dataset: Dataset-like object to inspect.
    :return bool: ``True`` when the object has a streaming iterator marker.
    """
    return bool(
        callable(getattr(dataset, "__iter__", None))
        and any(hasattr(dataset, marker) for marker in _STREAMING_ITERATOR_MARKERS)
    )


def is_streaming_dataset(dataset: object) -> bool:
    """Detect iterable datasets used for streaming-style iteration.

    :param object dataset: Dataset-like object to inspect.
    :return bool: ``True`` when the object should be treated as streaming.
    """
    return bool(
        isinstance(
            dataset,
            (torch.utils.data.IterableDataset, HuggingFaceIterableDataset),
        )
        or _has_streaming_iterator_marker(dataset)
    )


class TorchIterableDatasetAdapter(torch.utils.data.IterableDataset):
    """Expose a streaming iterable through PyTorch's ``IterableDataset`` API."""

    def __init__(self, dataset: object) -> None:
        """Initialize the adapter.

        :param object dataset: Streaming iterable dataset to adapt.
        :raises TypeError: If ``dataset`` is not recognized as streaming.
        """
        if not is_streaming_dataset(dataset):
            raise TypeError("TorchIterableDatasetAdapter requires a streaming dataset.")
        super().__init__()
        self.dataset = dataset

    def __getattr__(self, name: str) -> Any:
        """Delegate dataset-specific methods such as ``state_dict`` and ``set_epoch``.

        :param str name: Missing attribute name.
        :return Any: Attribute from the wrapped dataset.
        """
        try:
            dataset = self.__dict__["dataset"]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the wrapped streaming dataset.

        :return collections.abc.Iterator[Any]: Wrapped dataset iterator.
        """
        return iter(self.dataset)


def ensure_torch_iterable_dataset(dataset: object) -> object:
    """Return a PyTorch-compatible iterable dataset for streaming inputs.

    :param object dataset: Dataset-like object to inspect.
    :return object: Original dataset or a PyTorch iterable adapter.
    """
    if not is_streaming_dataset(dataset) or isinstance(
        dataset, torch.utils.data.IterableDataset
    ):
        return dataset
    return TorchIterableDatasetAdapter(dataset)


def supports_streaming_iteration_resume(dataset: object) -> bool:
    """Return whether a dataset exposes resumable iterator state hooks.

    Resume helpers assume the stateful-dataset contract HF iterable datasets
    implement: ``state_dict()`` returns a snapshot detached from live iterator
    state, while ``load_state_dict()`` may retain and later mutate the payload
    it is given (so callers defensively copy before loading).

    :param object dataset: Dataset-like object to inspect.
    :return bool: ``True`` when ``state_dict`` and ``load_state_dict`` are callable.
    """
    return bool(
        callable(getattr(dataset, "state_dict", None))
        and callable(getattr(dataset, "load_state_dict", None))
    )


def streaming_state_restore_drops_shuffle_buffer(dataset: object) -> bool:
    """Return whether restoring iterator state discards an in-memory shuffle buffer.

    HF iterable datasets do not serialize shuffle-buffer contents in
    ``state_dict()``; restoring a snapshot rewinds the source cursor but refills
    the buffer from new data, so buffered-but-unyielded examples are skipped.

    :param object dataset: Dataset-like object to inspect (the wrapped HF dataset).
    :return bool: ``True`` when the dataset carries an in-memory shuffle buffer.
    """
    return getattr(dataset, "_shuffling", None) is not None


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
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None


def is_transient_streaming_error(exc: BaseException) -> bool:
    """Return whether an exception looks like a transient remote-read failure.

    Message-fragment matching is restricted to network-layer exception types.
    Arbitrary exceptions whose text merely mentions e.g. ``timeout`` (config
    validation, auth failures inside dataset scripts) must fail fast instead of
    burning the retry budget.

    :param BaseException exc: Exception to classify.
    :return bool: ``True`` for retryable network/service failures.
    """
    for candidate in _iter_exception_chain(exc):
        if isinstance(candidate, (TimeoutError, ConnectionError)):
            return True
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
        if not isinstance(candidate, (OSError, requests.exceptions.RequestException)):
            continue
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

    Resumable streaming datasets are peeked non-destructively: the iterator state
    is snapshotted before inspection, each retry restarts from that snapshot, and
    the original cursor is restored before returning or raising.

    :param object dataset: Streaming dataset to read from.
    :param str context: Human-readable context for logs/errors.
    :param int max_retries: Maximum retry count after the initial failure.
    :param float base_backoff_seconds: Initial wait duration between retries.
    :param float max_backoff_seconds: Maximum capped wait duration.
    :param Callable[[float], None] sleep_fn: Sleep function for backoff delays.
    :return Any: First yielded example.
    """
    if not supports_streaming_iteration_resume(dataset):
        return retry_streaming_operation(
            lambda: next(iter(dataset)),
            context=context,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            sleep_fn=sleep_fn,
        )

    resume_state = dataset.state_dict()

    def _peek_once() -> Any:
        """Read the next example from the saved resume position.

        :return Any: Next example at the snapshotted cursor.
        """
        dataset.load_state_dict(deepcopy(resume_state))
        return next(iter(dataset))

    try:
        return retry_streaming_operation(
            _peek_once,
            context=context,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            sleep_fn=sleep_fn,
        )
    finally:
        dataset.load_state_dict(deepcopy(resume_state))


class RetryingStreamingDataset(torch.utils.data.IterableDataset):
    """Wrap a streaming dataset and restart iteration after transient failures.

    For unshuffled resumable streams, retry recovery is exactly-once: the cursor
    snapshot rewinds to the last yielded example. Shuffled HF streams do not
    serialize shuffle-buffer contents, so a retry restore refills the buffer and
    may skip up to ``shuffle_buffer_size`` buffered-but-unyielded examples; the
    wrapper logs a warning whenever such a lossy recovery happens.
    """

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
        self._epoch = torch.tensor(
            int(getattr(dataset, "epoch", 0)), dtype=torch.int64
        ).share_memory_()

    def _current_epoch(self) -> int:
        """Return the epoch stored in worker-shared memory.

        :return int: Current dataset epoch.
        """
        return int(self._epoch.item())

    def _set_shared_epoch(self, epoch: int) -> int:
        """Update the worker-shared epoch in place.

        :param int epoch: Epoch index to publish to persistent workers.
        :return int: Normalized epoch index.
        """
        normalized_epoch = int(epoch)
        self._epoch.fill_(normalized_epoch)
        return normalized_epoch

    def set_epoch(self, epoch: int) -> None:
        """Set the current dataset epoch and propagate to the wrapped dataset.

        :param int epoch: Epoch index to use for subsequent iterations.
        """
        epoch = self._set_shared_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable wrapper and dataset iteration state.

        :return dict[str, Any]: Serialized wrapper epoch and wrapped dataset state.
        """
        return {
            _WRAPPER_STATE_VERSION_KEY: _WRAPPER_STATE_VERSION,
            _WRAPPER_STATE_EPOCH_KEY: self._current_epoch(),
            _WRAPPER_DATASET_STATE_KEY: self.dataset.state_dict(),
        }

    def load_state_dict(self, state_dict: Any) -> None:
        """Restore wrapper and dataset iteration state.

        Wrapper-produced payloads restore both the wrapped dataset cursor and the
        wrapper epoch. Raw underlying dataset payloads are also accepted so runs
        can resume cleanly when retry wrapping is toggled on after earlier
        checkpoints were written without the wrapper.

        :param Any state_dict: Wrapper or wrapped-dataset resume payload.
        :raises ValueError: If a wrapper payload carries an unsupported version.
        """
        dataset_state = state_dict
        epoch = self._current_epoch()
        if isinstance(state_dict, Mapping):
            if _WRAPPER_DATASET_STATE_KEY in state_dict:
                version = state_dict.get(_WRAPPER_STATE_VERSION_KEY)
                if version != _WRAPPER_STATE_VERSION:
                    raise ValueError(
                        "Unsupported retrying-streaming wrapper state version "
                        f"{version!r}; this build reads version "
                        f"{_WRAPPER_STATE_VERSION}. Refusing to reinterpret "
                        "resume state written under a different format."
                    )
                dataset_state = state_dict[_WRAPPER_DATASET_STATE_KEY]
                epoch = int(state_dict.get(_WRAPPER_STATE_EPOCH_KEY, epoch))
            elif "epoch" in state_dict:
                raw_epoch = state_dict.get("epoch")
                if raw_epoch is not None:
                    epoch = int(raw_epoch)

        epoch = self._set_shared_epoch(epoch)
        self.dataset.load_state_dict(dataset_state)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the wrapped dataset with transient read recovery.

        The retry budget counts consecutive failures at a given resume point.
        Each successful yield resets the counter so isolated transient blips
        hours apart do not accumulate toward the budget. Recovery reloads a
        snapshot captured before the failed read, so the source cursor cannot
        skip past the failure boundary. Examples held in an in-memory shuffle
        buffer are not part of HF snapshot state, however: recovery on a
        shuffled stream refills the buffer and may skip buffered-but-unyielded
        examples (bounded by the shuffle buffer size), which is logged as a
        warning when it happens.

        :raises RuntimeError: If transient failures persist beyond the retry budget.
        :return collections.abc.Iterator[Any]: Example iterator.
        """
        retries = 0
        resume_state: Any | None = None
        epoch = self._current_epoch()
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        while True:
            if resume_state is not None:
                self.dataset.load_state_dict(deepcopy(resume_state))
                if hasattr(self.dataset, "set_epoch"):
                    self.dataset.set_epoch(epoch)
            try:
                # Snapshot before constructing/advancing the iterator so retries
                # restart from the last known-good cursor, even if the dataset
                # mutates its own state before surfacing a transient read error.
                # state_dict() returns a detached snapshot per the resume
                # contract, so no extra copy is taken on this per-example path.
                resume_state = self.dataset.state_dict()
                iterator = iter(self.dataset)
                while True:
                    example = next(iterator)
                    resume_state = self.dataset.state_dict()
                    retries = 0
                    yield example
            except StopIteration:
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
                    epoch,
                    exc,
                )
                if streaming_state_restore_drops_shuffle_buffer(self.dataset):
                    logger.warning(
                        "Retry recovery for %s reloads the stream cursor without "
                        "the in-memory shuffle buffer; up to shuffle_buffer_size "
                        "buffered examples may be skipped.",
                        self.label,
                    )
                self.sleep_fn(delay)
