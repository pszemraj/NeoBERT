#!/usr/bin/env python3
"""Tests for streaming dataset shuffle helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
import torch

from neobert.pretraining.trainer import (
    _maybe_shuffle_streaming_dataset,
    _load_streaming_split,
    _prepare_resume_dataloader,
)
from neobert.streaming import RetryingStreamingDataset, peek_streaming_example


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    """Construct an HTTPError with a response-like object.

    :param int status_code: HTTP status code to attach.
    :return requests.exceptions.HTTPError: HTTP error instance.
    """
    response = SimpleNamespace(status_code=status_code)
    return requests.exceptions.HTTPError(
        f"{status_code} transient failure",
        response=response,
    )


class DummyStreamingDataset:
    """Streaming dataset stub that records shuffle calls."""

    def __init__(self) -> None:
        """Initialize the dummy dataset."""
        self.shuffle_calls: list[tuple[int, int]] = []

    def shuffle(self, buffer_size: int, seed: int):
        """Record shuffle calls and return self.

        :param int buffer_size: Shuffle buffer size.
        :param int seed: Random seed.
        :return DummyStreamingDataset: Self for chaining.
        """
        self.shuffle_calls.append((buffer_size, seed))
        return self


class TestStreamingShuffle(unittest.TestCase):
    """Validate streaming shuffle helper behavior."""

    def test_shuffle_called_with_positive_buffer(self):
        """Ensure shuffle runs when buffer size is positive."""
        dataset = DummyStreamingDataset()
        out = _maybe_shuffle_streaming_dataset(dataset, buffer_size=32, seed=123)

        self.assertIs(out, dataset)
        self.assertEqual(dataset.shuffle_calls, [(32, 123)])

    def test_shuffle_skipped_with_zero_buffer(self):
        """Ensure shuffle is skipped when buffer size is zero."""
        dataset = DummyStreamingDataset()
        out = _maybe_shuffle_streaming_dataset(dataset, buffer_size=0, seed=123)

        self.assertIs(out, dataset)
        self.assertEqual(dataset.shuffle_calls, [])

    def test_prepare_resume_dataloader_streaming_skip_cases(self):
        """Ensure streaming resume skips using the expected counter per case."""

        class DummyDataloader:
            def __init__(self) -> None:
                self.set_epoch_called = False

            def set_epoch(self, epoch: int) -> None:
                self.set_epoch_called = True

            def __len__(self) -> int:
                raise TypeError("no length")

        class DummyAccelerator:
            def __init__(self) -> None:
                self.skip_called = False
                self.last_skip = None

            def skip_first_batches(self, dataloader, num_batches: int):
                self.skip_called = True
                self.last_skip = num_batches
                return dataloader

        cases = [
            ("global_batches", {"train/epochs": 1, "train/batches": 5}, 5),
            (
                "epoch_local_batches",
                {
                    "train/epochs": 3,
                    "train/batches": 105,  # Global batches across prior epochs.
                    "train/batches_in_epoch": 7,  # Epoch-local resume position.
                },
                7,
            ),
        ]

        for _name, metrics, expected_skip in cases:
            dataloader = DummyDataloader()
            accelerator = DummyAccelerator()
            skipped = _prepare_resume_dataloader(
                dataloader, metrics, accelerator, is_streaming=True
            )
            self.assertIs(skipped, dataloader)
            self.assertTrue(dataloader.set_epoch_called)
            self.assertTrue(accelerator.skip_called)
            self.assertEqual(accelerator.last_skip, expected_skip)

    def test_prepare_resume_dataloader_uses_len_for_non_streaming(self):
        """Ensure resume skip logic uses len and skip_first_batches when available."""

        class DummyDataloader:
            def __init__(self) -> None:
                self.set_epoch_called = False

            def set_epoch(self, epoch: int) -> None:
                self.set_epoch_called = True

            def __len__(self) -> int:
                return 10

        class DummyAccelerator:
            def __init__(self) -> None:
                self.skip_called = False
                self.last_skip = None

            def skip_first_batches(self, dataloader, num_batches: int):
                self.skip_called = True
                self.last_skip = num_batches
                return dataloader

        dataloader = DummyDataloader()
        accelerator = DummyAccelerator()
        metrics = {"train/epochs": 2, "train/batches": 17}

        skipped = _prepare_resume_dataloader(
            dataloader, metrics, accelerator, is_streaming=False
        )

        self.assertIs(skipped, dataloader)
        self.assertTrue(dataloader.set_epoch_called)
        self.assertTrue(accelerator.skip_called)
        self.assertEqual(accelerator.last_skip, 7)


class TestStreamingPercentSlice(unittest.TestCase):
    """Validate that percent slicing on streaming datasets fails fast."""

    def test_percent_slice_raises_without_total(self):
        """Ensure ValueError is raised when percent slicing cannot resolve total."""
        from unittest.mock import MagicMock, patch

        from neobert.pretraining.trainer import _load_streaming_split

        mock_dataset = MagicMock()
        mock_builder = MagicMock()
        mock_builder.info.splits = {}  # No split info available

        with (
            patch(
                "neobert.pretraining.trainer.load_dataset", return_value=mock_dataset
            ),
            patch(
                "neobert.pretraining.trainer.load_dataset_builder",
                return_value=mock_builder,
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                _load_streaming_split("dummy_dataset", "train[:1%]", {})

            self.assertIn("percent slicing", str(ctx.exception))

    def test_load_streaming_split_retries_transient_load_errors(self):
        """Ensure transient hub failures retry before the split loader gives up."""
        mock_dataset = MagicMock()
        calls = {"count": 0}

        def _flaky_load(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise _http_error(503)
            return mock_dataset

        with patch("neobert.pretraining.trainer.load_dataset", side_effect=_flaky_load):
            loaded = _load_streaming_split(
                "dummy_dataset",
                "train",
                {},
                streaming_read_retries=1,
                streaming_read_retry_backoff_seconds=0.01,
                streaming_read_retry_max_backoff_seconds=0.01,
            )

        self.assertIs(loaded, mock_dataset)
        self.assertEqual(calls["count"], 2)


class _PeekFlakyDataset(torch.utils.data.IterableDataset):
    """Yield once after a single transient failure."""

    def __init__(self) -> None:
        """Initialize the flaky dataset."""
        super().__init__()
        self.failed = False

    def __iter__(self):
        """Yield the first example after a transient failure.

        :return collections.abc.Iterator[dict[str, str]]: Example iterator.
        """
        if not self.failed:
            self.failed = True
            raise _http_error(503)
        yield {"text": "ok"}


class _RetryableFlakyDataset(torch.utils.data.IterableDataset):
    """Streaming stub that can resume from state after transient failures."""

    def __init__(
        self,
        values: list[int],
        *,
        fail_at: int,
        fail_times: int,
    ) -> None:
        """Initialize the retryable stub.

        :param list[int] values: Sequence of values to yield.
        :param int fail_at: Cursor position where failures should trigger.
        :param int fail_times: Number of transient failures before success.
        """
        super().__init__()
        self.values = values
        self.fail_at = fail_at
        self.failures_remaining = fail_times
        self.epoch = 0
        self._starting_state = {"cursor": 0, "epoch": 0}
        self._state = {"cursor": 0, "epoch": 0}

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch.

        :param int epoch: Epoch index.
        """
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        """Return the current iterator state.

        :return dict[str, int]: Cursor and epoch state.
        """
        return dict(self._state)

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Load the next iterator start state.

        :param dict[str, int] state_dict: Cursor and epoch state.
        """
        self._starting_state = dict(state_dict)

    def __iter__(self):
        """Iterate values and inject configured transient failures.

        :return collections.abc.Iterator[int]: Value iterator.
        """
        cursor = 0
        if self._starting_state.get("epoch", 0) == self.epoch:
            cursor = int(self._starting_state.get("cursor", 0))
        while cursor < len(self.values):
            if cursor == self.fail_at and self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise _http_error(503)
            value = self.values[cursor]
            cursor += 1
            self._state = {"cursor": cursor, "epoch": self.epoch}
            yield value


class TestStreamingRetryHelpers(unittest.TestCase):
    """Validate retry helpers for transient streaming read failures."""

    def test_peek_streaming_example_retries_transient_http_error(self):
        """Ensure peeking the first streaming example retries transient errors."""
        dataset = _PeekFlakyDataset()

        first = peek_streaming_example(
            dataset,
            context="unit-test peek",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(first, {"text": "ok"})

    def test_retrying_streaming_dataset_resumes_after_transient_failure(self):
        """Ensure transient failures resume from the last yielded example."""
        dataset = _RetryableFlakyDataset([0, 1, 2, 3], fail_at=2, fail_times=1)
        wrapped = RetryingStreamingDataset(
            dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(list(wrapped), [0, 1, 2, 3])

    def test_retrying_streaming_dataset_raises_after_exhausting_budget(self):
        """Ensure persistent transient failures still fail loudly after retries."""
        dataset = _RetryableFlakyDataset([0, 1, 2], fail_at=1, fail_times=3)
        wrapped = RetryingStreamingDataset(
            dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        with self.assertRaises(RuntimeError) as ctx:
            list(wrapped)

        self.assertIn("exhausted 1 retry attempt", str(ctx.exception))
