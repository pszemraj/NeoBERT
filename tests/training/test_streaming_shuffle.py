#!/usr/bin/env python3
"""Tests for streaming dataset shuffle helpers."""

import unittest
from unittest.mock import MagicMock, patch

import requests
import torch

from neobert.config import Config
from neobert.pretraining.trainer import (
    _maybe_wrap_streaming_dataset_for_retries,
    _maybe_shuffle_streaming_dataset,
    _load_streaming_split,
    _prepare_resume_dataloader,
)
from neobert.streaming import (
    RetryingStreamingDataset,
    is_streaming_dataset,
    is_transient_streaming_error,
    peek_streaming_example,
)
from tests.streaming_test_utils import http_error


class DummyStreamingDataset:
    """Streaming dataset stub that records shuffle calls."""

    _ex_iterable = object()

    def __init__(self) -> None:
        """Initialize the dummy dataset."""
        self.shuffle_calls: list[tuple[int, int]] = []

    def __iter__(self):
        """Yield no examples.

        :return collections.abc.Iterator[object]: Empty iterator.
        """
        return iter(())

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

    def test_shuffle_skipped_for_map_style_dataset(self):
        """Ensure streaming shuffle helper does not call map-style shuffle APIs."""

        class _MapStyleDataset:
            def __init__(self) -> None:
                self.shuffle_called = False

            def shuffle(self, *args, **kwargs):
                self.shuffle_called = True
                return self

        dataset = _MapStyleDataset()

        out = _maybe_shuffle_streaming_dataset(dataset, buffer_size=32, seed=123)

        self.assertIs(out, dataset)
        self.assertFalse(dataset.shuffle_called)

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
                raise http_error(503)
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
            raise http_error(503)
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
                raise http_error(503)
            value = self.values[cursor]
            cursor += 1
            self._state = {"cursor": cursor, "epoch": self.epoch}
            yield value


class _StatefulCursorStreamingDataset(torch.utils.data.IterableDataset):
    """Streaming stub whose dataset object owns the active cursor state."""

    def __init__(self, values: list[int], *, fail_once_after_advance: bool) -> None:
        """Store values and optional one-time transient failure behavior.

        :param list[int] values: Sequence of values to yield.
        :param bool fail_once_after_advance:
            Whether to advance the cursor once, raise a transient error, then
            succeed on the next attempt.
        """
        super().__init__()
        self.values = values
        self.cursor = 0
        self.fail_once_after_advance = fail_once_after_advance

    def state_dict(self) -> dict[str, int]:
        """Return the active dataset cursor.

        :return dict[str, int]: Current cursor position.
        """
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore the active dataset cursor.

        :param dict[str, int] state_dict: Cursor state to restore.
        """
        self.cursor = int(state_dict["cursor"])

    def __iter__(self):
        """Yield values while mutating the dataset-owned cursor in place.

        :return collections.abc.Iterator[int]: Value iterator.
        """
        if self.fail_once_after_advance:
            self.fail_once_after_advance = False
            self.cursor += 1
            raise http_error(503)
        while self.cursor < len(self.values):
            value = self.values[self.cursor]
            self.cursor += 1
            yield value


class _StatelessStreamingDataset(torch.utils.data.IterableDataset):
    """Iterable dataset stub without resumable iteration state."""

    def __init__(self, values: list[int]) -> None:
        """Store values to yield.

        :param list[int] values: Sequence of values to iterate.
        """
        super().__init__()
        self.values = values

    def __iter__(self):
        """Yield configured values in order.

        :return collections.abc.Iterator[int]: Value iterator.
        """
        yield from self.values


class _HuggingFaceStyleStreamingDataset:
    """Legacy Hugging Face-style iterable that is not a torch subclass."""

    def __init__(self, values: list[int]) -> None:
        """Store values and expose the Hugging Face streaming marker.

        :param list[int] values: Sequence of values to iterate.
        """
        self.values = values
        self._ex_iterable = object()
        self._starting_state = {"cursor": 0}
        self._state = {"cursor": 0}

    def state_dict(self) -> dict[str, int]:
        """Return the active dataset cursor.

        :return dict[str, int]: Current cursor position.
        """
        return dict(self._state)

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Load the next iterator start state.

        :param dict[str, int] state_dict: Cursor state.
        """
        self._starting_state = dict(state_dict)

    def __iter__(self):
        """Yield configured values while updating cursor state.

        :return collections.abc.Iterator[int]: Value iterator.
        """
        cursor = int(self._starting_state.get("cursor", 0))
        while cursor < len(self.values):
            value = self.values[cursor]
            cursor += 1
            self._state = {"cursor": cursor}
            yield value


class TestTransientErrorClassification(unittest.TestCase):
    """Validate transient-vs-permanent classification of streaming errors."""

    def test_network_typed_errors_are_transient(self):
        """Typed network failures classify as transient without message checks."""
        self.assertTrue(
            is_transient_streaming_error(requests.exceptions.ConnectionError("boom"))
        )
        self.assertTrue(is_transient_streaming_error(http_error(503)))

    def test_non_transient_http_status_is_permanent(self):
        """Client errors such as 404 must not be retried."""
        self.assertFalse(is_transient_streaming_error(http_error(404)))

    def test_message_match_requires_network_exception_type(self):
        """Network-sounding text in arbitrary exception types must fail fast."""
        self.assertFalse(
            is_transient_streaming_error(
                RuntimeError("Authentication failed: connection refused by auth proxy")
            )
        )
        self.assertFalse(
            is_transient_streaming_error(ValueError("read timeout must be > 0"))
        )
        self.assertFalse(
            is_transient_streaming_error(
                KeyError("missing config field: connect_timeout_seconds")
            )
        )

    def test_message_match_applies_to_network_exception_types(self):
        """OSError/requests errors without structured metadata match by message."""
        self.assertTrue(
            is_transient_streaming_error(
                OSError("Remote end closed connection without response")
            )
        )
        self.assertTrue(
            is_transient_streaming_error(
                requests.exceptions.RequestException("connection reset by peer")
            )
        )

    def test_transient_cause_inside_wrapper_exception_is_detected(self):
        """A permanent-looking wrapper with a transient network cause retries."""
        wrapper = RuntimeError("dataset generation failed")
        wrapper.__cause__ = requests.exceptions.ConnectionError("reset")

        self.assertTrue(is_transient_streaming_error(wrapper))


class TestStreamingRetryHelpers(unittest.TestCase):
    """Validate retry helpers for transient streaming read failures."""

    def test_is_streaming_dataset_recognizes_huggingface_marker(self):
        """Ensure legacy Hugging Face stream markers are treated as streaming."""
        dataset = _HuggingFaceStyleStreamingDataset([0, 1, 2])

        self.assertTrue(is_streaming_dataset(dataset))

    def test_retry_wrapper_accepts_huggingface_marker_streams(self):
        """Ensure retry setup still wraps Hugging Face-style resumable streams."""
        dataset = _HuggingFaceStyleStreamingDataset([0, 1, 2])
        cfg = Config()
        cfg.dataset.streaming_read_retries = 1
        cfg.dataset.streaming_read_retry_backoff_seconds = 0.01
        cfg.dataset.streaming_read_retry_max_backoff_seconds = 0.01

        wrapped = _maybe_wrap_streaming_dataset_for_retries(
            dataset,
            label="unit-test",
            cfg=cfg,
        )

        self.assertIsInstance(wrapped, RetryingStreamingDataset)
        self.assertEqual(list(wrapped), [0, 1, 2])

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

    def test_peek_streaming_example_restores_stateful_cursor_after_retry(self):
        """Peeking resumable streams must not consume examples or skew retries."""
        dataset = _StatefulCursorStreamingDataset(
            [10, 11, 12],
            fail_once_after_advance=True,
        )

        first = peek_streaming_example(
            dataset,
            context="stateful cursor peek",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(first, 10)
        self.assertEqual(list(dataset), [10, 11, 12])

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

    def test_retrying_streaming_dataset_warns_on_lossy_shuffle_buffer_recovery(self):
        """Shuffled streams must surface that retry recovery drops buffered examples."""
        dataset = _RetryableFlakyDataset([0, 1, 2, 3], fail_at=2, fail_times=1)
        dataset._shuffling = object()  # mimic HF IterableDataset.shuffle() marker
        wrapped = RetryingStreamingDataset(
            dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        with self.assertLogs("neobert.streaming", level="WARNING") as logs:
            self.assertEqual(list(wrapped), [0, 1, 2, 3])

        self.assertTrue(
            any("shuffle buffer" in message for message in logs.output),
            logs.output,
        )

    def test_retrying_streaming_dataset_recovery_stays_quiet_without_shuffle(self):
        """Unshuffled streams recover exactly-once and must not warn about buffers."""
        dataset = _RetryableFlakyDataset([0, 1, 2, 3], fail_at=2, fail_times=1)
        wrapped = RetryingStreamingDataset(
            dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        with self.assertLogs("neobert.streaming", level="WARNING") as logs:
            self.assertEqual(list(wrapped), [0, 1, 2, 3])

        self.assertFalse(
            any("shuffle buffer" in message for message in logs.output),
            logs.output,
        )

    def test_retrying_streaming_dataset_restores_cursor_advanced_before_failure(self):
        """Retries must not skip examples when the dataset advances before failing."""
        dataset = _StatefulCursorStreamingDataset(
            [10, 11, 12],
            fail_once_after_advance=True,
        )
        wrapped = RetryingStreamingDataset(
            dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(list(wrapped), [10, 11, 12])

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

    def test_retrying_streaming_dataset_resets_budget_after_successful_yields(self):
        """Isolated transient failures at different positions each get the full retry budget."""

        class _MultiFlakyDataset(torch.utils.data.IterableDataset):
            """Streaming stub that fails once at each of several cursor positions."""

            def __init__(self, values, fail_at_positions):
                super().__init__()
                self.values = values
                self.fail_at_positions = set(fail_at_positions)
                self.epoch = 0
                self._starting_state = {"cursor": 0, "epoch": 0}
                self._state = {"cursor": 0, "epoch": 0}

            def set_epoch(self, epoch):
                self.epoch = int(epoch)

            def state_dict(self):
                return dict(self._state)

            def load_state_dict(self, state_dict):
                self._starting_state = dict(state_dict)

            def __iter__(self):
                cursor = 0
                if self._starting_state.get("epoch", 0) == self.epoch:
                    cursor = int(self._starting_state.get("cursor", 0))
                while cursor < len(self.values):
                    if cursor in self.fail_at_positions:
                        self.fail_at_positions.discard(cursor)
                        raise http_error(503)
                    cursor += 1
                    self._state = {"cursor": cursor, "epoch": self.epoch}
                    yield self.values[cursor - 1]

        # Two isolated failures with max_retries=1.  Without retry-budget
        # reset this would exhaust the budget on the second failure.
        dataset = _MultiFlakyDataset([0, 1, 2, 3, 4, 5], fail_at_positions={1, 4})
        wrapped = RetryingStreamingDataset(
            dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(list(wrapped), [0, 1, 2, 3, 4, 5])

    def test_retrying_streaming_dataset_state_dict_round_trips_resume_state(self):
        """Checkpointed wrapper state should resume from the wrapped cursor."""
        wrapped = RetryingStreamingDataset(
            _RetryableFlakyDataset([0, 1, 2, 3], fail_at=99, fail_times=0),
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )
        wrapped.set_epoch(7)

        iterator = iter(wrapped)
        self.assertEqual(next(iterator), 0)
        self.assertEqual(next(iterator), 1)
        saved_state = wrapped.state_dict()

        resumed_dataset = _RetryableFlakyDataset([0, 1, 2, 3], fail_at=99, fail_times=0)
        resumed = RetryingStreamingDataset(
            resumed_dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        resumed.load_state_dict(saved_state)

        self.assertEqual(resumed_dataset.epoch, 7)
        self.assertEqual(list(resumed), [2, 3])

    def test_retrying_streaming_dataset_loads_raw_dataset_state(self):
        """Raw dataset resume payloads should remain loadable through the wrapper."""
        raw_state = {"cursor": 2, "epoch": 5}

        resumed_dataset = _RetryableFlakyDataset([0, 1, 2, 3], fail_at=99, fail_times=0)
        resumed = RetryingStreamingDataset(
            resumed_dataset,
            label="unit-test",
            max_retries=1,
            base_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
            sleep_fn=lambda _seconds: None,
        )

        resumed.load_state_dict(raw_state)

        self.assertEqual(resumed_dataset.epoch, 5)
        self.assertEqual(list(resumed), [2, 3])

    def test_retrying_streaming_dataset_requires_resumable_state(self):
        """Ensure the low-level retry wrapper remains strict about resume hooks."""
        dataset = _StatelessStreamingDataset([0, 1, 2])

        with self.assertRaises(TypeError) as ctx:
            RetryingStreamingDataset(
                dataset,
                label="unit-test",
                max_retries=1,
                base_backoff_seconds=0.01,
                max_backoff_seconds=0.01,
                sleep_fn=lambda _seconds: None,
            )

        self.assertIn("state_dict/load_state_dict", str(ctx.exception))

    def test_retry_wrapper_falls_back_for_stateless_streams(self):
        """Ensure trainer setup does not force retry wrapping onto stateless streams."""
        dataset = _StatelessStreamingDataset([0, 1, 2])
        cfg = Config()
        cfg.dataset.streaming_read_retries = 2
        cfg.dataset.streaming_read_retry_backoff_seconds = 0.01
        cfg.dataset.streaming_read_retry_max_backoff_seconds = 0.01

        with self.assertLogs("neobert.pretraining.trainer", level="WARNING") as logs:
            wrapped = _maybe_wrap_streaming_dataset_for_retries(
                dataset,
                label="unit-test",
                cfg=cfg,
            )

        self.assertIs(wrapped, dataset)
        self.assertIn(
            "does not expose resumable iteration state",
            "\n".join(logs.output),
        )
