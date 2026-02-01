#!/usr/bin/env python3
"""Tests for streaming dataset shuffle helpers."""

import unittest

from neobert.pretraining.trainer import _maybe_shuffle_streaming_dataset


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
