#!/usr/bin/env python3
"""Tests for streaming dataset shuffle helpers."""

import unittest

from neobert.pretraining.trainer import (
    _maybe_shuffle_streaming_dataset,
    _prepare_resume_dataloader,
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

    def test_prepare_resume_dataloader_skips_streaming(self):
        """Ensure resume skip logic avoids len/set_epoch on streaming dataloaders."""

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

            def skip_first_batches(self, dataloader, num_batches: int):
                self.skip_called = True
                return dataloader

        dataloader = DummyDataloader()
        accelerator = DummyAccelerator()
        metrics = {"train/epochs": 1, "train/batches": 5}

        skipped = _prepare_resume_dataloader(
            dataloader, metrics, accelerator, is_streaming=True
        )

        self.assertIsNone(skipped)
        self.assertFalse(dataloader.set_epoch_called)
        self.assertFalse(accelerator.skip_called)

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
