"""Tests for explicit long-sequence dataset filtering."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.pretraining.longer_seq import longer_seq


def test_longer_seq_honors_explicit_default_valued_threshold() -> None:
    """An explicit value of five must not be replaced by a hidden sentinel."""
    cfg = SimpleNamespace(dataset=SimpleNamespace(path="tokenized"))
    dataset = MagicMock()
    dataset.__len__.return_value = 3
    dataset.filter.return_value = dataset

    with (
        patch("scripts.pretraining.longer_seq.load_from_disk", return_value=dataset),
        patch("scripts.pretraining.longer_seq.os.sched_getaffinity", return_value={0}),
    ):
        longer_seq(cfg, min_length=5)

    first_filter = dataset.filter.call_args_list[0]
    second_filter = dataset.filter.call_args_list[1]
    assert first_filter.args[0]({"input_ids": [0] * 5})
    assert not first_filter.args[0]({"input_ids": [0] * 4})
    assert second_filter.args[0]({"input_ids": [0] * 10})
    assert not second_filter.args[0]({"input_ids": [0] * 9})
    assert [call.args[0] for call in dataset.save_to_disk.call_args_list] == [
        "tokenized+5",
        "tokenized+10",
    ]


def test_longer_seq_rejects_non_positive_threshold() -> None:
    """Invalid explicit thresholds should fail before loading data."""
    cfg = SimpleNamespace(dataset=SimpleNamespace(path="tokenized"))

    with pytest.raises(ValueError, match="must be positive"):
        longer_seq(cfg, min_length=0)
