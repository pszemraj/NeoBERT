"""Distributed regressions for variable-size packed pretraining batches."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from neobert.collator import DataCollatorWithPacking
from neobert.pretraining.metrics import Metrics


class _PackedPadCollator:
    """Convert already padded packing outputs to tensors."""

    pad_token_id = 0

    def __call__(self, features, return_tensors=None):
        """Tensorize packing outputs.

        :param list[dict] features: Fixed-width packed features.
        :param Any return_tensors: Unused compatibility argument.
        :return dict[str, torch.Tensor]: Tensorized packed batch.
        """
        del return_tensors
        return {
            "input_ids": torch.tensor(
                [feature["input_ids"] for feature in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [feature["attention_mask"] for feature in features], dtype=torch.long
            ),
        }


def _assert_all_ranks_equal(tensor: torch.Tensor, world_size: int) -> None:
    """Assert exact tensor equality across a process group.

    :param torch.Tensor tensor: Rank-local tensor.
    :param int world_size: Process-group size.
    """
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    for rank_value in gathered[1:]:
        torch.testing.assert_close(rank_value, gathered[0], rtol=0.0, atol=0.0)


def _packed_distributed_worker(rank: int, world_size: int, init_file: str) -> None:
    """Train with asymmetric packed row counts and validate checkpoint cursors.

    :param int rank: Process rank.
    :param int world_size: Process-group size.
    :param str init_file: File-store rendezvous path.
    """
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        collator = DataCollatorWithPacking(
            start_token_id=14,
            end_token_id=15,
            max_length=8,
            default_data_collator=_PackedPadCollator(),
        )
        features = (
            [{"input_ids": [1, 2, 3, 4, 5, 6]} for _ in range(4)]
            if rank == 0
            else [{"input_ids": []} for _ in range(4)]
        )
        packed = collator(features)
        expected_rows = 4 if rank == 0 else 1
        assert packed["input_ids"].shape == (expected_rows, 8)

        torch.manual_seed(2025)
        model = DistributedDataParallel(nn.Linear(8, 1, bias=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = Metrics()
        checkpoint_state = None

        for raw_pull in range(4):
            metrics["train/dataloader_batches_in_epoch"] += 1
            metrics["train/batches_in_epoch"] += 1
            is_accumulation_boundary = raw_pull % 2 == 1
            sync_context = (
                nullcontext() if is_accumulation_boundary else model.no_sync()
            )
            with sync_context:
                loss = model(packed["input_ids"].float()).sum()
                (loss / 2.0).backward()

            if is_accumulation_boundary:
                optimizer.step()
                optimizer.zero_grad()
                metrics["train/steps"] += 1
                if metrics["train/steps"] == 1:
                    checkpoint_state = metrics.state_dict()

            _assert_all_ranks_equal(model.module.weight.detach(), world_size)

        counters = torch.tensor(
            [
                metrics["train/steps"],
                metrics["train/batches_in_epoch"],
                metrics["train/dataloader_batches_in_epoch"],
                metrics["train/epochs"],
            ],
            dtype=torch.long,
        )
        _assert_all_ranks_equal(counters, world_size)

        assert checkpoint_state is not None
        restored = Metrics()
        restored.load_state_dict(checkpoint_state)
        assert restored["train/dataloader_batches_in_epoch"] == 2
        uninterrupted_suffix = list(range(2, 4))
        resumed_suffix = list(range(restored["train/dataloader_batches_in_epoch"], 4))
        assert resumed_suffix == uninterrupted_suffix

        metrics["train/dataloader_batches_in_epoch"] += rank
        caught = torch.zeros((), dtype=torch.int32)
        try:
            metrics.state_dict()
        except RuntimeError as exc:
            assert "disagree on checkpoint resume position" in str(exc)
            caught.fill_(1)
        dist.all_reduce(caught, op=dist.ReduceOp.SUM)
        assert int(caught.item()) == world_size
    finally:
        dist.destroy_process_group()


def test_variable_packed_batches_keep_distributed_control_flow_aligned(
    tmp_path: Path,
) -> None:
    """Asymmetric packing keeps model calls/cursors aligned and resumable."""
    init_file = tmp_path / "packed-gloo-init"
    mp.spawn(
        _packed_distributed_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )
