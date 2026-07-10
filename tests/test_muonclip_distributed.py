"""True-process regressions for replicated MuonClip Q/K clipping."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from neobert.model import NeoBERTConfig, NeoBERTLMHead
from neobert.optimizer.muon_clip import MuonClipConfig, MuonClipOptimizer


def _assert_replicated(tensor: torch.Tensor, world_size: int) -> None:
    """Assert a tensor is identical on every initialized rank.

    :param torch.Tensor tensor: Rank-local tensor.
    :param int world_size: Process-group size.
    """
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    for replica in gathered[1:]:
        torch.testing.assert_close(replica, gathered[0], rtol=0.0, atol=0.0)


def _run_ddp_clipping_case(rank: int, world_size: int, capture_last_only: bool) -> None:
    """Run two accumulated DDP updates and verify post-clip replica equality.

    :param int rank: Process rank.
    :param int world_size: Process-group size.
    :param bool capture_last_only: MuonClip activation-capture policy.
    """
    torch.manual_seed(1234)
    config = NeoBERTConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        dropout=0.0,
        vocab_size=16,
        max_length=4,
        attn_backend="sdpa",
        hidden_act="gelu",
        rope=False,
        tie_word_embeddings=False,
    )
    model = NeoBERTLMHead(config)
    with torch.no_grad():
        model.model.encoder.weight.zero_()
        model.model.encoder.weight[1].fill_(1.0)
        model.model.encoder.weight[2].fill_(3.0)
        model.model.positional_embedding.weight.zero_()
        model.model.transformer_encoder[0].qkv.weight.fill_(0.25)

    optimizer = MuonClipOptimizer(
        model,
        config,
        MuonClipConfig(
            enable_clipping=True,
            clipping_threshold=0.01,
            clipping_interval=1,
            capture_last_microbatch_only=capture_last_only,
            orthogonalization="newton_schulz",
            ns_steps=2,
        ),
    )
    ddp_model = DistributedDataParallel(model)
    input_ids = torch.full((1, 3), rank + 1, dtype=torch.long)

    for update_step in range(2):
        optimizer.zero_grad()
        for microstep in range(2):
            is_last_microbatch = microstep == 1
            optimizer.prepare_for_forward(
                update_step=update_step,
                is_last_microbatch=is_last_microbatch,
            )
            sync_context = nullcontext() if is_last_microbatch else ddp_model.no_sync()
            with sync_context:
                logits = ddp_model(src=input_ids)["logits"]
                (logits.float().sum() / 2.0).backward()

        optimizer.step()
        qkv = model.model.transformer_encoder[0].qkv.weight.detach()
        _assert_replicated(qkv, world_size)


def _distributed_muonclip_worker(rank: int, world_size: int, init_file: str) -> None:
    """Exercise global clipping statistics and DDP parameter mutation.

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
        local_max = (
            torch.tensor([[float("-inf")], [100.0]])
            if rank == 0
            else torch.tensor([[300.0]])
        )
        eta = MuonClipOptimizer._eta_from_per_step_max(local_max, 50.0)
        torch.testing.assert_close(eta, torch.tensor([0.25]))

        replicated_weight = torch.ones(1)
        replicated_weight.mul_(eta)
        _assert_replicated(replicated_weight, world_size)

        for capture_last_only in (True, False):
            _run_ddp_clipping_case(rank, world_size, capture_last_only)

        for invalid_rank, invalid_value in (
            (0, float("nan")),
            (1, float("inf")),
        ):
            local_value = invalid_value if rank == invalid_rank else 100.0
            caught = torch.zeros((), dtype=torch.int32)
            try:
                MuonClipOptimizer._eta_from_per_step_max(
                    torch.tensor([[local_value]]),
                    50.0,
                )
            except FloatingPointError:
                caught.fill_(1)
            dist.all_reduce(caught, op=dist.ReduceOp.SUM)
            assert int(caught.item()) == world_size
    finally:
        dist.destroy_process_group()


def test_ddp_qk_clipping_uses_global_statistics(tmp_path: Path) -> None:
    """DDP clipping keeps replicas equal and fails globally on invalid logits."""
    init_file = tmp_path / "muonclip-gloo-init"
    mp.spawn(
        _distributed_muonclip_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )
