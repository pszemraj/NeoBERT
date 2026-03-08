"""Manual smoke test for Accelerate FSDP2 Muon resume plumbing.

This validates the production checkpoint path rather than the lower-level DCP
helpers alone:

- construct MuonClip via ``get_optimizer(...)``,
- shard through Accelerate FSDP2 ``prepare`` / ``prepare_data_loader``,
- run one step and ``accelerator.save_state(...)``,
- rebuild fresh objects and ``accelerator.load_state(...)``,
- run the next step and compare against uninterrupted continuation.

Run on a CUDA machine with 2 ranks:
`conda run -s --name neobert torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_accelerate_fsdp2_resume.py`
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedType,
    FullyShardedDataParallelPlugin,
    ProjectConfiguration,
)
from torch.utils.data import DataLoader

from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import get_optimizer
from neobert.training_utils import (
    _reset_accelerate_runtime_state,
    _unwrap_optimizer,
    create_accelerator,
    validate_muon_distributed_compatibility,
    validate_muon_runtime_topology,
)

logger = logging.getLogger(__name__)
_MAX_DIFF = 5e-5


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", str(_get_rank())))


def _init_dist() -> tuple[int, int]:
    local_rank = _get_local_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def _make_shared_tmpdir() -> Path:
    root: str | None = None
    if dist.get_rank() == 0:
        root = tempfile.mkdtemp(prefix="neobert_muon_accel_resume_")
    payload = [root]
    dist.broadcast_object_list(payload, src=0)
    shared_root = payload[0]
    assert shared_root is not None
    return Path(shared_root)


def _make_config() -> NeoBERTConfig:
    return NeoBERTConfig(
        hidden_size=15,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=60,
        vocab_size=257,
        max_length=32,
        attn_backend="sdpa",
        hidden_act="gelu",
        rope=False,
    )


def _make_dataset(config: NeoBERTConfig) -> list[dict[str, torch.Tensor]]:
    batches = []
    for step in range(8):
        base = 1 + step * 22
        src = torch.arange(base, base + 22, dtype=torch.long).view(2, 11)
        src = src.remainder(config.vocab_size - 1).add_(1)
        batches.append({"src": src})
    return batches


def _build_accelerator(project_dir: Path):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["EncoderBlock"],
        state_dict_type="SHARDED_STATE_DICT",
    )
    # This smoke feeds Accelerate pre-batched dict samples (batch_size=None), so
    # newer Accelerate releases require even_batches=False when sharding it.
    dataloader_config = DataLoaderConfiguration(even_batches=False)
    project_config = ProjectConfiguration(
        project_dir=str(project_dir),
        automatic_checkpoint_naming=False,
    )
    accelerator = create_accelerator(
        use_cpu=False,
        log=logger,
        dataloader_config=dataloader_config,
        project_config=project_config,
        fsdp_plugin=fsdp_plugin,
        step_scheduler_with_optimizer=False,
    )
    if accelerator.distributed_type is not DistributedType.FSDP:
        raise RuntimeError(
            "Expected Accelerate FSDP runtime for this manual smoke, got "
            f"{accelerator.distributed_type!r}."
        )
    if int(getattr(accelerator.state.fsdp_plugin, "fsdp_version", 1)) != 2:
        raise RuntimeError("Manual smoke expects Accelerate FSDP v2.")
    return accelerator


def _build_prepared_run(project_dir: Path):
    accelerator = _build_accelerator(project_dir)
    config = _make_config()
    model = NeoBERT(config)
    optimizer = get_optimizer(
        model,
        accelerator.distributed_type,
        model_config=config,
        name="muonclip",
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        muon_config={
            "enable_clipping": False,
            "orthogonalization": "polar_express",
            "norm_factor": "spectral",
        },
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.5,
        total_iters=2,
    )
    dataloader = DataLoader(_make_dataset(config), batch_size=None, shuffle=False)
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    validate_muon_distributed_compatibility(
        accelerator=accelerator,
        optimizer_name="muonclip",
        log=logger,
        context="manual accelerate fsdp2 muon resume smoke",
    )
    validate_muon_runtime_topology(
        accelerator=accelerator,
        optimizer=optimizer,
        optimizer_name="muonclip",
        log=logger,
        context="manual accelerate fsdp2 muon resume smoke",
    )
    return accelerator, model, optimizer, scheduler, dataloader


def _run_step(
    accelerator: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    batch: dict[str, torch.Tensor],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    loss = output.float().square().mean()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    return float(loss.detach().item())


def _full_parameter_state(model: Any) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if hasattr(param, "full_tensor"):
            tensor = param.full_tensor().detach().cpu()
        else:
            tensor = param.detach().cpu()
        state[name] = tensor.clone()
    return state


def _max_state_diff(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[float, str]:
    max_diff = 0.0
    max_name = ""
    for name, ref_tensor in reference.items():
        diff = float((candidate[name] - ref_tensor).abs().max().item())
        if diff > max_diff:
            max_diff = diff
            max_name = name
    return max_diff, max_name


def _metadata_summary(optimizer: Any) -> list[tuple[bool, int, bool]]:
    summary: list[tuple[bool, int, bool]] = []
    for group in _unwrap_optimizer(optimizer).param_groups:
        param_info = list(group.get("param_info", ()))
        summary.append(
            (
                bool(group.get("use_muon", False)),
                len(param_info),
                "beta" in group,
            )
        )
    return summary


def main() -> None:
    rank = _get_rank()
    if not torch.cuda.is_available():
        if rank == 0:
            print("CUDA is required for this manual Accelerate FSDP2 smoke test.")
        return

    rank, world_size = _init_dist()
    if world_size != 2:
        raise RuntimeError(
            f"This manual smoke expects world_size=2, got world_size={world_size}."
        )

    ckpt_root = _make_shared_tmpdir()
    checkpoint_dir = ckpt_root / "checkpoint_step1"
    try:
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)

        accelerator, model, optimizer, scheduler, dataloader = _build_prepared_run(
            ckpt_root / "reference"
        )
        data_iter = iter(dataloader)
        batch_1 = next(data_iter)
        batch_2 = next(data_iter)

        _run_step(accelerator, model, optimizer, scheduler, batch_1)
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir=str(checkpoint_dir))
        metadata_after_save = _metadata_summary(optimizer)

        loss_2_reference = _run_step(accelerator, model, optimizer, scheduler, batch_2)
        reference_state = _full_parameter_state(model)
        reference_lr = float(_unwrap_optimizer(optimizer).param_groups[0]["lr"])
        reference_step = int(getattr(_unwrap_optimizer(optimizer), "_step"))

        accelerator.wait_for_everyone()
        del dataloader, scheduler, optimizer, model, accelerator, data_iter, batch_1
        gc.collect()
        torch.cuda.empty_cache()
        _reset_accelerate_runtime_state()

        accelerator2, model2, optimizer2, scheduler2, dataloader2 = _build_prepared_run(
            ckpt_root / "resume"
        )
        accelerator2.load_state(str(checkpoint_dir))
        loaded_inner = _unwrap_optimizer(optimizer2)
        if int(getattr(loaded_inner, "_step")) != 1:
            raise AssertionError(
                "Expected a single restored Muon step after Accelerate load_state."
            )
        if _metadata_summary(optimizer2) != metadata_after_save:
            raise AssertionError("Muon param-group metadata changed across load_state.")

        resumed_iter = iter(dataloader2)
        _ = next(resumed_iter)
        resumed_batch_2 = next(resumed_iter)
        loss_2_resumed = _run_step(
            accelerator2,
            model2,
            optimizer2,
            scheduler2,
            resumed_batch_2,
        )
        resumed_state = _full_parameter_state(model2)
        resumed_lr = float(loaded_inner.param_groups[0]["lr"])
        resumed_step = int(getattr(loaded_inner, "_step"))

        max_diff, max_name = _max_state_diff(reference_state, resumed_state)
        if max_diff > _MAX_DIFF:
            raise AssertionError(
                "Accelerate FSDP2 Muon resume drifted after load_state: "
                f"param={max_name} max_diff={max_diff:.6e}"
            )
        if resumed_step != reference_step:
            raise AssertionError(
                "MuonClip step counter mismatch after Accelerate resume: "
                f"resumed={resumed_step} reference={reference_step}"
            )
        if abs(resumed_lr - reference_lr) > 1e-12:
            raise AssertionError(
                "Scheduler/optimizer LR mismatch after Accelerate resume: "
                f"resumed={resumed_lr:.12f} reference={reference_lr:.12f}"
            )
        if abs(loss_2_resumed - loss_2_reference) > 1e-6:
            raise AssertionError(
                "Resumed second-step loss mismatch after Accelerate resume: "
                f"resumed={loss_2_resumed:.6e} reference={loss_2_reference:.6e}"
            )
        if rank == 0:
            print("PASS: Accelerate FSDP2 Muon save/load smoke succeeded.")
    finally:
        try:
            if dist.is_initialized():
                dist.barrier()
            _reset_accelerate_runtime_state()
            if rank == 0:
                shutil.rmtree(ckpt_root, ignore_errors=True)
            if dist.is_initialized():
                dist.barrier()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
