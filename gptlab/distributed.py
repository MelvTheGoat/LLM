"""Small helpers for DistributedDataParallel (DDP).

DDP runs one process per GPU. Each process holds a full copy of the model and
works on a different part of the batch. After the backward pass, DDP averages
the gradients over all processes, so every copy takes the same optimizer step.

`torchrun --nproc_per_node=2 -m gptlab.train ...` starts the processes and sets
RANK, LOCAL_RANK and WORLD_SIZE. Without torchrun we run as a single process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device_type(self) -> str:
        return self.device.type


def setup(device: str | None = None) -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and device != "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        # A long timeout: rank 0 may be busy saving a checkpoint or running the
        # final evaluation while the others wait at a barrier.
        kwargs = dict(backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=30))
        try:
            dist.init_process_group(**kwargs, device_id=dev if use_cuda else None)
        except TypeError:  # PyTorch older than 2.3 has no device_id argument
            dist.init_process_group(**kwargs)
    return DistInfo(rank, local_rank, world, dev)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


def broadcast_flag(flag: bool, device: torch.device) -> bool:
    """Rank 0 decides (for example "stop now"); every rank gets the same answer."""
    if not dist.is_initialized():
        return flag
    t = torch.tensor([1 if flag else 0], device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())
