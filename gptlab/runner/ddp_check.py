"""Quick check that the GPUs can talk to each other, and how fast.

Run with torchrun. Rank 0 prints one JSON line with the all-reduce bandwidth.
The runner uses this at the start of a GPU session: if it hangs or fails, it
tries again with NCCL_P2P_DISABLE=1 (send data through CPU memory instead of
directly between GPUs), and if that fails too it falls back to one GPU.
"""

from __future__ import annotations

import json
import time

import torch
import torch.distributed as dist

from gptlab import distributed as du


def main() -> None:
    info = du.setup()
    n_bytes = 256 * 2**20 if info.device_type == "cuda" else 4 * 2**20  # small on CPU (tests)
    x = torch.ones(n_bytes // 4, device=info.device)

    def sync():
        if info.device_type == "cuda":
            torch.cuda.synchronize(info.device)

    for _ in range(2):  # warm up
        dist.all_reduce(x)
    sync()
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(x)
    sync()
    dt = (time.perf_counter() - t0) / reps
    n = info.world_size
    # "Bus bandwidth": data each GPU sends and receives in a ring all-reduce.
    bus_gbps = 2 * (n - 1) / n * n_bytes / dt / 1e9
    if info.is_main:
        print(json.dumps({"ddp_ok": True, "world_size": n, "tensor_mb": n_bytes / 2**20, "allreduce_ms": 1000 * dt,
                          "bus_gb_per_s": bus_gbps}), flush=True)
    du.cleanup()


if __name__ == "__main__":
    main()
