"""Throughput benchmark: tokens per second, MFU and memory for many settings.

    torchrun --standalone --nproc_per_node=2 -m gptlab.bench --config configs/bench/smoke.yaml --out out/bench

Each case builds a model, then times full training steps (forward, backward,
AdamW update; one micro-batch per step) on random tokens. Random tokens are
fine here: speed does not depend on what the tokens are, and it keeps data
loading out of the measurement.

- micro_batch_size 0 means "the largest power of two that fits in memory".
  The search runs on each GPU on its own (no DDP), then all GPUs use the
  smallest size any of them found.
- world_size 1 cases run on GPU 0 only while the other processes wait. So one
  torchrun launch can compare 1 GPU against 2 GPUs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from gptlab import distributed as du
from gptlab.config import ModelConfig, TrainConfig, apply_overrides, from_dict, read_yaml
from gptlab.flops import flops_per_token, mfu, peak_flops_per_gpu
from gptlab.model import GPT, param_counts
from gptlab.optim import build_optimizer


@dataclass
class BenchCase:
    model: str  # a key of BenchConfig.models
    precision: str = "fp16"
    compile: bool = False
    world_size: int = 0  # 0 = every launched process
    micro_batch_size: int = 0  # 0 = largest power of two that fits


@dataclass
class BenchConfig:
    name: str
    base_model: dict = field(default_factory=dict)  # settings shared by all models
    models: dict = field(default_factory=dict)  # name -> model settings
    cases: list[BenchCase] = field(default_factory=list)
    grid: list[dict] = field(default_factory=list)  # each: lists of values, expanded to all combinations
    warmup_steps: int = 3
    timed_steps: int = 10
    max_micro_batch: int = 64

    def all_cases(self) -> list[BenchCase]:
        out = list(self.cases)
        for g in self.grid:
            keys = list(g)
            values = [v if isinstance(v, list) else [v] for v in g.values()]
            for combo in itertools.product(*values):
                out.append(from_dict(BenchCase, dict(zip(keys, combo))))
        return out

    def model_config(self, name: str) -> ModelConfig:
        cfg = from_dict(ModelConfig, {**self.base_model, **self.models[name]})
        cfg.validate()
        return cfg


def load_bench_config(path, overrides=None) -> BenchConfig:
    cfg = from_dict(BenchConfig, apply_overrides(read_yaml(path), overrides or []))
    for case in cfg.all_cases():
        if case.model not in cfg.models:
            raise ValueError(f"bench case uses unknown model {case.model!r}")
        cfg.model_config(case.model)
    return cfg


def _autocast(precision: str, device_type: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type, dtype=dtype)


class _Runner:
    """One model + optimizer, ready to run timed training steps."""

    def __init__(self, mcfg: ModelConfig, precision: str, use_compile: bool, device, ddp: bool, local_rank: int):
        torch.manual_seed(0)
        self.raw = GPT(mcfg).to(device)
        self.opt = build_optimizer(self.raw, TrainConfig(max_steps=1), device.type)
        self.scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
        self.ctx = _autocast(precision, device.type)
        self.model = torch.compile(self.raw) if use_compile else self.raw
        if ddp:
            self.model = DDP(self.model, device_ids=[local_rank] if device.type == "cuda" else None)
        self.device = device
        self.cfg = mcfg

    def step(self, x, y) -> None:
        with self.ctx:
            _, loss, _ = self.model(x, y)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _free(device) -> None:
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _batch(mcfg: ModelConfig, n: int, device):
    g = torch.Generator().manual_seed(0)
    t = torch.randint(0, mcfg.vocab_size, (n, mcfg.seq_len + 1), generator=g)
    return t[:, :-1].to(device), t[:, 1:].to(device)


def _fits(mcfg, precision, device, micro: int) -> bool:
    """Can one eager training step with this micro-batch run without running out of memory?"""
    runner = None
    try:
        runner = _Runner(mcfg, precision, False, device, False, 0)
        x, y = _batch(mcfg, micro, device)
        runner.step(x, y)
        runner.step(x, y)
        _sync(device)
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del runner
        _free(device)


def find_micro_batch(mcfg, precision, device, max_micro: int) -> int:
    if device.type != "cuda":
        return max_micro
    micro = max_micro
    while micro >= 1:
        if _fits(mcfg, precision, device, micro):
            return micro
        micro //= 2
    return 0


def run_case(bcfg: BenchConfig, case: BenchCase, info: du.DistInfo) -> dict | None:
    """Run one case. Returns the result on rank 0, None on other ranks."""
    mcfg = bcfg.model_config(case.model)
    world = case.world_size or info.world_size
    if world not in (1, info.world_size):
        raise ValueError(f"world_size {world} not possible with {info.world_size} processes")
    participates = world == info.world_size or info.rank == 0
    result = {"model": case.model, "precision": case.precision, "compile": case.compile, "world_size": world}
    counts = param_counts(GPT(mcfg))
    result.update(params=counts["total"], non_embedding_params=counts["non_embedding"],
                  d_model=mcfg.d_model, n_layer=mcfg.n_layer, seq_len=mcfg.seq_len)

    micro = case.micro_batch_size
    if micro == 0:
        found = find_micro_batch(mcfg, case.precision, info.device, bcfg.max_micro_batch) if participates else 10**6
        t = torch.tensor([found], device=info.device)
        if info.world_size > 1:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        micro = int(t.item())
    result["micro_batch_size"] = micro
    if micro == 0:
        result["status"] = "oom"
        return result if info.is_main else None

    status, error, times, peak_mem, first_step = "ok", None, [], None, None
    if participates:
        runner = None
        try:
            runner = _Runner(mcfg, case.precision, case.compile, info.device, world > 1, info.local_rank)
            x, y = _batch(mcfg, micro, info.device)
            for i in range(bcfg.warmup_steps + bcfg.timed_steps):
                _sync(info.device)
                t0 = time.perf_counter()
                runner.step(x, y)
                _sync(info.device)
                dt = time.perf_counter() - t0
                if i == 0:
                    first_step = dt
                if i >= bcfg.warmup_steps:
                    times.append(dt)
            if info.device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(info.device) / 2**30
        except torch.cuda.OutOfMemoryError:
            status = "oom"  # both GPUs have the same load, so both run out together
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {str(e)[:500]}"
        finally:
            del runner
            if case.compile:
                torch._dynamo.reset()
            _free(info.device)
    du.barrier()
    if not info.is_main:
        return None
    result["status"] = status
    if error:
        result["error"] = error
    if times:
        step_time = statistics.median(times)
        tokens_per_step = micro * mcfg.seq_len * world
        tok_s = tokens_per_step / step_time
        dev_name = torch.cuda.get_device_name(info.device) if info.device.type == "cuda" else "cpu"
        peak = peak_flops_per_gpu(dev_name, case.precision) if info.device.type == "cuda" else None
        fpt = flops_per_token(mcfg)
        result.update(
            tokens_per_step=tokens_per_step, step_time_ms=1000 * step_time,
            step_time_spread_ms=1000 * (max(times) - min(times)), tok_per_s=tok_s,
            tok_per_s_per_gpu=tok_s / world, mfu=mfu(tok_s, fpt["total"], world, peak),
            flops_per_token=fpt["total"], attention_flops_share=fpt["attention_term"] / fpt["total"],
            peak_mem_gb=peak_mem, first_step_s=first_step, device=dev_name,
        )
    return result


def run_bench(bcfg: BenchConfig, out_dir: str | Path, device: str | None = None, log=print) -> list[dict]:
    info = du.setup(device)
    out_dir = Path(out_dir)
    if info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "world_size": info.world_size}
    for case in bcfg.all_cases():
        r = run_case(bcfg, case, info)
        if info.is_main:
            results.append(r)
            with open(out_dir / "bench.json", "w") as f:
                json.dump({"name": bcfg.name, "env": env, "results": results}, f, indent=2)
            if r.get("tok_per_s"):
                m = f"{100 * r['mfu']:.1f}%" if r.get("mfu") else "n/a"
                log(f"[bench] {r['model']:>6} {r['precision']} compile={r['compile']!s:5} gpus={r['world_size']} "
                    f"micro={r['micro_batch_size']:>3} | {r['tok_per_s']:>10,.0f} tok/s | mfu {m} | "
                    f"mem {r['peak_mem_gb'] or 0:.1f} GB | first step {r['first_step_s']:.1f}s")
            else:
                log(f"[bench] {r['model']} {r['precision']} compile={r['compile']} gpus={r['world_size']}: "
                    f"{r['status']} {r.get('error', '')}")
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Measure training throughput.")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    try:
        run_bench(load_bench_config(args.config, args.set), args.out, args.device, log=lambda m: print(m, flush=True))
    finally:
        du.cleanup()


if __name__ == "__main__":
    main()
