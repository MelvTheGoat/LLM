"""The training loop.

Single process (CPU or one GPU):
    python -m gptlab.train --config configs/smoke/train.yaml
Two GPUs with DDP:
    torchrun --standalone --nproc_per_node=2 -m gptlab.train --config configs/smoke/train.yaml

What one optimizer step does:
1. Set the learning rate for this step (warmup, then cosine decay).
2. For each of `accum` micro-batches: forward pass under fp16 autocast, then
   backward on the loss scaled by the GradScaler. Gradients add up over
   micro-batches (gradient accumulation), so one step sees the whole global
   batch even when it does not fit in memory at once. With DDP, gradients are
   averaged across GPUs only on the last micro-batch (no_sync on the others).
3. Unscale the gradients, clip their total norm, and take the AdamW step. With
   fp16, if any gradient overflowed to inf/NaN, the scaler skips the step and
   lowers the scale.
4. Log loss, LR, gradient norm, tokens/s, MFU and memory. Now and then: run a
   validation pass, collect per-layer debug stats, save a checkpoint.

fp16 and the loss scaler: fp16 numbers can only get as small as about 6e-8.
Small gradients would round to zero ("underflow"). The GradScaler multiplies
the loss by a large factor (e.g. 65536) before backward, so gradients are
large enough to keep, then divides them back before the optimizer step. It
lowers the factor when it sees overflow and slowly raises it otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from gptlab import distributed as du
from gptlab.checkpoint import (
    LOG_FILES,
    latest_checkpoint,
    load_checkpoint,
    rng_state,
    save_checkpoint,
    save_final_weights,
    set_rng_state,
    step_dir_name,
)
from gptlab.config import RunConfig, load_config, save_config, to_dict
from gptlab.data.loader import TrainStream, ValData, resolve_data_dir
from gptlab.data.tokenizer import Tokenizer
from gptlab.evaluate import evaluate_val, final_evaluation, write_final_eval
from gptlab.flops import flops_per_token, mfu, peak_flops_per_gpu
from gptlab.metrics import JsonlWriter, forward_stats, grad_and_weight_norms, truncate_jsonl
from gptlab.model import GPT, param_counts
from gptlab.optim import build_optimizer, lr_at

# Settings that may change between sessions of the same run without changing
# the maths of training (so a resume is still exact).
_SAFE_TO_CHANGE = {
    "print_interval", "debug_interval", "ckpt_interval_minutes", "ckpt_interval_steps",
    "keep_checkpoints", "diverge_patience", "diverge_loss", "compile",
}


def training_hash(cfg: RunConfig) -> str:
    d = to_dict(cfg)
    d.pop("notes", None)
    d.pop("eval", None)
    for k in _SAFE_TO_CHANGE:
        d["train"].pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def git_commit() -> str | None:
    if os.environ.get("GPTLAB_GIT_COMMIT"):
        return os.environ["GPTLAB_GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info(info: du.DistInfo) -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "world_size": info.world_size,
        "device_type": info.device_type,
        "git_commit": git_commit(),
    }
    if info.device_type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(info.device)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(info.device))
        env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(info.device).total_memory / 2**30, 1)
    return env


def _autocast(precision: str, info: du.DistInfo):
    if precision == "fp32":
        return nullcontext()
    if info.device_type != "cuda":
        if precision == "fp16":
            raise ValueError("fp16 training needs a GPU; use train.precision=fp32 on CPU")
        return torch.autocast("cpu", dtype=torch.bfloat16)
    if precision == "bf16" and torch.cuda.get_device_capability(info.device) < (8, 0):
        raise ValueError("this GPU has no bf16 support (e.g. T4); use train.precision=fp16")
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def train(
    cfg: RunConfig,
    out_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    deadline: float | None = None,
    stop_after_steps: int | None = None,
    device: str | None = None,
    hellaswag_path: str | Path | None = None,
    log=None,
) -> dict:
    """Train (or resume) one run. Returns the final run state."""
    info = du.setup(device)
    rank, world, dev = info.rank, info.world_size, info.device
    is_main = info.is_main

    def say(msg: str) -> None:
        if is_main:
            (log or (lambda m: print(m, flush=True)))(f"[{cfg.name}] {msg}")

    out_dir = Path(out_dir)
    ckpt_root = out_dir / "ckpt"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    du.barrier()
    tc = cfg.train
    torch.backends.cuda.matmul.allow_tf32 = True  # no effect on T4; faster fp32 on newer GPUs
    torch.backends.cudnn.allow_tf32 = True

    # ---- data -----------------------------------------------------------
    data_dir = Path(data_dir) if data_dir else resolve_data_dir(cfg.data.name, cfg.data.root)
    tok_vocab = Tokenizer.load(data_dir / "tokenizer.json").vocab_size
    if tok_vocab > cfg.model.vocab_size:
        raise ValueError(f"tokenizer has {tok_vocab} tokens but model.vocab_size is {cfg.model.vocab_size}")
    T = cfg.model.seq_len
    stream = TrainStream.from_dir(data_dir, T, cfg.data.seed, cfg.data.max_train_shards)
    val_data = ValData.from_dir(data_dir, T)

    # ---- batch geometry -------------------------------------------------
    per_micro = tc.micro_batch_size * T * world
    if tc.global_batch_tokens % per_micro != 0:
        raise ValueError(
            f"global_batch_tokens ({tc.global_batch_tokens}) must be a multiple of "
            f"micro_batch_size * seq_len * world_size ({per_micro})"
        )
    accum = tc.global_batch_tokens // per_micro
    seqs_per_step = tc.global_batch_tokens // T
    total_steps = tc.max_steps or math.ceil(tc.tokens / tc.global_batch_tokens)

    # ---- model, optimizer, scaler ----------------------------------------
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)  # same initial weights on every GPU
    raw_model = GPT(cfg.model).to(dev)
    optimizer = build_optimizer(raw_model, tc, info.device_type)
    scaler = torch.amp.GradScaler(info.device_type, enabled=tc.precision == "fp16")
    autocast_ctx = _autocast(tc.precision, info)
    counts = param_counts(raw_model)
    fpt = flops_per_token(cfg.model)
    peak = peak_flops_per_gpu(torch.cuda.get_device_name(dev), tc.precision) if info.device_type == "cuda" else None
    thash = training_hash(cfg)

    # ---- resume or start fresh -------------------------------------------
    start_step, resumed, rng_exact = 0, None, True
    ckpt = latest_checkpoint(ckpt_root)
    if ckpt is not None:
        state = load_checkpoint(
            ckpt, model=raw_model, optimizer=optimizer, scaler=scaler, rank=rank, world_size=world, map_location=dev
        )
        if state.get("training_hash") != thash:
            raise ValueError(
                f"config changed since the checkpoint ({state.get('training_hash')} != {thash}); "
                "use a new run name for a different experiment"
            )
        if state.get("data_fingerprint") != stream.fingerprint():
            raise ValueError("training data changed since the checkpoint")
        start_step, resumed, rng_exact = state["step"], ckpt.name, state["rng_exact"]
        if is_main:  # restore logs exactly as they were at the checkpoint
            for name in LOG_FILES:
                src = ckpt / "logs" / name
                if src.exists():
                    (out_dir / name).write_bytes(src.read_bytes())
                truncate_jsonl(out_dir / name, start_step)
    elif is_main:
        for name in LOG_FILES + ["final_eval.json", "samples.txt"]:
            (out_dir / name).unlink(missing_ok=True)  # leftovers of a run that never saved
    if is_main:
        save_config(cfg, out_dir / "config.yaml")
    # Each GPU gets its own random stream for dropout.
    torch.manual_seed(cfg.seed + 1 + rank)
    if resumed and state.get("rng") is not None:
        set_rng_state(state["rng"])

    model = raw_model
    if tc.compile:
        model = torch.compile(model)
    if world > 1:
        model = DDP(model, device_ids=[info.local_rank] if info.device_type == "cuda" else None)

    metrics = JsonlWriter(out_dir / "metrics.jsonl", is_main)
    evals = JsonlWriter(out_dir / "eval.jsonl", is_main)
    debug = JsonlWriter(out_dir / "debug.jsonl", is_main)
    events = JsonlWriter(out_dir / "events.jsonl", is_main)
    env = environment_info(info)
    events.write({
        "event": "session_start", "step": start_step, "time": time.time(), "resumed_from": resumed,
        "rng_exact": rng_exact, "accum": accum, "total_steps": total_steps, "params": counts,
        "flops_per_token": fpt, "peak_flops_per_gpu": peak, "train_tokens_available": stream.total_tokens, **env,
    })
    if is_main:
        _write_json(out_dir / "env.json", env)
        _write_json(out_dir / "run_state.json", {"state": "running", "step": start_step, "total_steps": total_steps})
    say(
        f"{'resuming from ' + resumed if resumed else 'starting'} | step {start_step}/{total_steps} | "
        f"params {counts['total']:,} (non-embedding {counts['non_embedding']:,}) | world {world} | accum {accum} | "
        f"{env.get('gpu', 'cpu')}"
    )
    if resumed and not rng_exact:
        say("warning: world size changed, random states cannot be restored exactly")
    if info.device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    diverge_loss = tc.diverge_loss if tc.diverge_loss is not None else math.log(cfg.model.vocab_size) + 2.0
    step = start_step
    final_state, reason = "running", ""
    bad_steps = 0
    last_ckpt_time = time.time()
    interval_start, interval_tokens = time.perf_counter(), 0
    rng_before_step = rng_state()
    model.train()

    def checkpoint(at_step: int, rng=None) -> None:
        save_checkpoint(
            ckpt_root, at_step, model=raw_model, optimizer=optimizer, scaler=scaler,
            state={"training_hash": thash, "data_fingerprint": stream.fingerprint(), "world_size": world,
                   "seqs_consumed": at_step * seqs_per_step, "total_steps": total_steps},
            rank=rank, run_dir=out_dir, rng=rng, keep=tc.keep_checkpoints,
        )

    try:
        t_last = time.perf_counter()
        while step < total_steps:
            stop = False
            if is_main:
                if deadline is not None and time.time() >= deadline:
                    stop, reason = True, "deadline"
                if stop_after_steps is not None and step - start_step >= stop_after_steps:
                    stop, reason = True, "stop_after_steps"
            if du.broadcast_flag(stop, dev):
                # Log first, so the log copy inside the checkpoint includes this event.
                events.write({"event": "paused", "step": step, "reason": reason, "time": time.time()})
                checkpoint(step)
                final_state = "paused"
                say(f"pausing at step {step} ({reason}); checkpoint saved")
                break

            rng_before_step = rng_state()  # lets an emergency checkpoint resume exactly
            lr = lr_at(step, total_steps, tc)
            for group in optimizer.param_groups:
                group["lr"] = lr
            sums = torch.zeros(3, device=dev)  # loss, ce, z
            for micro in range(accum):
                x, y = stream.batch(step, micro, rank, world, tc.micro_batch_size, accum, dev)
                sync = model.no_sync() if (world > 1 and micro < accum - 1) else nullcontext()
                with sync:
                    with autocast_ctx:
                        _, loss, parts = model(x, y, z_loss_weight=tc.z_loss)
                    scaler.scale(loss / accum).backward()
                sums += torch.stack([loss.detach().float(), parts["ce"].float(), parts["z"].float()])
            scaler.unscale_(optimizer)
            do_debug = is_main and tc.debug_interval and (step + 1) % tc.debug_interval == 0
            layer_norms = grad_and_weight_norms(raw_model) if do_debug else None
            max_norm = tc.grad_clip if tc.grad_clip > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            du.all_reduce_mean(sums)
            loss_v, ce_v, z_v = (sums / accum).tolist()
            gn = grad_norm.item()
            now = time.perf_counter()
            dt = now - t_last
            tok_s = tc.global_batch_tokens / dt
            record = {
                "step": step, "tokens": step * tc.global_batch_tokens, "loss": loss_v, "ce": ce_v, "z": z_v,
                "lr": lr, "grad_norm": gn, "dt": dt, "tok_per_s": tok_s,
                "mfu": mfu(tok_s, fpt["total"], world, peak), "time": time.time(),
            }
            if scaler.is_enabled():
                record["scale"] = scale_before
                record["skipped"] = scaler.get_scale() < scale_before
            if info.device_type == "cuda" and (step % tc.print_interval == 0 or step == start_step + 1):
                record["mem_alloc_gb"] = torch.cuda.memory_allocated(dev) / 2**30
                record["mem_peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
                record["mem_reserved_gb"] = torch.cuda.memory_reserved(dev) / 2**30
            metrics.write(record)
            interval_tokens += tc.global_batch_tokens
            if step % tc.print_interval == 0 or step == total_steps:
                elapsed = time.perf_counter() - interval_start
                rate = interval_tokens / max(elapsed, 1e-9)
                m = mfu(rate, fpt["total"], world, peak)
                eta_h = (total_steps - step) * tc.global_batch_tokens / max(rate, 1e-9) / 3600
                say(
                    f"step {step}/{total_steps} | loss {loss_v:.4f} | lr {lr:.2e} | gnorm {gn:.3f} | "
                    f"{rate:,.0f} tok/s" + (f" | mfu {100 * m:.1f}%" if m else "") + f" | eta {eta_h:.2f}h"
                )
                interval_start, interval_tokens = time.perf_counter(), 0

            bad = not math.isfinite(loss_v) or loss_v > diverge_loss
            bad_steps = bad_steps + 1 if bad else 0
            if bad_steps >= tc.diverge_patience:
                final_state, reason = "diverged", f"{bad_steps} bad steps in a row (loss {loss_v})"
                events.write({"event": "diverged", "step": step, "reason": reason, "time": time.time()})
                say(f"stopping: diverged ({reason})")
                break

            if cfg.eval.interval and step % cfg.eval.interval == 0 and step < total_steps:
                v = evaluate_val(raw_model, val_data, cfg.eval.tokens, tc.micro_batch_size, dev, autocast_ctx,
                                 rank=rank, world_size=world)
                evals.write({"step": step, "tokens": step * tc.global_batch_tokens, **v})
                say(f"step {step} | val loss {v['val_loss']:.4f}")
            if do_debug:
                n = min(4, x.shape[0])
                s = forward_stats(raw_model, x[:n], y[:n], autocast_ctx)
                debug.write({"step": step, **s, **layer_norms})
            ckpt_due = False
            if is_main:
                ckpt_due = time.time() - last_ckpt_time >= tc.ckpt_interval_minutes * 60
            if tc.ckpt_interval_steps and step % tc.ckpt_interval_steps == 0:
                ckpt_due = True
            if du.broadcast_flag(ckpt_due, dev) and step < total_steps:
                checkpoint(step)
                last_ckpt_time = time.time()
            t_last = time.perf_counter()  # eval/debug/checkpoint time is not counted as step time

        if step >= total_steps and final_state == "running":
            need_ckpt = is_main and not (ckpt_root / step_dir_name(step)).exists()
            if du.broadcast_flag(need_ckpt, dev):  # one decision for all processes
                checkpoint(step)
            if is_main:
                save_final_weights(out_dir, raw_model, {"step": step, "tokens": step * tc.global_batch_tokens,
                                                        "params": counts, "training_hash": thash})
            events.write({"event": "training_done", "step": step, "time": time.time()})
            result = final_evaluation(raw_model, cfg, data_dir, dev, autocast_ctx, rank, world, hellaswag_path, say)
            if is_main:
                result.update({"step": step, "tokens": step * tc.global_batch_tokens, "params": counts})
                write_final_eval(out_dir, result)
            final_state = "done"
    except BaseException as e:
        final_state, reason = "failed", f"{type(e).__name__}: {e}"
        events.write({"event": "failed", "step": step, "error": reason, "traceback": traceback.format_exc(),
                      "time": time.time()})
        # A single process can save a checkpoint of the last finished step.
        # (With DDP the other processes may be stuck, so we rely on the
        # periodic checkpoints instead.)
        if world == 1 and step > start_step and not isinstance(e, KeyboardInterrupt):
            try:
                checkpoint(step, rng=rng_before_step)
                say(f"emergency checkpoint saved at step {step}")
            except Exception as e2:
                say(f"emergency checkpoint failed: {e2!r}")
        raise
    finally:
        run_state = {"state": final_state, "step": step, "total_steps": total_steps, "reason": reason,
                     "tokens": step * tc.global_batch_tokens, "updated_unix": time.time()}
        if is_main:
            _write_json(out_dir / "run_state.json", run_state)
        for w in (metrics, evals, debug, events):
            w.close()
    du.barrier()
    return run_state


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train a GPT model from a YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], help="override a config value, e.g. train.lr=3e-4")
    p.add_argument("--out-dir", default=None, help="run folder (default: out/<name>)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--deadline", type=float, default=None, help="unix time at which to save and stop")
    p.add_argument("--stop-after-steps", type=int, default=None, help="save and stop after N steps this session")
    p.add_argument("--device", default=None, help="'cpu' to force CPU")
    p.add_argument("--hellaswag-path", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    try:
        train(
            cfg, args.out_dir or f"out/{cfg.name}", data_dir=args.data_dir, deadline=args.deadline,
            stop_after_steps=args.stop_after_steps, device=args.device, hellaswag_path=args.hellaswag_path,
        )
    finally:
        du.cleanup()


if __name__ == "__main__":
    main()
