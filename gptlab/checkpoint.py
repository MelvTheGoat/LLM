"""Checkpoints with exact resume.

A checkpoint folder `ckpt/step_0001234/` holds everything needed to continue
training as if it had never stopped:

- model.pt       model weights (float32)
- optim.pt       AdamW state (moment estimates for every weight)
- state.pt       step number, fp16 loss scaler state, data position, and more
- rng_rank{r}.pt random number generator states of each process (Python,
                 numpy, torch CPU and CUDA). Dropout draws from these.
- logs/          a copy of the run's log files up to this step
- meta.json      small summary (step, time), written last

The LR schedule needs no saved state: it is a function of the step number. The
data position is also just the step number (see gptlab/data/loader.py).

Saving is atomic: files go into `step_X.tmp/`, which is renamed to `step_X/`
when complete. Then the text file `ckpt/LATEST` is updated. A crash while saving
leaves the previous checkpoint untouched.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from gptlab import distributed as du

LOG_FILES = ["metrics.jsonl", "eval.jsonl", "debug.jsonl", "events.jsonl"]
UPLOAD_MARKER = ".uploading"  # the Kaggle runner puts this in a folder while uploading it


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def step_dir_name(step: int) -> str:
    return f"step_{step:07d}"


def save_checkpoint(
    ckpt_root: Path,
    step: int,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    state: dict,
    rank: int,
    run_dir: Path,
    rng: dict | None = None,
    keep: int = 2,
) -> Path:
    """Save a checkpoint. Every process must call this (each saves its RNG state)."""
    ckpt_root = Path(ckpt_root)
    final = ckpt_root / step_dir_name(step)
    tmp = ckpt_root / (step_dir_name(step) + ".tmp")
    if rank == 0:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    du.barrier()
    torch.save(rng if rng is not None else rng_state(), tmp / f"rng_rank{rank}.pt")
    du.barrier()
    if rank == 0:
        torch.save(model.state_dict(), tmp / "model.pt")
        torch.save(optimizer.state_dict(), tmp / "optim.pt")
        torch.save({**state, "step": step, "scaler": scaler.state_dict()}, tmp / "state.pt")
        logs = tmp / "logs"
        logs.mkdir()
        for name in LOG_FILES + ["config.yaml"]:
            src = Path(run_dir) / name
            if src.exists():
                shutil.copy2(src, logs / name)
        with open(tmp / "meta.json", "w") as f:
            json.dump({"step": step, "saved_unix": time.time(), "world_size": state.get("world_size")}, f)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        _write_latest(ckpt_root, final.name)
        prune_checkpoints(ckpt_root, keep)
    du.barrier()
    return final


def _write_latest(ckpt_root: Path, name: str) -> None:
    tmp = ckpt_root / "LATEST.tmp"
    tmp.write_text(name + "\n")
    os.replace(tmp, ckpt_root / "LATEST")


def list_checkpoints(ckpt_root: Path) -> list[Path]:
    root = Path(ckpt_root)
    if not root.exists():
        return []
    dirs = [p for p in root.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")]
    return sorted((p for p in dirs if (p / "meta.json").exists()), key=lambda p: p.name)


def latest_checkpoint(ckpt_root: Path) -> Path | None:
    root = Path(ckpt_root)
    latest = root / "LATEST"
    if latest.exists():
        p = root / latest.read_text().strip()
        if (p / "meta.json").exists():
            return p
    found = list_checkpoints(root)
    return found[-1] if found else None


def prune_checkpoints(ckpt_root: Path, keep: int) -> None:
    """Delete all but the newest `keep` checkpoints (never one being uploaded)."""
    found = list_checkpoints(ckpt_root)
    for p in found[: max(len(found) - keep, 0)]:
        if not (p / UPLOAD_MARKER).exists():
            shutil.rmtree(p, ignore_errors=True)


def load_checkpoint(path: Path, *, model, optimizer, scaler, rank: int, world_size: int, map_location) -> dict:
    """Load a checkpoint into existing objects. Returns the saved training state."""
    path = Path(path)
    model.load_state_dict(torch.load(path / "model.pt", map_location=map_location, weights_only=True))
    optimizer.load_state_dict(torch.load(path / "optim.pt", map_location=map_location, weights_only=True))
    # Our own files: they hold Python and numpy RNG states, so not weights-only.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    scaler.load_state_dict(state["scaler"])
    rng_file = path / f"rng_rank{rank}.pt"
    state["rng"] = torch.load(rng_file, weights_only=False) if rng_file.exists() else None
    state["rng_exact"] = state.get("world_size") == world_size and rng_file.exists()
    return state


def save_final_weights(run_dir: Path, model: torch.nn.Module, meta: dict) -> Path:
    """Weights only (no optimizer state), for evaluation and sharing."""
    out = Path(run_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    cfg = Path(run_dir) / "config.yaml"
    if cfg.exists():
        shutil.copy2(cfg, out / "config.yaml")
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out
