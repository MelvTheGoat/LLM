"""Run logs (one JSON object per line) and per-layer debug statistics."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch


def _clean(v):
    """Make a value JSON-safe (NaN and inf become strings)."""
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


class JsonlWriter:
    """Appends records to a .jsonl file and flushes after each one."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._f = open(self.path, "a", encoding="utf-8") if enabled else None

    def write(self, record: dict) -> None:
        if self._f is not None:
            self._f.write(json.dumps(_clean(record)) + "\n")
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut off by a crash
    return out


def truncate_jsonl(path: str | Path, max_step: int) -> None:
    """Drop records with step > max_step (they will be redone after a resume)."""
    path = Path(path)
    if not path.exists():
        return
    keep = [r for r in read_jsonl(path) if r.get("step", 0) <= max_step]
    with open(path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")


def _group_name(param_name: str) -> str:
    parts = param_name.split(".")
    if parts[0] == "blocks":
        return f"layer{parts[1]}"
    return parts[0]  # wte, wpe, norm_f, lm_head


@torch.no_grad()
def grad_and_weight_norms(model: torch.nn.Module) -> dict:
    """L2 norms of gradients and weights, grouped by layer (call after unscaling)."""
    grads, weights = {}, {}
    for name, p in model.named_parameters():
        g = _group_name(name)
        weights[g] = weights.get(g, 0.0) + p.detach().float().pow(2).sum().item()
        if p.grad is not None:
            grads[g] = grads.get(g, 0.0) + p.grad.detach().float().pow(2).sum().item()
    return {
        "grad_norm": {k: math.sqrt(v) for k, v in grads.items()},
        "weight_norm": {k: math.sqrt(v) for k, v in weights.items()},
    }


@torch.no_grad()
def forward_stats(model, x, y, autocast_ctx=None) -> dict:
    """Run one forward pass with the explicit attention path and collect stats.

    Per layer: RMS ("activation size") of the attention output, MLP output and
    residual stream; mean attention entropy (low = attention focused on few
    tokens; near 0 = collapsed); lowest per-head entropy; and the largest
    attention logit (score before softmax). Plus the largest output logit and
    the mean log of the softmax normalizer (what z-loss pushes towards 0).
    """
    was_training = model.training
    model.eval()
    stats: list[dict] = []
    with autocast_ctx or nullcontext():
        _, loss, _ = model(x, y, stats=stats)
    model.train(was_training)
    layers = stats[:-1]
    out = {"loss": loss.item(), **stats[-1]}
    for key in layers[0]:
        out[key] = [s[key] for s in layers]
    return out
