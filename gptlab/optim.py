"""AdamW optimizer and the learning rate schedule."""

from __future__ import annotations

import math

import torch

from gptlab.config import TrainConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig, device_type: str) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Weight decay pulls weights towards zero. We apply it to weight matrices and
    embeddings (2D tensors) but not to norm gains or biases (1D tensors), which
    are few and where decay only hurts.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():  # tied weights appear once
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    if device_type == "cuda":
        kwargs["fused"] = True  # one kernel for the whole update: faster
    return torch.optim.AdamW(groups, **kwargs)


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Learning rate for optimizer step `step` (counting from 0).

    Linear warmup from near 0 to the peak over `warmup_steps`, then decay to
    `lr * min_lr_ratio` by the last step: cosine (smooth, the usual choice),
    linear, or none ("constant").
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return cfg.lr
    min_lr = cfg.lr * cfg.min_lr_ratio
    decay_steps = max(total_steps - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / decay_steps, 1.0)
    if cfg.schedule == "linear":
        return cfg.lr - (cfg.lr - min_lr) * progress
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))
