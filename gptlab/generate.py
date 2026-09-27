"""Text generation by sampling, one token at a time."""

from __future__ import annotations

import torch


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend `idx` (shape (B, T)) by `max_new_tokens` sampled tokens.

    temperature < 1 makes the output more predictable; top_k keeps only the k
    most likely next tokens before sampling. There is no key/value cache: every
    step reruns the whole (cropped) sequence. That is slow but simple, and fine
    for the few samples we make.
    """
    seq_len = model.cfg.seq_len
    for _ in range(max_new_tokens):
        logits, _, _ = model(idx[:, -seq_len:])
        logits = logits[:, -1, :].float() / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
