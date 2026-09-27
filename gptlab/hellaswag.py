"""Zero-shot HellaSwag evaluation, written from scratch.

HellaSwag (Zellers et al., 2019) gives a short context and four possible
endings; one is right. "Zero-shot" means the model gets no examples and no
fine-tuning: we just ask which ending it finds most likely. For each ending we
sum the log-probabilities of its tokens given the context and pick the highest.

- acc: highest total log-probability
- acc_norm: highest log-probability per byte of the ending. Long endings get
  lower totals just by having more tokens; this corrects for that.

Random guessing scores 25%. We follow the prompt format of EleutherAI's
lm-evaluation-harness ("<activity label>: <context>", then " <ending>", with the
same text cleanup), so our numbers are comparable to numbers made with it.
We use the 10,042-example validation set (the test labels are not public).
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import torch

from gptlab import distributed as du

VAL_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


def preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_examples(path: str | Path, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def ensure_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(VAL_URL, tmp)
        tmp.replace(path)
    return path


def render(example: dict, tok) -> tuple[list[int], list[list[int]], list[int], int]:
    """Token ids for the context and each ending, ending byte lengths, and the label."""
    ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
    query = preprocess(example["activity_label"] + ": " + ctx)
    endings = [" " + preprocess(e) for e in example["endings"]]
    return (
        tok.encode(query),
        [tok.encode(e) for e in endings],
        [len(e.encode("utf-8")) for e in endings],
        int(example["label"]),
    )


@torch.no_grad()
def ending_logprobs(model, rows: list[tuple[list[int], list[int]]], device, autocast_ctx) -> list[float]:
    """Sum of log p(ending tokens | context) for each (context, ending) pair."""
    seq_len = model.cfg.seq_len
    seqs, spans = [], []
    for ctx, end in rows:
        full = (ctx + end)[-(seq_len + 1) :]  # keep the end; cut the context from the left
        n_end = min(len(end), len(full) - 1)
        seqs.append(full)
        spans.append((len(full) - 1 - n_end, len(full) - 1))  # target positions of the ending
    width = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(seqs), width, dtype=torch.long)
    y = torch.zeros(len(seqs), width, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.bool)
    for i, (s, (a, b)) in enumerate(zip(seqs, spans)):
        t = torch.tensor(s)
        x[i, : len(s) - 1] = t[:-1]
        y[i, : len(s) - 1] = t[1:]
        mask[i, a:b] = True
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    with autocast_ctx:
        logits, _, _ = model(x)
    logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=1).tolist()


@torch.no_grad()
def evaluate_hellaswag(model, tok, examples: list[dict], device, autocast_ctx, rank=0, world_size=1, batch_examples=8) -> dict:
    was_training = model.training
    model.eval()
    counts = torch.zeros(3, dtype=torch.float64)  # n, correct, correct_norm
    mine = examples[rank::world_size]
    for i in range(0, len(mine), batch_examples):
        batch = [render(ex, tok) for ex in mine[i : i + batch_examples]]
        rows = [(ctx, end) for ctx, ends, _, _ in batch for end in ends]
        scores = ending_logprobs(model, rows, device, autocast_ctx)
        for j, (_, ends, nbytes, label) in enumerate(batch):
            s = scores[4 * j : 4 * j + 4]
            pred = max(range(4), key=lambda k: s[k])
            pred_norm = max(range(4), key=lambda k: s[k] / nbytes[k])
            counts += torch.tensor([1.0, float(pred == label), float(pred_norm == label)], dtype=torch.float64)
    counts = du.all_reduce_sum(counts.to(device)).cpu()
    model.train(was_training)
    n = int(counts[0].item())
    return {
        "hellaswag_n": n,
        "hellaswag_acc": counts[1].item() / max(n, 1),
        "hellaswag_acc_norm": counts[2].item() / max(n, 1),
        "random_chance": 0.25,
    }
