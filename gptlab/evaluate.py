"""Evaluation: validation loss, perplexity, bits per byte, HellaSwag, samples.

- Validation loss: average cross-entropy (in nats) of next-token prediction on
  held-out text the model never trained on.
- Perplexity = exp(loss). Roughly "how many tokens the model is choosing
  between" on average. Lower is better.
- Bits per byte = total loss in bits / number of UTF-8 bytes predicted. Unlike
  loss per token, it does not depend on the tokenizer, so it can be compared
  with models that use other tokenizers.

Can also be run on its own for a finished run:
    python -m gptlab.evaluate --run-dir out/<run>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gptlab import distributed as du
from gptlab.config import RunConfig, load_config
from gptlab.data.loader import ValData, resolve_data_dir
from gptlab.data.tokenizer import Tokenizer
from gptlab.generate import generate
from gptlab.hellaswag import ensure_file, evaluate_hellaswag, load_examples


@torch.no_grad()
def evaluate_val(model, val_data: ValData, n_tokens: int, micro_batch: int, device, autocast_ctx,
                 token_bytes: np.ndarray | None = None, rank: int = 0, world_size: int = 1) -> dict:
    was_training = model.training
    model.eval()
    sums = torch.zeros(3, dtype=torch.float64, device=device)  # loss sum, tokens, bytes
    tb = torch.from_numpy(token_bytes).to(device) if token_bytes is not None else None
    for x, y in val_data.batches(n_tokens, micro_batch, rank, world_size, device):
        with autocast_ctx:
            logits, _, _ = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        sums[0] += loss.double()
        sums[1] += y.numel()
        if tb is not None:
            sums[2] += tb[y].sum().double()
    du.all_reduce_sum(sums)
    model.train(was_training)
    loss_sum, n_tok, n_bytes = sums.tolist()
    loss = loss_sum / max(n_tok, 1)
    out = {"val_loss": loss, "val_ppl": math.exp(min(loss, 50.0)), "val_tokens": int(n_tok)}
    if tb is not None and n_bytes > 0:
        out["val_bpb"] = loss_sum / math.log(2) / n_bytes
    return out


@torch.no_grad()
def sample_texts(model, tok: Tokenizer, prompts: list[str], n_tokens: int, device, autocast_ctx, seed: int = 0) -> list[dict]:
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for prompt in prompts:
        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        with autocast_ctx:
            result = generate(model, idx, n_tokens, temperature=0.8, top_k=50, generator=gen)
        out.append({"prompt": prompt, "completion": tok.decode(result[0, idx.shape[1] :].tolist())})
    model.train(was_training)
    return out


def default_hellaswag_path() -> Path:
    return Path(os.environ.get("GPTLAB_DATA_ROOT", "data")) / "hellaswag_val.jsonl"


def final_evaluation(model, cfg: RunConfig, data_dir: Path, device, autocast_ctx, rank: int = 0,
                     world_size: int = 1, hellaswag_path: str | Path | None = None, log=print) -> dict:
    """Full evaluation at the end of a run. Every process must call this."""
    tok = Tokenizer.load(Path(data_dir) / "tokenizer.json")
    val = ValData.from_dir(data_dir, cfg.model.seq_len)
    micro = cfg.train.micro_batch_size
    result = evaluate_val(model, val, cfg.eval.final_tokens, micro, device, autocast_ctx, tok.token_bytes(), rank, world_size)
    log(f"final val loss {result['val_loss']:.4f}  ppl {result['val_ppl']:.2f}  bpb {result.get('val_bpb', float('nan')):.4f}")
    if cfg.eval.hellaswag:
        try:
            path = Path(hellaswag_path) if hellaswag_path else default_hellaswag_path()
            if rank == 0:
                ensure_file(path)
            du.barrier()
            examples = load_examples(path, cfg.eval.hellaswag_limit)
            hs = evaluate_hellaswag(model, tok, examples, device, autocast_ctx, rank, world_size)
            result.update(hs)
            log(f"hellaswag acc {hs['hellaswag_acc']:.4f}  acc_norm {hs['hellaswag_acc_norm']:.4f}  (n={hs['hellaswag_n']}, chance 0.25)")
        except Exception as e:  # no internet, for example; do not lose the rest
            result["hellaswag_error"] = repr(e)
            log(f"hellaswag skipped: {e!r}")
    if rank == 0 and cfg.eval.sample_prompts:
        result["samples"] = sample_texts(model, tok, cfg.eval.sample_prompts, cfg.eval.sample_tokens, device, autocast_ctx, seed=cfg.seed)
    return result


def write_final_eval(run_dir: Path, result: dict) -> None:
    run_dir = Path(run_dir)
    with open(run_dir / "final_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if result.get("samples"):
        with open(run_dir / "samples.txt", "w", encoding="utf-8") as f:
            for s in result["samples"]:
                f.write(f"### {s['prompt']}\n{s['prompt']}{s['completion']}\n\n")


def main(argv=None) -> None:
    from gptlab.model import GPT

    p = argparse.ArgumentParser(description="Evaluate a finished run from its final weights.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--hellaswag-path", default=None)
    p.add_argument("--device", default=None, help="'cpu' to force CPU")
    p.add_argument("--set", action="append", default=[], help="override config values, e.g. eval.hellaswag_limit=100")
    args = p.parse_args(argv)
    info = du.setup(args.device)
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "final" / "config.yaml", args.set)
    model = GPT(cfg.model).to(info.device)
    model.load_state_dict(torch.load(run_dir / "final" / "model.pt", map_location=info.device, weights_only=True))
    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_dir(cfg.data.name, cfg.data.root)
    if cfg.train.precision == "fp32" or info.device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.autocast("cuda", dtype=torch.float16 if cfg.train.precision == "fp16" else torch.bfloat16)
    result = final_evaluation(model, cfg, data_dir, info.device, ctx, info.rank, info.world_size, args.hellaswag_path)
    if info.is_main:
        write_final_eval(run_dir, result)
    du.cleanup()


if __name__ == "__main__":
    main()
