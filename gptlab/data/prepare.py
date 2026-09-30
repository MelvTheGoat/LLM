"""Build the token dataset: download, clean, de-duplicate, train the tokenizer,
tokenize, and write train/validation shards.

Run:
    python -m gptlab.data.prepare --config configs/data/fineweb_edu_16k.yaml --out data/fineweb-edu-16k

Two passes over the same (seeded, shuffled) stream of documents:

1. Tokenizer pass. Clean and de-duplicate documents until we have
   `tokenizer.train_chars` characters of *training-split* text. Train a few BPE
   tokenizers of different sizes on part of it and measure how well each
   compresses held-out validation text. Then train the final tokenizer.
2. Token pass. Start again from the beginning. Clean, de-duplicate, split by
   hash, tokenize and write shards until we have `train_tokens` training tokens.

The validation split is decided by a hash of each document, and the tokenizer is
trained only on training documents. Validation text is never used for training
anything. Everything is logged to `manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gptlab.config import apply_overrides, from_dict, read_yaml, to_dict
from gptlab.data.clean import CleanConfig, CleaningStats, Deduper, clean_document, doc_hash, is_val
from gptlab.data.shards import ShardWriter
from gptlab.data.sources import DocSource, SourceConfig
from gptlab.data.tokenizer import Tokenizer, bytes_per_token, train_tokenizer


@dataclass
class TokenizerPrepConfig:
    vocab_size: int = 16384
    train_chars: int = 1_000_000_000  # characters of training text for the final tokenizer
    min_frequency: int = 2
    compare_vocab_sizes: list[int] = field(default_factory=lambda: [8192, 16384, 32768])
    compare_train_chars: int = 200_000_000  # smaller sample for the size comparison
    compare_eval_docs: int = 2000  # validation documents used to measure compression


@dataclass
class DataPrepConfig:
    name: str
    seed: int = 1234
    source: SourceConfig = field(default_factory=SourceConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    tokenizer: TokenizerPrepConfig = field(default_factory=TokenizerPrepConfig)
    train_tokens: int = 2_600_000_000  # stop once this many training tokens are written
    val_fraction: float = 0.005  # share of documents that go to validation
    max_val_tokens: int = 20_000_000  # cap on validation tokens
    shard_tokens: int = 100_000_000  # tokens per shard file (200 MB)
    workers: int = 0  # worker processes; 0 means one per CPU core
    batch_docs: int = 256  # documents per work item
    batch_timeout_seconds: float = 900.0  # a worker slower than this on one batch is treated as stuck


def load_data_config(path: str | Path, overrides: list[str] | None = None) -> DataPrepConfig:
    cfg = from_dict(DataPrepConfig, apply_overrides(read_yaml(path), overrides or []))
    cfg.source.validate()
    return cfg


# ---------------------------------------------------------------------------
# Work done in worker processes
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(clean_cfg: CleanConfig, tokenizer_path: str | None) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _W["clean"] = clean_cfg
    _W["tok"] = Tokenizer.load(tokenizer_path) if tokenizer_path else None


def _process_batch(texts: list[str]) -> list[tuple]:
    """Clean, hash and (in the token pass) tokenize a batch of documents.

    Returns one tuple per document:
    (digest, reason, raw_chars, changed, clean_text_or_None, ids_or_None, n_bytes)
    Clean text is only returned when it is needed: for rejected documents (for
    examples in the log) and in the tokenizer pass.
    """
    cfg = _W["clean"]
    tok = _W["tok"]
    kept_idx, kept_text, out = [], [], []
    for raw in texts:
        text, reason = clean_document(raw, cfg)
        digest = doc_hash(text)
        changed = text != raw
        if reason is None:
            kept_idx.append(len(out))
            kept_text.append(text)
            out.append([digest, None, len(raw), changed, None if tok else text, None, len(text.encode("utf-8"))])
        else:
            out.append([digest, reason, len(raw), changed, text, None, 0])
    if tok is not None and kept_text:
        for i, ids in zip(kept_idx, tok.encode_batch(kept_text)):
            out[i][5] = np.asarray(ids, dtype=np.uint16)
    return [tuple(o) for o in out]


def _batches(docs, size: int):
    batch = []
    for d in docs:
        batch.append(d)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _processed(source: DocSource, cfg: DataPrepConfig, tokenizer_path: str | None):
    """Yield processed documents in stream order, using worker processes if asked.

    Workers are started with "spawn" (a fresh Python process), not "fork". A
    forked process copies only the thread that called fork. If another thread
    held a lock at that moment (for example the Hugging Face download engine,
    which keeps background threads alive), the copy can hang forever. That
    happened in the first Kaggle smoke run. A worker that dies or gets stuck now
    raises an error instead of hanging.
    """
    batches = _batches(source.iter_documents(cfg.seed), cfg.batch_docs)
    workers = cfg.workers or os.cpu_count() or 1
    if workers <= 1:
        _init_worker(cfg.clean, tokenizer_path)
        for b in batches:
            yield from _process_batch(b)
        return
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(cfg.clean, tokenizer_path))
    pending: deque = deque()
    finished = False

    def result(future):
        try:
            return future.result(timeout=cfg.batch_timeout_seconds)
        except FuturesTimeout:
            raise RuntimeError(f"a data worker spent over {cfg.batch_timeout_seconds:.0f}s on one batch; it looks stuck")
        except BrokenProcessPool as e:
            raise RuntimeError(f"a data worker process died: {e}") from e

    try:
        for b in batches:
            pending.append(pool.submit(_process_batch, b))
            if len(pending) >= 2 * workers:  # keep a few batches in flight, in order
                yield from result(pending.popleft())
        while pending:
            yield from result(pending.popleft())
        finished = True
    finally:
        if finished:
            pool.shutdown(wait=True)
        else:  # the caller stopped early, or something failed: do not wait for workers
            procs = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for proc in procs:
                proc.terminate()


# ---------------------------------------------------------------------------
# The two passes
# ---------------------------------------------------------------------------


def _tokenizer_pass(source, cfg, log):
    """Collect cleaned training documents for the tokenizer, plus some validation docs."""
    tcfg = cfg.tokenizer
    train_docs, val_docs = [], []
    chars = 0
    dedup = Deduper()
    stats = CleaningStats()
    last = time.time()
    for digest, reason, raw_chars, changed, text, _, _ in _processed(source, cfg, None):
        if time.time() - last > 60:
            last = time.time()
            log(f"tokenizer sample: {chars:,} / {tcfg.train_chars:,} chars ({stats.docs_in:,} docs read)")
        stats.add_input(raw_chars, changed)
        if reason is not None:
            stats.add_removed(reason, text)
            continue
        if dedup.is_duplicate(digest):
            stats.add_removed("exact_duplicate", text)
            continue
        stats.add_kept(len(text))
        if is_val(digest, cfg.val_fraction):
            if len(val_docs) < tcfg.compare_eval_docs:
                val_docs.append(text)
            continue
        train_docs.append(text)
        chars += len(text)
        if chars >= tcfg.train_chars:
            break
    log(f"tokenizer sample: {len(train_docs)} train docs, {chars:,} chars; {len(val_docs)} val docs")
    return train_docs, val_docs, stats


def _take_chars(docs: list[str], limit: int) -> list[str]:
    out, n = [], 0
    for d in docs:
        if n >= limit:
            break
        out.append(d)
        n += len(d)
    return out


def _train_tokenizers(train_docs, val_docs, cfg, out_dir: Path, log):
    tcfg = cfg.tokenizer
    comparison = []
    small_sample = _take_chars(train_docs, tcfg.compare_train_chars)
    for vs in tcfg.compare_vocab_sizes:
        _set_phase(f"training a {vs}-token tokenizer for the size comparison", log)
        t0 = time.time()
        tok = train_tokenizer(small_sample, vs, tcfg.min_frequency)
        row = {
            "vocab_size": vs,
            "actual_vocab_size": tok.vocab_size,
            "train_chars": sum(len(d) for d in small_sample),
            "val_docs": len(val_docs),
            "bytes_per_token_val": bytes_per_token(tok, val_docs) if val_docs else None,
            "train_seconds": round(time.time() - t0, 1),
        }
        comparison.append(row)
        log(f"vocab comparison: {row}")
    _set_phase(f"training the final {tcfg.vocab_size}-token tokenizer on {sum(len(d) for d in train_docs):,} chars", log)
    t0 = time.time()
    tok = train_tokenizer(train_docs, tcfg.vocab_size, tcfg.min_frequency)
    if tok.vocab_size != tcfg.vocab_size:
        log(f"warning: tokenizer has {tok.vocab_size} tokens, asked for {tcfg.vocab_size}")
    path = out_dir / "tokenizer.json"
    tok.save(path)
    info = {
        "vocab_size": tok.vocab_size,
        "eot_id": tok.eot_id,
        "train_docs": len(train_docs),
        "train_chars": sum(len(d) for d in train_docs),
        "train_seconds": round(time.time() - t0, 1),
        "bytes_per_token_val": bytes_per_token(tok, val_docs) if val_docs else None,
    }
    log(f"final tokenizer: {info}")
    return path, tok, comparison, info


def _token_pass(source, cfg, tokenizer_path, tok, out_dir: Path, log):
    train = ShardWriter(out_dir, "train", cfg.shard_tokens, tok.vocab_size, tok.eot_id)
    val = ShardWriter(out_dir, "val", cfg.shard_tokens, tok.vocab_size, tok.eot_id)
    split_docs = {"train": 0, "val": 0}
    split_bytes = {"train": 0, "val": 0}
    val_full = 0
    dedup = Deduper()
    stats = CleaningStats()
    eot = np.array([tok.eot_id], dtype=np.uint16)
    t0 = last = time.time()
    for digest, reason, raw_chars, changed, text, ids, n_bytes in _processed(source, cfg, str(tokenizer_path)):
        stats.add_input(raw_chars, changed)
        if reason is not None:
            stats.add_removed(reason, text)
            continue
        if dedup.is_duplicate(digest):
            stats.add_removed("exact_duplicate", f"[{len(ids)} tokens]")
            continue
        stats.add_kept(n_bytes)
        split = "val" if is_val(digest, cfg.val_fraction) else "train"
        writer = val if split == "val" else train
        if split == "val" and val.total_tokens >= cfg.max_val_tokens:
            val_full += 1
            continue
        writer.add(eot)  # every document starts with the end-of-text token
        writer.add(ids)
        split_docs[split] += 1
        split_bytes[split] += n_bytes
        if time.time() - last > 60:
            last = time.time()
            rate = train.total_tokens / (last - t0)
            log(f"train tokens {train.total_tokens:,} / {cfg.train_tokens:,} ({rate:,.0f} tok/s), val {val.total_tokens:,}")
        if train.total_tokens >= cfg.train_tokens:
            break
    train_paths, val_paths = train.close(), val.close()
    if train.total_tokens < cfg.train_tokens:
        log(f"warning: source ran out: {train.total_tokens:,} train tokens < target {cfg.train_tokens:,}")
    splits = {}
    for name, writer, paths in [("train", train, train_paths), ("val", val, val_paths)]:
        splits[name] = {
            "docs": split_docs[name],
            "tokens": writer.total_tokens,
            "text_bytes": split_bytes[name],
            "bytes_per_token": split_bytes[name] / max(writer.total_tokens, 1),
            "shards": [p.name for p in paths],
        }
    return splits, stats, val_full, round(time.time() - t0, 1)


def prepare(cfg: DataPrepConfig, out_dir: str | Path, work_dir: str | Path, log=print) -> dict:
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.bin")):
        raise FileExistsError(f"{out_dir} already has shards; use an empty folder")
    t_start = time.time()
    _set_phase("downloading the source files", log)
    source = DocSource(cfg.source, work_dir, log)
    source.prepare()
    log(f"source: {len(source.file_names)} files")

    _set_phase("reading and cleaning documents for the tokenizer sample", log)
    train_docs, val_docs, sample_stats = _tokenizer_pass(source, cfg, log)
    tok_path, tok, comparison, tok_info = _train_tokenizers(train_docs, val_docs, cfg, out_dir, log)
    del train_docs, val_docs

    _set_phase("tokenizing documents into shards", log)
    splits, stats, val_full, token_seconds = _token_pass(source, cfg, tok_path, tok, out_dir, log)
    _set_phase("done", log)
    manifest = {
        "name": cfg.name,
        "created_unix": int(time.time()),
        "config": to_dict(cfg),
        "source_files": source.file_names,
        "tokenizer": tok_info,
        "vocab_comparison": comparison,
        "splits": splits,
        "cleaning": stats.to_dict(),
        "val_docs_dropped_because_val_was_full": val_full,
        "tokenizer_sample_cleaning": sample_stats.to_dict(),
        "seconds": {"token_pass": token_seconds, "total": round(time.time() - t_start, 1)},
        "versions": _versions(),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"done: {splits['train']['tokens']:,} train tokens, {splits['val']['tokens']:,} val tokens")
    return manifest


_PHASE = {"name": "starting", "since": time.time()}


def _set_phase(name: str, log) -> None:
    _PHASE.update(name=name, since=time.time())
    log(name)


def _heartbeat(log, every_seconds: float = 300.0) -> None:
    """Print a line every few minutes, so a slow step is never mistaken for a stuck one."""
    start = time.time()
    while True:
        time.sleep(every_seconds)
        log(f"still working: {_PHASE['name']} ({(time.time() - _PHASE['since']) / 60:.0f} min in this step, "
            f"{(time.time() - start) / 60:.0f} min total)")


def _versions() -> dict:
    import tokenizers

    return {"python": platform.python_version(), "numpy": np.__version__, "tokenizers": tokenizers.__version__}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="output folder for shards, tokenizer and manifest")
    p.add_argument("--work", default=None, help="scratch folder for downloads (default: <out>/../work)")
    p.add_argument("--set", action="append", default=[], help="override a config value, e.g. train_tokens=1e6")
    args = p.parse_args(argv)
    cfg = load_data_config(args.config, args.set)
    work = args.work or str(Path(args.out).parent / "work")

    def log(m):
        print(f"[data] {m}", flush=True)

    threading.Thread(target=_heartbeat, args=(log,), daemon=True).start()
    prepare(cfg, args.out, work, log=log)


if __name__ == "__main__":
    main()
