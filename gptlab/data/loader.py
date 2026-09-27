"""Batches for training and validation.

Training data is cut into blocks of `seq_len + 1` tokens (input plus the
next-token targets). Every epoch uses one seeded random order (a permutation)
over all blocks from all training shards. So each batch is a random sample of
the whole dataset, and no block repeats before the epoch ends.

The data position is just a counter: "global sequence number g". Sequence g is
block perm[g]. At optimizer step s, the run reads sequences
s*S ... s*S + S - 1, where S is the number of sequences per step. Which GPU
reads which of those depends only on its rank. This has two nice effects:

- Exact resume needs nothing more than the step number.
- The same step reads the same sequences on 1 GPU or 2 GPUs (as long as the
  global batch size is the same), so runs can move between machines.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from gptlab.data.shards import list_shards, load_shard


def resolve_data_dir(name: str, root: str | None = None) -> Path:
    """Local folder of a dataset: <root>/<name>, root = config, $GPTLAB_DATA_ROOT or ./data."""
    root = root or os.environ.get("GPTLAB_DATA_ROOT", "data")
    return Path(root) / name


class _Blocks:
    """Blocks of `seq_len + 1` tokens across several shards, with stride seq_len."""

    def __init__(self, paths: list[Path], seq_len: int):
        if not paths:
            raise FileNotFoundError("no shards given")
        self.paths = [Path(p) for p in paths]
        self.seq_len = seq_len
        self.shards = [load_shard(p) for p in self.paths]
        self.blocks_per_shard = np.array([(len(s) - 1) // seq_len for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.blocks_per_shard)])
        self.total_blocks = int(self.offsets[-1])
        if self.total_blocks == 0:
            raise ValueError("shards are too small for one sequence")

    def gather(self, block_ids: np.ndarray) -> np.ndarray:
        """Stack the given blocks into an int64 array of shape (n, seq_len + 1)."""
        T = self.seq_len
        shard_ids = np.searchsorted(self.offsets, block_ids, side="right") - 1
        out = np.empty((len(block_ids), T + 1), dtype=np.int64)
        for row, (b, s) in enumerate(zip(block_ids, shard_ids)):
            start = int(b - self.offsets[s]) * T
            out[row] = self.shards[s][start : start + T + 1]
        return out

    def fingerprint(self) -> str:
        text = "|".join(f"{p.name}:{len(s)}" for p, s in zip(self.paths, self.shards))
        return hashlib.sha256(f"{text}|T={self.seq_len}".encode()).hexdigest()[:16]


def _to_xy(arr: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(arr)
    x, y = t[:, :-1], t[:, 1:]
    if device is not None and torch.device(device).type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    elif device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


class TrainStream:
    def __init__(self, paths: list[Path], seq_len: int, seed: int):
        bad = [p for p in paths if Path(p).name.startswith("val")]
        if bad:
            raise ValueError(f"validation shards must never be used for training: {bad}")
        self.blocks = _Blocks(paths, seq_len)
        self.seed = seed
        self._epoch = -1
        self._perm = None

    @classmethod
    def from_dir(cls, data_dir: str | Path, seq_len: int, seed: int, max_shards: int | None = None):
        paths = list_shards(data_dir, "train")
        if max_shards:
            paths = paths[:max_shards]
        return cls(paths, seq_len, seed)

    @property
    def total_blocks(self) -> int:
        return self.blocks.total_blocks

    @property
    def total_tokens(self) -> int:
        return self.total_blocks * self.blocks.seq_len

    def fingerprint(self) -> str:
        return f"{self.blocks.fingerprint()}-seed{self.seed}"

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = rng.permutation(self.total_blocks)
            self._epoch = epoch
        return self._perm

    def block_ids(self, first: int, count: int) -> np.ndarray:
        """Block ids of global sequences first ... first + count - 1."""
        out = np.empty(count, dtype=np.int64)
        for i in range(count):  # small loop; handles the epoch boundary simply
            epoch, pos = divmod(first + i, self.total_blocks)
            out[i] = self._permutation(epoch)[pos]
        return out

    def sequences(self, first: int, count: int) -> np.ndarray:
        return self.blocks.gather(self.block_ids(first, count))

    def batch(self, step: int, micro_step: int, rank: int, world_size: int, micro_batch: int, accum: int, device=None):
        """The (x, y) micro-batch for one GPU at one step. y is x shifted by one token."""
        seqs_per_step = world_size * micro_batch * accum
        first = step * seqs_per_step + (micro_step * world_size + rank) * micro_batch
        return _to_xy(self.sequences(first, micro_batch), device)

    def epoch_of_step(self, step: int, seqs_per_step: int) -> float:
        return step * seqs_per_step / self.total_blocks


class ValData:
    """A fixed, ordered set of validation sequences (no shuffling, no overlap)."""

    def __init__(self, paths: list[Path], seq_len: int):
        bad = [p for p in paths if not Path(p).name.startswith("val")]
        if bad:
            raise ValueError(f"expected validation shards, got {bad}")
        self.blocks = _Blocks(paths, seq_len)

    @classmethod
    def from_dir(cls, data_dir: str | Path, seq_len: int):
        return cls(list_shards(data_dir, "val"), seq_len)

    def num_sequences(self, n_tokens: int) -> int:
        return max(1, min(n_tokens // self.blocks.seq_len, self.blocks.total_blocks))

    def batches(self, n_tokens: int, micro_batch: int, rank: int = 0, world_size: int = 1, device=None):
        """Yield this rank's share of the first `n_tokens` validation tokens."""
        ids = np.arange(self.num_sequences(n_tokens))[rank::world_size]
        for i in range(0, len(ids), micro_batch):
            yield _to_xy(self.blocks.gather(ids[i : i + micro_batch]), device)
