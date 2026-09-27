"""Token shards: compact uint16 files of token ids.

Each shard is one file: a 1024-byte header followed by the tokens as uint16
(2 bytes per token, enough for a vocabulary up to 65,536). The header holds a
magic number, a format version and the token count, so a truncated or wrong file
is caught when it is opened instead of producing garbage batches.

Shards are read with numpy memory mapping, so opening one costs almost nothing
and only the parts we touch are read from disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MAGIC = 0x67707431  # the bytes "gpt1"
VERSION = 1
HEADER_INTS = 128  # int64 values -> 1024 bytes
HEADER_BYTES = HEADER_INTS * 8
TOKEN_DTYPE = np.uint16


class ShardError(ValueError):
    pass


def write_shard(path: str | Path, tokens: np.ndarray, vocab_size: int, eot_id: int) -> None:
    """Write one shard. The file appears only when it is complete."""
    tokens = np.asarray(tokens)
    if tokens.size and int(tokens.max()) >= vocab_size:
        raise ShardError("token id out of range for the vocabulary")
    if vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ShardError("vocab too big for uint16 shards")
    header = np.zeros(HEADER_INTS, dtype=np.int64)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = tokens.size
    header[3] = vocab_size
    header[4] = eot_id
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(TOKEN_DTYPE, copy=False).tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_header(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ShardError(f"{path}: file is shorter than the header")
    header = np.frombuffer(raw, dtype=np.int64)
    if header[0] != MAGIC:
        raise ShardError(f"{path}: not a token shard (bad magic number)")
    if header[1] != VERSION:
        raise ShardError(f"{path}: unsupported shard version {header[1]}")
    n_tokens = int(header[2])
    expected = HEADER_BYTES + n_tokens * np.dtype(TOKEN_DTYPE).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ShardError(f"{path}: size is {actual} bytes, header says {expected} (truncated?)")
    return {"n_tokens": n_tokens, "vocab_size": int(header[3]), "eot_id": int(header[4])}


def load_shard(path: str | Path) -> np.ndarray:
    """Memory-map a shard and return its tokens as a read-only uint16 array."""
    info = read_header(path)
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r", offset=HEADER_BYTES, shape=(info["n_tokens"],))


class ShardWriter:
    """Collects token ids and writes them out as fixed-size shards.

    Files are named `<prefix>_000000.bin`, `<prefix>_000001.bin`, and so on. The
    last shard may be smaller than `shard_tokens`.
    """

    def __init__(self, out_dir: str | Path, prefix: str, shard_tokens: int, vocab_size: int, eot_id: int):
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.vocab_size = vocab_size
        self.eot_id = eot_id
        self._buf = np.empty(shard_tokens, dtype=TOKEN_DTYPE)
        self._fill = 0
        self.paths: list[Path] = []
        self.total_tokens = 0

    def add(self, ids) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ShardError("token id out of range for the vocabulary")
        pos = 0
        while pos < ids.size:
            take = min(ids.size - pos, self.shard_tokens - self._fill)
            self._buf[self._fill : self._fill + take] = ids[pos : pos + take]
            self._fill += take
            pos += take
            if self._fill == self.shard_tokens:
                self._flush()
        self.total_tokens += ids.size

    def _flush(self) -> None:
        if self._fill == 0:
            return
        path = self.out_dir / f"{self.prefix}_{len(self.paths):06d}.bin"
        write_shard(path, self._buf[: self._fill], self.vocab_size, self.eot_id)
        self.paths.append(path)
        self._fill = 0

    def close(self) -> list[Path]:
        self._flush()
        return self.paths


def list_shards(data_dir: str | Path, split: str) -> list[Path]:
    """All shards of one split ("train" or "val"), in name order."""
    return sorted(Path(data_dir).glob(f"{split}_*.bin"))
