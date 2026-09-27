import numpy as np
import pytest

from gptlab.data.shards import (
    HEADER_BYTES,
    ShardError,
    ShardWriter,
    list_shards,
    load_shard,
    read_header,
    write_shard,
)


def test_write_and_load_one_shard(tmp_path):
    tokens = np.random.default_rng(0).integers(0, 16384, size=10_000)
    path = tmp_path / "train_000000.bin"
    write_shard(path, tokens, vocab_size=16384, eot_id=0)
    assert read_header(path) == {"n_tokens": 10_000, "vocab_size": 16384, "eot_id": 0}
    loaded = load_shard(path)
    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, tokens)
    assert path.stat().st_size == HEADER_BYTES + 2 * 10_000


def test_writer_splits_into_fixed_size_shards(tmp_path):
    rng = np.random.default_rng(1)
    docs = [rng.integers(1, 1000, size=rng.integers(1, 300)) for _ in range(200)]
    writer = ShardWriter(tmp_path, "train", shard_tokens=1000, vocab_size=1000, eot_id=0)
    for d in docs:
        writer.add(np.concatenate([[0], d]))
    paths = writer.close()
    expected = np.concatenate([np.concatenate([[0], d]) for d in docs])
    assert writer.total_tokens == expected.size
    sizes = [read_header(p)["n_tokens"] for p in paths]
    assert all(s == 1000 for s in sizes[:-1]) and 0 < sizes[-1] <= 1000
    assert np.array_equal(np.concatenate([load_shard(p) for p in paths]), expected)
    assert list_shards(tmp_path, "train") == paths
    assert list_shards(tmp_path, "val") == []


def test_bad_files_are_rejected(tmp_path):
    path = tmp_path / "train_000000.bin"
    write_shard(path, np.arange(100), vocab_size=128, eot_id=0)
    # Truncated file.
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    with pytest.raises(ShardError, match="truncated"):
        read_header(path)
    # Not a shard at all.
    path.write_bytes(b"x" * 2000)
    with pytest.raises(ShardError, match="magic"):
        read_header(path)
    # Token ids that do not fit the vocabulary.
    with pytest.raises(ShardError):
        write_shard(tmp_path / "bad.bin", np.array([5, 300]), vocab_size=256, eot_id=0)
