import numpy as np
import pytest
import torch

from gptlab.data.loader import TrainStream, ValData
from gptlab.data.shards import write_shard

T = 8


@pytest.fixture
def shards(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i, n in enumerate([97, 200, 57]):
        p = tmp_path / f"train_{i:06d}.bin"
        write_shard(p, rng.integers(0, 500, size=n), vocab_size=500, eot_id=0)
        paths.append(p)
    v = tmp_path / "val_000000.bin"
    write_shard(v, np.arange(100) % 500, vocab_size=500, eot_id=0)
    return paths, v


def all_blocks(paths):
    from gptlab.data.shards import load_shard

    out = []
    for p in paths:
        s = load_shard(p).astype(np.int64)
        for b in range((len(s) - 1) // T):
            out.append(tuple(s[b * T : b * T + T + 1]))
    return out


def test_one_epoch_visits_every_block_once(shards):
    paths, _ = shards
    stream = TrainStream(paths, T, seed=1)
    n = stream.total_blocks
    assert n == (96 // T) + (199 // T) + (56 // T)
    seen = [tuple(r) for r in stream.sequences(0, n)]
    assert sorted(seen) == sorted(all_blocks(paths))
    # The next epoch is a different order of the same blocks.
    nxt = [tuple(r) for r in stream.sequences(n, n)]
    assert sorted(nxt) == sorted(seen) and nxt != seen


def test_order_depends_only_on_seed(shards):
    paths, _ = shards
    a = TrainStream(paths, T, seed=1).sequences(5, 10)
    b = TrainStream(paths, T, seed=1).sequences(5, 10)
    c = TrainStream(paths, T, seed=2).sequences(5, 10)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_targets_are_inputs_shifted_by_one(shards):
    paths, _ = shards
    x, y = TrainStream(paths, T, seed=0).batch(0, 0, 0, 1, micro_batch=4, accum=1)
    assert x.shape == y.shape == (4, T)
    assert x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_same_step_reads_same_data_for_any_gpu_split(shards):
    """World 2 x micro 2 x accum 1 == world 1 x micro 2 x accum 2 == world 1 x micro 4."""
    paths, _ = shards
    s = TrainStream(paths, T, seed=3)
    step = 2

    def collect(world, micro, accum):
        rows = []
        for m in range(accum):
            for r in range(world):
                x, _ = s.batch(step, m, r, world, micro, accum)
                rows.append(x)
        return torch.cat(rows)

    a = collect(2, 2, 1)
    assert torch.equal(a, collect(1, 2, 2))
    assert torch.equal(a, collect(1, 4, 1))
    # Different ranks read different sequences.
    x0, _ = s.batch(step, 0, 0, 2, 2, 1)
    x1, _ = s.batch(step, 0, 1, 2, 2, 1)
    assert not torch.equal(x0, x1)


def test_validation_shards_are_refused_for_training(shards):
    paths, val = shards
    with pytest.raises(ValueError, match="never be used for training"):
        TrainStream(paths + [val], T, seed=0)
    with pytest.raises(ValueError):
        ValData(paths, T)


def test_validation_batches_are_fixed_and_split_across_ranks(shards):
    _, val = shards
    v = ValData([val], T)
    n_tokens = 64
    single = torch.cat([x for x, _ in v.batches(n_tokens, micro_batch=3)])
    assert single.shape == (n_tokens // T, T)
    assert torch.equal(single[0], torch.arange(T))  # first sequence = first tokens
    r0 = torch.cat([x for x, _ in v.batches(n_tokens, 3, rank=0, world_size=2)])
    r1 = torch.cat([x for x, _ in v.batches(n_tokens, 3, rank=1, world_size=2)])
    assert sorted(map(tuple, torch.cat([r0, r1]).tolist())) == sorted(map(tuple, single.tolist()))
