import json

import numpy as np
import pytest

from gptlab.data.clean import doc_hash, is_val, normalize
from gptlab.data.prepare import DataPrepConfig, TokenizerPrepConfig, prepare
from gptlab.data.shards import list_shards, load_shard
from gptlab.data.sources import SourceConfig
from gptlab.data.tokenizer import Tokenizer


def make_input(tmp_path, sample_docs):
    """Sample docs plus exact duplicates and junk that cleaning must remove."""
    docs = list(sample_docs)
    docs += [sample_docs[3], sample_docs[10].replace("\n", "  \r\n")]  # duplicates
    docs += ["tiny", "12 34 56 | 78 90 | " * 30, sample_docs[5] + "�" * 50]
    path = tmp_path / "docs.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    return path


def small_config(path, kind="jsonl") -> DataPrepConfig:
    return DataPrepConfig(
        name="test",
        source=SourceConfig(kind=kind, path=str(path), jsonl_group_size=8),
        tokenizer=TokenizerPrepConfig(
            vocab_size=400, train_chars=10**9, compare_vocab_sizes=[300, 400], compare_train_chars=50_000
        ),
        train_tokens=10**9,  # take everything
        val_fraction=0.15,
        shard_tokens=5000,
        workers=1,
        batch_docs=16,
    )


def read_docs(paths, tok):
    tokens = np.concatenate([load_shard(p) for p in paths]).astype(np.int64)
    assert tokens[0] == tok.eot_id
    starts = np.flatnonzero(tokens == tok.eot_id).tolist() + [len(tokens)]
    return [tok.decode(tokens[a + 1 : b]) for a, b in zip(starts[:-1], starts[1:])]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, sample_docs):
    tmp = tmp_path_factory.mktemp("prep")
    src = make_input(tmp, sample_docs)
    out = tmp / "out"
    manifest = prepare(small_config(src), out, tmp / "work", log=lambda m: None)
    return out, manifest, sample_docs


def test_manifest_counts_add_up(prepared):
    out, m, sample_docs = prepared
    c = m["cleaning"]
    assert c["docs_in"] == len(sample_docs) + 5
    assert c["removed_docs"]["exact_duplicate"] == 2
    assert c["removed_docs"]["too_short"] == 1
    assert c["removed_docs"]["few_letters"] == 1
    assert c["removed_docs"]["bad_characters"] == 1
    assert c["docs_kept"] == c["docs_in"] - sum(c["removed_docs"].values())
    assert m["splits"]["train"]["docs"] + m["splits"]["val"]["docs"] == c["docs_kept"]
    assert [r["vocab_size"] for r in m["vocab_comparison"]] == [300, 400]
    assert m["tokenizer"]["vocab_size"] == 400


def test_shards_decode_to_the_cleaned_documents(prepared):
    out, m, sample_docs = prepared
    tok = Tokenizer.load(out / "tokenizer.json")
    train = read_docs(list_shards(out, "train"), tok)
    val = read_docs(list_shards(out, "val"), tok)
    assert len(train) == m["splits"]["train"]["docs"]
    assert len(val) == m["splits"]["val"]["docs"] > 0
    expected = {normalize(d) for d in sample_docs}
    assert set(train) | set(val) == expected
    # No document is in both splits, and every document is in the split its hash says.
    assert not set(train) & set(val)
    assert all(is_val(doc_hash(d), 0.15) for d in val)
    assert not any(is_val(doc_hash(d), 0.15) for d in train)
    assert sum(m["splits"][s]["tokens"] for s in ["train", "val"]) == sum(
        len(load_shard(p)) for p in list_shards(out, "train") + list_shards(out, "val")
    )


def test_output_does_not_depend_on_worker_count(prepared, tmp_path, sample_docs):
    out, _, _ = prepared
    src = make_input(tmp_path, sample_docs)
    cfg = small_config(src)
    cfg.workers = 2
    out2 = tmp_path / "out2"
    prepare(cfg, out2, tmp_path / "work", log=lambda m: None)
    for split in ["train", "val"]:
        a, b = list_shards(out, split), list_shards(out2, split)
        assert [p.name for p in a] == [p.name for p in b]
        for pa, pb in zip(a, b):
            assert pa.read_bytes() == pb.read_bytes()


def test_parquet_source_and_token_limit(tmp_path, sample_docs):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"text": sample_docs, "id": list(range(len(sample_docs)))})
    pq.write_table(table, tmp_path / "part_000.parquet", row_group_size=10)
    cfg = small_config(tmp_path / "*.parquet", kind="parquet")
    cfg.train_tokens = 8000
    m = prepare(cfg, tmp_path / "out", tmp_path / "work", log=lambda m: None)
    # We stop at the first document that crosses the target.
    assert 8000 <= m["splits"]["train"]["tokens"] < 8000 + 2000
    assert m["cleaning"]["docs_in"] < len(sample_docs)


def test_refuses_to_overwrite_existing_shards(prepared, tmp_path, sample_docs):
    out, _, _ = prepared
    with pytest.raises(FileExistsError):
        prepare(small_config(make_input(tmp_path, sample_docs)), out, tmp_path / "w", log=lambda m: None)
