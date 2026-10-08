import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gptlab.config import ModelConfig
from gptlab.data.loader import ValData
from gptlab.data.shards import write_shard
from gptlab.evaluate import evaluate_val
from gptlab.generate import generate
from gptlab.hellaswag import ending_logprobs, evaluate_hellaswag, load_examples, preprocess, render
from gptlab.model import GPT

HS = Path(__file__).parent / "data" / "hellaswag_tiny.jsonl"


def tiny_model(vocab):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, seq_len=32, n_layer=2, n_head=2, d_model=32)).eval()


def test_preprocess_matches_harness_rules():
    # Same steps as lm-evaluation-harness: " [title]" -> ". ", drop [..], one pass of "  " -> " ".
    assert preprocess(" A [title] B  [step] c ") == "A. B  c"
    assert preprocess("Put the pan [title] on the stove.") == "Put the pan. on the stove."


def test_ending_logprobs_match_a_direct_computation(small_tokenizer):
    model = tiny_model(small_tokenizer.vocab_size)
    rows = [([5, 6, 7], [8, 9]), ([10], [11, 12, 13, 14]), ([1, 2, 3, 4, 5, 6], [7])]
    got = ending_logprobs(model, rows, "cpu", nullcontext())
    for (ctx, end), g in zip(rows, got):
        seq = torch.tensor([ctx + end])
        with torch.no_grad():
            logits, _, _ = model(seq[:, :-1])
        logp = torch.log_softmax(logits[0], -1)
        want = sum(logp[len(ctx) - 1 + i, end[i]].item() for i in range(len(end)))
        assert math.isclose(g, want, rel_tol=1e-4, abs_tol=1e-4)


class FavouriteTokenModel(torch.nn.Module):
    """A fake model that strongly prefers a fixed set of tokens."""

    def __init__(self, vocab, favourite_ids):
        super().__init__()
        self.cfg = ModelConfig(vocab_size=vocab, seq_len=64, n_layer=1, n_head=1, d_model=8)
        self.bias = torch.full((vocab,), -5.0)
        self.bias[list(favourite_ids)] = 5.0

    def forward(self, idx):
        return self.bias.expand(*idx.shape, -1).clone(), None, {}


def test_hellaswag_picks_the_ending_the_model_likes(small_tokenizer):
    examples = load_examples(HS)
    # Make the model love exactly the tokens of each correct ending.
    fav = set()
    for ex in examples:
        _, ends, _, label = render(ex, small_tokenizer)
        fav |= set(ends[label])
    model = FavouriteTokenModel(small_tokenizer.vocab_size, fav)
    result = evaluate_hellaswag(model, small_tokenizer, examples, "cpu", nullcontext())
    assert result["hellaswag_n"] == 3
    assert result["hellaswag_acc_norm"] == 1.0
    assert 0.0 <= result["hellaswag_acc"] <= 1.0
    # A real (untrained) model runs end to end.
    real = evaluate_hellaswag(tiny_model(small_tokenizer.vocab_size), small_tokenizer, examples, "cpu", nullcontext())
    assert real["hellaswag_n"] == 3


def test_val_loss_perplexity_and_bits_per_byte(tmp_path):
    rng = np.random.default_rng(0)
    write_shard(tmp_path / "val_000000.bin", rng.integers(0, 50, 400), vocab_size=50, eot_id=0)
    val = ValData.from_dir(tmp_path, 32)
    model = tiny_model(50)
    token_bytes = np.full(50, 2)
    out = evaluate_val(model, val, 320, 3, "cpu", nullcontext(), token_bytes)
    xs = torch.cat([x for x, _ in val.batches(320, 3)])
    ys = torch.cat([y for _, y in val.batches(320, 3)])
    with torch.no_grad():
        logits, _, _ = model(xs)
    want = F.cross_entropy(logits.view(-1, 50), ys.view(-1)).item()
    assert math.isclose(out["val_loss"], want, rel_tol=1e-5)
    assert math.isclose(out["val_ppl"], math.exp(want), rel_tol=1e-5)
    assert math.isclose(out["val_bpb"], want / math.log(2) / 2, rel_tol=1e-5)
    assert out["val_tokens"] == 320


def test_generate_is_reproducible_with_a_seed():
    model = tiny_model(50)
    idx = torch.tensor([[1, 2, 3]])
    a = generate(model, idx, 40, generator=torch.Generator().manual_seed(1))
    b = generate(model, idx, 40, generator=torch.Generator().manual_seed(1))
    assert a.shape == (1, 43) and torch.equal(a, b)
    assert torch.equal(a[:, :3], idx) and int(a.max()) < 50


def test_parquet_download_format_converts_to_the_same_examples(tmp_path, small_tokenizer):
    # The Hugging Face copy stores labels as strings; render() must give the same result.
    import pyarrow as pa
    import pyarrow.parquet as pq

    from gptlab.hellaswag import parquet_to_jsonl

    examples = load_examples(HS)
    rows = [dict(ex, label=str(ex["label"])) for ex in examples]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "val.parquet")
    out = parquet_to_jsonl(tmp_path / "val.parquet", tmp_path / "val.jsonl", expect_rows=len(rows))
    back = load_examples(out)
    assert [render(ex, small_tokenizer) for ex in back] == [render(ex, small_tokenizer) for ex in examples]
    with pytest.raises(ValueError):
        parquet_to_jsonl(tmp_path / "val.parquet", tmp_path / "bad.jsonl", expect_rows=len(rows) + 1)
