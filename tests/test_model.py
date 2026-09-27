import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from gptlab.config import ModelConfig
from gptlab.model import GPT, param_counts

V, T = 97, 16

VARIANTS = [
    dict(pos_encoding=p, norm=n, norm_placement=pl, mlp=m, tie_weights=t)
    for p, n, pl, m, t in itertools.product(
        ["rope", "learned"], ["rmsnorm", "layernorm"], ["pre", "post"], ["swiglu", "gelu"], [True, False]
    )
]


def variant_id(v):
    return "-".join(str(x) for x in v.values())


def tiny(**kw) -> GPT:
    base = dict(vocab_size=V, seq_len=T, n_layer=2, n_head=2, d_model=32)
    base.update(kw)
    torch.manual_seed(0)
    return GPT(ModelConfig(**base))


@pytest.mark.parametrize("variant", VARIANTS + [dict(qk_norm=True), dict(bias=True, dropout=0.1)], ids=variant_id)
def test_output_shapes(variant):
    model = tiny(**variant)
    idx = torch.randint(0, V, (3, T))
    tgt = torch.randint(0, V, (3, T))
    logits, loss, parts = model(idx, tgt)
    assert logits.shape == (3, T, V)
    assert loss.shape == () and torch.isfinite(loss)
    # At init the model is close to uniform, so loss is close to ln(V).
    assert abs(loss.item() - math.log(V)) < 0.5
    logits, loss, _ = model(idx[:, :5])  # shorter sequence, no targets
    assert logits.shape == (3, 5, V) and loss is None


@pytest.mark.parametrize("variant", VARIANTS + [dict(qk_norm=True)], ids=variant_id)
def test_future_tokens_do_not_change_past_outputs(variant):
    """Change tokens after position t: outputs at positions <= t must not move."""
    model = tiny(**variant).eval()
    torch.manual_seed(1)
    idx = torch.randint(0, V, (2, T))
    t = 6
    changed = idx.clone()
    changed[:, t + 1 :] = torch.randint(0, V, (2, T - t - 1))
    assert not torch.equal(changed, idx)
    with torch.no_grad():
        a, _, _ = model(idx)
        b, _, _ = model(changed)
        # Same check on the explicit attention path used for debug stats.
        a2, _, _ = model(idx, stats=[])
        b2, _, _ = model(changed, stats=[])
    assert torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-6)
    assert torch.allclose(a2[:, : t + 1], b2[:, : t + 1], atol=1e-6)
    # And the test is not vacuous: later positions do change.
    assert not torch.allclose(a[:, t + 1 :], b[:, t + 1 :], atol=1e-4)


def test_changing_a_past_token_changes_later_outputs():
    model = tiny().eval()
    idx = torch.randint(0, V, (1, T))
    changed = idx.clone()
    changed[0, 3] = (idx[0, 3] + 1) % V
    with torch.no_grad():
        a, _, _ = model(idx)
        b, _, _ = model(changed)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)
    assert not torch.allclose(a[:, 3:], b[:, 3:], atol=1e-4)


def test_debug_path_matches_fast_attention():
    model = tiny(qk_norm=True).eval()
    idx = torch.randint(0, V, (2, T))
    stats = []
    with torch.no_grad():
        fast, _, _ = model(idx)
        slow, _, _ = model(idx, stats=stats)
    assert torch.allclose(fast, slow, atol=1e-5)
    assert len(stats) == model.cfg.n_layer + 1  # one per layer, plus output logits
    for s in stats[:-1]:
        assert 0.0 <= s["attn_entropy_mean"] <= math.log(T)
        assert s["attn_entropy_min_head"] <= s["attn_entropy_mean"] + 1e-6
        assert math.isfinite(s["attn_logit_max"]) and s["resid_rms"] > 0
    assert stats[-1]["logits_abs_max"] > 0


def test_loss_matches_pytorch_cross_entropy_and_z_loss():
    model = tiny()
    idx = torch.randint(0, V, (2, T))
    tgt = torch.randint(0, V, (2, T))
    logits, loss, parts = model(idx, tgt, z_loss_weight=1e-2)
    ce = F.cross_entropy(logits.view(-1, V), tgt.view(-1))
    z = torch.logsumexp(logits, -1).pow(2).mean()
    assert torch.allclose(parts["ce"], ce, atol=1e-5)
    assert torch.allclose(parts["z"], z, atol=1e-5)
    assert torch.allclose(loss, ce + 1e-2 * z, atol=1e-5)


@pytest.mark.parametrize(
    "variant",
    [dict(), dict(pos_encoding="learned", norm="layernorm", norm_placement="post", mlp="gelu", tie_weights=False)],
    ids=["default", "gpt1-style"],
)
def test_tiny_model_overfits_one_batch(variant):
    model = tiny(**variant, n_layer=2, d_model=64, n_head=4)
    torch.manual_seed(0)
    idx = torch.randint(0, V, (4, T))
    idx[:, 0] = torch.tensor([1, 2, 3, 4])  # distinct first tokens, so every target is predictable
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(300):
        _, loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.02


def test_param_counts():
    tied = tiny()
    untied = tiny(tie_weights=False)
    learned = tiny(pos_encoding="learned")
    c_t, c_u, c_l = param_counts(tied), param_counts(untied), param_counts(learned)
    assert c_t["total"] == sum(p.numel() for p in tied.parameters())  # parameters() skips the tied copy
    assert c_u["total"] - c_t["total"] == V * 32
    assert c_t["non_embedding"] == c_u["non_embedding"]
    assert c_l["embedding"] == c_t["embedding"] + T * 32
    assert c_l["non_embedding"] == c_t["non_embedding"]
    assert tied.lm_head.weight.data_ptr() == tied.wte.weight.data_ptr()


def test_init_scales():
    model = tiny(n_layer=8, d_model=256, n_head=4, vocab_size=4096, init_std=0.02)
    assert model.wte.weight.std().item() == pytest.approx(0.02, rel=0.05)
    proj = model.blocks[0].attn.proj.weight.std().item()
    assert proj == pytest.approx(0.02 / math.sqrt(16), rel=0.05)
    qkv = model.blocks[0].attn.qkv.weight.std().item()
    assert qkv == pytest.approx(0.02, rel=0.05)
