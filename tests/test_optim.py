import pytest

from gptlab.config import ModelConfig, TrainConfig
from gptlab.model import GPT
from gptlab.optim import build_optimizer, lr_at


def cfg(**kw):
    base = dict(max_steps=100, lr=1e-3, min_lr_ratio=0.1, warmup_steps=10)
    base.update(kw)
    return TrainConfig(**base)


def test_warmup_is_linear_and_reaches_peak():
    c = cfg()
    assert lr_at(0, 100, c) == pytest.approx(1e-4)
    assert lr_at(4, 100, c) == pytest.approx(5e-4)
    assert lr_at(9, 100, c) == pytest.approx(1e-3)


def test_cosine_decays_to_min_lr():
    c = cfg()
    assert lr_at(10, 100, c) == pytest.approx(1e-3)
    assert lr_at(55, 100, c) == pytest.approx(1e-4 + 0.5 * 9e-4)  # halfway
    assert lr_at(100, 100, c) == pytest.approx(1e-4)
    assert lr_at(500, 100, c) == pytest.approx(1e-4)  # past the end it stays at min
    values = [lr_at(s, 100, c) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_no_warmup_linear_and_constant():
    assert lr_at(0, 100, cfg(warmup_steps=0)) == pytest.approx(1e-3)
    lin = cfg(warmup_steps=0, schedule="linear")
    assert lr_at(50, 100, lin) == pytest.approx(1e-3 - 0.5 * 9e-4)
    assert lr_at(70, 100, cfg(schedule="constant")) == pytest.approx(1e-3)


def test_weight_decay_only_on_matrices():
    model = GPT(ModelConfig(vocab_size=64, seq_len=8, n_layer=2, n_head=2, d_model=16, norm="layernorm", bias=True))
    opt = build_optimizer(model, cfg(weight_decay=0.1), "cpu")
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in model.parameters())
