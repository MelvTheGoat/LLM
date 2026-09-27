import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from gptlab.config import ModelConfig
from gptlab.flops import estimate_param_counts, flops_per_token, mfu, peak_flops_per_gpu
from gptlab.model import GPT, param_counts

CONFIGS = [
    dict(mlp="gelu"),
    dict(mlp="swiglu", tie_weights=False),
    dict(pos_encoding="learned", norm="layernorm", bias=True, qk_norm=True),
    dict(norm_placement="post", mlp="gelu", qk_norm=True),
]


def small(**kw):
    base = dict(vocab_size=128, seq_len=32, n_layer=2, n_head=2, d_model=64)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize("kw", CONFIGS)
def test_flop_formula_matches_pytorch_counter(kw):
    """Count real forward+backward FLOPs with PyTorch and compare to 6N + attention.

    We use the explicit attention path (stats=[]) because PyTorch's counter does
    not see inside the fused CPU attention kernel.
    """
    cfg = small(**kw)
    model = GPT(cfg)
    B, T = 2, cfg.seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with FlopCounterMode(display=False) as counter:
        _, loss, _ = model(idx, idx, stats=[])
        loss.backward()
    assert counter.get_total_flops() == flops_per_token(cfg)["total"] * B * T


@pytest.mark.parametrize("kw", CONFIGS + [dict(bias=True, norm="layernorm", mlp="gelu", tie_weights=False)])
def test_param_estimate_matches_model(kw):
    cfg = small(**kw)
    assert estimate_param_counts(cfg) == param_counts(GPT(cfg))


def test_attention_term_dominates_for_small_wide_context_models():
    f = flops_per_token(ModelConfig(d_model=128, n_head=2, n_layer=4, seq_len=1024, vocab_size=16384))
    assert f["attention_term"] > 0.3 * f["params_term"]
    big = flops_per_token(ModelConfig(d_model=768, n_head=12, n_layer=12, seq_len=1024, vocab_size=16384))
    assert big["attention_term"] < 0.25 * big["params_term"]


def test_peak_flops_table(monkeypatch):
    monkeypatch.delenv("GPTLAB_PEAK_TFLOPS", raising=False)
    assert peak_flops_per_gpu("Tesla T4", "fp16") == 65e12
    assert peak_flops_per_gpu("Tesla T4", "fp32") == 8.1e12
    assert peak_flops_per_gpu("Tesla T4", "bf16") is None  # T4 has no bf16 tensor cores
    assert peak_flops_per_gpu("Tesla P100-PCIE-16GB", "fp16") == 18.7e12
    assert peak_flops_per_gpu("NVIDIA L40S", "fp16") is None  # must not match "L4"
    assert peak_flops_per_gpu("cpu", "fp32") is None
    monkeypatch.setenv("GPTLAB_PEAK_TFLOPS", "10")
    assert peak_flops_per_gpu("anything", "fp16") == 10e12


def test_mfu():
    assert mfu(1000.0, 1e9, 2, 1e12) == pytest.approx(0.5)
    assert mfu(1000.0, 1e9, 1, None) is None
