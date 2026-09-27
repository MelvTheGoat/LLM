"""FLOP counting and MFU (model FLOPs utilization).

MFU = (FLOPs the model math needs per second) / (the GPU's peak FLOPs per second).
It tells us what fraction of the hardware we actually use.

The 6N rule
-----------
A matrix multiply y = x W with W of shape (m, n) costs 2*m*n FLOPs per token
(one multiply and one add per weight). So a forward pass costs about 2N FLOPs
per token, where N is the number of weights used in matrix multiplies. The
backward pass costs twice that: one matmul for the gradient with respect to the
input and one for the gradient with respect to the weights. Total: 6N per token.

N here counts the weights in all linear layers, including the output layer
(d_model x vocab). It leaves out the embedding lookup (a table read, not a
matmul), norms and biases (tiny element-wise work).

The attention term
------------------
Attention also does two matmuls that have no weights: scores = Q K^T and
out = softmax(scores) V. Per token and per layer, each costs 2 * T * d_model
FLOPs (T = context length) in the forward pass. Times 3 for forward plus
backward, that gives 12 * n_layer * T * d_model per token. For small models with
long context, this term is large. For d_model = 128 and T = 1024 it is bigger
than the 6N term.

We count the full T x T score matrix, as in the PaLM paper (appendix B). With a
causal mask, half of it is masked, so a kernel that skips masked blocks does less
real work. This convention is the common one, which makes our MFU numbers
comparable to other reports.

The formula is checked against PyTorch's own FLOP counter in tests/test_flops.py.
"""

from __future__ import annotations

import os
import re

from gptlab.config import ModelConfig


def matmul_params(cfg: ModelConfig) -> int:
    """Weights that take part in matrix multiplies (the N in 6N)."""
    d, h = cfg.d_model, cfg.resolved_mlp_hidden()
    attn = 4 * d * d  # q, k, v and output projections
    mlp = (3 if cfg.mlp == "swiglu" else 2) * d * h
    return cfg.n_layer * (attn + mlp) + d * cfg.vocab_size  # plus the output layer


def estimate_param_counts(cfg: ModelConfig) -> dict:
    """Parameter counts computed from the config alone (no model is built).

    Matches gptlab.model.param_counts; this is checked in the tests.
    """
    d, L, V, h = cfg.d_model, cfg.n_layer, cfg.vocab_size, cfg.resolved_mlp_hidden()
    norm = d * (2 if (cfg.norm == "layernorm" and cfg.bias) else 1)
    head_norm = cfg.head_dim * (2 if (cfg.norm == "layernorm" and cfg.bias) else 1)
    b = 1 if cfg.bias else 0
    attn = 4 * d * d + b * (3 * d + d)
    if cfg.mlp == "swiglu":
        mlp = 3 * d * h + b * (2 * h + d)
    else:
        mlp = 2 * d * h + b * (h + d)
    layer = attn + mlp + 2 * norm + (2 * head_norm if cfg.qk_norm else 0)
    embedding = V * d + (cfg.seq_len * d if cfg.pos_encoding == "learned" else 0)
    final_norm = norm if cfg.norm_placement == "pre" else 0
    head = 0 if cfg.tie_weights else V * d
    total = embedding + L * layer + final_norm + head
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding - head}


def flops_per_token(cfg: ModelConfig, seq_len: int | None = None) -> dict:
    """Training FLOPs (forward + backward) per token."""
    T = seq_len or cfg.seq_len
    params_term = 6 * matmul_params(cfg)
    attention_term = 12 * cfg.n_layer * T * cfg.d_model
    return {"params_term": params_term, "attention_term": attention_term, "total": params_term + attention_term}


# Dense peak TFLOPS from NVIDIA datasheets (no "2:4 sparsity" numbers).
# fp16 means tensor-core fp16 math with fp32 accumulation, which is what
# autocast uses. fp32 means plain fp32 (TF32 is off for matmuls by default).
_PEAK_TFLOPS = [
    (r"\bT4\b", {"fp16": 65.0, "fp32": 8.1}),
    (r"P100", {"fp16": 18.7, "fp32": 9.3}),  # PCIe 16 GB version (the one on Kaggle)
    (r"V100", {"fp16": 125.0, "fp32": 15.7}),  # SXM2 version
    (r"A100", {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5}),
    (r"\bL4\b", {"fp16": 121.0, "bf16": 121.0, "fp32": 30.3}),
    (r"H100", {"fp16": 989.0, "bf16": 989.0, "fp32": 67.0}),  # SXM version
]


def peak_flops_per_gpu(device_name: str, precision: str) -> float | None:
    """Peak FLOPs per second of one GPU, or None if we do not know the GPU.

    Set GPTLAB_PEAK_TFLOPS to override (for example for an unknown GPU).
    """
    override = os.environ.get("GPTLAB_PEAK_TFLOPS")
    if override:
        return float(override) * 1e12
    for pattern, table in _PEAK_TFLOPS:
        if re.search(pattern, device_name):
            value = table.get(precision)
            return value * 1e12 if value else None
    return None


def mfu(tokens_per_second: float, flops_per_tok: float, n_gpus: int, peak_per_gpu: float | None) -> float | None:
    if not peak_per_gpu or tokens_per_second <= 0:
        return None
    return tokens_per_second * flops_per_tok / (n_gpus * peak_per_gpu)
