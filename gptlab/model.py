"""A GPT-style decoder-only transformer, written from scratch.

The config can switch each of these parts, so we can run ablations:

- positional encoding: "learned" (a trained vector per position, added to the
  token embedding) or "rope" (rotary embeddings: queries and keys are rotated
  by an angle that grows with position, so attention scores depend on the
  distance between tokens)
- norm: "layernorm" (subtract mean, divide by std, scale and shift) or
  "rmsnorm" (only divide by the root-mean-square, then scale; cheaper)
- norm placement: "pre" (x + f(norm(x)), used by GPT-2 and most modern models)
  or "post" (norm(x + f(x)), the original 2017 transformer)
- MLP: "gelu" (Linear -> GELU -> Linear) or "swiglu" (a gated MLP:
  Linear_down(SiLU(Linear_gate(x)) * Linear_up(x)), used by LLaMA)
- weight tying: share one matrix between the token embedding and the output layer
- qk_norm: normalize queries and keys per head before attention (a known fix
  for attention logits that grow too large during training)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gptlab.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for accuracy under fp16, then cast back.
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def make_norm(cfg: ModelConfig, dim: int | None = None) -> nn.Module:
    dim = dim or cfg.d_model
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim, cfg.norm_eps)
    return nn.LayerNorm(dim, eps=cfg.norm_eps, bias=cfg.bias)


def rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (seq_len, head_dim / 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float64), inv_freq)
    return angles.cos().float(), angles.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features (i, i + d/2) of x, shape (B, H, T, d), by position."""
    T = x.shape[-2]
    cos, sin = cos[:T], sin[:T]
    xf = x.float()
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.proj.residual_proj = True  # gets a smaller init, see GPT._init_weights
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = make_norm(cfg, cfg.head_dim)
            self.k_norm = make_norm(cfg, cfg.head_dim)

    def forward(self, x, rope=None, stats: dict | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if stats is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            y = self._attention_with_stats(q, k, v, stats)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    @staticmethod
    def _attention_with_stats(q, k, v, stats: dict) -> torch.Tensor:
        """Plain causal attention that also records debug stats.

        Used only for the occasional debug forward pass, because it builds the
        full (T x T) attention matrix, which the fused kernel avoids.
        """
        T = q.shape[-2]
        logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = logits.softmax(dim=-1)
        plogp = torch.where(probs > 0, probs * probs.clamp_min(1e-30).log(), torch.zeros_like(probs))
        entropy = -plogp.sum(-1)  # (B, H, T), in nats
        stats["attn_logit_max"] = logits.max().item()
        stats["attn_entropy_mean"] = entropy.mean().item()
        stats["attn_entropy_min_head"] = entropy.mean(dim=(0, 2)).min().item()
        return (probs @ v.float()).type_as(v)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.resolved_mlp_hidden()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            self.gate_up = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias)  # gate and up, fused
        else:
            self.fc = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.proj.residual_proj = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = F.gelu(self.fc(x))
        return self.drop(self.proj(h))


def _rms(t: torch.Tensor) -> float:
    return t.float().pow(2).mean().sqrt().item()


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pre = cfg.norm_placement == "pre"
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None, stats: dict | None = None):
        if self.pre:
            a = self.attn(self.norm1(x), rope, stats)
            x = x + a
            m = self.mlp(self.norm2(x))
            x = x + m
        else:
            a = self.attn(x, rope, stats)
            x = self.norm1(x + a)
            m = self.mlp(x)
            x = self.norm2(x + m)
        if stats is not None:
            stats["attn_out_rms"] = _rms(a)
            stats["mlp_out_rms"] = _rms(m)
            stats["resid_rms"] = _rms(x)
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model) if cfg.pos_encoding == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        # With post-norm, every block already ends in a norm, so no final norm.
        self.norm_f = make_norm(cfg) if cfg.norm_placement == "pre" else nn.Identity()
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight
        if cfg.pos_encoding == "rope":
            cos, sin = rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2 style init.

        All linear and embedding weights start as N(0, init_std). The two
        projections that write into the residual stream (attention output and
        MLP output) use init_std / sqrt(2 * n_layer). There are 2 * n_layer of
        them adding up, so this keeps the size of the residual stream roughly
        independent of depth at the start of training.
        """
        std = self.cfg.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.lm_head and self.cfg.tie_weights:
                    continue  # shares the embedding matrix, which is set below
                s = std / math.sqrt(2 * self.cfg.n_layer) if getattr(module, "residual_proj", False) else std
                nn.init.normal_(module.weight, mean=0.0, std=s)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None, z_loss_weight: float = 0.0, stats: list | None = None):
        """Return (logits, loss, parts).

        loss = cross-entropy + z_loss_weight * mean(logsumexp(logits)^2). The
        z-loss pushes the log of the softmax normalizer towards 0, which keeps
        output logits from drifting to huge values (used in PaLM).
        `parts` holds the two terms separately for logging. If `stats` is a
        list, one dict of debug stats per layer is appended to it.
        """
        B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"sequence length {T} > model seq_len {self.cfg.seq_len}")
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(x)
        rope = (self.rope_cos, self.rope_sin) if self.rope_cos is not None else None
        for block in self.blocks:
            layer_stats = {} if stats is not None else None
            x = block(x, rope, layer_stats)
            if stats is not None:
                stats.append(layer_stats)
        logits = self.lm_head(self.norm_f(x))
        if stats is not None:
            lf = logits.float()
            stats.append({"logits_abs_max": lf.abs().max().item(), "logsumexp_mean": torch.logsumexp(lf, -1).mean().item()})
        if targets is None:
            return logits, None, {}
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        target_logit = logits_f.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ce = (lse - target_logit).mean()
        z = lse.pow(2).mean()
        loss = ce + z_loss_weight * z if z_loss_weight > 0 else ce
        return logits, loss, {"ce": ce.detach(), "z": z.detach()}


def param_counts(model: GPT) -> dict:
    """Parameter counts. Shared (tied) tensors are counted once.

    - total: every trainable number in the model
    - embedding: token and position embedding tables
    - non_embedding: total minus embeddings (and minus the output matrix when it
      is not tied). This is the "N" used by Kaplan et al. for scaling laws.
    """
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    embedding = model.wte.weight.numel() + (model.wpe.weight.numel() if model.wpe is not None else 0)
    head = 0 if model.cfg.tie_weights else model.lm_head.weight.numel()
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding - head}
