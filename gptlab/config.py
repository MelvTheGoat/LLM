"""Run configuration.

Every run is described by one small YAML file. The file only lists values that
differ from the defaults below. When a run starts, the full config (every field,
including defaults) is saved next to its logs, together with the git commit.
That saved file is enough to re-run the job exactly.

Loading is strict: an unknown key is an error. This stops a typo such as
`lr_warmup` from being silently ignored.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file has a wrong key or a bad value."""


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    seq_len: int = 1024  # context length in tokens
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    pos_encoding: str = "rope"  # "rope" or "learned"
    norm: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    norm_placement: str = "pre"  # "pre" (norm before each sub-layer) or "post"
    mlp: str = "swiglu"  # "swiglu" or "gelu"
    mlp_hidden: int | None = None  # None: 4*d for gelu, about 8/3*d for swiglu
    tie_weights: bool = True  # share the token embedding and output matrices
    qk_norm: bool = False  # normalize queries and keys (a stability fix)
    bias: bool = False  # add bias vectors to linear layers
    dropout: float = 0.0
    init_std: float = 0.02
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def resolved_mlp_hidden(self) -> int:
        """Width of the MLP hidden layer.

        GELU uses the usual 4*d. SwiGLU has three weight matrices instead of
        two, so it uses 2/3 of that width, rounded to the nearest multiple of
        32 (fast on tensor cores). This keeps the parameter count within a few
        percent, which makes the GELU vs SwiGLU ablation a fair comparison.
        """
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        if self.mlp == "gelu":
            return 4 * self.d_model
        return max(32, 32 * round(8 * self.d_model / 3 / 32))

    def validate(self) -> None:
        _choice("model.pos_encoding", self.pos_encoding, ["rope", "learned"])
        _choice("model.norm", self.norm, ["rmsnorm", "layernorm"])
        _choice("model.norm_placement", self.norm_placement, ["pre", "post"])
        _choice("model.mlp", self.mlp, ["swiglu", "gelu"])
        for name in ["vocab_size", "seq_len", "n_layer", "n_head", "d_model"]:
            if getattr(self, name) <= 0:
                raise ConfigError(f"model.{name} must be positive")
        if self.d_model % self.n_head != 0:
            raise ConfigError(
                f"model.d_model ({self.d_model}) must be divisible by model.n_head ({self.n_head})"
            )
        if self.pos_encoding == "rope" and self.head_dim % 2 != 0:
            raise ConfigError("RoPE needs an even head size (d_model / n_head)")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        if self.vocab_size > 65536:
            raise ConfigError("vocab_size must fit in uint16 token shards (<= 65536)")


@dataclass
class DataConfig:
    name: str = "fineweb-edu-16k"  # folder name, both locally and on the Hub
    root: str | None = None  # local data root; None means $GPTLAB_DATA_ROOT or ./data
    max_train_shards: int | None = None  # use only the first N train shards
    seed: int = 1234  # seed for the order in which training blocks are read

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("data.name must be set")
        if self.max_train_shards is not None and self.max_train_shards <= 0:
            raise ConfigError("data.max_train_shards must be positive")


@dataclass
class TrainConfig:
    tokens: int | None = None  # total training tokens (set this or max_steps)
    max_steps: int | None = None
    global_batch_tokens: int = 524288  # tokens per optimizer step, over all GPUs
    micro_batch_size: int = 16  # sequences per GPU per forward pass
    lr: float = 6e-4  # peak learning rate
    min_lr_ratio: float = 0.1  # final LR = lr * min_lr_ratio
    warmup_steps: int = 200
    schedule: str = "cosine"  # "cosine", "linear" or "constant" after warmup
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0  # max gradient norm; 0 turns clipping off
    z_loss: float = 0.0  # weight of the z-loss term (a stability fix); 0 = off
    precision: str = "fp16"  # "fp16", "bf16" or "fp32"
    compile: bool = False  # use torch.compile
    print_interval: int = 10  # print a progress line every N steps
    debug_interval: int = 100  # per-layer debug stats every N steps; 0 = off
    ckpt_interval_minutes: float = 30.0
    ckpt_interval_steps: int | None = None
    keep_checkpoints: int = 2  # local full checkpoints to keep
    diverge_patience: int = 100  # stop after this many bad steps in a row
    diverge_loss: float | None = None  # a step is "bad" above this loss; None = ln(vocab) + 2

    def validate(self) -> None:
        if (self.tokens is None) == (self.max_steps is None):
            raise ConfigError("set exactly one of train.tokens and train.max_steps")
        _choice("train.precision", self.precision, ["fp16", "bf16", "fp32"])
        _choice("train.schedule", self.schedule, ["cosine", "linear", "constant"])
        if self.lr <= 0:
            raise ConfigError("train.lr must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ConfigError("train.min_lr_ratio must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ConfigError("train.warmup_steps must be >= 0")
        if self.global_batch_tokens <= 0 or self.micro_batch_size <= 0:
            raise ConfigError("batch sizes must be positive")


@dataclass
class EvalConfig:
    interval: int = 250  # validation loss every N steps; 0 = only at the end
    tokens: int = 2_097_152  # validation tokens for the periodic check
    final_tokens: int = 20_971_520  # validation tokens for the final number
    hellaswag: bool = True
    hellaswag_limit: int | None = None  # use only the first N examples
    sample_tokens: int = 96  # tokens generated per prompt at the end
    sample_prompts: list[str] = field(
        default_factory=lambda: [
            "The most important idea in physics is",
            "In the year 1850, the city of",
            "To make bread at home, you need",
            "Photosynthesis is the process by which",
        ]
    )

    def validate(self) -> None:
        if self.tokens <= 0 or self.final_tokens <= 0:
            raise ConfigError("eval token counts must be positive")


@dataclass
class RunConfig:
    name: str
    seed: int = 1337  # seed for weight init and dropout
    notes: str = ""  # one line on what this run tests
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        if not self.name or "/" in self.name or " " in self.name:
            raise ConfigError("name must be non-empty with no spaces or slashes")
        self.model.validate()
        self.data.validate()
        self.train.validate()
        self.eval.validate()


# ---------------------------------------------------------------------------
# Generic loading helpers (also used by the data and benchmark configs)
# ---------------------------------------------------------------------------


def _choice(key: str, value: str, options: list[str]) -> None:
    if value not in options:
        raise ConfigError(f"{key} must be one of {options}, got {value!r}")


def _coerce(value: Any, hint: Any, key: str) -> Any:
    """Check a YAML value against a type hint, converting where it is safe."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{key} must be a mapping")
        return from_dict(hint, value, prefix=key + ".")

    # Optional[X] and X | None
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return _coerce(value, inner[0], key)
        raise ConfigError(f"unsupported type for {key}: {hint}")

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be a list")
        (inner,) = args or (Any,)
        return [_coerce(v, inner, f"{key}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{key} must be a mapping")
        return dict(value)

    if hint is Any:
        return value
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{key} must be true or false, got {value!r}")
    if hint is int:
        if isinstance(value, bool):
            raise ConfigError(f"{key} must be an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                as_float = float(value.replace("_", ""))
            except ValueError:
                pass
            else:
                if as_float.is_integer():
                    return int(as_float)
        raise ConfigError(f"{key} must be an integer, got {value!r}")
    if hint is float:
        if isinstance(value, bool):
            raise ConfigError(f"{key} must be a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # PyYAML reads "6e-4" (no dot) as a string. Accept it as a number.
            try:
                return float(value.replace("_", ""))
            except ValueError:
                pass
        raise ConfigError(f"{key} must be a number, got {value!r}")
    if hint is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{key} must be a string, got {value!r}")
    raise ConfigError(f"unsupported type for {key}: {hint}")


def from_dict(cls: type, data: dict, prefix: str = "") -> Any:
    """Build dataclass `cls` from a dict. Unknown keys are an error."""
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or 'config'} must be a mapping")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = [k for k in data if k not in names]
    if unknown:
        key = unknown[0]
        close = difflib.get_close_matches(key, names, n=1)
        hint = f" (did you mean {prefix}{close[0]}?)" if close else ""
        raise ConfigError(f"unknown config key {prefix}{key}{hint}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name], prefix + f.name)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"missing required config key {prefix}{f.name}")
    return cls(**kwargs)


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply command line overrides like `train.lr=3e-4` to a config dict."""
    data = json.loads(json.dumps(data))  # deep copy
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set {key}: {part} is not a section")
        node[parts[-1]] = value
    return data


def read_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load and check a run config from a YAML file."""
    data = apply_overrides(read_yaml(path), overrides or [])
    cfg = from_dict(RunConfig, data)
    cfg.validate()
    return cfg


def config_from_dict(data: dict) -> RunConfig:
    cfg = from_dict(RunConfig, data)
    cfg.validate()
    return cfg


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False, allow_unicode=True)


def config_hash(cfg: Any) -> str:
    """Short hash of everything that changes the result (the notes are left out)."""
    data = to_dict(cfg)
    data.pop("notes", None)
    text = json.dumps(data, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]
