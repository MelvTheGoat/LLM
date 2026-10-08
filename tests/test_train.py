"""Training loop tests on CPU with a tiny model and tiny data."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from gptlab.checkpoint import latest_checkpoint
from gptlab.config import RunConfig, config_from_dict
from gptlab.data.prepare import DataPrepConfig, TokenizerPrepConfig, prepare
from gptlab.data.sources import SourceConfig
from gptlab.metrics import read_jsonl
from gptlab.train import train

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "tests" / "data" / "hellaswag_tiny.jsonl"


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """A real (tiny) dataset built by the data pipeline from the sample docs."""
    tmp = tmp_path_factory.mktemp("data")
    cfg = DataPrepConfig(
        name="tiny",
        source=SourceConfig(kind="jsonl", path=str(ROOT / "tests" / "data" / "sample_docs.jsonl")),
        tokenizer=TokenizerPrepConfig(vocab_size=512, compare_vocab_sizes=[], train_chars=10**9),
        train_tokens=10**9,
        val_fraction=0.1,
        shard_tokens=20_000,
        workers=1,
    )
    out = tmp / "tiny"
    prepare(cfg, out, tmp / "work", log=lambda m: None)
    return out


def tiny_cfg(**train_overrides) -> RunConfig:
    d = {
        "name": "tiny",
        "seed": 7,
        "model": dict(vocab_size=512, seq_len=32, n_layer=2, n_head=2, d_model=32, dropout=0.1),
        "train": dict(max_steps=8, global_batch_tokens=128, micro_batch_size=2, lr=3e-3, warmup_steps=2,
                      precision="fp32", print_interval=100, debug_interval=3, ckpt_interval_minutes=1e9),
        "eval": dict(interval=4, tokens=256, final_tokens=512, hellaswag=True, sample_tokens=8,
                     sample_prompts=["Alice was"]),
    }
    d["train"].update(train_overrides)
    return config_from_dict(d)


def run(cfg, out, data, **kw):
    return train(cfg, out, data_dir=data, device="cpu", hellaswag_path=HS, log=lambda m: None, **kw)


def losses(out):
    return [(r["step"], r["loss"]) for r in read_jsonl(Path(out) / "metrics.jsonl")]


def test_training_runs_and_writes_all_logs(tiny_data, tmp_path):
    state = run(tiny_cfg(), tmp_path, tiny_data)
    assert state["state"] == "done" and state["step"] == 8
    m = read_jsonl(tmp_path / "metrics.jsonl")
    assert [r["step"] for r in m] == list(range(1, 9))
    assert m[-1]["loss"] < m[0]["loss"]
    for key in ["loss", "lr", "grad_norm", "tok_per_s", "tokens"]:
        assert key in m[0]
    assert [r["step"] for r in read_jsonl(tmp_path / "eval.jsonl")] == [4]
    dbg = read_jsonl(tmp_path / "debug.jsonl")
    assert [r["step"] for r in dbg] == [3, 6]
    assert len(dbg[0]["attn_entropy_mean"]) == 2 and "layer0" in dbg[0]["grad_norm"]
    final = json.loads((tmp_path / "final_eval.json").read_text())
    assert final["val_tokens"] > 0 and "val_bpb" in final
    assert final["hellaswag_n"] == 3
    assert (tmp_path / "samples.txt").exists() and (tmp_path / "final" / "model.pt").exists()
    assert json.loads((tmp_path / "run_state.json").read_text())["state"] == "done"


def test_resume_is_exact(tiny_data, tmp_path):
    """N steps straight == N/2 steps, stop, resume in fresh objects, N/2 more.

    Dropout is on, so this also checks that random states are restored.
    """
    cfg = tiny_cfg()
    run(cfg, tmp_path / "straight", tiny_data)

    first = run(cfg, tmp_path / "resumed", tiny_data, stop_after_steps=4)
    assert first["state"] == "paused" and first["step"] == 4
    assert latest_checkpoint(tmp_path / "resumed" / "ckpt").name == "step_0000004"
    second = run(tiny_cfg(), tmp_path / "resumed", tiny_data)
    assert second["state"] == "done"

    a, b = losses(tmp_path / "straight"), losses(tmp_path / "resumed")
    assert a == b  # exact equality, every step
    ev_a = read_jsonl(tmp_path / "straight" / "eval.jsonl")
    ev_b = read_jsonl(tmp_path / "resumed" / "eval.jsonl")
    assert [e["val_loss"] for e in ev_a] == [e["val_loss"] for e in ev_b]
    wa = torch.load(tmp_path / "straight" / "final" / "model.pt")
    wb = torch.load(tmp_path / "resumed" / "final" / "model.pt")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    events = [e["event"] for e in read_jsonl(tmp_path / "resumed" / "events.jsonl")]
    assert events == ["session_start", "paused", "session_start", "training_done"]


def test_resume_refuses_a_changed_config(tiny_data, tmp_path):
    run(tiny_cfg(), tmp_path, tiny_data, stop_after_steps=2)
    with pytest.raises(ValueError, match="config changed"):
        run(tiny_cfg(lr=1e-2), tmp_path, tiny_data)
    # Logging settings may change without breaking exact resume.
    assert run(tiny_cfg(print_interval=1), tmp_path, tiny_data)["state"] == "done"


def test_gradient_accumulation_matches_a_bigger_micro_batch(tiny_data, tmp_path):
    a = run(tiny_cfg(micro_batch_size=2, max_steps=4), tmp_path / "a", tiny_data)  # accum 2
    b = run(tiny_cfg(micro_batch_size=4, max_steps=4), tmp_path / "b", tiny_data)  # accum 1
    assert a["state"] == b["state"] == "done"
    for (_, la), (_, lb) in zip(losses(tmp_path / "a"), losses(tmp_path / "b")):
        # Dropout masks differ between the two layouts, so compare without dropout below.
        assert abs(la - lb) < 0.5
    cfg_a = tiny_cfg(micro_batch_size=2, max_steps=4)
    cfg_b = tiny_cfg(micro_batch_size=4, max_steps=4)
    cfg_a.model.dropout = cfg_b.model.dropout = 0.0
    run(cfg_a, tmp_path / "c", tiny_data)
    run(cfg_b, tmp_path / "d", tiny_data)
    for (_, la), (_, lb) in zip(losses(tmp_path / "c"), losses(tmp_path / "d")):
        assert abs(la - lb) < 1e-5


def test_deadline_saves_and_pauses(tiny_data, tmp_path):
    state = run(tiny_cfg(), tmp_path, tiny_data, deadline=time.time() - 1)
    assert state["state"] == "paused" and state["step"] == 0
    assert latest_checkpoint(tmp_path / "ckpt") is not None


def test_divergence_is_detected(tiny_data, tmp_path):
    state = run(tiny_cfg(diverge_loss=0.01, diverge_patience=3), tmp_path, tiny_data)
    assert state["state"] == "diverged" and state["step"] == 3


def test_crash_saves_an_emergency_checkpoint(tiny_data, tmp_path, monkeypatch):
    import gptlab.train as train_mod

    calls = {"n": 0}
    real = train_mod.lr_at

    def flaky(step, total, cfg):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("simulated crash")
        return real(step, total, cfg)

    monkeypatch.setattr(train_mod, "lr_at", flaky)
    with pytest.raises(RuntimeError, match="simulated"):
        run(tiny_cfg(), tmp_path / "crash", tiny_data)
    assert latest_checkpoint(tmp_path / "crash" / "ckpt").name == "step_0000003"
    monkeypatch.setattr(train_mod, "lr_at", real)
    run(tiny_cfg(), tmp_path / "crash", tiny_data)
    run(tiny_cfg(), tmp_path / "straight", tiny_data)
    assert losses(tmp_path / "crash") == losses(tmp_path / "straight")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_ddp_two_processes_match_one_process(tiny_data, tmp_path):
    """2 CPU processes (gloo) with micro 2 == 1 process with micro 2 x accum 2."""
    cfg = tiny_cfg(max_steps=6)
    cfg.model.dropout = 0.0
    cfg_path = tmp_path / "cfg.yaml"
    from gptlab.config import save_config

    save_config(cfg, cfg_path)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port", str(_free_port()),
        "-m", "gptlab.train", "--config", str(cfg_path), "--out-dir", str(tmp_path / "ddp"),
        "--data-dir", str(tiny_data), "--device", "cpu", "--hellaswag-path", str(HS),
    ]
    subprocess.run(cmd, check=True, env=env, cwd=ROOT, capture_output=True, timeout=600)
    run(cfg, tmp_path / "single", tiny_data)
    ddp, single = losses(tmp_path / "ddp"), losses(tmp_path / "single")
    assert len(ddp) == len(single) == 6
    for (sa, la), (sb, lb) in zip(ddp, single):
        assert sa == sb and abs(la - lb) < 1e-4
    ev = json.loads((tmp_path / "ddp" / "events.jsonl").read_text().splitlines()[0])
    assert ev["world_size"] == 2 and ev["accum"] == 1


@pytest.mark.slow
def test_compiled_ddp_with_accumulation_and_resume(tiny_data, tmp_path):
    """The real runs use torch.compile + DDP + accumulation + resume; check that mix on CPU."""
    cfg = tiny_cfg(max_steps=6, global_batch_tokens=256, compile=True)
    cfg.model.dropout = 0.0
    cfg_path = tmp_path / "cfg.yaml"
    from gptlab.config import save_config

    save_config(cfg, cfg_path)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    base = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port", "0",
        "-m", "gptlab.train", "--config", str(cfg_path), "--out-dir", str(tmp_path / "ddp"),
        "--data-dir", str(tiny_data), "--device", "cpu", "--hellaswag-path", str(HS),
    ]
    for extra in (["--stop-after-steps", "3"], []):  # stop halfway, then resume
        base[5] = str(_free_port())
        p = subprocess.run(base + extra, env=env, cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert p.returncode == 0, p.stderr[-3000:]
    cfg.train.compile = False
    run(cfg, tmp_path / "single", tiny_data)
    ddp, single = losses(tmp_path / "ddp"), losses(tmp_path / "single")
    assert [s for s, _ in ddp] == [s for s, _ in single] == list(range(1, 7))
    for (_, la), (_, lb) in zip(ddp, single):
        assert abs(la - lb) < 1e-3
    events = [json.loads(line) for line in (tmp_path / "ddp" / "events.jsonl").read_text().splitlines()]
    assert events[0]["accum"] == 2 and events[-1]["event"] == "training_done"
    assert (tmp_path / "ddp" / "final_eval.json").exists()
