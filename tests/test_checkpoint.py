import random

import numpy as np
import torch

from gptlab.checkpoint import (
    UPLOAD_MARKER,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    rng_state,
    save_checkpoint,
    set_rng_state,
)
from gptlab.config import ModelConfig, TrainConfig
from gptlab.metrics import JsonlWriter, forward_stats, grad_and_weight_norms, read_jsonl, truncate_jsonl
from gptlab.model import GPT
from gptlab.optim import build_optimizer


def make():
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=64, seq_len=8, n_layer=2, n_head=2, d_model=16))
    opt = build_optimizer(model, TrainConfig(max_steps=10), "cpu")
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, opt, scaler


def one_step(model, opt):
    x = torch.randint(0, 64, (2, 8))
    _, loss, _ = model(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad()


def test_save_load_round_trip(tmp_path):
    model, opt, scaler = make()
    one_step(model, opt)
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n')
    path = save_checkpoint(
        tmp_path / "ckpt", 1, model=model, optimizer=opt, scaler=scaler,
        state={"world_size": 1, "note": "x"}, rank=0, run_dir=tmp_path,
    )
    assert path.name == "step_0000001"
    assert (path / "logs" / "metrics.jsonl").exists()
    assert latest_checkpoint(tmp_path / "ckpt") == path

    model2, opt2, scaler2 = make()
    state = load_checkpoint(path, model=model2, optimizer=opt2, scaler=scaler2, rank=0, world_size=1, map_location="cpu")
    assert state["step"] == 1 and state["note"] == "x" and state["rng_exact"]
    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)
    # Optimizer moments restored: the next step gives identical weights.
    torch.manual_seed(5)
    one_step(model, opt)
    torch.manual_seed(5)
    one_step(model2, opt2)
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)


def test_prune_keeps_newest_and_uploading(tmp_path):
    model, opt, scaler = make()
    kw = dict(model=model, optimizer=opt, scaler=scaler, state={}, rank=0, run_dir=tmp_path, keep=2)
    first = save_checkpoint(tmp_path / "ckpt", 1, **kw)
    (first / UPLOAD_MARKER).touch()  # pretend the runner is uploading it
    for step in [2, 3, 4]:
        save_checkpoint(tmp_path / "ckpt", step, **kw)
    names = [p.name for p in list_checkpoints(tmp_path / "ckpt")]
    assert names == ["step_0000001", "step_0000003", "step_0000004"]


def test_half_written_checkpoint_is_ignored(tmp_path):
    model, opt, scaler = make()
    good = save_checkpoint(tmp_path / "ckpt", 5, model=model, optimizer=opt, scaler=scaler, state={}, rank=0, run_dir=tmp_path)
    (tmp_path / "ckpt" / "step_0000009.tmp").mkdir()  # crash while saving
    (tmp_path / "ckpt" / "step_0000008").mkdir()  # no meta.json: incomplete
    (tmp_path / "ckpt" / "LATEST").unlink()
    assert latest_checkpoint(tmp_path / "ckpt") == good


def test_rng_state_round_trip():
    s = rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_rng_state(s)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_jsonl_truncate_and_nan(tmp_path):
    w = JsonlWriter(tmp_path / "m.jsonl")
    for step in range(1, 6):
        w.write({"step": step, "loss": float("nan") if step == 3 else 1.0})
    w.close()
    rows = read_jsonl(tmp_path / "m.jsonl")
    assert rows[2]["loss"] == "nan"
    truncate_jsonl(tmp_path / "m.jsonl", 3)
    assert [r["step"] for r in read_jsonl(tmp_path / "m.jsonl")] == [1, 2, 3]


def test_debug_stats_have_one_value_per_layer():
    model, opt, _ = make()
    x = torch.randint(0, 64, (2, 8))
    s = forward_stats(model, x, x)
    assert len(s["attn_entropy_mean"]) == 2 and len(s["resid_rms"]) == 2
    assert model.training  # restored
    _, loss, _ = model(x, x)
    loss.backward()
    norms = grad_and_weight_norms(model)
    assert set(norms["grad_norm"]) >= {"layer0", "layer1", "wte"}
