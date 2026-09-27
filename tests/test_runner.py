"""End-to-end test of the Kaggle runner on CPU.

A local bare git repo stands in for GitHub (the results branch) and a local
folder stands in for the Hugging Face Hub. Each call to the runner is one
"Kaggle session" with a fresh work folder.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from gptlab.runner.run import copy_tail, main

ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def setup(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    cfgs = tmp_path / "cfgs"
    cfgs.mkdir()
    data_cfg = write_yaml(cfgs / "data.yaml", {
        "name": "tiny-data",
        "source": {"kind": "jsonl", "path": str(ROOT / "tests" / "data" / "sample_docs.jsonl")},
        "tokenizer": {"vocab_size": 512, "compare_vocab_sizes": [], "train_chars": 10**9},
        "train_tokens": 10**9, "val_fraction": 0.1, "shard_tokens": 20000, "workers": 1,
    })
    train_cfg = write_yaml(cfgs / "tiny-a.yaml", {
        "name": "tiny-a",
        "model": {"vocab_size": 512, "seq_len": 32, "n_layer": 2, "n_head": 2, "d_model": 32},
        "data": {"name": "tiny-data"},
        "train": {"max_steps": 8, "global_batch_tokens": 128, "micro_batch_size": 4, "precision": "fp32",
                  "print_interval": 2, "debug_interval": 4, "warmup_steps": 2},
        "eval": {"interval": 4, "tokens": 256, "final_tokens": 256, "hellaswag": False, "sample_tokens": 4},
    })
    broken_cfg = write_yaml(cfgs / "broken.yaml", {
        "name": "broken", "data": {"name": "no-such-data"}, "train": {"max_steps": 2, "precision": "fp32"},
    })
    bench_cfg = write_yaml(cfgs / "bench.yaml", {
        "name": "bench-tiny", "base_model": {"vocab_size": 128, "seq_len": 16},
        "models": {"t": {"d_model": 32, "n_layer": 1, "n_head": 2}},
        "cases": [{"model": "t", "precision": "fp32", "micro_batch_size": 2}],
        "warmup_steps": 1, "timed_steps": 2,
    })
    queue = write_yaml(tmp_path / "queue.yaml", {
        "settings": {"heartbeat_minutes": 0.01, "ckpt_check_seconds": 1},
        "jobs": [
            {"name": "data-tiny", "kind": "data", "config": str(data_cfg), "hardware": "cpu"},
            {"name": "tiny-a", "kind": "train", "config": str(train_cfg), "hardware": "cpu",
             "after": ["data-tiny"], "args": {"stop_after_steps": 5}},
            {"name": "broken", "kind": "train", "config": str(broken_cfg), "hardware": "cpu"},
            {"name": "bench-tiny", "kind": "bench", "config": str(bench_cfg), "hardware": "cpu"},
            {"name": "gpu-only", "kind": "bench", "config": str(bench_cfg), "hardware": "gpu"},
        ],
    })
    return tmp_path, remote, queue


def session(tmp_path, remote, queue, n):
    main([
        "--queue", str(queue), "--work", str(tmp_path / f"work{n}"), "--hardware", "cpu",
        "--local-store", str(tmp_path / "store"), "--results-url", str(remote),
    ])


def read_results(tmp_path, remote, n):
    clone = tmp_path / f"check{n}"
    subprocess.run(["git", "clone", "-q", "--branch", "results", str(remote), str(clone)], check=True)
    out = {}
    for d in (clone / "results").iterdir():
        out[d.name] = json.loads((d / "status.json").read_text())
    return clone, out


@pytest.mark.slow
def test_three_sessions(setup):
    tmp_path, remote, queue = setup
    store = tmp_path / "store"

    # Session 1: data job, training pauses after 5 steps, broken job fails, bench runs.
    session(tmp_path, remote, queue, 1)
    clone, st = read_results(tmp_path, remote, 1)
    assert st["data-tiny"]["state"] == "done"
    assert (store / "data" / "tiny-data" / "manifest.json").exists()
    assert (clone / "results" / "data-tiny" / "manifest.json").exists()
    assert st["tiny-a"]["state"] == "paused"
    assert st["tiny-a"]["run_state"]["step"] == 5
    assert (store / "checkpoints" / "tiny-a" / "latest" / "CHECKPOINT").read_text().strip() == "step_0000005"
    assert st["broken"]["state"] == "failed" and st["broken"]["attempts"] == 1
    assert st["bench-tiny"]["state"] == "done"
    assert (clone / "results" / "bench-tiny" / "bench.json").exists()
    assert "gpu-only" not in st
    metrics = (clone / "results" / "tiny-a" / "metrics.jsonl").read_text().splitlines()
    assert len(metrics) == 5

    # Session 2 (fresh disk): training resumes from the Hub copy and finishes; broken retries once.
    session(tmp_path, remote, queue, 2)
    clone, st = read_results(tmp_path, remote, 2)
    assert st["tiny-a"]["state"] == "done" and st["tiny-a"]["sessions"] == 2
    assert st["tiny-a"]["attempts"] == 1  # a resume is not a new attempt
    assert st["broken"]["state"] == "failed" and st["broken"]["attempts"] == 2
    assert (store / "checkpoints" / "tiny-a" / "final" / "model.pt").exists()
    assert not (store / "checkpoints" / "tiny-a" / "latest").exists()
    res = clone / "results" / "tiny-a"
    steps = [json.loads(line)["step"] for line in (res / "metrics.jsonl").read_text().splitlines()]
    assert steps == list(range(1, 9))
    assert json.loads((res / "final_eval.json").read_text())["val_tokens"] > 0
    events = [json.loads(line)["event"] for line in (res / "events.jsonl").read_text().splitlines()]
    assert events == ["session_start", "paused", "session_start", "training_done"]

    # Session 3: nothing left that a CPU session may run.
    session(tmp_path, remote, queue, 3)
    _, st3 = read_results(tmp_path, remote, 3)
    assert st3 == st


def test_copy_tail(tmp_path):
    src = tmp_path / "big.log"
    src.write_text("".join(f"line {i}\n" for i in range(100000)))
    dst = tmp_path / "small.log"
    copy_tail(src, dst, max_mb=0.01)
    text = dst.read_text()
    assert text.startswith("[... earlier output cut ...]\nline ")
    assert text.endswith("line 99999\n") and len(text) < 0.02 * 2**20


@pytest.mark.slow
def test_smoke_job_on_cpu(tmp_path):
    """The smoke job's code path, with local data and a tiny model instead of FineWeb and GPUs."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    data_cfg = write_yaml(tmp_path / "data.yaml", {
        "name": "smoke-data-test",
        "source": {"kind": "jsonl", "path": str(ROOT / "tests" / "data" / "sample_docs.jsonl")},
        "tokenizer": {"vocab_size": 512, "compare_vocab_sizes": [400], "train_chars": 10**9},
        "train_tokens": 10**9, "val_fraction": 0.1, "shard_tokens": 20000, "workers": 1,
    })
    train_cfg = write_yaml(tmp_path / "train.yaml", {
        "name": "smoke-train-test",
        "model": {"vocab_size": 512, "seq_len": 32, "n_layer": 2, "n_head": 2, "d_model": 32},
        "data": {"name": "smoke-data-test"},
        "train": {"max_steps": 6, "global_batch_tokens": 128, "micro_batch_size": 4, "precision": "fp32",
                  "ckpt_interval_steps": 2},
        "eval": {"interval": 3, "tokens": 256, "final_tokens": 256, "hellaswag": False, "sample_tokens": 4},
    })
    bench_cfg = write_yaml(tmp_path / "bench.yaml", {
        "name": "b", "base_model": {"vocab_size": 128, "seq_len": 16},
        "models": {"t": {"d_model": 32, "n_layer": 1, "n_head": 2}},
        "cases": [{"model": "t", "precision": "fp32", "micro_batch_size": 2}], "warmup_steps": 1, "timed_steps": 2,
    })
    spec = write_yaml(tmp_path / "smoke.yaml", {"data_config": str(data_cfg), "train_config": str(train_cfg),
                                                "bench_config": str(bench_cfg), "stop_after_steps": 3})
    queue = write_yaml(tmp_path / "queue.yaml", {
        "jobs": [{"name": "smoke", "kind": "smoke", "config": str(spec), "hardware": "cpu"}],
    })
    session(tmp_path, str(remote), queue, 1)
    clone, st = read_results(tmp_path, str(remote), 1)
    report = json.loads((clone / "results" / "smoke" / "smoke_report.json").read_text())
    assert st["smoke"]["state"] == "done", report
    resume = report["steps"]["resume"]
    assert resume["steps"] == 6 and resume["resumed_at"] == 3
    assert resume["identical_after_resume"]  # CPU is deterministic, so the resume must be exact
    assert resume["max_diff_straight_vs_straight2"] == 0.0
    for sub in ["straight", "straight2", "resume"]:
        assert (clone / "results" / "smoke" / sub / "metrics.jsonl").exists()
    assert (clone / "results" / "smoke" / "bench" / "bench.json").exists()
    assert (clone / "results" / "smoke" / "data" / "manifest.json").exists()
    assert (tmp_path / "store" / "checkpoints" / "smoke" / "resume" / "final" / "model.pt").exists()
