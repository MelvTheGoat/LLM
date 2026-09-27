import json
from pathlib import Path

import pytest

from gptlab.bench import BenchConfig, load_bench_config, run_bench
from gptlab.config import from_dict

ROOT = Path(__file__).resolve().parents[1]


def test_grid_expands_to_all_combinations():
    cfg = from_dict(BenchConfig, {
        "name": "b", "models": {"a": {}, "b": {}},
        "grid": [{"model": ["a", "b"], "compile": [False, True], "precision": "fp32"}],
        "cases": [{"model": "a", "world_size": 1}],
    })
    cases = cfg.all_cases()
    assert len(cases) == 5
    assert {(c.model, c.compile) for c in cases[1:]} == {("a", False), ("a", True), ("b", False), ("b", True)}


@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "bench").glob("*.yaml")), ids=lambda p: p.name)
def test_bench_configs_load(path):
    assert load_bench_config(path).all_cases()


def test_bench_runs_on_cpu(tmp_path):
    cfg = from_dict(BenchConfig, {
        "name": "cpu",
        "base_model": {"vocab_size": 128, "seq_len": 16},
        "models": {"t": {"d_model": 32, "n_layer": 2, "n_head": 2}},
        "cases": [{"model": "t", "precision": "fp32", "micro_batch_size": 2}],
        "warmup_steps": 1, "timed_steps": 2,
    })
    results = run_bench(cfg, tmp_path, device="cpu", log=lambda m: None)
    r = results[0]
    assert r["status"] == "ok" and r["tok_per_s"] > 0 and r["mfu"] is None
    assert r["tokens_per_step"] == 32
    saved = json.loads((tmp_path / "bench.json").read_text())
    assert saved["results"][0]["model"] == "t"
