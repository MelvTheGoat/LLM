"""Every config file in the repo must load. This catches typos before Kaggle does."""

from pathlib import Path

import pytest

from gptlab.data.prepare import load_data_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "data").glob("*.yaml")), ids=lambda p: p.name)
def test_data_configs_load(path):
    cfg = load_data_config(path)
    assert cfg.name


@pytest.mark.parametrize("path", sorted(ROOT.glob("configs/**/*.yaml")), ids=lambda p: str(p.relative_to(ROOT)))
def test_run_configs_load(path):
    """Every run config (anything with a `train` section) must load."""
    from gptlab.config import load_config, read_yaml

    if "train" in read_yaml(path):
        assert load_config(path).name


def test_queue_loads_and_smoke_spec_points_to_real_files():
    from gptlab.config import read_yaml
    from gptlab.runner.queue import load_queue

    queue = load_queue(ROOT / "runs" / "queue.yaml", ROOT)
    assert queue.jobs[0].kind == "smoke"  # the smoke test goes first
    for job in queue.jobs:
        if job.kind == "smoke":
            spec = read_yaml(ROOT / job.config)
            for key in ["data_config", "train_config", "bench_config"]:
                assert (ROOT / spec[key]).exists(), spec[key]
            assert int(spec["stop_after_steps"]) > 0
