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


def test_kaggle_notebook_is_valid_and_starts_the_runner():
    import json

    nb = json.loads((ROOT / "kaggle" / "runner.ipynb").read_text())
    code = "".join(c for cell in nb["cells"] if cell["cell_type"] == "code" for c in cell["source"])
    compile(code, "runner.ipynb", "exec")
    assert "gptlab.runner" in code and "SESSION_START" in code
    assert 'BRANCH = "main"' in code


def test_generated_experiment_configs_are_up_to_date():
    import subprocess
    import sys

    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_configs.py"), "--check"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_queue_jobs_point_to_configs_that_load():
    from gptlab.bench import load_bench_config
    from gptlab.config import load_config
    from gptlab.runner.queue import load_queue

    queue = load_queue(ROOT / "runs" / "queue.yaml", ROOT)
    names = [j.name for j in queue.jobs]
    assert len(names) == len(set(names)), "job names must be unique"
    for job in queue.jobs:
        assert all(a in names for a in job.after), f"{job.name} waits for a job that is not in the queue"
        if job.kind == "train":
            cfg = load_config(ROOT / job.config)
            assert cfg.name == job.name, f"{job.config}: config name should match the job name"
            assert cfg.model.vocab_size == 16384 and cfg.data.name == "fineweb-edu-16k"
        if job.kind == "bench":
            assert load_bench_config(ROOT / job.config).all_cases()


def test_kaggle_notebook_works_without_a_gpu():
    import json

    nb = json.loads((ROOT / "kaggle" / "runner.ipynb").read_text())
    code = "".join(c for cell in nb["cells"] if cell["cell_type"] == "code" for c in cell["source"])
    # CPU sessions have no nvidia-smi; calling it unguarded crashes the notebook before the runner starts.
    assert 'if shutil.which("nvidia-smi"):' in code
