import pytest

from gptlab.config import ConfigError
from gptlab.runner.queue import Job, Queue, Settings, load_queue, pick_next

NOW = 1_000_000.0


def q(*jobs):
    return Queue(settings=Settings(), jobs=list(jobs))


def test_first_unfinished_job_in_order():
    queue = q(Job("a", "bench", "x"), Job("b", "bench", "x"))
    job, _ = pick_next(queue, {}, "gpu", NOW)
    assert job.name == "a"
    job, reasons = pick_next(queue, {"a": {"state": "done"}}, "gpu", NOW)
    assert job.name == "b" and "already done" in reasons[0]


def test_hardware_and_dependencies():
    queue = q(Job("data", "data", "x", hardware="cpu"), Job("train", "bench", "x", after=["data"]))
    job, reasons = pick_next(queue, {}, "gpu", NOW)
    assert job is None
    assert "needs a CPU session" in reasons[0] and "waiting for data" in reasons[1]
    job, _ = pick_next(queue, {}, "cpu", NOW)
    assert job.name == "data"
    job, _ = pick_next(queue, {"data": {"state": "done"}}, "gpu", NOW)
    assert job.name == "train"


def test_running_jobs_are_skipped_until_stale():
    queue = q(Job("a", "bench", "x"), Job("b", "bench", "x"))
    fresh = {"a": {"state": "running", "heartbeat_unix": NOW - 60}}
    assert pick_next(queue, fresh, "gpu", NOW)[0].name == "b"
    stale = {"a": {"state": "running", "heartbeat_unix": NOW - 3600}}
    assert pick_next(queue, stale, "gpu", NOW)[0].name == "a"


def test_paused_resumes_and_failed_retries_until_limit():
    queue = q(Job("a", "bench", "x", max_attempts=2))
    assert pick_next(queue, {"a": {"state": "paused"}}, "gpu", NOW)[0].name == "a"
    assert pick_next(queue, {"a": {"state": "failed", "attempts": 1}}, "gpu", NOW)[0].name == "a"
    job, reasons = pick_next(queue, {"a": {"state": "failed", "attempts": 2}}, "gpu", NOW)
    assert job is None and "failed 2 times" in reasons[0]
    assert pick_next(queue, {"a": {"state": "diverged"}}, "gpu", NOW)[0] is None


def test_skip_set():
    queue = q(Job("a", "bench", "x"), Job("b", "bench", "x"))
    assert pick_next(queue, {}, "gpu", NOW, skip={"a"})[0].name == "b"


def test_load_queue_validates(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "c.yaml").write_text("name: t1\ntrain: {max_steps: 1}\n")
    path = tmp_path / "runs" / "queue.yaml"
    path.write_text("jobs:\n  - {name: t1, kind: train, config: c.yaml}\n")
    assert load_queue(path).jobs[0].name == "t1"
    path.write_text("jobs:\n  - {name: t2, kind: train, config: c.yaml}\n")
    with pytest.raises(ConfigError, match="names must match"):
        load_queue(path)
    path.write_text("jobs:\n  - {name: a, kind: bench, config: c.yaml, after: [zzz]}\n")
    with pytest.raises(ConfigError, match="unknown job"):
        load_queue(path)
    path.write_text("jobs:\n  - {name: a, kind: bench, config: missing.yaml}\n")
    with pytest.raises(ConfigError, match="not found"):
        load_queue(path)
    path.write_text("jobs:\n  - {name: a, kind: bench, config: c.yaml}\n  - {name: a, kind: bench, config: c.yaml}\n")
    with pytest.raises(ConfigError, match="unique"):
        load_queue(path)
