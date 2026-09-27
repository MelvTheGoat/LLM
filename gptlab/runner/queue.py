"""The job queue (runs/queue.yaml) and the rule for picking the next job.

Each job has a unique name. The name is also the folder on the `results`
branch and the folder in the checkpoint repo, so never reuse a name for a
different experiment: give the new one a new name.

Job states (stored in results/<job>/status.json on the `results` branch):
- (none)    never started
- running   a session is working on it (it pushes a heartbeat every few minutes)
- paused    stopped at the session time limit; the next session resumes it
- done      finished
- diverged  training blew up and was stopped (a real result for stability runs)
- failed    crashed; retried until `max_attempts` is used up
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from gptlab.config import ConfigError, apply_overrides, from_dict, read_yaml

KINDS = ("smoke", "data", "train", "bench", "eval")
FINISHED = ("done", "diverged")


@dataclass
class Job:
    name: str
    kind: str
    config: str | None = None
    hardware: str = "gpu"  # "gpu" or "cpu": the kind of Kaggle session that may run it
    after: list[str] = field(default_factory=list)  # jobs that must be finished first
    max_attempts: int = 2
    args: dict = field(default_factory=dict)  # extra settings for some kinds
    notes: str = ""


@dataclass
class Settings:
    repo_url: str = "https://github.com/MelvTheGoat/LLM.git"
    results_branch: str = "results"
    git_author_name: str = "MelvTheGoat"
    git_author_email: str = "110544695+MelvTheGoat@users.noreply.github.com"
    hf_data_repo: str = "gptlab-data"  # "<your hf user>/" is added in front
    hf_ckpt_repo: str = "gptlab-checkpoints"
    hf_private: bool = False
    session_hours: float = 11.0  # stop everything cleanly this long after the notebook started
    stop_margin_minutes: float = 20.0  # training stops this long before that, to save and upload
    min_minutes_to_start: float = 20.0  # do not start a new job with less time left
    heartbeat_minutes: float = 10.0  # push logs and status this often
    stale_minutes: float = 30.0  # a "running" job with no heartbeat for this long is free again
    ckpt_check_seconds: float = 60.0  # how often to look for a new checkpoint to upload
    chain_jobs: bool = True  # after a job ends, start the next one if time allows
    log_tail_mb: float = 5.0  # keep at most this much of each console log in results


@dataclass
class Queue:
    settings: Settings = field(default_factory=Settings)
    jobs: list[Job] = field(default_factory=list)

    def job(self, name: str) -> Job:
        for j in self.jobs:
            if j.name == name:
                return j
        raise KeyError(name)


def load_queue(path: str | Path, repo_root: str | Path | None = None, overrides=None) -> Queue:
    q = from_dict(Queue, apply_overrides(read_yaml(path), overrides or []))
    root = Path(repo_root) if repo_root else Path(path).resolve().parents[1]
    names = [j.name for j in q.jobs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ConfigError(f"job names must be unique: {sorted(dupes)}")
    for j in q.jobs:
        if not j.name or "/" in j.name or " " in j.name:
            raise ConfigError(f"bad job name {j.name!r}")
        if j.kind not in KINDS:
            raise ConfigError(f"job {j.name}: kind must be one of {KINDS}")
        if j.hardware not in ("gpu", "cpu"):
            raise ConfigError(f"job {j.name}: hardware must be gpu or cpu")
        for dep in j.after:
            if dep not in names:
                raise ConfigError(f"job {j.name} waits for unknown job {dep}")
        if j.kind in ("data", "train", "bench", "smoke") and not j.config:
            raise ConfigError(f"job {j.name}: needs a config")
        if j.config and not (root / j.config).exists():
            raise ConfigError(f"job {j.name}: config file {j.config} not found")
        if j.kind == "train":
            from gptlab.config import load_config

            cfg = load_config(root / j.config)
            if cfg.name != j.name:
                raise ConfigError(f"job {j.name}: its config is named {cfg.name!r}; the names must match")
        if j.kind == "eval" and "run" not in j.args:
            raise ConfigError(f"job {j.name}: eval jobs need args.run (the run to evaluate)")
    return q


def is_stale(status: dict, settings: Settings, now: float) -> bool:
    return now - status.get("heartbeat_unix", 0) > settings.stale_minutes * 60


def job_blocker(job: Job, statuses: dict[str, dict], hardware: str, settings: Settings, now: float) -> str | None:
    """Why this job cannot start now, or None if it can."""
    st = statuses.get(job.name) or {}
    state = st.get("state")
    if state in FINISHED:
        return f"already {state}"
    if state == "running" and not is_stale(st, settings, now):
        age = (now - st.get("heartbeat_unix", now)) / 60
        return f"running in another session (last heartbeat {age:.0f} min ago)"
    if state == "failed" and st.get("attempts", 0) >= job.max_attempts:
        return f"failed {st.get('attempts')} times; fix it, then rename the job or raise max_attempts"
    for dep in job.after:
        dep_state = (statuses.get(dep) or {}).get("state")
        if dep_state not in FINISHED:
            return f"waiting for {dep} ({dep_state or 'not started'})"
    if job.hardware != hardware:
        return f"needs a {job.hardware.upper()} session (this one is {hardware.upper()})"
    return None


def pick_next(queue: Queue, statuses: dict[str, dict], hardware: str, now: float | None = None,
              skip: set[str] | None = None) -> tuple[Job | None, list[str]]:
    """First job in queue order that can start now, plus the reasons others cannot."""
    now = time.time() if now is None else now
    reasons = []
    for job in queue.jobs:
        if skip and job.name in skip:
            reasons.append(f"{job.name}: already handled in this session")
            continue
        why = job_blocker(job, statuses, hardware, queue.settings, now)
        if why is None:
            return job, reasons
        reasons.append(f"{job.name}: {why}")
    return None, reasons
