"""The Kaggle job runner.

The Kaggle notebook clones the repo and runs `python -m gptlab.runner`. Then:

1. Clone the `results` branch and read every job's status.
2. Pick the first job in runs/queue.yaml that can run now (see queue.py) and
   claim it (status "running", pushed to `results`).
3. Download what it needs from the Hugging Face Hub (token data, the latest
   checkpoint) and run it. While it runs, a background thread uploads each new
   checkpoint and pushes logs and a heartbeat every few minutes.
4. Training stops by itself a little before the session limit (for example
   11 hours after the notebook started), saves, and is marked "paused". The
   next session resumes it.
5. Whatever happens (success, crash, time limit), the logs and status are
   pushed at the end. Then the next job starts if there is time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from gptlab.checkpoint import UPLOAD_MARKER, latest_checkpoint, list_checkpoints
from gptlab.runner.queue import Job, Queue, Settings, load_queue, pick_next
from gptlab.runner.results import ResultsRepo

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files copied from a training run folder to results/<job>/.
TRAIN_LOGS = ["config.yaml", "env.json", "run_state.json", "metrics.jsonl", "eval.jsonl", "debug.jsonl",
              "events.jsonl", "final_eval.json", "samples.txt", "train.log"]


def say(msg: str) -> None:
    print(f"[runner {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit(root: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def copy_tail(src: Path, dst: Path, max_mb: float) -> None:
    """Copy a (log) file, keeping only its last `max_mb` megabytes."""
    size = src.stat().st_size
    limit = int(max_mb * 2**20)
    if size <= limit:
        shutil.copy2(src, dst)
        return
    with open(src, "rb") as f:
        f.seek(size - limit)
        data = f.read()
    dst.write_bytes(b"[... earlier output cut ...]\n" + data[data.find(b"\n") + 1:])


def last_jsonl_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 8192))
        lines = f.read().decode("utf-8", "replace").strip().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def run_logged(cmd: list[str], log_path: Path, env: dict, cwd: Path = REPO_ROOT, timeout: float | None = None) -> int:
    """Run a command, showing its output live and saving it to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    say("$ " + " ".join(cmd))
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace")
        start = time.time()
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            if timeout and time.time() - start > timeout:
                proc.kill()
                break
        return proc.wait()


# ---------------------------------------------------------------------------
# Session context
# ---------------------------------------------------------------------------


@dataclass
class Context:
    queue: Queue
    results: ResultsRepo
    data_store: object
    ckpt_store: object
    work: Path
    hardware: str  # "gpu" or "cpu"
    n_gpus: int
    session_id: str
    session_end: float
    repo_root: Path = REPO_ROOT
    extra_env: dict = field(default_factory=dict)
    session_info: dict = field(default_factory=dict)

    @property
    def settings(self) -> Settings:
        return self.queue.settings

    def minutes_left(self) -> float:
        return (self.session_end - time.time()) / 60

    def child_env(self, keep_hf_token: bool = False) -> dict:
        env = dict(os.environ)
        env.pop("GH_TOKEN", None)  # training code never needs the GitHub token
        if not keep_hf_token:
            env.pop("HF_TOKEN", None)
        env.update(self.extra_env)
        env["PYTHONPATH"] = str(self.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["GPTLAB_DATA_ROOT"] = str(self.work / "data")
        commit = self.session_info.get("git_commit")
        if commit:
            env["GPTLAB_GIT_COMMIT"] = commit
        return env

    def launcher(self, module_args: list[str], n_procs: int | None = None) -> list[str]:
        """Command to run a module on all GPUs (torchrun) or on the CPU."""
        if self.hardware == "gpu":
            n = n_procs or self.n_gpus
            return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}",
                    *module_args]
        return [sys.executable, *module_args, "--device", "cpu"]


# ---------------------------------------------------------------------------
# One job: status, heartbeat, log publishing, checkpoint uploads
# ---------------------------------------------------------------------------


class CheckpointSync:
    """Uploads the newest local checkpoint to <run>/latest on the Hub."""

    def __init__(self, store, run_name: str, ckpt_root: Path, uploaded: str | None = None):
        self.store = store
        self.run_name = run_name
        self.ckpt_root = Path(ckpt_root)
        self.uploaded = uploaded
        self.lock = threading.Lock()

    def maybe_upload(self) -> bool:
        with self.lock:
            if (self.ckpt_root.parent / "final" / "meta.json").exists():
                return False  # training finished: final weights replace the full checkpoint
            p = latest_checkpoint(self.ckpt_root)
            if p is None or p.name == self.uploaded:
                return False
            marker = p / UPLOAD_MARKER
            marker.touch()  # stops the training loop from deleting it meanwhile
            try:
                (p / "CHECKPOINT").write_text(p.name + "\n")
                t0 = time.time()
                self.store.upload_folder(p, f"{self.run_name}/latest", message=f"{self.run_name}: {p.name}")
                self.uploaded = p.name
                say(f"uploaded checkpoint {self.run_name}/{p.name} in {time.time() - t0:.0f}s")
                try:
                    self.store.squash()  # drop older checkpoint versions from the repo history
                except Exception as e:
                    say(f"warning: could not squash checkpoint repo history: {e!r}")
            finally:
                marker.unlink(missing_ok=True)
            return True


def restore_checkpoint(store, run_name: str, run_dir: Path) -> str | None:
    """Bring <run>/latest from the Hub into run_dir/ckpt. Returns the checkpoint name."""
    ckpt_root = run_dir / "ckpt"
    local = latest_checkpoint(ckpt_root)
    if local is not None:
        return local.name
    tmp = run_dir / "ckpt_download"
    shutil.rmtree(tmp, ignore_errors=True)
    if not store.download_folder(f"{run_name}/latest", tmp):
        return None
    name = (tmp / "CHECKPOINT").read_text().strip()
    (tmp / "CHECKPOINT").unlink()
    ckpt_root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ckpt_root / name, ignore_errors=True)
    tmp.rename(ckpt_root / name)
    (ckpt_root / "LATEST").write_text(name + "\n")
    say(f"restored checkpoint {name} of {run_name} from {store.describe()}")
    return name


def ensure_data(ctx: Context, name: str, max_train_shards: int | None = None, train: bool = True) -> Path:
    """Make sure a token dataset is on local disk, downloading from the Hub if needed."""
    local = ctx.work / "data" / name
    tmp = ctx.work / "data" / (name + ".manifest")
    shutil.rmtree(tmp, ignore_errors=True)
    if not ctx.data_store.download_folder(name, tmp, patterns=["manifest.json"]):
        if (local / "manifest.json").exists():
            return local  # offline use with a local copy
        raise FileNotFoundError(f"dataset {name!r} not found in {ctx.data_store.describe()}; run its data job first")
    manifest = json.loads((tmp / "manifest.json").read_text())
    shutil.rmtree(tmp, ignore_errors=True)
    shards = manifest["splits"]["train"]["shards"] if train else []
    if max_train_shards:
        shards = shards[:max_train_shards]
    wanted = ["manifest.json", "tokenizer.json"] + manifest["splits"]["val"]["shards"] + shards
    missing = [f for f in wanted if not (local / f).exists()]
    if missing:
        t0 = time.time()
        ctx.data_store.download_folder(name, local, patterns=missing)
        say(f"downloaded {len(missing)} data files for {name} in {time.time() - t0:.0f}s")
    return local


class JobRun:
    def __init__(self, ctx: Context, job: Job, status: dict):
        self.ctx = ctx
        self.job = job
        self.status = status
        self.run_dir = ctx.work / "runs" / job.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.files: list[tuple[Path, str]] = []  # (local file, name in results folder)
        self.ckpt_sync: CheckpointSync | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def collect(self, src: Path, name: str | None = None) -> None:
        self.files.append((Path(src), name or Path(src).name))

    def _fill(self, folder: Path) -> None:
        for src, name in self.files:
            if src.exists() and src.is_file():
                dst = folder / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith(".log"):
                    copy_tail(src, dst, self.ctx.settings.log_tail_mb)
                else:
                    shutil.copy2(src, dst)
        (folder / "session.json").write_text(json.dumps(self.ctx.session_info, indent=2))
        (folder / "status.json").write_text(json.dumps(self.status, indent=2))

    def _update_progress(self) -> None:
        rs = self.run_dir / "run_state.json"
        if rs.exists():
            try:
                self.status["run_state"] = json.loads(rs.read_text())
            except json.JSONDecodeError:
                pass
        last = last_jsonl_record(self.run_dir / "metrics.jsonl")
        if last:
            self.status["last_metrics"] = {k: last.get(k) for k in ["step", "tokens", "loss", "tok_per_s", "mfu"]}

    def publish(self, message: str | None = None) -> None:
        self.status["heartbeat_unix"] = time.time()
        self._update_progress()
        msg = message or f"{self.job.name}: {self.status.get('state')} (heartbeat)"
        try:
            self.ctx.results.publish(self.job.name, self._fill, msg)
        except Exception as e:  # never let a push problem kill a training run
            say(f"warning: could not push results: {e!r}")

    def _loop(self) -> None:
        s = self.ctx.settings
        last_beat = time.time()
        while not self._stop.wait(timeout=min(s.ckpt_check_seconds, 30)):
            uploaded = False
            if self.ckpt_sync is not None:
                try:
                    uploaded = self.ckpt_sync.maybe_upload()
                except Exception as e:
                    say(f"warning: checkpoint upload failed, will retry: {e!r}")
            if uploaded or time.time() - last_beat >= s.heartbeat_minutes * 60:
                self.publish()
                last_beat = time.time()

    def start_background(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


# ---------------------------------------------------------------------------
# Job kinds
# ---------------------------------------------------------------------------


def _train_once(ctx: Context, jr: JobRun, config_path: Path, run_name: str, run_dir: Path, data_dir: Path,
                stop_after: int | None = None, overrides: list[str] | None = None) -> tuple[str, str]:
    """Run (or resume) one training run with checkpoint syncing. Returns (state, message)."""
    restored = restore_checkpoint(ctx.ckpt_store, run_name, run_dir)
    sync = CheckpointSync(ctx.ckpt_store, run_name, run_dir / "ckpt", uploaded=restored)
    jr.ckpt_sync = sync
    deadline = ctx.session_end - ctx.settings.stop_margin_minutes * 60
    args = ["-m", "gptlab.train", "--config", str(config_path), "--out-dir", str(run_dir), "--data-dir", str(data_dir),
            "--deadline", f"{deadline:.0f}", "--hellaswag-path", str(ctx.work / "data" / "hellaswag_val.jsonl")]
    for o in overrides or []:
        args += ["--set", o]
    if stop_after:
        args += ["--stop-after-steps", str(stop_after)]
    code = run_logged(ctx.launcher(args), run_dir / "train.log", ctx.child_env())
    jr.ckpt_sync = None
    try:
        sync.maybe_upload()  # the newest checkpoint, if the background thread has not sent it yet
    except Exception as e:
        say(f"warning: final checkpoint upload failed: {e!r}")
    try:
        run_state = json.loads((run_dir / "run_state.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        run_state = {"state": "failed", "reason": "no run_state.json (crashed at start?)"}
    state = run_state.get("state", "failed")
    if code != 0 and state not in ("paused", "done", "diverged"):
        state = "failed"
    if state == "running":  # killed without a clean exit
        state = "failed"
    if state == "done" and (run_dir / "final").exists():
        for name in ["final_eval.json", "samples.txt"]:
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, run_dir / "final" / name)
        ctx.ckpt_store.upload_folder(run_dir / "final", f"{run_name}/final", message=f"{run_name}: final weights")
        ctx.ckpt_store.delete_folder(f"{run_name}/latest")  # optimizer state is no longer needed
        try:
            ctx.ckpt_store.squash()
        except Exception as e:
            say(f"warning: could not squash checkpoint repo history: {e!r}")
    return state, f"exit code {code}; {run_state.get('reason', '')}".strip()


def run_train(ctx: Context, jr: JobRun) -> tuple[str, str]:
    from gptlab.config import load_config

    cfg_path = ctx.repo_root / jr.job.config
    cfg = load_config(cfg_path)
    data_dir = ensure_data(ctx, cfg.data.name, cfg.data.max_train_shards)
    for name in TRAIN_LOGS:
        jr.collect(jr.run_dir / name)
    stop_after = jr.job.args.get("stop_after_steps")
    return _train_once(ctx, jr, cfg_path, jr.job.name, jr.run_dir, data_dir, stop_after)


def _dataset_on_hub(ctx: Context, name: str) -> bool:
    return f"{name}/manifest.json" in ctx.data_store.list_files(name)


def _prepare_and_upload(ctx: Context, config_path: Path, log_path: Path, replace: bool = False) -> dict:
    from gptlab.data.prepare import load_data_config

    cfg = load_data_config(config_path)
    out = ctx.work / "data" / cfg.name
    scratch = ctx.work / "data_work" / cfg.name
    shutil.rmtree(out, ignore_errors=True)
    args = ["-m", "gptlab.data.prepare", "--config", str(config_path), "--out", str(out), "--work", str(scratch)]
    code = run_logged([sys.executable, *args], log_path, ctx.child_env(keep_hf_token=True))
    if code != 0:
        raise RuntimeError(f"data preparation failed with exit code {code}")
    manifest = json.loads((out / "manifest.json").read_text())
    (out / "README.md").write_text(
        f"# {cfg.name}\n\nToken shards (uint16) made from {cfg.source.repo} ({cfg.source.pattern}) by gptlab's data "
        "pipeline. See manifest.json for counts, cleaning statistics and the tokenizer comparison.\n\n"
        "Source data license: ODC-By 1.0 (FineWeb-Edu).\n"
    )
    t0 = time.time()
    ctx.data_store.upload_folder(out, cfg.name, message=f"dataset {cfg.name}")
    say(f"uploaded dataset {cfg.name} in {time.time() - t0:.0f}s")
    shutil.rmtree(scratch, ignore_errors=True)
    return manifest


def run_data(ctx: Context, jr: JobRun) -> tuple[str, str]:
    from gptlab.data.prepare import load_data_config

    cfg_path = ctx.repo_root / jr.job.config
    name = load_data_config(cfg_path).name
    jr.collect(jr.run_dir / "data.log")
    jr.collect(jr.run_dir / "manifest.json")
    if _dataset_on_hub(ctx, name):
        return "done", f"dataset {name} is already on the Hub; nothing to do"
    manifest = _prepare_and_upload(ctx, cfg_path, jr.run_dir / "data.log")
    (jr.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return "done", f"{manifest['splits']['train']['tokens']:,} train tokens"


def run_bench(ctx: Context, jr: JobRun) -> tuple[str, str]:
    jr.collect(jr.run_dir / "bench.json")
    jr.collect(jr.run_dir / "bench.log")
    args = ["-m", "gptlab.bench", "--config", str(ctx.repo_root / jr.job.config), "--out", str(jr.run_dir)]
    code = run_logged(ctx.launcher(args), jr.run_dir / "bench.log", ctx.child_env())
    return ("done" if code == 0 else "failed"), f"exit code {code}"


def run_eval(ctx: Context, jr: JobRun) -> tuple[str, str]:
    from gptlab.config import load_config

    run = jr.job.args["run"]
    run_dir = ctx.work / "runs" / run
    if not ctx.ckpt_store.download_folder(f"{run}/final", run_dir / "final"):
        return "failed", f"no final weights for {run} on the Hub"
    cfg = load_config(run_dir / "final" / "config.yaml")
    data_dir = ensure_data(ctx, cfg.data.name, train=False)
    jr.collect(run_dir / "final_eval.json")
    jr.collect(run_dir / "samples.txt")
    jr.collect(jr.run_dir / "eval.log")
    args = ["-m", "gptlab.evaluate", "--run-dir", str(run_dir), "--data-dir", str(data_dir),
            "--hellaswag-path", str(ctx.work / "data" / "hellaswag_val.jsonl")]
    for k, v in jr.job.args.get("set", {}).items():
        args += ["--set", f"{k}={v}"]
    code = run_logged(ctx.launcher(args), jr.run_dir / "eval.log", ctx.child_env())
    return ("done" if code == 0 else "failed"), f"exit code {code}"


def _losses(run_dir: Path) -> dict[int, float]:
    from gptlab.metrics import read_jsonl

    return {r["step"]: r["loss"] for r in read_jsonl(run_dir / "metrics.jsonl")}


def run_smoke(ctx: Context, jr: JobRun) -> tuple[str, str]:
    """End-to-end check of the whole pipeline before spending real GPU hours.

    1. data: build a tiny dataset from FineWeb-Edu, upload it, delete it
       locally, and download it again.
    2. resume: train a small model three times on 2 GPUs: twice straight
       through, and once stopped halfway, resumed from the checkpoint that was
       uploaded to (and downloaded back from) the Hub. On GPUs a few kernels
       are not bit-exact from run to run, so we compare the resume difference
       with the difference between the two straight runs.
    3. bench: throughput of the candidate model sizes.
    """
    from gptlab.config import read_yaml

    spec = read_yaml(ctx.repo_root / jr.job.config)
    report: dict = {"steps": {}}
    root = jr.run_dir

    def step(name, fn):
        t0 = time.time()
        say(f"smoke step: {name}")
        try:
            report["steps"][name] = {"ok": True, **(fn() or {})}
        except Exception as e:
            report["steps"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                     "traceback": traceback.format_exc()[-3000:]}
            say(f"smoke step {name} FAILED: {e!r}")
        report["steps"][name]["seconds"] = round(time.time() - t0, 1)
        (root / "smoke_report.json").write_text(json.dumps(report, indent=2))
        jr.publish(f"{jr.job.name}: step {name} finished")

    jr.collect(root / "smoke_report.json")
    data_cfg = ctx.repo_root / spec["data_config"]

    def data_step():
        from gptlab.data.prepare import load_data_config

        jr.collect(root / "data.log", "data/data.log")
        jr.collect(root / "manifest.json", "data/manifest.json")
        manifest = _prepare_and_upload(ctx, data_cfg, root / "data.log")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        name = load_data_config(data_cfg).name
        shutil.rmtree(ctx.work / "data" / name)
        local = ensure_data(ctx, name)
        n = len(list(local.glob("*.bin")))
        expected = len(manifest["splits"]["train"]["shards"]) + len(manifest["splits"]["val"]["shards"])
        if n != expected:
            raise RuntimeError(f"downloaded {n} shards, expected {expected}")
        return {"train_tokens": manifest["splits"]["train"]["tokens"], "shards": n}

    def resume_step():
        from gptlab.config import load_config

        cfg_path = ctx.repo_root / spec["train_config"]
        cfg = load_config(cfg_path)
        data_dir = ensure_data(ctx, cfg.data.name)
        runs = {}
        for tag in ["straight", "straight2", "resume"]:
            rd = root / tag
            shutil.rmtree(rd, ignore_errors=True)
            for name in TRAIN_LOGS:
                jr.collect(rd / name, f"{tag}/{name}")
            runs[tag] = rd
        store_name = f"{jr.job.name}/resume"
        for tag in runs:  # start clean, even if an earlier attempt left checkpoints behind
            ctx.ckpt_store.delete_folder(f"{jr.job.name}/{tag}/latest")
            ctx.ckpt_store.delete_folder(f"{jr.job.name}/{tag}/final")
        for tag in ["straight", "straight2"]:
            state, msg = _train_once(ctx, jr, cfg_path, f"{jr.job.name}/{tag}", runs[tag], data_dir)
            if state != "done":
                raise RuntimeError(f"{tag} run ended as {state}: {msg}")
        half = int(spec["stop_after_steps"])
        state, msg = _train_once(ctx, jr, cfg_path, store_name, runs["resume"], data_dir, stop_after=half)
        if state != "paused":
            raise RuntimeError(f"interrupted run ended as {state}: {msg}")
        shutil.rmtree(runs["resume"] / "ckpt")  # force the resume to come from the Hub copy
        state, msg = _train_once(ctx, jr, cfg_path, store_name, runs["resume"], data_dir)
        if state != "done":
            raise RuntimeError(f"resumed run ended as {state}: {msg}")
        a, b, r = _losses(runs["straight"]), _losses(runs["straight2"]), _losses(runs["resume"])
        steps = sorted(a)
        if sorted(r) != steps:
            raise RuntimeError("resumed run logged different steps")
        after = [s for s in steps if s > half]
        return {
            "steps": len(steps), "resumed_at": half,
            "max_diff_straight_vs_straight2": max(abs(a[s] - b[s]) for s in steps),
            "max_diff_straight_vs_resumed": max(abs(a[s] - r[s]) for s in steps),
            "max_diff_after_resume": max(abs(a[s] - r[s]) for s in after),
            "identical_after_resume": all(a[s] == r[s] for s in after),
            "final_loss": a[steps[-1]],
        }

    def bench_step():
        jr.collect(root / "bench" / "bench.json", "bench/bench.json")
        jr.collect(root / "bench" / "bench.log", "bench/bench.log")
        args = ["-m", "gptlab.bench", "--config", str(ctx.repo_root / spec["bench_config"]), "--out",
                str(root / "bench")]
        code = run_logged(ctx.launcher(args), root / "bench" / "bench.log", ctx.child_env())
        if code != 0:
            raise RuntimeError(f"bench exit code {code}")
        return {}

    step("data", data_step)
    step("resume", resume_step)
    step("bench", bench_step)
    ok = all(s["ok"] for s in report["steps"].values())
    bad = [k for k, s in report["steps"].items() if not s["ok"]]
    return ("done" if ok else "failed"), ("all smoke steps passed" if ok else f"failed steps: {bad}")


HANDLERS = {"train": run_train, "data": run_data, "bench": run_bench, "eval": run_eval, "smoke": run_smoke}


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


def claim(ctx: Context, job: Job) -> dict | None:
    """Mark the job as running in this session. None if another session got it first."""
    from gptlab.runner.queue import job_blocker

    status: dict = {}

    def check(old: dict | None) -> bool:
        statuses = {job.name: old} if old else {}
        # Dependencies were checked by pick_next moments ago; recheck this job only.
        why = job_blocker(Job(job.name, job.kind, job.config, job.hardware, [], job.max_attempts), statuses,
                          ctx.hardware, ctx.settings, time.time())
        if why is not None:
            say(f"cannot claim {job.name}: {why}")
            return False
        prev = old or {}
        status.clear()
        status.update(prev)
        new_attempt = prev.get("state") in (None, "failed")
        status.update({
            "job": job.name, "kind": job.kind, "config": job.config, "state": "running",
            "session": ctx.session_id, "hardware": ctx.session_info.get("gpu", ctx.hardware),
            "git_commit": ctx.session_info.get("git_commit"), "heartbeat_unix": time.time(),
            "attempts": prev.get("attempts", 0) + (1 if new_attempt else 0),
            "sessions": prev.get("sessions", 0) + 1, "message": "",
        })
        status.setdefault("first_started_unix", time.time())
        status["session_started_unix"] = time.time()
        return True

    def fill(folder: Path) -> None:
        (folder / "status.json").write_text(json.dumps(status, indent=2))
        (folder / "session.json").write_text(json.dumps(ctx.session_info, indent=2))

    ok = ctx.results.publish(job.name, fill, f"{job.name}: start (session {ctx.session_id})", check=check)
    return dict(status) if ok else None


def execute(ctx: Context, job: Job, status: dict) -> str:
    jr = JobRun(ctx, job, status)
    jr.start_background()
    try:
        state, message = HANDLERS[job.kind](ctx, jr)
    except Exception as e:
        state, message = "failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
        say(f"job {job.name} crashed: {e!r}")
    finally:
        jr.stop_background()
    jr.status.update({"state": state, "message": message[-4000:], "session_ended_unix": time.time()})
    jr.publish(f"{job.name}: {state}")
    say(f"job {job.name} finished as {state}: {message.splitlines()[0] if message else ''}")
    return state


def check_ddp(ctx: Context) -> None:
    """Make sure multi-GPU communication works; pick safe NCCL settings if not."""
    if ctx.hardware != "gpu" or ctx.n_gpus < 2:
        return
    log = ctx.work / "ddp_check.log"
    for extra in [{}, {"NCCL_P2P_DISABLE": "1"}]:
        ctx.extra_env = dict(extra)
        cmd = ctx.launcher(["-m", "gptlab.runner.ddp_check"])
        try:
            p = subprocess.run(cmd, cwd=ctx.repo_root, env=ctx.child_env(), capture_output=True, text=True,
                               timeout=240)
            for line in p.stdout.splitlines()[::-1]:
                if line.startswith("{"):
                    result = json.loads(line)
                    result["nccl_env"] = extra
                    ctx.session_info["ddp_check"] = result
                    say(f"DDP check passed with {extra or 'default settings'}: {result}")
                    return
            log.write_text(p.stdout + p.stderr)
        except subprocess.TimeoutExpired:
            say(f"DDP check timed out with {extra or 'default settings'}")
    say("warning: multi-GPU communication does not work here; using 1 GPU")
    ctx.extra_env = {}
    ctx.n_gpus = 1
    ctx.session_info["ddp_check"] = {"ddp_ok": False}


def session_info(work: Path, hardware: str, n_gpus: int, session_id: str) -> dict:
    import platform

    import torch

    info = {"session": session_id, "started_unix": time.time(), "hardware": hardware, "n_gpus": n_gpus,
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "git_commit": git_commit(), "cpu_count": os.cpu_count()}
    if n_gpus:
        info["gpu"] = torch.cuda.get_device_name(0)
    for path in [work, Path("/kaggle/working")]:
        if path.exists():
            du = shutil.disk_usage(path)
            info[f"disk_free_gb:{path}"] = round(du.free / 2**30, 1)
    try:
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        pass
    return info


def run_session(ctx: Context, max_jobs: int | None = None) -> list[tuple[str, str]]:
    say(f"session {ctx.session_id}: {ctx.hardware.upper()} x{ctx.n_gpus}, {ctx.minutes_left():.0f} min left")
    say(f"stores: data={ctx.data_store.describe()} checkpoints={ctx.ckpt_store.describe()}")
    check_ddp(ctx)
    handled: set[str] = set()
    outcomes = []
    while max_jobs is None or len(outcomes) < max_jobs:
        statuses = ctx.results.refresh()
        job, reasons = pick_next(ctx.queue, statuses, ctx.hardware, skip=handled)
        if job is None:
            say("no job can run in this session:")
            for r in reasons:
                say(f"  - {r}")
            break
        if ctx.minutes_left() < ctx.settings.min_minutes_to_start:
            say(f"only {ctx.minutes_left():.0f} min left; not starting {job.name}")
            break
        handled.add(job.name)
        status = claim(ctx, job)
        if status is None:
            continue
        say(f"starting job {job.name} ({job.kind}), attempt {status['attempts']}, session {status['sessions']}")
        outcomes.append((job.name, execute(ctx, job, status)))
        if not ctx.settings.chain_jobs:
            break
    say("session summary: " + (", ".join(f"{n}={s}" for n, s in outcomes) or "nothing ran"))
    return outcomes


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run the next jobs from runs/queue.yaml (used by the Kaggle notebook).")
    p.add_argument("--queue", default=str(REPO_ROOT / "runs" / "queue.yaml"))
    p.add_argument("--work", default=os.environ.get("GPTLAB_WORK", "/tmp/gptlab-work"))
    p.add_argument("--session-start", type=float, default=float(os.environ.get("SESSION_START", time.time())))
    p.add_argument("--hardware", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--local-store", default=None, help="use a local folder instead of the Hugging Face Hub")
    p.add_argument("--results-url", default=None, help="git remote for results (default: settings.repo_url)")
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--set", action="append", default=[], help="override queue settings, e.g. settings.session_hours=2")
    args = p.parse_args(argv)

    import torch

    queue = load_queue(args.queue, REPO_ROOT, args.set)
    s = queue.settings
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    hardware = args.hardware if args.hardware != "auto" else ("gpu" if n_gpus > 0 else "cpu")
    if hardware == "cpu":
        n_gpus = 0
    session_id = uuid.uuid4().hex[:8]

    gh_token = os.environ.get("GH_TOKEN")
    if not args.results_url and not gh_token:
        raise SystemExit("GH_TOKEN is not set: add it as a Kaggle secret (see RUNNING.md)")
    results = ResultsRepo(args.results_url or s.repo_url, None if args.results_url else gh_token,
                          work / "results_repo", s.results_branch, s.git_author_name, s.git_author_email, log=say)
    if args.local_store:
        from gptlab.hub import LocalStore

        data_store = LocalStore(Path(args.local_store) / "data")
        ckpt_store = LocalStore(Path(args.local_store) / "checkpoints")
    else:
        from gptlab.hub import HubStore, hub_username

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise SystemExit("HF_TOKEN is not set: add it as a Kaggle secret (see RUNNING.md)")
        user = hub_username(hf_token)

        def full(name):
            return name if "/" in name else f"{user}/{name}"

        data_store = HubStore(full(s.hf_data_repo), "dataset", hf_token, s.hf_private, log=say)
        ckpt_store = HubStore(full(s.hf_ckpt_repo), "model", hf_token, s.hf_private, log=say)

    ctx = Context(queue=queue, results=results, data_store=data_store, ckpt_store=ckpt_store, work=work,
                  hardware=hardware, n_gpus=n_gpus, session_id=session_id,
                  session_end=args.session_start + s.session_hours * 3600)
    ctx.session_info = session_info(work, hardware, n_gpus, session_id)
    results.sync()
    run_session(ctx, args.max_jobs)


if __name__ == "__main__":
    main()
