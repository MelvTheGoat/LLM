"""Read and write job logs on the `results` git branch.

The branch holds only logs, never code: results/<job>/status.json plus the
job's log files. Each session only ever writes its own job's folder. To push
safely while another session may also be pushing, every push starts from the
newest remote state: fetch, reset to it, copy our job's files in, commit, push.
If someone else pushed in between, we simply repeat.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


class GitError(RuntimeError):
    pass


def authed_url(repo_url: str, token: str | None) -> str:
    if token and repo_url.startswith("https://"):
        return repo_url.replace("https://", f"https://x-access-token:{token}@", 1)
    return repo_url


class ResultsRepo:
    def __init__(self, repo_url: str, token: str | None, local_dir, branch: str, author_name: str,
                 author_email: str, log=print):
        self.url = authed_url(repo_url, token)
        self.token = token
        self.dir = Path(local_dir)
        self.branch = branch
        self.author = (author_name, author_email)
        self.log = log
        self.lock = threading.Lock()  # the heartbeat thread and the main thread share the repo

    # -- git plumbing ------------------------------------------------------
    def _redact(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    def _git(self, *args, cwd=None, check=True) -> subprocess.CompletedProcess:
        cmd = ["git", *args]
        p = subprocess.run(cmd, cwd=cwd or self.dir, capture_output=True, text=True, timeout=300)
        if check and p.returncode != 0:
            raise GitError(self._redact(f"git {' '.join(args)} failed: {p.stderr.strip()[:1000]}"))
        return p

    def _remote_has_branch(self) -> bool:
        p = self._git("ls-remote", "--heads", self.url, self.branch, cwd=self.dir.parent)
        return bool(p.stdout.strip())

    def sync(self) -> None:
        """Make a local clone of the results branch (creating the branch if needed)."""
        with self.lock:
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            if self._remote_has_branch():
                self._git("clone", "--depth", "1", "--branch", self.branch, self.url, str(self.dir), cwd=self.dir.parent)
            else:
                self.dir.mkdir()
                self._git("init", "-q")
                self._git("checkout", "-q", "--orphan", self.branch)
                self._git("remote", "add", "origin", self.url)
            self._git("config", "user.name", self.author[0])
            self._git("config", "user.email", self.author[1])
            self._git("config", "commit.gpgsign", "false")
            readme = self.dir / "README.md"
            if not readme.exists():
                readme.write_text(
                    "# Results\n\nRun logs pushed by the Kaggle runner. One folder per job under `results/`.\n"
                    "This branch holds logs only; the code is on `main`.\n"
                )
                self._git("add", "README.md")
                self._git("commit", "-q", "-m", "Start results branch")
                self._push_head()

    def _push_head(self) -> bool:
        p = self._git("push", "origin", f"HEAD:refs/heads/{self.branch}", check=False)
        return p.returncode == 0

    def _reset_to_remote(self) -> None:
        self._git("fetch", "-q", "--depth", "1", "origin", self.branch)
        self._git("reset", "-q", "--hard", "FETCH_HEAD")
        self._git("clean", "-q", "-fd")

    # -- reading -----------------------------------------------------------
    def job_dir(self, job: str) -> Path:
        return self.dir / "results" / job

    def read_status(self, job: str) -> dict | None:
        path = self.job_dir(job) / "status.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def statuses(self) -> dict[str, dict]:
        root = self.dir / "results"
        if not root.exists():
            return {}
        out = {}
        for d in root.iterdir():
            st = self.read_status(d.name)
            if st is not None:
                out[d.name] = st
        return out

    def refresh(self) -> dict[str, dict]:
        with self.lock:
            self._reset_to_remote()
            return self.statuses()

    # -- writing -----------------------------------------------------------
    def publish(self, job: str, fill, message: str, attempts: int = 6, check=None) -> bool:
        """Write this job's folder and push it.

        `fill(folder)` writes the job's files into the folder. `check(status)`,
        if given, is called with the latest remote status before writing; if it
        returns False we stop without pushing (used to claim a job safely).
        """
        with self.lock:
            for i in range(attempts):
                self._reset_to_remote()
                if check is not None and not check(self.read_status(job)):
                    return False
                folder = self.job_dir(job)
                folder.mkdir(parents=True, exist_ok=True)
                fill(folder)
                self._git("add", "-A", f"results/{job}")
                if self._git("diff", "--cached", "--quiet", check=False).returncode == 0:
                    return True  # nothing changed
                self._git("commit", "-q", "-m", message)
                if self._push_head():
                    return True
                self.log(f"results push rejected (someone else pushed); retry {i + 1}")
                time.sleep(2 + 2 * i)
            raise GitError(f"could not push results for {job} after {attempts} tries")
