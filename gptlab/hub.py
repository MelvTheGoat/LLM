"""Storage for token data and checkpoints on the Hugging Face Hub.

Free Hugging Face accounts get best-effort (in practice very large) storage
for public repos and 100 GB for private repos. Every version of a file stays in
the repo history and counts as storage, so overwriting a 1 GB checkpoint every
30 minutes would pile up fast. To stay small we:

- keep only `<run>/latest/` (the newest full checkpoint, replaced each time)
  and `<run>/final/` (weights only, once the run is done), and delete `latest`
  when the run finishes;
- "super-squash" the checkpoint repo after uploads. This rewrites the history
  into a single commit, so old checkpoint versions are really deleted.

`LocalStore` has the same interface backed by a local folder. Tests and offline
runs use it.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def _retry(fn, attempts: int = 4, what: str = "hub call", log=print):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network errors, server busy, ...
            if i == attempts - 1:
                raise
            wait = 5 * 2**i
            log(f"{what} failed ({type(e).__name__}: {str(e)[:200]}); retrying in {wait}s")
            time.sleep(wait)


class LocalStore:
    """A folder that acts like a Hub repo."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.squash_count = 0

    def upload_folder(self, local_dir, remote_dir: str, message: str = "") -> None:
        dst = self.root / remote_dir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(local_dir, dst, ignore=shutil.ignore_patterns(".uploading"))

    def download_folder(self, remote_dir: str, local_dir, patterns: list[str] | None = None) -> bool:
        src = self.root / remote_dir
        if not src.exists():
            return False
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(src)
                if patterns and not any(rel.match(p) for p in patterns):
                    continue
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, Path(local_dir) / rel)
        return True

    def list_files(self, remote_dir: str) -> list[str]:
        src = self.root / remote_dir
        if not src.exists():
            return []
        return sorted(str(f.relative_to(self.root)) for f in src.rglob("*") if f.is_file())

    def delete_folder(self, remote_dir: str) -> None:
        shutil.rmtree(self.root / remote_dir, ignore_errors=True)

    def squash(self) -> None:
        self.squash_count += 1

    def describe(self) -> str:
        return f"local:{self.root}"


class HubStore:
    """One Hugging Face repo (a "model" repo for checkpoints, "dataset" for data)."""

    def __init__(self, repo_id: str, repo_type: str, token: str | None, private: bool = False, log=print):
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.log = log
        _retry(lambda: self.api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True),
               what="create_repo", log=log)

    def upload_folder(self, local_dir, remote_dir: str, message: str = "") -> None:
        """Upload a folder, replacing whatever was at `remote_dir`, in one commit."""
        _retry(lambda: self.api.upload_folder(
            folder_path=str(local_dir), path_in_repo=remote_dir, repo_id=self.repo_id, repo_type=self.repo_type,
            commit_message=message or f"upload {remote_dir}", delete_patterns="*", ignore_patterns=[".uploading"],
        ), what=f"upload {remote_dir}", log=self.log)

    def download_folder(self, remote_dir: str, local_dir, patterns: list[str] | None = None) -> bool:
        from huggingface_hub import snapshot_download

        files = self.list_files(remote_dir)
        if not files:
            return False
        allow = [f"{remote_dir}/{p}" for p in patterns] if patterns else [f"{remote_dir}/*"]
        staging = Path(local_dir).parent / (Path(local_dir).name + ".download")
        _retry(lambda: snapshot_download(
            self.repo_id, repo_type=self.repo_type, allow_patterns=allow, local_dir=str(staging), max_workers=8,
        ), what=f"download {remote_dir}", log=self.log)
        src = staging / remote_dir
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                dst = Path(local_dir) / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), dst)
        shutil.rmtree(staging, ignore_errors=True)
        return True

    def list_files(self, remote_dir: str) -> list[str]:
        prefix = remote_dir.rstrip("/") + "/"
        files = _retry(lambda: self.api.list_repo_files(self.repo_id, repo_type=self.repo_type),
                       what="list files", log=self.log)
        return sorted(f for f in files if f.startswith(prefix))

    def delete_folder(self, remote_dir: str) -> None:
        if not self.list_files(remote_dir):
            return
        _retry(lambda: self.api.delete_folder(remote_dir, repo_id=self.repo_id, repo_type=self.repo_type,
                                              commit_message=f"delete {remote_dir}"),
               what=f"delete {remote_dir}", log=self.log)

    def squash(self) -> None:
        """Squash the repo history into one commit so old file versions are freed."""
        _retry(lambda: self.api.super_squash_history(self.repo_id, repo_type=self.repo_type,
                                                     commit_message="squash history"),
               what="squash history", log=self.log)

    def describe(self) -> str:
        return f"hf:{self.repo_type}/{self.repo_id}"


def hub_username(token: str) -> str:
    from huggingface_hub import HfApi

    return _retry(lambda: HfApi(token=token).whoami()["name"], what="whoami")
