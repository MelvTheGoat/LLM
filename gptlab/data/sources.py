"""Where raw documents come from.

The main source is FineWeb-Edu on the Hugging Face Hub, stored as parquet files.
Inside each file, documents are grouped by crawl (long runs of documents from the
same Common Crawl dump). If we read files front to back, the first shards and the
tokenizer sample would only see a few crawls. So we read "row groups" (blocks of
about 1000 documents) in a seeded random order across all the files we use. Every
shard then holds a mix of crawls.

Local sources (a jsonl file or local parquet files) behave the same way. Tests
and offline runs use them.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceConfig:
    kind: str = "hf_parquet"  # "hf_parquet", "parquet" (local files) or "jsonl"
    repo: str = "HuggingFaceFW/fineweb-edu"
    pattern: str = "sample/10BT/*.parquet"  # files inside the Hub repo
    path: str | None = None  # glob of local files for "parquet" or "jsonl"
    num_files: int | None = None  # use only the first N files (in name order)
    text_field: str = "text"
    jsonl_group_size: int = 16  # documents per group for jsonl files

    def validate(self) -> None:
        if self.kind not in ("hf_parquet", "parquet", "jsonl"):
            raise ValueError(f"unknown source kind {self.kind!r}")
        if self.kind != "hf_parquet" and not self.path:
            raise ValueError("source.path is needed for local sources")


class DocSource:
    """A set of document groups that can be read in any order."""

    def __init__(self, cfg: SourceConfig, work_dir: str | Path, log=print):
        cfg.validate()
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.log = log
        self.files: list[Path] = []
        self.file_names: list[str] = []
        self._groups: list[tuple[int, int]] = []
        self._parquet = {}
        self._jsonl_cache: dict[int, list[str]] = {}

    def prepare(self) -> None:
        """Find (and for the Hub, download) the files, then list their groups."""
        cfg = self.cfg
        if cfg.kind == "hf_parquet":
            from huggingface_hub import HfApi, hf_hub_download

            names = sorted(
                f for f in HfApi().list_repo_files(cfg.repo, repo_type="dataset")
                if fnmatch.fnmatch(f, cfg.pattern)
            )
            if cfg.num_files:
                names = names[: cfg.num_files]
            if not names:
                raise FileNotFoundError(f"no files match {cfg.pattern} in {cfg.repo}")
            for name in names:
                self.log(f"downloading {cfg.repo}/{name}")
                local = hf_hub_download(
                    cfg.repo, name, repo_type="dataset", local_dir=str(self.work_dir / "source")
                )
                self.files.append(Path(local))
            self.file_names = [f"{cfg.repo}/{n}" for n in names]
        else:
            paths = sorted(Path(p) for p in glob.glob(cfg.path))
            if cfg.num_files:
                paths = paths[: cfg.num_files]
            if not paths:
                raise FileNotFoundError(f"no files match {cfg.path}")
            self.files = paths
            self.file_names = [str(p) for p in paths]

        self._groups = []
        for fi, path in enumerate(self.files):
            if cfg.kind == "jsonl":
                n_docs = sum(1 for line in open(path, encoding="utf-8") if line.strip())
                n_groups = (n_docs + cfg.jsonl_group_size - 1) // cfg.jsonl_group_size
            else:
                n_groups = self._parquet_file(fi).metadata.num_row_groups
            self._groups.extend((fi, g) for g in range(n_groups))

    def _parquet_file(self, fi: int):
        import pyarrow.parquet as pq

        if fi not in self._parquet:
            self._parquet[fi] = pq.ParquetFile(self.files[fi])
        return self._parquet[fi]

    def groups_in_order(self, seed: int) -> list[tuple[int, int]]:
        order = list(self._groups)
        random.Random(seed).shuffle(order)
        return order

    def read_group(self, group: tuple[int, int]) -> list[str]:
        fi, g = group
        if self.cfg.kind == "jsonl":
            if fi not in self._jsonl_cache:
                with open(self.files[fi], encoding="utf-8") as f:
                    self._jsonl_cache[fi] = [
                        json.loads(line)[self.cfg.text_field] for line in f if line.strip()
                    ]
            size = self.cfg.jsonl_group_size
            return self._jsonl_cache[fi][g * size : (g + 1) * size]
        table = self._parquet_file(fi).read_row_group(g, columns=[self.cfg.text_field])
        return [t for t in table.column(0).to_pylist() if t is not None]

    def iter_documents(self, seed: int):
        for group in self.groups_in_order(seed):
            yield from self.read_group(group)
