"""Basic text cleaning, exact de-duplication and the train/validation split.

FineWeb-Edu is already heavily filtered (language ID, quality classifier,
MinHash near-duplicate removal inside each crawl). So we only do light, cheap
checks here, and we count every document we drop and why:

- normalize: Unicode NFC form, "\\n" line endings, no trailing spaces on lines,
  at most one blank line in a row. The same text then always gets the same
  tokens and the same hash.
- too_short / too_long: very short pages carry little signal; huge ones are
  often logs or dumps and can use a lot of memory.
- bad_characters: control characters or U+FFFD (the "replacement character"
  left behind by broken text decoding) point to a broken page.
- few_letters: pages that are mostly digits, symbols or tables.
- exact_duplicate: the same cleaned text seen before (the MinHash step in
  FineWeb only removes duplicates inside one crawl, not across crawls).

The split is decided by a hash of the cleaned text, not by position. So a
document always lands in the same split no matter the order we read it in, and
exact duplicates can never end up in both splits.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_BAD_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]")
_LETTERS = re.compile(r"[^\W\d_]")  # any unicode letter


@dataclass
class CleanConfig:
    min_chars: int = 200
    max_chars: int = 1_000_000
    max_bad_char_frac: float = 0.001
    min_letter_frac: float = 0.25


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def rejection_reason(text: str, cfg: CleanConfig) -> str | None:
    """Return why a (normalized) document should be dropped, or None to keep it."""
    n = len(text)
    if n < cfg.min_chars:
        return "too_short"
    if n > cfg.max_chars:
        return "too_long"
    if len(_BAD_CHARS.findall(text)) > cfg.max_bad_char_frac * n:
        return "bad_characters"
    # Count letters without building a big list: remove them and compare lengths.
    letters = n - len(_LETTERS.sub("", text))
    if letters < cfg.min_letter_frac * n:
        return "few_letters"
    return None


def doc_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def is_val(digest: bytes, val_fraction: float) -> bool:
    """Deterministic split: about `val_fraction` of all documents go to validation."""
    bucket = int.from_bytes(digest[:8], "little") % 1_000_000
    return bucket < round(val_fraction * 1_000_000)


class Deduper:
    """Remembers 16-byte hashes of documents seen so far."""

    def __init__(self):
        self._seen: set[bytes] = set()

    def is_duplicate(self, digest: bytes) -> bool:
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False

    def __len__(self) -> int:
        return len(self._seen)


@dataclass
class CleaningStats:
    """Counts of what was kept and removed, with a few examples of each removal."""

    docs_in: int = 0
    docs_kept: int = 0
    chars_in: int = 0
    chars_kept: int = 0
    docs_changed_by_normalize: int = 0
    removed_docs: dict[str, int] = field(default_factory=dict)
    removed_chars: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    max_examples: int = 3

    def add_input(self, raw_chars: int, changed: bool) -> None:
        self.docs_in += 1
        self.chars_in += raw_chars
        self.docs_changed_by_normalize += int(changed)

    def add_kept(self, chars: int) -> None:
        self.docs_kept += 1
        self.chars_kept += chars

    def add_removed(self, reason: str, text: str) -> None:
        self.removed_docs[reason] = self.removed_docs.get(reason, 0) + 1
        self.removed_chars[reason] = self.removed_chars.get(reason, 0) + len(text)
        ex = self.examples.setdefault(reason, [])
        if len(ex) < self.max_examples:
            ex.append(text[:300])

    def to_dict(self) -> dict:
        return {
            "docs_in": self.docs_in,
            "docs_kept": self.docs_kept,
            "chars_in": self.chars_in,
            "chars_kept": self.chars_kept,
            "docs_changed_by_normalize": self.docs_changed_by_normalize,
            "removed_docs": dict(sorted(self.removed_docs.items())),
            "removed_chars": dict(sorted(self.removed_chars.items())),
            "removed_fraction_of_docs": {
                k: v / max(self.docs_in, 1) for k, v in sorted(self.removed_docs.items())
            },
            "examples": self.examples,
        }


def clean_document(raw: str, cfg: CleanConfig) -> tuple[str, str | None]:
    """Normalize one document and check it. Returns (clean_text, reason_or_None)."""
    text = normalize(raw)
    return text, rejection_reason(text, cfg)
