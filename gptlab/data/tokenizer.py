"""Byte-level BPE tokenizer.

BPE (byte pair encoding) starts from single bytes and repeatedly merges the most
common neighbouring pair into a new token, until the vocabulary has the size we
asked for. "Byte-level" means the base alphabet is the 256 possible bytes, so
any text can be encoded and decoded back with no unknown tokens and no loss.

We use the `tokenizers` library only to run the BPE training fast (it is written
in Rust). The model code never depends on it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"  # end-of-text token, placed before every document


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's map from each byte to a printable unicode character.

    The byte-level pre-tokenizer stores tokens as strings of these characters.
    We need the reverse map to know how many bytes each token stands for.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_UNICODE_TO_BYTE = {c: b for b, c in _bytes_to_unicode().items()}


class Tokenizer:
    """Thin wrapper that gives the rest of the code a small, fixed interface."""

    def __init__(self, hf_tokenizer: HFTokenizer):
        self._tok = hf_tokenizer
        # Treat a literal "<|endoftext|>" inside web text as normal text.
        # Only the data pipeline inserts the real end-of-text token.
        self._tok.encode_special_tokens = True
        eot = self._tok.token_to_id(EOT)
        if eot is None:
            raise ValueError("tokenizer has no end-of-text token")
        self.eot_id: int = eot
        self.vocab_size: int = self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def token_bytes(self) -> np.ndarray:
        """How many UTF-8 bytes each token id stands for (0 for special tokens).

        Used to turn loss per token into bits per byte, which does not depend
        on the tokenizer and so can be compared across tokenizers.
        """
        out = np.zeros(self.vocab_size, dtype=np.int64)
        special = {t.content for t in self._tok.get_added_tokens_decoder().values()}
        for i in range(self.vocab_size):
            piece = self._tok.id_to_token(i)
            if piece is None or piece in special:
                continue
            out[i] = sum(1 for ch in piece if ch in _UNICODE_TO_BYTE)
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(HFTokenizer.from_file(str(path)))


def train_tokenizer(texts: Iterable[str], vocab_size: int, min_frequency: int = 2) -> Tokenizer:
    """Train a byte-level BPE tokenizer. Token id 0 is the end-of-text token."""
    tok = HFTokenizer(models.BPE())
    # Split text GPT-2 style (words, numbers, punctuation, spaces) before BPE,
    # so merges never cross word boundaries. No prefix space is added, which
    # keeps encode -> decode an exact round trip.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return Tokenizer(tok)


def bytes_per_token(tokenizer: Tokenizer, texts: list[str]) -> float:
    """Average UTF-8 bytes per token on some texts. Higher means better compression."""
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(ids) for ids in tokenizer.encode_batch(texts))
    return n_bytes / max(n_tokens, 1)
