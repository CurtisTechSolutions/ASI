"""GPT-2's byte-level byte-pair encoding, in the standard library.

Run from the two files of the original release - ``encoder.json`` (token ->
id) and ``vocab.bpe`` (the merges, in rank order) - or from Hugging Face's
``vocab.json`` and ``merges.txt``, which hold the same content.  The three
parts of the algorithm are the ones in OpenAI's ``encoder.py``: the reversible
byte-to-unicode table, the pre-tokenizer, and the rank-ordered merging.

The pre-tokenizer is GPT-2's own regular expression with ``\\p{L}`` written
``[^\\W\\d_]`` and ``\\p{N}`` written ``\\d`` for Python's ``re`` (``\\d`` is the
decimal digits where ``\\p{N}`` is every numeric character, so a text full of
fractions or Roman numerals may cut differently); ``tests/test_codec.py``
holds the result to ``tiktoken``'s ``gpt2`` encoding id for id when that
package is installed.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence

__all__ = ["Gpt2BPE", "PRE_TOKENIZER", "bytes_to_unicode", "gpt2_files"]

PRE_TOKENIZER = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")

FILE_NAMES = (("encoder.json", "vocab.bpe"), ("vocab.json", "merges.txt"))
"""The two spellings of the two files: the original release's, and Hugging Face's."""


def bytes_to_unicode() -> dict[int, str]:
    """Every byte to a printable code point, reversibly - GPT-2's table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def gpt2_files(directory: str) -> tuple[str, str] | None:
    """The encoder and merges files in a directory, under either spelling, or ``None``."""
    for enc, bpe in FILE_NAMES:
        e, b = os.path.join(directory, enc), os.path.join(directory, bpe)
        if os.path.isfile(e) and os.path.isfile(b):
            return e, b
    return None


class Gpt2BPE:
    """The tokenizer: ``encode`` text to GPT-2 ids, ``decode`` them back."""

    def __init__(self, encoder: dict[str, int], merges: Sequence[tuple[str, str]]) -> None:
        self.encoder = dict(encoder)
        self.decoder = {i: t for t, i in self.encoder.items()}
        if len(self.decoder) != len(self.encoder):
            raise ValueError("the encoder maps two tokens to one id")
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: r for r, m in enumerate(self.merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.cache: dict[str, list[str]] = {}

    # -- files ------------------------------------------------------------------------

    @classmethod
    def load(cls, encoder_path: str, merges_path: str) -> "Gpt2BPE":
        with open(encoder_path, encoding="utf-8") as fh:
            encoder = json.load(fh)
        merges: list[tuple[str, str]] = []
        with open(merges_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                a, b = line.split(" ")
                merges.append((a, b))
        return cls(encoder, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # -- the algorithm ------------------------------------------------------------------

    def _bpe(self, token: str) -> list[str]:
        cached = self.cache.get(token)
        if cached is not None:
            return cached
        word = list(token)
        if len(word) < 2:
            self.cache[token] = word
            return word
        ranks = self.ranks
        while True:
            best = None
            best_rank = None
            for i in range(len(word) - 1):
                r = ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (word[i], word[i + 1]), r
            if best is None:
                break
            a, b = best
            merged = []
            i = 0
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
            if len(word) == 1:
                break
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        encoder = self.encoder
        byte_encoder = self.byte_encoder
        for piece in PRE_TOKENIZER.findall(text):
            token = "".join(byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(encoder[t] for t in self._bpe(token))
        return ids

    def token_bytes(self, i: int) -> bytes:
        """The bytes a GPT-2 id stands for."""
        return bytes(self.byte_decoder[c] for c in self.decoder[i])

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def display(self, i: int) -> str:
        """A readable form of one token (its bytes decoded with replacement)."""
        return self.token_bytes(i).decode("utf-8", errors="replace")

    # -- persistence ----------------------------------------------------------------------

    def to_dict(self) -> dict:
        tokens = [self.decoder[i] for i in range(len(self.decoder))]
        return {"tokens": tokens, "merges": [f"{a} {b}" for a, b in self.merges]}

    @classmethod
    def from_dict(cls, d: dict) -> "Gpt2BPE":
        encoder = {t: i for i, t in enumerate(d["tokens"])}
        merges = [tuple(m.split(" ")) for m in d["merges"]]
        return cls(encoder, merges)
