"""The repository's own byte-level byte-pair encoding: trained on a corpus to a chosen vocabulary size.

The algorithm the LLM tokenizers use, sized to what priming can afford.  A
text is cut into *pieces* - one optional leading space followed by a run of
non-space characters, or a run of whitespace (its last space going to the word
after it, as GPT-2 cuts) - and merges never cross a piece.
Training starts from the 256 bytes and merges the most frequent adjacent pair
``vocab_size - 256`` times (ties broken by the smaller pair, so training is
deterministic); the merge list *is* the tokenizer.  Encoding merges the
lowest-ranked pair present until none applies; decoding concatenates the
tokens' bytes.  ``decode(encode(b)) == b`` for any byte string, because every
byte is a token.

Byte-identical to no published tokenizer, and it does not claim to be:
``gpt2.py`` and the ``external`` codec are for that.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

__all__ = ["BPE", "PIECES"]

PIECES = re.compile(r" ?\S+|\s+(?!\S)|\s+")


class BPE:
    """``merges`` in rank order; token ``256 + r`` is the ``r``-th merge."""

    def __init__(self, merges: Sequence[tuple[int, int]] = ()) -> None:
        self.merges: list[tuple[int, int]] = [tuple(m) for m in merges]
        self.ranks: dict[tuple[int, int], int] = {m: r for r, m in enumerate(self.merges)}
        self.tokens: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.tokens.append(self.tokens[a] + self.tokens[b])
        self.cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    # -- training ----------------------------------------------------------------------

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BPE":
        if vocab_size < 256:
            raise ValueError(f"a byte-level vocabulary holds at least the 256 bytes, got {vocab_size}")
        words: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for piece in PIECES.findall(text):
                words[tuple(piece.encode("utf-8"))] += 1
        merges: list[tuple[int, int]] = []
        table: dict[tuple[int, ...], int] = dict(words)
        for step in range(vocab_size - 256):
            pairs: Counter[tuple[int, int]] = Counter()
            for word, freq in table.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            new = 256 + step
            merges.append(best)
            a, b = best
            merged: dict[tuple[int, ...], int] = {}
            for word, freq in table.items():
                if len(word) > 1:
                    out = []
                    i = 0
                    n = len(word)
                    while i < n:
                        if i < n - 1 and word[i] == a and word[i + 1] == b:
                            out.append(new)
                            i += 2
                        else:
                            out.append(word[i])
                            i += 1
                    word = tuple(out)
                merged[word] = merged.get(word, 0) + freq
            table = merged
        return cls(merges)

    # -- encoding / decoding -------------------------------------------------------------

    def _piece(self, piece: str) -> list[int]:
        cached = self.cache.get(piece)
        if cached is not None:
            return cached
        word = list(piece.encode("utf-8"))
        ranks = self.ranks
        while len(word) > 1:
            best = None
            best_rank = None
            for i in range(len(word) - 1):
                r = ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (word[i], word[i + 1]), r
            if best is None:
                break
            a, b = best
            new = 256 + best_rank
            out = []
            i = 0
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == a and word[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        self.cache[piece] = word
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in PIECES.findall(text):
            ids.extend(self._piece(piece))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.tokens[i] for i in ids).decode("utf-8", errors="replace")

    def display(self, i: int) -> str:
        return self.tokens[i].decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"merges": [[a, b] for a, b in self.merges]}

    @classmethod
    def from_dict(cls, d: dict) -> "BPE":
        return cls([(int(a), int(b)) for a, b in d["merges"]])
