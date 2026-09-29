"""How a corpus file is read, and how one is split for a held-out measurement.

A corpus is a text file, one text per line.  ``data/sample_corpus.txt`` is
the sixty sentences every model directory in this repository is shown, and
``data/corpus.txt`` the 800-line prose snapshot ``RadixAcyclicNN`` cut out of
the research papers (a copy, so this directory stands alone; its digest is
recorded there).
"""

from __future__ import annotations

import hashlib
import os

__all__ = ["GARBAGE", "PROSE", "SAMPLE", "digest", "read_texts", "split"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SAMPLE = os.path.join(DATA, "sample_corpus.txt")
GARBAGE = os.path.join(DATA, "sample_garbage.txt")
PROSE = os.path.join(DATA, "corpus.txt")


def read_texts(path: str) -> list[str]:
    """The non-blank lines of a text file, stripped: one text per line."""
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh.read().splitlines() if line.strip()]


def digest(texts) -> str:
    """SHA-256 of the exact texts, one per line."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def split(texts, every: int = 5) -> tuple[list[str], list[str]]:
    """``(read, held_out)``: every ``every``-th text is held out (none when ``every`` is 0)."""
    if every <= 0:
        return list(texts), []
    kept = [t for i, t in enumerate(texts) if i % every != every - 1]
    held = [t for i, t in enumerate(texts) if i % every == every - 1]
    return kept, held
