"""The prose corpus the comparison is run on, and how a corpus file is read.

Built out of the repository's own research papers - ordinary English prose,
no download, no dependency, nothing invented for the occasion - one text per
line of the markdown that is prose (headings, tables, code blocks and links
dropped, emphasis marks removed).  The file list is **pinned** rather than
globbed, and :func:`build` writes what it read to ``data/corpus.txt`` with a
digest, so a rerun can tell whether the input drifted when the repository
changed under it.  ``FilterBankRadix`` builds its corpus the same way, for the
same reason.
"""

from __future__ import annotations

import hashlib
import os
import re

__all__ = ["DEFAULT_LINES", "MAX_LINE", "MIN_LINE", "SNAPSHOT", "SOURCES", "build", "digest", "load", "read_texts", "repo_root", "save", "split"]

MIN_LINE = 24
"""Shortest text kept, in characters."""
MAX_LINE = 200
"""Longest text kept: a markdown line longer than this is a table or a paragraph pasted whole, not prose."""
DEFAULT_LINES = 800
"""How many texts the snapshot holds by default."""

SOURCES: tuple[str, ...] = (
    "Research/CyclesAreAFeature.md",
    "Research/SineWaveActivationFunction.md",
    "Research/VanishingGradientIsAFeature.md",
    "Research/2NRL.md",
)
"""The four papers, in this order; ``Research/Insights.md`` is a living index and is deliberately not one of them."""

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(os.path.dirname(HERE), "data", "corpus.txt")

_MARK = re.compile(r"^(?:[-*+]\s+|\d+\.\s+|>\s*)+")
_EMPHASIS = re.compile(r"[`*]")
_SPACES = re.compile(r"\s+")


def repo_root(start: str = HERE) -> str:
    """Walk up until the directory holding ``Research/`` and ``RadixCyclicNN/``."""
    d = start
    while True:
        if all(os.path.isdir(os.path.join(d, x)) for x in ("Research", "RadixCyclicNN")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("could not find the repository root above " + start)
        d = parent


def lines_of(text: str, min_len: int = MIN_LINE, max_len: int = MAX_LINE) -> list[str]:
    """The prose lines of a markdown document, cleaned, in order."""
    out: list[str] = []
    fenced = False
    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not line:
            continue
        if line[0] in "#|<!" or line.startswith("---") or line.startswith("===") or "http" in line:
            continue
        line = _MARK.sub("", line)
        line = _EMPHASIS.sub("", line)
        line = _SPACES.sub(" ", line).strip()
        if min_len <= len(line) <= max_len:
            out.append(line)
    return out


def build(limit: int | None = DEFAULT_LINES) -> list[str]:
    """The corpus from the pinned files: distinct prose lines, the first ``limit`` of them."""
    root = repo_root()
    seen: set[str] = set()
    texts: list[str] = []
    for rel in SOURCES:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            for line in lines_of(fh.read()):
                if line not in seen:
                    seen.add(line)
                    texts.append(line)
    return texts[:limit] if limit else texts


def digest(texts) -> str:
    """SHA-256 of the exact texts, one per line."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def save(texts, path: str = SNAPSHOT) -> str:
    """Write the corpus, one text per line; returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(texts) + "\n")
    return path


def read_texts(path: str) -> list[str]:
    """The non-blank lines of a text file, stripped: one text per line."""
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh.read().splitlines() if line.strip()]


def load(path: str = SNAPSHOT) -> list[str]:
    """The snapshot, or the corpus built (and written) if there is none yet."""
    if os.path.isfile(path):
        return read_texts(path)
    texts = build()
    save(texts, path)
    return texts


def split(texts, every: int = 5) -> tuple[list[str], list[str]]:
    """``(train, held_out)``: every ``every``-th text is held out (none when ``every`` is 0)."""
    if every <= 0:
        return list(texts), []
    train = [t for i, t in enumerate(texts) if i % every != every - 1]
    test = [t for i, t in enumerate(texts) if i % every == every - 1]
    return train, test
