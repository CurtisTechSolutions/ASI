"""How a text becomes grams, and how a path of node labels becomes text again.

The encoding is the one ``RadixCyclicNN`` was born with, kept to what this
model needs: a sliding window of ``n`` characters (three unless told otherwise)
at stride 1, so consecutive grams share ``n - 1`` characters and a run of them
merges back into the text it came from - ``"hello" -> ["hel", "ell", "llo"]``
and back.  ``Encoding(n=1)`` is the plain character trie.

Two labels are reserved for the sentinels.  A real node can legitimately carry
the same text (``"x<s>y"`` holds the trigram ``"<s>"``), which is why the tree
tells its sentinels apart by *kind* and never by label.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["END_LABEL", "START_LABEL", "WINDOW", "Encoding"]

WINDOW = 3
"""The default n of the n-gram: the trigram."""

START_LABEL = "<s>"
END_LABEL = "</s>"


@dataclass(frozen=True)
class Encoding:
    """A sliding window of ``n`` characters, stride 1.

    Fixed for a tree's life: every label and every offset is measured in the
    units of the encoding that built it, and it travels with the model file.
    """

    n: int = WINDOW

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", int(self.n))

    def validate(self) -> None:
        """``ValueError`` for a window that cannot hold a character."""
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    @property
    def overlap(self) -> int:
        """Characters two consecutive grams share: ``n - 1``."""
        return self.n - 1

    # -- encoder -------------------------------------------------------------

    def encode(self, text: str) -> list[str]:
        """The grams of ``text``; ``[]`` if it is shorter than ``n``.

        The tail that does not fill a whole gram is dropped, exactly as the
        trigram encoding drops nothing but an empty or two-character text.
        """
        n = self.n
        last = len(text) - n
        if last < 0:
            return []
        return [text[i : i + n] for i in range(last + 1)]

    def check_grams(self, grams: Sequence[str]) -> None:
        """``ValueError`` unless every gram has ``n`` characters and consecutive grams overlap."""
        n = self.n
        prev: str | None = None
        for g in grams:
            if len(g) != n:
                raise ValueError(f"expected a gram of {n} characters, got {g!r}")
            if prev is not None and prev[1:] != g[:-1]:
                raise ValueError(f"grams {prev!r} -> {g!r} do not overlap")
            prev = g

    # -- decoder -------------------------------------------------------------

    def decode_grams(self, grams: Iterable[str]) -> str:
        """The inverse of :meth:`encode`: the first gram in full, then the new character of each following one."""
        grams = list(grams)
        if not grams:
            return ""
        ov = self.overlap
        return grams[0] + "".join(g[ov:] for g in grams[1:])

    def grams_held(self, label: str) -> int:
        """How many grams a label holds: ``len(label) - n + 1``, and 0 for a label too short to hold one."""
        held = len(label) - self.overlap
        return held if held > 0 else 0

    def gram_at(self, label: str, i: int) -> str:
        """Gram number ``i`` of a label."""
        return label[i : i + self.n]

    def decode_path(self, labels: Iterable[str], start_offset: int = 0, include_context: bool = True) -> str:
        """Decode node labels in path order into text.

        Every label after the first contributes the part of it past the
        overlap.  The first label contributes ``label[start_offset:]`` when
        ``include_context`` is true (the whole context, used from START) or
        ``label[start_offset + n:]`` when it is false: the deterministic
        remainder of a compressed node *after* the matched gram, so a
        prediction returns only the continuation.  ``start_offset`` is a gram
        index, the way the tree locates a prefix.
        """
        if start_offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {start_offset}")
        first_cut = start_offset if include_context else start_offset + self.n
        parts: list[str] = []
        first = True
        ov = self.overlap
        for label in labels:
            parts.append(label[first_cut:] if first else label[ov:])
            first = False
        return "".join(parts)

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return {"unit": "char", "n": self.n, "stride": 1}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Encoding":
        """Inverse of :meth:`to_dict`; ``None`` is the default encoding."""
        if not d:
            return cls()
        if d.get("unit", "char") != "char" or int(d.get("stride", 1)) != 1:
            raise ValueError(f"unsupported encoding {d!r}: this model reads character grams at stride 1")
        return cls(int(d.get("n", WINDOW)))

    def __str__(self) -> str:
        return f"char:{self.n}:1"
