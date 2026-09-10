"""Sliding-window character encoding and its inverse.

``Encoder.encode`` turns text into overlapping windows of ``WINDOW`` (3)
characters with stride 1 (``"hello" -> ["hel", "ell", "llo"]``).
``Decoder`` reverses that for raw windows (:meth:`Decoder.decode_trigrams`) and
for the (possibly path-compressed) node labels of the graph
(:meth:`Decoder.decode_path`).
"""

from __future__ import annotations

from collections.abc import Iterable

WINDOW = 3
START_LABEL = "<s>"
END_LABEL = "</s>"


class Encoder:
    """Sliding-window encoder (stride 1)."""

    __slots__ = ("window",)

    def __init__(self, window: int = WINDOW) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)

    def encode(self, text: str) -> list[str]:
        """Return the overlapping windows of ``text``; ``[]`` if it is too short."""
        w = self.window
        n = len(text) - w + 1
        if n <= 0:
            return []
        return [text[i : i + w] for i in range(n)]

    def __repr__(self) -> str:
        return f"Encoder(window={self.window})"


class Decoder:
    """Inverse of :class:`Encoder` for windows and for compressed node labels."""

    __slots__ = ("window",)

    def __init__(self, window: int = WINDOW) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)

    def decode_trigrams(self, grams: Iterable[str]) -> str:
        """``[] -> ""``; first window in full, then the last char of each following window."""
        grams = list(grams)
        if not grams:
            return ""
        return grams[0] + "".join(g[-1] for g in grams[1:])

    def decode_path(
        self,
        labels: Iterable[str],
        start_offset: int = 0,
        include_context: bool = True,
        skip_sentinels: bool = True,
    ) -> str:
        """Decode node labels in path order into text.

        ``START_LABEL`` / ``END_LABEL`` entries are skipped while
        ``skip_sentinels`` is true.  Every label after the first *real* one
        contributes ``label[window - 1:]`` (the part that does not overlap its
        predecessor).  The first real label contributes

        * ``label[start_offset:]`` when ``include_context`` is true (full
          context, used when generating from START), or
        * ``label[start_offset + window:]`` when it is false: the deterministic
          remainder of a compressed node *after* the matched window, so a
          prediction returns only the continuation.

        A real node can legitimately carry a label equal to a sentinel label
        (the text ``"x<s>y"`` contains the trigram ``"<s>"``).  Callers that
        know the node ids strip START/END themselves and pass
        ``skip_sentinels=False`` so such a node is decoded like any other.
        """
        if start_offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {start_offset}")
        overlap = self.window - 1
        first_cut = start_offset if include_context else start_offset + self.window
        parts: list[str] = []
        first = True
        for label in labels:
            if skip_sentinels and (label == START_LABEL or label == END_LABEL):
                continue
            if first:
                first = False
                parts.append(label[first_cut:])
            else:
                parts.append(label[overlap:])
        return "".join(parts)

    def __repr__(self) -> str:
        return f"Decoder(window={self.window})"
