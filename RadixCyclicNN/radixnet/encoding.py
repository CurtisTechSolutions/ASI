"""Sliding-window character encoding and its inverse.

``Encoder.encode`` turns text into overlapping windows of ``WINDOW`` (3)
characters with stride 1 (``"hello" -> ["hel", "ell", "llo"]``).
``Decoder`` reverses that for raw windows (:meth:`Decoder.decode_trigrams`) and
for the (possibly path-compressed) node labels of the graph
(:meth:`Decoder.decode_path`).

:func:`repair_base64` is the shared tail of the *media* text formats
(``img:...`` in :mod:`radixnet.vision`, ``aud:...`` in :mod:`radixnet.speech`):
the base64 payload of a text the network predicted is rarely clean, so it is
repaired before it is decoded.
"""

from __future__ import annotations

import base64
import binascii
import re
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


_B64_JUNK = re.compile(r"[^A-Za-z0-9+/=]")


def repair_base64(body: str) -> tuple[bytes, bool]:
    """Decode the base64 tail of a media text, repairing it first; ``(payload, repaired)``.

    The media encoders (:mod:`radixnet.vision`, :mod:`radixnet.speech`) pack
    their payload as base64 into a text the network trains on and *predicts*,
    so what comes back may be cut off, padded with junk or interrupted by
    whitespace.  Characters outside the base64 alphabet are dropped, a single
    dangling character (which can never decode) is removed with them, the
    padding is completed, and ``repaired`` says whether any of that changed
    the text.  The payload is returned as it decodes - callers pad or truncate
    it to the length their format needs.
    """
    clean = _B64_JUNK.sub("", body).rstrip("=")
    if len(clean) % 4 == 1:  # a single dangling character can never decode
        clean = clean[:-1]
    padded = clean + "=" * (-len(clean) % 4)
    repaired = padded != body.strip()  # a clean text comes back unchanged, padding included
    try:
        payload = base64.b64decode(padded, validate=True)
    except (ValueError, binascii.Error) as exc:  # pragma: no cover - the junk filter makes this rare
        raise ValueError(f"the base64 part cannot be decoded: {exc}") from exc
    return payload, repaired
