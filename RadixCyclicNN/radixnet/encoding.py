"""How a text becomes the grams the graph is built from, and how it comes back.

:class:`Encoding` is three dials - what one *unit* of text is, how many units a
gram holds, and how far apart consecutive grams start:

* ``Encoding()`` is the default and the one the model was born with: character
  trigrams, stride 1 (``"hello" -> ["hel", "ell", "llo"]``).
* ``Encoding(n=5)`` is a sliding window of five characters, and any n works.
* ``Encoding(n=4, stride=4)`` is *tokenisation*: non-overlapping groups of four
  letters. So is ``Encoding(n=5, stride=5)``.
* ``Encoding(unit=WORDS, n=2)`` is the word bigram, ``n=3`` the word trigram.

:class:`Encoder` and :class:`Decoder` are the two halves of it kept as objects,
because that is how the rest of the package holds them; ``Encoder(window=5)``
still means what it always did.  :meth:`Decoder.decode_path` reverses the
(possibly path-compressed) node labels of the graph.

:func:`repair_base64` is the shared tail of the *media* text formats
(``img:...`` in :mod:`radixnet.vision`, ``aud:...`` in :mod:`radixnet.speech`):
the base64 payload of a text the network predicted is rarely clean, so it is
repaired before it is decoded.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

WINDOW = 3
"""The default n of the n-gram: the trigram the model was born with."""

CHARS = "char"
"""One unit is one character."""
WORDS = "word"
"""One unit is one whitespace-delimited word; the text is normalised to single spaces."""
UNIT_KINDS = (CHARS, WORDS)

START_LABEL = "<s>"
END_LABEL = "</s>"
BACK_LABEL = "<back>"
"""The third sentinel: where the graph has learned that a walk goes round (``graph.BACK``)."""

# A "unit view" is whatever slices by unit: a str for characters (Python slices
# strings by code point already), a list of words for words.  ``len`` is the
# unit count of both, and :func:`_piece` writes a slice of either back out.
UnitView = "str | list[str]"


def _piece(view: str | Sequence[str], lo: int, hi: int | None = None) -> str:
    """Units ``[lo:hi)`` of a view, as text."""
    part = view[lo:hi]
    return part if isinstance(part, str) else " ".join(part)


@dataclass(frozen=True)
class Encoding:
    """How a text becomes grams, and how labels become text again.

    Three dials, fixed for a graph's life (every label, every index key and
    every offset is measured in the units of the encoding that built it):

    * ``unit`` - :data:`CHARS` or :data:`WORDS`.
    * ``n`` - units per gram: the *n* of the n-gram, any ``n >= 1``.
    * ``stride`` - units between two consecutive grams.  ``1`` slides the
      window, so consecutive grams share ``n - 1`` units, which is what lets
      the graph chain them; ``n`` cuts the text into groups that share nothing.
    """

    unit: str = CHARS
    n: int = WINDOW
    stride: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "stride", int(self.stride))
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if this is not an encoding the graph can be built in."""
        if self.unit not in UNIT_KINDS:
            raise ValueError(f"unit must be one of {UNIT_KINDS}, got {self.unit!r}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.stride > self.n:
            raise ValueError(f"stride {self.stride} must be <= n {self.n}: a larger stride would skip units")

    # -- what it is ----------------------------------------------------------

    @property
    def overlap(self) -> int:
        """Units two consecutive grams share: ``n - stride``, so ``n - 1`` sliding and 0 in groups."""
        return self.n - self.stride

    @property
    def sliding(self) -> bool:
        """Do consecutive grams overlap at all?"""
        return self.stride < self.n

    def is_default(self) -> bool:
        """The character trigram of stride 1: what a model file leaves unwritten."""
        return self == Encoding()

    def __str__(self) -> str:
        return f"{self.unit}:{self.n}:{self.stride}"

    def describe(self) -> str:
        """The human form: ``'2-word grams, stride 1 (sliding)'``."""
        unit = "character" if self.unit == CHARS else "word"
        return f"{self.n}-{unit} grams, stride {self.stride} ({'sliding' if self.sliding else 'groups'})"

    def to_dict(self) -> dict:
        return {"unit": self.unit, "n": self.n, "stride": self.stride}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Encoding":
        """Read an ``encoding`` block of a model file; ``None`` is the default."""
        if not d:
            return cls()
        return cls(unit=str(d.get("unit", CHARS)), n=int(d.get("n", WINDOW)), stride=int(d.get("stride", 1)))

    # -- units ---------------------------------------------------------------

    def units(self, text: str) -> str | list[str]:
        """The text as something that slices by unit: itself, or its words."""
        return text if self.unit == CHARS else text.split()

    def length(self, text: str) -> int:
        """How many units a text holds."""
        return len(text) if self.unit == CHARS else len(text.split())

    def piece(self, text: str, lo: int, hi: int | None = None) -> str:
        """Units ``[lo:hi)`` of a text, as text."""
        return _piece(self.units(text), lo, hi)

    def join(self, *parts: str) -> str:
        """Glue unit-aligned pieces: nothing between characters, one space between words.

        Empty pieces are dropped, so a label contributing no unit adds no separator.
        """
        if self.unit != WORDS:
            return "".join(parts)
        return " ".join(p for p in parts if p)

    def truncate(self, text: str, n: int) -> str:
        """The first ``n`` units; ``n < 0`` leaves the text alone."""
        return text if n < 0 else self.piece(text, 0, n)

    def has_unit_prefix(self, text: str, prefix: str) -> bool:
        """Does ``text`` start with ``prefix`` *on a unit boundary*?

        For words that means whole words: ``"the ca"`` is not a prefix of
        ``"the cat sat"``, ``"the cat"`` is.
        """
        if not prefix:
            return True
        if self.unit != WORDS:
            return text.startswith(prefix)
        return text == prefix or text.startswith(prefix + " ")

    # -- encoder -------------------------------------------------------------

    def covered(self, units: int) -> int:
        """How many of ``units`` units the grams of a text actually cover."""
        if units < self.n:
            return 0
        return (units - self.n) // self.stride * self.stride + self.n

    def encode(self, text: str) -> list[str]:
        """The grams of a text: ``n`` units each, ``stride`` units apart; ``[]`` if it is too short.

        The tail that does not fill a whole gram is dropped, exactly as the
        trigram encoding drops the last two characters of a text.
        :meth:`normalize` is what is left.
        """
        view = self.units(text)
        last = len(view) - self.n
        if last < 0:
            return []
        return [_piece(view, i, i + self.n) for i in range(0, last + 1, self.stride)]

    def normalize(self, text: str) -> str:
        """The text this encoding can represent - what a round trip through the graph returns.

        For the default encoding that is the text itself; a word encoding loses
        the original spacing, and a grouping one the tail that does not fill a
        group.
        """
        view = self.units(text)
        covered = self.covered(len(view))
        return _piece(view, 0, covered) if covered else ""

    # -- decoder -------------------------------------------------------------

    def decode_grams(self, grams: Iterable[str]) -> str:
        """The inverse of :meth:`encode`, up to the tail it dropped."""
        grams = list(grams)
        if not grams:
            return ""
        return self.join(grams[0], *(self.piece(g, self.overlap) for g in grams[1:]))

    def decode_path(
        self,
        labels: Iterable[str],
        start_offset: int = 0,
        include_context: bool = True,
        skip_sentinels: bool = True,
    ) -> str:
        """Decode node labels in path order into text (see :meth:`Decoder.decode_path`)."""
        if start_offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {start_offset}")
        first_cut = start_offset if include_context else start_offset + self.n
        parts: list[str] = []
        first = True
        for label in labels:
            if skip_sentinels and label in (START_LABEL, END_LABEL, BACK_LABEL):
                continue
            parts.append(self.piece(label, first_cut if first else self.overlap))
            first = False
        return self.join(*parts)


DEFAULT_ENCODING = Encoding()
"""Character trigrams of stride 1: what every model used before the encoding became a choice."""

_ALIASES = {
    "": Encoding(),
    "default": Encoding(),
    "trigram": Encoding(),
    "trigrams": Encoding(),
    "bigram": Encoding(n=2),
    "bigrams": Encoding(n=2),
    "word": Encoding(unit=WORDS, n=1),
    "words": Encoding(unit=WORDS, n=1),
    "word-unigram": Encoding(unit=WORDS, n=1),
    "word-bigram": Encoding(unit=WORDS, n=2),
    "word-bigrams": Encoding(unit=WORDS, n=2),
    "word-trigram": Encoding(unit=WORDS, n=3),
    "word-trigrams": Encoding(unit=WORDS, n=3),
}
_UNIT_NAMES = {
    "char": CHARS, "chars": CHARS, "character": CHARS, "characters": CHARS, "letter": CHARS, "letters": CHARS,
    "word": WORDS, "words": WORDS,
}


def parse_encoding(spec: str) -> Encoding:
    """Read a spec: ``unit[:n[:stride]]``, or one of the names in :data:`_ALIASES`.

    ``"char:3:1"`` is the default, ``"char:5:groups"`` (or ``"char:5:5"``)
    non-overlapping groups of five letters, ``"word:2"`` the word bigram.
    """
    text = (spec or "").strip().lower()
    if text in _ALIASES:
        return _ALIASES[text]
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"encoding {spec!r}: expected unit[:n[:stride]]")
    if parts[0] not in _UNIT_NAMES:
        raise ValueError(f"encoding {spec!r}: unit must be char or word, got {parts[0]!r}")
    unit = _UNIT_NAMES[parts[0]]
    n = WINDOW
    if len(parts) > 1 and parts[1]:
        try:
            n = int(parts[1])
        except ValueError:
            raise ValueError(f"encoding {spec!r}: n must be a number, got {parts[1]!r}") from None
    stride = 1
    if len(parts) > 2 and parts[2]:
        if parts[2] in ("groups", "group", "blocks", "block"):
            stride = n
        elif parts[2] in ("sliding", "slide"):
            stride = 1
        else:
            try:
                stride = int(parts[2])
            except ValueError:
                raise ValueError(f"encoding {spec!r}: stride must be a number, got {parts[2]!r}") from None
    try:
        return Encoding(unit=unit, n=n, stride=stride)
    except ValueError as exc:
        raise ValueError(f"encoding {spec!r}: {exc}") from None


class Encoder:
    """The encoder half of an :class:`Encoding`, kept as an object.

    ``Encoder()`` is the sliding window of three characters; ``Encoder(5)`` a
    window of five; ``Encoder(encoding=Encoding(unit=WORDS, n=2))`` the word
    bigram.
    """

    __slots__ = ("encoding",)

    def __init__(self, window: int = WINDOW, encoding: Encoding | None = None) -> None:
        self.encoding = encoding if encoding is not None else Encoding(n=window)

    @property
    def window(self) -> int:
        """The n of the n-gram (the name this had when three was the only answer)."""
        return self.encoding.n

    def encode(self, text: str) -> list[str]:
        """Return the grams of ``text``; ``[]`` if it holds fewer than ``n`` units."""
        return self.encoding.encode(text)

    def __repr__(self) -> str:
        return f"Encoder(encoding={self.encoding})"


class Decoder:
    """Inverse of :class:`Encoder`, for raw grams and for compressed node labels."""

    __slots__ = ("encoding",)

    def __init__(self, window: int = WINDOW, encoding: Encoding | None = None) -> None:
        self.encoding = encoding if encoding is not None else Encoding(n=window)

    @property
    def window(self) -> int:
        """The n of the n-gram."""
        return self.encoding.n

    def decode_trigrams(self, grams: Iterable[str]) -> str:
        """``[] -> ""``; the first gram in full, then the part of each following one past the overlap."""
        return self.encoding.decode_grams(grams)

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
        contributes the part of it past the overlap (``n - stride`` units).
        The first real label contributes

        * ``label[start_offset:]`` when ``include_context`` is true (full
          context, used when generating from START), or
        * ``label[start_offset + n:]`` when it is false: the deterministic
          remainder of a compressed node *after* the matched gram, so a
          prediction returns only the continuation.

        A real node can legitimately carry a label equal to a sentinel label
        (the text ``"x<s>y"`` contains the trigram ``"<s>"``).  Callers that
        know the node ids strip START/END themselves and pass
        ``skip_sentinels=False`` so such a node is decoded like any other.
        """
        return self.encoding.decode_path(labels, start_offset, include_context, skip_sentinels)

    def __repr__(self) -> str:
        return f"Decoder(encoding={self.encoding})"


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
