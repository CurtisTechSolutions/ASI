"""How a text becomes grams, and how a path of node labels becomes text again.

An :class:`Encoding` is two dials: what one *unit* of text is, and how many
units a gram holds.  The window slides at stride 1, so consecutive grams share
``n - 1`` units and a run of them merges back into the text it came from.

* ``Encoding()`` is the one ``RadixCyclicNN`` was born with: character
  trigrams, ``"hello" -> ["hel", "ell", "llo"]`` and back.  ``Encoding(n=1)``
  is the plain character trie.
* ``Encoding(unit=PHONES)`` is the trigram of *sounds*: the text is read
  through the phonetic tokenizer (the sibling ``PhoneticTokenizer`` package,
  ``phonetok``), ``"the cat"`` becomes the units ``DH AH0 # K AE1 T``, and the
  tree is built over sounds rather than letters.  ``Encoding(unit=SYLLABLES)``
  is the trigram of syllables (``DH.AH0 # K.AE1.T``).  A label is that unit
  text, so a model file stays readable, and a prediction made of sounds is
  spelled back into words through the same tokenizer (:meth:`Encoding.spell`).

Two labels are reserved for the sentinels.  A real node can legitimately carry
the same text (``"x<s>y"`` holds the trigram ``"<s>"``), which is why the tree
tells its sentinels apart by *kind* and never by label.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "CHARS", "END_LABEL", "PHONES", "PHONETIC_UNITS", "START_LABEL", "SYLLABLES", "UNIT_KINDS", "WINDOW",
    "Encoding", "phonetic_tokenizer", "phonetok_module",
]

WINDOW = 3
"""The default n of the n-gram: the trigram."""

CHARS = "char"
"""One unit is one character."""
PHONES = "phone"
"""One unit is one sound: a phoneme (``DH``, ``AH0``, ``K``), the ``#`` between two words, or a pause."""
SYLLABLES = "syllable"
"""One unit is one syllable (``K.AE1.T``), the ``#`` between two words, or a pause."""
UNIT_KINDS = (CHARS, PHONES, SYLLABLES)
PHONETIC_UNITS = (PHONES, SYLLABLES)
_UNITS_NAMES = {CHARS: "chars", PHONES: "phones", SYLLABLES: "syllables"}
_UNIT_WORDS = {CHARS: "character", PHONES: "phone", SYLLABLES: "syllable"}

START_LABEL = "<s>"
END_LABEL = "</s>"

_PHONETIC: dict[str, object] = {}


def phonetok_module():
    """The ``phonetok`` package: from the environment, or from the checkout beside this project.

    The tokenizer lives in the sibling ``PhoneticTokenizer`` directory and a
    checkout finds it there on its own; anywhere else,
    ``pip install -e ../PhoneticTokenizer``.  Raises :class:`ValueError` when
    neither works, so a phonetic unit is refused at construction rather than
    mid-run.
    """
    try:
        return importlib.import_module("phonetok")
    except ImportError:
        pass
    sibling = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "PhoneticTokenizer")
    if os.path.isdir(os.path.join(sibling, "phonetok")) and sibling not in sys.path:
        sys.path.append(sibling)
    try:
        return importlib.import_module("phonetok")
    except ImportError as exc:
        raise ValueError(
            "a phonetic unit needs the phonetok package (the sibling PhoneticTokenizer directory: "
            f"pip install -e ../PhoneticTokenizer): {exc}"
        ) from None


def phonetic_tokenizer(unit: str):
    """The tokenizer a phonetic unit reads through: ``phonetok`` over its portable lexicon, made once per unit.

    The lexicon is the portable one - the bundled core plus the file
    ``PHONETOK_LEXICON`` names - because a model whose symbols are sounds is
    only as portable as the lexicon that made them.  What the tokenizer
    remembers of the words it sounded out is what lets a prediction be spelled
    back.
    """
    tok = _PHONETIC.get(unit)
    if tok is None:
        phonetok = phonetok_module()
        level = "phoneme" if unit == PHONES else "syllable"
        tok = phonetok.PhoneticTokenizer(level=level, lexicon=phonetok.Lexicon.portable())
        _PHONETIC[unit] = tok
    return tok


def _piece(view: str | Sequence[str], lo: int, hi: int | None = None) -> str:
    """Units ``[lo:hi)`` of a view - a string of characters or a list of unit tokens - as text."""
    part = view[lo:hi]
    return part if isinstance(part, str) else " ".join(part)


@dataclass(frozen=True)
class Encoding:
    """What a unit is, and how many make a gram.  Fixed for a tree's life; it travels with the model file."""

    unit: str = CHARS
    n: int = WINDOW

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "n", int(self.n))

    def validate(self) -> None:
        """``ValueError`` for a unit the tree cannot read or a window that cannot hold one."""
        if self.unit not in UNIT_KINDS:
            raise ValueError(f"unit must be one of {UNIT_KINDS}, got {self.unit!r}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.phonetic:
            phonetic_tokenizer(self.unit)

    # -- what it is ----------------------------------------------------------

    @property
    def overlap(self) -> int:
        """Units two consecutive grams share: ``n - 1``."""
        return self.n - 1

    @property
    def phonetic(self) -> bool:
        """Are the units sounds rather than letters?"""
        return self.unit in PHONETIC_UNITS

    @property
    def units_name(self) -> str:
        """What this encoding counts in: ``"chars"``, ``"phones"`` or ``"syllables"``."""
        return _UNITS_NAMES[self.unit]

    def describe(self) -> str:
        """The human form: ``'3-phone grams'``."""
        return f"{self.n}-{_UNIT_WORDS[self.unit]} grams"

    def __str__(self) -> str:
        return f"{self.unit}:{self.n}:1"

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return {"unit": self.unit, "n": self.n, "stride": 1}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Encoding":
        """Inverse of :meth:`to_dict`; ``None`` is the default encoding."""
        if not d:
            return cls()
        if int(d.get("stride", 1)) != 1:
            raise ValueError(f"unsupported encoding {d!r}: this model reads grams at stride 1")
        return cls(str(d.get("unit", CHARS)), int(d.get("n", WINDOW)))

    # -- units ---------------------------------------------------------------

    def units(self, text: str) -> str | list[str]:
        """A text as something that slices by unit: itself for characters, its sounds otherwise.

        A phonetic unit reads the text through the tokenizer: every word
        becomes its sounds, punctuation a pause, the gap between two words a
        ``#``.  Text that is already sounds passes through unchanged, and a
        text may mix the two (``"the K AE1 T sat"``).
        """
        if self.unit == CHARS:
            return text
        return phonetic_tokenizer(self.unit).text(text).split()

    def view(self, label: str) -> str | list[str]:
        """A *label* - unit text the tree wrote - as something that slices by unit, without the tokenizer.

        A phonetic label is its tokens joined by single spaces, so splitting on
        them gives the units it was made of; the tokenizer is idempotent on its
        own output, so this is what :meth:`units` would give, faster.
        """
        return label if self.unit == CHARS else label.split()

    def length(self, text: str) -> int:
        """How many units a text holds."""
        return len(self.units(text))

    def normalize(self, text: str) -> str:
        """The text as unit text: itself for characters, its sounds joined by spaces otherwise."""
        return text if self.unit == CHARS else " ".join(self.units(text))

    def piece(self, text: str, lo: int, hi: int | None = None) -> str:
        """Units ``[lo:hi)`` of a text, as text (the text is read through :meth:`units` first)."""
        return _piece(self.units(text), lo, hi)

    def piece_of(self, label: str, lo: int, hi: int | None = None) -> str:
        """Units ``[lo:hi)`` of a label or a prediction (unit text already), as text."""
        return _piece(self.view(label), lo, hi)

    def join(self, *parts: str) -> str:
        """Glue text pieces: nothing between characters, one space between sounds.

        A phonetic piece given as words joins as the sounds it makes, so a
        joined text is all sounds; empty pieces add no separator.
        """
        if self.unit == CHARS:
            return "".join(parts)
        return " ".join(p for p in (" ".join(self.units(p)) for p in parts) if p)

    def join_units(self, *parts: str) -> str:
        """Glue pieces that are unit text already (labels, grams, predictions): the fast :meth:`join`."""
        if self.unit == CHARS:
            return "".join(parts)
        return " ".join(p for p in parts if p)

    def truncate(self, text: str, k: int) -> str:
        """The first ``k`` units of unit text; ``k < 0`` leaves it alone."""
        return text if k < 0 else self.piece_of(text, 0, k)

    def has_unit_prefix(self, label: str, prefix: str) -> bool:
        """Does a label start with ``prefix`` on a unit boundary?  A phonetic prefix is looked for as its sounds."""
        if not prefix:
            return True
        if self.unit == CHARS:
            return label.startswith(prefix)
        prefix = " ".join(self.units(prefix))
        return not prefix or label == prefix or label.startswith(prefix + " ")

    # -- grams ---------------------------------------------------------------

    def encode(self, text: str) -> list[str]:
        """The grams of ``text``; ``[]`` if it holds fewer than ``n`` units.

        The tail that does not fill a whole gram is dropped, exactly as the
        trigram encoding drops nothing but an empty or two-character text.
        """
        view = self.units(text)
        n = self.n
        last = len(view) - n
        if last < 0:
            return []
        return [_piece(view, i, i + n) for i in range(last + 1)]

    def check_grams(self, grams: Sequence[str]) -> None:
        """``ValueError`` unless every gram has ``n`` units and consecutive grams overlap by ``n - 1``."""
        n = self.n
        prev: str | list[str] | None = None
        for g in grams:
            v = self.view(g)
            if len(v) != n:
                raise ValueError(f"expected a gram of {n} {self.units_name}, got {g!r}")
            if prev is not None and prev[1:] != v[:-1]:
                raise ValueError(f"grams {_piece(prev, 0)!r} -> {g!r} do not overlap")
            prev = v

    def last_unit(self, gram: str) -> str:
        """The unit a gram adds to the one before it: its last."""
        return gram[-1] if self.unit == CHARS else gram.rsplit(" ", 1)[-1]

    def first_gram(self, label: str) -> str:
        """The first ``n`` units of a label: what it is keyed by under its parent."""
        return label[: self.n] if self.unit == CHARS else _piece(label.split(), 0, self.n)

    def decode_grams(self, grams: Iterable[str]) -> str:
        """The inverse of :meth:`encode`: the first gram in full, then the new unit of each following one."""
        grams = list(grams)
        if not grams:
            return ""
        ov = self.overlap
        return self.join_units(grams[0], *(self.piece_of(g, ov) for g in grams[1:]))

    def grams_held(self, label: str) -> int:
        """How many grams a label holds: its units less the overlap, and 0 for a label too short to hold one."""
        held = len(self.view(label)) - self.overlap
        return held if held > 0 else 0

    def gram_at(self, label: str, i: int) -> str:
        """Gram number ``i`` of a label."""
        return self.piece_of(label, i, i + self.n)

    def decode_path(self, labels: Iterable[str], start_offset: int = 0, include_context: bool = True) -> str:
        """Decode node labels in path order into unit text.

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
        ov = self.overlap
        parts: list[str] = []
        first = True
        for label in labels:
            parts.append(self.piece_of(label, first_cut if first else ov))
            first = False
        return self.join_units(*parts)

    # -- spelling sounds back ------------------------------------------------

    def spell(self, text: str) -> str:
        """The words a phonetic text spells: ``"DH AH0 # K AE1 T"`` -> ``"the cat"``; other text as it is.

        The tokenizer spells each word back through its lexicon and its memory
        of what it read, and respells sounds no known word has, so a prediction
        made of sounds can be read.
        """
        if not self.phonetic:
            return text
        return phonetic_tokenizer(self.unit).decode(self.units(text))

    def spell_tail(self, whole: str, tail: str) -> str:
        """What ``tail`` - the last units of ``whole`` - spells, as the part of ``spell(whole)`` it wrote.

        A continuation spelled on its own loses the space before its first
        word and cannot finish a word the text before it began, so the whole
        text is spelled and what the text before the tail spells is cut off
        its front; when the tail finished a word the head began, the cut falls
        back to where that word starts.  A tail that does not end ``whole`` is
        spelled on its own.  Character text comes back as it is.
        """
        if not self.phonetic:
            return tail
        if not tail:
            return ""
        if not whole.endswith(tail):
            return self.spell(tail)
        spelled = self.spell(whole)
        head = self.spell(whole[: len(whole) - len(tail)])
        if not spelled.startswith(head):
            head = spelled[: os.path.commonprefix([spelled, head]).rfind(" ") + 1]
        return spelled[len(head):]
