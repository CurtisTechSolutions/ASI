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
BACK_LABEL = "<back>"
"""The third sentinel: where the graph has learned that a walk goes round (``graph.BACK``)."""


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
            if skip_sentinels and label in (START_LABEL, END_LABEL, BACK_LABEL):
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


# ---------------------------------------------------------------------------
# the word alphabet (see ``../SPEC-WordNGrams.md``)
# ---------------------------------------------------------------------------

WORD_BASE = 0x0100
"""The first word symbol: past ASCII and past Latin-1, so no label can be mistaken for text."""

SURROGATE_LO = 0xD800
SURROGATE_HI = 0xDFFF
_SURROGATES = SURROGATE_HI - SURROGATE_LO + 1

MAX_WORDS = 0x110000 - WORD_BASE - _SURROGATES
"""1 111 808: every code point above Latin-1 that a string can hold, the surrogate block excepted."""

UNKNOWN_WORD = "<unk>"
UNKNOWN_ID = 0
"""Id 0 is the unknown word: a word the model has never read maps to it at prediction and scoring time."""

_NOT_UNICODE_SPACE = "\x1c\x1d\x1e\x1f"
"""Python calls these whitespace; the Unicode ``White_Space`` property (Go and Rust) does not.

The three implementations have to split a text into the *same* words, so the
rule is the Unicode property and Python subtracts the four separators it adds.
"""


def word_symbol(word_id: int) -> str:
    """The code point that carries word ``word_id`` (``0 -> U+0100``, skipping the surrogates)."""
    if not 0 <= word_id < MAX_WORDS:
        raise ValueError(f"word id must lie in [0, {MAX_WORDS}), got {word_id}")
    code = WORD_BASE + word_id
    if code >= SURROGATE_LO:
        code += _SURROGATES
    return chr(code)


def symbol_word(symbol: str) -> int:
    """Inverse of :func:`word_symbol`; ``ValueError`` for a code point that carries no word."""
    if len(symbol) != 1:
        raise ValueError(f"a word symbol is one code point, got {symbol!r}")
    code = ord(symbol)
    if code < WORD_BASE or SURROGATE_LO <= code <= SURROGATE_HI:
        raise ValueError(f"U+{code:04X} is not a word symbol")
    return code - WORD_BASE - (_SURROGATES if code > SURROGATE_HI else 0)


def split_words(text: str) -> list[str]:
    """The words of ``text``: maximal runs of code points the Unicode ``White_Space`` property does not cover.

    Punctuation stays attached to the word it touches and case is kept - the
    split is the whole tokeniser (``SPEC-WordNGrams.md`` §4), and every
    refinement of it would be a step towards a vocabulary that has to be
    designed, versioned and defended.
    """
    if not any(ch in text for ch in _NOT_UNICODE_SPACE):
        return text.split()  # the common case: no code point the two rules disagree about
    words: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isspace() and ch not in _NOT_UNICODE_SPACE:
            if current:
                words.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        words.append("".join(current))
    return words


class Vocabulary:
    """The alphabet of a word model: a two-way map between words and single code points.

    A word model is the count / reward model over an alphabet whose symbols are
    words (``../SPEC-WordNGrams.md``), and this is that alphabet.  Id 0 is
    always :data:`UNKNOWN_WORD`; every other word is in the vocabulary because
    training read it, in the order it was first read.  Nothing is frozen,
    pruned or learned: there is no tokeniser here and no training run before
    the training run.
    """

    __slots__ = ("words", "_ids")

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.words: list[str] = [UNKNOWN_WORD]
        self._ids: dict[str, int] = {UNKNOWN_WORD: UNKNOWN_ID}
        if words is not None:
            items = list(words)
            if not items:
                return
            if items[0] != UNKNOWN_WORD:
                raise ValueError(f"a vocabulary starts with {UNKNOWN_WORD!r}, got {items[0]!r}")
            for word in items[1:]:
                if not isinstance(word, str):
                    raise TypeError(f"a vocabulary holds strings, got {type(word).__name__}")
                if word in self._ids:
                    raise ValueError(f"the vocabulary holds {word!r} twice")
                self.add(word)

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: object) -> bool:
        return word in self._ids

    def id(self, word: str) -> int:
        """The id of ``word``, or :data:`UNKNOWN_ID` if it has never been read."""
        return self._ids.get(word, UNKNOWN_ID)

    def add(self, word: str) -> int:
        """The id of ``word``, giving it the next one if it is new."""
        found = self._ids.get(word)
        if found is not None:
            return found
        if len(self.words) >= MAX_WORDS:
            raise ValueError(f"a word model holds at most {MAX_WORDS} words")
        new = len(self.words)
        self.words.append(word)
        self._ids[word] = new
        return new

    def encode(self, text: str, grow: bool = False) -> str:
        """``text`` as the graph's symbols, one code point per word.

        ``grow`` gives an unread word the next id (training); without it an
        unread word is :data:`UNKNOWN_WORD`, whose windows are not in the index,
        so the search and the score charge it what they charge any unknown
        transition.
        """
        pick = self.add if grow else self.id
        return "".join(word_symbol(pick(word)) for word in split_words(text))

    def decode(self, symbols: str) -> str:
        """The words of a symbol string, joined by single spaces (a word model normalises whitespace)."""
        words = self.words
        out: list[str] = []
        for symbol in symbols:
            index = symbol_word(symbol)
            if index >= len(words):
                raise ValueError(f"word {index} is past the end of a vocabulary of {len(words)}")
            out.append(words[index])
        return " ".join(out)

    def to_list(self) -> list[str]:
        """The vocabulary in id order, as the model file carries it."""
        return list(self.words)

    def __repr__(self) -> str:
        return f"Vocabulary(words={len(self.words)})"
