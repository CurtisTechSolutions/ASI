"""The pronouncing lexicon: which sounds a spelled word is made of, and which words a sound sequence spells.

The format is the CMU Pronouncing Dictionary's, one entry per line::

    word  PH ON ES
    word(2)  PH ON ES        an alternative pronunciation
    ;;; a comment            (a trailing ``# comment`` is dropped too)

Three sources, layered:

* the **core** lexicon bundled with the package (``data/core.dict``: the
  common words, the numbers, the letters - some 1 900 entries, always there);
* the **full** dictionary, when it can be found: a file named by the
  ``PHONETOK_LEXICON`` environment variable, or the ``cmudict`` package
  (``pip install cmudict``: 135 000 words);
* whatever the caller :meth:`Lexicon.add`\\ s.

A word the lexicon does not hold is sounded out by :mod:`phonetok.rules`.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable, Iterator
from typing import IO

from .phones import check_phones, strip_stress

CORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "core.dict")
ENV_LEXICON = "PHONETOK_LEXICON"

Phones = tuple[str, ...]


def parse_entry(line: str) -> tuple[str, Phones] | None:
    """One line of a CMU-format file as ``(word, phones)``; ``None`` for a comment or a blank."""
    if line.startswith(";;;"):
        return None
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    word = parts[0].lower()
    if word.endswith(")") and "(" in word:  # an alternative: cat(2)
        word = word[:word.rindex("(")]
    return word, tuple(p.upper() for p in parts[1:])


def read_entries(lines: Iterable[str]) -> Iterator[tuple[str, Phones]]:
    for line in lines:
        entry = parse_entry(line)
        if entry is not None:
            yield entry


_CORE_CACHE: list[tuple[str, Phones]] | None = None


def core_entries() -> list[tuple[str, Phones]]:
    """The bundled core lexicon's entries, parsed once per process."""
    global _CORE_CACHE
    if _CORE_CACHE is None:
        with open(CORE_PATH, encoding="utf-8") as f:
            _CORE_CACHE = list(read_entries(f))
    return list(_CORE_CACHE)


def find_full_dictionary(package: bool = True) -> tuple[str, IO[str] | None] | None:
    """Where the full dictionary is, if anywhere: ``(description, open file)``.

    The ``PHONETOK_LEXICON`` variable names a file; failing that the ``cmudict``
    package is used if it is installed (and ``package`` is true).  ``None`` when
    neither is there.
    """
    path = os.environ.get(ENV_LEXICON)
    if path:
        return f"file {path}", open(path, encoding="utf-8")
    if package and importlib.util.find_spec("cmudict") is not None:
        import cmudict  # type: ignore

        return "the cmudict package", cmudict.dict_stream()
    return None


class Lexicon:
    """A word -> pronunciations map, and its inverse.

    ``Lexicon()`` is empty; :meth:`core` is the bundled subset; :meth:`default`
    is the core plus the full dictionary when one can be found.  Every method
    lower-cases the word it is given, since the dictionary is lower case.
    """

    __slots__ = ("_entries", "_reverse", "_reverse_loose", "sources")

    def __init__(self, entries: Iterable[tuple[str, Iterable[str]]] = (), source: str | None = None) -> None:
        self._entries: dict[str, list[Phones]] = {}
        self._reverse: dict[Phones, list[str]] | None = None
        self._reverse_loose: dict[Phones, list[str]] | None = None
        self.sources: list[str] = []
        self.extend(entries)
        if source:
            self.sources.append(source)

    # -- construction --------------------------------------------------------

    @classmethod
    def core(cls) -> "Lexicon":
        """The bundled core lexicon."""
        return cls(core_entries(), source="the core lexicon")

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        """A CMU-format file."""
        with open(path, encoding="utf-8") as f:
            return cls(read_entries(f), source=f"file {path}")

    @classmethod
    def default(cls, full: bool = True, package: bool = True) -> "Lexicon":
        """The core lexicon, plus the full dictionary if ``full`` and one can be found.

        ``package=False`` ignores the ``cmudict`` package and reads only the file named by
        ``PHONETOK_LEXICON``: the two sources the Go and Rust ports can read too.
        """
        lex = cls.core()
        if full:
            found = find_full_dictionary(package=package)
            if found is not None:
                name, stream = found
                try:
                    lex.extend(read_entries(stream))
                finally:
                    if stream is not None:
                        stream.close()
                lex.sources.append(name)
        return lex

    @classmethod
    def portable(cls) -> "Lexicon":
        """The lexicon every port builds alike: the core, plus the ``PHONETOK_LEXICON`` file if it is set.

        A model whose symbols are sounds is only as portable as the lexicon that made them, so
        a Python model meant for Go or Rust is transcribed through this one.
        """
        return cls.default(full=True, package=False)

    def extend(self, entries: Iterable[tuple[str, Iterable[str]]]) -> int:
        """Add entries; a pronunciation the word already has is not added twice.  Returns how many were new."""
        added = 0
        table = self._entries
        for word, phones in entries:
            phones = tuple(phones)
            word = word.lower()
            known = table.get(word)
            if known is None:
                table[word] = [phones]
                added += 1
            elif phones not in known:
                known.append(phones)
                added += 1
        if added:
            self._reverse = self._reverse_loose = None
        return added

    def add(self, word: str, phones: Iterable[str], first: bool = False) -> None:
        """Add a pronunciation; ``first`` makes it the one :meth:`lookup` returns."""
        phones = tuple(phones)
        check_phones(phones)
        word = word.lower()
        known = self._entries.setdefault(word, [])
        if phones in known:
            known.remove(phones)
        if first:
            known.insert(0, phones)
        else:
            known.append(phones)
        self._reverse = self._reverse_loose = None

    def remove(self, word: str) -> bool:
        gone = self._entries.pop(word.lower(), None) is not None
        if gone:
            self._reverse = self._reverse_loose = None
        return gone

    # -- lookup --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word.lower() in self._entries

    def words(self) -> Iterator[str]:
        return iter(self._entries)

    def items(self) -> Iterator[tuple[str, Phones]]:
        """Every ``(word, pronunciation)`` pair, alternatives included."""
        for word, variants in self._entries.items():
            for phones in variants:
                yield word, phones

    def lookup(self, word: str) -> Phones | None:
        """The first pronunciation of ``word``, or ``None``."""
        variants = self._entries.get(word.lower())
        return variants[0] if variants else None

    def pronunciations(self, word: str) -> list[Phones]:
        """Every pronunciation of ``word``; ``[]`` if it is unknown."""
        return list(self._entries.get(word.lower(), ()))

    # -- the inverse ---------------------------------------------------------

    def _build_reverse(self) -> None:
        exact: dict[Phones, list[str]] = {}
        loose: dict[Phones, list[str]] = {}
        for word, variants in self._entries.items():
            for phones in variants:
                exact.setdefault(phones, []).append(word)
                loose.setdefault(tuple(strip_stress(phones)), []).append(word)
        self._reverse, self._reverse_loose = exact, loose

    def spellings(self, phones: Iterable[str], stress: bool = True) -> list[str]:
        """The words pronounced exactly like ``phones`` (homophones), in the order they were added.

        With ``stress=False`` the stress digits are ignored on both sides.
        """
        if self._reverse is None:
            self._build_reverse()
        assert self._reverse is not None and self._reverse_loose is not None
        phones = tuple(phones)
        if stress:
            return list(self._reverse.get(phones, ()))
        return list(self._reverse_loose.get(tuple(strip_stress(phones)), ()))

    def describe(self) -> str:
        return f"{len(self)} words from " + (", ".join(self.sources) if self.sources else "nowhere")

    def __repr__(self) -> str:
        return f"Lexicon({self.describe()})"
