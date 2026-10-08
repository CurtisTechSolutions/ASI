"""The regular languages the machine is taught, and how examples of them are drawn.

A language is a membership function over strings of ``a`` and ``b`` and the
accepting states a machine is given to learn it with (the start state is
``0``).  Each is small enough that a few states suffice and the machine's
credit has to find a transition table, not a lookup.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["ALPHABET", "LANGUAGES", "Language", "examples", "language"]

ALPHABET = "ab"


@dataclass(frozen=True)
class Language:
    name: str
    description: str
    member: Callable[[str], bool]
    accepting: tuple[int, ...]
    """Which states accept.  ``0`` is the start, so a language containing the empty string accepts at ``0``."""
    min_states: int
    """The smallest deterministic machine that recognises it."""


LANGUAGES: dict[str, Language] = {
    lang.name: lang
    for lang in (
        Language("even-b", "an even number of b's", lambda s: s.count("b") % 2 == 0, (0,), 2),
        Language("contains-aa", "two a's in a row somewhere", lambda s: "aa" in s, (2,), 3),
        Language("ends-ab", "ends with ab", lambda s: s.endswith("ab"), (2,), 3),
        Language("mod3-a", "a multiple of three a's", lambda s: s.count("a") % 3 == 0, (0,), 3),
    )
}


def language(name: str) -> Language:
    try:
        return LANGUAGES[name]
    except KeyError:
        raise KeyError(f"{name!r} is not one of {sorted(LANGUAGES)}") from None


def examples(lang: Language, count: int, rng: random.Random, max_length: int = 6,
             alphabet: Sequence[str] = ALPHABET) -> list[tuple[str, bool]]:
    """``count`` strings of length 0 to ``max_length`` drawn uniformly, each with its membership."""
    out = []
    for _ in range(count):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_length)))
        out.append((s, lang.member(s)))
    return out
