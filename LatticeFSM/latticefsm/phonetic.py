"""English in and out of a network, through the sounds it is made of.

The phonetic tokenizer of this repository (``../../PhoneticTokenizer``,
``phonetok``) reads English as ARPAbet phonemes - ``"cats"`` is ``K AE T S`` -
and spells phonemes back as English.  :class:`~latticefsm.network.PhoneticEncoder`
and :class:`~latticefsm.network.PhoneticDecoder` put it at the two ends of a
network: English goes in, the machines walk its sounds, and what comes out
is spelled back as English.

The sounds are the 39 phonemes without their stress digits (``AH``, not
``AH0`` / ``AH1``; ``stress=True`` keeps them), the word boundary ``#`` and
the three pauses ``, . ?`` punctuation becomes: 43 symbols.  The tokenizer
is the portable one - the bundled core lexicon and the letter-to-sound rules,
plus the file ``PHONETOK_LEXICON`` names - so the Rust crate, which uses the
same library's Rust port, hears every text the same way.

:data:`PLURALS` and :data:`PAST` are English-to-English tasks for a network
with both ends: a noun to its plural (``cat`` -> ``cats``, ``bus`` -> ``buses``)
and a verb to its past tense (``walk`` -> ``walked``, ``want`` -> ``wanted``),
each the same sounds with one or two more at the end, chosen by the sound
before them.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

__all__ = ["PAST", "PLURALS", "TASKS", "inventory", "split", "tokenizer"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBRARY = os.path.normpath(os.path.join(_HERE, "..", "..", "PhoneticTokenizer"))


def _phonetok():
    try:
        import phonetok
    except ImportError:
        if _LIBRARY not in sys.path:
            sys.path.insert(0, _LIBRARY)
        import phonetok
    return phonetok


_TOKENIZERS: dict[bool, object] = {}


def tokenizer(stress: bool = False):
    """The phonetic tokenizer (phoneme level, the portable lexicon), one per stress setting, made once."""
    if stress not in _TOKENIZERS:
        pt = _phonetok()
        _TOKENIZERS[stress] = pt.PhoneticTokenizer(stress=stress, lexicon=pt.Lexicon.default(package=False))
    return _TOKENIZERS[stress]


def inventory(stress: bool = False) -> list[str]:
    """The sounds, in a fixed order: ``#``, the pauses ``, . ?``, then the phonemes (alphabetical; with ``stress``,
    every vowel as ``AA AA0 AA1 AA2``, the tokenizer's own id order)."""
    pt = _phonetok()
    if stress:
        return [s for s in pt.SYMBOLS if s not in pt.SPECIALS]
    return [pt.BOUNDARY, *pt.PAUSES, *pt.PHONEMES]


def _pairs(a: str, b: str) -> tuple[tuple[str, str], ...]:
    xs, ys = a.split(), b.split()
    assert len(xs) == len(ys)
    return tuple(zip(xs, ys))


PLURALS = _pairs(
    "cat dog bus hat cup bed pig box rose sock map bag fox kiss pen car bird book ship tree cake bell lamp star fish "
    "dish wish bench horse key ring frog bat hand rock sea door song boat bike nose glass page road kite drum",
    "cats dogs buses hats cups beds pigs boxes roses socks maps bags foxes kisses pens cars birds books ships trees "
    "cakes bells lamps stars fishes dishes wishes benches horses keys rings frogs bats hands rocks seas doors songs "
    "boats bikes noses glasses pages roads kites drums")
"""Nouns and their plurals: ``-s`` after a voiceless sound, ``-z`` after a voiced one, ``-iz`` after a hiss."""

PAST = _pairs(
    "walk play want jump call need talk look help cook kiss wash fill rain love open start end paint wait pull push "
    "dance miss clean kick smile park",
    "walked played wanted jumped called needed talked looked helped cooked kissed washed filled rained loved opened "
    "started ended painted waited pulled pushed danced missed cleaned kicked smiled parked")
"""Verbs and their past tenses: ``-t`` after a voiceless sound, ``-d`` after a voiced one, ``-id`` after t or d."""

TASKS = {"plural": PLURALS, "past": PAST}


def split(pairs: Sequence[tuple[str, str]], every: int = 4) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Every ``every``-th pair held out for testing, the rest for teaching."""
    train = [p for i, p in enumerate(pairs) if i % every != every - 1]
    test = [p for i, p in enumerate(pairs) if i % every == every - 1]
    return train, test
