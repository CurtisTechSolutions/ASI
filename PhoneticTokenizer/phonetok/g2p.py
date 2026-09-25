"""Grapheme to phoneme: a spelled word in, its sounds out - and the naive way back.

:class:`Transcriber` answers *how does this word sound?* by trying, in order:

1. the **lexicon** (:mod:`phonetok.lexicon`);
2. its **memory** of words it has already sounded out;
3. **numbers** (:mod:`phonetok.numbers`): ``42`` is *forty two*;
4. **joins**: ``well-known``, ``a/b``, ``u.s.`` are read a part at a time;
5. **morphology**: an unknown word whose stem is known - ``cats``, ``walked``,
   ``running``, ``bigger``, ``quickly``, ``unhappy``, ``toothbrush`` - is the
   stem's sounds with the affix's, and the ``-s`` / ``-ed`` endings take the
   voicing the stem's last sound decides;
6. **letters**: a word with no vowel letters (``xq``, ``csr``) and an unknown
   word in capitals (``FBI``) are spelled out by letter name;
7. the **rules** (:mod:`phonetok.rules`) for everything else.

Every word that reaches step 3 or later is remembered with its sounds, so that
:meth:`Transcriber.spell` (sounds in, a word out) can give the spelling back.
:func:`respell` is the last resort of that direction: a readable, invented
spelling of sounds no known word has (``K AE1 T`` -> ``cat``, ``S T R EH1 NG TH``
-> ``strength``).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .lexicon import Lexicon
from .numbers import number_words, split_alphanumeric
from .phones import (
    CONSONANTS, VOWELS, base, is_vowel, past_suffix, plural_suffix, stress_of, strip_stress, with_stress,
)
from .rules import letter_to_sound

Phones = list[str]

_LETTERS = re.compile(r"[a-z]")
_JOINERS = re.compile(r"[-/_.]+")

SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ness", ("N", "AH0", "S")), ("ment", ("M", "AH0", "N", "T")), ("less", ("L", "AH0", "S")),
    ("ful", ("F", "AH0", "L")), ("able", ("AH0", "B", "AH0", "L")), ("ish", ("IH0", "SH")),
    ("ly", ("L", "IY0")), ("er", ("ER0",)), ("est", ("IH0", "S", "T")), ("ing", ("IH0", "NG")),
    ("y", ("IY0",)),
)
"""Suffixes with fixed sounds, tried on an unknown word whose remainder is a known stem."""

PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("under", ("AH2", "N", "D", "ER0")), ("inter", ("IH2", "N", "T", "ER0")), ("super", ("S", "UW2", "P", "ER0")),
    ("multi", ("M", "AH2", "L", "T", "IY0")), ("semi", ("S", "EH2", "M", "IY0")), ("over", ("OW2", "V", "ER0")),
    ("anti", ("AE2", "N", "T", "IY0")), ("auto", ("AO2", "T", "OW0")), ("micro", ("M", "AY2", "K", "R", "OW0")),
    ("non", ("N", "AA2", "N")), ("out", ("AW2", "T")), ("pre", ("P", "R", "IY2")), ("dis", ("D", "IH0", "S")),
    ("mis", ("M", "IH0", "S")), ("sub", ("S", "AH0", "B")), ("un", ("AH0", "N")), ("re", ("R", "IY0",)),
    ("de", ("D", "IY0",)), ("co", ("K", "OW0",)),
)
"""Prefixes with fixed sounds, tried the same way."""

CONTRACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("n't", ("N", "T")), ("'ll", ("L",)), ("'re", ("R",)), ("'ve", ("V",)), ("'d", ("D",)), ("'m", ("M",)),
)
"""Clitics: after a vowel they add just the consonant (*I'll*, *you've*); after a consonant a schwa comes first
(*it'll*, *would've*), and ``'s`` / ``s'`` take the voicing the stem's last sound decides (*cat's*, *dog's*)."""


class Transcriber:
    """Words to sounds, through the lexicon, the morphology and the rules; and sounds back to words."""

    def __init__(self, lexicon: Lexicon | None = None, rules: bool = True, remember: bool = True) -> None:
        self.lexicon = lexicon if lexicon is not None else Lexicon.default()
        self.use_rules = rules
        self.remember = remember
        self.memory: dict[str, tuple[str, ...]] = {}
        """Words the lexicon did not hold, with the sounds they were given."""
        self.counts: Counter[str] = Counter()
        """How often each spelled word has been transcribed: the tie-breaker among homophones."""

    # -- forward -------------------------------------------------------------

    def word(self, word: str) -> Phones:
        """The sounds of one spelled word (no spaces); ``[]`` if it has none."""
        return self.explain(word)[0]

    def explain(self, word: str) -> tuple[Phones, str]:
        """The sounds of a word and where they came from: ``lexicon``, ``memory``, ``number``,
        ``joined``, ``morphology``, ``letters``, ``rules`` or ``none``."""
        raw = word
        w = word.strip("'\"`‘’“”").lower()
        if not w:
            return [], "none"
        self.counts[w] += 1
        found = self.lexicon.lookup(w)
        if found is not None:
            return list(found), "lexicon"
        if w in self.memory:
            return list(self.memory[w]), "memory"
        phones, how = self._derive(w, raw)
        if phones and self.remember and how != "none":
            self.memory[w] = tuple(phones)
        return phones, how

    def _derive(self, w: str, raw: str) -> tuple[Phones, str]:
        if any(ch.isdigit() for ch in w):
            words = number_words(w)
            if words is not None:
                return self._join(words.split()), "number"
            return self._join(split_alphanumeric(w)), "number"
        if _JOINERS.search(w):
            parts = [p for p in _JOINERS.split(w) if p]
            if len(parts) > 1:
                return self._join(parts), "joined"
            w = parts[0] if parts else ""
            if not w:
                return [], "none"
        phones = self._contraction(w)
        if phones is not None:
            return phones, "morphology"
        phones = self._morphology(w)
        if phones is not None:
            return phones, "morphology"
        if not _LETTERS.search(w):
            return [], "none"
        if not any(ch in "aeiouy" for ch in w) or (raw.isupper() and 2 <= len(raw) <= 5):
            return self._letters(w), "letters"
        if not self.use_rules:
            return [], "none"
        return letter_to_sound(re.sub(r"[^a-z']", "", w)), "rules"

    def _join(self, parts: Sequence[str]) -> Phones:
        out: Phones = []
        for part in parts:
            phones, _ = self.explain(part)
            out.extend(phones)
        return out

    def _letters(self, w: str) -> Phones:
        out: Phones = []
        for ch in w:
            if ch.isalpha():
                found = self.lexicon.pronunciations(ch)
                # a letter's name is its stressed reading (the article "a" is AH0; the letter is EY1)
                named = [p for p in found if any(stress_of(x) == 1 for x in p)]
                out.extend(named[0] if named else (found[0] if found else letter_to_sound(ch)))
        return out

    def _contraction(self, w: str) -> Phones | None:
        if "'" not in w:
            return None
        for ending in ("'s", "s'"):
            if w.endswith(ending) and len(w) > 2:
                stem_phones = self._stem(w[:-2] + ("s" if ending == "s'" else ""))
                if stem_phones is not None:
                    return stem_phones + ([] if ending == "s'" else plural_suffix(stem_phones))
        for ending, sounds in CONTRACTIONS:
            if w.endswith(ending) and len(w) > len(ending):
                stem_phones = self._stem(w[: -len(ending)])
                if stem_phones is None:
                    continue
                if ending == "n't" and stem_phones and base(stem_phones[-1]) == "N":
                    return stem_phones + ["T"]  # can't, won't, don't: the stem already ends in the N
                if stem_phones and is_vowel(stem_phones[-1]) and ending != "n't":
                    return stem_phones + list(sounds)
                if ending == "'d" and stem_phones and base(stem_phones[-1]) in ("T", "D"):
                    return stem_phones + ["AH0", "D"]
                if ending == "'d" and stem_phones and not is_vowel(stem_phones[-1]):
                    return stem_phones + ["D"]
                return stem_phones + ["AH0"] + list(sounds)
        return None

    def _stem(self, stem: str) -> Phones | None:
        """The sounds of a candidate stem if it is a known word, else ``None``."""
        found = self.lexicon.lookup(stem)
        if found is not None:
            return list(found)
        if stem in self.memory:
            return list(self.memory[stem])
        return None

    def _stems(self, stem: str, suffix_starts_with_vowel: bool) -> list[str]:
        """The spellings a stem may have had before a suffix was added."""
        candidates = [stem]
        if suffix_starts_with_vowel:
            candidates.append(stem + "e")  # mak-ing -> make, lov-ed -> love
            if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] in "bdfgklmnprstz":
                candidates.append(stem[:-1])  # runn-ing -> run, stopp-ed -> stop
            if stem.endswith("i"):
                candidates.append(stem[:-1] + "y")  # carri-ed -> carry, happi-er -> happy
        return candidates

    def _morphology(self, w: str) -> Phones | None:
        if len(w) < 4:
            return None
        # plurals and third persons: cats, boxes, cities, goes
        if w.endswith("ies") and len(w) > 4:
            stem = self._stem(w[:-3] + "y")
            if stem is not None:
                return stem + plural_suffix(stem)
        if w.endswith("es") and len(w) > 3:
            stem = self._stem(w[:-2])
            if stem is not None:
                if stem and base(stem[-1]) in ("S", "Z", "SH", "ZH", "CH", "JH"):
                    return stem + ["IH0", "Z"]
                return stem + plural_suffix(stem)
        if w.endswith("s") and not w.endswith("ss"):
            stem = self._stem(w[:-1])
            if stem is not None:
                return stem + plural_suffix(stem)
        # the past: walked, loved, carried, stopped
        if w.endswith("ed") and len(w) > 4:
            for candidate in self._stems(w[:-2], True):
                stem = self._stem(candidate)
                if stem is not None:
                    return stem + past_suffix(stem)
        # the suffixes with fixed sounds
        for suffix, sounds in SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                for candidate in self._stems(w[: -len(suffix)], suffix[0] in "aeiouy"):
                    stem = self._stem(candidate)
                    if stem is not None:
                        if suffix == "ly" and stem and stem[-1] == "L":
                            return stem + list(sounds)[1:]  # real-ly, full-y: one L, as the dictionary says
                        return stem + list(sounds)
        # the prefixes
        for prefix, sounds in PREFIXES:
            if w.startswith(prefix) and len(w) - len(prefix) >= 3:
                stem = self._stem(w[len(prefix):])
                if stem is not None:
                    return list(sounds) + stem
        # a compound of two known words: toothbrush, keyboard
        best: tuple[float, Phones] | None = None
        for cut in range(3, len(w) - 2):
            left, right = self._stem(w[:cut]), self._stem(w[cut:])
            if left is not None and right is not None:
                demoted = [with_stress(p, 2) if stress_of(p) == 1 else p for p in right]
                balance = abs(cut - len(w) / 2)  # the cut nearest the middle wins
                if best is None or balance < best[0]:
                    best = (balance, left + demoted)
        return best[1] if best else None

    # -- backward ------------------------------------------------------------

    def spell(self, phones: Sequence[str]) -> str:
        """A word that sounds like ``phones``: a remembered one, a lexicon one, or a respelling.

        Among homophones the one transcribed most often wins, then the shorter, then the
        alphabetically first; stress is matched exactly first and ignored second.
        """
        phones = list(phones)
        if not phones:
            return ""
        key = tuple(phones)
        remembered = [w for w, p in self.memory.items() if p == key]
        candidates = remembered or self.lexicon.spellings(phones)
        if not candidates:
            loose = tuple(strip_stress(phones))
            remembered = [w for w, p in self.memory.items() if tuple(strip_stress(p)) == loose]
            candidates = remembered or self.lexicon.spellings(phones, stress=False)
        if candidates:
            return min(candidates, key=lambda w: (-self.counts[w], len(w), w))
        return respell(phones)



# ---------------------------------------------------------------------------
# respelling: sounds to an invented, readable spelling
# ---------------------------------------------------------------------------

_CONSONANT_SPELLING = {
    "B": "b", "CH": "ch", "D": "d", "DH": "th", "F": "f", "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}
_VOWEL_SPELLING = {
    "AA": "o", "AE": "a", "AH": "u", "AO": "aw", "AW": "ow", "AY": "i", "EH": "e", "ER": "er", "EY": "ay",
    "IH": "i", "IY": "ee", "OW": "o", "OY": "oy", "UH": "oo", "UW": "oo",
}
_TENSE_MAGIC = {"EY": "a", "AY": "i", "OW": "o", "UW": "u"}
_MAGIC_OK = frozenset("bdfgklmnprstvz")
_FRONT = frozenset({"EH", "IY", "IH", "EY", "AY"})


def respell(phones: Sequence[str]) -> str:
    """An invented spelling that reads back as ``phones``: ``K AE1 T -> cat``, ``M EY1 K -> make``.

    English conventions are followed where they are regular: *c* before a back vowel and
    *k* before a front one, the magic *e* after a tense vowel and one consonant, a doubled
    consonant after a stressed lax vowel, *y* for a final unstressed *ee*.
    """
    phones = list(phones)
    out: list[str] = []
    n = len(phones)
    for i, p in enumerate(phones):
        b = base(p)
        nxt = phones[i + 1] if i + 1 < n else None
        nxt_b = base(nxt) if nxt else None
        after = base(phones[i + 2]) if i + 2 < n else None
        if b in CONSONANTS:
            if b == "K":
                soft = nxt_b in _VOWEL_SPELLING and nxt_b not in _FRONT or nxt_b in ("L", "R", "Y")
                out.append("c" if soft else "k")
                if nxt is None and i and base(phones[i - 1]) in ("AE", "EH", "IH", "AH") and stress_of(phones[i - 1]) != 0:
                    out[-1] = "ck"
            elif b == "Y" and nxt_b == "UW":
                out.append("")  # the *u* below spells the glide
            else:
                out.append(_CONSONANT_SPELLING[b])
            # a doubled consonant after a stressed lax vowel, before a vowel
            prev = base(phones[i - 1]) if i else None
            if (
                prev in ("AE", "EH", "IH", "AH", "UH") and stress_of(phones[i - 1]) in (1, 2)
                and nxt_b in _VOWEL_SPELLING and b in ("B", "D", "G", "L", "M", "N", "P", "R", "S", "T", "Z")
            ):
                out[-1] = out[-1] + out[-1]
            continue
        # a vowel
        if b == "UW" and i and base(phones[i - 1]) == "Y":
            magic = nxt_b in CONSONANTS and after is None and _CONSONANT_SPELLING[nxt_b] in _MAGIC_OK
            spelling = "u\0" if magic else "u"
        elif b == "IY" and nxt is None and stress_of(p) == 0 and i:
            spelling = "y"
        elif b == "AY" and nxt is None:
            spelling = "y" if i else "i"
        elif b == "EY" and nxt is None:
            spelling = "ay"
        elif b == "OW" and nxt is None:
            spelling = "o"
        elif b in _TENSE_MAGIC and nxt_b in CONSONANTS and after is None and _CONSONANT_SPELLING[nxt_b] in _MAGIC_OK:
            spelling = _TENSE_MAGIC[b] + "\0"  # the magic e goes after the consonant
        elif b in _TENSE_MAGIC and nxt_b in CONSONANTS and after in _VOWEL_SPELLING and b != "IY":
            spelling = _TENSE_MAGIC[b]  # pa-per: one consonant then a vowel reads the vowel long
        elif b == "AH" and stress_of(p) == 0:
            spelling = "a" if (nxt is None or i == 0) else "u"
        else:
            spelling = _VOWEL_SPELLING[b]
        out.append(spelling)
    text = "".join(out)
    if "\0" in text:
        i = text.index("\0")
        text = text[:i] + text[i + 1:i + 2] + "e" + text[i + 2:]
        text = text.replace("\0", "")
    return text
