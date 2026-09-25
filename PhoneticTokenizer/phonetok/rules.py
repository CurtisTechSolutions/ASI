"""Letter-to-sound rules: how a word the dictionary does not know is sounded out.

English spelling is not phonetic, so no rule set reads it perfectly - but a
few hundred context rules of the classic form

    LEFT [ MATCH ] RIGHT = PHONES

read it *well enough* for a word nobody has listed, and they need no data at
all.  The form is the Naval Research Laboratory's (Elovitz, Johnson, McHugh
and Shore, 1976: "Automatic translation of English text to phonetics by means
of letter-to-sound rules"); the rules below are this package's own, written
to that form and producing ARPAbet, and tuned against the CMU dictionary
(``tests/test_rules.py`` pins the accuracy floor).

A rule fires when ``MATCH`` is the text at the cursor, ``LEFT`` matches what
comes before it and ``RIGHT`` what comes after; the rules of a letter are tried
in order and the first to fire wins, so the specific ones come first and the
bare letter last.  The context symbols:

| symbol | matches |
|---|---|
| `#` | one or more vowels (a e i o u y) |
| `:` | zero or more consonants |
| `^` | exactly one consonant |
| `.` | one voiced consonant (b d g j l m n r v w z) |
| `+` | one front vowel (e i y) |
| `&` | one sibilant: s, c, g, z, x, j, ch, sh |
| `@` | one of t s r d l z n j th ch sh - after these, *u* is /uw/ and not /yuw/ |
| `%` | a suffix: -e, -er, -es, -ed, -ing, -ely (then the end of the word, or an -s) |
| ` ` (space) | the edge of the word |
| a letter | itself |

The vowels a rule writes carry a stress digit only where the rule *knows* it
(``AH0`` for a schwa); :func:`assign_stress` places the primary stress
afterwards, by the shape of the word.  :func:`letter_to_sound` is the whole
thing: a spelled word in, phones with stress out.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .phones import VOWELS, base, is_vowel, stress_of

LETTER_VOWELS = frozenset("AEIOUY")
LETTER_CONSONANTS = frozenset("BCDFGHJKLMNPQRSTVWXZ")
LETTER_VOICED = frozenset("BDGJLMNRVWZ")
LETTER_FRONT = frozenset("EIY")
LETTER_SIBILANT_ONE = frozenset("SCGZXJ")
LETTER_SIBILANT_TWO = ("CH", "SH")
LETTER_U_PLAIN_ONE = frozenset("TSRDLZNJ")
LETTER_U_PLAIN_TWO = ("TH", "CH", "SH")
SUFFIXES = ("ING", "ELY", "ER", "ES", "ED", "E")


@dataclass(frozen=True)
class Rule:
    """One rewrite rule, parsed from ``LEFT[MATCH]RIGHT=PHONES``."""

    left: str
    match: str
    right: str
    phones: tuple[str, ...]
    source: str

    @classmethod
    def parse(cls, text: str) -> "Rule":
        m = re.fullmatch(r"(.*)\[(.+?)\](.*?)=(.*)", text)
        if not m:
            raise ValueError(f"not a rule: {text!r}")
        left, match, right, phones = m.groups()
        return cls(left, match, right, tuple(phones.split()), text)


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rules.lts")
"""The rule table, a data file so that the Go and Rust ports read the very same rules."""


def _read_rule_text(path: str = RULES_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def _parse_rules(text: str) -> dict[str, list[Rule]]:
    table: dict[str, list[Rule]] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        rule = Rule.parse(line)
        table.setdefault(rule.match[0], []).append(rule)
    return table


RULES: dict[str, list[Rule]] = _parse_rules(_read_rule_text())
"""The rules, grouped by the first letter of their match, in the order they are tried."""


def rule_count() -> int:
    return sum(len(rules) for rules in RULES.values())


# ---------------------------------------------------------------------------
# the matcher
# ---------------------------------------------------------------------------

def _is_vowel_letter(ch: str) -> bool:
    return ch in LETTER_VOWELS


def _is_consonant_letter(ch: str) -> bool:
    return ch in LETTER_CONSONANTS


def _match_right(word: str, pos: int, pattern: str, k: int) -> bool:
    """Does ``pattern[k:]`` match ``word`` forwards from ``pos``?  Backtracks over the quantifiers."""
    if k == len(pattern):
        return True
    sym = pattern[k]
    if sym == "#":  # one or more vowels
        j = pos
        while j < len(word) and _is_vowel_letter(word[j]):
            j += 1
            if _match_right(word, j, pattern, k + 1):
                return True
        return False
    if sym == ":":  # zero or more consonants
        j = pos
        while True:
            if _match_right(word, j, pattern, k + 1):
                return True
            if j < len(word) and _is_consonant_letter(word[j]):
                j += 1
            else:
                return False
    if pos >= len(word):
        return False
    ch = word[pos]
    if sym == "^":
        return _is_consonant_letter(ch) and _match_right(word, pos + 1, pattern, k + 1)
    if sym == ".":
        return ch in LETTER_VOICED and _match_right(word, pos + 1, pattern, k + 1)
    if sym == "+":
        return ch in LETTER_FRONT and _match_right(word, pos + 1, pattern, k + 1)
    if sym == "&":
        if word.startswith(LETTER_SIBILANT_TWO, pos):
            if _match_right(word, pos + 2, pattern, k + 1):
                return True
        return ch in LETTER_SIBILANT_ONE and _match_right(word, pos + 1, pattern, k + 1)
    if sym == "@":
        if word.startswith(LETTER_U_PLAIN_TWO, pos):
            if _match_right(word, pos + 2, pattern, k + 1):
                return True
        return ch in LETTER_U_PLAIN_ONE and _match_right(word, pos + 1, pattern, k + 1)
    if sym == "%":
        for suffix in SUFFIXES:
            if word.startswith(suffix, pos):
                end = pos + len(suffix)
                rest = word[end:]
                if rest in (" ", "S ") and _match_right(word, len(word), pattern, k + 1):
                    return True
        return False
    return ch == sym and _match_right(word, pos + 1, pattern, k + 1)


def _match_left(word: str, pos: int, pattern: str, k: int) -> bool:
    """Does ``pattern[:k]`` match ``word`` backwards, ending just before ``pos``?"""
    if k == 0:
        return True
    sym = pattern[k - 1]
    if sym == "#":
        j = pos
        while j > 0 and _is_vowel_letter(word[j - 1]):
            j -= 1
            if _match_left(word, j, pattern, k - 1):
                return True
        return False
    if sym == ":":
        j = pos
        while True:
            if _match_left(word, j, pattern, k - 1):
                return True
            if j > 0 and _is_consonant_letter(word[j - 1]):
                j -= 1
            else:
                return False
    if pos <= 0:
        return False
    ch = word[pos - 1]
    if sym == "^":
        return _is_consonant_letter(ch) and _match_left(word, pos - 1, pattern, k - 1)
    if sym == ".":
        return ch in LETTER_VOICED and _match_left(word, pos - 1, pattern, k - 1)
    if sym == "+":
        return ch in LETTER_FRONT and _match_left(word, pos - 1, pattern, k - 1)
    if sym == "&":
        if pos >= 2 and word[pos - 2:pos] in LETTER_SIBILANT_TWO:
            if _match_left(word, pos - 2, pattern, k - 1):
                return True
        return ch in LETTER_SIBILANT_ONE and _match_left(word, pos - 1, pattern, k - 1)
    if sym == "@":
        if pos >= 2 and word[pos - 2:pos] in LETTER_U_PLAIN_TWO:
            if _match_left(word, pos - 2, pattern, k - 1):
                return True
        return ch in LETTER_U_PLAIN_ONE and _match_left(word, pos - 1, pattern, k - 1)
    return ch == sym and _match_left(word, pos - 1, pattern, k - 1)


def apply_rules(word: str, trace: list[Rule] | None = None) -> list[str]:
    """Sound out ``word`` with the rules alone: phones without stress (schwas excepted).

    Letters the rules do not cover (digits, symbols) are skipped.  ``trace``, if given,
    collects the rule that fired at each step.
    """
    text = " " + word.upper() + " "
    out: list[str] = []
    pos = 1
    end = len(text) - 1
    while pos < end:
        ch = text[pos]
        rules = RULES.get(ch)
        if not rules:
            pos += 1
            continue
        for rule in rules:
            if not text.startswith(rule.match, pos):
                continue
            if not _match_right(text, pos + len(rule.match), rule.right, 0):
                continue
            if not _match_left(text, pos, rule.left, len(rule.left)):
                continue
            out.extend(rule.phones)
            if trace is not None:
                trace.append(rule)
            pos += len(rule.match)
            break
        else:  # pragma: no cover - every letter has a bare rule
            pos += 1
    return out


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------

_PREFIXES = (
    "a", "ab", "ac", "ad", "af", "ag", "al", "ap", "ar", "as", "at", "be", "com", "con", "cor", "de", "dis", "em",
    "en", "ex", "for", "im", "in", "inter", "ir", "mis", "ob", "per", "pre", "pro", "re", "sub", "sup", "sur",
    "trans", "un",
)
"""Unstressed prefixes: a two-syllable word starting with one is usually stressed on its second syllable."""

_PENULT_SUFFIXES = ("tion", "sion", "cian", "ic", "ics", "ical", "ity", "ities", "ify", "ial", "ual", "eous",
                    "ious", "ian", "ience", "ient", "ish", "ive", "ia", "ium", "ular", "itis", "ology", "ogy",
                    "ographer", "ography", "ometer", "osis", "atic", "ency", "ancy", "ent", "ant", "ator", "ators")
"""Suffixes that fix the stress on the syllable before them (or before their own first syllable)."""

_PENULT_SUFFIX_VOWELS = {
    "tion": 1, "sion": 1, "cian": 1, "ic": 1, "ics": 1, "ical": 2, "ity": 2, "ities": 2, "ify": 2, "ial": 1,
    "ual": 1, "eous": 1, "ious": 1, "ian": 1, "ience": 1, "ient": 1, "ish": 1, "ive": 1, "ia": 1, "ium": 1,
    "ular": 2, "itis": 1, "ology": 2, "ogy": 2, "ographer": 2, "ography": 2, "ometer": 2, "osis": 1, "atic": 1,
    "ency": 2, "ancy": 2, "ent": 1, "ant": 1, "ator": 3, "ators": 3,
}
"""How many syllables before the suffix's last vowel the primary stress falls, counted in the phones."""

_FINAL_STRESS_SUFFIXES = ("ee", "eer", "ese", "ette", "esque", "ique", "oon", "ain", "een", "ine", "aire")
"""Suffixes that carry the stress themselves."""


def assign_stress(word: str, phones: list[str]) -> list[str]:
    """Give ``phones`` (the rules' output for ``word``) their stress digits.

    Vowels a rule already marked keep their mark.  Of the rest, one becomes the primary stress:
    the only one, if there is one; the one a stressed suffix names; the second syllable of a
    word starting with an unstressed prefix; the first syllable otherwise.  The remaining full
    vowels become unstressed (``0``), except a tense vowel after the primary stress, which
    keeps a secondary stress (``2``) as the dictionary would write it.
    """
    vowels = [i for i, p in enumerate(phones) if is_vowel(p)]
    if not vowels:
        return list(phones)
    free = [i for i in vowels if stress_of(phones[i]) is None]
    out = list(phones)
    if not free:
        if not any(stress_of(phones[i]) == 1 for i in vowels):
            i = vowels[0]
            out[i] = base(out[i]) + "1"
        return out
    primary = free[0]
    if len(free) > 1:
        primary = _pick_primary(word.lower(), phones, vowels, free)
    primary_syllable = vowels.index(primary)
    for i in free:
        b = base(phones[i])
        if i == primary:
            out[i] = b + "1"
        elif i == vowels[0] and primary_syllable >= 2:
            out[i] = b + "2"  # a first syllable two or more away from the stress keeps a secondary one
        elif i > primary and b in _TENSE:
            out[i] = b + "2"
        else:
            out[i] = _REDUCED.get(b, b) + "0"
    return out


_TENSE = frozenset({"EY", "IY", "AY", "OW", "UW", "AW", "OY", "AO"})
_REDUCED = {"AE": "AH", "EH": "AH", "AA": "AH", "UH": "AH", "AO": "AH"}
"""An unstressed lax vowel is a schwa: what the dictionary writes for *-ness*, *-man*, *-ic-*."""


def _pick_primary(word: str, phones: list[str], vowels: list[int], free: list[int]) -> int:
    n_syl = len(vowels)
    for suffix in _FINAL_STRESS_SUFFIXES:
        if word.endswith(suffix) and n_syl >= 2:
            return free[-1]
    for suffix in _PENULT_SUFFIXES:
        if word.endswith(suffix):
            back = _PENULT_SUFFIX_VOWELS[suffix]
            # the suffix's own syllables sit at the end of the phones; count back from them
            idx = len(vowels) - 1 - back
            if 0 <= idx < len(vowels):
                candidate = vowels[idx]
                if candidate in free:
                    return candidate
                nearer = [i for i in free if i <= candidate]
                if nearer:
                    return nearer[-1]
            break
    if n_syl >= 2:
        for prefix in sorted(_PREFIXES, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                # the prefix's vowel is the first syllable; stress the next free one
                later = [i for i in free if i > vowels[0]]
                if later and n_syl <= 3:
                    return later[0]
                break
    if n_syl >= 4:
        # long Latinate words: the antepenult, when it is free
        idx = len(vowels) - 3
        candidate = vowels[idx]
        if candidate in free:
            return candidate
    return free[0]


def letter_to_sound(word: str) -> list[str]:
    """Phones with stress for a spelled word, from the rules alone.  ``""`` gives ``[]``."""
    phones = apply_rules(word)
    return assign_stress(word, phones)
