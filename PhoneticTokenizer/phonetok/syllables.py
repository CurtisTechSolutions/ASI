"""Syllables: the sounds that go together to build a word.

A syllable is an *onset* (the consonants before the vowel), a *nucleus* (the
vowel, which carries the stress) and a *coda* (the consonants after it); the
nucleus and the coda together are the *rime*, which is what rhymes.  English
lets very few consonant clusters start a syllable - :data:`ONSETS` is the
whole list - and that closed list is what decides where one syllable ends and
the next begins:

    *maximal onset*: between two vowels, as many consonants as can legally
    start a syllable go to the second one; the rest close the first.

So ``S T R EH1 NG TH S`` is one syllable (``S T R`` may start a syllable, and
``NG TH S`` closes it), ``AE1 S T R OW0`` (*astro*) breaks as ``AE1 . S T R OW0``,
and ``B AH1 T ER0`` (*butter*) as ``B AH1 T . ER0``: a stressed lax vowel
(``AE AH EH IH UH``) does not end a syllable in English, so it takes the
consonant that would otherwise have started the next.

Nothing here is learned: the onset list is the language's, and so the same
word breaks the same way everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .phones import CONSONANTS, base, is_vowel, sonority, stress_of

_SINGLE_ONSETS = frozenset((c,) for c in CONSONANTS if c != "NG")
_DOUBLE_ONSETS = frozenset(tuple(s.split()) for s in """
P L  P R  P Y  B L  B R  B Y  T R  T W  T Y  D R  D W  D Y  K L  K R  K W  K Y  G L  G R  G W  G Y
F L  F R  F Y  V Y  TH R  TH W  TH Y  SH R  SH M  SH N  SH L  SH W
S P  S T  S K  S M  S N  S L  S W  S F  M Y  N Y  L Y  HH Y
""".replace("\n", "  ").split("  ") if s.strip())
_TRIPLE_ONSETS = frozenset(tuple(s.split()) for s in """
S P L  S P R  S P Y  S T R  S T Y  S K L  S K R  S K W  S K Y
""".replace("\n", "  ").split("  ") if s.strip())

ONSETS: frozenset[tuple[str, ...]] = frozenset({()}) | _SINGLE_ONSETS | _DOUBLE_ONSETS | _TRIPLE_ONSETS
"""Every consonant cluster that may start an English syllable, the empty one included."""

LAX_VOWELS = frozenset({"AE", "AH", "EH", "IH", "UH"})
"""The vowels that do not end a stressed syllable."""

APPENDIX = frozenset({"S", "Z", "T", "D", "TH"})
"""Coronal obstruents that may close a syllable after anything (*sixths*, *texts*)."""


def is_legal_onset(cluster: Sequence[str]) -> bool:
    """May this consonant cluster start a syllable?  The empty cluster may."""
    return tuple(cluster) in ONSETS


def is_legal_coda(cluster: Sequence[str]) -> bool:
    """May this cluster close a syllable: does its sonority fall away from the vowel?

    A run of coronal obstruents at the very end (:data:`APPENDIX`) is allowed after
    anything, as *sixths* (``K S TH S``) and *texts* (``K S T S``) need it.  ``NG`` and
    ``HH``: the first closes but never opens, the second opens but never closes.
    """
    cluster = tuple(cluster)
    if not cluster:
        return True
    if "HH" in cluster or "W" in cluster or "Y" in cluster:
        return False
    if len(cluster) > 4:
        return False
    core = list(cluster)
    while len(core) > 1 and core[-1] in APPENDIX:
        core.pop()
    return all(sonority(a) >= sonority(b) for a, b in zip(core, core[1:]))


@dataclass(frozen=True)
class Syllable:
    """One syllable: ``onset`` + ``nucleus`` + ``coda``.

    ``nucleus`` is a vowel with its stress digit.  A word with no vowel at all
    (*hmm*, *shh*) gets its most sonorous consonant as the nucleus.
    """

    onset: tuple[str, ...]
    nucleus: str
    coda: tuple[str, ...]

    @property
    def stress(self) -> int:
        """0, 1 or 2 (0 for a vowel written without a digit, and for a consonant nucleus)."""
        s = stress_of(self.nucleus)
        return s if s is not None else 0

    @property
    def phones(self) -> tuple[str, ...]:
        return self.onset + (self.nucleus,) + self.coda

    @property
    def rime(self) -> tuple[str, ...]:
        """The nucleus and the coda: what a rhyme shares."""
        return (self.nucleus,) + self.coda

    @property
    def open(self) -> bool:
        return not self.coda

    def text(self) -> str:
        """The syllable as one token: its phones joined by dots, ``"S.T.R.EH1.NG.TH.S"``."""
        return ".".join(self.phones)

    @classmethod
    def parse(cls, text: str) -> "Syllable":
        """The inverse of :meth:`text`."""
        return syllabify(text.split("."))[0] if text else cls((), "", ())

    def __str__(self) -> str:
        return self.text()


def _split_cluster(cluster: Sequence[str]) -> int:
    """How many of the consonants between two vowels close the first syllable."""
    for keep in range(len(cluster) + 1):
        if is_legal_onset(cluster[keep:]):
            return keep
    return len(cluster)  # pragma: no cover - the empty onset is always legal


def syllabify(phones: Sequence[str], close_lax: bool = True) -> list[Syllable]:
    """Break a word's phones into syllables by maximal onset.

    ``close_lax`` applies the second rule of the module docstring: a stressed lax
    vowel with nothing after it takes the first consonant of the next onset.
    ``[]`` gives ``[]``.
    """
    phones = list(phones)
    if not phones:
        return []
    vowels = [i for i, p in enumerate(phones) if is_vowel(p)]
    if not vowels:
        peak = max(range(len(phones)), key=lambda i: (sonority(phones[i]), -i))
        return [Syllable(tuple(phones[:peak]), phones[peak], tuple(phones[peak + 1:]))]
    out: list[Syllable] = []
    onset_start = 0
    for k, vi in enumerate(vowels):
        onset = tuple(phones[onset_start:vi])
        if k + 1 < len(vowels):
            cluster = phones[vi + 1:vowels[k + 1]]
            keep = _split_cluster(cluster)
            if (
                close_lax and keep == 0 and cluster and base(phones[vi]) in LAX_VOWELS
                and stress_of(phones[vi]) in (1, 2) and is_legal_onset(cluster[1:])
            ):
                keep = 1
            coda = tuple(cluster[:keep])
            onset_start = vi + 1 + keep
        else:
            coda = tuple(phones[vi + 1:])
        out.append(Syllable(onset, phones[vi], coda))
    return out


def syllable_count(phones: Sequence[str]) -> int:
    """How many syllables: one per vowel, or one for a word with none."""
    n = sum(1 for p in phones if is_vowel(p))
    return n if n else (1 if phones else 0)


def stressed_syllable(syllables: Sequence[Syllable]) -> int | None:
    """The index of the syllable carrying primary stress; the last stressed one, else the last, if any."""
    if not syllables:
        return None
    for i, s in enumerate(syllables):
        if s.stress == 1:
            return i
    for i, s in enumerate(syllables):
        if s.stress == 2:
            return i
    return len(syllables) - 1


def rhymes(a: Sequence[str], b: Sequence[str]) -> bool:
    """Do two words rhyme: the same sounds from the last stressed vowel on?

    *cat* / *hat* do (``AE1 T``), *nation* / *station* do (``EY1 SH AH0 N``),
    *cat* / *cut* do not, and a word does not rhyme with itself's identical copy
    only in the sense that it does - the caller decides whether that counts.
    """
    sa, sb = syllabify(a), syllabify(b)
    ia, ib = stressed_syllable(sa), stressed_syllable(sb)
    if ia is None or ib is None:
        return False
    tail_a = [p for s in sa[ia:] for p in s.phones]
    tail_b = [p for s in sb[ib:] for p in s.phones]
    # everything from the stressed vowel on, the onset of the stressed syllable excepted
    tail_a = tail_a[len(sa[ia].onset):]
    tail_b = tail_b[len(sb[ib].onset):]
    return [base(p) for p in tail_a] == [base(p) for p in tail_b]


def alliterates(a: Sequence[str], b: Sequence[str]) -> bool:
    """Do two words start with the same consonant sound(s)?  Two vowel-initial words do not alliterate."""
    sa, sb = syllabify(a), syllabify(b)
    if not sa or not sb:
        return False
    return bool(sa[0].onset) and sa[0].onset == sb[0].onset
