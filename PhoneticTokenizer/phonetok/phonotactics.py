"""Phonotactics: which sounds go well together, and how to build a word out of them.

Two kinds of knowledge, one object:

* **the law** - what English lets a syllable be made of: the onset list and
  the sonority of the coda (:mod:`phonetok.syllables`).  :meth:`Phonotactics.well_formed`
  is that test, and it needs no data.
* **the habit** - how often one sound actually follows another, counted over
  every pronunciation the lexicon holds (or any pronunciations handed to
  :meth:`Phonotactics.fit`).  :meth:`affinity` is the pointwise mutual
  information of a pair, in bits: ``0`` is chance, positive is *these go
  together*, negative is *these avoid each other*.  :meth:`score` is the mean
  log-probability of a whole word's transitions, so *strength* scores well and
  *ngstrelk* does not.

:meth:`build` uses both to coin new words - an onset, a vowel and a coda per
syllable, each sampled by how often it follows what came before - and
:meth:`blend` finds the best joint for a portmanteau.  Counts, not a model:
:meth:`to_dict` writes them out and :meth:`from_dict` reads them back.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence

from .phones import BOUNDARY, CONSONANTS, VOWELS, base, is_vowel, strip_stress, with_stress
from .syllables import ONSETS, Syllable, is_legal_coda, is_legal_onset, syllabify

Phones = list[str]


def _sample(counter: Counter, rng: random.Random, exclude: set | None = None):
    """A weighted draw, in key order so that the same seed draws the same thing on every machine."""
    items = sorted((k, c) for k, c in counter.items() if not exclude or k not in exclude)
    if not items:
        return None
    total = sum(c for _, c in items)
    r = rng.random() * total
    acc = 0.0
    for k, c in items:
        acc += c
        if r < acc:
            return k
    return items[-1][0]


class Phonotactics:
    """Sound-to-sound habits counted from pronunciations, plus the law of the syllable."""

    def __init__(self, stress: bool = False, smoothing: float = 0.5) -> None:
        self.stress = stress
        """Whether the counts keep the vowels' stress digits (sparser, but truer to speech)."""
        self.smoothing = smoothing
        self.bigrams: Counter[tuple[str, str]] = Counter()
        self.unigrams: Counter[str] = Counter()
        self.followers: dict[str, Counter[str]] = {}
        self.onsets_initial: Counter[tuple[str, ...]] = Counter()
        self.onsets_medial: Counter[tuple[str, ...]] = Counter()
        self.nuclei: Counter[str] = Counter()
        self.codas_medial: Counter[tuple[str, ...]] = Counter()
        self.codas_final: Counter[tuple[str, ...]] = Counter()
        self.syllable_counts: Counter[int] = Counter()
        self.words = 0

    # -- fitting -------------------------------------------------------------

    def _key(self, phone: str) -> str:
        return phone if self.stress else base(phone)

    def fit(self, pronunciations: Iterable[Sequence[str]]) -> "Phonotactics":
        """Count the transitions and the syllable parts of every pronunciation.  Returns ``self``."""
        for phones in pronunciations:
            phones = list(phones)
            if not phones:
                continue
            keys = [BOUNDARY] + [self._key(p) for p in phones] + [BOUNDARY]
            for a, b in zip(keys, keys[1:]):
                self.bigrams[(a, b)] += 1
                self.unigrams[a] += 1
                self.followers.setdefault(a, Counter())[b] += 1
            self.unigrams[BOUNDARY] += 1
            syllables = syllabify(phones)
            self.syllable_counts[len(syllables)] += 1
            for i, s in enumerate(syllables):
                (self.onsets_initial if i == 0 else self.onsets_medial)[s.onset] += 1
                self.nuclei[s.nucleus if self.stress else with_stress(s.nucleus, 0) if is_vowel(s.nucleus) else s.nucleus] += 1
                (self.codas_final if i == len(syllables) - 1 else self.codas_medial)[s.coda] += 1
            self.words += 1
        return self

    @classmethod
    def from_lexicon(cls, lexicon, stress: bool = False) -> "Phonotactics":
        """Fit on every pronunciation of a :class:`phonetok.lexicon.Lexicon`."""
        return cls(stress=stress).fit(phones for _, phones in lexicon.items())

    @property
    def alphabet_size(self) -> int:
        return len(self.unigrams) or 1

    # -- the habit -----------------------------------------------------------

    def prob(self, a: str, b: str) -> float:
        """``P(b | a)``, add-k smoothed over the alphabet seen."""
        a, b = self._key(a), self._key(b)
        k = self.smoothing
        return (self.bigrams[(a, b)] + k) / (self.unigrams[a] + k * self.alphabet_size)

    def log_prob(self, a: str, b: str) -> float:
        return math.log(self.prob(a, b))

    def affinity(self, a: str, b: str) -> float:
        """How well ``b`` follows ``a``, as pointwise mutual information in bits (0 = chance).

        ``affinity("S", "T")`` is positive (the commonest cluster in the language);
        ``affinity("T", "L")`` is negative (no English syllable starts with *tl*).
        """
        a, b = self._key(a), self._key(b)
        total = sum(self.unigrams.values()) or 1
        k = self.smoothing
        p_ab = (self.bigrams[(a, b)] + k) / (total + k * self.alphabet_size ** 2)
        p_a = (self.unigrams[a] + k) / (total + k * self.alphabet_size)
        p_b = (self.unigrams[b] + k) / (total + k * self.alphabet_size)
        return math.log2(p_ab / (p_a * p_b))

    def transitions(self, phones: Sequence[str]) -> list[tuple[str, str, float]]:
        """Every transition of a word with its log-probability, the boundaries included."""
        keys = [BOUNDARY] + [self._key(p) for p in phones] + [BOUNDARY]
        return [(a, b, self.log_prob(a, b)) for a, b in zip(keys, keys[1:])]

    def score(self, phones: Sequence[str]) -> float:
        """The mean log-probability per transition: how English these sounds are in this order."""
        steps = self.transitions(phones)
        if not steps:
            return 0.0
        return sum(lp for _, _, lp in steps) / len(steps)

    def weakest(self, phones: Sequence[str]) -> tuple[int, float]:
        """The index (into the transitions) and log-probability of the least likely step."""
        steps = self.transitions(phones)
        i = min(range(len(steps)), key=lambda i: steps[i][2])
        return i, steps[i][2]

    def next(self, phone: str, k: int = 5) -> list[tuple[str, float]]:
        """The ``k`` likeliest sounds after ``phone`` (``"#"`` for the start of a word), with ``P``."""
        a = self._key(phone)
        table = self.followers.get(a)
        if not table:
            return []
        total = sum(table.values())
        return [(b, c / total) for b, c in table.most_common(k)]

    # -- the law -------------------------------------------------------------

    @staticmethod
    def well_formed(phones: Sequence[str]) -> bool:
        """Could an English speaker say this: every onset legal, every coda falling, a vowel somewhere?"""
        phones = list(phones)
        if not phones or not any(is_vowel(p) for p in phones):
            return False
        for s in syllabify(phones):
            if not is_legal_onset(s.onset) or not is_legal_coda(s.coda):
                return False
        return True

    @staticmethod
    def violations(phones: Sequence[str]) -> list[str]:
        """What is wrong with a sequence, in words; ``[]`` when nothing is."""
        phones = list(phones)
        if not phones:
            return ["empty"]
        out: list[str] = []
        if not any(is_vowel(p) for p in phones):
            out.append("no vowel")
        for s in syllabify(phones):
            if not is_legal_onset(s.onset):
                out.append(f"{' '.join(s.onset)} cannot start a syllable")
            if not is_legal_coda(s.coda):
                out.append(f"{' '.join(s.coda)} cannot end a syllable")
        return out

    # -- building words ------------------------------------------------------

    def build(
        self, rng: random.Random | int | None = None, syllables: int | None = None,
        min_syllables: int = 1, max_syllables: int = 3, avoid: Iterable[Sequence[str]] = (), tries: int = 50,
    ) -> Phones:
        """Coin a pronounceable word from the habits: one onset, vowel and coda per syllable.

        Each part is drawn by how often it was seen in that position, and the joints
        between parts are weighted by :meth:`prob` so the sounds that meet are sounds that
        go together.  One syllable carries the primary stress.  ``avoid`` is a set of
        pronunciations not to return (the lexicon's, to make the word new).
        """
        if not isinstance(rng, random.Random):
            rng = random.Random(rng)
        if not self.words:
            raise ValueError("fit the phonotactics on some pronunciations first")
        forbidden = {tuple(strip_stress(p)) for p in avoid}
        best: Phones = []
        for _ in range(tries):
            n = syllables if syllables else _sample_syllables(self, rng, min_syllables, max_syllables)
            stressed = 0 if n == 1 else (0 if rng.random() < 0.6 else rng.randrange(n))
            phones: Phones = []
            ok = True
            for i in range(n):
                prev = phones[-1] if phones else BOUNDARY
                onset = self._draw(self.onsets_initial if i == 0 else self.onsets_medial, rng, prev)
                if onset is None:
                    ok = False
                    break
                phones.extend(onset)
                prev = phones[-1] if phones else BOUNDARY
                nucleus = self._draw_nucleus(rng, prev, 1 if i == stressed else 0)
                phones.append(nucleus)
                last = i == n - 1
                coda = self._draw(self.codas_final if last else self.codas_medial, rng, nucleus)
                if coda is None:
                    ok = False
                    break
                phones.extend(coda)
            if not ok or not self.well_formed(phones):
                continue
            if tuple(strip_stress(phones)) in forbidden:
                continue
            return phones
        return best

    def _draw(self, table: Counter, rng: random.Random, prev: str):
        """A syllable part from ``table``, weighted by its count and by how well it follows ``prev``."""
        if not table:
            return ()
        weighted: Counter = Counter()
        for part, count in table.items():
            first = part[0] if part else None
            weight = count * (self.prob(prev, first) * self.alphabet_size if first else 1.0)
            weighted[part] = weight
        return _sample(weighted, rng)

    def _draw_nucleus(self, rng: random.Random, prev: str, stress: int) -> str:
        weighted: Counter = Counter()
        for nucleus, count in self.nuclei.items():
            if not is_vowel(nucleus):
                continue
            weighted[nucleus] = count * self.prob(prev, nucleus) * self.alphabet_size
        chosen = _sample(weighted, rng) or "AH0"
        return with_stress(chosen, stress)

    def blend(self, a: Sequence[str], b: Sequence[str]) -> tuple[Phones, int, int]:
        """The best portmanteau of two words: ``(phones, syllables kept of a, syllables dropped of b)``.

        Every cut that keeps at least one syllable of each is scored by how well the
        sounds at the joint go together and how close to half of each word survives;
        *breakfast* + *lunch* gives *brunch*-like joints, *smoke* + *fog* gives *smog*.
        """
        if not any(is_vowel(p) for p in a) or not any(is_vowel(p) for p in b):
            raise ValueError("both words need a vowel to be blended")
        sa, sb = syllabify(list(a)), syllabify(list(b))
        best: tuple[float, Phones, int, int] | None = None
        candidates: list[tuple[Phones, int, int]] = []
        # joints at syllable edges, and inside the syllables at the vowel
        for i in range(1, len(sa) + 1):
            for j in range(0, len(sb)):
                head = [p for s in sa[:i] for p in s.phones]
                tail = [p for s in sb[j:] for p in s.phones]
                candidates.append((head + tail, i, j))
                # keep a's onset (and vowel) and take b's rime
                head_open = [p for s in sa[: i - 1] for p in s.phones] + list(sa[i - 1].onset)
                candidates.append((head_open + list(sb[j].rime) + [p for s in sb[j + 1:] for p in s.phones], i, j))
        for phones, i, j in candidates:
            if not self.well_formed(phones) or len(phones) < 2:
                continue
            joint = self.score(phones)
            balance = -abs((i / len(sa)) - 0.5) - abs(((len(sb) - j) / len(sb)) - 0.5)
            length = len(phones) / max(len(a), len(b))  # a blend that keeps more of its parents reads as both
            total = joint + 0.5 * balance + 0.2 * length
            if best is None or total > best[0]:
                best = (total, phones, i, j)
        if best is None:
            return list(a) + list(b), len(sa), 0
        phones = best[1]
        # one primary stress
        if sum(1 for p in phones if p.endswith("1")) > 1:
            seen = False
            for k, p in enumerate(phones):
                if p.endswith("1"):
                    if seen:
                        phones[k] = with_stress(p, 2)
                    seen = True
        if not any(p.endswith("1") for p in phones):
            for k, p in enumerate(phones):
                if is_vowel(p):
                    phones[k] = with_stress(p, 1)
                    break
        return phones, best[2], best[3]

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "stress": self.stress, "smoothing": self.smoothing, "words": self.words,
            "bigrams": [[a, b, c] for (a, b), c in sorted(self.bigrams.items())],
            "onsets_initial": [[" ".join(k), c] for k, c in sorted(self.onsets_initial.items())],
            "onsets_medial": [[" ".join(k), c] for k, c in sorted(self.onsets_medial.items())],
            "nuclei": sorted(self.nuclei.items()),
            "codas_medial": [[" ".join(k), c] for k, c in sorted(self.codas_medial.items())],
            "codas_final": [[" ".join(k), c] for k, c in sorted(self.codas_final.items())],
            "syllable_counts": sorted(self.syllable_counts.items()),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Phonotactics":
        p = cls(stress=bool(d.get("stress", False)), smoothing=float(d.get("smoothing", 0.5)))
        p.words = int(d.get("words", 0))
        for a, b, c in d.get("bigrams", []):
            p.bigrams[(a, b)] = c
            p.unigrams[a] += c
            p.followers.setdefault(a, Counter())[b] += c
        p.unigrams[BOUNDARY] += p.words  # the final boundary of every word
        for name in ("onsets_initial", "onsets_medial", "codas_medial", "codas_final"):
            table = getattr(p, name)
            for k, c in d.get(name, []):
                table[tuple(k.split()) if k else ()] = c
        for k, c in d.get("nuclei", []):
            p.nuclei[k] = c
        for k, c in d.get("syllable_counts", []):
            p.syllable_counts[int(k)] = c
        return p

    def __repr__(self) -> str:
        return f"Phonotactics(words={self.words}, pairs={len(self.bigrams)}, stress={self.stress})"


def _sample_syllables(p: Phonotactics, rng: random.Random, lo: int, hi: int) -> int:
    table = Counter({n: c for n, c in p.syllable_counts.items() if lo <= n <= hi})
    return _sample(table, rng) or lo
