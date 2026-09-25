"""The sound inventory: the phonemes of English, what each one is made of, and their ids.

A *phone* here is a symbol of the CMU Pronouncing Dictionary's ARPAbet: 39
phonemes, the 15 vowels carrying a stress digit (``0`` none, ``1`` primary,
``2`` secondary) so that ``"AH0"`` (the schwa of *about*) and ``"AH1"`` (the
vowel of *hut*) are two sounds, which they are.  The consonants carry nothing.

Three things are decided here and nowhere else:

* **what a phone is** - :data:`PHONEMES`, :data:`VOWELS`, :data:`CONSONANTS`,
  and the grammar of a token (:func:`is_phone`, :func:`base`, :func:`stress_of`);
* **what a phone is made of** - :func:`features`, an articulatory feature
  vector (voice, place, manner, height, backness...) so that two sounds that
  are made the same way come out close, and :func:`sonority`, the scale the
  syllable is built on;
* **its id** - :data:`SYMBOLS` is the fixed alphabet: a few specials, the
  boundary and the pauses, then every ARPAbet symbol in the dictionary's own
  order.  Nothing here is learned from a corpus: the alphabet of a language's
  sounds is the language's, and it is the same on every machine.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# the inventory
# ---------------------------------------------------------------------------

VOWELS: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
)
"""The 15 vowel phonemes, in ARPAbet order.  *odd, at, hut, ought, cow, hide, Ed, hurt, ate, it, eat, oat, toy, hood, two*."""

CONSONANTS: tuple[str, ...] = (
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T",
    "TH", "V", "W", "Y", "Z", "ZH",
)
"""The 24 consonant phonemes, in ARPAbet order."""

PHONEMES: tuple[str, ...] = tuple(sorted(VOWELS + CONSONANTS))
"""All 39, in the dictionary's (alphabetical) order."""

STRESSES = ("0", "1", "2")
"""No stress, primary, secondary - the digit a vowel carries."""

_VOWEL_SET = frozenset(VOWELS)
_CONSONANT_SET = frozenset(CONSONANTS)

# -- the token grammar -------------------------------------------------------

BOUNDARY = "#"
"""The word boundary: what stands between two words' sounds (phonology's own symbol for it)."""

PAUSE_SHORT = ","
PAUSE_FULL = "."
PAUSE_QUESTION = "?"
PAUSES: tuple[str, ...] = (PAUSE_SHORT, PAUSE_FULL, PAUSE_QUESTION)
"""What punctuation becomes: a short pause, a full stop, a rising one.  A pause implies a boundary."""

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"
SPECIALS: tuple[str, ...] = (PAD, UNK, BOS, EOS)
"""Padding, the unknown token, start and end of a text - the four every tokeniser is expected to have."""

SYMBOLS: tuple[str, ...] = SPECIALS + (BOUNDARY,) + PAUSES + tuple(
    sym for v in VOWELS for sym in (v, v + "0", v + "1", v + "2")
) + CONSONANTS
"""The fixed alphabet, in id order: ``<pad> <unk> <s> </s> # , . ? AA AA0 AA1 AA2 AE ... ZH``.

The 84 ARPAbet entries are exactly ``cmudict.symbols`` (the bare vowel first, then its three stresses,
then the consonants), so an id is the same whether or not stress is kept.  92 symbols in all.
"""

SYMBOL_ID: dict[str, int] = {sym: i for i, sym in enumerate(SYMBOLS)}
"""Symbol -> id.  Ids 0-3 are the specials, 4 the boundary, 5-7 the pauses, 8-91 the phones."""

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
BOUNDARY_ID = SYMBOL_ID[BOUNDARY]


def base(phone: str) -> str:
    """The phoneme without its stress digit: ``"AH0" -> "AH"``, ``"K" -> "K"``."""
    return phone[:-1] if phone and phone[-1] in "012" else phone


def stress_of(phone: str) -> int | None:
    """The stress digit of a vowel as an int, or ``None`` for a consonant or an unstressed-by-omission vowel."""
    return int(phone[-1]) if phone and phone[-1] in "012" else None


def is_vowel(phone: str) -> bool:
    """Is this a vowel, with or without its stress digit?"""
    return base(phone) in _VOWEL_SET


def is_consonant(phone: str) -> bool:
    return phone in _CONSONANT_SET


def is_phone(token: str) -> bool:
    """Is ``token`` an ARPAbet symbol: a consonant, or a vowel with or without a stress digit?"""
    return base(token) in _VOWEL_SET or token in _CONSONANT_SET


def is_stressed(phone: str) -> bool:
    """Primary or secondary stress."""
    return stress_of(phone) in (1, 2)


def strip_stress(phones: "list[str] | tuple[str, ...]") -> list[str]:
    """The same phones with every stress digit removed."""
    return [base(p) for p in phones]


def with_stress(phone: str, stress: int) -> str:
    """A vowel with its stress set to ``stress``; a consonant is left alone."""
    b = base(phone)
    return f"{b}{stress}" if b in _VOWEL_SET else phone


def check_phones(phones: "list[str] | tuple[str, ...]") -> None:
    """Raise :class:`ValueError` at the first token that is not a phone."""
    for p in phones:
        if not is_phone(p):
            raise ValueError(f"{p!r} is not an ARPAbet phone")


# ---------------------------------------------------------------------------
# classes and articulation
# ---------------------------------------------------------------------------

# manner of every consonant (the dictionary's own classes, the liquids split)
STOPS = frozenset({"B", "D", "G", "K", "P", "T"})
AFFRICATES = frozenset({"CH", "JH"})
FRICATIVES = frozenset({"DH", "F", "S", "SH", "TH", "V", "Z", "ZH"})
ASPIRATES = frozenset({"HH"})
NASALS = frozenset({"M", "N", "NG"})
LIQUIDS = frozenset({"L", "R"})
GLIDES = frozenset({"W", "Y"})
OBSTRUENTS = STOPS | AFFRICATES | FRICATIVES | ASPIRATES
SONORANTS = NASALS | LIQUIDS | GLIDES
SIBILANTS = frozenset({"S", "Z", "SH", "ZH", "CH", "JH"})
VOICED_CONSONANTS = frozenset({"B", "D", "DH", "G", "JH", "L", "M", "N", "NG", "R", "V", "W", "Y", "Z", "ZH"})
VOICELESS_CONSONANTS = _CONSONANT_SET - VOICED_CONSONANTS

MANNER: dict[str, str] = {}
for _p in STOPS:
    MANNER[_p] = "stop"
for _p in AFFRICATES:
    MANNER[_p] = "affricate"
for _p in FRICATIVES:
    MANNER[_p] = "fricative"
for _p in ASPIRATES:
    MANNER[_p] = "aspirate"
for _p in NASALS:
    MANNER[_p] = "nasal"
for _p in LIQUIDS:
    MANNER[_p] = "liquid"
for _p in GLIDES:
    MANNER[_p] = "semivowel"
for _p in VOWELS:
    MANNER[_p] = "vowel"
del _p

PLACE: dict[str, str] = {
    "B": "bilabial", "P": "bilabial", "M": "bilabial", "W": "labial-velar",
    "F": "labiodental", "V": "labiodental",
    "TH": "dental", "DH": "dental",
    "T": "alveolar", "D": "alveolar", "S": "alveolar", "Z": "alveolar", "N": "alveolar", "L": "alveolar",
    "R": "alveolar",
    "SH": "postalveolar", "ZH": "postalveolar", "CH": "postalveolar", "JH": "postalveolar",
    "Y": "palatal",
    "K": "velar", "G": "velar", "NG": "velar",
    "HH": "glottal",
}
"""Where each consonant is made."""

SONORITY: dict[str, int] = {}
"""The sonority scale, 0 (a voiceless stop) to 10 (a low vowel): a syllable rises to its vowel and falls after it."""
for _p in STOPS:
    SONORITY[_p] = 0 if _p in VOICELESS_CONSONANTS else 1
for _p in AFFRICATES:
    SONORITY[_p] = 1 if _p in VOICELESS_CONSONANTS else 2
for _p in FRICATIVES:
    SONORITY[_p] = 2 if _p in VOICELESS_CONSONANTS else 3
SONORITY["HH"] = 2
for _p in NASALS:
    SONORITY[_p] = 4
SONORITY["L"] = 5
SONORITY["R"] = 6
for _p in GLIDES:
    SONORITY[_p] = 7
for _p in ("IY", "IH", "UW", "UH", "ER"):
    SONORITY[_p] = 8
for _p in ("EY", "EH", "OW", "AO", "AH", "OY"):
    SONORITY[_p] = 9
for _p in ("AE", "AA", "AY", "AW"):
    SONORITY[_p] = 10
del _p

MAX_SONORITY = 10


def sonority(phone: str) -> int:
    """Where a phone sits on the scale; the stress digit is ignored."""
    return SONORITY[base(phone)]


# the vowel space: (height, backness, round, tense), each 0..1, and the off-glide of a diphthong
_VOWEL_SPACE: dict[str, tuple[float, float, float, float, str]] = {
    #        height back  round tense glide
    "IY": (1.0, 0.0, 0.0, 1.0, ""),
    "IH": (0.8, 0.2, 0.0, 0.0, ""),
    "EY": (0.6, 0.0, 0.0, 1.0, "Y"),
    "EH": (0.5, 0.2, 0.0, 0.0, ""),
    "AE": (0.2, 0.2, 0.0, 0.0, ""),
    "AA": (0.0, 0.9, 0.0, 1.0, ""),
    "AO": (0.3, 1.0, 1.0, 1.0, ""),
    "OW": (0.6, 1.0, 1.0, 1.0, "W"),
    "UH": (0.8, 0.9, 1.0, 0.0, ""),
    "UW": (1.0, 1.0, 1.0, 1.0, ""),
    "AH": (0.4, 0.6, 0.0, 0.0, ""),
    "ER": (0.5, 0.5, 0.0, 1.0, ""),
    "AY": (0.1, 0.5, 0.0, 1.0, "Y"),
    "AW": (0.1, 0.5, 0.0, 1.0, "W"),
    "OY": (0.3, 1.0, 1.0, 1.0, "Y"),
}

DIPHTHONGS = frozenset({"AW", "AY", "EY", "OW", "OY"})
RHOTIC_VOWELS = frozenset({"ER"})

FEATURE_NAMES: tuple[str, ...] = (
    "syllabic", "consonantal", "sonorant", "voice", "nasal", "continuant", "strident", "lateral", "rhotic",
    "labial", "coronal", "dorsal", "glottal",
    "high", "low", "back", "round", "tense", "glide_y", "glide_w",
    "stress", "sonority",
)
"""The 22 dimensions of :func:`features`, in order.  Every value lies in ``[0, 1]``."""

FEATURE_DIM = len(FEATURE_NAMES)

_LABIAL = frozenset({"B", "P", "M", "F", "V", "W"})
_CORONAL = frozenset({"T", "D", "S", "Z", "N", "L", "R", "TH", "DH", "SH", "ZH", "CH", "JH"})
_DORSAL = frozenset({"K", "G", "NG", "W", "Y"})
_GLOTTAL = frozenset({"HH"})
_CONTINUANT = FRICATIVES | ASPIRATES | LIQUIDS | GLIDES
_STRIDENT = frozenset({"S", "Z", "SH", "ZH", "CH", "JH", "F", "V"})

_FEATURE_CACHE: dict[str, tuple[float, ...]] = {}


def features(phone: str) -> tuple[float, ...]:
    """The articulatory feature vector of a phone (:data:`FEATURE_NAMES`), stress included.

    Two sounds made the same way are close: ``features("P")`` and ``features("B")`` differ in
    voice alone, ``"IY1"`` and ``"IH1"`` in tense and a little height.  This is what lets a
    model that reads sounds be given vectors in which *sounding alike* is a short distance.
    """
    cached = _FEATURE_CACHE.get(phone)
    if cached is not None:
        return cached
    b = base(phone)
    if b not in _VOWEL_SET and b not in _CONSONANT_SET:
        raise ValueError(f"{phone!r} is not an ARPAbet phone")
    stress = stress_of(phone)
    stress_value = {None: 0.0, 0: 0.0, 1: 1.0, 2: 0.5}[stress]
    if b in _VOWEL_SET:
        height, back, rnd, tense, glide = _VOWEL_SPACE[b]
        vec = (
            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0 if b in RHOTIC_VOWELS else 0.0,
            0.0, 0.0, 0.0, 0.0,
            height, 1.0 - height, back, rnd, tense,
            1.0 if glide == "Y" else 0.0, 1.0 if glide == "W" else 0.0,
            stress_value, SONORITY[b] / MAX_SONORITY,
        )
    else:
        vec = (
            0.0,
            0.0 if b in GLIDES or b in ASPIRATES else 1.0,
            1.0 if b in SONORANTS else 0.0,
            1.0 if b in VOICED_CONSONANTS else 0.0,
            1.0 if b in NASALS else 0.0,
            1.0 if b in _CONTINUANT else 0.0,
            1.0 if b in _STRIDENT else 0.0,
            1.0 if b == "L" else 0.0,
            1.0 if b == "R" else 0.0,
            1.0 if b in _LABIAL else 0.0,
            1.0 if b in _CORONAL else 0.0,
            1.0 if b in _DORSAL else 0.0,
            1.0 if b in _GLOTTAL else 0.0,
            1.0 if b in ("Y", "W") else 0.0,  # glides are high
            0.0,
            1.0 if b in ("W", "K", "G", "NG") else 0.0,
            1.0 if b == "W" else 0.0,
            0.0, 0.0, 0.0,
            0.0, SONORITY[b] / MAX_SONORITY,
        )
    _FEATURE_CACHE[phone] = vec
    return vec


def distance(a: str, b: str) -> float:
    """How differently two phones are made: the Euclidean distance of their feature vectors, 0 for the same phone."""
    fa, fb = features(a), features(b)
    return sum((x - y) ** 2 for x, y in zip(fa, fb)) ** 0.5


def similarity(a: str, b: str) -> float:
    """``1`` for the same phone, falling towards ``0`` as two phones are made more differently."""
    return 1.0 / (1.0 + distance(a, b))


# ---------------------------------------------------------------------------
# the IPA
# ---------------------------------------------------------------------------

IPA: dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
    "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
"""ARPAbet -> IPA, one entry per phoneme.  The unstressed ``AH0`` and ``ER0`` become ``ə`` and ``ɚ``."""

_IPA_REDUCED = {"AH": "ə", "ER": "ɚ"}
_IPA_STRESS = {1: "ˈ", 2: "ˌ"}
_IPA_TO_ARPA: dict[str, str] = {v: k for k, v in IPA.items()}
_IPA_TO_ARPA.update({"ə": "AH0", "ɚ": "ER0", "ɜ": "ER", "ɜː": "ER", "ɹ": "R", "r": "R", "g": "G", "iː": "IY",
                     "uː": "UW", "ɑː": "AA", "ɔː": "AO", "ɛ": "EH", "e": "EH", "ɒ": "AA", "a": "AE"})


def to_ipa(phones: "list[str] | tuple[str, ...]", stress: bool = True) -> str:
    """Phones as one IPA string: ``["K", "AE1", "T"] -> "kæt"``, with stress marks before the syllable's vowel.

    The mark goes where the phones put it - before the vowel - since this does not syllabify;
    :func:`phonetok.syllables.syllabify` is what places it at a syllable's start.
    """
    out: list[str] = []
    for p in phones:
        b = base(p)
        s = stress_of(p)
        if b in _VOWEL_SET:
            if stress and s in _IPA_STRESS:
                out.append(_IPA_STRESS[s])
            if s == 0 and b in _IPA_REDUCED:
                out.append(_IPA_REDUCED[b])
            else:
                out.append(IPA[b])
        elif b in _CONSONANT_SET:
            out.append(IPA[b])
        else:
            out.append(p)  # a boundary or a pause passes through
    return "".join(out)


def from_ipa(text: str) -> list[str]:
    """IPA back to ARPAbet, greedily, longest symbol first; stress marks set the next vowel's digit.

    Only the symbols :func:`to_ipa` writes (and a few common variants) are read; anything else
    raises :class:`ValueError`.
    """
    symbols = sorted(_IPA_TO_ARPA, key=len, reverse=True)
    out: list[str] = []
    stress: int | None = None
    i = 0
    text = text.replace("ː", "ː")
    while i < len(text):
        ch = text[i]
        if ch in " .":
            i += 1
            continue
        if ch == "ˈ":
            stress, i = 1, i + 1
            continue
        if ch == "ˌ":
            stress, i = 2, i + 1
            continue
        for sym in symbols:
            if text.startswith(sym, i):
                arpa = _IPA_TO_ARPA[sym]
                if is_vowel(arpa):
                    b = base(arpa)
                    if stress is not None:
                        arpa = f"{b}{stress}"
                    elif stress_of(arpa) is None:
                        arpa = f"{b}0"
                    stress = None
                out.append(arpa)
                i += len(sym)
                break
        else:
            raise ValueError(f"cannot read IPA {text[i]!r} at {i} in {text!r}")
    return out


# ---------------------------------------------------------------------------
# the possessive / plural / past suffixes: the sound a word ends in decides them
# ---------------------------------------------------------------------------

def plural_suffix(phones: "list[str] | tuple[str, ...]") -> list[str]:
    """The ``-s`` / ``-'s`` ending after these phones: ``IH0 Z`` after a sibilant, ``Z`` after voice, ``S`` otherwise."""
    if not phones:
        return ["Z"]
    last = base(phones[-1])
    if last in SIBILANTS:
        return ["IH0", "Z"]
    if last in _VOWEL_SET or last in VOICED_CONSONANTS:
        return ["Z"]
    return ["S"]


def past_suffix(phones: "list[str] | tuple[str, ...]") -> list[str]:
    """The ``-ed`` ending: ``IH0 D`` after T or D, ``D`` after voice, ``T`` otherwise."""
    if not phones:
        return ["D"]
    last = base(phones[-1])
    if last in ("T", "D"):
        return ["IH0", "D"]
    if last in _VOWEL_SET or last in VOICED_CONSONANTS:
        return ["D"]
    return ["T"]
