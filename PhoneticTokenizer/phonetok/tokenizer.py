"""The phonetic tokenizer: text in, the sounds it is made of out - and back.

:class:`PhoneticTokenizer` reads a text as *sounds* rather than as letters or
words.  Every word is transcribed (:mod:`phonetok.g2p`), punctuation becomes
a pause, and what is between two words is a boundary.  The **level** decides
what one token is:

| level | one token is | ``"the cat sat"`` becomes |
|---|---|---|
| ``phoneme`` | one sound | ``DH AH0 # K AE1 T # S AE1 T`` |
| ``constituent`` | an onset, a vowel or a coda | ``DH- AH0 # K- AE1 -T # S- AE1 -T`` |
| ``syllable`` | one syllable | ``DH.AH0 # K.AE1.T # S.AE1.T`` |
| ``word`` | one word's sounds | ``DH.AH0 # K.AE1.T # S.AE1.T`` |

The **text form** (:meth:`text`) is those tokens joined by single spaces: it
is plain text a model can read, it is idempotent (:meth:`tokenize` reads its
own output back unchanged - any run of its tokens, a gram of a graph included,
even one starting or ending on a ``#`` or a pause - and a text may mix words
and sounds), and it is readable.  :meth:`encode` numbers the tokens: at the phoneme level the ids
are the fixed alphabet of :mod:`phonetok.phones`, the same everywhere; at the
other levels a token gets the next id the first time it is read, and
:meth:`save` keeps that table.  :meth:`decode` spells the sounds back into
words through the lexicon, the tokenizer's memory of what it read, and a
respelling for sounds no known word has.

The sound helpers - :meth:`ipa`, :meth:`syllables`, :meth:`rhymes`,
:meth:`affinity`, :meth:`score`, :meth:`coin`, :meth:`blend` - are the
phonotactics of :mod:`phonetok.phonotactics` reached through the same object.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .g2p import Transcriber
from .lexicon import Lexicon
from .phones import (
    BOUNDARY, BOUNDARY_ID, BOS, EOS, FEATURE_DIM, PAD, PAUSE_FULL, PAUSE_QUESTION, PAUSE_SHORT, PAUSES,
    SPECIALS, SYMBOL_ID, SYMBOLS, UNK, UNK_ID, base, features, is_phone, is_vowel, strip_stress, to_ipa,
)
from .phonotactics import Phonotactics
from .syllables import Syllable, alliterates, rhymes, syllabify

PHONEME, CONSTITUENT, SYLLABLE, WORD = "phoneme", "constituent", "syllable", "word"
LEVELS = (PHONEME, CONSTITUENT, SYLLABLE, WORD)

FIXED: tuple[str, ...] = SPECIALS + (BOUNDARY,) + PAUSES
"""The tokens every level's vocabulary starts with, in id order: ``<pad> <unk> <s> </s> # , . ?``."""

_TRAILING = ".,;:!?…\"')]}»’”"
_LEADING = "(\"'[{«‘“…"
_PAUSE_OF = {".": PAUSE_FULL, "!": PAUSE_FULL, "…": PAUSE_FULL, "?": PAUSE_QUESTION, ",": PAUSE_SHORT,
             ";": PAUSE_SHORT, ":": PAUSE_SHORT, ")": PAUSE_SHORT, "]": PAUSE_SHORT, "}": PAUSE_SHORT}
_DASHES = {"-", "--", "—", "–"}
_CLUSTER = re.compile(r"^(?:[A-Z]{1,2}[012]?)(?:\.[A-Z]{1,2}[012]?)*$")


@dataclass(frozen=True)
class Token:
    """One token: its text, what kind of thing it is, the phones it stands for, and its word."""

    text: str
    kind: str
    """``phone``, ``onset``, ``nucleus``, ``coda``, ``syllable``, ``word``, ``boundary``, ``pause`` or ``special``."""
    phones: tuple[str, ...] = ()
    word: int = -1
    """The index of the word this token belongs to; ``-1`` for a boundary, a pause or a special."""

    def __str__(self) -> str:
        return self.text


def parse_token(text: str) -> tuple[str, tuple[str, ...]] | None:
    """What a token string is: ``(kind, phones)``, or ``None`` if it is not a phonetic token.

    The grammar: a phone (``K``, ``AE1``); ``#``; a pause (``,`` ``.`` ``?``); an onset
    ``S.T.R-``; a coda ``-NG.TH.S``; a syllable or word ``S.T.R.EH1.NG.TH.S`` (phones joined
    by dots); the specials.  Anything else is spelled text.
    """
    if not text:
        return None
    if text in SPECIALS:
        return "special", ()
    if text == BOUNDARY:
        return "boundary", ()
    if text in PAUSES:
        return "pause", ()
    if is_phone(text):
        return ("nucleus" if is_vowel(text) else "phone"), (text,)
    onset = text.endswith("-") and not text.startswith("-")
    coda = text.startswith("-") and not text.endswith("-")
    body = text[:-1] if onset else text[1:] if coda else text
    if not _CLUSTER.match(body):
        return None
    phones = tuple(body.split("."))
    if not all(is_phone(p) for p in phones):
        return None
    if onset or coda:
        if any(is_vowel(p) for p in phones):
            return None
        return ("onset" if onset else "coda"), phones
    if len(phones) == 1:
        return ("nucleus" if is_vowel(phones[0]) else "phone"), phones
    return ("syllable" if sum(is_vowel(p) for p in phones) <= 1 else "word"), phones


class Vocab:
    """Token strings <-> ids.  The fixed head is :data:`FIXED`; the rest grows in the order it is read."""

    __slots__ = ("tokens", "_ids", "frozen")

    def __init__(self, tokens: Iterable[str] = FIXED, frozen: bool = False) -> None:
        self.tokens: list[str] = []
        self._ids: dict[str, int] = {}
        self.frozen = frozen
        for t in tokens:
            self.add(t)

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._ids

    def add(self, token: str) -> int:
        found = self._ids.get(token)
        if found is not None:
            return found
        self._ids[token] = len(self.tokens)
        self.tokens.append(token)
        return self._ids[token]

    def id(self, token: str, grow: bool = False) -> int:
        """The id of a token; a new token gets the next id if ``grow`` and the vocabulary is not frozen."""
        found = self._ids.get(token)
        if found is not None:
            return found
        if grow and not self.frozen:
            return self.add(token)
        return UNK_ID

    def token(self, i: int) -> str:
        return self.tokens[i] if 0 <= i < len(self.tokens) else UNK


class PhoneticTokenizer:
    """Text as sounds, at one of four levels; see the module docstring."""

    def __init__(
        self,
        level: str = PHONEME,
        lexicon: Lexicon | None = None,
        stress: bool = True,
        boundaries: bool = True,
        pauses: bool = True,
        rules: bool = True,
        phonotactics: Phonotactics | None = None,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.level = level
        self.stress = stress
        """Keep the vowels' stress digits (``AH0`` / ``AH1``): off, every vowel is bare (``AH``)."""
        self.boundaries = boundaries
        """Put ``#`` between the sounds of two words."""
        self.pauses = pauses
        """Turn punctuation into pause tokens; off, punctuation is dropped."""
        self.transcriber = Transcriber(lexicon, rules=rules)
        self._phonotactics = phonotactics
        self.vocab = Vocab(SYMBOLS if level == PHONEME else FIXED, frozen=(level == PHONEME))

    # -- properties -----------------------------------------------------------

    @property
    def lexicon(self) -> Lexicon:
        return self.transcriber.lexicon

    @property
    def phonotactics(self) -> Phonotactics:
        """The sound habits, fitted on the lexicon the first time they are needed."""
        if self._phonotactics is None:
            self._phonotactics = Phonotactics.from_lexicon(self.lexicon)
        return self._phonotactics

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def describe(self) -> str:
        return (f"{self.level} tokens, stress {'kept' if self.stress else 'dropped'}, "
                f"{len(self.vocab)} ids, lexicon of {self.lexicon.describe()}")

    def __repr__(self) -> str:
        return f"PhoneticTokenizer({self.describe()})"

    # -- reading text -----------------------------------------------------------

    def pieces(self, text: str) -> list[tuple[str, str]]:
        """The text cut into ``("word", spelling)``, ``("sounds", token)``, ``("boundary", "#")``,
        ``("pause", token)`` and ``("punct", token)`` pieces.

        Punctuation attached to a word becomes the ``punct`` after it (the strongest mark it
        carries); a token that is already a sound (``K``, ``AE1``, ``S.T.R-``, ``K.AE1.T``)
        passes through as ``sounds``, a bare ``#`` as a ``boundary`` and a bare ``,`` ``.`` ``?``
        as a ``pause``.  The last three are tokens already: :meth:`tokenize` keeps them as they
        are, wherever they are, so that any run of tokens - a gram of a graph, say - reads back
        as itself.
        """
        out: list[tuple[str, str]] = []
        for raw in text.split():
            parsed = parse_token(raw)
            if parsed is not None:
                kind, _ = parsed
                if kind == "boundary":
                    out.append(("boundary", raw))
                elif kind == "pause":
                    out.append(("pause", raw))
                elif kind == "special":
                    continue
                else:
                    out.append(("sounds", raw))
                continue
            if raw in _DASHES:
                out.append(("punct", PAUSE_SHORT))
                continue
            word = raw.lstrip(_LEADING)
            pause: str | None = None
            while word and word[-1] in _TRAILING:
                mark = _PAUSE_OF.get(word[-1])
                if mark is not None and (pause is None or _STRENGTH[mark] > _STRENGTH[pause]):
                    pause = mark
                word = word[:-1]
            if word:
                out.append(("word", word))
            if pause is not None:
                out.append(("punct", pause))
        return out

    def transcribe(self, text: str) -> list[list[str]]:
        """The phones of each word of the text, punctuation dropped."""
        words: list[list[str]] = []
        current: list[str] | None = None
        for kind, piece in self.pieces(text):
            if kind == "word":
                if current:
                    words.append(current)
                current = None
                phones = self.transcriber.word(piece)
                if phones:
                    words.append(self._stressed(phones))
            elif kind == "sounds":
                parsed = parse_token(piece)
                assert parsed is not None
                current = (current or []) + list(parsed[1])
            else:
                if current:
                    words.append(current)
                current = None
        if current:
            words.append(current)
        return words

    def _stressed(self, phones: Sequence[str]) -> list[str]:
        return list(phones) if self.stress else strip_stress(phones)

    def tokenize(self, text: str) -> list[Token]:
        """The tokens of a text at this level, boundaries and pauses included."""
        out: list[Token] = []
        word_index = 0
        current: list[str] | None = None
        last: str | None = None  # "word" or "pause"

        def flush() -> None:
            nonlocal current, word_index, last
            if current:
                if last == "word" and self.boundaries:
                    out.append(Token(BOUNDARY, "boundary"))
                out.extend(self._word_tokens(self._stressed(current), word_index))
                word_index += 1
                last = "word"
            current = None

        for kind, piece in self.pieces(text):
            if kind == "word":
                flush()
                phones = self.transcriber.word(piece)
                if phones:
                    current = list(phones)
                    flush()
            elif kind == "sounds":
                parsed = parse_token(piece)
                assert parsed is not None
                current = (current or []) + list(parsed[1])
            elif kind == "boundary":  # a boundary given as a token stays, wherever it is
                flush()
                out.append(Token(BOUNDARY, "boundary"))
                last = "boundary"
            elif kind == "pause":  # so does a pause given as a token
                flush()
                out.append(Token(piece, "pause"))
                last = "pause"
            else:  # punctuation: one pause, after something, never two in a row
                flush()
                if self.pauses and last != "pause" and out:
                    out.append(Token(piece, "pause"))
                    last = "pause"
        flush()
        return out

    def _word_tokens(self, phones: list[str], word: int) -> list[Token]:
        if self.level == PHONEME:
            return [Token(p, "nucleus" if is_vowel(p) else "phone", (p,), word) for p in phones]
        if self.level == WORD:
            return [Token(".".join(phones), "word", tuple(phones), word)]
        out: list[Token] = []
        for s in syllabify(phones):
            if self.level == SYLLABLE:
                out.append(Token(s.text(), "syllable", s.phones, word))
                continue
            if s.onset:
                out.append(Token(".".join(s.onset) + "-", "onset", s.onset, word))
            out.append(Token(s.nucleus, "nucleus", (s.nucleus,), word))
            if s.coda:
                out.append(Token("-" + ".".join(s.coda), "coda", s.coda, word))
        return out

    def tokens(self, text: str) -> list[str]:
        """The token strings."""
        return [t.text for t in self.tokenize(text)]

    def text(self, text: str) -> str:
        """The text form: the tokens joined by single spaces.  ``text(text(x)) == text(x)``."""
        return " ".join(self.tokens(text))

    def encode(self, text: str, grow: bool = True) -> list[int]:
        """The ids of the tokens; an unseen token grows the vocabulary (``grow``) or is ``<unk>``."""
        return [self.vocab.id(t, grow) for t in self.tokens(text)]

    # -- reading tokens back ----------------------------------------------------------

    def parse(self, tokens: Iterable[str]) -> list[Token]:
        """Token strings back into :class:`Token`\\ s (``ValueError`` for one that is not phonetic)."""
        out: list[Token] = []
        word = 0
        for text in tokens:
            parsed = parse_token(text)
            if parsed is None:
                raise ValueError(f"{text!r} is not a phonetic token")
            kind, phones = parsed
            if kind in ("boundary", "pause"):
                if out and out[-1].kind not in ("boundary", "pause"):
                    word += 1
                out.append(Token(text, kind))
            elif kind == "special":
                out.append(Token(text, kind))
            else:
                out.append(Token(text, kind, phones, word))
        return out

    def words_of(self, tokens: Iterable[str | int | Token]) -> list[list[str]]:
        """Group tokens (strings, ids or :class:`Token`\\ s) back into the phones of each word."""
        words: list[list[str]] = []
        current: list[str] = []
        for t in tokens:
            if isinstance(t, int):
                t = self.vocab.token(t)
            if isinstance(t, Token):
                kind, phones = t.kind, t.phones
            else:
                parsed = parse_token(t)
                if parsed is None:
                    continue
                kind, phones = parsed
            if kind in ("boundary", "pause", "special"):
                if current:
                    words.append(current)
                current = []
            else:
                current.extend(phones)
        if current:
            words.append(current)
        return words

    def decode(self, tokens: Iterable[str | int | Token]) -> str:
        """The sounds spelled back as words, with the pauses as punctuation.

        A word the lexicon or the tokenizer's memory knows comes back as itself; other
        sounds are respelled so that they read as they sound.
        """
        out: list[str] = []
        current: list[str] = []
        for t in tokens:
            if isinstance(t, int):
                t = self.vocab.token(t)
            if isinstance(t, Token):
                kind, phones, text = t.kind, t.phones, t.text
            else:
                parsed = parse_token(t)
                if parsed is None:
                    continue
                kind, phones = parsed
                text = t
            if kind in ("boundary", "pause", "special"):
                if current:
                    out.append(self.transcriber.spell(current))
                    current = []
                if kind == "pause" and out:
                    out[-1] = out[-1] + text
            else:
                current.extend(phones)
        if current:
            out.append(self.transcriber.spell(current))
        return " ".join(out)

    def features(self, token: str | int | Token) -> list[float]:
        """The articulatory features of a token: a phone's own; the mean over a cluster's or a
        syllable's phones; zeros for a boundary, a pause or a special (``FEATURE_DIM`` values)."""
        if isinstance(token, int):
            token = self.vocab.token(token)
        text = token.text if isinstance(token, Token) else token
        parsed = parse_token(text)
        if parsed is None or not parsed[1]:
            return [0.0] * FEATURE_DIM
        vectors = [features(p) for p in parsed[1]]
        return [sum(v[i] for v in vectors) / len(vectors) for i in range(FEATURE_DIM)]

    # -- sound ------------------------------------------------------------------------

    def pronounce(self, word: str) -> list[str]:
        """The phones of one word."""
        return self._stressed(self.transcriber.word(word))

    def ipa(self, text: str) -> str:
        """The text in the IPA, a stress mark before each stressed syllable: *the cat* -> ``ðə kæt``."""
        words: list[str] = []
        for phones in self.transcribe(text):
            parts: list[str] = []
            for s in syllabify(phones):
                mark = "ˈ" if s.stress == 1 else "ˌ" if s.stress == 2 else ""
                parts.append(mark + to_ipa(s.phones, stress=False))
            words.append("".join(parts))
        return " ".join(words)

    def syllables(self, word: str) -> list[Syllable]:
        return syllabify(self.pronounce(word))

    def rhymes(self, a: str, b: str) -> bool:
        return rhymes(self.pronounce(a), self.pronounce(b))

    def alliterates(self, a: str, b: str) -> bool:
        return alliterates(self.pronounce(a), self.pronounce(b))

    def affinity(self, a: str, b: str) -> float:
        """How well sound ``b`` follows sound ``a`` (bits of mutual information; 0 is chance)."""
        return self.phonotactics.affinity(a, b)

    def score(self, word: str) -> float:
        """How English a word sounds: the mean log-probability of its sound transitions."""
        return self.phonotactics.score(self.pronounce(word))

    def well_formed(self, word: str) -> bool:
        """Could an English speaker say it?"""
        return self.phonotactics.well_formed(self.pronounce(word))

    def coin(self, rng: random.Random | int | None = None, syllables: int | None = None) -> str:
        """A new, pronounceable word (spelled as it sounds), built from the sound habits."""
        phones = self.phonotactics.build(rng, syllables=syllables, avoid=(p for _, p in self.lexicon.items()))
        return self.transcriber.spell(phones) if phones else ""

    def blend(self, a: str, b: str) -> str:
        """A portmanteau of two words at the joint where their sounds meet best: *smoke* + *fog* -> *smog*."""
        phones, _, _ = self.phonotactics.blend(self.pronounce(a), self.pronounce(b))
        return self.transcriber.spell(phones)

    def explain(self, text: str) -> list[dict]:
        """One row per word: the spelling, its phones, where they came from, the syllables and the IPA."""
        rows: list[dict] = []
        for kind, piece in self.pieces(text):
            if kind != "word":
                continue
            phones, how = self.transcriber.explain(piece)
            phones = self._stressed(phones)
            rows.append({
                "word": piece, "phones": phones, "how": how,
                "syllables": [s.text() for s in syllabify(phones)],
                "ipa": self.ipa(piece), "score": round(self.phonotactics.score(phones), 3) if phones else None,
            })
        return rows

    # -- persistence -------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Everything that was learned by reading: the grown vocabulary, the memory, the counts."""
        return {
            "format": "phonetok", "version": 1,
            "level": self.level, "stress": self.stress, "boundaries": self.boundaries, "pauses": self.pauses,
            "vocab": list(self.vocab.tokens),
            "memory": {w: list(p) for w, p in self.transcriber.memory.items()},
            "counts": dict(self.transcriber.counts),
        }

    def load_dict(self, d: dict) -> "PhoneticTokenizer":
        if d.get("format") != "phonetok":
            raise ValueError("not a phonetok tokenizer document")
        if d.get("level", self.level) != self.level:
            raise ValueError(f"the file holds a {d.get('level')} tokenizer, this one is {self.level}")
        self.stress = bool(d.get("stress", self.stress))
        self.boundaries = bool(d.get("boundaries", self.boundaries))
        self.pauses = bool(d.get("pauses", self.pauses))
        tokens = d.get("vocab") or []
        if self.level != PHONEME:
            self.vocab = Vocab(tokens or FIXED)
        self.transcriber.memory.update({w: tuple(p) for w, p in d.get("memory", {}).items()})
        self.transcriber.counts.update(d.get("counts", {}))
        return self

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str, lexicon: Lexicon | None = None) -> "PhoneticTokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok = cls(level=d.get("level", PHONEME), lexicon=lexicon)
        return tok.load_dict(d)


_STRENGTH = {PAUSE_SHORT: 1, PAUSE_FULL: 2, PAUSE_QUESTION: 3}
