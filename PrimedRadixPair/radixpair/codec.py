"""The encoder/decoder: one tokenizer's two halves, plus what the tree needs to know about its ids.

A codec is chosen at priming, saved whole in the model file and never changed
afterwards: a tree's addresses are measured in its units.  ``encode(text)``
gives unit ids and never marks; ``decode(ids)`` gives text back and drops the
marks; ``padded(text)`` is ``[start] + encode(text) + [end]`` - what the trees
read.  ``emits()`` is what a walk may emit: every id but the start mark and
the dead ones (a tokenizer's ``<pad>``).

The presets (``DESIGN.md`` section 5); ``phones`` is the main one - the model
reads phones and writes phones, and ``spell`` gives the English they spell:

``chars``       3 marks then space, a-z, ``'`` ``.`` ``,``          R = 33
``bytes``       3 marks then the 256 byte values                  R = 259
``bpe``         3 marks then a byte-pair vocabulary trained here  R = vocab_size + 3
``phones``      the phonetic tokenizer's fixed phoneme alphabet   R = 92
``syllables``   its 8 specials then every syllable of a source    R = syllables + 8
``gpt2``        GPT-2's tokenizer from its two files, whole or capped
``external``    another published tokenizer via tiktoken / tokenizers (optional)
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence

from .bpe import BPE
from .gpt2 import Gpt2BPE, gpt2_files

__all__ = [
    "CODECS", "DEFAULT_CODEC", "DEFAULT_L", "Codec", "BpeCodec", "BytesCodec", "CharsCodec", "ExternalCodec", "Gpt2Codec", "PhonesCodec",
    "SyllablesCodec", "START", "END", "UNK", "MARKS", "CHAR_UNITS", "codec_from_dict", "make_codec", "phonetok_module",
]

START, END, UNK = 0, 1, 2
MARKS = ("<s>", "</s>", "<unk>")
CHAR_UNITS = " abcdefghijklmnopqrstuvwxyz'.,"
_WHITESPACE = re.compile(r"\s+")


def phonetok_module(name: str = "", what: str = "a phonetic codec"):
    """The ``phonetok`` package - or its submodule *name* - from the environment or the sibling checkout."""
    full = f"phonetok.{name}" if name else "phonetok"
    try:
        return importlib.import_module(full)
    except ImportError:
        sibling = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "PhoneticTokenizer")
        if os.path.isdir(os.path.join(sibling, "phonetok")) and sibling not in sys.path:
            sys.path.append(sibling)
        try:
            return importlib.import_module(full)
        except ImportError as exc:
            raise ValueError(
                f"{what} needs the phonetok package (the sibling PhoneticTokenizer directory: "
                f"pip install -e ../PhoneticTokenizer): {exc}"
            ) from None


def _texts(texts: Iterable[str] | None, what: str) -> list[str]:
    if texts is None:
        raise ValueError(f"{what} needs texts to close its vocabulary from (tokenizer data)")
    out = [t for t in (texts if not isinstance(texts, str) else [texts]) if t is not None]
    if not out:
        raise ValueError(f"{what} needs at least one text to close its vocabulary from")
    return out


class Codec:
    """The contract; every preset is a subclass.  ``symbols[i]`` is the text of id ``i``."""

    name = "codec"

    def __init__(self, symbols: Sequence[str], start: int, end: int, unk: int | None, dead: Iterable[int] = ()) -> None:
        self.symbols = list(symbols)
        self.R = len(self.symbols)
        self.start = int(start)
        self.end = int(end)
        self.unk = None if unk is None else int(unk)
        self.dead = frozenset(int(d) for d in dead)
        self._emits = [i for i in range(self.R) if i != self.start and i not in self.dead]

    # -- the two halves ------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    def padded(self, text: str) -> list[int]:
        return [self.start] + self.encode(text) + [self.end]

    def spell(self, ids: Iterable[int]) -> str:
        """The ids as the words they spell - the same as ``decode`` unless the units are sounds."""
        return self.decode(ids)

    def join(self, prefix: str, text: str) -> str:
        """A prefix and the text that continues it, as one text: plain concatenation - the prefix carries its own
        trailing space where one belongs.  A tokenizer of words joins them the way it joins words."""
        return prefix + text

    def units(self, ids: Iterable[int]) -> list[int]:
        """The ids that are units: the marks and the dead ids dropped."""
        skip = {self.start, self.end} | self.dead
        return [i for i in ids if i not in skip]

    def emits(self) -> list[int]:
        return list(self._emits)

    def symbol(self, i: int) -> str:
        return self.symbols[i] if 0 <= i < self.R else "<?>"

    def unk_share(self, ids: Sequence[int]) -> float:
        if self.unk is None or not ids:
            return 0.0
        return sum(1 for i in ids if i == self.unk) / len(ids)

    def describe(self) -> str:
        return f"{self.name}: {self.R} ids"

    # -- persistence ------------------------------------------------------------------

    def _state(self) -> dict:
        return {}

    def to_dict(self) -> dict:
        return {"name": self.name, **self._state()}

    @classmethod
    def from_dict(cls, d: dict) -> "Codec":
        return cls(**{k: v for k, v in d.items() if k != "name"})

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


class CharsCodec(Codec):
    """A closed character alphabet: casefolded, whitespace runs to one space, the rest ``<unk>``."""

    name = "chars"

    def __init__(self, units: str = CHAR_UNITS) -> None:
        if len(set(units)) != len(units):
            raise ValueError("chars: the units must be distinct")
        super().__init__(list(MARKS) + list(units), START, END, UNK)
        self.units_str = units
        self._ids = {c: 3 + i for i, c in enumerate(units)}

    @staticmethod
    def normalise(text: str) -> str:
        """Casefold, curly quotes to ``'``, every run of whitespace to one space - a trailing space is kept."""
        text = text.casefold().replace("’", "'").replace("‘", "'").replace("`", "'")
        return _WHITESPACE.sub(" ", text)

    def encode(self, text: str) -> list[int]:
        ids = self._ids
        return [ids.get(c, UNK) for c in self.normalise(text)]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            if i >= 3 and i < self.R:
                out.append(self.symbols[i])
            elif i == UNK:
                out.append("?")
        return "".join(out)

    def _state(self) -> dict:
        return {"units": self.units_str}


class BytesCodec(Codec):
    """UTF-8 bytes; every input is a unit, so ``<unk>`` is never produced."""

    name = "bytes"

    def __init__(self) -> None:
        symbols = list(MARKS) + [chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in range(256)]
        super().__init__(symbols, START, END, None)

    def encode(self, text: str) -> list[int]:
        return [3 + b for b in text.encode("utf-8")]

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(i - 3 for i in ids if 3 <= i < self.R).decode("utf-8", errors="replace")

    @classmethod
    def from_dict(cls, d: dict) -> "BytesCodec":
        return cls()


class BpeCodec(Codec):
    """The repository's byte-level BPE, trained at priming on the tokenizer data."""

    name = "bpe"

    def __init__(self, vocab_size: int = 1024, texts: Iterable[str] | None = None, bpe: BPE | None = None) -> None:
        self.bpe = bpe if bpe is not None else BPE.train(_texts(texts, "the bpe codec"), int(vocab_size))
        symbols = list(MARKS) + [self.bpe.display(i) for i in range(self.bpe.vocab_size)]
        super().__init__(symbols, START, END, None)
        self.vocab_size = self.bpe.vocab_size

    def encode(self, text: str) -> list[int]:
        return [3 + i for i in self.bpe.encode(text)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.bpe.decode(i - 3 for i in ids if 3 <= i < self.R)

    def describe(self) -> str:
        return f"bpe: {self.vocab_size} tokens, {self.R} ids"

    def _state(self) -> dict:
        return {"vocab_size": self.vocab_size, "bpe": self.bpe.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "BpeCodec":
        return cls(vocab_size=d["vocab_size"], bpe=BPE.from_dict(d["bpe"]))


class _Phonetic(Codec):
    """What the two phonetic codecs share: the tokenizer, its specials, encode / decode."""

    level = "phoneme"

    def _make(self, stress: bool, boundaries: bool, pauses: bool):
        phonetok = phonetok_module(what=f"the {self.name} codec")
        self.stress, self.boundaries, self.pauses = bool(stress), bool(boundaries), bool(pauses)
        self.tok = phonetok.PhoneticTokenizer(level=self.level, stress=self.stress, boundaries=self.boundaries,
                                              pauses=self.pauses, lexicon=phonetok.Lexicon.portable())
        self.phonetok = phonetok
        return self.tok

    def _finish(self) -> None:
        phones = self.phonetok.phones
        super().__init__(list(self.tok.vocab.tokens), phones.BOS_ID, phones.EOS_ID, phones.UNK_ID, (phones.PAD_ID,))

    def encode(self, text: str) -> list[int]:
        """English or phones in - the tokenizer reads its own text form unchanged - phone ids out."""
        return self.tok.encode(text, grow=False)

    def decode(self, ids: Iterable[int]) -> str:
        """Phones out: the tokenizer's text form, ``DH AH0 # K AE1 T .``, which ``encode`` reads back unchanged."""
        symbols = self.symbols
        return " ".join(symbols[i] for i in self.units(ids))

    def spell(self, ids: Iterable[int]) -> str:
        """The English the phones spell: ``the cat.``"""
        return self.tok.decode(self.units(ids))

    def join(self, prefix: str, text: str) -> str:
        """Two words apart: the tokenizer puts the boundary ``#`` between them, so an outcome ``mat`` after the
        prefix ``... the`` is ``# M AE1 T`` - the boundary is the outcome's first step, as the model would emit it."""
        if prefix.strip() and text.strip():
            return prefix.rstrip() + " " + text.lstrip()
        return prefix + text

    def _state(self) -> dict:
        return {"stress": self.stress, "boundaries": self.boundaries, "pauses": self.pauses}


class PhonesCodec(_Phonetic):
    """The phonetic tokenizer at the phoneme level: its vocabulary is frozen by the tokenizer itself (92 ids)."""

    name = "phones"
    level = "phoneme"

    def __init__(self, stress: bool = True, boundaries: bool = True, pauses: bool = True) -> None:
        self._make(stress, boundaries, pauses)
        self._finish()

    def describe(self) -> str:
        return f"phones: {self.R} ids, stress {'kept' if self.stress else 'dropped'}"


class SyllablesCodec(_Phonetic):
    """The syllable level, its vocabulary closed at priming from the lexicon or a corpus, then frozen.

    ``vocabulary="lexicon"`` takes every syllable of every word the tokenizer's
    lexicon knows; ``"corpus"`` every syllable the tokenizer data contains;
    ``top`` keeps the most frequent ``top`` of them.  A syllable outside the
    vocabulary reads as ``<unk>``.
    """

    name = "syllables"
    level = "syllable"

    def __init__(self, stress: bool = True, vocabulary: str = "lexicon", texts: Iterable[str] | None = None,
                 top: int | None = None, syllables: Sequence[str] | None = None, boundaries: bool = True,
                 pauses: bool = True) -> None:
        tok = self._make(stress, boundaries, pauses)
        self.vocabulary = vocabulary
        self.top = None if top is None else int(top)
        fixed = set(self.phonetok.tokenizer.FIXED)
        if syllables is None:
            counts: Counter[str] = Counter()
            if vocabulary == "lexicon":
                for word in tok.lexicon.words():
                    counts.update(t for t in tok.tokens(word) if t not in fixed)
            elif vocabulary == "corpus":
                for text in _texts(texts, "the syllables codec with vocabulary='corpus'"):
                    counts.update(t for t in tok.tokens(text) if t not in fixed)
            else:
                raise ValueError(f"vocabulary must be 'lexicon' or 'corpus', got {vocabulary!r}")
            kept = sorted(counts, key=lambda t: (-counts[t], t))
            if self.top is not None:
                kept = kept[: self.top]
            syllables = sorted(kept)
        self.syllables = list(syllables)
        tok.vocab = self.phonetok.tokenizer.Vocab(list(self.phonetok.tokenizer.FIXED) + self.syllables, frozen=True)
        self._finish()

    def describe(self) -> str:
        return (f"syllables: {len(self.syllables)} syllables from the {self.vocabulary}"
                f"{'' if self.top is None else f' (top {self.top})'}, {self.R} ids, "
                f"stress {'kept' if self.stress else 'dropped'}")

    def _state(self) -> dict:
        return {**super()._state(), "vocabulary": self.vocabulary, "top": self.top, "syllables": self.syllables}


class _Capped(Codec):
    """A published vocabulary, whole (its ids unchanged, marks appended) or capped to a corpus's most frequent tokens."""

    def _setup(self, V: int, display, top: int | None, texts: Iterable[str] | None, kept: Sequence[int] | None,
               raw_encode) -> None:
        self.V = int(V)
        self.top = None if top is None else int(top)
        if self.top is None and kept is None:
            symbols = [display(i) for i in range(self.V)] + list(MARKS)
            Codec.__init__(self, symbols, self.V, self.V + 1, self.V + 2)
            self.kept = None
            self._map = None
            return
        if kept is None:
            if self.top is not None and self.top < 1:
                raise ValueError(f"top must be >= 1, got {self.top}")
            freq: Counter[int] = Counter()
            for text in _texts(texts, f"the {self.name} codec with a cap"):
                freq.update(raw_encode(text))
            kept = sorted(freq, key=lambda i: (-freq[i], i))[: self.top]
        self.kept = [int(i) for i in kept]
        self._map = {i: 3 + r for r, i in enumerate(self.kept)}
        Codec.__init__(self, list(MARKS) + [display(i) for i in self.kept], START, END, UNK)

    def _to_ids(self, raw: list[int]) -> list[int]:
        if self._map is None:
            return raw
        m = self._map
        return [m.get(i, UNK) for i in raw]

    def _to_raw(self, ids: Iterable[int]) -> list[int]:
        if self._map is None:
            return [i for i in ids if 0 <= i < self.V]
        kept = self.kept
        return [kept[i - 3] for i in ids if 3 <= i < self.R]


class Gpt2Codec(_Capped):
    """GPT-2's tokenizer from ``encoder.json`` + ``vocab.bpe`` (or ``vocab.json`` + ``merges.txt``)."""

    name = "gpt2"

    def __init__(self, files: str | Sequence[str] | None = None, top: int | None = None,
                 texts: Iterable[str] | None = None, bpe: Gpt2BPE | None = None,
                 kept: Sequence[int] | None = None) -> None:
        if bpe is None:
            if files is None:
                raise ValueError("gpt2: give the directory holding encoder.json and vocab.bpe, or the two paths")
            if isinstance(files, str):
                found = gpt2_files(files)
                if found is None:
                    raise ValueError(f"gpt2: no encoder.json + vocab.bpe (or vocab.json + merges.txt) in {files}")
                files = found
            bpe = Gpt2BPE.load(files[0], files[1])
        self.bpe = bpe
        self._setup(bpe.vocab_size, bpe.display, top, texts, kept, bpe.encode)

    def encode(self, text: str) -> list[int]:
        return self._to_ids(self.bpe.encode(text))

    def decode(self, ids: Iterable[int]) -> str:
        return self.bpe.decode(self._to_raw(ids))

    def describe(self) -> str:
        cap = "whole" if self.kept is None else f"capped to {len(self.kept)} tokens"
        return f"gpt2: {self.V} tokens {cap}, {self.R} ids"

    def _state(self) -> dict:
        return {"top": self.top, "kept": self.kept, "bpe": self.bpe.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Gpt2Codec":
        return cls(bpe=Gpt2BPE.from_dict(d["bpe"]), top=d.get("top"), kept=d.get("kept"))


class ExternalCodec(_Capped):
    """Any other published tokenizer: ``tiktoken:<encoding>`` or ``hf:<tokenizer.json>``; optional packages."""

    name = "external"

    def __init__(self, spec: str, top: int | None = None, texts: Iterable[str] | None = None,
                 kept: Sequence[int] | None = None, digest: str | None = None) -> None:
        self.spec = str(spec)
        kind, _, arg = self.spec.partition(":")
        if kind == "tiktoken":
            try:
                import tiktoken  # type: ignore
            except ImportError as exc:
                raise ValueError(f"external: {self.spec} needs the tiktoken package: {exc}") from None
            enc = tiktoken.get_encoding(arg)
            V = enc.n_vocab
            self._raw_encode = lambda text: enc.encode(text, disallowed_special=())
            self._raw_decode = enc.decode

            def display(i: int) -> str:
                try:
                    return enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
                except Exception:
                    return f"<{i}>"
        elif kind == "hf":
            try:
                from tokenizers import Tokenizer  # type: ignore
            except ImportError as exc:
                raise ValueError(f"external: {self.spec} needs the tokenizers package: {exc}") from None
            tok = Tokenizer.from_file(arg)
            V = tok.get_vocab_size()
            self._raw_encode = lambda text: tok.encode(text, add_special_tokens=False).ids
            self._raw_decode = lambda ids: tok.decode(list(ids), skip_special_tokens=False)

            def display(i: int) -> str:
                return tok.id_to_token(i) or f"<{i}>"
        else:
            raise ValueError(f"external: spec must be 'tiktoken:<encoding>' or 'hf:<path>', got {self.spec!r}")
        self._display = display
        self._setup(V, display, top, texts, kept, self._raw_encode)
        h = hashlib.sha256()
        for i in range(self.V):
            h.update(f"{i}\t{display(i)}\n".encode("utf-8"))
        self.digest = h.hexdigest()
        if digest is not None and digest != self.digest:
            raise ValueError(f"external: the installed {self.spec} vocabulary digests differently from the model's")

    def encode(self, text: str) -> list[int]:
        return self._to_ids(list(self._raw_encode(text)))

    def decode(self, ids: Iterable[int]) -> str:
        return self._raw_decode(self._to_raw(ids))

    def describe(self) -> str:
        cap = "whole" if self.kept is None else f"capped to {len(self.kept)} tokens"
        return f"external {self.spec}: {self.V} tokens {cap}, {self.R} ids"

    def _state(self) -> dict:
        return {"spec": self.spec, "top": self.top, "kept": self.kept, "V": self.V, "digest": self.digest}

    @classmethod
    def from_dict(cls, d: dict) -> "ExternalCodec":
        return cls(d["spec"], top=d.get("top"), kept=d.get("kept"), digest=d.get("digest"))


CODECS: dict[str, type[Codec]] = {
    "chars": CharsCodec, "bytes": BytesCodec, "bpe": BpeCodec, "phones": PhonesCodec,
    "syllables": SyllablesCodec, "gpt2": Gpt2Codec, "external": ExternalCodec,
}


DEFAULT_CODEC = "phones"
"""The main tokenizer: the phonetic tokenizer at the phoneme level."""
DEFAULT_L = {"phones": 3, "syllables": 2, "chars": 4, "bytes": 2, "bpe": 2, "gpt2": 2, "external": 1}
"""The depth each preset primes to under the default ceiling."""


def make_codec(name: str = DEFAULT_CODEC, **options) -> Codec:
    """A codec by preset name (``phones`` by default); ``options`` are the preset's constructor arguments."""
    try:
        cls = CODECS[name]
    except KeyError:
        raise ValueError(f"unknown codec {name!r}; one of {', '.join(CODECS)}") from None
    return cls(**options)


def codec_from_dict(d: dict) -> Codec:
    try:
        cls = CODECS[d["name"]]
    except KeyError:
        raise ValueError(f"unknown codec {d.get('name')!r} in the model file") from None
    return cls.from_dict(d)
