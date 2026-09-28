"""The traditional LLM tokenizer: byte-level byte-pair encoding, the way GPT-2 and its descendants read text.

A text is cut into *pre-tokens* by a fixed rule (a word with the space before it, a run of digits,
a run of punctuation, a run of whitespace, an English contraction), each pre-token is turned into its
UTF-8 bytes, and the bytes are merged pairwise by a ranked list of **merges** learned from a corpus:
the pair with the best rank anywhere in the pre-token is merged everywhere it occurs, left to right,
until no pair of the list is left.  What remains are the **tokens**.  Every byte is a token of its
own, so nothing is unreadable - a word never seen is spelled out in the pieces the merges know - and
there is no ``<unk>``.

The pieces are the classic ones:

* the **byte alphabet** - GPT-2's ``bytes_to_unicode``: every byte is written as one printable
  character, ``Ġ`` for the space, ``Ċ`` for the newline, the printable ASCII and Latin-1 bytes as
  themselves - so a token is always a string of printable characters (``"Ġcat"``);
* the **pre-tokenizer** - GPT-2's pattern, ``'s|'t|'re|'ve|'m|'ll|'d| ?L+| ?D+| ?P+|\\s+(?!\\S)|\\s+``,
  over character classes this module defines exactly (:func:`char_class`), so the Go and Rust ports
  cut a text in the same places;
* the **merges** - ranked pairs, stored as GPT-2 stores them, one ``left right`` pair per line of a
  ``merges.txt`` file (:meth:`BPETokenizer.load`, :meth:`BPETokenizer.dumps`), and **learned** by
  counting the most frequent adjacent pair over a corpus (:meth:`BPETokenizer.learn`);
* **ids** - the 256 bytes are ids 0-255, the tokens the merges make follow in rank order, and the
  special token ``<|endoftext|>`` closes the vocabulary (:meth:`BPETokenizer.encode`,
  :meth:`BPETokenizer.decode`);
* a **leading space** is put before every text, as SentencePiece-based models do, so the first word
  of a text is the same token as the same word anywhere else; :meth:`BPETokenizer.decode` takes it
  away again, and a round trip gives back the text exactly.

**The text form.**  The graph of :mod:`radixnet` is built over units written as text - a gram is its
units joined by single spaces - so the tokens need a text form that the tokenizer reads back as the
very tokens it was made of, any run of them included (a label of the graph starts and ends wherever
a gram does).  A token that is a space followed by printable ASCII is written *bare*: ``"Ġcat"`` is
``cat``, so a text of plain words is already the text form of itself.  Every other token is written
*glued*: :data:`GLUE` (``⁀``, U+2040 CHARACTER TIE) and then the token in the byte alphabet -
``"ing"`` is ``⁀ing``, ``"."`` is ``⁀.``, a newline ``⁀Ċ``.  ``"The cat sat."`` is ``The cat sat ⁀.``
and ``"walking"`` is ``walk ⁀ing``.  ``⁀`` is outside the byte alphabet, so no token contains it,
and a text either is the text form (every piece bare or glued, single spaces between) - and is read
piece by piece - or is read as text through the tokenizer; where both readings apply they agree,
which is what makes the text form idempotent (``../SPEC-Tokens.md`` says why).

Standard library only, like the rest of the package; ``go/radixnet/bpe.go`` and ``rust/src/bpe.rs``
are the same tokenizer, token for token.
"""

from __future__ import annotations

import heapq
import os
import re
from collections.abc import Iterable, Sequence

__all__ = [
    "BPETokenizer",
    "ENDOFTEXT",
    "GLUE",
    "MERGES_FILE",
    "BYTE_CHARS",
    "byte_text",
    "text_bytes",
    "utf8",
    "char_class",
    "pretokenize",
    "bundled_path",
    "default_tokenizer",
]

GLUE = "⁀"
"""``⁀``: the mark of a glued token in the text form.  Outside the byte alphabet, so no token holds it."""

ENDOFTEXT = "<|endoftext|>"
"""The special token that closes the vocabulary: the end of a document, as GPT-2 has it."""

MERGES_FILE = "merges.txt"
"""The bundled merges, beside this module in ``data/``."""

FORMAT = "radixnet-bpe"
"""The first line of a merges file this module writes: ``#version: 0.2 radixnet-bpe``."""

MIN_VOCAB = 257
"""The smallest vocabulary: the 256 bytes and the special token."""


# ---------------------------------------------------------------------------
# the byte alphabet
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> list[str]:
    """GPT-2's map: printable ASCII and Latin-1 bytes are themselves, the rest move to U+0100 upwards."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = [""] * 256
    for b in keep:
        chars[b] = chr(b)
    moved = 0
    for b in range(256):
        if not chars[b]:
            chars[b] = chr(256 + moved)
            moved += 1
    return chars


BYTE_CHARS: tuple[str, ...] = tuple(_bytes_to_unicode())
"""The character each byte is written as: ``BYTE_CHARS[0x20] == "Ġ"``, ``BYTE_CHARS[0x0A] == "Ċ"``."""

_CHAR_BYTES: dict[str, int] = {c: b for b, c in enumerate(BYTE_CHARS)}
_BYTE_TABLE = {b: c for b, c in enumerate(BYTE_CHARS)}


def utf8(text: str) -> bytes:
    """A text's UTF-8 bytes.  A lone surrogate (which only Python can hold) is U+FFFD, as Go and Rust
    would have read it."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace").encode("utf-8")


def byte_text(data: bytes) -> str:
    """Bytes in the byte alphabet: ``b" cat" -> "Ġcat"``."""
    return data.decode("latin-1").translate(_BYTE_TABLE)


def text_bytes(text: str) -> bytes | None:
    """The bytes a string of the byte alphabet stands for; ``None`` if a character is not in it."""
    out = bytearray()
    for ch in text:
        b = _CHAR_BYTES.get(ch)
        if b is None:
            return None
        out.append(b)
    return bytes(out)


# ---------------------------------------------------------------------------
# the pre-tokenizer
# ---------------------------------------------------------------------------

SPACE, DIGIT, PUNCT, LETTER = "S", "D", "P", "L"

_WHITE_SPACE = "\t\n\x0b\x0c\r \x85\xa0            " \
               "    　"
"""The Unicode ``White_Space`` property: the 25 code points Go's ``unicode.IsSpace`` and Rust's
``char::is_whitespace`` answer yes for (Python's ``str.isspace`` adds four separators, ``split_words``
takes them away again)."""

_PUNCT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0000, 0x002F),  # controls, and ! " # $ % & ' ( ) * + , - . /  (the whitespace among them is SPACE first)
    (0x003A, 0x0040),  # : ; < = > ? @
    (0x005B, 0x0060),  # [ \ ] ^ _ `
    (0x007B, 0x00BF),  # { | } ~, DEL, the C1 controls, and Latin-1 punctuation and symbols (¡ « ° ± » ¿ ...)
    (0x00D7, 0x00D7),  # ×
    (0x00F7, 0x00F7),  # ÷
    (0x2000, 0x206F),  # General Punctuation: dashes, quotes, bullets, the ellipsis, zero-width marks
    (0x20A0, 0x20CF),  # currency symbols
    (0x2190, 0x2BFF),  # arrows, mathematical operators, technical, box drawing, shapes, symbols, dingbats
    (0x2E00, 0x2E7F),  # supplemental punctuation
    (0x3001, 0x3003),  # 、 。 〃
    (0x3008, 0x3011),  # CJK brackets
    (0x3014, 0x301F),  # more CJK brackets, the wave dash, CJK quotation marks
    (0xFE00, 0xFE0F),  # variation selectors (an emoji's presentation stays with the emoji)
    (0xFF01, 0xFF0F),  # fullwidth ! " # ... /
    (0xFF1A, 0xFF20),  # fullwidth : ; < = > ? @
    (0xFF3B, 0xFF40),  # fullwidth [ \ ] ^ _ `
    (0xFF5B, 0xFF65),  # fullwidth { | } ~ and halfwidth CJK punctuation
    (0x1F000, 0x1FAFF),  # emoji, pictographs and the symbols around them
    (0xE0020, 0xE007F),  # tag characters (the flags of emoji)
)
"""Where :data:`PUNCT` lies: every ASCII character that is not a letter, a digit or whitespace, and a
fixed list of Unicode blocks.  A table rather than the Unicode categories, because the Rust standard
library has no categories to ask, and a pre-tokenizer that cut differently in one port would give
that port different tokens."""


def char_class(ch: str) -> str:
    """The class of one character in the pre-tokenizer: :data:`SPACE`, :data:`DIGIT`, :data:`PUNCT` or :data:`LETTER`.

    ``SPACE`` is the Unicode ``White_Space`` property, ``DIGIT`` the ASCII digits, ``PUNCT`` the table
    :data:`_PUNCT_RANGES`, and ``LETTER`` everything else - the ASCII letters and every other character
    of every script.  The order matters: a space is a space before it is anything else.
    """
    if ch in _WHITE_SPACE:
        return SPACE
    code = ord(ch)
    if 0x30 <= code <= 0x39:
        return DIGIT
    for lo, hi in _PUNCT_RANGES:
        if code < lo:
            break
        if code <= hi:
            return PUNCT
    return LETTER


def _class_pattern() -> tuple[str, str, str]:
    space = "".join(re.escape(c) for c in _WHITE_SPACE)
    parts = []
    for lo, hi in _PUNCT_RANGES:
        parts.append(re.escape(chr(lo)) if lo == hi else f"{re.escape(chr(lo))}-{re.escape(chr(hi))}")
    punct = "".join(parts)
    return space, "0-9", punct


def _compile_pretokenizer() -> re.Pattern[str]:
    space, digit, punct = _class_pattern()
    s = f"[{space}]"
    not_s = f"[^{space}]"
    d = f"[{digit}]"
    # a punctuation character that is not whitespace: the table covers the ASCII controls, and some of
    # those are whitespace, which is SPACE first
    p = f"(?:(?!{s})[{punct}])"
    l_ = f"[^{space}{digit}{punct}]"
    contraction = "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])"
    return re.compile(f"{contraction}| ?{l_}+| ?{d}+| ?{p}+|{s}+(?!{not_s})|{s}+")


_PRETOKENIZER = _compile_pretokenizer()


def pretokenize(text: str) -> list[str]:
    """The pre-tokens of a text, GPT-2's way: ``"Hello, world!" -> ["Hello", ",", " world", "!"]``.

    A word, a run of digits or a run of punctuation takes the one space before it; a run of
    whitespace leaves its last space to the word after it; ``'s 't 're 've 'm 'll 'd`` stand alone
    (in either case).  The pre-tokens join back into the text exactly, and no merge ever crosses
    from one into the next.
    """
    return _PRETOKENIZER.findall(text)


# ---------------------------------------------------------------------------
# the tokenizer
# ---------------------------------------------------------------------------

_CACHE_LIMIT = 1 << 16


def _merge_word(word: Sequence[str], ranks: dict[tuple[str, str], int]) -> list[str]:
    """Byte-pair encode one pre-token given as its byte characters: GPT-2's ``bpe``.

    The adjacent pair with the best (lowest) rank is merged everywhere it occurs, left to right and
    without overlap, and the search starts again, until no adjacent pair has a rank.
    """
    parts = list(word)
    while len(parts) > 1:
        best = None
        best_rank = None
        for pair in zip(parts, parts[1:]):
            rank = ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best, best_rank = pair, rank
        if best is None:
            break
        first, second = best
        merged: list[str] = []
        i = 0
        n = len(parts)
        while i < n:
            if i < n - 1 and parts[i] == first and parts[i + 1] == second:
                merged.append(first + second)
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        parts = merged
    return parts


class BPETokenizer:
    """A byte-level BPE tokenizer: ranked merges over the byte alphabet, ids, and the text form.

    ``BPETokenizer()`` knows only the 256 bytes; :meth:`bundled`, :meth:`load` and :meth:`learn` give
    one worth having.  The vocabulary is the bytes (ids 0-255, one per byte value), then each token
    the merges make, in rank order (a merge that makes a token already there adds none), then the
    special tokens.
    """

    def __init__(
        self,
        merges: Iterable[tuple[str, str]] = (),
        special: Sequence[str] = (ENDOFTEXT,),
        note: str = "",
    ) -> None:
        self.vocab: list[str] = list(BYTE_CHARS)
        self._ids: dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        for left, right in merges:
            self._add_merge(left, right)
        self.special: list[str] = list(special)
        self._special_at: dict[str, int] = {}  # a special token's place among the special tokens
        for token in self.special:
            if token in self._special_at:
                raise ValueError(f"the special token {token!r} is listed twice")
            self._special_at[token] = len(self._special_at)
        self.note = note
        self._words: dict[str, tuple[str, ...]] = {}
        self._bare: dict[str, bool] = {}
        self._special_re = (
            re.compile("|".join(re.escape(s) for s in sorted(self.special, key=len, reverse=True))) if self.special else None
        )

    def _add_merge(self, left: str, right: str) -> None:
        pair = (left, right)
        if pair in self.ranks:
            raise ValueError(f"merge {len(self.merges) + 1} ({left} {right}) is listed twice")
        for side in pair:
            if side not in self._ids:
                raise ValueError(
                    f"merge {len(self.merges) + 1} ({left} {right}): {side!r} is not a token yet - a merge joins two "
                    f"tokens the bytes and the merges before it made"
                )
        self.ranks[pair] = len(self.merges)
        self.merges.append(pair)
        token = left + right
        if token not in self._ids:
            self._ids[token] = len(self.vocab)
            self.vocab.append(token)

    # -- what it is ------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Every id: the bytes, the tokens the merges made, and the special tokens."""
        return len(self.vocab) + len(self.special)

    def __len__(self) -> int:
        return self.vocab_size

    def token_id(self, token: str) -> int | None:
        """The id of a token written in the byte alphabet (``"Ġcat"``), or of a special token; ``None`` if it is neither."""
        found = self._ids.get(token)
        if found is not None:
            return found
        at = self._special_at.get(token)
        return None if at is None else len(self.vocab) + at

    def id_token(self, token_id: int) -> str:
        """The token an id stands for, in the byte alphabet (a special token as itself)."""
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        index = token_id - len(self.vocab)
        if 0 <= index < len(self.special):
            return self.special[index]
        raise ValueError(f"id {token_id} is outside a vocabulary of {self.vocab_size}")

    def describe(self) -> dict:
        """What this tokenizer is, as the CLI and the API report it."""
        return {
            "kind": "byte-level BPE",
            "vocab_size": self.vocab_size,
            "merges": len(self.merges),
            "tokens": len(self.vocab),
            "special": list(self.special),
            "note": self.note,
        }

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"

    # -- the encoder -------------------------------------------------------------

    def _word(self, pretoken: str) -> tuple[str, ...]:
        """The tokens of one pre-token, cached (the same word is met again and again)."""
        found = self._words.get(pretoken)
        if found is None:
            found = tuple(_merge_word(byte_text(utf8(pretoken)), self.ranks))
            if len(self._words) >= _CACHE_LIMIT:
                self._words.clear()
            self._words[pretoken] = found
        return found

    def _tokens_of(self, text: str) -> list[str]:
        out: list[str] = []
        for pretoken in pretokenize(text):
            out.extend(self._word(pretoken))
        return out

    def tokens(self, text: str) -> list[str]:
        """The tokens of a text, in the byte alphabet: ``"the cat." -> ["Ġthe", "Ġcat", "."]``.

        The text is read with a space before it (see the module), and never finds a special token:
        those are :meth:`encode` business, and only when it is asked to.
        """
        return self._tokens_of(" " + text)

    def encode(self, text: str, allowed_special: bool = False) -> list[int]:
        """The ids of a text.  ``allowed_special`` turns ``<|endoftext|>`` in the text into its id;
        without it the text is read as the characters it is, like any other."""
        framed = " " + text
        if not allowed_special or self._special_re is None:
            return [self._ids[t] for t in self._tokens_of(framed)]
        ids: list[int] = []
        at = 0
        for found in self._special_re.finditer(framed):
            ids.extend(self._ids[t] for t in self._tokens_of(framed[at:found.start()]))
            ids.append(len(self.vocab) + self._special_at[found.group()])
            at = found.end()
        ids.extend(self._ids[t] for t in self._tokens_of(framed[at:]))
        return ids

    # -- the decoder -------------------------------------------------------------

    def token_bytes(self, token: str) -> bytes:
        """The bytes a token stands for; a special token is its own text."""
        if token in self._special_at:
            return utf8(token)
        data = text_bytes(token)
        if data is None:
            raise ValueError(f"{token!r} is not a token of the byte alphabet")
        return data

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """The bytes of a run of ids, with the leading space the encoder put there taken away."""
        data = b"".join(self.token_bytes(self.id_token(int(i))) for i in ids)
        return data[1:] if data.startswith(b" ") else data

    def decode(self, ids: Iterable[int]) -> str:
        """The text of a run of ids: the inverse of :meth:`encode`, exactly.

        A run cut inside a character (ids the model predicted, say) decodes what is broken as
        U+FFFD, one for each maximal broken piece.
        """
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # -- the text form -------------------------------------------------------------

    @staticmethod
    def render(token: str) -> str:
        """A token as the text form writes it: ``"Ġcat" -> "cat"``, ``"ing" -> "⁀ing"``, ``"Ċ" -> "⁀Ċ"``."""
        if len(token) >= 2 and token[0] == "Ġ" and all("!" <= c <= "~" for c in token[1:]):
            return token[1:]
        return GLUE + token

    def read(self, piece: str) -> str | None:
        """The token a piece of the text form stands for, or ``None`` if it is not one.

        A glued piece (``⁀ing``) is the token after the mark, if the vocabulary has it.  A bare piece
        (``cat``) is printable ASCII, and stands for the token of a space and itself - but only when
        that is what the tokenizer would make of it as text (``" cat"`` is one pre-token and one
        token), so that reading a text of plain words piece by piece and reading it as text agree.
        """
        if not piece:
            return None
        if piece[0] == GLUE:
            token = piece[1:]
            return token if token in self._ids else None
        ok = self._bare.get(piece)
        if ok is None:
            ok = all("!" <= c <= "~" for c in piece) and self._tokens_of(" " + piece) == ["Ġ" + piece]
            if len(self._bare) >= _CACHE_LIMIT:
                self._bare.clear()
            self._bare[piece] = ok
        return "Ġ" + piece if ok else None

    def units(self, text: str) -> list[str]:
        """A text as the units of the text form: ``"The cat sat." -> ["The", "cat", "sat", "⁀."]``.

        Text that already is the text form - single spaces between pieces that each :meth:`read` -
        is read piece by piece and comes back as it is (a glued piece of a token that is written bare
        comes back bare); anything else is read as text.  So a label of the graph cuts into the units
        it was made of, and a prefix typed as text is looked for as the tokens it makes.
        """
        if not text:
            return []
        pieces = text.split(" ")
        tokens: list[str] = []
        for piece in pieces:
            token = self.read(piece)
            if token is None:
                break
            tokens.append(token)
        else:
            return [self.render(t) for t in tokens]
        return [self.render(t) for t in self.tokens(text)]

    def text(self, text: str) -> str:
        """The text form of a text: its units joined by single spaces.  Idempotent."""
        return " ".join(self.units(text))

    def token_ids(self, text: str) -> list[int]:
        """The ids of a text given as text or in the text form (its :meth:`units`)."""
        out = []
        for unit in self.units(text):
            token = unit[1:] if unit[0] == GLUE else "Ġ" + unit
            out.append(self._ids[token])
        return out

    def bytes_of(self, text: str) -> bytes:
        """The bytes of a text's :meth:`units`, the leading space kept: what a walk said, byte by byte."""
        return b"".join(self.token_bytes(unit[1:] if unit[0] == GLUE else "Ġ" + unit) for unit in self.units(text))

    def spell(self, text: str) -> str:
        """What a text of the text form says: ``"The cat sat ⁀."`` -> ``"The cat sat."``.

        The inverse of :meth:`units` on text: the tokens back to their bytes, the leading space
        taken away, the bytes read as UTF-8.
        """
        return self.decode(self.token_ids(text))

    # -- learning --------------------------------------------------------------------

    @classmethod
    def learn(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
        special: Sequence[str] = (ENDOFTEXT,),
        note: str = "",
    ) -> "BPETokenizer":
        """Learn the merges from a corpus, the classic way.

        Every text is pre-tokenized (with its leading space) and its pre-tokens counted; then, again
        and again, the adjacent pair of tokens that occurs most often across the corpus is merged
        wherever it occurs, and the merge is ranked next.  It stops when the vocabulary - the bytes,
        the tokens made and the special tokens - reaches ``vocab_size``, or when the best pair occurs
        fewer than ``min_frequency`` times.  Ties go to the pair whose bytes sort first, so the same
        corpus always learns the same merges.
        """
        if vocab_size < MIN_VOCAB:
            raise ValueError(f"vocab_size must be at least {MIN_VOCAB} (the 256 bytes and the special token), got {vocab_size}")
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")
        counts: dict[str, int] = {}
        for text in texts:
            for pretoken in pretokenize(" " + text):
                counts[pretoken] = counts.get(pretoken, 0) + 1
        words: list[list[str]] = []
        freq: list[int] = []
        for pretoken in sorted(counts):
            words.append(list(byte_text(utf8(pretoken))))
            freq.append(counts[pretoken])

        pair_counts: dict[tuple[str, str], int] = {}
        where: dict[tuple[str, str], set[int]] = {}
        for index, word in enumerate(words):
            f = freq[index]
            for pair in zip(word, word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + f
                where.setdefault(pair, set()).add(index)

        def key(pair: tuple[str, str]) -> tuple[bytes, bytes]:
            return text_bytes(pair[0]), text_bytes(pair[1])  # type: ignore[return-value]

        heap = [(-c, key(p), p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        tok = cls(special=special, note=note)
        while tok.vocab_size < vocab_size and heap:
            negative, _, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count != -negative:
                continue  # a stale entry: the pair went back on the heap with the count it has now
            if count < min_frequency:
                break
            if pair not in tok.ranks:
                tok._add_merge(*pair)
            # else: a later merge made a token equal to one side again, and the pair is back; the
            # encoder merges it at the rank it already has, so the corpus does too, and no rank is added
            first, second = pair
            joined = first + second
            touched: dict[tuple[str, str], int] = {}
            for index in sorted(where.pop(pair, ())):
                word = words[index]
                f = freq[index]
                for old in zip(word, word[1:]):
                    pair_counts[old] -= f
                    touched[old] = pair_counts[old]
                merged: list[str] = []
                i, n = 0, len(word)
                while i < n:
                    if i < n - 1 and word[i] == first and word[i + 1] == second:
                        merged.append(joined)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[index] = merged
                for new in zip(merged, merged[1:]):
                    pair_counts[new] = pair_counts.get(new, 0) + f
                    touched[new] = pair_counts[new]
                    where.setdefault(new, set()).add(index)
            pair_counts.pop(pair, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if p == pair:
                    continue
                if c > 0:
                    heapq.heappush(heap, (-c, key(p), p))
                else:
                    pair_counts.pop(p, None)
                    where.pop(p, None)
        return tok

    # -- the file --------------------------------------------------------------------

    def dumps(self) -> str:
        """The merges file: GPT-2's ``merges.txt`` - a version line, then one ``left right`` pair per line, by rank."""
        lines = [f"#version: 0.2 {FORMAT}"]
        for line in self.note.splitlines():
            lines.append(f"#note: {line}" if line else "#note:")
        lines.extend(f"{a} {b}" for a, b in self.merges)
        return "\n".join(lines) + "\n"

    def save(self, path: str) -> None:
        """Write :meth:`dumps` to ``path`` (atomically: a crash leaves the old file or the new one)."""
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.dumps())
        os.replace(tmp, path)

    @classmethod
    def loads(cls, text: str, special: Sequence[str] = (ENDOFTEXT,)) -> "BPETokenizer":
        """Read a merges file: lines of ``left right``; ``#`` starts a comment line, ``#note:`` lines are kept.

        GPT-2's own ``merges.txt`` is in this format.  Each side must be in the byte alphabet and a token
        by the time its merge comes, or the file is refused.
        """
        merges: list[tuple[str, str]] = []
        notes: list[str] = []
        for number, raw in enumerate(text.splitlines(), 1):
            line = raw.rstrip("\r")
            if not line.strip():
                continue
            if line.startswith("#"):
                if line.startswith("#note:"):
                    notes.append(line[len("#note:"):].strip())
                continue
            parts = line.split(" ")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"line {number}: a merge is two tokens separated by one space, got {line!r}")
            for side in parts:
                if text_bytes(side) is None:
                    raise ValueError(f"line {number}: {side!r} is not written in the byte alphabet")
            merges.append((parts[0], parts[1]))
        try:
            return cls(merges, special=special, note="\n".join(notes))
        except ValueError as exc:
            raise ValueError(f"merges: {exc}") from None

    @classmethod
    def load(cls, path: str, special: Sequence[str] = (ENDOFTEXT,)) -> "BPETokenizer":
        """Read a merges file from disk (:meth:`loads`)."""
        with open(path, encoding="utf-8") as fh:
            return cls.loads(fh.read(), special=special)

    @classmethod
    def bundled(cls) -> "BPETokenizer":
        """The merges that ship with the package (``data/merges.txt``)."""
        return cls.load(bundled_path())


def bundled_path() -> str:
    """Where the bundled merges file is."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", MERGES_FILE)


ENV = "RADIXNET_TOKENIZER"
"""The environment variable that names a merges file to read instead of the bundled one."""

_DEFAULT: list[BPETokenizer] = []


def default_tokenizer() -> BPETokenizer:
    """The tokenizer the ``token`` unit reads through: the file :data:`ENV` names, else the bundled merges.

    Made once and kept.  A model's tokens mean nothing without the merges that made them, so the
    rule is the one the phonetic lexicon and the acoustic codebook have: train and predict with the
    same file.  Raises :class:`ValueError` when the file cannot be read.
    """
    if not _DEFAULT:
        path = os.environ.get(ENV) or ""
        try:
            tok = BPETokenizer.load(path) if path else BPETokenizer.bundled()
        except OSError as exc:
            where = f"{ENV}={path}" if path else "the bundled merges"
            raise ValueError(f"the token unit needs its merges ({where}): {exc}") from None
        except ValueError as exc:
            where = f"{ENV}={path}" if path else "the bundled merges"
            raise ValueError(f"the token unit cannot read its merges ({where}): {exc}") from None
        _DEFAULT.append(tok)
    return _DEFAULT[0]


def reset_default_tokenizer() -> None:
    """Forget the tokenizer :func:`default_tokenizer` made, so the next call reads :data:`ENV` again (tests)."""
    _DEFAULT.clear()
