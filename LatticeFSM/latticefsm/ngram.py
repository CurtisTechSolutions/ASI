"""Predict the next thing: an N-gram model over the lattice's adaptive edges.

A finite state machine whose states are the last ``n - 1`` tokens *is* an
N-gram model: reading a token moves it to the context that ends in that token,
and what it predicts is the edge it is most likely to take next.  The dense
``S x A x S`` matrix of :mod:`latticefsm.lattice` cannot hold that machine for
a real vocabulary (a thousand words make a trillion cells of a trigram), so
:class:`NGram` keeps it sparse: an edge exists once ``(context, token)`` has
been seen, and it carries what a lattice edge carries that matters here - how
often it was taken (``count``), when (``last``), what it was rewarded and
punished, and a ``width`` that use and reward widen and punishment narrows.

The next-token distribution is **interpolated Kneser-Ney**, the standard
smoothing for N-grams (Chen & Goodman, 1998): the highest order from raw
counts, every lower order from *continuation* counts (in how many different
contexts a token was seen, not how often), each order discounted by Ney's
``D = n1 / (n1 + 2 n2)`` and interpolated down to a uniform floor, so every
token has some probability and a token seen in many places beats one seen
often in one.  On top of it the lattice's own idea: the **stimulation**
prefers the wide channel - ``P(w | h) ∝ P_KN(w | h) · width(h, w) ^ stimulation``,
renormalised - and credit (:meth:`NGram.credit_text`, from a judge such as an
LLM) widens or narrows the edges a text took.  At stimulation 0, or with no
edge credited, it is plain Kneser-Ney.

Tokens are ``word`` s (lower-cased words, punctuation on its own), ``char`` s,
``sound`` s (the phonetic tokenizer's phonemes, ``#`` between words, spelled
back as English) or ``base64`` letters (:mod:`latticefsm.b64`: any bytes - a
text, a picture, a sound, a video - in one 64-letter alphabet, decoded back
to bytes); every line (or, for ``base64``, every 76 letters of a file) is one
sequence, padded with ``<s>`` before it and ending with ``</s>``.  Every count is an integer and every probability is
computed in one fixed order, so the Rust crate (``ngram.rs``) gives the same
numbers for the same model, and reads and writes the same files.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
from collections.abc import Iterable, Sequence

__all__ = ["BOS", "EOS", "UNITS", "UNK", "NGram", "corpus_lines", "corpus_sequences", "detokenize", "load_ngram",
           "split_lines", "tokenize"]

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
UNITS = ("word", "char", "sound", "base64")
TEXT_EXTENSIONS = (".txt", ".md", ".text", ".jsonl", ".json", ".csv", ".tsv")

_FORMAT = "latticefsm-ngram"
_FORMAT_VERSION = 1
WIDTH_MIN, WIDTH_MAX = 0.05, 20.0
_NO_SPACE_BEFORE = set(".,!?;:)]}%'\"")
_NO_SPACE_AFTER = set("([{$\"")


# ---- tokens --------------------------------------------------------------------------------------------------------

_SOUND = None


def _sounds():
    """The phonetic tokenizer the ``sound`` unit reads and spells with: one for the process, the portable lexicon, no
    stress, and no memory of the words it has sounded out - so a word's sounds never depend on what was read before
    it (and the Rust crate, which keeps one the same way, hears every text alike)."""
    global _SOUND
    if _SOUND is None:
        from .phonetic import _phonetok
        pt = _phonetok()
        _SOUND = pt.PhoneticTokenizer(stress=False, lexicon=pt.Lexicon.default(package=False))
        _SOUND.transcriber.remember = False
    return _SOUND


def _word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def tokenize(line: str, unit: str = "word", lower: bool = True) -> list[str]:
    """One line as tokens: ``word`` - runs of letters, digits and ``_`` (an apostrophe between letters stays inside:
    *don't*), every other non-space character alone; ``char`` - every character, a space as ``_``; ``sound`` - the
    phonetic tokenizer's phonemes (no stress), ``#`` between words; ``base64`` - the letters of the line's UTF-8 in
    base64 (case kept: bytes are bytes)."""
    if unit == "base64":
        from .b64 import encode
        return list(encode(line.encode("utf-8")))
    if lower:
        line = line.lower()
    if unit == "char":
        return ["_" if c.isspace() else c for c in line.strip()]
    if unit == "sound":
        import unicodedata
        from .phonetic import inventory
        known = set(inventory(False))
        # numerals other than 0-9 (², ½, Ⅳ) are dropped: the tokenizer cannot say them
        line = "".join(" " if unicodedata.category(c) in ("Nd", "Nl", "No") and not "0" <= c <= "9" else c
                       for c in line)
        return [t for t in _sounds().tokens(line) if t in known]
    if unit != "word":
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
        elif _word_char(c):
            j = i + 1
            while j < n and (_word_char(line[j]) or (line[j] == "'" and j + 1 < n and line[j + 1].isalpha()
                                                     and _word_char(line[j - 1]))):
                j += 1
            out.append(line[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return out


def detokenize(tokens: Sequence[str], unit: str = "word") -> str:
    """Tokens written back as text: words with spaces (none before punctuation), characters run together, sounds
    spelled as English."""
    tokens = [t for t in tokens if t not in (BOS, EOS, UNK)]
    if unit == "base64":
        from .b64 import decode
        return decode("".join(tokens)).decode("utf-8", "replace")
    if unit == "char":
        return "".join(" " if t == "_" else t for t in tokens)
    if unit == "sound":
        return _sounds().decode(list(tokens))
    out = ""
    for t in tokens:
        if out and not (t in _NO_SPACE_BEFORE or out[-1] in _NO_SPACE_AFTER):
            out += " "
        out += t
    return out


# ---- corpora -------------------------------------------------------------------------------------------------------

def _json_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "sentence", "body"):
            if isinstance(value.get(key), str):
                return value[key]
        return " ".join(v for v in value.values() if isinstance(v, str))
    if isinstance(value, list):
        return "\n".join(_json_text(v) for v in value)
    return ""


def corpus_lines(paths: Iterable[str]) -> list[tuple[str, list[str]]]:
    """Every text file under ``paths`` (files, or directories walked in name order) as ``(path, lines)``: ``.txt`` /
    ``.md`` line by line; ``.jsonl`` one JSON value a line, its ``text`` (or ``content``, ...); ``.json`` likewise;
    empty lines dropped."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs.sort()
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.lower().endswith(TEXT_EXTENSIONS) and not n.startswith(".") and n != "README.md"]
        elif os.path.exists(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    out = []
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            text = f.read()
        lower = path.lower()
        if lower.endswith(".jsonl"):
            lines = []
            for raw in text.splitlines():
                raw = raw.strip()
                if raw:
                    try:
                        lines += _json_text(json.loads(raw)).splitlines()
                    except ValueError:
                        lines.append(raw)
        elif lower.endswith(".json"):
            try:
                lines = _json_text(json.loads(text)).splitlines()
            except ValueError:
                lines = text.splitlines()
        else:
            lines = text.splitlines()
        out.append((path, [ln.strip() for ln in lines if ln.strip()]))
    return out


def corpus_sequences(paths: Iterable[str], unit: str = "word", lower: bool = True) -> list[tuple[str, list[list[str]]]]:
    """Every file under ``paths`` as ``(path, token sequences)``: for ``base64``, every file of any kind, its bytes 76
    letters a sequence (:func:`latticefsm.b64.file_sequences`); for the other units, every text file line by line
    (:func:`corpus_lines`), each line tokenized, empty ones dropped."""
    if unit == "base64":
        from .b64 import file_sequences
        return [(path, [list(seq) for seq in seqs]) for path, seqs in file_sequences(paths)]
    out = []
    for path, lines in corpus_lines(paths):
        seqs = [tokenize(line, unit, lower) for line in lines]
        out.append((path, [q for q in seqs if q]))
    return out


def split_lines(lines: Sequence, every: int = 10) -> tuple[list, list]:
    """Every ``every``-th line held out (for measuring), the rest to train on."""
    if every <= 1:
        return list(lines), []
    return ([ln for i, ln in enumerate(lines) if i % every != every - 1],
            [ln for i, ln in enumerate(lines) if i % every == every - 1])


# ---- the model -----------------------------------------------------------------------------------------------------

class NGram:
    """An N-gram model of order ``order`` over ``unit`` tokens: sparse adaptive edges, interpolated Kneser-Ney."""

    def __init__(self, order: int = 3, unit: str = "word", lower: bool = True, stimulation: float = 1.0,
                 reward_widening: float = 0.2, punish_narrowing: float = 0.2) -> None:
        if not (1 <= order <= 12):
            raise ValueError(f"order is 1 to 12, got {order}")
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
        self.order = int(order)
        self.unit = unit
        self.lower = bool(lower)
        self.stimulation = float(stimulation)
        self.reward_widening = float(reward_widening)
        self.punish_narrowing = float(punish_narrowing)
        self.vocab: list[str] = [BOS, EOS, UNK]
        self.index: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.grams: dict[tuple[int, ...], dict[int, int]] = {}
        """The highest order's counts: context (``order - 1`` ids) -> next id -> how many times."""
        self.edges: dict[tuple[int, ...], dict[int, list[float]]] = {}
        """Edges with a record beyond their count: context -> id -> ``[width, rewarded, punished, last]``."""
        self.clock = 0
        """Tokens trained on (every ``</s>`` included)."""
        self.sentences = 0
        self.sources: list[list] = []
        self.history: list[dict] = []
        self._tables: list | None = None

    # ---- vocabulary and contexts -------------------------------------------------------------------------------

    def tokens(self, text: str) -> list[str]:
        return tokenize(text, self.unit, self.lower)

    def prompt_tokens(self, text: str) -> list[str]:
        """What a prompt is read as to be continued: its tokens - for ``base64`` only the letters whose bits are all
        known (:func:`latticefsm.b64.encode_prefix`), so the prompt is a true prefix of what continues it."""
        if self.unit == "base64":
            from .b64 import encode_prefix
            return list(encode_prefix(text.encode("utf-8")))
        return self.tokens(text)

    def _sequence(self, item: str | Sequence[str]) -> list[str]:
        return self.tokens(item) if isinstance(item, str) else list(item)

    def _id(self, token: str, grow: bool) -> int:
        i = self.index.get(token)
        if i is None:
            if not grow:
                return self.index[UNK]
            i = len(self.vocab)
            self.vocab.append(token)
            self.index[token] = i
        return i

    def context(self, tokens: Sequence[str]) -> tuple[int, ...]:
        """The last ``order - 1`` tokens as ids (unknown ones as ``<unk>``), padded with ``<s>``."""
        ids = [0] * (self.order - 1) + [self._id(t, False) for t in tokens]
        return tuple(ids[len(ids) - (self.order - 1):]) if self.order > 1 else ()

    # ---- training ----------------------------------------------------------------------------------------------

    def train_tokens(self, tokens: Sequence[str]) -> int:
        """One sequence: every ``(context, next)`` counted, ``</s>`` last.  Returns how many were counted."""
        ids = [0] * (self.order - 1) + [self._id(t, True) for t in tokens] + [1]
        k = self.order - 1
        for i in range(k, len(ids)):
            ctx = tuple(ids[i - k:i])
            row = self.grams.setdefault(ctx, {})
            row[ids[i]] = row.get(ids[i], 0) + 1
            self.clock += 1
            rec = self.edges.get(ctx, {}).get(ids[i])
            if rec is not None:
                rec[3] = self.clock
        self.sentences += 1
        self._tables = None
        return len(ids) - k

    def train_lines(self, lines: Iterable[str | Sequence[str]], source: str | None = None) -> int:
        """Every line (a text, or a sequence of tokens) one sequence; returns the tokens counted."""
        n = count = 0
        for line in lines:
            toks = self._sequence(line)
            if toks:
                n += self.train_tokens(toks)
                count += 1
        if source is not None:
            self.sources.append([source, count, n])
        return n

    def train_paths(self, paths: Iterable[str], held_out: int = 0) -> tuple[int, list[str]]:
        """Train on every file under ``paths`` (:func:`corpus_sequences`); with ``held_out = k`` every ``k``-th sequence
        is kept back and returned instead.  Returns (tokens counted, the held-out sequences)."""
        total, kept = 0, []
        for path, lines in corpus_sequences(paths, self.unit, self.lower):
            train, test = split_lines(lines, held_out) if held_out else (lines, [])
            total += self.train_lines(train, source=path)
            kept += test
        return total, kept

    # ---- Kneser-Ney ----------------------------------------------------------------------------------------------

    def _derive(self) -> list:
        """Per order k = 1..n: (table: context -> id -> count, totals, distinct, discount); the highest order from raw
        counts, each lower one from continuation counts of the order above."""
        if self._tables is not None:
            return self._tables
        tables: list = [None] * (self.order + 1)
        table = self.grams
        for k in range(self.order, 0, -1):
            totals = {h: sum(row.values()) for h, row in table.items()}
            n1 = n2 = 0
            for row in table.values():
                for c in row.values():
                    n1 += c == 1
                    n2 += c == 2
            d = n1 / (n1 + 2 * n2) if n1 > 0 and n2 > 0 else 0.75
            if not (0.0 < d < 1.0):
                d = 0.75
            tables[k] = (table, totals, {h: len(row) for h, row in table.items()}, d)
            if k > 1:
                lower: dict[tuple[int, ...], dict[int, int]] = {}
                for h, row in table.items():
                    sub = h[1:]
                    dst = lower.setdefault(sub, {})
                    for w in row:
                        dst[w] = dst.get(w, 0) + 1
                table = lower
        self._tables = tables
        return tables

    def _floor(self, w: int) -> float:
        return 0.0 if w == 0 else 1.0 / (len(self.vocab) - 1)

    def kn(self, ctx: tuple[int, ...], w: int) -> float:
        """``P_KN(w | ctx)``: interpolated Kneser-Ney, from the uniform floor up through every order."""
        tables = self._derive()
        p = self._floor(w)
        for k in range(1, self.order + 1):
            table, totals, distinct, d = tables[k]
            h = ctx[len(ctx) - (k - 1):] if k > 1 else ()
            row = table.get(h)
            if row:
                tot = totals[h]
                c = row.get(w, 0)
                p = max(c - d, 0.0) / tot + (d * distinct[h] / tot) * p
        return p

    def _boost(self, ctx: tuple[int, ...]) -> tuple[dict[int, list[float]] | None, float]:
        recs = self.edges.get(ctx)
        if not recs or self.stimulation == 0.0:
            return None, 1.0
        z = 1.0
        for w in sorted(recs):
            z += self.kn(ctx, w) * (recs[w][0] ** self.stimulation - 1.0)
        return recs, z

    def prob(self, ctx: tuple[int, ...], w: int) -> float:
        """``P(w | ctx)``: Kneser-Ney, every credited edge out of ``ctx`` weighted by its width ^ stimulation."""
        recs, z = self._boost(ctx)
        p = self.kn(ctx, w)
        if recs is None:
            return p
        rec = recs.get(w)
        return (p * rec[0] ** self.stimulation if rec else p) / z

    def distribution(self, ctx: tuple[int, ...]) -> list[float]:
        """``P(w | ctx)`` for every id (``<s>`` has 0): :meth:`kn` for all of them at once, order by order, the same
        operations in the same order, so every number is the one :meth:`prob` gives."""
        tables = self._derive()
        floor = 1.0 / (len(self.vocab) - 1)
        out = [floor] * len(self.vocab)
        out[0] = 0.0
        for k in range(1, self.order + 1):
            table, totals, distinct, d = tables[k]
            h = ctx[len(ctx) - (k - 1):] if k > 1 else ()
            row = table.get(h)
            if row:
                tot = totals[h]
                f = d * distinct[h] / tot
                out = [f * p for p in out]
                for w, c in row.items():
                    out[w] = max(c - d, 0.0) / tot + out[w]
        recs, z = self._boost(ctx)
        if recs is not None:
            for w in range(len(out)):
                rec = recs.get(w)
                out[w] = (out[w] * rec[0] ** self.stimulation if rec else out[w]) / z
        return out

    def predict(self, text: str | Sequence[str] = "", k: int = 10) -> list[tuple[str, float]]:
        """The ``k`` likeliest next tokens after ``text`` (a string, or tokens), likeliest first (ties: the token seen
        first)."""
        toks = self.prompt_tokens(text) if isinstance(text, str) else list(text)
        dist = self.distribution(self.context(toks))
        best = sorted(range(1, len(dist)), key=lambda w: (-dist[w], w))[:k]
        return [(self.vocab[w], dist[w]) for w in best]

    def generate(self, prompt: str = "", max_tokens: int = 30, temperature: float = 0.0,
                 rng: random.Random | None = None) -> list[str]:
        """Tokens continuing ``prompt`` until ``</s>`` or ``max_tokens``: the likeliest each time at temperature 0
        (deterministic: both ports write the same), else drawn from ``P ^ (1 / temperature)``."""
        return self.continue_tokens(self.prompt_tokens(prompt), max_tokens, temperature, rng)

    def continue_tokens(self, toks: Sequence[str], max_tokens: int = 30, temperature: float = 0.0,
                        rng: random.Random | None = None) -> list[str]:
        """Tokens continuing the tokens ``toks``, as :meth:`generate` does (never ``<s>`` or ``<unk>``)."""
        toks = list(toks)
        out: list[str] = []
        rng = rng or random.Random(1)
        for _ in range(max_tokens):
            dist = self.distribution(self.context(toks + out))
            dist[0] = dist[2] = 0.0
            if temperature <= 0:
                w = max(range(len(dist)), key=lambda i: (dist[i], -i))
            else:
                weights = [p ** (1.0 / temperature) if p > 0 else 0.0 for p in dist]
                r, acc, w = rng.random() * sum(weights), 0.0, 1
                for i, q in enumerate(weights):
                    acc += q
                    if r < acc:
                        w = i
                        break
            if w == 1:
                break
            out.append(self.vocab[w])
        return out

    def complete(self, prompt: str = "", max_tokens: int = 30, temperature: float = 0.0,
                 rng: random.Random | None = None) -> str:
        """``prompt`` continued, written back as text (for ``base64``, the prompt's letters and the new ones decoded
        together, so the prompt comes back with its continuation)."""
        toks = self.generate(prompt, max_tokens, temperature, rng)
        if self.unit == "base64":
            return detokenize(self.prompt_tokens(prompt) + toks, self.unit)
        return detokenize(toks, self.unit)

    def generate_bytes(self, prompt: bytes = b"", max_tokens: int = 4096, temperature: float = 0.0,
                       rng: random.Random | None = None) -> bytes:
        """A ``base64`` model continuing ``prompt`` (bytes: the start of a file, or nothing), decoded to bytes."""
        from .b64 import decode, encode_prefix
        if self.unit != "base64":
            raise ValueError("bytes come from a base64 model")
        start = list(encode_prefix(prompt))
        out = list(start)
        chunk = start
        while len(out) - len(start) < max_tokens:
            # a file was learned 76 letters a sequence: at a sequence's end the next one begins afresh
            more = self.continue_tokens(chunk, max_tokens - (len(out) - len(start)), temperature, rng)
            if not more:
                break
            out += more
            if "=" in more:
                break
            chunk = []
        return decode("".join(out))

    # ---- measuring ---------------------------------------------------------------------------------------------

    def evaluate(self, lines: Iterable[str], max_ranked: int = 2000) -> dict:
        """On held-out ``lines``: perplexity and cross-entropy (bits a token) over every token and ``</s>``; and, over
        the first ``max_ranked`` positions, how often the next token was the model's first guess (``accuracy``) or in
        its first five (``top5``).  Nothing moves."""
        logp, n, ranked, top1, top5 = 0.0, 0, 0, 0, 0
        for line in lines:
            toks = self._sequence(line)
            if not toks:
                continue
            ids = [self._id(t, False) for t in toks] + [1]
            hist: list[str] = []
            for j, w in enumerate(ids):
                ctx = self.context(hist)
                p = self.prob(ctx, w)
                logp += math.log(max(p, 1e-300))
                n += 1
                if ranked < max_ranked:
                    dist = self.distribution(ctx)
                    order = sorted(range(1, len(dist)), key=lambda i: (-dist[i], i))[:5]
                    top1 += order[0] == w
                    top5 += w in order
                    ranked += 1
                hist.append(toks[j] if j < len(toks) else EOS)
        if n == 0:
            return {"tokens": 0, "perplexity": None, "bits": None, "accuracy": None, "top5": None}
        return {"tokens": n, "perplexity": math.exp(-logp / n), "bits": -logp / n / math.log(2),
                "accuracy": top1 / ranked if ranked else None, "top5": top5 / ranked if ranked else None,
                "ranked": ranked}

    # ---- credit ------------------------------------------------------------------------------------------------

    def credit(self, ctx: tuple[int, ...], w: int, amount: float) -> None:
        """One edge credited: rewarded and widened (``amount > 0``) or punished and narrowed, as a lattice edge is."""
        rec = self.edges.setdefault(ctx, {}).setdefault(w, [1.0, 0.0, 0.0, float(self.clock)])
        if amount > 0:
            rec[0] = min(WIDTH_MAX, rec[0] * (1.0 + self.reward_widening * amount))
            rec[1] += amount
        elif amount < 0:
            rec[0] = max(WIDTH_MIN, rec[0] / (1.0 + self.punish_narrowing * -amount))
            rec[2] -= amount

    def credit_text(self, text: str | Sequence[str], amount: float) -> int:
        """Every edge a text takes, ``</s>`` included, credited ``amount``.  Returns how many."""
        toks = self.tokens(text) if isinstance(text, str) else list(text)
        hist: list[str] = []
        n = 0
        for t in toks + [EOS]:
            w = self._id(t, False) if t != EOS else 1
            if w != 2:
                self.credit(self.context(hist), w, amount)
                n += 1
            hist.append(t)
        return n

    # ---- describing --------------------------------------------------------------------------------------------

    def stats(self) -> dict:
        tables = self._derive()
        return {"order": self.order, "unit": self.unit, "lower": self.lower, "vocab": len(self.vocab) - 3,
                "contexts": len(self.grams), "ngrams": sum(len(r) for r in self.grams.values()),
                "tokens": self.clock, "sentences": self.sentences, "stimulation": self.stimulation,
                "credited_edges": sum(len(r) for r in self.edges.values()),
                "discounts": [round(tables[k][3], 6) for k in range(1, self.order + 1)],
                "sources": len(self.sources)}

    # ---- the file ----------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        grams = sorted([*ctx, w, c] for ctx, row in self.grams.items() for w, c in row.items())
        edges = sorted([*ctx, w, *rec] for ctx, recs in self.edges.items() for w, rec in recs.items())
        return {"format": _FORMAT, "version": _FORMAT_VERSION, "order": self.order, "unit": self.unit,
                "lower": self.lower, "stimulation": self.stimulation, "reward_widening": self.reward_widening,
                "punish_narrowing": self.punish_narrowing, "clock": self.clock, "sentences": self.sentences,
                "vocab": self.vocab, "grams": grams, "edges": edges, "sources": self.sources,
                "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> NGram:
        if d.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} file")
        m = cls(int(d["order"]), d.get("unit", "word"), bool(d.get("lower", True)), float(d.get("stimulation", 1.0)),
                float(d.get("reward_widening", 0.2)), float(d.get("punish_narrowing", 0.2)))
        m.vocab = list(d["vocab"])
        if m.vocab[:3] != [BOS, EOS, UNK]:
            raise ValueError("the vocabulary does not start <s> </s> <unk>")
        m.index = {t: i for i, t in enumerate(m.vocab)}
        k = m.order - 1
        for row in d.get("grams", []):
            m.grams.setdefault(tuple(int(x) for x in row[:k]), {})[int(row[k])] = int(row[k + 1])
        for row in d.get("edges", []):
            m.edges.setdefault(tuple(int(x) for x in row[:k]), {})[int(row[k])] = [float(x) for x in row[k + 1:k + 5]]
        m.clock = int(d.get("clock", 0))
        m.sentences = int(d.get("sentences", 0))
        m.sources = list(d.get("sources", []))
        m.history = list(d.get("history", []))
        return m

    def save(self, path: str) -> None:
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as f:
            f.write(text)


def load_ngram(path: str) -> NGram:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return NGram.from_dict(json.load(f))
