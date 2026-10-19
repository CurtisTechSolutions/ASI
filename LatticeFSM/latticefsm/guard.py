"""The guard: a negative network - only the failures, and *why* they were failures.

Every transition in it (the last ``order - 1`` tokens, then the next) exists because something went wrong there: a
text was blamed - a thumbs-down, a judge's low mark, a bad example, an explicit blame - and every token it took was
charged the blame, under the reason it came with.  A transition keeps

* ``blame`` - the summed severity of the failures that ran through it, ``fails`` - how many,
* ``clear`` - how much accepted text (a thumbs-up, a pass mark) ran through it since, and
* ``reasons`` - ``{reason: blame}``, at most :data:`MAX_EDGE_REASONS` of them (the smallest dropped).

The **evidence** against a transition is ``max(0, blame - clear)``: blame and clearing cancel, so a fragment good text
crosses as often as bad text carries no verdict - which keeps common words out of the judgement.

:meth:`Guard.judge` walks a text through it and reports how much of it is built from known failure:

* ``risk`` - the evidence summed over the text's transitions, divided by how many it has: repeating a failure blamed
  once scores about 1, sharing a third of its transitions with one about 0.33;
* ``coverage`` - the share of its transitions that are known failures; ``peak`` - the most evidence on one;
* ``reasons`` - the reasons behind that evidence, heaviest first; ``spans`` - the worst fragments, with the reason
  each carries;
* ``verdict`` - ``reject`` when at least ``min_coverage`` of it is known failure and ``risk`` reaches ``threshold``,
  ``suspect`` when some of it is, ``pass`` when none is - and ``why``, one sentence saying so.

A text the guard has never seen fail passes: it is a filter, not a censor.  The guard reads text with the N-gram
tokenizer (``word`` by default) whatever model it guards, so one guard sits in front of the chain, the N-gram and the
network alike.  The Rust crate's ``guard.rs`` judges the same, number for number, and reads the same files.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable

from .ngram import BOS, EOS, UNITS, tokenize

__all__ = ["Guard", "MAX_EDGE_REASONS", "MAX_JOURNAL", "UNSPECIFIED", "clean_reason", "load_guard"]

_FORMAT = "latticefsm-guard"
MAX_EDGE_REASONS = 8
MAX_JOURNAL = 200
UNSPECIFIED = "unspecified"
_JOURNAL_CHARS = 160


def clean_reason(reason: str | None) -> str:
    """A reason tag: whitespace collapsed, ASCII lower-cased, at most 60 characters, never empty."""
    text = " ".join(str(reason or "").split())
    text = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)
    return text[:60] or UNSPECIFIED


class Edge:
    __slots__ = ("blame", "fails", "clear", "reasons")

    def __init__(self) -> None:
        self.blame = 0.0
        self.fails = 0
        self.clear = 0.0
        self.reasons: dict[str, float] = {}

    @property
    def evidence(self) -> float:
        return max(0.0, self.blame - self.clear)

    def top_reason(self) -> str:
        if not self.reasons:
            return UNSPECIFIED
        return min(self.reasons, key=lambda r: (-self.reasons[r], r))


class Guard:
    """The failures a model made, transition by transition, and a judge of new text by them."""

    def __init__(self, unit: str = "word", order: int = 3, lower: bool = True, threshold: float = 1.0,
                 min_coverage: float = 0.5) -> None:
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
        if not 1 <= order <= 8:
            raise ValueError(f"order is 1 to 8, got {order}")
        self.unit = unit
        self.order = int(order)
        self.lower = bool(lower)
        self.threshold = float(threshold)
        self.min_coverage = float(min_coverage)
        self.edges: dict[tuple[str, ...], Edge] = {}
        """``(*context, next)`` -> its record."""
        self.totals: dict[str, float] = {}
        self.journal: list[dict] = []
        self.blamed = 0
        self.cleared = 0
        self.judgements = 0
        self.rejected = 0

    # ---- reading a text ------------------------------------------------------------------------------------------

    def transitions(self, text: str) -> tuple[list[str], list[tuple[str, ...]]]:
        """A text's tokens and its transitions, ``(*the last order - 1 tokens, next)``, ``</s>`` last."""
        toks = tokenize(text, self.unit, self.lower)
        k = self.order - 1
        padded = [BOS] * k + toks + [EOS]
        return toks, [tuple(padded[i - k:i + 1]) for i in range(k, len(padded))]

    # ---- learning what failed ------------------------------------------------------------------------------------

    def blame(self, text: str, reason: str | None = None, severity: float = 1.0) -> int:
        """Charge every transition of a failed text ``severity`` of blame under ``reason``.  Returns how many."""
        reason = clean_reason(reason)
        severity = max(0.0, float(severity))
        _, trans = self.transitions(text)
        if not trans or severity == 0.0:
            return 0
        for key in trans:
            e = self.edges.get(key)
            if e is None:
                e = self.edges[key] = Edge()
            e.blame += severity
            e.fails += 1
            e.reasons[reason] = e.reasons.get(reason, 0.0) + severity
            if len(e.reasons) > MAX_EDGE_REASONS:
                del e.reasons[min(e.reasons, key=lambda r: (e.reasons[r], r))]
        self.totals[reason] = self.totals.get(reason, 0.0) + severity
        self.blamed += 1
        self.journal.append({"text": text[:_JOURNAL_CHARS], "reason": reason, "severity": severity,
                             "transitions": len(trans)})
        del self.journal[:max(0, len(self.journal) - MAX_JOURNAL)]
        return len(trans)

    def clear(self, text: str, amount: float = 1.0) -> int:
        """An accepted text: every transition of it the guard knows is cleared by ``amount``.  Returns how many."""
        amount = max(0.0, float(amount))
        _, trans = self.transitions(text)
        n = 0
        for key in trans:
            e = self.edges.get(key)
            if e is not None:
                e.clear += amount
                n += 1
        if n:
            self.cleared += 1
        return n

    def forget(self, reason: str) -> int:
        """Take a reason's blame back out of every transition (dropping those left with none).  Returns how many."""
        reason = clean_reason(reason)
        n = 0
        for key in sorted(self.edges):
            e = self.edges[key]
            b = e.reasons.pop(reason, None)
            if b is None:
                continue
            e.blame = max(0.0, e.blame - b)
            n += 1
            if not e.reasons or e.blame <= 0.0:
                del self.edges[key]
        self.totals.pop(reason, None)
        return n

    def reset(self) -> None:
        self.edges.clear()
        self.totals.clear()
        self.journal.clear()
        self.blamed = self.cleared = self.judgements = self.rejected = 0

    # ---- judging -------------------------------------------------------------------------------------------------

    def judge(self, text: str, threshold: float | None = None, min_coverage: float | None = None,
              spans: int = 5) -> dict:
        """Why this text looks like a failure (see the module docstring)."""
        limit = self.threshold if threshold is None else float(threshold)
        floor = self.min_coverage if min_coverage is None else float(min_coverage)
        toks, trans = self.transitions(text)
        k = self.order - 1
        rows = []
        for i, key in enumerate(trans):
            e = self.edges.get(key)
            rows.append((i, e, e.evidence if e is not None else 0.0))
        total = len(rows)
        blamed = [r for r in rows if r[2] > 0]
        known = sum(1 for r in rows if r[1] is not None)
        evidence = 0.0
        for r in blamed:
            evidence += r[2]
        coverage = len(blamed) / total if total else 0.0
        risk = evidence / max(1, total)
        by_reason: dict[str, float] = {}
        for _, e, ev in blamed:
            share = ev / e.blame if e.blame > 0 else 0.0
            for name in sorted(e.reasons):
                by_reason[name] = by_reason.get(name, 0.0) + e.reasons[name] * share
        reasons = sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0]))
        weight = 0.0
        for _, b in reasons:
            weight += b
        weight = weight or 1.0
        worst = sorted(blamed, key=lambda r: (-r[2], r[0]))[:max(0, spans)]
        peak = worst[0][2] if worst else 0.0
        if blamed and coverage >= floor and risk >= limit:
            verdict = "reject"
        elif blamed:
            verdict = "suspect"
        else:
            verdict = "pass"
        self.judgements += 1
        if verdict == "reject":
            self.rejected += 1
        top = reasons[0][0] if reasons else None
        if verdict == "reject":
            why = (f"rejected: {coverage * 100:.0f}% of its transitions are known failures, {risk:.2f} blame each on "
                   f"average (the limit {limit:.2f}), mostly '{top}'")
        elif verdict == "suspect":
            why = (f"suspect: {len(blamed)} of its {total} transitions are known failures ({coverage * 100:.0f}%), "
                   f"{risk:.2f} blame each - under the limit")
        else:
            why = "pass: none of it is a known failure"
        out_spans = []
        for i, e, ev in worst:
            first = max(0, i - k)
            last = min(len(toks), i + 1)
            out_spans.append({"start": first, "end": last, "tokens": toks[first:last], "evidence": ev,
                              "reason": e.top_reason(), "next": trans[i][-1]})
        return {"text": text, "tokens": len(toks), "transitions": total, "known": known, "blamed": len(blamed),
                "evidence": evidence, "risk": risk, "coverage": coverage, "peak": peak,
                "reasons": [{"reason": n, "blame": b, "share": b / weight} for n, b in reasons],
                "spans": out_spans, "verdict": verdict, "why": why}

    def filter(self, texts: Iterable[str]) -> list[dict]:
        """Every text judged, in order."""
        return [self.judge(t) for t in texts]

    # ---- describing and the file ---------------------------------------------------------------------------------

    def stats(self) -> dict:
        evidence = 0.0
        for key in sorted(self.edges):
            evidence += self.edges[key].evidence
        return {"unit": self.unit, "order": self.order, "threshold": self.threshold, "min_coverage": self.min_coverage,
                "transitions": len(self.edges), "evidence": evidence, "blamed": self.blamed, "cleared": self.cleared,
                "judgements": self.judgements, "rejected": self.rejected,
                "reasons": [{"reason": r, "blame": b} for r, b in sorted(self.totals.items(), key=lambda kv: (-kv[1], kv[0]))],
                "journal": self.journal[-20:][::-1]}

    def to_dict(self) -> dict:
        rows = []
        for key in sorted(self.edges):
            e = self.edges[key]
            rows.append([list(key), e.blame, e.fails, e.clear, [[r, e.reasons[r]] for r in sorted(e.reasons)]])
        return {"format": _FORMAT, "version": 1, "unit": self.unit, "order": self.order, "lower": self.lower,
                "threshold": self.threshold, "min_coverage": self.min_coverage, "edges": rows,
                "totals": [[r, self.totals[r]] for r in sorted(self.totals)], "journal": self.journal,
                "blamed": self.blamed, "cleared": self.cleared, "judgements": self.judgements,
                "rejected": self.rejected}

    @classmethod
    def from_dict(cls, d: dict) -> Guard:
        if d.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} file")
        g = cls(d.get("unit", "word"), int(d.get("order", 3)), bool(d.get("lower", True)),
                float(d.get("threshold", 1.0)), float(d.get("min_coverage", 0.5)))
        for key, blame, fails, clear, reasons in d.get("edges", []):
            e = Edge()
            e.blame, e.fails, e.clear = float(blame), int(fails), float(clear)
            e.reasons = {str(r): float(b) for r, b in reasons}
            g.edges[tuple(str(t) for t in key)] = e
        g.totals = {str(r): float(b) for r, b in d.get("totals", [])}
        g.journal = list(d.get("journal", []))
        g.blamed = int(d.get("blamed", 0))
        g.cleared = int(d.get("cleared", 0))
        g.judgements = int(d.get("judgements", 0))
        g.rejected = int(d.get("rejected", 0))
        return g

    def save(self, path: str) -> None:
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as f:
            f.write(text)


def load_guard(path: str) -> Guard:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return Guard.from_dict(json.load(f))
