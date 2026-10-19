"""Auto-temperature: when a reply is no answer - nothing, no words, or a loop - draw it again a little hotter, rung by
rung up a ladder, until one is an answer or the ladder ends.  The same rules as ``rust/src/autotemp.rs``.

Temperature is a re-roll, not a repair: a hotter draw escapes a loop or an early ``</s>`` only by chance, and the
hotter it is the less it follows what the model learned (on ``data/corpus``, 0.3 still reads, 1.0 is word salad).
So the ladder starts where it was asked, climbs in small steps, stops early (``MAX_TEMPERATURE``), and draws a few
times at each rung before climbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = ["MAX_TEMPERATURE", "STEP", "TRIES", "MIN_LOOP", "why_draw", "why_no_answer", "repeats_context",
           "loop_period", "ladder", "attempts", "badness", "Tried", "Climbed", "climb", "climb_note"]

MAX_TEMPERATURE = 0.8
"""The hottest rung, by default: above it the replies stop following the model."""
STEP = 0.2
"""How much hotter each rung is than the one before."""
TRIES = 3
"""Draws at each rung above 0 before climbing (at 0 a draw is the same every time, so one)."""
MIN_LOOP = 12
"""The shortest stretch of text, in characters, that said again counts as a loop."""


def why_draw(text: str, temperature: float, tokens: Sequence[str], start: int, context: int) -> str | None:
    """Why a draw is no answer, or None when it is one: ``text`` the reply (without its prompt), ``tokens`` the
    prompt's tokens and the reply's, the reply's from ``start``, ``context`` how many tokens the model predicts from.
    A greedy draw (temperature 0) loops exactly when a context comes back (:func:`repeats_context`); a sampled one is
    read for a loop (:func:`why_no_answer`)."""
    if temperature <= 0:
        why = _why_empty(text)
        if why:
            return why
        return "a loop" if repeats_context(tokens, start, context) else None
    return why_no_answer(text)


def _why_empty(text: str) -> str | None:
    t = text.strip()
    if not t:
        return "nothing"
    if not any(c.isalnum() for c in t):
        return "no words"
    return None


def repeats_context(tokens: Sequence[str], start: int, k: int) -> bool:
    """Whether a greedy reply has begun going round: some token of it, from ``start``, was drawn after the same ``k``
    tokens as an earlier one.  A model at temperature 0 draws the same after the same context, so from there it writes
    what it wrote before, for ever - exact, where reading the text for a loop can only guess.  (Contexts still reaching
    back to ``<s>`` are each their own.)"""
    seen: set[tuple[str, ...]] = set()
    for g in range(max(start, k), len(tokens)):
        ctx = tuple(tokens[g - k:g])
        if ctx in seen:
            return True
        seen.add(ctx)
    return False


def why_no_answer(text: str) -> str | None:
    """Why ``text`` (a reply, without its prompt) is no answer, or None when it is one: ``"nothing"`` (empty), ``"no
    words"`` (no letter or digit, say "."), ``"a loop"`` (:func:`loop_period`: it says something again)."""
    why = _why_empty(text)
    if why:
        return why
    return "a loop" if loop_period(text.strip()) is not None else None


def loop_period(text: str) -> int | None:
    """Where ``text`` says something again: the shortest ``p`` such that a stretch of at least ``MIN_LOOP`` characters
    comes back ``p`` characters later, after no more new text than its own length (``r >= p / 2``).  A reply cut off
    by its token limit has seldom gone round twice in full - "is a directed cycle is recorded in a brain is a directed
    cycle is recorded in a" goes round its 43 characters not quite twice - so most of a period, said again, is enough;
    "the cat sat on the mat" says only "the " again."""
    s, n = text, len(text)
    for p in range(1, n):
        need = max(MIN_LOOP, (p + 1) // 2)
        if need + p > n:
            break
        run = 0
        for i in range(n - p):
            if s[i] == s[i + p]:
                run += 1
                if run >= need:
                    return p
            else:
                run = 0
    return None


def ladder(start: float, top: float) -> list[float]:
    """The temperatures to try, coolest first: ``start``, then ``STEP`` hotter each rung up to ``top`` (``start``
    alone when it is already as hot)."""
    start = max(0.0, start)
    out, k = [start], 1
    while True:
        t = round((start + STEP * k) * 1e9) / 1e9
        if t > top + 1e-9:
            return out
        out.append(t)
        k += 1


def attempts(start: float, top: float, tries: int) -> list[tuple[float, int]]:
    """Every draw to make, in order: (temperature, the seed's offset from the sample's own seed).  The first is the
    draw a plain request makes, so when it is an answer nothing changes."""
    out: list[tuple[float, int]] = []
    for t in ladder(start, top):
        for _ in range(1 if t <= 0 else max(1, tries)):
            out.append((t, 7919 * len(out)))
    return out


def badness(why: str | None) -> int:
    """How bad a reason is, for keeping the least bad when nothing was an answer: a loop still says something."""
    return {"a loop": 1, "rejected by the guard": 2, "no words": 3}.get(why or "", 4)


@dataclass
class Tried:
    """One draw on the way up the ladder: its temperature and seed, its text, and why it was no answer."""
    temperature: float
    seed: int
    text: str
    why: str | None


@dataclass
class Climbed:
    """A reply drawn up the ladder: the draw kept (``pick``, its temperature and seed), whether it is an answer
    (``found``; if not, ``why``), and every draw made."""
    pick: Any
    temperature: float
    seed: int
    found: bool
    why: str | None
    tried: list[Tried] = field(default_factory=list)


def climb(prompt: Sequence[str], context: int, start: float, top: float, tries: int, seed: int,
          draw: Callable[[float, int], tuple[list[str], str, Any]],
          check: Callable[[str, Any], str | None] | None = None) -> Climbed:
    """Draw a reply rung by rung (``attempts(start, top, tries)``, every seed ``seed`` plus its offset) until one is
    an answer: ``draw(temperature, seed)`` gives a reply's tokens, its text and anything else to keep;
    :func:`why_draw` reads it, with the prompt's tokens ``prompt`` and the model's ``context``; ``check`` may find
    more wrong with an answer (the guard, say).  When none is an answer the least bad is kept (:func:`badness`; the
    coolest of equals)."""
    kept: list[tuple[Any, Tried]] = []
    for t, offset in attempts(start, top, tries):
        s = seed + offset
        toks, text, extra = draw(t, s)
        why = why_draw(text, t, list(prompt) + list(toks), len(prompt), context)
        if why is None and check is not None:
            why = check(text, extra)
        kept.append((extra, Tried(t, s, text, why)))
        if why is None:
            break
    found = kept[-1][1].why is None
    pick = len(kept) - 1
    if not found:
        pick = 0
        for k, (_, x) in enumerate(kept):
            if badness(x.why) < badness(kept[pick][1].why):
                pick = k
    extra, x = kept[pick]
    return Climbed(extra, x.temperature, x.seed, found, x.why, [x for _, x in kept])


def climb_note(found: bool, temperature: float, why: str | None, tried: Sequence[Tried]) -> str:
    """How a climb went, in a line: "found at temperature 0.4: 0 nothing, 0.2 no words, 0.4 an answer" or "no answer
    up to 0.8 (kept the one at 0.2: a loop): ..."."""
    steps = ", ".join(f"{x.temperature:g} {x.why or 'an answer'}" for x in tried)
    if found:
        return f"found at temperature {temperature:g}: {steps}"
    top = max((x.temperature for x in tried), default=0.0)
    return f"no answer up to {top:g} (kept the one at {temperature:g}: {why or ''}): {steps}"
