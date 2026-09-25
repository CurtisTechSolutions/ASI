"""The attention band: where inside an n-gram a judgement lands.

A reader's eye does not take a line in evenly.  It fixes on one point and sees
that point sharply, and the letters to the left and the right of it - the lines
above and below it too - blur away with the distance.  A gram is this model's
fixation: ``n`` units read at once.  The **band** is how sharply each of those
``n`` positions is seen - 1 at the centre, falling off in a straight line to
``1 - blur`` at the first and the last unit::

    blur 0.5, n = 3   ->  0.5  1.0  0.5
    blur 0.5, n = 5   ->  0.5  0.75 1.0  0.75 0.5
    blur 0,   any n   ->  1.0 at every position: no focus, the gram seen evenly

A text is one line, so the band has a left and a right and nothing above or
below.  It is a setting of the model (:class:`AttentionBand`), saved in its
file, and **off** until it is switched on.

**What it changes: where a correction's blame and credit land.**  A judgement
that covers a whole text - a thumbs up, a thumbs down, a training pass - marks
every unit alike; every gram then sees nothing but marked units and no band
can tell one gram from another.  The band is inert there *by construction*,
not by a switch.  A correction is different: the diff (:mod:`radixnet.diff`)
marks only the units the teacher changed, and something has to decide which
steps of a path answer for them.

* **Off** - the rule every model had before this: each marked unit is charged,
  in full, to the step that *wrote* it, the step whose gram ends on it.  That
  is a band too, in the limit: all of the attention on the newest unit of the
  window, the one the eye has only just reached.
* **On** - each marked unit hands out exactly **one** charge, shared among the
  grams that *see* it, each in proportion to how sharply it sees it (its band
  weight at the unit's position).  The gram that has the unit at its centre
  takes the largest share; the grams that only glimpse it at an edge take
  less.  A step is charged what its grams collected, and never more than one
  full charge: a step is one decision, however much of it was wrong.

Sharing, rather than charging every gram that sees a unit in full, keeps a
correction as loud as it was: a unit that one step answered for is answered
for by several now, and their shares add up to the same one.  Where fewer
grams see a unit - the first and last units of a text, and every unit of a
grouping encoding, where each unit sits in exactly one gram - the shares are
larger, and where a single gram sees a unit it takes the whole charge, exactly
as the step that wrote it did.  The step into END is not a gram and sees
nothing: it answers for the position after the last unit (a sentence that
stopped too early) in full, band or no band.

The *verdict* a blamed step earns (the judged-path counters of
``CountRewardGraph.mark_steps``) is a count, not an amount, and cannot be
shared.  It goes to the **focus**: the gram or grams that see a marked unit
most sharply.  Off, that is the writer; on, the gram with the unit at its
centre.

:func:`spread` is the whole rule, over the text's grams and nothing else, so
the model (``GraphModel._charged_steps``), the preview the API and the CLI
show, and the Go and Rust ports compute the same numbers in the same order.
``../SPEC-AttentionBand.md`` is the specification.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .encoding import Encoding

__all__ = [
    "DEFAULT_BLUR", "AttentionBand", "Spread", "band_weights", "judged_units", "preview", "spread", "writer_marks",
]

DEFAULT_BLUR = 0.5
"""What switching the band on means when no blur is given: the ends of a gram seen half as sharply as its centre."""


def band_weights(n: int, blur: float) -> list[float]:
    """How sharply each of a gram's ``n`` positions is seen: 1 at the centre, ``1 - blur`` at both ends, linear between.

    ``n = 1`` is nothing but a centre.  ``n = 2`` has no centre at all - both
    positions are ends - so its band is flat whatever the blur, as every band
    is at ``blur = 0``.  Each weight is ``1 - blur * (d / c)``, with ``c =
    (n - 1) / 2`` and ``d`` the position's distance from it: every operation is
    one correctly rounded IEEE step, which is what lets Go and Rust compute the
    same bits.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return [1.0]
    centre = (n - 1) / 2
    return [1.0 - blur * (abs(j - centre) / centre) for j in range(n)]


@dataclass(frozen=True)
class AttentionBand:
    """The band a model credits corrections through: ``blur`` in ``[0, 1]``, or ``None`` for off (the writer rule).

    A value of the model, like its encoding, but not fixed for its life: it
    changes nothing the graph holds, only where the next correction lands, so
    it can be switched on, off or re-blurred at any time.
    """

    blur: float | None = None

    def __post_init__(self) -> None:
        if self.blur is None:
            return
        try:
            value = float(self.blur)
        except (TypeError, ValueError):
            raise ValueError(f"blur must be a number in [0, 1], got {self.blur!r}") from None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"blur must lie in [0, 1], got {self.blur!r}")
        object.__setattr__(self, "blur", value)

    @property
    def on(self) -> bool:
        """Is the band on?  Off, each marked unit is charged to the step that wrote it."""
        return self.blur is not None

    def weights(self, n: int) -> list[float] | None:
        """The band over one gram of ``n`` units, or ``None`` when it is off."""
        return band_weights(n, self.blur) if self.blur is not None else None

    def to_dict(self) -> dict | None:
        """The ``attention`` block of a graph document: ``{"blur": b}``, or ``None`` - nothing is written - when off."""
        return {"blur": self.blur} if self.blur is not None else None

    @classmethod
    def from_dict(cls, d: dict | None) -> "AttentionBand":
        """Read an ``attention`` block; a document without one was written with the band off."""
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise ValueError(f"an attention block is an object, got {type(d).__name__}")
        blur = d.get("blur")
        return cls(None if blur is None else blur)

    def __str__(self) -> str:
        return "off" if self.blur is None else f"blur {self.blur:g}"

    def describe(self, n: int | None = None) -> str:
        """The human form: ``'off: the writer takes each changed unit'`` or the band over a gram of ``n``."""
        if self.blur is None:
            return "off: each changed unit is charged to the step that wrote it"
        text = f"on, blur {self.blur:g}: the centre of each gram is charged most"
        if n is not None:
            text += " (" + " ".join(f"{w:g}" for w in band_weights(n, self.blur)) + ")"
        return text


def judged_units(spans: Iterable[tuple[int, int]], length: int) -> list[int]:
    """The units a correction marks, ascending, within ``[0, length]``.

    Every unit of every half-open span, and for an empty span - an insertion
    point - the unit it stands in front of: the step that walked straight past
    the position is the one at fault.  ``length`` itself is the position after
    the last unit, which only the step into END answers for.
    """
    marked: set[int] = set()
    for lo, hi in spans:
        lo, hi = int(lo), int(hi)
        top = hi if hi > lo else lo + 1
        for unit in range(max(0, lo), min(top, length + 1)):
            marked.add(unit)
    return sorted(marked)


@dataclass
class Spread:
    """What :func:`spread` hands each gram of a text."""

    shares: list[float]
    """Per gram, the charge it collected: the sum of its shares of the marked units, before any cap."""
    focus: list[bool]
    """Per gram: does it see some marked unit most sharply of all the grams that see that unit?"""
    end: bool
    """Is the position after the last unit marked?  The step into END answers for it, in full."""


def spread(
    n: int, stride: int, grams: int, length: int, spans: Iterable[tuple[int, int]], weights: Sequence[float]
) -> Spread:
    """Share every marked unit out over the grams that see it, in proportion to the band.

    ``grams`` is how many grams the text holds; gram ``g`` covers the units
    ``[g * stride, g * stride + n)`` and sees unit ``u`` at position ``u - g *
    stride``, with the weight ``weights`` gives that position.  Each marked
    unit hands out exactly one charge: ``weight / total`` to each gram that
    sees it, ``total`` being the sum of those grams' weights for it - or an
    even split when every one of them sees it at a weight of 0 (the limit of a
    blur of 1, where only the ends of the grams reach the unit).  A unit no
    gram covers (the tail a grouping encoding drops) is charged to no one.

    The order is part of the definition: units ascending, the grams that see
    one ascending, every sum left to right - so every implementation adds the
    same floats in the same order and gets the same bits.
    """
    shares = [0.0] * grams
    focus = [False] * grams
    end = False
    for unit in judged_units(spans, length):
        if unit >= length:
            end = True  # the position after the last unit: judged_units stops there
            continue
        lo = 0 if unit < n else (unit - n) // stride + 1
        hi = min(grams - 1, unit // stride)
        if lo > hi:
            continue
        total = 0.0
        best = 0.0
        for g in range(lo, hi + 1):
            weight = weights[unit - g * stride]
            total += weight
            if weight > best:
                best = weight
        viewers = hi - lo + 1
        for g in range(lo, hi + 1):
            weight = weights[unit - g * stride]
            shares[g] += weight / total if total > 0.0 else 1.0 / viewers
            if weight == best:
                focus[g] = True
    return Spread(shares, focus, end)


def writer_marks(
    n: int, stride: int, grams: int, length: int, spans: Iterable[tuple[int, int]]
) -> tuple[list[bool], bool]:
    """The rule with the band off, gram by gram: ``(does gram g write a marked unit, is the end marked)``.

    The first gram writes all of its units, every later one the ``stride``
    units past the overlap: what ``GraphModel._steps_over`` charges a path
    whose every node is one gram.
    """
    marked = judged_units(spans, length)
    overlap = n - stride
    out = [False] * grams
    for g in range(grams):
        lo = 0 if g == 0 else g * stride + overlap
        hi = g * stride + n
        out[g] = any(lo <= unit < hi for unit in marked)
    return out, length in marked


def preview(encoding: Encoding, weights: Sequence[float], text: str, spans: Sequence[tuple[int, int]]) -> dict:
    """One side of a correction, gram by gram: what the writer rule charges, and what the band would.

    No graph is involved: each gram is shown as if it were a step of its own,
    so a compressed node - which is charged what its grams add up to, capped at
    one - reads as the sum of the rows it covers.
    """
    grams = encoding.encode(text)
    length = encoding.length(text)
    writes, _ = writer_marks(encoding.n, encoding.stride, len(grams), length, spans)
    shared = spread(encoding.n, encoding.stride, len(grams), length, spans, weights)
    return {
        "text": text,
        "units": length,
        "grams": grams,
        "spans": [[int(lo), int(hi)] for lo, hi in spans],
        "writer": writes,
        "charges": [min(1.0, share) for share in shared.shares],
        "focus": shared.focus,
        "end": shared.end,
    }
