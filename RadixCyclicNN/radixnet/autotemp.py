"""Auto-temperature: a search that finds no answer walks again, hotter (``../SPEC-SearchAndTraining.md`` §10).

The temperature is the one dial that trades how likely a walk is for how far
it strays: a sampled step weighs every option ``p ** (1 / temperature)``, so a
cold walk keeps to the likeliest paths - and, in a cyclic graph, goes round
them - while a hot one reaches paths the likely ones crowd out.  When a search
comes back with no answer, this module asks again up a *ladder* of
temperatures, a few walks at every rung, and stops at the first answer:

* **The search as asked runs first**, exactly as it would have - any mode, any
  temperature - and when it answers, nothing else happens.
* **Then the ladder** (:func:`temperature_ladder`): the asked temperature, then
  doubling - from :data:`AUTO_FLOOR` when the asked one was greedy - up to a
  ceiling, the last rung clamped to it.  Each rung is :data:`AUTO_DRAWS`
  sampled walks (one at temperature 0, where every walk is the same): an
  empty or vetoed walk is mended by another draw as often as by a hotter one,
  and a hotter walk is a less likely one, so the temperature rises only when
  the draws at this one ran out.
* **An answer** is a walk that wrote something and, when the caller asked for
  the end of a text, reached END (:func:`answered`); a caller with a guard adds
  its veto (:func:`climb` takes any pick).  A search that offers several
  continuations - the beam's top K - has found one when *any* of them answers:
  the first that does becomes the prediction (:func:`first_answer`), as the
  guard promotes the first one it lets through, and nothing is walked.

A deterministic search - Dijkstra, the beam, the resonant model's k-best -
reads no temperature at all, so for it the ladder is where sampling starts:
the search found nothing at the cheapest paths, and walks at rising
temperature look among the others.  Where the graph has no way on at all, no
temperature finds one, and the record says so.

Off by default (a ceiling of 0): no extra search, no draw, the answer exactly
the one the search gave (D-078).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from .model import GraphModel
    from .search import PathResult

__all__ = [
    "AUTO_DRAWS", "AUTO_FLOOR", "AutoTemperature", "answered", "check_auto_temperature", "climb", "first_answer",
    "generate", "predict", "temperature_ladder",
]

AUTO_DRAWS = 3
"""Sampled walks drawn at every rung of the ladder before the temperature rises.

Measured on a count model of 1,500 lines of prose: an empty continuation comes
back from 4% of the walks at temperature 0.5, 6% at 1 and 13% at 4 - a hotter
walk is no likelier to say something, so a second and a third draw mend more
than the next rung does.  Three draws at a rung that answers 94% of the time
leave one search in five thousand with nothing to say.
"""

AUTO_FLOOR = 0.5
"""The first temperature above greedy: a ladder that starts at 0 climbs to this, then doubles."""

R = TypeVar("R")


def check_auto_temperature(ceiling: float) -> None:
    """``ValueError`` for a ceiling out of range: it must be a finite number ``>= 0`` (``0`` is off)."""
    if not (isinstance(ceiling, (int, float)) and math.isfinite(ceiling) and ceiling >= 0):
        raise ValueError(f"auto_temperature must be a finite number >= 0 (0 is off), got {ceiling!r}")


def temperature_ladder(start: float, ceiling: float) -> list[float]:
    """The temperatures a search climbs through: ``start``, then doubling up to ``ceiling``.

    The rung after ``t`` is ``min(max(2 * t, AUTO_FLOOR), ceiling)``, so a
    greedy start climbs to :data:`AUTO_FLOOR` first and the last rung is the
    ceiling itself: ``(1, 2)`` is ``[1, 2]``, ``(1, 3)`` is ``[1, 2, 3]``,
    ``(0, 2)`` is ``[0, 0.5, 1, 2]``.  A start at or above the ceiling is the
    one rung - the ladder never cools a walk - and a ceiling of 0 is no ladder.
    Temperature divides a log-probability, so the rungs are spaced by a factor
    rather than a step: each doubling is the same change in how flat a step is.
    """
    if not ceiling > 0:
        return []
    rungs = [float(start)]
    t = float(start)
    while t < ceiling:
        t = min(max(2.0 * t, AUTO_FLOOR), float(ceiling))
        rungs.append(t)
    return rungs


def answered(result: "PathResult", to_end: bool = False) -> bool:
    """Is this an answer: did the search write something - more than whitespace - and, when ``to_end`` asked for
    the end of a text, did it reach END?"""
    return bool(result.text.strip()) and (bool(result.reached_end) or not to_end)


def first_answer(result: R, to_end: bool = False) -> R | None:
    """The search's answer: ``result`` itself when it is one, else - for a search that offered several
    continuations (a :class:`~radixnet.beam.Prediction`'s ``top``) - the first of them that is, made the
    prediction (its ``text``, path and cost; ``top`` and ``bottom`` stay as the search ranked them), else ``None``.

    The beam can rank a path cut at the length cap above the ones that reached END - a walk round a cycle is
    cheap - so a request for the end of a text may find its answer second or third: that is still the search's
    own answer, and better than any walk at a higher temperature.
    """
    if answered(result, to_end):
        return result
    for candidate in getattr(result, "top", None) or ():
        if answered(candidate, to_end):
            from .search import PathResult

            carried = {f.name: getattr(candidate, f.name) for f in dataclasses.fields(PathResult) if f.name != "expanded"}
            return dataclasses.replace(result, **carried)
    return None


@dataclass
class AutoTemperature:
    """What a climb did: the ceiling it was given, whether the search as asked had to be retried (``climbed``),
    whether an answer came back (``found``), the temperature of the walk that gave it (``None`` when the answer is
    a deterministic search's, or there is none) and how many searches ran, the one as asked included (``tries``).

    With nothing found, what comes back is the search as asked, and ``temperature`` is ``None``: no walk answered.
    """

    ceiling: float
    climbed: bool = False
    found: bool = False
    temperature: float | None = None
    tries: int = 0

    def to_dict(self) -> dict:
        return {
            "ceiling": self.ceiling, "climbed": self.climbed, "found": self.found, "temperature": self.temperature,
            "tries": self.tries,
        }


def climb(
    search: Callable[[float | None], R],
    pick: Callable[[R], R | None],
    *,
    temperature: float,
    ceiling: float,
    sampled: bool,
) -> tuple[R, AutoTemperature]:
    """Run the search as asked and, while it gives no answer, sampled walks up the ladder.

    ``search(None)`` runs the caller's search exactly as it was asked for, and
    ``search(t)`` the same request as a sampled walk at temperature ``t``;
    ``pick`` makes an answer of what came back, or ``None`` when there is none
    in it: the model's own test (:func:`first_answer`), and whatever the caller
    adds - a guard's veto.  ``temperature`` is the one the request carries and
    ``sampled`` whether the search as asked was itself a sampled walk (then it
    is the temperature of an answer it gives).  The ladder is
    :func:`temperature_ladder` from ``temperature`` to ``ceiling``,
    :data:`AUTO_DRAWS` sampled walks at every rung (one at temperature 0), and
    the first answer ends it.  When none comes, the result is the search as
    asked - the most faithful thing there is to return - and the record says
    nothing was found.
    """
    first = search(None)
    record = AutoTemperature(ceiling=float(ceiling), tries=1)
    chosen = pick(first)
    if chosen is not None:
        record.found = True
        record.temperature = float(temperature) if sampled else None
        return chosen, record
    record.climbed = True
    for t in temperature_ladder(temperature, ceiling):
        for _draw in range(1 if t == 0 else AUTO_DRAWS):
            record.tries += 1
            chosen = pick(search(t))
            if chosen is not None:
                record.found = True
                record.temperature = t
                return chosen, record
    return first, record


# -- the two searches every kind has --------------------------------------------


def predict(model: "GraphModel", prefix: str, auto_temperature: float = 0.0, **options: Any):
    """``model.predict(prefix, **options)``, climbing the ladder when it finds no answer.

    Returns ``(result, record)``; the record is ``None`` when ``auto_temperature`` is 0 (off), and the search is
    then exactly the one :meth:`~radixnet.model.RadixNet.predict` runs.  Any kind, any mode: the walks up the
    ladder are the same request in ``"sample"`` mode at each rung's temperature.
    """
    check_auto_temperature(auto_temperature)
    if not auto_temperature:
        return model.predict(prefix, **options), None
    to_end = bool(options.get("to_end", False))

    def search(t: float | None):
        return model.predict(prefix, **(options if t is None else {**options, "mode": "sample", "temperature": t}))

    return climb(search, lambda result: first_answer(result, to_end), temperature=float(options.get("temperature", 1.0)),
                 ceiling=auto_temperature, sampled=str(options.get("mode", "")).lower() == "sample")


def wrote(model: "GraphModel", result: "PathResult", prefix: str) -> bool:
    """Did a generated text say anything beyond its prefix?  A generated ``text`` is the whole text, prefix
    included, so it is measured against the prefix in the model's units."""
    enc = model.encoding
    return bool(result.text.strip()) and enc.length(result.text) > enc.length(prefix)


def generate(model: "GraphModel", count: int = 1, auto_temperature: float = 0.0, **options: Any):
    """``model.generate(count=count, **options)``, with every sample that said nothing walked again up the ladder.

    Returns ``(results, record)``; the record is ``None`` when ``auto_temperature`` is 0 (off).  The generation
    as asked runs first; when fewer than ``count`` of its texts say anything beyond the prefix, the missing ones
    are drawn as sampled walks up the ladder, :data:`AUTO_DRAWS` generations a rung, each asking for as many as
    are still missing.  What comes back is then the texts that said something - those of the generation as asked,
    in its order, then the ones the ladder found - and fewer than ``count`` only when even the ceiling found too
    few.  A ``seed`` makes every generation draw from its own generator: the k-th retry draws from ``seed + k``,
    so a retry at the same temperature is a new draw and the whole climb is reproducible.
    """
    check_auto_temperature(auto_temperature)
    if not auto_temperature:
        return model.generate(count=count, **options), None
    prefix = str(options.get("prefix", "") or "")
    mode = str(options.get("mode") or "sample").lower()
    temperature = float(options.get("temperature", 1.0))
    seed = options.get("seed")
    results = model.generate(count=count, **options)
    record = AutoTemperature(ceiling=float(auto_temperature), tries=1)
    kept = [r for r in results if wrote(model, r, prefix)]
    if len(kept) >= count:
        record.found = True
        record.temperature = temperature if mode == "sample" else None
        return results, record
    record.climbed = True
    for t in temperature_ladder(temperature, auto_temperature):
        for _draw in range(1 if t == 0 else AUTO_DRAWS):
            need = count - len(kept)
            retry = {**options, "mode": "sample", "temperature": t}
            if seed is not None:
                retry["seed"] = int(seed) + record.tries
            record.tries += 1
            kept.extend([r for r in model.generate(count=need, **retry) if wrote(model, r, prefix)][:need])
            if len(kept) >= count:
                record.found = True
                record.temperature = t
                return kept, record
    return kept, record
