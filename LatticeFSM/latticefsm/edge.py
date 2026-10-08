"""Edge - one cell of the 3D matrix: a dense record of everything the machine knows about one transition.

An edge is ``(source, symbol, target)``: *in state* ``source``, *reading*
``symbol``, *the machine may move to* ``target``.  The matrix holds one for
every combination, so an edge is never created or deleted - it is only ever
written to.  What it holds:

* **a history** - ``seen`` (traversals, ever), ``first_seen`` and ``last_seen``
  (clock readings), and ``recent``, a trace that is set at each traversal and
  fades on the clock (:meth:`Edge.recent_at`);
* **a verdict** - ``rewarded`` and ``punished`` (credit received, ever, each
  signed positive), with ``last_rewarded`` and ``last_punished``;
* **a width** - how wide the channel is.  Use and reward widen it, punishment
  narrows it, and disuse lets it relax back toward one on the clock
  (:meth:`Edge.width_at`).  Under *stimulation* a wide edge is easier to
  traverse than a narrow one, and the more stimulation the more so (§3 of
  ``DESIGN.md``);
* **its own weighting function** - a :class:`Weighting`: the coefficients of
  the function that turns this edge's features into the log-weight the walk
  selects by.  Every edge starts from the machine's prototype and adapts its
  own copy with the credit it receives (:meth:`Weighting.adapt`), so two edges
  that have lived different lives weigh their features differently;
* ``features`` - the features as they were at the last traversal, which is
  what the adaptation credits.

Nothing here knows about the matrix or the machine; the clock and the
stimulation are passed in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

__all__ = ["COEFFICIENT_LIMIT", "FEATURES", "Edge", "Weighting", "WIDTH_MAX", "WIDTH_MIN", "WIDTH_REST"]

FEATURES = ("seen", "recent", "net", "age", "width")
"""The features an edge's weighting function reads, in the order :meth:`Edge.features_at` returns them.

``seen``: ``log1p(seen)``.  ``recent``: the traversal trace read through the clock.  ``net``: credit per
traversal, ``(rewarded - punished) / (seen + 1)``, in ``[-r, r]`` for credits of size ``r``.  ``age``: how recently
the edge was traversed, ``0.5 ** (elapsed / life)``, zero if never.  ``width``: ``log(width)``, zero at rest."""

WIDTH_REST, WIDTH_MIN, WIDTH_MAX = 1.0, 0.05, 20.0
"""A channel's width at rest, and the narrowest and widest it can be."""

COEFFICIENT_LIMIT = 8.0
"""Every coefficient of a weighting is clipped to ``[-limit, limit]`` after adapting, so no edge can become certain."""


@dataclass(slots=True)
class Weighting:
    """The coefficients of one edge's weighting function.

    ``log_weight = bias + seen * f_seen + recent * f_recent + net * f_net + age * f_age + width * f_width``

    - where ``f_*`` are the edge's features (:data:`FEATURES`).  The width term
    the *stimulation* adds is not here: stimulation is the machine's state,
    not the edge's opinion.  ``rate`` is this edge's own learning rate.

    The defaults are the prototype a fresh machine gives every edge: nothing
    but ``net``, so an unadapted edge weighs its own credit per traversal and
    nothing else.  Pass another :class:`Weighting` to the machine to change
    the prototype; each edge adapts its own copy from there.
    """

    bias: float = 0.0
    seen: float = 0.0
    recent: float = 0.0
    net: float = 1.0
    age: float = 0.0
    width: float = 0.0
    rate: float = 0.1

    def log_weight(self, features: tuple[float, ...]) -> float:
        """This edge's opinion of itself: its coefficients applied to its features."""
        f_seen, f_recent, f_net, f_age, f_width = features
        return (self.bias + self.seen * f_seen + self.recent * f_recent + self.net * f_net
                + self.age * f_age + self.width * f_width)

    def adapt(self, credit: float, features: tuple[float, ...]) -> None:
        """Move the coefficients with ``credit`` (signed: reward positive, punishment negative).

        The bias moves by ``rate * credit``; each coefficient moves by ``rate *
        credit * feature`` - the feature as it was when the edge was traversed
        - so a feature that was present when the edge was rewarded comes to
        count for it, and one present when it was punished comes to count
        against it.  Every coefficient is clipped to ``COEFFICIENT_LIMIT``.
        """
        step = self.rate * credit
        self.bias = _clip(self.bias + step)
        self.seen = _clip(self.seen + step * features[0])
        self.recent = _clip(self.recent + step * features[1])
        self.net = _clip(self.net + step * features[2])
        self.age = _clip(self.age + step * features[3])
        self.width = _clip(self.width + step * features[4])

    def copy(self) -> Weighting:
        return Weighting(self.bias, self.seen, self.recent, self.net, self.age, self.width, self.rate)

    def to_list(self) -> list[float]:
        return [self.bias, self.seen, self.recent, self.net, self.age, self.width, self.rate]

    @classmethod
    def from_list(cls, values: list[float]) -> Weighting:
        return cls(*[float(v) for v in values])


def _clip(value: float) -> float:
    return max(-COEFFICIENT_LIMIT, min(COEFFICIENT_LIMIT, value))


@dataclass(slots=True)
class Edge:
    """One transition of the machine and everything it has lived through."""

    source: int
    symbol: int
    target: int
    seen: int = 0
    """Traversals, ever."""
    first_seen: int = -1
    """The clock reading of the first traversal; ``-1`` never."""
    last_seen: int = -1
    """The clock reading of the latest traversal; ``-1`` never."""
    recent: float = 0.0
    """The traversal trace as it was written at ``last_seen``; :meth:`recent_at` reads it through the clock."""
    rewarded: float = 0.0
    """Reward credited, ever."""
    punished: float = 0.0
    """Punishment credited, ever (a positive number)."""
    last_rewarded: int = -1
    last_punished: int = -1
    width: float = WIDTH_REST
    """The channel's width as it was written at ``width_stamp``; :meth:`width_at` reads it through the clock."""
    width_stamp: int = 0
    weighting: Weighting = field(default_factory=Weighting)
    """This edge's own weighting function."""
    features: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    """The features as they were at the latest traversal: what :meth:`credit` adapts the weighting with."""

    # ---- what the edge is right now ------------------------------------------------------------------------------

    def recent_at(self, clock: int, life: float) -> float:
        """The traversal trace, faded by a half every ``life`` clock ticks since the last traversal."""
        if self.last_seen < 0:
            return 0.0
        return self.recent * 0.5 ** ((clock - self.last_seen) / life)

    def age_at(self, clock: int, life: float) -> float:
        """How recently the edge was traversed, one at the moment of traversal and a half a life later."""
        if self.last_seen < 0:
            return 0.0
        return 0.5 ** ((clock - self.last_seen) / life)

    def width_at(self, clock: int, life: float) -> float:
        """The width, relaxed toward :data:`WIDTH_REST` by a half every ``life`` ticks since it was last written."""
        if self.width == WIDTH_REST:
            return WIDTH_REST
        return WIDTH_REST + (self.width - WIDTH_REST) * 0.5 ** ((clock - self.width_stamp) / life)

    @property
    def net(self) -> float:
        """Credit per traversal: ``(rewarded - punished) / (seen + 1)``."""
        return (self.rewarded - self.punished) / (self.seen + 1)

    def features_at(self, clock: int, life: float) -> tuple[float, float, float, float, float]:
        """The five features of :data:`FEATURES`, as they are at ``clock``."""
        return (math.log1p(self.seen), self.recent_at(clock, life), self.net, self.age_at(clock, life),
                math.log(self.width_at(clock, life)))

    def log_weight(self, clock: int, life: float, stimulation: float) -> float:
        """What the walk selects by: the edge's own function of its features, plus ``stimulation * log(width)``.

        The stimulation term is the whole of §3: at zero stimulation width is
        not consulted; at one an edge twice as wide is twice as likely; at
        three, eight times.
        """
        features = self.features_at(clock, life)
        return self.weighting.log_weight(features) + stimulation * features[4]

    # ---- what happens to it ----------------------------------------------------------------------------------------

    def traverse(self, clock: int, life: float, trace: float = 1.0, widen: float = 0.0) -> None:
        """Record a traversal at ``clock``: features are noted, ``seen`` adds one, the trace is set, the channel widens."""
        self.features = self.features_at(clock, life)
        self.recent = self.recent_at(clock, life) + trace
        if self.first_seen < 0:
            self.first_seen = clock
        self.last_seen = clock
        self.seen += 1
        if widen:
            self._set_width(self.width_at(clock, life) * (1.0 + widen), clock)

    def credit(self, amount: float, clock: int, life: float, widen: float = 0.0, narrow: float = 0.0) -> None:
        """Receive ``amount`` of credit: reward if positive, punishment if negative.

        The verdict is written down, the channel widens by ``widen * amount``
        (or narrows by ``narrow * |amount|``), and the weighting adapts with
        the features of the traversal being credited.
        """
        if amount > 0:
            self.rewarded += amount
            self.last_rewarded = clock
            if widen:
                self._set_width(self.width_at(clock, life) * (1.0 + widen * amount), clock)
        elif amount < 0:
            self.punished -= amount
            self.last_punished = clock
            if narrow:
                self._set_width(self.width_at(clock, life) / (1.0 + narrow * -amount), clock)
        else:
            return
        self.weighting.adapt(amount, self.features)

    def set_width(self, width: float, clock: int) -> None:
        """Write the width outright (clipped), stamped at ``clock``."""
        self._set_width(width, clock)

    def _set_width(self, width: float, clock: int) -> None:
        self.width = max(WIDTH_MIN, min(WIDTH_MAX, width))
        self.width_stamp = clock

    @property
    def touched(self) -> bool:
        """Whether anything was ever written to this edge; a fresh matrix holds nothing but untouched edges."""
        return self.seen > 0 or self.rewarded != 0.0 or self.punished != 0.0 or self.width != WIDTH_REST

    # ---- persistence -----------------------------------------------------------------------------------------------

    def to_list(self) -> list:
        """The edge as a flat list: ``[source, symbol, target, seen, first_seen, last_seen, recent, rewarded, punished,
        last_rewarded, last_punished, width, width_stamp, [weighting], [features]]``."""
        return [self.source, self.symbol, self.target, self.seen, self.first_seen, self.last_seen, self.recent,
                self.rewarded, self.punished, self.last_rewarded, self.last_punished, self.width, self.width_stamp,
                self.weighting.to_list(), list(self.features)]

    @classmethod
    def from_list(cls, values: list) -> Edge:
        (source, symbol, target, seen, first_seen, last_seen, recent, rewarded, punished, last_rewarded,
         last_punished, width, width_stamp, weighting, features) = values
        return cls(int(source), int(symbol), int(target), int(seen), int(first_seen), int(last_seen), float(recent),
                   float(rewarded), float(punished), int(last_rewarded), int(last_punished), float(width),
                   int(width_stamp), Weighting.from_list(weighting), tuple(float(f) for f in features))


EDGE_FIELDS = tuple(f.name for f in fields(Edge))
"""Every field of an edge, in order - what ``stats`` and the README's table list."""
