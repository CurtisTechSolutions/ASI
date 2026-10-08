"""Auto prune: the graph lets go of the edges nothing walks, and the nodes those edges stranded.

Everything the graph learns, it keeps.  A text read once leaves its edges
behind for good; a split leaves a bridge that nothing crosses again; a text
registered for scoring (``count=False``) leaves nodes that were never walked
at all.  Compression (:meth:`radixnet.graph.RadixCyclicGraph.compress`) only
ever *coarsens* that structure - it merges, it never removes - so a model
taught for long enough carries every transition it ever saw, at the cost of
every search that has to price them.  **Auto prune** is the other half of
self-compression: the graph lets go of what it has stopped using.

The rule is two thresholds over what the graph already counts - an edge's
traversals, or in the negative network, which counts nothing, the evidence it
carries - applied to every edge leaving a node:

* **min_count** - an edge traversed fewer than this many times is pruned
  (``1``, the default, prunes only what was never traversed; ``0`` switches
  the rule off);
* **min_share** - an edge taking less than this share of its node's
  out-traversals is pruned (``0``, the default, switches the rule off; it is
  what thins a busy node of its rare continuations).

An edge is **never** pruned while something was *taught* about it rather than
observed: a hand-over into ``BACK`` or ``THINK`` (lessons from experience,
not from a corpus), the count model's rewards and judged contexts, the
negative network's blame and clearing, the phase model's rewards.  Those are
exactly the edges compression keeps apart as well.  And a count of 0 is read
as *unknown*, not never, while a node's out-edges do not account for what came
in: the bridge of a split in a kind that counts edges and not nodes starts at
0, and the deficit says something crossed it.  After the edges go, every real
node left with **no way in** (nothing reaches it any more) or **no way out**
(a walk through it could never end) goes too, with the edges it still had -
unless one of them is protected, in which case the node stays nameable, the
way a blamed fragment does.  Removing edges leaves unary chains behind, so the
model compresses once afterwards.

A prune happens **by hand** (``radixnet prune --now``, ``POST
/api/model/prune/now``) or **automatically**, at the end of every ``every``-th
training epoch while ``auto`` is set - after the compression and the window's
step the epoch already does.  Switched off (the default, and every model
before this existed) nothing here touches the graph.

The setting is the model's, like its dynamic window: saved in the graph
document while it is on (``"auto_prune": {"min_count": 1, "min_share": 0.0,
"every": 1, "auto": true}``) and switchable at any time.  ``../SPEC-AutoPrune.md``
is the specification.  Python only: the Go and Rust ports keep the setting on
file (they ignore the block) but do not prune.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_EVERY", "DEFAULT_MIN_COUNT", "DEFAULT_MIN_SHARE", "AutoPrune", "check_thresholds"]

DEFAULT_MIN_COUNT = 1
"""An edge traversed fewer than this many times is pruned: by default, only one that was never traversed."""

DEFAULT_MIN_SHARE = 0.0
"""An edge taking less than this share of its node's out-traversals is pruned: by default, none."""

DEFAULT_EVERY = 1
"""Prune at the end of every this-many-th training epoch: by default, every one."""


def check_thresholds(min_count: int, min_share: float, every: int) -> None:
    """``ValueError`` unless the thresholds make sense: a whole ``min_count >= 0``, a share in ``[0, 1)``, ``every >= 1``."""
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 0:
        raise ValueError(f"min_count is a whole number of traversals, 0 or more, got {min_count!r}")
    if isinstance(min_share, bool) or not isinstance(min_share, (int, float)) or not (0.0 <= float(min_share) < 1.0):
        raise ValueError(f"min_share is a share of a node's traversals, from 0 up to but not including 1, got {min_share!r}")
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError(f"every is a whole number of epochs, 1 or more, got {every!r}")


@dataclass(frozen=True)
class AutoPrune:
    """The pruning a graph is held to: off, or two thresholds and when they are applied.

    ``min_count`` and ``min_share`` are the rule (:mod:`radixnet.prune`);
    ``every`` says at the end of which epochs it runs and ``auto`` whether it
    runs by itself at all - off, the graph is pruned only when asked.  A value
    of the model, saved with it while it is on, and switchable at any time:
    the graph is pruned by a *prune*, never by the setting itself.
    """

    on: bool = False
    min_count: int = DEFAULT_MIN_COUNT
    min_share: float = DEFAULT_MIN_SHARE
    every: int = DEFAULT_EVERY
    auto: bool = True

    def __post_init__(self) -> None:
        try:
            min_count, every = int(self.min_count), int(self.every)
            min_share = float(self.min_share)
        except (TypeError, ValueError):
            raise ValueError(
                f"min_count and every are whole numbers and min_share a share, got "
                f"{self.min_count!r}, {self.min_share!r}, {self.every!r}"
            ) from None
        check_thresholds(min_count, min_share, every)
        object.__setattr__(self, "on", bool(self.on))
        object.__setattr__(self, "min_count", min_count)
        object.__setattr__(self, "min_share", min_share)
        object.__setattr__(self, "every", every)
        object.__setattr__(self, "auto", bool(self.auto))

    def due(self, epoch: int) -> bool:
        """Does the automatic prune run at the end of lifetime epoch ``epoch``?  Never while off or by hand."""
        return self.on and self.auto and epoch % self.every == 0

    def to_dict(self) -> dict | None:
        """The ``auto_prune`` block of a graph document, or ``None`` - nothing is written - while off."""
        if not self.on:
            return None
        return {"min_count": self.min_count, "min_share": self.min_share, "every": self.every, "auto": self.auto}

    @classmethod
    def from_dict(cls, d: dict | None) -> "AutoPrune":
        """Read an ``auto_prune`` block; a document without one was written with pruning off."""
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise ValueError(f"an auto_prune block is an object, got {type(d).__name__}")
        for key in ("min_count", "every"):
            value = d.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"auto_prune {key} must be a whole number, got {value!r}")
        share = d.get("min_share")
        if share is not None and (isinstance(share, bool) or not isinstance(share, (int, float))):
            raise ValueError(f"auto_prune min_share must be a number, got {share!r}")
        auto = d.get("auto", True)
        if not isinstance(auto, bool):
            raise ValueError(f"auto_prune auto must be true or false, got {auto!r}")
        return cls(
            True,
            DEFAULT_MIN_COUNT if d.get("min_count") is None else d["min_count"],
            DEFAULT_MIN_SHARE if share is None else share,
            DEFAULT_EVERY if d.get("every") is None else d["every"],
            auto,
        )

    def rule(self) -> str:
        """The thresholds in words: what an edge must fall short of to be pruned."""
        parts = []
        if self.min_count > 0:
            parts.append(f"traversed fewer than {self.min_count} time{'s' if self.min_count != 1 else ''}")
        if self.min_share > 0:
            parts.append(f"under {self.min_share:g} of its node's traversals")
        return " or ".join(parts) if parts else "nothing (both thresholds are off)"

    def __str__(self) -> str:
        if not self.on:
            return "off"
        return f"min_count {self.min_count}, min_share {self.min_share:g}, every {self.every}"

    def describe(self) -> str:
        """The human form: ``'off: nothing is pruned'`` or the rule, when it runs and whether it runs by itself."""
        if not self.on:
            return "off: nothing is pruned, the graph keeps every edge it ever learned"
        when = (
            f"at the end of every {'training epoch' if self.every == 1 else f'{self.every} training epochs'}"
            if self.auto else "by hand (prune --now)"
        )
        return f"on: an edge {self.rule()} is pruned, with the nodes it strands, {when}"
