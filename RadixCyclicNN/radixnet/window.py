"""The dynamic window: a ladder of node sizes, halving from 32 to 4 and back up.

A merged node (:meth:`radixnet.graph.RadixCyclicGraph.merge_child`) can hold
a whole sentence: compression turns every unary chain into one node, however
long, and a walk through it has nowhere to branch - the node is entered at
its first gram and left at its last.  The **dynamic window** is a ceiling on
that length, and the ceiling moves.  It is sized in the binary number system:
it starts at 32 units, then moves to 16, then 8, then 4 (the *floor*), and
then it goes back up to 32 and the process runs again::

    32 -> 16 -> 8 -> 4 -> 32 -> 16 -> ...

One **step** of the ladder does three things to the graph, in this order:

1. **merge** - compression runs, and merges nothing longer than the window
   (the ceiling is the current size);
2. **halve** - every node longer than the window is split into two halves at
   its middle gram (:meth:`radixnet.graph.RadixCyclicGraph.split_window`).
   Both halves carry the same data for traversal - the node's state, its
   activation parameters and its visit count - and the two are joined by a
   **heavy connection**: an edge that carries every traversal the node ever
   had, and in the sine model the heavy weight :data:`radixnet.graph.W_HEAVY`.
   A half still longer than the window is halved again, so after the step no
   node is longer than the window;
3. **move** - the window halves; below the floor it goes back to the top.

A step happens **manually** (``radixnet window --step``, ``POST
/api/model/window/step``, the Step button on the Model settings tab) or
**automatically**, at the end of every training epoch while ``auto`` is set -
where compression already runs, so there the step is the halving and the
move.  Switched off (the default, and every model before this existed) nothing
here touches the graph and compression is unbounded, as it always was.

The setting is the model's, like its attention band: saved in the graph
document while it is on (``"dynamic_window": {"top": 32, "floor": 4, "size":
16, "auto": true}``) and switchable at any time.  ``../SPEC-DynamicWindow.md``
is the specification; the Go (``go/radixnet/window.go``) and Rust
(``rust/src/window.rs``) ports keep it node for node.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_FLOOR", "DEFAULT_TOP", "DynamicWindow", "check_ladder", "is_power_of_two", "ladder"]

DEFAULT_TOP = 32
"""Where the ladder starts: the largest window, in the encoding's units."""

DEFAULT_FLOOR = 4
"""Where the ladder ends before it goes back up: the smallest window."""


def is_power_of_two(value: int) -> bool:
    """Whether ``value`` is ``1, 2, 4, 8, ...`` - a size in the binary number system."""
    return value >= 1 and value & (value - 1) == 0


def check_ladder(top: int, floor: int, size: int | None = None) -> None:
    """``ValueError`` unless ``top`` and ``floor`` are powers of two with ``floor <= top`` (and ``size`` on the ladder).

    The messages are the ones the Go and Rust ports raise, word for word.
    """
    if not is_power_of_two(top):
        raise ValueError(f"the window is sized in the binary number system: top must be a power of two, got {top}")
    if not is_power_of_two(floor):
        raise ValueError(f"the window is sized in the binary number system: floor must be a power of two, got {floor}")
    if floor > top:
        raise ValueError(f"floor must not exceed top, got floor {floor} over top {top}")
    if size is not None and (not is_power_of_two(size) or size < floor or size > top):
        raise ValueError(f"size must be a power of two on the ladder {floor}..{top}, got {size}")


def ladder(top: int, floor: int) -> list[int]:
    """The sizes the window takes, in order: ``top, top / 2, ..., floor``."""
    check_ladder(top, floor)
    sizes = []
    size = top
    while size >= floor:
        sizes.append(size)
        size //= 2
    return sizes


@dataclass(frozen=True)
class DynamicWindow:
    """The window a graph is held to: off (``top`` is ``None``), or a ladder ``top .. floor`` standing at ``size``.

    ``auto`` says whether the ladder steps by itself at the end of every
    training epoch; off, it moves only when a step is asked for.  A value of
    the model, saved with it while it is on, and switchable at any time: the
    graph is halved and merged by the *steps*, never by the setting itself.
    """

    top: int | None = None
    floor: int = DEFAULT_FLOOR
    size: int | None = None
    auto: bool = True

    def __post_init__(self) -> None:
        if self.top is None:
            object.__setattr__(self, "size", None)
            return
        try:
            top, floor = int(self.top), int(self.floor)
            size = top if self.size is None else int(self.size)
        except (TypeError, ValueError):
            raise ValueError(
                f"the window is sized in the binary number system: top, floor and size are whole numbers, got "
                f"{self.top!r}, {self.floor!r}, {self.size!r}"
            ) from None
        check_ladder(top, floor, size)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "floor", floor)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "auto", bool(self.auto))

    @property
    def on(self) -> bool:
        """Is the window on?  Off, compression is unbounded and no node is ever halved."""
        return self.top is not None

    def sizes(self) -> list[int]:
        """The ladder, top to floor; empty while off."""
        return ladder(self.top, self.floor) if self.top is not None else []

    def next_size(self) -> int | None:
        """The size after this one: half of it, or the top again from the floor (``None`` while off)."""
        if self.top is None:
            return None
        half = self.size // 2
        return half if half >= self.floor else self.top

    def advanced(self) -> "DynamicWindow":
        """The window one step down the ladder (back at the top from the floor)."""
        if self.top is None:
            return self
        return DynamicWindow(self.top, self.floor, self.next_size(), self.auto)

    def to_dict(self) -> dict | None:
        """The ``dynamic_window`` block of a graph document, or ``None`` - nothing is written - while off."""
        if self.top is None:
            return None
        return {"top": self.top, "floor": self.floor, "size": self.size, "auto": self.auto}

    @classmethod
    def from_dict(cls, d: dict | None) -> "DynamicWindow":
        """Read a ``dynamic_window`` block; a document without one was written with the window off."""
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise ValueError(f"a dynamic_window block is an object, got {type(d).__name__}")
        top = d.get("top")
        if top is None:
            return cls()
        for key in ("top", "floor", "size"):
            value = d.get(key, DEFAULT_FLOOR if key == "floor" else None)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"dynamic_window {key} must be a whole number, got {value!r}")
        auto = d.get("auto", True)
        if not isinstance(auto, bool):
            raise ValueError(f"dynamic_window auto must be true or false, got {auto!r}")
        return cls(top, d.get("floor", DEFAULT_FLOOR), d.get("size"), auto)

    def __str__(self) -> str:
        if self.top is None:
            return "off"
        return f"{' -> '.join(str(s) for s in self.sizes())}, at {self.size}"

    def describe(self, units: str = "units") -> str:
        """The human form: ``'off: compression is unbounded'`` or the ladder, where it stands and whether it steps by itself."""
        if self.top is None:
            return "off: compression is unbounded, and no node is halved"
        when = "stepping at the end of every training epoch" if self.auto else "stepping by hand (window --step)"
        return (
            f"on: {' -> '.join(str(s) for s in self.sizes())} {units}, at {self.size} "
            f"(next {self.next_size()}), {when}"
        )
