"""Intensity <-> pixel: how loud becomes how dark, and how to read it back.

The rule the format is built on is the one in the brief: **black is 1.0 and
white is 0.0**.  An intensity of 1 (the loudest thing in the clip) is ink; an
intensity of 0 (silence) is paper.  :data:`GRAY` is that rule and nothing else,
which is why it is the only map the codec uses by default - it is exact at any
bit depth and needs no lookup to invert::

    value = round((1 - intensity) * maxval)
    intensity = 1 - value / maxval

The other maps are for looking at.  They are 256-entry lookup tables built by
interpolating a handful of control points, and they are inverted by finding
which entry a pixel matches: a dictionary hit for a picture that came out of
this tool untouched, and a nearest-colour search for one that did not (a
screenshot, a resized copy, something rendered by another program).

That nearest-colour search is a k-nearest-neighbour query in RGB space against
256 points.  A KD-tree (``sklearn.neighbors.KDTree``) answers it, and was tried
here, but two table entries are regularly the *same* distance from an off-table
pixel - 55 times in 3000 random colours on the ``viridis`` ramp - and a tree
breaks those ties by its own layout.  A codec cannot decode one waveform on a
machine with scikit-learn installed and a different one on a machine without
it, so :class:`NearestColor` instead settles every tie the same way (the lowest
index wins) and gets its speed from doing the whole picture at once: numpy
scores all of an image's distinct colours against all 256 entries in one
vectorised ``argmin`` - which returns the first minimum, so the tie rule is
free - and falls back to a memoising scan when numpy is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "COLORMAPS",
    "ColorMap",
    "GRAY",
    "NearestColor",
    "get_colormap",
]

_Control = tuple[float, tuple[int, int, int]]

_CHUNK = 8192
"""How many unfamiliar colours are scored per numpy call, to bound the memory it takes."""

# Control points of each ramp: (position 0..1, RGB).  Interpolated linearly
# into a 256-entry table.  Every map keeps the brief's convention at the ends -
# 0.0 is the pale/quiet end and 1.0 the dark/loud end - except "viridis" and
# "magma", which are drawn in the direction people know them from.
_RAMPS: dict[str, tuple[_Control, ...]] = {
    "gray": ((0.0, (255, 255, 255)), (1.0, (0, 0, 0))),
    "gray-inv": ((0.0, (0, 0, 0)), (1.0, (255, 255, 255))),
    "fire": (
        (0.0, (255, 255, 255)),
        (0.25, (255, 237, 160)),
        (0.5, (254, 178, 76)),
        (0.75, (227, 26, 28)),
        (1.0, (0, 0, 0)),
    ),
    "ice": (
        (0.0, (255, 255, 255)),
        (0.25, (198, 234, 241)),
        (0.5, (94, 176, 214)),
        (0.75, (30, 76, 160)),
        (1.0, (0, 0, 0)),
    ),
    "viridis": (
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ),
    "magma": (
        (0.0, (0, 0, 4)),
        (0.25, (81, 18, 124)),
        (0.5, (183, 55, 121)),
        (0.75, (252, 137, 97)),
        (1.0, (252, 253, 191)),
    ),
}

COLORMAPS = tuple(_RAMPS)


def _build_lut(controls: Sequence[_Control], size: int = 256) -> list[tuple[int, int, int]]:
    """Interpolate control points into a table of ``size`` colours."""
    lut: list[tuple[int, int, int]] = []
    for i in range(size):
        t = i / (size - 1)
        lo = controls[0]
        hi = controls[-1]
        for a, b in zip(controls, controls[1:]):
            if a[0] <= t <= b[0]:
                lo, hi = a, b
                break
        span = hi[0] - lo[0]
        f = 0.0 if span <= 0 else (t - lo[0]) / span
        lut.append(tuple(int(round(lo[1][c] + (hi[1][c] - lo[1][c]) * f)) for c in range(3)))  # type: ignore[arg-type]
    return lut


# ---------------------------------------------------------------------------
# nearest colour
# ---------------------------------------------------------------------------


class NearestColor:
    """Which table entry a pixel is closest to, in RGB space.

    Exact colours - every pixel of a picture this tool wrote - are answered
    from a dictionary.  Anything else (a screenshot, a resized copy, a picture
    some other program drew) goes to a nearest-colour search, which numpy runs
    over every distinct colour in the image at once when it is installed.

    Ties are always broken towards the lowest table index, whichever path runs,
    so a picture decodes to the same waveform on every machine.
    """

    __slots__ = ("lut", "_exact", "_cache", "_array", "backend")

    def __init__(self, lut: Sequence[tuple[int, int, int]], use_numpy: bool = True) -> None:
        self.lut = list(lut)
        # a ramp can repeat a colour; the *first* index wins so inversion is stable
        self._exact: dict[tuple[int, int, int], int] = {}
        for i, rgb in enumerate(self.lut):
            self._exact.setdefault(rgb, i)
        self._cache: dict[tuple[int, int, int], int] = {}
        self._array: Any = None
        self.backend = "scan"
        if use_numpy:
            try:
                import numpy as np

                self._array = np.asarray(self.lut, dtype=np.int32)
                self.backend = "numpy"
            except Exception:
                self._array = None

    def _scan(self, rgb: tuple[int, int, int]) -> int:
        """Closest entry by brute force; the lowest index wins a tie (strict ``<``)."""
        r, g, b = rgb
        best = 0
        best_d = -1
        for i, (lr, lg, lb) in enumerate(self.lut):
            d = (r - lr) ** 2 + (g - lg) ** 2 + (b - lb) ** 2
            if best_d < 0 or d < best_d:
                best, best_d = i, d
                if d == 0:
                    break
        return best

    def index(self, rgb: tuple[int, int, int]) -> int:
        """The table index closest to ``rgb``."""
        hit = self._exact.get(rgb)
        if hit is not None:
            return hit
        hit = self._cache.get(rgb)
        if hit is not None:
            return hit
        idx = self._scan(rgb)
        self._cache[rgb] = idx
        return idx

    def index_many(self, colors: Sequence[tuple[int, int, int]]) -> list[int]:
        """The table index for every colour, de-duplicated first.

        This is the path a whole picture takes.  A picture this tool wrote has
        at most 256 distinct colours and never reaches the search at all; a
        foreign one may have thousands, and numpy scores them together.
        """
        unknown: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for c in colors:
            if c not in self._exact and c not in self._cache and c not in seen:
                seen.add(c)
                unknown.append(c)
        if unknown:
            if self._array is not None:
                import numpy as np

                # squared distance expanded as |a|^2 - 2a.b + |b|^2.  The |a|^2
                # term is constant across a row and cannot move the argmin, so
                # only the matrix product is computed - one BLAS call instead of
                # a colours x 256 x 3 intermediate.  Every value here is a small
                # integer held exactly in float64, so the arithmetic is exact
                # and argmin still ties to the lowest index, like _scan.
                lut = self._array
                lut_sq = (lut.astype(np.float64) ** 2).sum(axis=1)
                for start in range(0, len(unknown), _CHUNK):
                    block = unknown[start : start + _CHUNK]
                    pts = np.asarray(block, dtype=np.float64)
                    d = lut_sq[None, :] - 2.0 * (pts @ lut.astype(np.float64).T)
                    for c, i in zip(block, d.argmin(axis=1).tolist()):
                        self._cache[c] = int(i)
            else:
                for c in unknown:
                    self._cache[c] = self._scan(c)
        return [self._exact.get(c, self._cache.get(c, 0)) for c in colors]

    def __repr__(self) -> str:
        return f"NearestColor(entries={len(self.lut)}, backend={self.backend!r})"


# ---------------------------------------------------------------------------
# the maps
# ---------------------------------------------------------------------------


@dataclass
class ColorMap:
    """A named intensity ramp, and its inverse."""

    name: str
    lut: list[tuple[int, int, int]] = field(default_factory=list)
    _finder: NearestColor | None = field(default=None, repr=False, compare=False)

    @property
    def is_gray(self) -> bool:
        """``gray`` is handled arithmetically, so it is exact at 16 bits too."""
        return self.name == "gray"

    @property
    def mode(self) -> str:
        """The image mode a picture in this map needs."""
        return "L" if self.name in ("gray", "gray-inv") else "RGB"

    def to_value(self, intensity: float, maxval: int = 255) -> int:
        """Greyscale only: intensity -> pixel value (``1.0`` -> black)."""
        t = 0.0 if intensity < 0.0 else (1.0 if intensity > 1.0 else intensity)
        if self.name == "gray-inv":
            return int(round(t * maxval))
        return int(round((1.0 - t) * maxval))

    def from_value(self, value: int, maxval: int = 255) -> float:
        """Greyscale only: pixel value -> intensity."""
        t = value / maxval if maxval else 0.0
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return t if self.name == "gray-inv" else 1.0 - t

    def to_rgb(self, intensity: float) -> tuple[int, int, int]:
        """Intensity -> colour, through the table."""
        t = 0.0 if intensity < 0.0 else (1.0 if intensity > 1.0 else intensity)
        return self.lut[int(round(t * (len(self.lut) - 1)))]

    def from_rgb(self, rgb: tuple[int, int, int]) -> float:
        """Colour -> intensity, by finding the table entry it matches."""
        return self.finder.index(rgb) / (len(self.lut) - 1)

    def from_rgb_many(self, colors: Sequence[tuple[int, int, int]]) -> list[float]:
        """Colour -> intensity for a whole picture's worth of pixels at once."""
        scale = 1.0 / (len(self.lut) - 1)
        return [i * scale for i in self.finder.index_many(colors)]

    @property
    def finder(self) -> NearestColor:
        """The nearest-colour index (built on first use)."""
        if self._finder is None:
            self._finder = NearestColor(self.lut)
        return self._finder

    def swatch(self, steps: int = 8) -> list[tuple[int, int, int]]:
        """A few evenly spaced colours, for a legend."""
        return [self.to_rgb(i / (steps - 1)) for i in range(steps)]


_CACHE: dict[str, ColorMap] = {}


def get_colormap(name: str = "gray") -> ColorMap:
    """Look up a map by name."""
    key = (name or "gray").lower()
    if key not in _RAMPS:
        raise ValueError(f"unknown colormap {key!r}; choose from {', '.join(COLORMAPS)}")
    if key not in _CACHE:
        _CACHE[key] = ColorMap(key, _build_lut(_RAMPS[key]))
    return _CACHE[key]


GRAY = get_colormap("gray")
"""The canonical map: black is 1.0, white is 0.0."""
