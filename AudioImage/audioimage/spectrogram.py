"""The plane itself: time across, frequency up, loudness as ink.

This module holds the mapping the whole tool is built around, and nothing
else - no files, no pixels, no transforms.  A :class:`Spectrogram` (what the
STFT produced: a magnitude per frequency bin per frame) goes in, and a *plane*
comes out: a rectangle of numbers in ``[0, 1]`` where

* **X is time** - one column per analysis frame, left to right,
* **Y is frequency** - one row per frequency, low at the bottom by default,
* **the value is amplitude** - ``1.0`` is as loud as the clip gets and becomes
  black, ``0.0`` is silence and becomes white.

Three choices decide what the picture looks like, and all three are reversible:

``scale``
    How a magnitude becomes a number in ``[0, 1]``.  ``db`` (the default)
    spaces it the way hearing does - a quiet harmonic a thousandth of the peak
    is still clearly visible - across a ``top_db`` window below the loudest
    point.  ``linear`` is the raw ratio, and shows almost nothing but the
    fundamentals.  ``sqrt`` sits between the two.

``freq_scale``
    Which frequency each row stands for.  ``linear`` gives every row one FFT
    bin, which is the only setting that needs no interpolation and so is the
    only exactly invertible one.  ``log`` and ``mel`` compress the top of the
    range, which is where a picture of speech or music starts to look like
    what you hear; they resample, so they cost a little accuracy.

``height`` / ``width``
    The size of the picture.  Leave them alone and the plane is exactly one
    row per bin and one column per frame.  Set them and the plane is resampled
    to fit - useful when something downstream wants a fixed rectangle.

Everything is reversed by :func:`from_plane`, which is what makes the picture
an encoding rather than an illustration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

__all__ = [
    "FREQ_SCALES",
    "ORIGINS",
    "SCALES",
    "PlaneConfig",
    "Spectrogram",
    "from_plane",
    "hz_to_mel",
    "mel_to_hz",
    "row_frequencies",
    "to_plane",
]

SCALES = ("db", "linear", "sqrt")
FREQ_SCALES = ("linear", "log", "mel")
ORIGINS = ("lower", "upper")

MIN_LOG_HZ = 1.0
"""``log`` rows start here rather than at 0 Hz, which has no logarithm."""


def hz_to_mel(hz: float) -> float:
    """Hertz to mels (O'Shaughnessy's formula, the common one)."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Mels back to hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@dataclass
class Spectrogram:
    """Magnitudes from an STFT: ``mags[frame][bin]``, plus how they were made."""

    mags: list[list[float]]
    sample_rate: int
    n_fft: int = 1024
    hop: int = 256
    window: str = "hann"
    center: bool = True

    @property
    def frames(self) -> int:
        """Number of time slices (the width of the plane)."""
        return len(self.mags)

    @property
    def bins(self) -> int:
        """Number of frequency bins, ``n_fft // 2 + 1``."""
        return len(self.mags[0]) if self.mags else self.n_fft // 2 + 1

    @property
    def peak(self) -> float:
        """The largest magnitude anywhere (``0.0`` for an empty or silent clip)."""
        return max((max(row) for row in self.mags if row), default=0.0)

    @property
    def nyquist(self) -> float:
        """Half the sample rate - the highest frequency the clip can hold."""
        return self.sample_rate / 2.0

    def bin_hz(self, index: float) -> float:
        """The frequency a (possibly fractional) bin index stands for."""
        return index * self.sample_rate / self.n_fft

    def hz_bin(self, hz: float) -> float:
        """The fractional bin index a frequency falls on."""
        return hz * self.n_fft / self.sample_rate

    def frame_time(self, index: float) -> float:
        """When a (possibly fractional) frame happens, in seconds."""
        return index * self.hop / float(self.sample_rate)

    @property
    def duration(self) -> float:
        """How long the analysed signal is, in seconds."""
        return self.frames * self.hop / float(self.sample_rate)

    def describe(self) -> dict[str, Any]:
        """A summary for ``info`` / ``--json`` output."""
        return {
            "frames": self.frames,
            "bins": self.bins,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "window": self.window,
            "center": self.center,
            "peak": round(self.peak, 8),
            "duration": round(self.duration, 6),
        }

    def __repr__(self) -> str:
        return f"Spectrogram(frames={self.frames}, bins={self.bins}, sample_rate={self.sample_rate})"


@dataclass(frozen=True)
class PlaneConfig:
    """Every choice that turns magnitudes into a plane, and back again.

    ``None`` means "decide from the data": :meth:`resolved` fills those in, and
    the filled-in copy is what gets written into the picture so the decode can
    reproduce the mapping exactly.
    """

    scale: str = "db"
    top_db: float = 80.0
    ref: float | None = None
    """The magnitude that maps to 1.0 (black).  ``None`` -> the clip's peak."""
    freq_scale: str = "linear"
    f_min: float = 0.0
    f_max: float | None = None
    """``None`` -> the Nyquist frequency."""
    height: int | None = None
    """Rows.  ``None`` -> one per frequency bin (no resampling)."""
    width: int | None = None
    """Columns.  ``None`` -> one per frame (no resampling)."""
    origin: str = "lower"
    """``lower`` puts low frequencies at the bottom of the picture, as a spectrogram is normally drawn."""

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise ValueError(f"unknown scale {self.scale!r}; choose from {', '.join(SCALES)}")
        if self.freq_scale not in FREQ_SCALES:
            raise ValueError(f"unknown freq_scale {self.freq_scale!r}; choose from {', '.join(FREQ_SCALES)}")
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown origin {self.origin!r}; choose from {', '.join(ORIGINS)}")
        if self.top_db <= 0:
            raise ValueError(f"top_db must be > 0, got {self.top_db}")
        if self.height is not None and self.height < 1:
            raise ValueError(f"height must be >= 1, got {self.height}")
        if self.width is not None and self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")
        if self.ref is not None and self.ref <= 0:
            raise ValueError(f"ref must be > 0, got {self.ref}")
        if self.f_min < 0:
            raise ValueError(f"f_min must be >= 0, got {self.f_min}")
        if self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError(f"f_max must be greater than f_min, got {self.f_max} <= {self.f_min}")

    def resolved(self, spec: Spectrogram) -> "PlaneConfig":
        """A copy with every ``None`` replaced by what this spectrogram implies."""
        ref = self.ref
        if ref is None:
            peak = spec.peak
            ref = peak if peak > 0 else 1.0
        f_max = self.f_max if self.f_max is not None else spec.nyquist
        height = self.height if self.height is not None else spec.bins
        width = self.width if self.width is not None else max(1, spec.frames)
        return replace(self, ref=ref, f_max=f_max, height=height, width=width)

    @property
    def exact(self) -> bool:
        """True when no resampling happens on the frequency axis (the invertible case)."""
        return self.freq_scale == "linear" and self.f_min == 0.0

    def to_meta(self) -> dict[str, Any]:
        """The settings, ready to be written into the picture."""
        return {
            "scale": self.scale,
            "top_db": self.top_db,
            "ref": self.ref,
            "freq_scale": self.freq_scale,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "height": self.height,
            "width": self.width,
            "origin": self.origin,
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "PlaneConfig":
        """Rebuild a config from what was written into a picture."""
        known = {f: meta[f] for f in ("scale", "top_db", "ref", "freq_scale", "f_min", "f_max", "height", "width", "origin") if f in meta}
        return cls(**known)


# ---------------------------------------------------------------------------
# amplitude <-> intensity
# ---------------------------------------------------------------------------


def _to_intensity(mag: float, cfg: PlaneConfig) -> float:
    """One magnitude to one number in ``[0, 1]`` (1.0 = black = as loud as it gets)."""
    ref = cfg.ref or 1.0
    if mag <= 0.0:
        return 0.0
    if cfg.scale == "linear":
        t = mag / ref
    elif cfg.scale == "sqrt":
        t = math.sqrt(mag / ref)
    else:  # db
        db = 20.0 * math.log10(mag / ref)
        t = 1.0 + db / cfg.top_db
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def _to_magnitude(t: float, cfg: PlaneConfig) -> float:
    """The inverse of :func:`_to_intensity`.

    Intensity 0 means magnitude 0, not "``top_db`` below the peak": the floor
    of the picture is silence, and decoding it as a faint tone everywhere
    would add a hiss to the whole clip.
    """
    ref = cfg.ref or 1.0
    if t <= 0.0:
        return 0.0
    t = 1.0 if t > 1.0 else t
    if cfg.scale == "linear":
        return t * ref
    if cfg.scale == "sqrt":
        return t * t * ref
    return ref * (10.0 ** (((t - 1.0) * cfg.top_db) / 20.0))


# ---------------------------------------------------------------------------
# the frequency axis
# ---------------------------------------------------------------------------


def row_frequencies(cfg: PlaneConfig, nyquist: float) -> list[float]:
    """The frequency each plane row stands for, lowest first.

    The rows are evenly spaced in whichever domain ``freq_scale`` names - hertz,
    log hertz, or mels - between ``f_min`` and ``f_max``.
    """
    height = cfg.height or 1
    f_min = cfg.f_min
    f_max = cfg.f_max if cfg.f_max is not None else nyquist
    if height == 1:
        return [f_min]
    if cfg.freq_scale == "linear":
        step = (f_max - f_min) / (height - 1)
        return [f_min + step * i for i in range(height)]
    if cfg.freq_scale == "log":
        lo = math.log(max(f_min, MIN_LOG_HZ))
        hi = math.log(max(f_max, math.exp(lo) * 1.000001))
        step = (hi - lo) / (height - 1)
        return [math.exp(lo + step * i) for i in range(height)]
    lo, hi = hz_to_mel(f_min), hz_to_mel(f_max)
    step = (hi - lo) / (height - 1)
    return [mel_to_hz(lo + step * i) for i in range(height)]


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Linear interpolation of a sampled function, clamped at both ends.

    ``xs`` must increase.  A binary search keeps this usable for the ``log``
    and ``mel`` axes, whose sample positions are not evenly spaced.
    """
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1 or x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0:
        return ys[lo]
    f = (x - xs[lo]) / span
    return ys[lo] * (1.0 - f) + ys[hi] * f


def _resample_columns(rows: list[list[float]], width: int) -> list[list[float]]:
    """Stretch or squeeze the time axis to ``width`` columns."""
    src = len(rows[0]) if rows else 0
    if src == width or src == 0:
        return rows
    if width == 1:
        return [[row[0]] for row in rows]
    xs = list(range(src))
    scale = (src - 1) / (width - 1)
    return [[_interp(xs, row, i * scale) for i in range(width)] for row in rows]


# ---------------------------------------------------------------------------
# the plane
# ---------------------------------------------------------------------------


def to_plane(spec: Spectrogram, cfg: PlaneConfig) -> list[list[float]]:
    """Magnitudes -> a plane of intensities, in image row order.

    The result is indexed ``[row][column]``: row 0 is the *top* of the picture,
    which with the default ``origin="lower"`` is the highest frequency.
    """
    cfg = cfg.resolved(spec)
    height = cfg.height or spec.bins
    width = cfg.width or max(1, spec.frames)
    if not spec.mags:
        return [[0.0] * width for _ in range(height)]

    # frequency first: build rows of magnitudes, lowest frequency first
    exact = cfg.exact and height == spec.bins and abs((cfg.f_max or 0.0) - spec.nyquist) < 1e-9
    if exact:
        rows = [[frame[b] for frame in spec.mags] for b in range(spec.bins)]
    else:
        bin_hz = [spec.bin_hz(b) for b in range(spec.bins)]
        targets = row_frequencies(cfg, spec.nyquist)
        columns = [[frame[b] for b in range(spec.bins)] for frame in spec.mags]
        rows = [[_interp(bin_hz, col, f) for col in columns] for f in targets]

    rows = _resample_columns(rows, width)
    plane = [[_to_intensity(v, cfg) for v in row] for row in rows]
    if cfg.origin == "lower":  # row 0 is the top of the picture: highest frequency
        plane.reverse()
    return plane


def from_plane(
    plane: Sequence[Sequence[float]],
    cfg: PlaneConfig,
    sample_rate: int,
    n_fft: int,
    hop: int,
    window: str = "hann",
    center: bool = True,
    frames: int | None = None,
) -> Spectrogram:
    """A plane of intensities -> magnitudes, undoing :func:`to_plane`.

    ``frames`` is how many time slices the original had; the plane is stretched
    back to that many columns when the picture was written at another width.
    """
    rows = [list(r) for r in plane]
    if not rows:
        return Spectrogram([], sample_rate, n_fft, hop, window, center)
    if cfg.origin == "lower":
        rows.reverse()  # back to lowest frequency first

    bins = n_fft // 2 + 1
    nyquist = sample_rate / 2.0
    mags_rows = [[_to_magnitude(v, cfg) for v in row] for row in rows]
    if frames is not None:
        mags_rows = _resample_columns(mags_rows, frames)
    width = len(mags_rows[0]) if mags_rows else 0

    exact = cfg.exact and len(mags_rows) == bins and abs((cfg.f_max or nyquist) - nyquist) < 1e-9
    if exact:
        out = [[mags_rows[b][t] for b in range(bins)] for t in range(width)]
    else:
        row_hz = row_frequencies(cfg, nyquist)
        # the frequency grid must increase for the interpolation to be searchable
        if len(row_hz) != len(mags_rows):
            row_hz = row_hz[: len(mags_rows)] + [nyquist] * max(0, len(mags_rows) - len(row_hz))
        bin_hz = [b * sample_rate / n_fft for b in range(bins)]
        out = []
        for t in range(width):
            column = [row[t] for row in mags_rows]
            out.append([_interp(row_hz, column, f) for f in bin_hz])
    return Spectrogram(out, sample_rate, n_fft, hop, window, center)
