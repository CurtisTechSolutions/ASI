"""Pictures for looking at, as opposed to pictures for decoding.

:func:`encode` writes a *bare* plane: every pixel is data, nothing else is in
the file, which is what lets it be read back exactly.  That is not much help
when the question is "what does this clip look like".  This module wraps the
same plane in the furniture that makes it readable - a labelled frequency axis
up the side, a time axis along the bottom, a title, and a legend showing which
shade means how loud - and can draw the waveform itself as well.

Nothing here is reversible, and nothing here is meant to be: a view is an
illustration.  Decode the picture that :func:`encode` wrote, not a view of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .colormap import get_colormap
from .font import CHAR_HEIGHT, CHAR_WIDTH, SPACING, glyph, text_width
from .png import Image
from .spectrogram import PlaneConfig, row_frequencies
from .wav import Audio

__all__ = ["Canvas", "ViewConfig", "format_hz", "format_time", "nice_ticks", "render_waveform", "render_view"]

RGB = tuple[int, int, int]


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


class Canvas:
    """An RGB image with just enough drawing to label a chart."""

    __slots__ = ("image",)

    def __init__(self, width: int, height: int, background: RGB = (255, 255, 255)) -> None:
        self.image = Image(width, height, "RGB", 8)
        self.fill_rect(0, 0, width, height, background)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def pixel(self, x: int, y: int, rgb: RGB) -> None:
        """Set one pixel, ignoring anything outside the canvas."""
        if 0 <= x < self.image.width and 0 <= y < self.image.height:
            self.image.set_rgb(x, y, rgb)

    def fill_rect(self, x: int, y: int, w: int, h: int, rgb: RGB) -> None:
        """A solid rectangle."""
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.pixel(xx, yy, rgb)

    def hline(self, x: int, y: int, length: int, rgb: RGB) -> None:
        """A horizontal run of pixels."""
        for xx in range(x, x + length):
            self.pixel(xx, y, rgb)

    def vline(self, x: int, y: int, length: int, rgb: RGB) -> None:
        """A vertical run of pixels."""
        for yy in range(y, y + length):
            self.pixel(x, yy, rgb)

    def text(self, x: int, y: int, message: str, rgb: RGB = (0, 0, 0), scale: int = 1) -> int:
        """Draw ``message`` with its top-left at ``x, y``.

        Returns the width of the ink - the same number :func:`text_width`
        gives, not the cursor advance, so centring a label on one and
        right-aligning it with the other put it in the same place.
        """
        cx = x
        for char in message:
            bits = glyph(char)
            for col in range(CHAR_WIDTH):
                column = bits[col]
                for row in range(CHAR_HEIGHT):
                    if column >> row & 1:
                        if scale == 1:
                            self.pixel(cx + col, y + row, rgb)
                        else:
                            self.fill_rect(cx + col * scale, y + row * scale, scale, scale, rgb)
            cx += (CHAR_WIDTH + SPACING) * scale
        return text_width(message, scale)

    def text_right(self, right: int, y: int, message: str, rgb: RGB = (0, 0, 0), scale: int = 1) -> None:
        """Draw ``message`` ending at ``right``."""
        self.text(right - text_width(message, scale), y, message, rgb, scale)


# ---------------------------------------------------------------------------
# axis labels
# ---------------------------------------------------------------------------


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round numbers spanning ``lo``..``hi`` - the 1 / 2 / 5 sequence."""
    if hi <= lo or target < 2:
        return [lo]
    raw = (hi - lo) / (target - 1)
    power = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= power * mult:
            step = power * mult
            break
    else:  # pragma: no cover - the loop always breaks at 10
        step = power * 10.0
    start = math.ceil(lo / step) * step
    out: list[float] = []
    v = start
    while v <= hi + step * 1e-6:
        out.append(round(v, 10))
        v += step
    return out or [lo, hi]


def format_hz(hz: float) -> str:
    """``8000.0`` -> ``"8k"``, ``440.0`` -> ``"440"``."""
    if hz >= 1000:
        text = f"{hz / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"
    if hz >= 10 or hz == 0:
        return f"{hz:.0f}"
    return f"{hz:.1f}".rstrip("0").rstrip(".")


def format_time(seconds: float) -> str:
    """``1.5`` -> ``"1.5s"``, ``0.25`` -> ``"0.25s"``."""
    if seconds >= 10:
        return f"{seconds:.0f}s"
    if seconds >= 1:
        return f"{seconds:.1f}".rstrip("0").rstrip(".") + "s"
    return f"{seconds:.2f}".rstrip("0").rstrip(".") + "s"


def _row_for_hz(row_hz: Sequence[float], hz: float) -> float:
    """Which (fractional) plane row a frequency sits on, lowest row first."""
    n = len(row_hz)
    if n < 2:
        return 0.0
    if hz <= row_hz[0]:
        return 0.0
    if hz >= row_hz[-1]:
        return float(n - 1)
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if row_hz[mid] <= hz:
            lo = mid
        else:
            hi = mid
    span = row_hz[hi] - row_hz[lo]
    return lo + ((hz - row_hz[lo]) / span if span > 0 else 0.0)


# ---------------------------------------------------------------------------
# the view
# ---------------------------------------------------------------------------


@dataclass
class ViewConfig:
    """How a view is laid out."""

    colormap: str = "magma"
    title: str = ""
    axes: bool = True
    legend: bool = True
    scale: int = 1
    """Pixel zoom of the plane itself: 2 doubles every column and row."""
    width: int | None = None
    """Stretch the plane to this many columns before drawing (the view only)."""
    height: int | None = None
    background: RGB = (255, 255, 255)
    ink: RGB = (32, 32, 32)
    grid: RGB = (210, 210, 210)
    margins: tuple[int, int, int, int] = field(default=(58, 16, 26, 60))
    """Left, top, bottom, right - room for the labels, the title and the legend."""

    def __post_init__(self) -> None:
        if self.scale < 1:
            raise ValueError(f"scale must be >= 1, got {self.scale}")
        get_colormap(self.colormap)


def _stretch(plane: Sequence[Sequence[float]], width: int | None, height: int | None) -> list[list[float]]:
    """Nearest-neighbour resize, for fitting a view to a requested size."""
    rows = [list(r) for r in plane]
    if not rows:
        return rows
    if height and height != len(rows):
        src = len(rows)
        rows = [rows[min(src - 1, int(i * src / height))] for i in range(height)]
    if width and rows and width != len(rows[0]):
        src = len(rows[0])
        rows = [[r[min(src - 1, int(i * src / width))] for i in range(width)] for r in rows]
    return rows


def render_view(
    plane: Sequence[Sequence[float]],
    sample_rate: int,
    hop: int,
    cfg: PlaneConfig,
    view: ViewConfig | None = None,
) -> Image:
    """Draw a plane as a labelled chart.

    ``plane`` is in image row order (row 0 at the top), exactly as
    :func:`audioimage.spectrogram.to_plane` returns it.
    """
    view = view or ViewConfig()
    cmap = get_colormap(view.colormap)
    rows = _stretch(plane, view.width, view.height)
    if not rows or not rows[0]:
        raise ValueError("cannot draw an empty plane")
    ph, pw = len(rows), len(rows[0])
    zoom = view.scale
    plot_w, plot_h = pw * zoom, ph * zoom

    left, top, bottom, right = view.margins if view.axes else (1, 1, 1, 1)
    if not view.legend:
        right = 1
    if view.title and not view.axes:
        top = max(top, CHAR_HEIGHT + 6)

    canvas = Canvas(left + plot_w + right, top + plot_h + bottom, view.background)

    # the plane itself
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            rgb = cmap.to_rgb(value)
            if zoom == 1:
                canvas.pixel(left + x, top + y, rgb)
            else:
                canvas.fill_rect(left + x * zoom, top + y * zoom, zoom, zoom, rgb)

    if view.title:
        canvas.text(left, max(0, top - CHAR_HEIGHT - 5), view.title[:120], view.ink)

    if view.axes:
        canvas.hline(left - 1, top + plot_h, plot_w + 2, view.ink)
        canvas.vline(left - 1, top, plot_h + 1, view.ink)

        # frequency up the side.  The plane is drawn top-down, so a row's
        # distance from the bottom is what a frequency label points at.
        resolved = cfg
        nyquist = sample_rate / 2.0
        row_hz = row_frequencies(resolved, nyquist)
        if resolved.origin != "lower":
            row_hz = list(reversed(row_hz))
        f_lo, f_hi = row_hz[0], row_hz[-1]
        ticks = nice_ticks(f_lo, f_hi, 7) if resolved.freq_scale == "linear" else _log_ticks(f_lo, f_hi)
        drawn: list[int] = []
        for hz in ticks:
            r = _row_for_hz(row_hz, hz)
            y = top + plot_h - 1 - int(round(r * (plot_h - 1) / max(1, ph - 1)))
            if not (top <= y <= top + plot_h):
                continue
            # a log or mel axis bunches its round numbers together at one end;
            # drawing all of them there would overprint one label on the next
            if any(abs(y - other) < CHAR_HEIGHT + 2 for other in drawn):
                continue
            drawn.append(y)
            canvas.hline(left - 4, y, 3, view.ink)
            canvas.text_right(left - 6, y - CHAR_HEIGHT // 2, format_hz(hz), view.ink)
        canvas.text(2, top - CHAR_HEIGHT - 5, "Hz", view.ink)

        # time along the bottom
        duration = pw * hop / float(sample_rate) if sample_rate else 0.0
        placed: list[tuple[int, int]] = []
        for seconds in nice_ticks(0.0, duration, 7):
            x = left + int(round(seconds / duration * (plot_w - 1))) if duration > 0 else left
            if not (left <= x <= left + plot_w):
                continue
            label = format_time(seconds)
            half = text_width(label) // 2
            if any(x - half < prev_right + 4 and x + half > prev_left - 4 for prev_left, prev_right in placed):
                continue
            placed.append((x - half, x + half))
            canvas.vline(x, top + plot_h + 1, 3, view.ink)
            canvas.text(x - half, top + plot_h + 6, label, view.ink)

    if view.legend:
        _draw_legend(canvas, left + plot_w + 14, top, plot_h, cfg, view)
    return canvas.image


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Ticks for a log or mel axis: 1, 2, 5 at every power of ten."""
    lo = max(lo, 1.0)
    out: list[float] = []
    power = 10.0 ** math.floor(math.log10(lo))
    while power <= hi:
        for mult in (1.0, 2.0, 5.0):
            v = power * mult
            if lo <= v <= hi:
                out.append(v)
        power *= 10.0
    return out or [lo, hi]


def _draw_legend(canvas: Canvas, x: int, top: int, height: int, cfg: PlaneConfig, view: ViewConfig) -> None:
    """The ramp, with what its ends mean."""
    cmap = get_colormap(view.colormap)
    bar_w = 10
    for i in range(height):
        value = 1.0 - i / max(1, height - 1)  # loudest at the top
        canvas.fill_rect(x, top + i, bar_w, 1, cmap.to_rgb(value))
    canvas.vline(x - 1, top, height, view.ink)
    canvas.vline(x + bar_w, top, height, view.ink)
    canvas.hline(x - 1, top - 1, bar_w + 2, view.ink)
    canvas.hline(x - 1, top + height, bar_w + 2, view.ink)
    if cfg.scale == "db":
        hi, lo = "0 dB", f"-{cfg.top_db:.0f}"
    else:
        hi, lo = "1.0", "0.0"
    canvas.text(x + bar_w + 3, top, hi, view.ink)
    canvas.text(x + bar_w + 3, top + height - CHAR_HEIGHT, lo, view.ink)


# ---------------------------------------------------------------------------
# the waveform
# ---------------------------------------------------------------------------


def render_waveform(
    audio: Audio,
    width: int = 900,
    height: int = 220,
    view: ViewConfig | None = None,
) -> Image:
    """Draw the samples themselves: the loudness envelope over time.

    Each column covers a slice of the clip and is drawn from its quietest
    sample to its loudest, which is how an audio editor draws a waveform.
    """
    view = view or ViewConfig(colormap="gray", legend=False)
    if width < 16 or height < 16:
        raise ValueError(f"a waveform view must be at least 16x16, got {width}x{height}")
    left, top, bottom, right = view.margins if view.axes else (1, 1, 1, 1)
    right = 8
    plot_w, plot_h = width, height
    canvas = Canvas(left + plot_w + right, top + plot_h + bottom, view.background)
    mid = top + plot_h // 2
    samples = audio.samples
    n = len(samples)

    canvas.hline(left, mid, plot_w, (200, 200, 200))
    ink = view.ink
    if n:
        per = n / plot_w
        for x in range(plot_w):
            a = int(x * per)
            b = max(a + 1, int((x + 1) * per))
            chunk = samples[a:b] or [0.0]
            lo, hi = min(chunk), max(chunk)
            y0 = mid - int(hi * (plot_h // 2 - 1))
            y1 = mid - int(lo * (plot_h // 2 - 1))
            if y1 < y0:
                y0, y1 = y1, y0
            canvas.vline(left + x, y0, max(1, y1 - y0 + 1), ink)

    if view.axes:
        canvas.hline(left - 1, top + plot_h, plot_w + 2, ink)
        canvas.vline(left - 1, top, plot_h + 1, ink)
        for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
            y = mid - int(value * (plot_h // 2 - 1))
            canvas.hline(left - 4, y, 3, ink)
            canvas.text_right(left - 6, y - CHAR_HEIGHT // 2, f"{value:+.1f}", ink)
        duration = audio.duration
        for seconds in nice_ticks(0.0, duration, 7):
            x = left + int(round(seconds / duration * (plot_w - 1))) if duration > 0 else left
            canvas.vline(x, top + plot_h + 1, 3, ink)
            label = format_time(seconds)
            canvas.text(x - text_width(label) // 2, top + plot_h + 6, label, ink)
    if view.title:
        canvas.text(left, max(0, top - CHAR_HEIGHT - 5), view.title[:120], ink)
    return canvas.image
