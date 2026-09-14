"""Audio in, picture out, audio back: the two operations the tool exists for.

:func:`encode` runs the chain

    samples -> STFT -> magnitudes -> plane -> pixels -> PNG

and writes the settings it used into the file as a ``tEXt`` chunk, so the
picture is self-describing.  :func:`decode` runs it backwards

    PNG -> pixels -> plane -> magnitudes -> Griffin-Lim -> samples

reading those settings back out.  A picture with no settings in it still
decodes: the caller's defaults are used instead, which is what makes a
spectrogram drawn by hand, or produced by something else entirely, playable.

What survives and what does not
-------------------------------
The magnitudes survive: with the default ``linear`` frequency axis at 16 bits
the plane is the spectrum to about five decimal places.  The **phase** does
not - a picture has one number per point and phase is the second - so decoding
reconstructs it with Griffin-Lim (:func:`audioimage.dsp.griffin_lim`).  That is
the real loss in the round trip, and it is why a decoded clip measured against
the original sample by sample looks poor while sounding close: the spectrum is
right, the waveform underneath it is a different one that has the same
spectrum.  :func:`compare` reports both so the difference is visible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from . import __version__
from .backend import get_backend
from .colormap import get_colormap
from .png import Image, PngError, read_png, write_png
from .spectrogram import PlaneConfig, Spectrogram, from_plane, to_plane
from .wav import Audio, read_wav, write_wav

__all__ = [
    "DecodeResult",
    "EncodeConfig",
    "META_KEY",
    "META_VERSION",
    "compare",
    "decode",
    "decode_file",
    "encode",
    "encode_file",
    "image_to_plane",
    "plane_to_image",
    "read_meta",
]

META_KEY = "audioimage"
"""The ``tEXt`` keyword the settings live under."""
META_VERSION = 1


@dataclass(frozen=True)
class EncodeConfig:
    """Everything :func:`encode` needs, and everything :func:`decode` wants back."""

    n_fft: int = 1024
    """Window length in samples; a power of two.  Bigger = finer frequencies, coarser timing."""
    hop: int = 256
    """Samples between frames.  ``n_fft // 4`` overlaps enough to reconstruct exactly."""
    window: str = "hann"
    center: bool = True
    plane: PlaneConfig = field(default_factory=PlaneConfig)
    colormap: str = "gray"
    depth: int = 16
    """Bits per pixel: 16 keeps 65536 levels of loudness, 8 keeps 256."""

    def __post_init__(self) -> None:
        if self.n_fft < 2 or (self.n_fft & (self.n_fft - 1)):
            raise ValueError(f"n_fft must be a power of two >= 2, got {self.n_fft}")
        if self.hop < 1:
            raise ValueError(f"hop must be >= 1, got {self.hop}")
        if self.hop > self.n_fft:
            raise ValueError(f"hop ({self.hop}) must not exceed n_fft ({self.n_fft})")
        if self.depth not in (8, 16):
            raise ValueError(f"depth must be 8 or 16, got {self.depth}")
        get_colormap(self.colormap)  # validates the name

    @property
    def bins(self) -> int:
        """Frequency bins per frame."""
        return self.n_fft // 2 + 1


@dataclass
class DecodeResult:
    """What came back out of a picture."""

    audio: Audio
    spectrogram: Spectrogram
    meta: dict[str, Any]
    iterations: int = 0
    error: float = 0.0
    """Spectral convergence of the last Griffin-Lim pass: how well the phase guess fits."""
    had_meta: bool = True

    def describe(self) -> dict[str, Any]:
        """A summary for ``info`` / ``--json`` output."""
        return {
            **self.audio.describe(),
            "iterations": self.iterations,
            "griffin_lim_error": round(self.error, 8),
            "had_metadata": self.had_meta,
        }


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def _meta_for(
    cfg: EncodeConfig,
    spec: Spectrogram,
    plane_cfg: PlaneConfig,
    samples: int,
    level_peak: float,
    level_rms: float,
) -> dict[str, Any]:
    return {
        "v": META_VERSION,
        "tool": f"audioimage {__version__}",
        "sample_rate": spec.sample_rate,
        "samples": samples,
        "peak": round(level_peak, 8),
        "rms": round(level_rms, 8),
        "n_fft": cfg.n_fft,
        "hop": cfg.hop,
        "window": cfg.window,
        "center": cfg.center,
        "frames": spec.frames,
        "colormap": cfg.colormap,
        "depth": cfg.depth,
        **plane_cfg.to_meta(),
    }


def read_meta(image: Image) -> dict[str, Any]:
    """The settings stored in a picture, or ``{}`` when it has none.

    A picture from somewhere else has no chunk; one that has been through an
    editor may have a damaged one.  Neither is an error here - the caller falls
    back to its own defaults - so anything unparseable comes back empty.
    """
    raw = image.text.get(META_KEY)
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# plane <-> pixels
# ---------------------------------------------------------------------------


def plane_to_image(plane: Sequence[Sequence[float]], colormap: str = "gray", depth: int = 16) -> Image:
    """Intensities to pixels: ``1.0`` becomes black, ``0.0`` becomes white."""
    cmap = get_colormap(colormap)
    height = len(plane)
    width = len(plane[0]) if height else 0
    if height == 0 or width == 0:
        raise PngError("cannot make a picture out of an empty plane")
    if cmap.mode == "RGB":
        depth = 8
    image = Image(width, height, cmap.mode, depth)
    maxval = image.maxval
    flat = image.data
    if cmap.mode == "L":
        for y, row in enumerate(plane):
            base = y * width
            for x, v in enumerate(row):
                flat[base + x] = cmap.to_value(v, maxval)
    else:
        for y, row in enumerate(plane):
            base = y * width * 3
            for x, v in enumerate(row):
                r, g, b = cmap.to_rgb(v)
                flat[base + x * 3] = r
                flat[base + x * 3 + 1] = g
                flat[base + x * 3 + 2] = b
    return image


def image_to_plane(image: Image, colormap: str = "gray") -> list[list[float]]:
    """Pixels back to intensities."""
    cmap = get_colormap(colormap)
    width, height = image.width, image.height
    if cmap.mode == "L" and image.mode == "L":
        maxval = image.maxval
        flat = image.data
        return [[cmap.from_value(flat[y * width + x], maxval) for x in range(width)] for y in range(height)]
    if cmap.mode == "L":  # a grey ramp saved as RGB: the channels agree, use one
        return [[cmap.from_value(image.get_rgb(x, y)[0], 255) for x in range(width)] for y in range(height)]
    pixels = [image.get_rgb(x, y) for y in range(height) for x in range(width)]
    values = cmap.from_rgb_many(pixels)
    return [values[y * width : (y + 1) * width] for y in range(height)]


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def encode(audio: Audio, cfg: EncodeConfig | None = None, backend: str = "auto") -> Image:
    """A waveform to a picture of its spectrum."""
    cfg = cfg or EncodeConfig()
    dsp = get_backend(backend)
    if not audio.samples:
        raise ValueError("cannot encode an empty clip")
    frames = dsp.stft(audio.samples, cfg.n_fft, cfg.hop, cfg.window, cfg.center)
    spec = Spectrogram(dsp.magnitudes(frames), audio.sample_rate, cfg.n_fft, cfg.hop, cfg.window, cfg.center)
    plane_cfg = cfg.plane.resolved(spec)
    plane = to_plane(spec, plane_cfg)
    image = plane_to_image(plane, cfg.colormap, cfg.depth)
    image.text[META_KEY] = json.dumps(
        _meta_for(cfg, spec, plane_cfg, len(audio.samples), audio.peak, audio.rms),
        separators=(",", ":"),
        sort_keys=True,
    )
    return image


def encode_file(
    wav_path: str | os.PathLike[str],
    png_path: str | os.PathLike[str],
    cfg: EncodeConfig | None = None,
    backend: str = "auto",
) -> tuple[Image, Audio]:
    """Read a WAV, write the picture, and hand both back."""
    audio = read_wav(wav_path)
    image = encode(audio, cfg, backend)
    write_png(png_path, image)
    return image, audio


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def _config_from_meta(meta: dict[str, Any], fallback: EncodeConfig) -> tuple[EncodeConfig, PlaneConfig]:
    """Settings out of a picture, with the caller's defaults filling any gap."""
    n_fft = int(meta.get("n_fft", fallback.n_fft))
    hop = int(meta.get("hop", fallback.hop))
    cfg = EncodeConfig(
        n_fft=n_fft,
        hop=hop,
        window=str(meta.get("window", fallback.window)),
        center=bool(meta.get("center", fallback.center)),
        colormap=str(meta.get("colormap", fallback.colormap)),
        depth=int(meta.get("depth", fallback.depth)),
        plane=fallback.plane,
    )
    plane_cfg = PlaneConfig.from_meta({**fallback.plane.to_meta(), **meta}) if meta else fallback.plane
    return cfg, plane_cfg


def decode(
    image: Image,
    fallback: EncodeConfig | None = None,
    sample_rate: int | None = None,
    iterations: int = 48,
    momentum: float = 0.99,
    seed: int | None = 0,
    init: str = "random",
    backend: str = "auto",
    length: int | None = None,
    level: str = "auto",
    progress: Callable[[int, int, float], None] | None = None,
) -> DecodeResult:
    """A picture of a spectrum back to a waveform.

    Settings come from the picture when it carries them; ``fallback`` and
    ``sample_rate`` cover the case where it does not.  ``iterations`` is how
    many Griffin-Lim passes reconstruct the missing phase - more is closer and
    slower, and the returned :attr:`DecodeResult.error` says how close.

    ``level`` sets how loud the result comes out.  ``auto`` restores the
    loudness the original had, by matching the RMS the encode recorded - RMS,
    not peak, because a waveform rebuilt with new phase has the same energy as
    the original but not the same tallest sample.  ``peak`` normalises to just
    under full scale instead, which is what a picture carrying no level can do,
    and ``none`` leaves the samples as Griffin-Lim produced them.
    """
    if level not in ("auto", "peak", "none"):
        raise ValueError(f"unknown level {level!r}; choose from auto, peak, none")
    fallback = fallback or EncodeConfig()
    meta = read_meta(image)
    had_meta = bool(meta)
    cfg, plane_cfg = _config_from_meta(meta, fallback)
    rate = sample_rate or int(meta.get("sample_rate", 0)) or 22050

    plane = image_to_plane(image, cfg.colormap)
    # a picture with no settings tells us its own shape: one row per bin
    if not had_meta and plane_cfg.height is None:
        guessed = (len(plane) - 1) * 2
        if guessed >= 2 and not (guessed & (guessed - 1)):
            cfg = replace(cfg, n_fft=guessed, hop=max(1, guessed // 4))
    plane_cfg = replace(plane_cfg, height=len(plane), width=len(plane[0]) if plane else 0)

    frames = int(meta.get("frames", 0)) or (len(plane[0]) if plane else 0)
    spec = from_plane(plane, plane_cfg, rate, cfg.n_fft, cfg.hop, cfg.window, cfg.center, frames=frames)
    samples = int(meta.get("samples", 0)) or length or 0

    dsp = get_backend(backend)
    last: list[float] = [0.0]

    def _track(step: int, total: int, error: float) -> None:
        last[0] = error
        if progress is not None:
            progress(step, total, error)

    signal = dsp.griffin_lim(
        spec.mags,
        cfg.n_fft,
        cfg.hop,
        cfg.window,
        cfg.center,
        iterations=iterations,
        momentum=momentum,
        length=samples or None,
        seed=seed,
        init=init,
        progress=_track,
    )
    audio = Audio(signal, rate, source="decoded", meta={"colormap": cfg.colormap})
    want_rms = float(meta.get("rms", 0.0) or 0.0)
    if level == "auto" and want_rms > 0.0 and audio.rms > 0.0:
        gain = want_rms / audio.rms
        audio = Audio([s * gain for s in audio.samples], rate, "decoded", dict(audio.meta))
    elif level in ("auto", "peak") and audio.peak > 0.0:
        audio = audio.normalized()
    return DecodeResult(audio, spec, meta, iterations=iterations, error=last[0], had_meta=had_meta)


def decode_file(
    png_path: str | os.PathLike[str],
    wav_path: str | os.PathLike[str],
    bits: int = 16,
    **kwargs: Any,
) -> DecodeResult:
    """Read a picture, write the WAV it decodes to, and hand the result back."""
    result = decode(read_png(png_path), **kwargs)
    write_wav(wav_path, result.audio, bits=bits)
    return result


# ---------------------------------------------------------------------------
# how good was it?
# ---------------------------------------------------------------------------


def compare(
    reference: Audio, decoded: Audio, cfg: EncodeConfig | None = None, backend: str = "auto"
) -> dict[str, float]:
    """Measure a round trip, on the spectrum and on the waveform.

    ``spectral_convergence`` is the number that matters: the relative error of
    the magnitudes, which is what the picture actually stored.  ``waveform_snr``
    is reported next to it to make the point that it *should* look bad - the
    phase was thrown away and guessed again, so the samples differ even when
    the sound does not.
    """
    cfg = cfg or EncodeConfig()
    dsp = get_backend(backend)
    n = min(len(reference.samples), len(decoded.samples))
    if n == 0:
        return {"spectral_convergence": 0.0, "log_spectral_distance": 0.0, "waveform_snr": 0.0, "samples": 0}
    a = reference.samples[:n]
    b = decoded.samples[:n]
    # match the levels by RMS before measuring.  Peak would be the wrong ruler:
    # rebuilding a waveform from magnitudes alone keeps its energy but not its
    # tallest sample, so peak-matching would score a correct decode as wrong.
    ref_rms = (sum(v * v for v in a) / n) ** 0.5
    got_rms = (sum(v * v for v in b) / n) ** 0.5
    if got_rms > 0 and ref_rms > 0:
        gain = ref_rms / got_rms
        b = [v * gain for v in b]

    sa = dsp.magnitudes(dsp.stft(a, cfg.n_fft, cfg.hop, cfg.window, cfg.center))
    sb = dsp.magnitudes(dsp.stft(b, cfg.n_fft, cfg.hop, cfg.window, cfg.center))
    rows = min(len(sa), len(sb))
    num = 0.0
    den = 0.0
    log_sum = 0.0
    count = 0
    import math

    # the log distance is floored 80 dB under the loudest point, the same window
    # the picture stores.  Without a floor the score is dominated by bins that
    # are silent in both clips, where a difference between 1e-12 and 1e-9 counts
    # as 60 dB of error and nothing can be heard at all.
    peak = max((max(row) for row in sa if row), default=0.0)
    floor = max(peak * (10.0 ** (-80.0 / 20.0)), 1e-12)
    for i in range(rows):
        for x, y in zip(sa[i], sb[i]):
            num += (x - y) ** 2
            den += x * x
            log_sum += (20.0 * math.log10(max(x, floor)) - 20.0 * math.log10(max(y, floor))) ** 2
            count += 1
    noise = sum((x - y) ** 2 for x, y in zip(a, b))
    signal = sum(x * x for x in a)
    return {
        "spectral_convergence": (num**0.5) / (den**0.5) if den > 0 else 0.0,
        "log_spectral_distance": (log_sum / count) ** 0.5 if count else 0.0,
        "waveform_snr": 10.0 * math.log10(signal / noise) if noise > 0 and signal > 0 else float("inf"),
        "samples": n,
    }
