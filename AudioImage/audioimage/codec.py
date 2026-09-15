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
the plane is the spectrum to about five decimal places.  The **phase** is the
question, because a grey picture has one number per point and phase is the
second one.

``phase="none"`` (the default) stores only the magnitudes and rebuilds the
phase on the way out with Griffin-Lim (:func:`audioimage.dsp.griffin_lim`).
That is the whole of the loss in the round trip, and it is why a decoded clip
measured against the original sample by sample scores terribly while sounding
close: the spectrum is right, the waveform under it is a different one with the
same spectrum.

``phase="rgb"`` stores it instead, in the blue channel::

    R = G = magnitude     the picture stays grey, and stays readable
    B     = phase         only where there is energy to have a phase

and decoding then needs no guessing at all: the spectrum goes straight back
through the inverse transform.  Measured on a melody, against the same clip
through the grey format:

=========================  =============  ==================
                           ``none``       ``rgb``
=========================  =============  ==================
spectral error             2.51%          **0.45%**
waveform SNR               -2.8 dB        **+36.9 dB**
decode time                0.35 s         **0.11 s**
file size                  19.3 KiB       18.3 KiB
=========================  =============  ==================

The file does not grow because the masking pays for itself: phase is noise, and
noise does not compress, so it is written only where the magnitude is within
``phase_floor`` dB of the peak - 5.7% of the pixels on that clip.  Everywhere
else the blue channel simply repeats the grey, which is also what keeps the
picture legible: unmasked phase is confetti across the whole frame.

Noise is the case that does grow, and there is no trick for it - noise *is*
phase, so its picture has to carry phase everywhere.

:func:`compare` reports both the spectral and the waveform numbers so the
difference between the two modes is visible rather than described.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from . import __version__
from .backend import get_backend
from .colormap import GRAY, get_colormap
from .png import Image, PngError, read_png, write_png
from .spectrogram import PlaneConfig, Spectrogram, from_plane, phases_to_plane, plane_to_phases, to_plane
from .wav import Audio, read_wav, write_wav

__all__ = [
    "DecodeResult",
    "EncodeConfig",
    "META_KEY",
    "META_VERSION",
    "PHASE_MODES",
    "pack_phase",
    "unpack_phase",
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
META_VERSION = 2
"""Version 2 added ``phase``.  A version 1 picture is still read: it simply has none."""

PHASE_MODES = ("none", "rgb")


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
    """Bits per channel: 16 keeps 65536 levels of loudness, 8 keeps 256."""
    phase: str = "none"
    """``none`` rebuilds the phase when decoding; ``rgb`` stores it in the blue channel."""
    lossless: bool = False
    """Add a correction channel so the decode is bit-for-bit the original.

    The picture already gets within a handful of least-significant bits once it
    carries a phase.  ``lossless`` closes the gap the honest way: the encoder
    decodes its own picture, subtracts the result from the source, and writes
    that difference into the alpha channel.  Decoding adds it back, so what
    comes out is what went in, sample for sample, at ``lossless_bits``.

    It costs whatever the difference costs to store, which is why it is worth
    pairing with ``phase="rgb"`` - the better the picture, the smaller the
    correction and the smaller the file.
    """
    lossless_bits: int = 16
    """The sample resolution the correction makes exact (16 by default, as WAVs are)."""
    phase_floor: float = 60.0
    """With ``phase="rgb"``, how far under the peak still gets a phase written, in dB.

    Below this the blue channel repeats the grey instead.  It is what keeps the
    file small and the picture readable, and it costs nothing audible: a point
    60 dB down contributes a thousandth of the amplitude, so whatever phase it
    is given cannot be heard.
    """

    def __post_init__(self) -> None:
        if self.n_fft < 2 or (self.n_fft & (self.n_fft - 1)):
            raise ValueError(f"n_fft must be a power of two >= 2, got {self.n_fft}")
        if self.hop < 1:
            raise ValueError(f"hop must be >= 1, got {self.hop}")
        if self.hop > self.n_fft:
            raise ValueError(f"hop ({self.hop}) must not exceed n_fft ({self.n_fft})")
        if self.depth not in (8, 16):
            raise ValueError(f"depth must be 8 or 16, got {self.depth}")
        if self.phase not in PHASE_MODES:
            raise ValueError(f"phase must be one of {', '.join(PHASE_MODES)}, got {self.phase!r}")
        if self.phase_floor <= 0:
            raise ValueError(f"phase_floor must be > 0, got {self.phase_floor}")
        if self.lossless_bits not in (8, 16, 24, 32):
            raise ValueError(f"lossless_bits must be 8, 16, 24 or 32, got {self.lossless_bits}")
        get_colormap(self.colormap)  # validates the name
        if self.phase != "none":
            # the blue channel is the phase, so the other two have to be the
            # magnitude; and an angle cannot be interpolated, so the plane has
            # to be the layout that needs no interpolation
            if self.colormap != "gray":
                raise ValueError(f"phase={self.phase!r} needs colormap 'gray', not {self.colormap!r}")
            if not self.plane.exact:
                raise ValueError(
                    f"phase={self.phase!r} needs the exactly-invertible plane: "
                    f"freq_scale 'linear' and f_min 0, not {self.plane.freq_scale!r} from {self.plane.f_min}"
                )
            if self.plane.height is not None and self.plane.height != self.bins:
                raise ValueError(f"phase={self.phase!r} needs height {self.bins} (one row per bin), not {self.plane.height}")

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
    stored_phase: bool = False
    """True when the picture carried its own phase, so none had to be guessed."""
    lossless: bool = False
    """True when a correction channel was applied, so the samples are the originals."""

    def describe(self) -> dict[str, Any]:
        """A summary for ``info`` / ``--json`` output."""
        return {
            **self.audio.describe(),
            "iterations": self.iterations,
            "griffin_lim_error": round(self.error, 8),
            "had_metadata": self.had_meta,
            "stored_phase": self.stored_phase,
            "lossless": self.lossless,
        }


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def _quantise(samples: Sequence[float], bits: int) -> list[int]:
    """Samples to the integers a WAV of that depth would hold."""
    top = (1 << (bits - 1)) - 1
    low = -(1 << (bits - 1))
    return [low if v < low else (top if v > top else v) for v in (int(round(s * top)) for s in samples)]


def _dequantise(values: Sequence[int], bits: int) -> list[float]:
    """The inverse of :func:`_quantise`."""
    top = float((1 << (bits - 1)) - 1)
    return [v / top for v in values]


def _reconstruct(
    plane: Sequence[Sequence[float]],
    phase_rows: Sequence[Sequence[float]],
    plane_cfg: PlaneConfig,
    cfg: EncodeConfig,
    rate: int,
    frames: int,
    samples: int,
    dsp: Any,
) -> list[float]:
    """The base waveform a picture decodes to, before any correction.

    Both sides call this, and they have to agree exactly: the encoder subtracts
    its result from the source to make the correction, and the decoder adds the
    correction back to its result.  One shared function is what guarantees they
    are undoing the same arithmetic.
    """
    spec = from_plane(plane, plane_cfg, rate, cfg.n_fft, cfg.hop, cfg.window, cfg.center, frames=frames)
    angles = plane_to_phases(phase_rows, plane_cfg)
    complex_frames = [
        [complex(m * math.cos(p), m * math.sin(p)) for m, p in zip(mags, phs)]
        for mags, phs in zip(spec.mags, angles)
    ]
    return dsp.istft(complex_frames, cfg.n_fft, cfg.hop, cfg.window, cfg.center, samples or None)


def _pack_residual(image: Image, residual: Sequence[int]) -> None:
    """Write the correction into the alpha channel, one value per pixel."""
    capacity = image.width * image.height
    if len(residual) > capacity:
        raise ValueError(
            f"the correction needs {len(residual)} pixels but the picture has {capacity}; "
            f"use a smaller hop so the picture has more columns"
        )
    half = (image.maxval + 1) // 2
    limit = image.maxval - half
    flat = image.data
    for i, value in enumerate(residual):
        if not -half <= value <= limit:
            raise ValueError(
                f"the correction at sample {i} is {value}, which does not fit in {image.depth} bits; "
                f"store a phase (phase='rgb') so the picture starts closer to the original"
            )
        flat[i * 4 + 3] = value + half
    for i in range(len(residual), capacity):
        flat[i * 4 + 3] = half  # nothing to correct here


def _unpack_residual(image: Image, count: int) -> list[int]:
    """Read the correction back out of the alpha channel."""
    half = (image.maxval + 1) // 2
    flat = image.data
    return [flat[i * 4 + 3] - half for i in range(min(count, image.width * image.height))]


def _floor_intensity(cfg: EncodeConfig, plane_cfg: PlaneConfig) -> float:
    """The intensity ``phase_floor`` dB under the peak, in this plane's scale."""
    from .spectrogram import _to_intensity

    ref = plane_cfg.ref or 1.0
    return _to_intensity(ref * (10.0 ** (-cfg.phase_floor / 20.0)), plane_cfg)


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
        "phase": cfg.phase,
        "phase_floor": cfg.phase_floor,
        "lossless": cfg.lossless,
        "lossless_bits": cfg.lossless_bits,
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


def pack_phase(
    plane: Sequence[Sequence[float]],
    phases: Sequence[Sequence[float]],
    depth: int = 16,
    floor: float = 0.0,
) -> Image:
    """Magnitude and phase into one picture: ``R = G`` is the level, ``B`` is the angle.

    Where the level is below ``floor`` the blue channel repeats the grey rather
    than carrying an angle.  Two things come of that: the file stays small
    (phase is noise, and noise does not compress), and the picture stays
    readable (unmasked phase speckles the whole frame).  Nothing audible is
    lost - a point that far down cannot be heard whatever its phase.
    """
    height = len(plane)
    width = len(plane[0]) if height else 0
    if height == 0 or width == 0:
        raise PngError("cannot make a picture out of an empty plane")
    if len(phases) != height or (height and len(phases[0]) != width):
        raise PngError("the phase plane is not the same shape as the magnitude plane")
    image = Image(width, height, "RGB", depth)
    maxval = image.maxval
    steps = maxval + 1
    flat = image.data
    scale = steps / (2.0 * math.pi)
    for y in range(height):
        row = plane[y]
        prow = phases[y]
        base = y * width * 3
        for x in range(width):
            value = GRAY.to_value(row[x], maxval)
            if row[x] < floor:
                blue = value
            else:
                # wrap rather than clamp: +pi and -pi are the same angle, so the
                # last bucket has to meet the first one
                blue = int(round((prow[x] + math.pi) * scale)) % steps
            i = base + x * 3
            flat[i] = value
            flat[i + 1] = value
            flat[i + 2] = blue
    return image


def unpack_phase(image: Image) -> "tuple[list[list[float]], list[list[float]]]":
    """The inverse of :func:`pack_phase`: ``(intensities, phases in radians)``."""
    if image.mode not in ("RGB", "RGBA"):
        raise PngError(f"a picture carrying phase must be RGB or RGBA, not {image.mode}")
    width, height = image.width, image.height
    maxval = image.maxval
    steps = maxval + 1
    flat = image.data
    stride = image.channels  # 4 when a correction channel rides along
    scale = (2.0 * math.pi) / steps
    plane: list[list[float]] = []
    phases: list[list[float]] = []
    for y in range(height):
        base = y * width * stride
        prow = [0.0] * width
        frow = [0.0] * width
        for x in range(width):
            i = base + x * stride
            prow[x] = GRAY.from_value(flat[i], maxval)
            frow[x] = flat[i + 2] * scale - math.pi
        plane.append(prow)
        phases.append(frow)
    return plane, phases


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
    if cfg.phase == "none":
        if cfg.lossless:
            raise ValueError("lossless needs phase='rgb': without a phase the correction is as big as the sound")
        image = plane_to_image(plane, cfg.colormap, cfg.depth)
    else:
        if plane_cfg.width != spec.frames:
            raise ValueError(f"phase={cfg.phase!r} needs one column per frame ({spec.frames}), not {plane_cfg.width}")
        angles = [[math.atan2(v.imag, v.real) for v in row] for row in frames]
        phase_rows = phases_to_plane(angles, plane_cfg)
        image = pack_phase(plane, phase_rows, cfg.depth, floor=_floor_intensity(cfg, plane_cfg))
        if cfg.lossless:
            image = _add_correction(image, plane, phase_rows, plane_cfg, cfg, audio, spec, dsp)
    image.text[META_KEY] = json.dumps(
        _meta_for(cfg, spec, plane_cfg, len(audio.samples), audio.peak, audio.rms),
        separators=(",", ":"),
        sort_keys=True,
    )
    return image


def _add_correction(
    image: Image,
    plane: Sequence[Sequence[float]],
    phase_rows: Sequence[Sequence[float]],
    plane_cfg: PlaneConfig,
    cfg: EncodeConfig,
    audio: Audio,
    spec: Spectrogram,
    dsp: Any,
) -> Image:
    """Re-read the picture, see how far off it is, and write the difference into alpha."""
    read_plane, read_phase = unpack_phase(image)
    base = _reconstruct(read_plane, read_phase, plane_cfg, cfg, audio.sample_rate, spec.frames, len(audio.samples), dsp)
    want = _quantise(audio.samples, cfg.lossless_bits)
    got = _quantise(base, cfg.lossless_bits)
    residual = [want[i] - (got[i] if i < len(got) else 0) for i in range(len(want))]

    out = Image(image.width, image.height, "RGBA", image.depth, text=dict(image.text))
    for i in range(image.width * image.height):
        out.data[i * 4] = image.data[i * 3]
        out.data[i * 4 + 1] = image.data[i * 3 + 1]
        out.data[i * 4 + 2] = image.data[i * 3 + 2]
    _pack_residual(out, residual)
    return out


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

    # a picture that carries its phase needs no guessing: the spectrum goes
    # straight back through the inverse transform
    stored_phase = str(meta.get("phase", "none")) == "rgb" and image.mode in ("RGB", "RGBA")
    exact = False
    if stored_phase:
        plane, phase_rows = unpack_phase(image)
    else:
        plane = image_to_plane(image, cfg.colormap)
        phase_rows = []
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
    used_iterations = iterations

    def _track(step: int, total: int, error: float) -> None:
        last[0] = error
        if progress is not None:
            progress(step, total, error)

    if stored_phase:
        signal = _reconstruct(
            plane, phase_rows, plane_cfg, cfg, rate, frames, samples, dsp
        )
        used_iterations = 0
        # a picture with a correction channel says exactly what it should have
        # been; adding it back is what makes the round trip bit-for-bit
        if bool(meta.get("lossless")) and image.mode == "RGBA" and samples:
            bits = int(meta.get("lossless_bits", 16))
            corrected = _quantise(signal, bits)
            for i, fix in enumerate(_unpack_residual(image, samples)):
                if i < len(corrected):
                    corrected[i] += fix
            signal = _dequantise(corrected, bits)
            exact = True
        if progress is not None:
            progress(1, 1, 0.0)
    else:
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
    audio = Audio(signal, rate, source="decoded", meta={"colormap": cfg.colormap, "phase": "stored" if stored_phase else "rebuilt"})
    want_rms = float(meta.get("rms", 0.0) or 0.0)
    if exact:
        pass  # these samples are the originals; scaling them would undo that
    elif level == "auto" and want_rms > 0.0 and audio.rms > 0.0:
        gain = want_rms / audio.rms
        audio = Audio([s * gain for s in audio.samples], rate, "decoded", dict(audio.meta))
    elif level in ("auto", "peak") and audio.peak > 0.0:
        audio = audio.normalized()
    return DecodeResult(
        audio, spec, meta, iterations=used_iterations, error=last[0], had_meta=had_meta,
        stored_phase=stored_phase, lossless=exact,
    )


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
