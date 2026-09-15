"""Making sound to look at: tones, sweeps, noise, chords and little tunes.

The codec needs signals with *known* spectra to be tested against - a 440 Hz
sine has to come out as one horizontal line at 440 Hz, and a sweep has to come
out as a diagonal - and anyone trying the tool needs something to encode
before they have a WAV to hand.  Both come from here.

Everything returns an :class:`audioimage.wav.Audio`, so the output of any of
these goes straight into :func:`audioimage.codec.encode`.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .wav import Audio

__all__ = [
    "KINDS",
    "NOTE_NAMES",
    "adsr",
    "chirp",
    "chord",
    "fade",
    "generate",
    "melody",
    "mix",
    "noise",
    "note_to_hz",
    "sine",
]

KINDS = ("sine", "chirp", "noise", "chord", "melody", "harmonics")
DEFAULT_RATE = 22050
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLATS = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}


def note_to_hz(name: str, a4: float = 440.0) -> float:
    """``"A4"`` -> 440.0.  Sharps (``C#4``) and flats (``Bb3``) both work."""
    text = name.strip().upper()
    if not text:
        raise ValueError("empty note name")
    octave = 4
    i = len(text)
    while i > 0 and (text[i - 1].isdigit() or (text[i - 1] == "-" and i > 1)):
        i -= 1
    if i < len(text):
        octave = int(text[i:])
        text = text[:i]
    text = _FLATS.get(text, text)
    if text not in NOTE_NAMES:
        raise ValueError(f"unknown note {name!r}")
    semitones = NOTE_NAMES.index(text) - NOTE_NAMES.index("A") + (octave - 4) * 12
    return a4 * (2.0 ** (semitones / 12.0))


def _count(duration: float, rate: int) -> int:
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if rate < 1:
        raise ValueError(f"rate must be >= 1, got {rate}")
    return max(1, int(round(duration * rate)))


def sine(freq: float = 440.0, duration: float = 1.0, rate: int = DEFAULT_RATE, amplitude: float = 0.6, phase: float = 0.0) -> Audio:
    """A pure tone: one horizontal line in the picture."""
    n = _count(duration, rate)
    step = 2.0 * math.pi * freq / rate
    return Audio([amplitude * math.sin(step * t + phase) for t in range(n)], rate, "synth:sine")


def harmonics(
    freq: float = 220.0,
    duration: float = 1.0,
    rate: int = DEFAULT_RATE,
    count: int = 6,
    amplitude: float = 0.6,
    falloff: float = 1.0,
) -> Audio:
    """A fundamental and its overtones: a stack of evenly spaced lines."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    n = _count(duration, rate)
    gains = [1.0 / ((h + 1) ** falloff) for h in range(count)]
    total = sum(gains) or 1.0
    gains = [g * amplitude / total for g in gains]
    out = [0.0] * n
    for h, gain in enumerate(gains):
        step = 2.0 * math.pi * freq * (h + 1) / rate
        for t in range(n):
            out[t] += gain * math.sin(step * t)
    return Audio(out, rate, "synth:harmonics")


def chirp(
    f0: float = 100.0,
    f1: float = 4000.0,
    duration: float = 2.0,
    rate: int = DEFAULT_RATE,
    amplitude: float = 0.6,
    method: str = "linear",
) -> Audio:
    """A sweep from ``f0`` to ``f1``: a diagonal line, and the best test there is.

    A sweep exercises every frequency and every moment, so a mistake anywhere
    in the mapping - a flipped axis, an off-by-one row, a wrong sample rate -
    shows up as a line that bends, mirrors or stops in the wrong place.
    """
    n = _count(duration, rate)
    out = [0.0] * n
    phase = 0.0
    if method not in ("linear", "log"):
        raise ValueError(f"unknown chirp method {method!r}; choose from linear, log")
    if method == "log" and (f0 <= 0 or f1 <= 0):
        raise ValueError("a logarithmic sweep needs both frequencies above 0")
    for t in range(n):
        frac = t / n
        f = f0 + (f1 - f0) * frac if method == "linear" else f0 * ((f1 / f0) ** frac)
        phase += 2.0 * math.pi * f / rate
        out[t] = amplitude * math.sin(phase)
    return Audio(out, rate, "synth:chirp")


def noise(duration: float = 1.0, rate: int = DEFAULT_RATE, amplitude: float = 0.4, kind: str = "white", seed: int | None = 0) -> Audio:
    """Noise: white fills the picture evenly, pink fades towards the top."""
    n = _count(duration, rate)
    rng = random.Random(seed)
    if kind == "white":
        return Audio([amplitude * rng.uniform(-1.0, 1.0) for _ in range(n)], rate, "synth:noise")
    if kind != "pink":
        raise ValueError(f"unknown noise kind {kind!r}; choose from white, pink")
    # Voss-McCartney: a handful of random sources, each updating half as often
    rows = 12
    values = [rng.uniform(-1.0, 1.0) for _ in range(rows)]
    total = sum(values)
    out = [0.0] * n
    for t in range(n):
        changed = t ^ (t - 1) if t else 1
        for r in range(rows):
            if changed & (1 << r):
                total -= values[r]
                values[r] = rng.uniform(-1.0, 1.0)
                total += values[r]
        out[t] = total / rows
    peak = max((abs(v) for v in out), default=0.0) or 1.0
    return Audio([v * amplitude / peak for v in out], rate, "synth:noise")


def chord(
    notes: Sequence[str | float] = ("C4", "E4", "G4"),
    duration: float = 2.0,
    rate: int = DEFAULT_RATE,
    amplitude: float = 0.6,
) -> Audio:
    """Several notes at once: parallel lines, one per note."""
    freqs = [note_to_hz(n) if isinstance(n, str) else float(n) for n in notes]
    if not freqs:
        raise ValueError("a chord needs at least one note")
    n = _count(duration, rate)
    gain = amplitude / len(freqs)
    out = [0.0] * n
    for f in freqs:
        step = 2.0 * math.pi * f / rate
        for t in range(n):
            out[t] += gain * math.sin(step * t)
    return Audio(out, rate, "synth:chord")


def adsr(count: int, rate: int, attack: float = 0.01, decay: float = 0.05, sustain: float = 0.7, release: float = 0.1) -> list[float]:
    """An amplitude envelope, so a note starts and stops instead of clicking."""
    a = int(attack * rate)
    d = int(decay * rate)
    r = int(release * rate)
    out = [0.0] * count
    for i in range(count):
        if i < a and a:
            out[i] = i / a
        elif i < a + d and d:
            out[i] = 1.0 - (1.0 - sustain) * (i - a) / d
        elif i >= count - r and r:
            out[i] = sustain * max(0.0, (count - i) / r)
        else:
            out[i] = sustain
    return out


def melody(
    notes: Sequence[str | float] = ("C4", "E4", "G4", "C5", "G4", "E4", "C4"),
    note_duration: float = 0.35,
    rate: int = DEFAULT_RATE,
    amplitude: float = 0.6,
    overtones: int = 3,
) -> Audio:
    """A sequence of notes, each with a few overtones and an envelope.

    This is the most picture-like thing here: a staircase of bright blocks that
    is obviously music at a glance, which makes it the clearest demonstration
    of what the encoding does.
    """
    if not notes:
        raise ValueError("a melody needs at least one note")
    per = _count(note_duration, rate)
    env = adsr(per, rate)
    out: list[float] = []
    for item in notes:
        f = note_to_hz(item) if isinstance(item, str) else float(item)
        gains = [1.0 / (h + 1) ** 1.5 for h in range(max(1, overtones))]
        total = sum(gains) or 1.0
        for t in range(per):
            v = 0.0
            for h, g in enumerate(gains):
                v += (g / total) * math.sin(2.0 * math.pi * f * (h + 1) * t / rate)
            out.append(amplitude * v * env[t])
    return Audio(out, rate, "synth:melody")


def mix(*clips: Audio, normalize: bool = True) -> Audio:
    """Add clips together (they must share a sample rate); the longest wins."""
    if not clips:
        raise ValueError("mix needs at least one clip")
    rate = clips[0].sample_rate
    for c in clips:
        if c.sample_rate != rate:
            raise ValueError(f"cannot mix {rate} Hz with {c.sample_rate} Hz")
    n = max(len(c.samples) for c in clips)
    out = [0.0] * n
    for c in clips:
        for i, v in enumerate(c.samples):
            out[i] += v
    result = Audio(out, rate, "synth:mix")
    return result.normalized() if normalize else result


def fade(audio: Audio, seconds: float = 0.01) -> Audio:
    """Fade the ends in and out, so a clip does not start or stop with a click."""
    n = len(audio.samples)
    k = min(int(seconds * audio.sample_rate), n // 2)
    if k <= 0:
        return audio
    out = list(audio.samples)
    for i in range(k):
        g = i / k
        out[i] *= g
        out[n - 1 - i] *= g
    return Audio(out, audio.sample_rate, audio.source, dict(audio.meta))


def generate(kind: str = "melody", duration: float = 2.0, rate: int = DEFAULT_RATE, freq: float = 440.0, **kwargs: object) -> Audio:
    """Make one of :data:`KINDS` by name - what the command line calls."""
    kind = kind.lower()
    if kind == "sine":
        return sine(freq, duration, rate, float(kwargs.get("amplitude", 0.6)))
    if kind == "harmonics":
        return harmonics(freq, duration, rate, int(kwargs.get("count", 6)))
    if kind == "chirp":
        return chirp(float(kwargs.get("f0", freq)), float(kwargs.get("f1", rate / 4)), duration, rate, method=str(kwargs.get("method", "linear")))
    if kind == "noise":
        return noise(duration, rate, kind=str(kwargs.get("noise", "white")), seed=kwargs.get("seed", 0))  # type: ignore[arg-type]
    if kind == "chord":
        notes = kwargs.get("notes") or ("C4", "E4", "G4")
        return chord(notes, duration, rate)  # type: ignore[arg-type]
    if kind == "melody":
        notes = kwargs.get("notes") or ("C4", "E4", "G4", "C5", "G4", "E4", "C4")
        per = duration / max(1, len(notes))  # type: ignore[arg-type]
        return melody(notes, per, rate)  # type: ignore[arg-type]
    raise ValueError(f"unknown kind {kind!r}; choose from {', '.join(KINDS)}")
