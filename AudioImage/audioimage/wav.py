"""WAV files in and out, as plain floating-point samples.

The standard library's :mod:`wave` only understands integer PCM, and
``audioop`` - the usual companion for converting those integers - is
deprecated (gone in 3.13), so this module parses and writes RIFF/WAVE itself.

Read support: ``WAVE_FORMAT_PCM`` (8-bit unsigned, 16/24/32-bit signed),
``WAVE_FORMAT_IEEE_FLOAT`` (32/64-bit) and ``WAVE_FORMAT_EXTENSIBLE`` (whose
first GUID field carries one of the two codes above).  Write support: the
same PCM widths plus 32-bit float.

Samples always reach the rest of the package as a ``list[float]`` nominally in
``[-1.0, 1.0]``, one channel (:func:`read_wav` mixes multi-channel files down
by averaging), which is what :mod:`audioimage.dsp` expects.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Audio",
    "WavError",
    "read_wav",
    "read_wav_bytes",
    "write_wav",
    "write_wav_bytes",
]

FORMAT_PCM = 0x0001
FORMAT_FLOAT = 0x0003
FORMAT_EXTENSIBLE = 0xFFFE

MAX_READ_BYTES = 1 << 31
"""Refuse a ``data`` chunk larger than 2 GiB; pure-Python decoding of one would not finish."""


class WavError(ValueError):
    """A file that is not a WAV, or a WAV in a shape this module cannot decode."""


@dataclass
class Audio:
    """A mono signal: ``samples`` in ``[-1, 1]`` at ``sample_rate`` hertz."""

    samples: list[float]
    sample_rate: int
    source: str = ""
    """Where it came from (a path, ``"synth"``, ``"decoded"``), for reports only."""
    meta: dict[str, Any] = field(default_factory=dict)
    """Anything the reader learned and the caller may want back (bit depth, channel count)."""

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise WavError(f"sample_rate must be > 0, got {self.sample_rate}")
        if not isinstance(self.samples, list):
            self.samples = list(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return len(self.samples) / float(self.sample_rate)

    @property
    def peak(self) -> float:
        """Largest absolute sample (``0.0`` when empty)."""
        return max((abs(s) for s in self.samples), default=0.0)

    @property
    def rms(self) -> float:
        """Root mean square of the samples (``0.0`` when empty)."""
        if not self.samples:
            return 0.0
        return (sum(s * s for s in self.samples) / len(self.samples)) ** 0.5

    def normalized(self, target: float = 0.99) -> "Audio":
        """A copy scaled so the peak is ``target``; unchanged when it is silent."""
        peak = self.peak
        if peak <= 0.0:
            return Audio(list(self.samples), self.sample_rate, self.source, dict(self.meta))
        gain = target / peak
        return Audio([s * gain for s in self.samples], self.sample_rate, self.source, dict(self.meta))

    def clipped(self) -> "Audio":
        """A copy with every sample clamped into ``[-1, 1]``."""
        return Audio(
            [-1.0 if s < -1.0 else (1.0 if s > 1.0 else s) for s in self.samples],
            self.sample_rate,
            self.source,
            dict(self.meta),
        )

    def describe(self) -> dict[str, Any]:
        """A summary for ``info`` / ``--json`` output."""
        return {
            "source": self.source,
            "sample_rate": self.sample_rate,
            "samples": len(self.samples),
            "duration": round(self.duration, 6),
            "peak": round(self.peak, 6),
            "rms": round(self.rms, 6),
            **self.meta,
        }

    def __repr__(self) -> str:
        return f"Audio(samples={len(self.samples)}, sample_rate={self.sample_rate}, duration={self.duration:.3f}s)"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _chunks(data: bytes) -> "list[tuple[bytes, int, int]]":
    """``(id, start, size)`` of every chunk in a RIFF body, ignoring padding bytes."""
    out: list[tuple[bytes, int, int]] = []
    pos = 12  # past "RIFF" + size + "WAVE"
    n = len(data)
    while pos + 8 <= n:
        cid = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        start = pos + 8
        # a truncated final chunk is common in streamed files: keep what is there
        out.append((cid, start, min(size, n - start)))
        pos = start + size + (size & 1)
    return out


def _decode_samples(raw: bytes, fmt: int, bits: int) -> list[float]:
    """Interleaved sample bytes -> floats in ``[-1, 1]`` (or as stored, for float formats)."""
    if fmt == FORMAT_FLOAT:
        if bits == 32:
            count = len(raw) // 4
            return list(struct.unpack_from(f"<{count}f", raw, 0))
        if bits == 64:
            count = len(raw) // 8
            return list(struct.unpack_from(f"<{count}d", raw, 0))
        raise WavError(f"unsupported float width: {bits}-bit")
    if fmt != FORMAT_PCM:
        raise WavError(f"unsupported WAV format code 0x{fmt:04X} (PCM and IEEE float only)")
    if bits == 8:  # 8-bit PCM is unsigned, biased by 128
        return [(b - 128) / 128.0 for b in raw]
    if bits == 16:
        count = len(raw) // 2
        return [v / 32768.0 for v in struct.unpack_from(f"<{count}h", raw, 0)]
    if bits == 24:
        out: list[float] = []
        for i in range(0, len(raw) - 2, 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if v & 0x800000:
                v -= 0x1000000
            out.append(v / 8388608.0)
        return out
    if bits == 32:
        count = len(raw) // 4
        return [v / 2147483648.0 for v in struct.unpack_from(f"<{count}i", raw, 0)]
    raise WavError(f"unsupported PCM width: {bits}-bit")


def _mixdown(samples: list[float], channels: int) -> list[float]:
    """Average interleaved channels into one."""
    if channels <= 1:
        return samples
    frames = len(samples) // channels
    inv = 1.0 / channels
    return [sum(samples[i * channels : i * channels + channels]) * inv for i in range(frames)]


def read_wav_bytes(data: bytes, mono: bool = True, source: str = "") -> Audio:
    """Decode a WAV held in memory.  ``mono=False`` keeps the interleaved channels."""
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file (bad magic)")
    fmt_chunk: bytes | None = None
    data_chunk: bytes | None = None
    for cid, start, size in _chunks(data):
        if cid == b"fmt " and fmt_chunk is None:
            fmt_chunk = data[start : start + size]
        elif cid == b"data" and data_chunk is None:
            if size > MAX_READ_BYTES:
                raise WavError(f"data chunk too large: {size} bytes")
            data_chunk = data[start : start + size]
    if fmt_chunk is None:
        raise WavError("no 'fmt ' chunk")
    if data_chunk is None:
        raise WavError("no 'data' chunk")
    if len(fmt_chunk) < 16:
        raise WavError(f"'fmt ' chunk is {len(fmt_chunk)} bytes, need at least 16")

    fmt, channels, rate, _byte_rate, _align, bits = struct.unpack_from("<HHIIHH", fmt_chunk, 0)
    if fmt == FORMAT_EXTENSIBLE:
        if len(fmt_chunk) < 26:
            raise WavError("WAVE_FORMAT_EXTENSIBLE without its extension block")
        (fmt,) = struct.unpack_from("<H", fmt_chunk, 24)  # first field of the sub-format GUID
    if channels < 1:
        raise WavError(f"channel count must be >= 1, got {channels}")
    if rate < 1:
        raise WavError(f"sample rate must be >= 1, got {rate}")

    samples = _decode_samples(data_chunk, fmt, bits)
    frames = len(samples) // channels
    samples = samples[: frames * channels]
    if mono:
        samples = _mixdown(samples, channels)
    return Audio(
        samples,
        rate,
        source=source,
        meta={
            "channels": channels,
            "bits": bits,
            "format": "float" if fmt == FORMAT_FLOAT else "pcm",
            "mixed_down": bool(mono and channels > 1),
        },
    )


def read_wav(path: str | os.PathLike[str], mono: bool = True) -> Audio:
    """Read a WAV file from disk (multi-channel input is averaged into mono)."""
    with open(path, "rb") as fh:
        return read_wav_bytes(fh.read(), mono=mono, source=str(path))


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _encode_samples(samples: list[float], bits: int, floating: bool) -> bytes:
    """Floats in ``[-1, 1]`` -> the sample bytes of the requested width."""
    if floating:
        if bits != 32:
            raise WavError(f"float output must be 32-bit, got {bits}")
        return struct.pack(f"<{len(samples)}f", *samples)
    if bits == 8:
        return bytes(min(255, max(0, int(round(s * 127.0)) + 128)) for s in samples)
    if bits == 16:
        return struct.pack(
            f"<{len(samples)}h", *(min(32767, max(-32768, int(round(s * 32767.0)))) for s in samples)
        )
    if bits == 24:
        out = bytearray()
        for s in samples:
            v = min(8388607, max(-8388608, int(round(s * 8388607.0))))
            if v < 0:
                v += 0x1000000
            out += bytes((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF))
        return bytes(out)
    if bits == 32:
        return struct.pack(
            f"<{len(samples)}i",
            *(min(2147483647, max(-2147483648, int(round(s * 2147483647.0)))) for s in samples),
        )
    raise WavError(f"unsupported PCM width: {bits}-bit")


def write_wav_bytes(audio: Audio, bits: int = 16, floating: bool = False, clip: bool = True) -> bytes:
    """Encode ``audio`` as a complete WAV file.

    ``clip`` clamps to ``[-1, 1]`` first - integer PCM would otherwise wrap a
    sample that overshoots into the opposite sign.  Decoded audio regularly
    overshoots a little, so leave it on unless the caller has already scaled.
    """
    samples = audio.samples
    if clip:
        samples = [-1.0 if s < -1.0 else (1.0 if s > 1.0 else s) for s in samples]
    raw = _encode_samples(samples, bits, floating)
    fmt = FORMAT_FLOAT if floating else FORMAT_PCM
    channels = 1
    block_align = channels * (bits // 8)
    fmt_chunk = struct.pack(
        "<HHIIHH", fmt, channels, audio.sample_rate, audio.sample_rate * block_align, block_align, bits
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    body += b"data" + struct.pack("<I", len(raw)) + raw
    if len(raw) & 1:  # chunks are word-aligned
        body += b"\x00"
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_wav(
    path: str | os.PathLike[str],
    audio: Audio,
    bits: int = 16,
    floating: bool = False,
    clip: bool = True,
) -> str:
    """Write ``audio`` to ``path`` and return the path."""
    blob = write_wav_bytes(audio, bits=bits, floating=floating, clip=clip)
    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return str(path)
