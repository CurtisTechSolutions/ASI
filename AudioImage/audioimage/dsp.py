"""The signal processing, written from scratch on the standard library.

Everything the codec needs to move between a waveform and its short-time
spectrum lives here, in plain Python with plain ``complex`` numbers:

* :func:`fft` / :func:`ifft` - iterative radix-2 Cooley-Tukey, power-of-two sizes.
* :func:`rfft` / :func:`irfft` - the real-input pair.  A real signal of length
  ``N`` is packed into ``N/2`` complex numbers and transformed with a single
  half-length :func:`fft`, which is about twice as fast as transforming the
  real signal directly and returns the ``N/2 + 1`` non-redundant bins.
* :func:`stft` / :func:`istft` - the sliding window analysis and its
  weighted-overlap-add inverse.  With the default periodic Hann window the
  pair reconstructs the input to floating-point accuracy for any hop that
  divides the window evenly.
* :func:`griffin_lim` - magnitudes back to a waveform when the phase is gone,
  which is exactly the situation after a spectrogram has been through a
  picture.

:mod:`audioimage.dsp_numpy` mirrors this module for speed when numpy is
installed; :mod:`audioimage.backend` picks between the two.  The two are
expected to agree to ~1e-9, and ``tests/test_backends.py`` checks that.
"""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Callable, Sequence
from functools import lru_cache

__all__ = [
    "WINDOWS",
    "cola_sum",
    "fft",
    "frame_count",
    "griffin_lim",
    "ifft",
    "irfft",
    "magnitudes",
    "resample",
    "rfft",
    "stft",
    "istft",
    "window_values",
]

WINDOWS = ("hann", "hamming", "blackman", "rect")

NAME = "python"


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def window_values(name: str, size: int) -> tuple[float, ...]:
    """The ``size`` samples of a *periodic* window (the right choice for an STFT).

    Periodic (``cos(2*pi*n/N)``, not ``N-1``) is what makes overlapping frames
    sum to a constant, so analysis and synthesis cancel exactly.
    """
    if size < 1:
        raise ValueError(f"window size must be >= 1, got {size}")
    name = name.lower()
    if name in ("rect", "rectangular", "boxcar", "none"):
        return tuple([1.0] * size)
    two_pi_n = 2.0 * math.pi / size
    if name == "hann":
        return tuple(0.5 - 0.5 * math.cos(two_pi_n * i) for i in range(size))
    if name == "hamming":
        return tuple(0.54 - 0.46 * math.cos(two_pi_n * i) for i in range(size))
    if name == "blackman":
        return tuple(
            0.42 - 0.5 * math.cos(two_pi_n * i) + 0.08 * math.cos(2.0 * two_pi_n * i) for i in range(size)
        )
    raise ValueError(f"unknown window {name!r}; choose from {', '.join(WINDOWS)}")


def cola_sum(name: str, size: int, hop: int) -> list[float]:
    """One hop of the squared-window overlap sum - flat means perfect reconstruction."""
    w = window_values(name, size)
    out = [0.0] * hop
    for i in range(size):
        out[i % hop] += w[i] * w[i]
    return out


# ---------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@lru_cache(maxsize=16)
def _twiddles(n: int, inverse: bool) -> tuple[complex, ...]:
    """``exp(-2*pi*i*k/n)`` for ``k < n/2`` (conjugated when inverting)."""
    sign = 2.0 if inverse else -2.0
    return tuple(cmath.exp(complex(0.0, sign * math.pi * k / n)) for k in range(n // 2))


@lru_cache(maxsize=16)
def _bit_reversal(n: int) -> tuple[int, ...]:
    """The bit-reversed index permutation for a size-``n`` transform."""
    bits = n.bit_length() - 1
    out = [0] * n
    for i in range(n):
        r = 0
        v = i
        for _ in range(bits):
            r = (r << 1) | (v & 1)
            v >>= 1
        out[i] = r
    return tuple(out)


def fft(values: Sequence[complex], inverse: bool = False) -> list[complex]:
    """Radix-2 FFT of a power-of-two sequence.  ``inverse`` scales by ``1/n``."""
    n = len(values)
    if n == 0:
        return []
    if not _is_pow2(n):
        raise ValueError(f"fft size must be a power of two, got {n}")
    rev = _bit_reversal(n)
    buf = [complex(values[rev[i]]) for i in range(n)]
    tw = _twiddles(n, inverse)
    size = 2
    while size <= n:
        half = size >> 1
        step = n // size
        for start in range(0, n, size):
            k = 0
            for i in range(start, start + half):
                j = i + half
                t = tw[k] * buf[j]
                buf[j] = buf[i] - t
                buf[i] = buf[i] + t
                k += step
        size <<= 1
    if inverse:
        inv = 1.0 / n
        buf = [v * inv for v in buf]
    return buf


def ifft(values: Sequence[complex]) -> list[complex]:
    """Inverse of :func:`fft`."""
    return fft(values, inverse=True)


def rfft(values: Sequence[float]) -> list[complex]:
    """The ``n // 2 + 1`` non-redundant bins of a real, even-length signal.

    The even and odd samples are packed into one complex sequence of half the
    length, transformed once, and unpacked with the standard split-radix
    identity - the same answer as ``fft`` on the real values, at half the cost.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [complex(values[0], 0.0)]
    if not _is_pow2(n):
        raise ValueError(f"rfft size must be a power of two, got {n}")
    half = n // 2
    packed = [complex(values[2 * i], values[2 * i + 1]) for i in range(half)]
    z = fft(packed)
    tw = _twiddles(n, False)  # exp(-2*pi*i*k/n), k < n/2
    out: list[complex] = [0j] * (half + 1)
    for k in range(half):
        zk = z[k]
        zc = z[(half - k) % half].conjugate()
        even = (zk + zc) * 0.5
        odd = (zk - zc) * -0.5j  # divide by 2i
        out[k] = even + tw[k] * odd
        if k == 0:
            out[half] = even - odd  # tw[n/2] == -1, and E/O repeat with period n/2
    return out


def irfft(spectrum: Sequence[complex], size: int | None = None) -> list[float]:
    """Inverse of :func:`rfft`: ``n // 2 + 1`` bins back to ``n`` real samples."""
    bins = len(spectrum)
    if bins == 0:
        return []
    n = size if size is not None else (bins - 1) * 2
    if n == 1:
        return [spectrum[0].real]
    if not _is_pow2(n):
        raise ValueError(f"irfft size must be a power of two, got {n}")
    if bins != n // 2 + 1:
        raise ValueError(f"expected {n // 2 + 1} bins for size {n}, got {bins}")
    half = n // 2
    tw = _twiddles(n, True)  # exp(+2*pi*i*k/n)
    packed: list[complex] = [0j] * half
    for k in range(half):
        if k == 0:
            # DC and Nyquist are the two bins a real signal cannot give an
            # imaginary part.  A spectrum that has been through a picture (or
            # through Griffin-Lim's random phase) may carry one anyway, so drop
            # it - the same thing numpy's irfft does - instead of folding the
            # impossible part back into the samples.
            a = complex(spectrum[0].real, 0.0)
            b = complex(spectrum[half].real, 0.0)
        else:
            a = spectrum[k]
            b = spectrum[half - k].conjugate()
        even = (a + b) * 0.5
        odd = (a - b) * 0.5 * tw[k]
        packed[k] = even + 1j * odd
    z = ifft(packed)
    out = [0.0] * n
    for k in range(half):
        out[2 * k] = z[k].real
        out[2 * k + 1] = z[k].imag
    return out


# ---------------------------------------------------------------------------
# short-time Fourier transform
# ---------------------------------------------------------------------------


def frame_count(length: int, n_fft: int, hop: int, center: bool = True) -> int:
    """How many frames :func:`stft` produces for a signal of ``length`` samples."""
    if n_fft < 2 or not _is_pow2(n_fft):
        raise ValueError(f"n_fft must be a power of two >= 2, got {n_fft}")
    if hop < 1:
        raise ValueError(f"hop must be >= 1, got {hop}")
    padded = length + n_fft if center else length
    if padded < n_fft:
        return 1
    return 1 + -(-(padded - n_fft) // hop)  # ceil division: cover every sample


def _pad(samples: Sequence[float], n_fft: int, hop: int, center: bool) -> list[float]:
    """Half a window of silence at each end (when centred), then enough tail for whole frames."""
    pad = n_fft // 2 if center else 0
    x = [0.0] * pad + list(samples) + [0.0] * pad
    frames = frame_count(len(samples), n_fft, hop, center)
    need = n_fft + hop * (frames - 1)
    if len(x) < need:
        x.extend([0.0] * (need - len(x)))
    return x


def stft(
    samples: Sequence[float],
    n_fft: int = 1024,
    hop: int = 256,
    window: str = "hann",
    center: bool = True,
) -> list[list[complex]]:
    """Sliding-window spectrum: one row per frame, ``n_fft // 2 + 1`` bins per row."""
    x = _pad(samples, n_fft, hop, center)
    w = window_values(window, n_fft)
    frames = frame_count(len(samples), n_fft, hop, center)
    out: list[list[complex]] = []
    for f in range(frames):
        base = f * hop
        chunk = x[base : base + n_fft]
        out.append(rfft([chunk[i] * w[i] for i in range(n_fft)]))
    return out


def istft(
    frames: Sequence[Sequence[complex]],
    n_fft: int = 1024,
    hop: int = 256,
    window: str = "hann",
    center: bool = True,
    length: int | None = None,
) -> list[float]:
    """Weighted overlap-add inverse of :func:`stft`.

    Each frame is transformed back, multiplied by the same window again and
    added in; the running sum of squared windows then divides it out, which is
    what makes the round trip exact rather than merely close.
    """
    frames = list(frames)
    if not frames:
        return []
    w = window_values(window, n_fft)
    total = n_fft + hop * (len(frames) - 1)
    acc = [0.0] * total
    norm = [0.0] * total
    for f, spec in enumerate(frames):
        chunk = irfft(spec, n_fft)
        base = f * hop
        for i in range(n_fft):
            wi = w[i]
            acc[base + i] += chunk[i] * wi
            norm[base + i] += wi * wi
    eps = 1e-12
    out = [(acc[i] / norm[i]) if norm[i] > eps else 0.0 for i in range(total)]
    if center:
        pad = n_fft // 2
        out = out[pad : len(out) - pad]
    if length is not None:
        out = out[:length] + [0.0] * max(0, length - len(out))
    return out


def magnitudes(frames: Sequence[Sequence[complex]]) -> list[list[float]]:
    """``abs()`` of every bin: frames x bins, which is what the picture stores."""
    return [[abs(v) for v in row] for row in frames]


# ---------------------------------------------------------------------------
# phase reconstruction
# ---------------------------------------------------------------------------


def griffin_lim(
    mags: Sequence[Sequence[float]],
    n_fft: int = 1024,
    hop: int = 256,
    window: str = "hann",
    center: bool = True,
    iterations: int = 48,
    momentum: float = 0.99,
    length: int | None = None,
    seed: int | None = 0,
    init: str = "random",
    progress: Callable[[int, int, float], None] | None = None,
) -> list[float]:
    """Recover a waveform from magnitudes alone.

    A picture of a spectrogram keeps how loud each frequency was and throws
    away *where in its cycle* it was - the phase.  Griffin-Lim guesses: take
    the inverse transform of the magnitudes with the current phase, transform
    the result straight back, and keep the phase that came out (the magnitudes
    are forced back to the target every round).  Each pass lands closer to a
    spectrum that an actual waveform could have produced.

    ``momentum`` (the "fast Griffin-Lim" over-relaxation) pushes each update a
    little past the new estimate, which converges in far fewer rounds.
    ``progress`` is called as ``(iteration, total, error)`` with the spectral
    convergence after each pass.
    """
    rows = list(mags)
    if not rows:
        return []
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1], got {momentum}")

    bins = n_fft // 2 + 1
    rng = random.Random(seed)
    if init == "zeros":
        spec = [[complex(m, 0.0) for m in row] for row in rows]
    elif init == "random":
        spec = [
            [cmath.rect(m, rng.uniform(-math.pi, math.pi)) for m in row]
            for row in rows
        ]
    else:
        raise ValueError(f"unknown init {init!r}; choose from zeros, random")
    for row in spec:
        if len(row) != bins:
            raise ValueError(f"each frame needs {bins} bins for n_fft={n_fft}, got {len(row)}")

    target_norm = math.sqrt(sum(m * m for row in rows for m in row)) or 1.0
    # the over-relaxation is damped: a raw step of 0.99 overshoots so far that
    # the error climbs instead of falling.  alpha / (1 + alpha) is the step the
    # fast Griffin-Lim paper uses, and 0.99 then means a sensible 0.497.
    accel = momentum / (1.0 + momentum) if momentum else 0.0
    prev: list[list[complex]] | None = None
    error = 0.0
    for it in range(iterations):
        signal = istft(spec, n_fft, hop, window, center, length)
        rebuilt = stft(signal, n_fft, hop, window, center)
        if len(rebuilt) > len(rows):
            rebuilt = rebuilt[: len(rows)]
        while len(rebuilt) < len(rows):
            rebuilt.append([0j] * bins)

        diff = 0.0
        nxt: list[list[complex]] = []
        for f, row in enumerate(rows):
            got = rebuilt[f]
            out_row: list[complex] = [0j] * bins
            for k in range(bins):
                v = got[k]
                a = abs(v)
                diff += (a - row[k]) ** 2
                if accel and prev is not None:
                    v = v - accel * prev[f][k]
                    a = abs(v)
                out_row[k] = complex(row[k], 0.0) if a < 1e-12 else v * (row[k] / a)
            nxt.append(out_row)
        prev = rebuilt
        spec = nxt
        error = math.sqrt(diff) / target_norm
        if progress is not None:
            progress(it + 1, iterations, error)

    return istft(spec, n_fft, hop, window, center, length)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def resample(samples: Sequence[float], src_rate: int, dst_rate: int) -> list[float]:
    """Linear-interpolation sample-rate conversion (good enough for previews)."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be > 0")
    src = list(samples)
    if src_rate == dst_rate or not src:
        return src
    ratio = dst_rate / src_rate
    out_len = max(1, int(round(len(src) * ratio)))
    out = [0.0] * out_len
    last = len(src) - 1
    for i in range(out_len):
        pos = i / ratio
        lo = int(pos)
        if lo >= last:
            out[i] = src[last]
        else:
            frac = pos - lo
            out[i] = src[lo] * (1.0 - frac) + src[lo + 1] * frac
    return out
