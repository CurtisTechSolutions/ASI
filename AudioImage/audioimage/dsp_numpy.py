"""The same transforms as :mod:`audioimage.dsp`, vectorised with numpy.

Nothing here changes the maths - every function is the numpy spelling of its
pure-Python twin and returns the same plain lists, so the rest of the package
never learns which one ran.  It exists because Griffin-Lim is the expensive
part of decoding: fifty passes over a spectrogram means a hundred transforms
of the whole signal, which numpy does in the time pure Python needs for one.

The module raises :class:`ImportError` at import time when numpy is missing;
:mod:`audioimage.backend` catches that and falls back.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

import numpy as np  # noqa: F401  (an ImportError here is the point)

from .dsp import WINDOWS, cola_sum, frame_count, resample, window_values  # noqa: F401

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

NAME = "numpy"


def _window(name: str, size: int) -> "np.ndarray":
    return np.asarray(window_values(name, size), dtype=np.float64)


def fft(values: Sequence[complex], inverse: bool = False) -> list[complex]:
    """Radix-2 FFT; ``inverse`` scales by ``1/n`` (numpy's ``ifft`` convention)."""
    arr = np.asarray(values, dtype=np.complex128)
    if arr.size == 0:
        return []
    return (np.fft.ifft(arr) if inverse else np.fft.fft(arr)).tolist()


def ifft(values: Sequence[complex]) -> list[complex]:
    """Inverse of :func:`fft`."""
    return fft(values, inverse=True)


def rfft(values: Sequence[float]) -> list[complex]:
    """The ``n // 2 + 1`` non-redundant bins of a real signal."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    return np.fft.rfft(arr).tolist()


def irfft(spectrum: Sequence[complex], size: int | None = None) -> list[float]:
    """Inverse of :func:`rfft`."""
    arr = np.asarray(spectrum, dtype=np.complex128)
    if arr.size == 0:
        return []
    n = size if size is not None else (arr.size - 1) * 2
    return np.fft.irfft(arr, n=n).tolist()


# ---------------------------------------------------------------------------
# short-time Fourier transform
# ---------------------------------------------------------------------------


def _pad(samples: Sequence[float], n_fft: int, hop: int, center: bool) -> "np.ndarray":
    x = np.asarray(samples, dtype=np.float64)
    length = x.size
    if center:
        pad = n_fft // 2
        x = np.concatenate((np.zeros(pad), x, np.zeros(pad)))
    frames = frame_count(length, n_fft, hop, center)
    need = n_fft + hop * (frames - 1)
    if x.size < need:
        x = np.concatenate((x, np.zeros(need - x.size)))
    return x


def _frames_matrix(x: "np.ndarray", n_fft: int, hop: int, frames: int) -> "np.ndarray":
    """A ``frames x n_fft`` view of the padded signal, without copying it."""
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(frames, n_fft), strides=(hop * stride, stride))


def _stft_array(samples: Sequence[float], n_fft: int, hop: int, window: str, center: bool) -> "np.ndarray":
    x = _pad(samples, n_fft, hop, center)
    frames = frame_count(len(samples), n_fft, hop, center)
    mat = _frames_matrix(x, n_fft, hop, frames) * _window(window, n_fft)
    return np.fft.rfft(mat, axis=1)


def _istft_array(
    spec: "np.ndarray", n_fft: int, hop: int, window: str, center: bool, length: int | None
) -> "np.ndarray":
    frames = spec.shape[0]
    if frames == 0:
        return np.zeros(0)
    w = _window(window, n_fft)
    chunks = np.fft.irfft(spec, n=n_fft, axis=1) * w
    total = n_fft + hop * (frames - 1)
    acc = np.zeros(total)
    norm = np.zeros(total)
    wsq = w * w
    for f in range(frames):
        base = f * hop
        acc[base : base + n_fft] += chunks[f]
        norm[base : base + n_fft] += wsq
    out = np.divide(acc, norm, out=np.zeros_like(acc), where=norm > 1e-12)
    if center:
        pad = n_fft // 2
        out = out[pad : out.size - pad]
    if length is not None:
        if out.size >= length:
            out = out[:length]
        else:
            out = np.concatenate((out, np.zeros(length - out.size)))
    return out


def stft(
    samples: Sequence[float], n_fft: int = 1024, hop: int = 256, window: str = "hann", center: bool = True
) -> list[list[complex]]:
    """Sliding-window spectrum: one row per frame, ``n_fft // 2 + 1`` bins per row."""
    frame_count(len(samples), n_fft, hop, center)  # validates n_fft / hop the same way
    # .tolist() rather than list(): plain Python floats and complexes come out,
    # so the two backends return identical *types*, not just identical values
    return _stft_array(samples, n_fft, hop, window, center).tolist()


def istft(
    frames: Sequence[Sequence[complex]],
    n_fft: int = 1024,
    hop: int = 256,
    window: str = "hann",
    center: bool = True,
    length: int | None = None,
) -> list[float]:
    """Weighted overlap-add inverse of :func:`stft`."""
    spec = np.asarray([list(row) for row in frames], dtype=np.complex128)
    if spec.size == 0:
        return []
    return _istft_array(spec, n_fft, hop, window, center, length).tolist()


def magnitudes(frames: Sequence[Sequence[complex]]) -> list[list[float]]:
    """``abs()`` of every bin."""
    return np.abs(np.asarray([list(row) for row in frames], dtype=np.complex128)).tolist()


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
    """Magnitudes back to a waveform - see :func:`audioimage.dsp.griffin_lim`.

    The whole loop stays in numpy: the spectrogram is never converted back to
    Python objects between passes, which is where the speed comes from.
    """
    target = np.asarray([list(row) for row in mags], dtype=np.float64)
    if target.size == 0:
        return []
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1], got {momentum}")
    bins = n_fft // 2 + 1
    if target.shape[1] != bins:
        raise ValueError(f"each frame needs {bins} bins for n_fft={n_fft}, got {target.shape[1]}")

    if init == "zeros":
        spec = target.astype(np.complex128)
    elif init == "random":
        # the *standard library* RNG, drawn in the same order as the pure-Python
        # backend, so a given seed gives both backends the same starting phase
        # and therefore the same waveform - Griffin-Lim is not convex, and two
        # different random starts land on two different (equally valid) answers
        rng = random.Random(seed)
        phase = np.array(
            [[rng.uniform(-math.pi, math.pi) for _ in range(target.shape[1])] for _ in range(target.shape[0])]
        )
        spec = target * np.exp(1j * phase)
    else:
        raise ValueError(f"unknown init {init!r}; choose from zeros, random")

    target_norm = float(np.linalg.norm(target)) or 1.0
    # damped over-relaxation - see the note in audioimage.dsp.griffin_lim
    accel = momentum / (1.0 + momentum) if momentum else 0.0
    prev: "np.ndarray | None" = None
    for it in range(iterations):
        signal = _istft_array(spec, n_fft, hop, window, center, length)
        rebuilt = _stft_array(signal, n_fft, hop, window, center)
        if rebuilt.shape[0] > target.shape[0]:
            rebuilt = rebuilt[: target.shape[0]]
        elif rebuilt.shape[0] < target.shape[0]:
            pad = np.zeros((target.shape[0] - rebuilt.shape[0], bins), dtype=np.complex128)
            rebuilt = np.concatenate((rebuilt, pad), axis=0)

        error = float(np.linalg.norm(np.abs(rebuilt) - target)) / target_norm
        step = rebuilt if (not accel or prev is None) else rebuilt - accel * prev
        prev = rebuilt
        amp = np.abs(step)
        spec = np.where(amp < 1e-12, target.astype(np.complex128), step * (target / np.where(amp < 1e-12, 1.0, amp)))
        if progress is not None:
            progress(it + 1, iterations, error)

    return _istft_array(spec, n_fft, hop, window, center, length).tolist()
