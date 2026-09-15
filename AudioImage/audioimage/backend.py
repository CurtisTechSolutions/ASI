"""Which implementation of the transforms runs: the hand-written one, or numpy.

``"python"`` is :mod:`audioimage.dsp` - the standard library alone, so the tool
works on a bare interpreter.  ``"numpy"`` is :mod:`audioimage.dsp_numpy`, the
same maths vectorised, which matters for Griffin-Lim on anything longer than a
few seconds.  ``"auto"`` (the default everywhere) takes numpy when it imports
and the pure-Python core otherwise.  Setting ``$AUDIOIMAGE_BACKEND`` to
``python`` pins ``auto`` to the standard-library path without touching any
call site, which is how the test suite exercises it on a machine that has
numpy installed.

Both expose an identical surface, so callers simply do::

    from .backend import get_backend
    dsp = get_backend("auto")
    frames = dsp.stft(samples, n_fft, hop, window)
"""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any

from . import dsp

__all__ = ["BACKENDS", "ENV_VAR", "describe_backends", "get_backend", "numpy_available", "resolve"]

BACKENDS = ("auto", "python", "numpy")
ENV_VAR = "AUDIOIMAGE_BACKEND"
"""Set this to ``python`` or ``numpy`` to decide what ``auto`` resolves to."""

_numpy_backend: ModuleType | None = None
_numpy_error: str = ""
_numpy_tried = False


def _load_numpy() -> ModuleType | None:
    """Import the numpy backend once; remember the failure instead of retrying it."""
    global _numpy_backend, _numpy_error, _numpy_tried
    if not _numpy_tried:
        _numpy_tried = True
        try:
            from . import dsp_numpy
        except Exception as exc:  # pragma: no cover - depends on the install
            _numpy_error = f"{type(exc).__name__}: {exc}"
        else:
            _numpy_backend = dsp_numpy
    return _numpy_backend


def numpy_available() -> bool:
    """Whether the numpy backend can be used."""
    return _load_numpy() is not None


def resolve(name: str = "auto") -> str:
    """The backend ``name`` actually selects (``"auto"`` -> ``"numpy"`` or ``"python"``)."""
    name = (name or "auto").lower()
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    if name == "auto":
        override = (os.environ.get(ENV_VAR) or "").strip().lower()
        if override in ("python", "numpy"):
            name = override
        else:
            return "numpy" if numpy_available() else "python"
    return name


def get_backend(name: str = "auto") -> ModuleType:
    """The module implementing the transforms.  ``"numpy"`` raises when it is missing."""
    chosen = resolve(name)
    if chosen == "numpy":
        mod = _load_numpy()
        if mod is None:
            raise ImportError(f"the numpy backend is unavailable ({_numpy_error or 'numpy is not installed'})")
        return mod
    return dsp


def describe_backends() -> dict[str, Any]:
    """What is available, for ``info`` and ``--json`` output."""
    have = numpy_available()
    out: dict[str, Any] = {"backends": list(BACKENDS), "selected": resolve("auto"), "numpy": have}
    override = (os.environ.get(ENV_VAR) or "").strip().lower()
    if override:
        out["override"] = override
    if have:
        import numpy as np

        out["numpy_version"] = np.__version__
    elif _numpy_error:
        out["numpy_error"] = _numpy_error
    return out
