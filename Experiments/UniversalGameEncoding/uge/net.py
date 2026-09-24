"""Finding ``radixnet``, and the training settings this experiment uses.

``Experiments/`` directories stand alone, but the whole point of this one is
the network in ``RadixCyclicNN/``, so it has to be found rather than vendored.
The search order is ``$RADIXNET_PATH``, then the sibling directory two levels
up, then whatever is already importable (a ``pip install -e`` of the package).
"""

from __future__ import annotations

import os
import sys

__all__ = ["TRAIN_DEFAULTS", "import_radixnet", "new_net", "radixnet_root"]

TRAIN_DEFAULTS = {"epochs": 6, "lr": 0.05, "batch_size": 64, "auto_compress": True}
"""``RadixNet.train`` settings shared by every arm, so only the encoding varies."""


def radixnet_root() -> str | None:
    """Where ``RadixCyclicNN`` is, or ``None`` if ``radixnet`` is already importable."""
    env = os.environ.get("RADIXNET_PATH")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling = os.path.join(os.path.dirname(os.path.dirname(here)), "RadixCyclicNN")
    return sibling if os.path.isdir(os.path.join(sibling, "radixnet")) else None


def import_radixnet():
    """Import and return the ``radixnet`` package, putting it on the path first."""
    root = radixnet_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        import radixnet  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - the message is the point
        raise ImportError(
            "radixnet not found.  It lives in RadixCyclicNN/ two directories up; "
            "set RADIXNET_PATH to it if this experiment has been moved."
        ) from exc
    return radixnet


def new_net(seed: int = 0, backend: str = "auto"):
    """A fresh :class:`radixnet.RadixNet`."""
    return import_radixnet().RadixNet(seed=seed, backend=backend)
