"""UGE - a universal encoder and decoder for games, and the tapes it writes.

The claim this package makes is in :mod:`uge.tape`: every finite, discrete,
sequential game is a labelled transition system, a walk through one is what
RadixCyclicNN stores, and so encoding a game is not a string problem but the
choice of *which projection of the walk to write down*.  Six games implement
the interface (:mod:`uge.games`); nothing in :mod:`uge.tape` or
:mod:`uge.codec` knows which one it is looking at.

The package is deliberately free of any dependency on ``radixnet`` - only
``experiment.py`` and :mod:`uge.metrics` import the network.  The encoder is
meant to be liftable into ``RadixCyclicNN/radixnet/`` next to ``vision.py``
and ``speech.py``, which do the same job for images and audio.
"""

from .codec import DecodeResult, Play, decode, encode, encode_prefix
from .game import REGISTRY, Game, game_names, get_game
from .tape import CODES, LAYOUTS, W, TapeError, TapeSpec
from . import games as _games  # noqa: F401  - importing it is what registers the six

__all__ = [
    "CODES",
    "DecodeResult",
    "Game",
    "LAYOUTS",
    "Play",
    "REGISTRY",
    "TapeError",
    "TapeSpec",
    "W",
    "decode",
    "encode",
    "encode_prefix",
    "game_names",
    "get_game",
]
