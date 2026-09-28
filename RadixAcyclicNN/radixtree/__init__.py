"""radixtree - a radix tree neural network with no cycles: the acyclic sibling of ``RadixCyclicNN``.

Pure Python, standard library only.  See ``DESIGN.md`` for the specification
and ``README.md`` for what it measured against the cyclic graph.

    from radixtree import RadixTreeNet

    model = RadixTreeNet(seed=1)
    model.train(["the cat sat on the mat", "the dog sat on the log"], epochs=10, lr=0.5, batch_size=8)
    print(model.predict("the cat", to_end=True).text)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .activation import (
    DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K, MIN_B, SineActivation, edge_signal, sine_activation,
    sine_derivative, sine_partials,
)
from .encoding import END_LABEL, START_LABEL, WINDOW, Encoding
from .model import MAX_LEGS, UNKNOWN_PROB, RadixTreeNet, TrainConfig, load_model
from .search import COSTS, PathResult, cheapest_path, sample_walk
from .tree import FIRST, ROOT, START, RadixTree

__all__ = [
    "__version__",
    "RadixTreeNet", "TrainConfig", "load_model", "RadixTree", "Encoding", "SineActivation", "PathResult",
    "cheapest_path", "sample_walk",
    "ROOT", "START", "FIRST", "WINDOW", "START_LABEL", "END_LABEL", "COSTS", "MAX_LEGS", "UNKNOWN_PROB",
    "DEFAULT_A", "DEFAULT_B", "DEFAULT_H", "DEFAULT_K", "MIN_B",
    "edge_signal", "sine_activation", "sine_derivative", "sine_partials",
]
