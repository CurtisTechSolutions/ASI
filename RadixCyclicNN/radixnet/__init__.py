"""radixnet - a self-compressing cyclic-graph neural network (Radix Tree idea).

Pure Python (standard library only); ``torch`` is an optional accelerator that
is imported lazily.  See ``DESIGN.md`` for the full specification.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .activation import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_H,
    DEFAULT_K,
    SineActivation,
    edge_signal,
    sine_activation,
    sine_derivative,
    sine_partials,
)
from .backend import (
    CSR,
    Backend,
    NodeParams,
    PythonBackend,
    describe_backends,
    get_backend,
    torch_available,
)
from .checkpoint import CheckpointManager
from .encoding import END_LABEL, START_LABEL, WINDOW, Decoder, Encoder
from .gan import EvolveConfig, Evolver
from .graph import END, START, RadixCyclicGraph
from .beam import Prediction, beam_predict
from .countnet import CountRewardGraph, CountRewardNet
from .model import GraphModel, RadixNet, TrainConfig, load_model, model_class, model_kinds, new_model
from .search import PathResult, dijkstra_predict, sample_walk

__all__ = [
    "__version__",
    "RadixNet", "TrainConfig", "GraphModel", "CountRewardNet", "CountRewardGraph",
    "load_model", "model_class", "model_kinds", "new_model", "Prediction", "beam_predict",
    "Encoder", "Decoder", "SineActivation",
    "get_backend", "describe_backends", "torch_available",
    "CheckpointManager", "Evolver", "EvolveConfig",
    "DEFAULT_A", "DEFAULT_B", "DEFAULT_H", "DEFAULT_K",
    "edge_signal", "sine_activation", "sine_derivative", "sine_partials",
    "CSR", "Backend", "NodeParams", "PythonBackend",
    "END_LABEL", "START_LABEL", "WINDOW",
    "END", "START", "RadixCyclicGraph",
    "PathResult", "dijkstra_predict", "sample_walk",
]
