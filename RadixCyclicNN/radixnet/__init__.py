"""radixnet - a self-compressing cyclic-graph neural network (Radix Tree idea): the core model.

Pure Python (standard library only); ``torch`` is an optional accelerator that
is imported lazily.  See ``DESIGN.md`` for the full specification.

This package is the model and nothing else: the graph, the four kinds, the
encodings, training, the searches and the model file.  Everything built around
it - the command line, the HTTP API and the React frontend, the teaching loops,
the LLM clients, the agent and its tools, speech, images and voice, MCP - is
the ``modelkit`` package in ``../ModelKit``, which depends on this one and never
the other way round.  ``python -m radixnet`` still runs the command line: it
hands over to ``modelkit.cli``.
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
from .encoding import (
    ACOUSTIC, BACK_LABEL, CHARS, END_LABEL, PHONES, START_LABEL, SYLLABLES, THINK_LABEL, WINDOW, WORDS, Decoder, Encoder,
    Encoding, parse_encoding,
)
from .graph import BACK, END, FIRST, START, THINK, RadixCyclicGraph
from .beam import Prediction, beam_predict
from .countnet import CountRewardGraph, CountRewardNet
from .metacog import MetaLayer
from .model import GraphModel, RadixNet, TrainConfig, load_model, model_class, model_kinds, new_model
from .negative import NegativeGraph, NegativeNet
from .penalty import (
    DEFAULT_TRAVERSAL,
    TRAVERSALS,
    PenaltyCosts,
    PhasePenaltyCosts,
    phase_traversal_costs,
    resolve_traversal,
    traversal_costs,
)
from .phasesearch import phase_beam, phase_dijkstra, phase_walk
from .resonance import ResonantGraph, ResonantNet, trigram_phase
from .search import PathResult, dijkstra_predict, sample_walk
from .window import DEFAULT_FLOOR as WINDOW_FLOOR, DEFAULT_TOP as WINDOW_TOP, DynamicWindow

__all__ = [
    "__version__",
    "RadixNet", "TrainConfig", "GraphModel", "CountRewardNet", "CountRewardGraph",
    "NegativeNet", "NegativeGraph",
    "ResonantNet", "ResonantGraph", "MetaLayer", "trigram_phase",
    "load_model", "model_class", "model_kinds", "new_model", "Prediction", "beam_predict",
    "Encoder", "Decoder", "SineActivation",
    "get_backend", "describe_backends", "torch_available",
    "DEFAULT_A", "DEFAULT_B", "DEFAULT_H", "DEFAULT_K",
    "edge_signal", "sine_activation", "sine_derivative", "sine_partials",
    "CSR", "Backend", "NodeParams", "PythonBackend",
    "END_LABEL", "START_LABEL", "WINDOW", "CHARS", "WORDS", "PHONES", "SYLLABLES", "ACOUSTIC", "Encoding", "parse_encoding",
    "END", "START", "RadixCyclicGraph",
    "DynamicWindow", "WINDOW_TOP", "WINDOW_FLOOR",
    "PathResult", "dijkstra_predict", "sample_walk",
    "phase_beam", "phase_dijkstra", "phase_walk",
    "TRAVERSALS", "DEFAULT_TRAVERSAL", "PenaltyCosts", "PhasePenaltyCosts",
    "traversal_costs", "phase_traversal_costs", "resolve_traversal",
]
