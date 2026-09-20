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
from .encoding import (
    END_LABEL,
    MAX_WORDS,
    START_LABEL,
    UNKNOWN_WORD,
    WINDOW,
    Decoder,
    Encoder,
    Vocabulary,
    split_words,
    symbol_word,
    word_symbol,
)
from .gan import EvolveConfig, Evolver
from .graph import END, START, RadixCyclicGraph
from .beam import Prediction, beam_predict
from .countnet import CountRewardGraph, CountRewardNet
from .duo import FilterConfig, NegativeFilter
from .metacog import MetaLayer
from .model import GraphModel, RadixNet, TrainConfig, load_model, model_class, model_kinds, new_model
from .negative import NegativeGraph, NegativeNet
from .phasesearch import phase_beam, phase_dijkstra, phase_walk
from .resonance import ResonantGraph, ResonantNet, trigram_phase
from .wordnet import WordGraph, WordNGramNet
from .search import PathResult, dijkstra_predict, sample_walk
from .speech import (
    ASR_BACKENDS,
    CODECS as SPEECH_CODECS,
    DEFAULT_RATE as SPEECH_RATE,
    SPEECH_TOKEN,
    Audio,
    SpeechError,
    encode_audio,
    speech_texts,
    teach as teach_by_speech,
    transcribe,
    utterance_token,
)

__all__ = [
    "__version__",
    "RadixNet", "TrainConfig", "GraphModel", "CountRewardNet", "CountRewardGraph",
    "NegativeNet", "NegativeGraph", "NegativeFilter", "FilterConfig",
    "ResonantNet", "ResonantGraph", "MetaLayer", "trigram_phase",
    "WordNGramNet", "WordGraph", "Vocabulary", "split_words", "word_symbol", "symbol_word",
    "MAX_WORDS", "UNKNOWN_WORD",
    "load_model", "model_class", "model_kinds", "new_model", "Prediction", "beam_predict",
    "Encoder", "Decoder", "SineActivation",
    "get_backend", "describe_backends", "torch_available",
    "CheckpointManager", "Evolver", "EvolveConfig",
    "DEFAULT_A", "DEFAULT_B", "DEFAULT_H", "DEFAULT_K",
    "edge_signal", "sine_activation", "sine_derivative", "sine_partials",
    "CSR", "Backend", "NodeParams", "PythonBackend",
    "END_LABEL", "START_LABEL", "WINDOW",
    "SPEECH_TOKEN", "SPEECH_CODECS", "SPEECH_RATE", "ASR_BACKENDS", "Audio", "SpeechError",
    "encode_audio", "speech_texts", "teach_by_speech", "transcribe", "utterance_token",
    "END", "START", "RadixCyclicGraph",
    "PathResult", "dijkstra_predict", "sample_walk",
    "phase_beam", "phase_dijkstra", "phase_walk",
]
