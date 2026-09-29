"""radixdecay - a radix tree that remembers what it has read and said, and forgets the rest.

Pure Python, standard library only.  See ``DESIGN.md`` for the specification
and ``README.md`` for what it measured.

    from radixdecay import DecayNet

    model = DecayNet()
    model.read(["the cat sat on the mat", "the dog sat on the log"])
    model.say("the cat", to_end=True).text      # and now it has said it, and remembers that too
"""

from __future__ import annotations

__version__ = "0.1.0"

from .encoding import CHARS, END_LABEL, PHONES, START_LABEL, SYLLABLES, WINDOW, Encoding
from .model import MAX_LEGS, MAX_SLIDE, UNKNOWN_PROB, DecayNet, load_model
from .search import PathResult, cheapest_path
from .tree import DECAYS, FIRST, LIFE, MIN_SEEN, ROOT, START, DecayTree, Walks

__all__ = [
    "__version__", "DecayNet", "DecayTree", "Encoding", "PathResult", "Walks", "cheapest_path", "load_model",
    "ROOT", "START", "FIRST", "LIFE", "MIN_SEEN", "DECAYS", "MAX_SLIDE", "MAX_LEGS", "UNKNOWN_PROB",
    "CHARS", "PHONES", "SYLLABLES", "WINDOW", "START_LABEL", "END_LABEL",
]
