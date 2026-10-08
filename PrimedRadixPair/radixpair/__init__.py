"""radixpair - two primed radix trees, connected at every equal node: one written by reading, one by what worked.

    from radixpair import prime, load_model
    model = prime("chars", L=4)                # every sequence of up to 4 units exists from the start
    model.train(["the cat sat on the mat"])    # counting, and nothing else
    model.reward("the cat sat on the mat")     # what worked; a punishment is a negative reward
    model.predict("the cat", 10).full_text
    model.score("the cat sat on the log").bits

Standard library only.  The design is in ``../DESIGN.md``; ``../PRD.md`` says why.
"""

from .address import Address, node_count
from .codec import (CODECS, BpeCodec, BytesCodec, CharsCodec, Codec, ExternalCodec, Gpt2Codec, PhonesCodec,
                    SyllablesCodec, codec_from_dict, make_codec)
from .count import CountTree
from .model import KINDS, MODEL_FORMAT, PairModel, load_model, prime
from .pair import BACKOFFS, RadixPair, Score, Settings, TRAVERSALS
from .reward import RUNGS, RewardTree
from .search import MODES, PathResult, dijkstra, greedy

__all__ = [
    "Address", "node_count", "CODECS", "Codec", "BpeCodec", "BytesCodec", "CharsCodec", "ExternalCodec", "Gpt2Codec",
    "PhonesCodec", "SyllablesCodec", "codec_from_dict", "make_codec", "CountTree", "RewardTree", "RUNGS", "RadixPair",
    "Score", "Settings", "TRAVERSALS", "BACKOFFS", "PathResult", "MODES", "dijkstra", "greedy", "PairModel", "KINDS",
    "MODEL_FORMAT", "load_model", "prime",
]
__version__ = "0.1.0"
