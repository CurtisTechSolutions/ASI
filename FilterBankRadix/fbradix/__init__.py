"""FilterBankRadix: an activation filter in front of a bank of radix trees.

    from fbradix import FilteredRadixBank, FilterRouter, corpus

    data = corpus.load()
    train, test = corpus.split(data)
    bank = FilteredRadixBank(FilterRouter(bits=2), depth=5,
                             alphabet=corpus.alphabet(data))
    bank.fit(train, rounds=3, test=test)
    print(bank.bits_per_char(test))

``DESIGN.md`` is the specification, ``README.md`` is what it measured, and
``python3 -m fbradix.cli --help`` lists every command.
"""

from .activation import sine, sine_partials, tanh_partials
from .bank import (
    ConstantRouter,
    CycleRouter,
    FilteredRadixBank,
    FilterRouter,
    OracleRouter,
    router_from_dict,
)
from .checkpoint import CheckpointManager
from .filter import ActivationFilter, features
from .store import __version__, read_json, write_json_atomic
from .tree import RadixTreeNet

__all__ = [
    "ActivationFilter",
    "CheckpointManager",
    "ConstantRouter",
    "CycleRouter",
    "FilteredRadixBank",
    "FilterRouter",
    "OracleRouter",
    "RadixTreeNet",
    "__version__",
    "features",
    "read_json",
    "router_from_dict",
    "write_json_atomic",
    "sine",
    "sine_partials",
    "tanh_partials",
]
