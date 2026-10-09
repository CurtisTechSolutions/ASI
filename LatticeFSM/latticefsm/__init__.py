"""latticefsm - a finite state machine over a dense 3D matrix of adaptive edges.

The reference implementation, pure Python, standard library only; the Rust
crate in ``../rust`` is the same model with the CLI and the HTTP server.  See
``DESIGN.md`` for the specification and ``README.md`` for what it measured.

    from latticefsm import Machine

    m = Machine(3, "ab", accepting=[0])
    run = m.run("abab")                 # traverses four edges, lands somewhere
    m.reward(1) if run.accepted else m.punish(1)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .edge import COEFFICIENT_LIMIT, FEATURES, WIDTH_MAX, WIDTH_MIN, WIDTH_REST, Edge, Weighting
from .languages import ALPHABET, LANGUAGES, Language, examples, language
from .lattice import Lattice, State
from .machine import BASELINE, DEFAULT_ALPHABET, DEFAULT_STATES, DISCOUNT, LIFE, Machine, Run, Transition, load_machine

__all__ = [
    "__version__", "Machine", "Run", "Transition", "Lattice", "State", "Edge", "Weighting", "Language",
    "load_machine", "examples", "language", "LANGUAGES", "ALPHABET",
    "LIFE", "BASELINE", "DISCOUNT", "DEFAULT_STATES", "DEFAULT_ALPHABET", "FEATURES", "WIDTH_REST", "WIDTH_MIN", "WIDTH_MAX", "COEFFICIENT_LIMIT",
]
