"""Lattice - the 3D matrix: one :class:`Edge` for every ``(source, symbol, target)``.

The matrix is dense.  A machine of ``S`` states over an alphabet of ``A``
symbols holds exactly ``S * A * S`` edges from the moment it is made, every
one of them a full record (``edge.py``), and none is ever added or removed.
``lattice[s, a, t]`` is the edge; ``lattice.row(s, a)`` is the fibre of
``S`` edges the machine chooses among when it is in ``s`` and reads ``a``.

Everything is indexed by integers.  The alphabet maps symbols (strings) to
their indices and back; the states are ``0 .. S - 1`` and carry a
:class:`State` record each (visits, when, whether accepting).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .edge import Edge, Weighting

__all__ = ["Lattice", "State"]


@dataclass(slots=True)
class State:
    """One state of the machine: a cell's worth of bookkeeping beside the edges that leave it."""

    index: int
    accepting: bool = False
    visits: int = 0
    last_visited: int = -1

    def to_list(self) -> list:
        return [self.index, self.accepting, self.visits, self.last_visited]

    @classmethod
    def from_list(cls, values: list) -> State:
        index, accepting, visits, last_visited = values
        return cls(int(index), bool(accepting), int(visits), int(last_visited))


class Lattice:
    """The dense ``S x A x S`` matrix of edges, with the alphabet and the states around it."""

    def __init__(self, states: int, alphabet: Sequence[str], prototype: Weighting | None = None) -> None:
        if states < 1:
            raise ValueError(f"a machine needs at least one state, got {states}")
        symbols = list(alphabet)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("the alphabet must be a non-empty sequence of distinct symbols")
        self.n_states = int(states)
        self.symbols: list[str] = symbols
        self.index: dict[str, int] = {s: i for i, s in enumerate(symbols)}
        self.prototype = prototype.copy() if prototype is not None else Weighting()
        """The weighting every edge starts from; each adapts its own copy."""
        self.states: list[State] = [State(i) for i in range(self.n_states)]
        self.state_ids: list[int] = list(range(self.n_states))
        """Which state, by the number it was made with, sits at each position: rearranging moves them."""
        S, A = self.n_states, len(symbols)
        self._A = A
        self.edges: list[Edge] = [
            Edge(s, a, t, weighting=self.prototype.copy())
            for s in range(S) for a in range(A) for t in range(S)
        ]
        """Row-major: ``edges[(s * A + a) * S + t]`` is ``(s, a, t)``."""

    # ---- shape ------------------------------------------------------------------------------------------------------

    @property
    def n_symbols(self) -> int:
        return self._A

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(states, symbols, states)``: the three dimensions of the matrix."""
        return (self.n_states, self._A, self.n_states)

    def __len__(self) -> int:
        return len(self.edges)

    def symbol_index(self, symbol: str) -> int:
        try:
            return self.index[symbol]
        except KeyError:
            raise KeyError(f"{symbol!r} is not in the alphabet {self.symbols}") from None

    # ---- indexing ---------------------------------------------------------------------------------------------------

    def offset(self, source: int, symbol: int, target: int) -> int:
        S = self.n_states
        if not (0 <= source < S and 0 <= target < S and 0 <= symbol < self._A):
            raise IndexError(f"({source}, {symbol}, {target}) is outside the matrix {self.shape}")
        return (source * self._A + symbol) * S + target

    def __getitem__(self, key: tuple[int, int | str, int]) -> Edge:
        source, symbol, target = key
        if isinstance(symbol, str):
            symbol = self.symbol_index(symbol)
        return self.edges[self.offset(source, symbol, target)]

    def row(self, source: int, symbol: int | str) -> list[Edge]:
        """The ``S`` edges leaving ``source`` on ``symbol``: what the machine chooses among."""
        if isinstance(symbol, str):
            symbol = self.symbol_index(symbol)
        start = self.offset(source, symbol, 0)
        return self.edges[start:start + self.n_states]

    def leaving(self, source: int) -> list[Edge]:
        """Every edge leaving ``source``, on any symbol."""
        start = self.offset(source, 0, 0)
        return self.edges[start:start + self._A * self.n_states]

    def arriving(self, target: int) -> Iterator[Edge]:
        """Every edge arriving at ``target``, on any symbol, from any state."""
        return (e for e in self.edges[target::self.n_states])

    def touched(self) -> Iterator[Edge]:
        """The edges anything was ever written to."""
        return (e for e in self.edges if e.touched)

    def __iter__(self) -> Iterator[Edge]:
        return iter(self.edges)

    # ---- rearranging ------------------------------------------------------------------------------------------------

    def swap_states(self, i: int, j: int) -> None:
        """States ``i`` and ``j`` trade places: every edge from or to one is moved to the other's row and column, the
        two state records trade places, and nothing else changes.  Edge objects keep their identity."""
        if i == j:
            return
        S, A = self.n_states, self._A
        perm = list(range(S))
        perm[i], perm[j] = j, i
        new = [None] * len(self.edges)
        for e in self.edges:
            e.source, e.target = perm[e.source], perm[e.target]
            new[(e.source * A + e.symbol) * S + e.target] = e
        self.edges = new
        self.states[i], self.states[j] = self.states[j], self.states[i]
        self.states[i].index, self.states[j].index = i, j
        self.state_ids[i], self.state_ids[j] = self.state_ids[j], self.state_ids[i]

    def swap_symbols(self, a: int, b: int) -> None:
        """Symbols ``a`` and ``b`` trade places: their slices of the matrix and their labels."""
        if a == b:
            return
        S, A = self.n_states, self._A
        new = [None] * len(self.edges)
        for e in self.edges:
            if e.symbol == a:
                e.symbol = b
            elif e.symbol == b:
                e.symbol = a
            new[(e.source * A + e.symbol) * S + e.target] = e
        self.edges = new
        self.symbols[a], self.symbols[b] = self.symbols[b], self.symbols[a]
        self.index = {s: k for k, s in enumerate(self.symbols)}

    def state_load(self) -> list[float]:
        """How busy each state is: the traversals of every edge leaving it and every edge arriving at it."""
        load = [0.0] * self.n_states
        for e in self.edges:
            if e.seen:
                load[e.source] += e.seen
                load[e.target] += e.seen
        return load

    def symbol_load(self) -> list[float]:
        """How busy each symbol is: the traversals of every edge in its slice."""
        load = [0.0] * self._A
        for e in self.edges:
            if e.seen:
                load[e.symbol] += e.seen
        return load

    # ---- persistence ------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """The matrix as JSON: the shape, the alphabet, the states, and only the edges that were written to."""
        return {
            "states": self.n_states,
            "alphabet": list(self.symbols),
            "prototype": self.prototype.to_list(),
            "state_records": [s.to_list() for s in self.states],
            "state_ids": list(self.state_ids),
            "edges": [e.to_list() for e in self.touched()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Lattice:
        lattice = cls(int(data["states"]), data["alphabet"], Weighting.from_list(data["prototype"]))
        for values in data["state_records"]:
            state = State.from_list(values)
            lattice.states[state.index] = state
        ids = data.get("state_ids")
        if ids is not None:
            if sorted(ids) != list(range(lattice.n_states)):
                raise ValueError("state_ids must be a permutation of the states")
            lattice.state_ids = [int(x) for x in ids]
        for values in data["edges"]:
            edge = Edge.from_list(values)
            lattice.edges[lattice.offset(edge.source, edge.symbol, edge.target)] = edge
        return lattice

    def reset_edges(self, edges: Iterable[Edge]) -> None:
        """Put fresh edges (from the prototype) in place of ``edges``."""
        for e in edges:
            self.edges[self.offset(e.source, e.symbol, e.target)] = Edge(
                e.source, e.symbol, e.target, weighting=self.prototype.copy())
