"""Machine - the finite state machine that walks the matrix.

A machine is in one state.  It reads a symbol, looks along the row of edges
``lattice.row(state, symbol)`` - one edge per possible next state - weighs
each by its own weighting function and the machine's stimulation, draws one,
traverses it, and is in that state.  A string of symbols is a run; the run
ends in a state that is accepting or not.  That is the whole of the
machine's behaviour, and everything else here is what feeds back into it:

* **the clock** - every traversal is one tick; ``tick(k)`` lets time pass
  with nothing traversed.  Traces, ages, widths and the stimulation all fade
  on it.
* **credit** - ``reward(r)`` and ``punish(r)`` credit the edges of the
  current run, the last one in full and each earlier one discounted by
  ``discount`` per step back, and the edge adapts its weighting and its
  width with what it receives.
* **stimulation** - a level the machine is at.  It multiplies the
  log-width term of every weight: the higher it is, the more a wide channel
  is preferred to a narrow one.  ``stimulate(x)`` raises it; it relaxes
  toward ``baseline`` by a half every ``calm`` ticks.
* **quiet** - any run asked for ``quiet=True`` draws without traversing,
  credits nothing and moves no clock: a measurement.

``weight_fn`` replaces the weighting altogether: a callable
``(edge, clock, life, stimulation) -> log-weight``, for a machine whose
edges are weighed by something other than their own adapted function.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .edge import Edge, Weighting
from .lattice import Lattice

__all__ = ["BASELINE", "DISCOUNT", "LIFE", "Machine", "Run", "Transition", "load_machine"]

LIFE = 1_000
"""Ticks for a trace, an age, a width's excess or the stimulation's excess to fade by a half."""

BASELINE = 1.0
"""The stimulation a machine rests at: width counts in proportion."""

DISCOUNT = 0.8
"""How much of a run's credit each step further back from its end receives."""

_FORMAT = "latticefsm-machine"
_FORMAT_VERSION = 1

WeightFn = Callable[[Edge, int, float, float], float]


@dataclass(slots=True)
class Transition:
    """One step the machine took (or, quietly, would have taken)."""

    source: int
    symbol: str
    target: int
    probability: float
    """The probability the step had among the row's options under the stimulation it was taken at."""
    stimulation: float
    clock: int


@dataclass(slots=True)
class Run:
    """A string read from the start state: the steps, where it ended, and whether that is accepting."""

    transitions: list[Transition]
    final: int
    accepted: bool

    @property
    def states(self) -> list[int]:
        """The states visited, the start first."""
        if not self.transitions:
            return [self.final]
        return [self.transitions[0].source] + [t.target for t in self.transitions]

    @property
    def log_probability(self) -> float:
        return sum(math.log(t.probability) for t in self.transitions)


class Machine:
    """A finite state machine over a dense 3D matrix of adaptive edges."""

    def __init__(
        self,
        states: int,
        alphabet: Sequence[str],
        accepting: Iterable[int] = (),
        start: int = 0,
        life: float = LIFE,
        baseline: float = BASELINE,
        calm: float | None = None,
        temperature: float = 1.0,
        discount: float = DISCOUNT,
        trace: float = 1.0,
        use_widening: float = 0.01,
        reward_widening: float = 0.2,
        punish_narrowing: float = 0.2,
        prototype: Weighting | None = None,
        weight_fn: WeightFn | None = None,
        seed: int = 1,
    ) -> None:
        if not (life > 0) or not math.isfinite(life):
            raise ValueError(f"life must be a positive number of ticks, got {life}")
        if baseline < 0:
            raise ValueError(f"baseline stimulation must be >= 0, got {baseline}")
        if not (0.0 <= discount <= 1.0):
            raise ValueError(f"discount must be in [0, 1], got {discount}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        self.lattice = Lattice(states, alphabet, prototype)
        if not (0 <= start < states):
            raise ValueError(f"start state {start} is not one of 0..{states - 1}")
        for s in accepting:
            self.lattice.states[s].accepting = True
        self.start = int(start)
        self.life = float(life)
        self.baseline = float(baseline)
        self.calm = float(calm) if calm is not None else float(life)
        """Ticks for the stimulation's excess over the baseline to fade by a half."""
        self.temperature = float(temperature)
        self.discount = float(discount)
        self.trace = float(trace)
        self.use_widening = float(use_widening)
        self.reward_widening = float(reward_widening)
        self.punish_narrowing = float(punish_narrowing)
        self.weight_fn = weight_fn
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.clock = 0
        """Every traversal, plus time let pass."""
        self.state = self.start
        self.path: list[Edge] = []
        """The edges of the current run, in order: what credit lands on."""
        self._stimulation = self.baseline
        self._stimulation_stamp = 0
        self.runs = 0
        self.credits = 0
        self.lattice.states[self.start].visits += 1
        self.lattice.states[self.start].last_visited = 0

    # ---- shape and lookups -------------------------------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return self.lattice.n_states

    @property
    def alphabet(self) -> list[str]:
        return self.lattice.symbols

    @property
    def accepting(self) -> list[int]:
        return [s.index for s in self.lattice.states if s.accepting]

    def edge(self, source: int, symbol: int | str, target: int) -> Edge:
        return self.lattice[source, symbol, target]

    # ---- stimulation -------------------------------------------------------------------------------------------------

    @property
    def stimulation(self) -> float:
        """The level now: the baseline plus whatever excess is left, halved every ``calm`` ticks."""
        excess = self._stimulation - self.baseline
        if excess == 0.0:
            return self.baseline
        return self.baseline + excess * 0.5 ** ((self.clock - self._stimulation_stamp) / self.calm)

    @stimulation.setter
    def stimulation(self, level: float) -> None:
        if level < 0:
            raise ValueError(f"stimulation must be >= 0, got {level}")
        self._stimulation = float(level)
        self._stimulation_stamp = self.clock

    def stimulate(self, amount: float) -> float:
        """Raise (or, negative, lower) the stimulation by ``amount`` from where it is now; returns the new level."""
        self.stimulation = max(0.0, self.stimulation + amount)
        return self.stimulation

    # ---- weighing and choosing ---------------------------------------------------------------------------------------

    def log_weights(self, source: int, symbol: int | str, stimulation: float | None = None) -> list[float]:
        """The log-weight of every edge in the row ``(source, symbol)``, by target."""
        stim = self.stimulation if stimulation is None else float(stimulation)
        row = self.lattice.row(source, symbol)
        if self.weight_fn is not None:
            return [self.weight_fn(e, self.clock, self.life, stim) for e in row]
        return [e.log_weight(self.clock, self.life, stim) for e in row]

    def probabilities(self, source: int, symbol: int | str, stimulation: float | None = None,
                      temperature: float | None = None) -> list[float]:
        """The row's log-weights as a distribution over targets: softmax at ``temperature``; greedy at zero."""
        return _softmax(self.log_weights(source, symbol, stimulation),
                        self.temperature if temperature is None else float(temperature))

    def choose(self, source: int, symbol: int | str, stimulation: float | None = None,
               temperature: float | None = None) -> tuple[int, float]:
        """Draw a target from the row, returning it and the probability it had."""
        probs = self.probabilities(source, symbol, stimulation, temperature)
        r = self.rng.random()
        acc = 0.0
        for t, p in enumerate(probs):
            acc += p
            if r < acc:
                return t, p
        t = max(range(len(probs)), key=probs.__getitem__)
        return t, probs[t]

    # ---- moving ------------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        """Back to the start state with an empty path.  Not a tick."""
        self.state = self.start
        self.path = []

    def step(self, symbol: str, stimulation: float | None = None, temperature: float | None = None,
             quiet: bool = False) -> Transition:
        """Read one symbol: choose a next state from the current one, traverse the edge, and be there.

        Quiet, the choice is made and reported and nothing moves: not the
        state, not the edge, not the clock.
        """
        stim = self.stimulation if stimulation is None else float(stimulation)
        source = self.state
        target, p = self.choose(source, symbol, stim, temperature)
        t = Transition(source, symbol, target, p, stim, self.clock)
        if not quiet:
            self._traverse(self.lattice[source, symbol, target])
            self.state = target
        return t

    def _traverse(self, edge: Edge) -> None:
        edge.traverse(self.clock, self.life, trace=self.trace, widen=self.use_widening)
        self.path.append(edge)
        self.clock += 1
        s = self.lattice.states[edge.target]
        s.visits += 1
        s.last_visited = self.clock

    def run(self, symbols: Iterable[str], stimulation: float | None = None, temperature: float | None = None,
            quiet: bool = False) -> Run:
        """Read a string from the start state.  Quiet, a copy of the machine's state is walked and nothing moves."""
        stim = self.stimulation if stimulation is None else float(stimulation)
        if quiet:
            state = self.start
            transitions = []
            for sym in symbols:
                target, p = self.choose(state, sym, stim, temperature)
                transitions.append(Transition(state, sym, target, p, stim, self.clock))
                state = target
            return Run(transitions, state, self.lattice.states[state].accepting)
        self.reset()
        self.runs += 1
        transitions = [self.step(sym, stim, temperature) for sym in symbols]
        return Run(transitions, self.state, self.lattice.states[self.state].accepting)

    def tick(self, ticks: int = 1) -> None:
        """Let time pass: the clock advances with nothing traversed."""
        if ticks < 0:
            raise ValueError("time does not run backwards")
        self.clock += int(ticks)

    # ---- credit ------------------------------------------------------------------------------------------------------

    def credit(self, amount: float, path: Sequence[Edge] | None = None) -> int:
        """Credit the current run (or ``path``): its last edge receives ``amount``, each earlier one ``discount`` times
        the next.  Returns how many edges were credited."""
        edges = self.path if path is None else list(path)
        if not edges or amount == 0.0:
            return 0
        self.credits += 1
        share = float(amount)
        for edge in reversed(edges):
            edge.credit(share, self.clock, self.life, widen=self.reward_widening, narrow=self.punish_narrowing)
            share *= self.discount
            if abs(share) < 1e-12:
                break
        return len(edges)

    def reward(self, amount: float = 1.0) -> int:
        return self.credit(abs(amount))

    def punish(self, amount: float = 1.0) -> int:
        return self.credit(-abs(amount))

    def teach(self, source: int, symbol: str, target: int, amount: float = 1.0) -> Edge:
        """Traverse one edge deliberately and credit it ``amount``: a lesson rather than an experience."""
        edge = self.lattice[source, symbol, target]
        edge.traverse(self.clock, self.life, trace=self.trace, widen=self.use_widening)
        self.clock += 1
        edge.credit(amount, self.clock, self.life, widen=self.reward_widening, narrow=self.punish_narrowing)
        return edge

    # ---- measurements (quiet) ----------------------------------------------------------------------------------------

    def accepts(self, symbols: Iterable[str], temperature: float = 0.0) -> bool:
        """Whether the greedy (or, at a temperature, a sampled) quiet run ends in an accepting state."""
        return self.run(symbols, temperature=temperature, quiet=True).accepted

    def accuracy(self, examples: Iterable[tuple[Sequence[str], bool]], temperature: float = 0.0) -> float:
        """The share of ``(string, accept?)`` examples the quiet run classifies correctly."""
        right = total = 0
        for symbols, accept in examples:
            total += 1
            right += self.accepts(symbols, temperature) == bool(accept)
        return right / total if total else 0.0

    def transition_table(self, temperature: float = 0.0) -> dict[tuple[int, str], int]:
        """The deterministic machine the greedy walk is: ``(state, symbol) -> state``."""
        table = {}
        for s in range(self.n_states):
            for sym in self.alphabet:
                probs = self.probabilities(s, sym, temperature=temperature)
                table[(s, sym)] = max(range(len(probs)), key=probs.__getitem__)
        return table

    def stats(self) -> dict:
        lat = self.lattice
        touched = list(lat.touched())
        seen = [e for e in touched if e.seen]
        return {
            "shape": list(lat.shape),
            "edges": len(lat),
            "touched": len(touched),
            "traversed": len(seen),
            "clock": self.clock,
            "runs": self.runs,
            "credits": self.credits,
            "stimulation": self.stimulation,
            "baseline": self.baseline,
            "state": self.state,
            "accepting": self.accepting,
            "total_seen": sum(e.seen for e in seen),
            "total_rewarded": sum(e.rewarded for e in touched),
            "total_punished": sum(e.punished for e in touched),
            "widest": max((e.width_at(self.clock, self.life) for e in touched), default=1.0),
            "narrowest": min((e.width_at(self.clock, self.life) for e in touched), default=1.0),
        }

    # ---- persistence -------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "settings": {
                "start": self.start, "life": self.life, "baseline": self.baseline, "calm": self.calm,
                "temperature": self.temperature, "discount": self.discount, "trace": self.trace,
                "use_widening": self.use_widening, "reward_widening": self.reward_widening,
                "punish_narrowing": self.punish_narrowing, "seed": self.seed,
            },
            "clock": self.clock,
            "state": self.state,
            "runs": self.runs,
            "credits": self.credits,
            "stimulation": [self._stimulation, self._stimulation_stamp],
            "rng": {"mersenne": list(self.rng.getstate())},
            "lattice": self.lattice.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, weight_fn: WeightFn | None = None) -> Machine:
        """A machine from its JSON.  The generator state is each port's own: the Python port's is restored, and a
        file the Rust port wrote reseeds from the seed."""
        if data.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} file")
        lat = Lattice.from_dict(data["lattice"])
        st = data["settings"]
        m = cls.__new__(cls)
        m.lattice = lat
        m.start = int(st["start"])
        m.life = float(st["life"])
        m.baseline = float(st["baseline"])
        m.calm = float(st["calm"])
        m.temperature = float(st["temperature"])
        m.discount = float(st["discount"])
        m.trace = float(st["trace"])
        m.use_widening = float(st["use_widening"])
        m.reward_widening = float(st["reward_widening"])
        m.punish_narrowing = float(st["punish_narrowing"])
        m.weight_fn = weight_fn
        m.seed = int(st["seed"])
        m.rng = random.Random(m.seed)
        state = data.get("rng")
        if isinstance(state, dict) and "mersenne" in state:
            version, key, gauss = state["mersenne"]
            m.rng.setstate((version, tuple(key), gauss))
        # a file written by the Rust port carries that port's generator state: reseeded from the seed
        m.clock = int(data["clock"])
        m.state = int(data["state"])
        m.runs = int(data["runs"])
        m.credits = int(data["credits"])
        m.path = []
        m._stimulation, m._stimulation_stamp = float(data["stimulation"][0]), int(data["stimulation"][1])
        return m

    def save(self, path: str) -> None:
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)


def load_machine(path: str, weight_fn: WeightFn | None = None) -> Machine:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return Machine.from_dict(json.load(f), weight_fn)


def _softmax(logits: Sequence[float], temperature: float) -> list[float]:
    if temperature == 0.0:
        best = max(logits)
        winners = [i for i, v in enumerate(logits) if v == best]
        return [1.0 / len(winners) if i in winners else 0.0 for i in range(len(logits))]
    scaled = [v / temperature for v in logits]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]
