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
* **focus** - a number in ``[0, 1]`` that picks a node of the matrix's central
  vertical vector (``geometry.py``), and so the state every run starts from:
  none or low the top node (state 0), high the bottom one, ``0.5`` the
  central node.  Without a focus a run starts from ``start``.
* **compression** - every ``compress_every`` transitions the matrix is folded
  into its central node (``compress.py``) and the code kept on the machine
  (:attr:`Machine.core`); with ``compress_rebuild`` the matrix is then rebuilt
  from the code, so a lossy ``compress_precision`` is applied, not only
  recorded.

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

__all__ = ["BASELINE", "DEFAULT_ALPHABET", "DEFAULT_STATES", "DISCOUNT", "LIFE", "Machine", "Run", "Transition",
           "load_machine"]

LIFE = 1_000
"""Ticks for a trace, an age, a width's excess or the stimulation's excess to fade by a half."""

BASELINE = 1.0
"""The stimulation a machine rests at: width counts in proportion."""

DISCOUNT = 0.8
"""How much of a run's credit each step further back from its end receives."""

DEFAULT_STATES = 13
"""The states of a machine made without saying how many: with :data:`DEFAULT_ALPHABET`, a 13 x 13 x 13 matrix."""

DEFAULT_ALPHABET = "abcdefghijklm"
"""The alphabet of a machine made without saying which: thirteen symbols, ``a`` to ``m``.  The languages are over
``a`` and ``b``, so a default machine can be taught any of them; the other eleven symbols' edges wait unused."""

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
    skipped: int | None = None
    """A skip: the node passed through without stopping, the step reading two symbols (``symbol`` holds both)."""


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
        states: int = DEFAULT_STATES,
        alphabet: Sequence[str] = DEFAULT_ALPHABET,
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
        compress_every: int = 0,
        compress_precision: str = "exact",
        compress_rebuild: bool = False,
        focus: float | None = None,
        learn_focus: bool = False,
        focus_rate: float = 0.05,
        focus_explore: float = 0.15,
        skip: bool = False,
        skip_margin: float = 0.0,
        rearrange_every: int = 0,
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
        if compress_every < 0:
            raise ValueError(f"compress_every must be >= 0 (0 never), got {compress_every}")
        if compress_precision not in ("exact", "float32", "float16"):
            raise ValueError(f"compress_precision must be exact, float32 or float16, got {compress_precision!r}")
        from .focus import FocusLearner
        from .geometry import check_focus
        self._focus = check_focus(focus)
        if skip_margin < 0:
            raise ValueError(f"skip_margin must be >= 0, got {skip_margin}")
        if rearrange_every < 0:
            raise ValueError(f"rearrange_every must be >= 0 (0 never), got {rearrange_every}")
        self.learn_focus = bool(learn_focus)
        """Read the focus off the input through :attr:`focus_learner` instead of taking it as a setting."""
        self.focus_learner = FocusLearner(list(self.lattice.symbols), rate=focus_rate, explore=focus_explore)
        self._last_focus: tuple[float, float, list[float]] | None = None
        self.skip = bool(skip)
        """Take a skip - two symbols in one move, past the node between - when it is more efficient (``_decide``)."""
        self.skip_margin = float(skip_margin)
        self.skips = 0
        self.rearrange_every = int(rearrange_every)
        """Let the nodes rearrange themselves every this many transitions: one pass of swaps toward the centre."""
        self.swaps = 0
        self.since_rearrange = 0
        self._rearrange_due = False
        self.compress_every = int(compress_every)
        """Fold the matrix into its central node every this many transitions; 0 never."""
        self.compress_precision = compress_precision
        self.compress_rebuild = bool(compress_rebuild)
        self.core = None
        """The latest code the central node holds (:meth:`compress_now`)."""
        self.compressions = 0
        self.since_compression = 0
        self.last_compressed = -1
        self.rng = random.Random(seed)
        self.clock = 0
        """Every traversal, plus time let pass."""
        self.state = self.start
        self.path: list[Edge] = []
        """The edges of the current run, in order: what credit lands on."""
        self.last_focus: float | None = None
        """The focus the latest run started from, when it had one."""
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
    def focus(self) -> float | None:
        """The focus, a number in ``[0, 1]``, or ``None``: which node of the central vertical vector runs start from."""
        return self._focus

    @focus.setter
    def focus(self, value: float | None) -> None:
        from .geometry import check_focus
        self._focus = check_focus(value)

    def focus_state(self, focus: float | None) -> int:
        """The state ``focus`` starts a run from: its node on the central vertical vector."""
        from .geometry import focus_index
        return focus_index(self.n_states, focus)

    @property
    def origin(self) -> int:
        """Where a run starts: the focus node's state when there is a focus, else ``start``."""
        return self.start if self._focus is None else self.focus_state(self._focus)

    @property
    def center_state(self) -> int:
        """The middle state: the source and target of the matrix's central node (``geometry.center``)."""
        return self.n_states // 2

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
            self._settle_rearrange()
        return t

    def _traverse(self, edge: Edge, visit: bool = True) -> None:
        edge.traverse(self.clock, self.life, trace=self.trace, widen=self.use_widening)
        self.path.append(edge)
        self.clock += 1
        if visit:
            s = self.lattice.states[edge.target]
            s.visits += 1
            s.last_visited = self.clock
        self._after_transition()

    def _after_transition(self) -> None:
        """Count a transition; let the nodes rearrange themselves when ``rearrange_every`` have passed, and fold the
        matrix into its central node when ``compress_every`` have."""
        self.since_compression += 1
        self.since_rearrange += 1
        if self.rearrange_every and self.since_rearrange >= self.rearrange_every:
            self._rearrange_due = True
        if self.compress_every and self.since_compression >= self.compress_every:
            self.compress_now()

    # ---- rearranging ---------------------------------------------------------------------------------------------

    def swap_states(self, i: int, j: int) -> None:
        """States ``i`` and ``j`` trade places in the matrix.  A relabelling: every edge, the two state records, the
        start state, the current state and the run's path go with them, so the machine walks exactly as before from
        its start state.  What changes is where they are: which one the focus and the middle pick, which shell they
        sit in."""
        if not (0 <= i < self.n_states and 0 <= j < self.n_states):
            raise IndexError(f"states {i} and {j} are not both in 0..{self.n_states - 1}")
        if i == j:
            return
        self.lattice.swap_states(i, j)
        swap = {i: j, j: i}
        self.start = swap.get(self.start, self.start)
        self.state = swap.get(self.state, self.state)
        self.swaps += 1

    def swap_symbols(self, a: int | str, b: int | str) -> None:
        """Symbols ``a`` and ``b`` trade places: their slices and their labels.  Strings read the same as before."""
        a = self.lattice.symbol_index(a) if isinstance(a, str) else a
        b = self.lattice.symbol_index(b) if isinstance(b, str) else b
        if a == b:
            return
        self.lattice.swap_symbols(a, b)
        self.focus_learner.swap_symbols(a, b)
        self.swaps += 1

    def rearrange(self, full: bool = False, axes: Sequence[str] = ("states", "symbols")) -> list[tuple[str, int, int]]:
        """Let the nodes rearrange themselves toward the centre: one pass of swaps (``full``: passes until none is
        left), on the state axis and the symbol axis.

        On each axis, every pair of neighbours from the outside in swaps when the one farther from the centre is
        busier (``Lattice.state_load``, ``symbol_load``), so the busy nodes move inward a step per pass and the
        settled order is busiest at the centre, falling away on both sides.  Returns the swaps made."""
        done: list[tuple[str, int, int]] = []
        while True:
            made = []
            for axis in axes:
                if axis == "states":
                    made += [("states", i, j) for i, j in _inward_swaps(self.lattice.state_load(), self.swap_states)]
                elif axis == "symbols":
                    made += [("symbols", i, j) for i, j in _inward_swaps(self.lattice.symbol_load(), self.swap_symbols)]
                else:
                    raise ValueError(f"axes are states and symbols, got {axis!r}")
            done += made
            if not full or not made:
                break
        self.since_rearrange = 0
        return done

    def compress_now(self):
        """Fold the matrix into its central node now, at ``compress_precision``; with ``compress_rebuild``, rebuild
        the matrix from the code in place (every edge object is kept, so a run's path stays valid)."""
        from .compress import compress
        core = compress(self, precision=self.compress_precision)
        if self.compress_rebuild:
            core.rebuild_into(self.lattice)
        self.compressions += 1
        self.since_compression = 0
        self.last_compressed = self.clock
        self.core = core
        return core

    def run(self, symbols: Iterable[str], stimulation: float | None = None, temperature: float | None = None,
            quiet: bool = False, from_middle: bool = False, focus: float | None = None,
            skip: bool | None = None) -> Run:
        """Read a string.  It starts from the middle state (``from_middle``), else the node a ``focus`` given for this
        run picks, else - with ``learn_focus`` - the node the learned focus reads off the input, else :attr:`origin`
        (the machine's focus node, or ``start``).  With skips on (the machine's ``skip``, or ``skip`` for this run),
        two symbols are read in one move whenever that is more efficient.  Quiet, the same choices are made and
        nothing moves."""
        stim = self.stimulation if stimulation is None else float(stimulation)
        syms = [s for s in symbols if not (isinstance(s, str) and s.isspace())]
        learned = None
        if from_middle:
            origin = self.center_state
        elif focus is not None:
            origin = self.focus_state(focus)
        elif self.learn_focus:
            learned = self.focus_learner.choose(syms, lambda: self.rng.gauss(0.0, 1.0), explore=not quiet)
            origin = self.focus_state(self.focus_learner.clip(learned[0]))
        else:
            origin = self.origin
        do_skip = self.skip if skip is None else bool(skip)
        if not quiet:
            self.reset()
            self.state = origin
            self.runs += 1
            self._last_focus = learned
            self.last_focus = self.focus_learner.clip(learned[0]) if learned else (focus if focus is not None else self._focus)
        state = origin
        transitions: list[Transition] = []
        i = 0
        while i < len(syms):
            nxt = syms[i + 1] if do_skip and i + 1 < len(syms) else None
            move = self._decide(state, syms[i], nxt, stim, temperature)
            clock = self.clock
            if move[0] == "step":
                _, target, p = move
                if not quiet:
                    self._traverse(self.lattice[state, syms[i], target])
                    self.state = target
                transitions.append(Transition(state, syms[i], target, p, stim, clock))
                i += 1
            else:
                _, mid, target, p = move
                if not quiet:
                    self._traverse(self.lattice[state, syms[i], mid], visit=False)
                    self._traverse(self.lattice[mid, nxt, target])
                    self.state = target
                    self.skips += 1
                transitions.append(Transition(state, syms[i] + nxt, target, p, stim, clock, skipped=mid))
                i += 2
            state = self.state if not quiet else target
        if not quiet:
            state = self.state
        accepted = self.lattice.states[state].accepting
        if not quiet:
            self._settle_rearrange()
            state = self.state
        return Run(transitions, state, accepted)

    def _settle_rearrange(self) -> None:
        """A rearrangement that fell due during a run or a lesson happens when it is over, so nothing the run holds
        - a skip's target, the states it has passed - refers to a position that moved under it."""
        if self._rearrange_due:
            self._rearrange_due = False
            self.rearrange()

    def _decide(self, state: int, a: str, b: str | None, stim: float, temperature: float | None):
        """The next move from ``state``: ``("step", target, p)``, or ``("skip", passed, target, p)``.

        The machine draws its step on ``a`` as always.  With a next symbol ``b`` to look at, it compares the path
        that step begins - the step, then the best edge after it on ``b`` - with the best two-edge path on ``a``
        then ``b`` through any node, scoring each by its summed log-probability (at the run's temperature, or the
        machine's, or 1 when greedy).  When the best path beats the step's by more than ``skip_margin``, it skips:
        straight to that path's end, past the node between."""
        t1, p1 = self.choose(state, a, stim, temperature)
        if b is None:
            return ("step", t1, p1)
        temp = temperature if temperature is not None and temperature > 0 else (self.temperature or 1.0)
        pa = self.probabilities(state, a, stim, temp)
        best_t, best_u, best_v, step_v = 0, 0, -math.inf, -math.inf
        for t in range(self.n_states):
            pb = self.probabilities(t, b, stim, temp)
            u = max(range(len(pb)), key=pb.__getitem__)
            v = math.log(max(pa[t], 1e-300)) + math.log(max(pb[u], 1e-300))
            if v > best_v:
                best_t, best_u, best_v = t, u, v
            if t == t1:
                step_v = v
        if best_v > step_v + self.skip_margin:
            pb = self.probabilities(best_t, b, stim, temp)
            return ("skip", best_t, best_u, pa[best_t] * pb[best_u])
        return ("step", t1, p1)

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
        if path is None and self._last_focus is not None:
            self.focus_learner.learn(amount, *self._last_focus)
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
        self._after_transition()
        self._settle_rearrange()
        return edge

    # ---- measurements (quiet) ----------------------------------------------------------------------------------------

    def _focus_node(self) -> tuple[int, int, int]:
        from .geometry import focus_node
        return focus_node(self.lattice.shape, self._focus)

    def accepts(self, symbols: Iterable[str], temperature: float = 0.0, from_middle: bool = False) -> bool:
        """Whether the greedy (or, at a temperature, a sampled) quiet run ends in an accepting state."""
        return self.run(symbols, temperature=temperature, quiet=True, from_middle=from_middle).accepted

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
            "center": [self.center_state, len(self.alphabet) // 2, self.center_state],
            "focus": self._focus,
            "focus_node": list(self._focus_node()),
            "origin": self.origin,
            "learn_focus": self.learn_focus,
            "last_focus": self.last_focus,
            "skip": self.skip,
            "skip_margin": self.skip_margin,
            "skips": self.skips,
            "rearrange_every": self.rearrange_every,
            "swaps": self.swaps,
            "state_order": list(self.lattice.state_ids),
            "symbol_order": list(self.lattice.symbols),
            "compress_every": self.compress_every,
            "compress_precision": self.compress_precision,
            "compress_rebuild": self.compress_rebuild,
            "compressions": self.compressions,
            "since_compression": self.since_compression,
            "last_compressed": self.last_compressed,
            "core_bytes": self.core.bytes if self.core is not None else None,
        }

    # ---- persistence -------------------------------------------------------------------------------------------------

    def settings_dict(self) -> dict:
        """The settings as the machine file writes them."""
        return {
            "start": self.start, "life": self.life, "baseline": self.baseline, "calm": self.calm,
            "temperature": self.temperature, "discount": self.discount, "trace": self.trace,
            "use_widening": self.use_widening, "reward_widening": self.reward_widening,
            "punish_narrowing": self.punish_narrowing, "seed": self.seed,
            "compress_every": self.compress_every, "compress_precision": self.compress_precision,
            "compress_rebuild": self.compress_rebuild, "focus": self._focus, "learn_focus": self.learn_focus,
            "skip": self.skip, "skip_margin": self.skip_margin, "rearrange_every": self.rearrange_every,
        }

    def to_dict(self) -> dict:
        return {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "settings": self.settings_dict(),
            "clock": self.clock,
            "state": self.state,
            "runs": self.runs,
            "credits": self.credits,
            "compressions": self.compressions,
            "since_compression": self.since_compression,
            "last_compressed": self.last_compressed,
            "skips": self.skips,
            "swaps": self.swaps,
            "since_rearrange": self.since_rearrange,
            "focus_learner": self.focus_learner.to_dict(),
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
        m = cls._assemble(Lattice.from_dict(data["lattice"]), data, weight_fn)
        state = data.get("rng")
        if isinstance(state, dict) and "mersenne" in state:
            version, key, gauss = state["mersenne"]
            m.rng.setstate((version, tuple(key), gauss))
        # a file written by the Rust port carries that port's generator state: reseeded from the seed
        return m

    @classmethod
    def _assemble(cls, lattice: Lattice, data: dict, weight_fn: WeightFn | None = None) -> Machine:
        """A machine around ``lattice`` from the rest of a machine file (or a code's record), reseeded."""
        st = data["settings"]
        m = cls.__new__(cls)
        m.lattice = lattice
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
        from .focus import FocusLearner
        from .geometry import check_focus
        m._focus = check_focus(st.get("focus"))
        m.learn_focus = bool(st.get("learn_focus", False))
        m.skip = bool(st.get("skip", False))
        m.skip_margin = float(st.get("skip_margin", 0.0))
        m.rearrange_every = int(st.get("rearrange_every", 0))
        m.skips = int(data.get("skips", 0))
        m.swaps = int(data.get("swaps", 0))
        m.since_rearrange = int(data.get("since_rearrange", 0))
        m._rearrange_due = False
        fl = data.get("focus_learner")
        m.focus_learner = FocusLearner.from_dict(fl) if fl else FocusLearner(list(lattice.symbols))
        m._last_focus = None
        m.last_focus = None
        m.compress_every = int(st.get("compress_every", 0))
        m.compress_precision = st.get("compress_precision", "exact")
        m.compress_rebuild = bool(st.get("compress_rebuild", False))
        m.core = None
        m.compressions = int(data.get("compressions", 0))
        m.since_compression = int(data.get("since_compression", 0))
        m.last_compressed = int(data.get("last_compressed", -1))
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


def _inward_swaps(load: list[float], swap) -> list[tuple[int, int]]:
    """One pass of neighbour swaps toward the centre on an axis of ``len(load)`` nodes: from the outside in on each
    side, a pair swaps when the outer node is busier than the inner.  ``swap(i, j)`` makes each swap; ``load`` is
    kept in step.  Returns the pairs swapped."""
    n = len(load)
    c = n // 2
    made = []
    pairs = [(k - 1, k) for k in range(1, c + 1)] + [(k + 1, k) for k in range(n - 2, c - 1, -1)]
    for outer, inner in pairs:
        if load[outer] > load[inner]:
            swap(outer, inner)
            load[outer], load[inner] = load[inner], load[outer]
            made.append((min(outer, inner), max(outer, inner)))
    return made
