"""Game theory, as the actual decision procedure.

Not an analogy. Planning under uncertainty *is* a game against Nature: the rows
are goals the system could pursue, the columns are the states the world might be
in, and the entries are payoffs estimated from memory. Once the problem is
written in that form, the choice of *how* to choose stops being a vibe and
becomes a named solution concept with known properties:

- `maximin` -- best worst case. Correct when a failure is unrecoverable.
- `minimax_regret` -- Savage. Minimise the most you could wish you had done
  otherwise. This is the default, because regret is the quantity memory can
  actually estimate: every trace records what happened next.
- `expected_utility` -- correct when the belief over states is trustworthy,
  which early on it is not.
- `fictitious_play` -- the mixed equilibrium of a zero-sum game, for when the
  "opponent" adapts (a flaky oracle, a moving spec, an adversarial reviewer).
- `RegretMatching` -- Hart & Mas-Colell. No-regret learning over repeated
  choices; the average play converges to a coarse correlated equilibrium. This
  is how the system picks decomposition policies over a long run.
- `shapley` -- which subgoal actually mattered. The unique credit assignment
  satisfying efficiency, symmetry, null-player and additivity, which is why it
  is the one used rather than "the last step before it worked".

Everything here is exact, dependency-free, and unit tested against results with
known closed forms (matching pennies -> (1/2,1/2); an additive game -> Shapley
values equal to the individual contributions).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

Matrix = list[list[float]]


# --------------------------------------------------------------------------- #
# decision rules under uncertainty: rows = actions, cols = states of the world
# --------------------------------------------------------------------------- #

def _check(m: Matrix) -> None:
    if not m or not m[0]:
        raise ValueError("empty payoff matrix")
    if len({len(r) for r in m}) != 1:
        raise ValueError("ragged payoff matrix")


def maximin(m: Matrix) -> int:
    """Wald. Index of the action whose worst outcome is best."""
    _check(m)
    return max(range(len(m)), key=lambda i: min(m[i]))


def regret_matrix(m: Matrix) -> Matrix:
    """r[i][j] = (best any action achieves in state j) - m[i][j]. Non-negative,
    zero where the action was the right one for that state."""
    _check(m)
    best = [max(row[j] for row in m) for j in range(len(m[0]))]
    return [[best[j] - m[i][j] for j in range(len(m[0]))] for i in range(len(m))]


def minimax_regret(m: Matrix) -> int:
    r = regret_matrix(m)
    return min(range(len(r)), key=lambda i: max(r[i]))


def expected_utility(m: Matrix, belief: list[float]) -> list[float]:
    _check(m)
    if len(belief) != len(m[0]):
        raise ValueError("belief does not cover the state space")
    total = sum(belief)
    if total <= 0:
        belief = [1.0 / len(belief)] * len(belief)      # no belief is uniform belief
        total = 1.0
    return [sum(m[i][j] * belief[j] / total for j in range(len(m[0]))) for i in range(len(m))]


def hurwicz(m: Matrix, optimism: float = 0.5) -> int:
    """Interpolates maximin (optimism 0) and maximax (optimism 1)."""
    _check(m)
    return max(range(len(m)), key=lambda i: optimism * max(m[i]) + (1 - optimism) * min(m[i]))


@dataclass
class Decision:
    index: int
    rule: str
    scores: list[float]
    explanation: str


def choose(m: Matrix, belief: list[float] | None = None, aversion: float = 0.5) -> Decision:
    """Blend expected utility and minimax regret by `aversion`.

    The two rules answer different questions and the system does not always know
    which it is facing, so it interpolates on the normalised scores rather than
    switching hard at a threshold. At aversion 0 it maximises expected payoff; at
    1 it minimises worst-case regret; in between it trades the two smoothly and
    the blend is reported, so a decision is always attributable to a rule.
    """
    _check(m)
    n_states = len(m[0])
    belief = belief or [1.0 / n_states] * n_states
    eu = expected_utility(m, belief)
    worst_regret = [max(row) for row in regret_matrix(m)]
    eu_n = _unit(eu)
    reg_n = _unit([-w for w in worst_regret])          # negate: less regret is better
    scores = [(1 - aversion) * a + aversion * b for a, b in zip(eu_n, reg_n)]
    idx = max(range(len(scores)), key=lambda i: scores[i])
    rule = "expected-utility" if aversion < 0.25 else "minimax-regret" if aversion > 0.75 else "blended"
    return Decision(idx, rule, scores,
                    f"{rule}: eu={eu[idx]:.3f} worst-regret={worst_regret[idx]:.3f}")


def _unit(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


# --------------------------------------------------------------------------- #
# equilibria
# --------------------------------------------------------------------------- #

def fictitious_play(m: Matrix, iterations: int = 5000) -> tuple[list[float], list[float], float]:
    """Brown's fictitious play on a zero-sum game. `m` is the row player's payoff.

    Each side best-responds to the empirical distribution of the other's past
    play. Robinson proved this converges to the value of the game for zero-sum
    matrices; convergence is O(1/sqrt(t)), so the iteration count buys accuracy
    rather than correctness. Returns (row mixed strategy, column mixed strategy,
    game value).
    """
    _check(m)
    rows, cols = len(m), len(m[0])
    row_counts = [0] * rows
    col_counts = [0] * cols
    row_payoff = [0.0] * rows       # row's payoff against col's empirical play
    col_payoff = [0.0] * cols       # col's payoff (= -row's) against row's
    for _ in range(iterations):
        i = max(range(rows), key=lambda r: row_payoff[r])
        row_counts[i] += 1
        for c in range(cols):
            col_payoff[c] -= m[i][c]
        j = max(range(cols), key=lambda c: col_payoff[c])
        col_counts[j] += 1
        for r in range(rows):
            row_payoff[r] += m[r][j]
    p = [c / iterations for c in row_counts]
    q = [c / iterations for c in col_counts]
    value = sum(p[i] * m[i][j] * q[j] for i in range(rows) for j in range(cols))
    return p, q, value


@dataclass
class RegretMatching:
    """No-regret learning over a fixed set of actions (Hart & Mas-Colell 2000).

    Play proportional to positive cumulative regret; where there is none yet,
    play uniformly. The *average* strategy converges to a coarse correlated
    equilibrium, and the per-round strategy is what gets sampled. Used for the
    long-run choice of decomposition policy and experiment class, where the
    payoff is non-stationary because the system itself is changing -- which is
    exactly the regime a stationary bandit assumption gets wrong.
    """
    actions: list[str]
    regret: list[float] = field(default_factory=list)
    cumulative: list[float] = field(default_factory=list)
    rounds: int = 0

    def __post_init__(self) -> None:
        if not self.regret:
            self.regret = [0.0] * len(self.actions)
        if not self.cumulative:
            self.cumulative = [0.0] * len(self.actions)

    def strategy(self) -> list[float]:
        positive = [max(0.0, r) for r in self.regret]
        total = sum(positive)
        if total <= 0:
            return [1.0 / len(self.actions)] * len(self.actions)
        return [p / total for p in positive]

    def sample(self, rng: random.Random) -> int:
        s = self.strategy()
        roll, acc = rng.random(), 0.0
        for i, p in enumerate(s):
            acc += p
            if roll <= acc:
                return i
        return len(s) - 1

    def observe(self, utilities: list[float], played: int | None = None) -> None:
        """Counterfactual regret: for each action, what it would have paid minus
        what was actually paid. With full information the actual payoff is the
        strategy's expectation; with bandit feedback, pass the action played."""
        s = self.strategy()
        actual = utilities[played] if played is not None else sum(p * u for p, u in zip(s, utilities))
        for i, u in enumerate(utilities):
            self.regret[i] += u - actual
        self.cumulative = [c + p for c, p in zip(self.cumulative, s)]
        self.rounds += 1

    def average_strategy(self) -> list[float]:
        total = sum(self.cumulative)
        if total <= 0:
            return [1.0 / len(self.actions)] * len(self.actions)
        return [c / total for c in self.cumulative]

    def to_json(self) -> dict:
        return {"actions": self.actions, "regret": self.regret,
                "cumulative": self.cumulative, "rounds": self.rounds}

    @classmethod
    def from_json(cls, d: dict) -> "RegretMatching":
        return cls(actions=d["actions"], regret=d.get("regret", []),
                   cumulative=d.get("cumulative", []), rounds=d.get("rounds", 0))


# --------------------------------------------------------------------------- #
# credit
# --------------------------------------------------------------------------- #

def shapley(players: list[str], value: "callable", samples: int = 400,
            rng: random.Random | None = None) -> dict[str, float]:
    """Monte Carlo Shapley values by permutation sampling.

    `value(subset) -> float` is the coalition function: here, "does the parent
    goal pass with only these subgoals?". Exact Shapley is 2^n evaluations, so
    the estimator samples permutations and averages marginal contributions --
    unbiased, with error falling as 1/sqrt(samples). Below eight players it just
    enumerates, because at that size exactness is cheaper than the argument
    about whether the sample was large enough.

    This is the same credit mechanism `CyclicCortex` applies over regions; it is
    applied here over subgoals, which is the object whose contribution the
    reasoning layer needs to know.
    """
    rng = rng or random.Random(0)
    n = len(players)
    if n == 0:
        return {}
    contributions = {p: 0.0 for p in players}
    if n <= 8:
        import itertools
        perms = list(itertools.permutations(range(n)))
    else:
        perms = [rng.sample(range(n), n) for _ in range(samples)]
    cache: dict[frozenset, float] = {}

    def v(subset: frozenset) -> float:
        if subset not in cache:
            cache[subset] = value([players[i] for i in sorted(subset)])
        return cache[subset]

    for perm in perms:
        running: set[int] = set()
        before = v(frozenset())
        for idx in perm:
            running.add(idx)
            after = v(frozenset(running))
            contributions[players[idx]] += after - before
            before = after
    return {p: c / len(perms) for p, c in contributions.items()}


# --------------------------------------------------------------------------- #
# information
# --------------------------------------------------------------------------- #

def entropy(p: float) -> float:
    """Binary entropy in bits. Maximised at p = 0.5 -- which is the entire
    argument for running experiments you expect to fail half the time."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def information_gain(p_success: float, cost: float = 1.0) -> float:
    """Bits per unit cost. The ranking function for experiments (8.2).

    A thing you are certain will work teaches nothing. A thing you are certain
    will fail teaches nothing either -- and this is the half people skip. The
    experiment worth running is the one you genuinely cannot call, and the
    cheapest such experiment first. GREN derives the same 0.5 operating point
    from the same argument for probe policies; this is that result applied to
    self-improvement instead of to game identification.
    """
    return entropy(p_success) / max(cost, 1e-6)
