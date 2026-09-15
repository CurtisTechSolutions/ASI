# GTMNN — Design Specification

**G**ame **T**heory + **M**icro **N**eural **N**etworks.

There is no network here. There is a **population of thousands of tiny neural
networks**, and a **game they play against each other**. The answer the system
gives is the **equilibrium of that game**. Credit for a right or wrong answer is
not propagated backwards through layers — it is divided among the players by
**Shapley value**, the game-theoretic answer to "who actually contributed".
Players that keep losing money are **culled and replaced by mutated winners**.

Backpropagation is not used anywhere. Not as an approximation of it, not as a
special case of it — there is no chain rule spanning two micros, ever. The
deepest gradient in the system is two layers long, inside one micro, on roughly
a hundred and thirty floats.

Directory: `GTMNN/` (this directory). Python package: `gtmnn`.
Python 3.11+, standard library only. `torch` is an *optional* accelerator that
is imported lazily and never required. This document is the contract every
module is implemented against. Read it fully before writing code.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| Micro neural networks, not one network | A `Micro` is an `R -> H -> (K+1)` MLP: `R=8` inputs, `H=6` hidden, `K=4` symbols it is allowed to name plus `ABSTAIN`. ~133 floats each. The default population is `N=4096` of them, stored as flat parallel arrays (`MicroPool`), never as 4096 Python objects. |
| Partial information, like a cortical column | A micro does not see the input. It sees `R=8` fixed feature indices drawn at birth — its *receptive field*, its patch of cortex. Every micro is looking at a different slice of the same thing, which is what makes the game a game and not a vote. |
| Game theory | Every prediction is a **stage game**. Micros bid for a seat, seated micros choose an action, payoff depends on *what everyone else chose*, and the output is the equilibrium. The game catalogue (§8) is the congestion game, the second-price auction, the stag hunt, the prisoner's dilemma and rock-paper-scissors — each doing a specific job. |
| Specialisation, for free | The core game is a **congestion game**: the reward for a correct answer is *split among everyone who gave it*. A thousand micros all saying `"e"` earn a thousandth each. One micro that correctly calls something rare takes the whole pot. Nothing else in the design pushes micros apart; this does all of it. |
| Attention, as a market | Seats are sold in a **uniform-price auction** (§9). A micro bids its own confidence, the top `M=64` bidders are seated, and each pays the highest losing bid. Because the price does not depend on your own bid, bidding your true confidence is a dominant strategy — the micros are *calibrated by the mechanism*, not by a loss term. |
| Credit assignment without backprop | **Shapley value** (§12). `φ_i` = micro `i`'s average marginal contribution over all orders in which the coalition could have formed. Estimated by Monte-Carlo over permutations. Each micro then takes one local gradient step on its own weights to earn more; nothing crosses a micro boundary. |
| Accept vanishing gradients; they cannot occur | There is no depth to vanish through. Every gradient is two layers long and dies inside the micro that owns it. |
| Custom activation `-1 * sin(x / 3)` | Every neuron of every micro owns a parametric sine `f(x) = a*sin(b*(x-h)) + k`, initialised to `a=-1, b=1/3, h=0, k=0`. All four parameters are learnable per neuron, by the same Shapley-scaled rule as the weights. |
| Cycles are a feature | The inference game is *player-specific* (§3.2): a pure equilibrium is guaranteed to exist, but best-response dynamics is **not** guaranteed to reach it — it can cycle. The solver detects the cycle rather than iterating into it. |
| Metacognition | A detected cycle, or an equilibrium too close to call, **escalates to the meta-player** (§14): a second, small micro population playing a different game over the *statistics of the deliberation that just failed* — entropy, margin, cycle length, regret mass. The system thinking about how badly it is thinking. Recursion is depth-capped. |
| Constantly self-upgrading | `Evolver` (§17): micros with negative wealth are culled each generation, replaced by crossover-plus-mutation of the top earners, and the population is checked for **evolutionary stability** — whether a small invasion of mutants can out-earn the incumbents. |
| 2NRL | `two_nrl(bad, good)`: (1) train on garbage, (2) `invert()` the population, (3) fine-tune on correct data at a smaller learning rate. Inversion here means `a -> -a` on every sine *and* a sign flip of the payoff function — the population plays the anti-game for one pass. |
| Train & predict | `GTMNet.train / predict / generate / score` — the same four verbs as `RadixNet`, so the CLI, API and frontend of the two projects stay recognisably siblings. |
| CLI / API / React frontend | `gtmnn.cli` (argparse), `gtmnn.api` (stdlib `http.server`, JSON), `frontend/` (Vite + React). |
| Checkpointing / save / load | JSON (optionally gzip) model files, `CheckpointManager` with rotation + `latest` pointer + resume. |
| GPU acceleration / performance | `Backend` abstraction: `PythonBackend` (always available, flat `array('d')` storage) and `TorchBackend` (optional; the whole population is one batched `(N, R, H)` tensor contraction). |

---

## 2. Why a game

A neural network trained by backpropagation answers "who is responsible for this
error?" with the chain rule: responsibility is whatever the derivative says it
is. That answer is only available because the network is one differentiable
object. Break it into four thousand separate objects and the question becomes a
genuinely different one, and it has a genuinely different, older answer.

Shapley (1953) asked exactly this question — a coalition achieved something of
value `v(N)`, how much of it is each member's doing? — and proved there is
**exactly one** answer satisfying four axioms (§12.1). That uniqueness is the
whole reason this design is possible. Credit assignment is not being
approximated here; it is being computed, by the only formula that is fair.

The rest follows from taking the players seriously. If micros are players, they
need payoffs. If they have payoffs, they have strategies. If they have
strategies, the output is an equilibrium, and the character of that equilibrium
is now a *design surface* — which is where the congestion game, the auction and
the cycles come from.

### 2.1 What the structure buys, stated as theorems

These are not decorations. Each one is the reason a specific module is allowed
to be as simple as it is.

| Result | What it says here | Where it is used |
|---|---|---|
| **Rosenthal (1973)** — every congestion game is an exact potential game, has a pure-strategy Nash equilibrium, and has the finite improvement property. | With the true label known, every seated micro faces the *same* payoff function of `(action, load)`. The training game is therefore an exact potential game: best-response dynamics **cannot** cycle and terminates in finitely many steps at a pure equilibrium. | §11.2. Training never escalates to the meta-player. It does not need to. |
| **Milchtaich (1996)** — singleton congestion games with *player-specific*, non-increasing payoff functions always possess a pure Nash equilibrium, and a best-response improvement path to one exists from every profile; but arbitrary best-response paths may cycle. | At inference the label is unknown, so each micro scores actions by *its own* belief `q_i`. The game becomes player-specific. An equilibrium still exists — so the solver has a real target — but the path there can loop. | §11.3 and §14. This theorem is exactly the author's position in `Research/CyclesAreAFeature.md`, arrived at from the other direction: the cycle is not a failure of the solver, it is a property of the game, and the correct response is to leave the game rather than iterate harder. |
| **Hart & Mas-Colell (2000)** — regret matching drives the empirical distribution of play to the set of correlated equilibria. | Gives a solver that converges on *every* game in the catalogue, cyclic ones included, at `O(K)` per micro per iteration and no matrix inversion. | §11.4. The default solver. |
| **Shapley (1953)** — efficiency, symmetry, null player and linearity determine a unique value. | Credit is conserved: `Σ_i φ_i = v(N) - v(∅)`, exactly, per sampled permutation. The total credit handed out is precisely the log-likelihood the population gained over a uniform guess. | §12. Also the strongest test in the suite (§21, `test_shapley.py`). |
| **Vickrey (1961)** — in a second-price / uniform-price auction with unit demand, truthful bidding is a dominant strategy. | A micro maximises its own wealth by bidding what it actually believes. Calibration is a property of the mechanism, not something a loss function has to be talked into. | §9. |

### 2.2 The one-paragraph version

Hash the context into `F=256` features. Every micro reads 8 of them and names
at most 4 symbols or abstains. Micros bid for 64 seats; the price is the highest
losing bid; truthful bidding is dominant. The seated micros play a congestion
game where being right is worth `B` split among everyone who said the same
thing, so being right *and rare* is worth everything. Solve for the equilibrium
by regret matching. If the dynamics cycle, stop and ask the meta-player. The
equilibrium mixed profile, aggregated, is the prediction. When the truth
arrives, compute each seat's Shapley value against the coalition value function
`v(S) = log P_S(y)`, and let each micro take one two-layer gradient step to earn
more. Every generation, bankrupt micros are replaced by mutated winners.

---

## 3. Notation used throughout

| Symbol | Meaning | Default |
|---|---|---|
| `N` | population size (micros) | 4096 |
| `M` | seats per stage game | 64 |
| `R` | receptive field size (inputs per micro) | 8 |
| `H` | hidden units per micro | 6 |
| `K` | repertoire size (symbols a micro may name) | 4 |
| `F` | feature-vector dimension | 256 |
| `V` | output alphabet (symbol ids `0..|V|-1`) | from the corpus |
| `C` | context window (characters) fed to `features` | 12 |
| `A_i` | action set of micro `i` = `repertoire(i) ∪ {ABSTAIN}`, size `K+1` | |
| `q_i(a)` | micro `i`'s own probability for action `a` (softmax of its output layer) | |
| `ρ_i` | reputation of micro `i` in `[0, 1]` | |
| `n_a` | **load**: how many seated micros chose action `a` | |
| `B` | base reward for a correct action, before splitting | 1.0 |
| `λ` | penalty for a wrong action | 0.5 |
| `c` | participation cost (the auction's clearing price) | endogenous |
| `y` | the true symbol (training only) | |
| `φ_i` | Shapley value of micro `i` | |
| `ABSTAIN` | the reserved action, index `K` in every micro's action vector | |

---

## 4. Package layout

```
GTMNN/
  DESIGN.md                 this file
  README.md                 user docs (install, CLI, API, frontend, concepts)
  pyproject.toml            zero runtime deps; console script `gtmnn = gtmnn.cli:main`
  Makefile                  build / test / run / docker targets
  Dockerfile
  docker-compose.yml
  docker/entrypoint.sh
  gtmnn/
    __init__.py             exports GTMNet, TrainConfig, MicroPool, FeatureHasher, Alphabet,
                            StageGame, solve, shapley_values, Evolver, EvolveConfig,
                            CheckpointManager, get_backend, __version__
    __main__.py             `python -m gtmnn` -> cli.main()
    activation.py           the parametric sine + derivatives (identical contract to RadixCyclicNN)
    features.py             Alphabet, FeatureHasher, receptive fields
    micro.py                MicroPool - flat-array population, forward pass, local gradient step
    payoff.py               the game catalogue: payoff functions and their properties
    auction.py              uniform-price seat auction (§9)
    game.py                 StageGame - seating, actions, loads, payoff evaluation
    equilibrium.py          best response, fictitious play, regret matching(+), replicator, cycle detection
    shapley.py              coalition value v(S), Monte-Carlo Shapley, Banzhaf
    coalition.py            coalition formation, stability, the stag hunt
    meta.py                 the meta-player - metacognition on a cycle or a photo finish
    backend.py              Backend protocol, PythonBackend, TorchBackend, get_backend()
    model.py                GTMNet, TrainConfig, new_model / load_model
    evolve.py               Evolver, EvolveConfig - culling, crossover, mutation, ESS, 2NRL
    checkpoint.py           CheckpointManager
    bench.py                benchmarks (games/sec, Shapley permutations/sec, chars/sec)
    cli.py                  argparse CLI
    api.py                  HTTP JSON API + static file serving
  tests/                    unittest (run: `python -m unittest discover -s tests -v` from GTMNN/)
  frontend/                 Vite + React app (`npm install && npm run build` -> frontend/dist)
  data/
    sample_corpus.txt       good/correct data (one text per line)
    sample_garbage.txt      bad/garbage data (one text per line)
```

`GTMNN/` is the working directory for all commands. Tests import `gtmnn` from
the parent directory via a `sys.path` insert at the top of each test file.

**Module dependency order** (no cycles; a module may only import from those
above it): `activation` -> `features` -> `micro` -> `payoff` -> `auction` ->
`game` -> `equilibrium` -> `shapley` -> `coalition` -> `meta` -> `backend` ->
`model` -> `evolve` -> `checkpoint` -> `bench` -> `cli` -> `api`.

---

## 5. `activation.py`

Same contract as `RadixCyclicNN/radixnet/activation.py`, so the two projects
share the author's activation research and a fix in one transfers to the other by
inspection — **with one deliberate divergence**: `DEFAULT_B` is `1.0` here rather
than `1/3`, for the measured reason in §5.1.

```python
DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K = -1.0, 1.0, 0.0, 0.0   # b=1.0, not 1/3: see §5.1

def sine_activation(x: float, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K) -> float
    # a * sin(b * (x - h)) + k       ->  default is -sin(x). The author's -sin(x/3) is the
    # same function at b=1/3; §5.1 shows that frequency leaves every neuron in its linear
    # region once features are L2-normalised, and measures the cost at 17x.

def sine_derivative(x, a, b, h, k) -> float           # df/dx = a*b*cos(b*(x-h))

def sine_partials(x, a, b, h, k) -> tuple            # returns exactly 6 floats:
    #   (f, dfdx, dfda, dfdb, dfdh, dfdk)
    #   u = b*(x-h);  f = a*sin(u)+k;  dfdx = a*b*cos(u);  dfda = sin(u);
    #   dfdb = a*(x-h)*cos(u);  dfdh = -a*b*cos(u);  dfdk = 1.0

class SineActivation:            # convenience object (tests/docs, never in hot loops)
    __slots__ = ("a", "b", "h", "k")
    def __init__(self, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K)
    def __call__(self, x) -> float
    def derivative(self, x) -> float
    def partials(self, x) -> tuple
    def inverted(self) -> "SineActivation"     # a -> -a AND k -> -k: f = a*sin(u)+k, so
                                               # negating a alone leaves -f + 2k, which is
                                               # the negation only while k is still 0
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d)
```

**Why the sine matters more here.** In a deep net a periodic activation is a
liability — the same output can come from unboundedly many pre-activations and
depth compounds the ambiguity. A micro is two layers deep, so the ambiguity
never compounds; what is left is the useful half of periodicity. Two micros
whose pre-activations differ by `2π/b = 6π` are *indistinguishable at the
output and unrelated in weight space*, which means the population can hold many
genuinely different solutions that agree on the training data. That is diversity
the congestion game can then price. A monotone activation would give a
population whose members mostly agree, and a congestion game among players who
agree pays almost nothing to anyone.

### 5.1 The frequency `b` must be initialised against the input scale

`DEFAULT_B = 1/3` is **wrong for this network**, and measurably so. Reaching the
first peak of `sin(b·z)` requires `|z| = π/(2b)`, which at `b = 1/3` is **4.71**.
Features here are L2-normalised (§6.2) so `|x| ≈ 1`, and with weights initialised
in `[-0.5, 0.5]` over `R = 8` inputs the pre-activation `z` sits near zero. Every
neuron therefore operates in the **linear** region of its sine, and a population
of near-linear units is a population of linear models however wide it is.

Measured on a synthetic task whose target is itself a sine of a linear
combination — the case most favourable to this activation
(`Experiments/NeuralCompression/experiment.py sweep`):

| `b` | `\|z\|` to first peak | test MSE |
|---|---|---|
| **1/3** | 4.71 | **0.01674** |
| 1.0 | 1.57 | **0.00015** |
| 2.0 | 0.79 | **0.00004** |
| 3.0 | 0.52 | diverges |
| *tanh reference* | — | *0.00098* |

At `b = 1/3` the sine is **17× worse than tanh**; at `b = 1-2` it is **6-24×
better**. The activation is not the problem, the frequency is.

**Therefore `DEFAULT_B = 1.0` for GTMNN**, with the general rule

```
b  ≈  π / (2 · E|z|)        # initialise so the first peak falls inside the
                            # actual pre-activation range
```

`b` is learnable per neuron, but that does not rescue a bad initialisation:
`∂f/∂b = a(x−h)·cos(b(x−h))` is small exactly when `x` is small, so the gradient
that would correct the frequency is suppressed by the same condition that makes
it wrong. The initialisation is the trap, not the parameterisation.

Two consequences for text above and elsewhere:

* §6.2's claim that L2 normalisation "keeps the sine in its informative range... where `-sin(x/3)` is steepest and **most nearly linear**" is corrected there: near-linear is the failure mode, not the goal. Normalisation is still right — it makes `E|z|` *predictable*, which is what lets `b` be set correctly at all.
* The diversity argument above depends on the period `2π/b`, which at `b = 1` is `6.28` rather than `18.8`. Distinct solutions are therefore *closer together* in weight space than that paragraph assumed. The argument survives — periodicity still admits many non-identical solutions — but the spacing claim should be re-derived before being relied on, and `bench` should report the realised spread of `z` so this is measured rather than assumed. Open question 6.

`RadixCyclicNN` carries the same `b = 1/3` default. It is a working, merged
system on a different task with a different pre-activation scale, so this is
flagged there rather than changed: measure its realised `E|z|` before touching it.

---

## 6. `features.py`

### 6.1 Alphabet

```python
class Alphabet:
    symbols: list[str]                  # index -> symbol (single characters)
    index: dict[str, int]               # symbol -> index
    def __init__(self, symbols: Iterable[str] = ())
    def add(self, ch: str) -> int       # idempotent; returns the id
    def fit(self, texts: Iterable[str]) -> "Alphabet"   # adds every character seen, in first-seen order
    def __len__(self) -> int
    def encode(self, ch: str) -> int    # -> id, or UNK (= len-1 slot reserved at construction) if unknown
    def decode(self, i: int) -> str
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d)
```

Id `0` is reserved for `UNK`; id `1` is reserved for `END`. Real characters
start at `2`. `fit` is incremental and never reorders existing ids, so an
alphabet may be extended after training without invalidating a saved model.

### 6.2 FeatureHasher

The context is the last `C=12` characters before the position being predicted.
It is turned into a fixed `F=256`-dimensional vector by **signed feature
hashing** — no vocabulary, no growth, no lookup table:

```python
class FeatureHasher:
    def __init__(self, dim: int = 256, context: int = 12, ngrams: tuple[int, ...] = (1, 2, 3), seed: int = 0)
    dim: int ; context: int ; ngrams: tuple[int, ...] ; seed: int

    def transform(self, context: str, out: array | None = None) -> array   # array('d') of length dim
        # For each n in ngrams, for each n-gram g at offset o (o = 0 is the n-gram ending at the
        # last character, o = 1 the one before it, ...):
        #     token = f"{n}:{o}:{g}"
        #     hv    = _hash(token)                          # see below; deterministic across runs
        #     idx   = hv % dim
        #     sign  = +1.0 if (hv >> 32) & 1 else -1.0
        #     out[idx] += sign * decay ** o                 # decay = 0.9
        # Then L2-normalise out (if the norm is 0, leave it as zeros).
        # Positional offsets are part of the token, so "ab" one character back and "ab" five
        # characters back are different features. Context shorter than needed -> fewer tokens, no padding.

    def _hash(self, token: str) -> int      # 64-bit FNV-1a over token.encode("utf-8") mixed with seed.
        # MUST NOT be Python's hash(): that is salted per process (PYTHONHASHSEED) and would make a
        # saved model predict differently after a restart. This is a correctness requirement, not a
        # performance one, and test_features.py asserts it across a subprocess boundary.

    def to_dict(self) -> dict ; @classmethod from_dict(cls, d)
```

L2 normalisation is what makes the pre-activation scale **predictable**: feature
values land in roughly `[-1, 1]`, a micro sums `R=8` of them through weights
initialised in `[-0.5, 0.5]`, so `E|z|` is known in advance — which is precisely
what allows the sine frequency `b` to be initialised correctly against it (§5.1).
It is *not* a reason to sit near zero: that region is the sine's linear one, and
a population of near-linear units is a linear model. `b = 1.0` places the first
peak at `|z| = 1.57`, inside the operating range rather than far outside it. Without it, long contexts would
push pre-activations past `3π/2` and neighbouring inputs would start wrapping
onto the same output for no reason.

### 6.3 Receptive fields

```python
def sample_receptive_field(dim: int, size: int, rng: random.Random) -> list[int]
    # `size` DISTINCT indices in [0, dim), sorted ascending. rng.sample.

def coverage(pool: "MicroPool", dim: int) -> list[int]
    # How many micros read each feature index. Used by the frontend heat strip and by
    # Evolver: a feature no living micro reads is a blind spot, and newborns are biased
    # toward covering it (§17.3).
```

Fields are drawn uniformly at birth. With `N=4096`, `R=8` and `F=256` the
expected number of micros reading any given index is `4096*8/256 = 128`, and
the chance a given index is read by nobody is `(1 - 8/256)^4096 ≈ 3·10^-57`. The
blind-spot logic exists for *evolved* populations, where culling can strip
coverage from a region the corpus stopped exercising.

---

## 7. `micro.py` — MicroPool

### 7.1 Storage (performance: flat `array('d')`, one allocation per field)

There are no `Micro` objects in the hot path. `MicroPool` is a struct-of-arrays;
micro `i` owns the slice `[i*stride, (i+1)*stride)` of each array.

```python
from array import array

ABSTAIN = -1          # the action id meaning "I decline to play"; slot K of every action vector

class MicroPool:
    n: int                      # capacity (living + dead); ids are stable, never reused except by rebirth
    R: int ; H: int ; K: int    # 8, 6, 4
    F: int                      # feature dim the receptive fields index into

    rf:        array('i')       # n*R     receptive-field indices
    w1:        array('d')       # n*R*H   input -> hidden
    b1:        array('d')       # n*H
    w2:        array('d')       # n*H*(K+1)   hidden -> action
    b2:        array('d')       # n*(K+1)
    sa, sb, sh, sk: array('d')  # n*(H+K+1)   sine params per neuron (hidden then output)
    repertoire: array('i')      # n*K     symbol ids this micro may name; -1 = empty slot

    strategy:  array('d')       # n*(K+1) current mixed strategy (rows sum to 1)
    regret:    array('d')       # n*(K+1) cumulative regret, for regret matching
    wealth:    array('d')       # n       accumulated payoff, less auction payments
    hits:      array('d')       # n       discounted count of correct plays
    plays:     array('d')       # n       discounted count of non-abstaining plays
    age:       array('i')       # n       stage games seated in
    born:      array('i')       # n       generation of birth
    alive:     bytearray        # n       1 = alive

    rng: random.Random
    inverted: bool = False
    version: int                # bumped on any parameter change
    pop_version: int            # bumped only on birth / death / repertoire change
```

Derived, not stored:

```python
def reputation(self, i: int) -> float        # (hits[i] + 1.0) / (plays[i] + 2.0)   -- Laplace smoothed
```

`hits` and `plays` are **discounted**, not cumulative: after every stage game
both are multiplied by `gamma_rep = 0.999` before the update. A micro that was
excellent ten thousand games ago and has been wrong since must not keep trading
on it, because reputation is what it bids with (§9).

### 7.2 Initialisation

```python
def __init__(self, n, F, R=8, H=6, K=4, seed=0, alphabet_size=2)
```

Per micro:
* `rf` <- `sample_receptive_field(F, R, rng)`
* `w1`, `w2` <- `rng.uniform(-0.5, 0.5)`; `b1`, `b2` <- `0.0`
* sine params <- `(DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K)` per neuron
* `repertoire` <- `K` distinct symbol ids sampled from the alphabet (`-1` while the alphabet is smaller than `K`)
* `strategy` <- uniform `1/(K+1)`; `regret` <- `0.0`; `wealth` <- `0.0`
* `hits`, `plays` <- `0.0`; `age`, `born` <- `0`; `alive` <- `1`

If `inverted` is True at creation time (a micro born while the population is
inverted), `sa <- -DEFAULT_A`, matching `RadixCyclicNN`'s rule for nodes born
into an inverted graph.

### 7.3 Forward pass

```python
def forward(self, i: int, feat: array, out: array) -> None
    # out is length K+1 and receives q_i, the micro's own probabilities.
    #   for h in range(H):
    #       z = b1[i*H + h] + sum(w1[base + r*H + h] * feat[rf[i*R + r]] for r in range(R))
    #       a_h = sine(z, sa[..], sb[..], sh[..], sk[..])
    #   for o in range(K+1):
    #       z = b2[..] + sum(w2[..] * a_h)
    #       a_o = sine(z, ...)
    #   out[:] = softmax(a_o)       # numerically stable: subtract max before exp
    # Slots whose repertoire entry is -1 are masked to -inf BEFORE the softmax, so a micro
    # with an unfilled repertoire cannot name a symbol it does not have.

def forward_many(self, ids: Sequence[int], feat: array) -> list[array]
    # The batched form the backend actually calls. PythonBackend loops with locals bound;
    # TorchBackend does one (len(ids), R) gather, two matmuls and a softmax.

def cache_forward(self, i: int) -> tuple   # (a_h, z_h, z_o, q) retained from the last forward,
    # needed by the local gradient step. Stored in a per-seat scratch buffer of size M, not per micro:
    # only seated micros ever need it, and M << n.
```

### 7.4 The local learning step

This is the only place weights change, and it changes only micro `i`'s own
`R*H + H + H*(K+1) + (K+1)` weights and `(H+K+1)*4` sine parameters.

```python
def learn(self, i: int, action_slot: int, phi: float, lr: float, act_lr: float, clip: float = 5.0) -> None
    # phi is micro i's Shapley value for the stage game just resolved (§12), used directly as the
    # advantage of a REINFORCE step on the micro's own policy:
    #
    #     L_i = -phi * log q_i(action_slot)
    #     dL/dz_o[o] = -phi * (1[o == action_slot] - q_i[o])       # softmax-with-log-likelihood
    #
    # then ordinary two-layer backpropagation INSIDE the micro:
    #     dL/d(a_o) via sine_partials at z_o, dL/dw2, dL/db2, dL/d(a_h), dL/dz_h, dL/dw1, dL/db1
    #     and dL/d(sa,sb,sh,sk) for all H+K+1 neurons from the same partials.
    # Weights step by lr, sine parameters by act_lr (act_lr << lr: the activation is shared
    # structure and should move an order of magnitude slower than the weights that use it).
    # Every gradient is clipped elementwise to [-clip, clip] before the update.
    # version += 1 (batched: the backend bumps once per step, not once per micro).
```

**Why Shapley is the right coefficient and not merely a convenient one.**
REINFORCE needs an advantage — a payoff with a baseline subtracted — and the
choice of baseline is normally a variance-reduction heuristic. Here it is
forced. Shapley's efficiency axiom gives `Σ_i φ_i = v(N) - v(∅)`, so the
population-wide sum of learning signals is exactly the log-likelihood the
population earned over a uniform guess. A micro whose presence changed nothing
has `φ_i = 0` by the null-player axiom and takes no step at all — not a small
step, exactly none. Two micros that are interchangeable get identical `φ` by the
symmetry axiom, so nothing arbitrary breaks the tie between them. No hand-tuned
baseline has those three properties simultaneously, and Shapley is the *only*
value that does (§12.1).

Unseated micros do not learn. This is intentional and is the second force
driving specialisation: if you cannot win a seat on this kind of input, you do
not get shaped by this kind of input, and you drift toward whatever you *can*
win seats on.

### 7.5 Repertoire maintenance

```python
def note_symbol(self, i: int, symbol: int, credit: float) -> None
    # Maintains a small per-micro tally of which symbols it has been credited for. Every
    # `repertoire_every` stage games (default 512), the K highest-credit symbols become the
    # micro's repertoire. A symbol entering or leaving resets that slot's w2 column and b2 entry
    # to fresh random values (it is now a different action; its old weights mean nothing) and
    # bumps pop_version.
    # The ABSTAIN slot is never reassigned.
```

### 7.6 Serialisation

```python
def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> "MicroPool"
    # Dead micros are compacted out on save (ids remapped densely); floats are written as
    # lists; `rng.getstate()` is stored (tuple -> lists) so a resumed run is bit-reproducible.
    # A saved pool records R, H, K, F: loading into a different geometry is an error, not a
    # silent reshape.
def clone(self) -> "MicroPool"                # deep copy, used by ESS invasion tests
def invert(self) -> None                      # sa -> -sa for every neuron; inverted = not inverted; version += 1
```

### 7.7 Invariants (asserted by `test_micro.py`)

1. Every alive micro's `rf` slice holds `R` distinct indices in `[0, F)`.
2. `strategy` rows sum to `1.0 ± 1e-9` and are non-negative.
3. `forward` returns a distribution: non-negative, sums to `1.0 ± 1e-9`, and is `0.0` in every slot whose repertoire entry is `-1`.
4. `reputation` is in `(0, 1)` for every micro, including one that has never played.
5. `learn` with `phi == 0.0` leaves every array bit-identical.
6. `learn` with `phi > 0` increases `q_i(action_slot)` for a small enough `lr` (finite-difference check against the analytic gradient, `rtol=1e-5`).
7. `invert()` twice is the identity on every array.
8. `to_dict`/`from_dict` round-trips to an identical pool, and a pool with dead micros round-trips to a compacted pool whose *living* micros are identical.

---

## 8. `payoff.py` — the game catalogue

Every game in the system implements one protocol. Nothing downstream knows which
game it is holding.

```python
class Payoff(Protocol):
    name: str
    def utility(self, seat: int, action_slot: int, loads: Sequence[int], ctx: "GameContext") -> float
        # Payoff to `seat` for choosing `action_slot`, given how many seats chose each slot.
    def is_exact_potential(self) -> bool
        # True  -> Rosenthal applies: best-response dynamics is guaranteed to terminate (§2.1).
        # False -> a pure equilibrium may still exist (Milchtaich) but the path may cycle.
    def potential(self, loads: Sequence[int], ctx) -> float | None
        # The Rosenthal potential, or None when is_exact_potential() is False.
```

`loads` is indexed by **global symbol id**, not by seat-local slot: two seats
naming `"e"` from different repertoire slots must count as the same load. The
`GameContext` carries `alphabet_size`, the true symbol `y` (or `None` at
inference), `B`, `λ`, and the per-seat belief matrix `q`.

### 8.1 `CorrectnessCongestion` — the training game

The label is known. Every seat faces the same function of `(action, load)`:

```
u_i(a) =   B / n_a        if a == y
         = -λ             if a != y and a != ABSTAIN
         =  0             if a == ABSTAIN
```

Identical across players, and non-increasing in load, so by **Rosenthal** this is
an exact potential game with

```
Φ(loads) = Σ_{j=1}^{n_y} B/j  -  λ · Σ_{a ∉ {y, ABSTAIN}} n_a
         = B · H(n_y)  -  λ · W          where H is the harmonic number and W the wrong-count
```

`is_exact_potential() -> True`. Best-response dynamics therefore strictly
increases `Φ` on every accepted deviation, `Φ` is bounded above by `B·H(M)`, and
the action space is finite: the solver **terminates**, at a pure Nash
equilibrium, always. Training never escalates to the meta-player, and
`test_equilibrium.py` asserts it never does.

The split `B/n_a` is the entire specialisation mechanism. Being the ninth micro
to say `"e"` after a space is worth `B/9`; being the only micro that can call
the `"q"` in `"Iraq"` is worth `B`. Micros are not told to specialise and are not
regularised into it — they are paid for it.

### 8.2 `BeliefCongestion` — the inference game

The label is unknown, so each seat scores an action by its own expectation:

```
E_i[u_i(a)] =  q_i(a) · B/n_a  +  (1 - q_i(a)) · (-λ)      for a != ABSTAIN
E_i[u_i(ABSTAIN)] = 0
```

Still non-increasing in `n_a` (derivative `-q_i(a)·B/n_a² ≤ 0`), but now
**player-specific**: seat `i`'s payoff depends on `q_i`, so two seats on the same
action can value it differently. By **Milchtaich** a pure Nash equilibrium still
exists and a best-response improvement path to one exists from any profile, but
an arbitrary best-response path may cycle. `is_exact_potential() -> False`;
`potential() -> None`.

This is the formal home of `Research/CyclesAreAFeature.md`. The cycle is not the
solver failing to converge — the solver is behaving correctly on a game whose
improvement paths genuinely loop. Iterating longer does not help. Leaving does
(§14).

### 8.3 `PrisonersDilemma` — activation sharing

Used pairwise inside coalitions (§13). A micro may **share** its hidden vector
`a_h` with a partner (who may then read it as `R_shared` extra inputs) or
**withhold** while still reading the partner's.

```
              share     withhold
share      (   Rw,Rw )  (  S,T  )
withhold   (   T,S   )  (  P,P  )
        T = 1.6,  Rw = 1.0,  P = 0.2,  S = -0.4        (T > Rw > P > S, 2Rw > T + S)
```

One-shot: defection dominates. Repeated between the same two micros inside a
standing coalition, with `tit_for_tat` memory of the partner's last move, the
cooperative equilibrium is sustainable — and cooperation is what actually
widens a micro's effective receptive field from `R=8` to `2R=16` without
growing a single weight matrix. Sharing is therefore the population's only route
to a wider view, and it has to be *earned* from a partner rather than allocated.

### 8.4 `StagHunt` — coalition commitment

```
             stag      hare
stag      (  4, 4 )  ( 0, 3 )
hare      (  3, 0 )  ( 3, 3 )
```

Two equilibria: `(stag, stag)` is payoff-dominant, `(hare, hare)` is
risk-dominant. A coalition's joint action is the stag: worth more, worth nothing
unless the partners come. This is exactly the tension of committing to a
minority answer, and §13 uses the risk-dominance threshold to decide when a
coalition should form at all.

### 8.5 `RockPaperScissors` — the cycle fixture

Zero-sum, three actions, unique equilibrium at uniform, and no pure Nash. It is
not part of the prediction pipeline; it is the fixture that proves the cycle
detector and the meta-player work, because on it best-response dynamics is
*guaranteed* to cycle — with period 3 in sweeps (§11.1) — and fictitious play is
guaranteed to converge only in the time-average.

Genuine RPS structure does arise in the population, and it is worth naming why:
intransitive dominance among specialists is the normal case, not a pathology.
Micro `A` beats `B` on inputs where `A`'s receptive field is informative, `B`
beats `C` where `B`'s is, and `C` beats `A` where `C`'s is. Nothing forbids the
loop from closing. When it does, there is no single best micro to defer to, and
a system whose only move is "pick the best one" has no correct answer available.

### 8.6 `Inverted` — the 2NRL anti-game

```python
class Inverted(Payoff):
    def __init__(self, inner: Payoff)
    # utility = -inner.utility(...); is_exact_potential() = inner.is_exact_potential();
    # potential = -inner.potential(...)   (negating an exact potential leaves it exact)
```

Wraps any payoff and flips its sign. Phase 1 of 2NRL trains on garbage under the
ordinary game; `invert()` then flips both the sines (`a -> -a`) and the payoff;
phase 3 fine-tunes on correct data. Negating an exact potential function
preserves exactness, so the training game stays convergent throughout — the
inverted pass is still guaranteed to terminate.

---

## 9. `auction.py` — who gets a seat

Only `M=64` of `N=4096` micros play any given stage game. Choosing them by
"highest confidence" would be a scoring rule the micros could game by
overstating. Selling the seats cannot be gamed, because of Vickrey.

```python
@dataclass
class SeatAllocation:
    seats: list[int]          # micro ids, high bid first, len <= M
    bids: list[float]         # each seated micro's bid
    price: float              # uniform clearing price = the highest LOSING bid (0.0 if fewer than M bidders)
    bidders: int              # how many micros bid above the reserve
    def to_dict(self) -> dict

RESERVE = 0.01

def bid(pool: MicroPool, i: int, q: array) -> float
    # ρ_i * (max_{slot != ABSTAIN, repertoire[slot] != -1} q[slot] - q[ABSTAIN]), floored at 0.0.
    # "How much better than abstaining do I think my best call is, discounted by how often I am right."

def allocate(pool, ids, qs, m: int = 64, reserve: float = RESERVE) -> SeatAllocation
    # Bids above `reserve`, sorted descending (ties broken by reputation, then by micro id, so the
    # allocation is deterministic). Top `m` are seated. price = bids[m] if len(bids) > m else 0.0.
    # Every seated micro pays `price` from wealth, regardless of what it bid.

def charge(pool: MicroPool, alloc: SeatAllocation) -> None
    # wealth[i] -= alloc.price for each seated i.  Payment happens at seating, win or lose.
```

**Truthfulness.** A seated micro pays the highest losing bid, a quantity it
cannot influence by changing its own bid (raising your bid can only change
*whether* you are seated, never what you pay). Under unit demand this is
Vickrey's second-price rule, so bidding true value is a dominant strategy, and
`ρ_i·(q_max - q_abstain)` is a micro's true value for a seat. A micro that
inflates its bid wins seats on inputs it cannot answer and pays the price on
every one of them until it goes bankrupt and is culled (§17). Calibration is
enforced by bankruptcy, not by a loss term.

**Why a uniform price and not pay-your-bid.** Pay-your-bid destroys the dominant
strategy — under it the optimal bid is a shaded function of your belief about
the other 4095 bids, which is a harder problem than the one the micro was born
to solve, and it would make the bid uninformative as a confidence signal. The
frontend plots the clearing price directly as the network's "cost of attention",
and that plot is only meaningful because bids are honest.

**Reserve price.** A micro whose best call barely beats abstaining should not be
occupying a seat; `RESERVE = 0.01` clears them out. When fewer than `M` micros
clear the reserve the price is `0.0` and the game is played short-handed, which
is the correct behaviour for an input nobody in the population understands — and
`bidders` falling below `M` is itself a strong signal, forwarded to the
meta-player as a feature (§14.2).

---

## 10. `game.py` — the stage game

```python
@dataclass
class GameContext:
    alphabet_size: int
    y: int | None                 # the true symbol id; None at inference
    B: float = 1.0
    lam: float = 0.5
    q: list[array] = ...          # per-seat beliefs over its own K+1 slots
    slot_symbol: list[list[int]]  # slot_symbol[seat][slot] -> global symbol id, or ABSTAIN for slot K

@dataclass
class StageGame:
    seats: list[int]              # micro ids
    ctx: GameContext
    payoff: Payoff
    def loads(self, actions: Sequence[int]) -> list[int]
        # actions are SLOT indices; returns a list indexed by symbol id (plus one final entry
        # counting abstentions). Two seats naming the same symbol from different slots add to
        # the same load. This is the only place slot-space and symbol-space meet, and getting
        # it wrong silently turns the congestion game into a vote.
    def utilities(self, actions: Sequence[int]) -> list[float]
    def best_response(self, seat: int, actions: Sequence[int]) -> tuple[int, float]
        # The slot maximising this seat's utility holding the others fixed; ties -> lowest slot.
        # Loads are updated incrementally (remove this seat's current action, try each slot),
        # never recomputed from scratch: this is the innermost loop of the whole system.
    def potential(self, actions: Sequence[int]) -> float | None
```

```python
def build(pool, feat, ctx_alphabet_size, y, q=None, m=64, payoff=None, reserve=RESERVE,
          charge=True) -> tuple[StageGame, SeatAllocation, list[array]]
    # 1. q = q if q is not None else pool.forward_many(living_ids, feat)
    #    The caller normally passes the backend's population forward (§16.1 step 2); build()
    #    computes it only when called directly, as `gtmnn game` and POST /api/game do. It is
    #    the one full-population pass per prediction and must never happen twice.
    # 2. alloc = auction.allocate(pool, living_ids, q, m, reserve)
    # 3. if charge: auction.charge(pool, alloc)
    #    charge=False is the read-only path: POST /api/game and `gtmnn game` inspect a stage
    #    game without moving anyone's wealth (§21).
    # 4. payoff = CorrectnessCongestion() if y is not None else BeliefCongestion()
    # 5. return StageGame(alloc.seats, GameContext(...), payoff), alloc, q
```

Step 1 is `N` forward passes of an `8 -> 6 -> 5` network — about 90 multiply-adds
each, 370k per prediction. That is the cost floor of the design and §22 is about
keeping it there.

### 10.1 Aggregation — turning an equilibrium into an answer

```python
def aggregate(game: StageGame, profile: Sequence[array], pool: MicroPool,
              eps: float = 1e-3) -> array
    # Returns a distribution over the FULL alphabet:
    #   P(c) ∝ eps/|V| + Σ_{seat i} Σ_{slot s: slot_symbol[i][s] == c} ρ_i * profile[i][s]
    # The ABSTAIN slot contributes to nothing. `eps` smoothing guarantees P(c) > 0 for every c,
    # which is what makes log P(c) finite and therefore makes v(S) (§12.2) finite for every
    # coalition including the empty one.
```

Reputation weighting inside the aggregate is deliberate double-counting:
reputation already decided who got a seat, and it decides again how loudly they
speak once seated. A micro that is right 90% of the time and one that is right
55% should not be equal voices merely because both cleared the auction.

---

## 11. `equilibrium.py` — solving the stage game

```python
@dataclass
class Equilibrium:
    profile: list[array]        # per seat, a mixed strategy over its K+1 slots
    actions: list[int]          # per seat, a pure action slot (argmax of profile; ties -> lowest)
    iterations: int
    converged: bool
    cycle: int                  # 0 = none; otherwise the detected cycle length
    exploitability: float       # max over seats of (best-response utility - realised utility); 0 at Nash
    potential: float | None
    solver: str
    def to_dict(self) -> dict

def solve(game: StageGame, method: str = "regret", iters: int = 64,
          rng: random.Random | None = None, tol: float = 1e-6) -> Equilibrium
    # method in {"best_response", "fictitious", "regret", "regret_plus", "replicator"}
```

`exploitability` is the honest convergence measure: it is zero exactly at a Nash
equilibrium and is what the CLI, the API and the frontend all report. "The
solver ran 64 iterations" says nothing; "exploitability 0.004" says the profile
is within 0.004 of nobody wanting to move.

### 11.1 Cycle detection

Shared by every solver that produces pure profiles:

```python
class CycleDetector:
    def __init__(self, max_history: int = 256)
    def push(self, actions: Sequence[int]) -> int
        # Hashes the pure profile (a tuple of slot indices) and looks it up in a dict of
        # profile -> first sweep seen. Returns 0 if new, else (current_sweep - first_sweep)
        # = the cycle length. A returning profile in a deterministic dynamic means the whole
        # future is that loop, so the solver stops immediately rather than confirming it.
```

**Push once per completed sweep, never per seat move.** This is not a detail.
A sweep is one pass over all seats; mid-sweep the profile is a half-updated
state that is not a point of the dynamic, and hashing those states multiplies
every reported cycle length by the number of seats that move within the loop.
Two-player rock-paper-scissors is the worked example: pushed per sweep it has
period **3**, pushed per move it reports **6**, and the §11.6 invariant is
written against the former.

Detection is exact and costs one dict lookup per sweep, because the dynamics are
deterministic given the profile: if a state repeats, every subsequent state
repeats. There is nothing probabilistic to be careful about.

### 11.2 `best_response`

Round-robin over seats in a fixed order; each seat moves to its best response if
that strictly improves its utility by more than `tol`. A full sweep with no
accepted deviation means a pure Nash equilibrium — stop, `converged=True`.

On an exact potential game (`payoff.is_exact_potential()`) this terminates by the
finite improvement property, and the implementation **asserts** `potential`
strictly increased on every accepted deviation. If that assertion ever fires, the
payoff function has lied about being an exact potential, and finding that out
at the moment it happens is worth the arithmetic.

On a player-specific game the cycle detector is live — pushed once per completed
sweep (§11.1) — and a detected cycle returns immediately with
`converged=False, cycle=L`.

### 11.3 `fictitious`

Each seat keeps empirical counts of the loads it has faced and best-responds to
the time-average rather than the last profile. Converges on zero-sum and
potential games; on the general player-specific game it may not, but its
time-average is far better behaved than raw best response, and it is the
recommended fallback when `regret` is too slow for a given `M`.

### 11.4 `regret` / `regret_plus` — the default

Hart & Mas-Colell regret matching, per seat, over its `K+1` slots:

```
r_i^t(a)  = u_i(a, loads_{-i}^t) - u_i(a_i^t, loads^t)        # counterfactual regret for slot a
R_i^t(a)  = R_i^{t-1}(a) + r_i^t(a)                            # "regret"
          = max(R_i^{t-1}(a) + r_i^t(a), 0)                    # "regret_plus"
p_i^{t+1}(a) = R_i^t(a)^+ / Σ_b R_i^t(b)^+      if the denominator > 0, else uniform over A_i
```

The returned `profile` is the **time-average** of `p_i^t`, not the final
iterate; the convergence result is about the average, and returning the last
iterate is a common and silent error. `pool.regret` persists across stage games
(warm-started, decayed by `gamma_regret = 0.95` at the start of each game), so a
micro arrives at a new game already knowing which of its actions have
historically been under-played.

`regret_plus` clamps cumulative regrets at zero, which discards the memory of
actions that have been bad for a long time and in practice converges several
times faster. It is the default.

Both cost `O(M·K)` per iteration with no matrix operations, which is why `M=64`
seats and `iters=64` is affordable inside a per-character loop.

### 11.5 `replicator`

Discrete-time replicator dynamics, `p(a) <- p(a)·[1 + η(u(a) - ū)]` renormalised,
with `η = 0.1` and `ū = Σ_b p(b)u(b)`. Not used for inference — it is the
dynamic under which "evolutionarily stable" is defined, and §17.4 uses it to
test whether a mutant strategy can invade the incumbent population.

### 11.6 Invariants (asserted by `test_equilibrium.py`)

1. On `CorrectnessCongestion`, `best_response` always terminates with `converged=True, cycle=0`, and `potential` is strictly increasing along the accepted deviations. Asserted over 500 randomised games.
2. On two-player `RockPaperScissors`, `best_response` reports `cycle == 3` (sweeps, §11.1), and `regret`'s time-average converges to uniform within `0.02` in 10,000 iterations.
3. `exploitability >= 0` always, and `== 0 ± tol` exactly when `converged` and the profile is pure.
4. Every returned `profile` row is a distribution (non-negative, sums to 1).
5. Every solver is deterministic given a seeded `rng` and an identical game — same bytes in, same bytes out.
6. `regret` on a two-seat zero-sum game reaches the known minimax value within `1e-2`.

---

## 12. `shapley.py` — credit assignment

This module replaces backpropagation. It is the most important file in the
project and the one with the strongest tests.

### 12.1 The axioms, and why they are load-bearing

Shapley proved that exactly one function `φ` satisfies all four of:

| Axiom | Statement | What it means for GTMNN |
|---|---|---|
| **Efficiency** | `Σ_i φ_i = v(N) - v(∅)` | Credit is *conserved*. The learning signal handed to the population sums to exactly the log-likelihood gain the population produced. Nothing is invented and nothing leaks. |
| **Symmetry** | If `v(S ∪ {i}) = v(S ∪ {j})` for all `S` excluding both, then `φ_i = φ_j` | Two interchangeable micros get identical credit. No tie is broken by id, insertion order, or which one the loop reached first. |
| **Null player** | If `v(S ∪ {i}) = v(S)` for all `S`, then `φ_i = 0` | A micro that changed nothing takes *exactly zero* gradient step, not a small one. Micros are not nudged by outcomes they had no hand in. |
| **Linearity** | `φ(v + w) = φ(v) + φ(w)` | Credit from two games a micro played in adds, so a batch of stage games can be accumulated before stepping (§16.3) without distorting anyone's share. |

Uniqueness is the point. There is no tuning knob here and no alternative worth
comparing against: any other credit rule violates one of four properties that
are individually non-negotiable.

### 12.2 The coalition value function

```python
def coalition_value(game, profile, pool, y: int, eps: float = 1e-3) -> Callable[[set[int]], float]
```

`v(S) = log P_S(y)`, where `P_S` is exactly `game.aggregate` (§10.1, the module function) restricted to
the seats in `S`:

```
score_S(c) = eps/|V| + Σ_{i ∈ S} Σ_{slot s: symbol(i,s) == c} ρ_i · profile[i][s]
P_S(c)     = score_S(c) / Σ_{c'} score_S(c')
v(S)       = log P_S(y)
v(∅)       = log(1/|V|) = -log|V|         (the smoothing makes the empty coalition exactly uniform)
v(N)       = log P_N(y)
```

so `Σ_i φ_i = v(N) - v(∅) = log(|V| · P_N(y))` — the log-likelihood the seated
population earned over a uniform guess, divided fairly. Negative when the
population did worse than guessing, which is correct and which is how a micro
receives a *negative* `φ` and unlearns.

### 12.3 Estimation

Exact Shapley is `O(2^M)`. With `M=64` that is not a rounding error away from
feasible, so it is estimated by Monte-Carlo over permutations:

```python
@dataclass
class ShapleyResult:
    phi: array('d')             # length = len(seats)
    permutations: int
    total: float                # sum(phi)
    v_full: float ; v_empty: float
    efficiency_error: float     # abs(total - (v_full - v_empty)); 0.0 unless truncation is on
    def to_dict(self) -> dict

def shapley_values(game, profile, pool, y, permutations: int = 32,
                   rng=None, antithetic: bool = True, truncate: float = 0.0,
                   eps: float = 1e-3) -> ShapleyResult
```

For each sampled permutation `π` of the seats, walk it front to back
maintaining the running `score` vector and its total, and credit each seat its
marginal contribution:

```
score <- eps/|V| everywhere ;  total <- eps ;  prev <- log(1/|V|)
for i in π:
    for each slot s of seat i:  score[symbol(i,s)] += ρ_i · profile[i][s] ;  total += ρ_i · profile[i][s]
    cur = log(score[y]) - log(total)
    phi[i] += cur - prev
    prev = cur
```

**Cost.** One permutation is `O(M·K)` — about 320 float operations for `M=64,
K=5` — because `v` is maintained incrementally rather than recomputed. The
default 32 permutations cost roughly 10k operations per stage game, which is
about 3% of the 370k spent on the population forward pass (§10). Credit
assignment is *cheaper than the forward pass*, which is the opposite of the
usual arrangement and is what makes the whole design viable.

**Efficiency is exact per permutation, not just in expectation.** The marginal
contributions along a single permutation telescope: `Σ_i (cur_i - prev_i) =
v(N) - v(∅)`. So the estimator satisfies the efficiency axiom *identically*,
for any number of permutations including one, and `efficiency_error` is
`0.0` to floating-point rounding. `test_shapley.py` asserts this to `1e-9`.
Only sampling *variance* in the split between micros decreases with more
permutations; the total never moves.

**Antithetic sampling** (default on): permutations are drawn in pairs `(π,
reverse(π))`. Both are valid permutations, so efficiency is preserved exactly,
and the negative correlation between a seat's position in `π` and in
`reverse(π)` cuts the variance of `φ_i` substantially at no cost.

**Truncation** (`truncate > 0`): stop walking a permutation once
`|cur - v(N)| < truncate`, crediting the remaining seats zero for that
permutation. Saves time on games where a handful of seats already determine the
answer, but it **breaks efficiency** — `efficiency_error` becomes non-zero and
is reported rather than hidden. Off by default. The test that asserts exact
efficiency runs with `truncate=0.0`, deliberately.

### 12.4 Banzhaf

```python
def banzhaf_values(game, profile, pool, y, samples: int = 64, rng=None, eps=1e-3) -> array
    # Marginal contribution over uniformly random SUBSETS rather than permutations:
    #   φ^B_i = (1/2^{M-1}) Σ_{S ⊆ N\{i}} [v(S ∪ {i}) - v(S)]
```

Cheaper per sample and sometimes a better *ranking* of influence, but it does
**not** satisfy efficiency — the values do not sum to `v(N) - v(∅)` and would
silently rescale the population's learning rate with the population's size.
Provided for analysis and exposed in the frontend's influence view; never used
as the learning coefficient.

### 12.5 Invariants (asserted by `test_shapley.py`)

1. **Efficiency**: `abs(sum(phi) - (v_full - v_empty)) < 1e-9` with `truncate=0.0`, for any `permutations >= 1`, over 200 randomised games.
2. **Null player**: a seat whose profile is entirely on `ABSTAIN` (contributing nothing to any symbol) gets `phi == 0.0` exactly.
3. **Symmetry**: two seats with identical reputation, identical repertoire and identical profile rows get `phi` equal within Monte-Carlo error, and *exactly* equal when the estimator is run over all `M!` permutations for a small game (`M=5`, exhaustive).
4. **Exactness**: for `M <= 8`, MC Shapley over all permutations equals the closed-form subset-weighted sum to `1e-12`.
5. **Linearity**: `phi` of the summed value function equals the sum of the `phi`s, to `1e-9`.
6. **Sign**: when `P_N(y) > 1/|V|` the total is positive; when the population is worse than uniform, negative.
7. Determinism under a seeded `rng`.

---

## 13. `coalition.py`

A micro sees 8 of 256 features. Two micros that share hidden vectors see 16
between them, at the cost of one trust relationship and no new parameters. That
is the only mechanism in the design by which the effective receptive field can
grow, and it is a cooperative game.

```python
@dataclass
class Coalition:
    members: list[int]
    formed: int                 # generation formed
    joint_action: int           # the symbol id the coalition commits to, or ABSTAIN
    payoff_share: array('d')    # per member; sums to 1.0
    trust: dict[tuple[int, int], int]   # (a, b) -> partner b's last move toward a: 1 share, 0 withhold
    def to_dict(self) -> dict

def complementarity(pool, i: int, j: int) -> float
    # 1 - |rf_i ∩ rf_j| / R. Two micros reading disjoint features have complementarity 1.0.

def propose(pool, seats, profile, ctx, max_size: int = 4, min_complementarity: float = 0.5,
            rng=None) -> list[Coalition]
    # Greedy: seed on the highest-reputation unattached seat, add the partner maximising
    # complementarity * agreement (agreement = the partners' profiles overlapping on a symbol),
    # stop at max_size or when no candidate clears min_complementarity.

def should_commit(coalition, pool, ctx) -> bool
    # The stag hunt (§8.4). Committing to the coalition's joint action pays `B/n` if it is right
    # and the partners come, and -λ otherwise. Going alone pays the safe individual expectation.
    # Commit only when the payoff-dominant equilibrium is ALSO risk-dominant for this coalition:
    #     p_partners_commit ** (len(members) - 1) * stag_gain  >  hare_gain
    # estimated from the members' mutual trust history. Small, high-trust coalitions commit;
    # large or new ones do not, which is the correct asymmetry — a five-member coalition needs
    # four partners to show up and is exactly the case where hare is risk-dominant.

def share_step(coalition, pool) -> None
    # One round of the repeated prisoner's dilemma (§8.3), tit-for-tat: each member shares with
    # a partner iff the partner shared last round (first round: share). A member that receives
    # a partner's hidden vector reads it as R extra inputs on the NEXT forward pass through a
    # dedicated `w_share` block; withholding costs nothing this round and costs the partner's
    # cooperation next round.

def payoff_division(coalition, phi: Sequence[float]) -> array('d')
    # The Shapley values of the members WITHIN the coalition, normalised to sum to 1. A member
    # whose marginal contribution to its own coalition is zero receives none of the coalition's
    # payoff, by the null-player axiom.
```

**On the core.** The natural question for a cooperative game is whether the
division is in the *core* — whether any sub-coalition could do better by walking
out. For `v(S) = log P_S(y)` the core is frequently **empty**, and the Shapley
division is therefore not always core-stable. This is not a defect being papered
over; it is a real property of a game where a small expert subset can outscore
the whole group (adding a confidently wrong micro *lowers* `v`, so
super-additivity fails). The design's answer is that coalitions are
**re-proposed every stage game** rather than standing: a sub-coalition that
would rather walk out simply forms on its own next time, which is
`propose()`'s job. Stability is achieved by churn, not by a division nobody can
object to.

```python
def core_deficit(coalition, v: Callable[[set[int]], float]) -> float
    # max over sub-coalitions S of (v(S) - Σ_{i ∈ S} share_i · v(members)), clipped at 0.
    # Exhaustive for len(members) <= 4 (at most 15 subsets), which is why max_size is 4.
    # Reported by `gtmnn info` and plotted by the frontend. Not acted on.
```

---

## 14. `meta.py` — metacognition

> "When we encounter a cycle, we use metacognition or another part of the brain
> instead." — `Research/CyclesAreAFeature.md`

### 14.1 Triggers

The meta-player is invoked after `solve()` when any of:

| Trigger | Condition | Default |
|---|---|---|
| **Cycle** | `eq.cycle != 0` | — |
| **Photo finish** | `top1 - top2 < margin_tau` in the aggregate | `margin_tau = 0.05` |
| **No bidders** | `alloc.bidders < m` — the population has nobody who claims to understand this input | — |
| **Unconverged** | `not eq.converged and eq.exploitability > expl_tau` | `expl_tau = 0.05` |

If none fire, the aggregate stands and `meta` is skipped entirely — which is the
common case, and the reason the meta-player costs nothing on inputs the
population finds easy.

### 14.2 The meta-features

Sixteen summary statistics of the deliberation that just happened. Not of the
input — the meta-player never sees the text. It sees only *how the argument
went*, which is what makes it metacognition rather than a second opinion.

```python
META_F = 16

def meta_features(eq, alloc, game, agg, pool, depth: int) -> array('d')
    #  0  normalised entropy of `agg`            H(agg) / log|V|
    #  1  top1 - top2 margin of `agg`
    #  2  cycle length / 16                      (0.0 when no cycle)
    #  3  exploitability, clipped to [0, 1]
    #  4  fraction of seats on ABSTAIN
    #  5  bidders / N
    #  6  clearing price
    #  7  mean reputation of the seated
    #  8  population std of the seated reputations
    #  9  iterations used / iters
    # 10  mean |cumulative regret| over seats
    # 11  max load / M                           (how concentrated the answer is)
    # 12  distinct symbols named / M             (how scattered it is)
    # 13  converged flag, 0.0 or 1.0
    # 14  mean bid of the seated
    # 15  depth / max_depth
    # L2-normalised, exactly as FeatureHasher does, for the same reason.
```

### 14.3 The meta-population

A second `MicroPool` with `F=16, R=6, H=6, K=4, n=256`. Same class, same
forward pass, same auction, same congestion game, same Shapley credit, same
evolution. Nothing about the meta-player is a special case — it is the identical
machinery pointed at the network's own deliberation, and that is the claim being
made about metacognition.

Its `K=4` repertoire slots are not symbols but **resolution policies**:

| Policy | What it returns |
|---|---|
| `TRUST_EQUILIBRIUM` | the aggregate as it stands |
| `TRUST_REPUTATION` | the single highest-reputation seat's own `q_i`, mapped to the alphabet |
| `TRUST_COALITION` | the highest-value coalition's joint action as a point mass, smoothed |
| `TRUST_CYCLE_AVERAGE` | the mean aggregate over the profiles of the detected cycle |
| `ABSTAIN` (slot 4) | the smoothed uniform — "I do not know either" |

`TRUST_CYCLE_AVERAGE` is the principled response to the trigger that motivated
the whole module. A best-response cycle has no pure equilibrium to report, but
the time-average over the cycle is exactly the object fictitious play converges
to on such games, and it is a genuine mixed strategy rather than an arbitrary
point on the loop. When the network's reasoning goes round in a circle, the
answer is the circle's centre.

### 14.4 Resolution

```python
@dataclass
class MetaResult:
    distribution: array('d')    # over the full alphabet
    policy: int                 # the winning slot
    policy_name: str
    triggered: str              # "cycle" | "margin" | "bidders" | "exploitability" | ""
    depth: int
    inner: "MetaResult | None"  # the recursive call, when the meta-game itself escalated
    def to_dict(self) -> dict

def resolve(eq, alloc, game, agg, pool, meta_pool, depth: int = 0,
            max_depth: int = 2, rng=None) -> MetaResult
    # 1. feat = meta_features(...)
    # 2. Build and solve a stage game on `meta_pool` over `feat`, exactly as §10-11 do.
    # 3. If THAT game triggers (§14.1) and depth < max_depth, recurse with depth + 1.
    #    At depth == max_depth, do not recurse: return TRUST_REPUTATION deterministically.
    # 4. Apply the winning policy; return the resulting distribution.
```

The recursion cap is a hard stop, not a soft preference. The meta-population is
trained by the same Shapley rule against the same truth (§16.2), so "thinking
about how you are thinking" is learned, but "thinking about thinking about
thinking about..." is bounded at `max_depth = 2` and the fallback at the cap is
deterministic. A system that can always escalate is a system that can fail to
answer.

---

## 15. `backend.py`

The population forward pass is the cost floor (§10), so it is the one thing
worth vectorising. Everything else — the auction, the solver, Shapley — is
`O(M)` or `O(M·K)` work on 64 seats and stays in pure Python on both backends.

```python
class Backend(Protocol):
    name: str            # "python" | "torch"
    device: str          # "cpu" | "cuda" | "mps"
    def forward_population(self, pool: MicroPool, feat: array, ids: Sequence[int] | None = None) -> list[array]
    def learn_batch(self, pool: MicroPool, updates: Sequence[tuple[int, int, float]], lr, act_lr, clip) -> None
        # updates: (micro_id, action_slot, phi). Accumulated over a batch of stage games and
        # applied in one pass, which is valid by Shapley's linearity axiom (§12.1).
    def sync_from(self, pool) -> None ; def sync_to(self, pool) -> None
    def describe(self) -> dict

class PythonBackend:
    # Flat array('d') arithmetic with math.sin/math.cos/math.exp bound to locals, no attribute
    # lookups and no per-micro object allocation in any loop. Reference implementation: every
    # numeric result in the test suite is defined by this backend.

class TorchBackend:
    # The population is four tensors: W1 (N,R,H), B1 (N,H), W2 (N,H,K+1), B2 (N,K+1), plus the
    # sine parameters (N,H+K+1,4) and the receptive fields as an (N,R) index tensor.
    #   x  = feat[rf]                       # (N,R)   one gather
    #   zh = einsum("nr,nrh->nh", x, W1) + B1
    #   ah = SA*sin(SB*(zh-SH))+SK
    #   zo = einsum("nh,nho->no", ah, W2) + B2
    #   q  = softmax(SA*sin(SB*(zo-SH))+SK + mask)
    # Two einsums for the whole population. Picks cuda > mps > cpu. Imported lazily; a missing
    # or broken torch degrades to PythonBackend with a warning, never an exception.

def get_backend(name: str = "auto", device: str | None = None) -> Backend
def describe_backends() -> dict      # {"available": [...], "selected": {...}, "torch": version|None}
```

`test_backend.py` asserts the two backends agree to `1e-9` on `forward_population`
and to `1e-7` after a `learn_batch`, and skips when torch is unimportable.

---

## 16. `model.py` — GTMNet

```python
@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 0.05              # micro weights
    act_lr: float = 0.005         # sine parameters (an order of magnitude slower, §7.4)
    seats: int = 64               # M
    iters: int = 64               # equilibrium solver iterations
    solver: str = "regret_plus"
    permutations: int = 32        # Shapley permutations per stage game
    B: float = 1.0 ; lam: float = 0.5 ; reserve: float = 0.01
    batch: int = 64               # stage games accumulated before one learn_batch
    clip: float = 5.0
    meta: bool = True             # run the meta-player on a trigger
    max_depth: int = 2
    margin_tau: float = 0.05 ; expl_tau: float = 0.05
    evolve_every: int = 0         # stage games between cull/birth passes; 0 = no evolution while training
    shuffle: bool = True
    seed: int = 0

class GTMNet:
    pool: MicroPool ; meta_pool: MicroPool
    alphabet: Alphabet ; hasher: FeatureHasher
    backend: Backend
    history: list[dict]
    generation: int
    config: TrainConfig

    def __init__(self, n=4096, F=256, R=8, H=6, K=4, meta_n=256, seed=0, backend="auto", device=None)
```

### 16.1 The stage-game loop — one character, end to end

This is the whole system in one function; every other module exists to be called
from here.

```python
def step(self, context: str, y: int | None, cfg: TrainConfig, learn: bool) -> StepResult
    #  1. feat  = hasher.transform(context)
    #  2. q     = backend.forward_population(pool, feat)            # N forward passes; the ONLY one
    #  3. sg, alloc, _ = game.build(pool, feat, |V|, y, q=q, m=cfg.seats, reserve=cfg.reserve)
    #                                                               # auction: seats sold, price charged
    #  4. eq    = equilibrium.solve(sg, cfg.solver, cfg.iters)      # CorrectnessCongestion if y is
    #                                                               # not None, else BeliefCongestion
    #  5. coals = coalition.propose(pool, sg.seats, eq.profile, sg.ctx)
    #  6. agg   = game.aggregate(sg, eq.profile, pool)              # the module function, §10.1
    #  7. if cfg.meta and triggered(eq, alloc, agg): agg = meta.resolve(...).distribution
    #  8. if learn (y is not None):
    #       phi = shapley.shapley_values(sg, eq.profile, pool, y, cfg.permutations)
    #       pool.wealth[i] += sg.utilities(eq.actions)[seat]  for every seat
    #       pool.hits/plays updated (discounted by gamma_rep first)
    #       pool.note_symbol(i, y, phi[seat])  for every seat with phi > 0
    #       queue (micro_id, action_slot, phi[seat]) into the pending batch
    #       the same, recursively, for the meta-population against the SAME y
    #  9. return StepResult(agg, eq, alloc, coals, meta_result, phi)
```

Step 4's payoff choice is the only difference between training and inference,
and it is the difference between an exact potential game and a player-specific
one (§8.1 vs §8.2). Everything else on the path is identical, which is what
makes "the network is doing at inference what it did at training" true here in
a strong sense rather than a hand-waving one.

### 16.2 Training

```python
def train(self, texts: Sequence[str], config: TrainConfig | None = None,
          progress: Callable[[dict], None] | None = None,
          stop_event: threading.Event | None = None) -> list[dict]
    # alphabet.fit(texts) first (incremental, §6.1); micros whose repertoire slots are still -1
    # are filled from the new alphabet.
    # For each epoch, for each text, for each position p: context = text[max(0, p-C):p],
    # y = alphabet.encode(text[p]); step(context, y, cfg, learn=True). Position len(text) predicts END.
    # Every cfg.batch steps: backend.learn_batch(pending); clear.
    # Every cfg.evolve_every steps (if non-zero): Evolver.cull_and_birth(...).
    # progress(record) after each epoch; stop_event checked between texts so Ctrl-C and
    # POST /api/job/stop are clean and the model is still saveable.
```

Epoch record — the exact shape the API returns and the frontend charts:

```python
{"epoch", "loss", "perplexity", "accuracy", "chars", "games", "seconds",
 "alive", "culled", "born", "generation",
 "mean_wealth", "gini", "mean_price", "mean_bidders",
 "mean_exploitability", "converged_frac", "cycle_rate", "meta_rate",
 "meta_policies": {"TRUST_EQUILIBRIUM": int, ...},
 "diversity", "shapley_total", "mean_reputation", "coverage_gaps"}
```

`loss` is mean `-log P(y)` over characters — the same quantity `Σφ` accounts for,
so `shapley_total` and the epoch's loss improvement are two views of one number
and are asserted consistent in `test_model.py`.

`gini` is the Gini coefficient of `wealth` over living micros. It is the
specialisation readout: a population where everyone earns the same is a
population that has not differentiated, and a healthy run shows `gini` rising
early and then plateauing. `diversity` is the mean fraction of seats naming
distinct symbols. Rising `gini` with collapsing `diversity` means the population
is converging on a few winners and is the signature that `B` is too high
relative to `λ`.

### 16.3 Prediction, generation, scoring

```python
@dataclass
class CharStep:
    symbol: str ; p: float ; entropy: float
    seats: int ; price: float ; bidders: int
    exploitability: float ; cycle: int ; converged: bool ; iterations: int
    meta: dict | None ; coalitions: int
    top: list[tuple[str, float]]        # the `top_k` most likely symbols, for the frontend
    def to_dict(self) -> dict

@dataclass
class Prediction:
    prefix: str ; text: str ; full_text: str
    steps: list[CharStep]
    logprob: float ; perplexity: float
    def to_dict(self) -> dict

def predict(self, prefix: str, length: int | None = None, mode: str = "argmax",
            temperature: float = 1.0, top_k: int = 5, to_end: bool = True,
            cfg: TrainConfig | None = None, rng=None) -> Prediction
    # mode: "argmax" (the aggregate's mode), "sample" (softmax at `temperature`),
    #       "beam" (see below). Rolls the context forward one character at a time; stops at
    #       END when to_end, at `length` characters, or never (length=None and to_end=False
    #       is rejected with ValueError rather than looping forever).

def generate(self, count: int = 5, max_length: int = 60, prefix: str = "", mode: str = "sample",
             temperature: float = 1.0, cfg=None, seed=None) -> list[Prediction]

def beam_predict(self, prefix: str, length: int, beam: int = 8, k: int = 5, cfg=None) -> list[Prediction]
    # Standard beam over characters, scored by cumulative log P from the aggregate. Each beam
    # extension is a full stage game, so cost is O(beam * length) games: the CLI and API both
    # cap `beam * length` and refuse rather than quietly taking minutes.

def score(self, text: str, cfg=None) -> dict
    # {"text", "logprob", "per_char", "perplexity", "chars", "mean_exploitability", "cycle_rate"}
    # No learning, no wealth movement, no auction charges: scoring must not perturb the model.
```

### 16.4 2NRL

```python
def two_nrl(self, bad: Sequence[str], good: Sequence[str], neg_epochs=1, pos_epochs=1,
            neg_lr=0.05, pos_lr=0.01, cfg=None, progress=None, stop_event=None) -> dict
    # 1. train(bad,  epochs=neg_epochs, lr=neg_lr)          -- learn the garbage properly
    # 2. self.invert()                                      -- pool.invert() + meta_pool.invert(),
    #                                                          and the payoff is wrapped in Inverted
    #                                                          for the duration of phase 1's effect
    # 3. train(good, epochs=pos_epochs, lr=pos_lr)          -- fine-tune on truth, slower
    # returns {"negative": [records], "positive": [records], "inverted": bool}

def invert(self) -> None
    # pool.invert(); meta_pool.invert(); self.inverted = not self.inverted. Twice is the identity.
```

Inverting a *population* is cheaper and better behaved than inverting a graph:
`a -> -a` on every sine reflects every micro's response about its own operating
point without touching a single weight, so the population's structure — who
reads which features, who is coalitioned with whom, who is solvent — survives
the inversion intact. The wealth and reputation ledgers deliberately do **not**
invert: a micro that was trustworthy before the flip is still the micro you want
seated after it.

### 16.5 Stats and serialisation

```python
def stats(self) -> dict
    # {"alive", "capacity", "generation", "alphabet", "features", "R", "H", "K", "seats",
    #  "inverted", "mean_wealth", "gini", "mean_reputation", "coverage_gaps", "coalitions",
    #  "meta_alive", "backend": backend.describe(), "history_len"}

def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> "GTMNet"
def save(self, path: str, gzip: bool | None = None) -> int     # .gz suffix -> gzip; returns bytes
@classmethod load(cls, path: str, backend="auto", device=None) -> "GTMNet"
```

Model files carry `format_version`. Loading a file whose `R/H/K/F` differ from
the constructor's is an error with a message naming both geometries — never a
reshape.

---

## 17. `evolve.py` — Evolver

```python
@dataclass
class EvolveConfig:
    generations: int = 0            # 0 = forever (CLI Ctrl-C and API stop both exit cleanly and save)
    games_per_generation: int = 2048
    cull_fraction: float = 0.05     # hard cap on how much of the population can die per generation
    bankrupt: float = -1.0          # wealth below this is bankrupt
    wealth_decay: float = 0.99      # applied to every micro each generation
    mutation_sigma: float = 0.05
    crossover_rate: float = 0.5     # newborns from two parents rather than one
    immigrant_rate: float = 0.10    # newborns that are fresh random micros
    tournament: int = 4             # parent selection tournament size
    ess_samples: int = 16 ; ess_iters: int = 256
    checkpoint_every: int = 0
    seed: int = 0

class Evolver:
    def __init__(self, net: GTMNet, corpus: Sequence[str], config: EvolveConfig | None = None)
    generation: int ; history: list[dict]
    def run_generation(self, stop_event=None) -> dict
    def run(self, generations: int = 0, stop_event=None, progress=None) -> list[dict]
    def cull_and_birth(self) -> tuple[int, int]      # (culled, born); callable mid-training
```

### 17.1 Culling

Bankruptcy, not error. A micro is culled when `wealth < bankrupt`, and at most
`cull_fraction · alive` die per generation (the poorest first). Wealth is the
only survival criterion, and it already integrates everything the design cares
about: a micro that bids honestly and is often right accumulates `B/n_a` on rare
calls; one that bids high and is wrong pays the clearing price every game and
loses `λ` on top; one that always abstains earns nothing and slowly bleeds out
through `wealth_decay`. Being useless is fatal on a longer timescale than being
wrong, which is the right ordering.

The `cull_fraction` cap matters more than it looks. Without it, a corpus shift
can bankrupt most of the population in one generation and take the coverage
(§6.3) with it. With it, the population can lose at most 5% per generation and
the survivors' receptive fields keep the feature space covered while newborns
arrive.

### 17.2 Birth

Each culled slot is refilled in place (ids are stable; `pop_version` bumps):

* with probability `immigrant_rate`, a **fresh random micro**, its receptive field biased toward the feature indices with the lowest `coverage()` — new blood aimed at the population's blind spots;
* else with probability `crossover_rate`, a **crossover** of two parents chosen by `tournament` selection on wealth: uniform crossover per weight block, receptive field sampled from the union of both parents' fields, sine parameters from the fitter parent, repertoire from the fitter parent;
* else a **clone** of one tournament-selected parent.

Every newborn then takes Gaussian mutation `N(0, mutation_sigma)` on all weights
and `N(0, mutation_sigma/10)` on the sine parameters, and starts with
`wealth = 0`, `hits = plays = 0` (reputation `0.5` by Laplace smoothing, §7.1),
`age = 0`, `born = generation`.

Newborns are therefore *neither* advantaged nor protected: reputation `0.5` is
enough to win a seat on an input the incumbents are unsure about and not enough
to take one they are confident about, which is exactly the exploration schedule
the population needs and nobody had to write down as a schedule.

### 17.3 Evolutionary stability

```python
def ess_score(self, samples: int | None = None) -> dict
    # For each of `samples` mutants (a clone of a random incumbent plus mutation at 4 * sigma):
    #   seed a population mix at 5% mutant share, run replicator dynamics (§11.5) for ess_iters
    #   on the BeliefCongestion payoff over a sample of corpus contexts, and record whether the
    #   mutant share fell.
    # -> {"ess_score": fraction of mutants that FAILED to invade, "invaders": [...],
    #     "mean_final_share": float}
```

`ess_score` near 1.0 means the population is evolutionarily stable — no small
mutation pays to enter — which is a converged population and, past a point, a
stagnant one. Near 0.0 means anything can invade and the population has not
found structure worth defending. The useful reading is the *trajectory*: it
should climb through training and then sit high, and a sudden collapse means the
corpus changed under the population. It is reported, plotted, and never
optimised against directly.

### 17.4 The generation record

```python
{"generation", "games", "seconds", "culled", "born", "immigrants", "crossovers", "clones",
 "alive", "mean_wealth", "median_wealth", "gini", "mean_reputation", "mean_age",
 "loss", "accuracy", "diversity", "coverage_gaps",
 "ess_score", "cycle_rate", "meta_rate", "mean_price", "sample": str}
```

`sample` is one short generation from the current population, so the frontend's
evolve tab can show what the thing actually sounds like alongside the curves.

---

## 18. `checkpoint.py`

Identical contract to `RadixCyclicNN/radixnet/checkpoint.py`.

```python
@dataclass
class CheckpointRecord:
    name: str ; path: str ; bytes: int ; created_at: float
    generation: int ; epoch: int | None ; loss: float | None ; tag: str | None
    def to_dict(self) -> dict

class CheckpointManager:
    def __init__(self, directory: str, keep: int = 10, gzip: bool = True)
    def save(self, net: GTMNet, tag: str | None = None, **meta) -> CheckpointRecord
        # writes <dir>/<timestamp>-<tag>.json[.gz], rewrites <dir>/latest.json (a pointer, not a copy),
        # rotates out the oldest beyond `keep`.
    def list(self) -> list[CheckpointRecord]         # newest first
    def latest(self) -> CheckpointRecord | None
    def load(self, name: str, **kw) -> GTMNet
    def load_latest(self, **kw) -> GTMNet | None
    def restore(self, name: str, into: GTMNet) -> None    # in-place, preserves the object identity
                                                          # the API's ModelService holds
```

Writes are atomic: to `<name>.tmp`, `os.replace` onto the target. A run killed
mid-checkpoint leaves the previous checkpoint intact and a stray `.tmp` that
`list()` ignores.

---

## 19. `bench.py`

```python
def bench(net: GTMNet, chars: int = 2000, **cfg) -> dict
    # {"forward_per_sec", "games_per_sec", "chars_per_sec", "solver_iters_per_sec",
    #  "shapley_perms_per_sec", "backend", "device", "seats", "population",
    #  "breakdown": {"forward": pct, "auction": pct, "solve": pct, "shapley": pct, "other": pct}}
```

The `breakdown` is the point. A change that makes the solver twice as fast when
the forward pass is 80% of the budget is a change that was not worth making, and
this is the number that says so.

---

## 20. `cli.py`

`python -m gtmnn <command> [...]` (also the `gtmnn` console script). Global
options before the command: `--model PATH` (default `model.json`), `--backend
auto|python|torch`, `--device`, `--seed`, `--json` (machine-readable output only,
one JSON document on stdout).

| command | args | behaviour |
|---|---|---|
| `train` | `--data FILE [FILE...]` (one text per line; `--whole-file` treats each file as one text), `--epochs`, `--lr`, `--act-lr`, `--seats`, `--iters`, `--solver`, `--permutations`, `--batch`, `--reward B`, `--penalty λ`, `--no-meta`, `--evolve-every`, `--population`, `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--resume`, `--out` | trains (loads `--model` first if it exists), prints the per-epoch table (loss, perplexity, accuracy, alive, gini, diversity, cycle %, meta %), saves |
| `predict` | `--prefix TEXT`, `--length N`, `--mode argmax\|sample\|beam`, `--beam`, `--top-k`, `--temperature`, `--to-end`, `--explain` | prints the continuation; `--explain` prints the per-character table: p, entropy, seats, price, exploitability, cycle, meta policy |
| `generate` | `--count`, `--max-length`, `--prefix`, `--mode`, `--temperature` | samples with logprob and perplexity per sample |
| `score` | `--text` or `--data FILE` | logprob, per-char, perplexity, cycle rate |
| `game` | `--context TEXT`, `--solver`, `--iters`, `--seats` | **runs one stage game and shows all of it**: the bid distribution, who was seated and at what price, the equilibrium profile, exploitability, the cycle if any, the coalitions, every seat's Shapley value, and the meta-player's policy if it fired. The debugging tool for the entire design |
| `shapley` | `--context TEXT`, `--symbol CH`, `--permutations N`, `--exact` | the Shapley table for one stage game; `--exact` enumerates all `M!` orders (refuses above `M=8`) and prints the Monte-Carlo error against it |
| `evolve` | `--data FILE`, `--generations N` (0 = forever; Ctrl-C stops cleanly and saves), `--games-per-generation`, `--cull-fraction`, `--mutation-sigma`, `--immigrant-rate`, `--ess-samples`, `--checkpoint-dir`, `--checkpoint-every`, `--out` | the evolutionary loop |
| `ess` | `--samples`, `--iters` | the evolutionary-stability report |
| `2nrl` | `--bad FILE`, `--good FILE`, `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--out` | runs `two_nrl`, saves |
| `invert` | `--out` | inverts and saves |
| `population` | `--sort wealth\|reputation\|age`, `--limit 40`, `--micro ID` | the census: per-micro wealth, reputation, age, repertoire, receptive field, coverage gaps, the Gini curve. `--micro` dumps one micro's full parameters |
| `info` | | stats + the history tail |
| `checkpoints` | `--dir DIR`, `--restore NAME --out PATH` | list / restore |
| `bench` | `--chars N` | the §19 report |
| `serve` | `--host 127.0.0.1 --port 8000 --frontend-dir frontend/dist --checkpoint-dir checkpoints --upload-dir uploads` | starts the API |

Exit codes: 0 ok, 1 error (message on stderr). Every command supports `--json`.

---

## 21. `api.py` — HTTP JSON API (stdlib only)

`ThreadingHTTPServer` + `BaseHTTPRequestHandler`. All `/api/*` responses are JSON
with CORS headers and `OPTIONS` preflight. Errors are `{"error": "message"}` with
400/404/409/500. A single `ModelService` holds the model behind an `RLock`; long
operations run in one background thread at a time and are tracked as a **job**
(a second request gets 409). Job status is
`{"id", "type", "state": "running|done|error|stopped", "progress", "history", "error",
"started_at", "finished_at"}` — the same shape as RadixCyclicNN, so `useJob` is
portable between the two frontends unchanged.

| method & path | body | response |
|---|---|---|
| GET `/api/health` | | `{"ok": true, "version"}` |
| GET `/api/status` | | `stats()` + `{"job", "backends", "model_path", "checkpoint_dir"}` |
| POST `/api/train` | `{"texts"\|"text"\|"files"}` + TrainConfig fields | `{"job"}` (async) |
| GET `/api/job` | | current/last job status |
| POST `/api/job/stop` | | sets the stop event; returns job status |
| POST `/api/predict` | `{"prefix","length","mode","beam","top_k","temperature","to_end","explain"}` | `Prediction.to_dict()`; with `explain`, every `CharStep` including its meta result |
| POST `/api/generate` | `{"count","max_length","prefix","mode","temperature","seed"}` | `{"samples": [Prediction.to_dict()]}` |
| POST `/api/score` | `{"text"}` | the score dict |
| POST `/api/game` | `{"context","seats","iters","solver","symbol"}` | **the whole stage game**: `{"bids": [...], "allocation", "equilibrium", "coalitions", "aggregate", "meta", "shapley"}`. Read-only — no wealth moves, no seats are charged, nothing learns. This is what `GameView` renders |
| POST `/api/shapley` | `{"context","symbol","permutations","exact"}` | `ShapleyResult.to_dict()` + the per-seat table |
| GET `/api/population?sort=wealth&limit=100` | | `{"micros": [{"id","wealth","reputation","age","born","repertoire","rf","alive"}], "gini", "coverage", "coverage_gaps", "alive", "capacity"}` |
| GET `/api/micro/<id>` | | one micro in full: weights, sine parameters, receptive field, repertoire, ledger |
| POST `/api/2nrl` | `{"bad","good","neg_epochs","pos_epochs","neg_lr","pos_lr"}` | job (type `2nrl`) |
| POST `/api/feedback` | `{"good": [...], "bad": [...]}` | job (type `feedback`); both kinds -> `two_nrl`; good only -> a positive pass; bad only -> a negative pass then `invert()`. Identical semantics to RadixCyclicNN's endpoint |
| POST `/api/invert` | | `stats()` |
| POST `/api/evolve/start` | EvolveConfig fields + `{"corpus"\|"corpus_text"\|"corpus_files"}` + `"generations"` | job (type `evolve`) |
| POST `/api/evolve/stop` | | job status |
| GET `/api/evolve/history` | | `{"history": [...]}` |
| GET `/api/ess?samples=16` | | the evolutionary-stability report |
| POST `/api/save` \| `/api/load` \| `/api/reset` | `{"path"}` / `{"path"}` / `{"seed","population"}` | path+bytes / stats / stats |
| GET `/api/checkpoints`, POST `/api/checkpoints/save`, POST `/api/checkpoints/restore` | | list / record / stats |
| GET `/api/history` | | `{"history": net.history}` |
| GET `/api/uploads`, POST `/api/uploads`, POST `/api/uploads/delete` | | as RadixCyclicNN §20: JSON, multipart or raw bytes; names reduced to a safe base name; same name replaces |
| GET `/` and other paths | | serves `frontend_dir` (SPA fallback to `index.html`, correct mime types, no path traversal), else a built-in page explaining how to build the frontend |

```python
def create_server(host, port, model_path=None, checkpoint_dir=None, frontend_dir=None,
                  upload_dir=None, backend="auto", device=None, seed=0) -> tuple[server, ModelService]
def run_server(...)      # serve_forever, clean shutdown on KeyboardInterrupt (saves first)
```

Mutating endpoints refuse with 409 while a job is running. Tests start the
server on port 0 in a thread and talk to it with `urllib.request`.

---

## 22. `frontend/` — Vite + React

Dependencies: `react`, `react-dom`, `vite`, `@vitejs/plugin-react` only. No
TypeScript, no component library, no charting library — the charts are hand-rolled
SVG, as in RadixCyclicNN. `vite.config.js` proxies `/api` to
`http://127.0.0.1:8000` in dev; `npm run build` writes `frontend/dist`, which the
API serves.

* `StatusBar.jsx` — polls `/api/status` every 2s: alive / capacity, generation, Gini, mean reputation, coverage gaps, inverted flag, backend + device, job state.
* `TrainPanel.jsx` — corpus textarea or upload picker, the TrainConfig fields, start / stop; the live epoch table and two charts: loss, and Gini against diversity on one pair of axes (the specialisation readout, §16.2).
* `PredictPanel.jsx` — prefix, length, mode, temperature; the continuation with each character tinted by its probability, and a hover card per character showing seats, price, exploitability, the cycle, the top-k and the meta policy if it fired.
* **`GameView.jsx`** — the centrepiece, on `POST /api/game` for a typed context. Four stacked bands: the **bid curve** over the whole population with the clearing price drawn across it and the seated region shaded; the **seating table** (micro, bid, reputation, repertoire, chosen slot); the **equilibrium** as a seats × actions heat grid with exploitability and the cycle badge; and the **Shapley bars**, signed, sorted, with the efficiency check (`Σφ` against `v(N) - v(∅)`) printed underneath as a number the user can watch stay at zero.
* `PopulationView.jsx` — the census on `/api/population`: a wealth Lorenz curve with the Gini, a reputation histogram, an age histogram, and the `F=256` coverage heat strip with gaps marked. Clicking a micro opens `/api/micro/<id>`.
* `EvolvePanel.jsx` — corpus, EvolveConfig, start / stop; live charts of loss, Gini, ESS score and cull/birth counts per generation, plus the latest `sample` text.
* `MetaPanel.jsx` — how often the meta-player fired, split by trigger, with the policy mix as a stacked area over time. The honest readout of how much of the time the network is confused, which is a number worth looking at directly.
* `TwoNRLPanel.jsx` — bad / good textareas, the epoch and learning-rate fields, negative and positive phase losses, and a manual Invert button.
* `CheckpointPanel.jsx` — list, save with a tag, restore, save/load path, reset.
* `ScorePanel.jsx` — score a text, with the per-character logprob strip.

Plain readable CSS, responsive (single column under 800px), light and dark via
`prefers-color-scheme`.

---

## 23. Tests (unittest, no dependencies)

* `test_activation.py` — the default is exactly `-sin(x/3)`; every partial checked against finite differences.
* `test_features.py` — `Alphabet` round trips and incremental `fit` stability (ids never move); hashing is **stable across processes** (a subprocess computes the same vector — the `PYTHONHASHSEED` trap, §6.2); `transform` output is L2-normalised; positional offsets separate the same n-gram at different distances.
* `test_micro.py` — the §7.7 invariants; the analytic gradient checked against finite differences over all weights and all four sine parameters; `phi == 0` is a no-op on every byte.
* `test_payoff.py` — `CorrectnessCongestion` matches its potential function on 10,000 random load vectors; `BeliefCongestion` is non-increasing in load for every seat; `Inverted` negates exactly and preserves exactness.
* `test_auction.py` — the allocation is deterministic under ties; the price is the highest losing bid; **truthfulness**: for a random seat, over a sweep of bids at fixed opponents, no misreport earns more than the truthful bid (the dominant-strategy property checked numerically, not assumed).
* `test_equilibrium.py` — the §11.6 invariants, including that `best_response` on the training game terminates with `cycle == 0` over 500 randomised games, and that it reports `cycle == 3` on rock-paper-scissors.
* `test_shapley.py` — the §12.5 invariants. **Efficiency to `1e-9` is the single most important assertion in the suite**; also exhaustive-vs-Monte-Carlo agreement for `M <= 8`, the null player at exactly zero, and symmetry under exhaustive enumeration.
* `test_coalition.py` — `complementarity` bounds; tit-for-tat reciprocates and forgives; `payoff_division` sums to 1 and gives a null member zero; `core_deficit` is 0 on a superadditive fixture and positive on a known-empty-core fixture.
* `test_meta.py` — each trigger fires on a constructed case and only on it; `meta_features` is length 16, finite and normalised; recursion stops at `max_depth` and returns `TRUST_REPUTATION` there; `TRUST_CYCLE_AVERAGE` on a period-3 cycle returns the mean of the three aggregates.
* `test_backend.py` — python vs torch agreement to `1e-9` forward and `1e-7` after a learn batch (skipped without torch); a broken torch import degrades rather than raising.
* `test_model.py` — training reduces loss and perplexity on a small corpus; `predict` reproduces a memorised continuation; `score` ranks trained text above garbage; `shapley_total` over an epoch agrees with the epoch's loss improvement; `two_nrl` makes garbage less likely than before it ran; `invert` twice is the identity; save/load round-trips to identical predictions; a saved model reloaded in a **fresh process** predicts identically (the hashing trap again, end to end).
* `test_evolve.py` — one generation runs and records the §17.4 shape; `cull_fraction` is respected exactly at the cap; a bankrupt micro dies and its slot is refilled; newborns start at reputation 0.5; `ess_score` is in `[0, 1]`; `stop_event` is honoured mid-generation.
* `test_checkpoint.py` — rotation, the latest pointer, `load_latest`, resume, and an interrupted write leaving the previous checkpoint intact.
* `test_cli.py` — subprocess smoke test of `train`, `predict`, `game`, `shapley`, `population`, `info`, `2nrl`, `ess`, `checkpoints` and `bench`, each with `--json`, asserting the JSON parses and carries the documented keys.
* `test_api.py` — server in a thread on port 0; health, status, train with job polling, predict, generate, game, shapley, population, micro, score, 2nrl, invert, evolve start/stop, checkpoints, uploads, the 409 while a job runs, and the static fallback.

Every test is seeded. No test depends on wall-clock timing, network access, or
torch being present.

---

## 24. Performance notes (must be followed)

### 24.1 The budget

Per predicted character, at `N=4096, M=64, K=4, iters=64, permutations=32`, on
`PythonBackend`:

| Stage | Work | Share |
|---|---|---|
| Population forward | 4096 × (48 + 30) ≈ 320k multiply-adds, 45k `math.sin` calls | **~80%** |
| Auction | 4096 bids, one sort | ~2% |
| Equilibrium (`regret_plus`) | 64 iters × 64 seats × 5 slots ≈ 20k utility evaluations | ~14% |
| Shapley | 32 perms × 64 seats × 5 slots ≈ 10k operations | ~3% |
| Aggregate, coalitions, meta | `O(M·K)`, and meta only on a trigger | ~1% |

The forward pass dominates by an order of magnitude, and it is the only stage
worth optimising. **Target: 15+ characters/sec on `PythonBackend` and 80+ on
`TorchBackend` (GPU), at the default geometry.** `bench --json` reports the
breakdown and CI asserts the share of `forward` has not grown.

This is roughly a thousand times slower per character than RadixCyclicNN's graph
walk, and that is the honest price of the design: RadixCyclicNN answers by
walking a structure it already built, GTMNN answers by *holding an auction and
solving a game*. What the price buys is credit assignment that is fair by
construction rather than by convention, and a population that specialises because
specialising pays. It is not a trade worth making silently, so it is written
here in the contract.

### 24.2 Rules

* Hot loops bind `math.sin`, `math.cos`, `math.exp`, `math.log`, `len` and every array reference to locals before the loop. No attribute lookups per element, no method calls per element, no per-micro or per-seat object allocation anywhere in `step()`.
* `MicroPool` is `array('d')` throughout, never lists of lists and never objects. Slices are computed once per micro (`base = i * R * H`), not per element.
* Scratch buffers for the forward cache, the loads vector and the Shapley score vector are allocated once on the `GTMNet` and reused; `step()` performs **zero** allocations in steady state beyond the result objects it returns.
* `game.best_response` updates loads incrementally — remove this seat's action, try each slot, restore — and never recomputes the loads vector. It is the innermost loop of the system.
* Shapley maintains `v` incrementally along the permutation (§12.3). Recomputing `v(S)` from scratch per prefix would turn an `O(M·K)` walk into `O(M²·K)` and is the single easiest way to make this design look infeasible when it is not.
* `regret` and `strategy` live in the pool and are warm-started across stage games (§11.4); they are not reallocated per game.

### 24.3 The candidate pre-filter (optional, off by default)

A micro whose 8 features are all near zero in the current vector has a
pre-activation near zero, a near-uniform `q`, a bid near zero, and will not win a
seat. Forwarding it is waste.

```python
prefilter_top: int = 0      # 0 = off (exact). Non-zero = forward only the micros whose receptive
                            # field intersects the `prefilter_top` largest-|value| feature indices,
                            # found through an inverted index feature -> [micro ids] maintained by
                            # MicroPool and rebuilt on pop_version change.
```

At `prefilter_top = 8` roughly 23% of the population is forwarded — a ~4.4×
speedup on the dominant stage. It is an **approximation**: a micro reading eight
small-but-nonzero features could in principle have out-bid a candidate. It is off
by default, and `test_micro.py` measures the disagreement rate (the fraction of
stage games where the seat allocation differs from exact) on the sample corpus
and asserts it stays under 2% at `prefilter_top = 8`. Turn it on for evolution
runs, where throughput matters more than any single game's exact seating; leave
it off for scoring and for anything being compared against a baseline.

---

## 25. Cross-cutting invariants

Asserted somewhere in the suite, and true at every point in the system's life:

1. **Credit is conserved.** `Σ_i φ_i == v(N) - v(∅)` to `1e-9`, every stage game, with truncation off (§12.5).
2. **No gradient crosses a micro.** `learn()` writes only into micro `i`'s own slices. Asserted by checksumming every other micro's arrays around a `learn` call.
3. **Training terminates.** The training game is an exact potential game, so `best_response` always converges and the meta-player never fires during training (§8.1, §11.6).
4. **Inversion is an involution.** `invert()` twice is bit-identical on the pool, the meta-pool and the payoff wrapper. Wealth and reputation are untouched by it (§16.4).
5. **Determinism.** Same seed, same corpus, same config ⇒ same bytes out, across processes and across a save/load cycle. This is why the feature hash cannot be `hash()` (§6.2).
6. **Distributions are distributions.** Every `q`, every strategy row, every profile row and every aggregate is non-negative and sums to 1 within `1e-9`; `aggregate` is strictly positive everywhere so `log` is always finite.
7. **Population bookkeeping.** `alive.count(1) == stats()["alive"]`; ids are stable across a generation; a culled slot is either refilled in the same generation or stays dead, never half-initialised.
8. **The auction is honest.** No misreport beats the truthful bid, checked numerically over a bid sweep (§23).

---

## 26. What is deliberately not here

Named so that a future section adding one of them is a decision rather than a drift.

* **No backpropagation between micros, ever.** Not a deep variant, not a two-hop approximation, not a "just for the meta-player" exception. The moment credit flows by chain rule instead of by Shapley value, this is an ordinary network with extra steps.
* **No attention layer.** The auction is the attention mechanism. Adding a learned attention weight alongside it would give two mechanisms competing to decide who speaks, and the auction's truthfulness guarantee only holds while it is the only one.
* **No global loss.** There is a global *quantity* — `v(N) - v(∅)`, the log-likelihood gain — but nothing optimises it directly. It is divided and each micro optimises its share. A global optimiser would make the game decorative.
* **No shared weights between micros.** Weight sharing would make the null-player axiom false (a micro could matter through a copy of itself) and the Shapley values meaningless.
* **No fixed specialisation.** Micros are never assigned a topic, a symbol set or a region. Repertoires are earned (§7.5), receptive fields are inherited and mutated (§17.2), and the split reward (§8.1) is the only pressure toward differentiation.

---

## 27. Open questions

Honest gaps, to be resolved by implementation and measurement rather than by
guessing now.

1. **`B` against `λ`.** The reward-to-penalty ratio sets how much of the population dares to play. Too high and everyone bids on everything and `diversity` collapses; too low and everyone abstains and the aggregate is the `eps` smoothing. The defaults `B=1.0, λ=0.5` are a starting point, and §16.2 names the two readouts that diagnose a bad ratio. It may need to be scheduled over training rather than fixed — RadixCyclicNN's `schedule.py` (rates as graph functions of the epoch) is the obvious thing to reuse if so.
2. **Does `M=64` matter?** Shapley variance falls with permutations, not with seats, but coalition value is more concentrated with fewer seats. `M` may want to scale with the aggregate's entropy — more seats for harder characters — which would make seat count itself a decision the meta-player could own.
3. **Warm-started regret across stage games** (§11.4) is a real speedup and a mild theoretical liberty: Hart & Mas-Colell's result is about repeated play of *the same* game. Decaying by `gamma_regret` between different games is a heuristic bridge. Measure whether it helps or whether it drags stale regret into games where it does not belong.
4. **Meta-player training signal.** The meta-population is trained against the same `y` as the base population, which credits it for the base population's correctness rather than purely for its own resolution quality. A cleaner signal is the *counterfactual* — how much better the chosen policy did than `TRUST_EQUILIBRIUM` would have — which is itself a marginal contribution and therefore in the same family as everything else here. Worth trying second.
5. **Does the population actually specialise?** The whole design rests on the congestion game doing what §8.1 claims. `gini`, `diversity` and `coverage_gaps` are instrumented from the first commit specifically so this can be falsified early. If wealth stays flat and diversity stays high after real training, the split reward is not biting and the mechanism, not the hyperparameters, is what needs revisiting.
6. **Is the diversity argument still true at `b = 1.0`?** §5's case for a periodic activation is that micros whose pre-activations differ by one period are indistinguishable at the output and unrelated in weight space, giving the congestion game genuinely different solutions to price. §5.1 changed the period from `18.8` to `6.28`, so those solutions now sit three times closer together in weight space. The argument plausibly survives — periodicity still admits many non-identical solutions — but the spacing was asserted, never measured. `bench` should report the realised spread of `z` across the population; if it is much smaller than `2π/b`, no micro is exploiting periodicity at all and the activation is doing nothing a monotone one would not do better.
