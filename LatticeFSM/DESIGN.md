# LatticeFSM — Design

A finite state machine whose transition structure is a dense 3D matrix, and
whose every cell is a record. This document says exactly what the code does,
in both ports; `README.md` says what it measured.

The five requirements, in the author's words, and where each is realised:

| requirement | where |
|---|---|
| a 3D matrix | §1: `Lattice`, `states × symbols × states`, dense |
| finite state machine type design | §5: `Machine` — a state, a start, accepting states; a symbol moves it along an edge |
| edges are dense data classes with multiple values — recently traversed, seen, rewarded, punished, last seen, and more | §2: `Edge`, fifteen fields |
| stimulation makes it easier to traverse wider compared to narrower paths | §3: the `stimulation · log(width)` term; §4: how width moves |
| traverse based on a custom adaptive weighting function for each edge | §3: `Weighting`, one per edge, adapted at every credit; §5: `weight_fn` |
| a 13 × 13 × 13 matrix | §1: the default shape |
| compress the matrix into the central node with minimal loss | §11.1–11.3: the code the central node `(6, g, 6)` holds, exact by default |
| traverse from the middle outward as well | §11.4: the code laid out and rebuilt shell by shell from the centre; a walk from the middle state |
| run the compression every N uses / transitions | §11.5: `compress_every`, `compress_precision`, `compress_rebuild` |
| traverse from the central vertical vector by a finite "focus": high from the bottom node, low or none from the top | §12.1: `focus` in [0, 1], 13 equal bands on `(0..12, g, 6)` |
| allow edges to skip one node if a more efficient path is found | §12.3: a skip reads two symbols in one move past the node between |
| focus should also be a learned value based on the input | §12.2: `learn_focus`, a learner credited like the edges |
| allow nodes to rearrange themselves and swap | §12.4: `swap_states`, `swap_symbols`, `rearrange`, `rearrange_every` |

## 1. The matrix

`Lattice(states, alphabet)` holds `S · A · S` edges, one for every
`(source, symbol, target)`, created when the matrix is: dense, never added
to or removed from. A machine made without a shape — the CLIs, the server's
`serve` and `/api/new`, `Machine()` in Python, `Machine::default_shape` in
Rust — is **13 × 13 × 13**: `DEFAULT_STATES = 13` states over
`DEFAULT_ALPHABET = "abcdefghijklm"`, 2 197 edges. The four languages are
over `a` and `b`, which that alphabet holds, so a default machine can be
taught any of them; the slices for `c` to `m` stay at the prototype until
something is run over them. Storage is one flat vector in row-major order,
`edges[(s · A + a) · S + t]`, so the `S` edges the machine chooses among when
it is in `s` and reads `a` — the **row** `row(s, a)` — are one contiguous
slice. `leaving(s)` is every edge out of `s`; `arriving(t)` every edge into
`t`. The alphabet maps symbols (strings) to indices; `tokenize` splits a
text one character per symbol when every symbol is one character, else on
whitespace.

Beside the edges, one `State` per state: `accepting`, `visits`,
`last_visited`.

Persistence (§8) writes only the edges that were ever written to; the rest
are the prototype's.

## 2. The edge

Every edge holds:

| field | what |
|---|---|
| `source`, `symbol`, `target` | where it is in the matrix |
| `seen` | traversals, ever |
| `first_seen`, `last_seen` | clock readings; `−1` never |
| `recent` | the traversal trace as written at `last_seen`; read through the clock (§4) |
| `rewarded`, `punished` | credit received, ever, each a non-negative sum |
| `last_rewarded`, `last_punished` | clock readings; `−1` never |
| `width`, `width_stamp` | the channel's width as written, and when; read through the clock (§4) |
| `weighting` | its own `Weighting` (§3) |
| `features` | the five features as they were at the last traversal: what a credit adapts the weighting with |

Derived: `net = (rewarded − punished) / (seen + 1)`, the credit per
traversal, in `[−r, r]` for credits of size `r`.

The five **features** at clock `t`, in order:

| feature | value |
|---|---|
| `seen` | `log1p(seen)` |
| `recent` | `recent · 0.5^((t − last_seen) / life)`, zero if never traversed |
| `net` | as above |
| `age` | `0.5^((t − last_seen) / life)`, zero if never traversed |
| `width` | `log(width_at(t))`, zero at rest |

`traverse(t)` writes the features as they are at `t` to `features`, adds
`trace` (1) to the faded `recent`, sets `first_seen` if unset and
`last_seen`, adds one to `seen`, and widens by `use_widening` (§4).

`credit(r, t)` with `r > 0` adds `r` to `rewarded`, sets `last_rewarded`, and
widens; with `r < 0` adds `−r` to `punished`, sets `last_punished`, and
narrows; with `r = 0` does nothing. Then the weighting adapts with
`features` — the features of the traversal being credited, not the features
now.

## 3. The weighting function

Each edge's `Weighting` is seven numbers: `bias`, a coefficient per feature
(`seen`, `recent`, `net`, `age`, `width`), and `rate`, the edge's own
learning rate. The edge's log-weight is

```
log_weight = bias + k_seen·f_seen + k_recent·f_recent + k_net·f_net + k_age·f_age + k_width·f_width
           + stimulation · f_width
```

The last term is the machine's, not the edge's: the stimulation level in
force when the row is weighed, times the log of the width. It is the whole
of *stimulation makes the wide channel easier*: at stimulation 0 the width
is not consulted; at 1 the widths are the odds; at `σ` an edge `w` times as
wide is `w^σ` times as likely, other things equal.

**Adaptation.** At a credit `c` (signed), with the features `f` of the
traversal being credited:

```
bias  += rate · c
k_i   += rate · c · f_i          for each of the five features
```

each clipped to `[−8, 8]` (`COEFFICIENT_LIMIT`), so no edge can become
certain. A feature that was present when the edge was rewarded comes to
count for the edge; one present when it was punished comes to count against
it. The `bias` is the part that learns regardless of features — the
tabular value of the transition.

**The prototype.** Every edge starts from the machine's prototype
`Weighting`, by default `net = 1` and everything else 0, rate 0.1: an
unadapted edge weighs its own credit per traversal and nothing else, so a
fresh machine is uniform over every row and the first credit already moves
the next draw. The prototype is a setting; a machine can be given another.

**A custom function.** `weight_fn(edge, clock, life, stimulation)` on the
machine replaces every edge's own function in `log_weights` — the edges'
records are still kept and still adapt, but the walk is weighed by the
function given.

## 4. Width, trace and the clock

The machine's clock is the number of traversals anywhere, plus time let
pass by `tick(k)`. Three things on an edge fade on it, lazily — each is
stored as a value and a stamp and read through the elapsed ticks, so reading
writes nothing and there is no sweep:

| what | stored | read at `t` |
|---|---|---|
| the trace | `recent`, `last_seen` | `recent · 0.5^((t − last_seen) / life)` |
| the age | `last_seen` | `0.5^((t − last_seen) / life)` |
| the width | `width`, `width_stamp` | `1 + (width − 1) · 0.5^((t − width_stamp) / life)` |

`life` defaults to 1 000 ticks. **The verdicts never fade**: `seen`,
`rewarded`, `punished` and the clock readings are history, and `net` is read
from them unchanged at any time.

Width moves multiplicatively, always from its faded value at the moment:

| event | width becomes |
|---|---|
| a traversal | `width · (1 + use_widening)` — default 0.01 |
| a reward `r` | `width · (1 + reward_widening · r)` — default 0.2 |
| a punishment `r` | `width / (1 + punish_narrowing · r)` — default 0.2 |
| nothing | relaxes toward 1 by a half every `life` |

and is clipped to `[0.05, 20]` (`WIDTH_MIN`, `WIDTH_MAX`), so the
stimulation term is bounded: at most `σ · log 400` between the widest and
narrowest edge of a row.

## 5. The machine

`Machine(states, alphabet, accepting, start = 0, …)` is in `state` (the
start at first), holds the clock, the current `path` (the edges of the
current run, in order), a stimulation level, and a seeded generator.

**A step.** `step(symbol)`: the row `(state, symbol)` is weighed (§3) under
the stimulation in force (the machine's, or one passed for this step),
turned into a distribution by softmax at `temperature` (greedy at 0, ties
split evenly), a target is drawn, the edge `(state, symbol, target)` is
traversed (§2), the clock ticks, the target state's `visits` and
`last_visited` are written, the edge is appended to the path, and the
machine is in the target. The `Transition` returned records the source, the
symbol, the target, the probability the draw had, the stimulation and the
clock.

**A run.** `run(symbols)` resets to the start with an empty path and steps
through the symbols; the `Run` holds the transitions, the final state, and
whether it is accepting. `log_probability` is the sum of the steps' log
probabilities.

**Quiet.** Any `step` or `run` asked for `quiet` makes and reports the same
draws and moves nothing: not the state, not the edges, not the clock, not
the path. `accepts`, `accuracy`, `transition_table` (the deterministic
machine the greedy walk is) and `stats` are quiet by construction.

**Teach.** `teach(source, symbol, target, amount)` traverses one edge
deliberately (one tick) and credits it `amount`, without moving the machine
or touching the path: a lesson rather than an experience.

**Time.** `tick(k)` advances the clock by `k` with nothing traversed.

## 6. Credit

`credit(r)` lands on the current path: the last edge receives `r`, the one
before it `r · discount`, the one before that `r · discount²`, … (`discount`
defaults to 0.8; the walk back stops when a share is below `10⁻¹²`).
`reward(r)` is `credit(|r|)`, `punish(r)` is `credit(−|r|)`. An edge that
appears twice in a run is credited twice, once per appearance. With an empty
path (nothing run since the last reset, or only quiet runs) credit lands on
nothing and returns 0.

The learning experiment (`teach_language`) is exactly this: run a random
string, reward 1 if the run ended in an accepting state and the string is in
the language or in a non-accepting state and it is not, else punish 1. There
is no baseline, no normalisation, no schedule.

## 7. Stimulation

The machine's level is stored as a value and a stamp and read lazily, like
a width: `baseline + (value − baseline) · 0.5^((clock − stamp) / calm)`.
`baseline` defaults to 1 (the widths are the odds), `calm` to `life`.
`stimulate(x)` sets the level to the current reading plus `x`, floored at
0, stamped now; setting the level outright does the same. A step or a run
may be given a level for itself without changing the machine's.

## 8. Persistence

One JSON object (`.json`, or `.json.gz` gzipped): the format name
`latticefsm-machine` and version, the settings, the clock, the state, the
counts of runs and credits, the stimulation `[value, stamp]`, the
generator's state, and the matrix — its shape, alphabet, prototype, every
`State`, and every edge anything was written to as a flat list in the field
order of §2. An edge not in the file is the prototype's. Both ports write
and read this; the generator state is each port's own (`{"mersenne": …}`
from Python, `{"xoshiro256": …}` from Rust), and a port that finds the
other's reseeds from the seed.

## 9. The two ports

Rust (`rust/`) and Python (`latticefsm/`) implement §§1–8 the same way, down
to the order operations are applied in, and the parity test
(`tests/test_latticefsm.py`, `TestRustParity`) checks that the stimulation
and adaptation experiments — which draw nothing — produce the same records to
`10⁻⁹`, and that a machine trained in Rust loads in Python to the same
greedy table and saves back to a file Rust reads. The generators differ
(§8), so sampled runs do not agree between the ports and the learning
experiment's rows are two samples of one experiment.

The Rust crate has no dependencies: JSON, gzip (the module from
`RadixCyclicNN/rust`), the generator (xoshiro256** seeded by splitmix64) and
the HTTP/1.1 server are written out in `src/`. The server (`server.rs`)
holds one machine behind a mutex, answers JSON under `/api/` (the routes are
in `rust/README.md`) and serves the built frontend (`frontend/dist`) at
every other path, the way the ModelKit servers serve theirs: a path that
would climb out of the directory is refused, and an unknown path is one of
the single-page app's own routes and gets `index.html`.

The frontend (`frontend/`) is `../ModelKit/frontend`'s shape — Vite, React,
plain JSX, one stylesheet (ModelKit's, with a block for the matrix), no UI
or chart libraries, panel settings remembered in `localStorage` under
`latticefsm.v1.`, plain `node --test` tests for the pure modules — with this
model's panels: the matrix slice, runs and credit, training, time and
stimulation, the machine. `dist/` is built and committed, so a checkout
serves it with no `npm`.

## 10. Decisions

* **13 × 13 × 13 by default.** Asked for. It is four to six times the
  states the languages need, which costs learning reliability (README,
  the learning table) and is why training defaults to 4 000 episodes; a
  smaller machine is one flag away.
* **Dense, not sparse.** The request was a matrix, and a record on every
  cell is the point: an edge never traversed still has a width, a prototype
  weighting and a place in the row, so it can be drawn and can then begin
  to adapt. The cost is `S · A · S` records; the model is for machines of
  tens of states.
* **Stimulation as an exponent on width.** `w^σ` keeps the ratio between a
  wide and a narrow edge a function of stimulation alone, is exactly
  "width is the odds" at `σ = 1`, and is a single added term in the
  log-weight, so it composes with the adaptive function without touching
  it. An additive bonus would not make the *comparison* depend on
  stimulation; a threshold would be a different mechanism.
* **One function per edge, adapted by credit times feature.** The simplest
  rule under which an edge's function is its own and changes with its own
  life. The bias alone is enough to learn a transition table (the learning
  experiment); the feature coefficients are what make two edges' functions
  differ in shape, not only in level (the adaptation experiment).
* **Credit at the moment of traversal.** The weighting adapts with the
  features the edge had when it was traversed, not when it was credited,
  so what is learned is "this is what I looked like when this went well",
  which is what a feature coefficient means.
* **Verdicts never fade; traces, ages, widths and stimulation do.** The
  request names both "seen" and "recently traversed", and "rewarded" and
  "last seen": history and recency are different fields and treated
  differently. Forgetting of the verdict itself is a setting this model
  does not have.
* **No baseline in the credit.** Kept out to keep the model the edge
  records and nothing else; the cost is the stuck seeds in the learning
  table.
* **Lazy fading, no sweep.** Everything that fades is a stored value and a
  stamp read through elapsed ticks, as in `RadixDecayNN`, so `tick(k)` is
  one addition and reading never writes.
* **The central node, its code, and compression on a clock** have their own
  decisions in §11.6.
* **The Python port is the reference, the Rust crate is the build.** Python
  for the experiments' authorship and the tests that read like the spec;
  Rust for the CLI and the server, with no dependencies, as the repository's
  other crates. Both read one file format; sampling is not held to parity
  because the two generators are different by design.

## 11. The central node: folding in, walking out

### 11.1 The cube and its centre

A `S × A × S` matrix is a block of cells. Its **central node** is the cell
`(S / 2, A / 2, S / 2)` — `(6, g, 6)` on the default 13 × 13 × 13 machine,
the self-loop of state 6 on the seventh symbol, the one cell with as many
cells on every side as on the other. A cell's **shell** is its Chebyshev
distance from the centre, `max(|s − 6|, |a − 6|, |t − 6|)`. The default cube
has seven shells:

| shell | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| cells | 1 | 26 | 98 | 218 | 386 | 602 | 866 |

`geometry.center_out` lists every cell from the centre outward: shell by
shell, `(s, a, t)` order within a shell. It is the one order the code is
written in and read back in (`geometry.py`, `geometry.rs`).

### 11.2 The code

`compress(machine)` folds the whole matrix into a `Core`, the code the
central node holds. In centre-out order:

| part | what | cost |
|---|---|---|
| the structure | one bit per cell: was anything ever written to this edge | `⌈S·A·S / 8⌉` bytes — 275 for 13³ |
| the integers | per touched edge, `seen` and its five stamps (`first_seen`, `last_seen`, `last_rewarded`, `last_punished`, `width_stamp`), little-endian | 6 × 4 bytes when every one fits in int32, else 6 × 8 |
| the numbers | per touched edge, `recent`, `rewarded`, `punished`, `width`, the six weighting coefficients, the five features | 15 × 8, 4 or 2 bytes by precision |
| the record | the settings, clock, state, counters, stimulation, the states and where each sits, the prototype, the focus learner | about 1 KB of JSON, lossless |

An untouched edge is exactly the prototype's, so a bit is all it costs and it
comes back exactly; a weighting's `rate` is never adapted, so it is the
prototype's too and is not stored. In the file, the bitmap is hex and the
packed fields base64 (`latticefsm-core`, version 1).

| precision | the numbers as | loss |
|---|---|---|
| `exact` (the default) | float64 | none: every field of every edge comes back bit for bit |
| `float32` | IEEE single, round to nearest even | a relative error below 6 × 10⁻⁸ per number |
| `float16` | IEEE half, round to nearest even, clamped to ±65 504 | a relative error below 5 × 10⁻⁴ per number, more for a value past the clamp |

With a `budget` in bytes and no precision named, `compress` takes the least
lossy precision whose code fits, and `float16` when none does. Both ports
write the same code byte for byte, the half-float rounding included (the
Rust crate writes the conversion out; Python's is `struct`'s `"e"`), and
each rebuilds the other's.

### 11.3 Minimal loss, measured

`fidelity(original, rebuilt)` measures the loss two ways:

* **in state**: how many edges differ in any field, and the largest
  relative error over every number of every edge;
* **in behaviour**: per row `(s, a)`, the KL divergence between the
  original's and the rebuilt machine's distributions at the original's
  stimulation (temperature 1 when the original's is 0), and whether the
  rebuilt greedy choice is still among the original's best (two log-weights
  within 10⁻⁶ relative are a tie). The behaviour is *preserved* when no
  greedy choice changed and no row moved by more than 10⁻³ nats.

**Why this code and not a low-rank one.** Two low-rank codes were measured
while designing this, on a default machine taught `even-b` (338 touched
edges, every one on the `a` and `b` slices):

| code | the smallest that preserves the behaviour |
|---|---|
| Tucker decomposition (truncated HOSVD) of the cube's 20-channel tensor, ranks chosen greedily | ranks 13 × 2 × 13 × 19: 7 186 numbers |
| PCA of the touched edges' 20 channels | rank 19: 6 822 numbers |
| the touched edges stored exactly (this code) | 6 760 numbers, and lossless |

The information in a taught matrix is sparse, not low-rank: the symbol axis
collapses to the two slices in use — which the bitmap already captures —
but within them the states need full rank, and a learned table is close to a
permutation. The behaviour is also sensitive: a relative error of 0.24 % in
the channels already moved a row by 7.7 × 10⁻⁴ nats. So the minimal-loss code
is the exact one, and the lossy steps beyond it are precisions, not ranks.

### 11.4 From the middle outward

Because the code runs from the centre out, **rebuilding can stop at any
shell**: `decompress(shells=k)` gives back every cell within `k − 1` shells
of the central node exactly and every cell beyond as the prototype's edge;
`decompress()` is every shell. The machine can also be **walked straight from
the code** — `Core.probabilities` and `Core.run` decode only the cells of the
rows the walk reaches.

**A walk from the middle.** `run(…, from_middle=True)` (Rust:
`run_from_middle`) starts the walk at the middle state, `S / 2` — the
central node's source and target — instead of the start state, and walks out
from there by the same rule; `Core.run(text, from_middle)` does the same
from the code, and agrees with the machine's own greedy walk.

**What that does on the languages.** The four languages are over `a` and
`b`, the first two symbols, and so the outermost two layers of the symbol
axis: shells 6 and 5. A machine taught one of them keeps all it learned
there, and a partial rebuild from the middle gives back nothing it learned
until the last two shells. That is the geometry, measured, not a fault of
the code: a language over the middle symbols would be rebuilt from the
middle first.

### 11.5 Compression on a clock

`Machine(compress_every=N)` folds the matrix into its central node every `N`
transitions — a traversal in a run, in training, or in a lesson (`teach`);
a quiet run is no transition. The code is kept on the machine (`core`), with
`compressions`, `since_compression` and `last_compressed`; all of them and
the schedule are saved in the machine file.

* **Recorded** (`compress_rebuild` off, the default): the central node's code
  is kept current and the machine is untouched.
* **Rebuilt** (`compress_rebuild` on): after each compression the matrix is
  rebuilt from the code, so the precision's loss is *applied* to the machine
  and carried into what it learns next. At `exact` this changes nothing — a
  test holds a machine taught with an exact rebuild every 50 transitions to
  the same matrix, edge for edge, as one taught without. At `float16` the
  machine's numbers are rounded to three digits every `N` transitions.

The rebuild writes the decoded fields into the matrix in place (Python keeps
every edge object, Rust every offset), so a run's path is still valid and a
credit after the run lands where it should.

### 11.6 Decisions

* **The central node is the geometric centre.** Asked for, and well defined
  for any shape. It is where the code is held and where the outward order
  starts; it is not where a taught machine's information is (§11.4).
* **Exact by default.** "Minimal loss" read as no loss when no loss is
  possible, which it is, at 7.5 to 12 times smaller than the dense matrix on
  a taught machine and 1 342 times on a fresh one. The lossy precisions are
  there for a budget.
* **Low-rank codes rejected,** measured (§11.3), not assumed.
* **The code is laid out centre-out** so that "traverse from the middle
  outward" is a property of the code itself: a prefix of it is a rebuild to
  a shell.
* **Compression on a clock counts transitions,** not clock ticks: `tick`
  lets time pass without the machine doing anything, and a quiet run asks
  without traversing.

## 12. Where a run starts and how it moves

### 12.1 Focus on the central vertical vector

The **central vertical vector** is the column of cells through the central
node along the source axis — the axis the matrix is drawn with top to
bottom: `(s, A / 2, S / 2)` for `s` from 0 (the top) to `S − 1` (the
bottom), `(0..12, g, 6)` on the default machine (`geometry.central_vertical`).

**Focus** is a number in the finite range `[0, 1]` (`FOCUS_RANGE`; a value
outside it, infinite or not a number is refused). It translates to a node of
the vector by cutting the range into `S` equal bands:

```
focus_index(S, f) = min(S − 1, floor(f · S))      no focus: 0
```

| focus | none | [0, 1/13) | … | [6/13, 7/13) ∋ 0.5 | … | [12/13, 1] |
|---|---|---|---|---|---|---|
| node | top (0) | top (0) | … | the central node (6) | … | bottom (12) |

A run starts from the source state of the node the focus picks: low or no
focus the top node, high focus the bottom one. The machine holds a focus
(`focus`, saved; `None` by default), and a run may be given one for itself.
The order a run's start is decided in: the middle state (`from_middle`), a
focus given for the run, the learned focus (§12.2), the machine's focus, and
— with none of these — the machine's `start`, which is the top node unless
the nodes have rearranged themselves (§12.4).

### 12.2 The learned focus

With `learn_focus`, the focus is read off each input by a `FocusLearner`
(`focus.py`, `focus.rs`):

```
focus(text) = sigmoid(w · x(text))
x = [1, n / (n + 6), each symbol's share of the text, the first symbol one-hot, the last symbol one-hot]
```

`2 + 3A` weights, 41 on the default machine, starting at a bias of −3 and
nothing else: an untrained learner says 0.047, the top node, as no focus
does. A run that learns explores — it draws `f = mu + explore · N(0, 1)`
(`explore` 0.15) and starts from the node the clipped draw picks — and when
the run is credited `c` (a reward or a punishment, the same call that
credits the edges),

```
w += rate · (c − baseline) · (f − mu) / explore² · mu (1 − mu) · x       rate 0.05
baseline += 0.05 · (c − baseline)
```

the gradient of a Gaussian policy's log-likelihood through the sigmoid,
measured against the credit's running mean. Learning from the unclipped
draw and against a baseline matters: an earlier version that learned from
the clipped draw drifted to the bottom node, because noise clipped at 0
biases the draw upward while most credit is positive. Weights are clipped
to [−8, 8]; a quiet run reads the mean and does not explore. The learner,
its baseline and the switch are saved with the machine and in its code.

### 12.3 Skipping a node

With `skip` on, before each move with a symbol left to look at, the machine
compares two ways to read the next two symbols `a`, `b` from state `s`:

* the **step's path**: the step it would take on `a` anyway (greedy or a
  draw, as always), then the best edge after it on `b`;
* the **best path**: the best two-edge path on `a` then `b` through any
  node `t`.

Each is scored by its summed log-probability, at the run's temperature, or
the machine's, or 1 when greedy. If the best path beats the step's by more
than `skip_margin` (default 0: strictly), the machine **skips**: one move,
reading both symbols, from `s` straight to the best path's end, past the
node between. Both edges are traversed and credited, the clock ticks twice;
the node skipped is not visited. The move is recorded as one `Transition`
with both symbols and `skipped` naming the node passed. Without skips
nothing changes — not even the random draws — which a test holds edge for
edge. The walk from a compressed code (`Core.run`, `Core::walk`) skips as
the machine's quiet greedy run does.

### 12.4 Swapping and rearranging nodes

`swap_states(i, j)` makes states `i` and `j` trade places everywhere:
every edge from or to one moves to the other's row and column (an edge
object keeps its identity in Python; Rust remaps the run's path offsets),
the two state records trade places, and the start state and the current
state follow. `swap_symbols(a, b)` does the same for two symbols' slices
and labels, and keeps the learned focus's per-symbol weights with their
symbols. A swap is a relabelling: from its start state the machine walks
exactly as before (a test holds every edge's weight at its new
coordinates). What changes is **where** each node sits — which one a focus
picks, which is the middle state, which shell of the compressed code it
falls in. `lattice.state_ids` remembers which state, by the number it was
made with, sits at each position.

`rearrange()` lets the nodes do it themselves. A node's load is the
traversals of every edge through it (a state's edges in and out; a symbol's
slice). One pass, on each axis: from the outside in on each side of the
centre, a pair of neighbours swaps when the one farther from the centre is
busier. The busy nodes move inward a step per pass; `full=True` repeats
passes until none swaps, which leaves each axis busiest at the centre and
falling away on both sides. `rearrange_every=N` runs one pass every `N`
transitions — deferred to the end of the run or lesson it falls in, so that
nothing a run holds (the symbols still to read, a skip's target) refers to a
position that moved under it.

This is what makes the outward rebuild of §11.4 useful on a taught machine.
Before rearranging, every edge `even-b` taught lies in shells 5 and 6,
because `a` and `b` are the outermost symbols; after, `a` and `b` sit at the
centre of the symbol axis and the busiest states at the centre of the state
axis, and the touched edges reach every shell down to the central node
(README, "Where a run starts").

### 12.5 Decisions

* **"Vertical" is the source axis**, the one the matrix is drawn with top
  to bottom, so the top node is state 0, the default start: no focus and
  focus 0 agree on a default machine.
* **Equal bands, not rounding**, so every node of the vector gets the same
  share of the range, the ends included.
* **No focus falls back to `start`**, not to position 0: `start` is
  position 0 until the nodes rearrange, and the rearranged machine keeps
  starting from the state it learned from.
* **Learned focus as a Gaussian policy with a baseline,** the simplest
  learner that is credited by the same rewards and punishments as the
  edges; its measured effect is mixed (README), and it is off by default.
* **A skip is a lookahead of one symbol,** scored in probability, the
  direct reading of "a more efficient path"; it never changes what is read,
  only how many moves it takes and which edges carry it.
* **Rearrangement swaps neighbours toward the centre,** a local rule, so the
  arrangement emerges from use; a swap's relabelling is exact, so it costs
  nothing in what the machine has learned.
