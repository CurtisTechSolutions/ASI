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
| a module like TensorFlow, to connect several of these together | §13.1–13.3: `Network`, `Layer`, `Sequential`, `Parallel`, the architecture string, compile / fit / evaluate / predict |
| encoder and decoder layers: text to its phonetic representation with the repository's tokenizer, English in and out | §13.4: `PhoneticEncoder`, `PhoneticDecoder`, `translate`, `fit_pairs` |
| train on text, audio, images and video (images / differences and audio) | §14: the text, audio, image, diff and video encoders |
| predict the next thing with N-grams, trained automatically from a corpus and with Ollama | §15: `NGram`, Kneser-Ney over adaptive edges; `autotrain` |
| a base64 encoder and decoder for text, images, video..., and N-gram learning over it | §16: `b64`, the `base64` unit |
| a chain of nodes, one input a node, the output node predicting; prediction depth shrinking with confidence (accuracy and reward) | §17: `Chain`, the gathered-confidence walk |
| teach a reply, and train, until N of N | §18: `teach`, `drill`, `ceiling`; the network's `until` |
| parse zip files of unlimited size, and other file types, in the corpus | §19: `corpus`, `archive` |
| a frontend with the ModelKit frontend's features: predictions as chat, uploads, checkpoints as a tab | §20 |
| the guard (ModelKit's negative network) applied to this network | §21: `Guard`; `guard: true` on the routes |

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

## 13. Networks of machines

`network.py` and `network.rs`, the same in both ports, writing and reading the
same `latticefsm-network` files.

### 13.1 A layer writes

A machine reads a string; a **layer** is a machine that also writes one. With
`emit = "states"` it writes the state each move lands in (the target of every
transition: one symbol per move, a skip one symbol for its two); with
`emit = "final"` only the state the run ends in. State `i` is written as the
`i`-th of `a-z A-Z 0-9`, so a layer has at most 62 states, and the block after
it is built over exactly the letters it writes. Each layer's machine is
seeded with the network's seed plus the layer's index, depth first.

### 13.2 Blocks

A block is built over an input alphabet, reads a string, writes one, and has a
verdict. A `Layer`'s verdict is its run's. A `Sequential` passes what each
block writes to the next; its verdict is the last block's. A `Parallel` gives
each branch the same input and merges what they write - `concat` one branch
after another, `interleave` position by position - with branch `k`'s letters
shifted past the letters of the branches before it, so the merged alphabet is
the branches' alphabets side by side (62 at most). Its verdict is a vote of its
branches': `majority` (strictly more than half), `any` or `all`. The network's
root is a `Sequential`; its verdict is the network's. The architecture string
is `seq := atom ('>' atom)*`, `atom := N | Nf | 'enc' | 'dec' | '(' seq ('|' seq)* ')'`,
a group of one being just that sequence.

### 13.3 Credit

`teach(x, y)` runs every machine on `x` (traversing), compares the network's
verdict with `y`, and calls `credit(r)` on the root with `r = reward` when it
is right and `-punish` when it is wrong. A `Sequential` credits its last block
`r`, the one before it `r · layer_discount`, and so on back; a `Parallel` gives
every branch `r`; a `Layer` spends `r` on its machine's path (§6). With
`credit = "own"` the output layers - the last layer, or every branch's last in
a final `Parallel` - are credited by their own verdicts instead
(`reward` if theirs was `y`, else `-punish`); the layers before them by the
network's, as always. A network of one layer is therefore exactly its machine:
the same strings, the same credits, the same edges.

Greedy passes (`predict`, `evaluate`, `trace`, `translate`) run quietly at
temperature 0 with `Machine.greedy_first`: of tied targets the lowest is
taken and nothing is drawn. A machine on its own still draws among ties
(unchanged); a network's measurements are deterministic, leave the
generators alone, and agree between the ports.

### 13.4 English in and out

`PhoneticEncoder` reads English with the repository's phonetic tokenizer
(`../PhoneticTokenizer`: phoneme level, the portable lexicon - the core
lexicon, the letter-to-sound rules and `PHONETOK_LEXICON` - which its Python
and Rust ports read alike), keeps the tokens in its inventory - `#`, `, . ?`,
then the 39 phonemes without stress (`stress=True` keeps the digits: 88) -
and appends `pad = 2` boundaries `#`. It is the network's first block or
nowhere; it has no machine and takes no credit.

`PhoneticDecoder` is a layer whose states are that inventory, so what it
writes is sounds; the network spells them back as English with the
tokenizer's `decode`. It writes one sound per symbol, starts from `#` (state
0), skips nothing and is never rearranged - a state *is* its sound. Before
spelling, the tokenizer hears every word of every pair the network was taught
(`words`, saved in the file), so a word it was taught comes back as itself
from its memory in any process.

`teach_pair(x, y)` runs every machine on `x`; the decoder's targets are `y`'s
sounds, padded with `#` to the number of moves it made or cut to it. Each move
whose target is not the target sound has its edge credited `-punish`; then
the target path is taught edge by edge from `#` (`Machine.teach`, `+reward`).
The blocks before the decoder are credited `reward · (right / moves) ·
layer_discount`, discounted back as in §13.3 - a reward only: punishing them
for the decoder's errors kept reshuffling the walks the decoder was learning
to read, and no hidden layer learned at all (README). `evaluate_pairs` counts
a pair right when the sounds written, trailing `#` dropped, are the target's.

### 13.5 Decisions

* **A layer writes its states,** the one thing every machine has: no output
  alphabet to design, and the next machine reads the walk itself.
* **Parallel letters are shifted, not shared,** so branch 1's state 0 and
  branch 2's state 0 stay distinguishable to the block after.
* **Credit travels back through blocks the way it travels back along a run,**
  discounted, so a network adds one parameter (`layer_discount`) and no new
  learning rule; `own` exists because shared credit punishes a branch that
  was right whenever the vote was wrong.
* **The decoder is supervised, the rest reinforced.** A target sound at every
  step is information a single end-of-run verdict throws away, and the
  decoder's state being its last sound makes the plural and the past tense -
  a suffix chosen by the sound before it - a row each.
* **Sounds, not letters,** between English and the machines: 43 symbols that
  mean the same thing in every word, where spelling would make `-s`, `-es`
  and `-ies` three rules.

## 14. Text, audio, images and video

An encoder (§13.4) is any first block that turns an input into symbols;
`media.py` and `media.rs` add five, in exact integer arithmetic so the ports
agree symbol for symbol.

* **text**: lower-cased; whitespace is `_`; kept are `a-z 0-9 _ . , ? ! ' -`.
* **audio**: a WAV file through the phonetic tokenizer's `AcousticTokenizer`
  (its bundled 64-unit codebook, a run of one unit folded into one).
* **image**: decoded to brightness `Y = (299 R + 587 G + 114 B) // 1000`
  (PNG 8-bit of every colour type, unfiltered with all five filters; PPM /
  PGM `P2 P3 P5 P6`, scaled by `255 // max`; BMP 24/32-bit, bottom-up or
  top-down); averaged into `grid x grid` cells, cell `(cy, cx)` covering rows
  `cy·h // grid` up to `max(that + 1, (cy + 1)·h // grid)` and likewise
  columns, the integer mean; each cell's level `value · levels // 256`;
  read row by row, or `center-out`: by ring `max(|2r - (g-1)|, |2c - (g-1)|)`,
  then row, then column.
* **diff**: for consecutive grids, `d = after - before`; `|d| < 8` is `D0`,
  `< 48` is `D±1`, else `D±2`, the sign of `d`; each frame's after a `|`.
* **video**: frames are a directory's images in name order (and its
  `audio.wav`), or a video file ffmpeg turns into `fps` (8) frames a second of
  64 x 64 and a 16 kHz mono track; the first `frames` (16) are read; frame
  `t` is `|`, then itself as an image if `t = 0` or `t` is a multiple of
  `keyframe` (0: never again), else its difference from frame `t - 1`, then
  the audio units `units[t·U // F : (t+1)·U // F]`.

A media encoder keeps each file's symbols by absolute path, size and
modification time. Labels come from `input<TAB>verdict` files, paths
relative to the file; every fifth example is held out (every fourth would
hold out one class of the alternating synthetic sets). The synthetic sets
(`make_dataset`) are drawn from each port's own generator.

### 14.1 Decisions

* **Symbols, not features.** A machine reads a string; everything is turned
  into one, so nothing about the machine changes for a new kind of data.
* **Keyframes and deltas for video,** the representation video codecs use:
  most of a clip is what changed, and what changed is what a moving thing is.
* **Integers throughout,** so a picture is the same string in both ports and
  a network file classifies it the same way in either.
* **The repository's own acoustic units for audio,** the tokenizer the
  network already uses for English, rather than a second front end.

## 15. Predicting the next thing: N-grams

`ngram.py` and `ngram.rs`, the same model number for number.

### 15.1 The machine and its edges

The machine's state is the last `N - 1` token ids, the start padded with
`<s>` (id 0); `</s>` (id 1) ends every sequence and `<unk>` (id 2) stands for
a token never trained on. Ids are given in the order tokens are first seen.
`grams[context][token]` counts how often `token` followed `context` (the
highest order only; nothing else is stored). An edge that has been credited
also has a record `[width, rewarded, punished, last]`; `last` is the clock
(tokens trained on) when it was last taken.

### 15.2 Kneser-Ney

For order `k = N` the table is `grams`; for each `k < N` it is the
continuation counts of the order above: `cont_k[h'][w]` is the number of
distinct `x` with `count_{k+1}[x h'][w] > 0`. Per table: `total[h]`,
`distinct[h]` (tokens with a count), and the discount
`D_k = n1 / (n1 + 2 n2)` from its count-of-counts (0.75 when undefined or
outside (0, 1)). Then, with `p_0(w) = 1 / (V - 1)` (0 for `<s>`) and for `k`
from 1 to `N`, `h` the last `k - 1` ids of the context:

    p_k(w) = max(c_k(h, w) - D_k, 0) / total_k(h) + (D_k · distinct_k(h) / total_k(h)) · p_{k-1}(w)

when `h` has a row in table `k`, else `p_k = p_{k-1}`. `P_KN = p_N`. With
credited edges out of the context `h`:
`P(w | h) = P_KN(w | h) · width(h, w)^s / Z`, `Z = 1 + Σ P_KN(v | h)(width(h, v)^s - 1)`
over the credited `v` in id order (`s` the stimulation). The full distribution
is computed order by order with exactly these operations, so it equals the
token-by-token value bit for bit (a test checks it in each port).

### 15.3 Tokens

`word`: lower-cased; runs of alphanumerics and `_` (an apostrophe between a
word character and a letter stays inside), every other non-space character
alone. `char`: every character of the stripped line, whitespace as `_`.
`sound`: the phonetic tokenizer's phonemes without stress, through one
tokenizer with its word memory off, after numerals other than `0-9` are
removed. `base64`: §16. Detokenizing joins words with a space except before
`. , ! ? ; : ) ] } % ' "` and after `( [ { $ "`.

### 15.4 Measuring, generating, credit

`evaluate` sums `ln P` over every token and `</s>` of held-out sequences:
perplexity `exp(-Σ / n)`, bits `-Σ / n / ln 2`; over the first `max_ranked`
positions, accuracy is the share where the target is the top token (ties to
the lower id), top 5 where it is among the first five. Greedy generation
takes the top token each step (never `<s>` or `<unk>`) until `</s>`; sampling
draws from `P^(1/T)`. `credit_text(text, r)` credits every edge the text
takes, `</s>` included: `r > 0` multiplies the width by `1 + 0.2 r` (at most
20) and adds `r` to rewarded; `r < 0` divides it by `1 + 0.2 |r|` (at least
0.05) and adds `|r|` to punished.

### 15.5 Predicting while training

`train_tokens(tokens, predict, credit)` is prequential: at each position it
takes `top(ctx)` and `P(w | ctx)` from the model as it stands, adds
`[guess == w]`, `1` and `ln P` to the running `online` totals, and only then
counts `(ctx, w)`. `reset_online()` returns the tokens predicted, the share of
first guesses right, perplexity and bits, and starts over; `train`,
`/api/ngram/train`, `/api/ngram/text` and every autotrain round (`learned`,
`learned_accuracy`, `learned_perplexity`) report it.

Two things make it cheap. **The tables are incremental:** once derived, each
order keeps its rows, totals, distinct counts and `n1`, `n2`; adding `(h, w)`
at order `k` adjusts those, and when it is the first time `w` follows `h`
the continuation count of `(h[1:], w)` at order `k - 1` goes up by one, the
same step recursively - the tables always equal a rebuild from the counts.
**The first guess skips the distribution:** a token in no row of orders
`2..N` along the context, and not credited there, gets `P = Λ · p_1(w)` with
the same `Λ` for every such token, so among them the order-1 ranking decides.
Order 1 is kept as buckets count → ids, so `top` scores only the tokens in
those rows, the credited ones, and the best outsider by order 1 - the same
id the full argmax picks (ties to the lower id, never `<s>`), which the
tests check position by position.

With `credit = R`, before the count a right guess has `(ctx, w)` credited
`+R`, a wrong one `(ctx, guess)` credited `-R`. Measured on held-out lines it
does not help (word trigrams: perplexity 180.07, 178.45, 190.5 at R = 0,
0.25, 1; char 5-grams 4.485, 4.695, 5.085), so `R` defaults to 0 and
prediction itself is on by default: it costs little and changes nothing the
model learns.

A long run predicts a sample, so a big corpus is not slowed by it.
`predict_now(seen) = seen % (1 + seen / PREDICT_ALL) == 0`, with
`PREDICT_ALL = 50000`: every position of the first 50 000 since the last
`reset_online`, then every 2nd, every 3rd, and so on. The positions are the
same in both ports, so the measures still agree, and the first 50 000 are
exact. In Rust, `top` scores the rows level by level, highest order first,
and keeps every word within 1e-9 of that level's best. A word in a higher row,
or credited, is left to the end. The best outsider comes from the order-1
ranks, with the credited words left out. The exact `prob` of those few
candidates then decides, so the guess is the full argmax with no
distribution built. The chain's guess builds its distribution once, densely,
and takes both the argmax and `P(w)` from it.

Measured on 500 KB of text, training with prediction took 20.6 s for the
N-gram model and 5.45 s for the chain; it now takes 1.47 s and 1.77 s.
Training on a 10 MB file took 132 s and now takes 48.6 s.

### 15.6 Automatic training

`autotrain` trains on `corpus_sequences(data)` (every `held_out`-th sequence
of every file kept back), records round 0, and for each round `r` with Ollama:
for `topics_per_round` topics `topics[((r-1)·t + j) mod |topics|]` (given, or
the eight most frequent tokens of five or more letters that are not stop
words) asks `CORPUS_PROMPT` at temperature 0.8, seed `1000·seed + 10r + j`,
and trains on the parsed lines (numbering, bullets, quotes and preambles
stripped); takes `continuations` prefixes - the first `prefix_words` tokens
(base64: `4 ·` that many letters) of training sequences at a stride of
`⌊n / count⌋` from offset `7919 r` - asks `CONTINUE_PROMPT` (temperature
0.7, 32 tokens) and trains on prefix + the first line of the reply; writes
`judge` greedy texts from two-token prefixes (offset `104729 r`), asks
`JUDGE_PROMPT` (temperature 0) and credits each judged text `+1` or `-1`;
then records the round and stops after `patience` rounds without a lower
held-out perplexity.

### 15.7 Decisions

* **Sparse, not dense.** The lattice's dense matrix holds a trigram over 13
  tokens and no more; an N-gram over a real vocabulary has to keep only the
  edges it has seen.
* **Kneser-Ney, because it is the right baseline:** continuation counts are
  what make a low order useful, and the lattice's own credit sits on top of
  it rather than replacing it, so a model nobody has judged is a standard one.
* **Predict before learning,** so every token of training is also a test
  on text the model had not seen yet - a measure that comes free with
  training and needs no held-out set.
* **A held-out set that never changes during autotrain,** so each round's
  number means the same thing, and an early stop, so a teacher whose text
  hurts the corpus's likelihood is noticed instead of trusted.
* **An LLM as teacher three ways** - more text, the next words where the
  model asks, and a verdict on the model's own writing - each of which the
  model can use without the LLM's probabilities, which Ollama does not
  reliably expose.

## 16. Base64

`b64.py` and `b64.rs`: the RFC 4648 alphabet `A-Z a-z 0-9 + /` with `=`
padding. `decode` ignores every character outside the alphabet, reads a last
group of two or three letters as one or two bytes and drops a lone last
letter. `encode_prefix(data)` is the first `⌊8 · len / 6⌋` letters of
`encode(data)`: only letters whose six bits are all known, a prefix of the
encoding of every continuation of `data`.

The `base64` unit tokenizes a text as the letters of its UTF-8 (no
lower-casing), a prompt as `encode_prefix` of its UTF-8, and a file - any
file, every one under a directory but hidden files and `README.md` - as
consecutive 57-byte pieces, each encoded to 76 letters, one sequence each
(57 bytes is 19 groups of three: no group is split). Text is written back by
decoding the prompt's letters and the generated ones together;
`generate_bytes` continues a file's first bytes and starts a fresh sequence
whenever one ends, until `max_tokens` letters or a padded end.

* **One alphabet for everything:** the model and its machinery do not change
  for a picture or a video; what changes is only which letters follow which.
* **Cut at whole groups** so a letter always means the same six bits of the
  same byte position in its group, the structure an N-gram can learn.



## 17. The chain

`Chain(nodes, unit, lower, confidence)` (`chain.py`, `chain.rs`): `nodes - 1`
input nodes and the output node. `tables[k]` holds the nodes of depth `k`
by context - the last `k` inputs, `<s>` before the first (depth 0 is the
output node, context `()`); a sequence is `<s>`, its inputs, `</s>`, and at
position `i` the contexts of depths `0..min(nodes - 1, i)` exist.

A `Node` keeps `counts` (input id -> how often it came next), `total`,
`best` (the most frequent, ties to the lower id, kept current at every
`add`), `tested`, `right`, `rewarded`, `punished`. Its confidence is
`(right + rewarded) / (tested + rewarded + punished + PRIOR)`, `PRIOR = 2`.

**Training** (`train_tokens(tokens, predict, credit, reward)`), at every
position `i` with next input `w`: with `predict`, the whole chain first
predicts `w` - the walk, its guess and `P(w)` added to `online` (right,
predicted, `Σ ln P`, nodes traversed) - and with `credit` the nodes on the
walk whose `best` is the guess are rewarded `credit` (if it was right and
`reward`) or punished `credit` (if wrong); then every node of depth
`0..min(D, i)` along the context that already exists is **tested** (`tested
+= 1`, `right += best == w`), and every one, existing or new, learns `w`.

**The walk** after a history `ids` (`<s>` and the inputs), at threshold
`τ`: start at the output node, `doubt = 1 - c_0`; while `1 - doubt < τ` and
`k < min(D, len(ids))`, take the node of depth `k + 1` along `ids` - stop if
it does not exist - and `doubt *= 1 - c_{k+1}`. The walk is the nodes
taken. `τ = 0` stops at the output node; no node's confidence reaches 1, so
`τ = 1` goes as deep as the context exists.

**The distribution** of a walk `n_0..n_d`: `P_{-1}(w) = 1 / (V - 1)` (0 for
`<s>`), `P_k(w) = (c_k(w) + T_k P_{k-1}(w)) / (N_k + T_k)` (`T_k` distinct
inputs seen, `N_k` their total), in that order, integer sums before the
division. **The guess** is its argmax (ties to the lower id), found without
the whole distribution: an input no node past the output node has counted
is ranked by the output node's counts alone, so the candidates are the
inputs in nodes `1..d` and the best of the rest by the output node's ranks.

`credit_text(text, r)`: at every input of the text and `</s>`, every node on
the walk whose `best` is that input is rewarded `r` (or punished `-r`).
`evaluate` (perplexity, bits, accuracy, mean depth, walks per depth) and
`sweep` (evaluate at each threshold) move nothing. The file:
`{format: "latticefsm-chain", nodes, unit, lower, confidence, clock,
sentences, vocab, table: [[k, *context, tested, right, rewarded, punished,
w, c, w, c, ...]], sources, history}`, rows sorted by `[k, *context]`;
`best` and the output node's ranks are rebuilt on reading.

### 17.1 Decisions

* **Gathered, not first-above-threshold.** Stopping at the first node whose
  own confidence reaches `τ` lets one confident-but-shallow node decide;
  gathering `1 - Π(1 - c)` lets every node shorten the walk a little, so the
  depth falls continuously as confidence rises. Measured on letters it was
  as accurate or more at every depth, with a lower perplexity.
* **A node's own accuracy, measured while training,** is its confidence:
  what it would have said, not what the whole chain said, so a node earns
  trust by itself; two imagined misses keep an untested node at 0.
* **Credit to the nodes that guessed it,** not only to the node the walk
  stopped at: a reward that reached only the deepest node would never let
  the nearer nodes stop a walk sooner.

## 18. Teaching until N of N

**`teach(text, reply, max_rounds)`** (both models): `of = len(reply) + 1`.
A round: at every position of the reply and its `</s>`, the first guess
after `text + reply[:i]` is taken; `right` counted; if `right == of` or
`max_rounds` rounds have run, stop. Else every wrong position is corrected
by `a = 2^s` (`s`: the rounds that position has already been wrong in a
row) - the chain punishes the nodes on its walk whose `best` is the guess;
the N-gram narrows the guess's edge (`credit(ctx, guess, -a)`, `<unk>` never)
- and `text + reply` is learned once as a sequence.

**`ceiling(seqs)`**: the contexts of `seqs` (the N-gram's `order - 1`
tokens; the chain's whole reachable context, `min(D, i)` inputs) with what
follows each and how often: `Σ max` is the most first guesses any model of
that context length can get right. **`drill(seqs, max_rounds, patience)`**:
the target of a context is its most frequent next token (ties to the lower
id). A round counts the right first guesses over `seqs` (teacher-forced);
stop at the ceiling, `max_rounds`, or `patience` rounds without a gain (the
CLI and the server pass `patience = max_rounds`). Both take a `threshold`
(the command line and the server default to `THRESHOLD = 0.95`): the goal is
`goal_of(threshold, n) = ceil(threshold * n - 1e-9)` of the reply's `n`
tokens or of the ceiling (`n` itself at 1), and they stop once that many are
right; the network's teach stops at that share of a round (and, only at 1,
with its answer the reply), its fit at that share of the pairs. Else, contexts in sorted
order, every one whose guess is not its target, `a = 2^k` (`k`: the rounds
it was corrected before): the N-gram narrows the guess's edge by `a` and
widens the target's by `a`; the chain punishes by `a` every node on its walk
whose `best` is not the target, and if the deepest node of the context
prefers another input, or the walk reached it and still guessed wrong, adds
to it as many of the target as make it its `best` (`counts[best] - counts[t]
+ [t > best]`) plus `2^k - 1`. `train --until` is round 1 (the training pass,
measured while learning) and then the drill.

The network's `teach` with `until` repeats rounds of `times` teachings
until a round's sounds (or verdicts) are all right and its answer is the
reply; `fit` with `until` repeats epochs until every taught pair is exactly
right (or every taught example, by the quiet pass).

### 18.1 Decisions

* **A ceiling, not N of N:** when the data continues one context two ways no
  model that reads only that context can get both right; looping until N of
  N would never end. The ceiling is computed and printed, and reached.
* **Doubling.** A node right a thousand times elsewhere needs a large
  punishment to give way; a fixed one never gets there, doubling gets there
  in a handful of rounds and does not overshoot by more than a factor of two.
* **Targets, not the last wrong guess:** correcting toward each context's
  most frequent token settles; correcting toward whatever was just wrong
  flips a two-way context back and forth for ever.
* **No retraining in the drill:** a pass over the same data again raises
  every node's tested and right and so its confidence, shortening walks and
  undoing the punishments; teaching a reply learns it again because its
  tokens may never have been seen.

## 19. Reading a corpus

`corpus.files(paths)`: a named file as it is; a directory walked in name
order, hidden files and directories and `README.md` left out, and files of
no known kind too (unless everything is wanted, as for `base64`). The kind
of a file (or member) is read off its name (`file_kind`). An archive is
expanded: `.gz` into the file it holds, whose kind is the name without
`.gz`; `.tar` and `.tar.gz` / `.tgz` into their regular members in order;
`.zip` into its members in central-directory order - directories, hidden
members (any part beginning `.` or `__MACOSX`), encrypted ones and methods
other than stored and deflated skipped - each a document labelled
`archive/member`, archives followed `MAX_DEPTH = 4` deep. A text document's
lines: `text` and `jsonl` read `readline(MAX_LINE = 2^20)` at a time,
decoded UTF-8 with replacement, split as Python's `splitlines`, stripped,
empty dropped (`jsonl`: each line's JSON value's text, or the line); `json`,
markup and documents read whole. `strip_markup`: comments, `<!...>`,
`<?...?>`, scripts and styles dropped, CDATA kept, a tag whose local name
(lower-cased, after its last `:`) is a block (`p h h1..h6 div br li tr title
section article blockquote pre table ul ol dt dd hr header footer nav aside
caption figcaption`) becomes `\n`, `td th tab` a space, every other tag
nothing; then `&amp; &lt; &gt; &quot; &apos; &nbsp; &hellip; &mdash; &ndash;
&lsquo; &rsquo; &ldquo; &rdquo; &copy;` and `&#N;` / `&#xH;` decoded.

`archive.rs` reads what Python's `gzip`, `zipfile` and `tarfile` read: a
deflate decoder that keeps only its 32 KiB window and decodes block by block
into a buffer as it is read; gzip members one after another (zero padding
between allowed), each CRC and length checked; the zip end record (the last
`PK\x05\x06` in the last 64 KiB + 22 bytes), the ZIP64 end record before
its locator, ZIP64 extra fields, CP437 names unless flagged UTF-8, data
prepended to the archive allowed for, every member's CRC and length checked
at its end; tar headers with their checksums, ustar prefixes, GNU long names,
pax `path` and `size`, base-256 sizes, stopping at a zero block.

Training streams (`ngram.train_corpus`): every sequence of every document
goes straight to the model, every `held_out`-th of each document kept back
instead - at most `MAX_HELD_OUT = 10 000` - and one source per file (an
archive is one).

## 20. The frontend and the server around the models

The chain and the N-gram answer the same routes (`/api/{chain,ngram}/...`:
train, text, predict, generate, credit, feedback, score, teach, save; the
chain's new, load, confidence, sweep, graph; the N-gram's new, load,
ollama, autotrain) through one trait, `routes::Lm`. Uploads (`/api/uploads`)
are written under the upload directory, names sanitised part by part, a
file sent in pieces with `append`, so a file may be any size; a training
body's `uploads` names them. Checkpoints (`/api/checkpoints`) are model
files named `{model}--{tag}--{seconds}.json.gz` in the checkpoint directory,
listed newest first, restored into the model their name says, downloaded
with `Content-Disposition`, uploaded in pieces. `/api/converse` continues
each turn with the next; `/api/chat/llm` has Ollama say a line, the model
continue it, Ollama mark it out of 10 and the reply credited `(mark -
pass) / 5`; `/api/status` summarises every model. The frontend's tabs are
in README.md.

**Jobs.** A `POST` with `"job": true` is answered at once with `{job}`. The
same request, minus the flag, then runs through the router on a thread of
its own. Its result, or its error, is kept on the job: `GET /api/job` returns
the running job or the last one, and `POST /api/job/stop` asks it to stop.
Only one job runs at a time, and a second gets 409. The model code reports
progress through a thread-local handle (`job::tick`, `job::phase`,
`job::stopped`, in `job.rs`). On a thread with no job (the command line, the
tests) that handle is empty, so each call does nothing. Corpus training
stops between sequences, and drilling stops between rounds. What was
learned up to then is kept, and the job ends `stopped`. A panic ends the job
`failed`, never leaving it `running`.

**Never waiting behind a lock.** Outside a job, a request takes a model's
lock only when it is free (`Service::take`, `try_lock`). If it is held, the
request gets 409 at once, with `busy`, the message and the job. The job's
own thread waits for its locks as before. `/api/status` takes every lock
with `try_lock` and reports a held model as `{"busy": true}`. Before this,
one long training request held the model's lock: every other request to it
queued behind, the status bar's poll hung, and the page looked frozen.

**Memory.** Most of the chain's contexts are followed by one to three
inputs, so a node keeps its counts as a sorted list and switches to a map
past 32 (`chain::Counts`). Saving writes the table's text straight out
(`Chain::dump`, the same text as `to_json().dump()`) instead of first
building a tree of JSON numbers. Measured on a 10 MB text file, training
peaked at 641 MB instead of 769 MB, and saving a 2 MB one's chain at 293 MB
instead of 838 MB. The rest is the tables of contexts themselves.


## 21. The guard

`Guard(unit = "word", order = 3, lower, threshold = 1, min_coverage = 0.5)`
(`guard.py`, `guard.rs`). A text's transitions are `(*the last order - 1
tokens, next)` over `<s>`-padded tokens and `</s>`; `edges` maps each to
`blame`, `fails`, `clear` and `reasons` (`{reason: blame}`).

* `blame(text, reason, severity)`: the reason cleaned (whitespace collapsed,
  ASCII lower-cased, 60 characters, `unspecified` when empty); every
  transition gains `severity` blame, a fail and the reason's share; a ninth
  reason drops the smallest (ties: the alphabetically first); the reason's
  total and a journal entry (the last 200) are kept.
* `clear(text, amount)`: every transition of it that exists gains `amount`
  clear. `forget(reason)`: the reason's blame taken out of every
  transition, a transition left with none dropped.
* `judge(text, threshold, min_coverage, spans = 5)`: per transition the
  evidence `max(0, blame - clear)`; `risk = Σ evidence / max(1, n)`,
  `coverage = blamed / n`, `peak = max evidence`; reasons: each blamed
  transition's reasons weighted by `evidence / blame`, summed, sorted by
  blame then name, with shares; spans: the blamed transitions by evidence
  (ties: the earlier), each the token window `[i - (order - 1), i + 1)` and
  its heaviest reason; `reject` if any is blamed and `coverage >=
  min_coverage` and `risk >= threshold`, else `suspect` if any is, else
  `pass`. Every sum runs in one fixed order, so both ports agree.

Around the models: `routes::guarded_samples` draws `count · over_sample`
candidates (the first at the asked temperature, the rest at `max(t, 0.7)`),
judges each completion, keeps the first `count` not rejected, and when all
are keeps the least risky; generate, converse and chat take `guard: true`;
chat's `teach_guard` blames a reply marked under the pass mark ("marked
m/10", severity `(pass - m) / 5`) and clears one at or above it; feedback's
`guard` blames the bad texts and clears the good; the network's talk judges
its reply. The lock order is always model, then guard.

### 21.1 Decisions

* **Transitions, not texts:** a failure is remembered as its parts, so a
  new text that reuses enough of them is caught, and one that shares a
  common word or two is not (coverage) - unless that part keeps failing
  (risk).
* **Clearing cancels blame** so the guard does not learn to fear the
  language itself: what good text says often stops counting against it.
* **A filter, not a censor:** with nothing blamed every text passes, and
  when every candidate is vetoed the least risky is still served, marked.
