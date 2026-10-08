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

## 1. The matrix

`Lattice(states, alphabet)` holds `S · A · S` edges, one for every
`(source, symbol, target)`, created when the matrix is: dense, never added
to or removed from. Storage is one flat vector in row-major order,
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
* **The Python port is the reference, the Rust crate is the build.** Python
  for the experiments' authorship and the tests that read like the spec;
  Rust for the CLI and the server, with no dependencies, as the repository's
  other crates. Both read one file format; sampling is not held to parity
  because the two generators are different by design.
