# LatticeFSM

**A finite state machine over a dense 3D matrix of adaptive edges.** The
matrix is `states × symbols × states`, and every cell of it is an edge: *in
this state, reading this symbol, the machine may move to that state*. An edge
is not a number. It is a dense record of its own life — how often and when it
was traversed, a trace of recent traversals that fades, what it has been
rewarded and punished and when, how **wide** the channel is, and the
coefficients of its **own weighting function**, which adapts with the credit
the edge receives. The machine walks the matrix as a finite state machine:
in a state, reading a symbol, it looks along the row of `S` edges, weighs
each by its own function, draws one, traverses it, and is in that state.

Two things make the walk what it is:

* **Stimulation makes the wide channel easier.** Every weight carries a term
  `stimulation · log(width)`. At zero stimulation width is not consulted and
  a narrow path is as easy as a wide one; at one, an edge twice as wide is
  twice as likely; at four, sixteen times. Use and reward widen a channel,
  punishment narrows it, and disuse lets it relax. A surge of stimulation
  relaxes too, on the machine's own clock.
* **Every edge weighs itself its own way.** The weighting is a function of
  the edge's features (its history, its trace, its verdict, its age, its
  width) with coefficients that belong to that one edge, start from a shared
  prototype, and move with every reward and punishment the edge receives —
  so two edges that lived different lives end with different functions, and
  a whole different function can be dropped in from outside.

Reward and punishment land on the edges of the last run, the last edge in
full and each earlier one discounted. Nothing else is learned: no weights
beyond the edge's own, no gradient through the machine, no sweep. Time is the
clock of traversals; traces, ages, widths and stimulation fade on it; the
verdicts never do.

Two ports of one model, standard library only in each, reading and writing
the same machine files:

| port | where | what |
|---|---|---|
| Rust | `rust/` | crate `latticefsm`: the model, the `latticefsm` CLI, and the HTTP server (`make serve`) |
| Python | `latticefsm/` | the reference: the same model and CLI, and the experiments |
| JavaScript | `frontend/` | the React single-page app the server serves, in the shape of `../ModelKit/frontend` |

`DESIGN.md` is the specification both are written to. `make parity` holds
them to each other: the two deterministic experiments agree number for
number, and a machine file written by either reads in the other.

## The ideas, in one table

| Rule | Implementation |
|---|---|
| The matrix is dense and three-dimensional | `Lattice`: `S · A · S` edges from the moment it is made, row-major, none ever added or removed; `row(s, a)` is the fibre the machine chooses among |
| An edge is a record | `Edge`: `seen`, `first_seen`, `last_seen`, `recent` (a trace), `rewarded`, `punished`, `last_rewarded`, `last_punished`, `width`, `width_stamp`, `weighting`, `features` |
| The machine is a finite state machine | `Machine`: a state, a start, accepting states; `step(symbol)` draws the next state from the row and traverses the edge; `run(string)` from the start; the run is accepted or not |
| Traversal is by an adaptive weighting function per edge | `log_weight = bias + Σ kᵢ·fᵢ + stimulation·log(width)`; the five `f` are the edge's features and the `k` are its own `Weighting`, adapted by `rate · credit · fᵢ` at every credit |
| Stimulation prefers the wide channel | the `stimulation · log(width)` term; `stimulate(x)` raises the level, which relaxes toward `baseline` by a half every `calm` ticks |
| Use and reward widen, punishment narrows, disuse relaxes | `traverse` multiplies width by `1 + use_widening`; a credit by `1 + reward_widening · r` or divides by `1 + punish_narrowing · |r|`; the excess over one halves every `life` ticks; clipped to `[0.05, 20]` |
| Credit is discounted back from the end | `credit(r)`: the last edge of the run receives `r`, each earlier one `discount` times the next |
| The clock | every traversal is a tick; `tick(k)` lets time pass; everything that fades reads through elapsed ticks and writes nothing |
| Measurements are quiet | `run(quiet=True)`, `accepts`, `accuracy`, `transition_table`, `stats` move nothing |
| A different function altogether | `weight_fn(edge, clock, life, stimulation)` replaces every edge's own weighting |

## Quick start

```bash
cd LatticeFSM
make test                      # cargo test + clippy + fmt, then the Python tests with parity
make demo                      # learn a language, be stimulated, let time pass (Rust)
make serve                     # http://127.0.0.1:8000/ - the React frontend: the matrix, runs, credit, training, time
make train LANGUAGE=ends-ab STATES=3 && make run TEXT=aab STIM=3 CREDIT=1
make experiments               # the three tables below, a second
python3 -m latticefsm demo     # the same from the Python reference
rust/target/release/latticefsm --help
```

The demo, abridged:

```
a machine of 4 states over ["a", "b"]: 32 edges, each a record of its own; life 1000 ticks, baseline stimulation 1

learning "even-b" (an even number of b's) by reward and punishment over random strings:
  accuracy before: 0.570
  after   500 episodes: 1.000
  greedy table (accepting: [0]):
    0*: a -> 0   b -> 3
    3 : a -> 3   b -> 0
  clock 5950, 31 of 32 edges touched, widest channel 20.00, narrowest 0.99

stimulation prefers the wide channel.  A fresh fork of three edges, widths 4, 1 and 0.25:
  stimulation 0   : [0.333, 0.333, 0.333]
  stimulation 1   : [0.762, 0.190, 0.048]
  stimulation 4   : [0.996, 0.004, 0.000]

stimulated by +3: level 4.00; a life of silence later:
  level 2.50, the widest channel now 10.50; accuracy still 1.000
  four more lives: level 1.09, widest 1.59; accuracy 1.000 - the verdicts never fade, the widths and traces do
```

## The web server and the frontend

`make serve` (or `latticefsm serve --port 8000`) holds one machine behind a
lock, answers JSON under `/api/`, and serves the React frontend from
`frontend/dist` at every other path. The frontend is the ModelKit one's
shape — Vite, plain JSX, one stylesheet, no UI or chart libraries, settings
remembered in the browser — with five tabs for this model:

| tab | what it does |
|---|---|
| **Matrix** | the matrix one symbol-slice at a time: each row a state, each cell the probability of moving to that column's state, shaded by it, with the channel's width and `seen` beneath; click a cell for every field of the edge and its weighting function |
| **Run** | read a string at a stimulation of your choosing, traversing it or asking quietly; reward or punish the last run |
| **Train** | teach a language for some episodes, with the accuracy curve and the greedy table it ends in |
| **Time** | let ticks pass; raise or set the stimulation |
| **Machine** | a fresh machine of any shape; save and load; teach one edge deliberately |

The status bar polls the machine every two seconds: shape, how much of the
matrix is touched, the clock, the stimulation, the state, credits, and the
widest and narrowest channel.

`frontend/dist` is committed, so a checkout serves it with no build; `make
frontend-install` and `make frontend-build` rebuild it, `make frontend-dev`
runs Vite with hot reload against a running server, and `make frontend-test`
runs the frontend's own tests. `rust/README.md` lists every API route, and
`curl` works as well as the page:

```bash
curl -s localhost:8000/api/stats
curl -s -X POST localhost:8000/api/run -d '{"text":"abab","stimulation":2}'
curl -s -X POST localhost:8000/api/credit -d '{"amount":1}'
curl -s -X POST localhost:8000/api/train -d '{"language":"even-b","episodes":800}'
curl -s "localhost:8000/api/matrix?symbol=b"
```

## What it measured

Three experiments, deterministic given the seed, from both ports;
`results/` holds the Rust records and `results/python/` the Python ones.
The stimulation and adaptation records are identical to the last digit
across the ports (the parity test checks it); the learning rows draw their
strings from each port's own generator, so their numbers are two samples of
the same experiment. Settings unless stated: life 1 000 ticks, baseline
stimulation 1, discount 0.8, temperature 1, strings of up to six symbols.

### A language is learned by credit, and not quietly

For each of four regular languages and three machine sizes, five machines
(one per seed) run 4 000 random strings, are rewarded when they end in a
state of the right kind and punished when they do not, and are scored on 300
held-out strings by their greedy walk. The control runs the same strings
quietly and credits them the same way: a quiet run traverses nothing, so the
credit lands on nothing.

Rust:

| language | states (min) | before | after: mean | min | max | solved | quiet after |
|---|---|---|---|---|---|---|---|
| even-b (an even number of b's) | 3 (2) | 0.57 | **1.00** | 1.00 | 1.00 | 5/5 | 0.56 |
| even-b | 6 (2) | 0.55 | **1.00** | 1.00 | 1.00 | 5/5 | 0.55 |
| contains-aa (two a's in a row somewhere) | 3 (3) | 0.61 | **1.00** | 1.00 | 1.00 | 5/5 | 0.61 |
| contains-aa | 6 (3) | 0.64 | **0.87** | 0.62 | 1.00 | 3/5 | 0.63 |
| ends-ab (ends with ab) | 3 (3) | 0.64 | **0.92** | 0.77 | 1.00 | 3/5 | 0.66 |
| ends-ab | 6 (3) | 0.74 | **0.96** | 0.79 | 1.00 | 4/5 | 0.73 |
| mod3-a (a multiple of three a's) | 3 (3) | 0.61 | **1.00** | 1.00 | 1.00 | 5/5 | 0.63 |
| mod3-a | 6 (3) | 0.67 | **0.79** | 0.63 | 1.00 | 1/5 | 0.68 |

Python, the same experiment on its own strings: every language is solved
5/5 at three states; at six states `contains-aa` 0.93 (3/5), `ends-ab` 0.93
(2/5), `mod3-a` 0.82 (2/5). `results/*/experiments.log` has the full tables.

A machine of the minimal size solves every language on every seed. More
states than needed give the credit more ways to be wrong, and a run of seeds
settles into a table that is right on most strings and wrong on a few; there
is no baseline, no entropy bonus and no annealing to pull it out, because
the model is the edge records and nothing else. The quiet arm never moves
from chance, and its machines have not a single edge touched.

### Stimulation prefers the wide channel

A fork of three edges from one state on one symbol, widths 4, 1 and 0.25,
nothing else written to them. The probability of each by stimulation level:

| stimulation | width 4 | width 1 | width 0.25 |
|---|---|---|---|
| 0 | 0.333 | 0.333 | 0.333 |
| 0.5 | 0.571 | 0.286 | 0.143 |
| 1 | 0.762 | 0.190 | 0.048 |
| 2 | 0.938 | 0.059 | 0.004 |
| 4 | 0.996 | 0.004 | 0.000 |

At one the widths are the odds outright (`4 : 1 : 0.25`). Then a surge of
+3 over the baseline of 1, relaxing on the clock:

| lives since | stimulation | width 4 | width 1 | width 0.25 |
|---|---|---|---|---|
| 0 | 4.00 | 0.996 | 0.004 | 0.000 |
| 1 | 2.50 | 0.883 | 0.089 | 0.028 |
| 2 | 1.75 | 0.611 | 0.229 | 0.160 |
| 4 | 1.19 | 0.387 | 0.315 | 0.298 |

The widths relax too: by four lives the fork is nearly even because the
channels have narrowed and widened back toward rest, not only because the
surge has passed.

### An edge adapts its own function

Two edges from one prototype, each used twenty times: one rewarded on every
use, one punished on every use. A third edge in the row is never touched.

| stage | clock | P(rewarded) | P(punished) | P(untouched) | width + | width − | net + | net − | bias + | bias − |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh | 0 | 0.333 | 0.333 | 0.333 | 1.00 | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| after 1 use each | 2 | 0.602 | 0.126 | 0.272 | 1.21 | 0.84 | +0.50 | −0.50 | +0.10 | −0.10 |
| after 5 uses each | 10 | 1.000 | 0.000 | 0.000 | 2.61 | 0.42 | +0.83 | −0.83 | +0.50 | −0.50 |
| after 20 uses each | 40 | 1.000 | 0.000 | 0.000 | 19.99 | 0.05 | +0.95 | −0.95 | +2.00 | −2.00 |
| 1 life later | 1040 | 1.000 | 0.000 | 0.000 | 10.49 | 0.53 | +0.95 | −0.95 | +2.00 | −2.00 |
| 4 lives later | 4040 | 1.000 | 0.000 | 0.000 | 2.19 | 0.94 | +0.95 | −0.95 | +2.00 | −2.00 |

The two weighting functions at the end, `(bias, seen, recent, net, age, width, rate)`:

```
rewarded: [ 2.000,  4.234,  8.000, 2.640,  1.897, 3.496, 0.100]
punished: [-2.000, -4.234, -8.000, 2.640, -1.897, 3.179, 0.100]
```

One prototype, two lives, two functions. The rewarded edge has learned that
its own use (`seen`, `recent`, `age`) counts for it; the punished edge that
its use counts against it. Both have a positive `net` coefficient because
each was credited in the direction its net already pointed, and both a
positive `width` one because the rewarded edge was wide when rewarded and the
punished one narrow when punished. Four lives later the widths and traces
have relaxed and the verdicts and coefficients have not: the punished edge
stays at zero probability on its record alone.

## The pieces

| file | what it holds |
|---|---|
| `DESIGN.md` | the specification both ports are written to; §10 lists the decisions |
| `rust/src/edge.rs` | `Edge` and `Weighting`: the record, the features, the weight, traversal, credit, width |
| `rust/src/lattice.rs` | `Lattice` and `State`: the dense matrix and its persistence |
| `rust/src/machine.rs` | `Machine`: the walk, the clock, credit, stimulation, the quiet measurements, the file |
| `rust/src/languages.rs`, `experiment.rs` | the four languages; the three experiments and their tables |
| `rust/src/http.rs`, `server.rs` | the HTTP/1.1 server, the JSON routes, the static files of the frontend |
| `frontend/` | the React app: `src/App.jsx`, the five panels in `src/components/`, `src/api.js`, `src/matrix.js`; `dist/` built and committed |
| `rust/src/json.rs`, `gzip.rs`, `rng.rs` | JSON, gzip and the generator, written out: no dependencies |
| `rust/src/bin/latticefsm.rs` | the CLI: demo, train, run, teach, accuracy, table, stats, tick, stimulate, experiment, serve |
| `rust/tests/` | the machine and the server over a real socket |
| `latticefsm/` | the Python reference: `edge.py`, `lattice.py`, `machine.py`, `languages.py`, `experiment.py`, `cli.py` |
| `tests/test_latticefsm.py` | 35 tests, parity with the Rust crate among them; `frontend/test/` the frontend's (`tests/README.md`) |
| `results/` | the measured numbers from both ports, with a README |

## Limits

* Credit is a discounted trace back from a run's end, with no baseline: a
  machine larger than the language needs can settle into a partly wrong
  table and stay there (the learning table above).
* The matrix is dense by design, `S · A · S` records: a thousand states over
  a hundred symbols is a hundred million edges. The model is meant for
  machines of tens of states, where every edge's record is the point.
* The two ports draw from different generators (Python's Mersenne Twister,
  Rust's xoshiro256**), so sampled runs differ between them at the same
  seed; everything else agrees. A file written by one port reseeds the
  other's generator from the seed.
* Stimulation is one level for the whole machine, not per state.

## Where it sits

| directory | the structure | what is on an edge |
|---|---|---|
| `RadixCyclicNN/` | a cyclic graph of grams | a count and a learned weight |
| `RadixDecayNN/` | a tree of contexts | nothing: `seen` is on the node |
| `LatticeFSM/` (this) | a dense `S × A × S` matrix walked as a finite state machine | a record: history, trace, verdict, width, and its own adaptive weighting function |
