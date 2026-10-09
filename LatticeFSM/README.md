# LatticeFSM

**A finite state machine over a dense 3D matrix of adaptive edges.** The
matrix is `states × symbols × states` — **13 × 13 × 13** unless asked
otherwise: 13 states over the 13 symbols `a` to `m`, 2,197 edges — and every
cell of it is an edge: *in
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

**The central node.** The 13 × 13 × 13 cube has a centre, the cell
`(6, g, 6)`, and the whole matrix folds into a code that cell holds — laid
out from the centre outward, shell by shell, so it is rebuilt from the
middle outward too. By default the code is **exact**: every field of every
edge comes back bit for bit, at 7.5 to 11 times smaller than the dense
matrix on a taught machine. `float32` and `float16` make it 13 to 29 times
smaller with the behaviour still preserved. The machine can fold itself into
its central node every N transitions, and can walk from the middle state as
well as from the start state.

**Where a run starts and how it moves.** A **focus** in `[0, 1]` picks a
node of the central vertical vector `(0..12, g, 6)` — low or none the top
node, high the bottom, 0.5 the central node — and runs start from it; it can
be a setting, or **learned** from each input by a small function trained by
the same rewards and punishments as the edges. With **skips**, a run reads
two symbols in one move, past the node between, when that two-edge path is
more efficient than the step it would take. And the nodes can **swap** and
**rearrange themselves**, busiest toward the centre, which moves a taught
machine's information into the inner shells of its compressed code.

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
| The matrix is dense and three-dimensional | `Lattice`: `S · A · S` edges from the moment it is made, row-major, none ever added or removed; `row(s, a)` is the fibre the machine chooses among. The default is 13 × 13 × 13 (`DEFAULT_STATES`, `DEFAULT_ALPHABET`); `--states` and `--alphabet` give any other shape |
| An edge is a record | `Edge`: `seen`, `first_seen`, `last_seen`, `recent` (a trace), `rewarded`, `punished`, `last_rewarded`, `last_punished`, `width`, `width_stamp`, `weighting`, `features` |
| The machine is a finite state machine | `Machine`: a state, a start, accepting states; `step(symbol)` draws the next state from the row and traverses the edge; `run(string)` from the start; the run is accepted or not |
| Traversal is by an adaptive weighting function per edge | `log_weight = bias + Σ kᵢ·fᵢ + stimulation·log(width)`; the five `f` are the edge's features and the `k` are its own `Weighting`, adapted by `rate · credit · fᵢ` at every credit |
| Stimulation prefers the wide channel | the `stimulation · log(width)` term; `stimulate(x)` raises the level, which relaxes toward `baseline` by a half every `calm` ticks |
| Use and reward widen, punishment narrows, disuse relaxes | `traverse` multiplies width by `1 + use_widening`; a credit by `1 + reward_widening · r` or divides by `1 + punish_narrowing · |r|`; the excess over one halves every `life` ticks; clipped to `[0.05, 20]` |
| Credit is discounted back from the end | `credit(r)`: the last edge of the run receives `r`, each earlier one `discount` times the next |
| The clock | every traversal is a tick; `tick(k)` lets time pass; everything that fades reads through elapsed ticks and writes nothing |
| Measurements are quiet | `run(quiet=True)`, `accepts`, `accuracy`, `transition_table`, `stats` move nothing |
| The matrix folds into its central node | `compress(machine)` → `Core`, held at `(6, g, 6)`: a bitmap of touched edges and their fields, centre-out, exact by default (`compress.py`, `compress.rs`; `DESIGN.md` §11) |
| Traversal from the middle outward | the code is laid out and rebuilt shell by shell from the centre (`geometry.center_out`, `decompress(shells=k)`); `run(…, from_middle=True)` starts from the middle state |
| Compression every N transitions | `compress_every`, `compress_precision`, `compress_rebuild`: the code kept current, or the matrix rebuilt from it so a lossy precision is applied |
| Focus picks the start node | `focus` in [0, 1] → node `floor(focus · 13)` of the central vertical vector `(s, g, 6)`, top to bottom; none: the start state (`geometry.focus_index`; `DESIGN.md` §12.1) |
| Focus learned from the input | `learn_focus`: `focus = sigmoid(w · x(text))`, a Gaussian policy credited by the run's reward or punishment against a baseline (`focus.py`, `focus.rs`) |
| Skip a node when it is more efficient | `skip`, `skip_margin`: one move reading two symbols when the best two-edge path beats the step's path in log-probability; both edges traversed and credited, the node between not visited |
| Nodes swap and rearrange themselves | `swap_states`, `swap_symbols` (exact relabellings); `rearrange()` and `rearrange_every`: neighbours swap toward the centre when the outer one is busier |
| A different function altogether | `weight_fn(edge, clock, life, stimulation)` replaces every edge's own weighting |

## Quick start

```bash
cd LatticeFSM
make test                      # cargo test + clippy + fmt, then the Python tests with parity
make demo                      # learn a language, be stimulated, let time pass (Rust)
make serve                     # http://127.0.0.1:8000/ - the React frontend: the matrix, runs, credit, training, time
make train LANGUAGE=ends-ab STATES=3 && make run TEXT=aab STIM=3 CREDIT=1
make experiments               # the four sections below, a second
make train && make compress    # fold the taught machine into its central node, exact; CORE=core.json.gz
make expand SHELLS=6           # rebuild it from the middle outward, six shells of seven
make core-run TEXT=abba FROM_MIDDLE=1   # walk straight from the code, from the middle state
make demo EVERY=500 PRECISION=float16   # fold into the central node every 500 transitions
make train SKIP=1 && make focus FOCUS=0.8   # skips in training; runs start from node (10, g, 6)
make rearrange && make compress         # busy nodes to the centre, then fold into the central node
make demo LEARN_FOCUS=1                 # the focus read off each input, learned
python3 -m latticefsm demo     # the same from the Python reference
rust/target/release/latticefsm --help
```

The demo, abridged:

```
a machine of 13 states over ["a", "b", ... "m"]: 2197 edges, each a record of its own; life 1000 ticks, baseline stimulation 1

learning "even-b" (an even number of b's) by reward and punishment over random strings:
  accuracy before: 0.567
  after  2000 episodes: 0.507
  after  3000 episodes: 1.000
  greedy table (accepting: [0]):
     0*: a -> 0   b -> 5
     5 : a -> 5   b -> 0
     ...
    (never traversed on c d e f g h i j k l m: every next state still ties)
  clock 11900, 335 of 2197 edges touched, widest channel 20.00, narrowest 0.98

stimulation prefers the wide channel.  A fresh fork of three edges, widths 4, 1 and 0.25:
  stimulation 0   : [0.333, 0.333, 0.333]
  stimulation 1   : [0.762, 0.190, 0.048]
  stimulation 4   : [0.996, 0.004, 0.000]

stimulated by +3: level 4.00; a life of silence later:
  level 2.50, the widest channel now 10.50; accuracy still 1.000
  four more lives: level 1.09, widest 1.59; accuracy 1.000 - the verdicts never fade, the widths and traces do

folded into its central node (6, g, 6): 48515 bytes exact, lossless - 7.6x smaller than the dense matrix
  float16: 18365 bytes, max KL 3.8e-8, 0 greedy choices changed
  'abba' walked straight from the code, from the middle state outward: 6 -> 0 -> 5 -> 0 -> 0 (accepted)
```

The languages are over `a` and `b`, so a default machine traverses 2 of its
13 symbol-slices while it learns one; the other eleven wait at the
prototype, uniform, for whatever is run over them. The machine settles on a
two-state cycle (0 and 5 above) out of thirteen states, and leaves the
others as routes it once tried.


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
| **Walk** | the focus: a slider over the central vertical vector, none, or learned from the input (and what the learner reads off a string); skips and their margin; the nodes rearranging themselves (one pass, until settled, every N transitions) or two of them swapped |
| **Compress** | fold the matrix into its central node at a precision or a budget; the code's size, its loss in state and behaviour, and its rebuild shell by shell from the centre outward; rebuild the machine to any shell; walk straight from the code, from the start or the middle; save and load codes |
| **Machine** | a fresh machine of any shape; compress every N transitions (precision, rebuild); save and load; teach one edge deliberately |

The Run tab can start the walk from the middle state and shows a skip as one
move past the node it skips; the Matrix tab rings the central node on its
slice, shades the central vertical vector, marks the focus node, and shows
each state's original number once the nodes have rearranged.

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

Five experiments, deterministic given the seed, from both ports;
`results/` holds the Rust records and `results/python/` the Python ones.
The stimulation and adaptation records are identical to the last digit
across the ports (the parity test checks it); the learning rows draw their
strings from each port's own generator, so their numbers are two samples of
the same experiment. Settings unless stated: life 1 000 ticks, baseline
stimulation 1, discount 0.8, temperature 1, strings of up to six symbols.

### A language is learned by credit, and not quietly

For each of four regular languages and four machine sizes (13, the default, among them), five machines
(one per seed) run 4 000 random strings, are rewarded when they end in a
state of the right kind and punished when they do not, and are scored on 300
held-out strings by their greedy walk. The control runs the same strings
quietly and credits them the same way: a quiet run traverses nothing, so the
credit lands on nothing.

Rust:

| language | states (min) | before | after: mean | min | max | solved | quiet after |
|---|---|---|---|---|---|---|---|
| even-b (an even number of b's) | 3 (2) | 0.57 | **1.00** | 1.00 | 1.00 | 5/5 | 0.56 |
| even-b (an even number of b's) | 6 (2) | 0.55 | **1.00** | 1.00 | 1.00 | 5/5 | 0.55 |
| even-b (an even number of b's) | 13 (2) | 0.55 | **0.93** | 0.64 | 1.00 | 4/5 | 0.55 |
| contains-aa (two a's in a row somewhere) | 3 (3) | 0.61 | **1.00** | 1.00 | 1.00 | 5/5 | 0.61 |
| contains-aa (two a's in a row somewhere) | 6 (3) | 0.64 | **0.87** | 0.62 | 1.00 | 3/5 | 0.63 |
| contains-aa (two a's in a row somewhere) | 13 (3) | 0.65 | **0.81** | 0.64 | 1.00 | 1/5 | 0.65 |
| ends-ab (ends with ab) | 3 (3) | 0.64 | **0.92** | 0.77 | 1.00 | 3/5 | 0.66 |
| ends-ab (ends with ab) | 6 (3) | 0.74 | **0.96** | 0.79 | 1.00 | 4/5 | 0.73 |
| ends-ab (ends with ab) | 13 (3) | 0.79 | **0.92** | 0.81 | 1.00 | 1/5 | 0.78 |
| mod3-a (a multiple of three a's) | 3 (3) | 0.61 | **1.00** | 1.00 | 1.00 | 5/5 | 0.63 |
| mod3-a (a multiple of three a's) | 6 (3) | 0.67 | **0.79** | 0.63 | 1.00 | 1/5 | 0.68 |
| mod3-a (a multiple of three a's) | 13 (3) | 0.70 | **0.76** | 0.69 | 0.85 | 0/5 | 0.70 |

Python, the same experiment on its own strings: every language is solved
5/5 at three states; at thirteen states `even-b` 0.92 (4/5), `contains-aa`
0.80 (1/5), `ends-ab` 0.95 (0/5), `mod3-a` 0.80 (0/5).
`results/*/experiments.log` has the full tables, the four-state rows
included.

A machine of the minimal size solves every language on every seed. More
states than needed give the credit more ways to be wrong, and a run of seeds
settles into a table that is right on most strings and wrong on a few; there
is no baseline, no entropy bonus and no annealing to pull it out, because
the model is the edge records and nothing else. **The default 13-state
machine is four to six times the size these languages need**, and it shows:
`even-b` is solved on four seeds of five, `ends-ab` reaches 0.92 to 0.95 on
average, and `mod3-a`, which needs a three-state cycle found among thirteen,
is never solved outright. Pass `--states 3` (or `STATES=3`) to teach these
languages more reliably; the default is sized for the matrix, not for them. The
quiet arm never moves from chance, and its machines have not a single edge
touched.

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

### Folded into the central node

Default 13 × 13 × 13 machines, fresh and taught each language for 4 000
episodes, folded into the central node `(6, g, 6)` at each precision. The
dense matrix is 369 096 bytes: 2 197 edges of six 8-byte integers and
fifteen 8-byte numbers. Rust; the Python port writes the same codes byte
for byte.

| machine | precision | touched | code, bytes | smaller | lossless | max relative error | max KL | greedy changed | accuracy kept |
|---|---|---|---|---|---|---|---|---|---|
| fresh | any | 0 | 275 | 1 342× | yes | 0 | 0 | 0 | – |
| even-b | exact | 338 | 48 947 | 7.5× | **yes** | 0 | 0 | 0 | yes |
| even-b | float32 | 338 | 28 667 | 12.9× | no | 5.8 × 10⁻⁸ | 4.3 × 10⁻¹² | 0 | yes |
| even-b | float16 | 338 | 18 527 | 19.9× | no | 4.8 × 10⁻⁴ | 1.8 × 10⁻⁴ | 0 | yes |
| ends-ab | exact | 230 | 33 395 | 11.1× | **yes** | 0 | 0 | 0 | yes |
| ends-ab | float16 | 230 | 12 695 | 29.1× | no | 4.8 × 10⁻⁴ | 2.6 × 10⁻¹³ | 0 | yes |
| mod3-a | exact | 316 | 45 779 | 8.1× | **yes** | 0 | 0 | 0 | yes |
| mod3-a | float16 | 316 | 17 339 | 21.3× | no | 3.3 × 10⁻³ | 1.5 × 10⁻⁵ | 0 | yes |

Exact is the default: **no loss at all**, every field of every edge back bit
for bit. Even float16 changed no greedy choice and no language's accuracy on
any machine. A fresh matrix is its 275-byte bitmap. Low-rank codes were
measured against this one while designing it and lost: the smallest Tucker
decomposition that kept the behaviour was larger than storing the touched
edges exactly (`DESIGN.md` §11.3).

**From the middle outward.** The code is laid out shell by shell from the
central node, so a rebuild can stop at any shell. For `even-b`:

| shells rebuilt | cells | touched edges restored | greedy changed | accuracy |
|---|---|---|---|---|
| 1 (the central node) | 1 | 0 | 25 | 0.550 |
| 5 | 729 | 0 | 25 | 0.550 |
| 6 | 1 331 | 121 | 23 | 0.557 |
| 7 (all) | 2 197 | 338 | 0 | 0.580 |

The languages are over `a` and `b`, the outermost two layers of the symbol
axis, so a taught machine's edges all sit in shells 5 and 6 and come back
only at the end of an outward rebuild. A language over the middle symbols
would come back first.

**Every N transitions.** Each language taught again with the matrix folded
into its central node every 500 transitions and rebuilt from the code each
time (23 compressions per run):

| language | never | exact | float32 | float16 |
|---|---|---|---|---|
| even-b | 0.580 | 0.580 | 0.580 | 0.820 |
| contains-aa | 0.660 | 0.660 | 0.660 | 0.647 |
| ends-ab | 1.000 | 1.000 | 1.000 | 1.000 |
| mod3-a | 0.870 | 0.870 | 0.870 | 0.727 |

An exact rebuild on a clock changes nothing, edge for edge; float32 changed
nothing measurable. Float16 rounds the machine's numbers to three digits 23
times and so sends training down another path: better on one language,
worse on two, on one seed. That is a perturbation, not a method; the
Python port's run moves the same way on two of the three.

### Where a run starts

Every language taught on default machines for 4 000 episodes on three seeds,
six ways: plain, with skips, with a learned focus, rearranging every 500
transitions, and from a fixed focus of 0.5 (the central node) and of 1 (the
bottom node). Mean accuracy on 300 held-out strings; the two ports draw
from different generators, so each column pair is two samples of the same
experiment, and their disagreement is the noise in three seeds.

| language | plain (Rust · Python) | skip | learned focus | rearrange every 500 |
|---|---|---|---|---|
| even-b | 0.597 · 0.556 | **1.000 · 1.000** (3/3 · 3/3) | 0.619 · 0.546 | 0.729 · 1.000 |
| contains-aa | 0.628 · 0.888 | 0.681 · 0.948 | 0.577 · 0.633 | 0.680 · 0.838 |
| ends-ab | 0.968 · 0.959 | 0.947 · 0.968 | 0.926 · 0.920 | 0.913 · 0.929 |
| mod3-a | 0.774 · 0.724 | 0.716 · 0.868 | 0.751 · 0.813 | 0.717 · 0.708 |

* **Skips** solve `even-b` on every seed in both ports, where plain runs
  solve it on none: the lookahead finds the two-state parity cycle that a
  step-by-step walk among thirteen states misses. Elsewhere the effect
  goes either way between the ports. Training skips on 2 to 15 % of
  transitions.
* **The learned focus** shows no consistent gain: better on one language
  in each port, a different one, and worse on `contains-aa` in both. The
  languages' accepting states are fixed by number, so the learner mostly
  learns to stay at the top node — which is where no focus starts anyway.
* **Rearranging on a clock** relabels exactly, so it cannot change what a
  machine knows; it changes the order a sampled step draws its targets in,
  and so the training path. Its `even-b` row (3/3 in Python, 1/3 in Rust)
  is that, not a method.
* **A fixed focus away from the top** is worse on most languages in both
  ports (`walk_results.json`): the accepting states are numbered for a
  start at the top, and the empty string is judged where the run starts.

**Rearranging and the central node.** `even-b` taught plain, then left to
rearrange itself until settled — `a` and `b` move to the middle of the
symbol axis, the busiest states to the middle of the state axis, the start
state to the centre:

| shell | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| touched edges before | 0 | 0 | 0 | 0 | 0 | 121 | 217 |
| after | 1 | 17 | 32 | 48 | 64 | 80 | 96 |

Before, a rebuild from the middle outward gave back nothing until the last
two shells; after, the central node itself holds a learned edge and every
shell carries some. The machine walks exactly as before from its start
state — a swap is a relabelling — and the code is still exact.

## The pieces

| file | what it holds |
|---|---|
| `DESIGN.md` | the specification both ports are written to; §10 lists the decisions |
| `rust/src/edge.rs` | `Edge` and `Weighting`: the record, the features, the weight, traversal, credit, width |
| `rust/src/lattice.rs` | `Lattice` and `State`: the dense matrix and its persistence |
| `rust/src/machine.rs` | `Machine`: the walk, the clock, credit, stimulation, the quiet measurements, the file |
| `rust/src/languages.rs`, `experiment.rs` | the four languages; the four experiments and their tables |
| `rust/src/geometry.rs`, `compress.rs` | the cube's centre and shells, the centre-out order, the central vertical vector and focus; the code the central node holds, its precisions, the outward rebuild, the walk from the code, `fidelity` |
| `rust/src/focus.rs` | the learned focus |
| `rust/src/http.rs`, `server.rs` | the HTTP/1.1 server, the JSON routes, the static files of the frontend |
| `frontend/` | the React app: `src/App.jsx`, the five panels in `src/components/`, `src/api.js`, `src/matrix.js`; `dist/` built and committed |
| `rust/src/json.rs`, `gzip.rs`, `rng.rs` | JSON, gzip and the generator, written out: no dependencies |
| `rust/src/bin/latticefsm.rs` | the CLI: demo, train, run, teach, accuracy, table, stats, tick, stimulate, experiment, serve |
| `rust/tests/` | the machine and the server over a real socket |
| `latticefsm/` | the Python reference: `edge.py`, `lattice.py`, `machine.py`, `geometry.py`, `focus.py`, `compress.py`, `languages.py`, `experiment.py`, `cli.py` |
| `tests/test_latticefsm.py` | 67 tests, parity with the Rust crate among them; `frontend/test/` the frontend's (`tests/README.md`) |
| `results/` | the measured numbers from both ports, with a README |

## Limits

* Credit is a discounted trace back from a run's end, with no baseline: a
  machine larger than the language needs can settle into a partly wrong
  table and stay there (the learning table above).
* The default 13 × 13 × 13 machine is larger than any of the four languages
  needs, so it learns them less reliably than a machine of three or four
  states (the learning table); training defaults to 4 000 episodes for it.
* The matrix is dense by design, `S · A · S` records: a thousand states over
  a hundred symbols is a hundred million edges. The model is meant for
  machines of tens of states, where every edge's record is the point.
* The two ports draw from different generators (Python's Mersenne Twister,
  Rust's xoshiro256**), so sampled runs differ between them at the same
  seed; everything else agrees. A file written by one port reseeds the
  other's generator from the seed.
* Stimulation is one level for the whole machine, not per state.
* A partial rebuild from the middle outward gives back nothing a taught
  machine learned until the last two shells, because the languages use the
  outermost symbols (`DESIGN.md` §11.4).
* The learned focus is off by default and measured to help no language
  consistently; skips are off by default and help `even-b` reliably, the
  rest by chance (three seeds per port).
* A lossy precision with `compress_rebuild` on changes what the machine
  learns afterwards; the code's own loss is measured, the downstream effect
  only on one seed.

## Where it sits

| directory | the structure | what is on an edge |
|---|---|---|
| `RadixCyclicNN/` | a cyclic graph of grams | a count and a learned weight |
| `RadixDecayNN/` | a tree of contexts | nothing: `seen` is on the node |
| `LatticeFSM/` (this) | a dense `S × A × S` matrix walked as a finite state machine | a record: history, trace, verdict, width, and its own adaptive weighting function |
