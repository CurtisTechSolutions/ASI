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
| both | `ngram.py`, `ngram.rs` | predicting the next thing: N-grams over adaptive edges, trained from `data/corpus` and through Ollama; base64 for any file |
| both | `network.py`, `network.rs` | networks of machines, Keras-style: layers in a row and side by side, English in and out through `../PhoneticTokenizer` |
| both | `media.py`, `media.rs` | text, audio, images, frame differences and video as symbols, so a network trains on them |
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
make ngram-train && make ngram-predict TEXT="the brain is"   # predict the next thing
make autotrain                 # the corpus, then rounds with a local LLM through Ollama
make demo                      # learn a language, be stimulated, let time pass (Rust)
make serve                     # backend :8000 + frontend :5173 with hot reload - open http://127.0.0.1:5173/ (Ctrl-C stops both)
make serve-prod                # one process, no npm: the API and the committed frontend on http://127.0.0.1:8000/
make train LANGUAGE=ends-ab STATES=3 && make run TEXT=aab STIM=3 CREDIT=1
make experiments               # the four sections below, a second
make train && make compress    # fold the taught machine into its central node, exact; CORE=core.json.gz
make expand SHELLS=6           # rebuild it from the middle outward, six shells of seven
make core-run TEXT=abba FROM_MIDDLE=1   # walk straight from the code, from the middle state
make demo EVERY=500 PRECISION=float16   # fold into the central node every 500 transitions
make train SKIP=1 && make focus FOCUS=0.8   # skips in training; runs start from node (10, g, 6)
make rearrange && make compress         # busy nodes to the centre, then fold into the central node
make demo LEARN_FOCUS=1                 # the focus read off each input, learned
make network-fit ARCH="(3|3|3)" CREDIT_BY=own LANGUAGE=mod3-a   # three machines voting
make translate-fit && make translate TEXT="cat dog"             # English to English: cat -> cats
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

`make serve-prod` (or `latticefsm serve --port 8000`) holds one machine behind a
lock, answers JSON under `/api/`, and serves the React frontend from
`frontend/dist` at every other path. The frontend is the ModelKit one's
shape — Vite, plain JSX, one stylesheet, no UI or chart libraries, settings
remembered in the browser — with these tabs:

A selector in the header picks the language model the language tabs use -
the **chain** (below) or the **N-gram** - and the status bar shows both.

| tab | what it does |
|---|---|
| **Train** | a new chain (nodes, unit, confidence) or N-gram (N, unit); train on server paths and on **uploads** - pick many files or a whole folder, any kind (zip, tar, gz, HTML, docx, images...), sent in 8 MiB chunks so a file may be any size - or on pasted text; every token predicted before it is learned; **until N of N**; the held-out measure charted; for the chain, the confidence sweep (accuracy and depth against the threshold) and "use this confidence" |
| **Predict** | type, and the likeliest next tokens follow as bars - click one to take it; for the chain the input nodes holding the last inputs, the walk back through them (each node's confidence, the confidence gathered) and where it stopped |
| **Generate** | several samples at a temperature; 👍 / 👎 each, marks out of 10 applied as feedback, 🔊 heard |
| **Talk** | a chat whose replies are the model's **predictions**: what you say, continued; each token tinted by how deep the chain walked for it; 👍 / 👎; **teach this reply** - the reply it should have given, taught round after round until N of N |
| **Network** | the network of machines (a translator, `enc>dec`, or a classifier): reinforce or correct a reply - taught until N of N - and fit a task until every pair is right |
| **Converse** | the model talking to itself, turn after turn |
| **Chat** | a local LLM (Ollama) talking with the model and marking every reply out of 10, the marks credited |
| **Score** | how likely the model finds a text, token by token |
| **Nodes** | the chain's most-seen node contexts, their guesses and confidence, and each depth's accuracy |
| **Feedback** | good and bad texts, rewarded and punished in one go (and, ticked, taught to the guard) |
| **Guard** | the negative network: judge any text (verdict, risk, coverage, reasons, blamed spans highlighted), blame with a reason, clear, settings, the reasons (forget one, or all) and the journal; a Guard toggle on Generate, Talk, Converse, Chat, Feedback and Network |
| **Ollama** | is it there; a corpus from a prompt, trained on or saved as an upload; autotrain |
| **Voice** | the browser listens, the model answers, the browser speaks (Web Speech, no server audio) |
| **Checkpoints** | save any model (chain, N-gram, machine, network) as a checkpoint; list, restore, download, upload, delete; save and load each to a server path |
| **Settings** | the settings this browser remembers, and forgetting them |
| **Matrix** | the matrix one symbol-slice at a time: each row a state, each cell the probability of moving to that column's state, shaded by it, with the channel's width and `seen` beneath; click a cell for every field of the edge and its weighting function |
| **3D** | the whole matrix as a cube of cells - source top to bottom, symbol into the page, target across - drawn on a canvas by hand: drag to turn, scroll to zoom, hover to read, click for the edge's record; colour by probability, seen, width or net; only the inner shells, the touched edges or one symbol; the central node, the central vertical vector and the focus node outlined |
| **Run** | read a string at a stimulation of your choosing, traversing it or asking quietly; reward or punish the last run |
| **Walk** | the focus: a slider over the central vertical vector, none, or learned from the input (and what the learner reads off a string); skips and their margin; the nodes rearranging themselves (one pass, until settled, every N transitions) or two of them swapped |
| **Teach** | teach the machine a language for some episodes, with the accuracy curve and the greedy table it ends in |
| **Time** | let ticks pass; raise or set the stimulation |
| **Compress** | fold the matrix into its central node at a precision or a budget; the code's size, its loss in state and behaviour, and its rebuild shell by shell from the centre outward; rebuild the machine to any shell; walk straight from the code, from the start or the middle; save and load codes |
| **Machine** | a fresh machine of any shape; compress every N transitions (precision, rebuild); save and load; teach one edge deliberately |

Uploads go to `data/uploads` (or `$LATTICEFSM_UPLOADS`), checkpoints to
`checkpoints/` (or `$LATTICEFSM_CHECKPOINTS`), named
`{model}--{tag}--{unix seconds}.json.gz`.

The Run tab can start the walk from the middle state and shows a skip as one
move past the node it skips; the Matrix tab rings the central node on its
slice, shades the central vertical vector, marks the focus node, and shows
each state's original number once the nodes have rearranged.

The status bar polls the machine every two seconds: shape, how much of the
matrix is touched, the clock, the stimulation, the state, credits, and the
widest and narrowest channel.

`frontend/dist` is committed, so `make serve-prod` serves it with no build
and no npm. `make serve` runs the backend (`make serve-backend`, on
`HOST:PORT`, default 127.0.0.1:8000, loading `MODEL` when it exists) and the
Vite dev server (`make serve-frontend`, on `FRONTEND_HOST:FRONTEND_PORT`,
default 127.0.0.1:5173, proxying `/api` to the backend) together: open the
frontend's address, edit `frontend/src` and the page reloads; Ctrl-C stops
both, and if either stops the other is stopped too. `make install` builds
the binary and installs the frontend's dependencies; `make frontend-build`
rebuilds `dist/`, `make frontend-preview` serves that build through Vite,
and `make frontend-test` runs the frontend's own tests. `make help` lists
every target. `rust/README.md` lists every API route, and
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

## Predicting the next thing

The main use: an **N-gram model** that predicts the next token
(`latticefsm/ngram.py`, `rust/src/ngram.rs`). A finite state machine whose
states are the last `N - 1` tokens *is* an N-gram model; this one keeps that
machine sparse - an edge exists once `(context, token)` has been seen - and
gives every edge the parts of a lattice edge that matter here: its count,
when it was last taken, what it was rewarded and punished, and a width.

* The next-token distribution is **interpolated Kneser-Ney**: the highest
  order from raw counts, every lower order from continuation counts (how
  many different contexts a token follows), Ney's discount per order,
  interpolated down to a uniform floor - every token has some probability,
  and a token seen in many places outranks one seen often in one.
* **Credit** widens or narrows edges as it does a lattice edge, and the
  **stimulation** weighs each credited edge by `width ^ stimulation`,
  renormalised; at stimulation 0, or before any credit, it is plain
  Kneser-Ney. A judge (you, in the Predict tab, or an LLM) supplies it.
* Tokens are `word`s, `char`s, `sound`s (phonemes, spelled back as
  English), or **`base64` letters**: any file - text, image, audio, video -
  as base64 (`b64.py`, `b64.rs`), 76 letters a sequence, learned letter by
  letter and generated back to bytes. A prompt is encoded only as far as its
  bits are complete, so it is a true prefix of what continues it.
* **It predicts every token while it learns it.** Training walks each
  sequence once: the model first guesses the next token from what it has
  learned so far, then counts it - so every training run also says how well
  the model predicted text it had not yet seen ("predicted 12030 tokens
  before learning each, first guess right 0.172, perplexity 170.17" for
  word trigrams on `data/corpus`). The Kneser-Ney tables are kept up to date
  token by token rather than rebuilt, and the first guess is found without
  computing the whole distribution, so this costs little (Python: 1.6 s for
  12k words). Predicting changes nothing that is learned. `--self-credit R`
  also rewards a right guess and punishes a wrong one by `R`; it is off by
  default because it measured worse (held-out perplexity, word trigrams:
  180.07 off, 178.45 at 0.25, 190.5 at 1.0; char 5-grams: 4.485, 4.695,
  5.085). `--no-predict` just counts.
* Counts are integers and every probability is computed in one fixed order:
  both ports give the same probabilities, the same greedy texts and the same
  measurements, and read each other's files (`make parity`).

```bash
make ngram-train                                  # trigrams of words on data/corpus, every 10th line held out;
                                                  # each token predicted before it is learned
make ngram-predict TEXT="the brain is"            # a 0.553, the 0.053, </s> 0.043, not 0.031, ...
make ngram-generate TEXT="the brain"              # the brain is a directed cyclic graph can be.
make ngram-train UNIT=base64 ORDER=8 CORPUS=media/  # any files, as base64
make ngram-generate OUT=out.bin                   # a base64 model's letters decoded to bytes
make base64-encode FILE=photo.png OUT=photo.b64 && make base64-decode FILE=photo.b64 OUT=back.png
```

```python
from latticefsm.ngram import NGram
m = NGram(3, "word")
m.train_paths(["data/corpus"], held_out=10)
m.predict("the brain is", k=5)        # [('a', 0.5527), ('the', 0.0530), ...]
m.complete("the brain")               # 'is a directed cyclic graph can be.'
m.credit_text("the brain is a graph", +1)   # widen the edges this text takes
```

### Training it automatically: a corpus, and Ollama

`autotrain` (`latticefsm/autotrain.py`, `rust/src/autotrain.rs`) is the
whole loop. **Round 0** trains on every file under `data/corpus` (drop more
in; `.txt`, `.md`, `.jsonl`, `.json`, gzipped too - or any file at all for
`base64`), every tenth line held out, and measures the model on what was
held out. **Each round after it**, with a local language model served by
[Ollama](https://ollama.com):

1. the LLM writes sentences about the round's topics - given, or the
   corpus's own most frequent words (`graph`, `cycle`, `function`, ...) -
   and the model trains on them;
2. the model's own contexts (the first words of training lines) are handed
   to the LLM to continue, and the model trains on the continuations - the
   LLM shows what comes next exactly where the model asks;
3. the model writes texts of its own and the LLM judges which are fluent,
   sensible English; approved texts' edges are rewarded, rejected ones'
   punished;

then it is measured on the same held-out lines, checkpointed, and stopped
once the held-out perplexity has not improved for `patience` rounds.

```bash
ollama serve & ollama pull llama3.2               # once
make ollama-check                                 # is it there, which models
make autotrain ROUNDS=5 TOPICS="graphs,the brain" # or OLLAMA_MODEL=qwen3; NGRAM=ngram.json.gz
latticefsm ngram autotrain --ollama --rounds 5 --keep data/corpus/ollama.txt --save ngram.json.gz
make fake-ollama &  OLLAMA_HOST=127.0.0.1:11500 make autotrain   # a stand-in, to try the loop without a model
```

The address is `OLLAMA_HOST` (Ollama's own convention) or
`http://127.0.0.1:11434`; the model `--model`, `LATTICEFSM_OLLAMA_MODEL` or
`llama3.2`. The prompts and seeds are the same in both ports - a test runs
both against the stand-in and compares every prompt and every round. Ollama
was not available where this was built, so the loop is tested against the
stand-in (`tests/fake_ollama.py`, which answers like Ollama and records what
it is asked), not against a real model: what a real one does to the held-out
perplexity is not measured here. With the stand-in's repetitive text it
rises, as it should - text unlike the corpus makes the corpus less likely -
and the loop stops itself.

ngram: interpolated Kneser-Ney on data/corpus, every 10th line (base64: every 10th 76-letter sequence of the files) held out; accuracy over the first 2000 held-out positions; bits a byte to compare the units

| unit | N | vocabulary | n-grams | held-out perplexity | bits a token | **bits a byte** | next-token accuracy | top 5 |
|---|---|---|---|---|---|---|---|---|
| word | 1 | 1811 | 1812 | 360.80 | 8.495 | **1.882** | 0.069 | 0.261 |
| word | 2 | 1811 | 7210 | 195.87 | 7.614 | **1.687** | 0.170 | 0.371 |
| word | 3 | 1811 | 10269 | 180.07 | 7.492 | **1.660** | 0.181 | 0.382 |
| word | 4 | 1811 | 11051 | 178.39 | 7.479 | **1.657** | 0.185 | 0.387 |
| char | 3 | 79 | 4449 | 7.23 | 2.855 | **2.838** | 0.418 | 0.800 |
| char | 5 | 79 | 18577 | 4.48 | 2.165 | **2.152** | 0.588 | 0.860 |
| char | 7 | 79 | 31171 | 4.32 | 2.112 | **2.100** | 0.602 | 0.863 |
| sound | 3 | 43 | 5179 | 7.60 | 2.926 | **2.511** | 0.413 | 0.766 |
| sound | 5 | 43 | 19784 | 5.28 | 2.400 | **2.059** | 0.536 | 0.812 |
| base64 | 4 | 63 | 15989 | 4.91 | 2.295 | **3.101** | 0.520 | 0.887 |
| base64 | 8 | 63 | 50119 | 4.28 | 2.099 | **2.835** | 0.614 | 0.897 |
| base64 | 12 | 63 | 62754 | 4.29 | 2.103 | **2.840** | 0.610 | 0.897 |

What it says: on this corpus more context helps words up to trigrams and
characters up to 7-grams, then flattens - a 56 KB corpus has few 4-grams to
learn from. Word trigrams guess the next word right 18% of the time, the
next character 60% of the time at N = 7. Bits a byte compare the units on
the same text: words compress it best, but words, characters and sounds are
lower-cased (and sounds forget spelling), while base64 keeps every byte -
case, punctuation, line ends - and still comes within a bit a byte of
characters, from an alphabet of 64 letters that each carry 6 bits.

## The chain: one input to a node, walks that shorten with confidence

`latticefsm/chain.py` and `rust/src/chain.rs` (the same model, number for
number). Reading `"abc"`, input node 1 holds `a`, node 2 `b`, node 3 `c`,
and the **output node** after them predicts the next input. A chain of
`--nodes 8` has 7 input nodes; the inputs are letters by default (or words,
sounds, base64 letters).

Every input node, in its context (the inputs it and the nodes after it
hold), keeps what came next, its own guess, and how often that guess was
right when it was tested while training. Its **confidence** is
`(right + rewarded) / (tested + rewarded + punished + 2)`: accuracy and
reward for it, misses and punishment against it, and two imagined misses so
that a node has to earn it.

A prediction walks from the output node back through the input nodes,
nearest first, gathering confidence `1 - (1 - c0)(1 - c1)...`, and **stops as
soon as that reaches the threshold** (`--confidence`, default 0.9). Every
node's confidence shortens the walk, continuously: the more confident the
nodes near the output, the fewer are traversed - and as training and reward
make them more confident, paths keep getting shorter; the less confident, the
further the walk has to go. It predicts the distribution of the node it
stopped at, smoothed through the nodes nearer the output (Witten-Bell). A
reward goes to every node on the walk that guessed what was rewarded (so the
next walk stops there sooner), a punishment the same way (so it goes deeper).

On `data/corpus`, letters, 8 nodes, every tenth line held out:

| threshold | accuracy | perplexity | input nodes traversed |
|---|---|---|---|
| 0 (the output node alone) | 0.161 | 20.53 | 0.00 |
| 0.5 | 0.415 | 7.45 | 2.15 |
| 0.7 | 0.520 | 6.01 | 3.21 |
| 0.9 (default) | 0.584 | 5.61 | 4.23 |
| 0.95 | 0.594 | 5.59 | 4.54 |
| 1 (always as deep as it can) | 0.599 | 5.88 | 5.55 |

At 0.9 a walk traverses 24% fewer nodes than the deepest walk for 1.5 points
of accuracy, and its perplexity is lower. Trained a second time on the same
lines, the nodes grow more confident and the held-out walks shorten (at 0.7:
3.21 to 3.04 nodes, accuracy 0.520 to 0.514). Gathering confidence was as
accurate as stopping at the first node above the threshold, or more, at every
depth measured.

```bash
make chain-train && make chain-predict TEXT="the bra"
latticefsm chain predict "the bra" --load chain.json.gz
#   input nodes: t h e _ b r a
#   walk: node 8 (output) 0.166 > node 7 'a' 0.174 > node 6 'r' 0.179 > node 5 'b' 0.826; 3 of 7 input nodes, ...
#   i                0.843175
latticefsm chain sweep --load chain.json.gz          # the table above
latticefsm chain generate "the brain" --load chain.json.gz
```

## Teaching until N of N

Teaching a reply, and training, go on until every token is right:

* `teach TEXT REPLY` (`POST /api/{chain,ngram}/teach`, the Talk tab's
  "teach this reply") repeats rounds until the model's first guess is right
  on every token of the reply and on its end - its greedy continuation of the
  text is the reply. A round punishes what guessed wrong (the chain: the
  nodes on the walk that guessed it, so the walk goes deeper; the N-gram: the
  wrong edge narrowed) by 1, doubling every round a position stays wrong -
  so even a node right a thousand times elsewhere gives way - and learns
  text and reply once more. "a lattice of adaptive edges that predicts."
  after "what is the brain": 16 of 43 right before, 43 of 43 after 11 rounds
  (the chain); held-out accuracy unchanged (0.5845) after two such teachings.
* `train --until` (`"until": true` on the train and text routes) trains,
  then **drills** the model on what it learned until every token is right
  that can be: a context the data continues in more than one way can only
  ever be guessed one way, so the goal is the most possible (the ceiling,
  printed) - N of N when nothing is ambiguous. Every round, in every context
  whose guess is not the token that follows it most often, the chain's nodes
  that do not guess it are punished by `2^k` and its deepest node, when
  outvoted, is given enough of the token (and `2^k - 1` more); the N-gram's
  wrong edge narrowed and right one widened. On `data/corpus` (letters, 8
  nodes): 28 599 of 52 229 right while first learning, 44 715 of 44 715
  possible after 12 rounds, held-out accuracy 0.584 to 0.596.
* The network's "teach this reply" (`until`) repeats until a whole round is
  right and its answer is the reply; its fit (`until`) until every taught
  pair is exactly right.
* **A threshold** makes the goal a share instead of all of it: `--threshold`
  (`"threshold"` on the routes, "Goal %" in the frontend), **0.95 by
  default**, so teaching stops at 95% of a reply's tokens and training at
  95% of the most possible; 1 is exact. The same reply at 0.95: 41 of 43
  right after 8 rounds instead of 43 of 43 after 11. Right means right with
  the true text before it: a reply taught short of all of it may still be
  continued differently once one early token goes another way.

```bash
latticefsm chain teach "what is the brain" "a lattice of adaptive edges."
latticefsm chain train --data data/corpus/sentences.txt --until
#   round 1: 700 of 1673 predicted right before learning (the most possible: 1550)
#   ...
#   1550 of 1673 reached in 8 rounds: the most possible - the rest (123) come after contexts the data continues in more than one way
```

## The guard

`latticefsm/guard.py` and `rust/src/guard.rs`: a negative network - only the
failures, and why. Every transition of a blamed text (the last two words,
then the next) is charged the blame under its reason; accepted text clears
what it crosses, and the evidence against a transition is
`max(0, blame - clear)`, so a fragment good text crosses as often as bad
text carries no verdict. Judging a text gives its **risk** (evidence per
transition: about 1 for a repeat of a failure blamed once), **coverage**
(the share of its transitions that are known failures), peak, reasons and
worst spans, and a verdict: **reject** (coverage at least 0.5 and risk at
least 1, by default), **suspect** or **pass** - a text it never saw fail
passes.

The server holds one guard in front of everything that writes text. With
`guard: true`, generate draws three times as many candidates (the first at
the asked temperature, the rest at 0.7 or more, so a vetoed greedy reply has
others to give way to) and vetoes the rejected ones; converse and chat guard
every turn; the network's replies are judged and marked vetoed. What feeds
it: 👎 in the frontend, bad feedback, a local LLM marking a reply under the
pass mark ("marked 3/10"), and explicit blames; 👍, good feedback and pass
marks clear. Its tab judges any text (blamed spans highlighted), blames,
clears, forgets a reason, and lists the reasons and the journal; it saves
and restores as a checkpoint.

```bash
latticefsm guard blame "the cat ate the dog" --reason violent --severity 2 --save guard.json
latticefsm guard judge "the cat ate the fish" --load guard.json
#   reject: 'the cat ate the fish': risk 1.333, coverage 0.667, peak 2.000
#     rejected: 67% of its transitions are known failures, 1.33 blame each on average (the limit 1.00), mostly 'violent'
curl -s -X POST localhost:8000/api/chain/generate -d '{"text":"the brain","guard":true}'
```

## A corpus of any kind and size

`latticefsm/corpus.py` and `rust/src/corpus.rs` (with `archive.rs`, a
streaming inflate, gzip, zip and tar reader written out on the standard
library) read the same files into the same lines and bytes:

| kind | files |
|---|---|
| text | `.txt .md .text .csv .tsv .log .rst .tex .org .srt .vtt`, a line at a time |
| JSON | `.jsonl` (a value a line) and `.json`: its `text` / `content` / `sentence` / `body` |
| markup | `.html .htm .xhtml .xml`: the text, a block tag ending a line, scripts, styles and comments dropped, entities decoded |
| documents | `.docx` (word/document.xml), `.odt` (content.xml), `.epub` (every chapter) |
| archives | `.zip` (ZIP64 too: any size; stored or deflated), `.tar`, `.tar.gz` / `.tgz`, `.gz` - opened member by member, archives inside archives followed four deep |

Archives are read member by member and text a line at a time, and training
streams every sequence straight into the model, so the corpus's size is not
bounded by memory: a 135 MB text inside a 51 MB zip trains in 25 s in 47 MB
of memory (Rust). For `base64` every file is its bytes, archives giving their
members' bytes. `latticefsm corpus PATH` (`make corpus`) shows what a path
is read as; both ports print the same.

## Networks of machines

`latticefsm/network.py` and `rust/src/network.rs` connect machines the way
Keras connects layers. A **layer** is one machine that also *writes*: the
state each move lands in, one letter per move (`a` for state 0, `b` for 1,
...; `13f` writes only the final state), so the next machine reads the walk
the previous one took, and is built over exactly those letters.

```python
from latticefsm.network import Network, Layer, Parallel, PhoneticDecoder, PhoneticEncoder

net = Network("13>(13|13)>13", alphabet="ab")       # or Network([Layer(13), Parallel([Layer(13), Layer(13)]), Layer(13)])
net.compile(accepting=[0], layer_discount=0.8, credit="shared")   # credit="own": each output branch by its own verdict
history = net.fit(x, y, epochs=3, validation_data=(vx, vy))       # {"accuracy": [...], "val_accuracy": [...]}
net.evaluate(vx, vy); net.predict(["abab"]); net.trace("abab"); print(net.summary())
net.save("net.json.gz"); load_network("net.json.gz")              # the same file in the Rust crate

translator = Network("enc>dec")                                   # English in, English out
translator.fit_pairs([("cat", "cats"), ("bus", "buses"), ...], epochs=20)
translator.translate("dog")                                       # 'dogs'
```

| block | what it is |
|---|---|
| `Layer(states, emit="states")` | one machine; any machine option (`life`, `skip`, `learn_focus`, ...) passes through |
| `Sequential([...])`, `a>b` | blocks one after another |
| `Parallel([...], merge, vote)`, `(a\|b)` | blocks side by side on the same input; what they write concatenated or interleaved, each branch's letters shifted past the ones before; as the last block, a `majority` / `any` / `all` vote |
| `PhoneticEncoder()`, `enc` | English in: the text read as its sounds by this repository's phonetic tokenizer (`../PhoneticTokenizer`) - `cats` is `K AE T S` - plus two boundaries `#` of room; the first block only |
| `PhoneticDecoder()`, `dec` | English out: a layer whose 43 states are the sounds (39 phonemes, `#`, `, . ?`), so what it writes is phonemes, spelled back as English |

**Training** is the single machine's, spread over the layers: every machine
walks, the network's verdict is the last block's, the network is rewarded
when it is right and punished when wrong, and the credit travels back
through the blocks - the last in full, each block before it
`layer_discount` times the one after, every parallel branch the block's
share - and each machine spends its share on its own path. A network of one
layer *is* the machine it holds, edge for edge (a test in each port). A
decoder has a right answer at every step, so it is taught step by step: a
wrong sound is punished and the target's path is taught edge by edge; the
blocks before it are *rewarded* by the share of sounds it got right, never
punished (punishing them for the decoder's mistakes keeps reshuffling the
walks the decoder is learning to read). Quiet greedy passes (`predict`,
`evaluate`, `translate`) are deterministic: a tie goes to the lowest state
and nothing is drawn, so a file classifies and translates every string the
same way in both ports (`make parity`).

```bash
make network-fit ARCH="(3|3|3)" CREDIT_BY=own LANGUAGE=mod3-a   # NET=network.json.gz
make network-predict TEXT=aab
make translate-fit TASK=plural && make translate TEXT="cat dog bus"
rust/target/release/latticefsm network fit --arch "enc>dec" --pairs my_pairs.tsv --epochs 30 --save t.json
```

network: every language taught to each network for 4000 episodes (the learning experiment's strings), 5 seeds, 300 held-out strings; `>` one after another, `(a|b)` side by side (majority vote at the end), `f` writes only the final state

| language | network | credit | layers | accuracy: mean | min | max | solved | touched edges |
|---|---|---|---|---|---|---|---|---|
| even-b | `3` | shared | 1 | **1.00** | 1.00 | 1.00 | 5/5 | 18 |
| even-b | `3>3` | shared | 2 | **1.00** | 1.00 | 1.00 | 5/5 | 45 |
| even-b | `3f>3` | shared | 2 | **1.00** | 1.00 | 1.00 | 5/5 | 27 |
| even-b | `(3|3)>3` | shared | 3 | **0.89** | 0.44 | 1.00 | 4/5 | 90 |
| even-b | `(3|3|3)` | shared | 3 | **1.00** | 1.00 | 1.00 | 5/5 | 54 |
| even-b | `(3|3|3)` | own | 3 | **1.00** | 1.00 | 1.00 | 5/5 | 54 |
| even-b | `13` | shared | 1 | **0.81** | 0.51 | 1.00 | 3/5 | 338 |
| even-b | `(13|13|13)` | own | 3 | **0.84** | 0.55 | 1.00 | 3/5 | 1014 |
| contains-aa | `3` | shared | 1 | **1.00** | 1.00 | 1.00 | 5/5 | 18 |
| contains-aa | `3>3` | shared | 2 | **0.77** | 0.50 | 1.00 | 2/5 | 45 |
| contains-aa | `3f>3` | shared | 2 | **0.80** | 0.48 | 1.00 | 3/5 | 27 |
| contains-aa | `(3|3)>3` | shared | 3 | **0.72** | 0.58 | 1.00 | 1/5 | 90 |
| contains-aa | `(3|3|3)` | shared | 3 | **0.62** | 0.48 | 1.00 | 1/5 | 54 |
| contains-aa | `(3|3|3)` | own | 3 | **1.00** | 1.00 | 1.00 | 5/5 | 54 |
| contains-aa | `13` | shared | 1 | **0.76** | 0.63 | 0.99 | 0/5 | 338 |
| contains-aa | `(13|13|13)` | own | 3 | **0.75** | 0.63 | 0.99 | 0/5 | 1011 |
| ends-ab | `3` | shared | 1 | **0.99** | 0.97 | 1.00 | 4/5 | 18 |
| ends-ab | `3>3` | shared | 2 | **0.93** | 0.68 | 1.00 | 3/5 | 45 |
| ends-ab | `3f>3` | shared | 2 | **0.99** | 0.95 | 1.00 | 4/5 | 27 |
| ends-ab | `(3|3)>3` | shared | 3 | **0.95** | 0.83 | 1.00 | 3/5 | 90 |
| ends-ab | `(3|3|3)` | shared | 3 | **0.86** | 0.69 | 1.00 | 2/5 | 54 |
| ends-ab | `(3|3|3)` | own | 3 | **0.99** | 0.97 | 1.00 | 4/5 | 54 |
| ends-ab | `13` | shared | 1 | **0.91** | 0.88 | 0.98 | 0/5 | 288 |
| ends-ab | `(13|13|13)` | own | 3 | **0.95** | 0.92 | 0.98 | 0/5 | 851 |
| mod3-a | `3` | shared | 1 | **1.00** | 1.00 | 1.00 | 5/5 | 18 |
| mod3-a | `3>3` | shared | 2 | **0.75** | 0.43 | 1.00 | 2/5 | 45 |
| mod3-a | `3f>3` | shared | 2 | **0.70** | 0.39 | 1.00 | 2/5 | 27 |
| mod3-a | `(3|3)>3` | shared | 3 | **0.68** | 0.61 | 0.78 | 0/5 | 90 |
| mod3-a | `(3|3|3)` | shared | 3 | **0.88** | 0.64 | 1.00 | 3/5 | 54 |
| mod3-a | `(3|3|3)` | own | 3 | **1.00** | 1.00 | 1.00 | 5/5 | 54 |
| mod3-a | `13` | shared | 1 | **0.72** | 0.68 | 0.77 | 0/5 | 326 |
| mod3-a | `(13|13|13)` | own | 3 | **0.73** | 0.68 | 0.78 | 0/5 | 968 |

English to English: each task taught to each network for 20 epochs, every fourth pair held out; exact means every sound right

| task | network | pairs taught / held out | sounds right while taught | exact: taught | held out | translations (taught, taught, held out, held out) |
|---|---|---|---|---|---|---|
| plural | `enc>dec` | 35 / 11 | 0.90 | **0.80** | 0.09 | cat → cats, dog → dogs, hat → hats, box → b ks z (boxes) |
| plural | `enc>13>dec` | 35 / 11 | 0.11 | **0.03** | 0.00 | cat → ksiz (cats), dog → dawrs (dogs), hat → rocam (hats), box → boats g (boxes) |
| past | `enc>dec` | 21 / 7 | 0.89 | **0.76** | 0.00 | walk → walked, play → played, jump → pt (jumped), look → l kt (looked) |
| past | `enc>13>dec` | 21 / 7 | 0.04 | **0.00** | 0.00 | walk → st kt (walked), play → pulled (played), jump → fid st (jumped), look → cld s (looked) |

What it says: three parallel branches each credited by their own verdict
(`own`) do as well as one three-state machine (every language on 19 of the
20 seed runs in this port) and far better than the same branches sharing one
verdict, where a branch that was right is punished whenever the vote was
wrong. A vote of three thirteen-state machines is a few points better than one
on `even-b` and `ends-ab`, and level on the other two. Stacking does not
help these four languages - one three-state machine already solves them, and
a layer after it has to learn to read a code that is still changing. English
to English, a decoder reading the sounds directly learns its pairs (most of
them exactly, `cat → cats`, `walk → walked`) but generalises to unseen words
hardly at all: its rows are `(sound before, sound read)` pairs, each learned
on its own, and an unseen word brings unseen pairs. A hidden machine between
the encoder and the decoder still loses the signal at 20 epochs; at a lower
temperature and more epochs it reaches 0.4 of its pairs exactly (`Limits`).

## Training on text, audio, images and video

An **encoder** is a network's first block: it turns what the network reads
into symbols (`latticefsm/media.py`, `rust/src/media.rs`, standard library
only, exact integer arithmetic so both ports turn every file into the same
symbols - a test holds them to it, every format and a real video file
included).

| encoder | reads | writes |
|---|---|---|
| `text` | any text | 43 characters: `a-z 0-9`, `_` for a space, `. , ? ! ' -` |
| `enc` | English | its sounds (the phonetic tokenizer, above) |
| `audio` | a WAV file | the phonetic tokenizer's 64 acoustic units `q0`-`q63` (`../PhoneticTokenizer`, learned from recordings) |
| `image` | PNG (8-bit), PPM / PGM, BMP (24/32-bit) | a `grid x grid` (8) of brightness levels `L0`-`L3`, row by row or `center-out` |
| `diff` | frames: a directory of images, or a video file | `|` and each frame's cell-by-cell change from the last: `D-2 D-1 D0 D+1 D+2` |
| `video` | a directory of frames and its `audio.wav`, or a video file (ffmpeg splits it, when installed) | `|` per frame, the first frame as an image and the rest as differences (keyframe and deltas, as video codecs do), each followed by the audio units that sound during it |

```bash
make media-data KIND=video && make media-fit KIND=video ARCH="diff>3" EPOCHS=150
latticefsm network make-data --kind audio --out data/audio          # spoken nouns, singular / plural
latticefsm network fit --arch "audio>3" --data data/audio/labels.tsv --epochs 30 --save a.json
latticefsm network encode clip.mp4 --arch "video>13"                # what the network reads
```

```python
net = Network("image>13").compile(accepting=[0])
net.fit(["a.png", "b.ppm", ...], [True, False, ...], epochs=30)
Network([ImageEncoder(grid=16, levels=8, order="center-out"), Layer(13)])
```

A data file is `input<TAB>verdict` lines (`1/0`, `yes/no`), paths relative to
it; `make-data` writes a synthetic one of each kind. A file is decoded once
and its symbols kept (by path, size and time), however many epochs read it.

media: each synthetic data set taught to a network that reads it for 30 epochs (discount 0.95), every fifth example held out; text: is there a negation, image: are the stripes vertical, audio: is the spoken noun plural, video: does the square move right

| data | network | examples (held out) | symbols per example | alphabet | held out before | taught after | held out after |
|---|---|---|---|---|---|---|---|
| text | `text>13` | 60 (12) | 19 | 43 | 0.50 | 0.46 | **0.50** |
| image | `image>3` | 60 (12) | 64 | 4 | 0.50 | 0.40 | **0.25** |
| audio | `audio>3` | 60 (12) | 18 | 64 | 0.50 | 0.96 | **0.67** |
| video | `diff>3` | 60 (12) | 455 | 6 | 0.50 | 0.54 | **0.58** |
| video | `video>13` | 60 (12) | 533 | 74 | 0.50 | 0.50 | **0.50** |

What it says: every kind of data goes in and trains, and one task learns
clearly - spoken plurals, eighteen units a word, which a three-state machine
fits (0.96) and partly generalises (0.67). The others stay near chance at
30 epochs: a verdict credited once at the end of a 64-symbol picture or a
500-symbol clip reaches back only as far as the discount lets it (0.95¹⁰⁰ is
0.006), so a feature spread over the whole input is mostly not learned. Run
longer, `diff>3` reaches 0.83 held out at 150 epochs on one seed, not
reliably. A machine that reads a picture needs a reason to remember
something across it, and these do not yet have one (`Limits`).

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
| `rust/src/network.rs`, `phonetic.rs` | networks of machines: the blocks, the architecture string, compile / fit / evaluate / predict / translate, the file; English in and out through `../PhoneticTokenizer/rust` |
| `rust/src/ngram.rs`, `autotrain.rs`, `b64.rs` | the N-gram model (Kneser-Ney over adaptive edges; words, characters, sounds, base64), automatic training from a corpus and through Ollama, base64 for anything |
| `rust/src/media.rs` | text, WAV (acoustic units), PNG / PPM / BMP, frame differences and video (frames, ffmpeg) as symbols; the synthetic data sets |
| `rust/src/http.rs`, `server.rs` | the HTTP/1.1 server, the JSON routes, the static files of the frontend |
| `frontend/` | the React app: `src/App.jsx`, the five panels in `src/components/`, `src/api.js`, `src/matrix.js`; `dist/` built and committed |
| `rust/src/json.rs`, `gzip.rs`, `rng.rs` | JSON, gzip and the generator, written out: no dependencies |
| `rust/src/bin/latticefsm.rs` | the CLI: demo, train, run, teach, accuracy, table, stats, tick, stimulate, compress, expand, core-run, focus, skip, rearrange, network, experiment, serve |
| `rust/tests/` | the machine and the server over a real socket |
| `latticefsm/` | the Python reference: `edge.py`, `lattice.py`, `machine.py`, `geometry.py`, `focus.py`, `compress.py`, `network.py`, `phonetic.py`, `media.py`, `ngram.py`, `autotrain.py`, `b64.py`, `languages.py`, `experiment.py`, `cli.py` |
| `data/corpus/` | what the N-gram model trains on by default: prose and sentences (copies), and whatever is dropped in |
| `tests/test_latticefsm.py` | 98 tests, parity with the Rust crate among them; `frontend/test/` the frontend's (`tests/README.md`) |
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
* N-grams: autotrain with Ollama is tested against a stand-in, not a real
  model, so its effect on held-out perplexity is unmeasured here. The corpus
  is small (56 KB) and hard-wrapped, so a line often ends mid-sentence and
  `</s>` is often a likely guess. Vocabulary grows without limit (no pruning
  or cut-off); the whole distribution is computed per query, `O(V · N)`.
  Words, characters and sounds are lower-cased; sounds lose spelling.
  Sampling at a temperature draws from each port's own generator, so only
  greedy generation matches across ports.
* Networks: stacked layers learn less reliably than one machine on these
  languages, and a hidden layer between an encoder and a decoder mostly
  loses the signal (0.03-0.11 of the pairs exactly at 20 epochs; 0.4 at
  temperature 0.3 over 40). A decoder writes one sound per symbol it reads,
  so a translation is at most as many sounds as the input plus the
  encoder's two boundaries; a parallel block in front of it doubles what it
  reads. It generalises to unseen words hardly at all. The server holds one
  network beside its machine (the Talk tab); the other tabs show the machine.
* Media: images, frame differences and video train but mostly stay near
  chance (above); credit reaches back along a run only as far as its
  discount, and a verdict about a whole picture is credited at its last
  cell. Audio is read through one fixed codebook of 64 units learned from
  the phonetic tokenizer's own voice; images are brightness only, no colour;
  PNGs must be 8-bit and not interlaced; a video file needs ffmpeg.
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
