# GameLattice

**A RadixCyclicNN whose step is a game, feeding a LatticeFSM whose every
state is one of the encoder's outputs, read out by decoders that are
separate from the lattice - with a body around it.** The senses are its
encoders, the effectors its decoders, and every effector's output comes
back in through a sense. [`DESIGN.md`](DESIGN.md) is the contract;
[`results/README.md`](results/README.md) is what it measured.

**Status: a first build in Rust.** The crate in `rust/` (`gamelattice`,
standard library only, depending on the sibling crates `../LatticeFSM/rust`
and `../RadixCyclicNN/rust`) has the encoder (§4), the lattice kinds (§5),
the decoder protocol with its four mechanisms (§6), the credit landings
(§7), the efference copy (§8), a text sense over the four regular languages
and a move tape over Nim, the gate, the clock, the selector, babbling, and
experiments E1 and E2 of §17. The Python reference the design planned comes
after the numbers, not before (§21.8, decided by the author: Rust first).
An HTTP server in the crate and a React frontend in `frontend/` show the
stage game naming a state step by step (§16). Not built yet: the senses of §9
beyond text and the tape, the association lattice, the curriculum,
persistence of a whole body.

## Quick start

```bash
cd GameLattice
make demo                                   # even-b: the encoder names the lattice's states, the oracle credits each step
make train LANGUAGE=mod3-a EPISODES=4000    # held-out accuracy every tenth of the way
make plain LANGUAGE=mod3-a                  # the control: the lattice alone, per-step oracle
make nim                                    # babble 200 games, then 400 against a random opponent
make e1 && make e2                          # the measurements, to results/
make test                                   # cargo test, clippy, the formatter check, the frontend's tests
make serve-prod                             # the API and the built frontend on http://127.0.0.1:8000/
```

Needs Cargo. Everything is deterministic given `SEED`.

## What is here, in one paragraph

`GameGraph` (`rust/src/encoder.rs`) is a `radixnet::Graph` as it is, with a
belief over the lattice's `S` states plus abstain on every node, keyed by
where the node sits in the walk window (the current gram, the one before,
older, a child) and by the state the lattice was in. At every position the
window's nodes are sold seats in a uniform-price auction with a reserve,
play the congestion game over the states (`rust/src/game.rs`: right and rare
takes the pot, regret matching plus, sampled Shapley in antithetic pairs),
and the aggregate's argmax is the named state. Learning, the label is known
before the game: the lattice's own prediction when the oracle says it is
right, else the right state the lattice's row ranks highest. The lattice
(`rust/src/lattice.rs`) is a `latticefsm::Machine` with a kind, a
topographic flag, an abstain floor and a reasons table; its row is credited
per step by the oracle through `credit::land`, which lands a sharp verdict
on one edge in full, a dull one discounted within the run, and a reafferent
step at exactly zero. The decoders (`rust/src/decoders/`) are readouts,
repertoires on edges, propose-and-replay through a referee, and the taste
gate; the Nim referee (`rust/src/referee.rs`) refuses, replays and grades.

## The web server and the frontend

```bash
make serve-prod        # one process, no npm: `gamelattice serve` with the committed frontend/dist on :8000
make serve             # the backend on :8000 and Vite with hot reload on :5173 (/api proxied); Ctrl-C stops both
make frontend-test     # the frontend's unit tests (node --test)
```

`gamelattice serve [--host 127.0.0.1] [--port 8000] [--frontend-dir ../frontend/dist] [--results-dir ../results]
[--language even-b] [--states 13] [--window 8] [--n 1] [--seed 1]` holds one body for a language and one Nim
body behind locks, on `latticefsm::http` (the sibling's dependency-free server, reused). Outside a job a busy
body answers 409 at once; a `POST` with `"job": true` runs in the background (`GET /api/job`, `POST
/api/job/stop`), one at a time. Everything outside `/api/` is the frontend, `index.html` for any route of it.

| route | does |
|---|---|
| `GET /api/health`, `/api/status`, `/api/languages` | the shape and settings; everything in brief; the four languages |
| `POST /api/new` | a fresh body: language, states, window, n, seed, the E1 arms (and `nim` settings) |
| `POST /api/train` | `{episodes, max_length, job}`: random strings, held-out accuracy on a fixed 300 every tenth |
| `POST /api/classify`, `/api/run` | `{text}`: the verdicts and the naming trace, quiet or as one learning episode |
| `GET /api/matrix?symbol=`, `/api/edge`, `/api/table` | the lattice as `latticefsm serve` shows it |
| `GET /api/encoder` | the graph, the named-state histogram, diversity, wealth, thinks, the richest nodes |
| `GET /api/nim`, `POST /api/nim/new`, `/api/nim/babble`, `/api/nim/play` | the Nim body: babble, play, the totals |
| `GET /api/results` | `results/e1.json` and `e2.json` as they are |
| `GET /api/job`, `POST /api/job/stop` | the background job |

The tabs ([`frontend/README.md`](frontend/README.md)): **Body** (make, train with the held-out curve, classify
or run a string with its naming trace - per character the seats, their bids and the slots they played, the
aggregate over the states with the named state and the lattice's prediction), **Game** (one step in detail: the
seats as cards with their beliefs and Shapley values, the auction, the equilibrium), **Lattice** (the matrix
slice by symbol, the greedy table), **Encoder**, **Nim**, **Results** (E1 and E2 from `results/`), **Settings**.

## The numbers

E1, the four languages, three seeds, held-out: the game arm **0.961**
against the plain oracle lattice's **0.961** (LatticeFSM's marks: 0.894
whole-run, 0.924 oracle, 0.944 explore). The stage game matches the lattice
alone and does not beat it. E2, Nim by window width: wins 0.47-0.50 against
a random opponent, first-try legality 0.13-0.21, under 0.04 bits of
information about the position in the named state - contradicted as built,
with the diagnosis in `results/README.md`.

## The pieces

| file | what |
|---|---|
| `rust/src/game.rs` | the stage game: seats, the auction, the payoffs (correctness, an oracle's set of right states, belief, a refusal), regret matching plus, the best-response cycle check, sampled Shapley, the aggregate |
| `rust/src/encoder.rs` | `GameGraph`: the radix graph with beliefs per node and bucket, the window, `name`, `settle`, `think_at`, the diversity and wealth readouts |
| `rust/src/lattice.rs` | `GameLattice`: a machine with a kind, the topographic flag, the abstain floor, the reasons table |
| `rust/src/verdict.rs`, `credit.rs` | the one verdict record; where it lands |
| `rust/src/forward.rs`, `efference.rs` | the forward model's quiet prediction and its lesson; the efference copy and its three returns |
| `rust/src/decoders/` | the protocol and the four mechanisms |
| `rust/src/referee.rs` | the language oracle; Nim |
| `rust/src/select.rs`, `gate.rs`, `clock.rs` | the release rule; the reserves and surges; the shared clock |
| `rust/src/body.rs`, `babble.rs` | the organism wired for a language and for Nim; babbling |
| `rust/src/experiment.rs` | E1 and E2 |
| `rust/src/server.rs` | the HTTP API (§16) over `latticefsm::http`: the body and the Nim body behind locks, the jobs, the frontend |
| `frontend/` | the React frontend: Body, Game, Lattice, Encoder, Nim, Results, Settings; `dist/` committed |
| `rust/tests/` | the design choices as tests: the game's invariants, quiet classification, a trace that moves nothing, no collapse, lessons move the row and agreement does not, reafference at zero, sharp on one edge, a decoder must name its sense, the release rule, the topographic axis, Nim, taste; the HTTP API |
