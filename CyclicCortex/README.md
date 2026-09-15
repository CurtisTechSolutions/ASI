# CyclicCortex — V1

A cyclic graph over games where **distance is similarity**, partitioned into
**regions**, each region owning **one Self-Building Neural Network** that reads
the **common denominators** — generalised actions meaning the same thing in
every game in that region.

`DESIGN.md` is the specification. This is what runs today.

## Quick start

```bash
cd CyclicCortex
python3 -m cortex.cli demo          # build, train, play chess, add sudoku
python3 -m cortex.cli map           # the similarity graph and its regions
python3 -m cortex.cli play          # chess against the engine
python3 -m cortex.cli sudoku        # solve puzzles by the network's own ranking
python3 -m tests.test_v1            # 7 tests
```

Pure standard library, no dependencies.

## What V1 contains

| module | what it is |
|---|---|
| `cortex/board_games.py` | correct rules for chess, checkers and go, self-checked at import (pins, path blocking, forced capture, suicide, ko) |
| `cortex/sudoku.py` | sudoku rules, generator and solver |
| `cortex/engine.py` | the chess opponent — material plus one-ply reply, `depth=0` gives a random-legal control |
| `cortex/games.py` | adapters: `mechanics`, `candidates`, `generalise` (mechanic → features), `reward` |
| `cortex/vocabulary.py` | mechanic → input slots; extension only appends, slot indices are permanent |
| `cortex/sbnn.py` | the growing network, two heads (`p_valid`, `grade`), growth on a loss plateau |
| `cortex/graph.py` | Jaccard distance over mechanics, triangle-inequality check |
| `cortex/cortex.py` | regions, routing, training, evaluation |

## What it does

**Regions form as predicted.** Adding chess, checkers, go and sudoku:

```
chess     founded region 0
checkers  joined  region 0   (distance 0.529, vocabulary grew by 1 input)
go        founded region 1   (nearest 0.739)
sudoku    founded region 2   (nearest 0.778)
```

Chess and checkers share a region because they are both movement games; go and
sudoku each found their own. Checkers joining chess grew the shared vocabulary
by a **single input** (`FORCED_CAPTURE`) — everything else was already there,
which is the economy the design depends on.

**Sudoku is nearer go than chess** — 0.778 against 0.905 — as the mechanic sets
predict, and that is a test (`test_sudoku_is_nearer_go_than_chess`) rather than
an observation.

**Growth is an identity.** Hidden and input growth leave the network's output
bit-identical; `test_growth_identity` asserts it over 300 inputs. A region that
has learned chess does not get worse when another game joins it.

**Zero triangle-inequality violations**, so "distance is similarity" has a
consistent geometry.

## Measured results

Always reported against a baseline, because the candidate sets are not balanced
and raw accuracy would flatter a model that just predicts the majority class.

| game | legality accuracy | majority baseline | top pick legal | random pick |
|---|---|---|---|---|
| sudoku | **1.000** | 0.500 | **1.000** | 0.500 |
| chess | 0.805 | 0.636 | 0.833 | 0.364 |

**Sudoku is solved** — the network never misjudges a placement.

**Chess is genuinely learned but weak**: 0.805 against a 0.636 baseline is
+0.17, not the +0.30 the bare number suggests. Its top pick is legal 83% of the
time against a 36% random baseline, so it has learned a great deal about which
moves are legal, and not enough to play without occasional illegal picks. More
training does not fix it — 400, 1200 and 3000 episodes all land near 0.80 while
the hidden layer grows to 120 units, so this is a representation ceiling rather
than undertraining. `information.py` measures 0.977 as reachable with these
features under balanced sampling, so the gap is a training-distribution problem,
not a missing-feature one.

Against the engine, the cortex loses. The `grade` head is material delta with no
search, so that is expected at V1; the interesting number is the illegal-pick
rate, not the result.

## What is not here yet

Grade is a hand-supplied heuristic rather than a learned value, there is no
search, no replay when a game joins a region, no region-level auction or Shapley
credit (`DESIGN.md` §9.2), and no frontend. Regions do not yet transfer anything
between each other — which is `DESIGN.md` §15.1, the open question that decides
whether the graph is doing real work or only routing.
