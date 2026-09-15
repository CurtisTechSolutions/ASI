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
python3 -m tests.test_v1            # 14 tests
```

Pure standard library, no dependencies.

## What V1 contains

| module | what it is |
|---|---|
| `cortex/board_games.py` | correct rules for chess, checkers and go, self-checked at import (pins, path blocking, forced capture, suicide, ko) |
| `cortex/sudoku.py` | sudoku rules, generator and solver |
| `cortex/engine.py` | opponents for all three two-player games. Chess: material plus one-ply reply. Checkers: material, kings and advancement — **depth 2 wins 12/12 against random where depth 1 wins 3/12**, because one-ply greedy walks pieces into forced captures. Go: greedy on area score, and it genuinely passes, which is the only way a game can end. `depth=0` is a random-legal control everywhere |
| `cortex/games.py` | adapters: `mechanics`, `candidates`, `generalise` (mechanic → features), `reward` |
| `cortex/vocabulary.py` | mechanic → input slots; extension only appends, slot indices are permanent |
| `cortex/sbnn.py` | the growing network, two heads (`p_valid`, `grade`), growth on a loss plateau |
| `cortex/graph.py` | Jaccard distance over mechanics, triangle-inequality check |
| `cortex/cortex.py` | regions, routing, supervised training, **self-play with outcome credit**, evaluation |

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

| game | legality | majority baseline | top pick legal | random pick | illegal in play | vs engine (4 games) |
|---|---|---|---|---|---|---|
| **go** | **1.000** | 0.564 | **1.000** | 0.564 | **0.000** | **1 win** (55–26); losses 35–46, 39–42 |
| **checkers** | **0.975** | 0.702 | **0.900** | 0.298 | 0.05 – 0.11 | **2 wins** (8–0, 4–0) |
| sudoku | **1.000** | 0.500 | **1.000** | 0.500 | — (solitaire) | — |
| chess | 0.864 | 0.651 | 0.925 | 0.349 | **0.000 – 0.013** | 0 wins, −3700 to −5000 material |

**Go and sudoku are solved for legality** — the go region plays 125-ply games
against the engine without a single illegal move, and sudoku never misjudges a
placement.

**Three findings from making go and checkers playable:**

*The missing feature was the midpoint.* Checkers sat at 0.782 legality and a
top-choice-legal of 0.425 until the features included **the square between from
and to** — a jump is legal only if an enemy sits there, and without it the
network cannot tell a legal jump from an illegal jump-shaped move. This is the
same path predicate `information.py` found mattered most for chess. Adding it
took checkers to 0.906 / 0.725 **and chess to 0.921 / 0.975**, because they share
a region: a feature added for one game improved the other, which is
within-region transfer actually happening rather than being hoped for.

*No single ranking rule works for every game.* Measured:

| rule | chess | checkers | go |
|---|---|---|---|
| `p_valid` alone | **1.000** | 0.250 | **1.000** |
| gate on `p_valid`, then grade | 0.850 | **0.700** | **1.000** |

Chess and go want validity to decide outright; checkers does not, because forced
capture makes legality a property of the **move set** rather than the move. Each
region therefore selects its own rule by measurement (`select_rule`), on the
distribution it will actually face.

*Rewarding shape teaches shape.* The checkers grade head originally rewarded any
two-square move, so it learned that jump-SHAPED moves are good — and the ranking
then selected exactly the illegal ones. Rewarding only real captures fixed it.

**Chess is genuinely learned but weak**: 0.856 against a 0.636 baseline is
+0.22, not the +0.36 the bare number suggests. Its top pick is legal 82% of the
time against a 36% random baseline, so it has learned a great deal about which
moves are legal, and not enough to play without occasional illegal picks.

More training does not fix it — 400, 1200 and 3000 episodes all land near 0.80
while the hidden layer grows to 120 units, so this is not undertraining.
Balancing the training candidates does help, by **+0.037** (0.819 → 0.856), and
is now the default: chess candidates are only 36% legal, and training on that
distribution teaches the prior rather than the rule. But `information.py`
measures **0.977** as reachable with these same features under fully balanced
sampling, so roughly 0.12 of the gap remains unexplained and is the first thing
to chase.

## Grade, learned from outcomes

V1's `grade` was a hand-written heuristic, which is why go could play 125 legal
plies and lose 0–81: *legal* and *good* are different questions and only the
first was trained. `Cortex.selfplay` now plays games, waits for the result, and
credits every move the cortex made with the discounted outcome.

**Three things were each necessary, and none sufficient alone.** Measured on go
against the engine, 6 games:

| features | grade | γ | W–L | mean score gap |
|---|---|---|---|---|
| liberty only | heuristic | 0.95 | 0–6 | −80.2 |
| + influence, connection | heuristic | 0.95 | 0–6 | −80.2 |
| + influence, connection | outcome | 0.95 | 0–6 | −81.0 |
| **+ influence, connection** | **outcome** | **0.995** | **3–3** | **−14.3** |

*The vocabulary sets the ceiling.* Go's features were `x, y, empty, liberties,
captures, group size` — sufficient for legality (1.000) and structurally
incapable of expressing territory. `INFLUENCE` and `CONNECTION` (stone density
and distance by radius, edge proximity, whether the move joins two friendly
groups) are what a placement game needs to be *played*. This is the checkers
midpoint lesson one level up.

*The discount must match the game's length.* At γ=0.95 a 125-ply go game gives
its opening move `0.95¹²⁵ ≈ 0.0017` of the outcome — credit that has vanished.
The discount is now derived per game, `γ = 0.5^(1/plies)`, so the first move
keeps half the credit of the last. No constant can serve games of different
lengths.

*Unfinished games need a value.* Fifty self-play chess games once produced fifty
draws and therefore zero gradient, because they all hit the ply cap. Each game
now supplies a bounded, zero-sum `value(state, side)` used when no winner exists.

**Where it stands.** Checkers and go are real players — checkers wins 2 of 4
against a depth-1 engine and reaches 0.975 legality; go wins 1 of 4 and its
losses are close (35–46) where they were 0–81. Chess has excellent legality
(**0.0–0.013 illegal in play**, from 0.10–0.69) and still loses material
consistently: it is the hardest of the three and has no search, so the engine
takes free pieces the cortex cannot see coming.

## What is not here yet

Grade is a hand-supplied heuristic rather than a learned value, there is no
search, no replay when a game joins a region, no region-level auction or Shapley
credit (`DESIGN.md` §9.2), and no frontend. Regions do not yet transfer anything
between each other — which is `DESIGN.md` §15.1, the open question that decides
whether the graph is doing real work or only routing.
