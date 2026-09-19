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
python3 -m tests.test_v1            # 20 tests
```

Pure standard library, no dependencies. **Stockfish is optional** — install it
(`apt-get install stockfish`, or set `STOCKFISH_PATH`) to play and learn against
a real engine; everything degrades to the built-in opponents without it.

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
| `cortex/checkpoint.py` | save / load / rotating `CheckpointManager`. Atomic writes, gzip by extension |
| `cortex/credit.py` | vocabulary coverage, cross-region ensembling, **Shapley over regions** — exact below 12 regions, Monte-Carlo above |
| `cortex/routing.py` | the region-level **auction** (uniform price, so truthful bidding is dominant) and **congestion** settlement |
| `cortex/cortex.py` | regions, routing, supervised training, **self-play with outcome credit**, **distillation from a teacher**, evaluation |
| `cortex/stockfish.py` | Stockfish as opponent *and* teacher — persistent UCI process, FEN/UCI conversion, `evaluate`, and `score_moves` (MultiPV: every legal move scored in one search). Optional |

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

## Stockfish: what worked and what did not

Stockfish is wired in two ways — as an **opponent** (throttled by Skill Level)
and as a **teacher**, whose MultiPV search scores every legal move in a position
in one call. 120 positions yield ~1900 supervised targets in 6 seconds, where
self-play yields one number per game.

**The distillation did not work, and the diagnostic says why.** Spearman
correlation between the grade head and Stockfish's centipawns, on held-out
real-game positions:

| grade trained from | ρ vs Stockfish | picks Stockfish's best move |
|---|---|---|
| heuristic | **0.133** | 0.267 |
| distilled, 200 positions | −0.056 | 0.000 |
| distilled, 800 positions | −0.086 | 0.033 |

No correlation, and four times the data makes it slightly worse. The function is
not in the hypothesis class: Stockfish evaluates with an NNUE over the whole
board, and this network sees ~27 features describing one move through 36 hidden
units. Adding `EXCHANGE` and `POSITION` features (does the move hang the piece,
material balance, gives check, centre) did not close it either. **A teacher
cannot teach what the student's vocabulary cannot express** — the same ceiling
that the checkers midpoint and the go influence features ran into, reached here
from the strong side.

**Two real defects surfaced on the way, and both are fixed.**

*Distillation was corrupting the validity head.* Training `valid=1.0` on every
teacher-scored move floods it with positives — a teacher only ever scores legal
moves — and dropped legality from 0.925 to 0.748. Distillation now trains grade
only. Two heads, two signals, never conflated.

*Rule selection was optimising the wrong thing.* It picked the rule whose top
choice was most often legal, so after distillation it chose the validity-only
rule at a perfect 1.000 top-choice-legal, **discarded the grade head entirely**,
and produced the worst material of any configuration. A rule that reliably plays
a legal blunder is not a good rule. Selection now scores the position the move
leads to, with an illegal pick taking the worst value — legality enforced by
consequence rather than as the whole target.

**And one measurement worth having.** Checked against Stockfish, **99 of 200
randomly generated chess positions are unreachable** — the side not to move
already in check, or a missing king. Half the supervised training signal was
coming from boards no game can produce. `stockfish.playable()` now filters them,
and it is a pure function that works without Stockfish installed.

## Regions as players: the auction and Shapley credit

GTMNN divided credit among 4096 micros by Monte-Carlo because `2^4096` coalitions
cannot be enumerated. A dozen regions changes the arithmetic — `2^12 = 4096`
coalitions is a loop — so **exact Shapley is the default here** and Monte-Carlo
the fallback. That was the specific benefit claimed for moving from micros to
regions, and it holds.

Vocabulary coverage, which is what a region bids with:

| region | chess | checkers | go | sudoku |
|---|---|---|---|---|
| chess, checkers | 1.00 | 1.00 | 0.17 | **0.00** |
| go | 0.12 | 0.17 | 1.00 | 0.25 |
| sudoku | **0.00** | **0.00** | 0.17 | 1.00 |

Exact Shapley over those regions, efficiency error at machine precision:

| game | R0 (chess, checkers) | R1 (go) | R2 (sudoku) | eff. error |
|---|---|---|---|---|
| chess | **+0.473** | −0.141 | −0.000 | 5.6e-17 |
| checkers | **+0.487** | −0.123 | −0.000 | 0.0 |
| go | −0.116 | **+0.599** | +0.121 | 1.1e-16 |
| sudoku | −0.000 | −0.443 | **+1.071** | 2.2e-16 |

The numbers are semantically right, not merely well-formed. Each game's own
region takes the largest positive share. Wrong regions score **negative** — they
actively hurt. The go region earns **+0.121 on sudoku**, which is real
cross-region value from the placement mechanics they share. And the sudoku
region scores **exactly −0.000 on chess**, where its coverage is exactly 0.00:
**the null-player axiom appearing in measured data.**

The auction settles as it should — each game's own region wins its seat at a bid
of 0.50 — and congestion splits a correct claim among claimants, which is the
same split reward that made GTMNN's micros specialise, applied to regions.

## Cross-region transfer: real, and small

Regions share no weights, so an ensemble is the **only** channel by which one
region's learning can reach another's game (`DESIGN.md` §15.1):

| game | own region | auction top-2 | all regions | baseline |
|---|---|---|---|---|
| chess | 0.948 | 0.948 | 0.948 | 0.519 |
| **checkers** | 0.905 | **0.930** | **0.930** | 0.583 |
| go | 1.000 | 1.000 | 1.000 | 0.561 |
| sudoku | 1.000 | 1.000 | 1.000 | 0.500 |

**One case improves and the rest are flat.** Checkers gains 0.025 from the go
region. That is a real answer to the open question rather than a hopeful one:
the graph earns its place by *placing* games correctly — the wrong region is
measurably worse, per the negative Shapley values — and not by regions teaching
each other, which barely happens.

## Replay on join

`rehearse` interleaves every game in a region when one of them is new. Growth is
an identity, but the training that *follows* is not protected. Measured by
training chess alone, then admitting checkers:

| | chess legality | chess top-choice-legal |
|---|---|---|
| before checkers joins | 0.898 | 0.933 |
| after join, no rehearsal | 0.925 | 0.811 |
| after join, with rehearsal | 0.885 | 0.833 |

**Forgetting is mild and replay's benefit is not detectable at this scale.**
Joining checkers even *raised* chess legality. Reported as it measured rather
than as it was expected to: the mechanism is implemented and available, and this
corpus is too small to show it earning its keep.

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
