# GREN — V1

**G**ame **R**ule **E**ncoder **N**etwork. GREN does not play. It works out
**which game is being played**, by guessing, being refused, and treating every
*no* as the lesson.

`DESIGN.md` is the specification. This is what runs.

## Quick start

```bash
cd GREN
python3 -m gren.cli explore     # probe every game; what does each one refuse?
python3 -m gren.cli similar     # discovered similarity vs CyclicCortex's hand-written sets
python3 -m gren.cli policies    # failure rate and rule coverage by probe policy
python3 -m gren.cli tree        # the radix tree over discovered signatures
python3 -m gren.cli package     # the GamePackage handed to the player network
python3 -m tests.test_gren      # 10 tests
```

Pure standard library. Probes `CyclicCortex`'s games, so both must be present.

## The result that matters

**CyclicCortex hard-codes the mechanic set of every game it plays. GREN
discovers those sets by probing, and the two agree.**

Spearman correlation between GREN's discovered pairwise distances and the
hand-written mechanic sets: **ρ = 0.886**, with the closest pair (chess ↔
checkers) the same under both.

| pair | GREN discovered | hand-written |
|---|---|---|
| chess / checkers | **0.368** | **0.529** |
| chess / go | 0.579 | 0.684 |
| checkers / go | 0.632 | 0.667 |
| go / sudoku | 0.800 | 0.778 |
| chess / sudoku | 0.840 | 0.905 |
| checkers / sudoku | 0.880 | 0.900 |

Nothing about the ordering was given to GREN. It played moves, was told no, and
recovered the structure from the pattern of refusals — which is DESIGN §2's
claim (*a game's identity is its refusal boundary*) holding on real games.

## How it works

An oracle **names its own refusal kinds**, the way a compiler emits `E0308`
rather than a bare rejection. That is not GREN cheating — §15 of the spec says a
structured code *is* the class, because rediscovering clusters over `rustc`
output when `rustc` hands you the code is inventing a worse version of an
existing answer. What GREN discovers is **which codes a game actually has, and
in what proportion**:

| game | codes found | dominant refusals |
|---|---|---|
| chess | 7 | `WRONG_PATTERN` 80%, `BLOCKED_PATH` 8%, `SELF_CHECK` 7% |
| checkers | 7 | `EMPTY_SOURCE`, `NOT_YOURS`, `WRONG_PATTERN`, `FORCED_ALTERNATIVE` |
| go | 2 | `OCCUPIED_TARGET` 99%, `SUICIDE` 1% |
| sudoku | 4 | `OCCUPIED_TARGET` 47%, `CONSTRAINT_ROW` 31%, `CONSTRAINT_COL` 17% |

Sixteen codes exist in the shared vocabulary; a game has only the ones probing
finds. Go really does refuse in only two ways.

## The 50% failure rate

Choosing probes by expected information gain is provably maximised at
`p(legal) = 0.5` for a binary oracle, so the target failure rate is **derived,
not set**. Measured at a 1200-probe budget:

| game | failure rate under `eig` |
|---|---|
| chess | **0.480** |
| checkers | 0.545 |
| sudoku | 0.282 |
| go | 0.159 |

Chess and checkers land near the prediction. **Go and sudoku do not, and the
reason is structural rather than a tuning failure:** most points on a go board
are legal (legality density 0.84), so there is no 50/50 split available to find.
The prediction holds where the boundary is tight and does not where it is loose,
which is the honest scope of §4.

**Rule coverage is the stronger result.** On chess:

| policy | failure rate | codes found |
|---|---|---|
| random | 0.598 | **4** |
| boundary | 0.577 | **7** |
| eig | **0.445** | **7** |

Random probing finds four of chess's seven refusal kinds. Probing *at the
boundary* finds all seven. A policy that only ever trips the common refusal
never learns the rare rules, and that is what a probe policy is for.

## A radix tree with the mechanics of a trie

Both halves are load-bearing, and their interaction is one rule:
**a terminal node is never merged away.**

A general class can be a strict prefix of a specific one — `board /
two-player / perfect` is a real answer that can be used while identification
continues, and it has exactly one child, so without the clause compression
would silently delete it. `test_radix_never_merges_a_terminal` asserts it
directly.

Partial-evidence retrieval branches on unknown tokens rather than guessing, so
the answer is a cluster:

```
['category=board']          -> ['chess', 'checkers', 'go']
['refuses=CONSTRAINT_ROW']  -> ['sudoku', 'chess', 'checkers']
```

## What is not here yet

`sandbox.py` (no oracle currently executes generated code), `features.py` and
the operator→mechanic lifting of §8.4 — GREN discovers refusal *codes*, not
STRIPS operators, so minimality (§8.3) is unimplemented. No forest (§20), no
`similarity.py` MDS projection, no API or frontend. The `GamePackage` is built
but `CyclicCortex` does not yet consume it: the games there still carry
hand-written mechanics, and replacing them with discovered ones is the next
step, now that the two are known to agree.
