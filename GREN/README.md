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
python3 -m gren.cli grow        # one growing network across all four games
python3 -m gren.cli regress     # goal regression: backwards from the goal
python3 -m gren.cli transfer    # do the RULES transfer between games?
python3 -m tests.test_gren      # 29 tests

# hand the packages to CyclicCortex, which then builds its map from them
python3 -m gren.cli package --out ../CyclicCortex/data/gren_packages.json
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

**Caveat, measured later (Insights 41):** these policies are not blind. The
candidate pool is half legal by construction and the boundary policies perturb
the legal move list, so the 50% is partly handed over rather than found. Given
only a grammar and a yes/no, a policy that aims at p(accept) = ½ still maps
every rule of chess and go, and the policy that maximises failure is the worst
mapper in every game.

## A radix tree with the mechanics of a trie

**Ordered by the end goal, working backwards.** The order tokens are inserted in
is the order the tree asks about them, so it decides what sits at the root — and
`sorted()` put `adversarial=` there because "a" sorts first, which separates
nothing among four board-ish games. `goal_first()` asks the goal first:
`goal_type=reach-target` for chess, `outlast` for checkers, `maximise-score` for
go, `produce-artifact` for sudoku. Measured, that identifies a game from **one**
token against alphabetical's **2.5** (small N — four games and one
perfectly-discriminating axis makes it nearly free).

The argument that does not depend on N is **knowability**: the goal is the one
thing you can state about an unfamiliar game before you can play it. You know a
conversation is meant to get you a promotion; you do not know its payoff matrix.
Goal-first means the questions the tree asks first are the ones a new game can
answer — and when the goal is unclear, going *up* a level is the semantics rather
than a fallback, because "change how they see me" is a real goal with real
strategies even when "get a promotion" is not yet committed to.

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

## Self-Building Neural Networks

GREN starts knowing **nothing**: no features, and no refusal kinds at all. Both
vocabularies are discovered by probing, so both layers grow. One network across
all four games (`python3 -m gren.cli grow`):

| after | inputs | outputs | params | predict acc |
|---|---|---|---|---|
| chess | 18 | 8 | 656 | 0.710 |
| checkers | 31 | 11 | 1043 | 0.949 |
| go | 40 | 13 | 1309 | 0.889 |
| sudoku | 48 | 16 | 1576 | 0.782 |

It begins at **0 inputs and 0 outputs** and grows to 48 and 16 — an input the
first time a feature name appears, an output the first time a refusal code is
seen. The growth log shows the mechanism directly: sudoku's arrival adds inputs
44–48, and its `CONSTRAINT_ROW` / `CONSTRAINT_COL` / `CONSTRAINT_BOX` refusals
add outputs 14–16.

This is what makes the real expected-information-gain objective computable. The
histogram policy scores a probe from a running tally that knows nothing about
the probe itself — a prior, not a prediction. The learned policy asks the network
for `p(outcome | features(state, action))` and probes where that distribution is
most uncertain, which is §17.3 as specified.

### All three growths are identities, and two of them were subtle

**Hidden** is easy: new units enter with zero *outgoing* weight. **Inputs** are
easy: zero *incoming*. **Outputs are not**, and the obvious fixes both fail:

* Zero weight and zero bias gives the new class `exp(0) = 1` of the mass and disturbs every existing one.
* A large *fixed* negative bias is not enough either. A softmax cares only about **relative** logits, and after training the existing logits can sit far below any constant — so the new class becomes the maximum and takes everything. Measured: **0.244 of the distribution moved.**

It enters with zero weights and a bias 40 below the largest existing bias, so its
logit is constant and provably negligible whatever the input. New classes arrive
at a share of ~1e-18 and are still learnable afterwards (0.967 on a third class).

**And identity-preserving growth is wrong at initialisation.** Zero rows *and*
zero columns symmetry-lock the network: `ah = tanh(0) = 0` zeroes the W2 update,
which zeroes the hidden gradient, which zeroes the W1 update. Nothing but the
biases can ever move and the loss sits exactly where it started — measured at
0.7003, forever, on a trivially separable task. Rows and columns created during
warmup are therefore random; only later ones are identities. Asserted by
`test_sbnn_is_not_symmetry_locked_at_init`.

## The handoff: CyclicCortex now builds its map from these packages

`package --out` writes a JSON file that `CyclicCortex/cortex/discovered.py`
reads. It is a file and not an import because GREN imports CyclicCortex's
adapters in order to probe them; the reverse direction would close the cycle.

With the discovered signature substituted for the hand-written `mechanics` sets,
**CyclicCortex produces the same three regions, the same members, the same zero
triangle-inequality violations and the same playing strength.** The distances
move — GREN's signature has different cardinality — but chess is still nearest
checkers, sudoku is still the outlier, and sudoku is still nearer go than either
board game. See `CyclicCortex/README.md`.

Building the handoff exposed a confusion worth the whole exercise. The gate for
"may this game be placed on the map" had been identification confidence, and
sudoku scores 0.350 there — because it sits 0.80 away from everything GREN has
seen. That is *confidently novel*, not uncertain, and a novel game is exactly the
one that should found its own region. `GamePackage.characterisation` and
`placeable()` split the two questions: do we know what this game is like, versus
which known game is it.

`vocabulary.py` goes further and derives the *input* layout, matching feature
dimensions across games by their legality signature so a shared slot holds one
quantity rather than one name. It recovers real correspondences — sudoku's
`CONSTRAINT_UNIQUE` block separates cleanly into row, column and box — and it
makes chess-to-checkers transfer much worse, for a reason that is a finding about
the architecture rather than about the code: **a shared refusal code is not a
shared rule.** Chess and checkers both refuse `OCCUPIED_TARGET`, but chess
refuses only a target holding your own piece (taking an enemy is a capture) while
checkers refuses any occupied target. Refusals identify a rule's shape, not its
arguments — enough to place a game, not enough to share a weight. The numbers are
in `CyclicCortex/README.md`; the derived vocabulary is opt-in behind
`--discovered-vocab`.

## Goal regression — backwards from the goal to subgoals

Forward search asks "where can I get from here?". Regression asks "what would
have to be true for the goal to hold?", and keeps asking until the answer is
already true. It is how endgame tablebases are built — backwards from mate.

Its usual cost is writing every operator's preconditions by hand, and that hand
is where domain knowledge smuggles itself in. GREN does not need them written:

> **A refusal code is a precondition violation.**

`BLOCKED_PATH` means `move(from,to)` requires a clear path. `CONSTRAINT_ROW`
means `place(x,y,v)` requires `v` absent from row `y`. `why()` already answers
"what blocks this action?" — which is exactly the question regression asks at
every step. The preconditions are **measured, not authored**, and `DESIGN.md`
§8.3's STRIPS operators finally have a source.

### Sudoku: constraint propagation, derived

Asking what blocks every conceivable move on one puzzle:

```
OCCUPIED_TARGET   270      <- discovered by probing, not written down
CONSTRAINT_ROW    158
CONSTRAINT_COL     89
CONSTRAINT_BOX     25
APPLICABLE        187
```

Rank the empty cells by how few values clear all four, and a cell with exactly
one clearing value is a forced move. Nobody coded "naked single" — that
heuristic **is** constraint propagation, and regression produces it from what the
game refuses.

| clues | guided nodes | control nodes | | guided `candidates()` calls | control | |
|---|---|---|---|---|---|---|
| 30 | 53 | 1 436 | **27×** | 1 356 | 1 435 | 1.1× |
| 26 | 63 | 9 032 | **143×** | 1 745 | 9 031 | 5.2× |
| 22 | 81 | 100 129 | **1 240×** | 2 450 | 100 123 | 41× |

By **nodes** — the thing regression reduces — guidance wins everywhere. By raw
work it barely wins on easy puzzles, because scanning 51 cells to save a handful
of nodes is not worth it when almost any order solves them.

### Chess: a subgoal prunes exactly what it excludes

`mate ⟸ check ∧ no-escape`. The expensive conjunct enumerates every opponent
reply; the cheap one is a single `attacked()` call. Regress to the cheap one
first and only pay for the expensive one on moves that clear it:

| positions | % of moves giving check | expensive tests avoided | speedup |
|---|---|---|---|
| pieces scattered at random | 59.5% | 48.7% | 2.2× |
| **from actual play** | **4.3%** | **95.7%** | **7.7×** |

That is the whole law: **regression pays in proportion to how selective the cheap
subgoal is.** A subgoal that 96% of actions fail removes 96% of the expensive
work; one that 60% pass removes almost nothing.

### The measurement that came out at 1.0×, and why

The first attempt scored **exactly 1.0×** against its control. The control had
regressed all along: `is_mate()` is `in_check() and not legal_moves()`, and
Python's `and` short-circuits.

Goal regression over a conjunctive goal *is* a short-circuiting `and` with the
cheap, selective conjunct written first — which is why it is easy to have done
already without noticing, and why a baseline has to be checked for it before any
speedup is believed. `conjunctive()` and `expected_work()` make the ordering
explicit rather than incidental.

## Identify by mechanics, group by rules

Every similarity measure here — the refusal signature, the hand-written mechanic
sets, GTMNN's modifier — ranks chess/checkers the most similar pair and go/sudoku
among the least. Measured **rule transfer** says the opposite:

| pair | shared rule-bearing slots | rule transfer |
|---|---|---|
| chess / checkers | 7 | 0.32 |
| **go / sudoku** | 3 | **0.98** |
| every other pair | 0 | — (no channel) |

`gren.cli transfer` trains a growing code-predictor on one game's
(shared-vocabulary features → refusal code) and tests it *per code* on another.
Two facts decide it, both pinned as tests:

* **`go → sudoku`** transfers the one rule they share — occupied cell — through
  `GRID_PLACE[2]` at LEGAL 1.00 / OCCUPIED_TARGET 0.99, and on sudoku's constraint
  refusals says LEGAL, correctly by its own lights.
* **`chess → checkers`** calls **95% of checkers' legal moves `BLOCKED_PATH`**, a
  code checkers does not have. A jump's occupied midpoint is the piece being
  taken; to chess it is a blocked bishop. Same feature, opposite rule.

A rule transfers exactly when the feature carrying it means the same thing on
both sides, and a shared mechanic *name* does not tell you whether it does.
Agreement of the mechanic measures with rule transfer is **+0.30**; the goal
tree's is +0.07. So the signature stays what it is good at — identification —
and grouping needs either this operational measure or a decomposition deep enough
that a shared name implies a shared condition (§8.3, unbuilt, now with an
acceptance test in `Research/Insights.md`).

One guard, learned the hard way for a third time: a net given no familiar
features emits its bias, and `sudoku → chess` scored OCCUPIED_TARGET 1.00 with
zero shared features. A shared surface that can *carry* a rule — `SIDE` alone
cannot — or the number is a prior.

## What is not here yet

`sandbox.py` (no oracle currently executes generated code), `features.py` and
the operator→mechanic lifting of §8.4. §8.3's minimality criterion — keep
splitting a part while its halves are seen independently — is still unimplemented,
though the STRIPS operators it was for now have a source in `regress.py`. No
forest (§20), no `similarity.py` MDS projection, no API or frontend.
