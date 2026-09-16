# GTMNN — V1

A population of thousands of tiny neural networks that play a game against each
other. Every prediction is a **stage game**: micros bid for a seat, the seated
ones choose an action, the payoff depends on what everyone else chose, and the
output is the equilibrium. Credit is the **Shapley value**. No gradient ever
crosses a micro boundary.

`DESIGN.md` is the specification. This is the implementation of it, with the
measurements it asked for — including the ones that did not come out the way the
spec expected.

## Quick start

```bash
cd GTMNN
python3 -m gtmnn.cli demo        # train a population; every readout
python3 -m gtmnn.cli shapley     # the four axioms, checked numerically
python3 -m gtmnn.cli cycles      # rock-paper-scissors: the cycle detector
python3 -m gtmnn.cli auction     # is truthful bidding really dominant?
python3 -m gtmnn.cli modifier    # GREN's mechanic sets, hashed
python3 -m gtmnn.cli trie        # the game-theory trie; every solver compared
python3 -m gtmnn.cli transfer    # does chess transfer to checkers?
python3 -m gtmnn.cli evolve      # cull, birth, evolutionary stability
python3 -m tests.test_gtmnn      # 49 tests
```

Pure standard library. `gtmnn.games` and `gtmnn.play` read `CyclicCortex`'s game
adapters and GREN's handoff, so those two directories must be present beside
this one; nothing else in the package needs them.

## What holds

Every one of these is asserted in `tests/test_gtmnn.py` rather than claimed here.

| Claim | Measured |
|---|---|
| **Credit is conserved.** `Σφ = v(N) − v(∅)` | error `< 3e-15` over 200 games × 4 permutation counts, for **any** count including 1 — the contributions telescope along a permutation, so efficiency is exact per sample, not in expectation |
| **Null player** | a seat wholly on ABSTAIN gets `φ` of **exactly** `0.0`, so it takes exactly no step |
| **Symmetry** | two interchangeable seats, enumerated exhaustively: identical to `< 1e-12` |
| **The estimator is unbiased** | over all `M!` permutations it equals the closed-form subset-weighted sum to `4e-16`; any gap at finite samples is variance, falling as `1/√n` |
| **No gradient crosses a micro** | checksumming every other micro's arrays around a `learn` call: only micro `i` moves |
| **The gradient is right** | all 45 parameters of a micro — weights, biases and the four sine parameters — match finite differences to `1e-10` |
| **Training terminates** | Rosenthal (1973): the training game is an exact potential game. 200 random games, 0 unconverged, 0 cycles, and the solver *asserts* the potential rose on every accepted deviation |
| **Cycles are detected, not iterated into** | rock-paper-scissors reports cycle length **3** — per sweep. Pushed per *move* it reports 6, which is why the detector pushes once per completed sweep |
| **Truthful bidding is dominant** | over a bid sweep, no misreport beats the truth (Vickrey 1961) |
| **Inversion is an involution** | twice is bit-identical, and it leaves wealth and reputation alone |
| **Determinism** | the feature hash is FNV-1a, not `hash()`; identical output across a subprocess boundary with different `PYTHONHASHSEED` |

## A trie over what is provable

`gtmnn/trie.py` indexes the game catalogue by **game-theoretic structure**,
general at the root and specific at the leaves. The top level does nothing: it
asserts no property and guarantees only what holds of every finite game. Each
level down adds exactly one fact.

```
.                             regret_plus     every finite game
  players=n                   regret_plus
    payoff=common             regret_plus
      structure=congestion    regret_plus
        monotone=non-incr.    regret_plus
          potential=exact     best_response   <- correctness_congestion
    payoff=player-specific    regret_plus
      structure=congestion    regret_plus
        monotone=non-incr.    regret_plus
          potential=none      regret_plus     <- belief_congestion
  players=2                   regret_plus
    payoff=opposed            fictitious
      structure=matrix        fictitious
        ...
          potential=none      fictitious      <- rock_paper_scissors
    payoff=common             regret_plus
      structure=matrix        regret_plus
        ...
          potential=ordinal   best_response   <- stag_hunt, prisoners_dilemma
```

**A node's guarantees are a function of its path and nothing else.** Descending
is accumulating premises, and the theorems that fire at a node are exactly those
whose hypotheses the prefix satisfies:

| theorem | starts holding at | for |
|---|---|---|
| Nash 1950 | depth 0 — the root | everything |
| von Neumann 1928 | depth 2 (`players=2`, `payoff=opposed`) | rock-paper-scissors |
| Milchtaich 1996 | depth 4, the moment `monotone` is asserted | belief congestion |
| Rosenthal 1973 | depth 5, needing the whole congestion prefix | correctness congestion |

That is the trie half, the same one `GREN/gren/radix.py` takes: a terminal flag
on any node, so an internal node is a real answer — "a two-player zero-sum game"
already tells you fictitious play converges, and you can act on it before
reaching a leaf. There is deliberately **no** path compression here: GREN
compresses because its signatures are long and only a few tokens discriminate,
whereas these paths are five deep and every level discriminates, so compression
would buy nothing and would cost the property that depth equals facts asserted.

Each game *declares* its own properties (`zero_sum`, `player_specific`,
`monotone_in_load`, `ordinal_potential`), so a new game joins the index by
answering the same questions and nothing in the trie changes.

### Every solver against every other

The trie says which solver is *guaranteed* to work in each class. That is
falsifiable, so `cli trie` falsifies it — all five solvers, same games, seed for
seed. Mean **mixed** exploitability, 0 = the mixed profile is unexploitable:

| game class | best_response | fictitious | regret | regret_plus | replicator |
|---|---|---|---|---|---|
| correctness congestion | **0.0000** | 0.0000 | 0.0339 | 0.0365 | 0.4846 |
| belief congestion | **0.0000** | 0.0006 | 0.0292 | 0.0298 | 0.2220 |
| inverted correctness | **0.0000** | 0.0000 | 0.0877 | 0.0896 | 0.1422 |
| rock-paper-scissors | 2.0000 | **0.1855** | 0.3730 | 0.2472 | 0.3022 |
| stag hunt | **0.0000** | 0.0000 | 0.3350 | 0.3452 | 0.0717 |
| prisoner's dilemma | **0.0000** | 0.0000 | 0.0390 | 0.0390 | 0.1653 |

**The mixed measure is the point.** The pure one — exploitability of the
profile's argmax — is meaningless wherever the equilibrium is mixed: on
rock-paper-scissors *every* pure profile is exploitable (1.0 matched, 2.0
mismatched), so it scored all five solvers identically at their worst and hid
the fact that the uniform mixture is exactly unexploitable. I measured the wrong
thing first.

**5 of 6 classes vindicate the trie's recommendation.** The exception is
belief congestion, where the trie says `regret_plus` and `best_response`
measures better. Both are right: Milchtaich guarantees a pure equilibrium exists
and warns that an *arbitrary* best-response path may cycle — a worst-case claim.
Hunting for that cycle across seven geometries and **1 400 games found none**;
best response converged every time, in about two sweeps, and 20× faster. The
trie's job is to say which choice is provably safe, not to predict the mean.

### The solver is now derived, not configured

`TrainConfig.solver` defaults to `"auto"`, which asks the trie about the payoff
in hand: `best_response` for the training game (exact potential, Rosenthal),
`regret_plus` at inference (player-specific, Milchtaich), `fictitious` on a
two-player zero-sum game. An explicit request still wins.

| | belief loss | sec/epoch |
|---|---|---|
| `solver="auto"` | 3.153 ± 0.213 | **0.19** |
| `solver="regret_plus"` | 3.245 ± 0.315 | 0.35 |

**1.8× faster at no cost in loss** — the loss gap sits inside the seed spread, so
only the speed is a real difference. That is what makes the trie load-bearing
rather than a diagram: a new payoff gets the right solver by answering the same
questions, instead of someone remembering to add a branch.

### A network at every node

Reaching a node is how you find the network to ask. Each node of the trie may own
a population, and the trie itself still only *classifies* — it holds no
activations and computes no prediction. It says which population answers; the
population answers.

The mechanism that earns the terminal-flag-on-internal-nodes design is
**backoff**. A query walks as deep as the facts allow, then climbs back to the
nearest ancestor whose population has actually played:

```
correctness_congestion: walks to d5, that node is cold, so it backs off to d3
   players=n/payoff=common/structure=congestion
once the leaf has played: answers at d5, backed_off=False
```

"A congestion game with a common payoff" is a usable answer while "…with an exact
potential and non-increasing load" is still cold — a general class is a strict
prefix of a specific one, and the prefix is queryable on its own.

**The routing is verified; what backoff is worth here is not.** `resolve()` picks
the deepest trained node, backs off when the leaf is cold, and stops backing off
the moment the leaf has played — in every seed. But measured over 5 seeds,
backing off to the trained general class beats querying the cold leaf by
**+0.006**, with a per-seed swing of ±0.15. That is nothing. A single seed had
shown +0.22 and, as with the game modifier, it was noise.

The reason is not the mechanism: the class it backs off to has barely learned
either, which is the same weak-credit-path result this README opens with. Backoff
is worth measuring again once that is fixed, and not before.

### Scope

A trie belongs to game theory and game classification and to nothing else here.
There are exactly two in the repository:

| | indexes by | answers |
|---|---|---|
| `GREN/gren/radix.py` | what a game refuses | which game *is* this |
| `GTMNN/gtmnn/trie.py` | its game-theoretic structure | what may I *assume* about it |

Neither touches prediction, features, micro weights or credit. The only hook into
model code is `solver_for`, which classifies the payoff in hand to pick a solver.
The moment a trie starts carrying activations or standing in for a network it
stops being an index and the guarantees stop meaning anything.

## Two losses, and why both are reported

The epoch record carries `loss` **and** `belief_loss`, and they are not the same
thing:

* `loss` is the aggregate of the **training-game** equilibrium. That game is
  `CorrectnessCongestion`, whose payoff function *contains `y`* — the seats are
  being paid for naming the answer they were shown. It is the right partner for
  `shapley_total` (the two are views of `v(N) − v(∅)`) and it is **not** a
  prediction.
* `belief_loss` is the population's own beliefs, reputation-weighted, with no
  game solved at all. That is what the model would actually predict.

They diverge enormously. On a two-text corpus: `loss` **0.59** while the
population's predictive loss was **4.2**, against a uniform baseline of **2.77**.
Reading the first as the second is the mistake the pair exists to prevent — I
made it first, and the honest number is the one that matters.

## The credit path learns — after three corrections to how I measured it

The first version of this section said the population barely learns to predict.
That was wrong, and it was wrong because of the comparison rather than the thing
compared — the third time in this repository a control has decided an outcome.

Real inference loss (`y` withheld, `BeliefCongestion`, solver, aggregate), against
a uniform baseline of **2.708**. Geometry: `n = 64`, `M = 64` — every micro seated —
`R = 16`, three seeds. At the CLI's default 24 seats of 192 the same trend holds
but sits above uniform at short budgets, because only an eighth of the population
plays each game; that is the seats-per-game effect in the next section.

| condition | 6 epochs | 20 epochs | 60 epochs |
|---|---|---|---|
| Shapley credit, lr 0.05 (the old default) | 3.445 | 3.167 | 2.781 |
| **Shapley credit, lr 0.35** | **2.691** | **2.631** | **2.353** |
| same micros, direct supervised signal | 2.708 | 2.375 | 0.905 |

**It learns.** Below uniform at every budget, improving monotonically. The
credit path was never broken; its coefficient was mis-scaled. `φ` is a
*marginal contribution*, and a seat's mean `|φ|` is about 0.15, so at
`lr = 0.05` the population was learning roughly 7× slower than the same micros
under a unit supervised signal. `TrainConfig.lr` is now 0.35.

### The three corrections

1. **"Supervised reaches 1.50 where the credit path reaches 2.84."** Those were
   measured at 30–100 epochs and 4–6 epochs respectively. At matched 6 epochs
   the old credit path scores 2.886 against supervised's 2.820 — within 0.07.
   I compared two different budgets and called the gap architectural.
2. **"More training doesn't help."** That was measured on code that still had the
   empty-slot bug and argmax targets. On current code, loss falls monotonically
   through 60 epochs.
3. **"The abstain trap isn't real."** It is. Under Shapley credit nothing raises
   `q_abstain`, so it never triggers — bidders actually *rise* from 16 to 54 of
   64 over training. But give the population a signal that teaches abstention
   and it fires immediately (next section).

Two hypotheses tested and refuted on the way: Monte-Carlo noise in `φ` (SNR 3.3
at 8 permutations, zero sign-flips) and the `B`/`λ` ratio (already ruled out).
The gradient code was re-derived line by line and matches finite differences
across all 45 parameters of a micro; there was no bug in it.

### What is still not fixed, and the attempt that made it worse

Supervised reaches **0.905** at 60 epochs. The credit path's 2.353 is a real gap,
and the reason is structural: only *holders* — seats that hold `y` — receive a
non-zero `φ`. That is about 23% of seat-games. A non-holder abstains, is a null
player, gets exactly `φ = 0`, and is never taught that abstaining was *right*.
Measured: `q_abstain` **falls** over training (0.201 → 0.163), because the only
signal a micro ever gets is "push toward `y`" in the contexts where it holds `y`,
and normalisation drags ABSTAIN down with everything else.

`credit="regret"` supplies the missing term the principled way — counterfactual
regret (Hart & Mas-Colell, already the inference solver) with Shapley as the
payoff: advantage = `φ(what I did) − marginal(what I could have done)`. A
non-holder's advantage becomes `0 − marginal(playing its best wrong symbol) > 0`,
which pushes it toward ABSTAIN; a holder's stays `φ`. It does exactly what it was
designed to do — `q_abstain` rises to 0.33 — **and collapses the inference game to
uniform: `2.708 ± 0.00` at every budget.** Abstainers are 77% of seats and take
76% of the gradient; as they learn silence their bids fall under the reserve,
nobody is seated, the belief game plays short-handed to no one, and the aggregate
is the ε-smoothing. `belief_loss` frozen at 3.182 across 6, 20 and 60 epochs says
the pool stopped changing at all — that is the abstain trap, closed.

So the open question is sharper than it was: **how do you teach a micro to be
quiet without it going silent for good?** The `regret` option is kept, default
off, so the collapse stays reproducible.

## The GREN handoff: one modifier, three projects

`gtmnn/games.py` implements `GREN/DESIGN.md` §22.4–22.7. A micro's input is

```
F_state  hashed features of the position
F_cand   hashed features of the ONE candidate being scored
F_mod    the game modifier — the mechanic set, hashed
```

so the action-space size (chess ~4000 moves, go 361) never touches the geometry:
the micro is asked "this one: legal?", not "which of the 4000". The mechanic sets
come from **GREN's handoff** — the same `gren_packages.json` CyclicCortex reads —
so all three projects agree on what a game *is* by construction rather than by
three hand-written copies of one frozenset.

Hashing a *set* preserves overlap, so modifier cosine should track the Ochiai
similarity of the mechanic sets. It does:

| pair | Ochiai | cos@64 | cos@128 | cos@256 |
|---|---|---|---|---|
| chess / checkers | 0.775 | 0.827 | 0.791 | 0.791 |
| chess / go | 0.603 | 0.707 | 0.640 | 0.640 |
| checkers / sudoku | 0.215 | 0.336 | 0.215 | 0.215 |
| **mean abs. error** | | **0.102** | **0.031** | **0.017** |

`F_mod = 128` is the right size, matching the spec's own table. Chess is nearest
checkers and sudoku is furthest — the same ordering GREN found by probing and
CyclicCortex uses to build its regions, arrived at independently through a hash.

**But the modifier does not buy transfer.** Train on chess only, evaluate
checkers with no further training, 6 seeds:

| condition | chess acc | checkers acc |
|---|---|---|
| untrained | 0.517 ± 0.030 | 0.483 ± 0.037 |
| chess-trained, modifier **ON** | 0.551 ± 0.062 | 0.503 ± 0.062 |
| chess-trained, modifier **OFF** | 0.553 ± 0.036 | **0.566 ± 0.032** |

The modifier is worth **−0.062**, negative in 5 of 6 seeds. A single seed had
shown +0.117 and that was noise; six seeds reverse the sign.

The explanation that fits: the modifier is a *game-identity* signal, and identity
is exactly what lets a population specialise per game — the opposite of
transferring. §22.5 argues that similar games have similar modifiers so responses
carry over. At this scale the identity effect dominates the similarity effect.

This is the third independent measurement in this repository putting
chess→checkers transfer at roughly nothing: CyclicCortex's shared vocabulary
(+0.023 against its proper baseline), its cross-region ensemble (+0.025), and now
this.

## What is not here yet

`api.py` and `frontend/` (no HTTP surface; CyclicCortex and GREN have none
either). `TorchBackend` is specified in §15 and deliberately **not** implemented:
torch is not a dependency of this repository and an unexercised vectorised path
is a performance claim with nothing behind it — `get_backend("torch")` says so
rather than silently returning the slow one. `bench.py` is not written; the
realised spread of pre-activations (open question 6) is therefore still
unmeasured, and it is the cheapest remaining check on whether the periodic
activation is doing anything a monotone one would not.
