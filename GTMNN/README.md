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
python3 -m gtmnn.cli transfer    # does chess transfer to checkers?
python3 -m gtmnn.cli evolve      # cull, birth, evolutionary stability
python3 -m tests.test_gtmnn      # 36 tests
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

## What does not hold: the population barely learns to predict

This is the headline negative result and it is the thing to fix next.

| | belief loss | vs uniform 2.71 |
|---|---|---|
| untrained | 3.23 | worse |
| full game path — auction, equilibrium, Shapley, REINFORCE | **2.84** | still worse |
| **same architecture, direct supervised signal** | **1.50** | **far better** |

The supervised control is the important row. It uses the *identical* micros,
sine activation, feature hashing and aggregate, and simply tells each micro "if
you hold `y`, say it; if you do not, abstain". It reaches 1.14–2.47 across
geometries, comfortably below the baseline. So the architecture has ample
capacity for the task and **the credit path is what fails to deliver a signal
that sharpens the population**.

More training does not close it: at 6 900 learning steps per micro the belief
loss moved 3.226 → 3.204, and 150 epochs did *worse* than 20. It is not a signal
shortage.

Two candidate causes were tested and **both were wrong**, which is worth
recording so they are not re-tried from intuition:

* *B against λ* (`DESIGN.md` open question 1). Swept from 2 to ∞. Abstention
  falls from 56% to 0% as the ratio rises and the inference loss does not
  improve at any setting — 3.42 at best, still above uniform. The ratio is real
  but it is not the binding constraint.
* *The null-player axiom forbids rewarding a correct silence.* A micro that
  correctly abstains contributed nothing, so Shapley scores it exactly zero and
  it never learns to stay quiet. Deliberately breaking the axiom to reward it
  made things **worse** (2.838 → 3.050), so this is not it either.

What the population *does* do is specialise: `gini` rises to 0.54 when every
micro gets seated, against 0.05 when only 24 of 256 do. That is `DESIGN.md` open
question 5 — "does the population actually specialise?" — answered yes, and it
makes seat count the first thing to look at.

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
