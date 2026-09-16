# GTMNN

**G**ame **T**heory + **M**icro **N**eural **N**etworks — part two of a two-part
architecture. `GREN/` works out which game is being played; GTMNN plays it.

There is no network here. There is a **population of thousands of tiny neural
networks** and a **game they play against each other**, and the answer the
system gives is the **equilibrium of that game**. Credit for a right or wrong
answer is not propagated backwards through layers — it is divided among the
players by **Shapley value**, the game-theoretic answer to "who actually
contributed". Players that keep losing money are culled and replaced by mutated
winners.

**Back-propagation is not used anywhere.** Not as an approximation of it, not as
a special case of it — no chain rule ever spans two micros. The deepest gradient
in the system is two layers long, inside one micro, over roughly 133 floats.

## Contents

| file | what it is |
|---|---|
| `DESIGN.md` | the full specification — the contract every module is implemented against, in 27 sections |

**There is no code in this directory yet.** `DESIGN.md` is the whole of it: a
spec written to be implemented against, not notes. Read it fully before writing
anything here.

## What the spec covers

| § | what it settles |
|---|---|
| 1 | every requirement in the author's own words, mapped to the section that realises it |
| 2 | why a game — Shapley (1953) proved there is *exactly one* fair credit assignment, and that uniqueness is what makes the design possible |
| 5-7 | `activation.py` (a learnable per-neuron sine, initialised to `-sin(x/3)`), `features.py`, `micro.py` — the `MicroPool` of `N=4096` micros stored as flat parallel arrays, each an `8 → 6 → (4+1)` MLP seeing only its own receptive field |
| 8-11 | the game catalogue (congestion, second-price auction, stag hunt, prisoner's dilemma, rock-paper-scissors), the seat auction, the stage game, and solving it |
| 12-13 | `shapley.py` — Monte-Carlo credit assignment — and coalitions |
| 14 | `meta.py` — metacognition: a cycle or a too-close equilibrium escalates to a second micro population playing over the *statistics of the deliberation that just failed* |
| 15-17 | backends (pure Python, optional batched torch), `GTMNet`, and the `Evolver` with its evolutionary-stability check |
| 18-23 | checkpointing, bench, CLI, API, frontend, tests |
| 24-27 | performance rules, cross-cutting invariants, what is deliberately absent, open questions |

## The two mechanisms worth knowing about

**Specialisation is free.** The core game is a *congestion game*: the reward for
a correct answer is split among everyone who gave it. A thousand micros all
saying `"e"` earn a thousandth each; one micro that correctly calls something
rare takes the whole pot. Nothing else in the design pushes micros apart.

**Calibration is a mechanism, not a loss term.** Seats are sold in a
uniform-price auction — each seated micro pays the highest losing bid — so
bidding your true confidence is a dominant strategy.

## Planned shape

Python package `gtmnn`, Python 3.11+, **standard library only**; `torch` is an
optional accelerator, imported lazily and never required. `train / predict /
generate / score` — the same four verbs as `RadixNet`, so the CLI, API and
frontend stay recognisable siblings of `RadixCyclicNN/`.

## Related

* `GREN/DESIGN.md` — part one. GTMNN refuses a game package below a confidence
  threshold (§22.1), so "understand the game, then play it" is enforced.
* `Research/SineWaveActivationFunction.md` — the activation every micro carries.
* `Research/2NRL.md` — `two_nrl(bad, good)`: train on garbage, invert the
  population (`a → -a` on every sine *and* a sign flip of the payoff function,
  so the population plays the anti-game for one pass), then fine-tune.
* `Experiments/NeuralCompression/FINDINGS.md` §5 — measured evidence that this
  spec's `b = 1/3` initialisation is the wrong value.
