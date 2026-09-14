# Neural compression — does split-and-selector work?

**The scheme.** One network holds many games. Each game occupies a band of the
output range — one game takes `[0,1]`, two take `[0,0.5]` and `[0.5,1]`, four
take a quarter each, `N` take `1/N` each — and a **selector** picks the band.
On query, the output is expanded back to the game's own `0-1` range.

**Tested, not argued.** `experiment.py`, pure standard library, deterministic.
Sixteen synthetic games on a smooth curve in parameter space, so index order is
similarity order; a 16-hidden-unit MLP; test MSE always rescaled back to each
game's own `[0,1]` before comparison.

## Summary

| # | Question | Answer |
|---|---|---|
| 1 | Does one network hold many games? | **Yes, and it beats separate networks** — 3.2× fewer parameters *and* lower error at N=16. |
| 2 | Does the output partition do the compressing? | **No.** Plain conditioning is 7.5× more accurate at the same scale. The partition is the wrong mechanism for this job. |
| 3 | Does band ordering by similarity matter? | **Decisively, and it grows with N.** At N=16 ordered works and shuffled either diverges or is 10× worse. |
| 4 | Is there something the partition wins at? | **Yes — games it has never played.** It has the best zero-shot score of any condition, and one-hot conditioning cannot do this at all. |
| 5 | (Incidental) Is the sine activation's `b=1/3` right? | **No.** At `b=1/3` with normalised inputs it is 17× *worse* than tanh; at `b=1-2` it is 6-24× *better*. See §5 — this is a bug in two existing specs. |

## Two traps that would have produced a false result

Recorded because either one alone reverses the conclusion:

1. **A partitioned output has targets in a band of width `1/N`, so its raw MSE is automatically `N²` smaller.** Reported naively, the scheme looks brilliant and the advantage is pure arithmetic. Every number here is rescaled to the game's own `[0,1]` first (`yhat = o·N − band`).
2. **That same narrow band makes the gradient `N` times smaller**, so the partition trains `N` times slower at a shared learning rate and reports a *false negative*. Loss is therefore measured in game units (`gain=N`) for every condition. The first run of this experiment did not do this and materially understated the partition — the ordered/shuffled gap at N=16 only appeared after the fix. Gradient clipping at 5.0 (per `GTMNN/DESIGN.md` §7.4) is then required; without it the amplified gradient diverges.

## 1-2. Compression works; the partition is not what does it

Test MSE, rescaled, tanh:

| N | partition (ordered) | partition (shuffled) | scalar selector | **one-hot selector** | separate nets |
|---|---|---|---|---|---|
| 2 | 0.0311 | 0.0311 | 0.0267 | 0.0325 | 0.0278 |
| 4 | 0.0461 | 0.0767 | 0.0722 | **0.0103** | 0.0104 |
| 8 | 0.0553 | 0.1388 | 0.0202 | **0.0059** | 0.0064 |
| 16 | 0.0607 | *diverged* | 0.0741 | **0.0081** | 0.0101 |
| | 81 params | 81 | 81 | 321 | 1040 |

**The compression claim is confirmed.** At N=16 one network carries all sixteen
games at **321 parameters and 0.0081 error**, against sixteen separate networks
at **1040 parameters and 0.0101 error**. Fewer parameters *and* better accuracy —
because related games share structure, and sharing a trunk lets each game's data
inform the others.

**The partition is not the mechanism.** `partition_ordered` (0.0607) is 7.5×
worse than one-hot conditioning (0.0081). Note the slopes, which matter more than
any single row: as games are added, **the partition degrades** (0.031 → 0.046 →
0.055 → 0.061) while **conditioning improves** (0.033 → 0.010 → 0.006 → 0.008).
Opposite directions. Adding a game to a partitioned network costs the others;
adding a game to a conditioned network helps them.

The reason is precision. Rescaling multiplies output error by `N`, so holding
per-game accuracy fixed demands the raw output get `N` times more precise for
every game added. That is the opposite of compression — it asks more of the same
capacity rather than less.

`partition_onehot` — one-hot selector *and* partitioned output — was the worst
condition tested (0.218 at N=16). It pays the `N`× error amplification and gets
no smoothness in return, which is the clue to §3.

## 3. Ordering the bands by similarity is mandatory

| N | ordered | shuffled | ratio |
|---|---|---|---|
| 2 | 0.0311 | 0.0311 | 1.0× |
| 4 | 0.0461 | 0.0767 | 1.7× |
| 8 | 0.0553 | 0.1388 | 2.5× |
| 16 (tanh) | 0.0607 | *diverged* | — |
| 16 (sine b=1) | 0.0862 | 0.8342 | **9.7×** |

The effect is absent at N=2 and total at N=16, and the mechanism is clear:
**with bands ordered by similarity the target is a smooth function of the
selector coordinate; shuffled, it is a high-frequency sawtooth.** A small network
fits the first and cannot fit the second, and the `gain=N` amplification then
turns the misfit into divergence.

So the intuition that a game's position is a *plane on a continuum* is load
bearing, not decorative. The ordering has to come from somewhere real — a 1-D
projection of `GREN`'s mechanic-similarity space (`GREN/DESIGN.md` §21) is the
natural source, and it is already computed there.

## 4. What the partition actually wins: games never played

Train on even-indexed games only; evaluate on the odd-indexed ones the network
has never seen. N=16.

| activation | condition | seen | **UNSEEN** | degradation |
|---|---|---|---|---|
| tanh | **partition (ordered)** | 0.0707 | **0.0613** | **none** |
| tanh | scalar selector | 0.0771 | 0.0830 | 1.08× |
| tanh | one-hot selector | **0.0045** | 0.1185 | **26×** |
| sine b=1 | **partition (ordered)** | 0.0949 | **0.0801** | **none** |
| sine b=1 | scalar selector | 0.0750 | 0.0837 | 1.12× |
| sine b=1 | one-hot selector | 0.0326 | 0.1224 | 3.8× |

**This is the result worth keeping.** One-hot conditioning memorises known games
brilliantly (0.0045) and collapses on an unseen one (0.1185) — it structurally
cannot do better, because an unseen game has no slot to put a 1 in. A continuous
selector coordinate **interpolates**: it degrades by 8%, and the partitioned
version does not degrade at all, scoring the best zero-shot number of any
condition tested.

So the scheme's value is not compression — conditioning compresses better. Its
value is that a **continuous game coordinate supports playing a game you have
never played**, by placing it between games you have. For an AGI/ASI architecture
that is the more important property of the two, and it is the one that a one-hot
or per-game-head design cannot buy at any parameter budget.

**The recommendation follows from the two halves.** Use a continuous selector, of
a few dimensions rather than one, ordered by real similarity — which is exactly
`GTMNN`'s game modifier (`GTMNN/DESIGN.md` §22.5, the hashed mechanic set). Keep
the **full output range** rather than partitioning it: the partition's
interpolation advantage comes from the *continuous selector*, which the modifier
already provides, while its `N`× error amplification is a pure cost. The
experiment supports the modifier design and argues against carrying the output
partition into it.

## 5. Incidental finding: the sine activation's `b = 1/3` is wrong

The task family is `0.5 + 0.4·sin(a·x₀ + b·x₁ + c)` — a sine of a linear
combination, exactly what one sine-activated hidden unit computes. Sine should
dominate here. It did not, and a learning-rate sweep showed it *flat* from
lr=0.02 to lr=1.0 (0.0179 → 0.0179), which means stuck, not undertrained.

Sweeping the frequency `b` in `f(x) = a·sin(b(x−h)) + k`:

| `b` | \|z\| to first peak | sine MSE |
|---|---|---|
| **1/3 (as specified)** | 4.71 | **0.01674** |
| 1.0 | 1.57 | **0.00015** |
| 2.0 | 0.79 | **0.00004** |
| 3.0 | 0.52 | diverges |
| 5.0 | 0.31 | diverges |
| *tanh reference* | — | *0.00098* |

**At the specified `b = 1/3` the sine is 17× worse than tanh. At `b = 1-2` it is
6-24× better.** The activation is excellent; the frequency is wrong.

The mechanism: reaching the first peak of `sin(b·z)` needs `|z| = π/(2b)`, which
at `b = 1/3` is **4.71**. With L2-normalised inputs (`|x| ≈ 1`) and small
initial weights, pre-activations sit near zero — so every unit operates in the
**linear** region of the sine, and a network of near-linear units is a linear
network regardless of its width.

This is a live bug in two specs that pair `b = 1/3` with normalised features:

* `GTMNN/DESIGN.md` §5 states the normalisation "keeps the sine in its informative range... where `-sin(x/3)` is steepest and **most nearly linear**". That reasoning is backwards — "most nearly linear" is the failure, not the goal.
* `RadixCyclicNN/radixnet/activation.py` uses the same default. It is a working, merged system on a different task, so the finding is flagged rather than applied there; whether it bites depends on that project's pre-activation scale, which is worth measuring before changing anything.

`b` is learnable per neuron in both designs, so in principle a network could
escape. In practice it does not: `∂f/∂b = a(x−h)cos(b(x−h))` is small precisely
when `x` is small, so the gradient that would fix the frequency is suppressed by
the same condition that makes it wrong. **The initialisation is the trap, not the
parameterisation.**

The fix is one number: initialise `b` so that `π/(2b)` lands inside the actual
pre-activation range. For L2-normalised inputs that is `b ≈ 1`, not `1/3`. The
general rule is `b ≈ π / (2·E|z|)`.

## Scope

One synthetic task family, one small architecture, single seeds, N ≤ 16. The
compression and ordering results are large enough (up to 10×, and divergence) to
be robust to those limits. §5's mechanism is confirmed directly by the `b` sweep
rather than inferred. What is *not* established: whether any of this holds on
real games rather than smooth synthetic ones, and whether the interpolation of §4
survives when neighbouring games are genuinely discrete rather than samples from
a continuum. §4 is the result most likely to weaken on real data and the one most
worth re-running there first.

---

# Part 2 — how should the range be partitioned?

If games occupy bands, the layout is a design choice: equal `1/N` bands that are
re-laid-out whenever a game is added, or a **static** sequence (powers of two,
powers of ten) that reserves slots so existing games never move. `partition.py`.

## 6. Reserved slots are expensive: dust is error amplification

4 games, fixed-width bands, only the width differing. Seeds 0-3, tanh:

| layout | dust | amplification | mean MSE | vs. no dust |
|---|---|---|---|---|
| 4 bands of 1/4 | 0% | ×4 | **0.0401** | — |
| 8 bands of 1/8 | 50% | ×8 | 0.1248 | **3.1× worse** |
| 16 bands of 1/16 | 75% | ×16 | 0.1814 | **4.5× worse** |

**Dust is not wasted space — it is error amplification.** Reading a band back out
multiplies output error by the number of slots, so reserving slots for games that
are not there degrades the games that *are* there, in direct proportion. Fifty
percent dust costs 3.1×.

This settles the binary-sequence question against it. Padding to the next power
of two wastes up to 47% just past a boundary — 17 games in 32 slots — which by
this table is roughly a 3× accuracy penalty for nothing. **Use exactly `N` bands.
Never reserve.**

## 7. Re-laying-out is cheap — the thing reserving was meant to avoid

The case for a static layout was that `1/N` moves every band when a game is
added, destroying what was learned. Tested directly: train 4 games, then train
**only** the 4 new ones, then re-measure the original 4. Seeds 0-3, tanh:

| layout | phase-1 error | damage to the original 4 |
|---|---|---|
| `1/N`, re-laid-out | 0.0401 | **1.37×** |
| reserved slots, nothing moves | 0.0397 | **1.36×** |
| midpoint insertion (adaptive) | 0.0401 | 2.36× |

**Displacement barely matters.** Re-laying every band out costs 1.37×, and
freezing them costs 1.36× — indistinguishable. Most of that 1.37× is ordinary
catastrophic forgetting from training on new games only, which happens whatever
the layout does.

The reason is §4: the selector is *continuous*. Rescaling `s` is a smooth
reparameterisation the network already generalises across, not a permutation of
discrete slots. A network that can interpolate to an unseen game can also absorb
its coordinate system being stretched.

**So the two costs are 3.1× against 1.37×, and they point the same way:** take
the relayout, refuse the dust.

Midpoint insertion — placing a new game between its two nearest neighbours so
existing coordinates never move and bands still tile `[0,1]` — was my own
proposal for getting static, dust-free and similarity-ordered at once. It is the
**worst** option tested at 2.36×, because it produces bands of unequal and
irregular width, and the network must then learn a non-uniform map from selector
coordinate to output scale on top of everything else. Uniform bands that move
beat non-uniform bands that stay.

## 8. Predictions this section got wrong

Recorded because the corrections are the content:

* **"Displacement will cause catastrophic forgetting."** It does not — 1.37× against 1.36×. The continuous selector absorbs it (§7).
* **"Reserved slots cost ~14% at phase 1."** Single-seed noise. Across four seeds it is 0.0397 vs 0.0401 — no difference. The real cost of dust appears only when bands are genuinely held at fixed width (§6), which the first attempt did not do: its Voronoi band construction re-tiled `[0,1]` automatically and so removed the dust it was meant to be measuring.
* **"Midpoint insertion gets all three properties at once."** It gets them and is still worse, for a reason none of the three properties describes (§7).
* An earlier incremental test (train 4, then retrain on *all* 8) measured nothing at all: the original games improved from the extra training, which masks displacement entirely. Phase 2 must train only the new games.

## 9. What to build

**Exactly `N` equal bands, ordered by similarity, re-laid-out when a game is
added.** No padding, no reserved slots, no variable widths.

And the conclusion of Part 1 stands above all of it: the partition is worth
having only for the **continuous selector coordinate** that comes with it, which
is what buys interpolation to unseen games (§4). If that coordinate is supplied
some other way — as `GTMNN`'s game modifier is (`GTMNN/DESIGN.md` §22.5) — then
the full output range is better than any partition of it, and every number in
Part 2 becomes moot.
