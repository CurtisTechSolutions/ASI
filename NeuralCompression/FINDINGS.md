# Neural compression — can one network hold many games?

## Read this first

**The idea.** Normally you train one network per task. This tests the opposite:
one network that plays *many* games. You tell it which game to play through an
extra input — a **selector** — and optionally give each game its own slice of
the output range — a **band** — so game 3 answers somewhere in `[0.1875, 0.25]`
instead of the full `[0, 1]`. On query you stretch the band back out to the
game's own `0–1` scale.

**What came out of it, in one line each:**

1. **One network really does hold sixteen games**, using a third of the
   parameters of sixteen separate networks and making *fewer* errors — though
   that particular win holds for tanh and not for the sine activation. (§1)
2. **The bands are not what makes that work.** Simply telling the network which
   game to play works 7.5× better than slicing up the output. (§2)
3. **What the bands are good for is games the network has never played.**
   Because the selector is a *number* on a line rather than a *label*, an unseen
   game can sit between two seen ones. No other setup tested can do this. (§4)
4. **The sine activation's frequency `b = 1/3` is wrong** — 17× worse than tanh
   where it should have won. It is a live bug in two other specs in this
   repository, and it is *also* the cause of the vanishing gradient that Part 3
   was written to work around. (§5, §10)
5. **None of the escape mechanisms help.** Every way of "inverting out of a
   stall" that was tested either does nothing at all or destroys the network.
   The stall itself goes away when `b` is fixed. (§11)

**How to read the numbers.** Every figure is a test-set mean squared error,
always rescaled back to the game's own `0–1` range so conditions are comparable.
Lower is better. For scale: predicting a flat 0.5 for every input scores
**0.081** (measured over 2000 test points per game), so any number near or above
that means the network has learned nothing useful.

## The words used here

| term | meaning |
|---|---|
| **game** | one synthetic task, `0.5 + 0.4·sin(a·x₀ + b·x₁ + c)`. Sixteen of them sit on a smooth curve through `(a,b,c)` space, so game 5 really is more like game 6 than like game 12. |
| **selector** | the extra input that says which game to play. Either a **scalar** (one number, `(k+0.5)/N`) or **one-hot** (N inputs, a single 1 marking the game). |
| **band** | the slice of output range given to one game. With N games each band is `1/N` wide. |
| **partition** | the scheme that uses bands. "Full range" means no bands — every game answers in `[0, 1]`. |
| **dust** | band slots reserved for games that do not exist yet. 8 slots holding 4 games is 50% dust. |
| **specialist / query network** | Part 3's split: a small network that learns one game alone, and the big shared network that answers everything. |

## How to reproduce every number

```bash
python3 experiment.py     # §1-4 (experiments 1-2) and §5 (experiment 3)
python3 partition.py      # §6 (experiment 6) and §7 (experiment 5)
python3 vanishing.py      # §10 (part A) and §11 (part B)
python3 consolidate.py    # §12
```

Pure standard library, no dependencies, deterministic under fixed seeds. Part 4
(§15–18) is the one exception: `Rotating` lives in `vanishing.py` but is not in
its default run, so those numbers come from ad-hoc calls.

## Two traps that would have produced a false result

Recorded because either one alone reverses the conclusion:

1. **A banded output is automatically flattered.** Its targets live in a slice
   `1/N` as wide, so its raw error is `N²` smaller for free. Reported naively,
   the scheme looks brilliant and the whole advantage is arithmetic. Every
   number here is rescaled to the game's own `[0,1]` first (`yhat = o·N − band`).
2. **A banded output is then automatically punished.** That same narrow slice
   makes the gradient `N` times smaller, so at a shared learning rate the
   partition trains `N` times slower and looks bad for a reason that has nothing
   to do with the idea. So the loss is measured in game units for every
   condition (`gain=N`). The first run of this experiment skipped this and
   materially understated the partition — the ordered-versus-shuffled gap at
   N=16 only appeared once it was fixed. Amplifying the gradient then requires
   clipping at 5.0 (per `GTMNN/DESIGN.md` §7.4), or N=16 diverges.

---

# Part 1 — does one network hold many games?

## 1-2. It works, and the bands are not why

Test error with a tanh activation, rescaled, lower is better:

| N | bands, ordered | bands, shuffled | scalar selector | **one-hot selector** | separate nets |
|---|---|---|---|---|---|
| 2 | 0.0311 | 0.0311 | 0.0267 | 0.0325 | 0.0278 |
| 4 | 0.0461 | 0.0767 | 0.0722 | **0.0103** | 0.0104 |
| 8 | 0.0554 | 0.1388 | 0.0202 | **0.0059** | 0.0064 |
| 16 | 0.0607 | 0.9255 | 0.0741 | **0.0081** | 0.0101 |
| | 81 params | 81 | 81 | 321 | 1040 |

**Compression works.** At N=16 one network carries all sixteen games with
**321 parameters and 0.0081 error**, against sixteen separate networks at
**1040 parameters and 0.0101 error**. That is 3.2× fewer parameters *and* fewer
errors, and the reason is that these games are related: sharing a trunk lets
each game's data teach the others.

**The bands are not the mechanism.** Banding the output (0.0607) is 7.5× worse
than just naming the game (0.0081). The slopes say it more clearly than any
single row. As games are added the banded network gets **worse** (0.031 → 0.046
→ 0.055 → 0.061) while the named-game network gets **better** (0.033 → 0.010 →
0.006 → 0.008). Opposite directions. Adding a game to a banded network costs the
games already in it; adding a game to a conditioned network helps them.

The reason is precision. Stretching a band back out multiplies the output error
by `N`, so keeping per-game accuracy fixed requires the raw output to get `N`
times *more* precise for every game added. That is the opposite of compression:
it asks more of the same capacity rather than less.

Doing both at once — one-hot selector **and** banded output — was the worst
condition tested anywhere (0.2217 at N=16). It pays the `N`× error amplification
and gets none of the smoothness that §3 shows the bands depend on.

**One caveat on the headline.** The 3.2×-fewer-parameters win is measured with
tanh. Under `sine b=1` it does not appear: separate networks beat the shared
one at every N from 4 up (0.0103 vs 0.0151 at N=4, 0.0113 vs 0.0208 at N=8,
0.0121 vs 0.0333 at N=16). The direction of §2 — bands lose to conditioning —
holds under both activations, but "beats separate networks" is a tanh result.

## 3. The bands must be ordered by similarity

| N | ordered | shuffled | ratio |
|---|---|---|---|
| 2 | 0.0311 | 0.0311 | 1.0× |
| 4 | 0.0461 | 0.0767 | 1.7× |
| 8 | 0.0554 | 0.1388 | 2.5× |
| 16 (tanh) | 0.0607 | **0.9255** | **15×** |
| 16 (sine b=1) | 0.0862 | 0.8342 | 9.7× |

The effect is absent at N=2 and total at N=16. At 0.9255 the shuffled network is
not merely worse — it is **11.4× worse than predicting a flat 0.5**, so it has
learned nothing and is actively producing noise (its worst single game scores
1.66).

The mechanism is simple. With bands in similarity order, the target is a *smooth*
function of the selector: nudge the selector and the right answer moves a little.
Shuffled, it is a high-frequency sawtooth: nudge the selector and the right
answer jumps somewhere unrelated. A small network fits the first and cannot fit
the second, and the `gain=N` amplification turns that misfit into a blow-up.

So the intuition that a game's position is **a point on a continuum** is load
bearing, not decorative. The ordering has to come from somewhere real — a 1-D
projection of `GREN`'s mechanic-similarity space (`GREN/DESIGN.md` §21) is the
natural source, and it is already computed there.

## 4. What the bands actually win: games never played

Train on the even-numbered games only. Then test on the odd-numbered ones, which
the network has never seen. N=16.

| activation | condition | seen | **unseen** | degradation |
|---|---|---|---|---|
| tanh | **bands, ordered** | 0.0707 | **0.0613** | **none** |
| tanh | scalar selector | 0.0771 | 0.0830 | 1.08× |
| tanh | one-hot selector | **0.0045** | 0.1185 | **26×** |
| sine b=1 | **bands, ordered** | 0.0949 | **0.0801** | **none** |
| sine b=1 | scalar selector | 0.0750 | 0.0837 | 1.12× |
| sine b=1 | one-hot selector | 0.0326 | 0.1224 | 3.8× |

**This is the result worth keeping.** A one-hot selector memorises the games it
has seen brilliantly (0.0045) and then collapses completely on a new one
(0.1185, which is 1.5× worse than predicting a flat 0.5). It cannot do anything else:
an unseen game has no slot to put its 1 in, so the network is being asked about
an input pattern that never existed.

A selector that is a *number* interpolates instead. Asked about game 7 having
only ever seen 6 and 8, it answers with something between what it knows — and
the numbers say that answer is about as good as the ones it was trained on. The
banded version degrades not at all and posts the best zero-shot score of anything
tested.

So the scheme's value is **not** compression, which conditioning does better.
Its value is that a continuous game coordinate lets you play a game you have
never played, by placing it between games you have. For an AGI/ASI architecture
that is the more important of the two properties, and it is the one a one-hot or
per-game-head design cannot buy at any parameter budget.

**What to build follows from the two halves.** Use a continuous selector, of a
few dimensions rather than one, ordered by real similarity — which is exactly
what `GTMNN`'s game modifier already is (`GTMNN/DESIGN.md` §22.5, the hashed
mechanic set). Keep the **full output range** rather than banding it: the
interpolation advantage comes from the *continuous selector*, which the modifier
already provides, while the `N`× error amplification is a pure cost. The
experiment supports the modifier and argues against carrying the output
partition into it.

## 5. Incidental finding: the sine activation's `b = 1/3` is wrong

Every game here is `0.5 + 0.4·sin(a·x₀ + b·x₁ + c)` — a sine of a linear
combination, which is *exactly* what one sine-activated hidden unit computes. A
sine network should win this outright. It lost. A learning-rate sweep came back
flat from lr=0.02 to lr=1.0 (0.0179 → 0.0179), which means stuck, not
undertrained.

So sweep the frequency `b` in `f(x) = a·sin(b(x−h)) + k` instead:

| `b` | \|z\| needed to reach the first peak | error |
|---|---|---|
| **1/3 (as specified)** | 4.71 | **0.01674** |
| 1.0 | 1.57 | **0.00015** |
| 2.0 | 0.79 | **0.00004** |
| 3.0 | 0.52 | 9.46 (diverged) |
| 5.0 | 0.31 | 31.03 (diverged) |
| *tanh, for reference* | — | *0.00098* |

**At the specified `b = 1/3` the sine is 17× worse than tanh. At `b = 1–2` it is
6–24× better.** The activation is excellent; the frequency is wrong.

The mechanism is in the middle column. To reach the first peak of `sin(b·z)` you
need `|z| = π/(2b)`, which at `b = 1/3` is **4.71**. But inputs are L2-normalised
(`|x| ≈ 1`) and initial weights are small, so pre-activations sit near zero. Every
unit is therefore operating on the almost-straight part of the sine near the
origin — and a network of near-linear units is a linear network no matter how
wide it is.

This is a live bug in two specs that pair `b = 1/3` with normalised features:

* `GTMNN/DESIGN.md` §5 says the normalisation "keeps the sine in its informative
  range... where `-sin(x/3)` is steepest and **most nearly linear**". That
  reasoning is backwards: "most nearly linear" is the failure, not the goal.
* `RadixCyclicNN/radixnet/activation.py` uses the same default. That is a
  working, merged system on a different task, so this is flagged rather than
  applied — whether it bites there depends on that project's pre-activation
  scale, which is worth measuring before changing anything.

`b` is learnable per neuron in both designs, so in principle the network could
escape on its own. In practice it does not: `∂f/∂b = a(x−h)cos(b(x−h))` is small
exactly when `x` is small, so the gradient that would fix the frequency is
suppressed by the very condition that makes it wrong. **The initialisation is the
trap, not the parameterisation.**

The fix is one number: initialise `b` so `π/(2b)` lands inside the pre-activation
range the network actually sees. For L2-normalised inputs that is `b ≈ 1`, not
`1/3`. The general rule is `b ≈ π / (2·E|z|)`.

## Scope and limits

One synthetic family of games, one small architecture, N ≤ 16, and single seeds
for §1–5. The compression and ordering results are large enough (up to 15×, and
a blow-up) to survive those limits. §5's mechanism is confirmed directly by the
`b` sweep rather than inferred from it.

What is **not** established: whether any of this holds for real games rather than
smooth synthetic ones, and whether §4's interpolation survives when neighbouring
games are genuinely discrete instead of samples from a continuum. §4 is both the
most valuable result and the one most likely to weaken on real data, so it is the
one to re-run there first.

---

# Part 2 — if you do use bands, how should they be laid out?

Two choices to settle. When a game is added, do you re-lay-out every band, or
reserve empty slots in advance so nothing ever moves? `partition.py` measures
both costs directly.

## 6. Reserving slots is expensive, because empty slots amplify error

Four games, bands of fixed width, only the width differing. Seeds 0–3, tanh:

| layout | dust | amplification | error | vs. no dust |
|---|---|---|---|---|
| 4 bands of 1/4 | 0% | ×4 | **0.0401** | — |
| 8 bands of 1/8 | 50% | ×8 | 0.1248 | **3.1× worse** |
| 16 bands of 1/16 | 75% | ×16 | 0.1814 | **4.5× worse** |

**Dust is not wasted space — it is error amplification.** Reading a band back out
multiplies the output error by the number of slots, so reserving slots for games
that do not exist degrades the games that do, in direct proportion. Fifty percent
dust costs 3.1×.

That settles the binary-sequence question against it. Padding up to the next
power of two wastes up to 47% just past a boundary — 17 games in 32 slots —
which by this table is roughly a 3× accuracy penalty in exchange for nothing.
**Use exactly `N` bands. Never reserve.**

## 7. Moving the bands is cheap — the very thing reserving was meant to avoid

The case for reserved slots was that `1/N` bands move *every* game when a new one
arrives, destroying what was learned. Tested directly: train 4 games, then train
**only** the 4 new ones, then re-measure the original 4. Seeds 0–3:

| activation | layout | before | after | damage | worst seed |
|---|---|---|---|---|---|
| tanh | `1/N`, everything moves | 0.0401 | 0.0532 | **1.37×** | 2.28× |
| tanh | reserved slots, nothing moves | 0.0397 | 0.0484 | **1.36×** | 2.58× |
| tanh | midpoint insertion (adaptive) | 0.0401 | 0.0915 | **2.36×** | 3.74× |
| sine b=1 | `1/N`, everything moves | 0.0848 | 0.0794 | **0.96×** | 1.24× |
| sine b=1 | reserved slots | 0.0712 | 0.0799 | 1.12× | 1.34× |
| sine b=1 | midpoint insertion | 0.0848 | 0.1041 | 1.22× | 1.37× |

**Moving the bands barely matters.** Re-laying out every band costs 1.37×;
freezing them costs 1.36×. That is the same number. Most of the 1.37× is
ordinary catastrophic forgetting from training on new games only, which happens
whatever the layout does — and under `sine b=1` the moving layout actually comes
out *ahead* of where it started (0.96×).

The reason is §4: the selector is **continuous**. Rescaling it is a smooth
change of coordinates that the network already generalises across, not a
reshuffling of discrete slots. A network that can interpolate to a game it has
never seen can also absorb its coordinate system being stretched.

**So the two costs are 3.1× against 1.37×, and they point the same way:** take
the relayout, refuse the dust.

Midpoint insertion — putting a new game exactly between its two nearest
neighbours, so existing coordinates never move *and* the bands still tile `[0,1]`
with no dust — was my own proposal for getting every property at once. It is the
**worst** option tested, and it is worst on 7 of the 8 individual seed-activation
pairs, so this is not noise. The reason is that it produces bands of unequal and
irregular width, and the network then has to learn a lumpy map from selector
coordinate to output scale on top of everything else. **Uniform bands that move
beat non-uniform bands that stay.**

*Read the ordering, not the exact multiplier.* Per-seed damage ranges from 0.60×
to 3.74×, so a single seed can say almost anything — an earlier single-seed run
of this same experiment reported 1.1×/1.1×/1.5×. The layout *order* is stable
across seeds; the magnitudes are not.

## 8. Predictions this part got wrong

Recorded because the corrections are the content:

* **"Moving the bands will cause catastrophic forgetting."** It does not —
  1.37× against 1.36×. The continuous selector absorbs it (§7).
* **"Reserved slots cost about 14% up front."** Single-seed noise. Across four
  seeds it is 0.0397 against 0.0401 — no difference. The real cost of dust shows
  up only when bands are genuinely held at fixed width (§6), which the first
  attempt failed to do: its Voronoi band construction silently re-tiled `[0,1]`
  and so removed the dust it was supposed to be measuring.
* **"Midpoint insertion gets all three properties at once."** It gets them, and
  it is still the worst option, for a reason none of the three properties
  mentions (§7).
* An earlier version trained on *all eight* games in phase 2 and measured
  nothing at all: the original games improved from the extra training, which
  hides displacement completely. Phase 2 must train only the new games.

## 9. What to build

**Exactly `N` equal bands, ordered by similarity, re-laid-out whenever a game is
added.** No padding, no reserved slots, no variable widths.

And Part 1's conclusion stands above all of it: bands are worth having only for
the **continuous selector coordinate** that comes with them, which is what buys
interpolation to unseen games (§4). If that coordinate arrives some other way —
as `GTMNN`'s game modifier provides it (`GTMNN/DESIGN.md` §22.5) — then the full
output range beats any partition of it, and every number in Part 2 is moot.

---

# Part 3 — the vanishing gradient, accepted then escaped

Two mechanisms were proposed and both were built: flip the update sign when
training stalls (`vanishing.py`), and train a game in a small specialist network
before folding it into the big shared one (`consolidate.py`).

## 10. The vanishing gradient here is just `b = 1/3` again

A deep MLP, `2 → 8 → ... → 1`, measuring the first layer's gradient directly.
Near zero the sine's derivative is `−cos(b·z)·b ≈ −b`, so each layer shrinks the
gradient by about `b` and `depth` layers shrink it by `b^depth`.

| activation | depth | predicted shrink `b^depth` | measured gradient at layer 0 | error |
|---|---|---|---|---|
| sine `b=1/3` | 1 | 0.333 | 9.7e-03 | 0.0079 |
| sine `b=1/3` | 2 | 0.111 | 7.6e-03 | 0.0077 |
| **sine `b=1/3`** | **4** | **0.0123** | **7.6e-05** | **0.0858** |
| sine `b=1` | 4 | 1.0 | 1.7e-03 | **0.00016** |
| tanh | 4 | 1.0 | 2.3e-03 | 0.00036 |

At depth 4 the `b=1/3` gradient is **22× smaller** than at `b=1`, and the error
is **536× worse**. At `b=1` the gradient does not shrink at all, and depth 4
becomes the best result in the whole table — better than depth 1 or 2, which is
what depth is supposed to buy you.

**So in this architecture the vanishing gradient is not a property to design
around. It is the §5 bug seen from a second direction,** and it has the same
one-number fix. The general claim that vanishing gradients are worth accepting
may hold elsewhere; here the gradient vanishes because every neuron was
initialised onto the flat part of its activation, and it stops vanishing the
moment that is corrected.

## 11. Inversion: nothing tested helps, and most of it is catastrophic

The proposal: detect the stall, flip the update sign so the network climbs
*up* the loss for a burst of steps, then resume descending. Three ways of
detecting the stall were built and tested at depth 4, three seeds each.

| activation | policy | error | worst seed | fires | vs. doing nothing |
|---|---|---|---|---|---|
| sine `b=1/3` (genuinely stalled) | none | 0.0785 | 0.0858 | 0 | — |
| sine `b=1/3` | small gradient | 0.0789 | 0.0858 | 132 | 0.99× |
| sine `b=1/3` | small weights | 0.0785 | 0.0858 | 44 | 1.00× |
| sine `b=1/3` | loss plateau | 0.0786 | 0.0858 | 23 | 1.00× |
| sine `b=1` (healthy) | none | **0.00017** | 0.00022 | 0 | — |
| sine `b=1` | small gradient | 0.0884 | 0.0967 | 13 | **0.002×** |
| sine `b=1` | small weights | 0.0322 | 0.0961 | 1 | **0.01×** |
| sine `b=1` | loss plateau | 0.0564 | 0.0907 | 11 | **0.003×** |
| tanh (healthy) | none | **0.00038** | 0.00040 | 0 | — |
| tanh | small gradient | 0.0925 | 0.1010 | 132 | **0.004×** |
| tanh | small weights | 0.00067 | 0.00123 | 9 | 0.57× |
| tanh | loss plateau | 0.00039 | 0.00041 | 6 | 0.98× |

**The pattern is uniform and it is not encouraging. On a healthy network every
policy that fires does damage. On a genuinely stalled network no policy that
fires does any good.** The best outcome any of them achieves is 1.00× — no
effect. There is no cell in this table where inversion earned its place.

Two details that a mean would hide, which is why the table reports fires as a
total over the three seeds and shows the worst single seed:

* **One inversion is enough to ruin a network.** The "small weights" policy on
  `sine b=1` fired exactly **once**, on one seed out of three. That seed went
  from 0.00016 to 0.0961 — a 600× loss — while the other two were left untouched
  and identical to the baseline. The printed average of 0.0322 is one disaster
  and two non-events, not a mild effect.
* **Triggering on a small gradient is the worst of the three,** because a small
  gradient means *stuck* or *converged* or *only just started* and the magnitude
  cannot tell them apart. It fired 132 times, mostly during the early transient
  where gradients are briefly flat and the loss is legitimately still high, and
  each burst of climbing destroyed what had been learned.

**A plateau is the better *detector* and still not a useful *trigger*.** "Past a
warmup, the running loss has not improved for a window" does separate stuck from
starting from converged, and it shows: 6 fires on healthy tanh against the
gradient trigger's 132. But detecting the stall correctly does not make climbing
out of it work. On the healthy `sine b=1` network it still fired 11 times and
still cost 330×.

**Longer bursts make it worse, not better.** An earlier version of this document
reported that a plateau trigger with long bursts recovered 0.0635 on the stalled
network — a 1.24× improvement. **That does not reproduce.** Sweeping the burst
length on the stalled network, three seeds each:

| burst length | error on `sine b=1/3` | vs. doing nothing |
|---|---|---|
| 40 steps (the default) | 0.0786 | 1.00× |
| 100 | 0.0799 | 0.98× |
| 200 | 0.0896 | 0.88× |
| 400 | 404.6 | diverged |
| 800 | 8775 | diverged |

The trend is monotone downward: every increase in burst length makes it worse
until it blows up entirely. No burst length tested recovers the 1.24×, and the
committed defaults give 23 fires where the retired claim reported 15, so that
number came from a configuration that is not in the code. **It is withdrawn**
until a configuration that reproduces it is committed alongside it.

**The one thing that makes inversion safe also makes it pointless.** Gating the
trigger on "the loss is still bad" (`loss_tau`) protects the healthy networks
perfectly — they go to **zero fires** and become bit-identical to the baseline.
It protects them by never firing. On the genuinely stalled network the gate
cannot help, because there the loss really is bad, so it fires just as often and
achieves the same 1.00× nothing.

**The comparison that settles it.** Inversion's best result on the stalled
network is 1.00×. Setting `b = 1` on that same network takes it from 0.0785 to
0.00016 — **536×**. The escape mechanism is worth nothing on a problem the
initialisation fix removes entirely.

## 12. Dual network: replay beats grafting, and compute explains almost all of it

Four games, specialists of 4 hidden units each, one shared query network of 16.
The "updates" column counts SGD steps **on the shared network**, which is the
contended resource and the thing the three routes really differ on.

| activation | route | error | updates | vs. joint | vs. joint at matched budget |
|---|---|---|---|---|---|
| tanh | joint (the baseline) | 0.0626 | 115,200 | — | — |
| tanh | joint, **same budget as replay** | 0.0417 | 192,000 | 1.50× | — |
| tanh | **replay** | **0.0398** | 192,000 | 1.57× | **1.05×** |
| tanh | **graft** (fine-tune only) | 0.0542 | **38,400** | 1.16× | 0.77× |
| sine b=1 | joint | 0.0908 | 115,200 | — | — |
| sine b=1 | joint, same budget | 0.0547 | 192,000 | 1.66× | — |
| sine b=1 | replay | 0.0524 | 192,000 | 1.73× | **1.04×** |
| sine b=1 | graft | 0.0748 | 38,400 | 1.21× | 0.73× |
| *either* | *specialists alone* | *0.0081 / 0.0128* | — | — | — |

**Against a naive baseline both routes look strong — replay 1.57×, graft 1.16×.
Most of replay's margin is simply more compute.** Replay trains the shared
network on generated samples before fine-tuning, so it takes 192,000 updates
against joint's 115,200. Give joint the same budget and it reaches 0.0417
against replay's 0.0398: **1.05×, not 1.57×.** The mechanism is worth about five
percent — real, and small.

**Grafting's value is compute, not accuracy.** At matched budget it is actually
*behind* joint training (0.77×). What it does is reach 0.0542 using **38,400**
updates of the shared network — a fifth of what the matched baseline spends, and
a third of what naive joint training spends to reach a worse 0.0626. The
specialists train independently and can run in parallel; the shared network, the
bottleneck, only fine-tunes. For an architecture where many games are learned and
one network is queried, moving work off the contended resource onto parallel
independent learners is the win, and that is exactly what "slot the weights in,
then fine-tune" does.

**Replay is the better transfer where accuracy matters, and the sleep analogy is
why.** Grafted weights land in a network that has an input the specialist never
had (the selector) and an output squeezed into a band — they are in the wrong
coordinate system, and fine-tuning has to repair them. Replay transfers the
*function* instead of the *parameters*, so no coordinate system has to line up.
That is also what the neuroscience describes: consolidation during sleep is
replay-driven, the hippocampus regenerating experience for the neocortex, not
synapses being copied between structures. **The analogy predicted the better of
the two implementations before either was run.**

**What neither route fixes:** a specialist alone reaches 0.0081 where the best
shared network reaches 0.0398. Compression costs about **5×**, and no
consolidation route recovers it. That is the same cost Part 1 measured from the
other side, and it is the price of one network holding many games.

## 13. Predictions this part got wrong

* **"Replay gives 1.5×."** It gives 1.05× once joint training gets the same
  budget. The first comparison was against an under-trained baseline.
* **"A loss gate repairs the gradient trigger."** It does not — early training
  is flat *and* bad, which is exactly what the gate lets through. A plateau
  condition separates the cases properly, and still does not make inversion
  useful.
* **"A plateau trigger with long bursts gives 1.24× on a stalled network."**
  Withdrawn (§11). It does not reproduce at any burst length tested, and longer
  bursts make it monotonically worse until the network diverges.
* **"Grafting trades a little accuracy for a lot of compute."** Half right: the
  compute saving is real and large, but at matched budget grafting is *behind*
  plain joint training, not merely a little behind replay.
* Two implementation bugs were caught before they became results: a backward-pass
  index error (`as_[l-1]` where `as_[l]` was meant), and a probe-then-step
  pattern that gave the inversion conditions two updates per sample where the
  baseline got one.

## 14. What to build

* **Fix `b` first.** Everything in §10 sits upstream of both mechanisms.
* **Do not build inversion.** No trigger tested — gradient, weight, or plateau —
  improved any network at any burst length. The best case is no effect; the
  common case is destruction. Keep the *plateau detector* itself, which is a
  genuinely good stall detector (§11) and is useful for deciding when to **grow**
  a network, which is what `CyclicCortex/DESIGN.md` §8 uses it for.
* **Build the dual network, and transfer by replay**, using grafting where
  shared-network compute is the constraint rather than accuracy.
* **Do not expect consolidation to pay for compression.** The 5× gap between a
  specialist and a shared band is structural.

---

# Part 4 — rotating inversion

A third proposal: instead of inverting the whole network, invert alternating
*layers* each cycle. At depth 5 that means layers {1,3,5} invert, then {2,4},
then {1,3,5} again. Implemented as `Rotating` in `vanishing.py`, in two readings
of "invert":

* **`mode="grad"`** — the active layers climb while the others descend. An
  adversarial split inside one network.
* **`mode="weight"`** — the active layers have `w → −w` at each cycle boundary.

These are not part of `vanishing.py`'s default run; the numbers below come from
ad-hoc calls.

## 15. Period 4, verified

With an odd activation — and `−sin(bz)` is odd — negating a layer's weights is a
structural operation rather than noise. The parity alternation has **period 4**:

```
(odd)(even)(odd)(even)  ->  every weight bit-identical to its original value
```

Verified directly: cycles 4 and 8 return the network exactly; cycles 1–3 and 5–7
do not. It is a closed loop through the network's own sign structure, and it is
`Research/CyclesAreAFeature.md` appearing in weight space.

## 16. Safe, where every other ascent method is catastrophic

Depth 5, 3–4 seeds, on two landscapes: *smooth* (one low-frequency sine,
essentially no local minima) and *rugged* (a high-frequency product with many
basins — the regime an escape mechanism is actually for).

| landscape | activation | method | error | vs. none |
|---|---|---|---|---|
| smooth | sine `b=1` | none | 0.00029 | — |
| smooth | sine `b=1` | plateau inversion | 20.30 | **0.00×** |
| smooth | sine `b=1` | rotating **grad** | 0.0599 | **0.00×** |
| smooth | sine `b=1` | rotating **weight** | 0.00041 | **0.71×** |
| smooth | tanh | none | 0.00054 | — |
| smooth | tanh | rotating **grad** | 0.0920 | **0.01×** |
| smooth | tanh | rotating **weight**, p=10 | 0.00052 | **1.04×** |
| rugged | sine `b=1` | none | 0.01599 | — |
| rugged | sine `b=1` | plateau inversion | 0.0295 | 0.54× |
| rugged | sine `b=1` | rotating **weight**, p=50 | 0.01604 | **1.00×** |
| rugged | tanh | none | 0.01587 | — |
| rugged | tanh | plateau inversion | **22288** | 0.00× |
| rugged | tanh | rotating **weight**, p=50 | 0.01600 | **0.99×** |

**Rotating weight inversion is in a different category from everything else
tried.** Every gradient-*ascent* method — plateau-triggered, magnitude-triggered,
rotating-grad — is catastrophic on a healthy network, by 100× to a million×.
Rotating weight inversion is 0.71–1.04×: survivable, and sometimes
indistinguishable from not doing it at all. Negating weights **relocates** the
network; ascent **unlearns** it. Both are called inversion and they are not the
same operation.

## 17. And exactly neutral, for a reason worth having measured

It does not help. Not on the smooth task, and — the test that matters — **not on
the rugged one either**, where there genuinely are many basins to escape: 1.00×
and 0.99×. Long periods are neutral; short ones (p=10) are mildly harmful.

The explanation is not the obvious one. **It is not a loss-preserving symmetry.**
Measuring the loss across four negations with no training in between:

| event | sine `b=1` | tanh |
|---|---|---|
| trained | 0.00338 | 0.00039 |
| after negating {0,2,4} | 0.2513 | 0.1865 |
| after negating {1,3} | 0.2697 | 0.2129 |
| after negating {0,2,4} | 0.1405 | 0.0749 |
| after negating {1,3} | **0.00338** | **0.00039** |

The loss **jumps about 75×** on the first negation, so each hop is a large
perturbation, not a free move along a level set. What makes the method neutral is
the **closure**: after four cycles the weights are bit-identical, so the network
has been kicked out, partially repaired by training, kicked again, and returned
exactly to where it started. **A cycle that returns to its origin does no work**,
and the training spent inside the excursion mostly fits configurations that get
negated away again.

That also says what would have to change for it to help: **break the closure.**
Period 4 comes from two parities over an involution (`w → −w` twice is the
identity). Three or more phases, an asymmetric subset each cycle, or pairing the
negation with something non-involutive would leave the excursion open, and the
network would actually explore instead of returning. Whether an open version
beats plain descent is untested, and it is the obvious next thing to try.

## 18. Where this leaves the four methods

| method | on a healthy network | on a genuinely stalled one | verdict |
|---|---|---|---|
| gradient-magnitude inversion (§11) | 0.002–0.004× | 0.99× (no effect) | **do not build** |
| weight-magnitude inversion (§11) | 0.01–0.57× | 1.00× (no effect) | **do not build** |
| plateau-triggered inversion (§11) | 0.003–0.98× | 1.00× (no effect) | **do not build** |
| rotating **grad** (§16) | 0.00–0.01× | no effect | **do not build** |
| rotating **weight** (§16–17) | 0.71–1.04× | 1.00× | safe, no benefit as specified |

The plateau *detector* is worth keeping for deciding when to grow a network. The
inversion it was built to trigger is not.

And all of it sits under §10: the stall these methods were built to escape is
caused by `b = 1/3`, and correcting that is worth 536× where the best escape
mechanism is worth nothing at all.
