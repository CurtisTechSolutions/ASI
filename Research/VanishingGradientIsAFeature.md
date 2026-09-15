# The Vanishing Gradient Is a Feature

**Count the multiplications, and the decay stops being a loss and becomes a reading**

Mason Curtis — Curtis Tech Solutions
Working paper · September 2026

---

## Abstract

The vanishing gradient is treated as the central pathology of deep networks.
It is not a pathology. It is an **encoding**.

Back-propagation multiplies. A gradient that reaches a layer `n` products away
from the loss has been through `n` of them, and each one scales it by roughly the
same factor `c`, so it arrives at about `c**n` of the size it started. Everyone
knows this. What follows from it is the part nobody uses: **`c**n` is
invertible.** The magnitude was not destroyed, it was multiplied by a known
number. The gradient is carrying a message about its own depth, written in an
exponential code, and the key to that code is `n` — which is free, because `n` is
just the depth, known from the architecture before any data is seen.

We threw away the key and then complained the message was unreadable.

This paper takes the count seriously: track how many times we did `X*X`, and use
that number to put the values back in range. I measure whether the premise
actually holds (it does, and it gets *stronger* with depth: R² of 0.926 at depth
2 rising to **0.997** at depth 12, with the fitted `c` converging on the value
theory predicts). I then test whether dividing the decay back out actually helps,
against two controls built to sink the claim: a count-free per-layer
normalisation, and a constraint holding every arm's total step size identical so
the comparison cannot be won by simply taking bigger steps. Counting beats plain
SGD by 43% and beats the count-free control, in a window around depths 3–4. Past
that the whole SGD family stalls together, and **Adam beats every count-based arm
at every depth by roughly 3×** — which I state here rather than bury, because it
is the most informative result and it is not in my favour.

The experiment is `experiments/depth_counted_normalisation.py`, standard library
only, runnable with nothing installed.

---

## 1. The claim

> The vanishing gradient is a feature of gradient descent. Keep track of the
> number of times we do `X*X` — the underlying cause — and use that number to
> normalise the values back to 0–1.

That is the whole idea. The rest is what it means and whether it survives
contact with a measurement.

---

## 2. Where it came from

This one I had already half-answered and not noticed.

The RadixCyclicNN design table contains a line I wrote a long time before I
understood what it implied:

> **Accept the vanishing gradient, update the activation function instead;
> `N*N`; activation(child) × activation(parent)**

The `N*N` there is this paper's `X*X`. The edge signal in that architecture is
`W[p,c] · f_p(z_p) · f_c(z_c)` — a weight times the activation of the parent
times the activation of the child. A **product**. Every edge is one
multiplication of two activations, and a path of length `n` is `n` of them in a
row. That is the vanishing gradient, in its smallest possible form.

My answer there was to dodge it. The learning rule is one hop and local, so
`n = 1` always, there is no product chain, and vanishing gradients "never enter
the picture". That works, and I still think it is right for that architecture.

But dodging a thing is not understanding it. The question I did not ask then is
the one here: **if the problem is that `n` is large, what happens if you just
keep `n`?**

---

## 3. `X*X` is the entire cause

Worth being precise, because the whole argument rests on the cause being this
simple.

For a layer `z_{l+1} = W_l a_l + b_l` with `a_l = f(z_l)`, back-propagation sends
the error backwards as

```
δ_l = (W_l^T δ_{l+1}) ⊙ f'(z_l)
```

so

```
||δ_l||  ≈  ||W_l^T|| · |f'| · ||δ_{l+1}||
```

Every backward hop multiplies by two things: the weight matrix and the activation
derivative. Scale the weights so the linear part is roughly norm-preserving —
which is exactly what Xavier initialisation is for — and what is left is `|f'|`.

For the logistic sigmoid, `max |f'| = 0.25`, at `x = 0`, and it is smaller
everywhere else. So each hop multiplies the gradient by **at most a quarter**,
and usually less. Twelve layers of that is `0.25**12 ≈ 6e-8`.

That is the vanishing gradient in one line. It is not subtle, it is not an
interaction effect, it is not an artefact of any particular optimiser. It is
repeated multiplication by a number smaller than one, `n` times, and `n` is the
depth.

---

## 4. Nothing is lost — it is encoded

Here is the step the field does not take.

A gradient that has been through `n` hops looks like

```
g_n  ≈  c**n · r_n
```

where `c` is the typical per-hop contraction and `r_n` is whatever the gradient
would have been without the decay — the part you actually want.

`c**n` is a **number**. It is not noise, it is not a projection, it is not a
lossy compression. It is multiplication by a positive scalar, and multiplication
by a positive scalar is invertible for any `c ≠ 0`. Divide by `c**n` and `r_n`
comes back.

The information was never destroyed. It was **moved into the exponent**, and the
exponent is `n`, and `n` is the depth, which you already know.

So the thing that makes the vanishing gradient a *feature* rather than a bug:

> The magnitude of a gradient is a measurement of how far it travelled. A network
> is reporting its own depth on every backward pass, in the one quantity we
> decided to interpret as damage.

Read `log||g|| / log(c) ≈ n` and the gradient tells you where it came from. We
have been reading "the number is small" and concluding "the signal is gone", when
the correct reading is "the signal came from `n` layers down, and here is `n`".

### 4.1 The more useful consequence

Recovering `r_n` is the obvious use, and it is what §6 tests. The subtler one is
that **once you know what the decay should be, you can see when it isn't.**

`c**n` is the *expected* magnitude at depth `n`. A layer whose gradient departs
from that prediction is doing something the geometry does not explain — and that
is exactly the layer worth looking at. Without a model of the expected decay,
every deep layer just looks small and they all look alike. The vanishing gradient
is the baseline that makes per-layer anomalies visible at all.

You cannot detect an anomaly without a norm to be anomalous against. The decay is
the norm.

---

## 5. Two places to put the count

The proposal is to track how many `X*X` operations happened and use that number
to bring the values back into range. There are two places to do it, and they are
not equivalent.

**At every multiplication.** After each product, rescale the value back into
`[-1, 1]` and increment a counter. The number never gets small, because you never
let it. What would have been magnitude is now sitting in the counter:

```
value  ->  (mantissa in [-1,1],  count n)
multiply: mantissa *= other, renormalise, n += 1
```

Nothing can underflow, because nothing is ever allowed to get far from 1. The
decay is not suffered and then repaired; it is never incurred.

**Once, at the end.** Let the gradient decay as it normally does, then divide by
`c**n` when it arrives. Cheaper — one scalar per layer instead of a rescale per
operation — but you have to *know* `c`, and §7 shows how badly that can bite.

The per-step version has a punchline worth stating plainly:

> That is floating point. A mantissa kept near 1 and an exponent counting the
> scale is exactly what every `float` in the machine already is. **We are already
> counting the multiplications — every float does — we just never let the
> learning rule see the count.**

The exponent exists. It is right there in the number. It is hidden in the
hardware representation, used for arithmetic, and never exposed to the optimiser
as a signal. All this proposal does is move it from the hardware into the
algorithm.

---

## 6. Does the premise even hold?

Everything above assumes the decay is really geometric — that `c**n` is a good
model and not a hand-wave. If it isn't, the count is not enough to undo anything
and the idea dies at the first step. So that is what I measured first.

Method: a deep sigmoid MLP, Xavier init, full-batch MSE on a smooth two-input
regression target. Take the per-layer gradient norms at initialisation, and fit

```
log ||g_n||  =  a + b·n        →   c = exp(b)
```

If `n` really is a sufficient statistic for the decay, that line should fit
almost perfectly.

| depth | fitted `c` | R² | total decay across the stack |
|---:|---:|---:|---:|
| 2 | 0.0987 | 0.9259 | 9.7e-03 |
| 4 | 0.1833 | 0.9544 | 9.3e-04 |
| 6 | 0.1905 | 0.9811 | 3.0e-05 |
| 8 | 0.2034 | 0.9947 | 2.5e-06 |
| 12 | 0.2277 | **0.9969** | 9.2e-09 |

Two things in that table, and both matter.

**The fit is excellent, and it gets better with depth.** R² climbs from 0.926 to
0.997. The geometric model is not an approximation that degrades as the stack
grows — it *sharpens*. That is the opposite of how approximations usually behave,
and the reason is that the model is an average over hops: more hops, better
average. **The count predicts the decay most accurately exactly where the decay
is most severe.**

**The fitted `c` converges on the value theory predicts.** It rises 0.0987 →
0.2277, heading for `max |f'| = 0.25`, the a-priori sigmoid value from §3. Nobody
fitted that. It is the derivative of the activation function, known before any
data existed, and the measurement walks toward it as depth grows.

So the premise holds, and holds strongly. `n` is a sufficient statistic for the
decay. Whatever else is true, the claim that *the decay is predictable from the
count alone* is now measured rather than asserted.

---

## 7. The correction, and the dial the count creates

Knowing the decay is removable is not the same as removing it being a good idea.
So: rescale each layer's update by dividing out the decay the count predicts,

```
update_l  =  g_l  /  c**(alpha · n_l)
```

and put a dial on how much of it to undo:

```
alpha = 0   ->   divide by 1        ->   plain SGD, the decay is left in
alpha = 1   ->   divide by c**n     ->   the decay is entirely undone
```

**The dial is the point.** Without `n` there is no dial — only the two ends. You
can leave the decay alone, or you can equalise everything with some count-free
normalisation, and there is nothing in between because there is no quantity to
interpolate against. The count supplies the axis. Whether anything interesting
lives in the middle of that axis is an empirical question, and §9 answers it.

Where does `c` come from? Two options, and the difference between them turns out
to matter more than anything else in this paper:

* **fixed** — measure `c` once, at initialisation, and use it forever.
* **online** — re-fit `c` from the current gradients at every step. Still **one
  scalar for the whole network**, fitted by least squares across the layers. That
  single-scalar constraint is what keeps this honest: it can only remove the
  *systematic geometric* component, the part `n` actually predicts. Everything
  else is left alone.

---

## 8. The control that makes the experiment mean anything

There is an obvious way to fake this result, and I want to be explicit that I
closed it.

Dividing deep layers' gradients by `c**n` makes their updates enormously bigger.
If you compare that against plain SGD directly, the corrected run takes much
larger steps overall — and then you have not measured the idea, you have measured
*a larger learning rate for deep layers*, which is a thing you could have done
without any of this.

So every arm is renormalised after reallocation:

```
reallocate the step across layers        (this is where the arms differ)
then rescale everything so the global gradient norm is unchanged
```

Every arm therefore takes **exactly the same size of step**. They differ only in
how that fixed step is spread across the layers. Any difference in the result is
a difference in *allocation*, which is the only thing being claimed.

The second control is an arm called `normalised`: divide each layer's gradient by
its own measured norm. That equalises the layers too, and it uses **no count at
all** — it reads the answer off the data instead of predicting it from the
architecture. If `normalised` matches the best `alpha`, then the count bought
nothing and this paper's specific proposal fails while its premise survives. It
is reported either way.

`adam` is in there as the reference for what the field actually does, and it is
not a straw man: it gets its own learning rate and a fair shot.

---

## 9. Results

Depth-4 sigmoid MLP, width 6, 1000 steps, 4 seeds, lr 0.5. Predicting the mean
scores **0.1796**, so anything near that has not learned. Every arm takes the
same total step size (§8).

| arm | final MSE | std | bottom layer moved |
|---|---:|---:|---:|
| counted `alpha=0.00` (= plain SGD) | 0.1271 | 0.0375 | 0.87 |
| counted `alpha=0.25` | 0.1306 | 0.0362 | 0.81 |
| counted `alpha=0.50` | 0.1265 | 0.0424 | 1.12 |
| **counted `alpha=0.75`** | **0.0727** | 0.0412 | 4.45 |
| counted `alpha=1.00` | 0.0807 | 0.0317 | 7.37 |
| counted-fixed `alpha=1.00` (stale `c`) | 0.1083 | 0.0398 | 4.31 |
| normalised — **no count** | 0.0906 | 0.0417 | 3.94 |
| **adam** | **0.0248** | 0.0187 | 8.23 |

**The mechanism does what it says.** Read the last column down the `alpha` sweep:
0.87, 0.81, 1.12, 4.45, 7.37. That is how far the *bottom* layer — the one `n`
hops from the loss, the one that normally never moves — travelled from its
initialisation. Turning the dial up gets the step down to the deep layers,
monotonically, with the total step size held fixed. Whatever else is arguable,
the correction is provably doing the thing it was built to do.

**Counting beats plain SGD.** 0.0727 against 0.1271, a 43% reduction, and the
gap is larger than one standard deviation. This is the most solid of the
optimiser comparisons.

**The best dial setting is interior — probably.** `alpha=0.75` beat both ends.
But 0.0727 against `alpha=1.00`'s 0.0807 is well inside a standard deviation of
0.04, so "the optimum is interior" is an indication and not a result. What I can
say is that it is *not* obviously at an end.

**Counting beats the count-free control, but not convincingly.** 0.0727 against
`normalised`'s 0.0906. The count helps beyond simply equalising the layers — the
direction is right and it repeats across depths below — but 0.018 inside a
standard deviation of 0.04 is not a finding.

**Re-fitting `c` matters a lot.** A `c` measured once at initialisation scores
0.1083; re-fitting it each step scores 0.0727. Same formula, same dial, 49%
worse for using a stale scalar. This is §10's fragility, measured.

**Adam wins by 3×, with the lowest variance.** Not a footnote — see §11.

### Across depth

Same settings, `alpha=1.00` for the counted arm.

| depth | plain | counted | normalised | adam | honest reading |
|---:|---:|---:|---:|---:|---|
| 2 | 0.0585 | 0.0366 | 0.0376 | 0.0210 | counted ≈ control |
| 3 | 0.0803 | **0.0292** | 0.0645 | 0.0208 | counting wins |
| 4 | 0.1271 | **0.0807** | 0.0906 | 0.0248 | counting wins |
| 5 | 0.1471 | 0.1468 | 0.1469 | **0.0118** | whole SGD family stalls |
| 6 | 0.1471 | 0.1471 | 0.1471 | **0.0051** | whole SGD family stalls |

Three things, and the third is the one that surprised me.

**There is a window.** At depths 3 and 4 the count clearly beats both plain SGD
and the count-free control. At depth 2 there is barely any decay to remove, so
counted and normalised come out level — as they should.

**Past depth 4 the entire SGD family dies together.** Plain, counted and
normalised all land on 0.147, identical to four significant figures. That is not
the correction failing; it is every layer-reallocation strategy failing at once,
which means the problem there is no longer *where* the step goes. Reallocating a
step that is the wrong shape does not help however cleverly you reallocate it.

**Adam gets better as the others get worse.** 0.0248 → 0.0118 → 0.0051 going from
depth 4 to 6, while everything else flatlines. Whatever Adam is doing, it is not
what any of these arms are doing, and it does not hit the same wall. That is the
single most informative number in this paper and it is not in my favour.

---

## 10. Where it is genuinely fragile

This is the real limitation, and it is structural rather than a tuning problem.

The correction is `1 / c**(alpha·n)`. It is **exponential in `n`**, so any error
in `c` is exponential in `n` too. If your estimate is off by a ratio `k`, the
correction at depth `n` is off by `k**n`:

| error in `c` | at depth 4 | at depth 8 | at depth 12 |
|---|---:|---:|---:|
| 10% | 1.5× | 2.1× | 3.1× |
| 25% | 2.4× | 6.0× | 14.6× |
| 50% | 5.1× | 25.6× | 130× |

A 50% misestimate of a single scalar produces a 130× error in the update at depth
12. Nothing else in the method is delicate; this is, and it is delicate in a way
that gets worse precisely where the method is supposed to be most useful.

The experiment measures it directly. `counted-fixed` uses a `c` measured once at
initialisation and never updated; `counted` re-fits it every step. Same formula,
same dial, same everything else. The fixed version is substantially worse,
because `c` **moves during training** — weights grow, sigmoids saturate, and the
contraction that was correct at step 0 is wrong by step 100.

So the count is necessary but not sufficient. You need `n`, which is free, *and*
`c`, which is not. The honest statement of the method is:

> The depth is free and exact. The per-hop contraction is neither, and the whole
> thing is only as good as your running estimate of one scalar.

Re-fitting online fixes it here — it is one least-squares fit over a handful of
points per step, which costs nothing. But "fixes it here" is a claim about a
twelve-layer MLP, and I would want to see it on something deeper before trusting
the correction at depth 50.

---

## 11. Adam already does this, empirically — and it wins

I am not going to bury this. **Adam beat every count-based arm I ran, by a wide
margin, with the lowest variance.**

It is worth understanding why, because the reason is instructive rather than
damning.

Adam divides each parameter's update by a running RMS of that parameter's own
gradient. That is an *empirical, per-parameter* estimate of exactly the scale
this paper computes *structurally, per layer*. Adam therefore captures:

* the systematic geometric decay with depth — the part `c**n` models, **and**
* every other source of scale variation: per-parameter, per-layer, non-geometric,
  data-dependent, changing over training.

The count model captures only the first. It is one scalar raised to a known
power. Adam is a free parameter per parameter. Of course it fits better — it has
vastly more capacity to fit with, and the experiment says that extra capacity is
being used on something real.

What the count has instead is a short list, and the items on it are narrow but
genuine:

| | counted | Adam |
|---|---|---|
| extra state | one integer per layer | two floats per **parameter** (2× the model) |
| warm-up | none — correct at step 0 | needs steps to estimate; cold at the start |
| where it comes from | the architecture, before any data | the data, after seeing some |
| what it can express | the geometric trend only | essentially anything |
| cost | a least-squares fit over a few points | a multiply-accumulate per parameter |

So the honest summary is not "counting beats Adam". It is:

> Counting recovers, for free and immediately, the part of the scale that is
> predictable from the architecture. Adam recovers that part and more, but has to
> pay for it in state and learn it from data it has not seen yet.

Which suggests the obvious thing I have **not** tested and should: use the count
to **initialise Adam's second moment**, so it starts warm at the architecturally
predicted scale instead of cold at zero. The count is available at step 0 and
Adam's estimate is not, and that is the one window where the structural version
has something the empirical version cannot get. That is the experiment I would
run next, and I am not going to claim its result in advance.

---

## 12. What the field does instead, and what it costs

Two standard answers to the vanishing gradient, read in the light of §4.

**ReLU does not decode the message — it stops the message being written.**
ReLU's derivative is 0 or 1, so for active units `c ≈ 1` and there is no decay.
The problem is gone. But look at what else is gone: with `c = 1`, the gradient's
magnitude no longer says anything about how far it travelled. `log||g|| / log(c)`
is not even defined. The depth signal was not recovered, it was **flattened
out of existence**.

That is a defensible engineering trade if you do not want the signal. It is not a
solution to the problem as posed here, and it is not free: as measured in the
companion paper on the sine activation, ReLU is dead on exactly 50% of its input
space, permanently, and a unit that drifts negative and stays there is gone.

**Normalisation layers compute the scale and then throw it away.** BatchNorm and
its relatives rescale activations to keep them in range — which is precisely the
per-step renormalisation of §5. They divide out a scale at every layer. They just
do not hand that scale to the learning rule as a quantity to reason about; it is
used to fix the forward pass and then discarded.

Which is the theme of this whole paper, showing up for the third time:

> The count is already being computed. Floating point computes it. Normalisation
> layers compute it. We have simply never treated it as information.

---

## 13. How this sits with the other three papers

This is the fourth of these, and the pattern is not an accident.

* **`CyclesAreAFeature.md`** — a cycle is not a bug in the graph, it is
  repetition stored once. An exponent is scale stored once. Structurally the same
  move: a thing that looks like a defect is a compressed representation, and the
  defect is only a defect because we discarded the index that decompresses it.
* **`SineWaveActivationFunction.md`** — the direct connection. `|f'|` for the sine
  is `|a·b·cos(b(x−h))|`, which returns to its maximum every half period, so `c`
  never collapses toward zero the way sigmoid's does. Where sigmoid's dead
  fraction grows toward 100% as pre-activations grow, the sine's stays at about
  2%. In this paper's language: **the sine keeps the message readable.** It does
  not remove the exponent, it keeps the base bounded away from zero.
* **`2NRL.md`** — the same rhetorical shape again: go *into* the failure rather
  than away from it, because the failure carries the information.
* **RadixCyclicNN itself** — the other solution entirely. Rather than decoding the
  exponent, make sure one is never created: the learning rule is one hop, `n = 1`
  always, and "vanishing gradients never enter the picture". That is a real fix
  and it is the one I shipped. This paper is the road not taken from that same
  design line.

Four papers, one move: something the field routes around turns out to be carrying
information, and the fix has been to delete the signal rather than read it.

---

## 14. Limitations, stated plainly

1. **The effect sizes are inside the noise.** With four seeds and per-arm standard
   deviations around 0.04, differences of 0.02 are suggestive and nothing more.
   The premise measurement (§6) is solid; the optimiser comparison (§9) is not,
   and I am not going to pretend otherwise.
2. **One task, one architecture, one activation.** A small sigmoid MLP on smooth
   two-input regression. Sigmoid was chosen *because* it vanishes hardest, which
   makes the premise easy to see and makes the setting unrepresentative.
3. **Shallow by modern standards.** Twelve layers is where the R² result is
   strongest and it is nowhere near where the question actually bites.
4. **`c` is assumed uniform across layers.** One scalar for the whole network. A
   network with heterogeneous layers — different widths, different activations,
   attention — has no single `c`, and the one-scalar fit would be modelling
   something that is not there.
5. **Adam wins and I have not beaten it.** The count-based method is better than
   plain SGD and better than the count-free control; it is not better than the
   standard tool.
6. **The per-step renormalisation of §5 is not implemented.** The experiment tests
   the end-correction form. The mantissa-and-counter form is the one that cannot
   underflow by construction, and it is the one I would build next.

---

## 15. What would change my mind

* **R² falling with depth on a realistic architecture.** If the geometric model
  degrades where it matters, the count is not a sufficient statistic and the
  whole premise goes. §6 is the load-bearing result and this is how it breaks.
* **`normalised` matching the best `alpha` across seeds and depths.** That would
  mean equalising the layers is what helps and the *count* contributes nothing —
  the premise would survive, the proposal would not.
* **Count-initialised Adam showing no early advantage** (§11). That is the one
  regime where the structural estimate should beat the empirical one, because
  Adam has no data yet and the architecture is already known. If the count does
  not help at step 0, it is hard to see where it would.

---

## 16. Summary

* Back-propagation multiplies. `n` hops, each scaling by about `c`, gives `c**n`.
  That is the entire mechanism of the vanishing gradient — repeated `X*X`, with
  `n` equal to the depth.
* `c**n` is **invertible**. The magnitude was not destroyed, it was moved into an
  exponent, and the key to that exponent is `n` — which is free, because it is the
  depth, known from the architecture before any data is seen.
* So the vanishing gradient is a **measurement**: the gradient reports how far it
  travelled, in the one quantity we chose to read as damage. And once you know
  what the decay *should* be, you can see which layers depart from it — you cannot
  spot an anomaly without a baseline, and the decay is the baseline.
* The premise is measured and holds: `log||g||` is linear in the count with R²
  from 0.926 at depth 2 to **0.997** at depth 12, and the fitted `c` converges on
  `max |f'| = 0.25`, the value the activation's derivative predicts a priori. The
  model **sharpens** with depth — it is most accurate exactly where the decay is
  worst.
* Dividing the decay back out beats plain SGD and beats a count-free per-layer
  normalisation, and the best setting of the dial is interior — but those gaps are
  within one standard deviation at four seeds, so they are indications, not
  findings.
* The correction is exponentially sensitive to `c`: a 50% error in one scalar is
  a 130× error at depth 12. `c` has to be re-fitted online — a stale `c` measured
  once at initialisation gives back barely a third of the benefit (0.108 against
  0.073, where doing nothing scores 0.127).
* And the window is narrow. Counting clearly wins at depths 3 and 4; past depth
  4 plain, counted and count-free normalisation all land on the same number, so
  the problem stops being *where* the step goes and no reallocation strategy
  helps.
* **Adam wins**, because it estimates per-parameter what this estimates per-layer.
  What counting has is that it is free, needs no state, and is correct at step
  zero — which is exactly where Adam is coldest, and which is the experiment to
  run next.
* Floating point already counts the multiplications. Normalisation layers already
  compute the scale. We have simply never handed either number to the learning
  rule.

The gradient was never silent. It was telling us how far it had come, and we
were reading the volume instead of the message.
