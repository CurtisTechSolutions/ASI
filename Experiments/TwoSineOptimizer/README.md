# TwoSine — Adam, with the two moments read as two interfering sine waves

**Standard library only** — no numpy, no torch, no gym — so everything here runs
anywhere with nothing installed, the same rule `RadixCyclicNN/` and the rest of
`Experiments/` follow. Everything is seeded and deterministic: the same command
twice gives the same table.

| file | what it is |
|---|---|
| `two_sine_optimizer.py` | the optimiser — `TwoSine`, `Adam`, and the ablations |
| `test_two_sine_optimizer.py` | the benchmark that produced the numbers below |

```bash
python3 test_two_sine_optimizer.py            # full run (~1 min 40 s)
python3 test_two_sine_optimizer.py --quick    # ~15 s
python3 test_two_sine_optimizer.py --check    # algebra + gradient checks only
```

Every table below is from the default full run — 5 seeds, 1500 steps on the
closed-form problems, 1200 on the network. `--quick` cuts that to 3 seeds and
a third of the steps; it is a smoke test, and its numbers are noisier and will
not match these.

`Research/SineWaveActivationFunction.md` replaces the sigmoid with a wave, on the
argument that a sigmoid already *is* one — one single step of one, with
everything before and after thrown away. This applies the same move one level up,
to the optimiser instead of the activation.

## The idea

Adam's update is `m / sqrt(v)`: a first moment over a root second moment. That
ratio is a **signal-to-noise ratio** — near `±1` when the recent gradients all
point the same way, near `0` when they cancel. So Adam's step magnitude is
already a saturating, odd, bounded function of consistency: a sigmoid-shaped
response in disguise, clipped at `±1` and flat forever after.

So take two of them, at two different timescales, and let them interfere.

Three EMAs per parameter instead of Adam's two:

| state | rate | memory |
|---|---|---|
| `m_f` fast first moment | `0.9` | ~10 steps |
| `m_s` slow first moment | `0.99` | ~100 steps |
| `v` second moment | `0.999` | ~1000 steps — the shared scale |

Two ratios come out, both measured against the same `sqrt(v)`, and each maps to
a phase where saturated agreement is a quarter turn:

```
r_f = m_f / (sqrt(v) + eps)          phi_f = (pi/2) * clip(r_f, -1, 1)
r_s = m_s / (sqrt(v) + eps)          phi_s = (pi/2) * clip(r_s, -1, 1)
```

The step is the mean of the two waves — and that is where two sine waves become
one product, by an identity rather than by a choice:

```
step = (sin(phi_f) + sin(phi_s)) / 2
     = sin((phi_f + phi_s) / 2) * cos((phi_f - phi_s) / 2)
       \_______carrier________/   \_______envelope_______/
```

**This is a beat.** Two waves of nearby frequency sum to a carrier at their mean
modulated by an envelope at their difference; it is why two mistuned piano
strings throb. Here the two frequencies are the two timescales, and the beat
splits the update into the two questions an optimiser actually has:

- **carrier** — which way, and how sure, averaged over both timescales.
- **envelope** — how far the two timescales *agree*. It is `1` when they say the
  same thing and falls to `0` when they are fully opposed, which is the state of
  a parameter that has just reversed against its own trend: the signature of
  bouncing across a ravine rather than descending it.

Because `|phi_f - phi_s| <= pi` the envelope lies in `[0, 1]` and can only ever
brake. Nothing here is a schedule — the brake is read off the gradients, bites
after an overshoot and lets go once the timescales re-agree.

For small `r`, `sin((pi/2) r) -> (pi/2) r`, so the whole thing collapses to
`(pi/2) * (r_f + r_s)/2`: Adam's rule on the mean of two moments. **TwoSine is a
strict generalisation of Adam**, departing from it only where the SNR is large
enough for the curvature of the wave to matter.

**What it costs:** one extra EMA per parameter. Three slots against Adam's two,
+50% optimiser state. That is the price, and the tables below are what it buys.

## What is tested

Three problems — an ill-conditioned quadratic (κ=1000, 20-D), Rosenbrock, and a
`2 -> 8 -> 1` network whose hidden units are the repository's own
`a·sin(b(x−h))+k` with all four learnable per unit, trained on mini-batches.
The network is deliberately the awkward bed: §7 of the source paper predicts the
periodicity gives that loss surface many local minima and that the optimiser will
have to fight them.

Every arm is swept over the same 8 learning rates from `3.0` to `0.001` and
reported at its own best, because TwoSine's step is `pi/2` times Adam's near zero
and a shared learning rate would measure that constant instead of the idea.
Closed-form problems are scored at the **last** iterate, not the best one seen —
an arm that overshoots violently passes near the optimum on the way through, and
that is the behaviour being detected, not rewarded.

**Four controls, each of which can sink the claim.** All hold the state and the
timescales fixed and change only how the ratios combine, so anything that
separates them is the combining rule and not the extra memory:

| control | what it would prove if it ties |
|---|---|
| `two-ema-linear` | the same three EMAs, averaged with no sine anywhere → the third EMA did the work, the waves did nothing |
| `one-sine` | one wave, fast ratio only → the second wave did nothing |
| `sign-gate` | fast ratio, hard-zeroed when the moments disagree in sign → an `if` would have done |
| `slow-adam` | Adam with `beta1 = 0.99` → it is just a longer memory |

## Results

Paired over all 120 `(problem, learning rate, seed)` cells — the same seed gives
every arm the same initial weights and the same batch order, so each cell is a
matched pair. A win needs a factor of 1.25; `p` is a two-sided sign test on the
decisive cells.

| TwoSine against | win | loss | tie | reading |
|---|---|---|---|---|
| `adam` | 57 | 36 | 27 | **TwoSine ahead**, p = 0.038 |
| `two-ema-linear` | 57 | 35 | 28 | **TwoSine ahead**, p = 0.028 |
| `one-sine` | 47 | 41 | 32 | not distinguished, p = 0.594 |
| `sign-gate` | 53 | 39 | 28 | not distinguished, p = 0.175 |
| `slow-adam` | 79 | 17 | 24 | TwoSine ahead, p < 0.001 |
| `two-sine-full` | 4 | 3 | 113 | not distinguished, p = 1.000 |

Final MSE on the network, each arm at its own best learning rate, 5 seeds:

| arm | lr | mean MSE | std | worst seed |
|---|---|---|---|---|
| `slow-adam` | 0.1 | **0.000341** | 0.000242 | 0.000814 |
| `two-sine` | 0.1 | 0.001743 | 0.000837 | 0.002776 |
| `sign-gate` | 0.1 | 0.002549 | 0.000851 | 0.003709 |
| `two-sine-full` | 0.03 | 0.003274 | 0.006349 | 0.015971 |
| `adam` | 0.1 | 0.003835 | 0.002621 | 0.007857 |
| `two-ema-linear` | 0.03 | 0.004187 | 0.008146 | 0.020478 |
| `one-sine` | 0.1 | 0.008349 | 0.005351 | 0.014058 |

Learning rates out of 8 that still reach the target, summed over the three
problems — an optimiser needing less tuning:

| arm | quadratic | rosenbrock | network | total |
|---|---|---|---|---|
| `two-sine` | 7/8 | 4/8 | 3/8 | **14/24** |
| `two-ema-linear` | 7/8 | 5/8 | 2/8 | **14/24** |
| `sign-gate` | 5/8 | 4/8 | 2/8 | 11/24 |
| `slow-adam` | 6/8 | 3/8 | 2/8 | 11/24 |
| `adam` | 5/8 | 4/8 | 1/8 | 10/24 |
| `one-sine` | 5/8 | 3/8 | 2/8 | 10/24 |

The brake is real but rare: on the network the envelope averages `0.999`, drops
as low as `0.750`, and is engaged — below `0.99` — on **2.3%** of
(parameter, step) pairs. Transient, as designed. If that number were `0` the
optimiser would be Adam with extra state and the whole construction would be
decoration.

## The honest reading

**What survives.** Reading the two moments as interfering waves beats Adam
(p = 0.038) and beats *the same three EMAs combined linearly* (p = 0.028). That
second one is the result that matters: the third EMA is not what is doing the
work, the interference is. On the network it more than halves Adam's error
(0.00174 vs 0.00384) at the same learning rate, with a better worst seed.

**What does not.** The evidence does not separate TwoSine from `one-sine`
(p = 0.594) or from `sign-gate` (p = 0.175). Two of the four controls are
therefore unbeaten, and the honest version of the claim is "beats Adam and beats
its own linear ablation", not "beats everything". Both p-values that do clear
0.05 clear it thinly, and the sign test is optimistic anyway — cells at different
learning rates on one problem and seed share a trajectory and are not independent.

**Where it loses outright.** `slow-adam` — plain Adam with `beta1 = 0.99`, *fewer*
state slots than TwoSine — gets 0.000341 on the network, five times better than
TwoSine's best. It is a fragile win (the same arm is the worst on the grid at
`lr = 1.0`, and it loses 17–79 paired) but it is a win, and on this task a
one-line change to Adam beat the whole construction. On band width TwoSine ties
`two-ema-linear` at 14/24 rather than beating it; the paired test is the only
place the two separate.

**The unclipped wave.** `clip(r, -1, 1)` throws away everything past the first
quarter of the wave, which is exactly what the source paper argues against, so
`full_wave=True` removes it. It changes nothing in 113 of 120 cells — `|r| > 1`
is rare enough that the clip almost never binds. Where it does bind it is worse,
and for a reason worth recording: without the clip `|phi_f - phi_s|` can exceed
`pi`, so the envelope goes **negative** (measured minimum `-0.55`) and *reverses*
the step instead of braking it. For this particular application the rest of the
wave is not being thrown away — there is nothing out there to keep.

**What this is not.** Three toy problems, at most 20 dimensions, one architecture,
5 seeds, no deep network, no real dataset, no wall-clock or memory measurement
beyond the `+50%` state count. Nothing here says TwoSine scales.

---

## Related

| claim | where it is tested |
|---|---|
| the sine activation beats sigmoid and ReLU | `../ActivationFunctionTest/` |
| a learnable activation inside a conventional RNN | `../SBNN_RNN_ActivationFunction/main.py` |
| the vanishing gradient is a measurement | `../DepthCountedNormalisation/` |
