# 2 popular activation functions VS my activation function

In this directory, I provided 3 different python scripts to test two popular activation functions and my own activation function from my research.

The improvement is astounding with a ~200% to ~400% improvement in reward.

- Sigmoid: 9
- ReLU: 10
- Sinewave: 28-48


As you can tell, I don't write much about my research, I like to keep things simple and to the point without a lot of extra langauge and fluff.
---

## Self-building network test

`self_building_sinewave.py`. Standard library only — no TensorFlow, no PyTorch,
no gym — so it runs anywhere with nothing installed, the same rule
`RadixCyclicNN` and `Experiments/DepthCountedNormalisation` follow.

```
python3 self_building_sinewave.py            # full run, ~2 min
python3 self_building_sinewave.py --quick    # one seed, ~6 s
python3 self_building_sinewave.py --check    # gradient checks only
```

A self-building network starts too small and grows its hidden layer whenever
learning stalls — the rule in `Experiments/SBNN_RNN_ActivationFunction/main.py`. That turns
a question about an activation function into a question you can count:

> **how many units does the network have to build, and how many of the ones it
> built are still alive when it stops?**

Those are different numbers, and the gap between them is the point. A unit whose
derivative has gone to zero everywhere on the data is a unit the network paid to
build and can no longer train. Growth cannot recover it — growth adds *new*
units next to the dead one. So a self-building network wearing a saturating
activation does not merely learn slower: it builds capacity it cannot use,
stalls again, and builds more.

Five arms, identical in every respect except the shape of one unit. `1 -> H -> 1`,
linear output head, Adam, 5 seeds, 3000 epochs, `H` starting at 4 and growing by
4 per stall up to 48.

| arm | unit | learnable per unit |
|---|---|---|
| `sigmoid` | `1/(1+exp(-(z-h)))` | `h` |
| `tanh` | `tanh(z-h)` | `h` |
| `relu` | `max(0, z-h)` | `h` |
| `sine-fixed` | `-sin((z-h)/3)` | `h` |
| `sine` | `a*sin(b*(z-h)) + k` | `a, b, h, k` |

`tanh` is there because section 9 of the paper asks for it by name. `sine-fixed`
is the parameter-matched control — exactly the baselines' knob count — so the
`sine` vs `sine-fixed` gap is what *learning the activation* is worth, with the
shape held fixed.

### Results

`deaf` is the section 6 dead fraction — the share of (unit, sample) pairs with
`|df/dz| < 0.01` — except measured on the pre-activations the network actually
produced instead of over a range chosen by hand. `dead` is a unit below that
threshold at *every* training input: built, then lost.

**chirp** — a swept sine, solved at train MSE ≤ 1e-3:

| arm | test MSE | built | dead | deaf | units to target |
|---|---|---|---|---|---|
| sigmoid | 3.78e-05 ± 2.21e-05 | 7.2 ± 1.6 | 0.0 | 45.1% | 7.2 |
| tanh | 1.57e-04 ± 2.50e-04 | 4.8 ± 1.6 | 0.0 | 30.8% | 4.8 |
| relu | 8.27e-04 ± 3.02e-04 | 31.2 ± 13.9 | **17.8 ± 12.0** | 79.6% | 27.0 (4/5 seeds) |
| sine-fixed | 1.49e-03 ± 5.17e-04 | 48.0 | 0.0 | 0.4% | never |
| **sine** | **1.24e-04 ± 4.88e-05** | **6.4 ± 2.0** | **0.0** | **1.1%** | **6.4** |

**spline** — a smooth curve through fixed random knots, ≤ 5e-3:

| arm | test MSE | built | dead | deaf | units to target |
|---|---|---|---|---|---|
| sigmoid | 4.03e-04 ± 3.29e-04 | 13.6 ± 4.1 | 0.2 ± 0.4 | 46.6% | 13.6 |
| tanh | 1.04e-03 ± 6.58e-04 | 5.6 ± 2.0 | 0.0 | 44.3% | 5.6 |
| relu | 1.47e-03 ± 9.07e-04 | 16.8 ± 3.0 | 3.4 ± 2.1 | 71.1% | 16.8 |
| sine-fixed | 5.64e-03 ± 6.73e-05 | 48.0 | 0.0 | 0.6% | never |
| sine | 5.80e-03 ± 4.80e-04 | 48.0 | 0.2 ± 0.4 | 0.6% | never |

**steps** — a staircase, ≤ 1e-2:

| arm | test MSE | built | dead | deaf | units to target |
|---|---|---|---|---|---|
| sigmoid | 7.42e-03 ± 6.28e-05 | 24.8 ± 4.7 | 0.0 | 28.5% | 24.8 |
| tanh | 7.04e-03 ± 2.44e-04 | 12.8 ± 4.7 | 0.0 | 36.6% | 12.8 |
| relu | **6.59e-03 ± 3.03e-04** | 9.6 ± 2.0 | 2.6 ± 1.0 | 60.7% | 9.6 |
| sine-fixed | 2.78e-02 ± 1.29e-02 | 48.0 | 0.0 | 0.5% | never |
| sine | 1.91e-02 ± 1.57e-03 | 48.0 | 0.0 | 0.2% | never |

### What came out of it

**1. Section 6 holds, measured rather than asserted.** The sine's dead fraction
is 0.2–1.1% of the pre-activations the network really produced, against 28–47%
for sigmoid and tanh and 61–80% for ReLU. Across the 30 sine runs one single unit
died, on one seed of `spline`. The claim was made about a function on its own;
it survives being put inside a network that trains.

**2. A dying activation is much more expensive when the network can grow.** On
`chirp` ReLU built 31.2 units and 17.8 of them were dead at the end — **57% of
what it built**. That is the whole failure loop in one number: units die, the
loss stalls, the stall triggers growth, the new units die too. Half of ReLU's
`deaf` is its own gating and not a pathology; the `dead` column is the part that
cannot be undone.

**3. Learning the activation is where the win is — not the shape.** `sine`
against `sine-fixed` holds the function, the initialisation and the seed fixed
and changes only whether `a, b, k` may move. On `chirp`: **1.24e-04 with 6.4
units against 1.49e-03 at the 48-unit cap** — 12× the accuracy from 7.5× fewer
units, which is 38 parameters against 144. Run with `--act-lr 0.0` and the two
arms print identical numbers to the last digit, which is the check that nothing
else differs between them.

**4. The tenth in section 8 is load-bearing, and there is a failure mode the
paper does not name.** At `--act-lr 1.0` — the activation learning as fast as
the weights — the sine arm loses 13.2 of 48 units on `spline` and 12.2 on
`steps`, and `deaf` jumps from 0.6% to 28.1%. The mechanism is in the learned
parameters and it is exact: every dead sine unit in that run has `|a*b| <= 0.0100`
against a live median of 0.067, and since `|f'| = |a*b*cos(u)| <= |a*b|`, a unit
whose amplitude-frequency product has collapsed under the threshold is below it
at *every* input by construction. **The sine unit cannot die along `x`, but it
can die along `a` and `b`.** Section 6's liveness is a statement about the input axis only; the
tenth-rate update rule is what protects the other axis, and it earns its place.

**5. Where it loses: the staircase.** ReLU reaches 6.59e-03 with 9.6 units;
the sine sits at 1.91e-02 against the 48-unit cap and never gets there. The
mechanism is locality. A sigmoid-family unit is a *local* basis function — one
transition, at one place — so a target made of separated features decomposes
neatly across units. A sine unit is *global*: every unit speaks across the whole
domain, and they interfere. Section 3 says a sigmoid is one step of a rotated
sine wave, which makes `steps` a task built out of the baselines' own shape and
their home ground. `chirp`, where the structure is periodic, goes the other way,
and `spline` — which suits neither — leaves the sine 14× behind the best
baseline (4× behind ReLU) with a deaf fraction 75× lower.

**6. It is not the initialisation.** `--init shared` puts every arm on the
sine's own `Z_RANGE = 4.5` draw instead of its native He/Xavier one. The
baselines still go deaf at 35–69%, and the sine arms do not move. The liveness
difference is the shape of the function, not the draw.

### What it does not establish

One hidden layer, 1-D regression, one optimiser, five seeds, three targets I
chose. Nothing here says anything about depth, and it says nothing about the
CartPole numbers above — a different task, a different architecture, a different
failure mode. The script prints its own gradient check (max error 1.3e-09) before
every run, because the numbers are only worth reading if the derivatives are
right.
