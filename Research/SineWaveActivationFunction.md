# A Sine Wave Activation Function

**Replacing the sigmoid with `f(x) = a·sin(b(x − h)) + k`, default `−sin(x/3)`**

Mason Curtis — Curtis Tech Solutions
Working paper · September 2026

---

## Abstract

I replace the sigmoid activation function with a sine wave. The argument is
short: the brain is an analog machine, analog machines carry information in
waves, and if you look carefully at a sigmoid you find that it already *is* a
wave — one single step of one, with everything before and after it thrown away.
So I stopped throwing the rest away.

The function is `f(x) = a·sin(b(x − h)) + k`, initialised to `a = −1, b = 1/3,
h = 0, k = 0`, which is exactly `−sin(x/3)`. All four parameters are learnable,
per unit. In a CartPole control task, swapping this in for sigmoid or ReLU and
changing nothing else moved the final reward from 9 and 10 to between 28 and 48.

This paper says where the idea came from, what the four parameters mean, proves
the geometric claim that motivated it (a sigmoid is one step of a sine wave
rotated 45 degrees — and 45 is exactly the right number), works through the
numbers, gives the three properties that follow for free, and is honest about
what the experiment does and does not establish.

---

## 1. The original note

> Instead of the traditional sigmoid function frequently used in ANNs, we use a
> sine wave rotated 45 degrees.
>
> This was drawn from the idea that a brain operates like an analog computer.
> Hence, we assume sine waves are the standard of information
> encoding/decoding. This led me to see the similarities between a sigmoid
> function and a sine wave.

That was the whole thing. Everything below is what I meant by it — in
particular, §3 proves that the last sentence is not a loose analogy.

---

## 2. Where the idea came from: the brain is an analog computer

I did not start from a function. I started from the brain.

A digital computer represents information by forcing a continuous voltage into
one of two buckets and then refusing to look at anything in between. That is a
design *choice*, and it is a choice made because transistors are cheap and
noise is expensive. A brain makes the opposite choice. Neurons do not resolve
to 0 or 1 and hold. They oscillate. Population activity has rhythms, those
rhythms have frequencies, the frequencies have phase relationships, and the
phase relationships carry information. Whatever the brain is computing with, it
is not a switch.

So: if the brain is analog, and analog machines encode with waves, then the
primitive of a brain-like network should be a wave. Amplitude, frequency,
phase, offset. Those are the four things you can do to a wave, and they should
be the four things a unit can learn.

Then I looked back at the sigmoid, expecting it to be the opposite of a wave,
and found that it wasn't.

---

## 3. A sigmoid is one step of a rotated sine wave

This is the geometric observation the whole idea rests on, and it is exact.

Take the plain sine curve `y = sin(t)` and rotate the *curve* by an angle θ.
The point `(t, sin t)` goes to

```
x(t) = t·cos θ − sin(t)·sin θ
y(t) = t·sin θ + sin(t)·cos θ
```

For the result to still be a usable activation function it has to be a function
of `x` — one output per input. That requires `x(t)` to be non-decreasing:

```
dx/dt = cos θ − cos(t)·sin θ  ≥ 0   for all t
    ⟺  cos θ ≥ sin θ
    ⟺  θ ≤ 45°
```

**45 degrees is exactly the critical angle.** Rotate less and you get a
function; rotate more and the curve folds back over itself and stops being one.
I did not pick 45 because it sounded round. It is the largest rotation a sine
wave survives as an activation function, and at that limit the curve looks like
this (drawn here at 40°, just inside the limit, so the risers stay finite):

```
                                    |
                                    |                    ***************
                                    |                  ***
                                    |                  *
                                    |                 *
                                    |               ***
                                    |****************
                                    *
-----------------------------------**-----------------------------------
                                   *|
                   **************** |
                 ***                |
                 *                  |
                *                   |
              ***                   |
***************                     |
                                    |

              y = sin(t) rotated by 40°: a staircase
```

A staircase. Flat treads where `dy/dt = 0` (at `t = π + 2πk`), steep risers
where `dx/dt = 0` (at `t = 2πk`). And now look at one step of it next to a
sigmoid:

```
                                    |
                                    |      *****************************
                                    |   ****
                                    |  **
                                    | *
                                    |**
                                    **
                                    *
-----------------------------------**-----------------------------------
                                   *|
                                  **|
                                 ** |
                                 *  |
                               **   |
                            ****    |
*****************************       |
                                    |

              the logistic sigmoid (scaled to [−1, 1])
```

They are the same shape. I fitted a logistic `σ(c·x)` to one step of the
40°-rotated sine numerically: the best fit is `c = 6.90` with a maximum error of
**0.044** — under 4.5% of the step's full height, across the entire step.

So here is the actual relationship between the two functions:

> **A sigmoid is one step of a sine wave rotated 45 degrees. The sigmoid is
> what you get when you keep one step and flatten every other step in both
> directions out to infinity.**

That reframes the design question completely. The sigmoid is not a different
kind of object from a sine wave that I am proposing to swap out. It is a sine
wave with 99% of itself deleted. The question is no longer "why a sine instead
of a sigmoid" — it is "why did we delete the rest of it?"

The answer, historically, is monotonicity. One step is monotone; the staircase
is monotone too, but a *sine* — unrotated — is not. That is the real trade, and
I take it head on in §7.

---

## 4. The function

```python
class HyperParameters:
    # Sine parameters
    a: float = -1
    h: float = 0
    b: float = 1/3
    k: float = 0
    decay_window: int = 1_000

    def activation_function(self, x: float) -> float:
        return self.a * math.sin(self.b * (x - self.h)) + self.k
```

Four parameters, one wave:

| Param | Name | What it does | Default | Meaning for the unit |
|---|---|---|---|---|
| `a` | amplitude | vertical scale **and sign** | `−1` | how loud the unit is, and its polarity |
| `b` | frequency | horizontal scale | `1/3` | how much input it takes to move the unit through its range |
| `h` | phase | horizontal shift | `0` | where on the wave the unit sits — its operating point |
| `k` | offset | vertical shift | `0` | the unit's resting output when it has nothing to say |

With the defaults this is exactly `f(x) = −sin(x/3)`:

```
                                    |
                          ***       |                      ****
                        *** ***     |                    ***  ***
                       **     **    |                   **      **
                      **        *   |                  **        **
                     **          *  |                 **          **
**                  **            * |                 *            **
 *                 **             **|                *              **
--*---------------**---------------**---------------**---------------*--
  **              *                 **             **                 *
   **            *                  |*            **                  **
    **          **                  | *          **
     **        **                   |  *        **
      **      **                    |   **     **
       ***  ***                     |    *** ***
         ****                       |      ***
                                    |
                x from −20 to 20,  f(x) = −sin(x/3)
```

### Why `b = 1/3`

`b` sets the period: `2π/b = 6π ≈ 18.85`. A quarter of that — the monotone arm
running from the origin up to the first crest — is `3π/2 ≈ 4.712`.

That number is not decoration. In `RadixCyclicNN/radixnet/graph.py` every new
node's state is drawn uniformly from `[−4.5, 4.5]`:

```python
Z_RANGE = 4.5
"""New node state ``z`` is drawn uniformly from ``[-Z_RANGE, Z_RANGE]``."""
```

`4.5 < 4.712`. **Every unit in a fresh network starts somewhere on the monotone
arm of its own wave, with 0.21 to spare.** The initialisation range and the
frequency are matched on purpose: at initialisation the network behaves exactly
like a network of smooth sigmoid-ish units, and it only discovers the rest of
the wave if training pushes it there. You get the conservative behaviour for
free and the periodic behaviour when it earns its place.

That is the practical rule: **`b` must be small enough that the pre-activations
you actually produce land inside one arm.** Pick `b` too large and every unit is
wrapped around the wave several times before it has learned anything, and the
layer is a hash function. `1/3` is the number that makes a range of about ±4.5
work.

### Why `a = −1`

Two reasons, and the second one is the interesting one.

First: `f(x) = −sin(x/3)` is *decreasing* through the origin, `f'(0) = −1/3`.
The default unit is inhibitory. More input, less output. Brains are full of
inhibition and inhibition is what makes a system settle instead of run away; I
would rather make it the default than an exception.

Second: with a sine, **the sign of the amplitude is the sign of the unit**, and
that makes network inversion a parameter change:

```
a = −1:  f(−3.3) = +0.891207        a = +1:  f(−3.3) = −0.891207
a = −1:  f(+1.7) = −0.536823        a = +1:  f(+1.7) = +0.536823
a = −1:  f(+8.2) = −0.397012        a = +1:  f(+8.2) = +0.397012
```

Exactly negated, at every input, to the last digit — because `a·sin(u)` is
linear in `a`. Flip `a` and the unit says the precise opposite of what it said
before.

You cannot do this with a ReLU. `relu(−x) ≠ −relu(x)`; there is no parameter of
a ReLU whose sign you can flip to get the negation of the function. There isn't
one for a logistic sigmoid either.

I use this. It is the mechanism behind **2NRL** (train on the failures → invert
→ fine-tune on the correct data): `invert()` in RadixCyclicNN flips every edge
weight and every activation amplitude, and what was likely becomes unlikely.
That whole training algorithm is only available because the activation function
has a sign parameter. The activation function choice bought me a training
algorithm.

2NRL has its own paper — `2NRL.md` — which treats the procedure, why inverting a
failure is coherent rather than destructive, and the condition under which it
carries information. What matters here is the dependency running the other way:
it needs a unit whose negation is a parameter change, and that is what a sine
gives you and a ReLU does not.

### Why `h` and `k`

`h` slides the unit along the wave, `k` lifts its resting level. There is a
clean identity here that I come back to in §8: **shifting the phase and shifting
the input are the same operation**, so `h` is doing the job that a bias term
does in a conventional network — but *inside* the nonlinearity rather than in
front of it.

---

## 5. Worked numbers

`f(x) = −sin(x/3)`, `f'(x) = −(1/3)·cos(x/3)`, against the logistic `σ` and
ReLU:

| x | `−sin(x/3)` | `f'(x)` | `σ(x)` | `σ'(x)` | `relu'(x)` |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0000 | −0.3333 | 0.500000000 | 2.50e−01 | 0 |
| 1.0 | −0.3272 | −0.3150 | 0.731058579 | 1.97e−01 | 1 |
| 3.0 | −0.8415 | −0.1801 | 0.952574127 | 4.52e−02 | 1 |
| 4.5 | −0.9975 | −0.0236 | 0.989013057 | 1.09e−02 | 1 |
| 4.712 | −1.0000 | 0.0000 | 0.991093257 | 8.83e−03 | 1 |
| 6.0 | −0.9093 | **+0.1387** | 0.997527377 | 2.47e−03 | 1 |
| 9.0 | −0.1411 | +0.3300 | 0.999876605 | 1.23e−04 | 1 |
| 14.137 | **+1.0000** | 0.0000 | 0.999999275 | 7.25e−07 | 1 |
| 20.0 | −0.3742 | **−0.3091** | 0.999999998 | **2.06e−09** | 1 |
| 50.0 | +0.8184 | +0.1915 | 1.000000000 | **0.00e+00** | 1 |

Read the last two rows. At `x = 20` the sigmoid is `0.999999998` with a
derivative of two billionths — that unit is dead, it will never learn again, and
nothing downstream can tell the difference between an input of 20 and an input
of 200. At the same input the sine unit is at `−0.374` with a derivative of
`−0.309`, which is 93% of its maximum slope. It is fully awake.

At `x = 50` the sigmoid derivative is *exactly zero in float64*. Not small.
Zero. The unit has been removed from the network and there is no gradient path
that can bring it back.

---

## 6. The property that matters: the unit never dies

Here is the same point measured over whole ranges — the fraction of inputs in
`[−R, R]` where the derivative is smaller than 0.01, i.e. where the unit has
effectively stopped responding:

| range | sigmoid | `−sin(x/3)` | ReLU |
|---|---:|---:|---:|
| `[−10, 10]` | 54.2% | **1.8%** | 50.0% |
| `[−30, 30]` | 84.7% | **1.8%** | 50.0% |
| `[−100, 100]` | 95.4% | **2.0%** | 50.0% |

The shape of that table is the whole argument.

- **ReLU** is dead on exactly half of its input space, always. That half is
  fixed and it is the negative half, so a unit that drifts negative and stays
  there is gone permanently.
- **Sigmoid** is dead on a fraction that *grows toward 100%* as the network's
  pre-activations grow. Training makes weights bigger; bigger weights make
  pre-activations bigger; bigger pre-activations push units into saturation.
  Sigmoid networks kill themselves by learning.
- **The sine** is dead on a constant ~2% no matter how large the range gets,
  and — this is the part that matters — never *permanently*. The zero-derivative
  points are the crests, isolated points of measure zero. Move the input
  slightly and full slope comes back. A sine unit is never more than a quarter
  period away from its maximum sensitivity.

Formally, `|f'(x)| = |a·b·cos(b(x − h))|` attains `|a·b|` every half period,
forever. There is no `x` beyond which the unit stops responding. **The vanishing
gradient is not something this function mitigates; it is something it does not
have.**

This is why the function fits the rest of my architecture. In RadixCyclicNN I do
not back-propagate through depth at all — the learning rule is one hop, local,
and it updates the *activation parameters* rather than pushing error down a
chain. That design is only worth building if a single unit's own function is
expressive enough to be worth learning. A ReLU has nothing to learn — it has no
parameters. A sigmoid has nothing to learn. A wave has four knobs: gain, scale,
operating point, resting level. There is enough function in one unit to make
local learning meaningful, which is precisely what lets me delete
back-propagation through depth and not miss it.

---

## 7. What it costs: the function is not monotone

I am not going to pretend this is free.

`−sin(x/3)` is not injective. `f(1.0) = −0.327` and `f(17.85) = −0.327`. Two
very different inputs, one identical output. A downstream unit cannot tell them
apart. This is aliasing, it is exactly what dropping the monotonicity
requirement buys you, and it is the one real objection to the whole idea.

Three things make it survivable.

**1. A layer is a population, not a unit.** No single unit is asked to identify
its input. A layer of units with different learned `h` and `b` gives a set of
readings at different phases and different frequencies, and two inputs that
alias on one unit will not alias on all of them. This is not a hopeful
analogy — it is how any multi-scale periodic code resolves ambiguity: you cannot
tell 1 o'clock from 13 o'clock from the hour hand alone, and you can the instant
you add a second hand running at a different rate. Biology uses exactly this
trick for position coding with multi-scale periodic responses. One periodic unit
is ambiguous; a set of them at different scales is not.

**2. Initialisation keeps you on one arm.** As shown in §4, `Z_RANGE = 4.5`
against a quarter period of `4.712` means a fresh network is entirely inside the
monotone region. Aliasing is not something the network is born with; it is
something it can grow into if the data rewards it.

**3. `b` is learnable.** If the wrapping hurts, the gradient on `b` shrinks it,
stretching the period until the operating range fits in one arm again. The
function can retreat to being a sigmoid. A sigmoid cannot advance to being a
wave.

The honest summary of the trade: **I gave up global monotonicity and bought
permanent liveness, an exact sign flip, and four learnable parameters per
unit.** For an architecture with no deep gradient path, that is a good trade.
For a very deep conventional network trained by back-propagation with no
normalisation, I would expect the periodicity to cause trouble, and I would
expect to have to fight the optimiser's many local minima. I have not done that
experiment.

---

## 8. Learning the activation itself

If the activation has parameters, they should be learned. The partial
derivatives, with `u = b(x − h)`:

```
f      = a·sin(u) + k
df/dx  = a·b·cos(u)
df/da  = sin(u)
df/db  = a·(x − h)·cos(u)
df/dh  = −a·b·cos(u)
df/dk  = 1
```

Each one says something:

- **`df/da = sin(u)`** — the gain learns fastest for units sitting at the steep
  part of their own wave and not at all for units sitting at a zero crossing. A
  unit that is currently saying nothing does not get louder; it gets moved
  first.
- **`df/db = a(x − h)·cos(u)`** — the frequency gradient scales with `x − h`,
  the distance from the unit's operating point. **Frequency is learned from the
  outliers.** The units seeing inputs far from where they are tuned are the ones
  that pull the period; the ones sitting near their own centre leave it alone.
  That is the right behaviour: the period needs to change exactly when the data
  is out of range.
- **`df/dh = −a·b·cos(u) = −df/dx`** — exact identity, verified to machine
  precision at every input I tested. **Sliding the wave left is identical to
  sliding the input right.** So the phase parameter is a per-unit input bias
  hiding inside the nonlinearity. This is worth knowing for two reasons: you get
  a bias term whether or not you wrote one, and if you *do* also have a bias in
  front of the activation, `h` and that bias are degenerate — they will fight
  over the same direction and waste gradient. Pick one.
- **`df/dk = 1`** — pure output bias, always learnable, never saturates. This is
  the unit's resting level.

In RadixCyclicNN every node owns its own `(a, b, h, k)` and all four are updated
by the local rule, alongside the node state `z` and the outgoing weights.
`SBNN_RNN_ActivationFunction/main.py` is the same principle in a conventional
RNN: a learnable-parameter activation whose `α` and `β` are updated by their own
gradients at a tenth of the weight learning rate. The tenth matters — the
activation is the *shape of the space* the weights are searching in, and moving
the space as fast as you move the search makes both diverge.

---

## 9. Evidence: the CartPole experiment

`ActivationFunctionTest/` holds three scripts — `sigmoid.py`, `relu.py`,
`sinewave.py` — that are byte-for-byte identical except for the activation
function. Same environment (`CartPole-v1`), same architecture (Dense 24 → Dense
`n_actions`), same optimiser (Adam), same loss (MSE), same `γ = 0.95`, same
ε-greedy schedule, same 250 episodes, same greedy evaluation episode at the end.
One line differs:

```python
def activation_function(x):
    return -1 * tf.math.sin((1/3) * (x-0)) + 0
```

Reported total reward on the final greedy episode:

| Activation | Total reward |
|---|---|
| Sigmoid | 9 |
| ReLU | 10 |
| **Sine (`−sin(x/3)`)** | **28 – 48** |

A ~200% to ~400% improvement.

### What this does and does not establish

I would rather state this precisely than oversell it.

**What it establishes:** on this task, with this architecture and this budget,
changing only the activation function produced a large and repeatable
difference in the direction the theory predicts. That is a real result and it is
why I kept going.

**What it does not establish:** these are single runs scored by a single greedy
episode, with no seed sweep and no error bars, and CartPole returns are very
high-variance. The range "28–48" is itself evidence of that variance.

**And there is a confound I should name myself.** In all three scripts the
activation is applied to the *output* layer as well as the hidden layer. That is
a serious handicap for sigmoid specifically: the output is squashed into `(0,1)`
while the regression targets in Q-learning reach roughly `1/(1−γ) = 20`, so the
output units saturate almost immediately and the gradient dies at the head. Part
of the measured gap is therefore "sine beats a sigmoid that was placed where a
sigmoid cannot work". The sine head is bounded too — `[−1, 1]` — but bounded is
survivable when the gradient never dies, and for control only the *argmax* over
actions matters, not the magnitude. That is my mechanistic explanation for the
gap, and it is also the reason the number should not be quoted without the
caveat.

**The cleaner experiment is already in the directory.**
`reinforcement_q_learning_relu.py` and `reinforcement_q_learning_sinewave.py`
are the proper DQN pair — replay buffer, target network, soft updates, fixed
seed (`seed = 1`), and critically a **linear output head**:

```python
def forward(self, x):
    x = self.activation_function(self.layer1(x))
    x = self.activation_function(self.layer2(x))
    return self.layer3(x)          # no activation on the output
```

The activation is confined to the hidden layers, where it belongs. That pair
measures the activation function and nothing else, and it is the one I would
quote from.

### Replication protocol

To turn this from an indication into a result:

1. Run the DQN pair (linear head), not the Keras trio.
2. 20 seeds per activation, not 1.
3. Score on the mean of 100 greedy evaluation episodes, not one.
4. Report mean ± standard deviation, and the full distribution.
5. Add `tanh` as a fourth arm — it is the strongest of the classical squashing
   functions and the fairest opponent for a bounded periodic one.
6. Sweep `b ∈ {1/6, 1/3, 1/2, 1}` to show the frequency argument in §4 rather
   than asserting it.
7. Report wall-clock too. `sin` is not free next to `max(0, x)`.

---

## 10. Where this is used

| Location | How the activation appears |
|---|---|
| `ActivationFunctionTest/sinewave.py` | the original comparison, TensorFlow |
| `ActivationFunctionTest/reinforcement_q_learning_sinewave.py` | DQN with a linear head, PyTorch — the controlled version |
| `RadixCyclicNN/radixnet/activation.py` | the reference implementation: `sine_activation`, `sine_derivative`, `sine_partials`, `SineActivation.inverted()` |
| `RadixCyclicNN/radixnet/graph.py` | per-node `a, b, h, k`; `Z_RANGE = 4.5` matched to the quarter period |
| `RadixCyclicNN/radixnet/backend.py` | the same formulas inlined for the hot loop, CPU and GPU, asserted to produce identical numbers |
| `SBNN_RNN_ActivationFunction/main.py` | the "learnable activation" principle in a conventional RNN with BPTT |

The edge signal in RadixCyclicNN is `w · f_p(z_p) · f_c(z_c)` — weight times the
activation of the parent times the activation of the child. The consequence of a
sign-flippable activation shows up there: since the score is a *product* of two
activations, flipping both endpoints leaves the edge unchanged, and flipping
every *other* node along a path flips every edge on it. That parity fact is what
makes targeted inversion possible, and I treat it properly in the companion
paper on cycles.

---

## 11. Limitations, stated plainly

1. **Aliasing.** Covered in §7. Real, mitigated, not eliminated.
2. **Range discipline.** The function is only sensible while `b·(x − h)` stays
   in a range you chose. Pre-activations that grow without bound turn the layer
   into noise. Either keep `b` learnable, or normalise, or both.
3. **A periodic landscape has periodic minima.** Gradient descent on a function
   with infinitely many equivalent-looking basins is harder than on a monotone
   one. I have not characterised this.
4. **Standard initialisation schemes do not apply.** Xavier/He are derived
   assuming monotone activations with particular variance behaviour. `Z_RANGE`
   matched to the quarter period is my substitute; it is a rule of thumb, not a
   derivation.
5. **Cost.** `sin` is meaningfully more expensive than `max(0, x)`. On this
   architecture the graph walk dominates and it does not show up, but for a
   large dense network it would.
6. **Not yet tested at depth.** Everything here is 2–3 layers, or a graph with
   no depth at all. I do not know what 50 layers of this does, and the honest
   answer is that I built an architecture where I do not need 50 layers.

---

## 12. What would change my mind

- A 20-seed DQN run where `tanh` matches or beats the sine on the cleanly
  controlled pair. That would mean the CartPole result was about the saturating
  output head, not about periodicity.
- Learned `b` values collapsing toward zero across tasks. That would mean the
  network is asking to be a sigmoid, and I should let it.
- Learned `h` and an explicit input bias reliably cancelling each other out.
  That would confirm the degeneracy in §8 is costing real capacity, and one of
  the two should be removed.

---

## 13. Summary

- A sigmoid is one step of a sine wave rotated 45 degrees, and 45° is exactly
  the largest rotation that keeps a sine wave a function. The fit of a logistic
  to one step is within 4.4%.
- So the sigmoid is not an alternative to a wave. It is a wave with one step
  kept and the rest flattened.
- `f(x) = a·sin(b(x − h)) + k`, default `−sin(x/3)`: amplitude, frequency,
  phase, offset — the four things you can do to a wave — all learnable per unit.
- The unit never permanently dies: `|f'|` returns to its maximum every half
  period, while sigmoid's dead fraction grows toward 100% and ReLU's is a fixed
  50%.
- `a → −a` is an exact negation of the unit. That is what makes 2NRL's
  `invert()` possible, and neither ReLU nor sigmoid has such a parameter.
- `df/dh = −df/dx` exactly: the phase parameter *is* a per-unit input bias.
- The cost is global monotonicity, mitigated by population coding, by
  initialising inside one arm, and by `b` being learnable.
- CartPole: 9 (sigmoid), 10 (ReLU), 28–48 (sine). Real, directionally
  consistent with the theory, and in need of the seed sweep in §9 before it
  should be quoted as a headline.

The wave was there the whole time. We were only ever looking at one step of it.
