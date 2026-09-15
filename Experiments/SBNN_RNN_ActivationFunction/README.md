# SBNN_RNN_ActivationFunction

**S**elf-**B**uilding **N**eural **N**etwork: an RNN that *grows its own hidden
layer* when training stalls, driven by a **learnable activation function**.

Two of the repository's recurring claims meet here in the smallest possible
form. The activation is a parameter, not a fixture — so the network adapts the
shape of its own non-linearity rather than only its weights. And a stall is a
signal to be acted on rather than a failure: when the loss stops improving, the
architecture changes. (`Research/VanishingGradientIsAFeature.md` and
`Experiments/TwoNRL_CartPole/` take the other road out of a stall — invert instead of grow.)

## Contents

| file | what it is |
|---|---|
| `main.py` | the whole thing: `SineActivation`, `SelfBuildingRNN`, a toy dataset and a demo run |

### What is in `main.py`

| piece | what it does |
|---|---|
| `SineActivation` | `f(x) = a·sin(b·(x − h)) + k` — the activation from `Research/SineWaveActivationFunction.md`, at the defaults `a = −1, b = 1/3, h = 0, k = 0` exactly `−sin(x/3)`. All four parameters are **learnable per unit**, and `backward` returns `dx` plus the gradients for `a, b, h, k` using section 8's partials. Same formula and same partials as `RadixCyclicNN/radixnet/activation.py`, including its `b ≥ MIN_B` floor. |
| `gradient_check` | central differences against those five partials, printed before every run — the numbers are only worth reading if the derivatives are right. Currently 4.8e-09. |
| `SelfBuildingRNN` | a from-scratch NumPy RNN (`Wxh`, `Whh`, `bh`, `Why`, `by`) using `SineActivation` for the hidden-state update, with BPTT. The activation parameters are updated at `lr/10`, the tenth section 8 asks for by name. Growth control (`maybe_grow` / `_grow_hidden`): if the loss has not improved by `min_delta=1e-4` across the last `patience=15` epochs, the hidden layer gains `growth_amount=4` units, up to `max_hidden_size=256`. The old weights are copied into the top-left corner of the larger matrices and the new rows and columns are drawn fresh from `N(0, 0.1)`, so the network is perturbed by a growth step rather than continued exactly. A new unit's wave enters at the defaults — a unit built on a stall is born as exactly `−sin(x/3)` and learns its own `a, b, h, k` from there. `_grow_hidden` seeds its own unseeded generator, so growth is not reproducible run to run, and `min_delta` is an *absolute* improvement threshold, so once the loss is below `1e-4` the plateau test fires every epoch and the layer grows to `max_hidden_size` regardless of the activation. |
| `make_sequence_dataset` | a toy task — points on a sine wave, predict the next one. |
| `main` | prints the gradient check, trains for 400 epochs from `hidden_size=4`, then prints predictions against targets, the hidden size it ended up at, and how far the mean `a, b, h, k` moved off the defaults. |

## Running it

```bash
pip install numpy
python3 main.py
```

## Where it sits

This is the *conventional* RNN read of the activation idea — the author's
wave dropped into an otherwise ordinary recurrent network, so the effect can be
seen without the rest of the architecture in the way. The formula is the one in
the paper and nothing else: `a·sin(b·(x − h)) + k`.
`Research/SineWaveActivationFunction.md` is the argument, `Experiments/ActivationFunctionTest/`
is the head-to-head against sigmoid and ReLU, and
`RadixCyclicNN/radixnet/activation.py` is the version that ships.
