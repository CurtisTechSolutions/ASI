# SBNN_RNN_ActivationFunction

**S**elf-**B**uilding **N**eural **N**etwork: an RNN that *grows its own hidden
layer* when training stalls, driven by a **learnable activation function**.

Two of the repository's recurring claims meet here in the smallest possible
form. The activation is a parameter, not a fixture — so the network adapts the
shape of its own non-linearity rather than only its weights. And a stall is a
signal to be acted on rather than a failure: when the loss stops improving, the
architecture changes. (`Research/VanishingGradientIsAFeature.md` and
`TwoNRL_CartPole/` take the other road out of a stall — invert instead of grow.)

## Contents

| file | what it is |
|---|---|
| `main.py` | the whole thing: `CustomActivation`, `SelfBuildingRNN`, a toy dataset and a demo run |

### What is in `main.py`

| piece | what it does |
|---|---|
| `CustomActivation` | `f(x) = x·σ(αx) + β·tanh(x)` with **learnable** `α` and `β` — a swish whose gate steepness and a tanh whose weight are both trained. `backward` returns `dx` plus the gradients for `α` and `β`. |
| `SelfBuildingRNN` | a from-scratch NumPy RNN (`Wxh`, `Whh`, `bh`, `Why`, `by`) using `CustomActivation` for the hidden-state update, with BPTT. Growth control (`maybe_grow` / `_grow_hidden`): if the loss has not improved by `min_delta=1e-4` across the last `patience=15` epochs, the hidden layer gains `growth_amount=4` units, up to `max_hidden_size=256`. The old weights are copied into the top-left corner of the larger matrices and the new rows and columns are drawn fresh from `N(0, 0.1)`, so the network is perturbed by a growth step rather than continued exactly. `_grow_hidden` seeds its own unseeded generator, so growth is not reproducible run to run. |
| `make_sequence_dataset` | a toy task — points on a sine wave, predict the next one. |
| `main` | trains for 400 epochs from `hidden_size=4`, prints predictions against targets and the hidden size it ended up at. |

## Running it

```bash
pip install numpy
python3 main.py
```

## Where it sits

This is the *conventional* RNN read of the activation idea — a learnable
non-linearity dropped into an otherwise ordinary recurrent network, so the
effect can be seen without the rest of the architecture in the way.
`Research/SineWaveActivationFunction.md` is the argument, `ActivationFunctionTest/`
is the head-to-head against sigmoid and ReLU, and
`RadixCyclicNN/radixnet/activation.py` is the version that ships.
