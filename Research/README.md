# Research

Full write-ups of the ideas the code in this repository is built on. Each paper
expands one of my original short notes into the reasoning, the mathematics, the
worked examples, and an honest account of what the evidence does and does not
show.

| Paper | The one-line claim |
|---|---|
| [Cycles Are a Feature](CyclesAreAFeature.md) | The brain is a directed *cyclic* graph. A cycle is repetition stored once, and the loops are what make a metacognitive level both necessary and useful. |
| [A Sine Wave Activation Function](SineWaveActivationFunction.md) | A sigmoid is one step of a sine wave rotated 45 degrees. I stopped deleting the rest of the wave. |

The two are connected. The sine activation gives every unit a sign parameter,
which is what makes network inversion a parameter change; the cyclic structure
is what makes inverting a *path* a two-colouring problem, and what makes odd
cycles the case where that problem has no exact answer. Read either one first;
each says where it depends on the other.

## Where the ideas are implemented

| Idea | Code |
|---|---|
| the cyclic graph, split / merge, the invariants | `RadixCyclicNN/radixnet/graph.py` |
| shortest-path prediction over the unrolled graph | `RadixCyclicNN/radixnet/search.py` |
| the sine activation and its partial derivatives | `RadixCyclicNN/radixnet/activation.py` |
| the model conversing with itself, and the repeat monitor | `RadixCyclicNN/radixnet/dialogue.py` |
| 2NRL, inversion, the self-upgrade loop | `RadixCyclicNN/radixnet/gan.py`, `model.py` |
| the activation comparison on CartPole | `ActivationFunctionTest/` |
| a learnable activation inside a conventional RNN | `SBNN_RNN_ActivationFunction/main.py` |

`RadixCyclicNN/DESIGN.md` is the full specification of the implementation.
These papers are the reasoning behind it.
