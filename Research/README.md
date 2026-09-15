# Research

Full write-ups of the ideas the code in this repository is built on. Each paper
expands one of my original notes into the reasoning, the mathematics, the worked
examples, and an honest account of what the evidence does and does not show.

| Paper | The one-line claim |
|---|---|
| [Cycles Are a Feature](CyclesAreAFeature.md) | The brain is a directed *cyclic* graph. A cycle is repetition stored once, and the loops are what make a metacognitive level both necessary and useful. |
| [A Sine Wave Activation Function](SineWaveActivationFunction.md) | A sigmoid is one step of a sine wave rotated 45 degrees. I stopped deleting the rest of the wave. |
| [The Vanishing Gradient Is a Feature](VanishingGradientIsAFeature.md) | Back-propagation multiplies, so a gradient `n` hops deep arrives at `c**n` of its size — and `c**n` is invertible. The magnitude was not lost, it was moved into an exponent whose key is the depth. Count the multiplications and the decay becomes a reading. |
| [2NRL: Learning by Inverting Consistent Failure](2NRL.md) | *Double-Negative Reinforcement Learning.* Train **on** the failures at full rate, negate the network, then fine-tune on the correct data. Two negatives — fail consistently, then do the inverse of what failed. |

They depend on each other, and it is worth knowing in which direction:

- The **sine activation** gives every unit a sign parameter, so negating a unit
  is a parameter change rather than a structural one. **2NRL's** inversion
  operator is built on exactly that, and would not be available with a ReLU.
- The **cyclic structure** is what makes inverting a *path* a two-colouring
  problem — and what makes odd cycles the case where that problem has no exact
  answer, which is why targeted inversion is best-effort rather than exact.
- Cycles are also what 2NRL is repeatedly called on to break: a loop the search
  could not escape becomes the negative phase of the next training run.
- The **vanishing gradient** paper is the sine paper read from the other end.
  The sine keeps `|f'|` from collapsing, which keeps the per-hop contraction
  bounded away from zero — in that paper's language, it keeps the message
  readable. And it is the road not taken from RadixCyclicNN's own design line:
  the shipped architecture makes `n = 1` so no product chain exists, while that
  paper asks what happens if you keep `n` instead.

All four make the same move: something the field routes around turns out to be
carrying information, and the usual fix deletes the signal rather than reading
it. Read any one first; each says where it depends on the others.

## Where the ideas are implemented

| Idea | Code |
|---|---|
| the cyclic graph, split / merge, the invariants | `RadixCyclicNN/radixnet/graph.py` |
| shortest-path prediction over the unrolled graph | `RadixCyclicNN/radixnet/search.py` |
| the sine activation and its partial derivatives | `RadixCyclicNN/radixnet/activation.py` |
| conversing with itself; stutter detection and backtracking | `RadixCyclicNN/radixnet/dialogue.py` |
| 2NRL, inversion, the self-upgrade loop | `RadixCyclicNN/radixnet/gan.py`, `model.py` |
| the activation comparison on CartPole | `ActivationFunctionTest/` |
| depth-counted gradient normalisation | `Research/experiments/depth_counted_normalisation.py` |
| a learnable activation inside a conventional RNN | `SBNN_RNN_ActivationFunction/main.py` |

`RadixCyclicNN/DESIGN.md` is the full specification of the implementation and
`RadixCyclicNN/DECISIONS.md` the record of why it is that way. These papers are
the reasoning behind both.
