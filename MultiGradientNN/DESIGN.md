# MultiGradientNN — Design Specification

A neural network that keeps to traditional norms — ordinary layers, ordinary
backpropagation — and expands **gradient descent** by giving every layer its own
gradient and stacking those gradients into a connected volume.

Directory: `MultiGradientNN/` (this directory).

This document is the contract the implementation will be written against.
It is **concept stage**: sections 1–3 are settled, section 4 lists the decisions
that are still open. Nothing in section 4 has been chosen — where this document
has to name something to keep the notation usable, it says so explicitly.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| Traditional norms | Ordinary layers, ordinary forward pass, ordinary backpropagation. The architecture is not reinvented; the divergence is in the descent. |
| A gradient is an XY coordinate plane with weights as depth | **One gradient is a single plane.** `(i, j)` are the X and Y coordinates, and the weight at that point is the depth. |
| Multiple gradients (multiple layers) | `L` gradients, so `L` planes, one per layer — rather than every layer sharing one surface. |
| This adds another dimension | The planes stack: `G` is an `L × m × n` volume. The stack index `k` is a third axis, and descent moves *through* it as well as across each plane. |
| Each layer directly connected to each other vertically | All-to-all along `k`: every layer is one hop from every other, `L(L-1)/2` vertical connection sets, each with its own weights. |

---

## 2. Notation and geometry (settled)

```
L                 number of layers = number of gradients = number of planes
k, k'             layer indices, 0 <= k < L
G_k               layer k's gradient - a single plane, an m_k x n_k grid
G_k[i, j]         the depth at (i, j) on that plane: the weight there
G                 the full volume - the L planes stacked along the k axis
V[k, k']          the vertical connection between layers k and k'
```

**One gradient is one plane.** The two words name the same object. Layer `k` has
exactly one gradient, that gradient *is* a single plane, and the weights are the
depth on it — `L` layers, `L` gradients, `L` planes, and the three counts never
diverge. A gradient is not a field laid over a plane; there is nothing to hold
apart.

Two axes are horizontal (`i`, `j`, within a plane) and one is vertical (`k`,
through the stack).

**Horizontal descent** is conventional: it moves across plane `G_k`, from `(i, j)`
toward lower depth, by the usual step.

**Vertical descent** is the new part. It moves along `k` — between planes — rather
than across any single plane. Section 4.2 is where that step gets defined.

The two are not the same operation. A single gradient answers *which way is
downhill from here*; the stack must also answer *which layer is the right one to
be descending in*, and the pair is optimised together.

---

## 3. Vertical connectivity (settled)

Every layer connects **directly to every other layer**, not only to its immediate
neighbours. Layer 0 reaches layer 5 in one hop, exactly as it reaches layer 1.

```
connection sets   L(L-1)/2
reachability      any layer from any other in one hop, no propagation through
                  the layers in between
```

Three consequences follow, and they are properties of the design rather than
choices still to be made:

1. **The stack index is not a distance.** Layer 0 is no further from layer 5 than
   from layer 1, since both are a single connection away. How far apart two
   layers are is the *learned weight* on `V[k, k']`, not `|k - k'|`. The vertical
   axis carries no metric of its own; the network supplies one.

2. **Parameters grow quadratically in `L`.** `L = 4` is 6 connection sets and
   `L = 8` is 28, both negligible. `L = 64` is 2,016, at which point the vertical
   weights dominate the model. Whatever `V[k, k']` is made of, its size is
   multiplied by `L(L-1)/2`.

3. **Dense connectivity outranks depth.** In a conventional stack, information
   crosses the layers in order and depth is what separates them. Here nothing is
   separated by more than one hop, so ordering the layers `0 … L-1` is a labelling
   convention, not a structural claim.

---

## 4. Open decisions

None of these are settled. They are written as decisions with their trade-offs
rather than left as questions, so each can be closed by picking a branch.

### 4.1 What `V[k, k']` connects

"Vertically" is read geometrically here — along `k`, perpendicular to the planes.
Two readings remain:

- **Column-wise.** `G_k[i, j]` connects to `G_k'[i, j]`: the same `(i, j)`
  position through the stack, so each position owns a column of `L` cells,
  densely connected within the column. `V[k, k']` is then one weight per
  position, `m x n` per connection set. This is the literal reading of
  "vertical".
- **Plane-wise.** Every cell of plane `k` connects to every cell of plane `k'`.
  `V[k, k']` is then `(m x n) x (m x n)`, which at any realistic width is far
  larger than the network it is attached to.

Column-wise is the assumed reading below. It carries a constraint worth naming
before it bites: **columns only line up if the planes share a shape.** A
conventional network is ragged (`784 -> 128 -> 10`), so either the planes are
held to a common `m x n`, or a rule is needed for connecting positions that
exist on one plane and not another.

### 4.2 The cross-layer step

What it means to move along the vertical axis in one update. The horizontal step
is the conventional one; this is the operation that has no standard counterpart,
and it is the centre of the design.

### 4.3 Scheduling the two descents

Whether the horizontal and vertical steps alternate, run simultaneously, or run
at different rates — and whether the vertical step is taken every batch, every
epoch, or on a schedule of its own.

### 4.4 Training the vertical weights

`V[k, k']` is not part of any single plane's surface, so the horizontal rule does
not obviously apply to it. Either the same rule is extended to cover it, or the
vertical weights get a rule of their own.

### 4.5 What each layer's gradient is *of*

Conventionally every layer descends one shared scalar loss, and the gradient is a
separate object derived from it. Here the gradient *is* the plane and the depth on
it is the weight, so the array descended and the array learned are the same one.

What stays open is what drives that descent: per-layer losses, or one loss read
differently per layer. Which it is determines whether the vertical connections
carry gradient, activation, or both.

### 4.6 Tooling

`RadixCyclicNN` is deliberately standard-library only. Traditional backpropagation
over an `L x m x n` volume is the case where `numpy` earns its place. Whether this
model keeps the zero-dependency rule or takes the dependency is a decision about
the repository's conventions, not just this directory.

---

## 5. Package layout (provisional)

Depends on 4.6; the shape below assumes a Python package in the style of
`radixnet`, with the package name still to be confirmed.

```
MultiGradientNN/
  DESIGN.md                 this file
  README.md                 the idea, in short
  pyproject.toml            console script `mgradnet = mgradnet.cli:main`
  mgradnet/
    __init__.py             exports MultiGradientNet, TrainConfig, __version__
    volume.py               the L x m x n volume: the planes, their columns, indexing
    vertical.py             V[k, k'] - the dense vertical connections (section 3)
    descent.py              the horizontal step, the vertical step (4.2), the schedule (4.3)
    model.py                MultiGradientNet - forward, backward, train, predict
    cli.py                  argparse CLI
  tests/
    test_volume.py          geometry and indexing invariants
    test_vertical.py        all-to-all connectivity, L(L-1)/2 sets, one-hop reachability
    test_descent.py         the two steps and their scheduling
```
