# MultiGradientNN — Design Specification

A neural network that keeps to traditional norms — ordinary layers, ordinary
backpropagation — and expands **gradient descent** by giving every layer its own
gradient and stacking those gradients into a connected volume.

Directory: `MultiGradientNN/` (this directory).

This document is the contract the implementation will be written against.
It is **concept stage**: sections 1–4 are settled, section 5 lists the decisions
that are still open. Nothing in section 5 has been chosen — where this document
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
| The cross-layer step moves through the stack via the vertical weights | `V[k, k']` is the medium the step travels through, not a static coupling sitting between the planes. |
| Layers are accessed based on depth perception | At `(i, j)`, the column down the `k` axis holds one depth per plane; reading those depths is what selects the layer to access. |
| Depth perception is the human eye's | The analogy is literal: perception is comparative, never absolute, so the accessed layer is unchanged by a constant shift of the whole column. |

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
than across any single plane. Section 4 defines that step.

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

## 4. The cross-layer step (settled)

The operation that moves along the vertical axis. Two statements fix it.

**It travels via the vertical weights.** `V[k, k']` is not a static coupling
sitting between the planes; it is the medium the step moves through. Because
every layer is one hop from every other (section 3), the step reaches any plane
from any other directly, and `V[k, k']` is what it travels along to get there.

**Layers are accessed by depth perception.** Which layer the step reaches is
decided by perceiving depth through the stack. At a position `(i, j)`, the column
running down the `k` axis holds one depth per plane; reading those depths is what
selects the layer to access.

The two halves compose: **depth perception chooses the layer, the vertical weights
carry the step there.** Selection and transport are separate mechanisms doing
separate jobs, and neither substitutes for the other.

Four consequences follow.

1. **The access rule presupposes columns.** Perceiving depth "through the stack"
   is only defined if there is a column to look down — a fixed `(i, j)` with one
   depth per plane. This is independent evidence for the column-wise reading in
   5.1, which until now rested only on the word "vertical".

2. **`V[k, k']` carries two jobs at once.** Section 3 made it the learned
   separation between two layers; this section makes it the medium the step
   travels through. They are the same object, so a layer that has learned to be
   close is by that fact easier to reach. The metric and the transport are one.

3. **The descent shapes its own path.** The step travels via `V`, and `V` is
   itself learned (5.5). The route is therefore modified by the traffic on it.
   This is the design's most powerful property and its least stable one: the
   network can learn its own optimisation path, and it can also reinforce a path
   until nothing else is reachable. Whatever rule 5.5 settles on has to be read
   with this in mind.

4. **Depth is doing double duty.** `G_k[i, j]` is the weight being learned
   (section 2) and also the quantity perception reads to choose a layer. Changing
   a weight therefore changes which layers are reachable from it.

### 4.1 Depth perception is the eye's

The analogy is literal rather than decorative: depth perception here means what
the **human eye** does, and the mechanism is imported, not merely named.

That settles the first axis of 5.2. **Perception is comparative, never absolute.**
The eye does not read a distance off a single view; it derives depth from the
difference between views. So the access rule compares depths across the column and
never reads one plane's depth on its own. The consequence is concrete and
testable: **adding a constant to every depth in a column must leave the accessed
layer unchanged.** Depth perception is shift-invariant.

It also puts the eye's actual cues on the table as the menu for 5.2:

| Cue | What it would mean down a column |
|---|---|
| Binocular disparity (stereopsis) | Depth from the difference between two views of the same column. The primary cue — and the one needing a second view the design does not yet have. |
| Occlusion | A nearer plane hides those behind it: the shallowest depth at `(i, j)` wins and the rest are not reachable from there. Gives hard selection for free, and biologically. |
| Convergence | The eye points at a chosen depth and the plane there is the one accessed. Makes access an active aim rather than a passive read. |
| Motion parallax | Depth from how the column shifts between successive reads rather than between simultaneous views — a second view in time instead of in space. |

Two further consequences are properties of the eye, not choices:

1. **Stereopsis needs two views.** Depth from disparity is undefined from one
   viewpoint. The design has one column per position, so either a second view
   comes from somewhere (5.2), or the cue in use is monocular — occlusion and
   parallax both work with one eye.

2. **The eye's depth range is finite.** Stereo acuity falls off with distance;
   past a certain separation the disparity is too small to resolve and everything
   reads as equally far. Carried over, perception could not tell apart layers
   beyond some depth separation even though section 3 connects every layer to
   every other. **Connectivity and resolvability come apart:** `V[k, k']` still
   reaches a layer that perception can no longer distinguish from its neighbours.
   Whether to import that limit or drop it is a decision — and it is the one place
   where the eye and the dense stack actively disagree.

---

## 5. Open decisions

None of these are settled. They are written as decisions with their trade-offs
rather than left as questions, so each can be closed by picking a branch.

### 5.1 What `V[k, k']` connects

"Vertically" is read geometrically here — along `k`, perpendicular to the planes.
Two readings remain, though consequence 1 of section 4 now favours the first:

- **Column-wise.** `G_k[i, j]` connects to `G_k'[i, j]`: the same `(i, j)`
  position through the stack, so each position owns a column of `L` cells,
  densely connected within the column. `V[k, k']` is then one weight per
  position, `m x n` per connection set. This is the literal reading of
  "vertical", and the one depth perception needs.
- **Plane-wise.** Every cell of plane `k` connects to every cell of plane `k'`.
  `V[k, k']` is then `(m x n) x (m x n)`, which at any realistic width is far
  larger than the network it is attached to.

Column-wise is the assumed reading. It carries a constraint worth naming before
it bites: **columns only line up if the planes share a shape.** A conventional
network is ragged (`784 -> 128 -> 10`), so either the planes are held to a common
`m x n`, or a rule is needed for connecting positions that exist on one plane and
not another.

### 5.2 What depth perception computes

Section 4.1 settles that perception is comparative and puts the eye's cues on the
table. Three things remain:

- **Which cue.** Occlusion, convergence and parallax are monocular and work as the
  design stands; stereopsis is the eye's primary cue but needs a second view.
- **Where the second view comes from,** if the cue is stereopsis: two read-heads
  offset down the column, the disparity between neighbouring planes themselves, or
  two passes separated in time. The design has one column per position and stereo
  needs two views of it. This is the structural gap the eye analogy opens.
- **Hard or soft.** Whether perception selects exactly one layer, or ranks all of
  them and accesses them in proportion. Occlusion settles this toward hard
  selection if that is the cue; the others leave it open. Hard selection makes the
  step discrete and cheap, soft access keeps it differentiable.

### 5.3 What travels once a layer is accessed

Perception picks the layer and `V[k, k']` carries the step, but what is carried
is not yet fixed — the depth itself, its local slope, or something derived from
the accessed plane.

### 5.4 Scheduling the two descents

Whether the horizontal and vertical steps alternate, run simultaneously, or run
at different rates — and whether the vertical step is taken every batch, every
epoch, or on a schedule of its own.

### 5.5 Training the vertical weights

`V[k, k']` is not part of any single plane's surface, so the horizontal rule does
not obviously apply to it. Either the same rule is extended to cover it, or the
vertical weights get a rule of their own. Consequence 3 of section 4 makes this
the decision the stability of the whole scheme rests on.

### 5.6 What each layer's gradient is *of*

Conventionally every layer descends one shared scalar loss, and the gradient is a
separate object derived from it. Here the gradient *is* the plane and the depth on
it is the weight, so the array descended and the array learned are the same one.

What stays open is what drives that descent: per-layer losses, or one loss read
differently per layer.

### 5.7 Tooling

`RadixCyclicNN` is deliberately standard-library only. Traditional backpropagation
over an `L x m x n` volume is the case where `numpy` earns its place. Whether this
model keeps the zero-dependency rule or takes the dependency is a decision about
the repository's conventions, not just this directory.

---

## 6. Package layout (provisional)

Depends on 5.7; the shape below assumes a Python package in the style of
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
    perception.py           depth perception down a column: which layer is accessed (section 4)
    descent.py              the horizontal step, the cross-layer step (section 4), the schedule (5.4)
    model.py                MultiGradientNet - forward, backward, train, predict
    cli.py                  argparse CLI
  tests/
    test_volume.py          geometry and indexing invariants
    test_vertical.py        all-to-all connectivity, L(L-1)/2 sets, one-hop reachability
    test_perception.py      depth perception selects a layer from a column
    test_descent.py         the two steps and their scheduling
```
