# MultiGradientNN

A neural network that follows traditional norms — ordinary layers, ordinary
backpropagation — and expands on **gradient descent** itself by adding a
dimension to it.

Unlike the rest of this repository, this variation does not reinvent the
architecture. The divergence is in the descent.

## The idea

A gradient today is a single surface. Picture it as an XY coordinate plane
where the weights give the depth — one landscape, and descent walks it looking
for a minimum.

This model builds **multiple gradients as layers**. Each layer owns its own
gradient, its own XY plane with its own weight depth. Stacking them turns the
single surface into a volume: the layer index becomes a new axis, so descent
moves *through* the stack as well as across each plane in it.

That added axis is the whole point. A single gradient can only answer "which
way is downhill from here". A stack can also answer "which layer is the right
one to be descending in", and the two questions are optimised together.

## Vertical connections

The layers are not independent surfaces that descent merely hops between. Every
layer connects **directly to every other layer** along the vertical axis — not
just to the one above and below it, but to all of them, in a single hop.

The stack is therefore densely connected: with `L` layers there are `L(L-1)/2`
vertical connection sets, each carrying its own weights. Gradient information
reaches any layer from any other without propagating through the layers in
between.

This also settles what the new axis measures. The stack index is not a distance
— layer 1 is no further from layer 5 than it is from layer 2, since both are one
direct connection away. How far apart two layers are is a *learned weight* on the
connection between them, not a difference of indices.

## Status

Concept stage. Settled so far: the stack geometry, and the dense vertical
connectivity above.

Still to be written:

- how the cross-layer step is defined — what it means to move along the vertical
  axis in one update;
- how the horizontal descent (within a plane) and the vertical descent (across
  the stack) are scheduled against each other;
- whether the vertical weights are trained by the same rule as the horizontal
  ones, or by their own.
