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

## Status

Concept stage. The directory is scaffolded; the design — how the layers are
coupled, how the cross-layer step is defined, and how the two descents are
scheduled against each other — is still to be written.
