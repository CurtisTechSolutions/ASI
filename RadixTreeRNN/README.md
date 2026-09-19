# RadixTreeRNN

A radix tree with an RNN at each node — the recurrent sibling of
`RadixTrieLLM_RNN/`, where the per-node model is a small transformer instead.

The point of the whole radix-tree family is information compression: a path
through the tree *is* the sequence, a shared prefix is stored exactly once, and
the model attached to a node only ever has to explain the part of the string
that node owns. `RadixCyclicNN/` is where that idea is actually built out —
this directory is the RNN variant of it.

## Contents

| file | what it is |
|---|---|
| `main.py` | empty — the placeholder this directory was created around. Nothing is implemented here yet. |

## Status

**A stub.** See `RadixTrieLLM_RNN/main.py` for the working version of the same
shape (transformer-per-node), and `RadixCyclicNN/` for the full architecture
this line of thinking became.
