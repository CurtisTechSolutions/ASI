# PrimedRadixPair

**Two radix trees, primed with every option, connected at every equal node.**

A radix tree that is **primed** — every sequence of `1..L` units over a closed
alphabet is inserted before the tree sees a single text — never changes its
structure again: training is a counter going up at an address that is
*computed*, not found. A second primed tree over the same sequences reads them
from the other end, and the two are connected wherever they hold an equal
sequence: at every level, not only at the final nodes. Walking down the first
tree adds a unit to the end of what the model remembers; crossing to the second
and stepping toward its root drops a unit from the beginning. Together they can
walk a text of any length through a structure of fixed size, and fall back to a
shorter memory whenever the longer one is empty.

## Contents

| file | what it is |
|---|---|
| `PRD.md` | the requirement in the author's own words, the hypothesis, goals and non-goals, the requirements, the success criteria, the phases, and the decisions still to be made |
| `DESIGN.md` | the specification — the contract every module is implemented against, in 18 sections |

**There is no code in this directory yet.** The two documents are the whole of
it. Read `PRD.md` first — it is short and it names the decisions the design
rests on — then `DESIGN.md` before writing anything here.

## The three things worth knowing

**Priming is arithmetic.** A complete `R`-ary tree needs no pointers: the path
to a node *is* its address, read as a number in base `R` (the radix). Inserting
every option by brute force is therefore free in structure and costs only the
counters — one per sequence — and the literal brute-force insertion is kept as
the oracle the arithmetic is tested against (`DESIGN.md` §6, §14).

**The rungs are the fold.** Falling back from a long context to a shorter one is
a path: across to the second tree, one step toward its root, and back across.
Summed over every depth the walk can fall to, this reproduces exactly the
every-context-length prediction of `FilterBankRadix/DESIGN.md` §5.3 — checked to
`1e-12` while the design was written (`DESIGN.md` §8.3). Connected only at the
final nodes, the pair cannot even slide its window; the connections at every
level are what make it a model (`DESIGN.md` §8.5).

**One sequence, both ends.** The two trees hold the same counts read two ways:
what follows a sequence and what precedes it. A text is scored from either end,
and the unit that *both* readings find unlikely is where the text is wrong
(`DESIGN.md` §8.4) — the first hook for the negative network.

## Planned shape

Python package `radixpair`, Python 3.11+, **standard library only**. The
familiar four verbs — `train / predict / generate / score` — on the same CLI
shape as `RadixCyclicNN/` and `GTMNN/`. Two kinds on one node set: the `count`
kind first (deterministic, no learning rate, the measurements are made on it),
the `sine` kind second (the family's per-node sine, learnable edge and rung
weights, `invert` and 2NRL).

## Related

* `FilterBankRadix/DESIGN.md` §5 — the context tree this reproduces as a walk, and the smoothing constants shared so bits per unit are comparable.
* `RadixCyclicNN/DESIGN.md` §5, §7 — the grown graph this is the opposite end of, and the shortest-path search reused here.
* `RadixTrieLLM_RNN/main.py` — the smallest radix insertion in the repository; its splitting logic is what a primed tree never needs.
* `Research/Insights.md` §13 — "a radix tree with the mechanics of a trie": a terminal on an internal node is why the end of a text is a symbol here.
* `PhoneticTokenizer/` — the alphabet of sounds this is sized for.
