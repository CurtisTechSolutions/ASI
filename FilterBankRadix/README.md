# FilterBankRadix

**Layer 1 is an activation filter. It feeds a set of Radix Tree networks.**

```
                         ┌──────────────┐
  text ── features ──►   │  m sine units│ ──► address ──┐
                         │   (layer 1)  │               │
                         └──────────────┘               ▼
                                              ┌───────────────────┐
                                              │ radix tree  #r    │ ──► p(next char)
                                              │   (layer 2)       │
                                              └───────────────────┘
                            one of 2^m trees, and only that one
```

Layer 1 is a bank of learnable sine units, read for their **sign**. The signs
spell an address. The address picks one radix tree out of a bank of them, and
that tree — a path-compressed context trie with a learnable sine on every node
and `RadixCyclicNN`'s one-hop rule on every branch — predicts the next
character. Nothing else crosses between the layers: no representation goes
down, and no gradient comes back. What comes back is a **number** — how many
bits a tree spent on a segment — and the filter is fitted to the address that
number points at.

`DESIGN.md` is the specification. This is what it measured.

## Quick start

```bash
cd FilterBankRadix
make test         # 27 tests, standard library, seconds
make check        # finite-difference checks of all three learning rules
make demo         # fit a small bank and see what each address ended up holding
make quick        # a one-seed smoke run of every sweep, a few minutes
make experiment   # every arm and every sweep — the numbers below (slow)
```

Standard library only. No install, no download, nothing optional.

## The result that matters

**The architecture works, learns, and loses.** A learned filter beats the same
filter left frozen, and beats a random split by a wide margin — so the routing
is doing something. But one radix tree over the whole corpus beats a bank of
four almost everywhere, *even when the router is perfect*, and the experiment
says exactly where that stops being true:

| | one tree wins | the bank wins |
|---|---|---|
| **context depth** | 4, 5, 6 | **2 and 3** |
| **corpus size** | 100 and 200 segments per source | **400 and 800** |

Both boundaries are the same sentence read two ways. Routing hands the model a
fact — *which register this is* — that a deep tree already has in its context
and that a small corpus has too little data to exploit. Where the tree can work
the fact out for itself, a router is an overhead; where it cannot, a router is
worth a fifth of a bit a character.

And two things this was not built to find:

* **Counting beats the learning rule.** The same trees, the same compression,
  the same backoff, with the softmax of `w·f_p·f_c` replaced by relative
  traversal counts — and nothing trained at all — give better held-out bits
  than training does, under both routings.
* **The author's wave is a good activation and a poor gate.** A sine filter
  with its wave held fixed cannot be fitted to an address at all, and the
  reason is measurable: the hinge can only raise a response by pushing the
  projection, and a periodic response *comes back down*. It ends up sixteen
  periods out.

## What the experiment says

Every number is on held-out segments, stratified by source, that no arm trained
on, and every arm shares the corpus, the smoothing constants, the shared prior
and the seeds. `python3 summarize.py` prints these tables back out of
`results/*.json`.

(no results/main_results.json - run `make main`)

The ordering is stable across seeds. **`learned` beats `frozen`** — the same
filter, drawn the same way, differing only in whether the refilter step is
allowed to move it — which is the control that says learning the filter is
worth something. **`roundrobin` is the worst arm**, so the gain is routing and
not merely splitting. And **`single` beats `oracle`**: a bank with a *perfect*
router, told the answer, still loses to one tree at this depth and this size.
Splitting a radix tree's corpus costs every expert data and costs the bank the
prefixes the sources share, and at 58k characters that costs more than knowing
the register is worth.

### What the one-hop rule is worth

(no results/baseline_results.json - run `make baseline`)

Nothing is learned in the `counts` arms: the branch probabilities are relative
traversal counts, the trees are otherwise identical, and each run takes about a
second against forty. They win by 0.14–0.18 bits/char. Two conclusions, each
confirmed twice: **one tree beats four under either scoring rule**, so the
routing result does not depend on the learning rule; and **the learning rule
costs bits** on this task at this scale.

(no results/epochs_results.json - run `make epochs`)

### Context depth — where routing is worth something

(no results/depth_results.json - run `make depth`)

This is the mechanism, and it is monotone. At depth 2 a tree sees two
characters of context, which is not enough to tell Go from Python, and being
*told* the register is worth 0.24 bits/char. By depth 4 the tree has worked it
out for itself and the same knowledge is worth less than the data it costs. A
router buys context; if the model already has the context, the router is paying
for something it owns.

### Corpus size — the other boundary

(no results/scale_results.json - run `make scale`)

Splitting costs every expert data, so the bank's penalty should shrink as there
is more of it, and it does: the oracle's deficit narrows with every step and
turns into a win. The learned arms do not follow, because their problem is not
data.

### Where the routing gap is

(no results/probe_results.json - run `make probe`)

Four supervised measurements — things the architecture is never allowed to do —
that say where the gap between the learned router and the oracle sits.

**The signal is perfect.** Give the bank true experts and ask which one prices
each segment cheapest: it is the segment's own register, for every single
segment, NMI 1.000. That is exactly the target the refilter step chases, so
nothing is wrong with what layer 2 tells layer 1.

**Layer 1 is the bottleneck, and it is short of the answer twice over.** Fitted
*directly on the labels*, the filter reaches 0.71–0.74 purity, not 1.0 — the
17-number histogram and one unit per question do not contain the partition.
And the unsupervised rule reaches about 0.5, well short of its own supervised
ceiling. Roughly half of the gap is representation and half is the rule.

**A periodic unit is the wrong shape for a gate.** A sine filter with its wave
held fixed scores 0.245 against a chance of 0.25 — it cannot be fitted at all —
while the same filter with `a, b, h, k` free reaches 0.697, and a *monotone*
unit with its wave fixed is the best router of the four at 0.736. The mechanism
is measured, not inferred: a hinge raises a response by pushing the projection,
and the wave's argument `|u| = |b(z − h)|` goes from a median of 0.28 at
initialisation to **100.1** — **98.3%** of the units end up past their first
peak, where a sine is coming back down and only a monotone unit is still going
up. The tanh ends up past its first peak too (73.8%), and does not care: past
the peak it is merely saturated, and an argmax only reads the ordering.

What learning the wave mostly does in layer 1 is **flatten it**: `b` falls from
0.333 to 0.177, lengthening the period until the unit is monotone over the range
the data occupies, and the share past the first peak drops to 12.2%.

That is the same trap `Research/Insights.md` §24 names from the other end. There,
`b = 1/3` is too *low* — it puts every neuron in the sine's linear region and a
deep network collapses to a linear one. Here the same `b = 1/3` is what lets a
hinge push a gate right around the circle. One frequency, two opposite failures,
and the parameter that fixes both is the one the design already makes learnable.

### The refilter step's hinge passes

(no results/passes_results.json - run `make passes`)

The pricing half of the refilter step is expensive and the hinge half is nearly
free, so running the hinge more than once over the same prices costs almost
nothing — which is an argument for more passes, not a measurement. The
measurement says it barely matters. The default is 4 because the argument is
still sound, and this table is here because the argument is not evidence.

## How it works

**Layer 1.** A segment becomes 17 numbers: a 16-bucket histogram of character
codepoints, scaled by length so it is length-blind, plus a constant. Each unit
projects that and shapes it with the author's wave,
`f_j(x) = a_j·sin(b_j(v_j·x − h_j)) + k_j`, and is read for its sign. The signs
are the address. Every unit is born as exactly `−sin(z/3)` and learns its own
`v, a, b, h, k` from there, the activation parameters at a tenth of the
projection's rate.

**Layer 2.** Each address owns a radix tree: every window of up to `D+1`
characters inserted, a run of one-child nodes stored as one node, children
scored `w_c · f_p · f_c` and normalised by a softmax over the siblings. The
gradient of a decision touches that edge, those two nodes and their four wave
parameters — one hop, no chain, nothing to vanish. Prediction folds every
context length the tree knows, longest last, over a shared prior.

**Fitting.** Route, build, train, refilter — repeated. The experts are
**rebuilt** each round, never extended. Then every live expert prices every
training segment, the cheapest is the address layer 1 should have produced, and
each disagreeing bit is pushed across its boundary by a hinge whose step is the
size of the mistake in bits. That scalar is the only thing layer 2 ever tells
layer 1.

## What went wrong first

Four corrections, in the order they were needed. Three of them changed the
numbers and one of them changed a conclusion.

**1. Ignorance was cheaper than knowledge.** An expert that had never seen a
context paid the smoothing floor — about eleven bits a character — while an
expert that knew the context but was unsure paid four. So on some segments *not
knowing* scored better, and the filter learned to route text to the tree that
knew least about it. The fix is that every expert interpolates with a shared
prior by how often it has been at that context, which makes a knowing expert
better than an ignorant one **by construction** rather than by luck.
`test_ignorance_is_never_cheaper_than_knowledge` holds the line.

**2. Folding one level instead of the chain.** The first model mixed the prior
into the *deepest matching* context only. A backed-off prediction lands at a
node whose count is enormous — the root's count is the whole corpus — so its
interpolation weight is ≈ 1 and it drowns out the prior it was supposed to
defer to. Folding every context length, longest last, so the residual weight is
a product of `(1 − own)`, took one tree from 2.998 to **2.808** bits/char and
turned a deep shared prior from harmful (3.153) into neutral (2.880).

**3. Reinforcing the segments that were already right.** The first refilter rule
also pushed the weakest bit *away* from its boundary whenever the address was
already the best one. That is a constant force on a boundary with no error to
correct, and it does what constant forces do: on the synthetic corpus it took a
0.97-pure routing, dragged two clean clusters together and emptied an address,
in two rounds. It is off by default now (`reinforce = 0`).

**4. A ceiling measured before the fit had converged.** The probe's supervised
fits ran 25 epochs, and one of the three two-against-two splits read **0.500 —
exactly chance**. That looked like a finding: a question the filter cannot
answer, and therefore a hard limit on a sign-bit address. It was not. The
unit's `|a·b|` had collapsed to 0.0100 — the unit had *died* and was answering
a constant. At four times the epochs the same split reads 0.777. Every
supervised number here now runs 100 epochs, is averaged over three seeds, and
carries `|a·b|` beside it.

## What it does not establish

One corpus, cut out of this repository; four registers; 58k to 233k characters;
character-level prediction; one segmentation; 4 addresses in most runs; depths
2 to 6; three seeds. Nothing here says anything about routing at other
granularities (per character, per document), about more than four registers,
about a learned encoder in front of the filter, or about soft routing — a
softmax over experts would make the whole bank differentiable, and is the
obvious next thing to try, but it is not a *filter* and so it is not this
design.

The bits/char are a measurement of these models on this text, not of anything's
capability in general: a byte-level n-gram baseline with proper discounting
would beat every arm here, and that is not what any of these numbers are for.
What they are for is the comparison *between* the arms, which share every
constant, every smoothing rule and every seed.

## Contents

| file | what it is |
|---|---|
| `DESIGN.md` | the specification — the contract every module is written against |
| `fbradix/activation.py` | the parametric sine and its partials, plus the monotone control |
| `fbradix/filter.py` | **layer 1**: features, the units, the two address codes, the hinges |
| `fbradix/tree.py` | **layer 2**: the radix context tree and the one-hop rule |
| `fbradix/bank.py` | the routers and the route / build / train / refilter loop |
| `fbradix/corpus.py` | the four-source corpus and the labels it is scored against |
| `fbradix/experiment.py` | the arms, the metrics, the sweeps and the probe |
| `fbradix/check.py` | finite-difference checks of all three learning rules |
| `fbradix/cli.py` | `python3 -m fbradix.cli <command>` |
| `summarize.py` | the tables above, read back out of `results/*.json` |
| `data/`, `results/`, `tests/` | the corpus snapshot, the numbers, and 27 tests |

## Where it sits

`RadixCyclicNN/` is where the radix idea is built out properly — a single
self-compressing *cyclic* graph, with the one-hop rule this borrows. `GREN/`
uses a radix tree over discovered signatures and a network that grows its own
input and output layers. `Experiments/ActivationFunctionTest/` is where the
sine activation is measured head to head against sigmoid, tanh and ReLU, and
where the failure mode this architecture met again — a unit dying along `a` and
`b` rather than along `x` — was first measured.

This directory is the variant that asks a different question: not *what should
one network be*, but *what should choose between several of them* — and whether
the author's wave, which is a good activation, is also a good **gate**.
