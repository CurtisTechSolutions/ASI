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
spell an address (there is a second code, where the address is simply the unit
with the largest response; §4.5 of `DESIGN.md` says why that choice matters
more than it sounds). The address picks one radix tree out of a bank of them, and
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
make test         # 33 tests, standard library, seconds
make check        # finite-difference checks of all three learning rules
make demo         # fit a small bank and see what each address ended up holding
make quick        # a one-seed smoke run of every sweep, a few minutes
make experiment   # every arm and every sweep — the numbers below (slow)
```

Standard library only. No install, no download, nothing optional.

## The result that matters

**The architecture works, learns, and is beaten by one tree — except in the two
places where the experiment says it should not be.**

A learned filter beats the same filter left frozen by 0.22 bits/char, and beats
a random split by 0.44, so the routing does something. A single radix tree over
the whole corpus still beats the bank at the default depth and size. But the
deficit is not a constant, and both sweeps cross:

| | one tree wins | the bank wins |
|---|---|---|
| **context depth** (58k chars) | 5, 6 | **2 and 3** (a perfect router), **4** (the learned one) |
| **corpus size** (depth 5) | 100 – 400 segments per source | **800** (232k characters) |

Both boundaries say the same thing twice. Routing hands the model a fact —
*which register this is* — that a deep tree already has in its context and that
a small corpus has too little data to exploit. Where the tree can work the fact
out for itself, a router is overhead; where it cannot, it is worth up to 0.2
bits a character.

Three things this was not built to find:

* **The learned filter beats the true labels.** `monotone` routes on nothing
  but a character histogram and reaches **2.831** against the source-perfect
  oracle's **2.873**. The best partition for predicting text is not the
  partition a human would call correct.
* **Counting beats the learning rule.** The same trees with the softmax of
  `w·f_p·f_c` replaced by relative traversal counts — nothing trained at all,
  one second against forty — give **2.614** against **2.786**, and the rule
  closes the gap only slowly with ten times the training.
* **The author's wave is a good activation and a poor gate.** A sine filter
  with its wave held fixed cannot be fitted to an address at all — 0.243
  against a chance of 0.25 — and the reason is measurable: the hinge can only
  raise a response by pushing the projection, a periodic response *comes back
  down*, and 100% of its units end up past their first peak, a median of 143
  radians out.

## What the experiment says

Every number is on held-out segments, stratified by source, that no arm trained
on; every arm shares the corpus, the smoothing constants, the shared prior and
the seeds. `python3 summarize.py` prints these tables back out of
`results/*.json`.

| arm | bits/char (held out) | nodes | live | purity | nmi |
|---|---|---|---|---|---|
| `single` - one radix tree, no filter | **2.786** ± 0.012 | 34,317 | 1.0 | 0.250 | 0.000 |
| `monotone` - tanh units, same knobs | **2.831** ± 0.009 | 35,346 | 3.0 | 0.433 | 0.276 |
| `unbalanced` - no load pressure | **2.831** ± 0.040 | 35,862 | 3.0 | 0.488 | 0.329 |
| `deep-oracle` - oracle, deep shared prior | **2.873** ± 0.005 | 38,142 | 4.0 | 1.000 | 1.000 |
| `oracle` - routed by the true source | **2.873** ± 0.005 | 38,142 | 4.0 | 1.000 | 1.000 |
| argmax | **2.887** ± 0.057 | 36,846 | 3.7 | 0.473 | 0.305 |
| `deep` - learned, deep shared prior | **2.906** ± 0.119 | 36,236 | 3.0 | 0.388 | 0.178 |
| **`learned`** - the architecture | **2.908** ± 0.007 | 37,458 | 3.7 | 0.415 | 0.167 |
| `fixedwave` - only the projection learns | **3.088** ± 0.138 | 39,419 | 4.0 | 0.408 | 0.130 |
| `frozen` - calibrated random filter | **3.132** ± 0.067 | 40,320 | 4.0 | 0.465 | 0.180 |
| frozen-argmax | **3.143** ± 0.009 | 40,448 | 4.0 | 0.500 | 0.219 |
| `roundrobin` - split at random | **3.346** ± 0.003 | 43,915 | 4.0 | 0.331 | 0.021 |

800 segments / 58389 characters, alphabet 101, depth 5, 4 addresses, 3 seeds.

**`learned` beats `frozen`** (2.908 against 3.132) — the same filter, drawn the
same way, differing only in whether the refilter step may move it. **`frozen`
beats `roundrobin`** (3.132 against 3.346) — so even an unlearned filter routes
better than chance, because a random projection of a character histogram is
already correlated with the register. And **`single` beats `oracle`**: a bank
with a perfect router, told the answer, still loses to one tree here, because
splitting the corpus costs every expert data and costs the bank the prefixes
the sources share.

The ablations are more interesting than the headline:

| control | what it removes | costs |
|---|---|---|
| `fixedwave` | the wave's four parameters (only `v` learns) | **0.18 bits** — more than learning the projection is worth |
| `frozen` | all of layer 1's learning | 0.22 bits |
| `monotone` | the periodicity (tanh units, same knobs) | **−0.08 bits — removing it *helps*** |
| `unbalanced` | the load pressure on crowded addresses | **−0.08 bits — removing it helps too**, and it is what keeps addresses alive |
| `deep` | nothing; it deepens the shared prior | 0.00 bits — neutral, as §5.3 predicts |

### What the one-hop rule is worth

| arm | bits/char (held out) | nodes | live | purity | nmi |
|---|---|---|---|---|---|
| `counts` - one tree, **nothing learned** | **2.614** ± 0.000 | 34,317 | 1.0 | 0.250 | 0.000 |
| `counts-oracle` - routed by source, nothing learned | **2.740** ± 0.000 | 38,142 | 4.0 | 1.000 | 1.000 |
| `single` - one radix tree, no filter | **2.786** ± 0.012 | 34,317 | 1.0 | 0.250 | 0.000 |
| `oracle` - routed by the true source | **2.873** ± 0.005 | 38,142 | 4.0 | 1.000 | 1.000 |

800 segments / 58389 characters, alphabet 101, depth 5, 4 addresses, 3 seeds.

Nothing is learned in the `counts` arms: the branch probabilities are relative
traversal counts, the trees are otherwise identical. They win by 0.17 bits/char
unrouted and 0.13 routed.
Two conclusions, each confirmed twice: **one tree beats four under either
scoring rule**, so the routing result does not depend on the learning rule; and
**the learning rule costs bits** on this task at this scale.

It is not a representational limit. The parent's activation multiplies every
sibling equally, so it is an inverse temperature, and `w_c · f_c` is a free
per-child logit — `w_c f_c = (log p_c)/f_p` reproduces any distribution exactly.
So the gap is optimisation, and the next table is how much of it training
closes:

| branch probabilities | epochs | train bits/char | held-out bits/char |
|---|---|---|---|
| relative counts, nothing learned | — | 0.872 | **2.614** ± 0.000 |
| the one-hop rule | 1 | 1.066 | 2.842 ± 0.004 |
| the one-hop rule | 3 | 1.022 | 2.786 ± 0.012 |
| the one-hop rule | 10 | 0.977 | 2.726 ± 0.011 |
| the one-hop rule | 30 | 0.936 | 2.677 ± 0.003 |

**Under-trained, not over-fitted.** Counting has the lower *training* loss too
(0.872 against 1.022), and every extra epoch moves the rule towards it from
above without reaching it: ten times the training buys 0.11 bits and still
leaves 0.06 on the table. A rule that gets a worse answer than counting, more
slowly, is doing nothing here that counting does not — which is worth knowing
about a rule this repository uses everywhere.

### Context depth — where routing is worth something

| depth | `single` | `oracle` | `learned` | `argmax` | `frozen` | best |
|---|---|---|---|---|---|---|
| 2 | 3.517 | **3.320** | 3.618 | 3.516 | 3.684 | `oracle` |
| 3 | 2.972 | **2.958** | 3.110 | 2.978 | 3.326 | `oracle` |
| 4 | 2.814 | 2.878 | **2.807** | 2.849 | 3.239 | `learned` |
| 5 | **2.772** | 2.867 | 2.898 | 2.960 | 3.225 | `single` |
| 6 | **2.777** | 2.902 | 2.930 | 2.893 | 3.237 | `single` |

This is the mechanism, and it is monotone. At depth 2 a tree sees two
characters of context, which is not enough to tell Go from Python, and being
told the register is worth 0.20 bits/char. By depth 5 the tree has worked it
out for itself and the same knowledge costs more than it is worth. Depth 4 is
the crossing, and it is the one place the **learned** filter is the best arm in
the table.

### Corpus size — the other boundary

| segments per source | `single` | `oracle` | `learned` | `argmax` | best |
|---|---|---|---|---|---|
| 100 (28,990 chars) | **3.032** | 3.140 | 3.142 | 3.222 | `single` |
| 200 (58,389 chars) | **2.772** | 2.867 | 2.898 | 2.960 | `single` |
| 400 (115,475 chars) | **2.426** | 2.432 | 2.626 | 2.488 | `single` |
| 800 (232,105 chars) | 2.234 | **2.220** | 2.235 | 2.303 | `oracle` |

Splitting costs every expert data, so the bank's penalty should shrink as there
is more of it. It does, monotonically: the oracle's deficit runs 0.108, 0.095,
0.006 and then **−0.015** — at 232k characters a source-perfect router finally
beats one tree, and the learned filter draws level with it (2.235 against
2.234).

### Where the routing gap is

| measurement | purity | nmi | median \|u\| after | past first peak | mean `b` |
|---|---|---|---|---|---|
| the target layer 2 hands back (perfect experts) | **1.000** | **1.000** | — | — | — |
| supervised ceiling, sign code (best of 24 assignments) | 0.693 | 0.478 | 0.63 | 30.0% | 0.230 |
| supervised ceiling, argmax code, sine | 0.723 | 0.542 | 0.41 | 13.0% | 0.176 |
| supervised ceiling, argmax code, sine, wave held fixed | 0.292 | 0.008 | 143.54 | 100.0% | 0.333 |
| supervised ceiling, argmax code, tanh | 0.707 | 0.524 | 1.05 | 37.2% | 0.295 |
| supervised ceiling, argmax code, tanh, wave held fixed | 0.716 | 0.535 | 3.12 | 75.6% | 0.333 |

| one unit, supervised, asked for… | sine | tanh |
|---|---|---|
| `go + json` against the rest | 0.711 | 0.711 |
| `go + prose` against the rest | 0.797 | 0.816 |
| `go + python` against the rest | 0.636 | 0.733 |
| `go` alone against the rest | 0.667 | — |
| `json` alone against the rest | 0.967 | — |
| `prose` alone against the rest | 0.783 | — |
| `python` alone against the rest | 0.803 | — |

Four supervised measurements — things the architecture is never allowed to do —
that say where the gap between the learned router and the oracle sits.

**The signal is perfect.** Give the bank true experts and ask which one prices
each segment cheapest: it is the segment's own register, for every one of the
640 training segments, NMI 1.000. Nothing is wrong with what layer 2 tells
layer 1.

**Layer 1 is the bottleneck, and it falls short twice.** Fitted *directly on
the labels*, the filter reaches 0.69–0.72 purity, not 1.0 — a 17-number
histogram and one unit per question do not contain the partition. Unsupervised
it reaches 0.42–0.49. Roughly half the gap is representation and half is the
rule that has to find it without labels.

**A periodic unit is the wrong shape for a gate, and the measurement says why.**
A hinge raises a response by pushing the projection, so the wave's argument
`|u| = |b(z − h)|` grows. With the wave frozen it grows to a median of **143.5
radians — 100% of the units past their first peak**, twenty-two periods out,
and the filter scores 0.243 against a chance of 0.25: it cannot be fitted at
all. A tanh under the identical rule ends up past its own first peak too
(75.6%) and does not care, because saturation preserves an ordering and
periodicity does not — it scores 0.716. With the wave learnable the sine
recovers to 0.723, and the parameter that recovers it is `b`, which falls from
0.333 to **0.176**: what learning the wave mostly does in layer 1 is *flatten
it* until it is monotone over the range the data occupies.

That is `Research/Insights.md` §24's trap from the other end. There, `b = 1/3`
is too *low* — it puts every neuron in the sine's linear region and a deep
network collapses to a linear one. Here the same `b = 1/3` is what lets a hinge
push a gate right around the circle. One frequency, two opposite failures, and
the parameter that fixes both is the one the design already makes learnable.

### The refilter step's hinge passes

| hinge passes | bits/char (held out) | nmi | live addresses | seconds |
|---|---|---|---|---|
| 1 | 2.898 ± 0.019 | 0.194 | 3.3 | 125 |
| 2 | **2.812** ± 0.037 | 0.170 | 2.3 | 136 |
| 4 (default) | 2.908 ± 0.007 | 0.167 | 3.7 | 123 |
| 8 | 2.955 ± 0.124 | 0.234 | 3.0 | 122 |

The pricing half of the refilter step is expensive and the hinge half is nearly
free, so running the hinge more than once over the same prices costs almost
nothing — which is an argument for more passes, not a measurement. The
measurement finds no reliable effect: two passes score best, eight worst, and
the spread between seeds (up to ±0.124) is larger than the spread between
settings. The default stays at 4, and this table is here because that default
was chosen by an argument rather than by evidence.

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

## Saving and resuming

A fit costs minutes; asking the fitted bank a question should not. Everything
in a bank writes itself with `to_dict` and reads itself back with
`from_dict` — the trees as flat parallel lists, the filter's `v, a, b, h, k`,
the router, the shared prior — and every file goes through one atomic,
optionally-gzipped JSON writer.

```bash
make save                              # fit, write bank.json.gz, checkpoint each round
make checkpoints                       # list them, with the round's metrics
make predict TEXT='"'"'def main('"'"' LOAD=latest   # reuse it: 0.1s instead of 50
```

```
  /tmp/ckdemo  (keep=5)
    ckpt-round-000000.json.gz   step=0   1,767,118 B  2026-09-20T03:33:37  test=3.3426
  * ckpt-round-000001.json.gz   step=1   1,353,205 B  2026-09-20T03:33:54  test=3.0569
```

The layout is `radixnet`'s — `ckpt-<tag>-<step>.json[.gz]`, a `latest.json`
pointer, an `index.json` reconciled with the directory on every listing, and
rotation that never prunes the one `latest` points at. A reader who knows one
of these directories knows the other.

Two things are worth being precise about. **Both layers travel**, and every
expert's fallback is reattached to the one restored prior — an expert that lost
it would price an unknown context at the floor and quietly change every number.
And a restored bank **predicts identically but re-fits rather than resumes**:
the corpus does not travel, and `fit` rebuilds every expert from the segments
it is given. Resuming would mean *extending* experts, which is the one thing
the loop is not allowed to do (§6.1).

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

One corpus, cut out of this repository; four registers; 29k to 232k characters;
character-level prediction; one segmentation; 4 addresses in most runs; depths
2 to 6; three seeds on the headline table and one on the sweeps. Nothing here
says anything about routing at other granularities (per character, per
document), about more than four registers,
about a learned encoder in front of the filter, or about soft routing — a
softmax over experts would make the whole bank differentiable, and is the
obvious next thing to try, but it is not a *filter* and so it is not this
design.

The bits/char are a measurement of these models on this text, not of anything's
capability in general — the `counts` arm, which is an interpolated n-gram model
with no discounting worth the name, already beats every trained arm here. What
they are for is the comparison *between* the arms, which share every constant,
every smoothing rule and every seed.

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
| `fbradix/store.py` | JSON model files: atomic writes, transparent gzip, the format version |
| `fbradix/checkpoint.py` | `CheckpointManager` — rotation, the `latest` pointer, resume |
| `fbradix/cli.py` | `python3 -m fbradix.cli <command>` |
| `summarize.py` | the tables above, read back out of `results/*.json` |
| `data/`, `results/`, `tests/` | the corpus snapshot, the numbers, and 33 tests |

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
