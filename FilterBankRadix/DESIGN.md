# FilterBankRadix — Design Specification

**Layer 1 is an activation filter. It feeds a set of Radix Tree networks.**

That sentence is the whole architecture, and this document is the contract the
code is written against. One bank of learnable activation units reads a piece
of text and emits an **address**; the address selects one of `2^m` radix trees;
that tree, and only that tree, predicts the next character. Nothing else
crosses the boundary between the two layers — no representation, no gradient.

Directory: `FilterBankRadix/`. Python package: `fbradix`. Python 3.11+,
**standard library only**, no optional accelerator, deterministic given a seed.

---

## 1. The requirement, and where each part of it lives

| The requirement | How it is realised |
|---|---|
| **Layer 1 is an activation filter** | `filter.ActivationFilter`: `m` units, each `f_j(x) = a_j sin(b_j (v_j·x − h_j)) + k_j`, with `v_j` and all four activation parameters learnable. The unit is read for its **sign**, not its value: `s_j = [f_j > 0]`. |
| **…that feeds into…** | The only thing passed down is an **address** — the sign pattern of the units (`2^m` addresses) or the index of the largest response (`m` addresses), §4.5. Layer 2 never sees `x`, the responses, or anything derived from them; layer 1 never sees a character. |
| **…a set of Radix Tree networks** | `tree.RadixTreeNet` × `2^m`. Each is a path-compressed context trie over the characters routed to it, with a learnable sine on every node and the one-hop softmax rule of `RadixCyclicNN/radixnet/backend.py` on every branch. |
| the activation is the author's sine | `activation.py`: `f(x) = a·sin(b(x−h)) + k`, initialised to `a=−1, b=1/3, h=k=0`, i.e. exactly `−sin(x/3)`, in **both** layers. |
| the activation parameters learn slowly | `a, b, h, k` move at a tenth of the weights' rate in both layers (`ACT_RATE = 0.1`), and `b` is floored at `MIN_B = 1e-3` — section 8 of `Research/SineWaveActivationFunction.md`, and the failure mode `Experiments/ActivationFunctionTest/` finding 4 measured without it. |
| no back-propagation through depth | Layer 2's rule touches one edge, two nodes and their activation parameters. Layer 1's rule touches one unit. Between the layers passes one **number** — the difference in bits between two trees — never a gradient. Nothing in this architecture multiplies a chain of derivatives, so there is no vanishing gradient to fight. |
| compression | A run of nodes with one child each is one node holding the whole run: a deterministic continuation is stored once and costs no decision. |

---

## 2. Package layout

```
FilterBankRadix/
  DESIGN.md              this file — the contract
  README.md              what it measured, including what failed
  Makefile               make test | check | demo | experiment | quick | corpus | …
  summarize.py           the README's tables, read back out of results/*.json
  data/corpus.jsonl      the committed corpus snapshot (with a digest)
  results/               the numbers, committed beside the code that made them
  fbradix/
    __init__.py          the public surface
    __main__.py          python3 -m fbradix -> cli.main()
    store.py             JSON model files: atomic writes, transparent gzip, the version
    activation.py        the parametric sine, its partials, and the monotone control
    filter.py            LAYER 1: features, the units, the address, the hinge rule
    tree.py              LAYER 2: the radix context tree and the one-hop rule
    bank.py              the routers and the route/build/train/refilter loop
    corpus.py            the four-source corpus and the labels it is scored against
    experiment.py        the arms, the metrics, the sweeps and the probe
    checkpoint.py        CheckpointManager: rotation, the latest pointer, resume
    check.py             finite-difference checks of all three learning rules
    cli.py               python3 -m fbradix.cli <command>
  tests/test_fbradix.py  33 tests, standard library, seconds
```

---

## 3. `activation.py`

| symbol | meaning |
|---|---|
| `sine(x, a, b, h, k)` | `a·sin(b(x−h)) + k` |
| `sine_partials(x, …)` | `(f, df/dx, df/da, df/db, df/dh, df/dk)`, with `u = b(x−h)`: `a·b·cos u`, `sin u`, `a(x−h)cos u`, `−a·b·cos u`, `1` |
| `tanh_partials(x, …)` | the **monotone control**: `a·tanh(b(x−h)) + k`, the same four knobs in the same order, so the `monotone` arm differs in the shape of one function and in nothing else |
| `is_dead(a, b)` | `|a·b| ≤ 0.01`. Since `|f'| = |a·b·cos u| ≤ |a·b|`, this is a statement about *every* input at once |
| `MIN_B = 1e-3` | the floor applied to `b` after every update, in both layers |

The formulas are the ones in `RadixCyclicNN/radixnet/activation.py` and
`Experiments/SBNN_RNN_ActivationFunction/main.py`. This directory is
standalone, so they are repeated rather than imported.

---

## 4. `filter.py` — layer 1

### 4.1 The input

`features(text)` → 17 floats: a 16-bucket histogram of character codepoints
(`bucket = (ord(c) >> 3) % 16`) as **fractions scaled by 16**, plus a constant
`1.0` that gives each unit a bias. Scaling by the segment length is what makes
the filter length-blind — otherwise it learns to route by how long the cut was,
which is a property of the segmentation and not of the text. Nothing in the
vector is derived from the source labels.

### 4.2 The address

```
z_j = v_j · x                          m projections
f_j = a_j sin(b_j (z_j − h_j)) + k_j   m responses
s_j = [f_j > 0]                        m bits
r   = Σ_j s_j 2^j                      one address, 0 ≤ r < 2^m
```

`v_j ~ U(−1/√d, 1/√d)`; the activation parameters start at the defaults, so
every unit is born as `−sin(z/3)` and learns its own wave from there.

### 4.3 Calibration (once, before training)

`calibrate(xs)` subtracts each unit's **median response over the training
features** from its `k`. Every bit then splits the corpus in half. It reads the
features and never the labels, and the frozen arm gets it too: it is part of
drawing the filter, not part of learning it. Without it a unit can start with
every segment on one side of its boundary, and half the addresses are
unreachable before training begins.

### 4.4 The hinge

The only update layer 1 has. For a segment `x`, a unit `j` and a target sign
`y ∈ {+1, −1}`:

```
L = max(0, TAU − y·f_j(x))          TAU = 0.25
∂L/∂v_ji = −y · a b cos(u) · x_i    (zero when the hinge is inactive)
∂L/∂a = −y sin u   ∂L/∂b = −y a (z−h) cos u   ∂L/∂h = +y a b cos u   ∂L/∂k = −y
```

`v` steps at `lr`, `a, b, h, k` at `lr/10`, every gradient clipped to `±5`,
`b` floored at `MIN_B`. `hinge_grads` returns `None` exactly when the bit is
already on the right side with margin, which is also when every gradient is
zero. Checked against central differences by `check.check_filter` for both
`sine` and `monotone` (worst error ~1e-10).

### 4.5 The address code — `sign` or `argmax`

```
sign     r = Σ_j [f_j > 0] · 2^j        m units, 2^m addresses
argmax   r = argmax_j f_j               m units, m addresses
```

This is not a detail of encoding, and it is the single largest decision in
layer 1. **A sign-bit address makes every bit an independent dichotomy of the
data**: a partition is reachable only if it factors into `m` binary questions
the filter can answer, and the accuracy of the joint answer is roughly the
product of the parts. An argmax address has no such constraint — each expert
owns a unit, and the partition never has to factor — at a cost of `K` units
instead of `log2 K`.

Measured (`make probe`, §9): supervised on the labels, the sign code tops out
at 0.693 purity over the best of its 24 possible assignments, and the argmax
code at 0.723 with the sine. Both are far above what the unsupervised rule
reaches (0.42–0.49) and far below the oracle's 1.000, which is what puts the
gap in layer 1 rather than in the signal. In the bank the two codes score 2.908
and 2.887 bits/char — a difference inside the spread between seeds.

### 4.6 The argmax hinge

`push` asks one unit to be positive; `push_pair` asks one unit to **beat**
another:

```
L = max(0, TAU − (f_up − f_down))
∂L/∂f_up = −1        ∂L/∂f_down = +1
```

each chained through its own unit's partials — two one-unit updates in opposite
directions, still local, and checked by `check.check_pair` (worst error ~1e-10).

---

## 5. `tree.py` — layer 2, one expert

### 5.1 Structure

A **path-compressed context trie**. `insert_text(t)` inserts every window
`t[i : i+D+1]` (stride 1); a node's path spells a context, its children are the
characters seen after it, and a run of one-child nodes is one node holding the
run. `split(u, i)` is the radix split: `u` keeps `seg[:i]`, a new node takes
`seg[i:]` along with `u`'s children, count and learned state — it is the node
that still spells what `u` used to spell.

Per node: `seg`, `kids`, `par`, `cnt`, the in-edge weight `w` (a tree has one
parent per node, so the edge's weight lives on the child), and `z, a, b, h, k`.
`z ~ U(−4.5, 4.5)` and `w ~ U(0.5, 1.5)`, the draws `radixnet.graph` uses.

### 5.2 Scoring a branch

The children of `p` are scored `w_c · f_p(z_p) · f_c(z_c)` and normalised by a
softmax over the siblings — *the activation of the child times the activation
of the parent*, and the same rule as `radixnet`.

Worth noticing what that scoring can express. The parent's activation `f_p`
multiplies every sibling equally, so it acts as an **inverse temperature** on
the node's distribution, while `w_c · f_c` is a free per-child logit: setting
`w_c f_c = (log p_c) / f_p` reproduces any distribution `p` exactly. The rule
therefore has no representational disadvantage against counting the traversals,
and whatever gap the `baseline` sweep measures between them is **optimisation,
not capacity** — which is what the `epochs` sweep is for.

### 5.3 Prediction: every context length, folded longest-last

```
levels(ctx)  = the root, then each longer suffix of ctx the tree knows
p            = prior's answer (or 1/A)
for each level, shortest first:
    own = cnt / (cnt + ALPHA)        ALPHA = 2
    p   = own · p_here + (1 − own) · p
answer       = (1 − floor) · p + floor / A      floor = 0.02
```

`p_here` is 1 for the fixed continuation inside a run, the softmax entry at a
branch, and 0 for a character never seen after that context. Presence is
monotone — the tree holds every window of its own text, so if a suffix is
absent every longer one is too — which is why the walk can stop at the first
miss.

**This is a correction to an earlier design and the correction is
load-bearing.** Folding only the deepest matching level is worse in both
directions: the root's count is enormous, so `own` at the root is ≈ 1, and a
backed-off prediction then drowns out both the levels above it and the shared
prior. The product of `(1 − own)` down the chain is what stops that. Measured,
on the corpus of the day and before it was frozen: 2.998 → 2.808 bits/char for
one tree. It is also what makes a deep shared prior neutral instead of harmful
— `deep` against `learned` and `deep-oracle` against `oracle` now differ by
0.002 and 0.000 bits/char (`README.md`, "what went wrong first").

### 5.4 Training

`plan(texts)` collects `(branch node, chosen child)` **once per level per
position** — every node that takes part in a prediction is trained on it.
Structure alone decides the plan, so it is computed once per built tree and
reused by every epoch. A level inside a run contributes nothing: the softmax
over one child is 1 and every gradient is exactly zero.

`step(batch, lr, act_lr)` is the one-hop rule. For a decision `p → c*`:

```
g_c        = softmax_c − [c = c*]
∂L/∂w_c    = g_c · f_p · f_c
∂L/∂f_p    = Σ_c g_c · w_c · f_c        ∂L/∂f_c = g_c · w_c · f_p
```

chained into `z, a, b, h, k` of `p` and of every child through
`sine_partials`. Gradients are summed over the batch, divided by its size,
clipped to `±5` and applied once; `b` is floored. Checked against central
differences by `check.check_tree` (worst error ~3e-10) by reading the
analytic gradient *out of `step` itself*, so what is verified is the code that
trains.

---

## 6. `bank.py` — the two layers together

### 6.1 The loop

```
prepare   calibrate the filter on the training features
prior     build the shared fallback over all the training text
repeat rounds times:
  route    every training segment gets an address
  build    every expert is REBUILT from the segments addressed to it
  train    every expert runs the one-hop rule over its own text
  refilter layer 2 prices each segment under every live expert; layer 1 is
           pushed towards the address that priced it cheapest
```

**Rebuild, not extend.** A segment that moves must leave no trace behind, or
every expert slowly becomes the whole corpus and the partition stops meaning
anything. `test_the_bank_rebuilds_instead_of_extending` holds the line.

The filter is not updated after the last round: there would be no rebuild left
to act on it, and the reported numbers must belong to the routing that produced
the trees. A router that cannot learn partitions identically every round, so
`fit` runs exactly one round for it.

### 6.2 The refilter rule

For each training segment: price it under every **live** expert (one with
text), add `balance × load_e` in bits, take the cheapest as the address layer 1
should have produced, and hinge every disagreeing bit towards it with a step of
`lr · min(1, advantage / 1 bit)`. Advantages under `margin = 0.02` bits move
nothing. Where the filter already agrees, nothing is pushed (`reinforce`
defaults to 0; a full-strength push on the segments that are already right is a
constant force on a boundary with no error to correct, and measured it drags
clean clusters together).

`balance = 0.5` bits per unit of load is what stops the degenerate answer:
without it the cheapest expert for almost every segment is whichever tree holds
the most text, everything moves there, and the bank collapses to one tree with
a router in front of it. The `unbalanced` arm is that ablation.

### 6.3 The shared prior

One shallow tree (`prior_depth = 1`, a unigram) built over **all** the training
text and attached to every expert as its fallback, identically in every arm —
the single-tree baseline included, so it cannot flatter the bank. Its job is to
remove the *ignorance cliff*: without it an address with no text prices a
segment at ~11 bits/char against a trained expert's ~3, the counterfactual is
swamped by that constant, and the filter spends its updates avoiding a cliff
instead of learning a boundary. `test_ignorance_is_never_cheaper_than_knowledge`
is the regression: an expert must price its own register below what a foreign
expert prices it at. The `deep` and `deep-oracle` arms set `prior_depth = depth`
instead, which makes every expert a *correction to* a shared full model rather
than a replacement for it — at double the stored structure.

### 6.4 Addresses die; they are not born

A segment only moves to an address whose tree already prices it better, and an
empty tree prices nothing better than the prior. So the live set of addresses
is fixed by the first, random-but-calibrated routing and can only shrink.
`learn` therefore only proposes addresses that are already occupied — comparing
against a tree that does not exist wastes the update — and every result reports
`live_experts` rather than assuming `2^m`.

### 6.5 Routers

| router | address | learns | what it is for |
|---|---|---|---|
| `FilterRouter` | the filter's sign pattern | yes (`frozen=False`) | the architecture |
| `FilterRouter(frozen=True)` | the same, never updated | no | a calibrated random hash — the control that says whether *learning* the filter is worth anything |
| `FilterRouter(kind="monotone")` | `tanh` units, same knobs | yes | the control for periodicity |
| `FilterRouter(act_rate=0)` | the sine units with their wave held fixed | the projection only | the control for *learning the activation*: a filter unit can die along `(a, b)`, and a dead unit is a bit of the address stuck at a constant |
| `ConstantRouter` | always 0 | no | one radix tree: the baseline architecture |
| `CycleRouter` | round robin, stable per segment | no | an information-free split: the control that says routing, not splitting, would have to do the work |
| `OracleRouter` | the true source label | no | the ceiling, and the **only** arm ever shown a label |

---

## 6b. Persistence — one contract, and what a checkpoint of *this* contains

Every object that makes up a bank answers the same two methods, and every file
this package writes goes through the same two functions:

| | |
|---|---|
| `to_dict()` / `from_dict(d)` | `RadixTreeNet`, `ActivationFilter`, all four routers (each writes a `kind`, and `bank.router_from_dict` dispatches on it), and `FilteredRadixBank` itself |
| `store.write_json_atomic` / `store.read_json` | every file: models, checkpoints, the index, the `latest` pointer |
| `bank.save(path)` / `FilteredRadixBank.load(path)` | one bank, gzipped when the path ends in `.gz` |
| `CheckpointManager` | a directory of them: `ckpt-<tag>-<step:06d>.json[.gz]`, `latest.json`, `index.json`, rotation by `keep` |

The file rules are `radixnet`'s, deliberately — a model file is a thing you
lose work by getting wrong, and that package already settled it. Writes go to a
temporary file in the same directory, are `fsync`ed and then `os.replace`d, so
a reader never sees a half-written file and a crash leaves the previous version
intact. On reading, **the content decides, not the suffix**: a file is
gunzipped because it carries the gzip magic, so a plain-JSON file that happens
to end in `.gz` still loads.

**What travels in a checkpoint.** A bank is two layers fitted against each
other, so both go: the router with its filter's `v, a, b, h, k`, every expert
tree as flat parallel lists, the shared prior, and the routing that joined
them. Every expert's `fallback` is reattached to the one restored prior on
load — an expert that lost it would price an unknown context at the floor and
quietly change every number, which is §6.3's cliff coming back in through the
file format.

**What does not.** The corpus, and the fit history. So a restored bank
**predicts** exactly as it did — `test_a_saved_bank_predicts_identically`
checks routes, bits and a sampled string to the last digit — and **re-fits
rather than resumes**: `fit` rebuilds every expert from the segments it is
given (§6.1), which is the same rule that makes a round honest in the first
place. Resuming a fit from a checkpoint would mean extending experts, and that
is the one thing the loop is not allowed to do.

`fit(..., manager=cm, checkpoint_every=n)` writes a checkpoint every `n`
rounds and again at the end, carrying that round's metrics. A round is the
natural unit: it is the moment every expert has just been rebuilt and trained,
so a checkpoint is never half a fit.

---

## 7. Why a periodic unit, here

The sign of a monotone unit cuts its projection in two, so `m` monotone units
address the cells of an arrangement of `m` hyperplanes and no more. The sign of
a sine unit is a **square wave**: it cuts the same projection into unboundedly
many alternating bands, so one unit can separate sets that are not linearly
separable, and `m` units can in principle reach all `2^m` addresses along a
single projection direction. That is the reason to expect the periodic unit to
be the better filter, and `monotone` is the arm that tests it rather than
assuming it.

**Measured, the argument does not survive contact with the hinge.** Fitting an
address is done by pushing a response across a boundary, and a periodic
response *wraps*: the gradient points the right way locally, but pushing harder
carries the unit into the next lobe, where the answer is wrong again. With its
wave held fixed, a sine filter cannot be fitted to an argmax address at all —
0.243 supervised accuracy against a chance of 0.25, with **100% of its units
past their first peak**, a median of 143 radians out — while the same filter
with `a, b, h, k` free reaches 0.723, because what learning the wave mostly
does here is *flatten it* (`b` falls from 0.333 to 0.176) until it is monotone
over the range the data occupies. A tanh with its wave frozen goes just as far
out (75.6% past its own first peak) and does not care, because saturation
preserves an ordering and periodicity does not: it scores 0.716. In the bank
itself the tanh filter is the better router by 0.08 bits/char. The periodicity
that makes the sine a good *activation*
(`Research/SineWaveActivationFunction.md` §6: it does not die along `x`) is
what makes it an awkward *gate*.

The address is also, read as a binary number, a **radix-2 prefix over the
units** — a tree of experts keyed by that prefix is the same structure as the
character radix trees it selects between, one level up.

---

## 8. `corpus.py`

Four sources from the repository's own files — Python, Go, Markdown prose, JSON
results — cut into segments of 48…256 characters on line boundaries, the same
number of segments per source, spread across each source's files by a round
robin so a short file cannot starve the count. `split` is stratified by source:
an unstratified split can hand one register almost entirely to the test set,
and then every arm's bits/char is a measurement of that register alone.

The labels exist for two purposes only: scoring the routing afterwards
(`purity`, `nmi`), and the `oracle` arm. Nothing else is shown them.

`build` is pinned to an explicit file list, `digest` is the SHA-256 of the
corpus, and `save`/`load` keep a committed snapshot in `data/corpus.jsonl`, so
a rerun can tell whether the input drifted when the repository changed under it.
`synthetic` is the four made-up registers the tests use.

---

## 9. `experiment.py`

Metrics, all on held-out segments no arm trained on:

| metric | what it says |
|---|---|
| `test_bits_per_char` | the headline: `−log2 p` per character of the held-out text, under the expert each segment was routed to |
| `nodes`, `stored_chars` | the size of everything the arm built — a bank that wins on bits while storing twice as much has not won for free |
| `purity` | share of segments sitting with the majority source of their address; generous, since it rewards shattering one source across addresses |
| `nmi` | `2 I(R;L) / (H(R) + H(L))` — 0 when the address says nothing about the source, 1 when it determines it |
| `live_experts`, `balance` | how many addresses hold text, and the entropy of the load over `log2 k` |
| `dead_units`, `amp_freq` | `|a·b|` per filter unit, and which of them are below the dead threshold |

Sweeps: `main` (every arm × seeds, one corpus, one depth), `depth` (2…6 — a
router hands the model a fact the context would otherwise have to carry, so the
shallower the tree, the more a router should be worth), `scale` (100…800
segments per source — splitting costs every expert data, so whether a bank can
win at all is a question about how much text there is per expert), and `probe`.

`probe` is not an arm. It is four **supervised** measurements — things the
architecture is never allowed to do — that say *where* the gap between the
learned router and the oracle is:

1. give the bank perfect experts and ask which one prices each segment
   cheapest. That is exactly the target the refilter step chases;
2. fit the filter directly on the labels, under both address codes, and for the
   sign code under all 24 assignments of sources to addresses;
3. the three two-against-two splits of the four registers, one unit each — what
   a sign-bit address is made of;
4. each source against the rest — what an argmax unit is made of.

Every supervised fit runs `SUP_EPOCHS = 100` and is averaged over three seeds,
and reports `|a·b|` beside its accuracy. Both are there because of a measurement
that was wrong at 25 epochs: one split read 0.500, exactly chance, and the cause
was a unit whose `|a·b|` had collapsed to 0.0100 — it had died and was answering
a constant. At four times the epochs the same split read 0.777. A ceiling has
to be measured after the fit converges, and unit death has to be visible rather
than silently lowering the number.

---

## 10. Invariants

1. Layer 1 passes an **address** and nothing else; layer 2 passes a **scalar
   cost** back and nothing else. No gradient crosses between them.
2. Every gradient in the system is one hop. Nothing multiplies a chain.
3. The activation parameters move at a tenth of the weights' rate, and `b`
   never goes below `MIN_B`, in both layers.
4. Only `OracleRouter` ever sees a label. `FilterRouter.route` must give the
   same address whether or not one is passed.
5. Every arm gets the same smoothing constants (`floor`, `ALPHA`) and the same
   shared prior construction, or the bits/char are not comparable.
6. Experts are rebuilt each round, never extended.
7. `prob` is a probability distribution over the alphabet, for every context,
   in every arm (`test_the_prediction_is_a_probability_distribution`).
8. Everything is deterministic given the seeds: same call, same numbers.
9. Every part of a bank writes itself with `to_dict` and reads itself with
   `from_dict`, and every file goes through `store`. A restored bank predicts
   identically; nothing in the architecture is the one piece that cannot be
   written down.

---

## 11. Deliberately absent

* **No back-propagation, anywhere.** Not through the trees, not from the trees
  into the filter. The filter learns from a difference of two numbers.
* **No soft routing.** A segment goes to exactly one expert. A softmax over
  experts would make the whole bank differentiable and is the obvious next
  thing to try; it is not this design, because this design is a *filter*.
* **No growth.** Neither the number of addresses nor the depth of a tree
  changes during a fit. `2^m` is fixed by `m`.
* **No torch.** Standard library, one process, one thread.

---

## 12. Open questions

1. **Addresses cannot be born.** §6.4 is a real asymmetry. A rule that lets a
   dead address be reoccupied — seeding it from the segments the crowded
   experts fit worst, say — is untried.
2. **The filter optimises bits, not sources**, and the two come apart: it
   reaches a lower bits/char than a source-perfect router does in some
   settings while agreeing with the labels far less. What partition it is
   actually finding is measured (`contingency`) but not explained.
3. **One segment, one address.** Routing per character, per line or per
   document are all different architectures and only the middle one is built.
4. **The periodic unit is a liability for a router fitted by a hinge.** A sine
   with its wave held fixed cannot be fitted to an argmax address at all
   (0.245 against chance 0.25); it only works when `a, b, h, k` are free to
   flatten it. The gradient of a periodic unit points the right way locally,
   but the response wraps, so pushing harder can carry it into the next lobe.
   Whether a unit that is periodic in `z` but monotone over the *observed*
   range is the right compromise is untried.
5. **The features are a histogram.** A learned encoder in front of the filter
   would make layer 1 two layers; whether the bucket histogram is the binding
   constraint on routing quality is not known.
