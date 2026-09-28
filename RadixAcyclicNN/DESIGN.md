# RadixAcyclicNN — Design

The specification of `radixtree`: what the code must do, module by module.
`README.md` says how to run it and what it measured; the last section here
records the decisions and what each one cost.

The whole design is one sentence long. **`RadixCyclicNN`'s network, with a
node made a context instead of a gram.** Everything the cyclic graph does — the
sliding window of characters, the sine activation on every node, the one-hop
local learning rule, the `-log P` cost, the shortest-path prediction, the radix
split and merge, the inversion of 2NRL — is kept, operation for operation, so
that the only thing the two models differ in is the one thing this directory
exists to measure: what a cycle buys, and what it costs.

---

## 1. The structural change, and what follows from it

In `radixnet.graph.RadixCyclicGraph` a trigram lives in exactly one node
(`trigram_index`). The second occurrence of `"aaa"` in `"aaaa"` is therefore an
edge from the node back to itself, and the graph has cycles — by design
(`RadixCyclicNN/DECISIONS.md` D-001, `Research/CyclesAreAFeature.md`).

Here a node is identified by its **path from the root**. The second `"aaa"` is
a different node from the first, because it stands in a different context.
Every edge runs from a node to its child, one level deeper, so:

| in the cyclic graph | here |
|---|---|
| a gram lives in one node | a gram lives in as many nodes as it has contexts |
| repetition is an edge backwards: stored once, any length | repetition is unrolled: one node per occurrence, this length only |
| search state is `(node, chars_emitted)`, with an expansion budget | search state is the node; no budget, no visited set |
| costs must be `>= 0` (a negative cycle would be `-inf`) | costs may be signed; there is no cycle to go round |
| no topological order; the rule is local so it does not need one | depth is a topological order |
| targeted inversion of a path is best-effort (odd cycles) | exact: a tree is bipartite |
| `END` is one node with many parents | every context that ended a text has its own END leaf |
| a merge may create a self-loop | a merge cannot |

The price is stated in §16 and measured in `README.md`: the structure grows
with the corpus rather than with its diversity, it cannot say "this repeats",
and it trains every context a position passes through.

---

## 2. `encoding.py`

A sliding window of `n` characters at stride 1 (`Encoding(n=3)` by default).
`encode("hello") -> ["hel", "ell", "llo"]`; consecutive grams overlap by
`n - 1` characters, which is what lets a run of them merge back into text.
`Encoding(n=1)` is the plain character trie. Characters only: the word,
phonetic and acoustic units of the cyclic model are out of scope here (§17).

```python
class Encoding:            # frozen
    n: int = 3
    overlap -> n - 1
    encode(text) -> list[str]            # [] when len(text) < n
    check_grams(grams)                   # ValueError: wrong length, or no overlap
    decode_grams(grams) -> str           # first gram whole, then each one's new character
    grams_held(label) -> int             # len(label) - n + 1
    gram_at(label, i) -> str
    decode_path(labels, start_offset=0, include_context=True) -> str
    to_dict() / from_dict(d)
```

`decode_path` is the cyclic model's: every label after the first contributes
its part past the overlap; the first contributes `label[start_offset:]` with
the context or `label[start_offset + n:]` without it — the deterministic
remainder of a compressed node after the matched gram. `start_offset` is a
gram index.

`START_LABEL = "<s>"` and `END_LABEL = "</s>"` are reserved for the sentinels,
but a real node may carry the same text (`"x<s>y"` holds the trigram `"<s>"`),
so the tree tells sentinels apart by *kind* (§3), never by label.

---

## 3. `tree.py` — storage

Flat parallel lists indexed by node id. Removed nodes are tombstoned (ids never
reused) and compacted only by `to_dict`.

```python
ROOT, START = 0, 1                 # FIRST = 2: the first id that is a real node or an END leaf
KIND_ROOT, KIND_START, KIND_END, KIND_REAL = 0, 1, 2, 3

class RadixTree:
    labels:   list[str]            # "" for the root, "<s>" for START, "</s>" for an END leaf, grams merged for a real node
    kind:     list[int]
    parent:   list[int]            # -1 for the root
    children: list[dict[str, int]] # real children, keyed by the child's FIRST GRAM
    end_leaf: list[int]            # the END leaf under this node, or -1
    count:    list[int]            # visits: windows that passed through or ended here
    w:        list[float]          # the weight of the node's IN-EDGE (0.0 for the root)
    z, a, b, h, k: list[float]     # node state and activation parameters (defaults -1, 1/3, 0, 0)
    alive:    list[bool]
    grams:    set[str]             # every distinct gram observed
    rng:      random.Random        # seeded
    inverted: bool
    version, structure_version: int
    depth:    int | None           # symbols a root path may hold (START counts, END does not); None = unbounded
```

The root is the empty context. START is the root's child and the context *at
the beginning of a text*; it is never an option of a walk (`child_items(ROOT)`
does not list it — a walk begins at START, it does not go there). An END leaf
belongs to one context: a shared END would have many parents and the structure
would stop being a tree.

Because every node but the root has exactly one in-edge, the edge weight lives
on the child: `w[c]` is the weight of `parent[c] -> c`. There is no edge table.

New node: `z ~ U(-4.5, 4.5)`, `a, b, h, k` = defaults, `count = 0`; new in-edge
`w ~ U(0.5, 1.5)`. While `inverted` is true a new node gets `a = -DEFAULT_A`
and `w` negated — the cyclic graph's rule. The draws come from `rng` in the
order `z`, then `w`.

```python
is_real(i), is_end(i), held(i), label_len(i), first_gram(i)
num_nodes()      # alive real nodes (the sentinels are not counted)
num_ends()       # alive END leaves
num_edges()      # learnable in-edges: num_nodes() + num_ends()
num_grams()      # len(grams)
compression_ratio()   # grams / num_nodes() - the cyclic graph's metric, on a structure where it can fall below 1
label_chars()    # characters held in real labels: the tree's real size
depth_of(i), max_depth(), branches()
child_items(p)   # the options at p: its real children, then its END leaf
num_children(p)
```

---

## 4. Observation: every suffix, radix insertion, split

### 4.1 Windows

A text of grams `g0 .. gL-1` is observed as `L + 1` windows
(`RadixTree._windows`):

* from START: `START, g0, g1, ...` — with a bound, the first `depth - 1`
  grams; it *ends* (gets an END leaf) when the whole text fits;
* from the root, at every position `s`: `gs, gs+1, ...` — with a bound, `depth`
  grams; it ends when it reaches the end of the text.

So every substring of every text is a root path, the way a suffix tree holds a
text, and a walk from the root along the last few grams of any prefix lands on
the deepest context the corpus has seen (§6). Without a bound every root path
ends in an END leaf; with one a path may end where its window ran out. The
position `s = 0` is inserted from the root too: the beginning of a text is also
a place in it.

### 4.2 Insertion (`_insert`)

Standard radix insertion under an origin node, matching grams one at a time
(a gram matches when its last character matches the label's, given the
overlap):

* no child for the next gram → a **new leaf holding the whole remainder** of
  the window (plus an END leaf when the window ends);
* the window diverges inside a child's label → **split** the child there and
  add the new leaf under the first half;
* the window ends inside a child's label → if the text ends there, split so the
  END leaf hangs off exactly that context; otherwise only the count moves;
* the window matches the whole label → descend.

Returns the transitions the window took: every `(parent, child)` edge, END
leaves included; a step inside a label is deterministic and is not one. The
origin's counter is bumped once per window, each node's once per visit.

### 4.3 `split(node, i)`

The cyclic graph's split. `A` keeps `node`'s id, its in-edge and
`label[:i + n - 1]`; `B` is a new node with `label[i:]` that inherits `A`'s
children, END leaf, state, activation parameters and count, and gets a fresh
in-edge weight. `1 <= i < held(node)`; sentinels cannot be split.

### 4.4 Transitions that hold

A later window may split a node that an earlier transition named as a parent —
its children moved to the new deep half. `observe_sequence` therefore re-reads
every window of the text against the structure as it stands when a split
happened during the call, and `RadixTreeNet._observe` does the same across
texts (§9). A second pass creates nothing: every window is a root path by then.

### 4.5 Observation never leaves a unary chain

A real node is created holding a whole remainder, or cut at a branch; a leaf
that a later window extends already has an END leaf, so it gains a second
option, not a chain. This holds at every depth: a non-ended window always
reaches exactly `depth` symbols and a label never reaches past it. So a tree
built by `observe_sequence` alone is compressed already, and `compress()`
returns 0 on it. Only `split` (§4.3) leaves chains — the operation a dynamic
window would be built on.

---

## 5. Merge and compress

```python
merge_child(p) -> bool     # p real, exactly one option, that option a real node
compress() -> int          # merge until none remains
```

`p` takes `labels[p] + labels[c][n - 1:]`, `c`'s children and END leaf, and the
larger of the two counts; `c` is tombstoned. An END leaf is never merged away
(the trie clause of `GREN/gren/radix.py`, in the tree's currency).

The merge preserves the network function exactly as the cyclic graph's does:
the merged node keeps the activation of whichever endpoint has the larger
`|f|`, and the edges on the other side are rescaled by the ratio of the two
activations — `c`'s out-edges by `f_c / f_p` when `p`'s activation is kept,
`p`'s in-edge by `f_p / f_c` when `c`'s is adopted — so every remaining edge
score `w * f_parent * f_child`, and every probability and cost, is unchanged.
The edge `p -> c` itself was the only option at `p`: probability 1, cost 0,
nothing lost by removing it.

---

## 6. Walking and locating

```python
walk(origin, grams) -> (node, offset) | None
    # offset: index of the last matched gram inside the node's label; -1 for an empty walk on the origin
    # offset == held(node) - 1: the walk landed ON the node (its options begin); smaller: inside a run
at_end(node, offset) -> bool
trace(grams, origin=START) -> (transitions, node_path) | None   # a WHOLE text: [origin, ..., END leaf]
node_path(grams, origin=START)
usable(node, offset, min_count=1) -> bool
locate(grams, min_count=1) -> (node, offset, symbols) | None
```

`locate` finds **the deepest usable context of a history**: the history from
START first (the whole of it, when the window can hold it), then its suffixes
from the root, longest first — drop the oldest gram and ask again. A context is
*usable* when it has been seen `min_count` times and has something to continue
with: a run to finish, a child, or an END leaf. `symbols` counts START.
`None` when not even the last gram is known.

A context of exactly `depth` symbols stands at the end of its window: it can
never have seen a next gram, only whether texts ended there. So it is consulted
for END alone (`locate(..., ending=True)`, what `score` asks before the END),
and a gram is asked of a context of at most `depth - 1` symbols. Without that
distinction a history whose last `depth` grams had ended a line elsewhere would
match that line's END context and miss on its own next gram — a text the tree
was trained on must never miss, at any depth, and the tests hold it to that.

`min_count` is the trust knob. At 1 the deepest known context always decides;
at 2 a context seen once is never consulted and the walk falls back to the
longest suffix seen twice; large values collapse the tree toward the root's
order-0 model. `README.md` measures what it does to held-out text.

---

## 7. Activation, scores, probabilities, costs

`activation.py` is the reference sine, repeated rather than imported
(`f(x) = a * sin(b * (x - h)) + k`, defaults `-sin(x / 3)`, `MIN_B = 1e-3`;
`SineActivation.inverted()` negates `a` **and** `k`).

```python
activation_of(i)                       # f_i(z_i); cached per version
child_scores(p)   -> [(c, w[c] * f_p * f_c)]           over child_items(p)
child_probs(p)    -> softmax(child_scores)             numerically stable
child_costs(p, costs="logprob") -> [(c, cost)]
    # "logprob": -log softmax  (>= 0, the cyclic graph's cost; cached per version)
    # "signal":  -(w * f_p * f_c) — may be negative; legal only because there are no cycles
log_prob(p, c) -> float | None
```

---

## 8. `model.py` — the learning rule

The one-hop local rule of `radixnet.backend`, unchanged. For a transition
`p -> c` the loss is the negative log-softmax of the score `w_c * f_p * f_c`
among the options at `p`; the gradient touches the in-edge weights of `p`'s
children, and `z, a, b, h, k` of `p` and its children. Nothing propagates
deeper (`RadixCyclicNN/DECISIONS.md` D-004). With `u = b (z - h)`:

```
df/dz = a b cos u    df/da = sin u    df/db = a (z - h) cos u    df/dh = -a b cos u    df/dk = 1
```

`step(batch, lr, act_lr, clip=5.0)`: gradients summed over the batch, divided
by its size, clipped element-wise to `[-clip, clip]`, applied once — `w` and
`z` at `lr`, `a, b, h, k` at `act_lr`; `b` kept `>= MIN_B`. Returns the mean
loss before the update. A parent with one option contributes loss 0 and no
gradient (a softmax over one is 1) and is skipped.

Two implementation facts, both exact:

* **A batch is grouped by parent.** The loss and its gradient are sums over
  transitions, so a parent that occurs `m` times in the batch with its children
  as targets `n_c` times contributes `m * logsumexp - sum(n_c * score_c)` and
  `m * q_c - n_c` at every child: one softmax per parent per batch instead of
  one per transition, the same numbers to the last bit of arithmetic order
  (`tests`: `test_grouping_by_parent_is_exact`).
* **Accumulators are flat per-node arrays** kept between calls and reset entry
  by entry, so a batch costs its parents' fan-out and nothing of the tree's
  size — `radixnet.backend._PyState` does the same.

`loss_of(batch)` and `gradients_of(batch)` evaluate without updating;
`check.py` reads the analytic gradient out of `step` itself and holds it
against central differences (~1e-9).

`TrainConfig`: `epochs 5, lr 0.05, act_lr 0.005, batch_size 256, clip 5.0,
auto_compress True, shuffle True, verbose False` — the cyclic model's
defaults. The tests and the comparison train at `lr 0.5, act_lr 0.05, batch
32`, which learns visibly in a few epochs.

---

## 9. Training

`RadixTreeNet.train(texts, config=None, *, phase=None, progress=None, **overrides)`:

1. drop texts shorter than a gram (counted as `skipped_short`);
2. observe every text — every window, from START and from the root — with
   counting, and collect the transitions; re-read them when the structure
   changed during the pass (§4.4);
3. `compress()` when `auto_compress` (a no-op on a tree built by observation,
   §4.5); re-read the transitions if a merge moved nodes;
4. per epoch: shuffle the transitions with the tree's seeded RNG, feed them to
   `step` in mini-batches, `compress()` again, record
   `{epoch, loss, perplexity, nodes, ends, edges, grams, compression_ratio,
   merges, transitions, seconds, skipped_short, lr, act_lr[, phase]}`.

Every transition of every window trains: a corpus position trains every
context that predicts it — the root's order-0 choice, START's opening, and each
suffix context up to the depth. That is why the tree sees several times the
cyclic graph's transitions per epoch (§16).

---

## 10. Scoring

`score(text) -> {log_prob, per_char, chars, transitions, unknown_transitions}`.

The text is walked from START through **the deepest context it has at every
step**: along the tree while the context continues, and from the longest known
suffix of what was read so far whenever it does not — a window that ran out, a
context seen fewer than `min_count` times, or a gram the context was never
followed by. Every edge taken contributes `log softmax` of its score; a step
inside a compressed node is deterministic (cost 0) and not a transition; a
gram the deepest context was never followed by contributes `log(UNKNOWN_PROB)`
(`1e-6`, the cyclic model's constant) and counts as an unknown transition. The
context never peeks: a shorter context that would have known the gram does not
rescue a deeper one that did not. The END is scored the same way, from the
context the text ends in.

The walk is incremental — it relocates only when it must — and equals the rule
stated the slow way, "locate the deepest usable context at every symbol": a
usable context's continuation is the deepest usable context of the longer
history, because a prefix of a root path is a root path and counts never grow
down a path (`tests`: `test_incremental_scoring_equals_relocating_at_every_step`).

`bits_per_char(texts) -> {bits_per_char, chars, transitions, unknown_transitions, miss_rate}`.

What the number means: under both models a miss costs `-log2(1e-6) ≈ 19.9`
bits, so `bits_per_char` is dominated by the miss rate on held-out text and by
the softmaxes on trained text. The two are reported side by side for that
reason.

---

## 11. Search

### 11.1 `cheapest_path` (`search.py`)

Dijkstra from `(start_node, start_offset)`. A step over `p -> c` costs
`child_costs(p)[c] + step_penalty` and emits `len(label_c) - overlap`
characters; an END leaf emits nothing; the start node emits its remainder
after the matched gram. Goal: an END leaf, or (unless `to_end`) the first node
on a path with at least `min_chars` emitted; a path stops at its first goal.
`max_chars` caps expansion and the text. If no goal is reachable (a bounded
window ran out first) the visited node with the most characters emitted (ties:
lowest cost) comes back; the search never raises for a valid start.

What is *not* there, and why: no `(node, chars_emitted)` state, no `best`
dictionary, no expansion budget. Every node is reached by exactly one path, so
a node is pushed exactly once and the frontier is finite by construction.
`expanded` counts nodes visited and is at most the tree's size
(`tests`: `test_each_node_is_visited_at_most_once`).

Under non-negative costs (`costs="logprob"`, `step_penalty >= 0`) the first
goal popped is the cheapest and the search stops there. Under `costs="signal"`
or a negative penalty the early stop is invalid, so the search walks the whole
subtree below the start (still one visit per node) and keeps the cheapest goal
it saw. The cyclic graph must reject a negative penalty
(`radixnet.search.dijkstra_predict`); here it is a bonus per step.

### 11.2 `sample_walk`

A stochastic walk sampling each option from `softmax(-cost / temperature)`
(0 is greedy), stopping at an END leaf, at `max_chars`, or at a node with no
options. `step_costs` are the model's costs of the chosen edges.

### 11.3 `predict` — legs and the clock

`RadixTreeNet.predict(prefix, length=20, mode="dijkstra", step_penalty=0.0,
temperature=1.0, to_end=False, max_length=None, costs="logprob", seed=None,
max_legs=64)`.

The walk begins at `_start(prefix)`: the deepest usable context of the prefix,
continued from the matched gram (the continuation only); a prefix shorter than
a gram matches the most visited child of START whose label begins with it, and
the characters it already covers are cut from the first leg; nothing known →
START, a new text, with the context included.

On a tree of every suffix a walk is one leg: it ends at END or at `length`.
With a bounded `depth` a path can run out where its window did; the walk then
**re-enters** the tree at the deepest context of everything said so far
(`locate(prefix + emitted)`) and goes on, at most `max_legs` times, stopping
when a leg emits nothing. The tree has no cycles; the re-entry is a loop
*outside* it, and `max_legs` is that loop's clock — the same monotone resource
the cyclic search needs inside its graph, needed here only when the window is
bounded and the request is longer than the window.

`PathResult`: `text, labels, node_ids, cost, step_costs, expanded,
reached_end, full_text, legs`.

`generate(max_length=40, count=1, mode="sample", temperature=1.0, seed=None,
step_penalty=0.0, to_end=False, costs="logprob")` runs `predict("")`
`count` times (one, for the cheapest path).

---

## 12. Inversion and 2NRL

`invert()`: negate every in-edge weight and every node's `a` and `k`; toggle
`inverted`. Every edge signal is exactly negated, every ranking reverses
exactly, two inversions are the identity — the cyclic graph's operation and
reasoning (`RadixCyclicNN/DESIGN.md` §5.2).

`flip_nodes(nodes | {node: amount}, mode="activation" | "state", amount=1.0)`:
`value *= 1 - 2 * amount` on `(a, k)` or on `z`. An END leaf may be flipped: it
belongs to one context, so its sign is that path's alone. The root and START
are never flipped.

`invert_paths(texts, mode, amounts)`: for each text's path (registered
structurally first if it is not a root path yet), flip every other node after
START — the real nodes and the path's own END leaf — choosing the parity that
flips the most edges given what earlier texts already chose; a shared node
takes the largest amount. **On a tree this is exact for one text**: the path
is simple, the tree is bipartite, and the END leaf is private, so every edge of
the path changes sign (`tests`: `test_path_inversion_is_exact_on_a_tree`). In
the cyclic graph the same operation is best-effort — an odd cycle cannot be
two-coloured and a self-loop is sign-locked (`Research/CyclesAreAFeature.md`
§5.3) — and END is shared, so the last edge of an even-length path cannot be
covered.

`two_nrl(bad, good, neg_epochs=3, pos_epochs=3, neg_lr=0.05, pos_lr=0.01,
**overrides)`: train on `bad`, invert, fine-tune on `good` at `pos_lr` (the
activation parameters at `pos_lr / 10`). Records carry `phase`.

What 2NRL needs on a tree: **a branch**. The negative phase can only teach
where a node has more than one option; a garbage text the tree has never seen
is a corridor — one option at every step, no gradient — so the failures must
be shown to a model that already holds the good texts, where the swapped word
is a real branch. There the inversion reverses every pairwise preference
exactly (the count of branches preferring the good continuation after it is
the complement of the count before it), and the positive phase leaves the good
continuation preferred at nearly every branch (`tests`: `test_two_nrl`).

---

## 13. Invariants (`check_invariants(texts=None, compressed=False)`)

Asserted after every operation the tests make, and by the randomised test
after hundreds of random observe / split / compress operations.

1. The root is alive, kind ROOT, parent `-1`, label `""`; START is alive, kind
   START, the root's child, label `"<s>"`; START is not among the root's options.
2. Every alive node but the root has an alive parent that lists it under the
   right key (a real node by its first gram, an END leaf as `end_leaf`).
3. **No cycles, three ways:** following parents from any node reaches the root
   without meeting a node twice; a walk down from the root through the
   children visits every alive node exactly once; and the parents and the
   children agree on one tree. (`depth_of(child) == depth_of(parent) + 1` is a
   corollary the tests also check.)
4. A real label holds `n` or more characters; a real parent and a real child
   overlap by `n - 1` characters; an END leaf has no children.
5. Every gram of every real label is in `grams`.
6. Counts are non-negative; on a tree of every suffix, a node's count equals
   the sum of its options' counts (a window passes through, or ends).
7. With `compressed=True`: no real node has exactly one option that is a real
   node.
8. With `texts`: every window of every text is a root path, an ended window
   ends on a node with an END leaf, and — when the whole text fits under the
   depth — the text is a root path from START.
9. Dead nodes are empty: label `""`, no children, no END leaf, parent `-1`.

The counters `_n_real` / `_n_ends` match what is alive.

---

## 14. Persistence

`RadixTree.to_dict()` (format `radixtree-tree`, version 1): `seed`,
`encoding`, `depth`, `inverted`, `version`, `structure_version`, `grams`
(sorted), `rng_state`, and `nodes` — flat lists `labels, kind, parent, count,
w, z, a, b, h, k` over the alive nodes, compacted, the root and START first.
`children` and `end_leaf` are not written: a real node is keyed by its first
gram and an END leaf is its parent's, so `from_dict` rebuilds both from
`parent` and `kind`. The RNG state travels, so training after a load is the
training that would have followed a save.

`RadixTreeNet.to_dict()` (format `radixtree`, version 1): `kind`, `saved_at`,
`min_count`, `meta`, `history`, `tree`. `save(path)` writes JSON atomically
(temporary file, `fsync`, `os.replace`), gzipped when the path ends in `.gz`;
`load` reads by content (the gzip magic), not by suffix.

---

## 15. The comparison (`compare.py`)

`compare(train_texts, test_texts, seed, epochs, lr, act_lr, batch_size, depth,
min_count, prefixes)` trains a `radixnet.RadixNet` (imported from the sibling
`../RadixCyclicNN` checkout; the result says so when it cannot be) and a
`RadixTreeNet` on the same texts under the same numbers, and reports, for each:

| field | what it is |
|---|---|
| `stats` | real nodes, edges, distinct grams, label characters, grams per node (and for the tree: END leaves, branches, max depth) |
| `train_seconds`, `epoch_loss`, `transitions_per_epoch` | the cost of training, and the rule's own loss over its transitions |
| `train_bits`, `test_bits` | `bits_per_char` and the miss rate under each model's own `score` (§10; the cyclic model's `score` uses the same `1e-6`) |
| `recall` | how many training texts the model recites whole from their first 8 characters (`predict(..., to_end=True)`) |
| `predictions` | the continuation, cost, nodes expanded and time for each of `PREFIXES` |

and, once per run: `repetition` — the nodes and label characters each
structure needs for `"a" * n` and `"abc" * k` (the table of
`Research/CyclesAreAFeature.md` §3, measured) — and `generalisation`: shown
`"aaaa"` once, whether `"aaaaaaa"` is a path of each structure and what each
charges for it.

The corpus for the prose run is `corpus.py`'s: prose lines of the four research
papers, pinned by name, snapshotted with a digest (`data/README.md`), every
fifth line held out. `markdown(result)` prints the tables `README.md` carries;
if the two ever disagree, the JSON in `results/` is right.

---

## 16. Costs and limits

* **Labels are copied, not sliced.** A leaf holds the whole remainder of its
  window, so a tree of every suffix holds, across its labels, on the order of
  the square of the text — the prose run stores 1.5 million label characters
  for a 44-thousand-character corpus. A suffix tree proper keeps `(start, end)`
  offsets into the text instead; this one keeps the cyclic graph's
  representation so that a label reads the same way in both. A bound on
  `depth` caps a label at `depth` grams and the whole at `O(text * depth)`.
* **Training cost is the empty context's fan-out.** The root has an option per
  distinct gram (4228 on the prose corpus) and is in nearly every batch, and
  the one-hop rule costs a parent's fan-out per batch; that, and several times
  the graph's transitions (§9), is why the tree trains two orders of magnitude
  slower than the graph at batch 32. The levers are `batch_size` (the root is
  visited once per batch: 256 is five times cheaper than 32) and `depth`.
* **A held-out text pays for every corridor it leaves.** The deepest context
  decides (§10), and on a tree of every suffix most deep contexts have been
  seen once; a held-out text that enters one and leaves it pays a miss.
  `min_count` is the remedy, measured in `README.md`.
* No dynamic window, no word or phonetic units, no BACK / THINK sentinels, no
  server or frontend: this is the acyclic core, sized like `FilterBankRadix`,
  not the cyclic product.

---

## 17. Decisions

**T-01 — A node is a context.** The one change. Reversing it gives the cyclic
graph back. *Cost:* repetition unrolled; the structure grows with the text.

**T-02 — Every suffix is inserted, from the root at every position.** Without
it a node is only reachable from START, and the tree cannot answer for a prefix
that does not begin a text — the thing the cyclic graph answers through its
gram index. The beginning of a text is inserted from the root too. *Cost:* the
root's fan-out (§16), and `L + 1` windows per text.

**T-03 — START is a node under the root; every context has its own END leaf.**
START is a context (the beginning) and a walk's origin, never an option. A
shared END would be a node with many parents — not a tree — and its activation
would be one number for every ending; a private END leaf is a genuine node of
the context and can take part in a path's parity (§12). *Cost:* an END leaf
per ended context (one per text per position without a bound).

**T-04 — Same rule, same cost, same search, same draws.** Every operation the
cyclic graph has is kept with its numbers: the activation and its defaults, the
one-hop rule and its clipping, `-log softmax`, Dijkstra's goal and fallback,
`U(-4.5, 4.5)` states and `U(0.5, 1.5)` weights, `1e-6` for the unknown. A
comparison between two models that differ in two things measures neither.

**T-05 — The deepest context decides, and a miss costs `UNKNOWN_PROB`.** The
cyclic model's convention for its scores, so the two read on one scale.
Smoothing and back-off by escape are `FilterBankRadix`'s tree's business, not
this control's. *Cost:* the held-out miss rate in `README.md`.

**T-06 — `min_count` is the only trust knob.** A context seen fewer times is
not consulted, and the walk falls back by suffix. One integer, no escape
probabilities, no interpolation weights. *Cost:* it is coarse — the tree at
`min_count = 2` cannot use anything it saw once.

**T-07 — Signed costs are allowed.** They are what the absence of cycles buys
the search, so they are exposed (`costs="signal"`, a negative `step_penalty`)
and tested rather than forbidden for parity's sake. The default is the cyclic
graph's cost.

**T-08 — Re-entry is a loop outside the tree, with a clock.** A bounded window
generating past its window has to start again from what it said, and that
loop needs `max_legs`. Stated rather than hidden: it is the one place the
acyclic model needs the thing the cyclic one needs everywhere.

**T-09 — Standalone package.** Standard library only, no import from
`radixnet`; the sine and the JSON store are repeated, as `FilterBankRadix`
repeats them. The comparison alone imports the sibling, and says so when it
cannot.

**T-10 — Labels are copied.** The cyclic graph's representation, kept so a
label reads the same way in both models and the decoder is shared in spirit.
*Cost:* §16; the offset representation is the obvious next step if this
structure is ever asked to hold a large corpus.
