# The dynamic window — a ladder of node sizes, halving from 32 to 4 and back up

**Status** Built, in all three implementations: `radixnet/window.py` (with
`RadixCyclicGraph.split_window` and the window-aware `merge_child` in
`radixnet/graph.py`, and `GraphModel.window_config / configure_window /
window_step / _window_epoch` in `radixnet/model.py`), `go/radixnet/window.go`
and `rust/src/window.rs`. Off by default everywhere; switched with `radixnet
window --on` (`radixnet-count window`, the Rust `radixnet window`), `POST
/api/model/window` on all three servers, and the Dynamic window card on the
frontend's Model settings tab. D-087 is the decision.

**Answers** *"implement a dynamic window that is sized in the binary number
system. It starts at 32, then moves to 16, then 8, then maybe 4. This algorithm
will split nodes into two parts and assign the same weights and data for
traversal as well as a heavy connection for the two halves. This process
happens manually or automatically. Then we size up the windows back to 32 and
do this process again"* - applied to the actual nodes of the radix cyclic
graph: a node with the value `ABCD` becomes `AB` and `CD`.

---

## 1. The problem, precisely

Compression (D-007, `DESIGN.md` §5) merges every unary chain into one node,
however long: a sentence the corpus never branched inside is one node holding
every gram of it. A walk enters such a node at its first gram and leaves it at
its last; the steps inside are deterministic - no edge, no count, no weight,
nothing to learn and nowhere to branch. The graph's granularity is therefore
whatever the corpus's branching happened to leave, and it only ever coarsens:
a node is split when a transition is observed into or out of its middle, and
merged back the moment the chain is unary again.

The **dynamic window** is a ceiling on that length, and the ceiling moves.

## 2. The window

The window is a size in the encoding's **units** (characters by default, words
under a word encoding: what `label_len` measures) and it is **sized in the
binary number system**: it walks a ladder of powers of two from a `top` down to
a `floor`, and from the floor back to the top.

```
top 32, floor 4:   32 -> 16 -> 8 -> 4 -> 32 -> 16 -> ...
```

`DynamicWindow(top, floor, size, auto)`: `top` and `floor` are powers of two
with `floor <= top` (defaults `32` and `4`); `size` is where the window stands
now, a rung of that ladder (the top when it is switched on); `auto` says
whether it steps by itself at the end of every training epoch (the default) or
only by hand. Off - `DynamicWindow()`, the zero value, and every model before
this existed - it does nothing: compression is unbounded and no node is ever
halved. The refusals, word for word in the three implementations:

| given | refused with |
|---|---|
| a top or floor that is not a power of two | `the window is sized in the binary number system: top must be a power of two, got 20` |
| a floor over the top | `floor must not exceed top, got floor 64 over top 32` |
| a size off the ladder | `size must be a power of two on the ladder 4..32, got 12` |
| a step while off | `the dynamic window is off: switch it on first (window --on)` |
| fewer than one step | `steps must be >= 1, got 0` |

The window is a value of the **model** (D-079), like its attention band: saved
in the graph document while it is on and switchable at any time. The setting
itself never touches the graph; only a step does.

## 3. The step

One step of the ladder does three things to the graph, in this order:

1. **merge** - compression runs, and merges nothing longer than the window
   (§6: the ceiling is the current `size`);
2. **halve** - every node longer than the window is split into two halves
   (§4), until no node is longer than the window;
3. **move** - the window halves; below the floor it goes back to the top.

A step happens **by hand** - `radixnet window --step [N]`, `POST
/api/model/window/step {"steps": N}`, the Step button - or **automatically**
at the end of every training epoch while `auto` is set, in every kind's loop:
`RadixNet.train`, `CountRewardNet._passes`, `NegativeNet`'s blame passes (a
clearing pass is not a training pass) and `ResonantNet`'s passes, feedback
passes (`reward`, `punish`, `two_nrl`) included, since an epoch is an epoch.
Every loop but the phase model's has just compressed the graph when the step
comes, so there the step is the halving and the move; the phase model
compresses once before its passes, so its automatic step merges as well.

What a step did is reported as `{steps, sizes (the size each step applied),
from, to, merges, splits, nodes_before, nodes_after, edges_before, edges_after,
window}`, and an epoch whose loop stepped carries `"splits"` and `"window"` (the
size applied) in its record, right after `"merges"`; an epoch without a step
carries neither, so a record written with the window off is what it was.

## 4. The halving rule

A node's label of `L` units holds `G = (L - n) / stride + 1` grams. A node
longer than the window is cut **between its grams**, at its middle: the first
half keeps the first `⌈G / 2⌉` grams (the odd one, when there is one), its id
and its in-edges; the second half is a new node (the next free id) holding the
rest of the grams and the out-edges (`RadixCyclicGraph.split(node, ⌈G / 2⌉ *
stride)`). Both halves carry the node's data for traversal - its state `z`, its
activation parameters `a, b, h, k`, its visit count and reset count (and in the
count model the judged contexts, `q -> P -> c` becoming `q -> A -> B -> c`; in
the phase model its advance, refreshed) - and the edge between them is the
**heavy connection** of §5.

The scan visits node ids in increasing order and halves a node again while it
is still longer than the window; the second halves are appended to the id
range and halved in their turn when the scan reaches them. So a node of 100
units at a window of 32 becomes four nodes of about 25, and after the step no
node is longer than the window - except a node of **one gram**, which cannot be
halved (a gram is atomic: the index maps it to exactly one node) and is left
as it is however small the window says it should be. The ladder in order -
32, 16, 8, 4 - halves each node once per rung; a pass at a size the ladder
skipped halves as often as it takes.

**Where the halves meet.** Consecutive grams overlap by `n - stride` units
(D-006), so the halves share that overlap and the cut is exact only in grams:

| encoding | node | halves | why |
|---|---|---|---|
| `char:2:2` (groups of two) | `ABCD` | `AB`, `CD` | the grams `AB` and `CD` do not overlap |
| `char:4:4` | `ABCDEFGH` | `ABCD`, `EFGH` | likewise |
| `word:2:2` | `a b c d` | `a b`, `c d` | likewise, in words |
| `char:3:1` (the default trigram) | `ABCD` | `ABC`, `BCD` | `ABCD` *is* `ABC -> BCD` merged; the halves share the pivot `BC` |
| `char:3:1` | `abcdefghij` (8 grams) | `abcdef`, `efghij` (4 + 4 grams) | at window 8; at 4 they become `abcd`, `cdef`, `efgh`, `ghij` |

Under the sliding trigram a half is never shorter than three units - `AB` and
`CD` cannot exist in a trigram graph, because neither holds a trigram and the
trigram `BCD` would have to live in two nodes at once - so the ladder's lower
rungs bite less than their number says: a window of 2 leaves every node at one
gram of three. In units, a sliding node of `L` splits into about `L / 2 + 1`
and `L / 2 + 1`; the sum exceeds `L` by the overlap.

## 5. The same data, and the heavy connection

`split` gives the second half a copy of everything the node held (§4), and
gives the bridge `A -> B` a traversal count equal to the node's own visit
count: every walk that entered the node crossed the place the bridge now
stands. That count is the heavy connection in every kind that computes its
weights from counts - the bridge's share of its node's traffic is everything
the node ever saw - and each kind makes it heavy in its own currency:

| kind | the bridge | why it is heavy |
|---|---|---|
| sine (`radix`) | weight `W_HEAVY = 8`, negated while the network is inverted (as every fresh weight is, D-010) | the halves are copies of one node, so the score is `8 · f · f >= 0` whatever the sign of the activation: as heavy as the activation allows, against fresh competitors of at most `1.5 · f · f'`. It stays a learned weight: training moves it like any other |
| count / word | its traversal count, the node's visit count | share `(count + ½) / (count + ½) = 1` alone; a child attaching later starts against every traversal the node ever had |
| phase (`resonant`) | its traversal count, set to *what passed through the node*: the traversals of the out-edges the bridge now stands before, read before the halving | this kind counts its edges and not its nodes, so the node count a split hands over is 0 there; it fires at no phase (a step inside a merged node never fired at one), so it competes on its share alone, as any traversal whose phase is unknown does |
| negative | its traversal count, which is 0 | the negative network counts neither nodes nor traversals: its evidence is blame, and a bridge carries none - it honestly weighs nothing, and a `judge` over the halves finds exactly the blame it found over the node |

The halves being copies is what keeps traversal unchanged at the moment of the
split: the first half has one child, so its probability is 1 and its cost 0,
exactly the deterministic step inside the merged node. The heaviness matters
afterwards - when another child attaches to the first half, or a transition
into the second half's first gram no longer needs a split - which is the point:
the split creates places to branch, and the bridge keeps the old way through
the likeliest one until the data says otherwise.

## 6. Compression under the window

With the window on, `merge_child(p)` refuses a unary chain whose merged label
would be longer than the window: `len(p) + len(c) - overlap > size`. Everything
that fits still merges, so the structure stays a radix tree everywhere the
window allows. `check_invariants(compressed=True)` therefore exempts a chain
the window holds apart, in all three implementations.

That is what makes the ladder a cycle rather than a one-way shredding. Going
down, each rung cuts finer; back at the top, the epoch's compression (or the
step's) merges every chain that is still unary up to 32 again, and the halves
that nothing branched into in between grow together - keeping one side's
activation and rescaling the other's edges, as every merge does (D-007) - while
a half that gained a branch stays a node. Switched **off**, compression is
unbounded again and the same unary halves merge back at the next compression;
the graph is never left in a state the old rules could not have produced.

## 7. The file

```json
"graph": {"format": "radixnet-graph", "format_version": 4,
          "encoding": {"unit": "word", "n": 3, "stride": 1},
          "attention": {"blur": 0.5},
          "dynamic_window": {"top": 32, "floor": 4, "size": 16, "auto": true},
          "seed": 0, ...}
```

Written only while the window is on, right after `attention` - or after
`encoding` or `format_version` when those are absent - which is where Python
and Rust write it, byte for byte (Go writes the same keys; its documents are
compared as parsed JSON). A document without the block is off; a top that is
not a power of two, a floor over the top or a size off the ladder is refused on
load. `stats()` carries `dynamic_window` (the size, `null` while off) for every
kind, and `GET /api/status` with it.

## 8. The surfaces

| | |
|---|---|
| CLI | `window` (show) · `--on` (at the ladder it had, else 32 down to 4, standing at the top) · `--top N` · `--floor N` · `--size N` · `--auto` / `--manual` · `--off` (change, then save; not with `--dry-run`; `--out PATH`) · `--step [N]` (N steps, default 1, then save) · `info` has a `dynamic window` row. The same command in `radixnet-count` and in the Rust `radixnet` (`--step N` there) |
| API | `GET /api/model/window` → `{kind, window: {on, top, floor, size, auto, sizes, next, unit, units, ngram, longer (real nodes a step would halve; null while off), longest, nodes, heavy (the sine model's bridge weight; null where the bridge is heavy by its count), default_top, default_floor}}` · `POST /api/model/window {on, top, floor, size, auto}` → `{kind, window, stats}` (a new top or floor keeps the size on the ladder: above the top it is the top, below the floor the floor; 400 for a size off the ladder) · `POST /api/model/window/step {steps}` → `{kind, step, window, stats}` (400 while off) · `GET /api/model` carries `dynamic_window` |
| frontend | the Dynamic window card on Model settings: on / off, the top, the floor, where it stands and the automatic step, the ladder drawn as its rungs with the current one lit and the next one marked, how many nodes a step would halve and the longest label, and a Step button with what the step did in one line |
| Makefile | `make window` / `window-on WINDOW_TOP=32 WINDOW_FLOOR=4` / `window-step STEPS=4` / `window-off` |

## 9. Tests

| test | pins |
|---|---|
| `tests/test_window.py` | the ladder and what it refuses; `ABCD` into `AB` and `CD` under a grouping encoding and into `ABC` and `BCD` under the trigram; a node halved at its middle gram, again until it fits, never below one gram; the halves carrying the same state, parameters and count; the sine model's heavy weight (and its sign while inverted) keeping the bridge the likeliest way on against a fresh competitor; the counting kinds heavy by their count, the phase model by what passed through; compression stopping at the window and resuming when it is off; a step merging, halving and moving; the top regrowing what stayed unary; the settings; the automatic step on every kind, feedback passes included, and the negative network's blame passes; the file block beside the band; off is the old file to the bit; the CLI; the API |
| `go/radixnet/window_test.go`, `go/server/window_test.go` | the same, in Go |
| `rust/src/window.rs` (unit tests) | the same, in Rust, with the sine model's heavy weight |
| `frontend/test/window.test.mjs` | the frontend's ladder: the same sizes |
| `tests/test_go_parity.py` | Go against Python: the same nodes halved at the same grams, the same bridges with the same counts, the same chains held apart and regrown, by hand and at the end of every epoch; each reads the other's window |
| `tests/test_rust_parity_tools.py` | Rust against Python: the same, on the count, sine and phase kinds, the same model files byte for byte, the same refusals from the CLI and the same answers from the three routes |

## 10. What this does not do

* **It does not change the encoding.** The gram is the atom: `n` and the
  stride are fixed for a graph's life (D-071), the index is keyed by grams,
  and the window only decides how many of them a node may hold. A ladder that
  halved the *grams* would have to re-key the whole index at every rung, and
  could not climb back up once training at a small gram had left node lengths
  that are not multiples of the large one.
* **It does not split below a gram**, and under a sliding encoding the halves
  share the overlap (§4).
* **It does not pin the halves.** They are ordinary nodes: a half that stays
  unary merges back at the top, or the moment the window is switched off.
* **It does not say what the cycle buys.** Nothing here is graded (Q-20).

## 11. Alternatives rejected

See D-087: pinning the halves for good; measuring the window in grams; cutting
at the unit midpoint regardless of the overlap; a heavy weight scaled to the
activation; a reward on the count model's bridge; the window as the encoding's
`n`; stepping on corrections rather than epochs.
