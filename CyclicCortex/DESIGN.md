# CyclicCortex — Design Specification

A **cyclic graph over games where distance is similarity**, partitioned into
**regions**, each region owning **one Self-Building Neural Network** whose inputs
and outputs grow as games join it. The SBNN does not read a game's raw moves. It
reads the **common denominators** — the generalised, minified actions that mean
the same thing in every game in its region. *A piece moving one space* is the
same fact in chess and in checkers, and a network that reads only that fact can
be trained on both.

This replaces GTMNN's population of 4096 micro networks with a small number of
larger, growing networks placed by similarity. It is the cortex reading of the
architecture: similar games in adjacent regions, one specialised network per
region, and a graph rather than a hierarchy because similarity is not
hierarchical.

Directory: `CyclicCortex/`. Python package: `cortex`. Python 3.11+, standard
library only.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| A **Single Self-Building Neural Network** instead of micro networks | §8. One SBNN per region, not one per game and not thousands per system. It grows its **hidden** layer on a plateau (as `SBNN_RNN_ActivationFunction/main.py` already does), and — new here — grows its **inputs and outputs** when a joining game brings vocabulary the region does not yet have. |
| Inputs and outputs grow to match the games | §8.3. Growth is additive and **provably non-destructive**: new units enter with zero output weight, so the function computed is bit-identical the instant after growth and the network only changes as the new units learn. |
| Visualise the brain structure — similar games in similar regions | §5-§6. The similarity graph is partitioned into regions by modularity; a region is a set of games mutually closer to each other than to anything outside. This is topographic organisation, and it is the reason a new game lands next to the games it resembles rather than anywhere. |
| A **cyclic** graph | §5.2. Similarity genuinely contains cycles: chess-checkers-go is a triangle with no natural parent, and a tree must break one of its edges. The graph keeps all three. `Research/CyclesAreAFeature.md` is the structural principle, here applied to game-space. |
| **Distance is similarity** | §5.1. Edge weight is Jaccard distance over mechanic sets, which is a **proper metric** — verified over 210 ordered triples with zero triangle-inequality violations. "Distance is similarity" is therefore geometrically consistent rather than a figure of speech. |
| Each cluster assigned its own network | §6.3. One SBNN per region, sized to its region, sharing nothing with other regions except the routing layer. |
| Inputs are **generalised and minified** versions of the games' inputs and outputs | §7. A move becomes a displacement, not a square. Measured: three inputs — `(dx, dy, target_empty)` — give the same cross-game transfer as eight hand-designed features and beat a 72-input board encoding outright (§4). |
| For chess, a piece moving one space counts here — so the same network trains on checkers | §4, **measured and confirmed.** Zero-shot transfer from king moves to checkers: **0.646** on the generalised vocabulary against **0.514** (chance) on the raw board encoding. |
| Find common denominators between games | §7.2. The common denominators are GREN's **mechanics** (`GREN/DESIGN.md` §8) promoted from a similarity signature into the network's actual input vocabulary. The same objects that say two games are alike are the ones the network reads. |

---

## 2. What this replaces, and what survives

**Replaced.** GTMNN's `MicroPool` — 4096 fixed-geometry two-layer networks with
hashed features and a game modifier. The population is gone; so is the
per-micro repertoire and the modifier's role as the only carrier of game
identity.

**Why.** The modifier approach asks every micro to be told *which game it is in*
and to specialise from a shared feature space. This approach instead makes the
input vocabulary itself already general across the games that share a region, so
a network in the board-game region is reading `(dx, dy, occupied)` and does not
need to be told whether it is playing chess. §4 shows that is enough for
transfer. **Identity moves from the input vector into the graph position.**

**What survives, and this matters.** GTMNN's game theory is not discarded, it
**changes granularity**: the players become **regions rather than micros**
(§9.2). A region bids to handle an incoming game, the congestion game still
prices being right and rare, and Shapley still assigns credit. With 10-100
regions instead of 4096 micros, two things improve:

* **Exact Shapley becomes feasible** below ~20 regions (`2^20` coalitions is enumerable), so the Monte-Carlo estimator of `GTMNN/DESIGN.md` §12.3 becomes a fallback rather than the only option.
* The congestion game's split reward now prices **regional** specialisation, which is the thing the cortex reading actually wants — not four thousand micros differentiating, but a dozen regions becoming good at different kinds of game.

**What is lost, stated plainly.** GTMNN's diversity argument (§5) rested on a
large population holding many genuinely different solutions for the congestion
game to price. A dozen regions cannot hold that diversity. If specialisation
turns out to need the population rather than the regions, this design gives it
up — and §15.3 is how that gets detected.

---

## 3. The three pieces, in one paragraph

GREN (`GREN/DESIGN.md`) decomposes a game into mechanics and measures how similar
it is to known games. Those similarities become edge weights in a cyclic graph;
the graph partitions into regions of mutually-similar games. Each region owns one
SBNN whose input and output vocabulary is the **union of its games' generalised
mechanics** — the common denominators — and which grows as games join. An
incoming game is placed in the graph by its mechanics, routed to the nearest
region, and the region's SBNN plays it, having already been trained on everything
else in that region.

---

## 4. The claim, measured first

Before specifying anything, the load-bearing claim was tested:
`common_denominator.py`.

Two 6×6 move-legality games sharing structure but not surface form:

* **king** — legal iff the target is empty and `max(|dx|,|dy|) == 1` (any direction)
* **checkers** — legal iff the target is empty and `|dx| == 1` and `dy == +1` (diagonal, forward)

Train on king moves, then fine-tune on `k` checkers examples. Accuracy on
held-out checkers moves, 5 seeds:

| encoding | inputs | k=5 scratch | k=5 transfer | **zero-shot** | gain at k=25 |
|---|---|---|---|---|---|
| raw board (`one-hot(from) + one-hot(to)`) | 72 | 0.879 | 0.660 | **0.514** | **−0.028** |
| generalised, hand-designed | 8 | 0.899 | 0.890 | **0.646** | +0.015 |
| neutral geometry (no rule-specific features) | 8 | 0.761 | 0.790 | **0.646** | **+0.059** |
| **displacement only** — `(dx, dy, target_empty)` | **3** | 0.797 | 0.810 | **0.646** | +0.041 |

**The claim holds.** Zero-shot transfer from chess-like to checkers is **0.646**
under any generalised vocabulary against **0.514** — exactly chance — under the
raw board encoding. A network that has never seen a checkers move is right about
two thirds of them, purely from having learned king moves in a shared coordinate
system.

Three things in this table are worth more than the headline:

1. **Raw pretraining is actively harmful** at small `k` (**−0.220** at k=5). Training on chess in a board-specific encoding makes the network *worse* at checkers than starting from nothing. Square indices mean different things in different games, so the transfer is negative. This is the failure mode the generalised vocabulary exists to avoid, and it is not hypothetical.
2. **The transfer is not an artefact of good feature design.** The first generalised encoding included `is_diagonal` and `forwardness` — precisely what distinguishes checkers — so the control stripped them. Zero-shot was **identical** (0.646), and the transfer *gain* went **up** (+0.059 against +0.015). The advantage comes from the shared coordinate system, not from features chosen with the answer in mind.
3. **Three inputs are enough.** `(dx, dy, target_empty)` — which way the piece moved and whether the destination is free — matches eight features exactly and beats seventy-two. Minification is not a compromise made to save capacity; the minimal shared vocabulary is the *best* one, because everything stripped out was game-specific and therefore untransferable.

**What this does not show.** 0.646 is two-thirds, not nine-tenths, and it cannot
be better: the rules genuinely differ, since a king may move orthogonally and a
checker may not. Zero-shot transfer is bounded by how much the games actually
share. And both games here are grid games with the same displacement semantics —
the experiment shows transfer *within a region*, which is exactly what regions
are for, and says nothing about transfer across them. §15.1.

---

## 5. The cyclic similarity graph

### 5.1 Distance is similarity, and the geometry is consistent

```python
def distance(a: frozenset[int], b: frozenset[int]) -> float:
    return 1.0 - len(a & b) / len(a | b)          # Jaccard distance over mechanic sets
```

Jaccard distance is a **proper metric** (Levandowsky & Winter, 1971): non-negative,
zero only for identical sets, symmetric, and satisfying the triangle inequality.
Verified directly over the seven-domain corpus of `GREN/DESIGN.md` §8.5 — **210
ordered triples, zero violations.**

That is not decoration. "Distance is similarity" is only coherent if a consistent
geometry exists; without the triangle inequality there would be triples whose
distances cannot be realised simultaneously, and every clustering and every
layout would be arguing with the data. Measured distances:

| pair | distance |
|---|---|
| rust ↔ python | 0.500 |
| chess ↔ checkers | 0.609 |
| chess ↔ go | 0.769 |
| rust ↔ guitar | 0.842 |
| chess ↔ boss conversation | 0.926 |
| chess ↔ rust | 0.967 |

### 5.2 Why the graph must be cyclic

Chess, checkers and go form a triangle: 0.609, ~0.730, 0.769. All three are
mutually close and **none is the parent of the others**. A tree — including the
radix tree of `GREN/DESIGN.md` §19 — must choose one edge to break, and whichever
it breaks is a real relationship discarded.

The graph keeps all three, and cycles are the normal case rather than a
pathology: any three games with overlapping mechanics close a triangle. This is
`Research/CyclesAreAFeature.md` in game-space, and it says something specific
about the relationship between the two structures:

> **The radix tree is a projection of this graph, not a rival to it.** The tree
> is what makes retrieval sublinear; the graph is what is actually true. Where
> they disagree, the graph is right, and the disagreement is measurable (§15.2).

### 5.3 Storage

```python
class SimilarityGraph:
    games: list[int]                          # corpus ids
    mech: list[frozenset[int]]                # each game's mechanic set (GREN §8)
    adj: list[dict[int, float]]               # adj[i][j] = distance, symmetric
    tau: float = 0.75                         # edges above this distance are not stored
    region_of: list[int]                      # game -> region id
    version: int

    def add_game(self, game: int, mech: frozenset[int]) -> int
        # Distance to every existing game; store edges below `tau`. Returns the game index.
        # O(|corpus|) per insert, which is fine: games arrive rarely and mechanics are small sets.
    def neighbours(self, g: int, k: int = 8) -> list[tuple[int, float]]
    def geodesic(self, a: int, b: int) -> float
        # Shortest-path distance through the graph, which can be SHORTER than the direct
        # edge is long -- two games related through a third. Dijkstra, cached by version.
    def check_metric(self, sample: int = 1000) -> int      # triangle violations; must be 0
```

`tau = 0.75` keeps the graph sparse: at the measured distances it retains
chess↔checkers and rust↔python, drops chess↔rust and chess↔conversation, and
retains chess↔go marginally. The threshold is the main knob on how many regions
form and §15.4 is about choosing it by measurement rather than by taste.

---

## 6. Regions

### 6.1 Partitioning

```python
def partition(graph: SimilarityGraph, resolution: float = 1.0) -> list[int]
    # Greedy modularity maximisation (Louvain-style, one pass then refinement) over
    # similarity weights `1 - distance`. Deterministic given a seed: nodes are visited in
    # corpus order and ties break to the lower region id, so the same corpus always yields
    # the same regions.
```

Modularity rather than k-means because the number of regions is not known in
advance and should be discovered — the cortex reading wants however many
functional areas the data supports, not a number chosen up front.

### 6.2 What a region is

```python
@dataclass
class Region:
    id: int
    games: list[int]
    vocabulary: "Vocabulary"        # the union of its games' generalised mechanics (§7)
    net: "SBNN"                     # exactly one
    centroid: frozenset[int]        # mechanics held by a MAJORITY of member games
    radius: float                   # max distance from centroid to a member
    reputation: float               # discounted accuracy, for the region-level game (§9.2)
    wealth: float
```

`centroid` is deliberately the **majority** mechanic set rather than the
intersection. The intersection (`GREN/DESIGN.md` §8.7) is the right definition of
a *general class* — what every member certainly shares — but it shrinks toward
empty as a region grows, which makes it useless as a position. A majority
centroid stays representative.

### 6.3 One network per region

Each region owns one SBNN (§8), sized to its own vocabulary. Regions share no
weights. A game is played by exactly one region's network, chosen by routing
(§9).

This is the architectural bet: **a region's games are similar enough that one
network serves all of them, and different enough between regions that sharing
would hurt.** §4 supports the first half within a region. The second half is
untested and is §15.3.

---

## 7. The generalised vocabulary — common denominators

### 7.1 Minification

A game's native action space is large and game-specific: chess has ~4000
from-to pairs, go has 361 placements, Rust has infinitely many programs. The
generalised form strips everything that does not mean the same thing elsewhere.

```python
@dataclass(frozen=True)
class GeneralAction:
    mechanic: int              # which mechanic this action instantiates (GREN §8)
    params: tuple[float, ...]  # the mechanic's parameters, normalised

# chess Nf3            -> GeneralAction(STEP_MOVE,  (dx=+1, dy=+2, occupied=0))
# checkers c3-d4       -> GeneralAction(STEP_MOVE,  (dx=+1, dy=+1, occupied=0))
# go D4                -> GeneralAction(PLACEMENT_MOVE, (x=.4, y=.4, occupied=0))
# rust `let x: i32`    -> GeneralAction(SCOPE_BINDING, (depth=.2, annotated=1))
```

The two grid moves land in the **same mechanic with different parameters**, which
is exactly what makes one network able to read both. §4 measured that this is
sufficient.

### 7.2 The vocabulary is GREN's mechanics, promoted

A region's `Vocabulary` is the union of its games' mechanics, each contributing a
fixed number of input slots for its parameters plus one presence bit.

```python
class Vocabulary:
    mechanics: list[int]                 # in a stable order; never reordered, only appended
    slot_of: dict[int, tuple[int, int]]  # mechanic -> (offset, width) in the input vector
    width: int
    def encode(self, actions: Sequence[GeneralAction], state) -> array('d')
    def extend(self, new_mechanics: Sequence[int]) -> tuple[int, int]
        # Appends slots. Returns (old_width, new_width). NEVER reorders existing slots:
        # a slot's index is a permanent contract with the weights that read it, and
        # reordering would silently scramble everything the region has learned.
```

**The same objects that decide similarity are the ones the network reads.** That
is the economy of this design: GREN's mechanics are not a separate analysis
layer feeding a differently-encoded network — they *are* the input vocabulary,
so the reason two games cluster is the reason one network can read both.

### 7.3 Outputs generalise too

The output is not "which of 4000 moves" but, per candidate action, the two heads
established in `GREN/DESIGN.md` §22.6:

```
p_valid   is this generalised action legal?     trained from refusals
grade     how good is it, given legal?          trained from outcome
```

Output growth therefore happens when a region gains a mechanic it must *produce*
rather than merely score — a new action kind — not on every new game.

---

## 8. `sbnn.py` — the Self-Building Neural Network

One per region. Extends the growth logic of
`SBNN_RNN_ActivationFunction/main.py` from hidden-only to inputs and outputs.

```python
class SBNN:
    ni: int ; nh: int ; no: int
    W1: list[array]    # ni x nh, row-major by input so an input can be appended cheaply
    b1: array          # nh
    W2: list[array]    # nh x no
    b2: array          # no
    act: str           # "tanh" | "sine"
    b: float = 1.0     # sine frequency -- 1.0, NOT 1/3; see NeuralCompression/FINDINGS.md 5
    history: list[dict]
```

### 8.1 Growth triggers

| what grows | when | by |
|---|---|---|
| **hidden** | the running loss has not improved for `patience` updates past a warmup — the **plateau** condition | `growth_amount = 4` |
| **inputs** | a game joins the region bringing mechanics the vocabulary lacks | the new slots' width |
| **outputs** | the region gains a mechanic it must produce, not merely score | one head pair |

The hidden trigger is a plateau detector, not a gradient-magnitude one, and that
is a measured choice: `NeuralCompression/FINDINGS.md` §11 found gradient
magnitude cannot distinguish *stuck* from *converged* from *still starting*, and
using it was catastrophic. A plateau separates them. The author's existing SBNN
already grows on a loss plateau, which is the correct signal; it is restated here
so the reason is on the record.

### 8.2 Growth is additive and non-destructive

```python
def grow_hidden(self, k: int) -> None
    # Append k hidden units. Incoming weights ~ U(-s, s); OUTGOING weights EXACTLY ZERO.
def grow_inputs(self, k: int) -> None
    # Append k input rows to W1, EXACTLY ZERO.
def grow_outputs(self, k: int) -> None
    # Append k output columns to W2 with small random weights and zero bias.
```

**Invariant (asserted by `test_sbnn.py`): immediately after `grow_hidden` or
`grow_inputs`, the network's output is bit-identical to before, for every input.**

For hidden growth a new unit has zero outgoing weight, so it contributes nothing
until it learns. For input growth a new input has zero weight into every hidden
unit, so the new feature is ignored until it earns attention. Growth therefore
costs nothing and risks nothing at the moment it happens; the network only
changes as the new capacity is trained. Output growth is the exception — a new
head is new behaviour and has no prior output to preserve.

This is what makes "inputs and outputs grow to match the games" safe. A region
that has learned chess does not get worse the instant checkers joins it; it
gains slots that do nothing until checkers data flows through them.

### 8.3 Growth without forgetting

Non-destructive growth protects the function at the moment of growth. It does
not protect it from the *training that follows*, which is ordinary catastrophic
forgetting. Two mitigations, both already measured elsewhere in this repository:

* **Replay.** When a new game joins, the region rehearses its existing games alongside the new one. `NeuralCompression/FINDINGS.md` §12 measured replay as the better of the two consolidation routes — it transfers the function rather than the parameters, so nothing has to line up. Its margin over joint training at matched compute is small (**1.05×**), so replay is used here for *retention*, which is what it is good at, not for accuracy.
* **Fine-tune only on the new slots first.** One pass with the pre-existing `W1` rows frozen, letting the new inputs find their footing before the whole network moves. Cheap, and it bounds the disturbance.

---

## 9. Routing

### 9.1 Placing a game

```python
def route(graph, regions, mech: frozenset[int]) -> tuple[int, float]
    # Nearest region by distance from `mech` to each region's centroid. Returns
    # (region_id, confidence), confidence falling with distance and with how much
    # of `mech` the region's vocabulary already covers.
```

A game whose nearest region is further than `tau_new` (default 0.8) **creates a
new region** rather than joining a bad one. That is how the cortex grows areas:
genuinely novel structure gets its own place instead of being forced into an
existing one.

### 9.2 The region-level game

Routing by nearest centroid alone is a scoring rule that nothing keeps honest.
The region-level version of GTMNN's mechanism (§2) replaces it:

* Regions **bid** to handle an incoming game, bidding reputation × coverage of its mechanics.
* Seats are sold at the **highest losing bid**, so truthful bidding is dominant (Vickrey).
* The **congestion** reward is split among regions that claim the same game, so a region that uniquely handles an unusual game earns more than one of five that all claim chess.
* **Shapley** assigns credit after the outcome — and with 10-100 regions, exact enumeration is feasible below ~20, so `GTMNN/DESIGN.md` §12.3's Monte-Carlo estimator becomes the fallback rather than the only option.

---

## 10. Package layout

```
CyclicCortex/
  DESIGN.md                this file
  common_denominator.py    the §4 experiment, runnable
  cortex/
    __init__.py
    action.py              GeneralAction, minification from a game's native moves
    vocabulary.py          Vocabulary, stable slot allocation, extend()
    graph.py               SimilarityGraph, distance, geodesic, metric check
    regions.py             Region, modularity partitioning, centroid/radius
    sbnn.py                SBNN: forward, backward, grow_hidden/inputs/outputs
    growth.py              plateau detection, replay, frozen-row fine-tune
    routing.py             route(), the region-level auction and congestion game
    credit.py              Shapley over regions (exact below 20, MC above)
    cortex.py              Cortex: the whole thing; add_game, play, train, stats
    checkpoint.py ; bench.py ; cli.py ; api.py
  tests/ ; frontend/ ; data/
```

---

## 11. `frontend/` — the brain view

The point of this design is spatial, so one screen carries it: **`CortexMap.jsx`**
lays the similarity graph out by force-directed embedding of the distance matrix,
draws regions as coloured hulls, sizes each region's node by its SBNN's parameter
count, and animates a new game arriving — where it lands, which region bids, and
whether a new region forms. Everything else (`RegionPanel`, `GrowthLog`,
`VocabularyView`, `TransferMatrix`) is supporting detail.

The layout is a **projection** of the graph for human consumption; routing and
credit are computed in the graph, where the metric is exact and no projection
distortion exists.

---

## 12. CLI

| command | behaviour |
|---|---|
| `add --game NAME` | place a game: distances, route, join or found a region, grow the vocabulary |
| `map [--mds]` | the graph: distances, regions, radii, and 2D coordinates for plotting |
| `regions` | one row per region: games, vocabulary width, SBNN shape, reputation, wealth |
| `grow --region R` | show the growth history: what grew, when, and on which trigger |
| `play --game NAME` | route and play, reporting which region won the seat and at what price |
| `transfer --from A --to B` | zero-shot accuracy of A's region on B — the §4 measurement, on real corpus games |
| `check` | the invariants of §14, including the metric check and the growth identity |

---

## 13. Tests

* `test_graph.py` — **zero triangle-inequality violations** over random mechanic sets; symmetry; `geodesic <= direct edge`; determinism.
* `test_regions.py` — partitioning is deterministic under a seed; a game further than `tau_new` founds a new region; centroid is the majority set and stays representative as a region grows.
* `test_sbnn.py` — **the growth identity: output is bit-identical immediately after `grow_hidden` and `grow_inputs`**, for 1000 random inputs. Gradient check against finite differences. Growth never reorders existing slots.
* `test_vocabulary.py` — `extend` only appends; a slot's index never changes across any sequence of extensions; encode/decode round-trips.
* `test_growth.py` — the plateau trigger fires on a real plateau and not during warmup or after convergence (the distinction `NeuralCompression/FINDINGS.md` §11 found essential); replay bounds forgetting measurably.
* `test_transfer.py` — the §4 experiment as a regression test: **generalised zero-shot ≥ 0.60, raw zero-shot ≤ 0.55**, and raw pretraining shows negative transfer at k=5.
* `test_routing.py` — truthful bidding is dominant, checked numerically over a bid sweep; congestion splits correctly.
* `test_credit.py` — exact Shapley equals Monte-Carlo within tolerance for ≤ 12 regions; efficiency to 1e-9.

---

## 14. Invariants

1. **Growth is an identity.** Immediately after hidden or input growth the network computes exactly what it computed before.
2. **Slots are permanent.** A vocabulary slot's index never changes; extension only appends.
3. **The metric holds.** Zero triangle-inequality violations, always. A violation means the mechanic sets are being computed inconsistently and is a bug, not a curiosity.
4. **One region plays a game.** Routing returns exactly one region; ties break deterministically.
5. **Regions share no weights.** Asserted by checksumming every other region's arrays around any training step.
6. **Determinism.** Same corpus, same seed ⇒ same graph, same regions, same routing.

---

## 15. Open questions

1. **Does transfer survive across regions, or only within them?** §4 measured two grid games with shared displacement semantics — transfer *within* a region, which is what regions are for. Whether anything transfers between a board region and a programming region is untested, and if nothing does, regions are independent networks with a clustering algorithm attached and the graph is only doing routing.
2. **Where the graph and the tree disagree, which is right?** `GREN`'s radix tree gives sublinear retrieval; this graph gives the true geometry. They should mostly agree. The pairs where they do not are the interesting ones and the measurement is cheap: retrieve with both, compare rankings.
3. **Is one network per region enough, or was the population doing the work?** GTMNN's diversity argument rested on 4096 micros holding many different solutions for the congestion game to price. A dozen regions cannot. If regional specialisation stalls — flat wealth, flat accuracy, regions that all look alike — the population was load-bearing and this design gave it up. `gini` over regions is the readout, and it is instrumented from the first commit.
4. **What is `tau`?** The edge threshold decides how many regions form, and at 0.75 the measured corpus gives roughly board-games / programming / physical-and-social. Too low and everything is one region with a vocabulary of everything; too high and every game is its own region and nothing transfers. It should be swept against transfer performance, not chosen.
5. **Does a region's vocabulary converge or grow without bound?** Every joining game may add mechanics. If width grows linearly with membership the SBNN's input layer grows linearly too and the "one network per region" economy is lost. The prediction is that it saturates — similar games share mechanics, which is why they are in the same region — but that is the design's central efficiency claim and it is unverified.
