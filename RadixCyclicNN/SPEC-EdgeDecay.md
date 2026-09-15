# Edge decay — a node's edges fade on the graph's own clock

**Status** Proposed. Nothing in the code does this yet; this document is the
agreement to build from. When it is built, §3–§6 fold into `DESIGN.md` (a new
§5.1.2 beside the `BACK` sentinel) and §2/§9 become a `DECISIONS.md` entry.

**Answers** the part of **Q-1** that D-068 left open — *how fast a node should
learn to hand over, whether it should ever unlearn, and what a wrongly taught
hand-over costs* — and the consequence D-068 states plainly: *"Nothing decays
them yet, so a node taught in error stays taught until something retrains it."*

---

## 1. The problem, precisely

A voice that catches itself repeating teaches the graph where it goes round
(`observe_back`, D-068). After two or three lessons at a node `p`, the edge
`p -> BACK` is `p`'s cheapest child, and from then on `search.onward` drops any
branch that arrives there: the walk does not go through `p`, it goes round it.

That is the whole point, and it is also a trap. **The ordinary way this model
corrects itself is to walk, be wrong, and be corrected — and a hand-over closes
exactly that door.** Once `BACK` wins at `p`, no search walks `p` again, so no
walk through `p` can ever turn out fine, so nothing can contradict the lesson.
A hand-over taught in error is self-sealing: it is the one thing the model
learns that it has no mechanism to unlearn.

The same is true, less dramatically, of every other opinion a node holds. A
reward from a conversation six months and four corpora ago still counts for
exactly what it counted for on the day.

Measured, on the 60-line sample corpus (`data/sample_corpus.txt`, seed 1):

| | after 1 lesson | 2 | 3 | 4 |
|---|---|---|---|---|
| count model — `BACK` cost vs cheapest sibling | 3.215 vs 1.947 | **1.837 vs 2.079** | 0.870 vs 2.449 | 0.334 vs 3.164 |
| sine model — `BACK` cost vs cheapest sibling | 2.891 vs 2.882 | **2.880 vs 2.883** | 2.868 vs 2.883 | 2.857 vs 2.884 |

Two lessons close a node. Nothing reopens it.

## 2. What is being specified

One operation, and it is the request read literally:

> **A node's out-edges decay** — every edge leaving one node moves a fraction of
> the way back to the state it would be in if nothing had been learned about it —
> **on a clock that runs whether or not anything walks that node.**

That last clause is the whole design. A decay clocked on *walks through the
node* cannot fix the trap in §1, because the trap is precisely that the walks
stop. A decay clocked on *seeing a particular thing* is a trigger, not a decay.
So the clock is **graph time**: how much has happened in this model, anywhere,
since this node's edges were last settled.

Explicitly **not** specified here: no per-edge timers; no wall-clock or
epoch-count time; no decay of the visit **counts** (history is not rewritten —
see §3.3); no change to `search.onward` or to what a rethink teaches.

## 3. The algorithm

### 3.1 The clock

`graph.traversals` — the cyclic counter that already bounds every other counter
in the model (D-064), so it needs no new bookkeeping and wraps correctly by
construction. One epoch of the sample corpus is ~2 900 traversals (count model)
or ~3 500 (sine).

Each node that has something to forget carries a stamp: the exact traversal
total when its edges were last settled. Sparse, like `count_resets` — a
`dict[int, tuple[int, int]]` of `(value, resets)` read back through
`counter.total`, so a model with no hand-overs carries no stamps at all.

### 3.2 The factor

```
elapsed = total(traversals) - total(decayed_at[p])       # graph time since p was settled
factor  = 0.5 ** (elapsed / decay_half_life)             # 1.0 when decay_half_life <= 0
```

One `pow` per node per settle. `decay_half_life` is in traversals and **defaults
to 10 000 — the same number as the count model's sliding window** (D-022), and
deliberately so: the window is how far back this model calls *recent*, so the
default says *an opinion is worth half what it was once the model has read a
whole window of text without renewing it*. That also answers D-022's objection
to exponential decay — *"no crisp interpretation and no natural units"* — by
borrowing the units the window already established.

The property that makes everything else safe:

> **The decay a node has suffered depends only on elapsed graph time — never on
> when the bookkeeping ran.** Settling a node twice, or ten times, or not at
> all until much later, gives the same weights.

So the sweep can run anywhere, can be skipped, and can run at different moments
in Python and in Go without the two models diverging.

### 3.3 What moves

| | what moves | decays toward | untouched |
|---|---|---|---|
| sine graph (`RadixCyclicGraph`) | `edge_w[e]` | **the node's mean out-weight** | node activation params |
| count graph (`CountRewardGraph`) | `edge_reward[e]` | **0** | `edge_count`, `window_edge_count` |
| resonant graph (`ResonantGraph`) | `edge_reward[e]`, `(edge_cx, edge_cy)` | **0**, **no preferred phase** | `edge_cw`, `edge_count` |
| negative graph (`NegativeGraph`) | — out of scope — | | |

*Sine.* `m = sum(w) / deg` over the node's out-edges, computed **before** any of
them move; then `w <- m + (w - m) * factor` for each. The mean is the node's own
*no preference* point: the cost of an edge is a softmax over `w * f(p) * f(c)`,
which reads a node's weights against each other, so the mean is what "this node
has stopped having an opinion" means. It needs no new constant, and — this is
why it beats decaying toward a fixed birth weight — **it is equivariant under
`invert()`**: negating every weight negates the mean, so decay and 2NRL's
inversion commute.

*Count.* The weights are *derived*, so decaying them is erased by the next
`recompute_weights()`. What decays is the input that is an *opinion* rather than
a record: `edge_reward[e] <- edge_reward[e] * factor`, followed by one
`recompute_weights()` at the end of the sweep. The counts are **not** scaled:
they are what the model observed, history is not rewritten, they are cyclic
integer counters that `traversals` is supposed to bound, and scaling them would
make Python/Go parity depend on float-to-int rounding. This is also enough on
its own — the count model's hand-over is carried by the reward
(`observe_back` -> `add_reward`), which is why it wins in ~3 lessons instead of
the ~40 the count alone needed.

*Resonant.* Two things, for the same reason: the reward fades toward 0 as in the
count model, and the circular accumulator `(edge_cx, edge_cy)` is scaled by the
same factor — which is exactly `sharpen(edges, factor)`, an operation the model
already has, and which means *this edge no longer has a phase it likes to fire
at*. The accumulated weight `edge_cw` and the counts are left alone, as the
record of how much was seen; only the direction and strength of the preference
fade. `observe_back` there already counts the hand-over **without a phase**, so
what decays is precisely what the hand-over put in: a reward and a share.

*Negative.* It already has `forget(reason, factor)`, which is manual on purpose
— the tutor can be wrong, and blame is released deliberately. Left alone; named
here so the omission is a decision rather than an oversight.

### 3.4 Scope — whose edges

`decay_scope`, one of:

| scope | what decays | swept set |
|---|---|---|
| `"back"` | only the `p -> BACK` edges | `parents[BACK]` |
| **`"node"`** (default) | **every out-edge of a node that carries a hand-over**, the hand-over included | `parents[BACK]` |
| `"all"` | every out-edge of every node | every alive node |

`"node"` is the default because it is both the literal request and the
defensible one: **a node that has been caught looping is a node whose opinions
are suspect, so all of them fade** — the hand-over, the penalty on the step it
looped through, and the reward on the step it took instead — while a node no
voice has ever had to back out of is left alone entirely.

It is also free. The swept set is exactly `parents[BACK]`, a dict the graph
already keeps, so the default sweep is O(hand-overs × degree), not O(E). And a
real node's degree is bounded by the alphabet: a child's first trigram is the
parent's last trigram shifted one character, so distinct children mean distinct
characters, plus `END` and `BACK`. Measured over the real nodes of the sample
corpus: mean 1.73, max 17.

**No sentinel is ever swept, under any scope.** `START` is the only one with
out-edges, its degree is not bounded by the alphabet (its children are the
corpus's distinct opening trigrams — 18 on the sample corpus, thousands on a
large one), and what it holds is an observation about where texts begin, not an
opinion a node formed.

`"all"` is classic weight decay and **changes what training produces** — at the
default half-life, every epoch of the sample corpus multiplies a node's spread
by `0.5 ** (3000/10000) = 0.81` — so it is opt-in, and turning it on is a
statement about the model, not a tuning detail.

### 3.5 Removal

An edge that has decayed to nothing is dropped — but **only if no observation
supports it**, which today means only `p -> BACK`. An observed transition is a
fact about the corpus; removing it would break `trace`, `node_path` and
`check_invariants`. The rule:

> A hand-over is forgotten outright once it has faded below `DECAY_FLOOR = 0.05`
> of a single lesson: `abs(w - m) < 0.05` (sine) or `abs(reward) < 0.05` (count).

That is ~4.3 half-lives after one lesson never repeated — about 43 000
traversals, ~14 epochs of the sample corpus — and proportionally longer for a
trait taught over and over, which is the right shape: what was learned once is
forgotten soon, what was learned ten times takes ten times the silence.

Removal tombstones the edge exactly as `merge_child` does
(`edge_alive[e] = False`, `_n_alive_edges -= 1`, out of `children[p]` and
`parents[BACK]`), and bumps `structure_version`. Two payoffs:

* the node leaves the swept set, so it stops costing anything — and its other
  edges stop fading, because it is no longer a suspect node;
* a taught node has two children where it had one, so `compress` stops merging
  it. **Dropping the hand-over gives that compression back** — the cost D-068
  said this trait carries is refunded when the trait is unlearned.

### 3.6 Where it is invoked

The same safe points as `carry_counters`, and nowhere else:

* the end of an epoch — `model.py`, `countnet.py`, and their Go twins, beside
  the existing `graph.carry_counters()`;
* `to_dict` — *a saved file always holds a settled reading*, matching the
  existing *a saved file always holds a wrapped reading*;
* `from_dict` — normalise whatever the file carried;
* `observe_back`, for that one node first, so a new lesson is added to an
  up-to-date weight rather than to a stale one;
* an explicit `decay()` from the CLI and the API.

**Never from a query.** `child_costs`, `back_cost`, `onward`, `predict`,
`generate`, `converse` and every search stay pure: predicting must not change
the model, or "a model in the same state says the same thing" stops being true.
Between sweeps the search sees a slightly stale view of a decaying node; that is
the price, and it is bounded by one epoch or one save.

Hot-loop cost, per §15 of `DESIGN.md`: the sweep leaves after one comparison
when there is nothing to do — `decay_half_life <= 0`, or, under the default
scope, `not self.parents[BACK]`.

## 4. API

```python
# radixnet/graph.py
DECAY_HALF_LIFE = 10_000          # traversals; 0 or less turns decay off
DECAY_SCOPE     = "node"          # "back" | "node" | "all"
DECAY_FLOOR     = 0.05            # a twentieth of one lesson

class RadixCyclicGraph:
    decay_half_life: int
    decay_scope: str
    decayed_at: dict[int, tuple[int, int]]      # node -> the traversal reading it was settled at

    def settle(self, p: int) -> float:          # pay one node's debt now; returns the factor applied
    def decay(self, force: bool = False) -> dict:
        """Sweep -> {"nodes", "edges", "dropped", "elapsed", "half_life", "scope"}."""
```

`CountRewardGraph` and `ResonantGraph` override the per-node step (rewards, the
phase accumulator — not weights) and call `recompute_weights()` once at the end;
`configure()` accepts `decay_half_life` and `decay_scope` alongside the existing
scales, and `weight_config()` reports them.

* **CLI** — `radixnet decay --model M [--half-life N] [--scope back|node|all] [--save]`,
  printing *settled N node(s), moved M edge(s), forgot K hand-over(s)*; and
  `--half-life` / `--scope` on `train` and `weights`. `converse --save` already
  writes what a conversation taught, and now also what it forgot.
* **API** — `POST /api/model/decay` returns the same dict; `half_life` and
  `scope` join `POST /api/model/weights` and `GET /api/model/weights`.
* **Frontend** — the two settings on the weights card; the Converse panel's
  rethink line gains *and forgot the hand-over at N node(s)* when a sweep
  dropped one, beside the existing *and learned to hand over there*.

## 5. The model file

**Format 4**: `decay_half_life`, `decay_scope`, and the sparse `decayed_at`
stamps. A format-3 file loads with the defaults and every node's stamp set to
the file's own `traversals` reading — **a model does not decay for the time it
spent on disk**, only for the graph time it has lived through. The upgrade is a
`_with_decay(d)` beside `_with_back(d)`, mirrored by Go's `withDecay`.

## 6. Go parity

`go/radixnet/graph.go` (`DecayHalfLife`, `DecayScope`, `decayedAt`, `Settle`,
`Decay`), the count model's override, `json.go` (format 4 + `withDecay`),
`server/http.go`, `cmd/radixnet-count`. The resonant model and the
metacognitive layer are Python-only, so they carry no parity obligation until
they are ported. Two rules that decide whether parity holds:

* the factor is written as the *identical* expression in both — `0.5 ** (e/h)`
  and `math.Pow(0.5, e/h)` — and the mean is summed in the same explicit
  left-to-right order `shares()` and `recompute_weights()` already use;
* the swept set is iterated in **ascending node id**, not in map order, so the
  floating-point result does not depend on Go's randomised map iteration.

`tests/test_go_parity.py` gains: the same weights after the same sweep
(`places=9`, the tolerance the file already uses), the same edges dropped, and
the same `decay` dict.

## 7. Tests

* the factor depends only on elapsed graph time — settling once, twice or ten
  times over the same interval gives identical weights;
* a hand-over taught twice stops winning after one half-life, and is dropped
  after the floor is crossed;
* a hand-over taught ten times survives a half-life and is dropped later, in
  proportion;
* a dropped hand-over restores the node to a unary chain and `compress()`
  merges it again;
* scope `"back"` leaves the siblings alone, `"node"` moves them, `"all"` moves a
  node that was never taught anything;
* `decay_half_life = 0` is a no-op, and every existing seeded prediction is
  unchanged by the whole feature at its defaults on a model with no hand-overs;
* observed edges are never dropped, whatever the floor;
* a format-3 file loads and does not decay for the time it was on disk;
* prediction does not settle anything: `predict` twice on a decaying model
  returns the same thing and leaves `traversals`, weights and stamps untouched;
* the count model's counts are unchanged by a sweep, its rewards are not; the
  resonant model's `edge_cw` and counts are unchanged, its coherence falls;
* Go parity as in §6.

## 8. What this does not do

* **A model that only converses never forgets.** The clock is graph traversals,
  and a conversation adds about one per rethink. That is the honest position —
  nothing in the graph has moved, so nothing has been contradicted — but it
  means the cure for a wrong hand-over is *more text*, not more talk. Counting a
  spoken walk's steps as traversals would change what the count model's window
  means, so if that is wanted it is a separate decision, not a knob here.
* **It does not make a wrong hand-over cheap today.** Until the sweep has run
  enough times, the branch stays closed; §3.5 sets how long.
* **It does not touch the metacognitive layer** (`metacog.MetaLayer`), whose
  cycle signatures carry ride / escape / abort scores that never fade either.
  Deliberately: those scores are not self-sealing the way a hand-over is — a
  signature the layer scores `abort` on is still consulted every time that cycle
  comes round, and the corpus keeps scoring it — so they are corrected by the
  ordinary route. If they should fade too, it is the same algorithm on a
  different container and a separate decision.
* **It does not decide the rate from evidence.** 10 000 traversals is a
  defensible default with an interpretation, not a measured optimum. What Q-1
  asks for is a *measurement*: how long a hand-over should last before the model
  is worse off keeping it. That stays open, and this gives it a dial to measure
  with.

## 9. Alternatives rejected

* **Decay triggered by observation** — "when a training text walks through `p`,
  fade `p`'s hand-over". Same effect where it fires, but it is a trigger dressed
  as a decay: the amount forgotten depends on how often the corpus happens to
  cross that node, and the node in §1's trap is often the one the corpus crosses
  least. It also ties forgetting to the training loop, so a model that is only
  ever conversed with forgets nothing *and* has no way to say so.
* **Decay clocked on walks through the node** — the natural "use it or lose it"
  rule, and it cannot work here for the reason in §1: the hand-over's whole
  effect is that walks stop arriving.
* **Ticking the decay inside `child_costs` / `onward`** — it would make the
  search itself erode a hand-over every time it bounced off one, which is
  causally perfect and architecturally wrong: prediction would mutate the model,
  two identical `generate` calls would differ, and the cost cache would have to
  be invalidated by reading it.
* **A spoken turn decaying every node on its path** — the symmetric partner of
  `teach_back`, and cheap. Rejected because it does not reach the trapped node
  (it is not on any path, by construction) and it would degrade a well-trained
  model just by talking to it.
* **Decaying toward a fixed birth weight (`1.0`, the mean of the draw range)** —
  simpler, but not equivariant under `invert()`, and it drags a whole node
  toward a constant that means nothing once the node has been trained.
* **Decaying the counts as well as the rewards** — rewrites history, breaks
  `traversals` as the bound on every counter, and makes parity depend on
  float-to-int rounding.
* **A per-node scalar with lazily rescaled weights** (store `w / s_p`, decay
  `s_p` in O(1)) — genuinely O(1) per node, and rejected anyway: `edge_w` would
  stop being the weight, which breaks `to_csr`, `apply_csr_weights`,
  `recompute_weights`, `invert`, both backends, the Go port and every inspector.
  The measured degree bound (mean 1.73) makes the optimisation pointless.
