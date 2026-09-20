# The least-punished traversal — the search that follows the blame

**Status** Built. `Traversal.ByLeastPunished` in the Go port
(`go/radixnet/search.go`) and `Traversal::LeastPunished` in the Rust port
(`rust/src/search.rs`), off by default on both, selected with
`--traversal least-punished`. Python has it in neither model yet.

**Answers** the question D-047 and D-045 leave standing on the *search* side:
the negative network can veto an answer after the fact, but the walk that wrote
it was still chosen by what went right. This is the other half — a search that
chooses by what went wrong.

---

## 1. The problem, precisely

Every path this model has ever taken was chosen by one number. An edge's weight
is

```
count_scale · log(1 + count) + global_scale · log R_all + window_scale · log R_recent + reward_scale · reward
```

and the search takes the path where that is largest, softmax and `-log` away
(D-008, D-022). The rewards and the penalties 2NRL and the tutor apply all land
in the last term, **added together**: `reward` is one accumulator, so a step
rewarded five times and punished once carries `+4` and reads exactly like a step
rewarded four times and never punished at all.

That is a deliberate property of the count model and it is right for *likelihood*
— it is the model's estimate of what usually comes next. It is wrong for one
question the system keeps asking and cannot currently express:

> **Which way through has the least gone wrong on it?**

Not *the most likely way*, and not *the way with the best net score*. The way
carrying the least blame. The two differ exactly where it matters — at a step
that is both popular and known to be a mistake, which is what most tutor
corrections are about.

Measured, on the two-sentence corpus of the traversal tests: `"the cat sat on
the mat"` rewarded once at strength 5 and punished once at strength 1, against
`"the cat sat on the log"`, never judged.

| | continues with | cost |
|---|---|---|
| the ordinary search | `mat` — the rewarded step | 0.99 |
| least punished | `log` — the step nothing is held against | 5.24 |

The ordinary search cannot be made to answer the second question by tuning
`reward_scale`: the scale multiplies the *net*, and the net is where the
information was lost.

## 2. What is being specified

One traversal, selectable per call, that changes **what a walk is ranked by**
and nothing else:

> A step carries a **punishment** — what the model has been taught against it,
> counted on its own. A walk is ranked by its **worst** step's punishment first
> and by its summed cost only between walks whose worst step is the same; and at
> every node, a walk may only take the children the model has the least against.

Explicitly **not** specified here: no change to the weight function, the costs,
the counters, the model file or the structure; no new training; no change to
what the default traversal does — a graph on which nothing was ever punished
gives bit-identical answers, at identical cost, under both.

## 3. The algorithm

### 3.1 The punishment of a step

```
punish(prev, e) = reward_scale · max(0, -edge_reward[e])         # the edge's standing penalty
                + path_scale   · log(1 + incorrect(prev, e))     # what walks from prev got wrong here
```

Two terms, both one-sided, and the one-sidedness is the whole design.

**The edge term** is the penalty side of the edge's reward and nothing else. A
rewarded edge is not *less* punished than an edge nothing was ever said about —
it is exactly as unpunished, at zero. The traversal has no way to express
*better than clean*, which is what stops it from collapsing back into the
ordinary cost order.

**The context term** is the failure count of the path context `(prev, e)` — the
step in the company it kept (D-058): how often a walk that arrived from `prev`
and took `e` was part of an output judged wrong. It is counted **against
nothing**. The cost function's path term weighs `correct` against `incorrect`
and is symmetric on purpose; this one is not, because *a step that was wrong
here once is a step that was wrong here*, and no amount of being right
afterwards makes it a step nothing is held against.

That asymmetry is what makes the traversal unbuyable, and it is the property
the tests pin down: punish a walk, then reward it fifty times over, and the
edge's own penalty is indeed netted away — while the step stays punished.

`log1p` rather than the count itself so the term sits on the scale of the costs
it is ordered against, and so a step that failed a hundred times is worse than
one that failed twice without being fifty times worse.

A step whose caller is unknown — the first step out of the node a prefix landed
on, where the walk has no `prev` — carries the edge term only. This is the same
blind spot the cost function's path term has, for the same reason, and it means
a prediction's *first* step is never blamed by context. Recorded, not fixed.

### 3.2 Ranking a walk: the worst step, not the sum

A path's punishment is the **maximum** over its steps, not the sum.

Summing would say that ten steps punished at 0.1 are as bad as one step punished
at 1.0, and they are not: the model has one recorded failure in the second case
and ten in the first. It would also make a long clean path worse than a short
dirty one, which inverts the question being asked. The maximum is the same
reading the negative filter's `peak` already takes — *one corrected word vetoes
an otherwise clean sentence* (D-047) — so the two agree about what a path's blame
is.

Both numbers a walk carries are therefore monotone along it: the cost is a sum of
non-negative steps and the punishment is a running maximum. That is what keeps
the beam's early exit valid (§3.4).

### 3.3 The order

Lexicographic, `(punishment, cost)`:

```
a before b  ⟺  a.punish < b.punish  or  (a.punish == b.punish and a.cost < b.cost)
```

Under the default traversal the first component is 0 on every path, so the
comparison *is* the cost order — which is why the change is provably inert when
nothing is punished, rather than merely observed to be.

Two punishments within `PunishTolerance` (`1e-12`) count as equal. The same
penalty applied in a different order can land a bit or two apart, and a walk is
not "more punished" for that.

### 3.4 Where it applies

| | under `reward` | under `least-punished` |
|---|---|---|
| the children a node offers | `onward(costs)` — everything but a hand-over to `BACK` | `onward`, then **only the least punished of what is left** |
| a partial path in the beam | ordered by `(cost, chars, node, entry)` | `(punish, cost, chars, node, entry)` |
| a finished path | kept if cheaper than the k-th kept | kept if less punished, or as punished and cheaper |
| the bottom beam | the k **dearest** paths | the k **most punished** paths |
| the early exit | stop when the best partial is no cheaper than the k-th finished | same, on the lexicographic order |
| a sampled walk | softmax over `-cost / temperature` | the same softmax, over the least punished children alone |

The local filter and the global order do different work and both are needed. The
filter is what makes a walk *refuse* a blamed step while standing at the node; the
order is what makes the beam prefer a branch that never had to take one.

## 4. API

```go
// Go — go/radixnet
type Traversal int
const (ByReward Traversal = iota; ByLeastPunished)
func ParseTraversal(name string) (Traversal, error)   // "" / "reward" | "least-punished"
func LeastPunished(costs []ChildCost) []ChildCost     // the filter, exported for the same reason Onward is

func (g *Graph) EdgePunishment(e int) float64         // the edge term
func (g *Graph) StepPunishment(prev, e int) float64   // both terms
func (g *Graph) PathIncorrect(prev, edge int) int64   // the failures of one context, counted against nothing

PredictOptions{..., Traversal: "least-punished"}
GenerateOptions{..., Traversal: "least-punished"}
BeamOptions{..., Traversal: ByLeastPunished}
(*Graph).SampleWalkBy(..., traversal Traversal)
```

```rust
// Rust — rust/src
pub enum Traversal { Reward, LeastPunished }
pub fn parse_traversal(name: &str) -> Result<Traversal, String>
pub fn least_punished(costs: &mut Vec<ChildCost>)

impl Graph {
    pub fn edge_punishment(&self, e: usize) -> f64;
    pub fn step_punishment(&self, prev: Option<usize>, e: usize) -> f64;
    pub fn path_incorrect(&self, prev: usize, edge: usize) -> i64;
}
PredictOptions { traversal: Traversal::LeastPunished, .. }
```

A walk reports what it walked past: `PathResult.Punish` / `PathResult.punish` is
the worst step on it, and a `Prediction` names the traversal that wrote it when
it was not the ordinary one. Both fields are omitted from JSON when zero or
empty, so nothing that reads a prediction today sees a new field.

The CLI: `radixnet-count predict|generate --traversal least-punished`, and
`radixnet-count bench --traversal least-punished --punish-every N`.

## 5. The model file

Nothing. The traversal reads numbers the file already carries — `edge_reward`
and the path contexts — so a model trained by any of the three implementations
can be walked either way, and a model walked this way is not changed by it.

## 6. Parity

Go and Rust implement the same thing and are checked against each other by
`bench/compare.py`, which fails before reporting any timing if the two disagree
on the graph, the transitions, the loss, the expansions or the prediction.

Python does not have it. That is a gap of the *undone* kind, not the deliberate
kind (D-066's distinction): the count model's `search.py` would take the same
change, and until it does, a `--traversal` flag must not appear on the Python
CLI claiming to do nothing.

## 7. Tests

| test | what it pins |
|---|---|
| `TestLeastPunishedIsTheOldSearchUntilSomethingIsPunished` / `with_nothing_punished_the_two_traversals_agree` | the same text, the same cost, the same expansions on an unpunished graph |
| `TestLeastPunishedLeavesAJudgedStep` / `the_least_punished_walk_leaves_a_step_that_was_judged_wrong` | the blamed step is left even though it is five times rewarded and an order of magnitude cheaper |
| `TestPunishmentIsNotBoughtOff` / `punishment_is_not_bought_off_by_a_reward` | 50 units of reward on a punished step, and the step is still punished |
| `TestLeastPunishedFilter` | the filter keeps the minimum, the tolerance, an infinite punishment |
| `TestParseTraversal` / `the_traversal_names_round_trip` | the names the CLI and the API accept |

## 8. What this does not do

* **It does not learn anything.** Nothing about the traversal changes a counter,
  a reward or the structure. It is a way of reading the model, not of teaching it.
* **It does not replace the guard.** The negative network judges a *finished*
  text against a model of failure (D-045); this refuses a step *while walking*,
  out of the positive model's own record. They compose: the pair still vetoes
  what the least-punished walk comes back with.
* **It does not blame the first step of a prediction** (§3.1).
* **It is not a quality claim.** The benchmark measures that the two searches
  disagree on about a fifth of continuations and that the least-punished one
  expands an order of magnitude fewer nodes. Which answers are *better* is a
  question for the tutor, and nothing here has been graded.

## 9. Alternatives rejected

**Keep two accumulators per edge, `reward` and `penalty`.** The honest fix for
§1, and a model file change: every reader, writer and parity test in three
implementations, for a number the path contexts already carry. Rejected for now
— but it is what §3.1's first term is a stand-in for, and if the traversal earns
its place the edge should learn to remember its own failures.

**Punish by `max(0, -path_term)`.** The first thing tried: the mirror of the cost
function's `log((correct + s) / (incorrect + s))`, floored at zero. It is a net
again, so a reward buys the blame off, and the traversal degenerates into a
clipped copy of the cost order — as it did, visibly, in the first run of the
demonstration in §1.

**Sum the punishments along a path** rather than taking the worst (§3.2).

**Make it the default.** The model's own estimate of what comes next is the
likelihood, and that is what `predict` should mean without being asked. This
traversal answers a different question, and a question nobody asked should not
be answered by default.
