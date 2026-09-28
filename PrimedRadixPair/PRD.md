# PrimedRadixPair — Product Requirements

**Status** Proposed, 2026-09-28. No code yet. This document says *what* is
being built and *why*, and what will count as success. `DESIGN.md` is the
contract that says *how*; every requirement below names the section that
realises it.

---

## 1. The requirement, in the author's own words

> I want to build a new model in a new directory. Two Radix Tree's, connected
> at the final nodes. This is done by "Priming" the Radix Tree with every
> potential option (brute-force insertion of every option), then connect the
> 2nd Radix Tree to the first at every single level/node where the nodes are
> equal (not just the final nodes).

Term by term, as this document reads it. §9 lists where a different reading
would change the design, and what it would change.

| Term | Reading |
|---|---|
| **Radix tree** | as everywhere in this repository: a path from the root spells a sequence of units, and a node *is* the sequence spelled so far. `RadixTrieLLM_RNN/main.py` is the smallest example, `RadixCyclicNN/graph.py` the largest. |
| **Priming** | before the tree sees a single text, every possible sequence is inserted. The structure is then complete and **never changes again** — training only ever changes the numbers on nodes and edges. |
| **every potential option** | every sequence of length `1..L` over a closed alphabet of `R` units — letters, sounds or bytes, plus the marks for the start and end of a text and for a unit outside the alphabet. `R + R² + … + R^L` sequences. |
| **brute-force insertion** | the requirement is that every option is *present*. A complete tree needs no pointers — the path to a node is its address in base `R` — so the design realises the insertion by arithmetic and keeps a literal brute-force insertion as the oracle the arithmetic is tested against (`DESIGN.md` §6, §14). |
| **the 2nd Radix Tree** | the same primed sequences read **from the other end**: its path spells the sequence backwards. Its root-ward step therefore drops the *oldest* unit, where the first tree's drops the *newest* — which is what lets the pair slide a window. |
| **nodes are equal** | two nodes spell the same sequence. Path equality, not label equality (§9.2). |
| **connected at every level** | every node of the first tree has a connection — a *rung* — to its equal in the second, at every depth from the root to the final nodes. |
| **connected at the final nodes** | the special case the idea started from. It is shown (`DESIGN.md` §8.5) that a pair connected only there cannot continue past one window, which is the reason the connections at every level are the requirement. |

---

## 2. Why — the hypothesis

Every radix model in this repository so far **grows**. `RadixCyclicNN` creates
a node the first time it sees a gram, splits and merges as it learns, and its
dynamic window halves and regrows what it built; `FilterBankRadix` rebuilds its
trees from their text every round. Growth is where the structural decisions are
made — and every structural decision is a place to be wrong (a merge that hides
a branch point, a split that comes late) and a cost paid on every observation
(a lookup, an allocation, the compression pass).

This model is the other end of the same line: **decide nothing structurally,
ever.** Prime the tree with every option, and an observation is a counter going
up at an address that is computed, not found. Three things are claimed to
follow, and each is a requirement in §6.

1. **Learning is counting.** With the structure fixed and complete, training a
   text is `L` array increments per unit and nothing else — no lookup, no
   allocation, no split, no merge. Measured while writing the design, in pure
   Python: 2.4 million increments per second, which is 600 k units per second
   at `L = 4` (`DESIGN.md` §15).
2. **Every context length is present, so falling back is a walk, not a
   formula.** A grown tree holds the contexts it has seen; a primed one holds
   *every* context of every length up to `L − 1`, so the model can always answer
   from a shorter memory when the longer one is empty. The second tree is what
   makes that a *path*: a step toward its root forgets the oldest unit, the rung
   brings the walk back to the first tree, and the pair together does what
   `FilterBankRadix/DESIGN.md` §5.3 does with a fold — the same numbers, proved
   by identity and checked to `1e-12` (`DESIGN.md` §8.3).
3. **The same sequence, seen from both ends, is one node.** Reading forwards
   and backwards through the same primed nodes gives two conditionals per
   sequence — what follows it and what precedes it — and the rung is where they
   meet. A text can then be scored from either end, and the unit that *both*
   readings find unlikely is where the text is wrong: a localisation of blame
   that the negative network (`RadixCyclicNN/DESIGN.md` §24) currently gets by
   aligning characters.

The cost is stated as plainly. A primed tree is `R^L` big whatever the data, so
the alphabet must be small and the depth shallow (§6.2). That is exactly the
regime of letters and sounds the phonetic work has moved the repository into.

---

## 3. Goals

| | Goal | Realised in |
|---|---|---|
| G-1 | A model that is **primed, not grown**: its node set is a function of `(alphabet, L)` alone, fixed at creation. | `DESIGN.md` §6, §7 |
| G-2 | **Two trees** over those nodes, forwards and backwards, **connected at every equal node**. | §8 |
| G-3 | The **four verbs** of the family — `train / predict / generate / score` — on the same CLI shape as `RadixCyclicNN` and `GTMNN`, so it reads as a sibling. | §10, §13 |
| G-4 | **Scoring from both ends**, and the per-unit cost under each reading and under both. | §8.4 |
| G-5 | A **measurement against the nearest existing models** on the same corpus: bits per unit against `FilterBankRadix`'s single tree and `RadixCyclicNN`'s count model. | §7 of this document |
| G-6 | The **rungs' contribution measured, not assumed**: the pair with its rungs against the same counts without them. | §7, S-5 |

## 4. Non-goals — this version

* **Growth of any kind.** No split, no merge, no dynamic window, no node created after priming. A tree that grows is `RadixCyclicNN`.
* **Words as units.** "Every potential option" over words is unbounded; a word model is `RadixCyclicNN`'s `WordNGramNet`.
* **The meeting walk** — fill-in-the-middle and correction, where the forward walk from a left context and the backward walk from a right context meet on an equal node. The geometry is built for it and it is the first follow-up (§8, phase 3), specified on its own.
* **Back-propagation.** Nothing in this family multiplies a chain of derivatives, and nothing here starts to.
* **Torch, Go, Rust, the HTTP API and the frontend.** Python and the standard library first; ports once the Python numbers are worth porting.

---

## 5. Who uses it, and for what

The author, in three ways, in this order:

1. **As a language model over letters or sounds** — trained on the corpora the
   repository already has (`RadixCyclicNN/data/`, the Ollama-written corpora,
   the phonetic transcriptions), read with the same `units` the family already
   uses, and compared on bits per unit.
2. **As a fast scorer.** Counting needs no learning rate, no epochs and no
   seed; a primed model of a corpus is built in seconds and scores a text from
   both ends. That makes it a candidate judge for the teaching loops — the
   tutor, the critic, the negative network — where a cheap "how surprising is
   this, and where" is worth more than a good generator.
3. **As the ground for the meeting walk** (phase 3): a correction is the
   cheapest fill between what was read before the mistake and what was read
   after it, and equal nodes are where the two readings meet.

---

## 6. Requirements

### 6.1 Functional

| | Requirement | Realised in |
|---|---|---|
| FR-1 | **Priming.** `prime(alphabet, L)` produces a model holding every sequence of length `0..L` over the alphabet (`0` is the root). The node count is `(R^(L+1) − 1) / (R − 1)`, a pure function of `(R, L)`; no operation ever adds or removes a node. | `DESIGN.md` §6, §7.1 |
| FR-2 | **A closed alphabet.** The `R` units are explicit and fixed at priming: a preset (`letters`, `phones`, `bytes`) or a given list. The three marks — `<s>` (the start of a text), `</s>` (its end) and `<unk>` (a unit outside the alphabet) — are symbols of the alphabet, so a text is `<s> u₁ … u_T </s>` and its start and end are learned like anything else. | §5 |
| FR-3 | **Two trees, one node set.** Tree A's edges *append* a unit, tree B's *prepend* one. Every node of A is connected to its equal in B; because the connection holds at every node, the two trees are stored as one node set with two edge families, and the rung is the one place a connection can carry a weight of its own. | §8.1 |
| FR-4 | **Training is counting.** `train(texts)` counts every substring of length `1..L` of every padded text, once per occurrence, into the shared node table. Both trees are trained by the one pass. Deterministic; training a text twice doubles its counts; no other state changes. | §7.2 |
| FR-5 | **Prediction.** From a context of up to `L − 1` units: the next-unit distribution by the exact fold over every depth the walk can fall back to (`mode=greedy`), the cheapest single path through the pair (`mode=dijkstra`, the family's default), and a sampled walk with a temperature (`mode=sample`). Forwards and backwards. | §8.3, §9 |
| FR-6 | **Scoring.** `score(text)` returns bits per unit under the forward reading, under the backward reading, and per unit the cost under each and the *lesser of the two* — the cost both readings agree on. | §8.4 |
| FR-7 | **Fallback as a setting.** `backoff = all` (every rung; the default), `deepest` (one fall from the given context straight to the uniform — a pair joined at one level only) or `none` (no fall: the deepest context and the floor), so G-6 can be measured on the same counts. | §8.5 |
| FR-8 | **Persistence.** A model file holds the alphabet, `L`, the kind, the settings, and **only the counts that are not zero** (for the sine kind: only the parameters that moved from their deterministic initial values). Priming is recomputed on load. A restored model predicts identically, and the file's size is proportional to what was seen, never to `R^L`. | §12 |
| FR-9 | **CLI.** `python3 -m radixpair prime / train / predict / generate / score / info / bench / check`, `--json` on all of them, `make` targets for each. | §13 |
| FR-10 | **The brute-force oracle.** `check` primes a small tree by *literal* brute-force insertion — every option, one at a time, with the standard radix insertion — and asserts that the arithmetic tree is the same tree: the same node count, the same parent and children of every node, ids a bijection onto `0..N−1`. It runs in the test suite at several `(R, L)`. | §6.4, §14 |
| FR-11 | **A second kind on the same nodes** (phase 4): the `sine` kind — the family's per-node sine activation, learnable append, prepend and rung weights, the one-hop rule, `invert`, 2NRL. Same node set, same file format, same CLI. | §11 |

### 6.2 Non-functional

| | Requirement | Note |
|---|---|---|
| NFR-1 | **Standard library only.** Python 3.11+. No optional accelerator in this version. | as `FilterBankRadix` |
| NFR-2 | **Memory is `R^L` and is budgeted up front.** The table below is the whole budget; the Python port is for the rows under roughly 4 million nodes. The sine kind costs 64 bytes per node against the count kind's 8. | `DESIGN.md` §15 |
| NFR-3 | **Speed.** Training ≥ 1 million increments per second in pure Python (measured 2.4 M on the machine the design was written on). Prediction by the fold is `O(L · R)` reads per unit; the cheapest-path search is bounded by `max_expansions` and never raises. | §15 |
| NFR-4 | **Determinism.** Same alphabet, same `L`, same texts (and, for the sine kind, same seed) ⇒ same bytes out, across processes and across a save/load cycle. | §16 |
| NFR-5 | **The file is the data, not the tree.** A model trained on ten sentences is a file of a few kilobytes at any `L`. | §12 |

The budget (`N` = nodes per model; counts at 8 bytes per node; the sine kind at 64):

| alphabet | `R` | `L = 3` | `L = 4` | `L = 5` |
|---|---|---|---|---|
| letters: space, a–z, `'` `.` `,` + 3 marks | 33 | 37,060 nodes · 0.3 MiB | **1,222,981 · 9.3 MiB** (sine 75 MiB) | 40,358,374 · 308 MiB — a port's job |
| phones: 39 + word gap + pause + 3 marks | 44 | 87,165 · 0.7 MiB | 3,835,261 · 29 MiB (sine 234 MiB) | 168,751,485 — no |
| phones with stress | 94 | 839,515 · 6.4 MiB | 78,914,411 · 602 MiB — no | — |
| bytes + 2 marks | 258 | 17,240,335 · 132 MiB — a port's job | — | — |

`L` is the longest sequence held; the longest *context* is `L − 1`, so the
letters row in bold is a 4-gram model with fallback to every shorter order.

---

## 7. Success criteria

Each is a test or a measurement, and a criterion that fails is recorded as
having failed, in `README.md`, the way `Experiments/` does it.

| | Criterion | How it is checked |
|---|---|---|
| S-1 | **The primed tree is the brute-force tree.** | `check` at `(R, L) ∈ {(2,5), (3,4), (5,3), (7,2)}` and larger in `bench`; FR-10. |
| S-2 | **Training throughput** ≥ 1 M increments/s in Python at `letters, L = 4`. | `bench --json`. |
| S-3 | **The rungs are the fold.** The pair's forward distribution equals `FilterBankRadix/DESIGN.md` §5.3 computed on the same counts, to `1e-9`, for every context length, and sums to 1. | `test_pair.py`; the identity is `DESIGN.md` §8.3. |
| S-4 | **Bits per unit** on a held-out split of `RadixCyclicNN/data/sample_corpus.txt` at `letters, L = 4`: reported beside `FilterBankRadix`'s single tree at the same `L` and the same smoothing constants (the expectation is *equality* — it is the same predictor — and a gap is a bug), and beside `RadixCyclicNN`'s count model at its default trigram. | `bench compare`, numbers committed in `README.md`. |
| S-5 | **The rungs buy something.** `backoff = all` beats `deepest`, which beats `none`, in bits per unit on the held-out split. This is the experiment the pair exists to run; the size of each gap is the finding either way. | `bench compare`. |
| S-6 | **Both readings find the mistake.** On held-out sentences with one unit substituted at a random position, the position of the maximum *both-readings* cost (FR-6) is the substituted position — the hit rate is reported at `±0` and `±1`, whatever it is, and beside the hit rate of the forward reading alone. | `bench localise`. |
| S-7 | **Round trip.** Save, load, identical predictions and scores; file size proportional to the number of non-zero counts, checked at two corpus sizes. | `test_model.py`. |
| S-8 | **Backward is forward reversed.** A model's backward score of a text equals the forward score of the reversed text under a model trained on the reversed texts, to `1e-9`. | `test_pair.py`; `DESIGN.md` §16 invariant 5. |

---

## 8. Phases

| Phase | Delivers | Done when |
|---|---|---|
| **P0** | this document and `DESIGN.md` | reviewed; the decisions in §9 made |
| **P1 — the core** | `alphabet.py`, `address.py`, `nodes.py`, `tree.py`, `pair.py`; the count kind; the fold both ways; scoring; `check.py`; the tests of §14 for these | S-1, S-3, S-8 green; the brute-force oracle in the suite |
| **P2 — the verbs** | `search.py`, `model.py`, `checkpoint.py`, `cli.py`, `bench.py`, the `Makefile`; the measurements S-2, S-4, S-5, S-6, S-7 committed in `README.md` | every criterion of §7 has a number |
| **P3 — the meeting walk** | `SPEC-MeetingWalk.md`, then `meet.py`: the cheapest fill between a left and a right context, corrections, and the hook that hands a located mistake to `RadixCyclicNN`'s negative network as a fault | its own spec's criteria |
| **P4 — the sine kind** | `activation.py`, the parameters and their deterministic initialisation, the one-hop rule with the rung as a learned choice, `invert`, 2NRL; the same measurements as P2 for the second kind | the count kind's numbers are matched or the gap is explained |
| **P5 — ports and surfaces** | Go and Rust with bit-identical files, the HTTP API, the Voice / frontend integration — only if P2–P4's numbers justify them | parity tests as in `RadixCyclicNN/tests/` |

---

## 9. Decisions to make

Each has a recommendation; the design is written to the recommendation, and
says what changes if the decision goes the other way.

**9.1 What is an "option"?** *Recommended: every sequence of length `1..L`
over the unit alphabet.* The alternative is every entry of a **lexicon** —
every word of the dictionary, or every pronunciation — inserted into a prefix
tree and, reversed, into a suffix tree. That pair is not complete, so its
addresses are stored rather than computed, radix compression applies again,
and the rungs exist only at sequences that are both a prefix of some word and a
suffix of another (`cat` in *catalog* and *bobcat*). Everything after the
addressing — the walks, the fold, the scoring, the file — is unchanged. It is
the more interesting structure and the less general one; §18 of the design
keeps it as the first open question, and nothing in P1 forecloses it.

**9.2 What makes two nodes equal?** *Recommended: they spell the same
sequence.* The alternative, the same *last unit* regardless of path, would
connect every `t` in one tree to every `t` in the other — `(R^(ℓ−1))²` rungs
per unit per level, and a connection that says "the same symbol in any
context", which is the tree thrown away. Rejected.

**9.3 What does the second tree read?** *Recommended: the same texts,
backwards.* Only then do the two trees hold different information about the
same sequence, and only then does a root-ward step in the second tree do
something the first cannot (drop the oldest unit). The alternative is a second
*forward* tree over other data — the negative corpus, the other speaker of a
conversation — connected to the first at equal sequences. That pair is two
models of the same thing with a bridge between them; it cannot slide its
window, fallback would need edges of its own, and what the rungs would learn
is not obvious. It is the `duo.py` shape of `RadixCyclicNN`, and can be built
on the same node set later without touching the addressing.

**9.4 Two node sets with rungs, or one node set with two edge families?**
*Recommended: one node set.* When every node is connected to its equal and the
connection is free, two trees are one set of nodes with two sets of edges, and
the counts of a sequence are the same seen from either end. The design keeps
the trees' *edges* apart (append and prepend) and shares their *nodes*, and
keeps the rung as the one place a connection carries a weight (the sine kind's
learned "forget"). Two separate node sets would double the memory to store the
same numbers twice.

**9.5 The default alphabet and depth.** *Recommended: `letters`, `L = 4`* —
1,222,981 nodes, 9.3 MiB of counts, a 4-gram with fallback. For sounds,
`phones, L = 4` (3.8 million nodes, 29 MiB). Deeper needs a port (§6.2).

**9.6 The start and end of a text.** *Recommended: symbols of the alphabet.*
`<s>` and `</s>` are units, so the first unit of a text is predicted from a
context that says it is first, as `RadixCyclicNN`'s `START → …` does, and the
sequences that would put a mark in the middle simply stay at zero (about `2/R`
of the table, never touched, never written). The alternative — the end as a
terminal edge on every node, the trie's terminal flag of `Research/Insights.md`
§13 — is asymmetric between the two trees and is not taken.

**9.7 The name.** `PrimedRadixPair/`, package `radixpair`. Descriptive of the
two things that define it and silent on the second tree's role, which is the
one thing still open above.

---

## 10. Risks

| Risk | Where it bites | What is done about it |
|---|---|---|
| **`R^L` is the whole cost.** The alphabet is the budget; one unit too many at `L = 5` is 40 million nodes. | NFR-2 | the table in §6.2, a hard refusal above a configured node ceiling at `prime` time, and the ports as the way past it |
| **The pair is provably a fold.** If S-5 finds no gain over `backoff = none`, the rungs' value rests on backward reading and the meeting walk alone. | S-5, S-6 | S-6 and P3 exist for exactly that reason; the finding is recorded either way |
| **The reading of the requirement.** §9.1 and §9.3 are readings, not facts. | P1 | P1 is written so that a lexicon or a second forward tree changes one module; the decisions are asked for before P1 starts |
| **`<unk>` swallows the text.** A closed alphabet on real text may route too much through `<unk>`. | S-4 | `info` reports the share of `<unk>` units in what was trained; the `letters` preset is widened if it is large |
| **The sine kind in Python.** 64 bytes per node and `L · (R + 1)` sine evaluations per trained unit, per tree. | P4 | the count kind is the one that runs in Python; the sine kind is measured at `L = 3` first |

---

## 11. Open questions

Carried into `DESIGN.md` §18, where they are stated against the mechanism
each one questions. The ones a reader of this document should know exist:
whether priming beats growing on *little* data (the hypothesis in §2, and the
cheapest thing to measure once P2 runs); whether `L = 4` over letters is deep
enough to be worth the memory, or whether the model's place is sounds at
`L = 3`; and what the meeting walk's cost function is.
