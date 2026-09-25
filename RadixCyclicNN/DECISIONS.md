# RadixCyclicNN — Design Decisions

The record of **why** this system is built the way it is: the claim behind each
choice, what was rejected, and what the choice costs. It is the third of three
documents and deliberately overlaps neither:

| Document | Answers | Audience |
|---|---|---|
| `README.md` | *How do I run it?* | someone using the thing |
| `DESIGN.md` | *What exactly must the code do?* | someone implementing or changing a module |
| `DECISIONS.md` (this) | *Why is it like that, and what did it cost?* | someone questioning a choice, or reconstructing the reasoning later |

Where a decision carries an argument too long to fit an entry, it is written up
as a paper in `../Research/` (indexed in `../Research/README.md`):
`CyclesAreAFeature.md` for D-001 and D-028, `SineWaveActivationFunction.md` for
D-002 and D-003, and `2NRL.md` for D-009, D-010 and D-045.

A decision belongs here when reversing it would change the character of the
system, not just an implementation detail. Each entry is dated to the commit
that first carried it, so the file doubles as a chronological record of the
design's development.

**Status legend**

* **Accepted** — in force.
* **Research claim** — accepted and implemented, but resting on my own
  hypothesis rather than on external results. Flagged separately because these
  are the load-bearing ones: if a research claim is wrong, the engineering
  around it is wasted.
* **Superseded** — replaced; kept because the reversal is itself information.
* **Provisional** — in force, but a known question is open against it. Listed
  again in Part VIII.

---

## Index

**Part I — Foundations: the research claims** · D-001 cyclic not acyclic · D-002 sine activation ·
D-003 per-node learnable activation · D-004 accept the vanishing gradient · D-005 the edge signal ·
D-006 character trigrams · D-007 radix self-compression · D-008 shortest path · D-009 2NRL ·
D-010 inversion · D-011 perpetual self-upgrade · D-067 breadth is not annealed

**Part II — Implementation platform** · D-012 stdlib only · D-013 flat arrays and CSR ·
D-014 the backend protocol · D-015 JSON model files · D-016 stdlib HTTP server · D-017 one job at a time ·
D-018 React with a committed build · D-019 spec before code

**Part III — The second model** · D-020 two kinds on one graph · D-021 activations pinned to 1 ·
D-022 the dual frequency function · D-023 count-model 2NRL does not invert · D-024 top-K and bottom-K together ·
D-025 generation is prediction

**Part IV — Feedback, failure, self-improvement** · D-026 reward and punish · D-027 blatant failure ·
D-028 local inversion by parity · D-029 Ollama as an optional teacher · D-030 codegen as a grounded reward ·
D-031 the sandbox · D-032 self-conversation

**Part V — Data path and control surface** · D-033 rates as expressions · D-034 archives kept whole ·
D-035 no upload limits · D-036 images as text · D-037 SIGINT

**Part VI — The Go engine** · D-038 port the count model only · D-039 bit-identical interchange ·
D-040 racy counting · D-041 single-writer structure · D-042 lazy weights · D-043 streaming and chunking ·
D-044 one JSON contract, two engines

**Part IX — The negative network** · D-045 a model of failure · D-046 blame only what changed ·
D-047 the guard on every answer · D-048 teaching loops read unfiltered

**Part X — Teaching loops** · D-049 the tutor · D-050 rewards weighted by the mark ·
D-051 difficulty read off the marks · D-052 the auto run · D-053 the critic only reads · D-054 the recall tutor ·
D-055 chat with an LLM

**Part XI — Tool use and the agent** · D-056 a call is text · D-057 the LLM's four roles ·
D-058 blame at the right granularity · D-059 any provider, no stored key · D-060 browser and MCP

**Part XII — Metacognition** · D-061 the stutter · D-062 backing up and exploring · D-063 a record, not a mood ·
D-068 the BACK sentinel: where it goes round, learned

**Part XIII — Counters** · D-064 the odometer

**Part XIV — Memory and the Go gap** · D-065 bounded memory · D-066 what is left, and why

**Part XV — The traversal** · D-069 what a search looks for is an option · D-070 one home for a network setting ·
D-075 the least-punished traversal

**Part XVI — The encoding** · D-071 the encoding is a dial, and it belongs to the model ·
D-073 words as symbols (superseded by it)

**Part XVII — A third implementation** · D-072 the Rust port · D-074 how parity is measured ·
D-076 HTTPS through the system curl · D-077 the rest of Python, by area

**Part XVIII — More ways to search and to train** · D-078 every method off by default, and none draws a random
number · D-079 a setting's home is decided by who keeps it

**Part XIX — The copy editor** · D-082 a correction is a diff, and the LLM is asked for the smallest one ·
D-083 the veto can keep its provenance to itself

**Part VII — Superseded decisions** · **Part VIII — Open questions**

---

# Part I — Foundations: the research claims

These nine decisions are the system. Everything in Parts II–VI exists to serve
them. They come out of my own reverse-engineering of the brain rather than from
the literature, and my stated method is to *refuse current research and reinvent
the wheel* — so they are recorded as claims, with their
evidence and their exposure stated plainly.

### D-001 — The graph is directed and **cyclic**; cycles are a feature

**Status** Research claim · 2026-08-15 (`2c79fdb`), realised 2026-09-09 (`d180176`) · **Layer** theory

**Context** The first formulation of the brain model was a directed *acyclic*
graph. Working through it, I concluded that the brain is a directed
*cyclic* graph, and that a cycle is not a failure of the representation: when
the brain hits a cycle it hands over to metacognition or to another region
rather than looping.

**Decision** Let repeated trigrams create cycles and never break them. A node
may be its own ancestor; self-loops are legal. Handle the cycle at *search*
time, not at *structure* time.

**Alternatives rejected**
* A DAG (the original proposal) — requires either deduplicating repeated
  context or unrolling at build time; both destroy the compression that makes
  the structure worth having.
* Cycle detection and removal during training — throws away the observation
  that produced the cycle.

**Consequences**
* Prediction cannot be a plain graph traversal. Search runs over the
  *depth-unrolled* graph: the state is `(node_id, chars_emitted)`, not
  `node_id`, so a cycle is finite in state space because every traversal emits
  characters. `search.py` and `beam.py` are both written against that state.
* An expansion cap (`max_expansions`, default 200 000) is needed as a
  backstop, and the search must *never raise* — it falls back to the best
  partial path. A pathological graph degrades output quality instead of
  crashing.
* The "hand over to metacognition" half of the claim was, for most of the
  project's life, **not implemented** — search simply kept paying the cost of
  going round. As of 2026-09-14 it exists: a voice that detects it has walked a
  loop backs up to the point the loop started and re-plans from there, under a
  *different* procedure than the one that got stuck — and what it finds out is
  taught to the graph, so the *search* hands over there from then on. See D-061
  to D-063 and D-068.

**Lives in** `Research/CyclesAreAFeature.md`, `radixnet/search.py`,
`radixnet/beam.py`, `go/radixnet/search.go`, `go/radixnet/beam.go`

---

### D-002 — The activation is a sine, not a sigmoid or a ReLU

**Status** Research claim · 2026-08-13 (`9e168a1`) · **Layer** theory

**Context** The brain is taken to be an analog computer, so sine waves are
assumed to be the native encoding of information in it. Looking at a sigmoid
next to a sine, I read the sigmoid as a sine "rotated 45 degrees" —
a truncated special case of the more general periodic function.

**Decision** Use `f(x) = a·sin(b·(x − h)) + k`, defaulting to
`a = −1, b = 1/3, h = 0, k = 0` — that is, exactly `−sin(x/3)`.

**Evidence** My own comparison (`Experiments/ActivationFunctionTest/`) on a
reinforcement-learning control task at 250 episodes: sigmoid 9, ReLU 10,
sine 28–48 reward — a 2.8×–4.8× improvement over the better baseline.

**Alternatives rejected** Sigmoid and ReLU, on the measurement above.

**Consequences**
* The function is **not monotonic and not saturating**. A large `z` does not
  pin the output; it wraps. Saturation-based intuitions (and the tricks that
  depend on them) do not transfer.
* `b` is clamped to `≥ 1e-3` after every update: at `b = 0` the function is
  constant and the node stops carrying information irrecoverably.
* The evidence is one task, one episode count, no variance and no seeds
  reported, and the range 28–48 is itself wide — this being the most
  load-bearing claim in the project on the thinnest measurement. That is now
  addressed where it belongs: `Research/SineWaveActivationFunction.md` §9 sets
  out what the CartPole result does and does not establish, gives a replication
  protocol, and §12 states what would change my mind. See Q-2.

**Lives in** `Research/SineWaveActivationFunction.md`,
`Experiments/ActivationFunctionTest/`, `radixnet/activation.py`

---

### D-003 — Every node owns its four sine parameters, and all four are learnable

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** theory

**Context** If the activation is where learning lives (D-004), it cannot be a
fixed global function.

**Decision** `a, b, h, k` are per-node state, learned with their own rate
(`act_lr`, default `0.005`, an order of magnitude below `lr`). `sine_partials`
returns all six derivatives (`f, df/dx, df/da, df/db, df/dh, df/dk`) in one
call so the hot loop computes `sin` and `cos` once.

**Alternatives rejected**
* One global activation — leaves nothing for the local rule to learn.
* Learning amplitude `a` only — cheaper, but fixes the frequency, and `b` is
  what sets how fast a node's response cycles, i.e. the thing the analog
  hypothesis is actually about.

**Consequences**
* Four extra floats per node, and four extra gradients per transition per
  endpoint. Accepted: it is the point of the architecture.
* The *functional form* stays a sine. Only its shape is learned, never its
  family. See Q-3.
* `act_lr` being 10× smaller than `lr` is a stability guess, not a measured
  optimum — it is what made early training converge. The schedule system
  (D-033) exists partly so this can be varied without editing code.

**Lives in** `radixnet/activation.py`, `radixnet/backend.py`

---

### D-004 — Accept the vanishing gradient; update the activation function instead

**Status** Research claim · 2026-09-09 (`d180176`) · **Layer** theory

**Context** My framing: rather than fight the vanishing gradient with deeper
machinery, accept it, and move the learning into the activation.

**Decision** There is **no back-propagation through depth**. Each observed
transition `p → c` applies a one-hop gradient to exactly: the edge weight
`W[p,c]`, the node states `z_p` and `z_c`, and the four sine parameters of `p`
and of each of `p`'s children. Nothing propagates further.

**Alternatives rejected**
* BPTT / unrolled back-propagation — the graph is cyclic (D-001), so there is
  no finite unrolling; and it is precisely the thing being refused.
* Truncated BPTT at depth *k* — reintroduces a depth hyper-parameter and the
  vanishing problem in miniature.

**Consequences**
* **Vanishing gradients are impossible by construction**, not mitigated. There
  is no depth for a gradient to vanish along.
* Training cost is linear in transitions and independent of graph depth or
  cycle length. This is what makes a million-line corpus tractable at all.
* The loss the gradient is taken on is *local*: `−log softmax(scores_p)[target]`
  over one parent's children. The model never receives a gradient for a
  *sequence-level* error. Long-range coherence has to come from the structure
  (compression, D-007) and from the search (D-008), not from the learning rule.
* Gradients are summed over the batch, divided by batch size, clipped
  element-wise to `±5.0`, then applied once. The clip is load-bearing: `cos`
  terms make individual gradients large when `b` drifts.

**Lives in** `radixnet/backend.py::PythonBackend.step`, `radixnet/backend_torch.py`

---

### D-005 — The edge signal is the product of *both* endpoints' activations

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** theory

**Context** My specification: `N*N`, and "activation of the child × activation
of the parent".

**Decision** `score(p → c) = W[p,c] · f_p(z_p) · f_c(z_c)`. `N*N` is read as the
node-to-node weight matrix — N nodes, N² possible edges — stored sparsely as
per-node adjacency dicts and exported to CSR per epoch. The softmax over a
parent's scores is the next-node distribution; `−log` of it is the search cost.

**Alternatives rejected**
* `W[p,c] · f_p(z_p)` (parent only) — the conventional form; makes a child's
  own state irrelevant to whether it is reached.
* A dense N×N matrix — N reaches hundreds of thousands of nodes; dense is
  impossible without numpy, which is refused (D-012).

**Consequences**
* Both endpoints' parameters appear in every edge's gradient, which is what
  gives the one-hop rule (D-004) something to bite on at both ends.
* The score is **signed**. A negative weight or a negative activation flips an
  edge's contribution. This is what makes inversion (D-010) a one-line
  operation — and what makes the parity argument in D-028 work.
* Because the product is symmetric in its two activation factors, flipping the
  sign of *both* endpoints leaves the edge unchanged. Any inversion scheme has
  to respect that; D-028 is built on it.

**Lives in** `radixnet/activation.py::edge_signal`, `radixnet/graph.py::child_scores`

---

### D-006 — Input is a sliding window of three characters, stride 1

**Status** Research claim · 2026-09-09 (`d180176`) · **Layer** representation

**Context — my reason** Three was not chosen as "the smallest window that leaves
an overlap"; that is a consequence, not the motive. I chose the trigram because
**it reminded me of transformer architecture, in how the transformer pivots**. The structural echo is direct: with stride 1
every character is shared by the window that ends on it and the window that
begins on it, so each character is a *pivot* joining two contexts — and the
two-character overlap that results is what every edge in the graph is keyed on.
The relation a transformer computes with attention, this representation carries
in the shape of the window itself. (The author has confirmed this positional
reading: the shared character *is* the pivot. Three is therefore the smallest
window that gives a pivot plus context on either side of it.)

**Decision** `Encoder.encode("hello") → ["hel", "ell", "llo"]`. Characters, not
bytes and not tokens. Overlap of two characters between consecutive windows.

**Alternatives rejected**
* A learned sub-word tokenizer (BPE and friends) — needs a training corpus
  before training can start, freezes a vocabulary, and is the standard
  approach the project is explicitly diverging from.
* Bytes — would make the trigram overlap meaningless across multi-byte
  codepoints.
* A larger window — raises the node count and weakens compression, and widens
  the overlap past the single shared pivot character the choice is about.
* A window of two — leaves a one-character overlap; the pivot and the context
  collapse into the same character, so there is nothing for a node to carry
  across the join.

**Consequences**
* **No vocabulary and no unknown-token problem.** Any text at all can be
  trained on, including base64 (D-036) and source code (D-030).
* The two-character overlap is a structural invariant, not a convention: for
  every non-sentinel edge `p → c`, `label[p][-2:] == label[c][:2]`. Splits,
  merges and decoding all rely on it, and the test suite asserts it.
* Decoding is trivially exact — first label in full, then `label[2:]` of each
  following one — which is why a path can be turned back into text without
  any learned decoder.
* Text shorter than three characters cannot be represented at all; it is
  skipped and counted (`skipped_short`).
* The Go port indexes by **code point**, not by UTF-16 unit or byte, to keep
  parity with Python (`go/radixnet/encoding.go`).

**Lives in** `radixnet/encoding.py`, `go/radixnet/encoding.go`

---

### D-007 — The graph compresses itself like a radix tree, and the compression is lossy on purpose

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** representation

**Context** The lineage runs through `RadixTreeLLM` — "I chose a Radix Tree
specifically to focus on information compression."

**Decision** A unary chain (`p` has exactly one child `c`; `c` has exactly one
parent `p`; `p ≠ c`) is **merged** into a single node carrying a longer label.
A transition observed into or out of the middle of a merged node **splits** it
again. Compression runs after every epoch.

**Alternatives rejected**
* One node per trigram, never merged — no compression, and the node count
  grows with corpus size rather than with corpus *diversity*.
* Compressing once at the end of training — the structure would not be
  compressed while it is being learned on, which is when the cost matters.

**Consequences**
* **Merging is lossy and that is deliberate**: the merged node keeps the
  parent's parameters and discards the child's. The justification is that the
  chain was deterministic — probability 1, cost 0 — so predictions along it
  are unchanged. The *parameters* are discarded; the *predictions* are not.
  The test suite asserts exactly that: compression changes the node count but
  not the predictions of trained texts.
* Steps inside a compressed node are deterministic: no edge, no transition
  emitted, never counted. Everything downstream must agree on this or the
  counts diverge — the count model's passes are written around it, and the
  Go parity tests check it.
* Node ids are **not** stable across a merge. Anything holding an id across a
  compress is a bug; the training loop re-observes texts (with `count=False`)
  after any epoch that merged.
* The structure is self-limiting: repeated text costs nodes only where it
  *branches*.

**Lives in** `radixnet/graph.py::split / merge_child / compress`,
`go/radixnet/graph.go`

---

### D-008 — Prediction is a shortest-path search, not argmax and not sampling

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** inference

**Context** My specification named a cost function and Dijkstra.

**Decision** Edge cost is `−log P(c | p) + step_penalty ≥ 0`; prediction is
Dijkstra from the context node over the depth-unrolled graph, returning the
cheapest path that emits the requested number of characters (or reaches END
with `to_end`).

**Alternatives rejected**
* Greedy argmax per step — locally optimal, globally poor, and cannot honour
  a length target.
* Sampling as the primary mode — kept, but as `mode="sample"`, for diversity
  in the GAN loop (D-011), not as the default.

**Consequences**
* Non-negative costs are **required** for Dijkstra to be correct, which is why
  the cost is `−log` of a probability and `step_penalty ≥ 0`.
* The search systematically **prefers short, confident completions**, because
  cost accumulates per edge. `--step-penalty` and `--length` / `--to-end` are
  the levers that counteract it. This bias is a property of the method, not a
  bug to be fixed.
* Deterministic output for a given model and prefix — which makes the
  conversation loop (D-032) deterministic and repeat-free in beam mode, and
  makes testing tractable.
* Per-node edge costs are cached against `graph.version`; any mutation bumps
  the version and invalidates them.

**Lives in** `radixnet/search.py::dijkstra_predict`

---

### D-009 — 2NRL: train on garbage, invert, then fine-tune on the truth

**Status** Research claim · 2026-09-09 (`d180176`) · **Layer** learning

**Context — 2NRL (*Double-Negative Reinforcement Learning*) is my own learning
process, formalised.** This is not an algorithm I arrived at from the literature
and then justified. I am self-taught, and this is the method that produced that
education: **fail consistently, then do the inverse of what failed; and once a
thread worth pulling appears, pull hard.** 2NRL is that procedure written as an
update rule —
which is why the negative phase trains *on* the garbage at full rate rather than
away from it. You cannot invert a failure you have not first represented. The
full argument, the correspondence term by term, and where the analogy is
currently incomplete, are in `Research/2NRL.md`.

**Decision** `two_nrl(bad, good)` is three phases: (1) train on the bad data at
the full rate, (2) `invert()` the whole network, (3) fine-tune on the correct
data at a smaller rate (`pos_lr` default `0.01` against `neg_lr` `0.05`, with
`act_lr = pos_lr / 10`).

**Alternatives rejected**
* Negative sampling / contrastive loss — the conventional way to learn from
  wrong examples; it never inverts, so it cannot exploit D-010.
* Training on good data only — leaves the information in the failures unused.

**Consequences**
* Garbage is a **first-class input**, so the system needs sources of it:
  `data/sample_garbage.txt`, the LLM asked for deliberately wrong lines
  (D-029), failed generations (D-027), and rejected programs (D-030) all feed
  the same slot.
* The asymmetry between `neg_lr` and `pos_lr` matters: the negative phase moves
  the model hard, the positive phase corrects gently. Reversing the ratio makes
  the inversion dominate.
* Every downstream loop in the project is shaped to produce a `(bad, good)`
  pair, because that is the interface learning takes.
* The count model implements the *interface* but not the *mechanism* — see
  D-023.
* **"Pull hard on the thread" is not implemented, and is not a learning-rate
  decision.** The obvious reading is D-027's failure-proportional boosting.
  That is not what I meant: *explore rapidly and widely until a thread appears,
  then tighten the exploration and iterate.* That is a schedule over **search breadth** —
  temperature, beam width, sample count — not over learning rates, and nothing
  in the system currently anneals those. See D-067.

**Lives in** `radixnet/model.py::RadixNet.two_nrl`

---

### D-067 — Search breadth is fixed per call; my own process anneals it

**Status** Provisional — a recognised gap · **Layer** inference

**Context** The third commitment in the process 2NRL comes from (D-009) is
*explore rapidly and widely until you find a thread, then tighten the exploration
and iterate.* Wide-then-narrow: breadth first, then depth on whatever the breadth
turned up.

**Current behaviour** Every parameter that governs breadth — `temperature`, `k`,
`beam`, the sample `count`, `step_penalty` — is **fixed for the duration of a
call** and chosen by the caller. Nothing narrows as a run proceeds. A long evolve
or tutor run explores exactly as widely in its last generation as in its first.

**The trigger is settled; the schedule is not.** `Research/2NRL.md` §6.4 replaces
"what counts as finding a thread" with **what counts as finding a reward**, and
answers it with two events the system already emits: a judge paid out (the LLM
grader, the sandbox, a human mark, the discriminator), or the walk reached a
known area of completion (`reached_end`, a region an earlier run finished in).
Neither is an estimator inferred from the statistics of a search, so neither
needs a bar somebody sets in advance and neither can be tuned wrong. Nothing
currently listens to them for this purpose.

**What exists that is nearly right**
* D-033 already provides the machinery: rates as sandboxed expressions of the
  epoch, with `linear` / `geometric` / `cosine` / `step` / `warmup` helpers, a
  live preview and a reverse switch. It is pointed at the **learning rate**. The
  same evaluator applied to `temperature`, `k` and `beam` would be an annealing
  schedule over search breadth, which is what the process describes.
* D-062's exploration goes the *other* way on purpose, and correctly so: backing
  out of a loop, it widens (`k × (step + 2)`) the further back it goes. That is
  local recovery, not the global schedule — the two are compatible.

**Why this is recorded rather than built** It is a genuine feature, not a
documentation fix. Recorded so the gap stays visible rather than being lost.

**Would live in** `radixnet/schedule.py`, `radixnet/gan.py`, `radixnet/tutor.py`

---

### D-010 — Inversion negates every weight and every **whole** activation

**Status** Accepted · 2026-09-09 (`d180176`); corrected 2026-09-14 · **Layer** learning

**Decision** `invert()` sets `w → −w` on every alive edge, and `a → −a`
**together with `k → −k`** on every node, then flips the `inverted` flag.

**Why it works** The edge signal is `w · f_p · f_c` (D-005). Negating `w`
negates the product once; negating each endpoint's activation negates it twice
more — net, every edge signal changes sign, the softmax ordering reverses, and
the most likely continuation becomes the least likely. Two inversions are the
identity.

**Why `k` as well, which I had wrong** A unit is `f(x) = a·sin(b(x − h)) + k`.
Negating the amplitude alone gives `−a·sin(u) + k`, which is `−f(x) + 2k` — the
negation of the unit **only while `k` is 0**. And `k` is learned: `df/dk` is 1,
so every training step moves it. By the time 2NRL inverts anything, `k` is not 0.

The original rule negated `a` only. The consequence was not subtle once measured:
on randomised four-child nodes the ranking failed to reverse **87 %** of the
time, and after eight epochs of real training the edge signals came back as
−0.453 → +0.459 instead of +0.453. The most likely continuation was not becoming
the least likely, which is the one thing inversion exists to do — so the negative
phase's work was not being turned into avoidance, and 2NRL was quietly not doing
what it says.

**Why the tests did not catch it** They asserted the two properties that stay
true under the broken rule: `invert()` twice is the identity (negating `a` twice
restores it either way), and the sign flips of `w` and `a` themselves. Neither
looks at what inversion is *for*. And the graph-level test never trained, so
every `k` was still 0 and the bug could not appear. The regression tests now
assert the property that matters — every edge signal exactly negated, every child
ranking exactly reversed — with offsets deliberately moved off 0 first.

**Consequences**
* Nodes and edges created *while inverted* must be created with inverted
  defaults (`a = +1`, `w = −U(0.5, 1.5)`), or the graph becomes internally
  inconsistent. This is a real trap and is handled at creation time in
  `graph.py`.
* The `inverted` flag is persisted with the model, so a saved half-way-through
  2NRL model reloads correctly.
* Inversion is **global**. It cannot express "this one path was wrong" — which
  is what forced D-028.

**Lives in** `radixnet/graph.py::invert`

---

### D-011 — The system upgrades itself perpetually, GAN-style

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** learning

**Decision** `Evolver`: the model is the generator, a second `RadixNet` is the
discriminator. Each generation samples fakes, teaches the discriminator
real-vs-fake with 2NRL, scores the fakes, and feeds the worst ones back as the
generator's garbage with real corpus samples as the fine-tune pass. It runs
indefinitely — `generations=None` until a stop event.

**Alternatives rejected**
* A fixed training run — contradicts "constantly self-upgrading".
* A single network scoring itself — no adversary, so nothing pushes the
  generator anywhere.

**Consequences**
* Two models to persist, checkpoint and restore, not one.
* "Indefinitely" means the process must survive being told to stop: hence the
  stop events threaded through training, and D-037.
* The discriminator follows the generator's *kind* (D-020), so an evolve run
  on the count model gets a count-model discriminator.
* There is **no convergence criterion and no held-out evaluation**. The loop
  reports the real-vs-fake score gap, and a widening gap is taken as progress.
  Nothing in the system currently detects the generator getting worse in a way
  its own discriminator likes. See Q-4.

**Lives in** `radixnet/gan.py`

---

# Part II — Implementation platform

### D-012 — Pure Python, standard library only; torch optional; **no numpy**

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** platform

**Decision** `radixnet` imports nothing outside the standard library. `torch` is
imported lazily, never at module scope, and is never required. numpy is refused
outright — the torch backend even silences the linter about it
(`backend_torch.py`: *"this project deliberately has no numpy"*).

**Alternatives rejected** numpy as a baseline dependency — the obvious choice
for an N×N model, and rejected so the whole thing runs on a bare interpreter.

**Consequences**
* `pip install radixnet` pulls nothing. The Docker image without torch is small.
* All the performance work in D-013 exists *because* of this: no numpy means
  the hot loops must be hand-optimised Python over flat lists.
* Optional extras are genuinely optional and degrade rather than fail:
  `[images]` (Pillow) and `[diffusion]` (torch + diffusers) both fall back to a
  stand-in (D-036).
* The ceiling on pure-Python throughput is real — roughly 100k transitions/s —
  and is what eventually motivated the Go engine (D-038).

**Lives in** `pyproject.toml`, `radixnet/backend_torch.py`

---

### D-013 — Flat parallel arrays in the graph, CSR at the backend boundary

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** platform

**Decision** No per-node and no per-edge objects. The graph is parallel lists
indexed by id (`labels`, `z`, `a`, `b`, `h`, `k`, `count`, `alive`, `edge_w`,
`edge_count`, `edge_alive`) plus per-node dicts for adjacency. Once per epoch
the graph exports **CSR** (`indptr`, `indices`, `edge_ids`, `weights`,
`edge_pos`) and the backend trains on that.

**Alternatives rejected**
* `Node` / `Edge` classes — readable, and far too slow in CPython at this
  object count.
* CSR as the only representation — O(n) edge insertion, and training inserts
  constantly.

**Consequences**
* Two representations to keep in sync, with an explicit write-back
  (`apply_csr_weights`, `apply_node_params`). The boundary is the one place
  where a desync can happen, so it is narrow and explicit.
* Removal is by **tombstone** (`alive` / `edge_alive` false), not by deletion —
  ids stay valid. Ids are compacted only on save.
* Hot loops bind `math.sin`, `math.cos`, `math.exp`, `math.log` and the list
  references to locals before iterating. This is required by `DESIGN.md` §15,
  not left to taste.
* Insertion order of the adjacency dicts is **semantically significant**: it
  fixes the softmax summation order. The Go port had to replicate Python's dict
  ordering to match floating-point results (D-039).

**Lives in** `radixnet/graph.py`, `radixnet/backend.py::CSR`

---

### D-014 — A backend protocol, and `auto` picks torch only when there is a GPU

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** platform

**Decision** `Backend` is a protocol with four methods
(`prepare`/`step`/`finalize`/`node_activations`). `PythonBackend` is always
available; `TorchBackend` is optional. `get_backend("auto")` returns torch
**only if** torch imports *and* CUDA or MPS is available — never torch-on-CPU
by default.

**Rationale** Vectorised torch on CPU is not reliably faster than tuned pure
Python at these fan-outs, and picking it silently would add a heavyweight
dependency to the default path for no gain. `--backend torch --device cpu`
remains available for anyone who wants to measure it.

**Consequences**
* Two implementations of the same maths, which must agree. The contract is
  **1e-6 after one step on the same inputs**, enforced by
  `tests/test_backend_torch.py` and skipped when torch is absent.
* Dijkstra always runs on the CPU. Only the gradient step is accelerated.

**Lives in** `radixnet/backend.py::get_backend`, `radixnet/backend_torch.py`

---

### D-015 — Model files are JSON (optionally gzipped) and carry the RNG state

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** platform

**Decision** A model is a JSON document — `{format, version, saved_at, meta,
history, backend, graph}` — gzipped when the path ends `.gz`. The graph's
`random.Random` state is serialised with it.

**Alternatives rejected** `pickle` (unsafe to load, and not readable across
languages); a binary format (not inspectable, and would have to be re-specified
for Go).

**Consequences**
* **Training is reproducible across a save/load boundary**, which is what makes
  checkpoint-resume meaningful rather than approximate.
* The format is inspectable with `less` and diffable, which matters for a
  project whose point is to be examined.
* It is also the reason Go could be given the *same* file format rather than
  its own (D-038) — and the reason Go needed a CPython-compatible Mersenne
  Twister (D-039).
* Cost: large models are large files. Gzip is the mitigation.
* Dead nodes and edges are compacted on save, with ids remapped and START=0 /
  END=1 preserved.

**Lives in** `radixnet/model.py::save / load`, `go/radixnet/json.go`

---

### D-016 — The HTTP API is `http.server` from the standard library

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** platform

**Decision** `ThreadingHTTPServer` + `BaseHTTPRequestHandler`, hand-rolled JSON
routing, hand-rolled multipart parsing, hand-rolled static file serving with an
SPA fallback.

**Alternatives rejected** FastAPI, Flask — either would have violated D-012 for
what is, in the end, a route table.

**Consequences**
* ~2500 lines in `api.py` doing what a framework would do, including the
  multipart parser and the mime/traversal handling for static files. Accepted
  as the cost of zero dependencies.
* No schema validation for free, so request bodies are parsed through an
  explicit `Fields` helper with typed accessors and fixed error messages — and
  the Go server had to reproduce *those exact messages* (D-044).
* Tests start the real server on port 0 in a thread and talk to it over
  `urllib` — no test client to diverge from production behaviour.

**Lives in** `radixnet/api.py`

---

### D-017 — One long-running job at a time, behind an RLock, with 409 on conflict

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** platform

**Decision** A single `ModelService` owns the model behind an `RLock`. Train,
2NRL, evolve, feedback and codegen run on a background thread as a *job*; a
second request while one runs gets **409**. Jobs expose progress records, an
error, and a stop event.

**Alternatives rejected** A job queue — more machinery than a single-user
research tool needs; the 409 makes the constraint visible rather than hiding it
behind queueing.

**Consequences**
* The lock is released between epochs (`_yield_to_readers`) so `/api/status`,
  `/api/predict` and the frontend's 2-second poll keep answering during a long
  train. Without it the UI freezes for the whole run.
* The codegen trainer takes an `external` context-manager factory so the lock
  is also released around sandbox runs and LLM calls — those are seconds long
  and hold nothing the model needs.
* Every long operation must thread a `stop_event` all the way down, which is
  why `train`, `two_nrl`, `Evolver.run` and `CodeGenTrainer.run` all take one.
* The Go server reproduces this design, including the 409 and the mid-job
  release (D-044).

**Lives in** `radixnet/api.py::ModelService`, `go/server/service.go`

---

### D-018 — React + Vite, four dependencies, no TypeScript, and `dist/` is committed

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** platform

**Decision** The frontend depends on `react`, `react-dom`, `vite` and
`@vitejs/plugin-react` and nothing else — no UI kit, no chart library, no state
manager, no TypeScript. The built `frontend/dist` is committed and served by the
API.

**Consequences**
* Charts are hand-written SVG (`LineChart.jsx`), as is the graph view
  (`GraphView.jsx`, circular layout, edge opacity by probability, node radius by
  count). ~1300 lines of plain CSS instead of a framework.
* `python -m radixnet serve` works straight from a clone with no `npm install`.
  This is the reason `dist/` is committed and why several commits end "dist
  rebuilt" — the build artefact is part of the deliverable.
* The panels only ever speak the JSON contract, never model internals. That is
  what let the Go engine reuse the entire frontend unchanged (D-044).

**Lives in** `frontend/`

---

### D-019 — The specification was written before the code, and is maintained as a contract

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** process

**Context** The first RadixCyclicNN commit is `DESIGN.md` and sample data —
no implementation. The package arrived the next day.

**Decision** `DESIGN.md` is normative: *"the contract every module is
implemented against. Read it fully before writing code."* It carries exact
signatures, defaults, invariants and error behaviour, and every feature commit
updates it in the same commit as the code.

**Consequences**
* Invariants are stated once and asserted in tests (`DESIGN.md` §5.3 →
  `tests/test_graph.py`), so "correct" has a written definition.
* Performance requirements are *specified* (§15), not left to judgement.
* The spec is 1019 lines for ~12k lines of Python — a high ratio, and the
  reason a second implementation in another language (D-038) was feasible at
  all.
* This document exists because that one deliberately records *what*, not *why*.

**Lives in** `DESIGN.md`

---

# Part III — The second model

### D-020 — Two model kinds on one graph, selectable at run time, dispatched by the file's `format`

**Status** Accepted · 2026-09-10 (`29d2196`) · **Layer** architecture

**Context** The counting/reward idea is a genuinely different learning
algorithm, but it wants the same structure: the same self-compressing cyclic
trigram graph, the same search, the same decoding.

**Decision** Factor out `GraphModel` — graph, encoder/decoder, text cleaning,
observation, prefix location, `generate`, `score`, `compress`, `save`/`load`. A
*kind* supplies `train`, `predict`, `two_nrl`, `reward`, `punish`, `invert`,
`stats` and names itself (`kind` / `format` / `label` / `description`).
`load_model(path)` dispatches on the document's `format` field
(`"radixnet"` vs `"radixnet-count"`), so checkpoints, `POST /api/load` and the
CLI all restore whatever kind a file happens to hold.

**Alternatives rejected**
* A second project — would have forked the graph, the search and the frontend.
* A flag on one class — the two have *no* shared learning code; the branches
  would be everywhere.

**Consequences**
* The file itself declares its kind, so nothing has to be told out of band.
* The API keeps the inactive kinds parked in memory (`_parked`) and derives a
  per-kind path (`<stem>.<kind><ext>`), so switching kinds is instant and
  lossless.
* Both kinds can be loaded at once, which is what makes cross-kind conversation
  possible (D-032) — the radix model and the count model talking to each other.
* Every new capability now has to answer "what does this mean for both kinds?".
  Mostly this has been clarifying; occasionally it means a parameter is
  accepted and ignored (D-021).

**Lives in** `radixnet/model.py::GraphModel`, `radixnet/countnet.py`

---

### D-021 — The count model pins its activations to the constant 1

**Status** Accepted · 2026-09-10 (`29d2196`) · **Layer** architecture

**Context** The count model has no gradient and no activation learning — its
weight *is* a frequency statistic. But the base graph's scoring, softmax,
costs, Dijkstra, sampling, splitting and merging are all written against
`w · f_p · f_c`.

**Decision** Create every count-model node with `a = 0, k = 1`. Then
`f(z) = 0·sin(...) + 1 = 1` for every node and every `z`, so the score reduces
to exactly the edge weight — and **every inherited method works unchanged**.

**Alternatives rejected**
* Overriding `child_scores` / `child_costs` / `split` / `merge_child` in the
  subclass — four overrides, four places to drift out of sync with the base.
* A separate graph class — duplicates the radix machinery, which is the part
  worth keeping.

**Consequences**
* This is the single highest-leverage reuse decision in the codebase: ~700
  lines of `countnet.py` get a complete graph, search, beam, generation,
  scoring and persistence stack for free.
* Merging is safe because activation ratios are all 1, so a merge cannot change
  a score.
* `lr`, `act_lr` and `batch_size` are **accepted and ignored** on this kind, so
  a shared `TrainConfig` still works. Feedback magnitude is `strength` instead
  (default 1: one unit multiplies an edge's odds by *e*), which `RadixNet`
  accepts and ignores in return.
* The sine — the project's central research claim (D-002) — plays no part in
  this model at all. See Q-5.

**Lives in** `radixnet/countnet.py::CountRewardGraph._new_node`

---

### D-022 — The weight is a **dual** frequency function: all-time share × recent share, plus reward

**Status** Accepted · 2026-09-12 (`5961f6c`) · **Layer** learning

**Context** The first version weighted an edge by `log(1 + count)` — pure
accumulated frequency, with no way for the model to move on from what it saw
early.

**Decision** With `s = 0.5`, `C_p`/`W_p` the parent's all-time/windowed
traversals and `deg` its child count:

```
R_all    = (count + s) / (C_p + s·deg)          the edge's share, all time
R_recent = (window_count + s) / (W_p + s·deg)    the same share inside the window
weight   = count_scale·log1p(count) + global_scale·log R_all
         + window_scale·log R_recent + reward_scale·reward
```

Defaults `0, 0.5, 0.5, 1`. So `P(c | p) ∝ R_all^global · R_recent^window · e^reward`
— at `0.5 / 0.5` the **geometric mean** of the two shares, and when history and
the window agree, the probability *is* the share. The window is the last
`window` edge traversals seen anywhere in the graph (a deque, default 10 000).

**Alternatives rejected**
* Frequency alone (the previous version) — no recency, so the model cannot
  adapt.
* Recency alone — throws away everything learned before the window.
* An exponential decay per edge — equivalent in spirit, but has no crisp
  interpretation and no natural units; the window has both.

**Consequences**
* The scales are runtime-configurable (`configure(**scales, window=)`,
  `POST /api/model/weights`, the `weights` CLI command), so the balance between
  history and recency is a dial, not a rebuild. Shrinking the window trims it.
* The window state — its edge ids — is saved with the model and remapped like
  the edges, so a reloaded model has the same recent past.
* **Old files keep working**: a document from before this change loads with
  `count_scale = 1, global_scale = window_scale = 0`, reproducing the previous
  function exactly. Backwards compatibility is by construction, not by a
  migration.
* `recompute_weights()` is O(E) — walk every parent once. In Python this runs
  per text and became the dominant cost on large corpora; D-042 is the answer.
* Smoothing `s = 0.5` (Jeffreys) keeps an unseen edge's share finite rather
  than zero.

**Lives in** `radixnet/countnet.py::CountRewardGraph.edge_weight`,
`go/radixnet/weights.go`

---

### D-023 — The count model's 2NRL does **not** invert

**Status** Accepted · 2026-09-10 (`29d2196`) · **Layer** learning

**Decision** `CountRewardNet.two_nrl(bad, good)` is `punish(bad)` then
`reward(good)`. No inversion phase.

**Rationale** Inversion (D-010) exists because a gradient model has no way to
say "this specific path is wrong" — see D-028. The count model *does*: a
negative reward on the bad path's edges directly lowers their weight, hence
their probability. Inverting the whole graph on top would be a global answer to
a local problem.

**Consequences**
* The 2NRL *interface* is uniform across kinds, the *mechanism* is not. Callers
  (the feedback endpoint, the CLI, the codegen trainer, the evolve loop) are
  written against the interface and do not branch.
* `invert()` still exists on this kind — it negates the rewards — so the
  thumbs-down-only path (`punish` then `invert`) and the evolve loop's
  inversion still mean something.
* `neg_lr` / `pos_lr` are accepted and ignored here; `strength` is the knob.

**Lives in** `radixnet/countnet.py::CountRewardNet.two_nrl`

---

### D-024 — One search returns the K most likely **and** the K least likely continuations

**Status** Accepted · 2026-09-10 (`29d2196`) · **Layer** inference

**Decision** `beam_predict` runs **two** beams over the depth-unrolled graph in
one call: a *top* beam keeping the `beam` cheapest partial paths per step, and a
*bottom* beam keeping the `beam` dearest. It returns the `k` cheapest and the
`k` dearest complete paths.

**Rationale** With 2NRL (D-009), the least likely continuation is not a
curiosity — it is a candidate for the garbage half of a training pair. Getting
both ends of the distribution from one search makes that a single operation.

**Consequences**
* The top side stops early once no partial path can beat its k-th finished one.
* The bottom side **excludes anything already in the top list** — otherwise
  short paths that are both cheap and dear in absolute terms pollute it.
* With `to_end` and no cap, the bottom side is capped at about twice the
  longest top path, because "least likely" otherwise means "cycle for hundreds
  of steps" (D-001) and stops being comparable to the top.
* Available on both kinds. For the count model `"dijkstra"` is an alias of
  `"beam"`; for `RadixNet` the exact Dijkstra mode remains separate.

**Lives in** `radixnet/beam.py`, `go/radixnet/beam.go`

---

### D-025 — Generation *is* the prediction search, run to the end of a text

**Status** Accepted · 2026-09-12 (`40faaea`) · **Layer** inference

**Context** Generation and prediction had drifted into two code paths with
different behaviour — generation sampled, prediction searched — and the
difference showed up as inconsistent output between the tabs.

**Decision** `generate(mode="beam", count, ...)` is
`predict(prefix, length=0, mode="beam", k=count, to_end=True, max_length)`,
returning its `top`: the `count` most likely **distinct complete** texts, most
likely first. `"sample"` and `"dijkstra"` run through the same engine
(`GraphModel._search`). Every generated result is a whole text —
`text == full_text == prefix + continuation`.

**Consequences**
* With `k = 1`, beam generation agrees with Dijkstra's cheapest complete text.
  The modes are now points on one method rather than three methods.
* One place applies the partial-trigram lead and the cap rules, so the codegen
  trainer, the Ollama sampler and the evolve loop all see consistent text.
* Generation gained a **prefix** for free, which is what conversation (D-032)
  is built on.
* Beam generation is deterministic, so "generate 5 samples" gives five
  *distinct* texts rather than five draws that may coincide.

**Lives in** `radixnet/model.py::GraphModel._search / generate`

---

# Part IV — Feedback, failure, self-improvement

### D-026 — `reward` and `punish` are primitives on the model, not rules at the call site

**Status** Accepted · 2026-09-10 (`699e4fa`) · **Layer** architecture

**Context** Thumbs up/down appears in the Generate tab, the Predict tab, the
Converse tab, the CLI `feedback` command, the API and the codegen trainer. Each
of them wanting to spell out "positive-phase pass" or "negative pass then
invert" is five copies of the same rule.

**Decision** `reward(texts)` and `punish(texts)` are methods on the model.
`POST /api/feedback` dispatches: both sets rated → `two_nrl(bad, good)`; only
good → `reward`; only bad → `punish`. Each kind implements the pair its own way
(`RadixNet`: a positive-phase pass / a negative pass then `invert`; the count
model: a traversal plus reward / a penalty without one).

**Consequences**
* Human feedback and machine feedback take the same path. An LLM judge's
  verdict (D-029), a sandbox result (D-030) and a user's thumb all end at
  `reward`/`punish`.
* Feedback defaults are deliberately hotter than training defaults
  (`neg_lr=0.5`, `pos_lr=0.1`, `batch_size=4`) — a single rated text should
  visibly move the model, where a corpus should not.
* The frontend enforces one rating per distinct text and cannot reward the same
  text twice (`useRatings`), so the magnitude stays meaningful.

**Lives in** `radixnet/model.py`, `radixnet/api.py`,
`frontend/src/components/RatingsCard.jsx`

---

### D-027 — Train harder on worse failures: "blatantly fail on purpose, then invert"

**Status** Accepted · 2026-09-11 (`dc4eb78`) · **Layer** learning

**Context** The evolve loop fed the worst *half* of the fakes into 2NRL — a
fixed split that ignores how bad each one actually was.

**Decision** Score every fake against the real mean: `g = real_mean − score`.
Fakes with `g > 0` are failures; `g > blatant_margin` (default 1.0) are
*blatant*. In `fail_invert` mode the negative phase runs one pass per distinct
weight `w = min(boost, 1 + g/margin)` (default cap 4.0), heaviest first, with
`lr = w·neg_lr` and `act_lr = w·act_lr`. The worse the failure, the harder the
model is driven to reproduce it — *including its activation parameters* — before
the inversion turns all of it around.

**Alternatives rejected**
* The fixed worst-half split — discards the magnitude of the failure.
* Weighting the loss instead of the rate — `RadixNet.two_nrl` gained
  `bad_weights` and groups by distinct weight, which keeps the batched one-hop
  step (D-004) intact; a per-sample loss weight would not.

**Consequences**
* A generation with no failures does `reward(real)` only and **does not
  invert** — nothing is wrong, so nothing should be turned around.
* Records carry `failures`, `blatant`, `boost_mean`, `boost_max`, so the effect
  is visible in the CLI table and the Evolve tab rather than implicit.
* The count model scales its penalties by the same weights, so the mode means
  the same thing on both kinds.

**Lives in** `radixnet/gan.py`, `radixnet/model.py::RadixNet.two_nrl`

---

### D-028 — Local inversion flips **every other** node on a failed path

**Status** Accepted · 2026-09-11 (`dc4eb78`) · **Layer** learning

**Context** Inversion is global (D-010). Turning the whole network around
because one generated text was bad is a very large hammer.

**Decision** `invert_paths(texts, mode, amounts)` walks a failed text's path and
flips *alternating* nodes — `flip_nodes({node: amount}, mode)` applies
`value *= 1 − 2·amount` to `a` (mode `activation`) or `z` (mode `state`).

**Why alternating** An edge's score is `w · f_p · f_c` (D-005). Flipping the
sign of **both** endpoints leaves the product unchanged; flipping exactly one
negates it. So to make every edge of a path unlikely, you must flip every
*other* node — a two-colouring of the path.

**Why it is best-effort, and this is exact rather than sloppy** A path is always
two-colourable. A *graph with cycles* need not be: a graph is two-colourable if
and only if it contains no odd cycle, and a walk through a cyclic graph (D-001)
can close one. On a 3-cycle, no assignment of flips covers all three edges — the
best any of the eight patterns achieves is two. The obstruction is a theorem, not
an approximation. The extreme case is the self-loop: `score(p → p) = w · f_p²`,
and `f_p²` is non-negative whatever `f_p` is, so **flipping the node changes a
self-loop's score by exactly nothing** — a self-loop is sign-locked against node
inversion and can only be changed through its weight.

This is why the implementation says *the parity that covers the most edges, a
shared node takes the largest amount* rather than *the parity that covers every
edge*. It is the strongest statement available in a graph that has cycles in it —
and a clean instance of the general trade this project makes: allowing cycles
turned an operation that is exact on a DAG into one that is best-effort.
(`Research/CyclesAreAFeature.md` proves it.)

**Consequences**
* `amount` is continuous and gradual: 1 is a full sign flip, 0.5 zeroes the
  value (the node's transitions go neutral), less is attenuation. The evolve
  loop uses `min(1, g / (2·margin))` — attenuation for a mild failure, neutral
  at the margin, full flip at twice it.
* Blatant fakes handled this way **leave the 2NRL garbage set**; if every bad
  fake was blatant, the negative pass and the global inversion are skipped
  entirely. Local and global correction are alternatives, not layers.
* `version` bumps so the cost cache refreshes.
* This is a genuinely local correction in a model whose learning rule is
  already local (D-004) — arguably the more consistent of the two failure
  modes.
* The count model has no equivalent need: it penalises the path's edges
  directly by `2·strength·amount`.

**Lives in** `radixnet/graph.py::flip_nodes`, `radixnet/model.py::invert_paths`

---

### D-029 — A local LLM is an optional teacher and adversary, never a dependency

**Status** Accepted · 2026-09-10 (`36d00ef`) · **Layer** learning

**Context** 2NRL needs a supply of correct *and* deliberately wrong text, and
the evolve loop's discriminator is the model's own sibling — a weak critic.

**Decision** Talk to a local Ollama over its HTTP API with stdlib `urllib`
only. Two uses: **corpus from a prompt** (ask for N lines that are either
correct or deliberately wrong — exactly the two inputs of 2NRL), and
**adversarial review** (the LLM rates the network's own samples 0–10 with a
verdict and a critique; failures become garbage, passes become the fine-tune
set — an *external* discriminator).

**Alternatives rejected**
* A hosted API — sends the user's corpus off the machine and adds a key,
  a dependency and a bill to a source-available research project.
* An LLM client library — violates D-012 for what is two POST endpoints.

**Consequences**
* Everything degrades cleanly when Ollama is absent: `/api/ollama/models`
  returns 200 with `available: false` and an error string, never a 500.
* Ollama failures are surfaced as **502**, distinguishing "the LLM is down"
  from "your request was wrong" (400).
* Tests run against a fake Ollama server built on stdlib `http.server`, so the
  suite needs no model and no network.
* The LLM's judgement enters training through the same `reward`/`punish`
  primitives as everything else (D-026) — it is a *source of ratings*, not a
  special case.

**Lives in** `radixnet/ollama.py`

---

### D-030 — Code generation is the grounded reward signal: sandbox, objective style, then a judge

**Status** Accepted · 2026-09-10 (`1836687`) · **Layer** learning

**Context** Every feedback source so far is a judgement — a human thumb, an
LLM's opinion, a sibling discriminator. Code is different: a program either runs
and produces the expected output, or it does not.

**Decision** `problem → program → sandbox → judge → 2NRL`, in two
semi-supervised phases over one problem list. **Teacher phase**: the LLM writes
a program, the sandbox runs it, failures go back for a fix, wrong attempts
become 2NRL garbage and the correct one the fine-tune text. **Model phase**: the
network itself continues `"{problem}\n"` into code — first attempt by the
cheapest path, later attempts sampled — and is punished for every wrong attempt
and rewarded for the correct one.

**Key detail** The training text is
`solution_text(problem, code) = prompt + "\n" + code + "\n"`, so **the prefix
the network continues in phase 2 is exactly what it learned in phase 1**. The
two phases are the same string.

**Judging is layered, strongest first**: the sandbox (did it run? did stdout
match?), then an objective PEP 8 / naming checker written in-house
(`check_style` — W191, E111, E501, W291/W293, W292, E302; N801, N802, N803,
N806), then the LLM judge, which is given the *objective* report so it is
grading against facts rather than impressions. `judged_by` records which layer
decided — `sandbox | tests | ollama | none`.

**Consequences**
* This is the only reward in the system that does not ultimately reduce to
  someone's opinion.
* `strictness` lets style count or not; `fallback_teacher` supplies a correct
  answer when the network never gets there, so a hard problem still yields a
  training pair; `replay` adds up to `replay_limit` earlier correct texts to
  every positive phase, against forgetting.
* The LLM judge is optional (`use_judge`); with it off, the sandbox and the
  style checker still decide.

**Lives in** `radixnet/codegen.py`

---

### D-031 — The sandbox is defence in depth, and never raises

**Status** Accepted · 2026-09-10 (`1836687`) · **Layer** security

**Context** D-030 executes code written by a language model and by an untrained
neural network. The second is the more dangerous — its output is arbitrary
bytes that happen to parse.

**Decision** Layer the containment:
* `python -I -B` — isolated mode: no environment, no user site, no `.pyc`.
* A scratch working directory and an **empty** environment.
* `RLIMIT_AS` (memory, default 256 MB), `RLIMIT_CPU`, `RLIMIT_FSIZE` (16 MB)
  and `RLIMIT_CORE` (0), set **inside the child** before `runpy` runs the
  script — not on the parent.
* A wall-clock timeout on top of the CPU limit (a sleeping process burns no CPU).
* `unshare -rn` for a network namespace where the kernel allows it, probed once
  and skipped silently where it does not.
* It **never raises** for a bad program: a crash, a timeout or a hang comes back
  as a `RunResult` with `ok=False`.

**Consequences**
* Network isolation is **best-effort**, and the result says so
  (`network_isolated`). Under Docker's default seccomp profile `unshare` may be
  unavailable; the README documents the container boundary as the fallback
  protection. This is stated rather than assumed.
* Limits set in the child mean the API server itself is never constrained by
  them.
* Because the sandbox never raises, the trainer's control flow is
  straight-line: every attempt produces a verdict.

**Lives in** `radixnet/codegen.py::Sandbox`

---

### D-032 — The model converses with itself by predicting from the last words of the previous line

**Status** Accepted · 2026-09-12 (`4f21f88`) · **Layer** inference

**Context** Generation from START gives isolated texts. A conversation needs
each line to be *about* the previous one, and the only mechanism available is
the prefix.

**Decision** A reply is the prediction search (D-025) run from the tail of what
was just said. `tail_context(previous, context)` takes the last `context`
characters, cut *forward* to a word boundary when a word straddles the cut
("t on the mat" → "on the mat"). Then `predict(context, to_end=True)` offers the
`k` most likely complete continuations, and the first one that adds something —
and, with `avoid_repeats`, is neither already heard nor an *echo* of the
previous line — is spoken.

**The fallback chain is the design.** Nothing new follows, or the graph only
knows part of the last word → **drop the last word** and search again ("what
else could follow *on the*?") until the context is empty → then **change the
subject**: search fresh from START (`fresh=True`) → then, if even that only
offers repeats, speak the best repeat and flag it → and a voice with nothing at
all to say **ends the conversation**.

**Consequences**
* Beam conversations are deterministic and never repeat themselves — a direct
  consequence of D-008 and D-024.
* A second model can speak the second voice, so the radix model and the count
  model can talk to *each other* (enabled by D-020's parked models).
* A `history` parameter continues an existing conversation and returns only the
  new turns, so the UI's Continue button needs no server-side session.
* Every turn records what it picked up, what it added, its cost and probability,
  and its flags (`given`, `fresh`, `repeat`) — so a transcript is inspectable,
  not just readable. Turns are rated with the same thumbs as samples (D-026).

**Lives in** `radixnet/dialogue.py`, `go/radixnet/dialogue.go`

---

# Part V — Data path and control surface

### D-033 — Learning rates are **expressions of the epoch**, evaluated in a sandboxed AST

**Status** Accepted · 2026-09-10 (`0e4c9a7`) · **Layer** control

**Context** My framing was rates as *graph functions of the epoch* —
not a menu of named schedules, but the curve itself as an input. And rates that
*grow*, which most frameworks do not offer.

**Decision** `Schedule(expression)` parses with `ast.parse(mode="eval")`, walks
the tree, and rejects **every** node that is not a number, a whitelisted name, an
arithmetic/comparison/boolean operator, a conditional expression, or a call to a
whitelisted function. It evaluates with empty `__builtins__` and an environment
of `epoch` (1-based), `i` (0-based), `epochs`, `t`, `lr0`, `act_lr0`, `lr`, `pi`,
`e`, a list of `math` functions, and the helpers `linear`, `geometric`, `cosine`,
`step`, `warmup`.

**Two details worth keeping**
* **Integer literals are rewritten to floats**, so `2 ** 100000000` raises
  `OverflowError` instead of allocating a gigantic integer and hanging the
  server. A whitelist alone does not stop that.
* The activation schedule sees **the same epoch's** `lr`, so `lr / 10` follows
  whatever curve the learning rate has instead of being computed independently.

**Alternatives rejected**
* `eval()` — arbitrary code execution from an HTTP body.
* A fixed enum of schedules — the point was the curve as an input.

**Consequences**
* `validate()` evaluates every epoch **up front**, so a bad expression fails
  before training starts, not at epoch 40. `ScheduleError` is a `ValueError`,
  so the API answers 400 and the CLI exits 1 without special handling.
* The actual `lr` / `act_lr` are stamped on every epoch record, so the history
  shows what was used rather than what was asked for.
* `reverse_schedule` plays the list backwards keeping each epoch's pair
  together, so a ramp up becomes a ramp down without rewriting the expression.

**Lives in** `radixnet/schedule.py`

---

### D-034 — An uploaded archive is kept whole and unpacked in memory on demand

**Status** Accepted · 2026-09-11 (`219b1ad`), superseding the 2026-09-10 behaviour · **Layer** data

**Context** The first version unpacked a ZIP into one upload per text entry.
A 17 708-entry archive then became 17 708 rows in the upload list — unusable.

**Decision** Store the `.zip` itself as a **single** upload. Never write
unpacked entries to the upload directory. Every consumer goes through
`upload_entries(name)`, which unpacks the archive **in memory** on demand —
training, 2NRL, evolve, feedback and codegen problem files all take the same
path. The listing shows one row with a `ZIP · N files` badge, from a summary
cache keyed by size and mtime.

**Consequences**
* The upload directory mirrors what the user actually uploaded.
* An entry path **never touches the file system**: `flat_name` builds
  `<archive stem>__<dir>__<file>` from safe characters only, shortening
  over-long names to 128 characters with the extension kept and a hash of the
  full path appended. Zip-slip is structurally impossible, not filtered.
* Skip rules are explicit and shared with Go (D-039): directories, `__MACOSX`,
  `._*` / `.DS_Store` / `Thumbs.db`, nested archives, encrypted entries, binary
  content (a NUL in the first 8 KB), unreadable and empty entries — each with a
  recorded reason.
* `is_zip` decides by **magic bytes alone**, so a `.zip` that is really text is
  stored as text and a ZIP under any name is unpacked.

**Lives in** `radixnet/archive.py`, `radixnet/api.py`

---

### D-035 — No size limit on uploads

**Status** Accepted · 2026-09-11 (`c99bb80`), superseding the 2026-09-10 caps · **Layer** data

**Context** `go-master.zip` — 17 708 entries — was refused by a 10 000-entry cap
that had been added as a routine safety limit.

**Decision** Remove the defaults. `MAX_ARCHIVE_ENTRIES` and `MAX_UNPACKED_BYTES`
are `None`; `POST /api/uploads` is exempt from the 64 MB JSON body limit
(`MAX_UPLOAD_BYTES = None`). The limits remain available as *optional
arguments* for a caller that wants them.

**Rationale** This is a single-user research tool, usually on localhost. "Train
on a whole source tree" is a real use case; a denial-of-service cap protects
against a threat model that does not apply, at the cost of the use case.

**Consequences**
* A caller that *does* pass limits gets them enforced honestly: declared sizes
  are checked first, **and** the bytes actually read are counted against the
  same limit, so a lying ZIP header cannot get past it.
* Removing the cap forced the streaming work (D-043) — an unbounded archive
  cannot be held in memory, so the Go engine reads entries on demand and trains
  in chunks.
* JSON upload forms, which must be parsed whole, are still capped (512 MB in
  the Go server).
* If this were ever exposed beyond localhost, this decision would need
  revisiting first. See Q-6.

**Lives in** `radixnet/archive.py`, `radixnet/api.py`

---

### D-036 — Images become text by running the Stable Diffusion VAE **backwards**

**Status** Accepted · 2026-09-11 (`286bd56`) · **Layer** representation

**Context** The model consumes character trigrams and nothing else (D-006). To
train it on an image, the image has to *be* text — and a raw-pixel encoding
would be enormous and structureless.

**Decision** Run image generation in reverse. The SD VAE's **encoder** — the
inverse of the generation step — turns the image into its latent
(`4 × H/8 × W/8`), using the deterministic latent *mean* with no sampling noise.
Each number is quantised to one signed byte, and the bytes are base64-encoded as
`img:<encoder>:<w>x<h>:<base64>`. `decode_text` runs the forward process again
(base64 → latent → decoder → PNG).

**Consequences**
* An 8× linear reduction, and the latent is *semantically* structured — nearby
  values mean related content — which is what makes trigrams over the base64
  more than noise.
* **Predicted text can be decoded back into an image.** The decoder is written
  to repair what it gets: strip junk from the base64, complete the padding, pad
  or truncate the payload to the latent size. A model's half-learned output
  becomes a picture rather than an error.
* `TinyEncoder` — Pillow only, an RGB thumbnail at 1/8 scale — provides the
  same reduction and the same text format without torch, diffusers or the
  weights. `auto` prefers the VAE when it loads, records `sd_error` when it does
  not, and **does not retry the download on every image**.
* Optional extras `[images]` / `[diffusion]`; the core stays dependency-free
  (D-012).

**Lives in** `radixnet/vision.py`

---

### D-037 — Containers are stopped with SIGINT so they save

**Status** Accepted · 2026-09-10 (`b2adb78`) · **Layer** operations

**Context** The CLI handles Ctrl-C by finishing the current unit of work and
saving. Docker sends **SIGTERM**, so `docker stop` killed a long evolve run
after the grace period with nothing written — losing the whole run.

**Decision** `STOPSIGNAL SIGINT` on the api and evolve containers.

**Consequences** Verified: evolve stops in under a second and saves both the
model and the discriminator; the API shuts down cleanly. A perpetual loop
(D-011) is only viable if stopping it is safe, so this is load-bearing for the
central claim rather than an operational nicety.

**Lives in** `Dockerfile`, `docker-compose.yml`

---

# Part VI — The Go engine

### D-038 — Port the **count model only**, and leave Python untouched as the reference

**Status** Accepted · 2026-09-12 (`f8b8dd5`) · **Layer** platform

**Context** Pure Python (D-012) tops out near 100k transitions/s, and corpora
had grown to millions of lines. But the gradient model is where the research
claims live (Part I), and it is under active change.

**Decision** Port `CountRewardNet` to Go — a standalone module with a library
(`go/radixnet`) and a CLI (`go/cmd/radixnet-count`). Change nothing in Python.

**Rationale** The count model is the right half to port: no gradients, no
activation learning (D-021), no torch backend — just structure, counting and
search, which is exactly what parallelises. And it was the half hitting the
performance wall.

**Consequences**
* Two implementations of one model, which must stay in step. The contract is
  enforced by parity tests (D-039), not by discipline.
* Python remains the reference: when they disagree, Python is right.
* The Go module has **no dependencies beyond the standard library** — the same
  constraint as D-012, applied to a second language.
* The gradient model stays Python-only, so the frontend must know which engine
  it is talking to (D-044).

**Lives in** `go/`

---

### D-039 — Model files interchange **bit-identically**, which required a CPython MT19937 and exact summation

**Status** Accepted · 2026-09-12 (`f8b8dd5`) · **Layer** platform

**Context** "Both sides read the same file" is easy. "A model trained on one
side continues on the other with *identical numbers*" is not.

**Decision** Make interchange exact, and pay the three costs it implies:
* **`mt19937.go`** — a Mersenne Twister reimplementing CPython's seeding, its
  53-bit double generation and its `getstate()` layout, so `rng_state`
  round-trips between the languages and a continued run draws the same numbers.
* **`fsum.go`** — Shewchuk's exact summation, so path costs match
  `math.fsum` to the bit.
* **Insertion-ordered adjacency** in Go, because Python's dict order fixes the
  softmax summation order (D-013) and float addition is not associative.

**Alternatives rejected** "Close enough" (e.g. 1e-9) — the difference compounds
over epochs, and a tolerance would have hidden real divergence in the structure.

**Consequences**
* `tests/test_go_parity.py` can assert **equality**, not approximation: labels,
  counts, edges, rewards, window events, RNG state, weights (1e-12),
  predictions, generated texts, scores and conversation transcripts, with each
  side loading and continuing the other's file. That is a far stronger contract
  than a tolerance would allow, and it is what keeps the port honest as both
  sides change.
* Parity tests skip cleanly without a Go toolchain.
* Go's map iteration order is deliberately random, so insertion order had to be
  maintained explicitly — a real cost, accepted for reproducibility.

**Lives in** `go/radixnet/mt19937.go`, `go/radixnet/fsum.go`,
`go/radixnet/graph.go`, `tests/test_go_parity.py`

---

### D-040 — One goroutine per text, and counting is **racy by design**

**Status** Provisional · 2026-09-12 (`1d7e28d`) · **Layer** performance

**Decision** The unit of work is a **text**, not a batch: `SplitTexts` cuts a
corpus into lines, paragraphs, pages or files, and encoding, tracing and
counting fan one goroutine out per text (`--workers 0`, the default, spawns per
index; a positive cap gives a bounded pool). The counting pass bumps
`EdgeCount` / `Count` with **plain, unsynchronised increments** from all of
them. A collision loses an update, and that is accepted.

**The measurement, reported honestly** On a million-line corpus: **0.7 % of
traversals lost**, and **no speed-up** over a small exact pool — the per-text
goroutines and the contended cache lines cost more than they save on a 4-core
machine.

**Escape hatch** `Model.Exact` switches to `atomic.AddInt64`: no lost updates,
a result identical to the sequential run and to Python. `--exact` on the CLI and
the server; the parity tests use it.

**Consequences**
* The race is real, so the race-detector build (`race_on.go` / `race_off.go`)
  **skips** the racy test rather than pretending it is clean.
* Exact mode is required for D-039, so parity and speed are separate modes
  rather than a compromise between them.
* Marked Provisional: the measurement says the default costs 0.7 % accuracy for
  no measured gain. Whether the racy default is a *performance* choice (and
  should go) or a *research* position — that lossy, contended counting is what a
  brain does — is open. See Q-7.

**Lives in** `go/radixnet/model.go`, `go/radixnet/parallel.go`

---

### D-041 — Structure building stays single-writer

**Status** Accepted · 2026-09-12 (`f8b8dd5`) · **Layer** performance

**Context** Splits and merges reshape the shared trigram index and the adjacency
maps — the one phase that genuinely mutates shared structure.

**Decision** Keep it sequential. Detect walkable texts in parallel under a read
lock, then observe **only the novel ones**, in corpus order, under the write
lock.

**Two independent reasons** Go's runtime **aborts the process** on concurrent
map writes — not a data race to be tolerated like D-040, a crash. And a racy
structure would not be reproducible, which breaks D-039.

**Consequences**
* Amdahl's law applies: the sequential phase bounds the speed-up. It is
  mitigated by skipping texts that already walk, so a second epoch over the same
  corpus does almost no structural work.
* After `Compress()` every training text traces without a split, so the epochs
  never need the write lock at all — the sequential phase is paid once.
* "In corpus order" is not incidental: the sliding window (D-022) is a sequence,
  so its contents depend on the order traversals are applied in.

**Lives in** `go/radixnet/model.go::register`

---

### D-042 — Weights and costs are lazy, recomputed only for dirty rows

**Status** Accepted · 2026-09-12 (`f8b8dd5`) · **Layer** performance

**Context** The Python count model calls `recompute_weights()` — O(E) over every
parent — after every text. On a large corpus that is O(texts × edges), and it
was the measured hot spot.

**Decision** `RecordTraversals` and `AddReward` **mark** the parents whose rows
changed. `Prepare()` — called by every cost reader — recomputes only those rows;
after a structural change it recomputes every row in parallel (`parallelRanges`,
a goroutine per 64 nodes, uncapped). Edge costs (`−log softmax`) are cached per
version the same way.

**Consequences**
* Removes the dominant asymptotic cost of the Python implementation. This, not
  the goroutines, is where the Go engine's real win comes from.
* Correctness rests on *every* mutation marking its rows, and every reader
  calling `Prepare()`. A missed mark is a silent wrong answer, so the Go tests
  check lazy weights against a full recompute.
* The same optimisation has **not** been back-ported to Python. See Q-8.

**Lives in** `go/radixnet/weights.go`

---

### D-043 — Corpora stream through a `TextSource` and train in chunks

**Status** Accepted · 2026-09-12 (`1d7e28d`) · **Layer** performance

**Context** D-035 removed the size limits, so an archive may not fit in memory —
and training needs to re-read the corpus once per epoch.

**Decision** A `TextSource` yields texts in order and **can be re-read for every
pass**: `SliceSource` (memory), `FileSource` (line by line, any line length, BOM
and invalid UTF-8 repaired), `ZipSource` (opened from disk, central directory
only, entries streamed on demand), `MultiSource`. Training consumes it in chunks
of `ChunkSize` texts (default 8192): the structure pass walks the chunks once,
then every epoch re-streams them.

**The enabling trick** The loss is computed as the **traversal-weighted mean
edge cost** (`Σ traversals × cost / total`), so no list of transitions is kept.
That is what makes memory independent of corpus size; a transition list would
have defeated the whole scheme.

**Consequences**
* Measured: a million-line ZIP (39 MB unpacked) trains at a **flat 90 MB peak
  RSS**.
* The sliding window (D-022) is still applied in corpus order across chunks, so
  chunking does not change the model.
* The server streams multipart parts and raw bodies **straight to disk**
  (`StoreStream`, first four bytes decide text or ZIP), and inspects archives by
  streaming, so an upload is never held whole in memory either.

**Lives in** `go/radixnet/source.go`, `go/radixnet/model.go::passesSource`

---

### D-044 — Two engines, one JSON contract; the frontend detects which it is talking to

**Status** Accepted; the Python-only split **superseded by D-066** ·
2026-09-12 (`ec78de9`) · **Layer** architecture

**Context** The Go engine needed a server, and the React frontend (D-018)
already existed and worked.

**Decision** The Go server speaks the **Python API's contract exactly** —
including typed body fields with the *same error messages*, the same job
lifecycle with 409 and mid-job reader access, the same upload body forms, the
same checkpoint layout on disk, the same static-file behaviour and SPA fallback.
Endpoints only the Python engine has (evolve, ollama, images, codegen, schedule
preview) answer **404 with a message naming the Python server**.

The frontend reads `engine` from `/api/health` and `/api/status`. With `"go"` it
hides *and does not mount* the Python-only tabs, shows a **Go engine · N
goroutines** badge, and replaces the accelerator chips with engine / workers /
live goroutines.

**Consequences**
* **The entire frontend is reused unchanged** — "the panels only ever spoke the
  JSON contract" (D-018) paying off directly.
* A 404 that names the other server is a better failure than a hidden button:
  the capability difference is discoverable rather than invisible.
* The contract is tested from both sides: `go/server/server_test.go` (httptest,
  every endpoint, error shapes) and `tests/test_go_parity.py::TestGoServer` (a
  live `serve` process checked against the Python API tests' key sets, its saved
  model loaded in Python, and its checkpoints read by the Python
  `CheckpointManager`).
* `POST /api/train` gains `split` and `page_lines` on the Go side — the units
  the goroutines fan out over (D-040) — which the Train tab surfaces as
  "Texts are: lines | paragraphs | pages".

**Lives in** `go/server/`, `frontend/src/App.jsx::engineOf`

---

# Part IX — The negative network

The largest development since the count model, and a genuine evolution of D-009:
2NRL's negative phase, made **permanent**. Rather than training on failure,
inverting and discarding the representation, the system now keeps a standing
model of *how text goes wrong* and consults it at output time.

### D-045 — A second network that models failure, not correctness

**Status** Accepted · 2026-09-12 (`1f921e5`) · **Layer** architecture

**Context** Every failure the system found — a tutor's correction, a rejected
program, a bad review, a thumbs-down — was consumed once by a 2NRL pass and then
gone. The knowledge *that* a particular fragment tends to be wrong existed only
as a diffuse change to the positive model's weights.

**Decision** A third model kind: the same self-compressing cyclic graph, the same
trigram window, the same searches — but **every node and edge in it exists
because something went wrong there**. Each edge keeps the blame it collected, how
often it failed, how much *cleared* text crossed it, and the reasons behind that
blame.

The net evidence against an edge is

$$\text{evidence} = \max(0,\ \text{blame} - \text{clear\_scale} \cdot \text{clear})$$

so a fragment appearing in good and bad output alike stops carrying the verdict.
Weights are an edge's share of the failure mass leaving its parent, so **a
softmax over them is the failure distribution**: the network predicts the ways to
fail from a prefix, exactly as the positive model predicts the ways to succeed.

**Alternatives rejected**
* Keeping failures as a corpus and re-running 2NRL — no structure, no way to ask
  "is *this* text likely to be wrong?", and no way for evidence to accumulate or
  decay.
* A flag on the positive model's edges — cannot represent a fragment that is
  good in one context and bad in another, and pollutes the positive weights.

**Consequences**
* Failure becomes **queryable**. `judge(text)` walks a text through the failure
  structure and reports risk, peak, coverage, the reasons and the fragments
  carrying them — a verdict with evidence, not a score.
* The same graph machinery serves a third purpose, for the same reason D-021
  worked: the structure is independent of what the weights mean.
* It is a model kind like the others (`radixnet-negative`), so it saves, loads,
  checkpoints and crosses to Go by the same route (D-020, D-039).
* `forget` exists, because evidence that is never released makes the network
  permanently pessimistic.

**Lives in** `radixnet/negative.py`, `go/radixnet/negative.go`

---

### D-046 — Blame only the characters that actually changed

**Status** Accepted · 2026-09-12 (`27ac0ac`) · **Layer** learning

**Context** A grade reached the graph as two verdicts on two whole sentences: the
attempt was garbage, the teacher's correction was gospel. But most of a corrected
sentence is word for word what the network wrote — the teacher changes a tense,
an article, a plural. The whole-sentence penalty **taxed the trigrams that were
right**.

**Decision** Align the two sentences character by character (`diff.py`: common
prefix and suffix trimmed, the middle by a longest-common-subsequence walk,
changes closer together than a trigram merged so `"mat" → "park"` is one change
rather than two). Then charge each step of the traced path with the characters it
writes, and:

* steps of the wrong sentence that wrote a **struck-out** character lose
  `strength × weight` — and only those;
* steps of the correction that write the **teacher's version** gain
  `strength × reward`; the rest keep `keep_weight` times as much
  (`keep=0` teaches the fix alone, `keep=1` is the old whole-sentence thumbs up);
* the correction is still traversed once — it is correct English whatever
  changed — and **an edge both sentences walk is rewarded, never penalised**.

**Consequences**
* A correction is taught *as a correction*, at the granularity the teacher
  actually worked at. This is the same principle the agent later applies to tool
  calls (D-057): blame at the granularity the failure happened at.
* A blamed transition is **kept out of compression** (D-007), so the fragment
  stays nameable. Compression and blame are in direct tension, and blame wins.
* A failure the teacher left uncorrected is still whole-sentence garbage —
  the fine-grained path needs a correction to diff against.

**Lives in** `radixnet/diff.py`, `radixnet/negative.py::correct`, `go/radixnet/diff.go`

---

### D-047 — The pair guards every answer, and switching it **off** is what takes a flag

**Status** Accepted · 2026-09-14 (`f6585b8`) · **Layer** architecture

**Context** The negative network was something a caller had to ask for. Everything
else handed out whatever the positive model wrote — including the sentence the
tutor had corrected an hour earlier.

**Decision** `generate`, `predict` and `converse` run the two networks in tandem
**by default**. The positive model over-samples; the negative one vetoes by
blame, by **peak** blame on a single fragment (one corrected word vetoes an
otherwise clean sentence), or by the **likelihood ratio**
$\log P_{\text{neg}} - \log P_{\text{pos}}$ — all behind a *coverage gate*, so
text the tutor has never failed is never vetoed on no evidence.

**The guard stands aside rather than guessing.** It builds the pair only when
there is something to guard with: not when the negative network is the active
model (it is generating *from* the failures), not when there is none in memory or
saved beside the model, and not when the one there has never been taught a
failure and would veto nothing. **It never creates an empty negative network** —
an answer is not the place to bring one into being. An unguarded answer is byte
for byte what it was before, and a null `guard` in the payload says so.

**Consequences**
* This is the GAN idea (D-011) moved to **output time**: generator and
  discriminator on one path, rather than a training loop.
* Each surface re-ranks as little as it can. `generate` sorts on risk alone with
  a stable sort, keeping the model's own order among equally clean texts.
  `predict` re-ranks what the search already offered rather than searching again.
  `converse` treats a vetoed reply exactly like one already spoken — except a
  vetoed candidate may not even be the fallback.
* Every answer carries what was stopped and why, so a veto is inspectable.

**Lives in** `radixnet/duo.py`, `radixnet/negative.py`

---

### D-048 — The teaching loops read the model **unfiltered**

**Status** Accepted · 2026-09-14 (`f6585b8`) · **Layer** architecture

**Decision** The loops that *teach* the negative network — the critic, the tutor,
evolve — keep sampling the positive model directly, bypassing the guard.

**Rationale** *A reviewer that only ever saw what already passed the filter would
have nothing left to teach.* Filtering
the training signal with the filter being trained is a closed loop that converges
on its own blind spots.

**Consequences** Two sampling paths with different defaults — guarded for
answers, unguarded for teaching — which is a real asymmetry to keep in mind when
adding a new loop. The rule is: if the output goes to a person, guard it; if it
goes to a judge, do not.

**Lives in** `radixnet/critic.py`, `radixnet/tutor.py`, `radixnet/gan.py`

---

# Part X — Teaching loops

Six loops now teach the network, all reducing to the same `(bad, good)` pair
(D-009) and the same `reward`/`punish` primitives (D-026).

### D-049 — The tutor: an LLM sets the exercise, the network answers, the LLM marks it

**Status** Accepted · 2026-09-12 (`907a7af`) · **Layer** learning

**Decision** One round is
`topic → prefix (LLM) → completion (prediction search) → grade (LLM) → 2NRL`.
The teacher writes sentence openings, each drilling one point of grammar and each
with its own model answer; the network completes them with the ordinary
prediction search; the same model marks every sentence as an English teacher —
grammar, spelling and fluency out of 10, the worst mistake named **from a fixed
list**, one line of teaching, and the sentence written out correctly.

**Consequences**
* The network is graded on a *completion*, which is what it actually does. The
  exercise is shaped to the model rather than the model to the exercise.
* The fixed mistake list is what makes the negative network's reason table
  possible (D-045) — a free-text critique cannot be tallied.
* The teacher's own corrected English enters the positive phase at full weight:
  it is, at that moment, exactly what a good answer would have looked like.

**Lives in** `radixnet/tutor.py`, `go/radixnet/tutor.go`

---

### D-050 — Rewards are weighted by the mark, not by a single thumb

**Status** Accepted · 2026-09-12 (`907a7af`) · **Layer** learning

**Decision** `two_nrl` takes `good_weights`; `reward`/`punish` take `weights`;
**every kind** scales the pass per text, in whatever currency it learns in — a
learning rate (radix), a reward (count), a phase lock (resonant) or blame
(negative). **A 9-out-of-10 sentence is learned nine tenths as hard as a
perfect one; a 0 is skipped.** `/api/feedback` and `/api/2nrl` accept
`good_ratings` / `bad_ratings` (marks out of 10) or raw weights, and the
Ratings card carries a mark per rated text.

A kind that took the ratings but not the rest of the call was the same bug
twice: the resonant and negative models let `weights` / `progress` /
`stop_event` fall through into their `TrainConfig` overrides, and since the
Ratings card always sends marks, selecting either kind failed every feedback
job with `unknown train option(s)`. The four kinds are interchangeable by
design, so the whole feedback call is one interface and
`tests/test_feedback.py::KindParityTests` holds every kind to it.

**Rationale** A binary thumb throws away most of what a grader knows. It also
makes the boost mechanism of D-027 available on the *positive* side — see Q-13,
which this partly answers.

**Consequences** Feedback magnitude becomes continuous everywhere, so every
judge in the system (LLM grader, sandbox, discriminator, human) can express
confidence rather than only direction.

**Lives in** `radixnet/model.py`, `radixnet/countnet.py`, `radixnet/resonance.py`,
`radixnet/negative.py`

---

### D-051 — The difficulty step is read off the marks, never invented by the LLM

**Status** Accepted · 2026-09-13 (`448c7a1`, `6a70746`) · **Layer** learning

**Context** A run ends with a report card — the marks, the pass rate, how often
each mistake was the worst thing in a sentence — which is handed back to the
teacher, which answers with a syllabus that repairs it.

**Decision** The LLM writes the *prose*; the **marks** decide the difficulty. It
is not asked for a level at all.

| Step | Condition | Effect |
|---|---|---|
| `advance` | 80 % passed at 8/10 | next level, openings one rung up (3–6 → 5–8 → 7–12 → 10–16), pass mark +1 (max 9) |
| `stretch` | 50 % passed | same level, openings one rung up |
| `hold` | anything less | nothing harder: the weak points, and 3 correct sentences to imitate |

**Rationale, in one line:** *a student who is failing must not be given a harder
exercise.* An LLM asked "what level next?" will happily say "harder" because the
conversation has a forward momentum of its own.

**The floor.** `plan_from_card` — one lesson per weak point, worst first, no LLM
involved — is not a last resort but the **floor the teacher has to improve on**.
A weakness the teacher's plan skips displaces a lesson that drills nothing the
card marked down; an unreadable answer leaves the card's own plan standing.
`source` records which of the two wrote it, **so a plan is never a hallucination
presented as a syllabus.**

**Consequences** This is the general pattern for LLM use in this project: the
model supplies language, a deterministic rule supplies the decision, and the
record says which. Compare D-030 (the judge is given the objective style report)
and D-057 (blame granularity is computed, not asked for).

**Lives in** `radixnet/tutor.py::upgrade_from_card / plan_from_card`, `go/radixnet/plan.go`

---

### D-052 — The auto run: the report card writes the next batch's instructions

**Status** Accepted · 2026-09-13 (`e7921dc`) · **Layer** learning

**Decision** `lessons → report card → plan → apply → lessons → …`, with nobody in
the loop. The plan's **brief** — two or three sentences naming the points of
grammar to drill in order — becomes the standing instruction handed to the
exercise writer for every round of the next batch. One batch's report card is
literally the next batch's prompt. `--batches 0` runs until stopped.

**Consequences**
* A batch that cannot be planned **ends the run** rather than repeating itself —
  an explicit refusal to spin.
* The single-focus pin is released when a brief is in force, because the brief
  carries the grammar points in order and a stale pin would silently overrule it.
* This and the critic loop (D-053) are the two places the system genuinely runs
  unattended for long periods.

**Lives in** `radixnet/tutor.py::apply_plan`

---

### D-053 — The critic loop only ever *reads* the positive model

**Status** Accepted · 2026-09-14 (`6695e47`) · **Layer** learning

**Context** Every source of negatives arrived as a side effect of something else
running. The Negative tab was the one surface where a person had to type a
failure in by hand.

**Decision** A loop: the positive model writes texts; an LLM reviewer marks each
out of 10 and says what is wrong; everything below the pass mark blames the
negative network, with the critique picking the reason and the mark setting the
severity; what it passed **clears** blame off the fragments they share.

**Two invariants make it safe to leave running**, and they are worth stating as
invariants rather than as properties:
1. The positive model is **only read from** — nothing here trains, rewards or
   inverts it.
2. The slow call is wrapped so the server releases its model lock while the
   reviewer thinks (D-017).

**Consequences** An unattended loop that cannot damage the thing it is studying.
`--context` is the reviewer's yardstick and matters, because "is this good?"
means little without one.

**Lives in** `radixnet/critic.py`, `go/radixnet/critic.go`

---

### D-054 — The recall tutor needs no LLM, because the answer is already on file

**Status** Accepted · 2026-09-13 (`da9126c`) · **Layer** learning

**Context** Speech and images were encode-and-train paths with nothing in them
that could look at an output and say it went wrong.

**Decision** An utterance and a picture were **encoded into text before being
trained on** — so asking the network to write that text back is an exercise whose
correction already exists. The exercise is the opening of a text it was taught
(the utterance's own token, or an image header plus a few characters, since a
header alone names no picture); the completion is the ordinary prediction search;
the marking runs the result through the codec and compares it with the original.
Agreement over the payload is the mark out of 10, and the single worst fault is
named: *unreadable, truncated, overrun, garbled, silence/blank, clipping/noise,
mishearing, distortion/drift*.

**Consequences**
* No LLM, no Pillow, no torch, no transcriber — the marking is over payload bytes
  both codecs already produce, so none of its tests skip.
* Faults are judged **against the reference**: a recording that really was silent
  is not blamed for coming back silent, and mishearing is only diagnosed where
  the round-trip transcript is known.
* The original text is the correction, so D-046's character-level blame applies
  directly.

**Lives in** `radixnet/recall.py`, `go/radixnet/recall.go`

---

### D-055 — Chat: an LLM holds the other end of the conversation

**Status** Accepted · 2026-09-14 (`3845ae4`) · **Layer** learning

**Context** `converse` has the model talk to itself (D-032) — a good way to see
what it knows and a useless way to find out whether it *answers* anything, since
neither voice can tell the other that its reply did not follow on. Every teacher
until now talked *at* the network.

**Decision** Put a real language model on the other end. The partner is told to
keep its lines short, plain and easy to carry on from — *because that is the only
thing a character-level model can reply to; the prompt is doing the model a
favour, not flattering it.* The model replies the only way it can: the tail of
that line is located in the graph and continued, so **a reply is a real walk of
the network and not a prompt trick.** The judge marks every reply out of 10
against *the line it answered* — not against a style guide — and gives the
conversation as a whole a verdict, in one call rather than one per line.

**Consequences**
* `dialogue.reply` was extracted from `converse` so there is one implementation of
  "what does this model say next" and the other voice need not be a model at all.
  `converse` is now a loop over `reply`.
* **The partner's own lines go into the positive phase**, because in that
  conversation, at that moment, they are exactly what a good reply would have
  looked like. This is a neat use of the 2NRL good-set: the correct answer is
  free, and it is contextual.
* Unlike the critic (D-053) this loop *does* train the positive model.

**Lives in** `radixnet/chat.py`, `go/radixnet/chat.go`

---

# Part XI — Tool use and the agent

### D-056 — A tool call is text the network writes, so an attempt is one training text

**Status** Accepted · 2026-09-14 (`5abb608`) · **Layer** architecture

**Context** The network cannot decide to call a function. It can only emit
characters.

**Decision** Make that the whole design rather than working around it. A tool call
**is** text: `<tool>web_fetch {"url": "..."}</tool>`. The observation is text it
reads back. Therefore **a whole attempt — task, calls, observations, answer — is
one ordinary training text**, which is exactly what 2NRL can reward or punish as
a unit.

**Consequences**
* No new learning machinery at all. Tool use is trained by the same pass as
  everything else.
* The call format is characters the model must learn, which is why the Go port
  had to render JSON **exactly** as `json.dumps(ensure_ascii=False,
  sort_keys=True)` does — sorted keys, a space after every colon and comma,
  floats as `repr()` writes them. Go's encoder writes `{"url":"u"}` where Python
  writes `{"url": "u"}`, and *that one space is the difference between a
  transcript the other side can read and one it cannot* (D-039's principle,
  applied to a learned format).
* Parsing must be lenient — truncated JSON is repaired — because a generator
  will cut a call off mid-string.
* The first attempt is the **beam** search rather than Dijkstra's single cheapest
  path, which ends at END after a couple of characters that can never be a whole
  call. Candidates that closed their tag are preferred, and the first usable one
  is the network's own move, tracked as `autonomy`.

**Lives in** `radixnet/tools.py`, `radixnet/agent.py`

---

### D-057 — The LLM gets four roles, and solving the task is the **last** one

**Status** Accepted · 2026-09-14 (`5abb608`) · **Layer** learning

**Decision** The LLM is used in four distinct roles, ordered so that it does as
little of the work as possible:

1. **Criteria** — it writes the acceptance criteria *before anything is
   attempted*, so the bar is not set after the fact.
2. **Mediator** — it turns an emission the network cannot form into one valid
   call against the real schemas (native tool calling, then JSON, then a
   fallback), *so even an untrained network makes progress and learns
   well-formed calls*.
3. **Judge** — it marks the finished transcript against those criteria.
4. **Demonstrator** — and **only on failure** does it do the task itself, with
   the same real tools, as the correction.

**Consequences**
* The mediator solves the cold-start problem without giving the answer away: the
  network's malformed attempt is repaired into something that *works*, and the
  repaired form is what it learns.
* `explore()` takes the task-setting away too — the network continues `"TASK:"`
  into whatever it is reaching for, the LLM turns that into one concrete question
  browsing can settle, and links found along the way become the frontier, so it
  compounds.

**Lives in** `radixnet/agent.py`

---

### D-058 — Failure is blamed at the granularity it happened at

**Status** Accepted · 2026-09-14 (`6b8c776`) · **Layer** learning

**Decision** Three failures that look alike are blamed differently, *because they
are different failures*:

| Failure | What is blamed | Why |
|---|---|---|
| The answer was wrong | the transcript, diffed against a **correct run of the same task** | only characters differing from a run that worked are blamed; the shared task line and the calls that worked never become evidence |
| The mediator had to repair the call (`bad-call`) | **the emission the network wrote** | it is not in the transcript, which holds the *repaired* call. Blaming the transcript *would teach the network that a well-formed call is a mistake* |
| The tool refused the call (`tool-error`) | the call **only if the network wrote it** | one the mediator wrote is not the network's fault |

The reason comes from how far the attempt got (`no-call`, `bad-call`,
`tool-error`, `no-answer`, then the judge's own words); the severity from its gap.

**Consequences**
* What it learns comes back the other way: a candidate the negative network
  recognises as a known failure is passed over for the next one, so the failures
  stop it repeating them (D-047).
* This is D-046's principle generalised — blame the thing that was actually
  wrong — and it is the single most error-prone area in the system, because each
  of the three cases is plausible-looking and wrong in a different way.

**Lives in** `radixnet/agent.py`, `radixnet/blame.py`

---

### D-059 — Any LLM provider, and the key is never stored

**Status** Accepted · 2026-09-12 (`870d4bf`, `0dd6b57`) · **Layer** platform

**Decision** `LLMClient` is the shape both clients share; `make_client(provider)`
builds one from `"ollama"` or `"chatgpt"`. Teacher and judge choose independently,
so a ChatGPT teacher can be marked by a local model or the other way round, and
every grade records **which provider marked it**.

**Key handling is deliberate and strict.** The key comes from `$OPENAI_API_KEY`
(or `$OPENAI_API_KEY_FILE`, for a Docker secret), is read **per request**, and
never lands in a config, a job record, a report or a `repr`. **The HTTP API has
no field for it at all**, so the server always uses its own. A key is refused over
plain HTTP to a remote host. A model that rejects an optional field (reasoning
models refuse `temperature`, older ones `response_format`) is retried without it
rather than failing.

**Consequences** Still no dependency: both clients are `urllib` (D-012). An
answer that cannot be read is an `LLMError` rather than a provider-specific one,
*because it says nothing about the transport*.

**Lives in** `radixnet/llm.py`, `radixnet/chatgpt.py`, `go/radixnet/llm.go`

---

### D-060 — Browser and MCP, both on the standard library

**Status** Accepted · 2026-09-14 (`6b8c776`) · **Layer** platform

**Decision** `browser.py` drives Chrome over the **W3C WebDriver protocol** —
which is HTTP with JSON bodies, so `chromedriver` is started and spoken to with
the standard library and **no driver package is needed**. A page that draws itself
with JavaScript then reads as a person sees it rather than as "loading…".
`mcp.py` serves the tools and the network over the **Model Context Protocol** on
stdio (JSON-RPC 2.0, standard library only): predict, generate, score, stats,
judge against the negative network, and solve a task through the whole agent loop.

**Consequences**
* `chromedriver` refuses to drive another major version of Chrome — far and away
  the most common failure — so both versions are read and compared **up front**
  instead of surfacing a raw WebDriver error.
* `$RADIXNET_WEBDRIVER` attaches to a Selenium Grid or a standalone-chrome
  container instead of starting anything.
* A failing MCP tool is a **result with `isError`**, not a protocol error — the
  distinction matters to a client.
* The address guards (no credentials, no private / loopback / link-local, every
  redirect hop re-checked, byte and time caps) run first, on every path.

**Lives in** `radixnet/browser.py`, `radixnet/mcp.py`, `radixnet/tools.py`

---

# Part XII — Metacognition

This part **answers the question D-001 left open**. The claim was that on hitting
a cycle the brain hands over to metacognition rather than looping. As of
2026-09-14 that hand-over exists.

### D-061 — A stutter is the shape a cyclic graph falls into when it walks a loop

**Status** Accepted · 2026-09-14 (`ba055f6`) · **Layer** inference

**Decision** `stutter(text)` finds a run of one to four words repeated
**immediately** after itself — "the *the* west", "say morning *morning*", "*the
cat* the cat sat". Only immediate repetition counts, so English that repeats a
word and means it — "where there is a will there is a way", "a bird in the hand
is worth two in the bush" — is left alone.

**Why this is the right detector** It is the *signature of the failure mode*, not
a generic quality heuristic. A cyclic graph that walks a loop instead of going
somewhere emits exactly this shape. The detector is derived from D-001.

**Consequences**
* Such a candidate is skipped like one already heard; when every candidate
  repeats, the one spoken anyway is flagged, **punished through the 2NRL negative
  phase**, and ends a conversation that would only go round in circles.
* The flag is recorded whether or not the setting is on, *so a transcript says
  what happened either way* — switched off, the repetition is neither skipped nor
  punished, but the information survives.

**Lives in** `radixnet/dialogue.py::stutter`, `go/radixnet/dialogue.go`

---

### D-062 — A voice that catches itself repeating **backs up and explores**, rather than discarding the turn

**Status** Accepted · 2026-09-14 (`414bd1d`) · **Layer** inference

**Context** Skipping a looping candidate throws away everything it got right. The
words *before* the walk went round were said once and were the most likely thing
to say.

**Decision** A repeat is no longer a dead end. Three steps:

1. **Noticing.** `_pick` hands back `caught` — the best candidate rejected for
   repeating, of either kind. Where it backs up to depends on which: a *stutter*
   is cut at `stutter_at(text)`, where the walk went round; a *repeat* of
   something the conversation has already heard (`Heard.match`) is cut at its
   last word, because the line is a retread from end to end and that is the
   latest point at which it can still differ.
2. **Backing up.** `backtrack()` keeps everything said before that point and runs
   the search again from there. **The longer prefix is the whole trick**: it
   forces the walk to leave the loop at exactly the point it went round, where
   asking the same question again from the original context would only rank the
   same answers.
3. **Exploring.** Candidates that stutter, that the conversation has heard, or
   that the guard vetoes are passed over; the first that says something new is
   spoken as an ordinary turn **with nothing to punish**. Nothing new? It backs up
   one word further and looks wider — `k × (step + 2)` candidates, *so the further
   back it goes the more it weighs* — `explore` times over, then falls through to
   what it would have done anyway.

**A constraint worth naming: the voice may rethink what it said, never what it
heard.** What it picked up from the other voice it may not rewrite, and a repeat
inside those words is recorded and left alone.

**Consequences**
* This is the metacognitive hand-over of D-001, made concrete: on detecting that
  the search has gone round, control passes to a *different* procedure — one that
  re-plans from an earlier point and widens its search as it goes — rather than
  the same search being asked the same question again.
* **One rethink per turn**, so a conversation cannot spend itself thinking.
* It costs nothing when nothing loops. `explore=0` turns it off.

**Lives in** `radixnet/dialogue.py::backtrack`, `go/radixnet/dialogue.go`

---

### D-068 — Where it goes round is **learned into the graph**, like where texts end

**Status** Research claim · 2026-09-14 · **Layer** structure

**Context** D-062 hands over to a second procedure on detecting a loop, and that
procedure runs again from scratch every time the same loop comes round. A trait
that has to be re-derived at every turn has not been learned. `START` and `END`
are not re-derived: the graph *knows* where texts begin and end, because it was
taught by what it observed.

**Decision** A third sentinel, **`BACK`**, beside `START` and `END`, and an edge
`p -> BACK` meaning *walks that get to `p` go round*. It is an ordinary edge with
an ordinary weight and counter, competing for `p`'s probability mass like any
other child — but it is taught by **experience** rather than by observation,
because no corpus says where a walk loops. Every rethink teaches three edges at
once (`observe_back`): the hand-over itself, a *penalty* on the step it was about
to loop through, and a *reward* on the step it took instead. Where it goes round,
and what to do instead.

The search then consults it everywhere (`search.onward`): when `BACK` is the
cheapest child of a node, the model's most likely next step there is to stop, so
the branch offers nothing and the walk goes on with its others. **That answers
Q-1**: the hand-over is no longer a property of conversation. `predict`,
`generate`, the agent's loop and every other walk get it, because it is a
property of the graph.

**Alternatives rejected**
* *A learned cost with no sentinel* (a per-node "loops here" number) — same
  effect, but it would be a second kind of learned quantity bolted beside the
  edges, learned by its own rule and saved by its own code. An edge is the thing
  this system already knows how to learn, invert, reward, save and port.
* *Learning only into the existing weights* — the model would drift away from its
  loops without ever *knowing* about them, and nothing could be inspected, shown
  in a transcript or held to parity. Both were taken in the end: the sentinel
  says where, the weights say what to do instead.

**Consequences**
* **A conversation changes the model.** Two runs of the same conversation differ,
  because the first taught it something; determinism now means *a model in the
  same state says the same thing*. `learn=False` (`--no-learn`) keeps a
  conversation read-only, and the CLI, which cannot keep the model in memory,
  says so: *it learned to hand over at N node(s); --save writes that into the
  model*.
* The model file is **format 3**. Older files gain an unvisited `BACK` on load
  and their node ids shift by one; the sentinel takes `START`'s activation
  parameters so both kinds of model load unchanged.
* `BACK`'s state is *fixed* rather than drawn, so adding a third sentinel moved
  no random stream: every seeded model that existed still predicts exactly what
  it did.
* A node taught to hand over has two children where it had one, so compression
  stops merging it. That is correct — a loop point is a junction — but it means
  learning this trait slightly reduces compression.
* The rate is a research question, not a settled number: the sine model hands
  over after ~2 experiences at a node, the count model after ~3. Nothing decays
  them yet, so a node taught in error stays taught until something retrains it.

**Lives in** `radixnet/graph.py::observe_back`, `radixnet/search.py::onward`,
`radixnet/dialogue.py::teach_back`, `go/radixnet/graph.go`, `go/radixnet/search.go`

---

### D-063 — The metacognition is a record, not a mood

**Status** Accepted · 2026-09-14 (`414bd1d`) · **Layer** observability

**Decision** Every turn that thought twice carries a `Rethink`: what it caught
itself saying, what it kept, how many times it backed up, how many paths it
weighed, and whether it found a way on. Both CLIs and both tabs render it in a
line — *caught itself saying "ha" twice; kept "ha " and found another way on in 3
path(s)*.

**Rationale** A system that claims to reflect on itself has to be able to show
the reflection. If the only evidence of metacognition is better output, the claim
is unfalsifiable.

**Consequences** The parity tests hold both implementations to the same
transcript **and the same records**, exploring and not — so the reflection is part
of the cross-language contract, not a presentation detail.

**Lives in** `radixnet/dialogue.py::Rethink`

---

# Part XIII — Counters

### D-064 — Every counter is a two-digit odometer that wraps at 10^15

**Status** Accepted · 2026-09-14 (`2751287`) · **Layer** platform

**Context** Traversals, visit counts, epochs, trained characters, 2NRL runs,
feedback passes and the internal version stamps only ever count up. Each would
eventually leave the integer holding it — `int64` in Go, and *long before that*
the 53-bit mantissa of the JSON number carrying it through a model file, the HTTP
API and a JavaScript `Number`.

**Decision** No unbounded integers. Each counter is

$$\text{total} = \text{resets} \times \text{LIMIT} + \text{value}, \qquad 0 \le \text{value} < \text{LIMIT}$$

The count rises as before; on reaching the limit it is set back to 0 and `resets`
goes up by one. Nothing is lost — the exact number of events is still there, split
over two numbers that each stay small.

**Why 10^15** Exactly representable as a `float64` (< 2^53), so a counter
survives a model file, a JSON response and a JavaScript `Number` unchanged; and
four orders of magnitude under the `int64` maximum, so a whole epoch of counting
can land on a counter before the next wrap. The resets wrap at the same limit, so
the odometer comes full circle after 10^30 events.

**The design note the commit makes, and it is the right one:** *cycles are a
feature here too.* The same principle as D-001, applied to arithmetic.

**Consequences**
* **Wrapping never happens in a counting loop.** The loops add to a plain integer
  (in Go from one goroutine per text, plain or atomic — D-040); a *carry sweep* at
  the end of every epoch and before every save moves whatever crossed the limit
  into the resets. `graph.traversals` counts every increment made and so bounds
  each counter, which lets the sweep return after one comparison until a counter
  can actually have wrapped.
* The weight function, the shares, the loss and every ranking are computed from
  the **exact totals**, so a wrapped model behaves exactly as one that counted
  forever. Split carries a node's resets; merge keeps the larger exact count.
* Model files are `format_version 2`; format 1 files load unchanged with their
  counts carried on the way in, and `_resets` fields are written only once
  something has actually wrapped.
* A cached version stamp is still compared with `==` and stays exact across a
  reset, which is why the counter is a value type rather than a plain integer.

**Lives in** `radixnet/counter.py`, `go/radixnet/counter.go`

---

# Part XIV — Memory and the closing of the Go gap

### D-065 — Bounded memory: a sequencer, inflight slots, and a soft heap limit

**Status** Accepted · 2026-09-12 (`5a25799`) · **Layer** performance

**Context** D-043's streaming kept the *corpus* out of memory, but the Go server
was still being OOM-killed on large archives, for two independent reasons.

**Decision**

* **The reader outran the counting.** A chunk now holds one of `--inflight` slots
  (two per CPU by default) *from the moment it is read until the sequencer has
  applied it* — a finished chunk waiting its turn costs the same memory as one
  being counted. A **sequencer** restores corpus order by `(part, chunk)`, so the
  reader never waits for a chunk it spawned, and every part of an archive can be
  read at once with `--parallel-parts`. Parallel parts cannot share one pool (the
  slots would all be held by parts the sequencer cannot reach yet), so each open
  part gets its own budget, only a window of parts is open, and permits are taken
  in part order so the oldest part can always make progress.
* **The collector grows the heap to twice the live graph before it runs**, which
  on a container with a hard limit is a kill. Every process now sets a soft memory
  limit at 80 % of its cgroup limit or of available memory (`--memlimit`,
  `GOMEMLIMIT` wins).

**Measured** 39 MB corpus: **393 MB, down from 1.7 GB** (414 MB with parallel
parts, down from 3.2 GB). A 113 MB corpus inside a 1 GiB cgroup was **killed after
26.9 s; it now finishes in 32.5 s at 908 MiB.**

**Also from the profile:** adjacency is two parallel slices with a map index only
past degree 16 (*a Go map per node cost more than the rest of the graph*); models
are encoded straight into their file and decoded straight out of it rather than
buffered whole; the sliding window compacts in place; read buffers are pooled.

**Consequences** `/api/status` reports `heap_bytes`, `heap_sys_bytes` and
`memory_limit_bytes`, and the status bar shows heap against the limit — the
constraint is visible rather than discovered by a kill.

**Lives in** `go/radixnet/source.go`, `go/radixnet/model.go`, `go/server/service.go`

---

### D-066 — Close the Go gap almost completely, and say plainly what is left

**Status** Accepted · 2026-09-14 (`a34b822`, `737cd06`) · **Layer** platform ·
**Supersedes** the Python-only split in D-044

**Decision** Port everything, and record what is *not* ported together with
whether that is deliberate or merely undone. What remains Python-only:

| Not ported | Why |
|---|---|
| `schedule.py` (learning-rate preview) | **Deliberate.** Schedules are a RadixNet feature; the count model ignores learning rates (D-021), so a port would be a calculator nothing calls. |
| Whisper transcription | **Deliberate.** `faster-whisper` / `openai-whisper` are packages, not code to translate. |
| Image encoding **parity** | **Deliberate and documented.** Python's thumbnail resamples with Pillow's Lanczos, Go's with a box filter, *because reproducing Pillow's coefficients without Pillow to check against would be a guess rather than a port.* The **format** is shared — each side reads, trains on and decodes the other's — and `DescribeVision` says so in its own output. |

**Three things had to be written out rather than translated**, because the
standard libraries differ where it matters: sorted JSON rendering matching
`json.dumps` byte for byte (D-056); an HTML tokeniser (Go has no `html.parser`),
with raw-text elements taken whole so a `<` inside a script is not a tag; and a
recursive-descent arithmetic parser (Go has no `eval`) **keeping Python's
semantics where they differ** — `/` is true division, `//` floors, `%` takes the
sign of the divisor, `int op int` stays an `int`, `round` goes to even, and a
result prints as `str()` prints it: `29.0`, not `29`.

Speech **is** at exact parity, and had to be: the utterance token is a digest of
the waveform text and *the token is in the text the model trains on*, so two
encoders that disagree would teach two different things from one recording. That
required carrying `float32` through the sample pipeline (Python's `array("f")`
rounds every stored value) and writing out BLAKE2b, which Go's standard library
lacks.

**Consequences** The frontend hides no tab from the Go engine any more, and
D-044's "Python-only endpoints answer 404" now covers a single endpoint. The
honest accounting of *deliberate* versus *undone* is itself the decision: a gap
recorded with its reason is a design statement, an unrecorded one is debt.

**Lives in** `go/`, `DESIGN.md`

---

# Part XV — The traversal

### D-069 — What a search *looks for* is an option, separate from how it looks

**Status** Accepted · 2026-09-20 (`dc154c8`) · **Layer** search

**Context** Every reward the system hands out is also, implicitly, a
navigational instruction. The count / reward model's probability carries
`exp(reward_scale * reward)` (D-026), so once a path is rewarded the search
follows it: prediction, generation, conversation, the tutor's completions, the
agent's attempts. That is the intended behaviour and it is also the whole of
the behaviour — a network with a thousand corrections and a handful of thumbs
up still navigates by the handful, because the corrections only enter as the
same number with a minus sign in front, competing on the rewards' terms.

Rewards and penalties are not symmetric evidence, though. A reward says *this
was good once*; a penalty says *this was wrong, and here is the correction*.
The first is an invitation to repeat a success; the second is a boundary. A
walk that respects every boundary it has been taught is not the walk that
chases every reward it has been given, and there was no way to ask for it.

**Decision** The cost function a search reads the graph through is an option,
the **traversal**, beside the existing **mode**:

* `reward` (the default) — the model's own distribution, rewards and all.
  Exactly the previous behaviour, at exactly the previous cost: the option
  resolves to `None` and the searches call `graph.child_costs` as before.
* `punishment` — the rewards leave the score altogether and the penalties
  price every step, so the cheapest path is the one that accumulated the
  **least punishment**.

**What was rejected**

* **A fifth mode** (`dijkstra | kbest | beam | sample | punishment`). The
  wrong axis: it would have had to be written once for Dijkstra, once for each
  beam, once for the sampler and again for all four phase searches, and it
  would have made "least punished" and "top-K / bottom-K" mutually exclusive
  when they are orthogonal. Every search already reads the graph through one
  funnel — `[(child, edge, cost)]` for a node — so the option replaces the
  funnel instead. No search changed.
* **A second graph** holding the punishments. That already exists and is the
  negative network (D-045); this is about *navigating the model you have*, not
  about modelling failure.
* **Flipping the sign of the rewards.** `invert` (D-009) does that, and it is
  not the same thing: it makes the punished path *attractive*, whereas this
  makes it expensive and leaves everything else alone.
* **A lexicographic cost** (total punishment first, the model's own cost as a
  tie-break). Exact and tempting, but it would have made the cost a tuple
  through every heap in the package, and it forbids the trade-off that
  `penalty_scale` exists to express. A blended score with a knob says more.

**How the two currencies are separated** One hook,
`RadixCyclicGraph.child_evidence`, splits an edge's evidence into **merit** —
what speaks for the step with every reward taken out of it — and **penalty**
`>= 0`. Each kind implements it in its own currency: the count / reward model
and the phase model split `edge_reward` in half, the negative network weighs
net blame against cleared text, and the sine model reads the negative part of
`w · f_p · f_c`, because 2NRL trains a failure in and then inverts it, so what
a punishment leaves behind *is* a negative score on that path. A judged path
context — a verdict filed against the caller that reached the edge rather than
against the edge — splits the same way.

**Consequences**
* `score = merit_scale * merit − penalty_scale * penalty` and
  `cost = -log softmax(score)`, so costs stay `>= 0`, Dijkstra stays a true
  shortest path, `exp(-cost)` is still a probability and the two traversals'
  numbers are comparable. `merit_scale = 0` is the pure form.
* Every mode of every kind gains the traversal at once, in both languages, and
  the parity suite requires the same least-punished paths at the same costs.
* On the sine model the two traversals coincide at the default scales, because
  there the punishment *is* the negative score; they part company as soon as
  `penalty_scale` is raised. That is honest rather than convenient: a model
  that keeps no separate ledger of its punishments cannot be made to pretend
  it does.
* On the negative network the option reverses the network's purpose — the
  least blamed way through the failures rather than the likeliest one — which
  is a use nobody had before and falls out for free.
* **The cost:** a second cost cache per graph while a punishment search runs,
  and one more thing to choose. The default is unchanged behaviour, so nobody
  who does not want it pays for it.

**Lives in** `radixnet/penalty.py`, `radixnet/graph.py`, `go/radixnet/penalty.go`

### D-070 — A setting of the network gets one home and one value, wherever it is edited

**Status** Accepted · 2026-09-20 · **Layer** frontend · **Extended by** D-079 (the tab became two: the
browser's Settings and the model's Model settings; `useSiteSettings` generalises `useNetworkSettings`)

**Context** D-069 gave every search a traversal, and the frontend had nowhere
to put it. It went on the two tabs that use it, Predict and Generate, as a copy
each. That was already wrong in a way worth naming: the traversal is a property
of *the network's behaviour*, not an option of one prediction, and two copies of
one setting is two answers to the same question. The frontend had the same shape
elsewhere - the count model's weight function was a fieldset on the **Train**
tab, which is a training form, and it was missing `path_scale`.

**Decision** A **Network settings** tab: the settings of the network itself, as
opposed to the options of one run. The traversal, the score function of whichever
kind is active, and the encoder / decoder.

A setting several panels use lives **once** and is read through a provider
(`useNetworkSettings`), so the Network settings, Predict and Generate tabs show
one control in three places rather than three controls. `useStoredState`, which
every other field uses, cannot do this: two mounted components under one name
share the stored value and not the state, and every panel here stays mounted
while hidden, so two copies would drift apart within a session and only agree
again after a reload.

The score function left the Train tab for the same reason - two forms over one
server-side value, whichever is not touched showing what the function used to be.

**What the encoder card does *not* do** The window (3) is the one number on the
page that looks like a setting and is not: the graph's labels, the split and
merge rules, the model file and the Go port all assume it, so a model trained at
one window could not be read at another. `GET /api/encoding` reports
`configurable: false` and the card says why, then spends its space making the
encoding **visible** instead - `POST /api/encoding/preview` runs a text through
the encoder, back through the decoder, and through the graph's own node labels,
where a label longer than the window is a merged radix chain. Saying "this is
fixed, and here is what it does" is worth more than a disabled input.

**Alternatives rejected**
* **A modal or a header menu.** Settings that take a paragraph each to explain
  are not a menu; the tab strip already is the app's navigation.
* **Leaving the traversal only on Predict and Generate.** It would have kept two
  values for one thing and left the score function homeless.
* **Making the window settable** by re-encoding on change. It is a model-format
  change, not a setting, and pretending otherwise would break a saved file and
  the cross-language contract at once.
* **Mirroring the negative network's blame function here too.** It belongs
  beside the failures it weighs; the card links to the Negative tab instead of
  opening a second door onto a third value.

**Consequences** The frontend now distinguishes the two kinds of control it has
always had and never separated: what the network *is* (this tab, saved with the
model or remembered in the browser) and what one run *asks for* (the action
tabs). The traversal is the first setting to be shared rather than copied, and
`useNetworkSettings` is the pattern for the next one.

**Lives in** `frontend/src/components/NetworkSettingsPanel.jsx`,
`frontend/src/hooks/useNetworkSettings.jsx`, `radixnet/api.py`, `go/server/`
### D-075 — A walk can be ranked by what went **wrong** on it, and blame is not for sale

**Status** Accepted · 2026-09-20 · **Layer** search · **Extends** D-008, D-058 ·
**Specified in** `SPEC-LeastPunished.md`

**Context** Every path this model has taken was chosen by one number, and the
rewards and the penalties land in the same accumulator (D-022): a step rewarded
five times and punished once carries `+4` and is indistinguishable from a step
rewarded four times and never punished. That is right for *likelihood* and wrong
for a question the system asks constantly and could not express - **which way
through has the least gone wrong on it?**

**Decision** A second traversal, selected per call
(`--traversal least-punished`), that changes what a walk is *ranked by* and
nothing else. A step carries a punishment - the penalty side of its reward, plus
`path_scale · log(1 + incorrect)` of its judged path context. A walk is ranked by
its **worst** step first, by cost only between walks whose worst step ties, and
at every node it may take only the children with the least against them.

**Rationale** The context term is the load-bearing half, and it is deliberately
**asymmetric**: the cost function's path term weighs `correct` against
`incorrect` and is symmetric on purpose (D-058), while this one counts the
failures against nothing. A step that was wrong here once is a step that was
wrong here, and no amount of being right afterwards makes it a step nothing is
held against. Without that asymmetry the traversal is a clipped copy of the cost
order - which is exactly what the first implementation of it turned out to be,
netted away by a reward in the first test written against it.

The **worst step** rather than the sum, because ten small penalties are not one
real failure, and because it is the reading the negative filter's `peak` already
takes (D-047): the two now agree about what a path's blame is.

**Consequences**
* Provably inert where nothing was punished: the first component of the order is
  0 on every path, so the comparison *is* the cost order - same text, same cost,
  same expansions, pinned by a test on both sides.
* Not the default, and should not be: `predict` without an argument should mean
  the model's own estimate of what comes next.
* Measured on a 2M-character corpus with every 7th text punished, it disagrees
  with the ordinary search on **20% of continuations** and expands **15x fewer
  nodes** - because refusing a blamed step at the node prunes the beam. That is
  an effort result, not a quality result; nothing here has been graded.
* It reads numbers the model file already carries, so it costs the format
  nothing and changes nothing it walks.
* **All three implementations have it** - Python, Go and Rust - and both parity
  suites hold the ports to Python's answers under it, punishment included. The
  sine model accepts the argument and refuses anything but `reward`: it keeps no
  record of failure to rank a walk by, and says so rather than ignoring the
  option (the discipline of D-070's one-home rule).
* It is a way of *reading* the model, and nothing reads it that way on the
  model's behalf: the tutor, the conversation, the agent and the guard all still
  walk by cost.

**The honest gap it papers over.** An edge keeps one reward, so its own penalty
*is* netted - only the path contexts remember a failure as a failure. The edge
should learn to keep the two apart; that is a model file change, and it is what
the first term of the punishment is a stand-in for until then.

**Lives in** `radixnet/search.py`, `radixnet/beam.py`, `radixnet/countnet.py`,
`go/radixnet/search.go`, `go/radixnet/beam.go`, `go/radixnet/weights.go`,
`rust/src/search.rs`, `rust/src/beam.rs`, `rust/src/weights.rs`, `SPEC-LeastPunished.md`

---

# Part XVI — The encoding

### D-071 — The encoding is a dial of the model, not a constant of the package

**Status** Accepted · 2026-09-20 · **Layer** representation ·
**Extends** D-006, which stays the default

**Context** D-006 fixed the input at three characters with stride 1 and gave the
reason: the shared character is a *pivot*, and three is the smallest window that
gives a pivot with context either side. That argument says what the **default**
should be. It does not say the number should be a constant - and in both
implementations it was one: `WINDOW = 3` / `Window = 3`, read directly by the
graph, the models, the beams, the diff and the loader, with the overlap beside
it. Anyone wanting to ask "what does this corpus look like in fives?" had to
edit two constants, and nothing in a model file said how to read its labels.

**Decision** The encoding becomes a value - `Encoding(unit, n, stride)` - owned
by the graph, fixed when the graph is created, written into the model file and
read back from it, **in every implementation**. Three dials:

| dial | what it is | the default |
|---|---|---|
| `Unit` | what one position of a text is: a character, or a whitespace word | `char` |
| `N` | how many units one gram holds - the *n* of the n-gram | 3 |
| `Stride` | how far apart consecutive grams start | 1 |

`Stride` is the dial that makes the other two useful. At 1 the grams slide and
overlap by `N - 1`, which is D-006's pivot generalised. At `N` they do not
overlap at all, which is *tokenisation*: `char:4:4` cuts text into groups of
four letters, and the graph becomes a chain of groups that meet only at their
ends. `word:2:1` is the word bigram, `word:3:1` the word trigram.

Everything the graph measures is now measured in **units**, not characters: a
node's label length, the offset of a gram inside a label, the length of a
prediction, the spans of the correction diff. Under the default encoding a unit
*is* a character, so every one of those quantities is what it always was - which
is why the parity tests (`tests/test_go_parity.py`) still pass unchanged.

**Alternatives rejected**
* **A package-level variable instead of a field.** One process, one encoding -
  and a model loaded from a file could silently disagree with it. The encoding
  belongs to the graph because the graph's labels are written in it.
* **Keeping the label a `[]rune` and special-casing words.** Words are not
  characters of a different width; the split, the merge and the index all walk
  *positions*. A `Units` view (an index of byte offsets, sliced in O(1)) makes
  one code path serve both.
* **A learned sub-word vocabulary.** Still rejected, for D-006's reason: it
  needs a corpus before training can start, and freezes what the model can read.
  `word` is not a vocabulary - an unseen word is a new node, exactly as an
  unseen trigram is.
* **Writing the encoding into every file.** A file that says `char:3:1` is a
  file the Python loader would have to be taught to ignore. It is written only
  when it is *not* the default, so an ordinary model file is byte for byte what
  it always was.

**Consequences**
* **D-039's bit-identical interchange now covers every encoding.** Both sides
  write the same `encoding` block and read each other's, and
  `TestGoEncodingParity` holds them to the same graph, the same file and the
  same prediction under nine of them. A word model trained in Python continues
  in Go and back.
* **The dial reaches all four Python kinds.** RadixNet, the count model, the
  negative network and the resonant model share one graph, so none of them
  could have it alone. The sine model trains on word bigrams because the graph
  it trains on does.
* **The Rust port carries it too**, on its own branch, and paid the most for
  it: its index key was three code points packed into a `u64`, which four
  characters do not fit and a word does not fit at all. It becomes an enum -
  packed for character grams of up to three, the text itself otherwise - and
  the port's own tests pin the structure Python and Go build under nine
  encodings, since it writes no model file to compare.
* **The units leak into the vocabulary of the API.** `--length`, `--max-length`
  and `Score.chars` count units, so on a word model they count words. That is
  the honest reading - a "40-character" cap on a model that thinks in words is
  meaningless - but it does mean two models answer the same flag differently.
* **Compression means something different per encoding.** With no overlap there
  is no shared context for two nodes to be merged *through*; a grouping
  encoding compresses only the unary chains its corpus actually repeats.
* **A model cannot change its mind.** The encoding is fixed at creation: every
  label in the graph is written in it. The CLI refuses an encoding flag that
  disagrees with the model it loaded rather than ignoring it.

**Lives in** `radixnet/encoding.py`, `radixnet/graph.py`, `go/radixnet/encoding.go`,
`go/radixnet/graph.go`, `rust/src/encoding.rs`, `DESIGN.md` § 23.1

---

### D-073 — Superseded by the encoding dial: a word n-gram model is this model over an alphabet of words

**Status** Superseded 2026-09-20 by D-069's encoding dial · **Layer** input · **Beside** D-006

**How it was superseded** This decision and D-069's dial reached word n-grams
independently and differently, and one of them had to go. The reasoning below
is unchanged and still right about the *model*: not one structural rule of the
graph mentions a character, so a word n-gram model is this model over a
different alphabet rather than a different model. What is superseded is the
*mechanism*. This decision made words a **kind** — `WordNGramNet`, the format
`radixnet-word`, and a vocabulary mapping each word to one code point so a word
would fit the packed trigram. The dial makes them a **unit**, one of three
settings on the ordinary count model, and needs no vocabulary at all: a gram is
text, so the alphabet a graph has read is whatever its grams are made of.

The dial is strictly more general — any n, any stride, and groups of letters as
well as words — and it costs less: no second format, no second kind, no
vocabulary to keep in corpus order across three implementations. What was kept
from here is the part the dial had not built: `units` following the encoding
rather than the model class, every length and score counted in those units, and
the alphabet listing (`radixnet words`, `GET /api/words`, the Words tab) rebuilt
on the gram index. What was dropped is `radixnet/wordnet.py`, its Go and Rust
twins, `--kind word` and the `radixnet-word` format. The costs stated at the end
of this decision — normalised whitespace, and two unread words being one symbol
— are the dial's costs too, unchanged.

**Context** *"What about word n-grams?"* The obvious reading of that question is
that it asks for a second model. It does not. D-006 decides what a **symbol**
is - characters, window 3, stride 1 - and every structural rule in the graph is
stated in terms of that window and nothing else: a label is a sequence of
symbols, an edge exists where the last `WINDOW - 1` symbols of one label are the
first `WINDOW - 1` of another, the index maps a `WINDOW`-symbol key to
`(node, offset)`, compression merges a unary chain at that seam. Not one of
those rules mentions a character.

**Decision** Add one model kind, `word`, which is the count / reward model
(D-021, D-022) with the encoder and the decoder replaced: **a word is a
symbol**, carried as one code point (`id 0 -> U+0100` is `<unk>`, `id i ->
U+0100 + i`, the surrogate block skipped, 1 111 808 words). Text becomes symbols
on the way in and symbols become text on the way out; between those two points
the graph, the weight function, the counters, the paths, both traversals and the
search are untouched, in all three implementations. The file is a format of its
own, `radixnet-word`, carrying `units: "words"` and the `vocabulary` in id
order.

**Rationale** The alternative - making the graph generic over a sequence of
symbols - is what a type system would prefer, and it changes every structural
routine in three implementations to buy what a code-point alphabet already
buys. The current design *already* proves the graph is alphabet-agnostic; this
decision only names what was always true. The vocabulary grows as training reads
new words and is never frozen, pruned or learned, which is the part of D-006
that matters: there is no tokeniser, no merge table and no training run before
the training run. Tokenising is `text.split()` and nothing else, because every
refinement of that rule is a step towards a vocabulary that has to be designed,
versioned and defended.

**What it costs, stated plainly**
* **Whitespace is normalised.** `decode(encode(t))` joins the words with single
  spaces, so a corpus whose whitespace carries meaning - source code, base64, a
  waveform (D-036) - must stay on the character model. A word model cannot eat
  the corpora the character model was chosen to be able to eat, and that is why
  both kinds stay.
* **Two unread words are one symbol.** At prediction and scoring time an unread
  word is `<unk>`, so `"the qux sat"` and `"the quux sat"` score identically. It
  is visible in `unknown_transitions`, and it is why the character model remains
  the default.
* **N stays 3.** The overlap is two words and the pivot is the middle one -
  D-006's argument word for word. Word *bigrams* need `WINDOW = 2`, which
  collapses the pivot and the context that D-006 rejected collapsing, and in
  Rust additionally needs the packed-trigram representation rewritten. Out of
  scope; nothing here prevents it later.

**Consequences**
* Everything counted in symbols is counted in **words**: `--length 6` emits six
  words, `--max-length` caps words, `Score.chars` counts words and `per_char` is
  per word. `stats()["units"]` says which, because a number whose unit depends
  on the model is a number that will be read wrong; the CLI's score columns, the
  frontend's length fields and the API's status carry it too.
* Compression does to word chains what it already did to character chains: a
  repeated phrase becomes **one node whose label is that phrase**
  (`'sat on the mat'`), and the search walks phrases.
* Interchange is the contract the count model has: Python, Go and Rust read and
  write the file, and the parity tests require the same structure, the same
  counts, the same vocabulary **in the same order** and the same predictions.
  The Rust word document is Python's byte for byte, as the count one is.
* Three small seams were opened in shared code to make the alphabet a model's
  own business rather than the search's: `graph.text_of` / `graph.symbols_of`
  (a label as text, and back), a model-level `units`, and `_whole_text` no
  longer re-joining a prefix the search already joined.

**Lives in** `radixnet/wordnet.py`, `radixnet/encoding.py` (the alphabet),
`go/radixnet/words.go`, `rust/src/words.rs`, `SPEC-WordNGrams.md`,
`Makefile` (`word-*`)

---

# Part XVII — A third implementation

### D-072 — A third implementation, to price the language rather than the model

**Status** Accepted · 2026-09-20 · **Layer** platform · **Beside** D-038

**Context** D-038 ported the count model to Go because pure Python topped out
near 100k transitions/s. Go now runs the same model at ~6M. How much of what is
left is the model, and how much is the runtime? Nothing in the repository could
answer that, because nothing had ever run this model twice.

**Decision** Port the count model to Rust (`rust/`), with no dependencies, and
build the cross-language benchmark (`bench/`) that runs it against Go **over one
corpus** and refuses to report a timing until the two agree on the graph, the
transitions, the loss, the expansions and the prediction - to the bit, on the
cost of the path.

**Rationale** A speed comparison between two programs that computed different
things is not a comparison, and a port that quietly drifts is the normal failure
mode of having three implementations (D-038's "two implementations of one model,
which must stay in step"). Making the parity check a precondition of the
benchmark, rather than a separate test someone remembers to run, is the whole
design of `bench/compare.py`.

**What it measured** (4-core Xeon, 2M characters, 3 epochs; `bench/RESULTS.md`)

| | Go, one worker | Rust, one worker | Go, all cores | Rust, all cores |
|---|--:|--:|--:|--:|
| training | 6.6M transitions/s | **14.3M** | 7.4M | **20.6M** |
| prediction | 4.2k/s | **16.1k** | 3.8k | **16.1k** |

2.2-2.8x at counting, 3.8-6.2x at predicting (the wide end of the second range is
the least-punished traversal, where the search is small and the constant factors
are most of it). Three representation choices carry
most of it and none of them is algorithmic: a trigram is a packed `u64` rather
than a fresh string, a node's children are read into a buffer the search reuses
rather than a fresh slice per expansion, and the trigram index hashes with
`FxHasher`. They are listed in `rust/README.md` because a reader who takes the
table for "Rust is 4x faster than Go" has been misled by it.

**Consequences**
* Three implementations to keep in step, not two. The benchmark is the check for
  the third; `tests/test_go_parity.py` remains the check for the second.
* Go's `--workers 1` now runs the two beams of a prediction in turn rather than
  on two goroutines, so a one-worker row means the same thing on both sides.
* The Rust port reads and writes the `radixnet-count` model file, gzipped or
  not, and `tests/test_rust_parity.py` holds it to Python's: the same structure,
  counts, rewards, window, RNG state, judged paths and node ratios, the same
  predictions, generated texts and scores, and each side continuing the other's
  file. Its graph document is Python's **byte for byte** but for the `version`
  cache stamp - which the Go port's is not, because Go renders floats and orders
  keys its own way.
* The port carries **all three traversals** (D-075, and the punishment traversal
  of D-069 in `rust/src/penalty.rs`) and the **encoding dial** (D-073), so a
  traversal or a dial added to one implementation is now added to three.
* The dial cost the port one of the three representations this decision measured
  it on. A trigram was three code points packed into a `u64` - `Copy`, hashable,
  no allocation - and a gram of any n over any unit is arbitrary text, so the
  index is keyed by the gram as the other two key it. That is the honest trade:
  the packing was only ever possible because the window was fixed at three
  characters, and `bench/RESULTS.md` says what giving it up cost.
* The port serves the frontend: `rust/src/http.rs` is HTTP/1.1 written out over
  `TcpListener` and `rust/src/service.rs` answers the same JSON contract the
  Python and Go servers answer, so `frontend/dist` runs against
  `radixnet serve` unmodified. What a Rust server cannot fill, the frontend
  hides on `engine == "rust"`, and what it will not serve says so with a 400.
* What the port still does **not** have: the negative network, the tutors and
  the other teaching loops, the agent and its tools, images and speech, and MCP.
  Undone, not deliberate. **One gap is deliberate**: the LLM clients need HTTPS,
  and a crate with no dependencies cannot speak it. Porting them is not a task,
  it is a proposal to drop the no-dependency rule above - which would also drop
  what that rule buys, a `Cargo.lock` with nothing in it but this crate.
* Go keeps its racy-by-design counting (D-038); the comparison uses `--exact` on
  both sides, because a benchmark of a deliberate data race measures the race.

* **Update, 2026-09-23.** The port now carries the whole package - every model
  kind, the negative network, the teaching loops, the LLM clients, the tools,
  code generation, the agent, images and speech, MCP and the WebDriver browser -
  and the "one gap is deliberate" above no
  longer holds: D-076 has the LLM clients speak HTTPS through the system `curl`
  without taking a dependency. D-077 records how the rest was laid out.

**Lives in** `rust/`, `bench/`, `Makefile` (the `rust-*` targets, `bench-compare`)

---

### D-074 — A port is finished when a client cannot tell which one answered

**Status** Accepted · 2026-09-20 · **Layer** platform · **Beside** D-038, D-072

**Context** "Is everything ported?" had no answer anyone could check. Each
implementation's README listed what it had, in prose, written when it was
written. Two gaps found by walking the three surfaces rather than the prose:
Go had every HTTP route and every CLI command but `mcp` and
`speech transcribe`; the Rust server had thirty routes that each answered
*something*, but refused to switch model kind, read only the first of the three
ways the contract lets a client name its texts, answered a job 200 where the
other two answer 202, and saved whichever kind was active to the file of the
kind the binary had started on.

**Decision** Parity is measured against the **surfaces**, not the prose: the
route list, the CLI command list, and the fields and status code of each route.
A gap is either closed or named in the decision that owns it, with which of the
two kinds it is:

* **undone** — portable, not yet done (Rust's negative network, the teaching
  loops, the agent);
* **deliberate** — cannot or should not be ported, with the reason (the LLM
  clients need HTTPS, which D-072's no-dependency rule rules out; learning-rate
  schedules belong to the sine-activation model, which neither port has, so
  `radixnet schedule` is a command about a model Go and Rust do not run).

**Rationale** A port that answers every route with *something* looks finished
from the outside and is not. The three surfaces are enumerable and can be
diffed in a shell one-liner, which is the only reason the four Rust gaps above
were found at all — every one of them returned a 200 and a plausible document.
The distinction between undone and deliberate is what makes the remaining list
readable: a reader who cannot tell them apart reads every gap as neglect.

**Consequences**
* Go speaks MCP (`go/radixnet/mcp.go`, `radixnet-count mcp`): the same protocol
  revision, tool names and schemas as `radixnet/mcp.py`, and the same answers
  down to the error text — asserted by running both over one message stream.
  `/api/speech/transcribe` stays Python's, because local Whisper is not a thing
  a Go binary carries.
* The Rust server switches kind, parking the model that was running; reads
  `texts` / `text` / `files` (and the `good_*` / `bad_*` twins); answers 202 for
  a job it has started on a worker thread; and saves each kind to its own file.
* The gap lists in `rust/README.md`, `go/README.md` and DESIGN §33 say which
  kind of gap each remaining item is.
* **Update, 2026-09-23.** Rust's *undone* list is empty. What remains is
  deliberate and named in `rust/README.md`: the torch backend, the Stable
  Diffusion encoder and local Whisper (each needs Python packages or a GPU).
  The LLM clients left the deliberate list (D-076), and learning-rate schedules
  are no longer "about a model the ports do not run": the Rust port runs the
  sine-activation model and serves `radixnet schedule` and its preview.

**Lives in** `go/radixnet/mcp.go`, `go/cmd/radixnet-count/mcp.go`,
`rust/src/service.rs`, `rust/src/http.rs`, `Makefile` (`go-mcp`)

---

### D-076 — The Rust port speaks HTTPS through the system `curl`, and still depends on nothing

**Status** Accepted · 2026-09-23 · **Layer** platform · **Beside** D-072, D-074

**Context** D-072 kept the Rust crate free of dependencies and recorded the
price: the LLM clients need HTTPS, a crate with no dependencies cannot speak it,
so "porting them is not a task, it is a proposal to drop the no-dependency
rule". Then the port was asked for *all* of Python's functionality - the tutor,
the chat, the critic, the agent and its browsing tool, every one of which talks
to an LLM or a web page, and most of them over TLS.

**Decision** `rust/src/fetch.rs` is the port's one HTTP client. Plain `http://`
- a local Ollama, a transcription server, a test's own listener - is written
out over `TcpStream` and never leaves the process. `https://` is handed to the
system's `curl`, run as a child process with the request body on its stdin and
the answer on its stdout. The crate's `Cargo.lock` still holds nothing but the
crate.

**Rationale** A TLS stack is not a thing to write out the way the gzip inflate,
the Mersenne Twister and the HTTP server were: it is security code, and a
hand-rolled one would be worse than no HTTPS at all. Taking a TLS crate would
end D-072's rule for the sake of one feature and pull a dependency tree into a
crate whose point is that it has none. `curl` is already on every machine the
port runs on, already honours `HTTPS_PROXY`, `NO_PROXY` and `CURL_CA_BUNDLE`
the way every other tool there does, and is the client an operator already
trusts. The cost is a process per HTTPS request, which is noise next to an LLM
completion.

**Consequences**
* The LLM clients, the teaching loops and browsing are portable after all:
  D-072's "one gap is deliberate" no longer holds, and D-074's list of
  deliberate gaps loses its first entry.
* HTTPS needs `curl` on the `PATH`; without it an `https://` request fails
  with an error that says so, and plain HTTP is unaffected.
* An API key never goes on a command line (where the process list would show
  it): headers reach `curl` the way `fetch.rs` documents.
* Redirects are followed only when asked, because the browsing tool checks
  every hop against the private-address rule itself.

**Lives in** `rust/src/fetch.rs`

---

### D-077 — The rest of Python, ported by area: one module per area, a tab per route

**Status** Accepted · 2026-09-23 · **Layer** platform · **Beside** D-072, D-074, D-076

**Context** Everything the Rust port lacked was asked for at once: the negative
network's filter, the conversation, the LLM clients, the tutor and chat, the
critic, evolve, checkpoints, ZIP corpora, the tools and the sandbox, images and
speech, the agent and MCP, and two whole model kinds. Written as one piece it
would have been months of serial work through three shared files - the CLI's
command match, the server's route table and its state.

**Decision** The crate is laid out **by area**. `src/cli.rs` holds the command
line in the library, with a table sending each command to the module that
answers it; every area is a module that owns its commands (`cli(ctx)`), its
routes (`routes(server)`) and its server state, wired into the service once.
Each area has its own parity suite (`tests/test_rust_parity_<area>.py`, on
`tests/rust_harness.py`). The areas were then ported in parallel and merged,
and what only showed at the seams was fixed there:

* **One loader for the server's negative network**, which takes the model's lock
  before the negative network's and never the other way round - three areas had
  each written their own, and the first one deadlocked when the filter was the
  first thing asked of it.
* **Checkpoints through an epoch hook the model carries**, run at the end of
  every epoch by every kind's loop, instead of training an epoch at a time -
  which kept the count model exact and would have broken the sine model's
  learning-rate schedule.
* **Every learning loop through `crate::kinds`**, because the loops were ported
  against the count model and the sine and phase models arrived later.

The frontend shows a tab against the Rust server when **the route it needs is in
`/api/status`** (`routes`), instead of from a fixed list of what Rust "has".

**Rationale** Parallel work only pays when it does not collide, and the
collisions were all in three files; taking those apart first made every area a
set of files nobody else touched. A tab gated on the route list cannot drift
from the server: it appears the day the route does.

**Consequences**
* A new area is a module, one row in `cli::COMMANDS`, one `routes` call and one
  parity suite.
* `/api/status` carries `routes`; the frontend reads it only for the Rust engine.
* Rust jobs end `done`, `stopped` or `error`, the words the other two servers
  use (the Predict tab waited on `done`).

**Lives in** `rust/src/cli.rs`, `rust/src/service.rs` (`build`), `rust/src/duo.rs`
(`Service::ensure_negative`), `rust/src/checkpoint.rs`, `rust/src/kinds.rs`,
`frontend/src/App.jsx`, `tests/rust_harness.py`

---

# Part XVIII — More ways to search and to train

### D-078 — Every search and training method is off by default, and none of them draws a random number

**Status** Accepted · 2026-09-24 · **Layer** search, training · **Beside** D-008, D-024, D-039, D-074

**Context** The search had two knobs a language-model user reaches for first and did not find - a way to keep a
sampled walk off its long tail, and a beam that did not hand back five spellings of one sentence - and training
had one shape: every text, in corpus order, every epoch, until the epochs ran out. A model trained on a second
corpus forgot the first; nothing let a run start easy; nothing stopped a run that had stopped learning. Adding
them to one implementation is an afternoon. Adding them to three that are held to the same graph and the same file
byte for byte (D-039, D-074) is a contract problem first.

**Decision** Sampling filters (`top_k`, `top_p`, `min_p`), a diverse beam (`diversity`), and a training plan (an
`order`, a `curriculum`, a `replay` buffer kept with the model, early stopping by `patience` and `min_delta`), in
Python, Go and Rust, under one written contract (`SPEC-SearchAndTraining.md`). Three rules make it a contract
rather than three features:

* **Off means absent.** Every setting has an *off* value that is the default, and off changes nothing - the draw,
  the beam, the pass and the file are what they were. The frontend and the servers send and store a setting only
  when it is on.
* **Nothing new draws a random number.** The shuffle and the replay buffer order texts by SplitMix64 keys of the
  model's seed; the filters keep the cheapest option and draw once per step however many survive. The model's own
  generator - and so every weight, every sample, every seeded walk - is where it would have been, in every port.
* **Rules, not recipes.** Every rounding (`ceil` for the curriculum, half-up for the rehearsal), every order
  (`top_k` then `min_p` then `top_p`, ties by position) and every tie (`(priority, g)` in the buffer) is written
  down, because "about the same" is not a parity test.

**Alternatives rejected**
* **A frontier penalty for the diverse beam** - charging a partial path for ending in a node another kept path
  ends in. Implemented first: on real graphs it changed nothing (compressed endings differ node by node), and
  tuned harder it pruned the best path. Picking the K from a pool of *finished* paths by maximal marginal
  relevance cannot lose the best path - it is always the first pick - and trades exactly what it says it trades.
* **Reservoir sampling for the buffer.** It needs a random number per offer, which would move the model's
  generator; bottom-k by a hash is as uniform, deterministic, and mergeable.
* **Rehearsing punished texts.** A thumbs down, 2NRL's negative phase or the agent's punishment trains *against*
  a text; putting it in a buffer that later training rehearses *for* would teach the failure back. A pass stamped
  with a feedback phase is never planned and never touches the buffer.
* **Letting a partial epoch count toward the stop.** Its loss is over fewer (and, shortest-first, easier) texts;
  comparing it with a full epoch would stop a curriculum just as it got hard. The cost, stated in the spec: with the
  linear curriculum, early stopping watches only the epochs after the curriculum reaches every text.

**Consequences** A model file can now carry a `replay` block - only when it keeps a buffer, so every file written
before is still what it was. Fixing the units on the way (a prefix's lead, the resonant model's `chars` and
`trained_chars`, all counted in characters where the rest counts in units) was a precondition: a curriculum by
length is meaningless on a word model if length means characters.

**Lives in** `radixnet/training.py`, `radixnet/search.py`, `radixnet/beam.py`, `go/radixnet/training.go`,
`rust/src/training.rs`, `SPEC-SearchAndTraining.md`

### D-079 — A setting's home is decided by who keeps it: this browser, or the model

**Status** Accepted · 2026-09-24 · **Layer** frontend · **Extends** D-070

**Context** D-070 gave the settings of the network one home, a *Network settings* tab, and one value each. The
search and training methods (D-078) brought six more that shape every search and every run, and the tab mixed two
kinds of setting that behave differently: the traversal was remembered in the browser and stayed put when another
model was loaded, while the score function and the encoding belonged to the model, were saved in its file, and
changed with it. A tab that holds both invites the question "if I load another model, which of these change?" - and
answers it only by trial.

**Decision** Two tabs, split by owner. **Settings** holds what this browser keeps - the traversal, the sampling
filters and the diversity, how a run walks its texts - and says it is never saved with a model. **Model
settings** holds what the model keeps - its kind, its encoding, its size, its replay buffer, the score function -
and gains the one thing that was missing: a form that makes a **new model** in any kind and encoding, since an
encoding is chosen when a model is born (D-071) and a trained model cannot be moved to another. The action tabs
(Predict, Generate, Train) show the browser's settings as the same controls, D-070's one-value-several-doors rule
carried over through `useSiteSettings`; the rules themselves (ranges, what a mode reads, what a request carries)
are pure functions in `src/settings.js` with their own tests.

**Alternatives rejected**
* **Per-tab copies** of the filters and the plan. Two values for one setting, drifting apart within a session
  because every panel stays mounted - the problem D-070 already solved.
* **Saving the search defaults with the model.** They are how *this person* likes to search, not a property of
  the graph; a shared model file should not carry one user's taste for nucleus sampling.
* **One tab with two headings.** It leaves the loaded-another-model question to the reader; a tab boundary
  answers it.

**Consequences** `#network` links still work (they open Model settings). `/api/status` reports the replay buffer
(`{size, texts, seen}` or null) on all three servers so the model's tab can show it, and D-070's paragraph on the
encoder card being read-only is history: the encoding is chosen on the New model card instead.

**Lives in** `frontend/src/components/SettingsPanel.jsx`, `frontend/src/components/ModelSettingsPanel.jsx`,
`frontend/src/hooks/useSiteSettings.jsx`, `frontend/src/settings.js`

---

### D-080 — Thinking is a fourth sentinel that faces both ways, and a thought is a walk that begins there

**Status** Research claim · 2026-09-24 · **Layer** structure · **Extends** D-068

**Context** D-068 taught the graph *where it goes round* by experience: an edge into `BACK`, competing with a
node's real children, and a search that hands over when it wins. What it could not do is think - between
noticing a repeat and backing out of it there was nothing, and a model asked a question about a text had no
words of its own to answer in. The one source of thinking within reach is the local LLM, whose reasoning Ollama
returns beside its answer.

**Decision** A fourth sentinel, **`THINK`**, learned the same way as `BACK` and used the opposite way round as
well. Its in-edges `p -> THINK` are taught by experience (`observe_think`) whenever an *event* at `p` called for a
thought - a rethink, a question asked about a text, a thought questioning itself, a question the LLM asked itself
in its own thinking - and compete for `p`'s probability like `BACK`'s. Its out-edges are where **thoughts
begin**: a thought is trained as a text whose walk starts at `THINK` instead of `START` (`train(origin=THINK)`),
the same structure, counting and compression, so the model learns how thoughts open without a word of them
leaking into what it says. `think()` is one thought and always has a trigger: it teaches where it had to think,
thinks (the search from `THINK`), questions itself where its path crosses a node it has learned to think at
(`thinks_at`, bounded by a depth and a count, each question having to say something new), and when it stops
triggers the sentinel the event calls for - `BACK` for a repeat (D-068's lesson, now taught *after* thinking
rather than in place of it), the asking thought for a question, the end for a request. The thoughts come from
`ollama think`: a thinking model's reasoning about questions on a topic, with the questions it asked itself
teaching the network where to question.

**Alternatives rejected**
* *A second graph for thoughts.* Two files, two searches, and no way for a thought to share a node with the
  text it is about; one graph with two origins keeps every mechanism - the split, the merge, the counters, the
  ports - and separates the two by their first edge alone.
* *`THINK` handing over like `BACK`.* A walk that stopped wherever the model has learned to think would say
  less, not more; the search carries on and the thought happens beside it, asked for by whoever walked.
* *Training the LLM's whole answer.* Its answers are text and go in from `START` (`--with-answers`); it is the
  *thinking* that is in the right register - short, hedged, self-questioning - and it is kept apart.

**Consequences**
* Graph **format 4**; older files gain an unvisited `THINK` on load, `FIRST` is 4, and `THINK_Z` is fixed at the
  other edge of the drawn range so no random stream moved.
* A conversation now *thinks* before it backs up (`think`, on by default; `--no-think`), and the rethink carries
  the thought; a model taught no thoughts thinks nothing and says so (`stopped: "nothing"`), which is the honest
  answer rather than a made-up one.
* `onward` drops `THINK` from every continuation; only `BACK` hands over. `BACK` may not have children and
  `THINK` may, and the invariants say so.
* The rate of self-questioning is a research question: a node questions a thought once `THINK` is its cheapest
  way on, which the count model reaches after a few `observe_think`s and the sine model after a few more. Nothing
  decays it; a depth of 2 and one question per thought keep a thought from spending itself questioning.
* All three ports carry it, record for record: the Go parity tests compare the rethinks' thoughts, the Rust ones
  the model files.

**Lives in** `radixnet/thinking.py`, `radixnet/graph.py::observe_think`, `radixnet/search.py::onward`,
`radixnet/dialogue.py::think_back`, `radixnet/ollama.py::thoughts_from_prompt`, `go/radixnet/thinking.go`,
`rust/src/thinking.rs`

---

# Part XIX — The copy editor

### D-082 — A correction is a diff, and the LLM is asked for the smallest one

**Status** Accepted · 2026-09-24 · **Layer** feedback, negative network · **Beside** D-029, D-026

**Context** The adversarial reviewer (D-029) fails a *text*: its mark sets how
badly and its critique says roughly why, and every transition of the text is
blamed alike. The English tutor already knew better - it writes the sentence out
correctly and only the characters it changed are blamed (`NegativeNet.correct`,
`radixnet/diff.py`) - but that path was reachable only through a lesson. A
model that writes `"Hi howe are you??"` has one letter and one mark wrong; a
verdict on the sentence blames `"Hi "`, `" are "` and `"you"` too, and teaches
the negative network that greetings are failures.

**Decision** The LLM can be asked to be a **copy editor** instead of a critic
(`ollama correct`, `negative auto --correct`, `POST /api/ollama/correct`): it
returns each text written out correctly with the *smallest possible change*, and
the diff between the two is what the negative network learns. Only the
characters the editor struck out or replaced are blamed, at one failure per
corrected text (`--severity`); the correction itself never joins the failure
structure and clears blame where it is already known; a text handed back
unchanged clears blame; a text the editor said nothing usable about is neither
blamed nor cleared. The editor names the mistake in one word out of a fixed
vocabulary, with aliases the models actually use (`typo`, `capitalization`);
when it names none, the shape of the diff decides - punctuation, spacing, case,
letters inside a word, or words moved - so a reason is never invented from
nothing, and it is never `none` for a text that changed.

**Rejected**
* *Asking for the rating and the correction in one call.* Two jobs in one
  prompt is two chances to drift: a model asked to rate tends to rewrite, and
  one asked to rewrite stops rating. The reviewer and the editor are two modes,
  and the same loop runs either.
* *Scaling the severity by how much changed.* The diff already does that:
  more changed characters blame more edges. A per-text severity keeps the two
  numbers apart - how heavy a lesson, and how wide.

**Rationale** The negative network's value is *where* text goes wrong (D-026's
edge-level blame); a reviewer's verdict is the coarsest signal that can feed it
and a correction the finest. Asking for the smallest change keeps the diff an
honest map of the mistake rather than of the editor's taste.

**Consequences**
* The same prompt, the same parsing and the same reason rules in all three
  ports (`radixnet/ollama.py`, `rust/src/review.rs`, `go/radixnet/review.go`),
  held to Python byte for byte by the parity suites.
* The Automatic card and the report card grow an editor's vocabulary -
  `corrected`, `unchanged`, `uncorrected`, `edits`, `change_rate` - beside the
  reviewer's marks, and a round record says which mode ran it.
* An LLM that rewrites freely produces a wide diff and a wide lesson; the
  prompt forbids it, and the `changes` on every entry show what it did.

**Lives in** `radixnet/ollama.py` (`correct_texts`), `radixnet/blame.py`
(`faults_from_corrections`, `correction_reason`, `reason_from_changes`),
`radixnet/critic.py`, `radixnet/cli.py`, `radixnet/api.py`,
`frontend/src/components/OllamaPanel.jsx`, `rust/src/review.rs`,
`go/radixnet/review.go`

---

### D-083 — The veto can keep its provenance to itself

**Status** Accepted · 2026-09-24 · **Layer** negative network, output paths · **Beside** D-026, D-029

**Context** The guard (README, *The guard: both networks on every answer*)
was built so that nothing is filtered silently: every answer carries every
verdict - rule, risk, peak, ratio, reasons, blamed fragments, a sentence of
why - for every candidate it judged. That is right when the question is *why
was this dropped*, and wrong when the question was the answer: three texts
asked for come back with nine judgements, and a conversation with one per
candidate reply considered.

**Decision** The provenance of a veto is a setting, on by default and off on
request: `FilterConfig.provenance`, `--no-provenance` on the guard flags and
on `negative filter`, `{"provenance": false}` on `generate`, `predict`,
`converse` and `negative/filter` for one answer, and
`POST /api/negative/settings {"provenance": false}` for every answer a server
gives. Off, the veto applies exactly as before - the same candidates are
stopped, `learn` still blames them - but the pair reports each verdict as its
text, decision and rule alone, and the guard's report is the counts alone:
how many were judged, how many vetoed. Nothing is filtered silently still -
the count is always there - but nothing is explained unasked.

**Rejected** *A verbosity knob on the verdict (fewer spans, no `why`).* The
cost is the list itself, not the width of its rows; and a half-explained veto
is worse than a counted one, because it looks complete.

**Rationale** The judgement is computed either way (the decision needs it),
so the setting is about what is *reported*, which is the caller's business
and nobody else's - which is also why it is a per-answer field with a
server-wide default rather than a flag the negative network carries.

**Consequences**
* The guard report has two shapes, told apart by `provenance: false` and the
  presence of `judged`; the frontend's guard notice shows a count-only report
  without a *why* to open, and its verdict card renders a terse verdict.
* The same setting in all three ports, held to Python by the parity suites
  (`tests/test_rust_parity_negative.py`, `tests/test_go_parity.py`).

**Lives in** `radixnet/duo.py` (`FilterConfig.provenance`,
`NegativeFilter.terse` / `report`), `radixnet/api.py` (`guard`,
`_guard_report`, `negative_settings`), `radixnet/cli.py` (`add_guard_flags`,
`_guard_doc`), `frontend/src/components/GuardNotice.jsx`, `rust/src/duo.rs`,
`go/radixnet/duo.go`

---

# Part VII — Superseded decisions

Kept because the reversal is information.

| Was | Became | When | Why it turned over |
|---|---|---|---|
| The brain is a directed **acyclic** graph | Directed **cyclic**; cycles are a feature (D-001) | 2026-08-15 | Working through the model, cycles turned out to be what the structure is *for*, not an artefact to remove. |
| Prediction truncated at `2 × length` characters | `max_length` is an **optional** cap; without it the whole cheapest path is returned (D-008) | 2026-09-10 (`a20137d`) | An arbitrary multiple of the requested length silently cut off correct completions. |
| ZIP entries unpacked into one upload per file | The archive is kept whole and unpacked in memory on demand (D-034) | 2026-09-11 (`219b1ad`) | A 17 708-entry archive produced 17 708 upload rows. |
| 10 000 entries / 256 MB unpacked / 64 MB body caps | No limits by default; limits available as optional arguments (D-035) | 2026-09-11 (`c99bb80`) | A real corpus (`go-master.zip`) was refused by a cap defending against a threat model that does not apply on localhost. |
| Weight = `count_scale · log(1 + count)` | The dual frequency function with a sliding window (D-022) | 2026-09-12 (`5961f6c`) | Pure accumulated frequency cannot forget; the model could not move on from early data. Old files still load with the old behaviour. |
| Generation sampled; prediction searched | Generation **is** the prediction search run to the end (D-025) | 2026-09-12 (`40faaea`) | Two code paths drifted, and the tabs disagreed about what the model would say. |
| The evolve loop punished the worst **half** of fakes | Failure-proportional boosting, and blatant failures handled locally (D-027, D-028) | 2026-09-11 (`dc4eb78`) | A fixed split ignores *how* bad a failure was, which is exactly the signal available. |
| Docker's default SIGTERM | `STOPSIGNAL SIGINT` (D-037) | 2026-09-10 (`b2adb78`) | `docker stop` discarded an entire evolve run without saving. |
| A grade penalised the **whole** wrong sentence | Blame only the characters the teacher changed (D-046) | 2026-09-12 (`27ac0ac`) | Most of a corrected sentence is word for word what the network wrote; the whole-sentence penalty taxed the trigrams that were right. |
| The negative network was opt-in | The pair guards every answer; switching it **off** takes a flag (D-047) | 2026-09-14 (`f6585b8`) | Every surface handed out whatever the positive model wrote, including sentences the tutor had corrected an hour earlier. |
| A looping candidate was skipped | The voice backs up to where the loop started and re-plans (D-062) | 2026-09-14 (`414bd1d`) | Skipping threw away everything the candidate got right — the words before it went round were said once and were the most likely thing to say. |
| Go hid five tabs; Python-only endpoints 404'd (D-044) | Go serves everything but the schedule preview (D-066) | 2026-09-14 (`a34b822`) | The gap closed. What remains is recorded as deliberate rather than undone. |
| Counters were unbounded integers | Two-digit odometers wrapping at 10^15 (D-064) | 2026-09-14 (`2751287`) | Every counter would eventually leave the 53-bit mantissa carrying it through a model file, the API and a JavaScript Number. |
| A single thumb up or down | Rewards weighted by a mark out of 10 (D-050) | 2026-09-12 (`907a7af`) | A binary thumb throws away most of what a grader knows. |

---

# Part VIII — Open questions

Numbered for reference. These are genuinely open — each would change something
in the system, and none can be settled from the code as it stands.

**Q-1 — ~~The metacognition half of D-001~~ — RESOLVED (D-061 to D-063, D-068).** The
claim was that on hitting a cycle the brain hands over to metacognition rather
than looping. The hand-over now exists: a voice that detects a stutter backs up
to where the loop began and re-plans from there under a different procedure,
widening its search the further back it goes, once per turn, and records what it
did. The remaining question was **scope** — metacognition was a property of
*conversation* only — and D-068 answers it: what a rethink finds out is taught to
the graph as an edge into a third sentinel, and the search consults it in every
walk, so `predict`, `generate` and the agent's loop hand over too. What is open
now is the *rate*: how fast a node should learn to hand over, whether it should
ever unlearn, and what a wrongly taught hand-over costs.

**Q-2 — Evidence for the sine (D-002).** Now stated fully in
`Research/SineWaveActivationFunction.md` §9 and §12, with a replication protocol.
The decision left here is whether to *run* it — multiple seeds, a second task,
the 28–48 range explained — or to let the result stand as sufficient for its
purpose.

**Q-3 — Is the sine a fixed family?** Only `a, b, h, k` are learned; the
functional form is always a sine. Was a learnable *family* (sums of sines,
harmonics, a per-node blend) ever intended, or is "it is a sine" the claim
itself?

**Q-4 — What tells us the system is improving (D-011)?** The evolve loop has no
convergence criterion and no held-out evaluation; the real-vs-fake gap from the
model's own discriminator is the only signal, and that can widen while output
gets worse. Should there be a held-out set and a fixed metric, and if so what
counts as success for this architecture?

**Q-5 — The count model has no sine (D-021).** Its activations are pinned to the
constant 1, so the project's central research claim plays no part in it. Is that
permanent — the count model is a deliberately different algorithm that shares
only the structure — or is a count model *with* learned activations a direction?

**Q-6 — Threat model (D-035).** The no-limits decision is right for localhost
and wrong for anything exposed. Is this ever intended to run somewhere
reachable? If so, the caps, the sandbox's best-effort network isolation (D-031)
and the upload path all need a second pass.

**Q-7 — Is racy counting a performance choice or a research position (D-040)?**
It was measured at 0.7 % lost traversals for no speed-up on a 4-core machine. As
a pure optimisation it has not earned its place. As a claim — that lossy,
contended counting is what a brain actually does, and the 0.7 % is noise the
model should tolerate — it is a different kind of decision and should be stated
as one. Which is it?

**Q-8 — Back-port lazy weights to Python (D-042)?** The O(texts × edges)
recompute is still there on the Python side. Is Python meant to stay the simple
reference implementation, with Go the one that has to be fast — or should the
optimisation come back across?

**Q-9 — Where is this going next?** Go has the count model, a server and the
performance work. Is the intent that Go eventually carries the gradient model
too and becomes the primary engine (Python staying as the readable reference),
or that Go stays a specialised fast path for the count model only?

**Q-10 — Scope of this document.** It covers `RadixCyclicNN`. The repository
holds four other variations (`AbsorbingMarkovChain`, `RadixTreeRNN`,
`RadixTrieLLM_RNN`, `RadixTreeLLM`) described as further solutions to the same
problem, and the tests of the ideas themselves under `Experiments/`. Should
this become a repository-wide decision log — with a section on what each
variation tries and why it diverges — or stay scoped to this one?

**Q-11 — ~~What does "how the transformer pivots" mean?~~ — RESOLVED.** The
positional reading is the intended one: the character shared by two windows *is*
the pivot. Three is therefore the smallest window that gives a pivot with context
on either side, and D-006 records it as the reason rather than as a consequence.

**Q-12 — ~~What does 2NRL stand for?~~ — RESOLVED.**
*Double-Negative Reinforcement Learning.* The `2N` is two negatives: training
**on** the failures, then negating the representation that produced them —
phases 1 and 2 of D-009. The third phase is the positive consolidation, so the
name and the three-phase count do not conflict. Recorded in D-009 and in the
paper's title and §2.1; the expansion had appeared nowhere in the repository
before, and an earlier guess of *Two-phase Negative Reinforcement Learning* was
wrong (it also contradicted the three-phase procedure it labelled).

**Q-13 — ~~Is "pull hard on the thread" a positive-side boost?~~ — RESOLVED, and
the premise was wrong.** It is not a boost of any kind. My process is *explore
rapidly and widely until you find a thread, then tighten the exploration and
iterate* — a schedule over **search breadth**, not over learning rates.
Nothing in the system anneals temperature, beam width or sample count; D-033's
expression evaluator is the obvious mechanism and is pointed at the wrong
quantity. Recorded as D-067. The trigger, which looked like the hard part, is
settled too (`Research/2NRL.md` §6.4): the question was wrong. Not "what counts
as finding a *thread*", which is a judgement, but **what counts as finding a
reward** — the network is rewarded, or it reaches a known area of completion.
Both are events the system already emits, so neither needs a bar set in advance.
What is left is the schedule that listens to them.

**Q-14 — Does the process have a stopping rule?** "Fail consistently, then
invert" describes a loop. In a life it ends when the thing is learned; the answer
given in `Research/2NRL.md` §12 is *"when the thing worked"*. The evolve loop
never ends (D-011) and has no convergence criterion (Q-4). What remains open is
making that judgement computable — which is Q-13's problem in another form, since
both come down to recognising that something has started working.

**Q-15 — Does the negative network supersede inversion (D-045 vs D-009)?** The
negative network is 2NRL's negative phase made permanent: a standing model of
how text goes wrong, consulted at output time rather than trained-on and
discarded. If a failure can be kept, named and vetoed against, what is the
transient negative-phase-and-invert still buying? Are they two mechanisms for
one job, or does each do something the other cannot?

**Q-16 — Who arbitrates when blame and compression disagree (D-046 vs D-007)?**
A blamed transition is kept out of compression so the fragment stays nameable.
On a long-running system that accumulates blame, how much compression is being
given up — and should blame expire (`forget`) on a schedule rather than only on
request?

**Q-17 — How many teaching loops is the right number?** There are now six
(tutor, critic, recall, chat, codegen, agent), each with its own config surface,
its own report shape and its own parity test. They all reduce to `(bad, good)`
and `reward`/`punish`. Is there a loop *abstraction* worth extracting — or is
the variety the point, each being genuinely shaped to its subject?

**Q-18 — Is the guard's default right (D-047)?** Guarding every answer by
default, with the teaching loops exempt (D-048), means the output a person sees
and the output a judge sees are systematically different. That is deliberate and
well argued. But it also means the quality a user experiences is partly the
filter's, and no measurement currently separates "the model got better" from
"the filter got better at hiding it". Should there be one?
