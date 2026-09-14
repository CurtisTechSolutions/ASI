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
separately and linked: `../Research/2NRL.md` for D-009, and the notes in
`../Research/` for D-001 and D-002.

A decision belongs here when reversing it would change the character of the
system, not just an implementation detail. Each entry is dated to the commit
that first carried it, so the file doubles as a chronological record of the
design's development.

**Status legend**

* **Accepted** — in force.
* **Research claim** — accepted and implemented, but resting on the author's own
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
D-010 inversion · D-011 perpetual self-upgrade

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

**Part VII — Superseded decisions** · **Part VIII — Open questions**

---

# Part I — Foundations: the research claims

These nine decisions are the system. Everything in Parts II–VI exists to serve
them. They come out of the author's reverse-engineering of the brain rather than
from the literature, and the project's stated method is to *refuse current
research and reinvent the wheel* — so they are recorded as claims, with their
evidence and their exposure stated plainly.

### D-001 — The graph is directed and **cyclic**; cycles are a feature

**Status** Research claim · 2026-08-15 (`2c79fdb`), realised 2026-09-09 (`d180176`) · **Layer** theory

**Context** The first formulation of the brain model was a directed *acyclic*
graph. Working through it, the author concluded that the brain is a directed
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
* The "hand over to metacognition" half of the claim is **not implemented**.
  The system has no second faculty to defer to; search simply keeps paying the
  cost of going round. See Q-1.

**Lives in** `Research/CyclesAreAFeature.md`, `radixnet/search.py`,
`radixnet/beam.py`, `go/radixnet/search.go`, `go/radixnet/beam.go`

---

### D-002 — The activation is a sine, not a sigmoid or a ReLU

**Status** Research claim · 2026-08-13 (`9e168a1`) · **Layer** theory

**Context** The brain is taken to be an analog computer, so sine waves are
assumed to be the native encoding of information in it. Looking at a sigmoid
next to a sine, the author read the sigmoid as a sine "rotated 45 degrees" —
a truncated special case of the more general periodic function.

**Decision** Use `f(x) = a·sin(b·(x − h)) + k`, defaulting to
`a = −1, b = 1/3, h = 0, k = 0` — that is, exactly `−sin(x/3)`.

**Evidence** The author's own comparison (`ActivationFunctionTest/`) on a
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
  reported. The range 28–48 is itself wide. This is the single most
  load-bearing claim in the project and the thinnest measurement supporting
  it. See Q-2.

**Lives in** `Research/SineWaveActivationFunction.md`, `ActivationFunctionTest/`,
`radixnet/activation.py`

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

**Context** The author's framing: rather than fight the vanishing gradient with
deeper machinery, accept it, and move the learning into the activation.

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

**Context** The author's specification: `N*N`, and "activation of the child ×
activation of the parent".

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

**Context — the author's stated reason** Three was not chosen as "the smallest
window that leaves an overlap"; that is a consequence, not the motive. The
trigram was chosen because **it reminded the author of transformer architecture,
in how the transformer pivots**. The structural echo is direct: with stride 1
every character is shared by the window that ends on it and the window that
begins on it, so each character is a *pivot* joining two contexts — and the
two-character overlap that results is what every edge in the graph is keyed on.
The relation a transformer computes with attention, this representation carries
in the shape of the window itself. (The precise sense of "pivots" intended here
is recorded as Q-11.)

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

**Context** The author's specification named a cost function and Dijkstra.

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

**Context — 2NRL is the author's own learning process, formalised.** This is not
an algorithm arrived at from the literature and then justified. The author is
self-taught, and describes the method that produced that education directly:
**fail consistently, then do the inverse of what failed; and once a thread worth
pulling appears, pull hard.** 2NRL is that procedure written as an update rule —
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
* "Pull hard on the thread" is implemented as D-027's failure-proportional
  boosting, and arguably only half of it: the boost scales with how badly
  something *failed*, where the author's description is of pursuing a
  *promising* direction. See Q-13.
* The acronym's expansion is recorded nowhere in the repository; the code calls
  it only "the author's two-phase scheme". See Q-12.

**Lives in** `radixnet/model.py::RadixNet.two_nrl`

---

### D-010 — Inversion is negating every weight and every amplitude

**Status** Accepted · 2026-09-09 (`d180176`) · **Layer** learning

**Decision** `invert()` sets `w → −w` on every alive edge and `a → −a` on every
node, and flips the `inverted` flag.

**Why it works** The edge signal is `w · f_p · f_c` (D-005). Negating `w`
negates the product once; negating both `a`s negates it twice more — net, every
edge signal changes sign, the softmax ordering reverses, and the most likely
continuation becomes the least likely. Two inversions are the identity, which
the test suite asserts.

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
*other* node. The implementation picks the parity that covers the most edges of
the path, and where a node is shared it takes the largest amount.

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

**Context** The author's framing was rates as *graph functions of the epoch* —
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

**Status** Accepted · 2026-09-12 (`ec78de9`) · **Layer** architecture

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

---

# Part VIII — Open questions

Numbered for reference. These are genuinely open — each would change something
in the system, and none can be settled from the code as it stands.

**Q-1 — The metacognition half of D-001.** The claim is that on hitting a cycle
the brain *hands over* to metacognition or another region. The implementation
only makes cycles finite in state space and pays their cost. Is the hand-over a
component that should exist (a second faculty that takes over when the search
starts revisiting states), or was it always a description of *why* cycles are
tolerable rather than a mechanism to build?

**Q-2 — Evidence for the sine (D-002).** The strongest claim in the project
rests on one RL task at 250 episodes with no variance reported and a 28–48 range.
Is a firmer comparison worth running — multiple seeds, a second task, the range
explained — or is the result already settled enough for its purpose here?

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
holds five other variations (`AbsorbingMarkovChain`, `RadixTreeRNN`,
`RadixTrieLLM_RNN`, `RadixTreeLLM`, `SBNN_RNN_ActivationFunction`) which are
described as further solutions to the same problem. Should this become a
repository-wide decision log — with a section on what each variation tries and
why it diverges — or stay scoped to this one?

**Q-11 — What does "how the transformer pivots" mean precisely (D-006)?** The
reading recorded above is the *positional* one: with stride 1 each character is
shared by two windows and so joins two contexts, the way attention relates a
position to its neighbours. Other readings are available — the query/key/value
pivot around a single token, or the way attention re-centres the sequence on
whichever position it weights. Which one was the analogy? It decides whether
three is the right number or simply the first number that worked.

**Q-12 — What does 2NRL stand for?** The expansion appears nowhere in the
repository. `Research/2NRL.md` currently uses the acronym as a proper name.

**Q-13 — Is "pull hard on the thread" fully implemented (D-009, D-027)?** The
boost scales the negative phase by how badly a sample *failed*. The described
process is about recognising a *promising* direction and pursuing it hard. Those
are different signals. Should there be a positive-side boost — a promising
result training harder, not just a bad one?

**Q-14 — Does the process have a stopping rule?** "Fail consistently, then
invert" describes a loop. In a life it ends when the thing is learned. The
evolve loop never ends (D-011) and has no convergence criterion (Q-4). What
ended an iteration for the author?
