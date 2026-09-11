# RadixCyclicNN — Design Specification

A pure-Python (zero external dependency) neural network whose structure is a
**self-compressing cyclic graph** inspired by a Radix Tree. This document is the
contract every module is implemented against. Read it fully before writing code.

Directory: `RadixCyclicNN/` (this directory). Python package: `radixnet`.
Python 3.11+, standard library only. `torch` is an *optional* accelerator that is
imported lazily and never required.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| Encoding: sliding window of 3 characters | `Encoder.encode(text)` -> list of overlapping trigrams (`"hello"` -> `["hel","ell","llo"]`). Stride 1. |
| Decoding | `Decoder.decode_trigrams(["hel","ell","llo"])` -> `"hello"`: first trigram in full, then the last char of each following trigram. `Decoder.decode_path(labels)` does the same for (possibly compressed) node labels: first label in full, then `label[2:]` of each following label. |
| Self-compressing cyclic graph (Radix Tree idea) | Nodes carry a label of length >= 3. A node with label length L *contains* the L-2 trigrams `label[i:i+3]`. Unary chains (`p` has exactly one child `c`, `c` has exactly one parent `p`, `p != c`) are **merged** like a radix-tree path compression: `p.label += c.label[2:]`. A transition observed *into* or *out of* the middle of a compressed node **splits** it (radix-tree split). Repeated trigrams create cycles; cycles are a feature. |
| Accept vanishing gradient; update the activation instead; `N*N`; `act(child) * act(parent)` | Weights form an N x N (node x node) sparse matrix `W`. The signal on edge `p->c` is `score = W[p,c] * f_p(z_p) * f_c(z_c)` — the activation of the child times the activation of the parent. Learning is a **one-hop local rule** (no back-propagation through depth): each observed transition updates only `W[p,c]`, the node states `z`, and the **activation-function parameters** of `p` and its children. Nothing propagates further, so vanishing gradients are irrelevant by construction. |
| Custom activation `-1 * sin(x / 3.0)` | Every node owns a parametric sine `f(x) = a * sin(b * (x - h)) + k`, initialised to `a=-1, b=1/3, h=0, k=0` (exactly `-sin(x/3)`). All four parameters are learnable per node. |
| Shortest path prediction (cost function + Dijkstra) | Edge cost `cost(p->c) = -log(softmax_over_children(p)[c]) + step_penalty` (>= 0). Prediction runs Dijkstra from the context node over the depth-unrolled graph and returns the cheapest path that emits the requested number of characters, or the cheapest path to the END node. |
| Train & predict | `RadixNet.train / predict / generate / score`. |
| Constantly self-upgrading (GAN idea) | `Evolver`: generator = the model, discriminator = a second `RadixNet`. Each generation: sample fakes, discriminator learns real-vs-fake via 2NRL, low-scoring fakes become the generator's "garbage" for 2NRL, real corpus samples are the fine-tune pass. Runs indefinitely in a background thread (API) or for N generations (CLI), checkpointing as it goes. |
| 2NRL | `two_nrl(bad, good)`: (1) train on bad/garbage data, (2) `invert()` the network, (3) fine-tune on correct data with a smaller learning rate. |
| CLI / API / React frontend | `radixnet.cli` (argparse), `radixnet.api` (stdlib `http.server`, JSON), `frontend/` (Vite + React). |
| Checkpointing / save / load | JSON (optionally gzip) model files, `CheckpointManager` with rotation + `latest` pointer + resume. |
| GPU acceleration / performance | `Backend` abstraction: `PythonBackend` (always available, optimised pure Python over flat CSR arrays) and `TorchBackend` (optional; picks `cuda` > `mps` > `cpu`). The graph exports CSR arrays once per epoch; the backend runs vectorised mini-batch steps. |

---

## 2. Package layout

```
RadixCyclicNN/
  DESIGN.md                 this file
  README.md                 user docs (install, CLI, API, frontend, concepts)
  pyproject.toml            zero runtime deps; console script `radixnet = radixnet.cli:main`
  radixnet/
    __init__.py             exports RadixNet, TrainConfig, Encoder, Decoder, SineActivation, get_backend,
                            CheckpointManager, Evolver, EvolveConfig, __version__
    __main__.py             `python -m radixnet` -> cli.main()
    activation.py           sine activation (parametric sine + derivatives)
    encoding.py             Encoder / Decoder
    graph.py                RadixCyclicGraph (nodes, edges, trigram index, split/merge, CSR export/import, to_dict/from_dict)
    backend.py              CSR, NodeParams, Backend protocol, PythonBackend, TorchBackend, get_backend()
    search.py               PathResult, Dijkstra predictor + stochastic sampler
    beam.py                 Prediction, beam_predict (top-K / bottom-K continuations in one search; section 19)
    model.py                GraphModel (shared base), RadixNet, TrainConfig, model-kind factories (load_model, new_model, ...)
    countnet.py             CountRewardGraph, CountRewardNet - the count / reward model (section 19)
    schedule.py             learning-rate schedules as graph functions of the epoch (section 18)
    gan.py                  Evolver, EvolveConfig (GAN-style self-upgrade loop)
    checkpoint.py           CheckpointManager
    bench.py                benchmarks (chars/sec, predictions/sec)
    cli.py                  argparse CLI
    api.py                  HTTP JSON API + static file serving
  tests/                    unittest (run: `python -m unittest discover -s tests -v` from RadixCyclicNN/)
  frontend/                 Vite + React app (`npm install && npm run build` -> frontend/dist)
  data/
    sample_corpus.txt       good/correct data (one text per line)
    sample_garbage.txt      bad/garbage data (one text per line)
```

`RadixCyclicNN/` is the working directory for all commands. Tests import
`radixnet` from the parent directory (`tests/__init__.py` or a `conftest`-style
`sys.path` insert at the top of each test file).

---

## 3. `activation.py`

```python
DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K = -1.0, 1.0 / 3.0, 0.0, 0.0

def sine_activation(x: float, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K) -> float
    # a * sin(b * (x - h)) + k       ->  default is exactly -1 * sin(x / 3.0)

def sine_derivative(x, a, b, h, k) -> float           # df/dx = a*b*cos(b*(x-h))

def sine_partials(x, a, b, h, k) -> tuple            # returns exactly 6 floats:
    #   (f, dfdx, dfda, dfdb, dfdh, dfdk)
    #   u = b*(x-h);  f = a*sin(u)+k;  dfdx = a*b*cos(u);  dfda = sin(u);
    #   dfdb = a*(x-h)*cos(u);  dfdh = -a*b*cos(u);  dfdk = 1.0

class SineActivation:            # convenience object (used by tests/docs, not in hot loops)
    __slots__ = ("a", "b", "h", "k")
    def __init__(self, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K)
    def __call__(self, x) -> float
    def derivative(self, x) -> float
    def partials(self, x) -> tuple
    def inverted(self) -> "SineActivation"     # a -> -a
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d)

def edge_signal(w: float, fp: float, fc: float) -> float    # w * fp * fc  ("activation of child * activation of parent")
```

---

## 4. `encoding.py`

```python
WINDOW = 3
START_LABEL = "<s>"      # reserved label of the START node (id 0)
END_LABEL   = "</s>"     # reserved label of the END node   (id 1)

class Encoder:
    def __init__(self, window: int = WINDOW)
    def encode(self, text: str) -> list[str]
        # sliding window, stride 1. len(text) < window -> [] ; "hello" -> ["hel","ell","llo"]

class Decoder:
    def __init__(self, window: int = WINDOW)
    def decode_trigrams(self, grams: list[str]) -> str
        # [] -> ""; first gram in full, then gram[-1] for each following gram
    def decode_path(self, labels: list[str], start_offset: int = 0, include_context: bool = True) -> str
        # labels: node labels in path order (len >= 3 each, may be compressed). START/END labels are skipped.
        # include_context=True : first real label contributes label[start_offset:] in full, every following
        #                        label contributes label[2:]   (used by generate() from START, start_offset=0)
        # include_context=False: first real label contributes label[start_offset + 3:] (the deterministic
        #                        remainder of a compressed node after the matched trigram), every following
        #                        label contributes label[2:]   (used by predict(): returns only the continuation)
```

Round-trip property: `Decoder().decode_trigrams(Encoder().encode(t)) == t` for all `len(t) >= 3`.

---

## 5. `graph.py` — RadixCyclicGraph

### 5.1 Storage (performance: flat parallel lists indexed by node id)

```python
START, END = 0, 1

class RadixCyclicGraph:
    labels: list[str]            # labels[i] = label of node i; labels[0] = "<s>", labels[1] = "</s>"
    z: list[float]               # node state / pre-activation, learnable
    a, b, h, k: list[float]      # per-node activation params (defaults -1, 1/3, 0, 0)
    count: list[int]             # number of times the node was visited in training
    alive: list[bool]            # False for nodes removed by merge (ids are never reused until compaction on save)
    children: list[dict[int, int]]  # children[p][c] = edge id e
    parents:  list[dict[int, int]]  # parents[c][p]  = edge id e
    edge_w: list[float]          # edge weights (index = edge id)
    edge_count: list[int]
    edge_alive: list[bool]       # tombstones for removed edges
    trigram_index: dict[str, tuple[int, int]]   # trigram -> (node_id, offset); offset = index of the trigram inside the label
    rng: random.Random           # seeded
    inverted: bool = False
    version: int                 # bumped on ANY structural or parameter change (cost caches key on it)
    structure_version: int       # bumped only on node/edge creation, removal, split, merge
```

Node id 0 is START, 1 is END; both always exist, are never merged or split, and
their labels are not in `trigram_index`. `f_START` and `f_END` are ordinary
activations of their own `z`.

New node: `z = rng.uniform(-4.5, 4.5)`, `a,b,h,k = defaults`, `count = 0`.
New edge: `w = rng.uniform(0.5, 1.5)`, `count = 0`.
If `inverted` is True at creation time: new nodes get `a = -DEFAULT_A` and new
edges get `w = -rng.uniform(0.5, 1.5)` (consistent with the inverted network).

### 5.2 Public API

```python
def __init__(self, seed: int = 0)
def num_nodes(self) -> int          # alive nodes (incl. START/END)
def num_edges(self) -> int          # alive edges
def num_trigrams(self) -> int       # len(trigram_index)
def compression_ratio(self) -> float   # num_trigrams / max(1, alive nodes excluding START/END)
def alive_nodes(self) -> list[int]

def lookup(self, trigram: str) -> tuple[int, int] | None        # (node_id, offset) or None
def get_or_create(self, trigram: str) -> tuple[int, int]        # creates a fresh 3-char node if unknown
def activation_of(self, node_id: int) -> float                  # f_i(z_i)

def observe_sequence(self, trigrams: list[str], count: bool = True) -> list[tuple[int, int]]
    # Registers a training sequence structurally:  START -> t0 -> t1 -> ... -> tn -> END.
    # Returns the list of (parent_id, edge_id) transitions in order (what the backend trains on).
    # Empty trigram list -> returns [] and does nothing.
    # Rules, for consecutive trigrams (x, y):
    #   (px, ox) = get_or_create(x); (py, oy) = get_or_create(y)
    #   - if px == py and oy == ox + 1: the transition is INSIDE a compressed node -> deterministic,
    #     no edge, no transition emitted.
    #   - otherwise: ensure x is the LAST trigram of its node (else split(px, ox + 1) and re-lookup),
    #     ensure y is the FIRST trigram of its node (else split(py, oy) and re-lookup); then add/lookup
    #     the edge px->py, edge_count += 1 (if count), emit (px, e).
    #   START -> node(t0): split so t0 is the first trigram of its node; emit (START, e).
    #   node(tn) -> END : split so tn is the last trigram of its node; emit (node, e).
    #   if count: count[node] += 1 for each node visited (once per visit), edge_count += 1 per edge use.
    # Note: a split changes (node, offset) for trigrams; always re-lookup after splitting.

def split(self, node_id: int, i: int) -> tuple[int, int]
    # Split node between trigram i-1 and trigram i  (1 <= i <= len(label) - 3).
    # A keeps id `node_id` with label[:i+2]; B is a new node with label[i:].
    # A's out-edges move to B (edge ids preserved, parents[c] re-pointed); A gets ONE new edge A->B
    # (initialised like a new edge, edge_count = count[node_id]). B copies A's z/a/b/h/k and count.
    # trigram_index updated. Returns (A, B). Bad preconditions -> ValueError.

def merge_child(self, p: int) -> bool
    # Merge p's single child c into p when: p, c not in (START, END); p != c; len(children[p]) == 1;
    # len(parents[c]) == 1. New label = labels[p] + labels[c][2:]. count[p] = max(count[p], count[c]).
    # c's out-edges move to p (edge ids preserved). If c had a self-loop c->c it becomes p->p.
    # Edge p->c is tombstoned; c is tombstoned (alive False, dicts emptied); trigram_index entries
    # of c re-pointed to p with offset shifted by len(labels[p]) - 2 (computed BEFORE relabel).
    # Returns True if merged.

def compress(self) -> int
    # Repeatedly merge all mergeable unary chains until none remain. Returns number of merges.

def invert(self) -> None
    # w -> -w for every alive edge, a -> -a for every node (incl. START/END), inverted = not inverted, version += 1

def child_scores(self, p: int) -> list[tuple[int, float]]     # [(child_id, score)], score = w * f_p * f_c
def child_probs(self, p: int) -> list[tuple[int, float]]      # softmax over child_scores (numerically stable)
def child_costs(self, p: int) -> list[tuple[int, int, float]] # [(child_id, edge_id, -log prob)], cached by version

def to_csr(self) -> CSR              # see backend.CSR; over ALL node ids (dead ones have empty rows)
def apply_csr_weights(self, csr: CSR, weights: list[float]) -> None   # write back weights in CSR order; version += 1
def node_params(self) -> NodeParams  # copies of z, a, b, h, k in node-id order
def apply_node_params(self, params: NodeParams) -> None             # version += 1

def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> RadixCyclicGraph
    # JSON-serialisable. Dead nodes/edges are compacted on save (ids remapped), START=0/END=1 preserved.
    # Includes rng state (random.Random.getstate() converted to lists) so training is reproducible after load.
```

### 5.3 Invariants (tests assert these after random sequences of observe/split/compress)

1. Every alive node except START/END has `len(label) >= 3`.
2. `trigram_index` maps every trigram of every alive node label exactly once; no trigram maps to a dead node; for `(n, o) = trigram_index[t]`: `labels[n][o:o+3] == t`.
3. Two alive nodes never contain the same trigram (a trigram lives in exactly one node).
4. For every alive edge `p->c` with neither endpoint a sentinel: `labels[p][-2:] == labels[c][:2]` (window overlap).
5. `children`/`parents` are mirror images; every alive edge appears in both; dead nodes have empty dicts.
6. After `compress()`, no unary chain remains (no non-sentinel p with exactly one child c != p, c non-sentinel, where c has exactly one parent).
7. After `observe_sequence(encode(t))`, walking `t`'s trigrams through the index and decoding gives `t` (structure round trip), before and after `compress()`.

---

## 6. `backend.py`

```python
class CSR:
    __slots__ = ("indptr", "indices", "edge_ids", "weights", "edge_pos")
    # indptr: list[int] (len n_nodes+1); indices: list[int] child ids in CSR order; edge_ids: list[int] edge id
    # at each CSR position; weights: list[float] (copied from graph.edge_w in CSR order);
    # edge_pos: dict[int, int] edge_id -> CSR position

class NodeParams:
    __slots__ = ("z", "a", "b", "h", "k")     # list[float] each, node-id order

class Backend(Protocol):
    name: str          # "python" | "torch"
    device: str        # "cpu" | "cuda" | "mps"
    def prepare(self, csr: CSR, params: NodeParams) -> Any            # upload / wrap mutable state
    def step(self, state, parents: list[int], positions: list[int], lr: float, act_lr: float,
             clip: float = 5.0) -> float
        # One mini-batch gradient step over the given transitions.
        #   parents[i]   : parent node id p
        #   positions[i] : CSR position of the observed edge (target child = indices[positions[i]])
        # Loss = mean over transitions of  -log softmax(scores_p)[target]
        #   scores_p[j] = weights[j] * f_p(z_p) * f_c(z_c)   for j in indptr[p]..indptr[p+1]
        # Gradients (g_j = prob_j - 1[j == target]):
        #   dL/dw_j   = g_j * f_p * f_c_j
        #   dL/dz_c_j = g_j * w_j * f_p * f'_c_j
        #   dL/dz_p   = sum_j g_j * w_j * f_c_j * f'_p
        #   dL/d(a,b,h,k)_c_j = g_j * w_j * f_p * (df_c/da, df_c/db, df_c/dh, df_c/dk)
        #   dL/d(a,b,h,k)_p   = sum_j g_j * w_j * f_c_j * (df_p/da, df_p/db, df_p/dh, df_p/dk)
        # Gradients are SUMMED over the batch (a parent appearing twice contributes twice), divided by
        # batch size, clipped element-wise to [-clip, clip], then applied once:
        #   w -= lr * dw ; z -= lr * dz ; (a,b,h,k) -= act_lr * d(a,b,h,k).   Guard: b = max(b, 1e-3).
        # Returns the mean loss of the batch computed BEFORE the update.
    def finalize(self, state) -> tuple[list[float], NodeParams]      # weights in CSR order + params (download)
    def node_activations(self, state) -> list[float]                 # f_i(z_i) for all nodes

class PythonBackend:   # pure python; hot loop binds math.sin/cos/exp/log and lists to locals; no per-edge objects
class TorchBackend:    # lazy `import torch`; device auto: cuda -> mps -> cpu (override with device=...);
                       # vectorised with index_select / scatter_reduce("amax") / index_add_.
                       # Must match PythonBackend within 1e-6 after one step on the same inputs.

def get_backend(name: str = "auto", device: str | None = None) -> Backend
    # "auto": torch if importable AND (cuda or mps available) else python.
    # "torch": torch on any device (ImportError with a clear message if torch is missing).
    # "python": PythonBackend.  Unknown name -> ValueError.
def torch_available() -> bool
def describe_backends() -> dict   # {"python": True, "torch": bool, "cuda": bool, "mps": bool, "default": "python"|"torch"}
```

---

## 7. `search.py` — shortest-path prediction

```python
class PathResult:   # attributes: text: str, labels: list[str], node_ids: list[int], cost: float,
                    # step_costs: list[float], expanded: int, reached_end: bool, full_text: str (set by RadixNet)
                    # to_dict() -> JSON-serialisable

def dijkstra_predict(graph, start_node: int, start_offset: int, min_chars: int, max_chars: int | None = None,
                     step_penalty: float = 0.0, to_end: bool = False, max_expansions: int = 200_000) -> PathResult
    # State = (node_id, chars_emitted). Emission of a non-start, non-sentinel node n: len(labels[n]) - 2 chars;
    # the start node emits len(label) - (start_offset + 3) chars (its deterministic remainder; 0 if START).
    # Edge cost = cost from graph.child_costs(p) + step_penalty. Standard Dijkstra with heapq and a best-cost
    # dict keyed by state; max_chars is an OPTIONAL cap (None = no limit on emitted characters): states with
    # chars_emitted >= max_chars are not expanded further and the text is truncated to it.
    # Goal: if to_end: reaching END. Else: the first popped state with chars_emitted >= min_chars (Dijkstra pops
    # in cost order, so it is the cheapest such path); reaching END before min_chars is also accepted as a goal.
    # Fallback: if no goal is reached (dead end / expansion cap), return the popped state with the most emitted
    # chars (ties -> lowest cost) — never raise. Text is decoded with Decoder.decode_path(include_context=False)
    # and truncated to max_chars when a cap was given.

def sample_walk(graph, start_node: int, start_offset: int, max_chars: int | None, temperature: float = 1.0,
                rng: random.Random | None = None, stop_at_end: bool = True) -> PathResult
    # Stochastic walk: at each node sample a child from softmax(scores / temperature). cost = sum of -log p.
    # Stops at END, at max_chars (None = no limit), or at a node without children.
```

---

## 8. `model.py` — RadixNet

```python
@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 0.05
    act_lr: float = 0.005          # activation-parameter learning rate
    batch_size: int = 256          # transitions per backend step
    clip: float = 5.0
    auto_compress: bool = True     # run graph.compress() after every epoch
    shuffle: bool = True           # shuffle transitions each epoch (seeded)
    checkpoint_every: int = 0      # epochs; 0 = off
    verbose: bool = False
    lr_schedule: str | None = None      # graph function of the epoch for lr (section 18); None = constant
    act_lr_schedule: str | None = None  # same for act_lr; may use `lr`, the epoch's learning rate
    reverse_schedule: bool = False      # play the schedules backwards: the last epoch's rates first
    def rates(self) -> list[tuple[float, float]]   # (lr, act_lr) of every epoch; validate() evaluates them all

class RadixNet:
    def __init__(self, seed: int = 0, backend: str = "auto", device: str | None = None)
    graph: RadixCyclicGraph ; encoder: Encoder ; decoder: Decoder ; backend: Backend
    history: list[dict] ; meta: dict  # {"created": iso, "seed", "epochs_total", "trained_chars", "trained_texts", "twonrl_runs"}

    def train(self, texts: list[str] | str, config: TrainConfig | None = None, *, checkpoint_manager=None,
              progress: Callable[[dict], None] | None = None, stop_event: threading.Event | None = None,
              **overrides) -> list[dict]
        # str -> [str]. Texts shorter than 3 chars are skipped (counted in record["skipped_short"]).
        # 1. observe every text structurally (graph.observe_sequence(count=True)) -> transitions (parent_id, edge_id)
        # 2. per epoch: csr = graph.to_csr(); state = backend.prepare(csr, graph.node_params());
        #    parents/positions for all transitions (positions via csr.edge_pos); shuffle (seeded rng); mini-batches
        #    of batch_size -> backend.step; then graph.apply_csr_weights(...) and graph.apply_node_params(...)
        #    if auto_compress: merges = graph.compress(); if merges: re-observe all texts with count=False to get
        #    fresh transitions (structural no-op, edge ids may differ).
        #    record = {"epoch", "loss", "perplexity": exp(loss), "nodes", "edges", "trigrams", "compression_ratio",
        #              "merges", "transitions", "seconds", "skipped_short"} ; self.history.append(record); progress(record)
        #    if checkpoint_manager and checkpoint_every and epoch % checkpoint_every == 0: checkpoint_manager.save(self, epoch, "epoch", record)
        #    if stop_event and stop_event.is_set(): break
        # meta["epochs_total"] += epochs run; meta["trained_chars"] += sum(len(t)) once per call.
        # returns the list of epoch records

    def locate(self, prefix: str) -> tuple[int, int]
        #   len(prefix) >= 3: graph.lookup(prefix[-3:]); if None: the most-visited node containing a trigram whose
        #   first two chars == prefix[-2:] (scan trigram_index) -> (node, offset of that trigram); else the
        #   most-visited node containing a trigram whose first char == prefix[-1]; else (START, 0).
        #   0 < len(prefix) < 3: most-visited node whose label startswith(prefix); else (START, 0).
        #   prefix == "": (START, 0).

    def predict(self, prefix: str, length: int = 20, mode: str = "dijkstra", step_penalty: float = 0.0,
                temperature: float = 1.0, to_end: bool = False, max_length: int | None = None) -> PathResult
        # mode: "dijkstra" (default; shortest path, min_chars=length, max_chars=max_length — None = NO cap on the
        #       emitted characters, the whole cheapest path is returned) | "sample" (walks until END or length chars)
        # PathResult.text is the CONTINUATION only; result.full_text = prefix + text.

    def generate(self, max_length: int = 60, mode: str = "sample", temperature: float = 1.0, count: int = 1,
                 seed: int | None = None) -> list[PathResult]
        # from (START, 0), decode with include_context=True; mode "dijkstra": single cheapest path to END
        # (to_end=True, max_chars=max_length); mode "sample": `count` stochastic samples.

    def score(self, text: str) -> dict
        # {"log_prob", "per_char": log_prob / max(1, len(text)), "chars", "transitions", "unknown_transitions"}
        # Sum of log child_probs for every transition along the text's path (START -> ... -> END), following
        # compressed nodes (in-node steps cost 0). Unknown trigram / missing edge -> log(1e-6), counted as unknown.

    def invert(self) -> None              # graph.invert()
    def compress(self) -> int             # graph.compress()

    def two_nrl(self, bad: list[str], good: list[str], neg_epochs=3, pos_epochs=3, neg_lr=0.05, pos_lr=0.01,
                progress=None, checkpoint_manager=None, stop_event=None) -> dict
        # phase 1: train(bad, epochs=neg_epochs, lr=neg_lr)                      -> "negative" records (phase="negative")
        # phase 2: invert()
        # phase 3: train(good, epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10) -> "positive" records (phase="positive")
        # records passed to progress carry "phase". returns {"negative": [...], "positive": [...], "inverted": graph.inverted}
        # meta["twonrl_runs"] += 1

    def stats(self) -> dict   # {"nodes","edges","trigrams","compression_ratio","inverted","backend","device",
                              #  "epochs_total","trained_chars","trained_texts","twonrl_runs","history_len","last_loss"}

    def save(self, path: str) -> None
        # JSON; gzip if path endswith ".gz"; {"format": "radixnet", "version": 1, "saved_at", "meta", "history",
        # "backend": name, "graph": graph.to_dict()}
    @classmethod
    def load(cls, path: str, backend: str = "auto", device: str | None = None) -> RadixNet
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d, backend="auto", device=None)
```

All records / history entries are plain JSON-serialisable dicts.

---

## 9. `gan.py` — Evolver (GAN-style self-upgrading loop)

```python
@dataclass
class EvolveConfig:
    samples: int = 8            # fakes per generation
    real_per_generation: int = 8
    max_length: int = 40
    temperature: float = 1.0
    neg_epochs: int = 1 ; pos_epochs: int = 1 ; neg_lr: float = 0.05 ; pos_lr: float = 0.01
    disc_neg_epochs: int = 1 ; disc_pos_epochs: int = 1
    checkpoint_every: int = 0   # generations
    seed: int = 0

class Evolver:
    def __init__(self, generator: RadixNet, corpus: list[str], discriminator: RadixNet | None = None,
                 config: EvolveConfig | None = None)
        # discriminator None -> a fresh RadixNet(seed=config.seed + 1, same backend name)
    generation: int ; history: list[dict] ; discriminator: RadixNet
    def run_generation(self) -> dict
        # 1. fakes = [r.text for r in generator.generate(mode="sample", count=samples, max_length, temperature)]
        #    (fakes shorter than 3 chars are dropped; if none remain, use 1 dijkstra sample)
        # 2. real = rng.sample(corpus, min(real_per_generation, len(corpus)))
        # 3. discriminator.two_nrl(bad=fakes, good=real, neg_epochs=disc_neg_epochs, pos_epochs=disc_pos_epochs)
        # 4. fake_scores = [discriminator.score(f)["per_char"] for f in fakes]; real_scores likewise
        # 5. worst = fakes with score <= median(fake_scores) (at least 1)
        # 6. generator.two_nrl(bad=worst, good=real, neg_epochs, pos_epochs, neg_lr, pos_lr)
        # 7. record = {"generation", "fake_score_mean", "real_score_mean", "gap": real - fake, "gen_loss": last
        #              positive-phase loss, "nodes", "edges", "compression_ratio", "sample": fakes[0], "seconds"}
    def run(self, generations: int | None, checkpoint_manager=None, progress=None, stop_event=None) -> list[dict]
        # generations None -> until stop_event; checkpoint every config.checkpoint_every generations (tag "gen")
```

---

## 10. `checkpoint.py`

```python
class CheckpointManager:
    def __init__(self, directory: str, keep: int = 5, compress: bool = True)   # creates the directory
    def save(self, model, step: int, tag: str = "epoch", metrics: dict | None = None) -> dict
        # writes <dir>/ckpt-<tag>-<step:06d>.json[.gz] and <dir>/latest.json
        # {"name", "path", "step", "tag", "metrics", "saved_at", "bytes"}; prunes oldest beyond `keep`
        # (never the one referenced by latest.json). Returns the record.
    def list(self) -> list[dict]       # sorted by (step, saved_at)
    def latest(self) -> dict | None
    def load_latest(self, backend="auto", device=None) -> RadixNet | None
    def load(self, name_or_path: str, backend="auto", device=None) -> RadixNet
    def delete(self, name: str) -> bool
```

---

## 11. `cli.py`

`python -m radixnet <command> [...]` (also `radixnet` console script). Global
options (before the command): `--model PATH` (default `model.json`),
`--backend auto|python|torch`, `--device`, `--seed`, `--json` (machine-readable
output only, one JSON document on stdout).

| command | args | behaviour |
|---|---|---|
| `train` | `--data FILE [FILE...]` (one training text per line; `--whole-file` treats each file as one text), `--epochs`, `--lr`, `--act-lr`, `--batch-size`, `--no-compress`, `--checkpoint-dir`, `--checkpoint-every`, `--keep`, `--resume` (load latest checkpoint from dir first), `--out` (defaults to --model) | trains (loads --model first if it exists), prints per-epoch loss/perplexity/nodes/edges/compression, saves model |
| `predict` | `--prefix TEXT`, `--length N`, `--mode dijkstra\|sample`, `--to-end`, `--step-penalty`, `--temperature` | prints continuation + full text + cost + path |
| `generate` | `--count`, `--max-length`, `--mode`, `--temperature` | prints samples |
| `score` | `--text` or `--data FILE` | log-prob per text |
| `2nrl` | `--bad FILE`, `--good FILE`, `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--out` | runs two_nrl, saves |
| `invert` | `--out` | inverts and saves |
| `compress` | `--out` | compresses and saves, prints merges |
| `evolve` | `--data FILE`, `--generations N` (0 = forever, Ctrl-C stops cleanly and saves), `--samples`, `--max-length`, `--discriminator PATH` (load/save), `--checkpoint-dir`, `--checkpoint-every`, `--out` | GAN loop |
| `info` | | stats + history tail |
| `checkpoints` | `--dir DIR`, `--restore NAME --out PATH` | list / restore |
| `bench` | `--chars N`, `--epochs` | training transitions/sec + chars/sec, predictions/sec, Dijkstra expansions/sec |
| `serve` | `--host 127.0.0.1 --port 8000 --frontend-dir frontend/dist --checkpoint-dir checkpoints --upload-dir uploads` | starts the API |

Exit codes: 0 ok, 1 error (message on stderr). Every command supports `--json`.

---

## 12. `api.py` — HTTP JSON API (stdlib only)

`ThreadingHTTPServer` + `BaseHTTPRequestHandler`. All `/api/*` responses are JSON
with `Content-Type: application/json`, CORS headers (`Access-Control-Allow-Origin: *`,
`-Methods`, `-Headers`) and `OPTIONS` preflight support. Errors: `{"error": "message"}`
with 400/404/409/500. A single `ModelService` holds the model behind an `RLock`;
long operations (train / 2nrl / evolve) run in a background thread and are tracked
as a **job** (one at a time; a second request gets 409). Job status:
`{"id", "type", "state": "running|done|error|stopped", "progress": last record or null,
"history": [...records], "error": str|null, "started_at", "finished_at"}`.

| method & path | body | response |
|---|---|---|
| GET `/api/health` | | `{"ok": true, "version"}` |
| GET `/api/status` | | `stats()` + `{"job", "backends": describe_backends(), "model_path", "checkpoint_dir"}` |
| POST `/api/train` | `{"texts": [...]}` or `{"text": "..."}` (split on newlines, blank lines dropped) + `"epochs","lr","act_lr","batch_size","auto_compress"` | `{"job": {...}}` (async) |
| GET `/api/job` | | current/last job status |
| POST `/api/job/stop` | | sets the stop event; returns job status |
| POST `/api/predict` | `{"prefix","length","mode","to_end","step_penalty","temperature"}` | `{"prefix","continuation","full_text","cost","step_costs","path","node_ids","expanded","reached_end"}` |
| POST `/api/generate` | `{"count","max_length","mode","temperature"}` | `{"samples": [{"text","cost","path"}]}` |
| POST `/api/score` | `{"text"}` | score dict |
| POST `/api/2nrl` | `{"bad": [...],"good": [...],"neg_epochs","pos_epochs","neg_lr","pos_lr"}` (`bad_text`/`good_text` newline forms also accepted) | job (async, type "2nrl") |
| POST `/api/feedback` | rated texts `{"good": [thumbs up], "bad": [thumbs down]}` (also `*_text`, `*_files`), `neg_epochs=2`, `pos_epochs=3`, `neg_lr=0.5`, `pos_lr=0.1`, `batch_size=4` | `{"job" (type "feedback"), "action": "2nrl"\|"reward"\|"punish", "good", "bad"}` — both kinds: `two_nrl(bad, good)`; only good: a positive-phase `train`; only bad: a negative-phase `train` then `invert()`. Used by the frontend's Generate tab (thumbs up / down per sample) and the `feedback` CLI command |
| POST `/api/invert` | | `stats()` |
| POST `/api/compress` | | `{"merges", **stats}` |
| POST `/api/evolve/start` | EvolveConfig fields + `{"corpus": [...]}` or `"corpus_text"` + `"generations"` (null/0 = forever) | job (async, type "evolve") |
| POST `/api/evolve/stop` | | job status |
| GET `/api/evolve/history` | | `{"history": [...]}` |
| POST `/api/save` | `{"path"}` (default: model_path) | `{"path","bytes"}` |
| POST `/api/load` | `{"path"}` | stats |
| POST `/api/reset` | `{"seed"}` | stats of the fresh model |
| GET `/api/checkpoints` | | `{"checkpoints": [...], "latest": {...}|null}` |
| POST `/api/checkpoints/save` | `{"tag"}` | checkpoint record |
| POST `/api/checkpoints/restore` | `{"name"}` | stats |
| GET `/api/graph?limit=150` | | `{"nodes": [{"id","label","count","activation","z","a","b","h","k"}], "edges": [{"source","target","weight","count","prob","cost"}]}` — top-`limit` alive nodes by count plus START/END, edges among them |
| GET `/api/history` | | `{"history": model.history}` |
| GET `/api/uploads` | | `{"uploads": [{"name","bytes","chars","lines","modified"}], "upload_dir"}` — text files kept in the server's `upload_dir` (`--upload-dir`, default `uploads`; endpoints answer 400 when no directory is configured) |
| POST `/api/uploads` | JSON `{"name","content"}` / `{"name","content_base64"}` or `{"files": [...]}`; or `multipart/form-data` (every part with a filename, as bytes); or any other body with `?name=<file>` (raw bytes) | 201 `{"uploads": [record + "replaced": bool], "archives": [...]}`. Bytes that start with the ZIP magic are validated by `archive.extract_texts` (section 20) and stored as one `.zip` upload whose record carries `archive: true`, `files` (text entries), `skipped` and the summed `lines` / `chars`; `archives[]` summarises entries / extracted / skipped (path + reason); 400 for a corrupt archive, one without a text entry, or one over the entry / size limits. Names are reduced to a safe base name (no traversal); UTF-8 with BOM dropped; same name replaces the file |
| POST `/api/uploads/delete` | `{"name"}` | `{"deleted": name}` (404 when missing) |

`/api/train`, `/api/2nrl` and `/api/evolve/start` also accept upload names: `"files"` (train), `"bad_files"` / `"good_files"` (2NRL), `"corpus_files"` (evolve), each read as one text per non-blank line, or as one text per file with `"whole_file": true`. Inline texts and files combine; at least one text is required.
| GET `/` and other paths | | serves `frontend_dir` if it exists (SPA fallback to `index.html`, correct mime types, no path traversal), else a minimal built-in HTML page explaining how to build the frontend |

`create_server(host, port, model_path=None, checkpoint_dir=None, frontend_dir=None, backend="auto", device=None, seed=0) -> (server, service)`
and `run_server(...)` (serve_forever with clean shutdown on KeyboardInterrupt). Tests start the server on
port 0 in a thread and talk to it with `urllib.request`. If `model_path` exists it is loaded at start-up.
Mutating endpoints refuse (409) while a job is running.

---

## 13. `frontend/` — Vite + React

Dependencies: `react`, `react-dom`, `vite`, `@vitejs/plugin-react` only. `vite.config.js` proxies `/api`
to `http://127.0.0.1:8000` in dev; `npm run build` writes `frontend/dist`, which the API serves.
Files: `index.html`, `src/main.jsx`, `src/App.jsx`, `src/api.js` (fetch wrapper with JSON + error handling),
`src/styles.css`, `src/components/`:

* `StatusBar.jsx` — polls `/api/status` every 2s: nodes, edges, trigrams, compression ratio, inverted flag, backend/device, job state + latest progress.
* `TrainPanel.jsx` — textarea (one text per line), epochs, lr, start / stop; live epoch table (loss, perplexity, nodes, compression).
* `PredictPanel.jsx` — prefix, length, mode (dijkstra/sample; beam for the count model with K / beam width), to-end, step penalty; shows continuation (prefix + highlighted continuation), cost, probability, path chips with per-step costs, and the count model's top-K / bottom-K tables. A Like button (on the result and on every top / bottom row) rewards that text: `POST /api/feedback {good: [prefix + continuation]}` through the shared `useJob("feedback")` hook, i.e. `reward()` - a positive-phase pass for RadixNet, a traversal plus reward for the count model; the button shows the liked state and cannot reward the same text twice.
* `GeneratePanel.jsx` — count, max length, temperature, mode; list of samples with costs.
* `TwoNRLPanel.jsx` — bad textarea, good textarea, epochs/lrs; shows negative/positive losses; button to Invert manually.
* `EvolvePanel.jsx` — corpus textarea, samples, generations (blank = forever), start/stop; live SVG line chart of `gap` and `fake_score_mean` over generations + latest sample text.
* `CheckpointPanel.jsx` — list checkpoints, save checkpoint (tag), restore, save/load model path, reset.
* `GraphView.jsx` — SVG rendering of `/api/graph` (circular layout, edge opacity by prob, node radius by count, hover label).
* `ScorePanel.jsx` — score a text.

Plain readable CSS, responsive (single column under 800px). No TypeScript.

---

## 14. Tests (unittest, no deps)

* `test_activation.py` — default equals `-sin(x/3)`; numeric derivative check for all partials.
* `test_encoding.py` — round trips, short text, unicode, decode_path both modes.
* `test_graph.py` — invariants §5.3 under randomised observe/split/compress; merge/split round trip; CSR export/import; to_dict/from_dict equality; invert flips signs.
* `test_backend.py` — loss decreases on a tiny fixed graph; gradient check vs finite differences (python backend); torch vs python equality when torch is importable (skip otherwise).
* `test_search.py` — Dijkstra returns the trained sequence; to_end; fallback on dead end; sample_walk terminates.
* `test_model.py` — train reduces loss & perplexity; predict reproduces a training text continuation; score higher for trained text than garbage; two_nrl makes garbage continuation less likely than before; invert twice is identity; save/load round trip preserves predictions; compression changes node count but not predictions of trained texts.
* `test_checkpoint.py` — rotation, latest pointer, load_latest, resume.
* `test_gan.py` — one generation runs, history record shape, stop_event honoured.
* `test_cli.py` — subprocess smoke test of train/predict/info/2nrl/checkpoints/bench with `--json`.
* `test_api.py` — server in a thread; health/status/train(job polling)/predict/generate/score/2nrl/invert/compress/save/load/checkpoints/graph/evolve start-stop/static fallback.

---

## 15. Performance notes (must be followed)

* Hot loops (backend step, Dijkstra) bind `math.sin`, `math.cos`, `math.exp`, `math.log` and list references to locals; no attribute lookups or method calls per element; no per-edge object allocation.
* Graph storage is flat parallel lists; edges live in per-node dicts for O(1) lookup, exported to CSR per epoch.
* `child_costs` is cached per node and invalidated by `graph.version`.
* Dijkstra uses `heapq` with `(cost, tie_counter, node_id, chars)` tuples and a `best` dict; `max_expansions` guard.
* Training re-uses transition arrays across epochs unless `structure_version` changed.
* `bench` reports numbers; `PythonBackend` should reach roughly 100k+ transitions/sec on a modern CPU for small fan-outs.

---

## 16. Ollama integration (`ollama.py`) — prompt-driven corpora and an adversarial LLM review

Two ways of hooking the network into a local LLM served by [Ollama](https://ollama.com) (HTTP API, stdlib `urllib` only):

1. **Corpus from a prompt** — the LLM is asked for *N* lines about a prompt, either correct (`style="good"`) or
   deliberately wrong (`style="garbage"`): exactly the two inputs of 2NRL. The lines can be trained on, saved as an
   upload, or written to a file.
2. **Adversarial review** — the LLM plays the harsh critic: every sample the network generates (or any given text)
   gets a rating 0–10, a pass/fail verdict against a threshold and a one-sentence critique. Failed texts become 2NRL
   garbage, passed texts (plus an optional corpus) the fine-tune pass: an external discriminator for the GAN loop.

```python
DEFAULT_URL   = $OLLAMA_HOST (Ollama's own variable; "host:port" without scheme accepted) or "http://127.0.0.1:11434"
DEFAULT_MODEL = $RADIXNET_OLLAMA_MODEL or "llama3.2"
DEFAULT_TIMEOUT = 120.0

class OllamaError(Exception)          # unreachable / HTTP error / unusable answer; readable message
class OllamaClient(url=None, model=None, timeout=None)
    .url .model .timeout
    .models() -> list[dict]           # GET /api/tags -> [{"name","size","modified_at","details"}]
    .available() -> bool
    .generate(prompt, *, system=None, model=None, json_mode=False, options=None, timeout=None) -> str
                                      # POST /api/generate {model, prompt, system, stream: false, format: "json"?, options}
    .chat(messages, *, model=None, json_mode=False, options=None, timeout=None) -> str   # POST /api/chat

def parse_lines(text, limit=None) -> list[str]        # strips numbering / bullets / quotes, drops blanks, fences, duplicates
def corpus_from_prompt(client, prompt, lines=20, style="good", model=None) -> list[str]
def review_texts(client, texts, *, context=None, model=None, threshold=6.0, batch=20) -> list[dict]
    # JSON-mode prompt; each entry {"index","text","rating": 0..10 | None,"verdict": "pass"|"fail"|"unrated","critique"}
    # verdict recomputed from the rating; blank texts fail as "empty output" without asking the LLM
def sample_texts(model, count, prefix="", max_length=60, temperature=1.0, seed=None) -> list[str]
def adversarial_review(model, client, *, count=8, prefix="", max_length=60, temperature=1.0, texts=None,
                       threshold=6.0, context=None, ollama_model=None, seed=None) -> dict
    # {"source": "model"|"given", "model", "threshold", "texts", "reviews", "mean_rating", "pass_rate",
    #  "good": passed texts, "bad": failed + unrated texts}
```

CLI: `radixnet [globals] ollama [--url URL] [--ollama-model NAME] [--timeout S] <action>`:

| action | options | behaviour |
|---|---|---|
| `models` | | list installed models |
| `corpus` | `--prompt TEXT`, `--lines 20`, `--style good\|garbage`, `--out FILE`, `--train`, `--epochs 10`, `--lr 0.5`, `--batch-size 4`, `--model-out PATH` | prints the lines; writes / trains on them |
| `review` | `--count 8`, `--prefix`, `--max-length 60`, `--temperature`, `--text ...` / `--data FILE`, `--threshold 6`, `--context`, `--2nrl`, `--good FILE`, 2NRL options, `--out` | table of ratings + summary; `--2nrl` runs failed→invert→passed and saves |

`serve --ollama-url --ollama-model` set the API defaults.

API (`ollama_url` / `ollama_model` on `ModelService` / `create_server`; `/api/status` gains `"ollama": {"url","model"}`):

| method & path | body | response |
|---|---|---|
| GET `/api/ollama/models?url=` | | 200 always: `{"available", "url", "model", "models": [...], "error": null\|str}` |
| POST `/api/ollama/corpus` | `{prompt, lines=20, style="good", model?, url?, timeout?, save_as?: upload name, train=false, epochs, lr, act_lr, batch_size}` | `{"prompt","style","model","url","lines","texts","upload": record\|null,"job": job\|null}`; 202 when a train job started; 400 bad input; 409 job running; 502 Ollama failure |
| POST `/api/ollama/review` | `{count=8, prefix="", max_length=60, temperature=1, texts?\|text?, threshold=6, context?, model?, url?, apply="none"\|"2nrl", good?, good_text?, good_files?, neg_epochs, pos_epochs, neg_lr, pos_lr, batch_size}` | `adversarial_review` result + `"url"` + `"job"`; `apply="2nrl"` starts a 2NRL job with bad = failed+unrated and good = passed + given (400 when either set is empty) |

Frontend: an "Ollama" tab with a connection card (URL, model list), "Corpus from a prompt" (generate → train / save as upload / hold as 2NRL bad or good) and "Adversarial review" (ratings table, summary, apply as 2NRL with extra good files).

Docker: the API container gets `OLLAMA_HOST` (default `http://host.docker.internal:11434`, reachable through `extra_hosts`); the `ollama` profile runs the official `ollama/ollama` image with a model volume (`OLLAMA_HOST=http://ollama:11434`).

Tests (`tests/test_ollama.py`) use a fake Ollama server (stdlib `http.server`) that answers `/api/tags`, `/api/generate` (numbered lines for corpus prompts, JSON ratings for review prompts, configurable failures) and `/api/chat`.

---

## 17. Code generation (`codegen.py`) — sandbox, Ollama judge, 2NRL rewards

`problem -> Python program -> sandbox -> judge -> 2NRL`, in two semi-supervised phases over one problem list:

* **teacher**: the Ollama model (`CodeGenConfig.teacher_model`, default `gemma4`, env `RADIXNET_CODEGEN_MODEL`) writes a
  program (`teacher_generate`), the sandbox runs it, failures / rejections go back to the teacher (`teacher_fix`, up to
  `teacher_attempts` programs), the judge confirms. Learning: `two_nrl(bad=[texts of the wrong attempts], good=[text of the
  correct one])` — wrong answers first, then invert, then the correct answer.
* **model**: the network continues `model_prompt.format(problem)` (default `"{problem}\n"`) into code — attempt 0 by the
  cheapest path (`predict(mode="dijkstra", to_end=True)`), later attempts sampled — up to `model_attempts`; each attempt is
  run and judged; the loop stops at the first correct one; with `fallback_teacher` the teacher supplies the correct answer
  when the network never succeeds. Learning as above (all wrong attempts punished, the correct one rewarded).

The training text of an answer is `solution_text(problem, code) = prompt + "\n" + code + "\n"`, so the prefix the network
continues in phase 2 is exactly what it learned in phase 1.

```python
Problem(id, prompt, tests=None, expected_output=None); parse_problems(items); parse_problem_file(text, ext); load_problems(path)
    # .txt one prompt per line (# comments) | .jsonl objects | .json list or {"problems": [...]}; ids default to p1, p2, ...
Sandbox(python=None, timeout=10.0, memory_mb=256, cpu_seconds=None, isolate_network=True)
    .run(code, tests=None, expected_output=None, stdin="") -> RunResult(ok, exit_code, stdout, stderr, error, timed_out,
        seconds, expected_ok, network_isolated)
    # python -I -B, scratch cwd, empty environment, RLIMIT_AS / RLIMIT_CPU / RLIMIT_FSIZE set inside the child before
    # runpy runs the script, wall-clock timeout, `unshare -rn` when it works (probed once); never raises for a bad program.
check_style(code) -> StyleReport(ok, syntax_ok, pep8_ok, naming_ok, issues)
    # W191 tabs, E111 indentation, E501 > 79, W291/W293 trailing whitespace, W292 final newline, E302 two blank lines
    # before top-level defs; N801 CapWords classes, N802 snake_case functions, N803 arguments, N806 variables
    # (UPPER_CASE constants and CapWords aliases allowed)
extract_code(text)            # the longest ```python block, else the text
judge_with_ollama(client, problem, code, run, style, model=None) -> {"task","pep8","naming","score","issues","critique"}
    # JSON-mode adversarial reviewer prompt with the task, program, execution result and the objective style report
decide(run, style, llm, strictness) -> Verdict(correct, runs, task, pep8, naming, score, issues, critique, judged_by)
    # correct = runs and expected_ok is not False and task is not False and (lenient or pep8 and naming and style.ok)
    # task: False when the program fails or the stdout differs; the LLM's answer when asked; True with matching
    # expected output; None (counts as ok) when nothing can judge it. judged_by: sandbox | tests | ollama | none
CodeGenConfig(teacher_model, judge_model, phases=("teacher","model"), rounds=1, teacher_attempts=3, model_attempts=4,
    first_attempt_dijkstra=True, temperature=1.0, max_length=800, strictness="strict", use_judge=True,
    fallback_teacher=True, twonrl_per="problem"|"round", replay=True, replay_limit=64, teacher_prompt=None,
    model_prompt="{problem}\n", neg_epochs=2, pos_epochs=3, neg_lr=0.5, pos_lr=0.1, batch_size=4, checkpoint_every=0)
CodeGenTrainer(model, client, sandbox=None, config=None, external=None)
    .evaluate(problem, code, source, index) -> Attempt      # sandbox + style + (if it runs and use_judge) the LLM judge
    .solve_with_teacher / .generate_with_model / .solve_with_model
    .learn(bad, good) -> {"bad","good","action": "2nrl"|"reward"|"punish"|None,"neg_loss","pos_loss"}
        # bad and good: two_nrl; only good: train (reward); only bad: train + invert (punish);
        # replay adds up to replay_limit earlier correct texts to every positive phase
    .run(problems, progress=None, stop_event=None, checkpoint_manager=None) -> [problem and round records]
        # progress also receives attempt records: kinds "attempt", "problem", "round"; twonrl_per decides whether
        # learn() runs after each problem or once per phase-round; checkpoints every checkpoint_every problems
    # `external` is a context-manager factory entered around sandbox runs and LLM calls; the API passes
    # ModelService.pause_lock so readers are served while a program runs or the teacher thinks.
```

CLI `codegen --problems FILE` (section 11 of the README lists the options); the `ProblemPrinter` prints attempts as notes,
problems as table rows and round summaries. API: `POST /api/codegen/start` (job "codegen"; problems from `problems`,
`problems_text`, `problem_files` uploads), `GET /api/codegen/history`, `POST /api/codegen/solve` (one problem with the
model or the teacher, no training), `POST /api/codegen/run` (sandbox + style + verdict without an LLM). Frontend: the
"Code" tab. Tests (`tests/test_codegen.py`) use a fake Ollama whose teacher answers come from a scripted queue and whose
judge rejects programs containing `BAD_ANSWER`.

## 18. Learning-rate schedules (`schedule.py`) — rates as graph functions of the epoch

`Schedule(expression)` parses an expression with `ast.parse(mode="eval")`, walks the tree and rejects every node
that is not a number, a name from the whitelist, an arithmetic / comparison / boolean operator, a conditional
expression or a call of a whitelisted function; integer literals are rewritten to floats so `2 ** 100000000`
overflows (`OverflowError -> ScheduleError`) instead of allocating a huge integer. The compiled code object is
evaluated with an empty `__builtins__` and an environment holding `epoch` (1-based), `i` (0-based), `epochs`, `t`
(`i / (epochs - 1)`, 0 for a single epoch), `lr0` (the schedule's own base rate), `act_lr0`, `lr` (the epoch's
learning rate, for the activation schedule), the constants `pi` / `e`, the `math` functions listed in `FUNCTIONS`
and the helpers `linear(a, b)`, `geometric(a, b)`, `cosine(a, b)`, `step(a, factor, every)`, `warmup(a, b, n)`.
Every value must be a finite number `>= 0`; anything else is a `ScheduleError` (a `ValueError`, so the API answers
400 and the CLI exits 1) naming the epoch.

`preview_points(lr_schedule, act_lr_schedule, epochs, lr, act_lr, reverse=False)` gives `[{"epoch", "lr", "act_lr"}]`
(`reverse` plays the list backwards, keeping each epoch's `lr` / `act_lr` pair together - the "Reverse the schedule"
checkbox, `--reverse-schedule`, `reverse_schedule`): the
activation schedule sees the learning rate *of the same epoch* as `lr`, so `lr / 10` follows whatever curve the
learning rate has. `TrainConfig.rates()` uses it; `RadixNet.train` reads `(lr, act_lr) = rates[k]` per epoch and
stamps both on the epoch record (`"lr"`, `"act_lr"`), so the history shows what was actually used. `PRESETS`
(constant, linear / exponential / cosine ramps to 4x, step, 25 % compound growth, warm-up, "activation follows
lr / 10") feed the CLI listing, `GET /api/schedule` and the preset select of the Train tab; `POST
/api/schedule/preview` is what the tab's live chart calls (debounced) while an expression is typed. CLI:
`radixnet schedule` (table + `#` bar graph, or the preset listing) and `train --lr-schedule / --act-lr-schedule`.

## 19. The count / reward model (`countnet.py`, `beam.py`) — a second algorithm, selectable at run time

`GraphModel` (in `model.py`) is what both kinds share: `graph`, `encoder` / `decoder`, `_clean_texts`,
`_observe_grams` / `_observe`, prefix location (`_locate`, `locate`, `_prefix_start`), `generate`, `score`,
`compress`, `save` / `load`. A kind implements `train`, `predict`, `two_nrl`, `reward`, `punish`, `invert`, `stats`,
`to_dict` / `from_dict` and names itself with `kind` / `format` / `label` / `description`. `model_classes()`,
`model_kinds()`, `model_class(kind)`, `new_model(kind, ...)`, `model_from_dict(d)` and `load_model(path)` dispatch on
the document's `format` (`"radixnet"` vs `"radixnet-count"`), so checkpoints, `POST /api/load` and the CLI restore
whatever kind a file holds. `RadixNet.reward` / `punish` are the thumbs-up / thumbs-down primitives (a positive-phase
pass; a negative-phase pass followed by `invert`) that the API, the CLI `feedback` command and the codegen trainer
call instead of spelling the rule out themselves.

`CountRewardGraph(RadixCyclicGraph)` keeps a third per-edge list, `edge_reward`, next to `edge_w` / `edge_count`,
and derives every weight from the two tracked numbers: `weight(count, reward) = count_scale * log1p(count) +
reward_scale * reward`. Every node is created with `a = 0, k = 1`, so the sine activation is the constant 1 and the
base class's score `w * f_p * f_c` is the weight itself: `child_probs`, `child_costs`, Dijkstra, sampling, `split`
(the new internal edge gets the node's count and reward 0) and `merge_child` (activation ratios are 1) work
unchanged. `add_reward(edge_ids, amount)` and `recompute_weights()` keep `edge_w` in sync and bump `version` so the
cost cache refreshes; `invert()` negates the rewards; `to_dict` / `from_dict` carry `edges.reward` and the two scales
(`weights`) and rebuild the weights on load.

`CountRewardNet(GraphModel)`: one `_passes(texts, cfg, count, reward, phase)` routine underlies everything. It
registers the texts structurally first (no counting) and compresses, so every pass - the first included - walks the
same transitions (steps inside a compressed node are deterministic and never counted); then per epoch it re-observes
with `count=True` (`train`: one more traversal per path), adds `reward` to every edge of every path (`reward`:
`+strength` with a traversal; `punish`: `-strength` without one), recomputes the weights, measures the loss as the
mean `-log P` of the transitions and records the usual epoch fields plus `traversed` and `reward`. `two_nrl` is
`punish(bad)` then `reward(good)` - no inversion, a penalty already makes a path unlikely. `TrainConfig` fields
`lr` / `act_lr` / `batch_size` are accepted and ignored; feedback magnitude is `strength` (default 1: one unit
multiplies an edge's odds by `e`), which the 2NRL / feedback endpoints, `--strength` and the panels pass through
(RadixNet accepts and ignores it).

`beam.py` - `beam_predict(graph, start_node, start_offset, min_chars, k, beam, max_chars, step_penalty, to_end)`
runs two beams over the depth-unrolled graph, each a list of distinct partial paths kept in a parent-pointer table:
the *top* beam keeps the `beam` cheapest partial paths per step, the *bottom* beam the `beam` dearest. A path is
complete at END, at `min_chars` emitted characters (unless `to_end`) or at a cap; the `k` cheapest and the `k`
dearest complete paths are collected in bounded heaps, the top side stops early once no partial path can beat its
k-th finished one, and the bottom side excludes anything that is in the top list. With `to_end` and no cap the
bottom side is capped at about twice the longest top path so "least likely" stays comparable instead of cycling
for hundreds of steps. `Prediction(PathResult)` is the best path plus `top`, `bottom`, `k`, `beam`, `mode`;
`CountRewardNet.predict(prefix, length, mode="beam" | "sample", k=5, beam=None, ...)` applies the same partial-trigram
lead and cap rules as `RadixNet.predict` to every returned path, so `result.text` / `full_text` keep working for the
codegen trainer, the Ollama sampler and the evolve loop.

API: `ModelService` holds the active model, a `_parked` dict with the other kinds' models, and `_path_kind`;
`model_path_for(kind)` derives `<stem>.<kind><ext>` for the kinds the server path does not belong to. `GET /api/model`,
`POST /api/model/select {kind}` (parks the active model, restores the parked / saved / fresh one; 409 while a job
runs), `kind` on `POST /api/reset`, `k` / `beam` on `POST /api/predict` (ignored by RadixNet, `mode: "beam"` mapped to
dijkstra there), `strength` on `/api/2nrl` and `/api/feedback`, `reward` on `/api/graph` edges, `kind` / `kinds` /
`model_label` on `/api/status`. CLI: `--kind radix|count` (new models; `model.count.json` default), `predict --k /
--beam / --mode beam` with top / bottom tables, `--strength` on `2nrl` / `feedback`, `info` shows the kind and reward
totals. Frontend: `ModelSelector` in the header (bound to `status.kind`, `POST /api/model/select`), the status bar
shows the kind and reward totals, the Predict tab shows K / beam fields and the two tables, the Train tab hides the
rate fields, Generate and 2NRL get a strength field.

## 20. ZIP uploads (`archive.py`) — archives unpacked on the server

`extract_texts(data, archive_name, max_entries=10_000, max_bytes=256 MiB) -> (extracted, skipped)` opens the bytes
with `zipfile`, refuses archives over the entry limit or whose declared sizes exceed the byte limit (and counts the
bytes actually read against the same limit, so a lying header cannot get past it), and walks the entries: directories,
`__MACOSX` metadata, `._*` / `.DS_Store` / `Thumbs.db`, nested archives, encrypted entries, binary content (a NUL byte
in the first 8 KB), unreadable and empty entries are reported as `Skipped(path, reason)`; the rest become
`Extracted(name, path, text, bytes)` with the text decoded as UTF-8 (BOM dropped, undecodable bytes replaced).
`flat_name(archive, path)` builds `<archive stem>__<dir>__<file>` from safe characters only - an entry path never
touches the file system - and shortens over-long names to 128 characters with the extension kept and a short hash of
the full path appended. `is_zip(name, data)` decides by the magic bytes alone, so a `.zip` that is really text is
stored as text and a ZIP under any name is unpacked.

API: `Fields.upload_files()` now yields text (`content`) or bytes (`content_base64`, multipart parts, raw bodies);
`ModelService.upload_bytes` routes bytes to `store_archive` (validate, then keep the `.zip` as a single upload - the
upload directory never holds unpacked entries) or to a text upload. Every consumer of uploads goes through
`upload_entries(name) -> [(file, text)]`, which unpacks an archive in memory on demand: `upload_texts` (train / 2NRL /
evolve / feedback files), `read_upload`, and the codegen `problem_files` reader (each entry parsed by its own
extension). The listing (`_record`) reports an archive's text-entry count, ignored entries and summed lines from a
per-file-version cache (`_archive_cache`, keyed by size + mtime, seeded at upload time). CLI: `read_texts` (every
`--data` / `--good` / `--bad` / `--corpus` file) unpacks a ZIP in memory, one text per line or - with `--whole-file` -
one per entry. Frontend: the upload picker sends `.zip` files as multipart bytes (`api.uploadFile`), lists the
archive as one row with a `ZIP · N files` badge and reports what it holds and what was ignored.
