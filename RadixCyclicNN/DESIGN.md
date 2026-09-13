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
| The negative network | `NegativeNet` (section 24): a copy of the network keeping only its negative portions - every node and edge exists because something went wrong there, every edge keeps `blame`, `fails`, `clear` and the tutor's `reasons`, and the net evidence against it is `max(0, blame - clear)`. Trained on negative data only; `judge(text)` says how much of a text is built out of known failure, which reasons that blame carries and which fragments carry it. |
| The negatives come from the tutor | `blame.py` (section 24.2): the English tutor's marked mistake and correction, the Ollama reviewer's critique and rating, the code sandbox / style checker / judge, the evolve discriminator and a person's thumbs down become faults `(text, reason, severity, note, correction)`. The negative network never invents a failure. |
| A correction blames only what changed | `NegativeNet.correct(wrong, right)` (section 24.2): the same character alignment the count model's `correct` teaches from (`diff.py`) decides which steps are blamed - the words the teacher kept carry no verdict, and a blamed transition is never compressed away. |
| The pair as a GAN at output time | `NegativeFilter` (section 24.3): the positive model over-samples candidates, the negative one vetoes them by blame or by the likelihood ratio `log P_negative - log P_positive`; what survives is returned ranked, what does not comes back with its reason. |
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
    negative.py             NegativeGraph, NegativeNet - the negative network: the failures, and why (section 24)
    blame.py                the tutors' verdicts -> faults for the negative network (section 24.2)
    duo.py                  FilterConfig, NegativeFilter - the pair as a GAN at output time (section 24.3)
    dialogue.py             Turn, converse - the model conversing with itself (section 22)
    speech.py               teaching by talking: transcription, the waveform as text, the unique token (section 25)
    schedule.py             learning-rate schedules as graph functions of the epoch (section 18)
    gan.py                  Evolver, EvolveConfig (GAN-style self-upgrade loop)
    checkpoint.py           CheckpointManager
    bench.py                benchmarks (chars/sec, predictions/sec)
    llm.py                  provider layer: LLMClient / LLMError, make_client("ollama" | "chatgpt") (section 16)
    ollama.py               Ollama client, prompt-driven corpora, adversarial review (section 16.1)
    chatgpt.py              ChatGPT client over OpenAI's chat-completions API (section 16.2)
    tutor.py                automated English lessons: exercise -> completion -> grade -> 2NRL, and the lesson
                            plan the report card at the end calls for (section 16.3)
    diff.py                 character diff of a sentence against its correction (section 16.4)
    codegen.py              code generation with a sandbox, an LLM judge and 2NRL rewards (section 17)
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

```python
def repair_base64(body: str) -> tuple[bytes, bool]   # (payload, repaired)
```

The tail shared by the *media* text formats (`img:...`, section 21, and `aud:...`, section 25): a base64 payload the
network **predicted** is rarely clean, so characters outside the alphabet are dropped, a single dangling character
goes with them, the padding is completed and `repaired` reports whether any of that changed the text. The payload is
returned as it decodes; each format pads or truncates it to the length it needs.

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
        #       emitted characters, the whole cheapest path is returned) | "beam" (beam.beam_predict with the same
        #       goal: a Prediction = the best path plus the k most / least likely continuations, `k=5`, `beam=None`)
        #       | "sample" (walks until END or length chars)
        # PathResult.text is the CONTINUATION only; result.full_text = prefix + text.
        # "beam" and "sample" run through GraphModel._search (shared with CountRewardNet.predict); "dijkstra" is
        # the exact single-path search.

    def generate(self, max_length: int = 60, mode: str = "sample", temperature: float = 1.0, count: int = 1,
                 seed: int | None = None, prefix: str = "", step_penalty: float = 0.0,
                 beam: int | None = None) -> list[PathResult]
        # The prediction search used for generation: from (START, 0) or from where `prefix` ends (_prefix_start),
        # decode with include_context=True.  mode "beam": predict(prefix, length=0, mode="beam", k=count,
        # to_end=True, max_length) -> its `top`, the `count` most likely DISTINCT complete texts, most likely
        # first; "dijkstra": the single cheapest complete text; "sample": `count` stochastic walks.  Every result
        # is a whole text: text == full_text == prefix + continuation (`_whole_text`).

    def score(self, text: str) -> dict
        # {"log_prob", "per_char": log_prob / max(1, len(text)), "chars", "transitions", "unknown_transitions"}
        # Sum of log child_probs for every transition along the text's path (START -> ... -> END), following
        # compressed nodes (in-node steps cost 0). Unknown trigram / missing edge -> log(1e-6), counted as unknown.

    def invert(self) -> None              # graph.invert()
    def compress(self) -> int             # graph.compress()

    def two_nrl(self, bad: list[str], good: list[str], neg_epochs=3, pos_epochs=3, neg_lr=0.05, pos_lr=0.01,
                progress=None, checkpoint_manager=None, stop_event=None,
                bad_weights: list[float] | None = None, good_weights: list[float] | None = None) -> dict
        # phase 1: train(bad, epochs=neg_epochs, lr=neg_lr)                      -> "negative" records (phase="negative")
        # phase 2: invert()
        # phase 3: train(good, epochs=pos_epochs, lr=pos_lr, act_lr=pos_lr / 10) -> "positive" records (phase="positive")
        # records passed to progress carry "phase". returns {"negative": [...], "positive": [...], "inverted": graph.inverted}
        # meta["twonrl_runs"] += 1
        # bad_weights / good_weights (>= 0, one per text) make each phase a *rating* rather than one rate for all:
        # _weight_groups groups texts of equal (3-decimal) weight, heaviest first, drops zeros, and runs one pass per
        # group with lr * w (and act_lr * w; the positive phase pos_lr/10 * w). Those records carry "weight".
        # How bad a failure was decides how hard it is pushed away, how good a text was how much of it is kept.

    def reward(self, texts, *, epochs=3, lr=0.1, weights=None, ...) -> list[dict]
        # thumbs up: a positive-phase pass (act_lr = lr / 10); weights turn it into a mark out of 1 per text
    def punish(self, texts, *, epochs=2, lr=0.5, weights=None, ...) -> list[dict]
        # thumbs down: a negative-phase pass, then invert(); weights rate the failures the same way

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

### 9.1 Failure handling: train on failures, blatantly fail on purpose, then invert

`EvolveConfig.blatant_mode` (`none | fail_invert | activation | state`), `blatant_margin` (1.0) and `blatant_boost`
(4.0). Each generation the discriminator's per-char scores give every fake a gap `g = real_mean - score`; fakes with
`g > 0` are failures, those with `g > blatant_margin` are blatant.

* `fail_invert` - `RadixNet.two_nrl(bad=failures, good=real, bad_weights=[min(boost, 1 + g / margin)])`: the
  negative phase runs one pass per distinct weight (heaviest first) with `lr = w * neg_lr` and
  `act_lr = w * TrainConfig.act_lr`, so the worse a failure the harder the model is trained to reproduce it (the
  activation parameters `a, b, h, k` included); records carry `"weight"`. Then the ordinary `invert()` and the
  positive pass on real texts. No failures: `reward(real)` only, nothing inverted. `CountRewardNet.two_nrl` scales its
  penalties by the same weights.
* `activation` / `state` - `model.invert_paths(failures, mode, amounts=[min(1, g / (2 * margin))])`:
  `GraphModel._paths_of` walks (registering if needed) each text; because an edge score `w * f_p * f_c` only changes
  sign when exactly one endpoint changes, every *other* node of the path is chosen (the parity that covers the most
  edges, a shared node takes the largest amount) and `RadixCyclicGraph.flip_nodes({node: amount}, mode)` applies
  `value *= 1 - 2 * amount` to `a` (or `z`): 1 = sign flip, 0.5 = zero, less = attenuation; `version` bumps so the
  cost cache refreshes. Blatant fakes leave the 2NRL garbage set (`worst`); an empty set means the positive pass only.
  The count model penalises the path's edges by `2 * strength * amount` instead.

Generation records add `failures`, `blatant`, `flipped` (nodes / edges changed by the local variant), `boost_mean` /
`boost_max` (weights, or amounts) `twonrl` and `mode`; the CLI table shows `fails / blatant / boost`, the Evolve tab a
select with the four modes, the margin and the max boost, and the same columns.

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
| `predict` | `--prefix TEXT`, `--length N`, `--mode dijkstra\|beam\|sample`, `--k`, `--beam`, `--to-end`, `--step-penalty`, `--temperature` | prints continuation + full text + cost + path; beam: top / bottom tables |
| `generate` | `--count`, `--max-length`, `--mode beam\|sample\|dijkstra`, `--prefix TEXT`, `--temperature`, `--step-penalty`, `--beam` | prints samples (#, cost, probability, reached END, text) |
| `score` | `--text` or `--data FILE` | log-prob per text |
| `converse` | `--opening TEXT`, `--turns 6`, `--mode beam\|sample`, `--context 12`, `--max-length 60`, `--k 5`, `--beam`, `--temperature`, `--step-penalty`, `--speakers A,B`, `--partner FILE`, `--allow-repeats` | the model talks to itself (section 22); prints `speaker: text` lines with cost, probability, the picked-up words and flags (given / new topic / repeat); JSON: `turns`, `count`, `speakers`, `mode`, `opening`, `kind`, `partner_kind`, `transcript` |
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
| POST `/api/predict` | `{"prefix","length","mode","to_end","step_penalty","temperature"}`; `mode: "beam"` (both models): `k`, `beam` | `{"prefix","continuation","full_text","cost","step_costs","path","node_ids","expanded","reached_end"}`; beam: plus `top`, `bottom`, `k`, `beam`, `mode` |
| POST `/api/generate` | `{"count","max_length","mode": "beam"\|"sample"\|"dijkstra","prefix","temperature","step_penalty","beam","seed"}` | `{"samples": [{"text","full_text","cost","probability","path","node_ids","step_costs","reached_end"}]}` — beam: the `count` most likely complete texts from the prediction search |
| POST `/api/converse` | `{"opening","turns","mode","context","max_length","k","beam","temperature","step_penalty","seed","speakers","history","partner","avoid_repeats"}` | `{"kind","partner","speakers","count","turns": [Turn.to_dict()]}` — `partner` names another kind kept in memory (400 when it is not loaded); `history` continues a conversation and only the new turns are returned (section 22) |
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
| POST `/api/uploads` | JSON `{"name","content"}` / `{"name","content_base64"}` or `{"files": [...]}`; or `multipart/form-data` (every part with a filename, as bytes); or any other body with `?name=<file>` (raw bytes) | 201 `{"uploads": [record + "replaced": bool], "archives": [...]}`. Bytes that start with the ZIP magic are validated by `archive.extract_texts` (section 20) and stored as one `.zip` upload whose record carries `archive: true`, `files` (text entries), `skipped` and the summed `lines` / `chars`; `archives[]` summarises entries / extracted / skipped (path + reason); 400 for a corrupt archive, one without a text entry, (no size limit: neither the body limit nor an entry limit applies to uploads). Names are reduced to a safe base name (no traversal); UTF-8 with BOM dropped; same name replaces the file |
| POST `/api/uploads/delete` | `{"name"}` | `{"deleted": name}` (404 when missing) |
| GET `/api/speech` | | `speech.describe()`: the transcription backends and which one `auto` picks, ffmpeg, the microphone recorders, the codecs, the token and the text format (section 25) |
| POST `/api/speech/transcribe` | audio bytes (a binary route like `/api/uploads`: multipart, a raw body named `speech`, or JSON `content_base64`) + `backend`, `language`, `asr_model`, `asr_url`, `transcript` from the query string or the JSON body | `{"transcript", "backend", "model", "language", "seconds", "name"}` |
| POST `/api/speech/teach` | the same audio forms + `transcript` (what the browser dictated), `rate`, `codec`, `normalise`, `waveform`, `pair`, `token`, `unique`, `backend`, `language`, `asr_model`, `asr_url`, `train`, `epochs`, `lr`, `batch_size`, `save_as` | `{"token", "transcript", "asr", "audio", "texts", "chars", "pair", "name", "upload", "job"}`; 202 with a train job on the texts, `save_as` keeps them as one upload |
| POST `/api/speech/decode` | `{"text", "codec"}` | `{"wav_base64", "codec", "rate", "channels", "samples", "seconds", "bytes", "repaired"}` — an encoded *or predicted* waveform as playable audio |

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
* `PredictPanel.jsx` — prefix, length, mode (dijkstra / beam / sample; the count model's dijkstra is the beam search), K / beam width for beam, to-end, step penalty; shows continuation (prefix + highlighted continuation), cost, probability, path chips with per-step costs, and the top-K / bottom-K tables of a beam prediction (both models). A Like button (on the result and on every top / bottom row) rewards that text: `POST /api/feedback {good: [prefix + continuation]}` through the shared `useJob("feedback")` hook, i.e. `reward()` - a positive-phase pass for RadixNet, a traversal plus reward for the count model; the button shows the liked state and cannot reward the same text twice.
* `GeneratePanel.jsx` — prefix, count, max length, mode (beam = the K most likely complete texts from the prediction search, the default; sample; dijkstra), temperature; list of samples with cost and probability, thumbs up / down per sample (`RateButtons`).
* `ConversePanel.jsx` — the model talks to itself (section 22): opening line, turns, context, max length, mode (beam / sample), K, temperature, the two voices' names, "Second voice is" (the same model, or the other kind kept in memory - `GET /api/model` `in_memory`); Start / Start over runs `POST /api/converse`, Continue sends the transcript as `history` and appends the new turns, Clear empties it. The chat view puts the first voice left and the second right, dims the picked-up context inside each bubble, shows cost / probability / skipped candidates and badges (given, new topic, repeat), and every turn has the thumbs.
* `RatingsCard.jsx` — shared by Generate and Converse: `useRatings()` (one rating per distinct text, toggling), `RateButtons` (the thumbs pair) and the "Ratings → 2NRL" card (rated texts, the action that will run, epochs / learning rates / strength, Train on ratings → `POST /api/feedback` through the panel's `useJob("feedback")`, the job's phase table).
* `TwoNRLPanel.jsx` — bad textarea, good textarea, epochs/lrs; shows negative/positive losses; button to Invert manually.
* `EvolvePanel.jsx` — corpus textarea, samples, generations (blank = forever), start/stop; live SVG line chart of `gap` and `fake_score_mean` over generations + latest sample text.
* `CheckpointPanel.jsx` — list checkpoints, save checkpoint (tag), restore, save/load model path, reset.
* `GraphView.jsx` — SVG rendering of `/api/graph` (circular layout, edge opacity by prob, node radius by count, hover label).
* `ScorePanel.jsx` — score a text.
* `SpeechPanel.jsx` + `src/audio.js` — teaching by talking (section 25): the browser records the microphone
  (`MediaRecorder`) and dictates the words at the same time (`SpeechRecognition`, the Web Speech API); `audio.js`
  decodes the recording with the Web Audio API, mixes it to mono, resamples it to 16 kHz and writes a 16-bit PCM WAV,
  so the server reads it with the standard library alone and never needs ffmpeg for a browser recording. The panel
  shows the clip with a player and a peak meter, the transcript in an editable box (corrected by hand, or left empty
  for the server's Whisper), the waveform rate / codec / pair / unique-token / normalise switches and the training
  settings; "Teach the model" posts `/api/speech/teach` with `train`, "Preview the texts" the same request without it,
  and the result card shows the token, the ASR backend, the waveform's size and every text. A third card decodes any
  `aud:` text - including a prediction - back into audio through `/api/speech/decode` and plays it.

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
* `test_api.py` — server in a thread; health/status/train(job polling)/predict/generate/converse/score/2nrl/invert/compress/save/load/checkpoints/graph/evolve start-stop/static fallback.
* `test_negative.py` — the negative network (section 24): the blame weight function, evidence as blame minus clearing, blaming / clearing / two_nrl / invert_paths, corrections (only the changed characters blamed, nothing correct created), `judge` (risk, peak, coverage, reasons, spans, the thresholds), `crossings`, prediction over the failure distribution, `forget`, the capped per-edge reasons and journal, persistence and the kind registry, the `/api/negative/*` routes and the CLI's `negative` group.
* `test_blame.py` — where the negatives come from (section 24.2): reason classification from a critique, severity from a rating, code reasons from the sandbox / style / judge, faults from the English tutor's lessons (the named mistake, the mark, the correction), from reviews and from attempts, `teach`, and the tutor / evolve / codegen hooks.
* `test_duo.py` — the pair (section 24.3): the blame, peak and ratio rules with the coverage gate, strict, learn, `filter` / `generate` / `predict`, the count model as the positive half.
* `test_speech.py` — the text format (packing, the header found behind a token and before a transcript, repair of a
  cut-off prediction), unique tokens, `speech_texts`, both codecs (mu-law beats linear 8-bit on quiet audio, byte
  round trips), resampling / downmixing / normalising, every WAV sample format the reader accepts (PCM 8/16/24/32,
  IEEE float 32/64, A-law, mu-law, WAVE_FORMAT_EXTENSIBLE) and the ffmpeg hand-off, the waveform round trip
  (encode → text → WAV → the same tone), transcription (a given transcript, the auto order, fakes with the
  openai-whisper / faster-whisper / OpenAI-server interfaces, the error when nothing is installed), `teach` (both
  texts share the token, pair / shared token / transcript-only, a failing ASR still teaches the waveform, the model
  learns and continues what was spoken), recording (no recorder, `arecord` called and its file read), the four API
  routes in all three body forms and the CLI actions. Runs without a microphone, ffmpeg or any ASR backend.
* `test_dialogue.py` — `tail_context`, `converse`: alternating speakers, the opening as a given turn, every reply picks up (a whole-word part of) the previous line, no repeats / echoes in beam mode, determinism, history continuation, seeded sampling, speakers and a partner model, repeats on request, the empty model, validation.
* `test_tutor.py` — a fake Ollama plays the English teacher: `cue` / `overall_score` / the error-type mapping / the report card; the tolerant exercise and grade parsers; the marking (batches, an empty completion failed without a call, an unreadable answer left unrated); the loop over a real model and over a scripted one (what reaches the graph: corrections taught from their diff, weighted garbage for the rest and the mark-weighted rewards), adapting to the weakest points, drills, the dry run, per-lesson learning, the stop event, both model kinds; the next lesson plan (the weak points of a card, the upgrade ladder and the brief the marks write, the plan the marks alone imply, the tolerant plan parser, the teacher's plan merged with it - its brief kept, its difficulty ignored - a run that ends with one and a run taught to one); the five endpoints and the CLI.

---

## 15. Performance notes (must be followed)

* Hot loops (backend step, Dijkstra) bind `math.sin`, `math.cos`, `math.exp`, `math.log` and list references to locals; no attribute lookups or method calls per element; no per-edge object allocation.
* Graph storage is flat parallel lists; edges live in per-node dicts for O(1) lookup, exported to CSR per epoch.
* `child_costs` is cached per node and invalidated by `graph.version`.
* Dijkstra uses `heapq` with `(cost, tie_counter, node_id, chars)` tuples and a `best` dict; `max_expansions` guard.
* Training re-uses transition arrays across epochs unless `structure_version` changed.
* `bench` reports numbers; `PythonBackend` should reach roughly 100k+ transitions/sec on a modern CPU for small fan-outs.

---

## 16. LLM providers (`llm.py`, `ollama.py`, `chatgpt.py`) — a local model or ChatGPT

Everything that asks an LLM a question — the code-generation tutor and its judge, the corpus and review helpers —
takes a *client* and never looks at which provider it got:

```python
PROVIDERS = ("ollama", "chatgpt"); DEFAULT_PROVIDER = "ollama"
class LLMError(Exception)             # the base of OllamaError and ChatGPTError: "the LLM did not work"
class LLMClient(Protocol)             # .provider .url .model; .models() .available() .generate(...) .chat(...)
normalise_provider(name) -> str       # "" -> "ollama"; aliases "openai" / "gpt" / "chat-gpt" -> "chatgpt"; else ValueError
default_model(provider) / default_url(provider) -> str
make_client(provider=None, url=None, model=None, timeout=None, api_key=None) -> LLMClient   # lazy imports
provider_of(client) -> str            # the client's own label, "ollama" for anything unlabelled
loads_lenient(raw) -> Any             # JSON from an LLM answer: tolerates code fences and prose around the object
```

### 16.1 Ollama (`ollama.py`) — prompt-driven corpora and an adversarial LLM review

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

### 16.2 ChatGPT (`chatgpt.py`) — the hosted alternative, same interface

```python
DEFAULT_URL   = $OPENAI_BASE_URL (or $OPENAI_API_BASE) or "https://api.openai.com/v1"
DEFAULT_MODEL = $RADIXNET_OPENAI_MODEL or "gpt-4o-mini"
DEFAULT_TIMEOUT = 120.0

normalise_url(url)                    # "host" -> "https://host/v1"; refuses a plain-http remote host unless
                                      # RADIXNET_OPENAI_ALLOW_INSECURE is set (a key must not travel in clear)
api_key(explicit=None) -> str | None   # explicit, else $OPENAI_API_KEY, else the content of $OPENAI_API_KEY_FILE
api_key_configured(explicit=None) -> bool                      # never returns the key itself
class ChatGPTError(LLMError)          # .status .code .param from the API's {"error": {...}} body
class ChatGPTClient(url=None, model=None, timeout=None, *, api_key=None)
    .provider = "chatgpt"; .url .model .timeout .configured
    .models() -> [{"name","id","owned_by","created"}]          # GET /models, sorted by name
    .available() -> bool                                       # false without a key: nothing is sent
    .generate(prompt, *, system=None, model=None, json_mode=False, options=None, timeout=None) -> str
    .chat(messages, *, ...) -> str                             # POST /chat/completions, non-streaming
```

* The key is read per request and never stored in a config, a job record, a report or `repr`; the HTTP API has no
  field for it (the server uses its own `$OPENAI_API_KEY`).
* Ollama's `options` are translated (`num_predict` / `max_tokens` -> `max_completion_tokens`, `temperature`, `top_p`,
  `seed`, `stop`, the penalties; unknown names dropped); `json_mode` sends `response_format: {"type": "json_object"}`.
* A 400 that blames an optional field (`unsupported_parameter` / `unsupported_value` / "Unrecognized request
  argument", by `param` or by name in the message) drops that field and asks again, so the reasoning models (no
  `temperature`) and older ones (no `response_format`) work without configuration. Every field is dropped at most
  once, so the loop terminates.
* `available()` / `models()` also accept any OpenAI-compatible server through `$OPENAI_BASE_URL`; a loopback endpoint
  is never reached through the environment's proxy, the hosted API always is.
* `serve --chatgpt-url --chatgpt-model` set the API defaults; `/api/status` gains
  `"chatgpt": {"url","model","configured"}` and `GET /api/chatgpt/models?url=` answers 200 always with
  `{"available","configured","url","model","models","error"}`.
* CLI: `radixnet [globals] chatgpt [--url URL] [--chatgpt-model NAME] [--timeout S] models|ask --prompt TEXT
  [--system TEXT] [--temperature 0.7] [--json]`.
* Docker: the API container takes `OPENAI_API_KEY`, `RADIXNET_OPENAI_MODEL` and `OPENAI_BASE_URL` (empty key =
  ChatGPT tutoring stays off and the Code tab says so).
* Tests (`tests/test_chatgpt.py`) use a fake OpenAI server (stdlib `http.server`) answering `/v1/models` and
  `/v1/chat/completions`, recording the Authorization header, scripting tutor programs, judging by content and
  rejecting configurable fields or whole requests.

### 16.3 The tutor (`tutor.py`) — the teacher sets the exercise, the network answers, the teacher marks it

The prediction process run without a human: `topic -> prefix (LLM) -> completion (the prediction search) -> grade
(LLM) -> 2NRL`. Any provider client, one more loop: the teacher is a local Ollama model or ChatGPT
(`tutor_provider`), and the marking follows it unless `grader_provider` names the other one.

```python
DEFAULT_TUTOR_MODEL = $RADIXNET_TUTOR_MODEL or DEFAULT_MODEL (ollama.py)
default_tutor_model(provider) -> str      # DEFAULT_TUTOR_MODEL for ollama, the provider's own default otherwise
ERROR_TYPES = ("none", "agreement", "tense", "article", "preposition", "plural", "pronoun", "word-order",
               "spelling", "punctuation", "vocabulary", "fragment", "nonsense")   # "other" for anything else
MODES = ("dijkstra", "beam", "sample");  TWONRL_PER = ("round", "lesson");  TEACHER_WEIGHT = 1.0

def cue(prefix) -> str                    # the prefix with exactly one trailing space: what the search is given
def overall_score(grammar, spelling, fluency, grammar_weight=0.6) -> float | None
                                          # grammar_weight * grammar + (1 - grammar_weight) * mean(spelling, fluency)
                                          # missing sub-marks drop out; grammar is taught, so grammar dominates

@dataclass Exercise:  id, prefix, focus, answer        # .cue == cue(prefix)
@dataclass Grade:     score, grammar, spelling, fluency, passed, error, correction, comment, graded_by
                      # graded_by: the marking provider ("ollama" | "chatgpt"), "empty" or "unrated"
@dataclass Lesson:    exercise, attempt, mode, continuation, sentence, cost, probability, reached_end, seconds, grade

def write_exercises(client, topic, count=5, *, focus=None, level="beginner", weak=(), words="3 to 6", model=None)
    # JSON mode: {"exercises": [{"prefix", "focus", "answer"}]}; parse_exercises tolerates bare lists, plain lines,
    # "opening" / "stem" instead of "prefix", trailing punctuation, duplicates and an answer without the prefix
def drill_sentences(client, topic, count, *, weak=(), model=None) -> list[str]   # extra correct examples to imitate
def grade_completions(client, lessons, *, topic, threshold=6.0, grammar_weight=0.6, model=None, batch=10,
                      external=None, graded_by=None)   # graded_by: the provider behind the client
    # one JSON call per batch: {"grades": [{"index","grammar","spelling","fluency","error","correction","comment"}]}
    # empty completion -> failed without asking (graded_by="empty", the exercise's own answer as the correction);
    # unreadable answer -> score None, graded_by="unrated", counted as a failure
def report_card(lessons) -> dict          # {"lessons","graded","passed","failed","pass_rate","mean_score",
                                          #  "mean_grammar","mean_spelling","mean_fluency","errors","weakest"}
```

The report card at the end of a run is handed back to the teacher, which turns it into the syllabus of the
lessons that follow:

```python
ERROR_FOCUS = {"agreement": "subject-verb agreement", "tense": "verb tenses", ...}   # what fixes each mistake
LEVELS = ("beginner", "intermediate", "advanced");  DEFAULT_PLAN_LESSONS = 3
WORDS_LADDER = ("3 to 6", "5 to 8", "7 to 12", "10 to 16");  UPGRADE_STEPS = ("hold", "stretch", "advance")
STRONG_PASS_RATE = 0.8;  STRONG_SCORE = 8.0;  STRETCH_PASS_RATE = 0.5;  HOLD_DRILLS = 3

@dataclass PlannedLesson:  focus, topic, why, targets, exercises, drills, prefixes
@dataclass LessonPlan:     lessons, summary, prompt, upgrade, level, topic, weak, source   # .targets: what it drills

def focus_for(error) -> str               # ERROR_FOCUS[_error_type(error)], "" for "none" and the unknown
def weak_points(card, limit=3) -> list    # [{"error","count","share","focus"}] worst first, ties alphabetically;
                                          # reads a card that went through JSON (float counts, the teacher's wording)
def next_level(level, card) -> str        # one step up LEVELS when the card is strong, else level
def next_words(words, steps=1) -> str     # one rung up WORDS_LADDER; a setting off the ladder is left alone
def upgrade_from_card(card, *, level, words, threshold, drills) -> dict
    # {"step", "level", "words", "threshold", "drills", "note"} - the incremental upgrade, read off the marks:
    # "advance" (level up, longer openings, pass mark +1 capped at 9) at STRONG_PASS_RATE / STRONG_SCORE,
    # "stretch" (longer openings) at STRETCH_PASS_RATE, else "hold": nothing harder, HOLD_DRILLS correct
    # sentences to imitate.  A student who is failing never gets a harder exercise.
def plan_brief(weak, upgrade, topic="") -> str   # "Drill a, b and c, worst first: ... <note> Keep the sentences
                                          # about the sea." - the brief the marks alone write for the next batch
def card_lines(card) -> list[str]         # the card as the teacher reads it (both languages word for word)
def plan_from_card(card, *, topic, level, words, threshold, count=3, exercises=5, drills=0) -> LessonPlan
    # no LLM: one lesson per weak point, worst first, why = "5 of 10 lessons (50%) were marked down for agreement.",
    # plus the upgrade and plan_brief; a card with nothing to fix plans one lesson that keeps the topic
def plan_lessons(client, card, *, topic, level, words, threshold, count=3, exercises=5, drills=0, model=None)
    # JSON mode: {"summary", "prompt", "lessons": [{"focus", "targets", "topic", "why"}]}; parse_plan tolerates
    # {"plan": [...]}, bare lists, plain lines, "point" / "skill" instead of "focus", "reason" instead of "why",
    # "brief" / "instructions" instead of "prompt", and a "targets" in the teacher's own words (_error_type maps
    # it back; a focus alone implies it).  plan_from_card is the floor: a weakness the answer skips takes the
    # place of a lesson that drills nothing the card marked down (appended when there is none), a brief it does
    # not write is the one the marks wrote, and an answer that cannot be read - or that names no mistake and says
    # nothing about the student - leaves that plan standing.  How much harder the next batch gets is never read
    # from the answer: upgrade_from_card decides it and the teacher is asked to repeat it in the brief, so the
    # settings a plan carries are the ones the card earned.  `source`: the provider, or "report card".  A card of
    # no lessons is a ValueError.
```

`LessonPlan.prompt` is the point of the plan: `TutorConfig.brief` (`--brief TEXT`, `{"brief": ...}`) hands it to
`write_exercises` with **every** round of the run that follows (`"The plan for this batch of lessons: ..."` above
the topic, the focus and the weak points), so one batch's report card is the next batch's instructions.

`TutorConfig` (validated like every other config) holds the topic, `rounds`, `exercises`, `attempts`, `focus`,
`level`, `words`, `brief` (the previous batch's plan), the two providers and their model names (`tutor_provider`, `tutor_model`, `grader_provider`,
`grader_model`; `__post_init__` normalises the providers and fills in the models they imply, and
`resolved_grader_model` is what the marking runs on), the completion settings (`mode`, `length`, `max_length`, `temperature`,
`to_end`, `beam`), the marking settings (`threshold`, `grammar_weight`, `batch`, `adapt`, `drills`, `plan`, `teach_answer`,
`learn`), how a correction is taught (`diff_corrections`, `keep_weight`) and the 2NRL settings (`twonrl_per`,
`min_weight`, `neg_epochs`, `pos_epochs`, `neg_lr`, `pos_lr`, `batch_size`, `strength`, `replay`, `replay_limit`,
`checkpoint_every`).

`TutorTrainer(model, client, config, external=None, grader_client=None).run(...)` per round — `client` is the
teacher, the marking shares it unless `grader_client` is given or `grader_provider` names the other provider (then
one is built from the environment), and every grade records which one marked it:

1. `set_exercises` — the teacher writes the openings (with the previous round's weakest points as the syllabus when
   `adapt`).
2. `complete` — `model.predict(exercise.cue, ...)` per attempt (attempt 0 in `mode`, later ones sampled).
3. `grade` — one call per `batch` sentences.
4. `corrections_of` + `texts_of` + `learn` — a failure the teacher corrected is a `Correction(wrong, right,
   weight_of(grade))` taught by `model.correct` (section 16.4), not a whole sentence in `bad`; `weight_of(grade) =
   min_weight + (1 - min_weight) * (threshold - score) / threshold` (1 for an unrated one). What is left is the old
   split: `bad` = the failures with no correction to align, `good` = the sentences that passed with
   `reward_of(grade) = score / 10`, plus the model answers and the drill sentences at `TEACHER_WEIGHT`. A text
   offered twice keeps its largest weight. Nothing is punished when the network wrote nothing (the prefix itself is
   correct English). Both sides go into `two_nrl(bad_weights=, good_weights=)`, or `reward(weights=)` /
   `punish(weights=)` when only one side exists, and `action` names what ran ("correct+2nrl"). The corrections join
   the replay buffer like any taught text. `diff_corrections=False` goes back to the whole-sentence way;
   `learn=False` reports what it would have taught and touches nothing.

Records: `{"kind": "lesson", round, exercise, prefix, focus, attempt, mode, continuation, sentence, score, grammar,
spelling, fluency, passed, error, correction, changes, comment, graded_by, probability, seconds}` (`changes` is what
the teacher changed, span by span, and rides on the `Lesson` itself so a dry run carries it too), `{"kind": "round",
...}` (the report card plus `action`, `bad`, `good`, `corrections`, `edits`, `penalised`, `rewarded`, `neg_loss`,
`pos_loss`, `mean_weight`, `mean_reward`, `drills`) and a final `{"kind": "report", rounds, ...}`.  With
`plan = N` (`--plan N`, `POST /api/tutor/start {"plan": N}`) one more record closes the run:
`{"kind": "plan", rounds, ...LessonPlan.to_dict()}`, from `TutorTrainer.plan(card=None, count=N)` — the run's own
report card unless one is given.  A teacher that cannot plan costs only a `note`; the lessons stand.

With `--blame` (`TutorTrainer(negative=...)`, `POST /api/tutor/start {"blame": true}`) every failed sentence of a
round also teaches the **negative network** (section 24) why it failed: the mistake the teacher named is the reason,
its mark the severity, its sentence of teaching the note, and the correction is diffed so only the characters it
changed are blamed. The round records then carry `negative_blamed`, `negative_edges` and `negative_reasons`.

CLI `radixnet tutor` prints one row per marked sentence (round, exercise, score, grammar, spelling, fluency, mark,
mistake, sentence) with the correction and the teacher's line under a failure, a note per round and a report card at
the end; `--dry-run` marks without training or saving, `--report FILE` writes config, records, lessons, the card and
the plan.  `--plan [N]` (default 3) prints the planned lessons as a table, the step up, the brief, and the
`tutor --brief ...` command that teaches the next batch to it; `--brief TEXT` is the other end of that loop.

CLI flags for the teacher: `--tutor-provider ollama|chatgpt` (`--provider`), `--tutor-model`, `--grader-provider`,
`--grader-model`, `--url`, `--grader-url`; a `chatgpt` teacher without `$OPENAI_API_KEY` stops before anything is
sent.

API: `GET /api/tutor` (defaults, error types, modes, `levels`, `plan_lessons`, and `providers` — each teacher's url,
model and whether it is `configured`), `POST /api/tutor/start` (job), `GET /api/tutor/history`,
`POST /api/tutor/lesson` (one round, no training; `prefixes` skips the exercise writer) and `POST /api/tutor/plan`
(`{report, count, topic, level, exercises, drills}` -> `{plan, source, provider, model, url, report}`; without
`report` the card at the end of the last run — `ModelService.tutor_card()`, the last `"report"` record of the
history — is used, and 400 says so when there is none). All of them take `tutor_provider` / `grader_provider` (and
`url` / `grader_url`); a `chatgpt` teacher on a server without a key is refused with 400. The service releases the model
lock around every LLM call (`pause_lock`), so readers keep being served while the teacher thinks. The default
teacher is `$RADIXNET_TUTOR_MODEL`, else the model the server was started with (ChatGPT: the server's
`--chatgpt-model`).

Frontend: a **Tutor** tab (not Python-only - the Go server serves the same endpoints) with a **Teacher** selector
(Ollama or ChatGPT; the URL, the model placeholder and the notes follow it, and it says so when the server has no
key), the settings ("Teach corrections from the diff" and "Unchanged words keep" among them), "Dry run", a chart of the mean score and grammar
per round, the report card with the mistake histogram, a table of rounds (with what the corrections moved) and one
of every lesson (marks, mistake, what the network wrote, the correction, the changed words struck out against what
replaced them, the teacher's line).  **Plan** (`POST /api/tutor/plan`) sits under each report card - the dry run's
and the run's - and a **Lesson plan** card closes the tab: the summary, the step up (`hold` / `stretch` /
`advance` with its note), the brief for the next batch with **Teach the next batch** (which loads the brief and
the upgraded level, openings, pass mark and drills into the settings above), the weak points as pills, and a row
per planned lesson (the point of grammar, the mistake it fixes, the topic, the exercises, why) with *Use this
lesson*, which loads that one lesson instead.  Both leave **Start lessons** to the user.  The form's **Brief** box
and **Prefix words** field are `brief` and `words`; the **Plan** field is the run's own `plan` setting, so a run
that ends with a `"plan"` record shows it in the same card without asking again.

Tests: `tests/test_tutor.py` (a fake Ollama that writes exercises, marks by a rule, answers drill requests and
plans the next lessons; the parsers, the marking, the planner, the loop with a scripted model, the endpoints and
the CLI), the ChatGPT teacher of
`tests/test_chatgpt.py` (the same lessons against the fake OpenAI, including a ChatGPT teacher marked by a local
model) and `go/radixnet/tutor_test.go` + `go/server/tutor_test.go` for the port.

### 16.4 Learning from a correction (`diff.py`, `CountRewardNet.correct`) — only what changed moves

A grade used to reach the graph as two verdicts on two whole sentences: the attempt was garbage, the correction was
gospel. Most of a corrected sentence is however word for word what the network wrote - the teacher changes a tense,
an article, a plural - so the whole-sentence penalty taxed the trigrams that were right.

`diff.edits(wrong, right)` aligns the two character by character: the shared prefix and suffix are trimmed, the
middle is aligned by a longest-common-subsequence walk (`MAX_CELLS` guards a pathological pair, which then counts as
one change), neighbouring edits of one kind are merged, and an equal run shorter than a trigram between two changes
is swallowed (`MIN_EQUAL_RUN`), because no trigram fits in such a gap - "mat" -> "park", not "m" -> "p" and "t" ->
"rk". `changed_spans` returns the half-open rune ranges each side disagrees on; an insertion is an *empty* span at
the position where the text belongs. The Go port (`go/radixnet/diff.go`) is the same algorithm with the same
tie-breaks, and `tests/test_go_parity.py` compares the two on a set of sentence pairs.

`CountRewardNet.correct(wrong, right, strength=, weight=, reward=, keep=, count=)` turns that alignment into edge
updates. Both sentences join the structure first (observing one can split a node the other's path runs through), and
then every step of a traced path is charged with the characters it *writes*: the first step with the whole of its
node's label, a later one with everything past the two characters it overlaps its parent by, and the step into END
with the position just past the last character - where a sentence that stopped too early went wrong (`_steps_over`).
The steps of `wrong` charged with a changed character lose `strength * weight`; the steps of `right` charged with
one gain `strength * reward` and the rest of the correction `keep` times as much (0 teaches the fix alone, 1 is the
old whole-sentence thumbs up); an edge both sentences walk is rewarded, never penalised. `count` traverses the
correction once, as a training pass does. One `add_reward` call per distinct amount, so the weights are recomputed
three times at most. Returns `{edits, changes, penalised, rewarded, kept, penalty, reward, loss, wrong_chars,
right_chars}`.

CLI `radixnet correct --wrong ... --right ...` (and `radixnet-count correct`) teaches one correction by hand and
`--dry-run` prints the alignment alone. Tests: `tests/test_countnet.py::TestCorrections` and
`go/radixnet/correct_test.go` (the alignment rebuilds both sentences, only the differing steps move, `keep` spreads
the rest, an early end blames the step into END, a run of corrections keeps the graph sound), plus the cross-language
parity case.

---

## 17. Code generation (`codegen.py`) — sandbox, LLM tutor and judge, 2NRL rewards

`problem -> Python program -> sandbox -> judge -> 2NRL`, in two semi-supervised phases over one problem list:

* **teacher**: the tutor (`CodeGenConfig.teacher_provider`, `"ollama"` by default with `teacher_model` `gemma4` / env
  `RADIXNET_CODEGEN_MODEL`, or `"chatgpt"` with that provider's default model, section 16.2) writes a program
  (`teacher_generate`), the sandbox runs it, failures / rejections go back to the tutor (`teacher_fix`, up to
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
default_teacher_model(provider) -> str        # DEFAULT_TEACHER_MODEL for ollama, the provider's own default otherwise
judge_with_llm(client, problem, code, run, style, model=None) -> {"task","pep8","naming","score","issues","critique"}
    # JSON-mode adversarial reviewer prompt with the task, program, execution result and the objective style report
decide(run, style, llm, strictness, judged_by=DEFAULT_PROVIDER) -> Verdict(correct, runs, task, pep8, naming, score,
    issues, critique, judged_by)
    # correct = runs and expected_ok is not False and task is not False and (lenient or pep8 and naming and style.ok)
    # task: False when the program fails or the stdout differs; the LLM's answer when asked; True with matching
    # expected output; None (counts as ok) when nothing can judge it.
    # judged_by: sandbox | tests | ollama | chatgpt | none — an LLM verdict names the provider that gave it
CodeGenConfig(teacher_provider="ollama", teacher_model="", judge_provider="", judge_model=None,
    phases=("teacher","model"), rounds=1, teacher_attempts=3, model_attempts=4,
    first_attempt_dijkstra=True, temperature=1.0, max_length=800, strictness="strict", use_judge=True,
    fallback_teacher=True, twonrl_per="problem"|"round", replay=True, replay_limit=64, teacher_prompt=None,
    model_prompt="{problem}\n", neg_epochs=2, pos_epochs=3, neg_lr=0.5, pos_lr=0.1, batch_size=4, checkpoint_every=0)
    # __post_init__ normalises the providers (ValueError for anything else), fills teacher_model from the tutor's
    # provider when blank and lets judge_provider default to the tutor's, so to_dict() / reports name the real models
    .resolved_judge_model -> str          # judge_model, else the tutor's model (same provider) / that provider's default
CodeGenTrainer(model, client, sandbox=None, config=None, external=None, judge_client=None)
    # client is the tutor (any provider); the judge shares it unless judge_client is given or judge_provider names the
    # other provider, in which case a client for it is built from the environment. Attempts are labelled with
    # provider_of(client), verdicts with provider_of(judge_client), so a ChatGPT tutor can be judged by a local model.
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
model or the teacher, no training), `POST /api/codegen/run` (sandbox + style + verdict without an LLM). `start` and
`solve` take `teacher_provider` / `judge_provider` (and `url` / `judge_url` / `teacher_model` / `judge_model`); a
`chatgpt` tutor on a server without a key is refused with 400 before anything is sent. Frontend: the "Code" tab, whose
**Teacher** selector switches the provider (model, URL, notes and badges follow it). Tests (`tests/test_codegen.py`)
use a fake Ollama whose teacher answers come from a scripted queue and whose judge rejects programs containing
`BAD_ANSWER`; `tests/test_chatgpt.py` runs the same loop against the fake OpenAI server, including a ChatGPT tutor
judged by a local model.

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

`CountRewardGraph(RadixCyclicGraph)` keeps more per-edge lists next to `edge_w` / `edge_count`: `edge_reward` and
`window_edge_count` (traversals inside a sliding window, a deque `_window` of the last `window` edge ids traversed
anywhere in the graph, default 10 000), plus the global `total_traversals`. `observe_sequence(count=True)` is
overridden to `record_traversals` (all time, window, total) and recompute the weights. The weight is the *dual
frequency function* (`edge_weight`): with `C_p` / `W_p` the all-time / windowed traversals leaving the parent over
its `deg` children and `s = 0.5`, `R_all = (count + s) / (C_p + s * deg)`, `R_recent = (window_count + s) /
(W_p + s * deg)`, `weight = count_scale * log1p(count) + global_scale * log(R_all) + window_scale * log(R_recent) +
reward_scale * reward` (defaults 0, 0.5, 0.5, 1: the geometric mean of the two shares, so when they agree the
probability is the share) - the edge's share of its node's traversals, all time and recently, with the rewards. `recompute_weights()` walks every parent once (O(E)); `shares(p)` reports
the two ratios per edge for the graph view; `configure(**scales, window=)` changes the function at run time (the
window is trimmed when shrunk) and `weight_config()` describes it. `to_dict` stores the scales, `total_traversals`
and the window's edge ids (remapped like the edges); `from_dict` rebuilds the window counts from them, and a file
from before the dual function loads with `count_scale = 1, global_scale = window_scale = 0` so it behaves as it did. Every node is created with `a = 0, k = 1`, so the sine activation is the constant 1 and the
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
`GraphModel._search(prefix, length, mode, k, beam, ...)` (model.py) wraps it for both models and applies the same
partial-trigram lead and cap rules as the Dijkstra search to every returned path, so `result.text` / `full_text` keep
working for the codegen trainer, the Ollama sampler and the evolve loop. `CountRewardNet.predict(prefix, length,
mode="beam" | "sample", k=5, beam=None, ...)` is that search ("dijkstra" is an alias of beam there);
`RadixNet.predict(..., mode="beam", k=5, beam=None)` runs it next to the exact Dijkstra mode. **Generation is the
prediction search run to the end of a text**: `GraphModel.generate(mode="beam", count, max_length, prefix="",
step_penalty, beam)` calls `predict(prefix, length=0, mode="beam", k=count, to_end=True, max_length=max_length)`
and returns its `top` - the `count` most likely distinct complete texts, most likely first (with `k=1` it agrees
with Dijkstra's cheapest complete text); "sample" and "dijkstra" run through the same engine, and every generated
result is a whole text (`text == full_text == prefix + continuation`).

API: `ModelService` holds the active model, a `_parked` dict with the other kinds' models, and `_path_kind`;
`model_path_for(kind)` derives `<stem>.<kind><ext>` for the kinds the server path does not belong to. `GET /api/model`,
`POST /api/model/select {kind}` (parks the active model, restores the parked / saved / fresh one; 409 while a job
runs), `kind` on `POST /api/reset`, `k` / `beam` on `POST /api/predict` (`mode: "beam"` for both models), `prefix` /
`step_penalty` / `beam` and `mode: "beam"` on `POST /api/generate` (samples carry `full_text` and `probability`),
`strength` on `/api/2nrl` and `/api/feedback`, `reward` on `/api/graph` edges, `kind` / `kinds` /
`model_label` on `/api/status`. CLI: `--kind radix|count` (new models; `model.count.json` default), `predict --k /
--beam / --mode beam` with top / bottom tables (both kinds), `generate --mode beam --prefix --step-penalty --beam`,
`--strength` on `2nrl` / `feedback`, `info` shows the kind and reward totals. Frontend: `ModelSelector` in the
header (bound to `status.kind`, `POST /api/model/select`), the status bar shows the kind and reward totals, the
Predict tab shows K / beam fields and the two tables for a beam prediction, the Generate tab defaults to beam with a
prefix field, the Train tab hides the rate fields, Generate and 2NRL get a strength field.

## 20. ZIP uploads (`archive.py`) — archives unpacked on the server

`extract_texts(data, archive_name, max_entries=None, max_bytes=None) -> (extracted, skipped)` opens the bytes
with `zipfile` and walks the entries; there is no limit by default (`MAX_ARCHIVE_ENTRIES` / `MAX_UNPACKED_BYTES` are
`None`, and `POST /api/uploads` is exempt from the 64 MB JSON body limit - `MAX_UPLOAD_BYTES` is `None`), so a whole
source tree such as `go-master.zip` with 17 708 entries uploads and trains. A caller that passes limits gets the
declared sizes checked first and the bytes actually read counted against the same limit, so a lying header cannot get
past it. Entries: directories,
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

## 21. Images as text (`vision.py`) — the Stable Diffusion encoder run backwards

`encode_image(data, size=128, encoder="auto") -> {"text", "encoder", "width", "height", "latent_shape", "bytes",
"chars", "source_size"}`: the image (any format Pillow reads) is converted to RGB and resized to `size x size` (a
multiple of 8), the encoder turns it into a latent, the latent is quantised to one signed byte per number and packed
as `img:<encoder>:<w>x<h>:<base64>` (`pack_text` / `parse_text`). `decode_text(text, encoder=None) -> {"png",
"encoder", "width", "height", "bytes", "repaired"}` parses the header, strips junk from the base64 (a predicted text
may contain anything), completes the padding, pads or truncates the payload to the latent size and runs the decoder.

* `SDVaeEncoder` (`"sd"`) - `diffusers.AutoencoderKL.from_pretrained($RADIXNET_SD_VAE)` (default
  `stabilityai/sd-vae-ft-mse`), loaded once per process on first use, on CUDA / MPS when torch sees one. Encode:
  pixels to `[-1, 1]`, `vae.encode(x).latent_dist.mean * scaling_factor` (the deterministic inversion of the
  generation step - no sampling noise), clamp to `[-LATENT_SCALE, LATENT_SCALE]` (4.0), scale to `[-127, 127]`,
  round to int8: `4 x H/8 x W/8` bytes. Decode: bytes -> latent -> `vae.decode(z / scaling_factor).sample` ->
  pixels. A failed load records `error`; `auto` then falls back to the stand-in and does not retry the download on
  every image (`describe()` reports `sd_error`).
* `TinyEncoder` (`"tiny"`) - Pillow only: an RGB thumbnail at `1/8` of the size (`3 x H/8 x W/8` bytes), upscaled
  back on decode. Same reduction, same text format, so everything works without the diffusion weights.

API: `GET /api/images` (`describe()`), `POST /api/images/encode` (a binary route like `/api/uploads`: multipart,
raw body with a default name, or JSON `content_base64`; options from the query string or the JSON body: `size`,
`encoder`, `train`, `save_as`, `epochs`, `lr`, `batch_size`; `train` starts a train job on the one text, `save_as`
stores it as an upload so several images can be trained on together), `POST /api/images/decode`. CLI: `image info |
encode | decode`. Frontend: the Images tab (choose an image -> encoded text, latent size, copy, decode back to an image
side by side with the original; train on it; save as upload; decode any pasted text such as a prediction).
Dependencies: `pip install radixnet[images]` (pillow) or `radixnet[diffusion]` (pillow, torch, diffusers).

## 22. The model conversing with itself (`dialogue.py`) — replies are predictions picking up the last words

`converse(model, opening="", turns=6, mode="beam", max_length=60, context=12, temperature=1.0, k=5, beam=None,
step_penalty=0.0, seed=None, speakers=("A", "B"), history=(), partner=None, avoid_repeats=True) -> list[Turn]`
(also `GraphModel.converse(opening, turns, **options)`). Two voices take turns; the voice of turn `i` is
`speakers[i % len(speakers)]` and its model `model` for even `i`, `partner or model` for odd `i`, so the radix and
the count model can talk to each other. A given `opening` is spoken as turn 0 (`given=True`, its cost is
`-score(opening).log_prob`); `history` holds the utterances of a conversation being continued (indices and voices
carry on, only the new turns are returned).

Every reply is the prediction search run from the end of what was just said:

1. `tail_context(previous, context)` — the last `context` characters of the previous line, cut forward to a word
   boundary when a word straddles the cut ("t on the mat" → "on the mat"; a single long word keeps its tail).
2. The context is *usable* when `_prefix_start` locates its last trigram whole (a node other than START and no
   guessed partial-trigram lead). Then `predict(context, length=0, mode="beam", k, beam, to_end=True, max_length,
   step_penalty)` offers the `k` most likely complete continuations (`"sample"`: one stochastic walk per draw, up to
   `k` draws; `seed` gives the dialogue a private RNG). The first candidate that adds something (a non-empty
   continuation) and, with `avoid_repeats`, is neither an utterance the conversation already heard nor an *echo*
   (a piece of the previous line) is spoken: `text = context + continuation`.
3. Nothing (new) follows, or the context is not usable → the context loses its last word (`_shorter`) and the
   search runs again — what else could follow "on the"? — until it is empty.
4. Still nothing → the voice changes the subject: the same search from START (`fresh=True`); when even that only
   offers repeats the best repeat is spoken and flagged `repeat=True`. A voice with nothing at all to say (an
   untrained model) ends the conversation early.

A `Turn` records `index`, `speaker`, `text`, the `context` it picked up, the `reply` it added, `cost`,
`probability` (`exp(-cost)`), `reached_end`, `fresh`, `given`, `repeat`, `candidates` (continuations offered) and
`skipped` (rejected before the spoken one), plus the path (`labels`, `node_ids`, `step_costs`); `transcript(turns)`
renders `speaker: text` lines. Beam conversations are deterministic and never repeat themselves; the CLI `converse`
command, `POST /api/converse` (`ModelService.converse`, `partner` = another kind in memory) and the Converse tab
expose it, and turns are rated with the same thumbs as generated samples (`RatingsCard.jsx`).

## 23. The count / reward model in Go (`go/`) — goroutines over lines, paragraphs and pages

`go/` is a standalone Go module (`github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go`, Go 1.24, no dependencies
beyond the standard library) porting section 19's model: `go/radixnet` is the library, `go/cmd/radixnet-count` the
CLI (`train`, `predict`, `generate`, `score`, `feedback`, `2nrl`, `invert`, `weights`, `info`, `converse`). The
Python implementation is untouched; the two share the `radixnet-count` model file.

Files: `encoding.go` (code-point windows, `DecodePath`), `mt19937.go` (a Mersenne Twister with CPython's seeding,
53-bit doubles and `getstate()` layout - `rng_state` round-trips between the languages), `fsum.go` (Shewchuk's
exact summation, so path costs match `math.fsum` to the bit), `graph.go` (nodes, insertion-ordered adjacency -
Python dict order decides the softmax summation order -, trigram index, `Split` / `MergeChild` / `Compress`,
`ObserveSequence`, `Trace`, `CheckInvariants`), `weights.go` (counts, the sliding window, rewards, the dual frequency
function, lazy weights and edge costs), `search.go` (`PathResult`, `SampleWalk`), `beam.go` (`BeamPredict`),
`model.go` (training passes, feedback, prediction, generation, scoring, stats), `dialogue.go` (`Converse`),
`json.go` (the file format), `parallel.go` (`parallelFor`, `parallelRanges`, `SplitTexts`).

Concurrency (`--workers N` caps the goroutines; the default 0 is no cap):

* **Texts are the unit of work.** `SplitTexts` / `StreamSplit` cut a corpus into lines, paragraphs, pages (form
  feeds or every `--page-lines` lines) or one text per file. Encoding, tracing and counting fan **one goroutine per
  text** out (`parallelFor` with workers 0 spawns per index; a positive cap gives a bounded pool).
* **Counting is racy by design.** The counting pass traces every text of a chunk through the frozen structure in
  parallel and bumps `EdgeCount` / `Count` with plain increments from all those goroutines; a collision on the same
  counter loses an update, which is accepted (`Model.Exact` switches to `atomic.AddInt64`: no lost updates, the
  result identical to the sequential run and to Python; `--exact` on the CLI and the server; the parity tests use
  it). The race detector build tag (`race_on.go` / `race_off.go`) skips the racy test under `-race`. Measured on a
  million-line corpus: 0.7 % of the traversals lost, and no speed-up over a small exact pool - the per-text
  goroutines and the contended cache lines cost more than they save.
* **Structure building is the one sequential phase.** Splits and merges reshape the shared trigram index and the
  adjacency maps; Go's runtime aborts on concurrent map writes, and a racy structure would not be reproducible.
  The walkable texts are detected in parallel (read lock) and only the novel ones are observed, in corpus order,
  under the write lock (`register`). After `Compress()` every training text traces without a split, so the
  epochs never need the lock.
* **Weights are lazy.** `RecordTraversals` and `AddReward` mark the parents whose rows changed; `Prepare()` (called
  by every cost reader) recomputes only those rows, or every row in parallel over the nodes after a structural
  change (`parallelRanges`, a goroutine per 64 nodes without a cap). Edge costs (`-log softmax`) are cached per
  version the same way. This removes the Python implementation's per-text full recompute - the O(texts × edges)
  hot spot measured earlier.
* **Predictions run their two beams side by side** whenever the bottom cap is known up front; `ScoreAll` scores
  texts in parallel; the loss is a parallel reduction.

Streaming and chunking (`source.go`, `pipeline.go`): a `TextSource` yields texts in order and can be re-read for
every pass - `SliceSource` (in memory), `FileSource` (line by line, any line length, BOM and invalid UTF-8
repaired), `ZipSource` (`archive/zip` opened from disk: the central directory only, entries streamed on demand
through `WalkZip` with the Python module's skip rules and an 8 KB NUL sniff for binaries), `MultiSource`. A source
that has natural parts (archive entries, the files of a set) also implements `PartSource`, so the parts are opened
once and streamed independently. `Model.passesSource` consumes them in chunks of `TrainOptions.ChunkSize` texts
(default 8192, `--chunk`, `chunk_size`): the structure pass walks the chunks once; every epoch re-streams them -
encode, trace and count with one goroutine per text, the sliding window in corpus order, rewards per chunk - and
accumulates per-edge traversal counts so the loss (`weightedCost`, Σ traversals × cost / total) needs no list of
transitions. A `sequencer` restores corpus order by (part, chunk index) for the two steps that are ordered by
nature - observing novel texts and applying the window / rewards - while everything else runs in any order.

The reader never waits for a chunk it spawned, but it does wait for a slot: `TrainOptions.Inflight` (`--inflight`,
`inflight`; default two per CPU) bounds the chunks in flight, and a chunk holds its slot from the moment it is read
until the sequencer has applied it - a finished chunk waiting for its turn costs the same memory as one being
counted. Without that bound the reader outruns the counting: 1.7 GB on a 39 MB archive, 3.2 GB with `ParallelParts`
(a reader goroutine per part, `--parallel-parts`, off by default), against 393 MB now - and that pile-up is what
killed the server on a big upload. Parallel parts cannot share one pool, since the slots would all be taken by
parts the sequencer cannot reach yet while the part it waits for cannot start, so each open part gets its own
budget of chunks and only a window of parts is open at a time; permits are taken in part order, so the oldest
unfinished part always holds one and the pipeline cannot stall. The second half of that failure is the collector: Go grows the heap to
about twice the live set before collecting, so `ApplyMemoryLimit` (`memlimit.go`) sets `debug.SetMemoryLimit` to
80 % of the process's cgroup limit (its own path in `/proc/self/cgroup`, v2 and v1) or of `MemAvailable` at startup
(`--memlimit SIZE`, `--memlimit off`, `GOMEMLIMIT` wins), which trades speed for staying inside the box. Measured
on a 113 MB corpus (10,703 files, 421 K nodes, 2.34 M edges, 426 MB live heap): 1452 MB peak RSS with no limit,
1027 MB at a 1 GiB limit for the same wall time, 767 MB at 512 MiB with the collector running continuously (peak
RSS runs above the limit itself: it counts the runtime's unreturned spans and the page cache of the files being
read and written, while the anonymous memory stays inside it). Run inside a 1 GiB cgroup, the same training is
killed by the OOM killer after 26.9 s without a limit and finishes in 32.5 s at 908 MiB with the detected one -
the failure the user hit, and the fix. Saving and loading stream through the file (`encodeCompact` into the gzip /
temp file, `json.Decoder` out of it), the adjacency lists are two parallel slices with a map index only past
degree 16 (a Go map per node cost more than the rest of the graph), the sliding window compacts in place and the
read buffers are pooled, all of which keep the allocation rate off the collector's back.

The server's `POST /api/uploads` streams multipart parts and raw bodies into the upload directory
(`Uploads.StoreStream`: the first four bytes decide text or ZIP, an archive is validated by `InspectZip` streaming
its entries), the listing inspects archives by streaming, and `POST /api/train` builds a `MultiSource` of inline
texts and `Uploads.Source` (ZIP / file sources) for `StartTrainSource`. The JSON upload forms, which must be
parsed whole, are capped at 512 MB. `GET /api/status` reports `heap_bytes` and `memory_limit_bytes` so the status
bar can show how close a run is to its ceiling.

Parity (`tests/test_go_parity.py`, skipped without a Go toolchain): both implementations train the sample corpus
with the same seed and settings and must agree on labels, counts, edges, rewards, window events, RNG state (exact),
weights (1e-12), predictions, generated texts, scores and conversation transcripts; each side loads and continues
the other's file with identical results; feedback, invert and weight changes match too. `go test -race ./...`
covers the Go module (RNG vectors against CPython, exact summation, structure invariants, lazy weights against a
full recompute, 1 vs 8 workers giving the same model, search / generation / conversation, gzip round trips).
`TestGoTutorParity` runs both tutors against one fake Ollama: the calls the teacher receives (system and user
prompts, in order), the marks, the report card and the resulting graph must all match, a dry run on either side
must change nothing, with `--plan N` the report card is read out to the teacher in the same words and both sides
come back with the same syllabus (lessons, targets, weak points, level, summary, the brief and the upgrade), and
with `--brief TEXT` both hand that brief to the exercise writer identically.

### 23.1 The Go HTTP server (`go/server`) and the frontend hookup

`radixnet-count serve --port 8001 --frontend-dir frontend/dist --upload-dir uploads --checkpoint-dir checkpoints`
runs `go/server`: `service.go` (the `Service` - model, `Job`, uploads, checkpoints; `startJob` runs the work on a
goroutine holding the model's write lock and, in the progress hook, releases it between epochs so readers get a
turn - the Python `_yield_to_readers`; one job at a time, 409 otherwise), `http.go` (the route table, typed body
fields with the Python server's error messages, upload bodies as JSON / multipart / raw, static files with the SPA
fallback and immutable caching of hashed assets, 404 / 405 / 413 like the Python handler), `uploads.go` (the
upload directory, `ExtractTexts` with the same skip rules as `archive.py`, archives kept whole and unpacked in
memory), `checkpoints.go` (the `CheckpointManager` layout: `ckpt-<tag>-<step:06d>.json.gz`, `latest.json`,
`index.json`, pruning to `--keep`). Section 12's contract holds for every count-model endpoint; `POST /api/train`
additionally takes `split` (`lines | paragraphs | pages | file`) and `page_lines`, the units the goroutines fan out
over; `/api/health`, `/api/status` and `/api/model` carry `engine: "go"`, `workers` and `goroutines`; the Python-only
endpoints (evolve, the ollama corpus / review calls, images, speech, codegen, schedule preview) answer 404 with a message
naming the Python server.

The tutor is ported too (`go/radixnet/llm.go`, `ollama.go`, `chatgpt.go`, `tutor.go`, `plan.go`,
`go/server/tutor.go`): the same prompts, the same records and the same endpoints (`GET /api/tutor`,
`POST /api/tutor/start`, `GET /api/tutor/history`, `POST /api/tutor/lesson`, `POST /api/tutor/plan`,
`GET /api/chatgpt/models`), with the count / reward model
answering the exercises and `serve --ollama-url / --ollama-model / --chatgpt-url / --chatgpt-model` (or
`$OLLAMA_HOST` / `$RADIXNET_TUTOR_MODEL` / `$OPENAI_BASE_URL` / `$RADIXNET_OPENAI_MODEL`) setting the defaults.
Both teachers are here as well: `LLMClient` is the interface `OllamaClient` and `ChatGPTClient` implement,
`NewLLMClient(provider, ...)` builds one, `TutorConfig.TutorProvider` / `GraderProvider` choose them (resolved by
`Resolve`, which `Validate` calls), and `$OPENAI_API_KEY` is read per request and never stored - the key is the
server's own, never a request field. `radixnet-count tutor -tutor-provider chatgpt` is the CLI twin, `-plan N`
included; `plan.go` is `weak_points` / `plan_from_card` / `plan_lessons` word for word - the same system prompt,
the same report card read out to the teacher, the same fallbacks and the same JSON - so the parity test drives both
CLIs against one fake teacher and compares the plan they come back with.
The job releases the model lock around every LLM call, as the Python service does. Ratings reach the
model through `WeightGroups` + `RewardWeighted` / `PunishWeighted` / `TwoNRLWeighted` (one pass per distinct
weight, the reward or penalty scaled by it), which also back `good_ratings` / `bad_ratings` on `/api/2nrl` and
`/api/feedback`. The corrections go the diff way here too: `go/radixnet/diff.go` is the same alignment as
`diff.py` and `Model.Correct` (`go/radixnet/correct.go`) the same edge updates, down to the tie-breaks, so the
tutor parity test finds the same rewards on both sides; `radixnet-count correct` is the CLI twin of
`radixnet correct`.

Frontend (`App.jsx`): `engineOf(status, health)` reads the engine; with `"go"` the Python-only tabs (Evolve, Ollama,
Code, Images - the Tutor tab stays, both servers run the lessons) are neither shown nor mounted, the header shows a **Go engine · N goroutines** badge, the status bar
replaces the accelerator chips with `engine go · workers · goroutines` (`StatusBar.jsx`), and the Train tab
(`TrainPanel.jsx`) offers **Texts are** `lines | paragraphs | pages` (+ lines per page): with paragraphs or pages
the pasted text is sent whole as `text` together with `split`, and the server cuts it and the selected uploads.
Everything else is unchanged: the panels only ever spoke the JSON contract.

Tests: `go/server/server_test.go` (httptest: every endpoint, the job lifecycle with 409 and stop, readers answering
while a job runs, uploads in all three body forms with a ZIP built in the test, training from uploads with and
without `whole_file`, graph, save / load / reset, checkpoints in the Python layout, static files, the error shapes)
and `tests/test_go_parity.py::TestGoServer` (a live `serve` process: the key sets of `tests/test_api.py`, a
model saved by the server loaded in Python with identical predictions, ZIP uploads, `split: paragraphs`, and the
Python `CheckpointManager` reading the server's checkpoints).

---

## 24. The negative network (`negative.py`, `blame.py`, `duo.py`) — the failures, and why

A second copy of the network that keeps only its **negative portions**: the same self-compressing cyclic graph, the
same trigram window, the same Dijkstra / beam searches, but **every node and edge in it exists because something went
wrong there**. It is a model kind like the others (`kind = "negative"`, `format = "radixnet-negative"`, default file
`model.negative.json`), so `model_classes()`, `load_model`, checkpoints, `POST /api/model/select` and `--kind
negative` all reach it; what makes it different is that it is trained on negative data alone and that every edge
remembers *why*.

### 24.1 `NegativeGraph(RadixCyclicGraph)`

Per-edge lists beside `edge_w` / `edge_count` (appended in `_new_edge`, so they survive `split` and `merge_child`
exactly as the counts do): `edge_blame` (summed severity), `edge_fails` (failing texts through the edge),
`edge_clear` (how much cleared text crossed it) and `edge_reasons` (`{reason_id: blame}`, at most
`MAX_EDGE_REASONS = 8` entries; the smallest is dropped when a ninth appears and the graph-level totals keep it).
The graph owns the reason registry (`reasons`, `reason_ids`, `reason_blame`, `reason_fails`, `reason_id(label)`
normalising tags to one lower-case line) and the totals `total_blame`, `total_fails`, `total_clear`.

* `record_failure(edge_ids, severity, reason)` adds the severity to every listed alive edge, bumps its fail count and
  its reason entry, updates the totals and recomputes the weights; `record_clear(edge_ids, weight)` does the same for
  cleared text.
* `evidence(e) = max(0, blame - clear_scale * clear)` — **blame and clearing cancel**, so an edge the tutor's passes
  cross as often as its failures do carries no verdict at all. This is the only notion of evidence in the module:
  the weight function, `judge` and the filter all read it.
* `edge_weight(evidence, parent_evidence, degree) = share_scale * log((net + s) / (parent_net + s * deg)) +
  blame_scale * log1p(net)` with `s = SMOOTHING = 0.5`, `share_scale = 1`, `blame_scale = 0`, `clear_scale = 1`.
  `recompute_weights()` walks every parent once (O(E)) and bumps `version` so the cost cache refreshes;
  `configure(**scales)` changes the function at run time and `weight_config()` describes it.
* Every node is created with `a = 0, k = 1` (the sine activation is the constant 1), so the base class's score
  `w * f_p * f_c` is the weight itself and `child_probs`, `child_costs`, Dijkstra, sampling, splits and merges work
  unchanged. A softmax over the children is therefore `P(child | parent)` **under the failure distribution**.
* `merge_child(p)` refuses to merge a unary chain whose edge carries blame or clearing: compression turns a
  transition into a deterministic step *inside* a node, and a step inside a node has no edge to carry evidence, so a
  correction that blamed a single transition would be folded away and forgotten. Everything the tutor never ruled on
  still merges.
* `forget(reason=None, factor=0.0)` scales one reason's blame (or all of it) down — `factor` 0 drops it, 0.5 halves
  it — and returns `{reason, edges, blame_removed}`. `invert()` swaps `edge_blame` and `edge_clear` (the inverse of a
  network of failures is a network of what passed) and toggles `inverted`.
* `to_dict` / `from_dict` carry `edges.blame|fails|clear|reasons` (compacted in the same order as the base class's
  edge arrays), the scales, the totals and the reason registry, and rebuild the weights on load.

`RadixCyclicGraph.trace(trigrams)` (public wrapper around `_trace`) walks a sequence without creating, splitting or
counting anything — what clearing and judging need.

### 24.2 `NegativeNet(GraphModel)` and the tutor

One `_pass(texts, cfg, blame=..., amount, reason, source, note)` routine underlies everything. A **blame** pass
registers the texts structurally (`_register`: two passes, because a later text can split a node an earlier one
pointed at), compresses, and then per epoch blames every transition of every text; a **clear** pass creates nothing
and credits only the transitions a text shares with the existing failure structure (`_shared_edges`, built from
`crossings`), so ordinary correct text — which will never walk a graph of failures end to end — still counts. Each
pass appends a record (`phase: "negative" | "clear"`, `loss` = mean `-log P` of the transitions, `matched` /
`unmatched`, `edges_touched`, `reason`, `severity`, `source`) and, for blame passes, a journal entry
(`MAX_LOG_ENTRIES = 200`, newest kept: `at`, `text`, `reason`, `severity`, `source`, `note`).

| Method | What it does |
|---|---|
| `blame(texts, reason, severity, source, note, epochs)` | learn a failure — the only operation that adds structure |
| `clear(texts, weight, epochs)` | the tutor passed these: take blame off what they share, create nothing |
| `train(texts, ...)` | **is** a blame pass (`lr` / `act_lr` / `batch_size` accepted and ignored) |
| `punish` / `reward` | thumbs down = `blame` (reason `thumbs-down`), thumbs up = `clear` |
| `two_nrl(bad, good, ...)` | blame `bad` (optionally per-text `bad_weights` scaling the severity), then clear `good`; no inversion |
| `invert_paths(texts, amounts=...)` | the evolve loop's failures: blame each path by `strength * amount` |
| `predict(prefix, ...)` | the beam search of section 19 over the failure distribution: the K most likely ways to go wrong (and the K least likely as `bottom`) |
| `crossings(text)` | every transition of a text through the failure structure: `{index, start, end, fragment, parent, edge, blame, fails, clear, evidence, reason, reasons}`, `edge = None` where the network has never been |
| `correct(wrong, right, reason=, severity=, ...)` | the tutor wrote the sentence out correctly: blame only the steps that wrote a character it struck out (`_steps_over` over `diff.changed_spans`), never an edge the correction walks too, and clear everything the correction shares with the failure structure |
| `judge(text, threshold=, min_coverage=, spans=)` | the verdict (below) |
| `reasons()` / `recent(limit)` / `forget(...)` | the reason table / the journal / drop or fade a reason |

`judge` returns `{text, chars, transitions, known, blamed, coverage, blame, risk, peak, per_char, threshold,
min_coverage, verdict, reasons, spans, why}`: `blame` is the summed evidence of the blamed transitions, **`risk` that sum over
*all* of the text's transitions** (compression moves both together, so the ratio does not depend on it — repeating a
failure blamed once scores about 1, sharing a third of one's transitions with it about 0.33), `coverage` the share
of transitions that are known failures, `reasons` the aggregated reason histogram (each blamed edge contributes its
reasons scaled by `evidence / blame`), `spans` the worst fragments with their character range (`start = i - 1`,
`end = i + WINDOW` around the junction), and `verdict` is `"reject"` when `coverage >= min_coverage` **and**
`risk >= threshold` (defaults 0.5 and 1.0), `"suspect"` when something failed but not enough, `"pass"` when nothing
here ever failed; `peak` is the most evidence any single transition carries, which is what a correction moves and
what the filter's `peak` rule reads. `why` says it in one sentence. Judging is counted in `meta` (`judgements`,
`rejected`).

`GraphModel._steps_over(grams, length, spans)` and `_touches` live in `model.py` (moved up from `countnet.py`, which
inherits them): the edges of a traced text whose step wrote a character inside one of the changed spans, charging
every step with the characters it adds and the step into END with the position just past the last one. Both
corrections - the count model's reward-based `correct` and the negative network's blame-based one - mark the same
characters, and so does the Go port.

**`blame.py`** is where the negatives come from; nothing else may create them. A **fault** is
`{text, reason, severity, note, source}` plus, when the tutor rewrote the sentence, `correction`.
`faults_from_lessons(lessons, threshold)` reads a round of the English tutor (section 16.1): the mistake it named
(`grade.error`, one of `tutor.ERROR_TYPES`) is the reason, its mark is the severity, its sentence of teaching is the
note and its correction rides along so `teach` routes the fault through `NegativeNet.correct`; the sentences that
passed, the corrections themselves and the teacher's model answers all clear blame.
`classify(critique, verdict=, rating=)`
picks a reason out of `REASONS` (`empty`, `gibberish`, `repetition`, `truncated`, `grammar`, `spelling`,
`contradiction`, `false`, `incoherent`, `off-topic`, `other`, plus `unrated`) by matching the tutor's own words;
`severity_from_rating(rating, threshold)` maps 0 -> 2.0 and the pass threshold -> 0.25 (an unrated failure is 1.0);
`code_reason(attempt)` reads the sandbox, the style report and the judge into `CODE_REASONS` (`timeout`, `crash`,
`wrong-output`, `task-not-done`, `style`, `naming`) with the severities in `CODE_SEVERITY`.
`faults_from_reviews` / `faults_from_attempts` / `faults_from_lessons` turn a tutor's output into `(faults, passed)`,
and `teach(negative, faults, passed)` (with the wrappers `teach_reviews`, `teach_attempts` and `teach_lessons`)
blames each fault - through `correct` when it carries one - and clears the passes, returning
`{blamed, cleared, unmatched, edges, reasons, severity_mean, records}`.

The call sites: `tutor --blame` / `TutorTrainer(negative=...)` / `POST /api/tutor/start {"blame": true}`, which
blames every failed sentence of a round with the mistake the teacher named and adds `negative_blamed` /
`negative_edges` / `negative_reasons` to its round records; `ollama review --blame` /
`POST /api/ollama/review {"blame": true}` (the reviewer), `codegen --blame`
/ `CodeGenTrainer(negative=...)`, which blames the rejected attempts of every problem and adds `negative_blamed` /
`negative_reasons` to its problem records, `evolve --blame` / `Evolver(negative=...)`, which blames every fake the
discriminator scored below the real texts (reason `blatant` past `blatant_margin`, else `discriminator`, severity
`gap / margin` clamped to `[0.25, 2]`) and clears the real texts, and `POST /api/negative/blame` for a person.

### 24.3 `duo.py` — the pair as a GAN at output time

`NegativeFilter(positive, negative, config)` puts the two networks on one output path (`FilterConfig`: `threshold`,
`min_coverage`, `ratio` (`None` = off), `peak` (`None` = off), `over_sample`, `strict`, `spans`, `learn`, `reason`). `judge(text)` merges
the negative network's verdict with the likelihood ratio `negative.score(text)["per_char"] -
positive.score(text)["per_char"]` — the discriminator logit of two generative models — and rejects when the blame
rule fires, when `peak >= config.peak` (the evidence on one transition - a correction blames a handful of
characters, so this is how a single corrected word vetoes an otherwise clean sentence), or, behind the same coverage
gate, when `ratio >= config.ratio`; `rule` names which (`"blame"`, `"peak"`, `"ratio"`, `"suspect"` under `strict`). `filter(texts)` splits kept from rejected (blaming the rejects
when `learn`, which is off by default: the tutor supplies the negatives, the filter only applies them).
`generate(count, over_sample=, ...)` asks the positive model for `count * over_sample` candidates, filters them and
returns the `count` survivors with the least blame plus every rejected verdict; `predict(prefix, ...)` filters the
positive model's top-K continuations and adds `warning`, what the negative network predicts goes wrong from that
prefix. `describe()` reports both halves and the settings.

### 24.4 CLI, API, frontend, tests

* CLI: `negative <action>` with `--negative PATH` (before or after the action; default `model.negative.json` derived
  from `--model`): `blame`, `clear`, `why`, `filter`, `reasons`, `forget` — see the README's CLI table. `--blame`
  (with `--negative`) on `tutor`, `correct` (one hand-written correction teaches both networks from the same diff),
  `ollama review`, `codegen` and `evolve`. `--kind negative` makes it the ordinary model.
* API: `GET /api/negative` (stats, reasons, journal, settings), `POST /api/negative/blame|clear|judge|filter|forget|
  settings|reset|save`. The service keeps exactly one negative network, in the same `_parked` store as the other
  kinds, so selecting the `negative` kind hands back that very object; `positive_model()` finds the model it filters
  (the active one, else a parked or saved positive kind, else 409). `POST /api/save` writes the negative network
  beside the model when it holds blame.
* Frontend: the **Negative** tab (`NegativePanel.jsx`) — run the pair, judge a text with its blamed fragments marked
  (`<mark>`), blame / clear by hand, the reason table with per-reason *forget*, and the journal; the Ollama tab's
  review card has a *teach the negative network* checkbox and reports what it learned.
* Tests: `tests/test_negative.py` (weight function, evidence, blame / clear, corrections, judge, crossings,
  prediction, forget, inversion, persistence, the kind registry, the API routes, the CLI group),
  `tests/test_blame.py` (classification, severities, faults from lessons / reviews / attempts, teaching, and the
  tutor, evolve and codegen hooks) and `tests/test_duo.py` (all three rules, the coverage gate, strict, learn,
  generate / predict, the count model as the positive half).

---

## 25. Teaching by talking (`speech.py`) — the transcript and the waveform behind one unique token

One utterance becomes **two texts that start with the same token**, and both are trained on:

```
<speech:9f2a1c7d> the cat sat on the mat
<speech:9f2a1c7d> aud:mu:8000x1:gOXv7NgqFBAYTePu7dwvFRAXPuHu7d82FhAWNt/t7uE+FxAVL9zt7uNNGBAU…
```

`utterance_token(payload, unique=True)` is `SPEECH_TOKEN` (`<speech>`) with a short blake2b digest of the waveform
text folded in, so the token is unique to that utterance, deterministic (the same recording always gets the same
one) and **identical in both texts**: in the cyclic graph the words and the sound leave the *same* node, which is
what ties them together. `unique=False` falls back to the bare `<speech>` for every utterance (one entry node for
all speech), and an explicit `token=` overrides both. The token can never collide with the graph's `<s>` / `</s>`
sentinels (no 3-character window of `<speech…>` equals `<s>`).

`speech_texts(token, transcript, audio_text, pair=False) -> list[str]` builds them: the transcript (whitespace
collapsed) when there is one, the waveform text when there is one, and with `pair` a third text - the waveform
followed by its transcript - so the prediction search can run from the sound straight into the words.

### The waveform as text

`encode_audio(data, rate=8000, codec="auto", normalise=False) -> {"text", "codec", "rate", "channels", "samples",
"seconds", "bytes", "chars", "normalised", "source"}`: the audio is decoded, mixed down to mono, resampled (linear
interpolation - the next step quantises to a byte anyway), optionally peak-normalised to 0.99, quantised to one byte
per sample and packed as `aud:<codec>:<rate>x<channels>:<base64>` (`pack_text` / `parse_text`).

* `"mu"` (the default) — mu-law companding, `sign(x)·ln(1 + 255|x|)/ln(256)` mapped onto 256 levels: the classic
  8-bit speech quantisation (G.711, WaveNet). Its relative error stays at ~2 % of the amplitude from full scale down
  to a whisper, where linear 8-bit is already at 38 %, and it is a *stable* code (encoding a decoded byte gives the
  same byte back).
* `"pcm8"` — plain linear 8-bit, one signed byte per sample.

`parse_text` looks for the header **anywhere** in the text, because a spoken text carries its token first and a
paired one its transcript last; the payload runs from the header to the next whitespace and its base64 is repaired
with `encoding.repair_base64` (section 4). `decode_text(text, codec=None) -> {"wav", "codec", "rate", "channels",
"samples", "seconds", "bytes", "repaired"}` turns an encoded - or **predicted** - text back into a 16-bit PCM WAV
(`wav_bytes`), so what the graph says can be listened to.

`load_audio(data)` reads RIFF/WAVE itself: PCM 8 (unsigned) / 16 / 24 / 32, IEEE float 32 / 64, A-law and mu-law
(the G.711 expansion tables), `WAVE_FORMAT_EXTENSIBLE` resolved through its SubFormat tag, chunks walked in order
with the odd-length padding, a streamed `data` chunk of declared size 0 read to the end. Anything else (MP3, M4A,
WebM/Opus, Ogg, FLAC) goes through `ffmpeg` when it is installed (`$RADIXNET_FFMPEG` overrides the binary); without
it the error names the format it recognised from the magic bytes and says what to install. The browser never needs
it: the Speech tab converts its recording to WAV itself (section 13).

### Speech to text

`transcribe(data, backend="auto", text="", language=None, model=None, url=None) -> {"transcript", "backend",
"model", "language", "seconds"}` with `ASR_BACKENDS = (auto, given, faster-whisper, whisper, server)`:

* `given` — the transcript comes from the caller (the browser's Web Speech API, `--text`, the API's `transcript`).
  Always available; it is why the feature works with nothing installed, and under `auto` it wins over everything.
* `faster-whisper` / `whisper` — the two local Whisper packages, the model (`$RADIXNET_WHISPER_MODEL`, default
  `base`) loaded once per process and kept; a failed load is recorded in `.error` and raised as a `SpeechError`.
* `server` — any OpenAI-compatible `/v1/audio/transcriptions` endpoint (whisper.cpp's server, Speaches, ...) posted
  to with a hand-built multipart body over `urllib`; `$RADIXNET_ASR_URL`, `$RADIXNET_ASR_MODEL`,
  `$RADIXNET_ASR_KEY` / `$OPENAI_API_KEY`. `auto` only reaches for it when a URL is configured, so nothing ever
  blocks on a network call that was not asked for.

Audio that is not a WAV is converted with ffmpeg before it reaches a backend (16 kHz, what Whisper wants anyway).

### The flow

`teach(data, transcript="", *, backend, language, asr_model, asr_url, rate, codec, normalise, waveform, pair,
token, unique) -> {"token", "transcript", "asr", "audio", "texts", "chars", "pair"}` is what the CLI, the API and
the frontend all call: encode the waveform, transcribe when no transcript was given, derive the token from the
waveform text, build the texts. **A failing transcription is recorded, not raised** (`asr.error`): the waveform on
its own is still worth learning, so talking to a machine without an ASR backend still teaches it something.

`record(seconds, rate=16000, recorder=None)` records from the default input device with whichever of `arecord`,
`rec` / `sox` and `ffmpeg` (`-f alsa` / `-f avfoundation`) is installed; `recorders()` lists them and `describe()`
reports everything the feature can reach.

CLI: `speech info | transcribe | teach | listen | decode` (`listen` records first, then runs `teach`; `--train`
trains on the texts and saves). API: `GET /api/speech`, `POST /api/speech/transcribe`, `POST /api/speech/teach`,
`POST /api/speech/decode` (section 12). Frontend: the Speech tab (section 13). Dependencies: none for the waveform,
the token and the browser's dictation; `pip install radixnet[speech]` (faster-whisper) or `radixnet[whisper]`
(openai-whisper) for server-side transcription, ffmpeg for audio formats other than WAV.
