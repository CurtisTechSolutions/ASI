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
| 2NRL (*Double-Negative Reinforcement Learning*) | `two_nrl(bad, good)`: (1) train on bad/garbage data, (2) `invert()` the network, (3) fine-tune on correct data with a smaller learning rate. The two negatives of the name are (1) and (2); (3) is the positive one. |
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
    encoding.py             Encoder / Decoder, and the word alphabet (Vocabulary, section 34)
    attention.py            the attention band: where inside a gram a correction's blame and credit land (section 38)
    window.py               the dynamic window: a ladder of node sizes, halving from 32 to 4 and back up (section 40)
    counter.py              cyclic counters: every growing integer wraps at COUNTER_LIMIT and counts the reset (section 28)
    graph.py                RadixCyclicGraph (nodes, edges, trigram index, split/merge, CSR export/import, to_dict/from_dict)
    backend.py              CSR, NodeParams, Backend protocol, PythonBackend, TorchBackend, get_backend()
    search.py               PathResult, Dijkstra predictor + stochastic sampler
    beam.py                 Prediction, beam_predict (top-K / bottom-K continuations in one search; section 19)
    penalty.py              the punishment traversal: the merit / penalty split of an edge's evidence, and the
                            cost function that prices a step by the punishment it carries (section 31)
                            (ported to Go as go/radixnet/penalty.go)
    model.py                GraphModel (shared base), RadixNet, TrainConfig, model-kind factories (load_model, new_model, ...)
    countnet.py             CountRewardGraph, CountRewardNet - the count / reward model (section 19)
    wordnet.py              WordGraph, WordNGramNet - the same model over an alphabet of words (section 34)
    negative.py             NegativeGraph, NegativeNet - the negative network: the failures, and why (section 24)
                            (ported to Go as go/radixnet/negative.go + blame.go + duo.go, section 24.5)
    blame.py                the tutors' verdicts -> faults for the negative network (section 24.2)
    duo.py                  FilterConfig, NegativeFilter - the pair as a GAN at output time (section 24.3),
                            and the guard: the same pair on every output path (section 24.7)
    dialogue.py             Turn, Heard, stutter, backtrack, teach_back, Rethink, reply, converse, repeats -
                            the model conversing with itself, thinking twice about a repeat, and teaching the
                            graph where it goes round (section 22); StreamFn, the events a conversation is
                            watched through as it happens (section 22.1)
    thinking.py             Thought, think, think_on, questions_in, place, summarize - the THINK sentinel at
                            work: what makes the model think, what it thinks, how a thought questions itself
                            and what it triggers when it stops (section 36)
    chat.py                 Chat, ChatConfig - the model conversing with an LLM that marks it (section 28)
                            (ported to Go as go/radixnet/chat.go, section 28.1)
    speech.py               teaching by talking: transcription, the waveform as text, the unique token (section 25)
    recall.py               the speech / image recall tutor: ask for it back, mark it, blame it (section 26)
    critic.py               the negative network feeding itself: an LLM reviewer on a loop (section 24.6)
                            (ported to Go as go/radixnet/critic.go + review.go)
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
    def inverted(self) -> "SineActivation"     # a -> -a AND k -> -k (the negation of the unit: f = a*sin(u)+k)
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d)

def edge_signal(w: float, fp: float, fc: float) -> float    # w * fp * fc  ("activation of child * activation of parent")
```

---

## 4. `encoding.py`

The default is the character trigram of stride 1 (D-006); it is the default of a **dial**, and section 23.1 is
the full account of the dial in both implementations. Everything below is that default.

```python
WINDOW = 3               # the default n of the n-gram
CHARS, WORDS = "char", "word"   # what one unit is
PHONES, SYLLABLES = "phone", "syllable"   # ... or a sound, or a syllable: the text read through ../PhoneticTokenizer
START_LABEL = "<s>"      # reserved label of the START node (id 0)
END_LABEL   = "</s>"     # reserved label of the END node   (id 1)

@dataclasses.dataclass(frozen=True)
class Encoding:          # unit x n x stride; the zero value is the trigram
    unit: str = CHARS
    n: int = WINDOW
    stride: int = 1
    # overlap == n - stride; encode / decode_path / decode_grams / normalize / units / piece / join /
    # truncate / has_unit_prefix / to_dict / from_dict

def parse_encoding(spec: str) -> Encoding   # "word:2:1", "char:5:groups", "trigram", "phone:3:1", "syllable:2:1", ...
def phonetic_tokenizer(unit) -> PhoneticTokenizer   # the phonetok tokenizer a phonetic unit reads through, made once,
                                                    # over the portable lexicon (the core + PHONETOK_LEXICON)
# Encoding.phonetic, Encoding.spell(text): the words a text of sounds spells.  A phonetic unit's units(text) is
# split_words(tokenizer.text(text)): a word becomes its sounds, punctuation a pause, the gap between words "#",
# and text that is already sounds passes through - so a label cuts into the units it was made of (section 36).

class Encoder:           # the encoder half of an Encoding, kept as an object
    def __init__(self, window: int = WINDOW, encoding: Encoding | None = None)
    def encode(self, text: str) -> list[str]
        # sliding window, stride 1. len(text) < window -> [] ; "hello" -> ["hel","ell","llo"]

class Decoder:
    def __init__(self, window: int = WINDOW, encoding: Encoding | None = None)
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
WORD_BASE = 0x0100          # the first word symbol; MAX_WORDS = 1_111_808, UNKNOWN_WORD = "<unk>"

def word_symbol(word_id: int) -> str    # id -> the code point that carries it (the surrogates skipped)
def symbol_word(symbol: str) -> int     # and back
def split_words(text: str) -> list[str] # the whole tokeniser: the Unicode White_Space property

class Vocabulary:                       # the alphabet of a word model (section 34)
    def id(self, word) -> int           # 0 = <unk>: a word it has never read
    def add(self, word) -> int          # the next id, in first-read order
    def encode(self, text, grow=False) -> str   # text -> the graph's symbols, one code point per word
    def decode(self, symbols) -> str            # and back, words joined by single spaces
```

The word alphabet, used only by `wordnet.py`: it turns text into the symbols the
same `Encoder` then cuts into windows of three, so the graph never learns what a
symbol stands for. Section 34 and `SPEC-WordNGrams.md` are the whole of it.

```python
def repair_base64(body: str) -> tuple[bytes, bool]   # (payload, repaired)
```

The tail shared by the *media* text formats (`img:...`, section 21, and `aud:...`, section 25): a base64 payload the
network **predicted** is rarely clean, so characters outside the alphabet are dropped, a single dangling character
goes with them, the padding is completed and `repaired` reports whether any of that changed the text. The payload is
returned as it decodes; each format pads or truncates it to the length it needs.

---

### 4.1 `voice.py` — the model heard as it walks

```python
class Speaker:                          # what a model emits -> the tokens of the phonetic tokenizer -> the voice
    def __init__(self, encoding, rate=16000, pitch=120.0, tempo=1.0, gain=0.5)
    def feed(self, piece: str) -> bytes  # sounds pass through; words are transcribed; letters gather into words
    def end(self) -> bytes               # the final sentinel: the utterance is closed
def speak_walks(model, prefix="", count=1, max_length=60, temperature=1.0, seed=None, ...) -> Iterator[bytes]
    # sample `count` walks; every step's units reach the voice as the walk takes them (search.sample_walk's
    # on_step); END closes each utterance (D-081).  The CLI's `speak` writes the chunks to a WAV, a player or stdout.
```

---

## 5. `graph.py` — RadixCyclicGraph

### 5.1 Storage (performance: flat parallel lists indexed by node id)

```python
START, END, BACK = 0, 1, 2      # FIRST = 3: the first node id that is not a sentinel

class RadixCyclicGraph:
    labels: list[str]            # labels[i] = label of node i; labels[0..2] = "<s>", "</s>", "<back>"
    z: list[float]               # node state / pre-activation, learnable
    a, b, h, k: list[float]      # per-node activation params (defaults -1, 1/3, 0, 0)
    count: list[int]             # times the node was visited in training - a cyclic counter (section 28)
    count_resets: dict[int, int] # node id -> how often its counter wrapped (absent = never; usually empty)
    alive: list[bool]            # False for nodes removed by merge (ids are never reused until compaction on save)
    children: list[dict[int, int]]  # children[p][c] = edge id e
    parents:  list[dict[int, int]]  # parents[c][p]  = edge id e
    edge_w: list[float]          # edge weights (index = edge id)
    edge_count: list[int]        # cyclic counters, like count
    edge_count_resets: dict[int, int]
    edge_alive: list[bool]       # tombstones for removed edges
    trigram_index: dict[str, tuple[int, int]]   # trigram -> (node_id, offset); offset = index of the trigram inside the label
    rng: random.Random           # seeded
    inverted: bool = False
    traversals: CyclicCounter    # every increment made to the counters above; bounds each of them (section 28)
    version: CyclicCounter       # bumped on ANY structural or parameter change (cost caches key on it)
    structure_version: CyclicCounter  # bumped only on node/edge creation, removal, split, merge
```

Every integer above that only ever grows is a **cyclic counter**: it goes back
to 0 at `COUNTER_LIMIT` and the wrap is counted in the matching `_resets`
(section 28). `node_count(i)` / `edge_traversals(e)` give the exact totals and
`carry_counters()` does the wrapping.

Node id 0 is START, 1 is END; both always exist, are never merged or split, and
their labels are not in `trigram_index`. `f_START` and `f_END` are ordinary
activations of their own `z`.

New node: `z = rng.uniform(-4.5, 4.5)`, `a,b,h,k = defaults`, `count = 0`.
New edge: `w = rng.uniform(0.5, 1.5)`, `count = 0`.
If `inverted` is True at creation time: new nodes get `a = -DEFAULT_A` and new
edges get `w = -rng.uniform(0.5, 1.5)` (consistent with the inverted network).

### 5.1.1 The third sentinel: `BACK`, where the graph has learned it goes round

`START` and `END` are observed: every training text runs `START -> … -> END`, so the model learns where texts
begin and stop the way it learns everything else. **`BACK` is learned too, but from experience rather than from
a corpus** - nothing in a text says "a walk that gets here goes round in circles".

An edge `p -> BACK` is an ordinary edge: a weight, a counter, a share of `p`'s softmax. What it means is *this
is where my walks loop*, and it is taught by the voices that caught themselves repeating and had to back up
(`dialogue.teach_back`, section 22). `observe_back(p, went=…, instead=…)` teaches three things at once, all of
them ordinary learned quantities:

* `p -> BACK` is created on first use and bumped like an observed transition - the sine model nudges its weight
  in the direction that makes it likelier, the count model gives it a reward and lets the dual frequency
  function do the rest;
* `went`, the child the walk was about to loop through, gets *dearer*;
* `instead`, the child taken after backing up, gets *cheaper*.

The first says where it goes round; the other two say what to do instead. Because `BACK` competes for
probability with `p`'s real children, a few hand-overs are enough for it to become the model's most likely next
step there - and then the **search itself** stops walking through `p` (`search.onward`, section 7), in every
prediction, not only in conversation. The trait is the model's, and it is in the model file.

`BACK` never appears in a path, emits nothing and is never trained by a text; its state `z = BACK_Z` is fixed at
the edge of the drawn range rather than sampled, so adding the sentinel moved no random stream (every seeded
model that existed still predicts exactly what it did) and its activation stays firmly non-zero, which is what
lets an edge into it carry a learned weight at all. A node that has been taught to hand over has two children
where it had one, so the unary-chain rule stops merging it: a loop point is a junction, and compression leaves
it alone.

Graph format 3 carries it. Files written at format 1 or 2 gain an unvisited `BACK` on load and their node ids
shift up by one (`_with_back`, mirrored by the Go `withBack`); the sentinel takes `START`'s activation
parameters, so a count model's stays 1 and a sine model's stays the default.

### 5.1.2 The fourth sentinel: `THINK`, where the graph has learned to stop and think - and how thoughts begin

`THINK` (node id 3; `FIRST`, the first real node id, is now 4) faces both ways, and that is the whole idea.

Its **in-edges** are `BACK`'s twin: an edge `p -> THINK` means *something at `p` made me stop and think*, and it
is taught by experience - `observe_think(p, amount)` creates it on first use and bumps it like an observed
transition (the sine model nudges its weight, the count and resonant models reward it), so it competes for `p`'s
probability with the real children. The events that teach it are what section 36 calls **triggers**: a voice
that caught itself repeating, a question asked of the model about a text, a thought questioning itself, a
question an LLM asked itself in its own thinking. `think_cost(p)` reads it back and `thinks_at(p)` says whether it
has become the cheapest way on from `p` (`BACK` does not count: it is not a way on).

Its **out-edges** are how *thoughts* begin. A thought is a text whose walk starts at `THINK` instead of `START`:
`observe_sequence(trigrams, origin=THINK)`, `trace(..., origin)`, `node_path(..., origin)` and
`GraphModel.train(texts, origin=THINK)` run the same structure, the same counting or gradient and the same
compression from the other sentinel, and only the first edge differs (`ORIGINS = (START, THINK)`). So the model
learns how its thoughts open and go on without a word of them leaking into what it says from `START`: a thought
and a text may share every node after their first, but `generate` never begins with a thought and a thought never
begins with a text.

The search treats the two sentinels differently on purpose (`search.onward`, section 7). `BACK` hands a branch
over when it is the cheapest child. `THINK` is dropped from the continuations and hands nothing over: where the
model has learned to stop and think, the walk carries on and the *thought* is somebody else's business -
`thinking.think` asks `thinks_at` along a finished path (section 36). Neither sentinel appears in a path; a walk
started at `THINK` is decoded exactly like one started at `START` (`_build_result` includes the context for both
origins, and the judged costs read `came_from = origin`).

`THINK`'s state is fixed at the *other* edge of the drawn range (`THINK_Z = -Z_RANGE`; `BACK_Z = +Z_RANGE`), so
adding it moved no random stream and its activation is as firmly non-zero as `BACK`'s. Graph format 4 carries it:
files written at format 3 gain an unvisited `THINK` on load and their real node ids shift up by one, older files
gain `BACK` first and then `THINK` (`_with_sentinel`, mirrored by the Go `withSentinels` and the Rust
`with_sentinels`); the sentinel takes `START`'s activation parameters, as `BACK` did. `check_invariants` now asks
for four live sentinels, and for `BACK` to have no children (it never continues).

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
    # (initialised like a new edge, edge_count = count[node_id]). B copies A's z/a/b/h/k, count and its resets.
    # trigram_index updated. Returns (A, B). Bad preconditions -> ValueError.

def merge_child(self, p: int) -> bool
    # Merge p's single child c into p when: p, c not in (START, END); p != c; len(children[p]) == 1;
    # len(parents[c]) == 1. New label = labels[p] + labels[c][2:]. count[p] = max(count[p], count[c]),
    # compared as the EXACT counts (node_count) and taking the resets along.
    # c's out-edges move to p (edge ids preserved). If c had a self-loop c->c it becomes p->p.
    # Edge p->c is tombstoned; c is tombstoned (alive False, dicts emptied); trigram_index entries
    # of c re-pointed to p with offset shifted by len(labels[p]) - 2 (computed BEFORE relabel).
    # Returns True if merged.

def compress(self) -> int
    # Repeatedly merge all mergeable unary chains until none remain. Returns number of merges.
    # With the dynamic window on (section 40), merge_child refuses a chain whose merged label would be longer
    # than the window's size: len(p) + len(c) - overlap > size.

def grams_held(self, node: int) -> int          # (label_len - n) // stride + 1
def longer_than(self, size: int) -> tuple[int, int]   # (real nodes longer than size, the longest label)
def split_window(self, size: int) -> int
    # The step of the dynamic window (section 40): halve every real node longer than `size` units at its middle
    # gram - split(node, ceil(grams / 2) * stride) - again while it is still longer, visiting ids in increasing
    # order (the second halves are appended and halved in turn); a node of one gram is left as it is.  Each bridge
    # A -> B is the heavy connection: the sine model sets its weight to W_HEAVY (negated while inverted), the
    # phase model gives it the traversals of the out-edges it stands before, every other kind the node's count
    # that split already gave it.  Returns the number of splits.  ValueError for size < 1.

def invert(self) -> None
    # w -> -w for every alive edge; a -> -a AND k -> -k for every node (incl. START/END), which is the exact
    # negation of the unit: f = a*sin(b(x-h))+k, so flipping `a` alone leaves -f + 2k and only negates while
    # k == 0 - and k is learned (df/dk = 1). Then inverted = not inverted, version += 1

def child_scores(self, p: int) -> list[tuple[int, float]]     # [(child_id, score)], score = w * f_p * f_c
def child_probs(self, p: int) -> list[tuple[int, float]]      # softmax over child_scores (numerically stable)
def child_costs(self, p: int) -> list[tuple[int, int, float]] # [(child_id, edge_id, -log prob)], cached by version

def to_csr(self) -> CSR              # see backend.CSR; over ALL node ids (dead ones have empty rows)
def apply_csr_weights(self, csr: CSR, weights: list[float]) -> None   # write back weights in CSR order; version += 1
def node_params(self) -> NodeParams  # copies of z, a, b, h, k in node-id order
def apply_node_params(self, params: NodeParams) -> None             # version += 1

def carry_counters(self, force: bool = False) -> int
    # The ONLY place the visit counters wrap: every counter that reached COUNTER_LIMIT goes back to 0 and
    # its reset is counted. Called at the end of every epoch and before a save; skipped after one comparison
    # while `traversals` (which bounds every counter) has not wrapped. Returns how many wrapped. Section 28.

def node_count(self, i: int) -> int ; def edge_traversals(self, e: int) -> int
    # The exact counts across every reset: resets * COUNTER_LIMIT + the reading.

def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> RadixCyclicGraph
    # JSON-serialisable, format_version 2. Dead nodes/edges are compacted on save (ids remapped),
    # START=0/END=1 preserved. Includes rng state (random.Random.getstate() converted to lists) so training
    # is reproducible after load, `version` / `structure_version` / `traversals` with their `*_resets`, and
    # `nodes.count_resets` / `edges.count_resets` - dense arrays written only once something has wrapped.
    # A format 1 file loads with all resets 0 and its counts carried on the way in.
```

### 5.3 Invariants (tests assert these after random sequences of observe/split/compress)

1. Every alive node except START/END has `len(label) >= 3`.
2. `trigram_index` maps every trigram of every alive node label exactly once; no trigram maps to a dead node; for `(n, o) = trigram_index[t]`: `labels[n][o:o+3] == t`.
3. Two alive nodes never contain the same trigram (a trigram lives in exactly one node).
4. For every alive edge `p->c` with neither endpoint a sentinel: `labels[p][-2:] == labels[c][:2]` (window overlap).
5. `children`/`parents` are mirror images; every alive edge appears in both; dead nodes have empty dicts.
6. After `compress()`, no unary chain remains (no non-sentinel p with exactly one child c != p, c non-sentinel, where c has exactly one parent) - except a chain the dynamic window holds apart: with the window on, one whose merged label would be longer than its size (section 40).
7. After `observe_sequence(encode(t))`, walking `t`'s trigrams through the index and decoding gives `t` (structure round trip), before and after `compress()`.
8. Every counter is non-negative, every stored reset count is in `1..COUNTER_LIMIT - 1` and belongs to a real id; after `carry_counters()` every reading is below `COUNTER_LIMIT`.

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

**Every walk asks `onward(child_costs(node))` first** (`dijkstra_predict`, `sample_walk` and `beam.py` alike).
`BACK` is not a continuation - it emits nothing and no text passes through it - so it never appears in a path;
but it *competes* with the real children for probability, and when it is the cheapest of them the model's most
likely next step at that node is to stop rather than carry on. `onward` then returns nothing: the branch offers
no children, Dijkstra relaxes none, the beam drops it and carries on with its others, and the sampler ends the
walk. That is the hand-over of section 5.1.1, performed by the search itself wherever it is walking - and on a
model that has never been taught a `BACK` edge it is exactly the old behaviour, because there is nothing to
find (`onward` returns the list it was given).

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

Both take an optional `costs: CostFn` - `(parent[, prev]) -> [(child, edge, cost)]` - in place of
`graph.child_costs`, and so do `beam.beam_predict` and the four phase searches of `phasesearch.py`.  That one
parameter is the **traversal** option of section 31: `None` is the model's own cost function and nothing about
any search changes; the punishment traversal hands in a cost function built from the penalties alone.

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
    order, curriculum, replay, replay_size, patience, min_delta   # how a run walks its texts (SPEC-SearchAndTraining.md)
    reverse: bool = False               # read every text backwards, in the encoding's units (the spec's section 9)
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
        # meta_add(meta, "twonrl_runs", 1)
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
                              # every lifetime counter is reported with its "<name>_resets" as well (section 28)

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
| `converse` | `--opening TEXT`, `--turns 6`, `--mode beam\|sample`, `--context 12`, `--max-length 60`, `--k 5`, `--beam`, `--temperature`, `--step-penalty`, `--speakers A,B`, `--partner FILE`, `--allow-repeats`, `--allow-word-repeats`, `--explore 3`, `--no-learn`, `--save` / `--out`, `--stream` | the model talks to itself (section 22); prints `speaker: text` lines with cost, probability, the picked-up words and flags (given / new topic / repeat), then the `radixnet feedback --bad-text …` command that punishes the duplicates it could not avoid; JSON: `turns`, `count`, `speakers`, `mode`, `opening`, `kind`, `partner_kind`, `repeats`, `transcript`. `--stream` prints the conversation as it happens (section 22.1): each turn the moment it is spoken and, dimmed on a terminal, the window before it - the context it continues, the draft it caught itself on, where it backed up to, what it found; with `--json` one JSON object per line and the document last, as `{"event": "done", ...}` |
| `2nrl` | `--bad FILE`, `--good FILE`, `--neg-epochs`, `--pos-epochs`, `--neg-lr`, `--pos-lr`, `--out` | runs two_nrl, saves |
| `invert` | `--out` | inverts and saves |
| `compress` | `--out` | compresses and saves, prints merges (within the dynamic window, when it is on) |
| `window` | `--on`, `--off`, `--top N`, `--floor N`, `--size N`, `--auto` / `--manual`, `--step [N]`, `--dry-run`, `--out` | the dynamic window (section 40): without options it is shown; the settings switch it on (at the ladder given over the one it had, else 32 down to 4) or off and save; `--step` merges what fits the window, halves every node that is longer, moves the window down the ladder, N times, and saves; JSON: `window` (the config), `changed`, `step`, `saved` |
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
| POST `/api/converse` | `{"opening","turns","mode","context","max_length","k","beam","temperature","step_penalty","seed","speakers","history","partner","avoid_repeats","avoid_word_repeats","explore","learn","think","think_depth"}` | `{"kind","partner","speakers","count","turns": [Turn.to_dict()],"repeats"}` — `partner` names another kind kept in memory (400 when it is not loaded); `history` continues a conversation and only the new turns are returned; `repeats` are the duplicates spoken anyway, ready for POST `/api/feedback` `bad` (section 22); `think` (default on) has a voice that caught itself repeating think before it backs up, `think_depth` deep, and its `rethink` carries the `thought` (section 36) |
| POST `/api/converse/stream` | the same body as `/api/converse` | the same conversation as it happens (section 22.1): `application/x-ndjson`, one JSON object per line - `look` / `draft` / `caught` / `backtrack` / `found` / `stuck` (the window) and `turn` (the answer), each with `index` and `speaker`, then `{"event": "done", ...}` with the `/api/converse` document; a request refused before the first line is an ordinary 400, a failure after it the last line `{"event": "error", "error"}`. `ApiHandler._send_stream` writes it chunked, the headers waiting for the first event (`StreamedResponse`) |
| POST `/api/think` | `{"about","mode","k","beam","max_length","temperature","step_penalty","seed","depth","questions","learn"}` | `{"kind", **Thought.to_dict()}` — one thought from the `THINK` sentinel (`thinking.think`, section 36): `about` thinks at the node where that text ends and teaches the model to stop and think there (`learn`, default on - a thought changes the model); `depth` / `questions` bound how it questions itself; 400 for a mode, a depth or a node that is not one |
| POST `/api/score` | `{"text"}` | score dict |
| GET `/api/encoding` | | `{"window","stride","overlap","start_label","end_label","back_label","think_label","configurable": false,"note"}` — the text encoding every kind shares. Read-only: the window is part of the model format, not a setting (section 31.4) |
| POST `/api/encoding/preview` | `{"text"}` | the same document plus `{"chars","windows","count","decoded","round_trip","unknown_windows","kind","path": {"known","reason","labels","node_ids","decoded","nodes","compressed"}}` — one text through the encoder, back through `Decoder.decode_trigrams`, and through the graph's own (possibly merged) node labels with `Decoder.decode_path`. `path.known` is false with the reason: shorter than one window, windows never seen (listed in `unknown_windows`), or a text every window of which is known that still does not run from START to END |
| GET `/api/model/attention` | | `{"kind","attention": {"on","blur","weights","ngram","stride","unit","units","applies","default_blur"}}` — the active model's attention band (section 38); `weights` is the band over one gram, null while it is off; `applies` is false for a kind that is never corrected |
| POST `/api/model/attention` | `{"on","blur"}` | `{"kind","attention","stats"}` — a blur alone switches the band on, `on: true` without one uses the blur it had (else `default_blur`), `on: false` switches it off; 400 for a blur outside `[0, 1]` and for a kind that is never corrected |
| POST `/api/model/attention/preview` | `{"wrong","right","blur"}` | `{"kind","attention","blur","weights","changes","wrong","right"}`, each side `{"text","units","grams","spans","writer","charges","focus","end"}` — where one correction would land, gram by gram, under the writer rule and under a band (`blur`, else the model's, else the default); changes nothing |
| GET `/api/model/window` | | `{"kind","window": {"on","top","floor","size","auto","sizes","next","unit","units","ngram","longer","longest","nodes","heavy","default_top","default_floor"}}` — the active model's dynamic window (section 40): the ladder of node sizes and where it stands; `longer` is how many real nodes a step would halve (null while off), `heavy` the sine model's bridge weight (null where the bridge is heavy by its count) |
| POST `/api/model/window` | `{"on","top","floor","size","auto"}` | `{"kind","window","stats"}` — `on: false` switches it off (the graph stays as it is); `on: true` or any setting switches it on at the values given over the ones it had (else 32 down to 4, at the top, stepping every epoch), a new top or floor keeping the size on the ladder; 400 for a size that is not a power of two, a floor over the top or a size off the ladder |
| POST `/api/model/window/step` | `{"steps"}` (default 1) | `{"kind","step": {"steps","sizes","from","to","merges","splits","nodes_before","nodes_after","edges_before","edges_after","window"},"window","stats"}` — the window stepped by hand: each step merges what fits, halves every node that is longer and moves the window down the ladder, back to the top from the floor; 400 while it is off or for fewer than one step; 409 while a job runs |
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
| POST `/api/speech/tutor` | the same audio forms (encoded and quizzed in one call) *or* `{"texts": ["<speech:…> aud:…"]}` for utterances already encoded, + `transcript`, `rate`, `codec`, `normalise`, `token`, `unique`, `lead`, `length`, `attempts`, `mode`, `temperature`, `threshold`, `listen_back`, `blame` | the recall tutor of section 26: `{"modality", "lessons", "report", "negative"}` — each lesson carries the exercise, what was written back, the mark out of 10, the reason it failed and the facts behind it; `blame` teaches the negative network and fills `negative` |
| POST `/api/images/tutor` | the same image forms *or* `{"texts": ["img:…"]}`, + `size`, `encoder`, `lead`, `length`, `attempts`, `mode`, `temperature`, `threshold`, `blame` | the same, for a picture it was shown |
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
* `PredictPanel.jsx` — prefix, length, mode (dijkstra / beam / sample; the count model's dijkstra is the beam search), K / beam width for beam, to-end, step penalty; shows continuation (prefix + highlighted continuation), cost, probability, path chips with per-step costs, and the top-K / bottom-K tables of a beam prediction (both models). A Like button (on the result and on every top / bottom row) rewards that text: `POST /api/feedback {good: [prefix + continuation]}` through the shared `useJob("feedback")` hook, i.e. `reward()` - a positive-phase pass for RadixNet, a traversal plus reward for the count model; the button shows the liked state and cannot reward the same text twice. With **Query backwards** on (`BackwardsField.jsx`, for a model trained backwards) the prefix is sent turned around in the model's units (`src/backwards.js`) and every answer shown turned back round - what the model says came before, highlighted, then the prefix - while a Like still rewards the model's own, backwards, text.
* `GeneratePanel.jsx` — prefix, count, max length, mode (beam = the K most likely complete texts from the prediction search, the default; sample; dijkstra), temperature; list of samples with cost and probability, thumbs up / down per sample (`RateButtons`). With **Query backwards** on, the prefix is sent turned around - every text then *ends* with it - and every sample is shown turned back round; a rating keeps the model's own text (what feedback trains on) and carries `shown`, which the ratings card lists.
* `ConversePanel.jsx` — the model talks to itself (section 22): opening line, turns, context, max length, mode (beam / sample), K, temperature, the two voices' names, "Second voice is" (the same model, or the other kind kept in memory - `GET /api/model` `in_memory`), "Avoid repeated words" (`avoid_word_repeats`), "Explore" (`explore`), "Learn where it goes round" (`learn`), "Think before backing up" (`think`) with its "Think depth" (`think_depth`), "Punish duplicates"; Start / Start over runs `POST /api/converse`, Continue sends the transcript as `history` and appends the new turns, Clear empties it. The chat view is **newest first**: a new turn is appended to the top of the `<ol reversed>` and pushes the older ones down, so the latest reply is where the eye already is and nothing scrolls (the `RateButtons` label and the key keep counting from the start of the conversation). It puts the first voice left and the second right, dims the picked-up context inside each bubble, shows cost / probability / skipped candidates and badges (given, new topic, repeat, repeats itself, thought again, N vetoed), spells out any second thoughts in the meta line (`rethinkSays`), and every turn has the thumbs. "Punish duplicates" (on by default) passes the response's `repeats` to `useRatings().punish`, so the utterances the model could only repeat are marked 👎 and "Train on ratings" runs the 2NRL negative phase on them; a note above the transcript says how many were marked. "Stream" (on by default) runs `POST /api/converse/stream` instead (`api.converseStream`, section 22.1): every `turn` event is appended the moment it is spoken, and while a voice speaks a `LiveTurn` sits at the top of the list - the draft it caught itself on with what it backed out of struck through and the way on it found underlined, its meta line saying what it is doing (`applyEvent` in `src/stream.js` folds the events into that state; `LineParser` there cuts the chunks into lines, both with tests in `test/stream.test.mjs`). A committed turn keeps the draft it caught itself on in its meta line. A server without the route answers 404 and the panel falls back to `POST /api/converse`.
* `ThinkPanel.jsx` — **Think** (section 36): one thought per press - "About" (think at the node where a text ends, and with "Learn where it thinks" teach the model to stop and think there), mode (beam / sample), K, max length, depth, questions, and temperature / seed for sample - through `POST /api/think`; the thoughts are listed newest first, each a `ThoughtView`: its summary, the thought, the trigger, the node, how it stopped, what it triggered, what it taught, its cost and the path from `<think>` as chips, with the questions it asked itself nested one level deeper each. A model taught no thoughts says so and points at the Ollama tab's thinking cards.
* `ThoughtView.jsx` + `src/thinking.js` — the thought records read for display: `ThoughtView` (a thought in full, recursive over its questions) and `ThoughtLine` (the 💭 line under a Converse or Chat turn whose rethink thought before backing up); `thinking.js` holds the pure helpers - `summarizeThought` (radixnet.thinking.summarize's wording), `questionsIn` / `questionRuns` (radixnet.thinking.questions_in, to mark the questions in an LLM's thinking), the request builders for `/api/think` and `/api/ollama/think`, `isSentinel` (node ids 0-3) - tested with `node --test` in `test/thinking.test.mjs`.
* `ChatPanel.jsx` — the model in conversation with an LLM that marks it (section 28): the settings, the live transcript, the table of conversations and the report card. Its transcript is **newest first** too - a new exchange is appended to the top of the `<ol reversed>` and pushes the older ones down, so a running conversation never has to be scrolled to (the partner's line stays directly above the reply it drew) - and a line above it says how many replies were duplicates punished with the failures.
* `RatingsCard.jsx` — shared by Generate and Converse: `useRatings()` (one rating per distinct text, toggling; `punish(texts)` marks a whole batch thumbs-down without toggling and returns how many), `RateButtons` (the thumbs pair) and the "Ratings → 2NRL" card (rated texts, the action that will run, epochs / learning rates / strength, Train on ratings → `POST /api/feedback` through the panel's `useJob("feedback")`, the job's phase table).
* `TwoNRLPanel.jsx` — bad textarea, good textarea, epochs/lrs; shows negative/positive losses; button to Invert manually.
* `EvolvePanel.jsx` — corpus textarea, samples, generations (blank = forever), start/stop; live SVG line chart of `gap` and `fake_score_mean` over generations + latest sample text.
* `CheckpointPanel.jsx` — list checkpoints, save checkpoint (tag), restore, save/load model path, reset.
* `SettingsPanel.jsx` — **Settings** (section 31.4): the site-wide settings of this browser, what every search and every run starts from. Five cards: the **traversal** (`TraversalFields.jsx`), **sampling and diversity** (`SearchFields.jsx`), **how a run walks its texts** (`TrainingPlanFields.jsx`) - each with a button back to its defaults - **backwards** (`BackwardsField.jsx`: whether Predict and Generate ask the model backwards), and **this browser** (how many settings are remembered, and a two-click button that forgets them all).
* `ModelSettingsPanel.jsx` — **Model settings** (section 31.4): what belongs to the model and is saved with it. Five cards: **this model** (kind, encoding, size, file, and the replay buffer from `/api/status` `replay`), a **new model** in any kind and encoding (`POST /api/reset {kind, encoding, seed}`, two clicks, the encoding checked before it is sent), the **score function** of whichever kind is active (`GET /api/model` → `weights`, applied with `POST /api/model/weights`; the count model's six settings, the resonant model's seven, and for a kind without one - the sine model, the negative network - the sentence saying why and where its own settings are), the **attention band** (`AttentionBandCard.jsx`, section 38: `GET` / `POST /api/model/attention`, the band drawn over one gram, and a correction previewed under both rules with `POST /api/model/attention/preview`), the **dynamic window** (`DynamicWindowCard.jsx`, section 40: `GET` / `POST /api/model/window` for the ladder of node sizes, where it stands and whether it steps by itself, the ladder drawn as its rungs, and a Step button that steps it by hand with `POST /api/model/window/step`, saying what the step did), and the **encoder / decoder** (`GET /api/encoding` for the unit, the n, the stride, the overlap and the four sentinels, each with what it means - asked again whenever the model changes - and `POST /api/encoding/preview` for one text through the encoder, back through the decoder and through the graph's own node labels, with the grams the model has never seen marked and a label longer than one gram shown as the merged chain it is).
* `TraversalFields.jsx` — the traversal and its two scales, reading and writing the shared setting, so the Settings, Predict and Generate tabs show one control in three places; `compact` drops the explanation for the action tabs.
* `SearchFields.jsx` — the sampling filters (top-K, top-p, min-p) and the beam's diversity (`SPEC-SearchAndTraining.md` sections 1-2), over the shared `useSiteSettings().search`. Given the `mode` the search will really run in, the compact version shows only what that mode reads - the filters for `sample`, the diversity for `beam`, nothing for the exact searches - and an out-of-range value as an error the tab refuses to send.
* `TrainingPlanFields.jsx` — the order, the curriculum, the replay and the early stop (sections 3-6), over `useSiteSettings().training`; on the Train tab it previews how many texts each epoch walks and how many buffered texts it rehearses, and says what the model's buffer holds.
* `BackwardsField.jsx` — **Query backwards** (`SPEC-SearchAndTraining.md` section 9), over the shared `useSiteSettings().backwards`: for a model trained with the Train tab's *Read every text backwards* (`reverse` on `/api/train`), Predict and Generate turn the query around before it is sent and the answer back when it comes (`src/backwards.js`, pure, tested by `test/backwards.test.mjs`); nothing about it is sent. `compact` is the line the action tabs show.
* `GraphView.jsx` — SVG rendering of `/api/graph` (circular layout, edge opacity by prob, node radius by count - the *exact* count, `counterTotal(count, count_resets)` -, hover label; the tooltips show a counter's resets once it has any). The sentinels are drawn gold - START and END always, BACK and THINK once they are among the most visited nodes.
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
* `RecallCard.jsx` — the **What does it remember?** card, shared by the Speech and Images tabs (section 26): the
  exercise settings (how much of the payload to ask for, the lead, attempts, mode, the pass mark, listen back, blame
  it), one button, and the marked table — what was asked about, the mark out of 10, the agreement, the verdict and
  the sentence saying what went wrong — with the report card above it and what the negative network was taught below.

### 13.1 Settings remembered in the browser (`src/storage.js`, `src/hooks/useStoredState.js`)

Every settings field of every panel keeps its value in `localStorage`, so the forms survive a reload.

```js
STORAGE_PREFIX = "radixnet.v1."        // the namespace; the version retires an old format wholesale
MAX_VALUE_CHARS = 256 * 1024           // larger values (a pasted corpus) are not stored

storageKey(name)                       // "radixnet.v1." + name
sameShape(value, fallback)             // a stored value is used only while it still looks like the default
readSetting(storage, name, fallback)   // missing / unreadable / reshaped -> fallback; objects merge into it
writeSetting(storage, name, value)     // -> stored?  (unserialisable, too large, quota full: false, never throws)
removeSetting / settingNames / clearSettings(storage)   // only this app's keys
browserStorage()                       // localStorage probed once with a real write, else null
```

The settings that belong to the **site** rather than to a panel - the traversal and its two scales, the sampling
filters and the beam's diversity, how a run walks its texts, and whether a query is asked backwards - are the
exception, and they are why `src/hooks/useSiteSettings.jsx` exists: `SiteSettingsProvider` holds them once at the
top of `App.jsx` (persisted with `useStoredState` under `site.search.*`, `site.train.*` and `site.query.backwards`;
the traversal keeps its `network.*` names through `useNetworkSettings.jsx`, which the site provider wraps) and every
panel reads them through `useSiteSettings()` - `{network, search, training, backwards}`, each group with its values,
their problems, whether any is on, `set`, `reset` and `body`, the request fields (`backwards` is `{on, set}`: it is
never sent, it turns the query and the answer around). The rules themselves - ranges, what a mode reads, what a body carries - are pure
functions in `src/settings.js`, tested by `test/settings.test.mjs`. Two panels cannot share a `useStoredState`
name - they would share the stored value but not the state, and every panel here stays mounted while hidden, so
they would drift apart within a session - so a shared setting has to live in one place with several doors into it.
Outside the provider the hooks return the defaults with no-op setters.

`useStoredState(name, initialValue)` is `useState` with that store behind it: the initial value is the stored
one when a value of the same shape exists, every later change is written back 250 ms after the last keystroke,
and the value just read is *not* written again - a panel nobody touched leaves no trace. `name` is
`"<panel>.<field>"` (`tutor.topic`, `train.epochs`, `codegen.settings`, `generate.ratings.negLr`); two mounted
components must not share one, so components used by several tabs take a `namespace` prop (`RatingsCard`,
`RecallCard`). Where the browser has no usable storage - a private window, blocked site data - every call falls
back and the hook is plain `useState`.

What is *not* a setting and never stored: results, transcripts and recordings, ratings, job state, server data
(the model list, presets, checkpoints) and uploaded-file selections (the file may be gone by the next visit). The
footer's "settings saved in this browser" button (`SettingsReset` in `App.jsx`, two clicks) calls `clearSettings`
and reloads. Tests: `frontend/test/storage.test.mjs` (`npm test`, `make frontend-test`) runs the store against a
`localStorage` stand-in, including ones whose methods throw.

Plain readable CSS, responsive (single column under 800px). No TypeScript.

---

## 14. Tests (unittest, no deps)

* `test_activation.py` — default equals `-sin(x/3)`; numeric derivative check for all partials.
* `test_encoding.py` — round trips, short text, unicode, decode_path both modes.
* `test_graph.py` — invariants §5.3 under randomised observe/split/compress; merge/split round trip; CSR export/import; to_dict/from_dict equality; invert flips signs; the three sentinels (a fresh graph holds them and nothing else, and neither splits nor merges).
* `test_backend.py` — loss decreases on a tiny fixed graph; gradient check vs finite differences (python backend); torch vs python equality when torch is importable (skip otherwise).
* `test_search.py` — Dijkstra returns the trained sequence; to_end; fallback on dead end; sample_walk terminates; `onward` hands over where the model has learned it goes round.
* `test_model.py` — train reduces loss & perplexity; predict reproduces a training text continuation; score higher for trained text than garbage; two_nrl makes garbage continuation less likely than before; invert twice is identity; save/load round trip preserves predictions; compression changes node count but not predictions of trained texts.
* `test_checkpoint.py` — rotation, latest pointer, load_latest, resume.
* `test_gan.py` — one generation runs, history record shape, stop_event honoured.
* `test_cli.py` — subprocess smoke test of train/predict/info/2nrl/checkpoints/bench with `--json`.
* `test_api.py` — server in a thread; health/status/train(job polling)/predict/generate/converse/score/2nrl/invert/compress/save/load/checkpoints/graph/evolve start-stop/static fallback.
* `test_wordnet.py` — word n-grams (section 34): the alphabet (every id round-trips, the surrogate block skipped, the cap), the whitespace split and the round trip that normalises whitespace, an unread word as a new id while training and `<unk>` after, the graph invariants over words, a repeated phrase compressed into one node, predictions and lengths counted in words, a correction aligned word by word, the `radixnet-word` file (the vocabulary back in order, a count reader refusing it, a truncated vocabulary refused) and the kind registry. The cross-language half is `test_go_parity.py::TestGoWordParity` and `test_rust_parity.py::TestRustWordParity`, with `go/radixnet/words_test.go` and `rust/tests/words.rs` on their own sides.
* `test_counter.py` — the cyclic counters of section 28: wrapping at the limit, exact totals across a reset, weights / shares / rankings unchanged by a wrap, the save-load round trip with reset fields, a format 1 file, the carry guard, the lifetime counters.
* `test_negative.py` — the negative network (section 24): the blame weight function, evidence as blame minus clearing, blaming / clearing / two_nrl / invert_paths, corrections (only the changed characters blamed, nothing correct created), `judge` (risk, peak, coverage, reasons, spans, the thresholds), `crossings`, prediction over the failure distribution, `forget`, the capped per-edge reasons and journal, persistence and the kind registry, the `/api/negative/*` routes and the CLI's `negative` group.
* `test_blame.py` — where the negatives come from (section 24.2): reason classification from a critique, severity from a rating, code reasons from the sandbox / style / judge, faults from the English tutor's lessons (the named mistake, the mark, the correction), from reviews and from attempts, `teach`, and the tutor / evolve / codegen hooks.
* `test_codegen.py` — code generation (section 17): the problem formats, the sandbox, the style check, the verdict, the trainer's phases, the API routes and the CLI.
* `test_duo.py` — the pair (section 24.3): the blame, peak and ratio rules with the coverage gate, strict, learn, `filter` / `generate` / `predict`, the count model as the positive half.
* `test_chat.py` — the chat loop (section 28): the conversation, the marking against the line answered, the blame, the 2NRL, a reply it could only repeat punished however well the judge marked it, the guard's veto, stalling, the report card, the endpoint and the CLI.
* `test_guard.py` — the guard (section 24.7): `ready` / `rank` / `converse`, the three service methods, the three routes and the three CLI commands, on by default and off on request.
* `test_speech.py` — the text format (packing, the header found behind a token and before a transcript, repair of a
  cut-off prediction), unique tokens, `speech_texts`, both codecs (mu-law beats linear 8-bit on quiet audio, byte
  round trips), resampling / downmixing / normalising, every WAV sample format the reader accepts (PCM 8/16/24/32,
  IEEE float 32/64, A-law, mu-law, WAVE_FORMAT_EXTENSIBLE) and the ffmpeg hand-off, the waveform round trip
  (encode → text → WAV → the same tone), transcription (a given transcript, the auto order, fakes with the
  openai-whisper / faster-whisper / OpenAI-server interfaces, the error when nothing is installed), `teach` (both
  texts share the token, pair / shared token / transcript-only, a failing ASR still teaches the waveform, the model
  learns and continues what was spoken), recording (no recorder, `arecord` called and its file read), the four API
  routes in all three body forms and the CLI actions. Runs without a microphone, ffmpeg or any ASR backend.
* `test_critic.py` — the negative network feeding itself (section 24.6): the config and what it refuses, one round
  (write → review → blame) against a scripted reviewer and against the fake Ollama server, the blame arriving sourced
  to the loop with the critique kept, the mark setting the severity, what it passes not being blamed, rounds differing
  because the seed advances, the report card and its trend, stopping between rounds rather than inside one,
  `rounds=0` meaning until stopped, the positive model coming out untouched, `external` wrapping the slow call, the
  `critic` job, the `/api/negative/auto` endpoint and the `negative auto` CLI (including `--json` staying parsable
  because the progress goes to stderr).
* `test_recall.py` — the recall tutor (section 26): the exercise (a spoken cue is the token and the header, an image
  cue carries a lead, texts that are not encoded are skipped), every way a completion can be wrong in both
  modalities (unreadable, truncated, overrun, garbled, silence / blank, clipping / noise, mishearing, distortion /
  drift, and the two cases where a flat reference means flat is *not* a fault), the agreement curve, the mark (the
  overrun penalty, the caps, the cost of a repair), the quiz over a real model and over a perfect one (attempts stop
  at the first pass, the stop event, progress, a capped quiz marked against what it asked for), the report card, the
  faults it produces and what reaches the negative network, and the two CLI commands and two endpoints. Needs
  neither Pillow nor a transcriber, so nothing in it is skipped.
* `test_thinking.py` — `questions_in` (sentences ending in `?`, a terminator run with no sentence, a comma question), `place` (the node cut so the prefix ends there; an unknown text and an empty prefix place nowhere), `think` (a model taught no thoughts has nothing to think with; asking about a text teaches the node to think and ends; learning off writes nothing; `think_on` teaches thoughts from `THINK` and not from `START`, and the questions they asked themselves; a thought questions itself where the model learned to think, once, one level down, saying something new, bounded by depth and count; a thought out of a repeat hands over to `BACK`; validation; the model file keeps its thoughts), `POST /api/think` and the conversation's thoughts, the `think` and `converse --no-think` commands.
* `test_dialogue.py` — `tail_context`, `Heard` (said / added / echo, and a longer utterance that only contains an earlier one), `stutter` / `stutter_at` (a run said twice in a row, where it starts saying it again, and the English that repeats a word and means it), `backtrack` (both kinds and where each is cut, what it keeps, what it explores, the words it may not rethink, a one-word line, the settings off, a voice with nowhere to go, a way out it has already said, and a conversation backing out of its repeats), `teach_back` (the node it teaches, the search refusing by itself after enough hand-overs, a conversation leaving the model knowing more, the learning off, and a repeat the graph cannot place), `repeats`, `converse`: alternating speakers, the opening as a given turn, every reply picks up (a whole-word part of) the previous line, no repeats / echoes in beam mode, a long conversation that runs out of new things to say (its duplicates flagged once each, and it stops rather than looping), no reply repeating its own words unless `avoid_word_repeats` is off (and a voice that can only stutter punished for it), determinism, history continuation, seeded sampling, speakers and a partner model, repeats on request, the empty model, validation; `stream` (section 22.1): the turns streamed are the turns returned, the window between two turns belongs to the one that follows and shows the backing up event for event against the rethink record, a `backtrack` streamed on its own, and streaming changing nothing (the same turns and the same graph afterwards). `test_api.py::test_converse_stream` reads the route's JSON Lines, `test_cli.py` the `--stream` output in both modes, and the Go / Rust sides are held to the same events by `test_go_parity.py::test_the_same_conversation_streamed` and `test_rust_parity_dialogue.py`.
* `test_tutor.py` — a fake Ollama plays the English teacher: `cue` / `overall_score` / the error-type mapping / the report card; the tolerant exercise and grade parsers; the marking (batches, an empty completion failed without a call, an unreadable answer left unrated); the loop over a real model and over a scripted one (what reaches the graph: corrections taught from their diff, weighted garbage for the rest and the mark-weighted rewards), adapting to the weakest points, drills, the dry run, per-lesson learning, the stop event, both model kinds; the next lesson plan (the weak points of a card, the upgrade ladder and the brief the marks write, the plan the marks alone imply, the tolerant plan parser, the teacher's plan merged with it - its brief kept, its difficulty ignored - a run that ends with one and a run taught to one); the auto run (batches that plan and apply themselves, per-batch report cards, `apply_plan`, stopping between batches, a batch that cannot be planned); the five endpoints and the CLI.
* `test_window.py` — the **dynamic window** (section 40, `../SPEC-DynamicWindow.md`): the ladder and what it refuses; `ABCD` cut into `AB` and `CD` under a grouping encoding and into `ABC` and `BCD` under the trigram; a node halved at its middle gram, again until it fits, never below one gram; the halves carrying the same state, parameters and count, joined by the heavy connection (the sine model's weight and its sign while inverted, the counting kinds' count, the phase model's through-traffic); compression stopping at the window and resuming when it is off; a step merging, halving and moving, and the top regrowing what stayed unary; the settings; the automatic step on every kind; the file block beside the band, and off being the old file to the bit; the CLI and the HTTP API. The cross-language half is `test_go_parity.py::test_the_dynamic_window_halves_the_same_nodes` and `test_rust_parity_tools.py::TestRustWindowParity / TestRustWindowRoutes`, with `go/radixnet/window_test.go`, `go/server/window_test.go` and the unit tests of `rust/src/window.rs` on their own sides.

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

### 16.1 Ollama (`ollama.py`) — prompt-driven corpora, an adversarial LLM review, letter-level correction

Three ways of hooking the network into a local LLM served by [Ollama](https://ollama.com) (HTTP API, stdlib `urllib` only):

1. **Corpus from a prompt** — the LLM is asked for *N* lines about a prompt, either correct (`style="good"`) or
   deliberately wrong (`style="garbage"`): exactly the two inputs of 2NRL. The lines can be trained on, saved as an
   upload, or written to a file.
2. **Adversarial review** — the LLM plays the harsh critic: every sample the network generates (or any given text)
   gets a rating 0–10, a pass/fail verdict against a threshold and a one-sentence critique. Failed texts become 2NRL
   garbage, passed texts (plus an optional corpus) the fine-tune pass: an external discriminator for the GAN loop.
3. **Letter-level correction** — the LLM plays the copy editor: every sample (or given text) comes back written out
   correctly with the *smallest possible change*, and the diff between the two (`diff.edits`, character encoding)
   is what the negative network learns — `"Hi howe are you??"` against `"Hi, how are you?"` is the `e` and the
   second `?`, not the sentence (section 24.2, `blame.faults_from_corrections`). A text handed back unchanged
   clears blame; one the answer said nothing usable about is neither blamed nor cleared.

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
CORRECTION_VERDICTS = ("corrected", "unchanged", "uncorrected")
def correct_texts(client, texts, *, context=None, model=None, batch=20) -> list[dict]
    # JSON-mode prompt at temperature 0 asking for the SMALLEST change and one word for the mistake out of
    # blame.CORRECTION_REASONS; each entry {"index","text","correction": str | None,"verdict","reason","note",
    # "changes": [{"op","wrong","right","at": [a0, a1],"to": [b0, b1]}] (equal runs dropped),"edits","wrong_chars",
    # "right_chars"}; a corrected text's reason is blame.correction_reason(word, note, changes), an unchanged one's
    # "none"; blank texts are "uncorrected" ("empty output") without asking the LLM
def adversarial_correction(model, client, *, count=8, prefix="", max_length=60, temperature=1.0, texts=None,
                           context=None, ollama_model=None, seed=None) -> dict          # summarise_corrections(...)
def summarise_corrections(source, model, texts, corrections) -> dict
    # {"source", "model", "texts", "corrections", "corrected", "unchanged", "uncorrected" (texts), "edits",
    #  "wrong_chars", "right_chars", "change_rate": corrected / answered | None}
```

**Thinking.** A thinking model (qwen3, deepseek-r1, gpt-oss, ...) reasons before it answers, and Ollama hands
the reasoning back beside the answer - as a `thinking` field when the server separates it, or inline between
`<think>` tags when the model wrote it into its answer. `OllamaClient.complete(prompt, think=...)` returns both
(`{"response", "thinking"}`; `split_thinking` cuts the inline form out, `think_value` reads the flag leniently:
`true` / `false` / a level `low` | `medium` | `high`, `None` leaves it to the model), and `generate` is `complete`
keeping only the answer. `thoughts_from_prompt(client, prompt, lines, think=, temperature=)` is two calls a
question: `questions_from_prompt` has the LLM write `lines` short questions about the prompt, and `complete`
answers each one with its thinking; every entry is `{"question", "thinking", "answer"}`, the thinking collapsed
to one line and `""` for a model that does not think. That thinking is what section 36 teaches the network as
thoughts of its own.

CLI: `radixnet [globals] ollama [--url URL] [--ollama-model NAME] [--timeout S] <action>`:

| action | options | behaviour |
|---|---|---|
| `models` | | list installed models |
| `corpus` | `--prompt TEXT`, `--lines 20`, `--style good\|garbage`, `--out FILE`, `--train`, `--epochs 10`, `--lr 0.5`, `--batch-size 4`, `--model-out PATH` | prints the lines; writes / trains on them |
| `review` | `--count 8`, `--prefix`, `--max-length 60`, `--temperature`, `--text ...` / `--data FILE`, `--threshold 6`, `--context`, `--2nrl`, `--good FILE`, 2NRL options, `--out` | table of ratings + summary; `--2nrl` runs failed→invert→passed and saves |
| `correct` | `--count 8`, `--prefix`, `--max-length 60`, `--temperature`, `--text ...` / `--data FILE`, `--context`, `--blame`, `--severity 1`, `--negative PATH` | table of corrections as diffs + summary; `--blame` blames only the changed characters (`blame.teach_corrections`) and saves the negative network |
| `think` | `--prompt TEXT`, `--lines 5` (questions to think about), `--think true\|false\|low\|medium\|high`, `--temperature 0.7`, `--out FILE`, `--train`, `--with-answers`, `--no-questions`, `--epochs 10`, `--lr 0.5`, `--batch-size 4`, `--model-out PATH` | asks for `--lines` questions about the prompt and has the model *think* about each one; prints question, thinking and answer; `--train` teaches the thinking as thoughts and the questions it asked itself as places to stop and think (`thinking.think_on`, section 36), `--with-answers` the answers as texts |

`serve --ollama-url --ollama-model` set the API defaults.

API (`ollama_url` / `ollama_model` on `ModelService` / `create_server`; `/api/status` gains `"ollama": {"url","model"}`):

| method & path | body | response |
|---|---|---|
| GET `/api/ollama/models?url=` | | 200 always: `{"available", "url", "model", "models": [...], "error": null\|str}` |
| POST `/api/ollama/corpus` | `{prompt, lines=20, style="good", model?, url?, timeout?, save_as?: upload name, train=false, epochs, lr, act_lr, batch_size}` | `{"prompt","style","model","url","lines","texts","upload": record\|null,"job": job\|null}`; 202 when a train job started; 400 bad input; 409 job running; 502 Ollama failure |
| POST `/api/ollama/review` | `{count=8, prefix="", max_length=60, temperature=1, texts?\|text?, threshold=6, context?, model?, url?, apply="none"\|"2nrl", good?, good_text?, good_files?, neg_epochs, pos_epochs, neg_lr, pos_lr, batch_size}` | `adversarial_review` result + `"url"` + `"job"`; `apply="2nrl"` starts a 2NRL job with bad = failed+unrated and good = passed + given (400 when either set is empty) |
| POST `/api/ollama/think` | `{prompt, lines=5, think=true\|false\|"low"\|"medium"\|"high", temperature=0.7, model?, url?, timeout?, save_as?, train=false, with_answers=false, questions=true, epochs, lr, batch_size}` | `{"prompt","model","url","think","count","thinking","thoughts": [{"question","thinking","answer"}],"upload","job"}`; 202 when a train job started (`thinking.think_on`, section 36); 400 bad input or a negative network; 502 when Ollama fails, writes no questions, or - with `train` - returned no thinking |
| POST `/api/ollama/correct` | `{count=8, prefix="", max_length=60, temperature=1, seed?, texts?\|text?, context?, model?, url?, timeout?, blame=false, severity=1}` | `adversarial_correction` result + `"url"`, `"severity"`, `"negative"` (`null`, or `ModelService.negative_teach_corrections`: `{blamed, cleared, unmatched, edges, edits, uncorrected, reasons, lessons, severity_mean, stats, reason_table}`); the model samples under its lock, the editor answers outside it; 502 on an Ollama failure |

Frontend: an "Ollama" tab with a connection card (URL, model list), "Corpus from a prompt" (generate → train / save as upload / hold as 2NRL bad or good), "Adversarial review" (ratings table, summary, apply as 2NRL with extra good files), "Letter-level correction" (every correction as a diff — `<del>` what the editor struck out of the model's text, `<ins>` what it wrote instead, a caret where a change leaves nothing on that side — with the verdict, the reason and the note, and a *teach the negative network* checkbox with its severity) and "Thinking from a prompt" (`POST /api/ollama/think`: questions, a thinking level - on, low, medium, high, off or the model's choice, sent as `"default"` - and a temperature; "Teach the thinking to the network" starts the train job on the same request, with its epochs, "Learn where it questions itself" and "Train the answers as texts too"; "What it thought" lists question, thinking and answer with the questions the model asked itself marked, the saved upload and the teaching job's epochs).

Docker: the API container gets `OLLAMA_HOST` (default `http://host.docker.internal:11434`, reachable through `extra_hosts`); the `ollama` profile runs the official `ollama/ollama` image with a model volume (`OLLAMA_HOST=http://ollama:11434`).

Tests (`tests/test_ollama.py`) use a fake Ollama server (stdlib `http.server`) that answers `/api/tags`, `/api/generate` (numbered lines for corpus prompts, JSON ratings for review prompts, JSON corrections for copy-editing prompts — `fake_copy_edit`: `howe` → `how`, `??` → `?`, a comma after an opening `Hi` —, questions for a thinking prompt, configurable failures) and `/api/chat`; asked to think, it answers with its thinking as Ollama's field, inline between `<think>` tags, or not at all.

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
@dataclass Lesson:    exercise, attempt, mode, continuation, sentence, cost, probability, reached_end, seconds, grade,
                      why, variants                     # the widening (below): filled only for a blamed failure
@dataclass Correction: wrong, right, weight             # what was written, what should have been, how much it counts

def write_exercises(client, topic, count=5, *, focus=None, level="beginner", weak=(), words="3 to 6", model=None)
    # JSON mode: {"exercises": [{"prefix", "focus", "answer"}]}; parse_exercises tolerates bare lists, plain lines,
    # "opening" / "stem" instead of "prefix", trailing punctuation, duplicates and an answer without the prefix
def drill_sentences(client, topic, count, *, weak=(), model=None) -> list[str]   # extra correct examples to imitate
def grade_completions(client, lessons, *, topic, threshold=6.0, grammar_weight=0.6, model=None, batch=10,
                      external=None, graded_by=None, think=None, thinking=None)
    # graded_by: the provider behind the client; think: the thinking level asked of the marker (None sends none)
    # one JSON call per batch: {"grades": [{"index","grammar","spelling","fluency","error","correction","comment"}]}
    # a client that says what it thought (Ollama's complete) appends it to `thinking`, one line per call
    # empty completion -> failed without asking (graded_by="empty", the exercise's own answer as the correction);
    # unreadable answer -> score None, graded_by="unrated", counted as a failure
def report_card(lessons) -> dict          # {"lessons","graded","passed","failed","pass_rate","mean_score",
                                          #  "mean_grammar","mean_spelling","mean_fluency","errors","weakest"}

DEFAULT_VARIANTS = 3;  MAX_VARIANTS = 10;  MAX_WHY_CHARS = 400
def explain_mistakes(client, lessons, *, topic="", count=3, weight=0.5, model=None, batch=10, temperature=0.9,
                     external=None) -> list[Lesson]
    # one JSON call per batch of failures: {"mistakes": [{"index", "why", "again": [{"wrong", "right"}]}]}
    # fills lesson.why (the rule that was broken, clipped) and lesson.variants (Correction(wrong, right, weight));
    # blank completions are skipped (nothing was written to be wrong about), repeats of the student's own sentence
    # and of each other are dropped, a "right" equal to its "wrong" is treated as absent, and an unreadable answer
    # leaves the lesson exactly as it was
```

**Why it is wrong, and the same mistake again.**  A mark says *that* a sentence is wrong; the negative network
(section 24) wants to know *why*, and wants the mistake more than once.  So step 4 of a round is
`TutorTrainer.widen()`: every failed sentence goes back to the teacher with the mistake it was marked for and the
correction, and comes back with the **rule** it broke and `variants` more short sentences that break the same rule
the same way, each with its own correct form.  They are ordinary faults to `blame.faults_from_lessons`
(section 24) - the same reason, the diff against their own correction, `variant_weight` (0.5) of the failure's
severity because the student never wrote them - and they reach nothing else: the model being taught never sees a
sentence it did not write.  The question is only asked when a negative network is attached (`--blame`) and
`variants` > 0, so a run without one asks nothing extra; a teacher that cannot answer costs the widening, not the
round (a `"note"` record).  Cost: one call per batch of failures.  The Go tutor does the same, word for word
(`ExplainMistakes`, `TutorTrainer.Widen`, `FaultsFromLessons`, `-variants` / `-variant-weight`): `tests/test_go_parity.py`
runs both with `--blame --variants 2` against one fake teacher and asserts the same conversation, the same family
on every lesson and byte-identical negative networks afterwards.

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

`TutorConfig` (validated like every other config) holds the topic, `rounds`, `batches`, `exercises`, `attempts`,
`focus`, `level`, `words`, `brief` (the previous batch's plan), the two providers and their model names (`tutor_provider`, `tutor_model`, `grader_provider`,
`grader_model`; `__post_init__` normalises the providers and fills in the models they imply, and
`resolved_grader_model` is what the marking runs on), the completion settings (`mode`, `length`, `max_length`, `temperature`,
`to_end`, `beam`), the marking settings (`threshold`, `grammar_weight`, `batch`, `adapt`, `drills`, `plan`, `teach_answer`,
`learn`), the teacher's thinking (`think`, `learn_thinking`, `think_questions`; `resolved_think` is what the marking
asks for - `think` as given, else on while `learn_thinking`, else nothing), how a correction is taught (`diff_corrections`, `keep_weight`) and the 2NRL settings (`twonrl_per`,
`min_weight`, `neg_epochs`, `pos_epochs`, `neg_lr`, `pos_lr`, `batch_size`, `strength`, `replay`, `replay_limit`,
`checkpoint_every`).

`TutorTrainer(model, client, config, external=None, grader_client=None).run(...)` per round — `client` is the
teacher, the marking shares it unless `grader_client` is given or `grader_provider` names the other provider (then
one is built from the environment), and every grade records which one marked it:

1. `set_exercises` — the teacher writes the openings (with the previous round's weakest points as the syllabus when
   `adapt`).
2. `complete` — `model.predict(exercise.cue, ...)` per attempt (attempt 0 in `mode`, later ones sampled).
3. `grade` — one call per `batch` sentences, asking the marker for `resolved_think`; what it thought comes back
   with the marks, one line per call.
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
5. `teach_thinking` — with `learn_thinking` (and `learn`), what the marker thought is taught to the network as
   thoughts: `think_on(model, thinking, questions=think_questions)` at the fine-tune pass's `pos_epochs`, `pos_lr`
   and `batch_size` (section 36.2), so every question the teacher asked itself marks a node where the network stops
   to think. A dry run records the thinking and teaches none of it.

`run()` is a loop over **batches**: one batch is `rounds` rounds and the report card over them (its own lessons,
not the run's).  With `batches` > 1 - or 0, which keeps going until the stop event - it closes the loop itself:
`plan` from that card, `apply_plan(plan)` (the brief becomes `config.brief`, the upgrade sets `level`, `words`,
`threshold` and `drills`, and `focus` is released because the brief carries the points of grammar in order), then
the next batch, taught to it.  A batch that cannot be planned ends the run rather than repeating itself; a stop
between batches never starts one, and the batch it stops in still reports.

Records: `{"kind": "lesson", batch, round, exercise, prefix, focus, attempt, mode, continuation, sentence, score, grammar,
spelling, fluency, passed, error, correction, changes, comment, graded_by, probability, seconds, why, variants}` (`changes` is what
the teacher changed, span by span, and rides on the `Lesson` itself so a dry run carries it too), `{"kind": "round",
...}` (the report card plus `action`, `bad`, `good`, `corrections`, `edits`, `penalised`, `rewarded`, `neg_loss`,
`pos_loss`, `mean_weight`, `mean_reward`, `drills`, and `thinking`, `thoughts`, `thought_questions`,
`thought_nodes` - what the marker thought and what it taught) and a final `{"kind": "report", rounds, ...}`.  With
`plan = N` (`--plan N`, `POST /api/tutor/start {"plan": N}`) one more record closes the run:
`{"kind": "plan", batch, rounds, ...LessonPlan.to_dict()}`, from `TutorTrainer.plan(card=None, count=N)` — the
run's own report card unless one is given.  A teacher that cannot plan costs only a `note`; the lessons stand.
An auto run plans between batches whether or not `plan` was asked for (`DEFAULT_PLAN_LESSONS` then), and announces
each new batch with `{"kind": "batch", batch, step, brief, topic, level, words, threshold, drills, note}` — what
`apply_plan` just set, so a reader always knows what is being taught.

With `--blame` (`TutorTrainer(negative=...)`, `POST /api/tutor/start {"blame": true}`) every failed sentence of a
round also teaches the **negative network** (section 24) why it failed: the mistake the teacher named is the reason,
its mark the severity, its sentence of teaching the note, and the correction is diffed so only the characters it
changed are blamed. The round records then carry `negative_blamed`, `negative_edges` and `negative_reasons`, plus
`explained` and `similar` — how many mistakes the teacher explained and how many more sentences it wrote that are
wrong in the same way (`--variants N`, `--variant-weight X`, `{"variants": N, "variant_weight": X}`; the lesson
records carry them as `why` and `variants`).

**What the teacher thought.**  A mark is the teacher's conclusion; a thinking model (qwen3, deepseek-r1, gpt-oss)
also reasons its way to it, and Ollama hands the reasoning back beside the answer (`complete`, section 16.1).  That
reasoning is the one supply of *thoughts* the tutor has for free: it is about exactly the sentences the network
wrote, and it asks itself the questions a teacher asks (*does every verb agree with its subject?*).  So the
marking call asks for it (`think`, `--think LEVEL`: `true`, `false`, `low`, `medium`, `high` or `default`, the
model's own choice), every round records it (`thinking`, whitespace collapsed to one line per call), and with
`learn_thinking` (`--learn-thinking`) step 5 teaches it as thoughts - walks from the THINK sentinel - with every
question in it a node where the network learns to stop and think (`think_questions`, `--no-think-questions` to
leave them out; section 36).  The network then learns the English it got wrong and how its teacher reasoned about
it, and `think` (or a voice that catches itself repeating) thinks in those thoughts.  Asking a model to think
costs time, so the marker is asked only when something reads the answer: `think` unset means on while the
thinking is taught and not asked otherwise, and `--think` alone shows the thinking without teaching it.  A dry run
shows it too.  Only an Ollama client says what it thought (`_complete` asks a client for `complete` and falls
back to `generate`), so a ChatGPT marker records nothing, and the negative network is refused before a lesson is
set - it judges; it does not think (400 from the API, an error from the CLI).  The ports do the same: Go's
`TutorConfig.Think` / `LearnThinking` / `ThinkQuestions`, `TutorTrainer.TeachThinking` and the optional
`thoughtfulClient` interface, Rust's `LlmClient::complete_thinking` and `TutorTrainer::teach_thinking`.

CLI `radixnet tutor` prints one row per marked sentence (round, exercise, score, grammar, spelling, fluency, mark,
mistake, sentence) with the correction and the teacher's line under a failure, a note per round and a report card at
the end; `--dry-run` marks without training or saving, `--report FILE` writes config, records, lessons, the card and
the plan.  `--plan [N]` (default 3) prints the planned lessons as a table, the step up, the brief, and the
`tutor --brief ...` command that teaches the next batch to it; `--brief TEXT` is the other end of that loop, and
`--batches N` (0 = until Ctrl-C) runs it automatically, printing one line per batch as it starts.

CLI flags for the teacher: `--tutor-provider ollama|chatgpt` (`--provider`), `--tutor-model`, `--grader-provider`,
`--grader-model`, `--url`, `--grader-url`; a `chatgpt` teacher without `$OPENAI_API_KEY` stops before anything is
sent.  For its thinking: `--think LEVEL`, `--learn-thinking`, `--no-think-questions`; the settings table says what
will happen to it (a `thinking` row), and every round prints what the marker thought and what the network learned
from it (*taught 1 thought and 2 questions it asked itself; it now stops to think at 2 more nodes*).

API: `GET /api/tutor` (defaults, error types, modes, `levels`, `plan_lessons`, and `providers` — each teacher's url,
model and whether it is `configured`), `POST /api/tutor/start` (job), `GET /api/tutor/history`,
`POST /api/tutor/lesson` (one round, no training; `prefixes` skips the exercise writer; `thinking` is what the
marker thought, empty unless `think` asked it to) and `POST /api/tutor/plan`
(`{report, count, topic, level, exercises, drills}` -> `{plan, source, provider, model, url, report}`; without
`report` the card at the end of the last run — `ModelService.tutor_card()`, the last `"report"` record of the
history — is used, and 400 says so when there is none). All of them take `tutor_provider` / `grader_provider` (and
`url` / `grader_url`); a `chatgpt` teacher on a server without a key is refused with 400.  `start` and `lesson`
take `think`, `learn_thinking` and `think_questions` (a `null` `think` is no `think`, as in every other body). The service releases the model
lock around every LLM call (`pause_lock`), so readers keep being served while the teacher thinks. The default
teacher is `$RADIXNET_TUTOR_MODEL`, else the model the server was started with (ChatGPT: the server's
`--chatgpt-model`).

Frontend: a **Tutor** tab (not Python-only - the Go server serves the same endpoints) with a **Teacher** selector
(Ollama or ChatGPT; the URL, the model placeholder and the notes follow it, and it says so when the server has no
key), the settings ("Teach corrections from the diff" and "Unchanged words keep" among them), "Dry run", a chart of the mean score and grammar
per round, the report card with the mistake histogram, a table of rounds (with what the corrections moved) and one
of every lesson (marks, mistake, what the network wrote, the correction, the changed words struck out against what
replaced them, the teacher's line, and - when the negative network is being taught - why it is wrong with the
sentences the teacher wrote that are wrong the same way, each struck out against its correct form; "Same mistake
again" and "Their blame" sit beside the blame checkbox).  **Plan** (`POST /api/tutor/plan`) sits under each report card - the dry run's
and the run's - and a **Lesson plan** card closes the tab: the summary, the step up (`hold` / `stretch` /
`advance` with its note), the brief for the next batch with **Teach the next batch** (which loads the brief and
the upgraded level, openings, pass mark and drills into the settings above), the weak points as pills, and a row
per planned lesson (the point of grammar, the mistake it fixes, the topic, the exercises, why) with *Use this
lesson*, which loads that one lesson instead.  Both leave **Start lessons** to the user.  The form's **Brief** box
and **Prefix words** field are `brief` and `words`; the **Plan** field is the run's own `plan` setting, so a run
that ends with a `"plan"` record shows it in the same card without asking again.  **Batches** is the auto run: the
button becomes *Start auto run*, the Rounds table gains a `batch` column, the report card is the last batch's, and
every `"batch"` record fills the brief, the level, the openings, the pass mark and the drills into the form (once
per batch) so the settings show what the server is teaching.  **The teacher's thinking** settings are *The marker
thinks* (the level; blank leaves it to the server, and the blank option says which way that goes), *Train on the
teacher's thinking* and *Learn where it questions itself* (`tutorThinkingBody` in `frontend/src/thinking.js`); a
dry run shows *What the marker thought* under its report card, and a run adds a `thoughts` column to the Rounds
table (`1 (+2?)`: one thought and the two questions it asked itself) and a list of what the teacher thought round
by round, the questions marked as the Ollama tab marks them (`ThinkingText`) with what the network learned from
them (`roundThinkingSays`).

Tests: `tests/test_tutor.py` (a fake Ollama that writes exercises, marks by a rule, explains a mistake and writes
it again, answers drill requests and plans the next lessons; the parsers, the marking, the widening -
`_parse_explanations`, `explain_mistakes`, `TutorTrainer.widen`, a round that widens and one that does not - the
planner, the loop with a scripted model, the endpoints and the CLI), `tests/test_blame.py` (a widened lesson
becomes its whole family of faults) and the same in Go (`go/radixnet/tutor_test.go`, `blame_test.go`), the ChatGPT teacher of
`tests/test_chatgpt.py` (the same lessons against the fake OpenAI, including a ChatGPT teacher marked by a local
model) and `go/radixnet/tutor_test.go` + `go/server/tutor_test.go` for the port.  The fake teacher thinks aloud
when asked (in Ollama's `thinking` field, or inline in `<think>` tags), so the thinking cases - asked or not, read
from either place, taught with and without its questions, shown by a dry run, a bad level, the negative network
refused - run in Python and Go, and `test_go_parity.py` / `test_rust_parity_teach.py` hold the Go and Rust
tutors to Python's requests, records and models with `--learn-thinking`, and to its refusal of the negative
network (the Rust one on the server as well).

### 16.4 Learning from a correction (`diff.py`, `CountRewardNet.correct`) — only what changed moves, counted per path

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

**Where a step's charge comes from** is the attention band's to say (section 38). Off - the default - it is the
rule above: a step answers, in full, for the units it writes. On, every changed unit hands out one charge, shared
among the grams that *see* it by how centrally each sees it, so the penalty is `strength * weight * charge`, the
fix earns `strength * reward * (charge + (1 - charge) * keep)`, and the path verdicts go to the step that saw the
change most sharply. The spans and the length are both in the encoding's units, so under a word encoding a
sentence that stopped too early is an insertion at its word count.

CLI `radixnet correct --wrong ... --right ...` (and `radixnet-count correct`) teaches one correction by hand and
`--dry-run` prints the alignment alone. Tests: `tests/test_countnet.py::TestCorrections` and
`go/radixnet/correct_test.go` (the alignment rebuilds both sentences, only the differing steps move, `keep` spreads
the rest, an early end blames the step into END, a run of corrections keeps the graph sound), plus the cross-language
parity case.

### 16.5 Judged paths (`CountRewardGraph.paths`) — the node that called the step

An edge is the right move in one sentence and the wrong one in another, so a reward counted per edge blurs the two
together. Every judgement is counted **per path** instead: the key is `(the node before the edge's parent, the
edge)` - the step in the company it kept - and the row is `[seen, correct, incorrect]`.

* **Who writes it.** `record_path(transitions, outcome, create)` walks one text's `(parent, edge)` steps and takes
  the context of step *k* from the parent of step *k-1* (START for the first, which has none). `reward` marks the
  whole walk correct, `punish` marks it incorrect, a plain training pass marks nothing; `mark_steps` judges single
  steps, which is what `correct` uses - the blamed steps incorrect, the fix correct, so **the counters follow the
  reward**. A whole path is rewarded only when the output was correct: `keep` (and the tutor's `keep_weight`)
  defaults to 0.
* **What it costs.** A context is *born* when a path is judged and is kept up to date by every later traversal;
  an unjudged pass never creates one (`create=False`), so training a corpus cannot fill the table with the
  second-order counts of a whole language. Two indexes (by edge, by caller) make the structural fix-ups cheap, and
  `nodes_with_paths()` is rebuilt lazily so a whole epoch of splits costs one rebuild.
* **What it changes.** `path_term = log((correct + s) / (incorrect + s))` (zero until judged, symmetric) is added
  to the edge's weight, times `path_scale` (1 by default, in the weight config and the file), before the softmax
  over the node's children: `child_costs(p, prev)`. The searches carry the node they came from - the beam reads it
  off its entry table, the sampler remembers its last step, and the Python Dijkstra puts it in the state key *only*
  for the nodes where it makes a difference, so the search does not grow anywhere else. Go prices every judged
  context once in `Prepare` (`ensureContextCosts`), because the two beams run side by side and must only read.
* **Through splits and merges.** A split re-keys a moved edge's contexts to the new node and hands the bridge edge
  the caller's counts (`q -> P -> c` becomes `q -> A -> B -> c`); a merge drops the contexts of the dying edge and
  of the steps that were never a choice (a unary chain has one way through) and re-keys the ones that arrived
  through the absorbed node. `edge_parent` is maintained alongside, as Go has always had `EdgeParent`.
* **What it reports.** `path_stats` adds `correct_ratio` (of the judged traffic) and `seen_ratio` (of the edge's
  traversals), `path_totals` the four counters that reach `stats()` as `path_contexts`, `path_judged`, `path_seen`,
  `path_correct`, `path_incorrect`; `radixnet paths` / `radixnet-count paths` and `GET /api/paths` list the
  contexts with their labels, and the status bar shows the totals. The file carries a `paths` block (prev, edge,
  seen, correct, incorrect; ids remapped and sorted, so both implementations write the same bytes).

Tests: `tests/test_countnet.py::TestPathCounters` and `go/radixnet/paths_test.go` (training judges nothing, a
judgement starts the table and a later pass only updates it, the same edge is right after one word and wrong after
another and its cost follows, the round trip, compression, the scale off), and
`tests/test_go_parity.py::test_judged_paths_price_the_same_step_differently` for the two implementations.

### 16.6 Node ratios (`CountRewardGraph.node_ratios`) — the same numbers, read from the node

16.5 keys a judgement by who called the step. This reads the graph the other way round: one node, and what its
traffic and its reward look like shared out over the nodes on either side of it. Nothing new is stored - it is a
view over `edge_count`, `edge_reward` and the path table - so it costs nothing until it is asked for.

* **The two sides.** `node_ratios(node)` returns `from` (a row per previous node: the in-edge that arrives) and
  `to` (a row per next node: the out-edge that leaves), plus `in_totals` / `out_totals` over each. The pairs are
  sorted by neighbour id before anything is summed, so both implementations add the shares up in the same order
  and write the same floats; the rows then come back most walked first (`-seen`, `-reward`, `node`).
* **The shares.** `seen_ratio` is the edge's share of the traversals on *its side* (the exact totals, so a wrapped
  counter still divides correctly). `reward_ratio` is its share of the reward on that side, taken over the
  magnitudes and kept **signed**: `r / sum|r|`, so a penalty reads as a negative share of the pressure on the node
  and an arm holding all of it reads ±1. The denominators are the side's own, not `count[node]`: a node is entered
  without an in-edge whenever a text starts on it.
* **The verdicts.** `edge_paths(edge)` sums the path table over every caller that reached an edge, giving
  `path_seen` (how much of the edge's traffic a judged context has been watching, `path_ratio` as a share of the
  edge's traversals) and `correct` / `incorrect` / `correct_ratio`. So 16.5 answers "was this step right *after
  that word*" and this answers "of everything leaving this node, how much went the way that was right". `BACK`
  (5.1.1) needs no special case: its edge is an ordinary one, so it reads as the share of the walks leaving the
  node that have learned to go round, beside the penalty on the step they were about to loop through.
* **What reports it.** `node_ratio_rows(limit, node)` orders the nodes by visit count; `radixnet nodes` /
  `radixnet-count nodes` (`--limit`, `--node LABEL`, a node label or a trigram it holds) print both sides, and
  `GET /api/nodes?limit=20&node=LABEL` returns them - 404 for a label the graph does not hold. The Graph tab
  fetches the same endpoint when a node is clicked. Counts are reported as the odometer reading plus its resets
  (`seen` / `seen_resets`, `visits` / `visit_resets`), the convention `/api/graph` already uses.

Tests: `tests/test_countnet.py::TestNodeRatios` and `go/radixnet/nodes_test.go` (each side's shares add up to one,
the reward share is signed and its magnitudes add up to one, the judged paths land on the arm that was walked, an
unjudged graph has the shares but no verdicts, the shares are of the side and not of the visits, the table is most
visited first), the endpoint in both server test suites, and
`tests/test_go_parity.py::test_a_node_is_read_the_same_way_from_both_sides`.

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
anywhere in the graph, default 10 000), plus the global `total_traversals` - a cyclic counter like every other
(section 28); the window counts are bounded by `window` and so need no wrapping. `observe_sequence(count=True)` is
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
stores it as an upload so several images can be trained on together), `POST /api/images/decode`,
`POST /api/images/tutor` (the recall tutor of section 26). CLI: `image info | encode | tutor | decode`. Frontend:
the Images tab (choose an image -> encoded text, latent size, copy, decode back to an image side by side with the
original; train on it; save as upload; ask it for the picture back and blame what it misremembered; decode any
pasted text such as a prediction).
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
   continuation) and, with `avoid_repeats`, does not duplicate the conversation is spoken:
   `text = context + continuation`.
3. Nothing (new) follows, or the context is not usable → the context loses its last word (`_shorter`) and the
   search runs again — what else could follow "on the"? — until it is empty.
4. Still nothing → the voice changes the subject: the same search from START (`fresh=True`); when even that only
   offers duplicates the best one is spoken and flagged `repeat=True`. A voice with nothing at all to say (an
   untrained model) ends the conversation early.

### Duplicates: avoided, and punished when they cannot be

`Heard` is what the conversation has already heard and decides what counts as a duplicate. An utterance duplicates
it when it was **said** before (`normalize`: whitespace collapsed, case folded), when its reply **adds** what some
earlier reply added — the same continuation reached from another context, "blowers" answered with "blower" twice —
or when it **echoes** a line already spoken (the whole utterance sits inside one of them: "the west" after "on the
west" adds nothing). A *longer* utterance that happens to contain an earlier one is not a duplicate: it says more
than was heard. `Heard.remember(text, reply)` takes a spoken turn in; `_pick` skips every duplicate candidate and
counts it in `skipped`.

A reply is also a duplicate of **itself** when it *stutters*: `stutter(text)` is the run of one to
`LONGEST_STUTTER` (4) words an utterance says twice in a row - "the **the** west", "say morning **morning**",
"**the cat** the cat sat" - the shape a cyclic graph falls into when it walks a loop instead of going somewhere.
Only an immediate repetition counts, so the English that repeats a word and means it ("where there is a will
there is a way", "a bird in the hand is worth two in the bush") is left alone. Every turn carries the `stutter`
flag whether or not the setting is on, so the transcript says what happened either way.

Both kinds are settings: **`avoid_repeats`** for what the conversation has heard, **`avoid_word_repeats`** for
what one utterance says twice (`--allow-repeats` / `--allow-word-repeats`, `"avoid_repeats"` /
`"avoid_word_repeats"` in the body, two checkboxes in the tab; both default to on). A kind that is switched off
is neither skipped nor counted as a repeat, and so is never punished.

### Second thoughts: noticing a repeat and exploring out of it (`backtrack`, `Rethink`)

Dropping the best continuation because it repeats throws away everything it got *right*: the words before the
repetition were said once and were the most likely thing to say. So a voice that catches itself repeating does
not simply take the next answer down the list - it backs up and looks for another way on.

`_pick` hands back `caught`: the best candidate rejected for repeating, either kind.
`backtrack(voice, text, keep, heard, added=…, explore=…)` then

1. **notices** what it was about to do, which decides where it backs up to:
   * a **stutter** - its own words twice in a row - is cut at `stutter_at(text)`, where the walk went round
     ("say morning **morning**" → after `"say morning "`): everything before that was said once;
   * a **repeat** of something the conversation has heard (`Heard.match`, which names the utterance, the added
     words or the line it would echo) is cut at `_last_word_at(text)`: the line is a retread from end to end, so
     it keeps as much of it as it can and differs at the last word - the latest point at which it still can;
2. **backs up** to exactly there and keeps it. `keep` is the context it picked up, which it may not rewrite: a
   voice rethinks what it said, never what it heard, and a repeat inside the other voice's words is recorded and
   left alone (`steps` 0), as is a one-word utterance with nothing to keep;
3. **explores** from the cut - the search runs again with that longer prefix, which *forces* the walk to leave
   the line at that point (asking the same question again from the context would only rank the same
   answers). Candidates that stutter, that the conversation has heard, or that the guard vetoes are passed over;
   the first one that says something new is spoken.
4. Nothing? Then it backs up one word further and looks wider - `k * (step + 2)` candidates, so the further back
   it goes the more it weighs - `explore` times over (default 3, `--explore N`, `"explore"`, an "Explore" field
   in the tab; 0 turns it off).

Each kind is only caught while its own setting is on (`avoid_word_repeats` for the stutter, `avoid_repeats` for
the repeat), and the `Rethink`'s `kind` says which it was, so the transcripts can tell *caught itself saying "ha"
twice* from *caught itself repeating "on the west"*.

A voice that finds a way out speaks it as an ordinary turn - no `repeat`, no `stutter`, nothing to punish - and
one that does not falls through to what it would have done anyway: the next answer down the list, a shorter
context, a fresh text, or the flagged repeat. Either way the turn carries a **`Rethink`**: `kind`, `noticed`
(the words it caught itself on), `cut` (what it kept), `steps`, `explored` (paths weighed), `found` and
`taught`. That
record is the turn's metacognition, and the CLIs and both tabs say it in a line - *caught itself saying "ha"
twice; kept "ha " and found another way on in 3 path(s)*, *caught itself repeating "on the west"; kept "on the ",
weighed 10 path(s), took a lesser answer*. One rethink per turn, so a conversation cannot spend
itself thinking; the paths it weighed are counted in `candidates`.

A found way on is a path explored from the cut, so its `cost` and `probability` (and `labels` / `node_ids` /
`step_costs`) are that walk's, measured from where the voice backed up rather than from the context.

### The rethink is how the graph learns where it goes round

A procedure that has to be re-run every time has learned nothing, so the last thing a rethink does is **teach**
(`teach_back`, on by default; `learn=False`, `--no-learn`, `"learn": false`, a checkbox). The node the voice
backed up to is located with `_prefix_start(cut)` and handed to `observe_back` (section 5.1.1) with the step it
was about to loop through and the step it took instead. Three edges move, and that is the whole of it -
`Rethink.taught` is the node, or `-1` when the graph could not place the repeat (it began inside the words the
voice picked up, or in text the graph does not know).

From then on the model needs the procedure less: after a few hand-overs at a node, `BACK` is its cheapest child
and the search stops walking through it *by itself*, in `predict` and `generate` as much as in a conversation.
Three conversations on the sample corpus teach 3, then 2, then 1 node - the trait is being learned, and the
rethinks are its teacher, not its mechanism.

Two consequences worth stating plainly. **A conversation changes the model**: two runs of the same conversation
differ, because the first one taught it something (the determinism that remains is "a model in the same state
says the same thing"). And the learning lives wherever the model does - the server keeps it in memory until
something saves, and the CLI says *it learned to hand over at N node(s); --save writes that into the model*.

Since the `THINK` sentinel (section 36) a rethink **thinks before it backs up** (`think_back`, on by default;
`think=False`, `--no-think`, `"think": false`): the node it backed up to is the node it thinks at, the rethink is
the thought's trigger (`"stutter"` or `"repeat"`), and when the thought stops it hands over to `BACK` with exactly
the lesson `teach_back` used to write directly - the three edges still move, only *after* thinking rather than in
place of it. `Rethink.thought` carries the `Thought` record (`None` when the voice did not think, or could not
place the repeat), the CLI prints its summary under the rethink's line and adds *it learned ... to stop and think
at M node(s)*; `learn=False` still thinks, and teaches nothing.

When they are all duplicates the best one is spoken anyway — the cheapest candidate that was never said word for
word, else the cheapest of all — and the turn is flagged `repeat=True`. Saying that same duplicate a second time
would only go round in circles, so `converse` ends the conversation there instead (before this, a model that ran
out of things to say repeated one line for every remaining turn). `repeats(turns)` collects the flagged utterances,
each once: the texts to **punish**. They ride back on `POST /api/converse` and the CLI's JSON as `repeats`; the CLI
prints the `radixnet feedback --bad-text …` command that punishes them, and the Converse tab's "Punish duplicates"
marks them 👎 so "Train on ratings" runs the 2NRL negative phase (`two_nrl`, section 8) on them — the model is taught out of
the duplicates the search cannot avoid by itself, instead of offering them again next time.

A `Turn` records `index`, `speaker`, `text`, the `context` it picked up, the `reply` it added, `cost`,
`probability` (`exp(-cost)`), `reached_end`, `fresh`, `given`, `repeat`, `stutter`, `rethink` (second thoughts,
above), `candidates` (continuations offered) and
`skipped` (rejected before the spoken one), plus the path (`labels`, `node_ids`, `step_costs`); `transcript(turns)`
renders `speaker: text` lines. Beam conversations are deterministic and never repeat themselves; the CLI `converse`
command, `POST /api/converse` (`ModelService.converse`, `partner` = another kind in memory) and the Converse tab
expose it, and turns are rated with the same thumbs as generated samples (`RatingsCard.jsx`).  `reply(voice,
previous, ...)` is one turn of that loop on its own, and is what section 28 calls when the other voice is an LLM
rather than a model.

### 22.1 Streaming the conversation: the turns, and the window between them

A conversation can be watched as it happens: `converse(..., stream=fn)` (and `reply`, `backtrack`) hand every
event to `fn` the moment it occurs (`StreamFn`, a callable taking one dict).  `converse --stream` prints them,
`POST /api/converse/stream` writes them as `application/x-ndjson` (one JSON object per line, chunked, each line
flushed as it is written) and the Converse tab's "Stream" draws them.  Streaming is a *view* of the conversation
and not a second procedure: the events are emitted from inside the one search `converse` always ran, the turns
returned are exactly the ones streamed, and a conversation streamed teaches the graph exactly what a silent one
does (`TestStream.test_streaming_changes_nothing`).

The stream has two layers, and the split is the design.

* **The committed layer is the answer.** `{"event": "turn", "index", "speaker", "turn": {...}}` is a turn the
  moment it is spoken - the opening included - and a turn is never taken back: `converse` only ever *adds* turns,
  so a client can append them to a transcript as they arrive.  The `turn` field is `Turn.to_dict()`, the document
  `/api/converse` writes.
* **Everything between two turns is the window - what a backtrack may still rewrite.** A voice about to
  speak looks for continuations of a context (`look`, `from`: the context, `""` for a fresh text from START), and
  the context loses a word at a time while nothing new follows it - so there is one `look` per context tried.
  A candidate caught repeating is streamed as the `draft` it was about to say (`text`, `cost`), then `caught`
  (`kind`, `noticed`, and `cut`: what it keeps, `""` when it cannot back up - the exploring is off, the repeat
  lies in the words it picked up, or nothing of its own comes before it), one `backtrack` per step back (`step`,
  `cut`, `wider`: the candidates it weighs from there), and `found` (`text`, `cost`, `explored`) or `stuck`
  (`explored`).  Every window event carries the `index` and `speaker` of the turn it belongs to, which is the
  turn event that follows it.

Why the window is the whole turn rather than the last few words: the obvious reading of "stream the text but
hold back a window the backtracking can rewrite" would commit a draft's beginning and keep only its tail open.
That would lie.  A stutter is cut where the walk went round, which can be anywhere in the draft; a repeat of a
line already heard is cut at its last word and then a word further back per step; and a voice that finds
nothing from any cut does not keep the draft's beginning either - it drops the draft altogether, tries a
shorter context, and finally changes the subject with a fresh text from START.  Nothing in a draft is certain
until the turn is spoken, so nothing in it is streamed as the answer.  The response streams by the turn, the
window streams apart from it, and a client that only wants the answer ignores everything but `turn`.

What the window buys is the point of streaming this model at all: the backtracking can be seen in action.
The CLI prints the window dimmed and indented above the turn it belongs to (*was about to say "ha ha"* / *caught
itself saying "ha" twice* / *backs up to "ha " and weighs up to 10 paths (step 1)* / *found another way on: "ha
and the cat sat" (3 path(s) weighed)*, then `A: ha and the cat sat`); the tab shows the draft with what it backed
out of struck through and the way on underlined, then commits the turn with its draft kept in the meta line.

The transport is the same on the three servers: chunked NDJSON, the headers sent with the first event so a
request refused before anything was streamed (a bad `k`, a partner not in memory) is an ordinary 400, and a
failure after the first line the stream's last event, `{"event": "error", "error"}` - the status line has
already gone.  The last line of a good stream is `{"event": "done", ...}` carrying the document `/api/converse`
answers with, so a client has the transcript, the repeats and the guard's report without a second request.
The conversation runs under the model lock and writes to the socket between searches; a client that goes away
does not stop it (Go and Rust remember the failed write and drop what follows; Python's `BrokenPipeError` ends
the request), and the model comes out of it as any conversation leaves it.

The Go port (`Stream func(map[string]any)` on `ConverseOptions` / `ReplyOptions` / `BacktrackOptions`,
`rConverseStream`, `converse -stream`) and the Rust port (`Stream`, a `dyn FnMut(Json)`; `Server::stream_route`
and a `Sink` that chunks the lines; `converse --stream`) emit the same events with the same fields, and
`test_go_parity.py::test_the_same_conversation_streamed` and `test_rust_parity_dialogue.py` hold all three to
one stream, event for event, the costs to `1e-9`.  D-081 records the decision.

## 23. The count / reward model in Go (`go/`) — goroutines over lines, paragraphs and pages

(The negative network is ported too; section 24.5 covers what it adds to the types below.)

`go/` is a standalone Go module (`github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go`, Go 1.24, no dependencies
beyond the standard library) porting section 19's model: `go/radixnet` is the library, `go/cmd/radixnet-count` the
CLI (`train`, `predict`, `generate`, `score`, `feedback`, `2nrl`, `invert`, `compress`, `weights`, `info`,
`converse`, `correct`, `negative`, `codegen`, `tutor`, `evolve`, `ollama`, `chatgpt`, `checkpoints`, `bench`,
`serve`). The
Python implementation is untouched; the two share the `radixnet-count` model file.

What is *not* ported:

* **RadixNet itself** (the sine-activation model of sections 3-8, its backends and the `radix` kind).  Go is
  deliberately the count / reward port; the two are different algorithms over the same graph, and a Go RadixNet
  would be a second gradient trainer rather than a translation.
* The **Stable Diffusion encoder** (`vision.py`'s `sd`), the **local Whisper backends** (`speech.py`'s
  `faster-whisper` / `whisper`) and the **torch backend**: Python ML packages, not code to translate.  The parts of
  those modules that *are* code - the waveform codecs, the text formats, the thumbnail encoder, an
  OpenAI-compatible transcription server - have no such excuse and are simply not done yet.
* Nothing else is Python-only any more: `tools.go`, `web.go`, `calc.go`, `toolbox.go` and `agent.go` port
  section 27, so the Go server serves `/api/tools/*` and `/api/agent/*` and the frontend hides no tab.  The
  `pythonOnly` list the Go server answers 404-with-a-message for is down to `/api/schedule/preview`.
* `schedule.py` is deliberately not ported, and would be dead code if it were: learning-rate schedules are a
  RadixNet feature, the count / reward model ignores learning rates entirely, and the Train tab already hides the
  schedule fields for it (`!countKind`).  Porting the expression evaluator would add a calculator nothing calls.

`codegen.go` ports section 17, sandbox included, and the sandbox is the reason it can be ported at all: the
programs are Python either way.  Go runs the *same bootstrap string* as Python does
(`python3 -I -B -c <bootstrap>` under `unshare -rn`, with the rlimits set in the child), so a program sees the
same interpreter, the same limits and the same isolation whichever language started it - and a traceback comes
back cleaned the same way.  The rest follows: `ParseProblems` / `ParseProblemFile` / `LoadProblems` read the same
problem files, `CheckStyle` shells out to a small Python helper for the AST rules (PEP 8 line and naming checks
cannot be done on the text alone), `ExtractCode`, `TeacherGenerate`, `TeacherFix`, `JudgeWithLLM` and `Decide`
speak the same prompts and the same verdict, and `CodeGenTrainer` runs the same teacher / model phases with the
same records (`attempt`, `problem`, `round`).  Two differences, both of them the count model's:
`CodeGenConfig.Strength` replaces Python's `neg_lr` / `pos_lr` / `batch_size` (the count model pushes by a
strength, not a learning rate - the frontend already sends `strength` for the count kind and now does so for the
Code tab too), and there is no `checkpoint_every` because the Go CLI has no checkpoint manager on this path.
`blame.go` gains `CodeReasons`, `CodeSeverity`, `CodeReason`, `FaultsFromAttempts` and `TeachAttempts`, so
`--blame` teaches the negative network why a program was rejected on both sides.  `cmd/radixnet-count/codegen.go`
is the CLI command and `server/codegen.go` the four endpoints (`/api/codegen/start|history|solve|run`), which
takes the **Code** tab off the frontend's `pythonOnly` list.
`tests/test_go_parity.py::TestGoCodeGenParity` points both trainers at one fake teacher and holds them to the
same conversation prompt for prompt (masking the two things that cannot match: the sandbox's scratch directory,
which is named in tracebacks, and how long a program took), the same solutions and the same blame on disk.

`speech.go`, `vision.go` and `recall.go` port the rest of the media path, and one thing about them is worth stating
because it is the only place in this port where the two sides do not produce identical bytes:

* **Speech is at parity, exactly.**  The waveform text has to be, because the utterance token is a digest of it and
  the token is *in the text the model trains on* - if the two sides encoded a recording differently the same file
  would teach two different things.  Getting there took two things Go does not hand you: the sample pipeline carries
  **float32**, not float64, because Python's `array("f")` rounds every stored value to single precision, and
  `blake2b.go` is BLAKE2b written out (RFC 7693), because Go's standard library has none and this module takes no
  dependencies.  That is the same reasoning that put MT19937 and Shewchuk summation here by hand.
  `tests/test_go_parity.py::TestGoMediaParity` checks the token, the texts and the decoded WAV byte for byte.
* **Image encoding is not at parity, and cannot cheaply be.**  Python's thumbnail resamples with Pillow's Lanczos
  filter; the Go one uses a box filter, because reproducing Pillow's exact coefficients and rounding *without
  Pillow to check against* would be a guess rather than a port.  So the same picture gives two different (equally
  valid) texts.  The format is shared - each side parses, trains on, predicts and decodes the other's image texts -
  and `DescribeVision` says so in its own output rather than leaving it to be discovered.

Files: `encoding.go` (code-point windows, `DecodePath`), `mt19937.go` (a Mersenne Twister with CPython's seeding,
53-bit doubles and `getstate()` layout - `rng_state` round-trips between the languages), `fsum.go` (Shewchuk's
exact summation, so path costs match `math.fsum` to the bit), `graph.go` (nodes, insertion-ordered adjacency -
Python dict order decides the softmax summation order -, trigram index, `Split` / `MergeChild` / `Compress`,
`ObserveSequence`, `Trace`, `CheckInvariants`), `weights.go` (counts, the sliding window, rewards, the dual frequency
function, lazy weights and edge costs), `search.go` (`PathResult`, `SampleWalk`), `beam.go` (`BeamPredict`),
`model.go` (training passes, feedback, prediction, generation, scoring, stats), `dialogue.go` (`Converse`, `Reply`,
`Heard`, `Stutter`, `Backtrack`, `TeachBack`, `Rethink`, `Repeats`, `Stream` - the same duplicate rules, second thoughts,
learning and stream of events as Python, checked by `test_go_parity.py`),
`json.go` (the file format), `parallel.go` (`parallelFor`, `parallelRanges`, `SplitTexts`), `counter.go` (the
cyclic counters of section 28, identical to `radixnet/counter.py`).

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
texts and `Uploads.Source` (ZIP / file sources) for `StartTrainSource`. The JSON upload forms, which carry the file
inline, are parsed whole - held in memory while they are, as the Python server holds them - but not capped either
(D-035). `GET /api/status` reports `heap_bytes` and `memory_limit_bytes` so the status bar can show how close a run
is to its ceiling.

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
with `--brief TEXT` both hand that brief to the exercise writer identically; `--batches 2` runs the whole loop on
both sides and the batch records - brief, step, level, openings, pass mark, drills - must match.
`TestGoChatParity` does the same for the chat loop (section 28.1): one fake partner and judge answers both CLIs,
and the prompts, the transcripts, the marks, the report card and both saved networks must match.

### 23.1 The encoding (`radixnet/encoding.py`, `go/radixnet/encoding.go`) — n-grams of any size, groups of letters, words

The character trigram of stride 1 (section 4, D-006) becomes the **default of a dial** (D-071), in *both*
implementations. An `Encoding` is three numbers, owned by the graph, fixed when the graph is created and written
into the model file:

```go
type Encoding struct {
    Unit   UnitKind // Chars ("char") or Words ("word")
    N      int      // units per gram: the n of the n-gram
    Stride int      // units between two consecutive grams; 1 <= Stride <= N
}

func DefaultEncoding() Encoding        // {Chars, 3, 1} - the trigram
func (e Encoding) Overlap() int        // N - Stride: what two consecutive grams share
func ParseEncoding(spec string) (Encoding, error)   // "word:2:1", "char:5:groups", "trigram", ...
```

| spec | what it is |
|---|---|
| `char:3:1` | the trigram, the default, the only encoding Python reads |
| `char:5:1` | a sliding window of five letters (overlap four) |
| `char:4:4`, `char:5:5` | **tokenisation**: non-overlapping groups of four or five letters |
| `char:6:3` | groups of six, half of each shared with the next |
| `word:2:1`, `word:3:1` | word bigrams and word trigrams |
| `word:1:1` | one word per node |

**Units, not characters.** `Encoding.Units(text)` returns a `Units` view - the normalised text plus the byte
offset of every unit - so `Slice(from, to)` is O(1) whichever the unit is. For `Chars` a unit is a code point and
the text is untouched; for `Words` a unit is a `strings.Fields` word and the text is normalised to single spaces
(a word encoding keeps the words, not the layout). Everything the graph measures moves to units:

| quantity | was | is |
|---|---|---|
| `Graph.labelLen[n]` | characters in the label | units in the label |
| `loc.off`, the offset of a gram in a node | characters | units (always a multiple of `Stride`) |
| `Split(node, i)` | split before character `i` | split before the gram starting at unit `i` |
| `PredictOptions.Length`, `MaxLength`, `Score.Chars` | characters | units (words, on a word model) |
| `ChangedSpans`, `stepsOver` | character spans | unit spans (`Encoding.Edits` aligns words on a word model) |

The three structural operations generalise exactly:

* **Encode** cuts grams at `0, Stride, 2·Stride, …` while a whole gram fits; the ragged tail is dropped, as the
  trigram encoding already drops the last two characters. `Encoding.Normalize(text)` is what survives, and is what
  a round trip through the graph is checked against (`CheckInvariants`).
* **Split** at unit `i` (a positive multiple of `Stride`) leaves `A = label[:i+Overlap]` - the text through the
  end of the gram before `i` - and `B = label[i:]`, so `A` and `B` share `Overlap` units, which is the invariant
  every edge is keyed on. With `Stride == N` they share nothing and the edge is a plain join.
* **Merge** writes `label(p) + label(c)[Overlap:]`, joined with the unit separator, and re-indexes `c`'s grams at
  `len(p) - Overlap + j`.

**Persistence.** The `encoding` block of the graph document is written **only when it is not the default**, so an
ordinary model file is byte for byte what it always was and the existing parity tests are unaffected. Both sides
read the block, so a word model written by either continues in the other:
`tests/test_go_parity.py::TestGoEncodingParity` trains the same corpus under nine encodings on both sides and
compares labels, counts, edges, weights and the RNG state, has each side continue the other's file, and holds the
two to the same prediction in words.

**Reach.** In Go: the library (`NewModel` / `NewGraph` via `GraphOptions.Encoding`, `NewNegativeModel` via
`NegativeOptions.Encoding`), the CLI (`--encoding SPEC`, `--units`, `--ngram`, `--stride`) and the HTTP API. In
Python: `RadixCyclicGraph(seed, encoding=)` and every subclass, `new_model(kind, encoding=)` and all four model
kinds (RadixNet, count, negative, resonant — they share the graph, so the dial reaches the sine and the phase
models too), the same four CLI flags and the same `POST /api/reset` fields. On both sides the flags are honoured
when a model is *created* and refused when they disagree with one that is loaded, `GET /api/status` reports
`encoding` / `unit` / `ngram` / `stride`, and a model that trains a discriminator (`evolve`) or a negative
network beside itself hands them its own encoding: two networks reading one text differently would be judging
different grams.

**Python specifics.** A unit view is whatever slices by unit — the `str` itself for characters (Python slices
strings by code point already), `text.split()` for words — so `len(view)` is the unit count of both and
`_piece(view, lo, hi)` writes a slice of either back out; that is the whole of the machinery the Go side needs a
`Units` index for. `diff.py` takes an `encoding=` on every entry point and aligns units rather than characters,
so a word model's correction marks whole words. `GraphModel._adopt(graph)` is what a loaded model goes through:
a model is constructed before its file's graph is read, so its encoder and decoder have to be rebuilt from the
graph that arrives, or a word graph would be fed character trigrams.

### 23.2 The Go HTTP server (`go/server`) and the frontend hookup

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
over; `/api/health`, `/api/status` and `/api/model` carry `engine: "go"`, `workers` and `goroutines`; the one endpoint
this server does not implement (`/api/schedule/preview`, which would be dead code here - the count model ignores
learning rates) answers 404 with a message naming the Python server rather than blankly.

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

Frontend (`App.jsx`): `engineOf(status, health)` reads the engine; a tab that needed the Python server would carry
`pythonOnly` and be neither shown nor mounted with `"go"` - there are none left, since both servers now run the
lessons, the evolve loop, the Ollama calls, code generation, tool use, the chat loop and the media encoders.  The header shows a
**Go engine · N goroutines** badge, the status bar
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
| `punish` / `reward` | thumbs down = `blame` (reason `thumbs-down`), thumbs up = `clear`; `weights=` rates them per text, one pass each |
| `two_nrl(bad, good, ...)` | blame `bad` (optionally per-text `bad_weights` scaling the severity), then clear `good` (`good_weights` scaling how much blame each clears); no inversion |
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
passed, the corrections themselves and the teacher's model answers all clear blame.  A lesson the teacher was asked
*why* about (`tutor.explain_mistakes`, section 16.3) also carries its `variants` - more sentences that make the same
mistake, each with its correct form - and every one becomes a fault of its own: the same reason, the teacher's
explanation as the note, `<source>:similar` as the source, the variant's own `weight` (0.5) times the failure's
severity, and its correct form both the diff to blame against and one more sentence that clears blame.  One mistake
therefore teaches the *shape* of the mistake: a sentence the network never wrote is already suspect (`judge`), while
nothing synthetic reaches the model being taught.
`classify(critique, verdict=, rating=)`
picks a reason out of `REASONS` (`empty`, `gibberish`, `repetition`, `truncated`, `grammar`, `spelling`,
`contradiction`, `false`, `incoherent`, `off-topic`, `other`, plus `unrated`) by matching the tutor's own words;
`severity_from_rating(rating, threshold)` maps 0 -> 2.0 and the pass threshold -> 0.25 (an unrated failure is 1.0);
`code_reason(attempt)` reads the sandbox, the style report and the judge into `CODE_REASONS` (`timeout`, `crash`,
`wrong-output`, `task-not-done`, `style`, `naming`) with the severities in `CODE_SEVERITY`; `recall_reason(facts)`
does the same for the speech and image tutors of section 26, reading the round trip through the codec into
`SPEECH_REASONS` (`unreadable`, `truncated`, `overrun`, `garbled`, `silence`, `clipping`, `mishearing`,
`distortion`) and `IMAGE_REASONS` (the same four, then `blank`, `noise`, `drift`), with `RECALL_TRUNCATED` /
`RECALL_OVERRUN` / `RECALL_AGREEMENT` as its boundaries.
`faults_from_corrections(corrections, severity=1.0)` reads the copy editor of section 16.1: every text the editor
changed is a fault carrying its `correction`, so `teach` routes it through `NegativeNet.correct` and only the
characters the editor struck out or replaced are blamed, at `CORRECTION_SEVERITY` (1.0) per corrected text; the
texts it handed back unchanged clear blame, and an `uncorrected` one is neither.  The reason comes from
`correction_reason(word, note, changes)`: the editor's own word when it is in `CORRECTION_REASONS` (`spelling`,
`punctuation`, `capitalisation`, `spacing`, `agreement`, `tense`, `article`, `preposition`, `plural`, `pronoun`,
`word-order`, `vocabulary`, `repetition`, `fragment`, `nonsense`, `grammar`; aliases such as `typo` and
`capitalization` are accepted, spaces become hyphens), else its note read through `classify` (`gibberish` and
`incoherent` map to `nonsense`, `truncated` to `fragment`), else the shape of the diff (`reason_from_changes`: every
edit is read for what it moved - punctuation only, spaces only, a change of case only, letters inside one word
(`spelling`), or letters with a space (`grammar`, a word came, went or moved) - and the widest kind wins, `grammar`
> `spelling` > `punctuation` > `spacing` > `capitalisation`), and `grammar` when even that says nothing; it is
never `none` for a text that changed.
`faults_from_reviews` / `faults_from_attempts` / `faults_from_lessons` / `faults_from_recall` /
`faults_from_corrections` turn a tutor's output
into `(faults, passed)`, and `teach(negative, faults, passed)` (with the wrappers `teach_reviews`,
`teach_attempts`, `teach_lessons`, `teach_recall` and `teach_corrections`, whose report adds `severity`, `edits` -
the changed units over every correction - and `uncorrected`) blames each fault - through `correct` when it carries
one - and clears the passes, returning `{blamed, cleared, unmatched, edges, reasons, severity_mean, records}`.

The call sites: `tutor --blame` / `TutorTrainer(negative=...)` / `POST /api/tutor/start {"blame": true}`, which
blames every failed sentence of a round with the mistake the teacher named and adds `negative_blamed` /
`negative_edges` / `negative_reasons` to its round records; `ollama review --blame` /
`POST /api/ollama/review {"blame": true}` (the reviewer), `ollama correct --blame` / `POST /api/ollama/correct
{"blame": true}` (the copy editor, whose diff is the lesson), `codegen --blame`
/ `CodeGenTrainer(negative=...)`, which blames the rejected attempts of every problem and adds `negative_blamed` /
`negative_reasons` to its problem records, `evolve --blame` / `Evolver(negative=...)`, which blames every fake the
discriminator scored below the real texts (reason `blatant` past `blatant_margin`, else `discriminator`, severity
`gap / margin` clamped to `[0.25, 2]`) and clears the real texts; `negative auto` / `POST /api/negative/auto`, the
loop of section 24.6 that keeps the reviewer running so nobody has to type a failure in; `speech tutor --blame` /
`image tutor --blame` /
`POST /api/speech/tutor {"blame": true}` / `POST /api/images/tutor {"blame": true}`, which ask the network for a
recording or a picture back and blame what it misremembered (section 26); and `POST /api/negative/blame` for a
person.

### 24.3 `duo.py` — the pair as a GAN at output time

`NegativeFilter(positive, negative, config)` puts the two networks on one output path (`FilterConfig`: `threshold`,
`min_coverage`, `ratio` (`None` = off), `peak` (`None` = off), `over_sample`, `strict`, `spans`, `learn`, `reason`,
`provenance`).  With `provenance` off the veto applies exactly as before but every verdict the pair reports is
`terse(verdict)` - `{text, decision, rule}` - and the guard's report (`ModelService._guard_report`, the CLI's
`_guard_doc`) is the counts alone, `{on, provenance: false, judged, vetoed, negative, config, ...}`, with neither
`rejected` nor `verdicts` listed: the answer is not followed by pages of judgement when only the answer was wanted.
`--no-provenance` (the guard flags and `negative filter`), `{"provenance": false}` on `generate` / `predict` /
`converse` / `negative/filter` (per answer) and `POST /api/negative/settings {"provenance": false}` (every answer,
`ModelService.guard_config`) set it; the frontend's *Say why it vetoed* checkboxes send it and `GuardNotice` shows
a count-only report without a *why* to open. `judge(text)` merges
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
  `ollama review`, `codegen` and `evolve`. `--kind negative` makes it the ordinary model.  The Go CLI
  (`radixnet-count negative <action>`, `tutor --blame`, `correct --blame`) has the same commands and flags
  (section 24.5).
* API: `GET /api/negative` (stats, reasons, journal, settings), `POST /api/negative/blame|clear|judge|filter|forget|
  settings|reset|save`, plus `POST /api/negative/auto` + `GET /api/negative/auto/history` (the loop of section 24.6).
  The service keeps exactly one negative network, in the same `_parked` store as the other
  kinds, so selecting the `negative` kind hands back that very object; `positive_model()` finds the model it filters
  (the active one, else a parked or saved positive kind, else 409). `POST /api/save` writes the negative network
  beside the model when it holds blame.
* Frontend: the **Negative** tab (`NegativePanel.jsx`) — **Automatic** (the reviewer on a loop, section 24.6; its
  job history also refreshes the tables below, so the tab fills in by itself), run the pair, judge a text with its
  blamed fragments marked (`<mark>`), blame / clear by hand, the reason table with per-reason *forget*, and the
  journal; the Ollama tab's review card has a *teach the negative network* checkbox and reports what it learned.
* Tests: `tests/test_negative.py` (weight function, evidence, blame / clear, corrections, judge, crossings,
  prediction, forget, inversion, persistence, the kind registry, the API routes, the CLI group),
  `tests/test_blame.py` (classification, severities, faults from lessons / reviews / attempts, teaching, and the
  tutor, evolve and codegen hooks) and `tests/test_duo.py` (all three rules, the coverage gate, strict, learn,
  generate / predict, the count model as the positive half).

### 24.5 The negative network in Go (`go/radixnet/negative.go`, `blame.go`, `duo.go`)

The Go port carries the negative network too, and the two implementations are held to the same numbers by
`tests/test_go_parity.py::TestGoNegativeParity`: the same blame, the same corrections, the same file layout and the
same verdicts character for character (the sentence a verdict carries is quoted the way Python's `repr()` quotes it,
`pythonRepr`).  Go has one `Graph` type rather than a class hierarchy, so the negative arrays hang off it in an
optional block:

* `Graph.Neg *NegativeData` (nil on a count graph, so the count model pays nothing for it) holds `Blame`, `Fails`,
  `Clear` and `Reasons` per edge - the last as a `[]ReasonBlame` of at most `MaxEdgeReasons` entries rather than a
  map, because a map per edge is most of a graph's memory - plus the reason registry and the totals.  `newEdge`
  grows them, `recomputeRow` routes to `recomputeNegativeRow`, `Invert` to `invertNegative`, `Configure` to
  `ConfigureNegative` and `MergeChild` refuses to merge across an edge that carries evidence (`blocksMerge`).
* `Model.Neg *Negative` holds the journal and the thresholds; `NewNegativeModel`, `IsNegative`, `Kind() ==
  "negative"`, `Blame`, `Clear`, `BlameCorrection`, `Judge`, `Crossings`, `Reasons`, `Recent`, `Forget` and
  `negativeStats` mirror the Python methods, and `stepsOver` (already there for the count model's `Correct`) marks
  the same characters for both corrections.
* `ToDoc` / `FromDoc` write and read the `radixnet-negative` format: `edges.blame|fails|clear|reasons`, the
  `weights` block with the blame function and the reason registry, and the model-level `log` and `filter` blocks.
  `Load` dispatches on the format, so either kind loads from either language.
* `blame.go` is the Python `blame.py`: `Classify`, `SeverityFromRating`, `FaultsFromLessons`, `Teach` and
  `TeachLessons` (the code-generation reasons stay in Python, which is where the sandbox lives).
  `TutorTrainer.Negative` + `TeachNegative` blame every failed sentence of a round.
* `duo.go` is `duo.py`: `FilterConfig` (pointers for "unset" and "off"), `NewFilter`, `Judge` with the blame, peak
  and ratio rules, `Filter`, `Generate`, `Predict` and `Describe`.
* `server/negative.go` adds the `/api/negative/*` endpoints and the tutor's `blame` flag; the service keeps one
  negative network beside its model path, `POST /api/save` writes it alongside the model, and the frontend's
  **Negative** tab is therefore no longer Python-only.
* Go tests: `radixnet/negative_test.go`, `radixnet/blame_test.go`, `radixnet/duo_test.go`,
  `server/negative_test.go`.

The loop of section 24.6 is ported too (`go/radixnet/critic.go`), and with it the two things it is built out of
that Go did not have: `review.go` is the Python `ollama.py` - `CorpusFromPrompt`, `ReviewTexts`, `SampleTexts`,
`AdversarialReview` and `SummariseReviews` - duck-typed on `LLMClient`, so ChatGPT reviews as happily as a local
model, and `blame.go` gains `FaultsFromReviews` / `TeachReviews`.  `negative auto`, `ollama models | corpus |
review`, `chatgpt models | ask`, `POST /api/negative/auto`, `GET /api/negative/auto/history` and the three
`/api/ollama/*` endpoints all speak the Python server's contract, and
`tests/test_go_parity.py::TestGoCriticParity` holds the two loops to the same prompts, the same marks, the same
rounds and the same blame on disk.

Porting it turned up one thing worth stating, because it applies to both languages: a round **samples under the
lock and reviews without it**.  Sampling walks the graph, so doing it with the model lock released is a data race
in Go and a torn read in Python; only the reviewer's thinking - a network call that touches nothing of ours -
belongs outside.  That is why `SummariseReviews` / `summarise_reviews` exists as a step of its own: the loop needs
the two halves separately, while a one-shot `AdversarialReview` can still do both.

---

### 24.6 The negative network feeding itself (`critic.py`) — the reviewer on a loop

Every source of negatives in section 24.2 arrives as a *side effect* of something else running: a tutor round, a code
problem, a review, a recall quiz.  That left the Negative tab as the one surface where a person had to type a failure
in by hand — which is exactly the kind of work a loop should be doing.

`Critic(model, negative, client, config, external=...)` is that loop.  One round is three steps and nothing else:

1. the **positive model** writes `count` texts of its own (`ollama.sample_texts`, a stochastic walk, optionally
   continuing `prefix`);
2. an **LLM reviewer** marks each one out of 10, passes or fails it against `threshold` and writes a one-sentence
   critique (`ollama.adversarial_review`).  The client is any `LLMClient`, so `provider` picks a local Ollama model
   (the default) or ChatGPT — `review_texts` only ever calls `generate`, which is why it never had to care;
3. the failures **blame** and the passes **clear** (`blame.teach_reviews`, source `critic`): the critique picks the
   reason through `classify`, the mark sets the severity through `severity_from_rating`.  Nothing new is invented —
   this is the section 24.2 path, driven on a timer instead of by hand.

With `correct=True` (`--correct`, `{"correct": true}`) step 2 is the copy editor of section 16.1 instead of the
critic: `ollama.correct_texts` writes each text out correctly with the smallest change it can, and step 3 is
`blame.teach_corrections` — only the characters the editor changed are blamed, at `severity` (1.0) per corrected
text, the unchanged texts clear, and no pass mark applies.  The round record then says `"mode": "correct"` (a
reviewing round says `"review"`) and carries `corrections`, `corrected`, `unchanged`, `uncorrected`, `edits`,
`wrong_chars` and `change_rate` (the share of answered texts the editor changed) in place of the reviews and marks;
the report card sums them and adds `change_rate` and `change_trend` (last round minus first — negative when the
editor has less to put right than it had at the start).

`CriticConfig` carries those knobs plus `context` (what the reviewer is *told* the texts are meant to be — its
yardstick, and worth setting, because "is this good?" means little without one), `clear_passes`, `epochs`, `seed`,
`correct` and `severity`.
The seed advances by the round number (`_seed`), so a seeded run is reproducible *and* its rounds differ; an unseeded
one leaves the sampling alone.

`run(rounds=None, progress=, stop_event=)` loops until the count is reached, or forever when it is 0 / `None`, and
checks the stop event **between** rounds, so stopping finishes the round it is in rather than abandoning a half-taught
one.  The records are the round records followed by one `report` (`report_card`): rounds, reviewed, blamed, cleared,
edges, the mean mark, the pass rate, the reason histogram, and `trend` — the last round's mean mark minus the first's,
positive when the reviewer is marking the model's output better than it did at the start.

Two invariants are worth stating because they are what make the loop safe to leave running:

* **the positive model is only read from.**  Nothing here trains, rewards, inverts or compresses it, so the loop can
  run beside anything that *is* teaching it without fighting over the model;
* **`external` wraps the slow call**, exactly as the English tutor does, so the server releases its model lock while
  the reviewer thinks and readers keep being served.

CLI: `negative auto` (`--rounds 0` runs until Ctrl-C, which finishes the round and saves; the per-round progress goes
to stderr as a notice, so `--json` still writes one document to stdout).  API: `POST /api/negative/auto` starts a
`critic` job over `ModelService.start_critic` — which quizzes `positive_model()` and teaches `negative_model()` —
and `GET /api/negative/auto/history` is its record.  Frontend: the Negative tab's **Automatic** card, whose job
history also drives a refresh of the reason table, the stats and the journal below it, so the tab fills in by itself
as rounds land.  Tests: `tests/test_critic.py` — the config, one round, what reaches the negative network, the mark
setting the severity, stopping between rounds, the report card, a real `OllamaClient` against the fake server, the
job, the endpoint and the CLI.

---

### 24.7 The guard — the pair on every output path

Section 24.3 built the pair and left it as something a caller had to *ask* for: the Negative tab, `negative filter`,
`POST /api/negative/filter`.  Everything else — `generate`, `predict`, `converse` — still handed out whatever the
positive model wrote, including the sentence the tutor had corrected an hour earlier.  The guard closes that: the
two networks now work **in tandem on every answer**, and turning the filter *off* is what takes a flag.

`ModelService.guard()` (Python) / `Service.guard()` (Go) builds the pair when there is something to guard with and
returns `None`/`nil` otherwise — which is the whole of its judgement:

* the **negative network is the active model** — it is generating *from* the failures, and there is no positive half
  to guard;
* there is **none in memory and none saved** beside the model path.  The guard never *creates* an empty negative
  network: an answer is not the place to bring one into being (it would also park a model the caller never asked
  for, and `/api/model` would start reporting it);
* the one there **has never been taught a failure** (`NegativeFilter.ready` / `Filter.Ready()`, i.e.
  `total_blame > 0`).  An empty network vetoes nothing, so a filter around one costs the over-sampling and buys
  nothing.

An unguarded answer is therefore exactly what it was before this section, byte for byte, and a `"guard": null` in
the payload says so.  What each path does when there *is* a guard:

| path | what the guard does |
| --- | --- |
| `generate` | `NegativeFilter.generate`: the model is asked for `count * over_sample` candidates and the survivors come back, cleanest first (a **stable** sort on `risk` alone, so among equally clean texts the model's own order wins — the veto re-ranks as little as it can).  Fewer than `count` texts is information, not an error |
| `predict` | `NegativeFilter.rank` / `Filter.Rank`: the search has already run, so the guard *re-ranks* what it offered rather than asking for more — one judgement per continuation and no second search.  `top` keeps the survivors and the best of them becomes the prediction; when none survives the prediction is the prefix and nothing else (`expanded` still reports the search that ran) |
| `converse` | `NegativeFilter.converse` / `Filter.Converse`: `dialogue.converse` gained a `veto` hook (`ConverseOptions.Veto` in Go) and a vetoed candidate is skipped exactly like one already said — except that it may not even be the fallback.  A turn whose every candidate is vetoed falls back as any dead end does (a shorter context, then a fresh text), and the conversation stops when there is nothing left that may be said.  Each `Turn` counts its own `vetoed` |

Two things stayed deliberately outside it.  The **critic loop** (section 24.6) samples the positive model directly
and must keep doing so: a reviewer that only ever saw what already passed the filter would have nothing to teach.
The same goes for the tutor, evolve and the benchmark — the guard is on the way *out*, not on the way in.

The answer always carries the guard's own report, so nothing is dropped silently: `{"on", "vetoed", "rejected"
(the full verdicts), "verdicts", "negative" (its stats), "config"}` plus, per path, `candidates` / `kept` /
`asked` / `rate` / `refusals`.  `NegativeFilter.generate` grew a `results` key for this — the walks behind the
surviving texts, so a caller can still report cost, probability and path; it is popped before the explicit
`/api/negative/filter` endpoint answers, and in Go `FilterOutcome.Results` is `json:"-"` for the same reason.

* Python: `guard=False` on `ModelService.generate|predict|converse` and `{"guard": false}` on the three endpoints;
  `--no-guard` on the `generate`, `predict` and `converse` CLI commands, with `--negative PATH`, `--threshold`,
  `--min-coverage` and `--over-sample` beside it (`add_guard_flags` / `open_guard`).
* Go: the same three service methods take a `guard bool` and return `(answer, report, error)`; the CLI has the same
  flags (`addGuardFlags` / `openGuard`), and `negativeModel()` gained a `negMu` around its lazy load, because the
  guard reaches it from the read-locked output path where several callers can arrive at once.
* Frontend: `GuardNotice.jsx` under the Generate, Predict and Converse tabs — "the negative network judged N
  candidates and vetoed M", with *why* opening the full verdicts and their blamed fragments — and a *Filter with
  the negative network* checkbox on each of the three forms.
* Tests: `tests/test_guard.py` (the filter's additions, the three service methods, the three routes, the three CLI
  commands), `go/radixnet/guard_test.go`, `go/server/guard_test.go`, and
  `tests/test_go_parity.py::TestGoNegativeParity` holds the two guards to the same vetoes, the same survivors and
  the same order.

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

CLI: `speech info | transcribe | teach | listen | tutor | decode` (`listen` records first, then runs `teach`;
`--train` trains on the texts and saves; `tutor` is the recall tutor of section 26). API: `GET /api/speech`,
`POST /api/speech/transcribe`, `POST /api/speech/teach`, `POST /api/speech/tutor`, `POST /api/speech/decode`
(section 12). Frontend: the Speech tab (section 13). Dependencies: none for the waveform,
the token and the browser's dictation; `pip install radixnet[speech]` (faster-whisper) or `radixnet[whisper]`
(openai-whisper) for server-side transcription, ffmpeg for audio formats other than WAV.

---

## 26. The recall tutor (`recall.py`) — the tutor that needs no teacher

Sections 16 and 24.2 all end at the same place: a negative needs something outside the network to look at an output
and say it was wrong, **and why**.  For English that something has to be an LLM, because nobody knows in advance
what the right sentence is.  For speech and images it does not, because **the right answer is on file**: an
utterance or a picture was *encoded* into text (sections 25 and 11) before it was trained on, so asking the network
to write that text out again is an exercise whose correction already exists.

That is the whole idea of this module.  It is the same three steps the English tutor takes (section 16.2), with the
codec in the marker's chair:

1. **the exercise** — the opening of a text the network was taught.  `cue_of(text, lead)` is everything up to the
   payload plus `lead` characters of it, and `DEFAULT_LEAD` differs by modality for one reason: a spoken text
   carries a token of its own (`<speech:9f2a1c7d> aud:mu:8000x1:`), so the exercise can be the token alone — *what
   did that utterance sound like?* — while an image header (`img:tiny:128x128:`) is the same for every picture of
   that size, so without a lead there is nothing to say which one is wanted (16 characters by default);
2. **the completion** — `quiz` runs the prediction search from the cue with a budget of `RECALL_OVERRUN` times what
   is left, so rambling is bounded but detectable.  `length` caps how many payload characters are asked for and
   **trims the reference to match**, because a capped quiz has to be a fair one;
3. **the marking** — `check_speech` / `check_image` parse what came back with the modality's own `parse_text`, and
   report facts rather than a verdict: `readable`, `repaired`, `written_bytes`, `expected_bytes`, `length_ratio`,
   `agreement`, `flat`, `extreme`, the same two for the reference, and for speech `said` / `heard` / `match`.

### Agreement, and the mark

`_agreement` is the mean over the payload that was asked for of `max(0, 1 - |written - true| / TOLERANCE)`, with
`TOLERANCE = 32` (one eighth of the byte range) and a byte that was never written counting as 0.  Two properties
matter.  Truncation needs no separate penalty — the bytes that never arrived are already total disagreement — and
the curve is *generous about the last bits of a quantised sample and unforgiving about noise*: an exact recall
scores 1, one off by a quantisation step scores ~0.95, and random bytes score ~0.12.

`mark(facts)` turns that into the mark out of 10, and adds the three things agreement cannot see:

* what the network wrote **past** the end.  Agreement only asks whether the true bytes came back, so without
  `score *= expected / written` a completion that recalled the payload perfectly and then rambled would score full
  marks and never be blamed;
* repairing the base64 costs a point;
* silence, railing and wrong words are **caps** (2, 3 and 4 out of 10) rather than deductions, because they are
  failures of a different kind from being slightly off — a waveform that decodes to nothing is not 90 % right
  however its bytes line up.

`grade(facts, correction, threshold)` is the marking proper: the mark, `passed`, the reason
(`blame.recall_reason`), the original text as the **correction** and one sentence of teaching.  A pass carries no
reason, no correction and no comment — exactly like the English tutor's grade, which is what lets
`blame.faults_from_recall` read a `RecallLesson` the same way it reads a `Lesson`.

### What can be wrong, in the order it matters

`blame.recall_reason(facts)` names the single worst thing, and the order is the point: a completion that cannot be
read at all is not *also* judged on its length, and one that stopped early is not blamed for the base64 it never
got to.

| reason (speech / image) | what it means |
|---|---|
| `unreadable` | it wrote something that is not an encoded waveform or image at all: the header is gone |
| `truncated` / `overrun` | the payload is shorter than `RECALL_TRUNCATED` of the original, or longer than `RECALL_OVERRUN` |
| `garbled` | the base64 had to be repaired before it could be read |
| `silence` / `blank` | it decodes to nothing: a waveform whose peak is under `_FLAT_PEAK`, an image whose bytes have no spread |
| `clipping` / `noise` | more than `_RAILED` of it sits against either end of the byte range |
| `mishearing` | *speech only*: it decodes to speech, but the round trip transcribes to different words than were said |
| `distortion` / `drift` | readable, the right length, decodes cleanly and simply wrong: agreement below `RECALL_AGREEMENT` |

Both `flat` and `extreme` are compared against the *reference*: a recording that really was silent is not blamed for
coming back silent.  `mishearing` only ever appears when both the words that were said and the words the round trip
heard are known — `listen_back` decodes the recalled waveform and transcribes it, which needs an ASR backend and is
therefore off by default.  Without it `match` is `None` and nothing is claimed about the words.

### What it costs to run

Nothing here needs Pillow, torch or a transcriber: the marking is over the payload bytes both codecs already
produce, so the whole module works in an environment where neither the diffusion weights nor Whisper exist (which
is why `tests/test_recall.py` skips nothing).  Decoding a *picture* to look at needs Pillow and `listen_back` needs
a transcriber, but grading does not.

The one real cost is the search: a second of 8 kHz audio is ~10 700 characters, so `length` is usually what you
want.  It is why the option exists, and why it trims the reference too.

### Feeding the negative network

`blame.faults_from_recall(lessons, threshold, source)` is `faults_from_lessons` for these lessons: the reason is the
grade's own error (falling back to `recall_reason` over the facts, then `DEFAULT_REASON`), the severity is
`severity_from_rating` of the mark — the same curve the English tutor's marks go through — and the original text
rides along as the correction.  So `teach` routes it to `NegativeNet.correct` and **only the characters the network
actually got wrong are blamed**; the payload it did remember, and the original itself, clear blame.
`blame.teach_recall` is the one call the CLI, the API and the frontend all make.

CLI: `speech tutor FILE...` and `image tutor FILE...`, sharing `_add_recall_options` (`--lead`, `--length`,
`--attempts`, `--mode`, `--temperature`, `--threshold`, `--train`, `--blame` / `--negative PATH`) and
`_recall_tutor`, which trains first when asked, quizzes, prints the table and the report card, and saves the
negative network beside the model.  API: `POST /api/speech/tutor` and `POST /api/images/tutor` over
`ModelService.recall_quiz`, which quizzes `positive_model()` (quizzing the negative network itself is meaningless)
and teaches `negative_model()` under the mutating lock when `blame` is set.  Frontend: the **What does it
remember?** card (`RecallCard.jsx`), shared by the Speech and Images tabs.  Tests: `tests/test_recall.py` — the
exercise, every way a completion can be wrong, the agreement curve, the mark, the loop over a real model, the
faults, the CLI and the endpoints.
## 27. Tool use (`tools.py`, `agent.py`) — the network browses, Ollama sets the bar and teaches

The network cannot *decide* to call a function; it can only emit characters. So a tool call is text it writes and
the observation is text it reads, which makes a whole attempt one ordinary training text — and therefore something
2NRL can reward or punish as a unit:

```
TASK: How many legs does a cat have?
<tool>web_search {"query": "cat anatomy legs"}</tool>
<result>1. Cat - Wikipedia - https://en.wikipedia.org/wiki/Cat ...</result>
<tool>web_fetch {"url": "https://en.wikipedia.org/wiki/Cat"}</tool>
<result>Cat - Wikipedia The cat is a small domesticated carnivorous mammal ... four legs ...</result>
<answer>A cat has four legs.</answer>
```

Around that the LLM plays four roles, and the role of solving the task is the last one it is given:

```
task -> acceptance criteria (LLM, written first) -> the network calls tools -> judge against the criteria (LLM)
     -> teach with the same real tools when it failed (LLM) -> train on the failures, invert, fine-tune (2NRL)
```

### 27.1 `tools.py` — the call format, the registry, browsing

```python
CALL_OPEN/CLOSE = "<tool>"/"</tool>" ; RESULT_OPEN/CLOSE ; ANSWER_OPEN/CLOSE ; DEFAULT_OBSERVATION_CHARS = 600
task_header(prompt) -> "TASK: <one line>\n"      # what every transcript starts with, and what `explore` continues
call_text(name, arguments) -> '<tool>name {"a": "b"}</tool>\n'        # one line, JSON arguments, sorted keys
result_text(output, limit=600) ; answer_text(answer) ; format_observation(result, limit)   # a failure -> "ERROR: ..."
transcript_text(prompt, [(name, arguments, result)], answer=None, limit=600) -> one training text
parse_arguments(raw, tool=None) -> dict   # JSON (truncated JSON repaired: the first line, then the missing "}),
                                          # `key=value` pairs, or a bare value for a single-argument tool
parse_call(text, toolbox=None) -> ToolCall | None   # the FIRST call; never drops what it cannot read: the
                                          # ToolCall carries `error` so the caller can hand it to the mediator
find_answer(text) -> str | None           # an unterminated <answer> still counts

class ToolError(Exception)                                   # unknown tool, bad arguments, refused request
Param(name, type: string|number|integer|boolean, description, required=True, default=None, enum=None)
    .schema() .coerce(value)
Tool(name, description, params=(), handler=None, network=False)
    .signature() -> "name(a: string, [b: integer])"          # the line the prompts and the CLI show
    .schema()    -> Ollama's / OpenAI's `tools` format       # what OllamaClient.chat(tools=...) sends
    .coerce(arguments) -> dict                               # types, defaults, case-insensitive names, unknown dropped
ToolCall(name, arguments, raw, span, error) ; ToolResult(tool, arguments, ok, output, error, seconds, meta)
class ToolBox(tools=())
    .register .remove .has .get .names .tools .describe .schemas .catalogue
    .call(name, arguments) -> ToolResult       # NEVER raises: a failure is a result the model and the LLM can read
    .run(call: ToolCall) -> ToolResult         # a broken call becomes a failed result
```

Browsing is `WebClient(timeout=20, max_bytes=2_000_000, max_redirects=4, user_agent=None, allow_private=False,
search_url=None)`: `http`/`https` only, no credentials in the URL, no address that resolves into a private,
loopback, link-local, reserved or multicast range (`allow_private` is for a local test server), redirects followed
by hand so every hop is checked again, a byte cap and a timeout. `.fetch(url)` is the raw response, `.page(url)`
adds `html_to_text` (title, text, absolute `http(s)` links, each once and none back into the page itself) and
`.search(query, limit)` reads a JSON search API (SearxNG, Brave, MediaWiki, OpenSearch, ...) or an engine's HTML,
unwrapping DuckDuckGo's redirects, dropping links back into the engine and taking a hit's snippet from the engine's
other links to it. `search_url` may name several endpoints separated by spaces, tried in turn: one that refuses (an
HTTP error, DuckDuckGo's `202 Accepted` or a CAPTCHA page) is passed over for the next, and the default is
DuckDuckGo's HTML page, its lite page and Wikipedia's search API (`DEFAULT_SEARCH_ENGINES`).
`html_to_text` reads a page the way a reader meets it: what is never shown (`<head>`, `<script>`, `hidden`,
`display: none`, `aria-hidden`) and the page's chrome (`<nav>`, `<footer>`, `<aside>`, the page-level `<header>`,
buttons, menus, ARIA navigation / banner / search roles) are dropped with their links; the lines follow the blocks
(`<pre>` keeps its own); and runs of link-only lines (menus, language lists) and whatever lies outside `<main>` (or
`<article>`) move after the rest, links likewise, so the first few hundred characters - what the agent's transcript
keeps of an observation - are what the page is about. `<meta>` and `<link>` are void: counting them as open drops
once made every page that has one read as empty. `$RADIXNET_SEARCH_URL` (`{query}` is substituted) and
`$RADIXNET_USER_AGENT` are the defaults.

`default_toolbox(web=None, *, sandbox=None, upload_dir=None, offline=False, extra=()) -> ToolBox` installs
`web_search(query, [limit])`, `web_fetch(url, [max_chars])` (its `meta.links` is what an exploration follows next),
`web_links(url, [limit])`, `calculator(expression)` (`safe_eval`: an AST walk that allows numbers, operators and the
`math` functions and nothing else), plus `python(code)` when a :class:`~radixnet.codegen.Sandbox` is given and
`read_file(name, [max_chars])` when an upload directory is.

### 27.2 `agent.py` — criteria, mediation, judging, teaching

```python
DEFAULT_AGENT_MODEL = $RADIXNET_AGENT_MODEL or the Ollama default ; PHASES = ("teacher", "model")
MEDIATION = ("repair", "always", "never") ; MIN_EMISSION = 48 ; MAX_CRITERIA = 8
Task(id, prompt, criteria=(), answer=None, seeds=()) ; parse_tasks ; parse_task_file(.txt|.json|.jsonl) ; load_tasks
Step(index, call, result, source: "model"|"mediator"|"teacher", emission) ; Verdict(correct, score, met, critique, issues, judged_by)
Attempt(index, source, steps, answer, text, verdict, seconds)  # .calls .own_calls .autonomy .feedback()
Failure(text, gap, task, source)          # a failed transcript and how badly it failed (gap in [0, 1])

write_criteria(client, task, *, model=None, count=4) -> list[str]
    # JSON-mode, written BEFORE anything is attempted, so the bar does not depend on what the network produced.
    # A task that carries its own criteria keeps them (the LLM is not asked at all); an unusable answer falls
    # back to one criterion naming the task.
mediate_call(client, emission, toolbox, task, transcript="", *, model=None, hint=None) -> ToolCall
    # Ollama's native tool calling (chat with `tools`) first, then JSON mode for a model without it, then a
    # fallback to the first network tool with the task as its argument. The arguments are coerced against the
    # real schema; what still fails is reported on the call and executed as a failure.
judge_attempt(client, task, criteria, transcript, answer, *, model=None) -> {"met", "score", "correct", "critique"}
    # marks are read all-or-nothing (`_marks`): a partly readable list would misalign with the criteria
teach_task(client, task, toolbox, *, model=None, max_steps=6, criteria=None, feedback=None, external, observation_chars)
    -> (steps, answer)    # the LLM solves it with the REAL tools, so the demonstration is of things that happened
propose_task(client, emission, *, model=None, frontier=(), visited=(), index=1) -> Task   # exploration (24.4)
```

`AgentConfig`: `agent_model`, `judge_model`, `phases=("model",)`, `rounds=1`, `max_steps=6` (tool calls per
attempt), `model_attempts=2`, `teacher_attempts=1`, `first_attempt_dijkstra=True`, `candidates=5`,
`temperature=1.0`, `max_length=200` (characters per step), `mediation="repair"`, `criteria_count=4`, `strict=True`,
`use_judge=True`, `teach_on_failure=True`, `observation_chars=600`, `twonrl_per="task"|"round"`, `replay=True`,
`replay_limit=64`, `read_reward=False`, `blatant_mode="fail_invert"`, `blatant_margin=0.5`, `blatant_boost=4.0`,
`pass_score=6.0`, `neg_epochs=2`, `pos_epochs=3`, `neg_lr=0.5`, `pos_lr=0.1`, `batch_size=4`, `checkpoint_every=0`,
`seed=0`.

`AgentTrainer(model, client, toolbox, config=None, external=None)` (`external` is the context-manager factory the
API passes as `ModelService.pause_lock`, entered around tool and LLM calls so readers are served while the network
browses):

* `_emissions(transcript, attempt_index)` — what the network offers to write next, most likely first. The first
  attempt is the **beam search** (`predict(mode="beam", length=MIN_EMISSION, k=candidates)`), later attempts are
  samples; candidates that closed their tag come first, because an unterminated call only *started* one. A single
  cheapest path is no good here: Dijkstra accepts reaching END as a goal, so it answers with a couple of characters
  that can never be a whole call.
* `_next_call` — the first candidate that parses into a usable call (or gives an `<answer>`) is the network's own
  move; only when none of them can be used does the mediator step in, and the repaired call is executed *and
  written into the transcript*, so what the network learns is always well-formed. `mediation="always"` never asks
  the network, `"never"` executes its broken call as a failure instead of repairing it.
* `solve_with_model` / `solve_with_teacher` / `judge` / `criteria_for` — one attempt, one demonstration, one
  verdict, the criteria (written once per task and remembered).
* `run(tasks, progress=None, stop_event=None, checkpoint_manager=None)` — every round and phase; records of kind
  `criteria`, `step`, `attempt`, `task` and `round`. A task record carries `correct`, `solved_by`, `model_solved`,
  `taught`, `calls`, `own_calls`, **`autonomy`** (the share of calls the network wrote itself — the number that
  says whether it is learning), `failures`, `blatant`, `boost_mean` / `boost_max`, `action` and `gap_max`.

### 27.3 Failure first: how badly it failed decides how hard it is trained

Section 9.1 applied to tool use. `gap_of(attempt, criteria)` puts every failure on `[0, 1]`: the share of the
acceptance criteria it missed, or how far below `pass_score` the judge put it, whichever is worse; an attempt that
answered nothing has a gap of 1, and every failure keeps a floor of 0.1 so that a near miss still trains. Then
`learn(failures, good)`:

* `blatant_mode="fail_invert"` (the default) — `two_nrl(bad, good, bad_weights=[min(blatant_boost, 1 + gap /
  blatant_margin)])`: the negative phase runs one pass per distinct weight, heaviest first, so the worse an attempt
  was the harder the network is pushed to reproduce it — to fail blatantly on purpose — before `invert()` turns
  that into avoidance and the positive phase fine-tunes on the transcripts that were judged correct.
* `"activation"` / `"state"` — `invert_paths(failures, mode, amounts=[min(1, gap / (2 * blatant_margin))])`
  instead: every other node along a failed transcript is moved toward its negation, and blatant failures
  (`gap > blatant_margin`) leave the 2NRL garbage set.
* `"none"` — plain unweighted 2NRL.

Nothing failed -> the correct transcripts are rewarded and **nothing is inverted**. Nothing was right -> the
weighted negative passes run and the network is inverted (`_punish_weighted`). A transcript that is both a failure
and a success (the same text reached twice) is never trained as garbage. `replay` adds earlier correct transcripts
to every positive phase; `read_reward` adds the text of the pages that were read.

### 27.4 `explore` — the network chooses its own tasks

`AgentTrainer.explore(steps=10, progress=None, stop_event=None, checkpoint_manager=None)`; `steps` `None` or `0`
runs until the stop event, like the GAN loop. Each step:

1. `propose()` — the network continues `TASK:` (the prefix every transcript it has learned starts with) into
   whatever it is reaching for, and `propose_task` turns that emission into one concrete question browsing can
   settle, preferring the pages it has come across but not read (`frontier`) over the ones it has (`visited`).
   The record of kind `proposal` keeps the raw emission beside the question, so what the network actually wrote is
   visible.
2. The ordinary `run_task` cycle: criteria, attempts, judging, teaching, and 2NRL after every step.
3. `_note_urls` files the search results and page links of every tool result into `frontier` (capped at 200) and
   the pages read into `visited`, so the exploration compounds instead of circling.

### 27.5 CLI, API, frontend, tests

CLI: `tools list | describe --tool NAME | call (--tool NAME --arg k=v ... | --call 'name {...}')`,
`agent --tasks FILE` and `explore [--steps N] [--seed-url URL]`; `_add_tool_options` (`--offline`,
`--allow-private`, `--search-url`, `--web-timeout`, `--max-bytes`, `--python-tool`, `--sandbox-timeout`,
`--upload-dir`) and `_add_agent_options` (the LLM roles, the loop, `--blatant-mode|--blatant-margin|--blatant-boost`,
the 2NRL rates) are shared; `serve` takes the tool options too. `TaskPrinter` prints criteria, proposals, calls and
attempts as notes and every finished task as a row.

API (`ModelService.toolbox(**overrides)`, `describe_tools`, `start_agent`, `start_explore`, `agent_history`;
`/api/status` gains `"tools"`):

| method & path | body | response |
|---|---|---|
| GET `/api/tools` | | `{"tools", "names", "count", "options", "upload_dir", "call_format"}` |
| POST `/api/tools/call` | `{tool, arguments}` or `{call}` + tool overrides | the `ToolResult` (a failing tool is 200 with `ok: false`; an unknown tool 404) |
| POST `/api/agent/start` | `{tasks \| tasks_text \| task_files, phase, rounds, ...}` | 202 `{"job", "tasks", "tools", "config"}` |
| POST `/api/agent/explore` | `{steps, seed_urls, ...}` | 202 `{"job", "steps", "seeds", "tools", "config"}` |
| GET `/api/agent/history` | | `{"history": [...]}` |
| POST `/api/agent/criteria` | `{tasks}` | `{"tasks": [{task, prompt, criteria}], "model", "url"}` |
| POST `/api/agent/solve` | `{task, source: model\|teacher}` | `{"criteria", "attempt", "transcript", "correct", "gap", "frontier", "tools"}` (no training) |

Frontend: the **Agent** tab (`AgentPanel.jsx`) — the mode (tasks / explore), the tool list, the loop and mediation
options, the failure fieldset of section 27.3, a live log of criteria, proposals, tool calls (tagged by who wrote
them) and attempts, and a table of finished tasks with `own` calls, `failures`, `blatant` and `boost`.

Tests: `tests/test_tools.py` (a fake website: pages, a JSON and an HTML search endpoint, redirects — including one
to a refused scheme — a byte cap; the format, the registry, the guards, `safe_eval`, every built-in tool) and
`tests/test_agent.py` (a fake Ollama that plays all four roles and switches between native `tool_calls` and JSON,
the fake website, and a *real* untrained network: criteria, both mediation paths, judging, teaching, the gap and
the weighting, the whole loop, exploring, the API endpoints and the CLI).

### 27.6 Tool use in Go (`go/radixnet/tools.go`, `web.go`, `calc.go`, `toolbox.go`, `agent.go`)

The last Python-only module, and the one where the standard library had to be taken at its word: Go has no
`html.parser`, no `ast.literal_eval` and no `eval`, so three pieces are written out rather than translated.

* `tools.go` is the text format and the registry: `TaskHeader`, `CallText`, `ResultText`, `AnswerText`,
  `TranscriptText`, `ParseCall`, `ParseArguments` (JSON, `key=value` pairs, a bare value, and the same
  truncated-JSON repair), `Param` / `Tool` / `ToolCall` / `ToolResult` / `ToolBox`.  One thing had to be written
  by hand for the format to match character for character: `marshalSorted` renders a value the way
  `json.dumps(..., ensure_ascii=False, sort_keys=True)` does — sorted keys, a space after every colon and comma,
  floats as `repr()` writes them — because Go's encoder writes `{"url":"u"}` where Python writes `{"url": "u"}`,
  and the network learns the characters.
* `web.go` is the browsing half: a small HTML tokeniser (raw-text elements taken whole, so a `<` inside a script
  is not a tag), the same block / drop tag sets, the same title-inside-`<head>` special case, and a `WebClient`
  with the same guards — http / https only, no credentials, no private / loopback / link-local / reserved
  address unless `AllowPrivate`, a byte cap, and redirects followed by hand so every hop is checked again.
* `calc.go` is the calculator: a recursive-descent parser over the same grammar, keeping Python's arithmetic
  where the two differ — `/` is true division, `//` floors, `%` takes the sign of the divisor, `int op int` stays
  an int, `round` goes to even, and a result is rendered the way `str()` renders it (`29.0`, not `29`).
  `TestGoToolsParity` runs a list of expressions through both CLIs and compares the answers.
* `toolbox.go` is the built-in set (`web_search`, `web_fetch`, `web_links`, `calculator`, `python`,
  `read_file`) with the same names, parameters and descriptions, so a transcript written on one side is a
  transcript the other can read - and the JSON schemas the LLM is handed are equal, which the parity test
  asserts.
* `agent.go` is section 27.2: the four roles with the same prompts, the same lenient readers (`criterionMark`,
  `marksOf`), `decideAgent`, the mediator's fallbacks, `TeachTask`, `ProposeTask`, and `AgentTrainer` with the
  same records and the same blatant-failure handling (`TwoNRLWeighted`, `InvertPaths`).  The Go clients gained
  `Chat` / `ChatMessage` for it, so Ollama's native tool calling is used where the model has it and JSON mode
  where it does not - duck-typed on the client, exactly as Python's `getattr(client, "chat_message", None)`.
* `server/agent.go` adds `/api/tools`, `/api/tools/call` and the five `/api/agent/*` endpoints; the Go CLI gains
  `tools list | describe | call`, `agent` and `explore`, and `serve` takes the tool options.  The count model
  pushes by a `strength` rather than `neg_lr` / `pos_lr` / `batch_size`, which is the only setting that differs.
* Go tests: `radixnet/tools_test.go` (a fake website, the format, the registry, the guards, the calculator,
  every built-in tool), `radixnet/agent_test.go` (a fake Ollama playing all four roles, a real untrained
  network) and `server/agent_test.go` (the endpoints end to end).

---
### 27.7 The failures feeding the negative network (`blame.faults_from_agent`)

Section 24 applied to tool use: the judge is the tutor, and every attempt it rejects becomes blame. Reasons are
`AGENT_REASONS` (`no-call`, `bad-call`, `tool-error`, `no-answer`, then the judge's own words through `classify`),
picked by `agent_reason(attempt)` in the order of *how far it got*, and the severity of an attempt comes from its
gap (`severity_from_gap`: 0.25 at a near miss, 2.0 for an attempt that answered nothing).

`faults_from_agent(attempts, correction=, threshold=, source=, steps=)` blames a failure at every granularity it
went wrong at, because they are different failures:

* the **transcript**, for the answer. When a correct run of the same task exists — usually the teacher's
  demonstration — it rides along as the `correction`, so `teach` routes the fault through `NegativeNet.correct` and
  only the characters that differ from a run that worked are blamed. The shared task line and the calls that
  worked never become evidence of anything.
* each **emission the mediator had to repair** (`bad-call`, severity 0.75). What the network wrote is the failure,
  and it is *not* in the transcript — which holds the repaired call instead. Blaming the transcript for those
  characters would teach the network that a well-formed call is a mistake.
* each **call the network wrote itself that the tool refused** (`tool-error`): it chose that call and it did not
  work. A call the *mediator* wrote is not the network's fault and is not blamed.

The text used as the correction is left out of the cleared set: `correct` already clears what the two runs share,
and clearing it again would take off the blame the diff has just placed.

`AgentTrainer(negative=...)` calls `blame.teach_agent` after every task and adds `negative_blamed`,
`negative_edges`, `negative_cleared` and `negative_reasons` to the task record; the journal records `agent` or
`explore` as the source. What it learns comes back the other way: `AgentConfig.avoid_blamed` (on by default when a
negative network is attached) puts every candidate the network offers through `NegativeNet.judge` first, and one it
`reject`s is passed over for the next candidate, counted as `AgentTrainer.avoided`. The failures the agent finds
therefore stop it repeating them. `agent --blame` / `explore --blame` (with `--no-avoid` to keep offering them) and
`POST /api/agent/start` / `/api/agent/explore` `{"blame": true}` switch it on; the Agent tab has the checkbox.

---
## 28. Talking to something that answers back (`chat.py`) — the LLM converses, and marks the conversation

Section 22 has the model talk to *itself*.  That is a good way to see what it knows and a useless way to find out
whether it answers anything: neither voice can tell the other that its reply did not follow on.  Every teacher in
this project so far talks *at* the network — the English tutor writes a prefix and marks the completion (§21), the
critic reviews texts written alone (§24.6), the agent judges a transcript of tool calls (§27) — and none of them
holds up the other end of a conversation.

`Chat(model, client, config, negative=..., external=..., judge_client=...)` is that missing party.  One
conversation is:

1. the **partner** (`ollama.chat_line`) says a line.  Its system prompt asks for short, plain lines that end on
   words which are easy to carry on from, because that is the only thing a character-level model can reply to —
   the prompt is doing the model a favour, not flattering it.  `topic`, `persona` and a given `opening` steer it;
2. the **model replies** through `dialogue.reply` — extracted from `converse`'s loop for exactly this, so there is
   one implementation of "what does this model say next" and the other voice need not be a model at all.  The tail
   of the partner's line is located in the graph and continued, the context loses a word at a time while nothing
   follows it, and a voice with nothing left to add changes the subject from START.  With a negative network in
   hand `config.guard` puts the pair (§24.3) in the way as a `veto`, so a reply the network would be blamed for is
   never spoken;
3. they alternate for `turns` exchanges, the model always having the last word (there is no point asking the
   partner for a line nothing will answer);
4. the **judge** (`ollama.review_conversation`) marks every reply out of 10 **against the line it answered** and
   gives the conversation as a whole a verdict of its own.  One call per conversation, not one per line: the
   exchanges are numbered `[i] Partner: … / Model: …` and parsed by the same lenient `_parse_reviews` the critic
   uses, so the result is the `summarise_reviews` shape and `blame.teach_reviews` takes it unchanged.

Then both networks learn.  The failures blame the negative network and the passes clear it (§24.2, source
`chat`); and — unlike the critic, which only ever reads the positive model — this loop **trains it**:
`Chat._learn` runs 2NRL with the failed replies as garbage and the passed ones as correct, and with
`teach_partner` the partner's own lines join the positive phase, because in that conversation, at that moment,
they are exactly what a good reply would have looked like.  It takes the model lock like any other training job.

The **duplicates** are the one judgement the loop makes itself.  The model is answering an LLM, not itself, so the
same `Heard` (§22) decides what a reply would repeat — the conversation remembers both sides of it — and the
search skips those candidates, along with the replies that repeat their own words (`avoid_word_repeats`, the
Chat tab's "Avoid repeated words") - and a reply that catches itself repeating backs up and explores its way out
first (`explore`, §22), which the exchange records say.  What is left is `held["repeats"]`: the replies the model could only repeat.  They
are punished with the failures *whatever the judge made of them*, and taken out of the positive phase if it
passed them, because a duplicate said nothing new and rewarding it would only make it likelier next time.  The
`conversation` record counts them in `repeats`.

Lock discipline is §24.6's, applied twice per exchange: **the model replies under the lock, every LLM call happens
outside it**.  `_external` wraps the opening, each partner line and the judgement; `dialogue.reply` is a walk of
the graph and is made with the lock held.

Records are `exchange` (streamed as each reply is spoken: the line, the reply, the context it picked up, whether
it was a repeat, whether the guard vetoed anything), `conversation` (the transcript, every review, the marks, the
overall verdict, how many replies were duplicates, what was
blamed and what was learned — always carrying the learning keys so the shape is stable whether or not it learned)
and one `report` per run.  A conversation that ends early says which way it ended: the model had nothing left to
say, the guard vetoed everything it could say, or the partner went quiet.

CLI: `chat` (`--conversations 0` runs until Ctrl-C, which finishes the conversation in progress and saves).
API: `POST /api/chat/start` starts a `chat` job over `ModelService.start_chat`, `GET /api/chat/history` is its
record.  Frontend: the **Chat** tab — the settings, a live transcript that streams as the two of them talk (each
reply badged with its mark once the conversation has been judged), the table of conversations and the report card.
The transcript reads **newest first**: a new exchange is appended to the top and pushes the older ones down, so a
running conversation never has to be scrolled to, and a line above it says how many replies were duplicates
punished with the failures.
Tests: `tests/test_chat.py` — a fake partner and judge, a real trained model: the conversation, a given opening,
the reply really continuing the line, the marking, what reaches the negative network and the positive one, a
reply it could only repeat punished however well it was marked, the
guard silencing a reply, stalling, the report card, stopping, the endpoint and the CLI.

### 28.1 The conversation in Go (`go/radixnet/chat.go`, `review.go`, `dialogue.go`)

The Go half is the same four steps, from the same prompts, and the parity test holds them to it call by call.

* `dialogue.go` gained `Model.Reply(previous, ReplyOptions)` — the same extraction the Python side made, for the
  same reason: `Converse` is now a loop over it, and the chat loop calls it for the model's own lines while an LLM
  speaks the others.  One implementation of "what does this model say next", whoever is on the other side.
* `review.go` gained the two prompts character for character (`chatSystem`, `conversationSystem`), `ChatLine`
  with its `firstLine` reader (an answer that wrote several lines, or quoted itself, or prefixed a speaker name),
  and `ReviewConversation` — the same `[i] Partner: … / Model: …` numbering, the same lenient `parseReviews`, and
  `parseOverall` for the verdict on the conversation as a whole.  It returns the `SummariseReviews` shape plus
  `Overall`, so `TeachReviews` takes it unchanged, exactly as in Python.
* `chat.go` is `ChatConfig` / `Chat` / `ChatReportCard`: `Pair` (the guard, standing aside when there is no
  negative network or nothing has ever been blamed), the `chatVeto` that remembers its judgements so a candidate
  offered again after the context is shortened is not judged twice, `Converse`, `RunConversation`, `Learn` and
  `Run`.  The records carry the same keys in the same shape, the learning keys included whether or not it
  learned.  As everywhere in the port, the count model pushes by a `strength` rather than `neg_lr` / `pos_lr` /
  `batch_size`, which is the only setting that differs.
* `server/chat.go` adds `POST /api/chat/start` and `GET /api/chat/history` with the same request fields and the
  same 400s, releasing the model lock around the partner's and the judge's thinking; the Go CLI gains `chat` with
  the same flags, printing the transcript as it is spoken and saving only when it learned.
* Go tests: `radixnet/chat_test.go` (a fake playing the partner and the judge, a real trained model — the
  conversation, a given opening, the reply really continuing the line, the marking, the blame, the clearing, the
  guard, stalling, learning, the report card, stopping) and `server/chat_test.go` (the endpoints end to end).
  `TestGoChatParity` runs both CLIs against **one** fake partner and asserts the same prompts, the same
  transcripts, the same marks and the same two networks on disk afterwards.

---
## 24. Counter overflow (`counter.py`, `go/radixnet/counter.go`) — cyclic counters with a reset count

Every number in the model that only ever counts up - traversals, node and edge
visit counts, the negative network's per-edge and per-reason `fails`, epochs,
trained characters and texts, 2NRL runs, feedback passes, failures, clearings,
judgements, rejections, the per-source tally, the version stamps - would
eventually leave the range of whatever holds it:
`int64` in the Go port, and long before that the 53-bit mantissa of the JSON
number that carries it through a model file, the HTTP API and the frontend. So
**no counter in the model is an unbounded integer.** Each one is a two-digit
odometer in base `COUNTER_LIMIT`:

```
total = resets * COUNTER_LIMIT + value          with 0 <= value < COUNTER_LIMIT
```

The value counts up as before; the moment it reaches the limit it is **set back
to 0** and `resets` - how often that happened - goes up by one. Nothing is lost:
`total` is the exact number of events, `value` is what the odometer reads now,
`resets` is how often it went round. Cycles are a feature here too.

`COUNTER_LIMIT = 10 ** 15` because it is exactly representable as a `float64`
(`< 2 ** 53`), so every counter survives a model file, a JSON response and a
JavaScript `Number` unchanged; because it leaves four orders of magnitude of
head room under `int64`, so a whole epoch of increments can land on a counter
before the next carry without overflowing the Go port; and because it is a
round decimal - one quadrillion events per turn. `resets` wraps at the same
limit, so the pair itself cycles after `10 ** 30` events.

```python
COUNTER_LIMIT = 1_000_000_000_000_000

def carry(value: int, resets: int = 0) -> tuple[int, int]   # whole turns move from value into resets
def total(value: int, resets: int = 0) -> int               # the exact number of events
def as_float(value: int, resets: int = 0) -> float          # float(resets) * COUNTER_LIMIT + float(value)
def carry_series(values: list[int], resets: dict[int, int]) -> int   # wrap a parallel counter list in place

class CyclicCounter:                 # one scalar odometer; immutable, so `counter += 1` rebinds
    value: int ; resets: int         # compares and hashes as `total`, so version stamps stay exact
    @property total ; def as_float() ; def bumped(n=1) ; def to_pair() ; @classmethod from_pair(value, resets=0)
```

Go mirrors it exactly (`CounterLimit`, `Carry`, `Counter{Value, Resets}` - a
comparable struct, so a cached version stamp is still checked with `==` -,
`CarrySeries`, `counterTotal`). Both compute the float total as
`float(resets) * LIMIT + float(value)`, in that order, so the weights stay
identical to the bit across the two implementations.

**Wrapping never happens in a counting loop.** Hot loops add to a plain integer
(the Go port from many goroutines, with plain or atomic increments); a *carry
sweep* at a safe point moves whatever crossed the limit into the resets:

* `RadixCyclicGraph.carry_counters(force=False)` / `(*Graph).CarryCounters(force)` —
  the only place the visit counters wrap. Called at the end of every epoch and
  before every save. `graph.traversals` counts every increment made and so
  bounds each individual counter: while it has not wrapped, the sweep returns
  after one comparison. `force` sweeps anyway (used when loading a file).
* `meta_add(meta, key, delta)` (`(*Model).metaAddInt`) — the lifetime counters
  of `model.meta` listed in `META_COUNTERS`, each wrapping into `<key>_resets`;
  `meta_add_keyed(meta, key, name, delta)` does the same for the `{name: count}`
  maps of `META_COUNTER_MAPS` (`sources`), whose resets mirror it in
  `<key>_resets`. `carry_meta(meta)` / `(*Model).carryMeta` wrap them all when a
  file is loaded, in case it carries one that was never wrapped.
* `CyclicCounter` / `Counter` scalars (`total_traversals`, `traversals`,
  `version`, `structure_version`) wrap as they are bumped; they are cheap.

`NegativeGraph.carry_counters` extends the sweep to the negative network's
`edge_fails` and `reason_fails`, bounded the same way by its `total_fails`.

Per node / edge the reset counts live in a **sparse** map (`count_resets`,
`edge_count_resets`, `edge_fails_resets`, `reason_fails_resets`), so the usual
case - nothing ever came near the limit -
costs one comparison per entry and stores nothing. `split` copies a node's
resets to the new node and the new internal edge; `merge_child` keeps the
larger of the two *exact* counts; every ranking by visits (`_best_trigram`,
`_best_node_with_prefix`, the graph view) compares exact counts, never the bare
readings.

What crosses a boundary:

| Where | Fields |
|---|---|
| model file (`format_version` 2) | `version` / `structure_version` / `traversals` and their `_resets`; `nodes.count_resets` and `edges.count_resets` (dense arrays, written only once something has wrapped); `weights.total_traversals_resets`; `meta.<counter>_resets` and `meta.sources_resets`; on a negative graph also `edges.fails_resets`, `weights.total_fails_resets` and `weights.reasons.fails_resets` |
| `GET /api/status`, `/api/model` | `epochs_total_resets`, `trained_chars_resets`, `trained_texts_resets`, `twonrl_runs_resets`, `feedback_passes_resets`, `total_traversals_resets`; on the negative network `failures_total_resets`, `cleared_total_resets`, `judgements_resets`, `rejected_resets`, `sources_resets`, `edge_fails_total` / `edge_fails_total_resets` and `fails_resets` per reason row |
| `GET /api/graph` | `count_resets` per node and per edge, `total_traversals_resets` |
| frontend | `fmtCounter(value, resets)` renders `1,234 (+2 resets)` in the status bar, the Graph tab's tooltips and the Negative tab's counts; `counterTotal(value, resets)` is what node radius / edge opacity scale on |

A format 1 file (written before any of this) loads unchanged: its counts are
plain integers, are carried on the way in, and `traversals` is seeded with their
sum so the guard above stays correct. The float accumulators `rewards_total` /
`penalties_total` are sums of real numbers, not counters, and are left alone.

Tests: `tests/test_counter.py` (the odometer itself; counters wrapping mid-training
without losing history; weights, shares and rankings identical to what unbounded
integers would give; the save / load round trip; a format 1 file; the guard;
the lifetime counters), `go/radixnet/counter_test.go` (the same in Go) and
`tests/test_go_parity.py::test_a_wrapped_model_file_crosses_over_unchanged`
(a wrapped file trained on by both sides, with identical counters, weights and
predictions).

## 28. A real browser (`browser.py`) — Chrome over the WebDriver protocol

`urllib` reads a document; it cannot read a page that draws itself. `BrowserClient` runs the page in **Chrome** and
hands back the DOM once its scripts have run, so `web_fetch` / `web_links` / `web_search` see what a person sees.

WebDriver is HTTP with JSON bodies, so nothing is imported: the client starts `chromedriver` itself and talks to it
with the standard library, the way `OllamaClient` talks to Ollama. `selenium` speaks the same protocol to the same
binary and works too, but is not required and is not used.

```python
DEFAULT_PAGE_TIMEOUT = 30.0 ; DEFAULT_WINDOW = (1280, 900) ; SETTLE_SECONDS = 0.25
class BrowserError(Exception)
chromedriver_path() / chrome_path()   # $RADIXNET_CHROMEDRIVER / $RADIXNET_CHROME, then PATH, then a
                                      # Playwright browser directory (.../chromium-1194/chrome-linux/chrome)
describe() -> {"available", "chromedriver", "chrome", "chromedriver_version", "chrome_version", "headless", "error"}
browser_available() -> bool
class BrowserClient(page_timeout=30, window=(1280, 900), settle=0.25, headless=True, driver=None, chrome=None,
                    endpoint=None)
    .get(url) -> {"url", "title", "html", "status"}   # navigate, wait for readyState, then the outerHTML
    .script(source, *args) ; .close() ; .describe() ; context manager ; .running ; .pages
```

chromedriver refuses to drive a Chrome of a different major version, which is far and away the most common way
this fails, so `describe()` reads both versions and says so itself rather than leaving a raw WebDriver error
nobody can act on. `endpoint` (or `$RADIXNET_WEBDRIVER`) points at a WebDriver that is already running — a Selenium
Grid, a `selenium/standalone-chrome` container, a chromedriver of your own — and then nothing is started or stopped
here and the remote browser's own binary is used. Headless by default, `--no-sandbox` added when running as root
(containers), images off, one session opened on the first page and reused, stopped by `close()` and at exit.

`WebClient(browser=...)` routes `fetch` through it (`default_toolbox(..., browser=...)`, `--browser`,
`{"browser": true}` on `/api/tools/call`, `serve --browser`). The address guards run **first** either way: a
browser executes whatever a page sends it, so this is isolation from nothing — untrusted browsing belongs in the
Docker image. `radixnet tools browser` reports what would be driven without starting it; the API's `GET /api/tools`
carries the same under `"browser"`, and the server keeps one Chrome for every request (`ModelService.browser`),
closed with the server. Tests (`tests/test_browser.py`) drive a **fake WebDriver** over HTTP so the whole protocol
runs without a browser, and skip the one real-Chrome test when the versions do not match.

---

## 29. The Model Context Protocol (`mcp.py`) — the tools and the network, to any client

`radixnet mcp` speaks MCP on stdin / stdout, so any MCP client (Claude Desktop, an editor, another agent) can use
this instance. Two things are offered, and the second is the point:

* the **tools** of section 27.1 — the same registry the network calls by writing text;
* the **network** (`model_tools`) — `radixnet_predict`, `radixnet_generate`, `radixnet_score`, `radixnet_stats`,
  `radixnet_judge` (the negative network's verdict on a text: has this way of going wrong been seen before, and
  why) and `radixnet_solve` (one task through the whole agent loop — criteria, tool calls, a judged answer). A
  client can ask *this* network what it thinks, not just borrow its browser.

MCP is JSON-RPC 2.0 over a stream, so this is the standard library: newline-delimited JSON, `initialize`,
`tools/list`, `tools/call`, `ping` and the `notifications/*` that take no answer. `McpServer.handle(message)` turns
one request object into one response object (`None` for a notification), which is how it is tested; `run(in, out)`
is the stream loop. A tool that *fails* is a result with `isError: true`, not a protocol error — the client shows
it to its model — while an unknown tool or unreadable arguments are `INVALID_PARAMS`. Nothing is ever printed on
stdout but the protocol; the log goes to stderr.

```
radixnet --model model.json mcp                       # tools + the network
radixnet mcp --no-model --offline --python-tool       # the sandbox alone
radixnet mcp --browser                                # browsing in a real Chrome
```

```json
{"mcpServers": {"radixnet": {"command": "python", "args": ["-m", "radixnet", "--model", "model.json", "mcp"]}}}
```

## 30. The resonant model (`resonance.py`, `metacog.py`, `phasesearch.py`) — an analog phase, and cycles that hand off

A fourth kind on the same self-compressing graph, selectable at run time
(`--kind resonant`, `POST /api/model/select {kind: "resonant"}`, the header selector).  It takes up the two research
notes the other kinds leave on the table.

`Research/SineWaveActivationFunction.md` says a brain is an *analog computer*, so sine waves are how information is
encoded.  `RadixNet` reads that as a **pointwise** sine — every node passes its state through `-sin(z/3)`.  This model
reads the other half: a sine has a **phase**, phases **add** along a path, and signals that meet in phase reinforce
while signals that meet in antiphase cancel.  `Research/CyclesAreAFeature.md` says cycles are a feature and that *when
we encounter a cycle we use metacognition or another part of the brain instead*.  The phase is what makes that
actionable, and `metacog.py` is that other part.

### 30.1 The phase — a ring counter the text itself turns

A walk carries one number more than the node it stands on: its phase, one of `buckets` positions on a ring (so the
model is a finite automaton over `(node, phase)`, not an approximation of a continuous system).  Every **trigram**
advances it by a fixed integer:

```
advance(t) = round(buckets / period + kick_scale * trigram_phase(t) * buckets / TAU)   (mod buckets)
```

* `buckets / period` is a **clock**: with the default `period == buckets` one character is one bucket, so the phase
  says where in the rhythm the walk is — how far into a line, a word, an indent.
* `trigram_phase(t)` is a **kick**: a stable BLAKE2b hash of the trigram, in `[0, TAU)` (not `hash()`, which is
  randomised per process and would make a saved model read differently after a restart).  `kick_scale = 0` (the
  default) leaves a pure position clock — dense, quickly learned; turning it up makes the phase a rolling signature
  of the whole path — real long-range context on a three-character graph — at the price of far sparser statistics
  per phase.

Defining the advance **per trigram and summing** is what makes the phase survive the radix operations:
`ResonantGraph.advance[n] = label_advance(labels[n])` is the sum over the trigrams a node's label covers, a split and
a merge move trigrams between labels but never change which trigrams exist, so `split` and `merge_child` leave the
total across the path exactly where it was (`node_advance` forces 0 on every id below `FIRST`, whose labels `"<s>"`, `"</s>"` and
`"<back>"` are long enough to look like ordinary ones).  It also makes the phase a function of the emitted text alone —
`text_bucket(text)` — so a prediction's starting phase is read straight off the prefix with no walk, and it is
exactly the phase a walk that emitted that prefix would carry.

### 30.2 The edges — a circular mean, and coherence for free

An edge does not learn a phase *offset*; it learns the phases at which it was actually taken.  Each traversal at
bucket `b` adds the unit vector of that bucket to the edge's accumulator `(edge_cx, edge_cy)` with weight `edge_cw`:

```
mu        = atan2(cy, cx)                        # the mean phase at which this transition fires
coherence = hypot(cx, cy) / (cw + concentration) # how consistently, in [0, 1]
score(p -> c | psi) = amp_scale * log(share of the parent) + reward_scale * reward
                    + resonance_scale * coherence * cos(psi - mu)
```

`share` is the edge's smoothed share of its parent's traversals (`(count + 0.5) / (parent total + 0.5 * degree)`), so
with `resonance_scale = 0` the model *is* a clean smoothed frequency model and the phase adds to it.  `concentration`
(default 2) shrinks the resultant length, so a single traversal — whose resultant length is exactly 1 — reads as an
opinion, not a certainty.  `coherence` is free confidence: it measures how **context-dependent** a transition is,
with nothing added to the model to measure it.  An incoherent edge falls back to plain frequency; a coherent one is
cheap in phase and dear out of phase.  Children are drawn by a softmax over those scores, so
`P(child | parent, phase)` — the first distribution here that depends on more than the parent.

Every node is created with `a = 0, k = 1`, so its activation is the constant 1 and the base class's phase-free
scores, probabilities, costs, splits and merges all keep working; they read as the model's **phase-marginal** — what
it believes before the phase is taken into account.

Learning is counting: a traversal, a unit vector, a reward.  No gradient, no learning rate, no ordering effect
(`TrainConfig.lr` / `act_lr` / `batch_size` are accepted and ignored; feedback magnitude is `strength`).
`observe_back` (section 24's `BACK` sentinel) is overridden for the same reason the count model overrides it: a
weight here is a *function* of the counts, the rewards and the phase, so nothing may be written to `edge_w` by hand
— the next `recompute_weights` would erase it.  Going round is counted on the `BACK` edge and what to do instead is
a penalty on the step that looped and a reward on the step taken after backing up.  The hand-over is counted
**without a phase** (`record_traversal(e, None, ...)`, which grows the mean's weight but not its vector, so the
coherence falls towards 0): a voice that backed out of a repeat walked outside this model's search and cannot say
which phase it was in, so `BACK` competes on its share rather than pretending to a phase it never learned.
`invert()` negates the accumulators, which rotates every `mu` by `pi`: what resonated now cancels — 2NRL's inversion
in one line and with an exact meaning.  `reward` counts a traversal, adds `+strength` and *sharpens* the phase lock;
`punish` subtracts and *decoheres* it (taking away the context in which the path was right, this model's own way of
forgetting); `invert_paths` rotates a failed path's edges into antiphase (`activation`) or decoheres them (`state`).
`reward(weights=)` / `punish(weights=)` / `two_nrl(bad_weights=, good_weights=)` scale all three of those per text
(D-050): a text of weight `w` moves `w * strength` of reward and `w` of the sharpening, so how the phases lock
follows the mark.

### 30.3 Cycles — the handoff

Coming back to a node at a **new** phase is progress: the signal has moved on.  Coming back at the **same** phase is a
true loop that would repeat for ever.  So the phase is exactly what separates productive recurrence from being stuck,
and only the second kind is a cycle worth a decision.

`metacog.MetaLayer` holds, per **cycle signature** — the re-entered node's first trigram and how many steps the loop
closes over, clamped at 8, e.g. `"lol:4"` — a score for each of three actions: `ride` (go round again: right for
`"aaa"`, `"lol lol lol"`, `"----"`, indentation), `escape` (take the cheapest child that does not close the loop) and
`abort` (walk to END).  The scores are learned the way everything else here is: while training walks a real text,
every step where a child *would* close a phase-locked cycle is a decision the text made, and whatever it did scores
`+1`.  The policy is a softmax over `sign(s) * log1p(|s| / smoothing)` — evidence, not logits — so with the default
`smoothing = 1` it is exactly add-one smoothing over the counts: ten observations are confident, a hundred more so,
one is barely an opinion, and an untrained layer is uniform (costing `log 3` whatever the walk does, which changes no
ranking).  Unseen signatures fall back to a prior summed over every cycle the model has met — **rescaled to `prior_weight`
observations** (default 2).  Left raw that prior judges a cycle the model has *never* met with the confidence of all
of them at once: after three epochs of `data/sample_corpus.txt` it stands at `ride 78, escape 33`, so an unseen
cycle is decided as if by a hundred observations, and anything else with an opinion is drowned out.  Rescaling keeps
its *direction* — what cycles tend to look like — and gives it the weight it has actually earned about this one.
`invert()` negates every score, so 2NRL flips the layer with the graph.

### 30.3.1 `BACK` and the layer — the same knowledge at two resolutions

Section 24's `BACK` sentinel and this layer are both about going round, at different grains.  `BACK` is the
**reflex**: an edge `p -> BACK` competing for `p`'s probability, taught by voices that caught themselves repeating
and backed out, and when it is the cheapest child `search.onward` hands the branch over — it offers nothing.  The
layer is the **memory** of one particular cycle.  They meet in two places, and a third that turned out to be a
mistake.

**`BACK` primes the layer.**  `_meta_costs` passes the re-entered node's hand-over probability (`_back_probability`,
the `BACK` edge's share at that phase) into `MetaLayer.log_policy(signature, back)`, where it enters as
`back_scale * back` observations **against** riding — the same evidence scale as everything else, so the ordering
falls out on its own:

| what is known | decision |
|---|---|
| nothing | `ride` (weakly: the prior's direction, `prior_weight` of confidence) |
| 2 hand-overs at the node (`P(BACK) = 0.11`) | `ride` — a weak reflex is not enough |
| 6 hand-overs (`P(BACK) = 0.95`) | `escape` — the reflex speaks and wins |
| …plus **1** observed ride at this exact cycle | `escape` — one observation does not override it |
| …plus **3** observed rides | `ride` — the specific memory outranks the node's reflex |

**The layer overrules the reflex.**  When `onward` has handed a branch over, `_expand` asks `MetaLayer.rides(label)`
— what the layer remembers about cycles at *this node*, summed over every loop length it has seen there — and
restores the real children when the answer is that it rode them.  It has to be the node-level question rather than a
signature: a walk meeting the hand-over is on its **first** visit and has not closed a cycle yet, so there is no
signature to look up.  With no memory the hand-over stands.  Metacognition supervising the reflex is what the
research note describes, and this is the line where it happens.

**What does not work: teaching `BACK` from the corpus** (`teach_back`, default **off**).  A corpus declining a cycle
looks like the same lesson a voice reports by backing out, so `_teach_cycle` can pass it on.  It should not, by
default, and the measurement says why.  `observe_back` is one-directional by design — nothing calls it except
someone who already backed out — so an estimate fed from it alone only rises; over four epochs of the sample corpus
`"he "` reached `P(BACK) = 0.79` and began vetoing, which took `"the sun"` out of the model's reach and left
generation at `["lond", "ever", "water"]`.  Counting the rides too (`observe_onward`, the missing opposite) barely
helped, because at those nodes the corpus *always* declines.  The real mismatch is one of grain: the corpus declined
**one** child, and `BACK`'s veto drops **all seventeen** — including the escapes it actually took.  A ceiling below
which observation may push `BACK` (`back_ceiling`) keeps it under the veto and generation survives, but it still
costs corpus score for a speculative gain, so it is a dial rather than a default:

| `back_ceiling` | nodes vetoing | generation |
|---|---|---|
| off | 0 | `the sun`, `the field`, `the park` |
| 0.10 | 0 | `the sun`, `the park`, `the earth` |
| 0.15 | 1 | collapses at `back_strength` 0.25 |
| 0.25 | 2–3 | `lond`, `ever`, `water` |

`observe_onward` and `back_probability` stay: the first is a primitive `observe_back` was missing, and the second is
what primes the layer.

### 30.4 The search (`phasesearch.py`)

A state is `(node, chars_emitted, phase_bucket)`.  The cost of an edge now depends on the whole path that led to it,
and the search is still exact: the product graph is finite (`buckets` times bigger), so optimal substructure holds.
`kick_scale = 0` makes the phase a function of the emitted length alone and the product collapses back to the
ordinary unrolled graph at no extra cost.

* `phase_kbest` — **the default**: Dijkstra with `k` labels per state instead of one.  One label per state is what
  makes Dijkstra a shortest path and also what makes it blind — a single label cannot say *which* walk reached the
  state, so the layer has nothing to look at.  Letting a state be settled up to `k` times fixes both at once: the
  `k` goals pop in cost order and are the `k` cheapest walks *exactly* (the standard k-shortest-walks argument;
  loops are allowed, and a walk that goes round again is simply one of the candidates), and every label reads back
  to its own path, so the cycle it is standing in is visible and the layer can price it — **in the exact search,
  not only in the beam**.  `k = 1` is `phase_dijkstra` to the expansion.  Cost grows with `k`, not with the width
  of a frontier, because the search still stops at the `k`-th finished walk: on `data/sample_corpus.txt`, `k = 5`
  from `"the "` expands 159 states where the beam of the same `k` expands 1312, for an answer the beam can only
  approximate.  A trained layer prices a move by what is on the path, which no longer decomposes over states, so
  exactness then holds only up to the `k`-labels-per-state bound — but that bound is per state rather than per
  frontier, so it degrades where the graph *branches* instead of wherever the cheapest region happens to be.
* `phase_dijkstra` — the cheapest path, same contract as `search.dijkstra_predict` (goal, cap, fallback,
  `include_context`), `search.onward` included, so a node the model has learned to go round (its `BACK` edge
  cheapest, section 24's sentinel) offers nothing and the search goes on with its others.  **No metacognition**: one
  label per state cannot depend on which path reached it.  Kept for exactly that guarantee.
* `phase_beam` — the `k` cheapest and `k` dearest complete paths, same two-beam shape as `beam.beam_predict`.  Kept
  because it is the only mode that can answer the **bottom** half: in a cyclic graph the worst walk is unboundedly
  bad (loop once more and it is worse), so "least likely" needs a frontier's bound rather than a goal count, and a
  k-best search cannot give it.  Every entry carries the `(node, phase)` states already on its path, so a cycle is
  detected *per path*: the tightest loop names the signature, the child that closes it takes the layer's `ride`
  cost, END takes `abort` and every other child takes `escape`.  Nothing is forbidden — a cycle the corpus rides
  stays cheap to ride.
* `phase_walk` — one stochastic walk, layer included.

`ResonantNet.predict(prefix, mode="kbest" | "dijkstra" | "beam" | "sample", k, beam, ...)` is those four; `mode`
defaults to `"kbest"`, which is both exact and where the layer runs.  `_expand(graph, meta, node, phase, depth,
step_penalty)` is shared by every path-carrying search: `depth` maps each `(node, phase)` on the path to where it
first appeared, which is all the layer needs — a child landing on one closes a phase-locked cycle, and the
difference is the loop's length.  `score` walks the text `START -> ... -> END` carrying the phase, charges
`log(UNKNOWN_PROB)` for an unknown trigram or a missing edge exactly as `GraphModel.score` does (so an unseen text
scores *worse* than a trained one instead of looking deterministic) and never changes the model.
`generate` defaults to `"kbest"` too: generation asks for the `count` most likely *complete* texts, which is
precisely what a k-best search answers exactly and at a fraction of a beam's cost; the other modes are
`GraphModel.generate`'s.

### 30.5 What it changes, measured

Train `["the cat sat down", "a big cat ran away"]` and look at the node `"at "`, which three different contexts reach.
Phase-free its three children are exactly `1/3` each — the model cannot tell them apart.  Per phase they are not:
`"t sat"` is `0.807` at bucket 5, `"t down"` `0.807` at bucket 3, `"t ran aw"` `0.691` at bucket 0.  End to end the
model continues `"the cat "` with `"sat down"` and `"a big cat "` with `"ran away"`; the identical model with
`resonance_scale = 0` answers `"ran away"` to both (`tests/test_resonance.py`).

### 30.6 Wiring

`ResonantGraph.to_dict` adds `edges.cx / cy / cw / reward` in the same edge order as the base, and `weights` (the
seven settings plus `total_traversals`); `from_dict` rebuilds `advance` from the labels.  `ResonantNet.to_dict` adds
`metacog`.  `format` is `"radixnet-resonant"`, so `load_model`, `model_from_dict`, the checkpointer and `POST
/api/load` restore it like any other kind.  CLI: `--kind resonant` (default file `model.resonant.json`), `weights`
takes `--buckets / --period / --kick-scale / --resonance-scale / --amp-scale / --reward-scale / --concentration` and
**rejects** another kind's options by name instead of ignoring them, `info` shows the phase, the coherence and the
layer, `2nrl` describes the three phases.  API: `/api/model/weights` and `/api/reset` take the same settings (the
model rejects the ones it does not know), `/api/graph` reports `coherence` and `mu` per edge, `advance` per node and
`buckets` on the view, `/api/status` carries `coherence_mean`, `cycles_seen` and `meta`.  Frontend: the header
selector picks it up from `status.kinds` on its own; `util.countingKind` hides the learning-rate and schedule fields
and shows the `strength` field for every kind that learns by counting (only the count model aliases `dijkstra` to the
beam — on this model `dijkstra` is a real, exact mode), the status bar adds coherence / phases and cycles /
signatures, and the Evolve tab describes the failure modes in this model's terms.

Tests: `tests/test_resonance.py` — the phase (stability, sentinels, invariance under split and merge, clock vs kick,
`text_bucket`), the graph (unit activations, coherence growth and shrinkage, resonance in and out of phase,
`invert` / `rotate` / `sharpen`, `configure` rephasing), the layer (uniform when untrained, add-one policy, punish,
invert, prior fallback, round trip), the net (records, falling loss, the three prediction modes, generation, scoring,
cycle learning, the layer moving the beam while Dijkstra ignores it, reward / punish / decohere, 2NRL with weights,
`invert_paths`, stats, argument validation), persistence (file, gzip, wrong kind, checkpoints), the kind registry,
the evolver and `converse`, the CLI and the HTTP API.

---

## 31. The traversal (`penalty.py`, `go/radixnet/penalty.go`) — follow the rewards, or avoid the punishments

Every search in the package reads the graph through one funnel: `[(child, edge, cost)]` for a node, with
`cost = -log P(child | parent)`.  `search.dijkstra_predict`, `search.sample_walk`, `beam.beam_predict` and the four
phase searches of `phasesearch.py` all call `graph.child_costs(p[, prev])` (or `child_costs_at(p, bucket)`) and
nothing else.  **Which cost function fills that list is the traversal**, and it is an option:

| traversal | what the search looks for |
|---|---|
| `"reward"` (the default) | what the model believes.  The count / reward model's weight carries `reward_scale * reward`, so a path the tutor rewarded is cheap and the search follows the rewards.  Every release before this option behaved this way, and `traversal_costs` returns `None` for it — the searches call `graph.child_costs` exactly as they always did, at no cost at all. |
| `"punishment"` | what the model was punished for.  The rewards leave the score altogether and only the penalties price the step, so the cheapest path is the one that accumulated the **least punishment**. |

The traversal is *what* a search looks for; the mode (`dijkstra` / `kbest` / `beam` / `sample`) is *how* it looks.
They are independent by construction: the option replaces the funnel instead of adding a fifth mode, so every mode
of every kind gains it at once and no search code changes.

### 31.1 The split: `child_evidence`

```python
class RadixCyclicGraph:
    def child_evidence(self, p: int, prev: int | None = None) -> list[tuple[int, int, float, float]]
        # [(child, edge, merit, penalty)] over p's out-edges.
        #   merit    - what speaks FOR the step, with every reward taken out of it
        #   penalty  - what speaks AGAINST it, >= 0
        # prev is the node the walk arrived from, which a model that counts paths prices the step by.
```

| graph | merit | penalty |
|---|---|---|
| `RadixCyclicGraph` (the sine model) | `max(0, w * f_p * f_c)` | `max(0, -(w * f_p * f_c))` |
| `CountRewardGraph` | `edge_w[e] - reward_scale * reward` (the dual frequency function, rewards out) | `reward_scale * max(0, -reward)` |
| `ResonantGraph` | `edge_w[e] - reward_scale * reward`; `child_evidence_at(p, bucket)` adds the resonance | `reward_scale * max(0, -reward)` |
| `NegativeGraph` | `log1p(edge_clear[e])` | `log1p(evidence(e))` |

The sine model keeps no separate ledger of its punishments — 2NRL trains a failure in and then inverts it, so what
a punishment leaves behind *is* a negative score on that path's edges — which is why its penalty is read straight
off the score, and why at the default scales its two traversals coincide and part company as soon as
`penalty_scale` is raised.  A judged path context (`CountRewardGraph.path_term`) splits the same way: the part of
the term that says the step was right here is merit, the part that says it was wrong here is penalty.  On the
negative network the option reverses the network's purpose, which is the point: its ordinary traversal predicts the
likeliest way a prefix goes wrong, the punishment traversal walks the least blamed way through the same structure.

### 31.2 The cost function

```python
TRAVERSALS = ("reward", "punishment")
DEFAULT_TRAVERSAL = "reward"

def resolve_traversal(traversal: str | None) -> str          # None / "" -> the default; ValueError otherwise

class PenaltyCosts:                                          # a drop-in for graph.child_costs
    def __init__(self, graph, penalty_scale: float = 1.0, merit_scale: float = 1.0)
    def __call__(self, p: int, prev: int | None = None) -> list[tuple[int, int, float]]

class PhasePenaltyCosts:                                     # a drop-in for graph.child_costs_at
    def __call__(self, p: int, bucket: int) -> list[tuple[int, int, float]]

def traversal_costs(graph, traversal, penalty_scale=1.0, merit_scale=1.0)        # None for "reward"
def phase_traversal_costs(graph, traversal, penalty_scale=1.0, merit_scale=1.0)  # None for "reward"
```

`score(child) = merit_scale * merit - penalty_scale * penalty`, and the cost is `-log softmax(score)` over the
parent's children.  That keeps every invariant the searches rely on: costs are `>= 0` (Dijkstra stays a true
shortest path, `step_penalty >= 0` still holds), `exp(-cost)` is still a path's probability, the children still sum
to 1, and `onward()` can still compare a `BACK` edge against the real children.  The result is cached per
`(parent, prev)` until `graph.version` changes, exactly like the graph's own cost cache; the Go port instead prices
every edge once into a table before the search starts, because its two beams run in parallel goroutines.

`merit_scale = 0` is the pure form: nothing but the punishment decides, and among equally unpunished children the
walk is indifferent.  Both scales must be `>= 0`.

### 31.3 Where it is reachable

* Python: `predict(..., traversal=, penalty_scale=, merit_scale=)` and `generate(...)` on all four kinds, and the
  same three on `NegativeFilter.predict` / `.generate`.  A `Prediction` carries `traversal`, so a result says
  which one ran.
* CLI: `--traversal reward|punishment`, `--penalty-scale`, `--merit-scale` on `predict` and `generate`
  (`add_traversal_flags`), and the `--json` document carries `traversal`.
* HTTP API: `traversal`, `penalty_scale`, `merit_scale` on `POST /api/predict` and `POST /api/generate`
  (`_traversal_fields`), both servers.
* Frontend: the **Settings** tab (section 31.4) and, as the same control, the Predict and Generate tabs
  (`frontend/src/components/TraversalFields.jsx` over the shared `useNetworkSettings`), with the two scales shown
  only when the punishment traversal is chosen; the Result card reports the traversal that ran.
* Go: `Graph.ChildEvidence`, `Graph.PenaltyCosts`, `Graph.TraversalCosts`, a `CostFn` on `SampleWalk` and
  `BeamOptions`, and `Traversal` / `PenaltyScale` / `MeritScale` on `PredictOptions` and `GenerateOptions`.

Tests: `tests/test_penalty.py` (the split per kind, that the rewards really do leave the score, the cheapest path
being the least punished one, the distribution invariant, the cache, every mode of every kind, the CLI and the HTTP
API), `go/radixnet/penalty_test.go` (the same contract in Go) and
`tests/test_go_parity.py::TestGoParity::test_the_punishment_traversal_matches` (both sides walk the same
least-punished paths at the same costs, under three settings of the scales, for `predict` and `generate`).

### 31.4 The Settings and Model settings tabs

The traversal is a setting of the **network**, not an option of one run, and it was the first of those the
frontend had nowhere to put: it began on a *Network settings* tab beside the score function and the encoder. The
search and training methods (section 35) brought more settings of the same kind - they shape every search and
every run whichever tab starts it - and they split that tab along the line that matters: **who keeps the
setting**.

| tab | what it holds | kept by |
|---|---|---|
| **Settings** (`SettingsPanel.jsx`) | the traversal and its scales; the sampling filters and the diversity; whether a query is asked backwards; the order, the curriculum, the replay and the early stop; the browser's own store | this browser (`useSiteSettings`, section 13.1) - never saved with a model, the same whichever model is loaded |
| **Model settings** (`ModelSettingsPanel.jsx`) | this model (kind, encoding, size, replay buffer); a new model in any kind and encoding; the score function; the attention band; the encoder / decoder | the model - saved in its file, gone when another model is loaded |

| card | what it sets | how |
|---|---|---|
| **Traversal** | the traversal every search runs, and its penalty / merit scales | `TraversalFields.jsx` over the shared store, here in full and on the Predict and Generate tabs in its `compact` form |
| **Sampling and diversity** | `top_k`, `top_p`, `min_p`, `diversity` | `SearchFields.jsx`; Predict and Generate show only what their mode reads and send only what is on (`searchBody`) |
| **Backwards** | whether Predict and Generate ask the model backwards | `BackwardsField.jsx`; never sent - the query is turned around before it goes and the answer when it comes back (`src/backwards.js`). Its training half, `reverse`, is the Train tab's own *Read every text backwards* box beside the files |
| **How a run walks its texts** | `order`, `curriculum`, `replay`, `replay_size`, `patience`, `min_delta` | `TrainingPlanFields.jsx`; the Train tab shows the same fields with a preview of the curriculum and the rehearsal, and sends only what is on (`trainingBody`) |
| **This model** | nothing: it reports | `/api/status` - the kind, the encoding, the size, the file and `replay` (`{size, texts, seen}` or null, all three servers) |
| **New model** | a fresh model | `POST /api/reset {kind, encoding, seed}` - two clicks, since it replaces the model of that kind in memory; the encoding presets and the three dials by hand, checked (`encodingSpec`) before anything is sent |
| **Score function** | the active kind's weight function | `GET /api/model` → `weights` on mount and after every change, `POST /api/model/weights` to apply. The count model's `global_scale / window_scale / reward_scale / count_scale / path_scale / window`, the resonant model's `buckets / period / kick_scale / resonance_scale / amp_scale / reward_scale / concentration`. A kind without one - the sine model, whose score is learned rather than set, and the negative network, whose blame function is on its own tab - gets the sentence saying so and where to look instead |
| **Attention band** | where inside a gram a correction's blame and credit land (section 38) | `GET /api/model/attention` on mount and whenever the model changes, `POST /api/model/attention {on, blur}` to apply; the band drawn over one gram - a bar per position, and a gram's units blurred as the band sees them - redrawn from `src/attention.js` while the slider moves, and `POST /api/model/attention/preview` for a correction the user types, both rules side by side at the slider's blur, before anything is applied. A kind that is never corrected gets the sentence saying so |
| **Encoder / decoder** | nothing: it reports | `GET /api/encoding` for the unit, the n, the stride, the overlap and the four sentinels; `POST /api/encoding/preview` for a text the user types - the grams it becomes (the ones this model has never seen marked), the text the decoder reads back off them, and the walk through the graph's own labels, where a label longer than one gram is a merged radix chain |

The decisions worth stating:

**One setting, several doors.** Each site-wide setting is edited in several places and is one value: a provider at
the top of `App.jsx` holds it, every panel reads it through `useSiteSettings()`, and changing it anywhere changes it
everywhere at once. Two `useStoredState` calls under one name would have shared the *stored* value and not the
state, and every panel here stays mounted while hidden, so they would have disagreed until a reload. The old hash
`#network` still opens the model's tab.

**A tab sends only what its search reads.** The filters shape a sampled walk and the diversity a beam; the exact
searches read neither. So a request carries a setting only when its mode reads it and it is not off, and a value
out of range stops the tab from sending anything at all - with the reason beside the field - rather than being
sent to a 400. A server older than the setting never sees it.

**The score function lives with the model.** It was a fieldset on the Train tab once, count-model-only and missing
`path_scale`; how an edge is scored is a property of the model that is saved with it, not a setting of a training
run, and having it in two places would have had the same drift problem.

**The encoder is chosen once, not edited.** The encoding is a dial (section 34: the unit, the n and the stride),
but one a model is *born* with rather than one it can be moved between: the labels, the split and merge rules and
the saved file are all measured in its units, so a graph built at `char:3:1` could not be read at `word:2:1`. The
New model card is therefore where an encoding is chosen, and the encoder card spends its space on making the
encoding *visible* - the grams a text becomes, the ones this model has never seen, and the walk through the graph's
own labels - which is also the clearest demonstration of the radix compression anywhere in the frontend.

## 32. The least-punished traversal (`radixnet/search.py`, `go/radixnet/search.go`, `rust/src/search.rs`) — ranking a walk by what went wrong

The full argument, the alternatives rejected and the measured behaviour are in `SPEC-LeastPunished.md`; this is what
the code must do.

**This is not section 31's traversal, and the two do not compose.**  Section 31 replaces the *cost function*: the
punishments price a step and the walks are then ordered by summed cost exactly as always, so ten small penalties
and one large one can trade off against each other and a reward further along can pay for a punished step.  This
one leaves every price alone and replaces the *ranking*: a walk goes by the blame on its **worst** step first, and
a node offers only the children it has the least against, so blame cannot be bought off at all.  They are two
answers to one question, both selected by `--traversal` (`punishment` and `least-punished`), and a request for one
never silently gets the other: `penalty.traversal_costs` returns `None` for `least-punished` because there is no
cost function to hand out, and `search.parse_traversal` maps `"punishment"` to the *reward* ranking because the
blame is already in its costs.  The names are never aliases of each other.

**A traversal is a per-call choice**, `reward` (the default, nothing changes) or `least-punished`.  The name is
parsed from `""` / `"reward"` / `"rewards"` / `"cost"`, or `"least-punished"` / `"least_punished"` / `"punished"` /
`"punish"` / `"blame"`, and reaches the search through `predict(traversal=)` / `generate(traversal=)` /
`beam_predict(traversal=)` / `sample_walk(traversal=)` in Python, `PredictOptions.Traversal` /
`GenerateOptions.Traversal` / `BeamOptions.Traversal` / `SampleWalkBy` in Go and the same options in Rust.  The CLIs
carry it on `predict` and `generate` (and the Go and Rust `bench`) as `--traversal`; the HTTP API takes `traversal`
on `/api/predict` and `/api/generate`.

**A model that keeps no record of failure refuses it.**  `RadixNet` and `ResonantNet` accept the argument and raise
on `least-punished` (they take `reward` and `punishment`, which section 31 gives every kind): ranking a walk by
the blame on it needs failures to have been *counted per path*, and the sine and resonant models count none.  Their graphs
answer 0 to `edge_punishment` / `step_punishment`, so the machinery is inert rather than wrong where it does reach.

**The punishment of a step** is `punish(prev, e) = reward_scale · max(0, -edge_reward[e]) + path_scale ·
log1p(incorrect(prev, e))`, where `incorrect(prev, e)` is the failure count of the judged path context (section 16.5)
and is `0` where the walk's caller is unknown or the context has never been judged.  Both terms are **one-sided**:
the first ignores a positive reward, the second counts the failures *against nothing* (the cost function's path term
is the one that weighs them against the successes, and stays as it is).  `EdgePunishment` is the first term,
`StepPunishment` both, `PathIncorrect` the counter behind the second; `ChildCost.Punish` carries the answer beside
the cost, filled per edge by `ensureCosts` and per judged context by `ensureContextCosts`.

**A path's punishment is the maximum over its steps**, not the sum - the same reading the guard's `peak` takes
(section 24.7).  Both quantities a partial path carries are therefore monotone along it (a sum of non-negative costs,
a running maximum), which is what keeps the beam's early exit valid.

**The order is lexicographic**, `(punishment, cost)`, and under `ByReward` the first component is 0 on every path, so
the comparison is the cost order the beam has always used.  Concretely, in `runBeam`: a kept path is a
`(first, second, entry)` key whose `first` is `-punish · sign` only under `ByLeastPunished` and `0` otherwise; the
frontier sorts by `lessState` with punishment ahead of cost; the bottom beam reverses both, so the k worst paths are
the **most punished** ones; and the early exit stops when the best partial path is no better than the k-th finished
one on that same order.  Two punishments within `PunishTolerance` (`1e-12`) are the same punishment.

**A node offers fewer children.**  `Onward` drops a hand-over to `BACK` as before; under `ByLeastPunished`
`LeastPunished` then keeps only the children whose punishment is the minimum at that node, and returns the slice
untouched when they all tie (which is every node of a graph nothing was punished on, so the filter costs one pass and
no allocation there).  `SampleWalkBy` samples from the softmax over exactly those children: punishment decides what
may be walked, the cost decides which of them it is.

**What a walk reports.**  `PathResult.Punish` is its worst step, and `Prediction.Traversal` names the search when it
was not the ordinary one.  Both are omitted from JSON when zero or empty, so a reader of today's prediction sees no
new field.  Nothing in the traversal writes to the graph, and nothing about it touches the model file.

**Tests** `tests/test_countnet.py::TestLeastPunishedTraversal`, `go/radixnet/traversal_test.go` and
`rust/tests/model.rs`: identical answers, costs and expansions where nothing is punished; the blamed step left even
when it is five times rewarded and an order of magnitude cheaper; 50 units of reward failing to buy the blame off;
the filter's minimum, tolerance and infinity; the names; and a model with no record of failure refusing the option.
Both parity suites (`tests/test_go_parity.py`, `tests/test_rust_parity.py`) punish the same texts on both sides and
require the same continuation, cost, ranking **and punishment** under the traversal - and require that the punished
model answers the two searches differently somewhere, so neither can pass by the traversal doing nothing.

## 33. The Rust port (`rust/`) and the cross-language benchmark (`bench/`)

`rust/` is a standalone crate (edition 2021, **no dependencies**) porting section 19's model a second time: the
graph and its structural operations, the dual frequency weight function, the judged path contexts, **all three
traversals** (§31's punishment one in `src/penalty.rs`, §32's least-punished one in `src/search.rs`), training,
prediction, generation, scoring, reward / punish / 2NRL, **the encoding dial** (§34) and **the model file** - and,
since D-077, the rest of the package: the sine-activation and phase models, the negative network and the guard,
every teaching loop, the LLM clients (HTTPS through the system `curl`, D-076), the tools, code generation and the
agent, images and speech, MCP and the WebDriver browser, one module per area with one parity suite per area (`tests/test_rust_parity*.py`).  What it
deliberately leaves out - the torch backend, the Stable Diffusion encoder, local Whisper - `rust/README.md` names.

The dial is the one place the port paid a representation for generality.  A trigram used to be three code points
packed into a `u64` - `Copy`, hashable, no allocation - which is why its encoding pass allocated nothing and is one
of the three reasons D-072 measured it faster than Go.  A gram of *any* n over *any* unit is arbitrary text, so the
index is keyed by the gram as Python and Go key it, and what that costs is measured rather than argued
(`bench/RESULTS.md`).  `src/encoding.rs` carries `Encoding{unit, n, stride}` and a `Units` view that indexes a text
once so slicing by unit stays O(1), which is what the graph does on every split, merge and re-index.  Its binaries
are `radixnet` (the CLI: every command of Python's, the model's own in `src/bin/radixnet.rs` and every other area's
through `cli::COMMANDS`) and `radixnet-bench`.

**The server is here too** (`src/http.rs`, `src/service.rs`): HTTP/1.1 written out over `TcpListener` - the request
line, the headers, `Content-Length`, the static files of `frontend/dist` with the SPA fallback, and a thread per
connection - under the whole JSON contract §12 defines and the Python and Go servers answer, down to `engine` naming which one
is replying.  `frontend/dist` runs against `radixnet serve` unmodified: `/api/status` lists every route the server
has (`routes`) and a tab is shown against the Rust server when the route it needs is in that list, and what it
will not serve says so with a 400 rather than a half answer (`/api/words` on a character model).

**Uploads stream, as they do on the Go server** (D-035).  `POST /api/uploads` is the one route registered as
*streaming* (`Server::stream_route`): the request's head is read, the handler is given the body as it arrives -
ending where `Content-Length` says, and failing on a client that sent less - and `multipart::store_stream` writes a
multipart file part or a raw body straight to a `.part` file in the upload directory (`stream_parts` finds the
parts on the way, a window at a time, and the listing skips `.part` files), validates an archive from that file
(`source::inspect`, a batch of entries at a time, the pass remembered for the listing's record) and moves it into
place under a `.zip` name; a text file is converted into place a chunk at a time (byte-order mark dropped, bytes
that are not UTF-8 replaced) and counted a line at a time.  The JSON forms, which carry the file inline, are read
whole.  Every other route reads its body whole under a 16 MiB cap (`http::MAX_BODY`), which is what keeps a bad
`Content-Length` from asking for the machine's memory; a client that sends `Expect: 100-continue` (curl, for a
large file) is answered before the body goes; and an answer goes out in one write, so a connection closed on a
body the server did not read - a refused upload - still carries it.  So no server refuses an upload for its size.

Three things it has to get right for the frontend rather than for the model, each of them a rule of §12 rather than
of the port:

* **`/api/model/select` switches encoding**, between characters and words, the way the Python service switches
  kind: an encoding is fixed for a model's life, so changing it means a *different* model, and the one that was
  running is *parked* with its unsaved work rather than dropped.  The one selected is whichever comes first of the
  parked model of that encoding, its own file (`<stem>.<unit><ext>`) and a fresh one - which the answer says in
  `origin`.  A bare `{"kind": "word"}` names a unit and not the other two dials, so they come from the model that
  unit last had, which is what makes select-word, select-count, select-word come back to the same model rather than
  to a third.  `POST /api/reset` is where an encoding is *chosen*, as on the other two servers.  Without any of
  this a server started on characters could never reach the Words tab.
* **A job answers 202 and runs on its own thread.**  `train` and `2nrl` open the job, hand the model to a worker
  and reply at once, so the frontend follows the run on `/api/job` rather than waiting out a long training run on
  one blocked request.  The model lock is what makes a second request wait, and `ensure_idle` is what refuses a
  second job.
* **The texts of a request may come three ways** - `texts` as a list, `text` as one per line, `files` as the
  uploads to read, added together (`good` / `good_text` / `good_files` and their `bad` twins for the feedback
  routes).  A server that reads only the first of the three silently ignores the uploads a client ticked.

Not ported, and **undone** rather than deliberate: the negative network, the tutors and the other teaching loops,
the agent and its tools, images and speech, and MCP.  One gap is **deliberate**: the LLM clients (Ollama, ChatGPT)
need HTTPS, and a crate with no dependencies cannot speak it.  Porting them means giving up D-072's rule, which is
a decision to take rather than a thing to quietly do.

**The file is the contract** (`src/file.rs`, `src/json.rs`, `src/gzip.rs`, `src/clock.rs`).  Three things had to be
written out rather than translated, for the same reason the Go port had to write out BLAKE2b: JSON rendered as
Python's `json.dumps` renders it - compact separators, UTF-8 rather than `\u` escapes, and **floats formatted as
`repr(float)` formats them**, which is not how Rust prints a float (`0.0` prints as `0`, `1e-05` as `0.00001`); the
gzip container, inflate included, because `model.json.gz` is an ordinary model file and the standard library has no
gzip; and the ISO-8601 timestamp `saved_at` carries.  What comes out is Python's graph document **byte for byte**,
but for the `version` cache stamp that every load bumps - a stricter bar than the Go port meets, and free once the
writer is right.  `tests/test_rust_parity.py` asserts exactly that, alongside the structure, the predictions, the
judged paths, the node ratios, the wrapped counters and each side continuing the other's file, gzipped or not.

What the port has to get right, beyond the algorithm:

* **Counting is atomic** (`AtomicI64`, relaxed), the equivalent of Go's `--exact`.  A benchmark of Go's
  racy-by-design counting (D-038) would be measuring the race.
* **The order of every sum is the Python one** - the left-to-right row total in `recomputeRow`, `fsum` over the step
  costs, insertion-ordered adjacency - because the benchmark's parity check compares the *cost of a path* to the bit.
* **The fan-out is a thread pool**, not a thread per text: `parallel_for` / `parallel_fill` hand each worker a block
  of the chunk, `workers = 1` runs everything on the calling thread.  The two beams of a prediction run in turn on
  the calling thread; Go runs them on two goroutines unless `--workers 1`, which was changed to match so a
  one-worker row means the same thing on both sides.
* **The one `unsafe`** is `parallel::Disjoint`, a raw view of a slice shared by the weight and cost recomputes, which
  fan out over nodes and write per-edge values.  Every edge belongs to exactly one node (an invariant
  `check_invariants` proves), so no two threads write one index; the contract is written out where it is defined.
* **The traversal is a name, not a type.**  `PredictOptions.traversal` is a `String` resolved by
  `penalty::resolve_traversal`, exactly as Python's `parse_traversal` resolves it, so the CLI flag, the API field
  and the library argument all take the same three words and the same aliases.  `penalty::traversal_costs` hands
  back a cost function for `punishment` and `None` for the other two - `reward` because the graph's own costs are
  the reward traversal, `least-punished` because it is a *ranking* over those costs and never a second pricing of
  them (§32).  The two are never composed in any of the three implementations.

**`bench/`** is the comparison.  `make_corpus.py` writes `corpus.txt` and `corpus.prefixes.txt` once - neither
language can reproduce the other's RNG, so neither generates the corpus - and `compare.py` builds both, runs six
builds (Go racy, Go exact, Rust; one worker and all cores) under both traversals, repeats each and reports the
median into `RESULTS.md`.

**The parity check is a precondition, not a test.**  Before any timing is printed, the Go and Rust runs of each
configuration must agree exactly on `chars`, `texts`, `nodes`, `edges`, `trigrams`, `transitions`,
`dijkstra_expansions`, `punished_texts` and `punished_edges`; to `1e-9` on `compression_ratio`, `loss` and
`edge_reward_negative`; and on the sample prediction's continuation and its cost.  A disagreement stops the run and
names what differs.  `--punish-every N` punishes every Nth text before the predictions are timed, because a model
nothing was ever punished for gives the two traversals nothing to disagree about.

---

## 34. Word n-grams (`radixnet/wordnet.py`, `go/radixnet/words.go`, `rust/src/words.rs`) — the same graph over an alphabet of words

`SPEC-WordNGrams.md` and D-073.  D-006's input decision is about what a **symbol** is; every structural rule in
section 5's graph is stated in terms of a window of symbols and nothing else - a label is a sequence of them, an edge
exists where two labels overlap by `WINDOW - 1` of them, the index maps a `WINDOW`-symbol key to `(node, offset)`,
compression merges a unary chain at that seam.  Not one of those rules mentions a character.  So a word n-gram model
is not a different model: it is the count / reward model (section 19) over an alphabet whose symbols are **words**.

**The alphabet** (`encoding.Vocabulary`).  A label is a string and its length is counted in code points on all three
implementations, so a symbol is one code point:

```
id 0            -> U+0100          the unknown word, <unk>
id i (i >= 1)   -> U+0100 + i      skipping the surrogate block D800..DFFF
```

Starting past Latin-1 keeps every label out of the range text, sentinels (`<s>`, `</s>`, `<back>`) and punctuation
live in; skipping the surrogates keeps every label a string all three languages can hold and write (Go replaces a
lone surrogate with `U+FFFD`, which would silently corrupt a model).  The cap is **1 111 808 words**: every code
point above Latin-1 that is not a surrogate.  A `Trigram` in Rust packs three code points of 21 bits, which is
every one of them, so the fastest representation needed no change at all.

**Tokenising is the whitespace split and nothing else.**  Punctuation stays attached to the word it touches
(`"mat."` and `"mat"` are two words) and case is kept (`"The"` and `"the"` are two words).  The rule is the Unicode
`White_Space` property, so the three implementations cut a text into the *same* words - Python calls U+001C..U+001F
whitespace and the property does not, which `split_words` subtracts.  The vocabulary grows as training reads new
words and is never frozen, pruned or learned: a word is in it because it was seen.

**What a round trip costs.**  `decode(encode(t))` joins the words with single spaces, so a word model **normalises
whitespace** and a corpus whose whitespace carries meaning - source code, base64, a waveform (D-036) - belongs
on the character model.  At prediction and scoring time a word the model has never read is `<unk>`, whose windows are
not in the index, so it is charged what any unknown transition is charged and counted in `unknown_transitions`; two
different unread words are therefore one symbol and score identically.  Both kinds stay; neither replaces the other.

**Where the translation lives.**  Text becomes symbols at the model's own boundary - `_clean_texts` on the way in,
the search and `score` around their own calls - and symbols become text on the way out of the search.  Between those
two points the graph, the weights, the counters, the path contexts, both traversals and the beam are the count
model's, unchanged.  Three seams in shared code make the alphabet a model's business rather than the search's:
`graph.text_of(label)` (a label as text) and `graph.symbols_of(text)` (its inverse, for looking a node up by what a
user typed), a model-level `units`, and `_whole_text` no longer re-joining a prefix the search already joined.

**What it looks like.**  Compression does to word chains what it already did to character chains, so a repeated
phrase becomes one node whose label *is* that phrase:

```
$ make word-train word-predict
"<s>"  visited 180x  (0 in, 60 out)
to  "the cat sat on the mat"    3   2%
to  "the dog chased the cat around the garden"   3   2%
prefix        "the cat sat on"
continuation  "the mat"
```

**The file** is a format of its own, `radixnet-word`, so every reader written before it existed refuses it by the
check it already makes; the graph block gains `units: "words"` and `vocabulary` in id order, and a load checks that
every symbol every label carries is a word the vocabulary holds.  Interchange is the contract the count model has:
Python, Go and Rust read and write it, `tests/test_go_parity.py` and `tests/test_rust_parity.py` train a word model
on both sides of each pair and require the same structure, the same counts, the same vocabulary **in the same
order** and the same predictions, and the Rust word document is Python's byte for byte.

**Everything counted in symbols is counted in words**: `--length 6` emits six words, `--max-length` caps words,
`Score.chars` counts words and `per_char` is per word, and `trained_chars` counts words.  `stats()["units"]` says
which, and so do the CLI's score columns (`per_word words`), the API's status and the frontend's length hints,
because a number whose unit depends on the model is a number that will be read wrong.  The kind is `--kind word`
(default file `model.word.json`) in all three CLIs, `radixnet --kind word words` lists the alphabet, `GET
/api/words` serves it and the frontend's **Words** tab shows it whenever the active model counts in words.

**N stays 3**: the overlap is two words and the pivot is the middle one, which is D-006's argument word for
word.  Word bigrams need `WINDOW = 2`, a separate change with its own costs, and nothing here prevents it later.

## 35. Search and training methods (`radixnet/training.py`, `go/radixnet/training.go`, `rust/src/training.rs`) — sampling filters, a diverse beam, curricula, replay and early stopping

`SPEC-SearchAndTraining.md` is the contract, precise to the rounding and the tie; this is where it lives and what it
cost. Every setting is **off by default**, and off it changes nothing: the draw, the beam, the training pass and the
model file are exactly what they were, which the existing parity suites confirm unchanged.

**Search.** `sampling_filter` (`search.py`; `SamplingFilter` in Go and Rust) narrows the options of a sampled step -
`top_k`, then `min_p`, then `top_p` - ranking by `(cost, position)` and handing the survivors back in the order the
node offered them, so the draw that follows is the one it always was with fewer options. Two rules keep three
implementations on one random stream: the cheapest option always survives (so `lowest`, and every weight, is the
same number), and a step with a positive temperature and more than one option draws exactly once however few
survive. `sample_walk` and the resonant model's `phase_walk` both use it. `diverse_pick` (`beam.py`) is the diverse
beam: the top beam keeps a pool of `max(k, width)` finished paths instead of `k`, and the K are picked from it by
maximal marginal relevance - `(punish, cost + diversity · overlap, position)`, the overlap being the share of the
shorter path's nodes two paths share from the start. It replaced a first attempt that penalised paths ending in the
same node inside the frontier: that changed nothing on real graphs (the endings differ node by node) and, tuned
harder, pruned the best path; picking from finished paths cannot lose the best one, which is always the first
pick. Only the top side is spread; `phase_beam` applies the same pick.

**Training.** A `TrainingPlan` is built once per plain `train` call from the run's usable texts: the order
(`corpus`, `shortest-first`, `longest-first`, `shuffle` - the last by SplitMix64 keys of the seed and the epoch
number, so no random number is drawn and the model's generator stays where it was), the curriculum's share per
epoch (`ceil(frac_j · n)`, linear from `c` to 1), the rehearsal slice of the replay buffer, and the early stop
(skipping epochs still inside the curriculum). The structure pass still observes every text - and, when any is
rehearsed, the whole buffer - before the first epoch, so a plan changes what each epoch *counts*, never the graph's
shape. The `ReplayBuffer` is bottom-k sampling by `replay_key(seed, g)`: uniform over every text ever offered,
whatever order they came in, deterministic, and small enough to write into the model file (`replay`, last, only when
there is one; a reader recomputes the priorities rather than trusting the order). A pass stamped with a feedback
phase - 2NRL, a thumbs up or down, the agent's punishment - is not planned: a punished text is not one to rehearse.

**Where it runs.** Python: every kind that learns by walking a list of texts (count, sine, resonant). Go: the count
model (the only one it has); a streaming source is read into memory when a plan needs the whole list. Rust: every
kind, byte for byte with Python. CLI: `--top-k --top-p --min-p --diversity` on `predict` / `generate`, `--order
--curriculum --replay --replay-size --patience --min-delta --reverse` on `train`. HTTP: the same names on
`/api/predict`, `/api/generate` and `/api/train` on all three servers, with the same 400s, and `/api/status` reports
the buffer. Frontend: the Settings tab and the controls it shares with Predict, Generate and Train (section 31.4).

**Reading backwards** (`reverse`, the spec's section 9, D-084) is the first thing a training call does: every text
is turned around in the encoding's units (`Encoding.reverse` - code points, or whole words with single spaces), so
the rest of the run - the short texts dropped, the plan, the structure pass, the counters, the replay buffer - sees
the reversed texts, and a reversed run is byte for byte a run over the reversed texts. It is not a plan in the
sense above - the order of the texts is `order`'s - and it streams: Go wraps its source (`ReversedSource`, keeping
an archive's entries as parts), so a reversed run needs no list in memory. Every kind honours it, the negative
network (whose training is blaming) included; the feedback routes do not take it. Python applies it in each kind's
`train` (`GraphModel._read`), Go in `Train` and `TrainSource`, Rust in the three trainers the plan reaches (the count
model's passes, `radix_train`, `resonant_train`) and in `kinds::train` for the negative network. Asking a model
trained so is the caller's half: the frontend turns the query and the answer around (`src/backwards.js`).

**Fixed on the way.** The lead of a prefix (the unmatched rest of the located gram) was measured in characters where
`length` is in units - a word model asked for three words got three minus the lead's *characters*; the resonant
model's score counted `chars` in characters and its `trained_chars` too. All count units now, in every port.

Tests: `tests/test_search_training.py` (the rules, pinned keys, every kind's planned training, the phase rule, the
HTTP API and the CLI; reading backwards on every kind and both units, an upload trained on backwards over HTTP and a
file with `train --reverse`), `tests/test_rust_parity_methods.py` (Rust against Python on eight plans in three kinds,
two of them backwards - graph, history and replay block byte for byte - the filters and the diverse beam, and the
server), `tests/test_go_parity.py::TestGoSearchAndTraining`, `go/radixnet/training_test.go`,
`go/radixnet/sampling_test.go`, `go/server/server_test.go`, the `training`, `search`, `beam` and `cli` unit tests
and `tests/server.rs` in Rust, and `frontend/test/settings.test.mjs` and `frontend/test/backwards.test.mjs`.

## 36. Thinking (`radixnet/thinking.py`, `go/radixnet/thinking.go`, `rust/src/thinking.rs`) — the `THINK` sentinel at work

Section 5.1.2 gives the graph a fourth sentinel that faces both ways: `p -> THINK` is where the model has learned
to stop and think, `THINK -> ...` is how its thoughts begin. This section is what happens between the two - what
triggers a thought, what a thought is, how it questions itself, and what it triggers when it stops - and where
the thoughts come from in the first place.

### 36.1 Triggers, and the four things a thought does

`think(model, about=..., at=..., trigger=...)` is one thought, and it is always **triggered** by an event:

| trigger | the event | when the thought stops it triggers |
|---|---|---|
| `asked` | somebody asked (`radixnet think`, `POST /api/think`) | `end` - the thought ends |
| `stutter` / `repeat` | a voice caught itself repeating (`dialogue.think_back`, section 22) | `back` - `observe_back` on the node, with the step it was about to loop through and the one it took instead |
| `questioned` | a thought asked itself | `think` - it returns to the thought that asked |

In order: (1) it **teaches where** - `observe_think(at)` on the node the event happened at (`about` locates it:
`place` finds the node the text ends at and splits it so the text ends exactly there), unless the thought is a
question the model asked itself, which already knows to think there - that is why it asked; (2) it **thinks** -
the prediction search run from `THINK` to the end of a text (`_search(..., origin=THINK)`; beam for the most
likely thought, sample for a drawn one), so the thought is in the language of the thoughts it was taught, and
nothing when it was taught none (`stopped: "nothing"` - the honest answer rather than a made-up one); (3) it may
**question itself** - along its own `node_ids`, at every node where `thinks_at` is true, a nested thought with
trigger `questioned`, one level deeper, that must say something the chain of thoughts above it has not (up to
`max_depth`, default 2, and `max_questions` per thought, default 1, so a thought cannot spend itself questioning);
(4) it **stops**, and triggers the sentinel in the table. Every thought is a `Thought` record - `trigger`, `at`,
`about`, `text`, `depth`, `stopped` (`end` / `length` / `nothing`), `then`, `taught`, `handed_over`, `cost`,
`probability`, `expanded`, `questioned`, `questions` (the same records, nested), the path - and `learn=False`
thinks without writing anything into the model.

### 36.2 Where thoughts come from: an LLM's thinking (`think_on`, `ollama think`)

A model cannot think in words it has not learned, and no corpus is written in thoughts. Section 16.1's
`thoughts_from_prompt` asks a thinking model for questions about a topic and captures the *thinking* behind each
answer; `think_on(model, thoughts, questions=True)` trains every thought from `THINK` (`train(origin=THINK)`) and,
with `questions`, reads the thoughts for the sentences that end in `?` (`questions_in`): the node before each one
is taught to stop and think (`observe_think`, on the node `place` cut to end exactly there) and the question is
trained as a thought of its own. So a thought that later passes that node may question itself with it - the
network learns to question where its teacher did. `radixnet ollama think --prompt TOPIC --train` (`POST
/api/ollama/think {"train": true}`) is the whole pipeline; `--with-answers` trains the answers as ordinary texts
too, and the negative network refuses thoughts (it judges; it does not think).

The tutor is the second source (section 16.3): its marker reasons about the very sentences the network wrote before
it marks them, so `tutor --learn-thinking` (the Tutor tab's *Train on the teacher's thinking*, `{"learn_thinking":
true}`) asks for that reasoning while it marks and teaches it through the same `think_on` after every round - the
network learns to question itself where its teacher did, about its own mistakes.

### 36.3 What it touches, and the three ports

* `graph.py`: `THINK`, `FIRST`, `ORIGINS`, `THINK_Z`, `observe_think`, `think_cost`, `thinks_at`, `origin` on
  `observe_sequence` / `trace` / `node_path`, graph format 4 (section 5.1.2); `countnet.py` and `resonance.py`
  override `observe_think` the way they override `observe_back`, and thread `origin` through their passes.
* `search.py` (`onward` drops `THINK`; both origins decode alike), `beam.py`, `phasesearch.py`,
  `countnet._walks` / `record_path` (a path's first context is its origin).
* `model.py`: `_walk_start(prefix, origin)` (a located prefix wins over the origin), `origin` on `_search`,
  `_observe` and `train`.
* `dialogue.py`: `think_back`, `Rethink.thought`, `think` / `think_depth` on `backtrack`, `reply`, `converse`.
* `ollama.py`: `complete`, `think_value`, `split_thinking`, `questions_from_prompt`, `thoughts_from_prompt`, the
  `think` field on `generate` / `chat`.
* `cli.py`: `think`, `ollama think`, `converse --no-think --think-depth` (and *it learned to stop and think at N
  node(s)*); `api.py`: `POST /api/think`, `POST /api/ollama/think`, `think` / `think_depth` on `/api/converse`,
  `think_label` on `/api/encoding`.
* `tutor.py`: `_complete`, `think` / `thinking` on `grade_completions`, `TutorConfig.think` / `learn_thinking` /
  `think_questions` / `resolved_think`, `TutorTrainer.teach_thinking`; `tutor --think --learn-thinking
  --no-think-questions`, the same fields on `/api/tutor/start` and `/api/tutor/lesson` (section 16.3).
* Go - `thinking.go` (`Model.Think`, `ThinkOn`, `Place`, `QuestionsIn`, `Summarize`, `ThoughtsOf`), `graph.go`
  (`Think`, `First`, `IsOrigin`, `ThinkZ`, `ObserveThink`, `ThinkCost`, `ThinksAt`, `ObserveFrom` / `TraceFrom` /
  `NodePathFrom`), `json.go` (`withSentinels`), `search.go`, `model.go` (`TrainOptions.Origin`, `walkStart`),
  `dialogue.go` (`ThinkBack`, `Rethink.Thought`), `ollama.go` (`Complete`, `ThinkValue`, `SplitThinking`,
  `QuestionsFromPrompt`, `ThoughtsFromPrompt`), `tutor.go` (`TeachThinking`, `thoughtfulClient`), the same routes in
  `server/` and the same commands in `cmd/radixnet-count` - and Rust (`thinking.rs`, `graph.rs`, `file.rs`,
  `search.rs`, `model.rs`, `dialogue.rs`, `ollama.rs`, `llm.rs`'s `complete_thinking`, `tutor/trainer.rs`'s
  `teach_thinking`, the `think` command and routes) produce the same records: the Go parity tests compare the
  rethinks' thoughts field for field, the Rust ones the model files byte for byte.
* Tests: `tests/test_thinking.py`, `TestThinkSentinel` in `test_graph.py`, the origin cases in `test_search.py`,
  `TestThinksBeforeBackingUp` in `test_dialogue.py`, the thinking cases in `test_ollama.py` and `test_tutor.py`;
  `go/radixnet/thinking_test.go`, `go/server/thinking_test.go`, the thinking cases in `go/radixnet/tutor_test.go`
  and `go/server/tutor_test.go`; the unit tests in `rust/src/thinking.rs` and `rust/src/ollama.rs`; the tutor's
  thinking held to Python's by `test_go_parity.py` and `test_rust_parity_teach.py`.

## 37. Today's format (`radixnet/assistant.py`, `go/radixnet/assistant.go`, `rust/src/assistant.rs`) — messages in, thinking and a streamed reply out

Every language model is talked to through one shape now - a list of `{"role", "content"}` messages in, an assistant
message out, its *thinking* first and then the text, streamed over server-sent events - and every client, SDK and
front end speaks it. This area gives the model that shape in the two dialects that cover the ecosystem: **OpenAI's Chat
Completions** (`POST /v1/chat/completions`) and **Anthropic's Messages** (`POST /v1/messages`), with `GET /v1/models`
and `POST /v1/messages/count_tokens` beside them, the `talk` command and the Talk tab. It adds no prompt, no template
and no instruction following (D-080): everything it returns is a rendering of what §22's reply already produces.

### 37.1 The request (`Ask`)

`parse_openai(body)` / `parse_anthropic(body)` / `parse_request(body, dialect)` read a request into one `Ask`; a bad one
raises `AskError(message, param)`, `param` naming the field to blame.

* **`messages`** -> `Ask.messages`, a list of `Message(role, text)` with `role` in `user` / `assistant` / `tool`, and
  `Ask.system` (the `system` / `developer` lines joined by newlines - OpenAI - or the top-level `system` string or text
  blocks - Anthropic). Content is a string or a list of parts: `text` (and OpenAI's `refusal`) is read; `thinking` and
  `redacted_thinking` blocks are skipped (what the model thought is not what it said); an assistant's `tool_calls`
  (OpenAI) or `tool_use` blocks (Anthropic) are rendered as the `<tool>name {...}</tool>` lines §27.1's `call_text`
  writes; a `tool` message (OpenAI) or a `tool_result` block (Anthropic) as `result_text` (`<result>…</result>`, clipped
  at `DEFAULT_OBSERVATION_CHARS`); an image, a document, audio or any other part is refused
  (`content of type 'image_url' is not supported; this model reads text`). A message whose text is blank is dropped (it
  says nothing and is not heard). Anthropic's list may only hold `user` and `assistant`. At least one message must remain.
* **The caps and the sampling**: `max_tokens` (OpenAI also `max_completion_tokens`; default `DEFAULT_MAX_TOKENS` = 60,
  the dialogue's `max_length`, in the model's *units*), `temperature` (≥ 0, default 1), `stop` (OpenAI: a string or a
  list) / `stop_sequences` (Anthropic: a list) of non-empty strings, `n` (OpenAI, 1..8), `stream`,
  `stream_options.include_usage` (OpenAI), `tools` (OpenAI `{"type": "function", "function": {"name", "description",
  "parameters"}}`, Anthropic `{"name", "description", "input_schema"}`, normalised to `{"name", "description",
  "parameters"}`; names must be identifiers and distinct), `tool_choice` (`"none"` / `{"type": "none"}` withdraws them;
  anything else is accepted - the model cannot be forced to call), `thinking` (OpenAI: a boolean, or
  `reasoning_effort: "none"` for off; Anthropic: `{"type": "enabled" | "disabled" | "adaptive"}` or a boolean; default on).
* **The dialogue's dials**, by the names `/api/converse` uses, in either body: `mode` (`beam` | `sample`, `dijkstra`
  reads as `beam`), `context` (12), `k` (5), `beam`, `step_penalty` (0), `explore` (3), `avoid_repeats` (true),
  `avoid_word_repeats` (true), `learn` (true), `guard` (true), `seed`.
* Everything else a client sends (`top_p`, `top_k`, `logprobs`, `user`, `metadata`, `response_format`, `logit_bias`,
  `presence_penalty`, ...) is accepted and ignored.

`Ask.previous` is the last message's text - the line the reply continues; `Ask.prefill` is whether that message is the
assistant's own.

### 37.2 The reply (`respond`)

`respond(model, ask, *, pair=None, on_event=None, name=None) -> Reply` answers under whatever lock the caller holds
(`ModelService.respond` takes the session; with `ask.learn` the model may change). For each choice `0 ≤ i < n`:

1. `Heard` is every message's text (both sides of the conversation, the system prompt excluded); after each choice the
   turn spoken is remembered, so `n` alternatives differ because each hears the last.
2. `dialogue.reply(model, previous, heard=…, index=len(messages), speaker="assistant", rng=Random(seed), veto=…,
   trace=…, **dials)` - §22's turn, unchanged. `veto` is the pair's judgement when a guard is given (§24.7: a candidate
   is judged once; a rejected one is a line of the thinking, and with `pair.config.learn` the rejected texts are blamed
   as `vetoed in conversation`, as `NegativeFilter.converse` does). `trace` is the hook below.
3. The text is the turn's whole utterance (`turn.text`: the context picked up plus what was added), or - for a
   prefill whose turn picked up a context - only what it added (`turn.reply`). The earliest stop sequence cuts it
   (`stop_reason = stop_sequence`, `stop_sequence` set); a `<tool>` call at the front of what is left, parsed with
   `tools.parse_call` and, when its name was offered, re-read against the offered tools' schemas (`offered_toolbox`: the
   properties as `Param`s, so a bare value is the one argument of a one-argument tool), becomes a `ToolCall(token, name,
   input)` and cuts the text before the call (`stop_reason = tool_use`); a call to a tool not offered, or one whose
   arguments cannot be read, stays text and the thinking says why.
4. `stop_reason`: `end_turn` when the walk reached END, else `max_tokens`; `stop_sequence` / `tool_use` as above;
   `refusal` when the reply is `None` and something was vetoed (the guard vetoed everything it could say); `end_turn`
   with an empty text when the graph has no way on at all.
5. **Usage** is in the model's units (`Encoding.length`): `input_units` = the system prompt plus every message;
   per choice `thinking_units` = the thinking's length and `output_units` = the text's plus the thinking's.

`Reply(token, model, created, kind, units, input_units, choices, thinking)`; `Choice(index, thinking, text, tool_calls,
stop_reason, stop_sequence, turn, guard, output_units, thinking_units)` with `turn` the `Turn.to_dict()` record and
`guard` the `_guard_report` (`on`, `vetoed`, `rejected`, `verdicts`, `negative`, `config`) or `None`.

### 37.3 The thinking: the search's trace (`dialogue.Trace`, `_Narrator`)

`dialogue.reply` gained `trace: Callable[[dict], None] | None`. It receives, as they happen, `{"kind": "context",
"context", "usable"}` for each tail of the line tried, `{"kind": "candidates", "context", "offered", "mode"}` for each
search run from it, `{"kind": "pick", "skipped", "vetoed", "repeat", "spoken", "caught"}` for what one look through them
came to, `{"kind": "rethink", "kind_of", "noticed", "cut", "steps", "explored", "found", "taught"}` for a §22
`Rethink`, and `{"kind": "fresh"}` when the subject is changed. `converse` and the chat loop pass none; without one a
reply is exactly what it was. Go's `ReplyOptions.Trace` and Rust's `ReplyOptions::trace` are the same hook.

The narrator turns the trace into lines; the thinking is the lines joined by newlines, and the lines are **fixed
formats every port writes character for character** (`q(x)` is the text in JSON double quotes, `N item(s)` the plural
form, floats `%.4f`):

| when | line |
|---|---|
| first | `answering q(previous)` or, for a prefill, `continuing its own last line q(previous)`; with earlier messages ` (N earlier line(s) heard)` |
| a system prompt was given | `a system prompt was given; the network continues text and cannot follow instructions, so it is not read` |
| tools were offered | `N tool(s) offered (name, name); a reply that writes one comes back as a tool call` |
| `context`, not usable | `looking for q(ctx) in the graph: not there whole; dropping a word` |
| `candidates` | `looking for q(ctx) in the graph: found, N path(s) weighed` (sample: `N walk(s) drawn`); from START `starting a fresh text from the beginning: …` |
| a veto (from the guard, as it happens) | `the negative network vetoed q(text): <the verdict's why>` |
| `pick` | `every path repeats something already said; the best of them is kept as a last resort`, else `skipped N (empty, or already said)` for `N = skipped − vetoed > 0` |
| `rethink` | `caught itself saying q(noticed) twice` / `caught itself repeating q(noticed)`, then `; the words it picked up, not its own` (no steps), `; kept q(cut) and found another way on in N path(s)` (found) or `; kept q(cut), weighed N path(s), found nothing new`, plus ` (and learned to hand over there)` when it taught |
| `fresh` | `nothing new follows the line; changing the subject with a fresh text` |
| the turn | `saying q(text): cost C, probability P, reached the end of a text` / `cut at N units`, plus `; every path repeated something, so this is a repeat` |
| no turn | `nothing to say: the guard vetoed everything it could say` / `nothing to say: the graph has no way on from here` |
| a stop sequence | `stopped at the stop sequence q(seq)` |
| a call | `wrote a tool call: <tool>name {…}</tool>`, `wrote a call to NAME, which was not offered; it stays text`, `wrote a call to NAME, but no tools were offered; it stays text`, `wrote a tool call it could not finish (ERR); it stays text`, `wrote a call to NAME it could not finish (ERR); it stays text` |

### 37.4 The text, one node at a time (`deltas`)

`deltas(encoding, labels, node_ids, text)` cuts the reply into what each node of the walk added, so the pieces join back
into the text: the sentinels are dropped by id; every node after the first adds `piece(label, overlap)`; the first piece
is whatever precedes their join (the context picked up plus the located node's remainder). Under a word encoding the
later pieces carry their leading space. A text the walk does not line up with - one cut at the cap - is one piece. The
window `[start, end)` of the final text (a prefill's start, a stop or a call's end) is cut out of those pieces.

### 37.5 The events

`on_event` receives, per choice: `start` (`index`, and the reply's `id`, `model`, `created`, `input_units`, `thinking`,
so a streaming renderer can open the message); `thinking` deltas (`text`: the line, the second onwards led by a
newline, so the deltas join into the thinking); `text` deltas (one per piece of §37.4); `tool_use` (`id`, `name`,
`input`); `done` (`stop_reason`, `stop_sequence`, `output_units`, `thinking_units`, `turn`, `guard`). After the last
choice, `end` with the reply's usage. `stream(model, ask)` is the same as a generator on a worker thread.

### 37.6 The two documents and the two streams

* `to_openai(reply)`: `{"id": "chatcmpl-…", "object": "chat.completion", "created", "model", "choices": [{"index",
  "message": {"role": "assistant", "content" (null when there is only a call), "reasoning_content" (unless thinking is
  off), "tool_calls": [{"id": "call_…", "type": "function", "function": {"name", "arguments": JSON string}}]},
  "logprobs": null, "finish_reason"}], "usage": {"prompt_tokens", "completion_tokens", "total_tokens",
  "completion_tokens_details": {"reasoning_tokens"}}, "radixnet": {"kind", "units", "choices": [{"index",
  "stop_reason", "turn", "guard"}]}}`. `finish_reason` maps `end_turn` and `stop_sequence` to `stop`, `max_tokens`
  to `length`, `tool_use` to `tool_calls`, `refusal` to `content_filter`.
* `to_anthropic(reply)` (the first choice): `{"id": "msg_…", "type": "message", "role": "assistant", "model",
  "content": [{"type": "thinking", "thinking", "signature": ""}, {"type": "text", "text"}, {"type": "tool_use", "id":
  "toolu_…", "name", "input"}], "stop_reason", "stop_sequence", "usage": {"input_tokens", "output_tokens"},
  "radixnet"}`.
* `OpenAIStream.frames(event)`: `chat.completion.chunk` frames (`data: {…}\n\n`) - the first with `delta.role`,
  then `delta.reasoning_content`, `delta.content`, `delta.tool_calls`, the finishing chunk with `finish_reason` and a
  top-level `radixnet` (`stop_reason`, `turn`, `guard`), a usage chunk with `include_usage`, and `data: [DONE]`.
* `AnthropicStream.frames(event)`: `event: message_start` (the empty message with `input_tokens`), a `thinking`
  block (`content_block_start`, `thinking_delta`s, a `signature_delta`, `content_block_stop`), a `text` block
  (`text_delta`s), a `tool_use` block (one `input_json_delta`), `message_delta` (`stop_reason`, `stop_sequence`,
  `output_tokens`, `radixnet`) and `message_stop`. A reply with nothing to say still holds one empty text block.
  Only the first choice is streamed: a Messages reply is one message.
* A failure after the headers went out is one last frame: `{"error": {"message", "type": "server_error"}}` or
  `event: error` `{"type": "error", "error": {"type": "api_error", "message"}}`.

### 37.7 The routes, the errors, the CLI, the frontend

`/v1/*` paths go through the same route table as `/api/*` on all three servers. A streamed answer is written from
inside the search - the model lock is held for the whole stream - with `Content-Type: text/event-stream`,
`Cache-Control: no-cache`, `X-Accel-Buffering: no` and (Python, Rust) `Connection: close`, since no length is known in
advance. Errors take the dialect's envelope: OpenAI `{"error": {"message", "type": "invalid_request_error" |
"server_error", "param", "code": "model_not_found" for a 404 on `model`}}`, Anthropic `{"type": "error", "error":
{"type": "invalid_request_error" | "not_found_error" | "request_too_large" | "api_error", "message"}}` - for a bad
request, an unknown `/v1` route, a wrong method and a failure alike. The CORS allow-list gains the headers these
clients send (`Authorization`, `X-API-Key`, `anthropic-version`, `anthropic-beta`).

`model` names a kind: `ModelService.voice_for` answers the active model for `""`, `radixnet`, `default` or the active
kind's id, the parked model of another kind (`kind_of_id("radixnet-count") == "count"`; `word` is the count model
under a word encoding), and 404 with the ids in memory otherwise - the Go server, which runs the count model only, answers
the count model or 404. `GET /v1/models` lists the active model first, then every parked one (the negative network
included once it is in memory), each `{"id", "object": "model", "created" (the service's start), "owned_by":
"radixnet", "kind", "label", "encoding", "units", "active"}`.

CLI: `talk` (§11) - `--message` (repeatable: one conversation, each reply heard by the next), `--request FILE` (a
body in the dialect's shape, sent as it is), lines at the prompt otherwise; human mode prints `you: …`, the thinking
lines as `  · …` as they arrive, `model: ` with the text as it streams and `[stop; N units said, M thought]`; `--json`
prints the dialect's own document (a conversation: `{"format", "exchanges", "taught"}`); `--save` writes the model when a
rethink taught it. The same command in Go (`radixnet-count talk`) and Rust (`radixnet talk`).

Frontend: the Talk tab (`TalkPanel.jsx`, `src/sse.js`) streams `POST /v1/messages`, showing the thinking as a
collapsible list that fills line by line and the text as it arrives, newest first, with the turn's cost, probability,
context and rethink under it and the guard's report; the last card shows the `curl` for either dialect. The tab is shown
against the Rust server when `POST /v1/messages` is in `/api/status`'s `routes`.

Tests: `tests/test_assistant.py` (both requests, the reply, the thinking, the deltas, the guard, the tool calls, the
documents, the streams, the trace), `tests/test_api.py::TestTodaysFormat`, `tests/test_cli.py::TestTalk`,
`frontend/test/sse.test.mjs`, `go/radixnet/assistant_test.go`, `go/server/assistant_test.go`, the `assistant` tests
and `tests/server.rs` in Rust, and the parity suites `tests/test_rust_parity_assistant.py` and
`tests/test_go_parity.py::TestGoAssistantParity` (the same thinking, text, stop reason and units from the CLIs and the
servers; ids and timestamps aside, the turn's floats to 1e-9).

## 38. The attention band (`radixnet/attention.py`, `go/radixnet/attention.go`, `rust/src/attention.rs`) — where inside a gram a correction lands

`SPEC-AttentionBand.md` is the specification and D-086 the decision. A gram is read the way an eye reads a line:
sharp at its centre, blurred towards its first and last unit. The **band** is that, as `n` numbers - 1 at the
centre, `1 - blur` at both ends, linear between (`band_weights(n, blur)`; `n = 1` and `n = 2` are flat, being all
centre and all ends) - and it decides one thing: **which steps of a path answer for the units a correction
changed**.

### 38.1 The rule

`AttentionBand(blur)` - `None` for off, else a blur in `[0, 1]` - lives on the graph (`RadixCyclicGraph.attention`,
Go `Graph.Attention`, Rust `Graph::attention`) beside the encoding, but unlike it may change at any time: it
changes nothing the graph holds. `GraphModel._charged_steps(grams, length, spans)` returns `(prev, edge, charge,
focus)` per step of the traced text, in path order:

* **Off**: `_steps_over`'s steps, each `charge = 1`, each the focus - the writer rule of section 16.4.
* **On**: `spread(n, stride, grams, length, spans, weights)` first shares every marked unit out over the grams
  that see it (gram `g` covers `[g * stride, g * stride + n)` and sees unit `u` at weight `weights[u - g *
  stride]`): each takes `weight / total` of it - an even split when `total` is 0 - so every marked unit hands out
  exactly one charge, and a gram is the **focus** of a unit when no gram sees it more sharply. A step is then
  charged the sum of its node's grams' shares (`(label_len - n) / stride + 1` of them), capped at 1, and is the
  focus when one of them is. A unit no gram covers - the tail a grouping encoding drops - is charged to no one;
  the position after the last unit (`length`) is the END step's, in full, band or no band.

The marked units are every unit of every span, and an empty span - an insertion - marks the unit it stands in
front of (`judged_units`). The order is part of the definition - units ascending, the grams that see one
ascending, every sum left to right - so the three implementations write the same doubles.

Three properties the tests pin, over eight encodings and four blurs: a single marked unit's shares add up to 1; a
whole-text span leaves every gram at a full charge (so the cap makes the band inert on whole-text feedback); a unit
one gram sees is charged to it in full.

### 38.2 What it moves

* `CountRewardNet.correct`: an edge's charge is the largest any of its steps took. The penalty is `-strength *
  weight * charge`, one `add_reward` per distinct amount (first seen first, which is how Rust groups them too);
  the correction's own steps earn `strength * reward * factor` with `factor` 1 for a full charge, `charge + (1 -
  charge) * keep` for a partial one and `keep` for none; `mark_steps` gets only the focus steps. With the band off
  every charge is 1 and every number is what it was.
* `NegativeNet.correct`: `record_failure(edges, severity, reason, shares)` blames each edge `severity * share` and
  still counts one failure per edge; `blame_total` adds what was laid.
* Nothing else: `reward`, `punish`, 2NRL and training mark every unit of their texts alike.

### 38.3 The file, the CLI, the API, the frontend

* **File**: `"attention": {"blur": 0.5}` in the graph document, written only while the band is on, right after
  `encoding` (or `format_version`) - where Rust writes it too, byte for byte. A file without it is off.
  `stats()` reports `attention_blur` (null while off) for the count and negative kinds.
* **CLI**: `radixnet attention` shows the band; `--on`, `--blur X`, `--off` change it and save the model (not with
  `--dry-run`); `--wrong TEXT --right TEXT` prints where that correction lands, gram by gram, under both rules.
  `info` has an `attention` row. The same command in `radixnet-count` and in the Rust `radixnet`.
* **API**: `GET` / `POST /api/model/attention` and `POST /api/model/attention/preview` (section 12), on all
  three servers; `GET /api/model` carries `attention`. A kind that is never corrected - the sine and phase models -
  reports `applies: false` and refuses a band (400), as the CLI refuses it.
* **Frontend**: the Attention band card on Model settings (`AttentionBandCard.jsx`, section 31.4), with its
  display helpers in `src/attention.js` (`bandWeights` computes the same doubles as the servers, for drawing the
  band while the slider moves; where a correction lands is always the server's answer).

Tests: `tests/test_attention.py` (the band, `spread`'s three properties, the charged steps, both kinds'
corrections, off to the bit, the file, the refusals, the CLI and the API), `go/radixnet/attention_test.go`,
`go/server/attention_test.go`, the unit tests in `rust/src/attention.rs`, `frontend/test/attention.test.mjs`, and
the parity cases: `TestGoParity` / `TestGoWordParity` / `TestGoNegativeParity` in `test_go_parity.py` and
`TestRustCorrectionParity` / `TestRustAttentionRoutes` in `test_rust_parity_tools.py`, which hold both ports to
Python's previews, rewards, blame and - for Rust - model files byte for byte.

## 39. The phonetic units (`../PhoneticTokenizer`, `radixnet/encoding.py`, `go/radixnet/phonetic.go`, `rust/src/phonetic.rs`) — the same graph over an alphabet of sounds

Section 34's argument again, one alphabet over: not one structural rule of the graph mentions a character, so a
model over *sounds* is this model over a different alphabet. The alphabet is the phonetic tokenizer's - the ARPAbet
with stress, `#` between words, three pauses - and the unit is a **dial setting** (D-071's mechanism, D-080's
decision): `--encoding phone:3:1` or `syllable:2:1`, nothing new in the graph, the weights, the search or the file.

**How a text becomes units.** `Encoding.units(text)` for a phonetic unit is `split_words(tokenizer.text(text))`:
the tokenizer's *text form*, tokens joined by single spaces, cut by the word rule. The text form is idempotent on
any run of tokens - the tokenizer passes through a token that is already a sound, a `#` and a pause wherever they
are - so a label (a run of units) cuts into the units it was made of, a gram of `DH AH0 #` is three units, and the
round trip `decode(encode(t)) == normalize(t)` holds against the text of sounds. `join` writes single spaces and
turns any piece given as text into sounds first, so `full_text` is all sounds; `has_unit_prefix` matches a prefix
given as text by the sounds it makes; `spell` gives the words back (the lexicon, the tokenizer's memory of what it
read, a respelling for the rest).

**One tokenizer per unit, per process**, made on first use and kept (`phonetic_tokenizer`), so that what it
remembers of the words it sounded out can spell them back. Its lexicon is the *portable* one: the bundled core plus
the file `PHONETOK_LEXICON` names, which is what the Go and Rust ports read too. Both ports read the tokenizer's
ports beside them (a Go module by `replace`, a Rust crate by path; no third-party dependency), behind a mutex,
since training reads from many goroutines and threads.

**What the ports had to agree on.** The tokenizer's fixture (`../PhoneticTokenizer/tests/parity.json`) holds the
three ports to one token stream; this project's parity suites (`TestGoEncodingParity`, `test_the_other_dials_agree_too`)
then hold the three graphs built over it to one file and one prediction, and a Python model of sounds is read and
continued by Go and Rust. Every branch on the word unit that meant "not letters" - the alphabet listing, the
vocabulary count, the diff of a correction - now says `!= Chars`.

**The voice** (section 4.1, D-081): `radixnet speak` walks the model and speaks it as it goes; the graph's END
sentinel closes each utterance.

**Costs.** A text of sounds keeps no letters and no layout (the word unit's cost); an unread word is sounded out by
rules, which are a guess; a model means the sounds its lexicon gave it, so the same lexicon must serve training and
prediction. The frontend still names the units of a model of sounds *characters* - a gap for the next frontend build.

### The acoustic unit: learned from audio (D-082)

`Encoding(unit=ACOUSTIC)` is the same dial over the tokenizer's **acoustic
units** - a codebook learned from recordings by k-means over log-mel frames
(`../PhoneticTokenizer/DESIGN.md` §5), with nothing written down. Three things
differ from the phonetic units, and nothing else does:

* **What a text is.** A unit text is whitespace-separated tokens taken as they
  come (`q2 q28 q55`), so the acoustic unit behaves as the word unit does for
  every slice, join, prefix and length; `spell` returns the text itself, there
  being no words behind it. `validate` still imports the tokenizer, so
  `--encoding acoustic` fails early where the package is missing.
* **Where a text comes from.** `read_texts` hears a `.wav` (`is_audio_file`)
  through the codebook (`hear_audio`): one recording is one utterance, one
  text, runs of a unit collapsed. Everything downstream - the graph, the
  index, the file, the searches - sees a text like any other.
* **How it is heard back.** `Speaker` speaks acoustic units through the
  codebook's vocoder instead of the formant synthesizer, at the codebook's
  rate (`output_rate`), a unit's frames as the unit arrives, the END sentinel
  flushing the tail as for every other unit.

The codebook is the bundled one or the file `PHONETOK_CODEBOOK` names; it is
part of what a model means and is not written into the model file, which is
the cost noted in D-082.

## 40. The dynamic window (`radixnet/window.py`, `go/radixnet/window.go`, `rust/src/window.rs`) — a ladder of node sizes, halving from 32 to 4 and back up

`SPEC-DynamicWindow.md` is the specification and D-087 the decision. A merged node holds every gram of a chain
the corpus never branched inside, and a walk through it has nowhere to branch: it is entered at its first gram,
left at its last, and the steps inside are deterministic. The **dynamic window** is a ceiling on a node's length,
in the encoding's units, and the ceiling moves: it is sized in the binary number system, walking a ladder of
powers of two from a top down to a floor (32, 16, 8, 4 by default) and from the floor back to the top. It
applies to the actual nodes of the graph - a node with the value `ABCD` becomes `AB` and `CD`.

### 40.1 The setting

`DynamicWindow(top, floor, size, auto)` lives on the graph (`RadixCyclicGraph.dynamic_window`, Go
`Graph.DynamicWindow`, Rust `Graph::dynamic_window`) beside the encoding and the attention band, and like the band
it may change at any time: the setting touches nothing, only a step does. Off - `DynamicWindow()`, the zero value,
every model before this existed - it does nothing. `check_ladder` refuses a top or floor that is not a power of
two, a floor over the top and a size off the ladder, with one message per refusal shared by the three
implementations word for word. `GraphModel.window_config()` reports it (`on, top, floor, size, auto, sizes, next,
unit, units, ngram, longer, longest, nodes, heavy, default_top, default_floor`); `configure_window(on, top,
floor, size, auto)` switches it on at the values given over the ones it had - a new top or floor keeping the size
on the ladder - or off.

### 40.2 The step

`window_step(steps, compress)`: each step compresses the graph within the current size (`compress=False` leaves
that to a loop that has just done it), halves every node longer than the size (`split_window`, 40.3) and moves
the window down the ladder, back to the top from the floor; it returns `{steps, sizes, from, to, merges, splits,
nodes_before, nodes_after, edges_before, edges_after, window}`. `_window_epoch(compress)` is the automatic step
every kind's loop takes at the end of an epoch while `auto` is set - after the epoch's own compression in
`RadixNet.train`, `CountRewardNet._passes` and the negative network's blame passes (a clearing pass is not a
training pass), and with the merging in the phase model's passes, which compress once before the first - and the
epoch's record then carries `splits` and `window` (the size applied) right after `merges`; a record without a
step carries neither. Feedback passes are epochs and step like any other.

### 40.3 The halving, and the heavy connection

`split_window(size)` visits the real nodes in id order and cuts a node longer than `size` between its grams at
the middle - `split(node, ⌈G / 2⌉ · stride)`, `G` the grams it holds: the first half keeps the odd gram, the id
and the in-edges; the second half, appended and halved in its turn, takes the rest and the out-edges - again while
it is still longer; a node of one gram is never cut. Under a grouping encoding the cut is exact (`ABCD` into `AB`
and `CD`); under a sliding one the halves share the overlap (`ABCD` is `ABC -> BCD` merged, and comes apart into
those). Both halves carry the node's state, activation parameters, count and, in the count model, its judged
contexts (`split`). The bridge `A -> B` is the **heavy connection**, in each kind's currency: `split` already gives
it the node's whole visit count, which is what makes it heavy where weights are computed from counts; the sine
model sets its weight to `W_HEAVY = 8` (negated while inverted, as every fresh weight is), a score of `8 · f²`
between two copies of one activation; the phase model, which counts edges and not nodes, gives it what passed
through the node - the traversals of the out-edges it stands before, read before the halving (`_heavy_bridge`);
the negative network counts nothing, and its bridge carries no evidence. The first half has one child, so the
step across the bridge is probability 1 and cost 0: the cut is invisible to traversal the moment it is made.

### 40.4 Compression under the window

`merge_child` refuses a chain whose merged label would be longer than the window's size (`len(p) + len(c) -
overlap > size`), so a step's halves are held apart while the window stands at or below their size and grow
together again when the ladder is back at the top - keeping one side's activation and rescaling the other's edges,
as every merge does - or the moment the window is switched off. `check_invariants(compressed=True)` exempts a
chain the window holds apart. The graph is never left in a state the old rules could not have produced.

### 40.5 The file, the CLI, the API, the frontend

* **File**: `"dynamic_window": {"top": 32, "floor": 4, "size": 16, "auto": true}` in the graph document, only
  while it is on, right after `attention` (or `encoding`, or `format_version`) - where Rust writes it too, byte
  for byte. `stats()` reports `dynamic_window` (the size, null while off) for every kind.
* **CLI**: `radixnet window` shows it; `--on`, `--top`, `--floor`, `--size`, `--auto` / `--manual`, `--off` change
  it and save the model (not with `--dry-run`); `--step [N]` steps it and saves; `info` has a `dynamic window` row.
  The same command in `radixnet-count` and in the Rust `radixnet`.
* **API**: `GET` / `POST /api/model/window` and `POST /api/model/window/step` (section 12), on all three servers;
  `GET /api/model` carries `dynamic_window`.
* **Frontend**: the Dynamic window card on Model settings (`DynamicWindowCard.jsx`, section 13), with its display
  helpers in `src/window.js` (the same ladder the servers walk, drawn before it is applied).

Tests: `tests/test_window.py` (section 14), `go/radixnet/window_test.go`, `go/server/window_test.go`, the unit
tests in `rust/src/window.rs`, `frontend/test/window.test.mjs`, and the parity cases
`test_the_dynamic_window_halves_the_same_nodes` in `test_go_parity.py` and `TestRustWindowParity` /
`TestRustWindowRoutes` in `test_rust_parity_tools.py`, which hold both ports to Python's nodes, bridges, records
and - for Rust - model files byte for byte.
