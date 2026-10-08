# Integration — how a bare model becomes the product the kit serves

**Status** Describes what is built. RadixCyclicNN is the reference: the only model
in this repository that runs under every surface below (the command line, the
HTTP API, the React frontend, the OpenAI and Anthropic dialects, the teaching
loops, MCP, Docker), in three ports (Python, Go, Rust) held to one another byte
for byte. Everything around it is [ModelKit](ModelKit/README.md), broken out of
the model by D-093 so that other directories can use it. This document is the
map of that seam: what the kit *is*, what it *reads of a model*, and what a new
raw model therefore has to supply, in which order, before each surface lights
up. It takes its facts from the code and from
[RadixCyclicNN/DESIGN.md](RadixCyclicNN/DESIGN.md) and
[DECISIONS.md](RadixCyclicNN/DECISIONS.md), and points at them rather than
repeating them.

**Answers** *"Based on the RadixCyclicNN subdirectory, write an
integration/implementation spec document for the whole repo. This document
should contain the high-level ways that we implement and integrate the
frontend, backend, query spec (like popular providers), and more that would
need to be added to a new raw/bare model to make it functional."*

---

## 1. The problem, precisely

A model directory in this repository starts the same way: a package that can
read a corpus, predict and generate, and write a JSON file — RadixAcyclicNN,
RadixDecayNN and FilterBankRadix are at that stage today; GTMNN is specified
but has no code; CyclicCortex and GREN learn games rather than text. Each one
re-derives what it needs around that core: a `__main__`, a checkpoint directory
(FilterBankRadix, CyclicCortex, GREN and GTMNN each copy
`ModelKit/modelkit/checkpoint.py`'s contract by hand), a `Makefile`, and a
design note promising "an `api.py` the same shape as ModelKit's".

RadixCyclicNN shows what "functional" means here, and it is a long list: a
command line with some sixty commands; an HTTP API of about a hundred routes
with one-job-at-a-time semantics, uploads, checkpoints and two streaming
formats; a React frontend of twenty-odd tabs that mounts itself from what the
server reports; OpenAI Chat Completions and Anthropic Messages so any client,
agent framework or `agent.json` can talk to the model; teaching loops that put
an LLM, a sandbox, a browser, speech and images around it; an MCP server; a
Docker stack; and Go and Rust ports with parity suites. None of that is the
model. All of it is in ModelKit, and all of it is reached through one surface:
the model class.

So the question this document answers is: **what is that surface, exactly,
and what does a new model have to implement — or deliberately decline — for
each part of the kit to work?** The answer is layered, because the kit is:
some surfaces need almost nothing of the model, others reach into its graph.

## 2. The shape of the repository

```
ASI/
├── RadixCyclicNN/      the model: radixnet (Python), go/radixnet (Go), rust/ (crate radixnet)
│   ├── radixnet/       GraphModel and its kinds, encodings, searches, training, the file
│   ├── DESIGN.md       the normative specification     DECISIONS.md  why
│   ├── SPEC-*.md       one feature each, same template as this file
│   ├── agent.json      a chat descriptor pointing at the served /v1 API
│   ├── Dockerfile, docker-compose*.yml, docker/   the stack (builds from the repo root)
│   └── Makefile        every command; the kit targets reach into ../ModelKit
├── ModelKit/           everything around a model
│   ├── modelkit/       Python reference: cli, api, assistant, dialogue, tutor, agent, mcp, ...
│   ├── go/             module github.com/CurtisTechSolutions/ASI/ModelKit/go: kit/, server/, cmd/radixnet-count
│   ├── rust/           crate modelkit: binaries radixnet, radixnet-bench
│   ├── frontend/       the React + Vite single-page app every server serves (dist/ committed)
│   └── tests/          the kit's suite, run against the model beside it; Go and Rust parity
├── PhoneticTokenizer/  phonetok: sounds as units (phone, syllable, acoustic) — the one shared dependency
├── RadixAcyclicNN/     the same network kept as a tree (the control for Research/CyclesAreAFeature.md)
├── RadixDecayNN/       the tree with a decaying `seen` as its only memory
├── FilterBankRadix/    sine-unit filters routing to radix trees
├── CyclicCortex/, GREN/, GTMNN/   the game line (regions and SBNNs; identify a game; play it — spec only)
├── Experiments/        measurements; UniversalGameEncoding drives radixnet as a library
├── TwoNRL_Chess/, AudioImage/, DISTIL/   2NRL on chess; audio as images; an LLM agent (a *client* of /v1)
├── Research/           the papers       tools/  corpus downloaders       utils/
└── LICENSE             ASI Source-Available Non-Commercial License 1.1 (every package points at it)
```

Three rules hold the shape together:

1. **The kit depends on the model, never the reverse** (D-093). Go and Rust
   enforce it by module and crate boundaries; Python by
   `ModelKit/tests/test_layers.py`. The one sanctioned exception is the model's
   `python -m radixnet` entry point, which hands over to `modelkit.cli` (from
   the installed package, or the `../ModelKit` checkout beside it) so every
   command in the model's README keeps working.
2. **Standard library only, in every port** (D-012, D-072). `radixnet` and
   `modelkit` declare `dependencies = []`; torch is the model's optional `gpu`
   extra; the kit's media extras (`images`, `diffusion`, `speech`, `whisper`)
   are optional. The HTTP server is `http.server`; the Rust port carries its own
   JSON, gzip, BLAKE2b, MT19937 and HTTP; Rust reaches HTTPS through the system
   `curl` (D-076). The frontend has two dependencies, React 19 and Vite 7.
3. **One JSON contract, three servers, one frontend** (D-044, D-074). The Go
   and Rust servers speak the Python API exactly — fields, status codes, error
   text — and the frontend discovers which engine answered and which routes it
   serves. A port is finished when a client cannot tell which one answered.

A new model joins this shape by sitting beside RadixCyclicNN as a directory
with the same three-port layout where it has ports, and by being *loaded by*
the kit rather than by growing a kit of its own.

## 3. The layers

```
 clients:   browser (frontend/dist)   curl / SDKs on /v1   agent.json talk   MCP client (stdio)   CLI user
               │                          │                  │                  │                  │
 surfaces:  static + SPA fallback      /v1/chat/completions  /v1 again       mcp.py (tools/call)  cli.py
            /api/* (JSON, NDJSON)      /v1/messages          (agent_config)                        │
               └──────────────┬───────────────┴──────────────┘                  │                  │
 kit, model layer:      api.ModelService  ── assistant.respond ── dialogue.reply ── thinking ── voice ── gan ── tutor ── agent
                              │  one RLock, one job at a time, checkpoints, uploads, the guard (duo.NegativeFilter)
 kit, model-free layer: llm / ollama / chatgpt   tools + browser   speech / vision / media   critic / codegen / recall / chat / blame
                              │
 the model:             radixnet.model.GraphModel ─ kinds: radix | count | negative | resonant   (+ encoding dial: char | word | phone | syllable | acoustic)
                        model_classes() · load_model(path) · new_model(kind) · the model file (format → class)
                              │
 shared dependency:     phonetok (PhoneticTokenizer) — only when the encoding is a sound
```

The **model-free layer** (`llm`, `ollama`, `chatgpt`, `agent_config`, `tools`,
`browser`, `mcp`, `archive`, `speech`, `vision`, `media`, `critic`, `codegen`,
`recall`, `chat`, `blame`) imports nothing from the model and is *handed* one
when it needs one. The **model layer** (`api`, `cli`, `assistant`, `dialogue`,
`thinking`, `duo`, `gan`, `tutor`, `agent`, `voice`, `voicechat`,
`checkpoint`, `bench`) imports `radixnet` directly — thirteen modules do, so
there is no single seam in the kit; the seam is the model class itself (§4).
The three places a new model is *opened* are `cli.open_model`,
`api.ModelService` (which owns `self.model`) and `assistant.respond`, the only
way into `/v1`.

## 4. The model contract — what the kit reads of a model

There is no interface, protocol or trait on the kit side (D-093 rejected one:
the Go server alone touches 84 methods and 181 fields of the model, so an
interface that wide would be the model's API written twice). The contract is
the base class `radixnet.model.GraphModel`
(`RadixCyclicNN/radixnet/model.py`) and the functions beside it, plus a few
`hasattr` probes. It falls into levels; each surface in §6–§11 names the level
it needs.

### 4.1 Level 0 — identity, construction, the file

| what | where it is read |
|---|---|
| class attributes `kind` (the id in files, the API and the CLI), `label`, `description`, `units` (`"chars"` / `"words"`: what a *new* model counts in), `format` (the file's `format` string, unique per kind), `takes_corrections` (does it learn from a diff?) | `model_kinds()`, `/api/model`, `/v1/models`, the model selector, `--kind` |
| constructor `Cls(seed=0, backend="auto", device=None, encoding=None, **kind_options)` — kinds with no numeric rule accept `backend`/`device` and ignore them "for interface parity" | `new_model`, `/api/reset`, the discriminator in `gan.py` is `type(generator)(seed, backend, device, encoding)` |
| instance attributes `graph`, `encoder`, `decoder`, `backend` (with `.name`, `.device`), `history: list[dict]`, `meta: dict` (counters `epochs_total`, `trained_chars`, `trained_texts`, `twonrl_runs`, each with `*_resets`), `seed`, `replay` | status, info, checkpoints' step (`meta["epochs_total"]`), the Train tab's history chart |
| `stats() -> dict` with at least `kind, nodes, edges, trigrams, encoding, unit, units, compression_ratio, inverted, backend, device, epochs_total, history_len, last_loss` | `/api/status`, every CLI command's summary, the status bar |
| `to_dict()`, `from_dict(d, backend, device)`, `save(path)` (JSON, gzip when the path ends in `.gz`, atomic), `load(path, backend, device)`, `replay_summary()` | checkpoints, `/api/save`, `/api/load`, `--out`, `--resume` |
| `encoding` (an `Encoding`: `unit, n, stride, units_name, length(text), encode, join, piece, spell, reverse, describe, is_default`) | usage counting in `/v1`, `max_tokens` → units, spelled output for models of sounds, the Encoding card |

**Registration.** `model_classes()` in `model.py` is a closed dict —
`{radix, count, negative, resonant}` — and `model_from_dict` matches a file's
`format` against each class's `format`. A new kind is one more entry there, and
one more entry in the Rust `kinds.rs` `KINDS` table and the Go `Kinds()` list
(`ModelKit/go/server/service.go`) if it has those ports. There is no plug-in
discovery; the file declares its kind and the registry resolves it (D-020).

### 4.2 Level 1 — learn, continue, score

```python
train(texts, config: TrainConfig | None = None, *,
      checkpoint_manager=None, progress=None, stop_event=None, phase=None,
      origin=START, **overrides) -> list[dict]
```

* One **epoch record** per epoch, passed to `progress(record)` *and* returned:
  `{epoch, loss, perplexity, nodes, edges, trigrams, compression_ratio, merges,
  transitions, seconds, skipped_short, lr, act_lr[, phase][, early_stop]}`.
  The API's job stores each record and **yields the model lock to readers at
  that call** (D-017), so a model that never calls `progress` blocks every
  reader for the whole run.
* `stop_event` is a `threading.Event`, checked between epochs: the job's
  stop, and the CLI's first Ctrl-C, are cooperative and nothing else.
* `checkpoint_manager.save(self, epoch, "epoch", record)` every
  `config.checkpoint_every` epochs.
* `phase` labels the records (`"negative"`, `"positive"`); `origin` is `START`
  or `THINK` (thoughts are trained from the THINK sentinel).
* `TrainConfig` carries the dials every surface exposes: `epochs, lr, act_lr,
  batch_size, clip, auto_compress, shuffle, checkpoint_every, lr_schedule,
  act_lr_schedule, order, curriculum, replay, replay_size, patience,
  min_delta, reverse`. A kind that has no learning rate *accepts and ignores*
  `lr` (D-021); it does not reject it.

```python
predict(prefix, length=20, mode="dijkstra"|"beam"|"sample", step_penalty=0.0,
        temperature=1.0, to_end=False, max_length=None, k=5, beam=None,
        traversal="reward", penalty_scale=1.0, merit_scale=1.0,
        top_k=0, top_p=1.0, min_p=0.0, diversity=0.0) -> PathResult | Prediction
generate(max_length=60, mode="sample"|"beam"|"dijkstra", temperature=1.0, count=1,
         seed=None, prefix="", step_penalty=0.0, beam=None, traversal=..., ...) -> list[PathResult]
score(text) -> {"log_prob", "per_char", "chars", "transitions", "unknown_transitions", "units"}
compress() -> int
```

* `PathResult` (`radixnet/search.py`) is the currency of every answer:
  `text` (the continuation), `full_text`, `labels`, `node_ids`, `cost`
  (`-log P`), `step_costs`, `expanded`, `reached_end`. Beam mode returns a
  `Prediction` that adds `top`, `bottom`, `k`, `beam`, `mode`, `traversal`.
  `probability = exp(-cost)` everywhere (`beam.path_probability`).
* Lengths are in **encoding units** — characters under `char`, words under
  `word` — and `score()["units"]` says which, because a per-word number read
  as per-character is read wrong.
* Determinism: `generate(mode="sample", seed=…)` uses a private RNG; without a
  seed it consumes the model's own seeded Mersenne Twister, whose full state is
  saved in the file (§5). Dijkstra, beam and `score` are deterministic.

Level 1 is enough for: `train`, `predict`, `generate`, `score`, `info`, `bench`
on the CLI; `/api/train`, `/api/predict`, `/api/generate`, `/api/score`,
`/api/history`, `/api/compress`; the Train, Predict, Generate and Score tabs;
checkpoints; MCP's `radixnet_predict` / `generate` / `score` / `stats`; and the
Evolve discriminator's ranking.

### 4.3 Level 2 — feedback

```python
two_nrl(bad, good, neg_epochs=3, pos_epochs=3, neg_lr=0.05, pos_lr=0.01, *,
        progress=None, stop_event=None, checkpoint_manager=None, strength=1.0,
        bad_weights=None, good_weights=None, **overrides) -> dict   # with "positive": records
reward(texts, *, epochs=3, lr=0.1, weights=None, progress=None, stop_event=None, strength=1.0, **overrides) -> list[dict]
punish(texts, *, epochs=2, lr=0.5, weights=None, ...) -> list[dict]
invert() -> None
invert_paths(texts, mode="activation"|"state", amounts=None) -> {"flipped", "amount_mean", ...}
correct(wrong, right, *, strength=1.0, weight=1.0, reward=..., keep=...)   # optional: takes_corrections kinds
```

Every teaching loop ends in one of these: thumbs in the Predict and Generate
tabs (`/api/feedback` picks `two_nrl`, `reward` or `punish` by which texts it
was given), the 2NRL tab, Evolve, the tutor, the code generator, the agent,
the LLM chat partner. A kind without a learning rate uses `strength`
(one unit multiplies an edge's odds by *e*); a kind with one uses `neg_lr` /
`pos_lr`. The tutor prefers `correct` when `callable(getattr(model,
"correct", None))`, else falls back to the other three. A model that lacks
Level 2 should raise a `ValueError` naming the kind (the API turns it into a
400, the CLI into a message) rather than silently do nothing — the pattern the
count model uses for "does not count paths".

### 4.4 Level 3 — the walk the conversation needs

`/v1`, Converse, Think, Chat, Voice and the guard do not call `predict` on a
prompt. They call `dialogue.reply`, which locates the **tail of the last
line** in the structure and continues it, backtracks on repeats and teaches
what it learns (D-068, D-081, D-085). To do that the kit reaches past the
public verbs:

| the kit calls | from |
|---|---|
| `model._search(prefix, length, mode, k, beam, step_penalty, temperature, to_end, max_length, rng=, origin=START\|THINK, ...)` | `dialogue.reply` (sample mode), `thinking` |
| `model._prefix_start(prefix) -> (node, offset, lead)`, `locate(prefix)` | `dialogue`, `thinking`, `voice` |
| `model.graph.labels / alive / children[node] / child_costs(p) / split(node, i) / observe_back(node, went=, instead=, amount=) / observe_think(node, amount) / thinks_at(node) / trigram_index / text_of / symbols_of / lookup / node_path / edge_parent / path_totals() / num_nodes() / num_edges() / compression_ratio() / inverted` | `dialogue`, `thinking`, `/api/graph`, `/api/nodes`, `/api/encoding/preview`, `voice` |
| the sentinels `START, END, BACK, THINK, FIRST` (`radixnet/graph.py`) | everywhere a walk starts or stops |
| `search.sample_walk`, `penalty.traversal_costs` over the graph | `voice` (a walk heard as it goes) |

This is the level at which the kit is **graph-shaped**: it assumes a node /
edge structure with sentinels, `-log P` costs per child, and the ability to
teach a `BACK` or `THINK` edge at a node. A model that is not such a graph can
still serve `/v1` (§7.4), but only through a shim that answers these calls —
or by giving `dialogue.reply` a model-level `reply(prefix, ...)` to prefer,
which is not built today.

### 4.5 Level 4 — kind-specific surfaces, probed with `hasattr`

| optional method | surface that appears when it exists |
|---|---|
| `weight_config()`, `configure_weights(**scales)` | `/api/model/weights`, the score-function card, `weights` CLI |
| `paths(limit, node)`, `node_ratios(limit, node)` | `/api/paths`, `/api/nodes`, `paths` / `nodes` CLI (400 "does not count paths" otherwise) |
| `attention_config()`, `configure_attention(on=, blur=)`, `attention_preview(wrong, right, blur)` — `takes_corrections` kinds | `/api/model/attention*`, the Attention band card ([SPEC-AttentionBand](RadixCyclicNN/SPEC-AttentionBand.md)) |
| `window_config()`, `configure_window(...)`, `window_step(steps)` | `/api/model/window*`, the Dynamic window card ([SPEC-DynamicWindow](RadixCyclicNN/SPEC-DynamicWindow.md)) |
| the negative kind: `judge(text)`, `blame(texts, reason=, source=, note=, severity=)`, `clear`, `forget(reason, factor)`, `reasons()`, `recent(limit)`, `threshold`, `min_coverage` | `/api/negative/*`, the Negative tab, the **guard** on every answer path (`duo.NegativeFilter`), `radixnet_judge` in MCP |
| `graph.vocabulary` under a word encoding | `/api/words`, the Words tab (shown only when `status.units != "chars"`) |

The convention is uniform: a route for a capability a kind lacks answers **400
with a sentence naming the kind**, never 404 (404 is for a route a *server*
lacks, §6.4). The frontend shows the 400 in the tab rather than hiding the tab.

### 4.6 Two ways in

**(A) A new kind of the same graph** — the path RadixCyclicNN's own four kinds
took, and the one GTMNN's design assumes. Subclass `GraphModel`; keep its
`graph`, `encoder`, `decoder`, `_search`, `generate`, `score`, `save` / `load`;
implement Levels 1–2 (`train`, `predict`, `two_nrl`, `reward`, `punish`,
`invert`, `invert_paths`, `stats`); choose a `kind`, `label`, `description`,
`units`, a unique `format`; register in `model_classes()`. Every surface works
on day one, because the graph the kit reaches into is the same graph.

**(B) A model of another shape** (RadixAcyclicNN's `RadixTreeNet`,
RadixDecayNN's `DecayNet`, FilterBankRadix's bank, a future GTMNN). Implement
Levels 0–2 against the signatures above, in the model's own package. Then
either (i) expose a `graph` with the Level 3 surface — a tree *is* such a
graph, so `RadixTreeNet` is close — or (ii) accept that Converse, Think,
Chat, Voice, Graph, `/v1` and the guard stay off until a `reply`-level seam
exists in `dialogue.py`, and say so in the model's README the way
`rust/README.md` names its deliberate gaps. Either way the model is loaded by
the kit through `model_classes()` and its `format`, and its directory grows
no `api.py`, `cli.py`, `checkpoint.py` or frontend of its own — those are the
copies this document exists to end.

## 5. The model file and the checkpoints

The file is the contract between ports and between runs, so it is specified
more tightly than any API (D-015, D-039, D-071).

```jsonc
{ "format": "radixnet" | "radixnet-count" | "radixnet-negative" | "radixnet-resonant",   // picks the class
  "version": 1,                        // MODEL_FORMAT_VERSION; a reader rejects a newer one
  "saved_at": "2026-09-20T04:19:16+00:00",
  "kind": "count",                     // redundant with format, for humans
  "meta": { "created", "seed", "epochs_total", "epochs_total_resets", "trained_chars", ... },
  "history": [ /* epoch records */ ],
  "backend": { "name", "device" },     // kinds that have one
  "graph": {
    "format": "radixnet-graph", "format_version": 4,
    "encoding": {"unit","n","stride"},        // only when not char:3:1
    "attention": {...}, "dynamic_window": {...},   // only while on
    "seed", "inverted", "version", "structure_version", "traversals",   // each with *_resets
    "nodes": { "labels", "z", "a", "b", "h", "k", "count" },            // struct of arrays, sentinels first
    "edges": { "src", "dst", "w", "count" /* + reward | blame,fails,clear,reasons | cx,cy,cw */ },
    "rng_state": [3, [624 words, index], null],                          // CPython random.Random state
    "weights": { "function": "dual-frequency" | "blame", ...scales, "kind" },   // the score function; absent for radix
    "paths": { "prev", "edge", "seen", "correct", "incorrect" }          // count graphs
  },
  "replay": { "size", "seen", "index", "texts" },   // only when a replay buffer exists
  "log": [...], "filter": {...},                    // negative only
  "metacog": {...}, "cycles": {...}                 // resonant only
}
```

Rules a new model's file must keep:

* **`format` is the type.** `load_model(path)`, `POST /api/load`, the
  checkpoint restore and `--model` all dispatch on it; nothing is told out of
  band. Pick a new string per kind.
* **JSON, UTF-8, compact separators, `ensure_ascii=False`, floats as Python
  `repr`, insertion-ordered keys; gzip when the path ends in `.gz`; readers
  sniff the `1f 8b` magic regardless of extension; writes are atomic** (temp
  file, fsync, rename). The Rust port writes the same bytes; the parity suites
  compare documents byte for byte with only `version` and clock fields excused.
* **The RNG state travels in the file**, so a run resumed in another port
  continues the same sequence. Every growing integer is a cyclic counter that
  wraps at 10^15 with a `*_resets` twin (D-064).
* **Edge order is first-seen order and is the softmax summation order** — a
  port that stores edges differently will not reproduce costs to the last bit.
* Older graph `format_version`s are upgraded on load (sentinels inserted); a
  `version` above 1 is refused.

**Checkpoints** (`ModelKit/modelkit/checkpoint.py`, mirrored in
`go/server/checkpoints.go` and `rust/src/checkpoint.rs`): a checkpoint is an
ordinary model file written by `model.save` as
`ckpt-<tag>-<step:06d>.json.gz` in a directory beside `index.json` (name →
record) and `latest.json` (the newest record), each record
`{name, path, step, tag, metrics, saved_at, bytes}`. Tags in use: `epoch`
(train), `2nrl`, `gen` (evolve), `manual` (the API and the Checkpoints tab).
`keep` (default 5) prunes the oldest by `(step, saved_at, name)` and never the
one `latest` points to; the index is reconciled with the directory on every
listing. `--resume` and `RADIXNET_RESUME=1` restore `latest`. A model gets
all of this for free once Level 0 holds: the manager only calls `save` and
`load_model`.

## 6. The HTTP contract

`python -m radixnet serve --host 127.0.0.1 --port 8000` — `modelkit/api.py`,
a stdlib `ThreadingHTTPServer` (D-016). The Go server (`radixnet-count
serve`, port 8001) and the Rust server (`radixnet serve`, port 8000) answer
the same routes with the same bodies. The route table is `_ENDPOINTS` in
`api.py`; `GET /api` on the Go server lists it; the Rust server lists what it
serves in `/api/status` as `routes`.

### 6.1 Mechanics every route shares

* `/api/*` and `/v1/*` are dispatched; everything else is a static file from
  `--frontend-dir` (default `frontend/dist`) with an SPA fallback to
  `index.html`, or a help page when there is no build.
* JSON bodies must be objects, capped at 64 MiB; chunked request bodies get
  411. Six routes take binary or multipart bodies instead (`/api/uploads`,
  `/api/images/encode`, `/api/speech/transcribe`, `/api/speech/teach`,
  `/api/voice/turn`, `/api/voice/turn/stream`); uploads are streamed with no
  cap, and a ZIP is kept whole and unpacked in memory on demand (D-034, D-035).
* **Status codes:** 200 sync; **202** a job accepted; 400 bad input (any
  `ValueError` / `TypeError` / `OSError` from the model becomes one, with its
  message); 404 unknown route or file; 405 with `Allow`; **409** a job is
  running; 500 anything else.
* **Errors** on `/api/*` are `{"error": "message"}`. The frontend treats an
  `error` key as a failure even on a 2xx. Under `/v1` each dialect's own
  envelope is used (§7.3).
* **CORS** on every response: `Access-Control-Allow-Origin: *`, methods
  `GET, POST, OPTIONS`, headers `Content-Type, Accept, Authorization,
  X-API-Key, anthropic-version, anthropic-beta`; `OPTIONS` answers 204. The
  frontend can therefore be built once (`VITE_API_BASE`) and pointed at any
  server, and `agent.json` can be sent from a browser.
* **No authentication.** Keys a client sends are ignored; the server's own
  LLM keys come from its environment and are never accepted as request fields
  (D-059).

### 6.2 Jobs, the lock, cancellation (D-017)

`ModelService` holds one `RLock`. Readers take it per request. A long
operation — `train`, `2nrl`, `feedback`, `evolve`, `codegen`, `agent`,
`explore`, `tutor`, `critic`, `chat` — is a **job**: one at a time, started
with 202 `{"job": {...}}`, run on a daemon thread that holds the lock and
**releases it to queued readers at every `progress` record** for up to one
second, and around every LLM or sandbox call. While it runs, mutating requests
answer 409 (`"a train job (train-3) is running; wait for it to finish or POST
/api/job/stop"`); readers (`/api/status`, `/api/predict`, …) still answer.

```jsonc
GET /api/job  →  { "id": "train-3", "type": "train", "state": "running" | "done" | "stopped" | "error",
                   "progress": {…last record…}, "history": [records], "error": null,
                   "started_at", "finished_at", "stop_requested": false }
POST /api/job/stop   →  sets the job's stop_event; the model stops between epochs
```

The frontend's `useJob` hook polls `/api/job` every second while `state ==
"running"`, adopts a running job of its type on mount, and keeps polling after
a stop until a terminal state. The status bar summarises `progress` by its
shape (`epoch/loss/phase`, `generation/gap`, `kind: attempt|problem|round`).
A new model's `train` and feedback methods have to honour `progress` and
`stop_event` for any of this to be true.

### 6.3 Streaming — two formats, both over POST

* **NDJSON** (`application/x-ndjson`, chunked): one JSON object per line, each
  with `event`; the last line is `{"event": "done", ...}` carrying the same
  document the non-streaming route returns; a failure after the first line is
  a last line `{"event": "error", "error"}`. Used by `/api/converse/stream`
  (events `look, draft, caught, backtrack, found, stuck, turn` — D-081) and
  `/api/voice/turn/stream` (`heard, trained, reply, audio, spoken, taught`).
* **SSE** (`text/event-stream`, `Connection: close`, `X-Accel-Buffering:
  no`): the `/v1` dialects (§7.3).

The frontend reads both with `fetch` + `ReadableStream` (`EventSource`
cannot POST); a server that does not stream a route answers 404 and the tab
falls back to the plain route.

### 6.4 Engine and capability discovery

`GET /api/health` → `{"ok": true, "version"}` (Go adds `"engine": "go"`,
`workers`, `counting`; Rust answers `{"status": "ok", "version", "engine":
"rust"}`). `GET /api/status` → `stats()` plus `kind, model_label, units,
kinds, job, backends, model_path, checkpoint_dir, upload_dir, ollama,
chatgpt, tools, replay` and, on Rust, `routes`. The frontend reads `engine`
from status then health (default `"python"`); on `"go"` it hides tabs marked
Python-only, on `"rust"` it shows a non-model tab only when its route is in
`routes`. A route a server does not implement answers **404 with a sentence
naming the server that does** (D-044) — a discoverable gap, not a hidden
button.

### 6.5 The routes, by the level of the model they need

| level | routes |
|---|---|
| 0 | `GET /api/health`, `/api/status`, `/api/model`, `POST /api/model/select {kind}`, `/api/save {path}`, `/api/load {path}`, `/api/reset {seed, kind, encoding}`, `GET /api/encoding`, `/api/history`, `/api/checkpoints`, `POST /api/checkpoints/save {tag}`, `/api/checkpoints/restore {name}`, `GET/POST /api/uploads`, `/api/uploads/delete`, `GET /api/job`, `POST /api/job/stop`, `GET /api/schedule`, `POST /api/schedule/preview` |
| 1 | `POST /api/train` (202), `/api/predict`, `/api/generate`, `/api/score {text}`, `/api/compress`, `/api/encoding/preview {text}`, `GET /v1/models`, `POST /v1/messages/count_tokens` |
| 2 | `POST /api/2nrl` (202), `/api/feedback` (202), `/api/invert`, `/api/evolve/start` (202) `/stop` `/history`, `/api/tutor/*`, `/api/codegen/*`, `/api/agent/*`, `/api/chat/*`, `/api/ollama/corpus|review|correct|think`, `/api/images/tutor`, `/api/speech/teach|tutor`, `/api/negative/auto` |
| 3 | `POST /v1/chat/completions`, `/v1/messages`, `/api/converse`, `/api/converse/stream`, `/api/think`, `/api/voice/turn`, `/api/voice/turn/stream`, `/api/say`, `GET /api/graph`, `/api/nodes` |
| 4 | `/api/model/weights`, `/api/model/attention*`, `/api/model/window*`, `/api/paths`, `/api/words`, `/api/negative/*` |
| model-free | `GET /api/tools`, `POST /api/tools/call`, `GET /api/ollama/models`, `/api/chatgpt/models`, `GET /api/images`, `POST /api/images/encode|decode`, `GET /api/speech`, `POST /api/speech/transcribe|decode`, `GET /api/tutor`, `GET /api/voice` |

Field names are documented route by route in
[RadixCyclicNN/README.md § HTTP API](RadixCyclicNN/README.md) and in
`_ENDPOINTS`; the request body of every search route takes the same dials
(`mode, k, beam, step_penalty, temperature, traversal, penalty_scale,
merit_scale, top_k, top_p, min_p, diversity, guard`), which the frontend
fills from its stored Settings. A text may always be given three ways —
`texts`, `text`, or `files` (upload names) — and a server that reads only one
is not at parity (D-074).

## 7. The query spec — today's format, two dialects (D-085)

The model is talked to the way every language model is now: messages in, an
assistant message out, streamed as it is written. `modelkit/assistant.py`
gives it both dialects any client speaks; `go/kit/assistant.go` and
`rust/src/assistant.rs` serve the same bytes.

### 7.1 Model ids

`GET /v1/models` lists the models **in memory** as `radixnet-<kind>`
(`{"id", "object": "model", "created", "owned_by": "radixnet", "kind",
"label", "encoding", "units", "active"}`), the active one first. A request's
`model` may be `radixnet-<kind>` or `<kind>` (a parked kind answers), or any of
`""`, `radixnet`, `radixcyclicnn`, `default`, `active` for the active model;
an unknown id is a 404 `model_not_found` with `param: "model"`, checked before
any stream frame is sent. A new kind appears here by its `kind` alone.

### 7.2 Requests

**`POST /v1/chat/completions`** reads `model`, `messages` (`system` /
`developer` join the system text; `assistant.tool_calls` and `tool` results
are rendered into the transcript as `<tool>name {json}</tool>` /
`<result>…</result>`, the shape the network was trained on), `max_tokens` or
`max_completion_tokens` (default 60), `temperature`, `stop` (string or list),
`n` (1–8), `stream`, `stream_options.include_usage`, `tools`, `tool_choice:
"none"`, `thinking` (bool), `reasoning_effort: "none"`, and the dialogue's own
dials by their `/api/converse` names (`mode, context, k, beam, step_penalty,
explore, avoid_repeats, avoid_word_repeats, learn, guard, seed`). Unknown
fields (`top_p`, `response_format`, `logprobs`, …) are accepted and ignored;
image, audio and document parts are a 400.

**`POST /v1/messages`** reads `model`, `system` (string or blocks),
`messages` (user / assistant; `text`, `tool_use`, `tool_result` blocks;
`thinking` blocks skipped), `max_tokens`, `temperature`, `stop_sequences`,
`stream`, `tools` (`input_schema`), `tool_choice: {type: "none"}`, `thinking`
(`{type: enabled|disabled|adaptive}` or a bool), and the same dials.

**`POST /v1/messages/count_tokens`** → `{"input_tokens"}`: the encoding
units the system text and messages hold.

### 7.3 Replies, streams, errors

| | OpenAI dialect | Anthropic dialect |
|---|---|---|
| whole reply | `chat.completion`: `choices[].message.{content, reasoning_content, tool_calls}`, `finish_reason` `stop\|length\|tool_calls\|content_filter`, `usage.{prompt_tokens, completion_tokens, total_tokens, completion_tokens_details.reasoning_tokens}` | `message`: `content` of `thinking`, `text`, `tool_use` blocks, `stop_reason` `end_turn\|max_tokens\|stop_sequence\|tool_use\|refusal`, `usage.{input_tokens, output_tokens}` |
| stream | `chat.completion.chunk` events: `delta.reasoning_content` line by line as the search runs, `delta.content` **one node of the walk at a time**, `delta.tool_calls`, a finishing chunk with `finish_reason`, a usage chunk when asked, then `data: [DONE]` | `message_start`, `content_block_start` / `content_block_delta` (`thinking_delta`, `text_delta`, `input_json_delta`) / `content_block_stop`, `message_delta` (stop reason, output units), `message_stop` |
| errors | `{"error": {"message", "type": "invalid_request_error"\|"server_error", "param", "code"}}` | `{"type": "error", "error": {"type": "invalid_request_error"\|"not_found_error"\|"request_too_large"\|"api_error", "message"}}` |
| extension | a `radixnet` object on the reply and the finishing chunk: `{kind, units, choices: [{index, stop_reason, turn, guard}]}` | the same `radixnet` on `message_delta` |

A **token is one unit of the model's encoding** — a character of a character
model, a word of a word model — and `max_tokens` caps that many; usage is
counted the same way and includes the thinking units. The **thinking** is the
search's own trace, line by line (which tail it looked for, how many paths it
weighed, what the guard vetoed, where it caught itself repeating and backed
out, what it said at what cost); it is never prose the model did not produce.
A **system prompt is accepted and not read**, and the first thinking line says
so. A `<tool>name {...}</tool>` line the model writes becomes a `tool_call`
when the request offered that tool. **`learn` is on by default**: a
conversation changes the model (D-068).

### 7.4 What this asks of a new model

`assistant.respond` is `dialogue.reply` in a costume. It continues the
**last** message only (earlier ones count as *heard*, so the reply does not
echo them; a trailing assistant message is a prefill and only the addition
comes back), locates the last `context` units in the structure, asks
`predict(mode="beam", length=0, to_end=True, k, max_length=max_tokens)` or
`_search(... "sample" ...)`, shortens the context word by word until something
continues, falls back to a fresh text from `START`, applies `stop` to the
produced text, cuts the stream per node of the walk, and asks the negative
network to veto. So the dialects need Level 3; a Level 1 model gets them only
through a shim (§4.6 B). Nothing here is a prompt template, a chat markup or a
tokenizer — a new model brings none of those; it brings a structure that can
be entered at a prefix and walked to an end.

### 7.5 `agent.json` — the descriptor clients use to find the model

```json
{ "id": "radixcyclicnn-v1", "type": "chat", "base_url": "http://127.0.0.1:8000/v1",
  "model": "radixcyclicnn", "timeout": 60, "max_tokens": 128, "temperature": 0, "structured": true }
```

`id`, `type` (must be `"chat"`), `base_url` (http(s), no credentials, query or
fragment; a bare host gets `/v1`) and `model` are required; the rest default
as shown. `modelkit.agent_config` (model-free) validates it and POSTs to
`<base_url>/chat/completions` with the file's `model`, `max_tokens`,
`temperature`; `structured: true` adds `response_format: {"type":
"json_object"}`, **which the server accepts without enforcing** — the
descriptor does not turn the network into an instruction follower. Consumers:
`radixnet talk --agent-config FILE` and the Talk tab's "Load agent.json"
(sent from the browser, kept for the page session, never uploaded). A new
model needs no descriptor of its own unless it is served elsewhere; the one
file names the server, not the kind.

**What is deliberately absent:** `/v1/completions`, `/v1/embeddings`, and an
Ollama-*server* surface (`/api/tags`, `/api/chat`, `/api/show`). The kit is an
Ollama **client** (§10); its own `POST /api/generate` is not Ollama's and
shares only the path. DISTIL, this repository's LLM agent, can be pointed at a
served model with `OPENAI_BASE_URL=http://127.0.0.1:8000/v1` or
`ANTHROPIC_BASE_URL`, which is what the dialects are for.

## 8. The frontend

`ModelKit/frontend/`: React 19 + Vite 7, no TypeScript, `dist/` committed and
served by every port (D-018). `npm run dev` proxies `/api` and `/v1` to
`VITE_PROXY_TARGET` (default `http://127.0.0.1:8000`); `VITE_API_BASE` bakes
another API host into a build; `node --test` runs the pure modules without
`npm install`. The panels **only ever speak the JSON contract** — no panel
imports anything from a model — which is why one frontend serves three
engines unchanged.

### 8.1 How it mounts itself

1. `GET /api/health` once; `GET /api/status` every two seconds (the status
   bar, the job, the engine badge).
2. `TABS` in `src/App.jsx` is a list of `{id, label, Component, model?,
   route?, wordOnly?, single?}`. A tab is shown unless the engine lacks its
   `route` (Rust: `status.routes`; Go: `pythonOnly`) or it is `wordOnly` and
   `status.units == "chars"`. All panels stay mounted; inactive ones are
   hidden, so a running job keeps its panel's state.
3. The **model selector** is a `<select>` over `status.kinds` → `POST
   /api/model/select {kind}` → `{kind, label, model_path, origin:
   memory|file|new|active, stats}`. It chooses among kinds *inside one
   server*; there is no server picker (the Talk tab's `agent.json` is the one
   runtime redirection).
4. Settings live in `localStorage` under `radixnet.v1.*`: **network settings**
   (`traversal`, `penaltyScale`, `meritScale`) and **site settings**
   (`search.{topK,topP,minP,diversity}`, `train.{order,curriculum,replay,
   replaySize,patience,minDelta}`, `query.backwards`) are folded into every
   request body by `traversalBody`, `searchBody`, `trainingBody`; each panel's
   own fields use `useStoredState("panel.field")`. A footer link clears them.

### 8.2 The tabs and the level each needs

| tab | routes | level |
|---|---|---|
| Train | `/api/train`, `/api/history`, `/api/schedule/preview`, `/api/uploads*` | 1 |
| Predict, Generate, Score | `/api/predict`, `/api/generate`, `/api/score`, `/api/feedback` (thumbs) | 1 (+2 for thumbs) |
| Checkpoints | `/api/checkpoints*`, `/api/save`, `/api/load`, `/api/reset`, `/api/compress` | 0 |
| Model settings | `/api/model`, `/api/reset`, `/api/encoding/preview`, `/api/model/weights`, `/api/model/attention*`, `/api/model/window*` | 0 / 4 |
| Settings | none (browser only) | — |
| 2NRL | `/api/2nrl`, `/api/invert` | 2 |
| Evolve, Tutor, Code, Agent, Chat, Ollama, Images, Speech | the loops' `/start`, `/history`, one-shot routes | 2 + external services |
| Talk | `POST /v1/messages` streamed; `agent.json` → an external `/chat/completions` | 3 |
| Converse, Think, Voice | `/api/converse(/stream)`, `/api/think`, `/api/voice/turn(/stream)`, `/api/say` | 3 |
| Graph | `/api/graph`, `/api/nodes` (SVG of the top nodes and their edges) | 3 |
| Words | `/api/words` (word encoding only) | 4 |
| Negative | `/api/negative/*` | 4 (negative kind) |

The status bar reads from `stats()`: `nodes, edges, trigrams,
compression_ratio, inverted, backend, device, epochs_total, last_loss,
replay`, and kind-specific extras (`edge_reward_*`, `total_traversals`,
`window`, `coherence_mean`, `buckets`, `cycles_seen`). A new kind that fills
Level 0 gets the shell, the selector, Checkpoints and Model settings; Level 1
gets Train / Predict / Generate / Score; a **new card or tab** is a component
under `src/components/`, a line in `TABS` with its `route`, and the
corresponding functions in `src/api.js` — the Attention band and Dynamic
window cards are the template (each SPEC's "surfaces" section says what it
added). The title is hard-coded `RadixCyclicNN`; a repo-wide kit would read
`status.model_label` instead.

## 9. The command line

`python -m radixnet` / `python -m modelkit` / `modelkit` → `modelkit.cli`
(`PROG = "radixnet"`); the Go `radixnet-count` and Rust `radixnet` binaries
carry the same commands with byte-identical help. Global flags, before or
after the command: `--model PATH` (default `model.json`; per-kind defaults
`model.count.json`, `model.negative.json`, `model.resonant.json`), `--kind`,
`--encoding unit:n:stride` (or `--units/--ngram/--stride`), `--backend
auto|python|torch`, `--device`, `--seed`, `--json`. `open_model` loads the
checkpoint named by `--resume`, else the file, else makes a new model of
`--kind`; a loaded file's own kind and encoding override the flags.

Commands, grouped: **learn** `train`, `2nrl`, `feedback`, `correct`,
`invert`, `compress`, `evolve`; **read** `predict`, `generate`, `score`,
`converse`, `think`, `talk`, `chat`, `speak`, `say`; **inspect** `info`,
`weights`, `paths`, `nodes`, `words`, `attention`, `window`, `schedule`,
`checkpoints`, `bench`; **negative** `negative blame|clear|why|filter|auto|
reasons|forget`; **teachers and tools** `ollama …`, `chatgpt …`, `tutor`,
`codegen`, `tools …`, `agent`, `explore`, `image …`, `speech …`; **serve**
`serve`, `mcp`. `--json` prints the same documents the API returns, which is
what the parity suites diff. Long runs go through `run_interruptible`: the
first Ctrl-C sets the stop event and the run finishes its epoch and saves; the
second aborts. The Docker services therefore stop on `SIGINT`, not `SIGTERM`
(D-037).

A new model adds no command: its kind appears in `--kind`'s choices through
`model_kinds()`, and a kind-specific command follows the `paths` / `nodes`
pattern (probe with `hasattr`, refuse with a sentence).

## 10. The teaching loops and the services around them

Every loop is a function of the kit that is **handed** a model and calls
Level 1–2 on it; none is a method of the model (D-093). What each needs:

| loop | module | asks of the model | external |
|---|---|---|---|
| Evolve (GAN-style self-upgrade) | `gan.py` | `generate(sample)`, a discriminator `type(model)(...)` trained by `two_nrl` and ranking by `score()["per_char"]`, `two_nrl` / `invert_paths` / `reward`; checkpoints tagged `gen` | none |
| Tutor (English lessons) | `tutor.py` | `predict(cue, …)`, then `correct` if present else `two_nrl` / `reward` / `punish`; optional `train(origin=THINK)` | Ollama or ChatGPT |
| Code generation | `codegen.py` | `predict`, then feedback | Ollama (`gemma4` default) or ChatGPT; a sandbox (`python -I -B`, rlimits, `unshare` without network where available) |
| Agent / explore (tool use) | `agent.py`, `tools.py`, `browser.py` | `predict(mode="beam")` emitting `<tool>…</tool>` text; `two_nrl` / `reward` / `punish` / `train(phase="negative")` / `invert` | Ollama; web search (`RADIXNET_SEARCH_URL`), web fetch, calculator, the sandbox, Chrome over WebDriver |
| Chat (LLM partner and judge) | `chat.py` | `dialogue.reply` (Level 3), then feedback | Ollama or ChatGPT |
| Critic (adversarial review) | `critic.py` | `generate(sample)`; blames the negative network only | an LLM |
| Recall | `recall.py` | `predict` on the opening of a trained image or speech text | Pillow or an ASR |
| Dialogue / thinking / voice | `dialogue.py`, `thinking.py`, `voice.py`, `voicechat.py` | Level 3 | PhoneticTokenizer for voice |
| Speech / vision / media | `speech.py`, `vision.py`, `media.py` | nothing: audio and images become **text** the model trains on (`<speech:digest>` + waveform, `img:sd|tiny:WxH:base64`) | ffmpeg, faster-whisper / whisper / an ASR URL; torch + diffusers or Pillow |

**The LLM client shape** (`llm.py`, `LLMClient` Protocol): `provider`, `url`,
`model`, `models()`, `available()`, `generate(prompt, *, system, json_mode,
options, timeout)`, `chat(messages, …)`; `make_client("ollama"|"chatgpt",
url, model, timeout, api_key)`. Ollama: `OLLAMA_HOST` (default
`127.0.0.1:11434`), `RADIXNET_OLLAMA_MODEL` (`llama3.2`), over `/api/tags`,
`/api/generate`, `/api/chat`. ChatGPT or any OpenAI-compatible endpoint:
`OPENAI_API_KEY` (or `_FILE`), `OPENAI_BASE_URL` (default
`https://api.openai.com/v1`), `RADIXNET_OPENAI_MODEL` (`gpt-4o-mini`). Per-loop
overrides: `RADIXNET_TUTOR_MODEL`, `RADIXNET_AGENT_MODEL`,
`RADIXNET_CODEGEN_MODEL`. The API takes `provider`, `*_model`, `url`,
`timeout` in request bodies; the key never. `reader_text` spells a model of
sounds into English before an LLM reads it. A model of a new kind needs
nothing here but Levels 1–2.

## 11. MCP

`radixnet mcp` (`modelkit/mcp.py`; Go `radixnet-count mcp`; Rust `radixnet
mcp`): JSON-RPC 2.0, one object per line over stdio, protocol
`2024-11-05`, server name `radixnet`; `initialize`, `ping`, `tools/list`,
`tools/call` (→ `{content: [{type: "text", text}], isError}`), notifications
acted on and unanswered. Tools: the toolbox's `web_search`, `web_fetch`,
`web_links`, `calculator`, optionally `python` and `read_file`; and the
model's `radixnet_predict {prefix_text, length, mode, temperature}`,
`radixnet_generate {count, max_length, temperature}`, `radixnet_score
{text}`, `radixnet_stats`, `radixnet_judge {text}` (when a negative network is
loaded) and `radixnet_solve {task, max_steps, source}` (when an Ollama client
is). Go and Python are asserted to answer one message stream identically
(D-074). Level 1 lights the first four tools.

## 12. Ports and parity

A port of the model is a second implementation of §4 and §5 in Go
(`RadixCyclicNN/go/radixnet`, module
`github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go`) or Rust
(`RadixCyclicNN/rust`, crate `radixnet`); the kit's port
(`ModelKit/go`, `ModelKit/rust`) binds to it concretely — `server.Service`
wraps `*radixnet.Model`, `service::Service` holds `Mutex<Model>`, and Rust's
per-kind dispatch is one module, `kinds.rs`, "so that the CLI and the server
cannot answer a kind differently". The kit is consumed by path: Go `replace`
directives to `../../RadixCyclicNN/go` and `../../PhoneticTokenizer/go`
(repeated by any consumer, since `replace` applies only in the main module);
Rust `radixnet = { path = "../../RadixCyclicNN/rust" }`.

**Parity is measured on the surfaces, not the prose** (D-074): the route
list, the command list, and the fields and status code of each — a job
answers 202 everywhere, `texts` / `text` / `files` are read everywhere, each
kind saves to its own file. A gap is either **undone** or **deliberate** and
named in `go/README.md` / `rust/README.md` (today's deliberate list: the torch
backend, Stable Diffusion, local Whisper; Go runs the count kind only, D-038).
`make parity` runs `ModelKit/tests/test_go_parity.py` and
`test_rust_parity*.py`: both build the binaries, train the same corpus on each
side with `--json`, and compare structure, counts, rewards, window, RNG state,
predictions, generated texts, scores, conversations and blame verdicts; each
side loads the other's files; the Rust graph document must equal Python's
**byte for byte** except the `version` stamp and clock fields.

What that forces every port to carry, and a new model's ports with it:

| must agree to the bit | because |
|---|---|
| **MT19937 with CPython `random.Random` semantics** (`init_by_array`, 53-bit doubles, the 624-word state in the file) | training shuffles and sampled walks continue across ports |
| **`fsum`** (Shewchuk) for path costs | a `-log P` sum must match to the last bit |
| **BLAKE2b** (RFC 7693) | the resonant kind's gram phases; the speech `<speech:digest>` token in training text — "parity is a property of the file format" |
| **JSON** as Python writes it (compact, `repr` floats, insertion order); **gzip** readable everywhere (Rust writes stored blocks) | the file is the contract |
| edge order = first-seen = softmax summation order; node ids compacted on save as `START, END, sentinels, alive…` | costs and files |
| cyclic counters at 10^15 with `*_resets`; SplitMix64 keys for training order so no RNG draw is consumed (D-078) | numbers that never overflow and never shift the sequence |

`bench/compare.py` refuses to print a timing until parity holds
(`EXACT_KEYS` equal, `CLOSE_KEYS` within 1e-9, the same continuation at the
same cost); `bench/RESULTS.md` is the record.

## 13. Packaging and deployment

* **Python**: two packages, `pip install -e RadixCyclicNN -e ModelKit` (or
  both checkouts on `PYTHONPATH`; each finds the other beside it). Scripts:
  `radixnet` → `radixnet.__main__:main` (hands to the kit), `modelkit` →
  `modelkit.cli:main`. Python ≥ 3.11.
* **Image** (`RadixCyclicNN/Dockerfile`, context = the repository root,
  `Dockerfile.dockerignore` scopes it): stage 1 builds `frontend/dist` with
  Node; stage 2 is `python:3.11-slim` with `/app/RadixCyclicNN` beside
  `/app/ModelKit`, `pip install ./RadixCyclicNN ./ModelKit`, optional torch
  (`WITH_TORCH=1`), user `radixnet`, volume `/data`, port 8000, a healthcheck on
  `/api/health`, entrypoint `docker/entrypoint.sh` (restores the latest
  checkpoint first when `RADIXNET_RESUME=1`) then `radixnet "$@"`.
* **Compose** (`docker-compose.yml`, `.env.example` lists every knob):
  `api` (serve + frontend, `--checkpoint-dir /data/checkpoints`,
  `--upload-dir /data/uploads`, `stop_signal: SIGINT`); profiles `auto`
  (a one-shot sidecar POSTs `/api/evolve/start` once the API is healthy, so
  the self-upgrade loop runs *inside* the API and is visible in the Evolve
  tab), `tools` (`train`, `test`, `bench` one-shots), `evolve` (a headless
  CLI loop writing `/data/model.json` and `/data/discriminator.json`), `dev`
  (Vite on 5173 proxying to `api`), `ollama` (a local LLM; set
  `OLLAMA_HOST=http://ollama:11434`); `docker-compose.gpu.yml` overlays torch
  and every NVIDIA GPU. The API keeps its own in-memory model: what `train` /
  `evolve` write to `/data/model.json` shows up after `POST /api/load` or a
  restart with `RADIXNET_RESUME=1`.
* **Environment** the servers read: `RADIXNET_MODEL`,
  `RADIXNET_CHECKPOINT_DIR`, `RADIXNET_RESUME`, `RADIXNET_BACKEND`, the LLM
  variables of §10, `RADIXNET_WHISPER_MODEL`, `RADIXNET_ASR_URL/MODEL`,
  `RADIXNET_SEARCH_URL`, `RADIXNET_USER_AGENT`, `RADIXNET_CHROMEDRIVER`,
  `RADIXNET_CHROME`, `RADIXNET_WEBDRIVER`.

A new model's directory supplies the equivalent of RadixCyclicNN's
`Dockerfile`, `docker-compose.yml` and `Makefile` by **copying them and
changing the model directory name** — the kit half of each is identical, and
the compose file already speaks only in CLI arguments and `/api` calls.

## 14. The checklist for a new raw model

In the order the surfaces come alive. Each step names its test.

1. **A package beside the others**, `pyproject.toml` with `dependencies = []`
   (torch under an extra), a `README.md`, a `DESIGN.md` (D-019: normative,
   changed in the same commit as the code), `tests/` in plain `unittest`,
   `data/` with a sample corpus and a garbage twin, a `.gitignore` for
   `model*.json`, `checkpoints/`, `uploads/`. The license comment every
   `pyproject` carries.
2. **Level 0**: the class attributes, the constructor, `stats`, `to_dict` /
   `from_dict` / `save` / `load` with a unique `format` and `version: 1`, the
   RNG state in the file, cyclic counters in `meta`. Register in
   `model_classes()` — today that is an edit to
   `RadixCyclicNN/radixnet/model.py` (§16). *Test:* `load_model` round-trips
   the file; `ModelKit/tests/test_checkpoint.py` saves and restores it;
   `make test-layers` still holds.
3. **Level 1**: `train` honouring `progress`, `stop_event`,
   `checkpoint_manager` and ignoring dials it has no use for; `predict` and
   `generate` returning `PathResult`s with costs as `-log P` and lengths in
   units; `score`. *Test:* `radixnet --kind <new> train … && predict …`;
   `tests/test_api.py` (the 202 / 409 / progress / stop cycle); the Train,
   Predict, Generate, Score tabs.
4. **Level 2**: `two_nrl`, `reward`, `punish`, `invert`, `invert_paths`, and
   `correct` if the kind learns from a diff (then `takes_corrections = True`).
   *Test:* `tests/test_feedback.py`, `test_gan.py`, `test_tutor.py` with a fake
   Ollama.
5. **Level 3** if the model is a walkable structure: `graph` with the
   sentinels and the methods of §4.4, `_search`, `_prefix_start`. *Test:*
   `tests/test_dialogue.py`, `test_assistant.py`, `test_thinking.py`; `curl -N
   /v1/chat/completions` streams; `talk --agent-config agent.json` answers.
   Otherwise document the gap and let those routes 400 with the kind's name.
6. **Level 4** as the kind has them: weights, paths, attention, window, a
   word vocabulary. Each gets a SPEC in the house template.
7. **The frontend**: nothing, until the kind has a card of its own; then a
   component, a `TABS` entry with its `route`, `api.js` functions, a
   `node --test` for any pure logic, `make frontend-build` so `dist/` is
   committed with it.
8. **Ports** when wanted: a Go package and a Rust crate implementing §4–§5
   with the determinism table of §12; an entry in `kinds.rs` and `Kinds()`; a
   parity suite file per area under `ModelKit/tests/`; `make parity` green
   before `bench` is allowed to time anything.
9. **Deployment**: copy `Dockerfile`, `docker-compose.yml`, `.env.example`,
   `docker/entrypoint.sh`, the `Makefile`; change the directory name.
10. **Docs**: a row in this document's §2 tree, a D-entry in `DECISIONS.md`
    recording the kind and its format string, and the model's README pointing
    at the kit's for every command it does not own.

**Where the existing directories stand against it.** RadixAcyclicNN's
`RadixTreeNet` already has `kind`, `label`, `train(texts, config, *, phase,
progress)`, `predict`/`generate` returning `PathResult`s, `score`, `two_nrl`,
`invert`, `stats`, a JSON `format` — it lacks `stop_event`,
`checkpoint_manager`, `backend` / `device` in `load`, a `format` class
attribute the registry can read, and the Level 3 surface (its tree is a graph
without `BACK` / `THINK`); it is the nearest candidate. RadixDecayNN's
`DecayNet` has `read` where the kit expects `train`, no Level 2, and **its
queries mutate the model** unless `quiet=True` — a kit that assumes `predict`
is read-only (it does, under the reader lock) would need to know. FilterBankRadix
trains on `{"text", "source"}` samples and predicts one next character; it
needs text-level `train` / `predict` / `score` / `stats` adapters. GTMNN's
design is written against these very contracts and has no code. CyclicCortex,
GREN and AudioImage's `QNet` are not text models; the kit would need a
non-text seam for them, which this document does not specify.

## 15. Tests the integration must keep green

| suite | what it holds |
|---|---|
| `RadixCyclicNN/tests` (`make test-core`) | the model alone, with the kit **absent** from the path |
| `ModelKit/tests/test_layers.py` | the model-free modules import with `radixnet` refused; every model-layer module needs it; `import modelkit` loads no model; `radixnet` imports the kit nowhere but `__main__` |
| `ModelKit/tests/test_api.py`, `test_api_uploads.py`, `test_cli.py` | every route's fields and status codes; the job cycle; uploads and ZIPs; every command's output |
| `test_assistant.py`, `test_agent_config.py`, `test_dialogue.py`, `test_thinking.py`, `test_voice*.py` | the two dialects whole and streamed; `agent.json` end to end against a live server; the walks |
| `test_guard.py`, `test_feedback.py`, `test_gan.py`, `test_tutor.py`, `test_codegen.py`, `test_agent.py`, `test_chat.py`, `test_critic.py`, `test_recall.py`, `test_mcp.py` | the loops, against fake `http.server` LLMs — nothing external is ever contacted |
| `test_go_parity.py`, `test_rust_parity*.py` (`make parity`) | the ports held to Python's bytes; skipped without `go` / `cargo` |
| `frontend/test/*.test.mjs` (`make frontend-test`) | the SSE and NDJSON parsers, settings folding, storage, the agent descriptor |

Everything is seeded; the parity suites depend on identical Mersenne Twister
state in every port.

## 16. What this does not do

* It does not make the kit generic. There is no `Model` interface, protocol
  or trait, and D-093 says why; this document names the de facto contract so a
  new model can meet it, not so the kit can forget it.
* It does not open the registry. `model_classes()` is a hard-coded dict in the
  model package, so a new kind outside RadixCyclicNN still registers by
  editing it (or by the kit growing an entry-point / environment-variable
  loader, which is not built). Likewise Rust's `kinds.rs` and Go's `Kinds()`.
* It does not give `dialogue.reply` a model-level seam. Level 3 is the
  graph; a non-graph model cannot reach `/v1` without a shim.
* It does not specify a non-text model contract (games, audio strips).
* It does not add an Ollama-compatible *server* surface, `/v1/completions` or
  `/v1/embeddings`, or enforce `response_format`.
* It does not change any route, field, file key or command. Where it and
  `DESIGN.md` disagree, `DESIGN.md` wins and this file is wrong.

## 17. Open items this document surfaced

Found while writing, recorded here rather than fixed, because each is a
change to something other than a document:

* `RadixCyclicNN/checkpoints/ckpt-epoch-*.json.gz` are tracked (force-added
  past `.gitignore`) and carry `format: "radixnet-word"`, dropped by D-073;
  `load_model` refuses them, and `index.json` holds absolute paths from
  another machine.
* `SPEC-EdgeDecay.md` is *Proposed* and calls its file change "format 4",
  which the THINK sentinel already took.
* `DECISIONS.md` numbers D-080, D-081 and D-082 twice each, and `DESIGN.md`
  has two §24 and two §28; cite them by title or line.
* `README.md` still documents `--kind word`; the code reads it as
  `--encoding word:3:1`.
* The frontend's title is `RadixCyclicNN` rather than `status.model_label`.

## 18. Alternatives rejected

* **A kit of interfaces, generic over the model.** Rejected in D-093 for the
  reason above; this document's levels are the readable substitute for an
  interface that would be the model's API twice.
* **Each model directory keeping its own `api.py` / `cli.py` /
  `checkpoint.py` "of the same shape".** It is what FilterBankRadix,
  CyclicCortex, GREN and GTMNN do or plan, and it is four copies of a contract
  that already drifts (GTMNN's job shape is a snapshot of an older `api.py`).
  The kit exists so that a model is *loaded*, not re-wrapped.
* **An adapter layer in the kit that wraps any `predict`/`generate` object.**
  Tempting for Level 1, but Level 3 is where the kit earns its keep, and a
  wrapper cannot fake `observe_back` or a `THINK` sentinel. Better that a
  model declares what it lacks and the routes say so.
* **Ollama's server API as the query spec.** The two dialects already cover
  every client; an Ollama face would be a third rendering of the same search
  and a path collision with the kit's own `/api/generate`.
* **Putting this document in ModelKit.** The kit's README is the map of the
  kit; the design documents stay with the model because they describe one
  system (D-019), and a repository-wide integration spec belongs where the
  directories it ranges over can all see it: the root.
