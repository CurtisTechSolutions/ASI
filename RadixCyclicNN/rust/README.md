# rust

A Rust port of **everything the Python package does** - every model kind, every
teaching loop, the negative network, the LLM clients, the agent and its tools,
code generation, images and speech, MCP, the CLI and the HTTP API the frontend
talks to - held to Python by the cross-language suites in
`../tests/test_rust_parity*.py`.

It began as a port of the **count / reward model** (`CountRewardNet`), written
to answer one question - *how much of this model's cost is the language?* - and
it carries all three traversals: `reward`, the walk that follows what the model
believes; `punishment`, the same graph priced by the penalties alone
(`../radixnet/penalty.py`, DESIGN §31); and `least-punished`, the walk that
ranks by the blame a path carries before it looks at the cost at all
(`../SPEC-LeastPunished.md`).

A standalone crate. The Python implementation in `../radixnet/` and the Go port
in `../go/` are untouched by it.

**No dependencies**, like the other two: the hash, the Mersenne Twister, the
exact float sum, the worker pool, gzip, ZIP, PNG / JPEG / GIF, BLAKE2b, base64,
multipart, HTTP/1.1 (server and client) and the logger are written out in
`src/`. `Cargo.lock` has nothing in it but this crate. The one thing that is not
written out is TLS: an `https://` request goes through the system's `curl`
(`src/fetch.rs`, D-076).

## Contents

### The model

| file | what it is |
|---|---|
| `src/encoding.rs` | the encoding dial: what one unit is, how many units a gram holds, how far apart grams start |
| `src/words.rs` | the word view of a word encoding: the alphabet its grams are made of (`radixnet words`) |
| `src/graph.rs` | the self-compressing cyclic graph: split, merge, observe, trace, the invariants, and `observe_back` (what a voice backing out of a loop teaches it) |
| `src/weights.rs` | the dual frequency weight function, the softmax costs, and the punishment |
| `src/paths.rs` | what a *walk* did: the judged contexts and their counters |
| `src/search.rs` | the traversals, what a node offers a walk, the stochastic walk and the sampling filters it draws through (top-K, top-p, min-p) |
| `src/penalty.rs` | the punishment traversal: the merit / penalty split, and the cost function it prices a step with |
| `src/beam.rs` | the two beams - the k best paths and the k worst - and the diverse pick of the k best |
| `src/training.rs` | how a run walks its texts: the order, the curriculum, the replay buffer and early stopping (`../SPEC-SearchAndTraining.md`), keyed by SplitMix64 so no random number is drawn |
| `src/model.rs` | train, predict, generate, score, reward / punish / 2NRL, and the epoch hook checkpoints are written through |
| `src/kinds.rs` | the model kinds - `count` (this port's default), `radix`, `resonant`, `negative` - and training, 2NRL, reward, punish and `invert_paths` dispatched to each |
| `src/radix.rs`, `src/activation.rs`, `src/backend.rs`, `src/dijkstra.rs` | the sine-activation model: `f(z) = a*sin(b*(z-h)) + k` per node, a learned weight per edge, Python's one-hop learning rule over CSR, and the exact Dijkstra prediction |
| `src/schedule.rs` | learning-rate schedules - a small arithmetic language, parsed rather than evaluated - `radixnet schedule`, `/api/schedule` and its preview |
| `src/resonance.rs`, `src/phasesearch.rs`, `src/metacog.rs` | the phase model: edges learn the phase they fire at, the search runs over (node, units, phase), and a phase-locked cycle goes to the metacognitive layer |
| `src/negative.rs` | the negative network: a model built only out of failures, blame and clearing, verdicts and the "why" sentence |
| `src/duo.rs` | the pair on the way out: the positive model writes, the negative one vetoes; the guard on every answer and the `/api/negative` routes |
| `src/blame.rs`, `src/diff.rs` | where the negative network's data comes from - verdicts turned into faults - and the unit diff that blames only what a teacher changed |
| `src/correct.rs` | `Model::correct` and `radixnet correct`: teach one correction, only what changed moves |
| `src/attention.rs` | the attention band and `radixnet attention`: where inside a gram a correction lands, the charges shared out as Python shares them (`../SPEC-AttentionBand.md`) |
| `src/window.rs` | the dynamic window and `radixnet window`: the ladder of node sizes halving from 32 to 4 and back up, every node longer than the window halved at the same gram Python halves it, with the same heavy connection - the sine model's weight, the phase model's through-traffic - and the same file (`../SPEC-DynamicWindow.md`) |
| `src/dialogue.rs` | the model converses with itself: skipping what was heard, backing out of a repeat, and teaching the graph where it goes round; `converse --stream` and `POST /api/converse/stream` watch it happen, the turns as they are spoken and the backing up between them (a `Stream`, the same events as Python and Go); `reply` hears a `trace` |
| `src/assistant.rs` | today's format: messages in, an assistant message out - the search's trace as the thinking, the text one node of the walk at a time - in OpenAI's and Anthropic's dialects, `radixnet talk`, `/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models` |
| `src/counter.rs` | the cyclic counters, wrapping at `10^15` |
| `src/parallel.rs` | the worker pool, and the one `unsafe` in the crate (with its contract) |
| `src/fsum.rs`, `src/mt19937.rs`, `src/hash.rs`, `src/pyheap.rs`, `src/blake2b.rs` | the exact sum, CPython's RNG, the hash, `heapq`'s array layout and BLAKE2b, written out |
| `src/file.rs` | the model files of every kind - `radixnet-count`, `radixnet`, `radixnet-resonant`, `radixnet-negative` - what Python and Go read and write |
| `src/json.rs` | JSON as Python writes it - compact, UTF-8, and floats rendered as `repr(float)` renders them |
| `src/gzip.rs`, `src/zip.rs`, `src/source.rs` | gzip written out, ZIP archives read with its inflate, and corpora streamed entry by entry with Python's skip rules |
| `src/checkpoint.rs` | checkpoints in the Python `CheckpointManager` layout, written from inside a run of any kind; `radixnet checkpoints`, `/api/checkpoints/*` |
| `src/gan.rs` | the evolve (GAN) loop on every kind - the same draws, generator, discriminator and negative network as Python: `radixnet evolve`, `/api/evolve/*` |
| `src/report.rs` | the statistics, the judged paths, a node against its neighbours, and the weight knobs |
| `src/bench.rs`, `src/bin/radixnet-bench.rs` | the benchmark (`radixnet bench`) and its binary |

### The teachers and the tools

| file | what it is |
|---|---|
| `src/fetch.rs` | the HTTP client: plain HTTP over `TcpStream`, HTTPS through the system `curl` with the headers (an API key) in a private file, a peer checked where it is connected to |
| `src/llm.rs`, `src/llm/fields.rs` | what every LLM loop shares: `LlmClient` (models, generate, chat), options, errors, the server's teacher defaults, request fields read as the Python server reads them |
| `src/ollama.rs`, `src/chatgpt.rs` | the two providers: `radixnet ollama models\|corpus\|review`, `radixnet chatgpt models\|ask`, `/api/ollama/*`, `/api/chatgpt/models` |
| `src/review.rs` | a corpus from a prompt, the adversarial review and the conversation marking, with Python's prompts byte for byte |
| `src/critic.rs` | the Negative tab's automatic loop: the model writes, an LLM reviews, the failures blame - `radixnet negative auto`, `/api/negative/auto` |
| `src/tutor.rs`, `src/tutor/{trainer,serve}.rs`, `src/plan.rs` | the English tutor: exercises, marks, rewards weighted by the mark, the marker's thinking taught as thoughts, the auto run and the lesson plan - `radixnet tutor`, `/api/tutor/*` |
| `src/chat.rs` | an LLM converses with the model and marks every reply - `radixnet chat`, `/api/chat/*` |
| `src/calc.rs`, `src/web.rs` | the calculator (Python's grammar, whitelist and error texts) and browsing (a client that refuses anything but a public page, every redirect hop re-checked) |
| `src/tools.rs`, `src/toolbox.rs`, `src/toolbox/sandbox.rs` | the tool registry and the `<tool>` text format, the built-in tools, and the Python sandbox programs run in - `radixnet tools`, `/api/tools/*` |
| `src/vision.rs`, `src/vision/{png,jpeg,gif}.rs` | images as text: the Go port's thumbnail encoder over PNG / JPEG / GIF decoders written out - `radixnet image`, `/api/images/*` |
| `src/speech.rs`, `src/speech/asr.rs` | speech as text: WAV in every format Python reads, both codecs and the unique token; transcripts given or from an OpenAI-compatible server - `radixnet speech`, `/api/speech/*` |
| `src/recall.rs` | the recall tutor over images and speech, and what it teaches the negative network |
| `src/multipart.rs`, `src/multipart/base64.rs` | request bodies as Python reads them (multipart, raw, base64) and uploads stored as Python stores them, a ZIP kept whole; `POST /api/uploads` streams to disk, whatever the size |
| `src/codegen.rs` | code generation: problems run in the sandbox, style-checked and judged, blame and 2NRL through the model's kind - `radixnet codegen`, `/api/codegen/*` |
| `src/agent.rs` | tool use: criteria, mediated `<tool>` calls, judging, teaching, failure-first learning and exploring - `radixnet agent` / `explore`, `/api/agent/*` |
| `src/mcp.rs`, `src/mcp/pyjson.rs` | the tools and the network over MCP (JSON-RPC 2.0 on stdio), Python's answers line for line, and JSON read the way `json.loads` reads it, down to its error messages - `radixnet mcp` |
| `src/browser.rs` | Chrome over W3C WebDriver behind the web tools (`--browser`, `tools browser`) |

### The surfaces

| file | what it is |
|---|---|
| `src/cli.rs` | the command line's plumbing, and the table that sends a command to the module that answers it |
| `src/bin/radixnet.rs` | the CLI: the model's own commands, and every other module's through `cli::COMMANDS` |
| `src/http.rs` | HTTP/1.1 written out: the requests, the routes (a body read whole and parsed, or streamed for an upload of any size), the static files and the SPA fallback, and the routes that stream server-sent events |
| `src/service.rs` | the API the frontend talks to - the same JSON contract as the Python and Go servers; `/api/status` lists every route it serves |
| `src/log.rs` | logging, written out: levels, targets, timestamps, and never a byte on stdout |
| `src/clock.rs` | the one timestamp a model file carries |
| `tests/*.rs` | the model, the encodings, the words and the HTTP API end to end |
| `../tests/rust_harness.py`, `../tests/test_rust_parity*.py` | the contract with Python, one suite per area |

## What is here, and what is not

All of it, and each area answers to Python through its own parity suite: the
same model files (byte for byte, but for the `version` cache stamp and the
clock), the same predictions and scores, the same prompts sent to an LLM, the
same marks, and the same model afterwards.  The frontend runs against
`radixnet serve` unchanged: a tab is shown when the route it needs is in
`/api/status`.

**Deliberately not ported**, and why:

* **the torch backend** (`backend_torch.py`) - it is torch on a GPU; the
  sine model trains with Python's own CPU learning rule instead, and
  `--backend torch` says so;
* **the Stable Diffusion image encoder** - it needs torch; images go through the
  Go port's thumbnail encoder, as they do against the Go server;
* **local Whisper** (`faster-whisper`, `whisper`) - Python packages; speech is
  transcribed from a given transcript or by an OpenAI-compatible transcription
  server.

**Where it differs on purpose:** `count` stays this port's default kind (Python's
is `radix`); a CLI run cannot be interrupted with Ctrl-C (the standard library
cannot catch it), so the long loops save as they go; the calculator's integers
stop at 2**127; and where Python has a bug the port does what Python meant and
says so in the code (a word model's corrections,
the review's `lessons` key).

## Building and running

From `..`:

```bash
make rust-build        # cargo build --release -> rust/target/release/radixnet{,-bench}
make rust-test         # cargo test, cargo clippy, cargo fmt --check
make rust-train        # train the Rust model on DATA (RUST_KIND=count|word)
make rust-serve        # serve frontend/dist from the Rust model on http://HOST:PORT
make rust-parity       # the contract with Python
make bench-compare     # both ports over one corpus -> bench/RESULTS.md
```

## Logging

`RADIXNET_LOG` (or `--log`) takes a level, optionally per target:

```bash
radixnet --log info train --data corpus.txt      # what the run did
radixnet --verbose train --data corpus.txt       # ... and every epoch
radixnet --log warn,http=debug serve             # one line per request
RADIXNET_LOG=info,train=trace radixnet train ...
radixnet --quiet ...                             # nothing at all
```

The default is `warn`, so a CLI run says nothing unless something is wrong;
`serve` raises its own default to `info`, because a server that says nothing
while it runs cannot be debugged. Targets: `http`, `model`, `train`, `negative`,
`checkpoint`, `evolve`, `bench`, `llm`, `tools`, `media`, `tutor`, `chat`,
`critic`, `codegen`, `agent`, `mcp` and `browser`.

**Every line goes to stderr**, and there is no way to configure one onto
stdout. That is not a style choice: `--json` puts one document there, so a log
line landing in it corrupts the document.

`make help` lists the rest of the `rust-*` targets — predict, generate, score,
feedback, 2nrl, invert, compress, weights, paths, nodes, words and info — each
one the CLI subcommand of the same name.

or directly:

```bash
cd rust
cargo test
cargo run --release --bin radixnet -- --model model.count.json train --data ../data/sample_corpus.txt --epochs 5
cargo run --release --bin radixnet -- --model model.count.json predict --prefix "the cat" --k 5
cargo run --release --bin radixnet -- --model model.count.json predict --prefix "the cat" --traversal least-punished
cargo run --release --bin radixnet -- --model model.count.json predict --prefix "the cat" --traversal punishment --merit-scale 0
cargo run --release --bin radixnet -- --model model.count.json --seed 7 generate --mode sample --top-p 0.9 --seeded
cargo run --release --bin radixnet -- --model model.count.json generate --mode beam --count 5 --diversity 2
cargo run --release --bin radixnet -- --model model.count.json train --data ../data/sample_corpus.txt --order shortest-first --curriculum 0.3 --replay-size 256
cargo run --release --bin radixnet -- --model model.backwards.json train --data ../data/sample_corpus.txt --reverse   # every text read backwards
cargo run --release --bin radixnet -- --encoding word:3:1 --model model.word.json train --data ../data/sample_corpus.txt
cargo run --release --bin radixnet -- --model model.word.json words --limit 20    # the alphabet its grams are made of
cargo run --release --bin radixnet -- --model model.count.json serve --port 8000 --frontend-dir ../frontend/dist
python3 -m radixnet --model rust/model.count.json info     # ... and Python reads the same file
cargo run --release --bin radixnet-bench -- --chars 200000 --epochs 3
cargo run --release --bin radixnet-bench -- --texts ../bench/corpus.txt \
    --prefixes ../bench/corpus.prefixes.txt --punish-every 7 --traversal least-punished
```

```rust
use radixnet::{GraphOptions, Model, PredictOptions, TrainOptions, LEAST_PUNISHED};

let mut model = Model::new(0, GraphOptions::default())?;
model.train(&texts, &TrainOptions { epochs: 5, ..Default::default() })?;

let found = model.predict("the cat sat on the ", &PredictOptions {
    length: 6, k: 3, traversal: LEAST_PUNISHED.to_string(), ..Default::default()
})?;
println!("{} (worst step: {})", found.best.full_text, found.best.punish);
```

## The same model, checked

`../tests/test_rust_parity.py` is the contract with Python, and it is the one
`test_go_parity.py` holds the Go port to: both train the same corpus with the
same settings and must produce the same structure, counts, rewards, sliding
window and RNG state, the same predictions, generated texts and scores, the same
judged paths and node ratios — and each side must load and continue the other's
file, gzipped or not.

`../tests/test_rust_parity_methods.py` does the same for the search and training
methods on the count, sine and phase models: eight training plans - two of them
read backwards (`--reverse`, `Plan.reverse`), in characters and in words - the
same graph, history and `replay` block byte for byte, and the same texts from the
filters and the diverse beam.

One thing it asks for that the Go suite does not: **the graph document has to be
Python's byte for byte**, but for the `version` cache stamp that every load
bumps. Same key order, same float rendering (`repr(float)`, which is not how
Rust prints a float), same everything. It is a stricter bar, it is free once the
writer is right, and it turns "the numbers agree" into "it is the same file".

The benchmark checks the other side of it: `bench/compare.py` refuses to report a
timing until Go and Rust agree on the nodes, the edges, the trigrams, the
transitions, the compression ratio, the loss, the search expansions and the
prediction — down to the cost of the path, which comes out bit for bit identical
because the summation order and the exact sum (`fsum`) are the same on both sides.

They are compared with Go's `--exact` counting, for the same reason: Go's
default counting is racy by design (D-038), and a benchmark of a deliberate data
race measures the race.

## Where the two implementations genuinely differ

Two representation choices, none of them algorithmic. They are why the port is
faster than a line-by-line translation would be, and they are the first place to
look before reading the numbers as a language comparison. A third one is gone:
this port used to pack a trigram into a `u64` (three code points of 21 bits),
which is what made its encoding pass allocate nothing — the encoding dial's
grams are arbitrary text, so the index is keyed by the gram as the other two key
it, and `bench/RESULTS.md` says what that cost:

| | Go | Rust |
|---|---|---|
| a gram | a freshly allocated `string` per window | the same: a gram of any n over any unit does not fit in an integer |
| a node's children during a search | a fresh `[]ChildCost` per expansion | filled into a buffer the search reuses |
| the hash behind the trigram index | the runtime's (AES-assisted on amd64) | `FxHasher`, written out in `src/hash.rs` |

And two structural ones, which the benchmark handles by comparing both ports at
one worker as well as at their own default parallelism:

* **Fan-out.** Go spawns one goroutine per text; an OS thread is not a goroutine,
  so this port fans out over a pool the size of the machine.
* **The two beams.** Go runs the top and the bottom beam on two goroutines; here
  they run in turn on the calling thread. Go's `--workers 1` runs them in turn
  too, which is the row to read for a like-for-like comparison.

## The one `unsafe`

`parallel::Disjoint`, used by the two recomputes, which fan out over the nodes
and write the weight and the cost of each node's edges. Every edge belongs to
exactly one node — an invariant `Graph::check_invariants` proves — so no two
threads write the same index, which Rust cannot see from the types. The
alternative is an atomic per edge, paid for on every read the search makes.
Its safety contract is written out where it is defined.
