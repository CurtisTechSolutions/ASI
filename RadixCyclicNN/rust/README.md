# rust

A Rust port of the **count / reward model** (`CountRewardNet`), written to
answer one question — *how much of this model's cost is the language?* — and it
answers to all three traversals the other two do: `reward`, the walk that
follows what the model believes; `punishment`, the same graph priced by the
penalties alone (`../radixnet/penalty.py`, DESIGN §31); and `least-punished`,
the walk that ranks by the blame a path carries before it looks at the cost at
all (`../SPEC-LeastPunished.md`).

A standalone crate. The Python implementation in `../radixnet/` and the Go port
in `../go/` are untouched by it.

**No dependencies**, like the other two: the hash, the Mersenne Twister, the
exact float sum and the worker pool are written out in `src/`. `Cargo.lock` has
nothing in it but this crate.

## Contents

| file | what it is |
|---|---|
| `src/encoding.rs` | the encoding dial: what one unit is, how many units a gram holds, how far apart grams start |
| `src/words.rs` | the word view of a word encoding: the alphabet its grams are made of (`radixnet words`) |
| `src/graph.rs` | the self-compressing cyclic graph: split, merge, observe, trace, the invariants |
| `src/weights.rs` | the dual frequency weight function, the softmax costs, and the punishment |
| `src/paths.rs` | what a *walk* did: the judged contexts and their counters |
| `src/search.rs` | the traversals, what a node offers a walk, and the stochastic walk |
| `src/penalty.rs` | the punishment traversal: the merit / penalty split, and the cost function it prices a step with |
| `src/beam.rs` | the two beams — the k best paths and the k worst |
| `src/model.rs` | train, predict, generate, score, reward / punish / 2NRL |
| `src/counter.rs` | the cyclic counters, wrapping at `10^15` |
| `src/parallel.rs` | the worker pool, and the one `unsafe` in the crate (with its contract) |
| `src/fsum.rs`, `src/mt19937.rs`, `src/hash.rs` | the exact sum, the RNG and the hash, written out |
| `src/file.rs` | the `radixnet-count` and `radixnet-word` model files: what Python and Go read and write |
| `src/json.rs` | JSON as Python writes it - compact, UTF-8, and floats rendered as `repr(float)` renders them |
| `src/gzip.rs` | the gzip container, written out: inflate for reading, a stored-block writer for writing |
| `src/clock.rs` | the one timestamp a model file carries |
| `src/report.rs` | the statistics, the judged paths, a node against its neighbours, and the weight knobs |
| `src/http.rs` | HTTP/1.1 written out: the requests, the routes, the static files and the SPA fallback |
| `src/service.rs` | the API the frontend talks to — the same JSON contract as the Python and Go servers |
| `src/bin/radixnet.rs` | the CLI: train, predict, generate, score, feedback, 2nrl, invert, compress, weights, paths, nodes, words, info, serve |
| `src/bench.rs`, `src/bin/radixnet-bench.rs` | the benchmark and its binary |
| `tests/model.rs` | the model end to end, and all three traversals |
| `tests/words.rs` | the word model: the alphabet, the phrases compression makes of it, and its file |
| `tests/server.rs` | the HTTP API end to end: a real server on a real port, answering the JSON contract |

## What is here, and what is not

The model is here in full: the graph and its structural operations, the weight
function, the path contexts, all three traversals, training, prediction,
generation, scoring, reward / punish / 2NRL, the `radixnet-count` **model file**,
the CLI over all of it, and the **HTTP server** the frontend talks to — and, with
`--encoding`, the same model over any n-gram of characters *or* words
(`char:3:1` is the default, `char:5:5` groups of five letters, `word:3:1` the
word trigram; `../SPEC-WordNGrams.md`). A model trained here continues in Python
or in Go and back again, and `frontend/dist` runs against `radixnet serve` the
same way it runs against the other two — including switching between a character
and a word encoding from the model selector, which parks the model that was
running rather than dropping it.

Not ported: the negative network, the tutors and the other teaching loops, the
agent and its tools, the LLM clients, images and speech, and MCP. The first
group is a gap of the *undone* kind. The LLM clients are the deliberate kind:
they need HTTPS, and a crate with no dependencies cannot speak it — porting them
means giving up the rule in D-072, which is a decision to take rather than a
thing to quietly do.

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
