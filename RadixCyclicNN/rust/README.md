# rust

A Rust port of the **count / reward model** (`CountRewardNet`), written to
answer one question — *how much of this model's cost is the language?* — and to
carry the traversal the other two implementations did not have: the walk that
follows the **least punished** step instead of the best rewarded one
(`../SPEC-LeastPunished.md`).

A standalone crate. The Python implementation in `../radixnet/` and the Go port
in `../go/` are untouched by it.

**No dependencies**, like the other two: the hash, the Mersenne Twister, the
exact float sum and the worker pool are written out in `src/`. `Cargo.lock` has
nothing in it but this crate.

## Contents

| file | what it is |
|---|---|
| `src/encoding.rs` | the dial — how many symbols a gram holds and how far apart grams start — and the packed gram |
| `src/words.rs` | the word alphabet: a word is one code point, and a vocabulary maps between them (`--kind word`) |
| `src/graph.rs` | the self-compressing cyclic graph: split, merge, observe, trace, the invariants |
| `src/weights.rs` | the dual frequency weight function, the softmax costs, and the punishment |
| `src/paths.rs` | what a *walk* did: the judged contexts and their counters |
| `src/search.rs` | the traversals, what a node offers a walk, and the stochastic walk |
| `src/beam.rs` | the two beams — the k best paths and the k worst |
| `src/model.rs` | train, predict, generate, score, reward / punish / 2NRL |
| `src/counter.rs` | the cyclic counters, wrapping at `10^15` |
| `src/parallel.rs` | the worker pool, and the one `unsafe` in the crate (with its contract) |
| `src/fsum.rs`, `src/mt19937.rs`, `src/hash.rs` | the exact sum, the RNG and the hash, written out |
| `src/file.rs` | the `radixnet-count` and `radixnet-word` model files: what Python and Go read and write |
| `src/json.rs` | JSON as Python writes it - compact, UTF-8, and floats rendered as `repr(float)` renders them |
| `src/gzip.rs` | the gzip container, written out: inflate for reading, a stored-block writer for writing |
| `src/clock.rs` | the one timestamp a model file carries |
| `src/report.rs` | the statistics, the judged paths, and a node against its neighbours |
| `src/bin/radixnet.rs` | the CLI: train, predict, generate, score, feedback, 2nrl, invert, compress, weights, paths, nodes, words, info |
| `src/bench.rs`, `src/bin/radixnet-bench.rs` | the benchmark and its binary |
| `tests/model.rs` | the model end to end, and both traversals |
| `tests/encodings.rs` | every encoding end to end: any n, groups of letters, words |

## Two dials, and they compose

How a text becomes grams is two questions, answered separately:

| dial | what it is | the default |
|---|---|---|
| `GraphOptions::words` | whether one **symbol** is a word rather than a character — every word gets a code point of its own (`src/words.rs`, `../SPEC-WordNGrams.md`) | `false` |
| `Encoding { n, stride }` | how many symbols a **gram** holds, and how far apart consecutive grams start: **1** slides the window (they overlap by n−1), **n** cuts the text into non-overlapping groups | `3`, `1` |

Neither has to know about the other, which is the whole point: a word bigram is
`words: true` with `n: 2`, and `Encoding` never learns what a word is.

```rust
use radixnet::{Encoding, GraphOptions, Model};

let bigrams = GraphOptions { words: true, encoding: Encoding { n: 2, stride: 1 }, ..Default::default() };
let groups = GraphOptions { encoding: Encoding { n: 5, stride: 5 }, ..Default::default() };  // five letters
let (words, enc) = Encoding::parse_spec("word:2:1")?;                                        // from a spec
let mut model = Model::new(0, bigrams)?;
```

and on the benchmark binary: `--encoding word:2:1`, or `--units`, `--ngram`
and `--stride` one at a time. Lengths, offsets and the `chars` of a score are
then counted in symbols — words, on a word model.

| `tests/words.rs` | the word model: the alphabet, the phrases compression makes of it, and its file |

## What is here, and what is not

The model is: the graph and its structural operations, the weight function, the
path contexts, both traversals, training, prediction, generation, scoring,
reward / punish / 2NRL, the `radixnet-count` **model file** and the CLI over all
of it — and, with `--kind word`, the same model over an alphabet whose symbols
are words (`radixnet-word`, `../SPEC-WordNGrams.md`; a `Trigram` already packs
three code points of 21 bits, which is every code point there is, so the
representation needed no change). A model trained here continues in Python or in
Go and back again.

Not ported: the HTTP server, the negative network, the tutors, the agent and the
LLM clients. That is a gap of the *undone* kind, not the deliberate kind.

## Building and running

From `..`:

```bash
make rust-build        # cargo build --release -> rust/target/release/radixnet-bench
make rust-test         # cargo test, cargo clippy, cargo fmt --check
make bench-compare     # both ports over one corpus -> bench/RESULTS.md
```

or directly:

```bash
cd rust
cargo test
cargo run --release --bin radixnet -- --model model.count.json train --data ../data/sample_corpus.txt --epochs 5
cargo run --release --bin radixnet -- --model model.count.json predict --prefix "the cat" --k 5
cargo run --release --bin radixnet -- --model model.count.json predict --prefix "the cat" --traversal least-punished
python3 -m radixnet --model rust/model.count.json info     # ... and Python reads the same file
cargo run --release --bin radixnet-bench -- --chars 200000 --epochs 3
cargo run --release --bin radixnet-bench -- --texts ../bench/corpus.txt \
    --prefixes ../bench/corpus.prefixes.txt --punish-every 7 --traversal least-punished
```

```rust
use radixnet::{GraphOptions, Model, PredictOptions, TrainOptions, Traversal};

let mut model = Model::new(0, GraphOptions::default())?;
model.train(&texts, &TrainOptions { epochs: 5, ..Default::default() })?;

let found = model.predict("the cat sat on the ", &PredictOptions {
    length: 6, k: 3, traversal: Traversal::LeastPunished, ..Default::default()
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

Three representation choices, none of them algorithmic. They are why the port is
faster than a line-by-line translation would be, and they are the first place to
look before reading the numbers as a language comparison:

| | Go | Rust |
|---|---|---|
| a gram | a freshly allocated `string` per window | up to three code points packed into a `u64`, no allocation; longer grams and word grams carry their text |
| a node's children during a search | a fresh `[]ChildCost` per expansion | filled into a buffer the search reuses |
| the hash behind the trigram index | the runtime's (AES-assisted on amd64) | `FxHasher`, written out in `src/hash.rs` |

And two structural ones, which the benchmark handles by comparing both ports at
one worker as well as at their own default parallelism:

* **Fan-out.** Go spawns one goroutine per text; an OS thread is not a goroutine,
  so this port fans out over a pool the size of the machine.
* **The two beams.** Go runs the top and the bottom beam on two goroutines; here
  they run in turn on the calling thread. Go's `--workers 1` runs them in turn
  too, which is the row to read for a like-for-like comparison.

## What the dial costs

Making the index key hold *any* gram rather than exactly three characters is not
free: the key went from an 8-byte `u64` to a 16-byte enum, which the training
pass carries one of per gram of the corpus and hashes on every step. Measured
against the same build with the trigram hard-coded, six runs of each
interleaved:

| workload | before the dial | on its default encoding |
|---|--:|--:|
| `bench-compare`'s corpus (62,757 short texts, 3 epochs) | 19.65 M transitions/s | 19.58 M (no measurable difference) |
| `radixnet-bench --chars 2000000` (synthesised sentences) | 19.6 M | 17.8 M (91 %) |

So it is free where the graph is small and the per-text work dominates, and
costs about 9 % where the key itself is the hot thing. A *word* gram pays
nothing extra beyond that: a word is one code point, so word grams of up to
three pack exactly as character trigrams do. Two choices keep it to
that: a packed gram hashes as the one `u64` it always was, and the text variant
sits behind a thin pointer so the key is 16 bytes and not 24. Closing the rest
would mean making the graph generic over its key type, so that a trigram graph
is keyed on a `u64` again — a change to every signature in the crate.

## The one `unsafe`

`parallel::Disjoint`, used by the two recomputes, which fan out over the nodes
and write the weight and the cost of each node's edges. Every edge belongs to
exactly one node — an invariant `Graph::check_invariants` proves — so no two
threads write the same index, which Rust cannot see from the types. The
alternative is an atomic per edge, paid for on every read the search makes.
Its safety contract is written out where it is defined.
