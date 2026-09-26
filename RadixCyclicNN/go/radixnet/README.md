# radixnet (Go)

`package radixnet` — the Go port of the **count / reward model** of
RadixCyclicNN: a self-compressing cyclic n-gram graph whose edge weights are a
dual frequency function of traversal counts (all time and inside a sliding
window) plus rewards, with beam-search prediction (top-K **and** bottom-K
continuations), generation, scoring, 2NRL feedback and a self-conversation.

**No dependencies.** The standard library, plus a BLAKE2b and a Mersenne Twister
written out here so the numbers match Python exactly.

## Interchangeable with Python

Model files use the `radixnet-count` and `radixnet-negative` formats — the same
JSON layout on both sides, **including the Mersenne Twister state** — so a model
trained here continues in Python and back again with identical numbers.
`../../tests/test_go_parity.py` checks that in both directions.

That holds for **every encoding**, not only the default one: a graph built with
anything but the character trigram writes an `encoding` block into its file, and
both sides read it. `../../tests/test_go_parity.py::TestGoEncodingParity` trains
the same corpus on both sides under nine of them and compares the graphs, the
files and the predictions.

## The encoding is a dial

`Encoding{Unit, N, Stride}` is how a graph turns text into grams and its labels
back into text. It is owned by the graph (`Graph.Enc`), fixed when the graph is
created and carried in the model file.

```go
opts := radixnet.DefaultGraphOptions()
opts.Encoding = radixnet.Encoding{Unit: radixnet.Words, N: 2, Stride: 1} // word bigrams
m, err := radixnet.NewModel(1, opts)

enc, err := radixnet.ParseEncoding("char:5:groups")   // groups of five letters
```

* **`Unit`** — `Chars` (a code point) or `Words` (a whitespace word; the text is
  normalised to single spaces).
* **`N`** — units per gram, any n ≥ 1.
* **`Stride`** — units between two grams: `1` slides the window (grams overlap
  by `N-1`, the trigram's shape), `N` cuts the text into non-overlapping groups.

Everything downstream counts in that unit: `labelLen`, the offset of a gram
inside a label, `PredictOptions.Length`, `Score.Chars`, and the spans of the
correction diff. Under the default encoding a unit *is* a character, so every
one of those is what it always was.

## Concurrency

Training fans goroutines out over the texts (lines, paragraphs or pages of a
file) for encoding, tracing and counting, with lock-free atomic increments;
weights and edge costs recompute in parallel over the nodes; and the two beams of
a prediction run side by side. **Only a structural change — a split or a merge —
takes a write lock.**

| file | what it is |
|---|---|
| `parallel.go` | the worker pool every fan-out goes through |
| `race_on.go` / `race_off.go` | build-tagged switches so `-race` builds count exactly |
| `memlimit.go` | the soft heap limit (`--memlimit`), 80% of the machine or container by default |
| `fsum.go` | an exact float sum (Python's `math.fsum`), so path costs come out identical on both sides |

## The core

| file | what it is |
|---|---|
| `encoding.go` | the encoder and decoder — the unit, the n of the n-gram, the stride — and the package doc |
| `graph.go` | the self-compressing cyclic graph: split, merge, the invariants |
| `nodes.go` | a node against its neighbours — how much of the node's traffic went each way, counted per side rather than per visit |
| `weights.go` | the dual frequency weight function over counts and rewards |
| `counter.go` | the cyclic counters, wrapping at `10^15` |
| `model.go` | `CountRewardNet` — train, predict, generate, score, feedback / 2NRL, invert, compress |
| `search.go`, `beam.go`, `paths.go` | shortest-path prediction, the two beams, paths judged per caller - and the **least-punished traversal**, which ranks a walk by the blame on its worst step instead of by its cost (`../../SPEC-LeastPunished.md`) |
| `penalty.go` | the **punishment traversal**: the merit / penalty split of an edge's evidence, and the cost function that prices a step by the punishment it carries instead of by the reward. `--traversal punishment` on `predict` / `generate`, `traversal` on `/api/predict` and `/api/generate` |
| `source.go`, `pipeline.go` | `TextSource` streams a corpus in order; `PartSource` splits it into parts each goroutine can stream on its own, including entries of a ZIP opened once |
| `json.go` | the model file formats (`radixnet-count`, `radixnet-word`, `radixnet-negative`), written to match Python byte for byte |
| `mt19937.go`, `blake2b.go` | the RNG and the hash, ported so both sides agree |
| `calc.go` | the calculator tool's expression language. Python evaluates these with `ast` and a whitelist; Go has no `eval`, so this is a parser written to match Python's arithmetic where the two differ |

## The negative side

| file | what it is |
|---|---|
| `negative.go` | the negative network — the failures, and why |
| `blame.go` | the tutor's verdicts turned into blame |
| `correct.go`, `diff.go` | corrections, and the character diff that decides which steps to blame |
| `attention.go` | the attention band: where inside a gram a correction's blame and credit land (`BandWeights`, `SpreadCharges`, `chargedSteps`, the preview) |
| `window.go` | the dynamic window: the ladder of node sizes halving from 32 to 4 and back up (`DynamicWindow`, `Ladder`), the halving (`SplitWindow`), the step by hand and at the end of every epoch (`WindowStep`, `windowEpoch`) and what the API and the CLI show (`WindowConfig`, `ConfigureWindow`) |
| `duo.go` | the positive model writes, the negative one vetoes |
| `gan.go` | the self-upgrading loop |
| `critic.go` | the reviewer on a loop — the negative network feeding itself |
| `review.go` | Ollama beyond the chat client: a training corpus written from a prompt (good or garbage), and adversarial review of the model's own samples |

## Teaching it

| file | what it is |
|---|---|
| `tutor.go` | the English lessons — set, answer, mark, and the marker's thinking taught as thoughts |
| `recall.go` | what it remembers of what it was shown |
| `dialogue.go`, `chat.go` | conversing with itself, and with an LLM that marks the replies |
| `plan.go` | the lesson plan — the report card at the end of a run, the grammar focus each mistake drills, and the level it moves the student to |
| `codegen.go` | code generation with a sandbox and a judge |
| `agent.go`, `tools.go`, `toolbox.go` | tool use: the loop, the text format, and the built-in tools (browsing, the calculator, the sandbox, the uploads) |
| `web.go` | browsing on the standard library — HTML reduced to readable text, and its links |
| `llm.go`, `ollama.go`, `chatgpt.go` | the provider plumbing and the two clients |
| `speech.go`, `vision.go` | speech and images as text |
| `bench.go` | throughput benchmarks |

## Tests

24 `*_test.go` files beside the code they cover.

```bash
cd ..                 # the go/ directory
go test ./...
go test -race ./...   # or, from ../..: make go-test
```
