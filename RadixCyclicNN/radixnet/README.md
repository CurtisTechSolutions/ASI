# radixnet

The Python package: a **self-compressing cyclic-graph neural network**. Nodes
hold labels, edges hold weights, prediction is a shortest path, and a unary
chain of nodes merges itself back into one — so the structure compresses as it
learns.

A graph reads text in its own `Encoding` (`radixnet.Encoding`), fixed when it is
created and carried in its file: character trigrams by default, and any n, any
stride (groups of four or five letters) or whole words instead —
`new_model("count", encoding=Encoding(unit=WORDS, n=2))`, or `--encoding
word:2:1` on the CLI. All four model kinds share the graph, so all four have it.

**Standard library only.** `torch` accelerates it if installed and is never
required; Pillow, diffusers and a speech-to-text package are optional extras for
images and speech.

`../README.md` is the user-facing guide, `../DESIGN.md` the specification and
`../DECISIONS.md` the record of why it is that way. This file says which module
does what.

## The core

| module | what it is |
|---|---|
| `graph.py` | `RadixCyclicGraph` — the self-compressing cyclic graph: split, merge, the invariants |
| `activation.py` | the parametric sine `f(x) = a·sin(b(x−h)) + k`, all four parameters learnable per node |
| `encoding.py` | `Encoding` — how a text becomes grams and comes back: the unit (characters or words), the n of the n-gram and the stride (1 = the sliding window, n = non-overlapping groups), with the encoder and decoder halves |
| `attention.py` | the attention band: each gram read sharp at its centre and blurred towards its ends, so a correction's blame and credit land on the gram that has a changed unit at its centre (`AttentionBand`, `spread`, the preview; `../SPEC-AttentionBand.md`) |
| `counter.py` | cyclic counters — the odometer every growing integer in the model runs on, wrapping at `10^15` and counting the reset |
| `backend.py` | the training backends and the one-hop learning rule |
| `backend_torch.py` | the same rule fully vectorised over a mini-batch, on `cuda` / `mps` / `cpu`. Optional |
| `search.py` | shortest-path prediction (Dijkstra) and stochastic sampling over the graph |
| `beam.py` | beam search — the K best **and** the K worst continuations in one prediction |
| `schedule.py` | learning-rate schedules, expressed as a *graph function* of the epoch |

## The models

| module | what it is |
|---|---|
| `model.py` | `RadixNet` — training, prediction, generation, scoring, 2NRL and persistence |
| `countnet.py` | `CountRewardNet` — the count / reward model: a second algorithm on the same graph, weights a dual frequency function of traversal counts plus rewards |
| `wordnet.py` | `WordNGramNet` — the same count / reward model over an alphabet whose symbols are **words**: a word is one code point, the window is still three, and only the encoder and the decoder differ (`../SPEC-WordNGrams.md`) |
| `resonance.py` | `ResonantNet` — the phase model: an analog carrier on the same graph |
| `phasesearch.py` | search over the phase-unrolled graph, with the metacognitive handoff on cycles |
| `metacog.py` | `MetaLayer` — the part that takes over when the walk meets a phase-locked cycle. Its answer is a cost, never a prohibition |

## The negative side

| module | what it is |
|---|---|
| `negative.py` | `NegativeNet` — the failures, and *why* they were failures |
| `blame.py` | where the negative network's data comes from: the tutor's verdicts, turned into blame |
| `diff.py` | character diff between what the network wrote and what the teacher corrected, so only the steps that changed are blamed |
| `duo.py` | the two networks as one output path: the positive model writes, the negative one vetoes |
| `gan.py` | `Evolver` — the GAN-style self-upgrading loop: train on failures, fail on purpose, then invert |
| `critic.py` | the negative network feeding itself — an LLM reviewer on a loop |

## Teaching it

| module | what it is |
|---|---|
| `tutor.py` | automated English lessons: the tutor sets the exercise, the network completes it, the tutor marks it |
| `recall.py` | what the network remembers of what it was shown — the recall tutor for speech and images |
| `dialogue.py` | the model conversing with itself, with stutter detection and backtracking - and the stream of events a conversation is watched through as it happens (`stream=`: the turns as they are spoken, and the window a backtrack may still rewrite between them) |
| `chat.py` | the model in conversation with an LLM, and the LLM marking the conversation |
| `codegen.py` | code generation with a sandbox, an LLM teacher / judge and 2NRL rewards |
| `agent.py` | tool use: the network browses and solves on its own, an LLM sets the bar and teaches |

## Talking to the outside

| module | what it is |
|---|---|
| `llm.py` | provider-independent plumbing for the language models the network talks to |
| `ollama.py` | a local Ollama server: training corpora from a prompt, and adversarial review |
| `chatgpt.py` | ChatGPT over the hosted API, behind the same client interface |
| `tools.py` | the external tools the network can call, and the text format it learns them in |
| `browser.py` | a real browser behind the web tools — Chrome over the W3C WebDriver protocol |
| `mcp.py` | an MCP server: the network's tools *and* the network itself, over the Model Context Protocol |

## Other media

| module | what it is |
|---|---|
| `vision.py` | images as text — the Stable Diffusion image encoder (image generation run backwards) plus base64 |
| `speech.py` | speech as text — what was said *and* the waveform that said it, both behind one unique token |

## Plumbing

| module | what it is |
|---|---|
| `cli.py` | `python -m radixnet <command>` / `radixnet` — every command |
| `__main__.py` | dispatches `python -m radixnet` to the CLI |
| `api.py` | the HTTP JSON API and the static file serving, standard library only |
| `archive.py` | ZIP archives as training data: the text entries of an archive, unpacked safely |
| `checkpoint.py` | checkpoint directory management — rotation, the `latest` pointer, resume |
| `bench.py` | throughput benchmarks for the core |
| `__init__.py` | the public surface of the package |

## Related

* `../tests/` — a `test_*.py` per module, plus a cross-language parity suite
* `../go/radixnet/` — the Go port of the count / reward model and the negative
  network. Model files are interchangeable in both directions
* `../DESIGN.md` §12-13 — the API contract the frontend is written against
