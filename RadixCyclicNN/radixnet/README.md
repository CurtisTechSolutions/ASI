# radixnet

The Python package: a **self-compressing cyclic-graph neural network** — the
core model. Nodes hold labels, edges hold weights, prediction is a shortest
path, and a unary chain of nodes merges itself back into one — so the structure
compresses as it learns.

A graph reads text in its own `Encoding` (`radixnet.Encoding`), fixed when it is
created and carried in its file: character trigrams by default, and any n, any
stride (groups of four or five letters) or whole words instead —
`new_model("count", encoding=Encoding(unit=WORDS, n=2))`, or `--encoding
word:2:1` on the CLI. All four model kinds share the graph, so all four have it.

**Standard library only.** `torch` accelerates it if installed and is never
required.

**This package is the model and nothing else.** Everything built around it —
the command line, the HTTP API and the React frontend, the teaching loops, the
LLM clients, the agent and its tools, speech, images and voice, MCP,
checkpoints and the benchmark — is the `modelkit` package in `../../ModelKit`,
which imports this one; nothing here imports it. `python -m radixnet` still
runs the command line: `__main__.py` hands over to `modelkit.cli`.

`../README.md` is the user-facing guide, `../DESIGN.md` the specification and
`../DECISIONS.md` the record of why it is that way. This file says which module
does what.

## The core

| module | what it is |
|---|---|
| `graph.py` | `RadixCyclicGraph` — the self-compressing cyclic graph: split, merge, the invariants |
| `activation.py` | the parametric sine `f(x) = a·sin(b(x−h)) + k`, all four parameters learnable per node |
| `encoding.py` | `Encoding` — how a text becomes grams and comes back: the unit (characters, words or sounds), the n of the n-gram and the stride (1 = the sliding window, n = non-overlapping groups), with the encoder and decoder halves, and the words a model of sounds spells |
| `bpe.py` | the traditional LLM tokenizer the `token` unit reads through: byte-level byte-pair encoding, GPT-2's way - the byte alphabet, the pre-tokenizer, ranked merges, ids with `<|endoftext|>`, an exact decoder, learning merges from a corpus, GPT-2's `merges.txt` format - and the text form a graph over tokens is built from (`The walk ⁀ing cat ⁀.`); the bundled merges are `data/merges.txt` (`../SPEC-Tokens.md`) |
| `attention.py` | the attention band: each gram read sharp at its centre and blurred towards its ends, so a correction's blame and credit land on the gram that has a changed unit at its centre (`AttentionBand`, `spread`, the preview; `../SPEC-AttentionBand.md`) |
| `window.py` | the dynamic window: a ladder of node sizes in the binary number system - 32, 16, 8, 4 and back to 32 - that halves every node longer than the window into two halves with the same data and a heavy connection, and lets compression regrow them at the top (`DynamicWindow`, the ladder; the halving is `graph.py`'s `split_window`; `../SPEC-DynamicWindow.md`) |
| `prune.py` | auto prune: the thresholds under which the graph lets go of an edge - traversed fewer than `min_count` times, or taking less than `min_share` of its node's out-traversals - and when they run (`AutoPrune`; the pruning itself is `graph.py`'s `prune`, which also sweeps the nodes the removals strand; `../SPEC-AutoPrune.md`) |
| `counter.py` | cyclic counters — the odometer every growing integer in the model runs on, wrapping at `10^15` and counting the reset |
| `backend.py` | the training backends and the one-hop learning rule |
| `backend_torch.py` | the same rule fully vectorised over a mini-batch, on `cuda` / `mps` / `cpu`. Optional |
| `search.py` | shortest-path prediction (Dijkstra) and stochastic sampling over the graph |
| `beam.py` | beam search — the K best **and** the K worst continuations in one prediction |
| `penalty.py` | the punishment traversal: a walk priced by what the network was punished for (`../SPEC-LeastPunished.md`) |
| `schedule.py` | learning-rate schedules, expressed as a *graph function* of the epoch |
| `training.py` | the training plan: the order, the curriculum, the replay buffer and the early stop (`../SPEC-SearchAndTraining.md`) |

## The models

| module | what it is |
|---|---|
| `model.py` | `GraphModel` and `RadixNet` — training, prediction, generation, scoring, 2NRL and persistence; the kind registry (`model_kinds`, `new_model`, `load_model`) |
| `countnet.py` | `CountRewardNet` — the count / reward model: a second algorithm on the same graph, weights a dual frequency function of traversal counts plus rewards. Over a word encoding it is the word n-gram model (`../SPEC-WordNGrams.md`): words are an encoding, not a kind |
| `resonance.py` | `ResonantNet` — the phase model: an analog carrier on the same graph |
| `phasesearch.py` | search over the phase-unrolled graph, with the metacognitive handoff on cycles |
| `metacog.py` | `MetaLayer` — the part that takes over when the walk meets a phase-locked cycle. Its answer is a cost, never a prohibition |
| `negative.py` | `NegativeNet` — the failures, and *why* they were failures |
| `diff.py` | character diff between what the network wrote and what the teacher corrected, so only the steps that changed are blamed |

## Plumbing

| module | what it is |
|---|---|
| `__main__.py` | `python -m radixnet`: hands over to the command line in `modelkit.cli` (the one place this package reaches for the kit) |
| `__init__.py` | the public surface of the package |

## Related

* `../tests/` — a `test_*.py` per module; they pass with the kit refused at import
* `../go/radixnet/`, `../rust/` — the Go and Rust ports of the model. Model files
  are interchangeable between all three
* `../../ModelKit/modelkit/README.md` — the kit's modules, built on this package
