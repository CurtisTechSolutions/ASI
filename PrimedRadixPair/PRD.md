# PrimedRadixPair — Product Requirements

**Status** Built through phase P2, 2026-09-28 (v0.4; `README.md` has the numbers). This document says
*what* is being built and *why*, and what will count as success. `DESIGN.md` is
the contract that says *how*; every requirement below names the section that
realises it.

**What changed in v0.2.** The first draft read the second tree as the first
tree read backwards. The author's adjustments replace that: the second tree is
a **reward tree**, written by the outcomes of what the model produced, and the
model reads and writes text through an **encoder/decoder** behind a tokenizer —
the byte-pair kind LLMs use, or the repository's phonetic tokenizer. The mirror
reading, the backward scoring and the meeting walk are gone from this document;
§9.3 records why.

**What changed in v0.4.** The phonetic tokenizer is the **main tokenizer**:
`phones` is the default codec, the model reads phones and writes phones — its
output is the tokenizer's text form, which it reads back unchanged — and the
English the phones spell is a second reading of the same ids (`spell`). Two
phones of context by default (§9.5).

**What changed in v0.3.** Two decisions made and one correction. The first
codecs to try are **GPT-2's tokenizer** and the **syllable level of the phonetic
tokenizer** (§9.5, with the measured sizes); a **punishment is a negative
reward** (§1, §9.7). The correction: a tree of depth `L = 1` holds no context
at all — it is a unigram — so one token of context is `L = 2`, and the budget
table and the text now say so.

---

## 1. The requirement, in the author's own words

> I want to build a new model in a new directory. Two Radix Tree's, connected
> at the final nodes. This is done by "Priming" the Radix Tree with every
> potential option (brute-force insertion of every option), then connect the
> 2nd Radix Tree to the first at every single level/node where the nodes are
> equal (not just the final nodes).

and, on training:

> A few adjustments for training: use an encoder/decoder, support for a
> tokenizer (commonly used in LLMs) as well as my phonetic tokenizer in this
> repo. The second tree is a reward-focused tree that is rewarded based on
> correct outcomes.

Term by term, as this document reads it. §9 lists where a different reading
would change the design, and what it would change.

| Term | Reading |
|---|---|
| **Radix tree** | as everywhere in this repository: a path from the root spells a sequence of units, and a node *is* the sequence spelled so far. `RadixTrieLLM_RNN/main.py` is the smallest example, `RadixCyclicNN/graph.py` the largest. |
| **Priming** | before the tree sees a single text, every possible sequence is inserted. The structure is then complete and **never changes again** — training only ever changes the numbers on the nodes. |
| **every potential option** | every sequence of length `1..L` over the **codec's vocabulary** — characters, bytes, the tokens of a byte-pair encoding, or the sounds of the phonetic tokenizer, plus the marks for the start and end of a text and for a unit outside the vocabulary. `R + R² + … + R^L` sequences. |
| **brute-force insertion** | the requirement is that every option is *present*. A complete tree needs no pointers — the path to a node is its address in base `R` — so the design realises the insertion by arithmetic and keeps a literal brute-force insertion as the oracle the arithmetic is tested against (`DESIGN.md` §6, §15). |
| **the 2nd Radix Tree** | **the reward tree**: the same sequences, primed alike, holding not how often a step was read but what it earned — the rewards and the penalties from the outcomes the model's outputs were judged by. |
| **rewarded based on correct outcomes** | an output the model produced is judged — a tutor's mark, a sandbox, a person's thumb, 2NRL's good and bad texts — and every step of it is rewarded or punished in the reward tree, at every context length; nothing else there ever moves. The count tree is never written by an outcome, the reward tree never by a text merely read. |
| **the reward tree can also be punished — a negative reward** | `punish` is `reward` with a negative strength: the same primitive, the same nodes, the opposite sign. What a step is worth is its net — rewards minus penalties — and that net is what the model reads by default. The tree keeps the two sums apart underneath so the punishment traversal can ask which way the least has gone wrong on without the rewards buying it back (§9.4). |
| **nodes are equal** | two nodes spell the same sequence. Path equality, not label equality (§9.2). |
| **connected at every level** | every node of the count tree has a connection — a *rung* — to its equal in the reward tree, at every depth. The rung is where a step's two numbers — how often it was read, what it earned — are read together into one answer, at every context length the prediction visits; and a reward is written at every context length of the step it judges. |
| **connected at the final nodes** | the special case the idea started from, kept as a setting: `rungs = final` writes and reads rewards only at the deepest sequences. What the connections at every level buy over it is measured, not assumed (S-7). |
| **encoder/decoder** | the **codec**: the encoder turns text into unit ids, the decoder turns ids back into text — the two halves of one tokenizer, as `RadixCyclicNN`'s `Encoding` has its encoder and decoder halves. Which tokenizer is chosen at priming and written into the model file, because a tree's addresses are measured in its units. |
| **a tokenizer commonly used in LLMs** | byte-pair encoding: the repository's own byte-level BPE, trained on the corpus to a chosen vocabulary size and saved with the model; and, as optional extras, a published LLM tokenizer through `tiktoken` or `tokenizers` when either is installed. |
| **my phonetic tokenizer in this repo** | `../PhoneticTokenizer`'s `PhoneticTokenizer` at two levels. `phones`: the phoneme level, whose fixed alphabet of 92 ids — its own marks included — is the vocabulary. `syllables`: the syllable level, whose vocabulary is closed at priming — every syllable of every word the tokenizer's lexicon knows (1,502 with stress in the bundled core lexicon, measured), or every syllable a corpus contains — and frozen, so a syllable outside it reads as `<unk>`. |

If "encoder/decoder" meant instead an input tree and an output tree — one that
reads, one that writes — the design already has that shape: the count tree
learns by reading, the reward tree by acting, and the rung joins them (§9.3).

---

## 2. Why — the hypothesis

Every radix model in this repository so far **grows**. `RadixCyclicNN` creates
a node the first time it sees a gram, splits and merges as it learns, and its
dynamic window halves and regrows what it built; `FilterBankRadix` rebuilds its
trees from their text every round. Growth is where the structural decisions are
made — and every structural decision is a place to be wrong (a merge that hides
a branch point, a split that comes late) and a cost paid on every observation
(a lookup, an allocation, the compression pass).

This model is the other end of the same line: **decide nothing structurally,
ever.** Prime the tree with every option, and an observation is a number going
up at an address that is computed, not found. Three things are claimed to
follow, and each is a requirement in §6.

1. **Learning is counting, and rewarding is the same loop.** With the structure
   fixed and complete, training a text is `L` array increments per unit and
   nothing else — no lookup, no allocation, no split, no merge — and crediting
   a judged output is the same loop over the reward tree with a float. Measured
   while writing the design, in pure Python: 2.4 million increments per second
   (`DESIGN.md` §16).
2. **Every context length is present, so falling back is a walk, not a
   formula.** A grown tree holds the contexts it has seen; a primed one holds
   *every* context of every length up to `L − 1`. A tree of depth `L` reads a
   text of any length by *sliding* — after emitting at the deepest level the
   walk continues from the sequence without its oldest unit — and falling back
   to a shorter memory is the same move taken early, at a price the count tree
   sets. Summed over every depth the walk can fall to, this is the
   every-context-length prediction of `FilterBankRadix/DESIGN.md` §5.3, proved
   by identity and checked to `1e−12` (`DESIGN.md` §9.3).
3. **What was read and what worked are two trees, met at every level.** The
   count tree is written only by reading; the reward tree only by judged
   outcomes. Kept apart, each can be read alone — the punishment traversal
   walks the way the least has gone wrong on, the family's answer to a step
   that is popular and known to be a mistake (`RadixCyclicNN/SPEC-LeastPunished.md`)
   — and read together at the rung they give the family's dual function: the
   share of what was read, times `e` to the reward (`DECISIONS.md` D-022).
   Because the reward tree is primed too, an outcome can reward a sequence the
   corpus never contained. Because the rungs are at every level, a reward
   written at every context length generalises at the shallow ones and
   specialises at the deep.

The cost is stated as plainly, and it has a new edge. A primed tree is `R^L`
big whatever the data, and **the tokenizer sets `R`**: letters and phonemes
prime to three or four units of context; the core lexicon's syllables, or a
byte-pair vocabulary of about a thousand tokens, to one; and GPT-2's full
vocabulary to none — a unigram — unless it is capped to its two thousand most
frequent tokens (§6.2). The choice of tokenizer is the choice of depth.

---

## 3. Goals

| | Goal | Realised in |
|---|---|---|
| G-1 | A model that is **primed, not grown**: its node set is a function of `(vocabulary, L)` alone, fixed at creation. | `DESIGN.md` §6, §7 |
| G-2 | **Two trees** over those nodes — the count tree and the reward tree — **connected at every equal node**, each written by its own kind of event and read together at the rung. | §8, §9 |
| G-3 | An **encoder/decoder** over a chosen tokenizer: characters, bytes, GPT-2's tokenizer, the repository's own byte-pair encoding, and the phonetic tokenizer at the phoneme and the syllable level. | §5 |
| G-4 | The **four verbs** — `train / predict / generate / score` — and the family's **feedback primitives** — `reward / punish / two_nrl / feedback`, marks as weights (D-026, D-050) — on the same CLI shape as `RadixCyclicNN` and `GTMNN`. | §11, §14 |
| G-5 | A **measurement against the nearest existing models** on the same corpus: bits per unit against `FilterBankRadix`'s single tree and `RadixCyclicNN`'s count model. | §7 of this document |
| G-6 | The **rungs' contribution measured, not assumed**: rewards written and read at every level against the final nodes only, and the fold against no fold, on the same numbers. | §7, S-5, S-7 |

## 4. Non-goals — this version

* **Growth of any kind.** No split, no merge, no dynamic window, no node created after priming. A tree that grows is `RadixCyclicNN`.
* **Open vocabularies.** Every codec's vocabulary is closed at priming and frozen. The phonetic tokenizer gives a syllable its id the first time it is read, so the syllable codec closes its vocabulary from the lexicon or a corpus first (§9.5); the word level is not taken, and words as units are `RadixCyclicNN`'s `WordNGramNet`. The 113 million syllables English phonotactics would allow are measured and not taken either (`DESIGN.md` §5.4).
* **Back-propagation.** Nothing in this family multiplies a chain of derivatives, and nothing here starts to.
* **The least-punished ranking and the two beams** of the count model (worst step first; the `K` best and `K` worst continuations). Phase 3, once the two traversals exist.
* **Teachers wired from this side.** The model exposes `reward` and `punish`; the tutor, the critic, the sandbox and the chat loop of `RadixCyclicNN` call them (phase 3). Nothing here calls an LLM.
* **Torch, Go, Rust, the HTTP API and the frontend.** Python and the standard library first; ports once the Python numbers are worth porting.

---

## 5. Who uses it, and for what

The author, in three ways, in this order:

1. **As a language model over letters, sounds or tokens** — trained on the
   corpora the repository already has (`RadixCyclicNN/data/`, the
   Ollama-written corpora, phonetic transcriptions), read through the chosen
   codec, and compared on bits per unit.
2. **As a judged model.** Everything in `RadixCyclicNN` that marks an output —
   the English tutor, the critic, the code sandbox, a person's thumb, the 2NRL
   pairs — ends at `reward` / `punish` (D-026). Here those primitives have a
   tree of their own, so what the judges said can be read apart from what the
   corpus did, walked on its own, and inspected node by node.
3. **As a fast scorer.** Counting needs no learning rate, no epochs and no
   seed; a primed model of a corpus is built in seconds and prices a text by
   what was read and, separately, by what was judged.

---

## 6. Requirements

### 6.1 Functional

| | Requirement | Realised in |
|---|---|---|
| FR-1 | **Priming.** `prime(codec, L)` produces a model holding every sequence of length `0..L` over the codec's vocabulary (`0` is the root). The node count is `(R^(L+1) − 1) / (R − 1)`, a pure function of `(R, L)`; no operation ever adds or removes a node. | `DESIGN.md` §6, §7.1 |
| FR-2 | **The encoder/decoder.** A `Codec` with `encode(text) → ids`, `decode(ids) → text`, a closed vocabulary of `R` ids with its start, end and unknown marks, and `to_dict / from_dict` so the whole tokenizer travels in the model file. Presets: `chars`; `bytes`; `bpe(vocab_size)` — the repository's byte-level BPE trained at priming; `phones` — the phonetic tokenizer at the phoneme level; `syllables` — the same tokenizer at the syllable level, its vocabulary closed at priming from the lexicon or a corpus; `gpt2` — GPT-2's tokenizer from its two vocabulary files, in the standard library, whole or capped to the `top` most frequent tokens of a corpus; and `external` — any other published tokenizer through `tiktoken` or `tokenizers`, optional. | §5 |
| FR-3 | **Two trees, one address space.** The count tree holds a count per sequence; the reward tree holds a reward and a penalty per sequence, kept apart. Every sequence has one id, the same in both trees, so the rung between equal nodes is the same id read twice. | §7.1, §8.1, §9.1 |
| FR-4 | **Training is counting.** `train(texts)` counts every substring of length `1..L` of every padded text, once per occurrence, into the count tree and nowhere else. Deterministic; training a text twice doubles its counts. | §7.2 |
| FR-5 | **Rewarding is the same loop, and a punishment is a negative reward.** `reward(texts, strength, weights, prefix)` adds `strength × weight` to every step of every text **at every context length** (`rungs = all`) or at the deepest only (`rungs = final`); the `prefix` — what the model was given — is context for those steps and earns nothing itself, so an outcome credits what the model *produced*; `punish` is the same call with the sign reversed; `two_nrl(bad, good)` is punish then reward; `feedback(good, bad, good_weights, bad_weights)` dispatches as D-026. Weights are marks (D-050): a `0.9` text earns nine tenths, a `0` is skipped. Nothing in the count tree moves. | §8.2, §9.6 |
| FR-6 | **Prediction.** From a context of up to `L − 1` units: the exact fold over every depth the walk can fall to, one unit at a time (`mode = greedy`, the default — `DESIGN.md` §10 says why the family's cheapest path is not), the cheapest single path (`mode = dijkstra`) and a sampled walk with a temperature (`mode = sample`); under either **traversal** — `reward` (follow the rewards) or `punishment` (the rewards leave the score and only the penalties price a step), with `merit_scale` and `penalty_scale` as in `RadixCyclicNN/DESIGN.md` §31. | §9.2, §9.3, §10 |
| FR-7 | **Scoring.** `score(text)` returns bits per unit under the model's belief, and the reward tree's own readings of the same text: the mean net reward per step and the worst penalty on any step. | §9.5 |
| FR-8 | **Fallback as a setting.** `backoff = all` (the fold; the default), `deepest` (one fall from the given context straight to the uniform) or `none` (the deepest context and the floor), so G-6 can be measured on the same numbers. | §9.6 |
| FR-9 | **Persistence.** A model file holds the codec in full, `L`, the kind, the settings, and **only the counts, rewards and penalties that are not zero**. Priming is recomputed on load. A restored model predicts identically, and the file's size is proportional to what was read and judged, never to `R^L`. | §13 |
| FR-10 | **CLI.** `python3 -m radixpair prime / train / reward / punish / 2nrl / feedback / predict / generate / score / weights / info / bench / check`, `--json` on all of them, `make` targets for each. | §14 |
| FR-11 | **The brute-force oracle.** `check` primes a small tree by *literal* brute-force insertion — every option, one at a time, with the standard radix insertion — and asserts that the arithmetic tree is the same tree. It runs in the test suite at several `(R, L)`. | §6.4, §15 |
| FR-12 | **A second kind on the same nodes** (phase 4): the `sine` kind — the family's per-node sine activation and learnable weights on the count tree's edges, the one-hop rule, the reward tree unchanged as a ledger. | §12 |

### 6.2 Non-functional

| | Requirement | Note |
|---|---|---|
| NFR-1 | **Standard library only.** Python 3.11+. `tiktoken` and `tokenizers` are optional extras for the `external` codec, imported lazily, never required. | as the family's optional extras |
| NFR-2 | **Memory is `R^L` and is budgeted up front.** The table below is the whole budget: 24 bytes per node (a count and two reward numbers); the Python port is for the rows under the default ceiling of 4,194,304 nodes. | `DESIGN.md` §16 |
| NFR-3 | **Speed.** Training and rewarding ≥ 1 million increments per second in pure Python (measured 2.4 M). Prediction by the fold is `O(L · R)` reads per unit; the cheapest-path search is bounded by `max_expansions` and never raises. | §16 |
| NFR-4 | **Determinism.** Same codec, same `L`, same texts and outcomes (and, for sampling and the sine kind, same seed) ⇒ same bytes out, across processes and across a save/load cycle. | §17 |
| NFR-5 | **The file is the data, not the tree.** A model trained on ten sentences and judged on three is a file of a few kilobytes at any `L`. | §13 |

The budget (`N` = nodes per model, at 24 bytes each). `L` is the longest
sequence held; the longest *context* is `L − 1`, so **`L = 1` is a unigram**
and one unit of context is `L = 2`:

| codec | `R` | `L = 1` (no context) | `L = 2` | `L = 3` | `L = 4` |
|---|---|---|---|---|---|
| `chars`: space, a–z, `'` `.` `,` + 3 marks | 33 | 34 | 1,123 | 37,060 · 0.8 MiB | **1,222,981 · 28 MiB** |
| `phones`: the tokenizer's 92 ids | 92 | 93 | 8,557 · 0.2 MiB | **787,245 · 18 MiB** | 72,426,541 · 1.6 GiB — a port's job |
| `syllables` of the core lexicon, stress kept (measured) | 1,510 | 1,511 | **2,281,611 · 52 MiB** | 3,445,232,611 — no | — |
| `syllables` of `sample_corpus.txt` (measured) | 252 | 253 | 63,757 · 1.5 MiB | 16,066,765 · 368 MiB — a port's job | — |
| `bytes` + 3 marks | 259 | 260 | 67,341 · 1.5 MiB | 17,441,320 · 399 MiB — a port's job | — |
| `bpe` of 1,024 tokens + 3 marks | 1,027 | 1,028 | **1,055,757 · 24 MiB** | 1,084,262,440 — no | — |
| `bpe` of 2,048 tokens + 3 marks | 2,051 | 2,052 | 4,208,653 · 96 MiB — just over the ceiling | — | — |
| `gpt2` capped to 2,000 tokens + `<unk>` + 3 marks | 2,004 | 2,005 | **4,018,021 · 92 MiB** | 8,052,114,085 — no | — |
| `gpt2`, the whole vocabulary (50,257) + 3 | 50,260 | **50,261 · 1.2 MiB — a unigram** | 2,526,117,861 · 56 GiB — no | — | — |
| `external` `cl100k_base` (100,277) + 3 | 100,280 | 100,281 · 2.3 MiB — a unigram | 10,056,178,681 — no | — | — |

The largest vocabulary the default ceiling admits: 45 at `L = 4`, 160 at
`L = 3`, 2,047 at `L = 2`, anything at `L = 1`. The rows in bold are the
regimes this version is for, and the two decided in §9.5 come first.

---

## 7. Success criteria

Each is a test or a measurement, and a criterion that fails is recorded as
having failed, in `README.md`, the way `Experiments/` does it.

| | Criterion | How it is checked |
|---|---|---|
| S-1 | **The primed tree is the brute-force tree.** | `check` at `(R, L) ∈ {(2,5), (3,4), (5,3), (7,2)}` and larger in `bench`; FR-11. |
| S-2 | **Throughput** ≥ 1 M increments/s in Python, counting and rewarding alike, at `chars, L = 4`. | `bench --json`. |
| S-3 | **The rungs are the fold.** With `smoothing = 0` and no rewards, the pair's distribution equals `FilterBankRadix/DESIGN.md` §5.3 computed on the same counts, to `1e-9`, for every context length, and sums to 1. | `test_pair.py`; the identity is `DESIGN.md` §9.3. |
| S-4 | **Bits per unit** on a held-out split of `RadixCyclicNN/data/sample_corpus.txt` at `chars, L = 4`: reported beside `FilterBankRadix`'s single tree at the same `L` (at `smoothing = 0` the expectation is *equality* — it is the same predictor — and a gap is a bug; the default smoothing is reported beside it) and beside `RadixCyclicNN`'s count model at its default trigram. | `bench compare`, numbers committed in `README.md`. |
| S-5 | **The fold buys something.** `backoff = all` beats `deepest`, which beats `none`, in bits per unit on the held-out split; the size of each gap is the finding either way. | `bench compare`. |
| S-6 | **Rewards move the walk, and the two trees can be read apart.** The author's own case (`SPEC-LeastPunished.md` §1), fitted to a four-character window: `a cat sat on the mat` and `a cat sat on the log` read; the outcome `mat` after the prefix `a cat sat on the ` rewarded once at strength 5 and punished once at strength 1, `log` never judged. The `reward` traversal continues `mat`; the `punishment` traversal continues `log`; and rewarding `mat` fifty times more changes neither answer. Run at `smoothing = 0`, as the grown graph the case comes from. | `test_model.py`, `test_bench.py`, `bench feedback`. |
| S-7 | **Rewards at every level generalise.** A step rewarded under one context becomes likelier under a *different* context that shares its last unit with `rungs = all`, and not with `rungs = final`; the size of the effect, and its cost on contexts sharing nothing, are reported. This is the experiment the connections at every level exist to run. | `test_pair.py`, `bench rungs`. |
| S-8 | **Round trip.** Save, load, identical predictions and scores; the file holds no zero count, reward or penalty; its size grows with what was read and judged and not with `L`. | `test_model.py`. |
| S-9 | **The codecs are faithful.** Every codec decodes what it encodes (normalisation aside); the BPE reaches its vocabulary size and round-trips arbitrary bytes exactly; `phones` and `syllables` agree with `phonetok` token for token, and a syllable outside the frozen vocabulary reads as `<unk>`; `gpt2` agrees with `tiktoken`'s `gpt2` encoding id for id on every text in the test corpus when `tiktoken` is installed; `external` agrees with its library id for id. | `test_codec.py`. |

---

## 8. Phases

| Phase | Delivers | Done when |
|---|---|---|
| **P0** | this document and `DESIGN.md` | reviewed; the decisions in §9 made |
| **P1 — the core** | `codec.py` with `chars`, `syllables`, `phones` and `gpt2` — the two codecs to try first, plus the two the tests are cheapest on; `address.py`; `count.py`; `reward.py`; `pair.py` — the rung, the fold, the shift, scoring; `check.py`; the tests of §15 for these | S-1, S-3, S-7 (the test half), S-9 (four codecs) green |
| **P2 — the verbs** | `search.py` under both traversals, `model.py`, `checkpoint.py`, `cli.py`, `bench.py`, the `Makefile`; `bytes`, `bpe.py` and the `external` codec; the measurements S-2, S-4, S-5, S-6, S-7, S-8 committed in `README.md`, on `syllables, L = 2` and `gpt2 top 2000, L = 2` first | every criterion of §7 has a number |
| **P3 — the judged loops** | `feedback` wired as a reward source from `RadixCyclicNN`'s tutor, critic, sandbox and chat (they call `reward` / `punish`, D-026); the least-punished ranking; the two beams | the tutor's marks reach the reward tree and `bench feedback` reports what they did |
| **P4 — the sine kind** | `activation.py`, `sine.py`: learnable weights and activations on the count tree's edges, the one-hop rule, the reward tree unchanged; the same measurements as P2 | the count kind's numbers are matched or the gap is explained |
| **P5 — ports and surfaces** | Go and Rust with bit-identical files, the HTTP API, the frontend — only if P2–P4's numbers justify them | parity tests as in `RadixCyclicNN/tests/` |

---

## 9. Decisions

Each has a recommendation; the design is written to the recommendation, and
says what changes if the decision goes the other way. Two are already made.

**9.1 What is an "option"?** *Recommended: every sequence of length `1..L`
over the codec's vocabulary.* The alternative is every entry of a **lexicon** —
every word, or every pronunciation — inserted into a radix tree. That tree is
not complete, so its addresses are stored rather than computed and radix
compression applies again; everything after the addressing — the two trees,
the rung, the fold, the file — is unchanged. Kept as the design's first open
question; nothing in P1 forecloses it.

**9.2 What makes two nodes equal?** *Recommended: they spell the same
sequence.* The same *last unit* regardless of path would connect every `t` in
one tree to every `t` in the other — `(R^(ℓ−1))²` rungs per unit per level, and
a connection that says "the same symbol in any context", which is the tree
thrown away. Rejected.

**9.3 What is the second tree? — decided: the reward tree.** The first draft
read it as the same texts backwards, because that gave the two trees different
information about one sequence and made falling back a walk through the second.
The author's adjustment settles it the other way: the second tree holds what
the model's outputs *earned*, written by outcomes and by nothing else. What that
gives up: the backward reading of a text and the meeting walk built on it. What
it gives: the rung's two numbers are genuinely different *kinds* of thing — what
the corpus did and what a judge said — which is the split `RadixCyclicNN`'s
count / reward model makes inside one weight and its punishment traversal has
to undo; here it is the structure. Falling back is now a shift the address
arithmetic provides, not a corridor (`DESIGN.md` §9.4).

**9.4 Two node sets with rungs, or one address space with two trees?**
*Recommended: one address space.* Every sequence has one id; the count tree
writes one integer at it, the reward tree two floats — the reward and the
penalty **kept apart**, because their net loses exactly the information the
punishment traversal needs (`SPEC-LeastPunished.md` §1: a step rewarded five
times and punished once must not read like one rewarded four times and never
punished). Separate node sets would store the same addresses twice.

**9.5 Which codec, at what depth? — decided: the phonetic tokenizer is the
main one, phones in and phones out, at `L = 3`.** GPT-2's tokenizer and the
syllable level stay as options. The table in §6.2 says what each allows, and
the numbers were measured with the repository's own tokenizer:

| regime | vocabulary | `R` | `L` | nodes | what it is |
|---|---|---|---|---|---|
| **`phones`, the default** | the tokenizer's 92 ids | 92 | **3** | 787,245 · 18 MiB | two phones of context; the model's output is phones, spelled into English on request |

| `syllables`, the core lexicon, stress kept | 1,502 syllables + the tokenizer's 8 specials | 1,510 | **2** | 2,281,611 · 52 MiB | one syllable of context: the language-model regime to measure first |
| `syllables`, stress dropped | 1,389 + 8 | 1,397 | 2 | 1,953,007 · 45 MiB | the same, a little smaller |
| `syllables`, the corpus's own | e.g. 244 + 8 for `sample_corpus.txt` | 252 | 2 (3 is 16 million) | 63,757 · 1.5 MiB | a small corpus's syllables; `L = 3` only under 160 syllables |
| `gpt2`, capped to the corpus's 2,000 most frequent tokens | 2,000 + `<unk>` + 3 marks | 2,004 | **2** | 4,018,021 · 92 MiB | one GPT-2 token of context: the LLM-tokenizer regime |
| `gpt2`, the whole vocabulary | 50,257 + 3 | 50,260 | 1 | 50,261 · 1.2 MiB | **no context — a unigram over tokens.** Useless as a language model; the cheapest bed for the reward tree over an LLM's tokens |
| `gpt2`, one token of context, uncapped | | 50,260 | 2 | 2,526,117,861 · 56 GiB | not this version, and not Python |

The full CMU dictionary's syllables (tens of thousands) and the 113,530,725
syllables English phonotactics would allow prime no deeper than `L = 1` and are
not taken (`DESIGN.md` §5.4). *Recommended first runs: `syllables` from the
core lexicon at `L = 2`, and `gpt2` capped at 2,000 at `L = 2`; the whole GPT-2
vocabulary at `L = 1` only to exercise the reward tree.*

**9.6 The start and end of a text.** *Recommended: symbols of the vocabulary.*
`<s>` and `</s>` are units, so the first unit of a text is predicted from a
context that says it is first, as `RadixCyclicNN`'s `START → …` does. The
phonetic tokenizer has its own `<s>`, `</s>` and `<unk>`, and they are used as
they are; the other codecs put the three marks first.

**9.7 Does a reward also count a reading? — and a punishment is a negative
reward.** *Recommended: no reading on a reward.* The family's count model
traverses *and* rewards on a thumbs up. Here the two trees are the point:
`reward` writes the reward tree only, so what was read and what was judged
never mix before the rung. `reward(..., read=True)` counts the text as well,
and reproduces the family's behaviour for a comparison. A punishment is
`reward` with a negative strength — one primitive, one sign — and the net is
what a step is worth; the two sums are kept apart underneath for the reason in
§9.4, and dropping the second sum would drop the punishment traversal with it.

**9.8 What decides how much a context is trusted?** *Recommended: the count
tree alone.* `own(c)` — the share of the answer a context keeps for itself
before falling back — is a function of how often the context was *read*. How
often it was *judged* could add to it; that is an open question, not a default.

**9.9 The name.** `PrimedRadixPair/`, package `radixpair` — settled.

---

## 10. Risks

| Risk | Where it bites | What is done about it |
|---|---|---|
| **The tokenizer decides the depth.** GPT-2's full vocabulary primes to a unigram; one token of context over it is 56 GiB. The full CMU dictionary's syllables would be tens of thousands and prime no deeper. | §6.2, S-4 | the table, a hard refusal above the ceiling at `prime` time, the `top` cap that keeps a corpus's most frequent tokens and reads the rest as `<unk>`, the core lexicon's syllables measured at 1,502, and the ports for one level more |
| **Rewards are sparse.** A few hundred judgements over a million nodes leave the reward tree mostly zero. | S-6, S-7 | the rungs at every level are the answer — shallow nodes accumulate what deep ones cannot — and S-7 measures whether they are |
| **Smoothing changes the numbers.** The Jeffreys smoothing that lets a reward lift an unread step (`DESIGN.md` §9.2) makes the fold differ from `FilterBankRadix`'s. | S-3, S-4 | the identity and the comparison are measured at `smoothing = 0` and the default reported beside them |
| **The repository's BPE is not a published one.** Its merges are trained here; no byte-identity with any LLM's tokenizer is promised. | FR-2 | the `external` codec is for byte-identity, at `L = 1` |
| **`<unk>` swallows the text.** A closed vocabulary on real text may route too much through `<unk>`. | S-4 | `info` reports the share of `<unk>` in what was trained; `chars` is widened, or `bytes` / `bpe` chosen, if it is large |

---

## 11. Open questions

Carried into `DESIGN.md` §19, where they are stated against the mechanism each
one questions. The ones a reader of this document should know exist: whether
priming beats growing on *little* data (the hypothesis in §2, and the cheapest
thing to measure once P2 runs); whether judged evidence should count toward how
much a context is trusted (§9.8); whether the reward and the count should share
one smoothing or the reward tree needs its own; whether one syllable, or one
capped GPT-2 token, of context is a language model worth having, or whether
the model's home is phonemes at `L = 3`; and how the full dictionary's
syllables are to be primed at all.
