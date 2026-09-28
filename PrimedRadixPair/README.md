# PrimedRadixPair

**Two radix trees, primed with every option, connected at every equal node —
one written by reading, the other by what worked.**

A radix tree that is **primed** — every sequence of `1..L` units over a closed
vocabulary is inserted before the tree sees a single text — never changes its
structure again: training is a counter going up at an address that is
*computed*, not found. A second primed tree over the same sequences is the
**reward tree**: it is written only by the outcomes of what the model produced
— a step rewarded when the output was judged correct, punished when it was
not (a punishment is a negative reward) — and never by a text merely read. The
two are connected wherever they hold an equal sequence, at every level, and the
connection is where a step's two numbers — how often it was read, what it
earned — are read together into one answer. Text goes in and comes out through
an **encoder/decoder** behind a tokenizer of your choosing: characters, bytes,
GPT-2's own tokenizer or a byte-pair encoding of your own size, or the
repository's phonetic tokenizer at the phoneme or the syllable level.

`PRD.md` says what it is for and what counts as success; `DESIGN.md` is the
contract the code is written against; this is the manual, and the numbers.

## Quick start

```bash
cd PrimedRadixPair
make test                                        # 85 tests, seconds; standard library only
make check                                       # prime small trees by brute force and compare with the arithmetic

python3 -m radixpair prime   --model m.json.gz --codec chars --L 4          # 1,222,981 nodes, 28 MiB
python3 -m radixpair train   --model m.json.gz --data ../RadixCyclicNN/data/sample_corpus.txt
python3 -m radixpair reward  --model m.json.gz --text "mat" --prefix "a cat sat on the " --strength 5
python3 -m radixpair punish  --model m.json.gz --text "mat" --prefix "a cat sat on the "
python3 -m radixpair predict --model m.json.gz --prefix "a cat sat on the " --length 3
python3 -m radixpair predict --model m.json.gz --prefix "a cat sat on the " --length 3 --traversal punishment
python3 -m radixpair score   --model m.json.gz --text "the cat sat on the mat" --per-unit
python3 -m radixpair info    --model m.json.gz

# the two codecs decided on
python3 -m radixpair prime --model syl.json.gz --codec syllables --L 2                    # 1,502 syllables of the core lexicon: 2,281,611 nodes
python3 -m radixpair prime --model gpt.json.gz --codec gpt2 --L 2 --top 2000 \
                           --tokenizer-data corpus.txt                                   # GPT-2 capped to the corpus's 2,000 most frequent tokens
```

`--json` on any command prints one JSON document. `make help` lists the
targets; every variable is overridable (`make prime CODEC=syllables L=2`).

**GPT-2's tokenizer** runs in the standard library from its two vocabulary
files, `encoder.json` and `vocab.bpe` (or Hugging Face's `vocab.json` and
`merges.txt`), placed in `data/gpt2/` — `make gpt2-files` fetches them where
the network allows. They are not committed (`data/gpt2/README.md`). The
tokenizer's algorithm is tested on a synthetic vocabulary either way, and
held to `tiktoken`'s `gpt2` encoding id for id when both the files and that
package are present. **The phonetic codecs** read the sibling
`../PhoneticTokenizer` checkout, or the installed `phonetok` package.

## Contents

| file | what it is |
|---|---|
| `PRD.md` | the requirement in the author's own words, the hypothesis, goals and non-goals, the requirements, the success criteria, the phases, the decisions |
| `DESIGN.md` | the specification — the contract every module is implemented against, in 19 sections |
| `radixpair/codec.py` | the encoder/decoder: `chars`, `bytes`, `bpe`, `phones`, `syllables`, `gpt2`, `external`; the marks; the closed, frozen syllable vocabulary; the `top` cap |
| `radixpair/gpt2.py`, `bpe.py` | GPT-2's byte-level BPE from its two files; the repository's own BPE, trained on a corpus |
| `radixpair/address.py` | the arithmetic of a primed tree: `base`, `code`, `id`, `append`, `drop_oldest`, `drop_newest`, `level`, the rolling code |
| `radixpair/count.py`, `reward.py` | the count tree (counts, written by `observe` alone) and the reward tree (`plus` and `minus` kept apart, written by `credit` alone, at every level or the final nodes) |
| `radixpair/pair.py` | the rung — the score of a step under the `reward` and `punishment` traversals —, the fold, scoring, the settings |
| `radixpair/search.py` | the greedy walk of the exact fold (the default), the sampled walk, the cheapest path |
| `radixpair/model.py` | `PairModel`: `train`, `reward` / `punish` / `two_nrl` / `feedback` (marks as weights, a `prefix` that is context and not outcome), `predict`, `generate`, `score`, `weights`, `info`, save / load |
| `radixpair/check.py` | the brute-force oracle: every option inserted literally, then compared with the arithmetic |
| `radixpair/bench.py`, `cli.py`, `checkpoint.py` | the four benches; every command; rotated checkpoints |
| `tests/` | 85 tests, `unittest`, no dependencies |

## What was measured

On this container, pure Python, one process; `bench` reproduces every line.
The corpus is `../RadixCyclicNN/data/sample_corpus.txt` — 60 sentences,
1,800 characters — split 80 / 20 by `bench compare` at seed 0. Small, so the
numbers are the mechanism's, not a language model's.

**Throughput** (S-2: asked for 1 million increments per second):

| codec, `L` | nodes | counting | crediting | units read |
|---|---|---|---|---|
| `chars`, 4 | 1,222,981 | 4.36 M increments/s | 3.98 M entries/s | 1.09 M units/s |
| `phones`, 3 | 787,245 | 3.66 M | 3.23 M | 1.22 M |
| `syllables`, 2 | 2,281,611 | 3.26 M | 3.14 M | 1.63 M |

**Bits per unit on the held-out sentences** (S-4, S-5), `backoff` all against
one fall (`deepest`) against none:

| codec, `L` | smoothing | `all` | `deepest` | `none` | `<unk>` |
|---|---|---|---|---|---|
| `chars`, 4 | 0.5 | **3.204** | 3.748 | 3.604 | 0 % |
| `chars`, 4 | 0 | **2.888** | 3.242 | 3.824 | 0 % |
| `chars`, 3 | 0.5 | **3.123** | 3.410 | 3.318 | 0 % |
| `phones`, 3 | 0.5 | **4.102** | 4.749 | 4.603 | 0 % |
| `syllables`, 2 | 0.5 | **6.494** | 8.469 | 8.357 | 0.3 % |
| `bytes`, 2 | 0.5 | **4.280** | 4.400 | 4.365 | 0 % |

The fold wins every row: the rungs at every level buy between 0.1 and 2
bits per unit over one fall, on a corpus this small. Two things the table also
says. The Jeffreys smoothing that lets a reward lift an unread step costs 0.3
bits per character against the raw fold here (`smoothing = 0` is
`FilterBankRadix`'s fold exactly, `DESIGN.md` §9.3), and three characters of
context beat four: sixty sentences are not enough to trust a four-character
window. Both are expected of the corpus, not of the design, and both are what
`bench compare` on a real corpus will decide.

**The author's case** (S-6, `bench feedback`, `chars, L = 4`, smoothing 0):
`a cat sat on the mat` and `a cat sat on the log` read; the outcome `mat`
after `a cat sat on the ` rewarded once at strength 5 and punished once at
strength 1; `log` never judged.

| | `reward` traversal | `punishment` traversal |
|---|---|---|
| untouched | `log` (1.737) | `log` (1.737) |
| `mat` rewarded 5, punished 1 | **`mat`** (0.411) | **`log`** (1.360) |
| then rewarded 50 more times | `mat` (0.358) | `log` (1.360) — unbuyable |

The sentences are the spec's with `a` for `the`, so that the branch falls
inside a four-character window (`he ` must be followed by `m` or `l` alone);
the outcome is credited with the prefix as context and not as outcome, which
is what the `prefix` of `reward` and `punish` is for. At smoothing 0.5 the
punishment traversal continues `l o`: on two sentences the root has more say
than a once-seen deep step.

**Rewards at every level reach a sibling context** (S-7, `bench rungs`, 30
steps at strength 2, the lift in bits of the rewarded unit's probability):

| codec, `L` | sibling, `rungs = all` | sibling, `rungs = final` | control |
|---|---|---|---|
| `chars`, 4 | **+0.92** | 0.00 | +0.004 |
| `phones`, 3 | **+1.35** | 0.00 | +0.09 |
| `syllables`, 2 | **+1.15** | 0.00 | +1.21 |

A step rewarded under one context becomes about a bit likelier under another
context sharing its last unit, and not at all when the pair is connected at
the final nodes only. At one syllable of context the root is the only level
two contexts share, so the control — a context with a different last unit —
is lifted as much as the sibling: the connections at every level are doing
exactly what they are for, and at `L = 2` that is the root.

**One finding the design did not predict.** The cheapest single path — the
family's search — is not the right decoder on a primed tree: it can pay for a
rare unit at the deepest level, because the unread context it then stands in
falls to the root for nothing, where frequent units are cheap. The exact fold
sums over every fall and is not fooled, so the greedy walk of the fold is the
default and the cheapest path is the option (`DESIGN.md` §10).

## Status

Phases P1 and P2 of `PRD.md` §8 are built: every codec, both trees, the rung,
the fold, the walks, the verbs, persistence, checkpoints, the CLI, the four
benches and the tests. Not built: the judged loops of `RadixCyclicNN` calling
`reward` / `punish` (P3), the least-punished ranking and the beams (P3), the
sine kind (P4), the ports, the API and the frontend (P5).

## Related

* `RadixCyclicNN/radixnet/countnet.py` and `DECISIONS.md` D-022, D-026, D-050 — the count / reward model whose two halves this pair pulls apart into two trees, and the feedback primitives it keeps.
* `RadixCyclicNN/SPEC-LeastPunished.md` and `RadixCyclicNN/DESIGN.md` §31 — why rewards and punishments are kept in separate ledgers, and the two traversals that read them.
* `FilterBankRadix/DESIGN.md` §5 — the context tree whose every-context-length prediction the fold reproduces, and the smoothing constants shared so bits per unit are comparable.
* `RadixCyclicNN/DESIGN.md` §4, §7 — the encoder / decoder halves this codec is modelled on, and the shortest-path search kept as an option here.
* `RadixTrieLLM_RNN/main.py` — the smallest radix insertion in the repository; its splitting logic is what a primed tree never needs, and what `check.py` runs to prove it.
* `PhoneticTokenizer/` — the tokenizer of sounds this reads with, at the phoneme level and at the syllable level.
