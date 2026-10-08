# The growing window — a dynamic tokenization: first one letter, then two, then three

**Status** Proposed, 2026-09-28. Measured on the sample corpus with the built
package; not built. `DESIGN.md` §6 — the addressing — is what makes it
possible; this is the argument and the contract for building it.

**Answers** the author's proposal: *a new type of dynamic tokenization,
dynamic meaning the window size — first 1 letter, then 2 letters, then 3
letters, and so on.*

---

## 1. The idea, precisely

Two things grow, and one fact about the addressing lets them.

**The fact.** `base(ℓ) = (R^ℓ − 1) / (R − 1)` does not depend on `L`. So the id
of a sequence is the same at every depth, and **a tree of depth `L` is the
first `base(L + 1)` slots of a tree of depth `L + 1`**. Verified id for id at
`R ∈ {3, 33, 92}` and `L ∈ 1..3`. Growing the window is therefore an append —
`R^(L+1)` new slots on the end of every array — followed by one pass that
counts the new level's substrings of the texts. Nothing already learned moves,
and the grown tree is bit-identical to a tree primed at the new depth and
trained from scratch (measured: equal counts at every rung).

**The window that grows.** The model starts as small as a primed tree can be —
`L = 1`, the root and the single units: 93 slots for phones, 34 for letters, a
unigram — and
deepens one letter of context at a time: `L = 2`, `3`, `4` …, each rung `R`
times the memory of the last, each rung one more letter the model can see. It
grows when it is *ready* (§3.2): when the contexts at its deepest level have
been read often enough to be trusted, the window has been used up, and one more
letter will pay. When they have not, growing costs memory and buys nothing, and
the readiness numbers say so before the held-out set does.

**The token that grows with it.** The walk emits one unit per step, but where
the tree is *certain* — the observed continuation of a context has one choice
— the units run together into one token. At `L = 2` no run is longer than a
letter; at `L = 4`, on sixty sentences, a fifth of positions begin a run of two
or more and `bird ` is one token. A token is one letter at first, then two,
then three: **the tokenization is the window**, read off the tree at every
depth, and `R` never grows — the tree never sees a token, only units, which is
what keeps it primeable. (Byte-pair encoding grows `R`; this grows `L`.)

---

## 2. What was measured

On `../RadixCyclicNN/data/sample_corpus.txt`, 48 training and 12 held-out
sentences, letters (`R = 33`), the default smoothing:

| rung | nodes | held-out bits / char | train bits / char | mean `own` at the deepest context | grow step |
|---|---|---|---|---|---|
| `L = 1` | 34 | 4.132 | 4.142 | 1.00 | — |
| `L = 2` | 1,123 | 3.244 | 3.159 | 0.97 | 1 ms |
| `L = 3` | 37,060 | **3.123** | 2.835 | 0.80 | 10 ms |
| `L = 4` | 1,222,981 | 3.204 | 2.837 | 0.62 | 259 ms |

Each rung was grown from the one above by appending and one pass, and checked
equal to a fresh prime at that depth. The fourth letter of context made
held-out prediction *worse*: the deepest contexts were trusted at 0.62 on
average — read about three times each — and the fold fell through them to the
shallower levels it already had. The readiness number said so: 0.80 at three
letters, 0.62 at four.

The runs the tree is certain of, at smoothing 0 (the family's criterion for a
corridor: the observed continuation has one choice):

| depth | mean run after a context | positions beginning a run ≥ 2 | ≥ 4 | longest |
|---|---|---|---|---|
| `L = 2` | 0.00 letters | 0 % | 0 % | 1 |
| `L = 3` | 0.10 | 1 % | 0 % | 3 |
| `L = 4` | 0.79 | 20 % | 5 % | 7 |

A held-out sentence cut where the tree is certain, at `L = 4`:
`t h e ␣ e a r l y ␣ b i rd␣ c a t c h e s ␣ t h e ␣ w o r m` — one token of
four letters already, on a corpus of sixty sentences. Runs lengthen with depth
and with data; a real corpus is where the tokens become words.

The ladder's cost per rung, `N` nodes at 24 bytes each:

| codec | `L = 1` | `L = 2` | `L = 3` | `L = 4` | `L = 5` |
|---|---|---|---|---|---|
| `phones` (92), the main codec | 93 | 8,557 | 787,245 · 18 MiB | 72,426,541 · 1.6 GiB | — |
| `chars` (33) | 34 | 1,123 | 37,060 | 1,222,981 · 28 MiB | 40,358,374 · 924 MiB |
| `syllables` (1,510) | 1,511 | 2,281,611 · 52 MiB | 3,445,232,611 — no | — | — |
| `gpt2` capped (2,004) | 2,005 | 4,018,021 · 92 MiB | — | — | — |

---

## 3. What is being specified

### 3.1 `grow` — one rung up

```python
def grow(self, texts: Iterable[str] | None = None) -> dict
    # Deepen the model by one level.  N(R, L + 1) must be within the ceiling.
    # 1. new arrays of N(R, L + 1): cnt, plus, minus; the old arrays are their first N(R, L) entries.
    # 2. one pass over the texts read — the journal's (3.3), or the texts given — counting the substrings of
    #    length L + 1 only (the rolling code of one length; the shorter ones are already counted).
    # 3. one pass over the outcomes judged — the journal's — crediting level L + 1 only, with each outcome's
    #    prefix, sign, strength and weight, at the model's rungs (final: the new level is now the final one,
    #    and the old final level's credits are re-read as ordinary levels).
    # 4. L += 1; the address is rebuilt; every cache is dropped.  Returns {"L", "N", "counted", "credited", "seconds"}.
```

The fold, the walks, scoring, the file: unchanged. They read whatever `L` is.
A model grown from `L = 1` to `L = 4` is the model primed at `L = 4`, to the
bit, and the tests hold it there (§6).

### 3.2 Readiness — when to grow

```python
def ready(self) -> dict
    # {"L", "mean_own": the mean own(c) over the deepest contexts the read texts visit,
    #  "trusted": the share of those contexts with own(c) >= 0.5, "next_nodes": N(R, L + 1),
    #  "within_ceiling": bool, "ready": mean_own >= settings.ready and within_ceiling}
```

`Settings.ready = 0.75`: grow when the deepest level's contexts are trusted
three quarters on average — read six times each at `ALPHA = 2`. It is one
number from one corpus and is a setting for that reason. `train(texts,
grow=True)` asks `ready()` after the pass and grows at most once per call, so a
model fed a corpus in pieces climbs the ladder as the data earns it; `grow` by
hand climbs regardless. The ceiling stops the climb either way.

### 3.3 The journal

Growing counts a level the model has not counted, so the model must keep what
it read: `journal.read`, the texts, and `journal.judged`, the outcomes as
`(text, prefix, amount, rungs, weight)`. `train`, `reward` and `punish` append
to it; `grow` replays it. It is in the file (format version 2): a model's file
is the size of its data already, and this is the data. `Settings.journal =
False` keeps none, and `grow` then needs the texts given and cannot re-credit
the reward tree's new level (it says so). A version-1 file loads with an empty
journal.

### 3.4 `tokenize` — the token that grows

```python
def tokenize(self, text: str, certain: float = 1.0, start: bool = True) -> list[str]
    # Cut a text where the tree is certain.  Reading unit by unit with the context the walk would have, a unit
    # JOINS the token before it when its context has been read (ctx > 0) and it is that context's certain
    # continuation: q(x | c) >= certain — 1.0 is the family's corridor (one observed choice; exact at
    # smoothing 0), 0.9 tolerates the smoothing.  Otherwise it BEGINS a token.  Returns the tokens' texts.
```

and on the walks, `predict(..., tokens=True)`: the greedy walk emits its argmax
and keeps emitting while certain, and `PathResult.tokens` groups its units into
the runs — one decision per token, the rest is the corridor. Nothing in either
tree changes: a reward on a token credits the run's steps at every level as
now, and `R` stays what the codec made it.

### 3.5 Down the ladder

The family's dynamic window goes down as well as up — 32, 16, 8, 4 and back.
Here going down is `shrink()`: truncate every array to `N(R, L − 1)` and `L −=
1`; nothing is recounted, because the shallower tree is exactly the prefix.
Offered because it is free; when a model *should* shrink — the deepest level
untrusted, `mean_own < ready / 2`, say — is unmeasured, as it is in the family
(its Q-20).

---

## 4. What it costs

* **Memory** grows `R`-fold per rung; the ladder in §2 is the whole budget and
  the ceiling refuses the rung that would break it.
* **The journal**: the texts and the outcomes in the file. A model of a
  10 MB corpus carries 10 MB; gzip takes a third of it.
* **A rung** is one pass over the journal at the new length: at a million
  units per second (§16 of the design), ten seconds per 10 MB.
* **Nothing at prediction time.** The fold reads `L` levels as it always did.

---

## 5. Where it is reachable

* Python: `PairModel.grow(texts=None)`, `PairModel.ready()`, `PairModel.shrink()`, `train(texts, grow=False)`, `tokenize(text, certain=1.0)`, `predict(..., tokens=True)` and `PathResult.tokens`; `Settings.ready`, `Settings.journal`.
* CLI: `grow --model m.json [--data FILE]`, `ready --model m.json`, `shrink --model m.json`, `train --grow`, `tokenize --model m.json --text "..." [--certain 0.9]`, `predict --tokens`.
* The file: `"journal": {"read": [...], "judged": [[text, prefix, amount, rungs, weight], ...]}`, format version 2.
* `bench grow --data corpus.txt`: the table of §2 for any corpus and codec — bits per unit, readiness and the grow time at every rung up to the ceiling.

---

## 6. Tests

* `test_address.py` — the nesting: `Address(R, L).of(seq) == Address(R, L + 1).of(seq)` for every sequence of length `≤ L`, at several `(R, L)`; `bases` a prefix.
* `test_grow.py` — grown from `L = 1` to `L = 4` equals primed at `L = 4`, counts and rewards, with the journal and with texts given; a `rungs = final` model re-credits correctly; `grow` refuses the rung over the ceiling with its numbers; `shrink` then `grow` is the identity; `ready()` on the sample corpus says yes at `L = 3` and no at `L = 4`; `train(grow=True)` grows at most once per call.
* `test_tokenize.py` — every token's letters are the text's; a certain run is one token and an uncertain step begins one; runs lengthen with `L` on the sample corpus; `certain = 1.0` at smoothing 0 is the corridor exactly; `predict(tokens=True)` groups exactly the greedy walk's units.
* `test_model.py` — the journal round-trips; a version-1 file loads with an empty journal; `journal = False` grows only when given texts.

---

## 7. Decisions and open questions

1. **Which "dynamic tokenization" is meant.** This spec reads it as the *window* growing — the sliding `n` of the family's `Encoding` going 1, 2, 3 — with tokens read off the tree. The other reading is a *chunk codec*: non-overlapping groups of `k` letters as units, the family's `Encoding(n=k, stride=k)`, first `k = 1`, then `2`, then `3`. That needs no new machinery — a `chunks(k)` preset of the codec, a vocabulary of `33^k` — and primes to 33, 1,089 and 35,937 units: at three letters a chunk, only `L = 1`. It is one preset away if wanted; it is not this spec.
2. **`ready = 0.75`** is one number from one corpus. `bench grow` on a real corpus sets it.
3. **Should the reward tree grow with the count tree?** Yes by default — the journal re-credits the new level — or a new level could start clean, with only the counts. The first keeps "an outcome credits every level" true across growth; the second is cheaper and untried.
4. **Going down.** Free, offered, unmeasured (§3.5).
5. **Tokens that become units.** A run the tree is always certain of could be promoted to one unit of the vocabulary — the word-pair merge of the earlier discussion, driven by the tree. That grows `R`, which this spec keeps fixed; it is the next spec, not this one, and the two compose: grow `L` while the data earns it, then merge the runs it is certain of into units and grow again.
