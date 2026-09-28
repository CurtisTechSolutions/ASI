# PrimedRadixPair — Design Specification

**Two radix trees, primed with every option, connected at every equal node —
one written by reading, the other by what worked.**

A radix tree that is **primed** holds every sequence of `1..L` units over a
closed vocabulary before it sees a single text and never changes its structure
again. The **count tree** is written by reading: how often each sequence
occurred. The **reward tree** is written by outcomes: what each step of the
model's own outputs earned when a judge said the output was right or wrong. The
two are connected wherever they hold an equal sequence, at every level, and the
connection is where a step's two numbers are read together. Text goes in and
comes out through a **codec** — an encoder/decoder behind a tokenizer. This
document is the contract the code is written against - phases P1 and P2 of it
are built; `README.md` has what they measured. Read it fully before changing
code. `PRD.md` says what it is for and what will count as success; this
says what exactly the code must do.

Directory: `PrimedRadixPair/`. Python package: `radixpair`. Python 3.11+,
**standard library only**; `tiktoken` and `tokenizers` are optional extras for
one codec, imported lazily and never required. Deterministic given its inputs.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| **Two Radix Trees** | The **count tree** (§7) and the **reward tree** (§8), over the same primed sequences: a path from the root spells a sequence forwards, and a node *is* the sequence spelled so far. Both are radix trees in the sense of every other one in this repository. |
| **"Priming" the Radix Tree with every potential option** | `prime(codec, L)`: every sequence of length `0..L` over the codec's `R` ids exists from the start, as a slot in flat arrays. `N = (R^(L+1) − 1) / (R − 1)` nodes, a function of `(R, L)` alone. Nothing is ever added or removed. |
| **brute-force insertion of every option** | A complete `R`-ary tree needs no pointers: the path to a node, read as a number in base `R` — the *radix* — is its address (§6). The literal brute-force insertion — every sequence, one at a time, with the standard radix insertion of `RadixTrieLLM_RNN/main.py` — is kept in `check.py` as the oracle the arithmetic is tested against (§6.4, §15). |
| **connect the 2nd Radix Tree to the first where the nodes are equal** | Two nodes are equal when they spell the same sequence. Every sequence has exactly one id, the same in both trees, so the connection — the **rung** — is the same id read twice: once for its count, once for its reward (§9.1). |
| **at every single level/node, not just the final nodes** | A prediction visits every context length from the deepest down to the root (§9.3), and at each one the rung combines the count tree's share with the reward tree's reward into one answer (§9.2). A judged step is credited at every context length (§8.2). `rungs = final` keeps only the deepest level's rungs, so what the rest buy is measured (§9.6). |
| **the second tree is reward-focused, rewarded based on correct outcomes — and can be punished, a negative reward** | The reward tree is written only by `reward` / `punish` — the family's primitives (D-026), marks as weights (D-050) — and never by `train`. A punishment is `reward` with a negative amount: the same loop, the same nodes, the opposite sign (§8.2), and what a step is worth is its net. Underneath, a step keeps its **reward** and its **penalty** as two sums kept apart, because their net loses what the punishment traversal needs (`SPEC-LeastPunished.md` §1). |
| **use an encoder/decoder** | `Codec` (§5): `encode(text) → ids`, `decode(ids) → text`, the two halves of one tokenizer, chosen at priming and saved with the model — as `RadixCyclicNN`'s `Encoding` has its `Encoder` and `Decoder` halves. |
| **a tokenizer commonly used in LLMs, and my phonetic tokenizer** | `gpt2` — GPT-2's own tokenizer from its two vocabulary files, in the standard library, whole or capped to a corpus's most frequent tokens (§5.5); `bpe` — the repository's byte-level byte-pair encoding, trained at priming to a chosen vocabulary size (§5.3); `phones` and `syllables` — `../PhoneticTokenizer` at the phoneme level, its fixed alphabet of 92 ids as the vocabulary, and at the syllable level, its vocabulary closed at priming from the lexicon or a corpus (§5.4); `external` — any other published tokenizer through `tiktoken` or `tokenizers`, optional (§5.6). And `chars` and `bytes` for the small alphabets. |
| A model of the family | `train / predict / generate / score` and `reward / punish / two_nrl / feedback`; shortest-path prediction (Dijkstra, as `RadixCyclicNN/DESIGN.md` §7); the reward and punishment traversals (`RadixCyclicNN/DESIGN.md` §31); JSON model files; a CLI; `unittest` with no dependencies. |
| The score is the family's dual function | At a node, `P(x) ∝ share^share_scale · e^(reward_scale · reward)`: the edge's share of its node's traversals times `e` to the reward (D-022, without the recency window). Here the share comes from one tree and the reward from the other, and the rung is where they meet (§9.2). |
| No back-propagation | The count kind has no gradient at all. The sine kind's rule (§12) is the one-hop rule of `FilterBankRadix/DESIGN.md` §5.4 on the count tree's edges; the reward tree is a ledger, never a gradient's target. |

---

## 2. What this is, against what exists

**Grown against primed.** `RadixCyclicNN`'s graph and `FilterBankRadix`'s
trees are *grown*: a node exists because a text put it there, unary chains are
compressed away, and the structure is a record of what was seen. A primed tree
is the opposite end of that line: every context of every length up to `L − 1`
exists before any text, so no observation ever makes a structural decision.
There is nothing to compress — a complete tree has no unary chains — and no
lookup: the address of a sequence is arithmetic on its units (§6). What a grown
tree spends on structure, a primed tree spends on memory: `R^L` slots whatever
the data (§16).

**The count / reward model, pulled apart.** `RadixCyclicNN`'s count model keeps
one number per edge that adds `log` of a share to a reward; its punishment
traversal then has to take the reward back out to ask which way the least has
gone wrong on (`RadixCyclicNN/DESIGN.md` §31). Here the split is the structure: one tree is written
by reading and one by judging, and the rung is the one place they are added.
Each tree can be read alone, inspected node by node, and saved sparsely.

**The rungs are the fold.** Falling back from a context to a shorter one is a
shift the address arithmetic provides (§9.4), and summed over every depth the
walk can fall to it is exactly the every-context-length prediction of
`FilterBankRadix/DESIGN.md` §5.3 — at the settings where the two coincide
(§9.3). That identity is the design's main check: whatever the search does, the
exact answer is known and the pair must give it. What the reward tree adds is
a second term at every level of that fold.

**What it is not.** It is not a de Bruijn graph and not `RadixCyclicNN`'s gram
graph, though at a fixed depth all three encode a sliding window: those hold
one order and the seen grams; this holds every order and every gram. It is not
a policy-gradient method: a reward here is a number written on a node, never a
gradient's target.

---

## 3. Notation

| symbol | meaning |
|---|---|
| `Σ`, `R` | the codec's vocabulary and its size, marks included |
| `START`, `END`, `UNK` | the codec's ids for the start of a text, its end and a unit outside the vocabulary. `0, 1, 2` for `chars`, `bytes`, `bpe` and a capped `gpt2`; `V, V+1, V+2` for a whole `gpt2` and for `external`; the phonetic tokenizer's own `2, 3, 1` for `phones` and `syllables`, whose `<pad>` (`0`) is a *dead* id |
| `Σ_out`, `R'` | what a walk may emit: `Σ` without `START` and without the dead ids; its size |
| `L`, `D` | the longest sequence held, and `D = L − 1` the longest context |
| `s`, `\|s\|`, `ε` | a sequence of ids, its length, and the empty sequence (the root, id `0`) |
| `s·x` | append `x`: the child of `s`, and **the step "`x` after `s`"** — in a tree the edge into a node is unique, so a step and its node are one thing |
| `s[1:]` | drop the oldest unit: the shift (§9.4) |
| `base(ℓ)`, `code(s)`, `id(s)`, `N` | the addressing of §6; `N = base(L + 1)` |
| `cnt[i]` | how often sequence `i` occurred as a substring of the padded training texts (the count tree) |
| `plus[i]`, `minus[i]` | the rewards and the penalties the step `i` received (the reward tree), both `≥ 0`, kept apart |
| `reward(i)`, `penalty(i)` | `plus[i] − minus[i]` and `minus[i]` |
| `ctx(s)` | `Σ_{x∈Σ_out} cnt[s·x]`: how often `s` was a context |
| `share(x\|s)` | `(cnt[s·x] + σ) / (ctx(s) + σ·R')`, `σ = smoothing` (default `0.5`, Jeffreys, as D-022): the step's share of its context's traversals |
| `own(s)` | `ctx(s) / (ctx(s) + ALPHA)`, `ALPHA = 2`: the share of the answer a context keeps for itself before falling back |
| `FLOOR` | `0.02`, the uniform share mixed into every final answer. `ALPHA` and `FLOOR` are `FilterBankRadix`'s, so bits per unit are comparable (its invariant 5) |
| `share_scale`, `reward_scale`, `merit_scale`, `penalty_scale` | the scales of the two traversals (§9.2), all default `1` |
| a padded text | `START u₁ … u_T END` — what `train`, `reward` and `score` see |

---

## 4. Package layout

```
PrimedRadixPair/
  PRD.md                 what and why; the decisions and the success criteria
  DESIGN.md              this file — the contract
  README.md              the front page; the numbers, once there are numbers
  Makefile               make test | check | prime | train | reward | punish | feedback | predict | generate | score | info | bench | …
  radixpair/
    __init__.py          the public surface: Codec and the presets, PairModel, prime, load_model, Score, PathResult
    __main__.py          python3 -m radixpair -> cli.main()
    codec.py             the encoder/decoder: Codec, chars, bytes_, phones, syllables, gpt2, external; the marks
    gpt2.py              GPT-2's byte-level BPE from encoder.json + vocab.bpe: the byte table, the pre-tokenizer, the merges
    bpe.py               the byte-level byte-pair encoding: train, encode, decode
    address.py           the arithmetic of a primed tree: base, code, id, append, drop_oldest, drop_newest, level
    count.py             CountTree: the counts, observe(), ctx / share / own
    reward.py            RewardTree: plus and minus, credit() at every level, reward / penalty, invert
    pair.py              RadixPair: the rung (the score of a step under a traversal), the fold, the shift, scoring
    search.py            Dijkstra, greedy and sampling over the pair
    model.py             PairModel: the verbs, the feedback primitives, settings, kinds, persistence
    checkpoint.py        CheckpointManager: rotation, the latest pointer, resume
    check.py             the brute-force oracle: prime a small tree literally and compare
    bench.py             throughput; compare (bits per unit against the references); feedback; rungs
    cli.py               python3 -m radixpair <command>
    activation.py        (phase 4) the parametric sine and its partials
    sine.py              (phase 4) the sine kind: parameters on the count tree, the one-hop rule, invert
  tests/
    __init__.py          puts the project root on sys.path
    test_codec.py  test_bpe.py  test_address.py  test_count.py  test_reward.py  test_pair.py  test_search.py
    test_model.py  test_checkpoint.py  test_cli.py  test_bench.py  test_sine.py (phase 4)
```

---

## 5. `codec.py`, `bpe.py` — the encoder/decoder

### 5.1 The contract

A codec is one tokenizer's two halves plus what the tree needs to know about
its ids. It is chosen at priming, saved whole in the model file, and never
changed afterwards: the tree's addresses are measured in its units. **The
phonetic tokenizer is the main one**: `phones` is the default preset, the
model reads phones and writes phones — `decode` gives the tokenizer's text form,
`DH AH0 # K AE1 T .`, which `encode` reads back unchanged, so a prefix may be
given in English or in phones — and `spell` gives the English the phones spell.
For every other preset `spell` is `decode`.

```python
class Codec:
    name: str                       # "chars" | "bytes" | "bpe" | "phones" | "syllables" | "gpt2" | "external"
    R: int                          # the vocabulary size, marks included
    start: int ; end: int           # the ids of the marks
    unk: int | None                 # the id of "a unit outside the vocabulary"; None where every input is a unit (bytes)
    dead: frozenset[int]            # ids never read and never emitted (a tokenizer's <pad>)
    def encode(self, text: str) -> list[int]         # the encoder: units only, no marks; nothing outside Σ survives (-> unk)
    def decode(self, ids: Iterable[int]) -> str      # the decoder: marks and dead ids dropped; unk rendered as "?" where the tokenizer has no rendering
    def spell(self, ids: Iterable[int]) -> str       # the words the ids spell: the English behind phones or syllables; decode() elsewhere
    def join(self, prefix: str, text: str) -> str    # a prefix and its continuation as one text: concatenation, or two words apart for the phonetic codecs
    def padded(self, text: str) -> list[int]         # [start] + encode(text) + [end]
    def emits(self) -> list[int]                     # Σ_out, in id order
    def symbol(self, i: int) -> str                  # the text of one id, for display and the graph view
    def describe(self) -> str
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> Codec
```

`PairModel.encoder` and `PairModel.decoder` are the two halves as objects —
`encoder.encode(text)`, `decoder.decode(ids)` — so the model reads as the
family's do; both are views of the one codec.

### 5.2 The presets

| preset | vocabulary | `R` | encode / decode |
|---|---|---|---|
| `chars()` | 3 marks, then space, `a`–`z`, `'`, `.`, `,` | **33** | `casefold`; curly quotes to `'`; a run of whitespace to one space; everything else to `unk` |
| `bytes_()` | 3 marks, then the 256 byte values | **259** | UTF-8 bytes; `unk` is never produced (`unk = None`); decode with replacement for a walk that is not valid UTF-8 |
| `bpe(vocab_size=1024)` | 3 marks, then `vocab_size` tokens: the 256 bytes and `vocab_size − 256` merges (§5.3) | `vocab_size + 3` | byte-level BPE; `unk` is never produced |
| `phones(stress=True)` — **the default** | the phonetic tokenizer's fixed alphabet at the phoneme level: `<pad> <unk> <s> </s> # , . ?` then the 84 ARPAbet symbols of `cmudict.symbols` (§5.4) | **92** | `PhoneticTokenizer.encode` in; the phone text out (`decode`), the English on request (`spell`); `stress=False` keeps the same 92 ids and uses fewer of them |
| `syllables(stress=True, vocabulary="lexicon", top=None)` | the same 8 specials, then every distinct syllable token of the source — the tokenizer's lexicon (**1,502** with stress in the bundled core lexicon, 1,389 without; measured) or the tokenizer-data corpus — sorted, frozen (§5.4) | `syllables + 8` | the tokenizer at `level="syllable"` with a frozen `Vocab`: `encode(text, grow=False)`, so a syllable outside the vocabulary is `<unk>`; `decode` spells the sounds back |
| `gpt2(files=(encoder.json, vocab.bpe), top=None)` | GPT-2's 50,257 tokens, then the 3 marks appended; or, capped, the `top` most frequent tokens of the tokenizer-data corpus renumbered `3..top+2` after the marks, with `<unk>` for the rest (§5.5) | `50,260`, or `top + 3` | GPT-2's byte-level BPE in the standard library from the two files of the original release; verified against `tiktoken` when it is installed |
| `external(spec)` | any other published tokenizer's vocabulary, then the 3 marks appended (§5.6) | `V + 3` | the library's own; optional |

Every preset is deterministic and normalises before it maps, so
`decode(encode(text))` is the normalised text (`test_codec.py`, S-9).

### 5.3 `bpe.py` — the repository's byte-pair encoding

Byte-level, the algorithm the LLM tokenizers use, in the standard library and
sized to what priming can afford.

* **Pieces.** A text is cut into pieces before any merge: a piece is a run of
  whitespace, or one optional leading space followed by a run of non-space
  characters. Merges never cross a piece, so a token never spans two words.
* **Training** (`train(texts, vocab_size)`): every piece becomes its UTF-8
  bytes (ids `0..255` inside the tokenizer). Repeat `vocab_size − 256` times:
  count adjacent pairs over all pieces (each piece weighted by how often it
  occurs), merge the most frequent pair into a new id — ties broken by the
  smaller pair, so training is deterministic — and record `(a, b) → id` in
  rank order. The merge list *is* the tokenizer. Training runs once, at
  `prime`, on the training texts or the `--tokenizer-data` file, and takes
  seconds on a megabyte of text; a pair-count that is updated incrementally
  rather than recounted keeps it there.
* **Encoding**: per piece, bytes to ids, then repeatedly merge the adjacent
  pair with the lowest rank until none applies; a per-piece cache makes a
  repeated word one lookup.
* **Decoding**: the tokens' byte strings concatenated, UTF-8 decoded with
  replacement.
* **Round trip**: `decode(encode(b)) == b` for any byte string, because every
  byte is a token (S-9).

Its merges are trained here; it is byte-identical to no published tokenizer,
and does not claim to be — `external` is for that.

### 5.4 `phones` and `syllables` — the phonetic tokenizer

`PhoneticTokenizer(level, stress=stress)` from `../PhoneticTokenizer`
(`phonetok`), imported lazily; `test_codec.py` skips both codecs when the
package is not importable. The tokenizer's settings (`level`, `stress`,
`boundaries`, `pauses`) are saved in the codec block (§13) so a model reads
exactly as it was primed, and its own specials are used as they are: `<pad> = 0`
is dead, `<unk> = 1`, `<s> = 2`, `</s> = 3`, `# = 4` (the gap between words),
`, . ? = 5..7` (the pauses).

**`phones`** is the phoneme level, and the main tokenizer. Its vocabulary is
**frozen** by the tokenizer itself — 92 ids, the specials then the 84 ARPAbet
symbols, the same on every machine — which is what makes it primeable as it
stands. `encode` is `tok.encode(text, grow=False)`, and it reads English or
phones alike, because the tokenizer's text form is idempotent; `decode` is that
text form — the units joined by spaces, `DH AH0 # K AE1 T .` — so the model's
output is phones and `encode(decode(ids)) == ids`; `spell` is `tok.decode(ids)`,
the English the phones spell. It primes to `L = 3` — two phones of context,
787,245 nodes — under the default ceiling (`DEFAULT_L`), and `L = 4` is a
port's job (§16.1).

**`syllables`** is the syllable level (`decode` and `spell` as for phones),
where the tokenizer gives a syllable (`K.AE1.T`) its id the first time it reads one. "Every option" over an open
vocabulary is not a thing, so the codec **closes the vocabulary at priming**
and freezes it: the 8 specials, then every distinct syllable token of a source,
in sorted order so that priming is deterministic, as `Vocab(tokens,
frozen=True)`. Two sources: `vocabulary="lexicon"` — every syllable of every
word the tokenizer's lexicon knows, which is "every potential option" for the
language as the tokenizer knows it — and `vocabulary="corpus"` — every
syllable the tokenizer-data texts contain. `top=K` keeps the `K` most frequent
syllables of the corpus and reads the rest as `<unk>`. A syllable outside the
vocabulary is `<unk>` (`encode(text, grow=False)`), and `decode` drops it.
Measured with the package while writing this: the bundled core lexicon (2,057
entries) holds **1,502** distinct syllable tokens with stress and 1,389
without, so `L = 2` — one syllable of context — is 2,281,611 nodes; the
sample corpus holds 244. The full CMU dictionary (`pip install cmudict`, some
135,000 words) holds tens of thousands and would prime no deeper than `L = 1`
uncapped, and the 113,530,725 syllables the tokenizer's own onset and coda
rules would allow (77 legal onsets × 45 stressed vowels × 32,765 legal codas)
are not a vocabulary anything primes over. The word level is not taken.

### 5.5 `gpt2` — GPT-2's tokenizer, in the standard library

GPT-2's byte-level byte-pair encoding, run from the two files of the original
release — `encoder.json` (token → id) and `vocab.bpe` (the merges in rank
order), the same content as Hugging Face's `vocab.json` and `merges.txt` —
given by path at priming and copied into the codec block whole (about a
megabyte, gzip takes it to a quarter). `gpt2.py` carries the three parts of the
algorithm: the reversible byte-to-unicode table, the pre-tokenizer regex
(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
with `\p{L}` written `[^\W\d_]` and `\p{N}` written `\d` for Python's `re`),
and the rank-ordered merging of §5.3. `test_codec.py` holds it to `tiktoken`'s
`gpt2` encoding id for id on every text of the test corpus when `tiktoken` is
installed and skips that one test when it is not; exact parity on every unicode
category is not promised, and the test corpus says how far it holds.

Whole, the vocabulary is 50,257 tokens and the three marks are **appended** at
`50257..50259`, the tokenizer's ids unchanged: that primes to `L = 1` under
the ceiling (§16.1) — **no context at all, a unigram over tokens** — and one
token of context is 2.5 billion nodes. Capped, `top=K` keeps the `K` most
frequent GPT-2 tokens of the tokenizer-data corpus, renumbered `3..K+2` after
the marks in frequency order (the kept GPT-2 ids are saved in the codec block
so the renumbering is reproducible), and reads every other token as `<unk>`;
at `K = 2000` that is 4,018,021 nodes at `L = 2` — one GPT-2 token of
context — under the ceiling. What `<unk>` swallows is reported by `info`.

### 5.6 `external` — any other published tokenizer

`external("tiktoken:cl100k_base")` or `external("hf:/path/to/tokenizer.json")`:
the tokenizer's own ids, unchanged, with the three marks **appended** at
`V, V+1, V+2`. Imports `tiktoken` or `tokenizers` on first use and raises a
plain error naming the package when it is missing. The codec block stores the
spec, `V` and a digest of the vocabulary (the sorted `(id, bytes)` pairs
hashed with `hashlib.sha256`), and `from_dict` refuses a library whose
vocabulary digests differently — a model must read with the exact tokenizer it
was primed with. `V` is tens of thousands, so this codec primes to `L = 1`
under the default ceiling (§16.1) — a unigram — and takes the same `top=K` cap
as `gpt2` for one token of context. That is stated, not hidden.

### 5.7 The marks, and the sequences that cannot occur

The marks are units of the vocabulary on purpose (`PRD.md` §9.6): the first unit
of a text is predicted from a context that says it is first, and the end of a
text is a unit the model learns to predict. A sequence with a mark *inside* it —
`END` anywhere but last, `START` anywhere but first, a dead id anywhere — can
never occur in a padded text. Its slot exists, stays at zero in both trees, and
is never written to a file.

---

## 6. `address.py`

### 6.1 The numbering

Sequences are numbered by length, then lexicographically by id. The first unit
is the most significant digit:

```
base(ℓ) = (R^ℓ − 1) / (R − 1)         # the number of sequences shorter than ℓ; the id of the first of length ℓ
code(s) = Σ_{i=1..ℓ} s_i · R^(ℓ−i)    # the path as a number in base R
id(s)   = base(ℓ) + code(s)           # 0 ≤ id < N,  N = base(L + 1)
```

Worked example, `R = 3` (`a b c` as `0 1 2`), `L = 2`:

```
ε → 0      a → 1   b → 2   c → 3
aa → 4  ab → 5  ac → 6  ba → 7  bb → 8  bc → 9  ca → 10  cb → 11  cc → 12
```

`id` is a bijection from the sequences of length `0..L` onto `0..N−1`
(invariant 2) — every id is a sequence, so the arrays of §7 and §8 have no
holes.

### 6.2 The moves

For a node `i` at level `ℓ` with `k = i − base(ℓ)` its code:

| move | id | who uses it |
|---|---|---|
| `append(i, ℓ, x)` = `base(ℓ+1) + k·R + x` | the child `s·x`, `ℓ < L` | a walk emitting `x`; counting and crediting |
| `drop_oldest(i, ℓ)` = `base(ℓ−1) + k mod R^(ℓ−1)` | `s[1:]`, `ℓ ≥ 1` | the shift and the fall (§9.4) |
| `drop_newest(i, ℓ)` = `base(ℓ−1) + k // R` | `s[:-1]`, `ℓ ≥ 1` | the tree's parent; the graph view |
| `level(i)` | the `ℓ` with `base(ℓ) ≤ i < base(ℓ+1)`, by bisection over the `L + 2` bases | everything |
| `newest(i, ℓ)` = `k mod R`, `oldest(i, ℓ)` = `k // R^(ℓ−1)` | the last and first unit of `s` | decoding a path |

The children of `i` are the **contiguous** block `base(ℓ+1) + k·R … + R − 1`,
one entry per id in `0..R−1`; summing over `Σ_out` is one slice sum minus the
few excluded entries (§7.3).

```python
class Address:              # (R, L) bound once; bases and powers precomputed; every method is integer arithmetic
    R: int; L: int; N: int; bases: list[int]; powers: list[int]
    def of(self, seq: Sequence[int]) -> int ; def seq(self, i: int) -> tuple[int, ...] ; def level(self, i: int) -> int
    def append(self, i, ℓ, x) -> int ; def drop_oldest(self, i, ℓ) -> int ; def drop_newest(self, i, ℓ) -> int
    def block(self, i, ℓ) -> tuple[int, int]          # (start, stop) of the children block
    def substrings(self, ids: Sequence[int]) -> Iterator[int]   # the id of every substring of length 1..L, position by position (§6.3)
```

`Address` raises `ValueError` for `R < 2`, `L < 1`, a level out of range or an
id outside `0..R−1`; it never returns an id outside `0..N−1`.

### 6.3 The rolling code

Counting or crediting a text needs the id of every suffix of length `1..L` at
every position. They roll: the last `ℓ` units ending at position `t` are the
last `ℓ − 1` units ending at `t − 1` followed by `u_t`, so

```
code_ℓ(t) = code_{ℓ−1}(t − 1) · R + u_t,     code_0 = 0
```

computed for `ℓ = min(L, t) … 1` in place, in that order (each `code_ℓ` reads
the previous position's `code_{ℓ−1}` before it is overwritten). `L`
multiplications and `L` array writes per unit, no lookup, no allocation — the
loop of §7.2, shared by §8.2, and the whole of training.

### 6.4 The brute-force oracle — `check.py`

The requirement says every option is inserted by brute force. `check.py` does
exactly that, and is the only place that does: `brute_force(R, L)` inserts every
sequence of length `1..L`, one at a time, into a pointer trie with the standard
radix insertion of `RadixTrieLLM_RNN/main.py` (a partial match splits the
edge), then walks the result and asserts against `Address`:

1. the trie has exactly `N` nodes, every edge label has length 1, and no node
   has exactly one child — priming a complete set leaves nothing for radix
   compression to do;
2. `id(seq)` over the trie's nodes is a bijection onto `0..N−1`;
3. for every node, every pointer agrees with the arithmetic: each child with
   `append`, the parent with `drop_newest`, the node for `s[1:]` with
   `drop_oldest`; depth with `level`.

The test suite runs it at `(R, L) ∈ {(2,5), (3,4), (5,3), (7,2)}`; the CLI's
`check --R --L` runs it as large as memory allows. It is the definition of
correctness for §6, and §6 is the definition of both trees for everything else.

---

## 7. `count.py` — CountTree

### 7.1 Storage

```python
class CountTree:
    codec: Codec ; L: int ; address: Address
    cnt: array            # array('q'), N entries: cnt[i] = occurrences of sequence i as a substring of the padded texts
    texts: int ; units: int ; unk: int      # what was read: texts, units (marks excluded), of which unk
    version: int          # bumped by every observe(); the caches of §9 key on it
```

One `array('q')` of `N` signed 64-bit counts, allocated once at priming
(`array('q', bytes(8 * N))`), and nothing else per node in the count kind.
`prime` refuses an `N` above `Settings.node_ceiling` (default `4_194_304`)
with a `ValueError` naming `R`, `L`, `N` and the ceiling — the budget of §16
is enforced where the memory would be spent.

### 7.2 Counting — `observe`

```python
def observe(self, ids: Sequence[int]) -> int      # a padded text; returns the number of increments
def observe_text(self, text: str) -> int          # observe(codec.padded(text))
```

```
codes = [0] * (L + 1)
for t, x in enumerate(ids, 1):
    m = L if t >= L else t
    for ℓ in range(m, 0, -1):
        c = codes[ℓ - 1] * R + x
        codes[ℓ] = c
        cnt[bases[ℓ] + c] += 1
```

Every substring of length `1..L` of the padded text is counted once per
occurrence. A text is observed whole — substrings never cross texts — and the
loop is the hot loop of §16: `bases`, `cnt`, `R` and `codes` are locals, and
there is no call, no lookup and no allocation inside it. `observe` is additive
(a text twice doubles its counts), touches no other state than the three
tallies and `version`, and **is the only thing that writes `cnt`** (invariant 3).

### 7.3 The context count, the share and `own`

```python
def ctx(self, i, ℓ) -> int              # Σ over Σ_out: sum(cnt[start:stop]) minus the entries of START and the dead ids
def share(self, i, ℓ) -> list[float]    # (cnt[s·x] + σ) / (ctx + σ·R') for x in Σ_out; σ = settings.smoothing
def own(self, i, ℓ) -> float            # ctx / (ctx + ALPHA)
```

`ctx` is one slice sum over the contiguous children block less the two or
three excluded entries. For every sequence that does not touch a mark it equals
`cnt[i]` (a substring not at the very end of a text is followed by something);
`test_count.py` asserts it, and the definition stays the sum so that the root
and the marks need no special case. With `σ = 0` the share is the raw
frequency and an unread step's share is `0`; with `σ = 0.5` an unread step keeps
a small share, which is what lets a reward lift it (§9.2).

### 7.4 Invariants (asserted by `test_count.py`)

1. `len(cnt) == N` before and after any number of `observe` calls.
2. `cnt[id(s)]` equals a literal sliding-window count of `s` over the padded texts, for every `s` up to length `L`, on small texts.
3. `ctx(s) == cnt[s]` for `1 ≤ |s| ≤ D` whenever `s` neither starts with `START` nor ends with `END`.
4. Every sequence with a mark or a dead id inside it has `cnt == 0`.
5. `observe` is additive and bumps `version` exactly once; nothing else changes `cnt`.

---

## 8. `reward.py` — RewardTree

### 8.1 Storage

```python
class RewardTree:
    codec: Codec ; L: int ; address: Address
    plus: array           # array('d'), N entries: the rewards the step i received, >= 0
    minus: array          # array('d'), N entries: the penalties the step i received, >= 0
    judged: int ; rewards_total: float ; penalties_total: float
    version: int
```

Two `array('d')` — 16 bytes per node. The reward and the penalty are **kept
apart** on purpose: the net `plus − minus` is what the reward traversal reads
and it is right for likelihood, but a step rewarded five times and punished
once must not read like one rewarded four times and never punished when the
question is which way the least has gone wrong on (`SPEC-LeastPunished.md`
§1). Both arrays are monotone non-decreasing except through `invert` (§8.4).

### 8.2 Crediting a judged text — at every level; a punishment is a negative reward

```python
def credit(self, ids: Sequence[int], amount: float, rungs: str = "all", skip: int = 0) -> int
    # a padded text and a SIGNED amount: +strength·weight for a reward, −strength·weight for a punishment —
    # a punishment is a negative reward, and the same call writes both.
    # skip: positions at the start that are context only - their substrings roll the codes but are not
    #       credited. What the model was GIVEN is not what it PRODUCED: PairModel passes the prefix's length.
    # rungs="all":   every substring of length 1..L ending at every position — the same ids observe() counts —
    #                gets plus += amount (amount > 0) or minus += −amount (amount < 0).
    # rungs="final": only the substrings of length L (the final nodes): the step is credited at the deepest
    #                context alone.
    # Returns the number of entries written.  amount == 0 writes nothing.
```

The loop is §7.2's with a float and a sign. Where a step's credit lives is the
node of the sequence `s·x` — the step "`x` after `s`" — and with `rungs = all` a
step is credited at **every** context length: on `x` alone, on `s₁·x`, …, up
to the deepest. Shallow nodes therefore accumulate the credit of many outcomes
and generalise; deep nodes hold the credit of one context and specialise. That
is the whole of what "connected at every level, not just the final nodes" does
on the writing side, and `rungs = final` is the pair the idea started from,
kept so the difference is measured (S-7).

An outcome credits **what the model produced, not what it was given**: `reward`
and `punish` take the `prefix` the model continued, whose units are context
for the credited steps and earn nothing themselves (`skip`). The prefix and the
outcome are joined by the codec (`join`) before they are encoded, so a phonetic
model credits the word boundary as the outcome's first step — `mat` after
`... the` is `# M AE1 T`, the way the model itself would emit it. Rewarding a whole
sentence instead credits the prefix's steps too — at four characters, `he ` is
followed by `c` in `the cat` as well as by `m` in `the mat`, and both would
earn the reward.

`credit` **is the only thing that writes `plus` and `minus`** (invariant 4),
and it never touches `cnt`. `PairModel.reward` / `punish` / `two_nrl` /
`feedback` (§11) are the callers; the count tree is written by none of them
unless asked (`read=True`, `PRD.md` §9.7).

### 8.3 Reading a step

```python
def reward(self, i) -> float      # plus[i] − minus[i]: the net, what the reward traversal reads
def penalty(self, i) -> float     # minus[i]: what the punishment traversal reads
def rewards(self, i, ℓ) -> list[float] ; def penalties(self, i, ℓ) -> list[float]   # over the children block, Σ_out order
```

### 8.4 `invert`

`invert()` swaps `plus` and `minus` — every reward becomes a penalty of the
same size and every penalty a reward — and bumps `version`. It is an involution
(`test_reward.py`). It exists for parity with the family's count model, whose
`invert` negates the rewards; the pair's `two_nrl` does not use it (§11).

### 8.5 Invariants (asserted by `test_reward.py`)

1. `plus[i] ≥ 0` and `minus[i] ≥ 0` everywhere, always.
2. `credit(ids, a)` with `rungs = all` writes exactly the ids `observe(ids)` would count, each by `|a|`; with `rungs = final`, exactly those at level `L`.
3. `credit` is additive; a reward and a penalty of the same size on the same text leave `reward(i) == 0` and `penalty(i) > 0` — the net is gone, the punishment is not.
4. `invert` twice is bit-identical.
5. Nothing but `credit` and `invert` changes `plus` or `minus`; `credit` never changes `cnt`.

---

## 9. `pair.py` — the rungs

### 9.1 One address space, two trees

```
     the count tree  (what was read)                      the reward tree  (what it earned)

               ε  ---------------------------- rung ----------------------------  ε
            /  |  \                                                            /  |  \
           a   b   c  -------------------- rungs at level 1 ----------------  a   b   c
         / | \                                                              / | \
       aa ab ac  ----------------------- rungs at level 2 --------------  aa ab ac
        cnt[ab] = 12                                                       plus[ab] = 4.0, minus[ab] = 1.0
```

Every sequence is one node of each tree and the rung joins them. Both trees are
complete over the same `(codec, L)`, so the node sets are identical and the
rung between equal nodes is **the same id**: the implementation keeps one
`Address`, one `CountTree` and one `RewardTree`, and a rung is the pair
`(cnt[i], plus[i], minus[i])`. Nothing is stored for a rung; what it carries is
the rule of §9.2.

```python
class RadixPair:
    codec: Codec ; L: int ; address: Address
    count: CountTree ; reward: RewardTree
    settings: Settings                                  # alpha, floor, smoothing, the scales, backoff, rungs (§11)
    def scores(self, i, ℓ, traversal="reward") -> list[float]   # §9.2: the step scores of node i's children, Σ_out order
    def q(self, i, ℓ, traversal="reward") -> list[float]        # softmax(scores)
    def fold(self, context: Sequence[int], traversal="reward") -> list[float]   # §9.3: the exact next-unit distribution
    def score(self, text: str) -> Score                          # §9.5
```

### 9.2 The rung — the score of a step

At a context `c` (level `ℓ ≤ D`), for every `x ∈ Σ_out`, the count tree says
how often the step was taken and the reward tree what it earned; the rung adds
them the way `RadixCyclicNN`'s count model adds them inside one weight (D-022,
without the recency window and without the `log1p(count)` term that defaults
to zero there):

```
merit(x | c)   = share_scale · log share(x | c)                     the count tree's term
reward(x | c)  = plus[c·x] − minus[c·x]                              the reward tree's net
penalty(x | c) = minus[c·x]                                          its penalty side alone

traversal "reward"      (the default)
    score(x | c) = merit(x | c) + reward_scale · reward(x | c)
    q_c = softmax(score)          i.e.  q_c(x) ∝ share(x | c)^share_scale · e^(reward_scale · reward(x | c))

traversal "punishment"
    score(x | c) = merit_scale · merit(x | c) − penalty_scale · penalty(x | c)
    q_c = softmax(score)          the rewards leave the score; only what was punished prices a step
```

With `rungs = final`, `reward` and `penalty` are read as `0` at every level but
the deepest (`ℓ = D`), where the children are the final nodes.

Three things to notice. `share_scale = 1` and no rewards give `q_c` = the
smoothed share, the plain count model. The smoothing `σ` is what lets a reward
act on a step that was never read: with `σ = 0` such a step has `share = 0`,
`log 0 = −∞`, and no reward can lift it; with `σ = 0.5` it keeps
`σ / (ctx + σ·R')` and `e^(reward)` multiplies that — a rewarded output the
corpus never contained becomes likely, which is what "rewarded based on correct
outcomes" has to be able to do. And the punishment traversal is *unbuyable*:
adding rewards to a step changes nothing under it, and a step punished once
stays punished however often it is rewarded afterwards (`test_pair.py`, S-6).

`scores` is `O(R')` per node — one slice of each of three arrays — and is
cached per `(node, traversal)` in a dict keyed on the two trees' `version`s.

### 9.3 The fold — the exact next-unit distribution

From a context `c` of at most `D` units, for every `x ∈ Σ_out`:

```
P(x | c) = own(c) · q_c(x)  +  (1 − own(c)) · P(x | c[1:])          |c| ≥ 1
P(x | ε) = own(ε) · q_ε(x)  +  (1 − own(ε)) / R'
answer(x | c) = (1 − FLOOR) · P(x | c)  +  FLOOR / R'
```

Read as the walk it is: at `c`, stay with probability `own(c)` and emit from
`q_c` — the rung's answer at this level — or fall with `1 − own(c)` to the
context without its oldest unit and decide again; at the root, falling means
answering uniformly — knowing nothing. Every level from the deepest to the root
is visited, and at every one the rung is read: that is "connected at every
level" on the reading side. The floor is the one term that is not a path
(§19, question 7).

**This is `FilterBankRadix/DESIGN.md` §5.3, unrolled from the other end.** That
fold runs shortest level first, `p ← own · p_here + (1 − own) · p`, from the
prior `1/A`; expanding it from the longest level down gives the recursion above
term for term, with `q_c` in place of `p_here`. The two coincide exactly when
`q_c` is the raw share — `smoothing = 0`, `share_scale = 1` and no rewards —
and at those settings the identity was checked while writing this document:
200 random count tables, every context length, identical to `1e−12` and summing
to 1. That check is `test_pair.py::test_the_rungs_are_the_fold`, with the
shortest-first fold copied into the test as the reference (S-3). At the default
`σ = 0.5` the two differ by the Jeffreys smoothing inside each level, which is
reported beside the identity (S-4).

Cost: `|c| + 1` levels × `O(R')` = `O(D · R)` per unit.

### 9.4 The shift

A tree of depth `L` reads a text of any length because the walk **slides**: a
node at level `L` has no children, so a walk that has just emitted the `L`-th
unit of a window continues from `drop_oldest` — the same sequence without its
oldest unit, at level `D` — deterministically and at no cost, and emits again
from there. Falling back in the fold is the same move taken *before* the window
is full, at the price `1 − own(c)` the count tree sets. Both are arithmetic
(§6.2); neither is an edge, and neither tree has anything to say about them.

### 9.5 Scoring a text

For a padded text `u₁ … u_{T+2}`, at every position `t = 2 … T+2` with `c_t`
the last `≤ D` units before it:

```
bits_t    = −log₂ answer(u_t | c_t)           the model's belief, under the reward traversal
reward_t  = reward(c_t · u_t)                 the reward tree's net at the deepest node of the step
penalty_t = penalty(c_t · u_t)                and its penalty
```

```python
@dataclass
class Score:
    bits: float                 # Σ bits_t / (T + 1)
    mean_reward: float          # Σ reward_t / (T + 1)
    worst_penalty: float        # max_t penalty_t — the least-punished ranking's number for a whole text
    per_unit: list[tuple[str, float, float, float]]   # (unit, bits_t, reward_t, penalty_t)
    units: int
```

Two trees, two readings of one text: what the corpus makes of it, and what the
judges said about the steps it takes.

### 9.6 `rungs` and `backoff`

Two settings, both measured on the same numbers (`PRD.md` S-5, S-7):

| setting | values | what it decides |
|---|---|---|
| `rungs` | `all` (default), `final` | where a judged step is credited (§8.2) and where the reward terms are read (§9.2): at every level, or at the final nodes only — the pair connected everywhere, or only where the idea started |
| `backoff` | `all` (default), `deepest`, `none` | how far a prediction may fall: the fold of §9.3; one fall from the given context straight to the uniform (`own(c) · q_c + (1 − own(c)) / R'`, then the floor — the shape `FilterBankRadix` measured at 2.998 against 2.808 bits per character, its §5.3); or no fall (`q_c`, then the floor) |

`rungs` is a property of the model (it decides what was written) and is saved
in the file; `backoff` is a setting of a prediction and may be changed at any
time.

---

## 10. `search.py` — walks over the pair

**The greedy walk of the exact fold is the default** (`mode = greedy`), and the
cheapest path (`mode = dijkstra`) is the option, the other way round from the
family. Measured while building: on a primed tree the cheapest single path can
pay for a rare unit at the deepest level — an `<unk>` — because the unread
context it lands in then falls to the root for nothing (`own = 0`), where
frequent units are cheap; the fold sums over every fall and is not fooled. On a
grown graph the rare edge does not exist and the problem does not arise.

State: `(node, level, emitted)`. Moves from `(i, ℓ, e)`, under the traversal
the call names:

| move | to | cost | when |
|---|---|---|---|
| emit `x` | `(append(i, ℓ, x), ℓ + 1, e + 1)` | `−log(own(i) · q_i(x)) + step_penalty` | `ℓ ≤ D` |
| emit `x` from the root | `(append(0, 0, x), 1, e + 1)` | `−log(own(ε) · q_ε(x) + (1 − own(ε)) / R') + step_penalty` | `ℓ = 0` — the root's fall to the uniform is folded into its emissions, so every unit has a finite cost somewhere |
| fall | `(drop_oldest(i, ℓ), ℓ − 1, e)` | `−log(1 − own(i))` | `1 ≤ ℓ ≤ D` |
| shift | `(drop_oldest(i, L), D, e)` | `0` | `ℓ = L` — forced: a final node has no children |
| emit `END` | goal | as emit | |

The search's costs are the fold's terms without the floor: a path's cost is one
choice of how far to fall at every step, and the floor — a mixture over the
whole answer — does not decompose along a path. Consequently the cheapest
path's cost is never below the un-floored fold's cost of the same units (it
picks the best fall instead of summing over falls); `test_search.py` asserts
both the decomposition and the inequality against `P(x | c)` before the floor.

```python
class PathResult:      # text: str, units: list[int], node_ids: list[int], hops: list[tuple[str, int]] (every hop:
                       # ("emit", id) | ("fall", id) | ("shift", id)), cost: float, step_costs: list[float] (one per
                       # emitted unit: its emission plus the falls before it), traversal: str, expanded: int,
                       # reached_end: bool, full_text: str (set by PairModel: prefix + text)
                       # to_dict() -> JSON-serialisable

def dijkstra(pair, context: Sequence[int], min_units: int, max_units: int | None = None, to_end: bool = False,
             traversal: str = "reward", step_penalty: float = 0.0, max_expansions: int = 200_000) -> PathResult
    # Start at the node of the last ≤ D units of the context. heapq of (cost, tie, node, level, emitted); a
    # best-cost dict keyed by (node, emitted) — level is a function of node. Goal: to_end -> END; else the first
    # popped state with emitted >= min_units (Dijkstra pops in cost order, so it is the cheapest such path); END
    # before min_units is also a goal. States with emitted >= max_units are not expanded. Fallback on the
    # expansion cap: the popped state with the most emitted units (ties -> lowest cost) — never raise.

def greedy(pair, context, units: int, traversal="reward", rng=None, temperature: float = 0.0) -> PathResult
    # One unit at a time from the exact fold (answer(), floor included): the argmax at temperature 0, else a
    # sample from softmax(log answer / temperature) with the model's seeded rng. Stops at END or at `units`.
    # This is mode="greedy" and mode="sample"; it is exact where dijkstra is a cheapest path.
```

Every hop of a walk is reported, falls and shifts included, so a walk can be
watched the way `RadixCyclicNN`'s voice reports every step; `step_costs` fold
the falls into the unit they precede so the per-unit numbers line up with
`Score.per_unit`.

---

## 11. `model.py` — PairModel

```python
@dataclass
class Settings:
    alpha: float = 2.0             # ALPHA of own()
    floor: float = 0.02            # FLOOR of answer()
    smoothing: float = 0.5         # σ of share(); 0 reproduces FilterBankRadix's fold exactly
    share_scale: float = 1.0       # the count tree's scale in the reward traversal
    reward_scale: float = 1.0      # the reward tree's scale in the reward traversal
    merit_scale: float = 1.0       # the two scales of the punishment traversal
    penalty_scale: float = 1.0
    strength: float = 1.0          # one unit of reward multiplies a step's odds by e, as the family's count model
    rungs: str = "all"             # "all" | "final" — saved with the model (§9.6)
    backoff: str = "all"           # "all" | "deepest" | "none"
    step_penalty: float = 0.0
    max_expansions: int = 200_000
    node_ceiling: int = 4_194_304  # prime() refuses a larger N (§7.1)

class PairModel:
    kind: str                      # "count" | "sine"
    codec: Codec ; L: int ; seed: int ; settings: Settings
    pair: RadixPair ; encoder ; decoder          # the codec's two halves as objects
    history: list[dict]            # one record per train / reward / punish call: what it did, how much, how long

    @classmethod
    def prime(cls, codec: Codec, L: int, kind: str = "count", seed: int = 0, settings: Settings | None = None) -> PairModel
    # the module-level prime(codec="phones", L=None, ...) takes a preset name and defaults L to DEFAULT_L[codec]: phones 3

    # -- reading --
    def train(self, texts: Iterable[str], progress: ProgressFn | None = None) -> dict
        # count kind: observe_text per text; returns {"texts", "units", "unk", "unk_share", "increments", "seconds"}

    # -- outcomes: the family's primitives (D-026), marks as weights (D-050) --
    def reward(self, texts, *, strength: float | None = None, weights: Sequence[float] | None = None, read: bool = False,
               prefix: str | None = None) -> dict
        # credit(+strength·weight) per text at the model's `rungs`; a weight of 0 is skipped; prefix: what the model was
        # given - context for the texts, not credited (section 8.2); read=True also observes the text into the count
        # tree (the family's count model's behaviour; off by default — PRD.md section 9.7)
    def punish(self, texts, *, strength=None, weights=None, prefix=None) -> dict   # reward with the sign reversed; never reads
    def two_nrl(self, bad, good, *, strength=None, bad_weights=None, good_weights=None, prefix=None) -> dict   # punish(bad) then reward(good); no inversion
    def feedback(self, good=(), bad=(), *, good_weights=None, bad_weights=None, strength=None, prefix=None) -> dict
        # both -> two_nrl; only good -> reward; only bad -> punish — D-026's dispatch, so a tutor's marks, a sandbox's
        # verdict and a person's thumb all end here
    def invert(self) -> None                                                  # the reward tree's swap (§8.4)

    # -- prediction and scoring --
    def predict(self, prefix: str, length: int, mode: str = "greedy", traversal: str = "reward", start: bool = True,
                temperature: float = 1.0, to_end: bool = False, backoff: str | None = None) -> PathResult
        # start=True: the prefix begins a text (padded with START); False: a fragment. The context is the last ≤ D
        # units. mode: "greedy" (the exact fold, the default - section 10) | "dijkstra" | "sample".
        # full_text = codec.decode(prefix units + emitted units): phones for a phonetic model; result.spelled is
        # codec.spell(the same), the English they spell.
    def generate(self, prefix: str = "", length: int = 60, mode: str = "greedy", **options) -> PathResult
        # the walk runs to END, capped at `length`; from START alone when prefix == "". The cheapest complete text
        # (mode="dijkstra") is one step long whenever END is likelier than any continuation, as the family's is.
    def score(self, text: str, traversal: str = "reward") -> Score
    def distribution(self, context: Sequence[int], traversal: str = "reward") -> list[float]   # the fold; for tests and the API
    def weights(self, **scales) -> dict      # change smoothing / the four scales / backoff at run time and report them,
                                             # as the family's `weights` command; `rungs` is not changeable after priming
    def info(self) -> dict        # kind, codec, R, L, N, nonzero counts, nonzero rewards, texts, units, unk_share, judged,
                                  # rewards_total, penalties_total, settings, memory (bytes), history
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> PairModel
    def save(self, path: str) -> None ; @classmethod load(cls, path: str) -> PairModel
```

The count kind has no learning rate, no epochs, no shuffling and no seed in
training or feedback; `seed` is used by `mode=sample` and by the sine kind's
initialisation. `strength` defaults to the settings' (`1.0`: one unit of
reward multiplies a step's odds by `e` at `reward_scale = 1`, as the family's
count model).

---

## 12. The sine kind (phase 4) — `activation.py`, `sine.py`

The second kind on the same nodes; the count kind's numbers are the baseline
it must match (`PRD.md` phase P4). The reward tree is untouched: a reward is a
number written on a node, never a gradient's target. What changes is the count
tree's term at the rung.

* **Parameters**, seven `array('d')` of `N` — 56 bytes per node: `z, a, b, h, k`
  (the node's activation `f(x) = a·sin(b(x − h)) + k`, initialised to
  `−1, 1/3, 0, 0`) and `wa[i]`, the weight of the edge *into* node `i` from its
  parent — a tree has one parent per node, so the edge's weight lives on the
  child, as `FilterBankRadix` §5.1.
* **Initialisation by id**, so the file can stay sparse: every initial value is
  a function of `(seed, i, j)` — `u(seed, i, j) = (splitmix64(seed · 2^32 + i · 2 + j) >> 11) / 2^53`,
  with `splitmix64` the standard mixer (`x += 0x9E3779B97F4A7C15; x = (x ^ x>>30) · 0xBF58476D1CE4E5B9;
  x = (x ^ x>>27) · 0x94D049BB133111EB; x ^= x>>31`, all mod `2^64`); `z[i] = −4.5 + 9·u(·, 0)`,
  `wa[i] = 0.5 + u(·, 1)` — the family's ranges, spelled out to the bit so a port
  draws the same numbers.
* **The rung**: `merit(x | c) = wa[c·x] · f_c · f_{c·x}` — *the activation of the
  child times the activation of the parent*, the family's rule — in place of
  `share_scale · log share`; the reward and penalty terms, the softmax, the
  fold, `own` (still from the counts: the count tree keeps counting in this
  kind) and both traversals are unchanged.
* **The one-hop rule**: per position, every level's `q` against the unit that
  was read, `g = q − onehot(x)`, `∂/∂wa[c·y] = g_y · f_c · f_{c·y}`, `∂/∂f_c =
  Σ_y g_y · wa[c·y] · f_{c·y}`, `∂/∂f_{c·y} = g_y · wa[c·y] · f_c`, chained through
  `sine_partials` into `z, a, b, h, k` of `c` and its children; summed over a
  batch, divided by its size, clipped to `±5`, applied once; the activation
  parameters at `ACT_RATE = 0.1` of the weights' rate; `b ≥ MIN_B = 1e-3`.
  Nothing crosses further than a node and its children (`FilterBankRadix` §5.4).
  `check.check_sine` reads the analytic gradient out of `step` and compares it
  with central differences.
* **`invert`**: `wa → −wa`, `a → −a`, `k → −k` on the count side — the exact
  negation of the unit, as `RadixCyclicNN/DESIGN.md` §5.2 — *and* the reward
  tree's swap; twice is bit-identical. `two_nrl` on this kind is the family's:
  train on `bad`, `invert`, fine-tune on `good` at a smaller rate.

Provisional in one respect (§19, question 6): whether the fall should be
learned in this kind rather than read off the counts.

---

## 13. Persistence

JSON, format `"radixpair"` version `1`, gzip when the path ends in `.gz`;
`load` sniffs the gzip magic and ignores the suffix. Written through a
temporary file and `os.replace`, as `RadixCyclicNN`'s `write_bytes_atomic`.

```json
{"format": "radixpair", "version": 1,
 "codec": {"name": "bpe", "vocab_size": 1024, "merges": [[104, 101], [116, 104]], "start": 0, "end": 1, "unk": 2},
 "L": 2, "kind": "count", "seed": 0,
 "settings": {"alpha": 2.0, "floor": 0.02, "smoothing": 0.5, "share_scale": 1.0, "reward_scale": 1.0,
              "merit_scale": 1.0, "penalty_scale": 1.0, "strength": 1.0, "rungs": "all", "backoff": "all",
              "step_penalty": 0.0, "max_expansions": 200000, "node_ceiling": 4194304},
 "read":   {"texts": 12, "units": 913, "unk": 0},
 "judged": {"texts": 3, "rewards_total": 15.0, "penalties_total": 1.0},
 "history": [{"call": "train", "texts": 12, "units": 913, "increments": 1826, "seconds": 0.01},
             {"call": "reward", "texts": 2, "strength": 5.0, "entries": 40, "seconds": 0.0}],
 "counts":  {"ids": [1, 4, 17], "values": [12, 3, 1]},
 "rewards": {"ids": [4, 17], "plus": [5.0, 5.0], "minus": [0.0, 1.0]},
 "sine": {"inverted": false, "ids": [], "z": [], "a": [], "b": [], "h": [], "k": [], "wa": []}}
```

* The **codec block** holds the whole tokenizer: `chars` its symbols; `bytes`
  nothing but its name; `bpe` its merge list in rank order; `phones` the
  tokenizer's settings (`level`, `stress`, `boundaries`, `pauses`); `external`
  its spec, `V` and the vocabulary digest (§5.6); `gpt2` its two files whole and,
  capped, the kept ids (§5.5). `R` and `N` are recomputed, never read.
* `counts.ids` are sorted and hold **only the non-zero counts**; `rewards.ids`
  only the nodes with a non-zero `plus` or `minus`; `sine.ids` only the nodes
  whose parameters differ from §12's initial values (the block is absent in
  the count kind).
* `load` is `prime(codec, L, kind, seed, settings)` followed by writing the
  entries back.
* A file never holds `N` of anything: its size is proportional to what was
  read and judged (`PRD.md` S-8), and a model of ten sentences and three
  judgements is a few kilobytes at any `L`.
* `checkpoint.py` is `RadixCyclicNN`'s `CheckpointManager` shape: rotation, a
  `latest` pointer, `load_latest`, resume; a checkpoint is a model file.

---

## 14. `cli.py` and the `Makefile`

```
python3 -m radixpair prime    --model m.json [--codec phones|syllables|chars|bytes|bpe|gpt2|external] [--L 3]   # phones at L=3 by default
                              [--vocab-size 1024] [--tokenizer-data corpus.txt] [--vocabulary lexicon|corpus] [--top 2000]
                              [--gpt2-files encoder.json vocab.bpe] [--external tiktoken:cl100k_base] [--no-stress]
                              [--kind count|sine] [--seed 0] [--rungs all|final] [--ceiling N]
python3 -m radixpair train    --model m.json --data corpus.txt [--data more.txt ...] [--checkpoint-dir DIR]
                              [sine kind: --epochs 5 --lr 0.05 --act-lr 0.005 --batch 256 --clip 5]
python3 -m radixpair reward   --model m.json (--text "..." | --data file) [--strength 1] [--ratings 9 7 10] [--read]
                              [--prefix "what the model was given"]
python3 -m radixpair punish   --model m.json (--text "..." | --data file) [--strength 1] [--ratings ...] [--prefix ...]
python3 -m radixpair 2nrl     --model m.json --bad bad.txt --good good.txt [--strength 1] [--prefix ...]
python3 -m radixpair feedback --model m.json [--good ...] [--bad ...] [--good-ratings ...] [--bad-ratings ...] [--prefix ...]
python3 -m radixpair predict  --model m.json --prefix "the quick brown" --length 20 [--mode greedy|dijkstra|sample]
                              [--traversal reward|punishment] [--merit-scale 1] [--penalty-scale 1]
                              [--temperature 1.0] [--fragment] [--to-end] [--hops]
python3 -m radixpair generate --model m.json [--prefix ""] [--length 60] [the predict options]
python3 -m radixpair score    --model m.json (--text "..." | --data file) [--per-unit] [--traversal ...]
python3 -m radixpair weights  --model m.json [--smoothing 0.5] [--share-scale 1] [--reward-scale 1] [--merit-scale 1]
                              [--penalty-scale 1] [--backoff all|deepest|none]        show, or set and save
python3 -m radixpair info     --model m.json
python3 -m radixpair check    [--R 3 --L 4]                                   the brute-force oracle (§6.4)
python3 -m radixpair bench    [--codec chars --L 4 --units 200000]            throughput: counting and crediting (§16)
python3 -m radixpair bench compare  --data corpus.txt [--holdout 0.1] [--L 4] [--smoothing 0]   bits per unit: all / deepest / none, and the references (PRD S-4, S-5)
python3 -m radixpair bench feedback [--smoothing 0]                            the mat / log case under both traversals, at smoothing 0 (PRD S-6)
python3 -m radixpair bench rungs    --data corpus.txt                          rewards at every level against the final nodes (PRD S-7)
python3 -m radixpair invert   --model m.json
```

`--json` on every command prints one JSON document to stdout and progress to
stderr, as the family does. `--codec` defaults to `phones` and `--L` to the
depth the codec primes to (`DEFAULT_L`); `predict` and `generate` print the
phones and, on the line below, the English they spell. `--data` reads one text
per line, blank lines skipped; `--ratings` are marks out of 10, one per text, turned into weights as
D-050. The `Makefile` has a target per command with the variables overridable
on the command line (`make train DATA=… L=4`), plus `test`
(`python3 -m unittest discover -s tests`) and `check`.

---

## 15. Tests (`unittest`, no dependencies, seconds)

* `test_codec.py` — every preset: `R`, the marks, the dead ids, `emits`; normalisation; `unk`; `padded`; the round trip; `to_dict` / `from_dict`; `phones` and `syllables` agree with `phonetok` token for token, the syllable vocabulary is closed, sorted and frozen from either source, a syllable outside it is `<unk>` and `top` keeps the most frequent, skipped when the package is not importable; `gpt2` from the two files: the byte table is a bijection, the pre-tokenizer cuts the documented cases, the merges reproduce the release's own examples, the cap renumbers reproducibly, and id-for-id agreement with `tiktoken` on the test corpus when it is installed; `external` agrees with its library id for id and refuses a vocabulary that digests differently, skipped when neither library is installed.
* `test_bpe.py` — pieces never crossed by a merge; training reaches the vocabulary size and is deterministic (ties); `decode(encode(b)) == b` for random bytes and for unicode text; the merge list round-trips through the file; a repeated word is one cache hit.
* `test_address.py` — **the brute-force oracle** at `(2,5), (3,4), (5,3), (7,2)` (§6.4); the worked example of §6.1; `level` by bisection at every id; the children block contiguous; `append` undone by `drop_newest`; `substrings` against a naive enumeration; the refusals.
* `test_count.py` — the invariants of §7.4; the rolling code against a naive recount; the ceiling refused with the numbers in the message.
* `test_reward.py` — the invariants of §8.5: `credit` writes the ids `observe` counts (and, with `rungs = final`, the level-`L` ones only); reward and penalty kept apart; weights and a zero weight skipped; `invert` twice; `cnt` untouched.
* `test_pair.py` — `test_the_rungs_are_the_fold`: at `smoothing = 0` and no rewards, the pair against the shortest-first fold of `FilterBankRadix` §5.3 (copied into the test), 200 random tables × every context length, to `1e-9`, and every answer a distribution; `deepest` and `none` as §9.6 defines them; at `σ = 0.5` a reward lifts a never-read step and at `σ = 0` it cannot; the punishment traversal ignores rewards entirely and is unbuyable; a step credited with `rungs = all` moves a sibling context that shares its last unit, and with `rungs = final` does not (S-7's test half); the cache dropped on either `version`; the mat / log case of S-6 under both traversals.
* `test_search.py` — Dijkstra returns a trained text's continuation; a path's cost is the sum of its emissions and falls and is never below the un-floored fold's cost of the same units; the shift at level `L` is free and forced; the fallback on the expansion cap never raises; `to_end`; both traversals through every mode; `greedy` at temperature 0 is the fold's argmax; `sample` is seeded and terminates.
* `test_model.py` — train lowers a trained text's bits; `reward` lowers the bits of the rewarded text and `punish` raises them, at the model's `rungs`; `two_nrl` and `feedback` dispatch as D-026, with ratings as weights; `read=True` counts and the default does not; `start` against `fragment`; `weights` changes the fold and `rungs` cannot be changed; `history`; save / load identical predictions and scores; the file has no zero count, reward or penalty; file size grows with what was read and judged and not with `L`.
* `test_checkpoint.py` — rotation, the latest pointer, `load_latest`, resume.
* `test_cli.py` — subprocess smoke of every command with `--json`.
* `test_bench.py` — every bench runs in `--quick`; `compare` reports the three `backoff` settings and `rungs` the two `rungs` settings.
* `test_sine.py` (phase 4) — the default equals `−sin(x/3)` and the partials pass a finite-difference check; initialisation is a function of `(seed, id)` and the same across processes; the one-hop rule's analytic gradient, read out of `step`, against central differences; the reward terms unchanged by training; `invert` twice is identity; 2NRL makes garbage less likely; the file holds only the nodes that moved and loads to identical predictions.

---

## 16. Performance notes (must be followed)

### 16.1 The budget

`N` nodes per model, at 24 bytes each — a count and two reward numbers; the
sine kind adds 56. `L` is the longest sequence; the longest context is `L − 1`.

| codec | `R` | `L = 1` (no context) | `L = 2` | `L = 3` | `L = 4` |
|---|---|---|---|---|---|
| `chars` | 33 | 34 | 1,123 | 37,060 · 0.8 MiB | **1,222,981 · 28 MiB** |
| `phones` | 92 | 93 | 8,557 · 0.2 MiB | **787,245 · 18 MiB** | 72,426,541 · 1.6 GiB — a port's job |
| `syllables`, the core lexicon, stress kept (measured) | 1,510 | 1,511 | **2,281,611 · 52 MiB** | 3,445,232,611 — no | — |
| `syllables`, stress dropped (measured) | 1,397 | 1,398 | 1,953,007 · 45 MiB | 2,728,350,780 — no | — |
| `syllables` of `sample_corpus.txt` (measured) | 252 | 253 | 63,757 · 1.5 MiB | 16,066,765 · 368 MiB — a port's job | — |
| `bytes` | 259 | 260 | 67,341 · 1.5 MiB | 17,441,320 · 399 MiB — a port's job | — |
| `bpe` 1,024 | 1,027 | 1,028 | **1,055,757 · 24 MiB** | 1,084,262,440 — no | — |
| `bpe` 2,048 | 2,051 | 2,052 | 4,208,653 · 96 MiB — just over the ceiling | — | — |
| `gpt2` capped to 2,000 | 2,004 | 2,005 | **4,018,021 · 92 MiB** | 8,052,114,085 — no | — |
| `gpt2` whole | 50,260 | **50,261 · 1.2 MiB — a unigram** | 2,526,117,861 · 56 GiB — no | — | — |
| `external` cl100k | 100,280 | 100,281 · 2.3 MiB — a unigram | 10,056,178,681 — no | — | — |

The default ceiling (`4,194,304` nodes, §7.1) admits a vocabulary of at most
45 at `L = 4`, 160 at `L = 3`, 2,047 at `L = 2` and anything at `L = 1`, and
refuses the rest until there is a port or a torch backend to run them. **The
tokenizer decides the depth** — and `L = 1` holds no context at all: it is a
unigram, and one unit of context is `L = 2`.

### 16.2 What was measured while writing this

Pure Python, `array('q')`, one process, the container this document was written
in — a scratch figure that `bench` replaces:

| | |
|---|---|
| `chars, L = 4`, 200,000 random units | 800,000 increments in 0.33 s |
| increments per second | **2.43 million** |
| units per second | 609 thousand |

Crediting is the same loop over a float array and is expected within a factor
of two of it. `PRD.md` NFR-3 asks for 1 million per second; there is margin.

### 16.3 Rules

* The loops of §7.2 and §8.2 bind `cnt` (or `plus` / `minus`), `bases`, `R` and `codes` to locals and contain no call, no attribute lookup and no allocation. They are the whole of training and feedback.
* Context counts, shares and rewards are slice operations over `array` (`sum(cnt[a:b])`, `plus[a:b]`), never Python loops over children; `scores` is cached per `(node, traversal)` keyed on both `version`s.
* The fold allocates one list of `R'` floats per level and reuses it; nothing per unit inside.
* Dijkstra uses `heapq` with `(cost, tie, node, level, emitted)` tuples and a `best` dict; `max_expansions` is a guard, never a loop bound the search grows into.
* The BPE encoder caches the token ids of every piece it has seen; training keeps pair counts incrementally rather than recounting the corpus per merge.
* No per-node objects, ever. `N` is a million; a million Python objects is the one way to make this design look slow.
* `prime` allocates its arrays with `array(code, bytes(size * N))` — one allocation per array, zero-filled by the C library.

---

## 17. Invariants

Asserted in the suite, true at every point in a model's life:

1. **Primed means fixed.** `N = (R^(L+1) − 1) / (R − 1)`; no operation changes the length of any array; the arithmetic tree is the brute-force tree (§6.4).
2. **Every id is a sequence.** `id` is a bijection from the sequences of length `0..L` onto `0..N−1`, and every move of §6.2 lands inside it.
3. **The count tree is written by reading alone.** `cnt[s]` is the number of occurrences of `s` in the padded training texts; only `observe` changes it; child sums equal the node's count except across a mark; a sequence with a mark or a dead id inside stays at zero.
4. **The reward tree is written by outcomes alone.** Only `credit` and `invert` change `plus` or `minus`; `train` never does; `plus, minus ≥ 0` everywhere.
5. **Rewards and penalties are kept apart.** A reward and a penalty of equal size on one step leave its net at zero and its penalty standing; the punishment traversal's answer never changes when rewards are added.
6. **The rungs are the fold.** At `smoothing = 0` and no rewards, the pair's distribution equals `FilterBankRadix` §5.3 on the same counts to `1e-9`; at every setting every answer is a distribution over `Σ_out`.
7. **The shift is free and forced; the fall is priced by the count tree.** A walk at level `L` continues from `drop_oldest` at cost `0`; a fall from level `1..D` costs `−log(1 − own)`.
8. **Determinism.** Same codec, `L`, texts, outcomes and seed ⇒ the same bytes, across processes and across a save / load cycle; `mode=sample` with the same seed gives the same walk.
9. **The file is the data.** A restored model predicts and scores identically; the file holds no zero count, no zero reward, no unmoved parameter, and the whole codec.
10. **The codec is faithful.** `decode(encode(text))` is the normalised text for every preset; `bpe` round-trips bytes exactly; `phones` and `external` agree with their libraries id for id.
11. **The sine kind.** `invert` is an involution; every gradient is one hop; the activation parameters move at a tenth of the weights' rate; `b ≥ MIN_B`; the initial value of every parameter is a function of `(seed, id)`; the reward tree is unchanged by training.

---

## 18. Deliberately absent

Named so that adding one is a decision rather than a drift.

* **No growth.** No split, no merge, no dynamic window, no node after priming. The moment a node is created because a text needed it, this is `RadixCyclicNN`.
* **No open vocabularies.** Every codec is closed at priming and frozen; the syllable level is closed from the lexicon or a corpus first, the word level is not taken, and the phonotactic enumeration of every possible syllable (§5.4) is measured and not taken.
* **No recency window.** D-022's second share — an edge's share inside the last 10,000 traversals — is not carried; a primed tree could keep one, and does not yet (§19, question 8).
* **No `BACK`, no `THINK`.** The sentinels of `RadixCyclicNN` §5.1.1–§5.1.2 are learned from conversation; this model has no conversation yet.
* **No back-propagation.** Not through depth, not into the reward tree, not "just for the fall".
* **No least-punished ranking and no beams.** The worst-step ordering of `SPEC-LeastPunished.md` §3.2 and the count model's `K` best and `K` worst continuations are phase 3, once the two traversals exist.
* **No teachers called from here.** The tutor, the critic, the sandbox and the chat loop of `RadixCyclicNN` may *call* `reward` and `punish`; this model calls nothing.
* **No torch, no Go, no Rust, no HTTP API, no frontend** in this version (`PRD.md` phase P5).

---

## 19. Open questions

Honest gaps, to be resolved by measurement rather than by guessing now.

1. **What is an option?** (`PRD.md` §9.1) A lexicon-primed pair — every word, or every pronunciation, into a radix tree — is not complete, so its addresses are stored and radix compression returns. Everything from §7 on is unchanged. Whether that pair does anything the complete one does not is the most interesting question here and the one this version does not answer.
2. **Does priming beat growing on little data?** The hypothesis of `PRD.md` §2: with no structural decision to make, the model should be as good after one text as it will ever be on that text. Bits per unit after 1, 10, 100, 1000 texts against `RadixCyclicNN`'s count model on the same texts is the cheapest measurement once P2 runs.
3. **Should judged evidence count toward `own`?** (`PRD.md` §9.8) A context judged fifty times and read twice is trusted as a context read twice. Adding `plus + minus` to the evidence in `own` is a one-line change whose effect on S-5 and S-7 is unknown.
4. **One smoothing or two?** `σ` serves two masters: it keeps an unread step's share finite so a reward can lift it, and it changes the fold's numbers against `FilterBankRadix`. A reward tree with its own back-off — a reward read from the deepest level that has *any* credit, rather than the deepest level the count tree trusts — would let `σ` be `0` again. Untried.
5. **Should a deep reward weigh more than a shallow one?** `credit` writes the same amount at every level. The shallow levels then carry the sum of many outcomes and the deep ones a few, which is the generalisation S-7 measures — but it also means a specific context's judgement is drowned by the unigram's whenever the fold falls back. A per-level weight is a setting waiting for a reason.
6. **Should the sine kind learn the fall?** §12 keeps `own` on the counts. A learned fall — a weight per node competing with the children, trained by "the deepest level that knows best answers, everything deeper defers" — was the first draft's rule and is a guess; it is not taken without a measurement.
7. **Should the floor be a rung?** The floor is the one term of the answer that is not a path. A fall from the root to "nothing" with a weight of its own would make it one, and make every path's cost the fold's, which §10 currently only bounds.
8. **The recency window.** D-022's `R_recent` is what lets the family's count model move on from what it saw early. A primed tree could keep a per-node window count in one more array; whether recency matters on corpora that are read once is not known.
9. **Is one syllable, or one capped GPT-2 token, of context a language model worth having?** `L = 2` is the deepest either of the two decided codecs primes to under the ceiling, and the alternative — GPT-2's whole vocabulary at `L = 1` — is a unigram. The first `bench compare` on `syllables, L = 2` and `gpt2 top 2000, L = 2` against `chars, L = 4` and `phones, L = 3` on the same text decides where this model's home is.
12. **How is the full dictionary's syllables to be primed?** The core lexicon's 1,502 syllables prime to `L = 2`; the full CMU dictionary's would not. The `top` cap, a port that raises the ceiling, or a two-level vocabulary — the frequent syllables whole, the rare ones as their phonemes — are the three ways, and the last is the one the tokenizer's own idempotent text form makes possible.
10. **When to port.** `phones, L = 4` is 1.6 GiB and 72 million slots — Go or Rust, or torch over the arrays. The number that decides it is S-4's gap between `L = 3` and what a grown model reaches at the same memory.
11. **`<unk>`'s share.** A closed vocabulary on real text may route too much through `<unk>`; `info` reports it. Where the line is — at what share `<unk>` starts to carry the model — is not known, and `bytes` and `bpe` are the codecs that never produce one.
