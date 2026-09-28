# PrimedRadixPair — Design Specification

**Two radix trees, primed with every option, connected at every equal node.**

A radix tree that is **primed** holds every sequence of `1..L` units over a
closed alphabet before it sees a single text, and never changes its structure
again; a second primed tree over the same sequences reads them from the other
end; and the two are connected wherever they hold an equal sequence, at every
level. This document is the contract the code is written against. Read it
fully before writing code. `PRD.md` says what it is for and what will count as
success; this says what exactly the code must do.

Directory: `PrimedRadixPair/`. Python package: `radixpair`. Python 3.11+,
**standard library only**, no optional accelerator in this version,
deterministic given its inputs.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| **Two Radix Trees** | Tree **A** appends a unit to a sequence, tree **B** prepends one (§8.1). A path down A spells a sequence forwards, a path down B spells it backwards. Both are radix trees in the sense of every other one in this repository: a node *is* the sequence spelled so far. |
| **"Priming" the Radix Tree with every potential option** | `prime(alphabet, L)`: every sequence of length `0..L` over the `R` symbols of the alphabet exists from the start, as a slot in flat arrays. `N = (R^(L+1) − 1) / (R − 1)` nodes, a function of `(R, L)` alone. Nothing is ever added or removed (§7). |
| **brute-force insertion of every option** | A complete `R`-ary tree needs no pointers: the path to a node, read as a number in base `R` — the *radix* — is its address (§6). The literal brute-force insertion — every sequence, one at a time, with the standard radix insertion of `RadixTrieLLM_RNN/main.py` — is kept in `check.py` as the oracle the arithmetic is tested against (§6.4, §14). |
| **connect the 2nd Radix Tree to the first where the nodes are equal** | Two nodes are equal when they spell the same sequence. Every sequence has exactly one node in each tree, so the connection — a **rung** — is a bijection, and the same id names both ends (§8.1). Because every node is connected, the two trees are stored as one node set with two families of edges; the rung is the one place a connection carries a weight of its own (§11). |
| **at every single level/node, not just the final nodes** | The rungs at the two deepest levels are what let a walk slide its window: a node at level `L` has no children, so a forward walk crosses to B, steps toward B's root (dropping the *oldest* unit), and crosses back. Every shallower rung is a shorter memory the walk can fall back to. Connected only at the final nodes, the pair cannot continue past one window (§8.5). |
| A model of the family | `train / predict / generate / score`, shortest-path prediction over the structure (Dijkstra, as `RadixCyclicNN/DESIGN.md` §7), two kinds on one structure (the `count` kind first, the family's sine kind second, as `RadixCyclicNN/DECISIONS.md` D-020), JSON model files, a CLI, `unittest` with no dependencies. |
| No back-propagation | The count kind has no gradient at all. The sine kind's rule is the one-hop rule of `FilterBankRadix/DESIGN.md` §5.4, touching one node, its children and the far end of its rung; nothing multiplies a chain (§11). |
| Custom activation `−sin(x/3)` | The sine kind: every node owns `f(x) = a·sin(b(x − h)) + k`, initialised to `a = −1, b = 1/3, h = k = 0`, all four learnable (§11). |
| 2NRL | The sine kind: train on bad, `invert()`, fine-tune on good; the count kind does not invert (D-023) (§11.5). |

---

## 2. What this is, against what exists

**Grown against primed.** `RadixCyclicNN`'s graph and `FilterBankRadix`'s
trees are *grown*: a node exists because a text put it there, unary chains are
compressed away, and the structure is a record of what was seen. A primed tree
is the opposite end of that line: every context of every length up to `L − 1`
exists before any text, so no observation ever makes a structural decision.
There is nothing to compress — a complete tree has no unary chains — and no
lookup: the address of a sequence is arithmetic on its units (§6). What a
grown tree spends on structure, a primed tree spends on memory: `R^L` slots
whatever the data (§15).

**Two trees, one node set.** The requirement connects every node of B to its
equal in A. A connection that exists at every node and costs nothing to cross
is an identification: the two trees *are* one set of nodes with two sets of
edges. The design stores them that way (§7, §8.1) — one count per sequence,
read as "what follows it" by A and "what precedes it" by B — and keeps the
rung as the one place a connection can carry its own weight, which is what the
sine kind learns there (§11). Two separate node sets would hold the same
numbers twice.

**The rungs are the fold.** Falling back from a context to a shorter one is a
walk through the second tree (§8.2), and summed over every depth the walk can
fall to it is exactly the every-context-length prediction of
`FilterBankRadix/DESIGN.md` §5.3 (§8.3). That identity is the design's main
check: whatever the search does, the exact answer is known and the pair must
give it. What the pair adds over that fold is what a corridor cannot do and a
tree can: read from the other end (§8.4), and — the first follow-up — meet a
walk coming the other way (`PRD.md` §8, phase 3).

**What it is not.** It is not a de Bruijn graph and not `RadixCyclicNN`'s
gram graph, though at a fixed depth all three encode a sliding window: those
hold one order and the seen grams; this holds every order and every gram. It
is not a suffix automaton: no state is shared between different sequences.

---

## 3. Notation

| symbol | meaning |
|---|---|
| `Σ`, `R` | the alphabet and its size, marks included. Ids: `0 = <s>`, `1 = </s>`, `2 = <unk>`, then the units (§5). |
| `L`, `D` | the longest sequence held, and `D = L − 1` the longest context |
| `s`, `\|s\|`, `ε` | a sequence of unit ids, its length, and the empty sequence (the root, id `0`) |
| `s·x`, `x·s` | append `x` (an edge of A), prepend `x` (an edge of B) |
| `s[1:]`, `s[:-1]` | drop the oldest unit (B's step toward its root), drop the newest (A's) |
| `base(ℓ)`, `code(s)`, `id(s)` | the addressing of §6 |
| `N` | `base(L + 1)`, the number of nodes |
| `cnt[i]` | the number of occurrences of sequence `i` as a substring of the padded training texts (§7.2) |
| `Σ_A`, `Σ_B`, `R'` | what a forward walk can emit (`Σ` without `<s>`), what a backward walk can emit (`Σ` without `</s>`), and their size `R − 1` |
| `ctx_A(s)`, `ctx_B(s)` | `Σ_{x∈Σ_A} cnt[s·x]` and `Σ_{x∈Σ_B} cnt[x·s]`: how often `s` was a context in each direction (§7.3) |
| `p_A(x\|s)`, `p_B(x\|s)` | `cnt[s·x] / ctx_A(s)` and `cnt[x·s] / ctx_B(s)`; `0` when the context count is `0` |
| `own_A(s)` | `ctx_A(s) / (ctx_A(s) + ALPHA)`, `ALPHA = 2` — the share of the answer a context keeps for itself; `own_B` likewise |
| `FLOOR` | `0.02`, the uniform share mixed into every final answer. `ALPHA` and `FLOOR` are `FilterBankRadix`'s, so bits per unit are comparable (its invariant 5) |
| a padded text | `<s> u₁ … u_T </s>` — what `train` and `score` see (§5) |

---

## 4. Package layout

```
PrimedRadixPair/
  PRD.md                 what and why; the decisions and the success criteria
  DESIGN.md              this file — the contract
  README.md              the front page; the numbers, once there are numbers
  Makefile               make test | check | prime | train | predict | generate | score | info | bench | compare | localise
  radixpair/
    __init__.py          the public surface: Alphabet, PairModel, prime, load_model, Score, PathResult
    __main__.py          python3 -m radixpair -> cli.main()
    alphabet.py          the closed alphabet: presets, the marks, units(text) and text(units)
    address.py           the arithmetic of a primed tree: base, code, id, append, prepend, drop_*, level
    nodes.py             PrimedNodes: the node table, the counts, observe()
    tree.py              PrimedTree: one direction over the node table — children, parent, forget, own, dist
    pair.py              RadixPair: the two trees, the rungs, the corridor, the fold, scoring from both ends
    search.py            Dijkstra and sampling over the pair, forwards and backwards
    model.py             PairModel: the four verbs, settings, kinds, persistence
    checkpoint.py        CheckpointManager: rotation, the latest pointer, resume
    check.py             the brute-force oracle: prime a small tree literally and compare
    bench.py             throughput, compare (bits per unit against the references), localise
    cli.py               python3 -m radixpair <command>
    activation.py        (phase 4) the parametric sine and its partials
    sine.py              (phase 4) the sine kind: parameters, initialisation by id, the one-hop rule, invert
  tests/
    __init__.py          puts the project root on sys.path
    test_alphabet.py  test_address.py  test_nodes.py  test_pair.py  test_search.py
    test_model.py  test_localise.py  test_checkpoint.py  test_cli.py  test_bench.py  test_sine.py (phase 4)
```

---

## 5. `alphabet.py`

The alphabet is **closed and explicit**: fixed at priming, written into the
model file, and never extended. Three symbols are marks and come first, so
every preset agrees on their ids:

```python
START, END, UNK = 0, 1, 2

class Alphabet:
    name: str                  # "letters" | "phones" | "bytes" | "custom"
    symbols: list[str]         # symbols[i] is the text of id i; the marks first
    R: int                     # len(symbols)
    def units(self, text: str) -> list[int]       # normalised and mapped; nothing outside the alphabet survives (-> UNK); no marks
    def padded(self, text: str) -> list[int]      # [START] + units(text) + [END]
    def text(self, ids: Iterable[int]) -> str     # back to text; marks render as nothing, UNK as "?"
    def unk_share(self, ids) -> float
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> Alphabet
```

| preset | units | `R` | normalisation |
|---|---|---|---|
| `letters()` | space, `a`–`z`, `'`, `.`, `,` (30) | **33** | `casefold`; curly quotes to `'`; a run of whitespace to one space; everything else to `<unk>` |
| `phones(stress=False)` | the inventory of `../PhoneticTokenizer` — 39 phonemes, or with stress — plus its word gap `#` and its pause symbol, **taken from the tokenizer at priming** and written into the file, so the alphabet is whatever the tokenizer says it is | 44 without stress | text goes through the tokenizer exactly as `RadixCyclicNN`'s `Encoding(unit=PHONES)` reads it; `text()` joins units with spaces as that encoding does. Importable only when `PhoneticTokenizer` is; `test_alphabet.py` skips it otherwise |
| `bytes_()` | the 256 byte values | **258** | UTF-8; no `<unk>` — every byte is a unit, the marks are `0` and `1`, the bytes are ids `2..257`, so id `2` is a unit here (`has_unk = False`) |
| `custom(units)` | any list of distinct strings | `len + 3` | exact match per unit; anything else `<unk>` |

`bytes_()` is the one preset whose id `2` is a unit, not `<unk>`; `Alphabet`
carries `has_unk` so nothing tests for `UNK` by number. The marks are symbols
of the alphabet on purpose (`PRD.md` §9.6): the first unit of a text is
predicted from a context that says it is first, and the end of a text is a
unit the model learns to predict. A sequence with a mark *inside* it — `</s>`
anywhere but last, `<s>` anywhere but first — can never occur in a padded
text; its slot exists, stays at zero, and is never written to a file.

---

## 6. `address.py`

### 6.1 The numbering

Sequences are numbered by length, then lexicographically by unit id. The
first unit is the most significant digit:

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
(invariant 2) — every id is a sequence, so the arrays of §7 have no holes.

### 6.2 The moves

For a node `i` at level `ℓ` with `k = i − base(ℓ)` its code:

| move | id | who uses it |
|---|---|---|
| `append(i, ℓ, x)` = `base(ℓ+1) + k·R + x` | the child `s·x`, `ℓ < L` | A's edge; a forward walk emitting `x` |
| `prepend(i, ℓ, x)` = `base(ℓ+1) + x·R^ℓ + k` | the child `x·s`, `ℓ < L` | B's edge; a backward walk emitting `x` |
| `drop_newest(i, ℓ)` = `base(ℓ−1) + k // R` | `s[:-1]`, `ℓ ≥ 1` | A's parent; a *backward* walk forgetting |
| `drop_oldest(i, ℓ)` = `base(ℓ−1) + k mod R^(ℓ−1)` | `s[1:]`, `ℓ ≥ 1` | B's parent; a *forward* walk forgetting |
| `level(i)` | the `ℓ` with `base(ℓ) ≤ i < base(ℓ+1)`, by bisection over the `L + 2` bases | everything |
| `newest(i, ℓ)` = `k mod R`, `oldest(i, ℓ)` = `k // R^(ℓ−1)` | the last and first unit of `s` | decoding a path |

Two things worth noticing, both used in §7.3: A's children of `i` are the
**contiguous** block `base(ℓ+1) + k·R … + R − 1`, and B's children are the
**strided** block `base(ℓ+1) + k + x·R^ℓ`. Summing over either is one slice
of the count array.

```python
class Address:              # (R, L) bound once; bases and powers precomputed; every method is integer arithmetic
    R: int; L: int; N: int; bases: list[int]; powers: list[int]
    def of(self, seq: Sequence[int]) -> int ; def seq(self, i: int) -> tuple[int, ...] ; def level(self, i: int) -> int
    def append(self, i, ℓ, x) -> int ; def prepend(self, i, ℓ, x) -> int
    def drop_newest(self, i, ℓ) -> int ; def drop_oldest(self, i, ℓ) -> int
    def block_a(self, i, ℓ) -> tuple[int, int]        # (start, stop) of the contiguous append block
    def block_b(self, i, ℓ) -> tuple[int, int, int]   # (start, stop, step) of the strided prepend block
```

`Address` raises `ValueError` for `R < 2`, `L < 1`, a level out of range or a
unit outside `0..R−1`; it never returns an id outside `0..N−1`.

### 6.3 The rolling code

Counting a text needs the id of every suffix of length `1..L` at every
position. They roll: the last `ℓ` units ending at position `t` are the last
`ℓ − 1` units ending at `t − 1` followed by `u_t`, so

```
code_ℓ(t) = code_{ℓ−1}(t − 1) · R + u_t,     code_0 = 0
```

computed for `ℓ = min(L, t) … 1` in place, in that order (each `code_ℓ` reads
the previous position's `code_{ℓ−1}` before it is overwritten). `L`
multiplications and `L` increments per unit, no lookup, no allocation — the
loop of §7.2 and the whole of training.

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
   `append`, each `x·s` with `prepend`, the parent with `drop_newest`, and the
   node for `s[1:]` with `drop_oldest`; depth with `level`.

The test suite runs it at `(R, L) ∈ {(2,5), (3,4), (5,3), (7,2)}`; the CLI's
`check --R --L` runs it as large as memory allows. It is the definition of
correctness for §6, and §6 is the definition of the tree for everything else.

---

## 7. `nodes.py` — PrimedNodes

### 7.1 Storage

```python
class PrimedNodes:
    alphabet: Alphabet
    L: int
    address: Address
    cnt: array            # array('q'), N entries: cnt[i] = occurrences of sequence i as a substring of the padded texts
    texts: int            # texts observed
    units: int            # units observed, marks excluded
    unk: int              # of which <unk>
    version: int          # bumped by every observe(); the caches of §8 key on it
```

One `array('q')` of `N` signed 64-bit counts, allocated once at priming
(`array('q', bytes(8 * N))`), and nothing else per node in the count kind.
`prime` refuses an `N` above `Settings.node_ceiling` (default `4_194_304`)
with a `ValueError` naming `R`, `L`, `N` and the ceiling — the budget of §15
is enforced where the memory would be spent. Counts are plain integers: no
cyclic counter here, because `2^63` occurrences of one substring is not a
budget this design will reach before it is ported.

### 7.2 Counting — `observe`

```python
def observe(self, ids: Sequence[int]) -> int      # a padded text; returns the number of increments
def observe_text(self, text: str) -> int          # observe(alphabet.padded(text))
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
loop is the hot loop of §15: `bases`, `cnt`, `R` and `codes` are locals, and
there is no call, no lookup and no allocation inside it. Both trees are trained
by this one pass, because both read the same counts (§8.1). `observe` is
additive (a text twice doubles its counts) and touches no other state but the
three tallies and `version`.

### 7.3 The two context counts

```python
def ctx_a(self, i, ℓ) -> int      # Σ over A's children in Σ_A: sum(cnt[start + 1 : stop]) — the block minus <s>, which is never emitted
def ctx_b(self, i, ℓ) -> int      # Σ over B's children in Σ_B: sum(cnt[start : stop : step]) − cnt[the x = </s> entry]
```

Each is one slice sum over `R − 1` entries. For every sequence that does not
touch a mark the two are equal to each other and to `cnt[i]` (a substring
that is not at the very end of a text is followed by something, and one not
at the very start is preceded by something); `test_nodes.py` asserts it, and
the definition stays the sum so that the root and the marks need no special
case.

### 7.4 Invariants (asserted by `test_nodes.py`)

1. `len(cnt) == N` before and after any number of `observe` calls.
2. `cnt[id(s)]` equals a literal sliding-window count of `s` over the padded texts, for every `s` up to length `L`, on small texts.
3. `ctx_a(s) == ctx_b(s) == cnt[s]` for `1 ≤ |s| ≤ D` whenever `s` neither starts with `<s>` nor ends with `</s>`.
4. Every sequence with a mark inside it has `cnt == 0`.
5. `observe` is additive and bumps `version` exactly once.

---

## 8. `tree.py` and `pair.py` — the two trees and the rungs

### 8.1 One node set, two trees

```
        tree A  (append: a path reads forwards)         tree B  (prepend: a path reads backwards)

                    ε ============================ rung ============================ ε
                 /  |  \                                                         /  |  \
                a   b   c ===================== rungs ========================= a   b   c
              / | \                                                                 / | \
            aa ab ac ======================= rungs =============================== aa ba ca
                                                                                   ^
   A: ab's children are ab·a ab·b ab·c;  A's parent of ab is a         B: ab's children are a·ab b·ab c·ab; B's parent of ab is b
```

Every sequence is one node of A and one node of B, and the rung joins them.
Since that holds for every node, the implementation keeps **one** `PrimedNodes`
table and two `PrimedTree` views of it:

```python
FORWARD, BACKWARD = "forward", "backward"

class PrimedTree:                      # one direction over the node table; no storage of its own
    nodes: PrimedNodes ; direction: str
    def children(self, i, ℓ) -> list[tuple[int, int]]   # [(x, id)] over Σ_A (append) or Σ_B (prepend); [] at level L
    def parent(self, i, ℓ) -> int                       # toward THIS tree's root: drop_newest (A) / drop_oldest (B)
    def forget(self, i, ℓ) -> int                       # the far end of the corridor: drop_oldest (A) / drop_newest (B) — the OTHER tree's parent
    def ctx(self, i, ℓ) -> int ; def own(self, i, ℓ) -> float
    def dist(self, i, ℓ) -> list[float]                 # p_here over the tree's emitting alphabet; all zero when ctx == 0
    def emits(self) -> list[int]                        # Σ_A or Σ_B, as ids

class RadixPair:
    nodes: PrimedNodes
    forward: PrimedTree ; backward: PrimedTree
    rungs: str = "all"                                  # "all" | "final" — "final" exists only to show what it cannot do (§8.5)
    settings: Settings                                  # alpha, floor, backoff (§10)
    def tree(self, direction) -> PrimedTree
    def fold(self, direction, context: Sequence[int]) -> list[float]      # §8.3 — the exact next-unit distribution
    def score(self, text: str) -> Score                                    # §8.4
```

The rung's two ends have the same id, so the rung needs no storage in the count
kind. In the sine kind it is the weight `wr[i]` (§11).

### 8.2 The corridor

A forward walk standing on `s` in A wants to forget its oldest unit. A cannot
do that — A's parent of `s` is `s[:-1]`, the *newest* dropped. B can: B's
parent of `s` is `s[1:]`. So the walk goes

```
s (in A)  --rung-->  s (in B)  --B's parent-->  s[1:] (in B)  --rung-->  s[1:] (in A)
```

three hops, emitting nothing. That is the **corridor**, `PrimedTree.forget`,
and in the count kind its probability is the stay-or-cross decision at `s`:
`1 − own_A(s)`. A backward walk's corridor is the mirror image through A:
`s (B) → s (A) → s[:-1] (A) → s[:-1] (B)`, with `1 − own_B(s)`.

A node at level `L` has no children in either tree. A forward walk that has
just emitted the `L`-th unit of a window stands on one, and its only way on is
the corridor: cross at level `L`, step to level `D`, cross back, emit. Every
steady-state step of a walk is therefore *corridor, then emission* — the two
deepest rungs are crossed on every step, and the shallower ones whenever the
walk falls back further than one unit.

### 8.3 The fold — the exact next-unit distribution

Forward, from a context `c` of at most `D` units, for every `x ∈ Σ_A`:

```
P(x | c) = own_A(c) · p_A(x | c)  +  (1 − own_A(c)) · P(x | c[1:])          |c| ≥ 1
P(x | ε) = own_A(ε) · p_A(x | ε)  +  (1 − own_A(ε)) / R'
answer(x | c) = (1 − FLOOR) · P(x | c)  +  FLOOR / R'
```

Read as the walk it is: at `c`, stay with probability `own_A(c)` and emit `x`
with `p_A(x | c)`, or take the corridor with `1 − own_A(c)` and decide again at
`c[1:]`; at the root, taking the corridor means answering uniformly — knowing
nothing. The floor is the one term that is not a path (§18, question 6).

**This is `FilterBankRadix/DESIGN.md` §5.3, unrolled from the other end.** That
fold runs shortest level first, `p ← own · p_here + (1 − own) · p`, starting
from the prior `1/A`; expanding it from the longest level down gives the
recursion above term for term. The identity was checked while writing this
document — 200 random count tables, every context length, forward walk against
the shortest-first fold, identical to `1e−12` and summing to 1 — and that check
is `test_pair.py::test_the_rungs_are_the_fold`, with the shortest-first fold
copied into the test as the reference. Whatever `search.py` does, this is the
answer it approximates, and the pair must give it exactly.

Cost: `|c| + 1` levels × (one slice sum of `R'` counts + one read) = `O(D · R)`
per unit. `fold` caches `(ctx, own)` per `(node, direction)` in a dict keyed on
`nodes.version`; the cache is dropped when `version` changes.

Backward is the mirror: `c` is the units *following* the position, `p_B`,
`own_B`, `Σ_B`, and the corridor drops `c[:-1]`.

### 8.4 Scoring from both ends

For a padded text `u₁ … u_{T+2}` (`u₁ = <s>`, `u_{T+2} = </s>`):

```
f_t = −log₂ answer_A(u_t | the last ≤ D units before t)      t = 2 … T+2     (everything after <s>)
b_t = −log₂ answer_B(u_t | the first ≤ D units after t)      t = 1 … T+1     (everything before </s>)
both_t = min(f_t, b_t)                                        t = 2 … T+1     (the units themselves)
```

```python
@dataclass
class Score:
    forward_bits: float        # Σ f_t / (T + 1)
    backward_bits: float       # Σ b_t / (T + 1)
    per_unit: list[tuple[str, float, float, float]]   # (unit, f_t, b_t, both_t) for the T units
    units: int
```

Why the *lesser* of the two costs is the localiser: substitute one unit at
position `t`. The forward reading is surprised at `t`, and then at `t+1 … t+D`,
whose contexts contain the wrong unit (less and less, as the fold falls back to
contexts that no longer contain it). The backward reading is surprised at `t`
and at `t−D … t−1` for the same reason. Each reading blames a stretch on one
side of the mistake; **only the mistake itself is blamed by both**, so the
maximum of `both_t` is where the text is wrong. `PRD.md` S-6 measures the hit
rate; `bench localise` reports it beside the forward reading alone.

### 8.5 `backoff`, and why the final nodes are not enough

`Settings.backoff` selects how far a walk may fall, on the same counts:

| `backoff` | the answer | what it is |
|---|---|---|
| `all` (default) | the fold of §8.3 | every rung |
| `deepest` | `own(c) · p(x\|c) + (1 − own(c)) / R'` at the given `c`, then the floor | one fall, from the given context straight to knowing nothing — a pair joined at one level only. `FilterBankRadix` measured folding only the deepest level at 2.998 bits per character against 2.808 for every level (its §5.3) |
| `none` | `p(x\|c)`, then the floor | no fall: a table of the deepest context |

These are fold variants, computed for `PRD.md` S-5. The *structural* point is
separate and is asserted, not measured: `RadixPair(rungs="final")` keeps only
the level-`L` rungs, and a forward walk that reaches level `L` then crosses to
B, steps to level `D`, and finds no rung back. It can climb B to its root and
descend B's prepend edges — that is reading backwards — but it can never emit
forwards again. `test_pair.py::test_connected_only_at_the_final_nodes_cannot_slide`
asserts that `search.dijkstra` from any level-`L` node of such a pair emits
nothing, at any `min_units`. The rungs at every level are not an elaboration of
the requirement; they are what makes it a model.

---

## 9. `search.py` — walks over the pair

A walk lives on one tree and moves through the corridor when it must or when
it is cheaper to. State: `(node, level, emitted)`; the tree is fixed per
search. Moves from `(i, ℓ, e)` on tree `T` (forward shown; backward is the
mirror with `prepend`, `Σ_B`, `own_B`, `p_B`):

| move | to | cost | when |
|---|---|---|---|
| emit `x` | `(append(i, ℓ, x), ℓ + 1, e + 1)` | `−log(own(i) · p(x \| i)) + step_penalty` | `ℓ ≤ D`, `p(x \| i) > 0` |
| emit `x` from the root | `(append(0, 0, x), 1, e + 1)` | `−log(own(ε) · p(x \| ε) + (1 − own(ε)) / R') + step_penalty` | `ℓ = 0` — the root's fall to the uniform is folded into its emissions, so every unit has a finite cost somewhere |
| forget (the corridor) | `(forget(i, ℓ), ℓ − 1, e)` | `−log(1 − own(i))` | `ℓ ≥ 1` |
| emit `</s>` | goal | as emit | forwards; backwards the goal is `<s>` |

The search's costs are the fold's terms without the floor: a path's cost is
one choice of how far to fall at every step, and the floor — a mixture over
the whole answer — does not decompose along a path. Consequently the cheapest
path's cost is never below the un-floored fold's cost of the same units (it
picks the best fall instead of summing over falls); `test_search.py` asserts
both the decomposition and the inequality against `P(x | c)` before the floor.

```python
class PathResult:      # text: str, units: list[int], node_ids: list[int], hops: list[tuple[str, int]] (every hop,
                       # corridor hops included: ("emit", id) | ("rung", id) | ("up", id)), cost: float,
                       # step_costs: list[float] (one per emitted unit: its emission plus the corridors before it),
                       # expanded: int, reached_end: bool, full_text: str (set by PairModel: prefix + text)
                       # to_dict() -> JSON-serialisable

def dijkstra(pair, direction, context: Sequence[int], min_units: int, max_units: int | None = None,
             to_end: bool = False, step_penalty: float = 0.0, max_expansions: int = 200_000) -> PathResult
    # Start at the node of the last (first, backwards) ≤ D units of the context. heapq of
    # (cost, tie, node, level, emitted); best-cost dict keyed by (node, emitted) — level is a function of node.
    # Goal: to_end -> the end mark; else the first popped state with emitted >= min_units (Dijkstra pops in cost
    # order, so it is the cheapest such path); the end mark before min_units is also a goal. States with
    # emitted >= max_units are not expanded. Fallback on the expansion cap: the popped state with the most
    # emitted units (ties -> lowest cost) — never raise. Backwards, the units come out in reading order.

def greedy(pair, direction, context, units: int, rng=None, temperature: float = 0.0) -> PathResult
    # One unit at a time from the exact fold (answer(), floor included): the argmax at temperature 0, else a
    # sample from softmax(log answer / temperature) with the model's seeded rng. Stops at the end mark or at
    # `units`. This is `mode=greedy` and `mode=sample`; it is exact where dijkstra is a cheapest path.
```

Every hop of a walk is reported (`hops`), corridor hops included, so a walk can
be watched crossing and climbing the way `RadixCyclicNN`'s voice reports every
step; `step_costs` fold the corridors into the unit they precede so that the
per-unit numbers line up with `Score.per_unit`.

---

## 10. `model.py` — PairModel

```python
@dataclass
class Settings:
    alpha: float = 2.0            # ALPHA of own()
    floor: float = 0.02           # FLOOR of answer()
    backoff: str = "all"          # "all" | "deepest" | "none" (§8.5)
    step_penalty: float = 0.0
    max_expansions: int = 200_000
    node_ceiling: int = 4_194_304 # prime() refuses a larger N (§7.1)

class PairModel:
    kind: str                     # "count" | "sine"
    alphabet: Alphabet ; L: int ; seed: int ; settings: Settings
    nodes: PrimedNodes ; pair: RadixPair ; sine: SineParams | None
    history: list[dict]           # one record per train() call: texts, units, unk, increments (and the sine kind's epochs, losses)

    @classmethod
    def prime(cls, alphabet: Alphabet, L: int, kind: str = "count", seed: int = 0, settings: Settings | None = None) -> PairModel
    def train(self, texts: Iterable[str], progress: ProgressFn | None = None, **sine_options) -> dict
        # count kind: observe_text per text; returns {"texts", "units", "unk", "unk_share", "increments", "seconds"}
    def predict(self, prefix: str, length: int, mode: str = "dijkstra", direction: str = "forward",
                start: bool = True, temperature: float = 1.0, to_end: bool = False) -> PathResult
        # start=True: the prefix begins a text (padded with <s>); False: a fragment. The context is the last
        # (first, backwards) ≤ D units. mode: "dijkstra" | "greedy" | "sample". full_text = prefix + text
        # (text + prefix backwards).
    def generate(self, prefix: str = "", length: int = 60, **options) -> PathResult     # predict from <s> alone when prefix == ""
    def score(self, text: str) -> Score
    def distribution(self, context: Sequence[int], direction: str = "forward") -> list[float]   # the fold; for tests and the API
    def info(self) -> dict        # kind, alphabet, R, L, N, nonzero, texts, units, unk_share, settings, memory (bytes), history
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d) -> PairModel
    def save(self, path: str) -> None ; @classmethod load(cls, path: str) -> PairModel
```

The count kind has no learning rate, no epochs, no shuffling and no seed in
training; `seed` is used by `mode=sample` and by the sine kind's
initialisation. The same four verbs and the same result objects as the
family, so the CLI reads as `RadixCyclicNN`'s does.

---

## 11. The sine kind (phase 4) — `activation.py`, `sine.py`

The second kind on the same node set, the count kind's numbers being the
baseline it must match (`PRD.md` phase P4). Nothing in §5–§9 changes; what
changes is where `own` and `p` come from.

### 11.1 Parameters

Eight `array('d')` of `N` — 64 bytes per node:

| array | meaning |
|---|---|
| `z, a, b, h, k` | the node's state and its activation `f(x) = a·sin(b(x − h)) + k` |
| `wa[i]` | the weight of A's edge *into* node `i` from its A-parent — a tree has one parent per node, so the edge's weight lives on the child, as `FilterBankRadix` §5.1 |
| `wb[i]` | the weight of B's edge into `i` from its B-parent |
| `wr[i]` | the **rung** — the one weight a connection carries; the learned "forget" at `i`, shared by both directions (§18, question 4) |

`activation.py` repeats `sine`, `sine_partials` and `MIN_B = 1e-3` from
`RadixCyclicNN/radixnet/activation.py` rather than importing them; the
directory is standalone.

### 11.2 Initialisation by id — so the file can be sparse

A primed model must not write `N` random numbers to disk. Every initial value
is a **function of `(seed, i, j)`**, recomputed on load, so the file holds only
the nodes that moved (§12):

```
u(seed, i, j) = (splitmix64(seed · 2^32 + i · 4 + j) >> 11) / 2^53          uniform in [0, 1)
splitmix64(s): x = (s + 0x9E3779B97F4A7C15) mod 2^64
               x = (x ^ (x >> 30)) · 0xBF58476D1CE4E5B9 mod 2^64
               x = (x ^ (x >> 27)) · 0x94D049BB133111EB mod 2^64
               return x ^ (x >> 31)

z[i]  = −4.5 + 9 · u(seed, i, 0)          wa[i] = 0.5 + u(seed, i, 1)
wb[i] = 0.5 + u(seed, i, 2)               wr[i] = 0.5 + u(seed, i, 3)
a = −1, b = 1/3, h = 0, k = 0
```

The draws are the family's ranges (`z ~ U(−4.5, 4.5)`, `w ~ U(0.5, 1.5)`),
spelled out to the bit so that a port draws the same numbers.

### 11.3 Scores, and the learned fall

At a context `s` (level `ℓ ≤ D`) in A, the options are its `R'` children and
its rung:

```
score(x)    = wa[s·x] · f_s · f_{s·x}          x ∈ Σ_A
score(rung) = wr[s]   · f_s · f_{s[1:]}        (f of the corridor's far end; 1 at the root)
q           = softmax over the R' + 1 scores
own_A(s)    = 1 − q(rung)          p_A(x | s) = q(x) / own_A(s)
```

— *the activation of the child times the activation of the parent*, the
family's rule, with the rung as one more child. The fold, the corridor, the
search and `backoff` of §8–§9 then run unchanged on these `own` and `p`. The
mirror holds in B with `wb` and `s[:-1]`.

### 11.4 The one-hop rule (provisional — §18, question 5)

For one position with target `x` and the contexts `c_0 = ε, …, c_m` (`m =
min(D, t − 1)`), compute every level's `q`. Let `k* = argmax_k q_k(x)`, the
level that knows `x` best. Targets: at levels `k ≤ k*`, the child `x`; at
levels `k > k*`, the rung — *the deepest level that knows best answers, and
everything deeper defers*. Then per level, `g = q − onehot(target)` and

```
∂L/∂wa[c·y] = g_y · f_c · f_{c·y}        ∂L/∂wr[c] = g_r · f_c · f_far
∂L/∂f_c     = Σ_y g_y · wa[c·y] · f_{c·y} + g_r · wr[c] · f_far
∂L/∂f_{c·y} = g_y · wa[c·y] · f_c         ∂L/∂f_far = g_r · wr[c] · f_c
```

chained through `sine_partials` into `z, a, b, h, k` of `c`, its children and
the far node. Summed over a batch, divided by its size, clipped to `±clip`
(5), applied once; the activation parameters move at `ACT_RATE = 0.1` of the
weights' rate and `b` is floored at `MIN_B`. Both trees per position — B's
contexts are the units after the position, its target the unit at it. Every
gradient is one hop: a node, its children, the far end of its rung; nothing
crosses further. `check.check_sine` reads the analytic gradient out of `step`
itself and compares it with central differences.

### 11.5 `invert`, 2NRL

`invert()`: `wa, wb, wr → −w` for every node; `a → −a` and `k → −k` — the
exact negation of the unit, as `RadixCyclicNN/DESIGN.md` §5.2 explains it;
`inverted = not inverted`. Twice is bit-identical. `two_nrl(bad, good, lr,
fine_lr)`: train on `bad` at `lr`, `invert()`, train on `good` at `fine_lr`.
The count kind neither inverts nor has 2NRL in this version (D-023; §17).

---

## 12. Persistence

JSON, format `"radixpair"` version `1`, gzip when the path ends in `.gz`;
`load` sniffs the gzip magic and ignores the suffix. Written through a
temporary file and `os.replace`, as `RadixCyclicNN`'s `write_bytes_atomic`.

```json
{"format": "radixpair", "version": 1,
 "alphabet": {"name": "letters", "symbols": ["<s>", "</s>", "<unk>", " ", "a", "..."]},
 "L": 4, "kind": "count", "seed": 0,
 "settings": {"alpha": 2.0, "floor": 0.02, "backoff": "all", "step_penalty": 0.0, "max_expansions": 200000, "node_ceiling": 4194304},
 "trained": {"texts": 12, "units": 913, "unk": 3},
 "history": [{"texts": 12, "units": 913, "unk": 3, "increments": 3796, "seconds": 0.01}],
 "counts": {"ids": [1, 4, 17], "values": [12, 3, 1]},
 "sine": {"inverted": false, "ids": [], "z": [], "a": [], "b": [], "h": [], "k": [], "wa": [], "wb": [], "wr": []}}
```

* `counts.ids` are sorted and hold **only the non-zero counts**; `sine.ids`
  only the nodes whose parameters differ from §11.2's initial values (the
  `sine` block is absent in the count kind).
* `load` is `prime(alphabet, L, kind, seed, settings)` followed by writing the
  entries back. The stored `symbols` list is authoritative — a preset changing
  later does not change an old model — and `R`, `N` are recomputed, never read.
* A file never holds `N` of anything: its size is proportional to what was
  seen (`PRD.md` S-7), and a model of ten sentences is a few kilobytes at any
  `L`.
* `checkpoint.py` is `RadixCyclicNN`'s `CheckpointManager` shape: rotation, a
  `latest` pointer, `load_latest`, resume; a checkpoint is a model file.

---

## 13. `cli.py` and the `Makefile`

```
python3 -m radixpair prime    --model m.json --alphabet letters|phones|bytes --L 4 [--kind count|sine] [--seed 0] [--ceiling N] [--unit ...]
python3 -m radixpair train    --model m.json --data corpus.txt [--data more.txt ...] [--checkpoint-dir DIR]
                              [sine kind: --epochs 5 --lr 0.05 --act-lr 0.005 --batch 256 --clip 5]
python3 -m radixpair predict  --model m.json --prefix "the quick brown" --length 20 [--mode dijkstra|greedy|sample]
                              [--direction forward|backward] [--temperature 1.0] [--fragment] [--to-end] [--hops]
python3 -m radixpair generate --model m.json [--prefix ""] [--length 60] [the predict options]
python3 -m radixpair score    --model m.json --text "..." | --data file [--per-unit]
python3 -m radixpair info     --model m.json
python3 -m radixpair check    [--R 3 --L 4]                                   the brute-force oracle (§6.4)
python3 -m radixpair bench    [--alphabet letters --L 4 --units 200000]       throughput (§15)
python3 -m radixpair bench compare  --data corpus.txt [--holdout 0.1] [--L 4]  bits per unit: all / deepest / none, and the references (PRD S-4, S-5)
python3 -m radixpair bench localise --data corpus.txt [--n 200]                the planted-substitution hit rate (PRD S-6)
python3 -m radixpair invert / 2nrl  (sine kind)
```

`--json` on every command prints one JSON document to stdout and progress to
stderr, as the family does. `--data` reads one text per line, blank lines
skipped. The `Makefile` has a target per command with the variables
overridable on the command line (`make train DATA=… L=4`), plus `test`
(`python3 -m unittest discover -s tests`) and `check`.

---

## 14. Tests (`unittest`, no dependencies, seconds)

* `test_alphabet.py` — the presets' `R` and mark ids; normalisation; `<unk>`; `padded`; the round trip; `to_dict` / `from_dict`; `phones` skipped when the tokenizer is not importable; `bytes_` has no `<unk>`.
* `test_address.py` — **the brute-force oracle** at `(2,5), (3,4), (5,3), (7,2)` (§6.4); the worked example of §6.1; `level` by bisection at every id; A's blocks contiguous and B's strided; every move undone by its inverse (`append` then `drop_newest`, `prepend` then `drop_oldest`); the refusals.
* `test_nodes.py` — the invariants of §7.4; the rolling code against a naive recount; the ceiling refused with the numbers in the message.
* `test_pair.py` — `test_the_rungs_are_the_fold`: the pair against the shortest-first fold of `FilterBankRadix` §5.3 (copied into the test), 200 random tables × every context length, to `1e-9`, and every answer a distribution; `deepest` and `none` as §8.5 defines them; **backward is forward reversed** (§16 invariant 5); the corridor's three hops and its far end; forgetting at the root is the uniform; `test_connected_only_at_the_final_nodes_cannot_slide`; the cache dropped on `version`.
* `test_search.py` — Dijkstra returns a trained text's continuation; a path's cost is the sum of its emissions and corridors and is never below the un-floored fold's cost of the same units; a start at level `L` forgets first; the fallback on the expansion cap never raises; `to_end`; backward generation puts its units before the context in reading order; `greedy` at temperature 0 is the fold's argmax; `sample` is seeded and terminates.
* `test_model.py` — train lowers a trained text's bits; a trained text costs fewer bits than garbage, in both directions; `predict` reproduces a training continuation; `start` against `fragment`; `history`; save / load identical predictions, scores and `info`; the file has no zero count; file size grows with the non-zero counts and not with `L`; `distribution`.
* `test_localise.py` — a planted substitution in a text the model was trained around is the maximum of `both`; over 100 sentences the `both` hit rate is at least the forward-only hit rate.
* `test_checkpoint.py` — rotation, the latest pointer, `load_latest`, resume.
* `test_cli.py` — subprocess smoke of `prime / train / predict / generate / score / info / check` with `--json`.
* `test_bench.py` — every bench runs in `--quick`, and `compare` reports the three `backoff` settings.
* `test_sine.py` (phase 4) — the default equals `−sin(x/3)` and the partials pass a finite-difference check; initialisation is a function of `(seed, id)` and the same across processes; the one-hop rule's analytic gradient, read out of `step`, against central differences; the rung's target rule on a hand-built case; `invert` twice is identity; 2NRL makes garbage less likely; the file holds only the nodes that moved and loads to identical predictions.

---

## 15. Performance notes (must be followed)

### 15.1 The budget

`N` nodes per model; the count kind at 8 bytes per node, the sine kind at 64:

| alphabet | `R` | `L = 3` | `L = 4` | `L = 5` |
|---|---|---|---|---|
| `letters` | 33 | 37,060 · 0.3 MiB | **1,222,981 · 9.3 MiB** (sine 75 MiB) | 40,358,374 · 308 MiB (sine 2.4 GiB) — a port's job |
| `phones` | 44 | 87,165 · 0.7 MiB | 3,835,261 · 29 MiB (sine 234 MiB) | 168,751,485 · 1.3 GiB — no |
| `phones(stress=True)` | 94 | 839,515 · 6.4 MiB | 78,914,411 · 602 MiB — no | — |
| `bytes_` | 258 | 17,240,335 · 132 MiB — a port's job | — | — |

The ceiling of §7.1 (`4,194,304` nodes) admits the bold cell and `phones, L =
4`, and refuses the rest until there is a port or a torch backend to run them.
`L` is the longest sequence; the longest context is `L − 1`.

### 15.2 What was measured while writing this

Pure Python, `array('q')`, one process, the container this document was
written in — a scratch figure that `bench` replaces:

| | |
|---|---|
| `letters, L = 4`, 200,000 random units | 800,000 increments in 0.33 s |
| increments per second | **2.43 million** |
| units per second | 609 thousand |

`PRD.md` NFR-3 asks for 1 million increments per second; there is margin.

### 15.3 Rules

* The counting loop of §7.2 binds `cnt`, `bases`, `R` and `codes` to locals and contains no call, no attribute lookup and no allocation. It is the whole of training and the one loop that matters.
* Context counts are slice sums over `array` (`sum(cnt[a:b])`, `sum(cnt[a:b:step])`), never Python loops over children; `(ctx, own)` is cached per `(node, direction)` keyed on `nodes.version`.
* The fold allocates one list of `R'` floats per level and reuses it; nothing per unit inside.
* Dijkstra uses `heapq` with `(cost, tie, node, level, emitted)` tuples and a `best` dict; `max_expansions` is a guard, never a loop bound the search grows into.
* No per-node objects, ever. `N` is a million; a million Python objects is the one way to make this design look slow.
* `prime` allocates its arrays with `array(code, bytes(size * N))` — one allocation per array, zero-filled by the C library.

---

## 16. Invariants

Asserted in the suite, true at every point in a model's life:

1. **Primed means fixed.** `N = (R^(L+1) − 1) / (R − 1)`; no operation changes the length of any array; the arithmetic tree is the brute-force tree (§6.4).
2. **Every id is a sequence.** `id` is a bijection from the sequences of length `0..L` onto `0..N−1`, and every move of §6.2 lands inside it.
3. **Counts are substring counts.** `cnt[s]` is the number of occurrences of `s` in the padded training texts; child sums equal the node's count except across a mark; a sequence with a mark inside stays at zero (§7.4).
4. **The rungs are the fold.** The forward fold equals `FilterBankRadix` §5.3 on the same counts to `1e-9`, and every answer is a distribution over `Σ_A` (`Σ_B` backwards).
5. **Backward is forward reversed.** For a model `M` and a model `M'` primed alike and trained on the reversed texts (marks swapped), `M.score(text).backward_bits == M'.score(reversed text).forward_bits` to `1e-9`, unit by unit.
6. **The corridor is the only way between the trees, and it needs the rung it returns by.** With rungs only at level `L`, a forward walk from any level-`L` node emits nothing.
7. **Determinism.** Same alphabet, `L`, texts and seed ⇒ the same bytes, across processes and across a save / load cycle; `mode=sample` with the same seed gives the same walk.
8. **The file is the data.** A restored model predicts and scores identically; the file holds no zero count and no unmoved parameter.
9. **The sine kind.** `invert` is an involution; every gradient is one hop; the activation parameters move at a tenth of the weights' rate; `b ≥ MIN_B`; the initial value of every parameter is a function of `(seed, id)`.

---

## 17. Deliberately absent

Named so that adding one is a decision rather than a drift.

* **No growth.** No split, no merge, no dynamic window, no node after priming. The moment a node is created because a text needed it, this is `RadixCyclicNN`.
* **No words as units.** "Every option" over words is unbounded.
* **No `BACK`, no `THINK`.** The sentinels of `RadixCyclicNN` §5.1.1–§5.1.2 are learned from conversation; this model has no conversation yet.
* **No back-propagation.** Not through depth, not across the corridor, not "just for the rung".
* **No torch, no Go, no Rust, no HTTP API, no frontend** in this version (`PRD.md` phase P5).
* **No meeting walk.** Fill-in-the-middle and correction are the first follow-up and get their own `SPEC-MeetingWalk.md`; nothing here forecloses them — the pair's geometry is theirs.
* **No count-kind inversion, no count-kind 2NRL** (D-023).
* **No teachers.** The tutor, the critic and the negative network of `RadixCyclicNN` may *call* this model as a judge; it does not call them.

---

## 18. Open questions

Honest gaps, to be resolved by measurement rather than by guessing now.

1. **What is an option?** (`PRD.md` §9.1) A lexicon-primed pair — every word into a prefix tree and, reversed, into a suffix tree, connected where a sequence is both a prefix of some word and a suffix of another — is not complete, so its addresses are stored, radix compression returns, and the rungs are sparse and meaningful. Everything from §8 on is unchanged. Whether that pair does anything the complete one does not is the most interesting question here and the one this version does not answer.
2. **Does priming beat growing on little data?** The hypothesis of `PRD.md` §2: with no structural decision to make, the model should be as good after one text as it will ever be on that text. Bits per unit after 1, 10, 100, 1000 texts against `RadixCyclicNN`'s count model on the same texts is the cheapest measurement once P2 runs.
3. **Is `L = 4` over letters deep enough to be worth 9 MiB, or is the model's home sounds at `L = 3–4`?** S-4 answers the first half; the second needs a phonetic corpus scored both ways.
4. **One rung weight for both directions.** §11.1 shares `wr[i]` between a forward fall and a backward fall. Whether the two propensities are the same thing — how much `s` knows — or should be two weights is a measurement on the sine kind.
5. **The rung's target rule** (§11.4) is a guess: "the deepest level that knows best answers, everything deeper defers". The alternatives are training the rung toward the count-derived `own` (which makes the sine kind a curve fit of the count kind), or a sum-over-paths rule (which would be the first rule in the family that is not one hop, and is not taken without a reason).
6. **Should the floor be a rung?** The floor is the one term of the answer that is not a path. A rung from the root to "nothing", with a weight of its own, would make it one and make it learnable; it would also make every path's cost the fold's, which §9 currently only bounds.
7. **`<unk>`'s share.** A closed alphabet on real text may route too much through `<unk>`; `info` reports it, and the `letters` preset is widened if it is large. Where the line is — at what share `<unk>` starts to carry the model — is not known.
8. **The meeting walk's cost.** The sum of the forward and backward walks' costs to the equal node they meet on, or the fold of both readings at every filled unit? The former is a shortest path and the family's shape; the latter is exact. `SPEC-MeetingWalk.md` decides.
9. **When to port.** `letters, L = 5` is 308 MiB of counts and 40 million slots — Go or Rust, or torch over the count arrays. The number that decides it is S-4's gap between `L = 4` and what a grown model reaches at the same memory.
10. **Does the corridor need a direction in the count kind?** `ctx_A` and `ctx_B` differ only at the marks, so `own_A ≈ own_B` everywhere else; the two decisions could be one number. Kept as two until measured, because the marks are where texts begin and end and that is not nowhere.
