# RadixAcyclicNN

**`RadixCyclicNN`'s network, kept as a tree.** The same sliding window of three
characters, the same sine activation on every node, the same one-hop learning
rule, the same `-log P` cost and shortest-path prediction, the same radix
split and merge, the same inversion for 2NRL — with one thing changed. In the
cyclic graph a node *is a gram*, so the second `"aaa"` in `"aaaa"` is an edge
back to the first and the graph has cycles. Here a node *is a context*, the
path from the root, so the second `"aaa"` is a different node from the first
and every edge runs one level deeper. The structure is a rooted tree. It
cannot contain a cycle, and the invariant checks prove that after every
operation the tests make.

`Research/CyclesAreAFeature.md` argues that the cycles are the point. This
directory is the control that argument needs: the same model without them,
measured on the same text. What it found is below — the cycle buys exactly
what the paper says it buys, and the tree buys some things the paper does not
mention.

Pure Python, standard library only, sized like `FilterBankRadix/`: a package
(`radixtree`), a CLI, a Makefile, tests, and the comparison.

## The ideas, in one table

| Requirement | Implementation |
|---|---|
| Encoding: sliding window of 3 characters | `Encoding().encode("hello") -> ["hel", "ell", "llo"]`, stride 1; `Encoding(n=1)` is the plain character trie. |
| Decoding | `decode_path`: the first label in full, then the new character of every following node label — the cyclic model's decoder. |
| A radix tree with no cycles | A node is a context. Every text is inserted from START, and from the root at every position (every suffix, the way a suffix tree holds a text). A run of nodes with one child each is one node (path compression); a transition into the middle of a run splits it. Every edge runs from a node to a strictly deeper one; a walk from the root visits every node exactly once. |
| Sine activation, learned per node | `f(z) = a * sin(b * (z - h)) + k`, defaults `-sin(z / 3)`; all four parameters learn. Repeated rather than imported from `radixnet` so the directory stands alone. |
| The one-hop local rule | For a transition `p -> c` the loss is `-log softmax` of `w_c * f_p * f_c` among the options at `p`; the gradient touches `w` on the children's in-edges and `z, a, b, h, k` of `p` and its children, nothing deeper. Checked against finite differences to ~1e-9 (`make check`). A corpus position trains **every** context that predicts it. |
| Shortest-path prediction | Dijkstra from the deepest context of the prefix, cost `-log P + step_penalty`. No `(node, chars_emitted)` state, no expansion budget: a node is reached by one path, so it is pushed once and the search ends when the subtree does. |
| Negative costs | Legal here. `costs="signal"` walks by the raw edge signal `-(w * f_p * f_c)` and `step_penalty` may be below zero — no cycles, no negative cycles. The cyclic search must refuse both. |
| Scoring | The deepest context decides; a gram it was never followed by costs `UNKNOWN_PROB = 1e-6` — the cyclic model's convention, so the two read on one scale. `min_count` says how often a context must have been seen to be trusted. |
| Inversion and 2NRL | `invert()` negates every weight and every activation. `invert_paths` flips every other node of a text's path — **exact** on a tree (it is bipartite and each END leaf is private), best-effort on the graph. `two_nrl`: train on the bad, invert, fine-tune on the good. |
| Bounded windows | `depth=8` caps a root path at eight symbols. A walk that runs out of its window re-enters the tree at the deepest context of what it has said — a loop *outside* the tree, with `max_legs` as its clock. Unbounded, a walk never needs to. |
| Token by token | `mode="slide"`: one query of the tree per token — the deepest usable context of the last `window` grams, or of everything said so far — one decision from that context's options (the most likely token at temperature 0, a sample above it), the token fed back, the next query from the new context. A window that forgets can go round; `MAX_SLIDE` is its clock. |
| Sounds | `Encoding(unit=PHONES)` or `SYLLABLES`: the text is read through the sibling phonetic tokenizer (`PhoneticTokenizer/`), a gram is `n` sounds, the tree is built over sounds, and a prediction is spelled back into words (`spelled`, `full_spelled`). |
| Persistence | JSON, gzipped on request, written atomically; the RNG state travels, so training after a load is the training that would have followed a save. |

## Quick start

```bash
cd RadixAcyclicNN
make test            # 71 tests, standard library, about half a minute
make check           # the rule against finite differences
make demo            # train on the sample corpus and watch it recite
make compare-quick   # both models on the sample corpus, seconds
make compare         # both models on the prose corpus, one fifth held out - a quarter of an hour
```

Standard library only. No install, no download. The comparison imports the
cyclic model from the sibling `../RadixCyclicNN` checkout and says so if it
cannot.

```python
from radixtree import RadixTreeNet

model = RadixTreeNet(seed=1)                       # depth=None: every whole suffix; min_count=1
model.train(texts, epochs=10, lr=0.5, batch_size=32)
model.predict("the quick", to_end=True).text      # ' brown fox jumps over the lazy dog'
model.score("the quick brown fox jumps over the lazy dog")
model.invert_paths(["a text that was wrong"])     # every edge of its path changes sign
model.save("model.json.gz")
```

## What it measured

Both models, the same texts, the same numbers: seed 1, 10 epochs, `lr 0.5`,
`act_lr 0.05`, batch 32, the tree unbounded and trusting every context it has
seen. The tree's half is `radixtree`; the graph's is `radixnet.RadixNet` from
`../RadixCyclicNN`. Every number is in `results/`, with the SHA-256 of the
text it was measured on.

### The sample corpus

`data/sample_corpus.txt`: 60 sentences, 1 800 characters, nothing held out
(`make compare-quick`).

| | cyclic graph | tree |
|---|---:|---:|
| real nodes | 466 | 2 224 |
| END leaves | 1 | 1 682 |
| edges | 824 | 3 906 |
| distinct grams | 715 | 715 |
| label characters | 1 647 | 28 708 |
| grams per node | 1.53 | 0.32 |
| transitions per epoch | 1 418 | 5 810 |
| training seconds | 0.08 | 1.87 |
| train bits/char | 0.943 | 0.276 |
| texts recited whole from their first 8 characters | 1 / 60 | 57 / 60 |

| prefix | cyclic graph says | tree says |
|---|---|---|
| `the quick` | `''` | `' brown fox jumps over the lazy dog'` |
| `the cat` | `' gold'` | `' sat on the mat'` |
| `knowledge` | `' is mat'` | `' is power'` |
| `two plus` | `' forest'` | `' two equals four'` |
| `the moon` | `''` | `' orbits the earth'` |
| `th` | `'e quick'` | `'e quick brown fox jumps over the lazy dog'` |
| `zzzq` | `'uals nine'` | `'where there is a will there is a way'` |

The three sentences the tree does not recite share their first eight
characters with another (`the earl` opens both *the early bird* and *the earth
orbits*); at that branch the softmax picks one. The graph, being first-order,
forgets the sentence it is in at every shared gram: after `the cat` the most
likely thing it knows is `gold`.

### The prose corpus

`data/corpus.txt`: 800 lines of prose from the four research papers, 640 to
train on (44 335 characters) and every fifth line held out (160 lines, 10 897
characters). `make compare`.

| | cyclic graph | tree |
|---|---:|---:|
| real nodes | 3 516 | 61 213 |
| END leaves | 1 | 42 824 |
| edges | 9 699 | 104 037 |
| distinct grams | 4 228 | 4 228 |
| label characters | 11 260 | 1 510 919 |
| grams per node | 1.20 | 0.07 |
| transitions per epoch | 42 310 | 219 953 |
| training seconds | 9 | 871 |
| train bits/char | 1.924 | 0.145 |
| held-out bits/char | 4.057 | 6.514 |
| held-out miss rate | 12 % | 54 % |
| lines recited whole from their first 8 characters | 0 / 640 | 572 / 640 |

| prefix | cyclic graph says | tree says |
|---|---|---|
| `the quick` | `'ets,'` | `' 45 because it sounded round. It is the largest rotation a sine'` |
| `the cat` | `'e.'` | `'ches the loop directly:'` |
| `knowledge` | `'t'` | `'". It is not sloppiness. It is the only thing that can be said in a graph'` |
| `th` | `'e'` | `'eory predicts). I then test whether dividing the decay back out actually helps,'` |

The graph's continuations are short because its search prefers short, confident completions and the END edge is cheap (`RadixCyclicNN/DECISIONS.md` D-008); the tree's are the rest of the line the prefix landed in.

### Repetition, measured

The counting argument of `Research/CyclesAreAFeature.md` §3, run rather than
argued. Each text is shown once; the columns are what each structure needs to
hold it.

| text | chars | cyclic nodes | tree nodes | cyclic label chars | tree label chars |
|---|---:|---:|---:|---:|---:|
| `a` × 4 | 4 | 1 | 3 | 3 | 10 |
| `a` × 10 | 10 | 1 | 9 | 3 | 34 |
| `a` × 100 | 100 | 1 | 99 | 3 | 394 |
| `a` × 1000 | 1000 | 1 | 999 | 3 | 3 994 |
| `abc` × 2 | 6 | 2 | 5 | 7 | 23 |
| `abc` × 5 | 15 | 2 | 14 | 7 | 77 |
| `abc` × 50 | 150 | 2 | 149 | 7 | 887 |
| `abc` × 500 | 1500 | 2 | 1 499 | 7 | 8 987 |

Shown `aaaa` once and asked about `aaaaaaa`:

- the cyclic graph represents it — one node and a self-loop — and scores it
  with **0 unknown** of 6 transitions, log-probability −3.59;
- the tree does not: `aaaaaaa` is not a path of it, and the score charges
  **3 unknown** of 5 transitions, log-probability −41.45.

That is the paper's Prediction 1 — *the cyclic representation stays bounded on
inputs where the acyclic one grows linearly* — confirmed to the node.

## What a cycle buys, and what it costs

Read the tables together and the trade is plain.

**The cycle buys a finite structure and free generalisation.** A repetition is
one node with an edge back to it; the tree pays a node per occurrence and can
say nothing about a length it has not seen. On the prose corpus the graph holds
4 228 distinct grams in 3 516 nodes (1.2 grams per node); the tree holds the
same grams in 61 213 nodes, because a gram lives in every context it occurs in
(0.07 grams per node — fourteen contexts each, on average), and its labels run
to 1.5 million characters for a 44-thousand-character corpus. The graph trains
on 42 310 transitions an epoch; the tree, which trains every context a position
passes through, on 219 953 — and its root alone, the empty context, has 4 228
options in nearly every batch. Two orders of magnitude of training time follow.

**The tree buys certainty.** A context here is the whole history, so once a
walk is inside a line it knows the rest of it: the tree recites 57 of 60 sample
sentences and 572 of 640 prose lines from eight characters, at 0.28 and 0.15
bits per character on the training text, where the graph recites 1 and 0 at
0.94 and 1.92. The prediction search visits two to five nodes; the graph's
unrolled search visits four to seventy-three for the same prefixes, and has to
carry the characters emitted in its state to stay finite.

**On text it has not seen, the certainty is the problem.** The deepest context
decides, and on a tree of every suffix the deepest context has usually been
seen once; a held-out line that walks into that corridor and leaves it pays a
miss, then starts again from the longest suffix it knows. At `min_count 1` the
tree misses on 54 % of its held-out transitions against the graph's 12 %, and
its held-out bits per character are worse for it. The knobs the design has for
this are measured below: trust nothing seen once (`min_count 2`), bound the
window (`depth 8`), and bound it to the graph's own order (`depth 2`).

| | real nodes | label chars | training s | train bits/char | held-out bits/char | held-out misses | recited |
|---|---:|---:|---:|---:|---:|---:|---:|
| cyclic graph | 3 516 | 11 260 | 9 | 1.924 | 4.057 | 12 % | 0 / 640 |
| tree, unbounded, `min_count 1` | 61 213 | 1 510 919 | 871 | 0.145 | 6.514 | 54 % | 572 / 640 |
| tree, unbounded, `min_count 2` | 61 213 | 1 510 919 | 879 | 1.288 | 5.629 | 32 % | 198 / 640 |
| tree, `depth 8` | 51 609 | 276 651 | 641 | 0.296 | 6.360 | 51 % | 17 / 640 |
| tree, `depth 2` | 12 006 | 38 338 | 248 | 1.843 | 3.832 | 11 % | 0 / 640 |

Read down the table. Bounded to two symbols — a context is one gram, the
graph's own order — the tree holds the graph's information as a tree, in three
times the nodes because it has no corridors, and predicts the held-out lines
slightly *better* than the graph (3.83 against 4.06 bits per character, 11 %
against 12 % misses): a corridor is where a held-out line goes wrong, and at
this depth there are none. It recites nothing, as the graph recites nothing.
At eight symbols the labels fall by a factor of five and the training text is
certain again (0.30 bits per character, no misses), but the window is too short
to recite a line (17 of 640) and the held-out misses barely move (51 %): a
context of seven grams has usually been seen once too. Unbounded and trusting
only what it has seen twice, the tree gives up most of its recitation (198
lines) and still misses on a third of the held-out transitions — a context
seen twice has seen the continuations of those two occurrences and no others,
where the graph's node for a gram has seen every continuation of that gram
anywhere in the corpus. That is what the cycle is, read as statistics: **a
merge of every context a gram occurs in**, and merging contexts is what
generalises to text the model has not seen. The tree can only approach it by
throwing its depth away.

**What the tree has that the graph cannot.** No clock and no budget in the
search — "keep going until there is nowhere to go" is a complete algorithm. A
signed cost, and a negative step penalty, both of which the cyclic search must
refuse. A topological order (depth). And an exact targeted inversion: flipping
every other node of a path flips every edge of it, END leaf included, where the
paper shows the graph's version is best-effort — an odd cycle cannot be
two-coloured and a self-loop is sign-locked (`Research/CyclesAreAFeature.md`
§5.3). `make test` asserts all four.

**What 2NRL needs on a tree: a branch.** The negative phase can only teach
where a node has more than one option, and a garbage line the tree has never
seen is a corridor — no branch, no gradient — so the failures have to be shown
to a model that already holds the good texts, where the swapped word is a real
branch. There the inversion reverses every pairwise preference exactly, and the
positive phase leaves the good continuation preferred at 35 of 38 such
branches (`test_two_nrl`). In the cyclic graph, whose nodes are shared by every
sentence that uses the gram, the negative phase always has a branch to teach.

So: the cycle is what makes the structure finite and general; the tree is what
makes it certain and exact. The tree is the control that makes the paper's
claims measurable. It is not the better model of text, and the numbers say so.

## Token by token

`predict(prefix, mode="slide", window=k)` is the loop you would write by hand:
query the tree once with the last `k` grams of the context (everything said so
far when `k` is `None`), take the token it gives, append it, query again.
Inside a compressed run the token is fixed; at a branch it is the most likely
option at temperature 0, or a sample above it. With the whole history as
context the walk follows the deepest path the tree has and recites, as the
cheapest path does; with a window it decides afresh at every token and can
forget its way into a loop (`…sets in the east and sets in the east…`), which
is the cycle coming back outside the tree — `MAX_SLIDE` is its clock.

How deep the window should be is a measurement, and the unbounded tree holds
every depth at once. Next-gram prediction on the held-out prose, by counts
(the most frequent continuation, so the depth is isolated from the learning
rule), the cyclic graph's own walk as the reference:

| window | train accuracy | held-out accuracy | held-out: actual gram is an option |
|---|---:|---:|---:|
| cyclic graph | 0.591 | 0.514 | 0.88 |
| tree, k = 1 | 0.591 | 0.514 | 0.88 |
| tree, k = 2 | 0.707 | 0.568 | 0.81 |
| tree, k = 3 | 0.781 | 0.579 | 0.76 |
| tree, k = 4 | 0.832 | 0.581 | 0.73 |
| tree, k = 8 | 0.942 | 0.578 | 0.69 |
| tree, unbounded | 0.978 | 0.579 | 0.68 |

A window of one gram is the graph, to the digit. Held-out accuracy plateaus
at three to four grams; deeper windows only add recitation. And depth narrows
the options: the deepest window's top guess is as good as a shallow one's,
but the actual next gram is among its options far less often — which is where
the tree's held-out miss rate in the comparison came from. `python3 -m
radixtree accuracy` measures the same thing with the learned rule on any
saved model.

## Sounds

```bash
python3 -m radixtree predict --unit phone "the quick" --to-end
# the quick brown fox jumps over the lazy dog
#   sounds DH AH0 # K W IH1 K # B R AW1 N # F AA1 K S # JH AH1 M P S # OW1 V ER0 # DH AH0 # L EY1 Z IY0 # D AO1 G
python3 -m radixtree predict --unit syllable --n 2 "the cat" --to-end --slide --window 4
```

With `--unit phone` or `--unit syllable` the corpus is read through the
phonetic tokenizer in `../PhoneticTokenizer` (found beside this checkout, or
`pip install -e ../PhoneticTokenizer`): every word becomes its sounds, the gap
between two words a `#`, punctuation a pause. The tree is then a tree over
sounds — labels, grams, lengths and scores are all in phones or syllables —
and every prediction carries its words too (`spelled`, `full_spelled`), spelled
back through the same tokenizer's lexicon and its memory of what it read. On
the sample corpus a tree over phones holds 1 767 nodes against the character
tree's 2 224, recites the same lines, and a prefix the tree has never heard
is kept as the sounds it makes.

## The pieces

| module | what it is |
|---|---|
| `radixtree/encoding.py` | the sliding window of `n` units — characters, or phones and syllables through the phonetic tokenizer — the gram overlap, the path decoder, and the spelling of sounds back into words |
| `radixtree/activation.py` | the parametric sine and its partials — the reference implementation, repeated |
| `radixtree/tree.py` | `RadixTree`: the nodes, every-suffix insertion, `split` / `merge_child` / `compress`, `walk` / `trace` / `locate`, scores and costs, `invert` / `flip_nodes`, the invariants, JSON |
| `radixtree/search.py` | `cheapest_path` and `sample_walk` over the tree; `PathResult` |
| `radixtree/model.py` | `RadixTreeNet`: the one-hop rule, `train`, `predict` / `generate` (cheapest path, sampled, or token by token), `score` / `bits_per_char`, `next_token_accuracy`, `invert_paths`, `two_nrl`, `save` / `load` |
| `radixtree/check.py` | the finite-difference checks |
| `radixtree/corpus.py` | the prose corpus from the pinned papers, the snapshot and its digest, the split |
| `radixtree/compare.py` | the comparison against `radixnet`, and the tables above |
| `radixtree/cli.py` | `python3 -m radixtree <command>` |
| `tests/test_radixtree.py` | 71 tests; `tests/README.md` names what each holds the line on |
| `DESIGN.md` | the specification, module by module, and the decisions |
| `data/`, `results/` | the texts, and the numbers with their digests |

## Running it

```bash
python3 -m radixtree train   --data data/sample_corpus.txt --epochs 10 --save model.json.gz
python3 -m radixtree predict --load model.json.gz "the quick" --to-end
python3 -m radixtree predict --load model.json.gz "the" --sample --temperature 0.8 --sample-seed 3
python3 -m radixtree predict --load model.json.gz "the" --costs signal --step-penalty -0.2
python3 -m radixtree predict --load model.json.gz "the cat" --slide --window 4 --to-end
python3 -m radixtree accuracy --load model.json.gz --texts data/corpus.txt --windows 1,2,4,all
python3 -m radixtree train --unit phone --data data/sample_corpus.txt --save sounds.json.gz
python3 -m radixtree generate --load model.json.gz --count 3
python3 -m radixtree score   --load model.json.gz "the cat sat on the mat" "the cat sat on the sky"
python3 -m radixtree stats   --load model.json.gz
python3 -m radixtree invert  --load model.json.gz --save inverted.json.gz
python3 -m radixtree 2nrl    --bad data/sample_garbage.txt --good data/sample_corpus.txt
python3 -m radixtree check
python3 -m radixtree corpus                                   # rebuild data/corpus.txt
python3 -m radixtree compare --min-count 2 --out results/x.json
python3 -m radixtree demo
```

Every command that needs a model takes `--load`; without it the model is
trained from `--data` first (the sample corpus by default), so everything runs
straight out of a checkout. `--depth N` bounds a root path, `--min-count K`
sets the trust, `--n` the gram, `--unit` what a unit is; `make help` lists the
Makefile's targets.

## Limits

- **Labels are copied, not sliced.** A leaf holds the whole remainder of its
  window, so an unbounded tree holds on the order of the square of its text in
  label characters (1.5 million for 44 thousand, above). A suffix tree proper
  keeps offsets into the text; this one keeps the cyclic graph's representation
  so that a label reads the same in both. `--depth` caps it.
- **Training cost is the empty context's fan-out.** Batch 256 is five times
  cheaper an epoch than batch 32 on the prose corpus (the root is visited once
  per batch); the tests and the comparison use 32 because it learns visibly in
  ten epochs.
- Characters, phones and syllables — no word or acoustic units; no dynamic
  window; no `BACK` / `THINK` sentinels; no server, frontend or Go and Rust
  ports. This is the acyclic core, not the cyclic product.

## Where it sits

| directory | what a node is | what it is for |
|---|---|---|
| `RadixCyclicNN/` | a gram; repeats become cycles | the model, built out |
| `RadixAcyclicNN/` (this) | a context; no cycles | the same model as a tree, and what that measures |
| `FilterBankRadix/fbradix/tree.py` | a character context, depth 5 | one expert behind an activation filter; interpolated, not a stand-alone model |
| `GREN/gren/radix.py` | a run of signature tokens | a radix tree with trie mechanics over games, not text |
| `RadixTreeLLM/`, `RadixTreeRNN/`, `RadixTrieLLM_RNN/` | a trie edge with a transformer | where the idea started |
