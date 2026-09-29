# LatentRadixPair: design

A trained context tokenizer under a primed radix pair. The two trees of PrimedRadixPair (a count tree
written only by reading, a reward tree written only by judged outcomes, one address space) stay. What
changes is the address: a feed-forward network reads the last bytes of context and compresses them into a
short code, and the tree is primed over every prefix of that code. The network is the tokenizer; the code
is what the trees are keyed by; the units the trees count and predict are bytes, so the model reads and
writes anything.

This replaces the router layer that was planned on top of PrimedRadixPair (a market of slices, each with its
own tokenizer and trees, bidding for queries the way GTMNN's players bid). Section 7 says what became of
each part of that plan.

## 1. Why the context is what gets tokenized

PrimedRadixPair addresses a node by the last L-1 units exactly, so a vocabulary of R units costs R^L nodes.
Priming is brute force by design; the budget is a few million nodes. A learned tokenizer whose tokens are
the model's units runs into that budget from both sides:

- A learned token set that can encode *anything* losslessly needs an escape to raw bytes, so the 256 byte
  tokens are in the vocabulary whatever else is, and R^L with R >= 259 leaves L = 2: one token of context.
- A small learned vocabulary (R around 64, so L = 3 or 4 fits) cannot represent arbitrary byte strings
  exactly, and a lossy token stream decodes to text with the wrong letters in it.

The way out is to notice which of the two things a tokenizer does needs to be lossless. The *units* must be:
the model's outputs are read back as text. The *context* need not be: the context only has to select a
node, and a lossy, learned description of the context is exactly what a good node key is. So:

- The units are bytes. Every byte string encodes, every walk decodes; the outcome alphabet is the 256 bytes
  and an end mark (257 outcomes per node).
- The context is the tokenizer's code: D symbols, symbol s of radix R_s. The default is three symbols of
  radix 16 (12 bits) for 16 bytes of context.
- The tree is primed over every prefix of the code. Level 1 has R_1 nodes, level 2 has R_1 R_2, and so on;
  each node holds 257 outcome cells in both trees. The default address space is 4,369 nodes x 257 = 1.12
  million cells (27 MB for both trees).

"Compresses context down" is then literal: 16 bytes (128 bits) become 12 bits, chosen by training rather than
by keeping the last 1.5 bytes.

## 2. The tokenizer

A window of W bytes ending at a position (W = 16; positions before the text starts read as PAD, a 257th
input symbol) goes through:

1. **Embedding.** Each of the W symbols is looked up in a 257 x E table (E = 16) and the rows are
   concatenated: W E numbers.
2. **Encoder.** Two dense layers of width 256 with SiLU, then a dense layer to d numbers, one per latent
   dimension (d = 6 by default: three symbols of two dimensions).
3. **Quantiser.** Finite scalar quantisation: each latent dimension is squashed by tanh into [-1, 1] and
   rounded to one of its l levels (l = 4 by default). A symbol's dimensions form a mixed-radix integer, its
   value; the symbol's radix is the product of its level counts. Gradients pass straight through the
   rounding (the derivative of tanh is used as if the rounding were the identity).
4. **Decoder.** The quantised dimensions, plus one visibility flag per symbol, go through two dense layers
   of width 256 with SiLU and out to W vectors of width 32, each of which a shared 32 x 257 unembedding
   turns into logits over the window's symbols: the reconstruction.
5. **Next-byte head.** A dense layer from the decoder's second hidden layer to 257 logits over the byte
   that follows the window (the end of the text counts as a 257th value).

Training draws windows from the corpus (a text weighted by its length, 5% of windows from the first W
positions of a text so PAD contexts are learned) and minimises

    loss = sum_p w_p CE(window byte p) / sum_p w_p  +  predict * CE(next byte)        (predict = 4)

with three corruptions applied to each example:

- **Masking, coarse to fine.** A depth k is drawn uniformly from 1..D and the symbols after k are hidden
  from the decoder (dimensions zeroed, flag 0). Symbol 1 is always visible, symbol 2 two times in three,
  symbol 3 one time in three, so the first symbol learns to carry what reconstructs and predicts best on its
  own and each later symbol refines it. The tree's levels are exactly these prefixes, so backing off from
  level 3 to level 2 is backing off to a coarser description of the same context, not to a shorter one.
- **Byte noise.** A rate t is drawn uniformly from [0, noise] (noise = 0.1) and each non-PAD input byte is
  replaced by a random byte with probability t; the targets stay clean. Contexts that differ by a typo tend
  to share a code.
- **Recency.** The reconstruction weight of a byte is recency^age (recency = 0.6), normalised to mean 1, so
  the code spends its bits on the newest bytes, which carry most of what predicts the next one.

The optimiser is Adam (0.9, 0.99) with a linear warm-up over the first 5% of steps and a cosine decay to a
tenth of the peak rate, gradient norm clipped at 5. The default run is 2,000 steps of 256 windows.

### What came from image generation

| Image-generation idea | Here |
|---|---|
| Latent diffusion trains an autoencoder first and models only its latent | The tokenizer compresses context first; the trees model only its codes |
| Finite scalar quantisation (round bounded latents to a grid; the code is the cell) | The bottleneck: no codebook, no commitment loss, no dead codes |
| Patch embedding: a fixed window of pixels becomes one token | A fixed window of W bytes becomes one code |
| Masked-token and absorbing-state models corrupt inputs by hiding parts | Hiding the finer symbols during training orders them coarse to fine |
| A noise level drawn per example over a schedule | The byte-corruption rate is drawn per example from [0, noise] |
| A local receptive field for the autoencoder, global structure left to the generative model | The encoder sees W bytes; everything longer is the trees' business |

And one that is not from images: the next-byte head is an information-bottleneck term. A code that only
reconstructs its window is judged on the bytes it can echo back; a code that must also predict what follows
is judged on what the trees will ask it.

## 3. The address space

Codes are (c_1, ..., c_D) with 0 <= c_s < R_s. Level l holds one node per prefix of length l:

    base(0) = 0, base(l+1) = base(l) + R_1 ... R_l
    node(c_1..c_l) = base(l) + ((c_1 R_2 + c_2) R_3 + ...) 
    parent(node at level l) = base(l-1) + (node - base(l)) / R_l

N = base(D+1) context nodes, each with 257 cells: cell(node, x) = node * 257 + x. The root (level 0) is the
context of everything read. The chain of a code is its D+1 nodes from the root down; `Address.Chain` is one
multiply-add per symbol. Nothing is ever inserted or searched.

A cell ceiling (16,777,216 by default, 384 MB of trees) refuses address spaces that would not fit. Some
shapes:

| levels | radices | nodes | cells | memory |
|---|---|---|---|---|
| 4,4:4,4:4,4 (default) | 16,16,16 | 4,369 | 1.12 M | 27 MB |
| 8,4:8,4:8,4 | 32,32,32 | 33,825 | 8.69 M | 209 MB |
| 4,4:4,4:4,4:4,4 | 16,16,16,16 | 69,905 | 18.0 M | above the default ceiling |
| 16:16 | 16,16 | 273 | 70 k | 1.7 MB |

## 4. The two trees

**Count tree.** Reading a text encodes every position (the context before each byte and the context after
the last one, one batched encoder pass), then for each position adds one to the cell of the byte that
followed (or of the end mark) under every node of the code's chain, and one to each node's context count.

**Reward tree.** Judging a text does the same walk over the cells of its outcome positions, adding the
amount to the `plus` side (a reward) or the `minus` side (a punishment). The two sides are kept apart: a
punishment is never bought back. With rungs `all` every level of the chain is credited; with `final` only
the full code. A `prefix` is context, not outcome: its positions are skipped, so rewarding "mat" after
"a cat sat on the " credits m, a, t and the end mark and nothing before them. A rewarded text is also read
(counted) unless asked otherwise; a punished text never is.

**Step scores.** Under a node, an outcome's merit is share_scale * ln share, where share = (count +
smoothing) / (ctx + 257 smoothing). The `reward` traversal adds reward_scale * (plus - minus); the
`punishment` traversal uses merit_scale * merit - penalty_scale * minus. Softmax over the 257 outcomes gives
the node's own distribution q. The smoothing defaults to 0 here: a pseudo-count on each of 257 outcomes
swamps a context read a few times, and the fold below already covers what a node has not seen.

**The fold.** With own = ctx / (ctx + alpha) (alpha = 2):

    P_0 = own_0 q_root + (1 - own_0) uniform
    P_l = own_l q_l + (1 - own_l) P_{l-1}        (l = 1..D, along the chain)
    P   = (1 - floor) P_D + floor * uniform       (floor = 0.02)

Backoff `all` is the whole chain; `deepest` mixes only the full code's node with the uniform; `none` reads
the full code's node alone. This is PrimedRadixPair's fold with the parent chain in place of the
drop-oldest chain.

**Walks.** Greedy takes the fold's most probable outcome at every step (ties to the lowest byte), sampling
draws at a temperature; each step re-encodes the context (one encoder pass on one window), appends the byte,
and stops at the end mark or after the asked length (`to-end` walks until the end mark, 4,096 bytes at
most). There is no cheapest-path walk: on a primed tree the cheapest single path pays for one unread step to
fall back to the root, where frequent bytes are cheap.

**Scores.** A text is priced by the fold at each of its positions: bits per unit, the mean net reward and
the worst penalty of its steps at the full code.

## 5. Feedback

| call | what it does |
|---|---|
| `Train(texts)` | read into the count tree |
| `Reward(texts, strength, weights, read, prefix, outcomes)` | +strength * weight / outcomes per text on the outcome positions |
| `Punish(texts, strength, weights, prefix, outcomes)` | the same on the minus side; never read |
| `TwoNRL(bad, good, strength, prefix, outcomes)` | punish the bad, reward the good |
| `Feedback(texts, marks, prefix, outcomes)` | a mark in (0, 1] rewards with that weight, in [-1, 0) punishes with its size, 0 is ignored |

**Outcome units.** A verdict is worth strength / outcomes per rung, where outcomes is the number of
potential outcomes of the space the verdict comes from: the more finite the space, the more one verdict
says. A two-outcome game credits 1/2 per verdict at strength 1. English credits 1/69, one part per phone:
the 24 consonants and 15 vowels at three stress levels of the repository's phonetic tokenizer (39 phonemes
without stress, 88 emitted symbols with the word boundary and the three pauses, 26 letters and 257 bytes
are the other counts one could choose, and `outcomes` takes any of them). The model's `outcomes` setting
(69 by default) is the unit of verdicts that do not state their own; each call may state one, so one
reward tree holds verdicts from games and from English on a common scale. `outcomes` 1 restores raw
amounts. In the fold the amount lands in the log score, so from a uniform node a verdict moves its outcome
from 1/n to e^(1/n) / (e^(1/n) + n - 1): to 0.62 in a two-outcome game, and by a hundredth of a percent in
English, which is the point: a single English verdict is a small share of all the evidence a 69-way
position can produce.

## 6. Prediction and inspection

`Predict(prefix, length, mode, traversal, temperature, toEnd, backoff)` walks; `Fold(prefix)` is the next-byte
distribution; `Score(text)` prices; `Decode(prefix, known)` shows what the first `known` symbols of the
prefix's code decode to, which is how to see what a level of the tree "knows" about a context; `Coverage`
is the share of a text's chain nodes the count tree has met, depth-weighted.

## 7. What replaced the router

The router plan (PrimedPlanes) had many slices, each a PrimedRadixPair with its own tokenizer, a recogniser
choosing the slices a query fits (GREN's coverage), an auction among them (GTMNN's uniform-price seats),
settlement by outcome, and slices calling slices. The trained tokenizer takes the recogniser's job and
folds the slices into one tree:

- The **first symbol** of a code partitions every context into R_1 learned classes. It is trained to be the
  coarsest useful description of the context, so it plays the part the recogniser played: which kind of
  context is this. `latentpair tokenizer classes` lists the classes with their share of a corpus and the
  decoder's prototype for each.
- The **subtree** under a first symbol is that class's slice: its own count and reward cells, its own
  refinements. Nothing is bid for; the fold weighs a class's cells by how often they were read (own) and
  falls back to coarser classes.
- **Slices calling slices** is the fold's backoff: a fine context that has not been read defers to the
  coarser description of itself. There is no second address space to call into.
- **Adding slices by hand** is still possible the old way: prime a second model on another tokenizer or
  corpus and choose between models by their `Coverage` or `Score`. Nothing in this package does that for
  you; the intent is that one tokenizer trained on everything makes the classes itself.
- **Thinking, games, language** as named slices are not built in. Train the tokenizer on a corpus that
  contains all three and look at `classes`; the partition follows the data, not a label.

## 8. Files

Both files are JSON, gzipped when the name ends in `.gz`, written through a temporary file.

Tokenizer (`format: latent-tokenizer`): the config, seed, steps, corpus bytes, the training log, and every
weight matrix as rows, cols and little-endian float32 in base64.

Model (`format: latentpair`): the tokenizer file embedded, seed and random draws (so a reloaded model
samples on as before), settings, created, read and judged totals, the history of calls, and the non-zero
cells of both trees as parallel lists (`counts.cells/values`, `rewards.cells/plus/minus`).

## 9. Command line

`latentpair` (see `latentpair help`): `tokenizer train|info|encode|decode|classes`, `prime`, `train`,
`reward`, `punish`, `judge`, `predict`, `fold`, `score`, `info`, `demo`, `bench`. The Makefile wraps the
usual runs; `make demo` trains a small tokenizer on this repository's documents and shows the loop end to
end.

## 10. Measurements

Measured on this repository's Markdown files on four cores. Bits per byte are the fold's on the held-out
files, with alpha 2, floor 0.02, smoothing 0; the raw-byte rows use the same fold over the last k bytes, so
the only difference between rows is what the context is.

**The default tokenizer** (window 16, code 4,4:4,4:4,4, encoder width 256, predict 4, recency 0.6, 2,000
steps of 256 windows; 84 files, 2.10 MB, every fifth file held out: 16 files, 567 kB):

| context | contexts read | bits/byte | bits of context |
|---|---|---|---|
| latent code 16,16,16 | 3,564 | **3.189** | 12 |
| raw bytes, last 0 (the root alone) | 0 | 4.868 | 0 |
| raw bytes, last 1 | 162 | 3.892 | 8 |
| raw bytes, last 2 | 4,885 | 3.025 | 16 |
| raw bytes, last 3 | 36,603 | 2.408 | 24 |

Read against the raw rows, the 12 learned bits are worth about 14 raw bits. The gap to two raw bytes is
the price of the smaller tree: 4,369 primed nodes against 65,536 for a dense two-byte tree (4,885 of them
read here).

**Shape and objective** (an earlier split of 82 files with 247 kB held out, on which raw last-1 scored
3.797 and raw last-2 2.975; encoder width 128 and peak rate 2e-3 unless said):

| tokenizer | steps | bits/byte |
|---|---|---|
| reconstruction only (predict 0, recency 0.8) | 600 | 3.521 |
| predict 1, recency 0.8 | 600 | 3.515 |
| predict 4, recency 0.6 | 1,200 | 3.327 |
| predict 4, recency 0.6, binary dimensions (2,2,2,2 per symbol) | 1,200 | 3.371 |
| predict 4, recency 0.6, encoder width 256, peak rate 3e-3 (the default) | 1,200 | 3.235 |

A code trained only to reconstruct its window was worth about 11 raw bits, no better a key than the bytes
it replaced; the objective, more than the network, is what made it one.

**Training** the default tokenizer (412 k parameters) takes 220 s for 2,000 steps (0.11 s a step). The
reconstruction loss on evaluation windows fell to 2.43 bits per byte; the next-byte head reached 3.23 bits
from all three symbols and 3.89 from the first alone, which is the coarse-to-fine ordering at work.

**Throughput** with the default tokenizer: reading (encoder plus counting) 43,000 bytes/s, scoring
24,500 bytes/s, encoding alone 35,000 bytes/s, greedy prediction 11,000 bytes/s (one encoder pass per
byte). The width-128 encoder is about 2.5 times faster at 0.09 more bits per byte.

## 11. Limits and next steps

- The code is small on purpose (12 bits by default). The measured trade is in section 10: read it before
  choosing radices. A deeper or wider code buys context at the price of R_1...R_D nodes.
- Reconstruction is lossy and meant to be. `decode` shows what a level knows, not the original bytes;
  "encode anything" is a property of the byte units, not of the decoder.
- The window is fixed at W bytes. SPEC-DynamicTokenization in PrimedRadixPair describes growing windows;
  a tokenizer with several window sizes feeding one code is the natural next experiment.
- Decoding could be iterative (decode, re-encode, refine) the way masked-token models decode in a few
  passes; nothing here needs it yet.
- The trees are dense float64/int64 cells. A Rust port of the kernel measured 1.9x this Go on the fold.

## 12. Decisions

- D-1 Units are bytes; the tokenizer codes the context. Section 1.
- D-2 Finite scalar quantisation, not a codebook. No collapse, no auxiliary losses, the code is the grid.
- D-3 Masked coarse-to-fine training so the tree's parent chain is a backoff chain.
- D-4 Recency-weighted reconstruction and a next-byte head; a plain autoencoder spent its bits on the oldest
  bytes and was no better a context than raw bytes.
- D-5 Smoothing 0 by default over 257 outcomes; the fold backs off instead.
- D-6 Go, no dependencies: the numeric core is 300 lines and the matrices are small.
- D-7 Verdicts are in outcome units, strength / outcomes, English's 69 phones by default. Section 5.
