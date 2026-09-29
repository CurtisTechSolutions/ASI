# LatentRadixPair

A trained tokenizer for the context, under the two primed radix trees of
[PrimedRadixPair](../PrimedRadixPair). A feed-forward network reads the last 16 bytes and compresses them
into a 12-bit code; the count tree and the reward tree are primed over every prefix of that code; the units
they count, judge and predict are bytes, so the model reads and writes anything. This replaces the planned
router of slices: the code's first symbol is a learned partition of contexts, and the tree's levels below it
are the slices.

[DESIGN.md](DESIGN.md) has the reasoning, the network, the address space, the fold, and what came from
image generation. Pure Go, no dependencies.

## Quick start

```sh
cd LatentRadixPair
make build test                       # bin/latentpair; the tests take about 20 s
make demo                             # trains a small tokenizer on the repository's documents and shows the loop
```

The demo prints the training log, what a context's code decodes to symbol by symbol, the fold after it,
a greedy continuation, and what a reward and a punishment do to the fold.

With your own files (one file is one text):

```sh
bin/latentpair tokenizer train --out tok.json.gz --steps 2000 corpus/*.txt
bin/latentpair tokenizer decode --tokenizer tok.json.gz --text "the cat sat on the "
bin/latentpair prime --tokenizer tok.json.gz --out model.json.gz
bin/latentpair train --model model.json.gz corpus/*.txt
bin/latentpair predict --model model.json.gz --prefix "the cat sat on the " --length 20
bin/latentpair fold --model model.json.gz --prefix "the cat sat on the "
bin/latentpair reward --model model.json.gz --prefix "the cat sat on the " --text "mat" --strength 5
bin/latentpair punish --model model.json.gz --prefix "the cat sat on the " --text "log"
bin/latentpair reward --model model.json.gz --prefix "1. e4 e5 2. " --text "Nf3" --outcomes 2   # a game verdict
bin/latentpair predict --model model.json.gz --prefix "the cat sat on the " --traversal punishment
bin/latentpair score --model model.json.gz --text "the cat sat on the mat"
bin/latentpair tokenizer classes --tokenizer tok.json.gz corpus/*.txt   # what the first symbol stands for
bin/latentpair bench --tokenizer tok.json.gz corpus/*.txt              # held-out bits against raw byte contexts
```

`--text -` reads standard input. A verdict is worth `strength / outcomes` per rung, `outcomes` being the
number of potential outcomes of the space it comes from: 69 by default, one per English phone; a
two-outcome game says `--outcomes 2`. Settings (`--alpha`, `--floor`, `--smoothing`, `--outcomes`,
`--rungs all|final`, `--backoff all|deepest|none`, ...) are flags of `prime` and `bench`; tokenizer shape (`--window`,
`--levels 4,4:4,4:4,4`, `--recency`, `--predict`, `--noise`, ...) flags of `tokenizer train` and `demo`.

## What the numbers say

Held out: every fifth of this repository's 84 Markdown files (2.10 MB; 16 files, 567 kB held out), the rest
read; bits per byte of the fold with alpha 2, floor 0.02, smoothing 0. The raw rows use the same fold over
the last k bytes.

| context | contexts read | bits/byte | bits of context |
|---|---|---|---|
| latent code 16,16,16 (the default tokenizer, 2,000 steps) | 3,564 | **3.189** | 12 |
| raw bytes, last 1 | 162 | 3.892 | 8 |
| raw bytes, last 2 | 4,885 | 3.025 | 16 |
| raw bytes, last 3 | 36,603 | 2.408 | 24 |

Twelve learned bits of context are worth about fourteen raw bits; the same code trained only to
reconstruct its window was worth eleven. Training the tokenizer takes 0.11 s a step on four cores (220 s
for the default 2,000 steps); the model then reads 43 kB/s and predicts 11 kB/s in Go, 96 kB/s and 19 kB/s
in Rust. DESIGN.md section 10 has the full tables.

## Rust port and frontend

[`rust/`](rust/) is the same model in Rust, file-compatible with the Go implementation and checked
against it by parity tests, with a web frontend (`latentpair serve`) that predicts, folds, judges in
outcome units, reads, scores and trains tokenizers from the browser. `cd rust && make build test serve`.

## Layout

| path | what |
|---|---|
| `nn/` | float32 matrices with parallel products, dense layers, SiLU, softmax cross-entropy, Adam; a finite-difference gradient test |
| `tokenizer/` | the network, its training (masking, noise, recency, next-byte head), encode/decode, files |
| `pair/` | the mixed-radix address space, count and reward trees, the fold, walks, feedback, the model file |
| `cmd/latentpair/` | the command line, the demo and the benchmark |
| `rust/` | the Rust port, its parity tests and the web frontend |
| `DESIGN.md` | the design |

Files are JSON (`.gz` gzipped); a model file embeds its tokenizer, so one file is one model.
