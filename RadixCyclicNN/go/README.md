# go

A Go port of the **count / reward model** (`CountRewardNet`) and of the
**negative network**. A standalone module — the Python implementation in
`../radixnet/` stays exactly as it is.

**Model files are interchangeable.** Both sides read and write the
`radixnet-count` and `radixnet-negative` JSON formats, including the Mersenne
Twister state, so a model trained here continues in Python and vice versa with
identical numbers. `../tests/test_go_parity.py` enforces that in both directions.
That holds in every encoding: a model built with anything but the character
trigram says so in its file, and the Python implementation reads it.

**The encoding is a dial.** Python encodes one way; this implementation makes it
a choice, fixed when a model is created and carried in its file:

```bash
go/bin/radixnet-count --model m.json --ngram 5 train --data book.txt              # 5-character sliding window
go/bin/radixnet-count --model m.json --ngram 4 --stride 4 train --data book.txt   # groups of four letters
go/bin/radixnet-count --model m.json --encoding word:2:1 train --data book.txt    # word bigrams
go/bin/radixnet-count --model m.json --encoding word:3:1 train --data book.txt    # word trigrams
```

`--units char|word` says what one unit is, `--ngram N` how many units a gram
holds, `--stride N` how far apart two grams start (1 = the sliding window, n =
non-overlapping groups); `--encoding unit[:n[:stride]]` sets all three. Lengths,
offsets and diff spans are then counted in that unit — on a word model,
`--length 3` is three more words. Over HTTP it is `POST /api/reset` with
`{"encoding": "word:2:1"}`, and `GET /api/status` reports it.

Why Go: goroutines. Training fans out over the texts of a file, weights and edge
costs recompute in parallel over the nodes, and the two beams of a prediction run
side by side — with lock-free atomic counting, so only a structural change (a
split or a merge) takes a write lock.

## Contents

| directory | what it is |
|---|---|
| `radixnet/` | the library — the graph, the model, the negative network, the tutors, the tools, the LLM clients and the MCP server (`mcp.go`) |
| `server/` | the HTTP API — the same JSON contract as the Python server, so the prebuilt React frontend runs unchanged against it |
| `cmd/radixnet-count/` | the CLI binary |
| `go.mod` | the module — `github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go`, Go 1.24, **no dependencies** |

## Building and running

From `..`:

```bash
make go-build                                   # -> go/bin/radixnet-count (needs Go 1.24+)
make go-test                                    # cd go && go test -race ./...

go/bin/radixnet-count --model model.count.json train --data data/sample_corpus.txt --epochs 5
go/bin/radixnet-count --model model.count.json predict --prefix "the cat" --k 5
go/bin/radixnet-count --kind word train --data data/sample_corpus.txt --epochs 5   # -> model.word.json
go/bin/radixnet-count --kind word words --limit 20            # the alphabet it has read
go/bin/radixnet-count --model model.count.json serve          # the API and the frontend
go/bin/radixnet-count --model model.count.json mcp            # MCP on stdin / stdout, for any client
python -m radixnet --model model.count.json info              # the Python side reads the same file
```

`go/bin/` is ignored by git; the checked-in `cmd/radixnet-count/radixnet-count`
is a prebuilt binary for convenience.

`../README.md` § *Go implementation of the count / reward model* lists every
command and every global flag, and says where the goroutines go.

## The second traversal

`predict`, `generate` and `bench` take `--traversal least-punished`: the walk is
ranked by the **blame** on its worst step before its cost, and at every node it
may only take the children the model has the least against
(`../SPEC-LeastPunished.md`).  Where nothing has been punished it is the
ordinary search, to the bit.  Python and the Rust port have it too, and
`../tests/test_go_parity.py` holds this one to Python's answers under it -
the same continuations, the same costs and the same punishment per path.  `--workers 1` also runs the two beams of a
prediction in turn rather than side by side, so a one-worker run means the same
thing here as it does in the Rust port (`../rust/`), which the cross-language
benchmark compares this one against (`../bench/`).
