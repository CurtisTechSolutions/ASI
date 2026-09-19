# go

A Go port of the **count / reward model** (`CountRewardNet`) and of the
**negative network**. A standalone module — the Python implementation in
`../radixnet/` stays exactly as it is.

**Model files are interchangeable.** Both sides read and write the
`radixnet-count` and `radixnet-negative` JSON formats, including the Mersenne
Twister state, so a model trained here continues in Python and vice versa with
identical numbers. `../tests/test_go_parity.py` enforces that in both directions.

Why Go: goroutines. Training fans out over the texts of a file, weights and edge
costs recompute in parallel over the nodes, and the two beams of a prediction run
side by side — with lock-free atomic counting, so only a structural change (a
split or a merge) takes a write lock.

## Contents

| directory | what it is |
|---|---|
| `radixnet/` | the library — the graph, the model, the negative network, the tutors, the tools and the LLM clients |
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
go/bin/radixnet-count --model model.count.json serve          # the API and the frontend
python -m radixnet --model model.count.json info              # the Python side reads the same file
```

`go/bin/` is ignored by git; the checked-in `cmd/radixnet-count/radixnet-count`
is a prebuilt binary for convenience.

`../README.md` § *Go implementation of the count / reward model* lists every
command and every global flag, and says where the goroutines go.
