# go

A Go port of the **count / reward model** (`CountRewardNet`) and of the
**negative network**. A standalone module — the Python implementation in
`../radixnet/` stays exactly as it is.

**Model files are interchangeable.** Both sides read and write the
`radixnet-count`, `radixnet-word` and `radixnet-negative` JSON formats, including
the Mersenne Twister state, so a model trained here continues in Python and vice
versa with identical numbers. `../tests/test_go_parity.py` enforces that in both
directions.

`--kind word` is the same model over an alphabet whose symbols are **words**
rather than characters (`../SPEC-WordNGrams.md`): a word is one code point, the
window is still three, and the vocabulary is handed out in corpus order - which
is what makes a Go vocabulary the same vocabulary as a Python one.

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
