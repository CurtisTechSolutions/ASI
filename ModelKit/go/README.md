# go

The Go port of the kit: everything built on the Go count / reward model
(`../../RadixCyclicNN/go`) — the library, the HTTP server and the
`radixnet-count` CLI. A module of its own,
`github.com/CurtisTechSolutions/ASI/ModelKit/go`, which requires the model's
module and the phonetic tokenizer's through `replace` directives to the
checkouts beside it. Still no third-party dependency.

## Contents

| directory | what it is |
|---|---|
| `kit/` | the library — the tutors, the critic, the conversation and the thinking, the tools and the agent, the LLM clients, speech, images and voice, the MCP server and today's format (`kit/README.md`) |
| `server/` | the HTTP API — the same JSON contract as the Python server, so the prebuilt React frontend (`../frontend/dist`) runs unchanged against it |
| `cmd/radixnet-count/` | the CLI binary, and a prebuilt copy of it |
| `go.mod` | the module, Go 1.24 |

## Building and testing

From `..`:

```bash
make go-build                                  # -> go/bin/radixnet-count (needs Go 1.24+)
make go-test                                   # go vet and go test -race ./... here
make go-parity                                 # the Go port held to the Python model's output
go/bin/radixnet-count --model model.count.json serve   # the API and this kit's frontend
```

The model's `Makefile` (`../../RadixCyclicNN`) keeps every `go-*` target it
had (`make go-serve`, `make go-talk`, `make go-mcp`, ...) and builds this
module for them.
`../../RadixCyclicNN/go/README.md` and the model's README
(§ *Go implementation of the count / reward model*) are the manual.

`serve` looks for the frontend at `frontend/dist` under the working
directory: run it from `..`, or pass `--frontend-dir` (the make targets do).
