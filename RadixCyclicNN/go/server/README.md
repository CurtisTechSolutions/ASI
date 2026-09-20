# server (Go)

`package server` — the HTTP API of the Go count / reward model. **The same JSON
contract as the Python server** (`../../DESIGN.md` §12) for everything the count
model supports, so the prebuilt React frontend in `../../frontend/dist` runs
against it unchanged.

That is the point of this package: one frontend, two backends, no `#ifdef`. What
the count model does not support, the API says so about — it does not pretend.

## Contents

| file | what it is |
|---|---|
| `http.go` | the router and the handlers — every `/api/...` route, CORS, the static file serving, and the `/api/job` lifecycle |
| `service.go` | the `Service` behind the handlers: the model, its lock, the settings and the background jobs, and the encoding endpoints (`/api/encoding`, `/api/encoding/preview`) the Network settings tab reads |
| `uploads.go` | uploaded training files, including ZIP archives kept as one entry and unpacked when used. Carries the package doc |
| `checkpoints.go` | the checkpoint store — the index, rotation, the `latest` pointer and restore |
| `tutor.go` | the English lessons, the LLM client selection (Ollama or ChatGPT) and the report card |
| `negative.go` | the negative network: status, blame, clear, filter, judge, forget, settings |
| `codegen.go` | the code-generation loop — problems in, sandboxed programs out, 2NRL on the verdict |
| `agent.go` | tool use and exploration — the toolbox, the task runs, the trainer behind them |
| `chat.go` | an LLM conversing with the model, and the transcript it leaves |
| `critic.go` | the reviewer loop behind the Negative tab's automatic mode (`/api/negative/auto`), and the Ollama corpus / review endpoints |
| `evolve.go` | the self-upgrading loop (`/api/evolve/*`) |
| `media.go` | `/api/images/*` and `/api/speech/*` |

Tests sit beside the code: `server_test.go` plus one `*_test.go` per area, and
`guard_test.go` for what the negative network stops on the way out.

## Running it

From `../..`:

```bash
make go-serve                                  # builds, then serves on HOST:PORT
go/bin/radixnet-count --model model.count.json serve
```

Then open the address it prints — the same page the Python server serves.

```bash
cd ..                 # the go/ directory
go test ./server/...
```
