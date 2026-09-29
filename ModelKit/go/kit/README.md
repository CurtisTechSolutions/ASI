# kit (Go)

`package kit` — everything built on the Go count / reward model rather than the
model itself: the tutors and the critic, the conversation and the thinking, the
tools and the agent, the LLM clients, speech, images and voice, today's format
and the MCP server. The Go twin of Python's `modelkit`, held to it by
`../../tests/test_go_parity.py`.

The model is `github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet`
(`../../../RadixCyclicNN/go/radixnet`). What used to be a method of the model
and belongs to the kit is a function here that takes the model first:
`kit.Converse(m, opening, opts)`, `kit.Reply`, `kit.Backtrack`, `kit.TeachBack`,
`kit.ThinkBack`, `kit.Think`, `kit.ThinkOn`, `kit.Place` and `kit.SpeakWalks`.

## The negative side

| file | what it is |
|---|---|
| `blame.go` | the tutor's verdicts turned into blame |
| `duo.go` | the positive model writes, the negative one vetoes |
| `gan.go` | the self-upgrading loop |
| `critic.go` | the reviewer on a loop — the negative network feeding itself |
| `review.go` | Ollama beyond the chat client: a training corpus written from a prompt (good or garbage), and adversarial review of the model's own samples |

## Teaching it

| file | what it is |
|---|---|
| `tutor.go` | the English lessons — set, answer, mark, and the marker's thinking taught as thoughts |
| `recall.go` | what it remembers of what it was shown |
| `dialogue.go`, `chat.go` | conversing with itself, and with an LLM that marks the replies |
| `thinking.go` | the Think sentinel at work: what makes the model think, what it thinks, and what happens when it stops |
| `plan.go` | the lesson plan — the report card at the end of a run, the grammar focus each mistake drills, and the level it moves the student to |
| `codegen.go` | code generation with a sandbox and a judge |
| `agent.go`, `tools.go`, `toolbox.go` | tool use: the loop, the text format, and the built-in tools (browsing, the calculator, the sandbox, the uploads) |
| `calc.go` | the calculator tool's expression language. Python evaluates these with `ast` and a whitelist; Go has no `eval`, so this is a parser written to match Python's arithmetic where the two differ |
| `web.go` | browsing on the standard library — HTML reduced to readable text, and its links |
| `llm.go`, `ollama.go`, `chatgpt.go` | the provider plumbing and the two clients |

## Talking to it

| file | what it is |
|---|---|
| `assistant.go` | today's format: messages in, an assistant message out — thinking, text, tool calls, streamed (OpenAI's and Anthropic's dialects) |
| `mcp.go` | the Model Context Protocol server: the tools and the network, for any MCP client |
| `speech.go`, `vision.go` | speech and images as text |
| `blake2b.go` | the BLAKE2b behind a spoken text's utterance token, ported so both languages agree |
| `voice.go`, `voicechat.go` | the model heard as it walks, and talking with it by voice |
| `spelled.go` | the turns and the thoughts of a model of sounds as they are written out, with the words their sounds spell |

## Running it

| file | what it is |
|---|---|
| `bench.go` | throughput benchmarks |
| `memlimit.go` | the soft heap limit (`--memlimit`), 80% of the machine or container by default |
| `helpers.go` | two small helpers the model keeps to itself, copied rather than exported |

## Tests

26 `*_test.go` files beside the code they cover.

```bash
cd ..                 # the go/ directory of ModelKit
go test ./kit/...
go test -race ./...   # or, from ../..: make go-test
```
