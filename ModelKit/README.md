# ModelKit

Everything around a model: how it is taught, talked to, served and seen. The
teaching loops, the LLM clients, the agent and its tools, speech, images and
voice, the Model Context Protocol server, the command line, the HTTP API and
the React frontend — broken out of [RadixCyclicNN](../RadixCyclicNN) so other
directories and repositories can use them. RadixCyclicNN keeps the model
itself: the graph, its four kinds, the encodings, training, the searches and
the model file.

Like [PhoneticTokenizer](../PhoneticTokenizer) beside it, the kit comes in the
same ports as the model, standard library only in each:

| port | where | what |
|---|---|---|
| Python | `modelkit/` | the reference: every module below (`pip install -e .`, or the checkout on `PYTHONPATH`) |
| Go | `go/` | module `github.com/CurtisTechSolutions/ASI/ModelKit/go`: package `kit`, the HTTP `server`, and the `radixnet-count` command (`cmd/radixnet-count`) |
| Rust | `rust/` | crate `modelkit`, and the `radixnet` and `radixnet-bench` binaries |
| JavaScript | `frontend/` | the React single-page app every port's server serves |

## The one rule

The dependency runs one way: the kit imports the model, and nothing in the
model imports the kit. `radixnet` (Python), `go/radixnet` (Go) and the
`radixnet` crate (Rust) build, test and run on their own; the kit's model
layer is built on them. The one sanctioned exception is `python -m radixnet`,
the model's entry point, which hands over to this kit's command line so every
command in the model's README keeps working. `tests/test_layers.py` holds the
Python side to it, and the Go module and the Rust crate cannot break it: the
model's module and crate do not know this one exists.

## Two layers

**Model-free.** These load and work with no model installed at all, and are
the part to reach for from anywhere:

| module | what |
|---|---|
| `llm`, `ollama`, `chatgpt` | one LLM client shape for a local Ollama and OpenAI's hosted models: list models, complete, chat; `reader_text` shows a reader the words a model of sounds spells |
| `tools`, `browser` | the agent's tools (calculator, web search and fetch, a Python sandbox, uploads) and the Chrome-over-WebDriver browser behind them |
| `mcp` | a Model Context Protocol server over those tools (stdio) |
| `archive` | ZIP uploads unpacked in memory, entry by entry |
| `speech`, `vision`, `media` | audio and images as text a network can learn and predict, and back |
| `critic`, `codegen`, `recall`, `chat`, `blame` | teaching loops that are *handed* a model rather than importing one: an LLM critic, code generation judged by a sandbox, the recall tutor, the LLM chat partner, and the tutor's verdicts turned into blame |

```python
from modelkit.llm import make_client
from modelkit.tools import default_toolbox

client = make_client("ollama")                         # or "chatgpt" with OPENAI_API_KEY
print(client.generate("One sentence about radix trees."))
print(default_toolbox().call("calculator", {"expression": "2*(3+4)"}).output)   # 14
```

**Built on the model.** The tutor, the agent, evolve (the GAN-style
self-upgrade), the conversation and thinking walks, the assistant formats
(OpenAI's and Anthropic's), the negative network's filter, voice and voice
chat, checkpoints, the benchmark, the HTTP API and the command line. They take
their model from the `radixnet` package — installed, or found in the
RadixCyclicNN checkout beside this one when it is not — and they are how that
model is used day to day:

```bash
cd ../RadixCyclicNN
python -m radixnet train --data data/sample_corpus.txt --epochs 5      # the command line is this kit's
python -m radixnet serve --port 8000                                   # the API + the frontend on :8000
python -m modelkit --help                                              # the same command line, by its own name
```

`import modelkit` loads nothing but its `__init__`: `modelkit.CheckpointManager`,
`modelkit.converse`, `modelkit.NegativeFilter` and the other public names are
imported from their modules on first use, so the model-free layer stays free.

## Using it from another directory or repository

* **Python** — `pip install -e path/to/ModelKit` (and `path/to/RadixCyclicNN`
  for the model layer), or put the two checkouts on `PYTHONPATH`. The media
  extras are `modelkit[images]`, `[diffusion]`, `[speech]` and `[whisper]`.
* **Go** — require the module and point it at the checkout, the way
  RadixCyclicNN's own module requires the phonetic tokenizer:

  ```
  require github.com/CurtisTechSolutions/ASI/ModelKit/go v0.0.0
  replace github.com/CurtisTechSolutions/ASI/ModelKit/go => ../ModelKit/go
  replace github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go => ../RadixCyclicNN/go
  replace github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go => ../PhoneticTokenizer/go
  ```

  (`replace` directives only apply in the main module, so a consumer repeats
  the kit's own.)
* **Rust** — `modelkit = { path = "../ModelKit/rust" }`.
* **The frontend** — `frontend/` is a static app over the HTTP API
  (`modelkit/api.py`; the Go and Rust servers speak the same JSON). Any server
  that answers those routes can serve its `dist/`, and `npm run dev` proxies
  `/api` to `VITE_PROXY_TARGET` (default `http://127.0.0.1:8000`). A route a
  server does not implement answers 404, which the tab shows as an error (a
  few tabs fall back to an older route instead).

## Commands

`make help` lists them. The ones that matter:

```bash
make test               # the Python suite (the parity suites skip without go / cargo)
make go-build go-test   # the Go module: go/bin/radixnet-count, and its tests
make rust-build rust-test
make parity             # Go and Rust held to Python's output, byte for byte
make frontend-build     # rebuild frontend/dist, which every server serves
```

RadixCyclicNN's own `Makefile` keeps every command it had (`make serve`,
`make tutor`, `make go-serve`, ...); they run the kit from here.

## Documentation

The design documents stay with the model, because they describe one system:
`../RadixCyclicNN/README.md` (the manual), `DESIGN.md` (the specification),
`DECISIONS.md` (why; D-093 is this split) and the `SPEC-*.md` files.
`tests/README.md` says what each test covers.
