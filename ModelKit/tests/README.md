# tests

The kit's test suite: the command line, the HTTP API, the teaching loops, the
LLM clients, the agent and its tools, speech, images and voice, MCP, and the
cross-language parity of the Go and Rust ports. Plain `unittest`, no plugins.

## The model they run against

The kit is tested against a real model: the RadixCyclicNN checkout beside this
one (`../../RadixCyclicNN`). Every file puts both on `sys.path` — this project
for `modelkit`, the model's for `radixnet` — and reads the model's sample data
from its `data/` directory. A test that runs the command line runs
`python -m radixnet` in the model's directory, the way its README does; that
entry point hands over to `modelkit.cli`. The Go and Rust suites build the
kit's binaries from `../go` and `../rust`.

## Running them

```bash
cd ..
make test                                     # the whole suite (parity suites skip without go / cargo)
python3 -m unittest tests.test_tutor -v       # one module
make go-parity                                # Go against Python
make rust-parity                              # Rust against Python
```

## Layout

Most files are `test_<module>.py` for `../modelkit/<module>.py`, named to
match. Where a model feature has a command-line or HTTP side, the kit's file
carries the model's name and tests that side — `test_attention.py`,
`test_window.py`, `test_prune.py`, `test_penalty.py`, `test_countnet.py`, `test_negative.py`,
`test_resonance.py`, `test_schedule.py`, `test_search_training.py` and
`test_acoustic_units.py` here are the command line and the API of what the
model's files of the same name test in the model.

| file | covers |
|---|---|
| `test_agent_config.py` | chat descriptor validation, request defaults and overrides, timeout and HTTP failures, multi-turn CLI conversations, and an end-to-end request to the real RadixCyclicNN server |
| `test_layers.py` | **the line between the layers**: the model-free modules load with the model package refused at import; each model-layer module really does need it; `import modelkit` loads no model; and nothing in `radixnet` imports the kit except its `python -m radixnet` entry point |
| `test_go_parity.py` | **the cross-language contract.** Trains the same corpus on both implementations and compares structure, counts, rewards, window, RNG state, predictions, generated texts, scores and conversations; blames the same failures and corrections and compares the verdicts character for character; and has each side read the other's model files. Skipped when `go` is not on `PATH`, and when `go build` of the CLI fails; **talking by voice** (D-089): `speech talk` hears, learns, answers and speaks the same on both sides, and the Go server's `/api/voice` routes answer the Voice tab |
| `test_rust_parity*.py`, `rust_harness.py` | the same contract against the Rust port, one file per area; the harness builds the release binary once and skips without `cargo`. `test_rust_parity_methods.py` holds eight training plans to Python's graph, history and `replay` block byte for byte; `test_rust_parity_media.py` covers images and speech as text, the recall tutor, the media routes and talking by voice |
| `test_guard.py` | what the negative network stops on the way out, on every answer path |
| `test_feedback.py` | ratings and the 2NRL phases they drive |
| `test_api_uploads.py` | uploads through the API, including ZIP archives kept as one entry and unpacked behind the scenes |
| `test_voice.py` | the **voice** (D-081): a walk reports every step and its END sentinel, and is heard as it goes; words and letters are read through the tokenizer; the `speak` command writes a WAV; the **output decoder** (D-088): `say` speaks any text in the model's units, the streaming form makes the same bytes, a walk heard as it walks and its text said afterwards are the same audio; `--speak FILE` on `predict` and `generate`, and `speech decode` |
| `test_voicechat.py` | **talking with the model by voice** (D-089): one turn on the module alone, then over the API whole and streamed, the refusals, and `speech talk` on the command line with a conversation file |
| `test_teaching_in_words.py` | the teaching loops of a model of sounds show the LLM words (`reader_text`, now `../modelkit/llm.py`) |

Three modules have no file of their own, and are exercised through the callers
that use them:

| module | tested by |
|---|---|
| `archive.py` | `test_api_uploads.py` |
| `llm.py` | `test_chatgpt.py`, `test_tutor.py`, `test_teaching_in_words.py` |
| `media.py` | `test_speech.py`, `test_vision.py` |

## Conventions

* **Nothing external is contacted.** The LLM clients (`ollama`, `chatgpt`,
  `llm`), the browser and the web tools are exercised against fake servers
  built on `http.server` and run on localhost for the duration of a test.
* **Seeded.** Anything random takes a seed, and the parity suites depend on the
  Mersenne Twister state being identical in every port.
