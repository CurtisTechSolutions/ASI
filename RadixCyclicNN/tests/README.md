# tests

The RadixCyclicNN test suite. Plain `unittest`, no plugins — everything a test
needs it builds.

## Running them

```bash
cd ..
make test                                     # the whole suite
python3 -m unittest tests.test_graph -v       # one module
make go-test                                  # the Go side: cd go && go test -race ./...
```

`make test` includes the cross-language parity test when `go` is on `PATH`, and
skips it when it is not.

## Layout

Most files are `test_<module>.py` for `../radixnet/<module>.py`, named to match.
A few cover a behaviour rather than a module:

| file | covers |
|---|---|
| `__init__.py` | puts the project root on `sys.path` so `radixnet` imports from a checkout |
| `test_go_parity.py` | **the cross-language contract.** Trains the same corpus on both implementations and compares structure, counts, rewards, window, RNG state, predictions, generated texts, scores and conversations; blames the same failures and corrections and compares the verdicts character for character; and has each side read the other's model files. Skipped when `go` is not on `PATH`, and when `go build` of the CLI fails. |
| `test_guard.py` | what the negative network stops on the way out, on every answer path |
| `test_feedback.py` | ratings and the 2NRL phases they drive |
| `test_api_uploads.py` | uploads through the API, including ZIP archives kept as one entry and unpacked behind the scenes |
| `test_penalty.py` | the **punishment traversal** (`../radixnet/penalty.py`): the merit / penalty split per model kind, that the rewards really do leave the score, that the cheapest path is the least punished one, and the option's way through every search mode, the CLI and the HTTP API |

Six modules have no file of their own, and are exercised through the callers
that use them:

| module | tested by |
|---|---|
| `archive.py` | `test_api_uploads.py` |
| `beam.py` | `test_model.py`, `test_countnet.py`, `test_resonance.py` |
| `diff.py` | `test_countnet.py` |
| `llm.py` | `test_chatgpt.py`, `test_tutor.py` |
| `metacog.py` | `test_resonance.py` |
| `phasesearch.py` | `test_resonance.py` |

## Conventions

* **Nothing external is contacted.** The LLM clients (`ollama`, `chatgpt`,
  `llm`), the browser and the web tools are exercised against fake servers
  built on `http.server` and run on localhost for the duration of a test.
* **Seeded.** Anything random takes a seed, and the parity suite depends on the
  Mersenne Twister state being identical on both sides.
* **`torch` is optional.** `test_backend_torch.py` skips when it is absent.
