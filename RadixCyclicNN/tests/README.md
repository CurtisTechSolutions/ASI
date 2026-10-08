# tests

The core model's test suite: the graph, the four kinds, the encodings,
training, the searches and the model file. Plain `unittest`, no plugins —
everything a test needs it builds. Everything built around the model — the
command line, the HTTP API, the teaching loops, the LLM clients, the agent,
speech, images and voice, MCP, and the Go and Rust parity suites — is tested
with the kit, in `../../ModelKit/tests`.

## Running them

```bash
cd ..
make test-core                                # this suite (make test runs it, then the kit's)
python3 -m unittest tests.test_graph -v       # one module
make go-test                                  # the Go side: the core module, then the kit's
```

Nothing here imports `modelkit`: the suite passes with the kit refused at
import time, which is how the split between the two packages was checked, and
`../../ModelKit/tests/test_layers.py` keeps it that way.

## Layout

Most files are `test_<module>.py` for `../radixnet/<module>.py`, named to match.
A few cover a behaviour rather than a module:

| file | covers |
|---|---|
| `__init__.py` | puts the project root on `sys.path` so `radixnet` imports from a checkout |
| `test_attention.py` | the **attention band** (`../radixnet/attention.py`, `../SPEC-AttentionBand.md`): the band's shape; the rule's three properties (one charge per changed unit, a whole text left alone, a unit one gram sees charged to it in full) over eight encodings; the charged steps; both kinds' corrections under the band, and off to the bit; the file block; the refusals (the CLI and the HTTP API: the kit's `test_attention.py`) |
| `test_window.py` | the **dynamic window** (`../radixnet/window.py`, `../SPEC-DynamicWindow.md`): the ladder and what it refuses; `ABCD` into `AB` and `CD` under a grouping encoding, into `ABC` and `BCD` under the trigram; a node halved at its middle gram until it fits, never below one gram; the halves' same data and the heavy connection in every kind's currency; compression stopping at the window and the top regrowing what stayed unary; the automatic step on every kind; the file block; off to the bit (the CLI and the HTTP API: the kit's `test_window.py`) |
| `test_prune.py` | **auto prune** (`../radixnet/prune.py`, `../SPEC-AutoPrune.md`): the setting and what it refuses; a text registered but never walked pruned away while the trained texts still walk; a busy node thinned of its rare continuation; what was taught - a hand-over, a reward, a judged context, blame - never pruned and the node holding it never swept; the sweep until nothing is stranded; the file block and off to the bit; the automatic prune at the end of an epoch on every kind, every `every`-th epoch, never by hand; a prune by hand with the setting off (the CLI and the HTTP API: the kit's `test_prune.py`) |
| `test_penalty.py` | the **punishment traversal** (`../radixnet/penalty.py`): the merit / penalty split per model kind, that the rewards really do leave the score, that the cheapest path is the least punished one, and the option's way through every search mode (the CLI and the HTTP API: the kit's `test_penalty.py`) |
| `test_phonetic.py` | the **phonetic units** (`../../PhoneticTokenizer`, D-080): the text read as sounds, a label cut into the units it was made of, a prefix matched by its sounds, a prediction spelled back into words; skipped when the tokenizer is not importable |
| `test_acoustic_units.py` | the **acoustic unit** (D-082): a recording is heard as a text of learned units, and a model is trained on such texts alone, saved and loaded (spoken back through the vocoder, `say` and the command line: the kit's `test_acoustic_units.py`); skipped when the tokenizer is not importable |
| `test_search_training.py` | the **search and training methods** (`../SPEC-SearchAndTraining.md`): the sampling filters, the diverse beam, the keys (pinned - Go and Rust assert the same numbers), the curriculum, the replay buffer and early stopping on every kind that learns by walking texts, the rule that a feedback pass never touches the buffer; reading backwards (`reverse`) on every kind and both units - a reversed run is a run over the reversed texts (the settings through the HTTP API and the CLI: the kit's `test_search_training.py`) |

Three core modules have no file of their own, and are exercised through the
callers that use them:

| module | tested by |
|---|---|
| `beam.py` | `test_model.py`, `test_countnet.py`, `test_resonance.py` |
| `diff.py` | `test_countnet.py`, `test_attention.py` |
| `metacog.py` | `test_resonance.py` |

`phasesearch.py` is covered by `test_resonance.py` too.

## Conventions

* **Nothing external is contacted.**
* **Seeded.** Anything random takes a seed, and the parity suites (in the kit)
  depend on the Mersenne Twister state being identical in every port.
* **`torch` is optional.** `test_backend_torch.py` skips when it is absent.
