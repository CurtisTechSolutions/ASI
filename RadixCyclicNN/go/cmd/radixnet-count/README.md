# radixnet-count

`package main` — the CLI of the Go count / reward model. Train, predict,
generate, score, feedback / 2NRL, invert, weights, info and converse, with model
files interchangeable with the Python implementation.

## Contents

| file | what it holds |
|---|---|
| `main.go` | the package doc, the global flags (`--model`, `--json`, `--seed`, `--workers`, `--exact`, `--out`, `--memlimit`, `--memprofile`, and `--encoding` / `--units` / `--ngram` / `--stride` for a **new** model's encoding), the dispatcher, and the core commands — `train`, `predict`, `generate`, `score`, `feedback`, `2nrl`, `correct`, `invert`, `weights`, `nodes`, `paths`, `info`, `converse`, `tutor`, `serve`, `version` |
| `negative.go` | `negative` — `blame`, `clear`, `why`, `filter`, `reasons`, `forget`, `auto` |
| `ollama.go` | `ollama` (`models`, `corpus`, `review`) and `chatgpt` (`models`, `ask`) |
| `chat.go` | `chat` — an LLM converses with the model and marks every reply |
| `codegen.go` | `codegen` — write programs, run them in the sandbox, learn from the verdict |
| `agent.go` | `agent` and `explore` — the network browses, an LLM judges, 2NRL follows — plus `tools` (`list`, `describe`, `call`) |
| `evolve.go` | `evolve` — the self-upgrading loop |
| `media.go` | `image` (`info`, `encode`, `tutor`, `decode`) and `speech` (`info`, `teach`, `tutor`, `decode`) |
| `extras.go` | `bench`, `compress` and `checkpoints` |
| `radixnet-count` | a prebuilt binary, committed so the CLI runs without a Go toolchain |

## Building it

```bash
cd ../..                                # the go/ directory
go build -o bin/radixnet-count ./cmd/radixnet-count
```

Or, from `../../..`: `make go-build`. Needs Go 1.24+ and nothing else.

## Using it

```bash
radixnet-count --model model.count.json train --data data/sample_corpus.txt --epochs 5
radixnet-count --model model.count.json predict --prefix "the cat" --k 5
radixnet-count --model model.count.json negative filter --count 3   # write, then veto
radixnet-count --model model.count.json tutor --topic animals --rounds 3 --blame
radixnet-count --seed 1 bench --chars 200000
radixnet-count --model model.count.json predict --prefix "the cat" --traversal least-punished
radixnet-count --model model.word.json --encoding word:3:1 train --data data/sample_corpus.txt --epochs 5
radixnet-count --model model.word.json words --limit 20   # the alphabet it has read
```

`--encoding word:3:1` builds the same model over an alphabet whose symbols are
**words** (`../../../SPEC-WordNGrams.md`): every length, count and score is then
per word, `words` lists the alphabet the graph's grams are made of, and a node
is addressed in words (`nodes --node "sat on the mat"`). `--units`, `--ngram`
and `--stride` set the three dials one at a time. The encoding is fixed when a
model is created, so those flags apply to a new model and a loaded file's own
encoding always wins; `--kind word` only picks the default file name.

`--traversal least-punished` on `predict`, `generate` and `bench` walks by the
blame on a step rather than by its cost (`../../../SPEC-LeastPunished.md`); the
`bench` command also takes `--texts` / `--prefixes` to be handed a corpus rather
than build one, and `--punish-every N` to punish every Nth text before the
predictions are timed, which is how `../../../bench/compare.py` runs it against
the Rust port.

`../../../README.md` § *Go implementation of the count / reward model* has the
full command list, the flags and the Ctrl-C behaviour of the long-running loops.
