# results

The numbers in `../README.md`, committed beside the code that produced them.

| file | what it is |
|---|---|
| `sample_results.json` | both models on the 60-line sample corpus, nothing held out — `make compare-quick` |
| `prose_results.json`, `prose_run.log` | both models on the prose corpus, one fifth held out, the tree unbounded and trusting every context — `make compare` |
| `prose_min2_results.json`, `prose_min2_run.log` | the same run with `--min-count 2`: the tree consults no context it has seen only once |
| `prose_depth8_results.json`, `prose_depth8_run.log` | the same run with `--depth 8`: a root path holds at most eight symbols |
| `prose_depth2_results.json`, `prose_depth2_run.log` | the same run with `--depth 2`: a context is one gram — the cyclic graph's own order, held as a tree |
| `window_results.json` | the token-by-token walk's next-token accuracy on the held-out prose, over windows of one gram to the whole history, with the learned rule — `make window` |

Each JSON holds the corpus (size, split, SHA-256 of the exact text), the
settings, and per model: the size of everything it built, training seconds,
the loss per epoch, train and held-out bits per character with the miss rate,
how many training texts it recites whole, and the continuation of every probe
prefix. Then the repetition table (`"a" * n`, `"abc" * k`: nodes and label
characters each structure needs) and the generalisation row (shown `aaaa`,
what each says about `aaaaaaa`). `DESIGN.md` §15 says what each field means.

Reproduce with `make compare-all` (about an hour) or one at a time:

```bash
make compare-quick                  # the sample corpus, seconds
make compare                        # the prose corpus, about a quarter of an hour
make compare MIN_COUNT=2            # trusted twice (writes prose_results.json: rename it)
make compare DEPTH=8                # bounded to eight
make compare DEPTH=2                # bounded to two: the graph's order
```

`python3 -m radixtree.compare --help` lists every setting. The log files are
what the run printed: the markdown tables, and how long it took.
