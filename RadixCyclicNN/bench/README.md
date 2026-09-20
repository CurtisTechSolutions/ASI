# bench

The Go port and the Rust port, over one corpus, on one machine.

```bash
make bench-compare                      # from ..; writes bench/RESULTS.md
python3 bench/compare.py --repeat 5 --chars 5000000
python3 bench/compare.py --no-build --keep-corpus
```

| file | what it is |
|---|---|
| `make_corpus.py` | writes `corpus.txt` and `corpus.prefixes.txt` — the texts and the prefixes both ports are handed |
| `compare.py` | builds both, runs the matrix, checks the two against each other, writes `RESULTS.md` |
| `RESULTS.md` | the last run's numbers, with the machine and the versions they were measured on |

Neither port generates the corpus. Neither language can reproduce the other's
random number generator, and a speed comparison between two programs that
trained on different text is not a comparison, so the corpus is generated once
in Python and both are pointed at the file.

## What is compared

Six builds × two traversals: Go with its default racy counting, Go with
`--exact`, and Rust — each at one worker and at the machine's own parallelism —
under the reward traversal and the least-punished one
(`../SPEC-LeastPunished.md`). Every row trains for the same epochs, punishes
every Nth text so there is blame to walk by, and predicts the same prefixes;
each is run `--repeat` times and the median is reported.

## Parity first

Before a single number is printed, the Go and Rust runs of each configuration
are checked against each other: the same nodes, edges, trigrams, transitions,
punished texts and punished edges; the same compression ratio, loss and total
penalty to within `1e-9`; the same continuation for the sample prediction, at
the same cost. A disagreement stops the run and prints what differs — the
comparison is worthless if the two builds computed different things, and a
silently diverging port is exactly what a benchmark harness is in a position to
catch.

`corpus.txt` and `corpus.prefixes.txt` are generated files and are not checked
in; `RESULTS.md` is.
