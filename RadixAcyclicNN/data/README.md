# data

| file | what it is | used by |
|---|---|---|
| `sample_corpus.txt` | correct sentences, one per line — a copy of `RadixCyclicNN/data/sample_corpus.txt`, so the two models are shown the same text | `make demo`, `make compare-quick`, the tests, every command's default `--data` |
| `sample_garbage.txt` | the same sentences broken — a copy of `RadixCyclicNN/data/sample_garbage.txt` | the negative phase of `make 2nrl`, the tests |
| `corpus.txt` | 800 lines of prose cut out of the four research papers in `Research/` — the corpus the prose comparison in `../README.md` is measured on | `make compare`, `make compare-all` |

The two sample files are copies rather than links so that this directory
stands alone, the way every model directory in this repository does.

`corpus.txt` is a **snapshot** rather than a build step so that a result can be
reproduced after the papers it came from have changed. The sources are pinned
by name in `radixtree.corpus.SOURCES`, never globbed; `radixtree.corpus.digest`
is the SHA-256 of the exact text, and every result file records it.

```bash
make corpus                            # rebuild it from the pinned papers
python3 -m radixtree corpus --lines 2000 --out data/bigger.txt
```

The comparison holds out every fifth line (`--heldout-every 5`): 640 lines to
train on, 160 to score.
