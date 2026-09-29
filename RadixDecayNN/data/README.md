# data

| file | what it is | used by |
|---|---|---|
| `sample_corpus.txt` | sixty correct sentences, one per line — a copy of `RadixCyclicNN/data/sample_corpus.txt`, so every model directory is shown the same text | `make demo`, `make experiments`, the tests, every command's default `--data` |
| `sample_garbage.txt` | the same sentences broken — a copy of `RadixCyclicNN/data/sample_garbage.txt` | kept beside the corpus for the same reason the siblings keep it; nothing here trains on it |
| `corpus.txt` | 800 lines of prose cut out of the four research papers in `Research/` — a copy of `RadixAcyclicNN/data/corpus.txt`, whose `radixtree.corpus.digest` pins the exact text | `make experiments-prose` |

The files are copies rather than links so that this directory stands alone,
the way every model directory in this repository does. The prose corpus is a
snapshot for the same reason the sibling's is: a result can be reproduced
after the papers it came from have changed.
