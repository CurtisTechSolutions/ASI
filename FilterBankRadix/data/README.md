# data

`corpus.jsonl` — the committed snapshot of the four-source corpus, one
`{"source": ..., "text": ...}` per line. Python, Go, Markdown prose and JSON
results, cut out of this repository's own files by `fbradix/corpus.py` and
balanced at 200 segments per source.

It is a snapshot rather than a build step so that a result can be reproduced
after the files it came from have changed. `fbradix.corpus.digest` is the
SHA-256 of the exact text, and every result file records it.

```bash
make corpus                       # rebuild it from the pinned files
make corpus PER_SOURCE=800        # a bigger one
```

The **sources are pinned by name** in `corpus.SOURCES`, never globbed: a corpus
that changes when a file is added elsewhere in the repository is not a corpus
you can compare two runs on.

The four labels (`python`, `go`, `prose`, `json`) are what the routing is
*scored against*. Only the `oracle` arm is ever shown them.
