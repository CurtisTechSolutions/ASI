# data/corpus

What `latticefsm ngram train` and `autotrain` read when no `--data` is given:
every text file in this directory, in name order (this README excepted).
Drop more files in and they are trained on next time.

| file | what it is |
|---|---|
| `prose.txt` | 800 lines of prose cut out of the four research papers in `../../../Research/` - a copy of `RadixAcyclicNN/data/corpus.txt` |
| `sentences.txt` | sixty short correct sentences, one per line - a copy of `RadixCyclicNN/data/sample_corpus.txt` |

Formats read (`ngram.corpus_lines`): `.txt` / `.md` / `.text` line by line;
`.jsonl` one JSON value a line, its `text` (or `content`, `sentence`, `body`);
`.json` likewise; any of them gzipped (`.gz`). Every line is one sequence.
Every tenth line is held out to measure the model on unless `--held-out 0`.

The files are copies rather than links so that this directory stands alone,
the way every model directory in this repository does. `autotrain --ollama`
writes what the language model wrote to `ollama.txt` beside them (with
`--keep`), so a corpus grown by an LLM can be trained on again without it.
