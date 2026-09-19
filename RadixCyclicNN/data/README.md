# data

Small sample inputs, committed so every command in `../README.md` runs straight
out of a checkout with nothing to download. They are deliberately tiny — enough
to see a command work, not enough to train anything.

## Contents

| file | what it is | used by |
|---|---|---|
| `sample_corpus.txt` | correct sentences, one per line — "the quick brown fox jumps over the lazy dog" | `train`, `evolve`, the Train panel |
| `sample_garbage.txt` | the same sentences **broken** — "jumps *under* the lazy dog", "the cat sat on the *sky*" | the negative phase of `2nrl` / `feedback`, and the negative network |
| `sample_problems.jsonl` | programming problems as JSON — `id`, `prompt`, `expected_output`, optional `tests` | `codegen --problems` |
| `sample_problems.txt` | the same problems as plain text, one per line, `#` for comments | `codegen --problems` |
| `sample_tasks.jsonl` | questions for the agent as JSON — `id`, `prompt`, optional `answer`, `criteria`, `seeds` | `agent --tasks` |
| `sample_tasks.txt` | the same questions as plain text, one per line, `#` for comments | `agent --tasks` |

The `.txt` and `.jsonl` forms of each pair are interchangeable — the JSON one
carries the expected answer and the acceptance criteria, the text one leaves
those to the teacher to write.

## Why the pair of corpora

`sample_corpus.txt` and `sample_garbage.txt` are the same sentences with the
nouns and prepositions swapped, which is the whole of 2NRL in miniature: train
on the garbage at full rate, invert the network, then fine-tune on the correct
version. The two files differ only where it matters, so what inversion did is
legible in the output.

```bash
python -m radixnet train --data data/sample_corpus.txt --epochs 5
python -m radixnet 2nrl --bad data/sample_garbage.txt --good data/sample_corpus.txt
python -m radixnet codegen --problems data/sample_problems.txt --phase teacher
python -m radixnet agent --tasks data/sample_tasks.txt
```

Uploads made through the frontend land in `../uploads/`, which is ignored by git.
