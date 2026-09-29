# results

The numbers in `../README.md`, committed beside the code that produced them.

| file | what it is |
|---|---|
| `habit_results.json` | saying `'the'` ten times over on the sample corpus: what it says, the probability of it per repeat, the shares at the branch before and after; asked quietly as the control; thirty sampled sayings |
| `forgetting_results.json` | read once, then 0 to 8 lives of silence, under each decay: nodes remembered, texts recited, bits per character, misses |
| `use_results.json` | twenty silences of a tenth of a life each, broken by saying one text — said, said quietly, or not at all |
| `recovery_results.json` | fade four lives, re-read every third text once, then all of them once |
| `experiments.log` | the four tables as `make experiments` printed them |
| `prose/` | the same four on the 800-line prose corpus with a life of 100 000 traversals — `make experiments-prose` |

Each JSON holds the experiment's settings (life, decay, floor, depth,
encoding, seed), the corpus size, and its rows; `DESIGN.md` §13 says what
each experiment does and measures. Reproduce with:

```bash
make experiments                    # the sample corpus, a second
make experiments-prose              # the prose corpus, about a minute
make experiments LIFE=2000 DECAY=linear
python3 -m radixdecay experiment --which forgetting --data data/corpus.txt --life 100000
```
