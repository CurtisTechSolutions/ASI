# The long run: 200 rounds instead of 40

Five times the rounds of `../results/`, everything else identical. It answers
one question — does more training help? — and the answer is no, in both metrics
and in both directions.

| arm | refusals @40 | refusals @200 | best ever | at round |
|---|---|---|---|---|
| `positive` | 36.8 ± 8.2 | 151.2 ± 2.7 | 6.3 | 9 |
| `2nrl` | 133.2 ± 7.0 | 153.5 ± 1.3 | 87.4 | 27 |

Both arms peak early and decay, and they decay *toward each other*: the Elo
spread across the five arms falls from 79 points at 40 rounds to 30 at 200,
with every arm's interval still containing zero.

| arm | Elo @40 | Elo @200 |
|---|---|---|
| `positive` | 60 | 37 |
| `2nrl-random` | 37 | 22 |
| `2nrl` | 0 | 18 |
| `repulsion` | 13 | 7 |
| `2nrl-worst` | −19 | 7 |

This is overfitting to the sliding buffer of the network's own failures: the
training loss keeps falling while held-out refusals drift up. It is why runs
now keep the best network by a validation split and report on a disjoint test
split — without that, "the result" is wherever the run happened to stop.

| file | what produced it |
|---|---|
| `main.json` | `2nrl` and `positive`, three seeds, 200 rounds |
| `others.json` | `2nrl-worst`, `2nrl-random`, `repulsion`, same |
| `elo.json` | the rating ladder over these networks, 1540 games |
