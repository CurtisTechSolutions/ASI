# The 400-round run

Ten times the rounds of `../results/`, and the first configuration where a high
round count is worth running at all. Three things had to be true first:

**Model selection.** Every arm peaks early and then decays — `positive` reaches
6.3 refusals a move at round 9 and is back to 151 by round 200 (`../results_long/`).
Before selection existed, more rounds were strictly harmful, because the reported
number was whatever round the run stopped on. A run now keeps the network that
scored best on a validation split and grades it on a disjoint test split, so the
decayed tail is discarded and more rounds only widen the search for the peak.

**A trigger that does not fire on noise.** The flip used to test a single
round's value against the best round so far. Round-to-round noise in `p` is
0.042 against a 0.01 threshold, so it fired on noise: two of three seeds ended
phase 1 while `p` was still climbing, at 0.60 and 0.81. Every test now applies
to a mean over `trigger_window` rounds.

**Room for both phases.** At 40 rounds, two of three seeds hit the `n − 1`
deadline — phase 1 ran out of rounds rather than finishing. At 400 it can reach
its natural `reproduced` trigger *and* leave phase 3 a large budget, which is
the trade §12 item 4 leaves open.

| file | what produced it |
|---|---|
| `main.json` | `2nrl` and `positive`, three seeds, 400 rounds |
| `others.json` | `2nrl-worst`, `2nrl-random`, `repulsion`, same |
| `elo.json` | the rating ladder over the **selected** networks |

Reproduce with `./run_r400.sh`.
