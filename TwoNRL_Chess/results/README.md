# Runs

Every JSON here backs a table in `../README.md`, and `../summarize.py` rebuilds
those tables from exactly these files.  Reproduce the lot with `./run_all.sh`
(or one target at a time with `make run`, `make entropy`, and so on).

| file | what produced it |
|---|---|
| `main.json` | `2nrl` and `positive` - the headline comparison, three seeds |
| `entropy.json` | `2nrl-worst` and `2nrl-random` - the H(q) ladder of §11 |
| `repulsion.json` | arm E: the same negatives pushed away instead of inverted |
| `per_round.json` | ablation: §9.3's perpetual loop, inverting inside every round |
| `phase_depth_15.json` | ablation (§12 item 4): phase 1 is 15% of the run |
| `phase_depth_50.json` | ablation (§12 item 4): phase 1 is 50% of the run |
| `benchmark.json` | head to head, the common opponent, and the random-mover floor |

Each run JSON carries its full `config`, the per-round `history` (held-out
scores, what the phase was doing, the training batch and the play statistics),
the growth events, the learned activation parameters, and the engine call count.
