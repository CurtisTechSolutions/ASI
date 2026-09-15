# results

The run logs and JSON behind every table in `../README.md`. Committed so the
numbers in that document can be checked, and so `../summarize.py` can rebuild
the tables without re-running the experiment.

Each JSON file holds `{"runs": [...], "controls": [...]}` — one entry per seed,
each carrying the reward at all four stages (`untrained`, `after_negative`,
`after_invert`, `after_finetune`). The `.log` beside it is the full console
output of the same run: every negative-phase update with its mean episode
length, graded loss and venture radius.

## Contents

| file | the run that produced it |
|---|---|
| `twonrl_results.json` / `main_run.log` | the main experiment — `--seeds 0 1 2 --control`, delta-driven negative-phase starts. This is the table in `../README.md` § *Results*. |
| `ablation_no_venture.json` / `ablation_run.log` | the ablation with venturing out switched off — `--seeds 0 1 2 --fixed-starts`, standard negative-phase starts |
| `grading_speed_results.json` / `grading_speed_run.log` | the ablation that grades by speed ratio instead of graded loss — `--seeds 0 1 2 --grading speed` |

## Reproducing them

From `..`:

```bash
python3 twonrl_cartpole.py --seeds 0 1 2 --control \
        --out results/twonrl_results.json
python3 twonrl_cartpole.py --seeds 0 1 2 --fixed-starts \
        --out results/ablation_no_venture.json
python3 twonrl_cartpole.py --seeds 0 1 2 --grading speed \
        --out results/grading_speed_results.json
python3 summarize.py                       # rebuild the tables in ../README.md
```

Roughly three minutes a seed on one core, pure NumPy. Seeded, so a re-run should
land on the same numbers.

`--out` is what puts a run at the path `summarize.py` reads; without it the run
prints and is not saved. The names above are the three it looks for.
