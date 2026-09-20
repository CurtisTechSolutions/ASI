# results

The numbers in `../README.md`, committed beside the code that produced them.

| file | what it is |
|---|---|
| `main_results.json`, `main_run.log` | every arm × 3 seeds, one corpus, depth 5 — the headline table |
| `baseline_results.json` | the same trees with nothing learned: what the one-hop rule is worth |
| `passes_results.json` | how many hinge passes the refilter step should take |
| `probe_results.json` | the four supervised measurements that locate the routing gap |
| `depth_results.json`, `depth_run.log` | four arms across context depths 2…6 |
| `scale_results.json`, `scale_run.log` | four arms across 100…800 segments per source |
| `quick_results.json`, `quick_run.log` | what `make quick` writes — a one-seed smoke run, not a result |

Each JSON holds, per run: the arm, the seed, held-out bits per character, the
size of everything it built, the routing's agreement with the source labels
(`purity`, `nmi`), the per-round history, the load across addresses, the filter
units' `|a·b|`, and the contingency table of address against source. The
`corpus` block records the SHA-256 of the exact text the run was measured on.

Reproduce with `make experiment` (slow) or one sweep at a time:

```bash
make main      # the headline table
make baseline  # what the one-hop rule is worth
make passes    # the refilter step's hinge passes
make depth     # the depth sweep
make scale     # the scale sweep
make probe     # where the routing gap is
```

`python3 summarize.py` reads these files back and prints the tables that are in
`../README.md`. If the two ever disagree, the script is right.
