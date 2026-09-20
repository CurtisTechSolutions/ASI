# results

One JSON per arm, written by `experiment.py <arm>` and read back by
`experiment.py summary`, which regenerates the tables in
[`../README.md`](../README.md) between their `<!--NAME-->` markers. The tables
are never typed by hand, so the README and the runs cannot drift apart:

```bash
make all            # rewrite every file here (three seeds, ~40 minutes)
make summary        # print the tables
python3 experiment.py summary --into-readme    # splice them into README.md
```

| file | arm |
|---|---|
| `rules.json` | six games, one encoder, the same defaults for all of them |
| `phi.json` | the abstraction sweep, from a move-only tape to an injective one |
| `mix.json` | the same bits split between a position hash and a summary of the last move |
| `phase.json` | `plain` against `disjoint`, with the phase-ambiguity counts |
| `layout.json` | `struct` / `lsb` / `msb` - the digits inside a token |
| `multi.json` | one network holding all six games, and what it costs |
| `twonrl.json` | 2NRL against a matched positive control and a local-inversion arm |

## Reading one

Every arm records the `config` it ran under (corpus size, epochs, evaluation
positions) and the `seeds`, and every number is pooled across those seeds as a
mean plus `_ci` - the half-width of its 95 % interval. **A `_ci` of 0 means one
seed**, not agreement between three.

The measurement keys are the same everywhere and come from
`uge/metrics.py:evaluate`:

| key | what it is |
|---|---|
| `coverage` | fraction of the evaluated positions whose prefix ends on a trigram the graph has. The model is silent on the rest, and every other number has to be read against this |
| `legal_at_1` / `_covered` / `_baseline` / `_fallback` | the top token was legal: over all positions, over covered ones, what a uniform guess scores, and the model-where-it-knows-guess-where-it-does-not combination |
| `refusals` / `refusals_baseline` | moves the board refuses before accepting one, against what guessing costs. The tail the model did not rank is charged analytically, not sampled |
| `teacher_match` | the top token was the move the teacher actually played |
| `optimal_match` | (Nim only, which is solved) the move taken was one that wins from that position — and from a position that is already lost, any move counts, which is how `Nim.optimal` is defined and which inflates the number against a strong opponent |
| `greedy_plies` / `greedy_break` | how far the model gets playing both sides with no second chance, and what ended it |
| `selfplay_*` | the same rollout allowed to walk down its own ranking - measured on **its own** positions, which by move ten are not the teacher's distribution |
| `free_plies` / `free_break` / `free_best_plies` | the model writing a whole tape by itself. It cannot; see README limitation 3 |
| `ppl_*` | held-out log-probability per token, and what fraction of it was the unknown-transition charge rather than a real probability |
| `nodes` / `edges` / `trigrams` / `compression_ratio` | the graph the corpus built |

`rules.json` and `mix.json` additionally carry an `unseen` block: the same
battery over test positions whose `phi` token the training corpus never wrote.

## Reproducing

Everything is seeded. The corpus for a `(game, count, teacher, seed)` is
generated once and re-encoded for every spec, so two rows of an arm differ in
the encoding and in nothing else. `python3 tests.py determinism` asserts that a
corpus is a function of its seed and that a shorter corpus is a prefix of a
longer one.
