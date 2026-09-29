# tests

```bash
make test                          # or: python3 -m unittest tests.test_radixtree
python3 -m radixtree test          # the same, through the CLI
```

Plain `unittest`, standard library, about half a minute, deterministic (71
tests; the phonetic ones need `../PhoneticTokenizer` and skip without it). The
structural tests mirror `RadixCyclicNN/tests/test_graph.py` case for case, with
the tree's answers where the graph's were cycles. Each test names the invariant
it protects; the ones that hold the line on something the acyclic design is
specifically about:

| test | what it holds the line on |
|---|---|
| `TestNoCycles.test_aaaa_makes_no_self_loop` | the graph's `"aaaa"` self-loop is three context nodes here, and `"aaaaaaa"` is not a root path — the loop is exactly what the tree cannot say |
| `TestNoCycles.test_repetition_is_unrolled` | `n` a's cost `n - 1` nodes: the counting argument of `Research/CyclesAreAFeature.md` §3, measured |
| `TestNoCycles.test_every_edge_goes_one_level_deeper` and `test_invariants_catch_a_cycle` | every edge runs to a strictly deeper node, and the invariant check refuses a node made its own parent |
| `TestRandomisedInvariants` | hundreds of random observe / split / compress operations, and it is still one tree afterwards |
| `TestSuffixProperty` | every substring of every text is a root path; counts are conserved down every node; `locate` prefers the whole history and backs off by suffix |
| `TestSplitAndMerge.test_merge_preserves_every_score` | the rescale rule: a merge moves no probability, whichever activation is kept |
| `TestSplitAndMerge.test_observation_never_leaves_a_chain` | a tree built by observing is compressed already, at any depth; only `split` leaves chains |
| `TestLearning.test_one_hop_gradients_match_finite_differences` | the analytic gradient is read out of the code that trains, and agrees with central differences to ~1e-9 |
| `TestLearning.test_grouping_by_parent_is_exact` | grouping a batch by parent is the per-transition rule summed, to the bit |
| `TestLearning.test_observe_returns_transitions_that_hold` | a later text splits nodes earlier transitions named; what comes back names the structure as it stands |
| `TestPrediction.test_each_node_is_visited_at_most_once` | the search has no `(node, chars)` state and no budget because it needs none |
| `TestPrediction.test_bounded_depth_re_enters_the_tree` | a bounded window makes the generator loop *outside* the tree, and `max_legs` is that loop's clock |
| `TestPrediction.test_negative_costs_are_legal_here` | a signed cost and a negative step penalty both terminate: no cycles, no negative cycles |
| `TestScoring.test_incremental_scoring_equals_relocating_at_every_step` | the fast walk of `score` equals the rule stated the slow way (relocate to the deepest usable context at every symbol), at both depths and both trust settings |
| `TestScoring.test_the_repetition_the_graph_generalises_the_tree_does_not` | shown `"aaaa"`, the tree charges three unknowns for `"aaaaaaa"` where the graph charges none |
| `TestInversion.test_path_inversion_is_exact_on_a_tree` | flipping every other node of a path flips **every** edge of it, END leaf included — exact where the graph is best-effort |
| `TestInversion.test_two_nrl` | 2NRL read at the branches where a garbage line leaves its twin: the inversion reverses every preference exactly, and the positive phase leaves the good one preferred |
| `TestPersistence` | a saved model reloads to the same tree, the same predictions, and the same continued training (the RNG state travels) |
| `TestSlide` | token by token: one query per token and one for END; a window of one gram is the first-order walk; a window cut from a history never begins a text; the caps, the clock, the seeded sample; `next_token_accuracy` |
| `TestPhonetic` | the tree over sounds (skipped when the tokenizer is not importable): text read as sounds and spelled back, a tree over phones and one over syllables that recite in words, scoring in phones, token by token in sounds, save and load with the encoding, and the CLI's `--unit phone` |
