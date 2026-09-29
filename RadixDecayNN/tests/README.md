# tests

```bash
make test                          # or: python3 -m unittest tests.test_radixdecay
python3 -m radixdecay test         # the same, through the CLI
```

Plain `unittest`, standard library, a few seconds, deterministic (65 tests;
the phonetic ones need `../PhoneticTokenizer` and skip without it). The
structural tests are the acyclic sibling's, cut down to what this directory
repeats; the rest are the memory's. Each test names the rule it protects;
the ones that hold the line on something this model is specifically about:

| test | what it holds the line on |
|---|---|
| `TestTreeBasics.test_reading_one_text_visits_every_window` | one reading of `abcd` is nine traversals — three windows, each touching its origin, its node and its END leaf — and a second reading adds one to each |
| `TestMemory.test_seen_reads_through_time_and_changes_nothing` | `seen` is read through elapsed traversals and writes nothing; `settle` is the only write, and it changes the value read by nothing |
| `TestMemory.test_the_three_decays` | a lone visit is a half after one life under `half-life`, nothing under `linear`, itself under `none` |
| `TestMemory.test_touch_settles_adds_one_and_ticks` | a traversal is settle, add one, tick the clock — and the clock ticks between two touches of the same node |
| `TestMemory.test_shares_are_proportions_of_seen` and `test_faded_options_are_not_offered` | a share is the option's part of what the options hold; an option that has faded to exactly nothing is not on offer; a branch whose options have all faded is not usable, though a run through it still is |
| `TestMemory.test_min_seen_is_the_floor` | at exactly half a visit a context is still consulted; below it, not — unless the floor is zero |
| `TestMemory.test_split_keeps_seen_on_both_halves_and_merge_keeps_the_larger` | the split and merge rules of `DESIGN.md` §3 |
| `TestMemory.test_random_operations_keep_one_tree` | three hundred random reads, splits, touches, ticks and compressions, and it is still one tree with a sane memory |
| `TestMemory.test_option_total_cache_follows_the_clock_and_the_structure` | the cached option total is invalidated by a touch, a tick and a split |
| `TestSuffixProperty.test_walks_answer_what_locate_answers` | the incremental cursors of `Walks` find, at every token, exactly the context a fresh walk of every suffix finds — on a tree that has said things and faded |
| `TestSaying.test_saying_is_traversal_and_asking_quietly_is_not` | a saying adds exactly one to every node it arrives at, the context it starts from included, and moves the clock by that many; asked quietly, nothing moves |
| `TestSaying.test_reading_and_saying_weigh_the_same` | one reading and one saying each add one to the option they pass through, and the model's choice follows the larger total of the two |
| `TestSaying.test_a_habit_forms` | the probability of the text said rises with every saying; the branch it decides at sharpens |
| `TestSaying.test_a_faded_start_says_nothing` | a model whose START has faded below the floor says nothing from nothing |
| `TestCheapest.test_cheapest_touches_its_path_once_each` | the one-search walk traverses every node of the path it returns, once |
| `TestScoring.test_the_fast_walk_equals_the_definition_on_a_tree_built_by_reading` | the incremental scoring walk equals "the deepest usable context of the whole history decides at every step", fresh and faded, at two depths and two floors — with ROOT as the empty context of last resort |
| `TestScoring.test_measurements_are_quiet` | score, bits, accuracy, recital counts and stats leave the tree untouched |
| `TestForgetting.test_a_memory_fades_and_reading_brings_it_back` | two lives of silence and the corpus is gone; one reading and it is back |
| `TestForgetting.test_the_*_experiment` | each experiment of `README.md` reproduces its claim on the sample corpus in seconds |
| `TestPersistence.test_round_trip` | a saved model reloads to the same `seen` at every node, the same sayings, the same scores, and the same next draw |
| `TestPhonetic` | the tree over sounds (skipped without the tokenizer): read as phones, said and spelled back as words, scored in phones, saved with its encoding, and the CLI's `--unit phone` |
| `TestCLI` | every command runs out of a checkout: read, say (and its `--save` of the changed model), cheapest, generate, score, accuracy, stats, tick, experiment, demo, and the `train` / `predict` aliases |
