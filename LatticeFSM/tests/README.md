# tests

```bash
make test                          # the Rust tests, clippy and fmt; then these, with parity
make py-test                       # or: python3 -m unittest tests.test_latticefsm
python3 -m latticefsm test         # the same, through the CLI
make parity                        # TestRustParity alone (needs make build)
```

Plain `unittest`, standard library, a few seconds, deterministic (54 tests; the
parity ones need `rust/target/release/latticefsm` and skip without it). Each
test names the rule it protects; the ones that hold the line on something
this model is specifically about:

| test | what it holds the line on |
|---|---|
| `TestLattice.test_the_matrix_is_dense_and_three_dimensional` | `S · A · S` edges from the start, each knowing its place; `row`, `leaving`, `arriving` |
| `TestLattice.test_the_default_matrix_is_13_by_13_by_13` | a machine made without a shape is 13 states over `a` to `m`, 2 197 edges, and can still be taught every language |
| `TestLattice.test_every_edge_has_its_own_weighting` | the prototype is copied, not shared: adapting one edge's function moves no other's |
| `TestEdge.test_traversal_writes_the_history_and_the_trace` | a traversal writes `seen`, `first_seen`, `last_seen`, the trace, and the features of that moment |
| `TestEdge.test_credit_writes_the_verdict_and_adapts_the_weighting` | reward and punishment are kept apart, stamped, and move the bias by `rate · credit` |
| `TestEdge.test_the_adaptation_follows_the_features_at_traversal` | each coefficient moves by `rate · credit · feature`, the feature as it was when the edge was traversed |
| `TestEdge.test_width_widens_narrows_relaxes_and_is_bounded` | use and reward widen, punishment narrows, the excess halves in a life, the clip holds |
| `TestEdge.test_the_stimulation_term_is_stimulation_times_log_width` | the whole of §3's stimulation term, at four levels |
| `TestMachine.test_a_quiet_run_moves_nothing` | a quiet run makes the same draws and changes no field, no clock, no path |
| `TestMachine.test_credit_is_discounted_back_from_the_end` | the last edge receives the credit, each earlier one `discount` times the next |
| `TestMachine.test_stimulation_prefers_the_wide_channel_more_the_higher_it_is` | the wide edge's probability rises monotonically with stimulation, and is the odds at 1 |
| `TestMachine.test_stimulation_relaxes_to_the_baseline_on_the_clock` | a surge halves every `calm` ticks; the floor is 0 |
| `TestMachine.test_a_custom_weight_function_replaces_the_edges_own` | `weight_fn` is consulted and the edges' own functions are not |
| `TestMachine.test_time_fades_traces_and_widths_but_not_verdicts` | after ten lives the trace and the width are gone and `rewarded`, `seen`, `net` are as they were |
| `TestLearning.test_a_language_is_learned_by_credit_and_not_quietly` | 1 500 credited runs reach 95% on `even-b`; the same runs quietly stay at chance and touch no edge |
| `TestLearning.test_the_learned_table_is_the_language` | the greedy table of a two-state machine taught `even-b` is the parity automaton |
| `TestPersistence.test_round_trip` | a saved machine reloads to the same JSON, stats, table and next draws, `.json` and `.json.gz` alike |
| `TestPersistence.test_a_file_from_a_foreign_generator_reseeds` | a file carrying the Rust generator's state reseeds Python's from the seed |
| `TestGeometry` | the 13³ cube's centre `(6, 6, 6)`, its seven shells of 1, 26, 98, 218, 386, 602 and 866 cells, and a centre-out order that visits every cell once, never moving inward |
| `TestCompression.test_exact_is_lossless_bit_for_bit` | an exact code rebuilds every field of every edge, the states, the clock, the counters and the stimulation exactly |
| `TestCompression.test_smaller_precisions_lose_a_little_and_keep_the_behaviour` | float32 and float16 are smaller in turn, lossy in state, and change no greedy choice |
| `TestCompression.test_a_budget_takes_the_least_lossy_precision_that_fits` | exact when it fits, then float32, then float16 |
| `TestCompression.test_rebuilding_from_the_middle_outward` | a rebuild to `k` shells gives back exactly the cells within them and the prototype beyond |
| `TestCompression.test_walking_straight_from_the_code` | the greedy walk read from the code, from the start and from the middle, is the machine's own |
| `TestFromTheMiddle` | a run can start from the middle state, traversed or quiet |
| `TestCompressEvery` | the matrix is folded every N transitions (a lesson counts, a quiet run does not); an exact rebuild on a clock changes nothing; a float16 one is applied; a rebuild keeps the run's path valid; the schedule is saved |
| `TestRustCompressionParity` | the two ports write the same code byte for byte at every precision, half floats on rounding boundaries included, and each rebuilds the other's |
| `TestCLI.test_every_command` | every command runs out of a checkout, `compress`, `expand`, `core-run` and `run --from-middle` among them |
| `TestRustParity.test_the_deterministic_experiments_agree` | the stimulation and adaptation records from the two ports are equal to `10⁻⁹` |
| `TestRustParity.test_a_machine_file_crosses` | a machine trained by the Rust binary loads in Python to the same greedy table, and saves back to a file the binary reads |

The Rust crate's own tests (`rust/tests/machine.rs`, `rust/tests/server.rs`,
and the unit tests in `json.rs` and `gzip.rs`) cover the same rules from the
other side (`rust/tests/compress.rs` the central node), every HTTP route over a real socket, and the frontend's static
files (served from their directory, an unknown path answered as the app's
own route, a path that climbs out refused). The frontend's tests
(`frontend/test/*.test.mjs`, `make frontend-test`, plain `node --test`) cover
its pure modules: the settings store and the matrix helpers.
