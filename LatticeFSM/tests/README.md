# tests

```bash
make test                          # the Rust tests, clippy and fmt; then these, with parity
make py-test                       # or: python3 -m unittest tests.test_latticefsm
python3 -m latticefsm test         # the same, through the CLI
make parity                        # TestRustParity alone (needs make build)
```

Plain `unittest`, standard library, a second, deterministic (35 tests; the
parity ones need `rust/target/release/latticefsm` and skip without it). Each
test names the rule it protects; the ones that hold the line on something
this model is specifically about:

| test | what it holds the line on |
|---|---|
| `TestLattice.test_the_matrix_is_dense_and_three_dimensional` | `S · A · S` edges from the start, each knowing its place; `row`, `leaving`, `arriving` |
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
| `TestCLI.test_every_command` | every command runs out of a checkout |
| `TestRustParity.test_the_deterministic_experiments_agree` | the stimulation and adaptation records from the two ports are equal to `10⁻⁹` |
| `TestRustParity.test_a_machine_file_crosses` | a machine trained by the Rust binary loads in Python to the same greedy table, and saves back to a file the binary reads |

The Rust crate's own tests (`rust/tests/machine.rs`, `rust/tests/server.rs`,
and the unit tests in `json.rs` and `gzip.rs`) cover the same rules from the
other side, and every HTTP route over a real socket.
