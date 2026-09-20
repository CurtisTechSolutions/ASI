# tests

```bash
make test                      # or: python3 -m tests.test_fbradix
```

25 tests, standard library, a few seconds, deterministic. Each one names the
invariant it protects rather than the function it calls, and several are
regressions for bugs this architecture actually had — they say so in their
docstrings:

| test | the bug it holds the line on |
|---|---|
| `test_ignorance_is_never_cheaper_than_knowledge` | an expert that had never seen a context paid the floor, which made *not knowing* score better than a diffuse guess, and the filter learned to exploit it |
| `test_an_empty_expert_falls_back_to_the_prior` | the eleven-bit cliff at an empty address that drowned out the routing signal |
| `test_the_prediction_is_a_probability_distribution` | three smoothings are composed; the composition is only a model if it still sums to one |
| `test_the_bank_rebuilds_instead_of_extending` | experts that are extended rather than rebuilt each slowly become the whole corpus |
| `test_calibration_splits_every_bit_in_half` | an uncalibrated unit can start with every segment on one side, costing the bank half its addresses before training |
| `test_features_are_scale_free` | features that scale with length make the filter route by how long the cut was |
| `test_a_single_child_costs_nothing_to_train` | a softmax over one child is 1: those positions must not enter the plan |

The two gradient checks (`test_one_hop_gradients_match_finite_differences`,
`test_filter_gradients_match_finite_differences`) read the analytic gradient out
of the code that *trains*, not out of a second implementation of the same
formula, and both agree with central differences to ~1e-10.
