# tests

Run from `CyclicCortex/`:

```bash
python3 -m tests.test_v1        # the cortex itself — 20 tests
python3 -m tests.test_server    # the arena, the job runner and the API — 19 tests
```

No framework: each file collects its own `test_*` functions and runs them, so
there is nothing to install. Pure standard library, like the rest of the package.

| file | what it holds |
|---|---|
| `test_v1.py` | the invariants the design rests on — the **growth identity** (output bit-identical after growing hidden units or inputs), permanent vocabulary slots, **zero triangle-inequality violations**, exact Shapley with efficiency to machine precision and the null-player axiom, the rule selected on play quality rather than legality, and that every game learns above its majority baseline |
| `test_server.py` | that a move key names exactly one legal move in every game, that a human cannot play an illegal one, that **the cortex's pick is not filtered to legal moves** (the illegal rate only exists because it is not), that every pairing of human / cortex / bot plays every game, that an untrained region is reported as untrained, that a training job reports one measurement per chunk and can be stopped, that two writers on one model are refused, that a fork is a copy and not an alias, that job event cursors neither repeat nor skip, and that a checkpoint name from the browser cannot escape the checkpoint directory |

The rule both files follow: assert the thing that would be a **silent lie** if it
broke. A board that draws is not worth a test. A match that reports zero illegal
moves because it quietly filtered them is.
