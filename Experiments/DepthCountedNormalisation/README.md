# DepthCountedNormalisation

The experiment behind `Research/VanishingGradientIsAFeature.md`. A paper in
`Research/` is not allowed to rest on an argument alone — where a claim is
testable, the test lives here and the paper cites the numbers it produced.

**Standard library only** — no numpy, no torch — so the script runs anywhere
with nothing installed, the same rule `RadixCyclicNN/` follows.

## Contents

| file | tests |
|---|---|
| `depth_counted_normalisation.py` | `Research/VanishingGradientIsAFeature.md` — *is the vanishing gradient a measurement?* |

### `depth_counted_normalisation.py`

The claim: back-propagation multiplies, so a gradient `n` products away from the
loss arrives at about `c**n` of the size it began with. Nothing is *lost* there —
`c**n` is invertible. The magnitude was encoded, and the key to the code is `n`,
which is free: it is the depth, known from the architecture before any data is
seen. The vanishing gradient looks like information loss only because we throw
the key away. So: count the multiplications, divide the decay back out, and see
what happens.

What it measures, in order:

1. **The premise.** Does the magnitude really decay like `c**n`? Fit
   `log‖g_n‖ = a + b·n` across the layers and report `c = exp(b)` with R². A
   high R² means `n` is a *sufficient statistic* for the decay — the decay is
   fully predictable and therefore fully removable.
2. Then the consequences of removing it, against the usual baselines.

```bash
python3 depth_counted_normalisation.py            # full run (~8 min)
python3 depth_counted_normalisation.py --quick    # ~1 min
```

## Related

`../README.md` lists every experiment in this directory and the claim it tests.
`Research/README.md` maps every idea to the code that implements it.
