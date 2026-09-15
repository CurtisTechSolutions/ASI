# NeuralCompression

**Can one network hold many games?** Four experiments that test the
split-and-selector scheme — each game occupying a band of the output range, with
a selector picking the band — and then the two mechanisms proposed for getting
learning *into* such a network: accepting a vanishing gradient and inverting out
of it, and consolidating a fast specialist into a slow shared store.

Tested, not argued. **Pure standard library, deterministic, no dependencies.**

## Contents

| file | what it is |
|---|---|
| `FINDINGS.md` | the results — every number, every trap, and the predictions that turned out wrong. Read this first. |
| `experiment.py` | the core: the `MLP`, the synthetic game family, and experiments 1-4 (compression, what does the compressing, band ordering, zero-shot) plus the sine-frequency sweep |
| `partition.py` | experiment 5: when games are *added*, how much of what was learned is destroyed — `one_over_N` vs `static_pow2` vs `adaptive` layouts |
| `vanishing.py` | mechanism 1: let granular training stall, **detect** the stall, flip the update sign to climb back out, then resume descent. Two triggers (gradient magnitude, weight magnitude) because they fire in different regimes. Also holds `Rotating`, the alternating-parity inversion of part 4. |
| `consolidate.py` | mechanism 2: a specialist learns one game alone at full output range, then its learning is moved into one band of the shared query network. Three transfer routes — `joint`, `graft`, `replay` — the last being the REM reading: the fast learner replays experience, the slow store learns from the replay. |
| `__pycache__/` | compiled bytecode. Generated, and committed by accident — `.gitignore` already covers it. |

## Running it

```bash
python3 experiment.py            # experiments 1-4
python3 experiment.py sweep      # the sine-frequency sweep only
python3 partition.py             # experiment 5
python3 vanishing.py             # mechanism 1
python3 consolidate.py           # mechanism 2
```

Nothing to install. `consolidate.py` and `partition.py` import `MLP`,
`make_games`, `target` and `data` from `experiment.py`.

## The headline results

| # | Question | Answer |
|---|---|---|
| 1 | Does one network hold many games? | **Yes, and it beats separate networks** — 3.2× fewer parameters *and* lower error at N=16. |
| 2 | Does the output partition do the compressing? | **No.** Plain conditioning is 7.5× more accurate at the same scale. |
| 3 | Does band ordering by similarity matter? | **Decisively**, and it grows with N. At N=16 shuffled either diverges or is 10× worse. |
| 4 | Is there anything the partition wins at? | **Yes — games it has never played.** Best zero-shot score of any condition; one-hot conditioning cannot do this at all. |
| 5 | Is the sine activation's `b = 1/3` right? | **No.** 17× *worse* than tanh at `b = 1/3`; 6-24× *better* at `b = 1-2`. |

`FINDINGS.md` has the rest, including part 4 on rotating inversion: the parity
alternation has **period 4** (verified — cycles 4 and 8 return every weight
bit-identical, 1-3 and 5-7 do not), it is *safe* where plain gradient ascent is
catastrophic by 100× to a million×, and it is also exactly neutral — no benefit
as specified. A negative result, recorded as one.

## Two traps, recorded because either one alone reverses the conclusion

1. A partitioned output has targets in a band of width `1/N`, so its raw MSE is
   automatically `N²` smaller. Every error here is **rescaled** back into the
   game's own `[0,1]` first.
2. That same narrow band makes the gradient `N` times smaller, so the partition
   trains `N` times slower at a shared learning rate and reports a **false
   negative**. Loss is measured in game units (`gain=N`) for every condition,
   which then requires gradient clipping at 5.0 or the amplified gradient
   diverges at N=16.

## Related

* `Research/VanishingGradientIsAFeature.md` — the argument `vanishing.py` tests.
* `Research/Insights.md` — the index these findings are folded into.
* `GTMNN/DESIGN.md` §5 and `RadixCyclicNN/radixnet/activation.py` — the two
  specs that §5's `b = 1/3` finding lands on.
