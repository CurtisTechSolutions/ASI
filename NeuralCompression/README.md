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
| `experiment.py` | the core: the `MLP`, the synthetic game family, and the three experiments behind FINDINGS §1-5 — does compression work, what actually does the compressing, does band order matter, can it play an unseen game, and the sine-frequency sweep |
| `partition.py` | how the range should be laid out (FINDINGS §6-7): the cost of reserving empty slots, and the damage done to learned games when new ones are added — `one_over_N` vs `static_pow2` vs `adaptive` |
| `vanishing.py` | mechanism 1: let granular training stall, **detect** the stall, flip the update sign to climb back out, then resume descent. Three triggers (gradient magnitude, weight magnitude, loss plateau) because they fire in different regimes. None of them helps — see FINDINGS §11. Also holds `Rotating`, the alternating-parity inversion of part 4. |
| `consolidate.py` | mechanism 2: a specialist learns one game alone at full output range, then its learning is moved into one band of the shared query network. Three transfer routes — `joint`, `graft`, `replay` — the last being the REM reading: the fast learner replays experience, the slow store learns from the replay. |
| `__pycache__/` | compiled bytecode. Generated, and committed by accident — `.gitignore` already covers it. |

## Running it

```bash
python3 experiment.py            # FINDINGS §1-4 and §5
python3 experiment.py sweep      # the sine-frequency sweep only (§5)
python3 partition.py             # FINDINGS §6-7
python3 vanishing.py             # FINDINGS §10-11
python3 consolidate.py           # FINDINGS §12
```

Nothing to install. `consolidate.py` and `partition.py` import `MLP`,
`make_games`, `target` and `data` from `experiment.py`.

The scripts print their own `EXPERIMENT n` headings, which are a *different*
numbering from `FINDINGS.md`'s `§n` sections and do not line up with them — the
comments above are the mapping. Part 4 (§15-18) is the one set of numbers with no
default run: `Rotating` lives in `vanishing.py` but is not called from its
`__main__`.

## The headline results

| # | Question | Answer |
|---|---|---|
| 1 | Does one network hold many games? | **Yes, and it beats separate networks** — 3.2× fewer parameters *and* lower error at N=16 (with tanh; the win does not appear under `sine b=1`). |
| 2 | Does the output partition do the compressing? | **No.** Plain conditioning is 7.5× more accurate at the same scale. |
| 3 | Does band ordering by similarity matter? | **Decisively**, and it grows with N. At N=16 shuffled is 15× worse — worse than predicting a constant. |
| 4 | Is there anything the partition wins at? | **Yes — games it has never played.** Best zero-shot score of any condition; one-hot conditioning cannot do this at all. |
| 5 | Is the sine activation's `b = 1/3` right? | **No.** 17× *worse* than tanh at `b = 1/3`; 6-24× *better* at `b = 1-2`. |
| 6 | Does inverting the update escape a stall? | **No.** Of three triggers, the best result is *no effect*; on a healthy network one inversion can cost 600×. Fixing `b` is worth 536× on the same stall. |

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
