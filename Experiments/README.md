# Experiments

Every test of the ideas in this repository, in one place. The architectures —
`RadixCyclicNN/`, `GREN/`, `GTMNN/`, `CyclicCortex/` — are where the ideas are
*built*. The papers in `Research/` are where they are *argued*. This directory
is where they are **measured**, and a claim that fails here is recorded as
having failed.

Each experiment stands alone: its own directory, its own README, its own run
instructions, and the numbers it produced committed alongside the code that
produced them.

## Contents

| experiment | the claim it tests | the answer |
|---|---|---|
| [`ActivationFunctionTest/`](ActivationFunctionTest/) | the sine activation beats sigmoid and ReLU | **yes** — 28-48 reward against ReLU's 10 and sigmoid's 9 on CartPole |
| [`TwoNRL_CartPole/`](TwoNRL_CartPole/) | 2NRL works on a real control task | **yes** — trained only to fail (9.2 steps), one sign flip takes it to 499.4/500 |
| [`NeuralCompression/`](NeuralCompression/) | one network can hold many games; inversion as an escape from a stall | **mixed** — compression holds, but the output partition is *not* what does it |
| [`SBNN_RNN_ActivationFunction/`](SBNN_RNN_ActivationFunction/) | a learnable activation inside a conventional RNN that grows on a stall | the idea in its smallest form, gradient-checked to 4.8e-09 |
| [`DepthCountedNormalisation/`](DepthCountedNormalisation/) | the vanishing gradient is a measurement, not a loss | fit `log‖g_n‖ = a + b·n` and divide the decay back out |

## What each one needs

Most of this runs with nothing installed. The exceptions are called out below.

| experiment | dependencies |
|---|---|
| `NeuralCompression/` | **none** — standard library, deterministic |
| `DepthCountedNormalisation/` | **none** — standard library |
| `ActivationFunctionTest/self_building_sinewave.py` | **none** — standard library |
| `ActivationFunctionTest/` (the other five scripts) | `gymnasium`, plus `tensorflow` or `torch` depending on the script |
| `SBNN_RNN_ActivationFunction/` | `numpy` |
| `TwoNRL_CartPole/` | `numpy`, `gymnasium` (`make install`) |

## Running them

From this directory — each runs in a subshell, so the order does not matter:

```bash
(cd NeuralCompression && python3 experiment.py)
(cd DepthCountedNormalisation && python3 depth_counted_normalisation.py --quick)
(cd ActivationFunctionTest && python3 self_building_sinewave.py --quick)
(cd SBNN_RNN_ActivationFunction && python3 main.py)
(cd TwoNRL_CartPole && make quick)
```

Read each directory's README first — several have a `--quick` mode worth using
before committing to a full run, and `NeuralCompression/FINDINGS.md` is the one
document here that collects results rather than describing a procedure.

## Where the results are cited

`Research/Insights.md` tracks every claim against the experiment that
confirmed, refined or contradicted it. `Research/README.md` maps each paper to
the code that implements it. `RadixCyclicNN/DECISIONS.md` records which of
these numbers a design decision was made on.
