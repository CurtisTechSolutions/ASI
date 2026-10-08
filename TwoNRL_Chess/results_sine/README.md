# The inversion operator, three ways

`run_operators.sh`. Sine activation throughout, the rule heads included. 60
rounds, 3 seeds. The three `2nrl` rows flip at round 36 for the same reason in
every seed, so their phase-1 networks are **bit-identical** — the only thing that
differs between them is the operator applied at the flip.

| operator | what it does |
|---|---|
| complement | `W → 1 − W`, every layer, nothing else |
| per-unit flip | `a → −a`, `k → −k`, `W → −W` past the first layer |
| sign flip | `W → −W`, every layer, nothing else |

## The flip alone — same network, three operators

Phase 1 trains the network to get the rules backwards, and it does: legal AUC
**0.010**, which is a network that ranks illegal moves above legal ones almost
perfectly. Then one operation, no training:

| operator | refusals / move | legal AUC | capture | gives check |
|---|---|---|---|---|
| before the flip | 1163.4 | 0.010 | 0.003 | 0.015 |
| **complement** `1 − W` | 137.8 | **0.514** | 0.522 | 0.529 |
| per-unit flip | **1.6** | **0.989** | 0.997 | 0.985 |
| sign flip `−W` | **1.7** | **0.988** | 0.997 | 0.196 |

The negation-based operators **reverse** the learned rules — 0.010 becomes 0.989.
The complement **erases** them: every head lands at chance, 0.5, which is a
network that knows nothing rather than one that knows the opposite.

## Why: `1 − W` is `−W` plus a phase shift

`1 − W` is `−W + J`, where `J` is all ones. So the first layer computes

```
z' = x·(1 − W) + bias  =  −z  +  Σx  +  2·bias
```

exactly (checked to 2.1e-14). The `−z` term *is* the negation. But `Σx` is the
number of active inputs, which for this board encoding is **31 to 43** per
position, and the sine turns that offset into a phase shift of **10.3 to 14.3
radians** at `b = 1/3`. Since `sin(φ − u) = sin φ·cos u − cos φ·sin u`, the shift
mixes `cos u` into what was `sin u`, and the ordering the negation would have
reversed comes out scrambled instead. The shift is also different for every
position, because it depends on how many pieces are on the board.

Put another way: `1 − w` is a reflection about 0.5, the middle of `[0, 1]`. These
weights are not in `[0, 1]` — they are Glorot-initialised, centred on 0 — and the
reflection about the middle of *their* range is `−w`.

## After phase 3

| arm | refusals / move | legal AUC |
|---|---|---|
| `2nrl`, complement | 92.3 ± 42.8 | 0.633 |
| `2nrl`, per-unit flip | **1.6 ± 1.6** | 0.948 |
| `2nrl`, sign flip | 2.8 ± 2.6 | 0.939 |
| `positive` | 6.3 ± 1.5 | 0.964 |
| `repulsion` | 0.5 ± 0.4 | 0.969 |

24 rounds of phase 3 recover some of what the complement erased (AUC 0.514 →
0.633) but not most of it. With either negation, `2nrl` ends level with the
controls.

## Two smaller things worth knowing

- **The sign flip's output error is large and it doesn't matter.**
  `|f_after + f_before|` is about 5 for the sign flip against 0.000 for the
  per-unit flip, because `h`, `k` and the bias have moved off zero — yet it
  reverses the rules as well as the exact operator does (0.988 against 0.989).
  What 2NRL needs is the *ranking* reversed, not the output exactly negated.
  One head is the exception: *gives check* only reaches 0.196 under the sign
  flip.
- **`positive` is worse here than with sigmoid heads** — 6.3 refusals against
  0.9 in `results_rules/`, at the same legal AUC (0.964 against 0.966). MSE on
  the bounded wave gives the answers smaller margins than cross-entropy on an
  unbounded logit did, which costs the ranking even where it doesn't cost the
  AUC.
