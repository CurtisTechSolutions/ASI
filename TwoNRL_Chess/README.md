# 2NRL on chess: an SBNN that learns the rules by breaking them

Testing [`Research/2NRL.md`](../Research/2NRL.md) on a game it was not designed
for, against a baseline it has never had. A self-building network plays
Stockfish, is refused and graded by it, and learns — **with** 2NRL, and then
**without** it at matched compute.

§10 of the paper is blunt: *"Plainly: I have not measured 2NRL against a
baseline."* §11 says what the measurement needs — matched compute, negative
sets of different entropy, two controls, several seeds, variance reported. This
is that measurement.

## The network is not given the legal moves

That is the point of using chess. §6.5:

> A game is two things: rules and payoffs... I went after the first, and I went
> after it by losing on purpose... a rule is invisible while you are obeying it.
> Play a legal move and the game says nothing. Play an illegal one and the game
> answers exactly: not that, and here is the line.

So the network proposes a move out of all `64 × 64 = 4096` from-to pairs. The
board refuses or accepts. It keeps proposing down its own ranking until
something is accepted, and **how many refusals that took** is the rule metric.
About 30 of 4096 moves are legal, so guessing costs about 135 refusals a move.

Only then does quality matter, and Stockfish supplies that too: one `MultiPV`
search grades every legal move in centipawns, so a move that is legal but bad
is a second, different kind of failure.

Two things to learn, in the order the paper says they come: first the **rules**,
by being refused, then the **payoffs**, by being graded.

## Forcing the rules

Being refused teaches the rules, but it teaches them *two rows at a time*. Each
decision handed the softmax one refused action and one wild one, inside a
twelve-wide ranking whose target was move *quality* — so "this move is illegal"
and "this move is bad" arrived on the same wire, in the same units, and the
network had to separate them from about two bits a position. It worked, slowly:
400 rounds got refusals from 660 down to 41.8, which is a long way from zero for
a rule set a beginner learns in an afternoon.

That was leaving the teacher on the floor. Every position answers **all 4096
questions at once**, exactly and for free — `board.is_legal` is ground truth,
there is no label noise, and it costs no engine call. So the rules stop being a
by-product of the ranking and become what they are: a supervised problem with
perfect labels. `rules.py` is that, and the output layer becomes seven heads on
one trunk:

| head | ground truth | what it forces |
|---|---|---|
| `quality` | the graded softmax over candidates, as before | payoffs |
| `legal` | `board.is_legal(move)` | the rules, entire |
| `pseudo` | `board.is_pseudo_legal(move)` | how the pieces move |
| `capture` | `board.is_capture(move)` | what a move takes |
| `check` | `board.gives_check(move)` | forcing moves |
| `safe` | the to-square is unattacked afterwards | not hanging pieces |
| `threat` | the from-square is attacked right now | which pieces are loose |

`legal` and `pseudo` factor the rule set the way chess itself does:

```
legal  ==  pseudo  AND  the move does not leave our own king in check
```

`pseudo` is pure geometry and blocking — a knight's L, a bishop's diagonal, a
rook stopped by the piece in front of it, the pawn that moves one way and
captures another. The **gap** between the two heads is precisely the non-local
part of the rules: pins, and the obligation to answer a check. A network with
`pseudo` and not `legal` has learned how the pieces move and not yet that a
pinned knight is nailed down, and keeping them apart makes that distinction
measurable instead of a guess about what went wrong.

The last four are not rules at all. They are the first things anyone learns
*after* the rules, and the trunk that has to answer them cannot get by on
memorising which squares are usually fine.

### Which actions get asked about

All 4096 per position would be 31 MB of mostly-zero features, and a terrible
curriculum besides: about 4066 of them are illegal, so a head could answer
"illegal" to everything and be 99.3% right. Four groups are sampled instead,
and the second is the one that matters:

| group | what it is |
|---|---|
| legal | every legal move in the position — the positives, and there are only about thirty |
| **hard** | the highest-scoring **illegal** actions under the network as it stands — *literally the refusals it is about to make*, mined fresh every decision |
| pin | pseudo-legal but illegal: moves that obey the geometry and leave the king in check |
| wild | uniform over the 4096, so the easy majority stays represented |

About ninety rows a position instead of two, with the negatives chasing the
network's current error rather than sitting still.

### Why a sigmoid head is the ideal 2NRL object

This is the part worth reading twice. §4.3's inversion negates every unit, so
every head's logit `z` becomes `−z`. For a **sigmoid** head that is not an
approximation of anything:

```
sigma(-z)  ==  1 - sigma(z)        exactly, at every z
```

The negation of the network is **the complement of the probability**. A head
trained in phase 1 to answer *"is this move illegal?"* answers *"is this move
legal?"* the instant it is inverted, at the identical confidence, with no
training whatsoever.

So the inverting arms learn the complement of the truth in phase 1 — that legal
moves are illegal, that captures do not capture, that checks are not checks —
and flip. The controls learn the truth throughout. Same rows, same labels, same
number of updates; only the sign differs, which is the entire comparison.

It is also the one place in this experiment where the inversion is *literally* a
logical NOT rather than an order reversal. `softmax(-z)` is not `1 - softmax(z)`,
so the quality head can only ever show the weaker, ordinal version of the claim.
The rule heads show the strong one, and `test_sbnn.py` asserts it to machine
precision.

<!--RULES-->
### What the heads know, on the held-out exam

| arm | legal AUC | pseudo-legal AUC | capture | gives check | lands safely | piece is loose | refusals per move |
|---|---|---|---|---|---|---|---|
| `2nrl` | 0.962 ± 0.012 | 0.970 ± 0.008 | 0.957 ± 0.026 | 0.956 ± 0.003 | 0.659 ± 0.011 | 0.419 ± 0.041 | 1.1 ± 1.0 |
| `2nrl-worst` | 0.951 ± 0.014 | 0.952 ± 0.014 | 0.997 ± 0.003 | 0.921 ± 0.018 | 0.650 ± 0.019 | 0.415 ± 0.028 | 0.4 ± 0.2 |
| `2nrl-random` | 0.959 ± 0.015 | 0.967 ± 0.016 | 0.995 ± 0.001 | 0.956 ± 0.004 | 0.647 ± 0.019 | 0.427 ± 0.034 | 0.5 ± 0.2 |
| `positive` | 0.966 ± 0.005 | 0.974 ± 0.006 | 0.995 ± 0.005 | 0.934 ± 0.012 | 0.632 ± 0.021 | 0.407 ± 0.017 | 0.9 ± 0.5 |
| `repulsion` | 0.976 ± 0.011 | 0.978 ± 0.011 | 0.991 ± 0.013 | 0.941 ± 0.014 | 0.641 ± 0.011 | 0.475 ± 0.072 | 4.0 ± 1.9 |
| `2nrl (no rule heads)` | — | — | — | — | — | — | 97.9 ± 4.1 |
| `positive (no rule heads)` | — | — | — | — | — | — | 41.8 ± 9.5 |

0.5 is no knowledge of the rules and 1.0 is the rules. A network and its
inversion score `a` and `1 - a`, exactly.
<!--/RULES-->

`--rule-updates 0 --rule-weight 0` reproduces the single-score network exactly,
which is the ablation in the table and also a proof in the suite. It reproduces
the 400-round run **bit for bit** over all 61 shared rounds and selects the same
round in every seed, so the two configurations differ in the curriculum and in
nothing else — and, incidentally, the 400-round run added nothing after round 44.

### And then the rating fell

Refusals went from 97.9 to 1.1 a move, and the networks' rating against a random
legal mover went **down**. `2nrl` scored 0.464 against random before the heads
and 0.143 after.

The obvious reading — learning the rules made it play worse — is wrong, and
`ranking_ablation.py` is the experiment that shows what happened. The same saved
network can be asked to play two ways with nothing retrained and no weight
changed: at `rule_weight = 0` it ranks on the quality head alone, needs about a
hundred refusals to find a legal move, and therefore plays whatever legality
allows a long way down an ordering that was never about legality — close to a
uniform draw over the legal moves. At `rule_weight = 1` it plays the move it
actually wants and finds it in a handful of tries.

<!--EXPOSURE-->
### The same networks, ranked two ways (80 games a row, against a random legal mover)

| network | ranked by | score vs random | centipawn loss | refusals per move |
|---|---|---|---|---|
| `positive` | quality alone | 0.631 ± 0.053 | 772 | 108.7 |
| `positive` | quality + 1 × legal | 0.463 ± 0.055 | 710 | 4.2 |
| `2nrl` | quality alone | 0.581 ± 0.054 | 693 | 110.1 |
| `2nrl` | quality + 1 × legal | 0.237 ± 0.046 | 710 | 5.2 |
| `2nrl-worst` | quality alone | 0.506 ± 0.053 | 812 | 100.2 |
| `2nrl-worst` | quality + 1 × legal | 0.356 ± 0.052 | 811 | 4.4 |
| `2nrl-random` | quality alone | 0.537 ± 0.052 | 721 | 121.4 |
| `2nrl-random` | quality + 1 × legal | 0.275 ± 0.049 | 742 | 2.4 |
| `repulsion` | quality alone | 0.619 ± 0.052 | 535 | 571.1 |
| `repulsion` | quality + 1 × legal | 0.388 ± 0.054 | 693 | 18.8 |
| `random legal mover` | — | 0.481 ± 0.056 | 774 | 0.0 |

Nothing was retrained between the two rows of a pair and not one weight
differs. Only the ordering the moves are proposed in changes.
<!--/EXPOSURE-->

Every arm falls, by 0.15 to 0.34, and every fall is two to five standard errors.
Read against the random legal mover in the last row: ranked on quality alone
every arm scores **above** it, and ranked with the legal head every arm scores
**at or below** it.

Centipawn loss does not account for this. It barely moves, it moves in both
directions, and it is not bad — three of the five policies give away *less* than
the random mover's 774. So the network's own moves are not worse by the grader,
and its games are much worse.

One piece of this is genuinely unexplained, and is left that way rather than
tidied: why the refusal walk *beats* a random mover, 0.63 against 0.48, when the
two give away the same 772 and 774 centipawns a move. Something about the
distribution the walk produces is worth ~0.15 of score and is invisible to a
per-move centipawn average. Worth chasing; not chased here.

The direction is not a mystery. A player drawing near-uniformly from the legal
moves has no plan to be punished for; a player with a consistent bad policy does.
Expressing a policy is only an improvement if the policy is good, and at ~700
centipawns given away per move these are not.

So the earlier ratings — `positive` at 112 Elo, and every number in this
repository from before the rule heads — were not measuring what they appeared to.
A network needing 45 to 160 refusals a move was being randomised by its own
ignorance of the rules, and the rating was largely the rating of that
randomisation. **Learning the rules did not make these networks worse players. It
revealed that they were not playing.**

The rules half of §6.5 is genuinely solved here — 4096-way, from scratch, with no
legal-move list ever handed over. The payoffs half is not, and this is the
measurement that stops the first from flattering the second.

### And now the arms separate

While legality was the dominant term, no arm was distinguishable from any other
at the board. With it supervised away, the payoff half comes into view and the
ordering is unambiguous — and it is not the one §11's P1 predicts:

| | score for the first |
|---|---|
| `2nrl` vs `positive` | **0.229 ± 0.088** |
| `2nrl` vs `2nrl-worst` | 0.340 ± 0.097 |
| `2nrl` vs `repulsion` | 0.368 ± 0.084 |
| `2nrl` vs `2nrl-random` | 0.389 ± 0.099 |

`2nrl` — the full method, train on the failure, invert, fine-tune — loses to
every other arm, including to both of its own degraded negative sets. `positive`,
the control that only ever trains toward Stockfish's move, beats everything. The
same ordering appears against a common opponent: scoring against a random legal
mover, `positive` 0.535, `repulsion` 0.465, `2nrl-worst` 0.368, `2nrl-random`
0.340, `2nrl` 0.243.

This is the comparison §10 asks for and §11 specifies, and on this task the
answer is that 2NRL costs move quality rather than buying it. What the inversion
*does* buy is the rules — ×850 on refusals in one closed-form operation — and
that benefit is now obtainable more cheaply and more completely by supervising
legality directly.

One number keeps all of this in proportion. Against Stockfish, a random legal
mover gives away 397.5 centipawns a move. Every arm here gives away more:
`2nrl-worst` 420.8, `positive` 486.7, `2nrl-random` 491.2, `2nrl` 562.8,
`repulsion` 566.4. None of these networks has learned to play chess. They have
learned its rules, which is what §6.5 set out to test and is a different claim.

<!--HEADLINE-->
### Final, on the held-out positions (mean ± sd over 3 seeds)

| arm | H(q) | legal first try | refusals per move | centipawn loss | agrees with Stockfish |
|---|---|---|---|---|---|
| `2nrl` | 0.00 | 0.790 ± 0.017 | 1.1 ± 1.0 | 289.9 ± 17.4 | 0.044 ± 0.004 |
| `2nrl-worst` | 0.14 | 0.785 ± 0.061 | 0.4 ± 0.2 | 279.4 ± 17.4 | 0.052 ± 0.006 |
| `2nrl-random` | 0.72 | 0.817 ± 0.016 | 0.5 ± 0.2 | 296.1 ± 6.2 | 0.033 ± 0.009 |
| `positive` | — | 0.783 ± 0.029 | 0.9 ± 0.5 | 290.9 ± 10.1 | 0.058 ± 0.013 |
| `repulsion` | 1.52 | 0.777 ± 0.040 | 4.0 ± 1.9 | 321.6 ± 10.3 | 0.042 ± 0.007 |
| `2nrl (no rule heads)` | 0.00 | 0.008 ± 0.004 | 97.9 ± 4.1 | 315.4 ± 5.5 | 0.032 ± 0.006 |
| `positive (no rule heads)` | — | 0.065 ± 0.023 | 41.8 ± 9.5 | 290.0 ± 2.3 | 0.045 ± 0.012 |
<!--/HEADLINE-->

## Phase 1 ends when the failure is learned, not on a date

§12 item 4 is an open question in the paper:

> I train on the garbage for a fixed small number of epochs. The entropy account
> in §5 implies phase 1 should run until the failure mode is *well* represented,
> since a half-learned failure inverts into a half-useful signal. There is
> probably an optimal depth and it probably depends on `H(q)`. I have not looked.

So the sign flip is an **event, not a date**. It fires on the first of three,
each an observation rather than an estimate (§6.4), and which one fired is
recorded per run:

| trigger | what it means |
|---|---|
| `reproduced` | the network now puts ≥ 90% of its probability on the failure it is being trained toward — §3 read literally, *"until the model reproduces it"* |
| `plateau` | it has stopped getting better at that for 3 rounds running, having already passed 50% |
| `deadline` | round **n − 1**, so there is always at least one round of phase 3 left to repair with |

The floor under `plateau` is there because of §12 item 4's own warning. Without
it, two noisy rounds early on end phase 1 at p = 0.43 — a half-learned failure,
which is exactly the thing that inverts into a half-useful signal. A network
still reproducing its failure less than half the time has not finished phase 1;
it has merely stopped improving for a moment.

Every arm consults the same trigger on its own first-block loss, the controls
included, so the shape of the schedule is matched even though the round it turns
on is each arm's own. Only the inverting arms then flip. `--invert-trigger
schedule` restores the fixed `neg_fraction` date as an ablation.

## The three phases

Exactly [`TwoNRL_CartPole`](../TwoNRL_CartPole/)'s structure, carried from
control to a board game:

1. **negative** — for as many rounds as the trigger above allows, the network
   plays, fails, and is trained at the full `neg_lr`
   **toward the failures it produced**: the moves the board refused, and the
   blunders it settled for. It gets worse on purpose, and its games generate
   more failure to learn from. This is the "train on garbage" phase and the
   garbage is again generated by the network itself.
2. **invert** — `SineNet.invert()`, a logical NOT run through the network.
   Every unit is negated, every one of the 4096 move scores negates with them,
   so `argmax` becomes `argmin` and the move the failure policy most wanted is
   the one it now least wants. No training happens here.
3. **positive** — fine-tune toward Stockfish's move at `pos_lr = neg_lr / 5`,
   the activation parameters at a tenth of that, matching the research's 5:1
   ratio and `TrainConfig.lr` / `act_lr`.

<!--CURVE-->
### The curve: refusals and centipawn loss, by round

| arm | metric | r0 | r4 | r8 | r12 | r16 | r20 | r24 | r28 | r32 | r36 | r40 | r44 | r48 | r52 | r56 | r60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2nrl` | refusals/move | 114.2 | 2034.5 | 2303.2 | 2376.1 | 2406.3 | 2246.7 | 2022.8 | 2267.4 | 2440.0 | 2354.7 | 889.2 | 875.3 | 0.6 | 0.3 | 1.0 | 0.3 |
| `2nrl` | cp loss | 384 | 372 | 371 | 371 | 371 | 357 | 344 | 355 | 354 | 347 | 355 | 365 | 343 | 350 | 356 | 355 |
| `2nrl-worst` | refusals/move | 114.2 | 267.9 | 218.7 | 200.2 | 303.7 | 270.3 | 233.8 | 262.1 | 3.0 | 1.0 | 0.5 | 0.4 | 0.4 | 0.3 | 0.3 | 0.2 |
| `2nrl-worst` | cp loss | 384 | 412 | 402 | 429 | 409 | 391 | 392 | 381 | 364 | 350 | 346 | 358 | 351 | 359 | 366 | 366 |
| `2nrl-random` | refusals/move | 114.2 | 490.0 | 184.5 | 255.4 | 268.5 | 239.4 | 217.2 | 185.1 | 118.3 | 116.5 | 0.5 | 0.4 | 0.3 | 0.1 | 0.2 | 0.1 |
| `2nrl-random` | cp loss | 384 | 340 | 375 | 361 | 380 | 361 | 338 | 354 | 371 | 350 | 360 | 362 | 355 | 365 | 359 | 355 |
| `positive` | refusals/move | 114.2 | 1.7 | 1.6 | 2.4 | 2.7 | 1.8 | 0.6 | 0.6 | 1.8 | 0.6 | 2.2 | 0.7 | 0.5 | 0.6 | 0.5 | 0.3 |
| `positive` | cp loss | 384 | 345 | 340 | 350 | 343 | 344 | 346 | 340 | 312 | 354 | 353 | 328 | 342 | 344 | 359 | 345 |
| `repulsion` | refusals/move | 114.2 | 4.4 | 1.4 | 0.7 | 0.5 | 0.3 | 0.3 | 0.6 | 0.3 | 4.8 | 0.3 | 9.7 | 0.1 | 11.3 | 8.6 | 12.6 |
| `repulsion` | cp loss | 384 | 382 | 378 | 352 | 359 | 365 | 329 | 374 | 332 | 363 | 371 | 356 | 366 | 364 | 369 | 357 |
| `2nrl (no rule heads)` | refusals/move | 153.6 | 721.6 | 664.2 | 656.4 | 674.6 | 670.2 | 586.9 | 574.0 | 394.0 | 243.6 | 100.5 | 129.6 | 138.2 | 151.1 | 138.1 | 153.5 |
| `2nrl (no rule heads)` | cp loss | 367 | 382 | 383 | 340 | 373 | 367 | 345 | 344 | 342 | 334 | 343 | 357 | 378 | 341 | 365 | 373 |
| `positive (no rule heads)` | refusals/move | 153.6 | 10.7 | 6.4 | 11.7 | 11.2 | 22.0 | 23.3 | 24.3 | 28.3 | 33.1 | 35.1 | 38.2 | 43.7 | 55.4 | 64.2 | 73.5 |
| `positive (no rule heads)` | cp loss | 367 | 351 | 349 | 349 | 341 | 345 | 327 | 330 | 341 | 345 | 338 | 357 | 344 | 359 | 380 | 353 |
<!--/CURVE-->

## The arms

Every arm sees the same positions, the same candidate move sets, the same engine
budget, the same number of gradient updates and the same learning-rate schedule.
Only two things change: what the first phase trains toward, and whether the
network is inverted afterwards. `positive` is exactly `2nrl` with the negatives
replaced by positives and the inversion removed.

| arm | phase 1 trains toward | inverts | §11 arm |
|---|---|---|---|
| `2nrl` | its **own failure** — the move the board refused, or the blunder it settled for | yes | B |
| `2nrl-worst` | Stockfish's **worst legal move** — one consistent, engine-defined error mode | yes | A |
| `2nrl-random` | a **uniformly random legal move** | yes | C |
| `positive` | Stockfish's **best move** | no | D |
| `repulsion` | **away from** its own failure (unlikelihood) | no | E |

Three further arms are ablations of the method rather than of the negative set:
`--schedule per-round` inverts inside every round instead of once (§9.3's
perpetual loop), `--invert-trigger schedule` puts phase 1 back on a fixed date
so the learned trigger can be measured against one, and `--invert-mode readout`
swaps the operator for the read-out flip described above.

<!--ARMS-->
### Phase 2 on its own: one sign flip, no training

| arm | metric | before the flip | after the flip | change |
|---|---|---|---|---|
| `2nrl` | refusals per move | 2341.9 ± 416.1 | 1.7 ± 0.3 | ×1355.4 better |
| `2nrl` | legal first try | 0.003 ± 0.004 | 0.453 ± 0.028 | +0.450 |
| `2nrl` | legal AUC | 0.011 ± 0.000 | 0.989 ± 0.000 | +0.979 |
| `2nrl` | pseudo-legal AUC | 0.011 ± 0.000 | 0.989 ± 0.000 | +0.978 |
| `2nrl` | capture | 0.002 ± 0.001 | 0.998 ± 0.001 | +0.996 |
| `2nrl` | gives check | 0.016 ± 0.001 | 0.984 ± 0.001 | +0.967 |
| `2nrl` | centipawn loss | 336.9 ± 30.2 | 355.8 ± 7.8 | ×1.1 worse |
| `2nrl-worst` | refusals per move | 260.4 ± 105.0 | 3.9 ± 2.5 | ×67.5 better |
| `2nrl-worst` | legal first try | 0.011 ± 0.010 | 0.411 ± 0.090 | +0.400 |
| `2nrl-worst` | legal AUC | 0.014 ± 0.002 | 0.986 ± 0.002 | +0.972 |
| `2nrl-worst` | pseudo-legal AUC | 0.014 ± 0.002 | 0.986 ± 0.002 | +0.972 |
| `2nrl-worst` | capture | 0.004 ± 0.001 | 0.996 ± 0.001 | +0.992 |
| `2nrl-worst` | gives check | 0.015 ± 0.000 | 0.985 ± 0.000 | +0.971 |
| `2nrl-worst` | centipawn loss | 398.1 ± 2.7 | 350.0 ± 7.0 | ×1.1 better |
| `2nrl-random` | refusals per move | 247.2 ± 83.9 | 6.9 ± 7.0 | ×35.8 better |
| `2nrl-random` | legal first try | 0.019 ± 0.014 | 0.464 ± 0.055 | +0.444 |
| `2nrl-random` | legal AUC | 0.012 ± 0.003 | 0.988 ± 0.003 | +0.975 |
| `2nrl-random` | pseudo-legal AUC | 0.013 ± 0.003 | 0.987 ± 0.003 | +0.974 |
| `2nrl-random` | capture | 0.008 ± 0.005 | 0.992 ± 0.005 | +0.984 |
| `2nrl-random` | gives check | 0.015 ± 0.001 | 0.985 ± 0.001 | +0.970 |
| `2nrl-random` | centipawn loss | 352.5 ± 6.0 | 374.1 ± 12.7 | ×1.1 worse |
| `2nrl (no rule heads)` | refusals per move | 568.4 ± 17.0 | 85.3 ± 6.8 | ×6.7 better |
| `2nrl (no rule heads)` | legal first try | 0.003 ± 0.004 | 0.006 ± 0.004 | +0.003 |
| `2nrl (no rule heads)` | centipawn loss | 348.0 ± 6.5 | 329.9 ± 18.1 | ×1.1 better |

Measured negation error over every flip: **0.0e+00**.
<!--/ARMS-->

## What the network is

A **self-building** network on the sine-wave activation, `f(x) = a·sin(b(x−h)) + k`
with all four parameters learnable per neuron
([`Research/SineWaveActivationFunction.md`](../Research/SineWaveActivationFunction.md)).
It scores a *(position, move)* pair. Inputs are 775 board features — piece
planes, castling rights, whose turn, check, and whether the side to move has any
legal move — and 184 move features: from square, to square, file delta, rank
delta, and what stands on the from and to squares.

Nothing in there says a knight moves in an L or that a bishop cannot jump. The
deltas are geometry; which *(piece, delta)* pairs the board accepts is the rule
set, and that is what has to be learned.

When the training loss stalls the hidden layer grows. Growth here is
**identity-preserving** — new units enter with random incoming weights and
**zero** outgoing weights — which differs from
[`SBNN_RNN_ActivationFunction`](../SBNN_RNN_ActivationFunction/), where growth
perturbs the function. Two reasons, both specific to 2NRL: phase 1 spends its
whole budget building a representation of the failure and phase 2 negates *that*,
so a growth step must not damage it; and the inversion is exact only if every
path from input to output flips sign exactly once, which zeros survive and
random values would not. `test_sbnn.py` asserts both, at 0.0 error.

<!--GROWTH-->
### Self-building: where the network ended up

| arm | hidden layers | parameters | growth steps |
|---|---|---|---|
| `2nrl` | [56, 32] | 64,163 | 4.3 |
| `2nrl-worst` | [48, 32] | 64,163 | 4.7 |
| `2nrl-random` | [72, 32] | 74,787 | 5.3 |
| `positive` | [88, 32] | 80,099 | 6.7 |
| `repulsion` | [152, 32] | 149,155 | 14.7 |
| `2nrl (no rule heads)` | [80, 32] | 58,629 | 5.3 |
| `positive (no rule heads)` | [56, 32] | 55,973 | 6.0 |
<!--/GROWTH-->

## The inversion negates every unit, not the output

§4.3's primitive is a statement about a **unit**, and it is exact:

```
a → −a ,  k → −k        negates  a·sin(b(x−h)) + k  for every x
```

The unit's verdict flips and nothing else about it moves — not its phase `h`,
not its frequency `b`, not its bias. Inversion is that applied to every unit in
the network, which is what makes it a NOT rather than a sign on the read-out.

Composition adds exactly one piece of bookkeeping. A negated unit hands the next
layer a negated input, which would flip that layer's pre-activation and let the
odd sine undo the negation just applied; flipping that layer's weights cancels
it, so the unit still sees the `z` it saw before:

```
z′ = (−W)(−y) + bias = W·y + bias = z
```

The first layer is the whole of the exception: its input is the feature vector,
which nobody negated, so its weights stay put. Flip those too and the operation
cancels itself — that is `invert_literal()`, kept because the no-op is the
clearest way to show why the exception is there.

```
a → −a ,  k → −k                         every unit is negated
W → −W    for every layer but the first   so it still sees the same z
```

Three things hold afterwards, all asserted at 0.0 error: every pre-activation is
**unchanged**, so each unit looks at exactly the evidence it looked at before;
every unit's output is **exactly negated**, so the verdict on that evidence is
reversed throughout; and the network's output negates with them. Running it twice
is the identity, which is `¬¬P ⟹ P`.

There is a second operator that also produces `−output` exactly — flip every
weight and bias, break parity once at the last layer — and it is what
[`TwoNRL_CartPole`](../TwoNRL_CartPole/) does. It negates the *read-out*: every
hidden unit comes through it unchanged, so nothing inside the network has been
negated at all. `invert_readout()` is that one, and `--invert-mode readout` runs
it as an arm. The two agree on the function and disagree on where they leave the
parameters, which is a difference phase 2 cannot see and phase 3 can.

## Why the network scores moves and not positions

Scoring the position a move leads to — the obvious design — cannot express an
illegal move at all, because an illegal move leads to no position. That alone
settles it here.

There is a second reason and it is 2NRL's. A position-scoring player needs a
search on top, and a search has to commute with negation or the inversion stops
being order-reversing one level above where it was proved. The natural opponent
model does not commute:

```
S(m) = min_r v(leaf)        ->  S_inverted(m) = min_r(-v) = -max_r(v)  !=  -S(m)
```

The aggregators that *do* commute are the sign-symmetric ones, and the useful one
is the midrange, `(min_r v + max_r v) / 2`. Measured on 800 held-out positions
with a material evaluator at the leaves: a true minimax gives away 112.6
centipawns a move, the midrange 128.7, picking at random 289.0 and not searching
at all 283.5 — so nearly all the value of the search survives, and the involution
survives with it. That is the same shape of condition as §4.3's "the negation of
a learned structure has to be a coherent structure" and §12's odd-cycle
obstruction, one level up. `test_sbnn.py` keeps the demonstration that a minimax
opponent breaks it.

## Measurement

**The exam** is 400 positions from games between two Stockfish instances of
different strength, with *every* legal move pre-graded by `MultiPV`, so
evaluating a network costs no engine calls and every arm and seed answers the
identical questions. Four numbers, and the first two are §6.5's rules and the
last two its payoffs:

| | |
|---|---|
| `legal@1` | how often the first move proposed is legal |
| `refusals` | how many moves the board refuses before accepting one |
| `cp loss` | centipawns given away by the move finally played, clipped at 1000 |
| `agreement` | how often that move is Stockfish's own choice |
| `legal AUC` | the chance a random legal move outranks a random illegal one under the `legal` head |

`legal AUC` is the honest version of "has it learned the rules". Accuracy is
worthless here — answering "illegal" to everything scores 97% — and AUC is not
fooled by the imbalance: 0.5 is exactly no knowledge, 1.0 is the rules, and
because the inversion negates the logit, a network and its inverse score
`a` and `1 − a`. The rule exam is a **fixed** set of actions per position, drawn
once from a seed and never mined from the network being graded, so it is the
same exam for every arm.

**The benchmark** is three readings, because no single one is honest on its own:
head to head over shared openings with the colours swapped; a common opponent
(the same Stockfish) because a head-to-head between two weak players can be a
rock-paper-scissors artefact; and a random *legal* mover as the floor — a
generous floor, since it is handed the rules the networks had to learn.

The opponent is **Stockfish at Skill Level 20**, the engine with no handicap on
it at all. Skill and depth are separate knobs — skill is how well it plays the
search it does, depth is how much search it gets — and both are at the settings
in the tables. This is not a fair fight and is not meant to be one: the metric
the experiment turns on is *refusals per move*, which is a property of the
network and the rules of chess, and no opponent can affect it. What the games
measure is how long a network that has just learned the rules survives something
that has never needed to.

No engine anywhere is given a time limit, only depths, so the numbers do not move
when the machine is busy.

<!--BENCH-->
### Head to head

| match | score for the first | W–D–L | games | 95% CI | Elo |
|---|---|---|---|---|---|
| 2nrl vs positive | 0.229 | 12–9–51 | 72 | ±0.088 | -211 |
| 2nrl vs 2nrl-worst | 0.340 | 19–11–42 | 72 | ±0.097 | -115 |
| 2nrl vs 2nrl-random | 0.389 | 19–18–35 | 72 | ±0.099 | -78 |
| 2nrl vs repulsion | 0.368 | 12–29–31 | 72 | ±0.084 | -94 |
| positive vs 2nrl-worst | 0.632 | 37–17–18 | 72 | ±0.097 | +94 |
| positive vs 2nrl-random | 0.667 | 44–8–20 | 72 | ±0.103 | +120 |
| positive vs repulsion | 0.583 | 27–30–15 | 72 | ±0.088 | +58 |
| 2nrl-worst vs 2nrl-random | 0.653 | 43–8–21 | 72 | ±0.104 | +110 |
| 2nrl-worst vs repulsion | 0.507 | 28–17–27 | 72 | ±0.101 | +5 |
| 2nrl-random vs repulsion | 0.396 | 17–23–32 | 72 | ±0.090 | -74 |

### Common opponent

| player | vs Stockfish | vs a random *legal* mover | legal first try | refusals per move |
|---|---|---|---|---|
| `2nrl` | 0.000 ± 0.000 | 0.243 ± 0.099 | 0.884 | 20.3 |
| `positive` | 0.000 ± 0.000 | 0.535 ± 0.115 | 0.846 | 13.6 |
| `2nrl-worst` | 0.000 ± 0.000 | 0.368 ± 0.112 | 0.896 | 9.1 |
| `2nrl-random` | 0.000 ± 0.000 | 0.340 ± 0.107 | 0.926 | 10.1 |
| `repulsion` | 0.000 ± 0.000 | 0.465 ± 0.105 | 0.805 | 35.0 |
| `random` | 0.000 ± 0.000 | — | 1.000 | 0.0 |
<!--/BENCH-->

## Files

| file | what it is |
|---|---|
| `sbnn.py` | the network: per-neuron learnable sine activation, backprop, Adam with a separate `act_lr`, identity-preserving growth, and `invert()` |
| `moves.py` | the 4096-move action space, its features, and the six-gather fast path that scores all of them |
| `rules.py` | the rule curriculum: the seven heads, their exact labels, and the mined action sets |
| `features.py` | the board encoder, always from the mover's point of view |
| `agent.py` | the player — propose, be refused, propose again — and the training batch |
| `engine.py` | Stockfish in its two roles: the opponent that plays and the judge that grades |
| `dataset.py` | builds the held-out exam, every legal move pre-graded |
| `experiment.py` | the three phases, the arms, and the training loop |
| `benchmark.py` | head to head, common opponent, and the random-mover floor |
| `ranking_ablation.py` | the same weights ranked two ways: what the rating was actually measuring |
| `test_sbnn.py` | the correctness proofs |
| `summarize.py` | rebuilds the tables in this README from the run JSON |
| `export_viz.py` | scores the action space from the phase-boundary networks, for the page |
| `viz/` | the visualisation: one self-contained HTML file with the data inside it |
| `results/` | run logs and JSON behind every table here |

## Running it

```bash
make install     # numpy, python-chess, and Stockfish itself
make test        # the correctness proofs
make quick       # one seed, four rounds: the whole loop end to end
make data        # rebuild the held-out exam (already committed)
make run         # the headline: 2NRL against the same thing without it
make bench       # play the trained networks against each other
make viz         # build the page: viz/index.html
make all         # proofs, every arm, the benchmark, the tables, then the page
```

`make viz` captures the network at each of 2NRL's four states - `untrained`,
`phase1_end`, `after_invert`, `final` - and scores all 4096 moves from each, so
the page can put the two sides of the sign flip next to each other. Open
`viz/index.html` from a clone; it needs no server.

Every target takes overrides, as in `TwoNRL_CartPole`:

```bash
make run SEEDS="0 1 2 3 4" ROUNDS=40 JOBS=4
```

## Caveats

- **Three seeds, one task, one opponent.** The variance is reported rather than
  hidden, but three seeds is three seeds.
- **A rating below the rule heads is not a rating of a policy.** Any number in
  this repository produced by a network with tens of refusals a move is
  substantially a rating of the refusal walk, not of what the network wanted to
  play — see *And then the rating fell*. The pre-`rules.py` Elo tables are kept
  because they are what those runs did, not because they measure what their
  column headings say.
- **The network is weak in absolute terms.** It is a few tens of thousands of
  NumPy parameters trained for minutes against an engine rated well above any
  human club player. The interesting quantity is not its rating, it is the
  difference between arms trained identically apart from the inversion.
- **Under-promotion is outside the action space.** A pawn reaching the last rank
  promotes to a queen, so 4096 indices cover every move but three per promoting
  pawn.
- **Positions are not the same across arms.** Each arm plays its own games, which
  is what on-policy learning means. Engine calls, gradient updates, rates and
  candidate-set construction are matched exactly; the games themselves cannot be.
