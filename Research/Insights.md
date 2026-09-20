# Insights — index

A record of the ideas driving this repository, each with where it is specified,
whether it has been tested, and what the evidence says. Kept as an index rather
than an argument: the arguments live in the design documents, the numbers live in
`Experiments/NeuralCompression/FINDINGS.md`.

Status is one of **held** (stated, not yet tested), **confirmed** (tested, holds),
**refined** (tested, holds with a correction), or **contradicted** (tested, does
not hold as stated).

---

## The architecture

### 1. AGI is a game-solving problem
> To solve a game you must understand it. To understand it you must explore it.
> Once you understand the game, then and only then can you play the game.

The whole multi-network architecture follows from this ordering. It is enforced
mechanically rather than assumed: `GREN` produces a rule package and `GTMNN`
refuses to play one below a confidence floor.

*Where:* `GREN/DESIGN.md` §1, §22.1 · **held**

### 2. Game theory, not one big network — GTMNN
> A population of micro neural networks that play a game against each other. The
> answer is the equilibrium of that game.

*Where:* `GTMNN/DESIGN.md` · **held**

### 3. Credit belongs to game theory, not the chain rule
Backpropagation answers "who is responsible" with the derivative. Shapley (1953)
answered the same question in 1953 and proved the answer unique under four
axioms. No gradient crosses a micro boundary anywhere in GTMNN.

*Where:* `GTMNN/DESIGN.md` §12 · **held** — efficiency (`Σφ = v(N) − v(∅)`) is
exact per sampled permutation because marginal contributions telescope, which is
the suite's strongest assertion.

### 4. Everything can be gamified
*Refined.* The **encoding** is general — anything with a refusal oracle can be
probed and placed. The **learnability** is not, and the gap is six orders of
magnitude: the same table that says programming is learnable in about an hour
says a conversation would take 150 years. But *identification* needs only
`log2|G|` bits regardless of rule-set size, which collapses the intractable rows
to minutes. So everything can be *placed*; not everything can be *learned* by
probing.

*Where:* `GREN/DESIGN.md` §5 · **refined**

---

## Learning the rules

### 5. You learn the rules by failing on purpose
> When you fail at chess there are markers: no, you can't do that move. You learn
> the game by failing and guessing.

This became the central claim of GREN: **a game's identity is its refusal
boundary**, and two games are similar exactly to the degree they refuse the same
things.

*Where:* `GREN/DESIGN.md` §2 · **held**

### 6. The failure rate should be about 50%
Not a setting — a consequence. Choosing probes by expected information gain is
provably maximised, for a binary oracle, at `p(legal) = 0.5`. The most
informative probe is the one half the surviving candidates call legal. The
system reports its observed failure rate as a diagnostic: a run at 5% is
confirming what it already believes; at 95% it is thrashing outside the boundary.

*Where:* `GREN/DESIGN.md` §4 · **held** (derivation, not yet measured on a real oracle)

### 7. Programming is the game worth optimising for
> The compiler will tell you: this ain't working, bro.

*Confirmed by arithmetic.* It is the only domain where a rule set large enough to
matter meets an oracle fast enough to exhaust it: free, instant, complete, never
bored, and it ships with its own reason taxonomy already enumerated. The
diagnostic text is the highest-bandwidth channel in the system — `expected i32,
found &str` localises a type rule in one probe where a bare rejection would take
hundreds.

*Where:* `GREN/DESIGN.md` §5, §11.2 · **confirmed**

---

## Finding which game you are in

### 8. Identify the game by finding similar games
> We want to figure out which game we're playing by finding similar games, based
> on what we know about the game.

*Confirmed, and it is worth three to four orders of magnitude.* Learning Rust's
rules from nothing is ~10⁵ bits; recognising "this is Rust-like" against a corpus
of 1000 is ~10 bits — ten probes. The neighbour's rules are then inherited and
probing is spent only where the surviving candidates disagree. Most of a game's
rules are never probed at all.

*Where:* `GREN/DESIGN.md` §3 · **confirmed** (by derivation and by the
similarity numbers in §11 below)

### 9. Information clustering — chess and checkers are closer than chess and a difficult conversation
*Confirmed numerically.* Over mechanic sets: chess/checkers **0.39**,
chess/go **0.23**, chess/rust **0.03**, chess/boss-conversation **0.07**.
Correct ordering, reproduced from the axes rather than asserted.

*Where:* `GREN/DESIGN.md` §8.6 · **confirmed**

### 10. Use what we know about the game: inputs, outputs, category, goal
These four generalise better than the purely game-theoretic axes they replaced,
and they split into **declared** (free — you read the box before you play) and
**measured** (costs probes). Retrieval on declared axes alone routinely cuts a
thousand-game corpus to a handful, which is the single largest efficiency in the
system.

*Where:* `GREN/DESIGN.md` §9 · **held**

### 11. Break each game into its smallest possible parts first
*Confirmed, and it dissolved a defect rather than patching one.* Decomposing to
**mechanics** — the smallest unit comparable across domains — makes a game a
*set*, and set overlap is symmetric and order-free. The chess/go pathology (an
early axis disagreement making a single tree call them unrelated) cannot occur
over sets.

"Smallest possible" got a criterion that is measured rather than argued: **a part
is minimal when no proper sub-part is ever observed independently in the
corpus.** `CHAIN_CAPTURE` and `FORCED_CAPTURE` split, because draughts variants
exist with one and not the other. `CASTLING` does not, despite looking compound,
because it never varies in halves.

*Where:* `GREN/DESIGN.md` §8 · **confirmed**

### 12. An n-dimensional trie, with the game encoded into each path
### 13. A radix tree with the mechanics of a trie
Both halves are load-bearing and neither alone suffices. **Trie mechanics** give
terminal flags on internal nodes, so a general class can be a strict prefix of a
specific one — "certainly a two-player perfect-information capture game, which
one is still open" is a real answer whose shared rules can be played. **Radix
compression** means only the mechanics that actually discriminate cost a node.
Their interaction is one rule: *a terminal node is never merged away.*

*Where:* `GREN/DESIGN.md` §19 · **held**

### 14. Or invert the problem — a random forest
*Contradicted, honestly.* The forest existed to patch the chess/go ordering
problem. Insight 11 dissolved that problem at its source, so the forest's
original justification is gone. What survives is a weaker hedge against
frequency-ordering fragmentation; `T` dropped 64 → 16 and an open question asks
whether it should be 1 and the module deleted.

*Where:* `GREN/DESIGN.md` §20, §32.7 · **contradicted** (superseded by 11)

---

## Compression

### 15. Neural compression: split the output range, plus a selector
> One game takes 0 to 1. Two games take 0–0.5 and 0.5–1. Four games take 25%
> each. On query, expand the output back to the 0–1 range.

*Refined, and the refinement is the interesting part.* One network genuinely does
hold sixteen games at **321 parameters and 0.0081 error**, against sixteen
separate networks at **1040 parameters and 0.0101 error** — fewer parameters
*and* better accuracy. **But the output partition is not what compresses.** Plain
conditioning is 7.5× more accurate at the same scale, and the slopes run
opposite: as games are added the partition degrades while conditioning improves.

**What the partition actually wins is games it has never played.** Trained on
even-indexed games only, one-hot conditioning memorises the seen ones at 0.0045
and collapses to 0.1185 on unseen (26×) — it *structurally cannot* do better,
since an unseen game has no slot to put a 1 in. The continuous selector
coordinate degrades 8%, and the partitioned version does not degrade at all.

So the value is not compression. It is that **a continuous game coordinate lets
you play a game you have never played, by placing it between games you have.**
For an AGI architecture that is the more important of the two.

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §1–4 · **refined**

### 16. Order the partition by similarity
*Confirmed, and it is mandatory rather than a nicety.* Absent at N=2, 2.5× at
N=8, and at N=16 it is the difference between working and **numerically
diverging** (9.7× with sine). Ordered, the target is a smooth function of the
selector coordinate; shuffled, it is a high-frequency sawtooth a small network
cannot fit.

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §3 · **confirmed**

### 17. Avoid dust — binary doesn't work because of 16 and 32
*Confirmed, and the mechanism is worse than waste.* **Dust is error
amplification.** Reading a band back out multiplies output error by the slot
count, so reserving slots for games that are not there degrades the games that
are: 50% dust costs **3.1×**, 75% costs **4.5×**. Padding 17 games into 32 slots
is roughly a 3× penalty for nothing.

The static-sequence idea it was meant to enable turns out not to be needed:
re-laying every band out costs **1.37×** and freezing them costs **1.36×** —
indistinguishable, because the selector is continuous and rescaling it is a
smooth reparameterisation rather than a permutation. **3.1× against 1.37×, both
pointing the same way: take the relayout, refuse the dust.**

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §6–7 · **confirmed**

### 18. Same input and output size for every game, plus a game modifier
*Held, with a concrete design.* Score **one candidate at a time**, and action
space size never touches the geometry — 50 moves, 361, or infinitely many Rust
programs all work. The modifier is the mechanic set hashed into 128 dimensions,
and because hashing a *set* preserves overlap, similar games land at nearby
points automatically. **Transfer is a consequence of the encoding rather than a
mechanism added on top.** Width chosen by measurement: 0.177 mean error at 32
dimensions, 0.061 at 128.

*Where:* `GTMNN/DESIGN.md` via `GREN/DESIGN.md` §22.4–22.5 · **held**

### 19. Train validity first, then grade the move separately
> Focus on training on valid/invalid moves, then a separate grade/reward for the
> move itself. This lets us compress go, chess and checkers into the same space.

*Held, and it is the load-bearing half of 18.* Validity is binary, instant,
abundant and **transferable** — legality is structure, and the modifier says
which structure. Grade is sparse, delayed and game-specific. Training them
jointly would let the noisy signal contaminate the exact one. It also makes
"understand the game, then play it" true *inside a single micro's weights*, not
only at the system level.

*Where:* `GREN/DESIGN.md` §22.6 · **held** (the decisive test is stated: train
`p_valid` on chess only, measure it on checkers with the checkers modifier and
no further training)

---

## The vanishing gradient

### 20. The vanishing gradient is a feature, not a bug
*Contradicted in this architecture, specifically.* The sine's derivative near zero
is `−cos(b·z)·b ≈ −b`, so depth multiplies the gradient by `b^depth`. At depth 4
with `b = 1/3` the layer-0 gradient is `7.6e-05` and test MSE `0.0858`; at `b=1`
they are `1.7e-03` and `0.00016`. **22× the gradient, 536× the accuracy.** At
`b=1` the gradient does not vanish at all and depth 4 becomes the *best* result —
better than depth 1 or 2, which is what depth is supposed to buy.

The general case for designing around vanishing gradients may hold elsewhere.
Here the gradient vanishes because every neuron was initialised into the linear
region of its activation, and it stops when that is corrected. See 24.

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §10 · **contradicted**
(as a property to accept; the observation that it occurs was correct)

### 21. Invert the gradient when the vanishing threshold is reached
*Refined — the trigger was wrong, the mechanism is marginal.* Triggering on
**gradient magnitude** is catastrophic (243–520× worse on healthy networks): a
small gradient means stuck *or* converged *or* still starting, and magnitude
cannot separate them. The correct signal is a **plateau** — past a warmup, the
running loss has not improved for a window. That is harmless where the network is
healthy and gives **1.24×** where the gradient genuinely vanished.

But 1.24× recovers a condition that fixing `b` removes 536× of. *Build the
trigger if the stall is real; do not build it instead of fixing the stall.*

The stated form — *all weights below X* — is the better of the two magnitude
conditions, because a weight that never moved off its initialisation is
unambiguous in a way a small gradient is not.

*Where:* `Experiments/NeuralCompression/vanishing.py`, FINDINGS §11 · **refined**

### 22. Dual network: train in one, query the other — the REM analogy
*Confirmed, and the analogy predicted the better implementation before either was
run.* **Replay beats grafting.** Grafted weights arrive in a network with an
input the specialist never had and an output scaled to a band — wrong coordinate
system, which fine-tuning must repair. Replay transfers the *function* rather
than the *parameters*, so nothing has to match. That is also what the
neuroscience describes: consolidation in sleep is replay-driven, the hippocampus
regenerating experience for the neocortex, not synapses copied between
structures.

Two corrections to the magnitude. Replay's apparent 1.57× is mostly compute; at
matched budget it is **1.05×**. And grafting's real value is not accuracy but
**compute** — 0.0542 using 38,400 query-network updates, a third of what joint
training needs to reach a worse 0.0626. Specialists train independently and in
parallel; the shared network, the contended resource, only fine-tunes.

What neither fixes: specialists alone reach 0.0081 where the best shared network
reaches 0.0398. **Compression costs about 5× and no consolidation recovers it.**

*Where:* `Experiments/NeuralCompression/consolidate.py`, FINDINGS §12 ·
**confirmed**

### 23. Rotating inversion — invert alternating layers each cycle
*Safe, and for the same reason useless — tested.* With an odd activation
(`−sin(bz)` is odd), negating a layer's weights is a genuine **symmetry
operation**, and the parity alternation has **period 4**: odd, even, odd, even
returns every weight to its original value exactly (verified). That is a cycle
through the network's own symmetry orbit, and it connects directly to
`CyclesAreAFeature.md`.

It behaves completely differently from every gradient-ascent method: **0.58–1.04×
where ascent-based inversion is 0.00×** (100–1000× worse). Negating weights
relocates the network without unlearning; ascent destroys the learned function.

But it does not help, on either a smooth task or a deliberately rugged one with
many basins — **1.00×, exactly neutral.** The reason is not that it preserves the
loss: measured across four negations, the loss **jumps ~75×** (0.0034 → 0.25) and
only returns at cycle 4, when the weights do. Each hop is a large perturbation,
not a free move along a level set. What makes it neutral is the **closure** — the
network is kicked out, partly repaired by training, kicked again, and returns
exactly to where it began. *A cycle that returns to its origin does no work.*

Which says what would have to change for it to pay off: **break the closure.**
Period 4 comes from two parities over an involution. Three phases, an asymmetric
subset per cycle, or pairing the negation with something non-involutive would
make the excursion open, and the network would explore instead of returning.
Untested, and the obvious next thing to try.

*Where:* `Experiments/NeuralCompression/vanishing.py`, FINDINGS §15 · **refined**

---

## Activation

### 24. The sine wave activation instead of a sigmoid
*Confirmed as an activation, contradicted at the stated frequency.* Sweeping `b`
in `f(x) = a·sin(b(x−h)) + k` on a task whose target **is** a sine of a linear
combination — the case most favourable to it:

| `b` | \|z\| to first peak | test MSE |
|---|---|---|
| **1/3 (as written)** | 4.71 | **0.01674** |
| 1.0 | 1.57 | **0.00015** |
| 2.0 | 0.79 | **0.00004** |
| *tanh reference* | — | *0.00098* |

**At `b = 1/3` the sine is 17× worse than tanh. At `b = 1–2` it is 6–24×
better.** The activation is excellent; the frequency is wrong. Reaching the first
peak needs `|z| = π/(2b)`, which at `b = 1/3` is 4.71 — far outside the operating
range once features are L2-normalised, so every neuron sits in the sine's
**linear** region and the network is linear however wide it is.

Learnability does not rescue it: `∂f/∂b = a(x−h)·cos(b(x−h))` is small exactly
when `x` is small, so the gradient that would fix the frequency is suppressed by
the same condition that makes it wrong. **The initialisation is the trap, not the
parameterisation.** The rule is `b ≈ π / (2·E|z|)`.

*And in the other direction — as a **gate** it fails, for the same reason.*
`FilterBankRadix/` reads the wave for its *sign* and uses it to route: a hinge
raises a unit's response by pushing its projection, and the address is the sign
pattern. Held at `b = 1/3` with `a, b, h, k` frozen, that push carries the
wave's argument `|u| = |b(x−h)|` from a median of 0.28 to **100.1** — 98.3% of
units end up past their first peak, where a sine is coming *back down* — and the
filter scores **0.245 against a chance of 0.25**: it cannot be fitted at all. A
tanh under the identical rule ends up past its own first peak just as often
(73.8%) and does not care, because saturation preserves an ordering and
periodicity does not; it is the best router measured (0.736). With the wave
learnable the sine recovers to 0.697, and the parameter that recovers it is `b`,
which falls from 0.333 to **0.177** — *learning the wave, here, means flattening
it*. Same frequency, two opposite failures: too linear for a deep network, too
periodic for a gate.

*Where:* `Research/SineWaveActivationFunction.md`, FINDINGS §5,
`FilterBankRadix/README.md` · **refined**

### 25. Cycles are a feature, not a bug
Reached independently from the other direction by the game theory. At inference
GTMNN's game is *player-specific*, and Milchtaich (1996) says such games always
have a pure equilibrium but best-response paths may cycle. So the cycle is not the
solver failing — it is a property of the game, iterating harder does not help, and
the correct response is to leave and escalate to metacognition.

*Where:* `Research/CyclesAreAFeature.md`, `GTMNN/DESIGN.md` §2.1, §14 · **held**

---

## Cortical organisation

### 26. A Single Self-Building Neural Network instead of micro networks
> Inputs and outputs grow to match the games.

One SBNN per *region* rather than thousands of micros. Growth is additive and
**non-destructive**: new hidden units enter with zero outgoing weight and new
inputs with zero incoming weight, so the network computes bit-identically the
instant after growth and only changes as the new capacity learns. Growth on a
**plateau**, which the existing
`Experiments/SBNN_RNN_ActivationFunction/main.py` already does and which
insight 21 independently confirmed is the only trigger that separates *stuck*
from *converged* from *still starting*.

*Where:* `CyclicCortex/DESIGN.md` §8 · **held**

### 27. Similar games in similar regions — brain structure
The similarity graph partitions into regions by modularity; each region owns one
network; a new game lands next to what it resembles or founds a new region if
nothing is close enough. This changes what carries game identity: **it moves out
of the input vector and into the graph position**, which is why the region's
network does not need to be told which game it is playing.

*Where:* `CyclicCortex/DESIGN.md` §6, §9 · **held**

### 28. A cyclic graph where distance is similarity
*Confirmed.* Jaccard distance over mechanic sets is a **proper metric** — checked
over 210 ordered triples with **zero triangle-inequality violations** — so
"distance is similarity" has a consistent geometry rather than being a figure of
speech.

And it must be **cyclic**: chess/checkers 0.609, checkers/go ~0.730, chess/go
0.769 form a triangle in which none is the parent. A tree must break one edge.
This makes the relationship to insight 13 precise: **the radix tree is a
projection of this graph, not a rival to it** — the tree makes retrieval
sublinear, the graph is what is actually true, and where they disagree the graph
is right.

*Where:* `CyclicCortex/DESIGN.md` §5 · **confirmed**

### 29. Common denominators — generalised, minified inputs
> For chess, a piece moving one space would count here. This allows us to train
> the same network on checkers as well.

*Confirmed, and more strongly than stated.* Training on king moves then testing
on checkers, zero-shot: **0.646** on a generalised vocabulary against **0.514 —
exactly chance —** on a raw board encoding. Three findings beyond the headline:

* **Raw pretraining is actively harmful** (−0.220 at k=5). A board-specific encoding makes the network *worse* at the second game than starting from nothing, because square indices mean different things in different games. This is the failure mode the generalised vocabulary exists to prevent, and it is not hypothetical.
* **It is not an artefact of good feature design.** Stripping the features that encode the checkers rule (`is_diagonal`, `forwardness`) left zero-shot **identical** at 0.646 and made the transfer *gain* larger. The advantage is the shared coordinate system, not the chosen features.
* **Three inputs are enough.** `(dx, dy, target_empty)` matches eight hand-designed features and beats seventy-two. Minification is not a compromise for capacity — the minimal shared vocabulary is the *best* one, because everything removed was game-specific and therefore untransferable.

The common denominators are GREN's mechanics (insight 11) promoted from a
similarity signature into the network's actual input vocabulary: **the objects
that decide two games are alike are the objects the network reads.**

*Where:* `CyclicCortex/common_denominator.py`, `CyclicCortex/DESIGN.md` §4 · **confirmed**

---

## What to test next

Ordered by how much they would change, per unit of effort:

0. **Does transfer survive ACROSS regions, or only within them?** (29) The measured transfer was between two grid games sharing displacement semantics — within a region, which is what regions are for. If nothing transfers between a board region and a programming region, regions are independent networks with a clustering algorithm attached, and the graph is only doing routing. Cheap to run on the existing corpus.
1. **Does validity transfer?** (19) Train `p_valid` on chess only; measure it on checkers with the checkers modifier and no further training. If near chance, the modifier is not carrying what 18 claims and the composition is decorative. One day's work, falsifies the central claim.
2. **Fix `b` everywhere and re-measure.** (24, 20) Upstream of everything in the vanishing-gradient work. `RadixCyclicNN` carries the same default — measure its realised `E|z|` before changing it.
3. **Does the population actually specialise?** (2) GTMNN rests on the congestion game doing what it claims; `gini` and `diversity` are instrumented from the first commit so it can be falsified early.
4. **NCD against mechanic Jaccard.** (11) Where a feature-free similarity disagrees with the mechanic one, the vocabulary is blind — the only automatic check on the part of GREN hardest to verify.
5. **`universal_fraction`.** (18) The share of micros that are game-blind and transfer everywhere. Nothing predicts it; whether evolution finds a stable value is the sharpest test of whether GREN and GTMNN compose.
