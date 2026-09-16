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

*Implemented and measured.* The machinery all works and every game-theoretic
claim it rests on is asserted numerically (`GTMNN/tests/`, 36 tests). What does
not hold is the part the whole design is for: **the population barely learns to
predict.** Belief loss after the full auction / equilibrium / Shapley / REINFORCE
path is 2.84 against a uniform baseline of 2.71 — still worse than guessing —
while the *identical* micros given a direct supervised signal reach **1.50**. The
architecture has ample capacity; the credit path is what fails to deliver a
signal that sharpens the population.

More training does not close it (6 900 steps per micro moved belief loss 3.226 →
3.204, and 150 epochs did worse than 20), so it is not a signal shortage. Two
candidate causes were tested and **both were wrong**: the `B`/`λ` ratio (swept
2 → ∞; abstention falls 56% → 0% and the loss never improves) and the
null-player axiom forbidding a reward for correct silence (deliberately breaking
it made things *worse*, 2.838 → 3.050).

The population does specialise, which was the other thing in doubt: `gini` rises
to 0.54 when every micro is seated against 0.05 when 24 of 256 are. That makes
seat count the first thing to look at.

*Where:* `GTMNN/DESIGN.md`, `GTMNN/README.md` · **contradicted** (as a route to
prediction; the mechanism is sound and the credit is exact)

### 3. Credit belongs to game theory, not the chain rule
Backpropagation answers "who is responsible" with the derivative. Shapley (1953)
answered the same question in 1953 and proved the answer unique under four
axioms. No gradient crosses a micro boundary anywhere in GTMNN.

*Confirmed, and more exactly than stated.* Efficiency `Σφ = v(N) − v(∅)` holds to
**3e-15** over 200 games at 1, 2, 7 and 32 permutations — it is exact *per
sample*, not in expectation, because the marginal contributions telescope along a
single permutation. The null player gets **exactly** `0.0` and takes exactly no
step; two interchangeable seats get identical credit to `1e-12` when enumerated;
and over all `M!` permutations the incremental estimator equals the closed-form
subset-weighted sum to `4e-16`, so any gap at finite samples is variance falling
as `1/√n` rather than bias.

Also confirmed: no gradient crosses a micro (checksummed around every `learn`),
and all 45 parameters of a micro — weights, biases and the four sine parameters —
match finite differences to `1e-10`.

The credit is therefore not the reason 2 fails. It is exactly what it claims to
be, and the population still does not learn to predict — which makes the failure
more interesting, not less.

*Where:* `GTMNN/DESIGN.md` §12, `GTMNN/gtmnn/shapley.py` · **confirmed**

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
*Contradicted — no trigger tested helps, and most of them are catastrophic.*
Three detectors were built: gradient magnitude, weight magnitude, and a loss
**plateau**. Across three seeds and two healthy activations plus one genuinely
stalled one, **the best result any of them achieves is 1.00× — no effect at
all** — and on a healthy network every policy that fires does damage, by 1.8× to
520×. A single inversion is enough: the weight trigger fired *once*, on one seed,
and took that seed from 0.00016 to 0.0961.

Triggering on **gradient magnitude** is the worst of the three (243–520× worse on
healthy networks), because a small gradient means stuck *or* converged *or* still
starting and magnitude cannot separate them. A **plateau** is the better
*detector* — 6 fires against the gradient trigger's 132 on healthy tanh — but
detecting the stall correctly does not make climbing out of it work.

**An earlier 1.24× for plateau-with-long-bursts is withdrawn: it does not
reproduce.** Sweeping burst length on the stalled network gives 1.00×, 0.98×,
0.88×, then divergence — monotonically worse. Gating on "the loss is still bad"
does make the policy safe, by reducing it to **zero fires**.

Keep the plateau detector for deciding when to **grow** a network
(`CyclicCortex/DESIGN.md` §8 uses it for exactly that). Do not build the
inversion it was meant to trigger. Fixing `b` is worth 536× on the same stall.

*Where:* `Experiments/NeuralCompression/vanishing.py`, FINDINGS §11 · **contradicted**

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

*Where:* `Research/SineWaveActivationFunction.md`, FINDINGS §5 · **refined**

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

*Refined at cortex scale.* The purpose-built experiment above shares *all three*
of its inputs. A real region does not: region 0 gives chess and checkers 8
shared slots out of 33, and the other 25 are private. Measured there — train
chess alone, then evaluate checkers, against checkers' **majority class** —
transfer is **+0.023**, not +0.29. The untrained network is the wrong baseline:
with *nothing* shared at all, checkers still reaches its majority class, because
training chess moves the region's shared hidden layer and output bias and that
needs no transfer whatsoever. I made that mistake first and it inflated the
figure twelve-fold.

This does not contradict the headline; it is what the headline's third bullet
predicts. Everything removed was game-specific and untransferable, so a
vocabulary that is 25/33 game-specific transfers about as well as its private
part allows. The way to get the 0.646 result inside a region is to make the
shared vocabulary *most* of the vocabulary, not to add more of it.

*Where:* `CyclicCortex/common_denominator.py`, `CyclicCortex/README.md`,
`CyclicCortex/DESIGN.md` §4 · **refined**

### 30. A shared refusal code is not a shared rule
Not an insight of Curtis's but one the architecture forced, and the sharpest
limit found so far on identification-by-refusal (11, 12).

GREN can derive CyclicCortex's *input* vocabulary as well as its mechanics:
`gren/vocabulary.py` asks which feature dimensions separate the moves a game
refuses with code C from the moves it accepts, then matches dimensions **across**
games by their whole legality signature, so a shared slot holds one quantity
rather than one name. It works — chess's and checkers' `OCCUPANCY[0]` are matched
to each other, go's and sudoku's `GRID_PLACE[2]` are matched with a sign flip,
and sudoku's `CONSTRAINT_UNIQUE`, which I wrote as one 3-wide block, is separated
into row, column and box without being told there were three.

And it makes chess-to-checkers transfer **catastrophically worse**: −0.409
against the majority baseline, where the hand-written vocabulary gets +0.023.

The alignment is not the fault. It reproduces my own hand-written
occupancy-only ablation to within noise (−0.409 against −0.432) — which is what
a correct measurement of a bad idea looks like. Chess and checkers both refuse
`OCCUPIED_TARGET`, but chess refuses only a target holding your **own** piece,
because taking an enemy piece is a capture, while checkers refuses **any**
occupied target for a simple move. Same code, opposite rule. Sharing a weight
across them teaches chess's exception into checkers. The hand-written vocabulary
escapes this only by *diluting* those three slots with five harmless ones:
`GRID_MOVE` sharing alone contributes exactly 0.000.

**Refusals identify a rule's shape, not its arguments.** That is enough to place
a game on the map and not enough to share a weight — which is why the mechanics
swap is the default and the derived vocabulary is opt-in.

*Where:* `GREN/gren/vocabulary.py`, `CyclicCortex/cortex/discovered.py`,
`CyclicCortex/README.md` · **contradicted** (as a route to transfer;
the alignment itself is confirmed)

### 31. Identification and characterisation are different questions
Also forced rather than proposed. GREN gated a `GamePackage` on one confidence
number, and handing packages to CyclicCortex made the conflation visible: sudoku
scored **0.350** and was refused.

Sudoku scores badly because it sits 0.80 away from everything GREN has seen. That
is *confidently novel*, not uncertain — and a novel game is precisely the one
that should found its own region. Gating placement on identification confidence
refuses a correctly understood game for the crime of being new.

So the number splits in two. **Identification** — which known game is this —
falls when nothing is close. **Characterisation** — do we know what this game is
like — rises with probe coverage and a settled refusal taxonomy, and is the right
gate for placement. Sudoku: 0.350 and 0.977.

A confidence measure that answers two questions at once will be wrong about one
of them, and the integration is what exposed which.

*Where:* `GREN/gren/package.py` (`placeable`), `GREN/gren/axis.py` (`settled`),
`CyclicCortex/cortex/discovered.py` · **refined**

### 32. A game modifier is a game-IDENTITY signal, and identity is the opposite of transfer
`GREN/DESIGN.md` §22.5 argues that hashing a game's *mechanic set* into the micro's
input makes transfer automatic: hashing a set preserves overlap, so two similar
games land at nearby points and "a micro that learned 'a piece cannot move through
an occupant' generalises to every game with a similar modifier without being told
to".

**The first half is confirmed exactly.** Modifier cosine tracks the Ochiai
similarity of the mechanic sets with mean absolute error 0.102 at 64 dimensions,
**0.031 at 128** and 0.017 at 256 — matching the spec's own sizing table, and
putting chess nearest checkers and sudoku furthest, which is the same ordering
GREN found by probing and CyclicCortex uses to build its regions. Three projects
now agree on what a game *is*, through one JSON file rather than three
hand-written frozensets.

**The second half is contradicted.** Train on chess, evaluate checkers cold, six
seeds: with the modifier on, 0.503; with the modifier block zeroed, **0.566**.
The modifier is worth **−0.062**, negative in 5 of 6 seeds. One seed had shown
+0.117 and that was noise.

The reading that fits: the modifier tells the population *which game this is*, and
game identity is precisely what lets a population specialise per game — the
opposite of carrying a response across. Similarity in modifier space is real; it
is just dominated by the identity signal sitting in the same bits.

This is the third independent measurement in this repository putting
chess→checkers transfer at approximately nothing: CyclicCortex's shared
vocabulary (+0.023 against its proper baseline), its cross-region ensemble
(+0.025), and now this. Three different mechanisms, three different codebases,
one answer.

*Where:* `GTMNN/gtmnn/games.py`, `GTMNN/README.md`, `GREN/DESIGN.md` §22.5 ·
**refined** (the hash works; the transfer claim does not)

### 33. A training signal that contains the answer is not a loss
Not an insight of Curtis's — a mistake I made and had to measure my way out of,
recorded so it is not repeated.

GTMNN's epoch record reports `loss` as mean `−log P(y)`, per `DESIGN.md` §16.2,
where `P` is the aggregate of the training-game equilibrium. But that game is
`CorrectnessCongestion`, and **its payoff function contains `y`** — every seated
micro is paid `B/n_a` for naming the answer it was just shown. The resulting
number looks excellent and measures almost nothing: `loss` **0.59** while the
population's actual predictive loss was **4.2**, against a uniform baseline of
**2.77**. I read the first as the second for several rounds.

It is not a useless quantity — it is the right partner for `shapley_total`, since
the two are views of `v(N) − v(∅)`. It is just not a prediction. The fix is to
report both, side by side, so the contaminated one can never be mistaken for the
honest one again.

The general form: **when a metric is computed from a mechanism that was given the
label, it measures the mechanism's compliance, not the model's knowledge.** The
control that catches it is cheap — withhold the label and recompute.

*Where:* `GTMNN/gtmnn/model.py` (`belief_loss`), `GTMNN/README.md` · **confirmed**

### 34. A trie over what is PROVABLE, not over what a thing is
The radix tree in GREN indexes games by what they *refuse*. The same mechanic one
layer up indexes them by their game-theoretic structure — and there the levels
have a meaning the refusal tree does not have: **descending is accumulating
premises.**

The root does nothing. It asserts no property and guarantees only Nash 1950, true
of every finite game. Each level adds one fact, ordered so that every step
strictly narrows, and a node's guarantees are then a function of its path and
nothing else. The theorems fire at different depths: von Neumann at depth 2
(`players=2`, `payoff=opposed`), Milchtaich at depth 4 the moment monotone load
is asserted, Rosenthal only at depth 5 with the whole congestion prefix. **The
path is the proof**, which is why it is a trie and not a lookup table — an
internal node is a real answer you can act on before reaching a leaf.

Two things follow that a diagram would not have given:

* **The recommendation is falsifiable, and mostly holds.** Running all five
  solvers on every class, the trie's pick measures best in **5 of 6**. The
  exception is instructive rather than embarrassing: on the player-specific
  congestion game the trie says `regret_plus` because Milchtaich warns a
  best-response path *may* cycle, and `best_response` measures better. Both are
  right — one is a worst-case guarantee, the other an average case. Hunting the
  cycle across seven geometries and 1 400 games **found none**. The trie's job is
  to say which choice is provably safe, not to predict the mean.
* **The solver stopped being a configuration option.** `solver="auto"` classifies
  the payoff in hand and takes the deepest rule whose premises the path satisfies.
  Measured: 1.8× faster at a loss inside the seed spread. A new payoff now gets
  the right solver by answering the same questions, rather than by someone
  remembering to add a branch.

The general form: **an index whose levels are premises turns a classification
into a derivation.** GREN's tree answers "which game is this"; this one answers
"what am I allowed to assume", and the second is the one that changes code.

**A network at every node.** Reaching a node is how you find the population to
query, and a cold leaf backs off to the nearest trained ancestor — which is what
the terminal-flag-on-internal-nodes design is FOR. "A congestion game with a
common payoff" is a usable answer while the specific class below it is still
cold. The routing is verified in every seed: `resolve` takes the deepest trained
node, backs off when the leaf is cold, stops the moment it has played. What
backoff is *worth* is **+0.006 over 5 seeds with a ±0.15 swing** — nothing. One
seed had shown +0.22 and it was noise, the same trap as 32. The class it backs
off to has barely learned either, so this is downstream of insight 2 and is worth
re-measuring only once that is fixed.

**Scope, deliberately narrow.** A trie here indexes games and nothing else: GREN's
by what a game refuses, this one by its structure. Neither carries activations or
stands in for a network. The moment one does, it stops being an index and the
guarantees stop meaning anything.

*Where:* `GTMNN/gtmnn/trie.py`, `GTMNN/gtmnn/tournament.py`, `GTMNN/README.md` ·
**confirmed** (the index and the solver derivation; the backoff benefit is not
yet measurable)

### 35. Measure the mixed profile, not its argmax
Another mistake caught by its own control, recorded so it is not repeated.

Exploitability — max over players of (best-response utility − realised utility) —
is the honest convergence measure, zero exactly at Nash. I computed it on the
profile's **argmax**, which is meaningless wherever the equilibrium is mixed. On
rock-paper-scissors *every* pure profile is exploitable (1.0 when the two match,
2.0 when they do not), so the measure scored all five solvers identically at
their worst and hid the thing that matters: the uniform mixture is exactly
unexploitable, and the solvers differ enormously in how close they get to it
(fictitious 0.19, best response 2.00).

The tell was a column of identical bad numbers. A metric that cannot separate
five methods on the one game built specifically to separate them is measuring
the wrong object.

*Where:* `GTMNN/gtmnn/equilibrium.py` (`mixed_exploitability`) · **confirmed**

### 36. Memory is the trajectory, not the answer
> Solve memory by keeping track of the trajectory through the model, input and
> output data, then storing this in a new NN that is a stack-based idea where the
> top of the stack has precedence compared to similar paths further down.

Every network here discards the most informative thing it produces. GREN walks a
radix path to identify a game and keeps the answer. CyclicCortex routes to a
region and keeps the prediction. GTMNN seats 64 micros of 4096, solves an
equilibrium, and throws away *who was seated and what they argued*. The
trajectory is that discarded object — the path taken, not the destination — and
the claim is that it is what memory should be keyed on.

**The precedence order is the sharp part.** GREN's radix tree resolves a tie by
SPECIFICITY: a deeper terminal beats a shallower one, because "chess" beats "a
two-player board game" when both match. A stack resolves it by RECENCY: the top
beats everything below, whatever its specificity. Those are two different orders
over the same set of matching paths and they disagree — the specific-but-stale
entry and the general-but-fresh one are both real answers. Which should win is
empirical, and both policies can run over the same stack, so it is settleable.

A stack also buys something no weight-based memory has: **shadowing without
erasure**, exactly like a scope chain. Popping restores what was underneath, so
memory gets an undo.

Two existing threads meet here. `CyclicCortex::rehearse` exists because admitting
a new game overwrites the incumbents, and replay is the mitigation — a trajectory
store attacks the same problem from the other side and `rehearse` is the measured
baseline to beat. And insight 22's dual-network/REM consolidation has never had a
rule for *what* to consolidate; a stack supplies one, in that entries which keep
getting shadowed are what the base network should absorb, and entries that keep
being hit alone are what it still cannot represent.

**The premise is falsifiable and cheap to falsify, so that goes first.** If a
trajectory-keyed lookup does not beat an ordinary input-keyed one on the same
episodes, the trajectory carries no structure the input lacks and the whole
design reduces to a cache with extra steps. Given that "similar things
generalise" has now failed three independent measurements in this repository
(29, 30, 32), the experiment should be built to fail loudly.

Unresolved and load-bearing: what a trajectory IS across three networks with
three different path alphabets; what "similar" means for two paths (the phrase
"similar paths further down" is doing enormous work); one global stack or a stack
per prefix; and growth, since a stack that only grows is a log rather than a
memory. Also a scope collision — a path-keyed store is trie-shaped, and a trie
here is for game theory and game classification only. Either this uses a
different structure or that rule needs a stated exception; it should not quietly
become a third trie.

*Where:* `Memory/README.md` · **held**

### 37. Index a game by its END GOAL, and work backwards
> The Radix Tree for game solving should be about the end goal. We focus on the
> end goal of the game, and work backwards. For chess, it's to topple the king,
> for a conversation at work, it's determined by the conversation itself and can
> be a bit more ambiguous, however, we usually have an end goal in mind for the
> conversation (to learn more, get a promotion, etc...). If the goal is unclear,
> go up a level.

This corrects a proposal of mine. Asked how to make the trie more specific, I
offered more game-theoretic structure — information, horizon, dominance,
supermodularity. Every one of those is a real axis and every one is **the wrong
kind**: they require already knowing the game. You cannot state a conversation's
horizon or potential function before having it. Measured, that is exactly what
went wrong — a conversation with *every* structural fact supplied still stalled
at depth 1 of 9, because no branch existed for it.

The goal is the one thing you can state about an unfamiliar game **before you can
play it**. Same conversation, indexed by goal instead: depth 3 of 3, fully
placed, sitting next to the other conversations rather than nowhere.

**Going up a level is the semantics, not a fallback.** Each level is a usable
answer in its own right:

| what I know | depth | who it sits with |
|---|---|---|
| "get a promotion" | 3 | the 1:1 with my manager |
| "change how they see me" | 2 | every conversation with that shape |
| "talk to my manager" | 1 | everything |

"Change how they see me" is a real goal with real strategies even when "get a
promotion" is not yet committed to. That is the same trie mechanic as GREN's
terminal-on-internal-node rule, arrived at from the other direction: a general
class is a strict prefix of a specific one and is answerable on its own.

Two arguments for the ordering, and the second is the load-bearing one:

* **Compression.** `gren/cli.py` inserted `sorted(signature())`, so the tree's
  root discriminated on `adversarial=` for no better reason than that "a" sorts
  first — which separates nothing among four board-ish games. Goal-first
  identifies a game from **one** token against alphabetical's **2.5**, and the
  root branches four ways instead of two. *Small N*: four games and one
  perfectly-discriminating axis makes this nearly free, and `goal_type` would not
  stay a unique key at four hundred games.
* **Knowability.** Putting the goal at the root means the questions the tree asks
  first are the ones a new game can actually answer. This does not depend on the
  corpus size, and it is why the first argument is the weaker one.

"Work backwards" is also the traversal, not just the layout: from the goal, to
what must be true for it, to what must be true for *that*. Chess endgame
tablebases are built exactly this way, backwards from mate. It is goal regression,
and it is the natural home for `GREN/DESIGN.md` §8.3's STRIPS operators, still
unimplemented.

One honest limit: chess and checkers still share 2 of 3 goal axes
(`against=opponent`, `decided_by=terminal-state`) and diverge only at the leaf —
`reach-target` against `outlast`. The goal axis separates them where mechanic
similarity does not separate them at all, which is the right direction given that
chess→checkers transfer has measured at approximately nothing three ways
(29, 30, 32), but it is a modest separation rather than a dramatic one.

*Where:* `GREN/gren/radix.py` (`goal_first`), `GREN/gren/cli.py` · **confirmed**

### 38. A refusal code is a precondition violation
The follow-through on 37. If the tree is organised by the end goal, the way to
use it is to work backwards — and goal regression is the classical method for
that: ask what would have to be true for the goal to hold, and keep asking until
the answer is already true.

Regression's usual cost is writing every operator's preconditions by hand, and
that hand is where domain knowledge smuggles itself in. GREN does not need them
written. `BLOCKED_PATH` means `move` requires a clear path; `CONSTRAINT_ROW`
means `place` requires the value absent from the row. **`why()` already answers
"what blocks this action?", which is exactly the question regression asks at
every step.** The preconditions are measured, not authored, and `DESIGN.md`
§8.3's STRIPS operators finally have a source.

On sudoku this reproduces constraint propagation from nothing but the refusal
codes: rank cells by how few values clear all four preconditions, and a cell with
one clearing value is forced. Nobody coded "naked single". Search nodes fall
27×, 143× and **1 240×** at 30, 26 and 22 clues — though by raw work it barely
wins on easy puzzles, because scanning 51 cells to save a few nodes is not worth
it when almost any order solves them.

**The law, and it is quantitative: a subgoal prunes exactly what it excludes.**
On chess, `mate ⟸ check ∧ no-escape`, where the expensive conjunct enumerates
every opponent reply and the cheap one is a single `attacked()` call. In positions
from real play 4.3% of moves give check, 95.7% of the expensive tests are avoided,
and it runs 7.7× faster. With pieces scattered at random 59.5% give check, 48.7%
is avoided, and it runs 2.2× faster. Regression pays in proportion to how
selective the cheap subgoal is, and nothing else.

*Where:* `GREN/gren/regress.py`, `GREN/README.md` · **confirmed**

### 39. Check whether the baseline already does it
The first measurement of 38 came out at **exactly 1.0×**, which is the number
that should always be suspicious.

The control had regressed all along: `is_mate()` is `in_check() and not
legal_moves()`, and Python's `and` short-circuits. **Goal regression over a
conjunctive goal IS a short-circuiting `and` with the cheap, selective conjunct
written first.** The technique was already in the baseline, written by whoever
chose that conjunct order, probably without thinking of it as planning.

Two things follow. A classical technique can be present in ordinary code under
another name, so "does the baseline already do this?" is a question to ask before
building the thing, not after measuring it. And 1.0× is diagnostic: a real
improvement rarely lands on exactly parity, so that number usually means the two
arms are the same arm.

This is the third time in this repository a control has decided an outcome — the
untrained-network baseline that inflated GTMNN's transfer twelve-fold (33), the
argmax exploitability that scored five solvers identically (35), and now this. In
all three the mistake was in the comparison rather than in the thing being
compared.

*Where:* `GREN/gren/regress.py` (`conjunctive`, `expected_work`) · **confirmed**

---

## What to test next

Ordered by how much they would change, per unit of effort:

0. ~~**Does transfer survive ACROSS regions, or only within them?**~~ (29) **Answered: the graph is doing routing.** Cross-region transfer is +0.025 in one case of four; the within-region shared vocabulary is +0.023 against its proper baseline. Regions earn their place by *placing* games correctly — the wrong region is measurably worse, per the negative Shapley values — not by teaching each other.
1. ~~**Does validity transfer?**~~ (19) **Answered, and it is the same +0.023.** `p_valid` trained on chess alone leaves checkers at its majority class. The interesting follow-up is 29's refinement: build a region whose vocabulary is *mostly* shared, as `common_denominator.py` did with three inputs, and see whether 0.646 survives inside a cortex.
2. **Can the minimality criterion split a mechanic?** (38, and §8.3) Regression now has operators, but their preconditions are whole refusal codes. §8.3's rule — keep splitting a part while its halves are seen independently — would say whether `SELF_CHECK` is one precondition or two (king attacked, and no interposition). That changes what regression can plan through, and it is decidable from the corpus rather than argued about.
3. **Does goal-first ordering still pay at scale?** (37) The compression half of the result is small-N: four games and one perfectly-discriminating axis. Generate or gather thirty-odd games with overlapping goals and re-measure tokens-to-identification against alphabetical and against a highest-entropy-axis-first ordering. The knowability argument does not need this; the compression claim does.
4. **Is a trajectory worth more than the input that produced it?** (36) The premise of the whole memory design, and an afternoon to settle. Take CyclicCortex, which already has four games and trained regions: record input features, trajectory (region id, vocabulary slots written, hidden-activation pattern) and outcome per position, then build two lookup tables over the same episodes — one keyed by input, one by trajectory — and see which predicts the held-out outcome better. If trajectory-keyed does not win, the premise is wrong and nothing else in `Memory/` needs building.
5. **Can a mechanic be split when two games disagree about it?** (30) Chess and checkers both refuse `OCCUPIED_TARGET` under opposite conditions. If GREN probed *conditionally* — does this game still refuse when the occupant is an enemy? — the code would split into `OCCUPIED_TARGET_OWN` and `OCCUPIED_TARGET_ANY`, and the two games would stop sharing a slot they should never have shared. This is the cheapest test of whether the refusal channel can be widened enough to carry feature alignment, and it is a direct consequence of the only sharp negative result so far.
6. **Why does the credit path learn so weakly?** (2, 3) The sharpest open question in the repository. Credit is exact, the gradient is exact, the architecture has capacity — and the population still lands above the uniform baseline where a direct supervised signal on the same micros lands far below. Seats-per-game is the first suspect (`gini` 0.05 at 24/256 seated against 0.54 at 64/64), so sweep `M` against `N` before anything else. `B`/`λ` and the null-player/silence conflict are both already ruled out.
7. **Fix `b` everywhere and re-measure.** (24, 20) Upstream of everything in the vanishing-gradient work. `RadixCyclicNN` carries the same default — measure its realised `E|z|` before changing it.
8. ~~**Does the population actually specialise?**~~ (2) **Answered: yes, when it gets to play.** `gini` reaches 0.54 with every micro seated and 0.05 with 24 of 256 — so the split reward does bite, and seat count is what gates it. That is why 3 above is the sharper question.
9. **NCD against mechanic Jaccard.** (11) Where a feature-free similarity disagrees with the mechanic one, the vocabulary is blind — the only automatic check on the part of GREN hardest to verify.
10. **`universal_fraction`.** (18) The share of micros that are game-blind and transfer everywhere. Nothing predicts it; whether evolution finds a stable value is the sharpest test of whether GREN and GTMNN compose.
