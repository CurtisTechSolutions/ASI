# 2NRL: Learning by Inverting Consistent Failure

*Two-phase Negative Reinforcement Learning*

**A training procedure derived from autodidactic practice**

R. Curtis — Curtis Tech Solutions
Working paper · September 2026

---

## Abstract

2NRL — *Two-phase Negative Reinforcement Learning* — is a three-phase training
procedure: **train on the failures at full rate,
invert the network, then fine-tune on the correct data at a reduced rate.** It
inverts the conventional treatment of wrong examples. Where negative sampling,
unlikelihood training and contrastive objectives all move a model *away* from
a wrong answer, 2NRL moves the model *toward* the wrong answer — deliberately,
completely — and then negates the representation it built.

The procedure is not derived from the literature. It is a formalisation of how
its author learned, being self-taught: *fail consistently, then do the inverse of
what failed; explore widely until a thread appears, then tighten and iterate.*
The first half is implemented and is what this paper is mostly about; the second
is a schedule over search breadth that the system does not yet have (§5).

This paper states the procedure precisely, gives the mechanism that makes
inversion a coherent operation rather than a destructive one, and identifies its
governing condition. That condition falls directly out of the author's own
phrasing. The word **consistently** is load-bearing: the information recoverable
by inverting a failure is bounded by how *low-entropy* that failure is. A
systematic failure inverts into a usable signal. A random failure inverts into
nothing. This yields the paper's central claim in falsifiable form, and a
three-arm experiment that settles it with tooling the implementation already has.

We are explicit about status: 2NRL is implemented, in production use inside a
perpetual self-improvement loop, and its directional effect is asserted by
tests. It has **not** been measured against a baseline. Section 8 says what that
would take.

---

## 1. Origin

Most learning algorithms are proposed, then justified. This one was *lived*,
then written down.

The author is self-taught and describes the process that produced that education
as a repeating cycle:

> I learned by failing consistently, then doing the inverse/opposite. Once I
> found a thread to pull on, I would pull hard.

Three commitments are contained in that sentence, and 2NRL implements each:

1. **Failure is the primary input**, not an error to be minimised away. You go
   into it.
2. **The correction is inversion**, not gradual adjustment. Having understood
   what does not work, you do the opposite of it.
3. **Exploration narrows once it finds something.** In the author's fuller
   account: *explore rapidly and widely until you find a thread, then tighten the
   exploration and iterate.* Breadth first, then depth on whatever the breadth
   turned up.

The first two are implemented; the third, as §5 sets out, is not. The claim of
this paper is that these are implementable as an update rule,
that the rule is well-defined given an architecture in which inversion is
meaningful, and that its efficacy is governed by a single measurable property of
the failure distribution.

---

## 2. The procedure

Let $M$ be a model with parameters $\theta$. Let $B$ be a set of *failures* —
wrong, garbage or rejected outputs — and $G$ a set of correct examples.

**2NRL**$(B, G)$:

| Phase | Operation | Rate |
|---|---|---|
| 1. Negative | Train $M$ **on** $B$, maximising $p_\theta(B)$ | $\eta^-$ (full) |
| 2. Inversion | $\theta \leftarrow \mathcal{I}(\theta)$ | — |
| 3. Positive | Fine-tune $M$ on $G$ | $\eta^+ \ll \eta^-$ |

In the reference implementation $\eta^- = 0.05$ and $\eta^+ = 0.01$ — a 5:1
ratio — with the activation-parameter rate at $\eta^+/10$ in phase 3.

Note what phase 1 is *not*. It is not a penalty, not a negated gradient, not a
repulsion term. It is **ordinary training on the wrong answer**, at the full
learning rate, until the model reproduces it. The model is made to fail well
before it is made to succeed.

---

## 3. Why inversion, and why it is coherent here

### 3.1 The problem with moving away

Consider the conventional treatment. To discourage a wrong output $x^-$, one
applies $-\nabla_\theta \log p_\theta(x^-)$ — pushing probability mass off it.

This is a **repulsive** force, and repulsion is under-determined. It specifies a
direction to leave but no destination to arrive at. The mass displaced from
$x^-$ is redistributed by whatever the model's inductive bias happens to be, and
in a large output space the overwhelming majority of the places it can go are
*also wrong*. "Not that" is a weak constraint when the alternatives number in the
millions.

### 3.2 What inversion buys

2NRL replaces the repulsive force with a two-step construction:

1. **Represent the failure precisely.** After phase 1 the model does not hold a
   direction pointing away from the failure; it holds the failure itself, as a
   well-formed configuration of parameters. The failure mode has become a
   *location* in parameter space, not a gradient.
2. **Negate it.** Inversion maps that location to its opposite.

The gain is one of determinacy. Repulsion answers "where not to be"; inversion
answers "where to be instead" — provided the negation of a learned structure is
itself a coherent structure. That is an architectural precondition, not a
universal fact, and it is met here.

### 3.3 The mechanism in RadixCyclicNN

The host architecture scores a transition between graph nodes $p$ and $c$ as

$$s(p \to c) = W_{pc} \cdot f_p(z_p) \cdot f_c(z_c)$$

the edge weight times the activations of **both** endpoints. Each node's
activation is a parametric sine $f(x) = a\sin(b(x-h)) + k$.

Inversion is then two sign flips:

$$\mathcal{I}: \quad W \mapsto -W, \qquad a \mapsto -a \ \ \text{for every node}$$

Negating $W$ flips the product once; negating both endpoint amplitudes flips it
twice more. Net effect: **every edge signal changes sign.** The softmax over a
node's children reverses its ordering, and the most likely continuation becomes
the least likely. The operation is exact, is $O(N + E)$, and is its own inverse —
applying it twice is the identity, which the test suite asserts.

This is why the architecture and the algorithm are not separable. 2NRL requires
an inversion operator that is (i) cheap, (ii) exactly order-reversing, and
(iii) involutive. The bilinear-signed-score design supplies one. In an
architecture with, say, ReLU activations and unsigned attention weights, no such
operator exists, and 2NRL as stated cannot be run at all.

### 3.4 Why the positive phase must be gentler

Inversion is a **global, coarse** operation. It gets the direction right and the
details wrong: it reverses *everything*, including the parts of the model that
were fine. Phase 3 repairs the details.

The rate ratio is therefore not a tuning convenience but a structural
requirement. If $\eta^+ \gtrsim \eta^-$, phase 3 overwrites the inversion and the
procedure degenerates into ordinary supervised learning with a wasteful
pre-phase. The 5:1 default encodes a division of labour: **the inversion does
the work; the fine-tune polishes.**

---

## 4. The governing condition: failure must be consistent

This is the paper's central claim, and it is already present — precisely — in the
author's own sentence. Not "failing". *Failing consistently.*

### 4.1 Statement

Let $q$ be the distribution of the model's failures and $p$ the target
distribution. Phase 1 fits $\theta$ to $q$. Phase 2 negates it. The question is
how much of $p$ that recovers.

**Claim.** The information made available by inverting a failure distribution is
bounded above by that distribution's negative entropy. Writing $H(q)$ for the
entropy of the failure mode:

- $H(q)$ **low** — failure is *systematic*. The model consistently makes the
  same kind of mistake. Phase 1 captures a sharp, specific structure; its
  negation is correspondingly sharp and specific. Inversion is informative.
- $H(q)$ **high** — failure is *random*. Phase 1 fits noise. The negation of
  noise is noise. Inversion recovers nothing, and phases 1–2 have merely
  perturbed the model at full learning rate for no return.

In the limiting case, $q$ uniform: training on uniform noise moves the model
toward uniform, inverting uniform yields uniform, and 2NRL reduces to a costly
no-op followed by ordinary fine-tuning.

### 4.2 Consequence

2NRL is **not** a general-purpose replacement for supervised learning. It is a
procedure with a precondition, and the precondition is a property of the
*failures*, not of the task, the model or the data.

This reframes what the practitioner must supply. The question is no longer
"do I have wrong examples?" but **"are my wrong examples wrong in a consistent
way?"** Hand-written garbage that embodies a specific error mode is a good
negative set. Randomly corrupted text is a poor one, even though both are
"wrong".

It also explains something in the author's account that would otherwise look
like an incidental detail. One does not learn from failing *once*, nor from
failing *variously*. One learns from failing the *same way* repeatedly, until
the shape of the failure is clear enough to be negated. The repetition is what
drives the entropy down. Consistency is not a description of the author's
persistence — it is the mechanism's operating condition.

### 4.3 Where the implementation already respects this

Every source of negative examples in the system produces *structured* garbage,
never random noise:

- A language model asked for lines that are **deliberately wrong** — "false
  facts, scrambled reasoning" — which is a coherent error mode, not corruption.
- Generated samples that a discriminator scored **below the real distribution** —
  failures the model actually makes, which are by construction its systematic
  ones.
- Programs that **failed a sandbox or a judge** — wrong for a specific,
  reproducible reason.

This was not designed against the entropy condition; it fell out of building
the thing. That the condition retrodicts the design choices is weak evidence for
it, and worth recording as such.

---

## 5. Finding the thread: exploration, not effort

The third commitment is the one the implementation has **not** captured, and the
gap is worth stating precisely because the obvious reading of it is wrong.

### 5.1 What it is not

"Once I found a thread to pull on, I would pull hard" invites reading as a
weighting: pursue the promising example harder than the routine one. The system
does contain such a mechanism, and it is genuinely useful, so it is worth
describing before setting it aside.

In the self-improvement loop each generated sample is scored against the real
distribution, giving a gap $g = \bar{s}_{\text{real}} - s(x)$. Samples with
$g > 0$ are failures; those past a margin are **blatant**. The negative phase then
runs one pass per distinct weight

$$w = \min\left(\text{boost},\ 1 + g/\text{margin}\right), \qquad
\eta^- \leftarrow w \cdot \eta^-$$

heaviest first, with the activation-parameter rate scaled identically. The worse
the failure, the harder the model is driven to reproduce it — including its
activation parameters — before the inversion turns all of it around. The system
fails blatantly, on purpose, in proportion to how blatant the failure was. A
generation with no failures does not invert at all.

A positive-side analogue exists too: feedback is no longer a binary thumb but a
mark out of 10, so a 9-out-of-10 result is learned nine tenths as hard as a
perfect one and a 0 is skipped. Every judge in the system — LLM grader, sandbox,
discriminator, human — expresses confidence rather than only direction.

Both are sound mechanisms, and both belong to §2's two training phases: they say
*how hard* to represent an example. **Neither is the third commitment.**

### 5.2 What it actually is

The author's fuller account is a statement about **search**, not about rates:

> Explore rapidly and widely until you find a thread, then tighten up the
> exploration and iterate/fine-tune.

That is an *annealing schedule over exploration breadth*. Early on, sample
widely and cheaply — many candidates, high temperature, little commitment. When
something promising appears, **narrow**: fewer candidates, lower temperature,
concentrated on the region that produced it, and iterate there.

The distinction matters because the two are independent. Learning-rate weighting
decides how much a given example moves the model. Exploration breadth decides
*which examples are ever seen*. One can be maximal while the other is minimal.
Wide-then-narrow is a policy about where to look; boosting is a policy about what
to do once you have looked.

### 5.3 The gap

Every parameter governing breadth in this system — `temperature`, `k`, `beam`,
the sample `count`, `step_penalty` — is **fixed for the duration of a call** and
chosen by the caller. Nothing narrows as a run proceeds, and nothing detects that
a thread has appeared. A long self-improvement or tutoring run explores exactly
as widely in its final generation as in its first.

Two pieces of the machinery already exist, pointed elsewhere:

* **The schedule evaluator.** Learning rates are already expressible as sandboxed
  functions of the epoch, with linear, geometric, cosine, step and warm-up
  helpers, a live preview and a reverse switch. The same evaluator applied to
  `temperature`, `k` and `beam` would *be* the annealing schedule. It governs the
  wrong quantity.
* **Local re-exploration.** A conversational voice that catches itself looping
  backs up and widens its search — $k \times (\text{step} + 2)$ candidates, more
  the further back it goes. That is deliberately the opposite direction, and
  correctly so: it is local recovery from a dead end, not the global schedule.
  The two are compatible and would compose.

### 5.4 The harder half: what counts as finding a thread?

A schedule needs a trigger, and this is the genuinely open part. "A thread
appeared" is doing real work in the description and has no obvious formalisation.
Candidates, none yet tested:

* **A score threshold** — the first sample to clear some bar. Simple, and
  sensitive to a bar that has to be set in advance.
* **A plateau break** — narrow when the best-so-far improves after a stretch of
  not improving. Detects surprise rather than quality, which is closer to what a
  thread *is*.
* **A run of passes** — narrow after $n$ consecutive successes in one region.
  Robust, but slow to notice a single strong signal.
* **Discriminator disagreement** — narrow where the positive and negative models
  disagree most sharply, that being where the information is.

Without a detector the schedule has nothing to key on, so this is the part to
settle first. It is also the part where the human account is least directly
transferable: the author's recognition of a thread was a judgement, and the whole
exercise of this project is to ask what such a judgement is made of.

---

## 6. The negative phase made permanent

2NRL as stated in §2 is *transient*. Phase 1 builds a representation of the
failure, phase 2 negates it, and the representation itself is gone — what remains
is its effect on the weights. The knowledge that a particular fragment tends to be
wrong survives only as a diffuse change, unnameable and unqueryable.

The implementation has since taken the obvious next step: **keep it.**

### 6.1 A standing model of failure

A second network is maintained alongside the first. It has the same structure —
the same self-compressing cyclic graph, the same trigram windows, the same
searches — but every node and edge in it exists *because something went wrong
there*. Each edge accumulates the blame charged against it, how often it failed,
and how much **cleared** text has crossed it. The net evidence is

$$\text{evidence} = \max(0,\ \text{blame} - \lambda \cdot \text{clear})$$

so a fragment appearing in good and bad output alike stops carrying the verdict.
Weights are an edge's share of the failure mass leaving its parent, so a softmax
over them is the **failure distribution**: this network predicts the ways to fail
from a prefix, exactly as the positive model predicts the ways to succeed.

### 6.2 What this changes about the argument

Three things, and each is a strengthening rather than a replacement.

**Failure becomes queryable.** §3.2 argued that inversion's value is turning an
under-determined "away" into a determined "toward". A persistent failure model
goes further: one can ask of an arbitrary candidate *how* likely it is to be
wrong, and *which fragment* carries the risk. That is not available from a
transient phase at all.

**The entropy condition becomes measurable rather than assumed.** §4 claims the
recoverable information is bounded by the failure distribution's negative
entropy, and §4.3 could only observe that the implementation happens to generate
structured garbage. With an explicit model of $q$, its concentration is a
property one can compute. Q-E in §11 — estimate $H(q)$ online and skip the
inversion when the garbage is too diffuse to be worth inverting — becomes
straightforwardly implementable.

**Correction can be placed precisely.** A grade originally reached the graph as
two verdicts on two whole sentences: the attempt was garbage, the correction was
gospel. But most of a corrected sentence is word for word what the model wrote,
so the whole-sentence penalty *taxed the parts that were right*. Blame is now
placed by alignment — only the characters the teacher actually changed are
charged, the rest are cleared, and a fragment both sentences walk is rewarded
rather than penalised.

This last point deserves emphasis, because it qualifies §2. 2NRL's negative phase
trains on the failure *as a unit*. That is correct when the failure is a unit — a
rejected program, a hallucinated line, a fabricated fact. It is wrong when the
failure is local to a larger, mostly correct output. The persistent model
supports the granularity the transient phase cannot.

### 6.3 The pair at output time

Both networks now run on every answer by default: the positive model
over-samples, the negative one vetoes by accumulated blame, by peak blame on a
single fragment, or by the likelihood ratio
$\log P_{\text{neg}} - \log P_{\text{pos}}$, behind a coverage gate so text the
system has never failed is never vetoed on no evidence.

This is the adversarial idea relocated from *training time* to *inference time*.
The discriminator no longer only shapes the generator's weights; it sits on the
output path and refuses. Notably, the loops that *teach* the failure model
deliberately read the positive model **unfiltered** — a reviewer that only ever
saw what already passed the filter would have nothing left to teach.

### 6.4 The open question this raises

If a failure can be kept, named, and vetoed against, what is the transient
negative-phase-and-invert still buying?

Two defensible answers, and the paper does not settle between them:

* **They do different jobs.** Inversion changes what the model *tends to
  produce*; the failure model changes what is *allowed out*. A generation
  process that never proposes the failure is cheaper than one that proposes and
  filters it, and only inversion does the former.
* **The failure model subsumes it.** Inversion is a global, blunt operation
  (§10, item 2) that reverses the correct parts of the model along with the
  incorrect ones and relies on the positive phase to repair the damage. A precise,
  persistent, local account of failure may simply be the better instrument, with
  inversion a historical step toward it.

Settling this needs the same thing §9 needs: a measurement. The experiment in
§9.1 extends naturally — add an arm in which the negative set trains a
persistent failure model and guards the output, with no inversion anywhere, and
compare.

---

## 7. Relation to existing work

The project's stated method is to set current research aside and rebuild from
first principles. A paper should still say where its neighbours are, if only to
locate what is actually new.

| Approach | Treatment of wrong examples | Difference from 2NRL |
|---|---|---|
| **Negative sampling** (word2vec and descendants) | Gradient *away* from sampled negatives | Repulsive; never inverts; negatives are sampled, not the model's own failures |
| **Unlikelihood training** | Explicit penalty term on unwanted continuations | Repulsive; a loss modification, not a phase structure |
| **Contrastive learning** | Pull positives together, push negatives apart in embedding space | Operates on a metric embedding; no inversion operator; requires paired data |
| **GAN generators** | Discriminator signal backpropagated to the generator | Generator updated by *gradient*; 2NRL's generator is updated by *inversion*. 2NRL is used inside a GAN-style loop here, but the two are orthogonal |
| **Hopfield unlearning** (Crick–Mitchison "reverse learning"; Hopfield, Feinstein & Palmer 1983) | Anti-Hebbian update on spurious attractors to remove them | **The nearest prior art.** Both deliberately train toward an unwanted state, then reverse. But unlearning *subtracts a specific pattern*; 2NRL *negates the entire parameter set globally*, and then fine-tunes |
| **Self-correction / learning from mistakes** (LLM literature) | Mistakes fed back as context or as preference pairs | Operates at the data or prompt level; parameters are never inverted |

The distinguishing feature is narrow and specific: **global sign inversion of
the entire parameter set as a learning operator, positioned between a negative
and a positive training phase.** Anti-Hebbian unlearning is the closest thing to
it and is still local and subtractive rather than global and multiplicative.

The honest summary: 2NRL's *ingredients* have relatives; its *composition* — go
all the way into the failure, negate everything, then repair gently — does not
appear to be standard.

---

## 8. Implementation

2NRL is implemented in `RadixCyclicNN` and is the learning primitive the rest of
the system is shaped around. Every feedback source in the project — human thumbs
up/down, LLM judgements, sandbox results, discriminator scores — is funnelled
into the same $(B, G)$ pair, because that is the interface learning takes.

It now runs in eight places:

1. **Directly** — CLI `2nrl`, `POST /api/2nrl`, the 2NRL panel.
2. **Feedback** — rated texts dispatch to `two_nrl` (both sets rated),
   `reward` (good only) or `punish` (bad only, i.e. a negative pass then an
   inversion).
3. **The self-improvement loop** — a discriminator sorts generated samples;
   the worst become $B$, real corpus lines $G$; perpetually.
4. **Code generation** — wrong programs are $B$, the working program is $G$,
   with the sandbox providing a ground-truth verdict rather than an opinion.
5. **The tutor** — an LLM writes a sentence opening, the model completes it, the
   LLM marks the completion out of 10 and supplies the correction. Failed
   sentences are $B$ weighted by the mark; the corrections and drills are $G$.
6. **The critic** — the model writes freely, a reviewer marks it, and everything
   below the pass mark charges the failure model. This loop *only reads* the
   positive model; it trains nothing.
7. **Conversation with a language model** — the partner's own lines become $G$,
   because in that exchange, at that moment, they are exactly what a good reply
   would have looked like. A reply the model could only repeat is punished.
8. **The agent** — a whole tool-use attempt is one training text, so a failed
   attempt is $B$ and a successful one $G$ with no new machinery at all.

The eighth is worth noting for what it says about the interface. Because a tool
call is *text the model writes*, an entire episode of planning, calling and
answering reduces to a single string — and therefore to a single $(B, G)$
element. Nothing about 2NRL had to change to accommodate agency.

A second model kind in the same system (a count/reward network) implements the
2NRL *interface* with a different *mechanism*: penalise the bad paths, reward
the good ones, **no inversion**. There, a negative reward already makes a path
unlikely, so a global inversion would be a global answer to a local problem.
This is an instructive boundary: 2NRL's inversion earns its place precisely
where the model *cannot* express a local negative, and is redundant where it
can.

---

## 9. Empirical status — and what would settle it

**Stated plainly: 2NRL has not been measured against a baseline.**

What exists is a *directional* assertion, in `tests/test_model.py`: after
`two_nrl(bad, good)`, a garbage continuation is less likely than it was before.
That establishes the procedure does what it says. It does **not** establish that
it beats training on $G$ alone, nor that it beats negative sampling on the same
$(B, G)$.

The claim in §4 is, however, cheaply falsifiable, and the implementation already
contains every component needed.

### 8.1 Proposed experiment: the entropy condition

**Hypothesis.** 2NRL's advantage over positive-only training is a decreasing
function of the entropy of the negative set.

**Design.** Fix a corpus $G$ and a held-out evaluation set. Construct three
negative sets at matched size and matched character distribution, differing only
in structure:

| Arm | Negative set $B$ | Expected $H(q)$ |
|---|---|---|
| **A — systematic** | Hand-written garbage embodying one consistent error mode (`data/sample_garbage.txt`) | low |
| **B — semi-systematic** | An LLM asked for deliberately wrong lines (`ollama corpus --style garbage`) | medium |
| **C — random** | Character-level corruption of $G$ at matched edit distance | high |

Against two controls: **D** positive-only training on $G$ at matched total
compute, and **E** negative sampling on the same $(B, G)$ without inversion.

**Measurement.** Held-out per-character log-probability. Matched compute across
all arms, multiple seeds, variance reported.

**Predictions.**

- **P1.** A > D. Systematic failure, inverted, beats not using it.
- **P2.** A > B > C. Advantage decreases monotonically with negative-set entropy.
- **P3.** C ≈ D, or C < D. Random garbage yields no benefit, and may hurt —
  phases 1–2 perturb the model at full rate for nothing.
- **P4.** A > E. Inversion beats repulsion on identical data. *This is the load-
  bearing comparison*; without it, 2NRL's advantage could be nothing more than
  the negatives being used at all.

**P3 is the sharpest test.** Every competing account of why 2NRL might work —
regularisation, escaping local minima, the extra compute of phase 1 — predicts
that arm C helps roughly as much as arm A. Only the entropy account predicts
C fails while A succeeds. If C matches A, §4 is wrong.

### 8.2 Second experiment: the rate ratio

§3.4 argues $\eta^-/\eta^+ > 1$ is structurally required, not merely tuned.
Sweeping the ratio over $\{1/5, 1, 5, 25\}$ should show performance collapsing
toward the positive-only baseline as the ratio approaches and passes 1. If the
optimum sits at 1, the inversion is contributing nothing that the fine-tune is
not immediately undoing.

---

## 10. Limitations

1. **The architectural precondition is strong.** 2NRL needs an exact,
   cheap, involutive, order-reversing inversion operator. Most architectures do
   not have one. The procedure may not be portable beyond signed bilinear
   scoring of the kind used here.
2. **Inversion is global and therefore blunt.** It reverses the correct parts of
   the model along with the incorrect ones, and relies on phase 3 to repair the
   damage. The system already contains a partial admission of this: a *local*
   variant that flips alternating nodes along a single failed path, used instead
   of a global inversion when a failure is severe but isolated. Local and global
   correction are alternatives there, not layers — which suggests the global
   operator is not always the right tool.
3. **No stopping rule.** "Fail, invert, repeat" describes a loop. In a human
   life it terminates when the thing is learned. The self-improvement loop here
   runs indefinitely with no convergence criterion and no held-out evaluation;
   its only signal is a score gap from its own discriminator, which can widen
   while output quality falls.
4. **$n = 1$ on the human evidence.** The origin is one person's account of
   their own learning. That is a genuine existence proof — the procedure
   demonstrably produced a working education at least once — and it is not a
   controlled result about human learning generally. It is offered here as the
   *source* of the algorithm, not as evidence for it. The algorithm has to stand
   on §9.

---

## 11. Open questions

**Q-A. What does 2NRL stand for?** The expansion is recorded nowhere in the
implementation; the code calls it only "the author's two-phase scheme". This
paper uses the acronym as a proper name.

**Q-B. What counts as finding a thread?** See §5.4. The wide-then-narrow
schedule is well defined once there is a detector; the detector is the open part,
and it is where the human account transfers least directly.

**Q-C. Is there a principled stopping rule?** What ended an iteration for the
author — and can that be made a computable criterion?

**Q-D. Is phase 1 to convergence, or partial?** The implementation trains on $B$
for a fixed small number of epochs. The entropy account of §4 implies phase 1
should proceed until the failure mode is *well* represented, since a partially
learned failure inverts into a partially useful signal. Is there an optimal
depth, and does it depend on $H(q)$?

**Q-E. Can $H(q)$ be estimated online?** (See §6.2 — now tractable.) If the advantage is governed by the
negative set's entropy, the system could *measure* it and skip phases 1–2 when
the garbage is too diffuse to be worth inverting — making the precondition
self-enforcing rather than a caveat in a paper.

---

## References

Crick, F. & Mitchison, G. (1983). The function of dream sleep. *Nature* 304,
111–114.

Hopfield, J. J., Feinstein, D. I. & Palmer, R. G. (1983). "Unlearning" has a
stabilizing effect in collective memories. *Nature* 304, 158–159.

Mikolov, T. et al. (2013). Distributed representations of words and phrases and
their compositionality. *NeurIPS*.

Welleck, S. et al. (2020). Neural text generation with unlikelihood training.
*ICLR*.

Goodfellow, I. et al. (2014). Generative adversarial networks. *NeurIPS*.

---

## Companion documents

- `RadixCyclicNN/DECISIONS.md` — D-009 (2NRL), D-010 (inversion), D-027
  (proportional boosting), D-028 (local inversion), D-023 (the count model's
  non-inverting variant).
- `RadixCyclicNN/DESIGN.md` §8, §9.1 — the normative specification.
- `Research/SineWaveActivationFunction.md`, `Research/CyclesAreAFeature.md` —
  the architectural claims 2NRL's inversion operator depends on.
