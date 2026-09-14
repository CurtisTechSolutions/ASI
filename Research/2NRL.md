# 2NRL: Learning by Inverting Consistent Failure

**A training procedure derived from autodidactic practice**

R. Curtis — Curtis Tech Solutions
Working paper · September 2026

---

## Abstract

2NRL is a three-phase training procedure: **train on the failures at full rate,
invert the network, then fine-tune on the correct data at a reduced rate.** It
inverts the conventional treatment of wrong examples. Where negative sampling,
unlikelihood training and contrastive objectives all move a model *away* from
a wrong answer, 2NRL moves the model *toward* the wrong answer — deliberately,
completely — and then negates the representation it built.

The procedure is not derived from the literature. It is a formalisation of how
its author learned, being self-taught: *fail consistently, then do the inverse of
what failed; once a thread worth pulling appears, pull hard.*

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
3. **Effort is proportional to signal.** A promising direction is not pursued
   evenly with everything else; it is pursued *hard*.

The claim of this paper is that these three are implementable as an update rule,
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

## 5. Pulling the thread: proportional effort

The third commitment — *once I found a thread to pull on, I would pull hard* —
is implemented as failure-proportional boosting in the self-improvement loop.

Each generated sample is scored against the real distribution, giving a gap
$g = \bar{s}_{\text{real}} - s(x)$. Samples with $g > 0$ are failures; those
exceeding a margin are **blatant**. The negative phase then runs one pass per
distinct weight

$$w = \min\left(\text{boost},\ 1 + g/\text{margin}\right), \qquad
\eta^- \leftarrow w \cdot \eta^-$$

heaviest first, with the activation-parameter rate scaled identically. **The
worse the failure, the harder the model is driven to reproduce it** — including
its activation parameters — before the inversion turns all of it around. The
system fails blatantly, on purpose, in proportion to how blatant the failure
was.

A generation with no failures does not invert at all. Nothing was wrong, so
nothing is turned around.

### 5.1 An asymmetry worth naming

The implemented boost is driven by **failure** magnitude. The author's
description is of pursuing a **promising** direction — a thread is something
that looks like it is going somewhere, not something that went badly.

These are different signals, and only one of them is currently wired. A faithful
reading of "pull hard" might require a *positive*-side boost: a correct example
that arrived against expectation, or a fine-tune pass weighted by how much the
result improved, training harder than a routine success does.

Whether this asymmetry is an oversight or an intended simplification is open
(Q-13 in `RadixCyclicNN/DECISIONS.md`). It is the most concrete unexplored
extension the analogy suggests.

---

## 6. Relation to existing work

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

## 7. Implementation

2NRL is implemented in `RadixCyclicNN` and is the learning primitive the rest of
the system is shaped around. Every feedback source in the project — human thumbs
up/down, LLM judgements, sandbox results, discriminator scores — is funnelled
into the same $(B, G)$ pair, because that is the interface learning takes.

It runs in four places:

1. **Directly** — CLI `2nrl`, `POST /api/2nrl`, the 2NRL panel.
2. **Feedback** — rated texts dispatch to `two_nrl` (both sets rated),
   `reward` (good only) or `punish` (bad only, i.e. a negative pass then an
   inversion).
3. **The self-improvement loop** — a discriminator sorts generated samples;
   the worst become $B$, real corpus lines $G$; perpetually.
4. **Code generation** — wrong programs are $B$, the working program is $G$,
   with the sandbox providing a ground-truth verdict rather than an opinion.

A second model kind in the same system (a count/reward network) implements the
2NRL *interface* with a different *mechanism*: penalise the bad paths, reward
the good ones, **no inversion**. There, a negative reward already makes a path
unlikely, so a global inversion would be a global answer to a local problem.
This is an instructive boundary: 2NRL's inversion earns its place precisely
where the model *cannot* express a local negative, and is redundant where it
can.

---

## 8. Empirical status — and what would settle it

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

## 9. Limitations

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
   on §8.

---

## 10. Open questions

**Q-A. What does 2NRL stand for?** The expansion is recorded nowhere in the
implementation; the code calls it only "the author's two-phase scheme". This
paper uses the acronym as a proper name.

**Q-B. Should the positive side boost too?** See §5.1. The implemented boost
responds to failure magnitude; the described process responds to *promise*.

**Q-C. Is there a principled stopping rule?** What ended an iteration for the
author — and can that be made a computable criterion?

**Q-D. Is phase 1 to convergence, or partial?** The implementation trains on $B$
for a fixed small number of epochs. The entropy account of §4 implies phase 1
should proceed until the failure mode is *well* represented, since a partially
learned failure inverts into a partially useful signal. Is there an optimal
depth, and does it depend on $H(q)$?

**Q-E. Can $H(q)$ be estimated online?** If the advantage is governed by the
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
