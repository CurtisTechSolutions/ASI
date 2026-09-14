# 2NRL: Learning by Inverting Consistent Failure

*Double-Negative Reinforcement Learning*

Mason Curtis — Curtis Tech Solutions
Working paper · September 2026

---

## Abstract

I train on the failures at full rate, invert the whole network, then fine-tune on
the correct data at a lower rate. That is 2NRL — *Double-Negative Reinforcement
Learning*, for the two negatives in it — and it inverts the usual treatment of
wrong examples. Negative sampling, unlikelihood training and
contrastive objectives all move a model *away* from a wrong answer. I move the
model *toward* the wrong answer — deliberately, all the way — and then negate
what it built.

I did not get this from the literature. It is how I taught myself: fail
consistently, then do the inverse of what failed; explore widely until a thread
appears, then tighten and iterate. The first half is implemented and is most of
this paper. The second half is a schedule over search breadth that I have not
built yet, and §6 says so plainly.

The important part of this paper is the condition, and it came out of my own
sentence when I looked at it properly. The load-bearing word is **consistently**.
What you can recover by inverting a failure is bounded by how *low-entropy* that
failure is. Fail the same way every time and the inversion hands you something.
Fail randomly and it hands you nothing, because the negation of noise is noise.
That makes the claim falsifiable, and §11 gives the experiment — three arms, all
of which my existing tooling can run.

On status, blunt: 2NRL is implemented, it is running inside a perpetual
self-improvement loop, and the tests assert that it does what it says. **It has
never been measured against a baseline.** §10 is what that would take.

---

## 1. The original note

> I learned by failing consistently, then doing the inverse/opposite. Once I
> found a thread to pull on, I would pull hard.

That was the whole thing. Everything below is what I meant by it.

---

## 2. Where it came from: how I taught myself

Most learning algorithms get proposed and then justified. This one I lived first
and wrote down afterwards.

I am self-taught. I did not learn by being shown the right answer and copying it.
I learned by going at something, getting it wrong, going at it again the same way
and getting it wrong the same way — and then, once I could see the *shape* of
what I was doing wrong, doing the opposite of that. Not a small adjustment away
from it. The opposite.

Then, when something finally moved, I stopped casting around and went hard at
that one thing until it gave.

Three commitments are in there, and they need separating, because they are not
the same idea:

1. **Failure is the input, not the error term.** You do not minimise it away. You
   go into it until you understand it.
2. **The correction is inversion, not adjustment.** Once you know what does not
   work, you do the reverse of it. A small step away from a wrong answer is still
   near the wrong answer.
3. **Exploration narrows once it finds something.** Explore rapidly and widely
   until you find a thread; then tighten the exploration and iterate on it.
   Breadth first, then depth on whatever the breadth turned up.

The first two are implemented, and they are 2NRL. The third is not, and §6 is
about that gap rather than about pretending it is closed.

Where I learned this most plainly was games. I would fail at a game **on
purpose**, over and over, to work out its rules — and once I had the rules I
figured out the rest very quickly. That shape, a long deliberate flat stretch
followed by a fast one, is not incidental to the method. It is what the method
looks like from outside, and §6.5 is about why.

---

## 3. The procedure

Take a model `M` with parameters `θ`. Take `B`, a set of failures — wrong,
garbage or rejected outputs — and `G`, a set of correct examples.

**2NRL(B, G)** is three phases:

| Phase | What happens | Rate |
|---|---|---|
| 1. Negative | Train `M` **on** `B`. Maximise `p(B)`. | `η⁻` (full) |
| 2. Inversion | `θ → I(θ)` | — |
| 3. Positive | Fine-tune `M` on `G` | `η⁺`, much smaller |

In my implementation `η⁻ = 0.05` and `η⁺ = 0.01` — a 5:1 ratio — with the
activation parameters moving at `η⁺/10` in phase 3.

Be clear about what phase 1 is **not**. It is not a penalty. It is not a negated
gradient. It is not a repulsion term. It is **ordinary training on the wrong
answer, at full learning rate, until the model reproduces it.** I make the model
fail well before I make it succeed.

### 3.1 The name

2NRL is **Double-Negative Reinforcement Learning**. The `2N` is the two
negatives, and they are the first two rows of that table:

1. **A negative input** — the data is the failures, and I train *on* them rather
   than away from them.
2. **A negative operation** — I negate what that training built.

Two negatives make a positive. That is not a pun; it is the reason the procedure
*arrives* somewhere instead of merely leaving somewhere, which is the argument of
§4.2. One negative — push the mass off the wrong answer — leaves the model
anywhere at all. `¬¬P` comes back to `P`.

The third phase is not a third negative. It is the positive one, which is why
"double" and "three phases" do not contradict each other: two of the three are
negations and the last is not.

The name also states the condition. Double-negative *elimination* — `¬¬P ⟹ P` —
only holds where the negation is well defined, and two things have to be true for
that. Most of this paper is about them:

- **The negation of a learned structure has to be a coherent structure** (§4.3).
  Mine is, and exactly: flipping the amplitude sign of a sine activation negates
  that unit to machine precision, so `¬¬` comes back to precisely where it
  started rather than approximately. An architecture whose units cannot be
  negated cleanly has no double negative to eliminate.
- **The failure has to be consistent** (§5). A systematic failure has a coherent
  opposite. Noise does not — `¬(noise)` is not a place, and negating it twice
  restores nothing.

So the name carries the claim rather than labelling it. Where either condition
fails, 2NRL is a double negative in the looser English sense: two negations that
muddle the meaning instead of restoring it.

---

## 4. Why inversion, and why it works in this architecture

### 4.1 The problem with moving away

The conventional move is to apply `−∇ log p(x⁻)` and push probability mass off
the wrong answer.

That is a *repulsive* force, and repulsion is under-determined. It tells you a
direction to leave. It does not tell you where to arrive. The mass you displace
goes wherever the model's inductive bias happens to send it, and in a large
output space almost everywhere it can go is *also wrong*. "Not that" is a weak
instruction when there are millions of alternatives, and most of them are bad
too.

That is what is actually wrong with the standard treatment, and it matches what I
found teaching myself. Being told I was wrong never helped much. Working out
*exactly how* I was wrong did, because the opposite of a specific mistake is a
specific instruction.

### 4.2 What inversion buys

So I replace the repulsion with two steps:

1. **Represent the failure precisely.** After phase 1 the model is not holding a
   direction pointing away from the failure. It is holding the failure itself, as
   a well-formed set of parameters. The failure mode has become a *place*, not a
   gradient.
2. **Negate it.** Inversion maps that place to its opposite.

The gain is determinacy. Repulsion answers *where not to be*. Inversion answers
*where to be instead* — as long as the negation of a learned structure is itself
a coherent structure. That is an architectural precondition and not a free fact,
and my architecture happens to satisfy it. That is not luck; I built the two
together.

### 4.3 The mechanism

The network scores a transition between nodes `p` and `c` as

```
score(p → c) = W[p,c] · f_p(z_p) · f_c(z_c)
```

the edge weight times the activations of **both** endpoints. Every node's
activation is a parametric sine, `f(x) = a·sin(b(x − h)) + k`
(see [the sine paper](SineWaveActivationFunction.md)).

Inversion is then two sign flips:

```
W → −W          for every edge
a → −a          for every node
```

Negating `W` flips the product once. Negating both endpoint amplitudes flips it
twice more. Net effect: **every edge signal changes sign.** The softmax over a
node's children reverses its order, and the most likely continuation becomes the
least likely. The operation is exact, costs `O(N + E)`, and is its own inverse —
run it twice and you are back where you started, which the tests assert.

This is why the architecture and the algorithm do not come apart. 2NRL
needs an inversion operator that is cheap, exactly order-reversing, and
involutive. A signed bilinear score gives me one. Put ReLU activations and
unsigned weights in there and no such operator exists — 2NRL cannot be run at
all. The sine's sign parameter is what makes negating a unit a *parameter*
change rather than a structural one.

### 4.4 Why the positive phase has to be gentler

Inversion is global and blunt. It gets the direction right and the details wrong,
because it reverses *everything* — including the parts of the model that were
fine. Phase 3 repairs the details.

So the rate ratio is not a tuning convenience, it is structural. If `η⁺` gets
close to `η⁻`, phase 3 overwrites the inversion and the whole thing degenerates
into ordinary supervised learning with an expensive pointless pre-phase. The 5:1
default says who does what: **the inversion does the work, the fine-tune
polishes.**

---

## 5. The governing condition: the failure has to be consistent

This is the part that matters most, and it only came clear on going back to the
original sentence. Not "failing". *Failing consistently.*

### 5.1 The claim

Let `q` be the distribution of the model's failures. Phase 1 fits `θ` to `q`.
Phase 2 negates it. The question is how much of the target that recovers.

**What you can recover by inverting a failure distribution is bounded by that
distribution's negative entropy.** Write `H(q)` for the entropy of the failure
mode:

- **`H(q)` low — the failure is systematic.** The model makes the same kind of
  mistake every time. Phase 1 captures a sharp, specific structure, and its
  negation is just as sharp and just as specific. The inversion tells you
  something.
- **`H(q)` high — the failure is random.** Phase 1 fits noise. The negation of
  noise is noise. You recover nothing, and phases 1–2 have kicked the model at
  full learning rate for no return.

Take the limit: if `q` is uniform, training on uniform noise moves the model
toward uniform, inverting uniform gives uniform, and 2NRL has reduced to an
expensive no-op followed by ordinary fine-tuning.

### 5.2 What follows from it

2NRL is **not** a drop-in replacement for supervised learning. It is a procedure
with a precondition, and the precondition is a property of the *failures* — not
of the task, not of the model, not of the data.

That changes the question a practitioner has to answer. It is not "do I have
wrong examples?" It is **"are my wrong examples wrong in a consistent way?"**
Hand-written garbage embodying one specific error mode is a good negative set.
Randomly corrupted text is a bad one. Both are "wrong"; only one of them is
*informative*.

It also explains something in my own account that I had been treating as
incidental. You do not learn from failing *once*, and you do not learn from
failing *variously*. You learn from failing the *same way* repeatedly, until the
shape of it is clear enough to be negated. The repetition is what drives the
entropy down. Consistency is not a description of how stubborn I was — it is the
mechanism's operating condition — and it was in the sentence before the reason
for it was.

### 5.3 The implementation already respects this, by accident

Every source of negatives in the system produces *structured* garbage, never
random noise:

- An LLM asked for lines that are **deliberately wrong** — false facts, scrambled
  reasoning. That is a coherent error mode, not corruption.
- Generated samples a discriminator scored **below the real distribution** —
  failures the model actually makes, which are by construction its systematic
  ones.
- Programs that **failed a sandbox or a judge** — wrong for a specific,
  reproducible reason.

I did not design any of that against the entropy condition. It fell out of
building the thing. That the condition retrodicts choices I made for other
reasons is weak evidence for it, and I am recording it as weak.

---

## 6. Finding the thread: exploration, not effort

The third commitment is the one I have **not** built. The gap is worth stating
precisely, because the obvious reading of that sentence is wrong — and because
the part of it I used to call unformalisable turns out not to be. §6.1 to §6.3
say what the commitment is and what is missing; §6.4 says what the trigger is,
and §6.5 what the whole thing is a search *for* — the rules of the game, which
is where I learned all of this in the first place.

### 6.1 What it is not

"Once I found a thread to pull on, I would pull hard" reads like a weighting:
pursue the promising example harder than the routine one. The system does contain
a mechanism like that, and it is useful, so it is worth describing before setting
it aside.

In the self-improvement loop every generated sample is scored against the real
distribution, giving a gap `g = real_mean − score(x)`. Samples with `g > 0` are
failures; past a margin they are *blatant*. The negative phase then runs one pass
per distinct weight

```
w = min(boost, 1 + g / margin),      η⁻ ← w · η⁻
```

heaviest first, with the activation parameters scaled the same way. The worse the
failure, the harder I drive the model to reproduce it — activation functions
included — before the inversion turns all of it around. The system fails
blatantly, on purpose, in proportion to how blatant the failure was. A generation
with nothing wrong in it does not invert at all.

There is a positive-side analogue too: feedback is a mark out of 10 rather than a
thumb, so a 9-out-of-10 result is learned nine tenths as hard as a perfect one and
a 0 is skipped. Every judge in the system — LLM grader, sandbox, discriminator,
human — can express confidence and not just direction.

Both are sound, and both belong to §3's two training phases: they say *how hard*
to represent an example. **Neither of them is the third commitment.**

### 6.2 What it actually is

What I meant is a statement about **search**, not about rates:

> Explore rapidly and widely until you find a thread, then tighten up the
> exploration and iterate.

That is an annealing schedule over exploration *breadth*. Early on, sample widely
and cheaply — many candidates, high temperature, no commitment. When something
promising shows up, **narrow**: fewer candidates, lower temperature, concentrated
on whatever produced it, and iterate there.

The distinction matters because the two are independent. Learning-rate weighting
decides how much a given example moves the model. Exploration breadth decides
*which examples are ever seen at all*. Either can be maximal while the other is
minimal. Wide-then-narrow is a policy about where to look; boosting is a policy
about what to do once you have looked.

### 6.3 The gap

Every parameter that governs breadth in my system — `temperature`, `k`, `beam`,
the sample `count`, `step_penalty` — is **fixed for the duration of a call** and
chosen by whoever made the call. Nothing narrows as a run proceeds, and nothing
acts on a reward when one lands. A long self-improvement or tutoring run explores
exactly as widely in its last generation as in its first, which is not how I work
and not what I described. §6.4 settles what the trigger is; what is missing is
the schedule that listens to it.

Two pieces of the machinery already exist, pointed at the wrong thing:

* **The schedule evaluator.** Learning rates are already expressible as sandboxed
  functions of the epoch, with linear, geometric, cosine, step and warm-up
  helpers, a live preview and a reverse switch. Point that same evaluator at
  `temperature`, `k` and `beam` and it *is* the annealing schedule. It currently
  governs the wrong quantity.
* **Local re-exploration.** A conversational voice that catches itself looping
  backs up and widens its search — `k × (step + 2)` candidates, more the further
  back it goes. That is deliberately the opposite direction, and correctly so:
  it is local recovery from a dead end, not the global schedule. The two compose
  rather than conflict.

### 6.4 What counts as finding a reward

A schedule needs a trigger, and I had been asking the wrong question. "What
counts as finding a *thread*?" treats it as a judgement, and judgements are
exactly what this project exists to take apart. The question with an answer is
**what counts as finding a reward**:

> Mainly when the network is rewarded, or we reach a known area of completion.

Two triggers, both of them events rather than estimates, and the system already
emits both:

* **The network is rewarded.** A judge paid out — the LLM grader passed the
  attempt against its criteria, the sandbox ran the program, a human marked it
  up, the discriminator scored a sample as real. Reward is not inferred from the
  shape of a search curve. It happens, and the system already knows when.
* **A known area of completion is reached.** The walk arrived somewhere already
  known to be an end: `reached_end` on a path, a region of the graph that an
  earlier run finished in. Not "this looks promising" but "this is a place I have
  finished before".

The difference from what I had been considering matters. A score threshold, a
plateau break, a run of passes, discriminator disagreement — each of those is an
*estimator* of a thread, inferred from the statistics of the search, and each
needs a parameter somebody sets in advance. The two above are *observations* of
one. They need no bar, they cannot be tuned wrong, and both were already being
computed for other reasons. They remain the fallback for the case the two
triggers do not cover: a run that is genuinely getting warmer without having yet
been paid or arrived anywhere.

### 6.5 Failing at the game to learn its rules

The framing I actually work in is game-theoretic, and games are where the method
came from.

A game is two things: **rules** and **payoffs**. The rules say which moves exist
and which are legal. The payoffs say what the legal moves are worth. Starting
out you know neither. The usual approach goes after the second — play, observe
the payoff, fit a model of the reward, repeat.

I went after the first, and I went after it by losing on purpose:

> I failed on purpose at games to understand the rules, and then figured out the
> rest very quickly once I knew the rules.

#### Why deliberate failure is the efficient probe of a rule

Because **a rule is invisible while you are obeying it.**

Play a legal move and the game says nothing. You learn one bit — *that was
allowed* — and it does not tell you where the edge of *allowed* is. Play an
illegal one and the game answers exactly: not that, and here is the line. A
boundary is located by crossing it, never by staying well inside it.

So the information a move carries about the rules is not symmetric. Success is
cheap and says almost nothing about structure. Failure is expensive and says
precisely where the structure is. If what you are trying to recover is the rule
set, then deliberately walking off the board is the *targeted* experiment and
playing well is the wasteful one.

That is all "fail on purpose" means. It is not temperament and it is not
grit. It is experiment design, and it is the same design as phase 1 of §3: go
all the way into the failure, because the failure is where the information is.

Rules and payoffs are different objects, and they want opposite strategies:

| | Rules | Payoffs |
|---|---|---|
| shape | hard boundaries, usually deterministic | a landscape, usually noisy |
| how many | few | many |
| learned by | violating them | sampling them |
| one observation gives | the location of a boundary | one noisy value |
| once known | the action space collapses | you still have to search it |

The last row is the one that matters, and it is what "the rest went quickly"
means.

#### The phase transition

Knowing the rules of chess does not make you good at chess. Not knowing them
makes playing it impossible. Rules do not tell you what to do — they collapse
*anything at all* down to *the legal moves*, and that is an enormous reduction in
what is left to search.

So the curve has a shape, and it is not a smooth climb. There is a long flat
stretch that looks like failing and is actually rule-discovery, and then a fast
rise once the collapse happens and the remaining problem is small enough to
finish. I recognise that shape from my own learning and it is why I trust the
method: the flat part is not wasted, it is buying the collapse.

That is also the honest version of "explore widely, then pull hard" from §6.2.
The wide phase is not random casting about. It is probing for rules. The narrow
phase is possible because the rules arrived and shrank the space — so the reason
narrowing works is not that I got lucky, it is that there is less left to search.

And it says what §6.4's trigger is really detecting. A reward, or a known area of
completion, is not merely a good outcome to celebrate. **It is evidence that a
rule has been learned** — the first sign the collapse has happened and that
narrowing will now pay.

#### What the helper is for: it writes the rules down

The helper's job follows directly, and it is not teaching. In the agent loop the
LLM has four roles and *solving the task is the last one it is given*. Its
**first** role is to write the acceptance criteria before anything is attempted —
and the acceptance criteria **are the rules of the game, stated before play**.

```
task -> acceptance criteria (LLM, written first) -> the network calls tools
     -> judge against the criteria (LLM)
     -> teach with the same real tools when it failed (LLM)
     -> train on the failures, invert, fine-tune (2NRL)
```

Now read that against §5 and the reason for the ordering becomes clear. §5 says
what inverting a failure can recover is bounded by how *consistent* the failure
is: systematic failure inverts into signal, random failure inverts into nothing.

A network turned loose on a game whose rules were never stated fails
**randomly**. Its failures are high-entropy — wrong in a different way every
time — and there is nothing in them to invert. A network failing against criteria
written down first and judged by the same fixed standard every time fails
**against a specific rule**. Those failures are wrong in the same way, and that
is exactly the low-entropy negative set §5 requires.

So the helper is not a teacher. It is what makes failure *rule-shaped*, and
rule-shaped failure is what is worth negating. That is why it writes the criteria
first and solves last: a helper that solved the task would remove the failure,
and the failure is the input.

The clearest instance is already running. The network calls tools by *writing*
them — `<tool>web_fetch {"url": "..."}</tool>` — so the call format is a rule of
the game in the most literal sense, a syntax that is either legal or not. The
network cannot write it at first, and the helper **repairs the calls the network
cannot write yet**. That is failing at a rule and being shown the rule, and the
whole attempt becomes one training text either way.

#### What this predicts

- **Learning curves under this regime should be hockey-sticks, not smooth
  climbs.** A long flat stretch, then a fast rise. A smooth curve would mean the
  system is fitting payoffs incrementally rather than acquiring rules and
  collapsing the space, which is the opposite of what I claim it does.
- **Rule-failures should fall before reward rises,** and sharply. Malformed tool
  calls, criteria missed on a technicality, outputs in the wrong shape — those
  should disappear first, and the payoff should improve *after*. If reward climbs
  while rule-failures stay flat, the system is learning the payoff landscape
  directly and the rules story is decoration.
- **Stating the rules should matter more than stating them well.** Withholding
  the acceptance criteria — letting the network fail without a written standard —
  should hurt more than making the criteria vague, because the criteria's job is
  to make failure consistent, not to make it correct.

---

## 7. The negative phase, made permanent

2NRL as stated in §3 is *transient*. Phase 1 builds a representation of the
failure, phase 2 negates it, and then the representation is gone — all that
survives is its effect on the weights. The knowledge that some particular
fragment tends to be wrong lives on only as a diffuse change, with no name and no
way to query it.

The implementation has since taken the obvious next step: **keep it.**

### 7.1 A standing model of failure

I now keep a second network beside the first. Same structure — the same
self-compressing cyclic graph, the same trigram windows, the same searches — but
every node and edge in it exists *because something went wrong there*. Each edge
accumulates the blame charged against it, how often it failed, and how much
**cleared** text has crossed it. The net evidence is

```
evidence = max(0, blame − λ · clear)
```

so a fragment that turns up in good and bad output alike stops carrying the
verdict. Weights are an edge's share of the failure mass leaving its parent, so a
softmax over them is the **failure distribution**. This network predicts the ways
to fail from a prefix, exactly as the positive one predicts the ways to succeed.

### 7.2 What that changes about the argument

Three things, and each strengthens §4 and §5 rather than replacing them.

**Failure becomes queryable.** §4.2 said inversion's value is turning an
under-determined "away" into a determined "toward". A persistent failure model
goes further: I can ask of any candidate *how* likely it is to be wrong and
*which fragment* carries the risk. A transient phase cannot answer that at all.

**The entropy condition becomes measurable instead of assumed.** §5 claims the
recoverable information is bounded by the failure distribution's negative
entropy, and §5.3 could only point out that my implementation happens to produce
structured garbage. With an explicit model of `q`, its concentration is something
I can compute. Estimating `H(q)` online and skipping phases 1–2 when the garbage
is too diffuse to be worth inverting stops being a nice idea and becomes
straightforward.

**Correction can be placed precisely.** A grade used to reach the graph as two
verdicts on two whole sentences: the attempt was garbage, the correction was
gospel. But most of a corrected sentence is word for word what the model wrote,
so the whole-sentence penalty *taxed the parts that were right*. Blame is now
placed by alignment — only the characters the teacher actually changed are
charged, the rest are cleared, and a fragment both sentences walk gets rewarded
rather than penalised.

That last point qualifies §3, and leaving it unsaid would be dishonest. 2NRL's
negative phase trains on the failure *as a unit*. That is correct when the
failure *is* a unit — a rejected program, a hallucinated line, a fabricated fact.
It is wrong when the failure is local to a larger, mostly correct output. The
persistent model supports a granularity the transient phase cannot.

### 7.3 The pair at output time

Both networks now run on every answer by default. The positive model
over-samples; the negative one vetoes — by accumulated blame, by peak blame on a
single fragment, or by the likelihood ratio `log P_neg − log P_pos` — behind a
coverage gate, so text the system has never failed is never vetoed on no
evidence.

This is the adversarial idea moved from *training time* to *inference time*. The
discriminator no longer only shapes the generator's weights; it sits on the
output path and refuses. One detail matters: the loops that *teach* the failure
model deliberately read the positive model **unfiltered**, because a reviewer
that only ever saw what already passed the filter would have nothing left to
teach.

### 7.4 The question this raises, which I have not settled

If a failure can be kept, named and vetoed against, what is the transient
invert-and-discard still buying?

Two defensible answers, and I do not yet know which is right:

* **They do different jobs.** Inversion changes what the model *tends to
  produce*. The failure model changes what is *allowed out*. A generator that
  never proposes the failure is cheaper than one that proposes it and filters it,
  and only inversion does the former.
* **The failure model subsumes it.** Inversion is global and blunt (§12, item 2),
  reverses the correct parts along with the incorrect ones, and leans on phase 3
  to repair the damage. A precise, persistent, local account of failure may
  simply be the better instrument, with inversion a step on the way to it.

Settling it needs what §10 needs — a measurement. The experiment in §11 extends
naturally: add an arm where the negative set trains a persistent failure model
and guards the output, with no inversion anywhere, and compare.

---

## 8. Where this sits next to existing work

My method is to set the current literature aside and rebuild from first
principles, and I stand by that. The neighbours are still worth naming, if only
to be clear about what is actually new here and what is not.

| Approach | What it does with wrong examples | How 2NRL differs |
|---|---|---|
| **Negative sampling** (word2vec and descendants) | Gradient *away* from sampled negatives | Repulsive; never inverts; the negatives are sampled rather than being the model's own failures |
| **Unlikelihood training** | A penalty term on unwanted continuations | Repulsive; a loss modification rather than a phase structure |
| **Contrastive learning** | Pull positives together, push negatives apart in an embedding | Works on a metric embedding; no inversion operator; needs paired data |
| **GAN generators** | Discriminator signal backpropagated into the generator | The generator is updated by *gradient*. Mine is updated by *inversion*. I use 2NRL inside a GAN-style loop, but the two ideas are orthogonal |
| **Hopfield unlearning** (Crick & Mitchison; Hopfield, Feinstein & Palmer, 1983) | Anti-Hebbian update on spurious attractors to remove them | **The nearest prior art, and worth naming as such.** Both deliberately train toward an unwanted state and then reverse. But unlearning *subtracts a specific pattern*; 2NRL *negates the entire parameter set* and then fine-tunes |
| **Self-correction in LLMs** | Mistakes fed back as context or as preference pairs | Operates at the data or prompt level. Parameters are never inverted |

The distinguishing feature is narrow, and stating it narrowly: **global sign
inversion of the entire parameter set, used as a learning operator, positioned
between a negative and a positive training phase.** Anti-Hebbian unlearning is
the closest thing to it and is still local and subtractive where mine is global
and multiplicative.

Honest summary: the *ingredients* of 2NRL have relatives. The *composition* — go
all the way into the failure, negate everything, then repair gently — does not
appear to be standard.

---

## 9. Where it is used

2NRL is implemented in `RadixCyclicNN` and it is the primitive the rest of the
system is shaped around. Every feedback source — a human thumb, an LLM
judgement, a sandbox result, a discriminator score — is funnelled into the same
`(bad, good)` pair, because that is the interface learning takes.

It now runs in eight places:

1. **Directly** — the `2nrl` command, `POST /api/2nrl`, the 2NRL panel.
2. **Feedback** — rated texts dispatch to `two_nrl` when both sets are rated,
   `reward` for good only, `punish` for bad only (a negative pass, then invert).
3. **The self-improvement loop** — a discriminator sorts generated samples; the
   worst become the garbage, real corpus lines the fine-tune set; forever.
4. **Code generation** — wrong programs are the garbage, the working program is
   the correction, and the sandbox gives a ground-truth verdict rather than an
   opinion. This is the only reward in the system that is not ultimately
   somebody's judgement.
5. **The tutor** — an LLM writes a sentence opening, the model completes it, the
   LLM marks the completion out of 10 and supplies the correction. Failed
   sentences are garbage weighted by the mark; the corrections and drills are the
   fine-tune pass.
6. **The critic** — the model writes freely, a reviewer marks it, and everything
   under the pass mark charges the failure model. This loop only *reads* the
   positive model; it trains nothing, which is what makes it safe to leave
   running.
7. **Conversation with a language model** — the partner's own lines become the
   good set, because in that exchange, at that moment, they are exactly what a
   good reply would have looked like. A reply the model could only repeat is
   punished.
8. **The agent** — a whole tool-use attempt is one training text, so a failed
   attempt is garbage and a successful one is correct, with no new machinery.

The eighth is worth dwelling on for what it says about the interface. Because a
tool call is *text the network writes*, an entire episode of planning, calling and
answering collapses to a single string — and therefore to a single element of a
`(bad, good)` pair. Nothing about 2NRL had to change to accommodate agency, and I
take that as a sign the interface is the right shape.

There is also a boundary worth recording. A second model kind in the same system
— the count/reward network — implements the 2NRL *interface* with a different
*mechanism*: penalise the bad paths, reward the good ones, **no inversion**.
There, a negative reward already makes a path unlikely, so inverting the whole
graph would be a global answer to a local problem. That is instructive:
inversion earns its place precisely where the model *cannot* express a local
negative, and is redundant where it can.

---

## 10. Evidence: what I have, and what I do not

**Plainly: I have not measured 2NRL against a baseline.**

What I have is a *directional* assertion, in `tests/test_model.py`. After
`two_nrl(bad, good)`, a garbage continuation is less likely than it was before.
That establishes the procedure does what it says on the tin. It does **not**
establish that it beats training on the good set alone, and it does not establish
that it beats negative sampling on the same data.

Better to say that than imply more. The argument in §4 and §5 is an argument, and
arguments are worth something — but the claim in §5 is cheap to falsify and has
been neither falsified nor confirmed. Everything needed is already in the
repository.

---

## 11. Predictions, and what would change my mind

### 11.1 The experiment

**Hypothesis.** 2NRL's advantage over positive-only training decreases as the
entropy of the negative set rises.

**Design.** Fix a corpus and a held-out evaluation set. Build three negative sets
at matched size and matched character distribution, differing only in structure:

| Arm | Negative set | Expected `H(q)` |
|---|---|---|
| **A — systematic** | Hand-written garbage embodying one consistent error mode (`data/sample_garbage.txt`) | low |
| **B — semi-systematic** | An LLM asked for deliberately wrong lines (`ollama corpus --style garbage`) | medium |
| **C — random** | Character-level corruption of the corpus at matched edit distance | high |

Against two controls: **D**, positive-only training at matched total compute, and
**E**, negative sampling on the same data without any inversion.

**Measurement.** Held-out per-character log-probability. Matched compute across
arms, multiple seeds, variance reported — the things I did not do on CartPole and
got fairly criticised for.

**Predictions.**

- **P1.** A beats D. Systematic failure, inverted, beats not using it.
- **P2.** A > B > C. The advantage falls monotonically as negative-set entropy
  rises.
- **P3.** C ≈ D, or C is *worse* than D. Random garbage buys nothing and may cost
  something, because phases 1–2 kick the model at full rate for no return.
- **P4.** A beats E. Inversion beats repulsion on identical data. **This is the
  load-bearing comparison.** Without it, any advantage I claim could be nothing
  more than the negatives being used at all.

### 11.2 What would change my mind

- **If arm C matches arm A, §5 is wrong.** This is the sharp one. Every competing
  explanation of why 2NRL might work — regularisation, escaping local minima, the
  extra compute in phase 1 — predicts that random garbage helps about as much as
  systematic garbage. Only the entropy account predicts C fails while A succeeds.
  If C matches A, the mechanism is not what I say it is.
- **If arm E matches arm A**, then inversion is doing nothing that repulsion does
  not, and the interesting part of this paper collapses to "use your negatives".
- **If the rate ratio optimum sits at 1**, §4.4 is wrong. Sweep `η⁻/η⁺` over
  1/5, 1, 5, 25: performance should collapse toward the positive-only baseline as
  the ratio approaches and passes 1. If it does not, the inversion is contributing
  nothing that the fine-tune is not immediately undoing, and I should drop it.
- **If a persistent failure model with no inversion anywhere matches full 2NRL**,
  then §7.4's second answer is the right one, inversion was a step on the way, and
  I should say so.

---

## 12. Limitations, stated plainly

1. **The architectural precondition is strong.** 2NRL needs an inversion operator
   that is exact, cheap, involutive and order-reversing. Most architectures do
   not have one. Whether this ports beyond signed bilinear scoring of the kind I
   use is unknown, and I will not claim it does.
2. **Inversion is global, and therefore blunt.** It reverses the correct parts of
   the model along with the incorrect ones and leans on phase 3 to repair the
   damage. The system already half-admits this: there is a *local* variant that
   flips alternating nodes along one failed path, used instead of a global
   inversion when a failure is severe but isolated. They are alternatives, not
   layers — which tells me the global operator is not always the right tool.

   The local variant is also **provably** incomplete, and the proof is short
   enough to give. Flipping every edge of a path means two-colouring it, and a
   graph is two-colourable only if it contains no odd cycle. A cyclic graph can
   close one. On a 3-cycle the best any flip pattern manages is two edges out of
   three; on a self-loop, where `score(p → p) = w · f_p²`, flipping the node
   changes the score by *exactly nothing*. The obstruction is exact, not
   approximate. Allowing cycles is what turns an operation that is clean on a DAG
   into one that is best-effort — the same trade I made in
   [the cycles paper](CyclesAreAFeature.md), showing up somewhere else.
3. **No stopping rule.** "Fail, invert, repeat" describes a loop. In a life it
   ends when the thing is learned. My self-improvement loop runs indefinitely
   with no convergence criterion and no held-out evaluation, and its only signal
   is a score gap from its own discriminator — which can widen while the output
   gets worse. I know when I stopped: when the thing worked. I cannot yet make
   that computable, and §6.4 is the same problem wearing a different hat.
4. **Phase 1 depth is unexamined.** I train on the garbage for a fixed small
   number of epochs. The entropy account in §5 implies phase 1 should run until
   the failure mode is *well* represented, since a half-learned failure inverts
   into a half-useful signal. There is probably an optimal depth and it probably
   depends on `H(q)`. I have not looked.
5. **n = 1 on the human side.** The origin of this is my account of my own
   learning. That is a real existence proof — the procedure demonstrably produced
   a working education at least once — and it is not a controlled result about
   how people learn in general. I offer it as the *source* of the algorithm, not
   as evidence for it. The algorithm has to stand on §11.

---

## 13. Summary

- Train on the failures at full rate, invert the network, fine-tune on the
  correct data gently. That is 2NRL, and it is how I taught myself, written down.
- The name is the mechanism. **Double-Negative Reinforcement Learning**: trained
  **on** the failure, then negated, and a positive phase after. Two negatives
  make a positive, which is exactly why inversion gives a destination where
  repulsion only gives a direction.
- Moving *away* from a wrong answer is under-determined: it names a direction to
  leave and no place to arrive. Going all the way into the failure and negating it
  turns that into a destination.
- Inversion is two sign flips — `W → −W`, `a → −a` — because the edge signal is
  the product of the weight and both endpoints' activations. It is exact, cheap
  and its own inverse. The sine's sign parameter is what makes it possible at all.
- The condition is **consistency**, and it was in my own sentence before I
  understood why. What inverting a failure can recover is bounded by that
  failure's negative entropy. Systematic failure inverts into signal; random
  failure inverts into nothing.
- So the question to ask of a negative set is not "are these wrong?" but "are
  these wrong *in the same way*?"
- The negative phase has since been made permanent: a standing network that
  models how text goes wrong, which makes failure queryable, makes the entropy
  condition measurable, and lets blame be placed on the characters that were
  actually wrong.
- "Pull hard on the thread" is not a learning rate. It is a schedule over search
  breadth — wide, then narrow once something moves — and I have not built it. The
  trigger is settled, though: narrow when the network is **rewarded**, or when it
  reaches a **known area of completion**. Both are events the system already
  emits, not estimates it would have to infer.
- A game is rules plus payoffs, and you start out knowing neither. The usual
  move is to go after the payoffs by sampling them. I go after the **rules**, by
  losing on purpose — which is how I learned this in the first place. A rule is
  invisible while you obey it: a legal move tells you almost nothing, an illegal
  one tells you exactly where the line is. Failure is not the cost of learning
  the rules, it is the instrument.
- Which is why the curve is a hockey-stick and not a climb. A long flat stretch
  that looks like failing is rule-discovery; then the rules collapse the action
  space from *anything at all* down to *the legal moves*, and the rest goes fast.
  The flat part is buying the collapse. That is the honest version of "explore
  widely, then pull hard": narrowing works because there is less left to search.
- So the acceptance criteria the helper writes before play **are the rules,
  stated up front**, and that is its whole job — not teaching. Rules stated in
  advance make failure *rule-shaped* instead of random, and §5 says only
  consistent failure inverts into anything. A helper that solved the task would
  remove the failure, and the failure is the input.
- It has never been measured against a baseline. §11 is the experiment, and arm C
  is the one that would tell me I am wrong.

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

- [`SineWaveActivationFunction.md`](SineWaveActivationFunction.md) — why every
  unit carries a sign parameter, which is what makes §4.3's inversion operator a
  parameter change rather than a structural one. Without it 2NRL cannot run.
- [`CyclesAreAFeature.md`](CyclesAreAFeature.md) — why inverting a *path* is a
  two-colouring problem, and why odd cycles make it best-effort rather than exact
  (§12, item 2).
- [`README.md`](README.md) — how the three papers depend on each other.
- `RadixCyclicNN/DECISIONS.md` — D-009 (2NRL), D-010 (inversion), D-027
  (proportional boosting), D-028 (local inversion), D-045 (the negative network),
  D-067 (the breadth schedule I have not built).
- `RadixCyclicNN/DESIGN.md` §8, §9.1 — the specification.
