# DISTIL — Design Specification

**D**istillation **I**nto **S**ubgoals, with **T**ool **I**nduction and **L**earning.

A self-improving agent whose memory is an embedding layer, whose reasoning is
game-theoretic goal distillation, and whose capability grows by writing tools it
then has to prove work.

Directory: `DISTIL/`. Python package: `distil`. Python 3.11+, standard library
only. This document is the contract every module is implemented against.

> Understand the game. Distil it into goals. Question every premise. Take no
> refusal as final. Remember what happened, graded. Do it again, better.

---

## 0. The ordering, which is the design

Nine subsystems, but only one thing has to be remembered about how they fit:

```
FRAME ──▶ AGENDA ──▶ PRECEDENT ──▶ WHY ──▶ CHALLENGE ──▶ RETRIEVE
                                                            │
        CREDIT ◀── VERIFY ◀── ACT ◀── SELECT ◀── PAYOFF ◀── DISTIL
           │                                                  
           └──▶ MEMORY (graded) ──▶ COMPRESS ──▶ EXPLORE ──▶ SELF-EDIT
```

`FRAME` is first and the rest is downstream of it, because *which solution
concept is valid depends on what game this is* (§2). `CREDIT` closes the loop
back into memory, which is why there is no separate training phase: the loop that
answers and the loop that learns are the same loop.

| Requirement | Where it lives |
|---|---|
| An embedding layer that **is** memory — queries, tools, information, everything | `memory.py`, `embed.py`, `store.py` (§3–4) |
| Self-graded where possible, user-graded otherwise | `grade.py`, `sandbox.py` (§5) |
| Chain-of-thought built on game theory | `reason.py`, `game.py` (§6–7) |
| Continuously distil tasks into goals | `goals.py` (§8) |
| Question everything, ask why, never take no for an answer | `challenge.py` (§9) |
| Explore, brainstorm bad ideas, test them | `explore.py` (§10) |
| Write tools in Python, register them and what they solved | `toolsmith.py` (§11) |
| Invent by analogy; inversion as a learning method | `explore.py` §10.3 |
| Understand the game first — everything is a game | `frame.py` (§2) |
| Record past problems and their solutions | `casebook.py` (§12) |
| Compress by summarisation, keep details, track access | `compress.py` (§13) |
| Two-step memory: Redis short-term, Postgres long-term | `store.py` (§4.3) |
| Self-editing and self-upgrading | `selfedit.py`, `policy.py` (§14) |

---

## 1. Is this just RAG?

Partly, and the honest answer is more useful than a defensive one.

**The retrieval half is RAG.** Embed, index, nearest-neighbour, put the results in
front of the model. The kNN mechanics here are standard and there is nothing
clever about them; §4 even recommends handing them to pgvector.

**The write half is not, and that is where the system lives.** Four differences,
all on the write path:

1. **The corpus is self-written.** Classic RAG retrieves from a human-authored
   corpus that the system does not modify. Every trace here is produced by the
   system's own activity — the query it was asked, the goals it distilled, the
   chain it reasoned through, the tool it wrote, the error the interpreter
   returned, the idea it abandoned.
2. **Every entry carries a grade, and recall ranks by it.** Retrieval is
   `similarity × credibility × recency` (§4.2), not similarity. A vector store
   answers "what is closest?"; that question returns the answer that looked right
   and did not compile.
3. **Being wrong demotes rather than deletes.** A refuted answer stays
   retrievable at low rank, which is the difference between "nobody tried that"
   and "that was tried and it failed" — the second is worth far more, and an
   ungraded store cannot represent it.
4. **Credit propagates backwards.** One verified outcome re-weights the whole
   path that produced it, over the link graph, decayed per hop (§4.6). RAG has
   no write-back from outcomes at all.

Add that reasoning chains and tools are themselves first-class retrievable
objects, so recall returns *procedures*, not only passages. The short version:
**the embedding layer is ordinary; the grade on it is not.**

---

## 2. Understand the game before playing it

Everything is a game — chess, poker, a lease negotiation, shipping a parser. What
differs is *which* game, and that decides which solution concept is even valid. A
minimax play in a non-zero-sum game is a mistake; a cooperative play in a one-shot
zero-sum game is a donation. So the first object built is a `GameFrame` over seven
axes:

| axis | why it changes the play |
|---|---|
| players | solitaire, versus Nature, versus an adversary, or n-party |
| actions | the move set — what may legally be done |
| information | perfect, imperfect, or incomplete |
| payoff | zero-sum, common-interest, mixed-motive |
| horizon | one-shot or repeated — the single biggest lever |
| chance | deterministic or stochastic |
| referee | what says no, and how fast |

**2.1 The mapping.** `frame.classify` sends a frame to a named solution concept.
Order matters: incomplete information first (unknown payoffs mean no equilibrium
concept applies yet — form beliefs before optimising), then repeated play (the
folk theorem makes cooperation enforceable in a way the one-shot game cannot),
then the payoff structure.

**2.2 The referee is the most valuable axis.** It decides whether anything in this
game can be self-graded at all. A frame that reports *no referee* is reporting
that §5's verifier has nothing to attach to, and the agenda says so in those
words rather than proceeding as though grading were available.

**2.3 The play threshold.** `GameFrame.understood` is false below confidence 0.5
or with no objective or no move set. `GREN/DESIGN.md` §22.1 enforces the same
refusal for game packages handed to GTMNN, and for the same reason: a system that
plays before it understands is guessing with extra steps.

**2.4 Capabilities and the agenda.** `capabilities()` intersects what the game
requires with what this system can do — eight primitives (recall, interrogate,
challenge, distil, forge, check, experiment, select) plus every registered tool.
An action nothing covers is a **capability gap**, and gaps head the agenda,
because a move you cannot make is not a plan. The ordering is:

1. If the game is not understood, establish it. Work against the wrong frame is
   the most expensive kind.
2. Close the capability gaps — by forging a tool, which is how this system
   acquires a move it did not have.
3. Then play, in the order the solution concept chooses.

**2.5 Games are retrieved by structure, not subject.** A frame embeds its
players/information/horizon, so two tasks with nothing topical in common cluster
when the same solution concept applies to both. Identification against a corpus
costs `log2|G|` bits where induction from nothing costs `log2|Θ|` —
`GREN/DESIGN.md` §3's argument, applied to tasks rather than board games.

---

## 3. The embedding layer is the memory

Not a cache in front of one. One store, one vector space, one recall path, for
queries, answers, goals, reasoning chains, tools, facts, failures, ideas, games,
solutions and digests. There is no separate "tool index": a tool and the bug it
once fixed are retrieved by the same query, because at the moment of recall they
answer the same question — *what do I know that bears on this?*

**3.1 Two embedders.** `HashEmbedder` is lexical — signed hashing trick over words
and character 4-grams, sublinear tf, online idf. It has no idea "compile" and
"build" are related. It is deterministic, free, offline, and for the traffic this
system actually stores (identifiers, error strings, paths, stack frames) the
lexical match *is* the right match: `TypeError: unhashable type` should retrieve
by those exact tokens. `ProviderEmbedder` is semantic and earns its round trip on
prose.

**3.2 Every hash is blake2b.** Python's `hash()` for strings is salted per
process, so a vector written today would not match the same text embedded
tomorrow. This is the single most important line in `embed.py`.

**3.3 Signed hashing.** Unsigned hashing makes every collision constructive — two
unrelated tokens in one bucket always inflate similarity. With a sign bit from an
independent digest, collisions cancel in expectation. At 512 dimensions the
collision noise floor sits near cosine 0.17 for short texts while genuine matches
land at 0.25+; that gap is workable but narrow, and `frame.py` matches one-word
actions by exact surface form rather than relying on it.

---

## 4. Recall, and what makes it more than a vector search

**4.1** Vectors are unit-length `list[float]`, so cosine is a dot product. Exact
search is the default and is honest about its ceiling: ~100k traces, past which
the answer is an index, not a faster loop.

**4.2 The ranking.**

```
score = similarity^ws · credibility^wc · recency^wr
```

`credibility` is a shrunk posterior over the trace's grades — mapped to [0,1],
pulled toward a prior with pseudo-counts, user grades weighted double. Shrinkage
is not decoration: without it a single lucky `+1` outranks a memory verified
twenty times and recall becomes a recency lottery.

**The claim this rests on**, and it is tested: on a store holding a correct and an
incorrect answer to the same question, **similarity alone returns the wrong one**.
Credibility weighting returns the right one, despite it being *less* similar.

**4.3 Dedupe, and when it is wrong.** Near-duplicate writes merge and increment
`seen`. But two *different solutions to the same problem* embed almost
identically — the problem text dominates both vectors — so cosine dedupe merges
them and averages a `+1` and a `-1` into an opinionless `0`. That is
catastrophic precisely where it matters most, so records with a natural key pass
an `identity` and dedupe becomes exact (§12).

**4.4 Cross-backend comparison is refused.** Every trace records the embedder that
produced it; cosine between a lexical vector and a model vector is meaningless and
recall will not compute it.

**4.5 Approximate indexes must over-fetch.** ANN returns top-k by *distance*, then
this system re-ranks by credibility. A verified trace that ranks 12th by cosine is
invisible to a re-ranker that only sees the top 5. Approximate stores declare
themselves and `Memory` over-fetches 10× before re-ranking. Getting this wrong
produces a store that silently stops surfacing its best material as it grows.

**4.6 Credit propagation.** A grade pushes a decayed share back along `links`,
breadth-first with a visited set (a cycle terminates rather than amplifying) and a
noise floor below which the walk stops. Propagated grades keep the originating
*source*, so an opinion cannot launder itself into evidence two hops away.

**4.7 Backends.** `InProcessStore` (exact, default, tested) and adapters for
**pgvector** and **Redis**. Postgres is the recommended one: the ranking needs
similarity, credibility and recency, and in Postgres all three are columns of the
same row, so the score is one SQL expression and grade updates are transactional
against the vectors they modify. Redis cannot express that in-query, so it
over-fetches and re-ranks client-side.

**4.3 (tiered)** `TieredStore` puts Redis with a one-week TTL in front of Postgres.
The system writes far more than it should keep, so **expiry is automatic and
retention is earned**. A trace is promoted to long-term when its kind cannot be
lost (tools, cases, games), when it has been graded either way, or when it has
been recalled again — the store's own evidence that something wanted it. Every
write refreshes the TTL, so the week measures time since last *touch*. Redis as a
plain read-through cache is supported and is the wrong default: caching makes
reads faster without making the store smaller, and unbounded low-value
accumulation is the actual problem.

---

## 5. Grading, and the discipline of only grading what can be graded

**5.1 Self-grading** runs where a verifier exists. For Python it is real and free:
`parse → screen → run → contract tests → determinism`, five stages weighted into
`[-1, 1]`. Nothing about it is an opinion. Failing the contract tests scores worse
than having none, and having none is never a pass — an unverified tool must not
score the same as a verified one. Determinism is worth 15% on its own, because a
tool that is right intermittently is worse than one that is wrong consistently:
the second gets fixed.

**5.2 User grading** is everything else, and counts double in credibility — a
human bothering to grade is a stronger and rarer signal than a check firing.

**5.3 What is deliberately not done.** No LLM-as-judge. A model scoring a model's
prose produces a number with the *shape* of evidence and the *content* of a second
opinion, and once in the store recall cannot tell it from a passing test. So
`gradeable()` returns False for prose and the trace stays `ungraded`, sitting at
the credibility prior — neither promoted nor buried. That is an honest state.

**5.4 The sandbox is containment, not security.** Separate interpreter, fresh temp
cwd, wall-clock timeout with `kill()`, rlimits where the platform has them, an
environment stripped of credentials and proxies, and an AST screen that refuses
destructive source before it runs. It is **not** a security boundary: a subprocess
shares the kernel, and `getattr(__builtins__, 'ev'+'al')` defeats any denylist.
The screen exists to catch a model that cheerfully wrote `shutil.rmtree('/')`
because the docstring said "clean up" — which is the failure that actually
happens. For a real boundary, run the package in a container with no network.

---

## 6. Chain of thought as typed steps

Free-form CoT has one disqualifying property: it cannot be checked. It is prose
narrating a decision already made, and when it is wrong it is wrong persuasively.
Every step here is a typed object with its inputs attached, which buys **replay**
(re-execute against a changed memory; the step where the answer diverges is the
one that mattered), **retrieval** (chains are embedded, so the system recalls
*how it reasoned* about something similar), and **attribution** (SELECT carries
the payoff matrix it chose from, so "why that goal?" has a numeric answer).

---

## 7. Game theory as the decision procedure

Not an analogy. Planning under uncertainty **is** a game against Nature: rows are
goals, columns are states of the world, entries are payoffs estimated from memory.

**7.1 Nature's moves**: `as-stated`, `underspecified`, `harder`, `wrong-frame`,
`tool-missing`. Drawn from what actually goes wrong.

**7.2 The payoff table** is indexed by a goal's *role* — build, clarify,
interrogate, tool, check. The shape of it is the claim: BUILD pays best when the
task means what it says and is *actively harmful* when the frame is wrong, which
is the cost of building the right thing for the wrong problem. CLARIFY and
INTERROGATE are cheap insurance: mildly wasteful when all is well, decisive when
it is not. CHECK never hurts, which is why a regret-averse system drifts toward
verifying things.

**7.3 Beliefs come from memory and the interrogation.** Laplace-smoothed
frequencies of recorded outcomes on similar tasks, then shifted: a why-chain that
bottomed out CIRCULAR is direct evidence for `wrong-frame`; sharp open questions
are evidence for `underspecified`. This is the one place the interrogation layer
moves a number instead of producing commentary.

**7.4 The rule.** `game.choose` blends expected utility and minimax regret by
`policy.regret_aversion`. Minimax regret is the default because regret is the
quantity memory can actually estimate — every trace records what happened next.

**7.5 Implemented and tested against closed forms**: maximin, minimax regret,
Hurwicz, expected utility, fictitious play (matching pennies → ½,½; value 0),
regret matching (RPS → uniform), and Monte Carlo Shapley (additive game →
individual values; efficiency; null player; glove game → ⅔, ⅙, ⅙).

**7.6 Shapley for credit** over subgoals, not "the last step before it worked" —
the last step is what every naive scheme credits and is almost never the one that
mattered. Efficiency guarantees the shares sum to the outcome, so credit cannot be
conjured. `CyclicCortex` applies the same mechanism over regions; this applies it
over subgoals.

---

## 8. Distillation stops at verifiability

> A goal is atomic when a check can be written that decides whether it has been
> met.

Not when it feels primitive, not at a fixed depth. That rule terminates the
recursion on an objective criterion, guarantees every leaf is gradeable, and makes
"I don't know how to decompose this" and "I don't know how to check this" the same
complaint — which they always were.

The consequence is the useful part: an under-specified task **cannot** be distilled
to atoms. It bottoms out in goals nobody can write a check for, and that list is
the specification gap, named clause by clause. It is where §9 is pointed.

---

## 9. Question everything; never take no for an answer

Three mechanisms, none of them a prompt telling a model to be sceptical. A prompt
that says "be rigorous" produces prose that sounds rigorous; these produce objects
that can be checked.

**9.1 The why-chain.** Ask why, then why of the answer. What matters is the
classification of *where it stops*: GROUNDED (checkable, and here is the check),
ASSUMED (an axiom or a preference — not an insult, most chains should end there),
or CIRCULAR (the justification restates something already in the chain).
Circularity is detected by **embedding against ancestors**, not string equality,
because restating a claim in new words is how justifications actually loop.

**9.2 Premise attack.** Extract what a task takes for granted, attack each along
six axes — definition, necessity, evidence, alternative, scope, cost — then rank
by **expected information gain** and ask only the top few. This is what keeps
scepticism from becoming noise: a question whose answer you can predict is
rhetoric, not enquiry.

**9.3 Persistence.** "Never take no for an answer" needs a stopping rule or it is
an infinite loop with ambition:

> Keep re-attacking while the refusals are still *new*. Stop when they repeat.

Every refusal is embedded. A refusal near-identical to one already collected has
taught nothing, and a second identical no means the boundary has been found — not
that persistence failed. Until then the goal is reframed: relax the constraint,
substitute the means, decompose, invert, generalise, or question the refuser.
Order is fixed: RELAX first (the constraint may not be real), QUESTION_ORACLE last
(the checker is usually right, and assuming otherwise first is how a system talks
itself past a correct no).

**The deliverable of an unsuccessful run is the boundary.** A run that never
succeeds returns a description of why the goal is out of reach — which is a
result, and the one thing a system that stops at the first no can never produce.
This is `GREN/DESIGN.md` §2 turned from an observation into a control loop.

---

## 10. Exploration: curiosity, bad ideas, analogy, inversion

**10.1 Curiosity is a why-chain that did not bottom out.** Every chain ending
ASSUMED or CIRCULAR is an unexamined premise sitting in memory — a *known* hole
with a *known* location. Those are the first source of experiments.

**10.2 Bad ideas are the high-information ones, and this is not a metaphor.** The
value of an experiment is the entropy of its outcome, `H(p)`, maximised at
`p = 0.5`. An idea you are sure will work teaches nothing — and an idea you are
sure will *fail* teaches nothing either, which is the half people skip. Candidates
are ranked by `H(p)/cost × novelty`, and the system manufactures ideas it expects
to fail about half the time. `GREN/DESIGN.md` §4 derives the same 0.5 operating
point for probe policies; this is that result pointed at self-improvement.

**10.3 Invention by analogy, over a similarity *band*.** Pairs above ~0.6 are the
same thing said twice — transferring between them invents nothing. Pairs below
~0.22 share no structure to carry. The productive distance is the middle, and
nearest-neighbour retrieval is structurally incapable of finding it because it is
built to return the top of the range. The source is always something that
**worked**; the target is something unresolved. That direction matters:
transferring a known-good approach onto an open problem is invention; the reverse
is noise.

**10.4 Inversion.** A belief held with high credibility whose converse has never
been tested is a belief indistinguishable from a convention. Inverting it is cheap
and occasionally it holds — which is the only way that error is ever found.

**10.5 A claim is not tested by a function.** Curiosity and inversion ideas
*assert* something; gaps and proposals *propose building* something. A template
that builds a working function cannot refute "X is false", so pairing them would
record a confirmation nothing established. Offline, claims go untested and say so.
**An experiment tested by an unrelated check is worse than an untested one: it
manufactures evidence.**

**10.6 A failed experiment is a stored result.** Every refutation becomes a
`FAILURE` trace with its diagnostic: it stops the idea being re-derived, it gives
§7.2 evidence that pulls similar plans down, and it is raw material for §10.3.
The store gets better by being wrong, on the record.

**10.7 Regret matching over idea sources**, so the system learns which generator
earns its keep — rewarded for *information*, not success. A generator rewarded for
being right would stop proposing the ideas worth running. Unplayed arms are
credited with their own running mean, or the sampler collapses onto whichever
generator happened to fire first.

---

## 11. Tools

A tool enters the toolbox only by passing §5. There is no "probably fine" path:
the entire value of a tool over re-deriving the answer is that it has *already
been checked*, and an unchecked tool is a confident wrong answer with a name.

Contract tests are required and written by the same author as the tool. Not ideal
— an independent test author is better — but it is the honest trade, and the
failure it leaves (a test asserting the bug) is visible in the stored source
rather than hidden in a score.

**What gets embedded is purpose + signature + every problem the tool has solved.**
That is what makes retrieval work: a tool called `normalise_rows` is unreachable
to anyone who did not already know it exists; the same tool carrying "stripped
whitespace from a ragged CSV" is found by whoever has that problem next.

**Reuse does not fabricate arguments.** A goal says what is wanted, not what to
pass — "compute the median of a column" carries no column. An earlier version
invoked the match with an empty argument list, which raised `TypeError` for every
tool taking a parameter and made reuse silently impossible. Inventing arguments to
get a green result would have been the worse fix: it is exactly the fabrication
§5 exists to prevent.

---

## 12. The casebook: problems, and what solved them

The tool registry answers "what can I do?". The casebook answers the question that
comes up far more often: **"what did I do last time something looked like this?"**

Both halves of a case are embedded into one vector, so a case is reachable from
either end — by a problem resembling the problem, or by a solution resembling the
solution. The second direction is how one fix gets applied to the four other
places it applies to.

**12.1 Refuted cases are kept and ranked below.** "That has been tried and it does
not work" is worth more than "nobody has tried that", and an ungraded store cannot
tell them apart.

**12.2 Retrieval is two-stage.** The blended vector fans out; ranking is then
similarity to the recorded *problem* alone × credibility. The blend is wrong for
this query: asked "what looks like this problem?", a case whose *solution* shares
wording with the question outranks a case whose problem is the same one. The first
version did exactly that.

**12.3 A retrieved case reports what differs.** A precedent applied without
noticing what changed is how the right answer to last month's question becomes
this month's outage.

---

## 13. Compression: consolidate by access, not by age

An append-only memory gets slower and vaguer as it grows. So there is a
consolidation phase, and its rule is **compress by access**.

Age alone is wrong: a fact consulted daily for two years is old and is the last
thing to summarise; an idea written an hour ago and never re-read is new and is
the first. `Trace.heat` — saturating hit count × recency decay — means *neither
recently wanted nor often wanted*.

**13.1 Detail preservation is the part that is easy to get wrong.** A summariser
producing fluent prose loses exactly what made the traces worth keeping. So a
digest keeps: the highest-graded member **verbatim** (not a paraphrase); rare
terms ranked by IDF, because a high-IDF token is by construction what
distinguishes these traces from everything else (`ValueError`, `5432`,
`--no-verify` survive; "the" and "should" do not); member counts and grade
statistics; and a prose abstract **only if a provider is available**, appended to
the extractive core rather than replacing it. A model outage degrades readability,
never content.

**13.2 Nothing is destroyed.** Originals are appended to `archive.jsonl` before
leaving the store, and `restore()` brings them back. Compression means "moved out
of the retrieval path", which is what makes it safe to run automatically and makes
"maintain details" a property rather than a hope.

**13.3 Never compressed**: tools, cases, games. Summarising the accumulated
capability and judgement would be compressing the thing the rest of the store
exists to serve.

---

## 14. Self-editing and self-upgrading

Two surfaces, deliberately different in kind.

**14.1 Policy tuning** (`policy.py`): every number the system may change about
itself, each with an explicit range. A tuner that can set a weight to 1e9 has not
been tuned, it has been broken. Mutation is coordinate-wise, because attributing
an improvement to a change requires the change to be attributable.

**14.2 Self-editing** (`selfedit.py`): the system rewrites its own source. Four
things must be true for an edit to land — shadow copy, invariants, the **existing**
test suite, and a snapshot for rollback. The suite that runs is the one on disk,
not one the edit supplied; that distinction is the entire value of the step.

**14.3 The invariant that matters:**

> An optimiser that can edit its own grader will edit its own grader.

Every stage of self-improvement is scored by `grade.py`. Nothing stops a
well-meaning "improve the pass rate" from rewriting the weights so everything
passes, producing a system that reports monotone improvement while getting worse —
with no external signal to contradict it, because the signal *is* what was edited.
So `grade.py`, `sandbox.py` and `selfedit.py` are fixed points, and the test count
may not fall. **The system can rewrite any of its reasoning, retrieval, framing,
exploration or synthesis code. It cannot mark its own homework.**

**14.4 This was not hypothetical.** The first thing self-upgrade did was find a
hole in its own objective: the original `score()` averaged the grades of top hits
without conditioning on relevance, so a policy that *ignored the query* returned
the best-graded traces for every task and scored **0.61 against 0.24**. The fix
makes each hit contribute `grade × similarity`, so well-graded irrelevance earns
nothing. Both the hole and the fix are regression-tested.

---

## 15. Providers

`Ollama`, `OpenAI` (and any OpenAI-compatible server via `base_url`), `Anthropic`,
and `LocalProvider` — over `urllib`, no vendor SDKs. The seam is two methods, so
provider concepts cannot leak upward and comparing models on one task is a loop
over a list. Anthropic declares `can_embed = False` because there is no
first-party embeddings API — stated plainly rather than discovered as a 404
mid-run.

`LocalProvider` is not a mock. It is a deterministic offline decomposer
implementing the same interface with rules, and it is what the test suite runs
against. The system must be runnable, gradeable and improvable with no key and no
network, because a self-improvement loop whose every iteration costs money and
latency will never be run long enough to improve anything.

---

## 16. What this does not do

Stated because a specification that only lists strengths is marketing.

- **The offline synthesiser generalises to nothing.** It covers a fixed table of
  goal shapes and returns `None` outside it. That is correct behaviour — an
  unrecognised goal should reach a real model, not receive plausible code that was
  never going to run — but it means the offline loop demonstrates the *mechanism*,
  not the capability.
- **The lexical embedder does not understand synonyms.** "compile" and "build" are
  unrelated to it. Provider embeddings fix this; the default trades it for
  determinism and zero cost.
- **The sandbox is not a security boundary** (§5.4).
- **Prose is not graded** (§5.3), so most non-code work accumulates as `ungraded`.
  This is honest, and it is also a real ceiling: the system improves fastest at
  things a computer can check.
- **Postgres and Redis adapters are unexercised.** There is no database in the
  environment this was built in. The SQL and the index definitions are written out
  to be read; the promotion *rule* is tested with in-process stand-ins.
- **Self-editing is guarded, not safe.** The invariants stop the failure modes
  that were anticipated. They are not a proof.
