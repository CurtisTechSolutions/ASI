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
| Ask clarifying questions when the objective is not understood | `clarify.py` (§2.6) |
| Embed tools, MCPs, and its own tooling in one layer | `mcp.py`, `toolsmith.py` (§11.1–11.3) |
| A starter toolkit worth building first | `seed.py` (§11.4) |
| A frontend, with visualisations, to use all of it | `serve.py`, `project.py`, `ui/` (§16) |
| Think by clustering the embeddings; create new embeddings constantly | `think.py` (§20) |

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

**2.6 If the objective is not understood, ask.** `clarify.py` is what happens
below the play threshold. It does not guess, and it does not distil: a
well-organised plan for the wrong problem is the most expensive thing this system
can produce, and §7.2 prices it at −0.80.

Three rules keep the asking useful rather than tedious.

*Ask about what is missing, in blocking order.* The questions come from the axes
the frame could not fill, not from generic premise attacks (that is `challenge.py`,
and it is for a task that **is** understood). Objective before referee before
actions before everything else. A system that cannot say what winning means
should not be asking about input formats.

*Stop when the first step is actionable, not when the frame is complete.* Only
two things genuinely block: not being able to state what done means, and having
no move to make. A missing referee is a real limitation — nothing there can be
self-graded — but it is not a reason to refuse to start, and gating on it refused
plain instructions like *dedupe the records*. Everything else is cheaper to
discover by attempting the step than by asking about it.

*Incidentals are asked once; blockers come back.* Re-asking "what are your
inputs?" is pestering. Letting the objective go unasked because it was raised
once and ignored is the loop giving up on the only thing preventing progress.

The distinction between *understood* and *merely restated* is drawn by
`goals.checkability`, not by comparing the objective to the task. `Framer` falls
back to the task text whenever no explicit purpose clause is present, and for an
imperative like *build a csv parser that passes the test suite* that fallback is
correct — the task states its own objective. What separates it from *make the
thing better* is whether the verb admits a check, which is the same line
distillation already uses to decide where it may stop (§8). One definition of
"checkable" in the system, not two that drift.

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

### 8.1 Acting continues until the frontier is empty

Distillation stops at verifiability; *acting* does not stop at the first goal.
`solve` selects a goal from the frontier, acts, verifies, and — if it was met —
selects again from what remains, until the frontier is empty, a goal is
blocked, or a cap is reached. Only then does it report.

This was a defect for the first several rounds of this codebase. One selection
per task meant one goal per task: the second goal of every two-goal task sat
open forever, nothing ever came back for it, and `solved` reported the *goal*
rather than the task — `True` with half the tree still open. The directive
says to distil continuously, and a loop that stops after one leaf is not that.
`solved` is now `tree.solved`, which `GoalTree.mark` propagates upward only when
every child is met.

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

**11.1 One registry, two transports.** A tool exposed by an MCP server and a tool
this system forged in Python are both `Kind.TOOL` traces in one embedding layer,
found by one graded recall, invoked through one `Toolbox.invoke` that dispatches
on `transport`. At the moment of recall the question is *what can act on this?*,
and which process the answer lives in is an implementation detail. Two registries
would just be one more place every caller has to remember to look — and the one
they forget is the one that had the answer.

MCP tools enter **ungraded**. There are no contract tests to run against someone
else's server and its behaviour is not this system's to check, so they sit at the
credibility prior and earn or lose standing through use like anything else.
Registering them as verified would be manufacturing evidence, which is the same
rule §10.5 applies to experiments.

**11.2 The transport is stdio JSON-RPC**, and the handshake is not optional:
`initialize`, await the result, send `notifications/initialized`, and only then
`tools/list`. A server is entitled to ignore `tools/list` before the
notification, and that hang reads as a broken server rather than a missing
message. Everything degrades to a result object instead of raising — an agent
whose memory layer crashes because one of a dozen configured servers is down is
worse than one that records the outage as a fact and carries on.

One implementation note worth keeping, because it cost real debugging time: the
reader must not mix `select()` with buffered reads. `select` polls the file
descriptor while `readline()` reads from Python's text-mode buffer, so when one
read pulls several lines into that buffer — exactly what happens when a server
emits a log line, a notification and a response together — the next `select` sees
an idle descriptor and reports a timeout with a complete response already in
memory. `tools/list` returned empty about a third of the time. A daemon thread
doing blocking reads into a queue has no such split.

**11.3 Tools compose.** A `ToolSpec` may declare `deps`, and `Toolbox.bundle`
inlines them depth-first with a visited set (cycles terminate, diamonds get one
copy). The sandbox runs a single file with no import path back into the workshop,
so composition is by concatenation — crude, and correct: the dependency source
that runs is exactly the source that passed its own contract tests. Validation
bundles too, because grading `spec.source` alone would pass a composite whose
dependency is missing at call time.

**11.4 The starter toolkit.** `seed.py` is what is worth building before anything
has been asked, and the selection criteria come from the design rather than from
taste: it must be decidable (§8), it must be routed through often, and — the one
that matters most — **it should make more things gradeable**.

That last criterion is the highest-leverage one. §19 states the real ceiling:
prose is never self-graded, so the system improves fastest at what a computer can
check. A tool that *creates a referee* therefore buys more than a tool that does
work, because it moves a whole class of goals from `UNCHECKABLE` to `CHECKABLE`
in `goals.checkability`, which is what decides where distillation is allowed to
stop. `assert_schema` turns "the output is well-formed" from an opinion into an
assert. `normalise_error` and `extract_identifiers` turn a wall of diagnostic
text into a stable class label — `GREN/DESIGN.md` §15's point that a structured
code *is* the class.

Seeds are planted through the ordinary validate/register path. Nothing is trusted
for shipping with the package, and a seed that fails its own contract tests is
rejected exactly like a tool the system wrote for itself. That is the only way
the grade on the others means anything.

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

## 16. The interface

A system that reasons in twelve typed steps and ranks by three composed factors
has to be *looked at* to be believed, and a terminal renders a matrix badly. The
frontend (`serve.py`, `project.py`, `ui/`) exists to make each of this document's
claims checkable against a running instance rather than against prose.

### 16.1 What each view has to prove

These are not a tour of the API. Each is the visible form of one claim, and
§16.4 explains why they are turns in a conversation rather than tabs.

| claim | §  | where you see it |
|---|---|---|
| ranking is not similarity | 4 | every reply carries the hits it recalled, decomposed into similarity, credibility and recency with the weights the server used, and says so outright when the top hit is not the most similar one |
| the game is understood before it is played | 2 | the frame — players, payoff, horizon, information, referee — and its confidence, above anything that was planned |
| the choice is a decision under uncertainty | 7 | the payoff matrix, with Nature's states as columns and the belief over them printed on the headers |
| distillation stops at verifiability | 8 | the goal tree badges each leaf verifiable or needing a check |
| a grade is the one input it cannot re-derive | 5 | +1 / 0 / −1 on the answer itself, where the opinion is formed |
| it can work without being asked | 17 | the autonomous loop narrates into the same transcript |
| a bad idea is worth running | 10 | H(p) drawn, with each candidate experiment plotted on it |
| capability is earned | 11 | every tool shown with the grade it passed and the problem it solved, local, MCP and browser alike |
| the system may change its own numbers | 14 | the policy, every bound, and the button that tunes it |

### 16.2 Projection

512 dimensions onto a plane, by PCA — power iteration with deflation, in the
standard library. Classical MDS on a Euclidean distance matrix *is* PCA, and PCA
costs one pass over the vectors where MDS builds an n×n matrix first; at ten
thousand traces that is a hundred million entries to hold for a picture.

Two properties matter more than the method. The start vector is **fixed**, not
random, so the same store always draws the same picture and a point that moved
means a memory moved. And the sign of each component is pinned by the skew of the
projection, because an eigenvector is only defined up to sign and a picture that
mirrors between reloads is a picture a person cannot learn.

The projection runs over the **stored** vectors, never re-embedded text: the
lexical embedder learns its IDF online, so re-embedding an old trace gives a
slightly different vector than the one recall compares against, and a picture of
vectors the system does not use is a picture of nothing. For the same reason each
embedding backend is projected separately — two geometries share no plane.

The axes have no meaning. Distance is meaningful, direction is not.

### 16.3 What the interface is not

It is a local tool. It binds `127.0.0.1` and it drives an agent that writes files
and executes generated code, so exposing it hands anyone who can reach it both.
There is no authentication, and adding a token would suggest a level of hardening
this does not have.

Three defences are cheap enough to have anyway:

1. **Path containment.** A static path is resolved and confirmed to sit under
   `web/` before it is read, so `..` cannot escape into the filesystem.
2. **Host checking.** A request whose `Host` is not a loopback name is refused.
   This is what stops a page from resolving an attacker-controlled name to
   `127.0.0.1` and driving this server on the user's behalf.
3. **Cross-site refusal.** The `Host` check does *not* stop an ordinary CSRF:
   any page may address `127.0.0.1` directly, and the `Host` header it sends is
   then perfectly honest. So `Sec-Fetch-Site: cross-site` is refused, a non-
   loopback `Origin` is refused, and every POST must be `application/json` —
   which is not a "simple request", so the browser has to preflight it and the
   preflight finds no CORS headers here. Without this a page the user merely
   visited could POST `/api/forge` and have this agent write and execute code.
   The attacker could not read the reply; the side effect would already have
   happened.

None of the three is authentication. They close the attacks a browser can be
made to carry out, which is the threat a localhost tool actually faces.

### 16.4 One conversation, not six tabs

The first version of this interface was a row of tabs: Ask, Memory, Recall,
Tools, Explore, System. That is a dashboard, and a dashboard makes you navigate
*to* a system and hold the correlation between its views in your own head. You
could not get from a recall hit to its neighbourhood in the plot, or from a tool
to the case it solved; the link graph was in the data and in no view at all.

Everything is a message now. A question, a tool listing, an experiment run and an
autonomous cycle all land in one transcript in the order they happened, so the
record of what you did and the result of doing it are the same list. The
capabilities that were tabs are slash commands, which is a pattern nobody has to
be taught, and asking a question — the common case — costs no syntax.

Three consequences worth stating, because each was a defect before it was a
design:

1. **The verdict comes first.** Six expanded cards for one question put the
   answer at the bottom, underneath everything that produced it. The reasoning
   still has to be *there* — a chain nobody can inspect is an assertion — but it
   does not have to be in the way, so each section collapses behind a summary
   carrying the number that says whether to open it.
2. **Grading is on the answer.** The mechanism this system rests on is the user
   grade, weighted double. It used to live in a different tab, reachable by
   copying a trace id and finding a point in a scatter plot. A feedback loop with
   four steps of friction is one nobody closes.
3. **Recall rides on every reply.** §4's claim is that ranking is not similarity,
   and it was checkable only if you went looking. Worse, a question stopped at
   the clarification gate returned questions and nothing else — the case where
   "what do I already know about this?" is most valuable answered it least.

### 16.5 Speech

Two routes, and which one runs decides where the audio goes, so the interface
says which before it listens rather than putting a microphone icon on it and
leaving the question open.

A Whisper binary on `PATH` means the recording is converted to 16kHz mono wav in
the browser, POSTed to `/api/transcribe`, and never leaves the machine. Otherwise
the Web Speech API does it — free, instant, and in Chrome *not local*: the audio
goes to Google. For a system whose premise is that it runs on your machine with
no key and no network, that is a contradiction, and hiding it in a tooltip would
be lying by omission.

Ollama is deliberately not in that list. It serves language and embedding models
and does not transcribe audio; pointing this at an Ollama server would fail at
runtime with a confusing error instead of at startup with a clear one.

### 16.6 The one dependency

`ui/` is React built with Vite — the only third-party code anywhere in this
repository, and it is a *build*-time dependency. `web/`, what it compiles to, is
committed, so `distil.cli ui` runs on a machine with no node and no
`npm install`. The server that serves it is `http.server`.

---

---

## 17. Running by itself

Every other entry point waits to be asked. `auto.py` does not: it picks its own
next move, does it, grades the result and folds it back into the same memory
everything else reads from. Nothing in it is a new capability — it is the
existing ones, sequenced by something with an opinion about what is worth doing.

### 17.1 What makes it curious rather than busy

A loop that does the same thing every cycle is a cron job. Six moves compete and
the choice is regret-matched (§7), so the mix is learned from what each actually
returned — and because the system changes underneath it, no-regret learning is
right where a stationary bandit would be wrong.

| move | what it does | what it costs when it is wrong |
|---|---|---|
| QUESTION | why-chain a belief, then attack its premises | cheap; the payoff is finding something resting on nothing |
| EXPERIMENT | run one, ranked by information gain | cheap; a refutation is a stored result |
| BUILD | forge a tool for a capability gap | moderate; a rejected tool is still recorded |
| CONSOLIDATE | fold cold memory into digests | reversible — originals are archived first |
| TUNE | re-fit its own policy | bounded by `policy.BOUNDS` |
| PURSUE | set itself a problem and solve it | the only move that *produces* rather than consumes |

### 17.2 Rewarded for information, not success

A move that confirms what was already believed scores near zero however cleanly
it ran; one that refutes something scores highly. This is the same argument as
§10: rewarding correctness teaches a system to stop proposing the experiments
worth running, which is the failure mode the whole design is arranged against.

Two consequences that only showed up when it was run:

- **A repeated finding pays less than the first.** Offline the provider cannot
  really answer "why", so every chain returns circular — and QUESTION scored
  identically forever, starving the other five. Payoff now decays with how many
  times that terminal has already been seen, which is itself the honest measure:
  the first circular belief is a discovery, the tenth is a pattern you know.
- **An untried move is worth the best average seen, not zero.** With zero, one
  lucky first cycle drove the matcher to never sample the others again.

### 17.3 Running dry

The first five moves all consume pools that empty: there is a last unquestioned
belief, a last known gap, a last cold cluster. When they were gone every cycle
returned "nothing to do" and the loop spun — neither working nor finished, which
is the worst of both.

PURSUE is the answer. It invents a direction and runs the full solve loop on it,
which writes a frame, an agenda, goals, a chain, usually a tool and a case — the
things the other five had run out of. It is a normal competing move *and* it is
forced after two barren cycles, because a loop with nothing to do should change
what it is doing rather than wait to be told.

Directions come from five sources, round-robin so no one of them supplies all of
them, in descending order of how grounded they are in what the system actually
ran into: a disagreement between memories; a tool that has never solved anything;
a composition of two it has; something it failed at before; and — the floor — two
credible memories filed far apart. **That last source cannot exhaust**, which is
the property that matters: pairs are quadratic and every cycle adds to *n*.

On an empty store there is nothing to derive a direction *from*, and a fresh
instance would sit still waiting to be told to do the one thing it can always do.
So PURSUE bootstraps: it plants the starter toolkit (§11.4), twelve tools each
through the grader, which is enough for the disagreement, unused-tool and
composition sources to start producing. That is useful exactly once.

It runs with no `ask` callback, so a direction it cannot state the objective of
comes back gated rather than guessed at. That is not a failure. The gate firing
on a question it set *itself* is the system reporting that the direction was
vague, and the next cycle picks a different one.

**Self-reference is the recurring bug here, and it recurred twice.** QUESTION
writes a trace beginning "why <subject>", which was then eligible as the next
subject — so it asked why about why about why, escaping accumulating each turn.
PURSUE then hit the same shape from a different direction: solving a self-set
task writes goals and failures *containing* that task, and those carry no marker
saying the loop caused them, because the solver wrote them. Anything a loop
writes is input to that loop, and both fixes are the same idea: exclude your own
output, by marker where you wrote it and by subject where something downstream
did.

### 17.4 What it may not do

It cannot edit its own source. `selfedit` is reachable only from the command
line, deliberately: an unattended optimiser with write access to its own grader
will edit its own grader, and every grade in the store becomes meaningless. This
is the same invariant as §14, enforced by not building the door.

---

## 18. A browser it can drive

`auto.py` is rewarded for information, and the largest source of information it
does not already hold is the web. Without a browser every question it cannot
answer from memory terminates as an assumption.

`browser.py` speaks W3C WebDriver over `urllib`. That protocol is HTTP and JSON,
so Selenium buys nothing here and costs the claim that this package runs
anywhere. The actions are registered as ordinary `Kind.TOOL` traces with
`transport="browser"`, so `browser.read` is recalled by the same query that finds
a forged function or an MCP tool — the one embedding layer over every capability
that §11 is about.

They are registered **only when `DISTIL_WEBDRIVER` is set**. A capability that is
recalled and then fails is worse than one that was never offered.

### 18.1 It is not a sandbox

A driven browser fetches whatever it is pointed at and runs whatever that page
contains, in a process this package does not control. There is no URL allowlist
because a convincing one cannot be written — what it may reach is the network's
decision, not this module's.

That is the argument for `docker-compose.yml`: chromedriver runs in its own
container, its port is never published to the host, and the agent addresses it by
service name. The containment is the port mapping, not anything in this code.

## 19. What this does not do

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
- **The Seed literals in `seed.py` are not raw strings.** A valid Python escape
  inside one is processed before the tool source exists, which silently destroyed
  every `\b` word boundary in two tools. `\w` and `\d` survive because they are
  not valid escapes. A test asserts no seed contains a control character; the
  underlying sharp edge remains.
- **MCP is stdio only.** HTTP/SSE servers would be a second transport class, not
  a change to this one. No MCP server is exercised by the suite beyond a local
  fixture written for it.
- **The frontend has no authentication and is not hardened.** §16.3 closes the
  browser-carried attacks. It does nothing about a process already on the machine,
  which can reach the port like any other client. Treat it the way you would treat
  a shell.
- **Action extraction is a verb list with a leading-word fallback.** It will
  never be complete, and a task whose verb it misses gets its first content word
  taken as the move. That is a guess, and the frame's confidence does not count
  it as a settled axis.


---

## 20. Thinking by clustering the embedding space

§3 says the embedding layer *is* the memory. §4 makes it rank well. Neither makes
it a place where reasoning happens — both treat it as a store to read from, which
leaves it a lookup table with good ordering. `think.py` closes that: it clusters
what the system knows, names what forms, and writes the names back, so the space
grows from its own structure rather than only from what the system was handed.

### 20.1 Why a name is worth storing

The centroid of a cluster is a point near every member and identical to none of
them, which is what a concept is. Writing it back gives recall a single hop to a
whole neighbourhood where before it had to be similar to one specific member to
reach any of them. That is the difference between a store that gets *longer* and
one that gets *denser*.

### 20.2 Cluster context compression

A concept whose text is a label is an index entry: it tells you a group exists
and nothing about what is in it, so recalling it teaches nothing actionable. Each
cluster is therefore compressed into its own name, by
`Compressor.summarise_members` — the same code that compresses cold memory
(§13), producing the same three things: the IDF-ranked terms that distinguish
this cluster from the rest of the store, the best-graded member kept
**verbatim**, and the grade spread.

Sharing that implementation is the point. Grouping by structure and grouping by
access are different questions, but *what it means to summarise a group of traces
without losing the specifics* is one question, and two implementations of it
drift. The only difference is the heading, which is a parameter.

**It adds; it never replaces.** §13 folds cold clusters and archives their
members. This leaves every member exactly where it was. A cluster being coherent
is not evidence that anything in it should be forgotten — those are unrelated
properties, and conflating them would lose memories for being *well organised*.

### 20.3 Real clustering

An earlier version picked a trace, took its k nearest, and called that a cluster.
That is seed-dependent: two adjacent seeds give two clusters that are mostly the
same traces, and which you get depends on iteration order. Agglomerative
average-linkage replaces it — disjoint, deterministic, and needing no k, because
how many concepts a store contains is not something the caller knows.

Average rather than single linkage: single linkage chains, so one trace sitting
between two unrelated groups merges them into a cluster whose centre means
nothing. Average asks whether two groups are alike *on the whole*.

Two implementation notes, both found by the clustering silently returning too
little rather than by reading the code:

- **The full pairwise matrix is required.** Storing only pairs above the cut
  loses clusters, because average linkage averages over every cross-pair and a
  missing entry has to mean "low", not "absent". With them dropped, three obvious
  groups came back as one.
- **The cut cannot be a constant.** What counts as similar depends on the
  embedder and the corpus. Measured: the lexical embedder puts unrelated short
  text at ~0.01 and related text at 0.13–0.40; a provider embedder compresses
  into a much narrower, much higher band. The cut is `mean + 0.75σ` over this
  store's own observed pairs, floored — without the floor, a set with no
  structure has a tiny σ, the cut collapses toward the mean, and everything
  merges into one meaningless cluster.

### 20.4 Conjecture, not observation

Nothing written is graded, and every trace carries `derived: True`. These are read
off the shape of the store. They enter at the credibility prior exactly like an
MCP tool nobody has run, and earn credibility only if something later confirms
them.

Same rule as §10 for experiments and §11 for tools, for the same reason: a
derived trace entering as established would be the system manufacturing evidence
about its own contents — the failure the grading layer exists to prevent, turned
inward.

### 20.5 It must not feed on itself

A derived trace is a point in the same space the next pass reads. Without a
guard, the centroid of a set of centroids becomes a concept — the identical
failure that broke `auto.py`'s why-chains twice (§17.3).

Excluding derived traces from the source *pool* is not sufficient, and the
insufficiency is subtle enough to have shipped once. They still occupy slots in
the similarity search, so each pass saw a slightly different neighbourhood and
formed a slightly different cluster; the identity key is built from the members,
so that wrote a near-duplicate instead of merging.

### 20.6 Precision over recall

On a twelve-trace fixture with three known groups it finds two, both pure, no
mixed clusters. The third is four sentences sharing almost no vocabulary — at the
lexical embedder's noise floor, where within-group similarity (0.129) overlaps
across-group similarity (max 0.158).

Lowering the cut to catch it would start merging unrelated groups. That trade is
refused: a cluster becomes a `Kind.FACT` the system then believes about itself,
so a wrong concept costs more than a missing one. Provider embeddings raise the
recall; nothing here lowers the bar to chase it.

### 20.7 What it does not do

It does not name concepts well. `_shared_words` takes the words common to a
cluster, which is a label, not an understanding; a real model would do better and
this deliberately does not require one.

Clustering is O(n³) worst case, which is why the pool is capped at 400 by
credibility. `skipped()` reports what the cap left out, because clustering a
sample and calling it the store would be a quiet lie.
