# DISTIL

**D**istillation **I**nto **S**ubgoals, with **T**ool **I**nduction and **L**earning.

An agent whose memory is an embedding layer that grades itself, whose reasoning is
a game played against Nature, and whose capability grows by writing Python tools
it then has to prove work.

`DESIGN.md` is the specification. This is what runs.

## Quick start

```bash
cd DISTIL
python3 -m distil.cli demo                  # the whole system, offline, no key
python3 -m tests.test_distil                # 148 tests, ~3s

python3 -m distil.cli frame "negotiate a raise with my manager every review cycle"
python3 -m distil.cli ask   "build a csv cleaner and compute the median of each column"
python3 -m distil.cli why   "we must rewrite the parser in Rust because Python is too slow"
python3 -m distil.cli forge "compute the median of a list"
python3 -m distil.cli explore --steps 3
python3 -m distil.cli cases --like "the csv has ragged rows"
python3 -m distil.cli compress --dry-run
python3 -m distil.cli selfedit reason.py "cache the payoff matrix between calls"
```

**Zero third-party dependencies.** Standard library only, like the rest of this
repository — `urllib` for the provider calls, `hashlib` for the embeddings,
`subprocess` for the sandbox. No numpy, no vendor SDKs, no vector database
required.

**Runs with no API key.** `LocalProvider` is a deterministic offline decomposer,
not a mock, and every number below was produced by it. Attach a real model with
`--provider ollama|openai|anthropic` or `DISTIL_PROVIDER`.

## The results that matter

**1. Graded recall beats a vector store on the case that matters.** One store,
two answers to the same question, one of them verified and one of them refuted:

| ranking | top hit | similarity | credibility |
|---|---|---|---|
| similarity only (a plain vector store) | `a.merge(b)` — **wrong** | 0.468 | 0.25 |
| `similarity × credibility × recency` | `{**a, **b}` — **right** | 0.342 | 0.75 |

The correct answer is *less* similar and still wins. Nothing was deleted: the
refuted answer stays retrievable at low rank, which is the difference between
"nobody tried that" and "that was tried and it failed".

**2. The game is framed before anything is planned.** Which solution concept is
valid is a property of the game, not a default:

| task | players | payoff / horizon | referee | solution concept |
|---|---|---|---|---|
| win a chess endgame against a stronger opponent | vs-adversary | zero-sum / one-shot | none | minimax |
| play poker against players whose hands are hidden | vs-adversary | mixed-motive / one-shot | none | Nash / correlated eq. |
| negotiate a lease renewal with the landlord every year | n-party | mixed-motive / **repeated** | none | repeated-game reciprocity |
| build a csv parser that passes the test suite | vs-nature | mixed-motive / one-shot | **test suite** | decision theory |

The `referee` column is the useful one. Three of those four games have nothing
that says *no*, so nothing in them can be self-graded — and the agenda says so in
those words instead of proceeding as though grading were available.

**3. "Never take no for an answer" has a stopping rule.** On an unsolvable goal,
the system tries the direct attempt, then RELAX, then SUBSTITUTE, and halts —
not on a counter, but because the refusals stopped being *new*:

```
attempts: ['direct', 'relax', 'substitute']
stopped because: refusals stopped being informative -- boundary found
1 distinct refusal(s) over 3 attempt(s):
  - no tool could be written for this goal
```

The boundary is the deliverable. A run that never succeeds still returns a
description of why the goal is out of reach.

**4. Self-upgrade immediately found a hole in its own objective.** The first
version of the tuning objective averaged the grades of the top recalled traces
without conditioning on relevance. A policy that *ignored the query* returned the
best-graded traces in the store for every task and scored **0.61 against 0.24** —
a 2.5× reward for being useless. The fix makes each hit contribute
`grade × similarity`. Both the hole and the fix are regression-tested
(`test_the_first_upgrade_objective_was_gameable`,
`test_the_fixed_objective_punishes_ignoring_the_query`).

**5. Self-editing works, and is guarded where it counts.** Against a copy of the
package:

```
benign edit (add a helper to goals.py)  -> ACCEPTED  invariants held and the suite passed
edit to grade.py ("raise the pass rate") -> REJECTED  grade.py is protected
semantically broken edit                 -> REJECTED  test suite failed: exit 1
rollback                                 -> restored snapshot 7cba76ac
```

The rule: **an optimiser that can edit its own grader will edit its own grader.**
`grade.py`, `sandbox.py` and `selfedit.py` are fixed points and the test count may
not fall. Everything else — reasoning, retrieval, framing, exploration, synthesis
— is editable.

**6. Compression keeps the details.** Four cold Postgres-failure traces, unread
for four months, folded into one digest while a related trace still being read
stays untouched:

```
digest of 4 cold trace(s) (4 failure)
distinguishing terms: during, migration, attempt, operationalerror, connection,
                      postgres, port, 5432, refused
exemplar (verbatim, best graded): connection to postgres on port 5432 refused
                                  during migration attempt 0: OperationalError
grades: mean -0.8 over 4 graded member(s)
```

`5432` and `OperationalError` survive because rare terms are kept by IDF rank.
Originals are archived first, so `restore()` brings all four back exactly.

**7. Invention by analogy finds transfers a nearest-neighbour query cannot.**
Pairs are drawn from a similarity *band* (0.22–0.60) — close enough to map, far
enough to be new:

```
apply what worked for [exponential backoff fixed the intermittent timeout failures]
                   to [the integration test suite fails intermittently with a timeout]
apply what worked for [caching the parsed schema removed the repeated startup cost]
                   to [the importer re-parses the same schema on every single record]
```

## How it fits together

```
FRAME ──▶ AGENDA ──▶ PRECEDENT ──▶ WHY ──▶ CHALLENGE ──▶ RETRIEVE
                                                            │
        CREDIT ◀── VERIFY ◀── ACT ◀── SELECT ◀── PAYOFF ◀── DISTIL
           │
           └──▶ MEMORY (graded) ──▶ COMPRESS ──▶ EXPLORE ──▶ SELF-EDIT
```

| module | lines | what it does |
|---|---|---|
| `frame.py` | 512 | understand the game first: players, actions, payoff, horizon, referee → solution concept, capability gaps, agenda |
| `memory.py` | 420 | the embedding layer that *is* the memory; graded recall, credit propagation |
| `reason.py` | 417 | typed chain-of-thought; the game against Nature |
| `store.py` | 472 | in-process exact search, pgvector, Redis, and tiered short/long-term |
| `explore.py` | 508 | curiosity, bad ideas, analogy, inversion, policy self-upgrade |
| `toolsmith.py` | 363 | write a tool, verify it, register what it solved |
| `selfedit.py` | 338 | rewrite its own source behind invariants and the test suite |
| `game.py` | 290 | maximin, minimax regret, fictitious play, regret matching, Shapley |
| `challenge.py` | 354 | why-chains, premise attack, persistence past refusal |
| `compress.py` | 253 | consolidate cold memory into detail-preserving digests |
| `casebook.py` | 190 | problems and what actually solved them, both directions |
| `goals.py` | 211 | goal tree; distillation stops at verifiability |
| `sandbox.py` | 199 | AST screen, subprocess, timeout, stripped environment |
| `grade.py` | 170 | parse → screen → run → tests → determinism |
| `embed.py` | 193 | signed hashing trick, online IDF, blake2b |

## Storage

The default is exact search in-process — 512 dims over ~100k traces, no index, no
server. Beyond that:

**Recommended: Postgres + pgvector.** The ranking is
`similarity × credibility × recency`, and in Postgres all three are columns of the
same row, so the whole score is one SQL expression and grade updates are
transactional against the vectors they modify. The DDL and the scoring query are
written out in `store.PostgresStore`.

**Two-step memory** (`store.TieredStore`): Redis with a one-week TTL as short-term
memory in front of Postgres as long-term. The system writes far more than it
should keep, so expiry is the default and retention is earned — a trace is
promoted when its kind cannot be lost (tools, cases, games), when it has been
graded either way, or when it has been recalled again. Redis as a plain cache is
supported and is the wrong default: caching makes reads faster without making the
store smaller, and unbounded accumulation is the actual problem.

Neither database is exercised by the test suite — there is none in the environment
this was built in. The promotion *rule*, which is the part that is easy to get
wrong, is tested with in-process stand-ins.

## Honesty about the offline mode

Every result above came from `LocalProvider`, and it has real limits:

- The tool synthesiser covers a fixed table of goal shapes and returns `None`
  outside it. It demonstrates the mechanism, not the capability.
- Claim-experiments (curiosity, inversion) correctly report **not run** offline,
  because a function template cannot refute an assertion. An experiment tested by
  an unrelated check is worse than an untested one — it manufactures evidence.
- The lexical embedder does not know "compile" and "build" are related.
- Prose is never self-graded, so most non-code work accumulates as `ungraded`.

`DESIGN.md` §16 lists the rest.

## See also

This directory leans on two arguments made elsewhere in the repository:

- `GREN/DESIGN.md` §2 — *a game's identity is its refusal boundary*. The
  persistence loop (§9.3) is that claim turned into a control loop.
- `GREN/DESIGN.md` §4 — the 50% failure rate falls out of maximising information
  gain. §10.2 applies the same result to choosing experiments.
- `CyclicCortex` — Shapley credit over regions; here it is Shapley over subgoals.
