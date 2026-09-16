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
python3 -m tests.test_distil                # 243 tests, ~17s

python3 -m distil.cli seed                  # plant the starter toolkit (12 verified tools)
python3 -m distil.cli clarify "make the thing better"    # it asks instead of guessing
python3 -m distil.cli mcp --add fs npx -y @modelcontextprotocol/server-filesystem /tmp
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

**7. It asks rather than guessing when it cannot state the objective.** The gate
is not "is this vague?" but "can the first step be taken?", and only two things
block it:

| task | gated? | why |
|---|---|---|
| build a csv parser that passes the test suite | no | the task states its own objective; referee is the test suite |
| dedupe the records | no | no referee, but a missing referee is a grading limit, not a reason not to start |
| compute the median of a column | no | checkable verb, move set identified |
| make the thing better | **yes** | no statement of done exists, and none can be inferred |
| improve the design | **yes** | same |

Gated tasks return the questions and nothing is distilled — no goal tree, no
plan. Answer them and the same call proceeds. Incidental questions are asked
once; a *blocking* one comes back until answered, because giving up on the only
thing preventing progress is not politeness.

**8. MCP tools and local tools live in one embedding layer.** A goal reaches both
through the same graded recall, and `Toolbox.invoke` dispatches on transport:

```
toolbox: assert_schema, chunk, diff_lines, echo.add, echo.echo, extract_identifiers,
         jaccard, median, median_of_column, normalise_error, parse_csv, percentile, ...
invoke("median",   [[5,3,1,4]])          -> 3.5     (sandboxed subprocess)
invoke("echo.add", {"a":2,"b":40})       -> "42"    (JSON-RPC to another process)
```

MCP tools enter **ungraded** — there are no contract tests to run against someone
else's server, and registering them as verified would be manufacturing evidence.

**9. The starter toolkit, all twelve verified before entry.** `distil seed`
plants them through the ordinary grader; a seed that fails its own contract tests
is rejected like anything else. The selection rule is the interesting part: the
highest-leverage tools are the ones that **make more things gradeable**, because
that is what moves goals from uncheckable to checkable and decides where
distillation may stop.

```
median  parse_csv  chunk  percentile  jaccard  diff_lines  topological_sort
normalise_error  extract_identifiers  assert_schema  summarise_counts  median_of_column
```

`median_of_column` is composed from `median` and `parse_csv` — dependencies are
inlined at validation and at call time, so the source that runs is exactly the
source that passed its tests.

**10. Invention by analogy finds transfers a nearest-neighbour query cannot.**
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
| `clarify.py` | 507 | ask when the objective is not understood; stop when the first step is actionable |
| `mcp.py` | 561 | MCP servers over stdio JSON-RPC, embedded as ordinary tools |
| `seed.py` | 823 | the starter toolkit, planted through the grader |
| `frame.py` | 570 | understand the game first: players, actions, payoff, horizon, referee → solution concept, capability gaps, agenda |
| `memory.py` | 446 | the embedding layer that *is* the memory; graded recall, credit propagation |
| `reason.py` | 440 | typed chain-of-thought; the game against Nature |
| `store.py` | 477 | in-process exact search, pgvector, Redis, and tiered short/long-term |
| `explore.py` | 508 | curiosity, bad ideas, analogy, inversion, policy self-upgrade |
| `toolsmith.py` | 585 | write a tool, verify it, register what it solved |
| `selfedit.py` | 338 | rewrite its own source behind invariants and the test suite |
| `game.py` | 290 | maximin, minimax regret, fictitious play, regret matching, Shapley |
| `challenge.py` | 354 | why-chains, premise attack, persistence past refusal |
| `compress.py` | 253 | consolidate cold memory into detail-preserving digests |
| `casebook.py` | 195 | problems and what actually solved them, both directions |
| `goals.py` | 220 | goal tree; distillation stops at verifiability |
| `sandbox.py` | 199 | AST screen, subprocess, timeout, stripped environment |
| `grade.py` | 170 | parse → screen → run → tests → determinism |
| `embed.py` | 201 | signed hashing trick, online IDF, blake2b |

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

## What four rounds of adversarial review found

The code was reviewed by parallel agents across five dimensions, each finding
verified by an independent skeptic, four times. **86 defects were confirmed and
fixed.** The distribution is the interesting part:

| round | found | of which introduced by the previous round's fixes |
|---|---|---|
| 1 | 45 | — |
| 2 | 15 | 6 |
| 3 | 13 | 4 |
| 4 | 13 | 7 |

Fixing introduces bugs at roughly the rate you would fear, which is the argument
for re-reviewing the *fixed* tree rather than stopping at a green suite. Every
round here ran against the working tree, not against `HEAD`; the first round's
verification pass largely refuted its own findings because the code had already
moved under it.

Three worth naming, because they are the kind a test suite does not catch:

- **The tools that grade other work were themselves wrong.** `assert_schema`
  failed *open* — an unknown type name skipped the check, so a typo accepted
  every value. A referee that reports success for constraints it never checked is
  worse than no referee.
- **Two seed tools had their regex escapes destroyed at import.** The Seed source
  literals are not raw strings, so `\b` became a backspace character and every
  word-boundary anchor silently stopped anchoring. `\w` and `\d` are not valid
  Python escapes and passed through untouched — which is exactly why only `\b`
  broke, and why nothing noticed.
- **Reuse manufactured its own evidence.** The reuse path called
  `record_use(worked=True)`, writing a verifier-grade success for a tool that
  branch deliberately does not run. Every recognition pushed a `+1` that could
  outvote the real negative grades a failing tool had earned.

And one that made a feature pointless rather than wrong: clarification answers
reached the `GameFrame` and stopped there, because `solve()` ran the reasoner on
the raw task string. Someone who patiently explained what done means got the same
goal tree as someone who said nothing.

## Honesty about the offline mode

Every result above came from `LocalProvider`, and it has real limits:

- The tool synthesiser covers a fixed table of goal shapes and returns `None`
  outside it. It demonstrates the mechanism, not the capability.
- Claim-experiments (curiosity, inversion) correctly report **not run** offline,
  because a function template cannot refute an assertion. An experiment tested by
  an unrelated check is worse than an untested one — it manufactures evidence.
- The lexical embedder does not know "compile" and "build" are related.
- Prose is never self-graded, so most non-code work accumulates as `ungraded`.
- MCP is stdio only, and no real server is exercised by the suite — only a local
  fixture written to speak the protocol (including stray output and unsolicited
  notifications, because a client that only works against a well-behaved server
  works against exactly one).

`DESIGN.md` §16 lists the rest.

## See also

This directory leans on two arguments made elsewhere in the repository:

- `GREN/DESIGN.md` §2 — *a game's identity is its refusal boundary*. The
  persistence loop (§9.3) is that claim turned into a control loop.
- `GREN/DESIGN.md` §4 — the 50% failure rate falls out of maximising information
  gain. §10.2 applies the same result to choosing experiments.
- `CyclicCortex` — Shapley credit over regions; here it is Shapley over subgoals.
