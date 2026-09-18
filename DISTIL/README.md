# DISTIL

**D**istillation **I**nto **S**ubgoals, with **T**ool **I**nduction and **L**earning.

An agent whose memory is an embedding layer that grades itself, whose reasoning is
a game played against Nature, and whose capability grows by writing Python tools
it then has to prove work.

`DESIGN.md` is the specification. This is what runs.

## Quick start

```bash
cd DISTIL
make help                                   # every target, with its defaults
make demo                                   # the whole system, offline, no key
make test                                   # 338 tests, ~60s
```

There is a `Makefile` for all of it — `make ask TASK="..."`, `make recall
QUERY="..."`, `make ui` — with every task, goal and query overridable on the
command line. State goes to `./state` rather than `~/.distil`, so experimenting
here cannot quietly rewrite a real memory. The commands it wraps:

```bash
python3 -m distil.cli demo                  # the whole system, offline, no key
python3 -m tests.test_distil                # 338 tests, ~60s

python3 -m distil.cli seed                  # plant the starter toolkit (12 verified tools)
python3 -m distil.cli clarify "make the thing better"    # it asks instead of guessing
python3 -m distil.cli mcp --add fs npx -y @modelcontextprotocol/server-filesystem /tmp
python3 -m distil.cli frame "negotiate a raise with my manager every review cycle"
python3 -m distil.cli ask   "build a csv cleaner and compute the median of each column"
python3 -m distil.cli why   "we must rewrite the parser in Rust because Python is too slow"
python3 -m distil.cli forge "compute the median of a list"
python3 -m distil.cli explore --steps 3
python3 -m distil.cli auto --cycles 20     # it picks its own next move, and keeps going
python3 -m distil.cli cases --like "the csv has ragged rows"
python3 -m distil.cli compress --dry-run
python3 -m distil.cli selfedit reason.py "cache the payoff matrix between calls"

python3 -m distil.cli ui                    # the chat frontend on 127.0.0.1:8765
docker compose up -d                        # ... plus a Chrome container it may drive
```

**Zero third-party dependencies at runtime.** Standard library only, like the
rest of this repository — `urllib` for the provider calls, `hashlib` for the embeddings,
`subprocess` for the sandbox. No numpy, no vendor SDKs, no vector database
required. The frontend is the one exception and it is a *build*-time one: `ui/`
is React, and `web/` — what it compiles to — is committed, so running the UI
needs no node and no `npm install`.

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
| `clarify.py` | 618 | ask when the objective is not understood; stop when the first step is actionable |
| `mcp.py` | 581 | MCP servers over stdio JSON-RPC, embedded as ordinary tools |
| `seed.py` | 842 | the starter toolkit, planted through the grader |
| `frame.py` | 575 | understand the game first: players, actions, payoff, horizon, referee → solution concept, capability gaps, agenda |
| `memory.py` | 463 | the embedding layer that *is* the memory; graded recall, credit propagation |
| `reason.py` | 440 | typed chain-of-thought; the game against Nature |
| `store.py` | 477 | in-process exact search, pgvector, Redis, and tiered short/long-term |
| `explore.py` | 529 | curiosity, bad ideas, analogy, inversion, policy self-upgrade |
| `toolsmith.py` | 638 | write a tool, verify it, register what it solved |
| `selfedit.py` | 338 | rewrite its own source behind invariants and the test suite |
| `game.py` | 290 | maximin, minimax regret, fictitious play, regret matching, Shapley |
| `challenge.py` | 354 | why-chains, premise attack, persistence past refusal |
| `compress.py` | 279 | consolidate cold memory into detail-preserving digests |
| `casebook.py` | 195 | problems and what actually solved them, both directions |
| `goals.py` | 220 | goal tree; distillation stops at verifiability |
| `sandbox.py` | 200 | AST screen, subprocess, timeout, stripped environment |
| `grade.py` | 252 | parse → screen → run → tests → determinism |
| `embed.py` | 201 | signed hashing trick, online IDF, blake2b |
| `serve.py` | 700 | the local HTTP API the frontend drives, including two SSE streams |
| `auto.py` | 380 | the loop that runs itself: six moves, regret-matched, rewarded for information |
| `browser.py` | 200 | a Chrome it can drive, over W3C WebDriver, as tools in the same embedding layer |
| `think.py` | 250 | clustering the embedding space and naming what forms, as new memories |
| `speech.py` | 175 | local transcription when a Whisper binary exists, and honesty when it does not |
| `project.py` | 139 | PCA by power iteration: 512 dims down to a plane you can look at |

## The frontend

```bash
python3 -m distil.cli ui                    # http://127.0.0.1:8765
cd ui && npm install && npm run build       # only if you change the React source
```

**One conversation, not a dashboard.** This was six tabs. Tabs make you navigate
to a system and hold the correlation between views in your head; everything here
is a message instead, so the transcript is the record of what happened, in order,
and the thing you asked four questions ago is still on the page underneath its
answer.

Ask a question in plain language. Everything else is a slash command, with an
autocomplete menu — `/tools`, `/recall`, `/explore`, `/trace`, `/mcp add …` — and
each one renders its result as a turn in the same transcript.

An answer leads with the verdict. The frame, the payoff matrix, the goal tree and
all twelve reasoning steps sit underneath it, collapsed, each with a summary
saying whether it is worth opening. The previous version rendered six expanded
cards — about two thousand pixels for one question, with the answer at the
bottom, under everything that led to it.

**Grading is where the opinion is.** The verdict carries +1 / 0 / −1 buttons. It
used to require copying a trace id, switching tabs, finding the point in a
scatter plot and clicking it — and a feedback loop with four steps of friction is
one nobody closes. User grades count double, so this is the single highest-value
interaction in the product.

**Every question searches memory**, and the result rides on every reply, gated or
not. A question that got stopped for clarification used to return questions and
nothing else — the one case where "what do I already know about this?" is most
useful was the case that answered it least.

**Speech.** The microphone says where your voice is going *before* it listens. If
a Whisper binary is on `PATH`, audio is recorded, POSTed to `/api/transcribe` and
never leaves the machine. Otherwise the browser's Web Speech API does it — which
in Chrome means the audio goes to Google, and a local-first tool that hid that
would be lying by omission. Ollama is not an option here: it serves language and
embedding models and does not transcribe audio.

## Running by itself

`▸ run by itself` starts an autonomous loop (`distil/auto.py`) that narrates into
the same transcript. Six moves compete for each cycle, regret-matched, so the mix
is *learned* from what each returned rather than fixed:

| move | what it does |
|---|---|
| **question** | takes something it believes and asks why until the chain terminates, then attacks the premises |
| **experiment** | brainstorms and runs one, ranked by information gain |
| **build** | finds a capability gap and forges a tool through the grader |
| **consolidate** | folds cold memory into digests that keep the detail |
| **tune** | re-fits its own policy against measured outcomes |
| **pursue** | sets itself a new problem and runs the full solve loop on it |

It is rewarded for **information, not success**. A move that confirms what it
already believed scores near zero however cleanly it ran; one that refutes
something scores highly. Rewarding correctness would teach it to stop proposing
the experiments worth running.

**It never runs out of direction.** The first five moves all consume pools that
empty — a last unquestioned belief, a last known gap, a last cold cluster — and
when they did, every cycle returned "nothing to do" and the loop spun: neither
working nor finished. `pursue` is the answer, and it is forced after two barren
cycles. It draws directions from five sources, round-robin, the last of which
cannot exhaust (pairs of distant memories are quadratic, and every cycle adds to
*n*). Started against an empty store it plants the starter toolkit first, because
there is otherwise nothing to derive a direction from.

The dashboard above the log reports outcomes rather than activity: "63 cycles"
says how long it ran, "2 tools built, 9 beliefs found resting on nothing, 4 ideas
refuted" says whether that was worth it.

It **cannot edit its own source**. `selfedit` is reachable only from the command
line, deliberately: an unattended loop with write access to its own grader makes
every grade in the store meaningless.

## Driving a browser

`distil/browser.py` speaks W3C WebDriver over `urllib` — no Selenium, no
dependency. The actions are embedded as ordinary `Kind.TOOL` traces, so
`browser.read` is recalled by the same query that finds a locally forged function
or an MCP tool: one embedding layer over every capability.

```bash
docker compose up -d           # the agent, plus a Chrome container it may drive
```

A driven browser runs whatever the pages it visits contain, in a process this
package does not control. That is the argument for the compose file: chromedriver
belongs in a container, its port is never published, and the agent reaches it by
service name. Set `DISTIL_WEBDRIVER` to attach one; without it the browser tools
are not registered at all, because a capability that is recalled and then fails
is worse than one that was never offered.

The projection is PCA by power iteration from a *fixed* start vector, so the same
store always draws the same picture and a moved point means the memory moved.
The axes have no meaning: distance is meaningful, direction is not.

**It is a local tool, not a service.** It binds 127.0.0.1, and it drives an agent
that writes files and executes generated code. There is no authentication, and
adding a token would imply a level of hardening this does not have. What it does
have is three cheap defences: static paths are confirmed to resolve inside
`web/`; a non-loopback `Host` is refused (DNS rebinding); and a cross-site
request is refused outright — because any page you visit may address `127.0.0.1`
directly, and its `Host` header is then perfectly honest. Without the third, a
visited page could POST `/api/forge` and have this agent write and run code.
`--bind` exists because someone will want it in a container; the warning it
prints exists because they should know what they are opening.

## Thinking by clustering

Everything else treats the embedding layer as something to *read*: embed a query,
retrieve neighbours, rank them. `think.py` treats its **shape** as information —
it clusters what the system knows, names each cluster, and writes the name back
as a new memory.

A cluster's centroid is a point near every member and identical to none of them,
which is what a concept *is*. Storing it gives recall one hop to a whole
neighbourhood where before it had to be similar to one specific member to reach
any of them. The store gets denser rather than only longer.

**Agglomerative average-linkage, not nearest-neighbours-of-a-seed.** Clusters are
disjoint and deterministic: the same memories always produce the same concepts,
so a new one means the store changed. Average linkage rather than single, because
single linkage chains — one trace between two unrelated groups merges them into a
cluster whose centre means nothing.

**The cut comes from the store's own distribution** (`mean + 0.75σ` over observed
pairs), not a constant. What counts as "similar" depends on the embedder and the
corpus: measured here, the lexical embedder puts unrelated short text at ~0.01 and
related text at 0.13–0.40, while a provider embedder sits in a much narrower,
much higher band. No single number separates both; "unusually close *for this
store*" does.

Everything written is **ungraded and marked `derived`** — a conjecture read off
the geometry, entering at the credibility prior like an MCP tool nobody has run.
It earns credibility only if something later confirms it. Derived traces are
excluded from the pool *and* from the similarity search; filtering them from the
results is not enough, since they still occupy slots and shift which cluster
forms.

`think` is a seventh autonomous move, so this runs constantly.

**It favours precision over recall.** On a 12-trace fixture with three known
groups it finds two, both pure, with no mixed clusters. The third is four
sentences sharing almost no vocabulary — at the lexical embedder's noise floor.
A wrong concept is worse than a missing one, because it becomes a `Kind.FACT` the
system then believes about itself. Provider embeddings raise the recall; nothing
here lowers the bar to chase it.

## Choosing where memory lives

```bash
make stores                 # what is configured, and what each backend needs

export DISTIL_STORE=tiered  # memory | postgres | redis | tiered
export REDIS_URL=redis://localhost:6379
export DATABASE_URL=postgresql://localhost/distil
export DISTIL_REDIS_TTL=604800      # one week
```

`from_env` **raises rather than falling back**. A misconfigured backend that
quietly degrades to in-process memory is the same failure as a provider that
answers nothing while reporting healthy: the system keeps working and your data
goes somewhere other than where you asked.

```
$ DISTIL_STORE=postgres make stats
StoreError: the postgres store needs DATABASE_URL; set it, or unset
DISTIL_STORE to use the in-process store
```

Neither Redis nor Postgres is exercised by the test suite — there is no server in
the environment this was built in, so treat your first run against a real one as
the actual test.

## When it is not using your model

The provider is chosen **once, at startup**, so starting Ollama after the server
is already running changes nothing — restart it.

```bash
make providers          # says what is reachable and why the rest is not
#   ollama  not available  ollama is running but has no 'gemma4':
#                          pull it with `ollama pull gemma4`
```

`available()` requires that the daemon is reachable **and** holding the model it
is configured to use. That is deliberate, and it is the fix for a defect worth
knowing about: `/api/tags` returns 200 whatever is installed, so an un-pulled
model used to mean Ollama was selected, every `/api/chat` 404ed, every caller
swallowed the error and fell back to a heuristic — correctly, since a reasoning
step that cannot reach a model should degrade rather than crash — and the agent
reported `ollama` while running entirely on offline rules, silently.

Every provider now counts its calls and its failures. `GET /api/state` carries
`provider_health`, the header badge turns red and reads **"not answering"** when
every call has failed, and the transcript says so once in plain language. A
provider that is attached but answering nothing is no longer indistinguishable
from one that works.

```bash
curl -s localhost:8765/api/state | python3 -m json.tool | grep -A4 provider_health
```

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
verified by an independent skeptic, six times. **115 defects were confirmed and
fixed.** The distribution is the interesting part:

| round | found | of which introduced by the previous round's fixes |
|---|---|---|
| 1 | 45 | — |
| 2 | 15 | 6 |
| 3 | 13 | 4 |
| 4 | 13 | 7 |
| 5 | 2 | 1 |
| 6 — frozen tree | 27 | 5 |

Rounds 1–5 ran against a tree that was being edited underneath them, so their
*verifiers* were worthless — 16 of 16 refutations in round 4 read "the code no
longer exists in the working tree", because the bug had been fixed while they
checked. Only the reviewers' findings were usable. Round 6 froze the tree at a
commit and left it alone: **27 of 28 findings survived adversarial
verification.** A green suite and five rounds of review had not made the code
clean, and the only round that could prove it was the one that stopped moving.

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
- **The determinism stage never fired.** It re-ran the *module* and compared
  stdout, but the tool contract mandates no I/O outside the function — so a
  conforming module prints nothing, the stage compared `''` to `''`, and it
  passed unconditionally without ever calling the function. A tool returning a
  different answer on every call graded `+1.000 verified`, and all twelve seed
  tools collected the weight as a gift.
- **Two seed tools had their regex escapes destroyed at import.** The Seed source
  literals are not raw strings, so `\b` became a backspace character and every
  word-boundary anchor silently stopped anchoring. `\w` and `\d` are not valid
  Python escapes and passed through untouched — which is exactly why only `\b`
  broke, and why nothing noticed.
- **Reuse manufactured its own evidence.** The reuse path called
  `record_use(worked=True)`, writing a verifier-grade success for a tool that
  branch deliberately does not run. Every recognition pushed a `+1` that could
  outvote the real negative grades a failing tool had earned.

And two that were wrong in a way a passing suite cannot show:

- **Clarification was cosmetic.** The answers reached the `GameFrame` and stopped
  there, because `solve()` ran the reasoner on the raw task string. Someone who
  patiently explained what done means got the same goal tree as someone who said
  nothing.
- **The clarity gate inverted in both directions at once.** `handle it somehow
  please` proceeded while `write a parser that produces clean output` was
  refused, and `fix everything` was gated where `fix everything now` was not —
  one filler word decided it. The cause was using `goals.checkability` as a
  boundary when its own docstring calls it *a prior that only has to be right on
  average*. Sixteen tasks are now pinned as a discrimination table.

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

`DESIGN.md` §17 lists the rest.

## See also

This directory leans on two arguments made elsewhere in the repository:

- `GREN/DESIGN.md` §2 — *a game's identity is its refusal boundary*. The
  persistence loop (§9.3) is that claim turned into a control loop.
- `GREN/DESIGN.md` §4 — the 50% failure rate falls out of maximising information
  gain. §10.2 applies the same result to choosing experiments.
- `CyclicCortex` — Shapley credit over regions; here it is Shapley over subgoals.
