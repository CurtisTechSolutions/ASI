# GREN — Design Specification

**G**ame **R**ule **E**ncoder **N**etwork.

Part one of the multi-network architecture. GREN does not play. GREN **works out
which game is being played** — by guessing, being told no, and treating every
*no* as the lesson. When it knows, it hands a finished rule package to `GTMNN/`
(part two), which plays the game it is handed.

The job is **identification by similarity**, not induction from nothing. GREN
holds a corpus of games it already knows, and the question it actually answers is
*"which of these does this new thing refuse like?"* — a nearest-neighbour query
against a structure it already has, run on partial evidence, refined by probes
chosen to separate the candidates still standing. Learning a rule set from zero
is the degenerate case where nothing in the corpus is close, and it is by far the
expensive one (§3.1).

> Once you understand the game, then and only then can you play the game.

That ordering is enforced, not suggested: GTMNN refuses a game package below a
confidence threshold (§22.1). A system that plays before it understands is
guessing with extra steps.

Directory: `GREN/` (this directory). Python package: `gren`.
Python 3.11+, standard library only. This document is the contract every module
is implemented against. Read it fully before writing code.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| Solve games; to solve a game, understand it; to understand it, explore it | Four phases with a hard boundary before play: **retrieve** (§3, candidates from partial evidence), **explore** (§17, probes chosen to separate them), **induce** (§16, rules — only where the candidates disagree), **encode** (§19, the tree). Playing is not in this package at all. |
| Figure out which game we are playing by finding similar games | §3 is the spine. A partial signature retrieves a **candidate set** from the tree (§19.4); probes are chosen to split that set (§17.4); the nearest surviving neighbour's rule set is inherited as a prior and corrected only where it fails (§16). Identification costs `log2|G|` bits against a corpus of `G` games; induction from nothing costs `log2|Θ|`, which is three to four orders of magnitude worse (§3.1). |
| Clustering, based on what we know about the game so far | Evidence is **partial by default** (§3.2): most axes are unobserved most of the time. Retrieval branches over unknown axes and scores paths by bucket posterior weighted by information gain (§19.4), so the answer is always a ranked cluster with a confidence, never a single guess pretending to be certain. |
| You learn the rules by failing | The `Oracle` (§12) is the thing that says no. Every probe returns a `Verdict`, and an ILLEGAL verdict with a reason is worth more than a LEGAL one — it names a boundary, where a success only says "somewhere inside". |
| Failing *on purpose*, guessing | The probe policy maximises expected information gain **about which game this is** (§4). For a binary oracle that objective is provably maximised at `p(legal) = 0.5`, so **the target failure rate is 50%** — it falls out of the mathematics rather than being set as a knob. The probe that best splits the candidate set is the one half of them call legal and half call illegal (§17.4). |
| The compiler tells you "this ain't working, bro" | `CompilerOracle` / `InterpreterOracle` (§12.1). The diagnostic text is not discarded — it is the highest-bandwidth channel in the system, and §15 clusters the oracle's own words into a reason taxonomy that becomes the rule vocabulary. |
| Programming is the game worth optimising for | Quantified in §5, and it holds up: programming is the only domain whose rule set is large enough to be worth learning *and* whose oracle is fast enough to learn it from. |
| It is a classification task and a learning task at once | Learning = rule induction (§16). Classification = which leaf of the tree this game lands in (§19). The same structure does both, which is why they are not two systems. |
| Embedding space, information clustering — chess and checkers cluster closer than talking to your boss | Similarity is **shared prefix depth weighted by information gain** (§21), with forest proximity (§20.3) as the robust version. Chess/checkers agree on seven axes before diverging; chess and a difficult conversation diverge at axis 0. §21 projects the signatures to a 2D plane by classical MDS, which is the "plane of existence" made literal and is what the frontend renders. |
| Multiple different problems in one | One structure answers all four. The tree's levels are the axes of game-space; descending it is a decision tree; prefix overlap is the clustering metric; a forest of trees with randomised axis orderings is the random forest (§18). |
| An n-dimensional trie — a tree with ending states, with the game encoded into each path | §19. `n` is the number of game-space axes; each contributes one level; a root→node path is the game's **rule signature**; an ending state is a **game class** holding the induced rule set. |
| A radix tree with the mechanics of a trie | §19.1. Path compression from the radix tree so only the mechanics that actually *discriminate* cost a node; terminal-on-internal nodes from the trie so a general game class can be a strict prefix of a specific one. Neither structure alone can hold both properties. Split and merge follow `RadixCyclicNN/DESIGN.md` §5.2 exactly. |
| Or invert the problem — it could be a random forest | §20. Axis ordering is not unique, so a single tree is a single arbitrary ordering. `T` trees over randomised orderings and bootstrapped corpora, with Breiman proximity as the similarity measure. This is the robust version of §19 and the two agree where both apply. |
| Play chess and checkers with the same input and output size, with a game modifier | §22.4-22.5. Score **one candidate at a time**, so action-space size never touches the geometry. Input is `F_state ⊕ F_cand ⊕ F_mod` = 512; output is a fixed 3 units for every game. The **modifier** is the mechanic set hashed into 128 dims, and because hashing a set preserves overlap, similar games land at nearby points — so transfer is a consequence of the encoding rather than a mechanism bolted on. |
| Validity first, then a separate grade for the move | §22.6. Two heads on one micro: `p_valid` (legal?) trained by GREN from oracle refusals — binary, instant, abundant, **transferable**; `grade` (good?) trained by GTMNN from Shapley credit — sparse, delayed, game-specific. This is what actually compresses chess, go and checkers into one space, and it makes "understand the game, then play it" true inside a single micro rather than only at the system level. |
| You need an encoder for the game | §22 `GamePackage`: the signature, the learned legal action set, the legality predicate, the GTMNN payoff class and its calibrated constants. That object is the encoding, and it is the entire interface to part two. |
| Everything can be distilled down into a game | The *encoding* is general — anything with a refusal oracle can be probed. The *learnability* is not, and §5 says so with numbers. That refinement is the useful part of the claim. |

---

## 2. The claim: a game is identified by what it refuses

Take chess. You do not know the rules. You push a pawn three squares and are
told no. You move your king into check and are told no. You castle through an
attacked square and are told no. After enough of this you have not been told the
rules of chess — you have been told the *boundary*, and the boundary is the same
object.

This is the whole design:

> **A game's identity is its refusal boundary. Two games are similar exactly to
> the degree that they refuse the same things.**

Chess and checkers refuse overlapping sets: both refuse moving off-board, moving
out of turn, moving an opponent's piece, moving to an occupied friendly square.
They diverge on what a piece may do. The overlap is large, so they cluster.

A difficult conversation with your boss refuses almost nothing chess refuses. Its
action alphabet is open, its legality is graded rather than binary, and it has no
referee that halts you mid-sentence. It is a game, and it is a game the *encoder*
handles fine; it is just nowhere near as learnable by probing (§5).

Three consequences worth stating up front, because the rest of the package
follows from them:

1. **Failure is the signal, not the absence of one.** A LEGAL verdict says "you
   are somewhere inside the boundary". An ILLEGAL verdict *with a reason* names
   which wall you hit. The information asymmetry is enormous and it is the
   reason the probe policy deliberately aims at failure.
2. **The reason text is the highest-bandwidth channel in the system.** Most
   learners keep the bit and throw away the sentence. `expected i32, found &str`
   localises a type rule in one probe; a bare rejection would take hundreds of
   probes to localise the same rule. §15 treats the oracle's own words as a
   first-class observation.
3. **Identity is comparative, not absolute.** GREN never asks "what game is
   this" in the abstract. It asks "which known games does this refuse like",
   which is a nearest-neighbour query against a structure it already has, and
   which degrades gracefully to "none of them, here is a new leaf".

---

## 3. The task: work out which game this is by finding similar games

GREN is a **nearest-neighbour engine over game-space**. It holds a corpus of
games it already knows. Shown something new, it asks which known games this one
resembles, answers with a ranked cluster, and spends probes only on separating
the candidates still standing.

### 3.1 Identification is exponentially cheaper than induction

Two different questions, three orders of magnitude apart:

| Question | Cost | Rust, concretely |
|---|---|---|
| *What are this game's rules?* (induction from nothing) | `log2|Θ|` bits | ~10^5 bits ≈ hours of compiling |
| *Which known game is this?* (identification against a corpus of `G`) | `log2|G|` bits | ~10 bits for `G = 1000` ≈ **ten probes** |

Ten probes against a thousand known games, versus a hundred thousand bits from
scratch. Once identified, the neighbour's rule set is **inherited as a prior**
and probing continues only where the surviving candidates actually disagree
(§16.4) — which is a tiny fraction of the rule set, because games in the same
cluster are in the same cluster precisely *because* most of their rules coincide.

This is how a polymath actually learns, and it is worth being explicit that the
architecture is a claim about that: you do not learn each new domain from zero.
You recognise what it is *like*, import everything that transfers, and then grind
only on the differences. The grinding is real and it is where the time goes — but
it is spent on the delta, not the whole.

Induction from nothing still exists in the system (§16.5) as the cold-start path
when nothing in the corpus is within `τ_novel` of the evidence. It is the
expensive case, it is correctly rare once the corpus is populated, and every new
game it produces is added to the corpus so it is never paid for twice.

### 3.2 Evidence is partial, and that is the normal case

At any moment GREN holds an **Evidence** object: a partial, uncertain assignment
over the axes of §9. Most axes are unobserved most of the time.

```python
@dataclass
class Evidence:
    posterior: dict[int, array]     # axis id -> a distribution over that axis's buckets
                                    # absent key = wholly unobserved (uniform); a one-hot row = certain
    probes: list[tuple[Action, Verdict]]
    reasons: Counter                # reason class -> count, the refusal fingerprint so far
    declared: set[int]              # axes known without probing (§9.1)
    measured: set[int]              # axes resolved by probing
    def entropy(self) -> float              # total remaining uncertainty, IG-weighted
    def known_fraction(self) -> float
    def to_signature(self) -> tuple         # the MAP path; ties -> lowest bucket
    def to_dict(self) -> dict
```

The split between `declared` and `measured` carries real weight. A declared axis
is **free** — you are told "two players, alternating turns, capture the enemy
king" before you touch a piece, exactly as you read the box before playing. A
measured axis costs probes. Retrieval on declared axes alone routinely cuts a
thousand-game corpus to a handful, and the probe budget is then spent only on
what the box did not say.

Nothing in the pipeline requires complete evidence. Retrieval (§19.4) branches
over unknown axes, so the answer is always a **ranked cluster with a
confidence**, never a single guess wearing a confident face.

### 3.3 The loop

```
declared evidence  ─────────────────────────┐
                                            ▼
                          ┌──────► retrieve candidates (§19.4)
                          │              │
                          │              ▼
                          │      confident enough?  ──yes──►  inherit the winner's rules (§16.4)
                          │              │                              │
                          │              no                             ▼
                          │              ▼                    probe only where candidates
                          │      choose the probe that                  disagree
                          │      best splits the candidates             │
                          │      (§17.4)                                ▼
                          │              │                     GamePackage ──► GTMNN
                          │              ▼
                          │      oracle verdict + reason
                          │              │
                          └──────────────┘  update Evidence
```

Termination, in priority order: the candidate set collapses to one above
`τ_conf`; or the probe budget is spent; or the expected information gain of every
available probe falls below `eig_floor` — meaning no probe would tell you
anything, which is a real and reportable state rather than a failure.

---

## 4. Why failing on purpose is optimal, and why the rate is 50%

The intuition — guess, fail, learn from what breaks — has an exact form, and the
exact form makes a falsifiable prediction.

Let `C` be the candidate set with posterior `p(g)` over games `g`. Probing action
`a` yields verdict `V`. Choose the probe maximising expected information gain
**about which game this is**:

```
EIG(a) = H[V | a]  −  E_{g~p} H[V | a, g]
```

**Binary, noise-free.** Each candidate deterministically predicts legality, so
`H[V|a,g] = 0` and

```
EIG(a) = H(Bernoulli(p_legal(a)))     where  p_legal(a) = Σ_g p(g) · 1[g says a is legal]
```

maximised at `p_legal(a) = 0.5`. **The most informative probe is the one exactly
half the surviving candidates call legal and half call illegal.** A probe they
all accept teaches nothing. A probe they all reject teaches nothing. Maximum
learning sits exactly at maximum disagreement — and note that this is
disagreement *among the candidates*, not vagueness in your own head, which is
what makes it computable.

**Noisy oracle.** With a constant error rate `ε`, `H[V|a,g] = H(ε)` for every
candidate, so the maximiser is unchanged. Noise costs bits per probe; it does not
move the target.

**So the target failure rate is 50%,** and it is a *measurement*, not a setting.
`Explorer` reports the observed rate every epoch (§23.4). A policy running at 5%
or 95% is not exploring — it is confirming what it already believes, or thrashing
outside the boundary. This is the most useful single diagnostic the system
produces, and `test_probe.py` asserts the `eig` policy holds `0.5 ± 0.1` on the
synthetic oracles where ground truth is known.

**Version-space halving.** A probe at `p_legal = 0.5` halves the candidate set in
expectation, so identification takes `≈ log2|C₀|` probes — ten probes for a
thousand candidates, which is the §3.1 number derived rather than asserted.

**Where the clean result stops.** With structured reasons the verdict is not
binary: it takes one of `R+1` outcomes (legal, or one of `R` reason classes), and
`EIG` is then maximised where the predictive distribution spreads evenly over
*outcomes*, which is generally **not** at 50% legality. A reason class carries
`log2(R+1)` bits against a bare rejection's one, so rich oracles identify faster
than the halving bound suggests. The binary result is the special case `R = 1`.
The implementation computes the full multi-outcome `EIG` (§17.3); the 50% figure
is reported as a diagnostic that is exact in the binary case and approximate
otherwise. Targeting 50% directly would be fitting the knob instead of the
objective, and is explicitly not what the default policy does.

---

## 5. What is identifiable, and what is only placeable

"Everything in life can be gamified" is true of the *encoding* and misleading
about the *effort*. Two quantities decide whether probing finishes:

```
horizon       = log2|Θ|                  bits of rule to learn from nothing
learning_rate = reason_bits / latency    bits the oracle emits per second
time_to_learn = horizon / learning_rate
```

`reason_bits = log2(distinct reason classes + 1)`. Order-of-magnitude estimates:

| Oracle | reason_bits | latency | learning_rate | horizon | induction | **identification** |
|---|---|---|---|---|---|---|
| Chess referee | ~3 | ~10 µs | ~300k bit/s | ~20 bits | instant | instant |
| Regex engine | ~4 | ~10 µs | ~400k bit/s | ~30 bits | instant | instant |
| Python interpreter | ~7 | ~30 ms | ~230 bit/s | ~10^4 bits | ~1 minute | **~0.3 s** |
| Rust compiler | ~9 | ~300 ms | ~30 bit/s | ~10^5 bits | ~1 hour | **~3 s** |
| Codebase + test suite | ~10 | ~30 s | ~0.3 bit/s | ~10^6 bits | ~5 weeks | **~5 min** |
| A person, in conversation | ~12 | ~60 s | ~0.2 bit/s | ~10^9 bits | ~150 years | **~10 min** |

The right-hand column is the point of the whole design. Identification needs
`log2|G| ≈ 10` bits regardless of how large the rule set is, so it collapses the
intractable rows to minutes. **A conversation with your boss is not learnable by
probing — but it is identifiable in about ten exchanges**, and identification is
what licenses transferring a neighbour's rules instead of deriving your own.

Two honest consequences:

* **Programming is still the best training domain**, and the middle column says
  why: it is the only row where inducting a rule set large enough to matter is
  also *affordable*. The compiler is free, instant, complete, never bored, never
  lies, and explains itself in a taxonomy someone already enumerated. That is why
  `CompilerOracle` is the reference implementation and every other oracle is
  measured against it.
* **For the slow rows, GREN places rather than learns.** It puts the thing in the
  tree, names its nearest learnable neighbours, and reports which of their rules
  transfer and with what confidence. That is a smaller claim than "solve life"
  and it is the correct size for a contract — but it is also not a small thing,
  because placement is what turns an unlearnable domain into a solved one you
  already have the rules for.

---

## 6. Notation

| Symbol | Meaning | Default |
|---|---|---|
| `G` | games in the corpus | grows |
| `C` | the current candidate set (the retrieved cluster) | |
| `n` | game-space axes — the tree's dimensionality | 22 seeded, grows |
| `T` | trees in the forest | 64 |
| `R` | reason classes discovered for a domain | learned |
| `σ` | a **rule signature**: a tuple of `(axis, bucket)` pairs; a path through the tree | |
| `Θ` | version space: rule hypotheses consistent with the evidence | |
| `IG(d)` | information gain of axis `d` over the corpus | |
| `probe_budget` | probes per exploration run | 10 000 |
| `τ_conf` | posterior below which a `GamePackage` is not handed to GTMNN | 0.8 |
| `τ_novel` | similarity below which the corpus is considered to have no neighbour | 0.35 |
| `eig_floor` | expected-information-gain floor below which probing stops | 0.01 bits |

---

## 7. Package layout

```
GREN/
  DESIGN.md                 this file
  README.md                 user docs
  pyproject.toml            zero runtime deps; console script `gren = gren.cli:main`
  Makefile
  Dockerfile ; docker-compose.yml ; docker/entrypoint.sh
  gren/
    __init__.py             exports Explorer, Evidence, Oracle, Verdict, Axis, AXES,
                            RuleSet, RadixGameTree, GameForest, GamePackage, __version__
    __main__.py             `python -m gren` -> cli.main()
    verdict.py              Verdict, Outcome, reason normalisation
    action.py               Action, ActionSpace, grammar-driven action generation
    oracle.py               the Oracle protocol and its implementations
    sandbox.py              subprocess isolation, timeouts and resource caps
    features.py             the feature language: predicates over (state, action)
    primitive.py            decomposition: operators, mechanics, minimality, the vocabulary (§8)
    reasons.py              clustering the oracle's own words into a reason taxonomy
    rule.py                 Rule, RuleSet, inheritance, version space + IG induction
    probe.py                probe policies; the candidate-splitting objective
    axis.py                 the axis catalogue, discretisation, IG ordering, Evidence
    radix.py                RadixGameTree - a radix tree with the mechanics of a trie
    forest.py               GameForest - randomised axis orderings, Breiman proximity
    similarity.py           prefix similarity, the distance matrix, classical MDS
    package.py              GamePackage - the handoff to GTMNN
    explorer.py             Explorer - the retrieve/probe/induce/encode loop
    checkpoint.py           CheckpointManager (same contract as RadixCyclicNN)
    bench.py                probes/sec, bits/probe, probes-to-identify
    cli.py                  argparse CLI
    api.py                  HTTP JSON API + static file serving
  tests/
  frontend/
  data/
    grammars/               action grammars for the reference oracles
    corpus/                 known games with hand-written ground-truth signatures
```

**Module dependency order** (a module may import only from those above it):
`verdict` → `action` → `oracle` → `sandbox` → `features` → `reasons` → `rule` →
`primitive` → `probe` → `axis` → `radix` → `forest` → `similarity` → `package` → `explorer` →
`checkpoint` → `bench` → `cli` → `api`.

---

## 8. Decomposition — the smallest parts

Before asking which games are similar, break each game into its smallest parts.
Similarity is then **overlap of parts**, not agreement on coarse whole-game
properties. This section comes before the axis catalogue (§9) because the axes
turn out to be derived from it (§8.7).

### 8.1 Why this has to come first

§9.3 records a failure: chess and go agree on eleven of thirteen axes and one
early disagreement makes a single tree call them unrelated. That was not bad luck
in the ordering — it was a symptom of comparing games over **22 coarse
coordinates**, where one disagreement is 1/22 of the evidence and can swamp
eleven agreements if it sorts early.

Over a few hundred primitives, one difference is one difference. Decomposition
buys three things, and the first two are things the whole-game design could not
have at all:

1. **Order-free similarity.** A game becomes a *set*. Set overlap has no ordering, so §9.3's pathology cannot occur by construction, not by patching (§8.6).
2. **Per-primitive inheritance.** §16.4 stops being "copy the nearest game's rules" and becomes "assemble the rule set part by part, taking each part from whichever game in the corpus knows it best". A game is no longer limited to one donor.
3. **Compositional novelty.** A genuinely new game is usually a new *combination* of known parts, not a new kind of thing. Decomposed, it needs no cold start (§16.5) — every part is already known, only the arrangement is new. This is the case that most defeats nearest-neighbour identification and it becomes the easy case.

### 8.2 The decomposition hierarchy

Four levels, smallest first:

| level | unit | comparable across games? | source |
|---|---|---|---|
| **Predicate** | one test on state — `occupied(sq)`, `same_owner(a,b)`, `type_matches(x,y)` | no — domain-specific vocabulary | the feature language (§14) |
| **Operator** | a `(precondition, action, effect)` triple, STRIPS form — the smallest *complete* unit of game dynamics | no — written in one domain's predicates | rule induction (§16) |
| **Mechanic** | a named class of operators recurring across games — `DISPLACEMENT_CAPTURE`, `ALTERNATE_TURNS`, `PROMOTE_ON_RANK` | **yes** | discovered, §8.4 |
| **Game** | the set of its mechanics | — | |

**The mechanic is the atom that matters**, because it is the smallest unit that
is comparable *across domains*. Predicates and operators are how you get to it:
they are strictly finer, and strictly domain-locked. A predicate about squares
means nothing to a compiler. `ALTERNATE_TURNS` means the same thing in chess and
in a conversation.

An induced `Rule` (§16) is already almost an operator — its predicate is a
precondition and its reason class is an effect — so the first two levels of
decomposition fall out of machinery this document already specifies. Only the
lift from operator to mechanic is new.

### 8.3 When is a part smallest? A criterion that is measurable

"Smallest possible" needs a stopping rule that is not a person's judgment, or the
decomposition is just taste with extra steps.

> A part is **minimal** when no proper sub-part of it is ever observed
> independently in the corpus.

Keep splitting while the pieces are seen apart; stop when they are only ever seen
together. Formally, for a candidate part `P` with a proposed split into `A` and
`B`:

```
split(P) iff  support(A ∧ ¬B) + support(¬A ∧ B)  >=  min_independent        (default 2)
```

`CHAIN_CAPTURE` and `FORCED_CAPTURE` co-occur in checkers, but draughts variants
exist with one and not the other, so they split. `CASTLING` never appears
anywhere as half of itself, so it does not — it is atomic despite being visibly
compound to a person, and that is the correct answer, because a part that never
varies independently carries no independent information.

This is how morphemes are found in linguistics, it is decidable from the corpus
rather than argued about, and it has one consequence that must be handled rather
than discovered later:

**Minimality is corpus-relative and changes as the corpus grows.** A part that
looks atomic today splits the moment a game arrives that uses half of it.
`primitive.resplit()` runs on every corpus insert, and every signature carries a
`mechanic_epoch`. A re-split bumps the epoch and **invalidates cached signatures**
rather than leaving them silently referring to a vocabulary that no longer
exists. `test_primitive.py` asserts that a re-split followed by re-signing every
corpus game reproduces the same similarity ordering it had before, so the
vocabulary can move without the map moving under it.

### 8.4 Lifting operators to mechanics

Two operators from different games are the same mechanic when their
precondition/effect structures are **isomorphic after alpha-renaming of domain
constants** — same shape, different vocabulary.

```python
def canonical(op: Operator) -> str
    # Replace every domain constant and variable with a typed hole, numbered by first
    # occurrence; sort commutative conjuncts; render to a canonical string. Two operators
    # are the same mechanic iff their canonical forms match. This is a hash, not a search:
    # O(|op|) per operator, so the whole corpus lifts in one pass.

def lift(ops: Sequence[Operator], vocab: "MechanicVocab") -> list[int]
    # canonical() exactly; else structural match against known mechanics above `merge_tau`
    # (graph edit distance on the precondition DAG, normalised); else open a new mechanic,
    # named from its exemplar game and effect.
```

Exact canonical-form matching is the fast path and covers most cases; the graph
edit distance fallback is what catches "the same idea written slightly
differently", and it is capped at operators of fewer than `max_op_nodes = 24`
because graph edit distance is expensive and large operators are usually
compound and about to be split by §8.3 anyway.

### 8.5 Worked mechanic sets

Seeded with ~40 hand-written mechanics for bootstrap; grows by discovery.

| game | mechanics (abridged) |
|---|---|
| **chess** | ALTERNATE_TURNS, GRID_BOARD, PERFECT_INFO, ZERO_SUM, TWO_PLAYER, PIECE_OWNERSHIP, DISPLACEMENT_CAPTURE, PIECE_TYPES, PROMOTE_ON_RANK, BLOCKED_BY_OCCUPANT, SLIDING_MOVE, STEP_MOVE, ROYAL_PIECE, CHECK_CONSTRAINT, CASTLING, EN_PASSANT, STALEMATE_DRAW, REPETITION_DRAW |
| **checkers** | ALTERNATE_TURNS, GRID_BOARD, PERFECT_INFO, ZERO_SUM, TWO_PLAYER, PIECE_OWNERSHIP, JUMP_CAPTURE, PROMOTE_ON_RANK, BLOCKED_BY_OCCUPANT, STEP_MOVE, DIAGONAL_ONLY, FORCED_CAPTURE, CHAIN_CAPTURE, ELIMINATION_WIN |
| **go** | ALTERNATE_TURNS, GRID_BOARD, PERFECT_INFO, ZERO_SUM, TWO_PLAYER, PIECE_OWNERSHIP, PLACEMENT_MOVE, TERRITORY_SCORING, GROUP_LIBERTY, SURROUND_CAPTURE, KO_REPETITION, PASS_ALLOWED, NO_MOVEMENT, AREA_COUNT |
| **rust** | FREE_ORDER, SYMBOLIC_ARTIFACT, PERFECT_INFO, SOLITAIRE, TYPE_CONSTRAINT, SCOPE_BINDING, OWNERSHIP_CONSTRAINT, LIFETIME_CONSTRAINT, SYNTAX_GRAMMAR, NAME_RESOLUTION, ARITY_CONSTRAINT, MUTABILITY_CONSTRAINT, EXHAUSTIVE_MATCH |
| **python** | FREE_ORDER, SYMBOLIC_ARTIFACT, PERFECT_INFO, SOLITAIRE, SCOPE_BINDING, SYNTAX_GRAMMAR, NAME_RESOLUTION, ARITY_CONSTRAINT, INDENT_STRUCTURE, DUCK_TYPING, RUNTIME_FAILURE |
| **boss conversation** | ALTERNATE_TURNS, VERBAL_ACT, ASYMMETRIC_INFO, GENERAL_SUM, TWO_PLAYER, GRADED_REFUSAL, SOCIAL_NORM, FACE_SAVING, IRREVERSIBLE_UTTERANCE, GOAL_AGREEMENT, POWER_ASYMMETRY |

Note `ALTERNATE_TURNS` and `TWO_PLAYER` shared between chess and a difficult
conversation. That is correct and it is the point: they really do share those
mechanics, and the measure says they share almost nothing else.

### 8.6 The numbers, and a weighting that turned out wrong

Jaccard over the §8.5 sets, plain and IDF-weighted (`idf(m) = log(N/df(m)) + 1`):

| pair | shared | plain Jaccard | IDF-weighted |
|---|---|---|---|
| rust / python | 8 | **0.50** | 0.42 |
| chess / checkers | 9 | **0.39** | 0.28 |
| checkers / go | 6 | 0.27 | 0.18 |
| **chess / go** | 6 | **0.23** | 0.14 |
| chess / boss conversation | 2 | 0.07 | 0.04 |
| chess / rust | 1 | 0.03 | 0.02 |
| rust / boss conversation | 0 | 0.00 | 0.00 |

**The pathology is gone.** chess ↔ go now sits at 0.23, correctly ranked below
chess ↔ checkers (0.39) and far above anything cross-category (≤ 0.07). No
ordering, no forest, no patch — set overlap is symmetric and order-free, so there
is nothing left for an axis ordering to get wrong.

**IDF weighting is wrong here, and the numbers are how we know.** It makes every
similarity *lower* — chess ↔ checkers falls from 0.39 to 0.28 — because what
those two games **share** is the common mechanics (`GRID_BOARD`,
`ALTERNATE_TURNS`, low IDF) while what they **differ** on is the rare ones
(`CASTLING`, `EN_PASSANT`, `KO_REPETITION`, high IDF). IDF therefore amplifies
idiosyncratic detail at the expense of structural agreement, which is the exact
opposite of what a similarity measure should do. Chess and checkers are alike
*despite* castling.

The resolution is that three different jobs want three different measures, and
conflating them was the mistake:

| job | measure | why |
|---|---|---|
| **Similarity** — how alike are these two games | **plain Jaccard** `|A∩B| / |A∪B|` | symmetric, order-free, and not distracted by rare detail |
| **Hierarchy** — is A a generalisation of B | **containment** `|A∩B| / min(|A|,|B|)`, and exactly `A ⊆ B` for the strict case | asymmetric, which generalisation is |
| **Discrimination** — which mechanic to probe next | **IDF** | a rare mechanic splits the candidate set best, which is precisely what makes it useless for similarity and ideal here |

IDF keeps its place in §17 (probe selection), where being rare is the whole
point. It is removed from similarity.

### 8.7 The general class computes itself

§19.5 needed the deepest terminal node every candidate descends from. Decomposed,
that definition becomes an operation:

```
general(C) = ⋂_{g ∈ C} mechanics(g)
```

For `C = {chess, checkers, go}` that is exactly

```
{ALTERNATE_TURNS, GRID_BOARD, PERFECT_INFO, ZERO_SUM, TWO_PLAYER, PIECE_OWNERSHIP}
```

— "a two-player, perfect-information, zero-sum grid board game with piece
ownership". The general game class is the **intersection of the candidates'
mechanic sets**, it is guaranteed to be a valid class because every candidate has
every one of its mechanics, and its rule set is exactly the rules those mechanics
carry. It can be packaged and handed to GTMNN (§22) with honest confidence while
identification continues underneath, and the generalisation hierarchy as a whole
is **subset inclusion on mechanic sets** rather than a prefix relation that
depended on an ordering.

### 8.8 What this does to the axis catalogue

The axes of §9 become **derived**. `turn_structure = alternating` is just "has
`ALTERNATE_TURNS`"; `information = perfect` is "has `PERFECT_INFO`";
`legality_density` is a statistic over the operator set. The catalogue stops
being hand-authored primary data and becomes a **human-readable summary** of the
mechanic set — which is a real job worth keeping, for two reasons:

* **Declared evidence stays coarse and free.** A description says "two players, alternating turns, capture to win" long before any operator has been induced. The axes are the right shape for that, and §9.1's declared/measured split is unchanged.
* **People read axes; nobody reads 300 mechanics.** `gren axes` stays the human view; `gren mechanics` is the machine's.

What changes is that the axes are no longer the **basis of similarity**. That is
now the mechanic set, and §21 is written against it.

### 8.9 `primitive.py`

```python
@dataclass(frozen=True)
class Operator:
    precondition: tuple         # conjunction of predicate applications (§14)
    action: str
    effect: tuple
    reason_class: int | None    # the refusal this operator explains, when it is a prohibition
    game: int ; support: int ; confidence: float

@dataclass(frozen=True)
class Mechanic:
    id: int ; name: str
    exemplar: Operator
    canonical: str
    games: frozenset[int]       # corpus games exhibiting it
    df: int                     # document frequency, for IDF (§17 only)
    atomic: bool                # False while §8.3 still finds independent sub-parts
    epoch: int                  # bumped by resplit()

class MechanicVocab:
    mechanics: list[Mechanic] ; epoch: int
    def lift(self, ops, ) -> list[int]                  # §8.4
    def resplit(self, corpus) -> int                     # §8.3; returns splits made, bumps epoch
    def signature(self, game: int) -> frozenset[int]     # the game's mechanic set
    def jaccard(self, a, b) -> float                     # §8.6 — the similarity measure
    def containment(self, a, b) -> float                 # §8.6 — the hierarchy measure
    def idf(self, m: int) -> float                       # §8.6 — probe selection only
    def general(self, games) -> frozenset[int]           # §8.7 — the intersection
    def to_dict(self) / from_dict(cls, d)

def decompose(rules: RuleSet, features: FeatureLanguage) -> list[Operator]
    # Rule -> Operator: the rule's predicate becomes the precondition, its reason class the
    # effect, the action it was observed on the action. Rules with `source="inherited"` are
    # NOT decomposed -- a borrowed rule is not evidence about this game's parts, and
    # decomposing it would let one game's structure propagate through the corpus as if it
    # had been observed everywhere it was copied to.
```

That last clause is the one that keeps the corpus honest. Inheritance (§16.4) is
what makes identification cheap; letting inherited rules feed back into
decomposition would turn a single observation into apparent corroboration across
every game that borrowed it, and the vocabulary would converge on whatever the
first few games happened to look like.

---

## 9. The axis catalogue — what we know about a game

An axis is one question about a game with a small discrete answer set. The
catalogue is seeded with 22 and **grows**: `axis.discover()` (§18.3) proposes new
axes when the corpus contains games the existing ones cannot separate.

Each axis carries a `source`, and the distinction does real work:

| source | meaning | cost |
|---|---|---|
| `DECLARED` | known from the description before a single probe — you read the box before you play | **free** |
| `MEASURED` | resolved only by probing the oracle | probes |
| `INFERRED` | filled in from the nearest neighbours once the candidate set is narrow | free, and flagged as inherited |

Retrieval on declared axes alone routinely cuts a thousand-game corpus to a
handful. The probe budget is then spent only on what the description did not say,
which is the single largest efficiency in the system.

### 9.1 The seeded axes

**Cardinality — how big is it** (declared)

| # | axis | buckets |
|---|---|---|
| 0 | `players` | 1 / 2 / 3-6 / 7+ / open |
| 1 | `input_arity` | 1-4 / 5-32 / 33-1k / 1k-1M / open |
| 2 | `output_arity` | 1 / 2-4 / 5-32 / 33+ / open |
| 3 | `state_arity` | finite-small / finite-large / countable / continuous / unknown |

**Teleology — what are you trying to do** (declared)

| # | axis | buckets |
|---|---|---|
| 4 | `goal_type` | reach-target / maximise-score / outlast / produce-artifact / reach-agreement / discover / survive / none |
| 5 | `goal_horizon` | single-shot / episodic / continuous |
| 6 | `win_condition` | absorbing / threshold / relative-ranking / none |
| 7 | `adversarial` | solitaire / competitive / cooperative / mixed |

**Category — what kind of thing is it** (declared)

| # | axis | buckets |
|---|---|---|
| 8 | `category` | board / card / word-language / programming / physical / social / economic / puzzle / construction / negotiation |
| 9 | `medium` | symbolic / spatial / verbal / physical / mixed |

**Structure — the game-theoretic shape** (declared or measured)

| # | axis | buckets |
|---|---|---|
| 10 | `turn_structure` | simultaneous / alternating / free / real-time |
| 11 | `information` | perfect / imperfect / hidden / asymmetric |
| 12 | `determinism` | deterministic / stochastic / adversarial-stochastic |
| 13 | `sum` | zero-sum / constant-sum / general-sum / common-payoff |
| 14 | `termination` | bounded / unbounded / absorbing |
| 15 | `reversibility` | reversible / irreversible / partial |

**The refusal boundary — what it actually rejects** (measured; §2's claim, made into coordinates)

| # | axis | buckets |
|---|---|---|
| 16 | `legality_density` | <1% / 1-10% / 10-50% / 50-90% / >90% |
| 17 | `reason_taxonomy` | 1 / 2-4 / 5-16 / 17-64 / 65+ |
| 18 | `reason_entropy` | low / medium / high |
| 19 | `boundary_locality` | markov-1 / markov-k / full-history |
| 20 | `verdict_latency` | instant / fast / slow / human |
| 21 | `oracle_fidelity` | exact / noisy / graded / absent |

Axis 21 is the one that decides whether a domain is learnable at all. `graded`
means "kind of illegal" — the social case, where nobody stops you mid-sentence
but the refusal is real and arrives later. `absent` means no oracle: GREN can
place such a domain by its declared axes and can never measure it, and it says so
rather than inventing a boundary.

### 9.2 Worked signatures

Seven domains over the seeded catalogue, axes shown in information-gain order:

| axis | chess | checkers | go | rust | python | boss conversation | guitar |
|---|---|---|---|---|---|---|---|
| `category` | board | board | board | programming | programming | social | physical |
| `goal_type` | reach-target | reach-target | **maximise-score** | produce-artifact | produce-artifact | reach-agreement | produce-artifact |
| `adversarial` | competitive | competitive | competitive | solitaire | solitaire | mixed | solitaire |
| `medium` | spatial | spatial | spatial | symbolic | symbolic | verbal | physical |
| `information` | perfect | perfect | perfect | perfect | perfect | asymmetric | perfect |
| `players` | 2 | 2 | 2 | 1 | 1 | 2 | 1 |
| `determinism` | deterministic | deterministic | deterministic | deterministic | deterministic | stochastic | stochastic |
| `oracle_fidelity` | exact | exact | exact | exact | exact | **graded** | **graded** |
| `sum` | zero-sum | zero-sum | zero-sum | common-payoff | common-payoff | general-sum | common-payoff |
| `turn_structure` | alternating | alternating | alternating | free | free | alternating | real-time |
| `legality_density` | 1-10% | 1-10% | 50-90% | <1% | 10-50% | >90% | >90% |
| `input_arity` | 1k-1M | **33-1k** | 33-1k | open | open | open | open |
| `reason_taxonomy` | 5-16 | **2-4** | 2-4 | 65+ | **17-64** | 17-64 | 5-16 |

### 9.3 What the numbers say, including the part that breaks

Similarity computed two ways over the table above — `prefix` is agreement up to
the first divergence (what the tree gives you for free), `agreement` is
IG-weighted agreement across all axes:

| pair | divergence depth | prefix | agreement |
|---|---|---|---|
| chess ↔ checkers | 11 | 0.71 | **0.93** |
| rust ↔ python | 10 | 0.66 | **0.92** |
| checkers ↔ go | 1 | 0.08 | **0.78** |
| **chess ↔ go** | **1** | **0.08** | **0.71** |
| rust ↔ guitar | 0 | 0.00 | 0.43 |
| chess ↔ boss conversation | 0 | 0.00 | **0.13** |
| python ↔ boss conversation | 0 | 0.00 | 0.11 |

The first two rows and the last two are the design working exactly as claimed:
chess and checkers at 0.93, rust and python at 0.92, chess and a difficult
conversation at 0.13. Information clustering, reproduced numerically from the
axes rather than asserted.

**The chess ↔ go row is the important one, because it is the design failing.**
Chess and go agree on eleven of thirteen displayed axes — both are two-player,
perfect-information, deterministic, zero-sum, alternating, spatial board games —
and share 0.71 of the IG-weighted mass. But they differ on `goal_type` at depth
1, so **prefix similarity collapses to 0.08 and the tree says they are
unrelated.** They are not unrelated; they differ on one early high-information
axis and agree on nearly everything after it.

This is not a bug to patch. It is a property of committing to a single axis
ordering, and it has exactly one correct fix: **do not commit to one ordering.**
Build many trees over randomised orderings and let two games be similar if they
land together *often*, not if they happen to agree on whichever axis was sorted
first. That is a random forest, it is where §20 comes from, and the `chess ↔ go`
case is its regression test (`test_forest.py` asserts forest proximity recovers
`> 0.6` where prefix similarity gives `0.08`).

Prefix similarity is kept because it is free, it is what the radix tree computes
during retrieval anyway, and it is correct when the divergence is deep. It is not
the default similarity and it is never the basis of a `GamePackage`.

---

## 10. `verdict.py`

```python
class Outcome(IntEnum):
    LEGAL = 0        # accepted, play continues
    ILLEGAL = 1      # refused, with a reason — the informative case
    TERMINAL = 2     # the game ended (win/loss/draw/artifact accepted)
    ERROR = 3        # the oracle itself failed (timeout, crash, no answer) — never a rule signal

@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    reason: str | None          # the oracle's own words, verbatim and unmodified
    reason_code: str | None     # structured code when the oracle offers one ("E0308", "check")
    reason_class: int | None    # filled by reasons.py (§15); None until classified
    payoff: float | None        # only when TERMINAL
    latency: float
    raw: dict                   # anything else the oracle returned, untouched
    def to_dict(self) -> dict
```

`ERROR` is kept strictly separate from `ILLEGAL`. A compiler that times out has
not taught you a rule, and folding the two together poisons the boundary with the
oracle's own failures — the single most likely way to learn confident nonsense.
`Explorer` tracks an `error_rate` and aborts a run above `max_error_rate = 0.25`
rather than continuing to probe an oracle that is not answering.

`reason` is stored **verbatim**. Normalisation (stripping paths, line numbers,
timestamps, memory addresses, temp-file names) happens in `reasons.normalise()`
and produces a separate field; the original is never overwritten, because the
normalisation rules are themselves a guess that will be revised.

---

## 11. `action.py`

```python
@dataclass(frozen=True)
class Action:
    payload: Any                # domain-specific: a move, a source file, an utterance
    repr: str                   # a stable, hashable rendering, for logs and dedup
    derivation: tuple | None    # the grammar productions that built it (§11.2), or None

class ActionSpace(Protocol):
    def sample(self, state, rng) -> Action              # uniform-ish, the baseline
    def enumerate(self, state) -> Iterable[Action] | None   # None when the space is open
    def size(self, state) -> int | None                 # None = open; drives axis 1
    def neighbours(self, action, k: int) -> list[Action]    # small edits — the boundary prober
```

### 11.1 Open action spaces

Chess has ~4000 candidate moves; Rust has infinitely many programs. `enumerate`
returning `None` is the normal case for the interesting domains, and everything
downstream must work without it — which is why the probe policies (§17) are
generate-and-score rather than argmax-over-enumeration.

### 11.2 Grammar-driven generation

For open spaces the action space is a **grammar**, and a probe is a derivation:

```python
class GrammarActionSpace(ActionSpace):
    def __init__(self, grammar: dict[str, list[list[str]]], start: str, rng, max_depth: int = 12)
    # A probe is built by sampling productions from `start`. `derivation` records the
    # production path, which is what makes a refusal ATTRIBUTABLE: when a program is
    # rejected, the derivation says which productions were involved, and rule induction
    # (§16) searches over productions rather than over characters.
```

This is the difference between learning "this 400-character string was rejected"
and "a `let` binding without a type annotation, followed by a use at a different
type, is rejected". Grammars for the reference oracles live in `data/grammars/`.

### 11.3 `neighbours` — the boundary prober

Small edits to a known-legal action land near the boundary, which is where
`p_legal ≈ 0.5` lives (§4). `neighbours` is how the `boundary` policy (§17.5)
gets its candidates cheaply without a full posterior sweep, and it is the single
most effective policy on grammar-driven domains.

---

## 12. `oracle.py`

```python
class Oracle(Protocol):
    name: str
    fidelity: str               # "exact" | "noisy" | "graded" | "absent"  -> axis 21
    def reset(self, seed: int = 0) -> Any                      # returns an opaque state
    def probe(self, state, action: Action) -> Verdict
    def actions(self, state) -> ActionSpace
    def declared(self) -> dict[int, str]                       # the axes it knows about itself (§9.1)
    def describe(self) -> dict
```

`declared()` is how a domain hands over its free evidence. An oracle that knows
it is two-player, alternating and zero-sum says so, and those axes cost nothing.
An oracle that declares nothing is legitimate and simply makes identification
slower.

### 12.1 Reference implementations

| Oracle | fidelity | what it is |
|---|---|---|
| `GrammarOracle` | exact | A known regular or context-free language. **Ground truth is available**, so this is what validates GREN itself (§28). |
| `BoardGameOracle` | exact | A rules engine (chess, checkers, go). Small, instant, fully enumerable — the easy end. |
| `InterpreterOracle` | exact | `python -c` in a sandbox. Syntax *and* runtime errors, ~100 exception kinds. |
| `CompilerOracle` | exact | `rustc` / `gcc` / `tsc` in a sandbox. **The reference**: ~500 diagnostic codes, sub-second, self-documenting taxonomy. |
| `TestSuiteOracle` | noisy | A repository's own test suite. Slow, flaky, and the honest bridge to real work. |
| `GradedOracle` | graded | A wrapper returning a *degree* of refusal in `[0,1]` rather than a bit — the social case, where nothing stops you mid-sentence. Thresholded at `0.5` for legality with the degree kept in `raw`. |

### 12.2 Why `CompilerOracle` is the reference

Because of §5's middle column: it is the only oracle where a rule set large
enough to be worth inducting is also affordable to induct. It is free, instant,
complete, never bored, never lies, and — uniquely — **ships with its own reason
taxonomy already enumerated and documented**. `reason_code` for rustc is the
error index; §15 gets its clusters handed to it rather than having to discover
them. Every other oracle is measured against how close it comes to that.

---

## 13. `sandbox.py`

Real oracles run untrusted generated input. Probing a compiler means executing a
program GREN wrote, and the `eig` policy is *actively seeking* inputs that break
things.

```python
def run(argv: list[str], stdin: bytes = b"", timeout: float = 10.0,
        cwd: str | None = None, memory_mb: int = 512, env: dict | None = None) -> SandboxResult
    # subprocess with: a fresh temp cwd removed afterwards, RLIMIT_AS / RLIMIT_CPU / RLIMIT_NOFILE
    # via preexec_fn, a hard timeout with process-group kill, a scrubbed environment (no
    # inherited secrets), output truncated to `max_output = 1 MiB`, and NO network namespace
    # access where the platform allows dropping it.
```

Rules, non-negotiable:

* **Nothing generated is ever executed outside `run()`.** `InterpreterOracle` and `TestSuiteOracle` execute generated code by definition; that is exactly why they go through here.
* A timeout or a kill is `Outcome.ERROR`, never `ILLEGAL` (§10).
* Temp directories are removed on every path including the timeout path.
* `TestSuiteOracle` against a real repository runs against a **copy**, never the working tree.

The Docker Compose stack runs the oracle containers with `network_mode: none` and
a read-only root filesystem; `docker/entrypoint.sh` documents the boundary. This
is a design constraint rather than a deployment detail: a probe policy optimising
for "things that break the oracle" is a fuzzer, and it should be run like one.

---

## 14. `features.py`

Rule induction searches over predicates, and the predicates come from the domain.
This is where domain knowledge legitimately lives, and pretending otherwise would
be the design's biggest lie.

```python
class Feature(Protocol):
    name: str ; arity: int
    def __call__(self, state, action: Action) -> Any      # a small hashable value
    def buckets(self) -> list[Any] | None

class FeatureLanguage:
    def __init__(self, features: Sequence[Feature])
    def evaluate(self, state, action) -> tuple            # the feature vector, positionally stable
    def conjunctions(self, max_terms: int = 3) -> Iterator[Predicate]
        # Candidate predicates: conjunctions of at most `max_terms` (feature == value) tests.
        # Enumerated lazily in order of increasing arity, so induction finds the SIMPLEST
        # rule that explains a refusal class before a baroque one.
```

Built-in languages: `grammar_features` (which productions appear, at what depth,
how many times — domain-general for any `GrammarActionSpace`), `board_features`
(piece kinds, squares, distances, occupancy), `text_features` (token n-grams,
lengths, balance of delimiters).

**The honest limitation.** A rule outside the feature language cannot be induced,
only observed as an unexplained refusal class. `RuleSet.unexplained` (§16) counts
those and it is reported rather than buried — a large `unexplained` means the
feature language is too weak for the domain, which is a finding about the setup
and not a failure of the search. §32 keeps this open.

---

## 15. `reasons.py` — the oracle's own words

```python
def normalise(reason: str) -> str
    # Strips what varies between runs but not between rules: absolute paths, line/column
    # numbers, temp-file names, memory addresses, timestamps, durations, and integer
    # literals appearing outside quotes. Deterministic, order-independent, and total.

def signature(reason: str) -> str       # normalise() then collapse whitespace and lowercase

class ReasonTaxonomy:
    classes: list[ReasonClass]          # each: id, exemplar, signature set, count, code
    def classify(self, v: Verdict) -> int
        # 1. If `reason_code` is present, it IS the class. Compilers have already done this
        #    work; discovering clusters over rustc output when rustc hands you E0308 would be
        #    inventing a worse version of an existing answer.
        # 2. Else match the normalised signature exactly against known classes.
        # 3. Else agglomerate by token-level Jaccard >= `merge_tau` (0.6) against exemplars.
        # 4. Else open a new class.
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d)
```

The taxonomy is what turns a bit into `log2(R+1)` bits (§4) and it directly
populates axes 17 and 18. Its size and entropy are themselves coordinates: a
domain with one dominant failure mode is a genuinely different *kind* of domain
from one with 500 evenly-used diagnostics, and the tree can see that difference
without knowing anything about either.

Order-independence is asserted (`test_reasons.py`): classifying the same multiset
of verdicts in any order yields the same partition. An agglomerative clusterer
that quietly depends on arrival order would make every downstream signature
non-reproducible.

---

## 16. `rule.py`

```python
@dataclass
class Rule:
    predicate: Predicate        # from the feature language (§14)
    reason_class: int           # which refusal this predicate explains
    support: int                # probes matching the predicate that were refused this way
    contradictions: int         # probes matching the predicate that were NOT
    confidence: float           # Wilson lower bound at 95% on support/(support+contradictions)
    source: str                 # "induced" | "inherited" | "declared"
    inherited_from: int | None  # corpus game id, when source == "inherited"
    def predicts(self, state, action) -> bool

@dataclass
class RuleSet:
    rules: list[Rule]
    unexplained: Counter        # reason_class -> probes no rule explains (§14)
    coverage: float             # fraction of observed refusals some rule explains
    def legality(self, state, action) -> tuple[bool, int | None]   # (legal?, reason_class)
    def merge(self, other: "RuleSet", weight: float) -> "RuleSet"
    def to_dict(self) -> dict
```

Confidence is a **Wilson lower bound**, not a raw ratio. A rule at 3/3 and one at
300/300 are not equally trustworthy, and a raw ratio calls them both 1.0; Wilson
gives 0.44 and 0.99. Every threshold in the system compares against the bound.

### 16.1 Induction by information gain

Candidate predicates from `FeatureLanguage.conjunctions()`, scored by information
gain about `reason_class` over observed probes, greedily accepted while the
Wilson bound clears `min_confidence = 0.8` and the predicate is not subsumed by
an accepted one. This is decision-tree induction over the feature language — the
same algorithm as the axis ordering (§18.2), pointed at rules instead of games.

### 16.2 Version space

For domains small enough to enumerate (`GrammarOracle`, board games), a proper
candidate-elimination version space is maintained alongside: the most-general and
most-specific consistent boundaries. It is exact, it collapses to the true rule
set when one exists in the language, and it is what makes `test_rule.py` able to
assert *correctness* rather than merely plausibility. It is disabled above
`version_space_max = 10^4` candidates.

### 16.3 Consistency and revision

A contradiction (a probe refused by a rule but accepted by the oracle) decrements
confidence rather than deleting the rule — oracles are noisy and a `graded` one
is noisy by construction. A rule falling below `retire_confidence = 0.5` is
retired to `RuleSet.retired`, not discarded, so a rule that was right all along
and hit a run of noise can be revived rather than re-induced from nothing.

### 16.4 Inheritance — the point of the whole design

```python
def inherit(candidates: "Candidates", corpus: "Corpus", evidence: Evidence) -> RuleSet
    # Posterior-weighted merge of the candidates' rule sets:
    #   confidence(rule) <- Σ_g p(g) · confidence_g(rule)   over candidates g that hold it
    # Rules held by ALL candidates are inherited at full confidence: they do not discriminate
    # between the remaining possibilities, so no probe will ever be spent on them.
    # Rules held by SOME are inherited at partial confidence and become the probe targets (§17.4):
    #   they are exactly the places where the candidates disagree, which is exactly where a
    #   probe is worth spending.
    # Every inherited rule carries source="inherited" and inherited_from, so the provenance of
    # a rule GREN never verified itself is visible in the package rather than laundered.
```

That last sentence is the crux of §3.1. The reason identification is three orders
of magnitude cheaper than induction is that **most of a game's rules are never
probed at all** — they are inherited from neighbours because every surviving
candidate agrees on them. Probing is spent only on the disagreements. The risk is
equally clear and is why provenance is tracked: an inherited rule is a bet on the
neighbour being right, and `GamePackage.confidence` (§22) is reduced in proportion
to how much of the rule set was never verified.

### 16.5 Cold start

When no candidate clears `τ_novel`, there is nothing to inherit and induction
runs from nothing over the full probe budget. This is the expensive path, it is
correctly rare once the corpus is populated, and its product is a new corpus
entry — so a given cold start is paid for exactly once, ever.

---

## 17. `probe.py` — choosing what to try next

```python
class ProbePolicy(Protocol):
    name: str
    def propose(self, state, space: ActionSpace, evidence: Evidence,
                candidates: "Candidates", rules: RuleSet, rng, budget: int = 64) -> Action
        # `budget` = how many candidate actions may be generated and scored internally.

def get_policy(name: str) -> ProbePolicy      # "random" | "uncertainty" | "eig" | "adversarial" | "boundary" | "bandit"
```

### 17.1 `random`
Uniform from the action space. The honest baseline and the control condition —
every other policy is reported as a multiple of `random`'s probes-to-identify,
because a policy that cannot beat random is not a policy.

### 17.2 `uncertainty`
Maximises `H(Bernoulli(p_legal))`, i.e. picks the action closest to
`p_legal = 0.5` under the current rule set. Cheap, and the direct expression of
§4's binary result.

### 17.3 `eig` — the default
The full multi-outcome expected information gain about game identity:

```
p(V = v | a) = Σ_{g ∈ C} p(g) · p_g(V = v | a)              over v ∈ {LEGAL} ∪ reason classes
EIG(a)       = H[p(V | a)] − Σ_{g ∈ C} p(g) · H[p_g(V | a)]
```

Each candidate's `p_g(V|a)` comes from its inherited `RuleSet.legality`, which
returns a reason class as well as a bit — so a probe that splits candidates by
*which* refusal they predict scores far above one that merely splits legal from
illegal. This is where `log2(R+1)` bits per probe instead of 1 actually comes
from, and it is why rich oracles identify faster than the halving bound (§4).

Scored over `budget` sampled actions, not enumerated — open action spaces are the
normal case (§11.1).

### 17.4 The candidate-splitting objective
`eig` reduces exactly to "find the probe that half the surviving candidates call
legal and half call illegal" when reasons carry no information, and that reduction
is worth stating because it is the whole intuition: **probe the disagreement.**
The targets are the partially-inherited rules from §16.4 — the ones some
candidates hold and others do not. A rule every candidate agrees on is never
probed, because no verdict on it could change the answer.

### 17.5 `boundary`
Takes a known-legal action and walks `space.neighbours(a, k)` outward until the
rule set's confidence in legality crosses 0.5. Cheap, needs no posterior sweep,
and on grammar-driven domains it is the strongest policy in the suite because
small derivation edits land exactly on the boundary by construction.

### 17.6 `adversarial`
Maximises `p_legal(a) · 1[oracle refuses]` in expectation — deliberately probes
where the rule set is *confident and possibly wrong*, rather than where it is
uncertain. This is failing on purpose in the strong sense: not "try something you
don't know" but "try something you are sure about, to find out that you are not".
It finds rule-set errors that `eig` never targets, because `eig` avoids exactly
the confident regions where a wrong rule hides. Carries forward the same instinct
as `RadixCyclicNN`'s evolve-on-failures pass.

### 17.7 `bandit` — the default in `Explorer`
The policies are arms; the reward is bits of information actually gained per
probe (measured post-hoc as the entropy drop in `Evidence`), pulled by UCB1.
Policies have genuinely different strengths by domain and by phase — `boundary`
early, `eig` in the middle, `adversarial` to audit — and picking one up front is
a guess the system can make better than a person can.

---

## 18. `axis.py`

```python
@dataclass(frozen=True)
class Axis:
    id: int ; name: str ; buckets: tuple[str, ...]
    source: str                  # "declared" | "measured" | "inferred"
    measure: Callable | None     # for measured axes: (evidence, rules) -> bucket index | None

AXES: tuple[Axis, ...]           # the §9.1 catalogue, index == id
def ig(axis_id: int, corpus: "Corpus") -> float          # information gain over the corpus
def order(corpus, rng=None, jitter: float = 0.0) -> list[int]
    # Axis ids sorted by descending IG; `jitter` randomises among near-ties and is what
    # §20 uses to build a forest of genuinely different orderings.
```

### 18.1 `Evidence`
As §3.2. `update(verdict)` folds a probe into the posterior; `measure(axis)` runs
a measured axis's `measure()` when enough probes have accumulated.

### 18.2 Ordering by information gain
Standard decision-tree criterion over the corpus: the axis whose buckets best
partition known games goes first. This is the same algorithm as rule induction
(§16.1) pointed at games instead of refusals, which is why "descending the tree is
a decision tree" is an identity rather than an analogy.

### 18.3 Axis discovery
When two corpus games share a full signature but have materially different rule
sets (`RuleSet` Jaccard below `split_tau = 0.6`), the catalogue cannot separate
them and a new axis is proposed: the highest-IG feature-language predicate that
distinguishes their refusal sets, promoted to an axis with `source="measured"`.
The catalogue therefore grows in response to genuine confusion rather than
speculation, and `n` is a measured property of the corpus rather than a constant.

---

## 19. `radix.py` — a radix tree with the mechanics of a trie

The core structure. Both halves are load-bearing and neither alone suffices.

### 19.0 What the tree indexes, after §8

The tree is built over **mechanic sets**, not axis vectors. A game's mechanics are
sorted by descending corpus frequency and that sorted list is the signature
inserted — so games sharing the commonest mechanics share the longest prefixes.
This is an FP-tree (a prefix tree over frequency-sorted itemsets) with radix
compression, and it preserves both halves below exactly:

* **Terminal-on-internal** still marks a general game class, and §8.7 now *defines* it rather than describing it: an internal node's prefix is precisely the mechanic set every descendant shares, so `general` is the intersection and is computed, not searched for.
* **Path compression** pays better than before. Mechanic sets are sparse — a few hundred mechanics exist, a game has ~15 — so long runs of shared common mechanics collapse to single nodes.
* The **generalisation hierarchy is subset inclusion** on mechanic sets (§8.7), which is what a prefix relation was always approximating and could only do under a fixed ordering.

`axis_order` below reads `item_order`: the frequency ordering of mechanics.

### 19.1 Why both

| From the **trie** | Why it is required |
|---|---|
| **Terminal flag on any node, internal ones included** | A game class can be a strict prefix of another. "Two-player perfect-information deterministic board game" is a real, useful class *and* a prefix of chess. A structure that only stores games at leaves cannot represent the general class at all, and answering "I don't know if it's chess or checkers, but I'm certain it's a two-player perfect-information board game" is a genuinely useful answer (§19.5). |
| **Prefix semantics** | Retrieval on partial evidence walks a prefix and stops. That is the whole retrieval operation. |
| **Split on divergence** | A new game that agrees for a while then differs splits the shared run at exactly the divergence point. |

| From the **radix tree** | Why it is required |
|---|---|
| **Path compression** | With 22+ axes and most of them agreeing across a cluster, an uncompressed trie is a corridor of single-child nodes. Compression means **only the axes that actually discriminate cost a node**, which is the information-theoretically right structure and makes the tree readable by a person. |
| **Split / merge mechanics** | Identical in spirit to `RadixCyclicNN/DESIGN.md` §5.2 — the same operations the author already built, applied to game-space instead of character-space. |

**The interaction is the specification**, and it is one rule:

> A terminal node is never merged away.

Path compression merges unary chains; trie mechanics say a terminal node carries
meaning. So `merge_child(p)` requires `not terminal[p]` in addition to the usual
unary-chain conditions. Without that clause, compression would silently delete
the general game classes that are the reason for using a trie — which is the one
bug this structure is prone to, and `test_radix.py` asserts it directly.

### 19.2 Storage

```python
class RadixGameTree:
    segment:  list[tuple[tuple[int, int], ...]]   # node -> a compressed run of (axis, bucket) pairs
    children: list[dict[tuple[int, int], int]]    # node -> first (axis,bucket) of the child run -> node
    parent:   list[int]
    terminal: list[bool]                          # trie mechanics: an ending state
    game:     list[int | None]                    # corpus game id when terminal
    rules:    list[RuleSet | None]
    count:    list[int]                           # games at or beneath this node
    alive:    list[bool]                          # tombstones; compacted on save
    item_order: list[int]                         # mechanics by descending corpus frequency (§19.0)
    version: int ; structure_version: int
```

Node 0 is the root with an empty segment, is never terminal, and is never merged.

### 19.3 Operations

```python
def insert(self, signature: tuple, game: int, rules: RuleSet) -> int
    # Walk matching segments. Three cases, exactly as a radix tree:
    #   - the signature runs out mid-segment -> split(node, i), mark the FIRST half terminal
    #     (the new game is a generalisation of what was there)
    #   - the segment runs out and a child matches -> descend
    #   - they diverge at offset i -> split(node, i), add a new child for the remainder
    # Returns the terminal node id. Marking an existing internal node terminal is legal
    # and is how a general class is added above specific ones already present.

def split(self, node: int, i: int) -> tuple[int, int]
    # node keeps segment[:i]; a new node B takes segment[i:]. B inherits node's children,
    # terminal flag, game and rules; node becomes non-terminal. Follows `RadixCyclicNN/DESIGN.md` §5.2.

def merge_child(self, p: int) -> bool
    # Merge p's single child c into p when: len(children[p]) == 1, len(parents of c) == 1,
    # p != c, AND **not terminal[p]**  <-- the trie clause (§19.1). Returns True if merged.

def compress(self) -> int                # repeatedly merge_child until none apply
def lookup(self, signature: tuple) -> int | None                  # exact, terminal nodes only
def path(self, node: int) -> tuple                                # node -> its full signature
def compression_ratio(self) -> float     # axes stored / alive nodes -- how much is shared structure
def to_dict(self) / from_dict(cls, d)    # compacted on save, ids remapped
```

### 19.4 `retrieve` — the operation everything else is for

```python
@dataclass
class Candidates:
    games: list[int]            # corpus ids, best first
    scores: array('d')          # normalised posterior over `games`, sums to 1
    nodes: list[int]            # the tree nodes they came from
    general: int | None         # the deepest TERMINAL node every candidate descends from (§19.5)
    entropy: float              # of `scores`; the honest "how sure am I"
    expanded: int
    def top(self, k: int = 5) -> list[tuple[int, float]]
    def to_dict(self) -> dict

def retrieve(self, evidence: Evidence, beam: int = 32, corpus=None) -> Candidates
    # Best-first walk from the root with a heap, scoring each step by the evidence:
    #     score(path) = Σ_{(axis, bucket) ∈ path}  IG(axis) · log p_evidence(axis == bucket)
    # An axis the evidence has not observed has a uniform posterior, so it contributes
    # equally to every child -- the walk BRANCHES there rather than guessing, which is
    # precisely how partial evidence produces a cluster instead of a point.
    # Every terminal node encountered is collected, internal ones included. `beam` caps the
    # frontier; `expanded` is reported so a truncated search is visible rather than silent.
```

Compression pays off here directly: a run of axes the evidence is certain about
is matched in one step, and only genuinely uncertain axes cost branching.

### 19.5 Answering with an internal node

`Candidates.general` is the deepest terminal node that every surviving candidate
descends from. When the posterior over specific games is flat but they all sit
under one general class, **that class is the answer** — and it is a real answer,
not a hedge. Its `RuleSet` holds exactly the rules all its descendants share, so
it can be packaged and handed to GTMNN (§22) with honest confidence while
identification continues underneath.

This is the concrete payoff of insisting on trie mechanics. A flat clustering can
only say "35% chess, 33% checkers, 32% draughts". The tree says "certainly a
two-player perfect-information capture game; which one is still open" — and hands
over the rules that follow from the part it is certain about.

---

## 20. `forest.py`

**This module's original justification is gone, and that is worth saying plainly
rather than quietly keeping the module.** §20 existed to patch §9.3 — a single
tree calling chess and go unrelated because they differed on one early
high-information axis. Decomposition dissolves that at the source: set overlap is
symmetric and order-free, chess ↔ go scores 0.23 with no forest involved (§8.6),
and there is no axis ordering left for a forest to hedge against.

What survives is weaker but real. Mechanic sets are sparse and high-dimensional,
so a game missing one *frequent* mechanic diverges early from its own cluster in
the frequency ordering and retrieval must branch to find it again. A forest over
bootstrapped orderings hedges that, and it is cheap. `T` drops from 64 to **16**
accordingly, and §32.7 asks whether it should be 1.

```python
class GameForest:
    trees: list[RadixGameTree]          # T = 16
    def __init__(self, T: int = 16, jitter: float = 0.5, bootstrap: float = 0.8, seed: int = 0)
        # Tree t is built over the mechanic frequency ordering of bootstrap sample t, with
        # `jitter` randomising among near-equal frequencies (§19.0).
    def insert(self, signature, game, rules) -> None        # into every tree
    def retrieve(self, evidence, beam=32) -> Candidates
        # Retrieve in each tree; a game's score is its mean posterior across trees.
    def proximity(self, g1: int, g2: int) -> float          # §20.3
    def proximity_matrix(self, games: Sequence[int]) -> list[array]
```

### 20.3 Breiman proximity — the default similarity

```
proximity(g1, g2) = (number of trees where g1 and g2 land in the same terminal node) / T
```

No ordering is privileged, so a game lacking one frequent mechanic still lands
with its cluster in most orderings. Proximity is retained as a **robustness
check** on §8.6's Jaccard: the two measure different things and should correlate
strongly, and `test_forest.py` asserts Spearman correlation above 0.8 between
forest proximity and mechanic Jaccard across the corpus. Large divergence means
the frequency ordering is fragmenting a real cluster.

Mechanic Jaccard (§8.6) — not proximity, not prefix similarity — is the basis of
a `GamePackage`.

---

## 21. `similarity.py`

```python
def prefix_similarity(tree, g1, g2) -> float      # IG-weighted agreement up to first divergence
def weighted_agreement(sig1, sig2, ig) -> float   # IG-weighted agreement over ALL axes, order-free
def distance_matrix(games, metric="proximity") -> list[array]     # 1 - similarity
def mds(D, dim: int = 2, iters: int = 256) -> list[tuple[float, ...]]
    # Classical metric MDS: double-centre D∘D, then the top `dim` eigenvectors by power
    # iteration with deflation. Pure stdlib, ~60 lines, deterministic given a seed.
def neighbours(game, k=8, metric="proximity") -> list[tuple[int, float]]
```

`mds` is the "plane of existence" made literal — it is what the frontend plots,
and it is the picture in which chess and checkers sit next to each other and a
difficult conversation sits somewhere else entirely. It is a *projection* for
human consumption; nothing in the pipeline consumes the 2D coordinates, because
a projection to two dimensions discards real structure and decisions must be made
in the space where that structure still exists.

---

## 22. `package.py` — the handoff to GTMNN

The only interface between part one and part two. Everything above exists to
produce this object.

```python
@dataclass
class GamePackage:
    signature: tuple                    # the rule signature (the path)
    node: int                           # the tree node it came from; may be INTERNAL (§19.5)
    candidates: Candidates              # who else it might still be, and how sure
    alphabet: list[Action]              # the learned LEGAL action set
    rules: RuleSet                      # induced + inherited, each carrying provenance
    legality: Callable                  # (state, action) -> (bool, reason_class | None)
    payoff_class: str                   # which GTMNN Payoff to instantiate (§22.2)
    B: float ; lam: float               # GTMNN's reward / penalty, calibrated (§22.3)
    confidence: float                   # §22.1
    verified_fraction: float            # rules GREN probed itself / all rules
    provenance: dict                    # probes spent, inherited-from ids, unexplained classes
    def to_dict(self) -> dict ; @classmethod from_dict(cls, d)
```

### 22.1 Confidence, and the refusal to hand over

```
confidence = candidates.scores[0] · rules.coverage · (0.5 + 0.5 · verified_fraction)
```

Three independent ways of being wrong, multiplied rather than averaged so that
any one of them being bad is fatal: you might have the wrong game, your rules
might not explain what you have seen, or your rules might all be borrowed from a
neighbour you never checked.

**`GTMNet.load_package()` raises below `τ_conf = 0.8`.** This is the enforcement
of "once you understand the game, then and only then can you play the game" — a
package that does not know what game it is holding is refused, and `Explorer`
keeps probing instead. The internal-node package (§19.5) is the graceful path: it
is usually *more* confident than any specific candidate, because it only claims
what they all agree on.

### 22.2 Mapping a signature to a GTMNN payoff

| GREN axes | GTMNN `Payoff` (`GTMNN/DESIGN.md` §9) |
|---|---|
| `adversarial=solitaire` ∧ `sum=common-payoff` | `CorrectnessCongestion` — the standard training game |
| `adversarial=competitive` ∧ `sum=zero-sum` ∧ `turn_structure=alternating` | `CorrectnessCongestion` at play, `BeliefCongestion` at inference against the opponent model |
| `turn_structure=simultaneous` ∧ `sum=general-sum` | `PrisonersDilemma` / `StagHunt` per §9.3-8.4 |
| no pure equilibrium detected in the corpus neighbour | `RockPaperScissors` structure — GTMNN's cycle machinery and meta-player (§15) are the intended handler |
| `oracle_fidelity=graded` | `BeliefCongestion` only; the correctness game needs a hard label and there is none |

### 22.3 Calibrating `B` and `λ`, and masking repertoires

```python
def calibrate(rules: RuleSet, evidence: Evidence) -> tuple[float, float]
    # GTMNN's open question 1 (GTMNN/DESIGN.md §27) is the B:λ ratio. GREN can answer it,
    # because it has measured the thing the ratio should depend on: legality density (axis 16).
    #   B = 1.0                                   (the unit of reward, fixed)
    #   lam = B · legality_density / (1 - legality_density)
    # so the expected payoff of a uniformly random play is zero. In a domain where 1% of
    # actions are legal, a wrong guess must cost little or nobody bids; where 90% are legal,
    # being wrong must be expensive or everyone bids on everything. A constant cannot be right
    # for both, and GREN is the only part of the system that knows which it is facing.
```

`GamePackage.alphabet` becomes GTMNN's `Alphabet`, and `legality` masks micro
repertoires: `MicroPool.note_symbol` (`GTMNN/DESIGN.md` §7.5) may not promote an
action the rule set calls illegal into a repertoire slot. GTMNN already masks
unfilled slots to `-inf` before its softmax (`GTMNN/DESIGN.md` §7.3), so illegal actions use the
existing mechanism and need no new code on that side — **a micro is structurally
incapable of naming an illegal move.**

### 22.4 The universal interface — one micro geometry for every game

Chess has ~4 000 candidate moves, go has 361, checkers has ~50, Rust has
infinitely many programs. A GTMNN micro is an `R -> H -> out` network with **fixed
geometry**. For the same population to play all of them, every game must present
the same shaped interface, and the game's identity must arrive as part of the
input rather than as a different network.

**Score one candidate at a time.** The micro is never asked "which of the 4 000
moves" — it is asked "this one: legal? and how good?". Action-space size then
never touches the geometry, and chess, go and Rust differ only in how many times
the same network is evaluated.

```
micro input  (R indices drawn from F = F_state + F_cand + F_mod)

    F_state = 256   hashed features of the position / context
    F_cand  = 128   hashed features of the ONE candidate action being scored
    F_mod   = 128   the game modifier: the mechanic set, hashed (§22.5)
                    ------
                    512 total

micro output (fixed, 3 units, every game)

    p_valid   is this move legal?          <- trained by GREN against oracle refusals
    grade     how good is it, if legal?    <- trained by GTMNN against outcome
    abstain   I do not know
```

### 22.5 The game modifier

The modifier is the mechanic set (§8) hashed into `F_mod` dimensions by the same
signed hashing as GTMNN's `FeatureHasher` (`GTMNN/DESIGN.md` §6.2):

```python
def modifier(mechanics: frozenset[int], dim: int = 128) -> array('d')
    # v[h(m) % dim] += ±1 for each mechanic m; then L2-normalise.
```

Hashing a *set* preserves overlap, which is the whole reason this works: for two
games `A` and `B` with few collisions,

```
⟨v(A), v(B)⟩  ≈  |A ∩ B| / sqrt(|A|·|B|)      = the Ochiai similarity of their mechanic sets
```

so **two similar games land at nearby points in modifier space automatically**. A
micro that learned "in games shaped like this, a piece cannot move through an
occupant" generalises to every game with a similar modifier without being told
to, because its inputs barely moved. Transfer is not a mechanism that had to be
designed; it is a consequence of hashing a set instead of assigning an id.

**Sizing it, measured rather than assumed.** Mean absolute error between modifier
cosine and true Ochiai, over 600 random game pairs from a 300-mechanic vocabulary
with 10–20 mechanics per game:

| scheme | dims | mean abs. error | p95 |
|---|---|---|---|
| hashed | 32 | 0.177 | 0.437 |
| hashed | 64 | 0.128 | 0.333 |
| **hashed** | **128** | **0.061** | **0.175** |
| hashed | 256 | 0.037 | 0.134 |
| dedicated top-64 + 32 hashed | 96 | 0.114 | 0.346 |
| dedicated top-96 + 32 hashed | 128 | 0.087 | 0.292 |

`F_mod = 128`, plain hashing. The hybrid that gives dedicated collision-free bits
to the commonest mechanics was tried because §8.6 showed that what similar games
*share* is the common mechanics — and it **lost** at equal width (0.087 vs
0.061), because the rare mechanics it pushes into a smaller hashed remainder then
collide with each other and inject more noise than the dedicated bits remove.
Plain hashing is simpler and better; the table is here so the question is not
reopened from intuition.

### 22.6 Two heads, and why the split is the architecture

Separating **validity** from **grade** is what actually compresses chess, go and
checkers into one space, and it is not merely an output-layer convenience:

| | validity (`p_valid`) | grade (`grade`) |
|---|---|---|
| question | is this move legal? | is this move good? |
| signal | the oracle's refusal — binary, instant, abundant | game outcome — sparse, delayed, expensive |
| trained by | **GREN**, from probes (§16) | **GTMNN**, from Shapley credit |
| transfers across games? | **yes** — legality is structure, and the modifier says which structure | barely — evaluation is game-specific judgment |
| knowable? | yes, exactly | no, only estimated |

Three consequences follow, and each one is load-bearing:

1. **The cheap, abundant, transferable signal trains first and separately.** Validity comes from refusals, which cost one probe each and arrive by the thousand. Grade needs played-out games. Training them jointly would let the sparse, noisy signal contaminate the dense, exact one.
2. **"Understand the game, then play it" becomes true inside a single micro.** The system-level phase ordering of §1 is now also the ordering of the two heads' training, on the same weights. The validity head is the understanding; the grade head is the play; the shared hidden layer is what makes the second cheap once the first is done.
3. **Culling should be asymmetric** (GTMNN §17.1). A micro confidently wrong about *validity* is wrong about something knowable, and should be culled hard. One wrong about *grade* was exercising judgment under uncertainty, which is the job. Weighting both equally would select for timidity.

The congestion game is untouched. A seated micro still backs one candidate from
the `K` on offer — `argmax_k p_valid(k) · grade(k)` — and load is still how many
backed the same one, so `CorrectnessCongestion` (GTMNN §8.1) applies unchanged
with "correct" reading as "legal and best". What changes is that a micro's
**repertoire is filled per step from the candidate shortlist** rather than
learned as a fixed set of symbols, so specialisation becomes *mechanic-relative*
("I am good at candidates involving `DISPLACEMENT_CAPTURE`") rather than
*identity-relative* ("I am good at the letter e"). That is precisely what has to
be true for a micro to be worth anything in a game it has never seen.

### 22.7 Deltas required on the GTMNN side

`GTMNN/DESIGN.md` is written against `F = 256`, a learned per-micro repertoire of
symbol ids, and a single output distribution. Composing the two networks requires
these changes there, listed so they are unambiguous rather than inferred:

| GTMNN section | change |
|---|---|
| GTMNN §6.2 `FeatureHasher` | `F = 256` becomes `F = 512`, partitioned `F_state=256 ⊕ F_cand=128 ⊕ F_mod=128` (§22.4). The hashing itself is unchanged. |
| GTMNN §7.1 `MicroPool` | output width `K+1` becomes a fixed **3** (`p_valid`, `grade`, `abstain`). `repertoire` stops being a learned array of symbol ids and becomes the per-step candidate shortlist. |
| GTMNN §7.3 forward | two heads instead of one softmax; `p_valid` sigmoid, `grade` linear, `abstain` as before. |
| GTMNN §7.4 `learn` | two loss terms, trained in separate passes: validity against GREN's oracle labels, grade against Shapley credit. |
| GTMNN §7.5 `note_symbol` | becomes `note_mechanic`: specialisation is tracked against the mechanics a candidate exhibits, not the symbol it names. |
| GTMNN §6.3 receptive fields | draw with an explicit `universal_fraction` (default 0.5): that share of micros sample their `R` indices from `F_state ⊕ F_cand` **only** and are therefore game-blind and transfer everywhere; the rest may read modifier bits and specialise. Leaving the split to chance would make ~92% of micros game-aware at these widths, which is the wrong balance and not a decision worth making by accident. Evolution (§17) then moves the balance by which kind earns. |
| GTMNN §17.1 culling | asymmetric weighting of validity error against grade error, per §22.6.3. |

`universal_fraction` is the parameter this whole composition turns on, and
nothing here predicts its right value. It is §32.8.

---

## 23. `explorer.py`

```python
@dataclass
class ExploreConfig:
    probe_budget: int = 10_000
    beam: int = 32 ; policy: str = "bandit"
    tau_conf: float = 0.8 ; tau_novel: float = 0.35 ; eig_floor: float = 0.01
    max_error_rate: float = 0.25
    induce_every: int = 64          # probes between rule-induction passes
    report_every: int = 256
    seed: int = 0

class Explorer:
    def __init__(self, oracle: Oracle, corpus: Corpus, forest: GameForest, config=None)
    def run(self, stop_event=None, progress=None) -> GamePackage
    def step(self) -> dict          # one probe; the unit the API streams
```

The loop, per §3.3: seed `Evidence` from `oracle.declared()`; retrieve; if
confident, inherit and package; else choose a probe, run it, classify the reason,
fold it into the evidence, re-induce every `induce_every` probes, repeat.

### 23.4 The report record

```python
{"probes", "budget", "seconds", "failure_rate", "error_rate",
 "candidates": [(game, score)], "entropy", "general", "known_fraction",
 "bits_gained", "bits_per_probe", "policy_pulls": {name: count}, "policy_reward": {name: bits},
 "rules": {"induced", "inherited", "retired", "coverage", "unexplained"},
 "reason_classes", "confidence", "axes_measured", "axes_declared"}
```

`failure_rate` is the headline diagnostic (§4): it should sit near 0.5 under
`eig`, and a run at 0.05 or 0.95 is not exploring. `bits_per_probe` against
`log2(R+1)` says how much of the oracle's bandwidth is actually being used —
a low ratio means reasons are being wasted, which usually means the taxonomy
(§15) has collapsed distinct failures into one class.

---

## 24. `checkpoint.py`, `bench.py`

`CheckpointManager` is the same contract as `RadixCyclicNN/radixnet/checkpoint.py`
(rotation, `latest` pointer, atomic `os.replace` writes, resume). It checkpoints
the corpus, the forest and the in-flight `Evidence`, so a 10 000-probe run against
a slow oracle survives a restart without re-probing.

```python
def bench(oracle, corpus, forest, probes=1000) -> dict
    # {"probes_per_sec", "bits_per_probe", "probes_to_identify", "retrieve_per_sec",
    #  "oracle_latency_p50/p95", "breakdown": {"oracle": pct, "retrieve": pct,
    #  "induce": pct, "policy": pct}}
```

`probes_to_identify` against the `random` baseline is the number that says whether
any of this works.

---

## 25. `cli.py`

`python -m gren <command>`; global `--corpus DIR`, `--forest PATH`, `--oracle NAME`, `--seed`, `--json`.

| command | args | behaviour |
|---|---|---|
| `identify` | `--oracle`, `--declare k=v ...`, `--budget`, `--policy`, `--explain` | **the main verb**: probes until identified, prints the candidate table with posteriors, the general class, the failure rate, and the package |
| `probe` | `--oracle`, `--action TEXT`, `-n N` | raw probing; prints verdict, reason, reason class |
| `similar` | `--game NAME`, `-k 8`, `--metric proximity\|prefix\|agreement` | nearest neighbours with scores — the clustering, directly |
| `cluster` | `--metric`, `--mds`, `--out FILE` | the whole corpus: proximity matrix, and 2D MDS coordinates |
| `tree` | `--depth N`, `--node ID` | renders the radix tree: segments, terminals marked, counts, compression ratio |
| `axes` | `--corpus` | the catalogue with measured IG, sorted; flags axes that never discriminate |
| `rules` | `--game NAME`, `--source induced\|inherited` | the rule set with provenance and Wilson bounds |
| `corpus` | `--add NAME --oracle ...`, `--list`, `--remove NAME` | corpus management |
| `package` | `--game NAME`, `--out FILE` | writes a `GamePackage` for GTMNN |
| `bench` | `--probes N` | §24 |
| `serve` | `--host --port --frontend-dir` | the API |

---

## 26. `api.py`

Same shape as `RadixCyclicNN/radixnet/api.py` and `GTMNN/gtmnn/api.py`:
`ThreadingHTTPServer`, JSON, CORS, one background job at a time, 409 while busy,
`{"id","type","state","progress","history","error"}` job records so `useJob` is
portable across all three frontends unchanged.

| method & path | response |
|---|---|
| GET `/api/health`, `/api/status` | version; corpus size, forest size, axis count, job |
| POST `/api/identify` | starts an identification job; streams `step()` records |
| GET `/api/job`, POST `/api/job/stop` | job control |
| POST `/api/probe` | one probe, synchronous: verdict + reason + class |
| POST `/api/retrieve` | `{"declared": {...}, "beam": 32}` → `Candidates.to_dict()` — retrieval without probing, which is the cheap "what could this be" query |
| GET `/api/similar?game=chess&k=8&metric=proximity` | neighbours with scores |
| GET `/api/cluster?mds=1` | proximity matrix + 2D coordinates |
| GET `/api/tree?depth=6`, GET `/api/node/<id>` | the radix tree for rendering |
| GET `/api/axes` | the catalogue with measured IG |
| GET `/api/rules?game=...` | rule sets with provenance |
| GET/POST `/api/corpus` | list / add |
| POST `/api/package` | build and return a `GamePackage` |
| GET `/` | serves `frontend_dir`, SPA fallback |

---

## 27. `frontend/`

Vite + React, `react`/`react-dom`/`vite`/`@vitejs/plugin-react` only, hand-rolled
SVG, no TypeScript — matching the other two projects.

* **`GameMap.jsx`** — the centrepiece: the MDS projection (§21) as a scatter plot. Chess and checkers adjacent, rust and python adjacent, the conversation off on its own. Point size by corpus confidence, colour by `category`, hover for the signature, click to pin. This is the "plane of existence" and it is the one screen that shows whether the whole idea works.
* **`TreeView.jsx`** — the radix tree: compressed segments as multi-axis labels on one node, terminal nodes ringed, internal terminals ringed *and* expandable (§19.5), counts as node size, live highlight of the retrieval path during an identification run.
* **`IdentifyPanel.jsx`** — declare what you know, pick an oracle, watch the candidate list narrow probe by probe, with the failure rate needle against its 0.5 target and the entropy curve falling.
* **`ProbeLog.jsx`** — every probe: the action, the verdict, the reason verbatim, its class, and the bits it bought. The raw experience of the system learning by failing.
* **`RulesPanel.jsx`** — the rule set split by provenance, induced against inherited, with Wilson bounds and the `unexplained` count called out.
* **`AxesPanel.jsx`** — the catalogue by measured IG, with dead axes (IG ≈ 0) flagged for removal and discovered axes (§18.3) marked.
* **`PackagePanel.jsx`** — the `GamePackage`, its confidence decomposed into the three factors of §22.1, and whether GTMNN would accept it.

---

## 28. Tests

* `test_verdict.py` — `ERROR` never reaches rule induction; `reason` survives normalisation unmodified.
* `test_action.py` — grammar derivations are reproducible from a seed; `neighbours` returns actions at edit distance 1; open spaces report `size() is None`.
* `test_oracle.py` — every reference oracle satisfies the protocol; `GrammarOracle` agrees with its own ground-truth language on 10 000 strings.
* `test_sandbox.py` — a timeout is killed and reported as `ERROR`; a memory bomb is capped; temp dirs are removed on every exit path including timeout; a generated program cannot reach the network.
* `test_reasons.py` — normalisation is total and deterministic; **classification is order-independent** (same multiset in any order ⇒ same partition); `reason_code` short-circuits clustering.
* `test_rule.py` — against `GrammarOracle` with known ground truth: **rule recall and precision above 0.9 within budget**; Wilson bounds bracket the true rate; a contradiction lowers confidence without deleting; a retired rule revives.
* `test_probe.py` — **the `eig` policy holds a failure rate of `0.5 ± 0.1`** on the synthetic oracles (§4); every policy beats `random` on probes-to-identify; `adversarial` finds planted rule-set errors that `eig` misses.
* `test_primitive.py` — `canonical()` is invariant to alpha-renaming and stable across processes; minimality (§8.3) splits `CHAIN_CAPTURE`/`FORCED_CAPTURE` on a corpus containing a variant with one and not the other, and does not split `CASTLING`; **`resplit()` bumps the epoch, invalidates cached signatures, and re-signing the corpus reproduces the previous similarity ordering**; inherited rules are never decomposed (§8.9); mechanic Jaccard on the §8.5 fixtures reproduces §8.6's table to 0.01.
* `test_modifier.py` — hashing a set preserves overlap: modifier cosine tracks Ochiai with mean absolute error below 0.07 at `F_mod=128` over 600 random pairs (§22.5); the modifier is deterministic across processes; two games with identical mechanic sets get identical modifiers.
* `test_axis.py` — IG ordering matches a hand-computed corpus; `discover()` fires exactly when two games share a signature with disjoint rule sets.
* `test_radix.py` — **a terminal node is never merged away** (§19.1), asserted under randomised insert/compress sequences; split/merge round-trips; `path(insert(σ)) == σ`; an internal node can be terminal and survives compression; `retrieve` on complete evidence returns exactly `lookup`; `retrieve` on empty evidence returns the whole corpus ranked by prior; `to_dict`/`from_dict` round-trips.
* `test_forest.py` — forest proximity correlates with mechanic Jaccard at Spearman > 0.8 across the corpus (§20.3); proximity is symmetric, in `[0,1]`, and 1.0 for a game against itself.
* `test_similarity.py` — on the §8.5 mechanic fixtures the ordering chess↔checkers (0.39) > chess↔go (0.23) > chess↔boss-conversation (0.07) > chess↔rust (0.03) holds, **and holds under every permutation of the mechanic vocabulary** — the order-freeness claim of §8.1, asserted rather than argued; MDS preserves the rank order of the top-5 neighbours for 90% of corpus games; MDS is deterministic given a seed.
* `test_package.py` — `confidence` below `τ_conf` is refused; `calibrate` gives zero expected payoff for a random play; an illegal action cannot enter a GTMNN repertoire; `to_dict`/`from_dict` round-trips.
* `test_explorer.py` — identification on `GrammarOracle` converges within `2·log2|G|` probes; `stop_event` is honoured; a run resumes from a checkpoint without re-probing; `max_error_rate` aborts a broken oracle.
* `test_cli.py` / `test_api.py` — subprocess and in-thread smoke tests of every command and endpoint with `--json`, asserting the documented keys.

Every test is seeded; none depends on wall-clock timing or network access. The
`GrammarOracle` tests are the ones that matter: they are the only place where
ground truth exists, so they are the only place correctness can be asserted
rather than plausibility.

---

## 29. Performance notes

Probing dominates everything. At `CompilerOracle` latency (~300 ms) a
10 000-probe run is ~50 minutes of wall clock and ~99.9% of it is `rustc`.

| Stage | Cost | Share (compiler oracle) |
|---|---|---|
| Oracle probe | 300 ms | **~99.9%** |
| `retrieve` | `O(beam · depth)` ≈ 50 µs | <0.1% |
| Rule induction (every 64 probes) | `O(probes · |predicates|)` ≈ 20 ms | <0.1% |
| Policy scoring (`budget=64` actions) | ≈ 2 ms | <0.1% |

Consequences, which are the opposite of GTMNN's:

* **Optimise probe count, never probe cost.** A policy that takes 50 ms to pick a better probe pays for itself if it saves one probe in six. `eig` and `bandit` are extravagant by design and it is still free.
* **Probes are cached and deduplicated.** `(state, action)` → `Verdict` in a persistent store keyed by oracle version. Re-probing something already answered is pure waste and the cache hit rate is reported.
* **Probe in parallel where the oracle is stateless.** Compilers are: `Explorer` may run `jobs` probes concurrently through `sandbox.run`, defaulting to `os.cpu_count()`. This is the only optimisation in the system with an order-of-magnitude payoff.
* Retrieval, induction and scoring stay pure-Python with no vectorisation, because making a 0.1% stage faster is not an optimisation.

**Targets**: identification on `GrammarOracle` within `2·log2|G|` probes;
`probes_to_identify` at most one third of `random`'s; retrieve above 10 000/sec
on a 10 000-game corpus.

---

## 30. Other ways to solve this

The design above is one commitment among several. These are the alternatives that
were weighed, what each would buy, and what it would cost — recorded so that
adopting one later is a decision rather than a drift, and so the weaknesses of the
chosen design are on the record next to the things that would fix them.

The chosen design has three genuine weaknesses, and every alternative below is
best read as an attack on one of them:

* **W1 — the feature language (§14).** Domain knowledge hides there. A rule outside it cannot be induced. This is the biggest one.
* **W2 — axis commitment.** ~~Axes must be chosen and discretised; §9.3 showed one ordering calling chess and go unrelated.~~ **Dissolved by §8.** Decomposition makes a game a *set* of mechanics, set overlap is symmetric and order-free, and chess ↔ go scores 0.23 without a forest. Recorded here because the alternatives below were weighed against it and two of them (§30.1, §30.4) were attractive largely for solving it — they are correspondingly less attractive now.
* **W3 — discreteness.** Continuous properties get bucketed, and information is lost at every bucket boundary.

### 30.1 Learned embedding (metric learning) — attacks W2, W3

Drop axes entirely. Learn a continuous vector `e(game)` trained so similar games
are close, by a triplet or contrastive loss over known-similar pairs. Identify by
nearest neighbour in the embedding; handle partial evidence by encoding what is
known and projecting.

*Buys:* this is the author's "embedding space / plane of existence" taken
literally rather than projected at the end. No discretisation, no ordering, no
axis discovery, and continuous properties are handled natively.
*Costs:* needs labelled similar/dissimilar pairs to train on, which is the very
thing the corpus is supposed to produce — a chicken-and-egg the tree does not
have. And critically, **a vector cannot be inherited from.** §16.4 is the reason
identification is cheap, and it needs a neighbour's *rules*, not its coordinates.
*Verdict:* complementary, not alternative. The right combination is the tree for
rule inheritance and an embedding for similarity, with §21's MDS replaced by a
learned projection once the corpus is large enough to train one.

### 30.2 Bayesian posterior over rule programs — attacks W1, W2, W3

Model a game as a program in a DSL; put a prior over programs favouring short
ones; posterior-update on every probe. Identification is the posterior; the rule
set is the MAP program; the optimal probe is the max-EIG one.

*Buys:* it is the *correct* formulation. Everything in this document is a
tractable approximation of it — `Evidence` approximates the posterior, the tree
approximates the prior's structure, §17.3 is its probe rule exactly. Partial
evidence, inheritance and confidence all fall out rather than being engineered.
*Costs:* intractable beyond toy DSLs. Enumeration dies immediately; MCMC over
program space mixes badly; the prior over programs is doing enormous unexamined
work.
*Verdict:* keep as the **specification of correctness**, not the implementation.
`test_rule.py` on `GrammarOracle` can enumerate the posterior exactly for small
grammars, so the approximation can be measured against the truth rather than
assumed close to it. That is worth building even though the method is not.

### 30.3 Normalised compression distance — attacks W1 hardest

No axes, no features, no rules. Represent a game by the raw transcript of its
probes and verdicts, and define similarity by compressibility:

```
NCD(x, y) = (C(xy) − min(C(x), C(y))) / max(C(x), C(y))       C = zlib, stdlib
```

*Buys:* **zero feature engineering** — it removes W1 entirely, which is the
design's worst problem. It is domain-universal, needs no catalogue, no
discretisation and no ordering, and it is about forty lines of standard library.
*Costs:* opaque (it will say two games are similar and never say why); needs
substantial transcript before it is meaningful; nothing can be inherited from it;
and it is sensitive to transcript formatting in ways that are hard to audit.
*Verdict:* **adopt as a second opinion, and this is the recommendation.** Compute
NCD alongside forest proximity. Where they agree, confidence is real. Where they
**disagree, the axis catalogue is wrong** — NCD sees structure the axes are
blind to (or vice versa), and that disagreement is the only automatic signal
available that W1 is biting on this domain. A cheap, independent check on the
part of the design that is hardest to verify is worth more than a marginally
better clusterer. See §32.2.

### 30.4 MinHash over refusal fingerprints — attacks W1, scales

Take §2 literally: a game *is* its set of refusals. Sketch that set with MinHash
and estimate Jaccard in sublinear time.

*Buys:* the most direct possible operationalisation of the central claim, and it
scales to corpora far beyond what tree retrieval handles.
*Costs:* raw `(state, action)` pairs are not comparable across domains — chess
moves and Rust programs share no alphabet, so the Jaccard is trivially zero and
useless. The fix is to hash `(feature-vector, reason-class)` pairs instead of raw
actions, which reintroduces the feature language and gives back W1.
*Verdict:* the cross-domain comparability problem is fatal to the pure version.
Worth revisiting only if the corpus outgrows tree retrieval, which is not near.

### 30.5 A decision tree over probes rather than axes — attacks W2, collapses two modules

Skip the axis catalogue. Learn a tree whose internal nodes are *probes* and whose
leaves are games: "play this move; refused → left, accepted → right". Twenty
questions, directly.

*Buys:* elegant — the probe policy and the identification structure become one
object, and it optimises probes-to-identify directly rather than through the
proxy of axis information gain. §18 and §17 collapse into one module.
*Costs:* the optimal such tree is NP-hard (greedy IG is the standard
approximation, which is what §18.2 already does). Worse, the probes are
*domain-specific*: a tree whose nodes are chess moves says nothing about Rust, so
the structure cannot span the corpus — and spanning the corpus is the entire
point. It also generalises poorly to genuinely new games, where no stored probe
is meaningful.
*Verdict:* right idea at the wrong level. It is what §17.4 already does *within*
a candidate set, where the action space is shared. As the global structure it
cannot work.

### 30.6 Absorbing Markov chain over the boundary — reuses existing work

Model the game as a Markov chain whose absorbing states are refusals — directly
the author's `AbsorbingMarkovChain/`. The fundamental matrix `N = (I − Q)^-1`
gives expected steps to absorption, and a game is characterised by the spectrum
of its transition matrix. Similarity = spectral distance.

*Buys:* captures *dynamics* rather than static properties — how quickly and by
what routes a game refuses you — which no axis in §9.1 measures. Continuity with
existing work in the repository, and the fundamental matrix is a genuinely
different and informative view of a refusal boundary.
*Costs:* state spaces differ across games, so the matrices are not directly
comparable and need an alignment step that is itself unsolved. Spectral distance
is expensive on large state spaces.
*Verdict:* the strongest candidate for **new axes** rather than a new method.
Spectral summaries — top eigenvalue, mixing time, mean steps to absorption — are
scalars, so they bucket cleanly into axes 22+ and slot into the existing
catalogue with no architectural change. Cheapest good idea on this list.

### 30.7 Analysis by synthesis — the inversion

Rather than classifying the game, *generate* candidate rule sets and keep whichever
best predicts the observed verdicts. Score by likelihood, mutate, iterate.

*Buys:* it is the author's "invert the problem" instinct, and it composes with
machinery already built: it is `RadixCyclicNN`'s GAN-style `Evolver` and 2NRL
(train on garbage, invert, fine-tune) pointed at rule sets instead of text. It
finds rules outside the feature language, because a generator is not restricted
to a predicate vocabulary — which makes it the **only item on this list that
truly removes W1** rather than routing around it.
*Costs:* expensive, and the search space is unbounded without a strong prior —
which is §30.2's problem arriving by a different road.
*Verdict:* the most interesting long-term direction, and explicitly out of scope
for a first implementation. Revisit once `RuleSet.unexplained` has real numbers
on a real domain: if the feature language is covering 95% of refusals, this is
not needed; if it is covering 40%, this is the answer.

### 30.8 What is actually recommended

Build the tree as specified, and add §30.3's NCD as a cheap independent check
from the first commit. Take §30.6's spectral summaries as additional axes when
the catalogue stops discriminating. Hold §30.1 until the corpus is large enough
to train on, and §30.7 until `unexplained` says it is necessary.

The tree earns its place for one reason that none of the alternatives can match:
**it is the only structure here that supports rule inheritance with provenance.**
Every alternative gives a similarity number; only the tree hands over a
neighbour's actual rules along with an honest account of which ones were never
verified. §3.1's three-orders-of-magnitude saving depends entirely on that, and
without it the whole design collapses back into induction from nothing.

---

## 31. Cross-cutting invariants

1. **A terminal node is never merged away** (§19.1). The one bug this structure is prone to.
2. **`ERROR` never becomes a rule** (§10). Oracle failures do not enter the boundary.
3. **Provenance is never laundered.** Every rule carries `source`, and `verified_fraction` reflects it honestly in `GamePackage.confidence`.
4. **Nothing is played below `τ_conf`.** `GTMNet.load_package` refuses; understanding precedes play, mechanically.
5. **Reason classification is order-independent** (§15).
6. **Retrieval on complete evidence equals exact lookup**; on empty evidence it returns the corpus ranked by prior. The two limits of the same operation.
7. **Determinism.** Same seed, same corpus, same oracle transcript ⇒ same signatures, same clusters, same package — across processes and a save/load cycle.
8. **Nothing generated is executed outside the sandbox** (§13).

---

## 32. Open questions

1. **Is the feature language (§14) covering the domain?** `RuleSet.unexplained` answers it empirically and nothing else can. Instrumented from the first commit for exactly this reason. If coverage on Rust sits below ~0.6, §30.7 stops being a future direction and becomes the work.
2. **Do the two similarity measures agree?** §30.3's NCD against forest proximity, per game pair, plotted. Systematic disagreement localises where the axis catalogue is blind, and is the only automatic check on W1 available. This is the first experiment to run.
3. **How many axes are real?** 22 are seeded and `discover()` adds more, but an axis whose measured IG is ≈ 0 across the corpus is noise with a name. `gren axes` flags them; whether the catalogue converges to a stable dozen or keeps growing is unknown and interesting either way.
4. **Does `τ_novel` hold?** The claim that a populated corpus makes cold start rare is untested. If most new domains land below `τ_novel`, identification is not buying what §3.1 says it buys and the corpus needs seeding differently.
5. **Does `B:λ` calibration (§22.3) actually help GTMNN?** It answers GTMNN's own open question 1 from measurement rather than a constant. The test is whether GTMNN's `gini` and `diversity` readouts behave better under a GREN-calibrated ratio than under the default — the first real test of whether the two networks compose or merely concatenate.
6. **Does the minimality criterion (§8.3) converge?** A part is minimal when no sub-part is seen independently, so the vocabulary re-splits as the corpus grows. Whether it settles or churns indefinitely is unknown, and churn is expensive: every re-split invalidates every cached signature. If it does not settle, minimality needs a hysteresis term and the criterion stops being as clean as §8.3 claims.
7. **Should the forest be `T = 1`?** §20's original justification was dissolved by decomposition and what remains is a hedge against frequency-ordering fragmentation. If `test_forest.py`'s Spearman correlation against mechanic Jaccard stays above 0.95 at `T = 1`, the module is dead weight and should be deleted rather than kept for the reason it was originally built.
8. **What is the right `universal_fraction` (§22.7)?** The share of micros that are game-blind and therefore transfer everywhere. Too low and nothing transfers and each game is learned from scratch; too high and nothing specialises and every game is played averagely. Nothing in this design predicts it, evolution should find it, and whether it finds a *stable* value or oscillates with the corpus is the single most informative experiment about whether GREN and GTMNN actually compose.
9. **Does validity really transfer?** §22.6 asserts that legality is structure and transfers across games with similar modifiers, while grade does not. The clean test is available and cheap: train `p_valid` on chess only, then measure its accuracy on checkers with the checkers modifier and no further training. If it is near chance, the modifier is not carrying what §22.5 claims and the whole composition is decorative.
10. **Graded oracles.** `oracle_fidelity=graded` is specified and unproven. A refusal that arrives late, partially, and from a party with its own interests is the normal case outside programming, and thresholding it at 0.5 is a placeholder standing where the hard problem is.
