# GameLattice — Design Specification

A **RadixCyclicNN whose step is a game**, feeding a **LatticeFSM whose
every lattice state is one of the encoder's outputs**, read out by
**decoders that are separate from the lattice** — and a body around it: the
senses are its encoders, the effectors its decoders, and every effector's
output comes back in through a sense. This document is the contract the
package is implemented against. Read it fully before writing code; §21 lists
the decisions only the author can make, and nothing in §§4–12 is to be
"improved" past what §17 can measure.

Directory: `GameLattice/` (this directory). Python package: `gamelattice`.
Python 3.11+, standard library only. It imports its siblings from the
repository root — `RadixCyclicNN/radixnet`, `LatticeFSM/latticefsm` and
`Experiments/UniversalGameEncoding/uge` are required; `PhoneticTokenizer/phonetok`
and `ModelKit/modelkit` are optional extras that the ear, the mouth, the
hands and the monitors need (§13). `GTMNN/` is a specification with no code:
every auction, congestion, regret, Shapley and meta-player mechanism named
here is **built** from `GTMNN/DESIGN.md`, with `CyclicCortex/cortex/routing.py`
(the one running auction and settlement in the repository) as the precedent.

---

## 1. Concepts (in the author's own terms)

| Requirement | How it is realised |
|---|---|
| Combine the idea of a RadixCyclicNN with a LatticeFSM | Three parts, one organism (§2). The RadixCyclicNN's gram graph, path compression, cycles and sentinels are kept and become the **GameRadix encoder** (§4). The LatticeFSM's dense `S × A × S` matrix of adaptive edges is kept and becomes the **lattice** the encoder's outputs are states of (§5). Nothing in either sibling changes for this to work; the new code wraps them. |
| The RadixCyclicNN needs a new encoder | The encoder's *step* changes, not its graph: at every position the nodes of the walk window play a **stage game** whose outcome names a lattice state (§4.3). Encoding is no longer "the next gram" but "which state of the lattice this context is". |
| The LatticeFSM needs a new and separate decoder | A `Decoder` protocol the lattice knows nothing about (§6): `read(path) -> Output`, `observe(result) -> symbols`, `credit(verdict)`. One class per **output type**; four mechanisms (edge repertoires, propose-and-replay, a readout, a gate). A decoder that names no sense to read its output back is refused at construction. |
| Think about all potential output types and ways to output them | §6.2 is the table: verdict, label, state stream, symbol stream (text, sounds, bytes), action, distribution, value, structured, trace, another node, abstain — and the body's additions: gaze, expression, wait. §6.3 is how each is produced. |
| Turn the RadixCyclicNN into a model that focuses on game theory | §4.3–4.6 and §10. The congestion game (right *and rare* is worth everything), regret matching, Shapley credit, a seat auction with a reserve, coalitions, a meta-player on a cycle. The theorems that license each piece are GTMNN's (`GTMNN/DESIGN.md` §2.1) and are not restated here. |
| Each output of the RadixCyclicNN is a node in the LatticeFSM | The encoder emits a lattice **state** `s_t`, the *source* node of the next transition (§2.2). The lattice's symbol is the act or the receptor symbol (§5.1); the lattice chooses the target by its own adaptive edge. |
| The human body: it can see, hear, touch, smell, taste, and more; interaction is physical touch, talking | §9. Every sense is an organ that computes its own features and hands over **symbols** (`LatticeFSM/DESIGN.md` §14: "symbols, not features"); every effector is a walk on its own lattice whose symbol is the act. The verified mapping is one row per sense and per effector; what the repository already has as that organ is named in each. |
| Interaction (decoding) comes back as sensing | §8, the loop: an **efference copy** of every emission, three verdicts for what returns (reafferent, error, exafferent), credit only from the world. Babbling before any teacher. |
| Focus heavily on informational encoding and decoding | §7. Every encoder is a source code with a measured **rate** (symbols per second, bits per symbol), every lattice a channel with a **state capacity** (`log2 S` bits per step), every decoder a channel decoder whose **parity check** is the referee. Credit is information: the log-likelihood gain Shapley divides (`Σφ = v(N) − v(∅)`) is bits. The one-token memory cliff, the code-space partition and the locality rule are the three measured facts the section is built on. |
| Cycles are a feature | A best-response cycle in the inference game teaches `THINK` at that node; a looping walk teaches `BACK` (§4.6). The exits are the three of `Research/CyclesAreAFeature.md` §5: the clock, the monitor, the inversion — in that order of cost (§10.5). |
| You learn the rules by failing | The referee's refusal with a reason is the first and cheapest verdict and lands per step on the edge that proposed (§7.4, §6.4). Inversion (2NRL) is **not** used on any effector: it lost to its matched control on every game it was tried on (§17, E4). |

---

## 2. The shape

### 2.1 Three parts

```
                 ┌──────────────── sense organs (fixed transducers, §9.2) ───────────────┐
 world ─────────►│ retina / cochlea / skin / nose / tongue / the body's own gauges        │──► symbol streams
                 └────────────────────────────────────────────────────────────────────────┘
                                        │ receptor symbols, one code region per organ
                                        ▼
                 ┌──────────── GameRadix encoder (§4): a RadixCyclicNN whose step is a game ─┐
                 │ gram graph, compression, cycles, BACK / THINK                              │
                 │ the walk window's nodes play the congestion game  ──►  names state s_t    │
                 └────────────────────────────────────────────────────────────────────────────┘
                                        │ s_t  (a lattice state: the encoder's output IS a node)
                                        ▼
                 ┌──────────── the lattice (§5): LatticeFSM, one class, three kinds ──────────┐
                 │ sense lattice   row (s_t, receptor symbol) -> t_t          life per organ  │
                 │ act lattice     row (s_t, act a_t)         -> t_t  = the forward model      │
                 │ association     reads the sense lattices' states on a shared clock          │
                 └────────────────────────────────────────────────────────────────────────────┘
                                        │ the path: edges with their records
                                        ▼
                 ┌──────────── decoders (§6): separate, typed, one per effector ──────────────┐
                 │ repertoires on edges · propose-and-replay through a referee · readout · gate│
                 └────────────────────────────────────────────────────────────────────────────┘
                                        │ an act, copied (§8) into the alphabet of the sense that reads it back
                                        ▼
                                      world ──► (back to the top)
```

### 2.2 The split that makes it work

The encoder says **where we are**; the lattice says **where this takes us**;
the decoder says **what that means**. Concretely, per step:

1. The encoder names `s_t ∈ {0..S−1}` from the walk (§4). This is the
   *source* node of the next lattice transition.
2. The lattice reads a symbol `a_t` — on a sense lattice a receptor symbol,
   on an act lattice the act the decoder just emitted (§5.1) — and chooses
   `t_t` from the row `(s_t, a_t)` by its own adaptive edge
   (`LatticeFSM/DESIGN.md` §3–5). On an act lattice that row **is** the
   forward model `τ(φ(s), a) = φ(s′)`.
3. The decoder reads the path (the edges, with their fifteen fields each)
   and produces the output of its type (§6).
4. The output is copied into the alphabet of the sense that will read it
   back (§8). When the world answers, each returning symbol is classified
   *reafferent*, *error* or *exafferent*, and only the last may carry the
   world's credit.

The agreement `t_t == s_{t+1}` between the lattice's prediction and the
encoder's next named state is **a meta-feature, not credit** (§8.2). An
earlier draft of this design credited it; the body reading retired that
(§20, change 2).

### 2.3 What is kept from each sibling, unchanged

| from | kept as it is | used for |
|---|---|---|
| `radixnet/graph.py` | `RadixCyclicGraph`: labels, `observe_sequence`, split and merge, `trigram_index`, `BACK`, `THINK`, `observe_back`, `observe_think`, cyclic counters | the encoder's structure and the sentinels |
| `radixnet/search.py` | `onward` (the hand-over), the `(node, chars_emitted)` search state | the clock exit; the encoder's own continuation when no lattice is attached |
| `radixnet/encoding.py` | `Encoding` (unit × n × stride), the phonetic and acoustic units, `Vocabulary` | the text sense's transducer |
| `radixnet/window.py`, `attention.py` | the dynamic window, the attention band | the gram's grain; the *blame* fovea (not the gaze, §9.4) |
| `latticefsm/machine.py` | `Machine`: `step`, `run(quiet=)`, `credit`, **`credit_steps`**, `teach`, `tick`, `stimulate`, `focus`, `skip`, `rearrange`, `transition_table` | every lattice |
| `latticefsm/edge.py` | `Edge` (fifteen fields), `Weighting`, the five features, lazy fading | every edge |
| `latticefsm/network.py` | `Layer(states, emit, **machine_options)` — life, calm, mode, explore are already per layer; `Parallel` with its letter shift; `Encoder` classes; `PhoneticEncoder` / `PhoneticDecoder` | the lattice stack; the English front and back |
| `latticefsm/media.py` | `cells`, `order('center-out')`, `image_symbols`, `diff_symbols`, `video_symbols`, `audio_units` | the eye and the ear |
| `latticefsm/guard.py`, `autotemp.py` | `Guard.judge` (reject / suspect / pass), the temperature ladder | taste and the output reflex; prosody of a retry |
| `uge/game.py`, `codec.py`, `metrics.py`, `corpus.py` | `Game`, `decode` (the referee: header / syntax / illegal / phi / truncated), `propose`, `generate_plays(RandomPolicy)` | the action decoder, babbling |
| `cortex/routing.py` | `bid = reputation × coverage`, `allocate` at the highest losing bid, `RESERVE = 0.05`, `settle(base / n, −penalty 0.5)` | the running precedent for the auction and the congestion split |

---

## 3. Status, and the numbers that constrain it

Nothing in this directory is measured yet. Every claim below is **held**
(`Research/Insights.md`'s word) until §17's experiment for it has run. The
repository's existing numbers are the floor and the ceiling of what may be
claimed:

| measured | number | where | what it binds here |
|---|---|---|---|
| a 13-state lattice alone on the four languages | own 0.894, oracle per step 0.924, explore 0.944 | `LatticeFSM/README.md` "Credit by an oracle" | E1: the stage game must earn its cost against these |
| three parallel 3-state machines, own credit | 1.00 on every language; shared credit 0.62 on contains-aa | `LatticeFSM/README.md` networks table | own credit everywhere (§7.5); parallel beats stacked |
| a hidden machine between encoder and decoder | plural 0.03 exact (0.11 sounds right), past 0.00 (0.04) at 20 epochs; 0.4 at temperature 0.3 over 40 | same, English table | never a lattice *between* an encoder and its decoder (§11) |
| media | image>3 0.25 held out, audio>3 0.67, diff>3 0.58, video>13 0.50 (chance) | `LatticeFSM/README.md` media table | E8, E9: the two-sense stream is the layout control |
| the one-token memory cliff | Nim legality 0.943 / 0.514 / 0.939 at φ = 15 / 20 / 30 bits | `Experiments/UniversalGameEncoding/README.md` | §7.2; E2 |
| chess on a tape | 1724.84 refusals a move (guessing 1894.1), 98.63 with the last move's square in the token | same | §7.3 locality; E3 |
| 2NRL on games | Nim 0.943 → 0.438; chess 2NRL vs positive 0.229 | same; `TwoNRL_Chess/README.md` | no inversion on effectors |
| two heads for an act | refusals 97.9 → 1.1 a move with a legality head; 660 → 41.8 on one wire | `TwoNRL_Chess/README.md` | §7.4 |
| inversion of a continuous controller | 9.2 → 499.4 / 500 steps, one sign flip, no training | `Experiments/TwoNRL_CartPole/README.md` | the one place inversion is kept (§6.2, value) |
| transfer from a shared generalised vocabulary | 0.646 zero-shot against 0.514 | `CyclicCortex/DESIGN.md` §4 | §11: the association lattice's alphabet question |
| the acoustic round trip | the same unit 86 % streamed, 92 % polished | `PhoneticTokenizer/README.md` | §8.1's tolerance; E7 |
| the lattice's own code | exact, 7.5–12× smaller than the dense matrix; low-rank codes rejected | `LatticeFSM/DESIGN.md` §11.3 | §7.6 |

---

## 4. The encoder: GameRadix

### 4.1 What is kept

`GameGraph(RadixCyclicGraph)` in `encoder.py`. The gram graph is read,
split, merged and compressed exactly as `radixnet/graph.py` does; the
`Encoding` dial (`char:3:1` by default, words, phones, syllables, acoustic
units) is the graph's; `BACK` and `THINK` are the graph's; `locate` and
`onward` are the graph's. A `GameGraph` with no lattice attached predicts
text exactly as a `CountRewardGraph` would, which is the regression test.

### 4.2 What a node holds, beside its gram

Flat parallel lists indexed by node id, like every other per-node field:

| field | what |
|---|---|
| `strategy` | a mixed strategy over the attached lattice's `S` states plus `ABSTAIN` (`S + 1` floats, rows sum to 1) |
| `regret` | cumulative regret over the same slots, warm-started across stage games and decayed by `gamma_regret = 0.95` |
| `wealth` | payoff earned less clearing prices paid |
| `hits`, `plays` | discounted (`gamma_rep = 0.999`) counts of right and non-abstaining plays; `reputation = (hits + 1) / (plays + 2)` |

The node's sine activation (`a, b, h, k`) is kept and is its *voice*: the
aggregate weights a seat's strategy by `reputation × |f(z)|`, so a node the
walk has taught to be quiet speaks softly. Nothing here is a weight shared
between nodes; no gradient crosses a node (`GTMNN/DESIGN.md` §26).

### 4.3 The stage game, one position

1. **Seats.** The window is defined per modality by its encoder (§9.3):
   for text, the last `W` nodes of the walk plus the current node's
   children; for a fixation, the nodes the fixation's cells walked through;
   for a sniff, the nodes between two `|`. Seats are **nodes**. A symbol,
   a cell or a candidate is what a node *reads*, never a player.
2. **Bids.** A seat bids `reputation × (q_max − q_abstain)` from its own
   strategy; bids below the modality's reserve (`0.01`, the Gate's, §10.1)
   are not seated; `M` seats sell at the highest losing bid and each pays it
   (`cortex/routing.py allocate` is the precedent). With the window small
   the auction is usually uncontested and costs one sort.
3. **The game.** Each seat names a state. Payoff with the label `y` known
   (training): `B / n_a` if `a = y` (split among the seats that named it),
   `−λ` if wrong and not abstaining, `0` for abstaining; `B = 1`, `λ = 0.5`
   (rewards-only measured worse in every LatticeFSM arm, so `λ` stays).
   Without a label (play): each seat scores by its own belief, the
   player-specific game of `GTMNN/DESIGN.md` §8.2.
4. **Solver.** Regret matching plus (`regret_plus`), `iters = 64`, the
   time-average profile returned; a cycle detector pushed once per sweep.
5. **Aggregate.** `P(state) ∝ eps / S + Σ_seats reputation × voice × profile`;
   `s_t` is its argmax at play, the label at training. Reputation inside
   the aggregate is deliberate double counting (`GTMNN/DESIGN.md` §10.1).
6. **Credit.** Shapley values over the seated nodes against
   `v(C) = log P_C(y)`, estimated by 32 antithetic sampled permutations
   with efficiency exact per permutation (`Σφ = v(N) − v(∅)`), at any seat
   count. Each seat then takes one REINFORCE step on its own strategy with
   `φ` as the advantage. There is no exact mode: `2^16` coalition
   evaluations is not a per-step cost anywhere in this organism.

### 4.4 Where the label comes from

At training time `y` is **the lattice state whose edge was credited for
this step** (§7.5): the state the oracle, the referee or the teacher made
right. With `y` known the game is an exact potential game (Rosenthal) and
best response cannot cycle; training never escalates.

At play time there is no `y`, but there is a refusal. When the referee
refuses the act that followed and its reason is about the *belief*
(`illegal`, `phi`, a wrong output) rather than its *spelling* (`syntax`),
every seat that named `s_t` is charged `−λ`. A syntax refusal charges the
decoder only — `teach_pair` measured that punishing the blocks before the
decoder for the decoder's errors kept reshuffling the walks the decoder was
learning to read (`LatticeFSM/DESIGN.md` §13.4).

### 4.5 The state must have locality

`Experiments/UniversalGameEncoding/ENCODING.md` §6 measured it: a hashed
state recognises what it has seen and says nothing about a near miss
(chess coverage 0.00 off the corpus), while a meaningful digest of the last
move (65 buckets) generalises (refusals 1724.84 → 98.63). The encoder's
input therefore carries a **local digest** beside the grams where the
modality has one — `Game.action_summary` for a move, the fixation's gaze
vector for the eye, the syllable position for a sound — and §17 reports
*coverage* (the share of held-out positions whose state was ever named in
training) beside every number. A lattice whose states are hashes is a
transposition table, and the measurement will say so.

### 4.6 Cycles, THINK and BACK

The inference game may cycle (Milchtaich; `Research/CyclesAreAFeature.md`).
A detected cycle at position `t` calls `graph.observe_think(p)` on the
current node — the graph learns, as an ordinary edge competing with the
real children, that *something here made me stop and think* — and hands
the step to the meta-player (§10.4). A walk that loops in its output
(the monitor of `modelkit/dialogue.py`: a stutter, a repeat) calls
`observe_back`. Neither is a decoder symbol: `BACK` is a reflex **before**
the decoder's game, `THINK` the slow loop **after** the aggregate fails.

### 4.7 The one-token memory, and the window as memory

RadixNet conditions on the last gram alone (`_locate`), which is the
15-bit cliff of §3. Two things here can hold more than one token, and both
are claims to measure (E2), not properties the design already has:

* a **compressed node** is a whole history chunk, so a state named from it
  is named from more than one token;
* the **walk window's seats** carry the identities of the last `W` nodes
  into the game — but only if the seats' strategies are read *per node*,
  which §4.2 makes true, and only if `W` matters, which E2 measures at
  `W ∈ {1, 4, 8, 16}` on Nim's 20-bit case (0.514 should recover toward 0.94).

---

## 5. The lattice

### 5.1 One class, three kinds

| kind | its symbol `a_t` | its states | its row `(s, a) -> t` is | life | credited by |
|---|---|---|---|---|---|
| **sense lattice**, one per organ | that organ's receptor symbol (§9.2) | what the encoder names: 13 by default, 43 for the auditory lattice if the author so decides (§21.7) | the sensory dynamics: *after this receptor symbol in this state, the next state* | the organ's (§9.2) | its own verdict per step (§7.5) |
| **act lattice**, one per effector | the act the decoder just emitted (a sound, a sub-decision of a move, a tool call, a gaze vector) | the encoder's named state of the modality that reads the effector back | the **forward model** `τ(φ(s), a) = φ(s′)` | the longest: rules are permanent | **only** by the periphery: the world's replay (§8.2) — never by the task's payoff |
| **association lattice**, one | the sense lattices' written states, interleaved on a shared clock, letter-shifted per branch (`Parallel._wire`) | amodal: what two senses are about | binding | a few ticks: the binding window | its own verdict |

"The lattice knows nothing about outputs" still holds, corrected: an act
lattice reads the act **as its input symbol**, because the loop (§8) feeds
every emission back in, and nothing in `Machine` changes for that.

### 5.2 The symbol axis is a code, never a map

A dense `S × A × S` matrix is meant for machines of tens of states
(`LatticeFSM/DESIGN.md` §10). Three rules follow:

* **Actions are factored.** Chess's 20 480 moves as symbols is 3.46 million
  edges at `S = 13` and is refused. A move is a sequence of sub-decisions of
  at most 64 symbols each (`ENCODING.md` §11: `φ(s)`, from-square,
  to-square, promotion), each sub-decision one lattice row, the referee
  replaying the assembled move.
* **Topography lives beside the lattice.** A retina's or a skin's adjacency
  is a *reading order* and a neighbour table in the encoder; the symbol
  axis indexes levels, not patches, and `rearrange` sorts it by load into
  any permutation. A `topographic=True` layer refuses `rearrange` and
  `swap_symbols` on its symbol axis; the state axis may still rearrange.
* **Code regions are disjoint.** Two symbols that can be told apart only by
  where they sit in the stream cannot be told apart by this architecture
  (`ENCODING.md` §5). The fovea's levels, the periphery's, the change
  symbols, the gaze vectors, the boundary `|` are separate letters.

### 5.3 Per lattice: life, calm, stimulation, a clock shared

`life`, `calm`, `mode`, `explore` and the widening constants are already
constructor arguments of `Machine` and `Layer(**machine_options)`; this
design only gives them per-organ defaults (§9.2). Stimulation is one level
per machine (`LatticeFSM/README.md` Limits), which is exactly per-sense
arousal once each sense has its lattice. One `SharedClock` (`clock.py`)
ticks every lattice on every step of any lattice (`Machine.tick(k)`), so an
idle sense's traces fade; wall-clock idleness enters as explicit ticks
(§12.3).

### 5.4 Abstain on the lattice side

A row may abstain: a probability floor `abstain_floor` (0 by default) under
which the step emits nothing and the decoder's game is not played. Without a
**metabolic cost per traversal** a reserve converges to permanent silence
(`GTMNN/DESIGN.md` §17.1's ordering: being useless is fatal on a longer
timescale than being wrong); whether that cost exists is §21.4, because it
changes every measured LatticeFSM table.

### 5.5 Reasons on edges

Pain lands on edges (§7.4) and a reflex must be able to say why it withdrew:
a side table `reasons[(source, symbol, target)] -> {reason: blame}` beside
the lattice, mirroring `radixnet/negative.py`'s `edge_reasons`, with
`forget(reason, factor)` as `Guard.forget` has it. The edge's own
`punished` never fades (`LatticeFSM/DESIGN.md` §4); forgetting by reason is
the rule's second half, because the body habituates to a predictable
noxious stimulus over days (the touch verification, §9.2 pain).

---

## 6. The decoders

### 6.1 The protocol

```python
class Decoder(Protocol):
    output_type: str                 # one of §6.2
    reads_back: str                  # the sense that reads this output back; refused if empty
    def read(self, path: Sequence[Edge], lattice: Machine) -> Output
    def observe(self, result) -> list[str]   # the world's answer, as the read-back sense's symbols
    def credit(self, verdict: Verdict) -> int
```

`Verdict` is one record for every source of credit in the organism:

```python
@dataclass(frozen=True)
class Verdict:
    amount: float           # signed; the referee's severity, the oracle's ±1, the taste line's valence × intensity
    reason: str             # 'illegal', 'phi', 'syntax', 'timeout', 'unspecified', ...  (blame.py's taxonomy)
    sharp: bool             # True: per step, lands in full on the step's edge (credit_steps); False: dull, discounted back along the run
    source: str             # 'referee', 'sandbox', 'tool', 'oracle', 'taste', 'teacher', 'person', 'intero'
    origin: str | None      # what was consumed (a smell signature, a path): the identity-addressed path of §7.4
    step: int | None        # which step of the sensing act, for a sharp verdict
```

### 6.2 Output types

| type | one output is | the body's organ | mechanism (§6.3) | read back by |
|---|---|---|---|---|
| verdict | accept / reject | a reflex, the swallow-or-spit gate | the final state's `accepting`, or the taste gate | the next bite |
| label | one of `k` | recognition | a state-to-label map over accepting sets | the task |
| state stream | the states visited | corollary discharge to the layer above | `Layer.emit='states'`, a Moore output | the association lattice |
| symbol stream | text, sounds, base64 bytes | speech, writing | **repertoires on edges** on the effector's own act lattice; the guard on the output path | the text sense, the ear |
| action | a game move, a tool call, a saccade | the hand, the eye | **propose-and-replay** through a referee, released above a reserve | the new state, the `<result>`, the next fixation |
| distribution | a mixed strategy, a belief | readiness | the aggregate, or a row read quietly | nothing: a belief is not a move |
| value | a scalar: budget, confidence, stimulation, a continuous control | the autonomic dial; posture | a readout of edge records (`net`), or a reflex table; for a continuous controller the one place inversion is kept (CartPole 9.2 → 499.4) | the gauges |
| structured | JSON, a tool call | a skilled sequence | repertoires with the lattice as the grammar; accepting = well formed | the sandbox or schema |
| trace | the path, its edge records, blamed spans, the Rethink | introspective report | a readout | nobody; it is for the guard and the frontend |
| another node | a state handed to a second lattice or back to the encoder | the start state of an effector's walk | `Layer.emit='final'` | the layer after |
| gaze | a saccade vector; a sniff order; the next probe and its budget | orienting | **action** whose effect is on the organism's next input | vision, smell, touch |
| expression | the affect frame (stimulation, wealth, abstain share, the guard's verdict, the climbed rung) plus prosody dials | the face, the voice's tone | a **Moore readout**, credit a no-op by design: honest because uncreditable | the ear, interoception |
| abstain | nothing, with why | freezing, a pause | the resting state: a reserve price, `u(ABSTAIN) = 0`; `BACK` a reflex before the game, `THINK` the slow loop after; `wait` = `tick(k)` | the unpaid price, the clock |

### 6.3 The four mechanisms

1. **Repertoires on edges** (symbol streams, structured). Each edge of an
   act lattice carries `K = 4` output symbols plus `ABSTAIN`
   (`GTMNN/DESIGN.md` §7.5's repertoire, re-pointed at edges). The seats are
   the last few edges of the effector's walk; they play the congestion game
   over the output alphabet; Shapley (sampled) credits each edge; `END` and
   the pauses are members of every voice repertoire. The effector's walk is
   **its own** — an autoregressive walk on its act lattice whose symbol is
   the symbol just emitted — because output length is not input length
   (the `PhoneticDecoder` emits one sound per symbol read and caps at
   input + 2; 0.09 / 0.00 held out). The encoder hands the walk a start
   state (*another node*) and nothing per symbol.
2. **Propose-and-replay** (actions). The repertoires of the current
   sub-decision's row propose candidates; an `ActionSelector` seats at
   most 16 by `Edge.net × reputation`, plays the congestion game, and
   **releases** one channel if `top1 − top2 ≥ 0.05` and the bid clears the
   reserve, else `ABSTAIN` (`uge/metrics.py propose` then walks the ranking
   until the referee accepts, or falls back). The referee replays the
   assembled act and returns a `Verdict`. A distribution is a belief; a
   release is a move.
3. **A readout** (verdict by accepting state, label, state stream, value,
   trace, expression). Nothing to learn in the decoder; the lattice is
   read. The expression readout is deliberately uncredited.
4. **A gate** (taste). Five fixed lines with fixed signs — empty,
   unreadable, a loop, a known failure, clean — and an intensity each,
   executed in a fixed order before anything is learned, said or acted on;
   its thresholds are body state (`Guard.judge`'s `threshold` and
   `min_coverage`). It is not a stage game and no repertoire applies.

### 6.4 The referee

```python
class Referee(Protocol):
    def refuse(self, state, act) -> Verdict | None    # None: accepted
    def apply(self, state, act) -> state
```

Instances: `GameReferee` over `uge.Game` (break kinds `header`, `syntax`,
`illegal`, `phi`, `truncated` from `uge/codec.py`); `ToolBoxReferee` over
`modelkit/tools.py ToolBox.call` (`ok=False`, `ToolCall.error`, a `WebError`
text); `SandboxReferee` over `modelkit/codegen.py Sandbox.run`
(`timed_out`, a non-zero exit, `killed`); `EyeReferee` (a saccade off the
picture, pursuit on a still fovea); `PhonotacticReferee` over
`phonotactics.is_legal_onset / is_legal_coda` on the running cluster
(`well_formed` judges whole sequences and refuses a legal prefix). A refusal
is always a `Verdict` with a reason, never an exception. Decoding an act is
replay, not string manipulation: the act exists only when `apply` accepted
it (`ENCODING.md` §7).

---

## 7. Information: encoding and decoding as codes

This section is the quantitative spine. It treats the three parts as what
they are in Shannon's terms, and names the measurement that goes with each.

### 7.1 Three codes, one channel

| part | in coding terms | its rate | its capacity | measured by |
|---|---|---|---|---|
| a sense organ (§9.2) | a **source code**: the world compressed into symbols the lattice can read | symbols per second of world; bits per symbol = the entropy of its stream | fixed by the transducer: `log2 |alphabet|` per symbol at most | the stream's entropy and its conditional entropy given the last symbol (`ngram.evaluate`: bits) |
| the encoder (§4) | a **quantiser** of context into `S` states | one state per step | `log2 S` bits per step: 3.7 at `S = 13`, 5.4 at 43 | `I(s_t ; next symbol)`: how much naming the state tells about what comes next |
| the lattice (§5) | the **channel's state machine**: a stochastic transducer `(s, a) -> t` with memory one | one transition per tick | `log2 S` bits of memory; the row's entropy is what it does not know | the row's KL from its greedy choice (`compress.fidelity`'s measure), the agreement rate, coverage |
| a decoder (§6) | **channel decoding**: a received path mapped to a codeword of the output alphabet | one symbol per tick per effector | the output bottleneck: serial, one channel, `log2 |output alphabet|` per tick | refusals per act (the parity failures), legal@1, perplexity of the emitted stream |
| the referee (§6.4) | the **parity check** | — | — | the break kind: `syntax` is a code error, `illegal` a semantic one, `phi` the checksum (the tape's own state token disagreeing with the replay) |

Credit is information in the same units. Shapley's efficiency axiom gives
`Σφ = log P_N(y) − log(1 / |V|)`: the total credit handed to a stage game's
seats is exactly the bits the game gained over a uniform guess
(`GTMNN/DESIGN.md` §12.1). The N-gram's `bits` and the chain's
`Σ ln P` (`LatticeFSM/DESIGN.md` §15.4, §17) are the same quantity measured
on text. So "the stage game earns its cost" (E1) has a unit: bits per
symbol, against the plain lattice's.

### 7.2 Memory is one token wide, and a token is 15 bits

`ENCODING.md` §6 is the sharpest measured fact about the architecture: the
RadixNet conditions on the last trigram alone, a token of three base64url
characters carries 18 bits, and reserving the last character to its own
alphabet (so a trigram's phase is readable on its face) leaves **15 bits**.
Nim is 0.943 legal at 15 bits of state, 0.514 at 20 (the token holds five
bits of a two-token state and reads only the last), 0.939 at 30.

For this design that number is the budget per step of everything the
encoder is given, and the lattice adds `log2 S` bits of its own memory on
top. The question §4.7 poses is whether the walk window and the compressed
nodes raise the effective memory above one token; E2 measures it in bits:
the mutual information between the named state and the true state of the
game, at `W ∈ {1, 4, 8, 16}`.

### 7.3 Uniform, local, and the code-space partition

Three more facts from the same experiment are rules here:

* **Disjoint regions.** The outcome token and column 1 were both code 1
  once, and every tape's ending was a legal move. Header, outcome, action
  and φ live in disjoint ranges; here, every organ's symbols and every
  effector's acts are disjoint letters (§5.2).
* **Uniform beats injective.** Tic-tac-toe's exact board code lost to a
  15-bit hash (0.65 against 0.99) because a hash uses the token space
  evenly and a board code clusters in it. A source code that spreads its
  mass is worth something on its own; the entropy of each organ's stream
  is reported, and a stream that uses a tenth of its alphabet is a
  transducer to redesign.
* **Local beats hashed.** A hash has no near miss (§4.5). The state fed to
  a row must have locality, or coverage off the corpus is 0.00 and the
  lattice is a lookup table however well it scores on what it saw.

### 7.4 Two heads, sharp and dull, and the identity-addressed path

An act has **two verdicts on two wires**: *validity* (the referee's refusal,
per step, never fading) and *grade* (the payoff, sparse, later).
`TwoNRL_Chess` measured them on one wire at 660 → 41.8 refusals a move and
on two at 97.9 → 1.1; `Research/Insights.md` 19 files it as "validity
first, then grade". In information terms the refusal is a hard parity
failure at a known position, the grade a soft score over the whole word;
mixing them throws away the position.

Credit therefore has two landings:

* **sharp** (`Verdict.sharp`): a verdict with a known step lands in full on
  that step's edge, `Machine.credit_steps`, no discount — the oracle of
  `LatticeFSM/DESIGN.md` §24, measured as good or better and learned much
  sooner. The referee, the sandbox, a tool and the taste gate are sharp on
  every step they answer.
* **dull**: a verdict with no step — a teacher's mark on a whole reply, a
  benchmark drop hours later — lands discounted back along the run
  (`Machine.credit`, `DISCOUNT 0.8`), and **never across a sensing-act
  boundary `|`**: at 0.95 the first of 64 cells of a picture receives 0.04
  of the verdict and the last fourteen over half, which is the periphery,
  the opposite of the fovea, and is why `image>3` sits at chance.
  `credit_steps` is wired at the `Layer` level (today `Network.teach`
  credits through `Block.credit`, discounted, and `credit_steps` exists on
  `Machine` only).

A third addressing exists for a verdict that arrives too late for either:
**by identity of what was consumed** (`Verdict.origin`: a smell signature, a
path). Conditioned taste aversion is the body's version; here a crash or a
benchmark drop after ingesting a resource blames that resource's signature
in the guard (`Guard.blame`) and credits the sniff that chose it, hours
later, without asking the discount walk to reach it (`0.95^100 = 0.006`).

### 7.5 Own credit, per sensing act

Every part is credited by **its own verdict**: the decoder by the world,
the sense lattice by its per-step verdict, the encoder's game by the
credited state (§4.4), the act lattice by its forward-model error (§8.2).
Shared credit through a vote measured worse on every language (0.62 against
1.00 on contains-aa) and stays a setting. The unit of credit is the
**sensing act** — a fixation, an utterance, a tool call, a ply, marked `|`
in the stream — and nothing is discounted across one.

The lattice's credit has no baseline (`LatticeFSM/DESIGN.md` §10). This
design adds one as a **measured option**: `credit_baseline k` subtracts
`k × Edge.net` from a verdict so a predicted verdict moves an edge less
(an edge's own expectation is its baseline; `k ∈ {0, 0.5, 1}` in E1).

### 7.6 The lattice as a code of itself

`LatticeFSM/DESIGN.md` §11 folds the matrix into its central node exactly,
7.5–12× smaller than the dense matrix, and measured that the information in
a taught lattice is *sparse, not low-rank*: the bitmap of touched edges is
the structure, the touched edges need full rank. Two consequences here:

* **Touched edges per shell** is reported beside every number (E9), as the
  size of what was learned.
* **Consolidation is compression on a clock** (§12.2): `compress_every`,
  `rearrange_every` and the window's ladder are the sleep of this organism,
  and a precision below `exact` is a lossy channel to the next day whose
  loss `compress.fidelity` measures in nats per row.

### 7.7 The decoder as a channel decoder

Decoding is replay through the referee (§6.4), which is maximum-likelihood
decoding with the rules as the code: the received path is mapped to the
nearest act the world accepts, walking the ranking until one passes
(`uge/metrics.py propose`). The four break kinds are the syndrome. The
`phi` break — the tape's own state token disagreeing with the replayed
state — is a **checksum the encoder wrote for itself**, and it is the one
free, per-step, teacher-less verdict every action decoder has; the forward
model is taught by it (§8.2). `modelkit/media.py repair_base64` and
`uge/tape.py trim_to_tokens` are the erasure-correction step before the
syndrome is read: cut to the grid, then let the rules cut the rest.

### 7.8 Information as the gaze

Active sensing is a probe policy. GREN derived that for a binary oracle the
expected information gain is highest at `p(legal) = 0.5`
(`GREN/DESIGN.md` §4: *the target failure rate is 50 %*). The analogue for a
13-state prediction is the entropy of the row's predicted-state
distribution: the eye saccades, the nose sniffs and the hand probes where
the forward model is least sure (§9.4). Surprise — the forward model's
disagreement at the landing site — is the gaze's credit (bounded, because a
noisy periphery is always surprising: a `(location, fixations_spent)`
resource and the clock exit cap it).

### 7.9 Informational media

The body reads *information*, not only the world. The transducers for it
exist and are reused unchanged:

| medium | transducer | one symbol | notes |
|---|---|---|---|
| text | `radixnet/encoding.py` units; `latticefsm/media.py text` | a character, a word, a phone, a syllable | the text sense; also the hand's read-back (§9.2 touch) |
| English in and out | `PhoneticEncoder` / `PhoneticDecoder` (43 sounds) | one sound | sounds, not letters, between English and the machines |
| bytes of any file | `latticefsm/b64.py`, the `base64` unit | one base64 letter, cut at whole groups so a letter always means the same six bits | images, video, archives as one alphabet; `ngram.generate_bytes` writes bytes back |
| archives, documents, markup | `latticefsm/corpus.py`, `modelkit/archive.py` | — | zip, tar, gz, html, docx, epub opened; `file_kind` is the nose's first receptor |
| structured records | the lattice as the grammar (§6.2 structured) | a token of the record | accepting states = well formed; the sandbox or schema is the referee |
| a model file | `compress.Core` | — | a lattice is itself information the body can hand over (§7.6) |

### 7.10 The rate table

What each organ can carry per symbol and per second, so that a stream using a tenth of its alphabet, or a lattice asked to hold more than `log2 S` bits of context, is visible before it is trained. Measured entries are marked; the rest are the alphabets' capacities and are to be replaced by measured entropies (`ngram evaluate` reports bits per token; the LatticeFSM N-gram measured char 5-grams at perplexity 4.485, about 2.2 bits a character, and word trigrams at 180, about 7.5 bits a word, on its corpus).

| stream | alphabet | capacity per symbol | symbols per second | per second | measured |
|---|---|---|---|---|---|
| text, characters | 43 kept characters | 5.4 bits | — | — | about 2.2 bits a character on `data/corpus` (char 5-gram) |
| text, words | the vocabulary | `log2 V` | — | — | about 7.5 bits a word (word trigram) |
| English sounds | 43 phonemes (88 with stress) | 5.4 (6.5) bits | 10–15 segments | 55–100 bits | — |
| the ear | 64 acoustic units | 6 bits | ~36 units (2.8 frames a run, bundled codebook) | ~216 bits at most | the same unit back 86 % streamed, 92 % polished |
| the eye, one fixation | fovea 4 levels × 16 cells, periphery 4 × 8, surround 5, change 5 | ~2 bits a cell; ~50 bits a fixation | 3–4 fixations | ~150–200 bits | the stripes set is separable at 47/60 by a local two-cell rule in rows order |
| the gaze | 32 vectors + stay + pursue | 5.1 bits | 3–4 | ~18 bits | — |
| touch, one contact frame | 4 levels, 5 changes, slip, held | 3.6 bits a patch | one frame per contact | — | — |
| smell, one sniff | `k ≤ 8` of `F = 256` buckets | `log2 C(256, 8)` ≈ 46 bits a sniff at most | one per sniff | — | Ochiai error 0.128 at 64 dims (GREN's table) |
| taste, one bite | 15 | 3.9 bits | one per bite | — | — |
| interoception | 8 gauges × 4–5 bands | ~18 bits a frame | one per `k` steps | — | — |
| a game move, flat | `\|A\|` | 14.3 bits for chess | one per ply | — | **refused** as a lattice symbol: factored into ≤ 6-bit sub-decisions |
| the tape token | 3 base64url characters | 18 bits; **15** with a readable phase | one per token | — | the cliff at one token (0.943 / 0.514 / 0.939) |
| the encoder's state | `S` | `log2 S`: 3.7 bits at 13, 5.4 at 43 | one per step | — | E2 measures `I(state ; truth)` |
| the lattice's memory | one state | `log2 S` | — | — | — |
| an effector | its alphabet | `log2 \|alphabet\|` | **one symbol per tick, one channel** | the bottleneck | the voice: one sound per symbol read today (input + 2 at most) |
| credit | — | `Σφ = log P(y) − log(1/\|V\|)` bits per stage game | one per step | — | efficiency exact per permutation |

Two readings of the table are rules. First, the body's input is parallel and wide (the eye's fixation and the ear's units together run to hundreds of bits a second) and its output is serial and narrow (one symbol a tick), so the lattice between them is a **bottleneck by design**: what the encoder names is a compression of the fixation to `log2 S` bits, and the congestion game is the pressure that makes that compression spend its bits on what is rare and right. Second, nothing may be asked to carry more than its row: a symbol whose meaning needs the symbol before it (a phase, a position in the frame) is two symbols or a disjoint letter, never a convention the reader is expected to know (§7.3).

---

## 8. The loop: efference copy, reafference, babbling

### 8.1 The copy

At emission the decoder calls `Efference.copy(symbol, clock, modality, ttl)`
(`efference.py`), re-expressing the act in the alphabet of the sense that
will read the answer: a phoneme becomes the acoustic units `say()` will
make (a translation learned by babbling), a `<tool>` line and a tape token
are identity. The act lattice's row `(s_t, a_t)` is read quietly
(`Machine.step(quiet=True)`) as the predicted `t_t`. The `ttl` is the loop's
latency — one utterance, one call, one ply — never a life. The
`Transition` gains a `self` tag (`own` / `given` / `repaired`), the agency
tag: a step the organism did not choose is a null player and earns no
innate credit.

### 8.2 Three verdicts for what returns

| verdict | condition | what happens |
|---|---|---|
| **reafferent** | the returning symbol matches the copy within the measured tolerance (the ear's: the same unit 86 % streamed, 92 % polished) | the edge is traversed (`seen`, `use_widening 0.01`); `credit_steps` receives **exactly 0.0** (`mask_reafferent`). No seat is paid for naming a state the organism's own emission made predictable. The reafferent share of the step is a meta-feature. |
| **error** | the symbol differs, or the copy expired unmatched | a **lesson on the act lattice only**: `credit(−1)` on the predicted edge, `teach(s_t, a_t, s_actual, +1)` on the named one — never on the repertoire (it did what it was told), never on the encoder's game (it named what arrived). Compared against the drawn edge's target, not the greedy prediction; gated by the forward model's own reputation (the newborn's grace); silent on quiet runs. |
| **exafferent** | no copy: the world spoke | the only sensation that may carry the world's credit: the referee's break kind with its reason, the sandbox's `not ok`, a `ToolResult.error`, a person's words, a mark — onto the decoder's repertoires by Shapley among the seated candidates, and onto the encoder's game as `−λ` only for a belief-level reason (§4.4). |

This is why the earlier draft's "free self-supervised signal" is retired:
two internal estimates agreeing carry nothing about the world, and
rewarding the agreement rewards the dialogue loop and the dark room — the
two-voice fixed point the author recorded in `Research/CyclesAreAFeature.md`
§5.2 (its prediction 2 is unrun; E7 runs it with the copy on and off).

### 8.3 Babbling, listen-first

Before any teacher, each effector emits random symbols of its alphabet —
`uge/corpus.py generate_plays(teacher=RandomPolicy)` through the referee for
a move, the mouth through `say()`, the hand through the sandbox — reads
the return through its own sense, and the act lattice's rows learn by
lessons. Repertoires are uncredited during babbling: there is no right
babble. The sense lattice is trained quietly on its corpus to a plateau
*before* its effector babbles into it. The repository did this once by
hand: the bundled acoustic codebook was learned from the synthesizer's own
sentences.

### 8.4 The ear hears the mouth

`ModelKit` pauses the ear and mutes the dictation while the mouth speaks
(`frontend/src/voice.js pause / mute`), so today the organism never hears
itself. The target design is predict-and-suppress: the ear stays open,
own-voice units carry the copy, nothing of them is trained on as another's
speech, only disagreement is credited; the dictation's words stay muted so
a reply is never written down as the person's. The first build is the
`say → hear_audio` loop itself, which no package code closes today
(`recall.quiz(listen_back)` goes through ASR); half-duplex stays until the
agency tag is trusted end to end (E7).

---

## 9. The body

### 9.1 Tiers

| tier | body | repo piece or new class | fixed / learned / grown, by what |
|---|---|---|---|
| 0 world | contact, the opponent, another speaker | `uge/game.py`, `codec.py`; `Sandbox.run`; `ToolBox.call`; new `referee.py` | fixed; the referee is a property of games and compilers, not of the body |
| 1 receptors | retina, cochlea, skin, nose, tongue, spindles | `latticefsm/media.py` (+ new `fixate`, `surround_symbols`, `motion_symbols`); `phonetok/acoustic.py` (+ a frame gate, a running mean); new `senses/*` | fixed, deterministic; only the acoustic codebook is learned, unsupervised |
| 2 gate and copy | thalamic reticular gating (a model, not settled); corollary discharge | new `gate.py` over `stimulate` / `step(stimulation=)`; new `efference.py`; the `self` tag | one learned number per modality (the surge gain); the reserve set by the meta-player; the copy never credited |
| 3 cortex: naming | columns with partial information | new `encoder.py` (GameRadix) over `radixnet/graph.py`; the stage game from `GTMNN/DESIGN.md` §7–17 | the congestion payoff, regret matching, sampled Shapley |
| 4 sense lattices | V1 / A1 / S1 maps with their own time constants | `Layer(states, emit, **machine_options)` | own credit, `credit_steps` per step; verdicts never fade, traces do |
| 5 act lattices | the forward model (the cerebellum's role is the best-supported hypothesis, not settled) | `Machine` with the act alphabet; new `forward.py` | lessons from the periphery only |
| 6 association | binding window (superior colliculus deep layers as the shared aligned map: the body has both layouts) | `Layer(13, life=4)` over `Parallel(merge='interleave')`; new `clock.py`, `coalition.py` | own credit; cross-sense coalitions paid as one player; built only if E9 says voting is not enough |
| 7 selection | basal ganglia: release by disinhibition, default no (one model of their function) | new `select.py`; `uge/metrics.py propose` | Shapley over seated candidates; the released channel pays the price win or lose |
| 8 effectors | eye muscles, vocal tract, hands | `modelkit/voice.py Speaker`, `phonetok/synth.py`, `ToolBox.run`, `Game.apply`, `media.fixate` | fixed; one `ActionDecoder` with a `Referee` per effector |
| 9 the body's own senses | vagus, locus coeruleus, nociception, dorsal horn | new `senses/intero.py`, `temperature.py`, `pain.py` over `Machine.stats()`, `RunResult`, `guard.py`, `negative.py` | innate valence as the floor; blame / clear / reasons; `forget(reason, factor)` |
| 10 monitors | the inner monitor (whether it runs through comprehension is contested), the stop signal, freezing | `modelkit/dialogue.py` stutter / Heard / backtrack, `thinking.py`, `autotemp.py`, `radixnet/metacog.py` | BACK and THINK learned from experience; meta-policies credited counterfactually |
| 11 time and curriculum | a Weber clock, sleep replay, critical periods, the fetal order | `SharedClock`; new `curriculum.py` over `radixnet/training.py`, `schedule.py` | settings on a schedule; closure by saturation (`COEFFICIENT_LIMIT 8`, widths 0.05 / 20) |

### 9.2 The senses

Lives are proposals, not measurements; `LIFE = 1000` is the only number
`machine.py` has, and each life below is a setting E8–E10 sweep.

| sense | one symbol is | alphabet | map or code | life | active output | reuse | build |
|---|---|---|---|---|---|---|---|
| sight | one cell of one fixation in its region: fovea `L0..L3`, periphery `P0..P3`, isotropic surround `S−2..S+2` (the retina's centre-surround), oriented contrast `H/V` (V1, not the retina), change `D−2..D+2`, `\|` per fixation | 14–35 | map in the ring order; the symbol axis a code | retina ~24 ticks (one fixation; iconic memory 250–500 ms); scene 1000 | the saccade vector | `media.py cells, order('center-out'), image_symbols, diff_symbols`; `modelkit/vision.py` (the VAE eye for photographs) | `fixate(image, at, fine, coarse, levels)`, `surround_symbols`, `motion_symbols` from two consecutive diffs (one diff is polarity-confounded, measured on the clip generator); a fixation driver outside `Network` (which runs one pass, one encoder first) |
| hearing | one acoustic unit `q_k`: a run of 10 ms log-mel frames at one centroid (~36 units/s on the bundled codebook, 2.8 frames a run — two to four times the phone rate) | 64 units + `\|`; 43 phonemes when English is read | code; tonotopy lives in the 40-band front end | 64–128 ticks (one utterance; echoic memory 2–4 s); calm 16–32 | `attend(stream, window)`: the Endpointer's gain, the codebook, whether to pay for ASR | `phonetok/acoustic.py Analysis, Codebook, AcousticTokenizer.listen`; `media.py audio_units`; `AudioEncoder`; `AudioImage` (a tonotopic plane) | a frame RMS gate emitting `tick(k)` for silence (`listen` has none: quiet is units); a running mean (`phonetok` DESIGN §5.4, not done); a monotone aligner against the transcript as oracle; the `S = 43` preset |
| touch | one contact on one patch: `LEVEL L0..L3` (slowly adapting) or `CHANGE D−2..D+2` (rapidly adapting), `\|`, slip, held | 12 | map in the read order and a neighbour table *beside* the lattice | CHANGE ~64; LEVEL 1000 (two layers reading one stream) | the fingertip's position; the next probe | `media.py cells, diff_symbols`; `tools.py DEFAULT_OBSERVATION_CHARS` | `TouchEncoder` over a text surface (a file window, a terminal, a page); `ContactMemory` comparing `(size, mtime_ns)` — `Encoder.cache` only stores; the `topographic` flag |
| temperature | one gauge's signed step from its adapted baseline, `T−2..T+2`, framed by gauge (seconds, bytes, memory, a rate limit) | 10–25 | code; one level for the machine (the homeostatic thermal sense; the discriminative one is touch's) | adaptation over calm | the budget for the next contact | `machine.py stimulation`; `Sandbox` RLIMITs, `RunResult`; `tools.py max_chars` (100..20000 web, 100..100000 file) | `Gauges`; the clamp handed to pain; `stimulate(k · step)` on a warm step — a stressor, not heat |
| pain | **no symbol**: a fault `{reason, severity, span, source, origin}` that becomes `−severity` on the span's edges | ~23 reason tags (`blame.py`), ≤ 8 per edge | neither: a credit stream | the verdict never fades; it **habituates** by `forget(reason, factor)` and sensitises on repeats (wind-up, capped at 2.0) | failing on purpose where a reason is likeliest | `blame.py _fault, CODE_SEVERITY, severity_from_rating`; `guard.py`; `negative.py`; `ENCODING.md` §7 | `Pain.hurt`: sharp (span known, `credit_steps`) vs dull (`punish`); the reason side table; `origin` on the fault |
| smell | one receptor bucket that fired: the signed hash of an offset-free n-gram bag of the sample, the strongest `k ≤ 8` in a canonical order, then `\|` | `F = 256` read | map-less code: no adjacency between buckets | background tens of sniffs | the sniff order `(target, size)`: a verb from `{deeper, again, shallower, next, back, done}` and, for `next`, a target ranked over the open frontier by signature similarity | `GTMNN/DESIGN.md` §6.2 (a spec), `GREN/DESIGN.md` §22.5, `CyclicCortex/cortex/routing.py`, `corpus.py file_kind`, `tools.py _address_problem` | `Odotopes`, `BagHasher`, `Nose`; measure Ochiai error at `F ∈ {61, 128, 256, 1024}` first; novelty switches the reading to `mode='explore'`, never a machine-wide surge |
| taste | one innate line's verdict on one bite: `{empty, unreadable, loop, known-failure, clean} × 3 intensities` | 15 | neither; **read twice**, as a symbol in the stream and as an amount on the edge that consumed the bite | one tick; its ledger never fades | the gate: swallow / caution / spit; the next bite's size | `Guard.judge`, `autotemp.py why_no_answer`, `Encoder.encode` (drops unknown symbols), `duo.py FilterConfig.learn` | `Taste`, `Mouth.ingest` **on the way in**, before `Network.teach` / `fit` / `ngram.train_corpus` |
| proprioception | one transition as felt: the shell of the edge (`geometry.shell`), its probability band, skipped or not | 56 | topographic in the cube's shells | 5–10 ticks | the focus (where a run starts) | `machine.py Transition`, `geometry.py shell`, `focus.py FocusLearner` | `ProprioEncoder`, beside the named state, not instead of it |
| vestibular | the signed change of a gauge step to step (cost, stimulation, wealth) in five bands | 15 | code | ~10 ticks | the surge reflex | `media.py DIFFS`, `machine.py stimulate` | `VestibularEncoder`; the reflex gain learned by whether the next step's cost fell |
| interoception | one gauge band in a fixed-order frame `\| S W R X A G T K` (stimulation, wealth, reputation, exploitability, abstain share, the guard's verdict, time since the last reward, the generation) | 32–40 | code; the frame order is the map | 10 000–100 000 ticks | the sampling period `k`; the drives: hunger lowers the reserve, a surge raises it | `machine.py stats()`, `api.py status()`; `GTMNN/DESIGN.md` §7, §14.2, §17 (spec) | `InteroEncoder`; the ledgers (wealth, reputation, price) built first inside the encoder's stage game |
| time | the age of a situation in log bands of the reading layer's life, `T0..T4` | 15 | code; an exponential trace read in log bands is a Weber clock (a derivation, not a measurement) | none of its own: it is the fades | `tick(k)` on purpose; when to consolidate | `machine.py clock, tick, compress_every`; `radixnet/window.py` | an idle clock (wall seconds → explicit ticks); a two-process sleep gate (§12.2) |
| agency | the tag of the last step: `M` mine and as predicted, `X` mine but not, `G` given, `R` repaired | 4–12 | code | per step | the prediction committed before the step | `dialogue.py Turn.given / fresh`, `thinking.py Thought.taught`, `agent.py mediate_call` | the `self` tag on `Transition` and on every path record; a reader for it in the meta-frame |

**What the repository already is, organ by organ.** The eye is
`media.py`'s image, diff and video encoders and `vision.py`'s VAE. The ear
is the 64-unit codebook and the phoneme inventory. The mouth is `voice.py`
and the formant synthesizer. The hands are the tool box, the browser and
the sandbox. Pain is the sandbox's exit code, a tool refusal and a
referee's break flowing through `blame.py` into the negative network and
the guard. Taste is `Guard.judge`. Hearing yourself is `dialogue.py`'s
stutter and backtrack. Arousal is stimulation; time is the clock and
`tick`. What is missing is the loop and the gate, not the organs.

### 9.3 The effectors

| effector | output type | one output is | mechanism | read back by | reuse | build |
|---|---|---|---|---|---|---|
| voice | symbol stream + value stream (prosody) + abstain | one phoneme (or unit) to `Speaker.feed`, `</s>` to `end()`; pitch / tempo / gain per word (`Synthesizer` reads `Voice` live) | its own walk on a voice act lattice; repertoires `K = 4 + ABSTAIN` with `END` and the pauses always present; `PhonotacticReferee` before speaking | hearing (`say().wav()` → `hear_audio`), the free inner loop on tokens | `voice.py Speaker, speak_walks, say`; `phonetok/synth.py`; `phonotactics.py next, violations`; `syllables.py` | `SoundDecoder`, `ProsodyDials`; a `truncated` fault when `max_length` cuts (a breath budget later) |
| hands / tools | action, structured | one `<tool>name {...}</tool>` line | propose-and-replay over tool × argument chunk × procedure × budget; the guard vetoes per candidate before replay | the `<result>` read as a haptic frame plus the verdict symbol (`Readback`) | `tools.py call_text, parse_call, ToolBox.run`; `agent.py _next_call, mediate_call`; `codegen.py Sandbox.run` | `ActionDecoder`, `ToolBoxReferee`, `SandboxReferee`; the two missing hands: a confined `write_file`, a guarded browser script (today the hands are read-only on the world) |
| movement / game moves | action | one move `apply()` accepted, assembled from sub-decisions | repertoires per sub-decision row, release above a reserve, the referee walks the ranking | the new state's `state_key`, the opponent's reply | `uge/codec.py decode`, `metrics.py propose, rank_tokens, greedy_rollout(fallback=True)` | the factored tape (`ENCODING.md` §11, unimplemented) |
| writing / text | symbol stream | one token of a turn | repertoires on the text act lattice; the guard on the output path | the text sense reading its own line | `dialogue.py reply, Heard`; `tools.py transcript_text` | the `self` tag on written steps |
| gaze | action | one vector `G‹amplitude›‹direction›` (1, 2, 4, 8 × 8 directions), `stay`, `pursue` | repertoires on the fixation's last edges; `EyeReferee`; a `(location, fixations_spent)` resource bounds `stay` | vision: the next fixation; the command copy is exact | `focus.py FocusLearner` (the continuous fallback: two focuses) ; `graph.py observe_back / observe_think` | `GazeDecoder`, `FixationEncoder.fixate(image, at + vector)`, the loop driver |
| expression / affect | structured + prosody + trace | the affect frame as JSON beside the reply, the dials, the narrator's lines | **Moore**: a readout of the gauges the credit uses; `credit` a no-op | the ear (own prosody, tagged), interoception | `assistant.py _Narrator`, the `radixnet` field on every answer; `autotemp.py climb_note` | `ExpressionDecoder` |
| balance / continuous control | value | a continuous control into a physics loop | a sine controller; the one effector where **inversion is kept** (`TwoNRL_CartPole`: trained to fail 9.2 steps, one sign flip, 499.4 / 500) | the next observation | `Experiments/TwoNRL_CartPole/sinenet.py` | a `ValueDecoder` wrapping it; no lattice row (its alphabet is continuous) |
| abstain | abstain | no symbol, with why: nothing / no words / a loop / vetoed / abstained | the resting state (§6.2) | the unpaid price, the clock | `search.py onward`, `graph.py observe_back / observe_think`, `thinking.py think`, `autotemp.py climb`, `duo.py NegativeFilter.judge` | a lattice abstain (§5.4); a `wait` output with a tick count |

### 9.4 What is not the gaze

The attention band is the *blame* fovea (where a correction lands inside a
gram), the dynamic window is the gram's grain, the focus is where a run
starts. All three are kept as they are. Gaze is a new **action** output
that selects the encoder's next read and is read back by the sense; the
`FocusLearner`'s learning rule (a Gaussian policy against a baseline) is
reused as the gaze policy's continuous fallback.

---

## 10. Attention and selection as games

### 10.1 The gate: seats on the sense side

`gate.py`: a reserve per modality (0.01), `surge(modality, amount)` calling
that lattice's `stimulate`, onset hooks from the encoders (the Endpointer's
start, a `|` frame, a `<result>`). The seat auction of §4.3 is the thalamic
gate: bidders below the reserve are not read this step; `bidders < M` is
the No-bidders trigger. The meta-player may `SUPPRESS` or `ATTEND` a
modality by moving its reserve, **capped in duration and logged**, so a
sense is never starved to bankruptcy. Innate refusals (an address, a kind
of file) and the world's refusals with a reason bypass the auction. The
auction's truthfulness holds only while it is the only attention mechanism
(`GTMNN/DESIGN.md` §26): no learned attention weight sits beside it.

### 10.2 The congestion game, and what it does not do

`B / n_a` within one stage game: right and rare takes the pot; cells
specialise within a fixation, stop-word odotopes starve. The split says
nothing about re-reading across time: inhibition of return is a repeat rule
plus the clock, never a congestion payoff. Whether the `recent`
coefficient of an edge's own weighting turns negative under it is read
after training (E13), not assumed.

### 10.3 Coalitions are integration

Within a sense, adjacent seats with complementarity `1 − |rf_i ∩ rf_j| / R ≥ 0.5`
may form a coalition (an edge detector is two neighbours of opposite level,
if it happens — unmeasured). Across senses, seats of different lattices
naming one state are **paid as one player** with Shapley inside, committed
by the stag hunt only with trust history (`GTMNN/DESIGN.md` §13). Without
this rule integration would be punished as redundancy. Reliability-weighted
cue combination is reputation per context; `cortex/credit.py ensemble` is
the running precedent at region grain.

### 10.4 The output channel, and the meta-player

The `ActionSelector` of §6.3 is the basal-ganglia reading: the default is
no, one channel is released above a reserve with a margin, a photo finish
escalates. The meta-player (`meta.py`, from `GTMNN/DESIGN.md` §14) reads 16
features of the deliberation plus 6 slow body gauges normalised apart;
its triggers are Cycle, Photo finish, No bidders, Unconverged, Surprise,
Hurt and an interoceptive `stuck` (exploitability high and wealth falling);
its policies `TRUST_EQUILIBRIUM`, `TRUST_REPUTATION`, `TRUST_COALITION`,
`TRUST_CYCLE_AVERAGE`, `ABSTAIN`, and the body's `MOVE`, `SUPPRESS`,
`ATTEND`; depth 2 with `TRUST_REPUTATION` at the cap. For a mouth a cycle
resolves to a pause then `BACK`, never to the cycle's average: there is no
average of two sounds.

### 10.5 Reflex, freeze, flight, fight

Two latencies, never one. `BACK` and the guard's veto are reflexes that run
before the auction and never through it. `THINK` is the slow loop after the
aggregate fails, and it **cools first** (`think(temperature < 1.0)`) before
the auto-temperature ladder heats. The order is the body's and the paper's:
freeze (THINK), flight (BACK, a fresh text), fight (inversion, kept for the
guard's negative network and the continuous controller only) — cheapest
first.

### 10.6 The homunculus

Area by use has **no mechanism** in a fixed `S × A` lattice and is not
claimed. What use changes is width (the odds), the shell a busy state sits
in (`rearrange` on the state axis, so what compression keeps) and who wins
the congestion split. Blind spots are refilled by births only if the
players are a micro population (§21.1).

---

## 11. Layout

One class, three kinds (§5.1): sense lattices and act lattices with an
association lattice **beside** them, never between an encoder and its
decoder. From the repository's numbers: `enc>13>dec` 0.03 / 0.00 exact;
`(3|3|3)` own 1.00 where `3>3` is 0.77 and shared 0.62; the one two-sense
stream (`video>13`, one encoder interleaving image and audio units per
frame) exactly chance while `audio>3` alone reached 0.67; stimulation and
life per machine come free only with per-sense lattices.

Regions are derived, not asserted, and the rule is CyclicCortex's measured
one, not alphabet disjointness: `Cortex.add_game` founds a region when the
mechanics distance to every centroid exceeds `tau_new = 0.60`
(`cortex/cortex.py`). Here `found_lattice(encoder)` founds a lattice when
the new stream's *symbol distribution* is farther than `tau_new` from every
existing lattice's, and `rewire(encoder, lattice)` is a legal operation.
Founding is cheap; retiring is not — nothing in LatticeFSM deletes edges.

CyclicCortex's one measured result on regions argues the other way for the
association lattice: zero-shot transfer 0.646 against 0.514 came from a
**shared generalised vocabulary** across games. So the association
lattice's alphabet is an experiment, not a rule: modal letters shifted per
branch (the `Parallel` way) against a shared generalised alphabet of the
senses' states, both in E9. Under `SYMBOLS` (62 letters a block writes) the
association lattice holds at most four 13-state senses unless the file
format gains two-character labels — a format change in both LatticeFSM
ports (§21.9).

**What decides it (E9):** `video>13` (the measured shared-alphabet control,
0.50) against `(diff>13 | audio>13)` majority with own credit, against the
same with an association `>i 13`, against the same with a shared
generalised alphabet; three seeds, both ports where the patch has landed,
with the protect test (`audio>3` at 0.67 must not fall) and gini over the
branches' wealth and touched edges per shell reported.

---

## 12. Time and curriculum

### 12.1 The curriculum

`curriculum.py`: `Stage(modalities_live, rate, life, explore, share, replay)`.
The default `ORDER` is the body's — `('touch', 'body', 'chemical',
'hearing', 'vision')`, text read as touch — and the repository's habit,
cheapest oracle first (`text, games, sounds, acoustic, image`), is the
measured control, because the author's turn asked for the body and nothing
yet shows the order matters for a lattice. A new sense comes online when
the live ones plateau and the earlier sense's held-out score has not
fallen. Each effector babbles before it is credited (§8.3). Plasticity is a
schedule on rate, life and explore (`radixnet/schedule.py`) that closes by
saturation and reopens only by an explicit act.

### 12.2 Sleep

Consolidation is `compress_every`, `rearrange_every`, the window's ladder
and (if the players are a population) `cull_and_birth`, gated two-process:
a homeostatic counter **and** a circadian rung, with replay from the
`ReplayBuffer` rather than a pass over the whole corpus (Insight 22: train
in one, query the other).

### 12.3 The idle clock

Nothing ticks between requests today. Wall-clock idleness enters as
explicit `tick(k)` calls written to the model file (so a saved model reads
the same in both ports); the LatticeFSM server is Rust-only, so the idle
clock the server applies is a two-port build.

---

## 13. Package layout

```
GameLattice/
  DESIGN.md                 this file
  README.md                 what is measured and what is not (E1–E13), the quick start
  Makefile                  test / demo / experiments / the babbling and layout runs
  gamelattice/
    __init__.py             exports GameGraph, GameLattice, Decoder, Verdict, Referee, Efference, Body, __version__
    verdict.py              Verdict, the reason taxonomy re-exported from blame.py, the innate-sign table
    referee.py              Referee protocol; GameReferee (uge), ToolBoxReferee, SandboxReferee, EyeReferee, PhonotacticReferee
    credit.py               credit_steps per Layer, mask_reafferent, direct() for a refusal with a reason, the baseline option,
                            the identity-addressed path (blame by origin)
    efference.py            Efference.copy / match, the per-modality translator (decoder alphabet -> reading alphabet),
                            the self tag on Transition
    forward.py              predict(lattice, s, a) quietly; learn(s, a, s_actual): the lesson; the collapse guard (agreement rate)
    game.py                 the stage game: seats, bids, the congestion payoff, regret matching plus, sampled Shapley,
                            CycleDetector (from GTMNN/DESIGN.md §8-12, with cortex/routing.py as precedent)
    encoder.py              GameGraph(RadixCyclicGraph): strategy / regret / wealth / reputation per node; the window per modality;
                            name(s) -> s_t; observe_think / observe_back on a cycle
    lattice.py              GameLattice: the three kinds over Machine / Layer; the topographic flag; the abstain floor; the reasons table
    select.py               ActionSelector: the release rule; coalition.py: within and across senses; meta.py: the meta-player
    gate.py                 reserves per modality, surges on onsets, SUPPRESS / ATTEND with a cap
    clock.py                SharedClock over Machine.tick; the idle clock
    decoders/
      base.py               the Decoder protocol; readouts (verdict, label, state stream, value, trace); ExpressionDecoder
      repertoire.py         repertoires on edges: SymbolDecoder (text), SoundDecoder (+ ProsodyDials), structured output
      action.py             ActionDecoder: propose-and-replay; GazeDecoder; the factored tape for moves
      taste.py              Taste, Mouth.ingest: the gate on the way in
    senses/
      text.py               the text sense over radixnet.encoding (tier 1)
      eye.py                fixate, surround_symbols, motion_symbols, FixationEncoder (tier 1)
      ear.py                the frame gate, the running mean, the aligner (tier 1)
      readback.py           a (call, result) pair -> haptic frame + verdict symbol + gauge step (tier 1: the hand's feedback)
      touch.py, smell.py, intero.py, proprio.py, vestibular.py, temperature.py, pain.py, chrono.py   (tier 2)
    babble.py               babble(effector, sense, lattice, steps, rng)
    curriculum.py           Stage, ORDER, the plateau gate, the schedules
    body.py                 Body: the organism wired - encoders, lattices, decoders, the loop, the clock
    experiment.py           E1-E13 (§17), each writing results/<name>.json
    cli.py                  argparse: demo, babble, train, play, experiment
  tests/                    unittest (run: `python -m unittest discover -s tests -v` from GameLattice/)
  data/                     the four languages' examples come from latticefsm.languages; games from uge; media from media.make_dataset
  results/                  what E1-E13 wrote, and results/README.md reading them
```

**Module dependency order** (no cycles): `verdict` → `referee` → `credit`
→ `efference` → `forward` → `game` → `encoder` → `lattice` → `select`,
`coalition`, `meta` → `gate` → `clock` → `decoders/*` → `senses/*` →
`babble` → `curriculum` → `body` → `experiment` → `cli`.

**The one patch outside this directory:** `LatticeFSM/latticefsm/network.py`
gains a leading `Parallel` whose branches each start with their own
`Encoder` (today an encoder can only be the network's first block and
every branch reads the same input), per-branch layer options in the
architecture string (`3[life=10]`), a per-branch pause symbol for a
missing modality, and `credit_steps` through `Layer`. Its Rust twin lands
when §21.8 says so.

**Dependencies.** Required: `radixnet`, `latticefsm`, `uge`. Extras:
`phonetok` (the ear, the mouth, the phonotactic referee), `modelkit`
(the hands, the monitors, the narrator, the VAE eye). A `Body` built with
an extra missing says which organ it lacks and runs without it.

---

## 14. Persistence

One JSON object (`.json`, or `.json.gz`): `format: "gamelattice-body"`,
version, the settings (every life, calm, reserve, `K`, `W`, `B`, `λ`, the
curriculum stage), the shared clock, each lattice as its own
`latticefsm-machine` document under its name and kind, the `GameGraph` as a
radixnet graph document (format 4) plus its per-node strategy, regret,
wealth and reputation arrays, each decoder's repertoires keyed by
`(source, symbol, target)`, the reasons side table, the efference copies
still live (with their `ttl`), the gate's reserves and caps, and the
curriculum's position. A lattice document is readable by the LatticeFSM
ports on its own; a graph document by `radixnet`. Floats are written as
floats; nothing is a hash of a process-salted `hash()`.

---

## 15. Tests

The tests encode the design choices, not the code:

* `test_regression.py` — a `GameGraph` with no lattice attached predicts
  what `CountRewardGraph` predicts, edge for edge.
* `test_game.py` — on `CorrectnessCongestion` best response terminates with
  `cycle == 0` and the potential strictly increases; on two-player
  rock-paper-scissors the cycle has period 3 per sweep; `Σφ = v(N) − v(∅)`
  to `1e-9` per permutation.
* `test_forward.py` — the repertoire is unmoved by a mismatch-only step;
  the act lattice's row moves.
* `test_efference.py` — a model talking to itself credits no edge
  positively on a reafferent step; an unmatched copy becomes a lesson.
* `test_credit.py` — a sharp verdict lands on one edge in full and on no
  other; a dull one never crosses a `|`; `credit_baseline 1` moves a
  perfectly predicted edge by zero.
* `test_decoders.py` — every decoder names a `reads_back`; a decoder with
  none is refused; the release rule abstains on a photo finish; the
  taste gate never lets a `reject` into positive training.
* `test_layout.py` — the protect test: `audio>3` at 0.67 does not fall when
  a diff branch is added.
* `test_body.py` — a `Body` round-trips through `to_dict` / `from_dict`;
  a `Body` missing `phonetok` says so and runs without an ear.

---

## 16. The frontend and the server

Deferred until E1–E9 have numbers. When they do, the LatticeFSM frontend
gains a Body tab (the senses live, the gate's reserves, the clearing price
as the pupil, the affect frame) over the LatticeFSM Rust server's routes,
and nothing in this directory serves HTTP itself.

---

## 17. Measurement plan

Each experiment names its number to beat and its control; `experiment.py`
writes `results/<name>.json`, three seeds unless stated.

| id | question | number to beat | controls |
|---|---|---|---|
| E1 | does the stage game earn its cost? the four languages, 4 000 episodes | own 0.894 / oracle per step 0.924 / explore 0.944 | the plain oracle lattice; rewards-only; whole-run discount 0.8; `credit_baseline k ∈ {0, 0.5, 1}`; **bits per symbol** beside accuracy |
| E2 | the one-token cliff: Nim at φ = 20 | 0.943 / 0.514 / 0.939 at 15 / 20 / 30 | `W ∈ {1, 4, 8, 16}`; `I(named state ; true state)` in bits; coverage |
| E3 | chess refusals with a local digest | 98.63 (guessing 1894.1) | syntax vs illegal counted apart; legal@1; coverage; touched edges |
| E4 | babbling before the teacher | unaided plies chess 1.0 / Nim 7.7 / Pig 134.0 / Connect Four 12.0; the multi arm's 14–27 % cross-game moves | no babbling; 2NRL (expected to lose: Nim 0.943 → 0.438) |
| E5 | English pairs through a voice act lattice | 0.09 / 0.00 held out; `enc>13>dec` 0.03 / 0.00 must stay the loser | the `PhoneticDecoder` as is |
| E6 | hearing: audio plurals, 5 seeds | 0.67 held out | unit rate 36/s vs a minimum hold; `S = 43` vs 13 |
| E7 | the ear hearing the mouth | a false-error rate under the chosen tolerance; `CyclesAreAFeature` prediction 2 with the copy on and off | half-duplex |
| E8 | vision with credit per fixation | image>3 0.25 / 0.50 (ceiling 51/60 on the stripes set), diff>3 0.58 | end-discounted credit; one-diff vs two-diff motion symbols |
| E9 | layout (§11) | video>13 0.50 to raise; audio>3 0.67 to protect | shared alphabet; voting; association; generalised alphabet |
| E10 | the surge: one reflex table (comparator miss +0.5, jolt +1.0, punishment +0.5·\|amount\|, guard reject +0.5, bankruptcy +2.0, reward −0.25; soft ceiling; calm shorter than life) | the static dose-response of `LatticeFSM/results/stimulation_results.json` | no body (no surge); `step(stimulation=)` per step with no state |
| E11 | taste on the way in: a corpus salted with known failures | rejects blamed and never trained; held-out perplexity not worse | the guard on the way out only |
| E12 | the hand against `agent.py`'s beam | `agent.py`'s solve rate on its own tasks | the mediator's repair counted as the helper's hand |
| E13 | gaze: `GazeLearner` (continuous) vs `GazeDecoder` (repertoires); the `recent` coefficient after training | the fixed `center-out` order | random saccades |

Every number is reported with its seeds, its touched edges, and for a
game its coverage. A result below the number to beat is recorded as
**contradicted**, in the index's word, not tuned away.

---

## 18. Performance notes (must be followed)

* The population forward pass of GTMNN does not exist here: the seats are
  at most a window, so the stage game is `O(W · S)` per step and is pure
  Python. The costs that matter are the quiet re-runs Shapley needs over
  a sensing act (`2^F` is refused; 32 permutations at any seat count) and
  the acoustic ear (about a tenth of real time in pure Python: hearing
  back every utterance costs several times the voice).
* Per-node arrays are flat lists indexed by node id, like the graph's;
  no per-node objects in the hot path.
* Every search state carries a monotone resource; every generator stops
  on a resource or a monitor, never on exhaustion (`Research/CyclesAreAFeature.md` §8).
* Determinism: same seed, same corpus, same settings → same bytes, across
  a save and load; no process-salted hashing anywhere (the smell's hasher
  is FNV-1a).

---

## 19. Decisions

* **The encoder's output is the lattice's source state.** Asked for ("each
  output is a node"), and it is what makes the lattice the forward model
  rather than a second encoder. Alternative rejected: the encoder emitting
  the symbol, which leaves the lattice nothing to predict.
* **Players are graph nodes in the walk window** (recommended; §21.1 is
  the author's). Nodes already exist, are born from the corpus, and carry
  the gram the game is about; a micro population would add GTMNN's whole
  `MicroPool` and Evolver before the first number.
* **The lattice's symbol is per lattice kind;** actions are factored. A
  flat action alphabet is refused by arithmetic (3.46 M edges).
* **Reafference earns nothing.** The agreed free signal rewarded the
  dialogue loop; attenuation keeps traversal's `use_widening` and zeroes
  the credit, to be measured against a small attenuated credit (§21.5).
* **Two heads for an act** — validity per step, grade later — on the
  measured 97.9 → 1.1.
* **Shapley is always sampled.** There is no exact mode and `2^16` is not a
  per-step cost; efficiency is exact per permutation anyway.
* **Abstain is the resting state,** a reserve, not a fifth symbol; `BACK`
  a reflex before the auction, `THINK` the slow loop after; neither is
  the lattice's skip (a likelihood shortcut).
* **Credit per sensing act, never discounted across `|`;** sharp verdicts
  by `credit_steps` at the `Layer` level.
* **Taste on the way in as well as out.** The repository's stance was "the
  guard is on the way out, not on the way in" (`RadixCyclicNN/DESIGN.md`
  §24.7); the body judges every bite before swallowing. Spit from the
  positive, swallow into the negative (`FilterConfig.learn` is the
  precedent). The critic and the tutor still sample the positive model
  unfiltered on the way out; that part stands.
* **No inversion on a symbolic effector.** 2NRL lost to its matched control
  on Nim, Connect Four, Othello and chess. Inversion stays with the guard's
  negative network and with the one continuous controller where it won
  (CartPole), which has its own output type.
* **Half-duplex stays** until the agency tag is trusted end to end; the
  first build is the `say → hear_audio` loop.
* **The fetal order is the default curriculum,** cheapest-oracle-first the
  control, because the author asked for the body and the repo's order has
  no bodily warrant.
* **Regions by distance, not disjointness** (`tau_new = 0.60`, the measured
  rule); the association alphabet is an experiment because the measured
  transfer came from a shared vocabulary.
* **Topography beside the lattice, never on the symbol axis;** area by use
  is not claimed.
* **Contested biology is carried as a design choice with its control:**
  the thalamic gate, the cerebellar forward model, selection by
  disinhibition, the inner-speech monitor, primacy coding in smell, the
  strict serial bottleneck. None is a rule the body supplies; each has an
  experiment or is marked as needing one.
* **Dependencies:** three siblings required, two as extras; the GTMNN
  pieces are built here against GTMNN's DESIGN.md (§21.6 may move them).
* **Python reference first.** Rust parity follows the first numbers
  (§21.8), as every sibling did.

---

## 20. Changes to the earlier design, on the record

The design agreed in the first discussion is corrected here, not silently:

1. The lattice's symbol was "the stream's own symbol"; it is now **per
   lattice kind** (receptor symbols on a sense lattice, the act on an act
   lattice), and actions are factored.
2. "Agreement `t_t == s_{t+1}` is a free self-supervised signal" is
   **demoted to a meta-feature**; the forward model is taught by the
   periphery only (§8.2).
3. "Roughly 8–16 seats, so Shapley is exact" is **sampled Shapley at any
   seat count**; seats are per modality, not one walk.
4. "The decoder is Mealy, per edge, the last few edges of the path" holds
   for symbol streams only; actions are **propose-and-replay** with the
   repertoires as proposer and the referee as decoder; expression is a
   Moore readout; taste is a gate; the hearing side is Moore.
5. "One credit stream, each part by its own verdict" becomes **two
   landings** (sharp per step, dull discounted within a sensing act), a
   third addressing by identity, and the forward model's lesson apart.
6. "Abstain triggers auto-temperature or the meta-player" becomes
   **abstain is the default**, a reserve; `BACK` and `THINK` separated by
   latency.
7. "A hidden machine between enc and dec measured 0.03–0.11 exact" is
   corrected to **0.03 / 0.00 exact (0.11 / 0.04 sounds right)**; the
   conclusion stands.
8. "S = 13 by default but a setting" becomes **per lattice**, with a 43
   preset for the auditory lattice as the author's call.
9. "Importing radixnet, latticefsm and uge" gains **phonetok and modelkit
   as extras**: eight of the seventeen organ rows reuse them.
10. "First targets are the four languages and the UGE games" is widened:
    the layout is measured first on the two media tasks that already have
    numbers (E9), because the body's claims are claims about integration.
11. Stimulation had no afferent; it now has **a reflex table** (E10) and a
    soft ceiling, per lattice.
12. The guard moves **onto the way in** (taste).

---

## 21. Open questions for the author

1. **Players.** Graph nodes born from the corpus and never culled, or a
   micro population with births to the lowest coverage? Only the second
   gives bankruptcy, immigrants and a helplessness detector; it costs
   GTMNN's `MicroPool` and Evolver.
2. **What E1 must show.** Beating 0.924 / 0.944, or matching them at a
   per-symbol cost? If the game merely matches, the plain lattice with
   `credit_steps` is the encoder and the game lives in the decoders only.
3. **Curriculum default.** The fetal order (text read as touch, the
   languages delayed) or cheapest oracle first.
4. **The metabolic cost.** A cost per traversal the lattice never had, so
   an interoceptive floor exists and abstaining is not free — and every
   measured LatticeFSM table is rerun in both ports — or off by default.
5. **Reafference at exactly 0.0,** or a small attenuated credit measured in
   E7. Zero ends the two-voice conversations sooner, which may be the
   intended reading of the cycles paper.
6. **Where the game-theory code lives:** `gamelattice/game.py`, or GTMNN's
   own first code (a sixth sibling, implemented against its DESIGN.md).
7. **The auditory lattice's size:** 43 states, so a state is its sound and
   a transcript is a per-step oracle (120 k edges), or 13; and whether the
   voice act lattice shares it, which decides whether the forward-model
   lesson `teach(s, a, s_actual)` is well typed across the ear and the
   mouth.
8. **Port parity timing:** every new symbol, option and tick in both
   ports with a parity test from the start, or Python first and Rust after
   E1–E9. The LatticeFSM server is Rust-only.
9. **The association lattice:** from the start, or only if E9 says voting
   is not enough; and two-character labels in the file format to lift the
   62-letter cap.
10. **One attention market across effectors** (talk or act as a priced
    decision, a shared reserve) or separate games on separate clocks with
    a coalition rule for coordinated acts.
11. **A third kind of player:** a source ledger (wealth and reputation per
    smell signature, read by the sniff order), which is not a node and sits
    in no stage game.
12. **The hands that write:** a confined `write_file` and a guarded browser
    script make refusals costly and some acts irreversible; today every
    hand is read-only on the world.

---

## 22. What is deliberately not here

* **No backpropagation between parts**, and none between nodes: credit is
  Shapley inside a stage game and a lesson on a row; the sine's two-layer
  gradient never crosses a node (`GTMNN/DESIGN.md` §26).
* **No learned attention beside the auction**, for the reason GTMNN gives.
* **No second lattice between an encoder and its decoder** as the path;
  it stays an option on the measured number.
* **No 2NRL on a symbolic effector.**
* **No topography on the symbol axis**, and no claim of area by use.
* **No credit for reafference.**
* **No frontend or server of its own** until the numbers exist.
* **No organ admitted without a number already in the repository that it
  must beat and a control that would show the body reading bought
  nothing.** The author's index records insights as held, confirmed,
  refined or contradicted; this directory's results are to be read the
  same way.
