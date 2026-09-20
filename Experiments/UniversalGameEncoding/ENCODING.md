# Is there a way to encode *all* games?

Yes, for a precisely delimited class, and the delimitation is the interesting
part. This document is the argument; [`README.md`](README.md) is the
measurement; [`uge/tape.py`](uge/tape.py) is the code.

---

## 1. A game is not a special kind of thing

Every finite, discrete, sequential game is the same object:

```
G = (S, s₀, A, legal: S → 2^A, τ: S × A → S, terminal ⊆ S, payoff: terminal → ℝⁿ)
```

a **labelled transition system with payoffs**. Chess, Go, poker, Nim,
backgammon, Connect Four and a turn-based strategy game differ in `S`, `A` and
`τ` and in nothing else. A *play* is a walk through it:

```
s₀ --a₀--> s₁ --a₁--> s₂ ... --a_{n-1}--> s_n ∈ terminal
```

RadixCyclicNN stores exactly one thing: a walk through a labelled graph. So the
question is not "how do I turn chess into text". The question is

> **which projection of the walk do I write down?**

Everything below follows from taking that question literally.

---

## 2. There are only three things on the walk

The action `aᵢ`, the state `sᵢ`, and the payoff. The grammar writes each of them
as a fixed-width **token**:

```
tape  := head body tail
head  := one token          which game, which abstraction, which layout
body  := (φ-token* action-token)*    one group per ply
tail  := one token          how it ended
```

That is the whole format. It is game-agnostic: nothing in `uge/tape.py` or
`uge/codec.py` knows that chess has bishops or that Pig has a die.

---

## 3. The token is three characters because the window is three characters

RadixNet reads text through a sliding window of `W = 3` characters with stride
1, and **every trigram of the corpus lives in exactly one node**
(`radixnet/graph.py`). A token of exactly `W` characters therefore makes one
phase of the window a whole token: the graph's nodes become game objects and
its edges become game transitions.

Anything else smears a move across phases. UCI notation is four characters
(five with a promotion), so the same move lands at a different offset in
different games, the trigram index splits it across many nodes, and the graph
learns the *spelling* of the move instead of the move.

Three characters of base64url is `64³ = 262,144` codes — more than the action
space of any game here, chess's `64 × 64 × 5 = 20,480` included.

---

## 4. The window has phases, and the graph cannot see them

This is the part that is not obvious and that decides whether any of it works.

A stride-1 window over a `W`-wide grid produces `W` phases:

| phase | trigram | what it is |
|---|---|---|
| 0 | `(c₀, c₁, c₂)` | a whole token |
| 1 | `(c₁, c₂, c₀′)` | a **seam** — the tail of one token and the head of the next |
| 2 | `(c₂, c₀′, c₁′)` | a seam |

The seams are not waste. They are a trie over the token alphabet: they let the
network share structure between actions that begin with the same character, and
they are why the digit layout inside a token matters at all.

But the graph indexes a trigram by its three characters and by **nothing else**
— not by phase, not by position. Two occurrences in different phases that spell
the same three characters are *one node*, whose children mix "the rest of this
token" with "the start of the next one". A walk that enters in one phase leaves
in another, and the three characters it emits are not a token.

Measured on this directory's corpora, only 1–4 % of *distinct* trigrams are
ambiguous — and it still wrecks the small games, because the ambiguity is
concentrated on the hot path. A seven-action game pads two of its digits with a
constant, every token ends in the same character, and the seams then collide
systematically rather than by accident. By **occurrence**, 23–40 % of the
corpus is ambiguous. The first version of this encoder measured `nim` proposing
a syntactically impossible move as its *top* choice, from exactly this.

### The fix: make the phase readable from the trigram

Reserve the last character of every token to an alphabet that appears nowhere
else. Then

| phase | pattern |
|---|---|
| 0 | `(BIG, BIG, STOP)` |
| 1 | `(BIG, STOP, BIG)` |
| 2 | `(STOP, BIG, BIG)` |

and a trigram's phase is written on its face. `43 × 43 × 21 = 38,829` is the
largest such split of 64 symbols, so the cost is

* capacity: 38,829 codes per token instead of 262,144;
* abstraction: **15 bits of state per token instead of 18**;
* structured layouts: chess's `(from, to, promotion)` needs 64 values in two
  slots and 64 > 43, so under the disjoint codebook chess falls back to a flat
  code and loses the seams that meant "where did the last move land".

Whether that trade is worth it is `experiment.py phase`, and the answer is in
the README.

---

## 5. Every *kind* of token needs its own region of the code space

The same blindness bites three more times, and this is the lesson that cost the
most:

1. The outcome token `draw` and Connect Four's "column 1" were both code 1 — so
   every tape's ending was also a legal move, and the model's single cheapest
   complete game was two tokens long.
2. The header of game 0 was code 0, which was also that game's action 0 — the
   node every tape starts at was also a move.
3. `φ` wrote the position's own code, and the **empty board** is code 0, which
   was "play cell 0". An injective abstraction — which can only carry *more*
   information than a hashed one — measured **worse**: 0.72 against 0.99 on
   legality at known positions.

So the code space is partitioned:

| region | contents |
|---|---|
| `[0, bound)` | actions — `bound` is whatever the game's layout reaches |
| `[bound, meta)` | `φ` — what is left, so a large action space buys a smaller abstraction |
| `[meta, capacity)` | header and outcome tokens, 2 056 codes at the top |

Chess keeps 13 bits of `φ`; every other game here keeps 15.

**The general rule:** if two tokens can be told apart only by where they sit in
the tape, this architecture cannot tell them apart at all. Give them disjoint
codes.

---

## 6. `φ` is the one free choice, and it is one token wide

`RadixNet._locate` conditions a prediction on the **last trigram of the prefix
alone**. The network is first-order in the window whatever we do. What that
window *holds* is the design freedom, and it is the whole experiment:

| `φ` | what the model becomes |
|---|---|
| none | `P(aᵢ₊₁ \| aᵢ)` — a bigram over moves |
| *k* bits | `P(a \| φ(s))` — a policy over an abstraction |
| injective | a transposition table: perfect where it has been, silent elsewhere |

And it has to fit in **one** token. A `φ` that spans two tokens is tape the
model pays for and never reads, because only the *last* one is looked at. The
model's memory is one token wide: 18 bits, or 15 once you buy phase-disjointness.

That cap is the sharpest thing this directory has to say about the architecture,
and `experiment.py phi` walks through it rather than asserting it. The cliff is
at the last token, not at the second: Nim scores 0.94 legality at 15 bits, 0.51
at 20 and 0.94 again at 30 — twenty and thirty both cost two tokens, but at 20
the final token carries five bits and at 30 it carries fifteen.

One more thing the sweep found that the argument did not predict: an
**injective `φ` is not automatically the best one**. Tic-tac-toe's exact board
code loses to a 15-bit hash (0.65 against 0.99 at covered positions) although it
throws nothing away, because a hash uses the token space evenly and a board code
clusters in it — at matched corpus size the hashed tape has 875 distinct seam
trigrams and the exact one 677, so the exact tape's seams are shared by more
contexts and mix them. Uniformity is worth something on its own.

### A hash is a bad abstraction, and the token can be split

A hash has no **locality**. Two nearly identical chess positions land in
unrelated buckets, so a hashed `φ` can *recognise* a position it has seen and
say nothing at all about one it has not — there is no such thing as a near miss.
For a game whose state space dwarfs the corpus that is fatal, and it is why
chess's coverage falls as `φ` grows rather than rising.

The same bits buy something else. `phi_prev_bits` spends part of the token on
`Game.action_summary` — a small, *meaningful* code for the move just played;
for chess, **the square it landed on**. That is not a hash: "a capture on e5" is
one bucket across every position in which it happened, and the reply to it is
usually a recapture on e5. Sixty-five buckets that generalise, against 8 192
that do not.

| `phi_prev_bits` | the token holds |
|---|---|
| `0` | the position only — recognition, no generalisation |
| between | some of each |
| `phi_bits` | the last move only — the move bigram, with the moves bucketed |

`experiment.py mix` sweeps it. The point is not which end wins; it is that the
budget is fixed at one token and *what you spend it on* is the design.

### Result conditioning comes free

`φ` is a digest we control, so mixing the game's outcome (from the mover's point
of view) into it makes `P(a | φ(s), result)` learnable, and at play time you ask
for the abstraction that says *won*. Return conditioning with no architecture:
one byte of salt.

---

## 7. Decoding is replay, and replay is the referee

Decoding is **not** string manipulation. A tape is decoded by replaying it
through the game's own `τ` from its own `s₀`. That makes decoding

* **total** — every tape decodes to something, because the replay stops the
  moment the rules refuse;
* **self-checking** — *where* it stopped and *why* is the output;
* **the training signal** — a tape that broke at ply `k` is a tape the game
  refused, which is precisely the negative [`Research/2NRL.md`](../../Research/2NRL.md)
  §6.5 asks for, and it costs nothing to produce. No engine, no labels, no
  Stockfish.

Four ways a tape can be wrong, told apart:

| break | meaning |
|---|---|
| `header` | the walk picked the wrong game (only possible when one net holds several) |
| `syntax` | a token in an action slot names no action at all |
| `illegal` | it names an action this position does not allow |
| `phi` | every move was legal and the state token disagrees with the replay — the tape's own checksum caught the model writing down a position it is not in |

`radixnet.vision` needs `repair_base64` to survive a garbled prediction. Here
the repair is `trim_to_tokens` plus the replay: cut to the grid, then let the
rules cut the rest.

---

## 8. What "all games" covers

| class | how | lossless? |
|---|---|---|
| finite, discrete, sequential, perfect information | directly | **yes** |
| **chance** (dice, shuffles) | the dealer is a player; each outcome is an action with a token, and the tape records which happened. `pig` is in the game set to prove this is not a special case — nothing in `tape.py` or `codec.py` knows it has a die | **yes** — and the model *learns* the distribution instead of being told it |
| **imperfect information** | `φ` hashes the acting player's information set instead of the world state. One tape per player. The scheme does not change; only `φ` moves | yes, relative to each player's view |
| **simultaneous moves** | serialise into a fixed player order within a ply and hide the co-move via `φ` — i.e. reduce to imperfect information | yes |
| **more than two players** | `payoff` is already a vector; the outcome token widens | yes |
| **non-zero-sum, cooperative** | nothing above assumed zero-sum | yes |
| **continuous actions** | quantise. The quantiser is `φ` applied to `A` rather than to `S` | **no** — lossy, and stated as lossy |
| **real-time** | fix a tick; a ply is a tick | **no** — lossy in timing |
| **infinite / continuous state** | `φ` is already an abstraction; nothing changes | already lossy by construction |

So: **every finite discrete game exactly, and everything else through a stated
quantisation — and the scheme itself never changes. Only `φ` does.**

The claim is about *encoding*, not about learning. That a game can be written on
this tape does not mean a first-order chain over the tape can play it; §10 is
the honest list.

---

## 9. Adding a game is writing the game

Implement `uge.game.Game`:

```python
initial()                 s₀
legal(state)              legal: S → 2^A
apply(state, action)      τ, returning a new state
winner(state)             None | -1 | 0 | 1
to_move(state)            whose turn
action_count              |A|
state_key(state)          canonical bytes — what φ hashes
```

and optionally `is_chance` / `chance_weights` (a die), `struct_token` (your own
digit layout), `state_code` (an injective code, if the state space is small
enough), `action_summary` (what `phi_prev_bits` should write down), `heuristic`
(a teacher), `render` and `action_str`.

Everything else — the encoder, the decoder, the referee, the corpus generator,
the metrics, the 2NRL negatives — comes for free. The six games here are each
under 200 lines, which is the argument for the interface: the cost of adding a
game is the cost of writing the game.

---

## 10. What it cannot do

Stated plainly, because the measurements in the README show all of it.

1. **The memory is one token.** 15 bits. No encoding recovers information the
   window cannot hold. This is a property of the architecture, and the encoding
   merely makes it legible.

2. **A hash has no locality.** Two nearly identical chess positions hash to
   unrelated buckets, so a hashed `φ` cannot generalise between similar
   positions — it can only recognise ones it has seen. For a game whose state
   space dwarfs the corpus that means the
   model is silent almost everywhere. The φ sweep's optimum is therefore
   **interior and game-dependent**: chess does best with *no* state at all,
   Othello with as much as it can get.

3. **The tape can be read but not written.** Playing means "given the position,
   name the move", and the referee supplies the position — that works. Writing a
   whole tape means emitting the `φ` tokens too, and a hash is not invertible,
   so the model desynchronises almost immediately. Worse, asking for the
   *cheapest* complete tape returns a one-ply game: every tape ends
   `… action, outcome`, so "an action may be followed by the end" is an edge the
   graph genuinely observed, out of a node shared by every occurrence of that
   move. A first-order chain has nowhere to keep *"but it is only move one"*.

4. **Threefold repetition is not in the chess rules here**, because it is a
   function of the history and `winner(state)` is a function of the state. It is
   an adjudication in `uge/corpus.py` instead, and the decoder reports such a
   tape as `ok` but not `complete`.

---

## 11. The next thing to try

Limits 1 and 2 have the same shape: one 15-bit token has to condition a whole
move. The scheme already contains the fix, because nothing says an action is
*one* token:

```
φ(s)  from-square  φ(s, from-square)  to-square
```

Now the `to` decision is conditioned on a digest that already knows the
from-square, and that digest can be **local** — what a piece standing there can
reach — rather than a hash of the whole board. The tape becomes a walk through a
*decision* graph rather than a game graph, the factorisation is general (any
factored action space), and the window's one-token memory is spent on the one
sub-decision in front of it instead of on the entire position.

It is a real change: `token_bound`, the decoder's group arithmetic, and the
ranking in `metrics.rank_tokens` all widen from one token to a variable number.
It is not implemented here, and this directory says so rather than gesturing at
it in a results table.
