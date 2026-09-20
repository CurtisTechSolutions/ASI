# Using it

How to put a game on the tape, train RadixCyclicNN on it, and get moves back
out — and, at every knob, *why* the default is what it is.

Three documents, and they answer different questions:

| | |
|---|---|
| [`ENCODING.md`](ENCODING.md) | **why the scheme is what it is.** The argument: every finite discrete game is one object, and encoding one is choosing a projection of the walk |
| [`README.md`](README.md) | **what happened when it was measured.** Seven arms, six games, three seeds |
| **`USING.md`** (this file) | **how to actually use it**, and why each setting |

Everything here is executed, not pasted. The worked example is
[`examples/hexapawn.py`](examples/hexapawn.py) and it runs in about ten seconds.

---

## 1. Why you would use this

Because RadixCyclicNN eats text, and you have a game. The naive move — write
the moves out in whatever notation the game already has — **does not work**, and
the reasons are not obvious:

- The network's window is 3 characters with stride 1, and every trigram of the
  corpus lives in exactly one node. A notation whose tokens are 4 or 5
  characters (UCI, SAN, most game notations) puts the same move at a different
  offset in different games, so the trigram index splits it across many nodes
  and the graph learns the *spelling* rather than the move.
- A token can be read in three different window phases, and the graph cannot
  tell them apart. Up to **45 % of trigram occurrences** are ambiguous that way
  in a naive encoding, and the damage concentrates on the busiest nodes.
- Two *kinds* of token that spell the same three characters are one node. The
  first version of this encoder had the outcome token `draw` and Connect Four's
  "column 1" both encoding to `BAr`, so every game's ending was also a legal
  move.

This package is those problems already solved, plus a decoder that is also the
referee, plus a corpus generator, plus the measurement battery. The cost of
adding a game is the cost of writing the game.

### When not to use it

- **You want a strong player.** This is an encoding study on a first-order
  model. It learns Nim well and cannot play a legal move of chess unaided.
- **Your state does not fit 15 bits and your action does not generalise.**
  See §5 — chess is the worked failure.
- **You need the model to generate a whole game unprompted.** It cannot; see
  [README limitation 3](README.md#what-it-cannot-do).

---

## 2. Setup

Nothing to install. Standard library, plus `RadixCyclicNN/` from this
repository, which is found two directories up automatically:

```bash
cd Experiments/UniversalGameEncoding
make check     # byte-compiles, and confirms radixnet was found
make test      # perft, codebooks, the code-space partition, round trips
```

If you have moved this directory, set `RADIXNET_PATH` to wherever
`RadixCyclicNN/` is. `uge/` itself has **no** dependency on `radixnet` — only
`uge/metrics.py` imports it — so the encoder can be used on its own.

---

## 3. Quickstart

### Encode a game and read it back

```python
import uge
from uge.tape import TapeSpec
from uge.codec import decode
from uge.corpus import self_play
from uge.teachers import make_policy

game = uge.get_game("tictactoe")
spec = TapeSpec("tictactoe", game.index, phi_bits=15)

corpus = self_play(game, spec, 1, make_policy("search3"), seed=0)
tape = corpus.tapes[0]
print(tape)
# LM_CWtGArKm4IAriQxHArRT8DArIX8EArdY7CArXC6AArIR0FArmq_

result = decode(game, spec, tape)
print(result.plies, result.complete, result.outcome)
# 8 True second
```

Read the tape in threes: `LM_` is the header, then `CWt GAr` is
(position, move), eight times, then `mq_` is the result.

### Train, and ask for a move

```python
from uge.corpus import split
from uge.net import new_net, TRAIN_DEFAULTS
from uge.codec import encode_prefix
from uge.metrics import propose

corpus = self_play(game, spec, 300, make_policy("search3"), seed=0)
train, test = split(corpus, 0.2)

model = new_net(seed=0)
model.train(train.tapes, **dict(TRAIN_DEFAULTS, epochs=8))

state = game.initial()
prefix = encode_prefix(game, spec, [state], [])      # the question
answer = propose(model, game, spec, prefix, state)   # the answer, graded

print(answer["top"], answer["legal_at_1"], game.action_str(answer["action"]))
# FAr True 5
```

`encode_prefix` writes the tape up to and including the current position's `φ`
token. The network's next three characters are a move. That is the whole
interface.

---

## 4. The five things you touch

| object | what it is |
|---|---|
| `uge.get_game(name)` | one of the six, or your own once registered |
| `TapeSpec` | every encoding decision in one frozen object — §5 |
| `encode` / `decode` / `encode_prefix` | play ⇄ tape, and the referee |
| `uge.corpus.self_play` / `split` / `positions` | where training data comes from |
| `uge.metrics.propose` / `evaluate` / `rank_tokens` | asking, and grading |

Two that are worth knowing about:

**`rank_tokens(model, prefix)`** returns *every* three-character continuation
the model knows, cheapest first, by expanding the graph exhaustively. It returns
`[]` when the last trigram of the prefix is not in the graph — which is a
different answer from a wrong one, and is why `coverage` appears beside every
score. Do not use `RadixNet.predict(mode="beam")` for this: a beam of width `B`
silently truncates the ranking, and any "how far down the list" number you get
from it is a property of `B`.

**`decode` is the referee.** It replays the tape through the game's own
transition function, so it is total, self-checking, and tells you *where* the
tape stopped being a game:

| `break_kind` | meaning |
|---|---|
| `none` | the rules never refused |
| `header` | wrong game |
| `syntax` | a token in an action slot names no action at all |
| `illegal` | it names an action this position does not allow |
| `phi` | every move was legal and the state token disagrees with the replay |

That last one is the tape's own checksum. A tape that breaks at ply *k* is a
tape the game refused, which is a 2NRL negative produced for free — no engine,
no labels.

---

## 5. Choosing a `TapeSpec`, and why

```python
TapeSpec(
    game="tictactoe",
    game_index=game.index,
    phi_bits=15,          # how much position goes on the tape
    phi_prev_bits=0,      # ... of which, how much is the last move instead
    layout="lsb",         # how an action becomes three characters
    book="disjoint",      # which alphabet split
    condition_result=False,
    with_outcome=True,
)
```

### `book` — `"disjoint"` (default) or `"plain"`

**Use `disjoint` unless you need a structured layout.** It reserves the last
character of every token to an alphabet used nowhere else, so a trigram's window
phase is readable from the trigram and no trigram can be read two ways.

It costs 3 bits of abstraction (15 a token instead of 18) and chess's
`(from, to, promotion)` layout. Measured:

| game | ambiguous occurrences under `plain` | refusals `plain` → `disjoint` |
|---|---:|---|
| Connect Four | 45.4 % | 10.65 → **0.34** |
| Nim | 10.7 % | 0.93 → **0.13** |
| Othello | 34.4 % | 42.00 → **7.97** |
| chess | 6.6 % | 1727 → 1725 (**no change**) |

Chess is the exception that explains the rule: its tokens already use the whole
code space, so there is little ambiguity to remove. A game with a *small* action
space pads its tokens with a constant, every token then ends in the same
character, and the seams collide systematically.

### `phi_bits` — how much of the position goes on the tape

`RadixNet._locate` conditions a prediction on the **last trigram of the prefix
alone**, so the abstraction has to fit in **one token**.

- `0` — a move-only tape. The model becomes a bigram over moves.
- `1…15` — a digest of the position before every move: `P(a | φ(s))`.
- `phi_exact=True` — the game's own state code, if `state_code` returns one.

**Do not let it spill into a second token.** Only the last one is read, so the
others are tape you pay for and never use. The cliff is at the *final* token,
not at the second:

| Nim `phi_bits` | tokens | bits in the last token | legal@1 |
|---:|---:|---:|---:|
| 15 | 1 | 15 | **0.94** |
| 20 | 2 | 5 | **0.51** |
| 30 | 2 | 15 | **0.94** |

So `phi_bits` above what one token holds is a bug, not a trade. How much a token
holds depends on the game, because the actions take their share of the code
space first:

```python
probe = TapeSpec(name, game.index, phi_bits=0)
bound = game.token_bound(probe)
probe.codebook.phi_bits_for(bound)   # tictactoe: 15   chess: 13
```

`experiment.py`'s `default_spec` does exactly this, and it is worth copying.

**An injective `φ` is not automatically best.** Tic-tac-toe's exact board code
loses to a 15-bit hash (0.65 against 0.99 at covered positions) although it
throws nothing away — a hash uses the token space evenly and a board code
clusters in it, so the exact tape has 677 distinct seam trigrams where the
hashed one has 875, and its seams are shared by more contexts.

### `phi_prev_bits` — spend the budget on history instead

A hash has **no locality**. Two nearly identical positions land in unrelated
buckets, so a hashed `φ` recognises a position it has seen and says nothing at
all about one it has not; there is no such thing as a near miss.

`phi_prev_bits` moves part of the same budget onto `Game.action_summary` — a
small *meaningful* code for the move just played. For chess that is the square
it landed on: "a capture on e5" is one bucket across every position it happens
in, and the reply is usually a recapture on e5.

| chess, 13 bits | refusals/move |
|---|---:|
| `phi_prev_bits=0` (all position) | 1725 |
| `phi_prev_bits=6` | 1303 |
| `phi_prev_bits=13` (all history) | **98.6** |

Guessing costs 1894. **Rule of thumb: the bigger your state space relative to
your corpus, the more of the budget belongs on history.**

### `layout` — `"lsb"` (default), `"msb"`, `"struct"`

How an action becomes three characters. `struct` lets the game lay the fields
out itself and needs `book="plain"`.

`msb` buries the variation in the last character, and for a small action space
that is fatal — Othello scores 0.070 legality at covered positions under `msb`
against 0.599 under `lsb`. For chess all three are within noise, because its
action space is large enough that every layout discriminates. **Leave it on
`lsb` unless you have a reason.**

### `condition_result` — ask for the winning move

Mixes the game's outcome, from the mover's point of view, into `φ`. The same
position in a game its mover won and one they lost becomes two nodes, so
`P(a | φ(s), result)` is learnable, and at play time you pass
`assume_winner=<you>` to `encode_prefix`. Return conditioning for one byte of
salt and no architecture change.

---

## 6. Adding your own game

Implement `uge.game.Game` and call `register`. Required:

```python
initial()                 # s₀
legal(state)              # legal: S → 2^A, deterministic order
apply(state, action)      # τ, returning a NEW state
winner(state)             # None | -1 (draw) | 0 | 1
to_move(state)            # whose turn
action_count              # |A|
state_key(state)          # canonical bytes — what φ hashes
```

Optional, and each buys something specific:

| hook | what it buys |
|---|---|
| `struct_token` / `struct_action` | your own digit layout, so the seams mean something |
| `state_code` | an injective `φ`, if the state space fits one token |
| `action_summary` | what `phi_prev_bits` writes down — make it *meaningful* |
| `is_chance` / `chance_weights` | a die or a shuffle; see below |
| `heuristic` | a search teacher; without it you get a random one, which still contains the rules |
| `render` / `action_str` | readable output when something fails |

Then `game.validate()` checks the action space fits every codebook and
round-trips through every layout.

### The worked example

[`examples/hexapawn.py`](examples/hexapawn.py) is a complete new game — 3×3
chess with pawns, not one of the built-in six — in about seventy lines of rules
plus three encoding hints. Run it:

```bash
python3 examples/hexapawn.py
```

It prints the rules being exercised, one encoded game and its decode, the
decoder catching three different kinds of tampering, a trained model's numbers,
and the model playing a game. Real output from the last two sections:

```
coverage           0.990   (positions it had seen)
legal@1 covered    0.960   against guessing's 0.034
refusals/move      0.23     against guessing's 22.38
teacher match      0.405
```

```
  ply  0  2->5      model (rank 0, 3 ranked)
  ply  1  7->4      random opponent
  ply  2  0->3      model (rank 0, 2 ranked)
result: white (the model) wins after 3 plies; 0 refusals in total
```

### Chance is not a special case

A chance node is a node whose mover is the world. Its outcomes are ordinary
actions with ordinary tokens, and the tape records which one happened — so the
model *learns* the distribution instead of being told it. `uge/games.py`'s `Pig`
is in the set to prove it: nothing in `tape.py` or `codec.py` knows it has a die.

```python
def is_chance(self, state):      return state[4]          # a roll is pending
def legal(self, state):          return [2,3,4,5,6,7] if state[4] else [HOLD, ROLL]
def chance_weights(self, state): return [1/6] * 6          # uniform is the default
```

For **imperfect information**, hash the acting player's information set in
`state_key` instead of the world state, and write one tape per player. Nothing
else changes.

---

## 7. Training and measuring

```python
from uge.corpus import self_play, split, positions, unseen_positions
from uge.metrics import evaluate

corpus = self_play(game, spec, 200, make_policy("search2"), seed=0)
train, test = split(corpus, 0.2)          # whole games held out, never positions
model = new_net(seed=0)
model.train(train.tapes, **dict(TRAIN_DEFAULTS, epochs=8))

report = evaluate(model, game, spec, positions(game, test, limit=300), test.tapes)
```

Two things that will mislead you if you skip them.

**Quote legality against guessing.** For a game with seven actions, a random
token is legal about 83 % of the time, so a bare `legal@1` makes a model that
learned the rules look worse than one that learned nothing. `evaluate` returns
all four: `legal_at_1`, `legal_at_1_covered`, `legal_at_1_baseline`, and
`legal_at_1_fallback` (the model where it knows, a guess where it does not —
the honest single number).

**`refusals` and `legal@1` pull in opposite directions, and that is real.**
`legal@1` rewards a sharp ranking; `refusals` rewards a long one. A model with a
useless abstraction ranks nearly every move it has ever seen in rough frequency
order, and since only a fraction of the code space is ever legal anywhere, that
alone beats guessing handsomely. Chess at `phi_bits=15` scores 0.019 on
`legal@1` and 65.8 refusals against 1894 — it has learned the *vocabulary* of
chess moves and nothing about the position. Read them together.

For generalisation use `unseen_positions`, and know what it can tell you: under
a hashed `φ` it is **zero by construction**, because an unseen `φ` token is an
unseen node. That is not a broken measurement, it is the locality limit stated
as a set.

---

## 8. Playing with a trained model

The loop, in full — this is §5 of the Hexapawn example:

```python
state, states, actions = game.initial(), [game.initial()], []
while game.winner(state) is None:
    prefix = encode_prefix(game, spec, states, actions)
    answer = propose(model, game, spec, prefix, state)
    action = answer["action"]                       # first legal in its ranking
    if action is None:                              # model silent here
        action = rng.choice(game.legal(state))
    state = game.apply(state, action)
    states.append(state)
    actions.append(action)
```

`propose` walks down the model's own ranking to the first move the board
accepts, and reports how far down that was. **You supply the `φ` tokens from the
real position** — that is the difference between playing and generating. The
model can *read* this tape (given the position, name the move) and cannot
*write* one, because writing needs the state and only the referee has it. For
playing that is enough: the referee is always there.

If you want the strict version — the model's top token or nothing —
`metrics.greedy_rollout(model, game, spec)` is it, and `greedy_plies` is how
long it survives unaided.

---

## 9. When something looks wrong

| symptom | cause | fix |
|---|---|---|
| top-ranked token is *syntactically* impossible | phase ambiguity: a seam trigram and a whole token spell the same three characters | `book="disjoint"` |
| an injective `φ` scores *worse* than a hashed one | its codes collide with action codes, or cluster in the token space | check `phi_region(game, spec)`; `tests.py partition` asserts the regions are disjoint |
| `coverage` is high but `legal@1_covered` is low | the abstraction is too coarse to discriminate — it answers every position with the same node | raise `phi_bits`, or move bits to `phi_prev_bits` |
| `coverage` collapses as `phi_bits` rises | the abstraction is too fine for the corpus — every position is new | lower `phi_bits`, or move bits to `phi_prev_bits`, or get more games |
| adding `phi_bits` suddenly makes everything worse | it spilled into a second token | use `codebook.phi_bits_for(game.token_bound(spec))` |
| the model's cheapest complete tape is one ply | not a bug — "an action may be followed by the end" is a real edge, out of a node shared by every occurrence of that move | use `greedy_rollout`, not free generation |
| `decode` says `break_kind="header"` | the spec that encoded the tape differs from the one decoding it, anywhere | specs must match exactly; the header is a hash of all of them |
| numbers move between runs | a corpus is a function of its seed; a *spec* change re-encodes the same games | `tests.py determinism` asserts both |

---

## 10. Using the encoder somewhere else

`uge/` has no dependency on `radixnet` outside `uge/metrics.py`. It is meant to
be liftable into `RadixCyclicNN/radixnet/` next to `vision.py` and `speech.py`,
which do the same job for images and audio — "a modality, written as text the
network eats".

If you lift it, the pieces are:

- `uge/tape.py` — alphabet, codebooks, the code-space partition, `TapeSpec`.
  Knows nothing about games.
- `uge/codec.py` — encode, and decode-by-replay.
- `uge/game.py` — the interface, and the registry.

The rest (`games`, `teachers`, `corpus`, `metrics`) is experiment scaffolding,
not the encoder.
