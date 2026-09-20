# UniversalGameEncoding

**One encoder, six games, and RadixCyclicNN.** Chess, Othello, Connect Four,
tic-tac-toe, Nim and Pig are written onto the same tape by the same code, fed to
the network in `RadixCyclicNN/` unchanged, and measured against the rules of
each game by a decoder that *is* the referee.

Two questions, and they turned out to be one question:

1. **Can the Radix Tree network learn a game?**
2. **Is there a way to encode all games?**

The answer to the second is yes, for a class this directory delimits precisely,
and the argument is in [`ENCODING.md`](ENCODING.md). The answer to the first
turns entirely on the answer to the second: **what the network can learn is
decided by the encoding, not by the training.** The same network, the same
corpus and the same number of epochs go from 0.47 legality to 0.99 on one game
purely by changing how a move is spelled.

**Pure standard library.** No `python-chess`, no Stockfish, no numpy - the
chess rules are here and are proved by perft.

<!--HEADLINE-->

---

## The claim, and the answer

| # | claim | answer |
|---|---|---|
| 1 | every finite discrete sequential game can be written on one tape by one encoder | **yes** - six games, none of the encoder knows which |
| 2 | chance needs no extension of the scheme | **yes** - Pig's die is a player, its faces are actions, nothing in `tape.py` knows |
| 3 | the network can learn a game's *rules* from such a tape | **for four of six** - see the table below; chess and Othello are where it runs out |
| 4 | the decoder can be the referee, and produce the 2NRL negatives for free | **yes** - no engine, no labels |
| 5 | how much of the *position* the tape should carry has an optimum | **yes, and it is interior and game-dependent** |
| 6 | one network can hold all six games at once | **yes** - the header token is the whole selector |
| 7 | 2NRL beats a matched positive control on this task | **no** - recorded as a negative result |

---

## How a game becomes text

```
tape  := head body tail
head  := one token                 which game, which abstraction, which layout
body  := (φ-token  action-token)*  one group per ply
tail   := one token                how it ended
```

Every token is **exactly 3 characters**, because RadixNet's window is 3
characters with stride 1 and every trigram of the corpus lives in exactly one
node. A token of that width makes one phase of the window a whole token, so the
graph's nodes become game objects and its edges become game transitions.

A tic-tac-toe game, in full:

```
Jh_ iMr IAr AAt FAr FHt EAr iEt BAr WPr HAr bTs CAr SPs DAr amr AAr mq_
hdr φ   move φ   move φ   move φ   move φ   move φ   move φ   move φ   move result
```

Decoding is **replay**: the tokens are run through the game's own transition
function from its own initial state. That makes the decoder total, self-checking
and - the useful part - a referee. It reports *where* a tape stopped being a
game and *why*:

| break | meaning |
|---|---|
| `header` | the walk picked the wrong game |
| `syntax` | a token in an action slot names no action at all |
| `illegal` | it names an action this position does not allow |
| `φ` | every move was legal and the state token disagrees with the replay - the tape's own checksum caught the model writing down a position it is not in |

[`ENCODING.md`](ENCODING.md) is the full argument, including what happens to
imperfect information, simultaneous moves, continuous actions and real time.

---

## Three bugs, one lesson

The most transferable thing this directory produced is not a number. **The graph
identifies a token by its three characters and by nothing else** - not by where
it sits in the tape, not by what it means. Two *kinds* of token that spell the
same three characters are one node, with children from both roles, and a walk
that arrives as one leaves as the other.

That was learned three times:

1. The outcome token `draw` and Connect Four's "column 1" were both code 1. So
   every tape's *ending* was also a legal move, and the model's single most
   likely complete game was two tokens long.
2. The header of game 0 was code 0, which was also that game's action 0 - the
   node every tape starts at was also a move.
3. `φ` wrote the position's own code, and the **empty board is code 0**, which
   was "play cell 0". An injective abstraction, which can only carry *more*
   information than a hashed one, measured **worse**: 0.72 against 0.99 on
   legality at known positions.

The fix is a partition of the code space - actions low, `φ` in the middle,
header and outcome in a reserved region at the top - and `tests.py partition`
now asserts the three sets never intersect, for every game and every spec.

There is a fourth of the same family, one level down. A stride-1 window over a
token grid has **three phases**, and the graph cannot see them either: a trigram
that is a whole token in one place and a seam between two tokens in another is
one node. Only 1-4 % of distinct trigrams collide that way - and **23-40 % of
occurrences**, because a game with a seven-action space pads its tokens with a
constant and the collisions land on the hottest nodes. Reserving the last
character of every token to an alphabet that appears nowhere else makes the
phase of a trigram readable from the trigram, and the ambiguity goes to exactly
zero. It costs 3 bits of state and chess's structured `(from, to, promotion)`
layout.

---

## The arms

Every arm sees the same games, from the same seeds, played by the same teachers;
the corpus is generated once per `(game, seed)` and **re-encoded** per spec, so
an encoding comparison is a comparison of encodings and not of corpora.

| arm | the question |
|---|---|
| [`rules`](#rules) | six games, one encoder: how far above guessing does the tape get the network? |
| [`phi`](#phi) | how much of the position should the tape carry? |
| [`mix`](#mix) | the same budget split between a hash of the position and a summary of the last move |
| [`phase`](#phase) | does a trigram belonging to two window phases matter? |
| [`layout`](#layout) | does the arrangement of digits *inside* a token matter? |
| [`multi`](#multi) | one network, six games |
| [`twonrl`](#twonrl) | 2NRL on the refused moves, against a matched positive control |

### How to read the legality columns

For a game with a tiny action space, **guessing is a strong baseline**: all
seven of Connect Four's codes are actions and nearly all are legal, so a random
token is legal about 95 % of the time. Quoting `legal@1` alone for such a game
would make a model that has learned the rules look worse than one that has
learned nothing. So three numbers appear together:

* `legal@1` - the model's top token was legal, counting a position it has never
  seen as a miss;
* `legal@1 covered` - the same, over positions it *had* seen (`coverage` says
  how many those were);
* `guessing` - what a uniform guess over the action space scores here.

`refusals` is the other headline and the one [`Research/2NRL.md`](../../Research/2NRL.md)
§6.5 asks for: how many moves the board refuses before accepting one, against
what guessing costs. For chess that baseline is about 1960 - roughly 30 of
20 480 tokens are legal in a typical position.

<!--RULES-->

<!--PHI-->

<!--MIX-->

<!--PHASE-->

<!--LAYOUT-->

<!--MULTI-->

<!--TWONRL-->

---

## What it cannot do

Every one of these is measured above, not hedged.

1. **The memory is one token.** `RadixNet._locate` conditions a prediction on
   the last trigram of the prefix alone, so a `φ` that spans two tokens is tape
   the model pays for and never reads. The cap is 18 bits, or 15 once you buy
   phase-unambiguity, and the `phi` sweep walks straight through the cliff.

2. **A hash has no locality.** Two nearly identical chess positions land in
   unrelated buckets, so a hashed `φ` can recognise a position it has seen and
   say nothing about one it has not. That is why chess's coverage collapses as
   `φ` grows, and why the `mix` arm exists: the same bits spent on *the square
   the last move landed on* mean the same thing in every position they appear
   in.

3. **The tape can be read but not written.** Playing is "given the position,
   name the move", and the referee supplies the position - that works. Writing a
   whole tape means emitting the `φ` tokens too, and a hash is not invertible,
   so the model desynchronises almost at once (`free_break = "phi"`). Asking for
   the *cheapest* complete tape is worse: it returns a one-ply game, because
   "an action may be followed by the end" is an edge the graph genuinely
   observed, out of a node shared by every occurrence of that move. A
   first-order chain has nowhere to keep *"but it is only move one"*.

4. **Threefold repetition is not in the chess rules here.** It is a function of
   the history and `winner(state)` is a function of the state. It is an
   adjudication in `uge/corpus.py` instead, and the decoder reports such a tape
   as `ok` but not `complete`. The side effect is one this repository likes: a
   repeated position is a genuine **cycle** in the graph
   ([`Research/CyclesAreAFeature.md`](../../Research/CyclesAreAFeature.md))
   rather than a terminal state.

5. **The teachers are weak.** No Stockfish, so chess is taught by a depth-1
   search over material. For learning *rules* that is not a problem - a corpus
   of weak play contains the legal moves exactly as completely as a corpus of
   strong play - but no claim about playing *strength* is made anywhere here,
   and Nim is in the set because it is solved and therefore has an exact answer
   to "was that move good".

[`ENCODING.md` §11](ENCODING.md#11-the-next-thing-to-try) has the next step,
which addresses 1 and 2 together: nothing says an action must be *one* token.
Split it - `φ(s)`, from-square, `φ(s, from-square)`, to-square - and the one-token
memory is spent on the sub-decision in front of it instead of on the whole
position. It is not implemented here, and this directory says so rather than
gesturing at it in a results table.

---

## Contents

| file | what it is |
|---|---|
| [`ENCODING.md`](ENCODING.md) | **the argument**: why every finite discrete game is the same object, and what the encoding of one costs |
| `uge/tape.py` | the universal layer - alphabet, codebooks, the code-space partition, `φ`, `TapeSpec`. Knows nothing about games |
| `uge/codec.py` | encode, and decode-by-replay. The referee |
| `uge/game.py` | the interface a game implements, and the registry |
| `uge/games.py` | the six games |
| `uge/chess_rules.py` | chess in pure Python, proved by perft |
| `uge/teachers.py` | the seeded policies that generate corpora |
| `uge/corpus.py` | self-play, splits, and the unseen-position set |
| `uge/metrics.py` | the exhaustive ranker and the whole measurement battery. The only module that imports `radixnet` |
| `uge/net.py` | finding `RadixCyclicNN/`, and the shared training settings |
| `experiment.py` | the seven arms, and `summary` |
| `tests.py` | perft, the codebooks, the partition, round trips, the referee, determinism |
| `results/` | the JSON every table above is generated from - [`results/README.md`](results/README.md) says how to read one |

---

## Running it

Nothing to install. The experiment finds `RadixCyclicNN/` two directories up;
set `RADIXNET_PATH` if this directory has moved.

```bash
make test          # perft to depth 3, the codebooks, the partition, round trips, the referee
make test-full     # plus perft to depth 4 and every spec combination
make quick         # every arm at a tenth of the corpus, one seed - a few minutes
make all           # every arm, three seeds - this is what the tables above quote
make summary       # regenerate the tables from results/*.json
make show          # print one encoded tape per game, and its decode
```

One arm at a time: `make rules`, `make phi`, `make mix`, `make phase`,
`make layout`, `make multi`, `make twonrl` - or
`python3 experiment.py <arm> --seeds 0 1 2 3`.

### Adding a game

Implement `uge.game.Game`: `initial`, `legal`, `apply`, `winner`, `to_move`,
`action_count`, `state_key`. Optionally `is_chance` / `chance_weights` for a
die, `struct_token` for your own digit layout, `state_code` for an injective
abstraction, `action_summary` for what `mix` should write down, and `heuristic`
for a teacher. Everything else - encoder, decoder, referee, corpus, metrics,
2NRL negatives - comes for free. The six here are each under 200 lines.

---

## Where this sits

* [`RadixCyclicNN/`](../../RadixCyclicNN/) is the network. This directory does
  not modify it; `uge/` has no dependency on it at all except in
  `uge/metrics.py`, and is meant to be liftable into `radixnet/` next to
  `vision.py` and `speech.py`, which do the same job for images and audio.
* [`TwoNRL_Chess/`](../../TwoNRL_Chess/) also plays chess, and answers a
  different question with different equipment: it measures **2NRL against a
  baseline** using `python-chess` and Stockfish, and hands the network the legal
  moves' *space* but not the moves. This directory measures **the encoding**,
  has no dependencies, and covers six games.
* [`NeuralCompression/`](../NeuralCompression/) asked whether one network can
  hold many games and had to build an output partition and a selector to ask it.
  The [`multi`](#multi) arm asks the same question of real games, and needs no
  machinery: the header token is the selector, and it falls out of the encoding.
* [`Research/2NRL.md`](../../Research/2NRL.md) §6.5 is why the network is not
  handed the legal moves. The [`twonrl`](#twonrl) arm tests §11's comparison on
  a game the paper did not use, with the rules themselves as the judge.
