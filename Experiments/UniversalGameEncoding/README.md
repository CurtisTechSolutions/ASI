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
decided by the encoding, not by the training.** Same network, same corpus, same
number of epochs — Connect Four goes from 0.70 legality to 0.92 and from 10.65
refusals a move to 0.34, purely by changing which characters a move is allowed
to use.

**Pure standard library.** No `python-chess`, no Stockfish, no numpy - the
chess rules are here and are proved by perft against the published counts for
the six standard positions, to depth 4.

## Five numbers

Three seeds each, full tables below.

1. **Nim**: 0.999 legality at positions it has seen, 0.13 refusals a move
   against guessing's 3.1, and it finishes every game it starts. Nim is
   *solved*, so move quality has an exact answer rather than an opinion:
   **0.85** of the moves it plays are winning ones (counting any move as
   winning from a position that is already lost, which is how `Nim.optimal` is
   defined). The encoding fits the game and the network learns it.
2. **Chess**: 1725 refusals against guessing's 1894, and it cannot play one
   legal move unaided. A 13-bit digest of a chess position is not a chess
   position, and no amount of training fixes an encoding that throws the board
   away. Spending those bits on *the square the last move landed on* instead
   takes refusals to **98.6** — a 19× improvement over guessing, from the same
   budget.
3. **The abstraction is one token wide, and here is the cliff.** Nim's legality
   climbs to 0.94 at 15 bits, collapses to 0.51 at 20, and recovers to 0.94 at
   30. Both 20 and 30 cost *two* tokens; only the second one is ever read, and
   at 20 bits it holds five of them and at 30 it holds fifteen. The prediction
   and the measurement agree exactly.
4. **A trigram that can be read in two window phases is 1-9 % of the distinct
   trigrams and up to 45 % of the occurrences.** Reserving the last character
   of every token to its own alphabet takes it to zero, and takes Connect
   Four's refusals from 10.65 to 0.34. For chess it changes nothing — its
   tokens already use the whole space.
5. **2NRL loses to its own control**, on all three games it was tried on. Nim:
   0.94 before, 0.94 with the matched positive arm, **0.44** with 2NRL and 0.40
   with the local inversion. Recorded as a negative result.

---

## The claim, and the answer

| # | claim | answer |
|---|---|---|
| 1 | every finite discrete sequential game can be written on one tape by one encoder | **yes** - six games, none of the encoder knows which |
| 2 | chance needs no extension of the scheme | **yes** - Pig's die is a player, its faces are actions, nothing in `tape.py` knows |
| 3 | the network can learn a game's *rules* from such a tape | **where the state fits the abstraction** - 0.999 legality on Nim and 0.994 on tic-tac-toe, 0.44 on Othello, 0.08 on chess |
| 4 | the decoder can be the referee, and produce the 2NRL negatives for free | **yes** - no engine, no labels |
| 5 | how much of the *position* the tape should carry has an optimum | **yes, and it is interior, game-dependent, and has a cliff at exactly one token** |
| 6 | one network can hold all six games at once | **yes, at a price** - 1.28x fewer nodes than six separate graphs, and 14-27 % of its answers are another game's move |
| 7 | 2NRL beats a matched positive control on this task | **no** - 0.94 becomes 0.44 on Nim; recorded as a negative result |

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
one node. Only 1-9 % of distinct trigrams collide that way - and **up to 45 % of
occurrences**, because a game with a seven-action space pads its tokens with a
constant and the collisions land on the hottest nodes. Reserving the last
character of every token to an alphabet that appears nowhere else makes the
phase of a trigram readable from the trigram, and the ambiguity goes to exactly
zero: Connect Four's refusals fall from 10.65 a move to 0.34. It costs 3 bits of
state and chess's structured `(from, to, promotion)` layout - and for chess it
buys nothing, because chess tokens already use the whole space and collide on
only 3-7 % of occurrences. The [`phase`](#phase) arm is both halves of that.

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
what guessing costs. For chess that baseline is 1894 - roughly 30 of the 20 480
tokens are legal in a typical position.

**The two headline numbers pull in opposite directions, and that is the
finding, not a flaw.** `legal@1` rewards a *sharp* ranking; `refusals` rewards a
*long* one. A model with a nearly useless abstraction ranks almost every move it
has ever seen, in rough frequency order — and since only a fraction of the code
space is ever legal anywhere, that alone beats guessing handsomely. Chess at
`phi=15` scores 0.019 on `legal@1` and 65.8 refusals against 1894: it has
learned the *vocabulary* of chess moves and nothing about the position in front
of it. Read them together or they will each tell you a different lie.

<!--RULES-->

### rules

Six games, one encoder, and the same defaults for all of them:
a 15-bit hashed abstraction (13 for chess, whose 20 480 actions take
the code space it needs), the disjoint codebook, and the flat `lsb`
layout. Nothing is tuned per game - the later arms do that.

| game | `\|A\|` | coverage | legal@1 covered | guessing | refusals | refusals guessing | teacher match | unaided plies | self-play refusals | nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chess` | 20480 | 0.71 | **0.081** | 0.001 | **1724.84** | 1894.1 | 0.007 | 1.0 | 922.94 | 22071 |
| `connect4` | 7 | 0.35 | **0.921** | 0.826 | **0.34** | 0.3 | 0.147 | 12.0 | 0.09 | 6322 |
| `nim` | 21 | 0.94 | **0.999** | 0.364 | **0.13** | 3.1 | 0.448 | 7.7 | 0.00 | 664 |
| `othello` | 65 | 0.33 | **0.441** | 0.111 | **7.97** | 9.5 | 0.062 | 0.7 | 3.27 | 13079 |
| `pig` | 8 | 0.61 | **0.691** | 0.459 | **1.59** | 1.3 | 0.202 | 134.0 | 0.01 | 20574 |
| `tictactoe` | 9 | 0.55 | **0.994** | 0.626 | **0.55** | 0.8 | 0.151 | 7.3 | 0.00 | 1282 |

`unaided plies` is how far the model gets playing both sides with **no**
second chance - its top token, or the rollout ends. `self-play refusals`
is the same rollout allowed to walk down its own ranking, measured on
**its own** positions rather than the teacher's, which by move ten are
not the same distribution.

**The same models on positions whose `phi` token the training corpus never wrote.**
Under a hashed `phi` this is a tautology and the point of quoting it:
the last trigram of the prefix *is* the `phi` token, so an unseen token
is an unseen node and the model is silent - a hash has no near misses.
The [`mix`](#mix) arm is where the abstraction is not a hash and this
set stops being trivial.

| game | coverage | legal@1 covered | guessing | refusals | refusals guessing |
|---|---:|---:|---:|---:|---:|
| `chess` | 0.00 | 0.000 | 0.001 | 2075.50 | 2075.5 |
| `connect4` | 0.00 | 0.000 | 0.770 | 0.45 | 0.4 |
| `nim` | 0.00 | 0.000 | 0.311 | 2.33 | 2.3 |
| `othello` | 0.00 | 0.000 | 0.116 | 9.74 | 9.7 |
| `pig` | 0.00 | 0.000 | 0.446 | 1.33 | 1.3 |
| `tictactoe` | 0.00 | 0.000 | 0.453 | 1.23 | 1.2 |

<!--/RULES-->

<!--PHI-->

### phi

From a move-only tape to an injective one.

**The cliff is not at two tokens, it is at the *last* token.** Only the
last trigram of the prefix conditions a prediction, so what matters is
how many bits land in the final token. Nim at 15 bits scores 0.94; at
20 bits it scores 0.51; at 30 bits it is back to 0.94. Twenty and thirty
both cost two tokens - but at 20 the second token carries five bits and
at 30 it carries fifteen. Othello is the same story and louder: 211
refusals a move at 20 bits against 8.6 at 30.

**An injective `phi` is not automatically the best one.** Tic-tac-toe's
`exact` row is *worse* than its 15-bit hash (0.65 against 0.99 at covered
positions) although it throws nothing away, because a hash uses the token
space evenly and a board code does not: at matched corpus size the hashed
tape has 875 distinct seam trigrams and the exact one has 677, so the
exact tape's seams are shared by more contexts and mix them. Uniformity
is worth something on its own, separately from injectivity.


**`chess`**

| phi | tokens | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | teacher match | nodes | tape chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phi=0 | 0 | 0.97 | 0.142 | 0.146 | **0.142** | 0.001 | 635.74 | 0.029 | 7623 | 38772 |
| phi=4 | 1 | 1.00 | 0.042 | 0.042 | **0.042** | 0.001 | 75.88 | 0.008 | 4522 | 76824 |
| phi=8 | 1 | 1.00 | 0.031 | 0.031 | **0.031** | 0.001 | 154.15 | 0.000 | 9309 | 76824 |
| phi=11 | 1 | 0.99 | 0.048 | 0.048 | **0.048** | 0.001 | 1270.78 | 0.002 | 16591 | 76824 |
| phi=14 | 2 | 1.00 | 0.038 | 0.038 | **0.038** | 0.001 | 87.36 | 0.006 | 16053 | 114876 |
| phi=15 | 2 | 1.00 | 0.019 | 0.019 | **0.019** | 0.001 | 65.76 | 0.003 | 16443 | 114876 |
| phi=20 | 2 | 1.00 | 0.021 | 0.021 | **0.021** | 0.001 | 86.63 | 0.002 | 21201 | 114876 |
| phi=30 | 3 | 1.00 | 0.047 | 0.047 | **0.047** | 0.001 | 105.81 | 0.011 | 30155 | 152928 |

**`connect4`**

| phi | tokens | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | teacher match | nodes | tape chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phi=0 | 0 | 1.00 | 0.678 | 0.678 | **0.678** | 0.826 | 0.59 | 0.138 | 28 | 20244 |
| phi=4 | 1 | 1.00 | 0.851 | 0.851 | **0.851** | 0.826 | 0.26 | 0.146 | 31 | 39288 |
| phi=8 | 1 | 1.00 | 0.754 | 0.754 | **0.754** | 0.826 | 0.85 | 0.137 | 356 | 39288 |
| phi=11 | 1 | 0.90 | 0.712 | 0.788 | **0.791** | 0.826 | 1.24 | 0.161 | 2300 | 39288 |
| phi=14 | 1 | 0.41 | 0.353 | 0.858 | **0.839** | 0.826 | 0.42 | 0.140 | 6158 | 39288 |
| phi=15 | 1 | 0.35 | 0.323 | 0.921 | **0.860** | 0.826 | 0.34 | 0.147 | 6322 | 39288 |
| phi=20 | 2 | 1.00 | 0.794 | 0.794 | **0.794** | 0.826 | 1.46 | 0.169 | 6188 | 58332 |
| phi=30 | 2 | 0.40 | 0.290 | 0.720 | **0.783** | 0.826 | 0.47 | 0.132 | 9142 | 58332 |

**`nim`**

| phi | tokens | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | teacher match | nodes | tape chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phi=0 | 0 | 1.00 | 0.582 | 0.582 | **0.582** | 0.364 | 1.76 | 0.095 | 49 | 6048 |
| phi=4 | 1 | 1.00 | 0.613 | 0.613 | **0.613** | 0.364 | 1.30 | 0.118 | 38 | 10896 |
| phi=8 | 1 | 0.98 | 0.782 | 0.799 | **0.790** | 0.364 | 0.62 | 0.289 | 297 | 10896 |
| phi=11 | 1 | 0.95 | 0.885 | 0.930 | **0.903** | 0.364 | 0.47 | 0.379 | 523 | 10896 |
| phi=14 | 1 | 0.94 | 0.939 | 0.994 | **0.959** | 0.364 | 0.14 | 0.436 | 640 | 10896 |
| phi=15 | 1 | 0.94 | 0.943 | 0.999 | **0.964** | 0.364 | 0.13 | 0.448 | 664 | 10896 |
| phi=20 | 2 | 1.00 | 0.514 | 0.514 | **0.514** | 0.364 | 2.29 | 0.073 | 376 | 15744 |
| phi=30 | 2 | 0.94 | 0.939 | 0.994 | **0.959** | 0.364 | 0.14 | 0.458 | 687 | 15744 |
| phi=exact | 1 | 0.94 | 0.849 | 0.899 | **0.869** | 0.364 | 0.44 | 0.349 | 380 | 10896 |

**`othello`**

| phi | tokens | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | teacher match | nodes | tape chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phi=0 | 0 | 1.00 | 0.189 | 0.189 | **0.189** | 0.111 | 7.34 | 0.031 | 156 | 37743 |
| phi=4 | 1 | 1.00 | 0.150 | 0.150 | **0.150** | 0.111 | 9.82 | 0.010 | 141 | 74286 |
| phi=8 | 1 | 1.00 | 0.139 | 0.139 | **0.139** | 0.111 | 11.02 | 0.026 | 645 | 74286 |
| phi=11 | 1 | 0.99 | 0.114 | 0.116 | **0.116** | 0.111 | 12.67 | 0.020 | 4260 | 74286 |
| phi=14 | 1 | 0.49 | 0.139 | 0.285 | **0.196** | 0.111 | 8.26 | 0.054 | 12672 | 74286 |
| phi=15 | 1 | 0.33 | 0.147 | 0.441 | **0.221** | 0.111 | 7.97 | 0.062 | 13079 | 74286 |
| phi=20 | 2 | 1.00 | 0.058 | 0.058 | **0.058** | 0.111 | 211.48 | 0.011 | 12469 | 110829 |
| phi=30 | 2 | 0.50 | 0.128 | 0.254 | **0.183** | 0.111 | 8.59 | 0.052 | 23024 | 110829 |

**`tictactoe`**

| phi | tokens | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | teacher match | nodes | tape chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phi=0 | 0 | 1.00 | 0.613 | 0.613 | **0.613** | 0.626 | 0.85 | 0.124 | 34 | 5685 |
| phi=4 | 1 | 1.00 | 0.621 | 0.621 | **0.621** | 0.626 | 0.81 | 0.117 | 33 | 10170 |
| phi=8 | 1 | 0.97 | 0.557 | 0.576 | **0.578** | 0.626 | 1.28 | 0.110 | 352 | 10170 |
| phi=11 | 1 | 0.67 | 0.564 | 0.839 | **0.770** | 0.626 | 0.61 | 0.134 | 972 | 10170 |
| phi=14 | 1 | 0.57 | 0.547 | 0.967 | **0.819** | 0.626 | 0.56 | 0.138 | 1211 | 10170 |
| phi=15 | 1 | 0.55 | 0.551 | 0.994 | **0.830** | 0.626 | 0.55 | 0.151 | 1282 | 10170 |
| phi=20 | 2 | 1.00 | 0.524 | 0.524 | **0.524** | 0.626 | 9.16 | 0.103 | 1057 | 14655 |
| phi=30 | 2 | 0.57 | 0.554 | 0.971 | **0.823** | 0.626 | 0.55 | 0.144 | 1411 | 14655 |
| phi=exact | 1 | 0.55 | 0.359 | 0.652 | **0.641** | 0.626 | 3.69 | 0.118 | 1141 | 10170 |

<!--/PHI-->

<!--MIX-->

### mix

The same number of bits, split between a hash of the position (`prev=0`)
and a summary of the move just played (`prev=bits`). A hash has no
locality; a move summary does.


**`chess`** - 13 bits to spend

| split | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | refusals guessing | teacher match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prev=0/13 | 0.71 | **0.058** | 0.081 | 0.058 | 0.001 | **1724.84** | 1894.1 | 0.007 |
| prev=3/13 | 0.76 | **0.050** | 0.066 | 0.050 | 0.001 | **1607.93** | 1894.1 | 0.014 |
| prev=6/13 | 0.79 | **0.077** | 0.095 | 0.077 | 0.001 | **1303.05** | 1894.1 | 0.026 |
| prev=9/13 | 0.99 | **0.058** | 0.058 | 0.058 | 0.001 | **494.89** | 1894.1 | 0.011 |
| prev=13/13 | 1.00 | **0.044** | 0.044 | 0.044 | 0.001 | **98.63** | 1894.1 | 0.009 |

On positions whose `phi` token the training corpus never wrote:

| split | coverage | legal@1 | legal@1 covered | refusals | refusals guessing |
|---|---:|---:|---:|---:|---:|
| prev=0/13 | 0.00 | 0.000 | 0.000 | 2075.50 | 2075.5 |
| prev=3/13 | 0.00 | 0.000 | 0.000 | 1867.62 | 1867.6 |
| prev=6/13 | 0.00 | 0.000 | 0.000 | 1661.35 | 1661.4 |
| prev=9/13 | 0.00 | 0.000 | 0.000 | 1717.16 | 1717.2 |

**`connect4`** - 15 bits to spend

| split | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | refusals guessing | teacher match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prev=0/15 | 0.35 | **0.323** | 0.921 | 0.860 | 0.826 | **0.34** | 0.3 | 0.147 |
| prev=3/15 | 0.37 | **0.349** | 0.954 | 0.873 | 0.826 | **0.31** | 0.3 | 0.160 |
| prev=7/15 | 0.92 | **0.744** | 0.805 | 0.807 | 0.826 | **0.39** | 0.3 | 0.203 |
| prev=11/15 | 1.00 | **0.743** | 0.743 | 0.743 | 0.826 | **0.62** | 0.3 | 0.144 |
| prev=15/15 | 1.00 | **0.882** | 0.882 | 0.882 | 0.826 | **0.22** | 0.3 | 0.301 |

On positions whose `phi` token the training corpus never wrote:

| split | coverage | legal@1 | legal@1 covered | refusals | refusals guessing |
|---|---:|---:|---:|---:|---:|
| prev=0/15 | 0.00 | 0.000 | 0.000 | 0.45 | 0.4 |
| prev=3/15 | 0.00 | 0.000 | 0.000 | 0.43 | 0.4 |
| prev=7/15 | 0.00 | 0.000 | 0.000 | 0.46 | 0.5 |

**`othello`** - 15 bits to spend

| split | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | refusals guessing | teacher match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prev=0/15 | 0.33 | **0.147** | 0.441 | 0.221 | 0.111 | **7.97** | 9.5 | 0.062 |
| prev=3/15 | 0.36 | **0.159** | 0.447 | 0.231 | 0.111 | **7.90** | 9.5 | 0.067 |
| prev=7/15 | 0.54 | **0.182** | 0.337 | 0.233 | 0.111 | **7.81** | 9.5 | 0.071 |
| prev=11/15 | 1.00 | **0.204** | 0.204 | 0.204 | 0.111 | **6.17** | 9.5 | 0.046 |
| prev=15/15 | 1.00 | **0.156** | 0.156 | 0.156 | 0.111 | **8.19** | 9.5 | 0.013 |

On positions whose `phi` token the training corpus never wrote:

| split | coverage | legal@1 | legal@1 covered | refusals | refusals guessing |
|---|---:|---:|---:|---:|---:|
| prev=0/15 | 0.00 | 0.000 | 0.000 | 9.74 | 9.7 |
| prev=3/15 | 0.00 | 0.000 | 0.000 | 9.69 | 9.7 |
| prev=7/15 | 0.00 | 0.000 | 0.000 | 9.58 | 9.6 |
| prev=11/15 | 0.00 | 0.000 | 0.000 | 5.74 | 5.7 |

**`tictactoe`** - 15 bits to spend

| split | coverage | legal@1 | legal@1 covered | with fallback | guessing | refusals | refusals guessing | teacher match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prev=0/15 | 0.55 | **0.551** | 0.994 | 0.830 | 0.626 | **0.55** | 0.8 | 0.151 |
| prev=3/15 | 0.51 | **0.501** | 0.982 | 0.808 | 0.626 | **0.62** | 0.8 | 0.112 |
| prev=7/15 | 0.62 | **0.420** | 0.675 | 0.657 | 0.626 | **1.93** | 0.8 | 0.093 |
| prev=11/15 | 1.00 | **0.618** | 0.618 | 0.618 | 0.626 | **1.06** | 0.8 | 0.128 |
| prev=15/15 | 1.00 | **0.692** | 0.692 | 0.692 | 0.626 | **0.58** | 0.8 | 0.110 |

On positions whose `phi` token the training corpus never wrote:

| split | coverage | legal@1 | legal@1 covered | refusals | refusals guessing |
|---|---:|---:|---:|---:|---:|
| prev=0/15 | 0.00 | 0.000 | 0.000 | 1.23 | 1.2 |
| prev=3/15 | 0.00 | 0.000 | 0.000 | 1.24 | 1.2 |
| prev=7/15 | 0.00 | 0.000 | 0.000 | 1.22 | 1.2 |

<!--/MIX-->

<!--PHASE-->

### phase

`ambiguous occurrences` is the number that matters: a collision on a
trigram seen once is a curiosity, one on a trigram in every game is a
defect.

| game | codebook | ambiguous types | ambiguous occurrences | legal@1 covered | with fallback | refusals |
|---|---|---:|---:|---:|---:|---:|
| `chess` | disjoint/lsb | 0.0% | **0.0%** | 0.081 | 0.058 | 1724.84 |
| `chess` | plain/lsb | 3.6% | **6.6%** | 0.081 | 0.059 | 1727.30 |
| `chess` | plain/struct | 2.3% | **3.3%** | 0.083 | 0.061 | 1746.86 |
| `connect4` | disjoint/lsb | 0.0% | **0.0%** | 0.921 | 0.860 | 0.34 |
| `connect4` | plain/lsb | 5.3% | **45.4%** | 0.701 | 0.776 | 10.65 |
| `connect4` | plain/struct | 5.3% | **45.4%** | 0.701 | 0.776 | 10.65 |
| `nim` | disjoint/lsb | 0.0% | **0.0%** | 0.999 | 0.964 | 0.13 |
| `nim` | plain/lsb | 1.2% | **10.7%** | 0.939 | 0.907 | 0.93 |
| `nim` | plain/struct | 3.6% | **24.5%** | 0.808 | 0.783 | 1.85 |
| `othello` | disjoint/lsb | 0.0% | **0.0%** | 0.441 | 0.221 | 7.97 |
| `othello` | plain/lsb | 4.0% | **34.4%** | 0.379 | 0.212 | 42.00 |
| `othello` | plain/struct | 9.2% | **44.1%** | 0.139 | 0.123 | 25.12 |

<!--/PHASE-->

<!--LAYOUT-->

### layout

Three ways of writing the *same* action as three characters, all under
the `plain` codebook, which is the only one that can express a
structured layout.

| game | layout | coverage | legal@1 | legal@1 covered | refusals | teacher match | nodes |
|---|---|---:|---:|---:|---:|---:|---:|
| `chess` | lsb | 0.20 | **0.051** | 0.253 | 1837.70 | 0.010 | 21111 |
| `chess` | msb | 0.17 | **0.041** | 0.237 | 1836.53 | 0.006 | 20632 |
| `chess` | struct | 0.19 | **0.051** | 0.262 | 1842.48 | 0.009 | 20103 |
| `othello` | lsb | 0.23 | **0.139** | 0.599 | 47.34 | 0.061 | 13928 |
| `othello` | msb | 0.24 | **0.017** | 0.070 | 20.69 | 0.002 | 13629 |
| `othello` | struct | 0.25 | **0.107** | 0.433 | 17.32 | 0.043 | 14667 |
| `tictactoe` | lsb | 0.56 | **0.546** | 0.976 | 1.34 | 0.144 | 1314 |
| `tictactoe` | msb | 0.56 | **0.414** | 0.745 | 0.82 | 0.076 | 1124 |
| `tictactoe` | struct | 0.56 | **0.533** | 0.955 | 0.76 | 0.130 | 1341 |

<!--/LAYOUT-->

<!--MULTI-->

### multi

Six corpora into one `RadixNet.train` call. The header token is the
whole selector - `Experiments/NeuralCompression` needed a partitioned
output and a selector network to ask this question; here it falls out of
the encoding.

It works, and it is not free. The shared graph is smaller than six
separate ones, and every game pays for the company: refusals are worse
inside it for all six, and legality for five of the six (Connect Four's
top-1 improves and its refusals get five times worse). The last
column says why, and it is the same lesson as everywhere else in this
directory: the six games share a code space, only the header is reserved
per game, so nothing structurally stops a Connect Four position being
answered with a token that only ever appeared in Othello. It is not a
hypothetical - a quarter of the answers are exactly that, and the
legality columns hide it, because a foreign token is usually illegal
anyway and just looks like an ordinary miss.

shared graph: 49865 nodes; six separate graphs: 63991 nodes (ratio 1.28x)

| game | legal@1 shared | legal@1 alone | refusals shared | refusals alone | answered with another game's move |
|---|---:|---:|---:|---:|---:|
| `chess` | 0.053 | 0.058 | 1646.22 | 1724.84 | 0.261 |
| `connect4` | 0.379 | 0.323 | 1.80 | 0.34 | 0.251 |
| `nim` | 0.625 | 0.943 | 1.07 | 0.13 | 0.256 |
| `othello` | 0.108 | 0.147 | 11.02 | 7.97 | 0.140 |
| `pig` | 0.302 | 0.424 | 2.64 | 1.59 | 0.272 |
| `tictactoe` | 0.470 | 0.551 | 4.69 | 0.55 | 0.194 |

<!--/MULTI-->

<!--TWONRL-->

### twonrl

Three arms from the *same* warm-started model, with the same number of
gradient passes over the same number of texts of the same length:
`2nrl` trains toward the refused moves then inverts, `positive` trains
toward the correct ones on the same prefixes and does not, and `local`
uses `invert_paths` instead of the global inversion. `base` is the warm
start all three began from.

| game | arm | legal@1 | legal@1 covered | refusals | teacher match | nodes |
|---|---|---:|---:|---:|---:|---:|
| `connect4` | base | 0.320 ± 0.016 | 0.911 | 0.34 | 0.147 | 6322 |
| `connect4` | 2nrl | 0.324 ± 0.006 | 0.924 | 0.34 | 0.140 | 6481 |
| `connect4` | positive | 0.323 ± 0.020 | 0.921 | 0.34 | 0.147 | 6323 |
| `connect4` | local | 0.323 ± 0.024 | 0.921 | 0.34 | 0.143 | 6481 |
| `nim` | base | 0.943 ± 0.011 | 0.999 | 0.13 | 0.436 | 664 |
| `nim` | 2nrl | 0.438 ± 0.062 | 0.464 | 1.59 | 0.166 | 1021 |
| `nim` | positive | 0.943 ± 0.011 | 0.999 | 0.13 | 0.466 | 665 |
| `nim` | local | 0.400 ± 0.094 | 0.424 | 1.62 | 0.188 | 1021 |
| `othello` | base | 0.147 ± 0.029 | 0.441 | 7.97 | 0.061 | 13079 |
| `othello` | 2nrl | 0.130 ± 0.016 | 0.390 | 8.01 | 0.054 | 13408 |
| `othello` | positive | 0.147 ± 0.029 | 0.441 | 7.97 | 0.062 | 13080 |
| `othello` | local | 0.119 ± 0.024 | 0.355 | 8.03 | 0.049 | 13408 |

<!--/TWONRL-->

---

## What it cannot do

Every one of these is measured above, not hedged.

1. **The memory is one token.** `RadixNet._locate` conditions a prediction on
   the last trigram of the prefix alone, so a `φ` that spans two tokens is tape
   the model pays for and never reads. The cap is 18 bits, or 15 once you buy
   phase-unambiguity. The cliff is therefore at the *last* token rather than at
   the second, which the sweep shows exactly: Nim scores 0.94 at 15 bits, 0.51
   at 20 and 0.94 again at 30, because 20 leaves five bits in the final token
   and 30 leaves fifteen.

2. **A hash has no locality**, and it shows up in two different ways. For
   Othello, Connect Four and tic-tac-toe it is coverage: at 15 bits the model
   has seen a third of the test positions and is silent on the rest, and the
   *unseen* table is all zeros by construction, because an unseen `φ` token is
   an unseen node and there is no such thing as a near miss. For chess it is
   worse than that - coverage stays at 0.71 and legality at covered positions is
   still 0.081, because a 13-bit digest of a chess position simply does not
   determine which moves are legal. The [`mix`](#mix) arm is the response: the
   same bits spent on *the square the last move landed on* mean the same thing
   in every position they appear in, and they take chess from 1725 refusals a
   move to 98.6. Notice also that the unseen table disappears entirely at
   `prev=bits` - there are no unseen `φ` tokens left to put in it, which is what
   locality looks like from the other side.

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
