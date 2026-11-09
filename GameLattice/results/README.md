# Results

What `make experiments` wrote, read against the numbers `DESIGN.md` §3 and §17 say must be beaten.  Three seeds
each; `e1.log` and `e2.log` are the runs' output line by line.

## E1 - does the stage game earn its cost? (`e1.json`)

The four languages, 4 000 episodes, a 13-state lattice, held-out accuracy on 300 strings, the mean over the
languages and three seeds.  `encoder` is the verdict of the encoder's final named state; `lattice` the
verdict of the lattice's own final prediction.

| arm | encoder | lattice |
|---|---|---|
| plain-oracle (the lattice alone, credited per step; the control) | 0.961 | 0.961 |
| game (the encoder names the state the lattice continues from) | 0.961 | 0.958 |
| game, credit baseline k = 0.5 | 0.971 | 0.948 |
| game, credit baseline k = 1 | 0.964 | 0.959 |
| game, rewards only | **0.988** | 0.963 |
| game, the whole run credited at its end | 0.980 | 0.659 |
| game, forward lessons on the lattice | 0.969 | **0.972** |

The numbers to beat (`LatticeFSM/README.md`, "Credit by an oracle", the lattice alone): whole-run 0.894,
oracle per step 0.924, explore 0.944.

What it says.  The game arm **matches** the plain oracle lattice (0.961 against 0.961) and does not beat it:
`DESIGN.md` §21.2 is live, and the author decides whether matching at the stage game's per-symbol cost keeps
the GameRadix encoder as the architecture.  Both clear every LatticeFSM mark, which is the per-step credit's
doing (the plain arm is LatticeFSM's own oracle rule wired through `credit::land`).  Two controls went the other
way from the LatticeFSM measurement: rewards only was the best arm here (0.988; LatticeFSM measured it worse in
every arm), and a whole-run credit left the lattice at 0.659 while the encoder, credited per step by its own
game, still reached 0.980 - the encoder's verdict does not depend on the lattice's credit.  The forward
lessons (§8.2) raised the lattice to 0.972 and cost the encoder nothing.  Agreement between the lattice's
prediction and the encoder's naming sits at 0.90-1.00 in the game arms; diversity 1.6-3.4 bits over 13 states.
Per language and seed the game arm ranges 0.79-1.00 (contains-aa seed 3 is the 0.79); the plain arm
0.857-1.00.

## E2 - is the walk window a memory? (`e2.json`)

Nim on heaps of 3, 5 and 7, a 13-state act lattice over the 21 moves, 200 babbled games then 400 against a
random opponent, by window width; the mean over three seeds.

| W | legal first try | refusals a move | wins | wins, second half | optimal moves | I(state; nim-sum) bits | states used | agreement |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.13 | 0.70 | 0.50 | 0.52 | 0.40 | 0.029 | 13 | 0.87 |
| 4 | 0.16 | 1.19 | 0.50 | 0.53 | 0.39 | 0.026 | 13 | 0.76 |
| 8 | 0.21 | 1.14 | 0.47 | 0.47 | 0.39 | 0.031 | 13 | 0.75 |
| 16 | 0.21 | 1.30 | 0.49 | 0.50 | 0.39 | 0.027 | 13 | 0.75 |

What it says: **contradicted, as built.**  A random opponent wins half its games and a random move is optimal
about 0.4 of the time from these positions, and the body does no better on either; the named state carries
under 0.04 bits about the position's nim-sum, so the encoder is not reading the position off the tape; first-try
legality rises with the window (0.13 to 0.21) but a move is refused more than once on average.  Two things the
first build lacks explain it and are the next work: the repertoires seat the act lattice's last edges, whose
four random symbols rarely cover the legal moves of a position (the referee's refusals teach them one at a
time), and the encoder's game in a game has only the forward model's prediction as its label (`DESIGN.md`
§4.4), which carries nothing about the position until the forward model does.  The local digest of §4.5 (the
heaps as a feature) is the designed answer and is not built.
