# ASI

## READ THE LICENSE!!!

You automatically agree to the license if you visit this repository, or to whatever extent is legally allowable in a court of law.

The license is to protect my IP. This is source-available and to prove I'm the original inventor and author of these solutions. For more information, view the [LICENSE](LICENSE)

## The gist

I put in months of continuous extreme effort to research AGI/ASI, and this is the culmination of that work. Overall, I solved multiple different variations of AGI/ASI, all derived from my research on the human brain, and potential variations to the design of the human brain. This research diverges from the norm significantly, in that I refuse all current research and reinvent the wheel.

All rights reserved; all research is 100% my own, and none of this repo is allowed to be used for profit, unless used by my own company or myself [My company (CTS)](https://www.curtistechsolutions.com)

## Details

It all started when I was frustrated with the current AGI/LLM hype. I was building a company/product called Amy AI (Executive Email Assistant). I grew extremely frustrated with how much AI was parroted as the "Next big thing", and "AI *this*", "AI *that*". I was looking for hard problems to solve at the time, so I said: "screw it, I'll do it myself!". So, that's what I did! I reverse-engineered my entire brain, from memory to executive function, delved into neuroscience, and more. Overall, I have many pages of notes, scribbles, diagrams, methodologies, and more. I've also optimized the training process with my own proprietary custom training algorithms.

## Running it

Everything is driven from one Makefile at the root. `make help` lists every
target, grouped by project:

```bash
make help          # every target
make deps          # what the optional dependencies are, and whether they are here
make test          # every suite that needs nothing installed
make handoff       # GREN probes every game, then CyclicCortex rebuilds its map from it
make demo          # the CyclicCortex tour: train, play chess, add sudoku
```

Most of the repository is pure standard-library `python3` and runs with nothing
installed. `make deps` says which of the remaining pieces need `numpy`,
`gymnasium`, `torch`, Stockfish or Go, and which targets those are.

Three directories carry their own Makefile for their own commands — `AudioImage`,
`RadixCyclicNN` and `TwoNRL_CartPole`; the root delegates to them rather than
repeating them, so `make audioimage` lists AudioImage's targets.

Override any variable on the command line:

```bash
make handoff BUDGET=3000          # more probes per game
make demo EPISODES=1200 SEED=7
make map GAMES="chess checkers"
```
