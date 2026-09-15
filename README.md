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

Every project carries its own Makefile, and one at the root ties them together.
`make help` lists every target, at either level:

```bash
make help          # every root target
make deps          # what the optional dependencies are, and whether they are here
make test-quick    # GREN + GTMNN + CyclicCortex
make test          # every suite that needs nothing installed
make handoff       # GREN probes every game, then CyclicCortex rebuilds its map from it
make demo          # the CyclicCortex tour: train, play chess, add sudoku
```

Most of the repository is pure standard-library `python3` and runs with nothing
installed. `make deps` says which of the remaining pieces need `numpy`,
`gymnasium`, `torch`, Stockfish or Go, and which targets those are.

The root delegates with `make -C` rather than repeating commands, so each one
has a single home. To see what a project offers, ask it:

```bash
make gren          # GREN's targets — explore, similar, policies, tree, grow, package
make gtmnn         # GTMNN's — demo, shapley, cycles, auction, modifier, transfer
make cortex        # CyclicCortex's — demo, map, play, sudoku, transfer, credit
make compression   # NeuralCompression's four experiments
make audioimage    # AudioImage's
make radixcyclic   # RadixCyclicNN's
make cartpole      # TwoNRL_CartPole's
```

Or work inside one directly — `cd GREN && make test` does what you would expect.

Variables set on the command line flow down into the project Makefiles, and
their defaults live with them rather than at the root:

```bash
make handoff BUDGET=3000              # more probes per game, reaches GREN
make demo EPISODES=1200 SEED=7        # reaches CyclicCortex
make -C CyclicCortex map DISCOVERED=1 # route on GREN's measurement, not the hand-written sets
```
