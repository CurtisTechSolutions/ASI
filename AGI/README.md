# AGI — the networks that compose

Everything beside this directory is an independent line of work. These five are
one system, and they are here together because they **cross-import and pass one
another a handoff file** — not because they are thematically similar.

```
GREN          identify the game, by what it REFUSES
  |           probes an unknown game, records the refusal codes, and emits a
  |           GamePackage: a signature, a confidence, a goal type
  |
  |  gren_packages.json  <- the handoff. A file, never an import: GREN imports
  |                        the game adapters in order to probe them, so the
  v                        reverse direction would close a cycle.
  |
  +--> CyclicCortex     route it to a REGION of a cyclic similarity graph, where
  |                     distance is similarity, and play it with one
  |                     Self-Building network per region
  |
  +--> GTMNN            play it as a POPULATION: micros bid for seats, the
                        seated ones play a congestion game, and credit is the
                        Shapley value. The game modifier hashes the same
                        mechanic set, so all three agree on what a game IS
                        through one artifact rather than three hand-written
                        copies of the same frozenset.

Memory              the trajectory stack — specified, not built. Keyed on the
                    path through the model rather than the answer.

NeuralCompression   the substrate the others stand on: whether one network can
                    hold many games, and what the vanishing gradient is good
                    for. GTMNN's activation frequency comes from here.
```

## Reading order

`Research/Insights.md` is the index — 39 entries, each marked **held**,
**confirmed**, **refined** or **contradicted**, with the number behind it. Start
there rather than here, because several of the load-bearing claims did not
survive measurement and the index says which.

Then, in dependency order: `GREN/README.md`, `CyclicCortex/README.md`,
`GTMNN/README.md`.

## What is actually true, in one table

The honest summary, since three of these are negative and the negatives are the
more useful half.

| claim | verdict |
|---|---|
| A game can be identified by what it refuses | **holds** — the discovered signature reproduces the hand-written regions exactly |
| Credit can be assigned without backpropagation | **holds** — Shapley efficiency exact to 3e-15, per sample, not in expectation |
| Similar games share structure that transfers | **fails as measured** — every feature-based measure ranks chess/checkers most similar and go/sudoku near least; rule transfer is 0.32 and **0.98** respectively. A rule transfers exactly when the feature carrying it means the same thing, and mechanic *names* do not say whether it does |
| A population playing a game learns to predict | **holds, with a gap** — real inference loss 2.35 against uniform 2.71, improving monotonically; the same micros supervised reach 0.91, because only seats holding `y` ever receive credit |
| Index a game by its END GOAL and work backwards | **holds** — one token to identify against alphabetical's 2.5, and it is the one thing you can state about an unfamiliar game before playing it |
| Goal regression needs hand-written operators | **fails, usefully** — a refusal code *is* a precondition violation, so GREN already had them |

## Running it

From the repository root: `make gren`, `make cortex`, `make compression` list
each project's targets, and `make handoff` runs the GREN → CyclicCortex pipeline
end to end. Or work inside one directly — `cd AGI/GREN && make test`.
