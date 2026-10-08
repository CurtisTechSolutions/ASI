# frontend

The browser page for CyclicCortex. **One page, no build step, no framework** —
three files, served as static assets by `cortex/server.py`.

The server does every piece of thinking. Each button here is a call the command
line already makes: `/api/jobs` runs `Cortex.train`, `Cortex.selfplay` and
`Cortex.distill`; `/api/match` runs `cortex/arena.py`; `/api/model` reports
`Cortex.stats` and the similarity graph. What the browser adds is the part that
only works live — `p_valid` and `grade` for **every legal move** while it is
your turn, and the learning curve while it is still learning.

## Contents

| file | what it is |
|---|---|
| `index.html` | the page: four numbered tabs — the cortex, training, play, credit |
| `app.js` | all of the behaviour — the API calls, the multidimensional scaling that lays the map out, the SVG boards for all four games, the heat map, the learning curve, the Shapley bars |
| `style.css` | all of the styling, dark, one colour scale shared by the matrix and the board |

## Opening it

```bash
cd CyclicCortex
python3 -m cortex.cli serve --train 300      # train each game first, then serve
python3 -m cortex.cli serve                  # or serve an untrained cortex
```

Then open the address it prints (`127.0.0.1:8099` by default). There is nothing
to install or build — the files are served exactly as they are.

`--host` has to be given deliberately: training is unauthenticated CPU on
demand, so binding this to a public interface hands the machine to whoever can
reach the port.

## What each tab is for

**1 · The cortex.** The similarity graph as a picture rather than a table. The
map places the games by their distances *alone* — classical scaling of the
matrix, then stress majorisation, which on the four games here halves the
distance error and separates chess from checkers (0.45 apart instead of 0.06).
A node's size is its region's parameter count and the game that arrived last
pulses. The residual error is printed under the map: a plane cannot hold four
Jaccard distances exactly, the matrix is exact, and the drawing is a
**projection** — `DESIGN.md` §11 — so routing and credit are never computed
from it. Beside it: each region's vocabulary drawn as its slot layout, its
growth log, and how every game was routed.

**2 · Training.** The same runs the CLI does, chunked, with an evaluation after
each chunk. That is not only a progress bar: the README reports that 400, 1200
and 3000 episodes of chess all land near 0.80, which is a claim about a *curve*
made from three end points. This draws the curve, and it can be stopped.

**3 · Play.** Four kinds of player — human, cortex, the built-in engine, and
Stockfish where it is installed — and every pairing of them, in all four games.
So "human vs model", "model vs model" and "bot vs model" are not three features
but three rows of the same table; the chips at the top set a pairing in one
click. Two things are shown that a result cannot say:

* The cortex's pick is **not filtered to legal moves**. It ranks its own
  candidate set exactly as `Cortex.choose` does in `cli.play`, an illegal pick
  is counted, and the best legal alternative is substituted so the game goes
  on. The illegal rate under the board is what the network learned; hiding the
  substitution would make every cortex look perfect.
* Every legal move's `p_valid` and `grade` are on the board as a heat map and in
  the table beside it. A result says the cortex lost. The scores say it wanted
  to play a move that hangs a rook.

A region that has seen nothing says so, and offers to train itself: ranking
moves from a random initialisation is not a weak model but no model, and a page
that quietly let you play one would be showing you noise.

**4 · Credit.** Coverage, exact Shapley over regions, the auction, and the
cross-region transfer table — the numbers the README reports, recomputed on the
model that is loaded rather than quoted.

## Two deliberate divergences from `DESIGN.md` §11

*No React.* The spec names `CortexMap.jsx`. The package has no dependencies and
a front end is not the place to acquire a toolchain, so the map is inline SVG
built by `app.js`. Everything §11 asks for is here — the embedded layout, the
region colouring, the node sized by parameter count, the arrival of a new game,
the region panel, the growth log and the vocabulary view.

*Scaling, not springs.* §11 says force-directed. A spring layout answers "what
looks tidy"; classical scaling plus stress majorisation answers "what are the
distances", which is the question the graph is making a claim about — and it
lets the page print how wrong the drawing is.
