# GREN

**G**ame **R**ule **E**ncoder **N**etwork — part one of a two-part architecture.

GREN does not play. GREN works out **which game is being played**, by guessing,
being told no, and treating every *no* as the lesson. When it knows, it hands a
finished rule package to `GTMNN/`, which plays the game it is handed.

> Once you understand the game, then and only then can you play the game.

The job is **identification by similarity**, not induction from nothing. GREN
holds a corpus of games it already knows and asks *"which of these does this new
thing refuse like?"* — a nearest-neighbour query over partial evidence, refined
by probes chosen to separate the candidates still standing. Learning a rule set
from zero is the degenerate case where nothing in the corpus is close, and it
costs three to four orders of magnitude more.

## Contents

| file | what it is |
|---|---|
| `DESIGN.md` | the full specification — the contract every module is implemented against, in 32 sections |

**There is no code in this directory yet.** `DESIGN.md` is the whole of it: a
spec written to be implemented against, not notes. Read it fully before writing
anything here.

## What the spec covers

| § | what it settles |
|---|---|
| 1 | every requirement in the author's own words, mapped to the section that realises it |
| 2-3 | the claim (a game is identified by what it refuses) and the retrieval task built on it |
| 4 | why failing on purpose is optimal, and why the target failure rate is exactly **50%** — it falls out of the information-gain objective rather than being set as a knob |
| 5 | what is identifiable and what is only *placeable*; why programming is the domain worth optimising for |
| 7-8 | the package layout and the decomposition into smallest parts |
| 9 | the axis catalogue — the dimensions of game-space |
| 10-18 | the modules: `verdict`, `action`, `oracle`, `sandbox`, `features`, `reasons`, `rule`, `probe`, `axis` |
| 19-21 | `radix.py` (a radix tree with the mechanics of a trie), `forest.py`, `similarity.py` |
| 22 | `package.py` — the `GamePackage` handed to GTMNN, and the two-head micro (`p_valid` from GREN, `grade` from GTMNN) |
| 23-28 | `explorer`, `checkpoint`, `bench`, `cli`, `api`, `frontend/`, tests |
| 29-32 | performance, other ways to solve this, cross-cutting invariants, open questions |

## Planned shape

Python package `gren`, Python 3.11+, **standard library only** — the same rule
`RadixCyclicNN/` and `AudioImage/` follow. CLI, stdlib HTTP JSON API and a
Vite + React frontend, so it stays a recognisable sibling of the other projects.

## Related

* `GTMNN/DESIGN.md` — part two, which plays the game GREN identifies. It refuses
  a package below a confidence threshold, so the ordering is enforced rather
  than suggested.
* `RadixCyclicNN/DESIGN.md` §5.2 — the split and merge rules GREN's radix tree
  follows exactly.
* `Research/CyclesAreAFeature.md` — the structural argument underneath both.
