# Memory — the trajectory stack

> Solve memory by keeping track of the trajectory through the model, input and
> output data, then storing this in a new NN that is a stack-based idea where the
> top of the stack has precedence compared to similar paths further down.

**Status: specified, not built.** This file is the task, not the implementation.

## The idea

Every other network in this repository throws away the most informative thing it
produces. GREN walks a radix path to identify a game and keeps only the answer.
CyclicCortex routes to a region and keeps only the prediction. GTMNN seats 64
micros out of 4096, solves an equilibrium and keeps only the aggregate — the
*identity of who was seated and what they argued* is discarded at the end of
every stage game.

The trajectory is that discarded thing: **the path the model took**, not the
answer it arrived at. Memory here means storing `(trajectory, input, output)` and
being able to ask "have I been down a path like this before, and what happened?"

The store is a **stack**. New entries are pushed on top. A lookup walks down from
the top and the first sufficiently-similar path wins, so **recency shadows
history** — a path taken a moment ago outranks a similar path taken a thousand
episodes ago, without anything having to be deleted or decayed.

## Why the precedence order is the interesting part

`GREN/gren/radix.py` resolves ambiguity by **specificity**: a deeper terminal
node beats a shallower one, because "chess" is a better answer than "a
two-player board game" when both match.

This resolves ambiguity by **recency**: the top of the stack beats everything
below it, whatever their specificity.

Those are two different precedence orders over the same kind of object — a set of
matching paths — and they disagree. The specific-but-stale entry and the
general-but-fresh one are both real answers. Which one should win is an empirical
question and the repository already has the instrument to settle it, because both
policies can be run over the same stack.

This is also exactly a scope chain: an inner binding shadows an outer one without
erasing it, and popping restores what was underneath. A stack gives *undo* for
free, which no weight-based memory does.

## What it would connect to

* **Catastrophic forgetting.** `CyclicCortex/cortex/cortex.py::rehearse` exists
  because training a newly-admitted game overwrites the incumbents, and replay is
  the mitigation. A trajectory stack attacks the same problem from the other
  side: instead of replaying old data through the weights, keep the old
  *trajectories* and consult them. `rehearse` is the baseline to beat, and it is
  already measured.
* **Consolidation, insight 22.** The dual-network / REM idea — train in one
  network, query the other — has never had a mechanism for deciding *what* to
  consolidate. A stack supplies one: entries that keep getting shadowed are what
  the base network should absorb; entries that keep getting hit on their own are
  what it still cannot represent.
* **The trie scope rule.** A path-keyed store is structurally trie-shaped, and a
  trie in this repository is for game theory and game classification only. That
  tension is deliberate and unresolved: either this uses a different structure,
  or the rule needs a stated exception. It should not quietly become a third
  trie.

## What is genuinely unspecified

These are the decisions, not details. Each changes what gets built.

1. **What IS a trajectory?** It has to mean something in all three networks at
   once — GREN's radix path, CyclicCortex's region and vocabulary slots, GTMNN's
   seated micro ids plus their actions plus the trie node. Variable length, three
   different alphabets. Without one encoding there is no shared memory, only
   three private ones.
2. **What does "similar" mean for two paths?** Shared prefix length, Jaccard over
   visited nodes, edit distance, or a learned embedding. This is the crux: the
   whole design rests on "similar paths further down", and that phrase is doing
   an enormous amount of work.
3. **One stack, or a stack per prefix?** "Walk down from the top and take the
   first match" is one global stack and is O(depth) per lookup. A stack at each
   node of a path index is O(1) but changes the semantics — recency would then be
   scoped to a branch rather than global.
4. **Growth.** A stack that only grows is not a memory, it is a log. Eviction,
   consolidation into the base network, or both.
5. **Is the trajectory worth more than the input that produced it?** If not, this
   reduces to an ordinary input-keyed cache with extra steps.

## The first experiment

Question 5 is the cheapest and it can falsify the premise, so it goes first.

Take CyclicCortex, which already has four games, trained regions and a measured
baseline. For each position, record the input features, the trajectory (region
id, vocabulary slots written, the SBNN's hidden activations' argmax pattern) and
the outcome. Then build two lookup tables over the same episodes:

* keyed by **input** — an ordinary nearest-neighbour cache
* keyed by **trajectory** — the thing this design claims is better

and measure which predicts the held-out outcome more accurately.

**If trajectory-keyed does not beat input-keyed, the premise is wrong** and the
cost was an afternoon. If it does, the margin says how much structure the path
carries that the input does not, and that number is what justifies building the
rest.

Run it against the repository's own hard-won null results first: chess→checkers
transfer has now measured at approximately nothing three independent ways
(insights 29, 30, 32), so "similar paths generalise" is a claim this repository
has reason to be sceptical of, and the experiment should be built to fail loudly.
