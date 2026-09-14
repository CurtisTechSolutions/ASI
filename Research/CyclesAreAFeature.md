# Cycles Are a Feature

**The brain is a directed cyclic graph, and the loops are the point**

Mason Curtis — Curtis Tech Solutions
Working paper · September 2026

---

## Abstract

I originally proposed that the brain is a directed acyclic graph. I was wrong.
The brain is a directed *cyclic* graph, and the cycles are not damage to be
routed around — they are what makes the structure finite, what makes
generalisation free, and what creates the need for a supervisory level at all.

A cycle is repetition stored once. A graph with a loop in it represents an
infinite family of sequences in a fixed amount of space; the acyclic version of
the same data grows without bound. That is the benefit. The cost is that a
cyclic system can fail to halt, has no layer order, cannot use negative costs,
and — at the level of a mind — can ruminate.

The resolution is the part of my original note that needs the most unpacking:
*when we encounter a cycle, we use metacognition or another part of the brain
instead.* This paper makes that precise. A system inside a loop has, by
definition, no information inside the loop that tells it to leave — re-asking
the same question from the same place only returns the same answer. The exit
must come from outside, and there are exactly three places it can come from: a
clock, a monitor, or an inversion. All three are implemented in `RadixCyclicNN/`
and I walk through each one in the code, including the piece that notices a loop
from the shape of its own output, backs up to the point the walk went round, and
searches again from past it — which the implementation calls, without any
prompting from this paper, "the metacognition of a turn".

---

## 1. The original note

> I originally proposed that the brain is a directed acyclic graph. However, I
> believe the brain is actually a directed cyclic graph. Where cycles are a
> feature, not a bug. When we encounter a cycle, we use metacognition or another
> part of the brain instead.

Everything below is what I meant.

---

## 2. Why I changed my mind

The DAG was a convenience, and I should be honest that it was a convenience
borrowed from engineering rather than a conclusion drawn from the brain.

A directed acyclic graph is an extremely comfortable object. It has a
topological order, so you can evaluate it in one pass. It terminates for free.
It has clean layers. Back-propagation is defined on it without any special
handling. Every mainstream network architecture is a DAG for exactly these
reasons — not because anyone examined a brain and found one.

Then look at what a brain actually is. Feedback projections in cortex are not a
minor correction term; descending and lateral connectivity is abundant, in many
areas outnumbering the ascending fibres. Cortex and thalamus are wired in closed
loops. Cortex, basal ganglia and thalamus form closed loops. Local circuits are
densely recurrent. There is no direction you can point in a brain and say
"information goes this way and does not come back".

A model that forbids cycles is not a simplification of that. It is a different
object. And the specific things it cannot do are not edge cases:

| A brain does this | A DAG cannot |
|---|---|
| hold a thought (working memory) | hold anything; the wave passes through and is gone |
| repeat an action arbitrarily many times | repeat without a distinct node per repetition |
| think about its own thinking | ever return to a node — self-reference is a cycle by definition |
| get stuck | get stuck — which sounds like a feature until you notice that it also cannot *notice* being stuck |

That last row is the one that changed my mind. A DAG cannot ruminate, and
therefore a DAG has nothing for metacognition to supervise. If you build a mind
that cannot possibly loop, you have also built a mind that has no use for
self-monitoring. But self-monitoring is obviously there in a human, and it is
obviously *about something*. It is about loops.

So the cycles are not a defect I am tolerating. They are the thing the higher
level exists to manage.

---

## 3. What a cycle buys: repetition stored once

Here is the whole thesis in four characters.

`RadixCyclicNN` encodes text as a sliding window of three characters. The string
`"aaaa"` becomes the trigrams `["aaa", "aaa"]` — the same trigram twice. Since a
trigram lives in exactly one node, the second occurrence does not make a new
node. It makes an **edge from the node back to itself**:

```
   START ──▶ ┌─────┐ ──▶ END
             │ aaa │◀─┐
             └─────┘  │
                └─────┘   self-loop
```

From `RadixCyclicNN/tests/test_graph.py`:

```python
def test_self_loop(self):
    g = RadixCyclicGraph(seed=4)
    trans = g.observe_sequence(ENC.encode("aaaa"))
    n, _ = g.lookup("aaa")
    self.assertIn(n, g.children[n])          # the node is its own child
    self.assertEqual(len(trans), 3)          # START->aaa, aaa->aaa, aaa->END
    self.assertEqual(g.count[n], 2)          # visited twice
    self.assertEqual(g.compress(), 0)        # a self-loop is never merged
    self.assertEqual(round_trip(g, "aaaa"), "aaaa")
    self.assertEqual(round_trip(g, "aaaaaaa"), "aaaaaaa")
```

Read the last line carefully. **The graph was shown `"aaaa"` and it represents
`"aaaaaaa"`.** It was never shown seven a's. It does not contain seven a's
anywhere. It contains one node and one loop, and the loop *is* the statement
"this can repeat".

That is generalisation, and it did not come from a training procedure, a
regulariser, or a prior. It came from the structure. The moment you allow an
edge to point backwards, seeing a thing twice becomes a claim about seeing it
any number of times.

### The same thing one step up

`"abcabc"` gives the trigrams `abc, bca, cab, abc`. After path compression the
graph is two nodes:

```
            ┌──────────┐
            ▼          │
   START ──▶ abc ──▶ bcab
             │
             ▼
            END
```

```python
labels = sorted(g.labels[i] for i in g.alive_nodes() if i > END)
self.assertEqual(labels, ["abc", "bcab"])
self.assertIn(bcab, g.children[abc])
self.assertIn(abc, g.children[bcab])       # <- the cycle
self.assertEqual(round_trip(g, "abcabcabc"), "abcabcabc")
```

Two nodes, one 2-cycle, and the structure covers `"abcabc"`, `"abcabcabc"`, and
every longer repetition of the same motif.

### The counting argument

This is the part that makes it more than a nice trick.

To represent `n` repeated a's in an **acyclic** graph of trigram nodes, you need
`n − 2` distinct nodes, because every node can be entered at most once on the
path and each one emits one character. The structure grows linearly with how
long the repetition happens to be.

In a **cyclic** graph you need one node, regardless of `n`.

| String | DAG nodes | Cyclic graph nodes |
|---|---:|---:|
| `aaaa` | 2 | 1 |
| `aaaaaaaaaa` | 8 | 1 |
| `a` × 1000 | 998 | 1 |
| `a` × ∞ | impossible | 1 |

The last row is the important one. A finite acyclic graph cannot represent an
unbounded language. A finite cyclic graph can, and it is the loop that does it.
This is the same reason a regular expression with a Kleene star describes
infinitely many strings with a finite pattern, and the same reason a `for` loop
is shorter than the code it replaces. **A cycle is a loop in the literal
programming sense: repetition expressed once.**

The metric is in the codebase as `compression_ratio()` — trigrams known divided
by nodes alive. For `"abcabc"` it is `3 / 2 = 1.5`: three distinct trigrams held
in two nodes, and the ratio is above 1 precisely because of the cycle. Every
loop pushes that number up, and the ratio is high exactly when the data is
repetitive, which is to say, exactly when there is structure to find.

---

## 4. What a cycle costs

I am not going to claim this is free either. Four things break.

**1. Termination is no longer free.** A walk over a cyclic graph can go round
forever. In a DAG, "keep going until there is nowhere to go" is a complete
algorithm. In a cyclic graph it is an infinite loop.

**2. There is no layer order.** No topological sort exists, so "evaluate layer
1, then layer 2" is undefined. Back-propagation as normally written depends on
that order, which is why recurrent networks have to be unrolled in time before
they can be trained at all.

**3. Negative costs become ill-posed.** In a DAG you can happily give an edge a
negative cost — there is no way to exploit it. The instant a cycle exists, a
negative cycle means the cheapest path is `−∞`: go round forever and keep
collecting. Shortest-path search over a cyclic graph *requires* non-negative
edge costs.

This is why the cost function in `search.py` is what it is:

```
cost(p → c) = −log P(c | p) + step_penalty      ≥ 0
```

`P` is a probability, so `−log P ≥ 0`, and the step penalty only adds. It is not
a coincidence that the cost is non-negative by construction, and it is not a
stylistic preference for log-probabilities. **The decision to allow cycles is
what forces the cost function to be non-negative.** The structure dictated the
mathematics.

**4. At the level of a mind, a cycle is rumination.** The same thought, again,
with no new information and no exit. This is not a metaphor for the engineering
problem — it is the same problem, in the same graph, at a different scale.

---

## 5. The exit must come from outside

Here is the argument that organises everything else.

> **A system traversing a cycle cannot detect the cycle from inside it.**

Consider a walker at node `A` in a loop `A → B → C → A`. At `A`, the walker sees
`A`'s state and `A`'s outgoing edges. That is exactly what it saw the first
time. Nothing about the local state says "you have been here before" — that is
precisely what it means for the state to have returned. If the walker's
behaviour is a function of the current node, its behaviour on the second visit
is identical to its behaviour on the first, and therefore so is the third, and
the loop is eternal.

To get out, something must differ between visit one and visit two. That
something is by definition *not part of the cycle*. There are exactly three
kinds of thing it can be:

| Exit | What it is | In the code | In a mind |
|---|---|---|---|
| **a clock** | a quantity that only increases | `(node, chars_emitted)` search state; expansion budget | time passing, effort spent, fatigue, boredom |
| **a monitor** | a second process watching the trajectory | `dialogue.py` repeat detection | noticing you are going in circles |
| **an inversion** | flipping the landscape rather than searching it | `invert()`, 2NRL, `invert_paths`, punishing the duplicates that could not be avoided | deliberately doing the opposite; and, slower, no longer wanting to |

That table is the expansion of "we use metacognition or another part of the
brain instead". Each row is a different part of the brain, and each is
implemented.

---

### 5.1 The clock: unroll against something monotone

The cleanest exit, and the one that runs on every prediction.

In `search.py`, the search state is not a node. It is a pair:

```
state = (node_id, chars_emitted)
```

Moving over an edge `p → c` emits `len(label_c) − 2` characters, and
`chars_emitted` never decreases. So even though the *graph* is cyclic, the
*search space* is acyclic — you can revisit node `A`, but never in the same
state, because you have emitted more characters getting back there.

This is worth stating as a principle:

> **A cycle in the graph is not a cycle in the search, as long as every
> traversal consumes something that cannot be replenished.**

Which is the formal version of a fact about thinking: you can have the same
thought twice, but not at the same moment. Time has passed, and the context is
different, and that is what stops the second pass from being the first.

There is a second clock behind the first: `max_expansions = 200_000`. Even with
no character cap, the search gives up after a fixed budget and returns the best
partial answer it found rather than raising. Two guards, and they are the two
guards a brain has — **time and energy**. Neither one reasons about loops. They
do not have to. They just run out, and running out is sufficient.

Note what the fallback does: it returns *something*. The design never says "I
detected a loop, I have no answer". It says "here is the furthest I got". A
supervisory level that only ever reports failure is not much of a supervisor.

---

### 5.2 The monitor: metacognition, literally

`dialogue.py` is the piece of the system that most directly implements the
original note, and it has grown into the most complete answer to it. The model
talks to itself: two voices take turns, and every reply is the prediction search
run from the tail of what was just said.

The failure mode is obvious the moment you build it. The most likely
continuation of what was just said is *what was just said*. Unsupervised, two
copies of a model settle into a loop within a handful of turns, agreeing with
each other forever. This is a cycle — not in the graph this time, but in the
conversation, one level up. And it shows up at two scales at once, which is why
there are two monitors.

**Between turns**, `Heard` decides what counts as a duplicate: an utterance was
*said* before, or its reply *adds* what an earlier reply added (the same
continuation reached from a different context), or it *echoes* a line already
spoken. A longer utterance that happens to contain an earlier one is explicitly
not a duplicate — it says more than was heard.

**Inside a single utterance**, `stutter()` catches the loop directly:

```python
def stutter(text: str, longest: int = LONGEST_STUTTER) -> str:
    """The words an utterance says twice in a row, or ``""`` when it says each thing once.

    A *stutter* is a run of one to ``longest`` words repeated immediately after itself -
    "the **the** west", "say morning **morning**", "**the cat** the cat sat" - the shape a
    cyclic graph falls into when it walks a loop instead of going somewhere.  Words that
    come back later in the line are not a stutter: "where there is a will there is a way"
    says its words again, and says something with them.
    """
```

That docstring is the thesis of this paper stated from the other direction. A
stutter is not a language defect the system happens to produce; it is *what a
cycle looks like from the outside*. The walk went round, and the output says the
same words twice because the graph said the same nodes twice.

Note the discrimination, which is the hard part. Only an *immediate* repetition
counts. "Where there is a will there is a way" repeats its words and means
something by it; the monitor leaves it alone. A monitor that flagged all
repetition would be useless, because repetition is also how language emphasises,
balances and rhymes. The monitor has to distinguish *going round* from *coming
back*, and it does that by adjacency.

#### The escalation

```
1. speak the most likely continuation
   ↓ not if the conversation already heard it, and not if it stutters
2. shorten the context by one word and search again
   "nothing follows 'sat on the mat'? then what follows 'on the'?"
   ↓ repeat until the context is empty
3. change the subject entirely: search from START  (fresh = True)
   ↓ still nothing new
4. say the best duplicate anyway — flag it, and end the conversation there
   ↓ later
5. punish it: the flagged utterances become a 2NRL negative phase
```

Each step is a different escape, ordered cheapest to most drastic, and each one
is a different part of the system taking over from the one that failed.

**Step 1** is the monitor proper, and the separation is the whole idea. The
generator's job is to produce the most likely continuation, and producing a
repeat is not a bug in it — it is the generator working correctly on a graph
with a loop in it. A *separate* thing, holding a record of what has been said,
rejects the output. The generator cannot see its own loop, so something outside
it has to.

**Step 2** is re-framing: same question, less context. A narrower prompt has one
answer; a wider one has several.

**Step 3** is the other part of the brain. Give up on continuing and start
somewhere else entirely — changing the subject, which in cognition is what you
do after an hour of staring at a problem when you get up to make coffee.

**Step 4** is honest failure. It speaks, marks the turn `repeat=True`, and then
*ends the conversation* rather than saying the same duplicate again — because
saying it twice is the loop, one level higher again.

**Step 5** is treated in §5.3: the utterances it could not avoid are collected
and punished, so the loop is removed from the model rather than merely dodged.

#### Second thoughts: backing up to where the walk went round

The newest piece is the one I find most convincing, because it is the
impossibility argument of §5 turned into an algorithm.

Dropping a stuttering continuation throws away everything it got right. The
words *before* the walk went round were said once and were the most likely thing
to say; only the tail is the loop. So a voice that catches itself repeating does
not just take the next answer down the list. It backs up:

1. **Notice.** `stutter_at(text)` returns the character index where the
   utterance started saying itself again — "say morning **morning**" cuts after
   `"say morning "`. That index is *where the walk went round*.
2. **Back up to exactly there and keep it.** Everything before the cut was said
   once and is worth keeping.
3. **Explore from the cut.** Re-run the search with that longer prefix.

Step 3 is the crucial one, and the docstring says why better than I can:

> a longer prefix than the turn started with, which **forces** the walk to leave
> the loop at exactly the point it went round — asking the same question again
> from the context would only rank the same answers.

That is §5 exactly. *Asking the same question again from the same context would
only rank the same answers.* You cannot escape a cycle from inside it, because
inside it nothing has changed and the same query returns the same ranking
forever. The escape works only because the state is different — the prefix now
runs past the point where the loop closed, so the loop is no longer reachable
from where the search starts.

If that finds nothing, it backs up one more word and looks **wider** —
`k * (step + 2)` candidates, so the further back it goes the more it weighs —
up to `explore` times (default 3). Cheap and narrow first, expensive and broad
only when the cheap move failed.

And it is scoped. `keep` is the context picked up from the other voice, and it
may not be rewritten:

> a voice rethinks what it said, never what it heard.

A repeat inside the *other* voice's words is recorded and left alone. The
monitor's authority stops at its own output — which is both good manners and
good engineering, since the other voice's text is not this one's to explain.

The whole episode is recorded in an object whose docstring needs no gloss from
me:

```python
class Rethink:
    """A voice catching itself repeating, and what it did about it.

    The metacognition of a turn: ``noticed`` is the run of words it caught itself
    saying twice, ``cut`` what it kept of that attempt (everything said before the
    walk went round), ``steps`` how many times it backed up, ``explored`` the paths
    it weighed from there and ``found`` whether one of them said something new.
    A turn that never had to think twice has no record at all.
    """
```

`noticed`, `cut`, `steps`, `explored`, `found`. The system does not merely
escape its loops — it keeps a record of having noticed, of how far it backed up,
of how hard it looked, and of whether it got anywhere. A turn that never had to
think twice has no record at all, which is exactly right: metacognition is not
running all the time, it engages when the base level gets into trouble.

Nothing in any of this inspects the graph for cycles. There is no loop detector,
no visited set, no tortoise and hare. The monitor watches **outputs** and
notices that they repeat — which is precisely what metacognition has access to.
You do not perceive your own synapses. You notice that you have had this thought
before.


### 5.3 The inversion: flip the landscape instead of searching it

The third exit is the one I find most interesting, because it is not a search
strategy at all.

If a model keeps producing the same wrong output, no amount of searching *within*
its current landscape helps — the output is the minimum, that is why it keeps
coming back. So invert the landscape. **2NRL**: train on the failures, invert
the network so that everything likely becomes unlikely, then fine-tune on the
correct data at a smaller learning rate. `invert()` flips every edge weight and
every activation amplitude; what was an attractor becomes a repeller. The
procedure has its own paper (`2NRL.md`); what concerns this one is what the
cyclic structure does to it.

This works because of a property of the sine activation — the sign of the
amplitude `a` is the sign of the unit, so negating a unit is a parameter change
rather than a structural one. (That is treated properly in the companion paper,
`SineWaveActivationFunction.md`.)

The targeted version is where cycles bite back, and the mathematics here is
worth doing carefully.

An edge score is a **product of two activations**:

```
score(p → c) = w · f_p(z_p) · f_c(z_c)
```

To flip the sign of that score by touching nodes, you need *exactly one* of the
two endpoints to flip. So to flip every edge along a path, you flip **every
other node**:

```
path:            n0 ──▶ n1 ──▶ n2 ──▶ n3 ──▶ n4
flip alternate:   ✗      ·      ✗      ·      ✗
edge signs:        flipped flipped flipped flipped     ← all four
```

Verified numerically:

```
base             : -0.350  -0.360  -0.176  -0.048
flip alternate   : +0.350  +0.360  +0.176  +0.048     every edge flipped
flip every node  : -0.350  -0.360  -0.176  -0.048     nothing changed at all
```

Flipping *everything* achieves nothing, because each edge gets two sign changes
and they cancel. The naive move is precisely the useless one. You have to
two-colour the path.

**And now the cycles.** A path is always two-colourable. A graph with a cycle in
it may not be. A graph is two-colourable if and only if it contains no cycle of
odd length — and a walk through a cyclic graph can absolutely close an odd loop.
When it does, the conflict is unresolvable:

```
3-cycle  n0 → n1 → n2 → n0 , all 8 possible flip patterns:

   flip 000 : 0 of 3 edges flipped
   flip 001 : 2 of 3
   flip 010 : 2 of 3
   flip 011 : 2 of 3
   flip 100 : 2 of 3
   flip 101 : 2 of 3
   flip 110 : 2 of 3
   flip 111 : 0 of 3
                      best possible: 2 of 3. Never 3.
```

There is no assignment that flips all three. The obstruction is exact, not
approximate.

The smallest odd cycle is the self-loop, and it is the extreme case:

```
score(p → p) = w · f_p · f_p = w · f_p²
```

`f_p²` is non-negative no matter what `f_p` is, so **flipping the node changes a
self-loop's score by exactly nothing.** The sign of a self-loop is the sign of
its weight alone. A self-loop is sign-locked against node inversion.

This is why the implementation says "the parity that covers the most edges, a
shared node takes the largest amount" rather than "the parity that covers every
edge". It is not sloppiness. It is the only thing that can be said in a graph
that has cycles in it, and it is a clean example of the general shape of this
research: **allowing cycles made an operation that is exact on a DAG into one
that is best-effort, and the design absorbs that rather than banning the
cycles.**

For a self-loop the fix is the other knob: change `w`. Which is the right
answer anyway — a self-loop means "this repeats", and the thing you want to
adjust about a repetition is how attractive repeating is, not which direction
it points.

#### The slowest clock: punishing the loops that could not be escaped

The inversion exit also runs on a much slower timescale, and this is step 5 of
the ladder in §5.2.

When every candidate is a duplicate, the voice says the best one anyway and
flags the turn. `repeats(turns)` then collects those flagged utterances — *the
duplicates the search could not avoid* — and they become the bad half of a 2NRL
negative phase: the CLI prints the `radixnet feedback --bad-text …` that
punishes them, and the Converse tab's "Punish duplicates" marks them 👎 so the
next training run teaches the model out of them.

That closes a loop between the timescales, and it is worth being explicit about
what just happened:

| Timescale | Mechanism | What it changes |
|---|---|---|
| within one search | the monotone resource, the expansion budget | nothing; the walk just cannot go round forever |
| within one turn | `backtrack` — notice, cut, explore from past the loop | the query, not the model |
| within one conversation | `Heard`, changing the subject, ending early | what gets said |
| across training runs | punish the flagged duplicates via 2NRL | **the graph itself** |

The first three exits route *around* a loop. Only the last one removes it. A
loop the search could not escape is evidence about the model, and the system
treats it as evidence: the thing it could not stop saying becomes the thing it
is trained not to say.

This is the difference between coping and learning, and a mind needs both. You
can notice you are ruminating and change the subject — that is the monitor, and
it works this afternoon. Or the fact that you kept ruminating can change what is
attractive to you in the first place, which is slower and is the only one of the
two that means you do not have to keep noticing.

---

## 6. Why the loops are what make a higher level possible

Now the claim I actually care about.

A DAG mind needs no metacognition. Every computation terminates on its own;
there is no state it can fail to leave; there is nothing for a supervisor to
supervise. Add a monitoring level to a DAG and it has no job.

A cyclic mind needs one. And here is the part that makes cycles a *feature*
rather than a cost paid for compression: **the same property that makes the
monitor necessary also makes it powerful.**

Because the base level can loop, it can also sustain. A loop held deliberately
is working memory. A loop traversed cheaply is a habit. A loop with a counter is
iteration. A loop through your own output is self-reflection. Every one of those
is the thing the DAG could not do, and every one of them is the *same structural
fact* as rumination. The failure mode and the capability are the identical
mechanism, and which one you get depends entirely on whether something is
watching.

That is what I meant by cycles being a feature. Not "loops are tolerable if you
guard them". Rather:

> **The loop is the capability. The monitor is what turns it from a trap into a
> tool. Neither exists without the other, and a mind built from a DAG has
> neither.**

Metacognition is not a luxury layer bolted onto a working system. It is the
other half of the design. The instant you let the graph point backwards, you
have committed to building the thing that watches it, and you have also unlocked
everything the backward edges can do.

---

## 7. Loops that are the point, not the problem

Concretely, the things I want the architecture to be capable of, all of which
are cycles:

**Working memory.** Holding something available is a loop sustaining itself.
There is no "hold" operation in a feedforward pass — the activation goes through
and is gone. Persistence *is* recurrence.

**Skills and habits.** A well-learned action is a path that is cheap to
traverse, and a repeated action is a path you re-enter. Learning a skill is
partly making a loop cheap; over-learning it is making the loop so cheap you
enter it when you did not mean to.

**Counting and iteration.** "Do this until X" is a cycle plus a monotone
resource — precisely the `(node, chars_emitted)` construction from §5.1. Counting
is not a separate faculty bolted on; it is what a loop plus a clock *is*.

**Self-reference.** Thinking about your own thinking means the graph contains a
path from a node back to itself. There is no acyclic encoding of that. If you
want a system that can model itself, you have already chosen a cyclic graph
whether or not you admit it.

**Learning from your own output.** `converse` feeds the model's output back in
as its input; `Evolver` makes the model's own samples into its next training
data, with the worst of them becoming the garbage half of a 2NRL pass. Both are
cycles at the system level, and both are how the thing improves without new
external data. A DAG-shaped system has to be fed. A cyclic one can feed itself —
which is, incidentally, the only version of "self-improving" that means
anything.

---

## 8. Rules that follow, for anyone building on this

These are the invariants the codebase actually enforces, and each one exists
because of a cycle.

**1. Never merge a self-loop.** Path compression merges `p → c` into one node
when the chain is unary, but the code excludes `c == p`:

```python
c = next(iter(ch))
if c == p or c == START or c == END:
    return False
```

and the test asserts `g.compress() == 0` for `"aaaa"`. Merging a self-loop would
destroy the statement "this repeats", which is the one thing the node is there
to say. **Compression is allowed to shorten a chain; it is never allowed to
delete a loop.**

**2. Structural edits must preserve the network function, cycle edges included.**
When two nodes merge, the merged node keeps the activation of whichever endpoint
has the larger `|f|`, and the edges on the other side are rescaled by the ratio
of the two activations so every edge score `w · f_p · f_c` is unchanged. The
docstring is explicit that the cycle edge `c → p`, which becomes the self-loop
`p → p`, is covered by the same rescale. If you touch the structure, check what
it does to the loops — they are the cases that do not follow the general
argument.

**3. Costs must be non-negative.** §4. Non-negotiable once cycles exist.

**4. Every search state must carry the monotone resource.** Keying on `node`
alone is an infinite loop; keying on `(node, chars_emitted)` is a finite search.
This is the single most important line of the design.

**5. Every generator needs a stop condition that is not "nowhere left to go".**
In a cyclic graph there is always somewhere left to go. Stop on the resource, on
the budget, or on the monitor — never on exhaustion.

**6. Monitors watch outputs, not structure.** Do not write a cycle detector over
the graph. Watch what comes out and notice repetition, the way `dialogue.py`
does. It is cheaper, it catches loops that span levels the graph does not
represent, and it is what a real supervisory system has access to.

---

## 9. Predictions, and what would change my mind

Falsifiable, in order of how much each would cost me:

**Prediction 1.** Compression ratio rises with corpus repetitiveness, and the
cyclic representation stays bounded on inputs where the acyclic one grows
linearly. This one is nearly definitional and I would be surprised to be wrong.

**Prediction 2.** Disabling the `dialogue.py` monitors (`avoid_repeats=False`,
`avoid_word_repeats=False`, `explore=0`) collapses self-conversation into a
fixed point within a small number of turns, and the number is small — single
digits. If conversations stay varied without the monitors, then the generator is
escaping its own loops somehow and my impossibility argument in §5 has a hole in
it.

**Prediction 2a.** Of the three, `explore` should matter most per unit of cost.
Rejecting a stuttering candidate only moves down a ranking that was produced
from the same context, so the runner-up is drawn from the same loop; backing up
past the cut changes the query itself. Concretely: at `explore = 0` the
`Rethink.found` rate should be zero by construction, and turns that stutter
should mostly resolve by *changing the subject* (`fresh=True`) rather than by
continuing; at `explore = 3` most should resolve by continuing. If backing up
and simply taking the next candidate perform the same, then the state really was
not what mattered and §5 is wrong about where the escape comes from.

**Prediction 3.** Targeted inversion is measurably less effective on paths
containing odd cycles than on simple paths, because the parity cover is
incomplete (§5.3). This is the sharpest test: it is a specific, quantitative
consequence of the odd-cycle obstruction, and it is a direct prediction of the
mathematics rather than of the philosophy.

**Prediction 4.** Raising `step_penalty` shortens generated output by making
loop traversal expensive, monotonically. A penalty is a per-traversal tax on
loops, so this is the knob that trades compression against wandering.

**What would change my mind:** if the monotone-resource unrolling turned out to
be doing all the work — if a system with only the clock, and no monitor and no
inversion, performed as well as the full thing — then cycles would be merely
survivable rather than a feature, and the metacognition claim would be
decoration on a termination guard. I do not believe that, because Prediction 2
is the experiment that separates them, but it is the result that would cost me
the thesis.

---

## 10. Limitations

1. **This is an architectural argument supported by a small implementation.**
   The graph experiments are character-level and small. Nothing here establishes
   that the cyclic structure scales to where it would have to scale.
2. **The neuroscience is directional, not quantitative.** I am claiming that
   cortex is richly recurrent and organised in closed loops, which is not in
   dispute. I am not claiming a mapping from any specific loop to any specific
   mechanism here, and I would not defend one.
3. **"Metacognition" is doing real work in §5.2 and looser work in §6.** The
   implementation is a repeat-detector over utterances. That is a genuine
   instance of the pattern — a monitor outside the generator, watching outputs —
   but it is one narrow instance, and I should not let the word carry more than
   the code does.
4. **The three exits may not be exhaustive.** I argued that the differing
   quantity must come from outside the cycle, and then listed three kinds. The
   argument that it is *three* is by construction, not by proof. Noise is
   arguably a fourth and I currently file it under "inversion done badly".
5. **Cost accounting is missing.** Cyclic search is more expensive than a DAG
   pass, and I have not measured how much the unrolled state space costs against
   the compression it buys.

---

## 11. Summary

- The brain is a directed cyclic graph. I previously said acyclic; that was
  borrowed from engineering convenience, not observed in a brain.
- A cycle is repetition stored once. One node and one self-loop, learned from
  `"aaaa"`, represents `"aaaaaaa"` and every longer repetition — generalisation
  straight out of the structure, with no training procedure involved.
- A finite acyclic graph cannot represent an unbounded language. A finite cyclic
  graph can. That is what the loop is for.
- The costs are real: no guaranteed termination, no layer order, no negative
  costs (which is *why* the cost function is `−log P + penalty ≥ 0`), and
  rumination.
- A system inside a cycle cannot detect it from inside, because inside it
  nothing has changed and the same query returns the same ranking forever. The
  exit comes from a clock, a monitor, or an inversion — the
  `(node, chars_emitted)` search state, the ladder in `dialogue.py`, and
  `invert()` / 2NRL.
- A stutter is what a cycle looks like from outside: a run of words said twice
  in a row because the walk said the same nodes twice. The monitor finds the
  loop by watching output, never by inspecting the graph — and it has to tell
  *going round* from *coming back*, which it does by adjacency.
- The escape works by changing the state, not by re-asking. `backtrack` cuts the
  utterance at the point the walk went round and searches again from a prefix
  that runs past it, which forces the walk out of the loop; taking the
  next-ranked answer from the same context would only re-rank the same loop.
- What cannot be escaped gets punished. Duplicates the search could not avoid
  become a 2NRL negative phase, so the loop is removed from the graph instead of
  dodged again next time. Routing around a loop is coping; this is learning, and
  a mind needs both.
- Cycles change the mathematics, not just the engineering: flipping a path's
  edges requires flipping alternate nodes, an odd cycle makes that impossible
  (2 of 3 edges at best), and a self-loop is sign-locked entirely because
  `w · f²` cannot change sign by flipping `f`.
- Loops are what make a supervisory level necessary *and* what make it
  worthwhile. Working memory, habit, iteration and self-reference are the same
  structural fact as rumination. Which one you get depends on whether something
  is watching.

Cycles are a feature. The monitor is the other half of the feature.
