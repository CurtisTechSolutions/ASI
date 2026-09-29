# RadixDecayNN — Design

`RadixAcyclicNN`'s tree with its parameters taken out and one number put
in their place. A node holds nothing but `seen`. Every traversal that arrives
at a node adds one to it — a window of a text being read, and a token of a
text being said, alike. `seen` fades on the model's own clock, which is the
count of traversals anywhere in the tree. At a branch, each option's share
of the `seen` the options hold between them is its probability. That is the
whole model: no weights, no activation, no gradient, no learning rate.

The three rules were decided in conversation and are the specification:

1. **`seen` stays, and decays.** The count models kept `seen` as a count that
   only grew; here it is a value that fades with elapsed traversals.
2. **Every traversal adds one.** Recovery is by use: a node that is walked
   through is refreshed by that walk, whatever the walk was for.
3. **A query adds too.** Saying something is a traversal of what is said, so
   the model remembers what it has said as it remembers what it has read.
   Only *measurements* — scoring, accuracy, recital counts, and any walk
   asked for `quiet` — leave `seen` alone.

This document says exactly what the code does. `README.md` says what it
measured.

## 1. The structure, inherited

Everything structural is `RadixAcyclicNN/DESIGN.md` §§1–6, repeated in
`tree.py` so the directory stands alone:

* a node is a **context**, the path of grams from the root; the structure is
  a rooted tree, and `check_invariants` proves it after every operation the
  tests make (§11);
* a text is inserted from START and from the root at **every** position — a
  suffix tree — so every substring of every text read is a root path;
* a run of nodes with one child each is one node (path compression); a
  transition into the middle of a run **splits** it; `compress` merges unary
  chains back, and reading never leaves one;
* `depth` bounds a root path to that many symbols (START counts as one);
* node `0` is ROOT (the empty context), node `1` is START, and every context
  a text ended in has its own END leaf.

The encoding (`encoding.py`) is the sibling's: a gram is `n` units sliding
by one, a unit a character, a phone or a syllable (the last two through
`../PhoneticTokenizer`), and a prediction over sounds is spelled back into
words. Nothing below depends on which.

## 2. `tree.py` — storage

Per node, parallel lists: `labels`, `kind` (ROOT, START, END, REAL),
`parent`, `children` (first gram → child id), `end_leaf`, `alive`, and the
memory:

| field | meaning |
|---|---|
| `value[i]` | `seen` as it was last written down |
| `stamp[i]` | the clock reading it was written down at |
| `traversals` | the clock: every traversal anywhere, plus time let pass |

Settings: `life` (traversals), `decay` (`half-life`, `linear`, `none`),
`min_seen`, `depth`. `grams` is the set of every distinct gram read.
`_totals` caches §5's option totals.

## 3. Reading

`read(grams)` inserts every window of a text (`_windows`: from START the
whole text — the first `depth − 1` grams under a bound — then from ROOT at
every position, `depth` grams each under a bound) with the sibling's radix
insertion (`_insert`), and **touches** (§5) every node a window passes
through: its origin (START or ROOT), every node on the way, the leaf it
makes, and the END leaf when the window ends the text.

Consequences worth knowing:

* A node's `seen` after one reading is the number of windows that passed
  through it: START holds one per text, a ROOT-suffix context one per
  occurrence of its string as a substring, a START-anchored context one per
  text that begins with it.
* **A split copies `seen`.** Both halves of a run were passed through by the
  same windows, so both keep the run's value and stamp. `total_seen` can
  therefore exceed `traversals`; it is what the tree remembers, not what
  it was paid.
* A **merge keeps the larger** of the two, settled now.
* **Reading costs clock time**: the sample corpus (60 texts, 1 800
  characters) is 6 845 traversals the first time and 7 550 the second,
  because the tree is split more finely by then and a window passes through
  more nodes. A said text of 36 characters costs 6. §10 has the arithmetic.

## 4. The three decays

`seen(i)` is read through elapsed time, `e = traversals − stamp[i]`:

| `decay` | `seen(i)` | a lone visit is gone (below `min_seen = 0.5`) after |
|---|---|---|
| `half-life` | `value · 0.5^(e / life)` | one life (and never reaches zero) |
| `linear` | `max(0, value − e / life)` | half a life (and is exactly zero after one) |
| `none` | `value` | never |

Reading `seen` writes nothing: it is a pure function of the two stored
numbers and the clock, so there is no sweep, no schedule, and the value a
node holds depends only on how much has happened since it was last touched.
`settle(i)` writes the faded value down at the current reading (used by
`merge_child` and `to_dict`); `tick(k)` advances the clock by `k` with
nothing arriving anywhere — what happens to this tree while the world is
busy.

`life` defaults to 10 000 traversals, the cyclic count model's window
(`RadixCyclicNN/DECISIONS.md` D-022) borrowed for its units: how far back
the model calls recent. It is a setting; §10 says what it is worth in
texts.

## 5. Touching, shares, and what is usable

```
touch(i, amount=1):  value[i] = seen(i) + amount;  stamp[i] = traversals;  traversals += 1
```

is the only way `seen` grows. Reading and saying both come through it.

At a node `p`, `option_total(p)` is the sum of `seen` over its options — its
real children and its END leaf; START is never an option — cached under the
clock and structure it was computed for, so a quiet walk that asks a wide
node (ROOT has one child per distinct first gram) again and again computes
it once. `shares(p)` is each option's part of that total, **omitting an
option whose `seen` is exactly zero** (`linear` reaches zero; `half-life`
does not). `log_share(p, c)` is `log` of one share, `None` when `c` is not
on offer; `child_costs` is `−log share`, non-negative, what the search walks
by.

A located context `(node, offset)` is **usable** when `seen(node) ≥
min_seen` and it has something to continue with: it is inside a run (the
next gram is certain), or an option at its end still holds something.
`min_seen` is the floor of memory: below half a visit a context is not
consulted, though what it holds still counts among its parent's options and
still recovers when touched.

## 6. Locating a context

`walk(origin, grams)` and `step(node, offset, gram)` are the sibling's: a
gram at a time along a run or into the child that begins with it.
`locate(grams, ending, anchored, longest)` is the deepest usable context of a
history: the whole of it from START first (when the window can hold it, and
`anchored`), then its suffixes from the root, longest first. `ending`
consults a context of exactly `depth` symbols for END alone; `longest`
considers no suffix of more grams than that, and then not the walk from
START either — the back-off of §8.

**Deepest usable is not monotone here.** On a tree built by reading alone a
shorter suffix is always at least as fresh as a longer one (it was touched by
every window that touched the longer, and later in the same reading). A
saying touches only the deepest context it consults (§7), so after sayings a
long context can outlive its own suffixes; the model then knows a text from
its beginning and not from its middle. That is a property, not a bug, and it
is why `locate` walks every suffix rather than searching by length.

`Walks` makes that affordable for the token-by-token walk: every suffix of a
growing history is kept as a cursor and moved one gram per token (`push`), a
cursor that fails is dropped for good (no extension of a non-path is a
path), and `deepest()` answers exactly what `locate` answers for the history
— over its last `window` grams, matched from START only when the window
holds the whole history — in `O(history)` per token instead of a fresh walk
of every suffix. The test suite proves the two agree at every step on a tree
that has said things and faded.

## 7. Saying

`say(prefix, length, window, temperature, to_end, max_length, seed, quiet)`
is the acyclic model's `mode="slide"`, and it is the query rule 3 speaks of:

1. **Where it starts.** The deepest usable context of the prefix (`Walks`
   over its grams); for a prefix shorter than a gram, the most-seen usable
   child of START whose label begins with it, of which the walk says the
   rest of the first gram; for nothing known, START. If that start is START
   and START itself is not usable — its `seen` has faded below `min_seen`,
   or every first option has faded to nothing — the model says nothing at
   all: it has forgotten how texts begin.
2. **One query per token.** From the context, `next_token`: inside a run the
   run's next gram at cost 0; at a node's end the option with the largest
   share (`temperature = 0`; ties to the older node), or a draw from
   `share^(1/T)` above it. The token is fed back (`Walks.push`) and the next
   query is the deepest usable context of everything said so far, over the
   last `window` grams if one is given.
3. **What is traversed.** Every node the walk *arrives at*: the context it
   starts from, each option it takes, and each context it relocates to when
   the deepest usable context is a different node from the one it is in. A
   run is one node and is arrived at once; a step inside it is not a
   traversal. The ancestors of the start context are not touched — the
   prefix was given, not read. `quiet=True` is the same walk with no touch.
4. **Stops.** END; `length` units (unless `to_end`); `max_length`;
   `MAX_SLIDE = 2000` units when `to_end` has no `max_length` (a window that
   forgets can go round, and this is its clock); a context the tree knows
   nothing usable about.

`PathResult.traversals` is what a walk added; `expanded` counts its
queries. `generate(count)` says `count` texts from START; at temperature 0
a saying model can still differ from one to the next, because each one was
traversed and the tree has changed.

`cheapest(prefix, length, to_end, max_length, quiet, max_legs)` is the other
way to speak: one search (§9) instead of a query per token, and then, unless
quiet, every node on the path it returns is traversed once, the start
context included. A bounded tree re-enters at the deepest usable context of
what was said when a path runs out, at most `max_legs` times; a leg of a
tree bounded to `depth` symbols says one gram.

## 8. Scoring — a measurement

`score(text)` is the log-probability of a text under the shares. Nothing
moves. The walk:

* **follows** the context it is in while that context is usable: inside a
  run a matching gram costs 0 and is not a transition; at a node's end a
  gram that is an option costs `log share` and is one;
* **backs off** when the context is not usable — it has faded below
  `min_seen`, its options have all faded, or it holds `depth` symbols — to
  the longest usable suffix *shorter than the context it is leaving*
  (`locate(..., longest = symbols − 1)`), and to ROOT, the empty context,
  when there is none: at ROOT every first gram is an option and a text is
  scored as a bag of grams;
* **relocates** on a miss — a gram the context was never followed by, or
  whose option has faded to exactly nothing — to the deepest usable context
  of the whole history, charging `log(UNKNOWN_PROB) = log 10⁻⁶` and counting
  one unknown transition, the two sibling models' constant, so the three
  read on one scale;
* ends with END from the context it is in (consulted with `ending`), unknown
  when that context has no END leaf or its END has faded to nothing.

The back-off looks only at shorter suffixes because the walk got where it is
by following children from the deepest usable context; on a tree built by
reading alone this is exactly "the deepest usable context of the whole
history decides at every step" (the sibling's rule), and the tests check the
fast walk against that rule stated the slow way, at two depths and two
floors, fresh and faded. After sayings the two can differ (§6) and the walk
above is the definition.

`bits_per_unit(texts)` is `−log₂ P` per unit with the miss rate;
`next_token_accuracy(texts, window)` how often the walk's most likely token
is the text's next one (`Walks`, quiet); `recites(texts, prefix_units)` how
many texts the model says whole, quietly, from their first units.

## 9. Search (`search.py`)

`cheapest_path(tree, node, offset, min_units, max_units, to_end)` is
Dijkstra by `−log share`: non-negative costs, so the first goal popped is
the answer; every node reached by one path, so nothing is pushed twice and
no `(node, units)` state or budget is needed; the search ends when the
subtree does. It visits without arriving — it is `cheapest`'s decision, not
the search's, whether the path is then traversed.

## 10. The clock, in numbers

Everything fades against `traversals`, so what a life is worth depends on
what is put on the clock:

| event | traversals (sample corpus, `char:3:1`) |
|---|---|
| reading one 30-character text, first time | ≈ 114 |
| reading the 60-text corpus | 6 845 the first time, 7 550 the second |
| reading the 640-text prose corpus | ≈ 233 000 |
| saying a 36-character text from START | 6 |
| saying a text token by token under a window of 4 | ≈ 35 (a relocation per token) |
| asking anything quietly | 0 |

Reading is expensive on the clock because every suffix window is a walk;
saying is cheap because a run is one arrival. So a life of 10 000 is 1.5
readings of the sample corpus, and a twentieth of one reading of the prose
corpus — a model that reads the prose corpus once at that life has forgotten
most of it before it finishes (`README.md`, forgetting on prose, life
100 000). Choose `life` against the corpus and against how the model will be
used; the experiments show what each choice buys.

A reading is also *ordered*: START is touched first for each text and the
END leaf last, and the windows of the last text are the freshest. A model
that has just read a corpus once already remembers its first texts less than
its last.

## 11. Invariants (`check_invariants(texts=None, compressed=False)`)

The sibling's, plus the memory's:

* ROOT is `0` with no parent and no END; START is `1` under ROOT, never an
  option; every alive node's parent is alive, its parent's child for its
  first gram (or its parent's END leaf), and reachable from ROOT by parents
  without meeting itself — **no cycles**; parents and children agree exactly;
  every gram in every label was read; a child's label begins with its
  parent's last `n − 1` units; a real label holds at least one gram;
* `value ≥ 0` and `stamp ≤ traversals` on every alive node; dead nodes are
  empty tombstones;
* with `compressed`, no unary chain; with `texts`, every window of every text
  walks to a node, ending ones at a node's end with an END leaf, and every
  whole text short enough for the bound is a root path.

## 12. Persistence

`to_dict` writes every alive node compacted (ROOT and START first), with
`seen` **settled** at the clock it was written at, and the clock,
settings, grams and structure version. `from_dict` gives every node that
value with the file's clock as its stamp, so a loaded tree fades from the
moment it was saved exactly as the saved tree would have. The model file
adds the seed and the RNG state (a draw after a load is the draw that would
have followed the save), the reading history and the meta counters
(`texts_read`, `units_read`, `readings`, `traversals_said`, `ticks`). JSON,
gzipped when the name ends in `.gz`, written atomically.

## 13. The experiments (`experiment.py`)

Each is a function of a corpus and the settings, returns a JSON record, and
`run` writes `results/<name>_results.json`; `tables` renders the markdown
`README.md` quotes. All deterministic given the seed.

| experiment | what it does | what it measures |
|---|---|---|
| `habit` | says one prefix `times` over (temperature 0); the same asked quietly; `sampled` sayings at temperature 1 | the probability of the text said, per repeat; the shares and entropy at the first branch the walk decides at, before and after |
| `forgetting` | reads once, then ticks to each of `lives` since the reading ended, under each decay | nodes remembered (`≥ min_seen`), texts recited, bits per unit and miss rate, `total_seen` |
| `use` | after reading, `rounds` silences of `gap` traversals (a tenth of a life), each broken by saying one text from its first units — said, said quietly, or not at all | whether the used text is still recited; what else is; bits |
| `recovery` | reads, fades `fade_lives`, re-reads every `every`-th text once, then every text once | recited among the re-read and among the others, remembered, bits, at each stage |

## 14. Costs and limits

* A reading is `O(windows × path)`; `say` is `O(history)` per token through
  `Walks` (`O(window)` under a window); `score` follows in `O(1)` per gram
  and pays `O(L²)` for a back-off from a context of `L` grams and `O(T²)` for
  a miss over a history of `T`; `cheapest` is Dijkstra over a subtree.
* `seen` is a float and `half-life` never reaches zero: an option read once
  and never again is offered forever, at a vanishing share. `min_seen`
  decides what is *consulted*; `linear` decides what is *offered*.
* The clock is global. Two models in two files fade independently; there is
  no shared time.
* Same as the acyclic sibling: repetition is unrolled (`"a" × n` costs
  `n − 1` nodes), a substring never seen is unknown, and the tree does not
  generalise what the cyclic graph's cycles generalise. The cyclic graph
  with this memory is the next model, not this one.

## 15. Decisions

| # | decision | why |
|---|---|---|
| D-1 | `seen` is the whole memory; there are no weights and no activation | the model is the rule "every traversal adds one, and it fades"; anything learned by a gradient would obscure what that rule alone does |
| D-2 | every traversal adds exactly one, reading and saying alike | one unit of evidence per arrival; the two verbs weigh the same, so what is said can outweigh what was read (`README.md`, habit) |
| D-3 | a saying traverses the context it starts from, each option it takes, and each context it relocates to; not the ancestors of the start, not a run's interior | arrival is the unit; the prefix was given, not read; a run is one node |
| D-4 | measurements are quiet; `quiet=True` makes any walk one | so that measuring a model does not change it, and the experiments have a control |
| D-5 | the clock is traversals anywhere, `tick` lets time pass | the model's own time, with no wall clock and no epoch; a saying and a reading both spend it |
| D-6 | `half-life` by default, `life = 10 000`, `linear` and `none` beside it | the half-life never forgets outright and is the usual forgetting curve; `linear` is the "subtract" the idea began as, made proportional to time; `none` is the count model back |
| D-7 | `min_seen = 0.5`: half a visit is the floor of consulting a context | a natural unit: below it a lone visit is more forgotten than remembered; an option is still offered while it holds anything |
| D-8 | a split copies `seen` to both halves; a merge keeps the larger | both halves were passed through by every window that passed through the run; a merge loses nothing it held |
| D-9 | `locate` walks every suffix; `Walks` makes it incremental | deepest usable is not monotone after sayings (§6), so no search by length is exact |
| D-10 | `score` backs off to shorter suffixes only, then ROOT | the walk got there by following children; the empty context is the last resort, not a miss |
| D-11 | a model whose START has faded says nothing from nothing | it has forgotten how texts begin; anything else would invent a start |
| D-12 | option totals are cached under the clock and the structure | a quiet walk asks ROOT thousands of times; the cache turned a 30-second scoring into a fifth of one |
| D-13 | the structure is the acyclic tree; the cyclic graph follows | one change at a time: this directory measures the memory, the sibling measured the structure |
