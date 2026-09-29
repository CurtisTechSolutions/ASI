# RadixDecayNN

**A radix tree whose only memory is `seen` — fed by reading and saying alike,
and fading on its own clock.** `RadixAcyclicNN`'s tree of every suffix, with
the weights, the sine activation and the learning rule taken out and one
number put in their place. A node holds nothing but `seen`. Every traversal
that arrives at a node adds one: a window of a text being read, and a token
of a text being said, alike. `seen` decays with the traversals that have
happened since — a half-life of ten thousand of them by default — and at a
branch each option's share of the `seen` the options hold between them is
its probability. Measurements (scoring, accuracy, counting what it recites,
anything asked `quiet`) move nothing.

Three consequences follow, and this directory measures each of them: a
**habit** forms — what the model says, it is more likely to say again; a
memory **fades** on a forgetting curve whose shape is the decay's; and
**use** keeps a memory alive while **reading** brings a faded one back, one
traversal at a time. Pure Python, standard library only, sized like its
siblings: a package (`radixdecay`), a CLI, a Makefile, tests, and the four
experiments with their results.

## The ideas, in one table

| Rule | Implementation |
|---|---|
| `seen` is the memory | `DecayTree.value[i]`, `stamp[i]` and the clock `traversals`; `seen(i)` is read through the elapsed time and writes nothing. No weights, no activation, no gradient. |
| Every traversal adds one | `touch(i)`: settle, add one, tick the clock. `read` touches every node every window passes through — origin, path, leaf, END. |
| A query adds too | `say` (token by token) and `cheapest` (one search) traverse every node they arrive at: the context they start from, each option taken, each context relocated to. `quiet=True` asks without arriving. |
| It fades on the model's own clock | `life` traversals: `half-life` (`value · 0.5^(elapsed / life)`), `linear` (`value − elapsed / life`, floored at 0), or `none`. `tick(k)` lets time pass with nothing arriving. |
| Shares are the probabilities | `shares(p)`: each option's part of the `seen` at `p`; an option faded to exactly nothing is not offered. `−log share` is the search's cost. |
| A floor of memory | `min_seen = 0.5`: below half a visit a context is not consulted, and the walk backs off to a shorter, fresher one. |
| The structure | The acyclic sibling's: a node is a context, every suffix is inserted, runs are compressed and split, no cycles — proved by the invariant checks after every operation the tests make. A split copies `seen` to both halves; a merge keeps the larger. |
| Scoring, on one scale with the siblings | The deepest usable context decides, a miss costs `UNKNOWN_PROB = 10⁻⁶`, so bits per character compare across the three directories. |
| Sounds | `--unit phone` or `syllable`: read through `../PhoneticTokenizer`, said as sounds, spelled back as words. |

`DESIGN.md` is the specification; its §15 lists the decisions.

## Quick start

```bash
cd RadixDecayNN
make test                                # 65 tests, a few seconds
make demo                                # read the sample corpus; watch it remember, say, forget, recover
make experiments                         # the four experiments below, a second
make read && make say PREFIX="the"       # save a model, say something three times, save what that did to it
python3 -m radixdecay say "the cat" --to-end --quiet          # ask without arriving
python3 -m radixdecay tick --load model.json.gz --lives 2 --save model.json.gz
python3 -m radixdecay --help
```

The demo, abridged:

```
60 texts, 1800 chars; life 10000 traversals, decay half-life
  read 1×: 6845 traversals; 2224 nodes, 1682 END leaves, 715 grams, 607 branches
  recites 57/60 texts from their first eight chars; 0.197 bits per char

  saying 'the' three times - every saying is a traversal of what it says:
    'the bear sleeps through the winter'  probability 0.027  traversals 5  clock 6850
    'the bear sleeps through the winter'  probability 0.054  traversals 5  clock 6855
    'the bear sleeps through the winter'  probability 0.079  traversals 5  clock 6860

  two lives pass with nothing read or said:
    remembered 321/2224 nodes; recites 1/60; 4.55 bits per char
    'the' -> ' bear sleeps through the winter'  (what it said three times is what it still says)

  the corpus read once more:
    remembered 2224/2224 nodes; recites 57/60; 0.198 bits per char; clock 34570
```

## What it measured

Four experiments, each with a control, on the 60-line sample corpus (a
second) and on the 800-line prose corpus (`make experiments-prose`, a few
minutes). `results/` holds the JSON and the logs; every number below is in
them. Settings unless stated: `char:3:1`, `half-life`, `min_seen 0.5`, the
tree unbounded; life 10 000 on the sample corpus, 100 000 on the prose.

### A habit forms

Say `'the'` ten times over, to the end of a text. Each saying traverses what
it says, so the text said becomes more probable and the branch it is decided
at sharpens; asked quietly ten times, nothing moves; sampled at temperature
1, a habit forms by chance — and then feeds itself.

| arm | says | probability at first → last | distinct texts | top share at the branch | entropy there |
|---|---|---|---|---|---|
| said (10×) | `the bear sleeps through the winter` | 0.027 → **0.224** | 1 | 0.15 → 0.34 | 3.76 → 3.38 bits |
| asked quietly (10×) | the same | 0.027 → 0.027 | 1 | 0.15 → 0.15 | 3.76 → 3.76 bits |
| sampled at T = 1 (30×) | `the doctor heals the sick`, … | 0.025 → 0.043 | 15 | 0.15 → 0.17 | 3.76 → 3.68 bits |

The branch before: `'e b' 0.15, 'e c' 0.11, 'e f' 0.10, 'e do' 0.09, …`;
after ten sayings: `'e b' 0.34, 'e c' 0.09, 'e f' 0.08, 'e do' 0.07, …`.
On the prose corpus the same ten sayings take the text said from 0.067 to
0.467 and the branch from 0.81 to 0.52 bits.

A saying and a reading weigh the same: one traversal each, on the option
they pass through. Read one continuation twice and another twice, say the
first once, and the first wins; read the second twice more and it wins; say
it twice and it holds against two more readings of the first
(`tests/test_radixdecay.py`, `test_reading_and_saying_weigh_the_same`).

### A memory fades

Read the corpus once, then let time pass. The reading itself is 6 845
traversals — 0.68 lives — so a model that has just finished already
remembers its first text less than its last.

`half-life`:

| lives since the reading | nodes remembered | texts recited | bits / char | misses |
|---|---|---|---|---|
| 0 | 2224 / 2224 (100%) | 57 / 60 | 0.20 | 0% |
| 0.25 | 100% | 57 / 60 | 0.20 | 0% |
| 0.5 | 77% | 42 / 60 | 1.17 | 1% |
| 1 | 29% | 0 / 60 | 2.98 | 2% |
| 2 | 13% | 0 / 60 | 4.71 | 3% |
| 4 | 1% | 0 / 60 | 7.49 | 4% |
| 8 | 0% | 0 / 60 | 8.67 | 3% |

`linear` loses a visit's worth every life outright, so it is already
forgetting during the reading (77% remembered, 42 recited, at 0 lives) and
its misses climb to 81% by eight lives, where the half-life's stay near 3%:
under a half-life an option is never *gone*, only faint, and a text is
scored as a bag of grams from the root rather than as unknown. `none` is the
count model back: 100%, 57 / 60 and 0.20 bits at every row.

On the prose corpus (800 texts, life 100 000) the reading is 298 146
traversals — three lives — and the curve starts from a model that has
already forgotten most of what it read: 51% of nodes remembered and 232
texts recited at the end of the reading, 22% and none a life later, 2% and
4.93 bits at four lives; with `none`, 705 of 800 recited and 0.14 bits.
**A life shorter than a reading cannot hold the reading.** `DESIGN.md` §10
has the clock's arithmetic.

### Use keeps a memory

After reading, twenty silences of a tenth of a life each, and in each one
the model says one text — the first one it recites after reading — from its
first eight characters. Two lives pass in all. Said, the text is refreshed
by its own saying and survives while the rest of the corpus fades; said
quietly, it goes with the rest.

| arm | lives elapsed | traversals said | the used text still recited | recited | nodes remembered |
|---|---|---|---|---|---|
| said | 2.0 | 40 | **yes** | 1 / 60 | 13% |
| said quietly | 2.0 | 0 | no | 0 / 60 | 13% |
| silent | 2.0 | 0 | no | 0 / 60 | 13% |

Forty traversals — two per saying — keep a text that six thousand
traversals of reading could not keep past two lives: what is used is what
is kept. The prose corpus gives the same three rows (80 traversals; yes, no,
no; 10% remembered).

### Reading brings it back

Read, fade four lives, re-read every third text once, then every text once.
One reading restores a text whole, and only the texts re-read come back.

| stage | nodes remembered | texts recited | of the 20 re-read | of the other 40 | bits / char |
|---|---|---|---|---|---|
| read once | 100% | 57 / 60 | 18 | 39 | 0.20 |
| four lives later | 1% | 0 / 60 | 0 | 0 | 7.49 |
| every third text re-read | 40% | 19 / 60 | **19** | **0** | 4.41 |
| all re-read | 100% | 57 / 60 | 19 | 38 | 0.20 |

On the prose corpus the re-read third comes back better than it was after
the first reading — 232 of the 266 recited, against 80 when they had been
read as part of a three-life stretch — while none of the other 534 do;
re-reading all 800 takes three lives again, so the model ends that reading
as it ended the first, with 246 texts recited and the early ones already
fading.

## The clock, in numbers

Everything fades against `traversals`, so a life is worth what is put on
the clock:

| event | traversals |
|---|---|
| reading one 30-character text for the first time | ≈ 114 |
| reading the 60-text sample corpus | 6 845 the first time, 7 550 the second (the tree is split more finely by then) |
| reading the 800-text prose corpus | ≈ 298 000 |
| saying a 36-character text from START | 6 |
| saying a text token by token under a window of four grams | ≈ 35 |
| asking anything quietly | 0 |

Reading is expensive because every suffix window is a walk; saying is cheap
because a run is one arrival. A life of 10 000 is a reading and a half of
the sample corpus and a thirtieth of one reading of the prose. Pick `life`
against the corpus and against how the model will be used; `--life`,
`--decay` and `--min-seen` are on every command.

## What is different from the siblings

| | `RadixCyclicNN` count model | `RadixCyclicNN/SPEC-EdgeDecay.md` | `RadixAcyclicNN` | this |
|---|---|---|---|---|
| structure | cyclic graph over grams | the graph | acyclic tree of contexts | the tree |
| memory | `seen` counts that only grow, a learned weight beside them | weights decaying on graph time | learned weights and activations, `count` for trust | `seen` alone |
| what a query does | nothing | nothing — "never from a query" | nothing | traverses what it says |
| what decays | nothing | the weights, never the counts | nothing | `seen` itself |
| recovery | — | by training | by training | by any traversal: a reading, a saying |
| the clock | epochs | graph time | epochs | traversals anywhere |

## The pieces

| file | what it holds |
|---|---|
| `radixdecay/encoding.py` | grams over characters, phones or syllables; spelling back — the sibling's |
| `radixdecay/tree.py` | `DecayTree`: the suffix tree, the memory (`seen`, `touch`, `tick`, the decays, shares, the floor), split and merge, `locate`, the invariants, persistence; `Walks`, the incremental cursors the token-by-token walk sees with |
| `radixdecay/search.py` | `cheapest_path`: Dijkstra by `−log share`, visiting without arriving |
| `radixdecay/model.py` | `DecayNet`: `read`, `say` / `predict`, `cheapest`, `generate`, `tick`; the quiet measurements `score`, `bits_per_unit`, `next_token_accuracy`, `recites`; save and load |
| `radixdecay/experiment.py` | `habit`, `forgetting`, `use`, `recovery`; `run` and `tables` |
| `radixdecay/corpus.py` | reading a corpus file, splitting one, the digest |
| `radixdecay/cli.py` | `python3 -m radixdecay`: read, say, cheapest, generate, score, accuracy, stats, tick, experiment, demo, test |
| `tests/test_radixdecay.py` | 65 tests (`tests/README.md`) |
| `data/`, `results/` | the corpora and the measured numbers, each with a README |

## Running it

```bash
make help                 # every target and the defaults
make demo LIFE=2000       # a shorter life
make experiments DECAY=linear
make read DATA=data/corpus.txt LIFE=100000 MODEL=prose.json.gz
make say PREFIX="the signal" TIMES=5 MODEL=prose.json.gz     # said five times, and saved
make score MODEL=prose.json.gz DATA=data/corpus.txt
make tick LIVES=3 MODEL=prose.json.gz
python3 -m radixdecay say --unit phone "the cat" --to-end
python3 -m radixdecay accuracy --heldout-every 5 --windows 1,2,4,all
```

Every command that says something changes the model; `--save` writes it
back. `--quiet` asks without arriving.

## Limits

* `seen` is a float and a half-life never reaches zero: an option read once
  is offered forever, at a vanishing share. `min_seen` decides what is
  consulted; `linear` decides what is offered.
* The clock is global to one tree; two saved models fade independently.
* A saying refreshes only what it arrives at — the deepest context it
  consults and the options it takes — so after sayings the model can know a
  text from its beginning and not from its middle (`DESIGN.md` §6).
* Like the acyclic sibling, it unrolls repetition and does not generalise
  what the cyclic graph's cycles generalise. The graph with this memory is
  the next model; this directory measures the memory alone.

## Where it sits

| directory | what a node is | what it is for |
|---|---|---|
| `RadixCyclicNN/` | a gram; repeats become cycles | the model, built out |
| `RadixAcyclicNN/` | a context; no cycles | the same model as a tree, and what that measures |
| `RadixDecayNN/` (this) | a context; no cycles; no parameters | the tree with `seen` as its only memory, fed by reading and saying alike, fading on its own clock |
