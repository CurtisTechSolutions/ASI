# Search and training methods — the contract all three ports keep

**Status** Built, in all three implementations: Python (`radixnet/search.py`,
`radixnet/beam.py`, `radixnet/training.py`), Go (`go/radixnet/search.go`,
`go/radixnet/beam.go`, `go/radixnet/training.go`) and Rust
(`rust/src/search.rs`, `rust/src/beam.rs`, `rust/src/training.rs`). Every
setting here is **off by default** and changes nothing when it is off: the
draw, the beam and the training pass are then exactly the ones the model had
before, and a model file is byte for byte what it was.

**Answers** two requests at once: *more ways to search* — the sampling filters
language models are usually driven with, and a beam that does not hand back
five spellings of one sentence — and *more ways to train* — a curriculum, a
rehearsal of what the model read before, and a run that stops when it stops
learning. A later request added a third: *training on a file in reverse*, so
the model learns what comes before a text, and asking it backwards from the
frontend (§9). A fourth asked for *auto-temperature*: a search that finds no
answer raises the temperature and tries again (§10).

This document is the contract, and it is precise on purpose. The three ports
are held to the same graph, the same file and the same prediction by the
parity suites (`../ModelKit/tests/test_go_parity.py`, `../ModelKit/tests/test_rust_parity.py`), so
every rule below says what is compared, in what order, and how a tie breaks.

---

## 1. Sampling filters

A stochastic walk (`mode: "sample"` on `predict` and `generate`) picks every
step from the options a node offers, with weights

```
w_i = exp(-(cost_i - lowest) / temperature)          lowest = min cost_i
```

and one draw: `r = random() · fsum(w)`, then the first option whose running
sum passes `r`. Three filters now narrow the options **before** that draw.

| setting | range | off | what it keeps |
|---|---|---|---|
| `top_k` | integer `>= 0` | `0` | the `top_k` cheapest options |
| `min_p` | `0 <= min_p < 1` | `0` | the options with `w_i >= min_p` — a probability at least `min_p` times the best one's |
| `top_p` | `0 < top_p <= 1` | `1` | the smallest set of cheapest options whose weights reach `top_p` of the total (*nucleus* sampling) |

The rules that make three implementations agree:

1. **Rank.** Options are ranked by `(cost, position)` — cheapest first, and
   the option the node offered first where two costs are equal.
2. **Order of the filters.** `top_k`, then `min_p`, then `top_p`, each on what
   the previous one kept.
3. **`top_p` exactly.** `total = fsum(w)` over the options still kept;
   `acc` adds the kept weights one at a time **in rank order** with ordinary
   floating-point addition; the option at which `acc >= top_p · total` first
   holds is the last one kept.
4. **The draw is unchanged.** The kept options are drawn from **in the order
   the node offered them**, not in rank order, with the same weights and the
   same single `random()`. The cheapest option always survives every filter,
   so `lowest` — and with it every weight — is the same number it would have
   been unfiltered. With all three filters off the walk is therefore the walk
   it always was, draw for draw.
5. **Greedy is untouched.** `temperature = 0`, or a single option, takes the
   cheapest option before any filter is consulted.
6. **One draw per step.** Whenever the temperature is positive and the node
   offered more than one option, the step consumes exactly one `random()` —
   however few options the filters left, even one. The generator stays in the
   same place in all three ports whatever the filters did.

Under the least-punished traversal the filters run on what `least_punished()`
left, exactly as the draw does. Beam and Dijkstra ignore them, as they ignore
the temperature.

## 2. Diverse beam

The beam's top side hands back the K cheapest finished paths, and in a
compressed graph the cheapest few are often one text with its ending varied —
`capital of frog chased degrees`, `capital of frog chased degree `.
**`diversity`** (`>= 0`, off at `0`) trades a little likelihood for
novelty when the K are chosen, by *maximal marginal relevance*:

1. **A pool.** The top beam keeps up to `max(k, width)` finished paths
   instead of `k` — the same bounded heap, larger — and its early exit waits
   for the pool to fill rather than for `k`. The search itself, the frontier
   it keeps and the paths it expands, is otherwise unchanged.
2. **The first pick is the best path.** The pool is sorted in the plain
   beam's order (`(punish, cost, entry)`) and its first path is always kept.
3. **Every later pick** takes the path with the smallest

   ```
   (punish,  cost + diversity · overlap,  position in the pool)
   ```

   where `overlap` is the largest `path_overlap` with a path already picked:
   the number of leading nodes the two share (after the common start node,
   which does not count) divided by the length of the shorter one, in nodes.
   A path that only varies the ending of one already chosen pays nearly the
   whole `diversity`; one that parts from all of them at the first step pays
   nothing.
4. **The picks come back in the order they were picked.** Costs are not
   touched — a path still reports `-log p` plus its step penalty, and its
   probability — so with `diversity > 0` the list is no longer sorted by cost
   after the first entry: that is the trade it was asked to make.

Only the top side is diversified. The bottom beam — the K least likely
continuations — answers a question about what the model finds strange, and
spreading it out would answer a different one. The resonant model's phase
beam applies the same pick to its finished paths.

## 3. Training order and the curriculum

Every epoch of a training run walks a list of texts. The list is built from
the run's texts (after the ones too short for one gram are dropped) by two
settings.

**`order`** — the order of the list:

| value | the list |
|---|---|
| `corpus` (default) | as given |
| `shortest-first` | by length in the encoding's units, rising; ties in corpus order |
| `longest-first` | by length, falling; ties in corpus order |
| `shuffle` | a fresh permutation every epoch, by the keys of §5 |

**`curriculum`** (`0 < c <= 1`, off at `1`) — how much of the list an epoch
sees. Epoch `j` of a run of `E` epochs (counting from 0) takes the first

```
m_j = clamp(ceil(frac_j · n), 1, n)      frac_j = 1                          when E = 1
                                         frac_j = c + ((1 - c) · j) / (E - 1) otherwise
```

texts of the list, evaluated left to right in IEEE doubles. The run starts on
the first `c` of the list and ends on all of it: with `shortest-first` that is
the *baby steps* curriculum, the short texts first and the long ones added a
step at a time.

The structure pass is not affected: every text of the run is observed before
the first epoch, as before, so the graph a curriculum trains is the graph the
whole corpus builds. Only what each epoch *counts* changes. For the count
model the order also decides what the sliding window remembers, which is the
point: the end of an epoch is what "recent" means.

## 4. Replay

A run can rehearse what the model read in earlier runs. The model keeps a
**replay buffer** — a bounded, uniform sample of every text it has ever been
trained on — and a run with `replay > 0` walks some of it after the new texts
of every epoch.

**The buffer.** `replay_size` (integer `>= 0`, off at `0`) is its capacity and
is a property of the model: a run that gives it sets it (a smaller size drops
the entries with the largest priorities), and a run that does not leaves it
alone. After a training run finishes, every text of that run is offered to the
buffer in corpus order: the `g`-th text ever offered (`g` counting from 0,
across the model's life) gets the priority `replay_key(seed, g)` (§5), and the
buffer keeps the `replay_size` entries with the **smallest** priorities, ties by
`g`. That is bottom-k sampling: every text ever offered has the same chance of
being in it, whatever the order the corpus came in, and no random number is
drawn, so the model's generator is untouched.

**The rehearsal.** `replay` (`>= 0`, off at `0`) is how much of it an epoch
walks, as a fraction of the run's new texts:

```
r = min(len(buffer), floor(replay · n + 0.5))
```

texts, taken from the buffer (in priority order) starting at position
`(j · r) mod len(buffer)` and wrapping, so successive epochs walk successive
slices of it. They are appended **after** the epoch's new texts: the epoch ends
by refreshing old material, which is what the count model's sliding window will
remember. The buffer is read as it stood when the run began, so a run never
rehearses its own texts. When any are rehearsed, the structure pass observes
the whole buffer, in priority order, after the run's own texts. A replayed text is counted into the graph like any
other, but it is not new: it adds nothing to `trained_texts` or `trained_chars`.

**The file.** The buffer is saved with the model, at the end of the model
document, and **only when `replay_size > 0`** — an ordinary file is unchanged:

```json
"replay": {"size": 256, "seen": 1234, "index": [17, 3, ...], "texts": ["...", "..."]}
```

`seen` is how many texts were ever offered; `index` and `texts` are the kept
entries in priority order, `index[i]` the `g` of `texts[i]`. A reader
recomputes the priorities from the model's seed and the indices and sorts by
them; it does not trust the order it was given.

Replay, the order and the curriculum belong to plain training — `train` — and
to the kinds that learn by walking a list of texts: the count, resonant and
sine models. Reward, punishment, 2NRL and the negative network's blame keep
the passes they had. So does every training call stamped with a feedback
phase (`"negative"` or `"positive"`): the sine model's 2NRL and thumbs, and the
agent's punishment, train through `train` with one, and such a pass walks every
text in corpus order and neither reads the buffer nor offers it a text — a
punished text is not one to rehearse.

## 5. The keys

Both the shuffle and the replay buffer need an order that looks random and is
the same number in three languages. They use SplitMix64, on unsigned 64-bit
integers, with wrapping arithmetic:

```
splitmix64(x):  z = x + 0x9E3779B97F4A7C15
                z = (z ^ (z >> 30)) · 0xBF58476D1CE4E5B9
                z = (z ^ (z >> 27)) · 0x94D049BB133111EB
                return z ^ (z >> 31)

u64(x)                      = x mod 2^64   (a negative seed in two's complement)
shuffle_key(seed, epoch, i) = splitmix64(splitmix64(splitmix64(u64(seed) ^ 0x53485546464C45) ^ u64(epoch)) ^ u64(i))
replay_key(seed, g)         = splitmix64(splitmix64(u64(seed) ^ 0x5245504C4159) ^ u64(g))
```

`seed` is the model's seed. For `shuffle`, `epoch` is the number the epoch's
record will carry (`epochs_total` after it), and the list is sorted by
`(shuffle_key(seed, epoch, i), i)` where `i` is the text's corpus position —
so a shuffled run continues differently from where it stopped rather than
repeating its first epoch's order.

## 6. Early stopping

**`patience`** (integer `>= 0`, off at `0`) and **`min_delta`** (`>= 0`)
end a run that has stopped improving. After every epoch:

```
if loss < best - min_delta:   best = loss;  stale = 0
else:                         stale = stale + 1
if patience > 0 and stale >= patience:  stop after this epoch
```

with `best` starting at `+inf`. An epoch that walked only part of the list —
one still inside the curriculum — neither improves nor tires the run: its loss
is over fewer texts and is not comparable, so it is skipped. The record of the
epoch that stops the run carries `"early_stop": true`, and no other record
gains a key.

Under the linear pacing of §3 an epoch walks the whole list only once
`ceil(frac_j · n)` reaches `n` — for a long list, the last epoch — so a run
with a curriculum is watched only from there on, and `patience` is meant for
runs without one (or with a curriculum close enough to 1 that it reaches the
whole list early: 20 texts at `c = 0.9` over 6 epochs walk 18, 19, 19, 20, 20
and 20).

## 7. Where the settings go

| | Python | Go | Rust | HTTP | CLI |
|---|---|---|---|---|---|
| `top_k`, `top_p`, `min_p` | `predict`, `generate` | `PredictOptions`, `GenerateOptions` | `PredictOptions`, `GenerateOptions` | `/api/predict`, `/api/generate` | `--top-k`, `--top-p`, `--min-p` |
| `diversity` | `predict`, `generate` (beam) | the same | the same | the same | `--diversity` |
| `order`, `curriculum` | `TrainConfig` | `TrainOptions` | `TrainOptions` | `/api/train` | `--order`, `--curriculum` |
| `replay`, `replay_size` | `TrainConfig` | `TrainOptions` | `TrainOptions` | `/api/train` | `--replay`, `--replay-size` |
| `patience`, `min_delta` | `TrainConfig` | `TrainOptions` | `TrainOptions` | `/api/train` | `--patience`, `--min-delta` |
| `reverse` | `TrainConfig` | `TrainOptions.Plan.Reverse` | `Plan.reverse` (in `TrainOptions`, the sine model's `TrainConfig`) | `/api/train` | `--reverse` |
| `auto_temperature` | `radixnet.autotemp` (`predict`, `generate`, `climb`); the kit's `guarded_predict`, `NegativeFilter.generate`, `dialogue.reply` / `converse`, `Ask` | the same, in the core and the kit | the same, in the core and the kit | `/api/predict`, `/api/generate`, `/api/converse` (and `/stream`), `/v1/messages`, `/v1/chat/completions` | `--auto-temperature` on `predict`, `generate`, `converse`, `talk` |

A value out of range is an error with the same meaning everywhere: `top_k < 0`,
`top_p` outside `(0, 1]`, `min_p` outside `[0, 1)`, `diversity < 0`, an
unknown `order`, `curriculum` outside `(0, 1]`, `replay < 0`,
`replay_size < 0`, `patience < 0` and `min_delta < 0` - and, over HTTP, a
`reverse` that is not a boolean, and an `auto_temperature` that is not a
finite number `>= 0`.

A streaming source (the Go and Rust readers of large files and archives) is
read into memory first when a run asks for a non-corpus order, a curriculum or
replay: each needs the whole list before the first epoch. Reading backwards
does not: every text is turned around as it streams in (§9).

`GET /api/status` reports the buffer on all three servers: `"replay": {"size",
"texts", "seen"}` - its capacity, how many texts it holds, how many were ever
offered - or `null` when the model keeps none.

## 8. What the frontend does with them

The **Settings** tab keeps the search and training settings of this browser —
the traversal, the sampling filters and the diversity, the auto-temperature
ceiling (§10), whether a query is asked backwards (§9), and how a run walks its
texts — the defaults every tab starts
from, shared by Predict, Generate and Train the way the traversal already was
(`../ModelKit/frontend/src/hooks/useSiteSettings.jsx`, the rules in
`../ModelKit/frontend/src/settings.js`). A tab sends a setting only when its mode reads it
and it is not off, and refuses to send one out of range. The
**Model settings** tab holds what belongs to the model and is saved with it:
the model's kind, encoding, size and replay buffer, a form that makes a new
model in any kind and encoding, the score function, and the encoder / decoder.

## 9. Reading backwards

**`reverse`** (a boolean, off by default) reads every text of a training run
backwards, so the model learns what comes *before* a text instead of what
follows it. It is the file read from its end to its start: with `whole_file`
(Go: `split: file`) each file is one text, turned around whole; one text per
line turns every line around.

**The turn.** A text is reversed in the units of the model's encoding, by
`Encoding.reverse` (Go `Encoding.Reverse`, Rust `Encoding::reverse`):

| unit | `reverse(text)` |
|---|---|
| characters | the code points in the opposite order - `"the cat"` → `"tac eht"`; a combining accent is a code point of its own, so turning a text around twice gives it back exactly |
| words | the words in the opposite order, written with single spaces - `"the  cat sat"` → `"sat cat the"` - each word's letters in their order; the words are cut on Unicode `White_Space`, as the word encoding cuts them (`SPEC-WordNGrams.md` §4) |

**Where.** The turn is the first thing a training call does, before anything
else sees a text: the texts too short for one gram are dropped from the
reversed list (the length is the same), the order and the curriculum walk it,
the structure pass observes it, `trained_texts` and `trained_chars` count it,
and the replay buffer is offered the reversed texts - the buffer keeps the
texts as they were read, so a later run, reversed or not, rehearses them as
they were learned. The order of the texts is not touched: `order` decides it,
as it always did. A run with `reverse` is therefore exactly a run over the
reversed texts: the same graph, history, counters and file, byte for byte, in
all three ports.

**Every kind.** The count, sine and phase models read their texts backwards,
and so does the negative network, whose training is blaming: a model that
reads backwards needs its failures backwards too. The turn belongs to the
training call, not to its phase: a call stamped with a feedback phase and
`reverse` reads its texts backwards as well - though no feedback route sets it.
Reward, punishment, 2NRL and the feedback routes take their texts as given.

**Off** changes nothing: the texts are the ones given, and no record gains a
key. The flag is not saved with the model; the CLI's `train` document reports
it in its `config` (`"reverse": false` when off), as the last field of the
config in Python and Rust alike.

**Streaming.** Go's readers stream a corpus of any size through training, and a
reversed run streams too: `ReversedSource` turns every text around as it is
read and keeps an archive's entries as parts that stream side by side, so
reading backwards costs no memory. Python and Rust train on the run's list of
texts, and turn the list around.

**Asking it.** A model trained backwards continues what it was given
backwards: the query has to be turned around too, and the answer turned back.
The servers leave that to the caller - `Encoding.reverse` is public in every
port - and the frontend does it (`../ModelKit/frontend/src/backwards.js`): with **Query
backwards** on (a site-wide setting beside the traversal), Predict and Generate
send the prefix turned around, in the model's units, and show every answer
turned back round, so a user types the end of a text and reads what precedes
it the right way round. A thumbs up or down goes to the feedback routes as the
model's own text, the backwards one, since that is the text feedback trains
on. The Train tab sends `reverse` for every kind from its **Read every text
backwards** box.

## 10. Auto-temperature

**`auto_temperature`** (a number `>= 0`, off at `0`) is a ceiling. A search
that finds **no answer** is asked again up a *ladder* of temperatures, a few
sampled walks at every rung, and the first answer ends it. Off, nothing
changes: one search, no extra draw, no record (D-078).

**Why the temperature, and how far.** A sampled step weighs its options
`p ** (1 / temperature)` (§1), so the temperature is the one dial that trades
how likely a walk is for how far it strays. Measured on a count model of 1,500
lines of prose, sampling 300 continuations of 60 prefixes:

| temperature | walks that reach END | walks that say nothing | nats per character |
|---|---|---|---|
| 0 (greedy) | 1.7% | 1.7% | 1.01 |
| 0.5 | 35% | 4.0% | 1.13 |
| 1 | 89% | 5.7% | 1.48 |
| 2 | 93% | 7.0% | 1.62 |
| 4 | 95% | 13% | 1.72 |

A cold walk goes round the graph's cycles and never ends; warming it up to 1
mends that. Past 2 it stops helping: END is reached no more often, an empty
answer gets *likelier* (a flatter step takes END sooner), and every
character is less likely. So the ceiling is the caller's choice, 2 is the
recommended one, and every rung is three draws rather than one: an empty or
vetoed walk is mended by another draw as often as by a hotter one.

**What is not mended.** Dijkstra, the beam and the resonant model's k-best read
no temperature, so for them the ladder is where *sampling* starts: the search
found nothing among the cheapest paths, and walks at rising temperature look
among the others. Where the graph has no way on at all, no temperature finds
one, and the record says so. A sampled walk keeps its cap (`length`, or
`max_length` when given), as it always has: a request for the end of a text
needs room to reach it.

### 10.1 The ladder

`temperature_ladder(start, ceiling)`: with `ceiling <= 0`, no rungs. Otherwise
the rungs are `start`, then `t <- min(max(2 * t, 0.5), ceiling)` while
`t < ceiling` - doubling, from `AUTO_FLOOR = 0.5` when the start is greedy, the
last rung the ceiling itself. A start at or above the ceiling is the one rung:
the ladder never cools a walk.

| `(start, ceiling)` | rungs |
|---|---|
| `(1, 2)` | `1, 2` |
| `(1, 3)` | `1, 2, 3` |
| `(0, 2)` | `0, 0.5, 1, 2` |
| `(0.7, 2)` | `0.7, 1.4, 2` |
| `(3, 2)` | `3` |

Every rung is `AUTO_DRAWS = 3` sampled searches, except a rung at temperature
0, which is one: a greedy walk is the same every time.

### 10.2 An answer

A search's result **answers** when its `text` holds more than whitespace and,
when the caller asked for the end of a text (`to_end`), it reached END.

A search that offered several continuations - a `Prediction`'s `top` - has
found an answer when any of them answers: the first that does, in the order
the search ranked them, becomes the prediction (`first_answer`). Its `text`,
`labels`, `node_ids`, `cost`, `step_costs`, `reached_end`, `full_text` and
`punish` replace the prediction's own; `expanded`, `top`, `bottom`, `k`,
`beam`, `mode` and `traversal` stay as the search left them. (The beam can rank
a path cut at the length cap above the ones that reached END - a walk round a
cycle is cheap - so a request for the end may find its answer second: that is
still the search's own, and nothing is walked.) The guard promotes the same
way: it re-ranks first (`rank`), and `first_answer` then reads what survived.

### 10.3 The climb

1. **The search as asked** runs first, exactly as it would have: its mode, its
   temperature, everything. When it answers, that is the whole of it.
2. **Then the ladder**, from the temperature the request carries (`1` when it
   carries none) to the ceiling: at every rung, its draws, each the same request
   in `sample` mode at the rung's temperature. The first that answers is
   returned.
3. **Nothing anywhere**: the search as asked is returned, unchanged.

The walks draw from the generator the request was already drawing from - the
model's own, or a conversation's seeded one - one search after the other, so
the climb is reproducible and the three ports draw alike. A **generation**
re-seeds per call (`seed` gives each generation a generator of its own), so the
`k`-th retry of a climb that was given a seed draws from `seed + k`: a retry at
the same temperature is a new draw.

**The record** of a climb is `{"ceiling", "climbed", "found", "temperature",
"tries"}`: the ceiling; whether the search as asked had to be retried; whether
an answer came back; the temperature of the walk that gave it (`null` when the
answer is the deterministic search's own, or there is none - and the asked
temperature when the search as asked was a sampled walk that answered); and how
many searches ran, the one as asked included. It is reported as
`auto_temperature` beside the answer whenever the ceiling is on, and never when
it is off.

### 10.4 Where it climbs

| path | the search | "no answer" |
|---|---|---|
| **predict** (CLI `predict`, `POST /api/predict`) | `model.predict`, through the guard when it is on (`guarded_predict`): every search is ranked, and its verdicts are added to the guard's report | `first_answer` of what survived finds none |
| **generate**, unguarded (`autotemp.generate`) | `model.generate` | fewer than `count` texts say anything beyond the prefix (the whole text is longer than the prefix, in the model's units). The ladder asks for as many as are still missing; what comes back is the texts that said something - the first generation's in its order, then the ladder's - possibly fewer than `count` |
| **generate**, guarded (`NegativeFilter.generate`) | the pair's over-sampled, filtered generation | fewer than `count` survivors say anything beyond the prefix. Each retry asks for the missing ones; a survivor whose text is already kept is skipped; `kept`, `rejected` and `verdicts` are concatenated, `candidates` and `asked` summed, `rate = len(kept) / candidates` |
| **a reply** (`dialogue.reply`: `converse`, today's format, chat, voice) | the whole reply - every context, the fresh start, the rethinks | the reply has nothing to say (`None`): every candidate empty or vetoed. A retry is the same reply in `sample` mode at the rung's temperature, so each one draws `k` walks from every context it tries. The turn carries the record (`Turn.auto_temperature`, in its `to_dict` only when set); `trace` hears `{"kind": "climb", "temperature", "draws"}` at the first retry of every rung |

**Today's format** reads `auto_temperature` from either dialect's body, beside
the dialogue's other dials (`Ask.auto_temperature`, in `Ask.to_dict` only when
it is not 0), and narrates the climb in the thinking:

```
nothing to say yet; trying again at temperature 1.00 (at most 3 replies)
found something to say at temperature 1.00, on try 2
```

- the first line for every `climb` event (`{t:.2f}`; `1 reply` when the rung
has one draw), the second when a climbed turn is spoken.

**The CLI** takes `--auto-temperature T` on `predict`, `generate`, `converse`
and `talk` (which sends it in the request body), and prints one line for the
climb; under `--json` the record is the document's `auto_temperature`, and a
conversation's turns carry their own.

