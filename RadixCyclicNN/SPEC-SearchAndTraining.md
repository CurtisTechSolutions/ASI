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
learning.

This document is the contract, and it is precise on purpose. The three ports
are held to the same graph, the same file and the same prediction by the
parity suites (`tests/test_go_parity.py`, `tests/test_rust_parity.py`), so
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

A value out of range is an error with the same meaning everywhere: `top_k < 0`,
`top_p` outside `(0, 1]`, `min_p` outside `[0, 1)`, `diversity < 0`, an
unknown `order`, `curriculum` outside `(0, 1]`, `replay < 0`,
`replay_size < 0`, `patience < 0` and `min_delta < 0`.

A streaming source (the Go and Rust readers of large files and archives) is
read into memory first when a run asks for a non-corpus order, a curriculum or
replay: each needs the whole list before the first epoch.

`GET /api/status` reports the buffer on all three servers: `"replay": {"size",
"texts", "seen"}` - its capacity, how many texts it holds, how many were ever
offered - or `null` when the model keeps none.

## 8. What the frontend does with them

The **Settings** tab keeps the search and training settings of this browser —
the traversal, the sampling filters and the diversity, and how a run walks its
texts — the defaults every tab starts from, shared by Predict, Generate and
Train the way the traversal already was (`frontend/src/hooks/useSiteSettings.jsx`,
the rules in `frontend/src/settings.js`). A tab sends a setting only when its
mode reads it and it is not off, and refuses to send one out of range. The
**Model settings** tab holds what belongs to the model and is saved with it:
the model's kind, encoding, size and replay buffer, a form that makes a new
model in any kind and encoding, the score function, and the encoder / decoder.
