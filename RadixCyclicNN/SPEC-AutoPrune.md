# Auto prune — the graph lets go of the edges nothing walks, and the nodes they strand

**Status** Built, in Python: `radixnet/prune.py` (the setting), `RadixCyclicGraph.prune /
prune_candidates / stranded_nodes / remove_edge / remove_node / edge_protected / edge_traffic` in
`radixnet/graph.py` with the kinds' overrides in `countnet.py`, `negative.py` and `resonance.py`,
and `GraphModel.prune_config / configure_prune / prune / _prune_epoch` in `radixnet/model.py`.
Off by default; switched with `radixnet prune --on`, `POST /api/model/prune`, and pruned by hand
with `radixnet prune --now`, `POST /api/model/prune/now`. **The Go and Rust ports do not prune**:
they keep the `auto_prune` block of a model file as they keep any block they do not read, and a
model pruned in Python loads in both as the smaller graph it is. D-094 is the decision.

**Answers** *"add an auto prune functionality to the RadixCyclicNN"* - the graph that
self-compresses (D-007) but never shrinks.

---

## 1. The problem, precisely

Everything the graph learns, it keeps. Compression (D-007, `DESIGN.md` §5) only ever coarsens
the structure: a unary chain becomes one node, and nothing is ever removed. So:

* a text read once leaves its edges behind for good, however rare its continuation is among
  its node's children;
* a split leaves a bridge that nothing crosses again once the text that caused it is gone;
* a text registered for scoring or for a path inversion (`observe_sequence(count=False)`) leaves
  nodes that were never walked at all;
* a split caused by a *new* text cuts an *old* node in two, and in a kind that counts edges and
  not nodes (the phase model, §4) the bridge between the halves starts at a count of 0.

A model taught for long enough carries every transition it ever saw, and every search prices
them all. **Auto prune** is the other half of self-compression: the graph lets go of what it
has stopped using.

## 2. The rule

Two thresholds over what the graph already counts, read in the kind's own currency
(§4), applied to every edge leaving a node:

| threshold | an edge is pruned when | default |
|---|---|---|
| `min_count` | it carries less than this much traffic | `1` - only what was never traversed; `0` switches the rule off |
| `min_share` | it takes less than this share of its node's out-traffic | `0` - off; it is what thins a busy node of its rare continuations |

The rules are independent: an edge under either is a candidate. A node nothing has traversed
has no shares, and only the count rule speaks there.

### 2.1 What is never pruned

An edge is **protected** while something was *taught* about it rather than observed - the same
evidence that keeps an edge out of compression (D-046, D-092):

| kind | protected edges |
|---|---|
| every kind | an edge into `BACK` or `THINK`: a lesson a voice learned by experience (D-068), which no corpus could teach again |
| count model | an edge with a reward or penalty (`edge_reward != 0`), or one a judged walk went over (a row in `paths`) |
| negative network | an edge carrying blame or clearing |
| phase model | an edge with a reward or penalty |
| sine model | nothing beyond the sentinels' edges: its feedback moves its weights and leaves no record on the edge, so the thresholds alone decide |

### 2.2 A count of 0 is unknown, not never

Every walk that enters a real node leaves it. So when a node's **in-traffic exceeds its
out-traffic**, the difference crossed an edge whose count was never written - the bridge of
a split in a kind that counts edges and not nodes, or that counts nothing but what a judge
said - and **none of that node's zero-count edges is a candidate**, since nothing says which of
them carried it. The comparison allows a float's last bit: a sum of shares in a different
order is the same sum. A registered-but-never-walked text gets no shelter from this: its
edges leave nodes whose in-traffic is accounted for, or nodes that have none.

### 2.3 The sweep

After the candidates go, every real node left with **no way in** (no parent but itself) or
**no way out** (no child but itself) goes too, with the edges it still had - a walk cannot
reach the first and can never end through the second. The sweep repeats until nothing is
stranded, since a node that goes can strand the next. A node holding a protected edge is never
swept: what was taught stays nameable, the way a blamed fragment does. A self-loop is no way in
or out.

A removed node's grams leave the index: the text is unknown to the graph again, and the next
`observe_sequence` that reads it creates fresh nodes for them (from the seeded stream, so a
pruned-and-retaught model is not byte for byte the untouched one; nothing promises that).

### 2.4 Afterwards

Removing edges opens unary chains, so the model **compresses** once after a prune (within the
dynamic window when it is on, and never through blame). The kinds that compute their weights
from counts recompute them; the count model also forgets the pruned edges' events in its
sliding window. Node and edge ids are tombstoned as a merge tombstones them, and compacted by
the file.

## 3. When it runs

* **By hand**: `radixnet prune --now`, `POST /api/model/prune/now`, `GraphModel.prune()`. At the
  model's thresholds, or the ones given for that prune alone, or the defaults while the setting
  is off - a prune by hand needs no setting.
* **Automatically**: at the end of every `every`-th training epoch (lifetime epoch count, so a
  model's third run is pruned on the same clock as its first) while the setting is on and
  `auto` is set - after the compression and the dynamic window's step the epoch already does,
  so what it removes is measured on the settled structure. Every kind's loop, feedback passes
  included, as the window steps there; the negative network's clearing passes add no structure
  and prune none away. The epoch's record carries `pruned_edges` and `pruned_nodes` whenever
  the prune ran, and its `merges` include the merges that followed.

## 4. The currency

`edge_traffic(e)` is what the thresholds read:

| kind | traffic |
|---|---|
| sine, count, phase | the edge's traversals, exactly (`edge_traversals`, across every reset of its counter) |
| negative network | the evidence it carries, blame plus clearing: it counts nothing else |

In the negative network evidence protects, so the thresholds only ever reach the edges that
carry none - the structure a judgement registered and never ruled on - and the bridge of a
split inside a blamed chain, which carries none either, is told from those by §2.2: the blame
that entered its node left over it.

## 5. The setting, the file, the surfaces

`AutoPrune(on, min_count, min_share, every, auto)` lives on the graph
(`RadixCyclicGraph.auto_prune`) beside the encoding, the band and the window, and like them it
may change at any time: the setting touches nothing, only a prune does. Off - `AutoPrune()`, the
zero value - nothing is written: **off is the old file to the bit**.

* **File**: `"auto_prune": {"min_count": 1, "min_share": 0.0, "every": 1, "auto": true}` in the graph
  document, only while it is on, right after `dynamic_window`. `stats()` does **not** report it:
  the stats are what every port answers alike, and only Python prunes - `prune_config()` is
  the setting's own view.
* **CLI**: `radixnet prune` shows it; `--on`, `--min-count N`, `--min-share X`, `--every N`,
  `--auto` / `--manual`, `--off` change it and save the model (not with `--dry-run`); `--now` prunes
  and saves; `info` has an `auto prune` row. `make prune` / `prune-on PRUNE_MIN_COUNT=1
  PRUNE_MIN_SHARE=0` / `prune-now` / `prune-off`.
* **API**: `GET` / `POST /api/model/prune` and `POST /api/model/prune/now`. `GET /api/model` and
  `GET /api/status` do not carry it, for the same reason the stats do not.
* **Frontend**: nothing yet - the Model settings tab does not show it.

Refused: a negative `min_count`, a `min_share` outside `[0, 1)`, an `every` under 1, a boolean
where a number goes; `--off` with any other setting; `--auto` with `--manual`.

## 6. Explicitly not specified

* **No decay.** Pruning removes; it never fades. An edge's weight, reward or blame is what it
  was until the edge goes (`SPEC-EdgeDecay.md` is the fading).
* **No pruning by weight or by cost.** A learned weight is an opinion; a count is a record. The
  thresholds read the record.
* **No clock.** An edge is measured by how much went over it, not by when: a count model's
  sliding window is the nearest thing to recency the graph has, and it is one kind's.
* **No port.** The Go and Rust ports read the block and leave it alone.

## 7. Tests

`tests/test_prune.py` (the model), `../ModelKit/tests/test_prune.py` (the CLI and the HTTP API).
