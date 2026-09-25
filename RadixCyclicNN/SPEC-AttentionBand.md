# The attention band — where inside a gram a correction lands

**Status** Built, in all three implementations: `radixnet/attention.py` (with
`GraphModel._charged_steps` in `radixnet/model.py`), `go/radixnet/attention.go`
and `rust/src/attention.rs`. Off by default everywhere; switched with
`radixnet attention --blur X` (`radixnet-count attention`, the Rust
`radixnet attention`), `POST /api/model/attention` on all three servers, and the
Attention band card on the frontend's Model settings tab. D-086 is the decision.

**Answers** *"add an attention adjustment band for each n-gram, rewarding the
centre of the window more than its beginning and end - the way the eye focuses
on the centre and blurs the left, the right, and the lines above and below
while it reads."*

---

## 1. The problem, precisely

A correction (D-046, `DESIGN.md` §16.4) aligns what the network wrote with what
the teacher wrote and marks the **units** that differ. Something must then decide
which steps of each path answer for them. The rule has been: a step answers, in
full, for the units it **writes** - the first step for its whole label, every
later one for the units past its overlap.

```
the cat sat  ->  the bat sat          the changed unit is 'c', unit 4

grams    the  he␣  e␣c  ␣ca  cat  at␣  t␣s  ␣sa  sat
writer              1                                     <- 'c' at the END of "e␣c": only just reached
```

Under a sliding window every unit is seen by up to `n` grams, at every position
from the last to the first. The writer is the gram that sees it at its **edge** -
the newest unit, the one the eye has only just reached. The gram that has it at
its **centre** (`␣ca`) sees it best, and is never charged at all.

## 2. The band

A gram is a fixation of `n` units. The band says how sharply each position is
seen: **1 at the centre, `1 - blur` at the first and the last unit, linear
between**.

```
weight(j) = 1 - blur * (|j - c| / c)        c = (n - 1) / 2,   j = 0 .. n - 1
```

| n | blur 0.5 | blur 1 |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 0.5 0.5 | 0 0 |
| 3 | 0.5 1 0.5 | 0 1 0 |
| 4 | 0.5 0.8333333333333334 0.8333333333333334 0.5 | 0 0.6666666666666667 0.6666666666666667 0 |
| 5 | 0.5 0.75 1 0.75 0.5 | 0 0.5 1 0.5 0 |

A gram of one is all centre, a gram of two all ends: both are flat whatever the
blur, as every band is at blur 0. The formula is computed in exactly that order
- `d = |j - c|`, then `d / c`, then `blur * that`, then `1 - that` - one
correctly rounded IEEE operation at a time. Go converts the product explicitly
(`1 - float64(blur*(d/c))`) so the compiler cannot fuse it into a multiply-add.

`blur` is a finite number in `[0, 1]`. The band is **off** (`None`, the zero
value) until it is switched on; `0.5` is what switching it on means when no blur
is given.

## 3. The rule

### 3.1 The marked units

Every unit of every half-open span `[lo, hi)` of the diff; for an **empty** span
- an insertion - the unit it stands in front of (`[lo, lo + 1)`): the step that
walked past the position is the one at fault. Units are clamped to `[0, length]`,
and `length` itself - the position after the last unit - is the end marker.

### 3.2 Sharing a unit out

The text has `G` grams; gram `g` covers `[g * stride, g * stride + n)`. For each
marked unit `u < length`, ascending:

* its viewers are the grams `g` with `lo <= g <= hi`, where `lo = 0` if `u < n`
  else `(u - n) / stride + 1`, and `hi = min(G - 1, u / stride)` (integer
  division); none (the tail a grouping encoding drops) means nobody is charged;
* viewer `g` sees it at weight `w_g = weight(u - g * stride)`; `total` is their
  sum, left to right;
* each viewer's **share** grows by `w_g / total` - or `1 / viewers` when `total`
  is 0 (every viewer sees it at an end the band gives nothing);
* each viewer whose `w_g` equals the largest is a **focus** of the grams.

The end marker `u = length` is not shared: the step into END answers for it in
full.

**Three properties follow, and the tests pin them** (eight encodings, four blurs):

1. **One unit, one charge.** A single marked unit's shares add up to exactly 1 -
   as loud as the writer rule, spread rather than multiplied.
2. **A whole text is left alone.** When every unit is marked, every gram's share
   is at least 1 - so, capped, every step is charged in full, exactly as without
   a band. A thumbs up, a thumbs down, 2NRL and training mark every unit alike and
   so never meet the band at all.
3. **One viewer takes it all.** A unit that one gram sees - the first and last
   units of a text, every unit under a grouping encoding or a gram of one - is
   charged to that gram in full, as its writer was.

### 3.3 From grams to steps

A traced path is `START, n1, ..., nk, END`. The step into node `n_i` owns the
text's grams from where the previous node's end left off: a node whose label is
`L` units long holds `(L - n) / stride + 1` grams. A step's **charge** is the sum
of its grams' shares, in gram order, **capped at 1**; it is a **focus** step when
any of its grams is a focus. The step into END is charged 1, and is a focus, when
the end marker is marked. Steps charged 0 are left out.

With the band off, the steps and charges are the writer rule's - every charge 1,
every step a focus - so every number downstream is what it was.

## 4. What a charge moves

**The count model** (`correct(wrong, right, strength, weight, reward, keep, count)`):

| | band off | band on |
|---|---|---|
| an edge's charge | 1 if a step of it wrote a changed unit | the largest charge any of its steps took |
| penalty on a step of `wrong` | `strength * weight` | `strength * weight * charge` |
| reward on a step of `right` | `strength * reward`, else `* keep` | `strength * reward * factor`, `factor` = 1 for charge 1, `charge + (1 - charge) * keep` for a partial charge, `keep` for none |
| path verdict (`mark_steps`) | every blamed / taught step | the focus steps |

The penalties are applied one `add_reward` per distinct amount, first seen first;
an edge the correction also rewards is never penalised. The lengths are the texts'
lengths **in the encoding's units** - a word model's sentence that stopped too
early is an insertion at its word count, which Python's count model had measured
in characters.

**The negative network** (`correct(wrong, right, reason, severity, ...)`): each
charged step's edge is blamed `severity * charge` (`record_failure(..., shares)`)
and still counts one failure; `blame_total` adds what was laid, in step order.

## 5. The file

```json
"graph": {"format": "radixnet-graph", "format_version": 4,
          "encoding": {"unit": "word", "n": 3, "stride": 1},
          "attention": {"blur": 0.5},
          "seed": 0, ...}
```

Written only while the band is on, right after `encoding` - or after
`format_version` when the encoding is the default - which is where Python and
Rust write it, byte for byte (Go writes the same keys; its documents are compared
as parsed JSON). A document without the block, or with `"blur": null`, is off. A
blur outside `[0, 1]` is refused on load. `stats()` carries `attention_blur`
(null while off) for the count model and the negative network.

## 6. The surfaces

| | |
|---|---|
| CLI | `attention` (show) · `--on` · `--blur X` · `--off` (change, then save; not with `--dry-run`) · `--wrong TEXT --right TEXT` (where one correction lands, both rules) · `info` has an `attention` row |
| API | `GET /api/model/attention` → `{kind, attention: {on, blur, weights, ngram, stride, unit, units, applies, default_blur}}` · `POST /api/model/attention {on, blur}` → `{kind, attention, stats}` · `POST /api/model/attention/preview {wrong, right, blur}` → `{kind, attention, blur, weights, changes, wrong, right}`, each side `{text, units, grams, spans, writer, charges, focus, end}` · `GET /api/model` carries `attention` |
| frontend | the Attention band card on Model settings: on / off, the blur slider, the band drawn over one gram, and a correction previewed under both rules at the slider's blur |

The preview needs only the encoding - each gram shown as if it were a step of its
own - and changes nothing. A kind that is never corrected (the sine model, the
phase model) reports `applies: false` and refuses a band: 400 from the servers,
an error from the CLI.

## 7. Tests

| test | pins |
|---|---|
| `tests/test_attention.py` | the band's shape; `spread`'s three properties; off is `_steps_over`; compressed nodes capped; the centre takes most of the penalty and the whole verdict; `keep` tops up a partial charge; a word model blames END; the negative network blamed by share; off is the old file to the bit; the file block; the refusals; the CLI; the API |
| `go/radixnet/attention_test.go`, `go/server/attention_test.go` | the same, in Go |
| `rust/src/attention.rs` (unit tests) | the same, in Rust |
| `frontend/test/attention.test.mjs` | the frontend's band: the same doubles |
| `tests/test_go_parity.py` | Go against Python: the same previews, the same rewards to the bit, the same paths, the same blame; a word model with and without the band |
| `tests/test_rust_parity_tools.py` | Rust against Python: the same previews, the same totals, the same model files byte for byte (count, word, negative); the same answers from the three routes |

## 8. What this does not do

* **It does not touch whole-text feedback** - property 2 - nor training, nor the
  search. It is inert everywhere a text is judged as a whole.
* **It does not blur recognition.** An unknown gram is still unknown; reading it
  as the known gram that agrees at the centre is a different feature (Q-19).
* **It has no up and down.** A text is one line.
* **It does not say which blur is right.** Nothing here is graded; the preview
  shows both rules side by side so the question can be asked.

## 9. Alternatives rejected

See D-086: charging every viewer in full by its weight (louder, and no
gradation); a gram's charge as the weighted fraction of its window that is marked
(under-charges the ends of a text); a Gaussian band (`exp` is not bit-identical
across the three languages); a field of the encoding (the encoding is fixed for a
graph's life, the band is not).
