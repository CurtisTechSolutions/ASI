# Word n-grams — the same graph over a different alphabet

**Status** Built, 2026-09-20. `DESIGN.md` §34 is what was built and
`DECISIONS.md` D-075 is why; this document stays as the argument the two were
written from, and §10's table is the test suite
(`tests/test_encodings_end_to_end.py`, `go/radixnet/words_test.go`,
`rust/tests/words.rs`, `rust/tests/encodings.rs`, and the encoding classes of
the two parity suites).

**Two designs, one feature.** §1 below was written once and reached twice, at
the same time and independently: once as a **word model kind** carrying a
vocabulary that maps each word to one code point (D-073), and once as a
**unit of the encoding dial** (D-075). The second won, and this document
describes it. D-073 is kept in `DECISIONS.md` because the argument in it is the
argument this rests on, and because the reversal is information.

**Answers** *"what about word n-grams?"* — and answers it without touching the
graph, the weight function, the search or any of the three ports' structural
code.

---

## 1. The claim this rests on

The model's input decision (D-006) is about **what a symbol is**: characters,
not bytes and not tokens, in a sliding window of three with stride 1. Every
structural rule in the graph is stated in terms of that window and nothing else:

* a node's label is a sequence of **units**, at least `n` of them;
* an edge `p -> c` exists only where the last `n - stride` units of `p` are
  the first `n - stride` of `c`;
* the index maps an `n`-unit key to `(node, offset)`;
* compression merges a unary chain by overlapping the two labels at that seam;
* decoding walks the path and takes each label's units past the overlap.

**Not one of those rules mentions a character.** They are true of any alphabet.
So a word n-gram model is not a different model — it is this model over an
alphabet whose symbols are words:

```
"the cat sat on the mat"
  characters →  ["the", "he ", "e c", " ca", ...]        the alphabet is ~100 symbols
  words      →  ["the cat sat", "cat sat on", ...]       the alphabet is the corpus
```

The compression then does to word chains what it already does to character
chains. Today a recurring word becomes one node — the sample corpus produces
labels like `'lays eggs'` and `'s celsius'` out of nothing but unary merging.
Under a word encoding the same merge produces a node whose label is a
**phrase**, and the same search walks phrases instead of fragments.

## 2. What is being specified

No new model kind and no new file format. **One dial of the encoding**
(D-075), which every model of every kind already carries:

> **A word is a unit.** `Encoding.unit` says whether one position of a text is
> a character or a whitespace word; `n` says how many of them a gram holds and
> `stride` how far apart grams start. A gram of words is *text*, so the graph
> below the dial is handed labels exactly as it always was and never learns
> what a word is.

So `char:3:1` is the character trigram the model was born with, `word:3:1` the
word trigram, `word:2:1` the word bigram and `char:5:5` groups of five letters
— one mechanism, and word n-grams are one setting of it.

Explicitly **not** specified here: no change to the graph, the weights, the
counters, the paths, either traversal, the negative network or the search; no
change to what `kind: "count"` does; no learned tokenizer, ever (D-006 rejects
that, and this does not reopen it).

## 3. The alphabet

There is no vocabulary object, and that is the point of the design. A gram is
text, so **the alphabet a graph knows is whatever its grams are made of**. It
grows as training reads new words and there is nothing to freeze, prune, learn,
version or check on load.

What a *listing* of the alphabet needs is therefore derived rather than stored:
`Encoding.vocabulary(grams)` counts the units of the graph's gram index — the
one count compression cannot change, since a phrase merged into one node is
still made of the grams that built it. `radixnet words`, `GET /api/words` and
the frontend's Words tab read it.

A word's `id` is its **rank** in that listing, not an identity: nothing records
the order words were first read. The order is written down once — most read
first, ties alphabetically — so Python, Go and Rust hand back the same rows.

## 4. Tokenising

The whole tokeniser is: **maximal runs of code points the Unicode `White_Space`
property does not cover.** Punctuation stays attached to the word it touches
and case is kept, so `"mat."` and `"mat"` are two words and `"The"` and `"the"`
are two words.

Every refinement of that rule is a step towards a vocabulary that has to be
designed, versioned and defended. Python subtracts the four separators
(`U+001C`–`U+001F`) that `str.split()` treats as whitespace and the Unicode
property does not, so the three implementations split a text into the same
words.

## 5. What a round trip costs

`decode(encode(text))` returns the words joined by single spaces. So

```
"a  b\n c"   ->   "a b c"
```

**A word encoding normalises whitespace.** Training normalises the text first,
so the structural round trip the invariants check holds against the normalised
text. A text whose whitespace carries meaning — source code, base64, a waveform
(D-036) — must not be trained under a word encoding, and that is the honest
cost: **a word model cannot represent the corpora a character model was chosen
to be able to eat.** `char:3:1` stays the default for that reason.

## 6. The unknown word

There is no `<unk>` symbol, because there is nothing to map an unread word
*to*. A gram holding a word the graph has never read is simply not in the
index, so the existing machinery charges it exactly what it charges any unknown
transition: `log(UNKNOWN_PROB)` in `score`, and no walk in `predict`.

Two different unread words are therefore *not* the same thing, which is one
cost the alphabet design paid and this one does not — but two unread words in
the same position are both unknown transitions, so a score still cannot tell
them apart by cost. It is visible in `unknown_transitions` either way.

## 7. n is a dial, and 3 is its default

`n` is 3 by default for the same reason D-006 gives: the overlap is two units
and the pivot is the middle one. Under a word encoding that argument holds word
for word, so `word:3:1` is the word n-gram this document is named for.

`word:2:1` — the word **bigram** — is now reachable, and it is a real change to
what the graph is: a one-unit overlap collapses the pivot and the context that
D-006 rejected collapsing. It is offered, not recommended, and `char:3:1`
remains the default everywhere.

## 8. The model file

**One format, `radixnet-count`, whatever the encoding.** The dial is a property
of the graph, so it rides in the graph block:

```json
{
  "format": "radixnet-count",
  "kind": "count",
  "graph": {
    "format": "radixnet-graph",
    "encoding": {"unit": "word", "n": 3, "stride": 1},
    "nodes": {"labels": ["<s>", "</s>", "<back>", "the cat sat on", ...]},
    ...
  }
}
```

The labels are text, readable as they stand. The `encoding` block is written
**only when it is not `char:3:1`**, so an ordinary model file is byte for byte
what it always was and a reader written before the dial existed never meets a
file it would misread.

Interchange is the contract the count model has: Python, Go and Rust read and
write the file, and the parity suites require the same structure, the same
counts, the same alphabet **in the same order** and the same predictions, under
nine encodings.

## 9. API and CLI

```bash
python -m radixnet --encoding word:3:1 train --data corpus.txt --epochs 5
python -m radixnet --encoding word:3:1 predict --prefix "the cat sat on" --length 6
python -m radixnet words --limit 20                     # the alphabet it has read
go/bin/radixnet-count --encoding word:3:1 train --data corpus.txt
rust/target/release/radixnet --encoding word:2:1 predict --prefix "the cat" --length 6
```

`--units`, `--ngram` and `--stride` set the three dials one at a time;
`--encoding` sets all three at once. The encoding is **fixed for a model's
life** — every label is written in it — so the flags apply to a *new* model and
a loaded file's own encoding always wins; a flag that disagrees with the file is
an error rather than something quietly ignored. Over HTTP the same choice is
`POST /api/reset` with `{"encoding": "word:2:1"}` or `{unit, ngram, stride}`,
and `GET /api/encoding` reports what the running model is in.

`GET /api/words` serves the alphabet and the frontend grows a **Words** tab
whenever the active model counts in words; every length field there says
*words* rather than *characters*, for the same reason.

Everything counted in units is counted in **words**: `--length 6` emits six
words, `--max-length` caps words, `Score.chars` counts words and `per_char` is
per word. The reports say which (`units` rides in `stats`), because a number
whose unit depends on the model is a number that will be read wrong.

## 10. Tests

| test | what it pins |
|---|---|
| the split | the tokeniser is the Unicode property, and the three ports agree on it |
| the encoder | `"the cat sat on"` is two grams of three words; whitespace normalises; the tail no whole gram covers is dropped |
| the graph | a word corpus builds a graph whose invariants hold and whose texts round-trip — the same `check_invariants` the character model uses |
| compression | a repeated phrase becomes one node whose label is that phrase |
| prediction | a word model continues a prefix with words, and `score` charges an unread word as unknown |
| the alphabet | it is whatever the grams are made of, most read first, ties alphabetically, `id` = rank |
| the file | one format; the `encoding` block is written only when it is not the default, and read back |
| parity | Go and Rust against Python: the same graph, file, alphabet and predictions under nine encodings |

## 11. Alternatives rejected

**A word model kind with a code-point vocabulary** (D-073). Each distinct word
gets one code point above Latin-1, so a label stays a string the existing
packed-trigram representation holds unchanged, and the graph is untouched
literally rather than only structurally. It works, it was built, and it lost on
generality: it buys words and nothing else, while the dial buys any `n`, any
stride and letter groups as well — and it needs a vocabulary, which has to be
written into the file in order, read back, checked on load and kept in step
across three implementations. The dial needs none of that, because a gram is
text.

**Make the graph generic over a sequence of symbols** (labels become lists of
words rather than strings). Honest, and the version of this that a type system
would prefer — but it changes every structural routine in three
implementations, changes the file format's `labels` from strings to arrays, and
buys nothing that indexing the text by unit does not already buy. A `Units`
view — an index of byte offsets, sliced in O(1) — makes one code path serve
both alphabets.

**A learned sub-word vocabulary.** Still rejected, for D-006's reason: it needs
a corpus before training can start, and freezes what the model can read. A word
encoding is not a vocabulary — an unread word is a new node, exactly as an
unseen trigram is.

**Words *and* characters in one model** (a unit is either). It is a real idea —
a graph that backs off from a word to its characters where the word is unknown
— and it is a different research claim, not this one.
