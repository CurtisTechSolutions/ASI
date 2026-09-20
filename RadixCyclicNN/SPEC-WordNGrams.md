# Word n-grams — the same graph over a different alphabet

**Status** Built, 2026-09-20. `DESIGN.md` §34 is what was built and
`DECISIONS.md` D-073 is why; this document stays as the argument the two were
written from, and §10's table is the test suite
(`tests/test_wordnet.py`, `go/radixnet/words_test.go`, `rust/tests/words.rs`,
and the word classes of the two parity suites).

**Answers** *"what about word n-grams?"* — and answers it without touching the
graph, the weight function, the search or any of the three ports' structural
code.

---

## 1. The claim this rests on

The model's input decision (D-006) is about **what a symbol is**: characters,
not bytes and not tokens, in a sliding window of three with stride 1. Every
structural rule in the graph is stated in terms of that window and nothing else:

* a node's label is a sequence of **symbols**, at least `WINDOW` of them;
* an edge `p -> c` exists only where the last `WINDOW - 1` symbols of `p` are
  the first `WINDOW - 1` of `c`;
* the index maps a `WINDOW`-symbol key to `(node, offset)`;
* compression merges a unary chain by overlapping the two labels at that seam;
* decoding walks the path and takes each label's symbols past the overlap.

**Not one of those rules mentions a character.** They are true of any alphabet.
So a word n-gram model is not a different model — it is this model over an
alphabet whose symbols are words:

```
"the cat sat on the mat"
  characters →  ["the", "he ", "e c", " ca", ...]        the alphabet is ~100 symbols
  words      →  [("the","cat","sat"), ("cat","sat","on"), ...]   the alphabet is the corpus
```

The compression then does to word chains what it already does to character
chains. Today a recurring word becomes one node — the sample corpus produces
labels like `'lays eggs'` and `'s celsius'` out of nothing but unary merging.
In word mode the same merge produces a node whose label is a **phrase**, and
the same search walks phrases instead of fragments.

## 2. What is being specified

One new model kind, `word`, which is the count / reward model (D-021, D-022)
with the encoder and decoder replaced:

> **A word is a symbol.** The encoder splits text on whitespace, maps each
> distinct word to a symbol the graph can hold, and hands the graph the same
> sliding window of three it always gets. The decoder maps back. Everything
> between those two points is unchanged, in all three implementations.

Explicitly **not** specified here: no change to the graph, the weights, the
counters, the paths, either traversal, the negative network or the search; no
change to what `kind: "count"` does; no new value of `WINDOW` (§7); no learned
tokenizer, ever (D-006 rejects that, and this does not reopen it).

## 3. The alphabet

A symbol has to be something a label can be made of. In all three
implementations a label is a string and its length is counted in code points
(`runeLen`, `char_len`, `len`), so **a symbol is one code point** and a word
model's alphabet is a map between words and code points:

```
id 0            -> U+0100          the unknown word
id i (i >= 1)   -> U+0100 + i      skipping the surrogate block D800..DFFF
```

Starting at `U+0100` puts every word symbol past ASCII and past Latin-1, so no
label can be mistaken for text, for a sentinel (`<s>`, `</s>`, `<back>`) or for
punctuation while it is being read. Skipping the surrogates keeps every label a
string the three languages can hold and a file the three can write: Go replaces
a lone surrogate with `U+FFFD`, which would silently corrupt a model.

**The vocabulary holds 1 111 808 words.** It grows as training reads new words
and is never frozen, pruned or learned — a word is in it because it was seen.
That is the part of D-006 this keeps: there is no tokenizer, no merge table and
no training run before the training run.

## 4. Tokenising

`text.split()`, and nothing else. Whitespace separates words; punctuation stays
attached to the word it touches (`"mat."` and `"mat"` are two words); case is
kept (`"The"` and `"the"` are two words).

This is a decision, not an oversight. Every refinement — splitting punctuation,
folding case, stripping accents — is a step towards a tokenizer, and the moment
the project ships one it owns a vocabulary that has to be designed, versioned
and defended. A whitespace split is the only rule that needs no defending, and
a corpus that wants `"mat ."` can be written that way.

## 5. What a round trip costs

`decode(encode(text))` returns the words joined by single spaces. So

```
"a  b\n c"   ->   "a b c"
```

**A word model normalises whitespace.** Training normalises the text first, so
the structural round trip the invariants check (`decode(encode(t)) == t`) holds
against the normalised text, and `train` reports what it normalised. A text
whose whitespace carries meaning — source code, base64, a waveform (D-036) —
must not be trained on a word model, and that is the honest cost of the
alphabet: **a word model cannot represent the corpora a character model was
chosen to be able to eat.** Both kinds stay; neither replaces the other.

## 6. The unknown word

Id 0 is `<unk>`. At training time nothing is unknown — a new word joins the
vocabulary. At prediction and scoring time a word the model has never read maps
to `<unk>`, whose windows are not in the index, so the existing machinery
charges it exactly what it charges any unknown transition: `log(UNKNOWN_PROB)`
in `score`, and no walk in `predict`.

Two different unseen words are the same symbol, so `"the qux sat"` and
`"the quux sat"` score identically. That is what a vocabulary costs, it is
visible in the output (`unknown_transitions`), and it is why the character model
remains the default.

## 7. N stays 3

`WINDOW` is 3 for a word model exactly as it is for a character model, so the
overlap is two words and the pivot is the middle one — D-006's argument holds
word for word. Word **bigrams** need `WINDOW = 2`, which is a separate change
with its own costs (a one-symbol overlap collapses the pivot and the context,
which D-006 rejected), and in Rust it additionally needs the packed-trigram
representation rewritten. Out of scope here, and nothing in this document
prevents it later.

## 8. The model file

A new format, so that **every existing reader refuses it by the check it already
makes**:

```json
{
  "format": "radixnet-word",
  "version": 1,
  "kind": "word",
  "meta": {...}, "history": [...],
  "graph": {
    "format": "radixnet-graph", "format_version": 3,
    "nodes": {...}, "edges": {...}, "weights": {...}, "paths": {...},
    "units": "words",
    "vocabulary": ["<unk>", "the", "cat", "sat", ...]
  }
}
```

`model_from_dict` dispatches on `format`, so a `radixnet-count` reader written
before this existed says *"not a radixnet-count model document"* rather than
reading word symbols as text. `vocabulary` is written in id order and is the
only new block; `units` names the alphabet for anything that reports on a model.
Both ride at the end of the graph block, where the three writers put them, and a
load checks that every symbol every label carries is a word the vocabulary
holds - a truncated vocabulary is an error, not a decoding surprise later.

Interchange is the same contract the count model has: Python, Go and Rust read
and write this file, `tests/test_go_parity.py` and `tests/test_rust_parity.py`
train a word model on both sides of each pair and require the same structure,
the same counts, the same vocabulary **in the same order**, and the same
predictions.

## 9. API and CLI

```bash
python -m radixnet --kind word train --data corpus.txt --epochs 5
python -m radixnet --kind word predict --prefix "the cat sat on" --length 6
python -m radixnet --kind word words --limit 20        # the alphabet it has read
go/bin/radixnet-count --kind word train --data corpus.txt
rust/target/release/radixnet --kind word predict --prefix "the cat" --length 6
make word-demo                                          # train, predict, generate, list
```

The default `--model` follows the kind (`model.word.json`), so a word model
never overwrites a character one. `GET /api/words` serves the alphabet and the
frontend grows a **Words** tab whenever the active model counts in words; every
length field there says *words* rather than *characters*, for the same reason.

Everything counted in symbols is counted in **words**: `--length 6` emits six
words, `--max-length` caps words, `Score.chars` counts words and `per_char` is
per word. The reports say which (`units` rides in `stats`), because a number
whose unit depends on the model is a number that will be read wrong.

## 10. Tests

| test | what it pins |
|---|---|
| the alphabet | every id round-trips through its code point; the surrogate block is skipped; the cap is what §3 says |
| the encoder | `"the cat sat on"` is two windows of three words; whitespace normalises; an unseen word is `<unk>` at inference and a new id in training |
| the graph | a word corpus builds a graph whose invariants hold and whose texts round-trip - the same `check_invariants` the character model uses |
| compression | a repeated phrase becomes one node whose label is that phrase |
| prediction | a word model continues a prefix with words, and `score` charges an unseen word as unknown |
| the file | a word model round-trips; a `radixnet-count` reader refuses it; a word file's vocabulary comes back in order |
| parity | Go and Rust against Python: same graph, same vocabulary, same predictions |

## 11. Alternatives rejected

**Make the graph generic over a sequence of symbols** (labels become lists of
words rather than strings). Honest, and the version of this that a type system
would prefer — but it changes every structural routine in three implementations,
changes the file format's `labels` from strings to arrays, and buys nothing the
code-point alphabet does not already buy. Rejected for cost, and because the
current design *already* proves the graph is alphabet-agnostic: this document
only names what was always true.

**A private-use code-point range.** Unicode's private use areas hold 137 468
code points, which a real corpus exhausts. The range chosen here is the whole
space above Latin-1, which is not "private" but is unambiguous within a model
whose every symbol is a word.

**A `units` flag on the count model rather than a new kind.** One flag, one
format, one file that an old reader would accept and misread. The failure would
be silent - the labels would decode as nonsense text - and a silent failure
across three implementations is not worth one fewer class.

**Words *and* characters in one model** (a symbol is either). It is a real idea
- a graph that backs off from a word to its characters where the word is unknown
- and it is a different research claim, not this one.
