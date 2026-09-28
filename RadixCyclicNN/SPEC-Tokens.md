# The token unit — the traditional LLM tokenizer as an encoder and decoder layer

**Status** Built, 2026-09-28. `DECISIONS.md` D-090 is why, `DESIGN.md` §41 is where it lives, and
`tests/tokens_fixture.json` is the contract the three ports are held to.

**Asked for** *"Implement a new encoder and decoder layer that follows the traditional LLM
tokenizer."*

**Answers** it the way this project answers every question about what a symbol is (D-071): as a
setting of the encoding dial. `--encoding token:3:1` (or `bpe:3:1`) reads every text through a
byte-level byte-pair encoder - GPT-2's tokenizer, piece for piece - and builds the graph over the
tokens it makes; the decoder turns the tokens of a prediction back into the text they are. Nothing
in the graph, the search, the weights or the file format changes: a gram is still its units joined
by single spaces, and the file says `"encoding": {"unit": "token", "n": 3, "stride": 1}`.

---

## 1. What is being specified

Five things, each exactly enough that three implementations make the same tokens:

1. the **byte alphabet** a token is written in (§2);
2. the **pre-tokenizer** that cuts a text into the pieces merges never cross (§3);
3. the **merges**, the ids and the round trip (§4);
4. the **text form**: how a run of tokens is written as the text a graph label is, and read back (§5);
5. the **merges file**, where the vocabulary comes from, and how it is learned (§6, §7).

`radixnet/bpe.py` is the reference; `go/radixnet/bpe.go` and `rust/src/bpe.rs` are the ports.

## 2. The byte alphabet

GPT-2's `bytes_to_unicode`, unchanged. A byte `b` is written as the character `b` itself when it is
printable ASCII (`0x21`-`0x7E`) or printable Latin-1 (`0xA1`-`0xAC`, `0xAE`-`0xFF`); the other 68 bytes,
in increasing order, are written as `U+0100`, `U+0101`, ... `U+0143`. So the space is `Ġ` (`U+0120`), the
newline `Ċ`, the tab `ĉ`, the soft hyphen `Ń`. No character of the alphabet is whitespace (by any of the
three languages' rules), and none is `⁀`.

A token is a non-empty byte string, written in this alphabet: `" cat"` is `Ġcat`.

## 3. The pre-tokenizer

### 3.1 The classes

Every character is in exactly one class, decided in this order:

| class | the characters |
|---|---|
| `S` space | the Unicode `White_Space` property, written out: `\t \n \v \f \r`, the space, `U+0085`, `U+00A0`, `U+1680`, `U+2000`-`U+200A`, `U+2028`, `U+2029`, `U+202F`, `U+205F`, `U+3000` |
| `D` digit | `0`-`9` |
| `P` punctuation | every other ASCII character that is not a letter (the controls too), and the blocks `U+007F`-`U+00BF`, `U+00D7`, `U+00F7`, `U+2000`-`U+206F`, `U+20A0`-`U+20CF`, `U+2190`-`U+2BFF`, `U+2E00`-`U+2E7F`, `U+3001`-`U+3003`, `U+3008`-`U+3011`, `U+3014`-`U+301F`, `U+FE00`-`U+FE0F`, `U+FF01`-`U+FF0F`, `U+FF1A`-`U+FF20`, `U+FF3B`-`U+FF40`, `U+FF5B`-`U+FF65`, `U+1F000`-`U+1FAFF`, `U+E0020`-`U+E007F` |
| `L` letter | everything else: the ASCII letters and every other character of every script |

GPT-2 asks the Unicode categories (`\p{L}`, `\p{N}`); this table asks nothing, because the Rust
standard library has no categories to ask and a class that differed in one port would give that
port different tokens. A digit of another script is a letter here, and a curly quote, a dash, an
arrow and an emoji are punctuation, as they are in GPT-2.

### 3.2 The pattern

GPT-2's, over those classes, applied left to right, the first alternative that matches winning:

```
's|'t|'re|'ve|'m|'ll|'d| ?L+| ?D+| ?P+|\s+(?!\S)|\s+
```

The contractions match in either case (`'LL`, as tiktoken's later patterns do). As steps, at
position `i`:

1. a contraction starts at `i`: it is a pre-token;
2. otherwise let `j = i + 1` if the character at `i` is the space `U+0020` and the next one is not
   `S`, else `j = i`; if the character at `j` is not `S`, the pre-token runs from `i` over every
   following character of the same class as the one at `j`;
3. otherwise the character at `i` begins a run of `S`: the pre-token is the whole run if the run
   ends the text or is one character long, and the run but its last character otherwise - which is
   left to be the ` ?` of what follows it (if it is a space) or a pre-token of its own (if it is not).

`"Hello, world!\nIt's"` is `Hello` `,` ` world` `!` `\n` `It` `'s`. The pre-tokens join back into
the text exactly, and no merge crosses from one into the next.

## 4. The merges, the ids and the round trip

**Merges.** A ranked list of pairs of tokens. A pre-token is turned into its UTF-8 bytes, each byte
a token; then, while any adjacent pair of tokens is in the list, the pair with the best (lowest) rank
is merged **everywhere it occurs, left to right, without overlap** (`a a a` under the merge `a a` is
`aa a`), and the search starts again. This is GPT-2's `bpe()`.

**The leading space.** Every text is read with one space put before it, as SentencePiece-based
models do, so that the first word of a text is the same token as the same word anywhere else
(`"The cat"` is `ĠThe Ġcat`, not `The Ġcat`). The decoder takes one leading space away again.

**Ids.** `0`-`255` are the bytes, by value. Then each token a merge makes, in rank order - a merge
whose result is already a token adds no id. Then the special tokens, of which there is one,
`<|endoftext|>`. The bundled vocabulary is 4096 ids: 256 bytes, 3839 merged tokens and the special
token. The encoder reads `<|endoftext|>` in a text as the special token only when it is asked to
(`allowed_special`); otherwise it is the thirteen characters it is. The special token never
appears in the text form: the graph's own sentinels (`<s>`, `</s>`) are its specials.

**The round trip.** `decode(encode(t)) == t` for every text - no `<unk>`, no normalisation: every
byte is a token. A run of ids that ends inside a character (ids the model predicted, say) decodes
the broken bytes as `U+FFFD`, one for each *maximal broken piece* - the Unicode practice Python's
decoder follows, which the Go and Rust ports write out by hand.

## 5. The text form

A graph label is text: a gram is its units joined by single spaces, and every label is read back by
cutting it into units. So the tokens need a way of being written that the tokenizer reads back as
the very tokens it was made of - **any run** of them included, because a label starts and ends
wherever a gram does, inside a word or on a space.

### 5.1 Writing a token

| the token | is written | example |
|---|---|---|
| a space followed by one or more printable ASCII bytes (`0x21`-`0x7E`) | **bare**: the bytes after the space | `Ġcat` → `cat`, `Ġ.` → `.` |
| anything else | **glued**: `⁀` (`U+2040` CHARACTER TIE), then the token in the byte alphabet | `ing` → `⁀ing`, `.` → `⁀.`, `Ċ` → `⁀Ċ`, `Ġ` → `⁀Ġ`, `ĠcafÃ©` → `⁀ĠcafÃ©` |

`"The walking cat."` is `The walk ⁀ing cat ⁀.` A text of plain words is its own text form: `the cat
sat on the mat` is written `the cat sat on the mat`, one token a word. The bare form is GPT-2's `Ġ`
read as what it means - *a space before this* - so the space a token begins with is the space before
the piece; the glue says the opposite, *no space before this*.

### 5.2 Reading a text

A piece (a run of non-space characters) **reads** as a token when

* it is glued - `⁀` then a string that is a token of the vocabulary: that token; or
* it is bare - printable ASCII - and the tokenizer, given a space and the piece as text, makes
  exactly one token, the space and the piece: that token.

A text **is the text form** when it is non-empty, its only whitespace is single spaces between
pieces, and every piece reads. Then its units are its pieces' tokens, written as §5.1 writes them.
Every other text is read **as text**: its tokens (with the leading space, §4), written as §5.1
writes them. The empty text has no units.

### 5.3 Why that is idempotent

* **A run of units reads back as itself.** Every piece of a run is a token written by §5.1. A glued
  piece reads because its token is in the vocabulary. A bare piece `p` was written for a token
  `Ġp` the encoder made inside some pre-token; `Ġp` starts with its pre-token's space, so the text
  `" " + p` is one pre-token on its own (a space and a run of one class), and byte-pair encoding a
  stretch of a pre-token that the encoder left whole gives that stretch back whole - no pair across
  its edge was ever merged, and the pairs inside it were merged in the same rank order - so it reads.
  A run of units is single-spaced. So a run is the text form, and reads as its own pieces.
* **Where both readings apply, they agree.** A text of single-spaced bare pieces that each read is
  `" " + p1 + " " + p2 ...` as text: the pre-tokenizer cuts it at every space (a space stops every
  run), each pre-token `" " + pk` is one token by the reading rule, so reading it as text gives the
  same tokens as reading it piece by piece. That is why the bare form is kept to printable ASCII -
  for those bytes a character *is* its byte, so a bare piece means the same written by a person as
  written by the tokenizer - and why `⁀` is outside the byte alphabet: no text of the byte alphabet
  can contain it, so a glued piece is never text in disguise.

What it costs: a text typed with `⁀` in it is read as tokens where it can be, and a text of plain
words with two spaces between two of them is text, not the text form (it reads as its tokens, which
keep the two spaces: `cat ⁀Ġ sat`). Both are what a person writes by accident rarely; the second is
exact anyway.

### 5.4 The encoding dial over it

The token unit is a unit like the others (`radixnet/encoding.py`): `units` is the text form's units,
`length` their number, `join` reads each piece as tokens and joins them with single spaces,
`has_unit_prefix` reads the prefix as tokens (so `"the cat sat."` is a prefix of the label
`the cat sat ⁀.` and `"the ca"` is not), `reverse` reverses the tokens, and `spell` - the decoder -
gives the text a text of tokens is: `the cat sat ⁀.` spells `the cat sat.`. What the model says is
spelled wherever a phonetic model's sounds are (`predict`, `generate`: the `spelled` line and field),
and a walk is spoken by the voice as the letters its tokens' bytes spell, a character held until
its last byte arrives.

## 6. The merges file, the bundled vocabulary, and the rule that goes with them

**The file** is GPT-2's `merges.txt`: one merge per line, `left right`, the two tokens in the byte
alphabet separated by one space, in rank order; a line beginning with `#` is a comment, and a
`#note:` line is kept as the file's note - GPT-2's own `merges.txt` is in this format. A line that is not
two tokens, a character outside the byte alphabet, a merge listed twice and a merge whose side is
not yet a token when its line comes are refused, naming the line.

**The bundled merges** (`radixnet/data/merges.txt`, embedded in the Go port from its copy
`go/radixnet/data/merges.txt`, included in the Rust port from the Python one) were learned by
`tests/make_merges.py` from the 77 Markdown files of this repository at commit `34a64a1` - the only
sizeable English prose a checkout has, and the author's own - read one text per non-blank line, to
4096 ids. They read the repository's prose at about 3.0-3.5 characters a token, 1.7 tokens a word.
The commit is pinned, so the same command learns the same merges however the documents change.

**`RADIXNET_TOKENIZER=path`** points all three ports at another merges file; `radixnet tokenizer
learn` makes one from a corpus (§7).

**The rule.** A model's tokens mean nothing without the merges that made them, and the file does not
name them: **train and predict with the same merges file**, in every port - the rule the phonetic
lexicon and the acoustic codebook already have (D-080, D-082). A model over the bundled merges is
portable to any checkout; one over your own merges travels with its file. Relearning the bundled
merges is a new vocabulary: every model trained over the old one reads its labels wrongly under it
(a bare piece that is no longer one token reads as text, a glued one the vocabulary has lost reads
as text too), which is why `make tokenizer-merges` exists but is never run on a whim.

## 7. Learning

`BPETokenizer.learn(texts, vocab_size, min_frequency=2)`, the classic way. Every text is
pre-tokenized with its leading space and the pre-tokens counted. Then, again and again, the adjacent
pair of tokens that occurs most often across the corpus - counting every position, so `a a a` holds
the pair `a a` twice - is merged everywhere it occurs (left to right, as the encoder merges) and
ranked next. It stops when the vocabulary - bytes, merged tokens, the special token - reaches
`vocab_size`, or when the best pair occurs fewer than `min_frequency` times. Ties go to the pair whose
bytes sort first, so the same corpus learns the same merges in any order. A pair that reappears after
its merge (a later merge made one of its sides again) is merged at the rank it already has, as the
encoder would. The sample corpus learns 400 ids in milliseconds, the repository's prose 4096 in a
second and a half, in pure Python.

Learning is the Python package's (`radixnet tokenizer learn`); the Go and Rust ports read what it
writes. The encoder, the decoder and the text form are in all three.

## 8. The surfaces

* **Encoding**: `--encoding token:3:1` (`token`, `tokens`, `bpe`, `subword`, `subwords`), `--units token`,
  on all three CLIs; `{"encoding": "token:3:1"}` or `{"unit": "token", ...}` on all three servers'
  `/api/reset`. `info` and `/api/status` report `units: "tokens"`.
* **CLI** (Python): `radixnet tokenizer info | encode TEXT... [--special] | decode ID... | decode --text
  TEXT | learn --data FILE... --vocab N --out PATH`, each with `--merges FILE` to read another file.
  `words` lists the tokens a model has read, most read first.
* **Frontend**: the New model card offers `token:3:1` and `token:2:1` and the unit `token`; a model's
  lengths read in tokens.
* **Make**: `token-train`, `token-predict`, `token-generate`, `token-words`, `token-demo`, `tokenize`,
  `tokenizer-learn`, `tokenizer-merges`, `token-go`, `token-rust`, `token-test`, `token-parity`.

## 9. Tests

* `tests/test_tokens.py` - the byte alphabet; GPT-2's pattern on hard cases and the pre-tokens joining
  back; the classes; the bundled vocabulary; the exact round trip over random hard text under three
  vocabularies (none, a small learned one, the bundled one); the special token; `U+FFFD` for broken
  characters; writing a token; plain words as their own text form; **every run of up to six units of
  every text reading back as itself**; the two readings agreeing on 300 random texts of bare pieces;
  learning (determinism, the most frequent pair first, the stop); the file (round trip, GPT-2's own
  format, the refusals, `RADIXNET_TOKENIZER`); the unit on the dial, a model over tokens trained,
  predicted, spelled, saved and loaded; and that the fixture is up to date.
* `tests/test_encoding.py`, `tests/test_encodings_end_to_end.py` - `token:3:1` and `token:2:2` beside
  every other encoding, through the graph and all four model kinds.
* `tests/tokens_fixture.json` (`tests/make_tokens_fixture.py`) - what Python makes of 28 hard texts
  and 12 texts to read, under the bundled merges and a small learned vocabulary, plus the class of 72
  code points and 12 broken byte strings: `go/radixnet/bpe_test.go` and `rust/src/bpe.rs` reproduce
  every pre-token, token, id, unit and decoded text in it.
* `tests/test_go_parity.py`, `tests/test_rust_parity.py` - the same graph from the same corpus under
  `token:3:1` and `token:2:2`, the same predictions, each side reading the other's file, the same
  tokens listed, and the same audio from `say`.

## 10. Alternatives rejected

* **The GPT-2 form as the label** (`the Ġcat Ġsat .`). It marks the space and leaves the glue
  unmarked, which is backwards for text: a word typed after a space would read as a glued token, so
  a prefix typed as text could never match a label, and the text form would not be idempotent on
  text a person writes. §5.1 is the same information with the mark moved to where text needs it.
* **Joining a label's tokens with nothing between them** (`TheĠwalkingĠcat.`) and cutting it by
  encoding it again. Byte-pair encoding a stretch of text is not always the stretch of the encoding:
  the pre-tokenizer's context (a contraction, the last space of a run) cuts a stretch differently,
  and a label would come back as other tokens.
* **The Unicode categories** for the classes (§3.1): one port could not ask them.
* **GPT-2's vocabulary itself.** It is not in this repository, cannot be fetched from a checkout, and
  would make every model depend on a file this project did not write; its `merges.txt` is in the
  format §6 reads, if you bring it (`RADIXNET_TOKENIZER`), though the ids would not be GPT-2's ids (§4
  numbers the bytes by value).
* **Learning the vocabulary from a model's first training corpus and storing it in the model file.**
  The traditional workflow, and the most self-contained - but the encoding is a value compared and
  copied everywhere (the CLI refuses a flag that disagrees with a file by comparing it), and a
  vocabulary inside it would make it one; and the servers keep several models at once, so a tokenizer
  per model would be a registry rather than a dial. The file and the environment variable are the
  rule the lexicon and the codebook already follow.
* **Learning in all three ports.** The merges are a data file three ports read; one learner writes it.
