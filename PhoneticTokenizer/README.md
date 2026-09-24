# PhoneticTokenizer

A tokenizer that reads text as **the sounds it is made of**, and the phonotactics
that say which sounds go well together to build words. Three ports of one design
- Python (`phonetok/`), Go (`go/`) and Rust (`rust/`) - that read the same data
files and turn the same text into the same tokens, sample for sample; and a voice
that speaks the tokens back, as they arrive.

```
$ phonetok tokenize "The cat sat on the mat."
DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # M AE1 T .

$ phonetok tokenize --level syllable "butter strengths"
B.AH1.T ER0 # S.T.R.EH1.NG.K.TH.S

$ phonetok decode "DH AH0 # K AE1 T # S AE1 T ."
the cat sat.

$ phonetok blend smoke fog
smoke + fog = smog  (S M AA1 G, /smˈɑɡ/)

$ phonetok say "Hello world." --play
```

Standard library only, in every port. The full CMU Pronouncing Dictionary is
optional (`pip install cmudict`, or `PHONETOK_LEXICON=/path/to/cmudict.dict`);
without it a bundled core lexicon and a set of letter-to-sound rules do the work.
`DESIGN.md` is the argument; this is the manual.

## What a token is

The alphabet is the ARPAbet of the CMU dictionary: 39 phonemes, the 15 vowels
carrying a stress digit (`AH0` the schwa of *about*, `AH1` the vowel of *hut*),
plus `#` between two words and three pauses (`,` `.` `?`) that punctuation
becomes. The **level** decides what one token is:

| level | one token is | `"the cat sat"` |
|---|---|---|
| `phoneme` | one sound | `DH AH0 # K AE1 T # S AE1 T` |
| `constituent` | an onset, a vowel or a coda | `DH- AH0 # K- AE1 -T # S- AE1 -T` |
| `syllable` | one syllable | `DH.AH0 # K.AE1.T # S.AE1.T` |
| `word` | one word's sounds | `DH.AH0 # K.AE1.T # S.AE1.T` |

The **text form** - the tokens joined by single spaces - is plain text a model can
read, and it is *idempotent*: the tokenizer reads its own output back unchanged,
any run of its tokens included (a gram of a graph, one that starts or ends on a
`#` or a pause), and a text may mix words and sounds (`"the K AE1 T sat"`).
That property is what lets a model store sounds as its labels and read them back
as the units they were made of.

**Ids.** At the phoneme level the ids are a fixed alphabet of 92 symbols
(`<pad> <unk> <s> </s> # , . ?` then `cmudict.symbols` in its own order), the
same on every machine. At the other levels a token gets the next id the first
time it is read, and `save` keeps the table - there is no vocabulary to design,
learn or freeze.

## Installing

```bash
cd PhoneticTokenizer
make install              # pip install -e .   (nothing else)
make install-cmudict      # ... with the full 135 000-word dictionary
make test                 # the Python suite; the port comparisons run when go / cargo are on PATH
make go-build rust-build  # go/bin/phonetok and rust/target/release/phonetok
make test-all             # every suite of every port
```

## The library (Python)

```python
from phonetok import PhoneticTokenizer

tok = PhoneticTokenizer()                    # level="phoneme"; "constituent" | "syllable" | "word"
tok.tokens("The cat sat.")                   # ['DH', 'AH0', '#', 'K', 'AE1', 'T', '#', 'S', 'AE1', 'T', '.']
tok.text("The cat sat.")                     # 'DH AH0 # K AE1 T # S AE1 T .'
ids = tok.encode("The cat sat.")             # [71, 17, 4, 76, 14, 85, 4, 83, 14, 85, 6]
tok.decode(ids)                              # 'the cat sat.'
tok.ipa("The cat sat.")                      # 'ðə ˈkæt ˈsæt'
tok.syllables("computer")                    # [K.AH0.M, P.Y.UW1, T.ER0]
tok.explain("Strengths, 42!")                # each word: phones, where they came from, syllables, IPA
tok.features("K")                            # 22 articulatory features (voice, place, manner, height, ...)
tok.save("tok.json"); PhoneticTokenizer.load("tok.json")
```

The sounds that go well together:

```python
tok.affinity("S", "T")                       # +1.84 bits: they go together
tok.affinity("T", "L")                       # -3.80 bits: no English syllable starts with tl
tok.score("strength") > tok.score("ngstrelk")  # how English a word sounds
tok.well_formed("strength")                  # could a speaker say it?
tok.rhymes("cat", "hat"), tok.alliterates("big", "bad")
tok.coin(seed=3)                             # 'pring': a new pronounceable word, the same in every port
tok.blend("breakfast", "lunch")              # 'brunch'
```

And the voice:

```python
from phonetok.synth import Synthesizer, write_wav
pcm = Synthesizer().speak(tok.tokens("The cat sat."))   # 16-bit mono PCM, 16 kHz
write_wav("cat.wav", pcm)

synth = Synthesizer()
for token in tok.tokens("Hello world."):                 # one token at a time
    chunk = synth.feed(token)                            # a word's audio comes back when the token after it arrives
chunk = synth.end()                                      # the final sentinel: the utterance is closed
```

Under the facade: `Lexicon` (CMU-format files, layered), `Transcriber` (the
lexicon, then morphology - *cats*, *walked*, *unhappy*, *toothbrush* - then
numbers, letters and the rules; and `spell`, the way back), `syllabify`,
`Phonotactics` (counts, affinities, `build`, `blend`), `letter_to_sound`.

## How a word is sounded out

1. **The lexicon.** The bundled core (the common words, the numbers, the letters,
   the contractions, some 2 000 entries of the CMU dictionary) plus the full
   dictionary when it is installed or `PHONETOK_LEXICON` names it.
2. **Its memory** of words it has already sounded out (so that `decode` can give
   the spelling back).
3. **Numbers** (`42` is *forty two*, `$2.50` *two dollars fifty cents*, `21st`),
   **joins** (`well-known`, `u.s.`), **contractions** (`cat's`, `it'll`, `can't`).
4. **Morphology**: an unknown word whose stem is known takes the stem's sounds and
   the affix's - the `-s` and `-ed` endings with the voicing the stem's last sound
   decides - and a compound of two known words is both.
5. **Letters** for a word with no vowel letters or an unknown word in capitals
   (`FBI`).
6. **The rules**: some 380 context rules in the classic letter-to-sound form
   (`data/rules.lts`), with the stress placed by the shape of the word. Against
   the full dictionary they sound out about 38 % of words exactly and 82 % of
   phones (the dictionary is half names); against the common words of English,
   about 63 % of the top thousand (`phonetok eval`).

## The three ports

| | Python | Go | Rust |
|---|---|---|---|
| code | `phonetok/` | `go/phonetok/` | `rust/src/` |
| command | `phonetok` / `python -m phonetok` | `go/bin/phonetok` | `rust/target/release/phonetok` |
| the data | `phonetok/data/` | embedded copies (`make sync-data`) | `include_str!` of the same files |
| tests | `make test` | `make go-test` | `make rust-test` |

They agree token for token. `tests/parity.json` is what Python says about a
battery of texts - tokens at every level, ids, decoding, syllables, rules,
respellings, affinities, coined words, blends - and the Go and Rust suites assert
it; `tests/test_parity.py` then builds the two command lines and compares them
with Python over the sample corpora, the whole core lexicon and a spoken
sentence. The seeded word-coiner draws from a Mersenne Twister with CPython's
semantics, so a seed coins the same word in all three; the voice's noise is a
shared xorshift, so the three voices are one voice to within the last bits of a
`libm`.

The same commands on every port:

```
tokenize | explain | ipa | decode | pronounce | rhymes | affinity | coin | blend | lexicon | eval | say
options: --level L  --no-stress  --no-boundaries  --no-pauses  --core  --lexicon FILE  --json
```

## The voice

`synth.py` (and `synth.go`, `synth.rs`) is a formant synthesizer - a source-filter
vocoder of the kind speech was synthesized with before there were models to train.
A glottal pulse at the pitch, or noise, runs through a cascade of resonators tuned
to each sound's formants (`data/voice.tsv`: the targets, bandwidths, durations,
voicing and noise of every phoneme); the formants glide from one sound to the next,
stressed syllables get a higher pitch and a longer stay, the pitch declines over an
utterance, falls at a full stop and rises at a question. It does not sound like a
person; every sound is where it should be, which is what a model whose symbols are
sounds needs to be heard.

It streams. Fed one token at a time, it commits a word's audio as soon as the
token after the word arrives, and the **final sentinel** - `</s>`, or `end()` -
closes the utterance with the closing intonation. ~70 x real time in pure Python,
thousands of times in the ports.

```bash
phonetok say "The cat sat." --out cat.wav        # a WAV
phonetok say "The cat sat." --play               # aplay / paplay / ffplay / play / afplay
echo "DH AH0 # K AE1 T ." | phonetok say --raw | aplay -r 16000 -f S16_LE   # tokens on stdin, PCM out
```

## In RadixCyclicNN

The tokenizer is an option of the encoding dial of the sibling project: a model
trained with `--encoding phone:3:1` (or `syllable:2:1`) reads every text as
sounds, its labels are sounds, and its predictions are spelled back into words;
`radixnet speak` walks the model and speaks the walk as it goes, each utterance
closed by the graph's END sentinel. All three ports of that project read the same
sounds through this package. See `../RadixCyclicNN/README.md`, *The encoding*.

## Layout

| path | what |
|---|---|
| `phonetok/phones.py` | the inventory, the ids, the features, the IPA |
| `phonetok/rules.py` | the letter-to-sound engine and the stress |
| `phonetok/numbers.py` | numbers as words |
| `phonetok/lexicon.py` | the CMU-format lexicon, layered |
| `phonetok/g2p.py` | the transcriber (lexicon, morphology, rules) and the respelling |
| `phonetok/syllables.py` | maximal onset over the closed list of English onsets; rhyme |
| `phonetok/phonotactics.py` | affinities, well-formedness, building and blending words |
| `phonetok/tokenizer.py` | the four levels, the text form, the ids, the way back |
| `phonetok/synth.py` | the voice |
| `phonetok/cli.py` | the command line |
| `phonetok/data/` | `core.dict`, `rules.lts`, `voice.tsv`, the CMU licence |
| `tests/` | the suite, `make_parity.py` and `parity.json` |
| `go/`, `rust/` | the ports |

Licensed under the repository's source-available licence (`../LICENSE`); the
bundled dictionary entries are CMU's, under their BSD licence
(`phonetok/data/LICENSE-cmudict`).
