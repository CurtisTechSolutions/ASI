# PhoneticTokenizer - the design

**Status** Built, 2026-09-24, in Python, Go and Rust. `README.md` is the manual;
this is the argument: what a phonetic tokenizer is for, what was decided, and what
was rejected.

## 1. The claim

A tokenizer decides what a symbol is. Sub-word tokenizers decide it by counting
letters in a corpus: whatever byte pairs recur become the alphabet, and the
alphabet is an accident of the data. This tokenizer decides it by **how words
sound**. Its alphabet is the sound inventory of the language - 39 phonemes, with
the stress a vowel carries - and its larger units are the ones the language's own
phonotactics build: the onset, the vowel and the coda of a syllable, and the
syllable. None of that is learned from a corpus; it is what English is.

Two things follow that a letter tokenizer cannot give:

* **Sounding alike is being close.** *Night* and *knight* are the same token
  sequence; *cat* and *cut* differ in one symbol whose feature vectors differ in
  one dimension. A model over these symbols learns sound patterns - rhyme,
  alliteration, the way a sentence ends - rather than spelling accidents.
* **Which sounds go together is knowable.** English lets `S T R` start a syllable
  and never `T L`; it lets `NG` end one and never start one. The tokenizer knows
  this outright (`syllables.py`) and measures the habits beyond the law
  (`phonotactics.py`), which is what makes it able to *build* words - to coin a
  pronounceable new one, or blend two at the joint where their sounds meet best.

## 2. What was decided

**The alphabet is ARPAbet with stress.** The CMU Pronouncing Dictionary's symbols,
because the dictionary is the one free, large, well-tested pronunciation resource
for English, and its symbols are plain ASCII that a label can hold and a person
can read. Stress is kept (`AH0` and `AH1` are different sounds in speech) and can
be dropped (`stress=False`) when a smaller alphabet matters more. The IPA is a
rendering (`to_ipa`, `ipa`), not the representation: two code points for a
diphthong make bad labels.

**Four levels, one text form.** Phoneme, constituent, syllable and word are four
ways of cutting the same sound sequence, and every one writes the same kind of
text: tokens joined by single spaces, clusters joined by dots, an onset marked
`X-` and a coda `-X`. The text form is idempotent on *any run of tokens*, because
a graph stores runs of tokens as labels and reads them back, and a run may start
or end on a boundary or a pause. A token that is already a sound passes through;
a word is transcribed; punctuation becomes a pause; the gap between words a `#`.

**Boundaries and pauses are tokens.** A model that sees `#` learns where words
end; one that sees `.` and `?` learns how sentences end. Without them the
decoder could not tell *a nice cream* from *an ice cream*, and a synthesizer
could not breathe.

**No learned vocabulary.** The ids of the phoneme level are a fixed table; the
other levels number tokens as they are read, and there is nothing to train before
training. This is the constraint RadixCyclicNN's D-006 placed on any tokenizer
the project would accept, and it is met: the only tables are the language's.
(The acoustic units of section 5 are the one exception, and they are learned
from audio, where no table of the language's exists; see there for why that is
a different question.)

**The lexicon first, the rules last.** A pronouncing dictionary is right; rules
are a guess. The bundled core (some 2 000 entries: the common words, numbers,
letters, contractions, the repository's sample corpora) makes the package work
with nothing installed; the full dictionary is one `pip install cmudict` or one
environment variable away; the morphology stretches the lexicon over the
inflections and compounds it does not list; the rules cover the rest. The rules
are the classic NRL form (Elovitz et al., 1976) with this package's own table,
tuned against the full dictionary, and the stress placed by suffix, prefix and
syllable count. They are a floor, and the tests pin it.

**The lexicon is portable, and it is the same in every port.** A model whose
symbols are sounds means the sounds its lexicon gave it. `Lexicon.portable()` -
the core plus the `PHONETOK_LEXICON` file - is what every port can build alike,
and is what RadixCyclicNN reads through; the Python-only `cmudict` package is a
convenience of the standalone tokenizer, not of the shared models.

**The syllable is maximal onset with one exception.** Between two vowels, as many
consonants as may legally start a syllable go to the second one. The onset list
is closed (77 clusters, the empty one included) and is the language's, so every
port breaks every word the same way. The exception is English's own: a stressed
lax vowel does not end a syllable, so it takes the consonant that would have
started the next (*butter* is `B.AH1.T ER0`, not `B.AH1 T.ER0`).

**Phonotactics are counts plus the law.** The habits are bigram counts over the
lexicon's pronunciations with add-k smoothing; `affinity` is pointwise mutual
information in bits, which reads as "these go together" above zero and "these
avoid each other" below it. The law - a legal onset, a coda whose sonority falls
away from the vowel (with the coronal appendix that *sixths* needs) - is applied
to everything built, so a coined word is one a speaker could say.

**Three ports, two data files, one fixture.** The rule table and the lexicon are
data files read by every port, so a rule is changed once. The Go module embeds
copies (Go cannot embed from outside a module; `make sync-data` and a test keep
them equal); Rust includes the originals. `tests/parity.json` is Python's answer
over a battery of texts and both ports assert it; `test_parity.py` then drives
the two command lines over the corpora. The seeded word-coiner draws from a
Mersenne Twister with CPython's exact semantics, copied into both ports, so that
a seed means the same word everywhere.

**The voice is a formant synthesizer, streaming.** The decoder from sounds to
speech is a source-filter vocoder over a table of formant targets, because it
needs no data, runs anywhere the tokenizer runs, and is the same in every port to
within a `libm`. It is fed one token at a time and commits a word's audio when
the token after the word arrives (phrase-final lengthening and the final pitch
movement need to know whether a pause follows); the **final sentinel** (`</s>`)
closes the utterance. So a model walking its graph is heard while it walks, and
an utterance ends the moment the walk reaches its END.

## 3. Alternatives rejected

* **A learned sub-word tokenizer** (BPE over phones, say). It would need a corpus
  before anything could start and would freeze an alphabet that is an accident
  of that corpus; D-006's reasons hold for sounds as for letters. (They do not
  hold for audio, which has no alphabet to begin with: section 5.)
* **`espeak`, `phonemizer`, a neural G2P.** Each is a dependency with its own
  build, its own phone set and its own drift; none can be made byte-identical in
  three languages. The rules are weaker than a neural G2P on rare words and
  wholly transparent.
* **Bundling the whole dictionary.** 3.6 MB in the repository, and a copy in each
  port. The core subset is 40 KB, the full dictionary is one command away, and
  the models that matter are portable either way.
* **IPA as the representation.** Diphthongs and affricates are two code points
  each, stress marks are prefixes, and the text stops being something a graph can
  slice by symbol.
* **A neural vocoder.** Weights to ship, a framework to run them, and nothing a
  Go or Rust port could reproduce. The formant synthesizer is a hundred lines of
  arithmetic on a table, and it speaks the moment a token arrives.

## 4. Costs

* English only, and the rules are English's; another language is another lexicon
  and another rule table.
* Homophones are one token sequence, so `decode` chooses a spelling - the word
  read most often, then the shortest - and a word the lexicon never held comes
  back respelled as it sounds (`Z IH1 B R AH0` is *zibra*).
* The voice is intelligible, not natural.
* The transcriber's memory of what it read makes `decode` better and the object
  stateful; a fresh process spells only what the lexicon knows.

## 5. The acoustic units: the one thing that is learned

### 5.1 The claim

Everything above starts from text. Speech starts from sound, and no rule table
maps a waveform to units: there is no dictionary of waveforms and no alphabet of
them, so for audio the question is not *learned or rule-based* but *what is
learned, and what the units should be*. The claim is that a model whose symbols
are a small inventory of learned acoustic units can be trained on recordings
alone - no transcript, no recogniser - with the same machinery that trains it on
phones, and can be heard back through a vocoder.

### 5.2 What was decided

**Log-mel frames.** 25 ms every 10 ms, 40 triangular filters on the mel scale,
the log of each band's energy, and the utterance's mean subtracted from every
band. This is the front end of forty years of speech recognition, for the same
reasons: the mel scale spends its bands where hearing does, the log makes the
level additive, and the subtraction removes the level and the channel so the
same word in a quiet room and a loud one is the same frames. It is plain
arithmetic - a Hann window, a radix-2 FFT, a table of filters - that three ports
compute in the same order, so a frame is the same frame everywhere.

**k-means, a small inventory, runs collapsed.** The units are the centroids of a
k-means over the frames of the recordings (k-means++ seeding, Lloyd until the
assignments settle), 64 of them by default, and a run of one unit is one token.
A count graph learns exact repetitions: it needs symbols that recur, at a rate
close to the rate at which sounds change. Forty-odd phones at ten to fifteen a
second recur; a thousand-way codec token at fifty a second never does. Sixty-four
units with runs collapsed sit where the phones do, which is why an acoustic
model can use the graph, the search and the voice the phone model uses.

**A codebook is a data file that carries its analysis.** `data/acoustic.tsv`
holds the settings that made the frames, the centroids, each unit's count and
typical run, and the mean log-mel of the training audio. A codebook learned in
any port is read by every port, the same seed learns the same codebook in every
port (the k-means draws come from the same Mersenne Twister the coiner uses),
and a model's units mean nothing without the codebook that made them, so the
two travel together.

**Decoding is a vocoder, not a codec.** A unit is its centroid held for its
typical run; the mel frame is spread back over the linear spectrum through the
same triangles (adjacent filters sum to one, so nothing is invented at the
seams); a pulse train at a chosen pitch, mixed with noise by how voiced the
frame looks, lends its phases; the frames are overlap-added as they come. Nothing
is stored but the codebook, the output is speech-like rather than a copy, and it
streams - a model is heard as it walks - with Griffin-Lim iterations as an
optional polish once the utterance is whole.

**The bundled codebook is a bootstrap.** The package ships no recordings, so its
default codebook is learned from the synthesizer's own speech of a small corpus.
That makes everything work out of the box and makes the parity fixture possible
(every port synthesizes the same audio); it also means the default units are
the synthesizer's voice, and a codebook for real speech is learned from real
speech, which `learn` does from nothing but WAV files.

### 5.3 Alternatives rejected

* **Neural codec tokens** (EnCodec, SoundStream). The best reconstruction there
  is, and the wrong shape for this model: a framework and pretrained weights in
  every port, and a stream of high-entropy tokens at 50-75 a second that a count
  graph cannot generalise over.
* **Self-supervised units** (HuBERT or wav2vec features under k-means). Closer to
  phones than log-mel units, and again a network to run; the front end here is
  the part of that recipe that needs no network, and the k-means is the same.
* **MFCCs with deltas.** The cepstral transform mostly decorrelates for the sake
  of diagonal Gaussians, which k-means does not need; deltas double the
  dimension for a modest gain and would be the first thing to add if the
  units prove too static.
* **Units that carry duration** (`q7x3`). They would multiply the inventory and
  thin the graph; the typical run stored per unit gives decoding what it needs.

### 5.4 Costs

* The units are a recording's own: a codebook learned on one voice or one
  microphone fits others less well, and the bundled one, learned on synthetic
  speech, fits real speech least well of all.
* Pitch is not in the units (a mean-normalised log-mel frame carries little of
  it), so a replay is spoken at one pitch of the listener's choosing.
* Per-utterance normalisation needs the whole utterance before the first unit
  is known; a running mean would make hearing streamable at the cost of the first
  second, and is not done yet.
* Pure Python hears at about ten times real time and learns a 64-unit codebook
  from three minutes of audio in about three minutes; the ports do both in
  seconds.
