# data

| file | what it is |
|---|---|
| `core.dict` | the bundled core lexicon: some 2 000 entries in the CMU Pronouncing Dictionary's format - the common words of English, the numbers, the letters, the days and the months, the contractions, and the words of the repository's sample corpora. A word's alternative pronunciations follow it as `word(2)`, `word(3)`. The word list was written by hand (`Lexicon.core()` reads the file); the pronunciations are the dictionary's. |
| `LICENSE-cmudict` | the dictionary's BSD licence, which the redistribution of these entries requires. |
| `voice.tsv` | the voice: per phoneme, the formant targets and bandwidths, the duration, the voicing and the noise the synthesizer speaks it with (`synth.py`). |
| `acoustic.tsv` | the bundled codebook of acoustic units: 64 centroids over 40 log-mel bands, each unit's count and typical run, the analysis settings that made the frames and the mean log-mel of the training audio (`acoustic.py`). Learned from the synthesizer's speech of 28 sentences in two voices by `../../tests/make_codebook.py`; a codebook learned from real recordings (`phonetok learn`) suits real speech better. |

The full dictionary (135 000 words) is not bundled: `pip install cmudict` or
`PHONETOK_LEXICON=/path/to/cmudict.dict` gives the tokenizer the whole of it
(`Lexicon.default()` finds either), and everything else is sounded out by the
letter-to-sound rules in `../rules.py`.
