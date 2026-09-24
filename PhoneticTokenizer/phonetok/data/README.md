# data

| file | what it is |
|---|---|
| `core.dict` | the bundled core lexicon: some 2 000 entries in the CMU Pronouncing Dictionary's format - the common words of English, the numbers, the letters, the days and the months, the contractions, and the words of the repository's sample corpora. A word's alternative pronunciations follow it as `word(2)`, `word(3)`. The word list was written by hand (`Lexicon.core()` reads the file); the pronunciations are the dictionary's. |
| `LICENSE-cmudict` | the dictionary's BSD licence, which the redistribution of these entries requires. |

The full dictionary (135 000 words) is not bundled: `pip install cmudict` or
`PHONETOK_LEXICON=/path/to/cmudict.dict` gives the tokenizer the whole of it
(`Lexicon.default()` finds either), and everything else is sounded out by the
letter-to-sound rules in `../rules.py`.
