"""phonetok - a phonetic tokenizer: text as the sounds it is made of.

    >>> from phonetok import PhoneticTokenizer
    >>> tok = PhoneticTokenizer()
    >>> tok.text("The cat sat.")
    'DH AH0 # K AE1 T # S AE1 T .'
    >>> tok.decode(tok.encode("The cat sat."))
    'the cat sat.'
    >>> PhoneticTokenizer(level="syllable").tokens("butter")
    ['B.AH1.T', 'ER0']

Standard library only.  ``pip install cmudict`` (or ``PHONETOK_LEXICON=/path/to/cmudict.dict``)
gives it the full 135 000-word dictionary; without it the bundled core lexicon and the
letter-to-sound rules do the work.  See ``README.md`` and ``DESIGN.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .g2p import Transcriber, respell
from .lexicon import Lexicon
from .numbers import cardinal, number_words, ordinal
from .phones import (
    BOUNDARY, CONSONANTS, FEATURE_DIM, FEATURE_NAMES, IPA, PAUSES, PHONEMES, SPECIALS, SYMBOL_ID, SYMBOLS, VOWELS,
    base, distance, features, from_ipa, is_phone, is_vowel, similarity, sonority, stress_of, strip_stress, to_ipa,
)
from .phonotactics import Phonotactics
from .rules import letter_to_sound
from .syllables import ONSETS, Syllable, alliterates, is_legal_coda, is_legal_onset, rhymes, syllabify
from .tokenizer import CONSTITUENT, LEVELS, PHONEME, SYLLABLE, WORD, PhoneticTokenizer, Token, Vocab, parse_token

__all__ = [
    "__version__", "PhoneticTokenizer", "Token", "Vocab", "parse_token", "LEVELS", "PHONEME", "CONSTITUENT",
    "SYLLABLE", "WORD",
    "Transcriber", "respell", "Lexicon", "Phonotactics", "letter_to_sound",
    "Syllable", "syllabify", "rhymes", "alliterates", "is_legal_onset", "is_legal_coda", "ONSETS",
    "PHONEMES", "VOWELS", "CONSONANTS", "SYMBOLS", "SYMBOL_ID", "SPECIALS", "BOUNDARY", "PAUSES", "IPA",
    "FEATURE_NAMES", "FEATURE_DIM",
    "base", "stress_of", "strip_stress", "is_phone", "is_vowel", "sonority", "features", "distance", "similarity",
    "to_ipa", "from_ipa", "cardinal", "ordinal", "number_words",
]
