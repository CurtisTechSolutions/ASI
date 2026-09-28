"""Test package: make ``radixpair`` importable from the project root, and the phonetic tokenizer from beside it."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PHONETIC = os.path.join(os.path.dirname(ROOT), "PhoneticTokenizer")
if os.path.isdir(PHONETIC) and PHONETIC not in sys.path:
    sys.path.append(PHONETIC)
