"""Test package: make ``radixnet`` importable from the project root."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# the phonetic tokenizer lives beside this project; a checkout imports it from there
PHONETIC = os.path.join(os.path.dirname(ROOT), "PhoneticTokenizer")
if os.path.isdir(PHONETIC) and PHONETIC not in sys.path:
    sys.path.append(PHONETIC)
