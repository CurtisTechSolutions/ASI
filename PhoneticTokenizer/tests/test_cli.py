"""Tests for the phonetok command line."""

import contextlib
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.cli import main  # noqa: E402


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["--core" if False else a for a in argv])
    return code, out.getvalue(), err.getvalue()


class TestCommands(unittest.TestCase):
    def test_tokenize(self):
        code, out, _ = run("tokenize", "--core", "The cat sat.")
        self.assertEqual((code, out.strip()), (0, "DH AH0 # K AE1 T # S AE1 T ."))
        code, out, _ = run("tokenize", "--core", "--level", "syllable", "--ids", "the cat sat")
        self.assertEqual(out.strip(), "8 4 9 4 10")
        code, out, _ = run("tokenize", "--core", "--json", "the cat")
        doc = json.loads(out)
        self.assertEqual(doc["tokens"], ["DH", "AH0", "#", "K", "AE1", "T"])
        self.assertEqual(doc["vocab_size"], 92)

    def test_explain_ipa_decode_pronounce(self):
        code, out, _ = run("explain", "--core", "Strengths, butter?")
        self.assertIn("Strengths", out)
        self.assertIn("lexicon", out)
        code, out, _ = run("ipa", "--core", "the cat")
        self.assertEqual(out.strip(), "ðə ˈkæt")
        code, out, _ = run("decode", "--core", "DH AH0 # K AE1 T .")
        self.assertEqual(out.strip(), "the cat.")
        code, out, _ = run("decode", "--core", "71 17 4 76 14 85")
        self.assertEqual(out.strip(), "the cat")
        code, out, _ = run("pronounce", "--core", "strength", "zebra")
        self.assertIn("strength: S T R EH1 NG K TH", out)
        self.assertIn("zebra: Z EH1 B R AH0  [rules]", out)

    def test_rhymes_affinity_coin_blend(self):
        code, out, _ = run("rhymes", "--core", "cat", "hat")
        self.assertIn("rhyme", out)
        code, out, _ = run("rhymes", "--core", "cat", "--limit", "3")
        self.assertEqual(len(out.split()), 3)
        code, out, _ = run("affinity", "--core", "s", "t")
        self.assertIn("go together", out)
        code, out, _ = run("coin", "--core", "--count", "3", "--seed", "1")
        self.assertEqual(len(out.strip().splitlines()), 3)
        code, out, _ = run("blend", "--core", "smoke", "fog")
        self.assertIn("= smog", out)

    def test_lexicon_and_errors(self):
        code, out, _ = run("lexicon", "--core", "cat", "zzz")
        self.assertEqual(code, 0)
        self.assertIn("cat: K AE1 T", out)
        self.assertIn("zzz: (not in the lexicon)", out)
        code, _, err = run("tokenize", "--core", "--lexicon", "/nonexistent/file.dict", "cat")
        self.assertEqual(code, 1)
        self.assertIn("phonetok:", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
