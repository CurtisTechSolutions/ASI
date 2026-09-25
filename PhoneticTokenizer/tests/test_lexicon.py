"""Tests for phonetok.lexicon."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.lexicon import CORE_PATH, ENV_LEXICON, Lexicon, find_full_dictionary, parse_entry  # noqa: E402


class TestParsing(unittest.TestCase):
    def test_entries(self):
        self.assertEqual(parse_entry("cat K AE1 T"), ("cat", ("K", "AE1", "T")))
        self.assertEqual(parse_entry("cat(2)  K AE1 T # a comment"), ("cat", ("K", "AE1", "T")))
        self.assertEqual(parse_entry("CAT k ae1 t"), ("cat", ("K", "AE1", "T")))
        self.assertIsNone(parse_entry(";;; a comment"))
        self.assertIsNone(parse_entry(""))
        self.assertIsNone(parse_entry("word"))


class TestCore(unittest.TestCase):
    def test_the_core_is_there(self):
        lex = Lexicon.core()
        self.assertGreater(len(lex), 1500)
        self.assertEqual(lex.lookup("the"), ("DH", "AH0"))
        self.assertEqual(lex.lookup("cat"), ("K", "AE1", "T"))
        self.assertEqual(lex.lookup("THE"), ("DH", "AH0"))
        self.assertIn("the", lex)
        self.assertNotIn("qzxv", lex)
        self.assertIsNone(lex.lookup("qzxv"))
        self.assertGreater(len(lex.pronunciations("the")), 1)
        self.assertEqual(lex.pronunciations("qzxv"), [])
        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(CORE_PATH), "LICENSE-cmudict")))

    def test_every_core_entry_is_phones(self):
        from phonetok.phones import is_phone

        for word, phones in Lexicon.core().items():
            self.assertTrue(all(is_phone(p) for p in phones), (word, phones))

    def test_homophones(self):
        lex = Lexicon.core()
        self.assertEqual(lex.spellings(["T", "UW1"]), ["to", "too", "two"])
        self.assertEqual(lex.spellings(["R", "AY1", "T"]), ["right", "rite", "wright", "write"])
        self.assertEqual(lex.spellings(["T", "UW0"]), [])
        self.assertEqual(lex.spellings(["T", "UW0"], stress=False), ["to", "too", "two"])


class TestEditing(unittest.TestCase):
    def test_add_and_remove(self):
        lex = Lexicon()
        self.assertEqual(len(lex), 0)
        lex.add("radix", ["R", "EY1", "D", "IH0", "K", "S"])
        self.assertEqual(lex.lookup("Radix"), ("R", "EY1", "D", "IH0", "K", "S"))
        lex.add("radix", ["R", "AE1", "D", "IH0", "K", "S"], first=True)
        self.assertEqual(lex.lookup("radix")[1], "AE1")
        self.assertEqual(len(lex.pronunciations("radix")), 2)
        self.assertEqual(lex.spellings(["R", "AE1", "D", "IH0", "K", "S"]), ["radix"])
        self.assertTrue(lex.remove("radix"))
        self.assertFalse(lex.remove("radix"))
        with self.assertRaises(ValueError):
            lex.add("bad", ["X"])

    def test_extend_does_not_duplicate(self):
        lex = Lexicon()
        self.assertEqual(lex.extend([("a", ["AH0"]), ("a", ["AH0"]), ("a", ["EY1"])]), 2)
        self.assertEqual(lex.pronunciations("a"), [("AH0",), ("EY1",)])


class TestSources(unittest.TestCase):
    def test_load_a_file_and_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mini.dict")
            with open(path, "w", encoding="utf-8") as f:
                f.write(";;; mini\nzork Z AO1 R K\nzork(2) Z AO2 R K\n")
            lex = Lexicon.load(path)
            self.assertEqual(len(lex), 1)
            self.assertEqual(lex.pronunciations("zork"), [("Z", "AO1", "R", "K"), ("Z", "AO2", "R", "K")])
            with mock.patch.dict(os.environ, {ENV_LEXICON: path}):
                found = find_full_dictionary()
                self.assertIsNotNone(found)
                found[1].close()
                full = Lexicon.default()
                self.assertEqual(full.lookup("zork"), ("Z", "AO1", "R", "K"))
                self.assertEqual(full.lookup("cat"), ("K", "AE1", "T"))
                self.assertIn(f"file {path}", full.describe())

    def test_default_without_a_full_dictionary_is_the_core(self):
        with mock.patch.dict(os.environ, {ENV_LEXICON: ""}):
            with mock.patch("phonetok.lexicon.importlib.util.find_spec", return_value=None):
                self.assertIsNone(find_full_dictionary())
                self.assertEqual(len(Lexicon.default()), len(Lexicon.core()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
