"""Tests for phonetok.rules: the letter-to-sound rules and the stress they get."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.lexicon import core_entries  # noqa: E402
from phonetok.phones import strip_stress  # noqa: E402
from phonetok.rules import RULES, Rule, apply_rules, assign_stress, letter_to_sound, rule_count  # noqa: E402


class TestTheEngine(unittest.TestCase):
    def test_rules_parse(self):
        r = Rule.parse("#:[E]D =")
        self.assertEqual((r.left, r.match, r.right, r.phones), ("#:", "E", "D ", ()))
        r = Rule.parse(" :[ANY]=EH N IY0")
        self.assertEqual((r.left, r.match, r.right, r.phones), (" :", "ANY", "", ("EH", "N", "IY0")))
        with self.assertRaises(ValueError):
            Rule.parse("no brackets")

    def test_every_letter_has_a_bare_rule_last(self):
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            last = RULES[letter][-1]
            self.assertEqual((last.left, last.match, last.right), ("", letter, ""), letter)
        self.assertGreater(rule_count(), 250)

    def test_contexts(self):
        """The silent e, the suffix, the sibilant, the voiced consonant, the word edge."""
        self.assertEqual(apply_rules("bake"), ["B", "EY", "K"])
        self.assertEqual(apply_rules("baked"), ["B", "EY", "K", "T"])
        self.assertEqual(apply_rules("baking"), ["B", "EY", "K", "IH", "NG"])
        self.assertEqual(apply_rules("boxes"), ["B", "AA", "K", "S", "IH0", "Z"])
        self.assertEqual(apply_rules("loved"), ["L", "AH", "V", "D"])
        self.assertEqual(apply_rules("waited"), ["W", "EY", "T", "IH0", "D"])
        self.assertEqual(apply_rules("knight"), ["N", "AY", "T"])
        self.assertEqual(apply_rules("cheese"), ["CH", "IY", "Z"])
        self.assertEqual(apply_rules("nation"), ["N", "EY", "SH", "AH0", "N"])
        self.assertEqual(apply_rules("quick"), ["K", "W", "IH", "K"])
        self.assertEqual(apply_rules("physics"), ["F", "IH", "Z", "IH", "K", "S"])

    def test_non_letters_are_skipped(self):
        self.assertEqual(apply_rules("ca3t"), ["K", "AE", "T"])
        self.assertEqual(apply_rules(""), [])


class TestStress(unittest.TestCase):
    def test_one_syllable(self):
        self.assertEqual(letter_to_sound("cat"), ["K", "AE1", "T"])
        self.assertEqual(letter_to_sound("strength"), ["S", "T", "R", "EH1", "NG", "TH"])

    def test_exactly_one_primary_stress(self):
        for word in ("computer", "understand", "beautiful", "photograph", "tokenizer", "syllable", "the", "a"):
            phones = letter_to_sound(word)
            self.assertEqual(sum(p.endswith("1") for p in phones), 1, (word, phones))

    def test_a_marked_schwa_stays(self):
        self.assertEqual(letter_to_sound("nation"), ["N", "EY1", "SH", "AH0", "N"])
        self.assertEqual(letter_to_sound("tuna"), ["T", "UW1", "N", "AH0"])

    def test_suffix_and_prefix_stress(self):
        self.assertEqual(letter_to_sound("information")[-5:], ["M", "EY1", "SH", "AH0", "N"])
        self.assertEqual(letter_to_sound("rebuild"), ["R", "IH0", "B", "IH1", "L", "D"])
        self.assertEqual(letter_to_sound("specific"), ["S", "P", "AH0", "S", "IH1", "F", "IH0", "K"])

    def test_assign_stress_keeps_marks(self):
        self.assertEqual(assign_stress("x", ["K", "AH0"]), ["K", "AH1"])  # something must carry the stress
        self.assertEqual(assign_stress("x", ["K", "AE", "T", "AH0"]), ["K", "AE1", "T", "AH0"])
        self.assertEqual(assign_stress("x", ["K"]), ["K"])


class TestAgainstTheLexicon(unittest.TestCase):
    """The rules alone against the core lexicon: a floor, so that a change to them cannot quietly regress."""

    def test_accuracy_floor(self):
        seen = set()
        words = right = 0
        errors = total = 0
        for word, gold in core_entries():
            if word in seen or not word.isalpha() or len(word) < 3:
                continue
            seen.add(word)
            got = strip_stress(letter_to_sound(word))
            gold = strip_stress(gold)
            words += 1
            right += got == gold
            errors += _edit(got, gold)
            total += len(gold)
        self.assertGreater(words, 1000)
        self.assertGreater(right / words, 0.55, f"{right}/{words} words right")
        self.assertLess(errors / total, 0.16, f"phone error rate {errors / total:.3f}")


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
