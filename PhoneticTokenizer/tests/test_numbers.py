"""Tests for phonetok.numbers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok.numbers import cardinal, digits, number_words, ordinal, split_alphanumeric  # noqa: E402


class TestNumbers(unittest.TestCase):
    def test_cardinal(self):
        self.assertEqual(cardinal(0), "zero")
        self.assertEqual(cardinal(13), "thirteen")
        self.assertEqual(cardinal(42), "forty two")
        self.assertEqual(cardinal(100), "one hundred")
        self.assertEqual(cardinal(1234), "one thousand two hundred thirty four")
        self.assertEqual(cardinal(1000000), "one million")
        self.assertEqual(cardinal(-7), "minus seven")
        self.assertEqual(cardinal(10 ** 15), "one quadrillion")

    def test_ordinal(self):
        self.assertEqual(ordinal(1), "first")
        self.assertEqual(ordinal(2), "second")
        self.assertEqual(ordinal(3), "third")
        self.assertEqual(ordinal(12), "twelfth")
        self.assertEqual(ordinal(20), "twentieth")
        self.assertEqual(ordinal(21), "twenty first")
        self.assertEqual(ordinal(100), "one hundredth")

    def test_number_words(self):
        self.assertEqual(number_words("42"), "forty two")
        self.assertEqual(number_words("3.14"), "three point one four")
        self.assertEqual(number_words("1,000"), "one thousand")
        self.assertEqual(number_words("21st"), "twenty first")
        self.assertEqual(number_words("50%"), "fifty percent")
        self.assertEqual(number_words("$5"), "five dollars")
        self.assertEqual(number_words("$1"), "one dollar")
        self.assertEqual(number_words("$2.50"), "two dollars fifty cents")
        self.assertEqual(number_words("-7"), "minus seven")
        self.assertIsNone(number_words("mp3"))
        self.assertIsNone(number_words("abc"))
        self.assertIsNone(number_words("3.1st"))

    def test_split_alphanumeric(self):
        self.assertEqual(split_alphanumeric("mp3"), ["mp", "3"])
        self.assertEqual(split_alphanumeric("3d"), ["3", "d"])
        self.assertEqual(split_alphanumeric("b2b"), ["b", "2", "b"])
        self.assertEqual(split_alphanumeric("abc"), ["abc"])
        self.assertEqual(digits("007"), "zero zero seven")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
