"""Tests for the word n-gram model (``radixnet.wordnet``) and its alphabet.

``../SPEC-WordNGrams.md`` §10 is the table these follow: the alphabet, the
encoder, the graph, compression, prediction, the file.  The cross-language half
of that table lives in ``test_go_parity.py`` and ``test_rust_parity.py``.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.countnet import COUNT_MODEL_FORMAT, CountRewardNet  # noqa: E402
from radixnet.encoding import (  # noqa: E402
    MAX_WORDS,
    SURROGATE_HI,
    SURROGATE_LO,
    UNKNOWN_WORD,
    WORD_BASE,
    Vocabulary,
    split_words,
    symbol_word,
    word_symbol,
)
from radixnet.model import load_model, model_class, model_from_dict, new_model  # noqa: E402
from radixnet.wordnet import WORD_MODEL_FORMAT, WORD_UNITS, WordGraph, WordNGramNet  # noqa: E402

TEXTS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ate the bone in the garden",
    "a bird sang in the garden",
]


def taught(texts=TEXTS, epochs=3, seed=0) -> WordNGramNet:
    model = WordNGramNet(seed=seed)
    model.train(texts, epochs=epochs)
    return model


class TestAlphabet(unittest.TestCase):
    """Every id round-trips through its code point; the surrogate block is skipped; the cap is what §3 says."""

    def test_every_id_round_trips_through_its_code_point(self):
        for word_id in (0, 1, 2, 1000, 55039, 55040, 55041, MAX_WORDS - 1):
            self.assertEqual(symbol_word(word_symbol(word_id)), word_id)

    def test_the_first_symbol_is_past_latin_1_and_the_surrogates_are_skipped(self):
        self.assertEqual(ord(word_symbol(0)), WORD_BASE)
        below = ord(word_symbol(SURROGATE_LO - WORD_BASE - 1))
        first_above = ord(word_symbol(SURROGATE_LO - WORD_BASE))
        self.assertEqual(below, SURROGATE_LO - 1)
        self.assertEqual(first_above, SURROGATE_HI + 1)
        for code in range(SURROGATE_LO, SURROGATE_HI + 1):
            with self.assertRaises(ValueError):
                symbol_word(chr(code))

    def test_the_cap_is_every_code_point_above_latin_1_that_is_not_a_surrogate(self):
        self.assertEqual(MAX_WORDS, 1_111_808)
        self.assertEqual(MAX_WORDS, 0x110000 - WORD_BASE - 2048)
        self.assertEqual(ord(word_symbol(MAX_WORDS - 1)), 0x10FFFF)
        with self.assertRaises(ValueError):
            word_symbol(MAX_WORDS)

    def test_a_character_is_not_a_word_symbol(self):
        for ch in ("a", " ", "<", "ÿ"):
            with self.assertRaises(ValueError):
                symbol_word(ch)

    def test_the_unknown_word_is_id_zero_and_is_always_there(self):
        v = Vocabulary()
        self.assertEqual(len(v), 1)
        self.assertEqual(v.words[0], UNKNOWN_WORD)
        self.assertEqual(v.id("never read"), 0)

    def test_a_vocabulary_loads_back_in_order(self):
        v = Vocabulary([UNKNOWN_WORD, "the", "cat", "sat"])
        self.assertEqual(v.to_list(), [UNKNOWN_WORD, "the", "cat", "sat"])
        self.assertEqual([v.id(w) for w in ("the", "cat", "sat")], [1, 2, 3])
        with self.assertRaises(ValueError):
            Vocabulary(["the", "cat"])  # a vocabulary starts with the unknown word
        with self.assertRaises(ValueError):
            Vocabulary([UNKNOWN_WORD, "the", "the"])


class TestTheSplit(unittest.TestCase):
    """Whitespace separates words; punctuation stays attached and case is kept (§4)."""

    def test_punctuation_and_case_are_part_of_the_word(self):
        self.assertEqual(split_words("The mat. the mat"), ["The", "mat.", "the", "mat"])

    def test_the_rule_is_the_unicode_property_so_the_three_ports_agree(self):
        # Python calls U+001C..U+001F whitespace; Unicode's White_Space property does not
        self.assertEqual(split_words("a\x1cb"), ["a\x1cb"])
        self.assertEqual(split_words("a\tb\nc\r\nd\x0be\x0cf g\u0085h"), list("abcdefgh"))
        self.assertEqual(split_words("   "), [])


class TestTheEncoder(unittest.TestCase):
    """Windows of three words, whitespace normalised, an unread word ``<unk>`` (§5, §6)."""

    def test_four_words_are_two_windows_of_three(self):
        model = WordNGramNet(seed=0)
        symbols = model.symbols("the cat sat on", grow=True)
        grams = model.encoder.encode(symbols)
        self.assertEqual(len(symbols), 4)
        self.assertEqual([model.words(g) for g in grams], ["the cat sat", "cat sat on"])

    def test_a_round_trip_normalises_whitespace_and_nothing_else(self):
        model = WordNGramNet(seed=0)
        text = "a  b\n c"
        self.assertEqual(model.words(model.symbols(text, grow=True)), "a b c")
        self.assertEqual(model.normalise(text), "a b c")
        self.assertEqual(model.words(model.symbols("a b c")), "a b c")

    def test_an_unread_word_is_a_new_id_while_training_and_unknown_after(self):
        model = WordNGramNet(seed=0)
        model.train(["the cat sat on the mat"], epochs=1)
        size = len(model.vocabulary)
        self.assertEqual(model.words(model.symbols("the qux sat")), "the <unk> sat")
        self.assertEqual(len(model.vocabulary), size)  # predicting never grows it
        model.train(["the qux sat on the mat"], epochs=1)
        self.assertEqual(len(model.vocabulary), size + 1)
        self.assertEqual(model.words(model.symbols("the qux sat")), "the qux sat")

    def test_two_different_unread_words_are_the_same_symbol(self):
        model = taught()
        self.assertEqual(model.score("the qux sat"), model.score("the quux sat"))

    def test_a_text_of_fewer_than_three_words_is_skipped(self):
        model = WordNGramNet(seed=0)
        record = model.train(["a b", "the cat sat on the mat"], epochs=1)[-1]
        self.assertEqual(record["skipped_short"], 1)
        self.assertEqual(record["transitions"], 2)  # the six-word text alone: START, its one node, END


class TestTheGraph(unittest.TestCase):
    """The same invariants the character model holds to, over words."""

    def test_the_invariants_hold_and_every_text_round_trips(self):
        model = taught()
        symbols = [model.symbols(t) for t in TEXTS]
        model.graph.check_invariants(texts=symbols, compressed=True)

    def test_a_repeated_phrase_becomes_one_node_whose_label_is_that_phrase(self):
        model = WordNGramNet(seed=0)
        model.train(["the cat sat on the mat", "a dog sat on the mat"], epochs=1)
        labels = {
            model.graph.text_of(model.graph.labels[n])
            for n in range(len(model.graph.labels))
            if model.graph.alive[n]
        }
        self.assertIn("sat on the mat", labels)  # the shared tail, merged into one node
        self.assertIn("the cat sat on", labels)

    def test_the_words_are_what_the_graph_holds(self):
        model = taught()
        rows = model.top_words(limit=3)
        self.assertEqual(rows[0]["word"], "the")
        self.assertTrue(all(r["trigrams"] > 0 for r in rows))
        self.assertEqual([r["id"] for r in rows], [model.vocabulary.id(r["word"]) for r in rows])


class TestPrediction(unittest.TestCase):
    """A word model continues a prefix with words, and lengths are counted in words."""

    def test_a_prediction_is_words(self):
        model = taught()
        found = model.predict("the cat sat on", length=2, k=3)
        self.assertEqual(found.full_text, "the cat sat on " + found.text)
        self.assertIn(found.text, ("the mat", "the rug"))
        self.assertTrue(all(" " in label or label.startswith("<") for label in found.labels))

    def test_length_is_counted_in_words(self):
        model = taught()
        found = model.predict("the cat sat", length=3, max_length=3)
        self.assertLessEqual(len(split_words(found.text)), 3)

    def test_a_generated_text_is_whole_and_spaced(self):
        model = taught()
        for result in model.generate(count=2, mode="beam", max_length=12):
            self.assertEqual(result.text, result.full_text)
            self.assertNotIn("  ", result.text)
            self.assertEqual(result.text, model.normalise(result.text))

    def test_the_prefix_comes_back_normalised(self):
        model = taught()
        found = model.predict("the   cat\tsat on", length=2)
        self.assertTrue(found.full_text.startswith("the cat sat on "))

    def test_score_charges_an_unread_word_as_unknown_and_counts_in_words(self):
        model = taught()
        known = model.score("the cat sat on the mat")
        unknown = model.score("the cat sat on the flurb")
        self.assertEqual(known["unknown_transitions"], 0)
        self.assertGreater(unknown["unknown_transitions"], 0)
        self.assertLess(unknown["log_prob"], known["log_prob"])
        self.assertEqual(known["chars"], 6)  # words, not characters
        self.assertAlmostEqual(known["per_char"], known["log_prob"] / 6)

    def test_the_least_punished_traversal_works_over_words(self):
        model = taught()
        model.punish(["the dog sat on the mat"], epochs=1, strength=3.0)
        rewarded = model.predict("the dog sat", length=3, traversal="reward")
        punished = model.predict("the dog sat", length=3, traversal="least-punished")
        self.assertEqual(punished.traversal, "least-punished")
        self.assertGreaterEqual(rewarded.punish, punished.punish)


class TestFeedbackAndCorrections(unittest.TestCase):
    """Feedback moves rewards over word paths, and a correction is aligned word by word."""

    def test_reward_and_punish_move_the_same_paths(self):
        model = taught()
        before = model.score("the cat sat on the mat")["log_prob"]
        model.reward(["the cat sat on the mat"], epochs=1, strength=2.0)
        after = model.score("the cat sat on the mat")["log_prob"]
        self.assertGreater(after, before)

    def test_a_correction_is_aligned_word_by_word(self):
        model = taught()
        result = model.correct("the dog sat on the mat", "the dog sat on the rug")
        self.assertEqual(result["wrong_chars"], 1)  # one word differs
        self.assertEqual(result["right_chars"], 1)
        self.assertEqual([c["op"] for c in result["changes"]], ["replace"])
        self.assertEqual(result["changes"][0]["wrong"], "mat")
        self.assertEqual(result["changes"][0]["right"], "rug")


class TestTheFile(unittest.TestCase):
    """A new format, the vocabulary in id order, and every older reader refusing it (§8)."""

    def test_a_word_model_round_trips_through_its_document(self):
        model = taught()
        model.reward(["the cat sat on the mat"], epochs=1, strength=1.5)
        doc = model.to_dict()
        self.assertEqual(doc["format"], WORD_MODEL_FORMAT)
        self.assertEqual(doc["kind"], "word")
        self.assertEqual(doc["graph"]["units"], WORD_UNITS)
        self.assertEqual(doc["graph"]["vocabulary"], model.vocabulary.to_list())
        again = model_from_dict(doc)
        self.assertIsInstance(again, WordNGramNet)
        self.assertEqual(again.vocabulary.to_list(), model.vocabulary.to_list())
        self.assertEqual(again.stats(), model.stats())
        self.assertEqual(again.predict("the cat sat on", length=2).text, model.predict("the cat sat on", length=2).text)
        self.assertEqual(again.score("the cat sat on the mat"), model.score("the cat sat on the mat"))

    def test_the_vocabulary_comes_back_in_order(self):
        model = taught()
        words = model.vocabulary.to_list()
        again = model_from_dict(model.to_dict())
        self.assertEqual(again.vocabulary.to_list(), words)
        self.assertEqual([again.vocabulary.id(w) for w in words], list(range(len(words))))

    def test_a_count_reader_refuses_a_word_file_and_a_word_reader_refuses_a_count_file(self):
        word = taught().to_dict()
        count = CountRewardNet(seed=0)
        count.train(TEXTS, epochs=1)
        with self.assertRaises(ValueError) as caught:
            CountRewardNet.from_dict(word)
        self.assertIn(COUNT_MODEL_FORMAT, str(caught.exception))
        with self.assertRaises(ValueError):
            WordNGramNet.from_dict(count.to_dict())

    def test_a_word_graph_document_carries_its_vocabulary(self):
        doc = taught().to_dict()
        graph = dict(doc["graph"])
        graph.pop("vocabulary")
        with self.assertRaises(ValueError):
            WordGraph.from_dict(graph)
        short = dict(doc["graph"])
        short["vocabulary"] = [UNKNOWN_WORD, "the"]
        with self.assertRaises(ValueError):
            WordGraph.from_dict(short)

    def test_a_saved_file_loads_as_a_word_model(self):
        model = taught()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.word.json")
            model.save(path)
            again = load_model(path)
        self.assertIsInstance(again, WordNGramNet)
        self.assertEqual(again.vocabulary.to_list(), model.vocabulary.to_list())


class TestTheKind(unittest.TestCase):
    """The registry, the units and what the reports say."""

    def test_the_kind_is_registered(self):
        self.assertIs(model_class("word"), WordNGramNet)
        self.assertIsInstance(new_model("word"), WordNGramNet)

    def test_stats_name_the_units(self):
        model = taught()
        stats = model.stats()
        self.assertEqual(stats["units"], WORD_UNITS)
        self.assertEqual(stats["vocabulary"], len(model.vocabulary))
        self.assertEqual(WordNGramNet.units, WORD_UNITS)
        self.assertEqual(CountRewardNet.units, "chars")

    def test_a_node_report_is_words(self):
        model = taught()
        rows = model.node_ratios(limit=2)
        for row in rows:
            self.assertNotIn("Ā", row["label"])
            for side in ("from", "to"):
                for edge in row[side]:
                    self.assertNotIn("Ā", edge["label"])


if __name__ == "__main__":
    unittest.main()
