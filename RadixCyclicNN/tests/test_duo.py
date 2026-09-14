"""Tests for radixnet.duo - the pair at output time: the positive model writes, the negative one vetoes.

The filter is the GAN applied to an answer instead of to a training step: two
signals can reject a candidate (blame from the failure structure and the
likelihood ratio between the two networks), both behind a coverage gate so
text the tutor has never failed is never vetoed on a hunch.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import RadixNet  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.duo import FilterConfig, NegativeFilter  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat ran to the door",
    "the dog ate the bone",
]
FAILURE = "the the the the the"


def pair(config=None, kind=RadixNet):
    positive = kind(seed=1)
    positive.train(CORPUS, epochs=3, lr=0.5, batch_size=4)
    negative = NegativeNet(seed=2)
    negative.blame([FAILURE], reason="repetition", source="review", note="it repeats the same word")
    return NegativeFilter(positive, negative, config)


class TestConfig(unittest.TestCase):
    def test_validation(self):
        FilterConfig().validate()
        with self.assertRaises(ValueError):
            FilterConfig(over_sample=0).validate()
        with self.assertRaises(ValueError):
            FilterConfig(min_coverage=1.5).validate()
        with self.assertRaises(ValueError):
            FilterConfig(spans=-1).validate()

    def test_the_two_models_must_differ(self):
        negative = NegativeNet(seed=1)
        with self.assertRaises(ValueError):
            NegativeFilter(negative, negative)
        with self.assertRaises(TypeError):
            NegativeFilter(RadixNet(seed=1), RadixNet(seed=2))

    def test_describe(self):
        described = pair().describe()
        self.assertEqual(described["positive"]["kind"], "radix")
        self.assertEqual(described["negative"]["kind"], "negative")
        self.assertEqual(described["config"]["over_sample"], 3)
        self.assertIn("NegativeFilter", repr(pair()))


class TestJudge(unittest.TestCase):
    def test_a_known_failure_is_rejected_with_its_reason(self):
        verdict = pair().judge(FAILURE)
        self.assertEqual(verdict["decision"], "reject")
        self.assertEqual(verdict["rule"], "blame")
        self.assertEqual(verdict["reasons"][0]["reason"], "repetition")
        self.assertGreater(verdict["ratio"], 0)  # it reads more like failure than like the corpus
        self.assertTrue(verdict["spans"])
        self.assertIn("rejected", verdict["why"])

    def test_corpus_text_passes(self):
        verdict = pair().judge(CORPUS[0])
        self.assertNotEqual(verdict["decision"], "reject")
        self.assertLess(verdict["ratio"], 0)
        self.assertLess(verdict["negative"], verdict["positive"])

    def test_unseen_text_passes(self):
        verdict = pair().judge("a wholly unrelated sentence")
        self.assertEqual(verdict["decision"], "pass")
        self.assertIsNone(verdict["rule"])
        self.assertEqual(verdict["blame"], 0.0)

    def test_the_ratio_rule_can_reject_on_its_own(self):
        duo = pair(FilterConfig(threshold=1e9, min_coverage=0.0, ratio=0.0))
        verdict = duo.judge(FAILURE)
        self.assertEqual(verdict["decision"], "reject")
        self.assertEqual(verdict["rule"], "ratio")
        self.assertIn("nats/char", verdict["why"])

    def test_the_ratio_rule_can_be_turned_off(self):
        duo = pair(FilterConfig(threshold=1e9, min_coverage=0.0, ratio=None))
        self.assertEqual(duo.judge(FAILURE)["decision"], "suspect")

    def test_the_peak_rule_catches_a_single_corrected_fragment(self):
        positive = RadixNet(seed=1)
        positive.train(CORPUS, epochs=3, lr=0.5, batch_size=4)
        negative = NegativeNet(seed=2)
        negative.correct("the cat sit on the mat", "the cat sits on the mat", reason="agreement", severity=1.5)
        lenient = NegativeFilter(positive, negative, FilterConfig(ratio=None))
        verdict = lenient.judge("the cat sit on the mat")
        self.assertEqual(verdict["decision"], "suspect")  # one wrong fragment in a long sentence
        self.assertAlmostEqual(verdict["peak"], 1.5)
        strict = NegativeFilter(positive, negative, FilterConfig(ratio=None, peak=1.0))
        rejected = strict.judge("the cat sit on the mat")
        self.assertEqual((rejected["decision"], rejected["rule"]), ("reject", "peak"))
        self.assertIn("sit", rejected["why"])
        self.assertEqual(strict.judge("nothing like it at all")["decision"], "pass")
        with self.assertRaises(ValueError):
            FilterConfig(peak=float("nan")).validate()

    def test_the_coverage_gate_protects_unknown_text(self):
        duo = pair(FilterConfig(threshold=0.0, min_coverage=1.0, ratio=0.0))
        self.assertEqual(duo.judge("nothing here ever failed")["decision"], "pass")

    def test_strict_drops_suspects(self):
        duo = pair(FilterConfig(threshold=1e9, ratio=None, strict=True))
        verdict = duo.judge(CORPUS[0])
        if verdict["coverage"] > 0:
            self.assertEqual(verdict["decision"], "reject")
            self.assertEqual(verdict["rule"], "suspect")
            self.assertIn("strict", verdict["why"])


class TestFilter(unittest.TestCase):
    def test_filter_splits_kept_from_rejected(self):
        outcome = pair().filter([FAILURE, CORPUS[0], "a wholly unrelated sentence"])
        self.assertEqual(outcome["rejected"], [FAILURE])
        self.assertEqual(len(outcome["kept"]), 2)
        self.assertAlmostEqual(outcome["rate"], 2 / 3)
        self.assertEqual(len(outcome["verdicts"]), 3)
        self.assertEqual(pair().filter(FAILURE)["rejected"], [FAILURE])  # one string is one text

    def test_learn_is_off_by_default(self):
        duo = pair()
        before = duo.negative.graph.total_blame
        duo.filter([FAILURE])
        self.assertEqual(duo.negative.graph.total_blame, before)

    def test_learn_blames_what_it_rejects(self):
        duo = pair(FilterConfig(learn=True, reason="filtered"))
        before = duo.negative.graph.total_blame
        duo.filter([FAILURE])
        self.assertGreater(duo.negative.graph.total_blame, before)
        self.assertIn("filtered", [r["reason"] for r in duo.negative.reasons()])
        self.assertEqual(duo.negative.recent(1)[0]["source"], "filter")


class TestOutput(unittest.TestCase):
    def test_generate_returns_the_cleanest_survivors(self):
        duo = pair()
        out = duo.generate(count=2, max_length=30, seed=5)
        self.assertLessEqual(len(out["texts"]), 2)
        self.assertEqual(out["asked"], 6)
        self.assertEqual(len(out["verdicts"]), out["candidates"])
        self.assertTrue(set(out["texts"]) <= set(out["kept"]))
        risk_of = {v["text"]: v["risk"] for v in out["verdicts"]}
        risks = [risk_of[text] for text in out["texts"]]  # the cleanest come back first
        self.assertEqual(risks, sorted(risks))
        self.assertEqual(duo.generate(count=0)["texts"], [])
        with self.assertRaises(ValueError):
            duo.generate(count=-1)
        with self.assertRaises(ValueError):
            duo.generate(count=1, over_sample=0)

    def test_generate_can_come_back_empty(self):
        duo = pair(FilterConfig(threshold=0.0, min_coverage=0.0, over_sample=1))
        duo.negative.blame(CORPUS, reason="gibberish")  # everything the generator can write is now known-bad
        out = duo.generate(count=3, max_length=30, seed=7)
        self.assertEqual(out["texts"], [])
        self.assertTrue(out["rejected"])
        self.assertEqual(out["rate"], 0.0)

    def test_predict_filters_the_continuations_and_warns(self):
        duo = pair()
        out = duo.predict("the ", length=10, k=3)
        self.assertEqual(out["prefix"], "the ")
        self.assertEqual(len(out["verdicts"]), out["candidates"])
        self.assertTrue(all(v["text"].startswith("the ") for v in out["verdicts"]))
        self.assertTrue(out["warning"] is None or out["warning"].startswith("the "))
        if out["text"] is not None:
            self.assertIn(out["text"], out["kept"])

    def test_the_count_model_can_be_the_positive_half(self):
        duo = pair(kind=CountRewardNet)
        self.assertEqual(duo.judge(FAILURE)["decision"], "reject")
        self.assertLessEqual(len(duo.generate(count=1, max_length=30, seed=3)["texts"]), 1)

    def test_teach_goes_through_the_tutor(self):
        duo = pair()
        report = duo.teach([
            {"text": "the cat cat cat", "rating": 2.0, "verdict": "fail", "critique": "it repeats words"},
            {"text": CORPUS[1], "rating": 8.0, "verdict": "pass", "critique": "no flaw found"},
        ])
        self.assertEqual(report["blamed"], 1)
        self.assertEqual(report["reasons"], {"repetition": 1})


if __name__ == "__main__":
    unittest.main()
