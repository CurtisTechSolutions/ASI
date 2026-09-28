"""PairModel: the verbs, the feedback primitives, settings and persistence."""

import gzip
import json
import os
import tempfile
import unittest

import unittest as _ut

from radixpair.codec import CharsCodec, phonetok_module
from radixpair.model import PairModel, load_model, prime
from radixpair.pair import Settings

try:
    phonetok_module()
    HAVE_PHONETOK = True
except ValueError:
    HAVE_PHONETOK = False

TEXTS = ["the cat sat on the mat", "the cat sat on the log", "the dog ate the bone", "a cat and a dog"]


class TestVerbs(unittest.TestCase):
    def test_train_lowers_the_bits_of_what_was_read(self):
        m = prime("chars", L=4)
        before = m.score(TEXTS[0]).bits
        record = m.train(TEXTS)
        self.assertLess(m.score(TEXTS[0]).bits, before)
        self.assertEqual(record["texts"], 4)
        self.assertEqual(record["call"], "train")
        self.assertEqual(m.history[-1]["call"], "train")
        self.assertLess(m.score(TEXTS[0]).bits, m.score("zq xk vv").bits)

    def test_reward_lowers_and_punish_raises(self):
        m = prime("chars", L=4)
        m.train(TEXTS)
        base = m.score(TEXTS[1]).bits
        m.reward(TEXTS[1], strength=2.0)
        rewarded = m.score(TEXTS[1]).bits
        self.assertLess(rewarded, base)
        m.punish(TEXTS[1], strength=4.0)
        self.assertGreater(m.score(TEXTS[1]).bits, rewarded)
        self.assertEqual(m.pair.count.texts, 4)        # neither read anything

    def test_read_counts_and_the_default_does_not(self):
        m = prime("chars", L=3)
        m.reward("abc")
        self.assertEqual(m.pair.count.texts, 0)
        m.reward("abc", read=True)
        self.assertEqual(m.pair.count.texts, 1)

    def test_weights_are_marks_and_zero_is_skipped(self):
        m = prime("chars", L=3)
        r = m.reward(["ab", "cd", "ef"], weights=[1.0, 0.0, 0.5])
        self.assertEqual((r["texts"], r["skipped"]), (2, 1))
        ab = m.pair.address.of(m.codec.encode("ab"))
        ef = m.pair.address.of(m.codec.encode("ef"))
        self.assertAlmostEqual(m.pair.reward.reward(ab), 1.0)
        self.assertAlmostEqual(m.pair.reward.reward(ef), 0.5)
        with self.assertRaises(ValueError):
            m.reward(["ab"], weights=[1.0, 1.0])
        with self.assertRaises(ValueError):
            m.reward(["ab"], weights=[-1.0])

    def test_two_nrl_and_feedback_dispatch(self):
        m = prime("chars", L=3)
        r = m.two_nrl(["bad text"], ["good text"])
        self.assertEqual(r["call"], "two_nrl")
        self.assertEqual(m.feedback(good=["g"])["call"], "reward")
        self.assertEqual(m.feedback(bad=["b"])["call"], "punish")
        self.assertEqual(m.feedback(good=["g"], bad=["b"], good_weights=[0.9], bad_weights=[1.0])["call"], "two_nrl")
        with self.assertRaises(ValueError):
            m.feedback()
        m.invert()
        self.assertEqual(m.history[-1]["call"], "invert")

    def test_the_mat_and_log_case(self):
        m = prime("chars", L=4, settings=Settings(smoothing=0.0))
        m.train(["a cat sat on the mat", "a cat sat on the log"])
        prefix = "a cat sat on the "
        m.reward("mat", strength=5.0, prefix=prefix)
        m.punish("mat", strength=1.0, prefix=prefix)
        self.assertEqual(m.predict(prefix, 3, traversal="reward").text, "mat")
        self.assertEqual(m.predict(prefix, 3, traversal="punishment").text, "log")
        m.reward(["mat"] * 50, strength=5.0, prefix=prefix)
        self.assertEqual(m.predict(prefix, 3, traversal="punishment").text, "log")
        # the prefix's own steps were context, not outcome: "the cat" earned nothing
        c = m.pair.address.of(m.codec.encode("a ca"))
        self.assertEqual(m.pair.reward.reward(c), 0.0)
        self.assertEqual(m.history[-1]["prefix"], prefix)

    def test_predict_modes_start_and_fragment(self):
        m = prime("chars", L=4)
        m.train(TEXTS)
        for mode in ("dijkstra", "greedy", "sample"):
            r = m.predict("the cat", 5, mode=mode)
            self.assertEqual(r.full_text[:7], "the cat")
            self.assertLessEqual(len(r.units), 5)
        a = m.predict("cat", 3, start=True)
        b = m.predict("cat", 3, start=False)
        self.assertEqual(a.full_text[:3], "cat")
        self.assertEqual(b.full_text[:3], "cat")
        with self.assertRaises(ValueError):
            m.predict("x", 3, mode="beam")
        with self.assertRaises(ValueError):
            m.predict("x", 3, mode="dijkstra", backoff="none")   # the cheapest path walks the model's own fold
        self.assertLessEqual(len(m.predict("x", 3, mode="greedy", backoff="none").units), 3)

    def test_generate(self):
        m = prime("chars", L=4)
        for _ in range(3):
            m.train(["the dog"])
        g = m.generate("", 30)
        self.assertTrue(g.reached_end)
        self.assertEqual(g.full_text, "the dog")
        s = m.generate("the", 30, mode="sample")
        self.assertLessEqual(len(s.units), 30)

    def test_next_units_and_distribution(self):
        m = prime("chars", L=4)
        m.train(TEXTS)
        top = m.next_units("the cat sat on the ", 3)
        self.assertEqual(len(top), 3)
        self.assertGreaterEqual(top[0][1], top[1][1])
        dist = m.distribution(m.codec.encode("the"))
        self.assertAlmostEqual(sum(p for _, p in dist), 1.0)

    def test_weights_change_the_fold_and_rungs_cannot_change(self):
        m = prime("chars", L=3)
        m.train(TEXTS)
        before = m.score(TEXTS[0]).bits
        m.weights(smoothing=0.0)
        self.assertNotEqual(m.score(TEXTS[0]).bits, before)
        self.assertEqual(m.settings.smoothing, 0.0)
        with self.assertRaises(ValueError):
            m.weights(rungs="final")
        with self.assertRaises(ValueError):
            prime("chars", L=3, kind="sine")


@_ut.skipUnless(HAVE_PHONETOK, "the phonetic tokenizer is not importable")
class TestPhonesAreTheMainTokenizer(unittest.TestCase):
    def test_the_default_model_reads_and_writes_phones(self):
        m = prime()                                   # phones at L=3
        self.assertEqual(m.codec.name, "phones")
        self.assertEqual(m.L, 3)
        m.train(["the cat sat on the mat"] * 3)
        r = m.predict("the cat", 4)
        self.assertTrue(r.full_text.startswith("DH AH0 # K AE1 T"))         # phones out
        self.assertTrue(r.spelled.startswith("the cat"))                    # and the English they spell
        self.assertEqual(m.codec.encode(r.full_text), m.codec.encode("the cat") + r.units)   # phones read back
        s = m.score("the cat sat")
        self.assertEqual(s.per_unit[0][0], "DH")
        p = m.predict("DH AH0 # K AE1 T", 2)          # a prefix given in phones
        self.assertEqual(p.units, m.predict("the cat", 2).units)
        # an outcome after a prefix is joined as words: the boundary is the outcome's first credited step
        m.reward("mat", strength=5.0, prefix="the cat sat on the")
        boundary = m.codec.encode("#")[0]
        node = m.pair.address.of(m.codec.encode("the")[-2:] + [boundary])
        self.assertGreater(m.pair.reward.reward(node), 0.0)
        self.assertEqual(m.codec.join("the", "mat"), "the mat")
        self.assertEqual(CharsCodec().join("the ", "mat"), "the mat")


class TestPersistence(unittest.TestCase):
    def test_save_and_load_are_identical(self):
        m = prime("chars", L=4, seed=3)
        m.train(TEXTS)
        m.reward(TEXTS[0], strength=2.0)
        m.punish(TEXTS[2])
        m.weights(smoothing=0.25)
        with tempfile.TemporaryDirectory() as d:
            for name in ("m.json", "m.json.gz"):
                path = os.path.join(d, name)
                m.save(path)
                n = load_model(path)
                self.assertEqual(n.predict("the cat", 4).text, m.predict("the cat", 4).text)
                self.assertEqual(n.score(TEXTS[3]).to_dict(), m.score(TEXTS[3]).to_dict())
                self.assertEqual(n.info()["nonzero_counts"], m.info()["nonzero_counts"])
                self.assertEqual(n.info()["nonzero_rewards"], m.info()["nonzero_rewards"])
                self.assertEqual(n.settings.smoothing, 0.25)
                self.assertEqual(n.history, m.history)
                self.assertEqual(n.predict("a", 5, mode="sample").units, m.predict("a", 5, mode="sample").units)
            with open(os.path.join(d, "m.json.gz"), "rb") as fh:
                doc = json.loads(gzip.decompress(fh.read()))
            self.assertTrue(all(v != 0 for v in doc["counts"]["values"]))
            self.assertTrue(all(p or q for p, q in zip(doc["rewards"]["plus"], doc["rewards"]["minus"])))
            self.assertEqual(doc["codec"]["name"], "chars")

    def test_the_file_is_the_data_not_the_tree(self):
        with tempfile.TemporaryDirectory() as d:
            sizes = {}
            for L in (2, 4):
                m = prime("chars", L=L)
                m.train(TEXTS)
                path = os.path.join(d, f"L{L}.json")
                m.save(path)
                sizes[L] = os.path.getsize(path)
            self.assertLess(sizes[4] / sizes[2], 10)      # N grows a thousandfold; the file does not
            small = prime("chars", L=4)
            small.train(TEXTS[:1])
            small.save(os.path.join(d, "small.json"))
            self.assertLess(os.path.getsize(os.path.join(d, "small.json")), sizes[4])

    def test_refusals(self):
        with self.assertRaises(ValueError):
            PairModel.from_dict({"format": "radixnet"})
        with self.assertRaises(ValueError):
            PairModel.from_dict({"format": "radixpair", "version": 99, "codec": {"name": "chars"}, "L": 2,
                                 "counts": {"ids": [], "values": []}})
        with self.assertRaises(ValueError):
            prime(CharsCodec(), L=3, top=5)


if __name__ == "__main__":
    unittest.main()
