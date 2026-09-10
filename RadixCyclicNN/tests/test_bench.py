"""Smoke tests for radixnet.bench (synthetic corpus, benchmark dict, entry point)."""

import contextlib
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import bench  # noqa: E402

REQUIRED_KEYS = (
    "backend", "device", "chars", "texts", "epochs",
    "train_seconds", "transitions", "transitions_per_sec", "chars_per_sec",
    "predict_count", "predict_seconds", "predictions_per_sec", "dijkstra_expansions_per_sec",
    "nodes", "edges", "compression_ratio", "sample_prediction",
)

RATE_KEYS = ("transitions_per_sec", "chars_per_sec", "predictions_per_sec", "dijkstra_expansions_per_sec")


class TestSyntheticCorpus(unittest.TestCase):
    def test_deterministic_and_sized(self):
        a = bench.synthetic_corpus(3000, seed=7)
        b = bench.synthetic_corpus(3000, seed=7)
        self.assertEqual(a, b)
        self.assertNotEqual(a, bench.synthetic_corpus(3000, seed=8))
        self.assertGreaterEqual(sum(len(t) for t in a), 3000)
        self.assertLess(sum(len(t) for t in a[:-1]), 3000)  # stops as soon as the budget is met
        self.assertTrue(all(len(t) >= 3 for t in a))

    def test_edge_cases(self):
        self.assertEqual(bench.synthetic_corpus(0), [])
        single = bench.synthetic_corpus(50, lines=["abc def"])
        self.assertGreaterEqual(sum(len(t) for t in single), 50)
        for text in single:  # verbatim, spliced or mutated: only base vocabulary, never too short
            self.assertGreaterEqual(len(text), 3)
            self.assertTrue(set(text.split()) <= {"abc", "def"}, text)
        with self.assertRaises(ValueError):
            bench.synthetic_corpus(-1)
        with self.assertRaises(ValueError):
            bench.synthetic_corpus(10, lines=["ab"])

    def test_prefixes_come_from_the_texts(self):
        texts = bench.synthetic_corpus(1000, seed=1)
        prefixes = bench.prediction_prefixes(texts, 50, seed=1)
        self.assertEqual(len(prefixes), 50)
        self.assertEqual(prefixes, bench.prediction_prefixes(texts, 50, seed=1))
        for prefix in prefixes:
            self.assertGreaterEqual(len(prefix), 3)
            self.assertTrue(any(prefix in t for t in texts))
        self.assertEqual(bench.prediction_prefixes(texts, 0), [])
        with self.assertRaises(ValueError):
            bench.prediction_prefixes([], 1)

    def test_base_lines_fallback(self):
        self.assertTrue(bench.load_base_lines())
        fallback = bench.load_base_lines(os.path.join(os.path.dirname(__file__), "no-such-corpus.txt"))
        self.assertTrue(fallback)
        self.assertTrue(all(len(line) >= 3 for line in fallback))


class TestRunBenchmark(unittest.TestCase):
    def test_smoke(self):
        result = bench.run_benchmark(chars=2000, epochs=1, backend="python", seed=0, quiet=True)
        for key in REQUIRED_KEYS:
            self.assertIn(key, result)
        self.assertEqual(result["backend"], "python")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["epochs"], 1)
        self.assertGreaterEqual(result["chars"], 2000)
        self.assertGreater(result["texts"], 0)
        self.assertGreater(result["transitions"], 0)
        self.assertGreater(result["train_seconds"], 0.0)
        self.assertGreater(result["predict_count"], 0)
        self.assertGreater(result["predict_seconds"], 0.0)
        for key in RATE_KEYS:
            self.assertGreater(result[key], 0.0, key)
        self.assertGreaterEqual(result["nodes"], 2)
        self.assertGreater(result["edges"], 0)
        self.assertGreater(result["compression_ratio"], 0.0)
        sample = result["sample_prediction"]
        self.assertIsInstance(sample["prefix"], str)
        self.assertIsInstance(sample["continuation"], str)
        self.assertGreaterEqual(sample["cost"], 0.0)
        json.dumps(result)  # JSON-serialisable for --json and the API

    def test_validation(self):
        with self.assertRaises(ValueError):
            bench.run_benchmark(chars=0, epochs=1)
        with self.assertRaises(ValueError):
            bench.run_benchmark(chars=100, epochs=0)
        with self.assertRaises(ValueError):
            bench.run_benchmark(chars=100, epochs=1, predictions=0)


class TestMain(unittest.TestCase):
    def test_json_output(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bench.main(["--chars", "1000", "--epochs", "1", "--backend", "python", "--predictions", "20", "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out.getvalue())
        for key in REQUIRED_KEYS:
            self.assertIn(key, doc)
        self.assertEqual(doc["predict_count"], 20)

    def test_table_output(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bench.main(["--chars", "1000", "--epochs", "1", "--backend", "python", "--predictions", "20"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        for key in REQUIRED_KEYS:
            self.assertIn(key, text)
        self.assertIn("sample_prediction.prefix", text)

    def test_format_report_flattens_nested_dicts(self):
        report = bench.format_report({"a": 1234, "b": 0.5, "c": {"d": "x y", "e": True}})
        self.assertEqual(report.splitlines(), ["a    1,234", "b    0.5000", "c.d  'x y'", "c.e  yes"])


if __name__ == "__main__":
    unittest.main()
