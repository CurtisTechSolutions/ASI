"""A subprocess smoke test of every command, with --json."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args, expect=0):
    proc = subprocess.run([sys.executable, "-m", "radixpair", "--json", *args], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != expect:
        raise AssertionError(f"exit {proc.returncode} for {args}: {proc.stderr}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {}


class TestCli(unittest.TestCase):
    def test_every_command(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.json")
            data = os.path.join(d, "corpus.txt")
            with open(data, "w") as fh:
                fh.write("the cat sat on the mat\nthe cat sat on the log\n\nthe dog ate the bone\na cat and a dog\n")
            out = run("prime", "--model", model, "--codec", "chars", "--L", "3")
            self.assertEqual(out["N"], 37060)
            out = run("train", "--model", model, "--data", data, "--checkpoint-dir", os.path.join(d, "ckpt"))
            self.assertEqual(out["texts"], 4)
            self.assertEqual(len(run("checkpoints", "--checkpoint-dir", os.path.join(d, "ckpt"))["checkpoints"]), 1)
            out = run("reward", "--model", model, "--text", "the cat sat on the mat", "--strength", "5", "--ratings", "9")
            self.assertEqual(out["texts"], 1)
            run("punish", "--model", model, "--text", "the cat sat on the mat")
            good, bad = os.path.join(d, "good.txt"), os.path.join(d, "bad.txt")
            with open(good, "w") as fh:
                fh.write("the dog ate the bone\n")
            with open(bad, "w") as fh:
                fh.write("the dog ate the mat\n")
            self.assertEqual(run("2nrl", "--model", model, "--bad", bad, "--good", good)["call"], "two_nrl")
            self.assertEqual(run("feedback", "--model", model, "--good", "a cat", "--good-ratings", "8")["call"], "reward")
            out = run("predict", "--model", model, "--prefix", "the cat", "--length", "4", "--hops")
            self.assertEqual(out["full_text"][:7], "the cat")
            out = run("predict", "--model", model, "--prefix", "the cat", "--length", "4", "--mode", "greedy",
                      "--traversal", "punishment", "--backoff", "deepest")
            self.assertEqual(out["traversal"], "punishment")
            out = run("generate", "--model", model, "--prefix", "the", "--length", "10", "--mode", "sample")
            self.assertLessEqual(len(out["units"]), 10)
            out = run("score", "--model", model, "--text", "the cat sat on the mat", "--per-unit")
            self.assertEqual(len(out["scores"]), 1)
            out = run("weights", "--model", model, "--smoothing", "0", "--backoff", "deepest")
            self.assertEqual(out["backoff"], "deepest")
            out = run("info", "--model", model)
            self.assertEqual(out["L"], 3)
            self.assertEqual(out["settings"]["smoothing"], 0.0)
            run("invert", "--model", model)
            out = run("check", "--R", "3", "--L", "3")
            self.assertTrue(out["checks"][0]["ok"])
            out = run("bench", "throughput", "--codec", "chars", "--L", "3", "--units", "3000")
            self.assertGreater(out["count"]["per_second"], 0)
            out = run("bench", "feedback", "--codec", "chars", "--L", "4")
            self.assertTrue(out["unbuyable"])
            out = run("bench", "compare", "--codec", "chars", "--L", "3", "--data", data, "--holdout", "0.25")
            self.assertIn("all", out["bits"])
            out = run("bench", "rungs", "--codec", "chars", "--L", "3", "--data", data, "--samples", "3")
            self.assertGreaterEqual(out["samples"], 1)
            run("predict", "--model", os.path.join(d, "none.json"), "--prefix", "x", expect=2)
            run("train", "--model", model, expect=2)


if __name__ == "__main__":
    unittest.main()
