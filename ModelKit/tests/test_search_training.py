"""Search and training settings through the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_search_training.py).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from radixnet.model import load_model  # noqa: E402

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the model's checkout: its sample data, and the directory `python -m radixnet` runs in
ROOT = os.path.join(os.path.dirname(KIT), "RadixCyclicNN")


def _lines(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


CORPUS = _lines("sample_corpus.txt")


class TestApiAndCli(unittest.TestCase):
    """The settings over HTTP and on the command line: accepted in range, a 400 / usage error out of it."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, kind="count", seed=1)
        status, _, _ = cls.client.post("/api/train", {"texts": CORPUS[:30], "epochs": 1, "replay_size": 8})
        assert status == 202
        cls.wait()

    @classmethod
    def wait(cls):
        from tests.test_api import wait_for_job

        return wait_for_job(cls.client)

    def test_search_settings_out_of_range_are_a_400(self):
        for route, body in (
            ("/api/predict", {"prefix": "the", "top_p": 0}),
            ("/api/predict", {"prefix": "the", "min_p": 1}),
            ("/api/predict", {"prefix": "the", "top_k": -1}),
            ("/api/generate", {"mode": "sample", "top_p": 1.5}),
            ("/api/generate", {"mode": "beam", "diversity": -0.5}),
        ):
            with self.subTest(route=route, body=body):
                status, data, _ = self.client.post(route, body)
                self.assertEqual(status, 400, data)

    def test_search_settings_in_range_are_used(self):
        status, plain, _ = self.client.post("/api/generate", {"mode": "beam", "count": 5, "max_length": 40, "guard": False})
        self.assertEqual(status, 200, plain)
        status, spread, _ = self.client.post(
            "/api/generate", {"mode": "beam", "count": 5, "max_length": 40, "diversity": 5.0, "guard": False}
        )
        self.assertEqual(status, 200, spread)
        self.assertEqual(plain["samples"][0]["text"], spread["samples"][0]["text"])
        status, data, _ = self.client.post(
            "/api/generate", {"mode": "sample", "count": 3, "seed": 4, "top_k": 2, "top_p": 0.8, "min_p": 0.1,
                              "guard": False}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual(len(data["samples"]), 3)

    def test_training_settings_out_of_range_are_a_400(self):
        for body in ({"order": "random"}, {"curriculum": 0}, {"curriculum": 2}, {"replay": -1},
                     {"replay_size": -2}, {"patience": -1}, {"min_delta": -0.1}, {"reverse": "yes"}):
            with self.subTest(body=body):
                status, data, _ = self.client.post("/api/train", {"texts": CORPUS[:3], "epochs": 1, **body})
                self.assertEqual(status, 400, data)

    def test_a_planned_run_over_http(self):
        # 20 texts, a curriculum of 0.9 over 6 epochs: 18, 19, 19, 20, 20, 20 texts
        status, _, _ = self.client.post("/api/train", {
            "texts": CORPUS[30:50], "epochs": 6, "order": "Shortest-First", "curriculum": 0.9, "replay": 0.5,
            "patience": 1, "min_delta": 100,
        })
        self.assertEqual(status, 202)
        job = self.wait()
        self.assertEqual(job["state"], "done", job)
        history = job["history"]
        # the epochs inside the curriculum never count toward the stop; the first full one sets the best
        # and the next one, no better by 100, stops the run
        self.assertEqual(len(history), 5)
        self.assertEqual([r.get("early_stop") for r in history], [None] * 4 + [True])
        self.assertEqual(self.service.model.replay.seen, 50)

    def test_the_cli_refuses_settings_out_of_range(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.json")
            data = os.path.join(ROOT, "data", "sample_corpus.txt")
            for args, flag in ((["generate", "--top-p", "0"], "--top-p"), (["generate", "--min-p", "1"], "--min-p"),
                               (["predict", "--diversity", "-1"], "--diversity"),
                               (["train", "--data", data, "--order", "random"], "--order"),
                               (["train", "--data", data, "--curriculum", "0"], "--curriculum"),
                               (["train", "--data", data, "--replay-size", "-1"], "--replay-size")):
                with self.subTest(args=args):
                    proc = subprocess.run([sys.executable, "-m", "radixnet", "--model", model, *args],
                                          cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(f"argument {flag}", proc.stderr)
                    self.assertFalse(os.path.exists(model))  # refused before anything ran


class TestReadingBackwardsOverHttpAndCli(unittest.TestCase):
    """A file trained on backwards through ``POST /api/train`` (an upload) and ``train --reverse``."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, kind="count", seed=1, upload_dir=os.path.join(cls.tmp.name, "uploads")
        )

    def test_an_upload_read_backwards(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.post("/api/train", {"texts": ["lazy dog sleeps"], "reverse": "yes"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/uploads", {"name": "fox.txt", "content": "quick brown fox\n"})
        self.assertEqual(status, 201, data)
        status, data, _ = self.client.post(
            "/api/train", {"texts": ["lazy dog sleeps"], "files": ["fox.txt"], "reverse": True, "epochs": 2}
        )
        self.assertEqual(status, 202, data)
        self.assertEqual(wait_for_job(self.client)["state"], "done")
        for end, whole in (("fox", "quick brown fox"), ("sleeps", "lazy dog sleeps")):
            status, data, _ = self.client.post(
                "/api/predict", {"prefix": end[::-1], "length": 1, "to_end": True, "guard": False}
            )
            self.assertEqual(status, 200, data)
            self.assertEqual(data["full_text"][::-1], whole)

    def test_the_cli_reads_a_file_backwards(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "fox.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("quick brown fox\nlazy dog sleeps\n")
            model = os.path.join(tmp, "m.json")
            proc = subprocess.run(
                [sys.executable, "-m", "radixnet", "--json", "--kind", "count", "--model", model, "train",
                 "--data", data, "--reverse", "--epochs", "2"],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIs(json.loads(proc.stdout)["config"]["reverse"], True)
            found = load_model(model).predict("xof", length=1, to_end=True)
            self.assertEqual(found.full_text[::-1], "quick brown fox")


if __name__ == "__main__":
    unittest.main()
