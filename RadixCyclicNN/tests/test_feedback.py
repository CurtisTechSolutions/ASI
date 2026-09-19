"""Tests for the ``feedback`` job (thumbs up / thumbs down -> 2NRL) in the API and the CLI.

Both kinds rated: 2NRL (bad, invert, good).  Only thumbs up: reward (a
positive-phase pass).  Only thumbs down: punish (a negative-phase pass, then
the network is inverted).  Every model kind has to answer the same call, which
:class:`KindParityTests` pins down.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.api import feedback_action  # noqa: E402
from radixnet.model import model_classes  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import CORPUS, GARBAGE, start_server  # noqa: F401
except ImportError:
    from tests.test_api import CORPUS, GARBAGE, start_server  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = {"neg_epochs": 1, "pos_epochs": 1, "batch_size": 4}


def wait_job(client, timeout=90.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, job, _ = client.get("/api/job")
        assert status == 200, (status, job)
        if job and job.get("state") != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish in time")


class FeedbackActionTests(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(feedback_action(["a"], ["b"]), "2nrl")
        self.assertEqual(feedback_action(["a"], []), "reward")
        self.assertEqual(feedback_action([], ["b"]), "punish")
        self.assertIsNone(feedback_action([], []))


class ApiFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="radixnet-feedback-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.client, self.server, self.service = start_server(self.addCleanup, upload_dir=self.dir)

    def stats(self):
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(status, 200)
        return stats

    def test_both_kinds_run_two_nrl(self):
        body = {"good": CORPUS[:3], "bad": GARBAGE[:2], **SETTINGS}
        status, data, _ = self.client.post("/api/feedback", body)
        self.assertEqual(status, 202, data)
        self.assertEqual((data["action"], data["good"], data["bad"], data["job"]["type"]), ("2nrl", 3, 2, "feedback"))
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual({r.get("phase") for r in job["history"]}, {"negative", "positive"})
        stats = self.stats()
        self.assertEqual((stats["twonrl_runs"], stats["inverted"], stats["trained_texts"]), (1, True, 5))

    def test_thumbs_up_only_rewards(self):
        status, data, _ = self.client.post("/api/feedback", {"good_text": "\n".join(CORPUS[:2]), **SETTINGS})
        self.assertEqual(status, 202, data)
        self.assertEqual(data["action"], "reward")
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        self.assertTrue(all(r.get("phase") == "positive" for r in job["history"]))
        stats = self.stats()
        self.assertEqual((stats["inverted"], stats["trained_texts"], stats["twonrl_runs"]), (False, 2, 0))

    def test_thumbs_down_only_punishes_and_inverts(self):
        status, data, _ = self.client.post("/api/feedback", {"bad": GARBAGE[:2], **SETTINGS})
        self.assertEqual(status, 202, data)
        self.assertEqual(data["action"], "punish")
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        self.assertTrue(all(r.get("phase") == "negative" for r in job["history"]))
        self.assertTrue(self.stats()["inverted"])
        # punishing again inverts back: the network is never left one-sided by accident
        status, data, _ = self.client.post("/api/feedback", {"bad": GARBAGE[2:4], **SETTINGS})
        self.assertEqual(status, 202, data)
        wait_job(self.client)
        self.assertFalse(self.stats()["inverted"])

    def test_uploads_and_errors(self):
        self.client.post("/api/uploads", {"name": "ups.txt", "content": "\n".join(CORPUS[:2]) + "\n"})
        status, data, _ = self.client.post("/api/feedback", {"good_files": ["ups.txt"], "bad": ["asdf qwer"], **SETTINGS})
        self.assertEqual(status, 202, data)
        self.assertEqual((data["action"], data["good"], data["bad"]), ("2nrl", 2, 1))
        wait_job(self.client)
        status, data, _ = self.client.post("/api/feedback", {})
        self.assertEqual(status, 400, data)
        self.assertIn("thumbs", data["error"])
        status, data, _ = self.client.post("/api/feedback", {"good": [], "bad": []})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/feedback", {"good": ["x y z"], "neg_lr": -1})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/feedback", {"good_files": ["missing.txt"]})
        self.assertEqual(status, 404, data)

    def test_ratings_weight_what_is_learned(self):
        status, data, _ = self.client.post("/api/feedback", {
            "good": [CORPUS[0], CORPUS[1]], "good_ratings": [10, 5],
            "bad": ["asdf qwer"], "bad_ratings": [8], **SETTINGS,
        })
        self.assertEqual(status, 202, data)
        self.assertEqual(data["good_weights"], [1.0, 0.5])
        self.assertEqual(data["bad_weights"], [0.8])
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        weights = [r.get("weight") for r in job["history"] if r.get("phase") == "positive"]
        self.assertEqual(weights, [1.0, 0.5])  # the better the mark, the larger the share of the learning rate
        # 0..1 shares say the same thing
        status, data, _ = self.client.post("/api/feedback", {"good": [CORPUS[0]], "good_weights": [0.25], **SETTINGS})
        self.assertEqual(status, 202, data)
        self.assertEqual(data["good_weights"], [0.25])
        wait_job(self.client)

    def test_rating_errors(self):
        for body, expected in (
            ({"good": [CORPUS[0]], "good_ratings": [1, 2]}, "1 text"),
            ({"good": [CORPUS[0]], "good_ratings": ["nine"]}, "list of numbers"),
            ({"good": [CORPUS[0]], "good_ratings": [-1]}, ">= 0"),
            ({"good": [CORPUS[0]], "good_ratings": [10], "good_weights": [1]}, "not both"),
        ):
            with self.subTest(**body):
                status, data, _ = self.client.post("/api/feedback", {**body, **SETTINGS})
                self.assertEqual(status, 400, data)
                self.assertIn(expected, data["error"])


class KindParityTests(unittest.TestCase):
    """Every model kind answers the same feedback call the API, the tutor and the chat loop make.

    The resonant and negative models used to let ``weights`` / ``progress`` /
    ``stop_event`` fall through into their :class:`TrainConfig` overrides, so
    rating the texts - which is all the ratings card ever does - killed the job
    with ``unknown train option(s): stop_event, weights``.
    """

    TEXTS = ["the sun rises in the east", "the sun sets in the west", "the moon rises at night"]
    GARBAGE = ["qx zz vv qq", "zzzz xxxx vvvv"]

    def fresh(self, cls):
        model = cls(seed=0)
        model.train(self.TEXTS, epochs=1)
        return model

    def test_rated_reward_and_punish(self):
        """What ``ApiService.start_feedback`` passes, for every kind: weights, progress, stop_event and all."""
        for kind, cls in model_classes().items():
            for action, texts in (("reward", self.TEXTS), ("punish", self.GARBAGE)):
                with self.subTest(kind=kind, action=action):
                    seen = []
                    records = getattr(self.fresh(cls), action)(
                        texts[:2], epochs=1, lr=0.1, strength=None, weights=[1.0, 0.5], batch_size=4,
                        progress=seen.append, stop_event=threading.Event(),
                    )
                    # heaviest first, one pass each, and the job history sees every one of them
                    self.assertEqual([r["weight"] for r in records], [1.0, 0.5])
                    self.assertEqual(len(seen), len(records))

    def test_rated_two_nrl(self):
        for kind, cls in model_classes().items():
            with self.subTest(kind=kind):
                seen = []
                out = self.fresh(cls).two_nrl(
                    self.GARBAGE[:2], self.TEXTS[:2], neg_epochs=1, pos_epochs=1, neg_lr=0.5, pos_lr=0.1,
                    strength=None, bad_weights=[1.0, 0.5], good_weights=[1.0, 0.5], batch_size=4,
                    progress=seen.append, stop_event=threading.Event(),
                )
                self.assertEqual([r["weight"] for r in out["negative"]], [1.0, 0.5])
                self.assertEqual([r["weight"] for r in out["positive"]], [1.0, 0.5])
                self.assertEqual(len(seen), len(out["negative"]) + len(out["positive"]))

    def test_unrated_records_carry_no_weight(self):
        """No weights given is a thumbs up, not a rating - nothing in the records says otherwise."""
        for kind, cls in model_classes().items():
            with self.subTest(kind=kind):
                out = self.fresh(cls).two_nrl(self.GARBAGE[:2], self.TEXTS[:2], neg_epochs=1, pos_epochs=1)
                self.assertTrue(all("weight" not in r for r in out["negative"] + out["positive"]))

    def test_a_set_stop_event_stops_every_kind(self):
        """Stopping the job mid-rating stops the passes; the inverting kinds still never stay in the garbage."""
        for kind, cls in model_classes().items():
            with self.subTest(kind=kind):
                model = self.fresh(cls)
                stop = threading.Event()
                stop.set()
                self.assertEqual(model.reward(self.TEXTS[:2], weights=[1.0, 0.5], stop_event=stop), [])
                self.assertEqual(model.punish(self.GARBAGE[:2], weights=[1.0, 0.5], stop_event=stop), [])
                out = model.two_nrl(
                    self.GARBAGE[:2], self.TEXTS[:2], neg_epochs=1, pos_epochs=1,
                    bad_weights=[1.0, 0.5], good_weights=[1.0, 0.5], stop_event=stop,
                )
                self.assertEqual((out["negative"], out["positive"]), ([], []))

    def test_the_job_runs_for_every_kind(self):
        """End to end: the ratings card posts marks, and the job finishes instead of dying in the worker thread."""
        for kind in model_classes():
            with self.subTest(kind=kind):
                client, _server, _service = start_server(self.addCleanup, kind=kind)
                client.post("/api/train", {"texts": self.TEXTS, "epochs": 1, "batch_size": 4})
                wait_job(client)
                status, data, _ = client.post("/api/feedback", {
                    "good": self.TEXTS[:2], "good_ratings": [10, 5],
                    "bad": self.GARBAGE[:1], "bad_ratings": [8], "neg_epochs": 1, "pos_epochs": 1, "batch_size": 4,
                })
                self.assertEqual(status, 202, data)
                job = wait_job(client)
                self.assertEqual(job["state"], "done", job)
                self.assertTrue(job["history"], job)

    def test_weights_have_to_match_the_texts(self):
        for kind, cls in model_classes().items():
            with self.subTest(kind=kind):
                model = self.fresh(cls)
                with self.assertRaises(ValueError):
                    model.reward(self.TEXTS[:2], weights=[1.0])
                with self.assertRaises(ValueError):
                    model.punish(self.GARBAGE[:2], weights=[1.0, -1.0])
                with self.assertRaises(ValueError):
                    model.two_nrl(self.GARBAGE[:2], self.TEXTS[:2], bad_weights=[1.0])


class CliFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="radixnet-feedback-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.model = os.path.join(self.dir, "model.json")

    def run_cli(self, *args, expect=0):
        cmd = [sys.executable, "-m", "radixnet", "--model", self.model, "--backend", "python", "--json", *args]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PYTHONWARNINGS="ignore"), timeout=300)
        self.assertEqual(proc.returncode, expect, proc.stderr[-2000:])
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def test_feedback_command(self):
        doc = self.run_cli("feedback", "--good-text", CORPUS[0], "--good-text", CORPUS[1], "--bad-text", GARBAGE[0],
                           "--neg-epochs", "1", "--pos-epochs", "1")
        self.assertEqual((doc["action"], doc["good_texts"], doc["bad_texts"], doc["inverted"]), ("2nrl", 2, 1, True))
        self.assertEqual(doc["stats"]["twonrl_runs"], 1)
        self.assertTrue(os.path.isfile(self.model))
        doc = self.run_cli("feedback", "--bad-text", GARBAGE[1], "--neg-epochs", "1")
        self.assertEqual((doc["action"], doc["inverted"]), ("punish", False))  # inverted twice = back to normal
        good = os.path.join(self.dir, "good.txt")
        with open(good, "w", encoding="utf-8") as fh:
            fh.write("\n".join(CORPUS[:2]) + "\n")
        doc = self.run_cli("feedback", "--good", good, "--pos-epochs", "1")
        self.assertEqual((doc["action"], doc["positive"][-1]["phase"]), ("reward", "positive"))
        self.run_cli("feedback", expect=1)

    def test_marks_scale_the_learning(self):
        doc = self.run_cli("feedback", "--good-text", CORPUS[0], "--good-text", CORPUS[1], "--good-ratings", "10,5",
                           "--pos-epochs", "1", "--pos-lr", "0.2")
        self.assertEqual(doc["good_weights"], [1.0, 0.5])
        self.assertEqual([r["weight"] for r in doc["positive"]], [1.0, 0.5])
        self.assertAlmostEqual(doc["positive"][1]["lr"], 0.1)
        doc = self.run_cli("feedback", "--bad-text", GARBAGE[0], "--bad-ratings", "4", "--neg-epochs", "1", "--neg-lr", "0.5")
        self.assertEqual(doc["bad_weights"], [0.4])
        self.assertAlmostEqual(doc["negative"][0]["lr"], 0.2)
        self.run_cli("feedback", "--good-text", CORPUS[0], "--good-ratings", "10,5", expect=1)
        self.run_cli("feedback", "--good-text", CORPUS[0], "--good-ratings", "11", expect=1)


if __name__ == "__main__":
    unittest.main()
