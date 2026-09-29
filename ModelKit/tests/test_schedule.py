"""Learning-rate schedules through the HTTP API and the command line (the model side is RadixCyclicNN's tests/test_schedule.py).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from radixnet.model import TrainConfig  # noqa: E402
from radixnet.schedule import PRESETS, describe  # noqa: E402

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the model's checkout: its sample data, and the directory `python -m radixnet` runs in
ROOT = os.path.join(os.path.dirname(KIT), "RadixCyclicNN")


def _lines(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


CORPUS = _lines("sample_corpus.txt")


def close(a, b):
    return all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(a, b)) and len(a) == len(b)


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup)

    def test_describe_endpoint(self):
        status, data, _ = self.client.get("/api/schedule")
        self.assertEqual(status, 200)
        self.assertEqual(data, describe())

    def test_preview_endpoint(self):
        status, data, _ = self.client.post(
            "/api/schedule/preview",
            {"lr_schedule": "linear(lr0, 5 * lr0)", "act_lr_schedule": "lr / 10", "epochs": 5, "lr": 0.1, "act_lr": 0.005},
        )
        self.assertEqual(status, 200, data)
        self.assertEqual(set(data), {"lr_schedule", "act_lr_schedule", "epochs", "lr", "act_lr", "reverse_schedule", "points"})
        self.assertEqual((data["lr_schedule"], data["act_lr_schedule"], data["epochs"]), ("linear(lr0, 5 * lr0)", "lr / 10", 5))
        self.assertEqual([p["epoch"] for p in data["points"]], [1, 2, 3, 4, 5])
        self.assertTrue(close([p["lr"] for p in data["points"]], [0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertTrue(close([p["act_lr"] for p in data["points"]], [0.01, 0.02, 0.03, 0.04, 0.05]))

    def test_preview_defaults_and_blank_schedules(self):
        status, data, _ = self.client.post("/api/schedule/preview", {})
        self.assertEqual(status, 200)
        self.assertEqual((data["lr_schedule"], data["act_lr_schedule"], data["epochs"]), (None, None, TrainConfig.epochs))
        self.assertEqual(len(data["points"]), TrainConfig.epochs)
        self.assertEqual(data["points"][0], {"epoch": 1, "lr": TrainConfig.lr, "act_lr": TrainConfig.act_lr})
        status, data, _ = self.client.post("/api/schedule/preview", {"lr_schedule": "   ", "act_lr_schedule": None, "epochs": 0})
        self.assertEqual(status, 200)
        self.assertEqual((data["lr_schedule"], data["points"]), (None, []))

    def test_preview_errors(self):
        for body, fragment in (
            ({"lr_schedule": "lr0 - 1"}, "must be finite and >= 0"),
            ({"act_lr_schedule": "__import__('os')"}, "only these functions"),
            ({"lr_schedule": "lr0 *"}, "invalid schedule expression"),
            ({"lr_schedule": 5}, "'lr_schedule' must be a string"),
            ({"epochs": -1}, "'epochs' must be >= 0"),
            ({"lr": -0.1}, "'lr' must be >= 0"),
        ):
            with self.subTest(body=body):
                status, data, _ = self.client.post("/api/schedule/preview", body)
                self.assertEqual(status, 400, data)
                self.assertIn(fragment, data["error"])

    def test_train_with_schedules(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.post(
            "/api/train",
            {"texts": CORPUS[:12], "epochs": 4, "lr": 0.1, "act_lr": 0.005, "batch_size": 8,
             "lr_schedule": "geometric(lr0, 8 * lr0)", "act_lr_schedule": "lr / 10"},
        )
        self.assertEqual(status, 202, data)
        done = wait_for_job(self.client)
        self.assertEqual(done["state"], "done", done)
        self.assertTrue(close([r["lr"] for r in done["history"]], [0.1, 0.2, 0.4, 0.8]))
        self.assertTrue(close([r["act_lr"] for r in done["history"]], [0.01, 0.02, 0.04, 0.08]))
        status, data, _ = self.client.post(
            "/api/train",
            {"texts": CORPUS[:12], "epochs": 4, "lr": 0.1, "act_lr": 0.005, "batch_size": 8,
             "lr_schedule": "geometric(lr0, 8 * lr0)", "act_lr_schedule": "lr / 10", "reverse_schedule": True},
        )
        self.assertEqual(status, 202, data)
        done = wait_for_job(self.client)
        self.assertTrue(close([r["lr"] for r in done["history"]], [0.8, 0.4, 0.2, 0.1]))
        self.assertTrue(close([r["act_lr"] for r in done["history"]], [0.08, 0.04, 0.02, 0.01]))

    def test_preview_reverse(self):
        status, data, _ = self.client.post(
            "/api/schedule/preview", {"lr_schedule": "linear(lr0, 3 * lr0)", "epochs": 3, "lr": 0.1, "reverse_schedule": True},
        )
        self.assertEqual(status, 200, data)
        self.assertTrue(data["reverse_schedule"])
        self.assertTrue(close([p["lr"] for p in data["points"]], [0.3, 0.2, 0.1]))
        status, data, _ = self.client.post("/api/schedule/preview", {"lr_schedule": "lr0", "reverse_schedule": "yes"})
        self.assertEqual(status, 400)

    def test_train_rejects_a_bad_schedule_without_starting_a_job(self):
        status, before, _ = self.client.get("/api/job")
        self.assertEqual(status, 200)
        self.assertTrue(before is None or before["state"] != "running")
        status, data, _ = self.client.post("/api/train", {"texts": CORPUS[:4], "epochs": 2, "lr_schedule": "lr0 - 1"})
        self.assertEqual(status, 400, data)
        self.assertIn("must be finite and >= 0", data["error"])
        status, data, _ = self.client.post("/api/train", {"texts": CORPUS[:4], "epochs": 2, "act_lr_schedule": 7})
        self.assertEqual(status, 400, data)
        self.assertIn("'act_lr_schedule' must be a string", data["error"])
        status, after, _ = self.client.get("/api/job")
        self.assertEqual(after, before)


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_cli import run_cli, run_json

        cls.run_cli = staticmethod(run_cli)
        cls.run_json = staticmethod(run_json)

    def test_schedule_lists_presets_without_expressions(self):
        doc = self.run_json("schedule")
        self.assertEqual(set(doc), {"presets", "variables", "functions", "helpers"})
        self.assertEqual(doc["presets"], [dict(p) for p in PRESETS])
        proc = self.run_cli("schedule", json_mode=False)
        self.assertIn("linear ramp x4", proc.stdout)
        self.assertIn("warmup(a, b, n)", proc.stdout)

    def test_schedule_previews_a_table_and_graph(self):
        doc = self.run_json("schedule", "--epochs", 4, "--lr", 0.1, "--act-lr", 0.005,
                            "--lr-schedule", "linear(lr0, 4 * lr0)", "--act-lr-schedule", "lr / 10")
        self.assertEqual((doc["lr_schedule"], doc["act_lr_schedule"], doc["epochs"], doc["lr"], doc["act_lr"]),
                         ("linear(lr0, 4 * lr0)", "lr / 10", 4, 0.1, 0.005))
        self.assertTrue(close([p["lr"] for p in doc["points"]], [0.1, 0.2, 0.3, 0.4]))
        self.assertTrue(close([p["act_lr"] for p in doc["points"]], [0.01, 0.02, 0.03, 0.04]))
        doc = self.run_json("schedule", "--epochs", 4, "--lr", 0.1, "--lr-schedule", "linear(lr0, 4 * lr0)", "--reverse-schedule")
        self.assertTrue(doc["reverse_schedule"])
        self.assertTrue(close([p["lr"] for p in doc["points"]], [0.4, 0.3, 0.2, 0.1]))
        proc = self.run_cli("schedule", "--epochs", 4, "--lr-schedule", "linear(lr0, 4 * lr0)", json_mode=False)
        self.assertIn("lr graph", proc.stdout)
        bars = [line.split()[2] for line in proc.stdout.splitlines() if line.strip()[:1].isdigit()]
        self.assertEqual(len(bars), 4)
        self.assertEqual(bars[-1], "#" * 30)
        self.assertTrue(all(set(b) == {"#"} for b in bars))
        self.assertEqual([len(b) for b in bars], sorted(len(b) for b in bars))

    def test_schedule_errors(self):
        proc = self.run_cli("schedule", "--lr-schedule", "lr0 - 1", expect=1)
        self.assertIn("must be finite and >= 0", proc.stderr)
        proc = self.run_cli("schedule", "--act-lr-schedule", "open('x')", expect=1)
        self.assertIn("only these functions may be called", proc.stderr)
        proc = self.run_cli("schedule", "--epochs", -1, expect=1)  # usage errors exit 1 in this CLI
        self.assertIn("usage", proc.stderr)
        self.assertIn("must be >= 0", proc.stderr)

    def test_train_with_schedules(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="radixnet-schedule-") as tmp:
            model = os.path.join(tmp, "model.json")
            doc = self.run_json(
                "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 3, "--batch-size", 8,
                "--lr", 0.1, "--act-lr", 0.005, "--lr-schedule", "lr0 * 2 ** i", "--act-lr-schedule", "lr / 10", model=model,
            )
            self.assertEqual(doc["config"]["lr_schedule"], "lr0 * 2 ** i")
            self.assertEqual(doc["config"]["act_lr_schedule"], "lr / 10")
            self.assertTrue(close([r["lr"] for r in doc["records"]], [0.1, 0.2, 0.4]))
            self.assertTrue(close([r["act_lr"] for r in doc["records"]], [0.01, 0.02, 0.04]))
            self.assertTrue(os.path.exists(model))
            doc = self.run_json(
                "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 3, "--batch-size", 8,
                "--lr", 0.1, "--lr-schedule", "lr0 * 2 ** i", "--reverse-schedule", model=os.path.join(tmp, "rev.json"),
            )
            self.assertTrue(doc["config"]["reverse_schedule"])
            self.assertTrue(close([r["lr"] for r in doc["records"]], [0.4, 0.2, 0.1]))
            proc = self.run_cli(
                "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 2, "--batch-size", 8,
                "--lr-schedule", "linear(lr0, 3 * lr0)", model=model, json_mode=False,
            )
            self.assertIn("lr_schedule=linear(lr0, 3 * lr0)", proc.stdout)
            self.assertIn("act_lr", proc.stdout)  # the epoch table has lr / act_lr columns
            proc = self.run_cli("train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 2,
                                "--lr-schedule", "lr0 - 1", model=os.path.join(tmp, "other.json"), expect=1)
            self.assertIn("must be finite and >= 0", proc.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "other.json")))


if __name__ == "__main__":
    unittest.main()
