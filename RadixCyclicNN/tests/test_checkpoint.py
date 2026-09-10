"""Tests for radixnet.checkpoint (CheckpointManager)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.checkpoint import CheckpointManager  # noqa: E402
from radixnet.model import RadixNet  # noqa: E402

TEXTS = ["hello world", "hello there", "help me out", "the cat sat on the mat"]
FAST = {"lr": 1.0, "batch_size": 1}


def make_model(seed=0):
    model = RadixNet(seed=seed, backend="python")
    model.train(TEXTS, epochs=1, **FAST)
    return model


def fingerprint(model):
    return [model.predict(p, length=8).text for p in ("hel", "the cat", "he", "")]


class TestSaveAndList(unittest.TestCase):
    def test_save_record_and_files(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "ckpts", "nested")
            cm = CheckpointManager(directory, keep=3)
            self.assertTrue(os.path.isdir(directory))
            self.assertEqual(cm.list(), [])
            self.assertIsNone(cm.latest())
            self.assertIsNone(cm.load_latest(backend="python"))
            rec = cm.save(model, 1, metrics={"loss": 0.5})
            self.assertEqual(set(rec), {"name", "path", "step", "tag", "metrics", "saved_at", "bytes"})
            self.assertEqual(rec["name"], "ckpt-epoch-000001.json.gz")
            self.assertEqual(rec["path"], os.path.join(directory, rec["name"]))
            self.assertEqual((rec["step"], rec["tag"], rec["metrics"]), (1, "epoch", {"loss": 0.5}))
            self.assertTrue(os.path.isfile(rec["path"]))
            self.assertEqual(rec["bytes"], os.path.getsize(rec["path"]))
            with open(os.path.join(directory, "latest.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), rec)
            self.assertEqual(cm.latest(), rec)
            self.assertEqual(cm.list(), [rec])
            json.dumps(rec)
            # uncompressed variant and custom tags
            plain = CheckpointManager(os.path.join(tmp, "plain"), compress=False)
            rec2 = plain.save(model, 7, tag="gen")
            self.assertEqual(rec2["name"], "ckpt-gen-000007.json")
            with open(rec2["path"], "rb") as fh:
                self.assertEqual(fh.read(1), b"{")
            with self.assertRaises(ValueError):
                plain.save(model, 1, tag="../evil")
            with self.assertRaises(ValueError):
                plain.save(model, -1)
            with self.assertRaises(ValueError):
                CheckpointManager(os.path.join(tmp, "x"), keep=0)

    def test_list_sorted_and_reconciled(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            cm = CheckpointManager(tmp, keep=10)
            for step in (3, 1, 2):
                cm.save(model, step)
            cm.save(model, 2, tag="gen")
            self.assertEqual([(r["step"], r["tag"]) for r in cm.list()], [(1, "epoch"), (2, "epoch"), (2, "gen"), (3, "epoch")])
            # same (tag, step) again overwrites instead of duplicating
            cm.save(model, 2)
            self.assertEqual(len(cm.list()), 4)
            # files removed or added by hand are picked up
            os.unlink(os.path.join(tmp, "ckpt-epoch-000003.json.gz"))
            model.save(os.path.join(tmp, "ckpt-manual-000009.json.gz"))
            names = [r["name"] for r in cm.list()]
            self.assertNotIn("ckpt-epoch-000003.json.gz", names)
            self.assertIn("ckpt-manual-000009.json.gz", names)
            manual = cm.list()[-1]
            self.assertEqual((manual["step"], manual["tag"], manual["metrics"]), (9, "manual", None))
            self.assertEqual(fingerprint(cm.load("ckpt-manual-000009", backend="python")), fingerprint(model))


class TestRotation(unittest.TestCase):
    def test_prunes_oldest_but_never_latest(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            cm = CheckpointManager(tmp, keep=2)
            for step in (1, 2, 3):
                cm.save(model, step)
            self.assertEqual([r["step"] for r in cm.list()], [2, 3])
            self.assertEqual(sorted(n for n in os.listdir(tmp) if n.startswith("ckpt-")),
                             ["ckpt-epoch-000002.json.gz", "ckpt-epoch-000003.json.gz"])
            # a lower step saved last is the latest: rotation removes the older-by-step 2 instead
            cm.save(model, 1, tag="gen")
            self.assertEqual([(r["step"], r["tag"]) for r in cm.list()], [(1, "gen"), (3, "epoch")])
            self.assertEqual(cm.latest()["name"], "ckpt-gen-000001.json.gz")
            tiny = CheckpointManager(os.path.join(tmp, "one"), keep=1)
            tiny.save(model, 5)
            tiny.save(model, 1)
            self.assertEqual([r["step"] for r in tiny.list()], [1])
            self.assertEqual(tiny.latest()["step"], 1)

    def test_delete_moves_latest(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            cm = CheckpointManager(tmp, keep=5)
            a = cm.save(model, 1)
            b = cm.save(model, 2)
            self.assertTrue(cm.delete(b["name"]))
            self.assertFalse(os.path.exists(b["path"]))
            self.assertEqual(cm.latest()["name"], a["name"])
            self.assertFalse(cm.delete(b["name"]))
            self.assertFalse(cm.delete("nope"))
            self.assertTrue(cm.delete("ckpt-epoch-000001"))
            self.assertIsNone(cm.latest())
            self.assertEqual(cm.list(), [])
            self.assertFalse(os.path.exists(os.path.join(tmp, "latest.json")))
            with self.assertRaises(FileNotFoundError):
                cm.load("ckpt-epoch-000001", backend="python")


class TestLoadAndResume(unittest.TestCase):
    def test_load_latest_and_by_name_or_path(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            cm = CheckpointManager(tmp)
            rec = cm.save(model, 4, metrics={"loss": 0.1})
            for ref in (rec["name"], rec["path"], "ckpt-epoch-000004"):
                with self.subTest(ref=ref):
                    loaded = cm.load(ref, backend="python")
                    self.assertEqual(loaded.stats(), model.stats())
                    self.assertEqual(fingerprint(loaded), fingerprint(model))
            latest = cm.load_latest(backend="python")
            self.assertEqual(latest.history, model.history)
            self.assertEqual(latest.meta, model.meta)
            # a missing file behind latest.json is reported as "no latest"
            os.unlink(rec["path"])
            self.assertIsNone(cm.latest())
            self.assertIsNone(cm.load_latest(backend="python"))

    def test_resume_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = CheckpointManager(tmp, keep=2)
            reference = RadixNet(seed=3, backend="python")
            reference.train(TEXTS, epochs=3, checkpoint_manager=cm, checkpoint_every=1, **FAST)
            self.assertEqual([r["step"] for r in cm.list()], [2, 3])
            latest = cm.latest()
            self.assertEqual((latest["step"], latest["tag"], latest["metrics"]["epoch"]), (3, "epoch", 3))
            resumed = cm.load_latest(backend="python")
            self.assertEqual(resumed.meta["epochs_total"], 3)
            self.assertEqual(len(resumed.history), 3)
            more_ref = reference.train(TEXTS, epochs=2, **FAST)
            more_res = resumed.train(TEXTS, epochs=2, checkpoint_manager=cm, checkpoint_every=2, **FAST)
            self.assertEqual([r["loss"] for r in more_ref], [r["loss"] for r in more_res])
            self.assertEqual([r["epoch"] for r in more_res], [4, 5])
            self.assertEqual(fingerprint(resumed), fingerprint(reference))
            self.assertEqual([r["step"] for r in cm.list()], [3, 4])  # epoch 4 checkpointed (4 % 2 == 0)
            # 2NRL writes one checkpoint tagged "2nrl" when given a manager
            resumed.two_nrl(bad=["zzz zzz"], good=TEXTS[:2], neg_epochs=1, pos_epochs=1, checkpoint_manager=cm)
            self.assertEqual(cm.latest()["tag"], "2nrl")
            self.assertEqual(cm.latest()["step"], 1)


class TestRobustness(unittest.TestCase):
    def test_delete_refuses_manager_files_and_outside_paths(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            cm = CheckpointManager(tmp, keep=3)
            rec = cm.save(model, 1)
            for name in ("latest.json", "index.json", os.path.join(tmp, "latest.json")):
                self.assertFalse(cm.delete(name), name)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "latest.json")))
            self.assertEqual(cm.latest()["name"], rec["name"])
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
                outside = fh.name
            try:
                self.assertFalse(cm.delete(outside))
                self.assertTrue(os.path.exists(outside))
            finally:
                os.unlink(outside)
            self.assertTrue(cm.delete(rec["name"]))

    def test_resolve_accepts_either_suffix(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            gz = CheckpointManager(tmp, keep=3, compress=True).save(model, 1)
            plain = CheckpointManager(tmp, keep=3, compress=False).save(model, 2)
            cm = CheckpointManager(tmp, keep=3)
            self.assertEqual(cm.resolve("ckpt-epoch-000001.json"), gz["path"])
            self.assertEqual(cm.resolve("ckpt-epoch-000002.json.gz"), plain["path"])
            self.assertEqual(cm.resolve("ckpt-epoch-000002"), plain["path"])
            self.assertEqual(cm.load("ckpt-epoch-000001.json", backend="python").stats(), model.stats())
            with self.assertRaises(FileNotFoundError):
                cm.resolve("ckpt-epoch-000003")

    def test_plain_json_behind_gz_suffix_loads(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plain.json.gz")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(model.to_dict(), fh)
            loaded = RadixNet.load(path, backend="python")
            self.assertEqual(fingerprint(loaded), fingerprint(model))
            # a corrupt latest.json is reported as "no latest" rather than raising
            cm = CheckpointManager(tmp, keep=2)
            cm.save(model, 1)
            with open(os.path.join(tmp, "latest.json"), "w", encoding="utf-8") as fh:
                fh.write("{not json")
            self.assertIsNone(cm.latest())
            self.assertIsNone(cm.load_latest(backend="python"))
            self.assertEqual([r["step"] for r in cm.list()], [1])


if __name__ == "__main__":
    unittest.main()
