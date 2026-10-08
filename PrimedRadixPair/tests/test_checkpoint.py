"""Checkpoints: rotation, the latest pointer, resolve, resume, delete."""

import os
import tempfile
import unittest

from radixpair.checkpoint import CheckpointManager
from radixpair.model import prime


class TestCheckpoints(unittest.TestCase):
    def test_rotation_latest_resolve_and_delete(self):
        with tempfile.TemporaryDirectory() as d:
            manager = CheckpointManager(d, keep=3)
            m = prime("chars", L=3)
            for step in range(1, 6):
                m.train([f"text {step}"])
                manager.save(m, step=step, tag="epoch", metrics={"step": step})
            names = [r["name"] for r in manager.list()]
            self.assertEqual(len(names), 3)
            self.assertEqual(names[-1], "ckpt-epoch-000005.json.gz")
            self.assertEqual(manager.latest()["step"], 5)
            latest = manager.load_latest()
            self.assertEqual(latest.pair.count.texts, 5)
            self.assertEqual(manager.resolve("latest"), os.path.join(d, names[-1]))
            self.assertEqual(manager.resolve("4"), os.path.join(d, "ckpt-epoch-000004.json.gz"))
            self.assertEqual(manager.load("ckpt-epoch-000003.json.gz").pair.count.texts, 3)
            with self.assertRaises(FileNotFoundError):
                manager.resolve("1")
            self.assertTrue(manager.delete(names[-1]))
            self.assertEqual(manager.latest()["step"], 4)
            self.assertFalse(manager.delete("nothing"))
            with self.assertRaises(ValueError):
                manager.save(m, step=1, tag="bad tag")


if __name__ == "__main__":
    unittest.main()
