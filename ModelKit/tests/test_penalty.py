"""The punishment traversal through the kit's front doors: the command line and the HTTP API (the model side is RadixCyclicNN's tests/test_penalty.py).
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from modelkit import cli  # noqa: E402
from modelkit.api import ModelService  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402


TEXTS = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "the cat ran up the hill",
]


PRAISED = "the cat sat on the mat"


PUNISHED = "the cat ate the rat"


def counted(epochs=4, seed=1):
    """A count / reward model with one branch praised, one punished and one left alone."""
    model = CountRewardNet(seed=seed)
    model.train(TEXTS, epochs=epochs)
    model.reward([PRAISED], strength=4.0)
    model.punish([PUNISHED], strength=4.0)
    return model


def _json_cli(argv):
    """Run the CLI with ``--json`` and give back the document it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(argv)
    if code != 0:
        raise AssertionError(f"cli exited {code}: {buffer.getvalue()}")
    return json.loads(buffer.getvalue())


class TestFrontDoors(unittest.TestCase):
    """The option where a person reaches it: the CLI and the HTTP API."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "model.count.json")
        counted().save(self.path)
        self.addCleanup(self.dir.cleanup)

    def test_the_cli_carries_the_traversal(self):
        doc = _json_cli(["--model", self.path, "--json", "predict", "--prefix", "the cat", "--length", "16"])
        self.assertEqual(doc["traversal"], "reward")
        punished = _json_cli([
            "--model", self.path, "--json", "predict", "--prefix", "the cat", "--length", "16",
            "--traversal", "punishment", "--merit-scale", "0",
        ])
        self.assertEqual(punished["traversal"], "punishment")
        self.assertNotEqual(doc["continuation"], punished["continuation"])

    def test_the_cli_generates_with_it(self):
        doc = _json_cli([
            "--model", self.path, "--json", "generate", "--count", "2", "--mode", "beam",
            "--max-length", "40", "--traversal", "punishment",
        ])
        self.assertEqual(doc["traversal"], "punishment")
        self.assertTrue(doc["samples"])

    def test_the_cli_rejects_an_unknown_traversal(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["--model", self.path, "--json", "predict", "--prefix", "a", "--traversal", "nope"])
        self.assertNotEqual(code, 0)

    def test_the_api_carries_the_traversal(self):
        svc = ModelService(self.path, backend="python")
        plain = svc.predict("the cat", length=16, k=3, mode="beam", guard=False)
        punished = svc.predict(
            "the cat", length=16, k=3, mode="beam", guard=False, traversal="punishment", merit_scale=0.0,
        )
        self.assertEqual(plain["traversal"], "reward")
        self.assertEqual(punished["traversal"], "punishment")
        self.assertNotEqual(plain["continuation"], punished["continuation"])

    def test_the_api_generates_with_it(self):
        svc = ModelService(self.path, backend="python")
        out = svc.generate(count=2, mode="beam", max_length=40, guard=False, traversal="punishment")
        self.assertTrue(out["samples"])


if __name__ == "__main__":
    unittest.main()
