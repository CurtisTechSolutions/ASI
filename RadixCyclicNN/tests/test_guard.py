"""Tests for the guard - the negative network filtering the positive model on every output path.

The pair (:mod:`radixnet.duo`) used to be something a caller asked for.  It is
now what ``generate``, ``predict`` and ``converse`` do: the positive model
writes, the negative one vetoes what it recognises as a failure the tutor has
already corrected, and the answer says what it stopped.  Covered here: the
filter's own additions (``ready``, ``rank``, ``converse``), the veto hook in
the conversation, the three service methods, the three HTTP routes and the
three CLI commands.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.api import ModelService  # noqa: E402
from radixnet.duo import FilterConfig, NegativeFilter  # noqa: E402
from radixnet.model import RadixNet  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a bird flew over the hill",
]
FAILURE = "the cat sat on the mat"  # blamed in most of these tests, so the guard has something to veto


def service(tmp=None, train=True, seed=5):
    """A service over a trained radix model, with the negative network beside it when ``tmp`` is given."""
    svc = ModelService(model_path=os.path.join(tmp, "model.json") if tmp else None, seed=seed, quiet=True,
                       backend="python")
    if train:
        svc.model.train(CORPUS, epochs=3)
    return svc


class TestFilterAdditions(unittest.TestCase):
    """``ready``, ``rank`` and ``converse``: what the filter needed to stand on the output path."""

    def pair(self, blamed=True, config=None):
        positive = RadixNet(seed=1)
        positive.train(CORPUS, epochs=3)
        negative = NegativeNet(seed=2)
        if blamed:
            negative.blame([FAILURE], reason="gibberish")
        return NegativeFilter(positive, negative, config)

    def test_ready_is_false_until_a_failure_is_blamed(self):
        self.assertFalse(self.pair(blamed=False).ready)
        self.assertTrue(self.pair().ready)

    def test_ready_goes_false_again_when_the_blame_is_forgotten(self):
        duo = self.pair()
        duo.negative.forget("gibberish")
        self.assertFalse(duo.ready)

    def test_generate_carries_the_walks_behind_the_texts(self):
        duo = self.pair()
        out = duo.generate(count=2, max_length=30, seed=5)
        self.assertEqual([r.text for r in out["results"]], out["texts"])
        self.assertTrue(all(r.cost >= 0 for r in out["results"]))
        self.assertEqual(duo.generate(count=0)["results"], [])

    def test_generate_keeps_the_models_own_order_among_equally_clean_texts(self):
        duo = self.pair(blamed=False)  # nothing is blamed: every candidate has risk 0
        duo.negative.blame(["nothing here matches the corpus at all"], reason="unrelated")
        out = duo.generate(count=5, max_length=30, seed=11, over_sample=2)
        self.assertEqual(out["texts"], [t for t in out["kept"]][: len(out["texts"])])

    def test_rank_drops_the_vetoed_continuations(self):
        duo = self.pair()
        prediction = duo.positive.predict("the ", length=10, mode="beam", k=4)
        ranked, verdicts = duo.rank("the ", prediction)
        self.assertEqual(len(verdicts), len(prediction.top))
        self.assertTrue(all(v["text"].startswith("the ") for v in verdicts))
        self.assertEqual(len(ranked.top), sum(v["decision"] != "reject" for v in verdicts))
        if ranked.top:
            self.assertEqual(ranked.text, ranked.top[0].text)
        self.assertEqual(ranked.expanded, prediction.expanded)  # the search that ran is still reported
        self.assertEqual((ranked.k, ranked.beam, ranked.mode), (prediction.k, prediction.beam, prediction.mode))

    def test_rank_leaves_nothing_when_every_continuation_is_vetoed(self):
        duo = self.pair()
        duo.negative.blame(CORPUS, reason="gibberish")
        duo.config.threshold, duo.config.min_coverage = 0.0, 0.0
        prediction = duo.positive.predict("the ", length=10, mode="beam", k=4)
        ranked, verdicts = duo.rank("the ", prediction)
        self.assertTrue(all(v["decision"] == "reject" for v in verdicts))
        self.assertEqual((ranked.text, ranked.full_text, ranked.top), ("", "the ", []))

    def test_rank_handles_a_single_path_result(self):
        duo = self.pair()
        found = duo.positive.predict("the ", length=8, mode="dijkstra")
        ranked, verdicts = duo.rank("the ", found)
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(ranked.text, found.text if verdicts[0]["decision"] != "reject" else "")

    def test_converse_leaves_the_vetoed_replies_unsaid(self):
        duo = self.pair()
        duo.negative.blame(CORPUS, reason="gibberish")
        duo.config.threshold, duo.config.min_coverage = 0.0, 0.0  # everything the model can say is known-bad
        out = duo.converse("", turns=3, mode="beam")
        self.assertEqual(out["turns"], [])  # nothing may be spoken, so nothing is
        self.assertTrue(out["rejected"])
        self.assertTrue(all(v["decision"] == "reject" for v in out["verdicts"]))

    def test_converse_speaks_what_is_allowed(self):
        duo = self.pair()
        out = duo.converse("the cat sat on the mat", turns=3, mode="beam")
        self.assertTrue(out["turns"])
        self.assertEqual(out["turns"][0].text, "the cat sat on the mat")
        self.assertEqual(out["vetoed"], sum(t.vetoed for t in out["turns"]))
        for turn in out["turns"][1:]:
            self.assertNotIn(turn.text, [v["text"] for v in out["rejected"]])

    def test_converse_can_blame_what_it_refused(self):
        duo = self.pair(config=FilterConfig(learn=True, reason="vetoed"))
        duo.negative.blame(CORPUS, reason="gibberish")
        duo.config.threshold, duo.config.min_coverage = 0.0, 0.0
        before = duo.negative.graph.total_blame
        duo.converse("", turns=2, mode="beam")
        self.assertGreater(duo.negative.graph.total_blame, before)
        self.assertIn("vetoed", [r["reason"] for r in duo.negative.reasons()])


class TestServiceGuard(unittest.TestCase):
    """``ModelService.guard`` and the three output paths."""

    def test_no_negative_network_means_no_guard(self):
        svc = service()
        self.assertIsNone(svc.guard())
        self.assertIsNone(svc.generate(count=2, mode="sample", seed=1)["guard"])
        self.assertIsNone(svc.predict("the ", length=6)["guard"])
        self.assertIsNone(svc.converse("the cat sat on the mat", turns=2)["guard"])

    def test_an_untaught_negative_network_guards_nothing(self):
        svc = service()
        svc.negative_model()  # in memory, but never taught a failure
        self.assertIsNone(svc.guard())

    def test_the_guard_never_creates_a_negative_network(self):
        svc = service()
        svc.guard()
        self.assertNotIn("negative", svc.describe_model()["in_memory"])

    def test_generate_reports_what_the_negative_network_saw(self):
        svc = service()
        svc.negative_model().blame([FAILURE], reason="gibberish")
        out = svc.generate(count=3, mode="sample", seed=1, max_length=40)
        report = out["guard"]
        self.assertTrue(report["on"])
        self.assertEqual(report["candidates"], len(report["verdicts"]))
        self.assertEqual(report["asked"], 9)  # three wanted, over-sampled three times
        self.assertEqual(report["vetoed"], len(report["rejected"]))
        kept = [s["text"] for s in out["samples"]]
        self.assertFalse(set(kept) & {v["text"] for v in report["rejected"]})
        self.assertEqual(report["negative"]["kind"], "negative")

    def test_generate_drops_every_candidate_the_tutor_has_failed(self):
        svc = service()
        svc.negative_model().blame(CORPUS, reason="gibberish")
        svc.guard_config = FilterConfig(threshold=0.0, min_coverage=0.0)  # everything it can write is known-bad
        out = svc.generate(count=3, mode="sample", seed=1, max_length=40)
        self.assertEqual(out["samples"], [])
        self.assertGreater(out["guard"]["vetoed"], 0)
        self.assertEqual(out["guard"]["rejected"][0]["rule"], "blame")
        self.assertEqual(out["guard"]["rate"], 0.0)

    def test_generate_can_be_asked_not_to_guard(self):
        svc = service()
        svc.negative_model().blame([FAILURE], reason="gibberish")
        out = svc.generate(count=3, mode="sample", seed=1, max_length=40, guard=False)
        self.assertIsNone(out["guard"])

    def test_predict_returns_the_best_continuation_that_survives(self):
        svc = service()
        svc.negative_model().blame([FAILURE], reason="gibberish")
        out = svc.predict("the ", length=12, mode="beam", k=4)
        report = out["guard"]
        self.assertTrue(report["on"])
        self.assertEqual(len(out["top"]), report["kept"])
        for row in out["top"]:
            self.assertNotIn("the " + row["continuation"], [v["text"] for v in report["rejected"]])
        if out["top"]:
            self.assertEqual(out["continuation"], out["top"][0]["continuation"])

    def test_predict_comes_back_empty_when_everything_is_vetoed(self):
        svc = service()
        svc.negative_model().blame(CORPUS, reason="gibberish")
        svc.guard_config = FilterConfig(threshold=0.0, min_coverage=0.0)
        out = svc.predict("the ", length=12, mode="beam", k=4)
        self.assertEqual((out["continuation"], out["full_text"]), ("", "the "))
        self.assertEqual(out["top"], [])
        self.assertEqual(out["guard"]["kept"], 0)
        self.assertGreater(out["guard"]["vetoed"], 0)

    def test_converse_counts_its_vetoes_per_turn(self):
        svc = service()
        svc.negative_model().blame([FAILURE], reason="gibberish")
        out = svc.converse("the cat sat on the mat", turns=4, mode="beam")
        self.assertTrue(out["guard"]["on"])
        self.assertEqual(out["guard"]["refusals"], sum(t["vetoed"] for t in out["turns"]))
        for turn in out["turns"][1:]:
            self.assertNotIn(turn["text"], [v["text"] for v in out["guard"]["rejected"]])

    def test_the_negative_model_as_the_active_one_is_not_guarded(self):
        svc = service()
        svc.negative_model().blame([FAILURE], reason="gibberish")
        svc.select_kind("negative")
        self.assertIsNone(svc.guard())
        self.assertIsNone(svc.generate(count=1, mode="sample", seed=1)["guard"])

    def test_the_guard_loads_the_negative_model_beside_the_model_file(self):
        with tempfile.TemporaryDirectory(prefix="radixnet-guard-") as tmp:
            negative = NegativeNet(seed=2)
            negative.blame([FAILURE], reason="gibberish")
            negative.save(os.path.join(tmp, "model.negative.json"))
            svc = service(tmp)
            pair = svc.guard()
            self.assertIsNotNone(pair)
            self.assertEqual(pair.negative.reasons()[0]["reason"], "gibberish")


class TestGuardApi(unittest.TestCase):
    """The guard over HTTP: on by default, off on request, reported in every answer."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-guard-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, model_path=os.path.join(cls.tmp.name, "model.json")
        )
        cls.service.model.train(CORPUS, epochs=3)

    def setUp(self):
        status, data, _ = self.client.post("/api/negative/reset")
        self.assertEqual(status, 200, data)

    def blame(self, texts=(FAILURE,), reason="gibberish"):
        status, data, _ = self.client.post("/api/negative/blame", {"texts": list(texts), "reason": reason})
        self.assertEqual(status, 200, data)

    def test_generate_reports_the_guard(self):
        status, data, _ = self.client.post("/api/generate", {"count": 3, "mode": "sample", "seed": 1})
        self.assertEqual(status, 200, data)
        self.assertIsNone(data["guard"])  # an empty negative network vetoes nothing
        self.blame()
        status, data, _ = self.client.post("/api/generate", {"count": 3, "mode": "sample", "seed": 1})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["guard"]["on"])
        self.assertNotIn(FAILURE, [s["text"] for s in data["samples"]])
        self.assertTrue(json.dumps(data))  # the walks behind the texts never reach the wire

    def test_guard_false_turns_it_off(self):
        self.blame()
        status, data, _ = self.client.post("/api/generate", {"count": 2, "guard": False})
        self.assertEqual(status, 200, data)
        self.assertIsNone(data["guard"])

    def test_predict_and_converse_report_the_guard(self):
        self.blame()
        status, data, _ = self.client.post("/api/predict", {"prefix": "the ", "mode": "beam", "k": 4})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["guard"]["on"])
        self.assertEqual(len(data["top"]), data["guard"]["kept"])
        status, data, _ = self.client.post("/api/converse", {"opening": FAILURE, "turns": 3})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["guard"]["on"])
        self.assertTrue(all("vetoed" in t for t in data["turns"]))


class TestGuardCli(unittest.TestCase):
    """``generate`` / ``predict`` / ``converse`` filter through the negative model beside --model."""

    @classmethod
    def setUpClass(cls):
        from tests.test_cli import run_json

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-guard-cli-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model = os.path.join(cls.tmp.name, "model.json")
        cls.negative = os.path.join(cls.tmp.name, "model.negative.json")
        run_json("train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 2,
                 "--batch-size", 8, "--lr", 0.5, model=cls.model)

    def run_json(self, *args, **kwargs):
        from tests.test_cli import run_json

        kwargs.setdefault("model", self.model)
        return run_json(*args, **kwargs)

    def test_no_negative_model_means_no_guard(self):
        self.assertFalse(os.path.isfile(self.negative))
        doc = self.run_json("generate", "--count", 2, "--mode", "sample", "--seed", 3)
        self.assertIsNone(doc["guard"])

    def test_the_three_commands_filter_once_a_failure_is_blamed(self):
        blamed = "the quick brown fox"
        self.run_json("negative", "blame", "--text", blamed, "--reason", "gibberish")
        self.assertTrue(os.path.isfile(self.negative))

        doc = self.run_json("generate", "--count", 3, "--mode", "sample", "--seed", 3, "--max-length", 40)
        self.assertTrue(doc["guard"]["on"])
        self.assertEqual(doc["guard"]["asked"], 9)
        self.assertEqual(len(doc["guard"]["verdicts"]), doc["guard"]["candidates"])
        self.assertNotIn(blamed, [s["text"] for s in doc["samples"]])

        doc = self.run_json("predict", "--prefix", "the ", "--mode", "beam", "--k", 4)
        self.assertTrue(doc["guard"]["on"])
        self.assertEqual(len(doc["top"]), doc["guard"]["kept"])

        doc = self.run_json("converse", "--turns", 3, "--opening", "the quick brown fox")
        self.assertTrue(doc["guard"]["on"])
        self.assertTrue(all("vetoed" in t for t in doc["turns"]))

        doc = self.run_json("generate", "--count", 3, "--mode", "sample", "--seed", 3, "--no-guard")
        self.assertIsNone(doc["guard"])

    def test_the_guard_takes_its_thresholds_from_the_command_line(self):
        self.run_json("negative", "blame", "--text", "the quick brown fox", "--reason", "gibberish")
        doc = self.run_json("generate", "--count", 2, "--mode", "sample", "--seed", 3, "--over-sample", 5,
                            "--threshold", 1e9, "--min-coverage", 1.0)
        self.assertEqual(doc["guard"]["asked"], 10)
        self.assertEqual(doc["guard"]["config"]["threshold"], 1e9)


if __name__ == "__main__":
    unittest.main()
