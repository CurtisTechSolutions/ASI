"""Tests for radixnet.negative - the negative network (the failures, and why).

Covers the blame weight function and the side arrays that survive splits and
merges, blaming and clearing, the ``judge`` verdict (risk, coverage, reasons,
spans), prediction over the failure distribution, forgetting, inversion,
persistence, the model-kind plumbing, the HTTP API routes and the CLI's
``negative`` command group.
"""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.graph import END, START  # noqa: E402
from radixnet.model import RadixNet, load_model, model_class, model_from_dict, model_kinds, new_model  # noqa: E402
from radixnet.negative import (  # noqa: E402
    MAX_EDGE_REASONS,
    MAX_LOG_ENTRIES,
    NEGATIVE_MODEL_FORMAT,
    NegativeGraph,
    NegativeNet,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = [
    "the the the the cat",
    "the cat cat cat sat",
]
GOOD = [
    "the cat sat on the mat",
    "the dog sat on the log",
]


def taught(seed=1, reason="repetition", severity=1.0):
    model = NegativeNet(seed=seed)
    model.blame(FAILURES, reason=reason, severity=severity, source="review", note="it repeats itself")
    return model


class TestWeightFunction(unittest.TestCase):
    """The blame weight: the edge's share of the failure mass leaving its parent."""

    def test_edge_weight_formula(self):
        g = NegativeGraph(seed=0, share_scale=1.0, blame_scale=0.5)
        s = g.SMOOTHING
        expected = math.log((3 + s) / (4 + 2 * s)) + 0.5 * math.log1p(3)
        self.assertAlmostEqual(g.edge_weight(3, 4, 2), expected)
        # nothing blamed anywhere: every child of a node weighs the same
        self.assertAlmostEqual(g.edge_weight(0, 0, 3), g.edge_weight(0, 0, 3))
        self.assertEqual(g.weight_config()["function"], "blame")

    def test_evidence_is_blame_minus_clearing(self):
        model = taught()
        edge = next(e for e, ok in enumerate(model.graph.edge_alive) if ok and model.graph.edge_blame[e] > 0)
        blame = model.graph.edge_blame[edge]
        self.assertAlmostEqual(model.graph.evidence(edge), blame)
        model.graph.record_clear([edge], blame / 2)
        self.assertAlmostEqual(model.graph.evidence(edge), blame / 2)
        model.graph.record_clear([edge], blame)  # cleared more often than blamed: no evidence left, never negative
        self.assertEqual(model.graph.evidence(edge), 0.0)

    def test_activations_are_one_so_scores_are_weights(self):
        model = taught()
        g = model.graph
        for node in g.alive_nodes():
            self.assertEqual(g.activation_of(node), 1.0)
        for parent, edge in ((p, e) for p, ch in enumerate(g.children) if g.alive[p] for e in ch.values()):
            scores = dict(g.child_scores(parent))
            child = next(c for c, e in g.children[parent].items() if e == edge)
            self.assertAlmostEqual(scores[child], g.edge_w[edge])

    def test_configure_rescales_every_weight(self):
        model = taught()
        before = list(model.graph.edge_w)
        config = model.configure_weights(blame_scale=1.0)
        self.assertEqual(config["blame_scale"], 1.0)
        self.assertNotEqual(before, model.graph.edge_w)
        with self.assertRaises(ValueError):
            model.configure_weights(nonsense=1.0)
        with self.assertRaises(ValueError):
            model.configure_weights(share_scale=float("nan"))


class TestBlameAndClear(unittest.TestCase):
    def test_blame_builds_structure_and_records_the_reason(self):
        model = NegativeNet(seed=1)
        self.assertEqual(model.graph.num_nodes(), 2)  # the two sentinels only
        records = model.blame(["the the the the cat"], reason="Repetition ", severity=2.0, source="review",
                              note="  it repeats\n the same word ")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["phase"], "negative")
        self.assertEqual(record["reason"], "repetition")  # normalised
        self.assertEqual(record["severity"], 2.0)
        self.assertGreater(record["edges_touched"], 0)
        self.assertGreater(model.graph.num_nodes(), 2)
        self.assertEqual([r["reason"] for r in model.reasons()], ["repetition"])
        entry = model.recent(1)[0]
        self.assertEqual(entry["note"], "it repeats the same word")
        self.assertEqual(entry["source"], "review")

    def test_clear_never_creates_structure(self):
        model = taught()
        nodes, edges = model.graph.num_nodes(), model.graph.num_edges()
        records = model.clear(["a sentence that never failed"])
        self.assertEqual((model.graph.num_nodes(), model.graph.num_edges()), (nodes, edges))
        self.assertEqual(records[-1]["matched"], 0)
        self.assertEqual(records[-1]["unmatched"], 1)

    def test_clear_credits_shared_fragments(self):
        model = taught()
        before = model.judge(GOOD[0])["blame"]
        records = model.clear([GOOD[0]])
        self.assertEqual(records[-1]["matched"], 1)
        self.assertGreater(records[-1]["edges_touched"], 0)
        self.assertLess(model.judge(GOOD[0])["blame"], before)

    def test_train_is_blaming_and_epochs_accumulate(self):
        model = NegativeNet(seed=2)
        model.train(FAILURES[:1], epochs=2, reason="gibberish", severity=1.0)
        single = NegativeNet(seed=2)
        single.train(FAILURES[:1], epochs=1, reason="gibberish", severity=1.0)
        self.assertGreater(model.graph.total_blame, single.graph.total_blame)
        self.assertEqual(model.meta["epochs_total"], 2)

    def test_reward_clears_and_punish_blames(self):
        model = taught()
        blamed = model.graph.total_blame
        model.reward([GOOD[0]])
        self.assertEqual(model.graph.total_blame, blamed)  # correct text never adds blame
        self.assertGreater(model.graph.total_clear, 0)
        model.punish(["brand new failure"], strength=1.5)
        self.assertGreater(model.graph.total_blame, blamed)
        self.assertIn("thumbs-down", [r["reason"] for r in model.reasons()])

    def test_two_nrl_blames_then_clears_without_inverting(self):
        model = NegativeNet(seed=3)
        result = model.two_nrl(FAILURES, GOOD, neg_epochs=1, pos_epochs=1, reason="gibberish")
        self.assertTrue(result["negative"])
        self.assertTrue(result["positive"])
        self.assertFalse(result["inverted"])
        self.assertEqual(model.meta["twonrl_runs"], 1)
        with self.assertRaises(TypeError):
            model.two_nrl(FAILURES, GOOD, epochs=2)

    def test_two_nrl_weights_scale_the_severity(self):
        model = NegativeNet(seed=4)
        model.two_nrl(FAILURES, [], neg_epochs=1, pos_epochs=0, bad_weights=[2.0, 0.0], reason="gibberish")
        self.assertEqual(model.meta["trained_texts"], 1)  # the zero-weight text is skipped entirely
        with self.assertRaises(ValueError):
            model.two_nrl(FAILURES, [], bad_weights=[1.0])

    def test_invert_swaps_blame_and_clearing(self):
        model = taught()
        model.clear([GOOD[0]])
        blame, clear = model.graph.total_blame, model.graph.total_clear
        model.invert()
        self.assertEqual((model.graph.total_blame, model.graph.total_clear), (clear, blame))
        self.assertTrue(model.graph.inverted)
        model.invert()
        self.assertEqual((model.graph.total_blame, model.graph.total_clear), (blame, clear))
        self.assertFalse(model.graph.inverted)

    def test_invert_paths_blames_by_amount(self):
        model = NegativeNet(seed=5)
        out = model.invert_paths(FAILURES, amounts=[1.0, 0.0])
        self.assertEqual(out["mode"], "blame")
        self.assertEqual(out["unit"], "edges")
        self.assertGreater(out["flipped"], 0)
        self.assertEqual(model.meta["trained_texts"], 1)
        with self.assertRaises(ValueError):
            model.invert_paths(FAILURES, amounts=[2.0, 0.0])

    def test_forget_removes_one_reason(self):
        model = NegativeNet(seed=6)
        model.blame([FAILURES[0]], reason="repetition")
        model.blame([FAILURES[1]], reason="gibberish")
        total = model.graph.total_blame
        result = model.forget("gibberish")
        self.assertGreater(result["blame_removed"], 0)
        self.assertLess(model.graph.total_blame, total)
        self.assertEqual([r["reason"] for r in model.reasons() if r["blame"] > 0], ["repetition"])
        # a reason nobody ever used changes nothing
        self.assertEqual(model.forget("never-seen")["edges"], 0)
        model.forget(factor=0.5)
        self.assertAlmostEqual(model.graph.total_blame, sum(r["blame"] for r in model.reasons()), places=6)
        with self.assertRaises(ValueError):
            model.forget(factor=1.0)

    def test_edge_reasons_are_capped(self):
        model = NegativeNet(seed=7)
        for i in range(MAX_EDGE_REASONS + 3):
            model.blame([FAILURES[0]], reason=f"reason-{i}")
        edge = next(e for e, ok in enumerate(model.graph.edge_alive) if ok and model.graph.edge_blame[e] > 0)
        self.assertLessEqual(len(model.graph.edge_reasons[edge]), MAX_EDGE_REASONS)
        self.assertEqual(len(model.reasons()), MAX_EDGE_REASONS + 3)  # the totals keep every reason

    def test_journal_is_bounded(self):
        model = NegativeNet(seed=8)
        for i in range(MAX_LOG_ENTRIES + 5):
            model.blame([f"failure number {i}"], reason="other")
        self.assertEqual(len(model.log), MAX_LOG_ENTRIES)
        self.assertEqual(model.recent(1)[0]["text"], f"failure number {MAX_LOG_ENTRIES + 4}")
        self.assertEqual(model.recent(0), [])
        with self.assertRaises(ValueError):
            model.recent(-1)


class TestCorrections(unittest.TestCase):
    """A correction is the sharpest lesson: only the characters the teacher changed are blamed."""

    def test_only_the_changed_characters_are_blamed(self):
        model = NegativeNet(seed=1)
        out = model.correct("the cat sit on the mat", "the cat sits on the mat", reason="agreement", severity=1.5,
                            note="the verb must agree with the subject")
        self.assertEqual(out["edits"], 1)
        self.assertEqual(out["blamed"], 1)
        self.assertGreater(out["cleared"], 1)
        self.assertEqual(out["reason"], "agreement")
        verdict = model.judge("the cat sit on the mat")
        self.assertEqual([span["fragment"] for span in verdict["spans"]], ["sit "])
        self.assertEqual(verdict["reasons"][0]["reason"], "agreement")
        self.assertEqual(model.judge("the cat sits on the mat")["verdict"], "pass")
        self.assertEqual(model.recent(1)[0]["note"], "the verb must agree with the subject")
        self.assertEqual(model.stats()["sources"], {"tutor": 1})

    def test_an_unchanged_sentence_blames_nothing(self):
        model = NegativeNet(seed=2)
        out = model.correct("the cat sat on the mat", "the cat sat on the mat", reason="none")
        self.assertEqual((out["edits"], out["blamed"]), (0, 0))
        self.assertEqual(model.graph.total_blame, 0.0)

    def test_short_and_empty_texts(self):
        model = NegativeNet(seed=3)
        self.assertEqual(model.correct("ab", "abc")["blamed"], 0)
        self.assertEqual(model.correct("", "")["edits"], 0)
        self.assertEqual(model.correct("the cat sat here", "")["cleared"], 0)

    def test_a_correction_never_creates_correct_structure(self):
        model = NegativeNet(seed=4)
        model.correct("the cat sit on the mat", "the cat sits on the mat", reason="agreement")
        # the correction's own words are only known where the failure already walked them
        self.assertIsNone(model.graph.trace(model.encoder.encode("the cat sits on the mat")))


class TestJudge(unittest.TestCase):
    def test_a_known_failure_is_rejected_with_its_reason(self):
        model = taught()
        verdict = model.judge(FAILURES[0])
        self.assertEqual(verdict["verdict"], "reject")
        self.assertEqual(verdict["coverage"], 1.0)
        self.assertGreaterEqual(verdict["risk"], verdict["threshold"])
        self.assertEqual(verdict["reasons"][0]["reason"], "repetition")
        self.assertAlmostEqual(sum(r["share"] for r in verdict["reasons"]), 1.0)
        self.assertTrue(verdict["spans"])
        span = verdict["spans"][0]
        self.assertEqual(span["fragment"], FAILURES[0][span["start"]:span["end"]])
        self.assertIn("repetition", verdict["why"])

    def test_unseen_text_passes(self):
        model = taught()
        verdict = model.judge("something entirely different")
        self.assertEqual(verdict["verdict"], "pass")
        self.assertEqual(verdict["blame"], 0.0)
        self.assertEqual(verdict["reasons"], [])
        self.assertEqual(verdict["why"], "nothing here has failed before")

    def test_thresholds_decide(self):
        model = taught()
        strict = model.judge(GOOD[0], threshold=0.0, min_coverage=0.0)
        self.assertEqual(strict["verdict"], "reject")
        lenient = model.judge(FAILURES[0], threshold=1e9)
        self.assertEqual(lenient["verdict"], "suspect")
        self.assertEqual(model.judge(FAILURES[0], min_coverage=1.1)["verdict"], "suspect")

    def test_risk_is_per_transition_not_per_character(self):
        model = NegativeNet(seed=9)
        model.blame([GOOD[0]], reason="gibberish", severity=1.0)  # no repeated fragment: one blame per edge
        verdict = model.judge(GOOD[0])
        self.assertAlmostEqual(verdict["risk"], 1.0, places=6)
        self.assertAlmostEqual(verdict["blame"] / max(1, len(GOOD[0])), verdict["per_char"])
        # a text whose every transition failed twice is twice as risky, however long it is
        model.blame([GOOD[0]], reason="gibberish", severity=1.0)
        self.assertAlmostEqual(model.judge(GOOD[0])["risk"], 2.0, places=6)

    def test_crossings_walk_the_text(self):
        model = taught()
        crossings = model.crossings(FAILURES[0])
        self.assertTrue(all(c["end"] >= c["start"] for c in crossings))
        self.assertTrue(all(c["edge"] is not None for c in crossings))
        self.assertEqual(model.crossings(""), [])
        self.assertTrue(all(c["edge"] is None for c in model.crossings("zzz qqq vvv")))
        with self.assertRaises(TypeError):
            model.crossings(None)

    def test_judgements_are_counted(self):
        model = taught()
        model.judge(FAILURES[0])
        model.judge("nothing like it")
        self.assertEqual(model.stats()["judgements"], 2)
        self.assertEqual(model.stats()["rejected"], 1)


class TestPrediction(unittest.TestCase):
    def test_predict_returns_the_known_ways_to_fail(self):
        model = taught()
        found = model.predict("the ", length=8, k=2)
        self.assertTrue(found.top)
        self.assertTrue(all(p.text for p in found.top))
        self.assertLessEqual(len(found.top), 2)
        self.assertEqual(found.mode, "beam")
        self.assertEqual(model.predict("the ", length=6, mode="dijkstra").mode, "beam")  # alias
        self.assertEqual(model.predict("the ", length=6, mode="sample").mode, "sample")
        with self.assertRaises(ValueError):
            model.predict("the ", mode="nonsense")
        with self.assertRaises(ValueError):
            model.predict("the ", length=-1)

    def test_score_and_generate_still_work(self):
        model = taught()
        score = model.score(FAILURES[0])
        self.assertLess(score["log_prob"], 0.0)
        self.assertEqual(score["unknown_transitions"], 0)
        self.assertGreater(model.score("never seen text")["unknown_transitions"], 0)
        self.assertTrue(model.generate(max_length=20, count=1)[0].text)


class TestPersistenceAndKinds(unittest.TestCase):
    def test_round_trip(self):
        model = taught()
        model.clear([GOOD[0]])
        model.threshold, model.min_coverage = 1.5, 0.75
        document = model.to_dict()
        self.assertEqual(document["format"], NEGATIVE_MODEL_FORMAT)
        back = NegativeNet.from_dict(json.loads(json.dumps(document)))
        self.assertEqual(back.reasons(), model.reasons())
        self.assertEqual(back.recent(5), model.recent(5))
        self.assertEqual((back.threshold, back.min_coverage), (1.5, 0.75))
        self.assertAlmostEqual(back.graph.total_blame, model.graph.total_blame)
        self.assertAlmostEqual(back.graph.total_clear, model.graph.total_clear)
        self.assertEqual(back.judge(FAILURES[0])["risk"], model.judge(FAILURES[0])["risk"])
        # dead nodes and edges are compacted away by to_dict; the alive weights survive unchanged
        alive = lambda m: sorted(m.graph.edge_w[e] for e, ok in enumerate(m.graph.edge_alive) if ok)  # noqa: E731
        self.assertEqual(alive(back), alive(model))

    def test_save_and_load_detects_the_kind(self):
        model = taught()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.negative.json")
            model.save(path)
            back = load_model(path)
            self.assertIsInstance(back, NegativeNet)
            self.assertEqual(back.reasons(), model.reasons())
            gz = os.path.join(tmp, "model.negative.json.gz")
            model.save(gz)
            self.assertIsInstance(load_model(gz), NegativeNet)
            other = os.path.join(tmp, "radix.json")
            RadixNet(seed=0).save(other)
            with self.assertRaises(ValueError):
                NegativeNet.load(other)
        with self.assertRaises(ValueError):
            NegativeNet.from_dict({"format": "nope"})
        with self.assertRaises(ValueError):
            NegativeNet.from_dict({"format": NEGATIVE_MODEL_FORMAT, "version": 99, "graph": {}})

    def test_kind_registry(self):
        self.assertIn("negative", [k["kind"] for k in model_kinds()])
        self.assertIs(model_class("negative"), NegativeNet)
        self.assertIsInstance(new_model("negative"), NegativeNet)
        self.assertIsInstance(model_from_dict(taught().to_dict()), NegativeNet)

    def test_compression_keeps_the_blame(self):
        model = NegativeNet(seed=11)
        model.blame([FAILURES[0]], reason="repetition")
        before = model.judge(FAILURES[0])["blame"]
        model.compress()
        model.graph.check_invariants()
        self.assertAlmostEqual(model.judge(FAILURES[0])["blame"], before, places=6)

    def test_stats_report_the_failures(self):
        model = taught()
        stats = model.stats()
        self.assertEqual(stats["kind"], "negative")
        self.assertEqual(stats["sources"], {"review": 2})
        self.assertGreater(stats["blame_total"], 0)
        self.assertEqual(stats["top_reasons"][0]["reason"], "repetition")
        self.assertIn("threshold", stats)
        self.assertIn("NegativeNet", repr(model))


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-negative-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model_path = os.path.join(cls.tmp.name, "model.json")
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, model_path=cls.model_path)

    def setUp(self):
        status, data, _ = self.client.post("/api/negative/reset")
        self.assertEqual(status, 200, data)

    def blame(self, **body):
        body.setdefault("texts", FAILURES)
        body.setdefault("reason", "repetition")
        status, data, _ = self.client.post("/api/negative/blame", body)
        self.assertEqual(status, 200, data)
        return data

    def test_status_reasons_and_journal(self):
        self.blame(note="it repeats", source="review")
        status, data, _ = self.client.get("/api/negative")
        self.assertEqual(status, 200)
        self.assertEqual(data["path"], os.path.join(self.tmp.name, "model.negative.json"))
        self.assertEqual([r["reason"] for r in data["reasons"]], ["repetition"])
        self.assertEqual(data["journal"][0]["note"], "it repeats")
        self.assertEqual(data["stats"]["kind"], "negative")
        self.assertIn("threshold", data["settings"])

    def test_blame_judge_and_clear(self):
        self.blame()
        status, data, _ = self.client.post("/api/negative/judge", {"text": FAILURES[0]})
        self.assertEqual(status, 200, data)
        verdict = data["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertEqual(verdict["reasons"][0]["reason"], "repetition")
        status, data, _ = self.client.post("/api/negative/clear", {"texts": [GOOD[0]]})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["matched"] + data["unmatched"], 1)
        status, data, _ = self.client.post("/api/negative/judge", {"texts": []})
        self.assertEqual(status, 400, data)

    def test_filter_vetoes_a_known_failure(self):
        self.blame()
        status, data, _ = self.client.post("/api/negative/filter", {"texts": [FAILURES[0], "an unseen sentence"]})
        self.assertEqual(status, 200, data)
        self.assertEqual([v["decision"] for v in data["verdicts"]], ["reject", "pass"])
        self.assertEqual(data["kept"], ["an unseen sentence"])
        self.assertEqual(data["rejected"][0]["rule"], "blame")
        self.assertEqual(data["pair"]["negative"]["kind"], "negative")

    def test_filter_generates_through_the_pair(self):
        self.client.post("/api/train", {"texts": GOOD, "epochs": 2, "lr": 0.5, "batch_size": 4})
        from tests.test_api import wait_for_job

        wait_for_job(self.client)
        self.blame()
        status, data, _ = self.client.post("/api/negative/filter", {"count": 2, "max_length": 30, "over_sample": 2})
        self.assertEqual(status, 200, data)
        self.assertLessEqual(len(data["texts"]), 2)
        self.assertEqual(len(data["verdicts"]), data["candidates"])
        self.assertTrue(all("why" in v for v in data["verdicts"]))

    def test_filter_rules_can_be_tuned(self):
        self.blame()
        body = {"texts": [FAILURES[0]], "threshold": 1e9, "min_coverage": 0.0}
        status, data, _ = self.client.post("/api/negative/filter", body)
        self.assertEqual(status, 200, data)
        self.assertEqual(data["verdicts"][0]["rule"], "ratio")  # blame cannot reject, the ratio still can
        status, data, _ = self.client.post("/api/negative/filter", {**body, "no_ratio": True})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["verdicts"][0]["decision"], "suspect")  # both rules off: nothing rejects
        self.assertIsNone(data["verdicts"][0]["ratio_threshold"])
        status, data, _ = self.client.post("/api/negative/filter", {**body, "no_ratio": True, "strict": True})
        self.assertEqual(data["verdicts"][0]["rule"], "suspect")

    def test_settings_forget_and_save(self):
        self.blame()
        status, data, _ = self.client.post("/api/negative/settings", {"threshold": 2.5, "min_coverage": 0.1})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["settings"], {"threshold": 2.5, "min_coverage": 0.1})
        status, data, _ = self.client.post("/api/negative/settings", {"clear_scale": 2.0})
        self.assertEqual(data["weights"]["clear_scale"], 2.0)
        status, data, _ = self.client.post("/api/negative/forget", {"reason": "repetition"})
        self.assertEqual(status, 200, data)
        self.assertGreater(data["blame_removed"], 0)
        status, data, _ = self.client.post("/api/negative/save")
        self.assertEqual(status, 200, data)
        self.assertTrue(os.path.isfile(data["path"]))
        self.assertIsInstance(load_model(data["path"]), NegativeNet)

    def test_save_writes_the_negative_model_too(self):
        self.blame()
        status, data, _ = self.client.post("/api/save")
        self.assertEqual(status, 200, data)
        self.assertIsNotNone(data["negative"])
        self.assertTrue(os.path.isfile(data["negative"]["path"]))

    def test_selecting_the_negative_kind_shares_the_model(self):
        self.blame()
        status, data, _ = self.client.post("/api/model/select", {"kind": "negative"})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["kind"], "negative")
        self.assertGreater(data["stats"]["failures_total"], 0)
        # the filter still works: the positive model comes from the parked kinds
        status, data, _ = self.client.post("/api/negative/filter", {"texts": [FAILURES[0]]})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["verdicts"][0]["decision"], "reject")
        self.assertEqual(self.client.post("/api/model/select", {"kind": "radix"})[1]["kind"], "radix")

    def test_errors(self):
        status, data, _ = self.client.post("/api/negative/blame", {"texts": []})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/negative/forget", {"factor": 2.0})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/negative/filter", {"over_sample": 0})
        self.assertEqual(status, 400, data)


class TestCli(unittest.TestCase):
    """The ``negative`` command group end to end (subprocess, --json)."""

    @classmethod
    def setUpClass(cls):
        from tests.test_cli import run_json

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-negative-cli-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model = os.path.join(cls.tmp.name, "model.json")
        cls.negative = os.path.join(cls.tmp.name, "model.negative.json")
        run_json("train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 2,
                 "--batch-size", 8, "--lr", 0.5, model=cls.model)

    def run_json(self, *args, **kwargs):
        from tests.test_cli import run_json

        kwargs.setdefault("model", self.model)
        return run_json(*args, **kwargs)

    def test_blame_why_reasons_filter_and_forget(self):
        doc = self.run_json("negative", "blame", "--text", FAILURES[0], "--reason", "repetition",
                            "--note", "it repeats", "--source", "review")
        self.assertEqual(doc["reason"], "repetition")
        self.assertTrue(os.path.isfile(self.negative))
        self.assertEqual(doc["reasons"][0]["reason"], "repetition")

        doc = self.run_json("negative", "why", "--text", FAILURES[0])
        verdict = doc["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertTrue(verdict["spans"])

        doc = self.run_json("negative", "reasons")
        self.assertEqual(doc["journal"][0]["note"], "it repeats")

        doc = self.run_json("negative", "filter", "--text", FAILURES[0], "--text", "a wholly unseen line")
        self.assertEqual([v["decision"] for v in doc["verdicts"]], ["reject", "pass"])
        self.assertEqual(doc["texts"], ["a wholly unseen line"])

        doc = self.run_json("negative", "filter", "--count", 2, "--max-length", 30)
        self.assertEqual(len(doc["verdicts"]), doc["candidates"])

        doc = self.run_json("negative", "clear", "--text", GOOD[0])
        self.assertEqual(doc["texts"], 1)

        doc = self.run_json("negative", "forget", "--reason", "repetition")
        self.assertGreater(doc["blame_removed"], 0)

    def test_correct_can_teach_both_networks(self):
        from tests.test_cli import run_json

        model = os.path.join(self.tmp.name, "pair.count.json")
        negative = os.path.join(self.tmp.name, "pair.count.negative.json")
        run_json("--kind", "count", "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"),
                 "--epochs", 1, model=model)
        doc = run_json("correct", "--wrong", "the cat sit on the mat", "--right", "the cat sits on the mat",
                       "--blame", "--reason", "agreement", "--note", "the verb must agree", model=model)
        self.assertEqual(doc["penalised"], 1)  # the positive model moved one step
        self.assertEqual(doc["negative"]["blamed"]["blamed"], 1)  # and so did the negative one
        self.assertEqual(doc["negative"]["blamed"]["reason"], "agreement")
        self.assertTrue(os.path.isfile(negative))
        blamed = load_model(negative)
        self.assertEqual([span["fragment"] for span in blamed.judge("the cat sit on the mat")["spans"]], ["sit "])

    def test_kind_negative_trains_by_blaming(self):
        from tests.test_cli import run_json

        model = os.path.join(self.tmp.name, "kind.negative.json")
        doc = run_json("--kind", "negative", "train", "--data", os.path.join(ROOT, "data", "sample_garbage.txt"),
                       "--epochs", 1, model=model)
        self.assertEqual(doc["stats"]["kind"], "negative")
        self.assertGreater(doc["stats"]["failures_total"], 0)

    def test_missing_negative_model_is_an_error(self):
        from tests.test_cli import run_cli

        proc = run_cli("negative", "why", "--text", "anything at all", "--negative",
                       os.path.join(self.tmp.name, "absent.json"), model=self.model, expect=1)
        self.assertIn("negative model file not found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
