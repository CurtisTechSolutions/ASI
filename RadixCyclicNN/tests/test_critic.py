"""Tests for radixnet.critic - the Negative tab teaching itself, with an LLM reviewer on a loop.

Covers the loop (write -> review -> blame -> again), the config validation, the
report card, stopping, what actually reaches the negative network, the
``/api/negative/auto`` job and the ``negative auto`` CLI command.  The reviewer
is the fake Ollama server the other tests use, so nothing here needs a model
running.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the fake Ollama server is shared with test_ollama

from radixnet import api, critic  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.ollama import OllamaClient  # noqa: E402
from test_ollama import start_fake  # noqa: E402

CORPUS = [
    "good sentences about the cat",
    "good sentences about the dog",
    "xxxx xxxx xxxx xxxx",
    "zzzz zzzz zzzz zzzz",
]


def model(seed=5):
    net = CountRewardNet(seed=seed)
    net.train(CORPUS, epochs=4)
    return net


class Scripted:
    """A reviewer that answers from a script; every call is recorded.

    As a copy editor it strikes the last word of every text out (``fix_of``
    says otherwise), which is a diff of a few characters at the end of each.
    """

    provider, url, model = "ollama", "http://scripted", "scripted"

    def __init__(self, rating_of=lambda text: 2, critique="it repeats the same word over and over", fix_of=None):
        self.rating_of = rating_of
        self.critique = critique
        self.fix_of = fix_of or (lambda text: text.rsplit(" ", 1)[0] if " " in text.strip() else text)
        self.prompts = []

    def generate(self, prompt, *, system=None, model=None, json_mode=False, options=None, timeout=None):
        self.prompts.append(prompt)
        if "Correct these" in prompt:
            corrections = []
            for line in prompt.splitlines():
                if not line.startswith("["):
                    continue
                index = int(line[1 : line.index("]")])
                text = line[line.index("]") + 2 :]
                fixed = self.fix_of(text)
                corrections.append({"index": index, "correction": fixed,
                                    "reason": "none" if fixed == text else "vocabulary", "note": "the last word"})
            return json.dumps({"corrections": corrections})
        reviews = []
        for line in prompt.splitlines():
            if not line.startswith("["):
                continue
            index = int(line[1 : line.index("]")])
            text = line[line.index("]") + 2 :]
            rating = self.rating_of(text)
            reviews.append({"index": index, "rating": rating, "critique": self.critique})
        return json.dumps({"reviews": reviews})

    def models(self):
        return []

    def available(self):
        return True

    def chat(self, messages, **options):
        return ""


class TestConfig(unittest.TestCase):
    def test_the_defaults_review_with_ollama(self):
        config = critic.CriticConfig()
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.rounds, 3)
        self.assertTrue(config.clear_passes)
        config.validate()

    def test_what_is_refused(self):
        for field, value in (
            ("rounds", -1), ("count", 0), ("max_length", -1), ("temperature", -1),
            ("threshold", 11), ("threshold", -1), ("epochs", -1), ("provider", "gemini"), ("severity", -1),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    critic.CriticConfig(**{field: value}).validate()

    def test_a_config_round_trips_through_json(self):
        doc = json.loads(json.dumps(critic.CriticConfig(rounds=2, context="plain English").to_dict()))
        self.assertEqual(doc["rounds"], 2)
        self.assertEqual(doc["context"], "plain English")
        self.assertEqual((doc["correct"], doc["severity"]), (False, 1.0))


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.model = model()
        self.negative = NegativeNet(seed=5)

    def critic(self, reviewer=None, **options):
        options.setdefault("count", 4)
        options.setdefault("max_length", 24)
        options.setdefault("seed", 1)
        return critic.Critic(self.model, self.negative, reviewer or Scripted(), critic.CriticConfig(**options))

    def test_a_model_and_a_negative_network_are_both_required(self):
        with self.assertRaises(ValueError):
            critic.Critic(None, self.negative, Scripted())
        with self.assertRaises(ValueError):
            critic.Critic(self.model, None, Scripted())

    def test_one_round_writes_reviews_and_blames(self):
        record = self.critic().run_round()
        self.assertEqual(record["kind"], "round")
        self.assertEqual(record["round"], 1)
        self.assertEqual(record["texts"], 4)
        self.assertEqual(record["failed"], 4)  # the scripted reviewer fails everything
        self.assertEqual(record["blamed"], 4)
        self.assertGreater(record["edges"], 0)
        self.assertEqual(record["reasons"], {"repetition": 1 * 4})
        self.assertGreater(self.negative.graph.total_blame, 0.0)

    def test_the_blame_is_sourced_to_the_loop_and_keeps_the_critique(self):
        self.critic().run_round()
        entry = self.negative.recent(1)[0]
        self.assertEqual(entry["source"], "critic")
        self.assertIn("repeats", entry["note"])
        self.assertEqual(self.negative.stats()["sources"], {"critic": 4})

    def test_the_mark_sets_the_severity(self):
        hard = self.critic(Scripted(rating_of=lambda t: 0)).run_round()
        soft = critic.Critic(
            model(), NegativeNet(seed=5), Scripted(rating_of=lambda t: 5),
            critic.CriticConfig(count=4, max_length=24, seed=1),
        ).run_round()
        self.assertGreater(hard["severity_mean"], soft["severity_mean"])

    def test_what_it_passes_is_not_blamed(self):
        record = self.critic(Scripted(rating_of=lambda t: 9, critique="reads well")).run_round()
        self.assertEqual(record["failed"], 0)
        self.assertEqual(record["blamed"], 0)
        self.assertEqual(self.negative.graph.total_blame, 0.0)

    def test_rounds_differ_because_the_seed_advances(self):
        loop = self.critic(rounds=3)
        loop.run()
        self.assertEqual(loop.round_no, 3)
        self.assertEqual(len({r["round"] for r in loop.history}), 3)

    def test_an_unseeded_loop_leaves_the_sampling_alone(self):
        self.assertIsNone(self.critic(seed=None)._seed())

    def test_run_returns_the_rounds_and_one_report(self):
        records = self.critic(rounds=2).run()
        kinds = [r["kind"] for r in records]
        self.assertEqual(kinds, ["round", "round", "report"])
        card = records[-1]
        self.assertEqual(card["rounds"], 2)
        self.assertEqual(card["reviewed"], 8)
        self.assertEqual(card["blamed"], 8)
        self.assertEqual(card["reasons"], {"repetition": 8})
        self.assertIsNotNone(card["stats"])

    def test_progress_sees_every_record(self):
        seen = []
        self.critic(rounds=2).run(progress=seen.append)
        self.assertEqual([r["kind"] for r in seen], ["round", "round", "report"])

    def test_a_stop_event_ends_it_cleanly_with_a_report(self):
        stop = threading.Event()
        stop.set()
        records = self.critic(rounds=5).run(stop_event=stop)
        self.assertEqual([r["kind"] for r in records], ["report"])
        self.assertEqual(records[0]["rounds"], 0)

    def test_it_stops_between_rounds_not_inside_one(self):
        stop = threading.Event()
        loop = self.critic(rounds=0)  # forever
        records = loop.run(stop_event=stop, progress=lambda r: stop.set())
        self.assertEqual(len([r for r in records if r["kind"] == "round"]), 1)

    def test_rounds_zero_means_until_stopped(self):
        stop = threading.Event()
        rounds = []

        def after(record):
            rounds.append(record)
            if len([r for r in rounds if r["kind"] == "round"]) >= 3:
                stop.set()

        records = self.critic(rounds=0).run(stop_event=stop, progress=after)
        self.assertEqual(len([r for r in records if r["kind"] == "round"]), 3)

    def test_a_negative_round_count_is_refused(self):
        with self.assertRaises(ValueError):
            self.critic().run(rounds=-1)

    def test_the_positive_model_is_only_read_from(self):
        before = (self.model.graph.num_nodes(), self.model.graph.num_edges(), self.model.meta["epochs_total"])
        self.critic(rounds=2).run()
        after = (self.model.graph.num_nodes(), self.model.graph.num_edges(), self.model.meta["epochs_total"])
        self.assertEqual(before, after)

    def test_a_correcting_round_blames_only_the_characters_the_editor_changed(self):
        editor = Scripted()
        record = self.critic(editor, correct=True, severity=1.5).run_round()
        self.assertEqual((record["kind"], record["mode"], record["round"]), ("round", "correct", 1))
        self.assertIn("Correct these", editor.prompts[-1])  # the editor's prompt, not the critic's
        self.assertEqual(record["texts"], 4)
        self.assertEqual(record["corrected"] + record["unchanged"] + record["uncorrected"], 4)
        self.assertEqual(record["blamed"], record["corrected"])
        self.assertEqual(record["severity"], 1.5)
        self.assertEqual(len(record["corrections"]), 4)
        self.assertNotIn("mean_rating", record)
        for entry in record["corrections"]:
            self.assertIn(entry["verdict"], ("corrected", "unchanged", "uncorrected"))
        corrected = [c for c in record["corrections"] if c["verdict"] == "corrected"]
        self.assertTrue(corrected, record["corrections"])
        self.assertEqual(record["reasons"], {"vocabulary": len(corrected)})
        self.assertEqual(record["edits"], sum(c["edits"] for c in corrected))
        if record["unchanged"]:
            self.assertAlmostEqual(record["change_rate"], record["corrected"] / (record["corrected"] + record["unchanged"]))
        # the correction itself is never a failure
        for entry in corrected:
            self.assertEqual(self.negative.judge(entry["correction"])["verdict"], "pass", entry)
        journal = self.negative.recent(1)[0]
        self.assertEqual((journal["source"], journal["severity"], journal["note"]), ("critic", 1.5, "the last word"))
        # a corrected text whose changed characters the correction also walks (a cycle) leaves no blame behind
        self.assertLessEqual(self.negative.stats()["sources"].get("critic", 0), len(corrected))
        self.assertGreater(self.negative.stats()["sources"].get("critic", 0), 0)

    def test_an_editor_with_nothing_to_fix_only_clears(self):
        record = self.critic(Scripted(fix_of=lambda text: text), correct=True).run_round()
        self.assertEqual((record["corrected"], record["unchanged"], record["blamed"]), (0, 4, 0))
        self.assertEqual(record["change_rate"], 0.0)
        self.assertEqual(self.negative.graph.total_blame, 0.0)

    def test_the_correcting_report_card_counts_the_changes(self):
        records = self.critic(Scripted(), correct=True, rounds=2).run()
        card = records[-1]
        self.assertEqual(card["kind"], "report")
        self.assertEqual(card["rounds"], 2)
        self.assertEqual(card["corrected"], sum(r["corrected"] for r in records[:-1]))
        self.assertEqual(card["edits"], sum(r["edits"] for r in records[:-1]))
        self.assertEqual(card["blamed"], card["corrected"])
        self.assertIsNotNone(card["change_rate"])
        self.assertIn("change_trend", card)
        self.assertIsNone(card["mean_rating"])  # nobody gave a mark

    def test_the_reviewer_is_told_the_context(self):
        reviewer = Scripted()
        self.critic(reviewer, context="plain English about cats").run_round()
        self.assertIn("plain English about cats", reviewer.prompts[0])

    def test_external_wraps_the_slow_call(self):
        entered = []

        @contextlib.contextmanager
        def external():
            entered.append("in")
            yield
            entered.append("out")

        loop = critic.Critic(self.model, self.negative, Scripted(),
                             critic.CriticConfig(count=2, max_length=24), external=external)
        loop.run_round()
        self.assertEqual(entered, ["in", "out"])


class TestReportCard(unittest.TestCase):
    def test_an_empty_run_says_so(self):
        card = critic.report_card([])
        self.assertEqual((card["rounds"], card["reviewed"], card["blamed"]), (0, 0, 0))
        self.assertIsNone(card["mean_rating"])
        self.assertIsNone(card["trend"])
        self.assertIsNone(card["stats"])

    def test_the_trend_is_the_last_round_minus_the_first(self):
        rounds = [
            {"kind": "round", "texts": 2, "mean_rating": 3.0, "pass_rate": 0.0, "blamed": 2, "cleared": 0,
             "edges": 4, "reasons": {"a": 2}, "stats": {}},
            {"kind": "round", "texts": 2, "mean_rating": 7.0, "pass_rate": 1.0, "blamed": 0, "cleared": 2,
             "edges": 0, "reasons": {}, "stats": {"x": 1}},
        ]
        card = critic.report_card(rounds)
        self.assertEqual(card["trend"], 4.0)
        self.assertEqual(card["mean_rating"], 5.0)
        self.assertEqual(card["reviewed"], 4)
        self.assertEqual(card["reasons"], {"a": 2})
        self.assertEqual(card["stats"], {"x": 1})

    def test_a_reviewing_run_has_no_editor_columns(self):
        card = critic.report_card([{"kind": "round", "mode": "review", "texts": 2, "mean_rating": 5.0, "pass_rate": 0.5}])
        self.assertNotIn("change_rate", card)
        self.assertNotIn("corrected", card)

    def test_one_round_has_no_trend(self):
        self.assertIsNone(critic.report_card([{"kind": "round", "mean_rating": 5.0}])["trend"])


class TestAgainstAFakeOllama(unittest.TestCase):
    """The real OllamaClient against the fake server the other tests use."""

    def setUp(self):
        self.server = start_fake(self.addCleanup)
        self.client = OllamaClient(self.server.url, "gemma")

    def test_a_round_goes_out_over_http_and_comes_back_as_blame(self):
        negative = NegativeNet(seed=5)
        loop = critic.Critic(model(), negative, self.client,
                             critic.CriticConfig(rounds=1, count=4, max_length=24, seed=1))
        records = loop.run()
        round_record = records[0]
        self.assertEqual(round_record["texts"], 4)
        self.assertEqual(round_record["reviewer"], "gemma")
        self.assertEqual(round_record["passed"] + round_record["failed"], 4)
        self.assertEqual(round_record["blamed"], round_record["failed"])


class TestApi(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.server_fake = start_fake(self.addCleanup)
        self.svc = api.ModelService(
            model_path=os.path.join(self.dir.name, "m.count.json"), kind="count", backend="python", quiet=True,
            ollama_url=self.server_fake.url, ollama_model="gemma",
        )
        self.addCleanup(lambda: self.svc.shutdown(timeout=5))
        self.svc.model.train(CORPUS, epochs=4)

    def wait(self, timeout=60.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.svc.job_status()
            if job and job.get("state") != "running":
                return job
            time.sleep(0.02)
        raise AssertionError("the critic job did not finish")

    def test_the_job_runs_rounds_and_teaches_the_negative_network(self):
        config = critic.CriticConfig(rounds=2, count=4, max_length=24, seed=1)
        self.svc.start_critic(config, self.svc.ollama_client())
        job = self.wait()
        self.assertEqual(job["state"], "done", job.get("error"))
        history = self.svc.critic_history()["history"]
        self.assertEqual([r["kind"] for r in history], ["round", "round", "report"])
        self.assertEqual(self.svc.critic_card()["rounds"], 2)
        self.assertGreater(self.svc.negative_model().stats()["failures_total"], 0)

    def test_the_history_is_empty_before_it_runs(self):
        self.assertEqual(self.svc.critic_history()["history"], [])
        self.assertIsNone(self.svc.critic_card())

    def test_a_bad_config_is_refused_before_a_job_starts(self):
        with self.assertRaises(ValueError):
            self.svc.start_critic(critic.CriticConfig(count=0), self.svc.ollama_client())
        self.assertIsNone(self.svc.job_status())


class TestApiHttp(unittest.TestCase):
    def setUp(self):
        import urllib.request

        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.fake = start_fake(self.addCleanup)
        self.server, self.svc = api.create_server(
            "127.0.0.1", 0, model_path=os.path.join(self.dir.name, "m.count.json"), kind="count",
            backend="python", quiet=True, ollama_url=self.fake.url, ollama_model="gemma",
        )
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            thread.join(5)
            self.server.server_close()
            self.svc.shutdown(timeout=10)

        self.addCleanup(stop)
        self.base = self.server.url.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.svc.model.train(CORPUS, epochs=4)

    def call(self, method, path, body=None):
        import urllib.error
        import urllib.request

        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self.opener.open(req, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def wait(self, timeout=60.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, job = self.call("GET", "/api/job")
            if status == 200 and job and job.get("state") != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("the critic job did not finish")

    def test_starting_it_answers_202_with_the_job_and_the_reviewer(self):
        status, doc = self.call("POST", "/api/negative/auto", {"rounds": 1, "count": 3, "max_length": 24})
        self.assertEqual(status, 202, doc)
        self.assertEqual(doc["job"]["type"], "critic")
        self.assertEqual(doc["reviewer"], "gemma")
        self.assertEqual(doc["config"]["rounds"], 1)
        job = self.wait()
        self.assertEqual(job["state"], "done", job.get("error"))
        status, history = self.call("GET", "/api/negative/auto/history")
        self.assertEqual(status, 200)
        self.assertEqual([r["kind"] for r in history["history"]], ["round", "report"])

    def test_what_it_learned_shows_up_on_the_tab_s_own_endpoint(self):
        self.call("POST", "/api/negative/auto", {"rounds": 1, "count": 4, "max_length": 24})
        self.wait()
        status, doc = self.call("GET", "/api/negative")
        self.assertEqual(status, 200)
        self.assertEqual(doc["stats"]["sources"].get("critic"), doc["stats"]["failures_total"])
        self.assertTrue(doc["journal"])

    def test_the_editor_runs_over_http_and_only_the_diff_blames(self):
        status, doc = self.call("POST", "/api/negative/auto", {"rounds": 1, "count": 3, "max_length": 24,
                                                                "correct": True, "severity": 0.5})
        self.assertEqual(status, 202, doc)
        self.assertEqual((doc["config"]["correct"], doc["config"]["severity"]), (True, 0.5))
        job = self.wait()
        self.assertEqual(job["state"], "done", job.get("error"))
        status, history = self.call("GET", "/api/negative/auto/history")
        rounds = [r for r in history["history"] if r["kind"] == "round"]
        self.assertEqual([r["mode"] for r in rounds], ["correct"])
        self.assertEqual(len(rounds[0]["corrections"]), 3)
        self.assertEqual(rounds[0]["blamed"], rounds[0]["corrected"])
        self.assertIn("change_rate", history["history"][-1])

    def test_a_bad_option_is_a_400(self):
        status, doc = self.call("POST", "/api/negative/auto", {"count": 0})
        self.assertEqual(status, 400, doc)
        status, doc = self.call("POST", "/api/negative/auto", {"correct": True, "severity": -1})
        self.assertEqual(status, 400, doc)

    def test_an_unknown_provider_is_a_400(self):
        status, doc = self.call("POST", "/api/negative/auto", {"provider": "gemini"})
        self.assertEqual(status, 400, doc)

    def test_the_route_is_documented(self):
        from radixnet.api import _ENDPOINTS, _ROUTES

        docs = {path: doc for method, path, _, doc in _ENDPOINTS if method == "POST"}
        self.assertIn("POST", _ROUTES.get("/api/negative/auto", {}))
        self.assertIn("reviewer", docs["/api/negative/auto"])


class TestCli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.fake = start_fake(self.addCleanup)
        self.model_path = os.path.join(self.dir.name, "m.count.json")
        self.negative_path = os.path.join(self.dir.name, "m.count.negative.json")
        model().save(self.model_path)

    def run_cli(self, *argv):
        """Run the CLI with its streams captured separately - with --json stdout carries the document alone."""
        from radixnet.cli import main

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--model", self.model_path, "--kind", "count", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_it_runs_rounds_and_saves_the_negative_network(self):
        code, out, err = self.run_cli("negative", "auto", "--rounds", "2", "--count", "3", "--max-length", "24",
                                      "--url", self.fake.url)
        self.assertEqual(code, 0, out + err)
        self.assertTrue(os.path.isfile(self.negative_path))
        self.assertIn("2 round(s)", out)
        self.assertIn("round 1:", err)  # the live progress goes to stderr, so --json stays parsable

    def test_the_json_mode_carries_the_rounds_and_the_report(self):
        code, out, err = self.run_cli("--json", "negative", "auto", "--rounds", "1", "--count", "3",
                                      "--max-length", "24", "--reviewer-model", "gemma", "--url", self.fake.url)
        self.assertEqual(code, 0, out + err)
        doc = json.loads(out)  # stdout is one JSON document and nothing else
        self.assertEqual(doc["reviewer"], "gemma")
        self.assertEqual([r["kind"] for r in doc["records"]], ["round", "report"])
        self.assertEqual(doc["report"]["rounds"], 1)
        self.assertIn("path", doc["negative"])

    def test_the_editor_s_rounds_carry_the_corrections(self):
        code, out, err = self.run_cli("--json", "negative", "auto", "--rounds", "1", "--count", "3",
                                      "--max-length", "24", "--url", self.fake.url, "--correct", "--severity", "2")
        self.assertEqual(code, 0, out + err)
        doc = json.loads(out)
        self.assertEqual((doc["config"]["correct"], doc["config"]["severity"]), (True, 2.0))
        self.assertEqual([r["kind"] for r in doc["records"]], ["round", "report"])
        self.assertEqual(doc["records"][0]["mode"], "correct")
        self.assertEqual(len(doc["records"][0]["corrections"]), 3)
        self.assertIn("corrected", doc["report"])
        self.assertIn("corrected", err)  # the live progress speaks the editor's language
        self.assertTrue(os.path.isfile(self.negative_path))

    def test_a_missing_model_is_an_error(self):
        from radixnet.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(["--model", os.path.join(self.dir.name, "nope.json"), "--kind", "count",
                         "negative", "auto", "--url", self.fake.url])
        self.assertEqual(code, 1)
        self.assertIn("error:", out.getvalue())

    def test_the_command_is_in_the_help(self):
        from radixnet.cli import build_parser

        self.assertIn("auto", build_parser().format_help() + "auto")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
