"""Tests for radixnet.chat - the model in conversation with an LLM that marks it.

Covers one conversation end to end (the partner opens, the model replies, the
judge marks every reply and the conversation as a whole), the config, what
reaches the negative network and the positive one, the guard vetoing a reply
before it is spoken, stalling when the model has nothing to say, the report
card, stopping between conversations, the HTTP job and the CLI command.  The
partner and the judge are a fake server, so nothing here needs a model running.
"""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from radixnet import api, chat  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.ollama import OllamaClient  # noqa: E402

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a bird flew over the hill",
]
# lines that end on words the corpus knows, so the model has something to continue
PARTNER_LINES = ["tell me about the cat", "and what about the dog", "now tell me about a bird", "say more"]
EXCHANGE = re.compile(r"^\[(\d+)\] Partner: (.*)$")


class _Handler(BaseHTTPRequestHandler):
    server: "FakePartner"

    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200, {"models": [{"name": "fake:latest"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
            return
        self._json(200, {"response": self._answer(body)})

    def _answer(self, body):
        system, prompt = body.get("system", ""), body.get("prompt", "")
        if "marking a conversation" in system:  # the judge
            if self.server.judge_response is not None:
                return self.server.judge_response
            reviews = []
            lines = prompt.splitlines()
            for i, line in enumerate(lines):
                match = EXCHANGE.match(line)
                if match is None:
                    continue
                reply = lines[i + 1].split("Model:", 1)[-1].strip() if i + 1 < len(lines) else ""
                good = self.server.passes(reply)
                reviews.append({"index": int(match.group(1)), "rating": 9 if good else 2,
                                "critique": "it follows on" if good else "it repeats the same words"})
            self.server.marked += 1
            return json.dumps({"reviews": reviews,
                               "overall": {"rating": 7, "critique": "it kept up, mostly"}})
        if "conversation with a very small" in system:  # the partner
            self.server.lines += 1
            return self.server.line_for(self.server.lines - 1)
        return "unexpected prompt"


class FakePartner(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests = []
        self.fail_with = None
        self.judge_response = None
        self.lines = 0
        self.marked = 0
        self.passes = lambda reply: "sat" in reply or "flew" in reply
        self.line_for = lambda i: PARTNER_LINES[i % len(PARTNER_LINES)]

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def start_partner(add_cleanup):
    fake = FakePartner()
    threading.Thread(target=fake.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    add_cleanup(fake.server_close)
    add_cleanup(fake.shutdown)
    return fake


def model(seed=5):
    net = CountRewardNet(seed=seed)
    net.train(CORPUS, epochs=4)
    return net


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        chat.ChatConfig().validate()
        self.assertEqual(chat.DEFAULT_SPEAKERS, ("Partner", "Model"))

    def test_validation(self):
        for change in (
            {"conversations": -1}, {"turns": 0}, {"max_length": 0}, {"mode": "dijkstra"}, {"k": 0},
            {"temperature": -1}, {"threshold": 11}, {"neg_epochs": -1}, {"batch_size": 0},
            {"strength": -1}, {"provider": "nobody"}, {"context": -1}, {"neg_lr": -1},
        ):
            with self.assertRaises(ValueError, msg=change):
                chat.ChatConfig(**change).validate()


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_partner(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "fake:latest", timeout=20)
        self.model = model()

    def loop(self, negative=None, **options):
        options.setdefault("turns", 2)
        options.setdefault("learn", False)
        return chat.Chat(self.model, self.client, chat.ChatConfig(**options), negative=negative)

    def test_the_partner_opens_and_the_model_replies(self):
        held = self.loop().converse()
        speakers = [speaker for speaker, _text in held["transcript"]]
        self.assertEqual(speakers[0], "Partner")
        self.assertEqual(speakers[1], "Model")
        self.assertEqual(len(held["exchanges"]), 2)
        self.assertEqual(held["transcript"][0][1], PARTNER_LINES[0])
        for said, reply in held["exchanges"]:
            self.assertTrue(said.strip() and reply.strip())
        self.assertFalse(held["stalled"])

    def test_a_given_opening_is_spoken_as_it_is(self):
        held = self.loop(opening="the cat sat on the mat", turns=1).converse()
        self.assertEqual(held["transcript"][0], ("Partner", "the cat sat on the mat"))
        self.assertEqual(self.fake.lines, 0)  # the partner was never asked to open

    def test_the_model_replies_by_continuing_the_line(self):
        held = self.loop(opening="tell me about the cat", turns=1).converse()
        turn = held["turns"][0]
        self.assertTrue(turn.context)  # it picked up the tail of the line
        self.assertTrue(turn.text.startswith(turn.context))
        self.assertIn(turn.context, "tell me about the cat")

    def test_a_conversation_is_marked_reply_by_reply(self):
        record = self.loop().run_conversation()
        self.assertEqual(record["kind"], "conversation")
        self.assertEqual(len(record["reviews"]), record["exchanges"])
        self.assertEqual(record["overall_rating"], 7)
        self.assertEqual(record["overall_critique"], "it kept up, mostly")
        self.assertEqual(record["passed"] + record["failed"], record["exchanges"])
        self.assertIsNotNone(record["mean_rating"])
        for review in record["reviews"]:  # every reply was marked: none came back unrated
            self.assertIn(review["verdict"], ("pass", "fail"))
            self.assertIsNotNone(review["rating"])
            self.assertTrue(review["said"])  # and each carries the line it was answering
        self.assertEqual(self.fake.marked, 1)  # one judgement per conversation, not one per line

    def test_a_stalled_conversation_says_so(self):
        loop = self.loop(opening="nothing here matches the corpus at all", turns=2)
        loop.model = CountRewardNet(seed=1)  # untrained: it has nothing to say at all
        held = loop.converse()
        self.assertEqual(held["exchanges"], [])
        self.assertTrue(held["stalled"])

    def test_the_failures_blame_the_negative_network(self):
        negative = NegativeNet(seed=2)
        loop = self.loop(negative=negative, turns=3)
        self.fake.passes = lambda reply: False  # every reply is a failure
        record = loop.run_conversation()
        self.assertGreater(record["blamed"], 0)
        self.assertGreater(negative.graph.total_blame, 0)
        self.assertIn("repetition", record["reasons"])  # "it repeats the same words"
        self.assertEqual(negative.recent(1)[0]["source"], "chat")

    def test_the_passes_clear_blame(self):
        negative = NegativeNet(seed=2)
        negative.blame(["the cat sat on the mat"], reason="gibberish")
        before = negative.graph.total_clear
        loop = self.loop(negative=negative, turns=2)
        self.fake.passes = lambda reply: True
        loop.run_conversation()
        self.assertGreater(negative.graph.total_clear, before)

    def test_the_guard_vetoes_a_reply_before_it_is_spoken(self):
        negative = NegativeNet(seed=2)
        negative.blame(CORPUS, reason="gibberish")  # everything it can say is known-bad
        loop = self.loop(negative=negative, turns=2, guard=True)
        loop.pair().config.threshold = 0.0
        loop.pair().config.min_coverage = 0.0
        held = loop.converse()
        self.assertGreater(held["vetoed"], 0)

    def test_learning_trains_the_positive_model(self):
        loop = self.loop(learn=True, turns=2, neg_epochs=1, pos_epochs=1)
        self.fake.passes = lambda reply: "sat" in reply
        before = self.model.meta["twonrl_runs"]
        record = loop.run_conversation()
        self.assertIn(record["action"], ("2nrl", "reward", "punish"))
        if record["action"] == "2nrl":
            self.assertGreater(self.model.meta["twonrl_runs"], before)
        self.assertGreater(record["good"], 0)  # the partner's lines are correct data

    def test_a_reply_it_could_only_repeat_is_punished_whatever_the_judge_said(self):
        loop = self.loop(learn=True, turns=1, teach_partner=False, neg_epochs=1, pos_epochs=1)
        self.fake.passes = lambda reply: True  # the judge is happy with everything
        held = {"transcript": [("Partner", "tell me about the cat")], "turns": [],
                "repeats": ["the cat sat on the mat"]}
        review = {"good": ["the cat sat on the mat"], "bad": []}
        learned = loop._learn(review, held)
        # a duplicate is never rewarded, however well it was marked
        self.assertEqual((learned["action"], learned["bad"], learned["good"], learned["repeats"]),
                         ("punish", 1, 0, 1))

    def test_the_partners_lines_can_be_kept_out_of_the_positive_phase(self):
        loop = self.loop(learn=True, turns=1, teach_partner=False, neg_epochs=1, pos_epochs=1)
        self.fake.passes = lambda reply: False
        record = loop.run_conversation()
        self.assertEqual(record["good"], 0)
        self.assertEqual(record["action"], "punish")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_partner(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "fake:latest", timeout=20)
        self.loop = chat.Chat(model(), self.client, chat.ChatConfig(conversations=2, turns=2, learn=False))

    def test_the_run_holds_every_conversation_and_reports(self):
        seen = []
        records = self.loop.run(progress=seen.append)
        held = [r for r in records if r["kind"] == "conversation"]
        card = records[-1]
        self.assertEqual(len(held), 2)
        self.assertEqual(card["kind"], "report")
        self.assertEqual(card["conversations"], 2)
        self.assertEqual(card["exchanges"], sum(r["exchanges"] for r in held))
        self.assertIsNotNone(card["mean_rating"])
        self.assertEqual(card["overall_rating"], 7)
        self.assertEqual([r["conversation"] for r in held], [1, 2])
        self.assertTrue(any(r["kind"] == "exchange" for r in seen))  # the turns stream as they happen

    def test_stopping_between_conversations(self):
        stop = threading.Event()
        stop.set()
        records = self.loop.run(stop_event=stop)
        self.assertEqual([r["kind"] for r in records], ["report"])
        self.assertEqual(records[0]["conversations"], 0)

    def test_report_card_of_nothing(self):
        card = chat.report_card([])
        self.assertEqual((card["conversations"], card["mean_rating"], card["trend"]), (0, None, None))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-chat-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, model_path=os.path.join(cls.tmp.name, "model.json"), kind="count",
        )

    def setUp(self):
        self.fake = start_partner(self.addCleanup)
        self.service.ollama_url = self.fake.url
        self.service.model.train(CORPUS, epochs=4)

    def test_start_and_history(self):
        from test_api import wait_for_job

        status, data, _ = self.client.post("/api/chat/start", {
            "conversations": 1, "turns": 2, "model": "fake:latest", "url": self.fake.url,
            "topic": "animals", "learn": False,
        })
        self.assertEqual(status, 202, data)
        self.assertEqual(data["job"]["type"], "chat")
        self.assertEqual(data["config"]["topic"], "animals")
        job = wait_for_job(self.client)
        self.assertEqual(job["state"], "done", job)
        status, data, _ = self.client.get("/api/chat/history")
        self.assertEqual(status, 200, data)
        kinds = [r["kind"] for r in data["history"]]
        self.assertIn("conversation", kinds)
        self.assertIn("report", kinds)
        held = next(r for r in data["history"] if r["kind"] == "conversation")
        self.assertTrue(held["transcript"])
        self.assertEqual(held["transcript"][0]["speaker"], "Partner")

    def test_bad_requests(self):
        status, data, _ = self.client.post("/api/chat/start", {"turns": 0})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/chat/start", {"provider": "nobody"})
        self.assertEqual(status, 400, data)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_partner(self.addCleanup)
        self.tmp = tempfile.TemporaryDirectory(prefix="radixnet-chat-cli-")
        self.addCleanup(self.tmp.cleanup)
        self.model = os.path.join(self.tmp.name, "model.count.json")
        net = model()
        net.save(self.model)

    def run_cli(self, *args):
        from radixnet.cli import main

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--model", self.model, "--kind", "count", "--json", "chat", "--url", self.fake.url,
                         "--partner-model", "fake:latest", *[str(a) for a in args]])
        self.assertEqual(code, 0, err.getvalue())
        return json.loads(out.getvalue())

    def test_chat_command(self):
        doc = self.run_cli("--conversations", 1, "--turns", 2, "--topic", "animals",
                           "--neg-epochs", 1, "--pos-epochs", 1)
        self.assertEqual(doc["config"]["topic"], "animals")
        self.assertEqual(doc["report"]["conversations"], 1)
        self.assertTrue(doc["records"])
        held = next(r for r in doc["records"] if r["kind"] == "conversation")
        self.assertEqual(held["transcript"][0]["speaker"], "Partner")
        self.assertEqual(doc["speakers"], ["Partner", "Model"])
        self.assertTrue(os.path.isfile(doc["saved"]))  # it learned, so the model was written back
        self.assertTrue(os.path.isfile(doc["negative"]["path"]))

    def test_marking_without_learning_leaves_the_model_alone(self):
        doc = self.run_cli("--conversations", 1, "--turns", 1, "--no-learn", "--no-blame")
        self.assertIsNone(doc["saved"])
        self.assertIsNotNone(doc["report"]["mean_rating"])


if __name__ == "__main__":
    unittest.main()
