"""Tests for ``radixnet.agent``: criteria, mediation, judging, teaching, failure-first 2NRL and exploring.

A fake Ollama (standard library ``http.server``) plays all four roles and can
be switched between answering with Ollama's native ``tool_calls`` and with
plain JSON, so both mediation paths are covered; the fake website of
``test_tools`` stands in for the internet.  The network itself is real: an
untrained :class:`~radixnet.model.RadixNet` that writes noise, which is exactly
the case the loop is built for.
"""

import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import RadixNet, blame  # noqa: E402
from radixnet.agent import (  # noqa: E402
    AgentConfig,
    AgentTrainer,
    Attempt,
    Failure,
    Task,
    Verdict,
    _decide,
    judge_attempt,
    load_tasks,
    mediate_call,
    parse_task_file,
    parse_tasks,
    propose_task,
    teach_task,
    write_criteria,
)
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.ollama import OllamaClient, OllamaError  # noqa: E402
from radixnet.tools import ToolBox, default_toolbox  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_tools import local_web, start_site
except ImportError:
    from tests.test_tools import local_web, start_site

FAST = {"max_steps": 2, "model_attempts": 1, "criteria_count": 2, "neg_epochs": 1, "pos_epochs": 1, "batch_size": 4}


# ---------------------------------------------------------------------------
# fake Ollama: criteria writer, mediator, judge and teacher
# ---------------------------------------------------------------------------


class _AgentHandler(BaseHTTPRequestHandler):
    server: "FakeAgentOllama"

    def log_message(self, *args):  # silence
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.requests.append(("GET", self.path, None))
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
        elif self.path == "/api/tags":
            self._json(200, {"models": [{"name": "fake:latest", "size": 1}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        self.server.requests.append(("POST", self.path, body))
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
        elif self.path == "/api/chat":
            self._json(200, {"model": body.get("model"), "message": self._chat(body)})
        elif self.path == "/api/generate":
            self._json(200, {"model": body.get("model"), "response": self._generate(body), "done": True})
        else:
            self._json(404, {"error": "not found"})

    # -- the roles -----------------------------------------------------------

    def _cats_url(self):
        return self.server.site + "/cats"

    def _chat(self, body):
        messages = body.get("messages") or [{}]
        system = messages[0].get("content", "")
        if "tool mediator" in system:
            self.server.mediations += 1
            if self.server.mediator_response is not None:
                return {"role": "assistant", "content": self.server.mediator_response}
            if self.server.native_tools and body.get("tools"):
                return {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": "web_fetch", "arguments": {"url": self._cats_url()}}}]}
            # a model without tool calling: the JSON is asked for in the content
            return {"role": "assistant",
                    "content": json.dumps({"tool": "web_fetch", "arguments": {"url": self._cats_url()}})}
        if "solve a task with the tools" in system:
            used = [m for m in messages if m.get("role") == "tool"]
            if not used and not self.server.teacher_answers_at_once:
                return {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": "web_fetch", "arguments": {"url": self._cats_url()}}}]}
            return {"role": "assistant", "content": self.server.answer}
        return {"role": "assistant", "content": self.server.answer}

    def _generate(self, body):
        system, prompt = body.get("system", ""), body.get("prompt", "")
        if "judge of an attempt" in system:  # matched first: the judge's prompt also says "acceptance criteria"
            if self.server.judge_response is not None:
                return self.server.judge_response
            answer = prompt.rsplit("Answer:", 1)[-1].lower()
            good = "four" in answer
            count = len(re.findall(r"^\d+\. ", prompt, re.M)) or 2
            return json.dumps({"met": [good] * count, "score": 9 if good else 1, "correct": good,
                               "critique": "fine" if good else "it never says how many"})
        if "acceptance criteria" in system:
            if self.server.criteria_response is not None:
                return self.server.criteria_response
            return json.dumps({"criteria": ["The answer says how many legs a cat has", "The answer names four"]})
        if "choosing what to look into next" in system:
            self.server.proposals += 1
            return json.dumps({"question": f"Question {self.server.proposals} about cats?", "seed": self._cats_url()})
        return "unexpected prompt"


class FakeAgentOllama(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, site_url):
        super().__init__(("127.0.0.1", 0), _AgentHandler)
        self.site = site_url
        self.requests = []
        self.fail_with = None
        self.native_tools = True
        self.teacher_answers_at_once = False
        self.criteria_response = None
        self.judge_response = None
        self.mediator_response = None
        self.answer = "A cat has four legs."
        self.mediations = 0
        self.proposals = 0

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def start_fake_ollama(add_cleanup, site_url):
    fake = FakeAgentOllama(site_url)
    thread = threading.Thread(target=fake.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    add_cleanup(fake.server_close)
    add_cleanup(fake.shutdown)
    return fake


class AgentCase(unittest.TestCase):
    """A fake site, a fake Ollama, a real (untrained) network and the real toolbox."""

    def setUp(self):
        self.site = start_site(self.addCleanup)
        self.fake = start_fake_ollama(self.addCleanup, self.site.url)
        self.client = OllamaClient(self.fake.url, "fake:latest", timeout=20)
        self.toolbox = default_toolbox(local_web(self.site))
        self.model = RadixNet(seed=1, backend="python")

    def trainer(self, **options):
        return AgentTrainer(self.model, self.client, self.toolbox, AgentConfig(**{**FAST, **options}))

    def task(self, prompt="How many legs does a cat have?", **kwargs):
        return Task("t1", prompt, **kwargs)


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


class TaskTests(unittest.TestCase):
    def test_strings_and_objects(self):
        tasks = parse_tasks(["first?", {"prompt": "second?", "criteria": ["a", "b"], "answer": "x", "seeds": ["u"]}])
        self.assertEqual([t.id for t in tasks], ["t1", "t2"])
        self.assertEqual(tasks[1].criteria, ("a", "b"))
        self.assertEqual(tasks[1].answer, "x")
        self.assertEqual(tasks[1].seeds, ("u",))
        self.assertEqual(tasks[0].to_dict()["prompt"], "first?")

    def test_alternative_keys_and_ids(self):
        tasks = parse_tasks([{"question": "q?", "id": "mine"}, {"goal": "g", "urls": "http://x"}])
        self.assertEqual(tasks[0].id, "mine")
        self.assertEqual(tasks[1].seeds, ("http://x",))

    def test_rejects_bad_input(self):
        for bad in ([""], [{}], [{"prompt": "x", "criteria": 5}], [{"prompt": "x", "answer": 5}], [5]):
            with self.assertRaises(ValueError):
                parse_tasks(bad)
        with self.assertRaises(ValueError):
            parse_tasks([{"prompt": "a", "id": "d"}, {"prompt": "b", "id": "d"}])

    def test_file_formats(self):
        self.assertEqual([t.prompt for t in parse_task_file("# a comment\nfirst?\n\nsecond?\n")], ["first?", "second?"])
        jsonl = '{"prompt": "a?"}\n{"prompt": "b?"}\n'
        self.assertEqual(len(parse_task_file(jsonl, ".jsonl")), 2)
        self.assertEqual(len(parse_task_file('{"tasks": ["a?", "b?"]}', ".json")), 2)
        self.assertEqual(len(parse_task_file('["a?"]', ".json")), 1)
        with self.assertRaises(ValueError):
            parse_task_file("", ".txt")
        with self.assertRaises(ValueError):
            parse_task_file('{"nope": 1}', ".json")

    def test_load_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tasks.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("only one?\n")
            self.assertEqual(load_tasks(path)[0].prompt, "only one?")


class ConfigTests(unittest.TestCase):
    def test_defaults_focus_on_failure(self):
        config = AgentConfig()
        config.validate()
        self.assertEqual(config.blatant_mode, "fail_invert")
        self.assertEqual(config.mediation, "repair")
        self.assertTrue(config.teach_on_failure)
        self.assertIn("phases", config.to_dict())

    def test_validation(self):
        for field, value in (
            ("phases", ()), ("phases", ("nope",)), ("mediation", "x"), ("rounds", 0), ("max_steps", 0),
            ("model_attempts", 0), ("max_length", 0), ("temperature", -1), ("criteria_count", 0),
            ("criteria_count", 99), ("twonrl_per", "x"), ("blatant_mode", "x"), ("blatant_margin", 0),
            ("blatant_boost", 0.5), ("pass_score", 0), ("pass_score", 11), ("batch_size", 0),
            ("neg_epochs", -1), ("neg_lr", -1), ("observation_chars", -1),
        ):
            with self.assertRaises(ValueError, msg=f"{field}={value}"):
                AgentConfig(**{field: value}).validate()


# ---------------------------------------------------------------------------
# the four LLM roles
# ---------------------------------------------------------------------------


class CriteriaTests(AgentCase):
    def test_written_before_anything_is_attempted(self):
        criteria = write_criteria(self.client, self.task(), count=2)
        self.assertEqual(criteria, ["The answer says how many legs a cat has", "The answer names four"])

    def test_a_task_that_carries_criteria_keeps_them(self):
        task = self.task(criteria=("mine",))
        self.assertEqual(write_criteria(self.client, task), ["mine"])
        self.assertEqual(self.fake.requests, [])  # the LLM is not asked at all

    def test_plain_lines_are_accepted(self):
        self.fake.criteria_response = "1. first thing\n2. second thing"
        self.assertEqual(write_criteria(self.client, self.task(), count=2), ["first thing", "second thing"])

    def test_an_unusable_answer_still_yields_a_criterion(self):
        self.fake.criteria_response = "   "
        criteria = write_criteria(self.client, self.task())
        self.assertEqual(len(criteria), 1)
        self.assertIn("How many legs", criteria[0])

    def test_the_count_is_capped(self):
        self.fake.criteria_response = json.dumps({"criteria": [f"c{i}" for i in range(20)]})
        self.assertEqual(len(write_criteria(self.client, self.task(), count=3)), 3)


class MediationTests(AgentCase):
    def test_native_tool_calls(self):
        call = mediate_call(self.client, "grbl ntoo", self.toolbox, self.task())
        self.assertTrue(call.ok)
        self.assertEqual(call.name, "web_fetch")
        self.assertEqual(call.arguments["url"], self.site.url + "/cats")
        self.assertTrue(any(body.get("tools") for _m, path, body in self.fake.requests if path == "/api/chat"))

    def test_json_fallback_for_a_model_without_tool_calling(self):
        self.fake.native_tools = False
        call = mediate_call(self.client, "", self.toolbox, self.task())
        self.assertTrue(call.ok)
        self.assertEqual(call.name, "web_fetch")

    def test_an_unusable_answer_falls_back_to_a_network_tool(self):
        self.fake.mediator_response = "I refuse"  # neither tool_calls nor JSON
        call = mediate_call(self.client, "", self.toolbox, self.task())
        self.assertEqual(call.name, "web_search")  # the first network tool
        self.assertEqual(call.arguments["query"], "How many legs does a cat have?")

    def test_a_tool_the_box_does_not_have_falls_back(self):
        box = default_toolbox(offline=True)  # only the calculator: the fake's web_fetch is unknown here
        call = mediate_call(self.client, "", box, self.task())
        self.assertEqual(call.name, "calculator")
        self.assertEqual(call.arguments, {"expression": "How many legs does a cat have?"})

    def test_an_unreachable_llm_raises(self):
        self.fake.fail_with = 500
        with self.assertRaises(OllamaError):
            mediate_call(self.client, "", self.toolbox, self.task())


class JudgeTests(AgentCase):
    def test_marks_each_criterion(self):
        judged = judge_attempt(self.client, self.task(), ["a", "b"], "transcript", "A cat has four legs.")
        self.assertEqual(judged["met"], [True, True])
        self.assertEqual(judged["score"], 9.0)
        self.assertTrue(judged["correct"])

    def test_a_wrong_answer_fails(self):
        judged = judge_attempt(self.client, self.task(), ["a", "b"], "transcript", "purple")
        self.assertEqual(judged["met"], [False, False])
        self.assertFalse(judged["correct"])
        self.assertIn("never says", judged["critique"])

    def test_an_unusable_answer_is_reported(self):
        self.fake.judge_response = "not json at all"
        judged = judge_attempt(self.client, self.task(), ["a"], "t", "x")
        self.assertIsNone(judged["correct"])
        self.assertIsNone(judged["score"])

    def test_marks_written_as_words(self):
        self.fake.judge_response = json.dumps({"met": ["yes", "no"], "score": 5, "correct": False})
        self.assertEqual(judge_attempt(self.client, self.task(), ["a", "b"], "t", "x")["met"], [True, False])

    def test_unreadable_marks_are_dropped_whole(self):
        # a criterion counted as met because its text is a non-empty string would misalign the marking
        self.fake.judge_response = json.dumps({"met": ["the answer says four", "maybe"], "score": 5})
        self.assertEqual(judge_attempt(self.client, self.task(), ["a", "b"], "t", "x")["met"], [])

    def test_decide_needs_an_answer(self):
        verdict = _decide({"met": [True], "score": 10, "correct": True}, ["a"], None, strict=True)
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.judged_by, "answer")

    def test_decide_strictness(self):
        judged = {"met": [True, False], "score": 8, "correct": True, "critique": ""}
        self.assertFalse(_decide(judged, ["a", "b"], "x", strict=True).correct)
        self.assertTrue(_decide(judged, ["a", "b"], "x", strict=False).correct)
        self.assertEqual(_decide(judged, ["a", "b"], "x", strict=True).issues, ("b",))

    def test_decide_without_a_verdict_falls_back_to_the_marks(self):
        judged = {"met": [True, True], "score": None, "correct": None, "critique": ""}
        self.assertTrue(_decide(judged, ["a", "b"], "x", strict=True).correct)


class TeachTests(AgentCase):
    def test_the_teacher_really_calls_the_tools(self):
        steps, answer = teach_task(self.client, self.task(), self.toolbox, max_steps=3)
        self.assertEqual([s.call.name for s in steps], ["web_fetch"])
        self.assertTrue(steps[0].result.ok)
        self.assertIn("A cat has four legs.", steps[0].result.output)
        self.assertEqual(steps[0].source, "teacher")
        self.assertEqual(answer, "A cat has four legs.")

    def test_an_answer_without_any_call(self):
        self.fake.teacher_answers_at_once = True
        steps, answer = teach_task(self.client, self.task(), self.toolbox, max_steps=2)
        self.assertEqual(steps, [])
        self.assertEqual(answer, "A cat has four legs.")


class ProposeTests(AgentCase):
    def test_turns_an_emission_into_a_question(self):
        task = propose_task(self.client, "cats legs", frontier=["http://x/a"], index=3)
        self.assertEqual(task.id, "x3")
        self.assertTrue(task.prompt.endswith("about cats?"))
        self.assertEqual(task.seeds, (self.site.url + "/cats",))

    def test_falls_back_when_the_llm_is_unusable(self):
        self.fake.criteria_response = None
        original = _AgentHandler._generate
        try:  # an LLM that answers with nothing usable at all
            _AgentHandler._generate = lambda self, body: "   "
            task = propose_task(self.client, "", frontier=["http://x/a"])
            self.assertIn("http://x/a", task.prompt)
            self.assertEqual(propose_task(self.client, "").seeds, ())
        finally:
            _AgentHandler._generate = original


# ---------------------------------------------------------------------------
# failure weighting: train on failures, then invert
# ---------------------------------------------------------------------------


class GapTests(AgentCase):
    def _attempt(self, verdict, answer="something"):
        return Attempt(0, "model", [], answer, "text", verdict)

    def test_a_correct_attempt_has_no_gap(self):
        trainer = self.trainer()
        self.assertEqual(trainer.gap_of(self._attempt(Verdict(True, 9.0, (True,))), ["a"]), 0.0)

    def test_no_answer_is_the_worst_failure(self):
        trainer = self.trainer()
        self.assertEqual(trainer.gap_of(self._attempt(Verdict(False), answer=None), ["a"]), 1.0)

    def test_the_gap_follows_the_score_and_the_unmet_criteria(self):
        trainer = self.trainer(pass_score=6.0)
        self.assertAlmostEqual(trainer.gap_of(self._attempt(Verdict(False, 0.0, (False, False))), ["a", "b"]), 1.0)
        self.assertAlmostEqual(trainer.gap_of(self._attempt(Verdict(False, 3.0, (True, False))), ["a", "b"]), 0.5)
        # a near miss still trains: every failure keeps a floor
        self.assertAlmostEqual(trainer.gap_of(self._attempt(Verdict(False, 9.0, (True, True))), ["a", "b"]), 0.1)


class LearnTests(AgentCase):
    def failure(self, text, gap):
        return Failure(text, gap, "t1", "model")

    def test_failures_are_weighted_then_the_model_is_inverted(self):
        trainer = self.trainer(blatant_margin=0.5, blatant_boost=4.0)
        result = trainer.learn(
            [self.failure("a failed transcript", 1.0), self.failure("a near miss", 0.1)],
            ["a correct transcript"],
        )
        self.assertEqual(result["action"], "2nrl")
        self.assertEqual(result["failures"], 2)
        self.assertEqual(result["blatant"], 1)  # only the gap of 1.0 is beyond the margin
        self.assertAlmostEqual(result["boost_max"], 3.0)  # 1 + 1.0 / 0.5
        self.assertAlmostEqual(result["boost_mean"], (3.0 + 1.2) / 2)
        self.assertEqual(self.model.meta["twonrl_runs"], 1)

    def test_the_worst_gap_of_a_repeated_transcript_wins(self):
        trainer = self.trainer()
        result = trainer.learn([self.failure("same", 0.2), self.failure("same", 0.9)], ["good"])
        self.assertEqual(result["failures"], 1)
        self.assertAlmostEqual(result["boost_max"], 1 + 0.9 / 0.5)

    def test_nothing_failed_rewards_without_inverting(self):
        trainer = self.trainer()
        before = self.model.graph.inverted
        result = trainer.learn([], ["a correct transcript"])
        self.assertEqual(result["action"], "reward")
        self.assertEqual(result["failures"], 0)
        self.assertEqual(self.model.graph.inverted, before)

    def test_only_failures_punishes_and_inverts(self):
        trainer = self.trainer()
        before = self.model.graph.inverted
        result = trainer.learn([self.failure("bad one", 1.0), self.failure("bad two", 0.2)], [])
        self.assertEqual(result["action"], "punish")
        self.assertNotEqual(self.model.graph.inverted, before)

    def test_nothing_at_all(self):
        self.assertIsNone(self.trainer().learn([], [])["action"])

    def test_a_transcript_that_is_both_is_not_trained_as_garbage(self):
        trainer = self.trainer()
        result = trainer.learn([self.failure("shared", 1.0)], ["shared"])
        self.assertEqual(result["failures"], 0)
        self.assertEqual(result["action"], "reward")

    def test_local_inversion_mode(self):
        trainer = self.trainer(blatant_mode="activation", blatant_margin=0.5)
        self.model.train(["a failed transcript"], epochs=1, lr=0.5, batch_size=4)
        result = trainer.learn([self.failure("a failed transcript", 1.0)], ["a correct transcript"])
        self.assertEqual(result["mode"], "activation")
        self.assertGreater(result["flipped"], 0)
        # the blatant failure left the garbage set, so the positive pass ran alone
        self.assertEqual(result["action"], "reward")

    def test_plain_2nrl_when_failure_weighting_is_off(self):
        trainer = self.trainer(blatant_mode="none")
        result = trainer.learn([self.failure("bad", 1.0)], ["good"])
        self.assertEqual(result["action"], "2nrl")
        self.assertIsNone(result["boost_max"])

    def test_replay_keeps_earlier_successes(self):
        trainer = self.trainer(replay=True)
        trainer.learn([], ["first correct"])
        result = trainer.learn([self.failure("bad", 1.0)], ["second correct"])
        self.assertEqual(result["good"], 2)  # the earlier success rides along


# ---------------------------------------------------------------------------
# the whole loop
# ---------------------------------------------------------------------------


class RunTests(AgentCase):
    def test_an_untrained_network_is_mediated_judged_and_taught(self):
        trainer = self.trainer()
        records = []
        out = trainer.run([self.task()], progress=records.append)
        task_record = next(r for r in out if r["kind"] == "task")
        self.assertTrue(task_record["correct"])
        self.assertEqual(task_record["solved_by"], "teacher")  # the network could not do it yet
        self.assertTrue(task_record["taught"])
        self.assertFalse(task_record["model_solved"])
        self.assertEqual(task_record["own_calls"], 0)  # every call had to be repaired
        self.assertEqual(task_record["autonomy"], 0.0)
        self.assertEqual(task_record["action"], "2nrl")
        self.assertGreaterEqual(task_record["failures"], 1)
        self.assertEqual([r["kind"] for r in records][:2], ["criteria", "step"])
        self.assertEqual(self.model.meta["twonrl_runs"], 1)

    def test_the_transcript_it_learns_is_well_formed(self):
        trainer = self.trainer()
        trainer.run([self.task()])
        learned = trainer.solved["t1"]
        self.assertTrue(learned.startswith("TASK: How many legs does a cat have?\n"))
        self.assertIn('<tool>web_fetch {"url"', learned)
        self.assertIn("<result>", learned)
        self.assertIn("<answer>A cat has four legs.</answer>", learned)

    def test_a_network_that_writes_its_own_call_is_not_mediated(self):
        # teach the network the exact call first; the beam search then offers it among its candidates
        call = '<tool>calculator {"expression": "6*7"}</tool>'
        model = RadixNet(seed=2, backend="python")
        model.train([f"TASK: What is 6*7?\n{call}\n"], epochs=60, lr=0.5, batch_size=8)
        trainer = AgentTrainer(model, self.client, self.toolbox, AgentConfig(**{**FAST, "max_steps": 1}))
        attempt = trainer.solve_with_model(Task("t1", "What is 6*7?"), 0, None, "model")
        self.assertEqual([s.source for s in attempt.steps], ["model"])
        self.assertEqual(attempt.steps[0].call.name, "calculator")
        self.assertEqual(attempt.steps[0].result.output, "42")
        self.assertEqual(attempt.autonomy, 1.0)
        self.assertEqual(self.fake.mediations, 0)

    def test_mediation_always_never_asks_the_network(self):
        trainer = self.trainer(mediation="always", max_steps=1)
        attempt = trainer.solve_with_model(self.task(), 0, None, "model")
        self.assertEqual([s.source for s in attempt.steps], ["mediator"])
        self.assertEqual(attempt.steps[0].emission, "")

    def test_mediation_never_executes_the_broken_call_as_a_failure(self):
        trainer = self.trainer(mediation="never", max_steps=1)
        attempt = trainer.solve_with_model(self.task(), 0, None, "model")
        self.assertEqual(self.fake.mediations, 0)
        self.assertLessEqual(len(attempt.steps), 1)

    def test_teacher_phase(self):
        trainer = self.trainer(phases=("teacher",))
        out = trainer.run([self.task()])
        record = next(r for r in out if r["kind"] == "task")
        self.assertEqual(record["solved_by"], "teacher")
        self.assertEqual(record["calls"], 0)  # the network attempted nothing in this phase

    def test_no_teaching_leaves_the_failure_unsolved(self):
        trainer = self.trainer(teach_on_failure=False)
        record = next(r for r in trainer.run([self.task()]) if r["kind"] == "task")
        self.assertFalse(record["correct"])
        self.assertEqual(record["action"], "punish")  # nothing was right: the failure is trained on, then inverted

    def test_rounds_and_2nrl_per_round(self):
        trainer = self.trainer(rounds=2, twonrl_per="round")
        out = trainer.run([self.task()])
        self.assertEqual(len([r for r in out if r["kind"] == "round"]), 2)
        self.assertNotIn("action", next(r for r in out if r["kind"] == "task"))  # learning is per round here
        self.assertEqual([r for r in out if r["kind"] == "round"][0]["action"], "2nrl")

    def test_a_stop_event_ends_the_run(self):
        stop = threading.Event()
        stop.set()
        out = self.trainer().run([self.task()], stop_event=stop)
        self.assertEqual([r["kind"] for r in out], ["round"])

    def test_no_tasks_and_an_empty_toolbox(self):
        with self.assertRaises(ValueError):
            self.trainer().run([])
        with self.assertRaises(ValueError):
            AgentTrainer(self.model, self.client, ToolBox(), AgentConfig())

    def test_known_answers_can_judge_without_the_llm(self):
        trainer = self.trainer(use_judge=False)
        task = Task("t1", "How many legs does a cat have?", answer="four")
        record = next(r for r in trainer.run([task]) if r["kind"] == "task")
        self.assertTrue(record["correct"])
        self.assertFalse(any(path == "/api/generate" and "judge" in (body.get("system") or "")
                             for _m, path, body in self.fake.requests))

    def test_checkpoints(self):
        from radixnet.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(folder, keep=2)
            self.trainer(checkpoint_every=1).run([self.task()], checkpoint_manager=manager)
            self.assertTrue(manager.list())
            self.assertEqual(manager.list()[-1]["tag"], "agent")


class ExploreTests(AgentCase):
    def test_the_network_chooses_its_own_tasks(self):
        trainer = self.trainer()
        records = []
        out = trainer.explore(steps=2, progress=records.append)
        self.assertEqual(len(out), 2)
        self.assertEqual([r["kind"] for r in out], ["explore", "explore"])
        self.assertEqual([r["task"] for r in out], ["x1", "x2"])
        proposals = [r for r in records if r["kind"] == "proposal"]
        self.assertEqual(len(proposals), 2)
        self.assertIn("about cats", proposals[0]["prompt"])
        self.assertEqual(self.fake.proposals, 2)

    def test_pages_it_reads_become_the_frontier(self):
        trainer = self.trainer()
        trainer.frontier.append(self.site.url + "/")
        trainer.explore(steps=1)
        self.assertIn(self.site.url + "/cats", trainer.visited)

    def test_a_stop_event_ends_the_exploration(self):
        stop = threading.Event()
        stop.set()
        self.assertEqual(self.trainer().explore(steps=5, stop_event=stop), [])

    def test_it_learns_from_every_step(self):
        trainer = self.trainer()
        out = trainer.explore(steps=1)
        self.assertIn(out[0]["action"], ("2nrl", "reward", "punish"))
        self.assertGreater(self.model.graph.num_nodes(), 2)


# ---------------------------------------------------------------------------
# the negative network: the failures the agent finds
# ---------------------------------------------------------------------------

WRONG = (
    'TASK: How many legs does a cat have?\n<tool>web_fetch {"url": "http://x/cats"}</tool>\n'
    "<result>A cat has four legs.</result>\n<answer>A cat has seven legs.</answer>\n"
)
RIGHT = WRONG.replace("seven", "four")


_A_STEP = {"source": "model", "ok": True, "tool": "web_fetch", "emission": '<tool>web_fetch {"url": "http://x/cats"}</tool>',
           "text": '<tool>web_fetch {"url": "http://x/cats"}</tool>\n'}


def _attempt_dict(text, *, correct=False, gap=0.9, steps=(_A_STEP,), critique="the number is false", score=1.0,
                  answer="an answer"):
    """One attempt as the agent reports it; by default it called a tool and got an answer back."""
    return {
        "text": text, "correct": correct, "gap": gap, "answer": answer, "steps": list(steps),
        "verdict": {"correct": correct, "score": score, "critique": critique, "issues": ["names four"]},
    }


class FaultsFromAgentTests(unittest.TestCase):
    def test_the_judge_names_the_reason_and_the_gap_sets_the_severity(self):
        faults, passed = blame.faults_from_agent([_attempt_dict(WRONG)])
        self.assertEqual([f["reason"] for f in faults], ["false"])
        self.assertAlmostEqual(faults[0]["severity"], blame.severity_from_gap(0.9))
        self.assertIn("names four", faults[0]["note"])
        self.assertEqual(faults[0]["source"], "agent")
        self.assertEqual(passed, [])

    def test_a_correct_run_of_the_same_task_becomes_the_correction(self):
        faults, passed = blame.faults_from_agent(
            [_attempt_dict(WRONG), _attempt_dict(RIGHT, correct=True, gap=0.0)], correction=RIGHT,
        )
        self.assertEqual(faults[0]["correction"], RIGHT)
        # the correction already clears what the two share: clearing it again would undo the diff's blame
        self.assertEqual(passed, [])

    def test_without_a_correction_the_correct_run_clears(self):
        _faults, passed = blame.faults_from_agent([_attempt_dict(WRONG), _attempt_dict(RIGHT, correct=True)])
        self.assertEqual(passed, [RIGHT])

    def test_a_repaired_emission_is_blamed_for_itself(self):
        # the transcript holds the mediator's *correct* call, so blaming it there would teach the
        # network that a well-formed call is a mistake; what it actually wrote is the failure
        steps = [{"source": "mediator", "ok": True, "tool": "web_fetch", "emission": "wbf tch {{{", "text": "<tool>web_fetch {}</tool>\n"}]
        faults, _passed = blame.faults_from_agent([_attempt_dict(WRONG, steps=steps)])
        step_fault = next(f for f in faults if f["text"] == "wbf tch {{{")
        self.assertEqual(step_fault["reason"], "bad-call")
        self.assertEqual(step_fault["severity"], blame.AGENT_SEVERITY["bad-call"])
        self.assertIn("web_fetch", step_fault["note"])
        # the transcript holds the mediator's repaired call, so that text is never the one blamed for it
        self.assertNotIn('<tool>web_fetch {}</tool>', [f["text"] for f in faults])

    def test_a_failed_call_the_network_wrote_itself_is_blamed(self):
        steps = [{"source": "model", "ok": False, "tool": "web_fetch", "error": "refusing 10.0.0.1",
                  "text": '<tool>web_fetch {"url": "http://10.0.0.1/"}</tool>\n'}]
        faults, _passed = blame.faults_from_agent([_attempt_dict(WRONG, steps=steps)])
        step_fault = next(f for f in faults if "10.0.0.1" in f["text"])
        self.assertEqual(step_fault["reason"], "tool-error")
        self.assertIn("refusing", step_fault["note"])
        self.assertEqual(step_fault["severity"], blame.AGENT_SEVERITY["tool-error"])

    def test_a_failed_call_the_mediator_wrote_is_not_the_networks_fault(self):
        steps = [{"source": "mediator", "ok": False, "tool": "web_fetch", "error": "boom", "emission": "", "text": "<tool>x {}</tool>\n"}]
        faults, _passed = blame.faults_from_agent([_attempt_dict(WRONG, steps=steps)])
        # only the attempt itself: the call was the mediator's, and the empty emission is too short to blame
        self.assertEqual([f["text"] for f in faults], [WRONG])

    def test_steps_false_keeps_only_the_attempt(self):
        steps = [{"source": "mediator", "ok": True, "emission": "wbf tch {{{", "tool": "web_fetch"}]
        faults, _passed = blame.faults_from_agent([_attempt_dict(WRONG, steps=steps)], steps=False)
        self.assertEqual(len(faults), 1)

    def test_agent_reason_reads_how_far_it_got(self):
        self.assertEqual(blame.agent_reason(_attempt_dict(WRONG, steps=())), "no-call")
        self.assertEqual(blame.agent_reason(_attempt_dict(WRONG)), "false")  # it got all the way to a wrong answer
        self.assertEqual(blame.agent_reason(_attempt_dict(WRONG, steps=[{"source": "mediator", "ok": True}])), "bad-call")
        self.assertEqual(blame.agent_reason(_attempt_dict(WRONG, steps=[{"source": "model", "ok": False}])), "tool-error")
        self.assertEqual(
            blame.agent_reason(_attempt_dict(WRONG, steps=[{"source": "model", "ok": True}], answer=None)), "no-answer"
        )
        self.assertEqual(blame.agent_reason(_attempt_dict(WRONG, steps=[{"source": "model", "ok": True}])), "false")

    def test_severity_from_gap(self):
        self.assertAlmostEqual(blame.severity_from_gap(0.0), 0.25)
        self.assertAlmostEqual(blame.severity_from_gap(1.0), 2.0)
        self.assertAlmostEqual(blame.severity_from_gap(None), 1.0)
        self.assertAlmostEqual(blame.severity_from_gap(5.0), 2.0)  # clamped


class TeachNegativeTests(unittest.TestCase):
    def test_only_the_characters_that_differ_from_a_correct_run_are_blamed(self):
        negative = NegativeNet(seed=0)
        report = blame.teach_agent(
            negative, [_attempt_dict(WRONG), _attempt_dict(RIGHT, correct=True, gap=0.0)], correction=RIGHT,
        )
        self.assertEqual(report["blamed"], 1)
        self.assertEqual(report["reasons"], {"false": 1})
        blamed = {c["fragment"] for c in negative.crossings(WRONG) if (c.get("evidence") or 0) > 0}
        self.assertTrue(blamed, "the wrong word should carry evidence")
        self.assertTrue(any("sev" in f or "ev" in f for f in blamed), blamed)
        # the task line is shared with the correct run, so it is never evidence of anything
        self.assertEqual(negative.judge("TASK: How many legs does a cat have?")["verdict"], "pass")
        self.assertEqual(negative.judge(RIGHT)["verdict"], "pass")
        self.assertGreater(negative.judge("A cat has seven legs.")["risk"], 0.0)


class TrainerNegativeTests(AgentCase):
    def setUp(self):
        super().setUp()
        self.negative = NegativeNet(seed=3)

    def trainer(self, **options):
        return AgentTrainer(
            self.model, self.client, self.toolbox, AgentConfig(**{**FAST, **options}), negative=self.negative,
        )

    def test_a_run_blames_its_failures_and_reports_what_it_taught(self):
        record = next(r for r in self.trainer().run([self.task()]) if r["kind"] == "task")
        self.assertGreaterEqual(record["negative_blamed"], 1)
        self.assertGreater(record["negative_edges"], 0)
        self.assertTrue(record["negative_reasons"])
        self.assertGreater(self.negative.graph.total_blame, 0.0)
        self.assertTrue(self.negative.recent(1))
        self.assertEqual(self.negative.recent(1)[0]["source"], "agent")

    def test_exploring_records_its_own_source(self):
        self.trainer().explore(steps=1)
        self.assertEqual(self.negative.recent(1)[0]["source"], "explore")

    def test_no_negative_network_means_no_blame_keys(self):
        plain = AgentTrainer(self.model, self.client, self.toolbox, AgentConfig(**FAST))
        record = next(r for r in plain.run([self.task()]) if r["kind"] == "task")
        self.assertNotIn("negative_blamed", record)

    def test_a_candidate_the_negative_network_has_seen_fail_is_passed_over(self):
        trainer = self.trainer(max_steps=1, mediation="never")
        known = '<tool>calculator {"expression": "1+1"}</tool>'
        self.negative.blame([known], reason="tool-error", severity=5.0)
        trainer._emissions = lambda transcript, index: [known, '<tool>web_fetch {"url": "%s/cats"}</tool>' % self.site.url]
        call, source, _emission, _answer = trainer._next_call(self.task(), "TASK: x\n", 0, 0)
        self.assertEqual(call.name, "web_fetch")  # the blamed candidate was skipped for the next one
        self.assertEqual(source, "model")
        self.assertEqual(trainer.avoided, 1)

    def test_the_veto_can_be_turned_off(self):
        trainer = self.trainer(max_steps=1, mediation="never", avoid_blamed=False)
        known = '<tool>calculator {"expression": "1+1"}</tool>'
        self.negative.blame([known], reason="tool-error", severity=5.0)
        trainer._emissions = lambda transcript, index: [known]
        call, _source, _emission, _answer = trainer._next_call(self.task(), "TASK: x\n", 0, 0)
        self.assertEqual(call.name, "calculator")
        self.assertEqual(trainer.avoided, 0)

    def test_the_veto_never_fires_without_a_negative_network(self):
        plain = AgentTrainer(self.model, self.client, self.toolbox, AgentConfig(**FAST))
        self.assertFalse(plain._blamed("anything at all"))

    def test_a_negative_network_that_cannot_judge_never_vetoes(self):
        trainer = self.trainer()

        class Broken:
            def judge(self, *args, **kwargs):
                raise RuntimeError("no")

        trainer.negative = Broken()
        self.assertFalse(trainer._blamed("some candidate"))


# ---------------------------------------------------------------------------
# the API endpoints
# ---------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from test_api import start_server, wait_for_job
        except ImportError:
            from tests.test_api import start_server, wait_for_job
        self.wait_for_job = wait_for_job
        self.site = start_site(self.addCleanup)
        self.fake = start_fake_ollama(self.addCleanup, self.site.url)
        self.client, _server, self.service = start_server(
            self.addCleanup, ollama_url=self.fake.url, ollama_model="fake:latest",
            tool_options={"allow_private": True, "search_url": self.site.url + "/search?q={query}"},
        )

    def start(self, path, body, expect=202):
        status, data, _ = self.client.post(path, body)
        self.assertEqual(status, expect, data)
        return data

    def test_tools_endpoint(self):
        status, data, _ = self.client.get("/api/tools")
        self.assertEqual(status, 200)
        self.assertEqual(data["names"], ["web_search", "web_fetch", "web_links", "calculator"])
        self.assertEqual(data["tools"][0]["name"], "web_search")
        self.assertIn("<tool>", data["call_format"])

    def test_status_reports_the_tools(self):
        _status, data, _ = self.client.get("/api/status")
        self.assertIn("calculator", data["tools"]["names"])
        self.assertTrue(data["tools"]["allow_private"])

    def test_tool_call(self):
        status, data, _ = self.client.post("/api/tools/call", {"tool": "calculator", "arguments": {"expression": "6*7"}})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["output"], "42")

    def test_tool_call_in_the_text_form_the_network_writes(self):
        status, data, _ = self.client.post("/api/tools/call", {"call": f'web_fetch {{"url": "{self.site.url}/cats"}}'})
        self.assertEqual(status, 200)
        self.assertIn("A cat has four legs.", data["output"])

    def test_tool_call_errors(self):
        status, data, _ = self.client.post("/api/tools/call", {"tool": "nope"})
        self.assertEqual(status, 404, data)
        status, data, _ = self.client.post("/api/tools/call", {"call": "not a call"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/tools/call", {"tool": "calculator", "arguments": "text"})
        self.assertEqual(status, 400, data)
        # a tool that fails is a 200 with ok=false: the failure is the answer, not an API error
        status, data, _ = self.client.post("/api/tools/call", {"tool": "calculator", "arguments": {"expression": "1/0"}})
        self.assertEqual(status, 200)
        self.assertFalse(data["ok"])

    def test_tool_overrides(self):
        status, data, _ = self.client.post("/api/tools/call", {"tool": "calculator", "arguments": {"expression": "1"}, "offline": True})
        self.assertEqual(status, 200)
        status, data, _ = self.client.post("/api/tools/call", {"tool": "web_fetch", "arguments": {"url": "http://x/"}, "offline": True})
        self.assertEqual(status, 404, data)  # browsing is off for this request

    def test_criteria(self):
        status, data, _ = self.client.post("/api/agent/criteria", {"tasks": ["How many legs does a cat have?"]})
        self.assertEqual(status, 200)
        self.assertEqual(len(data["tasks"][0]["criteria"]), 2)

    def test_solve_without_training(self):
        before = self.service.model.meta["twonrl_runs"]
        status, data, _ = self.client.post(
            "/api/agent/solve", {"tasks": ["How many legs does a cat have?"], "source": "teacher", "max_steps": 2},
        )
        self.assertEqual(status, 200, data)
        self.assertTrue(data["correct"])
        self.assertIn("<answer>", data["transcript"])
        self.assertEqual(data["attempt"]["source"], "teacher")
        self.assertEqual(self.service.model.meta["twonrl_runs"], before)  # nothing was learned

    def test_solve_with_the_model_reports_how_badly_it_failed(self):
        status, data, _ = self.client.post(
            "/api/agent/solve", {"tasks": ["How many legs does a cat have?"], "max_steps": 1, "criteria": 2},
        )
        self.assertEqual(status, 200, data)
        self.assertFalse(data["correct"])  # an untrained network
        self.assertGreater(data["gap"], 0.0)
        self.assertEqual(data["attempt"]["steps"][0]["source"], "mediator")

    def test_start_runs_the_loop_and_records_history(self):
        data = self.start("/api/agent/start", {
            "tasks": ["How many legs does a cat have?"], "max_steps": 2, "model_attempts": 1,
            "criteria": 2, "neg_epochs": 1, "pos_epochs": 1,
        })
        self.assertEqual(data["job"]["type"], "agent")
        self.assertEqual(data["config"]["blatant_mode"], "fail_invert")
        job = self.wait_for_job(self.client)
        self.assertEqual(job["state"], "done", job)
        _status, history, _ = self.client.get("/api/agent/history")
        kinds = [r["kind"] for r in history["history"]]
        self.assertEqual(kinds[0], "criteria")
        self.assertIn("task", kinds)
        record = next(r for r in history["history"] if r["kind"] == "task")
        self.assertTrue(record["correct"])
        self.assertEqual(record["action"], "2nrl")
        self.assertGreaterEqual(record["failures"], 1)

    def test_explore_chooses_its_own_tasks(self):
        data = self.start("/api/agent/explore", {
            "steps": 1, "max_steps": 1, "model_attempts": 1, "criteria": 2, "neg_epochs": 1, "pos_epochs": 1,
            "seed_urls": [self.site.url + "/"],
        })
        self.assertEqual(data["job"]["type"], "explore")
        self.assertEqual(data["steps"], 1)
        job = self.wait_for_job(self.client)
        self.assertEqual(job["state"], "done", job)
        _status, history, _ = self.client.get("/api/agent/history")
        self.assertIn("proposal", [r["kind"] for r in history["history"]])

    def test_bad_requests(self):
        for body, field in (
            ({}, "tasks"), ({"tasks": "not a list"}, "tasks"), ({"tasks": [""]}, "empty"),
            ({"tasks": ["q"], "phase": "nope"}, "phase"), ({"tasks": ["q"], "mediation": "nope"}, "mediation"),
            ({"tasks": ["q"], "blatant_mode": "nope"}, "blatant_mode"), ({"tasks": ["q"], "max_steps": 0}, "max_steps"),
            ({"tasks": ["q"], "twonrl_per": "nope"}, "twonrl_per"),
        ):
            status, data, _ = self.client.post("/api/agent/start", body)
            self.assertEqual(status, 400, (body, data))
            self.assertIn(field, data["error"])

    def test_solve_takes_exactly_one_task(self):
        status, data, _ = self.client.post("/api/agent/solve", {"tasks": ["a?", "b?"]})
        self.assertEqual(status, 400)
        self.assertIn("exactly one task", data["error"])

    def test_an_unreachable_ollama_is_a_502(self):
        self.fake.fail_with = 500
        status, data, _ = self.client.post("/api/agent/criteria", {"tasks": ["q?"]})
        self.assertEqual(status, 502, data)


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def setUp(self):
        self.site = start_site(self.addCleanup)
        self.fake = start_fake_ollama(self.addCleanup, self.site.url)
        self.folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.folder, True)
        self.model = os.path.join(self.folder, "m.json")

    def run_cli(self, *args, expect=0):
        from radixnet.cli import main

        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in args])
        self.assertEqual(code, expect, f"{args}\n{out.getvalue()}\n{err.getvalue()}")
        return out.getvalue()

    def json_cli(self, *args, **kwargs):
        return json.loads(self.run_cli("--json", *args, **kwargs))

    def agent_args(self):
        return [
            "--url", self.fake.url, "--agent-model", "fake:latest", "--timeout", "20",
            "--allow-private", "--search-url", self.site.url + "/search?q={query}",
            "--max-steps", "1", "--model-attempts", "1", "--criteria", "2",
            "--neg-epochs", "1", "--pos-epochs", "1",
        ]

    def test_tools_list_and_describe(self):
        doc = self.json_cli("tools", "list")
        self.assertEqual([t["name"] for t in doc["tools"]][:2], ["web_search", "web_fetch"])
        self.assertEqual(self.json_cli("tools", "list", "--offline")["tools"][0]["name"], "calculator")
        doc = self.json_cli("tools", "describe", "--tool", "web_fetch")
        self.assertEqual(doc["schema"]["function"]["name"], "web_fetch")

    def test_tools_call(self):
        doc = self.json_cli("tools", "call", "--tool", "calculator", "--arg", "expression=6*7", "--offline")
        self.assertEqual(doc["output"], "42")
        doc = self.json_cli("tools", "call", "--call", f'web_fetch {{"url": "{self.site.url}/cats"}}', "--allow-private")
        self.assertIn("four legs", doc["output"])
        self.run_cli("--json", "tools", "call", "--tool", "calculator", "--arg", "expression=1/0", "--offline", expect=1)
        self.run_cli("--json", "tools", "call", "--offline", expect=1)  # neither --tool nor --call
        self.run_cli("--json", "tools", "call", "--tool", "calculator", "--arg", "nonsense", "--offline", expect=1)

    def test_agent_solves_and_saves(self):
        tasks = os.path.join(self.folder, "tasks.txt")
        with open(tasks, "w", encoding="utf-8") as fh:
            fh.write("How many legs does a cat have?\n")
        report = os.path.join(self.folder, "report.json")
        doc = self.json_cli("--model", self.model, "agent", "--tasks", tasks, "--report", report, *self.agent_args())
        self.assertEqual(doc["solved"], 1)
        self.assertEqual(doc["failures"], 1)  # the network's own attempt failed and was trained on
        self.assertEqual(doc["autonomy"], 0.0)
        self.assertTrue(os.path.isfile(self.model))
        with open(report, encoding="utf-8") as fh:
            self.assertIn("records", json.load(fh))

    def test_agent_needs_a_readable_task_file(self):
        self.run_cli("--json", "agent", "--tasks", os.path.join(self.folder, "missing.txt"), *self.agent_args(), expect=1)

    def test_explore_chooses_its_own_tasks(self):
        doc = self.json_cli(
            "--model", self.model, "explore", "--steps", "1", "--seed-url", self.site.url + "/", *self.agent_args()
        )
        self.assertEqual(len(doc["records"]), 1)
        self.assertEqual(doc["records"][0]["kind"], "explore")
        self.assertIn("about cats", doc["records"][0]["prompt"])
        self.assertTrue(doc["visited"])


if __name__ == "__main__":
    unittest.main()
