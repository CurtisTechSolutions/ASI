"""Tests for ``radixnet.codegen``: problems, sandbox, style check, judge, the 2NRL trainer, API and CLI.

A fake Ollama answers teacher requests from a queue of scripted programs and
judges programs by content (``BAD_ANSWER`` marks a wrong program), so the
whole reward / punish loop runs without a real LLM.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import RadixNet  # noqa: E402
from radixnet.codegen import (  # noqa: E402
    Attempt,
    CodeGenConfig,
    CodeGenTrainer,
    Problem,
    RunResult,
    Sandbox,
    StyleReport,
    check_style,
    decide,
    extract_code,
    judge_with_llm,
    load_problems,
    model_prefix,
    parse_problem_file,
    parse_problems,
    solution_text,
    teacher_fix,
    teacher_generate,
)
from radixnet.ollama import OllamaClient, OllamaError  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server  # noqa: F401
except ImportError:
    from tests.test_api import start_server  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELLO = Problem("hello", "Print the word hello.", expected_output="hello")
GOOD = 'print("hello")\n'
BROKEN = "print(hello)\n"
WRONG = 'print("BAD_ANSWER")\n'


# ---------------------------------------------------------------------------
# fake Ollama: scripted teacher, content-based judge
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server: "FakeOllama"

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
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "gemma4:latest", "size": 1}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
            return
        if self.path != "/api/generate":
            self._json(404, {"error": "not found"})
            return
        prompt = body.get("prompt", "")
        if body.get("format") == "json":  # the judge
            match = re.search(r"```python\n(.*?)```", prompt, re.S)
            code = match.group(1) if match else ""
            task = "BAD_ANSWER" not in code
            answer = {
                "task_accomplished": task, "pep8": True, "naming": "sloppyName" not in code,
                "score": 9 if task else 2, "issues": [] if task else ["prints the wrong text"],
                "critique": "fine" if task else "does not solve the task",
            }
            self._json(200, {"response": json.dumps(answer)})
            return
        if self.server.solutions:  # the teacher (generate or fix)
            response = self.server.solutions.pop(0)
        else:
            response = "```python\nprint(\"hello\")\n```"
        self._json(200, {"response": response})


class FakeOllama(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests = []
        self.solutions = []
        self.fail_with = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def start_fake(add_cleanup):
    server = FakeOllama()
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    add_cleanup(server.server_close)
    add_cleanup(server.shutdown)
    return server


def wait_job(client, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, job, _ = client.get("/api/job")
        assert status == 200, (status, job)
        if job and job.get("state") != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish in time")


FAST = dict(neg_epochs=2, pos_epochs=3, neg_lr=0.5, pos_lr=0.1, batch_size=4)


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


class ProblemTests(unittest.TestCase):
    def test_from_any_and_dedupe(self):
        problems = parse_problems(["Print 1.", {"id": "x", "prompt": "Print 2.", "expected_output": "2"}, {"task": "Print 3."}, {"id": "x", "question": "again"}])
        self.assertEqual([p.id for p in problems], ["p1", "x", "p3", "x-2"])
        self.assertEqual(problems[1].expected_output, "2")
        self.assertEqual(problems[3].prompt, "again")
        for bad in ("   ", {"id": "a"}, {"prompt": "x", "tests": 5}, 7):
            with self.assertRaises(ValueError):
                parse_problems([bad])
        with self.assertRaises(ValueError):
            parse_problems([])

    def test_parse_problem_file(self):
        txt = "# comment\nPrint a.\n\nPrint b.\n"
        self.assertEqual([p.prompt for p in parse_problem_file(txt, ".txt")], ["Print a.", "Print b."])
        jsonl = '{"id": "one", "prompt": "Print 1."}\n\n{"prompt": "Print 2.", "tests": "assert True"}\n'
        problems = parse_problem_file(jsonl, ".jsonl")
        self.assertEqual([(p.id, p.tests) for p in problems], [("one", None), ("p2", "assert True")])
        self.assertEqual(len(parse_problem_file('{"problems": ["a", "b"]}', ".json")), 2)
        self.assertEqual(len(parse_problem_file('["a"]', ".json")), 1)
        with self.assertRaises(ValueError):
            parse_problem_file('{"nope": 1}', ".json")

    def test_sample_files_load(self):
        self.assertEqual(len(load_problems(os.path.join(ROOT, "data", "sample_problems.txt"))), 10)
        problems = load_problems(os.path.join(ROOT, "data", "sample_problems.jsonl"))
        self.assertEqual(problems[0].id, "hello")
        self.assertEqual(problems[0].expected_output, "hello, world")


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sandbox = Sandbox(timeout=3.0, memory_mb=128)

    def test_success_and_output(self):
        run = self.sandbox.run('print("hi")\nprint(2 + 2)\n')
        self.assertTrue(run.ok)
        self.assertEqual((run.exit_code, run.stdout, run.error, run.timed_out), (0, "hi\n4\n", None, False))
        self.assertIsNone(run.expected_ok)
        self.assertIsInstance(run.to_dict()["network_isolated"], bool)

    def test_error_is_reported(self):
        run = self.sandbox.run("print(undefined_name)\n")
        self.assertFalse(run.ok)
        self.assertEqual(run.exit_code, 1)
        self.assertEqual(run.error, "NameError: name 'undefined_name' is not defined")
        self.assertNotIn("runpy", run.stderr)
        self.assertIn("solution.py", run.stderr)

    def test_timeout(self):
        run = Sandbox(timeout=1.0).run("while True:\n    pass\n")
        self.assertFalse(run.ok)
        self.assertTrue(run.timed_out)
        self.assertIn("did not finish", run.error)

    def test_expected_output_and_tests(self):
        self.assertTrue(self.sandbox.run("print(4)\n", expected_output="4\n").expected_ok)
        self.assertFalse(self.sandbox.run("print(5)\n", expected_output="4").expected_ok)
        good = self.sandbox.run("def add(a, b):\n    return a + b\n", tests="assert add(1, 2) == 3")
        self.assertTrue(good.ok)
        bad = self.sandbox.run("def add(a, b):\n    return a - b\n", tests="assert add(1, 2) == 3")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.error, "AssertionError")

    def test_memory_limit(self):
        run = self.sandbox.run("x = bytearray(300 << 20)\nprint(len(x))\n")
        self.assertFalse(run.ok)
        self.assertIn("MemoryError", run.error or "")

    def test_isolated_environment(self):
        run = self.sandbox.run("import os, sys\nprint(sorted(os.environ)[:3])\nprint(sys.flags.isolated)\n")
        self.assertTrue(run.ok, run.stderr)
        self.assertIn("1", run.stdout.splitlines()[-1])
        self.assertNotIn("OLLAMA", run.stdout)

    def test_network_is_blocked_when_isolated(self):
        if not self.sandbox.network_isolated:
            self.skipTest("unshare is not usable here")
        run = self.sandbox.run("import socket\nsocket.create_connection(('127.0.0.1', 9), timeout=2)\n")
        self.assertFalse(run.ok)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            Sandbox(timeout=0)
        with self.assertRaises(ValueError):
            Sandbox(memory_mb=-1)


# ---------------------------------------------------------------------------
# style, texts, verdicts
# ---------------------------------------------------------------------------


class StyleTests(unittest.TestCase):
    def test_clean_code_passes(self):
        code = 'MAX_ITEMS = 3\n\n\ndef add(a, b):\n    return a + b\n\n\nclass Counter:\n    pass\n\n\nprint(add(1, 2))\n'
        report = check_style(code)
        self.assertTrue(report.ok, report.issues)
        self.assertTrue(report.pep8_ok and report.naming_ok and report.syntax_ok)

    def test_formatting_issues(self):
        report = check_style("def f():\n\treturn 1  \nx = 1\n" + "y = " + "1 + " * 40 + "1\n")
        codes = [issue.split()[0] for issue in report.issues]
        self.assertIn("W191", codes)
        self.assertIn("W291", codes)
        self.assertIn("E501", codes)
        self.assertFalse(report.pep8_ok)
        self.assertTrue(report.naming_ok)
        self.assertIn("W292", [i.split()[0] for i in check_style("x = 1").issues])
        self.assertIn("E111", [i.split()[0] for i in check_style("if True:\n   x = 1\n").issues])

    def test_naming_and_blank_lines(self):
        report = check_style("def AddNumbers(A, b):\n    return A + b\nclass foo:\n    pass\nmyVar = 1\n")
        codes = [issue.split()[0] for issue in report.issues]
        self.assertEqual(sorted(set(codes)), ["E302", "N801", "N802", "N803", "N806"])
        self.assertTrue(report.pep8_ok is False and report.naming_ok is False)

    def test_syntax_error(self):
        report = check_style("def (:\n")
        self.assertFalse(report.syntax_ok)
        self.assertTrue(report.issues[0].startswith("syntax error"))


class TextTests(unittest.TestCase):
    def test_extract_code(self):
        self.assertEqual(extract_code("Sure!\n```python\nprint(1)\n```\nDone."), "print(1)\n")
        self.assertEqual(extract_code("```\nx = 1\n```"), "x = 1\n")
        self.assertEqual(extract_code("```py\na = 1\n```\n```python\nb = 1\nc = 2\n```"), "b = 1\nc = 2\n")
        self.assertEqual(extract_code("print(2)"), "print(2)\n")
        self.assertEqual(extract_code("   "), "")

    def test_solution_text_and_prefix(self):
        self.assertEqual(solution_text(HELLO, 'print("hello")\n\n'), 'Print the word hello.\nprint("hello")\n')
        self.assertEqual(model_prefix("{problem}\n", HELLO), "Print the word hello.\n")
        self.assertEqual(model_prefix("Q: {problem}\nA:\n", HELLO), "Q: Print the word hello.\nA:\n")
        with self.assertRaises(ValueError):
            model_prefix("no placeholder", HELLO)


def _run(ok=True, expected_ok=None, error=None):
    return RunResult(ok, 0 if ok else 1, "out", "", error, False, 0.01, expected_ok)


def _style(ok=True):
    return StyleReport(ok, True, ok, ok, [] if ok else ["E501 line 1: line too long (90 > 79)"])


class DecideTests(unittest.TestCase):
    def test_sandbox_failures_decide_alone(self):
        verdict = decide(_run(ok=False, error="NameError: x"), _style(), None)
        self.assertEqual((verdict.correct, verdict.runs, verdict.task, verdict.judged_by), (False, False, False, "sandbox"))
        self.assertEqual(verdict.issues[0], "NameError: x")
        verdict = decide(_run(expected_ok=False), _style(), {"task": True, "pep8": True, "naming": True, "score": 9, "issues": [], "critique": ""})
        self.assertFalse(verdict.correct)
        self.assertIn("stdout differs from the expected output", verdict.issues)

    def test_llm_and_strictness(self):
        llm = {"task": True, "pep8": True, "naming": True, "score": 8.0, "issues": ["nit"], "critique": "ok"}
        self.assertTrue(decide(_run(), _style(), llm).correct)
        self.assertFalse(decide(_run(), _style(ok=False), llm).correct)
        self.assertTrue(decide(_run(), _style(ok=False), llm, strictness="lenient").correct)
        llm["task"] = False
        self.assertFalse(decide(_run(), _style(), llm, strictness="lenient").correct)
        llm["task"] = True
        llm["naming"] = False
        verdict = decide(_run(), _style(), llm)
        self.assertFalse(verdict.correct)
        self.assertFalse(verdict.naming)
        self.assertEqual(verdict.score, 8.0)
        with self.assertRaises(ValueError):
            decide(_run(), _style(), llm, strictness="odd")

    def test_without_judge(self):
        self.assertEqual(decide(_run(expected_ok=True), _style(), None).judged_by, "tests")
        verdict = decide(_run(), _style(), None)
        self.assertTrue(verdict.correct)
        self.assertEqual((verdict.task, verdict.judged_by), (None, "none"))


# ---------------------------------------------------------------------------
# teacher and judge against the fake
# ---------------------------------------------------------------------------


class TeacherJudgeTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "gemma4")

    def test_teacher_generate_and_fix(self):
        self.fake.solutions = ["```python\nprint(hello)\n```", "print(\"hello\")"]
        code = teacher_generate(self.client, HELLO, extra="Keep it short.")
        self.assertEqual(code, "print(hello)\n")
        body = self.fake.requests[-1]
        self.assertIn("Task:\nPrint the word hello.", body["prompt"])
        self.assertIn("must be exactly:\nhello", body["prompt"])
        self.assertIn("Keep it short.", body["prompt"])
        self.assertIn("PEP 8", body["system"])
        run = Sandbox(timeout=3).run(code)
        attempt = Attempt(0, "ollama", code, solution_text(HELLO, code), run, check_style(code), decide(run, check_style(code), None), 0.0)
        fixed = teacher_fix(self.client, HELLO, attempt)
        self.assertEqual(fixed, 'print("hello")\n')
        body = self.fake.requests[-1]
        self.assertIn("NameError", body["prompt"])
        self.assertIn("print(hello)", body["prompt"])

    def test_judge(self):
        sandbox = Sandbox(timeout=3)
        run = sandbox.run(GOOD)
        opinion = judge_with_llm(self.client, HELLO, GOOD, run, check_style(GOOD))
        self.assertEqual((opinion["task"], opinion["pep8"], opinion["naming"], opinion["score"]), (True, True, True, 9.0))
        self.assertEqual(self.fake.requests[-1]["format"], "json")
        self.assertIn("stdout:\nhello", self.fake.requests[-1]["prompt"])
        opinion = judge_with_llm(self.client, HELLO, WRONG, sandbox.run(WRONG), check_style(WRONG))
        self.assertFalse(opinion["task"])
        self.assertEqual(opinion["issues"], ["prints the wrong text"])
        self.fake.fail_with = 500
        with self.assertRaises(OllamaError):
            judge_with_llm(self.client, HELLO, GOOD, run, check_style(GOOD))


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "gemma4")
        self.sandbox = Sandbox(timeout=3.0, memory_mb=128)
        self.model = RadixNet(seed=0, backend="python")

    def trainer(self, **overrides):
        config = CodeGenConfig(teacher_model="gemma4", **{**FAST, **overrides})
        return CodeGenTrainer(self.model, self.client, self.sandbox, config)

    def test_teacher_phase_punishes_wrong_and_rewards_correct(self):
        self.fake.solutions = ["```python\nprint(hello)\n```", "```python\nprint(\"BAD_ANSWER\")\n```", "```python\nprint(\"hello\")\n```"]
        trainer = self.trainer(phases=("teacher",))
        records = trainer.run([HELLO])
        attempts = [r for r in trainer.history if r["kind"] == "attempt"]
        self.assertEqual([(a["runs"], a["correct"]) for a in attempts], [(False, False), (True, False), (True, True)])
        self.assertTrue(attempts[0]["error"].startswith("NameError: name 'hello' is not defined"), attempts[0]["error"])
        self.assertEqual(attempts[1]["judged_by"], "sandbox")  # the expected output already rejects it
        problem = [r for r in records if r["kind"] == "problem"][0]
        self.assertEqual((problem["correct"], problem["solved_by"], problem["attempts"], problem["bad"], problem["good"]), (True, "ollama", 3, 2, 1))
        self.assertEqual(problem["action"], "2nrl")
        self.assertIsNotNone(problem["neg_loss"])
        self.assertIsNotNone(problem["pos_loss"])
        self.assertTrue(self.model.graph.inverted)
        self.assertEqual(trainer.solved["hello"], 'Print the word hello.\nprint("hello")\n')
        summary = records[-1]
        self.assertEqual((summary["kind"], summary["solved"], summary["problems"], summary["model_solved"]), ("round", 1, 1, 0))
        self.assertGreater(self.model.stats()["trained_texts"], 0)

    def test_model_reproduces_a_learned_solution(self):
        # reward only: the model learns the question + answer text and its cheapest path returns the code
        trainer = self.trainer(phases=("model",), fallback_teacher=False)
        text = solution_text(HELLO, GOOD)
        self.model.train([text], epochs=8, lr=0.5, batch_size=4)
        self.assertEqual(trainer.generate_with_model(HELLO, 0), GOOD)
        records = trainer.run([HELLO])
        problem = [r for r in records if r["kind"] == "problem"][0]
        self.assertEqual((problem["correct"], problem["solved_by"], problem["model_solved"], problem["action"]), (True, "model", True, "reward"))

    def test_model_phase_punishes_and_falls_back_to_the_teacher(self):
        trainer = self.trainer(phases=("model",), model_attempts=2, fallback_teacher=True)
        records = trainer.run([HELLO])  # an untrained model produces nothing usable
        attempts = [r for r in trainer.history if r["kind"] == "attempt"]
        self.assertEqual([a["source"] for a in attempts][:2], ["model", "model"])
        self.assertEqual(attempts[-1]["source"], "ollama")
        problem = [r for r in records if r["kind"] == "problem"][0]
        self.assertEqual((problem["correct"], problem["solved_by"], problem["model_solved"], problem["action"]), (True, "ollama", False, "2nrl"))
        self.assertGreaterEqual(problem["bad"], 2)

    def test_punish_only_when_nothing_is_correct(self):
        trainer = self.trainer(phases=("model",), model_attempts=1, fallback_teacher=False)
        inverted_before = self.model.graph.inverted
        records = trainer.run([HELLO])
        problem = [r for r in records if r["kind"] == "problem"][0]
        self.assertEqual((problem["correct"], problem["action"], problem["good"]), (False, "punish", 0))
        self.assertNotEqual(self.model.graph.inverted, inverted_before)

    def test_round_mode_and_replay(self):
        self.fake.solutions = ["```python\nprint(hello)\n```", "```python\nprint(\"hello\")\n```"]
        trainer = self.trainer(phases=("teacher",), twonrl_per="round", teacher_attempts=2, rounds=2)
        records = trainer.run([HELLO, Problem("two", "Print the word hello twice.", expected_output="hello\nhello")])
        problems = [r for r in records if r["kind"] == "problem"]
        self.assertTrue(all("action" not in r for r in problems))
        rounds = [r for r in records if r["kind"] == "round"]
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["action"], "2nrl")
        self.assertGreaterEqual(rounds[1]["good"], 1)  # replay keeps earlier successes in the positive phase
        self.assertLessEqual(len(trainer.replay_buffer), trainer.config.replay_limit)

    def test_stop_event_and_validation(self):
        stop = threading.Event()
        stop.set()
        records = self.trainer(phases=("teacher",)).run([HELLO], stop_event=stop)
        self.assertEqual([r["kind"] for r in records], ["round"])
        self.assertEqual(records[0]["problems"], 0)
        for bad in (dict(phases=("later",)), dict(rounds=0), dict(strictness="odd"), dict(twonrl_per="epoch"), dict(model_prompt="x")):
            with self.assertRaises(ValueError):
                self.trainer(**bad)
        with self.assertRaises(ValueError):
            self.trainer().run([])

    def test_ollama_failure_propagates(self):
        self.fake.fail_with = 500
        with self.assertRaises(OllamaError):
            self.trainer(phases=("teacher",)).run([HELLO])

    def test_config_to_dict(self):
        data = CodeGenConfig().to_dict()
        self.assertEqual(data["phases"], ["teacher", "model"])
        self.assertEqual(data["teacher_model"], CodeGenConfig().teacher_model)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.dir = tempfile.mkdtemp(prefix="radixnet-codegen-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.client, self.server, self.service = start_server(
            self.addCleanup, upload_dir=self.dir, ollama_url=self.fake.url, ollama_model="gemma4",
        )

    def test_run_endpoint(self):
        status, data, _ = self.client.post("/api/codegen/run", {"code": 'print("hi")', "expected_output": "hi"})
        self.assertEqual(status, 200, data)
        self.assertTrue(data["run"]["ok"])
        self.assertTrue(data["run"]["expected_ok"])
        self.assertTrue(data["style"]["ok"])
        self.assertTrue(data["verdict"]["correct"])
        status, data, _ = self.client.post("/api/codegen/run", {"code": "print(x)", "sandbox_timeout": 3})
        self.assertEqual(status, 200, data)
        self.assertFalse(data["run"]["ok"])
        self.assertIn("NameError", data["run"]["error"])
        self.assertFalse(data["verdict"]["correct"])
        status, data, _ = self.client.post("/api/codegen/run", {})
        self.assertEqual(status, 400, data)

    def test_solve_endpoint(self):
        self.fake.solutions = ["```python\nprint(\"hello\")\n```"]
        body = {"problem": {"prompt": HELLO.prompt, "expected_output": "hello"}, "source": "teacher"}
        status, data, _ = self.client.post("/api/codegen/solve", body)
        self.assertEqual(status, 200, data)
        self.assertTrue(data["correct"])
        self.assertEqual(data["attempts"][0]["code"], GOOD)
        self.assertEqual(data["attempts"][0]["verdict"]["judged_by"], "ollama")
        status, data, _ = self.client.post("/api/codegen/solve", {"problem": HELLO.prompt, "source": "model", "judge": False, "attempts": 2})
        self.assertEqual(status, 200, data)
        self.assertFalse(data["correct"])
        self.assertEqual(len(data["attempts"]), 2)
        self.assertEqual(data["attempts"][0]["source"], "model")
        status, data, _ = self.client.post("/api/codegen/solve", {"problem": "x", "source": "oracle"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/codegen/solve", {"source": "model"})
        self.assertEqual(status, 400, data)

    def test_start_job_with_texts_and_uploads(self):
        self.fake.solutions = ["```python\nprint(hello)\n```", "```python\nprint(\"hello\")\n```"]
        self.client.post("/api/uploads", {"name": "problems.jsonl", "content": json.dumps({"id": "up", "prompt": "Print the word hello.", "expected_output": "hello"}) + "\n"})
        body = {"problems_text": "Print the word hello.\n", "problem_files": ["problems.jsonl"], "phases": "teacher", "teacher_attempts": 2, **FAST}
        status, data, _ = self.client.post("/api/codegen/start", body)
        self.assertEqual(status, 202, data)
        self.assertEqual(data["job"]["type"], "codegen")
        self.assertEqual(data["problems"], ["p1", "up"])
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        kinds = [r["kind"] for r in job["history"]]
        self.assertEqual(kinds.count("problem"), 2)
        self.assertEqual(kinds[-1], "round")
        status, hist, _ = self.client.get("/api/codegen/history")
        self.assertEqual(len(hist["history"]), len(job["history"]))
        status, stats, _ = self.client.get("/api/status")
        self.assertGreater(stats["trained_texts"], 0)
        problems = [r for r in job["history"] if r["kind"] == "problem"]
        self.assertTrue(all(r["correct"] for r in problems), problems)

    def test_start_errors(self):
        status, data, _ = self.client.post("/api/codegen/start", {})
        self.assertEqual(status, 400, data)
        self.assertIn("problems", data["error"])
        status, data, _ = self.client.post("/api/codegen/start", {"problems": ["x"], "phases": "later"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/codegen/start", {"problems": ["x"], "strictness": "odd"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/codegen/start", {"problems": "x"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/codegen/start", {"problem_files": ["missing.txt"]})
        self.assertEqual(status, 404, data)

    def test_readers_are_served_while_the_teacher_thinks(self):
        # a job releases the model lock around sandbox / LLM calls, so status stays responsive
        self.fake.solutions = ["```python\nprint(\"hello\")\n```"] * 3
        body = {"problems": [HELLO.prompt] * 3, "phases": "teacher", **FAST}
        status, data, _ = self.client.post("/api/codegen/start", body)
        self.assertEqual(status, 202, data)
        t0 = time.time()
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(status, 200)
        self.assertLess(time.time() - t0, 5.0)
        wait_job(self.client)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.dir = tempfile.mkdtemp(prefix="radixnet-codegen-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.model = os.path.join(self.dir, "model.json")
        self.problems = os.path.join(self.dir, "problems.txt")
        with open(self.problems, "w", encoding="utf-8") as fh:
            fh.write("Print the word hello.\n")

    def run_cli(self, *args, expect=0):
        cmd = [sys.executable, "-m", "radixnet", "--model", self.model, "--backend", "python", "--json", *args]
        env = dict(os.environ, PYTHONWARNINGS="ignore", OLLAMA_HOST=self.fake.url)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(proc.returncode, expect, proc.stderr[-3000:])
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def test_codegen_teacher_phase(self):
        self.fake.solutions = ["```python\nprint(hello)\n```", "```python\nprint(\"hello\")\n```"]
        report = os.path.join(self.dir, "report.json")
        doc = self.run_cli("codegen", "--problems", self.problems, "--phase", "teacher", "--teacher-model", "gemma4",
                           "--teacher-attempts", "2", "--sandbox-timeout", "3", "--report", report)
        self.assertEqual((doc["solved"], doc["model_solved"], doc["attempts"]), (1, 0, 2))
        self.assertEqual(doc["config"]["phases"], ["teacher"])
        self.assertEqual(doc["solutions"], {"p1": 'Print the word hello.\nprint("hello")\n'})
        self.assertTrue(os.path.isfile(self.model))
        with open(report, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["solved"], 1)
        self.run_cli("codegen", "--problems", os.path.join(self.dir, "missing.txt"), expect=1)
        self.run_cli("codegen", "--problems", self.problems, "--twonrl-per", "epoch", expect=1)


if __name__ == "__main__":
    unittest.main()
