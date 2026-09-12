"""Tests for ``radixnet.chatgpt``: the client, the ChatGPT tutor in codegen, its API endpoint and CLI.

A fake OpenAI server (standard library) stands in for the real API: it answers
``/v1/models`` and ``/v1/chat/completions``, records every request with the
Authorization header it arrived with, writes scripted programs for the tutor,
judges programs by content (``BAD_ANSWER`` marks a wrong one) and can reject
optional request fields the way a picky model does or fail outright.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import RadixNet  # noqa: E402
from radixnet import chatgpt  # noqa: E402
from radixnet.chatgpt import (  # noqa: E402
    ChatGPTClient,
    ChatGPTError,
    _options_body,
    api_key,
    api_key_configured,
    normalise_url,
)
from radixnet.codegen import (  # noqa: E402
    CodeGenConfig,
    CodeGenTrainer,
    Problem,
    Sandbox,
    default_teacher_model,
)
from radixnet.llm import LLMError, default_model, make_client, normalise_provider, provider_of  # noqa: E402
from radixnet.ollama import OllamaClient  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server  # noqa: F401
    from test_codegen import start_fake as start_fake_ollama  # noqa: F401
except ImportError:
    from tests.test_api import start_server  # noqa: F401
    from tests.test_codegen import start_fake as start_fake_ollama  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELLO = Problem("hello", "Print the word hello.", expected_output="hello")
GOOD = 'print("hello")\n'
KEY = "sk-test-key"
FAST = dict(neg_epochs=2, pos_epochs=3, neg_lr=0.5, pos_lr=0.1, batch_size=4)


# ---------------------------------------------------------------------------
# fake OpenAI API
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server: "FakeOpenAI"

    def log_message(self, *args):  # silence
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, code=None, param=None):
        self._json(status, {"error": {"message": message, "type": "invalid_request_error", "code": code, "param": param}})

    def do_GET(self):
        self.server.requests.append(("GET", self.path, None, self.headers.get("Authorization")))
        if self.server.fail_with:
            self._error(self.server.fail_with, "boom")
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "fake-gpt", "object": "model", "created": 1735689600, "owned_by": "openai"},
                {"id": "another-gpt", "object": "model", "created": 1735689601, "owned_by": "system"},
            ]})
        else:
            self._error(404, "unknown path")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(("POST", self.path, body, self.headers.get("Authorization")))
        if self.server.fail_with:
            self._error(self.server.fail_with, "boom")
            return
        if self.path != "/v1/chat/completions":
            self._error(404, "unknown path")
            return
        rejected = next((field for field in self.server.reject if field in body), None)
        if rejected is not None:
            if self.server.reject_style == "argument":  # the shape that names no param
                self._error(400, f"Unrecognized request argument supplied: {rejected}")
            else:
                self._error(400, f"Unsupported parameter: '{rejected}' is not supported with this model.",
                            code="unsupported_parameter", param=rejected)
            return
        if self.server.answer is not None:
            self._json(200, _completion(self.server.answer))
            return
        self._json(200, _completion(self._answer(body)))

    def _answer(self, body):
        messages = body.get("messages") or []
        system = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
        user = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if body.get("response_format", {}).get("type") == "json_object":  # the judge
            match = re.search(r"```python\n(.*?)```", user, re.S)
            code = match.group(1) if match else ""
            task = "BAD_ANSWER" not in code
            return json.dumps({
                "task_accomplished": task, "pep8": True, "naming": True, "score": 9 if task else 2,
                "issues": [] if task else ["prints the wrong text"],
                "critique": "fine" if task else "does not solve the task",
            })
        if "Python programmer" in system:  # the tutor (a first program or a fix)
            return self.server.solutions.pop(0) if self.server.solutions else '```python\nprint("hello")\n```'
        return f"answer to: {user.strip()}"


def _completion(content):
    return {
        "id": "chatcmpl-fake", "object": "chat.completion", "model": "fake-gpt",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class FakeOpenAI(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests = []
        self.solutions = []
        self.reject = ()
        self.reject_style = "parameter"
        self.answer = None
        self.fail_with = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    @property
    def bodies(self):
        return [body for _method, _path, body, _auth in self.requests if body is not None]


def start_fake(add_cleanup):
    server = FakeOpenAI()
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    add_cleanup(server.server_close)
    add_cleanup(server.shutdown)
    return server


def closed_port_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def set_env(test, **values):
    """Set environment variables for one test (``None`` removes one); restored afterwards."""
    for name, value in values.items():
        previous = os.environ.get(name)
        test.addCleanup(lambda n=name, p=previous: os.environ.__setitem__(n, p) if p is not None else os.environ.pop(n, None))
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def wait_job(client, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, job, _ = client.get("/api/job")
        assert status == 200, (status, job)
        if job and job.get("state") != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish in time")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        set_env(self, OPENAI_API_KEY=KEY, OPENAI_API_KEY_FILE=None)

    def client(self, **kwargs):
        return ChatGPTClient(kwargs.pop("url", self.fake.url), kwargs.pop("model", "fake-gpt"), **kwargs)

    def test_normalise_url(self):
        self.assertEqual(normalise_url("api.openai.com"), "https://api.openai.com/v1")
        self.assertEqual(normalise_url("https://api.openai.com/v1/"), "https://api.openai.com/v1")
        self.assertEqual(normalise_url("  http://localhost:1234/v1  "), "http://localhost:1234/v1")
        self.assertEqual(normalise_url("http://127.0.0.1:8080"), "http://127.0.0.1:8080/v1")
        for bad in ("", "   ", "ftp://host/v1", "https://"):
            with self.assertRaises(ValueError):
                normalise_url(bad)
        with self.assertRaises(ValueError) as ctx:  # a key must not travel unencrypted to a remote host
            normalise_url("http://gateway.example.com/v1")
        self.assertIn("unencrypted", str(ctx.exception))
        set_env(self, RADIXNET_OPENAI_ALLOW_INSECURE="1")
        self.assertEqual(normalise_url("http://gateway.example.com/v1"), "http://gateway.example.com/v1")

    def test_api_key_sources(self):
        self.assertEqual(api_key(), KEY)
        self.assertEqual(api_key("  explicit  "), "explicit")
        self.assertTrue(api_key_configured())
        directory = tempfile.mkdtemp(prefix="radixnet-key-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "key.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("from-file\n")
        set_env(self, OPENAI_API_KEY=None, OPENAI_API_KEY_FILE=path)
        self.assertEqual(api_key(), "from-file")
        set_env(self, OPENAI_API_KEY_FILE=os.path.join(directory, "missing.txt"))
        self.assertIsNone(api_key())
        self.assertFalse(api_key_configured())

    def test_models_and_available(self):
        client = self.client()
        self.assertTrue(client.available())
        self.assertEqual([m["name"] for m in client.models()], ["another-gpt", "fake-gpt"])
        self.assertEqual(client.models()[1]["owned_by"], "openai")
        self.assertEqual(self.fake.requests[0][3], f"Bearer {KEY}")
        down = ChatGPTClient(closed_port_url(), "fake-gpt", timeout=2)
        self.assertFalse(down.available())
        with self.assertRaises(ChatGPTError) as ctx:
            down.models()
        self.assertIn("cannot reach the OpenAI API", str(ctx.exception))

    def test_without_a_key_nothing_is_sent(self):
        set_env(self, OPENAI_API_KEY=None)
        client = self.client()
        self.assertFalse(client.configured)
        self.assertFalse(client.available())
        with self.assertRaises(ChatGPTError) as ctx:
            client.generate("hello")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        self.assertEqual(self.fake.requests, [])  # never left the process
        self.assertNotIn(KEY, repr(client))

    def test_generate_and_chat(self):
        client = self.client()
        self.assertEqual(client.generate("hi there", system="be brief"), "answer to: hi there")
        body = self.fake.bodies[-1]
        self.assertEqual(body["model"], "fake-gpt")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["messages"][0]["content"], "be brief")
        self.assertNotIn("response_format", body)

        self.assertEqual(client.chat([{"role": "user", "content": "and again"}], model="another-gpt"), "answer to: and again")
        body = self.fake.bodies[-1]
        self.assertEqual((body["model"], len(body["messages"])), ("another-gpt", 1))

    def test_options_are_translated(self):
        client = self.client()
        client.generate("hi", options={"temperature": 0.3, "num_predict": 64, "top_p": 0.9, "mirostat": 2})
        body = self.fake.bodies[-1]
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["max_completion_tokens"], 64)
        self.assertEqual(body["top_p"], 0.9)
        self.assertNotIn("mirostat", body)  # an Ollama-only knob is dropped, not sent
        self.assertEqual(_options_body({"num_predict": -1, "max_tokens": "nope", "seed": 7}), {"seed": 7})
        self.assertEqual(_options_body(None), {})

    def test_json_mode(self):
        self.fake.answer = '{"ok": true}'
        client = self.client()
        self.assertEqual(client.generate("give me json", json_mode=True), '{"ok": true}')
        self.assertEqual(self.fake.bodies[-1]["response_format"], {"type": "json_object"})

    def test_fields_a_model_rejects_are_dropped_and_retried(self):
        client = self.client()
        self.fake.reject = ("temperature", "response_format")
        self.assertEqual(client.generate("hi", json_mode=True, options={"temperature": 0.3}), "answer to: hi")
        self.assertEqual(len(self.fake.bodies), 3)  # temperature, then response_format, then an accepted body
        self.assertNotIn("temperature", self.fake.bodies[-1])
        self.assertNotIn("response_format", self.fake.bodies[-1])

    def test_fields_rejected_without_a_param_name(self):
        client = self.client()
        self.fake.reject = ("response_format",)
        self.fake.reject_style = "argument"
        self.assertEqual(client.generate("hi", json_mode=True), "answer to: hi")
        self.assertEqual(len(self.fake.bodies), 2)

    def test_http_errors_carry_the_reason(self):
        client = self.client()
        self.fake.fail_with = 401
        with self.assertRaises(ChatGPTError) as ctx:
            client.generate("hi")
        self.assertIn("HTTP 401", str(ctx.exception))
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))  # the hint
        self.assertEqual(ctx.exception.status, 401)
        self.fake.fail_with = 429
        with self.assertRaises(ChatGPTError) as ctx:
            client.generate("hi")
        self.assertIn("rate limit", str(ctx.exception))
        self.assertIsInstance(ctx.exception, LLMError)

    def test_unusable_answers(self):
        client = self.client()
        self.fake.answer = ""
        with self.assertRaises(ChatGPTError) as ctx:
            client.generate("hi")
        self.assertIn("empty answer", str(ctx.exception))

    def test_provider_factory(self):
        self.assertEqual(normalise_provider("OpenAI"), "chatgpt")
        self.assertEqual(normalise_provider(""), "ollama")
        with self.assertRaises(ValueError):
            normalise_provider("bard")
        client = make_client("chatgpt", self.fake.url, "fake-gpt")
        self.assertIsInstance(client, ChatGPTClient)
        self.assertEqual(provider_of(client), "chatgpt")
        self.assertEqual(provider_of(OllamaClient()), "ollama")
        self.assertEqual(default_model("chatgpt"), chatgpt.DEFAULT_MODEL)
        self.assertEqual(default_teacher_model("chatgpt"), chatgpt.DEFAULT_MODEL)
        self.assertNotEqual(default_teacher_model("ollama"), default_teacher_model("chatgpt"))


# ---------------------------------------------------------------------------
# the tutor
# ---------------------------------------------------------------------------


class TutorTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        set_env(self, OPENAI_API_KEY=KEY)
        self.client = ChatGPTClient(self.fake.url, "fake-gpt")
        self.sandbox = Sandbox(timeout=3.0, memory_mb=128)
        self.model = RadixNet(seed=0, backend="python")

    def trainer(self, judge_client=None, **overrides):
        config = CodeGenConfig(teacher_provider="chatgpt", teacher_model="fake-gpt", **{**FAST, **overrides})
        return CodeGenTrainer(self.model, self.client, self.sandbox, config, judge_client=judge_client)

    def test_config_defaults_follow_the_provider(self):
        config = CodeGenConfig(teacher_provider="openai")
        self.assertEqual(config.teacher_provider, "chatgpt")
        self.assertEqual(config.teacher_model, chatgpt.DEFAULT_MODEL)
        self.assertEqual(config.judge_provider, "chatgpt")  # the judge follows the tutor
        self.assertEqual(config.resolved_judge_model, chatgpt.DEFAULT_MODEL)
        self.assertEqual(config.to_dict()["teacher_provider"], "chatgpt")
        mixed = CodeGenConfig(teacher_provider="chatgpt", teacher_model="gpt-x", judge_provider="ollama")
        self.assertEqual(mixed.resolved_judge_model, default_teacher_model("ollama"))
        with self.assertRaises(ValueError):
            CodeGenConfig(teacher_provider="bard")

    def test_teacher_phase_is_solved_and_labelled_chatgpt(self):
        self.fake.solutions = ['```python\nprint(hello)\n```', '```python\nprint("hello")\n```']
        trainer = self.trainer(phases=("teacher",), teacher_attempts=2)
        records = trainer.run([HELLO])
        attempts = [r for r in trainer.history if r["kind"] == "attempt"]
        self.assertEqual([(a["source"], a["correct"]) for a in attempts], [("chatgpt", False), ("chatgpt", True)])
        self.assertEqual(attempts[-1]["judged_by"], "chatgpt")
        problem = [r for r in records if r["kind"] == "problem"][0]
        self.assertEqual((problem["correct"], problem["solved_by"], problem["action"]), (True, "chatgpt", "2nrl"))
        self.assertEqual(trainer.solved["hello"], "Print the word hello.\n" + GOOD)
        # the tutor saw the problem, the judge saw the program
        teacher_bodies = [b for b in self.fake.bodies if "response_format" not in b]
        self.assertIn("Print the word hello.", teacher_bodies[0]["messages"][-1]["content"])
        self.assertIn("What went wrong", teacher_bodies[1]["messages"][-1]["content"])

    def test_a_local_judge_can_review_a_chatgpt_tutor(self):
        ollama_fake = start_fake_ollama(self.addCleanup)
        judge = OllamaClient(ollama_fake.url, "gemma4")
        trainer = self.trainer(judge_client=judge, phases=("teacher",), teacher_attempts=1, judge_provider="ollama")
        records = trainer.run([HELLO])
        attempt = [r for r in trainer.history if r["kind"] == "attempt"][-1]
        self.assertEqual((attempt["source"], attempt["judged_by"]), ("chatgpt", "ollama"))
        self.assertTrue([r for r in records if r["kind"] == "problem"][0]["correct"])
        self.assertTrue(ollama_fake.requests)  # the judge really ran on the other provider

    def test_failures_propagate_as_llm_errors(self):
        self.fake.fail_with = 500
        with self.assertRaises(LLMError):
            self.trainer(phases=("teacher",)).run([HELLO])


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        set_env(self, OPENAI_API_KEY=KEY, OPENAI_API_KEY_FILE=None)
        self.dir = tempfile.mkdtemp(prefix="radixnet-chatgpt-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.client, self.server, self.service = start_server(
            self.addCleanup, upload_dir=self.dir, chatgpt_url=self.fake.url, chatgpt_model="fake-gpt",
        )

    def test_status_reports_the_provider(self):
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(status, 200, stats)
        self.assertEqual(stats["chatgpt"], {"url": self.fake.url + "/v1", "model": "fake-gpt", "configured": True})

    def test_models_endpoint(self):
        status, data, _ = self.client.get("/api/chatgpt/models")
        self.assertEqual(status, 200, data)
        self.assertEqual((data["available"], data["configured"], data["error"]), (True, True, None))
        self.assertEqual([m["name"] for m in data["models"]], ["another-gpt", "fake-gpt"])
        self.assertEqual(set(data["models"][0]), {"name", "owned_by", "created"})

        self.fake.fail_with = 401
        status, data, _ = self.client.get("/api/chatgpt/models")
        self.assertEqual(status, 200, data)  # never fails: it reports
        self.assertFalse(data["available"])
        self.assertIn("HTTP 401", data["error"])

    def test_models_endpoint_without_a_key(self):
        set_env(self, OPENAI_API_KEY=None)
        status, data, _ = self.client.get("/api/chatgpt/models")
        self.assertEqual(status, 200, data)
        self.assertEqual((data["available"], data["configured"], data["models"]), (False, False, []))
        self.assertIn("OPENAI_API_KEY", data["error"])
        self.assertEqual(self.fake.requests, [])

    def test_solve_with_a_chatgpt_teacher(self):
        self.fake.solutions = ['```python\nprint("hello")\n```']
        body = {"problem": {"prompt": HELLO.prompt, "expected_output": "hello"}, "source": "teacher",
                "teacher_provider": "chatgpt"}
        status, data, _ = self.client.post("/api/codegen/solve", body)
        self.assertEqual(status, 200, data)
        self.assertTrue(data["correct"])
        self.assertEqual(data["attempts"][0]["source"], "chatgpt")
        self.assertEqual(data["attempts"][0]["verdict"]["judged_by"], "chatgpt")
        self.assertEqual(self.fake.bodies[0]["model"], "fake-gpt")

    def test_start_job_with_a_chatgpt_teacher(self):
        self.fake.solutions = ['```python\nprint(hello)\n```', '```python\nprint("hello")\n```']
        body = {"problems": [{"prompt": HELLO.prompt, "expected_output": "hello"}], "phases": "teacher",
                "teacher_provider": "openai", "teacher_attempts": 2, **FAST}
        status, data, _ = self.client.post("/api/codegen/start", body)
        self.assertEqual(status, 202, data)
        self.assertEqual(data["config"]["teacher_provider"], "chatgpt")
        self.assertEqual(data["config"]["teacher_model"], "fake-gpt")
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        status, history, _ = self.client.get("/api/codegen/history")
        attempts = [r for r in history["history"] if r["kind"] == "attempt"]
        self.assertEqual([a["source"] for a in attempts], ["chatgpt", "chatgpt"])

    def test_bad_provider_and_missing_key_are_refused(self):
        status, data, _ = self.client.post("/api/codegen/solve", {"problem": "x", "teacher_provider": "bard"})
        self.assertEqual(status, 400, data)
        self.assertIn("teacher_provider", data["error"])
        set_env(self, OPENAI_API_KEY=None)
        status, data, _ = self.client.post(
            "/api/codegen/solve", {"problem": "x", "source": "teacher", "teacher_provider": "chatgpt"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("OPENAI_API_KEY", data["error"])
        self.assertEqual(self.fake.requests, [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.dir = tempfile.mkdtemp(prefix="radixnet-chatgpt-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.model = os.path.join(self.dir, "model.json")

    def run_cli(self, *args, expect=0, key=KEY):
        cmd = [sys.executable, "-m", "radixnet", "--model", self.model, "--backend", "python", "--json", *args]
        env = dict(os.environ, PYTHONWARNINGS="ignore", OPENAI_BASE_URL=self.fake.url, RADIXNET_OPENAI_MODEL="fake-gpt")
        env.pop("OPENAI_API_KEY_FILE", None)
        if key is None:
            env.pop("OPENAI_API_KEY", None)
        else:
            env["OPENAI_API_KEY"] = key
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(proc.returncode, expect, proc.stderr[-2000:])
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def test_models(self):
        doc = self.run_cli("chatgpt", "models")
        self.assertEqual((doc["url"], doc["model"]), (self.fake.url + "/v1", "fake-gpt"))
        self.assertEqual([m["name"] for m in doc["models"]], ["another-gpt", "fake-gpt"])
        self.assertIsNone(self.run_cli("chatgpt", "models", expect=1, key=None))  # no key: nothing is sent
        self.assertIsNone(self.run_cli("chatgpt", "--url", closed_port_url(), "--timeout", "2", "models", expect=1))

    def test_ask(self):
        doc = self.run_cli("chatgpt", "ask", "--prompt", "what is 2 + 2", "--system", "be brief")
        self.assertEqual(doc["answer"], "answer to: what is 2 + 2")
        self.assertEqual(self.fake.bodies[-1]["messages"][0], {"role": "system", "content": "be brief"})

    def test_codegen_with_a_chatgpt_teacher(self):
        problems = os.path.join(self.dir, "problems.jsonl")
        with open(problems, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": "hello", "prompt": HELLO.prompt, "expected_output": "hello"}) + "\n")
        self.fake.solutions = ['```python\nprint("hello")\n```']
        doc = self.run_cli("codegen", "--problems", problems, "--phase", "teacher", "--teacher-provider", "chatgpt",
                           "--teacher-attempts", "1", "--neg-epochs", "1", "--pos-epochs", "1")
        self.assertEqual(doc["config"]["teacher_provider"], "chatgpt")
        self.assertEqual(doc["config"]["teacher_model"], "fake-gpt")
        self.assertEqual(doc["solved"], 1)
        self.assertEqual([r["solved_by"] for r in doc["records"] if r["kind"] == "problem"], ["chatgpt"])
        self.assertTrue(os.path.isfile(self.model))

    def test_codegen_without_a_key_stops_before_asking(self):
        problems = os.path.join(self.dir, "problems.txt")
        with open(problems, "w", encoding="utf-8") as fh:
            fh.write(HELLO.prompt + "\n")
        self.assertIsNone(self.run_cli("codegen", "--problems", problems, "--teacher-provider", "chatgpt",
                                       expect=1, key=None))
        self.assertEqual(self.fake.requests, [])


if __name__ == "__main__":
    unittest.main()
