"""Cross-language parity of the Rust port's code generation, agent loop, MCP server and browser.

Skipped when no Rust toolchain is available.  The fake servers of the Python
suites play every outside party for both ports - the scripted teacher and
content-based judge of ``test_codegen``, the criteria writer, mediator, judge,
teacher and proposer of ``test_agent``, the website of ``test_tools`` - and a
fake WebDriver on a local socket plays Chrome.  Both ports are pointed at the
same fakes, so what has to agree is:

* **the conversation** - every request the LLM (or the WebDriver) receives
  from Rust is the one it received from Python, body for body;
* **the verdicts** - the same attempts, runs, style reports, criteria, steps,
  judgements and records;
* **the effect on the models** - the positive model 2NRL trained (count and
  radix) and the negative network the failures blamed are Python's, graph for
  graph;
* **the MCP stream** - one stream of requests, notifications and garbage
  answered line for line the same by ``python -m radixnet mcp`` and
  ``radixnet mcp``;
* **the routes** - ``/api/codegen/*``, ``/api/agent/*`` and the browser fields
  of ``/api/tools`` and ``/api/status`` answer the frontend's JSON.

It mirrors ``TestGoCodeGenParity`` and ``TestGoToolsParity`` of
``test_go_parity.py`` and goes further, because the Rust port's documents are
held to Python's rather than to a handful of their keys.
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
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.rust_harness import CORPUS, ROOT, build, load_json, py, rust, serve, tmpdir  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_agent import start_fake_ollama as start_agent_llm  # noqa: E402
    from test_codegen import start_fake as start_codegen_llm  # noqa: E402
    from test_tools import start_site  # noqa: E402
except ImportError:
    from tests.test_agent import start_fake_ollama as start_agent_llm  # noqa: E402
    from tests.test_codegen import start_fake as start_codegen_llm  # noqa: E402
    from tests.test_tools import start_site  # noqa: E402


def setUpModule():
    build()  # SkipTest without a Rust toolchain


def compact(doc):
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


def graph_of(path):
    """A model file's graph as both ports write it, without the cache stamp every load bumps."""
    return compact({k: v for k, v in load_json(path)["graph"].items() if k != "version"})


def negative_beside(path):
    root, ext = os.path.splitext(path)
    return f"{root}.negative{ext}"


def fresh(*paths):
    for path in paths:
        for name in (path, negative_beside(path)):
            if os.path.exists(name):
                os.remove(name)


def without_timing(value):
    """A document with the wall-clock fields taken out, which no two runs share."""
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def close(test, a, b, path="doc"):
    """Two documents equal, keys in order, the floats to 1e-9 (a loss summed in another order may differ in its last
    bit)."""
    if isinstance(a, float) or isinstance(b, float):
        test.assertIsInstance(a, (int, float), path)
        test.assertIsInstance(b, (int, float), path)
        test.assertLessEqual(abs(a - b), 1e-9, f"{path}: {a} != {b}")
    elif isinstance(a, dict):
        test.assertIsInstance(b, dict, path)
        test.assertEqual(list(a), list(b), f"{path}: the keys, in order")
        for key in a:
            close(test, a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        test.assertIsInstance(b, list, path)
        test.assertEqual(len(a), len(b), f"{path}: the length")
        for i, (x, y) in enumerate(zip(a, b)):
            close(test, x, y, f"{path}[{i}]")
    else:
        test.assertEqual(a, b, path)


def the_run(doc):
    """What a run's document says about the run itself - not where the files went or how the backend is called."""
    doc = without_timing(doc)
    for key in ("model", "out", "saved", "stats"):
        doc.pop(key, None)
    if isinstance(doc.get("negative"), dict):
        doc["negative"] = {k: v for k, v in doc["negative"].items() if k in ("blamed", "edges", "reasons")}
    return doc


# ---------------------------------------------------------------------------
# code generation
# ---------------------------------------------------------------------------


class TestRustCodeGenParity(unittest.TestCase):
    """One fake teacher, both code-generation trainers: the same prompts, attempts, solutions and models."""

    def setUp(self):
        self.fake = start_codegen_llm(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_CODEGEN_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.problems = os.path.join(tmpdir(), "codegen_problems.txt")
        with open(self.problems, "w", encoding="utf-8") as fh:
            fh.write("Print the word hello.\n")
        # the teacher's first program does not run; its fix does
        self.script = ["```python\nprint(hello)\n```", '```python\nprint("hello")\n```']

    def prompts(self):
        """Every request so far, with the two things in them that cannot match masked out: the sandbox's scratch
        directory (a fresh one per run, named in the tracebacks) and how long a program took."""
        out = []
        for body in self.fake.requests:
            body = dict(body)
            prompt = re.sub(r"radixnet-sandbox-\w+", "radixnet-sandbox-X", body.pop("prompt", ""))
            out.append((re.sub(r"exit code (-?\d+), [\d.]+s", r"exit code \1, Ts", prompt), compact(body)))
        self.fake.requests.clear()
        return out

    def both(self, kind, *options):
        py_path = os.path.join(tmpdir(), f"codegen_py.{kind}.json")
        rs_path = os.path.join(tmpdir(), f"codegen_rs.{kind}.json")
        fresh(py_path, rs_path)
        self.fake.requests.clear()
        self.fake.solutions = list(self.script)
        a = py("--kind", kind, *options, model=py_path, env=self.env)
        py_prompts = self.prompts()
        self.fake.solutions = list(self.script)
        b = rust("--kind", kind, *options, model=rs_path, env=self.env)
        return a, b, py_prompts, self.prompts(), py_path, rs_path

    def test_both_trainers_ask_the_same_and_solve_the_same(self):
        options = ("codegen", "--problems", self.problems, "--phase", "teacher", "--teacher-attempts", 2,
                   "--sandbox-timeout", 20, "--blame")
        a, b, py_prompts, rs_prompts, py_path, rs_path = self.both("count", *options)
        # the same conversation with the teacher and the judge, prompt for prompt
        self.assertGreaterEqual(len(py_prompts), 3)  # write, fix, judge the fix
        self.assertEqual(py_prompts, rs_prompts)
        # the same attempts, records and solution, found the same way
        close(self, the_run(a), the_run(b))
        self.assertEqual(b["solutions"], {"p1": 'Print the word hello.\nprint("hello")\n'})
        self.assertEqual((a["solved"], b["solved"]), (1, 1))
        # and the same models afterwards: the 2NRL the verdicts drove, the blame the crash earned
        self.assertEqual(graph_of(py_path), graph_of(rs_path))
        self.assertEqual(graph_of(negative_beside(py_path)), graph_of(negative_beside(rs_path)))
        self.assertEqual([r["reason"] for r in b["negative"]["reasons"]], ["crash"])

    def test_a_radix_model_learns_from_the_verdicts_as_python_s_does(self):
        """2NRL goes through the kind: the sine model descends at the configured rates, as Python's does."""
        options = ("codegen", "--problems", self.problems, "--phase", "teacher", "--teacher-attempts", 2,
                   "--sandbox-timeout", 20)
        a, b, py_prompts, rs_prompts, py_path, rs_path = self.both("radix", *options)
        self.assertEqual(py_prompts, rs_prompts)
        close(self, the_run(a), the_run(b))
        self.assertEqual([r.get("action") for r in b["records"] if r["kind"] == "problem"], ["2nrl"])
        self.assertEqual(graph_of(py_path), graph_of(rs_path))

    def test_both_sandboxes_report_the_same_failure(self):
        """The sandbox is the same Python bootstrap on both sides, so a crash reads the same."""
        self.script = ["```python\nraise ValueError('boom')\n```"]
        options = ("codegen", "--problems", self.problems, "--phase", "teacher", "--teacher-attempts", 1,
                   "--sandbox-timeout", 20)
        a, b, py_prompts, rs_prompts, _, _ = self.both("count", *options)
        self.assertEqual((a["solved"], b["solved"]), (0, 0))
        self.assertEqual(py_prompts, rs_prompts)
        py_record = [r for r in a["records"] if r["kind"] == "problem"][0]
        rs_record = [r for r in b["records"] if r["kind"] == "problem"][0]
        close(self, without_timing(py_record), without_timing(rs_record))
        self.assertIn("ValueError: boom", " ".join(rs_record["issues"]))


# ---------------------------------------------------------------------------
# the agent loop
# ---------------------------------------------------------------------------


class TestRustAgentParity(unittest.TestCase):
    """One fake LLM in all its roles and one fake website: the same calls, verdicts, lessons and models."""

    def setUp(self):
        self.site = start_site(self.addCleanup)
        self.fake = start_agent_llm(self.addCleanup, self.site.url)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "PYTHONWARNINGS": "ignore"}
        self.tasks = os.path.join(tmpdir(), "agent_tasks.txt")
        with open(self.tasks, "w", encoding="utf-8") as fh:
            fh.write("How many legs does a cat have?\n")

    def requests(self):
        seen = [(method, path, compact(body)) for method, path, body in self.fake.requests]
        self.fake.requests.clear()
        self.fake.mediations = self.fake.proposals = 0
        return seen

    def both(self, kind, command, *options):
        py_path = os.path.join(tmpdir(), f"agent_py.{kind}.json")
        rs_path = os.path.join(tmpdir(), f"agent_rs.{kind}.json")
        fresh(py_path, rs_path)
        self.requests()
        common = ("--agent-model", "fake:latest", "--allow-private", "--search-url", self.site.url + "/search?q={query}",
                  "--max-steps", 2, "--model-attempts", 1, "--criteria", 2, "--neg-epochs", 1, "--pos-epochs", 1)
        a = py("--kind", kind, command, *common, *options, model=py_path, env=self.env)
        py_seen = self.requests()
        b = rust("--kind", kind, command, *common, *options, model=rs_path, env=self.env)
        return a, b, py_seen, self.requests(), py_path, rs_path

    def check(self, kind, command, *options, negative=False):
        a, b, py_seen, rs_seen, py_path, rs_path = self.both(kind, command, *options)
        self.assertGreaterEqual(len(py_seen), 3)
        self.assertEqual(py_seen, rs_seen)  # criteria, mediation, judging and teaching, body for body
        close(self, the_run(a), the_run(b))
        self.assertEqual(graph_of(py_path), graph_of(rs_path))
        if negative:
            self.assertEqual(graph_of(negative_beside(py_path)), graph_of(negative_beside(rs_path)))
        return b

    def test_both_agents_ask_the_same_and_learn_the_same(self):
        doc = self.check("count", "agent", "--tasks", self.tasks, "--blame", negative=True)
        task = [r for r in doc["records"] if r["kind"] == "task"][0]
        self.assertEqual((task["solved_by"], task["action"], task["mode"]), ("teacher", "2nrl", "fail_invert"))
        self.assertEqual(task["negative_reasons"], {"bad-call": 1})
        self.assertEqual(list(doc), ["model", "out", "config", "tools", "records", "attempts", "solved", "model_solved",
                                     "calls", "own_calls", "autonomy", "failures", "criteria", "solutions",
                                     "interrupted", "saved", "stats", "avoided", "negative"])

    def test_every_failure_mode_learns_the_same(self):
        # nothing went right: the weighted negative passes, then the inversion
        doc = self.check("count", "agent", "--tasks", self.tasks, "--no-teach")
        self.assertEqual([r["action"] for r in doc["records"] if r["kind"] == "task"], ["punish"])
        # the local variant: the failed paths lose reward, the blatant ones leave the 2NRL set
        self.check("count", "agent", "--tasks", self.tasks, "--blatant-mode", "activation")

    def test_a_radix_model_learns_from_the_judge_as_python_s_does(self):
        doc = self.check("radix", "agent", "--tasks", self.tasks, "--no-teach")
        self.assertEqual([r["action"] for r in doc["records"] if r["kind"] == "task"], ["punish"])
        self.check("radix", "agent", "--tasks", self.tasks, "--blatant-mode", "activation")

    def test_both_explorers_choose_the_same(self):
        doc = self.check("count", "explore", "--steps", 2, "--seed-url", self.site.url + "/cats", "--blame",
                         negative=True)
        self.assertEqual([r["prompt"] for r in doc["records"] if r["kind"] == "explore"],
                         ["Question 1 about cats?", "Question 2 about cats?"])
        self.assertIn("frontier", doc)
        self.assertIn("visited", doc)


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


def rpc(ident, method, **params):
    return json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})


class TestRustMcpParity(unittest.TestCase):
    """One stream, two servers: ``python -m radixnet mcp`` and ``radixnet mcp`` answer it line for line alike."""

    @classmethod
    def setUpClass(cls):
        cls.site = start_site(cls.addClassCleanup)
        cls.fake = start_agent_llm(cls.addClassCleanup, cls.site.url)
        cls.dir = tempfile.mkdtemp(prefix="radixnet-rust-mcp-")
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)
        cls.models = {}
        for kind in ("count", "radix"):
            path = os.path.join(cls.dir, f"trained.{kind}.json")
            py("--kind", kind, "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=path)
            py("negative", "blame", "--text", "the cat sat on the sky", "--reason", "false", "--severity", 2,
               model=path)
            cls.models[kind] = path

    def serve_stream(self, kind, lines, *options, no_model=False):
        """Both servers over the same stream: ``{side: (answers, stderr, LLM requests)}``."""
        env = {**os.environ, "OLLAMA_HOST": self.fake.url, "PYTHONWARNINGS": "ignore"}
        out = {}
        for side, command in (("py", [sys.executable, "-m", "radixnet"]), ("rs", [build()])):
            model = os.path.join(self.dir, f"{side}.{kind}.json")
            shutil.copyfile(self.models[kind], model)
            shutil.copyfile(negative_beside(self.models[kind]), negative_beside(model))
            self.fake.requests.clear()
            self.fake.mediations = 0
            args = [*command, "--model", model, "mcp", *options]
            proc = subprocess.run(args, input="\n".join(lines) + "\n", capture_output=True, text=True, cwd=ROOT,
                                  env=env, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            seen = [(method, path, compact(body)) for method, path, body in self.fake.requests]
            out[side] = ([json.loads(line) for line in proc.stdout.splitlines()], proc.stderr, seen)
        return out

    def assert_alike(self, out):
        (a, _, py_seen), (b, _, rs_seen) = out["py"], out["rs"]
        self.assertEqual(len(a), len(b))
        for first, second in zip(a, b):
            text = (first.get("result") or {}).get("content", [{}])[0].get("text", "")
            if text.startswith("{\n") and '"backend"' in text:  # radixnet_stats: the backend is named for what it is
                doc = [json.loads(x["result"]["content"][0]["text"]) for x in (first, second)]
                for stats in doc:
                    for key in ("backend", "device", "counting"):
                        stats.pop(key, None)
                self.assertEqual(sorted(doc[0]), sorted(doc[1]))
                close(self, {k: doc[0][k] for k in sorted(doc[0])}, {k: doc[1][k] for k in sorted(doc[1])})
                continue
            self.assertEqual(first, second)
        self.assertEqual(py_seen, rs_seen)
        return b

    def test_one_stream_is_answered_alike(self):
        lines = [
            rpc(1, "initialize", protocolVersion="2024-11-05", capabilities={}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            rpc(2, "tools/list"),
            rpc(3, "tools/call", name="calculator", arguments={"expression": "6*7"}),
            rpc(4, "tools/call", name="radixnet_predict", arguments={"prefix_text": "the cat", "length": 8}),
            rpc(5, "tools/call", name="radixnet_generate", arguments={"count": 2, "max_length": 20}),
            rpc(6, "tools/call", name="radixnet_score", arguments={"text": "the cat sat on the mat"}),
            rpc(7, "tools/call", name="radixnet_judge", arguments={"text": "the cat sat on the sky"}),
            rpc(8, "tools/call", name="radixnet_stats"),
            "{not json}", "[1, 2", '{"a" 1}', '{"a": tru}', '"unterminated', "{}", "[]", "3",
            rpc(9, "tools/call", name="nope"),
            rpc(10, "tools/call"),
            rpc(11, "tools/call", name="calculator", arguments="text"),
            rpc(12, "tools/call", name="calculator", arguments={}),
            rpc(13, "tools/call", name="calculator", arguments={"expression": "1/0"}),
            rpc(14, "nosuch"),
            json.dumps({"jsonrpc": "2.0", "id": 15, "method": "ping", "params": [1]}),
            json.dumps({"jsonrpc": "2.0", "id": "x", "method": "ping", "params": []}),
            json.dumps({"jsonrpc": "1.0", "id": 16, "method": "ping"}),
            json.dumps({"jsonrpc": "2.0", "id": 17}),
            rpc(18, "tools/call", name="radixnet_predict", arguments={"prefix_text": "é ü", "mode": "nope"}),
            rpc(19, "tools/call", name="radixnet_solve", arguments={"task": "How many legs does a cat have?",
                                                                    "max_steps": 2}),
            rpc(20, "tools/call", name="radixnet_solve", arguments={"task": "How many legs does a cat have?",
                                                                    "source": "teacher"}),
            rpc(21, "tools/call", name="web_fetch", arguments={"url": self.site.url + "/cats"}),
        ]
        for kind in ("count", "radix"):
            with self.subTest(kind=kind):
                out = self.serve_stream(kind, lines, "--allow-private", "--search-url",
                                        self.site.url + "/search?q={query}")
                answers = self.assert_alike(out)
                self.assertEqual(len(answers), len(lines) - 1)  # the notification is never answered
                by_id = {a.get("id"): a for a in answers}
                self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "radixnet")
                self.assertEqual([t["name"] for t in by_id[2]["result"]["tools"]],
                                 ["web_search", "web_fetch", "web_links", "calculator", "radixnet_predict",
                                  "radixnet_generate", "radixnet_score", "radixnet_stats", "radixnet_judge",
                                  "radixnet_solve"])
                self.assertTrue(by_id[4]["result"]["content"][0]["text"].startswith("the cat"))
                self.assertIn("false", by_id[7]["result"]["content"][0]["text"])
                self.assertIn("A cat has four legs.", by_id[20]["result"]["content"][0]["text"])
                errors = [a["error"]["message"] for a in answers if a.get("id") is None]
                self.assertEqual(errors[0], "invalid JSON: Expecting property name enclosed in double quotes: "
                                            "line 1 column 2 (char 1)")
                # nothing but the protocol on stdout; the notes on stderr
                self.assertIn("MCP server", out["rs"][1])
                self.assertIn("negative network attached", out["rs"][1])

    def test_the_flags_offer_the_same_tools(self):
        lines = [rpc(1, "tools/list")]
        for options, names in (
            (("--no-model", "--offline"), ["calculator"]),
            (("--offline", "--no-solve"), ["calculator", "radixnet_predict", "radixnet_generate", "radixnet_score",
                                           "radixnet_stats", "radixnet_judge"]),
        ):
            with self.subTest(options=options):
                answers = self.assert_alike(self.serve_stream("count", lines, *options))
                self.assertEqual([t["name"] for t in answers[0]["result"]["tools"]], names)


# ---------------------------------------------------------------------------
# the browser
# ---------------------------------------------------------------------------


PAGE = ("<html><head><title>Cats</title></head><body><p>A cat has four legs.</p><a href='/dogs'>Dogs</a>"
        "<script>draw()</script></body></html>")


class _Driver(BaseHTTPRequestHandler):
    """A WebDriver with one page in it: every request is logged, the answers are a Chrome's."""

    server: "FakeWebDriver"

    def log_message(self, *args):
        pass

    def _json(self, status, doc):
        body = json.dumps(doc).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.log.append(("GET", self.path, ""))
        self._json(200, {"value": "Cats page" if self.path.endswith("/title") else {"ready": True}})

    def do_DELETE(self):
        self.server.log.append(("DELETE", self.path, ""))
        self._json(200, {"value": None})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8")
        self.server.log.append(("POST", self.path, raw))
        body = json.loads(raw or "{}")
        if self.path == "/session":
            self._json(200, {"value": {"sessionId": "s1", "capabilities": {}}})
        elif self.path.endswith("/url"):
            self.server.at = body["url"]
            self._json(200, {"value": None})
        elif self.path.endswith("/execute/sync"):
            script = body["script"]
            value = "complete" if "readyState" in script else PAGE if "outerHTML" in script else self.server.at
            self._json(200, {"value": value})
        else:
            self._json(404, {"value": {"error": "unknown command", "message": self.path}})


class FakeWebDriver(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Driver)
        self.log = []
        self.at = ""

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def taken(self):
        seen, self.log = self.log, []
        return seen


def fake_binary(folder, name, says):
    """A 'chromedriver' or 'chrome' that only says its version - enough for the version check."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"#!/bin/sh\necho '{says}'\n")
    os.chmod(path, 0o755)
    return path


class TestRustBrowserParity(unittest.TestCase):
    """``--browser`` on both sides against one fake WebDriver: the same requests, the same pages."""

    @classmethod
    def setUpClass(cls):
        cls.driver = FakeWebDriver()
        threading.Thread(target=cls.driver.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        cls.addClassCleanup(cls.driver.server_close)
        cls.addClassCleanup(cls.driver.shutdown)
        cls.dir = tempfile.mkdtemp(prefix="radixnet-rust-browser-")
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)
        cls.browser_env = {
            "RADIXNET_CHROMEDRIVER": fake_binary(cls.dir, "chromedriver", "ChromeDriver 120.0.6099.109 (abc)"),
            "RADIXNET_CHROME": fake_binary(cls.dir, "chrome", "Chromium 120.0.6099.0"),
            "RADIXNET_WEBDRIVER": cls.driver.url + "/",
        }
        cls.env = {**os.environ, **cls.browser_env, "PYTHONWARNINGS": "ignore"}
        cls.model = os.path.join(cls.dir, "m.count.json")

    def both(self, *args, env=None, expect=0):
        env = env or self.env
        self.driver.taken()
        a = py(*args, model=self.model, env=env, expect=expect)
        py_log = self.driver.taken()
        b = rust(*args, model=self.model, env=env, expect=expect)
        return a, b, py_log, self.driver.taken()

    def test_the_browser_is_described_alike(self):
        a, b, _, _ = self.both("tools", "browser")
        self.assertEqual(a, b)
        self.assertTrue(b["available"])
        self.assertEqual(b["chrome_version"], "Chromium 120.0.6099.0")
        # and the machine's own binaries (or their absence) are read the same way
        env = {k: v for k, v in os.environ.items() if k not in self.browser_env}
        a, b, _, _ = self.both("tools", "browser", env=env)
        self.assertEqual(a, b)
        # a mismatch refuses --browser, in the same words
        mismatched = {**self.env, "RADIXNET_CHROME": fake_binary(self.dir, "chrome119", "Google Chrome 119.0.1")}
        a, b, _, _ = self.both("tools", "list", "--browser", env=mismatched, expect=1)
        self.assertIn("--browser: chromedriver 120 cannot drive Chrome 119", b["error"])
        self.assertEqual(a["error"].replace("radixnet: error: ", ""), b["error"].replace("radixnet: ", ""))

    def test_the_web_tools_read_the_page_chrome_drew(self):
        for tool in ("web_fetch", "web_links"):
            with self.subTest(tool=tool):
                a, b, py_log, rs_log = self.both("tools", "call", "--browser", "--allow-private", "--tool", tool,
                                                 "--arg", "url=http://127.0.0.1:9/cats")
                a.pop("seconds", None)
                b.pop("seconds", None)
                self.assertEqual(a, b)
                self.assertEqual(py_log, rs_log)  # the session, the page, the scripts and the title, body for body
                self.assertEqual([path for _, path, _ in rs_log][:2], ["/session", "/session/s1/url"])
                self.assertIn("A cat has four legs." if tool == "web_fetch" else "1. Dogs", b["output"])

    def test_a_server_draws_its_pages_in_one_browser(self):
        from tests.test_api import start_server

        patcher = unittest.mock.patch.dict(os.environ, self.browser_env)
        patcher.start()
        self.addCleanup(patcher.stop)
        python, _server, _service = start_server(
            self.addCleanup, model_path=os.path.join(self.dir, "py.count.json"), kind="count",
            tool_options={"browser": True, "allow_private": True},
        )
        server = serve(self, os.path.join(self.dir, "rs.count.json"), "--browser", "--allow-private", "--log", "warn",
                       env=self.env)
        _, a, _ = python.get("/api/tools")
        _, b, _ = server.get("/api/tools")
        self.assertEqual(a["browser"], b["browser"])
        self.assertEqual(a["options"], b["options"])
        self.assertTrue(b["options"]["browser"])
        _, a, _ = python.get("/api/status")
        _, b, _ = server.get("/api/status")
        self.assertEqual(a["tools"], b["tools"])
        call = {"tool": "web_fetch", "arguments": {"url": "http://127.0.0.1:9/cats"}}
        for _ in range(2):  # the second page reuses the session the first opened
            self.driver.taken()
            status, a, _ = python.post("/api/tools/call", call)
            py_log = self.driver.taken()
            status2, b, _ = server.post("/api/tools/call", call)
            self.assertEqual((status, status2), (200, 200))
            close(self, without_timing(a), without_timing(b))
            self.assertEqual(py_log, self.driver.taken())
        self.assertIn("A cat has four legs.", b["output"])
        # a request may still fetch the plain way
        self.driver.taken()
        status, b, _ = server.post("/api/tools/call", {**call, "browser": False})
        self.assertEqual(status, 200)
        self.assertFalse(b["ok"])  # nothing listens on port 9, and the browser was not asked
        self.assertEqual(self.driver.taken(), [])


# ---------------------------------------------------------------------------
# the routes
# ---------------------------------------------------------------------------


class TestRustAgentRoutes(unittest.TestCase):
    """``radixnet serve`` answers the code-generation and agent routes as the Python server does."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.site = start_site(cls.addClassCleanup)
        cls.agent_llm = start_agent_llm(cls.addClassCleanup, cls.site.url)
        cls.codegen_llm = start_codegen_llm(cls.addClassCleanup)
        cls.dir = tempfile.mkdtemp(prefix="radixnet-rust-agent-routes-")
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)
        trained = os.path.join(cls.dir, "trained.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=trained)
        cls.paths = {}
        for side in ("py", "rs"):
            cls.paths[side] = os.path.join(cls.dir, side, "m.count.json")
            os.makedirs(os.path.dirname(cls.paths[side]))
            shutil.copyfile(trained, cls.paths[side])
        cls.search = cls.site.url + "/search?q={query}"
        cls.python, _server, cls.service = start_server(
            cls.addClassCleanup, model_path=cls.paths["py"], kind="count", seed=1,
            tool_options={"allow_private": True, "search_url": cls.search},
        )
        cls.rust = serve(cls, cls.paths["rs"], "--seed", "1", "--log", "warn", "--allow-private", "--search-url",
                         cls.search)

    def setUp(self):
        self.sync()

    def sync(self):
        """Hands the Python server's model to the Rust one, so both start every test from the same network."""
        path = os.path.join(self.dir, "sync.count.json")
        status, doc, _ = self.python.post("/api/save", {"path": path})
        self.assertEqual(status, 200, doc)
        status, doc, _ = self.rust.post("/api/load", {"path": path})
        self.assertEqual(status, 200, doc)

    def seen(self, fake):
        if fake is self.agent_llm:  # the proposer numbers its questions: every side starts from the first
            fake.mediations = fake.proposals = 0
        if fake is self.codegen_llm:
            bodies = [compact(body) for body in fake.requests]
        else:
            bodies = [(method, path, compact(body)) for method, path, body in fake.requests]
        fake.requests.clear()
        return [re.sub(r"exit code (-?\d+), [\d.]+s", r"exit code \1, Ts",
                       re.sub(r"radixnet-sandbox-\w+", "radixnet-sandbox-X", str(b))) for b in bodies]

    def both(self, method, path, body, fake, script=None):
        """The same request to both servers: ``((status, doc) python, (status, doc) rust)`` and the LLM calls."""
        fake.requests.clear()
        if script is not None:
            fake.solutions = list(script)
        a = self.python.request(method, path, body=body)[:2]
        py_seen = self.seen(fake)
        if script is not None:
            fake.solutions = list(script)
        b = self.rust.client.request(method, path, body=body)[:2]
        return a, b, py_seen, self.seen(fake)

    def wait(self, client, timeout=120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, job, _ = client.get("/api/job")
            self.assertEqual(status, 200)
            if job and job.get("state") != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("the job did not finish")

    def job_alike(self, path, body, fake, script=None, history=None):
        """A 202 job on both servers: the same acceptance, the same LLM calls, the same records."""
        if script is not None:
            fake.solutions = list(script)
        fake.requests.clear()
        status, a, _ = self.python.post(path, body)
        self.assertEqual(status, 202, a)
        py_job = self.wait(self.python)
        py_seen = self.seen(fake)
        if script is not None:
            fake.solutions = list(script)
        status, b, _ = self.rust.post(path, body)
        self.assertEqual(status, 202, b)
        self.assertEqual(list(a), list(b))
        rs_job = self.wait(self.rust.client)
        self.assertEqual(rs_job["error"], None, rs_job)
        self.assertEqual(py_job["type"], rs_job["type"])
        self.assertEqual(py_seen, self.seen(fake))
        close(self, without_timing(py_job["history"]), without_timing(rs_job["history"]), "history")
        if history:
            _, py_history, _ = self.python.get(history)
            status, rs_history, _ = self.rust.get(history)
            self.assertEqual(status, 200)
            mine = without_timing(rs_history["history"][-len(rs_job["history"]):])
            self.assertEqual(mine, without_timing(rs_job["history"]))
            close(self, without_timing(py_history["history"][-len(mine):]), mine, "kept history")
        return a, b, rs_job

    def test_a_program_is_run_and_judged_alike(self):
        for body in ({"code": 'print("hello")', "expected_output": "hello"},
                     {"code": "def sloppyName():\n  return 1\nprint(x)\n"},
                     {"code": "print(1)", "tests": "assert 1 == 1"}):
            (s1, a), (s2, b), _, _ = self.both("POST", "/api/codegen/run", body, self.codegen_llm)
            self.assertEqual((s1, s2), (200, 200), b)
            self.assertEqual(list(a), ["run", "style", "verdict"])
            a["run"] = {k: v for k, v in a["run"].items() if k not in ("seconds", "stderr", "error")}
            b["run"] = {k: v for k, v in b["run"].items() if k not in ("seconds", "stderr", "error")}
            close(self, a, b)

    def test_a_problem_is_solved_alike(self):
        body = {"problem": "Print the word hello.", "source": "teacher", "attempts": 2, "url": self.codegen_llm.url,
                "teacher_model": "fake:latest", "sandbox_timeout": 20}
        (s1, a), (s2, b), py_seen, rs_seen = self.both(
            "POST", "/api/codegen/solve", body, self.codegen_llm,
            script=["```python\nprint(hello)\n```", '```python\nprint("hello")\n```'])
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(list(a), list(b))
        for doc in (a, b):
            for attempt in doc["attempts"]:
                attempt["run"] = {k: v for k, v in attempt["run"].items() if k not in ("stderr", "error")}
        close(self, without_timing(a), without_timing(b))
        self.assertTrue(b["correct"])

    def test_a_codegen_job_runs_and_its_history_is_kept(self):
        body = {"problems": ["Print the word hello."], "phases": "teacher", "teacher_attempts": 2, "blame": True,
                "url": self.codegen_llm.url, "teacher_model": "fake:latest", "sandbox_timeout": 20,
                "neg_epochs": 1, "pos_epochs": 1}
        a, b, job = self.job_alike("/api/codegen/start", body, self.codegen_llm,
                                   script=["```python\nprint(hello)\n```", '```python\nprint("hello")\n```'],
                                   history="/api/codegen/history")
        self.assertEqual(list(b), ["job", "problems", "config", "sandbox"])
        close(self, a["config"], b["config"], "config")
        self.assertEqual(b["job"]["type"], "codegen")
        self.assertIn("problem", [r["kind"] for r in job["history"]])

    def test_criteria_and_a_solved_task_come_back_alike(self):
        body = {"tasks": ["How many legs does a cat have?", {"prompt": "And a dog?", "criteria": ["says four"]}],
                "url": self.agent_llm.url, "model": "fake:latest"}
        (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/agent/criteria", body, self.agent_llm)
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(a, b)
        for source in ("teacher", "model"):
            body = {"tasks": ["How many legs does a cat have?"], "source": source, "url": self.agent_llm.url,
                    "agent_model": "fake:latest", "max_steps": 2, "criteria": 2}
            (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/agent/solve", body, self.agent_llm)
            self.assertEqual((s1, s2), (200, 200), b)
            self.assertEqual(py_seen, rs_seen, source)
            close(self, without_timing(a), without_timing(b), source)
        self.assertEqual(list(b), ["task", "source", "criteria", "attempt", "transcript", "correct", "gap", "frontier",
                                   "tools"])

    def test_an_agent_job_and_an_exploration_run_alike(self):
        body = {"tasks": ["How many legs does a cat have?"], "url": self.agent_llm.url, "agent_model": "fake:latest",
                "max_steps": 2, "model_attempts": 1, "criteria": 2, "neg_epochs": 1, "pos_epochs": 1, "blame": True}
        a, b, job = self.job_alike("/api/agent/start", body, self.agent_llm, history="/api/agent/history")
        close(self, a["config"], b["config"], "config")
        self.assertEqual(b["job"]["type"], "agent")
        self.assertIn("task", [r["kind"] for r in job["history"]])
        self.sync()
        body = {"steps": 1, "seed_urls": [self.site.url + "/cats"], "url": self.agent_llm.url,
                "agent_model": "fake:latest", "max_steps": 2, "model_attempts": 1, "criteria": 2, "neg_epochs": 1,
                "pos_epochs": 1}
        a, b, job = self.job_alike("/api/agent/explore", body, self.agent_llm)
        self.assertEqual(b["job"]["type"], "explore")
        self.assertIn("explore", [r["kind"] for r in job["history"]])

    def test_refusals_match(self):
        for path, body in (
            ("/api/codegen/start", {}),
            ("/api/codegen/start", {"problems": ["x"], "phases": "sideways"}),
            ("/api/codegen/start", {"problems": ["x"], "rounds": 0}),
            ("/api/codegen/solve", {"problem": "x", "source": "nobody"}),
            ("/api/codegen/run", {}),
            ("/api/codegen/run", {"code": 5}),
            ("/api/agent/start", {}),
            ("/api/agent/start", {"tasks": ["a"], "phase": "sideways"}),
            ("/api/agent/start", {"tasks": [{"prompt": "a", "id": "t2"}, "b"]}),
            ("/api/agent/start", {"tasks": ["a"], "mediation": "sometimes"}),
            ("/api/agent/explore", {"steps": -1}),
            ("/api/agent/criteria", {"tasks": []}),
            ("/api/agent/solve", {"tasks": ["a", "b"]}),
            ("/api/agent/solve", {"tasks": ["a"], "source": "x"}),
        ):
            with self.subTest(path=path, body=body):
                (s1, a), (s2, b), _, _ = self.both("POST", path, body, self.agent_llm)
                self.assertEqual(s1, s2, (a, b))
                self.assertEqual(s1, 400)
                self.assertEqual(a["error"], b["error"])
        # nothing runs, so both histories answer (and nothing was started)
        for path in ("/api/codegen/history", "/api/agent/history"):
            status, doc, _ = self.rust.get(path)
            self.assertEqual(status, 200)
            self.assertIn("history", doc)


if __name__ == "__main__":
    unittest.main()
