"""Cross-language parity of the Rust port's tools, calculator, browsing and corrections.

Skipped when no Rust toolchain is available.  These are the checks
``tests/test_go_parity.py`` holds the Go port to - ``TestGoToolsParity`` and the
correction checks of ``TestGoParity`` / ``TestGoNegativeParity`` - held to the
Rust port, plus what Go is not asked for: the browsing tools against one fake
site on both sides, the sandbox and the uploads, the graph document after a run
of corrections byte for byte, and the two routes the frontend's AgentPanel calls,
answered by a real ``radixnet serve``.
"""

import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.rust_harness import CORPUS, assert_close, build, load_json, py, rust, serve, tmpdir  # noqa: E402

try:  # the fake website of the tools tests stands in for the internet on both sides
    from test_tools import start_site
except ImportError:
    from tests.test_tools import start_site


def setUpModule():
    build()


def compact(doc):
    import json

    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


def without_timing(doc):
    """A tool result minus how long it took, which no two runs share."""
    doc = dict(doc)
    doc.pop("seconds", None)
    if isinstance(doc.get("meta"), dict):
        doc["meta"] = {k: v for k, v in doc["meta"].items() if k != "seconds"}
    return doc


class TestRustToolsParity(unittest.TestCase):
    """The tools both sides offer, and the text format the network calls them in."""

    def setUp(self):
        self.py_model = os.path.join(tmpdir(), "tools_py.count.json")
        self.rs_model = os.path.join(tmpdir(), "tools_rs.count.json")

    def both(self, *args, expect=0):
        return py(*args, model=self.py_model, expect=expect), rust(*args, model=self.rs_model, expect=expect)

    def test_the_same_tools_with_the_same_schemas(self):
        for options in (("--offline",), (), ("--python-tool", "--upload-dir", tmpdir())):
            a, b = self.both("tools", "list", *options)
            self.assertEqual(a["tools"], b["tools"], options)  # names, descriptions, parameters, defaults and all
            self.assertEqual(a["offline"], b["offline"])
        a, b = self.both("tools", "describe", "--tool", "web_fetch")
        self.assertEqual(a, b)  # the tool and what the LLM is handed for tool calling
        self.assertEqual([t["name"] for t in b["tool"]["params"]], ["url", "max_chars"])

    def test_the_calculator_answers_the_same(self):
        for expression in ("2 * (3 + 4)", "sqrt(841)", "7 / 2", "7 // 2", "-7 % 3", "round(2.5)",
                           "round(3.14159, 2)", "min(3, 1, 2)", "sum([1, 2, 3])", "2 ^ 8", "1 < 2",
                           "abs(-3)", "int(3.9)", "float(3)", "log(100, 10)", "hypot(3, 4)", "0.1 + 0.2",
                           # where a one-pass calculator and Python part company
                           "2 ** 100", "1 or 1/0", "1, 2", "[1] + [2]", "max([1, 2], [3])", "(-8) ** (1/3)",
                           "round(2.675, 2)", "round(1234.5, -2)", "-1 % inf", "1e16", "sqrt", "0x10 + 1_000"):
            a, b = self.both("tools", "call", "--tool", "calculator", "--arg", f"expression={expression}")
            self.assertEqual((a["ok"], a["output"], a["arguments"]), (b["ok"], b["output"], b["arguments"]), expression)
        # both refuse the same expressions, in the same words, on stderr with exit 1
        for bad in ("10 / 0", "open('x')", "nosuchname", "(1", "[1 2]", "'a' * open(1)", "lambda: 1",
                    "sqrt(-1)", "min(1, [1])", "1 if 2", "~1", "x.y", "sqrt(x=1)", "[1] * 2.0"):
            a, b = self.both("tools", "call", "--tool", "calculator", "--arg", f"expression={bad}", expect=1)
            self.assertIn("calculator", b["error"], bad)
            self.assertEqual(a["error"].replace("radixnet: error: ", ""), b["error"].replace("radixnet: ", ""), bad)

    def test_a_call_is_written_and_read_the_same(self):
        """The text format is the contract: a call one side writes is the call the other reads."""
        written = 'calculator {"expression": "6*7"}'
        a, b = self.both("tools", "call", "--call", written)
        self.assertEqual(without_timing(a), without_timing(b))
        # the lenient reader: a bare value, key=value pairs and a truncated call all arrive the same
        for raw in ("calculator 5*5", "calculator expression=2+2", 'calculator {"expression": "9"',
                    "<tool>calculator  expression: '3'</tool>", 'calculator {"Expression": 4}'):
            a, b = self.both("tools", "call", "--call", raw)
            self.assertEqual((a["arguments"], a["output"]), (b["arguments"], b["output"]), raw)
        for broken in ("nosuchtool 1", "{}", "calculator"):
            a, b = self.both("tools", "call", "--call", broken, expect=1)
            self.assertEqual(a["error"].replace("radixnet: error: ", ""), b["error"].replace("radixnet: ", ""), broken)

    def test_the_sandbox_answers_the_same(self):
        """The same bootstrap in the same kind of scratch directory: the same output, the same failure."""
        base = ("tools", "call", "--python-tool", "--sandbox-timeout", 20, "--tool", "python", "--arg")
        a, b = self.both(*base, "code=print(sum(range(10)))")
        self.assertEqual((a["output"], a["meta"]["exit_code"]), (b["output"], b["meta"]["exit_code"]))
        self.assertEqual(b["output"], "45")
        a, b = self.both(*base, "code=x = 1")
        self.assertEqual(a["output"], b["output"])
        for failing in ("code=raise ValueError('boom')", "code=raise SystemExit(3)", "code=import sys; sys.exit('bye')"):
            a, b = self.both(*base, failing, expect=1)
            first = lambda doc: doc["error"].splitlines()[0].replace("radixnet: error: ", "").replace("radixnet: ", "")  # noqa: E731
            self.assertEqual(first(a), first(b), failing)

    def test_read_file_reads_the_same(self):
        uploads = os.path.join(tmpdir(), "tool-uploads")
        os.makedirs(uploads, exist_ok=True)
        with open(os.path.join(uploads, "notes.txt"), "w", encoding="utf-8") as fh:
            fh.write("﻿the cat sat on the mat\n" + "y" * 3000)
        base = ("tools", "call", "--offline", "--upload-dir", uploads, "--tool", "read_file", "--arg")
        for name in ("name=notes.txt", "name=../../notes.txt"):
            a, b = self.both(*base, name)
            self.assertEqual(without_timing(a), without_timing(b), name)
        a, b = self.both(*base, "name=nope.txt", expect=1)
        self.assertEqual(a["error"].replace("radixnet: error: ", ""), b["error"].replace("radixnet: ", ""))


class TestRustBrowsingParity(unittest.TestCase):
    """One fake site, both clients: the same page text, the same links, the same hits, the same refusals."""

    def setUp(self):
        self.site = start_site(self.addCleanup)
        self.py_model = os.path.join(tmpdir(), "web_py.count.json")
        self.rs_model = os.path.join(tmpdir(), "web_rs.count.json")

    def call(self, tool, *args, expect=0, search="/search?q={query}", extra=()):
        options = ("tools", "call", "--allow-private", "--search-url", self.site.url + search, *extra,
                   "--tool", tool, *[x for arg in args for x in ("--arg", arg)])
        return (py(*options, model=self.py_model, expect=expect),
                rust(*options, model=self.rs_model, expect=expect))

    def test_a_page_reads_the_same(self):
        for path in ("/", "/cats", "/plain", "/redirect"):
            a, b = self.call("web_fetch", f"url={self.site.url}{path}")
            self.assertEqual(without_timing(a), without_timing(b), path)
        a, b = self.call("web_fetch", f"url={self.site.url}/cats", "max_chars=100")
        self.assertEqual(a["output"], b["output"])
        self.assertTrue(b["output"].startswith("Cats — "))
        a, b = self.call("web_links", f"url={self.site.url}/")
        self.assertEqual(without_timing(a), without_timing(b))
        self.assertIn("All about cats", b["output"])

    def test_a_search_reads_the_same(self):
        a, b = self.call("web_search", "query=cats")
        self.assertEqual(without_timing(a), without_timing(b))
        self.assertIn("1. Cats — http://example.org/cats", b["output"])
        a, b = self.call("web_search", "query=cats", "limit=1")
        self.assertEqual(a["output"], b["output"])
        a, b = self.call("web_search", "query=cats and dogs", search="/search-html?q={query}")
        self.assertEqual(without_timing(a), without_timing(b))  # the DuckDuckGo wrapper unwrapped, the engine skipped

    def test_the_same_refusals(self):
        for path in ("/nope", "/loop", "/away"):
            a, b = self.call("web_fetch", f"url={self.site.url}{path}", expect=1)
            self.assertEqual(a["error"].replace("radixnet: error: ", ""), b["error"].replace("radixnet: ", ""), path)
        # the byte cap
        a, b = self.call("web_fetch", f"url={self.site.url}/big", extra=("--max-bytes", 2048))
        self.assertEqual(without_timing(a), without_timing(b))
        self.assertTrue(b["meta"]["truncated"])
        # and without --allow-private the site is refused outright, in the same words
        options = ("tools", "call", "--tool", "web_fetch", "--arg", f"url={self.site.url}/cats")
        a = py(*options, model=self.py_model, expect=1)
        b = rust(*options, model=self.rs_model, expect=1)
        self.assertEqual(a["error"].replace("radixnet: error: ", ""), b["error"].replace("radixnet: ", ""))
        self.assertIn("a loopback address", b["error"])


class TestRustCorrectionParity(unittest.TestCase):
    """The tutor's correction: the same alignment, the same edges moved, the same file afterwards."""

    PAIRS = [
        ("the cat sit on the mat", "the cat sits on the mat"),
        ("the dogs run in the mat", "the dogs run in the park"),
        ("a apple a day", "an apple a day"),
        ("we was happy", "we were happy"),
        ("the cat sat", "the cat sat on the mat"),
        ("the mat on sat cat", "the cat sat on the mat"),
        ("i have ate the bone", "i have eaten the bone"),
        ("the cat sat on the mat", "the cat sat on the mat"),
    ]

    @classmethod
    def setUpClass(cls):
        cls.py_model = os.path.join(tmpdir(), "corr_base_py.count.json")
        cls.rs_model = os.path.join(tmpdir(), "corr_base_rs.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.py_model)
        rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.rs_model)

    def test_the_alignment_is_the_same(self):
        for wrong, right in self.PAIRS + [("", "hello there"), ("he go to school", "he goes to school")]:
            with self.subTest(wrong=wrong):
                a = py("correct", "--wrong", wrong, "--right", right, "--dry-run", model=self.py_model)
                b = rust("correct", "--wrong", wrong, "--right", right, "--dry-run", model=self.rs_model)
                self.assertEqual(a, b)

    def test_corrections_move_the_same_trigram_nodes(self):
        py_path = os.path.join(tmpdir(), "corr_py.count.json")
        rs_path = os.path.join(tmpdir(), "corr_rs.count.json")
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.rs_model, rs_path)
        for wrong, right in self.PAIRS:
            a = py("correct", "--wrong", wrong, "--right", right, "--keep", 0.25, "--strength", 0.5, model=py_path)
            b = rust("correct", "--wrong", wrong, "--right", right, "--keep", 0.25, "--strength", 0.5, model=rs_path)
            for key in ("edits", "penalised", "rewarded", "kept", "wrong_chars", "right_chars", "changes",
                        "marked_correct", "marked_incorrect", "wrong", "right"):
                self.assertEqual(a[key], b[key], f"{key} for {wrong!r} -> {right!r}")
            for key in ("penalty", "reward", "loss"):
                self.assertLessEqual(abs(a[key] - b[key]), 1e-9, key)
        # not merely the same numbers: the same file, but for the cache stamp every load bumps
        a_doc, b_doc = load_json(py_path), load_json(rs_path)
        strip = lambda doc: compact({k: v for k, v in doc["graph"].items() if k != "version"})  # noqa: E731
        self.assertEqual(strip(a_doc), strip(b_doc))
        for key in ("trained_texts", "trained_chars", "feedback_passes"):
            self.assertEqual(a_doc["meta"][key], b_doc["meta"][key], key)
        for key in ("rewards_total", "penalties_total"):
            self.assertLessEqual(abs(a_doc["meta"][key] - b_doc["meta"][key]), 1e-9, key)
        a_stats = py("info", model=py_path)["stats"]
        b_stats = rust("info", model=rs_path)["stats"]
        for key in ("feedback_passes", "total_traversals", "window_traversals", "trained_texts",
                    "path_contexts", "path_judged", "path_seen", "path_correct", "path_incorrect"):
            self.assertEqual(a_stats[key], b_stats[key], key)
        # and they go on to predict the same continuations
        for prefix in ("the cat", "i have", "we w"):
            x = py("predict", "--prefix", prefix, "--length", 8, "--k", 3, model=py_path)
            y = rust("predict", "--prefix", prefix, "--length", 8, "--k", 3, model=rs_path)
            self.assertEqual(x["continuation"], y["continuation"], prefix)
            self.assertLessEqual(abs(x["cost"] - y["cost"]), 1e-9)

    def test_the_options_move_the_same_edges(self):
        """--no-count leaves the correction untraversed; --weight 0 blames nothing; --keep 1 rewards it all."""
        for options in (("--no-count",), ("--weight", 0), ("--keep", 1, "--reward", 2)):
            py_path = os.path.join(tmpdir(), "opt_py.count.json")
            rs_path = os.path.join(tmpdir(), "opt_rs.count.json")
            shutil.copy(self.py_model, py_path)
            shutil.copy(self.rs_model, rs_path)
            args = ("correct", "--wrong", "the cat sit on the mat", "--right", "the cat sits on the mat", *options)
            a, b = py(*args, model=py_path), rust(*args, model=rs_path)
            for key in ("penalised", "rewarded", "kept", "marked_correct", "marked_incorrect"):
                self.assertEqual(a[key], b[key], f"{key} with {options}")
            self.assertLessEqual(abs(a["reward"] - b["reward"]), 1e-9)
            x, y = load_json(py_path)["graph"], load_json(rs_path)["graph"]
            self.assertEqual(x["edges"]["reward"], y["edges"]["reward"], options)
            self.assertEqual(x["edges"]["count"], y["edges"]["count"], options)
            self.assertEqual(x["paths"], y["paths"], options)

    def test_a_node_is_read_the_same_way_from_both_sides(self):
        """The node ratios after a correction: both languages share a node's traffic and reward out the same way."""
        py_path = os.path.join(tmpdir(), "ratios_py.count.json")
        rs_path = os.path.join(tmpdir(), "ratios_rs.count.json")
        corpus = os.path.join(tmpdir(), "ratios_corpus.txt")
        with open(corpus, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\na cat ran to the park\nthe cat sat on the log\n")
        py("--kind", "count", "--seed", 1, "train", "--data", corpus, "--epochs", 2, model=py_path)
        rust("--seed", 1, "train", "--data", corpus, "--epochs", 2, model=rs_path)
        for path, run in ((py_path, py), (rs_path, rust)):
            run("correct", "--wrong", "the cat ran to the mat", "--right", "the cat sat on the mat", model=path)
        a = py("nodes", "--limit", 0, model=py_path)["nodes"]
        b = rust("nodes", "--limit", 0, model=rs_path)["nodes"]
        self.assertEqual(len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            self.assertEqual(x, y, f"node row {i}")
        branch = next((row for row in a if len(row["to"]) > 1 and row["out_totals"]["incorrect"]), None)
        self.assertIsNotNone(branch, "the correction should have split a branch's verdicts")

    def test_a_correction_blames_the_same_characters(self):
        """--blame teaches the negative network the same diff on both sides."""
        py_path = os.path.join(tmpdir(), "blame_py.count.json")
        rs_path = os.path.join(tmpdir(), "blame_rs.count.json")
        for path in (py_path, rs_path):
            negative = path.replace(".count.json", ".count.negative.json")
            if os.path.exists(negative):
                os.remove(negative)
        shutil.copy(self.py_model, py_path)
        shutil.copy(self.rs_model, rs_path)
        py("negative", "blame", "--text", "the cat sat here", "--reason", "other", model=py_path)
        rust("negative", "blame", "--text", "the cat sat here", "--reason", "other", model=rs_path)
        args = ("correct", "--wrong", "the cat sit on the mat", "--right", "the cat sits on the mat",
                "--blame", "--reason", "agreement", "--note", "the verb must agree")
        a, b = py(*args, model=py_path), rust(*args, model=rs_path)
        for key in ("blamed", "cleared", "reason", "severity", "edits", "wrong_chars", "right_chars", "changes"):
            self.assertEqual(a["negative"]["blamed"][key], b["negative"]["blamed"][key], key)
        self.assertEqual([r["reason"] for r in a["negative"]["reasons"]], [r["reason"] for r in b["negative"]["reasons"]])
        x = py("negative", "why", "--text", "the cat sit on the mat", model=py_path)["verdicts"][0]
        y = rust("negative", "why", "--text", "the cat sit on the mat", model=rs_path)["verdicts"][0]
        self.assertEqual([s["fragment"] for s in x["spans"]], [s["fragment"] for s in y["spans"]])
        self.assertEqual([s["reason"] for s in x["spans"]], [s["reason"] for s in y["spans"]])
        self.assertAlmostEqual(x["risk"], y["risk"], places=9)
        self.assertEqual(x["why"], y["why"])
        a_neg = load_json(py_path.replace(".count.json", ".count.negative.json"))["graph"]
        b_neg = load_json(rs_path.replace(".count.json", ".count.negative.json"))["graph"]
        self.assertEqual(a_neg["edges"]["blame"], b_neg["edges"]["blame"])
        assert_close(self, a_neg["edges"]["clear"], b_neg["edges"]["clear"])

    def test_a_word_model_corrects_word_by_word(self):
        """Under a word encoding the alignment, the blame and the rewards are in words, on both sides."""
        py_path = os.path.join(tmpdir(), "words_py.word.json")
        rs_path = os.path.join(tmpdir(), "words_rs.word.json")
        for path in (py_path, rs_path):
            if os.path.exists(path):
                os.remove(path)
        py("--kind", "count", "--encoding", "word:2:1", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2,
           model=py_path)
        rust("--encoding", "word:2:1", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=rs_path)
        for wrong, right in (("the cat sit on the mat", "the cat sits on the mat"),
                             ("he go to school", "he goes to school"), ("we was happy", "we were happy")):
            a = py("correct", "--wrong", wrong, "--right", right, "--keep", 0.25, model=py_path)
            b = rust("correct", "--wrong", wrong, "--right", right, "--keep", 0.25, model=rs_path)
            for key in ("edits", "penalised", "rewarded", "kept", "wrong_chars", "right_chars",
                        "marked_correct", "marked_incorrect"):
                self.assertEqual(a[key], b[key], f"{key} for {wrong!r} -> {right!r}")
        strip = lambda doc: compact({k: v for k, v in doc["graph"].items() if k != "version"})  # noqa: E731
        self.assertEqual(strip(load_json(py_path)), strip(load_json(rs_path)))
        # A sentence that stopped too early is blamed on its step into END.  Python's count model passes
        # len(wrong) - characters - where the word positions are meant, so under a word encoding it never
        # blames that step; the Rust port measures in units, as Go and Python's own negative network do.
        b = rust("correct", "--wrong", "the cat sat", "--right", "the cat sat on the mat", model=rs_path)
        self.assertEqual((b["penalised"], b["marked_incorrect"]), (1, 1))

    def test_the_cli_refuses_the_same_way(self):
        a = py("correct", "--wrong", " ", "--right", "", model=self.py_model, expect=1)
        b = rust("correct", "--wrong", " ", "--right", "", model=self.rs_model, expect=1)
        self.assertEqual(a["error"].replace("radixnet: error: ", ""), b["error"].replace("radixnet: ", ""))
        b = rust("correct", "--wrong", "a", "--right", "b", "--strength", -1, model=self.rs_model, expect=1)
        self.assertIn("--strength", b["error"])


class TestRustToolRoutes(unittest.TestCase):
    """The routes the frontend's AgentPanel calls, from a real `radixnet serve`."""

    @classmethod
    def setUpClass(cls):
        cls.site = start_site(cls.addClassCleanup)
        cls.model = os.path.join(tmpdir(), "routes_tools.count.json")
        cls.uploads = os.path.join(tmpdir(), "routes-uploads")
        os.makedirs(cls.uploads, exist_ok=True)
        with open(os.path.join(cls.uploads, "notes.txt"), "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\n")
        cls.search = cls.site.url + "/search?q={query}"
        cls.server = serve(cls, cls.model, "--allow-private", "--search-url", cls.search)
        cls.plain = serve(cls, cls.model, "--allow-private", "--search-url", cls.search, "--upload-dir", cls.uploads,
                          "--python-tool", "--offline")

    def test_the_tools_endpoint(self):
        status, data, _ = self.server.get("/api/tools")
        self.assertEqual(status, 200)
        self.assertEqual(data["names"], ["web_search", "web_fetch", "web_links", "calculator"])
        self.assertEqual(data["count"], 4)
        self.assertEqual(data["tools"][0]["name"], "web_search")
        self.assertIn("<tool>", data["call_format"])
        self.assertTrue(data["options"]["allow_private"])
        self.assertEqual(data["options"]["search_url"], self.search)
        self.assertFalse(data["browser"]["available"])
        # the same tools Python describes with the same settings
        expected = py("tools", "list", "--allow-private", "--search-url", self.search, model=self.model)["tools"]
        self.assertEqual(data["tools"], expected)
        status, data, _ = self.plain.get("/api/tools")
        self.assertEqual(data["names"], ["calculator", "python", "read_file"])
        self.assertEqual(data["upload_dir"], self.uploads)
        _, doc, _ = self.server.get("/api/status")
        self.assertIn("GET /api/tools", doc["routes"])
        self.assertIn("POST /api/tools/call", doc["routes"])

    def test_tool_calls(self):
        status, data, _ = self.server.post("/api/tools/call", {"tool": "calculator", "arguments": {"expression": "6*7"}})
        self.assertEqual(status, 200)
        self.assertEqual((data["ok"], data["output"], data["error"]), (True, "42", None))
        self.assertEqual(set(data), {"tool", "arguments", "ok", "output", "error", "seconds", "meta"})
        status, data, _ = self.server.post("/api/tools/call", {"call": f'web_fetch {{"url": "{self.site.url}/cats"}}'})
        self.assertEqual(status, 200, data)
        self.assertIn("A cat has four legs.", data["output"])
        status, data, _ = self.server.post("/api/tools/call", {"tool": "web_search", "arguments": {"query": "cats"}})
        self.assertEqual(len(data["meta"]["results"]), 2)
        status, data, _ = self.plain.post("/api/tools/call", {"tool": "python", "arguments": {"code": "print(2**10)"}})
        self.assertEqual((status, data["output"]), (200, "1024"))
        status, data, _ = self.plain.post("/api/tools/call", {"tool": "read_file", "arguments": {"name": "notes.txt"}})
        self.assertEqual(data["output"], "the cat sat on the mat\n")

    def test_tool_call_errors(self):
        status, data, _ = self.server.post("/api/tools/call", {"tool": "nope"})
        self.assertEqual(status, 404, data)
        status, data, _ = self.server.post("/api/tools/call", {"call": "not a call"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.server.post("/api/tools/call", {"tool": "calculator", "arguments": "text"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.server.post("/api/tools/call", {"tool": "calculator", "max_bytes": 10})
        self.assertEqual(status, 400, data)
        # a tool that fails is a 200 with ok=false: the failure is the answer, not an API error
        status, data, _ = self.server.post("/api/tools/call", {"tool": "calculator", "arguments": {"expression": "1/0"}})
        self.assertEqual(status, 200)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "division by zero")

    def test_tool_overrides(self):
        body = {"tool": "calculator", "arguments": {"expression": "1"}, "offline": True}
        status, _data, _ = self.server.post("/api/tools/call", body)
        self.assertEqual(status, 200)
        body = {"tool": "web_fetch", "arguments": {"url": "http://x/"}, "offline": True}
        status, data, _ = self.server.post("/api/tools/call", body)
        self.assertEqual(status, 404, data)  # browsing is off for this request
        body = {"tool": "web_fetch", "arguments": {"url": f"{self.site.url}/cats"}, "allow_private": False}
        status, data, _ = self.server.post("/api/tools/call", body)
        self.assertEqual(status, 200)
        self.assertFalse(data["ok"])
        self.assertIn("loopback", data["error"])


if __name__ == "__main__":
    unittest.main()
