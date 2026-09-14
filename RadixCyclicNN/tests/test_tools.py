"""Tests for ``radixnet.tools``: the call format, the registry, browsing and the built-in tools.

A fake website (standard library ``http.server``) stands in for the internet:
it serves a couple of HTML pages, a JSON search endpoint, a redirect chain and
a page that is larger than the byte cap.  Nothing here reaches the network —
:class:`WebClient` is pointed at ``127.0.0.1`` with ``allow_private=True``,
which is exactly the guard the tests also check refuses by default.
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.tools import (  # noqa: E402
    ANSWER_OPEN,
    CALL_OPEN,
    Param,
    Tool,
    ToolBox,
    ToolError,
    ToolResult,
    WebClient,
    WebError,
    answer_text,
    call_text,
    default_toolbox,
    find_answer,
    format_observation,
    html_to_text,
    parse_arguments,
    parse_call,
    result_text,
    safe_eval,
    task_header,
    transcript_text,
)

HOME = (
    "<html><head><title> Example Home </title><style>p{color:red}</style>"
    "<script>alert('x')</script></head><body><h1>Welcome</h1><p>The cat sat on the mat.</p>"
    "<a href='/cats'>All about cats</a> <a href='mailto:x@y.z'>mail</a>"
    "<a href='http://127.0.0.1:1/other'>absolute</a></body></html>"
)
CATS = "<html><head><title>Cats</title></head><body><p>A cat has four legs.</p><br><p>And a tail.</p></body></html>"
SEARCH_JSON = json.dumps({"results": [
    {"title": "Cats", "url": "http://example.org/cats", "content": "cats have four legs"},
    {"title": "Dogs", "url": "http://example.org/dogs", "content": "dogs bark"},
]})
SEARCH_HTML = (
    "<html><body><a href='/internal'>engine link</a>"
    "<a href='https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fhit'>A wrapped hit</a>"
    "<a href='https://example.com/plain'>A plain hit</a></body></html>"
)


class _SiteHandler(BaseHTTPRequestHandler):
    server: "FakeSite"

    def log_message(self, *args):  # silence
        pass

    def _send(self, status, body, content_type="text/html", headers=()):
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?")[0]
        self.server.requests.append(self.path)
        if path == "/":
            self._send(200, HOME)
        elif path == "/cats":
            self._send(200, CATS)
        elif path == "/search":
            self._send(200, SEARCH_JSON, "application/json")
        elif path == "/search-html":
            self._send(200, SEARCH_HTML)
        elif path == "/plain":
            self._send(200, "just text", "text/plain")
        elif path == "/big":
            self._send(200, "x" * 200_000)
        elif path == "/redirect":
            self._send(302, "", headers=[("Location", "/cats")])
        elif path == "/loop":
            self._send(302, "", headers=[("Location", "/loop")])
        elif path == "/away":  # a redirect to a host that is not allowed
            self._send(302, "", headers=[("Location", "file:///etc/passwd")])
        else:
            self._send(404, "<html><body>not found</body></html>")


class FakeSite(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _SiteHandler)
        self.requests = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def start_site(add_cleanup):
    site = FakeSite()
    thread = threading.Thread(target=site.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    add_cleanup(site.server_close)
    add_cleanup(site.shutdown)
    return site


def local_web(site, **kwargs):
    """A :class:`WebClient` pointed at the fake site (private addresses allowed)."""
    kwargs.setdefault("allow_private", True)
    kwargs.setdefault("search_url", site.url + "/search?q={query}")
    return WebClient(**kwargs)


# ---------------------------------------------------------------------------
# the text format
# ---------------------------------------------------------------------------


class FormatTests(unittest.TestCase):
    def test_call_text_is_one_json_line(self):
        self.assertEqual(call_text("web_fetch", {"url": "u"}), '<tool>web_fetch {"url": "u"}</tool>\n')
        self.assertEqual(call_text("now"), "<tool>now {}</tool>\n")

    def test_result_text_collapses_and_clips(self):
        self.assertEqual(result_text("a\n  b\tc"), "<result>a b c</result>\n")
        clipped = result_text("x" * 50, limit=10)
        self.assertEqual(clipped, "<result>" + "x" * 10 + " ...</result>\n")
        self.assertEqual(result_text("x" * 50, limit=None), "<result>" + "x" * 50 + "</result>\n")

    def test_answer_and_header(self):
        self.assertEqual(answer_text(" four\nlegs "), "<answer>four legs</answer>\n")
        self.assertEqual(task_header("  how   many?\n"), "TASK: how many?\n")

    def test_find_answer(self):
        self.assertEqual(find_answer("noise <answer> 42 </answer> more"), "42")
        self.assertEqual(find_answer("<answer>truncated"), "truncated")  # the generator stopped early
        self.assertIsNone(find_answer("nothing here"))
        self.assertIsNone(find_answer("<answer>   </answer>"))

    def test_format_observation_marks_failures(self):
        ok = ToolResult("t", {}, True, output="fine")
        bad = ToolResult("t", {}, False, error="boom")
        self.assertEqual(format_observation(ok), "<result>fine</result>\n")
        self.assertEqual(format_observation(bad), "<result>ERROR: boom</result>\n")

    def test_transcript_is_one_training_text(self):
        steps = [("calculator", {"expression": "1+1"}, ToolResult("calculator", {}, True, output="2"))]
        text = transcript_text("What is 1+1?", steps, "two")
        self.assertEqual(
            text,
            'TASK: What is 1+1?\n<tool>calculator {"expression": "1+1"}</tool>\n'
            "<result>2</result>\n<answer>two</answer>\n",
        )
        self.assertIn(CALL_OPEN, text)
        self.assertIn(ANSWER_OPEN, text)
        self.assertEqual(transcript_text("q", []), "TASK: q\n")  # no calls, no answer


class ParseCallTests(unittest.TestCase):
    def setUp(self):
        self.box = default_toolbox(offline=True)

    def test_json_arguments(self):
        call = parse_call('<tool>calculator {"expression": "1+1"}</tool>', self.box)
        self.assertTrue(call.ok)
        self.assertEqual((call.name, call.arguments), ("calculator", {"expression": "1+1"}))

    def test_bare_value_for_a_single_argument_tool(self):
        call = parse_call("<tool>calculator 5*5</tool>", self.box)
        self.assertEqual(call.arguments, {"expression": "5*5"})

    def test_key_value_pairs(self):
        call = parse_call("<tool>calculator expression=2+2</tool>", self.box)
        self.assertEqual(call.arguments, {"expression": "2+2"})

    def test_unterminated_call_is_still_read(self):
        call = parse_call('<tool>calculator {"expression": "9"}', self.box)
        self.assertTrue(call.ok)
        self.assertEqual(call.arguments, {"expression": "9"})

    def test_truncated_json_is_repaired_to_the_last_brace(self):
        self.assertEqual(parse_arguments('{"a": "b"} trailing junk'), {"a": "b"})

    def test_no_call_at_all(self):
        self.assertIsNone(parse_call("just some text", self.box))
        self.assertIsNone(parse_call("", self.box))

    def test_unknown_tool_is_reported_not_dropped(self):
        call = parse_call("<tool>nosuchtool {}</tool>", self.box)
        self.assertFalse(call.ok)
        self.assertIn("unknown tool", call.error)
        self.assertEqual(call.name, "nosuchtool")

    def test_a_bare_value_needs_a_single_argument_tool(self):
        box = ToolBox([Tool("pair", "Two.", (Param("a"), Param("b")), lambda a, b: a + b)])
        call = parse_call("<tool>pair ???</tool>", box)
        self.assertFalse(call.ok)
        self.assertIn("cannot read the arguments", call.error)
        with self.assertRaises(ToolError):
            parse_arguments("???", box.get("pair"))

    def test_garbage_without_a_name(self):
        call = parse_call("<tool>{}</tool>", self.box)
        self.assertFalse(call.ok)
        self.assertIn("no tool name", call.error)

    def test_first_call_wins(self):
        call = parse_call("<tool>calculator 1</tool><tool>calculator 2</tool>", self.box)
        self.assertEqual(call.arguments, {"expression": "1"})


# ---------------------------------------------------------------------------
# tools and the registry
# ---------------------------------------------------------------------------


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tool = Tool(
            "demo", "A demo.",
            (Param("text", "string", "some text"),
             Param("count", "integer", "how many", required=False, default=2),
             Param("flag", "boolean", "on or off", required=False),
             Param("mode", "string", "which mode", required=False, enum=("a", "b"))),
            lambda text, count=2, flag=False, mode="a": text * count,
        )

    def test_signature_and_schema(self):
        self.assertEqual(self.tool.signature(), "demo(text: string, [count: integer], [flag: boolean], [mode: string])")
        schema = self.tool.schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "demo")
        self.assertEqual(schema["function"]["parameters"]["required"], ["text"])
        self.assertEqual(schema["function"]["parameters"]["properties"]["mode"]["enum"], ["a", "b"])

    def test_coerce_types_defaults_and_aliases(self):
        values = self.tool.coerce({"text": 5, "count": "3", "flag": "yes", "MODE": "b"})
        self.assertEqual(values, {"text": "5", "count": 3, "flag": True, "mode": "b"})
        self.assertEqual(self.tool.coerce({"text": "x"}), {"text": "x", "count": 2})  # the default fills in

    def test_coerce_rejects_missing_bad_and_out_of_enum(self):
        with self.assertRaises(ToolError) as ctx:
            self.tool.coerce({})
        self.assertIn("missing argument(s) text", str(ctx.exception))
        with self.assertRaises(ToolError):
            self.tool.coerce({"text": "x", "count": "many"})
        with self.assertRaises(ToolError):
            self.tool.coerce({"text": "x", "mode": "c"})
        with self.assertRaises(ToolError):
            self.tool.coerce(["not", "a", "dict"])

    def test_unknown_arguments_are_dropped(self):
        self.assertEqual(self.tool.coerce({"text": "x", "nonsense": 1})["text"], "x")

    def test_bad_param_type(self):
        with self.assertRaises(ValueError):
            Param("x", "list")


class ToolBoxTests(unittest.TestCase):
    def setUp(self):
        self.box = ToolBox()
        self.box.register(Tool("echo", "Echo.", (Param("text"),), lambda text: text.upper()))

    def test_registry(self):
        self.assertIn("echo", self.box)
        self.assertEqual(self.box.names(), ["echo"])
        self.assertEqual(len(self.box), 1)
        self.assertEqual([t.name for t in self.box], ["echo"])
        self.assertEqual(self.box.describe()[0]["name"], "echo")
        self.assertEqual(self.box.schemas()[0]["function"]["name"], "echo")
        self.assertIn("echo(text: string) — Echo.", self.box.catalogue())
        self.assertTrue(self.box.remove("echo"))
        self.assertFalse(self.box.remove("echo"))

    def test_register_validates(self):
        with self.assertRaises(TypeError):
            self.box.register("not a tool")
        with self.assertRaises(ValueError):
            self.box.register(Tool("bad name!", "x"))

    def test_get_unknown(self):
        with self.assertRaises(ToolError):
            self.box.get("nope")

    def test_call_succeeds(self):
        result = self.box.call("echo", {"text": "hi"})
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "HI")
        self.assertEqual(self.box.calls, 1)
        self.assertIn("tool", result.to_dict())

    def test_a_failing_tool_never_raises(self):
        def boom(text):
            raise RuntimeError("kaboom")

        self.box.register(Tool("boom", "Fails.", (Param("text"),), boom))
        result = self.box.call("boom", {"text": "x"})
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError: kaboom", result.error)

    def test_unknown_tool_and_bad_arguments_become_results(self):
        self.assertFalse(self.box.call("missing").ok)
        self.assertIn("missing argument(s)", self.box.call("echo", {}).error)

    def test_no_handler(self):
        self.box.register(Tool("empty", "Nothing."))
        self.assertIn("no handler", self.box.call("empty").error)

    def test_run_a_broken_call(self):
        call = parse_call("<tool>unknown x</tool>", self.box)
        result = self.box.run(call)
        self.assertFalse(result.ok)
        self.assertIn("unknown tool", result.error)

    def test_handler_may_return_meta(self):
        self.box.register(Tool("meta", "Meta.", (), lambda: ("out", {"n": 1})))
        result = self.box.call("meta")
        self.assertEqual((result.output, result.meta), ("out", {"n": 1}))


# ---------------------------------------------------------------------------
# HTML and browsing
# ---------------------------------------------------------------------------


class HtmlTests(unittest.TestCase):
    def test_title_text_and_links(self):
        page = html_to_text(HOME)
        self.assertEqual(page["title"], "Example Home")
        self.assertIn("The cat sat on the mat.", page["text"])
        self.assertNotIn("alert", page["text"])  # <script> dropped
        self.assertNotIn("color:red", page["text"])  # <style> dropped
        self.assertIn(("All about cats", "/cats"), page["links"])

    def test_blocks_become_lines(self):
        self.assertEqual(html_to_text("<p>a</p><p>b</p>")["text"], "a\nb")
        self.assertEqual(html_to_text("a<br>b")["text"], "a\nb")

    def test_broken_markup_does_not_raise(self):
        self.assertIn("hello", html_to_text("<p>hello<<<>")["text"])
        self.assertEqual(html_to_text("")["text"], "")


class WebClientGuardTests(unittest.TestCase):
    def setUp(self):
        self.web = WebClient()

    def test_private_and_loopback_are_refused(self):
        for url in ("http://127.0.0.1/", "http://10.0.0.1/", "http://192.168.1.1/", "http://localhost/"):
            with self.assertRaises(WebError) as ctx:
                self.web.check(url)
            self.assertIn("private", str(ctx.exception))

    def test_unresolvable_host_is_refused(self):
        with self.assertRaises(WebError):
            self.web.check("http://no-such-host.invalid/")

    def test_scheme_and_credentials(self):
        with self.assertRaises(WebError):
            self.web.check("file:///etc/passwd")
        with self.assertRaises(WebError):
            self.web.check("ftp://example.com/x")
        with self.assertRaises(WebError):
            self.web.check("http://user:pw@example.com/")
        with self.assertRaises(WebError):
            self.web.check("")

    def test_a_scheme_is_added(self):
        self.assertEqual(self.web.check("example.com/x"), "https://example.com/x")

    def test_construction_validates(self):
        with self.assertRaises(ValueError):
            WebClient(timeout=0)
        with self.assertRaises(ValueError):
            WebClient(max_bytes=10)


class WebClientTests(unittest.TestCase):
    def setUp(self):
        self.site = start_site(self.addCleanup)
        self.web = local_web(self.site)

    def test_page(self):
        page = self.web.page(self.site.url + "/")
        self.assertEqual(page["title"], "Example Home")
        self.assertEqual(page["status"], 200)
        self.assertIn("The cat sat on the mat.", page["text"])
        hrefs = [href for _text, href in page["links"]]
        self.assertIn(self.site.url + "/cats", hrefs)  # made absolute
        self.assertNotIn("mailto:x@y.z", hrefs)  # only http(s) links are kept

    def test_plain_text_and_missing_page(self):
        self.assertEqual(self.web.page(self.site.url + "/plain")["text"], "just text")
        with self.assertRaises(WebError) as ctx:
            self.web.page(self.site.url + "/nope")
        self.assertIn("404", str(ctx.exception))

    def test_byte_cap(self):
        web = local_web(self.site, max_bytes=2048)
        response = web.fetch(self.site.url + "/big")
        self.assertTrue(response["truncated"])
        self.assertEqual(response["bytes"], 2048)

    def test_redirects_are_followed_and_re_checked(self):
        self.assertEqual(self.web.page(self.site.url + "/redirect")["title"], "Cats")
        with self.assertRaises(WebError) as ctx:
            self.web.page(self.site.url + "/loop")
        self.assertIn("redirect loop", str(ctx.exception))
        with self.assertRaises(WebError) as ctx:
            self.web.page(self.site.url + "/away")  # the new location must pass the guard too
        self.assertIn("only http and https", str(ctx.exception))

    def test_search_from_json(self):
        results = self.web.search("cats")
        self.assertEqual([r.title for r in results], ["Cats", "Dogs"])
        self.assertEqual(results[0].url, "http://example.org/cats")
        self.assertEqual(results[0].snippet, "cats have four legs")
        self.assertEqual(len(self.web.search("cats", limit=1)), 1)

    def test_search_from_html_unwraps_and_skips_the_engine(self):
        web = local_web(self.site, search_url=self.site.url + "/search-html?q={query}")
        urls = [r.url for r in web.search("cats")]
        self.assertIn("https://example.net/hit", urls)  # the DuckDuckGo redirect wrapper is unwrapped
        self.assertIn("https://example.com/plain", urls)
        self.assertNotIn(self.site.url + "/internal", urls)  # links back into the engine are dropped

    def test_empty_query(self):
        with self.assertRaises(WebError):
            self.web.search("  ")


# ---------------------------------------------------------------------------
# the built-in tools
# ---------------------------------------------------------------------------


class SafeEvalTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(safe_eval("2 * (3 + 4)"), 14)
        self.assertEqual(safe_eval("sqrt(841)"), 29.0)
        self.assertEqual(safe_eval("2^10"), 1024)  # ^ is read as a power
        self.assertEqual(safe_eval("max(1, 2)"), 2)

    def test_refuses_everything_else(self):
        for expression in ("__import__('os')", "open('x')", "[].__class__", "x = 1", "lambda: 1", "'a' * 3", "e.real"):
            with self.assertRaises(ToolError):
                safe_eval(expression)

    def test_arithmetic_errors(self):
        with self.assertRaises(ToolError):
            safe_eval("1/0")
        with self.assertRaises(ToolError):
            safe_eval("")
        with self.assertRaises(ToolError):
            safe_eval("9" * 600)


class DefaultToolboxTests(unittest.TestCase):
    def setUp(self):
        self.site = start_site(self.addCleanup)
        self.box = default_toolbox(local_web(self.site))

    def test_the_default_set(self):
        self.assertEqual(self.box.names(), ["web_search", "web_fetch", "web_links", "calculator"])
        self.assertEqual(default_toolbox(offline=True).names(), ["calculator"])
        self.assertTrue(self.box.get("web_fetch").network)
        self.assertFalse(self.box.get("calculator").network)

    def test_web_fetch(self):
        result = self.box.call("web_fetch", {"url": self.site.url + "/cats"})
        self.assertTrue(result.ok)
        self.assertIn("A cat has four legs.", result.output)
        self.assertTrue(result.output.startswith("Cats — "))
        self.assertEqual(result.meta["status"], 200)

    def test_web_fetch_reports_the_links_for_the_frontier(self):
        result = self.box.call("web_fetch", {"url": self.site.url + "/"})
        self.assertEqual(result.meta["link_count"], 2)
        self.assertIn({"text": "All about cats", "url": self.site.url + "/cats"}, result.meta["links"])

    def test_web_fetch_clips(self):
        result = self.box.call("web_fetch", {"url": self.site.url + "/cats", "max_chars": 100})
        self.assertLess(len(result.output), 250)

    def test_web_links_and_search(self):
        links = self.box.call("web_links", {"url": self.site.url + "/"})
        self.assertIn("All about cats", links.output)
        search = self.box.call("web_search", {"query": "cats"})
        self.assertIn("1. Cats — http://example.org/cats", search.output)
        self.assertEqual(len(search.meta["results"]), 2)

    def test_a_refused_address_is_a_readable_failure(self):
        box = default_toolbox(WebClient())  # the default guard: no private addresses
        result = box.call("web_fetch", {"url": self.site.url + "/"})
        self.assertFalse(result.ok)
        self.assertIn("private", result.error)

    def test_calculator(self):
        self.assertEqual(self.box.call("calculator", {"expression": "6*7"}).output, "42")
        self.assertIn("division by zero", self.box.call("calculator", {"expression": "1/0"}).error)

    def test_extra_tools(self):
        extra = Tool("hello", "Greets.", (), lambda: "hi")
        self.assertIn("hello", default_toolbox(offline=True, extra=[extra]).names())


class ReadFileToolTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.dir, True)
        with open(os.path.join(self.dir, "notes.txt"), "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\n")
        self.box = default_toolbox(offline=True, upload_dir=self.dir)

    def test_reads_an_uploaded_file(self):
        result = self.box.call("read_file", {"name": "notes.txt"})
        self.assertTrue(result.ok)
        self.assertIn("the cat sat", result.output)
        self.assertEqual(result.meta["name"], "notes.txt")

    def test_missing_file_lists_what_there_is(self):
        result = self.box.call("read_file", {"name": "nope.txt"})
        self.assertFalse(result.ok)
        self.assertIn("notes.txt", result.error)

    def test_no_path_traversal(self):
        result = self.box.call("read_file", {"name": "../../etc/passwd"})
        self.assertFalse(result.ok)
        self.assertIn("no uploaded file named 'passwd'", result.error)
        self.assertFalse(self.box.call("read_file", {"name": "  "}).ok)

    def test_clips_to_max_chars(self):
        with open(os.path.join(self.dir, "big.txt"), "w", encoding="utf-8") as fh:
            fh.write("y" * 5000)
        result = self.box.call("read_file", {"name": "big.txt", "max_chars": 100})
        self.assertTrue(result.output.endswith(" ..."))


class PythonToolTests(unittest.TestCase):
    def setUp(self):
        from radixnet.codegen import Sandbox

        self.box = default_toolbox(offline=True, sandbox=Sandbox(timeout=20, isolate_network=False))

    def test_runs_a_program(self):
        result = self.box.call("python", {"code": "print(6 * 7)"})
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output, "42")

    def test_a_failing_program_is_a_readable_failure(self):
        result = self.box.call("python", {"code": "raise SystemExit(3)"})
        self.assertFalse(result.ok)
        self.assertIn("the program failed", result.error)


if __name__ == "__main__":
    unittest.main()
