"""Tests for ``radixnet.tools``: the call format, the registry, browsing and the built-in tools.

A fake website (standard library ``http.server``) stands in for the internet:
it serves a couple of HTML pages, a JSON search endpoint, a redirect chain and
a page that is larger than the byte cap.  Nothing here reaches the network —
:class:`WebClient` is pointed at ``127.0.0.1`` with ``allow_private=True``,
which is exactly the guard the tests also check refuses by default.
"""

import contextlib
import json
import os
import socket
import sys
import threading
import unittest
import unittest.mock
import urllib.request
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
# A page as real sites write them: a <meta charset> before the title (which once made every page read
# as empty), a banner, menus, a language list, hidden parts, a navbox and a footer around the article.
ARTICLE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cat - Encyclopedia</title>
<link rel="stylesheet" href="/style.css">
<script>var config = {"title": "Cat"};</script>
</head>
<body>
<a class="skip" href="#content">Jump to content</a>
<header class="site-header">
  <a href="/" class="logo"><svg><title>Encyclopedia logo</title><path d="M0 0"/></svg></a>
  <form role="search" action="/search"><input name="q" placeholder="Search"><button>Search</button></form>
  <nav aria-label="Personal tools"><a href="/login">Log in</a> <a href="/signup">Create account</a></nav>
</header>
<nav id="sidebar"><ul><li><a href="/main">Main page</a></li><li><a href="/random">Random article</a></li></ul></nav>
<main id="content">
  <header class="title-bar">
    <h1>Cat</h1>
    <div class="languages">
      <span aria-hidden="true">3 languages</span>
      <ul>
        <li><a href="https://de.example.org/wiki/Katze" hreflang="de">Deutsch</a></li>
        <li><a href="https://fr.example.org/wiki/Chat" hreflang="fr">Fran&ccedil;ais</a></li>
        <li><a href="https://es.example.org/wiki/Gato" hreflang="es">Espa&ntilde;ol</a></li>
      </ul>
    </div>
  </header>
  <p>The <b>cat</b> (<i>Felis catus</i>) is a small
     <a href="/wiki/Domestication">domesticated</a> carnivorous
     <a href="/wiki/Mammal">mammal</a>.</p>
  <table class="infobox"><tr><th>Kingdom:</th><td><a href="/wiki/Animal">Animalia</a></td></tr><tr><th>Legs:</th><td>4</td></tr></table>
  <h2 id="Anatomy">Anatomy <span class="edit">[<a href="/w/edit?section=1">edit</a>]</span></h2>
  <p>A cat has four legs and a tail; see <a href="/wiki/Mammal">mammal</a> and <a href="#Anatomy">above</a>.</p>
  <div hidden>Hidden until the reader asks.</div>
  <div style="display: none">Invisible.</div>
  <input type="hidden" name="token" value="x" hidden>
  <pre>meow()
purr()</pre>
  <div role="navigation" class="navbox"><a href="/wiki/Lion">Lion</a> <a href="/wiki/Tiger">Tiger</a></div>
</main>
<footer><a href="/privacy">Privacy policy</a><p>Text is available under a licence.</p></footer>
</body>
</html>
"""
# DuckDuckGo's HTML results, as its /html/ page writes them: an ad, then every hit linked three times
# (its title, its address, a snippet), each through the duckduckgo.com/l/ redirect.
DDG_RESULTS = """<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
<meta name="referrer" content="origin">
<title>cats at DuckDuckGo</title>
<link rel="stylesheet" href="/dist/h.css" type="text/css">
</head>
<body class="body--html">
<div id="header" class="header">
  <form name="x" action="/html/" method="post"><input name="q" type="text" value="cats">
  <select name="kl"><option value="">All Regions</option><option value="uk-en">UK</option></select></form>
</div>
<div id="links" class="results">
  <div class="result results_links result--ad">
    <h2 class="result__title"><a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=shop.example&amp;u3=x">Cat food deals</a></h2>
    <a class="result__snippet" href="https://duckduckgo.com/y.js?ad_domain=shop.example&amp;u3=x">Buy cat food online.</a>
  </div>
  <div class="result results_links results_links_deep web-result">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1a">Cat - Encyclopedia</a>
    </h2>
    <div class="result__extras__url">
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1a"><img alt="" src="//external-content.duckduckgo.com/ip3/en.example.org.ico"></a>
      <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1a">en.example.org/wiki/Cat</a>
    </div>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1a">The <b>cat</b> is a small domesticated carnivorous mammal with four legs.</a>
  </div>
  <div class="result results_links results_links_deep web-result">
    <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvets.example.com%2Fcat-anatomy&amp;rut=2b">Cat anatomy explained</a></h2>
    <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvets.example.com%2Fcat-anatomy&amp;rut=2b">vets.example.com/cat-anatomy</a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvets.example.com%2Fcat-anatomy&amp;rut=2b">Cats walk on four legs, and their skeleton has 230 bones.</a>
  </div>
  <div class="nav-link"><form action="/html/" method="post"><input type="submit" value="Next"></form></div>
  <div class="feedback-btn"><a href="//duckduckgo.com/feedback.html">Feedback</a></div>
</div>
</body>
</html>
"""
# What DuckDuckGo answers a client it takes for a bot with: HTTP 202 and a challenge instead of results.
DDG_BLOCKED = (
    '<!DOCTYPE html><html><head><meta charset="utf-8"><title>DuckDuckGo</title></head><body>'
    '<form id="challenge-form" action="//duckduckgo.com/anomaly.js?sv=html&amp;cc=botnet" method="POST">'
    '<div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>'
    '<div class="anomaly-modal__description">Please complete the following challenge to confirm this search was '
    "made by a human.</div></form></body></html>"
)
CAPTCHA = (
    "<html><head><title>Sorry</title></head><body><p>Our systems have detected unusual traffic from your "
    "computer network.</p><a href='/help'>Why did this happen?</a></body></html>"
)
NO_RESULTS = '<html><head><meta charset="utf-8"><title>No results</title></head><body><p>No results.</p></body></html>'
# Wikipedia's search API (generator=search, formatversion=2): the pages come out of rank, "index" ranks them.
WIKI_API = json.dumps({"batchcomplete": True, "continue": {"gsroffset": 2, "continue": "gsroffset||"}, "query": {"pages": [
    {"pageid": 20, "ns": 0, "title": "Cat anatomy", "index": 2, "fullurl": "https://en.example.org/wiki/Cat_anatomy",
     "extract": "Cat anatomy comprises the anatomic studies of the visible parts of the body of a domestic cat."},
    {"pageid": 10, "ns": 0, "title": "Cat", "index": 1, "fullurl": "https://en.example.org/wiki/Cat",
     "extract": "The cat (Felis catus) is a small domesticated carnivorous mammal."},
]}})
OPENSEARCH = json.dumps(["cat", ["Cat", "Catalonia"], ["", "A region of Spain"],
                         ["https://en.example.org/wiki/Cat", "https://en.example.org/wiki/Catalonia"]])
# A JSON answer the way Chrome draws one: the document is a page, the answer is the text of its <pre>.
JSON_AS_PAGE = (
    '<html><head><meta name="color-scheme" content="light dark"></head><body><pre style="word-wrap: break-word">'
    + json.dumps({"results": [{"title": "Cats", "url": "http://example.org/cats", "content": "cats have four legs"}]})
    .replace("&", "&amp;").replace("<", "&lt;")
    + "</pre></body></html>"
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
        elif path == "/article":
            self._send(200, ARTICLE, "text/html; charset=utf-8")
        elif path == "/ddg":
            self._send(200, DDG_RESULTS)
        elif path == "/ddg-blocked":
            self._send(202, DDG_BLOCKED)
        elif path == "/search-203":  # a proxy that rewrote the answer: still an answer
            self._send(203, SEARCH_JSON, "application/json")
        elif path == "/captcha":
            self._send(200, CAPTCHA)
        elif path == "/no-results":
            self._send(200, NO_RESULTS)
        elif path == "/wiki-api":
            self._send(200, WIKI_API, "application/json; charset=utf-8")
        elif path == "/opensearch":
            self._send(200, OPENSEARCH, "application/x-suggestions+json")
        elif path == "/json-page":
            self._send(200, JSON_AS_PAGE)
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


@contextlib.contextmanager
def _no_dns():
    """Every name lookup fails, as it does in a process whose DNS goes through a proxy."""
    def fail(*args, **kwargs):
        raise OSError(-2, "Name or service not known")

    with unittest.mock.patch.object(socket, "getaddrinfo", fail):
        yield


@contextlib.contextmanager
def _resolves_to(address):
    """Every name resolves to one address (a DNS sinkhole, or an internal answer)."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    infos = [(family, socket.SOCK_STREAM, 6, "", (address, 0))]
    with unittest.mock.patch.object(socket, "getaddrinfo", lambda *a, **k: infos):
        yield


@contextlib.contextmanager
def _proxies(mapping):
    with unittest.mock.patch.object(urllib.request, "getproxies", lambda: dict(mapping)), \
         unittest.mock.patch.object(urllib.request, "proxy_bypass", lambda host: False):
        yield


@contextlib.contextmanager
def _no_proxies():
    with _proxies({}):
        yield


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

    def test_a_meta_tag_does_not_swallow_the_page(self):
        # <meta> and <link> are void: counting one as an open drop hid every page that has one, links and all
        page = html_to_text(
            '<html><head><meta charset="utf-8"><link rel=x href=y><title>T &amp; U</title></head>'
            '<body><p>Hello &amp; bye&nbsp;now</p><a href="/q?a=1&amp;b=2">Link &lt;1&gt;</a></body></html>'
        )
        self.assertEqual(page["title"], "T & U")
        self.assertEqual(page["text"], "Hello & bye now\nLink <1>")
        self.assertEqual(page["links"], [("Link <1>", "/q?a=1&b=2")])

    def test_the_main_text_comes_first(self):
        page = html_to_text(ARTICLE)
        self.assertEqual(page["title"], "Cat - Encyclopedia")  # not the logo's <svg><title>
        self.assertEqual(page["text"].split("\n"), [
            "Cat",
            "The cat (Felis catus) is a small domesticated carnivorous mammal.",  # one line, as the reader sees it
            "Kingdom: Animalia",                                                  # cells set apart, not glued
            "Legs: 4",
            "Anatomy [edit]",
            "A cat has four legs and a tail; see mammal and above.",
            "meow()", "purr()",                                                   # a <pre> keeps its lines
            "Jump to content",                                                    # outside <main>: after it
            "Deutsch", "Français", "Español",                                     # a run of links: a menu, last
        ])
        # the banner, the menus, the search form, the hidden parts, the navbox and the footer are gone
        for gone in ("Log in", "Main page", "Search", "3 languages", "Hidden", "Invisible", "Lion", "Privacy", "licence"):
            self.assertNotIn(gone, page["text"])
        # the links in the running text come first, the menus' after them
        self.assertEqual([href for _text, href in page["links"]], [
            "/wiki/Domestication", "/wiki/Mammal", "/wiki/Animal", "/w/edit?section=1", "/wiki/Mammal", "#Anatomy",
            "#content", "https://de.example.org/wiki/Katze", "https://fr.example.org/wiki/Chat",
            "https://es.example.org/wiki/Gato",
        ])

    def test_reading_order_can_be_turned_off(self):
        page = html_to_text(ARTICLE, reading_order=False)
        self.assertEqual(page["text"].split("\n")[:3], ["Jump to content", "Cat", "Deutsch"])
        self.assertEqual(page["links"][0], ("Jump to content", "#content"))

    def test_what_is_never_shown_is_dropped(self):
        text = html_to_text(
            "<p>a</p><div hidden><div>nested</div>still hidden</div><p>b</p>"
            "<span style='DISPLAY: none'>x</span><span style='visibility:hidden'>y</span><span aria-hidden=true>z</span>"
            "<div role='navigation'>menu</div><div role='Search box'>find</div><dialog>cookies?</dialog>"
            "<p>c<button>Click</button><select><option>One</option></select><textarea>typed</textarea></p>"
        )["text"]
        self.assertEqual(text, "a\nb\nc")

    def test_what_hiding_could_run_away_with_is_kept(self):
        # a void element cannot hide what follows it, and neither can one whose end tag may be left out
        self.assertEqual(html_to_text('<p>before</p><input name="m" hidden><img hidden src=x><p>after</p>')["text"],
                         "before\nafter")
        self.assertEqual(html_to_text("<ul><li hidden>one<li>two</ul><p>three")["text"], "one\ntwo\nthree")
        self.assertEqual(html_to_text('<div hidden="until-found">findable</div>')["text"], "findable")

    def test_the_banner_goes_but_an_articles_header_stays(self):
        page = html_to_text("<header>Site name</header><article><header><h1>Story</h1></header><p>Body.</p></article>")
        self.assertEqual(page["text"], "Story\nBody.")
        self.assertEqual(html_to_text("<section><header>Part one</header><p>x</p></section>")["text"], "Part one\nx")

    def test_an_unclosed_head_ends_where_the_body_begins(self):
        page = html_to_text("<html><head><title>T</title><body><p>Body.</p>")
        self.assertEqual((page["title"], page["text"]), ("T", "Body."))

    def test_isolated_link_lines_stay_in_place(self):
        # one link on its own line is part of the text; only a run of them is a menu
        text = html_to_text("<p>First.</p><p><a href='/a'>A link</a></p><p>Last.</p>")["text"]
        self.assertEqual(text, "First.\nA link\nLast.")
        text = html_to_text("<ul><li><a href='/a'>One</a><li><a href='/b'>Two</a></ul><p>Body.</p>")["text"]
        self.assertEqual(text, "Body.\nOne\nTwo")

    def test_the_main_region_can_be_a_role(self):
        page = html_to_text("<div>Promo</div><div role='main'><div>Body<div hidden>x</div></div>more</div><div>After</div>")
        self.assertEqual(page["text"], "Body\nmore\nPromo\nAfter")


class WebClientGuardTests(unittest.TestCase):
    def setUp(self):
        self.web = WebClient()

    def test_private_and_loopback_are_refused(self):
        for url in ("http://127.0.0.1/", "http://10.0.0.1/", "http://192.168.1.1/", "http://localhost/"):
            with self.assertRaises(WebError) as ctx:
                self.web.check(url)
            self.assertIn("private", str(ctx.exception))

    def test_unresolvable_host_is_refused_when_nothing_else_can_resolve_it(self):
        with _no_dns(), _no_proxies():
            with self.assertRaises(WebError) as ctx:
                self.web.check("https://example.com/")
        self.assertIn("cannot be resolved here", str(ctx.exception))

    def test_a_host_this_process_cannot_resolve_still_goes_through_a_proxy(self):
        # behind a proxy (Docker, a corporate network, a sandbox) the proxy resolves the name and
        # this process often has no DNS at all; refusing there would make browsing impossible
        with _no_dns(), _proxies({"https": "http://127.0.0.1:3128"}):
            self.assertEqual(self.web.check("https://example.com/"), "https://example.com/")

    def test_each_refusal_says_which_kind_of_address_it_is(self):
        for address, expected in (
            ("0.0.0.0", "an unspecified address"),  # what a DNS sinkhole answers with
            ("127.0.0.1", "a loopback address"),
            ("169.254.169.254", "a link-local address"),  # the cloud metadata endpoint
            ("10.1.2.3", "a private address"),
        ):
            with _resolves_to(address):
                with self.assertRaises(WebError) as ctx:
                    self.web.check("https://example.com/")
            message = str(ctx.exception)
            self.assertIn(expected, message)
            self.assertIn(address, message)
            self.assertIn("allow_private", message)

    def test_allow_private_skips_the_check_entirely(self):
        web = WebClient(allow_private=True)
        with _resolves_to("127.0.0.1"):
            self.assertEqual(web.check("https://example.com/"), "https://example.com/")

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

    def test_page_links_are_absolute_unique_and_leave_the_page(self):
        page = self.web.page(self.site.url + "/article")
        self.assertEqual(page["title"], "Cat - Encyclopedia")
        self.assertTrue(page["text"].startswith("Cat\nThe cat (Felis catus) is a small domesticated"))
        self.assertEqual([href for _text, href in page["links"]], [
            self.site.url + "/wiki/Domestication", self.site.url + "/wiki/Mammal", self.site.url + "/wiki/Animal",
            self.site.url + "/w/edit?section=1", "https://de.example.org/wiki/Katze",
            "https://fr.example.org/wiki/Chat", "https://es.example.org/wiki/Gato",
        ])  # the second link to /wiki/Mammal, and "#Anatomy" / "#content" back into the page, are gone

    def _engines(self, *paths):
        return local_web(self.site, search_url=" ".join(self.site.url + path + "?q={query}" for path in paths))

    def test_search_reads_a_duckduckgo_page(self):
        results = self._engines("/ddg").search("cats")
        self.assertEqual([(r.title, r.url, r.snippet) for r in results], [
            ("Cat - Encyclopedia", "https://en.example.org/wiki/Cat",
             "The cat is a small domesticated carnivorous mammal with four legs."),
            ("Cat anatomy explained", "https://vets.example.com/cat-anatomy",
             "Cats walk on four legs, and their skeleton has 230 bones."),
        ])  # the ad and the feedback link are DuckDuckGo's own, the address link is no snippet

    def test_search_falls_back_when_an_engine_refuses(self):
        results = self._engines("/ddg-blocked", "/wiki-api").search("cat legs")
        self.assertEqual([(r.title, r.url) for r in results], [
            ("Cat", "https://en.example.org/wiki/Cat"), ("Cat anatomy", "https://en.example.org/wiki/Cat_anatomy"),
        ])  # ranked by "index", not in the order the API lists the pages
        self.assertEqual(results[0].snippet, "The cat (Felis catus) is a small domesticated carnivorous mammal.")
        self.assertEqual([p.split("?")[0] for p in self.site.requests], ["/ddg-blocked", "/wiki-api"])
        # the first engine that has results ends the search
        self._engines("/ddg", "/wiki-api").search("cats")
        self.assertEqual(self.site.requests[-1].split("?")[0], "/ddg")

    def test_a_single_engine_that_refuses_is_an_error(self):
        with self.assertRaises(WebError) as ctx:
            self._engines("/ddg-blocked").search("cats")
        self.assertEqual(str(ctx.exception),
                         "127.0.0.1 answered HTTP 202 instead of results - it may be turning automated searches away")
        with self.assertRaises(WebError) as ctx:
            self._engines("/captcha").search("cats")
        self.assertIn("answered with a CAPTCHA instead of results", str(ctx.exception))

    def test_every_engine_refusing_is_an_error_naming_each(self):
        with self.assertRaises(WebError) as ctx:
            self._engines("/ddg-blocked", "/captcha", "/nope").search("cats")
        message = str(ctx.exception)
        self.assertTrue(message.startswith("no search engine answered: "), message)
        for part in ("HTTP 202", "CAPTCHA", "HTTP 404"):
            self.assertIn(part, message)

    def test_only_a_202_is_a_refusal(self):
        results = self._engines("/search-203").search("cats")
        self.assertEqual([r.title for r in results], ["Cats", "Dogs"])

    def test_an_engine_that_finds_nothing_is_no_results(self):
        self.assertEqual(self._engines("/ddg-blocked", "/no-results").search("cats"), [])
        self.assertEqual(self._engines("/no-results").search("cats"), [])

    def test_search_reads_opensearch_and_json_drawn_as_a_page(self):
        results = self._engines("/opensearch").search("cat")
        self.assertEqual([(r.title, r.url, r.snippet) for r in results], [
            ("Cat", "https://en.example.org/wiki/Cat", ""),
            ("Catalonia", "https://en.example.org/wiki/Catalonia", "A region of Spain"),
        ])
        results = self._engines("/json-page").search("cats")  # what a JSON API looks like through --browser
        self.assertEqual([(r.title, r.url) for r in results], [("Cats", "http://example.org/cats")])

    @unittest.skipIf(os.environ.get("RADIXNET_SEARCH_URL", "").strip(), "$RADIXNET_SEARCH_URL overrides the default")
    def test_the_default_search_engines(self):
        from radixnet.tools import DEFAULT_SEARCH_ENGINES

        self.assertEqual(WebClient().search_url.split(), list(DEFAULT_SEARCH_ENGINES))
        self.assertIn("duckduckgo.com", DEFAULT_SEARCH_ENGINES[0])
        self.assertIn("wikipedia.org/w/api.php", DEFAULT_SEARCH_ENGINES[-1])
        self.assertEqual(WebClient(search_url="  http://a/?q={query}\n http://b/  ").search_url,
                         "http://a/?q={query} http://b/")


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

    def test_web_fetch_reads_a_real_page_article_first(self):
        result = self.box.call("web_fetch", {"url": self.site.url + "/article", "max_chars": 120})
        self.assertTrue(result.ok, result.error)
        self.assertEqual(
            result.output,
            f"Cat - Encyclopedia — {self.site.url}/article\n"
            "Cat\nThe cat (Felis catus) is a small domesticated carnivorous mammal.\nKingdom: Animalia\nLegs: 4\n"
            "Anatomy [edit]\nA cat has ...",
        )
        self.assertEqual(result.meta["link_count"], 7)
        self.assertEqual(result.meta["links"][0], {"text": "domesticated", "url": self.site.url + "/wiki/Domestication"})
        links = self.box.call("web_links", {"url": self.site.url + "/article", "limit": 2})
        self.assertEqual(links.output, f"1. domesticated — {self.site.url}/wiki/Domestication\n"
                                       f"2. mammal — {self.site.url}/wiki/Mammal")

    def test_web_search_carries_the_snippets(self):
        box = default_toolbox(local_web(self.site, search_url=self.site.url + "/ddg?q={query}"))
        result = box.call("web_search", {"query": "cats"})
        self.assertEqual(result.output.split("\n")[0], "1. Cat - Encyclopedia — https://en.example.org/wiki/Cat — "
                                                       "The cat is a small domesticated carnivorous mammal with four legs.")
        blocked = default_toolbox(local_web(self.site, search_url=self.site.url + "/ddg-blocked?q={query}"))
        result = blocked.call("web_search", {"query": "cats"})
        self.assertFalse(result.ok)
        self.assertIn("HTTP 202", result.error)

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
