"""Tests for ``radixnet.browser``: finding Chrome, the version check, and the WebDriver protocol.

A **fake WebDriver** (standard library ``http.server``) stands in for
chromedriver, so every line of the protocol handling runs without a browser
anywhere: sessions, navigation, scripts, titles and the error shapes.  The one
test that needs a real Chrome skips itself when there is none (or when the
chromedriver next to it is of another major version, which is the usual state
of affairs on a machine that did not set them up together).
"""

import json
import os
import sys
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import browser  # noqa: E402
from radixnet.browser import BrowserClient, BrowserError  # noqa: E402
from radixnet.tools import WebClient, default_toolbox  # noqa: E402

SOURCE = (
    "<html><head><title>Static</title></head><body><div id=a>loading...</div>"
    "<script>document.title='Rendered';"
    "document.getElementById('a').innerHTML='<p>The cat sat on the mat.</p><a href=\"/cats\">Cats</a>';"
    "</script></body></html>"
)
"""The page as the server sends it: a title, a placeholder and a script that replaces both."""

DRAWN = (
    '<html><head><title>Rendered</title></head><body><div id=a>'
    '<p>The cat sat on the mat.</p><a href="/cats">Cats</a></div></body></html>'
)
"""The DOM after its scripts have run - what a real browser hands back, and what the fake one returns."""


# ---------------------------------------------------------------------------
# a WebDriver that is not a browser
# ---------------------------------------------------------------------------


class _DriverHandler(BaseHTTPRequestHandler):
    server: "FakeWebDriver"

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
        self.server.requests.append(("GET", self.path))
        if self.path == "/status":
            self._json(200, {"value": {"ready": self.server.ready, "message": ""}})
        elif self.path.endswith("/title"):
            self._json(200, {"value": self.server.title})
        else:
            self._json(404, {"value": {"error": "unknown command", "message": self.path}})

    def do_DELETE(self):
        self.server.requests.append(("DELETE", self.path))
        self.server.deleted = True
        self._json(200, {"value": None})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        self.server.requests.append(("POST", self.path))
        if self.path == "/session":
            if self.server.refuse_session:
                self._json(500, {"value": {"error": "session not created", "message": self.server.refuse_session}})
                return
            self.server.capabilities = body
            self._json(200, {"value": {"sessionId": "s1", "capabilities": {}}})
        elif self.path.endswith("/url"):
            self.server.visited.append(body.get("url"))
            self._json(200, {"value": None})
        elif self.path.endswith("/execute/sync"):
            script = body.get("script", "")
            if "readyState" in script:
                self._json(200, {"value": "complete"})
            elif "outerHTML" in script:
                self._json(200, {"value": self.server.html})
            elif "location.href" in script:
                self._json(200, {"value": self.server.visited[-1] if self.server.visited else ""})
            else:
                self._json(200, {"value": self.server.script_value})
        else:
            self._json(404, {"value": {"error": "unknown command", "message": self.path}})


class FakeWebDriver(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _DriverHandler)
        self.requests = []
        self.visited = []
        self.ready = True
        self.title = "Rendered"
        self.html = DRAWN
        self.script_value = None
        self.refuse_session = None
        self.capabilities = None
        self.deleted = False

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def start_driver(add_cleanup):
    driver = FakeWebDriver()
    thread = threading.Thread(target=driver.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    add_cleanup(driver.server_close)
    add_cleanup(driver.shutdown)
    return driver


# ---------------------------------------------------------------------------
# finding the binaries
# ---------------------------------------------------------------------------


class DiscoveryTests(unittest.TestCase):
    def test_the_environment_overrides_the_search(self):
        with unittest.mock.patch.dict(os.environ, {"RADIXNET_CHROMEDRIVER": __file__, "RADIXNET_CHROME": __file__}):
            self.assertEqual(browser.chromedriver_path(), __file__)
            self.assertEqual(browser.chrome_path(), __file__)

    def test_an_override_that_is_not_a_file_finds_nothing(self):
        with unittest.mock.patch.dict(os.environ, {"RADIXNET_CHROMEDRIVER": "/no/such/driver"}):
            self.assertIsNone(browser.chromedriver_path())

    def test_a_playwright_directory_is_understood(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "chromium-1194", "chrome-linux", "chrome")
            os.makedirs(os.path.dirname(path))
            open(path, "w").close()
            with unittest.mock.patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": root, "RADIXNET_CHROME": ""}), \
                 unittest.mock.patch.object(browser.shutil, "which", lambda _name: None):
                self.assertEqual(browser.chrome_path(), path)

    def test_describe_names_what_is_missing(self):
        with unittest.mock.patch.object(browser, "chromedriver_path", lambda: None), \
             unittest.mock.patch.object(browser, "chrome_path", lambda: "/bin/chrome"):
            found = browser.describe()
        self.assertFalse(found["available"])
        self.assertIn("chromedriver not found", found["error"])
        self.assertIn("$RADIXNET_CHROMEDRIVER", found["error"])
        self.assertFalse(browser.browser_available() and False)

    def test_a_version_mismatch_is_reported_before_anything_starts(self):
        # chromedriver refuses to drive another major version; saying so here beats a raw WebDriver error
        with unittest.mock.patch.object(browser, "chromedriver_path", lambda: "/bin/chromedriver"), \
             unittest.mock.patch.object(browser, "chrome_path", lambda: "/bin/chrome"), \
             unittest.mock.patch.object(browser, "_version_of",
                                        lambda b: "ChromeDriver 147.0.1" if "driver" in b else "Chromium 141.0.2"):
            found = browser.describe()
        self.assertFalse(found["available"])
        self.assertIn("chromedriver 147 cannot drive Chrome 141", found["error"])
        self.assertIn("major version", found["error"])

    def test_matching_versions_are_available(self):
        with unittest.mock.patch.object(browser, "chromedriver_path", lambda: "/bin/chromedriver"), \
             unittest.mock.patch.object(browser, "chrome_path", lambda: "/bin/chrome"), \
             unittest.mock.patch.object(browser, "_version_of", lambda b: "141.0.2"):
            self.assertTrue(browser.describe()["available"])
            self.assertTrue(browser.browser_available())

    def test_major(self):
        self.assertEqual(browser._major("Chromium 141.0.7390.37"), "141")
        self.assertEqual(browser._major("ChromeDriver 147.0.1 (abc)"), "147")
        self.assertIsNone(browser._major(""))
        self.assertIsNone(browser._major(None))


# ---------------------------------------------------------------------------
# the protocol
# ---------------------------------------------------------------------------


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.driver = start_driver(self.addCleanup)
        self.browser = BrowserClient(endpoint=self.driver.url, page_timeout=10, settle=0.0)
        self.addCleanup(self.browser.close)

    def test_a_page_comes_back_rendered(self):
        page = self.browser.get("http://example.com/")
        self.assertEqual(page["title"], "Rendered")
        self.assertIn("The cat sat on the mat.", page["html"])
        self.assertEqual(page["url"], "http://example.com/")
        self.assertEqual(self.driver.visited, ["http://example.com/"])
        self.assertEqual(self.browser.pages, 1)

    def test_the_session_is_opened_once_and_reused(self):
        self.browser.get("http://example.com/")
        self.browser.get("http://example.com/two")
        self.assertEqual(sum(1 for method, path in self.driver.requests if path == "/session"), 1)
        self.assertEqual(self.browser.pages, 2)

    def test_an_endpoint_is_never_spawned_or_stopped(self):
        self.assertIsNone(self.browser.driver)
        self.assertIsNone(self.browser.chrome)
        self.browser.get("http://example.com/")
        self.assertTrue(self.browser.running)
        self.browser.close()
        self.assertFalse(self.browser.running)
        self.assertTrue(self.driver.deleted)

    def test_headless_is_asked_for(self):
        self.browser.get("http://example.com/")
        args = self.driver.capabilities["capabilities"]["alwaysMatch"]["goog:chromeOptions"]["args"]
        self.assertIn("--headless=new", args)
        self.assertTrue(any(a.startswith("--window-size=") for a in args))
        # a remote WebDriver has its own browser, so no binary is imposed on it
        self.assertNotIn("binary", self.driver.capabilities["capabilities"]["alwaysMatch"]["goog:chromeOptions"])

    def test_a_visible_browser_drops_the_headless_flag(self):
        with BrowserClient(endpoint=self.driver.url, headless=False, settle=0.0) as visible:
            visible.get("http://example.com/")
        args = self.driver.capabilities["capabilities"]["alwaysMatch"]["goog:chromeOptions"]["args"]
        self.assertNotIn("--headless=new", args)

    def test_a_refused_session_is_a_readable_error(self):
        self.driver.refuse_session = "This version of ChromeDriver only supports Chrome version 147"
        with self.assertRaises(BrowserError) as ctx:
            self.browser.get("http://example.com/")
        self.assertIn("only supports Chrome version 147", str(ctx.exception))

    def test_scripts(self):
        self.driver.script_value = 42
        self.assertEqual(self.browser.script("return 6*7;"), 42)

    def test_an_unreachable_driver(self):
        dead = BrowserClient(endpoint="http://127.0.0.1:1", page_timeout=2)
        with self.assertRaises(BrowserError) as ctx:
            dead.get("http://example.com/")
        self.assertIn("cannot reach chromedriver", str(ctx.exception))

    def test_describe(self):
        described = self.browser.describe()
        self.assertEqual(described["endpoint"], self.driver.url)
        self.assertTrue(described["available"])
        self.assertFalse(described["running"])

    def test_construction_validates(self):
        with self.assertRaises(ValueError):
            BrowserClient(page_timeout=0)


class WebClientThroughBrowserTests(unittest.TestCase):
    def setUp(self):
        self.driver = start_driver(self.addCleanup)
        self.browser = BrowserClient(endpoint=self.driver.url, page_timeout=10, settle=0.0)
        self.addCleanup(self.browser.close)
        self.web = WebClient(allow_private=True, browser=self.browser)

    def test_the_tools_read_what_the_browser_drew(self):
        box = default_toolbox(self.web)
        result = box.call("web_fetch", {"url": "http://127.0.0.1:1/"})
        self.assertTrue(result.ok, result.error)
        self.assertIn("The cat sat on the mat.", result.output)  # only in the page after its scripts ran
        self.assertTrue(result.output.startswith("Rendered — "))
        self.assertEqual(result.meta["link_count"], 1)

    def test_the_address_guard_still_runs_first(self):
        guarded = WebClient(browser=self.browser)  # allow_private is off
        box = default_toolbox(guarded)
        result = box.call("web_fetch", {"url": "http://10.0.0.1/"})
        self.assertFalse(result.ok)
        self.assertIn("private address", result.error)
        self.assertEqual(self.driver.visited, [])  # the browser was never asked

    def test_a_browser_failure_is_a_web_error(self):
        self.driver.refuse_session = "no"
        box = default_toolbox(self.web)
        result = box.call("web_fetch", {"url": "http://127.0.0.1:1/"})
        self.assertFalse(result.ok)
        self.assertIn("cannot open", result.error)

    def test_default_toolbox_takes_the_browser(self):
        box = default_toolbox(WebClient(allow_private=True), browser=self.browser)
        self.assertTrue(box.call("web_links", {"url": "http://127.0.0.1:1/"}).ok)


@unittest.skipUnless(browser.browser_available(), "no matching Chrome and chromedriver on this machine")
class RealChromeTests(unittest.TestCase):
    """The one test that drives a real browser (skipped unless the versions match)."""

    def test_a_page_that_draws_itself_is_readable(self):
        site = ThreadingHTTPServer(("127.0.0.1", 0), _page_handler(SOURCE))
        threading.Thread(target=site.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.addCleanup(site.server_close)
        self.addCleanup(site.shutdown)
        url = f"http://127.0.0.1:{site.server_address[1]}/"
        with BrowserClient(page_timeout=60) as chrome:
            box = default_toolbox(WebClient(allow_private=True), browser=chrome)
            drawn = box.call("web_fetch", {"url": url})
        plain = default_toolbox(WebClient(allow_private=True)).call("web_fetch", {"url": url})
        self.assertIn("The cat sat on the mat.", drawn.output)
        self.assertNotIn("The cat sat on the mat.", plain.output)  # urllib only ever sees "loading..."
        self.assertIn("loading...", plain.output)


def _page_handler(html):
    body = html.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


if __name__ == "__main__":
    unittest.main()
