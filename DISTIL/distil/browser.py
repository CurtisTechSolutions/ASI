"""A browser the system can drive, as tools in the same embedding layer.

The W3C WebDriver protocol is HTTP and JSON, so this talks to it with `urllib`
and no Selenium. That is not asceticism for its own sake: the rest of this
package has no third-party runtime dependency, and the moment one module needs
`pip install` the claim that the whole thing runs anywhere stops being true.

**Why a browser at all.** `auto.py` is rewarded for information, and the cheapest
large source of information it does not already hold is the web. Without this,
every question it cannot answer from memory terminates as an assumption; with it,
a why-chain can end in something it went and read.

**The shape.** `Browser` is the driver client. `register` writes each action into
memory as a `Kind.TOOL` trace with `transport="browser"`, so `browser.open` is
recalled by the same query that finds a locally forged function or an MCP tool --
one embedding layer over every capability, which is the property the design is
built on.

**What it is not.** It is not a sandbox. A driven browser fetches whatever it is
pointed at and runs whatever that page contains, in a process this package does
not control. That is the argument for the compose file: chromedriver belongs in a
container with no access to the host, and `DISTIL_WEBDRIVER` pointed at it. There
is no URL allowlist here because a convincing one cannot be written -- what it is
allowed to reach is the network's decision, not this module's.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .memory import Kind

DEFAULT_URL = "http://127.0.0.1:9515"
#: How much page text one read returns. A whole page would dominate every
#: embedding it lands in -- the vector would be about the volume of the text
#: rather than its subject -- and the useful part of a page is near the top.
MAX_TEXT = 4000


class BrowserError(RuntimeError):
    """The driver refused, or is not there. The message is shown to the user."""


def endpoint() -> str:
    return os.environ.get("DISTIL_WEBDRIVER", DEFAULT_URL).rstrip("/")


class Browser:
    """One WebDriver session, opened lazily and reused.

    Sessions are expensive and the loop makes many small requests, so the session
    outlives any single call. `close` is called from `Distil.save`-adjacent
    teardown the same way MCP servers are.
    """

    def __init__(self, url: str | None = None, headless: bool = True,
                 timeout: float = 30.0) -> None:
        self.url = (url or endpoint()).rstrip("/")
        self.headless = headless
        self.timeout = timeout
        self.session: str | None = None

    # -- protocol ------------------------------------------------------------

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.url}{path}", method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raise BrowserError(_why(exc)) from exc
        except urllib.error.URLError as exc:
            raise BrowserError(
                f"no webdriver at {self.url} ({exc.reason}). Start one with "
                f"`docker compose up -d chrome`, or set DISTIL_WEBDRIVER.") from exc

    def start(self) -> str:
        if self.session:
            return self.session
        args = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        if self.headless:
            args.insert(0, "--headless=new")
        out = self._call("POST", "/session", {
            "capabilities": {"alwaysMatch": {
                "browserName": "chrome",
                "goog:chromeOptions": {"args": args},
                # A page that never finishes loading must not hang the loop.
                "timeouts": {"pageLoad": 20000, "script": 10000, "implicit": 2000},
            }}})
        value = out.get("value") or {}
        self.session = value.get("sessionId") or out.get("sessionId")
        if not self.session:
            raise BrowserError(f"the driver did not return a session: {out}")
        return self.session

    def close(self) -> None:
        if not self.session:
            return
        try:
            self._call("DELETE", f"/session/{self.session}")
        except BrowserError:
            pass                      # already gone; nothing to reclaim
        finally:
            self.session = None

    def _in_session(self, method: str, path: str, body: dict | None = None):
        return self._call(method, f"/session/{self.start()}{path}", body).get("value")

    # -- the actions, which are the tools ------------------------------------

    def open(self, url: str) -> dict:
        """Go to a URL and report what arrived."""
        if not str(url).lower().startswith(("http://", "https://")):
            raise BrowserError("only http(s) URLs can be opened")
        self._in_session("POST", "/url", {"url": url})
        return {"url": self._in_session("GET", "/url"),
                "title": self._in_session("GET", "/title")}

    def read(self, selector: str = "body") -> str:
        """The visible text under a CSS selector, truncated."""
        found = self._in_session("POST", "/element",
                                 {"using": "css selector", "value": selector})
        if not isinstance(found, dict) or not found:
            raise BrowserError(f"nothing matched {selector!r}")
        element = next(iter(found.values()))
        text = self._in_session("GET", f"/element/{element}/text") or ""
        clean = " ".join(str(text).split())
        return clean[:MAX_TEXT] + (" …" if len(clean) > MAX_TEXT else "")

    def links(self, limit: int = 40) -> list[dict]:
        """Every link on the page, as text and href."""
        script = ("return [...document.querySelectorAll('a[href]')]"
                  ".slice(0, arguments[0]).map(a => ({text: a.innerText.trim(), href: a.href}))")
        out = self._in_session("POST", "/execute/sync", {"script": script, "args": [limit]})
        return [l for l in (out or []) if l.get("href")]

    def find(self, selector: str, limit: int = 20) -> list[str]:
        """The text of everything matching a selector."""
        script = ("return [...document.querySelectorAll(arguments[0])]"
                  ".slice(0, arguments[1]).map(e => e.innerText.trim()).filter(Boolean)")
        return self._in_session("POST", "/execute/sync",
                                {"script": script, "args": [selector, limit]}) or []

    def screenshot(self) -> str:
        """The page as a base64 PNG."""
        return self._in_session("GET", "/screenshot") or ""


def _why(exc) -> str:
    try:
        payload = json.loads(exc.read() or b"{}").get("value") or {}
        return payload.get("message") or f"webdriver returned {exc.code}"
    except Exception:
        return f"webdriver returned {exc.code}"


#: What gets embedded. Kept as data rather than reflection so the description a
#: recall matches against is written deliberately, not derived from a docstring.
ACTIONS = [
    ("browser.open", "browser.open(url)",
     "fetch a web page and report its final URL and title"),
    ("browser.read", "browser.read(selector='body')",
     "read the visible text of a page, or of one part of it, as plain text"),
    ("browser.links", "browser.links(limit=40)",
     "list the links on the current page, with their text and destination"),
    ("browser.find", "browser.find(selector, limit=20)",
     "pull the text of every element matching a CSS selector"),
    ("browser.screenshot", "browser.screenshot()",
     "capture the current page as a base64 PNG"),
]


def register(memory) -> list[str]:
    """Embed the browser actions so they are recalled like any other capability.

    Ungraded, for the same reason MCP tools are: there is no contract test that
    can be run against the live web, and entering them as verified would be
    manufacturing evidence. They earn credibility by being used and graded, which
    is the same path every other tool takes.
    """
    for name, signature, purpose in ACTIONS:
        memory.remember(
            Kind.TOOL, f"tool {name}: {purpose} {signature}",
            meta={"tool": name, "signature": signature, "purpose": purpose,
                  "transport": "browser", "solved": []},
            identity=f"tool:{name}")
    return [name for name, _, _ in ACTIONS]


def invoke(browser: Browser, name: str, args: list, kwargs: dict):
    """Dispatch a `browser.*` tool call onto a session."""
    action = name.split(".", 1)[-1]
    if action not in {a for _, a in ((n, n.split(".", 1)[-1]) for n, _, _ in ACTIONS)}:
        raise BrowserError(f"no such browser action: {name}")
    return getattr(browser, action)(*args, **kwargs)
