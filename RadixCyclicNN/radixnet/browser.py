"""A real browser behind the web tools: Chrome driven over the W3C WebDriver protocol.

:mod:`radixnet.tools` fetches pages with ``urllib``, which is enough for
ordinary documents and nothing at all for a page that draws itself with
JavaScript — a search engine, a single-page app, anything behind a consent
banner.  This module runs the page in **Chrome** instead and hands back the DOM
after the scripts have had their say, so the same ``web_fetch`` / ``web_links``
/ ``web_search`` tools see what a person would see.

WebDriver is an HTTP protocol with JSON bodies, so no driver library is needed:
:class:`BrowserClient` starts ``chromedriver`` itself and talks to it with the
standard library, exactly the way :class:`~radixnet.ollama.OllamaClient` talks
to Ollama.  ``selenium`` works too if it happens to be installed — it speaks the
same protocol to the same binary — but nothing here imports it.

What it needs: a ``chromedriver`` binary and a Chrome or Chromium to drive.
Both are found automatically (``$RADIXNET_CHROMEDRIVER`` / ``$RADIXNET_CHROME``
override, and a Playwright browser directory is understood), and
:func:`describe` reports what was found without starting anything.  A session is
opened on the first page and reused for every page after it, so the browser
starts once per run; :meth:`BrowserClient.close` (or the context manager) stops
it, and it is stopped at interpreter exit in any case.

Headless by default.  A browser executes whatever a page sends it, so this is
not a sandbox: the address guards of :class:`~radixnet.tools.WebClient` still
run first, and untrusted browsing belongs in the Docker image.
"""

from __future__ import annotations

import atexit
import glob
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

__all__ = [
    "DEFAULT_PAGE_TIMEOUT",
    "DEFAULT_WINDOW",
    "BrowserClient",
    "BrowserError",
    "browser_available",
    "chrome_path",
    "chromedriver_path",
    "describe",
]

DEFAULT_PAGE_TIMEOUT = 30.0
"""Seconds a page may take to load before the browser gives up on it."""
DEFAULT_WINDOW = (1280, 900)
STARTUP_TIMEOUT = 20.0
SETTLE_SECONDS = 0.25
"""How long to let a page keep drawing after ``readyState`` says it is complete."""

_CHROME_NAMES = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
_CHROME_ARGS = (
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",       # /dev/shm is small in a container
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-sync",
    "--no-first-run",
    "--no-default-browser-check",
    "--mute-audio",
    "--blink-settings=imagesEnabled=false",  # the tools read text; images are bytes and time
)


class BrowserError(Exception):
    """Chrome could not be found, started, or asked to do something."""


# ---------------------------------------------------------------------------
# finding the binaries
# ---------------------------------------------------------------------------


def chromedriver_path() -> str | None:
    """The ``chromedriver`` binary (``$RADIXNET_CHROMEDRIVER``, else ``PATH``)."""
    explicit = os.environ.get("RADIXNET_CHROMEDRIVER", "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    return shutil.which("chromedriver")


def chrome_path() -> str | None:
    """The Chrome / Chromium binary to drive (``$RADIXNET_CHROME``, then ``PATH``, then Playwright's browsers)."""
    explicit = os.environ.get("RADIXNET_CHROME", "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for name in _CHROME_NAMES:
        found = shutil.which(name) if os.sep not in name else (name if os.path.isfile(name) else None)
        if found:
            return found
    # a Playwright browser directory: .../chromium-1194/chrome-linux/chrome
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    for pattern in (
        os.path.join(root, "chromium*", "chrome-linux", "chrome"),
        os.path.join(root, "chromium*", "chrome-mac", "Chromium.app", "Contents/MacOS/Chromium"),
        os.path.join(root, "chromium*", "chrome-win", "chrome.exe"),
    ) if root else ():
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _version_of(binary: str | None) -> str | None:
    """``<binary> --version``, or ``None`` when it cannot be run."""
    if not binary:
        return None
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return " ".join((out.stdout or out.stderr or "").split()) or None


def _major(version: str | None) -> str | None:
    """The major version out of ``Chromium 141.0.7390.37`` / ``ChromeDriver 147.0...``."""
    for word in (version or "").split():
        head = word.split(".")[0]
        if head.isdigit():
            return head
    return None


def browser_available() -> bool:
    """True when both binaries are there and their major versions match (nothing is started to find out)."""
    return describe()["available"]


def describe() -> dict:
    """What browsing with Chrome would use, and why it would not work.

    ``{"available", "chromedriver", "chrome", "chromedriver_version",
    "chrome_version", "headless", "error"}``.  chromedriver refuses to drive a
    Chrome of a different major version, which is far and away the most common
    way this fails, so the versions are read and compared here rather than left
    to a WebDriver error nobody can act on.
    """
    driver, chrome = chromedriver_path(), chrome_path()
    missing = [name for name, found in (("chromedriver", driver), ("Chrome", chrome)) if not found]
    driver_version = _version_of(driver)
    chrome_version = _version_of(chrome)
    error = None
    if missing:
        variables = " / ".join(
            "$RADIXNET_CHROMEDRIVER" if m == "chromedriver" else "$RADIXNET_CHROME" for m in missing
        )
        error = f"{' and '.join(missing)} not found: install it, or point {variables} at it"
    elif _major(driver_version) and _major(chrome_version) and _major(driver_version) != _major(chrome_version):
        error = (
            f"chromedriver {_major(driver_version)} cannot drive Chrome {_major(chrome_version)}: they must share a "
            f"major version. Install the matching chromedriver (chrome-for-testing has one per Chrome release) and "
            f"point $RADIXNET_CHROMEDRIVER at it, or $RADIXNET_CHROME at a Chrome {_major(driver_version)}"
        )
    return {
        "available": error is None,
        "chromedriver": driver,
        "chrome": chrome,
        "chromedriver_version": driver_version,
        "chrome_version": chrome_version,
        "headless": True,
        "error": error,
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------


class BrowserClient:
    """One headless Chrome, started on demand and reused for every page.

    ``page_timeout`` caps a navigation, ``window`` is the viewport (a real size
    matters: sites lay out differently at 400px), and ``settle`` is how long the
    page is given to keep drawing after it reports itself complete.  Every
    method raises :class:`BrowserError` and never leaves a process behind.
    """

    __slots__ = ("page_timeout", "window", "settle", "headless", "driver", "chrome", "endpoint", "_base",
                 "_process", "_session", "_lock", "pages")

    def __init__(
        self,
        page_timeout: float = DEFAULT_PAGE_TIMEOUT,
        window: tuple[int, int] = DEFAULT_WINDOW,
        settle: float = SETTLE_SECONDS,
        headless: bool = True,
        driver: str | None = None,
        chrome: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        if page_timeout <= 0:
            raise ValueError("page_timeout must be > 0")
        self.page_timeout = float(page_timeout)
        self.window = (int(window[0]), int(window[1]))
        self.settle = max(0.0, float(settle))
        self.headless = bool(headless)
        self.endpoint = (endpoint or os.environ.get("RADIXNET_WEBDRIVER", "")).strip().rstrip("/") or None
        """A WebDriver already running (a Selenium Grid, a ``selenium/standalone-chrome`` container, a
        chromedriver of your own): it is used as it is and nothing is started or stopped here."""
        self.driver = None if self.endpoint else (driver or chromedriver_path())
        self.chrome = None if self.endpoint else (chrome or chrome_path())
        self._base: str | None = self.endpoint
        self._process: subprocess.Popen | None = None
        self._session: str | None = None
        self._lock = threading.RLock()
        self.pages = 0

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "BrowserClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._session is not None

    def _start(self) -> None:
        """Start chromedriver and open one session (idempotent)."""
        if self._session is not None:
            return
        if self.endpoint is None:
            problem = describe()
            if not self.driver or not self.chrome or problem["error"]:
                raise BrowserError(problem["error"] or "no browser available")
            self._spawn()
        self._open_session()

    def _spawn(self) -> None:
        """Start our own chromedriver on a free port and wait for it."""
        port = _free_port()
        try:
            self._process = subprocess.Popen(
                [self.driver, f"--port={port}", "--silent"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise BrowserError(f"cannot start {self.driver}: {exc}") from exc
        self._base = f"http://127.0.0.1:{port}"
        atexit.register(self.close)
        self._await_driver()

    def _open_session(self) -> None:
        """Ask the WebDriver for a session (ours or somebody else's)."""
        args = list(_CHROME_ARGS) + [f"--window-size={self.window[0]},{self.window[1]}"]
        if not self.headless:
            args.remove("--headless=new")
        if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0:
            args.append("--no-sandbox")  # Chrome refuses to run as root otherwise (containers, CI)
        options: dict[str, Any] = {"args": args}
        if self.chrome:
            options["binary"] = self.chrome  # a remote WebDriver has its own browser
        body = {"capabilities": {"alwaysMatch": {
            "browserName": "chrome",
            "pageLoadStrategy": "normal",
            "goog:chromeOptions": options,
            "timeouts": {"pageLoad": int(self.page_timeout * 1000), "script": int(self.page_timeout * 1000)},
        }}}
        try:
            answer = self._request("POST", "/session", body)
        except BrowserError:
            self.close()
            raise
        session = (answer or {}).get("sessionId") or (answer or {}).get("value", {}).get("sessionId")
        if not session:
            self.close()
            raise BrowserError(f"chromedriver did not open a session: {str(answer)[:200]}")
        self._session = session

    def _await_driver(self) -> None:
        """Wait for chromedriver to answer ``/status`` (it binds its port a moment after starting)."""
        deadline = time.monotonic() + STARTUP_TIMEOUT
        last = "no answer"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise BrowserError(f"{self.driver} exited with code {self._process.returncode} on start-up")
            try:
                status = self._request("GET", "/status", timeout=2.0)
            except BrowserError as exc:
                last = str(exc)
                time.sleep(0.05)
                continue
            if (status or {}).get("value", {}).get("ready", True):
                return
            time.sleep(0.05)
        self.close()
        raise BrowserError(f"{self.driver} did not become ready in {STARTUP_TIMEOUT:g}s ({last})")

    def close(self) -> None:
        """Close the session and stop chromedriver (safe to call twice)."""
        with self._lock:
            if self._session is not None:
                try:
                    self._request("DELETE", f"/session/{self._session}", timeout=5.0)
                except BrowserError:
                    pass
                self._session = None
            process, self._process = self._process, None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            self._base = self.endpoint

    # -- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> Any:
        if self._base is None:
            raise BrowserError("the browser is not running")
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base}{path}", data=data, method=method,
            headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local: never via a proxy
        try:
            with opener.open(request, timeout=timeout or self.page_timeout + 5.0) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                value = json.loads(detail).get("value", {})
                detail = f"{value.get('error', '')}: {value.get('message', '')}".strip(": ") or detail
            except (ValueError, AttributeError):
                pass
            raise BrowserError(detail.split("\n")[0][:300] or f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            raise BrowserError(f"cannot reach chromedriver: {getattr(exc, 'reason', None) or exc}") from exc

    def _session_request(self, method: str, path: str, body: dict | None = None) -> Any:
        self._start()
        return self._request(method, f"/session/{self._session}{path}", body)

    # -- pages ---------------------------------------------------------------

    def script(self, source: str, *args: Any) -> Any:
        """Run JavaScript in the page and return what it gives back."""
        answer = self._session_request("POST", "/execute/sync", {"script": source, "args": list(args)})
        return (answer or {}).get("value")

    def get(self, url: str) -> dict:
        """Open ``url`` and return ``{"url", "title", "html", "status"}`` once the page has drawn itself."""
        with self._lock:
            self._session_request("POST", "/url", {"url": url})
            self._settle()
            html = self.script("return document.documentElement.outerHTML;")
            final = self.script("return document.location.href;")
            title = (self._session_request("GET", "/title") or {}).get("value") or ""
            self.pages += 1
        if not isinstance(html, str):
            raise BrowserError(f"the page at {url} returned no HTML")
        return {"url": str(final or url), "title": str(title), "html": html, "status": 200}

    def _settle(self) -> None:
        """Wait for ``document.readyState`` to be complete, then a moment more for the scripts."""
        deadline = time.monotonic() + self.page_timeout
        while time.monotonic() < deadline:
            try:
                if self.script("return document.readyState;") == "complete":
                    break
            except BrowserError:
                break
            time.sleep(0.05)
        if self.settle:
            time.sleep(self.settle)

    def describe(self) -> dict:
        """What this client is (and whether its browser is up)."""
        if self.endpoint is not None:
            found = {"available": True, "chromedriver": self.endpoint, "chrome": "(the endpoint's own)",
                     "chromedriver_version": None, "chrome_version": None, "error": None}
        else:
            found = describe()
        return {
            **found, "endpoint": self.endpoint, "headless": self.headless, "running": self.running,
            "pages": self.pages, "page_timeout": self.page_timeout, "window": list(self.window),
        }
