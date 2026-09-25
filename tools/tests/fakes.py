"""Test doubles shared by the tools' tests: a fake web site, a fake clock and a real local server.

Nothing here touches the network. :class:`FakeSite` replaces urllib's opener
inside the process; :class:`LocalServer` is a real HTTP server on 127.0.0.1, for
the few tests that must go through urllib's own sockets, headers and redirects.
"""

from __future__ import annotations

import gzip
import http.server
import io
import threading
import urllib.error
import urllib.request
import urllib.response

REASONS = {
    403: "Forbidden",
    404: "Not Found",
    416: "Range Not Satisfiable",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


def http_error(url, code, headers=None):
    """The exception urllib raises for an HTTP error status."""
    return urllib.error.HTTPError(url, code, REASONS.get(code, "Error"), headers or {}, io.BytesIO(b""))


class FakeSite:
    """Stands in for urllib's opener: serves canned responses by URL and records each request.

    A page is the body (bytes), a ``(body, headers)`` or ``(body, headers,
    status)`` tuple, an exception to raise, a callable taking the request and
    returning one of those, or a list of any of them served in turn (the last
    one repeats). A URL that is not listed is a 404.
    """

    def __init__(self, pages=None):
        self.pages = dict(pages or {})
        self.requests = []
        self.headers = []

    def open(self, request, timeout=None):
        url = request.full_url
        self.requests.append(url)
        self.headers.append(dict(request.header_items()))
        page = self.pages.get(url)
        if isinstance(page, list):
            page = page.pop(0) if len(page) > 1 else page[0]
        if callable(page) and not isinstance(page, BaseException):
            page = page(request)
        if page is None:
            raise http_error(url, 404)
        if isinstance(page, BaseException):  # before the next test: an HTTPError can read(), too
            raise page
        if hasattr(page, "read"):  # a ready-made response
            return page
        if isinstance(page, tuple):
            body, headers, status = (*page, 200)[:3]
        else:
            body, headers, status = page, {}, 200
        return urllib.response.addinfourl(io.BytesIO(body), headers, url, status)

    def asked_for(self, fragment):
        """The requested URLs containing ``fragment``."""
        return [url for url in self.requests if fragment in url]


class FakeClock:
    """A clock that moves only when something sleeps on it, recording every sleep."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def no_proxy_opener():
    """urllib's opener without the environment's proxies - the local server is local."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LocalServer:
    """A real HTTP server on 127.0.0.1 serving ``pages`` (path -> bytes), in a thread.

    It gzips a response when the client accepts gzip (and the file is not
    already compressed), and honours ``Range: bytes=N-`` with a 206, so the
    code under test meets the same headers it would meet on the real sites.
    """

    def __init__(self, pages):
        self.pages = pages
        self.requests = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - the name http.server calls
                owner.requests.append(self.path)
                body = owner.pages.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                status, headers = 200, {}
                wanted = self.headers.get("Range", "")
                if wanted.startswith("bytes=") and wanted.endswith("-"):
                    start = int(wanted[len("bytes=") : -1])
                    if start >= len(body):
                        self.send_error(416)
                        return
                    headers["Content-Range"] = f"bytes {start}-{len(body) - 1}/{len(body)}"
                    body, status = body[start:], 206
                elif "gzip" in self.headers.get("Accept-Encoding", "") and not self.path.endswith((".gz", ".bz2")):
                    body = gzip.compress(body)
                    headers["Content-Encoding"] = "gzip"
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
