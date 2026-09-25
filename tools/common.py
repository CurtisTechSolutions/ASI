"""Plumbing shared by the download tools in this directory.

Nothing here knows about Project Gutenberg or Wikipedia. It is how both of them
talk to servers they do not own - one request at a time, a pause between
requests, patient retries, and a hard stop the moment the server (or a proxy in
front of it) says no - plus the file handling both need: names that are safe on
every filesystem, writes that are never left half-done, and a JSON-lines
manifest of what was written.

Standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import functools
import gzip
import http.client
import json
import os
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zlib
from typing import Any, Callable, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\w+")


def fold(text: str) -> str:
    """``text`` case-folded and without accents, so "Brontë" and "BRONTE" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@functools.lru_cache(maxsize=1 << 16)
def words(text: str) -> tuple[str, ...]:
    """The folded words of ``text``. Cached: subjects, shelves and categories repeat a lot."""
    return tuple(_WORD.findall(fold(text)))


def slugify(text: str, limit: int = 60) -> str:
    """``text`` as a file-name-safe run of folded words joined by hyphens.

    Letters outside ASCII are kept (a Greek title stays Greek) but accents,
    punctuation and anything a filesystem might object to are not. A slug longer
    than ``limit`` is cut at a word boundary when there is one near the end.
    """
    slug = re.sub(r"[\W_]+", "-", fold(re.sub(r"['’]", "", text))).strip("-")
    if len(slug) > limit:
        cut = slug.rfind("-", 0, limit + 1)
        slug = slug[: cut if cut > limit // 2 else limit].strip("-")
    return slug


class TextQuery:
    """One value of a text filter.

    By default every word of the query must begin a word of the field, in any
    order, ignoring case and accents: "austen jane" matches "Austen, Jane,
    1775-1817" and "detect" matches "Detective and mystery stories". With
    ``regex`` the query is a case-insensitive regular expression instead
    (``re.error`` if it is not one).
    """

    def __init__(self, text: str, regex: bool = False):
        self.text = text
        self.pattern = re.compile(text, re.IGNORECASE) if regex else None
        self.words = () if regex else words(text)
        if not regex and not self.words:
            raise ValueError(f"{text!r} has no words to search for")

    def __call__(self, value: str) -> bool:
        if self.pattern is not None:
            return bool(self.pattern.search(value) or self.pattern.search(fold(value)))
        have = words(value)
        return all(any(word.startswith(want) for word in have) for want in self.words)

    def __repr__(self) -> str:
        return f"TextQuery({self.text!r})"


def any_match(queries: Sequence[TextQuery], values: Iterable[str]) -> bool:
    """Whether any of ``queries`` matches any of ``values``."""
    values = tuple(values)
    return any(query(value) for query in queries for value in values)


def text_queries(values: Sequence[str] | None, regex: bool = False) -> tuple[TextQuery, ...]:
    """A command line's text filter values as queries; ValueError for a bad one."""
    out = []
    for value in values or ():
        try:
            out.append(TextQuery(value, regex))
        except re.error as exc:
            raise ValueError(f"bad regular expression {value!r}: {exc}") from None
    return tuple(out)


def clip(text: str, width: int) -> str:
    """``text`` cut to ``width`` characters, ending in an ellipsis if anything was lost."""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def human_bytes(count: float) -> str:
    """``1234567`` as ``"1.2 MB"`` (decimal units, as download sizes are quoted)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(count) < 1000 or unit == "TB":
            return f"{count:,.0f} {unit}" if unit == "B" else f"{count:,.1f} {unit}"
        count /= 1000
    raise AssertionError("unreachable")


def utc_now() -> str:
    """The current time as an ISO-8601 UTC timestamp, to the second."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def write_atomic(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file, so ``path`` is never half-written."""
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


class Manifest:
    """A JSON-lines record of what a tool wrote: one object per item, keyed by an integer.

    Each entry is appended the moment its item is written, so an interrupted run
    loses nothing; when a key repeats, the later line wins. :meth:`compact`
    rewrites the file with one line per key, sorted, at the end of a run.
    """

    def __init__(self, path: str, key: str = "id"):
        self.path = path
        self.key = key
        self.entries: dict[int, dict[str, Any]] = {}
        self._changed = False
        self._needs_newline = False
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            return
        # A run killed mid-append leaves a partial last line; the next entry
        # must not be glued onto it.
        self._needs_newline = bool(text) and not text.endswith("\n")
        for line in text.splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get(key), int):
                self.entries[entry[key]] = entry

    def __contains__(self, item: object) -> bool:
        return item in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, item: int) -> dict[str, Any] | None:
        return self.entries.get(item)

    def add(self, entry: dict[str, Any]) -> None:
        """Record ``entry`` (which must carry the key) and append it to the file now."""
        self.entries[entry[self.key]] = entry
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            if self._needs_newline:
                fh.write("\n")
                self._needs_newline = False
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._changed = True

    def compact(self) -> None:
        """Rewrite the file as one line per key, sorted - if anything was added."""
        if not self._changed:
            return
        lines = (json.dumps(self.entries[k], ensure_ascii=False) + "\n" for k in sorted(self.entries))
        write_atomic(self.path, "".join(lines).encode("utf-8"))
        self._changed = False


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """A request failed, after whatever retries it deserved. ``status`` is the HTTP code, if any."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NotFound(FetchError):
    """The server has no such file (404 or 410). Not retried: look somewhere else."""


class Fatal(FetchError):
    """Nothing further will work: the site or a proxy refuses us, or TLS will not verify.

    Every later request would get the same answer, and retrying a refusal is
    how an address gets banned, so a run that meets this stops.
    """


@dataclasses.dataclass
class Reply:
    """A response body and its headers."""

    body: bytes
    headers: Mapping[str, str]


_TUNNEL_REFUSED = re.compile(r"Tunnel connection failed: 40[37]")


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or url


def _retry_after(headers: Any, cap: float) -> float | None:
    """The ``Retry-After`` header in seconds (the date form is ignored), at most ``cap``."""
    try:
        value = float(headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None
    return min(cap, max(0.0, value))


class Fetcher:
    """HTTP GET for tools that download from sites they do not own.

    * One request at a time, at least ``delay`` seconds apart, retries included.
    * Failures worth retrying - timeouts, dropped connections, 408/425/429/5xx -
      are retried ``retries`` times with exponential backoff, honouring
      ``Retry-After``.
    * 404 and 410 raise :class:`NotFound` at once.
    * A refusal - 401/403/407/451, a proxy refusing the tunnel, a certificate
      that does not verify - raises :class:`Fatal`.

    ``opener``, ``sleep`` and ``clock`` can be replaced, so the tests run it
    with no network and no real waiting.
    """

    RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
    REFUSED_STATUS = frozenset({401, 403, 407, 451})
    MAX_RETRY_AFTER = 300.0

    def __init__(
        self,
        user_agent: str,
        *,
        delay: float = 1.0,
        retries: int = 3,
        timeout: float = 60.0,
        opener: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ):
        self.user_agent = user_agent
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener()
        self.sleep = sleep
        self.clock = clock
        self.log = log or (lambda message: None)
        self.requests = 0
        self._last: float | None = None

    def fetch(self, url: str, headers: Mapping[str, str] | None = None) -> Reply:
        """The whole of ``url``, gunzipped if the server compressed it on the way."""

        def once() -> Reply:
            with self._open(url, {"Accept-Encoding": "gzip", **(headers or {})}) as response:
                body = response.read()
                reply_headers = response.headers
            if (reply_headers.get("Content-Encoding") or "").lower() == "gzip":
                body = gzip.decompress(body)
            return Reply(body, reply_headers)

        return self._retry(url, once)

    def get(self, url: str, headers: Mapping[str, str] | None = None) -> bytes:
        """The body of ``url``."""
        return self.fetch(url, headers).body

    def open(self, url: str, headers: Mapping[str, str] | None = None) -> Any:
        """``url`` opened for streaming, for files too big to hold in memory. The caller closes it.

        Only opening is retried here; a transfer that breaks halfway is the
        caller's to resume.
        """
        return self._retry(url, lambda: self._open(url, dict(headers or {})))

    def wait(self, seconds: float) -> None:
        """Pause for ``seconds`` the way this fetcher pauses (so tests need not)."""
        if seconds > 0:
            self.sleep(seconds)

    def _open(self, url: str, headers: Mapping[str, str]) -> Any:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, **headers})
        self.requests += 1
        return self.opener.open(request, timeout=self.timeout)

    def _wait_turn(self) -> None:
        now = self.clock()
        if self._last is not None:
            remaining = self.delay - (now - self._last)
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self._last = now

    def backoff(self, attempt: int) -> float:
        """Seconds to wait before retry number ``attempt + 1``: doubling, from the delay, up to a minute."""
        return min(60.0, max(1.0, self.delay) * 2**attempt)

    def _retry(self, url: str, action: Callable[[], Any]) -> Any:
        attempt = 0
        while True:
            self._wait_turn()
            try:
                return action()
            except urllib.error.HTTPError as exc:
                with contextlib.suppress(Exception):
                    exc.close()
                if exc.code in (404, 410):
                    raise NotFound(f"HTTP {exc.code} for {url}", exc.code) from None
                if exc.code in self.REFUSED_STATUS:
                    raise Fatal(
                        f"HTTP {exc.code} ({exc.reason}) from {_host(url)}: the server refuses this client",
                        exc.code,
                    ) from None
                if exc.code not in self.RETRY_STATUS or attempt >= self.retries:
                    raise FetchError(f"HTTP {exc.code} ({exc.reason}) for {url}", exc.code) from None
                pause = _retry_after(exc.headers, self.MAX_RETRY_AFTER) or self.backoff(attempt)
                problem = f"HTTP {exc.code}"
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, ssl.SSLCertVerificationError):
                    raise Fatal(f"cannot verify the TLS certificate of {_host(url)}: {reason}") from None
                if _TUNNEL_REFUSED.search(str(reason)):
                    raise Fatal(
                        f"the proxy refused to connect to {_host(url)} ({reason}) - "
                        "is that host allowed on this network?"
                    ) from None
                if attempt >= self.retries:
                    raise FetchError(f"{url}: {reason}") from None
                pause, problem = self.backoff(attempt), str(reason)
            except (OSError, EOFError, zlib.error, http.client.HTTPException) as exc:
                if attempt >= self.retries:
                    raise FetchError(f"{url}: {exc or type(exc).__name__}") from None
                pause, problem = self.backoff(attempt), str(exc) or type(exc).__name__
            except ValueError as exc:  # a malformed URL; no retry will fix it
                raise FetchError(f"{url}: {exc}") from None
            attempt += 1
            self.log(f"  {problem} from {_host(url)}; retrying in {pause:.0f}s ({attempt}/{self.retries})")
            self.sleep(pause)


# ---------------------------------------------------------------------------
# command lines
# ---------------------------------------------------------------------------


def at_least(minimum: float, cast: Callable[[str], Any] = int) -> Callable[[str], Any]:
    """An argparse ``type``: a number (``cast`` of the text) no smaller than ``minimum``."""

    def parse(text: str) -> Any:
        try:
            value = cast(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
        if not value >= minimum:  # `not >=` also refuses NaN
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {text}")
        return value

    return parse


def printer(stream: Any, quiet: bool = False) -> Callable[[str], None]:
    """A one-argument print to ``stream``, or a no-op when ``quiet``."""
    if quiet:
        return lambda message: None
    return lambda message: print(message, file=stream, flush=True)


def is_url(text: str) -> bool:
    return text.startswith(("http://", "https://"))
