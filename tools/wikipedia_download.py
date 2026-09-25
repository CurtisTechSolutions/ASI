#!/usr/bin/env python3
"""Download Wikipedia - all of it from the official dumps, or chosen articles through the API.

    python3 tools/wikipedia_download.py dump --lang simple --list      what is in the newest dump?
    python3 tools/wikipedia_download.py dump --lang simple             download it: resumable, checksummed
    python3 tools/wikipedia_download.py extract wikipedia_dumps/simplewiki-*-pages-articles.xml.bz2
    python3 tools/wikipedia_download.py fetch --category Physics --depth 1 -n 200
    python3 tools/wikipedia_download.py fetch --search "black hole" -n 20 --format jsonl

All of Wikipedia means the database dumps Wikimedia publishes twice a month at
https://dumps.wikimedia.org. Wikimedia asks people not to crawl the site for
that, and a crawl would take months where the dump takes an afternoon. ``dump``
finds the newest finished dump, downloads it with resume and checks it against
the published SHA-1; ``extract`` streams it into plain text - one article per
line of JSON, or one file per article - without holding it in memory, on as many
processes as there are cores.

For a few thousand articles the API is the better road: ``fetch`` takes them by
title, category (and subcategories), search or at random, and saves the plain
text Wikipedia's own TextExtracts produce - one request at a time, backing off
whenever the servers say they are busy.

Standard library only; Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import bz2
import collections
import dataclasses
import gzip
import hashlib
import html
import http.client
import itertools
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Callable, Iterator, Sequence, TextIO

from common import (
    FetchError,
    Fetcher,
    Manifest,
    NotFound,
    TextQuery,
    any_match,
    at_least,
    clip,
    fold,
    human_bytes,
    printer,
    slugify,
    text_queries,
    utc_now,
    write_atomic,
)

__version__ = "1.0.0"

DUMPS = "https://dumps.wikimedia.org"
DEFAULT_DUMP_DIR = "wikipedia_dumps"
DEFAULT_TEXT_DIR = "wikipedia_text"
DEFAULT_FETCH_DIR = "wikipedia_articles"
MANIFEST = "manifest.jsonl"
EXTRACT_MANIFEST = "manifest.json"
FETCHED_JSONL = "articles.jsonl"

DUMP_KINDS: dict[str, tuple[str | None, str | None, str]] = {
    "articles": (
        r"pages-articles\.xml\.bz2",
        r"pages-articles\d+\.xml-p\d+p\d+\.bz2",
        "the current text of every article, template and project page (what 'all of Wikipedia' usually means)",
    ),
    "multistream": (
        r"pages-articles-multistream(?:-index)?\.(?:xml|txt)\.bz2",
        r"pages-articles-multistream(?:-index)?\d+\.(?:xml|txt)-p\d+p\d+\.bz2",
        "the same, compressed in independent blocks with an index, for random access",
    ),
    "meta-current": (
        r"pages-meta-current\.xml\.bz2",
        r"pages-meta-current\d+\.xml-p\d+p\d+\.bz2",
        "the current text of every page, talk and user pages included",
    ),
    "history": (
        r"pages-meta-history\.xml\.bz2",
        r"pages-meta-history\d+\.xml-p\d+p\d+\.bz2",
        "every revision of every page - terabytes, for the big wikis",
    ),
    "abstracts": (r"abstract\.xml\.gz", r"abstract\d+\.xml\.gz", "the opening of every article"),
    "titles": (r"all-titles-in-ns0\.gz", None, "the title of every article"),
}
"""Dump kinds: the whole file, the parts big wikis split it into, and what it holds."""


class DumpError(Exception):
    """A dump could not be found, is not finished, or did not arrive intact."""


class ExtractError(Exception):
    """A dump could not be read, or the output directory is not safe to write into."""


class ApiError(Exception):
    """The MediaWiki API answered with an error."""


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def user_agent(contact: str | None) -> str:
    """The User-Agent Wikimedia's policy asks for: the tool, how to reach its user, the library."""
    about = f" ({contact})" if contact else ""
    return f"wikipedia_download/{__version__}{about} Python-urllib/{sys.version_info[0]}.{sys.version_info[1]}"


# ---------------------------------------------------------------------------
# dumps
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DumpFile:
    """One file of a dump, as ``dumpstatus.json`` describes it."""

    name: str
    size: int | None = None
    sha1: str | None = None
    md5: str | None = None


def wiki_name(lang: str) -> str:
    """The dump name of a language's Wikipedia: "en" is "enwiki", "zh-yue" is "zh_yuewiki"."""
    return lang.strip().lower().replace("-", "_") + "wiki"


_DATE_LINK = re.compile(r'href="(?:\./)?(\d{8})/?"')


def dump_dates(fetcher: Fetcher, wiki: str, base: str = DUMPS) -> list[str]:
    """The dates of the dumps listed for ``wiki``, newest first."""
    try:
        page = fetcher.get(f"{base}/{wiki}/")
    except NotFound:
        raise DumpError(f"{base}/{wiki}/ does not exist - is {wiki!r} the right wiki?") from None
    return sorted(set(_DATE_LINK.findall(page.decode("utf-8", "replace"))), reverse=True)


def _natural(name: str) -> list[Any]:
    """A sort key that puts part 2 before part 10."""
    return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", name)]


def select_files(status: dict, wiki: str, date: str, kind: str, parts: bool = False) -> tuple[list[DumpFile], bool]:
    """The files of ``kind`` in a dump's ``dumpstatus.json``, and whether all of them are finished.

    The whole file is preferred and the parts are the fallback (``parts``
    reverses that): for small wikis only the whole file exists, and for big
    ones the whole file is stitched together from the parts hours after they
    are finished - in between, the finished parts are the answer.
    """
    whole, split, _ = DUMP_KINDS[kind]
    listed: dict[str, tuple[dict, str]] = {}
    for job in (status.get("jobs") or {}).values():
        if isinstance(job, dict):
            for name, info in (job.get("files") or {}).items():
                listed[name] = (info if isinstance(info, dict) else {}, job.get("status", ""))
    unfinished: list[DumpFile] = []
    for pattern in (split, whole) if parts else (whole, split):
        if pattern is None:
            continue
        regex = re.compile(rf"{re.escape(wiki)}-{re.escape(date)}-{pattern}")
        names = sorted((name for name in listed if regex.fullmatch(name)), key=_natural)
        if not names:
            continue
        files = [
            DumpFile(name, _int(listed[name][0].get("size")), listed[name][0].get("sha1"), listed[name][0].get("md5"))
            for name in names
        ]
        if all(listed[name][1] == "done" for name in names):
            return files, True
        unfinished = unfinished or files
    return unfinished, False


def find_dump(
    fetcher: Fetcher,
    wiki: str,
    kind: str,
    *,
    date: str = "latest",
    parts: bool = False,
    base: str = DUMPS,
    tries: int = 5,
    log: Callable[[str], None] = lambda message: None,
) -> tuple[str, list[DumpFile]]:
    """The date and files of the newest finished ``kind`` dump of ``wiki`` (or of ``date``).

    A dump run takes days, so the newest date is often still in progress; this
    walks back through the ``tries`` newest until one is finished.
    """
    latest = date == "latest"
    dates = dump_dates(fetcher, wiki, base)[:tries] if latest else [date]
    if not dates:
        raise DumpError(f"no dumps are listed at {base}/{wiki}/")
    for day in dates:
        try:
            status = json.loads(fetcher.get(f"{base}/{wiki}/{day}/dumpstatus.json"))
        except NotFound:
            if not latest:
                raise DumpError(f"there is no {day} dump of {wiki}") from None
            continue
        except ValueError:
            raise DumpError(f"{base}/{wiki}/{day}/dumpstatus.json is not JSON") from None
        files, done = select_files(status, wiki, day, kind, parts)
        if files and done:
            return day, files
        why = "is not finished yet" if files else "has no such files"
        if not latest:
            raise DumpError(f"the {day} {kind} dump of {wiki} {why} - try another --date")
        log(f"the {day} {kind} dump of {wiki} {why}; looking at the one before")
    raise DumpError(f"none of the {len(dates)} newest dumps of {wiki} has a finished {kind} dump")


def file_digest(path: str, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class Progress:
    """A download's progress on one line: redrawn in place on a terminal, every tenth otherwise."""

    def __init__(self, name: str, total: int | None, stream: TextIO | None = None, clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.total = total
        self.stream = stream or sys.stderr
        self.clock = clock
        self.tty = self.stream.isatty()
        self.started = clock()
        self.first: int | None = None
        self.shown = 0.0
        self.tenth = -1

    def __call__(self, done: int) -> None:
        now = self.clock()
        if self.first is None:
            self.first = done
        if self.tty:
            if now - self.shown >= 0.5 or done == self.total:
                self.shown = now
                self.stream.write("\r\033[K" + self.line(done, now))
                self.stream.flush()
        elif self.total:
            tenth = 10 * done // self.total
            if tenth > self.tenth:
                self.tenth = tenth
                print(self.line(done, now), file=self.stream, flush=True)

    def line(self, done: int, now: float) -> str:
        rate = (done - (self.first or 0)) / max(now - self.started, 1e-3)
        text = f"  {clip(self.name, 60)}  {human_bytes(done)}"
        if self.total:
            text += f" of {human_bytes(self.total)} ({100 * done / self.total:.0f}%)"
            if rate > 0 and done < self.total:
                text += f"  {human_bytes(rate)}/s, {_duration((self.total - done) / rate)} left"
        return text

    def finish(self) -> None:
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()


def download_file(
    fetcher: Fetcher,
    url: str,
    path: str,
    *,
    size: int | None = None,
    sha1: str | None = None,
    md5: str | None = None,
    verify: bool = True,
    progress: Callable[[int], None] | None = None,
) -> str:
    """Download ``url`` to ``path`` through ``path.part``, resuming whatever a previous run left.

    A transfer that breaks is picked up where it stopped (``Range``), up to the
    fetcher's retries in a row. The finished file must have the published size
    and checksum, or it is deleted - a corrupt dump is worse than none - and
    DumpError says so. Returns "exists" if ``path`` was already complete, else
    "saved".
    """
    if os.path.exists(path) and (size is None or os.path.getsize(path) == size):
        return "exists"
    part = path + ".part"
    failures = 0
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size is not None and have >= size:
            break
        try:
            response = fetcher.open(url, {"Range": f"bytes={have}-"} if have else None)
        except FetchError as exc:
            if exc.status == 416 and have:  # nothing past what we hold: it is complete
                break
            raise
        try:
            if have and getattr(response, "status", 200) != 206:
                have = 0  # the server ignored the range: start again
            with open(part, "ab" if have else "wb") as fh:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    fh.write(chunk)
                    have += len(chunk)
                    if progress:
                        progress(have)
        except (OSError, http.client.HTTPException) as exc:
            failures += 1
            if failures > fetcher.retries:
                raise FetchError(f"{url}: {exc} ({failures} interrupted transfers in a row)") from None
            fetcher.log(f"  the transfer broke ({exc or type(exc).__name__}); resuming at {human_bytes(have)}")
            fetcher.wait(fetcher.backoff(failures - 1))
            continue
        finally:
            response.close()
        if size is None or have >= size:
            break
        failures += 1  # the server closed the transfer early without an error
        if failures > fetcher.retries:
            raise FetchError(f"{url}: the transfer keeps ending early")
    actual = os.path.getsize(part)
    if size is not None and actual != size:
        os.remove(part)
        raise DumpError(f"{os.path.basename(path)} arrived as {actual:,} bytes, not {size:,}; deleted it - run again")
    if verify and (sha1 or md5):
        algorithm, want = ("sha1", sha1) if sha1 else ("md5", md5)
        got = file_digest(part, algorithm)
        if got != want:
            os.remove(part)
            raise DumpError(f"{os.path.basename(path)}: {algorithm} {got} is not the published {want}; deleted it - run again")
    os.replace(part, path)
    return "saved"


# ---------------------------------------------------------------------------
# reading a dump
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SiteInfo:
    """The ``<siteinfo>`` that opens a dump: which wiki, where it lives, its namespaces."""

    name: str = ""
    dbname: str = ""
    base: str = ""
    namespaces: dict[int, str] = dataclasses.field(default_factory=dict)

    def article_url(self, title: str) -> str:
        if "/" not in self.base:
            return ""
        root = self.base.rsplit("/", 1)[0] + "/"
        return root + urllib.parse.quote(title.replace(" ", "_"), safe="/:()_,'!*-.~@$;")


@dataclasses.dataclass
class Page:
    """One ``<page>`` of a dump, with the text of its last revision."""

    id: int
    ns: int
    title: str
    text: str = ""
    redirect: str | None = None
    revision: int | None = None
    timestamp: str = ""


def open_dump(path: str) -> Any:
    """``path`` opened for reading as bytes, decompressed if it is bzip2 or gzip."""
    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic.startswith(b"BZh"):
        return bz2.open(path, "rb")
    if magic.startswith(b"\x1f\x8b"):
        return gzip.open(path, "rb")
    if magic.startswith(b"7z\xbc\xaf\x27\x1c"):
        raise ExtractError(f"{path} is 7-Zip, which Python cannot read; use the .bz2 dump instead")
    return open(path, "rb")


class DumpReader:
    """Streams the pages out of a MediaWiki XML dump without keeping them in memory.

    ``site`` is read on opening; :meth:`pages` then yields every page. Works on
    any export schema version, and on the multistream files, which are a series
    of bzip2 streams that Python reads as one.
    """

    def __init__(self, path: str):
        self.path = path
        try:
            self._fh = open_dump(path)
            self._events = ET.iterparse(self._fh, events=("start", "end"))
            _, self._root = next(self._events)
        except (OSError, EOFError, ET.ParseError, StopIteration) as exc:
            raise ExtractError(f"{path} is not a readable MediaWiki dump: {exc or type(exc).__name__}") from None
        tag = self._root.tag
        self.ns = tag[: tag.index("}") + 1] if "}" in tag else ""
        self.site = SiteInfo()
        # One stream for the header and the pages alike. (Not `yield from` the
        # parser: a wrapper that is dropped would close it on Pythons where
        # iterparse's iterator has a close().)
        self._stream = self._read()
        for event, elem in self._stream:
            if event == "end" and elem.tag == self.ns + "siteinfo":
                self.site = self._siteinfo(elem)
                self._root.clear()
                break
            if event == "start" and elem.tag == self.ns + "page":
                break  # a dump without a siteinfo

    def __enter__(self) -> DumpReader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._fh.close()

    def pages(self) -> Iterator[Page]:
        for event, elem in self._stream:
            if event == "end" and elem.tag == self.ns + "page":
                yield self._page(elem)
                self._root.clear()

    def _read(self) -> Iterator[tuple[str, ET.Element]]:
        while True:
            try:
                item = next(self._events)
            except StopIteration:
                return
            except (OSError, EOFError, ET.ParseError) as exc:
                raise ExtractError(f"{self.path} is damaged or cut short: {exc or type(exc).__name__}") from None
            yield item

    def _siteinfo(self, elem: ET.Element) -> SiteInfo:
        q = self.ns
        spaces = {}
        for space in elem.iter(q + "namespace"):
            key = _int(space.get("key"))
            if key is not None:
                spaces[key] = space.text or ""
        return SiteInfo(
            name=elem.findtext(q + "sitename") or "",
            dbname=elem.findtext(q + "dbname") or "",
            base=elem.findtext(q + "base") or "",
            namespaces=spaces,
        )

    def _page(self, elem: ET.Element) -> Page:
        q = self.ns
        revisions = elem.findall(q + "revision")
        last = revisions[-1] if revisions else None
        redirect = elem.find(q + "redirect")

        def revision(tag: str) -> str:
            return (last.findtext(q + tag) if last is not None else None) or ""

        return Page(
            id=_int(elem.findtext(q + "id")) or 0,
            ns=_int(elem.findtext(q + "ns")) or 0,
            title=elem.findtext(q + "title") or "",
            text=revision("text"),
            redirect=None if redirect is None else redirect.get("title", ""),
            revision=_int(revision("id")),
            timestamp=revision("timestamp"),
        )


# ---------------------------------------------------------------------------
# wikitext to plain text
# ---------------------------------------------------------------------------

DROP_SECTIONS = frozenset(
    {
        "references", "notes", "footnotes", "citations", "sources", "bibliography", "works cited",
        "notes and references", "references and notes", "general references", "further reading",
        "external links", "external link", "see also", "related pages",
    }
)  # fmt: skip
"""Sections that hold lists of links and citations rather than prose; ``--keep-sections`` keeps them."""

_DROPPED_TAGS = (
    "ref|references|gallery|timeline|imagemap|graph|mapframe|maplink|templatedata|templatestyles|score|math|chem"
    "|ce|hiero|inputbox|categorytree|syntaxhighlight|source|includeonly|indicator|section"
)
_DROPPED_EMPTY = re.compile(rf"<(?:{_DROPPED_TAGS})\b[^>]*/>", re.IGNORECASE)
_DROPPED = re.compile(rf"<({_DROPPED_TAGS})\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
_BRACES = re.compile(r"\{\{|\}\}")
_BRACKETS = re.compile(r"\[\[|\]\]")
_PARAM_TOKENS = re.compile(r"\{\{|\}\}|\[\[|\]\]|\|")
_NAMED = re.compile(r"\s*([^=\[\]{}|]+?)\s*=")
_TABLE_EDGE = re.compile(r"^[ \t:]*(\{\||\|\})", re.MULTILINE)
_LINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_INTERLANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]+)*:")
_EXTERNAL = re.compile(r"\[(?:(?:https?|ftp)://|//|mailto:)[^\s\]]*(?:[ \t]+([^\]\n]*))?\]")
_QUOTES = re.compile(r"'{2,}")
_BR = re.compile(r"<br\s*/?\s*>", re.IGNORECASE)
_HTML_TAGS = (
    "abbr|b|bdi|bdo|big|blockquote|caption|center|cite|code|col|colgroup|data|dd|del|dfn|div|dl|dt|em|font"
    "|h[1-6]|hr|i|ins|kbd|li|mark|noinclude|nowiki|ol|onlyinclude|p|poem|pre|q|rb|rp|rt|rtc|ruby|s|samp"
    "|small|span|strike|strong|sub|sup|table|tbody|td|tfoot|th|thead|time|tr|tt|u|ul|var|wbr"
)
_TAG = re.compile(rf"</?(?:{_HTML_TAGS})\b[^<>]*>", re.IGNORECASE)
_MAGIC = re.compile(r"__[A-Z]+__")
_HEADING = re.compile(r"^(={1,6})\s*(.+?)\s*\1\s*$")
_REDIRECT = re.compile(r"\s*#\s*redirect\b", re.IGNORECASE)
_DISAMBIGUATION = re.compile(
    r"\{\{\s*(?:disambiguation|disambig|dab|disamb|hndis|geodis|numberdis|mathdab|begriffsklärung|homonymie"
    r"|desambiguación|disambigua|doorverwijspagina|неоднозначность|многозначность)\s*[|}]",
    re.IGNORECASE,
)
_FILE_ALIASES = frozenset(
    {"file", "image", "media", "bild", "datei", "fichier", "imagen", "archivo", "immagine", "afbeelding", "plik",
     "arquivo", "файл", "изображение", "ファイル", "画像", "文件", "图像", "파일"}
)  # fmt: skip
_FIRST_ARGUMENT = frozenset(
    {"nowrap", "nobr", "small", "smaller", "big", "larger", "nobold", "noitalic", "em", "strong", "var", "mvar",
     "math", "abbr", "sic", "flag", "flagcountry", "flagu", "nihongo", "keypress", "tooltip"}
)  # fmt: skip
_SECOND_ARGUMENT = frozenset({"lang", "script", "wikt-lang"})
_LAST_ARGUMENT = frozenset({"transl", "transliteration"})
_QUOTE_TEMPLATES = frozenset({"quote", "blockquote", "cquote", "quotation"})
_LITERALS = {
    "snd": " – ", "spaced ndash": " – ", "spaced en dash": " – ", "ndash": "–", "mdash": "—",
    "nbsp": " ", "dot": " · ", "middot": "·", "'": "'",
}  # fmt: skip
_RANGE_WORDS = frozenset({"-", "–", "to", "and", "or", "by", "x", "×", "+/-", "±"})
_MONTHS = ("January February March April May June July August September October November December").split()


def _split_params(inner: str) -> list[str]:
    """A template's name and parameters, split on the pipes not inside nested links or templates."""
    parts, depth, begin = [], 0, 0
    for match in _PARAM_TOKENS.finditer(inner):
        token = match.group()
        if token in ("{{", "[["):
            depth += 1
        elif token in ("}}", "]]"):
            depth = max(0, depth - 1)
        elif depth == 0:
            parts.append(inner[begin : match.start()])
            begin = match.end()
    parts.append(inner[begin:])
    return parts


def _apostrophes(match: re.Match) -> str:
    """Wiki bold and italic: 2, 3 and 5 apostrophes are markup, 4 is markup plus a real one."""
    count = len(match.group())
    return "'" if count == 4 else "'" * max(0, count - 5)


def strip_tables(text: str) -> str:
    """``text`` without ``{| ... |}`` tables, nested or not. An unclosed table is left in."""
    if "{|" not in text:
        return text
    out: list[str] = []
    depth = start = last = 0
    for match in _TABLE_EDGE.finditer(text):
        if match.group(1) == "{|":
            if depth == 0:
                start = match.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                out.append(text[last:start])
                end = text.find("\n", match.end())
                last = len(text) if end == -1 else end
    out.append(text[last:])
    return "".join(out)


_OPEN_PUNCTUATION = re.compile(r"\(\s*[,;:]+\s*")
_CLOSE_PUNCTUATION = re.compile(r"\s*[,;:]+\s*\)")
_EMPTY_PARENS = re.compile(r"\(\s*\)")
_SPACE_BEFORE = re.compile(r"[ \t]+([,.)])")
_SPACES = re.compile(r"[ \t]{2,}")


def tidy_text(text: str, keep_sections: bool = False) -> str:
    """Plain text from what is left once the markup is gone.

    Headings become lines of their own; the reference-and-link sections go
    unless ``keep_sections``, and so does any heading left with nothing under
    it; list markers, table debris and the stray punctuation of removed
    templates ("Paris ( ; French: ...)") are cleaned up.
    """
    items: list[tuple[int, str]] = []  # (heading level, or 0 for text; the line)
    skip_below: int | None = None
    for raw in text.split("\n"):
        line = raw.strip()
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_below is not None and level > skip_below:
                continue
            skip_below = None
            title = heading.group(2).strip()
            if not keep_sections and fold(title).strip(" :") in DROP_SECTIONS:
                skip_below = level
                continue
            items.append((level, title))
            continue
        if skip_below is not None:
            continue
        if line.startswith(("{|", "|}", "|-", "|", "!")) or re.fullmatch(r"-{4,}", line):
            continue
        line = re.sub(r"^[*#:;]+\s*", "", line)
        line = _OPEN_PUNCTUATION.sub("(", line)
        line = _CLOSE_PUNCTUATION.sub(")", line)
        line = _EMPTY_PARENS.sub("", line)
        line = _SPACE_BEFORE.sub(r"\1", line)
        line = _SPACES.sub(" ", line).strip()
        if line and not re.search(r"\w", line):
            continue
        items.append((0, line))
    lines: list[str] = []
    for index, (level, line) in enumerate(items):
        if level:
            following = next(((lv, ln) for lv, ln in items[index + 1 :] if lv or ln), None)
            if following is None or (following[0] and following[0] <= level):
                continue  # nothing under this heading
            lines += ["", line, ""]
        else:
            lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class WikitextConverter:
    """Wikitext to plain text, without expanding templates.

    Templates, tables, references, files and categories go - the categories are
    returned alongside the text - links become their labels, and a few inline
    templates that carry words ({{convert}}, {{lang}}, {{nowrap}}, ...) are
    rendered so a sentence does not lose its middle. ``namespaces`` (from the
    dump's siteinfo) supplies the local names of File: and Category:.
    """

    def __init__(self, namespaces: dict[int, str] | None = None, *, keep_sections: bool = False):
        names = namespaces or {}
        self.keep_sections = keep_sections
        self._file_prefixes = _FILE_ALIASES | {fold(names[k]) for k in (6, -2) if names.get(k)}
        categories = {"Category"} | ({names[14]} if names.get(14) else set())
        pattern = "|".join(re.escape(name) for name in sorted(categories, key=len, reverse=True))
        self._category = re.compile(rf"\[\[\s*(?:{pattern})\s*:\s*([^\]|]*)(?:\|[^\]]*)?\]\]", re.IGNORECASE)

    def convert(self, wikitext: str) -> tuple[str, list[str]]:
        """The plain text of ``wikitext``, and the categories it puts itself in."""
        text = _COMMENT.sub("", wikitext)
        text = _DROPPED_EMPTY.sub("", text)
        text = _DROPPED.sub("", text)
        categories = list(dict.fromkeys(name.strip() for name in self._category.findall(text) if name.strip()))
        text = self._category.sub("", text)
        text = self.strip_templates(text)
        text = strip_tables(text)
        text = self._strip_files(text)
        text = _LINK.sub(self._link, text)
        text = _EXTERNAL.sub(lambda m: m.group(1) or "", text)
        text = _QUOTES.sub(_apostrophes, text)
        text = _BR.sub("\n", text)
        text = _TAG.sub("", text)
        text = _MAGIC.sub("", text)
        text = html.unescape(text).replace(" ", " ")
        return tidy_text(text, self.keep_sections), categories

    def strip_templates(self, text: str) -> str:
        """``text`` without templates, however deeply nested. An unclosed ``{{`` is left in."""
        if "{{" not in text:
            return text
        out: list[str] = []
        depth = start = last = 0
        for match in _BRACES.finditer(text):
            if match.group() == "{{":
                if depth == 0:
                    start = match.start()
                depth += 1
            elif depth:
                depth -= 1
                if depth == 0:
                    out.append(text[last:start])
                    out.append(self._render(text[start + 2 : match.start()]))
                    last = match.end()
        out.append(text[last:])
        return "".join(out)

    def _render(self, inner: str) -> str:
        """What an inline template reads as in running text; nothing, for every other template."""
        params = _split_params(inner)
        name = " ".join(fold(params[0]).replace("_", " ").split())
        args = [p.strip() for p in params[1:] if not _NAMED.match(p)]
        named = {m.group(1).strip().lower(): p[m.end() :].strip() for p in params[1:] if (m := _NAMED.match(p))}
        if name in _FIRST_ARGUMENT or name.startswith("lang-"):
            text = args[0] if args else ""
        elif name in _SECOND_ARGUMENT:
            text = args[1] if len(args) > 1 else ""
        elif name in _LAST_ARGUMENT:
            text = args[-1] if args else ""
        elif name == "ill":
            text = named.get("lt") or (args[0] if args else "")
        elif name in ("convert", "cvt"):
            text = self._convert_units(args)
        elif name in ("circa", "c."):
            text = f"c. {args[0]}" if args else "c."
        elif name == "as of":
            month = _int(args[1]) if len(args) > 1 else None
            text = "As of " + (f"{_MONTHS[month - 1]} " if month and 1 <= month <= 12 else "") + (args[0] if args else "")
        elif name in ("frac", "sfrac"):
            text = {1: "1/{0}", 2: "{0}/{1}", 3: "{0} {1}/{2}"}.get(len(args), "").format(*args)
        elif name in _QUOTE_TEMPLATES:
            text = named.get("text") or named.get("quote") or (args[0] if args else "")
        elif name == "val":
            text = " ".join(part for part in (args[0] if args else "", named.get("u", "")) if part)
        elif name == "sortname":
            text = " ".join(args[:2])
        elif name in _LITERALS:
            return _LITERALS[name]
        else:
            return ""
        return self.strip_templates(text)

    @staticmethod
    def _convert_units(args: list[str]) -> str:
        if len(args) >= 4 and args[1] in _RANGE_WORDS:
            joiner = "–" if args[1] in ("-", "–") else f" {args[1]} "
            return f"{args[0]}{joiner}{args[2]} {args[3]}"
        return " ".join(args[:2])

    def _strip_files(self, text: str) -> str:
        """``text`` without [[File:...]] links, captions and all, however many links the captions hold."""
        if "[[" not in text:
            return text
        out: list[str] = []
        depth = start = last = 0
        for match in _BRACKETS.finditer(text):
            if match.group() == "[[":
                if depth == 0:
                    start = match.start()
                depth += 1
            elif depth:
                depth -= 1
                if depth == 0 and self._is_file(text[start + 2 : match.start()]):
                    out.append(text[last:start])
                    last = match.end()
        out.append(text[last:])
        return "".join(out)

    def _is_file(self, target: str) -> bool:
        prefix, colon, _ = target.partition(":")
        return bool(colon) and fold(prefix).strip() in self._file_prefixes

    @staticmethod
    def _link(match: re.Match) -> str:
        target, label = match.group(1).strip(), match.group(2)
        if label is not None:
            if label.strip():
                return label.strip()
            # the pipe trick: [[Paris, Texas|]] reads "Paris"
            short = target.lstrip(":").split(":", 1)[-1].split(",")[0]
            return re.sub(r"\s*\([^)]*\)\s*$", "", short).strip()
        if target.startswith(":"):
            return target[1:]
        if _INTERLANGUAGE.match(target):
            return ""
        return target


# ---------------------------------------------------------------------------
# extracting
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExtractOptions:
    """Which pages of a dump become articles."""

    namespaces: frozenset[int] = frozenset({0})
    titles: tuple[TextQuery, ...] = ()
    not_titles: tuple[TextQuery, ...] = ()
    categories: tuple[TextQuery, ...] = ()
    not_categories: tuple[TextQuery, ...] = ()
    min_chars: int = 0
    keep_sections: bool = False
    keep_disambiguation: bool = False


def convert_page(page: Page, converter: WikitextConverter, options: ExtractOptions) -> tuple[str, dict | None]:
    """``("ok", article)`` for a page that becomes an article, else ``(why not, None)``."""
    if not options.keep_disambiguation and _DISAMBIGUATION.search(page.text):
        return "disambiguation pages", None
    try:
        text, categories = converter.convert(page.text)
    except RecursionError:  # templates nested thousands deep: one page must not end a run of millions
        return "nested too deeply to convert", None
    if options.categories and not any_match(options.categories, categories):
        return "outside the categories", None
    if options.not_categories and any_match(options.not_categories, categories):
        return "in an excluded category", None
    if len(text) < max(1, options.min_chars):
        return "too short", None
    return "ok", {
        "id": page.id,
        "title": page.title,
        "revision": page.revision,
        "timestamp": page.timestamp,
        "categories": categories,
        "text": text,
    }


_worker: tuple[WikitextConverter, ExtractOptions] | None = None


def _start_worker(namespaces: dict[int, str], options: ExtractOptions) -> None:
    global _worker
    _worker = (WikitextConverter(namespaces, keep_sections=options.keep_sections), options)


def _convert_in_worker(page: Page) -> tuple[str, dict | None]:
    assert _worker is not None
    return convert_page(page, *_worker)


def _candidates(paths: Sequence[str], options: ExtractOptions, counts: collections.Counter) -> Iterator[Page]:
    """The pages of every dump that pass the checks needing no conversion."""
    for path in paths:
        with DumpReader(path) as reader:
            for page in reader.pages():
                counts["pages read"] += 1
                if page.ns not in options.namespaces:
                    counts["other namespaces"] += 1
                elif page.redirect is not None or _REDIRECT.match(page.text):
                    counts["redirects"] += 1
                elif (options.titles and not any_match(options.titles, [page.title])) or (
                    options.not_titles and any_match(options.not_titles, [page.title])
                ):
                    counts["filtered by title"] += 1
                else:
                    yield page


def _pooled(pool: Any, pages: Iterator[Page], processes: int) -> Iterator[tuple[str, dict | None]]:
    """``pages`` converted by ``pool``, in order, one batch in flight while the next is read.

    ``Pool.imap`` would read the whole dump ahead into its queue; batches keep
    memory flat while the main process parses and the workers convert.
    """
    batch = processes * 64
    pending = None
    while True:
        chunk = list(itertools.islice(pages, batch))
        job = pool.map_async(_convert_in_worker, chunk, chunksize=max(1, len(chunk) // (4 * processes))) if chunk else None
        if pending is not None:
            yield from pending.get()
        if job is None:
            return
        pending = job


class JsonlShards:
    """Articles as JSON lines, about ``shard_bytes`` to a file: wiki_00000.jsonl, wiki_00001.jsonl, ..."""

    def __init__(self, directory: str, shard_bytes: int = 100_000_000, compress: bool = False):
        self.directory = directory
        self.shard_bytes = shard_bytes
        self.compress = compress
        self.files: list[str] = []
        self._fh: TextIO | None = None
        self._size = 0

    def write(self, article: dict) -> None:
        if self._fh is None or self._size >= self.shard_bytes:
            self.close()
            name = f"wiki_{len(self.files):05d}.jsonl" + (".gz" if self.compress else "")
            path = os.path.join(self.directory, name)
            self._fh = gzip.open(path, "wt", encoding="utf-8") if self.compress else open(path, "w", encoding="utf-8")
            self.files.append(name)
            self._size = 0
        line = json.dumps(article, ensure_ascii=False) + "\n"
        self._fh.write(line)
        self._size += len(line)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class TextFiles:
    """One .txt file per article - the title, a blank line, the text - a thousand to a folder."""

    PER_FOLDER = 1000

    def __init__(self, directory: str):
        self.directory = directory
        self.files: list[str] = []  # the folders
        self.count = 0

    def write(self, article: dict) -> None:
        folder = f"{self.count // self.PER_FOLDER:05d}"
        if not self.files or self.files[-1] != folder:
            os.makedirs(os.path.join(self.directory, folder), exist_ok=True)
            self.files.append(folder)
        name = f"{article['id']}-{slugify(article['title']) or 'untitled'}.txt"
        with open(os.path.join(self.directory, folder, name), "w", encoding="utf-8") as fh:
            fh.write(f"{article['title']}\n\n{article['text']}\n")
        self.count += 1

    def close(self) -> None:
        pass


def extract(
    paths: Sequence[str],
    sink: JsonlShards | TextFiles,
    options: ExtractOptions,
    counts: collections.Counter,
    *,
    processes: int = 1,
    limit: int | None = None,
    on_progress: Callable[[collections.Counter], None] | None = None,
) -> SiteInfo:
    """Write the articles of ``paths`` (dumps of one wiki, in order) to ``sink``; tally in ``counts``."""
    with DumpReader(paths[0]) as first:
        site = first.site
    pages = _candidates(paths, options, counts)
    pool = multiprocessing.Pool(processes, _start_worker, (site.namespaces, options)) if processes > 1 else None
    results: Iterator[tuple[str, dict | None]] | None = None
    try:
        if pool is not None:
            results = _pooled(pool, pages, processes)
        else:
            converter = WikitextConverter(site.namespaces, keep_sections=options.keep_sections)
            results = (convert_page(page, converter, options) for page in pages)
        for status, article in results:
            if article is None:
                counts[status] += 1
            else:
                sink.write({"id": article["id"], "title": article["title"], "url": site.article_url(article["title"]), **article})
                counts["articles"] += 1
            if on_progress:
                on_progress(counts)
            if limit and counts["articles"] >= limit:
                break
    finally:
        if results is not None:
            results.close()
        pages.close()  # closes the dump being read, if the limit stopped us early
        if pool is not None:
            pool.terminate()
            pool.join()
    return site


def prepare_output(directory: str, overwrite: bool) -> None:
    """Make ``directory`` ready for a fresh extract; with ``overwrite``, remove the last one's output."""
    if os.path.isdir(directory) and os.listdir(directory):
        if not overwrite:
            raise ExtractError(f"{directory} is not empty; choose another -o or pass --overwrite")
        try:
            with open(os.path.join(directory, EXTRACT_MANIFEST), encoding="utf-8") as fh:
                previous = json.load(fh).get("files", [])
        except (OSError, ValueError, AttributeError):
            previous = []
        for name in previous:  # only what the last extract wrote, by its own record
            if isinstance(name, str) and re.fullmatch(r"wiki_\d{5}\.jsonl(?:\.gz)?|\d{5}", name):
                path = os.path.join(directory, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.exists(path):
                    os.remove(path)
    os.makedirs(directory, exist_ok=True)


def _extract_progress(stream: TextIO, clock: Callable[[], float] = time.monotonic) -> Callable[[collections.Counter], None]:
    started = clock()
    shown = [started]
    tty = stream.isatty()

    def show(counts: collections.Counter) -> None:
        now = clock()
        if now - shown[0] < (0.5 if tty else 30):
            return
        shown[0] = now
        rate = counts["pages read"] / max(now - started, 1e-3)
        line = f"  {counts['pages read']:,} pages read, {counts['articles']:,} articles written ({rate:,.0f} pages/s)"
        if tty:
            stream.write("\r\033[K" + line)
            stream.flush()
        else:
            print(line, file=stream, flush=True)

    return show


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------


class WikiApi:
    """The MediaWiki Action API, called politely: JSON, one request at a time, ``maxlag`` honoured.

    ``maxlag=5`` asks the servers to refuse the request while their replicas
    lag by more than five seconds; when they do, this waits as long as they say
    and asks again.
    """

    MAXLAG_TRIES = 10

    def __init__(self, fetcher: Fetcher, endpoint: str, log: Callable[[str], None] = lambda message: None):
        self.fetcher = fetcher
        self.endpoint = endpoint
        self.log = log

    def call(self, **params: str) -> dict:
        query = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        url = f"{self.endpoint}?{urllib.parse.urlencode(query)}"
        for attempt in range(self.MAXLAG_TRIES + 1):
            reply = self.fetcher.fetch(url)
            try:
                data = json.loads(reply.body)
            except ValueError:
                raise ApiError(f"{self.endpoint} did not answer with JSON - is it a MediaWiki API?") from None
            error = data.get("error") if isinstance(data, dict) else None
            if not error:
                return data if isinstance(data, dict) else {}
            if error.get("code") == "maxlag" and attempt < self.MAXLAG_TRIES:
                wait = _int(reply.headers.get("Retry-After")) or 5
                self.log(f"  the servers are busy ({error.get('info', 'maxlag')}); waiting {wait}s")
                self.fetcher.wait(wait)
                continue
            raise ApiError(f"{error.get('code', 'error')}: {error.get('info', '')}".rstrip(": "))
        raise ApiError("the servers stayed too busy to answer (maxlag); try again later")

    def query(self, **params: str) -> Iterator[dict]:
        """Every batch of a query's results, following the API's continuation."""
        more: dict = {}
        while True:
            data = self.call(**{"action": "query", **params, **more})
            yield data.get("query") or {}
            if "continue" not in data:
                return
            more = data["continue"]


def title_pages(api: WikiApi, titles: Sequence[str], log: Callable[[str], None]) -> Iterator[dict]:
    """The articles named by ``titles``, redirects followed, fifty to a request."""
    for start in range(0, len(titles), 50):
        data = api.call(action="query", titles="|".join(titles[start : start + 50]), redirects="1")
        for page in (data.get("query") or {}).get("pages", []):
            if page.get("missing") or page.get("invalid") or "pageid" not in page:
                log(f"  no article called {page.get('title', '?')!r}")
                continue
            yield {"pageid": page["pageid"], "title": page["title"]}


def category_pages(api: WikiApi, category: str, depth: int = 0) -> Iterator[dict]:
    """The articles in ``category`` and, ``depth`` levels down, in its subcategories."""
    start = category if re.match(r"(?i)category\s*:", category) else f"Category:{category}"
    queue = collections.deque([(start, 0)])
    seen: set[str] = set()
    while queue:
        title, level = queue.popleft()
        if title in seen:
            continue
        seen.add(title)
        for batch in api.query(list="categorymembers", cmtitle=title, cmtype="page|subcat", cmnamespace="0|14", cmprop="ids|title|type", cmlimit="max"):
            for member in batch.get("categorymembers", []):
                if member.get("type") == "subcat" or member.get("ns") == 14:
                    if level < depth:
                        queue.append((member["title"], level + 1))
                elif member.get("ns") == 0:
                    yield {"pageid": member["pageid"], "title": member["title"]}


def search_pages(api: WikiApi, query: str) -> Iterator[dict]:
    """The articles a Wikipedia search finds, best first (the API stops at 10,000)."""
    for batch in api.query(list="search", srsearch=query, srnamespace="0", srlimit="max", srprop="", srinfo=""):
        for hit in batch.get("search", []):
            yield {"pageid": hit["pageid"], "title": hit["title"]}


def random_pages(api: WikiApi, count: int) -> Iterator[dict]:
    """``count`` different articles picked at random (fewer if the wiki runs out)."""
    seen: set[int] = set()
    stale = 0
    while len(seen) < count and stale < 5:
        data = api.call(action="query", list="random", rnnamespace="0", rnfilterredir="nonredirects", rnlimit=str(min(500, count - len(seen))))
        fresh = 0
        for page in (data.get("query") or {}).get("random", []):
            if page["id"] not in seen and len(seen) < count:
                seen.add(page["id"])
                fresh += 1
                yield {"pageid": page["id"], "title": page["title"]}
        stale = 0 if fresh else stale + 1


def fetch_extracts(api: WikiApi, pages: Sequence[dict], intro: bool = False) -> list[dict]:
    """TextExtracts' plain text for ``pages`` (1 at a time; up to 20 with ``intro``), with URL and revision."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "redirects": "1",
        "pageids": "|".join(str(page["pageid"]) for page in pages),
    }
    if intro:
        params.update(exintro="1", exlimit=str(len(pages)))
    data = api.call(**params)
    return [page for page in (data.get("query") or {}).get("pages", []) if "extract" in page]


class ArticleSaver:
    """Writes fetched articles - a .txt each, or lines of one articles.jsonl - and keeps the manifest."""

    def __init__(self, output: str, fmt: str = "txt", *, keep_sections: bool = False, min_chars: int = 0):
        self.output = output
        self.fmt = fmt
        self.keep_sections = keep_sections
        self.min_chars = min_chars
        os.makedirs(output, exist_ok=True)
        self.manifest = Manifest(os.path.join(output, MANIFEST), key="pageid")

    @staticmethod
    def filename(pageid: int, title: str) -> str:
        return f"{pageid}-{slugify(title) or 'untitled'}.txt"

    def have(self, page: dict) -> bool:
        """Whether ``page`` was saved by an earlier run (and is still there)."""
        entry = self.manifest.get(page["pageid"])
        if self.fmt == "txt":
            if os.path.exists(os.path.join(self.output, self.filename(page["pageid"], page["title"]))):
                return True
            return entry is not None and os.path.exists(os.path.join(self.output, entry.get("file", "")))
        return entry is not None and os.path.exists(os.path.join(self.output, FETCHED_JSONL))

    def save(self, page: dict) -> tuple[str, str]:
        """Save one TextExtracts page; returns (status, detail)."""
        text = tidy_text(page.get("extract") or "", self.keep_sections)
        if len(text) < max(1, self.min_chars):
            return "short", f"{len(text):,} characters"
        pageid, title, url = page["pageid"], page["title"], page.get("fullurl", "")
        if self.fmt == "txt":
            name = self.filename(pageid, title)
            body = f"{title}\n\n{text}\n".encode("utf-8")
            write_atomic(os.path.join(self.output, name), body)
        else:
            name = FETCHED_JSONL
            record = {"id": pageid, "title": title, "url": url, "revision": page.get("lastrevid"), "text": text}
            body = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            with open(os.path.join(self.output, name), "ab") as fh:
                fh.write(body)
        self.manifest.add(
            {
                "pageid": pageid,
                "title": title,
                "url": url,
                "revision": page.get("lastrevid"),
                "file": name,
                "chars": len(text),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "fetched": utc_now(),
            }
        )
        return "saved", human_bytes(len(body))


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

EPILOG = """\
All of Wikipedia is 'dump' then 'extract': the official dumps, not a crawl of
the site. 'fetch' is for chosen articles.

examples:
  %(prog)s dump --lang simple --list        the newest Simple English dump, without downloading
  %(prog)s dump --lang en                   all of English Wikipedia (over 20 GB compressed)
  %(prog)s extract wikipedia_dumps/enwiki-*-pages-articles.xml.bz2 --min-chars 500
  %(prog)s fetch -c "Quantum mechanics" --depth 1 -n 300
  %(prog)s fetch --random 100 --lang de --format jsonl

Wikimedia asks automated clients to say who is running them: pass --contact
(an email address or a URL) or set WIKIMEDIA_CONTACT. Run '%(prog)s COMMAND -h'
for each command's options.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wikipedia_download.py",
        description="Download Wikipedia: all of it from the official dumps, or chosen articles through the API.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    network = argparse.ArgumentParser(add_help=False)
    net = network.add_argument_group("network")
    net.add_argument("--delay", type=at_least(0, float), metavar="SECONDS", help="pause between requests")
    net.add_argument("--retries", type=at_least(0), default=3, metavar="N", help="retries for a failed request (default: %(default)s)")
    net.add_argument("--timeout", type=at_least(1, float), default=60.0, metavar="SECONDS", help="network timeout (default: %(default)s)")
    net.add_argument("--contact", default=os.environ.get("WIKIMEDIA_CONTACT"), metavar="WHO", help="email or URL for the User-Agent (default: $WIKIMEDIA_CONTACT)")
    net.add_argument("-q", "--quiet", action="store_true", help="print only warnings, errors and the summary")

    dump = commands.add_parser(
        "dump",
        parents=[network],
        help="download the official database dumps - all of Wikipedia (resumable, checksummed)",
        description="Download a Wikimedia database dump: the newest finished one, resumable, checked against its SHA-1.",
        epilog="kinds:\n" + "\n".join(f"  {k:<13} {v[2]}" for k, v in DUMP_KINDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dump.add_argument("--lang", default="en", metavar="CODE", help="which Wikipedia: en, de, fr, simple, ... (default: en)")
    dump.add_argument("--wiki", metavar="NAME", help="any wiki by dump name instead: enwiktionary, dewikibooks, ...")
    dump.add_argument("--date", default="latest", metavar="YYYYMMDD", help="a particular dump (default: the newest finished one)")
    dump.add_argument("--kind", choices=DUMP_KINDS, default="articles", help="what to download (default: articles)")
    dump.add_argument("--parts", action="store_true", help="the numbered parts rather than one whole file, where both exist")
    dump.add_argument("-o", "--output", default=DEFAULT_DUMP_DIR, metavar="DIR", help=f"where dumps go (default: {DEFAULT_DUMP_DIR})")
    dump.add_argument("--list", action="store_true", help="show the files and their sizes; download nothing")
    dump.add_argument("--mirror", metavar="URL", help="a mirror of dumps.wikimedia.org (https://dumps.wikimedia.org/mirrors.html)")
    dump.add_argument("--no-verify", action="store_true", help="skip the checksum (the size is still checked)")
    dump.add_argument("--force", action="store_true", help="download even if the disk looks too full")

    ext = commands.add_parser(
        "extract",
        help="turn a downloaded dump into plain text (JSON lines, or a file per article)",
        description="Stream the articles of a dump (.xml.bz2, .xml.gz or .xml) into plain text.",
    )
    ext.add_argument("dumps", nargs="+", metavar="DUMP", help="dump files of one wiki, e.g. all the parts of a split dump")
    ext.add_argument("-o", "--output", default=DEFAULT_TEXT_DIR, metavar="DIR", help=f"where the text goes (default: {DEFAULT_TEXT_DIR})")
    ext.add_argument("--format", choices=("jsonl", "txt"), default="jsonl", help="JSON lines in shards, or a .txt per article (default: jsonl)")
    ext.add_argument("--gzip", action="store_true", help="compress the JSON-lines shards")
    ext.add_argument("--shard-mb", type=at_least(1), default=100, metavar="MB", help="size of a JSON-lines shard (default: %(default)s)")
    ext.add_argument("--namespace", type=int, action="append", metavar="N", help="namespace to take (default: 0, the articles)")
    ext.add_argument("-t", "--title", action="append", metavar="TEXT", help="keep articles whose title matches TEXT")
    ext.add_argument("--exclude-title", action="append", metavar="TEXT", help="drop articles whose title matches TEXT")
    ext.add_argument("-c", "--category", action="append", metavar="TEXT", help="keep articles in a category matching TEXT")
    ext.add_argument("--exclude-category", action="append", metavar="TEXT", help="drop articles in a category matching TEXT")
    ext.add_argument("--regex", action="store_true", help="title and category filters are regular expressions, not words")
    ext.add_argument("--min-chars", type=at_least(0), default=0, metavar="N", help="drop articles shorter than N characters of text")
    ext.add_argument("--keep-sections", action="store_true", help="keep References, External links, See also and the like")
    ext.add_argument("--keep-disambiguation", action="store_true", help="keep disambiguation pages")
    ext.add_argument("-n", "--limit", type=at_least(1), metavar="N", help="stop after N articles")
    ext.add_argument("-p", "--processes", type=at_least(1), default=max(1, min(8, (os.cpu_count() or 2) - 1)), metavar="N", help="conversion processes (default: %(default)s)")
    ext.add_argument("--overwrite", action="store_true", help="replace the output of an earlier extract in DIR")
    ext.add_argument("-q", "--quiet", action="store_true", help="print only warnings, errors and the summary")

    get = commands.add_parser(
        "fetch",
        parents=[network],
        help="download chosen articles through the API: by title, category, search or at random",
        description="Download articles as plain text through the MediaWiki API, one request at a time.",
    )
    source = get.add_argument_group("which articles (any combination)")
    source.add_argument("-t", "--title", action="append", metavar="TITLE", help="an article by title")
    source.add_argument("--titles-file", metavar="FILE", help="titles, one per line")
    source.add_argument("-c", "--category", action="append", metavar="NAME", help="the articles in a category")
    source.add_argument("--depth", type=at_least(0), default=0, metavar="N", help="and in its subcategories, N levels down (default: 0)")
    source.add_argument("-s", "--search", action="append", metavar="QUERY", help="the articles a search finds")
    source.add_argument("--random", type=at_least(1), metavar="N", help="N articles at random")
    where = get.add_argument_group("where from and to")
    where.add_argument("--lang", default="en", metavar="CODE", help="which Wikipedia (default: en)")
    where.add_argument("--site", metavar="HOST", help="another wiki, e.g. en.wiktionary.org")
    where.add_argument("--api", metavar="URL", help="the full api.php URL of any MediaWiki")
    where.add_argument("-o", "--output", default=DEFAULT_FETCH_DIR, metavar="DIR", help=f"where articles go (default: {DEFAULT_FETCH_DIR})")
    where.add_argument("--format", choices=("txt", "jsonl"), default="txt", help="a .txt per article, or lines of articles.jsonl (default: txt)")
    how = get.add_argument_group("what to keep")
    how.add_argument("-n", "--limit", type=at_least(1), metavar="N", help="stop once N articles are saved")
    how.add_argument("--intro", action="store_true", help="only each article's lead section (20 to a request)")
    how.add_argument("--min-chars", type=at_least(0), default=0, metavar="N", help="skip articles shorter than N characters")
    how.add_argument("--keep-sections", action="store_true", help="keep See also, References and the like")
    how.add_argument("--overwrite", action="store_true", help="fetch articles already saved again")
    return parser


def _fetcher(args: argparse.Namespace, opener: Any, sleep: Callable[[float], None], delay: float) -> Fetcher:
    return Fetcher(
        user_agent(args.contact),
        delay=delay if args.delay is None else args.delay,
        retries=args.retries,
        timeout=args.timeout,
        opener=opener,
        sleep=sleep,
        log=printer(sys.stderr),
    )


def run_dump(args: argparse.Namespace, parser: argparse.ArgumentParser, opener: Any, sleep: Callable[[float], None]) -> int:
    if args.date != "latest" and not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD or 'latest'")
    note = printer(sys.stderr, quiet=args.quiet)
    fetcher = _fetcher(args, opener, sleep, delay=1.0)
    wiki = args.wiki or wiki_name(args.lang)
    base = (args.mirror or DUMPS).rstrip("/")
    date, files = find_dump(fetcher, wiki, args.kind, date=args.date, parts=args.parts, base=base, log=note)
    total = sum(f.size or 0 for f in files)
    print(f"{wiki} {date}, {args.kind}: {len(files)} file{'' if len(files) == 1 else 's'}, {human_bytes(total)}")
    if args.list:
        for f in files:
            print(f"  {f.name}  {human_bytes(f.size) if f.size is not None else 'size unknown'}")
        return 0
    os.makedirs(args.output, exist_ok=True)
    needed = 0
    for f in files:
        path = os.path.join(args.output, f.name)
        if f.size is not None and not (os.path.exists(path) and os.path.getsize(path) == f.size):
            part = path + ".part"
            needed += f.size - (os.path.getsize(part) if os.path.exists(part) else 0)
    free = shutil.disk_usage(args.output).free
    if needed > free and not args.force:
        raise DumpError(f"this needs {human_bytes(needed)} more and {args.output} has {human_bytes(free)} free (--force to try anyway)")
    saved = []
    for f in files:
        path = os.path.join(args.output, f.name)
        progress = None if args.quiet else Progress(f.name, f.size)
        status = download_file(
            fetcher,
            f"{base}/{wiki}/{date}/{f.name}",
            path,
            size=f.size,
            sha1=f.sha1,
            md5=f.md5,
            verify=not args.no_verify,
            progress=progress,
        )
        if progress:
            progress.finish()
        checked = "sha1" if f.sha1 else "md5" if f.md5 else None
        detail = f" ({checked} verified)" if status == "saved" and checked and not args.no_verify else ""
        print(f"{status:<7} {f.name}{detail}")
        saved.append(path)
    print(f"{len(files)} file{'' if len(files) == 1 else 's'} in {args.output}")
    xml = [p for p in saved if re.search(r"\.xml[^/]*\.(?:bz2|gz)$", p) and "-index" not in p]
    if xml:
        script = os.path.abspath(__file__)
        try:
            relative = os.path.relpath(script)
            script = script if relative.startswith("..") else relative
        except ValueError:  # another drive, on Windows
            pass
        note(f"next, for plain text: python3 {script} extract {' '.join(xml)}")
    return 0


def run_extract(args: argparse.Namespace, parser: argparse.ArgumentParser, opener: Any, sleep: Callable[[float], None]) -> int:
    try:
        options = ExtractOptions(
            namespaces=frozenset(args.namespace or [0]),
            titles=text_queries(args.title, args.regex),
            not_titles=text_queries(args.exclude_title, args.regex),
            categories=text_queries(args.category, args.regex),
            not_categories=text_queries(args.exclude_category, args.regex),
            min_chars=args.min_chars,
            keep_sections=args.keep_sections,
            keep_disambiguation=args.keep_disambiguation,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for path in args.dumps:
        if not os.path.isfile(path):
            raise ExtractError(f"no such dump: {path}")
    prepare_output(args.output, args.overwrite)
    sink: JsonlShards | TextFiles = (
        JsonlShards(args.output, args.shard_mb * 1_000_000, args.gzip) if args.format == "jsonl" else TextFiles(args.output)
    )
    counts: collections.Counter = collections.Counter()
    started = utc_now()
    site = SiteInfo()
    finished = False
    progress = None if args.quiet else _extract_progress(sys.stderr)
    try:
        site = extract(args.dumps, sink, options, counts, processes=args.processes, limit=args.limit, on_progress=progress)
        finished = True
    finally:
        sink.close()
        if progress and sys.stderr.isatty():
            sys.stderr.write("\n")
        record = {
            "tool": f"wikipedia_download.py {__version__}",
            "dumps": [{"file": os.path.basename(p), "bytes": os.path.getsize(p)} for p in args.dumps],
            "site": {"name": site.name, "dbname": site.dbname, "base": site.base},
            "format": args.format + (".gz" if args.gzip and args.format == "jsonl" else ""),
            "options": {
                "namespaces": sorted(options.namespaces),
                "titles": args.title or [],
                "exclude_titles": args.exclude_title or [],
                "categories": args.category or [],
                "exclude_categories": args.exclude_category or [],
                "regex": args.regex,
                "min_chars": args.min_chars,
                "keep_sections": args.keep_sections,
                "keep_disambiguation": args.keep_disambiguation,
                "limit": args.limit,
            },
            "counts": dict(counts),
            "files": sink.files,
            "started": started,
            "finished": utc_now() if finished else None,
        }
        write_atomic(os.path.join(args.output, EXTRACT_MANIFEST), (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    skipped = ", ".join(f"{why} {n:,}" for why, n in counts.most_common() if why not in ("articles", "pages read"))
    print(f"{counts['articles']:,} articles from {counts['pages read']:,} pages - in {args.output}" + (f" (skipped: {skipped})" if skipped else ""))
    return 0


def run_fetch(args: argparse.Namespace, parser: argparse.ArgumentParser, opener: Any, sleep: Callable[[float], None]) -> int:
    titles = list(args.title or [])
    if args.titles_file:
        with open(args.titles_file, encoding="utf-8") as fh:
            titles += [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    if not (titles or args.category or args.search or args.random):
        parser.error("say which articles: --title, --titles-file, --category, --search or --random")
    note = printer(sys.stderr, quiet=args.quiet)
    warn = printer(sys.stderr)
    if not args.contact:
        note("note: Wikimedia asks automated clients for contact details; pass --contact or set WIKIMEDIA_CONTACT")
    endpoint = args.api or f"https://{args.site or args.lang.strip().lower() + '.wikipedia.org'}/w/api.php"
    api = WikiApi(_fetcher(args, opener, sleep, delay=0.5), endpoint, log=warn)
    saver = ArticleSaver(args.output, args.format, keep_sections=args.keep_sections, min_chars=args.min_chars)
    log = printer(sys.stdout, quiet=args.quiet)
    counts: collections.Counter = collections.Counter()
    pending: list[dict] = []
    goal = f"/{args.limit}" if args.limit else ""

    def done() -> int:
        return counts["saved"] + counts["exists"]

    def flush() -> None:
        if not pending:
            return
        got = fetch_extracts(api, pending, args.intro)
        counts["missing"] += len(pending) - len(got)
        for page in got:
            status, detail = saver.save(page)
            counts[status] += 1
            log(f"[{done()}{goal}] {status:<6} {page['pageid']:>9}  {clip(page['title'], 60)} ({detail})")
        pending.clear()

    def sources() -> Iterator[dict]:
        if titles:
            yield from title_pages(api, titles, warn)
        for category in args.category or ():
            yield from category_pages(api, category, args.depth)
        for query in args.search or ():
            yield from search_pages(api, query)
        if args.random:
            yield from random_pages(api, args.random)

    seen: set[int] = set()
    try:
        for page in sources():
            if page["pageid"] in seen:
                continue
            seen.add(page["pageid"])
            if args.limit and done() + len(pending) >= args.limit:
                flush()
                if done() >= args.limit:
                    break
            if not args.overwrite and saver.have(page):
                counts["exists"] += 1
                log(f"[{done()}{goal}] exists {page['pageid']:>9}  {clip(page['title'], 60)}")
                continue
            pending.append(page)
            if len(pending) >= (20 if args.intro else 1):
                flush()
        flush()
    finally:
        saver.manifest.compact()
        parts = [f"{counts['saved']} saved"] + [
            f"{counts[k]} {label}" for k, label in (("exists", "already there"), ("short", "too short"), ("missing", "gone")) if counts[k]
        ]
        print(", ".join(parts) + f" - in {args.output}")
    return 0


COMMANDS = {"dump": run_dump, "extract": run_extract, "fetch": run_fetch}


def main(argv: Sequence[str] | None = None, *, opener: Any = None, sleep: Callable[[float], None] = time.sleep) -> int:
    """Run the command line; returns the exit status. ``opener`` and ``sleep`` are for tests."""
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args, parser, opener, sleep)
    except KeyboardInterrupt:
        print("\ninterrupted - run the same command again to carry on", file=sys.stderr)
        return 130
    except BrokenPipeError:
        raise
    except (DumpError, ExtractError, ApiError, FetchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
