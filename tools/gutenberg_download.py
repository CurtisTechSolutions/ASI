#!/usr/bin/env python3
"""Download plain-text books from Project Gutenberg, chosen by filters.

    python3 tools/gutenberg_download.py                          the top 100 books in English
    python3 tools/gutenberg_download.py --facet bookshelf -l fr
    python3 tools/gutenberg_download.py -b "science fiction" --list
    python3 tools/gutenberg_download.py -b "science fiction" -n 20
    python3 tools/gutenberg_download.py -a "jane austen" --strip -o austen

With no options it downloads the 100 most downloaded books in English. The
defaults are ``--language en``, ``--sort popular`` and ``--limit 100``, and any
of them can be changed - ``-l any``, ``--sort id``, ``-n 0`` for no limit.

Choosing books asks little of the site: the filters run over Project
Gutenberg's own catalog (``pg_catalog.csv``), downloaded once and cached for a
week, and the popularity order comes from its Top 100 page, fetched at most
once a day. Only the books themselves are fetched - one at a time, ``--delay``
seconds apart, which is what the site asks of automated clients.

Every book is saved as UTF-8 with ``\\n`` line endings, whatever it was
published in; ``--strip`` also cuts off Project Gutenberg's header and licence.
``manifest.jsonl`` in the output directory records each book's catalog entry,
the URL it came from and the SHA-256 of the file written.

Standard library only; Python 3.10 or newer.
"""

from __future__ import annotations

import argparse
import calendar
import codecs
import collections
import csv
import dataclasses
import datetime
import gzip
import hashlib
import io
import os
import random
import re
import shutil
import string
import sys
import time
import urllib.parse
from typing import Any, Callable, Iterable, Iterator, Sequence, TextIO

from common import (
    Fatal,
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
    is_url,
    printer,
    slugify,
    text_queries,
    utc_now,
    write_atomic,
)

__version__ = "1.0.0"

SITE = "https://www.gutenberg.org"
CATALOG_URLS = (
    f"{SITE}/cache/epub/feeds/pg_catalog.csv.gz",
    f"{SITE}/cache/epub/feeds/pg_catalog.csv",
)
"""The official CSV catalog - compressed first, the plain file if that is missing."""

CATALOG_MAX_AGE = 7 * 24 * 3600
"""Seconds a cached catalog is used before a fresh one is fetched."""

TOP_URL = f"{SITE}/browse/scores/top"
"""Project Gutenberg's Top 100: the most downloaded books yesterday, last week and last month."""
TOP_LISTS = ("books-last30", "books-last7", "books-last1")
"""The page's book lists, in the order they rank: the month leads, the shorter spans fill in."""
TOP_MAX_AGE = 24 * 3600
"""The lists change daily; a cached copy is used for a day."""

USER_AGENT = f"gutenberg_download/{__version__} (Python urllib)"
DEFAULT_LANGUAGE = "en"
DEFAULT_LIMIT = 100
DEFAULT_SORT = "popular"
DEFAULT_OUTPUT = "gutenberg_books"
DEFAULT_NAME = "{id}-{title}"
DEFAULT_DELAY = 2.0
"""Seconds between requests: the pause in Project Gutenberg's own example for robots (``wget -w 2``)."""
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 60.0
MANIFEST = "manifest.jsonl"
FAILURES_BEFORE_STOPPING = 3
"""Books in a row that may fail before a run stops knocking."""

SORTS = ("popular", "id", "title", "author", "released", "random")
FACETS = ("language", "author", "subject", "bookshelf", "locc", "type")
NAME_FIELDS = ("id", "title", "author", "language")

# ---------------------------------------------------------------------------
# the catalog
# ---------------------------------------------------------------------------


class CatalogError(Exception):
    """The catalog could not be fetched or read, or is not the file it should be."""


_ROLE = re.compile(r"\s*\[([^\[\]]+)\]\s*$")
_YEAR = r"(?:(?:approximately|approx\.|circa|ca\.)\s*)?(\d{1,4})\s*\??\s*(BCE|BC)?"
_LIFESPAN = re.compile(rf"^\s*(?:{_YEAR})?\s*-\s*(?:{_YEAR})?\s*$", re.IGNORECASE)


def _year(digits: str | None, era: str | None) -> int | None:
    if not digits:
        return None
    return -int(digits) if era else int(digits)


@dataclasses.dataclass(frozen=True, slots=True)
class Creator:
    """One name from the catalog's Authors column, such as "Austen, Jane, 1775-1817".

    ``role`` is None for an author and the bracketed role otherwise
    ("Translator", "Editor", "Illustrator", ...). Years before the common era
    are negative; a year the catalog does not give is None.
    """

    raw: str
    name: str
    birth: int | None = None
    death: int | None = None
    role: str | None = None

    @classmethod
    def parse(cls, raw: str) -> Creator:
        text = raw.strip()
        role = None
        bracket = _ROLE.search(text)
        if bracket:
            role, text = bracket.group(1).strip(), text[: bracket.start()].rstrip()
        head, comma, tail = text.rpartition(",")
        span = _LIFESPAN.match(tail) if comma else None
        if span is None or not (span.group(1) or span.group(3)):
            # No dates, or dates in a form this does not read ("active 12th
            # century"): the whole thing is the name, and no year filter can
            # count this person as alive at any time.
            return cls(raw.strip(), text, role=role)
        return cls(raw.strip(), head.strip(), _year(*span.group(1, 2)), _year(*span.group(3, 4)), role)

    def alive_within(self, start: int | None, end: int | None) -> bool:
        """Whether this person lived at some point from ``start`` to ``end`` (either may be open).

        A missing birth or death year is taken to be the one that is known;
        someone with neither is never counted as alive.
        """
        first = self.birth if self.birth is not None else self.death
        last = self.death if self.death is not None else self.birth
        if first is None or last is None:
            return False
        return (start is None or last >= start) and (end is None or first <= end)


@dataclasses.dataclass(frozen=True, slots=True)
class Book:
    """One catalog entry."""

    id: int
    title: str
    type: str = "Text"
    released: str = ""
    languages: tuple[str, ...] = ()
    creators: tuple[Creator, ...] = ()
    subjects: tuple[str, ...] = ()
    locc: tuple[str, ...] = ()
    bookshelves: tuple[str, ...] = ()

    @property
    def authors(self) -> tuple[Creator, ...]:
        """The authors proper; everyone credited when the book names no author."""
        primary = tuple(c for c in self.creators if c.role is None)
        return primary or self.creators

    @property
    def author(self) -> str:
        """Who to credit in one line: the first author, and "et al." if there are more."""
        people = self.authors
        if not people:
            return ""
        return people[0].name + (" et al." if len(people) > 1 else "")

    def metadata(self) -> dict[str, Any]:
        """The catalog entry as JSON-ready data, for the manifest."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": [c.raw for c in self.creators],
            "languages": list(self.languages),
            "subjects": list(self.subjects),
            "bookshelves": list(self.bookshelves),
            "locc": list(self.locc),
            "released": self.released,
            "type": self.type,
        }


_COLUMNS = {
    "id": ("text#", "text #", "etext#", "id"),
    "type": ("type",),
    "released": ("issued", "release date", "released"),
    "title": ("title",),
    "languages": ("language", "languages"),
    "creators": ("authors", "author"),
    "subjects": ("subjects", "subject"),
    "locc": ("locc",),
    "bookshelves": ("bookshelves", "bookshelf"),
}
"""Where each field comes from, by (folded) column name - the order of columns does not matter."""


def _split(value: str, separators: str = ";") -> tuple[str, ...]:
    """The non-empty, stripped items of a delimited catalog cell."""
    return tuple(item.strip() for item in re.split(f"[{separators}]", value) if item.strip())


def _open_text(path: str) -> TextIO:
    """``path`` opened as UTF-8 text for the csv module, gunzipped if it is gzip."""
    with open(path, "rb") as fh:
        gzipped = fh.read(2) == b"\x1f\x8b"
    if gzipped:
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return open(path, encoding="utf-8-sig", newline="")


def _cell(row: list[str], column: dict[str, int], field: str) -> str:
    index = column.get(field)
    return row[index].strip() if index is not None and index < len(row) else ""


def read_catalog(path: str) -> Iterator[Book]:
    """Every entry of a ``pg_catalog.csv`` (plain or gzipped), as a :class:`Book`.

    Columns are found by name, and a row whose number is not a number is
    skipped rather than trusted.
    """
    csv.field_size_limit(max(csv.field_size_limit(), 1 << 24))
    try:
        with _open_text(path) as fh:
            rows = csv.reader(fh)
            header = [fold(name).strip() for name in next(rows, [])]
            column = {}
            for field, names in _COLUMNS.items():
                found = next((header.index(name) for name in names if name in header), None)
                if found is not None:
                    column[field] = found
            if "id" not in column or "title" not in column:
                raise CatalogError(
                    f"{path} is not a Project Gutenberg catalog (its columns are: {', '.join(header) or 'none'})"
                )
            for row in rows:
                try:
                    number = int(_cell(row, column, "id"))
                except ValueError:
                    continue
                yield Book(
                    id=number,
                    title=" ".join(_cell(row, column, "title").split()),
                    type=_cell(row, column, "type") or "Text",
                    released=_cell(row, column, "released"),
                    languages=tuple(code.lower() for code in _split(_cell(row, column, "languages"), ";,")),
                    creators=tuple(Creator.parse(name) for name in _split(_cell(row, column, "creators"))),
                    subjects=_split(_cell(row, column, "subjects")),
                    locc=tuple(code.upper() for code in _split(_cell(row, column, "locc"), ";,")),
                    bookshelves=_split(_cell(row, column, "bookshelves")),
                )
    except (OSError, EOFError, UnicodeDecodeError, csv.Error) as exc:
        raise CatalogError(f"cannot read the catalog {path}: {exc}") from None


def default_cache_dir() -> str:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "gutenberg_download")


def _looks_like_catalog(data: bytes) -> bool:
    """Whether ``data`` starts like the CSV catalog - not, say, an HTML error page."""
    head = data[:4096]
    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                head = gz.read(4096)
        except (OSError, EOFError):
            return False
    first_line = head.split(b"\n", 1)[0].decode("utf-8", "replace").lower()
    return "title" in first_line and "," in first_line


def ensure_catalog(
    fetcher: Fetcher,
    cache_dir: str,
    urls: Sequence[str] = CATALOG_URLS,
    *,
    refresh: bool = False,
    max_age: float = CATALOG_MAX_AGE,
    log: Callable[[str], None] = lambda message: None,
    warn: Callable[[str], None] = lambda message: None,
    now: Callable[[], float] = time.time,
) -> str:
    """The path of a usable catalog, downloading one when the cached copy is missing or stale.

    ``urls`` are tried in order, moving on only when one does not exist. When
    the download fails and a stale copy is cached, the stale copy is used - a
    week-old catalog is a better answer than none.
    """
    paths = [os.path.join(cache_dir, os.path.basename(urllib.parse.urlsplit(u).path) or "pg_catalog.csv") for u in urls]
    cached = [p for p in paths if os.path.isfile(p)]
    newest = max(cached, key=os.path.getmtime, default=None)
    if newest and not refresh and now() - os.path.getmtime(newest) < max_age:
        return newest
    problem: Exception | None = None
    for url, path in zip(urls, paths):
        log(f"downloading the catalog: {url}")
        try:
            data = fetcher.get(url)
        except NotFound as exc:
            problem = exc
            continue
        except FetchError as exc:
            problem = exc
            break
        if not _looks_like_catalog(data):
            problem = CatalogError(f"{url} did not return a CSV catalog")
            break
        os.makedirs(cache_dir, exist_ok=True)
        write_atomic(path, data)
        return path
    if newest:
        day = datetime.date.fromtimestamp(os.path.getmtime(newest)).isoformat()
        warn(f"warning: could not download a fresh catalog ({problem}); using the cached one from {day}")
        return newest
    raise CatalogError(f"could not download the catalog: {problem}")


_BOOK_LINK = re.compile(r'href="(?:https?://www\.gutenberg\.org)?/ebooks/(\d+)"')


def parse_top(page: str) -> list[int]:
    """The book numbers on Project Gutenberg's Top 100 page, most downloaded first.

    The page ranks the books most downloaded over the last 30 days, the last 7
    and yesterday. The 30-day list leads, being the steadiest, and books only on
    the shorter lists follow it - so that a filter such as English, which drops
    a few, still leaves a hundred to choose from. If the lists cannot be found,
    the page's book links are taken in the order they appear.
    """
    ranking: list[int] = []
    for anchor in TOP_LISTS:
        heading = re.search(rf'(?:id|name)="{anchor}"', page)
        listing = re.search(r"<ol\b.*?</ol>", page[heading.end() :], re.DOTALL | re.IGNORECASE) if heading else None
        if listing:
            ranking += [int(number) for number in _BOOK_LINK.findall(listing.group())]
    if not ranking:
        ranking = [int(number) for number in _BOOK_LINK.findall(page)]
    return list(dict.fromkeys(ranking))


def _read_ranking(path: str) -> list[int]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_top(fh.read())


def ensure_ranking(
    fetcher: Fetcher,
    cache_dir: str,
    *,
    refresh: bool = False,
    max_age: float = TOP_MAX_AGE,
    log: Callable[[str], None] = lambda message: None,
    warn: Callable[[str], None] = lambda message: None,
    now: Callable[[], float] = time.time,
) -> list[int]:
    """Project Gutenberg's popularity ranking, from its Top 100 page, cached for a day.

    The catalog has no download counts, so this page is the only place the
    order comes from. A failed download falls back to the cached page with a
    warning; with nothing cached it is an error, since "the most downloaded"
    cannot be guessed.
    """
    path = os.path.join(cache_dir, "top100.html")
    cached = os.path.isfile(path)
    if cached and not refresh and now() - os.path.getmtime(path) < max_age:
        return _read_ranking(path)
    log(f"downloading the Top 100: {TOP_URL}")
    problem: object
    try:
        data = fetcher.get(TOP_URL)
    except FetchError as exc:
        problem = exc
    else:
        ranking = parse_top(data.decode("utf-8", "replace"))
        if ranking:
            os.makedirs(cache_dir, exist_ok=True)
            write_atomic(path, data)
            return ranking
        problem = f"{TOP_URL} lists no books"
    if cached:
        day = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
        warn(f"warning: could not download the Top 100 ({problem}); using the one from {day}")
        return _read_ranking(path)
    raise CatalogError(f"could not download Project Gutenberg's Top 100 ({problem}); --sort id works without it")


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


IdRange = tuple[int | None, int | None]


@dataclasses.dataclass
class Filters:
    """What a book must be to be chosen. Empty fields accept everything.

    The values of one filter are alternatives - a book in any of the languages
    passes the language filter - and a book must pass every filter. Exclusions
    drop a book when any of their values match it.
    """

    ids: tuple[IdRange, ...] = ()
    types: frozenset[str] = frozenset({"text"})
    languages: frozenset[str] = frozenset()
    locc: tuple[str, ...] = ()
    released_from: str | None = None
    released_to: str | None = None
    alive_from: int | None = None
    alive_to: int | None = None
    authors: tuple[TextQuery, ...] = ()
    titles: tuple[TextQuery, ...] = ()
    subjects: tuple[TextQuery, ...] = ()
    bookshelves: tuple[TextQuery, ...] = ()
    not_authors: tuple[TextQuery, ...] = ()
    not_titles: tuple[TextQuery, ...] = ()
    not_subjects: tuple[TextQuery, ...] = ()
    not_bookshelves: tuple[TextQuery, ...] = ()

    @property
    def narrows(self) -> bool:
        """Whether anything was asked for beyond the default type."""
        return bool(
            self.ids
            or self.languages
            or self.locc
            or self.released_from
            or self.released_to
            or self.alive_from is not None
            or self.alive_to is not None
            or self.authors
            or self.titles
            or self.subjects
            or self.bookshelves
        )

    def __call__(self, book: Book) -> bool:
        if self.ids and not any((lo is None or lo <= book.id) and (hi is None or book.id <= hi) for lo, hi in self.ids):
            return False
        if self.types and book.type.casefold() not in self.types:
            return False
        if self.languages and self.languages.isdisjoint(book.languages):
            return False
        if self.locc and not any(code.startswith(prefix) for code in book.locc for prefix in self.locc):
            return False
        if self.released_from or self.released_to:
            day = book.released[:10]
            if not day:
                return False
            if (self.released_from and day < self.released_from) or (self.released_to and day > self.released_to):
                return False
        if self.alive_from is not None or self.alive_to is not None:
            if not any(c.alive_within(self.alive_from, self.alive_to) for c in book.authors):
                return False
        people = [c.raw for c in book.creators]
        if self.authors and not any_match(self.authors, people):
            return False
        if self.titles and not any_match(self.titles, [book.title]):
            return False
        if self.subjects and not any_match(self.subjects, book.subjects):
            return False
        if self.bookshelves and not any_match(self.bookshelves, book.bookshelves):
            return False
        return not (
            any_match(self.not_authors, people)
            or any_match(self.not_titles, [book.title])
            or any_match(self.not_subjects, book.subjects)
            or any_match(self.not_bookshelves, book.bookshelves)
        )


_ID_RANGE = re.compile(r"([0-9]*)-([0-9]*)")


def parse_ids(spec: str) -> list[IdRange]:
    """Book numbers and ranges - "84,1342", "1-100", "70000-" - as inclusive (low, high) pairs."""
    ranges: list[IdRange] = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        if re.fullmatch(r"[0-9]+", part):
            ranges.append((int(part), int(part)))
            continue
        span = _ID_RANGE.fullmatch(part)
        if not span or not any(span.groups()):
            raise ValueError(f"--ids: {part!r} is not a book number or a range like 1-100")
        low, high = (int(g) if g else None for g in span.groups())
        if low is not None and high is not None and low > high:
            raise ValueError(f"--ids: {part!r} runs backwards")
        ranges.append((low, high))
    if not ranges:
        raise ValueError("--ids: no book numbers given")
    return ranges


_DATE = re.compile(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?")


def date_bound(text: str, *, end: bool) -> str:
    """A partial date as the first day it covers (or, with ``end``, the last), as YYYY-MM-DD."""
    found = _DATE.fullmatch(text.strip())
    if not found:
        raise ValueError(f"{text!r} is not a date like 2020, 2020-06 or 2020-06-30")
    year, month, day = (int(g) if g else None for g in found.groups())
    try:
        if day is not None:
            return datetime.date(year, month, day).isoformat()
        if month is not None:
            last = calendar.monthrange(year, month)[1] if end else 1
            return datetime.date(year, month, last).isoformat()
        return f"{year:04d}-12-31" if end else f"{year:04d}-01-01"
    except (ValueError, calendar.IllegalMonthError):
        raise ValueError(f"{text!r} is not a real date") from None


def filters_from_args(args: argparse.Namespace) -> Filters:
    """The :class:`Filters` a parsed command line asks for; ValueError for a bad value."""

    def queries(values: Sequence[str] | None) -> tuple[TextQuery, ...]:
        return text_queries(values, args.regex)

    types = {fold(t) for spec in (args.type or ["Text"]) for t in _split(spec, ",")}
    released_from = date_bound(args.released_from, end=False) if args.released_from else None
    released_to = date_bound(args.released_to, end=True) if args.released_to else None
    if released_from and released_to and released_from > released_to:
        raise ValueError("--released-from is after --released-to")
    if args.author_alive_from is not None and args.author_alive_to is not None:
        if args.author_alive_from > args.author_alive_to:
            raise ValueError("--author-alive-from is after --author-alive-to")
    languages = {code.lower() for spec in args.language or () for code in _split(spec, ",;")}
    return Filters(
        ids=tuple(r for spec in args.ids or () for r in parse_ids(spec)),
        types=frozenset() if "any" in types else frozenset(types),
        languages=frozenset() if "any" in languages else frozenset(languages),
        locc=tuple(code.upper() for spec in args.locc or () for code in _split(spec, ",;")),
        released_from=released_from,
        released_to=released_to,
        alive_from=args.author_alive_from,
        alive_to=args.author_alive_to,
        authors=queries(args.author),
        titles=queries(args.title),
        subjects=queries(args.subject),
        bookshelves=queries(args.bookshelf),
        not_authors=queries(args.exclude_author),
        not_titles=queries(args.exclude_title),
        not_subjects=queries(args.exclude_subject),
        not_bookshelves=queries(args.exclude_bookshelf),
    )


def apply_defaults(args: argparse.Namespace, filters: Filters) -> Filters:
    """Fill in what was not asked for: books in English, a hundred of them.

    ``--limit 0`` means no limit. Naming books with ``--ids`` chooses them
    exactly, so neither default applies there.
    """
    exact = bool(filters.ids)
    if args.limit is None:
        args.limit = None if exact else DEFAULT_LIMIT
    elif args.limit == 0:
        args.limit = None
    if args.language is None and not exact:
        return dataclasses.replace(filters, languages=frozenset({DEFAULT_LANGUAGE}))
    return filters


def sort_books(
    books: Iterable[Book],
    key: str = "id",
    *,
    reverse: bool = False,
    seed: int | None = None,
    ranking: Sequence[int] = (),
) -> list[Book]:
    """``books`` in the order asked for. Books without the sort field go last, in number order.

    ``popular`` follows ``ranking``, the book numbers of :func:`ensure_ranking`.
    """
    ordered = sorted(books, key=lambda b: b.id)
    if key == "popular":
        place = {number: index for index, number in enumerate(ranking)}
        ranked = sorted((b for b in ordered if b.id in place), key=lambda b: place[b.id], reverse=reverse)
        return ranked + [b for b in ordered if b.id not in place]
    if key == "random":
        random.Random(seed).shuffle(ordered)
    elif key != "id":
        field: Callable[[Book], str] = {
            "title": lambda b: fold(b.title),
            "author": lambda b: fold(b.authors[0].name) if b.authors else "",
            "released": lambda b: b.released,
        }[key]
        known = sorted((b for b in ordered if field(b)), key=field, reverse=reverse)
        return known + [b for b in ordered if not field(b)]
    return ordered[::-1] if reverse else ordered


def facet_counts(books: Iterable[Book], field: str) -> list[tuple[str, int]]:
    """How many of ``books`` carry each value of ``field``, most common first."""
    pick: Callable[[Book], Iterable[str]] = {
        "language": lambda b: b.languages,
        "author": lambda b: (c.raw for c in b.authors),
        "subject": lambda b: b.subjects,
        "bookshelf": lambda b: b.bookshelves,
        "locc": lambda b: b.locc,
        "type": lambda b: (b.type,),
    }[field]
    counts = collections.Counter(value for book in books for value in set(pick(book)))
    return sorted(counts.items(), key=lambda item: (-item[1], fold(item[0])))


# ---------------------------------------------------------------------------
# the text of a book
# ---------------------------------------------------------------------------


def text_urls(book_id: int, mirror: str | None = None) -> list[str]:
    """Where a book's plain text may be, best first.

    The generated UTF-8 file comes first - it exists for nearly every book -
    then the files volunteers uploaded: UTF-8, then ASCII, then Latin-1. A
    mirror keeps the uploaded files in the collection's numbered folders
    (book 12345 in ``1/2/3/4/12345/``).
    """
    if mirror:
        root = mirror.rstrip("/")
        digits = str(book_id).zfill(2)
        stem = f"{root}/{'/'.join(digits[:-1])}/{book_id}/{book_id}"
    else:
        root = SITE
        stem = f"{SITE}/files/{book_id}/{book_id}"
    return [f"{root}/cache/epub/{book_id}/pg{book_id}.txt", f"{stem}-0.txt", f"{stem}.txt", f"{stem}-8.txt"]


def fetch_text(fetcher: Fetcher, book_id: int, mirror: str | None = None) -> tuple[bytes, str]:
    """The first plain-text file of the book that exists, and its URL. NotFound if none does."""
    for url in text_urls(book_id, mirror):
        try:
            data = fetcher.get(url)
        except NotFound:
            continue
        if data:
            return data, url
    raise NotFound(f"no plain-text file for book {book_id}")


_DECLARED = re.compile(rb"character set encoding:[ \t]*([A-Za-z0-9_.:-]+)", re.IGNORECASE)


def _declared_encoding(data: bytes) -> str | None:
    found = _DECLARED.search(data[:20000])
    if not found:
        return None
    try:
        return codecs.lookup(found.group(1).decode("ascii")).name
    except LookupError:
        return None


def decode_text(data: bytes) -> tuple[str, str]:
    """``data`` as text, and the encoding it turned out to be in.

    UTF-8 is tried first because valid UTF-8 practically never happens by
    accident; then whatever the header declares ("Character set encoding:
    ..."), then Windows-1252, then Latin-1, which accepts anything. A file that
    declares UTF-8 but has a stray bad byte keeps the rest of its text.
    """
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8) :].decode("utf-8", "replace"), "utf-8"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", "replace"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    declared = _declared_encoding(data)
    if declared == "utf-8":
        return data.decode("utf-8", "replace"), "utf-8"
    for encoding in (declared, "cp1252"):
        if encoding:
            try:
                return data.decode(encoding), encoding
            except (UnicodeDecodeError, LookupError):
                pass
    return data.decode("latin-1"), "latin-1"


_START = (
    re.compile(
        r"^[ \t]*\*+[ \t]*START OF (?:THE |THIS )?(?:COPYRIGHTED )?PROJECT GUTENBERG[^\n]*",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^[ \t]*\*END\*?[ \t]*THE SMALL PRINT[^\n]*", re.IGNORECASE | re.MULTILINE),
)
"""Where the header ends, most reliable first: today's marker, then the 1990s "small print"."""

_END = re.compile(
    r"^[ \t]*(?:\*+[ \t]*END OF (?:THE |THIS )?(?:COPYRIGHTED )?PROJECT GUTENBERG"
    r"|END OF (?:THE |THIS )?PROJECT GUTENBERG"
    r"|ENDE DIESES PROJE[CK]T GUTENBERG)",
    re.IGNORECASE | re.MULTILINE,
)
"""Where the licence begins: "*** END OF THE PROJECT GUTENBERG EBOOK", or the older
"End of the Project Gutenberg EBook of ..." line that precedes it."""

_CREDITS = re.compile(
    r"(?:produced by|e-?text prepared by|this e-?(?:text|book) was (?:prepared|produced) by|transcribed from)\b",
    re.IGNORECASE,
)


def _drop_credits(body: str) -> str:
    """``body`` without a leading "Produced by ..." paragraph of at most a few lines."""
    rest = body[re.match(r"(?:[ \t]*\n)*", body).end() :]
    gap = re.search(r"\n[ \t]*\n", rest)
    if gap and _CREDITS.match(rest.lstrip()) and rest[: gap.start()].count("\n") < 6:
        return rest[gap.end() :]
    return body


def strip_boilerplate(text: str) -> tuple[str, bool]:
    """Cut Project Gutenberg's header and licence off ``text``.

    Returns the book's own text and whether any marker was found; with none,
    ``text`` comes back untouched rather than guessed at. A short "Produced by"
    credit at the top of the book goes too.
    """
    start = 0
    for marker in _START:
        found = marker.search(text)
        if found:
            start = found.end()
            # A marker too long for one line ends on the next with "***".
            if not text[found.start() : start].rstrip().endswith("*"):
                following = text[start + 1 :].split("\n", 1)[0]
                if following.rstrip().endswith("***"):
                    start += 1 + len(following)
            break
    end = _END.search(text, start)
    if not start and end is None:
        return text, False
    body = _drop_credits(text[start : end.start() if end else len(text)])
    body = re.sub(r"\A(?:[ \t]*\n)+", "", body).rstrip()
    return (body + "\n" if body else ""), True


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------


def book_filename(book: Book, name_format: str = DEFAULT_NAME) -> str:
    """The path, relative to the output directory, that ``book`` is saved at."""
    author = book.authors[0].name if book.authors else ""
    return (
        name_format.format(
            id=book.id,
            title=slugify(book.title) or "untitled",
            author=slugify(author) or "anonymous",
            language=slugify("-".join(book.languages)) or "unknown",
        )
        + ".txt"
    )


def check_name_format(name_format: str) -> None:
    """ValueError unless ``name_format`` gives every book its own file inside the output directory."""
    try:
        fields = [name for _, name, _, _ in string.Formatter().parse(name_format) if name is not None]
    except ValueError as exc:
        raise ValueError(f"bad --name-format {name_format!r}: {exc}") from None
    unknown = sorted({name for name in fields if name not in NAME_FIELDS})
    if unknown:
        raise ValueError(
            f"--name-format: unknown field(s) {', '.join('{' + n + '}' for n in unknown)}; "
            f"use {', '.join('{' + n + '}' for n in NAME_FIELDS)}"
        )
    if "id" not in fields:
        raise ValueError("--name-format must include {id}, so that every book gets a file of its own")
    try:
        sample = book_filename(Book(id=1, title="t"), name_format)
    except (ValueError, IndexError, KeyError) as exc:
        raise ValueError(f"bad --name-format {name_format!r}: {exc}") from None
    parts = sample.replace("\\", "/").split("/")
    if os.path.isabs(sample) or sample.startswith(("/", "\\")) or ".." in parts:
        raise ValueError("--name-format must name a file inside the output directory")


@dataclasses.dataclass
class Result:
    """What happened to one book."""

    book: Book
    status: str  # "saved", "exists", "no text" or "failed"
    path: str = ""
    detail: str = ""


class Downloader:
    """Fetches books one at a time into ``output``, keeping ``manifest.jsonl`` up to date."""

    def __init__(
        self,
        fetcher: Fetcher,
        output: str,
        *,
        name_format: str = DEFAULT_NAME,
        strip: bool = False,
        overwrite: bool = False,
        mirror: str | None = None,
        ranking: Sequence[int] = (),
        log: Callable[[str], None] = print,
    ):
        self.fetcher = fetcher
        self.output = output
        self.name_format = name_format
        self.strip = strip
        self.overwrite = overwrite
        self.mirror = mirror
        self.ranks = {number: place for place, number in enumerate(ranking, 1)}
        self.log = log
        self.results: list[Result] = []

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.status == "failed"]

    def run(self, books: Sequence[Book], limit: int | None = None) -> list[Result]:
        """Download ``books`` in order until ``limit`` of them are on disk.

        Books already on disk count towards the limit, so running the same
        command twice fetches nothing new; books without a plain-text file, or
        that fail, do not, so the next book takes their place.
        """
        goal = min(limit or len(books), len(books))
        os.makedirs(self.output, exist_ok=True)
        manifest = Manifest(os.path.join(self.output, MANIFEST))
        done = failures = 0
        try:
            for book in books:
                if done >= goal:
                    break
                result = self._one(book, manifest)
                self.results.append(result)
                if result.status in ("saved", "exists"):
                    done += 1
                failures = failures + 1 if result.status == "failed" else 0
                self.log(_progress_line(done, goal, result))
                if failures >= FAILURES_BEFORE_STOPPING:
                    raise FetchError(f"{failures} books in a row failed, so stopping; the last: {result.detail}")
        finally:
            manifest.compact()
        return self.results

    def _one(self, book: Book, manifest: Manifest) -> Result:
        relative = book_filename(book, self.name_format)
        path = os.path.join(self.output, relative)
        if not self.overwrite and os.path.exists(path):
            return Result(book, "exists", path)
        try:
            data, url = fetch_text(self.fetcher, book.id, self.mirror)
        except NotFound:
            return Result(book, "no text")
        except Fatal:
            raise
        except FetchError as exc:
            return Result(book, "failed", detail=str(exc))
        text, encoding = decode_text(data)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        stripped = False
        if self.strip:
            text, stripped = strip_boilerplate(text)
        body = text.encode("utf-8")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_atomic(path, body)
        entry = {
            **book.metadata(),
            "file": relative.replace(os.sep, "/"),
            "url": url,
            "encoding": encoding,
            "stripped": stripped,
            "chars": len(text),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "downloaded": utc_now(),
        }
        if book.id in self.ranks:  # its place in the Top 100 lists on the day it was fetched
            entry["popularity_rank"] = self.ranks[book.id]
        manifest.add(entry)
        detail = human_bytes(len(body))
        if self.strip and not stripped:
            detail += ", no Gutenberg header found so kept whole"
        return Result(book, "saved", path, detail)

    def summary(self) -> str:
        counts = collections.Counter(r.status for r in self.results)
        parts = [f"{counts['saved']} saved"]
        for status, label in (("exists", "already there"), ("no text", "without a plain-text file"), ("failed", "failed")):
            if counts[status]:
                parts.append(f"{counts[status]} {label}")
        lines = [", ".join(parts) + f" - in {self.output}"]
        if self.failed:
            numbers = ", ".join(str(r.book.id) for r in self.failed)
            lines.append(f"failed: {numbers} (running the same command again retries them)")
        return "\n".join(lines)


def _progress_line(done: int, goal: int, result: Result) -> str:
    width = len(str(goal))
    detail = f" ({result.detail})" if result.detail else ""
    return f"[{done:>{width}}/{goal}] {result.status:<7} {result.book.id:>6}  {clip(result.book.title, 60)}{detail}"


def print_list(books: Sequence[Book], out: TextIO | None = None) -> None:
    """``books`` as a table: number, languages, author, title."""
    out = out or sys.stdout
    width = shutil.get_terminal_size((120, 24)).columns if out.isatty() else 0
    print(f"{'id':>6}  {'lang':<5}  {'author':<28}  title", file=out)
    for book in books:
        line = f"{book.id:>6}  {','.join(book.languages) or '-':<5}  {clip(book.author or '-', 28):<28}  {book.title}"
        print(clip(line, width) if width else line, file=out)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

EPILOG = """\
With no options: the 100 most downloaded books in English. The defaults are
--language en --sort popular --limit 100; change any of them (-l any for every
language, -n 0 for no limit). --ids names books exactly, so the language and
the limit are not applied to it.

Text filters ignore case and accents and match words: every word you give must
begin a word of the field, in any order - so --author "austen jane" finds
"Austen, Jane, 1775-1817" and --subject detect finds "Detective and mystery
stories". Repeat an option to accept any of several values; different options
must all match. --regex makes the text filters regular expressions instead.

examples:
  %(prog)s                                        the top 100 books in English
  %(prog)s --list                                 ...shown, not downloaded
  %(prog)s --facet bookshelf                      what is there to filter on?
  %(prog)s -b "science fiction" -n 20             the 20 most popular sci-fi books
  %(prog)s -a "jane austen" --strip -o austen     her books, without the licence
  %(prog)s -l de --author-alive-to 1900 --sort random --seed 7

"Popular" is the order of Project Gutenberg's Top 100 lists (the last 30 days,
then the last 7, then yesterday), fetched once a day; the books on none of them
follow in catalog order. The catalog is downloaded once and cached for a week
(--cache-dir). Books are fetched one at a time, --delay seconds apart, as
Project Gutenberg asks of robots; for big jobs use a mirror (--mirror; see
https://www.gutenberg.org/MIRRORS.ALL).
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gutenberg_download.py",
        description="Download plain-text books from Project Gutenberg, chosen by filters - by default the top 100 in English.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    pick = parser.add_argument_group("filters")
    pick.add_argument(
        "-l", "--language", action="append", metavar="CODE", help="language code (default: en): fr, de, ... commas allowed; 'any' for all"
    )
    pick.add_argument("-a", "--author", action="append", metavar="TEXT", help="an author, translator, editor or illustrator")
    pick.add_argument("-t", "--title", action="append", metavar="TEXT", help="words of the title")
    pick.add_argument("-s", "--subject", action="append", metavar="TEXT", help="a Library of Congress subject heading")
    pick.add_argument("-b", "--bookshelf", action="append", metavar="TEXT", help="a Project Gutenberg bookshelf")
    pick.add_argument("--locc", action="append", metavar="CLASS", help="Library of Congress class prefix: PR, PS, Q, ...")
    pick.add_argument("--ids", action="append", metavar="LIST", help="book numbers and ranges: 84,1342,1-100,70000-")
    pick.add_argument("--released-from", metavar="DATE", help="released on Project Gutenberg on or after DATE (YYYY[-MM[-DD]])")
    pick.add_argument("--released-to", metavar="DATE", help="released on or before DATE")
    pick.add_argument("--author-alive-from", type=int, metavar="YEAR", help="an author was alive in or after YEAR (negative: BCE)")
    pick.add_argument("--author-alive-to", type=int, metavar="YEAR", help="an author was alive in or before YEAR")
    pick.add_argument("--type", action="append", metavar="TYPE", help="catalog type (default: Text; 'any' for all)")
    pick.add_argument("--exclude-author", action="append", metavar="TEXT", help="drop books by anyone matching TEXT")
    pick.add_argument("--exclude-title", action="append", metavar="TEXT", help="drop books whose title matches TEXT")
    pick.add_argument("--exclude-subject", action="append", metavar="TEXT", help="drop books with a subject matching TEXT")
    pick.add_argument("--exclude-bookshelf", action="append", metavar="TEXT", help="drop books on a bookshelf matching TEXT")
    pick.add_argument("--regex", action="store_true", help="text filters are regular expressions, not words")

    order = parser.add_argument_group("choosing")
    order.add_argument(
        "--sort", choices=SORTS, default=DEFAULT_SORT, help="order to take books in (default: popular, most downloaded first)"
    )
    order.add_argument("--reverse", action="store_true", help="reverse that order")
    order.add_argument("--seed", type=int, help="seed for --sort random, to get the same books again")
    order.add_argument(
        "-n", "--limit", type=at_least(0), metavar="N", help=f"stop once N books are on disk, or list N (default: {DEFAULT_LIMIT}; 0: no limit)"
    )

    mode = parser.add_argument_group("instead of downloading").add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="show the matching books and download nothing")
    mode.add_argument("--facet", choices=FACETS, help="count the values of a field among the matching books")

    save = parser.add_argument_group("downloading")
    save.add_argument("-o", "--output", default=DEFAULT_OUTPUT, metavar="DIR", help=f"where books go (default: {DEFAULT_OUTPUT})")
    save.add_argument("--strip", action="store_true", help="cut off Project Gutenberg's header and licence")
    save.add_argument(
        "--name-format",
        default=DEFAULT_NAME,
        metavar="FORMAT",
        help="file name from {id} {title} {author} {language}; '/' makes folders (default: %(default)s)",
    )
    save.add_argument("--overwrite", action="store_true", help="download books that are already on disk again")
    save.add_argument("--mirror", metavar="URL", help="fetch books from this mirror instead of www.gutenberg.org")
    save.add_argument("--delay", type=at_least(0, float), default=DEFAULT_DELAY, metavar="SECONDS", help="pause between requests (default: %(default)s)")
    save.add_argument("--retries", type=at_least(0), default=DEFAULT_RETRIES, metavar="N", help="retries for a failed request (default: %(default)s)")
    save.add_argument("--timeout", type=at_least(1, float), default=DEFAULT_TIMEOUT, metavar="SECONDS", help="network timeout (default: %(default)s)")
    save.add_argument("-q", "--quiet", action="store_true", help="print only warnings, errors and the summary")

    catalog = parser.add_argument_group("catalog")
    catalog.add_argument("--catalog", metavar="FILE|URL", help="use this pg_catalog.csv[.gz] instead of the cached official one")
    catalog.add_argument(
        "--cache-dir", default=default_cache_dir(), metavar="DIR", help="where the catalog and the Top 100 are cached (default: %(default)s)"
    )
    catalog.add_argument("--refresh", action="store_true", help="download the catalog and the Top 100 again, however recent")
    return parser


def _catalog_path(args: argparse.Namespace, fetcher: Fetcher, note: Callable[[str], None]) -> str:
    if args.catalog and not is_url(args.catalog):
        return args.catalog
    urls = (args.catalog,) if args.catalog else CATALOG_URLS
    return ensure_catalog(fetcher, args.cache_dir, urls, refresh=args.refresh, log=note, warn=printer(sys.stderr))


def main(argv: Sequence[str] | None = None, *, opener: Any = None, sleep: Callable[[float], None] = time.sleep) -> int:
    """Run the command line; returns the exit status. ``opener`` and ``sleep`` are for tests."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        filters = filters_from_args(args)
        check_name_format(args.name_format)
        if args.mirror and not is_url(args.mirror):
            raise ValueError("--mirror must be an http:// or https:// URL")
    except ValueError as exc:
        parser.error(str(exc))
    if not (args.list or args.facet) and args.limit == 0 and not filters.narrows:
        parser.error(
            "-n 0 lifts the limit, and with no filter that is tens of thousands of books. Add a filter "
            "or a limit (--list shows what would match). To copy everything, mirror the collection: "
            "https://www.gutenberg.org/help/mirroring.html"
        )
    filters = apply_defaults(args, filters)

    note = printer(sys.stderr, quiet=args.quiet)
    fetcher = Fetcher(
        USER_AGENT,
        delay=args.delay,
        retries=args.retries,
        timeout=args.timeout,
        opener=opener,
        sleep=sleep,
        log=printer(sys.stderr),
    )
    downloader: Downloader | None = None
    code = 0
    try:
        total, matches = 0, []
        for book in read_catalog(_catalog_path(args, fetcher, note)):
            total += 1
            if filters(book):
                matches.append(book)
        found = f"{len(matches):,} of {total:,} catalog entries match"
        if args.facet:
            note(found)
            rows = facet_counts(matches, args.facet)
            for value, count in rows[: args.limit] if args.limit else rows:
                print(f"{count:>7}  {value}")
            return 0
        if not args.list:
            note(found)
            if not matches:
                return 0
        seed = args.seed
        if args.sort == "random" and seed is None:
            seed = random.randrange(1 << 32)
            note(f"random order, seed {seed} (pass --seed {seed} to get the same order again)")
        ranking: list[int] = []
        if args.sort == "popular" and matches:
            ranking = ensure_ranking(fetcher, args.cache_dir, refresh=args.refresh, log=note, warn=printer(sys.stderr))
            ranked = len(set(ranking).intersection(book.id for book in matches))
            if ranked < min(args.limit or len(matches), len(matches)):
                note(f"{ranked:,} of the {len(matches):,} matching books are on the Top 100 lists; the rest follow in catalog order")
        matches = sort_books(matches, args.sort, reverse=args.reverse, seed=seed, ranking=ranking)
        if args.list:
            shown = matches[: args.limit] if args.limit else matches
            print_list(shown)
            note(found + (f"; showing {len(shown):,}" if len(shown) < len(matches) else ""))
            return 0
        downloader = Downloader(
            fetcher,
            args.output,
            name_format=args.name_format,
            strip=args.strip,
            overwrite=args.overwrite,
            mirror=args.mirror,
            ranking=ranking,
            log=printer(sys.stdout, quiet=args.quiet),
        )
        downloader.run(matches, args.limit)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        code = 130
    except BrokenPipeError:
        raise
    except (CatalogError, FetchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 1
    if downloader is not None:
        print(downloader.summary())
        if code == 0 and downloader.failed:
            code = 1
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:  # e.g. `--list | head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
