"""Tests for gutenberg_download.py.

Run: python3 -m unittest discover -s tools/tests  (from the repository root)

Offline and deterministic: the catalog and every book the site would serve are
canned here, and nothing sleeps. One test class goes through a real HTTP server
on 127.0.0.1, so urllib's own headers, gzip and sockets are exercised too.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gutenberg_download as gd  # noqa: E402
from common import Fetcher, TextQuery  # noqa: E402
from fakes import FakeClock, FakeSite, LocalServer, http_error, no_proxy_opener  # noqa: E402

CATALOG = """\
Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves
10,Text,1989-08-01,The King James Version of the Bible,en,,Bible,BS,"Banned Books from Anne Haight's list; Browsing: Religion/Spirituality/Paranormal"
84,Text,1993-10-01,"Frankenstein; Or, The Modern Prometheus",en,"Shelley, Mary Wollstonecraft, 1797-1851","Science fiction; Horror tales; Gothic fiction; Monsters -- Fiction",PR,"Gothic Fiction; Precursors of Science Fiction; Browsing: Science-Fiction & Fantasy"
1260,Text,1998-03-01,Jane Eyre: An Autobiography,en,"Brontë, Charlotte, 1816-1855","Governesses -- Fiction; Love stories; Bildungsromans",PR,Browsing: Fiction
1342,Text,1998-06-01,Pride and Prejudice,en,"Austen, Jane, 1775-1817","Courtship -- Fiction; Love stories; England -- Fiction",PR,"Best Books Ever Listings; Browsing: Fiction"
2000,Text,1999-12-01,Don Quijote,es,"Cervantes Saavedra, Miguel de, 1547-1616","Knights and knighthood -- Spain -- Fiction; Romances",PQ,Browsing: Fiction
2600,Text,2001-04-01,War and Peace,en,"Tolstoy, Leo, graf, 1828-1910; Maude, Aylmer, 1858-1938 [Translator]; Maude, Louise, 1855-1939 [Translator]","Napoleonic Wars, 1800-1815 -- Campaigns -- Russia -- Fiction; War stories",PG,Browsing: Fiction
6130,Text,2004-07-01,The Iliad,en,"Homer, 751? BCE-651? BCE; Pope, Alexander, 1688-1744 [Translator]","Epic poetry, Greek -- Translations into English; Trojan War -- Poetry",PA,"Browsing: Poetry; Classical Antiquity"
17989,Text,2006-03-06,"Le comte de Monte-Cristo, Tome I",fr,"Dumas, Alexandre, 1802-1870; Maquet, Auguste, 1813-1888","Adventure stories; Historical fiction",PQ,Browsing: Fiction
19033,Sound,2006-08-12,Alice's Adventures in Wonderland,en,"Carroll, Lewis, 1832-1898",Fantasy fiction,PR,
70000,Text,2023-02-05,"A Reader in Two
Languages",en; la,"Chaucer, Geoffrey, -1400",Latin language -- Readers,PA,Browsing: Language & Communication
"""

BODY = "CHAPTER I.\n\nIt is a truth universally acknowledged, that a single man in\npossession of a good fortune, must be in want of a wife."


def modern(title: str, body: str = BODY) -> str:
    """A book the way Project Gutenberg generates it today: header, markers, licence, CRLF."""
    return (
        f"The Project Gutenberg eBook of {title}\r\n    \r\n"
        "This ebook is for the use of anyone anywhere in the United States and\r\n"
        "most other parts of the world at no cost and with almost no restrictions\r\n"
        "whatsoever.\r\n\r\n"
        f"Title: {title}\r\n\r\nRelease date: June 1, 1998 [eBook #1342]\r\n\r\nLanguage: English\r\n\r\n"
        f"*** START OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***\r\n\r\n\r\n"
        + body.replace("\n", "\r\n")
        + f"\r\n\r\n\r\n*** END OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***\r\n\r\n"
        "Updated editions will replace the previous one--the old editions will\r\nbe renamed.\r\n"
    )


def pg(number: int) -> str:
    return f"{gd.SITE}/cache/epub/{number}/pg{number}.txt"


def uploaded(number: int, suffix: str = "-0") -> str:
    return f"{gd.SITE}/files/{number}/{number}{suffix}.txt"


def setUpModule():
    global BOOKS, _MODULE_TMP
    _MODULE_TMP = tempfile.TemporaryDirectory()
    path = os.path.join(_MODULE_TMP.name, "pg_catalog.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(CATALOG)
    BOOKS = list(gd.read_catalog(path))


def tearDownModule():
    _MODULE_TMP.cleanup()


def parse(*argv):
    return gd.build_parser().parse_args(list(argv))


def choose(*argv):
    """The numbers of the fixture books a command line's filters keep."""
    filters = gd.filters_from_args(parse(*argv))
    return sorted(book.id for book in BOOKS if filters(book))


def by_id(number):
    return next(book for book in BOOKS if book.id == number)


class TestCreators(unittest.TestCase):
    def test_dates_come_off_the_name(self):
        c = gd.Creator.parse("Austen, Jane, 1775-1817")
        self.assertEqual((c.name, c.birth, c.death, c.role), ("Austen, Jane", 1775, 1817, None))

    def test_years_before_the_common_era_are_negative(self):
        c = gd.Creator.parse("Homer, 751? BCE-651? BCE")
        self.assertEqual((c.name, c.birth, c.death), ("Homer", -751, -651))

    def test_a_bracketed_role_is_not_part_of_the_name(self):
        c = gd.Creator.parse("Maude, Louise, 1855-1939 [Translator]")
        self.assertEqual((c.name, c.role, c.birth, c.death), ("Maude, Louise", "Translator", 1855, 1939))

    def test_a_name_with_commas_keeps_everything_but_the_dates(self):
        self.assertEqual(gd.Creator.parse("Tolstoy, Leo, graf, 1828-1910").name, "Tolstoy, Leo, graf")

    def test_one_known_year_is_enough(self):
        chaucer = gd.Creator.parse("Chaucer, Geoffrey, -1400")
        self.assertEqual((chaucer.name, chaucer.birth, chaucer.death), ("Chaucer, Geoffrey", None, 1400))
        living = gd.Creator.parse("Doe, Jane, 1950-")
        self.assertEqual((living.birth, living.death), (1950, None))

    def test_dates_it_cannot_read_stay_in_the_name(self):
        c = gd.Creator.parse("Marie, de France, active 12th century")
        self.assertEqual((c.name, c.birth, c.death), ("Marie, de France, active 12th century", None, None))

    def test_alive_within(self):
        homer = gd.Creator.parse("Homer, 751? BCE-651? BCE")
        self.assertTrue(homer.alive_within(None, -700))
        self.assertTrue(homer.alive_within(-700, -690))
        self.assertFalse(homer.alive_within(-600, None))
        self.assertFalse(gd.Creator.parse("Anonymous").alive_within(None, None))


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, data):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def test_every_row_is_read_including_a_title_on_two_lines(self):
        self.assertEqual(len(BOOKS), 10)
        self.assertEqual(by_id(70000).title, "A Reader in Two Languages")

    def test_list_columns_are_split(self):
        war = by_id(2600)
        self.assertEqual([c.role for c in war.creators], [None, "Translator", "Translator"])
        self.assertEqual([c.name for c in war.authors], ["Tolstoy, Leo, graf"])
        self.assertEqual(by_id(70000).languages, ("en", "la"))
        self.assertEqual(by_id(84).subjects[0], "Science fiction")
        self.assertEqual(by_id(10).authors, ())

    def test_a_gzipped_catalog_reads_the_same(self):
        path = self.write("pg_catalog.csv.gz", gzip.compress(CATALOG.encode("utf-8")))
        self.assertEqual(list(gd.read_catalog(path)), BOOKS)

    def test_column_order_and_a_byte_order_mark_do_not_matter(self):
        path = self.write("c.csv", "﻿Title,Language,Text#\nPride and Prejudice,en,1342\n".encode("utf-8"))
        (book,) = gd.read_catalog(path)
        self.assertEqual((book.id, book.title, book.languages, book.type), (1342, "Pride and Prejudice", ("en",), "Text"))

    def test_a_file_that_is_not_the_catalog_is_refused(self):
        path = self.write("c.csv", b"<html><body>Service unavailable</body></html>\n")
        with self.assertRaises(gd.CatalogError):
            list(gd.read_catalog(path))
        with self.assertRaises(gd.CatalogError):
            list(gd.read_catalog(os.path.join(self.tmp, "missing.csv")))


class TestMatching(unittest.TestCase):
    def test_words_match_in_any_order(self):
        self.assertTrue(TextQuery("jane austen")("Austen, Jane, 1775-1817"))

    def test_words_match_the_start_of_a_word_not_its_middle(self):
        self.assertTrue(TextQuery("detect")("Detective and mystery stories"))
        self.assertFalse(TextQuery("tective")("Detective and mystery stories"))

    def test_every_word_must_match(self):
        self.assertFalse(TextQuery("jane smith")("Austen, Jane, 1775-1817"))

    def test_case_and_accents_are_ignored(self):
        self.assertTrue(TextQuery("bronte")("Brontë, Charlotte, 1816-1855"))
        self.assertTrue(TextQuery("BRONTË")("Bronte"))

    def test_regular_expressions_on_request(self):
        self.assertTrue(TextQuery(r"^pride\b", regex=True)("Pride and Prejudice"))
        self.assertFalse(TextQuery(r"^prejudice", regex=True)("Pride and Prejudice"))

    def test_a_query_without_words_is_refused(self):
        with self.assertRaises(ValueError):
            TextQuery("--")


class TestFilters(unittest.TestCase):
    def test_only_text_unless_asked(self):
        self.assertEqual(choose("-a", "carroll"), [])
        self.assertEqual(choose("-a", "carroll", "--type", "any"), [19033])

    def test_a_language_matches_any_of_a_books_languages(self):
        self.assertEqual(choose("-l", "la"), [70000])
        self.assertEqual(choose("-l", "fr"), [17989])

    def test_values_of_one_option_are_alternatives_and_options_combine(self):
        self.assertEqual(choose("-l", "en,es", "-s", "fiction"), [84, 1260, 1342, 2000, 2600])
        self.assertEqual(choose("-l", "es", "-l", "fr"), [2000, 17989])

    def test_a_name_matches_translators_too(self):
        self.assertEqual(choose("-a", "maude"), [2600])

    def test_lifetimes_count_authors_not_translators(self):
        self.assertEqual(choose("--author-alive-to", "-500"), [6130])
        self.assertEqual(choose("--author-alive-from", "1800", "--author-alive-to", "1899"), [84, 1260, 1342, 2600, 17989])
        # Tolstoy died in 1910; his translators lived on, but they did not write the book.
        self.assertEqual(choose("--author-alive-from", "1911"), [])

    def test_release_dates_can_be_partial(self):
        self.assertEqual(choose("--released-from", "1998", "--released-to", "1998"), [1260, 1342])
        self.assertEqual(choose("--released-from", "1998-04"), [1342, 2000, 2600, 6130, 17989, 70000])

    def test_numbers_and_ranges(self):
        self.assertEqual(choose("--ids", "1-1000,17989", "--ids", "70000-"), [10, 84, 17989, 70000])

    def test_a_class_is_a_prefix(self):
        self.assertEqual(choose("--locc", "pq"), [2000, 17989])
        self.assertEqual(len(choose("--locc", "P")), 8)

    def test_exclusions_drop_books(self):
        self.assertEqual(choose("-b", "fiction", "--exclude-subject", "love"), [84, 2000, 2600, 17989])
        self.assertEqual(choose("-l", "en", "--exclude-author", "homer", "--exclude-title", "bible"), [84, 1260, 1342, 2600, 70000])

    def test_titles_and_regular_expressions(self):
        self.assertEqual(choose("-t", "pride"), [1342])
        self.assertEqual(choose("--regex", "-t", "^the "), [10, 6130])

    def test_bad_values_are_refused(self):
        for argv in (
            ("--regex", "-t", "("),
            ("--released-from", "2001", "--released-to", "1999"),
            ("--released-from", "2021-02-30"),
            ("--author-alive-from", "1900", "--author-alive-to", "1800"),
            ("--ids", "5-3"),
            ("--ids", "x"),
            ("-a", "!!"),
        ):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                gd.filters_from_args(parse(*argv))

    def test_partial_dates_cover_whole_periods(self):
        self.assertEqual(gd.date_bound("2020", end=False), "2020-01-01")
        self.assertEqual(gd.date_bound("2020-02", end=True), "2020-02-29")
        self.assertEqual(gd.date_bound("2020-2-3", end=True), "2020-02-03")


class TestSorting(unittest.TestCase):
    def ids(self, key, **options):
        return [book.id for book in gd.sort_books(BOOKS, key, **options)]

    def test_title(self):
        self.assertEqual(self.ids("title"), [70000, 19033, 2000, 84, 1260, 17989, 1342, 6130, 10, 2600])

    def test_author_puts_books_without_one_last_either_way(self):
        self.assertEqual(self.ids("author"), [1342, 1260, 19033, 2000, 70000, 17989, 6130, 84, 2600, 10])
        self.assertEqual(self.ids("author", reverse=True), [2600, 84, 6130, 17989, 70000, 2000, 19033, 1260, 1342, 10])

    def test_released(self):
        self.assertEqual(self.ids("released"), [10, 84, 1260, 1342, 2000, 2600, 6130, 17989, 19033, 70000])

    def test_random_is_repeatable_with_a_seed(self):
        first = self.ids("random", seed=7)
        self.assertEqual(first, self.ids("random", seed=7))
        self.assertEqual(sorted(first), self.ids("id"))
        self.assertNotEqual(first, self.ids("random", seed=8))


class TestStripping(unittest.TestCase):
    def test_todays_header_and_licence_come_off(self):
        text, found = gd.strip_boilerplate(modern("Pride and Prejudice").replace("\r\n", "\n"))
        self.assertTrue(found)
        self.assertEqual(text, BODY + "\n")

    def test_the_older_markers_and_the_credits_come_off(self):
        old = (
            "The Project Gutenberg EBook of Pride and Prejudice, by Jane Austen\n\n"
            "This eBook is for the use of anyone anywhere at no cost and with\nalmost no restrictions whatsoever.\n\n"
            "*** START OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n\n\n\n\n"
            "Produced by Anonymous Volunteers\n\n\n\n\n" + BODY + "\n\n\n\n"
            "End of the Project Gutenberg EBook of Pride and Prejudice, by Jane Austen\n\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n\n"
            "***** This file should be named 1342.txt or 1342.zip *****\n"
        )
        self.assertEqual(gd.strip_boilerplate(old), (BODY + "\n", True))

    def test_the_1990s_small_print_comes_off(self):
        old = "**The Project Gutenberg Etext**\nlegalese\n*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*\n\n" + BODY
        self.assertEqual(gd.strip_boilerplate(old), (BODY + "\n", True))

    def test_a_marker_that_wraps_onto_a_second_line(self):
        text = "header\n*** START OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF\nWILLIAM SHAKESPEARE ***\n\n" + BODY
        self.assertEqual(gd.strip_boilerplate(text), (BODY + "\n", True))

    def test_text_without_markers_is_left_alone(self):
        self.assertEqual(gd.strip_boilerplate(BODY), (BODY, False))

    def test_a_long_paragraph_that_starts_like_credits_is_kept(self):
        paragraph = "Produced by the committee, which met\n" + "and met again\n" * 7
        text = "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n" + paragraph + "\n" + BODY
        self.assertTrue(gd.strip_boilerplate(text)[0].startswith("Produced by the committee"))


class TestDecoding(unittest.TestCase):
    def test_utf8_with_and_without_a_byte_order_mark(self):
        self.assertEqual(gd.decode_text(b"\xef\xbb\xbfBront\xc3\xab"), ("Brontë", "utf-8"))
        self.assertEqual(gd.decode_text("Brontë".encode()), ("Brontë", "utf-8"))

    def test_eight_bit_text_falls_back_to_windows_1252(self):
        self.assertEqual(gd.decode_text("Brontë – “quoted”".encode("cp1252")), ("Brontë – “quoted”", "cp1252"))

    def test_the_declared_character_set_is_used(self):
        data = b"Character set encoding: ISO-8859-7\n\n" + "Όμηρος".encode("iso8859-7")
        text, encoding = gd.decode_text(data)
        self.assertEqual((text.split("\n")[-1], encoding), ("Όμηρος", "iso8859-7"))

    def test_a_stray_bad_byte_does_not_cost_a_utf8_book_its_text(self):
        text, encoding = gd.decode_text(b"Character set encoding: UTF-8\n\nna\xefve caf\xc3\xa9")
        self.assertEqual((text.split("\n")[-1], encoding), ("na�ve café", "utf-8"))


class TestPlaces(unittest.TestCase):
    def test_the_generated_file_first_then_the_uploaded_ones(self):
        self.assertEqual(gd.text_urls(1342), [pg(1342), uploaded(1342), uploaded(1342, ""), uploaded(1342, "-8")])

    def test_a_mirror_keeps_uploads_in_numbered_folders(self):
        self.assertEqual(gd.text_urls(5, "http://m/")[1], "http://m/0/5/5-0.txt")
        self.assertEqual(gd.text_urls(11, "http://m")[1], "http://m/1/11/11-0.txt")
        self.assertEqual(gd.text_urls(12345, "http://m")[0], "http://m/cache/epub/12345/pg12345.txt")
        self.assertEqual(gd.text_urls(12345, "http://m")[3], "http://m/1/2/3/4/12345/12345-8.txt")

    def test_file_names(self):
        pride = by_id(1342)
        self.assertEqual(gd.book_filename(pride), "1342-pride-and-prejudice.txt")
        self.assertEqual(gd.book_filename(pride, "{language}/{author}/{id:05d}"), "en/austen-jane/01342.txt")
        self.assertEqual(gd.book_filename(by_id(10), "{author}-{id}"), "anonymous-10.txt")

    def test_file_names_that_would_collide_or_escape_are_refused(self):
        for name_format in ("{title}", "{id}-{nope}", "../{id}", "/tmp/{id}", "{id:q}", "{}{id}"):
            with self.subTest(name_format=name_format), self.assertRaises(ValueError):
                gd.check_name_format(name_format)


class TestCatalogCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name
        self.gz, self.plain = gd.CATALOG_URLS
        self.warnings = []

    def tearDown(self):
        self._tmp.cleanup()

    def ensure(self, site, **options):
        clock = FakeClock()
        fetcher = Fetcher("t", delay=0, retries=0, opener=site, sleep=clock.sleep, clock=clock)
        return gd.ensure_catalog(fetcher, self.cache, warn=self.warnings.append, **options)

    def cached(self, name="pg_catalog.csv.gz", age=0.0):
        path = os.path.join(self.cache, name)
        with open(path, "wb") as fh:
            fh.write(gzip.compress(CATALOG.encode()))
        os.utime(path, (time.time() - age, time.time() - age))
        return path

    def test_a_fresh_copy_is_used_without_asking(self):
        path = self.cached()
        site = FakeSite()
        self.assertEqual(self.ensure(site), path)
        self.assertEqual(site.requests, [])

    def test_a_stale_copy_is_replaced(self):
        path = self.cached(age=8 * 24 * 3600)
        site = FakeSite({self.gz: gzip.compress(CATALOG.encode())})
        self.assertEqual(self.ensure(site), path)
        self.assertEqual(site.requests, [self.gz])
        self.assertLess(time.time() - os.path.getmtime(path), 60)

    def test_a_stale_copy_beats_no_catalog_at_all(self):
        path = self.cached(age=8 * 24 * 3600)
        site = FakeSite({self.gz: http_error(self.gz, 503)})
        self.assertEqual(self.ensure(site), path)
        self.assertIn("using the cached one", self.warnings[0])

    def test_the_plain_file_when_the_compressed_one_is_missing(self):
        site = FakeSite({self.plain: CATALOG.encode()})
        self.assertTrue(self.ensure(site).endswith("pg_catalog.csv"))
        self.assertEqual(site.requests, [self.gz, self.plain])

    def test_no_copy_and_no_network_is_an_error(self):
        with self.assertRaises(gd.CatalogError):
            self.ensure(FakeSite({self.gz: http_error(self.gz, 503)}))

    def test_an_error_page_is_not_cached_as_the_catalog(self):
        with self.assertRaises(gd.CatalogError):
            self.ensure(FakeSite({self.gz: b"<html>Please try again later</html>"}))
        self.assertEqual(os.listdir(self.cache), [])


class CommandLineCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.out = os.path.join(self.tmp, "books")
        self.cache = os.path.join(self.tmp, "cache")
        self.catalog = os.path.join(self.tmp, "pg_catalog.csv")
        with open(self.catalog, "w", encoding="utf-8", newline="") as fh:
            fh.write(CATALOG)
        self.site = FakeSite(
            {
                pg(84): modern("Frankenstein; Or, The Modern Prometheus").encode("utf-8"),
                pg(1342): modern("Pride and Prejudice").encode("utf-8"),
                uploaded(1260): "Jane Eyre by Charlotte Brontë\n".encode("utf-8"),
                pg(6130): http_error(pg(6130), 500),
            }
        )

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *argv, catalog=True, site=None):
        """Run the command line against the fake site: ``(exit status, stdout, stderr)``."""
        argv = [*argv, "--delay", "0", "--retries", "0", "-o", self.out, "--cache-dir", self.cache]
        if catalog:
            argv += ["--catalog", self.catalog]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = gd.main(argv, opener=site or self.site, sleep=lambda seconds: None)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def manifest(self):
        with open(os.path.join(self.out, gd.MANIFEST), encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]


class TestDownloading(CommandLineCase):
    def test_books_are_saved_with_a_manifest_that_vouches_for_them(self):
        code, out, err = self.run_cli("-a", "shelley", "--strip")
        self.assertEqual(code, 0, err)
        name = "84-frankenstein-or-the-modern-prometheus.txt"
        with open(os.path.join(self.out, name), "rb") as fh:
            data = fh.read()
        self.assertEqual(data.decode("utf-8"), BODY + "\n")  # stripped, and CRLF gone
        (entry,) = self.manifest()
        self.assertEqual((entry["id"], entry["file"], entry["url"], entry["stripped"]), (84, name, pg(84), True))
        self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(entry["authors"], ["Shelley, Mary Wollstonecraft, 1797-1851"])
        self.assertIn("1 saved", out)

    def test_without_strip_the_whole_file_is_kept(self):
        self.run_cli("-t", "pride")
        with open(os.path.join(self.out, "1342-pride-and-prejudice.txt"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertTrue(text.startswith("The Project Gutenberg eBook of Pride and Prejudice\n"))
        self.assertNotIn("\r", text)

    def test_a_second_run_fetches_nothing(self):
        self.run_cli("-t", "pride")
        again = FakeSite(self.site.pages)
        code, out, _ = self.run_cli("-t", "pride", site=again)
        self.assertEqual((code, again.requests), (0, []))
        self.assertIn("0 saved, 1 already there", out)

    def test_overwrite_fetches_again(self):
        self.run_cli("-t", "pride")
        again = FakeSite(self.site.pages)
        self.run_cli("-t", "pride", "--overwrite", site=again)
        self.assertEqual(again.requests, [pg(1342)])

    def test_a_book_without_the_generated_file_falls_back_to_the_upload(self):
        code, _, err = self.run_cli("-a", "bronte")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.manifest()[0]["url"], uploaded(1260))

    def test_the_limit_counts_books_on_disk_not_books_tried(self):
        """War and Peace has no text here, so Pride and Prejudice takes its place; Frankenstein is never asked for."""
        code, out, _ = self.run_cli("--ids", "84,1342,2600", "--reverse", "-n", "1")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(os.listdir(self.out)), ["1342-pride-and-prejudice.txt", gd.MANIFEST])
        self.assertEqual(self.site.asked_for("/84/"), [])
        self.assertIn("no text", out)
        self.assertIn("1 without a plain-text file", out)

    def test_a_failed_book_fails_the_run_but_not_the_others(self):
        code, out, _ = self.run_cli("--ids", "1342,6130")
        self.assertEqual(code, 1)
        self.assertTrue(os.path.exists(os.path.join(self.out, "1342-pride-and-prejudice.txt")))
        self.assertIn("failed: 6130", out)

    def test_a_refusal_stops_the_run_at_once(self):
        self.site.pages[pg(84)] = http_error(pg(84), 403)
        code, out, err = self.run_cli("--ids", "84,1342")
        self.assertEqual(code, 1)
        self.assertEqual(self.site.requests, [pg(84)])
        self.assertIn("refuses", err)
        self.assertIn("0 saved", out)

    def test_three_failures_in_a_row_stop_the_run(self):
        for number in (84, 1260, 1342, 2000, 2600):
            self.site.pages[pg(number)] = http_error(pg(number), 500)
        code, _, err = self.run_cli("--ids", "84-3000")
        self.assertEqual(code, 1)
        self.assertEqual(len(self.site.requests), 3)
        self.assertIn("in a row", err)

    def test_folders_from_the_name_format(self):
        code, _, err = self.run_cli("-t", "pride", "--name-format", "{language}/{id}")
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(os.path.join(self.out, "en", "1342.txt")))
        self.assertEqual(self.manifest()[0]["file"], "en/1342.txt")

    def test_the_catalog_is_downloaded_once_and_then_cached(self):
        self.site.pages[gd.CATALOG_URLS[0]] = gzip.compress(CATALOG.encode())
        code, _, err = self.run_cli("-t", "pride", catalog=False)
        self.assertEqual(code, 0, err)
        again = FakeSite(self.site.pages)
        self.run_cli("-t", "pride", "--list", catalog=False, site=again)
        self.assertEqual(again.requests, [])


class TestOtherModes(CommandLineCase):
    def test_list_shows_the_matches_and_fetches_nothing(self):
        code, out, err = self.run_cli("-l", "fr", "--list")
        self.assertEqual(code, 0)
        self.assertIn("17989", out)
        self.assertIn("Dumas, Alexandre", out)
        self.assertIn("1 of 10 catalog entries match", err)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.site.requests, [])

    def test_facets_count_what_there_is_to_filter_on(self):
        code, out, _ = self.run_cli("--facet", "language")
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines(), ["      7  en", "      1  es", "      1  fr", "      1  la"])

    def test_random_lists_repeat_with_a_seed(self):
        first = self.run_cli("--list", "--sort", "random", "--seed", "3")[1]
        self.assertEqual(first, self.run_cli("--list", "--sort", "random", "--seed", "3")[1])

    def test_downloading_everything_needs_a_filter_or_a_limit(self):
        code, _, err = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("whole catalog", err)
        code, _, _ = self.run_cli("-n", "1")
        self.assertEqual(code, 0)
        self.assertEqual(len([n for n in os.listdir(self.out) if n.endswith(".txt")]), 1)

    def test_no_arguments_prints_the_help(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(gd.main([]), 0)
        self.assertIn("usage:", out.getvalue())

    def test_bad_arguments_are_usage_errors(self):
        for argv in (("-t", "x", "--name-format", "{title}"), ("-t", "x", "--mirror", "ftp://x"), ("-n", "0")):
            with self.subTest(argv=argv):
                self.assertEqual(self.run_cli(*argv)[0], 2)


class TestOverRealHttp(unittest.TestCase):
    """The same download through urllib's real sockets, from a mirror-shaped local server."""

    def test_a_download_from_a_mirror(self):
        pages = {
            "/cache/epub/feeds/pg_catalog.csv.gz": gzip.compress(CATALOG.encode()),
            "/cache/epub/1342/pg1342.txt": modern("Pride and Prejudice").encode(),
            "/1/2/6/1260/1260-0.txt": "Jane Eyre\n".encode(),
        }
        with tempfile.TemporaryDirectory() as tmp, LocalServer(pages) as server:
            argv = [
                "-l", "en", "--released-from", "1998", "--released-to", "1998", "--strip",
                "--mirror", server.url, "--catalog", server.url + "/cache/epub/feeds/pg_catalog.csv.gz",
                "--cache-dir", os.path.join(tmp, "cache"), "-o", os.path.join(tmp, "out"), "--delay", "0",
            ]  # fmt: skip
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = gd.main(argv, opener=no_proxy_opener())
            self.assertEqual(code, 0, err.getvalue())
            names = sorted(os.listdir(os.path.join(tmp, "out")))
            self.assertEqual(names, ["1260-jane-eyre-an-autobiography.txt", "1342-pride-and-prejudice.txt", gd.MANIFEST])
            with open(os.path.join(tmp, "out", "1342-pride-and-prejudice.txt"), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), BODY + "\n")
            # the pg1260 file is missing, so the numbered folder was the fallback
            self.assertIn("/cache/epub/1260/pg1260.txt", server.requests)
            self.assertIn("/1/2/6/1260/1260-0.txt", server.requests)


if __name__ == "__main__":
    unittest.main()
