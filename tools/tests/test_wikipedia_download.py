"""Tests for wikipedia_download.py.

Run: python3 -m unittest discover -s tools/tests  (from the repository root)

Offline and deterministic. The dumps are small XML files written here, the dump
server is a fake of dumps.wikimedia.org's index pages and dumpstatus.json, and
the API is a fake MediaWiki that answers the queries the tool makes. The last
class goes through a real HTTP server on 127.0.0.1.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
import urllib.parse
import urllib.response
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock
from xml.sax.saxutils import escape, quoteattr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wikipedia_download as wd  # noqa: E402
from common import Fetcher  # noqa: E402
from fakes import FakeClock, FakeSite, LocalServer, http_error, no_proxy_opener  # noqa: E402

PARIS = """{{Short description|Capital of France}}
{{Infobox settlement
| name = Paris
| image = {{Multiple image|a|b}}
}}
'''Paris''' ({{IPA|fr|paʁi|lang}}; {{lang|fr|Paris}}) is the [[capital city|capital]] of [[France]], with an area of \
{{convert|105|km2|sqmi}}.<ref>{{cite web|url=x}}</ref> It was founded {{circa|250 BC}}.<ref name="a" />
[[File:Paris.jpg|thumb|The [[Eiffel Tower]] in [[Paris]]]]

== History ==
The [[Parisii (Gaul)|Parisii]] lived there.<!-- comment --> See [[Paris, Texas|]] and [[Seine]]s.
{| class="wikitable"
|-
| a || b
|}
* first item
* second ''item'' &ndash; with&nbsp;entities

== Empty section ==
{{main|Something}}

== See also ==
* [[Île-de-France]]

== References ==
{{reflist}}

[[Category:Capitals in Europe]]
[[Category:Cities in France|Paris]]
[[fr:Paris]]
"""

PARIS_TEXT = """\
Paris (Paris) is the capital of France, with an area of 105 km2. It was founded c. 250 BC.

History

The Parisii lived there. See Paris and Seines.

first item
second item – with entities"""

PAGES = [
    {"id": 10, "title": "Paris", "text": PARIS},
    {"id": 11, "title": "Paris, France", "text": "#REDIRECT [[Paris]]", "redirect": "Paris"},
    {"id": 12, "title": "Talk:Paris", "ns": 1, "text": "Let us talk about Paris."},
    {"id": 13, "title": "Mercury", "text": "'''Mercury''' may refer to:\n* [[Mercury (planet)]]\n{{disambiguation}}"},
    {"id": 14, "title": "Stub", "text": "'''Stub''' is short.\n[[Category:Stubs]]"},
    {"id": 15, "title": "Berlin", "text": "'''Berlin''' is the capital of [[Germany]].\n[[Category:Capitals in Europe]]"},
    {"id": 16, "title": "Template:Infobox", "ns": 10, "text": "{{{name}}}"},
]


def dump_xml(pages=PAGES, *, schema="0.11", file_ns="File", category_ns="Category") -> bytes:
    """A MediaWiki XML dump of ``pages``, the way the dumps are written."""
    lines = [
        f'<mediawiki xmlns="http://www.mediawiki.org/xml/export-{schema}/" version="{schema}" xml:lang="en">',
        "  <siteinfo>",
        "    <sitename>Wikipedia</sitename>",
        "    <dbname>testwiki</dbname>",
        "    <base>https://en.wikipedia.org/wiki/Main_Page</base>",
        "    <case>first-letter</case>",
        "    <namespaces>",
        '      <namespace key="0" case="first-letter" />',
        '      <namespace key="1" case="first-letter">Talk</namespace>',
        f'      <namespace key="6" case="first-letter">{file_ns}</namespace>',
        '      <namespace key="10" case="first-letter">Template</namespace>',
        f'      <namespace key="14" case="first-letter">{category_ns}</namespace>',
        "    </namespaces>",
        "  </siteinfo>",
    ]
    for page in pages:
        redirect = f"    <redirect title={quoteattr(page['redirect'])} />" if page.get("redirect") else ""
        lines += [
            "  <page>",
            f"    <title>{escape(page['title'])}</title>",
            f"    <ns>{page.get('ns', 0)}</ns>",
            f"    <id>{page['id']}</id>",
            redirect,
            "    <revision>",
            f"      <id>{page['id'] * 10}</id>",
            "      <timestamp>2026-09-01T12:00:00Z</timestamp>",
            f'      <text bytes="{len(page["text"])}" xml:space="preserve">{escape(page["text"])}</text>',
            "    </revision>",
            "  </page>",
        ]
    lines.append("</mediawiki>")
    return "\n".join(line for line in lines if line).encode("utf-8")


def run(*argv, opener=None, sleep=lambda seconds: None):
    """Run the command line: ``(exit status, stdout, stderr)``."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = wd.main([str(a) for a in argv], opener=opener, sleep=sleep)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def write(self, name, data):
        with open(self.path(name), "wb") as fh:
            fh.write(data)
        return self.path(name)


# ---------------------------------------------------------------------------
# wikitext
# ---------------------------------------------------------------------------


class TestConverter(unittest.TestCase):
    def setUp(self):
        self.converter = wd.WikitextConverter({6: "File", 14: "Category"})

    def text(self, wikitext):
        return self.converter.convert(wikitext)[0]

    def test_an_article(self):
        text, categories = self.converter.convert(PARIS)
        self.assertEqual(text, PARIS_TEXT)
        self.assertEqual(categories, ["Capitals in Europe", "Cities in France"])

    def test_nested_templates_go_whole(self):
        self.assertEqual(self.text("a {{outer|{{inner|{{deepest}}}}|x}} b"), "a b")

    def test_inline_templates_that_carry_words_are_rendered(self):
        cases = {
            "{{nowrap|10 June}}": "10 June",
            "{{lang|de|Straße}}": "Straße",
            "{{lang-fr|Paris}}": "Paris",
            "{{convert|5|-|10|km|mi}}": "5–10 km",
            "{{convert|3|and|4|m}}": "3 and 4 m",
            "{{frac|1|2}}": "1/2",
            "{{As of|2019|5}}": "As of May 2019",
            "{{ill|Foo|de|Foo (Begriff)}}": "Foo",
            "{{nowrap|{{convert|2|kg}}}}": "2 kg",
            "{{snd}}": "–",
            "{{cite book|title=Gone}}": "",
        }
        for wikitext, expected in cases.items():
            with self.subTest(wikitext=wikitext):
                self.assertEqual(self.text(f"x {wikitext} y"), re.sub(" +", " ", f"x {expected} y"))

    def test_references_comments_and_formulas_go(self):
        self.assertEqual(self.text('A<ref name="x">{{cite|y}}</ref> B<ref name=x/> C<!-- hidden --> D<math>x^2</math>.'), "A B C D.")

    def test_a_self_closing_reference_does_not_swallow_what_follows(self):
        self.assertEqual(self.text('One.<ref name="a" /> Two.<ref>cite</ref> Three.'), "One. Two. Three.")

    def test_links(self):
        cases = {
            "[[France]]": "France",
            "[[capital city|capital]]": "capital",
            "[[Paris, Texas|]]": "Paris",
            "[[Pipe (computing)|]]": "Pipe",
            "[[bus]]es": "buses",
            "[[:Category:Cities]]": "Category:Cities",
            "[http://example.org the site]": "the site",
            "[http://example.org]": "",
            "[[File:X.jpg|thumb|A [[b|c]] d]]": "",
            "[[Image:Y.png]]": "",
            "[[de:Paris]]": "",
        }
        for wikitext, expected in cases.items():
            with self.subTest(wikitext=wikitext):
                self.assertEqual(self.text(f"x {wikitext} y"), re.sub(" +", " ", f"x {expected} y"))

    def test_bold_and_italic_go_but_a_real_apostrophe_stays(self):
        self.assertEqual(
            self.text("'''Bold''' and ''italic'' and '''''both''''' and the '''Smiths''''s cat"),
            "Bold and italic and both and the Smiths's cat",
        )

    def test_tables_go_nested_or_not(self):
        self.assertEqual(self.text("Before.\n{| class=x\n| a\n{|\n| inner\n|}\n| b\n|}\nAfter."), "Before.\n\nAfter.")

    def test_headings_become_lines_and_reference_sections_go(self):
        wikitext = (
            "Intro.\n== History ==\nOld.\n=== Early ===\nOlder.\n== References ==\n* A book\n"
            "=== Under references ===\n* more\n== Legacy ==\nNew."
        )
        self.assertEqual(self.text(wikitext), "Intro.\n\nHistory\n\nOld.\n\nEarly\n\nOlder.\n\nLegacy\n\nNew.")
        kept = wd.WikitextConverter(keep_sections=True).convert(wikitext)[0]
        self.assertIn("References", kept)
        self.assertIn("A book", kept)

    def test_a_heading_left_with_nothing_under_it_goes(self):
        wikitext = "Text.\n== Gallery ==\n<gallery>\nA.jpg\n</gallery>\n== Next ==\nMore."
        self.assertEqual(self.text(wikitext), "Text.\n\nNext\n\nMore.")

    def test_lists_keep_their_words(self):
        self.assertEqual(self.text("* one\n** two\n# three\n; term : definition"), "one\ntwo\nthree\nterm : definition")

    def test_entities_are_decoded(self):
        self.assertEqual(self.text("A&nbsp;B &amp; C&ndash;D &lt;tag&gt;"), "A B & C–D <tag>")

    def test_files_and_categories_in_the_wikis_own_language(self):
        german = wd.WikitextConverter({6: "Datei", 14: "Kategorie"})
        text, categories = german.convert(
            "Berlin ist die [[Hauptstadt]].\n[[Datei:Tor.jpg|mini|Das [[Brandenburger Tor]]]]\n"
            "[[Kategorie:Hauptstadt in Europa]]\n[[Category:Canonical names work too]]"
        )
        self.assertEqual(text, "Berlin ist die Hauptstadt.")
        self.assertEqual(categories, ["Hauptstadt in Europa", "Canonical names work too"])

    def test_broken_markup_does_not_cost_the_rest_of_the_article(self):
        self.assertIn("The rest of the article.", self.text("Start {{broken template\n\nThe rest of the article."))
        self.assertIn("The rest.", self.text("Start [[broken link\n\nThe rest."))

    def test_textextracts_output_is_tidied_the_same_way(self):
        extract = (
            "Energy is a quantity.\nIt is conserved.\n\n\n== History ==\nThe word comes from Greek.\n\n\n"
            "== See also ==\nPower\n\n\n== References ==\n"
        )
        self.assertEqual(wd.tidy_text(extract), "Energy is a quantity.\nIt is conserved.\n\nHistory\n\nThe word comes from Greek.")


# ---------------------------------------------------------------------------
# reading and extracting dumps
# ---------------------------------------------------------------------------


class TestDumpReader(TempCase):
    def test_the_siteinfo_and_every_page(self):
        path = self.write("d.xml.bz2", bz2.compress(dump_xml()))
        with wd.DumpReader(path) as reader:
            site = reader.site
            pages = list(reader.pages())
        self.assertEqual((site.dbname, site.base), ("testwiki", "https://en.wikipedia.org/wiki/Main_Page"))
        self.assertEqual(site.namespaces, {0: "", 1: "Talk", 6: "File", 10: "Template", 14: "Category"})
        self.assertEqual([p.id for p in pages], [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual((pages[1].redirect, pages[0].redirect), ("Paris", None))
        self.assertEqual((pages[2].ns, pages[0].revision, pages[0].timestamp), (1, 100, "2026-09-01T12:00:00Z"))
        self.assertEqual(pages[0].text, PARIS)

    def test_gzip_plain_and_an_older_schema_read_the_same(self):
        expected = [p.id for p in wd.DumpReader(self.write("a.xml.bz2", bz2.compress(dump_xml()))).pages()]
        for name, data in (
            ("b.xml.gz", gzip.compress(dump_xml())),
            ("c.xml", dump_xml()),
            ("d.xml", dump_xml(schema="0.10")),
        ):
            with self.subTest(name=name), wd.DumpReader(self.write(name, data)) as reader:
                self.assertEqual([p.id for p in reader.pages()], expected)

    def test_a_damaged_dump_is_reported_not_half_read_in_silence(self):
        whole = bz2.compress(dump_xml() * 1)
        path = self.write("cut.xml.bz2", whole[: len(whole) // 2])
        with self.assertRaises(wd.ExtractError):
            with wd.DumpReader(path) as reader:
                list(reader.pages())

    def test_article_urls(self):
        site = wd.SiteInfo(base="https://en.wikipedia.org/wiki/Main_Page")
        self.assertEqual(site.article_url("Paris, France"), "https://en.wikipedia.org/wiki/Paris,_France")
        self.assertEqual(site.article_url("C++"), "https://en.wikipedia.org/wiki/C%2B%2B")


class TestExtract(TempCase):
    def setUp(self):
        super().setUp()
        self.dump = self.write("testwiki-20260901-pages-articles.xml.bz2", bz2.compress(dump_xml()))
        self.runs = 0

    def extract(self, *argv, out=None):
        """Extract into a fresh directory; returns (exit status, stdout, stderr, directory)."""
        self.runs += 1
        out = out or self.path(f"text{self.runs}")
        return (*run("extract", self.dump, "-o", out, "-p", "1", "-q", *argv), out)

    def articles(self, out, name="wiki_00000.jsonl"):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(os.path.join(out, name), "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    def titles(self, *argv):
        code, _, err, out = self.extract(*argv)
        self.assertEqual(code, 0, err)
        return [a["title"] for a in self.articles(out)]

    def test_articles_become_json_lines_with_a_manifest(self):
        code, stdout, err, out = self.extract()
        self.assertEqual(code, 0, err)
        paris, stub, berlin = self.articles(out)
        self.assertEqual((paris["id"], paris["title"], paris["text"]), (10, "Paris", PARIS_TEXT))
        self.assertEqual((paris["url"], paris["revision"]), ("https://en.wikipedia.org/wiki/Paris", 100))
        self.assertEqual(paris["categories"], ["Capitals in Europe", "Cities in France"])
        self.assertEqual((stub["text"], berlin["text"]), ("Stub is short.", "Berlin is the capital of Germany."))
        with open(os.path.join(out, wd.EXTRACT_MANIFEST), encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(
            manifest["counts"],
            {"pages read": 7, "articles": 3, "redirects": 1, "other namespaces": 2, "disambiguation pages": 1},
        )
        self.assertEqual(manifest["files"], ["wiki_00000.jsonl"])
        self.assertIsNotNone(manifest["finished"])
        self.assertIn("3 articles from 7 pages", stdout)

    def test_filters(self):
        self.assertEqual(self.titles("-c", "capitals europe"), ["Paris", "Berlin"])
        self.assertEqual(self.titles("--exclude-category", "stubs"), ["Paris", "Berlin"])
        self.assertEqual(self.titles("-t", "berlin"), ["Berlin"])
        self.assertEqual(self.titles("--min-chars", "20"), ["Paris", "Berlin"])
        self.assertEqual(self.titles("-n", "1"), ["Paris"])
        self.assertEqual(self.titles("--keep-disambiguation"), ["Paris", "Mercury", "Stub", "Berlin"])
        self.assertEqual(self.titles("--namespace", "0", "--namespace", "1"), ["Paris", "Talk:Paris", "Stub", "Berlin"])
        self.assertEqual(self.titles("--regex", "-t", "^B"), ["Berlin"])

    def test_one_text_file_per_article(self):
        code, _, err, out = self.extract("--format", "txt")
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(os.listdir(os.path.join(out, "00000"))), ["10-paris.txt", "14-stub.txt", "15-berlin.txt"])
        with open(os.path.join(out, "00000", "10-paris.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "Paris\n\n" + PARIS_TEXT + "\n")

    def test_several_processes_write_the_same_articles(self):
        _, _, _, one = self.extract()
        code, _, err, out = (*run("extract", self.dump, "-o", self.path("many"), "-p", "2", "-q"), self.path("many"))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.articles(out), self.articles(one))

    def test_compressed_shards(self):
        code, _, err, out = self.extract("--gzip")
        self.assertEqual(code, 0, err)
        self.assertEqual([a["id"] for a in self.articles(out, "wiki_00000.jsonl.gz")], [10, 14, 15])

    def test_shards_rotate_at_their_size(self):
        shards = wd.JsonlShards(self.tmp, shard_bytes=100)
        for number in range(3):
            shards.write({"id": number, "text": "x" * 80})
        shards.close()
        self.assertEqual(shards.files, ["wiki_00000.jsonl", "wiki_00001.jsonl", "wiki_00002.jsonl"])

    def test_it_will_not_mix_with_an_earlier_extract(self):
        _, _, _, out = self.extract()
        with open(os.path.join(out, wd.EXTRACT_MANIFEST), encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["files"].append("wiki_00001.jsonl")  # as if the last run wrote two shards
        self.write(os.path.join(out, "wiki_00001.jsonl"), b"{}\n")
        self.write(os.path.join(out, wd.EXTRACT_MANIFEST), json.dumps(manifest).encode())
        self.write(os.path.join(out, "notes.txt"), b"mine")
        code, _, err, _ = self.extract(out=out)
        self.assertEqual(code, 1)
        self.assertIn("not empty", err)
        code, _, err, _ = self.extract("--overwrite", out=out)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(os.listdir(out)), [wd.EXTRACT_MANIFEST, "notes.txt", "wiki_00000.jsonl"])

    def test_a_pathological_page_is_skipped_not_fatal(self):
        deep = "{{nowrap|" * 5000 + "deep" + "}}" * 5000
        self.dump = self.write("deep.xml.bz2", bz2.compress(dump_xml([{"id": 1, "title": "Deep", "text": deep}, *PAGES])))
        code, _, err, out = self.extract()
        self.assertEqual(code, 0, err)
        with open(os.path.join(out, wd.EXTRACT_MANIFEST), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["counts"]["nested too deeply to convert"], 1)
        self.assertEqual(len(self.articles(out)), 3)

    def test_a_missing_dump_is_an_error(self):
        self.assertEqual(run("extract", self.path("nope.xml.bz2"), "-o", self.path("o"))[0], 1)


# ---------------------------------------------------------------------------
# finding and downloading dumps
# ---------------------------------------------------------------------------

WIKI, NEW, OLD = "testwiki", "20260920", "20260901"
BASE = f"{wd.DUMPS}/{WIKI}"


def index_html(dates):
    """dumps.wikimedia.org's listing of a wiki's dump dates."""
    rows = "".join(f'<a href="{d}/">{d}/</a>                                          01-Sep-2026 00:00    -\n' for d in dates)
    return (
        f"<html><head><title>Index of /{WIKI}/</title></head><body><h1>Index of /{WIKI}/</h1><hr><pre>"
        f'<a href="../">../</a>\n{rows}<a href="latest/">latest/</a>\n</pre><hr></body></html>'
    ).encode()


def status_json(jobs):
    """A dumpstatus.json: ``jobs`` maps a job name to (status, {file name: bytes})."""
    return json.dumps(
        {
            "jobs": {
                job: {
                    "status": state,
                    "updated": "2026-09-02 00:00:00",
                    "files": {
                        name: {"size": len(data), "url": f"/{WIKI}/x/{name}", "md5": hashlib.md5(data).hexdigest(), "sha1": hashlib.sha1(data).hexdigest()}
                        for name, data in files.items()
                    },
                }
                for job, (state, files) in jobs.items()
            },
            "version": "0.8",
        }
    ).encode()


def ranged(data):
    """A page that honours ``Range: bytes=N-`` the way a real server does."""

    def page(request):
        wanted = request.get_header("Range")
        if not wanted:
            return data
        start = int(wanted[len("bytes=") : -1])
        if start >= len(data):
            return http_error(request.full_url, 416)
        return data[start:], {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"}, 206

    return page


class DropsAfter(io.RawIOBase):
    """A response body whose connection drops after ``cut`` bytes."""

    def __init__(self, data, cut):
        self.data, self.cut, self.pos = data, cut, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.pos >= self.cut:
            raise ConnectionResetError("connection reset by peer")
        count = min(len(buffer), self.cut - self.pos)
        buffer[:count] = self.data[self.pos : self.pos + count]
        self.pos += count
        return count


def fetcher(site, retries=2):
    clock = FakeClock()
    return Fetcher("t", delay=0, retries=retries, opener=site, sleep=clock.sleep, clock=clock)


class TestFindingDumps(unittest.TestCase):
    def parts_status(self, whole_state="in-progress"):
        return {
            "articlesdump": ("done", {f"{WIKI}-{NEW}-pages-articles10.xml-p9p12.bz2": b"b", f"{WIKI}-{NEW}-pages-articles2.xml-p1p8.bz2": b"a"}),
            "articlesdumprecombine": (whole_state, {f"{WIKI}-{NEW}-pages-articles.xml.bz2": b"ab"}),
            "articlesmultistreamdump": ("done", {f"{WIKI}-{NEW}-pages-articles-multistream.xml.bz2": b"m", f"{WIKI}-{NEW}-pages-articles-multistream-index.txt.bz2": b"i"}),
        }

    def select(self, kind, parts=False, whole_state="in-progress"):
        files, done = wd.select_files(json.loads(status_json(self.parts_status(whole_state))), WIKI, NEW, kind, parts)
        return [f.name.split(f"{NEW}-")[1] for f in files], done

    def test_dates_come_newest_first(self):
        site = FakeSite({f"{BASE}/": index_html([OLD, NEW])})
        self.assertEqual(wd.dump_dates(fetcher(site), WIKI), [NEW, OLD])

    def test_finished_parts_stand_in_for_an_unfinished_whole(self):
        self.assertEqual(self.select("articles"), (["pages-articles2.xml-p1p8.bz2", "pages-articles10.xml-p9p12.bz2"], True))
        self.assertEqual(self.select("articles", whole_state="done"), (["pages-articles.xml.bz2"], True))
        self.assertEqual(self.select("articles", parts=True, whole_state="done")[0][0], "pages-articles2.xml-p1p8.bz2")

    def test_kinds_do_not_bleed_into_each_other(self):
        self.assertEqual(self.select("multistream"), (["pages-articles-multistream-index.txt.bz2", "pages-articles-multistream.xml.bz2"], True))
        self.assertEqual(self.select("meta-current"), ([], False))

    def test_the_newest_finished_dump_is_the_one_chosen(self):
        name = lambda date: f"{WIKI}-{date}-pages-articles.xml.bz2"  # noqa: E731
        site = FakeSite(
            {
                f"{BASE}/": index_html([OLD, NEW]),
                f"{BASE}/{NEW}/dumpstatus.json": status_json({"articlesdump": ("in-progress", {name(NEW): b"x"})}),
                f"{BASE}/{OLD}/dumpstatus.json": status_json({"articlesdump": ("done", {name(OLD): b"x"})}),
            }
        )
        said = []
        date, files = wd.find_dump(fetcher(site), WIKI, "articles", log=said.append)
        self.assertEqual((date, [f.name for f in files]), (OLD, [name(OLD)]))
        self.assertIn("not finished", said[0])
        with self.assertRaises(wd.DumpError):
            wd.find_dump(fetcher(site), WIKI, "articles", date=NEW)

    def test_an_unknown_wiki(self):
        with self.assertRaises(wd.DumpError):
            wd.dump_dates(fetcher(FakeSite()), "nosuchwiki")


class TestDownloadFile(TempCase):
    URL = "https://dumps.example/testwiki/file.bz2"
    DATA = bytes(range(256)) * 40

    def download(self, site, **options):
        options.setdefault("size", len(self.DATA))
        options.setdefault("sha1", hashlib.sha1(self.DATA).hexdigest())
        return wd.download_file(fetcher(site), self.URL, self.path("file.bz2"), **options)

    def saved(self):
        with open(self.path("file.bz2"), "rb") as fh:
            return fh.read()

    def test_a_download_is_checked_and_moved_into_place(self):
        self.assertEqual(self.download(FakeSite({self.URL: ranged(self.DATA)})), "saved")
        self.assertEqual(self.saved(), self.DATA)
        self.assertFalse(os.path.exists(self.path("file.bz2.part")))

    def test_a_partial_download_is_resumed_from_where_it_stopped(self):
        self.write("file.bz2.part", self.DATA[:1000])
        site = FakeSite({self.URL: ranged(self.DATA)})
        self.download(site)
        self.assertEqual(site.headers[0]["Range"], "bytes=1000-")
        self.assertEqual(self.saved(), self.DATA)

    def test_a_server_that_ignores_the_range_is_downloaded_from_the_start(self):
        self.write("file.bz2.part", b"junk" * 250)
        self.download(FakeSite({self.URL: self.DATA}))
        self.assertEqual(self.saved(), self.DATA)

    def test_a_dropped_connection_is_resumed(self):
        dropping = lambda request: urllib.response.addinfourl(DropsAfter(self.DATA, 3000), {}, self.URL, 200)  # noqa: E731
        site = FakeSite({self.URL: [dropping, ranged(self.DATA)]})
        self.download(site)
        self.assertEqual(site.headers[1]["Range"], "bytes=3000-")
        self.assertEqual(self.saved(), self.DATA)

    def test_a_corrupt_download_is_deleted(self):
        with self.assertRaises(wd.DumpError):
            self.download(FakeSite({self.URL: ranged(self.DATA)}), sha1="0" * 40)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_a_complete_file_is_not_downloaded_again(self):
        self.write("file.bz2", self.DATA)
        site = FakeSite()
        self.assertEqual(self.download(site), "exists")
        self.assertEqual(site.requests, [])

    def test_without_a_known_size_a_complete_part_is_recognised(self):
        self.write("file.bz2.part", self.DATA)
        self.assertEqual(self.download(FakeSite({self.URL: ranged(self.DATA)}), size=None), "saved")
        self.assertEqual(self.saved(), self.DATA)


class TestDumpCommand(TempCase):
    def setUp(self):
        super().setUp()
        self.data = bz2.compress(dump_xml())
        self.name = f"{WIKI}-{NEW}-pages-articles.xml.bz2"
        self.site = FakeSite(
            {
                f"{BASE}/": index_html([NEW]),
                f"{BASE}/{NEW}/dumpstatus.json": status_json({"articlesdump": ("done", {self.name: self.data})}),
                f"{BASE}/{NEW}/{self.name}": ranged(self.data),
            }
        )

    def dump(self, *argv):
        return run("dump", "--wiki", WIKI, "-o", self.path("dumps"), "--delay", "0", "-q", *argv, opener=self.site)

    def test_list_shows_the_files_and_downloads_nothing(self):
        code, out, _ = self.dump("--list")
        self.assertEqual(code, 0)
        self.assertIn(f"{WIKI} {NEW}, articles: 1 file", out)
        self.assertIn(self.name, out)
        self.assertEqual(self.site.asked_for(".bz2"), [])

    def test_a_dump_is_downloaded_verified_and_not_downloaded_twice(self):
        code, out, err = self.dump()
        self.assertEqual(code, 0, err)
        self.assertIn(f"saved   {self.name} (sha1 verified)", out)
        with open(self.path("dumps", self.name), "rb") as fh:
            self.assertEqual(fh.read(), self.data)
        code, out, _ = self.dump()
        self.assertIn(f"exists  {self.name}", out)
        self.assertEqual(len(self.site.asked_for(".bz2")), 1)

    def test_it_refuses_to_fill_the_disk_unless_forced(self):
        with mock.patch.object(wd.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10)):
            code, _, err = self.dump()
            self.assertEqual(code, 1)
            self.assertIn("free", err)
            self.assertEqual(self.dump("--force")[0], 0)

    def test_the_language_names_the_wiki(self):
        self.assertEqual((wd.wiki_name("en"), wd.wiki_name("zh-yue"), wd.wiki_name("simple")), ("enwiki", "zh_yuewiki", "simplewiki"))
        self.assertEqual(self.dump("--date", "2026")[0], 2)


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------

ARTICLES = {
    "Energy": (1, "Energy is the capacity to do work.\nIt is conserved.\n\n\n== History ==\nThe word is Greek.\n\n\n== See also ==\nPower\n"),
    "Force": (2, "A force is a push or a pull.\n"),
    "Momentum": (3, "Momentum is mass times velocity.\n"),
    "Stub": (4, "Tiny.\n"),
    "Paris": (5, "Paris is the capital of France.\n"),
}
CATEGORIES = {
    "Category:Physics": ["Energy", "Category:Mechanics", "Force", "Stub"],
    "Category:Mechanics": ["Momentum", "Force", "Category:Physics"],  # a loop back up
}
REDIRECTS = {"Paris, France": "Paris"}


class FakeWiki:
    """A tiny MediaWiki API over the articles above: categories, search, random, titles, TextExtracts.

    Lists come ``per_batch`` at a time, so continuation is exercised, and the
    first ``lagged`` calls answer "maxlag", as a busy Wikipedia does. It works
    as an opener for the fetcher and as the pages of a LocalServer.
    """

    ENDPOINT = "https://en.wikipedia.org/w/api.php"

    def __init__(self, lagged=0, per_batch=2):
        self.lagged = lagged
        self.per_batch = per_batch
        self.urls, self.calls, self.headers = [], [], []
        self.ids = {title: pageid for title, (pageid, _) in ARTICLES.items()}

    def open(self, request, timeout=None):
        self.headers.append(dict(request.header_items()))
        body, headers = self.answer(request.full_url)
        return urllib.response.addinfourl(io.BytesIO(body), headers, request.full_url, 200)

    def get(self, path):
        return self.answer(path)[0] if path.startswith("/w/api.php") else None

    def extract_calls(self):
        return [call for call in self.calls if call.get("prop") == "extracts|info"]

    def answer(self, url):
        self.urls.append(url)
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.calls.append(q)
        if self.lagged:
            self.lagged -= 1
            return self.json({"error": {"code": "maxlag", "info": "Waiting for a database server: 7 seconds lagged."}}), {"Retry-After": "3"}
        if q.get("list") == "categorymembers":
            rows = [
                {"pageid": 100 + i, "ns": 14, "title": t, "type": "subcat"} if t.startswith("Category:") else {"pageid": self.ids[t], "ns": 0, "title": t, "type": "page"}
                for i, t in enumerate(CATEGORIES.get(q["cmtitle"], []))
            ]
            return self.batch(rows, "categorymembers", "cmcontinue", q), {}
        if q.get("list") == "search":
            wanted = q["srsearch"].lower().split()
            rows = [{"ns": 0, "title": t, "pageid": p} for t, (p, text) in ARTICLES.items() if all(w in f"{t} {text}".lower() for w in wanted)]
            return self.batch(rows, "search", "sroffset", q), {}
        if q.get("list") == "random":
            rows = [{"id": p, "ns": 0, "title": t} for t, (p, _) in ARTICLES.items()][: int(q["rnlimit"])]
            return self.json({"batchcomplete": True, "continue": {"rncontinue": "0|1", "continue": "-||"}, "query": {"random": rows}}), {}
        if "titles" in q:
            pages = []
            for title in q["titles"].split("|"):
                target = REDIRECTS.get(title, title)
                pages.append({"pageid": self.ids[target], "ns": 0, "title": target} if target in self.ids else {"ns": 0, "title": title, "missing": True})
            return self.json({"batchcomplete": True, "query": {"pages": pages}}), {}
        if q.get("prop") == "extracts|info":
            titles = {p: t for t, p in self.ids.items()}
            pages = []
            for pageid in map(int, q["pageids"].split("|")):
                text = ARTICLES[titles[pageid]][1]
                if q.get("exintro"):
                    text = text.split("\n")[0]
                url = f"https://en.wikipedia.org/wiki/{titles[pageid]}"
                pages.append({"pageid": pageid, "ns": 0, "title": titles[pageid], "extract": text, "fullurl": url, "lastrevid": 1000 + pageid})
            return self.json({"batchcomplete": True, "query": {"pages": pages}}), {}
        return self.json({"error": {"code": "badvalue", "info": "Unrecognized value for parameter."}}), {}

    def batch(self, rows, key, cursor, q):
        start = int(q.get(cursor, 0))
        data = {"batchcomplete": True, "query": {key: rows[start : start + self.per_batch]}}
        if start + self.per_batch < len(rows):
            data["continue"] = {cursor: str(start + self.per_batch), "continue": "-||"}
        return self.json(data)

    @staticmethod
    def json(data):
        return json.dumps(data).encode()


class TestFetch(TempCase):
    def fetch(self, *argv, wiki=None, api=True):
        self.wiki = wiki or FakeWiki()
        self.clock = FakeClock()
        extra = ["--api", FakeWiki.ENDPOINT] if api else []
        return run("fetch", *argv, "-o", self.path("out"), "--delay", "0", *extra, opener=self.wiki, sleep=self.clock.sleep)

    def saved(self):
        return sorted(name for name in os.listdir(self.path("out")) if name.endswith(".txt"))

    def read(self, name):
        with open(self.path("out", name), encoding="utf-8") as fh:
            return fh.read()

    def test_a_category_and_its_subcategories(self):
        code, out, err = self.fetch("-c", "Physics", "--depth", "1")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.saved(), ["1-energy.txt", "2-force.txt", "3-momentum.txt", "4-stub.txt"])
        self.assertEqual(self.read("1-energy.txt"), "Energy\n\nEnergy is the capacity to do work.\nIt is conserved.\n\nHistory\n\nThe word is Greek.\n")
        self.assertIn("4 saved", out)
        self.assertEqual(self.fetch("-c", "Category:Physics")[0], 0)  # the prefix is optional

    def test_depth_zero_is_the_category_alone(self):
        self.fetch("-c", "Physics")
        self.assertEqual(self.saved(), ["1-energy.txt", "2-force.txt", "4-stub.txt"])

    def test_short_articles_do_not_count_towards_the_limit(self):
        code, out, _ = self.fetch("-c", "Physics", "--depth", "1", "--min-chars", "10", "-n", "3")
        self.assertEqual(code, 0)
        self.assertEqual(self.saved(), ["1-energy.txt", "2-force.txt", "3-momentum.txt"])
        self.assertIn("3 saved, 1 too short", out)

    def test_the_limit_stops_the_listing_too(self):
        self.fetch("-c", "Physics", "--depth", "1", "-n", "1")
        self.assertEqual(self.saved(), ["1-energy.txt"])
        self.assertEqual(len(self.wiki.extract_calls()), 1)

    def test_a_second_run_fetches_no_text(self):
        self.fetch("-c", "Physics")
        code, out, _ = self.fetch("-c", "Physics")
        self.assertEqual((code, self.wiki.extract_calls()), (0, []))
        self.assertIn("0 saved, 3 already there", out)

    def test_titles_follow_redirects_and_report_what_is_missing(self):
        code, _, err = self.fetch("-t", "Paris, France", "-t", "Nowhere")
        self.assertEqual(code, 0)
        self.assertEqual(self.saved(), ["5-paris.txt"])
        self.assertIn("no article called 'Nowhere'", err)

    def test_search_and_random(self):
        self.fetch("-s", "capital france")
        self.assertEqual(self.saved(), ["5-paris.txt"])
        code, out, _ = self.fetch("--random", "2")
        self.assertEqual(code, 0)
        self.assertIn("2 saved", out)
        self.assertEqual(len(self.saved()), 3)

    def test_the_lead_section_only_twenty_to_a_request(self):
        self.fetch("-c", "Physics", "--depth", "1", "--intro")
        self.assertEqual(len(self.wiki.extract_calls()), 1)
        self.assertEqual(self.read("1-energy.txt"), "Energy\n\nEnergy is the capacity to do work.\n")

    def test_json_lines(self):
        code, _, err = self.fetch("-c", "Physics", "--format", "jsonl")
        self.assertEqual(code, 0, err)
        with open(self.path("out", wd.FETCHED_JSONL), encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        self.assertEqual([(r["id"], r["title"], r["revision"]) for r in records], [(1, "Energy", 1001), (2, "Force", 1002), (4, "Stub", 1004)])
        with open(self.path("out", wd.MANIFEST), encoding="utf-8") as fh:
            self.assertEqual({json.loads(line)["file"] for line in fh}, {wd.FETCHED_JSONL})

    def test_a_busy_wiki_is_waited_for(self):
        code, _, err = self.fetch("-t", "Force", wiki=FakeWiki(lagged=2))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.clock.sleeps, [3, 3])
        self.assertIn("busy", err)

    def test_who_is_asking_goes_in_the_user_agent(self):
        _, _, err = self.fetch("-t", "Force", "--contact", "me@example.org")
        self.assertIn("(me@example.org)", self.wiki.headers[0]["User-agent"])
        self.assertNotIn("contact details", err)
        _, _, err = self.fetch("-t", "Force", "--contact", "")
        self.assertIn("contact details", err)

    def test_the_language_picks_the_wiki(self):
        self.fetch("-t", "Force", "--lang", "de", api=False)
        self.assertTrue(self.wiki.urls[0].startswith("https://de.wikipedia.org/w/api.php?"))
        self.assertIn("maxlag=5", self.wiki.urls[0])

    def test_it_needs_to_be_told_which_articles(self):
        self.assertEqual(self.fetch()[0], 2)

    def test_something_that_is_not_an_api(self):
        html_page = types.SimpleNamespace(
            open=lambda request, timeout=None: urllib.response.addinfourl(io.BytesIO(b"<html>Hello</html>"), {}, request.full_url, 200)
        )
        code, _, err = self.fetch("-t", "Force", wiki=html_page)
        self.assertEqual(code, 1)
        self.assertIn("did not answer with JSON", err)
        code, _, err = self.fetch("-t", "Force", wiki=FakeSite())  # every URL a 404
        self.assertEqual(code, 1)
        self.assertIn("HTTP 404", err)


# ---------------------------------------------------------------------------
# through real sockets
# ---------------------------------------------------------------------------


class TestOverRealHttp(TempCase):
    def test_a_dump_resumed_from_a_mirror_then_extracted(self):
        data = bz2.compress(dump_xml())
        name = f"{WIKI}-{NEW}-pages-articles.xml.bz2"
        pages = {
            f"/{WIKI}/": index_html([NEW]),
            f"/{WIKI}/{NEW}/dumpstatus.json": status_json({"articlesdump": ("done", {name: data})}),
            f"/{WIKI}/{NEW}/{name}": data,
        }
        os.makedirs(self.path("dumps"))
        self.write(os.path.join("dumps", name + ".part"), data[:100])
        with LocalServer(pages) as server:
            code, out, err = run("dump", "--wiki", WIKI, "--mirror", server.url, "-o", self.path("dumps"), "--delay", "0", opener=no_proxy_opener())
        self.assertEqual(code, 0, err)
        self.assertIn("sha1 verified", out)
        self.assertIn("extract", err)  # the hint for what to do next
        code, out, err = run("extract", self.path("dumps", name), "-o", self.path("text"), "-p", "1", "-q")
        self.assertEqual(code, 0, err)
        self.assertIn("3 articles from 7 pages", out)

    def test_fetching_from_an_api(self):
        with LocalServer(FakeWiki()) as server:
            code, out, err = run(
                "fetch", "-c", "Physics", "--api", f"{server.url}/w/api.php", "-o", self.path("out"), "--delay", "0", "-q",
                opener=no_proxy_opener(),
            )  # fmt: skip
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(os.listdir(self.path("out"))), ["1-energy.txt", "2-force.txt", "4-stub.txt", wd.MANIFEST])


if __name__ == "__main__":
    unittest.main()
