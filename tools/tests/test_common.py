"""Tests for common.py - the fetcher, the manifest and the text helpers both tools share.

Run: python3 -m unittest discover -s tools/tests  (from the repository root)
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import Fatal, FetchError, Fetcher, Manifest, NotFound, clip, fold, human_bytes, slugify, words  # noqa: E402
from fakes import FakeClock, FakeSite, http_error  # noqa: E402

URL = "https://example.org/a"


def fetcher(site, **options):
    clock = FakeClock()
    options.setdefault("delay", 0)
    return Fetcher("test-agent/1.0", opener=site, sleep=clock.sleep, clock=clock, **options), clock


class TestFetcher(unittest.TestCase):
    def test_requests_are_spaced_by_the_delay(self):
        """The pause is between requests, not after each one: the first goes at once."""
        site = FakeSite({URL: b"a", URL + "2": b"b"})
        f, clock = fetcher(site, delay=2.0)
        f.get(URL)
        f.get(URL + "2")
        self.assertEqual(clock.sleeps, [2.0])

    def test_server_errors_are_retried_with_growing_pauses(self):
        site = FakeSite({URL: [http_error(URL, 503), http_error(URL, 503), b"ok"]})
        f, clock = fetcher(site, delay=1.0, retries=3)
        self.assertEqual(f.get(URL), b"ok")
        self.assertEqual(len(site.requests), 3)
        self.assertEqual(clock.sleeps, [1.0, 2.0])

    def test_retry_after_is_honoured(self):
        site = FakeSite({URL: [http_error(URL, 429, {"Retry-After": "7"}), b"ok"]})
        f, clock = fetcher(site, retries=1)
        self.assertEqual(f.get(URL), b"ok")
        self.assertEqual(clock.sleeps, [7.0])

    def test_it_gives_up_after_its_retries(self):
        site = FakeSite({URL: http_error(URL, 500)})
        f, _ = fetcher(site, retries=2)
        with self.assertRaises(FetchError) as caught:
            f.get(URL)
        self.assertEqual(caught.exception.status, 500)
        self.assertEqual(len(site.requests), 3)

    def test_a_missing_file_is_not_retried(self):
        site = FakeSite()
        f, clock = fetcher(site, retries=5)
        with self.assertRaises(NotFound):
            f.get(URL)
        self.assertEqual((len(site.requests), clock.sleeps), (1, []))

    def test_a_refusal_is_fatal_and_not_retried(self):
        """Retrying a 403 is how an address gets banned."""
        site = FakeSite({URL: http_error(URL, 403)})
        f, _ = fetcher(site, retries=5)
        with self.assertRaises(Fatal):
            f.get(URL)
        self.assertEqual(len(site.requests), 1)

    def test_a_proxy_refusing_the_tunnel_is_fatal(self):
        refused = urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))
        site = FakeSite({URL: refused})
        f, _ = fetcher(site, retries=5)
        with self.assertRaises(Fatal) as caught:
            f.get(URL)
        self.assertIn("example.org", str(caught.exception))
        self.assertEqual(len(site.requests), 1)

    def test_dropped_connections_are_retried(self):
        site = FakeSite({URL: [ConnectionResetError("reset by peer"), b"ok"]})
        f, _ = fetcher(site, retries=1)
        self.assertEqual(f.get(URL), b"ok")

    def test_gzip_on_the_wire_is_undone_and_asked_for(self):
        site = FakeSite({URL: (gzip.compress(b"hello"), {"Content-Encoding": "gzip"})})
        f, _ = fetcher(site)
        self.assertEqual(f.get(URL), b"hello")
        self.assertEqual(site.headers[0]["Accept-encoding"], "gzip")
        self.assertEqual(site.headers[0]["User-agent"], "test-agent/1.0")

    def test_a_malformed_url_fails_without_a_retry(self):
        f, clock = fetcher(FakeSite(), retries=3)
        with self.assertRaises(FetchError):
            f.get("not a url")
        self.assertEqual(clock.sleeps, [])


class TestManifest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "manifest.jsonl")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_latest_entry_wins_and_compacting_keeps_one_per_key(self):
        manifest = Manifest(self.path)
        manifest.add({"id": 2, "v": "old"})
        manifest.add({"id": 1, "v": "one"})
        manifest.add({"id": 2, "v": "new"})
        self.assertEqual(Manifest(self.path).get(2)["v"], "new")  # before compacting, too
        manifest.compact()
        with open(self.path, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh]
        self.assertEqual(lines, [{"id": 1, "v": "one"}, {"id": 2, "v": "new"}])

    def test_a_torn_last_line_is_skipped_and_nothing_is_glued_to_it(self):
        """A run killed mid-write leaves half a line; the next run must not corrupt its own."""
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write('{"id": 1}\n{"id": 2, "tit')
        manifest = Manifest(self.path)
        self.assertEqual(sorted(manifest.entries), [1])
        manifest.add({"id": 3})
        self.assertEqual(sorted(Manifest(self.path).entries), [1, 3])


class TestText(unittest.TestCase):
    def test_folding_ignores_case_and_accents(self):
        self.assertEqual(fold("BRONTË"), fold("bronte"))
        self.assertEqual(words("Austen, Jane, 1775-1817"), ("austen", "jane", "1775", "1817"))

    def test_slugs_are_safe_file_names(self):
        self.assertEqual(slugify("Frankenstein; Or, The Modern Prometheus"), "frankenstein-or-the-modern-prometheus")
        self.assertEqual(slugify("Alice's Adventures in Wonderland"), "alices-adventures-in-wonderland")
        self.assertEqual(slugify("Ὀδύσσεια"), "οδυσσεια")
        self.assertEqual(slugify("../../etc/passwd"), "etc-passwd")

    def test_a_long_slug_is_cut_at_a_word(self):
        slug = slugify("word " * 30, limit=22)
        self.assertEqual(slug, "word-word-word-word")

    def test_clip_and_sizes(self):
        self.assertEqual(clip("abcdef", 4), "abc…")
        self.assertEqual(clip("abc", 4), "abc")
        self.assertEqual(human_bytes(999), "999 B")
        self.assertEqual(human_bytes(1_234_567), "1.2 MB")


if __name__ == "__main__":
    unittest.main()
