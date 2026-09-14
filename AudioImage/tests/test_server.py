"""Tests for audioimage.server - the HTTP API behind the browser page."""

import base64
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage.png import read_png_bytes  # noqa: E402
from audioimage.server import MAX_SECONDS, Params, ServerError, frontend_dir, make_server  # noqa: E402
from audioimage.wav import read_wav_bytes  # noqa: E402

FAST = "n_fft=256&hop=64"


class ServerCase(unittest.TestCase):
    """One server for the whole class, on a port the OS picks."""

    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0, quiet=True)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)

    def post(self, path, body=b""):
        request = urllib.request.Request(self.base + path, data=body, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read(), dict(response.headers)

    def failure(self, path, body=b"", method="POST"):
        """Return (status, parsed error) for a request expected to fail."""
        request = urllib.request.Request(self.base + path, data=body if method == "POST" else None, method=method)
        try:
            urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())
        self.fail(f"{path} was expected to fail")

    def meta(self, headers):
        return json.loads(base64.b64decode(headers["X-AudioImage-Meta"]))

    def tone(self, query="kind=sine&duration=0.4&rate=16000"):
        return self.post("/api/tone?" + query)[1]


class TestParams(unittest.TestCase):
    def test_text_and_defaults(self):
        p = Params("a=1&b=&c=hi")
        self.assertEqual(p.text("a"), "1")
        self.assertEqual(p.text("b", "fallback"), "fallback")
        self.assertEqual(p.text("missing", "d"), "d")

    def test_numbers(self):
        p = Params("n=3.5&i=7")
        self.assertEqual(p.number("n", 0.0), 3.5)
        self.assertEqual(p.integer("i", 0), 7)
        self.assertEqual(p.number("missing", 1.25), 1.25)

    def test_ranges_and_rubbish(self):
        p = Params("n=99&bad=abc&inf=inf")
        for call in (
            lambda: p.number("n", 0.0, high=10.0),
            lambda: p.number("n", 0.0, low=200.0),
            lambda: p.number("bad", 0.0),
            lambda: p.number("inf", 0.0),
            lambda: p.choice("n", ("a", "b"), "a"),
        ):
            with self.assertRaises(ServerError):
                call()

    def test_flags(self):
        self.assertTrue(Params("x=1").flag("x"))
        self.assertTrue(Params("x=true").flag("x"))
        self.assertFalse(Params("x=0").flag("x", True))
        self.assertTrue(Params("").flag("x", True))

    def test_optional(self):
        self.assertIsNone(Params("").optional_int("h"))
        self.assertEqual(Params("h=64").optional_int("h"), 64)


class TestStatic(ServerCase):
    def test_index(self):
        status, body, headers = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"AudioImage", body)
        self.assertTrue(headers["Content-Type"].startswith("text/html"))

    def test_assets(self):
        for path, kind in (("/app.js", "text/javascript"), ("/style.css", "text/css")):
            with self.subTest(path=path):
                status, body, headers = self.get(path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith(kind))
                self.assertGreater(len(body), 100)

    def test_security_headers(self):
        _, _, headers = self.get("/")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["X-Frame-Options"], "DENY")

    def test_missing_file(self):
        status, payload = self.failure("/nope.js", method="GET")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_directory_traversal_is_refused(self):
        """A path that climbs out of the front-end directory must not be served."""
        for path in ("/../cli.py", "/%2e%2e/cli.py", "/../../pyproject.toml", "/..%2f..%2fsetup.py"):
            with self.subTest(path=path):
                status, _ = self.failure(path, method="GET")
                self.assertEqual(status, 404)

    def test_frontend_dir_default(self):
        self.assertTrue(os.path.isfile(os.path.join(frontend_dir(), "index.html")))


class TestInfo(ServerCase):
    def test_shape(self):
        status, body, _ = self.get("/api/info")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["tool"], "audioimage")
        for key in ("colormaps", "scales", "freq_scales", "windows", "kinds", "depths"):
            self.assertIn(key, payload["options"])
        for key in ("n_fft", "hop", "window", "scale", "iters"):
            self.assertIn(key, payload["defaults"])
        self.assertIn(payload["backend"]["selected"], ("python", "numpy"))

    def test_unknown_route(self):
        self.assertEqual(self.failure("/api/nope", method="GET")[0], 404)


class TestTone(ServerCase):
    def test_makes_a_wav(self):
        status, body, headers = self.post("/api/tone?kind=sine&duration=0.5&rate=16000&freq=440")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        audio = read_wav_bytes(body)
        self.assertEqual(audio.sample_rate, 16000)
        self.assertAlmostEqual(audio.duration, 0.5, places=3)
        self.assertAlmostEqual(self.meta(headers)["duration"], 0.5, places=3)

    def test_every_kind(self):
        for kind in ("sine", "chirp", "noise", "chord", "melody", "harmonics"):
            with self.subTest(kind=kind):
                body = self.post(f"/api/tone?kind={kind}&duration=0.2&rate=8000")[1]
                self.assertGreater(len(read_wav_bytes(body).samples), 0)

    def test_notes(self):
        body = self.post("/api/tone?kind=chord&notes=C4%20E4%20G4&duration=0.3&rate=8000")[1]
        self.assertGreater(len(read_wav_bytes(body).samples), 0)

    def test_rejections(self):
        for query, why in (
            ("kind=didgeridoo", "unknown kind"),
            ("duration=0", "no duration"),
            (f"duration={MAX_SECONDS + 10}", "too long"),
            ("rate=1", "rate too low"),
        ):
            with self.subTest(why=why):
                self.assertEqual(self.failure("/api/tone?" + query)[0], 400)


class TestEncodeDecode(ServerCase):
    def test_encode(self):
        status, png, headers = self.post(f"/api/encode?{FAST}", self.tone())
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        image = read_png_bytes(png)
        self.assertEqual(image.height, 129)
        report = self.meta(headers)
        self.assertEqual(report["meta"]["n_fft"], 256)
        self.assertEqual(report["meta"]["sample_rate"], 16000)

    def test_decode(self):
        png = self.post(f"/api/encode?{FAST}", self.tone())[1]
        status, wav, headers = self.post("/api/decode?iters=6", png)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        audio = read_wav_bytes(wav)
        self.assertEqual(audio.sample_rate, 16000)
        self.assertTrue(self.meta(headers)["had_metadata"])

    def test_decode_a_picture_with_no_settings(self):
        """What the page sends after someone paints on the plane."""
        from audioimage.png import Image, write_png_bytes

        image = Image(24, 129, "L", 8)
        image.fill(255)
        for x in range(24):
            image.set(x, 64, 0)
        status, wav, headers = self.post("/api/decode?rate=8000&n_fft=256&iters=4", write_png_bytes(image))
        self.assertEqual(status, 200)
        self.assertFalse(self.meta(headers)["had_metadata"])
        self.assertEqual(read_wav_bytes(wav).sample_rate, 8000)

    def test_settings_are_honoured(self):
        png = self.post(f"/api/encode?{FAST}&height=64&width=40&depth=8", self.tone())[1]
        image = read_png_bytes(png)
        self.assertEqual((image.width, image.height, image.depth), (40, 64, 8))

    def test_colormaps(self):
        for name in ("gray", "fire", "viridis"):
            with self.subTest(name=name):
                png = self.post(f"/api/encode?{FAST}&colormap={name}", self.tone())[1]
                self.assertEqual(read_png_bytes(png).mode, "L" if name == "gray" else "RGB")

    def test_roundtrip(self):
        status, body, _ = self.post(f"/api/roundtrip?{FAST}&iters=20", self.tone())
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertLess(payload["metrics"]["spectral_convergence"], 0.5)
        self.assertTrue(payload["image"].startswith("data:image/png;base64,"))
        self.assertTrue(payload["audio"].startswith("data:audio/wav;base64,"))

    def test_view_and_waveform(self):
        wav = self.tone()
        for route in (f"/api/view?{FAST}", f"/api/waveform?{FAST}"):
            with self.subTest(route=route):
                status, png, headers = self.post(route, wav)
                self.assertEqual(status, 200)
                self.assertEqual(read_png_bytes(png).mode, "RGB")

    def test_view_accepts_a_picture_too(self):
        png = self.post(f"/api/encode?{FAST}", self.tone())[1]
        self.assertEqual(read_png_bytes(self.post("/api/view", png)[1]).mode, "RGB")


class TestErrors(ServerCase):
    def test_empty_body(self):
        status, payload = self.failure(f"/api/encode?{FAST}")
        self.assertEqual(status, 400)
        self.assertIn("no audio", payload["error"])

    def test_rubbish_body(self):
        self.assertEqual(self.failure(f"/api/encode?{FAST}", b"not a wav at all")[0], 400)
        self.assertEqual(self.failure("/api/decode", b"not a png at all")[0], 400)

    def test_a_wav_sent_to_decode(self):
        status, payload = self.failure("/api/decode", self.tone())
        self.assertEqual(status, 400)
        self.assertIn("PNG", payload["error"])

    def test_bad_settings(self):
        wav = self.tone()
        for query, expect in (
            ("n_fft=1000", "power of two"),
            ("n_fft=256&hop=0", "hop"),
            ("scale=loud", "scale"),
            ("colormap=rainbow", "colormap"),
            ("freq_scale=bark", "freq_scale"),
            ("n_fft=256&top_db=0", "top_db"),
        ):
            with self.subTest(query=query):
                status, payload = self.failure("/api/encode?" + query, wav)
                self.assertEqual(status, 400)
                self.assertIn(expect, payload["error"])

    def test_unknown_post_route(self):
        self.assertEqual(self.failure("/api/nope", b"x")[0], 404)

    def test_errors_are_json(self):
        _, payload = self.failure(f"/api/encode?{FAST}")
        self.assertIsInstance(payload.get("error"), str)


if __name__ == "__main__":
    unittest.main()
