"""Images and speech as text, and the recall tutor over both: the Rust port against Python (and Go).

Skipped without a Rust toolchain.  What ``tests/test_go_parity.py`` asks of Go
(``TestGoMediaParity``) is asked of Rust here, and more of it:

* **speech** - the same recording encodes to the same waveform text and the
  same utterance token (a digest of that text, and *in the text the model
  trains on*), whatever the sample format, channel count, rate, codec or
  normalising; the same text decodes to the same WAV file, byte for byte;
* **images** - Python's thumbnail goes through Pillow's Lanczos filter and
  Go's through a box filter, so the pixel reduction is the one place the texts
  of the two differ.  Rust is Go's twin there: when a Go toolchain is on PATH
  the same picture must give Go's text exactly (PNG of every colour type, and
  the JPEGs and GIFs of Go's own test data).  The *format* is shared by all
  three, which Python checks without Pillow;
* **the recall tutor** - the same quiz on the same model file marks the same:
  the whole document, every lesson, grade and fact, and what ``--blame``
  teaches the negative network;
* **the routes** - the JSON the Images and Speech tabs read, from multipart,
  raw and JSON bodies alike, and ``POST /api/uploads`` taking all three.
"""

from __future__ import annotations

import array
import base64
import glob
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import unittest
import zlib

from tests.rust_harness import ROOT, build, py, rust, serve, tmpdir

sys.path.insert(0, ROOT)

from radixnet import recall, speech, vision  # noqa: E402
from radixnet.model import load_model  # noqa: E402

RATE = 8000
GO = shutil.which("go")


def setUpModule():
    build()  # raises SkipTest without a Rust toolchain


def tone(seconds=0.05, rate=RATE, freq=220.0, amplitude=0.6, channels=1):
    """A sine as Python's ``array("f")`` holds it, interleaved over ``channels``."""
    return array.array("f", [
        amplitude * math.sin(2 * math.pi * freq * (i // channels) / rate) * (1 if i % channels == 0 else 0.5)
        for i in range(int(rate * seconds) * channels)
    ])


def wav_file(path, samples, rate=RATE, channels=1, fmt=1, bits=16):
    """A WAV of any sample format, written by hand so the readers are what is tested."""
    if fmt == 3:
        body = (array.array("f", samples) if bits == 32 else array.array("d", samples)).tobytes()
    elif bits == 8:
        body = bytes(max(0, min(255, round(v * 127 + 128))) for v in samples)
    elif bits == 24:
        body = b"".join(max(-8388608, min(8388607, round(v * 8388607))).to_bytes(3, "little", signed=True)
                        for v in samples)
    elif bits == 32:
        body = array.array("i", [max(-2**31, min(2**31 - 1, round(v * 2**31))) for v in samples]).tobytes()
    else:
        body = array.array("h", [max(-32768, min(32767, round(v * 32767))) for v in samples]).tobytes()
    width = bits // 8
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(body), b"WAVE", b"fmt ", 16, fmt, channels, rate,
                         rate * channels * width, channels * width, bits, b"data", len(body))
    with open(path, "wb") as fh:
        fh.write(header + body)
    return path


def png_bytes(width, height, color_type, depth, pixels, palette=None, trns=None, interlace=False):
    """A PNG of any colour type, built by hand (``pixels(x, y)`` gives the samples of one pixel)."""
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    def row_bytes(xs, y):
        bits = []
        for x in xs:
            bits.extend(pixels(x, y))
        if depth == 16:
            return b"".join(struct.pack(">H", v) for v in bits)
        if depth == 8:
            return bytes(bits)
        per_byte, out, acc, n = 8 // depth, bytearray(), 0, 0
        for v in bits:
            acc = (acc << depth) | v
            n += 1
            if n == per_byte:
                out.append(acc)
                acc, n = 0, 0
        if n:
            out.append(acc << (depth * (per_byte - n)))
        return bytes(out)

    raw = bytearray()
    passes = [(8, 8, 0, 0), (8, 8, 4, 0), (4, 8, 0, 4), (4, 4, 2, 0), (2, 4, 0, 2), (2, 2, 1, 0), (1, 2, 0, 1)] \
        if interlace else [(1, 1, 0, 0)]
    for xs, ys, xo, yo in passes:
        columns = list(range(xo, width, xs))
        if not columns:
            continue
        for y in range(yo, height, ys):
            raw += b"\x00" + row_bytes(columns, y)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0,
                                                              1 if interlace else 0))
    if palette is not None:
        out += chunk(b"PLTE", bytes(v for rgb in palette for v in rgb))
    if trns is not None:
        out += chunk(b"tRNS", trns)
    return out + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")


def gradient_png(width=64, height=64):
    """The Go parity suite's picture: an RGB gradient."""
    return png_bytes(width, height, 2, 8, lambda x, y: ((x * 255) // width, (y * 255) // height, 128))


def read_png(data):
    """``(width, height, rgb rows)`` of an 8-bit RGB, non-interlaced PNG - enough to read what Rust draws."""
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    at, idat, width, height = 8, b"", 0, 0
    while at < len(data):
        length, tag = struct.unpack(">I4s", data[at:at + 8])
        body = data[at + 8:at + 8 + length]
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", body[:10])
            assert (depth, color_type) == (8, 2), (depth, color_type)
        elif tag == b"IDAT":
            idat += body
        at += 12 + length
    raw, stride, rows, prev = zlib.decompress(idat), width * 3, [], bytes(width * 3)
    for y in range(height):
        kind, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            left = line[i - 3] if i >= 3 else 0
            line[i] = (line[i] + (left if kind == 1 else prev[i] if kind == 2 else 0)) & 0xFF
        rows.append(bytes(line))
        prev = bytes(line)
    return width, height, rows


def fresh(name):
    """A model path in the scratch directory, with no file (nor negative file) behind it yet."""
    path = os.path.join(tmpdir(), name)
    for stale in (path, path.replace(".count.json", ".count.negative.json")):
        if os.path.exists(stale):
            os.remove(stale)
    return path


def without_seconds(doc):
    """A tutor document with the timings taken out: everything else must agree."""
    doc = json.loads(json.dumps(doc))
    for lesson in doc["lessons"]:
        lesson.pop("seconds")
    return doc


class TestRustSpeechParity(unittest.TestCase):
    """The waveform text and the token are the same bytes on both sides, for every recording."""

    def setUp(self):
        self.model = fresh("speech.count.json")
        self.wav = wav_file(os.path.join(tmpdir(), "probe.wav"), tone())

    def test_both_sides_encode_the_same_utterance_identically(self):
        options = ("speech", "teach", self.wav, "--text", "the cat sat on the mat", "--rate", RATE)
        a = py(*options, model=self.model)
        b = rust(*options, model=self.model)
        self.assertEqual(a["token"], b["token"], "the utterance token must match: it is in the trained text")
        self.assertEqual(a["texts"], b["texts"])
        self.assertEqual(a["audio"], b["audio"])
        self.assertEqual((a["chars"], a["pair"], a["asr"]), (b["chars"], b["pair"], b["asr"]))

    def test_every_sample_format_encodes_the_same(self):
        stereo = tone(0.1, 44100, 330.0, 0.4, channels=2)
        recordings = {
            "pcm8": wav_file(os.path.join(tmpdir(), "p8.wav"), tone(), bits=8),
            "pcm24-stereo-44k": wav_file(os.path.join(tmpdir(), "p24.wav"), stereo, 44100, 2, bits=24),
            "pcm32": wav_file(os.path.join(tmpdir(), "p32.wav"), tone(0.05, 11025), 11025, bits=32),
            "float32": wav_file(os.path.join(tmpdir(), "f32.wav"), tone(0.05, 22050, 500.0, 0.9), 22050, fmt=3,
                                bits=32),
            "float64": wav_file(os.path.join(tmpdir(), "f64.wav"), tone(0.05, 16000), 16000, fmt=3, bits=64),
        }
        for name, path in recordings.items():
            for extra in ([], ["--normalise"], ["--codec", "pcm8"], ["--rate", 16000], ["--pair", "--text", "hi"]):
                with self.subTest(recording=name, options=extra):
                    options = ("speech", "teach", path, *extra)
                    a, b = py(*options, model=self.model), rust(*options, model=self.model)
                    self.assertEqual((a["token"], a["texts"], a["audio"]), (b["token"], b["texts"], b["audio"]))

    def test_each_side_decodes_the_other_s_waveform(self):
        text = py("speech", "teach", self.wav, "--rate", RATE, model=self.model)["texts"][0]
        for codec, cut in (("", 0), ("pcm8", 0), ("", 37)):
            with self.subTest(codec=codec, cut=cut):
                out_py = os.path.join(tmpdir(), "back_py.wav")
                out_rs = os.path.join(tmpdir(), "back_rs.wav")
                body = text[:-cut] if cut else text
                extra = ("--codec", codec) if codec else ()
                a = py("speech", "decode", "--text", body, *extra, "--out", out_py, model=self.model)
                b = rust("speech", "decode", "--text", body, *extra, "--out", out_rs, model=self.model)
                self.assertEqual({**a, "out": None}, {**b, "out": None})
                with open(out_py, "rb") as fa, open(out_rs, "rb") as fb:
                    self.assertEqual(fa.read(), fb.read(), "the decoded WAV must be identical")

    def test_a_given_transcript_is_transcribed_the_same(self):
        options = ("speech", "transcribe", self.wav, "--text", "  hello   there ")
        a, b = py(*options, model=self.model), rust(*options, model=self.model)
        self.assertEqual((a["transcript"], a["backend"], a["model"]), (b["transcript"], b["backend"], b["model"]))

    def test_the_description_has_every_key_python_reports(self):
        a = py("speech", "info", model=self.model)
        b = rust("speech", "info", model=self.model)
        self.assertLessEqual(set(a), set(b))
        for key in ("codecs", "default_codec", "default_rate", "token", "token_example", "text_format"):
            self.assertEqual(a[key], b[key], key)


class TestRustRecallParity(unittest.TestCase):
    """The same quiz on the same model file marks the same, and blames the same."""

    def setUp(self):
        self.wav = wav_file(os.path.join(tmpdir(), "recall.wav"), tone())
        self.model = fresh("recall.count.json")
        py("--kind", "count", "--seed", 1, "speech", "teach", self.wav, "--text", "hello there", "--rate", RATE,
           "--train", "--epochs", 4, model=self.model)

    def test_both_recall_tutors_mark_the_same(self):
        for extra in ([], ["--length", 120], ["--attempts", 3, "--length", 60],
                      ["--lead", 8, "--length", 40, "--mode", "sample"], ["--threshold", 2.5]):
            with self.subTest(options=extra):
                options = ("speech", "tutor", self.wav, "--text", "hello there", "--rate", RATE, *extra)
                a = without_seconds(py(*options, model=self.model))
                b = without_seconds(rust(*options, model=self.model))
                self.assertEqual(a, b)

    def test_the_negative_network_learns_the_same_from_what_it_forgot(self):
        docs = {}
        for side, run in (("py", py), ("rs", rust)):
            model = os.path.join(tmpdir(), f"blame-{side}.count.json")
            shutil.copy(self.model, model)
            negative = model.replace(".count.json", ".count.negative.json")
            if os.path.exists(negative):
                os.remove(negative)
            docs[side] = run("speech", "tutor", self.wav, "--text", "hello there", "--rate", RATE, "--length", 120,
                             "--attempts", 2, "--blame", model=model)["negative"]
            self.assertTrue(os.path.exists(negative), side)
        self.assertEqual(docs["py"]["taught"], docs["rs"]["taught"])
        self.assertEqual(docs["py"]["reasons"], docs["rs"]["reasons"])

    def test_an_image_quiz_marks_the_same_as_python_s_quiz(self):
        """Python's image tutor needs Pillow to encode; the quiz over a text needs nothing."""
        payload = bytes((i * 7 + 11) % 256 for i in range(3 * 8 * 8))
        text = vision.pack_text("tiny", 64, 64, payload)
        data = os.path.join(tmpdir(), "picture.txt")
        with open(data, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        model = fresh("picture.count.json")
        py("--kind", "count", "--seed", 3, "train", "--data", data, "--epochs", 6, model=model)
        server = serve(self, model)
        for body in ({"texts": [text]}, {"texts": [text], "length": 40, "attempts": 2}, {"text": text, "lead": 4}):
            with self.subTest(body=body):
                status, doc, _ = server.post("/api/images/tutor", body)
                self.assertEqual(status, 200, doc)
                options = {"lead": body.get("lead"), "length": body.get("length", 0),
                           "attempts": body.get("attempts", 1)}
                lessons = recall.quiz(load_model(model), [text], labels=["text 1"], **options)
                expected = {"modality": "image", "lessons": [lesson.to_dict() for lesson in lessons],
                            "report": recall.report_card(lessons), "negative": None}
                self.assertEqual(without_seconds(doc), without_seconds(expected))


class TestRustImageParity(unittest.TestCase):
    """The image text format is shared with Python; the pixels are Go's."""

    def setUp(self):
        self.model = fresh("images.count.json")
        self.png = os.path.join(tmpdir(), "probe.png")
        with open(self.png, "wb") as fh:
            fh.write(gradient_png(64, 64))

    def test_both_sides_read_the_same_image_text(self):
        encoded = rust("image", "encode", self.png, "--size", 64, "--encoder", "tiny", model=self.model)
        self.assertEqual((encoded["encoder"], encoded["latent_shape"], encoded["source_size"]), ("tiny", [3, 8, 8],
                                                                                                 [64, 64]))
        name, width, height, payload, repaired = vision.parse_text(encoded["text"])
        self.assertEqual((name, width, height, repaired), ("tiny", 64, 64, False))
        self.assertEqual(len(payload), encoded["bytes"])
        self.assertEqual(vision.pack_text("tiny", 64, 64, payload), encoded["text"])

    def test_a_text_is_drawn_back_as_its_thumbnail(self):
        payload = bytes((i * 37) % 256 for i in range(3 * 4 * 4))
        text = vision.pack_text("tiny", 32, 32, payload)
        out = os.path.join(tmpdir(), "drawn.png")
        doc = rust("image", "decode", "--text", text, "--out", out, model=self.model)
        self.assertEqual({k: doc[k] for k in ("encoder", "width", "height", "bytes", "repaired")},
                         {"encoder": "tiny", "width": 32, "height": 32, "bytes": 48, "repaired": False})
        with open(out, "rb") as fh:
            width, height, rows = read_png(fh.read())
        self.assertEqual((width, height), (32, 32))
        for y in range(32):
            for x in range(32):
                at = ((y // 8) * 4 + x // 8) * 3
                self.assertEqual(rows[y][x * 3:x * 3 + 3], payload[at:at + 3], (x, y))
        cut = rust("image", "decode", "--text", text[:-5], "--out", out, model=self.model)
        self.assertTrue(cut["repaired"])

    def test_bad_input_is_refused_as_python_refuses_it(self):
        for args in (("image", "encode", self.png, "--size", 30), ("image", "decode", "--text", "hello", "--out",
                                                                     os.path.join(tmpdir(), "x.png"))):
            with self.subTest(args=args):
                self.assertIn("error", rust(*args, model=self.model, expect=1))


@unittest.skipIf(GO is None, "no Go toolchain on PATH")
class TestRustImagesAgainstGo(unittest.TestCase):
    """The same picture gives Go's text exactly: the same decoders, the same box filter."""

    @classmethod
    def setUpClass(cls):
        cls.binary = os.path.join(tmpdir(), "radixnet-count")
        if not os.path.exists(cls.binary):
            env = {**os.environ, "GOFLAGS": os.environ.get("GOFLAGS", "-mod=mod"),
                   "GOCACHE": os.environ.get("GOCACHE", os.path.join(tmpdir(), "gocache"))}
            proc = subprocess.run([GO, "build", "-o", cls.binary, "./cmd/radixnet-count"],
                                  cwd=os.path.join(ROOT, "go"), capture_output=True, text=True, env=env, timeout=600)
            if proc.returncode != 0:
                raise unittest.SkipTest(f"go build failed:\n{proc.stderr}")
        goroot = subprocess.run([GO, "env", "GOROOT"], capture_output=True, text=True).stdout.strip()
        cls.go_testdata = sorted(glob.glob(os.path.join(goroot, "src", "image", "testdata", "*.*")))

    def go(self, *args):
        proc = subprocess.run([self.binary, "--json", "--model", fresh("go.count.json"), *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True, timeout=300)
        return proc.returncode, (json.loads(proc.stdout) if proc.returncode == 0 else proc.stderr)

    def pictures(self):
        """PNGs of every colour type Go reads, built here, plus Go's own JPEG / GIF / PNG test data."""
        w, h = 37, 23
        rgb = lambda x, y: ((x * 7) % 256, (y * 11) % 256, (x * y) % 256)  # noqa: E731
        made = {
            "rgb8": png_bytes(w, h, 2, 8, rgb),
            "rgb8-interlaced": png_bytes(w, h, 2, 8, rgb, interlace=True),
            "rgb16": png_bytes(w, h, 2, 16, lambda x, y: (x * 1771, y * 2819, (x + y) * 999)),
            "rgba8": png_bytes(w, h, 6, 8, lambda x, y: (*rgb(x, y), (x * 13 + y) % 256)),
            "rgba16": png_bytes(w, h, 6, 16, lambda x, y: (x * 999, y * 1999, 30000, (x * y * 97) % 65536)),
            "gray1": png_bytes(w, h, 0, 1, lambda x, y: ((x + y) % 2,)),
            "gray4-trns": png_bytes(w, h, 0, 4, lambda x, y: ((x * y) % 16,), trns=b"\x00\x05"),
            "gray8-trns": png_bytes(w, h, 0, 8, lambda x, y: ((x * 5 + y) % 256,), trns=b"\x00\x10"),
            "gray16": png_bytes(w, h, 0, 16, lambda x, y: ((x * 1500 + y * 900) % 65536,)),
            "graya8": png_bytes(w, h, 4, 8, lambda x, y: ((x * 5) % 256, (y * 9) % 256)),
            # (100, 0, 200) is the transparent colour: Go reads it as black
            "rgb8-trns": png_bytes(w, h, 2, 8, lambda x, y: (x % 3 * 100, 0, y % 2 * 200),
                                   trns=b"\x00\x64\x00\x00\x00\xc8"),
            "palette2": png_bytes(w, h, 3, 2, lambda x, y: ((x + y) % 4,),
                                  palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (9, 9, 9)]),
            "palette8-trns": png_bytes(w, h, 3, 8, lambda x, y: ((x * y) % 6,),
                                       palette=[(i * 40, 255 - i * 40, 7) for i in range(4)], trns=b"\xff\x80\x00"),
        }
        paths = []
        for name, data in made.items():
            path = os.path.join(tmpdir(), f"go-{name}.png")
            with open(path, "wb") as fh:
                fh.write(data)
            paths.append(path)
        return paths + [p for p in self.go_testdata if p.endswith((".png", ".jpeg", ".gif"))]

    def test_the_same_picture_gives_go_s_text(self):
        pictures = self.pictures()
        self.assertGreater(len(pictures), 20)
        decoded = 0
        for path in pictures:
            for size in (8, 64):
                with self.subTest(picture=os.path.basename(path), size=size):
                    code, go_doc = self.go("image", "encode", path, "--size", size, "--encoder", "tiny")
                    proc = subprocess.run([build(), "--json", "--model", fresh("rs.count.json"), "image", "encode",
                                           path, "--size", str(size), "--encoder", "tiny"],
                                          cwd=ROOT, capture_output=True, text=True, timeout=300)
                    self.assertEqual(code == 0, proc.returncode == 0, (go_doc, proc.stderr))
                    if code != 0:
                        continue
                    rs_doc = json.loads(proc.stdout)
                    self.assertEqual((rs_doc["text"], rs_doc["source_size"]), (go_doc["text"], go_doc["source_size"]))
                    decoded += 1
        self.assertGreater(decoded, 30)


class TestRustMediaServer(unittest.TestCase):
    """The routes the Images and Speech tabs call, answered in the JSON the Python server answers with."""

    @classmethod
    def setUpClass(cls):
        cls.dir = os.path.join(tmpdir(), "media-server")
        os.makedirs(cls.dir, exist_ok=True)
        cls.model = os.path.join(cls.dir, "served.count.json")
        cls.uploads = os.path.join(cls.dir, "uploads")
        cls.server = serve(cls, cls.model, "--upload-dir", cls.uploads)
        cls.wav = speech.wav_bytes(tone(0.1), RATE)
        cls.png = gradient_png(64, 40)

    def raw(self, path, data, content_type):
        return self.server.client.request("POST", path, raw=data, headers={"Content-Type": content_type})

    def multipart(self, path, name, data, content_type="application/octet-stream"):
        boundary = "----radixnet-rust"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"note\"\r\n\r\nignored\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        return self.raw(path, body, f"multipart/form-data; boundary={boundary}")

    def test_the_tabs_are_served(self):
        status, info, _ = self.server.get("/api/status")
        self.assertEqual(status, 200)
        for route in ("GET /api/images", "POST /api/images/encode", "POST /api/images/decode",
                      "POST /api/images/tutor", "GET /api/speech", "POST /api/speech/teach",
                      "POST /api/speech/decode", "POST /api/speech/tutor", "POST /api/speech/transcribe"):
            self.assertIn(route, info["routes"])
        status, images, _ = self.server.get("/api/images")
        self.assertEqual((status, images["auto"], images["pillow"], images["sd_loaded"]), (200, "tiny", True, False))
        status, spoken, _ = self.server.get("/api/speech")
        self.assertEqual((status, spoken["token"], spoken["faster_whisper"]), (200, "<speech>", False))
        self.assertIn("text_format", spoken)

    def test_an_image_encodes_the_same_from_every_body(self):
        status, a, _ = self.raw("/api/images/encode?size=32&encoder=tiny", self.png, "image/png")
        self.assertEqual(status, 200, a)
        self.assertEqual((a["name"], a["width"], a["latent_shape"], a["source_size"], a["job"], a["upload"]),
                         ("image", 32, [3, 4, 4], [64, 40], None, None))
        status, b, _ = self.multipart("/api/images/encode?size=32&encoder=tiny", "photo.png", self.png, "image/png")
        self.assertEqual((status, b["name"], b["text"]), (200, "photo.png", a["text"]))
        status, c, _ = self.server.post("/api/images/encode", {
            "name": "b64.png", "content_base64": base64.b64encode(self.png).decode(), "size": 32, "encoder": "tiny"})
        self.assertEqual((status, c["text"]), (200, a["text"]))
        for bad, why in ((("/api/images/encode?encoder=tiny", b"not an image", "image/png"), "readable"),
                         (("/api/images/encode?size=30", self.png, "image/png"), "multiple of 8")):
            status, doc, _ = self.raw(*bad)
            self.assertEqual(status, 400, doc)
            self.assertIn(why, doc["error"])
        status, doc, _ = self.server.post("/api/images/encode", {"name": "x.png", "content": "text, not bytes"})
        self.assertEqual(status, 400)
        self.assertIn("bytes", doc["error"])

    def test_an_image_can_be_saved_and_trained_on(self):
        from tests.test_api import wait_for_job

        status, doc, _ = self.raw("/api/images/encode?size=32&encoder=tiny&train=true&save_as=photo.txt&epochs=2",
                                  self.png, "image/png")
        self.assertEqual(status, 202, doc)
        self.assertEqual((doc["upload"]["name"], doc["job"]["type"]), ("photo.txt", "train"))
        job = wait_for_job(self.server.client)
        self.assertEqual(len(job["history"]), 2, job)
        with open(os.path.join(self.uploads, "photo.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), doc["text"] + "\n")

    def test_a_text_is_drawn_back(self):
        text = vision.pack_text("tiny", 32, 32, bytes(range(48)))
        status, doc, _ = self.server.post("/api/images/decode", {"text": text})
        self.assertEqual(status, 200, doc)
        width, height, rows = read_png(base64.b64decode(doc["png_base64"]))
        self.assertEqual((width, height, doc["repaired"], rows[0][:3]), (32, 32, False, bytes([0, 1, 2])))
        status, doc, _ = self.server.post("/api/images/decode", {"text": text[:-20]})
        self.assertEqual((status, doc["repaired"]), (200, True))
        status, doc, _ = self.server.post("/api/images/decode", {"text": "hello"})
        self.assertEqual(status, 400)

    def test_speech_teaches_the_same_as_python_from_every_body(self):
        spoken = "the cat sat on the mat"
        expected = speech.teach(self.wav, transcript=spoken, rate=4000)
        status, a, _ = self.raw(f"/api/speech/teach?rate=4000&transcript={spoken.replace(' ', '%20')}", self.wav,
                                "audio/wav")
        self.assertEqual(status, 200, a)
        self.assertEqual((a["token"], a["texts"], a["audio"], a["asr"]),
                         (expected["token"], expected["texts"], expected["audio"], expected["asr"]))
        self.assertEqual((a["name"], a["job"], a["audio"]["samples"]), ("speech", None, 400))
        status, b, _ = self.multipart(f"/api/speech/teach?rate=4000&transcript={spoken.replace(' ', '%20')}",
                                      "utterance.wav", self.wav, "audio/wav")
        self.assertEqual((status, b["name"], b["texts"]), (200, "utterance.wav", a["texts"]))
        status, c, _ = self.server.post("/api/speech/teach", {
            "name": "b64.wav", "content_base64": base64.b64encode(self.wav).decode(), "rate": 4000,
            "transcript": spoken, "pair": True})
        self.assertEqual((status, len(c["texts"]), c["pair"]), (200, 3, True))
        status, doc, _ = self.raw("/api/speech/teach?rate=2", self.wav, "audio/wav")
        self.assertEqual(status, 400)
        self.assertIn("sample rate", doc["error"])
        status, doc, _ = self.raw("/api/speech/teach", b"not audio at all", "audio/wav")
        self.assertEqual(status, 400, doc)

    def test_speech_decodes_and_transcribes(self):
        taught = speech.teach(self.wav, transcript="hello", rate=4000)
        status, doc, _ = self.server.post("/api/speech/decode", {"text": taught["texts"][1]})
        self.assertEqual(status, 200, doc)
        expected = speech.decode_text(taught["texts"][1])
        self.assertEqual(base64.b64decode(doc["wav_base64"]), expected.pop("wav"))
        self.assertEqual({k: doc[k] for k in expected}, expected)
        status, doc, _ = self.raw("/api/speech/transcribe?transcript=hello%20there", self.wav, "audio/wav")
        self.assertEqual((status, doc["transcript"], doc["backend"], doc["name"]), (200, "hello there", "given",
                                                                                    "speech"))

    def test_the_recall_tutors_answer_the_frontend(self):
        status, doc, _ = self.multipart("/api/images/tutor?size=32&lead=4&length=16", "pic.png", self.png, "image/png")
        self.assertEqual(status, 200, doc)
        self.assertEqual((doc["modality"], doc["negative"], len(doc["lessons"])), ("image", None, 1))
        self.assertEqual(doc["lessons"][0]["exercise"]["label"], "pic.png")
        for key in ("lessons", "passed", "mean_score", "mean_agreement", "reasons", "modality"):
            self.assertIn(key, doc["report"])
        status, doc, _ = self.server.post("/api/speech/tutor", {
            "name": "a.wav", "content_base64": base64.b64encode(self.wav).decode(), "transcript": "hello",
            "rate": RATE, "length": 80, "blame": True})
        self.assertEqual(status, 200, doc)
        self.assertIn("<speech:", doc["lessons"][0]["exercise"]["label"])
        self.assertIn("blamed", doc["negative"]["taught"])
        self.assertIn("reasons", doc["negative"])
        status, doc, _ = self.server.post("/api/speech/tutor", {"texts": []})
        self.assertEqual(status, 400, doc)
        status, doc, _ = self.server.post("/api/speech/tutor", {"texts": ["aud:mu:8000x1:AAAA"], "attempts": "many"})
        self.assertEqual(status, 400, doc)

    def test_uploads_take_every_form_python_takes(self):
        status, doc, _ = self.multipart("/api/uploads", "multi.txt", b"the owl flew\nover the hill\n", "text/plain")
        self.assertEqual(status, 200, doc)
        self.assertEqual((doc["uploads"][0]["name"], doc["uploads"][0]["lines"]), ("multi.txt", 2))
        status, doc, _ = self.raw("/api/uploads?name=raw.txt", "﻿café au lait\n".encode(), "text/plain")
        self.assertEqual((status, doc["uploads"][0]["name"], doc["uploads"][0]["chars"]), (200, "raw.txt", 13))
        status, doc, _ = self.server.post("/api/uploads", {"files": [
            {"name": "a.txt", "content": "one line"},
            {"name": "../b.txt", "content_base64": base64.b64encode(b"two\nlines\n").decode()}]})
        self.assertEqual(status, 200, doc)
        self.assertEqual([u["name"] for u in doc["uploads"]], ["a.txt", "b.txt"])
        status, doc, _ = self.raw("/api/uploads", b"raw bytes without a name", "application/octet-stream")
        self.assertEqual(status, 400, doc)
        status, listing, _ = self.server.get("/api/uploads")
        self.assertLessEqual({"multi.txt", "raw.txt", "a.txt", "b.txt"}, {u["name"] for u in listing["uploads"]})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
