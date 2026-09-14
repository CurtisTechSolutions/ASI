"""Tests for audioimage.cli - every command, end to end."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioimage.cli import EXIT_ERROR, EXIT_OK, build_parser, main  # noqa: E402
from audioimage.png import read_png  # noqa: E402
from audioimage.wav import read_wav  # noqa: E402

FAST = ["--n-fft", "256", "--hop", "64"]


def run(*argv):
    """Run the command line, returning ``(code, stdout, stderr)``."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


class CliCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.wav = self.path("tone.wav")
        code, _, err = run("tone", "-o", self.wav, "--kind", "chord", "--duration", "0.4", "--rate", "16000", "-q")
        self.assertEqual(code, EXIT_OK, err)

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestParser(unittest.TestCase):
    def test_help_without_a_command(self):
        code, out, _ = run()
        self.assertEqual(code, EXIT_OK)
        self.assertIn("usage", out)

    def test_every_command_is_registered(self):
        parser = build_parser()
        actions = [a for a in parser._actions if getattr(a, "choices", None) and "encode" in getattr(a, "choices", {})]
        self.assertTrue(actions)
        self.assertEqual(
            set(actions[0].choices),
            {"tone", "encode", "decode", "roundtrip", "view", "waveform", "info"},
        )


class TestTone(CliCase):
    def test_writes_a_wav(self):
        audio = read_wav(self.wav)
        self.assertEqual(audio.sample_rate, 16000)
        self.assertAlmostEqual(audio.duration, 0.4, places=3)

    def test_every_kind(self):
        for kind in ("sine", "chirp", "noise", "chord", "melody", "harmonics"):
            with self.subTest(kind=kind):
                out = self.path(f"{kind}.wav")
                code, _, err = run("tone", "-o", out, "--kind", kind, "--duration", "0.2", "--rate", "8000", "-q")
                self.assertEqual(code, EXIT_OK, err)
                self.assertTrue(os.path.exists(out))

    def test_notes(self):
        out = self.path("notes.wav")
        code, _, _ = run("tone", "-o", out, "--kind", "melody", "--notes", "C4,E4 G4", "--duration", "0.3", "-q")
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(os.path.exists(out))

    def test_bit_depths(self):
        for bits in (8, 16, 24, 32):
            with self.subTest(bits=bits):
                out = self.path(f"b{bits}.wav")
                self.assertEqual(run("tone", "-o", out, "--duration", "0.1", "--bits", bits, "-q")[0], EXIT_OK)
                self.assertEqual(read_wav(out).meta["bits"], bits)


class TestEncodeDecode(CliCase):
    def test_encode(self):
        png = self.path("out.png")
        code, out, err = run("encode", self.wav, "-o", png, *FAST)
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn("wrote", out)
        image = read_png(png)
        self.assertEqual(image.height, 129)
        self.assertEqual(image.depth, 16)

    def test_default_output_path(self):
        code, _, _ = run("encode", self.wav, *FAST, "-q")
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(os.path.exists(self.path("tone.png")))

    def test_decode(self):
        png = self.path("out.png")
        back = self.path("back.wav")
        run("encode", self.wav, "-o", png, *FAST, "-q")
        code, out, err = run("decode", png, "-o", back, "--iters", "6", "-q")
        self.assertEqual(code, EXIT_OK, err)
        self.assertEqual(len(read_wav(back).samples), len(read_wav(self.wav).samples))

    def test_roundtrip_reports_metrics(self):
        code, out, err = run(
            "--json", "roundtrip", self.wav, "--image", self.path("rt.png"), "-o", self.path("rt.wav"), *FAST, "--iters", "20", "-q"
        )
        self.assertEqual(code, EXIT_OK, err)
        payload = json.loads(out)
        self.assertIn("metrics", payload)
        self.assertLess(payload["metrics"]["spectral_convergence"], 0.5)
        self.assertTrue(os.path.exists(self.path("rt.png")))
        self.assertTrue(os.path.exists(self.path("rt.wav")))

    def test_every_colormap_and_depth(self):
        for colormap in ("gray", "gray-inv", "fire", "ice", "viridis", "magma"):
            with self.subTest(colormap=colormap):
                png = self.path(f"{colormap}.png")
                code, _, err = run("encode", self.wav, "-o", png, "--colormap", colormap, *FAST, "-q")
                self.assertEqual(code, EXIT_OK, err)
                back = self.path(f"{colormap}.wav")
                self.assertEqual(run("decode", png, "-o", back, "--iters", "4", "-q")[0], EXIT_OK)

    def test_scales_and_axes(self):
        for scale in ("db", "linear", "sqrt"):
            for freq_scale in ("linear", "log", "mel"):
                with self.subTest(scale=scale, freq_scale=freq_scale):
                    png = self.path(f"{scale}-{freq_scale}.png")
                    code, _, err = run(
                        "encode", self.wav, "-o", png, "--scale", scale, "--freq-scale", freq_scale, *FAST, "-q"
                    )
                    self.assertEqual(code, EXIT_OK, err)

    def test_explicit_size(self):
        png = self.path("small.png")
        run("encode", self.wav, "-o", png, "--height", "64", "--width", "32", *FAST, "-q")
        image = read_png(png)
        self.assertEqual((image.width, image.height), (32, 64))

    def test_8_bit(self):
        png = self.path("8bit.png")
        run("encode", self.wav, "-o", png, "--depth", "8", *FAST, "-q")
        self.assertEqual(read_png(png).depth, 8)

    def test_both_backends_agree(self):
        results = {}
        for backend in ("python", "numpy"):
            png = self.path(f"{backend}.png")
            code, _, err = run("--backend", backend, "encode", self.wav, "-o", png, *FAST, "-q")
            if code != EXIT_OK and "unavailable" in err:
                self.skipTest("numpy is not installed")
            self.assertEqual(code, EXIT_OK, err)
            results[backend] = list(read_png(png).data)
        self.assertEqual(results["python"], results["numpy"])

    def test_global_flags_after_the_command(self):
        code, out, _ = run("encode", self.wav, "-o", self.path("x.png"), "--json", *FAST)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["command"], "encode")

    def test_decode_without_metadata(self):
        from audioimage.png import Image, write_png

        image = Image(24, 129, "L", 8)
        image.fill(255)
        for x in range(24):
            image.set(x, 64, 0)
        png = self.path("drawn.png")
        write_png(png, image)
        code, out, err = run("decode", png, "-o", self.path("drawn.wav"), "--rate", "8000", "--iters", "4", "-q")
        self.assertEqual(code, EXIT_OK, err)
        self.assertIn("no settings", out)
        self.assertGreater(len(read_wav(self.path("drawn.wav")).samples), 0)


class TestViews(CliCase):
    def test_view_from_a_wav(self):
        out = self.path("v.png")
        code, _, err = run("view", self.wav, "-o", out, *FAST, "-q")
        self.assertEqual(code, EXIT_OK, err)
        self.assertEqual(read_png(out).mode, "RGB")

    def test_view_from_a_picture(self):
        png = self.path("enc.png")
        run("encode", self.wav, "-o", png, *FAST, "-q")
        code, _, err = run("view", png, "-o", self.path("v2.png"), "-q")
        self.assertEqual(code, EXIT_OK, err)

    def test_view_options(self):
        code, _, err = run(
            "view", self.wav, "-o", self.path("v3.png"), "--view-colormap", "ice",
            "--zoom", "2", "--no-legend", "--title", "hello", *FAST, "-q",
        )
        self.assertEqual(code, EXIT_OK, err)

    def test_view_default_path(self):
        run("view", self.wav, *FAST, "-q")
        self.assertTrue(os.path.exists(self.path("tone-view.png")))

    def test_waveform(self):
        out = self.path("w.png")
        code, _, err = run("waveform", self.wav, "-o", out, "-q")
        self.assertEqual(code, EXIT_OK, err)
        self.assertTrue(os.path.exists(out))

    def test_waveform_without_axes(self):
        code, _, _ = run("waveform", self.wav, "-o", self.path("w2.png"), "--no-axes", "--view-width", "120", "-q")
        self.assertEqual(code, EXIT_OK)


class TestInfo(CliCase):
    def test_wav(self):
        code, out, _ = run("info", self.wav)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("audio", out)
        self.assertIn("16000 Hz", out)

    def test_picture(self):
        png = self.path("i.png")
        run("encode", self.wav, "-o", png, *FAST, "-q")
        code, out, _ = run("info", png)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("picture", out)
        self.assertIn("n_fft 256", out)

    def test_json(self):
        code, out, _ = run("--json", "info", self.wav)
        payload = json.loads(out)
        self.assertEqual(payload["kind"], "audio")
        self.assertIn("backends", payload)

    def test_json_of_a_picture_carries_the_settings(self):
        png = self.path("j.png")
        run("encode", self.wav, "-o", png, *FAST, "-q")
        payload = json.loads(run("--json", "info", png)[1])
        self.assertEqual(payload["meta"]["n_fft"], 256)


class TestErrors(CliCase):
    def test_missing_file(self):
        code, _, err = run("encode", self.path("nope.wav"))
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("no such file", err)

    def test_wrong_file_type(self):
        code, _, err = run("decode", self.wav, "-o", self.path("x.wav"))
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("not a PNG", err)

    def test_png_given_to_an_audio_command(self):
        png = self.path("p.png")
        run("encode", self.wav, "-o", png, *FAST, "-q")
        code, _, err = run("waveform", png)
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("PNG", err)

    def test_bad_settings(self):
        for argv, expect in (
            (("encode", "--n-fft", "1000"), "power of two"),
            (("encode", "--hop", "0"), "hop"),
            (("encode", "--top-db", "0"), "top_db"),
        ):
            with self.subTest(argv=argv):
                code, _, err = run(argv[0], self.wav, "-o", self.path("bad.png"), *argv[1:])
                self.assertEqual(code, EXIT_ERROR)
                self.assertIn(expect, err)

    def test_errors_are_one_line(self):
        _, _, err = run("encode", self.path("nope.wav"))
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertNotIn("Traceback", err)

    def test_unreadable_image(self):
        broken = self.path("broken.png")
        with open(broken, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"rubbish")
        code, _, err = run("decode", broken, "-o", self.path("x.wav"))
        self.assertEqual(code, EXIT_ERROR)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
