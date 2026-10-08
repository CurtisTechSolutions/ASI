"""The neural vocoder for the acoustic units (``phonetok/neural.py``): the excitation, the model, the file, the
trainer, the commands - and the bundled vocoder against the bundled codebook."""

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonetok import Lexicon, PhoneticTokenizer  # noqa: E402
from phonetok.acoustic import (  # noqa: E402
    DEFAULT_CODEBOOK, PITCH, AcousticTokenizer, Codebook, _istft, default_codebook, learn, pcm_samples, synthesize,
)
from phonetok.neural import (  # noqa: E402
    CHIRP, CLIP, DEFAULT_VOCODER, Spec, UnitVocoder, chirp_kernel, codebook_checksum, codes_of, disperse,
    excitation_length, excitation_spectra, find_vocoder, noise_train, pulse_train, vocoder_path_for,
)
from phonetok.synth import Synthesizer, Voice, _Noise, wav_bytes  # noqa: E402

try:
    import numpy as np
except ImportError:  # the vocoder runs without it; the trainer does not
    np = None

TEXTS = ["the cat sat on the mat", "the quick brown fox jumps over the lazy dog", "practice makes perfect"]


def spoken(text: str, pitch: float = 120.0) -> list[float]:
    tok = PhoneticTokenizer(lexicon=Lexicon.core())
    return pcm_samples(Synthesizer(voice_settings=Voice(pitch=pitch)).speak(tok.tokens(text)))


_CLIPS: list[list[float]] = []
_BOOK: list[Codebook] = []


def clips() -> list[list[float]]:
    if not _CLIPS:
        _CLIPS.extend(spoken(t) for t in TEXTS)
    return _CLIPS


def small_book() -> Codebook:
    """A codebook of eight units learned from the three clips (once per run)."""
    if not _BOOK:
        _BOOK.append(learn(clips(), k=8, seed=3, iterations=30, note="test"))
    return _BOOK[0]


def write_wavs(tmp: str) -> list[str]:
    paths = []
    for i, samples in enumerate(clips()):
        path = os.path.join(tmp, f"c{i}.wav")
        with open(path, "wb") as fh:
            fh.write(wav_bytes(struct.pack(f"<{len(samples)}h", *[int(s * 32767) for s in samples])))
        paths.append(path)
    return paths


class TestTheExcitation(unittest.TestCase):
    def test_the_pulse_train_is_the_vocoders(self):
        pulses = pulse_train(16000, 100.0, 16000)
        # the phase accumulates 0.00625 a sample, not exact in binary: the first pulse lands on sample 160 and
        # ninety-nine fit in a second - the same in every port, since each runs the same loop
        self.assertEqual(sum(1 for v in pulses if v == 1.0), 99)
        self.assertEqual(pulses.index(1.0), 160)
        gaps = [j - i for i, j in zip([n for n, v in enumerate(pulses) if v], [n for n, v in enumerate(pulses) if v][1:])]
        self.assertEqual(set(gaps), {160})
        self.assertEqual(pulse_train(0, 100.0, 16000), [])
        with self.assertRaises(ValueError):
            pulse_train(10, 0.0, 16000)

    def test_every_pulse_becomes_the_chirp(self):
        kernel = chirp_kernel()
        self.assertEqual(len(kernel), CHIRP)
        self.assertAlmostEqual(sum(v * v for v in kernel), 1.0, places=12)
        self.assertEqual(kernel[0], 0.0)  # the window starts at zero
        dispersed = disperse(pulse_train(1600, 100.0, 16000))
        self.assertAlmostEqual(sum(v * v for v in dispersed), 9.0, places=9)  # nine pulses, unit energy each
        self.assertEqual(dispersed[:160], [0.0] * 160)
        self.assertNotEqual(dispersed[161], 0.0)
        self.assertEqual(disperse([]), [])
        # a pulse at the very end is cut short, not spilled
        self.assertEqual(len(disperse([0.0, 1.0])), 2)

    def test_the_noise_is_the_vocoders(self):
        noise = _Noise()
        self.assertEqual(noise_train(5), [noise.next() for _ in range(5)])
        self.assertEqual(noise_train(8)[:5], noise_train(5))  # extended, the prefix stays
        self.assertTrue(all(-1.0 <= v < 1.0 for v in noise_train(1000)))

    def test_the_spectra(self):
        a = default_codebook().analysis
        pulses, noises = excitation_spectra(3, a, 120.0)
        self.assertEqual((len(pulses), len(noises)), (3, 3))
        self.assertEqual(len(pulses[0][0]), a.fft // 2 + 1)
        self.assertEqual(pulses[0][1][0], 0.0)  # a real frame: no imaginary part at DC
        self.assertEqual(excitation_spectra(0, a), ([], []))
        self.assertEqual(excitation_length(3, a), 2 * a.hop + a.frame)
        self.assertEqual(excitation_length(0, a), 0)


class TestTheModel(unittest.TestCase):
    def test_a_fresh_vocoder(self):
        book = small_book()
        voc = UnitVocoder.new(book, channels=6, kernel=3, dilations=(1, 2), seed=2)
        a = book.analysis
        self.assertEqual(voc.spec, Spec(units=8, bins=a.fft // 2 + 1, channels=6, kernel=3, dilations=(1, 2)))
        self.assertEqual(voc.spec.context, 3)
        d = voc.describe()
        self.assertEqual(d["parameters"], 8 * 6 + 2 * (6 * 6 * 3 + 6 + 6 * 6 + 6) + 2 * 257 * 6 + 2 * 257 + 8 * 2 * 257)
        self.assertAlmostEqual(d["context_seconds"], 3 * a.hop / a.rate)
        self.assertTrue(voc.matches(book))
        self.assertFalse(voc.matches(default_codebook()))
        codes = [0, 1, 1, 2, 7, 7, 7]
        lp, ln = voc.filters(codes)
        self.assertEqual((len(lp), len(lp[0]), len(ln), len(ln[0])), (7, 257, 7, 257))
        self.assertTrue(all(CLIP[0] <= v <= CLIP[1] for row in lp + ln for v in row))
        samples = voc.samples(codes)
        self.assertEqual(len(samples), excitation_length(7, a))
        self.assertEqual(len(voc.synthesize(codes, gain=0.5, pitch=150.0)), 2 * len(samples))
        self.assertEqual((voc.filters([]), voc.samples([]), voc.synthesize([])), (([], []), [], b""))
        with self.assertRaises(ValueError):
            voc.filters([8])
        with self.assertRaises(ValueError):
            Spec(units=8, bins=257, kernel=4).validate()
        with self.assertRaises(ValueError):
            Spec(units=8, bins=257, dilations=(0,)).validate()
        # two pitches, two excitations: the same filters, other samples
        self.assertNotEqual(voc.samples(codes, 100.0), voc.samples(codes, 200.0))

    def test_the_file_round_trip(self):
        book = small_book()
        voc = UnitVocoder.new(book, channels=6, kernel=3, dilations=(1, 2), seed=2)
        voc.note = "a note"
        voc.trained = {"steps": 3}
        text = voc.dumps()
        doc = json.loads(text)
        self.assertEqual((doc["phonetok"], doc["model"], doc["version"], doc["units"]), ("acoustic vocoder", "source-filter", 1, 8))
        self.assertEqual((doc["weights"]["head"]["w"]["shape"], doc["weights"]["template"]["shape"]), ([2 * 257, 6], [8, 2 * 257]))
        again = UnitVocoder.loads(text)
        self.assertEqual((again.note, again.trained, again.checksum), ("a note", {"steps": 3}, codebook_checksum(book)))
        self.assertEqual(again.spec, voc.spec)
        codes = [3, 3, 4, 0]
        for mine, theirs in zip(voc.filters(codes)[0], again.filters(codes)[0]):
            self.assertLess(max(abs(x - y) for x, y in zip(mine, theirs)), 1e-5)  # float32 in the file
        self.assertTrue(again.matches(book))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.json")
            voc.dump(path)
            self.assertEqual(UnitVocoder.load(path).spec, voc.spec)
        with self.assertRaises(ValueError):
            UnitVocoder.loads(json.dumps({"phonetok": "codebook"}))
        doc["weights"]["head"]["b"]["shape"] = [3]
        with self.assertRaises(ValueError):
            UnitVocoder.loads(json.dumps(doc))
        doc = json.loads(text)
        doc["model"] = "something else"
        with self.assertRaises(ValueError):
            UnitVocoder.loads(json.dumps(doc))

    @unittest.skipIf(np is None, "numpy is not installed")
    def test_python_and_numpy_agree(self):
        book = small_book()
        voc = UnitVocoder.new(book, channels=6, kernel=3, dilations=(1, 2), seed=2)
        codes = [0, 1, 1, 2, 7, 7, 7, 5]
        lp, ln = voc.filters(codes)
        lp2, ln2 = voc._filters_python(codes)
        self.assertLess(max(abs(x - y) for a, b in zip(lp + ln, lp2 + ln2) for x, y in zip(a, b)), 1e-9)
        pure = _istft(voc.spectra(codes, 130.0), book.analysis)
        fast = voc.samples(codes, 130.0)
        self.assertLess(max(abs(x - y) for x, y in zip(pure, fast)), 1e-9)

    def test_codes_of(self):
        book = small_book()
        self.assertEqual(codes_of(["q1", "q3"], book), [1] * book.hold(1) + [3] * book.hold(3))
        self.assertEqual(codes_of([], book), [])
        with self.assertRaises(ValueError):
            codes_of(["q1", "nope"], book)

    def test_finding_a_vocoder(self):
        book = small_book()
        voc = UnitVocoder.new(book, channels=4, kernel=3, dilations=(1,), seed=2)
        self.assertEqual(vocoder_path_for("mine.tsv"), "mine.vocoder.json")
        self.assertEqual(vocoder_path_for("dir/mine.TSV"), "dir/mine.vocoder.json")
        self.assertEqual(vocoder_path_for("mine"), "mine.vocoder.json")
        with tempfile.TemporaryDirectory() as tmp:
            book_path = os.path.join(tmp, "book.tsv")
            book.dump(book_path)
            self.assertIsNone(find_vocoder(book, book_path))  # nothing beside it yet
            self.assertIsNone(find_vocoder(book))  # the bundled vocoder is not this codebook's
            voc.dump(vocoder_path_for(book_path))
            self.assertEqual(find_vocoder(book, book_path).spec, voc.spec)
            elsewhere = os.path.join(tmp, "elsewhere.json")
            voc.dump(elsewhere)
            self.assertEqual(find_vocoder(book, book_path, elsewhere).checksum, voc.checksum)
            with self.assertRaises(ValueError):
                find_vocoder(book, book_path, os.path.join(tmp, "missing.json"))
            old = os.environ.get("PHONETOK_VOCODER")
            try:
                os.environ["PHONETOK_VOCODER"] = elsewhere
                self.assertIsNotNone(find_vocoder(book))
                os.environ["PHONETOK_VOCODER"] = os.path.join(tmp, "missing.json")
                with self.assertRaises(ValueError):
                    find_vocoder(book)
            finally:
                if old is None:
                    os.environ.pop("PHONETOK_VOCODER", None)
                else:
                    os.environ["PHONETOK_VOCODER"] = old
            # a file beside the codebook that belongs to another codebook is a mistake, not a fallback
            other = learn(clips()[:2], k=5, seed=9, iterations=5)
            UnitVocoder.new(other, channels=4, kernel=3, dilations=(1,)).dump(vocoder_path_for(book_path))
            with self.assertRaises(ValueError):
                find_vocoder(book, book_path)
            # the facade
            voc.dump(vocoder_path_for(book_path))
            tok = AcousticTokenizer(book_path)
            self.assertEqual((tok.source, tok.vocoder_name()), (book_path, "neural"))
            self.assertEqual(tok.neural.checksum, voc.checksum)
            units = tok.hear(clips()[0])[:6]
            pcm = tok.synthesize(units)
            self.assertEqual(pcm, tok.neural.synthesize(codes_of(units, book), pitch=PITCH))
            self.assertEqual(len(pcm), len(tok.synthesize(units, neural=False)))
            self.assertEqual(tok.synthesize(units, neural=False, polish=2), synthesize(units, book, polish=2))
            self.assertEqual(tok.synthesize(units, neural=True), pcm)
            plain = AcousticTokenizer(book)
            self.assertEqual((plain.source, plain.neural, plain.vocoder_name()), (None, None, "centroid"))
            self.assertEqual(plain.synthesize(units), synthesize(units, book))
            with self.assertRaises(ValueError):
                plain.vocoder_name(True)
            with self.assertRaises(ValueError):
                plain.synthesize(units, neural=True)


@unittest.skipIf(np is None, "numpy is not installed")
class TestTraining(unittest.TestCase):
    def test_the_gradients(self):
        from phonetok.neural import gradient_check

        self.assertLess(gradient_check(small_book()), 1e-4)

    def test_pitch_tracking(self):
        from phonetok.neural import track_pitch

        a = default_codebook().analysis
        n = np.arange(a.rate)
        saw = 0.3 * (2.0 * ((n * 130.0 / a.rate) % 1.0) - 1.0)  # a sawtooth at 130 Hz
        count = (len(saw) - a.frame) // a.hop + 1
        f0, voiced = track_pitch(np, saw, a, count)
        self.assertEqual((len(f0), len(voiced)), (count, count))
        self.assertGreater(sum(voiced), 0.9 * count)
        self.assertLess(abs(np.median([f for f, v in zip(f0, voiced) if v]) - 130.0), 3.0)
        # silence is not voiced, and gets the default pitch to keep the pulse train going
        f0, voiced = track_pitch(np, np.zeros(a.rate), a, count)
        self.assertFalse(any(voiced))
        self.assertEqual(set(f0), {PITCH})
        # the synthesizer's speech: voiced where it speaks, near its pitch
        y = np.asarray(spoken(TEXTS[0], pitch=140.0))
        count = (len(y) - a.frame) // a.hop + 1
        f0, voiced = track_pitch(np, y, a, count)
        self.assertGreater(sum(voiced), count // 4)
        self.assertLess(abs(np.median([f for f, v in zip(f0, voiced) if v]) - 140.0), 25.0)

    def test_training_moves_the_loss(self):
        from phonetok.neural import SpectralLoss, _example_loss, evaluate, prepare, templates, train

        book = small_book()
        a = book.analysis
        prepared = prepare(np, book, clips())
        loss_fn = SpectralLoss(np, a, 1.0)

        def loss_over(vocoder):
            """The loss over the first sixty frames of each training clip: a fixed measure, no draws in it."""
            w = vocoder._numpy_weights(np)
            return sum(_example_loss(np, w, vocoder.spec, a, loss_fn, c, 0, min(60, c.frames))[0] for c in prepared[:2])

        fresh = UnitVocoder.new(book, channels=8, kernel=3, dilations=(1, 2), seed=1,
                                template=templates(np, prepared[:2], book.k, a))
        before = loss_over(fresh)
        voc, report = train(book, clips(), steps=40, batch=2, segment=60, lr=3e-3, seed=1, channels=8, kernel=3,
                            dilations=(1, 2), hold_out=1, polish=1, note="tiny")
        self.assertTrue(voc.matches(book))
        self.assertEqual((report["recordings"], report["steps"], voc.note), (2, 40, "tiny"))
        self.assertEqual(len(report["history"]), 40)
        after = loss_over(voc)
        self.assertLess(after, before, (before, after))
        held = report["held_out"]
        self.assertEqual(set(held), {"neural_logmel", "griffin_logmel", "neural_logmel_speech", "griffin_logmel_speech",
                                     "neural_round_trip", "griffin_round_trip", "recordings", "polish"})
        self.assertEqual(held["recordings"], 1)
        self.assertEqual(voc.trained["steps"], 40)
        self.assertNotIn("history", voc.trained)
        again = UnitVocoder.loads(voc.dumps())
        self.assertEqual(again.trained["held_out"], held)
        # continued training starts from the weights it was given, and a vocoder of another codebook is refused
        more, report2 = train(book, clips()[:1], steps=2, batch=1, segment=30, vocoder=voc, polish=1)
        self.assertIs(more, voc)
        with self.assertRaises(ValueError):
            train(default_codebook(), clips()[:1], steps=1, batch=1, vocoder=voc)
        with self.assertRaises(ValueError):
            train(book, clips()[:1], steps=1, hold_out=1)
        measured = evaluate(voc, book, clips()[:1], polish=1)
        self.assertEqual(measured["recordings"], 1)
        self.assertEqual(len(prepare(np, book, clips()[:2])), 2)


class TestTheCommands(unittest.TestCase):
    def run_cli(self, *argv):
        from phonetok.cli import main

        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_info_and_replay(self):
        book = small_book()
        voc = UnitVocoder.new(book, channels=4, kernel=3, dilations=(1,), seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            book_path = os.path.join(tmp, "book.tsv")
            book.dump(book_path)
            code, out, err = self.run_cli("vocoder", "info", "--codebook", book_path)
            self.assertEqual(code, 1)
            self.assertIn("no vocoder", err)
            voc.dump(vocoder_path_for(book_path))
            code, out, err = self.run_cli("vocoder", "info", "--codebook", book_path, "--json")
            self.assertEqual(code, 0, err)
            doc = json.loads(out)
            self.assertEqual((doc["path"], doc["matches"], doc["units"], doc["channels"]), (vocoder_path_for(book_path), True, 8, 4))
            code, out, err = self.run_cli("vocoder", "info", vocoder_path_for(book_path), "--codebook", DEFAULT_CODEBOOK)
            self.assertEqual(code, 0, err)
            self.assertIn("NOT for this codebook", out)
            back = os.path.join(tmp, "back.wav")
            for use, want in (("auto", "neural"), ("neural", "neural"), ("centroid", "centroid")):
                code, out, err = self.run_cli("replay", "q1", "q2", "--codebook", book_path, "--out", back, "--use", use, "--json")
                self.assertEqual(code, 0, err)
                self.assertEqual(json.loads(out)["vocoder"], want)
            code, out, err = self.run_cli("replay", "q1", "--codebook", book_path, "--vocoder", os.path.join(tmp, "no.json"), "--out", back)
            self.assertEqual(code, 1)
            self.assertIn("not found", err)

    @unittest.skipIf(np is None, "numpy is not installed")
    def test_train_and_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            wavs = write_wavs(tmp)
            book_path = os.path.join(tmp, "book.tsv")
            small_book().dump(book_path)
            code, out, err = self.run_cli("vocoder", "train", *wavs, "--codebook", book_path, "--steps", "3", "--batch", "1",
                                          "--segment", "40", "--channels", "4", "--kernel", "3", "--dilations", "1",
                                          "--hold-out", "1", "--polish", "1", "--quiet", "--json", "--note", "three clips")
            self.assertEqual(code, 0, err)
            doc = json.loads(out)
            self.assertEqual((doc["path"], doc["recordings"], doc["channels"], doc["note"]), (vocoder_path_for(book_path), 3, 4, "three clips"))
            self.assertEqual(doc["held_out"]["recordings"], 1)
            self.assertTrue(os.path.exists(doc["path"]))
            code, out, err = self.run_cli("vocoder", "eval", wavs[0], "--codebook", book_path, "--polish", "1", "--json")
            self.assertEqual(code, 0, err)
            doc = json.loads(out)
            self.assertEqual((doc["recordings"], doc["polish"]), (1, 1))
            self.assertIn("neural_logmel", doc)


class TestTheBundledVocoder(unittest.TestCase):
    def test_it_belongs_to_the_bundled_codebook(self):
        self.assertTrue(os.path.exists(DEFAULT_VOCODER), "no bundled vocoder (tests/make_vocoder.py writes it)")
        book = default_codebook()
        voc = UnitVocoder.load()
        self.assertTrue(voc.matches(book))
        self.assertEqual(voc.spec.units, book.k)
        self.assertIn("synthesizer", voc.note)
        held = voc.trained["held_out"]
        # the claim: held-out speech comes back nearer the original than through the centroid vocoder, over every
        # frame and over the speech alone (the round trip is the centroid vocoder's by construction: see evaluate)
        self.assertLess(held["neural_logmel"], held["griffin_logmel"], held)
        self.assertLess(held["neural_logmel_speech"], 0.7 * held["griffin_logmel_speech"], held)
        tok = AcousticTokenizer()
        self.assertEqual((tok.source, tok.vocoder_name()), (DEFAULT_CODEBOOK, "neural"))
        units = tok.hear(clips()[2])
        pcm = tok.synthesize(units)
        self.assertEqual(len(pcm) // 2, excitation_length(len(codes_of(units, book)), book.analysis))
        peak = max(abs(v) for v in struct.unpack(f"<{len(pcm) // 2}h", pcm))
        self.assertGreater(peak, 2000)
        self.assertLess(peak, 32767)


if __name__ == "__main__":
    unittest.main()
