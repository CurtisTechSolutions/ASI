"""The cross-port contract, from the Python side.

``tests/parity.json`` is what the Python tokenizer says about a battery of
texts (``tests/make_parity.py`` writes it); the Go and Rust test suites hold
their ports to it.  This module holds Python to the same file, so that a change
here is a change everywhere on purpose - and, when ``go`` and ``cargo`` are on
``PATH``, builds the two ports' command lines and compares them with Python
over the sample corpora and the whole core lexicon, token for token, at every
level.  Those tests are skipped when the toolchains are absent.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phonetok import LEVELS, PhoneticTokenizer, Lexicon, Phonotactics  # noqa: E402
from phonetok.acoustic import (  # noqa: E402
    AcousticTokenizer, Codebook, default_codebook, frames_of, learn, pcm_samples, read_wav, synthesize,
)
from phonetok.synth import Synthesizer, Voice  # noqa: E402
from phonetok.g2p import Transcriber  # noqa: E402
from phonetok.phones import SYMBOLS  # noqa: E402
from phonetok.rules import letter_to_sound  # noqa: E402
from phonetok.syllables import syllabify  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "parity.json")
SCRATCH = os.path.join(ROOT, "build", "parity")


def fixture() -> dict:
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


class TestPythonHoldsToTheFixture(unittest.TestCase):
    """What the file says is what Python says now (regenerate it on a deliberate change)."""

    def test_levels(self):
        doc = fixture()
        self.assertEqual(doc["symbols"], list(SYMBOLS))
        for level in LEVELS:
            tok = PhoneticTokenizer(level=level, lexicon=Lexicon.core())
            for row in doc["levels"][level]["rows"]:
                toks = tok.tokenize(row["text"])
                self.assertEqual([t.text for t in toks], row["tokens"], (level, row["text"]))
                self.assertEqual([t.kind for t in toks], row["kinds"], (level, row["text"]))
                self.assertEqual([t.word for t in toks], row["words"], (level, row["text"]))
                self.assertEqual(tok.encode(row["text"]), row["ids"], (level, row["text"]))
                self.assertEqual(tok.decode(tok.tokens(row["text"])), row["decoded"], (level, row["text"]))
            self.assertEqual(list(tok.vocab.tokens), doc["levels"][level]["vocab"], level)

    def test_words_and_phonotactics(self):
        doc = fixture()
        tok = PhoneticTokenizer(lexicon=Lexicon.core())
        for i, row in enumerate(doc["levels"]["phoneme"]["rows"]):
            rows = tok.explain(row["text"])
            self.assertEqual([{k: r[k] for k in ("word", "phones", "how", "syllables", "ipa")} for r in rows],
                             doc["explain"][i], row["text"])
            self.assertEqual(tok.ipa(row["text"]), doc["ipa"][row["text"]])
        for word, want in doc["rules"].items():
            self.assertEqual(letter_to_sound(word), want, word)
        for word, want in doc["syllables"].items():
            self.assertEqual([s.text() for s in syllabify(tok.pronounce(word))], want, word)
        p = Phonotactics.from_lexicon(Lexicon.core())
        for pair, (affinity, prob) in doc["affinity"].items():
            a, b = pair.split()
            self.assertAlmostEqual(p.affinity(a, b), affinity, places=9)
            self.assertAlmostEqual(p.prob(a, b), prob, places=12)
        for key, want in doc["coin"].items():
            if key.startswith("seed 5 syllables "):
                got = p.build(random.Random(5), syllables=int(key[-1]))
            else:
                got = p.build(random.Random(int(key)), avoid=(ph for _, ph in Lexicon.core().items()))
            self.assertEqual(got, want, key)
        t = Transcriber(Lexicon.core())
        for w in ("two", "two", "right", "zebra"):
            t.word(w)
        for phones, want in doc["transcriber_spell"].items():
            self.assertEqual(t.spell(phones.split()), want, phones)


# -- the ports' command lines ---------------------------------------------------------

def assert_close(test, got, want, tolerance=1e-9):
    """Floats within tolerance of each other, relative to their size."""
    got, want = list(got), list(want)
    test.assertEqual(len(got), len(want))
    for g, w in zip(got, want):
        test.assertLessEqual(abs(g - w), tolerance * max(1.0, abs(w)), (g, w))


class TestPythonHoldsToTheAcousticFixture(unittest.TestCase):
    """The acoustic units: what the file says the synthesized utterances are heard as, Python hears now."""

    def test_acoustic(self):
        import struct

        doc = fixture()["acoustic"]
        tok = PhoneticTokenizer(lexicon=Lexicon.core())
        clips = []
        for text, tokens in zip(doc["texts"], doc["tokens"]):
            self.assertEqual(tok.tokens(text), tokens)
            clips.append(pcm_samples(Synthesizer(voice_settings=Voice(**doc["voice"])).speak(tokens)))
        self.assertEqual([len(c) for c in clips], doc["samples"])
        book = default_codebook()
        frames = frames_of(clips[0], book.analysis)
        self.assertEqual(len(frames), doc["frames"]["count"])
        for key, row in (("0", frames[0]), ("50", frames[50]), ("last", frames[-1])):
            assert_close(self, row, doc["frames"]["rows"][key])
        heard = AcousticTokenizer(book).listen(clips[0])
        self.assertEqual(heard.units, doc["units"])
        self.assertEqual(heard.codes, doc["codes"])
        want = doc["learn"]
        small = learn(clips, k=want["k"], seed=want["seed"], iterations=want["iterations"])
        for got, w in zip(small.centroids, want["centroids"]):
            assert_close(self, got, w)
        self.assertEqual(small.counts, want["counts"])
        self.assertEqual(small.runs, want["runs"])
        assert_close(self, small.mean, want["mean"])
        assert_close(self, [small.inertia], [want["inertia"]])
        for key in ("replay", "polished"):
            spec = doc[key]
            pcm = synthesize(spec["units"], book, polish=spec["polish"])
            values = struct.unpack(f"<{len(pcm) // 2}h", pcm)
            self.assertEqual(len(values), spec["samples"], key)
            got = values[spec["at"]:spec["at"] + len(spec["values"])]
            self.assertLessEqual(max(abs(g - w) for g, w in zip(got, spec["values"])), 2, key)


def _build_go():
    if shutil.which("go") is None:
        return None
    os.makedirs(SCRATCH, exist_ok=True)
    exe = os.path.join(SCRATCH, "phonetok-go")
    try:
        subprocess.run(["go", "build", "-o", exe, "./cmd/phonetok"], cwd=os.path.join(ROOT, "go"), check=True,
                       capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return exe


def _build_rust():
    if shutil.which("cargo") is None:
        return None
    try:
        subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=os.path.join(ROOT, "rust"), check=True,
                       capture_output=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    exe = os.path.join(ROOT, "rust", "target", "release", "phonetok")
    return exe if os.path.isfile(exe) else None


def corpora() -> list[str]:
    """The texts the ports are compared on: the fixture's, the sample corpora, and every core word."""
    texts = [row["text"] for row in fixture()["levels"]["phoneme"]["rows"] if row["text"].strip()]
    data = os.path.join(os.path.dirname(ROOT), "RadixCyclicNN", "data")
    if os.path.isdir(data):
        for name in sorted(os.listdir(data)):
            if name.endswith(".txt"):
                with open(os.path.join(data, name), encoding="utf-8") as f:
                    texts.extend(line.strip() for line in f if line.strip())
    words = sorted(Lexicon.core().words())
    for i in range(0, len(words), 200):
        texts.append(" ".join(words[i:i + 200]))
    texts.append(" ".join(f"{w}s {w}ed {w}ing {w}ly {w}ness un{w}" for w in words[::40]))
    return texts


class PortParity(unittest.TestCase):
    exe: str | None = None
    name = ""

    def run_cli(self, *args: str) -> str:
        env = dict(os.environ, PHONETOK_LEXICON="")
        out = subprocess.run([self.exe, *args], capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(out.returncode, 0, f"{self.name}: {out.stderr}")
        return out.stdout

    def test_every_level_over_the_corpora(self):
        if not self.exe:
            self.skipTest(f"no {self.name} toolchain, or the build failed")
        texts = corpora()
        for level in LEVELS:
            tok = PhoneticTokenizer(level=level, lexicon=Lexicon.core())
            for text in texts:
                with self.subTest(level=level, text=text[:40]):
                    theirs = self.run_cli("tokenize", "--core", "--level", level, text).rstrip("\n")
                    self.assertEqual(theirs, tok.text(text))

    def test_the_other_commands(self):
        if not self.exe:
            self.skipTest(f"no {self.name} toolchain, or the build failed")
        tok = PhoneticTokenizer(lexicon=Lexicon.core())
        text = "The quick brown fox jumps over the lazy dog, 42 times!"
        theirs = json.loads(self.run_cli("explain", "--core", "--json", text))
        ours = tok.explain(text)
        self.assertEqual([{k: r[k] for k in ("word", "phones", "how", "syllables", "ipa")} for r in theirs["words"]],
                         [{k: r[k] for k in ("word", "phones", "how", "syllables", "ipa")} for r in ours])
        self.assertEqual(self.run_cli("ipa", "--core", text).strip(), tok.ipa(text))
        form = tok.text(text)
        # a command line has no memory of what it read in another run, so it is compared with a fresh tokenizer
        fresh = PhoneticTokenizer(lexicon=Lexicon.core())
        self.assertEqual(self.run_cli("decode", "--core", form).strip(), fresh.decode(form.split()))
        ids = " ".join(map(str, tok.encode(text)))
        self.assertEqual(self.run_cli("decode", "--core", ids).strip(), fresh.decode(tok.encode(text)))
        self.assertEqual(self.run_cli("rhymes", "--core", "cat", "hat").strip(), "cat / hat: rhyme")
        self.assertEqual(self.run_cli("blend", "--core", "smoke", "fog").split()[4], "smog")
        coins = [line.split()[0] for line in self.run_cli("coin", "--core", "--count", "5", "--seed", "3").splitlines()]
        rng = random.Random(3)
        ours = []
        for _ in range(5):
            phones = tok.phonotactics.build(rng, avoid=(p for _, p in tok.lexicon.items()))
            ours.append(tok.transcriber.spell(phones))
        self.assertEqual(coins, ours)


    def test_say_speaks_the_same(self):
        """The three voices are one voice: the same samples, give or take the last bits of a libm."""
        if not self.exe:
            self.skipTest(f"no {self.name} toolchain, or the build failed")
        import struct
        import wave

        from phonetok.synth import Synthesizer

        tok = PhoneticTokenizer(lexicon=Lexicon.core())
        text = "The quick brown fox jumps over the lazy dog. Is the cat on the mat? Strengths, butter!"
        ours = Synthesizer().speak(tok.tokens(text))
        os.makedirs(SCRATCH, exist_ok=True)
        path = os.path.join(SCRATCH, f"say-{self.name}.wav")
        self.run_cli("say", "--core", "--out", path, text)
        with wave.open(path) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 16000))
            theirs = w.readframes(w.getnframes())
        self.assertEqual(len(theirs), len(ours), "the same number of samples")
        a = struct.unpack(f"<{len(ours) // 2}h", ours)
        b = struct.unpack(f"<{len(theirs) // 2}h", theirs)
        worst = max(abs(x - y) for x, y in zip(a, b))
        mean = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
        self.assertLessEqual(worst, 64, f"{self.name}: the worst sample differs by {worst}")
        self.assertLessEqual(mean, 2.0, f"{self.name}: the mean difference is {mean:.3f}")
        # streamed, token by token, it is the same audio
        form = tok.text(text)
        raw = subprocess.run([self.exe, "say", "--core", "--raw"], input=(form + "\n").encode(), capture_output=True,
                             env=dict(os.environ, PHONETOK_LEXICON=""), timeout=600)
        self.assertEqual(raw.returncode, 0, raw.stderr)
        self.assertEqual(raw.stdout, theirs)


    def test_hear_learn_and_replay_agree(self):
        """The acoustic units: the port hears a recording as the same units, learns the same codebook from
        the same recordings, and speaks units back as the same samples."""
        if not self.exe:
            self.skipTest(f"no {self.name} toolchain, or the build failed")
        import struct
        import tempfile

        from phonetok.synth import write_wav

        tok = PhoneticTokenizer(lexicon=Lexicon.core())
        with tempfile.TemporaryDirectory() as tmp:
            wavs = []
            for i, text in enumerate(("the cat sat on the mat", "a bird in the hand")):
                path = os.path.join(tmp, f"{i}.wav")
                write_wav(path, Synthesizer(voice_settings=Voice(pitch=130.0)).speak(tok.tokens(text)))
                wavs.append(path)
            book = default_codebook()
            ours = AcousticTokenizer(book)
            with open(wavs[0], "rb") as fh:
                heard = ours.listen(fh.read())
            self.assertEqual(self.run_cli("hear", wavs[0]).split(), heard.units)
            theirs = json.loads(self.run_cli("hear", "--json", "--frames", *wavs))
            self.assertEqual([d["path"] for d in theirs], wavs)
            self.assertEqual(theirs[0]["codes"], heard.codes)
            self.assertEqual(theirs[0]["frames"], heard.frames)
            path = os.path.join(tmp, "theirs.tsv")
            self.run_cli("learn", *wavs, "--units", "5", "--seed", "4", "--iterations", "15", "--out", path)
            theirs_book = Codebook.load(path)
            recordings = []
            for w in wavs:
                with open(w, "rb") as fh:
                    recordings.append(ours.samples(fh.read()))
            ours_book = learn(recordings, k=5, seed=4, iterations=15)
            self.assertEqual(theirs_book.k, 5)
            for got, want in zip(theirs_book.centroids, ours_book.centroids):
                assert_close(self, got, want)
            self.assertEqual((theirs_book.counts, theirs_book.runs), (ours_book.counts, ours_book.runs))
            assert_close(self, theirs_book.mean, ours_book.mean)
            back = os.path.join(tmp, "back.wav")
            units = heard.units[:15]
            for polish in (0, 2):
                self.run_cli("replay", *units, "--out", back, "--polish", str(polish))
                with open(back, "rb") as fh:
                    samples, rate = read_wav(fh.read())
                pcm = synthesize(units, book, polish=polish)
                want = struct.unpack(f"<{len(pcm) // 2}h", pcm)
                self.assertEqual((rate, len(samples)), (16000, len(want)), polish)
                self.assertLessEqual(max(abs(s * 32768 - w) for s, w in zip(samples, want)), 2, polish)
            # and the port's codebook file is read back the same
            self.assertEqual(Codebook.loads(theirs_book.dumps()).centroids, theirs_book.centroids)


class TestGoParity(PortParity):
    name = "Go"

    @classmethod
    def setUpClass(cls):
        cls.exe = _build_go()


class TestRustParity(PortParity):
    name = "Rust"

    @classmethod
    def setUpClass(cls):
        cls.exe = _build_rust()


del PortParity  # the base class is not a test of its own


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
