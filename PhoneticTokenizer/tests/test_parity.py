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
