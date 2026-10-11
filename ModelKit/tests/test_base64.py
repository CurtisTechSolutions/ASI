"""The base64 unit through the kit: the command line and the HTTP API read any file as its bytes.

``../RadixCyclicNN/tests/test_base64_unit.py`` holds the unit itself; this holds
what the kit does with it.  A model of base64 symbols (``--units base64``) is
trained on the bytes of whatever it is given - a text file line by line, an
image or any other file whole - is asked prefixes as the symbols their bytes
fix (``--prefix``, ``--prefix-file``), says what its answers hold (``decoded``,
``decoded_continuation``, ``decoded_bytes``) and writes them back out as bytes
(``--decode-to``).  The API does the same, keeps a binary upload byte for byte,
and takes the model's own words back as they are when a request says
``symbols``.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from radixnet.encoding import BASE64_ALPHABET, base64_bytes, base64_prefix, base64_text  # noqa: E402
from radixnet.graph import FIRST  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import Client, start_server, wait_for_job  # noqa: F401
except ImportError:  # ``python -m unittest tests.test_base64``
    from tests.test_api import Client, start_server, wait_for_job  # noqa: F401

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(KIT), "RadixCyclicNN")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

TEXTS = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the log",
    "a quick brown fox jumps over the lazy dog",
]
# the head of a PNG and some bytes that are no text: what an image, a video or a recording is to the unit
BLOB = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF, 0xFE, 0x80, 0x10, 0x20, 0x30, 0xC0, 0xC1])
ALPHABET = set(BASE64_ALPHABET)


def run(*args, model, expect=0):
    """``python -m radixnet --json`` from the model's checkout; the parsed document (``{"error"}`` on a failure)."""
    cmd = [sys.executable, "-m", "radixnet", "--backend", "python", "--json", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=300)
    if proc.returncode != expect:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {"error": proc.stderr.strip()}


def symbols_only(labels):
    """Is every label made of the 64 symbols (sentinels and dead nodes aside)?"""
    return all(set(label) <= ALPHABET for label in labels[FIRST:] if label)


class TestTheCommandLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="radixnet-base64-")
        cls.text_file = os.path.join(cls.tmp, "corpus.txt")
        with open(cls.text_file, "w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(TEXTS) + "\n\n   \n")
        cls.blob_file = os.path.join(cls.tmp, "image.png")
        with open(cls.blob_file, "wb") as fh:
            fh.write(BLOB)
        cls.model = os.path.join(cls.tmp, "b64.json")
        cls.trained = run("--kind", "count", "--units", "base64", "--seed", 1, "train",
                          "--data", cls.text_file, cls.blob_file, "--epochs", 2, model=cls.model)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def test_the_bytes_are_what_it_learned(self):
        """Four lines of text and one image: five texts, every label made of the symbols."""
        self.assertEqual(self.trained["texts"], 5)
        with open(self.model, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["graph"]["encoding"], {"unit": "base64", "n": 3, "stride": 1})
        self.assertTrue(symbols_only(doc["graph"]["nodes"]["labels"]))
        info = run("info", model=self.model)
        self.assertEqual((info["stats"]["encoding"], info["stats"]["units"]), ("base64:3:1", "symbols"))

    def test_a_prefix_is_asked_in_symbols_and_answered_in_bytes(self):
        doc = run("predict", "--prefix", "the cat sat on the", "--length", 8, "--to-end", "--k", 2, model=self.model)
        self.assertEqual(doc["prefix"], "the cat sat on the")
        self.assertEqual(doc["prefix_symbols"], base64_prefix(b"the cat sat on the"))
        self.assertTrue(doc["full_text"].startswith(doc["prefix_symbols"]))
        self.assertIn(doc["decoded"], ("the cat sat on the mat", "the cat sat on the floor"))
        self.assertTrue(doc["decoded"].endswith(doc["decoded_continuation"]))
        self.assertIn(doc["decoded_continuation"], (" mat", " floor"))
        self.assertEqual(doc["decoded_bytes"], len(doc["decoded"].encode()))
        for row in doc["top"] + doc["bottom"]:
            self.assertEqual(row["decoded"], base64_bytes(row["full_text"]).decode("utf-8", "replace"))
            self.assertTrue(row["decoded"].endswith(row["decoded_continuation"]))
        self.assertNotIn("prefix_file", doc)
        self.assertNotIn("decoded_to", doc)

    def test_a_file_is_continued_and_written_back_byte_for_byte(self):
        """The head of the image as the prefix: the walk writes the rest of it, and --decode-to the file again."""
        head = self.path("head.bin")
        with open(head, "wb") as fh:
            fh.write(BLOB[:8])
        out = self.path("again.png")
        doc = run("predict", "--prefix-file", head, "--length", 4, "--to-end", "--decode-to", out, model=self.model)
        self.assertEqual(doc["prefix_file"], head)
        self.assertEqual(doc["prefix_symbols"], base64_prefix(BLOB[:8]))
        self.assertEqual(doc["prefix"], doc["prefix_symbols"])
        self.assertEqual(doc["decoded_to"], [{"path": out, "bytes": len(BLOB)}])
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), BLOB)
        self.assertEqual(doc["decoded_bytes"], len(BLOB))

    def test_generate_decodes_every_sample(self):
        out = self.path("sample.txt")
        doc = run("generate", "--mode", "beam", "--count", 2, "--prefix", "the", "--max-length", 40,
                  "--decode-to", out, model=self.model)
        self.assertEqual(doc["prefix"], "the")
        self.assertEqual(doc["prefix_symbols"], "dGhl")
        self.assertEqual([w["path"] for w in doc["decoded_to"]], [self.path("sample-1.txt"), self.path("sample-2.txt")])
        for sample, written in zip(doc["samples"], doc["decoded_to"]):
            self.assertTrue(sample["decoded"].startswith("the"), sample)
            with open(written["path"], "rb") as fh:
                data = fh.read()
            self.assertEqual(data, base64_bytes(sample["full_text"]))
            self.assertEqual(written["bytes"], len(data))
            self.assertEqual(sample["decoded_continuation"], data[3:].decode("utf-8", "replace"))

    def test_a_score_reads_the_bytes(self):
        known = run("score", "--text", TEXTS[0], model=self.model)["results"][0]
        self.assertEqual(known["text"], base64_text(TEXTS[0].encode()))
        self.assertEqual(known["unknown_transitions"], 0)
        noise = run("score", "--text", "qzx wqzj vbn", model=self.model)["results"][0]
        self.assertGreater(noise["unknown_transitions"], 0)
        from_file = run("score", "--data", self.blob_file, model=self.model)["results"]
        self.assertEqual([r["text"] for r in from_file], [base64_text(BLOB)])

    def test_feedback_learns_the_bytes(self):
        out = self.path("liked.json")
        run("feedback", "--good-text", "the cat ate the rat", "--out", out, model=self.model)
        with open(out, encoding="utf-8") as fh:
            labels = json.load(fh)["graph"]["nodes"]["labels"]
        self.assertTrue(symbols_only(labels))
        doc = run("predict", "--prefix", "the cat ate", "--length", 6, "--to-end", model=out)
        self.assertEqual(doc["decoded"], "the cat ate the rat")

    def test_there_are_no_words_to_list(self):
        self.assertIn("no words to list", run("words", model=self.model, expect=1)["error"])

    def test_one_prefix_or_the_other(self):
        run("predict", "--prefix", "a", "--prefix-file", self.blob_file, model=self.model, expect=1)
        run("predict", model=self.model, expect=1)

    def test_whole_file_keeps_every_byte(self):
        model = self.path("whole.json")
        doc = run("--kind", "count", "--units", "base64", "train", "--data", self.text_file, "--whole-file",
                  "--epochs", 1, model=model)
        self.assertEqual(doc["texts"], 1)
        with open(self.text_file, "rb") as fh:
            whole = base64_text(fh.read())  # the blank lines and the last newline included
        self.assertEqual(run("info", model=model)["stats"]["trained_chars"], len(whole))

    def test_a_text_model_reads_a_prefix_file_as_text(self):
        model = self.path("chars.json")
        run("--kind", "count", "train", "--data", self.text_file, "--epochs", 1, model=model)
        head = self.path("head.txt")
        with open(head, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on")
        out = self.path("chars-out.txt")
        doc = run("predict", "--prefix-file", head, "--length", 4, "--decode-to", out, model=model)
        self.assertEqual(doc["prefix"], "the cat sat on")
        self.assertTrue(doc["full_text"].startswith("the cat sat on"))
        for key in ("prefix_symbols", "decoded", "decoded_continuation", "decoded_bytes"):
            self.assertNotIn(key, doc)
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), doc["full_text"])
        self.assertIn("not valid UTF-8", run("predict", "--prefix-file", self.blob_file, model=model, expect=1)["error"])


class TestTheApi(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="radixnet-base64-uploads-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.client, self.server, self.service = start_server(self.addCleanup, upload_dir=self.dir)
        status, data, _ = self.client.post("/api/reset", {"kind": "count", "encoding": "base64:3:1", "seed": 1})
        self.assertEqual(status, 200, data)

    def post(self, path, body, expect=200):
        status, data, _ = self.client.post(path, body)
        self.assertEqual(status, expect, data)
        return data

    def train(self, **body):
        self.post("/api/train", {"epochs": 2, **body}, expect=202)
        job = wait_for_job(self.client)
        self.assertEqual(job["state"], "done", job)

    def labels(self):
        with self.service.session() as model:
            return list(model.graph.labels)

    def test_the_encoding_is_base64(self):
        data = self.client.get("/api/encoding")[1]
        self.assertEqual((data["encoding"], data["unit"]), ("base64:3:1", "base64"))
        self.assertEqual(self.client.get("/api/model")[1]["units"], "symbols")

    def test_texts_are_learned_as_their_bytes_and_answers_decoded(self):
        self.train(texts=TEXTS)
        self.assertTrue(symbols_only(self.labels()))
        data = self.post("/api/predict", {"prefix": "the cat sat on the", "length": 8, "to_end": True, "k": 2,
                                          "guard": False})
        self.assertEqual(data["prefix"], "the cat sat on the")
        self.assertEqual(data["prefix_symbols"], base64_prefix(b"the cat sat on the"))
        self.assertIn(data["decoded"], ("the cat sat on the mat", "the cat sat on the floor"))
        self.assertTrue(data["decoded"].endswith(data["decoded_continuation"]))
        for row in data["top"] + data["bottom"]:
            self.assertEqual(row["decoded"], base64_bytes(row["full_text"]).decode("utf-8", "replace"))
        data = self.post("/api/generate", {"count": 2, "mode": "beam", "prefix": "the", "guard": False})
        self.assertEqual(data["prefix_symbols"], "dGhl")
        for sample in data["samples"]:
            self.assertTrue(sample["decoded"].startswith("the"), sample)
            self.assertEqual(sample["decoded_bytes"], len(base64_bytes(sample["full_text"])))

    def test_a_binary_upload_is_kept_byte_for_byte_and_learned_whole(self):
        data = self.post("/api/uploads", {"name": "image.png", "content_base64": base64.b64encode(BLOB).decode()},
                         expect=201)
        self.assertEqual(data["uploads"][0]["bytes"], len(BLOB))
        with open(os.path.join(self.dir, "image.png"), "rb") as fh:
            self.assertEqual(fh.read(), BLOB)
        self.train(files=["image.png"])
        data = self.post("/api/predict", {"prefix": "", "length": 1, "to_end": True, "guard": False})
        self.assertEqual(base64_bytes(data["full_text"]), BLOB)
        self.assertEqual(data["decoded_bytes"], len(BLOB))

    def test_a_text_upload_is_still_text(self):
        bom = "﻿" + "\n".join(TEXTS) + "\n"
        self.post("/api/uploads", {"name": "corpus.txt", "content_base64": base64.b64encode(bom.encode()).decode()},
                  expect=201)
        with open(os.path.join(self.dir, "corpus.txt"), "rb") as fh:
            self.assertEqual(fh.read(), bom[1:].encode())  # the BOM dropped, as it always was
        self.train(files=["corpus.txt"])
        data = self.post("/api/score", {"text": TEXTS[2]})
        self.assertEqual(data["unknown_transitions"], 0)

    def test_the_models_own_words_come_back_as_they_are(self):
        """A like or a rating gives back what the model said - symbols already - and ``symbols`` says so."""
        self.train(texts=TEXTS)
        said = base64_text(b"the dog sat on the log")
        self.assertEqual(self.post("/api/score", {"text": said, "symbols": True})["unknown_transitions"], 0)
        self.assertGreater(self.post("/api/score", {"text": said})["unknown_transitions"], 0)  # its symbols' bytes
        self.post("/api/feedback", {"good": [base64_text(b"the cat ate the rat")], "symbols": True}, expect=202)
        self.assertEqual(wait_for_job(self.client)["state"], "done")
        data = self.post("/api/predict", {"prefix": base64_prefix(b"the cat ate"), "symbols": True, "length": 6,
                                          "to_end": True, "guard": False})
        self.assertEqual(data["prefix_symbols"], base64_prefix(b"the cat ate"))
        self.assertEqual(data["decoded"], "the cat ate the rat")
        # fourteen symbols fix ten bytes and half the eleventh: the walk finished the "e"
        self.assertEqual(data["decoded_continuation"], "e the rat")

    def test_the_preview_shows_the_symbols(self):
        data = self.post("/api/encoding/preview", {"text": "the cat"})
        self.assertEqual((data["text"], data["symbols"]), ("the cat", "dGhlIGNhdA"))
        self.assertEqual(data["windows"][:2], ["dGh", "Ghl"])
        self.assertTrue(data["round_trip"])
        data = self.post("/api/encoding/preview", {"text": "dGhl", "symbols": True})
        self.assertEqual((data["symbols"], data["windows"]), ("dGhl", ["dGh", "Ghl"]))

    def test_there_are_no_words_to_list(self):
        status, data, _ = self.client.get("/api/words")
        self.assertEqual(status, 400, data)

    def test_a_text_model_adds_nothing(self):
        self.post("/api/reset", {"kind": "count", "seed": 1})
        self.train(texts=TEXTS)
        data = self.post("/api/predict", {"prefix": "the cat", "length": 4, "guard": False})
        for key in ("prefix_symbols", "decoded", "decoded_continuation", "decoded_bytes"):
            self.assertNotIn(key, data)
        self.assertNotIn("symbols", self.post("/api/encoding/preview", {"text": "the cat"}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
