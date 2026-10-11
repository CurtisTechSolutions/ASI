"""The base64 unit of the Rust port held to Python's: anything - a text, an image - as the n-grams of its bytes.

Skipped when no Rust toolchain is available (``tests/rust_harness.py`` builds the binary once).

A model of base64 symbols (``--units base64``) reads every file as the bytes it holds - a UTF-8 text file one
text per line, anything else one text of every byte - learns n-grams of their base64 symbols, is asked a prefix
as the symbols its bytes fix, and says what its answers hold (``decoded``, ``decoded_continuation``,
``decoded_bytes``).  ``tests/test_base64.py`` holds Python to that; this holds the Rust port to Python:

* **the CLI** - both train on the sample corpus and a small binary file and write the same model file, byte for
  byte but for the cache stamp; ``predict --prefix`` and ``--prefix-file`` (a binary prefix whose decoded text
  is no UTF-8 included), ``generate``, the bytes ``--decode-to`` writes, ``score --text`` / ``--data``,
  ``feedback --good-text`` and ``2nrl`` agree, and ``words`` is refused on both;
* **the server** - a binary upload is kept byte for byte, training on uploads builds the same graph, predict and
  generate answer the same decoded fields, score, feedback and the encoding preview agree with and without
  ``symbols``, and ``/api/words`` is a 400 on both.

The encoding-parity list of ``tests/test_rust_parity.py`` (``test_the_other_dials_agree_too``) holds
``base64:3:1`` and ``base64:4:4`` to the same graph from the sample corpus.
"""

import base64
import json
import os
import shutil
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from tests.rust_harness import CORPUS, GARBAGE, assert_close, build, load_json, py, rust, serve, tmpdir  # noqa: E402

from radixnet.encoding import base64_bytes, base64_prefix, base64_text  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server  # noqa: E402
except ImportError:
    from tests.test_api import start_server  # noqa: E402

# the head of a PNG and some bytes that are no text: what an image, a video or a recording is to the unit
BLOB = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF, 0xFE, 0x80, 0x10, 0x20, 0x30, 0xC0, 0xC1])
DECODED = ("decoded", "decoded_continuation", "decoded_bytes")
"""What a model of base64 symbols adds to every answer, and to every row of one."""


def setUpModule():
    build()


def scratch(name: str) -> str:
    path = os.path.join(tmpdir(), "base64", name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def write(path: str, data: bytes) -> str:
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def graph_of(path: str) -> str:
    """A model file's graph as both implementations write it, without the cache stamp every load bumps."""
    graph = load_json(path)["graph"]
    graph.pop("version", None)
    return json.dumps(graph, separators=(",", ":"), ensure_ascii=False)


def rows(doc: dict, *keys: str) -> list:
    """The ``top`` and ``bottom`` rows of a prediction, each as the values of ``keys``."""
    return [[r.get(k) for k in keys] for r in doc.get("top", []) + doc.get("bottom", [])]


class TestRustBase64Cli(unittest.TestCase):
    """``radixnet --units base64``: the same model from the same bytes, and the same answers out of it."""

    @classmethod
    def setUpClass(cls):
        cls.dir = scratch("cli")
        cls.blob = write(os.path.join(cls.dir, "image.png"), BLOB)
        cls.head = write(os.path.join(cls.dir, "head.bin"), BLOB[:8])
        cls.py_model = os.path.join(cls.dir, "py.b64.json")
        cls.rs_model = os.path.join(cls.dir, "rs.b64.json")
        # Python takes the files after one --data, the Rust port (as the Go one) one --data per file
        cls.py_train = py("--kind", "count", "--units", "base64", "--seed", 1, "train", "--data", CORPUS, cls.blob,
                          "--epochs", 2, model=cls.py_model)
        cls.rs_train = rust("--units", "base64", "--seed", 1, "train", "--data", CORPUS, "--data", cls.blob,
                            "--epochs", 2, "--workers", 4, model=cls.rs_model)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def both(self, *args, expect: int = 0) -> tuple[dict, dict]:
        """The same command on each side's own model."""
        return py(*args, model=self.py_model, expect=expect), rust(*args, model=self.rs_model, expect=expect)

    def test_the_same_bytes_make_the_same_model(self):
        """The corpus a line at a time and the image whole: the same texts, the same file."""
        self.assertEqual(graph_of(self.py_model), graph_of(self.rs_model))
        with open(CORPUS, encoding="utf-8") as fh:
            lines = sum(1 for line in fh if line.strip())
        self.assertEqual((self.py_train["texts"], self.rs_train["texts"]), (lines + 1, lines + 1))
        py_doc, rs_doc = load_json(self.py_model), load_json(self.rs_model)
        for doc in (py_doc, rs_doc):
            self.assertEqual(doc["graph"]["encoding"], {"unit": "base64", "n": 3, "stride": 1})
        for key in ("epochs_total", "trained_texts", "trained_chars"):
            self.assertEqual(py_doc["meta"][key], rs_doc["meta"][key], key)
        a, b = self.both("info")
        for stats in (a["stats"], b["stats"]):
            self.assertEqual((stats["encoding"], stats["unit"], stats["units"]), ("base64:3:1", "base64", "symbols"))
        self.assertNotIn("vocabulary", b["stats"], "symbols are letters, not words")

    def test_a_prefix_is_asked_in_symbols_and_answered_in_bytes(self):
        keys = ("prefix", "prefix_symbols", "continuation", "full_text", "path", "reached_end", *DECODED)
        for prefix in ("the cat sat on", "the d", "the", "zzq", ""):
            with self.subTest(prefix=prefix):
                a, b = self.both("predict", "--prefix", prefix, "--length", 8, "--k", 3, "--mode", "beam")
                self.assertEqual({k: a.get(k) for k in keys}, {k: b.get(k) for k in keys})
                self.assertEqual(b["prefix"], prefix)
                self.assertEqual(b["prefix_symbols"], base64_prefix(prefix.encode()))
                self.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9)
                self.assertEqual(rows(a, "full_text", *DECODED), rows(b, "full_text", *DECODED))
                assert_close(self, [r["cost"] for r in a["top"]], [r["cost"] for r in b["top"]])
                self.assertNotIn("prefix_file", b)
                self.assertNotIn("decoded_to", b)

    def test_a_file_is_continued_and_written_back_as_bytes(self):
        """The head of the image as the prefix: its decoded text is no UTF-8, and both write the same U+FFFDs."""
        keys = ("prefix", "prefix_file", "prefix_symbols", "continuation", "full_text", *DECODED)
        a = py("predict", "--prefix-file", self.head, "--length", 4, "--to-end", "--k", 2, "--decode-to",
               self.path("py-again.png"), model=self.py_model)
        b = rust("predict", "--prefix-file", self.head, "--length", 4, "--to-end", "--k", 2, "--decode-to",
                 self.path("rs-again.png"), model=self.rs_model)
        self.assertEqual({k: a.get(k) for k in keys}, {k: b.get(k) for k in keys})
        self.assertEqual(rows(a, "full_text", *DECODED), rows(b, "full_text", *DECODED))
        self.assertEqual(b["prefix_symbols"], base64_prefix(BLOB[:8]))
        self.assertEqual(b["prefix"], b["prefix_symbols"])
        self.assertIn("�", b["decoded"])
        self.assertEqual(b["decoded"], base64_bytes(b["full_text"]).decode("utf-8", "replace"))
        self.assertEqual(read(self.path("py-again.png")), read(self.path("rs-again.png")))
        self.assertEqual(read(self.path("rs-again.png")), BLOB)
        self.assertEqual(b["decoded_to"], [{"path": self.path("rs-again.png"), "bytes": len(BLOB)}])
        self.assertEqual(a["decoded_to"][0]["bytes"], b["decoded_to"][0]["bytes"])

    def test_generate_decodes_every_sample_and_writes_them_numbered(self):
        sample = ("text", "full_text", "reached_end", *DECODED)
        for given in (["--prefix", "the"], [], ["--prefix-file", self.head]):
            with self.subTest(prefix=given):
                a = py("generate", "--mode", "beam", "--count", 3, "--max-length", 40, *given,
                       "--decode-to", self.path("py-sample.bin"), model=self.py_model)
                b = rust("generate", "--mode", "beam", "--count", 3, "--max-length", 40, *given,
                         "--decode-to", self.path("rs-sample.bin"), model=self.rs_model)
                self.assertEqual([[s[k] for k in sample] for s in a["samples"]],
                                 [[s[k] for k in sample] for s in b["samples"]])
                for key in ("prefix", "prefix_symbols", "prefix_file", "count"):
                    self.assertEqual(a.get(key), b.get(key), key)
                # one sample is written where it was asked, several numbered before the extension
                count = len(b["samples"])
                if given[:1] != ["--prefix-file"]:
                    self.assertEqual(count, 3, "three samples to number")
                where = [self.path("rs-sample.bin")] if count == 1 else [
                    self.path(f"rs-sample-{i}.bin") for i in range(1, count + 1)]
                self.assertEqual([w["path"] for w in b["decoded_to"]], where)
                for x, y, s in zip(a["decoded_to"], b["decoded_to"], b["samples"]):
                    self.assertEqual(read(x["path"]), read(y["path"]))
                    self.assertEqual((x["bytes"], y["bytes"]), (s["decoded_bytes"], s["decoded_bytes"]))
        a, b = self.both("generate", "--mode", "dijkstra", "--max-length", 40, "--prefix", "the cat")
        self.assertEqual([[s[k] for k in sample] for s in a["samples"]], [[s[k] for k in sample] for s in b["samples"]])

    def test_a_score_reads_the_bytes(self):
        for args in (["--text", "the cat sat on the mat"], ["--text", "qzx wqzj vbn"], ["--data", self.blob],
                     ["--data", GARBAGE]):
            with self.subTest(args=args[0]):
                a, b = self.both("score", *args)
                self.assertEqual([r["text"] for r in a["results"]], [r["text"] for r in b["results"]])
                assert_close(self, [r["log_prob"] for r in a["results"]], [r["log_prob"] for r in b["results"]])
                for key in ("chars", "transitions", "unknown_transitions"):
                    self.assertEqual([r[key] for r in a["results"]], [r[key] for r in b["results"]], key)
        self.assertEqual(rust("score", "--data", self.blob, model=self.rs_model)["results"][0]["text"],
                         base64_text(BLOB))

    def test_feedback_and_2nrl_learn_the_bytes(self):
        for tag, args in (("feedback", ["feedback", "--good-text", "the cat ate the rat", "--good-text", "   ",
                                        "--bad", self.blob]),
                          ("2nrl", ["2nrl", "--bad", GARBAGE, "--good", self.blob])):
            with self.subTest(command=tag):
                py_out, rs_out = self.path(f"py-{tag}.json"), self.path(f"rs-{tag}.json")
                a = py(*args, "--out", py_out, model=self.py_model)
                b = rust(*args, "--out", rs_out, model=self.rs_model)
                self.assertEqual(graph_of(py_out), graph_of(rs_out))
                if tag == "feedback":  # the blank text dropped, the image read whole
                    for key in ("action", "good_texts", "bad_texts"):
                        self.assertEqual(a.get(key), b.get(key), key)
                    self.assertEqual((b["action"], b["good_texts"], b["bad_texts"]), ("2nrl", 1, 1))
        # the rated sentence is said back, in bytes, on both sides
        model = self.path("rs-feedback.json")
        doc = rust("predict", "--prefix", "the cat ate", "--length", 6, "--to-end", model=model)
        self.assertEqual(doc["decoded"], "the cat ate the rat")
        self.assertEqual(py("predict", "--prefix", "the cat ate", "--length", 6, "--to-end",
                            model=self.path("py-feedback.json"))["decoded"], doc["decoded"])

    def test_whole_file_keeps_every_byte(self):
        py_model, rs_model = self.path("py.whole.json"), self.path("rs.whole.json")
        a = py("--kind", "count", "--units", "base64", "--seed", 1, "train", "--data", CORPUS, "--whole-file",
               "--epochs", 1, model=py_model)
        b = rust("--units", "base64", "--seed", 1, "train", "--data", CORPUS, "--whole-file", "--epochs", 1,
                 model=rs_model)
        self.assertEqual((a["texts"], b["texts"]), (1, 1))
        self.assertEqual(graph_of(py_model), graph_of(rs_model))
        self.assertEqual(b["stats"]["trained_chars"], len(base64_text(read(CORPUS))))

    def test_what_is_refused_is_refused_on_both(self):
        a, b = self.both("words", expect=1)
        for doc in (a, b):
            self.assertIn("no words to list", doc["error"])
        self.both("predict", "--prefix", "a", "--prefix-file", self.blob, expect=1)
        self.both("predict", expect=1)
        a, b = self.both("score", "--data", self.path("no-such-file.txt"), expect=1)
        for doc in (a, b):
            self.assertIn(f"data file not found: {self.path('no-such-file.txt')}", doc["error"])

    def test_a_text_model_reads_a_prefix_file_as_text(self):
        py_model, rs_model = self.path("py.chars.json"), self.path("rs.chars.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 1, model=py_model)
        rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 1, model=rs_model)
        head = write(self.path("head.txt"), "the cat sat on".encode())
        a = py("predict", "--prefix-file", head, "--length", 4, "--decode-to", self.path("py-chars.txt"), model=py_model)
        b = rust("predict", "--prefix-file", head, "--length", 4, "--decode-to", self.path("rs-chars.txt"),
                 model=rs_model)
        for key in ("prefix", "prefix_file", "continuation", "full_text"):
            self.assertEqual(a.get(key), b.get(key), key)
        self.assertEqual(b["prefix"], "the cat sat on")
        for key in ("prefix_symbols", *DECODED):
            self.assertNotIn(key, b)
        self.assertEqual(read(self.path("rs-chars.txt")), b["full_text"].encode())
        self.assertEqual(read(self.path("py-chars.txt")), read(self.path("rs-chars.txt")))
        for doc in (py("predict", "--prefix-file", self.blob, model=py_model, expect=1),
                    rust("predict", "--prefix-file", self.blob, model=rs_model, expect=1)):
            self.assertIn("is not valid UTF-8 text", doc["error"])


class TestRustBase64Server(unittest.TestCase):
    """Both servers over a model of base64 symbols, the same requests in the same order: the same answers.

    The tests run in their numbered order: each one leaves both models where the next one starts.
    """

    @classmethod
    def setUpClass(cls):
        cls.dir = scratch("server")
        cls.py_uploads = os.path.join(cls.dir, "py-uploads")
        cls.rs_uploads = os.path.join(cls.dir, "rs-uploads")
        os.makedirs(cls.py_uploads)
        os.makedirs(cls.rs_uploads)
        python, _server, _service = start_server(cls.addClassCleanup, model_path=os.path.join(cls.dir, "py.json"),
                                                 seed=1, upload_dir=cls.py_uploads)
        mine = serve(cls, os.path.join(cls.dir, "rs.json"), "--seed", "1", "--upload-dir", cls.rs_uploads)
        cls.clients = (python, mine.client)  # Python's first, always
        cls.reset("base64:3:1")

    @classmethod
    def reset(cls, encoding: str) -> None:
        """A fresh count model of ``encoding`` on both servers."""
        for client in cls.clients:
            status, doc, _ = client.post("/api/reset", {"kind": "count", "encoding": encoding, "seed": 1})
            assert status == 200, doc

    def both(self, method: str, route: str, body: dict | None = None, **kwargs) -> tuple:
        """The same request to each server: ``((status, doc), (status, doc))``, Python's first."""
        a, b = (client.request(method, route, body=body, **kwargs)[:2] for client in self.clients)
        return a, b

    def job(self, route: str, body: dict) -> None:
        """A job on both servers, waited for: it must end done on each."""
        (a, a_doc), (b, b_doc) = self.both("POST", route, body)
        self.assertEqual((a, b), (202, 202), (a_doc, b_doc))
        for client in self.clients:
            for _ in range(6000):
                _, job, _ = client.get("/api/job")
                if job and job.get("state") != "running":
                    break
                time.sleep(0.02)
            self.assertEqual(job["state"], "done", job)

    def same_models(self, tag: str) -> None:
        """Both models saved and compared: the same graph, byte for byte but for the cache stamp."""
        paths = [os.path.join(self.dir, f"{side}-{tag}.json") for side in ("py", "rs")]
        for client, path in zip(self.clients, paths):
            status, doc, _ = client.post("/api/save", {"path": path})
            self.assertEqual(status, 200, doc)
        self.assertEqual(graph_of(paths[0]), graph_of(paths[1]), tag)

    def test_1_the_dial_is_base64(self):
        (_, a), (_, b) = self.both("GET", "/api/encoding")
        for doc in (a, b):
            self.assertEqual((doc["encoding"], doc["unit"]), ("base64:3:1", "base64"))
        (_, a), (_, b) = self.both("GET", "/api/model")
        self.assertEqual((a["units"], b["units"]), ("symbols", "symbols"))

    def test_2_a_binary_upload_is_kept_byte_for_byte(self):
        """However it arrives - inline as JSON, a raw body, a multipart part - a file that is no text is kept as
        it is, and one that is text is kept as its text (the byte-order mark dropped, as it always was)."""
        inline = {"name": "image.png", "content_base64": base64.b64encode(BLOB).decode()}
        (a, a_doc), (b, b_doc) = self.both("POST", "/api/uploads", inline)
        self.assertEqual((a // 100, b // 100), (2, 2), (a_doc, b_doc))  # Python answers 201, this port 200
        for doc in (a_doc, b_doc):
            self.assertEqual((doc["uploads"][0]["name"], doc["uploads"][0]["bytes"]), ("image.png", len(BLOB)))
        boundary = "----radixnet-base64"
        multipart = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"part.png\"\r\n"
                     "Content-Type: image/png\r\n\r\n").encode() + BLOB + f"\r\n--{boundary}--\r\n".encode()
        for route, data, kind in (("/api/uploads?name=raw.png", BLOB, "image/png"),
                                  ("/api/uploads", multipart, f"multipart/form-data; boundary={boundary}")):
            (a, a_doc), (b, b_doc) = self.both("POST", route, raw=data, headers={"Content-Type": kind})
            self.assertEqual((a // 100, b // 100), (2, 2), (a_doc, b_doc))
        corpus = "﻿".encode() + read(CORPUS)
        (a, _), (b, _) = self.both("POST", "/api/uploads", {"name": "corpus.txt",
                                                            "content_base64": base64.b64encode(corpus).decode()})
        self.assertEqual((a // 100, b // 100), (2, 2))
        for uploads in (self.py_uploads, self.rs_uploads):
            for name in ("image.png", "raw.png", "part.png"):
                self.assertEqual(read(os.path.join(uploads, name)), BLOB, f"{uploads}/{name}")
            self.assertEqual(read(os.path.join(uploads, "corpus.txt")), read(CORPUS))
        (_, a), (_, b) = self.both("GET", "/api/uploads")
        listed = lambda doc: [(u["name"], u["bytes"], u.get("lines")) for u in doc["uploads"]]  # noqa: E731
        self.assertEqual(listed(a), listed(b))

    def test_3_training_on_uploads_reads_their_bytes(self):
        self.job("/api/train", {"files": ["corpus.txt", "image.png"], "epochs": 2})
        self.same_models("trained")
        self.job("/api/train", {"files": ["raw.png"], "whole_file": True, "epochs": 1})
        self.same_models("whole")

    def test_4_predict_and_generate_say_what_they_hold(self):
        keys = ("prefix", "prefix_symbols", "continuation", "full_text", "path", "reached_end", *DECODED)
        for body in ({"prefix": "the cat sat on"}, {"prefix": "the d"}, {"prefix": ""},
                     {"prefix": base64_prefix(b"the cat"), "symbols": True}, {"prefix": base64_text(BLOB[:6]),
                                                                               "symbols": True}):
            with self.subTest(body=body):
                request = {**body, "length": 8, "k": 3, "mode": "beam", "guard": False}
                (a, a_doc), (b, b_doc) = self.both("POST", "/api/predict", request)
                self.assertEqual((a, b), (200, 200), (a_doc, b_doc))
                self.assertEqual({k: a_doc.get(k) for k in keys}, {k: b_doc.get(k) for k in keys})
                self.assertEqual(rows(a_doc, "full_text", *DECODED), rows(b_doc, "full_text", *DECODED))
                self.assertLessEqual(abs(a_doc["cost"] - b_doc["cost"]), 1e-9)
        for body in ({"count": 3, "mode": "beam", "prefix": "the"}, {"count": 2, "mode": "beam"},
                     {"count": 2, "mode": "beam", "prefix": "dGhl", "symbols": True},
                     {"mode": "dijkstra", "prefix": "the cat"}):
            with self.subTest(body=body):
                (a, a_doc), (b, b_doc) = self.both("POST", "/api/generate", {**body, "max_length": 40, "guard": False})
                self.assertEqual((a, b), (200, 200), (a_doc, b_doc))
                sample = ("text", "full_text", "reached_end", *DECODED)
                self.assertEqual([[s[k] for k in sample] for s in a_doc["samples"]],
                                 [[s[k] for k in sample] for s in b_doc["samples"]])
                self.assertEqual(a_doc["prefix_symbols"], b_doc["prefix_symbols"])
                self.assertEqual(set(a_doc), set(b_doc))

    def test_5_scores_feedback_and_the_preview_with_and_without_symbols(self):
        said = base64_text(b"the dog sat on the log")
        for body in ({"text": "the dog sat on the log"}, {"text": said, "symbols": True}, {"text": said},
                     {"text": "qzx wqzj vbn"}):
            with self.subTest(body=body):
                (a, a_doc), (b, b_doc) = self.both("POST", "/api/score", body)
                self.assertEqual((a, b), (200, 200), (a_doc, b_doc))
                self.assertLessEqual(abs(a_doc["log_prob"] - b_doc["log_prob"]), 1e-9)
                for key in ("chars", "transitions", "unknown_transitions"):
                    self.assertEqual(a_doc[key], b_doc[key], key)
        preview = ("text", "symbols", "chars", "windows", "count", "decoded", "round_trip")
        for body in ({"text": "the cat"}, {"text": "dGhl", "symbols": True}, {"text": ""}, {"text": "the cat sat on"}):
            with self.subTest(body=body):
                (_, a_doc), (_, b_doc) = self.both("POST", "/api/encoding/preview", body)
                self.assertEqual({k: a_doc.get(k) for k in preview}, {k: b_doc.get(k) for k in preview})
                path = ("known", "labels", "node_ids", "decoded", "nodes", "compressed")
                self.assertEqual({k: a_doc["path"].get(k) for k in path}, {k: b_doc["path"].get(k) for k in path})
        self.job("/api/feedback", {"good": [base64_text(b"the cat ate the rat")], "symbols": True})
        self.job("/api/feedback", {"good": ["the dog ate the log"], "bad_text": "zzq qqz zzq\n\nqqz zzq"})
        self.same_models("feedback")
        (a, a_doc), (b, b_doc) = self.both("POST", "/api/predict", {"prefix": "the cat ate", "length": 6,
                                                                    "to_end": True, "guard": False})
        self.assertEqual(a_doc["decoded"], b_doc["decoded"])
        self.assertIn(b_doc["decoded"], ("the cat ate the rat", "the cat ate the log"))
        (a, _), (b, _) = self.both("GET", "/api/words")
        self.assertEqual((a, b), (400, 400))

    def test_6_a_text_model_reads_a_binary_upload_as_text(self):
        self.reset("char:3:1")
        self.job("/api/train", {"files": ["corpus.txt", "image.png"], "epochs": 1})
        self.same_models("text")
        (_, a_doc), (_, b_doc) = self.both("POST", "/api/predict", {"prefix": "the cat", "length": 4, "guard": False})
        self.assertEqual(a_doc["full_text"], b_doc["full_text"])
        for key in ("prefix_symbols", *DECODED):
            self.assertNotIn(key, b_doc)
        (_, a_doc), (_, b_doc) = self.both("POST", "/api/encoding/preview", {"text": "the cat"})
        self.assertNotIn("symbols", b_doc)
        self.assertEqual(a_doc["windows"], b_doc["windows"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
