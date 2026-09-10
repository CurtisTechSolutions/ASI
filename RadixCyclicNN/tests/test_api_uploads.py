"""Tests for the upload endpoints of ``radixnet.api`` and the ``files`` inputs of train / 2nrl / evolve.

Uploads are text files kept in the server's ``upload_dir``; ``POST /api/uploads``
accepts JSON (``{name, content}`` or ``{files: [...]}``), ``multipart/form-data``
and a raw body named by ``?name=``.  Train, 2NRL and evolve read them through
``files`` / ``bad_files`` / ``good_files`` / ``corpus_files``.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import CORPUS, GARBAGE, Client, start_server  # noqa: F401
except ImportError:  # ``python -m unittest tests.test_api_uploads``
    from tests.test_api import CORPUS, GARBAGE, Client, start_server  # noqa: F401


def wait_job(client, timeout=90.0):
    """Poll ``GET /api/job`` until the job leaves the running state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, job, _ = client.get("/api/job")
        assert status == 200, (status, job)
        if job and job.get("state") != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish in time")


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="radixnet-uploads-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.client, self.server, self.service = start_server(self.addCleanup, upload_dir=self.dir)

    # -- upload forms --------------------------------------------------------

    def test_json_upload_list_replace_delete(self):
        content = "\n".join(CORPUS[:5]) + "\n\n   \n"
        status, data, _ = self.client.post("/api/uploads", {"name": "corpus.txt", "content": content})
        self.assertEqual(status, 201, data)
        record = data["uploads"][0]
        self.assertEqual(record["name"], "corpus.txt")
        self.assertEqual(record["lines"], 5)
        self.assertEqual(record["chars"], len(content))
        self.assertEqual(record["bytes"], len(content.encode("utf-8")))
        self.assertFalse(record["replaced"])
        self.assertIn("modified", record)
        with open(os.path.join(self.dir, "corpus.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), content)

        status, data, _ = self.client.get("/api/uploads")
        self.assertEqual(status, 200, data)
        self.assertEqual([u["name"] for u in data["uploads"]], ["corpus.txt"])
        self.assertEqual(data["upload_dir"], os.path.abspath(self.dir))
        self.assertNotIn("replaced", data["uploads"][0])

        status, data, _ = self.client.post("/api/uploads", {"name": "corpus.txt", "content": "x y z\n"})
        self.assertEqual(status, 201, data)
        self.assertTrue(data["uploads"][0]["replaced"])
        self.assertEqual(data["uploads"][0]["lines"], 1)

        status, data, _ = self.client.post("/api/uploads/delete", {"name": "corpus.txt"})
        self.assertEqual(status, 200, data)
        self.assertEqual(data, {"deleted": "corpus.txt"})
        status, data, _ = self.client.get("/api/uploads")
        self.assertEqual(data["uploads"], [])
        status, data, _ = self.client.post("/api/uploads/delete", {"name": "corpus.txt"})
        self.assertEqual(status, 404, data)
        self.assertIn("corpus.txt", data["error"])

    def test_json_batch_upload(self):
        body = {"files": [{"name": "a.txt", "content": "one\ntwo\n"}, {"name": "b.txt", "content": "three\n"}]}
        status, data, _ = self.client.post("/api/uploads", body)
        self.assertEqual(status, 201, data)
        self.assertEqual([(u["name"], u["lines"]) for u in data["uploads"]], [("a.txt", 2), ("b.txt", 1)])
        status, data, _ = self.client.get("/api/uploads")
        self.assertEqual([u["name"] for u in data["uploads"]], ["a.txt", "b.txt"])

        status, data, _ = self.client.post("/api/uploads", {"files": []})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/uploads", {"files": ["a.txt"]})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/uploads", {"name": "c.txt"})
        self.assertEqual(status, 400, data)
        self.assertIn("content", data["error"])

    def test_raw_upload(self):
        raw = b"\xef\xbb\xbfline one\nline two\n"  # UTF-8 BOM is dropped
        status, data, _ = self.client.request(
            "POST", "/api/uploads?name=raw.txt", raw=raw, headers={"Content-Type": "text/plain; charset=utf-8"}
        )
        self.assertEqual(status, 201, data)
        record = data["uploads"][0]
        self.assertEqual((record["name"], record["lines"], record["chars"]), ("raw.txt", 2, len("line one\nline two\n")))

        status, data, _ = self.client.request(
            "POST", "/api/uploads", raw=b"no name\n", headers={"Content-Type": "application/octet-stream"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("?name=", data["error"])

    def test_multipart_upload(self):
        boundary = "----radixnet-test-boundary"
        good = "\n".join(CORPUS[:3])
        bad = "\n".join(GARBAGE[:2])
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="good.txt"\r\n'
            "Content-Type: text/plain\r\n\r\n"
            f"{good}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="note"\r\n\r\n'
            "just a form field\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="bad.txt"\r\n\r\n'
            f"{bad}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        status, data, _ = self.client.request("POST", "/api/uploads", raw=body, headers=headers)
        self.assertEqual(status, 201, data)
        self.assertEqual([(u["name"], u["lines"]) for u in data["uploads"]], [("good.txt", 3), ("bad.txt", 2)])
        with open(os.path.join(self.dir, "good.txt"), encoding="utf-8") as fh:
            self.assertEqual([line for line in fh.read().splitlines() if line.strip()], CORPUS[:3])

        no_files = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="note"\r\n\r\n'
            "only a field\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        status, data, _ = self.client.request("POST", "/api/uploads", raw=no_files, headers=headers)
        self.assertEqual(status, 400, data)
        self.assertIn("no file parts", data["error"])

    def test_upload_names_are_sanitised(self):
        status, data, _ = self.client.post("/api/uploads", {"name": "../../etc/passwd", "content": "root\n"})
        self.assertEqual(status, 201, data)
        self.assertEqual(data["uploads"][0]["name"], "passwd")
        self.assertEqual(sorted(os.listdir(self.dir)), ["passwd"])

        status, data, _ = self.client.post("/api/uploads", {"name": "C:\\Users\\me\\odd name!?.txt", "content": "x\n"})
        self.assertEqual(status, 201, data)
        self.assertEqual(data["uploads"][0]["name"], "odd name__.txt")

        for bad_name in ("", "   ", "..", "...", "\\"):
            status, data, _ = self.client.post("/api/uploads", {"name": bad_name, "content": "x\n"})
            self.assertEqual(status, 400, (bad_name, data))

        status, data, _ = self.client.post("/api/uploads/delete", {"name": "../passwd"})
        self.assertEqual(status, 200, data)
        self.assertEqual(os.listdir(self.dir), ["odd name__.txt"])

    # -- training from uploads -------------------------------------------------

    def test_train_from_files(self):
        self.client.post("/api/uploads", {"name": "corpus.txt", "content": "\n".join(CORPUS[:5]) + "\n"})
        status, data, _ = self.client.post("/api/train", {"files": ["corpus.txt"], "epochs": 1, "lr": 0.5, "batch_size": 4})
        self.assertEqual(status, 202, data)
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["trained_texts"], 5)
        self.assertEqual(stats["upload_dir"], os.path.abspath(self.dir))

    def test_train_combines_texts_and_files(self):
        self.client.post("/api/uploads", {"name": "corpus.txt", "content": "\n".join(CORPUS[:2]) + "\n"})
        body = {"texts": CORPUS[2:5], "files": ["corpus.txt"], "epochs": 1, "lr": 0.5, "batch_size": 4}
        status, data, _ = self.client.post("/api/train", body)
        self.assertEqual(status, 202, data)
        self.assertEqual(wait_job(self.client)["state"], "done")
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["trained_texts"], 5)

    def test_train_whole_file(self):
        self.client.post("/api/uploads", {"name": "poem.txt", "content": "the cat sat\non the mat\n"})
        body = {"files": ["poem.txt"], "whole_file": True, "epochs": 1, "lr": 0.5, "batch_size": 4, "texts": []}
        status, data, _ = self.client.post("/api/train", body)
        self.assertEqual(status, 202, data)
        self.assertEqual(wait_job(self.client)["state"], "done")
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["trained_texts"], 1)
        self.assertEqual(stats["trained_chars"], len("the cat sat\non the mat\n"))

    def test_train_input_errors(self):
        status, data, _ = self.client.post("/api/train", {"epochs": 1})
        self.assertEqual(status, 400, data)
        self.assertIn("'files'", data["error"])
        status, data, _ = self.client.post("/api/train", {"files": ["missing.txt"], "epochs": 1})
        self.assertEqual(status, 404, data)
        self.assertIn("missing.txt", data["error"])
        status, data, _ = self.client.post("/api/train", {"files": "corpus.txt", "epochs": 1})
        self.assertEqual(status, 400, data)
        self.client.post("/api/uploads", {"name": "blank.txt", "content": "\n\n  \n"})
        status, data, _ = self.client.post("/api/train", {"files": ["blank.txt"], "epochs": 1})
        self.assertEqual(status, 400, data)
        self.assertIn("no texts", data["error"])

    def test_two_nrl_and_evolve_from_files(self):
        self.client.post("/api/uploads", {"name": "good.txt", "content": "\n".join(CORPUS[:6]) + "\n"})
        self.client.post("/api/uploads", {"name": "bad.txt", "content": "\n".join(GARBAGE[:6]) + "\n"})
        body = {"bad_files": ["bad.txt"], "good_files": ["good.txt"], "neg_epochs": 1, "pos_epochs": 1, "batch_size": 4}
        status, data, _ = self.client.post("/api/2nrl", body)
        self.assertEqual(status, 202, data)
        self.assertEqual(wait_job(self.client)["state"], "done")
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["twonrl_runs"], 1)

        body = {"corpus_files": ["good.txt"], "generations": 1, "samples": 2, "max_length": 20, "batch_size": 4}
        status, data, _ = self.client.post("/api/evolve/start", body)
        self.assertEqual(status, 202, data)
        job = wait_job(self.client)
        self.assertEqual(job["type"], "evolve")
        self.assertEqual(job["state"], "done", job)


class UploadsDisabledTests(unittest.TestCase):
    def test_endpoints_refuse_without_upload_dir(self):
        client, _server, _service = start_server(self.addCleanup)
        status, data, _ = client.get("/api/uploads")
        self.assertEqual(status, 400, data)
        self.assertIn("uploads are disabled", data["error"])
        status, data, _ = client.post("/api/uploads", {"name": "a.txt", "content": "x\n"})
        self.assertEqual(status, 400, data)
        status, data, _ = client.post("/api/train", {"files": ["a.txt"], "epochs": 1})
        self.assertEqual(status, 400, data)
        status, stats, _ = client.get("/api/status")
        self.assertIsNone(stats["upload_dir"])
        # inline texts keep working exactly as before
        status, data, _ = client.post("/api/train", {"texts": CORPUS[:3], "epochs": 1, "lr": 0.5, "batch_size": 4})
        self.assertEqual(status, 202, data)
        self.assertEqual(wait_job(client)["state"], "done")


if __name__ == "__main__":
    unittest.main()


class ZipUploadTests(unittest.TestCase):
    """ZIP archives are unpacked on the server: every text entry becomes its own upload."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="radixnet-zips-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.client, self.server, self.service = start_server(self.addCleanup, upload_dir=self.dir)

    @staticmethod
    def archive(entries):
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, content in entries:
                z.writestr(name, content)
        return buf.getvalue()

    def sample(self):
        return self.archive([
            ("corpus/part1.txt", "\n".join(CORPUS[:4]) + "\n"),
            ("corpus/sub/part2.txt", "﻿" + "\n".join(CORPUS[4:7]) + "\n"),
            ("corpus/notes.md", "# notes\n\nabout the corpus\n"),
            ("image.png", b"\x89PNG\r\n\x1a\n\x00\x00binary"),
            ("__MACOSX/corpus/._part1.txt", "resource fork"),
            ("corpus/.DS_Store", "junk"),
            ("empty.txt", "  \n\n"),
            ("inner.zip", b"PK\x03\x04nested"),
            ("corpus/dir/", ""),
        ])

    def test_raw_zip_is_unpacked(self):
        status, data, _ = self.client.request(
            "POST", "/api/uploads?name=sample.zip", raw=self.sample(), headers={"Content-Type": "application/zip"}
        )
        self.assertEqual(status, 201, data)
        self.assertEqual(
            [(u["name"], u["lines"], u["entry"], u["archive"]) for u in data["uploads"]],
            [
                ("sample__corpus__part1.txt", 4, "corpus/part1.txt", "sample.zip"),
                ("sample__corpus__sub__part2.txt", 3, "corpus/sub/part2.txt", "sample.zip"),
                ("sample__corpus__notes.md", 2, "corpus/notes.md", "sample.zip"),
            ],
        )
        archive = data["archives"][0]
        self.assertEqual((archive["name"], archive["extracted"], archive["entries"], archive["bytes"]), ("sample.zip", 3, 9, len(self.sample())))
        self.assertEqual(
            {(s["path"], s["reason"]) for s in archive["skipped"]},
            {
                ("image.png", "binary"), ("__MACOSX/corpus/._part1.txt", "macOS metadata"), ("corpus/.DS_Store", "system file"),
                ("empty.txt", "empty"), ("inner.zip", "nested archive"), ("corpus/dir/", "directory"),
            },
        )
        self.assertEqual(sorted(os.listdir(self.dir)), sorted(u["name"] for u in data["uploads"]))  # the archive itself is not kept
        with open(os.path.join(self.dir, "sample__corpus__sub__part2.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "\n".join(CORPUS[4:7]) + "\n")  # the BOM is dropped
        status, data, _ = self.client.get("/api/uploads")
        self.assertEqual(len(data["uploads"]), 3)

    def test_json_base64_and_multipart_zip(self):
        import base64

        payload = base64.b64encode(self.sample()).decode("ascii")
        status, data, _ = self.client.post("/api/uploads", {"name": "b64.zip", "content_base64": payload})
        self.assertEqual(status, 201, data)
        self.assertEqual([u["name"] for u in data["uploads"]][:1], ["b64__corpus__part1.txt"])
        status, data, _ = self.client.post(
            "/api/uploads", {"files": [{"name": "plain.txt", "content": "a line\n"}, {"name": "batch.zip", "content_base64": payload}]}
        )
        self.assertEqual(status, 201, data)
        self.assertEqual([u["name"] for u in data["uploads"]], ["plain.txt", "batch__corpus__part1.txt", "batch__corpus__sub__part2.txt", "batch__corpus__notes.md"])
        self.assertEqual([a["name"] for a in data["archives"]], ["batch.zip"])

        boundary = "----radixnet-zip-boundary"
        body = (
            f"--{boundary}\r\n".encode("ascii")
            + b'Content-Disposition: form-data; name="file"; filename="multi.zip"\r\nContent-Type: application/zip\r\n\r\n'
            + self.sample()
            + f"\r\n--{boundary}--\r\n".encode("ascii")
        )
        status, data, _ = self.client.request(
            "POST", "/api/uploads", raw=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        self.assertEqual(status, 201, data)
        self.assertEqual(len(data["uploads"]), 3)
        self.assertTrue(all(u["name"].startswith("multi__") for u in data["uploads"]))

    def test_zip_errors_and_text_fallbacks(self):
        status, data, _ = self.client.post("/api/uploads", {"name": "x.zip", "content_base64": "@@not base64@@"})
        self.assertEqual(status, 400, data)
        self.assertIn("base64", data["error"])
        status, data, _ = self.client.post("/api/uploads", {"name": "x.zip"})
        self.assertEqual(status, 400, data)
        self.assertIn("content_base64", data["error"])
        # a ".zip" name with text bytes is just a text upload: the magic bytes decide
        status, data, _ = self.client.request(
            "POST", "/api/uploads?name=notreally.zip", raw=b"just text\n", headers={"Content-Type": "application/zip"}
        )
        self.assertEqual(status, 201, data)
        self.assertEqual((data["uploads"][0]["name"], data["uploads"][0]["lines"]), ("notreally.zip", 1))
        self.assertNotIn("archives", data)
        status, data, _ = self.client.request(
            "POST", "/api/uploads?name=empty.zip", raw=self.archive([]), headers={"Content-Type": "application/zip"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("no text files", data["error"])
        status, data, _ = self.client.request(
            "POST", "/api/uploads?name=bin.zip", raw=self.archive([("a.bin", b"\x00\x01\x02")]),
            headers={"Content-Type": "application/zip"},
        )
        self.assertEqual(status, 400, data)
        self.assertIn("a.bin: binary", data["error"])
        truncated = self.sample()[:40]
        status, data, _ = self.client.request(
            "POST", "/api/uploads?name=broken.zip", raw=truncated, headers={"Content-Type": "application/zip"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("not a valid ZIP archive", data["error"])

    def test_zip_limits(self):
        from unittest import mock

        from radixnet import archive as archive_module

        big = self.archive([("a.txt", "x" * 100 + "\n"), ("b.txt", "y" * 100 + "\n")])
        with mock.patch.object(archive_module, "MAX_UNPACKED_BYTES", 150), mock.patch.object(
            self.service, "unpack_archive", wraps=self.service.unpack_archive
        ):
            with mock.patch("radixnet.api.extract_texts", lambda data, name: archive_module.extract_texts(data, name, max_bytes=150)):
                status, data, _ = self.client.request(
                    "POST", "/api/uploads?name=big.zip", raw=big, headers={"Content-Type": "application/zip"}
                )
        self.assertEqual(status, 400, data)
        self.assertIn("limit is 150", data["error"])
        many = self.archive([(f"f{i}.txt", "line\n") for i in range(5)])
        with mock.patch("radixnet.api.extract_texts", lambda data, name: archive_module.extract_texts(data, name, max_entries=3)):
            status, data, _ = self.client.request(
                "POST", "/api/uploads?name=many.zip", raw=many, headers={"Content-Type": "application/zip"}
            )
        self.assertEqual(status, 400, data)
        self.assertIn("5 entries", data["error"])
        self.assertEqual(os.listdir(self.dir), [])

    def test_train_on_an_unpacked_archive(self):
        status, data, _ = self.client.request(
            "POST", "/api/uploads?name=sample.zip", raw=self.sample(), headers={"Content-Type": "application/zip"}
        )
        self.assertEqual(status, 201, data)
        names = [u["name"] for u in data["uploads"] if u["name"].endswith(".txt")]
        status, data, _ = self.client.post("/api/train", {"files": names, "epochs": 1, "batch_size": 8})
        self.assertEqual(status, 202, data)
        job = wait_job(self.client)
        self.assertEqual(job["state"], "done", job)
        status, data, _ = self.client.get("/api/status")
        self.assertEqual(data["trained_texts"], 7)

    def test_flat_names_are_safe(self):
        from radixnet.archive import flat_name

        self.assertEqual(flat_name("My Data (v2).zip", "../../etc/passwd"), "My Data (v2)__etc__passwd")
        self.assertEqual(flat_name("/tmp/dir/a.zip", "x\\y\\z.txt"), "a__x__y__z.txt")
        self.assertEqual(flat_name("a.ZIP", "./notes.txt"), "a__notes.txt")
        long = flat_name("a.zip", "x" * 300 + ".txt")
        self.assertLessEqual(len(long), 128)
        self.assertTrue(long.endswith(".txt"))
        self.assertNotEqual(long, flat_name("a.zip", "y" * 300 + ".txt"))
