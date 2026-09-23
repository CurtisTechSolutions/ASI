"""The Rust port's learning loops held to Python's: checkpoints, the evolve (GAN) loop, ZIP corpora, the benchmark.

Skipped when no Rust toolchain is available (``tests/rust_harness.py`` builds the binary once).

* **Checkpoints** are the Python ``CheckpointManager``'s layout, so each side lists, restores and resumes the
  other's directory; a training run that checkpoints ends with the graph Python's ends with.
* **Evolve** is Python's loop to the number: the same real texts drawn, the same fakes sampled, the same
  discriminator, generator and negative network written - only the wall-clock ``seconds`` (and the last bit of a
  loss, which the two sum in a different order) differ.  The routes answer the frontend's JSON, and the server's
  loop is the Python service's.
* **ZIP corpora** are read as Python reads them: the same entries skipped, the same texts trained on, from the
  CLI's ``--data`` and from an upload kept whole on the server (D-034).
* **bench** reports every key Python's ``run_benchmark`` does, in its order.
"""

import gzip
import io
import json
import os
import shutil
import sys
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.rust_harness import CORPUS, build, load_json, py, rust, serve, tmpdir  # noqa: E402

LOSSY = ("gen_loss", "loss", "perplexity")


def setUpModule():
    build()


def scratch(name: str) -> str:
    path = os.path.join(tmpdir(), "learn", name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def graph_of(path: str) -> dict:
    """A model file's graph (gunzipped when it is a checkpoint), without the cache stamp every load bumps."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            graph = json.load(fh)["graph"]
    else:
        graph = load_json(path)["graph"]
    graph.pop("version", None)
    return graph


def corpus_lines() -> list[str]:
    with open(CORPUS, encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


def corpus_zip(compression=zipfile.ZIP_STORED) -> bytes:
    """The sample corpus in three parts, with what an archive also holds: a picture, macOS metadata, a folder."""
    lines = corpus_lines()
    third = len(lines) // 3
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as z:
        z.writestr("part1.txt", "\n".join(lines[:third]) + "\n")
        z.writestr("more/", "")
        z.writestr("more/part2.txt", "\n".join(lines[third:2 * third]) + "\n")
        z.writestr("part3.txt", "\n".join(lines[2 * third:]) + "\n")
        z.writestr("cover.png", b"\x89PNG\x00\x00")
        z.writestr("__MACOSX/._part1.txt", b"meta")
    return buf.getvalue()


class TestRustCheckpoints(unittest.TestCase):
    """``train --checkpoint-dir`` and ``checkpoints``: the Python manager's layout, read by both sides."""

    @classmethod
    def setUpClass(cls):
        cls.dir = scratch("checkpoints")
        cls.py_model = os.path.join(cls.dir, "py.count.json")
        cls.rs_model = os.path.join(cls.dir, "rs.count.json")
        cls.py_ck = os.path.join(cls.dir, "py-ck")
        cls.rs_ck = os.path.join(cls.dir, "rs-ck")
        cls.py_doc = py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 3,
                        "--checkpoint-dir", cls.py_ck, "--checkpoint-every", 2, model=cls.py_model)
        cls.rs_doc = rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 3,
                          "--checkpoint-dir", cls.rs_ck, "--checkpoint-every", 2, model=cls.rs_model)

    def test_a_run_that_checkpoints_is_pythons_run(self):
        names = lambda doc: [c["name"] for c in doc["checkpoints"]]  # noqa: E731
        self.assertEqual(names(self.rs_doc), names(self.py_doc))
        self.assertEqual(names(self.rs_doc), ["ckpt-epoch-000002.json.gz"])
        self.assertEqual(self.rs_doc["checkpoint_dir"], os.path.abspath(self.rs_ck))
        mine, theirs = self.rs_doc["checkpoints"][0], self.py_doc["checkpoints"][0]
        self.assertEqual(set(mine), set(theirs))
        self.assertEqual((mine["step"], mine["tag"], mine["metrics"]["epoch"]), (2, "epoch", 2))
        # checkpointing trains an epoch at a time, and still ends with Python's graph
        self.assertEqual(graph_of(self.rs_model), graph_of(self.py_model))
        self.assertEqual(graph_of(mine["path"]), graph_of(theirs["path"]))
        self.assertEqual(load_json(self.rs_model)["meta"]["trained_texts"],
                         load_json(self.py_model)["meta"]["trained_texts"])

    def test_each_side_lists_and_restores_the_other(self):
        from radixnet.checkpoint import CheckpointManager

        manager = CheckpointManager(self.rs_ck)
        self.assertEqual([r["name"] for r in manager.list()], ["ckpt-epoch-000002.json.gz"])
        self.assertEqual(manager.latest()["step"], 2)
        self.assertEqual(manager.load("ckpt-epoch-000002").meta["epochs_total"], 2)
        listing = rust("checkpoints", "--dir", self.py_ck, model=self.rs_model)
        self.assertEqual([r["name"] for r in listing["checkpoints"]], ["ckpt-epoch-000002.json.gz"])
        self.assertEqual(listing["latest"]["name"], "ckpt-epoch-000002.json.gz")
        self.assertEqual(listing["directory"], os.path.abspath(self.py_ck))
        out = os.path.join(self.dir, "restored.json")
        restored = rust("checkpoints", "--dir", self.py_ck, "--restore", "latest", "--out", out, model=self.rs_model)
        theirs = py("checkpoints", "--dir", self.py_ck, "--restore", "latest", "--out",
                    os.path.join(self.dir, "py-restored.json"), model=self.py_model)
        self.assertEqual(set(restored), set(theirs))
        self.assertEqual(restored["restored"]["name"], theirs["restored"]["name"])
        self.assertEqual(restored["stats"]["epochs_total"], 2)
        self.assertEqual(graph_of(out), graph_of(os.path.join(self.dir, "py-restored.json")))
        # by its bare stem too, and a name that is not there is an error
        stem = rust("checkpoints", "--dir", self.rs_ck, "--restore", "ckpt-epoch-000002", "--out",
                    os.path.join(self.dir, "stem.json"), model=self.rs_model)
        self.assertEqual(stem["restored"]["step"], 2)
        rust("checkpoints", "--dir", self.rs_ck, "--restore", "nope", model=self.rs_model, expect=1)

    def test_resume_continues_from_the_latest(self):
        ck = os.path.join(self.dir, "resume-ck")
        shutil.copytree(self.py_ck, ck)
        doc = rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 1, "--checkpoint-dir", ck, "--resume",
                   model=os.path.join(self.dir, "fresh.json"))
        self.assertEqual(doc["stats"]["epochs_total"], 3, "two from the checkpoint, one more")
        self.assertEqual([c["name"] for c in doc["checkpoints"]],
                         ["ckpt-epoch-000002.json.gz", "ckpt-epoch-000003.json.gz"])

    def test_the_options_are_checked_as_python_checks_them(self):
        model = os.path.join(self.dir, "x.json")
        for args in (("train", "--data", CORPUS, "--checkpoint-every", 2),
                     ("train", "--data", CORPUS, "--resume"),
                     ("checkpoints", "--dir", os.path.join(self.dir, "missing")),
                     ("checkpoints", "--dir", self.rs_ck, "--out", model)):
            with self.subTest(args=args):
                rust(*args, model=model, expect=1)


class TestRustEvolve(unittest.TestCase):
    """``evolve``: Python's loop, generation for generation."""

    @classmethod
    def setUpClass(cls):
        cls.dir = scratch("evolve")
        cls.base = os.path.join(cls.dir, "base.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.base)

    def run_both(self, tag: str, *args) -> tuple[dict, dict, dict]:
        paths = {}
        docs = []
        for side, runner in (("py", py), ("rs", rust)):
            model = os.path.join(self.dir, f"{tag}-{side}.count.json")
            shutil.copy(self.base, model)
            disc = os.path.join(self.dir, f"{tag}-{side}.disc.json")
            docs.append(runner("--seed", 4, "evolve", "--data", CORPUS, "--generations", 3, "--samples", 5,
                               "--real-per-generation", 4, "--max-length", 30, "--discriminator", disc, *args,
                               model=model))
            paths[side] = (model, disc, os.path.join(self.dir, f"{tag}-{side}.count.negative.json"))
        return docs[0], docs[1], paths

    def same_records(self, theirs: list[dict], mine: list[dict]) -> None:
        self.assertEqual(len(mine), len(theirs))
        for a, b in zip(theirs, mine):
            self.assertEqual(list(a), list(b), "the same keys in the same order")
            for key in a:
                if key == "seconds":
                    continue
                if key in LOSSY and a[key] is not None:
                    self.assertAlmostEqual(a[key], b[key], delta=1e-12, msg=key)
                else:
                    self.assertEqual(a[key], b[key], f"record {a.get('generation', a.get('epoch'))}: {key}")

    def same_models(self, a: str, b: str) -> None:
        p, r = graph_of(a), graph_of(b)
        self.assertEqual(p["nodes"]["labels"], r["nodes"]["labels"])
        self.assertEqual(p["nodes"]["count"], r["nodes"]["count"])
        for key in ("src", "dst", "count", "reward", "w"):
            self.assertEqual(p["edges"][key], r["edges"][key], key)
        self.assertEqual(p["rng_state"], r["rng_state"])

    def test_every_blatant_mode_is_pythons(self):
        for mode, extra in (("none", ()), ("fail_invert", ()), ("activation", ("--blatant-margin", "0.2")),
                            ("state", ())):
            with self.subTest(mode=mode):
                theirs, mine, paths = self.run_both(mode, "--blatant-mode", mode, *extra)
                self.assertEqual(list(mine), list(theirs), "the command's document")
                self.same_records(theirs["generations"], mine["generations"])
                self.assertEqual(mine["generation"], 3)
                self.assertEqual({k: v for k, v in mine["config"].items() if k != "strength"}, theirs["config"])
                self.same_models(paths["py"][0], paths["rs"][0])
                self.same_models(paths["py"][1], paths["rs"][1])  # the discriminator
                self.same_records(load_json(paths["py"][0])["history"], load_json(paths["rs"][0])["history"])

    def test_the_critic_teaches_the_negative_network_as_pythons_does(self):
        theirs, mine, paths = self.run_both("blame", "--blatant-mode", "fail_invert", "--blatant-margin", "0.2",
                                            "--blame")
        self.same_records(theirs["generations"], mine["generations"])
        self.assertTrue(any(r["negative_blamed"] for r in mine["generations"]))
        self.assertEqual(mine["negative"]["reasons"], theirs["negative"]["reasons"])
        p, r = graph_of(paths["py"][2]), graph_of(paths["rs"][2])
        for key in ("src", "dst"):
            self.assertEqual(p["edges"][key], r["edges"][key], key)
        self.assertEqual(p["nodes"]["labels"], r["nodes"]["labels"])
        # and Python reads the negative network Rust taught
        from radixnet.model import load_model

        negative = load_model(paths["rs"][2])
        self.assertEqual(negative.kind, "negative")
        self.assertEqual(negative.reasons(), theirs["negative"]["reasons"])

    def test_it_checkpoints_every_n_generations(self):
        model = os.path.join(self.dir, "ck.count.json")
        shutil.copy(self.base, model)
        ck = os.path.join(self.dir, "ck")
        doc = rust("evolve", "--data", CORPUS, "--generations", 4, "--samples", 3, "--real-per-generation", 2,
                   "--checkpoint-dir", ck, "--checkpoint-every", 2, model=model)
        self.assertEqual(doc["checkpoint_dir"], os.path.abspath(ck))
        from radixnet.checkpoint import CheckpointManager

        manager = CheckpointManager(ck)
        self.assertEqual([r["name"] for r in manager.list()], ["ckpt-gen-000002.json.gz", "ckpt-gen-000004.json.gz"])
        latest = manager.latest()
        self.assertEqual((latest["step"], latest["tag"], latest["metrics"]["generation"]), (4, "gen", 4))
        self.assertEqual(manager.load(latest["name"]).kind, "count")

    def test_what_it_refuses(self):
        model = os.path.join(self.dir, "refused.json")
        shutil.copy(self.base, model)
        for args in (("evolve",), ("evolve", "--data", CORPUS, "--blatant-mode", "sideways"),
                     ("evolve", "--data", CORPUS, "--max-length", 2),
                     ("evolve", "--data", CORPUS, "--checkpoint-every", 1)):
            with self.subTest(args=args):
                rust(*args, model=model, expect=1)


class TestRustZipCorpus(unittest.TestCase):
    """``--data corpus.zip``: the same entries read, the same entries skipped, the same graph."""

    def test_an_archive_trains_the_graph_python_trains(self):
        tmp = scratch("zip")
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            with self.subTest(compression=compression):
                archive = os.path.join(tmp, f"corpus-{compression}.zip")
                with open(archive, "wb") as fh:
                    fh.write(corpus_zip(compression))
                py_model = os.path.join(tmp, f"py-{compression}.count.json")
                rs_model = os.path.join(tmp, f"rs-{compression}.count.json")
                py("--kind", "count", "--seed", 1, "train", "--data", archive, "--epochs", 2, model=py_model)
                doc = rust("--seed", 1, "train", "--data", archive, "--epochs", 2, model=rs_model)
                self.assertEqual(doc["texts"], len(corpus_lines()))
                self.assertEqual(graph_of(rs_model), graph_of(py_model))
                self.assertEqual(load_json(rs_model)["meta"]["trained_texts"],
                                 load_json(py_model)["meta"]["trained_texts"])

    def test_an_archive_under_any_name_is_an_archive(self):
        tmp = scratch("zip-name")
        disguised = os.path.join(tmp, "corpus.txt")
        with open(disguised, "wb") as fh:
            fh.write(corpus_zip(zipfile.ZIP_DEFLATED))
        doc = rust("--seed", 1, "train", "--data", disguised, "--epochs", 1, model=os.path.join(tmp, "m.json"))
        self.assertEqual(doc["texts"], len(corpus_lines()))
        whole = rust("--seed", 1, "train", "--data", disguised, "--split", "file", "--epochs", 1,
                     model=os.path.join(tmp, "w.json"))
        self.assertEqual(whole["texts"], 3, "one text per text entry")


class TestRustBench(unittest.TestCase):
    def test_it_reports_every_key_python_does_in_its_order(self):
        from radixnet.bench import RESULT_KEYS

        doc = rust("bench", "--chars", 2000, "--epochs", 1, "--predictions", 20, "--workers", 1,
                   model=os.path.join(tmpdir(), "bench-unused.json"))
        self.assertEqual(list(doc)[:len(RESULT_KEYS)], list(RESULT_KEYS))
        for key in ("seed", "trigrams", "mean_fanout", "loss_first", "loss_last", "epoch_seconds",
                    "backend_transitions_per_sec", "dijkstra_expansions"):
            self.assertIn(key, doc)
        self.assertEqual((doc["backend"], doc["epochs"], doc["predict_count"]), ("rust", 1, 20))
        self.assertGreaterEqual(doc["chars"], 2000)
        self.assertEqual(set(doc["sample_prediction"]), {"prefix", "continuation", "cost", "reached_end"})


class TestRustLearnServer(unittest.TestCase):
    """The routes: uploads kept whole, checkpoints, the evolve job - the JSON the frontend reads."""

    @classmethod
    def setUpClass(cls):
        cls.dir = scratch("server")
        cls.model = os.path.join(cls.dir, "model.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=cls.model)
        cls.snapshot = os.path.join(cls.dir, "snapshot.count.json")
        shutil.copy(cls.model, cls.snapshot)
        cls.uploads = os.path.join(cls.dir, "uploads")
        cls.ck = os.path.join(cls.dir, "ck")
        os.makedirs(cls.uploads)
        cls.server = serve(cls, cls.model, "--seed", "1", "--upload-dir", cls.uploads, "--checkpoint-dir", cls.ck)

    def status(self) -> dict:
        status, doc, _ = self.server.get("/api/status")
        self.assertEqual(status, 200)
        return doc

    def train(self, body: dict) -> dict:
        status, doc, _ = self.server.post("/api/train", body)
        self.assertEqual(status, 202, doc)
        job = self.server.wait_job()
        self.assertEqual(job["state"], "done", job)
        return job

    def test_1_the_routes_are_listed_for_the_frontend(self):
        routes = self.status()["routes"]
        for route in ("GET /api/checkpoints", "POST /api/checkpoints/save", "POST /api/checkpoints/restore",
                      "POST /api/evolve/start", "POST /api/evolve/stop", "GET /api/evolve/history"):
            self.assertIn(route, routes)

    def test_2_an_evolve_job_is_the_python_services_loop(self):
        from radixnet.api import ModelService
        from radixnet.gan import EvolveConfig

        status, doc, _ = self.server.post("/api/evolve/stop", {})
        self.assertEqual(status, 404, doc)
        body = {"corpus": corpus_lines(), "generations": 2, "samples": 4, "real_per_generation": 3,
                "max_length": 30, "blatant_mode": "activation", "blatant_margin": 0.2, "checkpoint_every": 0}
        status, doc, _ = self.server.post("/api/evolve/start", body)
        self.assertEqual(status, 202, doc)
        self.assertEqual((doc["job"]["type"], doc["job"]["state"]), ("evolve", "running"))
        self.assertEqual((doc["generations"], doc["corpus"]), (2, len(corpus_lines())))
        job = self.server.wait_job()
        self.assertEqual((job["type"], len(job["history"])), ("evolve", 2), job)
        status, history, _ = self.server.get("/api/evolve/history")
        self.assertEqual(status, 200)
        mine = history["history"]
        self.assertEqual(len(mine), 2)
        for key in ("generation", "gap", "fake_score_mean", "real_score_mean", "sample", "failures", "mode"):
            self.assertIn(key, mine[0])
        # the Python service runs the same loop over the same model
        svc = ModelService(self.snapshot, backend="python", seed=1)
        config = EvolveConfig(samples=4, real_per_generation=3, max_length=30, blatant_mode="activation",
                              blatant_margin=0.2)
        svc.start_evolve(corpus_lines(), 2, config)
        svc.job.thread.join(60)
        theirs = svc.evolve_history()["history"]
        self.assertEqual(len(theirs), 2)
        for a, b in zip(theirs, mine):
            for key in a:
                if key == "seconds":
                    continue
                if key in LOSSY:
                    self.assertAlmostEqual(a[key], b[key], delta=1e-12, msg=key)
                else:
                    self.assertEqual(a[key], b[key], key)
        # a stop now finds the finished evolve job; a run with nothing to learn from is refused
        status, doc, _ = self.server.post("/api/evolve/stop", {})
        self.assertEqual((status, doc["type"]), (200, "evolve"))
        status, doc, _ = self.server.post("/api/evolve/start", {"generations": 1})
        self.assertEqual(status, 400)
        status, doc, _ = self.server.post("/api/evolve/start", {"corpus": corpus_lines(), "blatant_mode": "x"})
        self.assertEqual(status, 400)

    def test_3_uploads_keep_an_archive_whole(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a/one.txt", "the cat sat on the mat\nthe cat ran to the door\n")
            z.writestr("two.txt", "the dog sat on the log\n")
            z.writestr("img.png", b"\x89PNG\x00\x00")
        with open(os.path.join(self.uploads, "corpus.zip"), "wb") as fh:
            fh.write(buf.getvalue())
        with open(os.path.join(self.uploads, "notes.txt"), "w", encoding="utf-8") as fh:
            fh.write("para one line a\npara one line b\n\npara two\n")
        status, listing, _ = self.server.get("/api/uploads")
        self.assertEqual(status, 200)
        self.assertEqual([u["name"] for u in listing["uploads"]], ["corpus.zip", "notes.txt"])
        record = listing["uploads"][0]
        self.assertEqual((record["archive"], record["files"], record["lines"], record["skipped"]), (True, 2, 3, 1))
        for key in ("bytes", "chars", "modified"):
            self.assertIn(key, record)
        before = self.status()["trained_texts"]
        self.train({"files": ["corpus.zip"], "epochs": 1})
        self.assertEqual(self.status()["trained_texts"] - before, 3)
        self.train({"files": ["corpus.zip"], "whole_file": True, "epochs": 1})
        self.assertEqual(self.status()["trained_texts"] - before, 3 + 2, "one text per entry")
        # a large deflated archive: 24 000 lines in three entries
        big = io.BytesIO()
        with zipfile.ZipFile(big, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for part in range(3):
                z.writestr(f"vol{part}.txt",
                           "".join(f"the cat number {i % 53} sat on mat {(i * 7) % 61}\n" for i in range(8000)))
        with open(os.path.join(self.uploads, "big.zip"), "wb") as fh:
            fh.write(big.getvalue())
        status, listing, _ = self.server.get("/api/uploads")
        record = next(u for u in listing["uploads"] if u["name"] == "big.zip")
        self.assertEqual((record["files"], record["lines"]), (3, 24000))
        later = self.status()["trained_texts"]
        self.train({"files": ["big.zip"], "epochs": 1})
        self.assertEqual(self.status()["trained_texts"] - later, 24000)
        # a corrupt archive is a 400, not a crash
        with open(os.path.join(self.uploads, "broken.zip"), "wb") as fh:
            fh.write(b"PK\x03\x04 and nothing else")
        status, doc, _ = self.server.post("/api/train", {"files": ["broken.zip"], "epochs": 1})
        self.assertEqual(status, 400, doc)
        status, listing, _ = self.server.get("/api/uploads")
        self.assertIn("error", next(u for u in listing["uploads"] if u["name"] == "broken.zip"))

    def test_4_checkpoints_in_the_python_managers_layout(self):
        from radixnet.checkpoint import CheckpointManager

        status, ck, _ = self.server.post("/api/checkpoints/save", {"tag": "manual"})
        self.assertEqual(status, 200, ck)
        self.assertTrue(ck["name"].startswith("ckpt-manual-") and ck["name"].endswith(".json.gz"))
        self.assertEqual(set(ck), {"name", "path", "step", "tag", "metrics", "saved_at", "bytes"})
        status, listing, _ = self.server.get("/api/checkpoints")
        self.assertEqual(listing["latest"]["name"], ck["name"])
        manager = CheckpointManager(self.ck)
        self.assertIn(ck["name"], [r["name"] for r in manager.list()])
        epochs = self.status()["epochs_total"]
        self.assertEqual(manager.load(ck["name"]).stats()["epochs_total"], epochs)
        # training that checkpoints every epoch
        self.train({"texts": corpus_lines(), "epochs": 2, "checkpoint_every": 1})
        names = [r["name"] for r in manager.list()]
        for step in (epochs + 1, epochs + 2):
            self.assertIn(f"ckpt-epoch-{step:06d}.json.gz", names)
        # restoring puts the model back where the checkpoint left it
        status, restored, _ = self.server.post("/api/checkpoints/restore", {"name": ck["name"]})
        self.assertEqual(status, 200, restored)
        self.assertEqual(restored["epochs_total"], epochs)
        self.assertEqual(self.status()["epochs_total"], epochs)
        status, doc, _ = self.server.post("/api/checkpoints/restore", {"name": ""})
        self.assertEqual(status, 400)
        status, doc, _ = self.server.post("/api/checkpoints/restore", {"name": "ckpt-nope-000001"})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
