"""The Rust port's radix and resonant kinds against ``radixnet/model.py`` and ``radixnet/resonance.py``.

Both implementations train each kind on ``data/sample_corpus.txt`` with the same
seed and settings and must write the same model document - byte for byte but
for the ``version`` cache stamp, the timestamps, the epoch timings and the
``backend`` field, which names the implementation that ran the learning rule
(``python`` there, ``rust`` here).  They must then make the same predictions in
every mode, generate the same texts, give the same scores, learn the same from
feedback and 2NRL, and each continue the other's file into the same model.
The schedule language previews the same rates and refuses the same
expressions in the same words, and the server selects, resets and reports the
kinds in the JSON the frontend reads.

See ``rust/README.md`` and ``rust/src/kinds.rs``.
"""

import json
import os
import shutil
import sys
import unittest

from tests.rust_harness import CORPUS, GARBAGE, build, load_json, py, rust, serve, tmpdir

KINDS = ("radix", "resonant")
MODES = {"radix": ("dijkstra", "beam"), "resonant": ("kbest", "dijkstra", "beam")}
GENERATE_MODES = {"radix": ("beam", "dijkstra", "sample"), "resonant": ("kbest", "beam", "dijkstra", "sample")}


def setUpModule():
    build()


def normalised(doc: dict) -> dict:
    """The document minus what legitimately differs between two runs and two implementations."""
    doc = dict(doc)
    for key in ("version", "saved_at", "backend"):
        doc.pop(key, None)
    doc["meta"] = {k: v for k, v in doc["meta"].items() if k != "created"}
    doc["history"] = [{k: v for k, v in r.items() if k != "seconds"} for r in doc["history"]]
    doc["graph"] = {k: v for k, v in doc["graph"].items() if k not in ("version", "version_resets")}
    return doc


def compact(doc: dict) -> str:
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


class TestRustKindsParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(tmpdir(), "kinds")
        os.makedirs(cls.root, exist_ok=True)
        cls.models = {}
        for kind in KINDS:
            pm = os.path.join(cls.root, f"py.{kind}.json")
            rm = os.path.join(cls.root, f"rs.{kind}.json")
            py("--kind", kind, "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=pm)
            rust("--kind", kind, "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=rm)
            cls.models[kind] = (pm, rm)

    def copies(self, kind: str, tag: str, source: str | None = None) -> tuple[str, str]:
        src = source or self.models[kind][0]
        a = os.path.join(self.root, f"{tag}.py.{kind}.json")
        b = os.path.join(self.root, f"{tag}.rs.{kind}.json")
        shutil.copy(src, a)
        shutil.copy(src, b)
        return a, b

    def assert_same_model(self, a_path: str, b_path: str):
        a, b = normalised(load_json(a_path)), normalised(load_json(b_path))
        self.assertEqual(a["graph"]["nodes"]["labels"], b["graph"]["nodes"]["labels"])
        self.assertEqual(a["graph"]["edges"]["w"], b["graph"]["edges"]["w"])  # the same doubles
        self.assertEqual(compact(a), compact(b))

    def test_the_documents_are_pythons_byte_for_byte(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                pm, rm = self.models[kind]
                p, r = load_json(pm), load_json(rm)
                self.assertEqual(r["format"], {"radix": "radixnet", "resonant": "radixnet-resonant"}[kind])
                self.assertEqual(r["backend"], "rust")
                self.assertEqual(list(p), list(r))
                self.assertEqual(compact(normalised(p)), compact(normalised(r)))
                # and rendered as Python renders it: every float, every key, no whitespace
                with open(rm, encoding="utf-8") as fh:
                    raw = fh.read()
                self.assertEqual(raw, compact(json.loads(raw)))

    def test_the_negative_kind_trains_by_blaming(self):
        # `--kind negative train` is an ordinary kind in Python: training it *is* blaming
        a = os.path.join(self.root, "neg.py.negative.json")
        b = os.path.join(self.root, "neg.rs.negative.json")
        for path in (a, b):
            if os.path.exists(path):
                os.remove(path)
        py("--kind", "negative", "--seed", 1, "train", "--data", GARBAGE, "--epochs", 2, model=a)
        rust("--kind", "negative", "--seed", 1, "train", "--data", GARBAGE, "--epochs", 2, model=b)
        x, y = load_json(a), load_json(b)
        self.assertEqual(y["format"], "radixnet-negative")
        for doc in (x, y):
            for entry in doc.get("log", []):
                entry.pop("at", None)
        self.assertEqual(compact(normalised(x)), compact(normalised(y)))

    def test_predictions_agree_in_every_mode(self):
        keys = ("continuation", "full_text", "cost", "step_costs", "path", "node_ids", "reached_end")
        for kind in KINDS:
            model = self.models[kind][0]
            for mode in MODES[kind]:
                for prefix in ("", "the ", "the cat", "q", "xyz"):
                    for extra in ([], ["--max-length", 30], ["--to-end"], ["--traversal", "punishment"]):
                        args = ["predict", "--prefix", prefix, "--mode", mode, "--length", 12, "--k", 3, *extra]
                        with self.subTest(kind=kind, mode=mode, prefix=prefix, extra=extra):
                            a = py(*args, model=model)
                            b = rust(*args, model=model)
                            for key in keys:
                                self.assertEqual(a[key], b[key], key)
                            for side in ("top", "bottom"):
                                if side in a:
                                    self.assertEqual([(t["text"], t["cost"]) for t in a[side]],
                                                     [(t["text"], t["cost"]) for t in b[side]], side)

    def test_generations_and_scores_agree(self):
        for kind in KINDS:
            model = self.models[kind][0]
            for mode in GENERATE_MODES[kind]:
                for prefix in ("", "the "):
                    with self.subTest(kind=kind, mode=mode, prefix=prefix):
                        args = ["generate", "--mode", mode, "--count", 3, "--prefix", prefix, "--max-length", 40]
                        a = py("--seed", 5, *args, model=model)
                        b = rust("--seed", 5, *args, "--seeded", model=model)
                        self.assertEqual([(s["text"], s["cost"]) for s in a["samples"]],
                                         [(s["text"], s["cost"]) for s in b["samples"]])
            for text in ("the cat sat on the mat", "zzz qqq", "the"):
                with self.subTest(kind=kind, text=text):
                    a = py("score", "--text", text, model=model)
                    b = rust("score", "--text", text, model=model)
                    x, y = a["results"][0], b["results"][0]
                    for key in ("log_prob", "per_char", "chars", "transitions", "unknown_transitions"):
                        self.assertEqual(x[key], y[key], key)

    def test_feedback_and_2nrl_learn_the_same(self):
        good = os.path.join(self.root, "good3.txt")
        bad = os.path.join(self.root, "bad3.txt")
        with open(good, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\nthe dog ran in the park\na bird sang a song\n")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("the the the the\nmat mat on on on\nsat sat sat cat\n")
        commands = (
            ["feedback", "--good", CORPUS, "--bad", GARBAGE],
            ["feedback", "--good", CORPUS],
            ["feedback", "--bad", GARBAGE],
            ["feedback", "--good", good, "--good-ratings", "8,10,6", "--bad", bad, "--bad-ratings", "3,0,9"],
            ["feedback", "--bad", bad, "--bad-ratings", "10,2,2", "--neg-lr", 0.3, "--batch-size", 2],
            ["2nrl", "--bad", GARBAGE, "--good", CORPUS],
            ["invert"],
            ["compress"],
        )
        for kind in KINDS:
            for command in commands:
                with self.subTest(kind=kind, command=" ".join(map(str, command[:3]))):
                    a, b = self.copies(kind, "fb")
                    x = py(*command, model=a)
                    y = rust(*command, model=b)
                    self.assert_same_model(a, b)
                    if "action" in x:
                        self.assertEqual(x["action"], y["action"])
                        self.assertEqual(len(x["negative"]), len(y["negative"]))
                        self.assertEqual(len(x["positive"]), len(y["positive"]))

    def test_each_side_continues_the_others_file(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                pm, rm = self.models[kind]
                a = os.path.join(self.root, f"cont.py.{kind}.json")
                b = os.path.join(self.root, f"cont.rs.{kind}.json")
                shutil.copy(rm, a)  # Python continues the Rust file
                shutil.copy(pm, b)  # and Rust the Python one
                py("train", "--data", CORPUS, "--epochs", 1, model=a)
                rust("train", "--data", CORPUS, "--epochs", 1, model=b)
                self.assert_same_model(a, b)
                # a loaded file's own kind wins over --kind
                c = rust("--kind", "count", "info", model=b)
                self.assertEqual(c["stats"]["kind"], kind)

    def test_schedules_train_and_preview_the_same(self):
        a = os.path.join(self.root, "sched.py.radix.json")
        b = os.path.join(self.root, "sched.rs.radix.json")
        for path in (a, b):
            if os.path.exists(path):
                os.remove(path)
        args = ["--kind", "radix", "--seed", 3, "train", "--data", CORPUS, "--epochs", 4, "--lr", 0.2,
                "--lr-schedule", "cosine(lr0, lr0 / 10)", "--act-lr-schedule", "lr / 10", "--batch-size", 64,
                "--reverse-schedule"]
        py(*args, model=a)
        rust(*args, model=b)
        self.assert_same_model(a, b)
        model = os.path.join(self.root, "none.json")
        for expressions in (["--lr-schedule", "linear(lr0, 4 * lr0)"],
                            ["--lr-schedule", "step(lr0, 0.5, 2)", "--act-lr-schedule", "lr * t + act_lr0"],
                            ["--lr-schedule", "warmup(0, lr0, 3) if epoch < 5 else lr0 ** 2", "--reverse-schedule"],
                            ["--lr-schedule", "geometric(lr0, lr0 / 100)", "--epochs", 7, "--lr", 0.3],
                            ["--act-lr-schedule", "max(1e-4, cos(pi * t) * lr / 3)", "--epochs", 1]):
            with self.subTest(expressions=expressions):
                x = py("schedule", *expressions, model=model)
                y = rust("schedule", *expressions, model=model)
                self.assertEqual(x, y)
        self.assertEqual(py("schedule", model=model), rust("schedule", model=model))
        for bad in ("lr0 +", "__import__('os')", "lr0 / (t - t)", "foo(1)", "[1, 2]", "0x1g", "(lr0"):
            with self.subTest(bad=bad):
                x = py("schedule", "--lr-schedule", bad, model=model, expect=1)["error"]
                y = rust("schedule", "--lr-schedule", bad, model=model, expect=1)["error"]
                said = x.split("error: ", 1)[-1]
                self.assertIn(said, y)

    def test_the_resonant_weights_command(self):
        a, b = self.copies("resonant", "w")
        args = ["weights", "--buckets", 12, "--kick-scale", 0.5, "--concentration", 1]
        x = py(*args, model=a)
        y = rust(*args, model=b)
        self.assertEqual(x["weights"], y["weights"])
        self.assertEqual(x["changed"], y["changed"])
        self.assert_same_model(a, b)
        for kind, args in (("radix", ["weights", "--reward-scale", 2]), ("resonant", ["weights", "--window", 5])):
            with self.subTest(kind=kind):
                x = py(*args, model=self.models[kind][0], expect=1)["error"]
                y = rust(*args, model=self.models[kind][0], expect=1)["error"]
                tail = x.split("; ", 1)[-1]
                self.assertIn(tail, y)


class TestRustKindsServer(unittest.TestCase):
    """Selecting, resetting and reporting the kinds: the JSON the frontend reads, Python's keys."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from radixnet.api import ModelService

        cls.root = os.path.join(tmpdir(), "kinds-server")
        os.makedirs(cls.root, exist_ok=True)
        cls.rs_path = os.path.join(cls.root, "rs.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 1, model=cls.rs_path)
        cls.server = serve(cls, cls.rs_path)
        py_path = os.path.join(cls.root, "py.count.json")
        shutil.copy(cls.rs_path, py_path)
        cls.python = ModelService(model_path=py_path, seed=0, quiet=True)

    def test_status_and_model_carry_pythons_keys(self):
        for kind in ("resonant", "radix", "count"):
            with self.subTest(kind=kind):
                self.server.post("/api/model/select", {"kind": kind})
                self.python.select_kind(kind)
                self.assert_status_like_pythons()

    def assert_status_like_pythons(self):
        _, status, _ = self.server.get("/api/status")
        python = self.python.status()
        missing = {"ollama", "chatgpt", "tools"}  # other areas' blocks
        self.assertEqual(set(python) - set(status) - missing, set())
        self.assertEqual(status["kinds"], python["kinds"])
        self.assertEqual(status["backends"]["torch"], False)
        _, model, _ = self.server.get("/api/model")
        described = self.python.describe_model()
        self.assertEqual(set(described) - set(model), set())
        self.assertEqual(set(described["paths"]) - set(model["paths"]), set())

    def test_select_and_reset_answer_as_pythons_do(self):
        for kind in ("radix", "resonant", "negative", "count"):
            with self.subTest(kind=kind):
                status, chosen, _ = self.server.post("/api/model/select", {"kind": kind})
                self.assertEqual(status, 200, chosen)
                expected = self.python.select_kind(kind)
                self.assertEqual(set(expected) - set(chosen), set())
                self.assertEqual(chosen["kind"], expected["kind"])
                self.assertEqual(chosen["label"], expected["label"])
                self.assertEqual(chosen["origin"], expected["origin"])
                self.assertEqual(set(expected["stats"]) - set(chosen["stats"]) - {"device", "counting"}, set())
        # a reset takes the kind and its score function
        status, _, _ = self.server.post("/api/reset", {"kind": "resonant", "buckets": 6, "concentration": 3})
        self.assertEqual(status, 200)
        _, model, _ = self.server.get("/api/model")
        self.python.reset(kind="resonant", buckets=6, concentration=3.0)
        self.assertEqual(model["weights"], self.python.describe_model()["weights"])
        status, error, _ = self.server.post("/api/reset", {"kind": "radix", "reward_scale": 2})
        self.assertEqual(status, 400)
        self.assertEqual(error["error"], "weight options (reward_scale) do not apply to the radix model")
        status, error, _ = self.server.post("/api/model/select", {"kind": "sine"})
        self.assertEqual(status, 400)
        self.assertIn("unknown model kind 'sine'", error["error"])
        self.server.post("/api/model/select", {"kind": "count"})
        self.python.select_kind("count")

    def test_the_schedule_routes_are_pythons(self):
        from radixnet.schedule import describe, preview_points

        _, described, _ = self.server.get("/api/schedule")
        self.assertEqual(described, json.loads(json.dumps(describe())))
        body = {"lr_schedule": "cosine(lr0, lr0 / 10)", "act_lr_schedule": "lr / 10", "epochs": 6, "lr": 0.1,
                "act_lr": 0.01, "reverse_schedule": True}
        status, preview, _ = self.server.post("/api/schedule/preview", body)
        self.assertEqual(status, 200, preview)
        self.assertEqual(preview["points"], preview_points(body["lr_schedule"], body["act_lr_schedule"], 6, 0.1,
                                                           0.01, reverse=True))
        status, error, _ = self.server.post("/api/schedule/preview", {"lr_schedule": "import os"})
        self.assertEqual(status, 400)
        self.assertIn("invalid syntax", error["error"])


if __name__ == "__main__":
    unittest.main()
