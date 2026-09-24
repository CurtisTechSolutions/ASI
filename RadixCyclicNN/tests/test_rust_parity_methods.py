"""Rust parity of the search and training methods (``SPEC-SearchAndTraining.md``).

Every kind that learns by walking a list of texts - the count, sine and phase
models - is trained by both implementations with every training setting (the
order, the curriculum, the replay buffer and early stopping) and must write the
same graph, byte for byte, the same history and the same ``replay`` block at the
end of the same document.  Then both search the models they wrote with the
sampling filters and the diverse beam and must return the same texts at the
same costs.  The server takes the same settings and refuses the same values.
"""

import json
import os
import unittest

from tests.rust_harness import CORPUS, build, load_json, py, rust, serve, tmpdir

KINDS = ("count", "radix", "resonant")

# (name, settings of a new model, flags of the first run, [(corpus, flags) of the runs after it])
PLANS = (
    ("curriculum", [], ["--epochs", 4, "--order", "shortest-first", "--curriculum", 0.3], []),
    ("shuffle", [], ["--epochs", 3, "--order", "shuffle"], []),
    ("patience", [], ["--epochs", 8, "--order", "longest-first", "--patience", 1, "--min-delta", 0.5], []),
    ("replay-size", [], ["--epochs", 2, "--replay-size", 7], []),
    ("rehearsal", [], ["--epochs", 2, "--replay-size", 6], [
        ("second", ["--epochs", 3, "--replay", 0.4, "--order", "shuffle", "--curriculum", 0.5]),
        ("first", ["--epochs", 2, "--replay", 2.5, "--replay-size", 4]),
    ]),
    ("words", ["--encoding", "word:2:1"], ["--epochs", 3, "--order", "shortest-first", "--curriculum", 0.5,
                                           "--replay-size", 5], []),
)

SAMPLES = (
    ["--temperature", 1.3, "--top-k", 2],
    ["--temperature", 1.0, "--top-p", 0.6],
    ["--temperature", 2.0, "--min-p", 0.3],
    ["--temperature", 1.5, "--top-k", 3, "--top-p", 0.9, "--min-p", 0.05],
)


def compact(doc):
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


def setUpModule():
    build()


def halves():
    """The corpus cut in two: the texts the later runs train on."""
    with open(CORPUS, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    out = {}
    for name, part in (("first", lines[: len(lines) // 2]), ("second", lines[len(lines) // 2:])):
        path = os.path.join(tmpdir(), f"methods-{name}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(part) + "\n")
        out[name] = path
    return out


class TestRustTrainingPlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        parts = halves()
        for kind in KINDS:
            for name, new, first, later in PLANS:
                pm = os.path.join(tmpdir(), f"methods-{kind}-{name}.py.json")
                rm = os.path.join(tmpdir(), f"methods-{kind}-{name}.rs.json")
                py("--kind", kind, "--seed", 3, *new, "train", "--data", CORPUS, *first, model=pm)
                rust("--kind", kind, "--seed", 3, *new, "train", "--data", CORPUS, *first, model=rm)
                for corpus, flags in later:
                    py("train", "--data", parts[corpus], *flags, model=pm)
                    rust("train", "--data", parts[corpus], *flags, model=rm)
                cls.models[(kind, name)] = (load_json(pm), load_json(rm))

    def test_the_same_graph_byte_for_byte(self):
        for (kind, name), (a, b) in self.models.items():
            with self.subTest(kind=kind, plan=name):
                strip = lambda doc: compact({k: v for k, v in doc["graph"].items() if k != "version"})  # noqa: E731
                self.assertEqual(strip(a), strip(b))
                self.assertEqual(list(a), list(b))  # the same document, key for key: `replay` last

    def test_the_same_history_and_counters(self):
        for (kind, name), (a, b) in self.models.items():
            with self.subTest(kind=kind, plan=name):
                self.assertEqual([list(r) for r in a["history"]], [list(r) for r in b["history"]])
                for x, y in zip(a["history"], b["history"]):
                    self.assertEqual((x["epoch"], x["transitions"], x["skipped_short"], x.get("early_stop")),
                                     (y["epoch"], y["transitions"], y["skipped_short"], y.get("early_stop")))
                    self.assertLessEqual(abs(x["loss"] - y["loss"]), 1e-9)
                for key in ("epochs_total", "trained_texts", "trained_chars"):
                    self.assertEqual(a["meta"][key], b["meta"][key], key)

    def test_the_same_replay_block(self):
        for (kind, name), (a, b) in self.models.items():
            with self.subTest(kind=kind, plan=name):
                self.assertEqual(compact(a.get("replay")), compact(b.get("replay")))
        a, b = self.models[("count", "rehearsal")]
        self.assertEqual((a["replay"]["size"], len(a["replay"]["texts"])), (4, 4))

    def test_the_settings_did_something(self):
        for kind in KINDS:
            a, _ = self.models[(kind, "patience")]
            self.assertTrue(a["history"][-1].get("early_stop"), kind)
            self.assertLess(len(a["history"]), 8)
            self.assertNotIn("replay", self.models[(kind, "shuffle")][0])
            self.assertEqual(len(self.models[(kind, "replay-size")][0]["replay"]["texts"]), 7)

    def test_each_side_continues_the_others_buffer(self):
        pm = os.path.join(tmpdir(), "methods-cross.py.json")
        rm = os.path.join(tmpdir(), "methods-cross.rs.json")
        py("--kind", "count", "--seed", 8, "train", "--data", CORPUS, "--epochs", 1, "--replay-size", 5, model=pm)
        rust("--kind", "count", "--seed", 8, "train", "--data", CORPUS, "--epochs", 1, "--replay-size", 5, model=rm)
        # each continues the other's file, rehearsing the buffer the other wrote
        parts = halves()
        cross_py = os.path.join(tmpdir(), "methods-cross-py-continues-rs.json")
        cross_rs = os.path.join(tmpdir(), "methods-cross-rs-continues-py.json")
        with open(rm, encoding="utf-8") as src, open(cross_py, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        with open(pm, encoding="utf-8") as src, open(cross_rs, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        py("train", "--data", parts["second"], "--epochs", 2, "--replay", 1.0, model=cross_py)
        rust("train", "--data", parts["second"], "--epochs", 2, "--replay", 1.0, model=cross_rs)
        a, b = load_json(cross_py), load_json(cross_rs)
        self.assertEqual(compact(a["replay"]), compact(b["replay"]))
        self.assertEqual(a["graph"]["edges"]["count"], b["graph"]["edges"]["count"])
        self.assertEqual([r["transitions"] for r in a["history"]], [r["transitions"] for r in b["history"]])


class TestRustSearchMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for kind in (*KINDS, "words"):
            new = ["--encoding", "word:2:1"] if kind == "words" else []
            pm = os.path.join(tmpdir(), f"methods-search-{kind}.py.json")
            rm = os.path.join(tmpdir(), f"methods-search-{kind}.rs.json")
            base = "count" if kind == "words" else kind
            py("--kind", base, "--seed", 3, *new, "train", "--data", CORPUS, "--epochs", 2, model=pm)
            rust("--kind", base, "--seed", 3, *new, "train", "--data", CORPUS, "--epochs", 2, model=rm)
            cls.models[kind] = (pm, rm)

    @staticmethod
    def texts(doc):
        return [(s["text"], round(s["cost"], 9)) for s in doc["samples"]]

    def test_the_sampling_filters_draw_the_same_texts(self):
        for kind, (pm, rm) in self.models.items():
            for flags in SAMPLES:
                with self.subTest(kind=kind, flags=flags):
                    args = ["generate", "--mode", "sample", "--count", 6, "--max-length", 40, *flags]
                    a = py("--seed", 11, *args, model=pm)
                    b = rust("--seed", 11, *args, "--seeded", model=rm)
                    self.assertEqual(self.texts(a), self.texts(b))

    def test_the_diverse_beam_picks_the_same_texts(self):
        for kind, (pm, rm) in self.models.items():
            for flags in (["--diversity", 2.0], ["--diversity", 0.5, "--beam", 12]):
                with self.subTest(kind=kind, flags=flags):
                    args = ["generate", "--mode", "beam", "--count", 5, "--max-length", 40, *flags]
                    self.assertEqual(self.texts(py(*args, model=pm)), self.texts(rust(*args, model=rm)))
            for prefix in ("the cat", "the"):
                with self.subTest(kind=kind, prefix=prefix):
                    args = ["predict", "--prefix", prefix, "--length", 10, "--k", 4, "--mode", "beam", "--diversity", 3.0]
                    a, b = py(*args, model=pm), rust(*args, model=rm)
                    self.assertEqual(a["continuation"], b["continuation"])
                    for side in ("top", "bottom"):
                        self.assertEqual([(t["full_text"], round(t["cost"], 9)) for t in a[side]],
                                         [(t["full_text"], round(t["cost"], 9)) for t in b[side]], side)

    def test_the_settings_change_what_comes_back(self):
        _, rm = self.models["resonant"]
        plain = rust("generate", "--mode", "beam", "--count", 5, "--max-length", 40, model=rm)
        spread = rust("generate", "--mode", "beam", "--count", 5, "--max-length", 40, "--diversity", 2.0, model=rm)
        self.assertEqual(plain["samples"][0]["text"], spread["samples"][0]["text"])
        self.assertNotEqual(self.texts(plain), self.texts(spread))
        _, rm = self.models["count"]
        args = ["generate", "--mode", "sample", "--count", 6, "--max-length", 40, "--temperature", 1.3]
        self.assertNotEqual(self.texts(rust("--seed", 11, *args, "--seeded", model=rm)),
                            self.texts(rust("--seed", 11, *args, "--top-k", 2, "--seeded", model=rm)))

    def test_the_cli_refuses_the_same_values(self):
        _, rm = self.models["count"]
        for args in (["generate", "--top-p", 0], ["generate", "--min-p", 1], ["predict", "--diversity", -1],
                     ["generate", "--top-k", -1], ["train", "--data", CORPUS, "--order", "random"],
                     ["train", "--data", CORPUS, "--curriculum", 0], ["train", "--data", CORPUS, "--replay-size", -1],
                     ["train", "--data", CORPUS, "--replay", -1]):
            with self.subTest(args=args):
                rust(*args, model=rm, expect=1)


class TestRustServerMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.path.join(tmpdir(), "methods-server.json")
        cls.server = serve(cls, cls.model)

    def test_out_of_range_is_a_400(self):
        for route, body in (
            ("/api/predict", {"prefix": "the", "top_p": 0}),
            ("/api/predict", {"prefix": "the", "min_p": 1}),
            ("/api/predict", {"prefix": "the", "top_k": -1}),
            ("/api/generate", {"mode": "sample", "top_p": 1.5}),
            ("/api/generate", {"mode": "beam", "diversity": -0.5}),
            ("/api/train", {"texts": ["hello there"], "order": "random"}),
            ("/api/train", {"texts": ["hello there"], "curriculum": 0}),
            ("/api/train", {"texts": ["hello there"], "replay": -1}),
            ("/api/train", {"texts": ["hello there"], "replay_size": -2}),
            ("/api/train", {"texts": ["hello there"], "patience": -1}),
            ("/api/train", {"texts": ["hello there"], "min_delta": -0.1}),
        ):
            with self.subTest(route=route, body=body):
                status, data, _ = self.server.post(route, body)
                self.assertEqual(status, 400, data)

    def test_a_planned_run_is_the_cli_run(self):
        with open(CORPUS, encoding="utf-8") as fh:
            texts = [line for line in fh.read().splitlines() if line.strip()]
        body = {"texts": texts, "epochs": 3, "order": "longest-first", "curriculum": 0.5, "replay_size": 6,
                "seed": 3}
        self.assertEqual(self.server.post("/api/reset", {"seed": 3, "kind": "count"})[0], 200)
        status, data, _ = self.server.post("/api/train", body)
        self.assertEqual(status, 202, data)
        job = self.server.wait_job()
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(self.server.post("/api/save", {"path": self.model})[0], 200)
        via_cli = os.path.join(tmpdir(), "methods-server-cli.json")
        py("--kind", "count", "--seed", 3, "train", "--data", CORPUS, "--epochs", 3, "--order", "longest-first",
           "--curriculum", 0.5, "--replay-size", 6, model=via_cli)
        a, b = load_json(via_cli), load_json(self.model)
        self.assertEqual(compact(a["replay"]), compact(b["replay"]))
        self.assertEqual(a["graph"]["edges"]["count"], b["graph"]["edges"]["count"])


if __name__ == "__main__":
    unittest.main()
