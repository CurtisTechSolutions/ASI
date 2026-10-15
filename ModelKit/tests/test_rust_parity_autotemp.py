"""The Rust port's auto-temperature against Python's: the same climbs, the same answers, the same records.

``../RadixCyclicNN/SPEC-SearchAndTraining.md`` §10 is the contract: a search that finds no answer is asked again as
sampled walks up a ladder of temperatures - the asked one, then doubling up to the ceiling, three walks a rung - and
the first answer wins; the guard's veto counts as no answer, and a reply with nothing to say climbs too.  The walks
draw from the generator the search was already drawing from - the model's own, saved in its file, or a seeded one -
so both ports must walk the same walks, find the same answer at the same rung on the same try, and write the same
record (``../RadixCyclicNN/rust/src/autotemp.rs`` in the model, ``rust/src/duo.rs``, ``dialogue.rs``,
``assistant.rs`` and ``service.rs`` in the kit, against ``../RadixCyclicNN/radixnet/autotemp.py`` and ``modelkit/``):

* the CLI - ``predict`` climbing out of a greedy loop, on every kind, ``predict`` with the ceiling off exactly as it
  was, a seeded ``generate`` drawing again for the samples that said nothing, a guarded ``predict`` whose only
  continuation is vetoed, a guarded ``generate`` and a guarded ``converse`` that would have stopped;
* the server - ``/api/predict``, ``/api/generate`` and ``/api/converse`` carrying the record, the same 400 for a
  ceiling out of range, and ``/v1/messages`` thinking the same lines when a reply has nothing to say as asked.
"""

import json
import math
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit

from tests.rust_harness import build, py, rust, serve, tmpdir  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server  # noqa: E402
except ImportError:
    from tests.test_api import start_server  # noqa: E402

# a strong cycle - "ab " comes round again far more often than it ends - so a greedy walk goes round for ever
LOOP = ["ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab end"] * 3 + ["ab end"]
# a text that ends as often as it goes on, so a walk from its end says nothing half the time
ENDS = ["the cat sat on the mat", "the cat sat on the mat and slept"]
CORPUS = ["the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill"]
FAILURE = "the cat sat on the mat"
MAT = "the cat sat on the mat"
GREEDY = ("predict", "--prefix", "ab", "--mode", "sample", "--temperature", 0, "--to-end", "--max-length", 30)
VETOED = ("predict", "--prefix", "the cat", "--mode", "beam", "--k", 1, "--length", 30, "--to-end")
KEYS = ("text", "speaker", "context", "reply", "fresh", "given", "repeat", "stutter", "candidates", "skipped",
        "vetoed", "labels", "node_ids", "reached_end")
"""What two turns must share word for word; their cost is held to 1e-9."""
RUN_NAMES = ("backend", "device")
"""What the guard's report says about the run, not the answer: the negative network's ``stats`` name the port."""


def setUpModule():
    build()


def write_lines(path: str, lines: list[str]) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def trained(name: str, lines: list[str], epochs: int, *blamed: str, kind: str = "count") -> str:
    """A model of ``lines`` trained by Python (a count model unless ``kind`` says otherwise), and - with ``blamed`` -
    a negative network beside it that has been taught those texts are gibberish (the guard every answer of the
    model then goes through)."""
    root = os.path.join(tmpdir(), "autotemp")
    os.makedirs(root, exist_ok=True)
    model = os.path.join(root, f"{name}.count.json")
    data = write_lines(os.path.join(root, f"{name}.txt"), lines)
    py("--kind", kind, "--seed", 1, "train", "--data", data, "--epochs", epochs, model=model)
    if blamed:
        args = ["negative", "blame", "--reason", "gibberish"]
        for text in blamed:
            args += ["--text", text]
        py(*args, model=model)
    return model


def copy(model: str, name: str) -> str:
    """A private copy of a model - and of its negative network, when it has one - so neither side's run can change
    what the other reads.  The copy is named after its model, so two models never share a copy (or the guard
    beside it)."""
    stem = os.path.basename(model)[: -len(".count.json")]
    target = os.path.join(os.path.dirname(model), f"{stem}-{name}.count.json")
    shutil.copy(model, target)
    negative = model.replace(".count.json", ".count.negative.json")
    guard = target.replace(".count.json", ".count.negative.json")
    if os.path.exists(negative):
        shutil.copy(negative, guard)
    elif os.path.exists(guard):
        os.remove(guard)
    return target


def beams_texts(model: str, k: int, max_length: int) -> list[str]:
    """Every text the beam offers from the start: a reply as asked, all of them vetoed, has nothing it may say."""
    from radixnet.model import load_model

    found = load_model(model).predict("", length=0, mode="beam", k=k, to_end=True, max_length=max_length)
    return [r.full_text for r in found.top]


def same_path(test: unittest.TestCase, a: dict, b: dict, *, where: str = "") -> None:
    """Two answers walked the same path: the same text, nodes and END, the costs to 1e-9."""
    for key in ("continuation", "full_text", "path", "node_ids", "reached_end"):
        test.assertEqual(a[key], b[key], f"{where}{key}")
    test.assertLessEqual(abs(a["cost"] - b["cost"]), 1e-9, where)
    test.assertEqual(len(a["step_costs"]), len(b["step_costs"]), where)
    for x, y in zip(a["step_costs"], b["step_costs"]):
        test.assertLessEqual(abs(x - y), 1e-9, where)


def same_samples(test: unittest.TestCase, a: list[dict], b: list[dict]) -> None:
    test.assertEqual([s["text"] for s in a], [s["text"] for s in b])
    test.assertEqual([s["reached_end"] for s in a], [s["reached_end"] for s in b])
    for x, y in zip(a, b):
        test.assertLessEqual(abs(x["cost"] - y["cost"]), 1e-9, x["text"])


def same_turns(test: unittest.TestCase, a: list[dict], b: list[dict]) -> None:
    """Two conversations said the same turns, each found by the same climb."""
    test.assertEqual([t["text"] for t in a], [t["text"] for t in b])
    for x, y in zip(a, b):
        for key in KEYS:
            test.assertEqual(x[key], y[key], (key, x["text"]))
        test.assertLessEqual(abs(x["cost"] - y["cost"]), 1e-9, x["text"])
        test.assertEqual("auto_temperature" in x, "auto_temperature" in y, x["text"])
        test.assertEqual(x.get("auto_temperature"), y.get("auto_temperature"), x["text"])


def same_guard(test: unittest.TestCase, a: dict | None, b: dict | None, *keys: str) -> None:
    """The guard judged the same texts the same way, and counted them alike."""
    test.assertEqual(a is None, b is None)
    if a is None:
        return
    for key in ("on", "vetoed", *keys):
        test.assertEqual(a[key], b[key], key)
    test.assertEqual([(v["text"], v["decision"], v["rule"]) for v in a["verdicts"]],
                     [(v["text"], v["decision"], v["rule"]) for v in b["verdicts"]])


class TestRustAutoTemperatureCli(unittest.TestCase):
    """``predict``, ``generate`` and ``converse`` with ``--auto-temperature``, on both sides."""

    @classmethod
    def setUpClass(cls):
        cls.loop = trained("loop", LOOP, 2)
        cls.ends = trained("ends", ENDS, 2)
        cls.vetoing = trained("vetoing", CORPUS, 3, FAILURE)

    def test_predict_climbs_out_of_a_greedy_loop(self):
        a = py(*GREEDY, "--auto-temperature", 2, model=copy(self.loop, "gp"))
        b = rust(*GREEDY, "--auto-temperature", 2, model=copy(self.loop, "gr"))
        same_path(self, a, b)
        for key in ("expanded", "mode", "k", "beam", "traversal", "guard"):
            self.assertEqual(a[key], b[key], key)
        self.assertTrue(b["reached_end"], "a warmer walk ends")
        self.assertEqual(a["auto_temperature"], b["auto_temperature"])
        record = b["auto_temperature"]
        self.assertTrue(record["climbed"] and record["found"])
        self.assertGreater(record["temperature"], 0)
        self.assertEqual(record["ceiling"], 2.0)
        self.assertEqual(list(b)[-1], "auto_temperature", "the record is the document's last word, as in Python")

    def test_every_kind_climbs_alike(self):
        """The sine, phase and negative models walk the ladder as the count model does: the same walks from the
        model's own generator, the same answer, the same record - and a deterministic search that answers is not
        walked past."""
        for kind in ("radix", "resonant", "negative"):
            model = trained(f"loop-{kind}", LOOP, 2, kind=kind)
            for mode in (("--mode", "sample", "--temperature", 0), ("--mode", "beam"), ("--mode", "dijkstra")):
                with self.subTest(kind=kind, mode=mode[1]):
                    args = ("predict", "--prefix", "ab", *mode, "--to-end", "--max-length", 30,
                            "--auto-temperature", 2)
                    a = py(*args, model=copy(model, "kp"))
                    b = rust(*args, model=copy(model, "kr"))
                    same_path(self, a, b)
                    self.assertEqual(a["auto_temperature"], b["auto_temperature"])
                    self.assertTrue(b["auto_temperature"]["found"] and b["reached_end"])
                    self.assertEqual(b["auto_temperature"]["climbed"], mode[1] == "sample")

    def test_predict_with_the_ceiling_off_is_unchanged(self):
        a = py(*GREEDY, model=copy(self.loop, "op"))
        b = rust(*GREEDY, model=copy(self.loop, "or"))
        same_path(self, a, b)
        self.assertFalse(b["reached_end"], "greedy goes round and never ends")
        self.assertNotIn("auto_temperature", a)
        self.assertNotIn("auto_temperature", b)
        zero = rust(*GREEDY, "--auto-temperature", 0, model=copy(self.loop, "oz"))
        self.assertEqual(zero, b, "a ceiling of 0 is no ceiling: the same walk, the same document")

    def test_generate_draws_again_for_the_samples_that_said_nothing(self):
        args = ("generate", "--count", 6, "--mode", "sample", "--prefix", MAT)
        plain_py = py("--seed", 4, *args, model=copy(self.ends, "sp0"))
        plain_rs = rust("--seed", 4, *args, "--seeded", model=copy(self.ends, "sr0"))
        same_samples(self, plain_py["samples"], plain_rs["samples"])
        self.assertNotIn("auto_temperature", plain_rs)
        self.assertTrue(any(s["text"] == MAT for s in plain_rs["samples"]), "some walks end where the prefix does")
        a = py("--seed", 4, *args, "--auto-temperature", 2, model=copy(self.ends, "sp"))
        b = rust("--seed", 4, *args, "--auto-temperature", 2, "--seeded", model=copy(self.ends, "sr"))
        same_samples(self, a["samples"], b["samples"])
        self.assertEqual(a["count"], b["count"])
        self.assertEqual(a["auto_temperature"], b["auto_temperature"])
        self.assertTrue(b["auto_temperature"]["climbed"])
        self.assertTrue(all(len(s["text"]) > len(MAT) for s in b["samples"]))
        # the beam reads no temperature: the ladder is where its sampling starts, from greedy when asked so
        for extra in ((), ("--temperature", 0)):
            with self.subTest(extra=extra):
                args = ("generate", "--count", 3, "--mode", "beam", "--prefix", MAT, "--max-length", 40, *extra,
                        "--auto-temperature", 2)
                a = py(*args, model=copy(self.ends, "bp"))
                b = rust(*args, model=copy(self.ends, "br"))
                same_samples(self, a["samples"], b["samples"])
                self.assertEqual(a["auto_temperature"], b["auto_temperature"])

    def test_a_generation_climbs_only_for_what_it_would_have_given(self):
        """Dijkstra's generation is one text however many are asked for, and the beam's texts are distinct: the
        ladder fills only what is missing of that - Dijkstra's one text when it says nothing, a beam's text that
        says nothing with one the beam did not already give - and both ports fill it alike."""
        cases = (("--mode", "dijkstra"), ("--mode", "dijkstra", "--prefix", MAT), ("--mode", "beam", "--prefix", MAT))
        for seed in (0, 9):
            for case in cases:
                with self.subTest(seed=seed, case=case):
                    args = ("generate", "--count", 3, *case, "--max-length", 40, "--auto-temperature", 2)
                    a = py("--seed", seed, *args, model=copy(self.ends, "wp"))
                    b = rust("--seed", seed, *args, "--seeded", model=copy(self.ends, "wr"))
                    same_samples(self, a["samples"], b["samples"])
                    self.assertEqual(a["auto_temperature"], b["auto_temperature"])
                    texts = [s["text"] for s in b["samples"]]
                    self.assertEqual(len(set(texts)), len(texts), "no text twice")
                    if case[1] == "dijkstra":
                        self.assertEqual(len(texts), 1)
                        self.assertEqual(b["auto_temperature"]["climbed"], MAT in case, "only when it said nothing")

    def test_nothing_asked_for_is_found_as_asked(self):
        """A generation of no texts has nothing missing: with the ceiling on, the record says so, guarded or not."""
        found = {"ceiling": 2.0, "climbed": False, "found": True, "temperature": 1.0, "tries": 1}
        for model in (self.ends, self.vetoing):
            with self.subTest(model=os.path.basename(model)):
                args = ("generate", "--count", 0, "--auto-temperature", 2)
                a = py(*args, model=copy(model, "none-py"))
                b = rust(*args, model=copy(model, "none-rs"))
                self.assertEqual((a["samples"], b["samples"]), ([], []))
                self.assertEqual((a["auto_temperature"], b["auto_temperature"]), (found, found))
                same_guard(self, a["guard"], b["guard"], "candidates", "asked", "kept", "rate")

    def test_a_guarded_predict_climbs_past_the_veto(self):
        """The guard vetoes the beam's only continuation: off, nothing is said; on, a warmer walk the guard lets
        through is the answer - and every search the climb ran was judged."""
        plain_py = py(*VETOED, model=copy(self.vetoing, "vp0"))
        plain_rs = rust(*VETOED, model=copy(self.vetoing, "vr0"))
        same_path(self, plain_py, plain_rs)
        self.assertEqual(plain_rs["continuation"], "")
        self.assertNotIn("auto_temperature", plain_rs)
        same_guard(self, plain_py["guard"], plain_rs["guard"], "candidates", "kept")
        a = py(*VETOED, "--auto-temperature", 2, model=copy(self.vetoing, "vp"))
        b = rust(*VETOED, "--auto-temperature", 2, model=copy(self.vetoing, "vr"))
        same_path(self, a, b)
        for key in ("expanded", "mode", "k", "beam"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a["auto_temperature"], b["auto_temperature"])
        record = b["auto_temperature"]
        self.assertTrue(record["climbed"] and record["found"])
        self.assertTrue(b["continuation"].strip() and b["reached_end"])
        self.assertNotEqual(b["full_text"], FAILURE)
        same_guard(self, a["guard"], b["guard"], "candidates", "kept")
        self.assertEqual(b["guard"]["candidates"], record["tries"], "every search was judged")
        self.assertEqual(b["guard"]["verdicts"][0]["decision"], "reject")

    def test_a_guarded_generation_draws_again_for_what_it_vetoed(self):
        offered = py("generate", "--count", 3, "--mode", "beam", "--max-length", 40, "--no-guard",
                     model=copy(self.vetoing, "go"))
        model = trained("own-best", CORPUS, 3, *[s["text"] for s in offered["samples"][:2]])
        args = ("generate", "--count", 3, "--mode", "beam", "--max-length", 40, "--over-sample", 1)
        plain_py = py(*args, model=copy(model, "gp0"))
        plain_rs = rust(*args, model=copy(model, "gr0"))
        same_samples(self, plain_py["samples"], plain_rs["samples"])
        self.assertLess(len(plain_rs["samples"]), 3, "the beam's own texts were vetoed")
        a = py("--seed", 3, *args, "--auto-temperature", 2, model=copy(model, "gp"))
        b = rust("--seed", 3, *args, "--auto-temperature", 2, "--seeded", model=copy(model, "gr"))
        same_samples(self, a["samples"], b["samples"])
        self.assertEqual(a["auto_temperature"], b["auto_temperature"])
        self.assertTrue(b["auto_temperature"]["climbed"])
        same_guard(self, a["guard"], b["guard"], "candidates", "asked", "kept", "rate")
        self.assertGreater(b["guard"]["candidates"], plain_rs["guard"]["candidates"])

    def test_a_conversation_goes_on_where_it_would_have_stopped(self):
        model = trained("silenced", CORPUS, 3, *beams_texts(self.vetoing, 3, 60))
        args = ("--seed", 4, "converse", "--turns", 2, "--k", 3, "--no-learn")
        stopped = rust(*args, model=copy(model, "cr0"))
        self.assertEqual(stopped["turns"], [], "every text the beam offers is vetoed")
        a = py(*args, "--auto-temperature", 2, model=copy(model, "cp"))
        b = rust(*args, "--auto-temperature", 2, model=copy(model, "cr"))
        same_turns(self, a["turns"], b["turns"])
        self.assertGreaterEqual(len(b["turns"]), 1)
        self.assertTrue(all(t["auto_temperature"]["ceiling"] == 2.0 for t in b["turns"]))
        self.assertTrue(b["turns"][0]["auto_temperature"]["climbed"])
        self.assertEqual(a["transcript"], b["transcript"])
        same_guard(self, a["guard"], b["guard"], "refusals")

    def test_a_ceiling_out_of_range_is_refused(self):
        for command in (("predict", "--prefix", "ab"), ("generate",), ("converse", "--turns", 1),
                        ("talk", "--message", "ab")):
            for bad in (-1, "nan", "hot"):
                with self.subTest(command=command[0], bad=bad):
                    doc = rust(*command, "--auto-temperature", bad, model=copy(self.loop, "bad"), expect=1)
                    self.assertIn("--auto-temperature", doc["error"])


class TestRustAutoTemperatureRoutes(unittest.TestCase):
    """Both servers over copies of the same model: the same answers, records and refusals."""

    @classmethod
    def setUpClass(cls):
        cls.vetoing = trained("routes", CORPUS, 3, FAILURE)
        cls.silenced = trained("routes-silenced", CORPUS, 3, *beams_texts(cls.vetoing, 5, 60))

    def servers(self, model: str, name: str):
        """A Python server and a Rust one, each over its own copy of ``model``."""
        client, _server, _service = start_server(self.addCleanup, model_path=copy(model, f"{name}-py"), seed=1)
        return client, serve(self, copy(model, f"{name}-rs"), "--seed", "1")

    def both(self, servers, path: str, body: dict, expect: int = 200) -> tuple[dict, dict]:
        answers = []
        for server in servers:
            status, doc, _ = server.post(path, body)
            self.assertEqual(status, expect, (path, body, doc))
            answers.append(doc)
        return answers[0], answers[1]

    def test_predict_climbs_past_the_veto(self):
        servers = self.servers(self.vetoing, "predict")
        body = {"prefix": "the cat", "mode": "beam", "k": 1, "length": 30, "to_end": True}
        a, b = self.both(servers, "/api/predict", body)
        self.assertEqual((a["continuation"], b["continuation"]), ("", ""))
        self.assertNotIn("auto_temperature", b)
        a, b = self.both(servers, "/api/predict", {**body, "auto_temperature": 2})
        same_path(self, a, b)
        for key in ("expanded", "mode", "k", "beam", "traversal"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a["auto_temperature"], b["auto_temperature"])
        self.assertTrue(b["continuation"].strip())
        self.assertTrue(b["auto_temperature"]["climbed"] and b["auto_temperature"]["found"])
        same_guard(self, a["guard"], b["guard"], "candidates", "kept")
        self.assertEqual(b["guard"]["candidates"], b["auto_temperature"]["tries"])

    def test_generate_and_converse_carry_the_record(self):
        servers = self.servers(self.vetoing, "generate")
        a, b = self.both(servers, "/api/generate", {"count": 2, "mode": "sample", "max_length": 40, "seed": 2,
                                                    "auto_temperature": 2})
        same_samples(self, a["samples"], b["samples"])
        self.assertEqual(a["auto_temperature"], b["auto_temperature"])
        same_guard(self, a["guard"], b["guard"], "candidates", "asked", "kept", "rate")
        a, b = self.both(servers, "/api/generate", {"count": 2, "mode": "sample", "max_length": 40, "seed": 2})
        self.assertNotIn("auto_temperature", b)
        # nothing asked for is nothing missing: found as asked, the record all the same
        a, b = self.both(servers, "/api/generate", {"count": 0, "mode": "beam", "auto_temperature": 2})
        self.assertEqual(a["auto_temperature"], b["auto_temperature"])
        self.assertEqual(b["auto_temperature"], {"ceiling": 2.0, "climbed": False, "found": True, "temperature": None,
                                                 "tries": 1})
        a, b = self.both(servers, "/api/converse", {"opening": "the cat", "turns": 2, "learn": False,
                                                    "auto_temperature": 2})
        same_turns(self, a["turns"], b["turns"])
        self.assertNotIn("auto_temperature", b["turns"][0], "the opening was given, not found")
        for turn in b["turns"][1:]:
            self.assertEqual(turn["auto_temperature"]["ceiling"], 2.0)

    def test_a_conversation_that_would_stop_goes_on_the_same_way(self):
        servers = self.servers(self.silenced, "converse")
        body = {"turns": 2, "seed": 4, "learn": False}
        a, b = self.both(servers, "/api/converse", body)
        self.assertEqual((a["turns"], b["turns"]), ([], []))
        a, b = self.both(servers, "/api/converse", {**body, "auto_temperature": 2})
        same_turns(self, a["turns"], b["turns"])
        self.assertTrue(b["turns"] and b["turns"][0]["auto_temperature"]["climbed"])
        # the stream says the same, turn for turn, and ends with the document
        streamed = []
        for server in servers:
            status, raw, _ = server.post("/api/converse/stream", {**body, "auto_temperature": 2})
            self.assertEqual(status, 200, raw)
            events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            streamed.append([e["turn"] for e in events if e["event"] == "turn"])
        same_turns(self, streamed[0], streamed[1])
        same_turns(self, streamed[1], b["turns"])

    def test_a_ceiling_out_of_range_is_a_400(self):
        """Negative, not a number, or not finite: a 400 on every route.  A finite refusal is worded alike; ``NaN``
        and ``Infinity`` are no JSON to Python's server, which refuses the body before it reads the field, while
        Rust's reads them and refuses the field - a 400 either way."""
        servers = self.servers(self.vetoing, "refused")
        for path, body in (("/api/predict", {"prefix": "the"}), ("/api/generate", {}), ("/api/converse", {}),
                           ("/api/converse/stream", {})):
            for bad in (-1, -0.5, "hot", True, math.inf, math.nan):
                with self.subTest(path=path, bad=bad):
                    a, b = self.both(servers, path, {**body, "auto_temperature": bad}, expect=400)
                    if math.isfinite(bad if isinstance(bad, (int, float)) else 0):
                        self.assertEqual(a, b, "refused in the same words")
                    else:
                        self.assertIn("'auto_temperature' must be a finite number", b["error"])
        for path in ("/v1/messages", "/v1/chat/completions"):
            for bad in (-1, "hot", math.inf):
                with self.subTest(path=path, bad=bad):
                    a, b = self.both(servers, path, {"messages": [{"role": "user", "content": "the cat"}],
                                                     "auto_temperature": bad}, expect=400)
                    if math.isfinite(bad if isinstance(bad, (int, float)) else 0):
                        self.assertEqual(a, b, "refused in the dialect's shape, in the same words")
                    else:
                        self.assertEqual(set(a), set(b), "refused in the dialect's shape")
                        self.assertIn("auto_temperature must be a finite number >= 0", json.dumps(b))

    def test_todays_format_narrates_the_climb_the_same_way(self):
        """Every text the beam can start a reply with is vetoed, so the reply as asked has nothing to say; a warmer
        walk does, and both ports think the same lines on the way there."""
        servers = self.servers(self.silenced, "talk")
        body = {"messages": [{"role": "user", "content": "zzq"}], "max_tokens": 60, "learn": False, "seed": 3}
        a, b = self.both(servers, "/v1/messages", body)
        self.assertEqual((a["stop_reason"], b["stop_reason"]), ("refusal", "refusal"), "as asked: nothing to say")
        for path in ("/v1/messages", "/v1/chat/completions"):
            with self.subTest(path=path):
                a, b = self.both(servers, path, {**body, "auto_temperature": 2})
                ra, rb = a.pop("radixnet"), b.pop("radixnet")
                for doc in (a, b):
                    doc.pop("id", None)
                    doc.pop("created", None)
                self.assertEqual(a, b)
                ta, tb = ra["choices"][0]["turn"], rb["choices"][0]["turn"]
                self.assertIsNotNone(tb, "a warmer walk had something to say")
                for key in (*KEYS, "auto_temperature"):
                    self.assertEqual(ta[key], tb[key], key)
                self.assertLessEqual(abs(ta["cost"] - tb["cost"]), 1e-9)
                record = tb["auto_temperature"]
                self.assertTrue(record["climbed"] and record["found"])
                ga, gb = ra["choices"][0]["guard"], rb["choices"][0]["guard"]
                self.assertEqual({k: v for k, v in ga.items() if k != "negative"},
                                 {k: v for k, v in gb.items() if k != "negative"})
                self.assertEqual({k: v for k, v in ga["negative"].items() if k not in RUN_NAMES},
                                 {k: v for k, v in gb["negative"].items() if k not in RUN_NAMES})
                thinking = (b["content"][0]["thinking"] if path == "/v1/messages"
                            else b["choices"][0]["message"]["reasoning_content"])
                self.assertIn("nothing to say yet; trying again at temperature 1.00 (at most 3 replies)", thinking)
                self.assertIn(f"found something to say at temperature {record['temperature']:.2f}, "
                              f"on try {record['tries']}", thinking)


if __name__ == "__main__":
    unittest.main()
