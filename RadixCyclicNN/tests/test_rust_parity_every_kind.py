"""Every learning loop of the Rust port held to Python's on the radix (sine) and resonant (phase) kinds.

Skipped when no Rust toolchain is available (``tests/rust_harness.py`` builds the binary once).

Python writes each loop once, against the model interface - ``two_nrl``, ``reward``, ``punish``,
``invert_paths``, ``train``, ``correct`` - and every kind answers in its own way: the sine model learns at the
loop's rates and batch size and inverts, the phase model rotates its locks, the count model rewards and
penalises by a strength.  The Rust port wrote its loops against the count model first; ``rust/src/kinds.rs`` is
where each of them now hands the call to the kind.  So this suite runs the same loop on both sides, on a radix
and on a resonant model, and asks for the same result:

* **evolve**, in every blatant mode - the same generation records (``flipped`` counts nodes on the sine model,
  edges on the phase model), the same generator, the same discriminator (a fresh model of the generator's own
  kind), and with ``--blame`` the same negative network;
* **the English tutor and the chat**, against the fake teachers of ``test_tutor`` and ``test_chat`` - the same
  conversation, the same marks and the same model afterwards: the marks reach the model as rated 2NRL through its
  kind, and a corrected failure is 2NRL on these kinds because only the count model has ``correct``;
* **the routes** - ``/api/feedback`` (a job, as Python's is, for every kind), ``/api/2nrl`` and
  ``/api/evolve/start``: the same job records and the same model, from both servers;
* **the side paths** that teach a model a few texts - ``speech teach --train``, the recall tutor's ``--train``,
  ``ollama corpus --train``, ``ollama review --2nrl`` - and ``correct``, which both refuse on these kinds.

"The same" is the whole document, but for what names the run rather than the model: the ``version`` stamps,
timestamps, ``seconds``, the ``backend`` (``python`` / ``rust``) and ``device`` names, and the files' paths.
"""

import json
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.rust_harness import CORPUS, GARBAGE, Server, build, load_json, py, rust, tmpdir  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server  # noqa: E402
    from test_chat import start_partner  # noqa: E402
    from test_ollama import start_fake as start_ollama  # noqa: E402
    from test_rust_parity_media import RATE, tone, wav_file  # noqa: E402
    from test_tutor import start_fake as start_teacher  # noqa: E402
except ImportError:
    from tests.test_api import start_server  # noqa: E402
    from tests.test_chat import start_partner  # noqa: E402
    from tests.test_ollama import start_fake as start_ollama  # noqa: E402
    from tests.test_rust_parity_media import RATE, tone, wav_file  # noqa: E402
    from tests.test_tutor import start_fake as start_teacher  # noqa: E402

KINDS = ("radix", "resonant")
BLATANT_MODES = ("none", "fail_invert", "activation", "state")
RUN_NAMES = frozenset({"version", "version_resets", "saved_at", "at", "created", "seconds", "backend", "device"})
FILE_NAMES = frozenset({"path", "bytes", "out"})
"""Where a command wrote its files (and how many bytes the timings took there): a command's document only."""


def setUpModule():
    build()


def scrub(value, names=RUN_NAMES, digits=None):
    """A document without what names the run instead of the model (see the module docstring).

    ``digits`` rounds every float to that many significant digits: the count model's losses are means Python
    and Rust sum in a different order, and may differ in their last bit (``test_rust_parity_learn``'s ``LOSSY``).
    """
    if isinstance(value, dict):
        return {k: scrub(v, names, digits) for k, v in value.items() if k not in names}
    if isinstance(value, list):
        return [scrub(v, names, digits) for v in value]
    if digits is not None and isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def compact(doc, names=RUN_NAMES, digits=None) -> str:
    return json.dumps(scrub(doc, names, digits), separators=(",", ":"), ensure_ascii=False)


def lines_of(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


class Trained:
    """One model of each kind, trained by Python, that every test copies for both sides."""

    _paths: dict[str, str] = {}

    @classmethod
    def model(cls, kind: str) -> str:
        if kind not in cls._paths:
            path = os.path.join(tmpdir(), "every-kind", f"base.{kind}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            py("--kind", kind, "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=path)
            cls._paths[kind] = path
        return cls._paths[kind]


def pair(kind: str, tag: str) -> tuple[str, str]:
    """``(python copy, rust copy)`` of the trained model of ``kind``, with nothing beside them yet."""
    root = os.path.join(tmpdir(), "every-kind", tag)
    os.makedirs(root, exist_ok=True)
    paths = []
    for side in ("py", "rs"):
        path = os.path.join(root, f"{side}.{kind}.json")
        shutil.copy(Trained.model(kind), path)
        for stale in (path.replace(".json", ".negative.json"), path.replace(".json", ".disc.json")):
            if os.path.exists(stale):
                os.remove(stale)
        paths.append(path)
    return paths[0], paths[1]


class Same:
    """The assertions every test class shares."""

    def assert_same_doc(self, theirs, mine, what: str, names=RUN_NAMES, digits=None):
        self.assertEqual(compact(theirs, names, digits), compact(mine, names, digits), what)

    def assert_same_files(self, theirs: str, mine: str, what: str, digits=None):
        self.assert_same_doc(load_json(theirs), load_json(mine), what, digits=digits)


class TestEvolveEveryKind(Same, unittest.TestCase):
    """``evolve`` on the sine and phase models: the generator's kind judges it and it learns its own way."""

    def run_both(self, kind: str, tag: str, *args) -> tuple[dict, dict, tuple[str, str]]:
        docs = []
        models = pair(kind, f"evolve-{tag}")
        for runner, model in zip((py, rust), models):
            docs.append(runner("--seed", 4, "evolve", "--data", CORPUS, "--generations", 3, "--samples", 4,
                               "--real-per-generation", 4, "--max-length", 30,
                               "--discriminator", model.replace(".json", ".disc.json"), *args, model=model))
        return docs[0], docs[1], models

    def test_every_blatant_mode_is_pythons_on_every_kind(self):
        for kind in KINDS:
            for mode in BLATANT_MODES:
                with self.subTest(kind=kind, mode=mode):
                    theirs, mine, (py_model, rs_model) = self.run_both(
                        kind, f"{kind}-{mode}", "--blatant-mode", mode, "--blatant-margin", "0.2")
                    self.assertEqual(list(mine), list(theirs), "the command's document")
                    self.assertEqual(len(mine["generations"]), 3)
                    self.assert_same_doc(theirs["generations"], mine["generations"], "the generation records")
                    self.assertEqual({k: v for k, v in mine["config"].items() if k != "strength"}, theirs["config"])
                    self.assert_same_doc(theirs["stats"], mine["stats"], "the generator's stats")
                    self.assert_same_files(py_model, rs_model, "the generator")
                    # the critic is a fresh model of the generator's own kind, and learns as Python's does
                    self.assertEqual(mine["discriminator"]["stats"]["kind"], kind)
                    self.assert_same_files(py_model.replace(".json", ".disc.json"),
                                           rs_model.replace(".json", ".disc.json"), "the discriminator")

    def test_the_local_modes_count_in_the_kinds_own_unit(self):
        """The sine model flips every other *node* of a failed path; the phase model turns its *edges*."""
        for kind in KINDS:
            with self.subTest(kind=kind):
                theirs, mine, _ = self.run_both(kind, f"{kind}-unit", "--blatant-mode", "activation",
                                                "--blatant-margin", "0.05")
                flipped = [r["flipped"] for r in mine["generations"]]
                self.assertEqual(flipped, [r["flipped"] for r in theirs["generations"]])
                self.assertTrue(any(flipped), "a margin this small leaves failures to flip")

    def test_the_critic_teaches_the_negative_network_as_pythons_does(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                theirs, mine, (py_model, rs_model) = self.run_both(
                    kind, f"{kind}-blame", "--blatant-mode", "fail_invert", "--blatant-margin", "0.2", "--blame")
                self.assert_same_doc(theirs["generations"], mine["generations"], "the generation records")
                self.assertTrue(any(r["negative_blamed"] for r in mine["generations"]))
                self.assert_same_doc(theirs["negative"], mine["negative"], "the negative network's report",
                                     RUN_NAMES | FILE_NAMES)
                self.assert_same_files(py_model, rs_model, "the generator")
                # the whole negative network: blame, clearing, weights, journal
                self.assert_same_files(py_model.replace(".json", ".negative.json"),
                                       rs_model.replace(".json", ".negative.json"), "the negative network")


class TestTeachersEveryKind(Same, unittest.TestCase):
    """The tutor and the chat on the sine and phase models, against the fake teachers."""

    def test_the_tutor_rates_every_kind_through_its_own_2nrl(self):
        fake = start_teacher(self.addCleanup)
        env = {**os.environ, "OLLAMA_HOST": fake.url, "RADIXNET_TUTOR_MODEL": "fake:latest",
               "PYTHONWARNINGS": "ignore"}
        options = ("tutor", "--topic", "animals", "--rounds", 2, "--exercises", 2, "--attempts", 1, "--mode", "beam",
                   "--threshold", 9.5, "--drills", 2, "--neg-epochs", 1, "--pos-epochs", 1)
        for kind in KINDS:
            with self.subTest(kind=kind):
                py_model, rs_model = pair(kind, f"tutor-{kind}")
                fake.requests.clear()
                a = py(*options, model=py_model, env=env)
                py_seen = [body for _method, _path, body in fake.requests if body]
                fake.requests.clear()
                b = rust(*options, model=rs_model, env=env)
                rs_seen = [body for _method, _path, body in fake.requests if body]
                # the second round's completions are written by the model the first round taught
                self.assertEqual(py_seen, rs_seen, "the same conversation with the teacher")
                self.assertEqual(list(a), list(b))
                for key in ("records", "lessons", "report", "config", "stats"):
                    self.assert_same_doc(a[key], b[key], key)
                # these kinds have no `correct`: a corrected failure is rated 2NRL garbage, as in Python
                rounds = [r for r in b["records"] if r["kind"] == "round"]
                self.assertTrue(all(r["corrections"] == 0 for r in rounds))
                self.assertIn("2nrl", [r["action"] for r in rounds])
                self.assert_same_files(py_model, rs_model, "the model the marks taught")

    def test_the_chat_learns_through_every_kinds_own_2nrl(self):
        fake = start_partner(self.addCleanup)
        env = {**os.environ, "OLLAMA_HOST": fake.url, "RADIXNET_OLLAMA_MODEL": "fake:latest",
               "PYTHONWARNINGS": "ignore"}
        options = ("chat", "--url", fake.url, "--partner-model", "fake:latest", "--conversations", 2, "--turns", 2,
                   "--topic", "animals", "--neg-epochs", 1, "--pos-epochs", 1)
        for kind in KINDS:
            with self.subTest(kind=kind):
                py_model, rs_model = pair(kind, f"chat-{kind}")
                seen = []
                docs = []
                for runner, model in ((py, py_model), (rust, rs_model)):
                    fake.requests.clear()
                    fake.lines = 0
                    fake.marked = 0
                    docs.append(runner(*options, model=model, env=env))
                    seen.append(list(fake.requests))
                a, b = docs
                self.assertEqual(seen[0], seen[1], "the same conversation with the partner and the judge")
                for key in ("records", "report", "config"):
                    self.assert_same_doc(a[key], b[key], key)
                self.assertTrue(any(r.get("action") for r in b["records"] if r["kind"] == "conversation"))
                self.assert_same_files(py_model, rs_model, "the model the marks taught")
                self.assert_same_files(py_model.replace(".json", ".negative.json"),
                                       rs_model.replace(".json", ".negative.json"), "the negative network")


class TestRoutesEveryKind(Same, unittest.TestCase):
    """The learning routes of both servers, the same requests: the same jobs and the same model."""

    def both(self, kind: str, tag: str, route: str, body: dict) -> tuple[tuple, tuple, tuple[str, str]]:
        py_model, rs_model = pair(kind, f"route-{tag}")
        client, _server, _service = start_server(self.addCleanup, model_path=py_model, seed=1)
        mine = Server(rs_model, "--seed", "1")
        self.addCleanup(mine.close)
        answers = []
        for server, model in ((client, py_model), (mine, rs_model)):
            status, doc, _ = server.post(route, body)
            self.assertEqual(status, 202, doc)
            job = self.wait(server)
            self.assertEqual(job["state"], "done", job)
            status, saved, _ = server.post("/api/save", {"path": model + ".saved.json"})
            self.assertEqual(status, 200, saved)
            answers.append((doc, job, server.get("/api/evolve/history")[1]))
        return answers[0], answers[1], (py_model + ".saved.json", rs_model + ".saved.json")

    @staticmethod
    def wait(server) -> dict:
        import time

        for _ in range(6000):
            _, job, _ = server.get("/api/job")
            if job and job.get("state") != "running":
                return job
            time.sleep(0.02)
        raise AssertionError("the job did not finish")

    def test_feedback_is_a_job_that_rates_every_kind_as_pythons_does(self):
        good, bad = lines_of(CORPUS)[:3], lines_of(GARBAGE)[:3]
        bodies = {
            "both": {"good": good, "bad": bad},
            "rated": {"good": good, "good_ratings": [8, 10, 6], "bad": bad, "bad_ratings": [3, 0, 9]},
            "down": {"bad": bad[:2], "neg_epochs": 1, "batch_size": 2},
            "up": {"good_text": "\n".join(good), "good_weights": [0.5, 1.0, 0.25]},
        }
        for kind in (*KINDS, "count"):
            for tag, body in bodies.items():
                with self.subTest(kind=kind, body=tag):
                    (a, a_job, _), (b, b_job, _), (pm, rm) = self.both(kind, f"feedback-{kind}-{tag}",
                                                                       "/api/feedback", body)
                    self.assertEqual({k: v for k, v in a.items() if k != "job"},
                                     {k: v for k, v in b.items() if k != "job"})
                    self.assertEqual((b_job["type"], a_job["type"]), ("feedback", "feedback"))
                    digits = 12 if kind == "count" else None  # the sine and phase models: to the bit
                    self.assert_same_doc(a_job["history"], b_job["history"], "the job's records", digits=digits)
                    self.assert_same_files(pm, rm, "the model", digits=digits)

    def test_2nrl_takes_pythons_defaults_on_every_kind(self):
        body = {"bad": lines_of(GARBAGE)[:4], "good": lines_of(CORPUS)[:4], "pos_epochs": 2}
        for kind in (*KINDS, "count"):
            with self.subTest(kind=kind):
                (_, a_job, _), (_, b_job, _), (pm, rm) = self.both(kind, f"2nrl-{kind}", "/api/2nrl", body)
                digits = 12 if kind == "count" else None
                self.assert_same_doc(a_job["history"], b_job["history"], "the job's records", digits=digits)
                # Python's three negative epochs on every kind (the count model's used to be two here)
                self.assertEqual(sum(1 for r in b_job["history"] if r.get("phase") == "negative"), 3)
                self.assert_same_files(pm, rm, "the model", digits=digits)

    def test_an_evolve_job_on_every_kind_is_the_python_services(self):
        body = {"corpus": lines_of(CORPUS), "generations": 2, "samples": 4, "real_per_generation": 3,
                "max_length": 30, "blatant_mode": "activation", "blatant_margin": 0.2}
        for kind in KINDS:
            with self.subTest(kind=kind):
                (_, a_job, a_history), (_, b_job, b_history), (pm, rm) = self.both(
                    kind, f"evolve-{kind}", "/api/evolve/start", body)
                self.assertEqual(len(b_history["history"]), 2)
                self.assert_same_doc(a_history, b_history, "the generation records")
                self.assert_same_files(pm, rm, "the generator")


class TestSidePathsEveryKind(Same, unittest.TestCase):
    """The commands that teach a model a few texts on the side, and ``correct``, on the sine and phase models."""

    @classmethod
    def setUpClass(cls):
        cls.fake = start_ollama(cls.addClassCleanup)
        cls.env = {**os.environ, "OLLAMA_HOST": cls.fake.url, "RADIXNET_OLLAMA_MODEL": "fake:latest",
                   "PYTHONWARNINGS": "ignore"}
        root = os.path.join(tmpdir(), "every-kind")
        os.makedirs(root, exist_ok=True)
        cls.wav = wav_file(os.path.join(root, "hello.wav"), tone())
        cls.good = os.path.join(root, "good.txt")
        with open(cls.good, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\nthe dog ran in the park\n")

    def same_on_both(self, kind: str, tag: str, *args):
        py_model, rs_model = pair(kind, f"side-{tag}-{kind}")
        a = py(*args, model=py_model, env=self.env)
        b = rust(*args, model=rs_model, env=self.env)
        self.assertEqual(list(a), list(b))
        self.assert_same_doc(a, b, "the command's document", RUN_NAMES | FILE_NAMES)
        self.assert_same_files(py_model, rs_model, "the model")
        return b

    def test_the_side_paths_learn_at_the_kinds_own_rates(self):
        cases = {
            "speech-teach": ("speech", "teach", self.wav, "--text", "hello there", "--rate", RATE, "--train",
                             "--epochs", 2),
            "recall": ("speech", "tutor", self.wav, "--text", "hello there", "--rate", RATE, "--train",
                       "--epochs", 2),
            "corpus": ("ollama", "corpus", "--prompt", "rivers", "--lines", 4, "--train", "--epochs", 2),
            "review": ("ollama", "review", "--text", "zzz nonsense", "--text", "good text", "--2nrl",
                       "--good", self.good, "--neg-epochs", 1, "--pos-epochs", 1),
        }
        for kind in KINDS:
            for tag, args in cases.items():
                with self.subTest(kind=kind, path=tag):
                    self.same_on_both(kind, tag, *args)

    def test_correct_is_refused_as_pythons_is(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                py_model, rs_model = pair(kind, f"correct-{kind}")
                args = ("correct", "--wrong", "the cat sit on the mat", "--right", "the cat sat on the mat")
                theirs = py(*args, model=py_model, expect=1)["error"]
                mine = rust(*args, model=rs_model, expect=1)["error"]
                said = "model cannot learn from a diff; use feedback instead"
                self.assertIn(said, theirs)
                self.assertIn(said, mine)
                self.assertEqual(mine.split(": ", 1)[1], theirs.split("error: ", 1)[1])
                self.assert_same_files(py_model, rs_model, "the untouched model")


if __name__ == "__main__":
    unittest.main()
