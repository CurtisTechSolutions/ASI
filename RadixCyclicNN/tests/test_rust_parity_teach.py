"""Cross-language parity of the Rust port's teaching loops: the English tutor, its lesson plan, and the chat.

Skipped when no Rust toolchain is available.  The fake servers of the Python
suites play the LLM for both ports - the fake teacher of ``test_tutor`` writes
exercises, marks sentences, explains mistakes, writes drills and plans the
next lessons; the fake partner of ``test_chat`` talks and judges - and both
ports are pointed at the same one, so what has to agree is:

* **the conversation** - every request body the fake receives from Rust is the
  one it received from Python: the same prompts and system prompts byte for
  byte, the same model, options and format;
* **the marks** - the same lessons, grades, corrections, explanations,
  variants, report cards, plans and conversation records;
* **the effect on the models** - the positive model the grades trained and the
  negative network the failures blamed are Python's (but for the ``version``
  cache stamp every load bumps);
* **the routes** - ``/api/tutor``, ``/api/tutor/lesson``, ``/api/tutor/plan``,
  ``/api/tutor/start`` with ``/api/tutor/history``, and ``/api/chat/start``
  with ``/api/chat/history`` answer the frontend's JSON.

It mirrors ``TestGoTutorParity`` and ``TestGoChatParity`` of
``test_go_parity.py`` and goes further where it can, because the Rust port's
documents are held to Python's rather than to a handful of their keys.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.rust_harness import CORPUS, build, load_json, py, rust, serve, tmpdir  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_chat import start_partner  # noqa: E402
    from test_ollama import closed_port_url  # noqa: E402
    from test_tutor import start_fake as start_teacher  # noqa: E402
except ImportError:
    from tests.test_chat import start_partner  # noqa: E402
    from tests.test_ollama import closed_port_url  # noqa: E402
    from tests.test_tutor import start_fake as start_teacher  # noqa: E402


def setUpModule():
    build()  # SkipTest without a Rust toolchain


def compact(doc):
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


def graph_of(path):
    """A model file's graph as both ports write it, without the cache stamp every load bumps."""
    return compact({k: v for k, v in load_json(path)["graph"].items() if k != "version"})


def negative_beside(path):
    return path.replace(".count.json", ".count.negative.json")


def fresh(path):
    for name in (path, negative_beside(path)):
        if os.path.exists(name):
            os.remove(name)


def trained_pair(name):
    """The same corpus trained the same way on both sides: ``(python model, rust model)``."""
    py_path = os.path.join(tmpdir(), f"{name}_py.count.json")
    rs_path = os.path.join(tmpdir(), f"{name}_rs.count.json")
    fresh(py_path)
    fresh(rs_path)
    py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=py_path)
    rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=rs_path)
    return py_path, rs_path


def without_timing(value):
    """A document with the wall-clock fields taken out, which no two runs share."""
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def close(test, a, b, path="doc"):
    """Two documents equal, the floats to 1e-9 (a mean summed in another order may differ in its last bit)."""
    if isinstance(a, float) or isinstance(b, float):
        test.assertIsInstance(a, (int, float), path)
        test.assertIsInstance(b, (int, float), path)
        test.assertLessEqual(abs(a - b), 1e-9, f"{path}: {a} != {b}")
    elif isinstance(a, dict):
        test.assertIsInstance(b, dict, path)
        test.assertEqual(list(a), list(b), f"{path}: the keys, in order")
        for key in a:
            close(test, a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        test.assertIsInstance(b, list, path)
        test.assertEqual(len(a), len(b), f"{path}: the length")
        for i, (x, y) in enumerate(zip(a, b)):
            close(test, x, y, f"{path}[{i}]")
    else:
        test.assertEqual(a, b, path)


class TestRustTutorParity(unittest.TestCase):
    """One fake teacher, both tutors: the same prompts, the same marks, the same models afterwards."""

    def setUp(self):
        self.fake = start_teacher(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_TUTOR_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.py_path, self.rs_path = trained_pair("tutor")

    def bodies(self):
        """Every request body the fake has seen since the last call, which is then forgotten."""
        seen = [body for _method, _path, body in self.fake.requests if body]
        self.fake.requests.clear()
        return seen

    def both(self, *options):
        """The same command on both sides; ``(python doc, rust doc, python requests, rust requests)``."""
        self.fake.requests.clear()
        a = py(*options, model=self.py_path, env=self.env)
        py_seen = self.bodies()
        b = rust(*options, model=self.rs_path, env=self.env)
        return a, b, py_seen, self.bodies()

    def same_run(self, a, b):
        """The documents both commands answer with, key for key."""
        self.assertEqual(list(a), list(b))
        close(self, a["config"], b["config"], "config")
        close(self, without_timing(a["records"]), without_timing(b["records"]), "records")
        close(self, without_timing(a["lessons"]), without_timing(b["lessons"]), "lessons")
        close(self, a["report"], b["report"], "report")
        close(self, without_timing(a["plan"]), without_timing(b["plan"]), "plan")
        self.assertEqual(a["model"], {**b["model"], "path": a["model"]["path"]})

    def test_both_tutors_ask_the_same_and_learn_the_same(self):
        options = ("tutor", "--topic", "animals", "--rounds", 2, "--exercises", 2, "--attempts", 1,
                   "--mode", "beam", "--threshold", 9.5, "--drills", 2, "--neg-epochs", 1, "--pos-epochs", 1,
                   "--strength", 1.0)
        a, b, py_seen, rs_seen = self.both(*options)

        # the same conversation with the teacher: same system prompts, same exercises, same sentences to mark
        self.assertGreaterEqual(len(py_seen), 6)  # exercises + marking + drills, twice
        self.assertEqual(len(py_seen), len(rs_seen))
        for i, (first, second) in enumerate(zip(py_seen, rs_seen)):
            self.assertEqual(first, second, f"call {i} differs between the two tutors")

        # the same marks, lessons, rounds and report card
        self.same_run(a, b)
        self.assertEqual(len(b["lessons"]), 4)  # two rounds of two exercises

        # and the same model: the grades reached the graph as the same rewards and penalties
        self.assertEqual(graph_of(self.py_path), graph_of(self.rs_path))
        for key in ("twonrl_runs", "feedback_passes", "total_traversals", "nodes", "edges"):
            self.assertEqual(a["stats"][key], b["stats"][key], key)
        for key in ("rewards_total", "penalties_total", "edge_reward_positive", "edge_reward_negative"):
            self.assertLessEqual(abs(a["stats"][key] - b["stats"][key]), 1e-9, key)
        self.assertEqual(a["saved"]["path"], self.py_path)
        self.assertEqual(b["saved"]["path"], self.rs_path)

    def test_lesson_by_lesson_and_whole_sentence_corrections_learn_the_same(self):
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 3, "--attempts", 2,
                   "--mode", "beam", "--threshold", 9.5, "--twonrl-per", "lesson", "--no-diff-corrections",
                   "--neg-epochs", 1, "--pos-epochs", 1)
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        self.same_run(a, b)
        self.assertEqual(graph_of(self.py_path), graph_of(self.rs_path))

    def test_both_tutors_widen_a_mistake_the_same_way(self):
        """--blame --variants: the same question to the teacher, and the same family in the negative network."""
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam",
                   "--threshold", 9.5, "--blame", "--variants", 2, "--variant-weight", 0.5,
                   "--neg-epochs", 1, "--pos-epochs", 1)
        a, b, py_seen, rs_seen = self.both(*options)

        # the same conversation, the extra "why" call included
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(sum(1 for body in py_seen if "explaining a beginner's mistake" in body["system"]), 1)

        # the same explanation and the same family on every lesson
        self.same_run(a, b)
        self.assertTrue(all(len(lesson["variants"]) == 2 for lesson in b["lessons"]))
        rounds = [r for r in b["records"] if r["kind"] == "round"]
        self.assertEqual(rounds[0]["similar"], 2 * rounds[0]["lessons"])

        # and the same negative network afterwards, reason for reason
        py_neg, rs_neg = negative_beside(self.py_path), negative_beside(self.rs_path)
        self.assertEqual(graph_of(py_neg), graph_of(rs_neg))
        self.assertEqual([(r["reason"], r["note"], r["source"]) for r in load_json(py_neg)["log"]],
                         [(r["reason"], r["note"], r["source"]) for r in load_json(rs_neg)["log"]])
        self.assertEqual(a["negative"]["reasons"], b["negative"]["reasons"])
        self.assertEqual(set(a["negative"]), set(b["negative"]))
        # and both judge a sentence neither network ever wrote the same way
        verdicts = [runner("negative", "why", "--text", "the dogs sits on the mat 0", "--negative", negative,
                           model=model)["verdicts"][0]["verdict"]
                    for runner, model, negative in ((py, self.py_path, py_neg), (rust, self.rs_path, rs_neg))]
        self.assertEqual(verdicts[0], verdicts[1])
        self.assertNotEqual(verdicts[1], "pass")

    def test_both_tutors_plan_the_same_next_lessons(self):
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam",
                   "--threshold", 9.5, "--dry-run", "--plan", 2)
        a, b, py_seen, rs_seen = self.both(*options)

        # the report card is handed back to the teacher with the same words on both sides
        self.assertEqual(py_seen, rs_seen)
        plans = [body["prompt"] for body in py_seen if "planning the next lessons" in body["system"]]
        self.assertEqual(len(plans), 1)
        self.assertIn("Lessons marked: 2", plans[0])

        # and the same syllabus comes out, down to the brief the next batch would be taught to
        self.same_run(a, b)
        self.assertEqual(b["plan"]["upgrade"]["step"], "hold")  # nothing passed at 9.5
        self.assertEqual(b["plan"]["lessons"][0]["targets"], "agreement")
        self.assertIsNone(b["saved"])
        self.assertIsNone(b["out"])

    def test_both_tutors_teach_to_the_same_brief(self):
        brief = "Drill plural nouns first. Stay at beginner. Keep the sentences about animals."
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam", "--dry-run",
                   "--brief", brief)
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        written = [body["prompt"] for body in rs_seen if "writing exercises" in body["system"]]
        self.assertEqual(len(written), 1)
        self.assertIn(f"The plan for this batch of lessons: {brief}", written[0])
        self.same_run(a, b)

    def test_both_tutors_auto_run_the_same_way(self):
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam",
                   "--threshold", 9.5, "--dry-run", "--batches", 3)
        a, b, py_seen, rs_seen = self.both(*options)

        # the same conversation: exercises, marking, the plan between the batches, the next batch's exercises
        self.assertEqual(py_seen, rs_seen)
        kinds = [r["kind"] for r in b["records"]]
        self.assertEqual(kinds, ["round", "report", "plan", "batch", "round", "report", "plan", "batch", "round",
                                 "report"])
        self.same_run(a, b)
        started = [r for r in b["records"] if r["kind"] == "batch"]
        self.assertEqual([r["batch"] for r in started], [2, 3])
        written = [body["prompt"] for body in rs_seen if "writing exercises" in body["system"]]
        self.assertEqual(len(written), 3)
        self.assertIn(f"The plan for this batch of lessons: {started[-1]['brief']}", written[-1])

    def test_dry_runs_agree_and_change_nothing(self):
        before = (graph_of(self.py_path), graph_of(self.rs_path))
        options = ("tutor", "--topic", "animals", "--rounds", 1, "--exercises", 2, "--mode", "beam", "--dry-run")
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        self.same_run(a, b)
        self.assertEqual((graph_of(self.py_path), graph_of(self.rs_path)), before)

    def test_a_teacher_that_writes_nothing_usable_is_an_error_on_both_sides(self):
        self.fake.exercise_response = "```\n```"
        options = ("tutor", "--topic", "animals", "--rounds", 1)
        py(*options, model=self.py_path, env=self.env, expect=1)
        err = rust(*options, model=self.rs_path, env=self.env, expect=1)
        self.assertIn("returned no usable exercises", err["error"])
        err = rust("tutor", "--rounds", 0, model=self.rs_path, env=self.env, expect=1)
        self.assertIn("rounds", err["error"])


class TestRustChatParity(unittest.TestCase):
    """One fake partner, both chat loops: the same conversation, the same marks, the same learning."""

    def setUp(self):
        self.fake = start_partner(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_OLLAMA_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.py_path, self.rs_path = trained_pair("chat")

    def restart(self):
        """A fresh fake for the second side: the same lines, in the same order, from the start."""
        seen = list(self.fake.requests)
        self.fake.requests.clear()
        self.fake.lines = 0
        self.fake.marked = 0
        return seen

    def both(self, *options):
        self.restart()
        a = py(*options, model=self.py_path, env=self.env)
        py_seen = self.restart()
        b = rust(*options, model=self.rs_path, env=self.env)
        return a, b, py_seen, self.restart()

    def test_both_loops_converse_the_same_and_learn_the_same(self):
        options = ("chat", "--url", self.fake.url, "--partner-model", "fake:latest", "--conversations", 2,
                   "--turns", 2, "--topic", "animals", "--neg-epochs", 1, "--pos-epochs", 1)
        a, b, py_seen, rs_seen = self.both(*options)

        # the same conversation with the partner and the judge, body for body
        self.assertGreaterEqual(len(py_seen), 4)  # open, carry on, open again, and the two judgements
        self.assertEqual(py_seen, rs_seen)

        # the same conversations, marked the same and learned from the same
        self.assertEqual(list(a), list(b))
        close(self, without_timing(a["records"]), without_timing(b["records"]), "records")
        close(self, a["report"], b["report"], "report")
        close(self, a["config"], b["config"], "config")
        self.assertEqual((a["url"], a["partner"], a["judge"]), (b["url"], b["partner"], b["judge"]))
        self.assertEqual(b["speakers"], ["Partner", "Model"])
        self.assertEqual(len([r for r in b["records"] if r["kind"] == "conversation"]), 2)

        # and the same two networks on disk: 2NRL trained the positive one, the marks taught the negative one
        self.assertEqual(graph_of(self.py_path), graph_of(self.rs_path))
        py_neg, rs_neg = negative_beside(self.py_path), negative_beside(self.rs_path)
        self.assertEqual(graph_of(py_neg), graph_of(rs_neg))
        self.assertEqual({r["source"] for r in load_json(rs_neg)["log"]}, {"chat"})
        self.assertEqual(a["negative"]["reasons"], b["negative"]["reasons"])
        self.assertEqual(a["saved"], self.py_path)
        self.assertEqual(b["saved"], self.rs_path)

    def test_a_given_opening_is_spoken_as_it_is_on_both_sides(self):
        options = ("chat", "--url", self.fake.url, "--partner-model", "fake:latest", "--conversations", 1,
                   "--turns", 1, "--opening", "tell me about the cat", "--no-learn", "--no-blame")
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        py_held = next(r for r in a["records"] if r["kind"] == "conversation")
        rs_held = next(r for r in b["records"] if r["kind"] == "conversation")
        self.assertEqual(rs_held["transcript"][0], {"speaker": "Partner", "text": "tell me about the cat"})
        close(self, without_timing(py_held), without_timing(rs_held), "conversation")
        self.assertIsNone(a["saved"])  # nothing was learned, so neither model was written back
        self.assertIsNone(b["saved"])

    def test_neither_side_learns_with_no_learn(self):
        before = (graph_of(self.py_path), graph_of(self.rs_path))
        options = ("chat", "--url", self.fake.url, "--partner-model", "fake:latest", "--conversations", 1,
                   "--turns", 2, "--no-learn", "--no-blame")
        self.both(*options)
        self.assertEqual((graph_of(self.py_path), graph_of(self.rs_path)), before)

    def test_a_guarded_conversation_vetoes_what_python_vetoes(self):
        # teach both negative networks the same failures first, so the guard has something to say
        for runner, model in ((py, self.py_path), (rust, self.rs_path)):
            runner("negative", "blame", "--text", "the dog sat on the log", "--reason", "repetition",
                   "--negative", negative_beside(model), model=model, env=self.env)
        options = ("chat", "--url", self.fake.url, "--partner-model", "fake:latest", "--conversations", 1,
                   "--turns", 3, "--no-learn", "--no-blame")
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        close(self, without_timing(a["records"]), without_timing(b["records"]), "records")


class TestRustTeachServer(unittest.TestCase):
    """``radixnet serve`` answers the tutor's and the chat's routes as the Python server does."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.teacher = start_teacher(cls.addClassCleanup)
        cls.dir = tempfile.mkdtemp(prefix="radixnet-rust-teach-")
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)
        trained = os.path.join(cls.dir, "trained.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=trained)
        cls.py_model = os.path.join(cls.dir, "py", "m.count.json")
        cls.rs_model = os.path.join(cls.dir, "rs", "m.count.json")
        for path in (cls.py_model, cls.rs_model):
            os.makedirs(os.path.dirname(path))
            shutil.copyfile(trained, path)
        cls.python, _server, cls.service = start_server(
            cls.addClassCleanup, model_path=cls.py_model, kind="count", ollama_url=cls.teacher.url,
            ollama_model="fake:latest", seed=1,
        )
        env = {**os.environ, "OLLAMA_HOST": cls.teacher.url}
        env.pop("RADIXNET_TUTOR_MODEL", None)
        cls.rust = serve(cls, cls.rs_model, "--seed", "1", "--log", "warn", "--ollama-model", "fake:latest",
                         env=env)

    def setUp(self):
        self.teacher.fail_with = None
        self.teacher.requests.clear()
        self.sync()

    def bodies(self):
        seen = [body for _method, _path, body in self.teacher.requests if body]
        self.teacher.requests.clear()
        return seen

    def sync(self):
        """Hands the Python server's model to the Rust one, so both complete the same exercises the same way."""
        path = os.path.join(self.dir, "sync.count.json")
        status, doc, _ = self.python.post("/api/save", {"path": path})
        self.assertEqual(status, 200, doc)
        status, doc, _ = self.rust.post("/api/load", {"path": path})
        self.assertEqual(status, 200, doc)

    def both(self, method, path, body=None):
        """The same request to both servers: ``((status, doc) python, (status, doc) rust)`` and the LLM calls."""
        self.teacher.requests.clear()
        py_answer = self.python.request(method, path, body=body)[:2]
        py_seen = self.bodies()
        rs_answer = self.rust.client.request(method, path, body=body)[:2]
        return py_answer, rs_answer, py_seen, self.bodies()

    def wait(self, client, timeout=120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, job, _ = client.get("/api/job")
            self.assertEqual(status, 200)
            if job and job.get("state") != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("the job did not finish")

    def test_the_tutor_is_described_alike(self):
        (s1, a), (s2, b), _, _ = self.both("GET", "/api/tutor")
        self.assertEqual((s1, s2), (200, 200))
        self.assertEqual(list(a), list(b))
        for key in ("url", "model", "providers", "error_types", "modes", "twonrl_per", "levels", "words_ladder",
                    "upgrade_steps", "plan_lessons"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(list(a["defaults"]), list(b["defaults"]))
        self.assertEqual({**a["defaults"], "tutor_model": ""}, {**b["defaults"], "tutor_model": ""})
        _, status, _ = self.rust.get("/api/status")
        for route in ("GET /api/tutor", "POST /api/tutor/start", "POST /api/tutor/lesson", "POST /api/tutor/plan",
                      "GET /api/tutor/history", "POST /api/chat/start", "GET /api/chat/history"):
            self.assertIn(route, status["routes"])

    def test_a_lesson_is_set_completed_and_marked_alike(self):
        body = {"topic": "animals", "exercises": 2, "mode": "beam", "threshold": 9.5}
        (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/tutor/lesson", body)
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(py_seen, rs_seen)
        close(self, without_timing(a), without_timing(b), "lesson")
        self.assertEqual(b["source"], "ollama")
        # given prefixes skip the exercise writer
        body = {"prefixes": ["the cat sat on", "the dogs run"], "exercises": 5, "mode": "beam"}
        (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/tutor/lesson", body)
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(py_seen, rs_seen)
        self.assertFalse([x for x in rs_seen if "writing exercises" in x["system"]])
        close(self, without_timing(a), without_timing(b), "given")
        self.assertEqual(b["source"], "given")

    def test_a_plan_is_written_alike(self):
        card = {"lessons": 6, "passed": 1, "pass_rate": 1 / 6, "mean_score": 3.0,
                "errors": {"agreement": 4, "tense": 1}}
        body = {"report": card, "topic": "animals", "count": 2, "exercises": 3}
        (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/tutor/plan", body)
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(py_seen, rs_seen)
        close(self, a, b, "plan")
        self.assertEqual(b["plan"]["upgrade"]["step"], "hold")

    def test_refusals_match(self):
        for path, body, status in (
            ("/api/tutor/start", {"topic": " "}, 400),
            ("/api/tutor/start", {"mode": "nope"}, 400),
            ("/api/tutor/start", {"twonrl_per": "hourly"}, 400),
            ("/api/tutor/start", {"threshold": 99}, 400),
            ("/api/tutor/start", {"exercises": 0}, 400),
            ("/api/tutor/start", {"tutor_provider": "bard"}, 400),
            ("/api/tutor/start", {"strength": -1}, 400),
            ("/api/tutor/plan", {"report": "a card", "topic": "animals"}, 400),
            ("/api/tutor/plan", {"report": {"lessons": 3}, "count": 0}, 400),
            ("/api/tutor/plan", {"report": {"lessons": 0}}, 400),
            ("/api/tutor/plan", {"report": {"lessons": 2, "errors": {"tense": 1}}, "url": closed_port_url(),
                                 "timeout": 2}, 502),
            ("/api/tutor/lesson", {"topic": "animals", "url": closed_port_url(), "timeout": 2}, 502),
            ("/api/chat/start", {"turns": 0}, 400),
            ("/api/chat/start", {"provider": "nobody"}, 400),
            ("/api/chat/start", {"mode": "dijkstra"}, 400),
        ):
            with self.subTest(path=path, body=body):
                (s1, a), (s2, b), _, _ = self.both("POST", path, body)
                self.assertEqual((s1, s2), (status, status), (a, b))
                if status == 400:
                    self.assertEqual(a["error"], b["error"])

    def test_a_tutor_job_runs_and_its_history_is_kept(self):
        body = {"topic": "animals", "rounds": 1, "exercises": 2, "mode": "beam", "threshold": 9.5, "drills": 2,
                "blame": True, "variants": 2, "neg_epochs": 1, "pos_epochs": 1, "plan": 2}
        status, a, _ = self.python.post("/api/tutor/start", body)
        self.assertEqual(status, 202, a)
        self.wait(self.python)
        py_seen = self.bodies()
        status, b, _ = self.rust.post("/api/tutor/start", body)
        self.assertEqual(status, 202, b)
        self.assertEqual(list(a), list(b))
        close(self, {**a["config"], "tutor_model": ""}, {**b["config"], "tutor_model": ""}, "config")
        self.assertEqual(b["job"]["type"], "tutor")
        job = self.wait(self.rust.client)
        self.assertEqual((job["type"], job["error"]), ("tutor", None))
        self.assertEqual(py_seen, self.bodies())  # the same lessons, asked about the same way

        _, py_history, _ = self.python.get("/api/tutor/history")
        status, rs_history, _ = self.rust.get("/api/tutor/history")
        self.assertEqual(status, 200)
        # this run's records: the history keeps every run's, the job only its own
        mine = without_timing(rs_history["history"][-len(job["history"]):])
        theirs = without_timing(py_history["history"][-len(mine):])
        self.assertEqual([r["kind"] for r in mine], ["lesson", "lesson", "round", "report", "plan"])
        self.assertEqual(without_timing(job["history"]), mine)
        for first, second in zip(theirs, mine):
            # what the negative network had already been taught differs from server to server
            if first["kind"] == "round":
                for key in ("negative_blamed", "negative_edges", "negative_reasons"):
                    first.pop(key, None)
                    second.pop(key, None)
            close(self, first, second, first["kind"])
        # and a plan can be asked for from the card the run ended with
        (s1, a), (s2, b), _, _ = self.both("POST", "/api/tutor/plan", {"topic": "animals", "count": 1})
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(b["report"]["kind"], "report")
        self.assertEqual(len(b["plan"]["lessons"]), 1)

    def test_a_running_auto_run_stops_with_its_card_and_blocks_a_second_job(self):
        body = {"topic": "animals", "rounds": 1, "exercises": 2, "mode": "beam", "batches": 0, "learn": False}
        status, doc, _ = self.rust.post("/api/tutor/start", body)
        self.assertEqual(status, 202, doc)
        status, doc, _ = self.rust.post("/api/tutor/start", body)
        self.assertEqual(status, 409, doc)
        status, doc, _ = self.rust.post("/api/chat/start", {"turns": 1})
        self.assertEqual(status, 409, doc)
        self.rust.post("/api/job/stop")
        job = self.wait(self.rust.client)
        self.assertEqual((job["type"], job["error"]), ("tutor", None))
        kinds = [r["kind"] for r in job["history"]]
        self.assertIn("report", kinds)  # a batch always ends with its card, stopped or not
        self.assertIn(kinds[-1], ("report", "plan"))  # and no batch is announced that will not run

    def test_a_chat_job_runs_and_its_history_is_kept(self):
        partner = start_partner(self.addCleanup)
        body = {"conversations": 1, "turns": 2, "model": "fake:latest", "url": partner.url, "topic": "animals",
                "learn": False, "blame": False}
        status, a, _ = self.python.post("/api/chat/start", body)
        self.assertEqual(status, 202, a)
        self.wait(self.python)
        partner.requests.clear()
        partner.lines = 0
        status, b, _ = self.rust.post("/api/chat/start", body)
        self.assertEqual(status, 202, b)
        self.assertEqual(list(a), list(b))
        close(self, a["config"], b["config"], "config")
        self.assertEqual((a["url"], a["partner"], a["judge"], a["speakers"]),
                         (b["url"], b["partner"], b["judge"], b["speakers"]))
        self.assertEqual(b["job"]["type"], "chat")
        job = self.wait(self.rust.client)
        self.assertEqual((job["type"], job["error"]), ("chat", None))
        _, py_history, _ = self.python.get("/api/chat/history")
        status, rs_history, _ = self.rust.get("/api/chat/history")
        self.assertEqual(status, 200)
        mine = without_timing(rs_history["history"][-len(job["history"]):])
        self.assertEqual([r["kind"] for r in mine], ["exchange", "exchange", "conversation", "report"])
        close(self, without_timing(py_history["history"][-len(mine):]), mine, "history")


if __name__ == "__main__":
    unittest.main()
