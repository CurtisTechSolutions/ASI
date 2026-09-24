"""Cross-language parity of the Rust port's LLM clients, the review and the negative network's automatic loop.

Skipped when no Rust toolchain is available.  The fake servers of the Python
suites stand in for the LLMs - the fake Ollama of ``test_ollama`` reviews and
writes corpora, the fake OpenAI API of ``test_chatgpt`` answers as ChatGPT -
and both ports are pointed at the same one, so what has to agree is:

* **the conversation** - every request body the fake receives from Rust is the
  one it received from Python: the same prompt and system prompt byte for
  byte, the same model, options and format;
* **what comes back** - the same reviews, verdicts, marks and corpus lines, and
  the same JSON documents from the CLI and from the server routes the frontend
  calls (``/api/ollama/*``, ``/api/chatgpt/models``, ``/api/negative/auto``);
* **the effect on the models** - the negative network the automatic loop or a
  blamed review teaches is Python's byte for byte (but for the ``version``
  cache stamp), and the positive model comes out untouched.

It mirrors ``TestGoCriticParity`` of ``test_go_parity.py`` and goes further,
because the Rust port's documents are held to Python's byte for byte.
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
    from test_chatgpt import KEY  # noqa: E402
    from test_chatgpt import start_fake as start_fake_openai  # noqa: E402
    from test_ollama import closed_port_url, start_fake  # noqa: E402
except ImportError:
    from tests.test_chatgpt import KEY  # noqa: E402
    from tests.test_chatgpt import start_fake as start_fake_openai  # noqa: E402
    from tests.test_ollama import closed_port_url, start_fake  # noqa: E402


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


class TestRustCriticParity(unittest.TestCase):
    """One fake reviewer, both loops: the same prompts, the same marks, the same blame."""

    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.env = {**os.environ, "OLLAMA_HOST": self.fake.url, "RADIXNET_OLLAMA_MODEL": "fake:latest",
                    "PYTHONWARNINGS": "ignore"}
        self.py_path, self.rs_path = trained_pair("critic")

    def bodies(self):
        """Every request body the fake has seen since the last call, which is then forgotten."""
        seen = [(path, body) for _method, path, body in self.fake.requests]
        self.fake.requests.clear()
        return seen

    def both(self, *options, py_model=None, rs_model=None):
        """The same command on both sides; ``(python doc, rust doc, python requests, rust requests)``."""
        self.fake.requests.clear()
        a = py(*options, model=py_model or self.py_path, env=self.env)
        py_seen = self.bodies()
        b = rust(*options, model=rs_model or self.rs_path, env=self.env)
        return a, b, py_seen, self.bodies()

    def test_both_loops_ask_the_same_and_blame_the_same(self):
        options = ("negative", "auto", "--rounds", 2, "--count", 4, "--max-length", 40, "--threshold", 6)
        a, b, py_seen, rs_seen = self.both(*options)

        # the same texts written, so the same conversation with the reviewer, body for body
        self.assertEqual(len(py_seen), 2)
        self.assertEqual(py_seen, rs_seen)

        # the same rounds, marked the same and blamed the same
        py_rounds = [r for r in a["records"] if r["kind"] == "round"]
        rs_rounds = [r for r in b["records"] if r["kind"] == "round"]
        self.assertEqual(len(py_rounds), 2)
        self.assertEqual(len(py_rounds), len(rs_rounds))
        for first, second in zip(py_rounds, rs_rounds):
            self.assertEqual(set(first), set(second))
            for key in ("round", "reviewer", "threshold", "texts", "reviews", "passed", "failed", "blamed", "cleared",
                        "unmatched", "edges", "reasons", "pass_rate"):
                self.assertEqual(first[key], second[key], f"round {first['round']}: {key}")
            self.assertAlmostEqual(first["mean_rating"], second["mean_rating"], places=12)
            self.assertAlmostEqual(first["severity_mean"], second["severity_mean"], places=12)

        # the same report card and settings
        self.assertEqual(set(a["report"]), set(b["report"]))
        for key in ("kind", "rounds", "reviewed", "blamed", "cleared", "edges", "reasons", "pass_rate", "trend"):
            self.assertEqual(a["report"][key], b["report"][key], key)
        self.assertAlmostEqual(a["report"]["mean_rating"], b["report"]["mean_rating"], places=12)
        self.assertEqual(a["config"], b["config"])
        self.assertEqual((a["url"], a["reviewer"]), (b["url"], b["reviewer"]))
        self.assertEqual(set(a), set(b))
        self.assertEqual(set(a["negative"]), set(b["negative"]))
        self.assertEqual(a["negative"]["reasons"], b["negative"]["reasons"])

        # and the same negative network on disk, byte for byte
        py_neg, rs_neg = negative_beside(self.py_path), negative_beside(self.rs_path)
        self.assertEqual(graph_of(py_neg), graph_of(rs_neg))
        py_doc, rs_doc = load_json(py_neg), load_json(rs_neg)
        self.assertEqual(py_doc["format"], rs_doc["format"])
        self.assertEqual([(r["reason"], r["note"], r["source"]) for r in py_doc["log"]],
                         [(r["reason"], r["note"], r["source"]) for r in rs_doc["log"]])
        self.assertEqual({r["source"] for r in rs_doc["log"]}, {"critic"})

    def test_the_positive_model_comes_out_untouched_on_both_sides(self):
        before = (load_json(self.py_path), load_json(self.rs_path))
        options = ("negative", "auto", "--rounds", 1, "--count", 3, "--max-length", 40)
        self.both(*options)
        self.assertEqual(before[0], load_json(self.py_path))
        self.assertEqual(before[1], load_json(self.rs_path))

    def test_a_context_a_prefix_and_a_reviewer_model_reach_the_prompt_the_same_way(self):
        options = ("negative", "auto", "--rounds", 1, "--count", 3, "--max-length", 30, "--prefix", "the cat",
                   "--context", "nursery rhymes", "--reviewer-model", "other:7b", "--threshold", 6.5, "--no-clear")
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        body = rs_seen[0][1]
        self.assertTrue(body["prompt"].startswith("Context: nursery rhymes\n\n"))
        self.assertIn("a rating of 6.5 or more", body["system"])
        self.assertEqual(body["model"], "other:7b")
        self.assertEqual(b["reviewer"], "other:7b")
        self.assertEqual(a["config"], b["config"])

    def test_both_reviews_mark_the_same_given_texts(self):
        texts = ["the cat sat on the mat", "a good one", "xxxx xxxx xxxx", "   "]
        options = ("ollama", "review", "--threshold", 6, "--context", "plain English",
                   *sum((("--text", t) for t in texts), ()))
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(a, b)  # the whole document: reviews, marks, the split, the mean

    def test_both_reviews_sample_the_same_texts_from_the_model(self):
        options = ("--seed", 3, "ollama", "review", "--count", 3, "--max-length", 30)
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(a["source"], "model")
        self.assertEqual(a, b)
        # sampling needs a model: none is an error on both sides
        missing = os.path.join(tmpdir(), "missing.count.json")
        py("ollama", "review", "--count", 2, model=missing, env=self.env, expect=1)
        rust("ollama", "review", "--count", 2, model=missing, env=self.env, expect=1)

    def test_a_blamed_review_teaches_what_python_teaches(self):
        from radixnet import blame
        from radixnet.negative import NegativeNet

        texts = ["the cat sat on the mat", "a good one", "zzz nonsense zzz"]
        given = sum((("--text", t) for t in texts), ())
        result = py("ollama", "review", *given, model=self.py_path, env=self.env)
        py_seen = self.bodies()
        # Python's own `--blame` stops at a KeyError ('lessons') after teaching, so the lesson is given here
        negative = NegativeNet(seed=0)
        report = blame.teach_reviews(negative, result, threshold=6.0, source="review")
        expected = os.path.join(tmpdir(), "blamed_py.negative.json")
        negative.save(expected)

        out = rust("ollama", "review", *given, "--blame", model=self.rs_path, env=self.env)
        self.assertEqual(py_seen, self.bodies())
        self.assertEqual(graph_of(expected), graph_of(negative_beside(self.rs_path)))
        taught = out["negative"]
        self.assertEqual((taught["blamed"], taught["cleared"], taught["edges"]),
                         (report["blamed"], report["cleared"], report["edges"]))
        self.assertEqual(taught["reasons"], negative.reasons())
        self.assertEqual([f["text"] for f in taught["lessons"]], [f["text"] for f in report["faults"]])
        self.assertEqual(taught["path"], negative_beside(self.rs_path))

    def test_both_editors_correct_the_same_given_texts(self):
        texts = ["Hi howe are you??", "the cat sat on the mat", "howe??", "   "]
        options = ("ollama", "correct", "--context", "short greetings", *sum((("--text", t) for t in texts), ()))
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)  # the editor's prompt, byte for byte
        self.assertEqual(a, b)  # the whole document: the corrections, their diffs, reasons and the counts
        self.assertEqual([c["verdict"] for c in b["corrections"]], ["corrected", "unchanged", "corrected", "uncorrected"])
        self.assertEqual(b["corrections"][0]["correction"], "Hi, how are you?")
        self.assertEqual([(c["op"], c["wrong"], c["right"]) for c in b["corrections"][0]["changes"]],
                         [("insert", "", ","), ("delete", "e", ""), ("delete", "?", "")])

    def test_both_editors_correct_the_same_samples_from_the_model(self):
        options = ("--seed", 3, "ollama", "correct", "--count", 3, "--max-length", 30)
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(a["source"], "model")
        self.assertEqual(a, b)
        missing = os.path.join(tmpdir(), "missing.count.json")
        py("ollama", "correct", "--count", 2, model=missing, env=self.env, expect=1)
        rust("ollama", "correct", "--count", 2, model=missing, env=self.env, expect=1)

    def test_a_blamed_correction_teaches_what_python_teaches(self):
        texts = ["Hi howe are you??", "the cat sat on the mat", "howe now"]
        given = sum((("--text", t) for t in texts), ())
        options = ("ollama", "correct", *given, "--blame", "--severity", 1.5)
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(graph_of(negative_beside(self.py_path)), graph_of(negative_beside(self.rs_path)))
        taught_a, taught_b = a["negative"], b["negative"]
        self.assertEqual(set(taught_a), set(taught_b))
        for key in ("blamed", "cleared", "edges", "edits", "reasons", "lessons"):
            self.assertEqual(taught_a[key], taught_b[key], key)
        self.assertEqual((taught_b["blamed"], taught_b["cleared"]), (2, 1))
        self.assertEqual([r["reason"] for r in taught_b["reasons"]], ["spelling"])
        self.assertEqual(taught_b["lessons"][0]["severity"], 1.5)
        # only the changed characters are known failures on either side
        for side, path in ((py, self.py_path), (rust, self.rs_path)):
            verdicts = side("negative", "why", "--text", "Hi, how are you?", "--text", "Hi howe are you??",
                            model=path, env=self.env)["verdicts"]
            self.assertEqual(verdicts[0]["verdict"], "pass")
            self.assertEqual(verdicts[1]["reasons"][0]["reason"], "spelling")

    def test_both_correcting_loops_ask_the_same_and_blame_the_same(self):
        options = ("negative", "auto", "--rounds", 2, "--count", 4, "--max-length", 40, "--correct", "--severity", 2)
        a, b, py_seen, rs_seen = self.both(*options)
        self.assertEqual(len(py_seen), 2)
        self.assertEqual(py_seen, rs_seen)
        py_rounds = [r for r in a["records"] if r["kind"] == "round"]
        rs_rounds = [r for r in b["records"] if r["kind"] == "round"]
        self.assertEqual(len(py_rounds), 2)
        self.assertEqual(len(py_rounds), len(rs_rounds))
        for first, second in zip(py_rounds, rs_rounds):
            self.assertEqual(set(first), set(second))
            self.assertEqual(first["mode"], "correct")
            for key in ("round", "reviewer", "mode", "severity", "texts", "corrections", "corrected", "unchanged",
                        "uncorrected", "edits", "wrong_chars", "blamed", "cleared", "unmatched", "edges", "reasons",
                        "change_rate"):
                self.assertEqual(first[key], second[key], f"round {first['round']}: {key}")
            self.assertAlmostEqual(first["severity_mean"], second["severity_mean"], places=12)
        self.assertEqual(set(a["report"]), set(b["report"]))
        for key in ("kind", "rounds", "reviewed", "blamed", "cleared", "edges", "reasons", "corrected", "unchanged",
                    "uncorrected", "edits", "change_rate", "change_trend", "mean_rating", "pass_rate", "trend"):
            self.assertEqual(a["report"][key], b["report"][key], key)
        self.assertEqual(a["config"], b["config"])
        self.assertEqual((a["config"]["correct"], a["config"]["severity"]), (True, 2.0))
        self.assertEqual(graph_of(negative_beside(self.py_path)), graph_of(negative_beside(self.rs_path)))

    def test_a_corpus_is_asked_for_and_trained_on_the_same_way(self):
        py_out = os.path.join(tmpdir(), "corpus_py.txt")
        rs_out = os.path.join(tmpdir(), "corpus_rs.txt")
        options = ("ollama", "--ollama-model", "other:7b", "corpus", "--prompt", "mountains", "--lines", 3,
                   "--style", "garbage")
        self.fake.requests.clear()
        a = py(*options, "--out", py_out, model=self.py_path, env=self.env)
        py_seen = self.bodies()
        b = rust(*options, "--out", rs_out, model=self.rs_path, env=self.env)
        self.assertEqual(py_seen, self.bodies())
        self.assertEqual(a["texts"], [f"garbage line {i} about mountains" for i in range(1, 4)])
        self.assertEqual({**a, "out": None}, {**b, "out": None})
        with open(py_out, encoding="utf-8") as one, open(rs_out, encoding="utf-8") as other:
            self.assertEqual(one.read(), other.read())

        # --train: the lines are trained on and the model saved, the same model on both sides
        py_new = os.path.join(tmpdir(), "corpus_py.count.json")
        rs_new = os.path.join(tmpdir(), "corpus_rs.count.json")
        fresh(py_new)
        fresh(rs_new)
        a = py("--kind", "count", "--seed", 1, "ollama", "corpus", "--prompt", "rivers", "--lines", 4, "--train",
               "--epochs", 2, model=py_new, env=self.env)
        b = rust("--seed", 1, "ollama", "corpus", "--prompt", "rivers", "--lines", 4, "--train", "--epochs", 2,
                 model=rs_new, env=self.env)
        self.assertEqual(a["texts"], b["texts"])
        self.assertEqual(len(b["trained"]["epochs"]), 2)
        self.assertEqual(b["trained"]["stats"]["trained_texts"], a["trained"]["stats"]["trained_texts"])
        self.assertEqual(set(a["trained"]), set(b["trained"]))
        self.assertEqual(graph_of(py_new), graph_of(rs_new))

    def test_a_failed_review_feeds_2nrl_the_same_way(self):
        good = os.path.join(tmpdir(), "good.txt")
        with open(good, "w", encoding="utf-8") as fh:
            fh.write("the cat sat on the mat\n")
        py_out = os.path.join(tmpdir(), "twonrl_py.count.json")
        rs_out = os.path.join(tmpdir(), "twonrl_rs.count.json")
        options = ("ollama", "review", "--text", "zzz nonsense", "--text", "good text", "--2nrl", "--good", good,
                   "--neg-epochs", 1, "--pos-epochs", 1)
        a = py(*options, "--out", py_out, model=self.py_path, env=self.env)
        b = rust(*options, "--out", rs_out, model=self.rs_path, env=self.env)
        for key in ("bad_texts", "good_texts", "inverted", "interrupted"):
            self.assertEqual(a["two_nrl"][key], b["two_nrl"][key], key)
        self.assertEqual((b["two_nrl"]["bad_texts"], b["two_nrl"]["good_texts"]), (1, 2))
        self.assertEqual(graph_of(py_out), graph_of(rs_out))
        # nothing passed and no --good: there is no positive phase, on either side
        py("ollama", "review", "--text", "zzz", "--2nrl", model=self.py_path, env=self.env, expect=1)
        rust("ollama", "review", "--text", "zzz", "--2nrl", model=self.rs_path, env=self.env, expect=1)

    def test_the_models_are_listed_the_same_and_a_missing_ollama_is_an_error(self):
        a, b, py_seen, rs_seen = self.both("ollama", "models")
        self.assertEqual(a, b)
        self.assertEqual([m["name"] for m in b["models"]], ["fake:latest", "other:7b"])
        self.assertEqual(py_seen, rs_seen)
        down = ("ollama", "--url", closed_port_url(), "--timeout", 2, "models")
        py(*down, model=self.py_path, env=self.env, expect=1)
        err = rust(*down, model=self.rs_path, env=self.env, expect=1)
        self.assertIn("cannot reach Ollama", err["error"])

    def test_an_ollama_error_is_reported_not_swallowed(self):
        self.fake.fail_with = 500
        err = rust("ollama", "review", "--text", "x", model=self.rs_path, env=self.env, expect=1)
        self.assertIn("HTTP 500", err["error"])
        self.assertIn("boom", err["error"])
        py("ollama", "review", "--text", "x", model=self.py_path, env=self.env, expect=1)


class TestRustChatGPTParity(unittest.TestCase):
    """The fake OpenAI API answers both: the same bodies, the same key, the same answers."""

    def setUp(self):
        self.fake = start_fake_openai(self.addCleanup)
        self.env = {**os.environ, "OPENAI_BASE_URL": self.fake.url, "RADIXNET_OPENAI_MODEL": "fake-gpt",
                    "OPENAI_API_KEY": KEY, "PYTHONWARNINGS": "ignore"}
        self.env.pop("OPENAI_API_KEY_FILE", None)
        self.model = os.path.join(tmpdir(), "chatgpt.count.json")

    def seen(self):
        seen = [(method, path, body, auth) for method, path, body, auth in self.fake.requests]
        self.fake.requests.clear()
        return seen

    def test_the_models_are_listed_the_same(self):
        a = py("chatgpt", "models", model=self.model, env=self.env)
        py_seen = self.seen()
        b = rust("chatgpt", "models", model=self.model, env=self.env)
        self.assertEqual(py_seen, self.seen())
        self.assertEqual(a, b)
        self.assertEqual(b["url"], self.fake.url + "/v1")
        self.assertEqual([m["name"] for m in b["models"]], ["another-gpt", "fake-gpt"])

    def test_one_question_is_asked_the_same_way_with_the_same_key(self):
        options = ("chatgpt", "ask", "--prompt", "what is 2 + 2", "--system", "be brief")
        a = py(*options, model=self.model, env=self.env)
        py_seen = self.seen()
        b = rust(*options, model=self.model, env=self.env)
        rs_seen = self.seen()
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(rs_seen[0][3], f"Bearer {KEY}")
        self.assertEqual(rs_seen[0][2]["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(rs_seen[0][2]["temperature"], 0.7)
        self.assertEqual(a, b)
        self.assertEqual(b["answer"], "answer to: what is 2 + 2")

    def test_a_model_that_refuses_a_field_is_asked_again_without_it(self):
        self.fake.reject = ("temperature",)
        options = ("chatgpt", "ask", "--prompt", "hi")
        a = py(*options, model=self.model, env=self.env)
        py_seen = self.seen()
        b = rust(*options, model=self.model, env=self.env)
        self.assertEqual(py_seen, self.seen())
        self.assertEqual(len(py_seen), 2)
        self.assertEqual(a["answer"], b["answer"])

    def test_without_a_key_nothing_is_sent(self):
        env = {k: v for k, v in self.env.items() if k != "OPENAI_API_KEY"}
        py("chatgpt", "models", model=self.model, env=env, expect=1)
        err = rust("chatgpt", "models", model=self.model, env=env, expect=1)
        self.assertIn("OPENAI_API_KEY", err["error"])
        err = rust("negative", "auto", "--provider", "chatgpt", model=self.model, env=env, expect=1)
        self.assertIn("OPENAI_API_KEY", err["error"])
        self.assertEqual(self.fake.requests, [])

    def test_a_key_is_never_sent_unencrypted_to_a_remote_host(self):
        env = {**self.env, "OPENAI_BASE_URL": ""}
        err = rust("chatgpt", "--url", "http://gateway.example.com/v1", "models", model=self.model, env=env,
                   expect=1)
        self.assertIn("unencrypted", err["error"])

    def test_chatgpt_reviews_the_negative_loop_as_ollama_does(self):
        py_path, rs_path = trained_pair("chatgpt_critic")
        options = ("negative", "auto", "--provider", "chatgpt", "--rounds", 1, "--count", 3, "--max-length", 30)
        a = py(*options, model=py_path, env=self.env)
        py_seen = self.seen()
        b = rust(*options, model=rs_path, env=self.env)
        self.assertEqual(py_seen, self.seen())
        self.assertEqual(a["config"], b["config"])
        self.assertEqual(a["reviewer"], b["reviewer"])
        self.assertEqual(graph_of(negative_beside(py_path)), graph_of(negative_beside(rs_path)))


class TestRustLLMServer(unittest.TestCase):
    """``radixnet serve`` answers the frontend's LLM routes as the Python server does."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.fake = start_fake(cls.addClassCleanup)
        cls.openai = start_fake_openai(cls.addClassCleanup)
        cls.dir = tempfile.mkdtemp(prefix="radixnet-rust-llm-")
        cls.addClassCleanup(shutil.rmtree, cls.dir, True)
        trained = os.path.join(cls.dir, "trained.count.json")
        py("--kind", "count", "--seed", 1, "train", "--data", CORPUS, "--epochs", 2, model=trained)
        cls.py_model = os.path.join(cls.dir, "py", "m.count.json")
        cls.rs_model = os.path.join(cls.dir, "rs", "m.count.json")
        for path in (cls.py_model, cls.rs_model):
            os.makedirs(os.path.dirname(path))
            shutil.copyfile(trained, path)
        # the Python server reads the key from its own process's environment
        cls._key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = KEY
        cls.addClassCleanup(cls._restore_key)
        cls.python, _server, cls.service = start_server(
            cls.addClassCleanup, model_path=cls.py_model, kind="count", upload_dir=os.path.join(cls.dir, "py-up"),
            ollama_url=cls.fake.url, ollama_model="fake:latest", chatgpt_url=cls.openai.url,
            chatgpt_model="fake-gpt", seed=1,
        )
        env = {**os.environ, "OLLAMA_HOST": cls.fake.url, "OPENAI_BASE_URL": cls.openai.url,
               "RADIXNET_OPENAI_MODEL": "fake-gpt", "OPENAI_API_KEY": KEY}
        env.pop("OPENAI_API_KEY_FILE", None)
        cls.rust = serve(cls, cls.rs_model, "--seed", "1", "--log", "warn", "--upload-dir",
                         os.path.join(cls.dir, "rs-up"), "--ollama-model", "fake:latest", env=env)

    @classmethod
    def _restore_key(cls):
        if cls._key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = cls._key

    def setUp(self):
        self.fake.fail_with = None
        self.fake.requests.clear()
        self.openai.requests.clear()

    def bodies(self):
        seen = [(path, body) for _method, path, body in self.fake.requests]
        self.fake.requests.clear()
        return seen

    def wait(self, client, timeout=120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, job, _ = client.get("/api/job")
            self.assertEqual(status, 200)
            if job and job.get("state") != "running":
                return job
            time.sleep(0.05)
        raise AssertionError("the job did not finish")

    def sync(self):
        """Hands the Python server's model to the Rust one, so a comparison that samples walks one graph.

        The tests that train (a corpus job, a 2NRL job) change one server's
        model only, and the order the tests run in is not theirs to rely on.
        """
        path = os.path.join(self.dir, "sync.count.json")
        status, doc, _ = self.python.post("/api/save", {"path": path})
        self.assertEqual(status, 200, doc)
        status, doc, _ = self.rust.post("/api/load", {"path": path})
        self.assertEqual(status, 200, doc)

    def both(self, method, path, body=None):
        """The same request to both servers: ``((status, doc) python, (status, doc) rust)`` and the LLM calls."""
        self.fake.requests.clear()
        py_answer = self.python.request(method, path, body=body)[:2]
        py_seen = self.bodies()
        rs_answer = self.rust.client.request(method, path, body=body)[:2]
        return py_answer, rs_answer, py_seen, self.bodies()

    def test_the_status_names_both_providers(self):
        _, py_status, _ = self.python.get("/api/status")
        status, rs_status, _ = self.rust.get("/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(rs_status["ollama"], py_status["ollama"])
        self.assertEqual(rs_status["chatgpt"], py_status["chatgpt"])
        self.assertEqual(rs_status["ollama"], {"url": self.fake.url, "model": "fake:latest"})
        self.assertIn("POST /api/negative/auto", rs_status["routes"])

    def test_the_models_routes_answer_alike(self):
        (s1, a), (s2, b), py_seen, rs_seen = self.both("GET", "/api/ollama/models")
        self.assertEqual((s1, s2), (200, 200))
        self.assertEqual(a, b)
        self.assertEqual(set(b["models"][0]), {"name", "size", "modified_at", "details"})
        down = closed_port_url()
        (s1, a), (s2, b), _, _ = self.both("GET", "/api/ollama/models?url=" + down)
        self.assertEqual((s1, s2), (200, 200))
        self.assertEqual((b["available"], b["models"], b["url"]), (False, [], down))
        self.assertIn("cannot reach Ollama", b["error"])
        self.assertEqual(set(a), set(b))
        (s1, a), (s2, b), _, _ = self.both("GET", "/api/chatgpt/models")
        self.assertEqual((s1, s2), (200, 200))
        self.assertEqual(a, b)
        self.assertEqual([m["name"] for m in b["models"]], ["another-gpt", "fake-gpt"])

    def test_a_corpus_answers_alike(self):
        body = {"prompt": "the sea", "lines": 3, "style": "garbage", "model": "other:7b"}
        (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/ollama/corpus", body)
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(a, b)
        self.assertEqual(py_seen, rs_seen)
        # saved as an upload, and trained on in a job
        body = {"prompt": "the sea", "lines": 2, "save_as": "sea.txt", "train": True, "epochs": 1}
        status, doc, _ = self.rust.post("/api/ollama/corpus", body)
        self.assertEqual(status, 202, doc)
        self.assertEqual((doc["upload"]["name"], doc["upload"]["lines"]), ("sea.txt", 2))
        self.assertEqual(set(doc["upload"]), {"name", "bytes", "chars", "lines", "modified", "replaced"})
        self.assertEqual(doc["job"]["type"], "train")
        job = self.wait(self.rust.client)
        self.assertEqual((job["type"], job["error"]), ("train", None))
        _, uploads, _ = self.rust.get("/api/uploads")
        self.assertIn("sea.txt", [u["name"] for u in uploads["uploads"]])

    def test_corpus_refusals_match(self):
        for body, status, words in (
            ({"lines": 3}, 400, "prompt"),
            ({"prompt": "x", "style": "weird"}, 400, "style"),
            ({"prompt": "x", "url": "   "}, 400, "empty"),
            ({"prompt": "x", "lines": 0}, 400, "lines"),
            ({"prompt": "x", "url": closed_port_url(), "timeout": 2}, 502, "cannot reach Ollama"),
        ):
            with self.subTest(body=body):
                (s1, a), (s2, b), _, _ = self.both("POST", "/api/ollama/corpus", body)
                self.assertEqual((s1, s2), (status, status), (a, b))
                self.assertIn(words, b["error"])
        self.fake.fail_with = 500
        (s1, a), (s2, b), _, _ = self.both("POST", "/api/ollama/corpus", {"prompt": "x"})
        self.assertEqual((s1, s2), (502, 502))
        self.assertEqual(a["error"], b["error"])

    def test_a_review_answers_alike(self):
        body = {"texts": ["a good one", "zzz nonsense", "a good two"], "threshold": 6, "context": "plain"}
        (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/ollama/review", body)
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(a, b)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(set(b["reviews"][0]), {"index", "text", "rating", "verdict", "critique"})
        # the model's own samples, seeded: the same texts, so the same review
        self.sync()
        body = {"count": 3, "max_length": 30, "seed": 5}
        (s1, a), (s2, b), py_seen, rs_seen = self.both("POST", "/api/ollama/review", body)
        self.assertEqual((s1, s2), (200, 200), b)
        self.assertEqual(py_seen, rs_seen)
        self.assertEqual(a, b)
        self.assertEqual(b["source"], "model")

    def test_a_review_blames_and_feeds_2nrl(self):
        status, doc, _ = self.rust.post("/api/ollama/review", {"texts": ["a good one", "zzz nonsense"],
                                                               "blame": True})
        self.assertEqual(status, 200, doc)
        taught = doc["negative"]
        self.assertEqual((taught["blamed"], taught["reasons"]), (1, {"other": 1}))
        self.assertEqual(set(taught), {"blamed", "cleared", "unmatched", "edges", "reasons", "lessons",
                                       "severity_mean", "stats", "reason_table"})
        self.assertEqual(taught["reason_table"][0]["reason"], "other")
        self.rust.post("/api/uploads", {"name": "good.txt", "content": "the cat sat on the mat\n"})
        body = {"texts": ["a good one", "zzz nonsense"], "apply": "2nrl", "good": ["one more good line"],
                "good_files": ["good.txt"], "neg_epochs": 1, "pos_epochs": 1}
        status, doc, _ = self.rust.post("/api/ollama/review", body)
        self.assertEqual(status, 202, doc)
        self.assertEqual(doc["job"]["type"], "2nrl")
        job = self.wait(self.rust.client)
        self.assertEqual((job["type"], job["error"]), ("2nrl", None))

    def test_review_refusals_match(self):
        for body, status, words in (
            ({"texts": ["x"], "apply": "maybe"}, 400, "apply"),
            ({"texts": ["all good", "still good"], "apply": "2nrl"}, 400, "nothing failed"),
            ({"texts": ["zzz", "qqq"], "apply": "2nrl"}, 400, "no text passed"),
            ({"count": 0}, 400, "count"),
            ({"texts": "not a list"}, 400, "must be a list of strings"),
            ({"texts": ["x"], "threshold": -1}, 400, "threshold"),
        ):
            with self.subTest(body=body):
                (s1, a), (s2, b), _, _ = self.both("POST", "/api/ollama/review", body)
                self.assertEqual((s1, s2), (status, status), (a, b))
                self.assertIn(words, b["error"])
                self.assertEqual(a["error"], b["error"])
        self.fake.fail_with = 503
        (s1, a), (s2, b), _, _ = self.both("POST", "/api/ollama/review", {"texts": ["x"]})
        self.assertEqual((s1, s2), (502, 502))
        self.assertEqual(a["error"], b["error"])

    def test_the_automatic_loop_runs_as_a_job_and_asks_what_python_asks(self):
        body = {"rounds": 1, "count": 3, "max_length": 24, "seed": 1}
        self.sync()
        self.fake.requests.clear()
        status, a, _ = self.python.post("/api/negative/auto", body)
        self.assertEqual(status, 202, a)
        self.wait(self.python)
        py_seen = self.bodies()
        status, b, _ = self.rust.post("/api/negative/auto", body)
        self.assertEqual(status, 202, b)
        self.assertEqual(set(a), set(b))
        self.assertEqual(a["config"], b["config"])
        self.assertEqual((a["url"], a["reviewer"]), (b["url"], b["reviewer"]))
        self.assertEqual(b["job"]["type"], "critic")
        job = self.wait(self.rust.client)
        self.assertEqual((job["type"], job["error"]), ("critic", None))
        self.assertEqual([r["kind"] for r in job["history"]], ["round", "report"])
        self.assertEqual(py_seen, self.bodies())  # the same texts, the same review

        _, py_history, _ = self.python.get("/api/negative/auto/history")
        status, rs_history, _ = self.rust.get("/api/negative/auto/history")
        self.assertEqual(status, 200)
        self.assertEqual([r["kind"] for r in rs_history["history"]][-2:], ["round", "report"])
        first, second = py_history["history"][-2], rs_history["history"][-2]
        self.assertEqual(set(first), set(second))
        # what was blamed and why; how much it cleared depends on what each
        # server's negative network had already been taught by other tests
        for key in ("texts", "reviews", "passed", "failed", "blamed", "reasons", "reviewer", "threshold"):
            self.assertEqual(first[key], second[key], key)
        self.assertEqual(set(py_history["history"][-1]), set(rs_history["history"][-1]))

    def test_loop_refusals_match(self):
        for body in ({"count": 0}, {"provider": "gemini"}, {"threshold": 11}, {"rounds": -1}):
            with self.subTest(body=body):
                (s1, a), (s2, b), _, _ = self.both("POST", "/api/negative/auto", body)
                self.assertEqual((s1, s2), (400, 400), (a, b))
                self.assertEqual(a["error"], b["error"])


class TestRustServerWithoutAKey(unittest.TestCase):
    """A server with no OpenAI key says so, and refuses ChatGPT before anything is sent."""

    @classmethod
    def setUpClass(cls):
        cls.openai = start_fake_openai(cls.addClassCleanup)
        model = os.path.join(tmpdir(), "nokey.count.json")
        rust("--seed", 1, "train", "--data", CORPUS, "--epochs", 1, model=model)
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "OPENAI_API_KEY_FILE")}
        env["OPENAI_BASE_URL"] = cls.openai.url
        cls.server = serve(cls, model, "--log", "warn", env=env)

    def test_chatgpt_is_reported_off_and_refused(self):
        status, doc, _ = self.server.get("/api/chatgpt/models")
        self.assertEqual(status, 200, doc)
        self.assertEqual((doc["available"], doc["configured"], doc["models"]), (False, False, []))
        self.assertIn("OPENAI_API_KEY", doc["error"])
        _, st, _ = self.server.get("/api/status")
        self.assertFalse(st["chatgpt"]["configured"])
        status, doc, _ = self.server.post("/api/negative/auto", {"provider": "chatgpt"})
        self.assertEqual(status, 400, doc)
        self.assertIn("not configured", doc["error"])
        self.assertEqual(self.openai.requests, [])


if __name__ == "__main__":
    unittest.main()
