"""Tests for ``radixnet.thinking``: the THINK sentinel at work, in the library, the CLI and the API."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.graph import BACK, FIRST, START, THINK  # noqa: E402
from radixnet.model import new_model  # noqa: E402
from radixnet.thinking import (  # noqa: E402
    ASKED,
    QUESTIONED,
    STUTTER,
    THEN_BACK,
    THEN_END,
    THEN_THINK,
    Thought,
    place,
    questions_in,
    summarize,
    think,
    think_on,
)

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server, wait_for_job  # noqa: F401
except ImportError:
    from tests.test_api import start_server, wait_for_job  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = {"epochs": 3, "lr": 1.0, "batch_size": 1}
TEXTS = ["ha ha ha ha ha", "ha ha ho ho hum", "ha ha and then the cat sat", "the cat sat on the mat"]
THOUGHTS = [
    "okay, the cat sat. is that right? yes it did.",
    "the sky is blue because of light",
    "hmm. what did the cat do? it sat on the mat.",
]


def trained(kind="radix", seed=3):
    model = new_model(kind, seed=seed)
    model.train(TEXTS, **FAST)
    return model


class TestQuestions(unittest.TestCase):
    def test_questions_are_the_sentences_ending_in_a_question_mark(self):
        self.assertEqual(questions_in("Hmm. Is that right? Yes. Why not?"), [(5, "Is that right?"), (25, "Why not?")])
        self.assertEqual(questions_in("no questions here."), [])
        self.assertEqual(questions_in("  really?!  "), [(2, "really?!")])
        self.assertEqual(questions_in("a? b. c?"), [(0, "a?"), (6, "c?")])
        self.assertEqual(questions_in(""), [])
        self.assertEqual(questions_in("???"), [])  # a terminator run with no sentence before it
        self.assertEqual(questions_in("hmm, what did the cat do? it sat."), [(0, "hmm, what did the cat do?")])

    def test_place_cuts_the_node_so_the_prefix_ends_there(self):
        model = trained()
        graph = model.graph
        graph.compress()
        node = place(model, "the cat ")
        self.assertGreaterEqual(node, FIRST)
        # the prefix's last gram is now the node's last: an edge taught from it fires exactly here
        enc = graph.encoding
        _n, offset, _matched = model._locate("the cat ")
        self.assertEqual(offset + enc.n, graph.label_len(node))
        graph.check_invariants(texts=TEXTS)
        self.assertEqual(place(model, "zzz qqq"), -1)  # a text the graph never saw
        self.assertEqual(place(model, ""), -1)


class TestThink(unittest.TestCase):
    def test_a_model_taught_no_thoughts_has_nothing_to_think_with(self):
        model = trained()
        thought = think(model)
        self.assertIsInstance(thought, Thought)
        self.assertEqual((thought.trigger, thought.at, thought.text, thought.stopped), (ASKED, -1, "", "nothing"))
        self.assertEqual((thought.then, thought.taught, thought.handed_over, thought.questioned), (THEN_END, -1, -1, 0))
        self.assertEqual(model.graph.children[THINK], {})
        self.assertIn("nothing to think with", summarize(thought))

    def test_asking_about_a_text_teaches_the_model_to_think_there(self):
        model = trained()
        graph = model.graph
        thought = think(model, about="the cat")
        self.assertGreaterEqual(thought.at, FIRST)
        self.assertEqual(thought.taught, thought.at)
        self.assertIn(THINK, graph.children[thought.at])
        self.assertEqual(graph.count[THINK], 1)
        self.assertEqual(thought.then, THEN_END)  # a thought asked for ends; nothing is backed out of
        self.assertEqual(thought.handed_over, -1)
        self.assertEqual(graph.parents[BACK], {})
        # and the doc says all of it
        doc = thought.to_dict()
        self.assertEqual(doc["trigger"], ASKED)
        self.assertEqual(doc["at"], thought.at)
        self.assertEqual(doc["about"], "the cat")
        self.assertEqual(doc["questions"], [])
        graph.check_invariants(texts=TEXTS)

    def test_learning_off_thinks_without_writing_anything(self):
        model = trained()
        before = model.to_dict()
        thought = think(model, about="the cat", learn=False)
        self.assertGreaterEqual(thought.at, FIRST)
        self.assertEqual(thought.taught, -1)
        self.assertEqual(model.graph.children[THINK], {})
        self.assertEqual(model.graph.parents[THINK], {})
        # place() may split a node to say where the prefix ends, which is structure, not learning:
        # compressed again, the graph is the one it was
        model.graph.compress()
        untouched = type(model).from_dict(before)
        untouched.graph.compress()
        self.assertEqual(model.graph.num_edges(), untouched.graph.num_edges())
        self.assertEqual(model.graph.num_nodes(), untouched.graph.num_nodes())

    def test_think_on_teaches_thoughts_and_where_they_question_themselves(self):
        for kind in ("radix", "count", "resonant"):
            with self.subTest(kind=kind):
                model = trained(kind)
                graph = model.graph
                learned = think_on(model, THOUGHTS, **FAST)
                self.assertEqual(learned["thoughts"], 3)
                self.assertEqual(learned["questions"], 2)  # "is that right?" and "what did the cat do?"
                self.assertEqual(learned["taught"], 2)
                self.assertTrue(learned["epochs"])
                self.assertGreater(len(graph.children[THINK]), 0)
                self.assertGreaterEqual(graph.count[THINK], 2)
                # the thoughts begin at THINK: START saw none of them (a thought opening like a text - "the sky
                # is blue" after "the cat sat" - shares the text's nodes, one graph with two origins)
                start_count = graph.count[START]
                for thought in THOUGHTS:
                    grams = model.encoder.encode(thought)
                    self.assertIsNotNone(graph.node_path(grams, origin=THINK), thought)
                    if not thought.startswith("the "):
                        self.assertIsNone(graph.node_path(grams), thought)
                self.assertEqual(graph.count[START], start_count)
                graph.check_invariants(texts=TEXTS)
                # now it has thoughts to think with
                thought = think(model, learn=False)
                self.assertNotEqual(thought.text, "")
                self.assertEqual(thought.node_ids[0], THINK)
                self.assertNotIn(START, thought.node_ids)
                self.assertIn(thought.stopped, ("end", "length"))
                self.assertGreater(thought.probability, 0.0)
                sampled = think(model, mode="sample", seed=1, learn=False)
                self.assertNotEqual(sampled.text, "")

    def test_a_thought_questions_itself_where_the_model_learned_to_think(self):
        for kind in ("count", "radix", "resonant"):
            with self.subTest(kind=kind):
                self.questions_itself(trained(kind))

    def questions_itself(self, model):
        graph = model.graph
        think_on(model, ["the sky is blue because of light", "why is that so?"], questions=False, **FAST)
        # teach it, by experience, to stop and think on the path of whatever it thinks first: a node that
        # thinks a lot makes the thought through it dearer, so the best thought may change - then teach the
        # new one's path too; within a few rounds the thought it thinks passes a node it has learned to think at
        thought = think(model, learn=False)
        node = -1
        for _round in range(4):
            first = next((n for n in thought.node_ids if n >= FIRST), -1)
            self.assertGreaterEqual(first, FIRST)
            while not graph.thinks_at(first):
                graph.observe_think(first)
            thought = think(model, learn=False)
            if any(n >= FIRST and graph.thinks_at(n) for n in thought.node_ids):
                node = next(n for n in thought.node_ids if n >= FIRST and graph.thinks_at(n))
                break
        self.assertGreaterEqual(node, FIRST, "no thought passed a node it learned to think at")
        self.assertIn(node, thought.node_ids)
        self.assertEqual(thought.questioned, 1)
        question = thought.questions[0]
        self.assertEqual((question.trigger, question.at, question.depth, question.then), (QUESTIONED, node, 1, THEN_THINK))
        self.assertEqual(question.taught, -1)  # it already knew to think here: that is why it asked
        # a question must say something the thought above it did not
        self.assertNotEqual(question.text.strip().lower(), thought.text.strip().lower())
        self.assertIn("questioned itself once", summarize(thought))
        doc = thought.to_dict()
        self.assertEqual(doc["questioned"], 1)
        self.assertEqual(doc["questions"][0]["trigger"], QUESTIONED)
        # the depth is bounded, and so is the number of questions
        shallow = think(model, learn=False, max_depth=0)
        self.assertEqual(shallow.questioned, 0)
        none = think(model, learn=False, max_questions=0)
        self.assertEqual(none.questioned, 0)

    def test_a_thought_out_of_a_repeat_hands_over_to_back_when_it_stops(self):
        model = trained()
        graph = model.graph
        node = graph.lookup("ha ")[0] if graph.lookup("ha ") else [n for n in graph.alive_nodes() if n >= FIRST][0]
        went = next(iter(graph.children[node]))
        thought = think(model, at=node, trigger=STUTTER, went=went)
        self.assertEqual(thought.then, THEN_BACK)
        self.assertEqual((thought.taught, thought.handed_over), (node, node))
        self.assertIn(THINK, graph.children[node])
        self.assertIn(BACK, graph.children[node])
        self.assertIsNotNone(graph.back_cost(node))
        self.assertIn("then backed up", summarize(thought))
        # read-only: the same thought, nothing written
        fresh = trained()
        node = fresh.graph.lookup("ha ")[0] if fresh.graph.lookup("ha ") else node
        quiet = think(fresh, at=node, trigger=STUTTER, learn=False)
        self.assertEqual(quiet.then, THEN_BACK)
        self.assertEqual((quiet.taught, quiet.handed_over), (-1, -1))
        self.assertEqual(fresh.graph.parents[BACK], {})
        self.assertEqual(fresh.graph.parents[THINK], {})

    def test_validation(self):
        model = trained()
        with self.assertRaises(ValueError):
            think(model, mode="walk")
        with self.assertRaises(ValueError):
            think(model, at=START)
        with self.assertRaises(ValueError):
            think(model, k=0)
        with self.assertRaises(ValueError):
            think(model, max_depth=-1)
        with self.assertRaises(TypeError):
            think(model, about=3)
        negative = new_model("negative", seed=1)
        with self.assertRaises(ValueError):
            think_on(negative, THOUGHTS)
        with self.assertRaises(TypeError):
            think_on(model, [3])

    def test_the_model_file_keeps_its_thoughts(self):
        model = trained("count")
        think_on(model, THOUGHTS, **FAST)
        before = think(model, learn=False)
        loaded = type(model).from_dict(model.to_dict())
        loaded.graph.check_invariants(texts=TEXTS)
        after = think(loaded, learn=False)
        self.assertEqual(after.text, before.text)
        self.assertEqual(after.labels, before.labels)  # the ids are compacted by the file; the labels are the same
        self.assertAlmostEqual(after.cost, before.cost, places=9)


class TestThinkApi(unittest.TestCase):
    def setUp(self):
        self.client, self.server, self.service = start_server(self.addCleanup)
        status, _data, _ = self.client.post("/api/train", {"texts": TEXTS, **FAST})
        self.assertEqual(status, 202)
        wait_for_job(self.client)

    def test_the_route_thinks(self):
        status, data, _ = self.client.post("/api/think", {"about": "the cat"})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["kind"], "radix")
        self.assertEqual(data["trigger"], ASKED)
        self.assertGreaterEqual(data["at"], FIRST)
        self.assertEqual(data["taught"], data["at"])
        self.assertEqual(data["then"], THEN_END)
        self.assertEqual(data["stopped"], "nothing")
        self.assertEqual(data["questions"], [])
        # learning off: nothing taught
        status, data, _ = self.client.post("/api/think", {"about": "the cat", "learn": False})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["taught"], -1)
        status, data, _ = self.client.post("/api/think", {"mode": "walk"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/think", {"depth": -1})
        self.assertEqual(status, 400, data)

    def test_the_encoding_names_the_sentinel(self):
        status, info, _ = self.client.get("/api/encoding")
        self.assertEqual(status, 200)
        self.assertEqual(info["think_label"], "<think>")

    def test_a_conversation_thinks_and_says_so(self):
        status, data, _ = self.client.post("/api/converse", {"turns": 6, "guard": False})
        self.assertEqual(status, 200, data)
        rethinks = [t["rethink"] for t in data["turns"] if t["rethink"] is not None]
        self.assertTrue(rethinks, data["turns"])
        thoughts = [r["thought"] for r in rethinks if r["thought"] is not None]
        self.assertTrue(thoughts)
        self.assertEqual(thoughts[0]["then"], THEN_BACK)
        self.assertIn(thoughts[0]["trigger"], ("stutter", "repeat"))
        status, data, _ = self.client.post("/api/converse", {"turns": 4, "guard": False, "think": False})
        self.assertEqual(status, 200, data)
        self.assertTrue(all(r["thought"] is None for r in (t["rethink"] for t in data["turns"]) if r is not None))


class TestThinkCli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="radixnet-think-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.model = os.path.join(self.dir, "model.json")

    def run_cli(self, *args, expect=0):
        cmd = [sys.executable, "-m", "radixnet", "--model", self.model, "--backend", "python", "--json", *args]
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(proc.returncode, expect, proc.stderr[-2000:])
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def test_think_and_converse(self):
        self.run_cli("think", expect=1)  # no model yet
        self.run_cli("train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", "1",
                     "--lr", "0.5", "--batch-size", "4")
        doc = self.run_cli("think", "--about", "the cat")
        self.assertEqual((doc["kind"], doc["trigger"], doc["then"]), ("radix", ASKED, THEN_END))
        self.assertGreaterEqual(doc["taught"], FIRST)
        self.assertIsNone(doc["saved"])
        doc = self.run_cli("think", "--about", "the cat", "--save")
        self.assertIsNotNone(doc["saved"])
        doc = self.run_cli("think", "--about", "the cat", "--no-learn", "--depth", "0")
        self.assertEqual(doc["taught"], -1)
        doc = self.run_cli("converse", "--turns", "4", "--no-guard")
        self.assertIn("thought_at", doc)
        doc = self.run_cli("converse", "--turns", "2", "--no-guard", "--no-think", "--think-depth", "0")
        self.assertEqual(doc["thought_at"], [])


if __name__ == "__main__":
    unittest.main()
