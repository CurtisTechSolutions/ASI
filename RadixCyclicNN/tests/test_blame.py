"""Tests for radixnet.blame - where the negative network's data comes from.

The tutors are the English tutor (:mod:`radixnet.tutor`), the Ollama
reviewer, the code-generation teacher / judge and the evolve loop's
discriminator: this module turns their verdicts into faults (text, reason,
severity, note) and hands them to the negative network.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import blame  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402

REVIEWS = [
    {"index": 0, "text": "the the the the cat", "rating": 1.0, "verdict": "fail",
     "critique": "The text repeats the same word over and over."},
    {"index": 1, "text": "the cat sat on the mat", "rating": 9.0, "verdict": "pass", "critique": "no flaw found"},
    {"index": 2, "text": "xqz frm mat cat", "rating": 0.0, "verdict": "fail",
     "critique": "Pure gibberish, garbled characters."},
    {"index": 3, "text": "the dog sat on", "rating": None, "verdict": "unrated", "critique": "no review returned"},
]


def attempt(**over):
    data = {
        "text": "def solve():\n    return 1\n",
        "correct": False,
        "run": {"ok": True, "timed_out": False, "expected_ok": True, "error": None},
        "style": {"pep8_ok": True, "naming_ok": True},
        "verdict": {"task": True, "pep8": True, "naming": True, "critique": "", "issues": []},
    }
    for key, value in over.items():
        if key in ("run", "style", "verdict"):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


class TestClassify(unittest.TestCase):
    def test_the_tutors_own_words_pick_the_reason(self):
        cases = {
            "The text repeats the same word over and over.": "repetition",
            "Pure gibberish with garbled characters": "gibberish",
            "It is cut off mid-sentence": "truncated",
            "Broken grammar and wrong word order": "grammar",
            "Several misspellings": "spelling",
            "It contradicts itself": "contradiction",
            "The claim is factually wrong": "false",
            "Completely incoherent": "incoherent",
            "Irrelevant to the prompt": "off-topic",
            "empty output": "empty",
            "Something nobody has a word for": "other",
        }
        for critique, reason in cases.items():
            self.assertEqual(blame.classify(critique), reason, critique)
        self.assertIn(blame.classify("Something nobody has a word for"), blame.REASONS)

    def test_unrated_and_unspoken_failures(self):
        self.assertEqual(blame.classify("", verdict="unrated"), "unrated")
        self.assertEqual(blame.classify(None, rating=0.0), "gibberish")
        self.assertEqual(blame.classify(None), blame.DEFAULT_REASON)
        self.assertEqual(blame.classify("nothing matching", default="mine"), "mine")

    def test_severity_follows_the_rating(self):
        self.assertAlmostEqual(blame.severity_from_rating(0.0, 6.0), 2.0)
        self.assertAlmostEqual(blame.severity_from_rating(6.0, 6.0), 0.25)
        self.assertAlmostEqual(blame.severity_from_rating(3.0, 6.0), (0.25 + 2.0) / 2)
        self.assertEqual(blame.severity_from_rating(None), 1.0)
        self.assertAlmostEqual(blame.severity_from_rating(99.0, 6.0), 0.25)  # clamped
        self.assertAlmostEqual(blame.severity_from_rating(1.0, 0.0), 0.25)  # a zero threshold never divides by zero

    def test_code_reasons_come_from_the_sandbox_and_the_judge(self):
        self.assertEqual(blame.code_reason(attempt(run={"timed_out": True, "ok": False})), "timeout")
        self.assertEqual(blame.code_reason(attempt(run={"ok": False, "error": "NameError"})), "crash")
        self.assertEqual(blame.code_reason(attempt(run={"expected_ok": False})), "wrong-output")
        self.assertEqual(blame.code_reason(attempt(verdict={"task": False})), "task-not-done")
        self.assertEqual(blame.code_reason(attempt(verdict={"naming": False})), "naming")
        self.assertEqual(blame.code_reason(attempt(style={"pep8_ok": False})), "style")
        self.assertEqual(blame.code_reason(attempt(verdict={"critique": "it repeats the same block"})), "repetition")
        self.assertIn(blame.code_reason(attempt()), blame.CODE_REASONS)


class TestFaults(unittest.TestCase):
    def test_reviews_become_faults_and_passes(self):
        faults, passed = blame.faults_from_reviews(REVIEWS, threshold=6.0)
        self.assertEqual([fault["reason"] for fault in faults], ["repetition", "gibberish", "unrated"])
        self.assertEqual(passed, ["the cat sat on the mat"])
        self.assertGreater(faults[1]["severity"], faults[0]["severity"])  # rated 0 is worse than rated 1
        self.assertEqual(faults[0]["source"], "review")
        self.assertIn("repeats", faults[0]["note"])
        self.assertEqual(blame.faults_from_reviews([{"text": ""}])[0], [])
        self.assertEqual(blame.faults_from_reviews(["not a dict"])[0], [])

    def test_attempts_become_faults(self):
        faults, correct = blame.faults_from_attempts([
            attempt(run={"ok": False, "error": "NameError: x"}, verdict={"issues": ["NameError"]}),
            attempt(correct=True, text="def solve():\n    return 2\n"),
        ])
        self.assertEqual([fault["reason"] for fault in faults], ["crash"])
        self.assertEqual(faults[0]["severity"], blame.CODE_SEVERITY["crash"])
        self.assertEqual(faults[0]["source"], "codegen")
        self.assertEqual(len(correct), 1)

    def test_teach_blames_and_clears(self):
        model = NegativeNet(seed=1)
        report = blame.teach_reviews(model, REVIEWS)
        self.assertEqual(report["blamed"], 3)
        self.assertEqual(report["passed"], 1)
        self.assertGreater(report["edges"], 0)
        self.assertEqual(set(report["reasons"]), {"repetition", "gibberish", "unrated"})
        self.assertEqual(model.judge(REVIEWS[0]["text"])["reasons"][0]["reason"], "repetition")
        self.assertEqual(model.stats()["sources"], {"review": 3})

    def test_teach_accepts_the_whole_review_result(self):
        model = NegativeNet(seed=2)
        report = blame.teach_reviews(model, {"reviews": REVIEWS, "threshold": 8.0})
        self.assertEqual(report["threshold"], 8.0)
        self.assertEqual(report["blamed"], 3)

    def test_clear_passes_can_be_turned_off(self):
        model = NegativeNet(seed=3)
        report = blame.teach_reviews(model, REVIEWS, clear_passes=False)
        self.assertEqual(report["cleared"], 0)
        self.assertEqual(model.graph.total_clear, 0.0)

    def test_teach_attempts(self):
        model = NegativeNet(seed=4)
        report = blame.teach_attempts(model, [attempt(run={"timed_out": True, "ok": False})])
        self.assertEqual(report["reasons"], {"timeout": 1})
        self.assertEqual(model.stats()["sources"], {"codegen": 1})

    def test_teach_stops_on_the_stop_event(self):
        import threading

        model = NegativeNet(seed=5)
        stop = threading.Event()
        stop.set()
        report = blame.teach(model, [{"text": "a failing text", "reason": "other", "severity": 1.0}], stop_event=stop)
        self.assertEqual(report["blamed"], 0)


LESSONS = [
    {
        "exercise": {"id": "r1e1", "prefix": "the cat", "focus": "subject-verb agreement",
                     "answer": "the cat sleeps on the mat"},
        "sentence": "the cat sit on the mat",
        "grade": {"score": 4.0, "passed": False, "error": "agreement", "graded_by": "ollama",
                  "correction": "the cat sits on the mat", "comment": "The verb must agree with the subject."},
    },
    {
        "exercise": {"id": "r1e2", "prefix": "the dogs", "focus": "plural nouns", "answer": ""},
        "sentence": "the dogs run in the park",
        "grade": {"score": 9.0, "passed": True, "error": "none", "graded_by": "ollama", "correction": "",
                  "comment": "Good."},
    },
    {
        "exercise": {"id": "r1e3", "prefix": "she go", "focus": "past tense", "answer": "she went to the shop"},
        "sentence": "she go to the shop yesterday",
        "grade": {"score": 2.0, "passed": False, "error": "tense", "graded_by": "ollama", "correction": "",
                  "comment": "The past tense of 'go' is 'went'."},
    },
]


class TestTutorLessons(unittest.TestCase):
    """The English tutor is the richest tutor: it names the mistake and writes the sentence out correctly."""

    def test_lessons_become_faults(self):
        faults, passed = blame.faults_from_lessons(LESSONS, threshold=6.0)
        self.assertEqual([f["reason"] for f in faults], ["agreement", "tense"])
        self.assertEqual(faults[0]["correction"], "the cat sits on the mat")
        self.assertNotIn("correction", faults[1])  # the teacher named the mistake without rewriting the sentence
        self.assertGreater(faults[1]["severity"], faults[0]["severity"])  # 2/10 is worse than 4/10
        self.assertEqual(faults[0]["source"], "tutor")
        self.assertIn("agree", faults[0]["note"])
        # what passed, every correction and every model answer clear blame
        self.assertIn("the dogs run in the park", passed)
        self.assertIn("the cat sits on the mat", passed)
        self.assertIn("she went to the shop", passed)

    def test_a_correction_blames_only_what_the_teacher_changed(self):
        model = NegativeNet(seed=7)
        report = blame.teach_lessons(model, LESSONS)
        self.assertEqual(report["blamed"], 2)
        self.assertEqual(set(report["reasons"]), {"agreement", "tense"})
        self.assertEqual(model.stats()["sources"], {"tutor": 2})
        verdict = model.judge("the cat sit on the mat")
        self.assertEqual(verdict["reasons"][0]["reason"], "agreement")
        self.assertEqual([span["fragment"] for span in verdict["spans"]], ["sit "])
        self.assertEqual(model.judge("the cat sits on the mat")["verdict"], "pass")

    def test_an_unrecognised_mistake_falls_back_to_the_critique(self):
        faults, _ = blame.faults_from_lessons([{
            "sentence": "the the the cat", "exercise": {},
            "grade": {"score": 1.0, "passed": False, "error": "none", "correction": "",
                      "comment": "It repeats the same word over and over."},
        }])
        self.assertEqual(faults[0]["reason"], "repetition")

    def test_the_tutor_trainer_teaches_the_negative_network(self):
        from radixnet.ollama import OllamaClient
        from radixnet.tutor import Exercise, Grade, Lesson, TutorConfig, TutorTrainer

        model = NegativeNet(seed=8)
        trainer = TutorTrainer(None, OllamaClient("http://127.0.0.1:1"), TutorConfig(), negative=model)
        lesson = Lesson(
            exercise=Exercise(id="r1e1", prefix="the cat", answer="the cat sleeps on the mat"),
            attempt=0, mode="dijkstra", continuation="sit on the mat", sentence="the cat sit on the mat",
            grade=Grade(score=4.0, passed=False, error="agreement", correction="the cat sits on the mat",
                        comment="The verb must agree with the subject."),
        )
        report = trainer._teach_negative([lesson])
        self.assertEqual(report["reasons"], {"agreement": 1})
        self.assertEqual(model.judge("the cat sit on the mat")["reasons"][0]["reason"], "agreement")
        self.assertIsNone(TutorTrainer(None, OllamaClient("http://127.0.0.1:1"))._teach_negative([lesson]))


class TestEvolveAndCodegenHooks(unittest.TestCase):
    def test_the_discriminator_teaches_the_negative_network(self):
        from radixnet import RadixNet
        from radixnet.gan import EvolveConfig, Evolver

        corpus = ["the cat sat on the mat", "the dog sat on the log", "the cat ran to the door"]
        generator = RadixNet(seed=1)
        generator.train(corpus, epochs=2, lr=0.5, batch_size=4)
        negative = NegativeNet(seed=2)
        evolver = Evolver(generator, corpus, config=EvolveConfig(samples=4, max_length=30, seed=3), negative=negative)
        record = evolver.run_generation()
        self.assertIn("negative_blamed", record)
        self.assertEqual(record["negative_blamed"], negative.meta["trained_texts"])
        if record["negative_blamed"]:
            self.assertTrue(set(record["negative_reasons"]) <= {"discriminator", "blatant"})
            self.assertEqual(negative.stats()["sources"], {"evolve": record["negative_blamed"]})

    def test_evolve_without_a_negative_network_is_unchanged(self):
        from radixnet import RadixNet
        from radixnet.gan import EvolveConfig, Evolver

        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        generator = RadixNet(seed=1)
        generator.train(corpus, epochs=1, lr=0.5, batch_size=4)
        record = Evolver(generator, corpus, config=EvolveConfig(samples=2, max_length=20, seed=1)).run_generation()
        self.assertNotIn("negative_blamed", record)

    def test_codegen_trainer_teaches_the_negative_network(self):
        from radixnet.codegen import Attempt, CodeGenTrainer, RunResult, StyleReport, Verdict
        from radixnet.ollama import OllamaClient

        negative = NegativeNet(seed=6)
        trainer = CodeGenTrainer(
            RadixNetStub(), OllamaClient("http://127.0.0.1:1"), config=None, negative=negative,
        )
        run = RunResult(ok=False, exit_code=1, stdout="", stderr="NameError", error="NameError: x", timed_out=False,
                        seconds=0.1)
        style = StyleReport(ok=True, syntax_ok=True, pep8_ok=True, naming_ok=True, issues=[])
        verdict = Verdict(correct=False, runs=False, task=False, pep8=True, naming=True, score=None,
                          issues=["NameError"], critique="", judged_by="sandbox")
        bad = Attempt(index=0, source="model", code="x", text="def solve():\n    return x\n", run=run, style=style,
                      verdict=verdict, seconds=0.1)
        report = trainer._teach_negative([bad])
        self.assertEqual(report["reasons"], {"crash": 1})
        self.assertGreater(negative.graph.total_blame, 0)
        self.assertIsNone(CodeGenTrainer(RadixNetStub(), OllamaClient("http://127.0.0.1:1"))._teach_negative([bad]))


class RadixNetStub:
    """Enough of a model for CodeGenTrainer's constructor (nothing here trains it)."""

    kind = "radix"


if __name__ == "__main__":
    unittest.main()
