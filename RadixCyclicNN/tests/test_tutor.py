"""Tests for ``radixnet.tutor`` (exercises, marking, the lesson loop), its API endpoints and CLI.

A fake Ollama server (standard library) stands in for the real one and plays
the English teacher: it answers exercise requests with JSON openings and model
answers, marks completions by a simple rule (a sentence ending in "the mat" is
correct English, anything else is an agreement mistake), and returns plain
lines for drill sentences.  Every request is recorded, and the answers can be
replaced or made to fail.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import CountRewardNet, RadixNet  # noqa: E402
from radixnet.llm import LLMError  # noqa: E402
from radixnet.ollama import OllamaClient, OllamaError  # noqa: E402
from radixnet.search import PathResult  # noqa: E402
from radixnet.tutor import (  # noqa: E402
    ERROR_TYPES,
    LEVELS,
    Exercise,
    Grade,
    Lesson,
    TutorConfig,
    TutorTrainer,
    _error_type,
    _parse_grades,
    card_lines,
    cue,
    drill_sentences,
    focus_for,
    grade_completions,
    next_level,
    overall_score,
    parse_exercises,
    parse_plan,
    plan_from_card,
    plan_lessons,
    report_card,
    weak_points,
    write_exercises,
)

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server, wait_for_job  # noqa: F401
    from test_ollama import closed_port_url  # noqa: F401
except ImportError:
    from tests.test_api import start_server, wait_for_job  # noqa: F401
    from tests.test_ollama import closed_port_url  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = ["the cat sat on the mat", "the dogs run in the park", "the cat likes the mat"]
FAST = {"neg_epochs": 1, "pos_epochs": 1, "batch_size": 4}


# ---------------------------------------------------------------------------
# the fake teacher
# ---------------------------------------------------------------------------


def _mark(prefix, continuation):
    """The fake teacher's rule: only the corpus sentences are correct English, everything else fails on agreement."""
    sentence = " ".join((prefix + continuation).split())
    if sentence in CORPUS:
        return {"grammar": 9, "spelling": 8, "fluency": 7, "error": "none",
                "correction": sentence, "comment": "Well written."}
    correction = next((t for t in CORPUS if t.startswith(prefix.strip())), prefix.strip() + " the mat")
    return {"grammar": 2, "spelling": 3, "fluency": 1, "error": "subject-verb agreement",
            "correction": correction, "comment": "A plural subject takes a plural verb."}


class _FakeHandler(BaseHTTPRequestHandler):
    server: "FakeTeacher"

    def log_message(self, *args):  # silence
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.requests.append(("GET", self.path, None))
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
        elif self.path == "/api/tags":
            self._json(200, {"models": [{"name": "fake:latest", "size": 1, "modified_at": "2026-01-01T00:00:00Z"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(("POST", self.path, body))
        if self.server.fail_with:
            self._json(self.server.fail_with, {"error": "boom"})
            return
        self._json(200, {"model": body.get("model"), "response": self._answer(body), "done": True})

    def _answer(self, body):
        prompt, system = body.get("prompt", ""), body.get("system", "")
        if "writing exercises" in system:
            if self.server.exercise_response is not None:
                return self.server.exercise_response
            match = re.search(r"exactly (\d+) entries", system)
            count = int(match.group(1)) if match else 2
            pool = [
                {"prefix": "the cat sat on", "focus": "prepositions of place", "answer": "the cat sat on the mat"},
                {"prefix": "the dogs run", "focus": "subject-verb agreement", "answer": "the dogs run in the park"},
                {"prefix": "the cat likes", "focus": "verb agreement", "answer": "the cat likes the mat"},
            ]
            return json.dumps({"exercises": [pool[i % len(pool)] for i in range(count)]})
        if "marking sentence completions" in system:
            if self.server.grade_response is not None:
                return self.server.grade_response
            grades = []
            for line in prompt.splitlines():
                match = re.match(r"\[(\d+)\] <<(.*?)>>(.*)", line)
                if match:
                    index, prefix, rest = int(match.group(1)), match.group(2), match.group(3)
                    continuation = rest.split("   (drilling:")[0]
                    grades.append({"index": index, **_mark(prefix, continuation)})
            return json.dumps({"grades": grades})
        if "planning the next lessons" in system:
            if self.server.plan_response is not None:
                return self.server.plan_response
            match = re.search(r"exactly (\d+) lessons", system)
            count = int(match.group(1)) if match else 2
            pool = [
                {"focus": "subject-verb agreement", "targets": "agreement", "topic": "animals",
                 "why": "Nearly every sentence lost marks here."},
                {"focus": "plural nouns", "targets": "plural", "topic": "the market", "why": "Plurals were shaky."},
                {"focus": "past tense", "targets": "tense", "topic": "yesterday", "why": "Tenses drifted."},
            ]
            return json.dumps({
                "summary": "The student writes verbs badly.", "level": "beginner",
                "lessons": [pool[i % len(pool)] for i in range(count)],
            })
        if "model sentences" in system:  # drills
            match = re.search(r"exactly (\d+) lines", system)
            count = int(match.group(1)) if match else 3
            return "\n".join(f"{i + 1}. the cat sat on the mat number {i}" for i in range(count))
        return "unexpected request"


class FakeTeacher(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _FakeHandler)
        self.requests = []
        self.fail_with = None
        self.exercise_response = None
        self.grade_response = None
        self.plan_response = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def prompts(self, kind):
        """The user prompts of the recorded calls of one kind: exercises, grades or drills."""
        marker = {
            "exercises": "writing exercises", "grades": "marking sentence completions",
            "drills": "model sentences", "plan": "planning the next lessons",
        }[kind]
        return [body["prompt"] for _method, _path, body in self.requests if body and marker in body.get("system", "")]


def start_fake(add_cleanup):
    server = FakeTeacher()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    add_cleanup(server.server_close)
    add_cleanup(server.shutdown)
    return server


class ScriptedModel:
    """A stand-in for the network: fixed completions, and every learning call recorded instead of run."""

    kind = "scripted"

    def __init__(self, *continuations):
        self.continuations = list(continuations) or [""]
        self.asked = []
        self.calls = []

    def predict(self, prefix, **options):
        text = self.continuations[len(self.asked) % len(self.continuations)]
        self.asked.append(prefix)
        return PathResult(text=text, cost=1.0, reached_end=True, full_text=prefix + text)

    def two_nrl(self, bad, good, **options):
        self.calls.append(("two_nrl", list(bad), list(good), options))
        return {"negative": [{"loss": 1.25}], "positive": [{"loss": 0.5}]}

    def reward(self, texts, **options):
        self.calls.append(("reward", list(texts), options))
        return [{"loss": 0.5}]

    def punish(self, texts, **options):
        self.calls.append(("punish", list(texts), options))
        return [{"loss": 1.5}]


def trained_model(cls=RadixNet, seed=7):
    model = cls(seed=seed)
    model.train(CORPUS, epochs=6, lr=0.6, batch_size=4)
    return model


# ---------------------------------------------------------------------------
# marks and mistakes
# ---------------------------------------------------------------------------


class HelperTests(unittest.TestCase):
    def test_cue_adds_one_trailing_space(self):
        self.assertEqual(cue("the cat"), "the cat ")
        self.assertEqual(cue("the cat  "), "the cat ")
        self.assertEqual(cue(""), "")
        self.assertEqual(cue(None), "")

    def test_overall_score_is_grammar_weighted(self):
        self.assertAlmostEqual(overall_score(10, 0, 0, 0.6), 6.0)
        self.assertAlmostEqual(overall_score(0, 10, 10, 0.6), 4.0)
        self.assertAlmostEqual(overall_score(4, 8, 6, 0.5), 5.5)
        # missing sub-marks drop out of the mean
        self.assertEqual(overall_score(7, None, None), 7)
        self.assertEqual(overall_score(None, 6, 8), 7)
        self.assertIsNone(overall_score(None, None, None))

    def test_error_types_are_mapped_onto_the_list(self):
        self.assertEqual(_error_type("subject-verb agreement"), "agreement")
        self.assertEqual(_error_type("Verb Tense"), "tense")
        self.assertEqual(_error_type("word_order"), "word-order")
        self.assertEqual(_error_type(""), "none")
        self.assertEqual(_error_type("no error"), "none")
        self.assertEqual(_error_type("style"), "other")
        for name in ERROR_TYPES:
            self.assertEqual(_error_type(name), name)

    def test_report_card_counts_marks_and_mistakes(self):
        lessons = [
            Lesson(Exercise("e1", "a"), 0, "dijkstra", "x", "a x", grade=Grade(8.0, 8, 8, 8, True, "none")),
            Lesson(Exercise("e2", "b"), 0, "dijkstra", "y", "b y", grade=Grade(2.0, 1, 3, 2, False, "agreement")),
            Lesson(Exercise("e3", "c"), 0, "dijkstra", "z", "c z", grade=Grade(None, None, None, None, False, "agreement")),
        ]
        card = report_card(lessons)
        self.assertEqual((card["lessons"], card["graded"], card["passed"], card["failed"]), (3, 2, 1, 2))
        self.assertAlmostEqual(card["pass_rate"], 1 / 3)
        self.assertAlmostEqual(card["mean_score"], 5.0)
        self.assertAlmostEqual(card["mean_grammar"], 4.5)
        self.assertEqual(card["errors"], {"agreement": 2})
        self.assertEqual(card["weakest"], ["agreement"])
        self.assertEqual(report_card([])["lessons"], 0)


class ParsingTests(unittest.TestCase):
    def test_parse_exercises_reads_the_documented_shape(self):
        raw = json.dumps({"exercises": [
            {"prefix": "the cat sat on", "focus": "prepositions", "answer": "the cat sat on the mat"},
            {"prefix": "the dogs run", "focus": "agreement", "answer": "in the park"},
        ]})
        first, second = parse_exercises(raw, 5, round_no=2)
        self.assertEqual((first.id, first.prefix, first.focus), ("r2e1", "the cat sat on", "prepositions"))
        self.assertEqual(first.answer, "the cat sat on the mat")
        self.assertEqual(first.cue, "the cat sat on ")
        # an answer that is only the continuation is joined onto the prefix
        self.assertEqual(second.answer, "the dogs run in the park")

    def test_parse_exercises_tolerates_llm_drift(self):
        raw = json.dumps([
            "  the children were playing.  ",  # a bare string, trailing punctuation, stray spaces
            {"opening": "The children were playing"},  # the same opening under another key
            {"stem": "every morning she", "point": "present simple", "solution": "every morning she walks"},
            {"nothing": "usable"},
            42,
        ])
        exercises = parse_exercises(raw, 5)
        self.assertEqual([e.prefix for e in exercises], ["the children were playing", "every morning she"])
        self.assertEqual(exercises[1].focus, "present simple")
        self.assertEqual(exercises[1].answer, "every morning she walks")

    def test_parse_exercises_falls_back_to_plain_lines(self):
        exercises = parse_exercises("1. the cat sat on\n2. the dogs run\n", 5)
        self.assertEqual([e.prefix for e in exercises], ["the cat sat on", "the dogs run"])
        self.assertEqual(parse_exercises("", 3), [])

    def test_parse_exercises_honours_the_limit(self):
        raw = json.dumps({"exercises": [{"prefix": f"opening {i}"} for i in range(10)]})
        self.assertEqual(len(parse_exercises(raw, 3)), 3)

    def test_parse_grades_reads_marks_and_alternatives(self):
        raw = json.dumps({"grades": [
            {"index": 0, "grammar": 9, "spelling": 8, "fluency": 7, "error": "none", "correction": "a", "comment": "good"},
            {"index": 1, "score": 3, "mistake": "plural", "corrected": "b", "critique": "bad"},
            {"index": 9, "grammar": 5},  # out of range
            "nonsense",
        ]}, )
        grades = _parse_grades(raw, 2, 0.6, 6.0)
        self.assertEqual(sorted(grades), [0, 1])
        self.assertTrue(grades[0].passed)
        self.assertAlmostEqual(grades[0].score, 0.6 * 9 + 0.4 * 7.5)
        self.assertEqual((grades[0].error, grades[0].correction, grades[0].comment), ("none", "a", "good"))
        self.assertFalse(grades[1].passed)
        self.assertEqual((grades[1].score, grades[1].error, grades[1].correction, grades[1].comment), (3.0, "plural", "b", "bad"))

    def test_parse_grades_survives_junk(self):
        self.assertEqual(_parse_grades("not json at all", 2, 0.6, 6.0), {})
        self.assertEqual(_parse_grades(json.dumps({"grades": [{"index": 0}]}), 1, 0.6, 6.0), {})
        # a single object without a wrapper still counts
        grades = _parse_grades(json.dumps({"grammar": 8, "spelling": 8, "fluency": 8}), 1, 0.6, 6.0)
        self.assertEqual(list(grades), [0])
        self.assertTrue(grades[0].passed)


# ---------------------------------------------------------------------------
# talking to the teacher
# ---------------------------------------------------------------------------


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "fake:latest")

    def test_write_exercises(self):
        exercises = write_exercises(self.client, "animals", 2, focus="plurals", weak=["agreement", "none"])
        self.assertEqual([e.id for e in exercises], ["r1e1", "r1e2"])
        self.assertEqual(exercises[0].prefix, "the cat sat on")
        self.assertTrue(all(e.answer for e in exercises))
        prompt = self.fake.prompts("exercises")[0]
        self.assertIn("Topic: animals", prompt)
        self.assertIn("must drill: plurals", prompt)
        self.assertIn("agreement", prompt)
        self.assertNotIn("none", prompt.split("mistakes, so drill them:")[1].split("\n")[0])

    def test_write_exercises_rejects_bad_input_and_empty_answers(self):
        with self.assertRaises(ValueError):
            write_exercises(self.client, "animals", 0)
        with self.assertRaises(ValueError):
            write_exercises(self.client, "   ", 2)
        self.fake.exercise_response = json.dumps({"exercises": []})
        with self.assertRaises(LLMError):  # an unusable answer is not the provider's transport failing
            write_exercises(self.client, "animals", 2)
        self.fake.fail_with = 500
        with self.assertRaises(OllamaError):
            write_exercises(self.client, "animals", 2)

    def test_drill_sentences(self):
        lines = drill_sentences(self.client, "animals", 3, weak=["plural"])
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(line.startswith("the cat sat on the mat") for line in lines))
        self.assertIn("plural", self.fake.prompts("drills")[0])
        self.assertEqual(drill_sentences(self.client, "animals", 0), [])

    def _lessons(self, *pairs):
        return [
            Lesson(Exercise(f"e{i}", prefix, "focus", f"{prefix} the mat"), 0, "dijkstra", continuation,
                   cue(prefix) + continuation)
            for i, (prefix, continuation) in enumerate(pairs)
        ]

    def test_grade_completions_marks_every_lesson(self):
        lessons = self._lessons(("the cat sat on", "the mat"), ("the dogs run", "the mat"))
        grade_completions(self.client, lessons, topic="animals", threshold=6.0)
        self.assertTrue(lessons[0].grade.passed)
        self.assertEqual(lessons[0].grade.error, "none")
        self.assertFalse(lessons[1].grade.passed)
        self.assertEqual(lessons[1].grade.error, "agreement")
        self.assertEqual(lessons[1].grade.correction, "the dogs run in the park")
        self.assertIn("A plural subject", lessons[1].grade.comment)
        self.assertEqual([l.grade.graded_by for l in lessons], ["ollama", "ollama"])
        prompt = self.fake.prompts("grades")[0]
        self.assertIn("Topic of the lesson: animals", prompt)
        self.assertIn("[0] <<the cat sat on >>the mat", prompt)
        self.assertIn("(drilling: focus)", prompt)

    def test_grade_completions_batches(self):
        lessons = self._lessons(*[("the cat sat on", "the mat")] * 5)
        grade_completions(self.client, lessons, batch=2)
        self.assertEqual(len(self.fake.prompts("grades")), 3)
        self.assertTrue(all(l.grade.passed for l in lessons))
        with self.assertRaises(ValueError):
            grade_completions(self.client, lessons, batch=0)

    def test_empty_completion_fails_without_asking(self):
        lessons = self._lessons(("the cat sat on", ""))
        grade_completions(self.client, lessons)
        grade = lessons[0].grade
        self.assertEqual((grade.score, grade.passed, grade.error, grade.graded_by), (0.0, False, "nonsense", "empty"))
        self.assertEqual(grade.correction, "the cat sat on the mat")  # the teacher's own answer teaches instead
        self.assertEqual(self.fake.prompts("grades"), [])  # nothing to mark: no LLM call

    def test_unreadable_marking_leaves_the_lesson_unrated(self):
        self.fake.grade_response = "the teacher wandered off"
        lessons = self._lessons(("the cat sat on", "the sofa"))
        grade_completions(self.client, lessons)
        grade = lessons[0].grade
        self.assertEqual((grade.score, grade.passed, grade.graded_by), (None, False, "unrated"))
        self.assertEqual(grade.correction, "the cat sat on the mat")


# ---------------------------------------------------------------------------
# the next lesson plan
# ---------------------------------------------------------------------------


CARD = {
    "lessons": 10, "graded": 10, "passed": 2, "failed": 8, "pass_rate": 0.2, "mean_score": 3.4,
    "mean_grammar": 3.0, "mean_spelling": 6.0, "mean_fluency": 4.0,
    "errors": {"agreement": 5, "tense": 3, "article": 1}, "weakest": ["agreement", "tense", "article"],
}


class WeakPointTests(unittest.TestCase):
    def test_weak_points_rank_the_mistakes_worst_first(self):
        points = weak_points(CARD)
        self.assertEqual([p["error"] for p in points], ["agreement", "tense", "article"])
        self.assertEqual([p["count"] for p in points], [5, 3, 1])
        self.assertAlmostEqual(points[0]["share"], 0.5)
        self.assertEqual(points[0]["focus"], "subject-verb agreement")
        self.assertEqual([p["error"] for p in weak_points(CARD, limit=2)], ["agreement", "tense"])
        self.assertEqual(weak_points({"lessons": 3, "errors": {}}), [])
        self.assertEqual(weak_points({}), [])

    def test_weak_points_read_a_card_that_went_through_json(self):
        card = {"lessons": 4.0, "errors": {"subject-verb agreement": 2.0, "none": 9, "tense": "1"}}
        points = weak_points(card)
        self.assertEqual([(p["error"], p["count"]) for p in points], [("agreement", 2), ("tense", 1)])
        self.assertAlmostEqual(points[0]["share"], 0.5)
        self.assertIsNone(weak_points({"errors": {"tense": 1}})[0]["share"])  # no lessons: no share

    def test_ties_break_alphabetically(self):
        card = {"lessons": 4, "errors": {"tense": 2, "article": 2, "plural": 2}}
        self.assertEqual([p["error"] for p in weak_points(card)], ["article", "plural", "tense"])

    def test_focus_for_names_the_grammar_point(self):
        self.assertEqual(focus_for("subject-verb agreement"), "subject-verb agreement")
        self.assertEqual(focus_for("plural"), "plural nouns")
        self.assertEqual(focus_for("none"), "")
        self.assertEqual(focus_for(""), "")
        for name in ERROR_TYPES[1:]:
            self.assertTrue(focus_for(name), name)

    def test_next_level_only_moves_a_strong_card_up(self):
        strong = {"pass_rate": 0.9, "mean_score": 8.5}
        self.assertEqual(next_level("beginner", strong), "intermediate")
        self.assertEqual(next_level("intermediate", strong), "advanced")
        self.assertEqual(next_level("advanced", strong), "advanced")  # the ladder ends
        self.assertEqual(next_level("beginner", CARD), "beginner")
        self.assertEqual(next_level("beginner", {"pass_rate": 0.9}), "beginner")  # no mean: no promotion
        self.assertEqual(next_level("fluent", strong), "fluent")  # off the ladder: left alone

    def test_card_lines_read_like_a_report_card(self):
        lines = card_lines(CARD)
        self.assertEqual(lines[0], "Lessons marked: 10")
        self.assertEqual(lines[1], "Passed: 2 (20%)")
        self.assertEqual(lines[2], "Mean marks out of 10: overall 3.4, grammar 3.0, spelling 6.0, fluency 4.0")
        self.assertEqual(lines[3], "Mistakes, worst first: agreement x5 (50% of the lessons), "
                                   "tense x3 (30% of the lessons), article x1 (10% of the lessons)")
        self.assertEqual(card_lines({"lessons": 1})[-1], "Mistakes: none were named.")
        self.assertIn("none recorded", card_lines({"lessons": 1})[2])


class PlanFromCardTests(unittest.TestCase):
    def test_every_weak_point_gets_a_lesson(self):
        plan = plan_from_card(CARD, topic="animals", level="beginner", count=3, exercises=4, drills=2)
        self.assertEqual(plan.source, "report card")
        self.assertEqual([l.targets for l in plan.lessons], ["agreement", "tense", "article"])
        self.assertEqual(plan.targets, ["agreement", "tense", "article"])
        self.assertEqual([l.focus for l in plan.lessons][0], "subject-verb agreement")
        self.assertTrue(all(l.topic == "animals" for l in plan.lessons))
        self.assertTrue(all((l.exercises, l.drills) == (4, 2) for l in plan.lessons))
        self.assertEqual(plan.lessons[0].why, "5 of 10 lessons (50%) were marked down for agreement.")
        self.assertEqual(plan.lessons[2].why, "1 of 10 lessons (10%) was marked down for article.")
        self.assertIn("2 of 10 lessons passed", plan.summary)
        self.assertEqual(plan.level, "beginner")

    def test_count_caps_the_lessons(self):
        self.assertEqual(len(plan_from_card(CARD, count=2).lessons), 2)

    def test_a_clean_card_moves_the_student_up_instead(self):
        card = {"lessons": 4, "passed": 4, "pass_rate": 1.0, "mean_score": 9.0, "errors": {}}
        plan = plan_from_card(card, topic="the sea", level="beginner")
        self.assertEqual(plan.level, "intermediate")
        self.assertEqual([l.targets for l in plan.lessons], ["none"])
        self.assertEqual(plan.lessons[0].topic, "the sea")
        self.assertIn("nothing was marked down", plan.summary)

    def test_a_plan_is_json(self):
        doc = plan_from_card(CARD, topic="animals").to_dict()
        self.assertEqual(sorted(doc), ["lessons", "level", "source", "summary", "targets", "topic", "weak"])
        self.assertEqual(sorted(doc["lessons"][0]),
                         ["drills", "exercises", "focus", "prefixes", "targets", "topic", "why"])
        self.assertEqual(json.loads(json.dumps(doc))["weak"][0]["error"], "agreement")


class ParsePlanTests(unittest.TestCase):
    def test_reads_the_documented_shape(self):
        raw = json.dumps({"summary": "Verbs are the trouble.", "level": "Intermediate", "lessons": [
            {"focus": "subject-verb agreement", "targets": "agreement", "topic": "the market", "why": "Most marks."},
            {"focus": "past tense", "targets": "verb tense", "topic": "yesterday", "prefixes": ["last week we"]},
        ]})
        plan = parse_plan(raw, 3, topic="animals", exercises=4, drills=1)
        self.assertEqual(plan.summary, "Verbs are the trouble.")
        self.assertEqual(plan.level, "intermediate")
        self.assertEqual([(l.focus, l.targets, l.topic) for l in plan.lessons], [
            ("subject-verb agreement", "agreement", "the market"),
            ("past tense", "tense", "yesterday"),
        ])
        self.assertEqual([(l.exercises, l.drills) for l in plan.lessons], [(4, 1), (4, 1)])
        self.assertEqual(plan.lessons[1].prefixes, ["last week we"])
        self.assertEqual(plan.source, "llm")

    def test_tolerates_the_shapes_an_llm_drifts_into(self):
        for raw in (
            json.dumps({"plan": [{"point": "plural nouns", "reason": "shaky"}]}),
            json.dumps([{"skill": "plural nouns", "because": "shaky"}]),
            json.dumps({"focus": "plural nouns", "note": "shaky"}),
            "```json\n{\"lessons\": [{\"title\": \"plural nouns\", \"rationale\": \"shaky\"}]}\n```",
        ):
            with self.subTest(raw=raw[:40]):
                plan = parse_plan(raw, 3, topic="animals")
                self.assertEqual([(l.focus, l.targets, l.why) for l in plan.lessons],
                                 [("plural nouns", "plural", "shaky")])
                self.assertEqual(plan.lessons[0].topic, "animals")

    def test_plain_lines_are_still_a_syllabus(self):
        plan = parse_plan("1. plural nouns\n2. prepositions of place", 5, topic="animals")
        self.assertEqual([(l.focus, l.targets) for l in plan.lessons],
                         [("plural nouns", "plural"), ("prepositions of place", "preposition")])

    def test_drops_duplicates_empties_and_the_overflow(self):
        raw = json.dumps({"lessons": [
            {"focus": "plural nouns"}, {"focus": "Plural Nouns"}, {"why": "no focus at all"},
            {"targets": "tense"}, {"focus": "articles"},
        ]})
        plan = parse_plan(raw, 3, topic="animals")
        self.assertEqual([(l.focus, l.targets) for l in plan.lessons],
                         [("plural nouns", "plural"), ("verb tenses", "tense"), ("articles", "article")])

    def test_an_unreadable_answer_plans_nothing(self):
        self.assertEqual(parse_plan("", 3).lessons, [])
        self.assertEqual(parse_plan(json.dumps({"lessons": []}), 3).lessons, [])


class PlanLessonsTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "fake:latest")

    def test_the_teacher_sees_the_card_and_plans_from_it(self):
        plan = plan_lessons(self.client, CARD, topic="animals", level="beginner", count=2, exercises=4, drills=1)
        self.assertEqual(plan.source, "ollama")
        self.assertEqual(plan.summary, "The student writes verbs badly.")
        self.assertEqual(plan.level, "beginner")
        self.assertEqual([(l.focus, l.targets) for l in plan.lessons][0], ("subject-verb agreement", "agreement"))
        self.assertEqual([(l.exercises, l.drills) for l in plan.lessons], [(4, 1), (4, 1)])
        self.assertEqual([p["error"] for p in plan.weak], ["agreement", "tense"])
        prompt = self.fake.prompts("plan")[0]
        self.assertIn("The lessons so far were about: animals", prompt)
        self.assertIn("Mistakes, worst first: agreement x5 (50% of the lessons)", prompt)
        self.assertIn("Level so far: beginner", prompt)
        self.assertIn("Plan the next 2 lesson(s) now.", prompt)

    def test_a_weakness_the_teacher_skipped_takes_an_off_card_slot(self):
        # the teacher plans plurals (not on the card) and agreement; tense must still get a lesson
        self.fake.plan_response = json.dumps({"summary": "Verbs.", "lessons": [
            {"focus": "plural nouns", "targets": "plural"},
            {"focus": "subject-verb agreement", "targets": "agreement"},
        ]})
        plan = plan_lessons(self.client, CARD, topic="animals", count=2)
        self.assertEqual([l.targets for l in plan.lessons], ["tense", "agreement"])
        self.assertEqual(len(plan.lessons), 2)  # the plan stays the size that was asked for

    def test_a_weakness_is_appended_when_no_slot_can_be_taken(self):
        self.fake.plan_response = json.dumps({"summary": "Verbs.", "lessons": [
            {"focus": "subject-verb agreement", "targets": "agreement"},
        ]})
        plan = plan_lessons(self.client, CARD, topic="animals", count=3)
        self.assertEqual([l.targets for l in plan.lessons], ["agreement", "tense", "article"])

    def test_the_worst_mistake_wins_the_only_slot(self):
        self.fake.plan_response = json.dumps({"summary": "Verbs.", "lessons": [{"focus": "verb tenses",
                                                                                "targets": "tense"}]})
        plan = plan_lessons(self.client, CARD, topic="animals", count=1)
        self.assertEqual([l.targets for l in plan.lessons], ["agreement"])

    def test_an_unusable_answer_leaves_the_card_to_plan(self):
        for answer in ("the teacher wandered off", json.dumps({"lessons": []}), ""):
            with self.subTest(answer=answer[:20]):
                self.fake.plan_response = answer
                plan = plan_lessons(self.client, CARD, topic="animals", count=2)
                self.assertEqual(plan.source, "report card")
                self.assertEqual([l.targets for l in plan.lessons], ["agreement", "tense"])

    def test_missing_fields_fall_back_to_the_card(self):
        self.fake.plan_response = json.dumps({"lessons": [{"focus": "verb tenses", "targets": "tense"}],
                                              "level": "wherever"})
        plan = plan_lessons(self.client, CARD, topic="animals", level="beginner", count=3)
        self.assertEqual(plan.lessons[0].why, "3 of 10 lessons (30%) were marked down for tense.")
        self.assertEqual(plan.lessons[0].topic, "animals")
        self.assertIn(plan.level, LEVELS)
        self.assertTrue(plan.summary.startswith("2 of 10 lessons passed"))

    def test_bad_input_and_a_teacher_that_fails(self):
        with self.assertRaises(ValueError):
            plan_lessons(self.client, CARD, count=0)
        for card in ({}, {"lessons": 0}, None):
            with self.assertRaises(ValueError):
                plan_lessons(self.client, card, count=2)
        self.fake.fail_with = 500
        with self.assertRaises(OllamaError):
            plan_lessons(self.client, CARD, count=2)


# ---------------------------------------------------------------------------
# the lesson loop
# ---------------------------------------------------------------------------


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.client = OllamaClient(self.fake.url, "fake:latest")

    def config(self, **overrides):
        settings = dict(topic="animals", rounds=1, exercises=2, tutor_model="fake:latest", **FAST)
        settings.update(overrides)
        return TutorConfig(**settings)

    def test_a_round_teaches_the_corrections(self):
        model = trained_model()
        before = model.stats()
        trainer = TutorTrainer(model, self.client, self.config(rounds=2))
        records = trainer.run()
        kinds = [r["kind"] for r in records]
        self.assertEqual(kinds, ["round", "round", "report"])
        self.assertEqual([r["kind"] for r in trainer.history].count("lesson"), 4)
        first = records[0]
        self.assertEqual((first["round"], first["lessons"], first["exercises"]), (1, 2, 2))
        self.assertIn(first["action"], ("2nrl", "reward", "punish"))
        self.assertEqual(records[-1]["rounds"], 2)
        self.assertEqual(records[-1]["lessons"], 4)
        after = model.stats()
        self.assertGreater(after["epochs_total"], before["epochs_total"])

    def test_a_run_can_end_with_the_next_lesson_plan(self):
        trainer = TutorTrainer(trained_model(), self.client, self.config(rounds=1, threshold=9.5, plan=2))
        seen = []
        records = trainer.run(progress=seen.append)
        self.assertEqual([r["kind"] for r in records], ["round", "report", "plan"])
        plan = records[-1]
        self.assertEqual((plan["rounds"], plan["source"]), (1, "ollama"))
        self.assertEqual(len(plan["lessons"]), 2)
        self.assertIn("agreement", plan["targets"])
        self.assertEqual(plan["lessons"][0]["exercises"], 2)  # the run's own settings ride along
        self.assertEqual([r["kind"] for r in seen][-1], "plan")  # and it reaches the progress callback
        self.assertIn("Lessons marked: 2", self.fake.prompts("plan")[0])

    def test_a_teacher_that_cannot_plan_leaves_the_lessons_standing(self):
        self.fake.plan_response = json.dumps({"lessons": []})  # unusable: the marks alone plan instead
        trainer = TutorTrainer(trained_model(), self.client, self.config(rounds=1, threshold=9.5, plan=2))
        self.assertEqual(trainer.run()[-1]["source"], "report card")

        trainer = TutorTrainer(trained_model(), self.client, self.config(rounds=1, plan=2))
        trainer.plan = lambda *a, **k: (_ for _ in ()).throw(LLMError("the teacher went home"))
        records = trainer.run()
        self.assertEqual([r["kind"] for r in records], ["round", "report"])  # the run still finished
        self.assertIn("no lesson plan", trainer.history[-1]["message"])

    def test_the_trainer_plans_from_its_own_lessons(self):
        trainer = TutorTrainer(trained_model(), self.client, self.config(rounds=1, threshold=9.5))
        trainer.run()
        plan = trainer.plan(count=2)
        self.assertEqual(len(plan.lessons), 2)
        self.assertEqual(plan.lessons[0].topic, "animals")
        self.assertEqual(plan.lessons[0].exercises, 2)
        self.assertIn("Lessons marked: 2", self.fake.prompts("plan")[-1])

    def test_lesson_records_carry_the_marking(self):
        trainer = TutorTrainer(trained_model(), self.client, self.config())
        seen = []
        trainer.run(progress=seen.append)
        lessons = [r for r in seen if r["kind"] == "lesson"]
        self.assertEqual(len(lessons), 2)
        for record in lessons:
            self.assertEqual(set(record) >= {"score", "grammar", "spelling", "fluency", "passed", "error",
                                             "correction", "comment", "sentence", "prefix", "focus"}, True)
            self.assertTrue(record["sentence"].startswith(record["prefix"]))
            self.assertIn(record["error"], (*ERROR_TYPES, "other"))

    def test_failed_sentences_are_weighted_garbage_and_corrections_are_taught(self):
        # the whole-sentence path; the diff has tests of its own
        trainer = TutorTrainer(trained_model(), self.client, self.config(threshold=9.5, diff_corrections=False))
        exercise = Exercise("e1", "the dogs run", "agreement", "the dogs run in the park")
        good_lesson = Lesson(exercise, 0, "dijkstra", "the park", "the dogs run the park",
                             grade=Grade(9.9, 10, 10, 10, True, "none", "the dogs run the park"))
        bad_lesson = Lesson(exercise, 0, "dijkstra", "run run", "the dogs run run run",
                            grade=Grade(1.0, 1, 1, 1, False, "agreement", "the dogs run in the park", "plural verb"))
        empty_lesson = Lesson(exercise, 0, "dijkstra", "", "the dogs run ",
                              grade=Grade(0.0, 0, 0, 0, False, "nonsense", "the dogs run in the park", "", "empty"))
        bad, weights, good, rewards = trainer.texts_of([good_lesson, bad_lesson, empty_lesson])
        self.assertEqual(bad, ["the dogs run run run"])  # the prefix of an empty completion is never punished
        self.assertEqual(len(weights), 1)
        self.assertIn("the dogs run the park", good)  # a passed sentence is learned as it stands
        self.assertIn("the dogs run in the park", good)  # the correction, and the teacher's own answer
        self.assertEqual(len(good), len(set(good)))
        # the reward follows the mark: 9.9/10 for the sentence the network wrote, full weight for the teacher's
        self.assertAlmostEqual(rewards[good.index("the dogs run the park")], 0.99)
        self.assertEqual(rewards[good.index("the dogs run in the park")], 1.0)

    def test_the_grades_reach_2nrl_as_weighted_garbage_and_corrections(self):
        model = ScriptedModel("the mat", "the sofa")  # exercise 1 is correct English, exercise 2 is not
        trainer = TutorTrainer(model, self.client, self.config(exercises=2, drills=2))
        record, lessons = trainer.run_round(1)
        self.assertEqual([l.grade.passed for l in lessons], [True, False])
        self.assertEqual(model.asked, ["the cat sat on ", "the dogs run "])
        (action, bad, good, options), = model.calls
        self.assertEqual(action, "two_nrl")
        self.assertEqual(bad, ["the dogs run the sofa"])
        self.assertEqual(options["bad_weights"], [trainer.weight_of(lessons[1].grade)])
        rewards = dict(zip(good, options["good_weights"]))
        self.assertAlmostEqual(rewards["the cat sat on the mat"], lessons[0].grade.score / 10)  # the mark it earned
        self.assertEqual(rewards["the dogs run in the park"], 1.0)  # the teacher's own English
        self.assertIn("the cat sat on the mat", good)  # the passed sentence
        self.assertIn("the dogs run in the park", good)  # its correction and the teacher's answer
        self.assertEqual(sum(1 for t in good if t.startswith("the cat sat on the mat number")), 2)  # the drills
        self.assertEqual((options["neg_epochs"], options["pos_epochs"]), (1, 1))
        self.assertEqual((record["neg_loss"], record["pos_loss"]), (1.25, 0.5))

    def test_reward_follows_the_mark(self):
        trainer = TutorTrainer(None, self.client, self.config())
        self.assertAlmostEqual(trainer.reward_of(Grade(score=10.0, passed=True)), 1.0)
        self.assertAlmostEqual(trainer.reward_of(Grade(score=6.0, passed=True)), 0.6)
        self.assertAlmostEqual(trainer.reward_of(Grade(score=0.0, passed=False)), 0.0)
        self.assertEqual(trainer.reward_of(Grade(score=None)), 0.0)

    def test_the_positive_phase_is_weighted_by_the_rating(self):
        model = ScriptedModel("the mat")  # every completion is correct English, but only worth its mark
        trainer = TutorTrainer(model, self.client, self.config(exercises=1))
        record, lessons = trainer.run_round(1)
        (action, good, options), = model.calls  # ("reward", texts, options)
        self.assertEqual(action, "reward")  # nothing failed: a rewarded pass, weighted by the mark
        self.assertEqual(good, ["the cat sat on the mat"])
        self.assertAlmostEqual(options["weights"][0], lessons[0].grade.score / 10)
        self.assertAlmostEqual(record["mean_reward"], lessons[0].grade.score / 10)

    def test_replay_keeps_teaching_earlier_corrections(self):
        model = ScriptedModel()
        trainer = TutorTrainer(model, self.client, self.config())
        trainer.learn(["bad one"], [1.0], ["the cat sat on the mat"], [0.7])
        trainer.learn(["bad two"], [1.0], ["the dogs run in the park"], [1.0])
        (_, _, first, _), (_, _, second, options) = model.calls
        self.assertEqual(first, ["the cat sat on the mat"])
        self.assertEqual(second, ["the dogs run in the park", "the cat sat on the mat"])
        # this round's text keeps its mark, the replayed correction is taught at the full rate
        self.assertEqual(options["good_weights"], [1.0, 1.0])
        trainer = TutorTrainer(ScriptedModel(), self.client, self.config(replay=False))
        trainer.learn(["bad one"], [1.0], ["the cat sat on the mat"], [1.0])
        trainer.learn(["bad two"], [1.0], ["the dogs run in the park"], [1.0])
        self.assertEqual(trainer.model.calls[1][2], ["the dogs run in the park"])  # replay off: only this round

    def test_replay_limit_zero_means_no_limit(self):
        texts = ["the cat sat on the mat", "the dogs run in the park", "the cat likes the mat"]
        for limit, replayed in ((0, 2), (1, 1)):
            with self.subTest(replay_limit=limit):
                model = ScriptedModel()
                trainer = TutorTrainer(model, self.client, self.config(replay_limit=limit))
                for text in texts:
                    trainer.learn(["garbage"], [1.0], [text], [1.0])
                self.assertEqual(len(trainer.replay_buffer), len(texts) if limit == 0 else limit)
                last = model.calls[-1][2]
                self.assertEqual(last[0], texts[-1])  # this round's text first, then the replayed ones
                self.assertEqual(len(last) - 1, replayed)

    def test_weight_of_grows_with_the_mistake(self):
        trainer = TutorTrainer(None, self.client, self.config(threshold=6.0, min_weight=0.25))
        near_miss = trainer.weight_of(Grade(score=5.0, passed=False))
        hopeless = trainer.weight_of(Grade(score=0.0, passed=False))
        self.assertLess(near_miss, hopeless)
        self.assertAlmostEqual(hopeless, 1.0)
        self.assertAlmostEqual(trainer.weight_of(Grade(score=6.0, passed=True)), 0.25)
        self.assertEqual(trainer.weight_of(Grade(score=None, passed=False)), 1.0)  # unrated counts as a full failure

    def test_adapt_drills_the_weakest_points_next_round(self):
        trainer = TutorTrainer(ScriptedModel("the sofa"), self.client, self.config(rounds=2))
        trainer.run()
        prompts = self.fake.prompts("exercises")
        self.assertEqual(len(prompts), 2)
        self.assertNotIn("keeps making these mistakes", prompts[0])
        self.assertIn("keeps making these mistakes", prompts[1])
        self.assertIn("agreement", prompts[1])
        self.assertEqual(trainer.weak, ["agreement"])

    def test_no_adapt_keeps_the_syllabus(self):
        trainer = TutorTrainer(ScriptedModel("the sofa"), self.client, self.config(rounds=2, adapt=False))
        trainer.run()
        self.assertTrue(all("keeps making these mistakes" not in p for p in self.fake.prompts("exercises")))
        self.assertEqual(trainer.weak, [])

    def test_drills_join_the_positive_phase(self):
        trainer = TutorTrainer(trained_model(), self.client, self.config(drills=3, threshold=9.5))
        record, _lessons = trainer.run_round(1)
        self.assertEqual(record["drills"], 3)
        self.assertEqual(len(self.fake.prompts("drills")), 1)
        self.assertGreaterEqual(record["good"], 3)

    def test_dry_run_leaves_the_model_alone(self):
        model = trained_model()
        before = model.stats()
        trainer = TutorTrainer(model, self.client, self.config(learn=False, threshold=9.5))
        records = trainer.run()
        self.assertIsNone(records[0]["action"])
        self.assertGreater(records[0]["bad"], 0)  # what it would have punished is still reported
        self.assertEqual(model.stats()["epochs_total"], before["epochs_total"])
        self.assertEqual(model.stats()["twonrl_runs"], before["twonrl_runs"])

    def test_twonrl_per_lesson(self):
        trainer = TutorTrainer(trained_model(), self.client, self.config(twonrl_per="lesson", threshold=9.5))
        record, lessons = trainer.run_round(1)
        self.assertEqual(len(lessons), 2)
        self.assertEqual(record["bad"], 2)  # summed over the lessons, not the last one only
        self.assertIsNotNone(record["action"])

    def test_attempts_ask_the_network_more_than_once(self):
        trainer = TutorTrainer(trained_model(), self.client, self.config(attempts=3, mode="dijkstra"))
        _record, lessons = trainer.run_round(1)
        self.assertEqual(len(lessons), 6)
        self.assertEqual([l.mode for l in lessons[:3]], ["dijkstra", "sample", "sample"])

    def test_stop_event_ends_the_run(self):
        stop = threading.Event()
        stop.set()
        trainer = TutorTrainer(trained_model(), self.client, self.config(rounds=3))
        records = trainer.run(stop_event=stop)
        self.assertEqual([r["kind"] for r in records], ["report"])
        self.assertEqual(self.fake.prompts("exercises"), [])

    def test_count_model_learns_too(self):
        model = trained_model(CountRewardNet)
        trainer = TutorTrainer(model, self.client, self.config(threshold=9.5, strength=1.0, diff_corrections=False))
        record, _lessons = trainer.run_round(1)
        self.assertEqual(record["action"], "2nrl")
        self.assertGreater(model.stats()["twonrl_runs"], 0)

    def test_a_correction_moves_only_the_words_it_changes(self):
        model = trained_model(CountRewardNet)
        trainer = TutorTrainer(model, self.client, self.config(threshold=9.5, strength=1.0, keep_weight=0.0))
        record, lessons = trainer.run_round(1)
        self.assertEqual(record["action"], "correct+reward")  # the diff, then the teacher's own English
        self.assertEqual(record["corrections"], sum(1 for l in lessons if not l.grade.passed))
        self.assertGreater(record["edits"], 0)
        self.assertGreater(record["penalised"], 0)
        self.assertGreater(record["rewarded"], 0)
        # what the teacher changed rides along with the lesson, for the panel to show
        changes = [l.changes for l in lessons if not l.grade.passed]
        self.assertTrue(any(changes), "a failed lesson should say what the teacher changed")
        for change in [c for spans in changes for c in spans]:
            self.assertEqual(set(change), {"op", "wrong", "right"})
        self.assertTrue(all(l.changes == [] for l in lessons if l.grade.passed))
        # a failed sentence is no longer punished whole: the penalty is the few steps the diff blames
        stats = model.stats()
        self.assertGreater(stats["penalties_total"], 0)
        self.assertLessEqual(stats["penalties_total"], record["penalised"] * 1.0)
        self.assertEqual(record["bad"], 0)  # nothing went to 2NRL as whole-sentence garbage

    def test_the_diff_can_be_turned_off(self):
        model = trained_model(CountRewardNet)
        trainer = TutorTrainer(model, self.client, self.config(threshold=9.5, diff_corrections=False))
        lesson = Lesson(Exercise("e1", "the dogs run", "agreement", "the dogs run in the park"), 0, "dijkstra",
                        "run run", "the dogs run run run",
                        grade=Grade(1.0, 1, 1, 1, False, "agreement", "the dogs run in the park"))
        self.assertEqual(trainer.corrections_of([lesson]), [])
        bad, _weights, good, _rewards = trainer.texts_of([lesson])
        self.assertEqual(bad, ["the dogs run run run"])  # the whole sentence is garbage again
        self.assertIn("the dogs run in the park", good)

    def test_invalid_configuration_is_refused(self):
        for overrides in ({"topic": " "}, {"rounds": 0}, {"mode": "nope"}, {"twonrl_per": "hourly"},
                          {"threshold": 11}, {"grammar_weight": 2}, {"min_weight": -1}, {"batch": 0},
                          {"exercises": 0}, {"attempts": 0}, {"neg_lr": -1}, {"batch_size": 0},
                          {"keep_weight": 1.5}, {"keep_weight": -0.1}):
            with self.subTest(**overrides):
                with self.assertRaises(ValueError):
                    TutorTrainer(None, self.client, self.config(**overrides))

    def test_ollama_failure_propagates(self):
        self.fake.fail_with = 503
        trainer = TutorTrainer(trained_model(), self.client, self.config())
        with self.assertRaises(OllamaError):
            trainer.run()


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.client, self.server, self.service = start_server(
            self.addCleanup, ollama_url=self.fake.url, ollama_model="fake:latest",
        )
        self.service.model.train(CORPUS, epochs=6, lr=0.6, batch_size=4)

    def test_describe(self):
        status, body, _ = self.client.get("/api/tutor")
        self.assertEqual(status, 200)
        self.assertEqual((body["url"], body["model"]), (self.fake.url, "fake:latest"))
        self.assertEqual(body["error_types"], list(ERROR_TYPES))
        self.assertEqual(body["modes"], ["dijkstra", "beam", "sample"])
        self.assertEqual(body["defaults"]["threshold"], 6.0)

    def test_lesson_dry_run(self):
        status, body, _ = self.client.post("/api/tutor/lesson", {"topic": "animals", "exercises": 2})
        self.assertEqual(status, 200)
        self.assertEqual((body["source"], body["model"]), ("ollama", "fake:latest"))
        self.assertEqual(len(body["lessons"]), 2)
        first = body["lessons"][0]
        self.assertEqual(first["exercise"]["prefix"], "the cat sat on")
        self.assertTrue(first["sentence"].startswith("the cat sat on "))
        self.assertIn(first["grade"]["error"], (*ERROR_TYPES, "other"))
        self.assertEqual(body["report"]["lessons"], 2)
        self.assertIsNone(self.service.job_status())  # nothing was trained

    def test_lesson_with_given_prefixes_skips_the_exercise_writer(self):
        status, body, _ = self.client.post(
            "/api/tutor/lesson", {"prefixes": ["the cat sat on", "the dogs run"], "exercises": 5},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["source"], "given")
        self.assertEqual([e["prefix"] for e in body["exercises"]], ["the cat sat on", "the dogs run"])
        self.assertEqual(self.fake.prompts("exercises"), [])

    def test_start_history_and_report(self):
        status, body, _ = self.client.post(
            "/api/tutor/start",
            {"topic": "animals", "rounds": 2, "exercises": 2, "threshold": 9.5, "drills": 2, **FAST},
        )
        self.assertEqual(status, 202)
        self.assertEqual(body["job"]["type"], "tutor")
        self.assertEqual(body["config"]["topic"], "animals")
        job = wait_for_job(self.client)
        self.assertEqual((job["state"], job["error"]), ("done", None))
        status, history, _ = self.client.get("/api/tutor/history")
        self.assertEqual(status, 200)
        kinds = [r["kind"] for r in history["history"]]
        self.assertEqual(kinds.count("lesson"), 4)
        self.assertEqual(kinds.count("round"), 2)
        self.assertEqual(kinds[-1], "report")
        report = history["history"][-1]
        self.assertEqual((report["rounds"], report["lessons"]), (2, 4))
        self.assertGreaterEqual(report["errors"].get("agreement", 0), 1)
        self.assertEqual(report["weakest"][0], "agreement")
        self.assertGreater(self.service.model.stats()["twonrl_runs"], 0)

    def test_plan_from_a_report_card(self):
        card = {"lessons": 6, "passed": 1, "pass_rate": 1 / 6, "mean_score": 3.0,
                "errors": {"agreement": 4, "tense": 1}}
        status, body, _ = self.client.post(
            "/api/tutor/plan", {"report": card, "topic": "animals", "count": 2, "exercises": 3},
        )
        self.assertEqual(status, 200)
        self.assertEqual((body["source"], body["provider"], body["model"]), ("ollama", "ollama", "fake:latest"))
        self.assertEqual(body["report"], card)
        plan = body["plan"]
        self.assertEqual(len(plan["lessons"]), 2)
        self.assertEqual(plan["lessons"][0]["targets"], "agreement")
        self.assertEqual(plan["lessons"][0]["exercises"], 3)
        self.assertEqual([p["error"] for p in plan["weak"]], ["agreement", "tense"])
        self.assertIn("agreement x4", self.fake.prompts("plan")[0])

    def test_plan_falls_back_to_the_last_run(self):
        status, answer, _ = self.client.post("/api/tutor/plan", {"topic": "animals"})
        self.assertEqual(status, 400, answer)  # nothing has been marked yet
        self.assertIn("no report card yet", answer["error"])
        self.client.post("/api/tutor/start",
                         {"topic": "animals", "rounds": 1, "exercises": 2, "threshold": 9.5, **FAST})
        wait_for_job(self.client)
        status, body, _ = self.client.post("/api/tutor/plan", {"topic": "animals", "count": 1})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["report"]["kind"], "report")
        self.assertEqual(body["report"]["lessons"], 2)
        self.assertEqual(len(body["plan"]["lessons"]), 1)

    def test_a_run_can_plan_its_own_next_lessons(self):
        status, _body, _ = self.client.post(
            "/api/tutor/start",
            {"topic": "animals", "rounds": 1, "exercises": 2, "threshold": 9.5, "plan": 2, **FAST},
        )
        self.assertEqual(status, 202)
        wait_for_job(self.client)
        _status, history, _ = self.client.get("/api/tutor/history")
        record = history["history"][-1]
        self.assertEqual(record["kind"], "plan")
        self.assertEqual(len(record["lessons"]), 2)
        self.assertIn("agreement", record["targets"])

    def test_plan_bad_requests(self):
        for body, expected in (
            ({"report": "a card", "topic": "animals"}, "'report' must be an object"),
            ({"report": {"lessons": 3}, "count": 0}, "count"),
            ({"report": {"lessons": 0}}, "report card of at least one lesson"),
        ):
            with self.subTest(**body):
                status, answer, _ = self.client.post("/api/tutor/plan", body)
                self.assertEqual(status, 400, answer)
                self.assertIn(expected, answer["error"])
        status, answer, _ = self.client.post(
            "/api/tutor/plan", {"report": {"lessons": 2, "errors": {"tense": 1}}, "url": closed_port_url(),
                                "timeout": 2},
        )
        self.assertEqual(status, 502, answer)

    def test_bad_requests(self):
        for body, expected in (
            ({"topic": " "}, "topic"),
            ({"mode": "nope"}, "mode"),
            ({"twonrl_per": "hourly"}, "twonrl_per"),
            ({"threshold": 99}, "threshold"),
            ({"exercises": 0}, "exercises"),
        ):
            with self.subTest(**body):
                status, answer, _ = self.client.post("/api/tutor/start", body)
                self.assertEqual(status, 400, answer)
                self.assertIn(expected, answer["error"])

    def test_unreachable_ollama_is_a_502(self):
        status, answer, _ = self.client.post(
            "/api/tutor/lesson", {"topic": "animals", "url": closed_port_url(), "timeout": 2},
        )
        self.assertEqual(status, 502)
        self.assertIn("cannot reach Ollama", answer["error"])

    def test_a_running_job_blocks_a_second_one(self):
        body = {"topic": "animals", "rounds": 3, "exercises": 2, **FAST}
        status, _first, _ = self.client.post("/api/tutor/start", body)
        self.assertEqual(status, 202)
        status, answer, _ = self.client.post("/api/tutor/start", body)
        self.assertEqual(status, 409, answer)
        self.client.post("/api/job/stop")
        wait_for_job(self.client)


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def setUp(self):
        self.fake = start_fake(self.addCleanup)
        self.dir = tempfile.mkdtemp(prefix="radixnet-tutor-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.model = os.path.join(self.dir, "model.json")

    def run_cli(self, *args, expect=0):
        cmd = [sys.executable, "-m", "radixnet", "--model", self.model, "--backend", "python", "--json", *args]
        env = dict(os.environ, PYTHONWARNINGS="ignore", OLLAMA_HOST=self.fake.url, RADIXNET_TUTOR_MODEL="fake:latest")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(proc.returncode, expect, proc.stderr[-2000:])
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def train(self):
        data = os.path.join(self.dir, "corpus.txt")
        with open(data, "w", encoding="utf-8") as fh:
            fh.write("\n".join(CORPUS) + "\n")
        self.run_cli("train", "--data", data, "--epochs", "6", "--lr", "0.6", "--batch-size", "4")

    def test_dry_run_saves_nothing(self):
        self.train()
        before = os.path.getmtime(self.model)
        doc = self.run_cli("tutor", "--topic", "animals", "--rounds", "1", "--exercises", "2", "--dry-run")
        self.assertIsNone(doc["out"])
        self.assertIsNone(doc["saved"])
        self.assertEqual(doc["config"]["tutor_model"], "fake:latest")
        self.assertEqual(len(doc["lessons"]), 2)
        self.assertEqual(doc["report"]["lessons"], 2)
        self.assertEqual(os.path.getmtime(self.model), before)

    def test_lessons_train_and_report(self):
        self.train()
        report = os.path.join(self.dir, "report.json")
        doc = self.run_cli("tutor", "--topic", "animals", "--rounds", "2", "--exercises", "2", "--threshold", "9.5",
                           "--neg-epochs", "1", "--pos-epochs", "1", "--report", report)
        self.assertEqual(doc["out"], self.model)
        self.assertEqual([r["kind"] for r in doc["records"]], ["round", "round", "report"])
        self.assertGreaterEqual(doc["report"]["errors"].get("agreement", 0), 1)
        self.assertGreater(doc["stats"]["twonrl_runs"], 0)
        with open(report, encoding="utf-8") as fh:
            written = json.load(fh)
        self.assertEqual(sorted(written), ["config", "lessons", "plan", "records", "report"])
        self.assertIsNone(written["plan"])  # nothing was planned without --plan

    def test_plan_writes_the_next_lessons(self):
        self.train()
        report = os.path.join(self.dir, "planned.json")
        doc = self.run_cli("tutor", "--topic", "animals", "--rounds", "1", "--exercises", "2", "--threshold", "9.5",
                           "--dry-run", "--plan", "2", "--report", report)
        self.assertEqual([r["kind"] for r in doc["records"]], ["round", "report", "plan"])
        plan = doc["plan"]
        self.assertEqual(len(plan["lessons"]), 2)
        self.assertEqual(plan["source"], "ollama")
        self.assertIn("agreement", plan["targets"])
        with open(report, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["plan"]["lessons"], plan["lessons"])

    def test_bad_options_and_unreachable_ollama(self):
        self.run_cli("tutor", "--topic", " ", expect=1)
        self.run_cli("tutor", "--url", closed_port_url(), "--timeout", "2", "--rounds", "1", expect=1)


if __name__ == "__main__":
    unittest.main()
