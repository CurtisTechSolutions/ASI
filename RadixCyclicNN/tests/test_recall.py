"""Tests for radixnet.recall - the speech and image tutors, and the blame they hand the negative network.

The recall tutor needs no LLM: an utterance and a picture were encoded into
text and trained on, so asking the network to write that text back is an
exercise whose correction is on file.  These tests cover the exercise, every
way a completion can be wrong, the mark, the loop over a real model, the
faults it produces and the CLI / API surfaces.  None of it needs Pillow, torch
or a transcriber - the marking is over the payload bytes both codecs already
produce - so nothing here is skipped.
"""

import array
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import blame, recall, speech, vision  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402

RATE = 8000


def tone(seconds=0.05, freq=220.0, amplitude=0.6, rate=RATE):
    """A sine wave as floats in [-1, 1]."""
    return array.array("f", [amplitude * math.sin(2 * math.pi * freq * i / rate) for i in range(int(rate * seconds))])


def utterance(seconds=0.05, freq=220.0, transcript="hello there"):
    """``(waveform_text, taught)`` for one short recording."""
    taught = speech.teach(speech.wav_bytes(tone(seconds, freq), RATE), transcript=transcript, rate=RATE)
    return next(text for text in taught["texts"] if "aud:" in text), taught


def waveform(payload, rate=RATE):
    return speech.pack_text("mu", rate, 1, payload)


def picture(n=3 * 8 * 8, step=7):
    """An image text with a repeatable payload (no Pillow needed: the format is pure)."""
    return vision.pack_text("tiny", 64, 64, bytes((x * step + 11) % 256 for x in range(n))), n


class TestExercise(unittest.TestCase):
    def test_modality_comes_from_the_header(self):
        text, _ = utterance()
        self.assertEqual(recall.modality_of(text), "speech")
        self.assertEqual(recall.modality_of(picture()[0]), "image")
        with self.assertRaises(ValueError):
            recall.modality_of("the cat sat on the mat")

    def test_a_spoken_cue_is_the_token_and_the_header(self):
        text, taught = utterance()
        cue = recall.cue_of(text)
        self.assertEqual(cue, f"{taught['token']} aud:mu:{RATE}x1:")
        self.assertTrue(text.startswith(cue))
        self.assertNotIn("=", cue)  # nothing of the payload is given away

    def test_an_image_cue_carries_a_lead_so_the_picture_is_identified(self):
        text, _ = picture()
        cue = recall.cue_of(text)
        self.assertTrue(cue.startswith("img:tiny:64x64:"))
        self.assertEqual(len(cue), len("img:tiny:64x64:") + recall.DEFAULT_LEAD["image"])
        self.assertEqual(recall.cue_of(text, lead=0), "img:tiny:64x64:")

    def test_the_cue_never_runs_past_a_short_text(self):
        text = vision.pack_text("tiny", 64, 64, b"ab")
        self.assertEqual(recall.cue_of(text, lead=999), text)

    def test_exercises_skip_what_is_not_encoded(self):
        text, _ = utterance()
        items = recall.exercises([text, "the cat sat on the mat", "", picture()[0]], labels=["a", "b", "c", "d"])
        self.assertEqual([item.modality for item in items], ["speech", "image"])
        self.assertEqual(items[0].label, "a")
        self.assertEqual(items[0].reference, text)
        self.assertEqual(items[0].cue + items[0].answer, text)
        # the index is where it came from, not where it ended up: "b" and "c" were skipped
        self.assertEqual([item.index for item in items], [0, 3])
        self.assertEqual([item.label for item in items], ["a", "d"])


class TestMarkingSpeech(unittest.TestCase):
    def setUp(self):
        self.text, self.taught = utterance()
        self.token = self.taught["token"]
        self.samples = len(speech.parse_text(self.text)[3])

    def check(self, written, **kw):
        return recall.check_speech(written, self.text, **kw)

    def test_a_perfect_recall_scores_ten_and_names_nothing(self):
        facts = self.check(self.text)
        self.assertEqual(recall.mark(facts), 10.0)
        self.assertEqual(blame.recall_reason(facts), "none")
        self.assertTrue(recall.grade(facts, self.text).passed)

    def test_prose_is_unreadable(self):
        facts = self.check(f"{self.token} the cat sat on the mat")
        self.assertFalse(facts["readable"])
        self.assertEqual(recall.mark(facts), 0.0)
        self.assertEqual(blame.recall_reason(facts), "unreadable")

    def test_stopping_early_is_truncated(self):
        facts = self.check(self.text[: len(self.text) // 2])
        self.assertLess(facts["length_ratio"], blame.RECALL_TRUNCATED)
        self.assertEqual(blame.recall_reason(facts), "truncated")
        self.assertLess(recall.mark(facts), 6.0)

    def test_rambling_past_the_end_is_overrun_and_costs_the_mark(self):
        payload = speech.parse_text(self.text)[3]
        facts = self.check(f"{self.token} " + waveform(payload + payload))
        self.assertGreater(facts["length_ratio"], blame.RECALL_OVERRUN)
        self.assertEqual(blame.recall_reason(facts), "overrun")
        self.assertAlmostEqual(recall.mark(facts), 5.0, places=6)  # perfect bytes, half of them waste

    def test_silence_is_named_even_when_the_bytes_almost_agree(self):
        quiet = speech.mulaw_encode(array.array("f", [0.0] * self.samples))
        facts = self.check(f"{self.token} " + waveform(quiet))
        self.assertTrue(facts["flat"])
        self.assertFalse(facts["reference_flat"])
        self.assertEqual(blame.recall_reason(facts), "silence")
        self.assertLessEqual(recall.mark(facts), 2.0)

    def test_silence_is_not_a_fault_when_the_recording_was_silent(self):
        quiet_text, taught = utterance(freq=220.0, transcript="")
        quiet = speech.mulaw_encode(array.array("f", [0.0] * self.samples))
        reference = f"{taught['token']} " + waveform(quiet)
        facts = recall.check_speech(reference, reference)
        self.assertTrue(facts["flat"] and facts["reference_flat"])
        self.assertEqual(blame.recall_reason(facts), "none")

    def test_railing_against_the_limits_is_clipping(self):
        railed = bytes([0 if (i // 8) % 2 else 255 for i in range(self.samples)])
        facts = self.check(f"{self.token} " + waveform(railed))
        self.assertTrue(facts["extreme"])
        self.assertEqual(blame.recall_reason(facts), "clipping")

    def test_different_words_are_a_mishearing(self):
        facts = self.check(self.text, said="hello there", heard="yellow bear")
        self.assertIs(facts["match"], False)
        self.assertEqual(blame.recall_reason(facts), "mishearing")
        self.assertLessEqual(recall.mark(facts), 4.0)

    def test_the_same_words_are_not_a_mishearing(self):
        facts = self.check(self.text, said="Hello there", heard="hello  there")
        self.assertIs(facts["match"], True)
        self.assertEqual(blame.recall_reason(facts), "none")

    def test_nothing_is_claimed_about_the_words_without_a_transcript(self):
        self.assertIsNone(self.check(self.text)["match"])

    def test_right_length_but_wrong_is_a_distortion(self):
        payload = speech.parse_text(self.text)[3]
        wrong = bytes((b + 90) % 256 for b in payload)
        facts = self.check(f"{self.token} " + waveform(wrong))
        self.assertTrue(facts["readable"])
        self.assertAlmostEqual(facts["length_ratio"], 1.0)
        self.assertLess(facts["agreement"], blame.RECALL_AGREEMENT)
        self.assertEqual(blame.recall_reason(facts), "distortion")

    def test_repairing_the_base64_is_garbled_and_costs_a_point(self):
        clean = self.text + "=="  # the payload is complete; the padding is junk
        facts = self.check(clean)
        self.assertTrue(facts["repaired"])
        self.assertEqual(facts["written_bytes"], facts["expected_bytes"])
        self.assertEqual(blame.recall_reason(facts), "garbled")
        self.assertAlmostEqual(recall.mark(facts), 9.0, places=6)


class TestMarkingImages(unittest.TestCase):
    def setUp(self):
        self.text, self.size = picture()

    def check(self, written):
        return recall.check_image(written, self.text)

    def test_a_perfect_recall_scores_ten(self):
        facts = self.check(self.text)
        self.assertEqual(recall.mark(facts), 10.0)
        self.assertEqual(blame.recall_reason(facts), "none")

    def test_prose_is_unreadable(self):
        facts = self.check("the cat sat on the mat")
        self.assertEqual(blame.recall_reason(facts), "unreadable")

    def test_a_flat_payload_is_blank_not_silence(self):
        facts = self.check(vision.pack_text("tiny", 64, 64, bytes([128] * self.size)))
        self.assertTrue(facts["flat"])
        self.assertEqual(blame.recall_reason(facts), "blank")

    def test_all_extremes_is_noise(self):
        facts = self.check(vision.pack_text("tiny", 64, 64, bytes([0 if i % 2 else 255 for i in range(self.size)])))
        self.assertTrue(facts["extreme"])
        self.assertEqual(blame.recall_reason(facts), "noise")

    def test_right_length_but_wrong_is_a_drift(self):
        truth = vision.parse_text(self.text)[3]
        facts = self.check(vision.pack_text("tiny", 64, 64, bytes((b + 90) % 256 for b in truth)))
        self.assertEqual(blame.recall_reason(facts), "drift")

    def test_an_image_never_claims_to_have_heard_anything(self):
        self.assertIsNone(self.check(self.text)["match"])
        self.assertEqual(self.check(self.text)["heard"], "")


class TestAgreement(unittest.TestCase):
    def test_an_exact_payload_agrees_completely(self):
        self.assertEqual(recall._agreement(bytes(range(64)), bytes(range(64))), 1.0)

    def test_bytes_never_written_count_as_total_disagreement(self):
        self.assertAlmostEqual(recall._agreement(bytes(32), bytes(64)), 0.5)

    def test_a_quantisation_step_barely_costs_anything(self):
        truth = bytes(range(64))
        near = bytes((b + 1) % 256 for b in truth)
        self.assertGreater(recall._agreement(near, truth), 0.95)

    def test_a_byte_further_than_the_tolerance_agrees_not_at_all(self):
        self.assertEqual(recall._agreement(bytes([200]), bytes([int(200 - recall.TOLERANCE)])), 0.0)

    def test_an_empty_truth_is_matched_only_by_an_empty_answer(self):
        self.assertEqual(recall._agreement(b"", b""), 1.0)
        self.assertEqual(recall._agreement(b"abc", b""), 0.0)


class TestGrade(unittest.TestCase):
    def test_a_pass_carries_no_reason_correction_or_comment(self):
        text, _ = picture()
        g = recall.grade(recall.check_image(text, text), text)
        self.assertTrue(g.passed)
        self.assertEqual((g.error, g.correction, g.comment), ("none", "", ""))

    def test_a_failure_carries_the_original_as_its_correction(self):
        text, _ = picture()
        g = recall.grade(recall.check_image("img:tiny:64x64:AAAA", text), text)
        self.assertFalse(g.passed)
        self.assertEqual(g.correction, text)
        self.assertEqual(g.error, "truncated")
        self.assertIn("stops after", g.comment)

    def test_the_threshold_decides(self):
        text, _ = picture()
        truth = vision.parse_text(text)[3]
        near = vision.pack_text("tiny", 64, 64, bytes((b + 4) % 256 for b in truth))
        facts = recall.check_image(near, text)
        self.assertTrue(recall.grade(facts, text, threshold=6.0).passed)
        self.assertFalse(recall.grade(facts, text, threshold=10.0).passed)

    def test_the_grade_round_trips_through_json(self):
        text, _ = picture()
        g = recall.grade(recall.check_image("img:tiny:64x64:AAAA", text), text)
        doc = json.loads(json.dumps(g.to_dict()))
        self.assertEqual(doc["error"], "truncated")
        self.assertEqual(doc["graded_by"], "recall")
        self.assertIn("agreement", doc["facts"])


class TestQuiz(unittest.TestCase):
    def setUp(self):
        self.text, self.taught = utterance()
        self.model = CountRewardNet(seed=7)
        self.model.train([self.text], epochs=3)

    def test_a_quiz_marks_what_the_network_writes_back(self):
        lessons = recall.quiz(self.model, [self.text], length=120)
        self.assertEqual(len(lessons), 1)
        lesson = lessons[0]
        self.assertEqual(lesson.exercise.modality, "speech")
        self.assertTrue(lesson.text.startswith(lesson.exercise.cue))
        self.assertIsNotNone(lesson.grade.score)
        self.assertEqual(lesson.sentence, lesson.text)

    def test_a_capped_quiz_is_marked_against_what_it_asked_for(self):
        lessons = recall.quiz(self.model, [self.text], length=40)
        # 40 base64 characters of the payload were asked for, so 30 bytes are expected - not the whole utterance
        self.assertEqual(lessons[0].grade.facts["expected_bytes"], 30)
        self.assertLess(lessons[0].grade.facts["expected_bytes"], len(speech.parse_text(self.text)[3]))

    def test_a_model_that_remembers_perfectly_passes(self):
        class Perfect:
            def __init__(self, answer):
                self.answer = answer

            def predict(self, prefix, **options):
                class R:
                    text = self.answer[len(prefix):]
                    cost = 0.0
                    reached_end = True
                return R()

        lessons = recall.quiz(Perfect(self.text), [self.text])
        self.assertTrue(lessons[0].grade.passed)
        self.assertEqual(lessons[0].grade.score, 10.0)

    def test_attempts_stop_at_the_first_pass(self):
        class Perfect:
            def __init__(self, answer):
                self.answer = answer
                self.calls = 0

            def predict(self, prefix, **options):
                self.calls += 1
                class R:
                    text = self.answer[len(prefix):]
                    cost = 0.0
                    reached_end = True
                return R()

        model = Perfect(self.text)
        recall.quiz(model, [self.text], attempts=4)
        self.assertEqual(model.calls, 1)

    def test_a_transcript_follows_its_own_text_past_the_skipped_ones(self):
        class Perfect:
            def __init__(self, answer):
                self.answer = answer

            def predict(self, prefix, **options):
                class R:
                    text = self.answer[len(prefix):]
                    cost = 0.0
                    reached_end = True
                return R()

        # the first text is not encoded and is skipped, so the second must still get the second transcript
        lessons = recall.quiz(Perfect(self.text), ["not encoded at all", self.text], said=["wrong words", "hello there"])
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0].grade.facts["said"], "hello there")

    def test_progress_sees_every_lesson(self):
        seen = []
        recall.quiz(self.model, [self.text], length=40, progress=seen.append)
        self.assertEqual(len(seen), 1)
        self.assertIn("grade", seen[0])

    def test_a_stop_event_ends_the_quiz(self):
        import threading

        stop = threading.Event()
        stop.set()
        self.assertEqual(recall.quiz(self.model, [self.text], stop_event=stop), [])

    def test_the_report_card_counts_the_reasons(self):
        lessons = recall.quiz(self.model, [self.text], length=80)
        card = recall.report_card(lessons)
        self.assertEqual(card["lessons"], len(lessons))
        self.assertEqual(card["modality"], "speech")
        self.assertLessEqual(card["passed"], card["lessons"])
        self.assertEqual(sum(card["reasons"].values()), card["lessons"] - card["passed"])

    def test_a_mixed_report_card_says_so(self):
        self.assertEqual(recall.report_card([])["modality"], "")


class TestBlameFromRecall(unittest.TestCase):
    def setUp(self):
        self.text, _ = picture()
        self.negative = NegativeNet(seed=7)

    def lesson(self, written, threshold=6.0):
        facts = recall.check_image(written, self.text)
        return recall.RecallLesson(
            exercise=recall.RecallExercise("r1", "image", self.text, recall.cue_of(self.text), "a.png"),
            attempt=0, mode="beam", continuation="", text=written,
            grade=recall.grade(facts, self.text, threshold),
        )

    def test_a_failure_becomes_a_fault_with_the_original_as_its_correction(self):
        faults, passed = blame.faults_from_recall([self.lesson("img:tiny:64x64:AAAA")])
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0]["reason"], "truncated")
        self.assertEqual(faults[0]["correction"], self.text)
        self.assertEqual(faults[0]["source"], "recall")
        self.assertGreater(faults[0]["severity"], 1.0)  # a bad mark is blamed hard
        self.assertIn(self.text, passed)  # the original is correct by construction

    def test_a_pass_only_clears(self):
        faults, passed = blame.faults_from_recall([self.lesson(self.text)])
        self.assertEqual(faults, [])
        self.assertEqual(passed, [self.text])

    def test_dicts_work_as_well_as_lessons(self):
        faults, _ = blame.faults_from_recall([self.lesson("img:tiny:64x64:AAAA").to_dict()])
        self.assertEqual(faults[0]["reason"], "truncated")

    def test_a_lesson_without_a_named_reason_is_classified_from_its_facts(self):
        doc = self.lesson("img:tiny:64x64:AAAA").to_dict()
        doc["grade"]["error"] = "none"
        faults, _ = blame.faults_from_recall([doc])
        self.assertEqual(faults[0]["reason"], "truncated")

    def test_a_failure_whose_facts_name_nothing_falls_back(self):
        doc = self.lesson("img:tiny:64x64:AAAA").to_dict()
        doc["grade"]["error"] = "none"  # marked a failure, but nothing about it is wrong
        doc["grade"]["facts"] = recall.check_image(self.text, self.text)
        faults, _ = blame.faults_from_recall([doc])
        self.assertEqual(faults[0]["reason"], blame.DEFAULT_REASON)

    def test_facts_that_say_nothing_at_all_are_unreadable(self):
        self.assertEqual(blame.recall_reason({}), "unreadable")
        self.assertEqual(blame.recall_reason(None), "unreadable")

    def test_teaching_blames_only_what_the_network_got_wrong(self):
        truth = vision.parse_text(self.text)[3]
        half = len(truth) // 2
        # the same picture, remembered correctly for the first half and badly for the second
        wrong = vision.pack_text("tiny", 64, 64, truth[:half] + bytes((b + 90) % 256 for b in truth[half:]))
        report = blame.teach_recall(self.negative, [self.lesson(wrong)], source="vision")
        self.assertEqual(report["blamed"], 1)
        self.assertEqual(report["source"], "vision")
        self.assertEqual(report["reasons"], {"drift": 1})
        self.assertGreater(report["edges"], 0)
        self.assertGreater(self.negative.judge(wrong)["blame"], 0.0)
        # the opening both texts share is not the mistake, so it carries no blame
        self.assertEqual(self.negative.judge(self.text[: len("img:tiny:64x64:") + 40])["blame"], 0.0)

    def test_what_it_remembered_clears_blame(self):
        blame.teach_recall(self.negative, [self.lesson("img:tiny:64x64:AAAA")])
        before = self.negative.judge(self.text)["blame"]
        report = blame.teach_recall(self.negative, [self.lesson(self.text)])
        self.assertEqual(report["blamed"], 0)
        self.assertGreaterEqual(report["cleared"], 0)
        self.assertLessEqual(self.negative.judge(self.text)["blame"], before)

    def test_the_reasons_are_the_modality_vocabulary(self):
        self.assertIn("silence", blame.SPEECH_REASONS)
        self.assertIn("blank", blame.IMAGE_REASONS)
        for tag in ("unreadable", "truncated", "overrun", "garbled"):
            self.assertIn(tag, blame.SPEECH_REASONS)
            self.assertIn(tag, blame.IMAGE_REASONS)


class TestEndToEnd(unittest.TestCase):
    def test_teach_quiz_blame_and_judge(self):
        text, _ = utterance()
        model = CountRewardNet(seed=11)
        model.train([text], epochs=3)
        lessons = recall.quiz(model, [text], length=120)
        negative = NegativeNet(seed=11)
        report = blame.teach_recall(negative, lessons, source="speech")
        if report["blamed"]:
            self.assertGreater(negative.graph.total_blame, 0.0)
            self.assertTrue(negative.reasons())
            self.assertEqual(negative.recent(1)[0]["source"], "speech")


class TestCli(unittest.TestCase):
    """`speech tutor` and `image tutor`: the two commands the recall tutor is driven by."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.model = os.path.join(self.dir.name, "m.count.json")
        self.negative = os.path.join(self.dir.name, "m.count.negative.json")
        self.wav = os.path.join(self.dir.name, "a.wav")
        with open(self.wav, "wb") as fh:
            fh.write(speech.wav_bytes(tone(), RATE))

    def run_cli(self, *argv):
        """Run the CLI with its console captured; returns ``(exit code, stdout)``."""
        from radixnet.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(["--model", self.model, "--kind", "count", *argv])
        return code, out.getvalue()

    def test_the_speech_tutor_trains_quizzes_and_reports(self):
        code, out = self.run_cli("speech", "tutor", self.wav, "--text", "hello", "--train", "--length", "80")
        self.assertEqual(code, 0, out)
        self.assertTrue(os.path.isfile(self.model))
        self.assertIn("report card:", out)
        self.assertIn("the token and the header alone", out)

    def test_blame_writes_the_negative_network_beside_the_model(self):
        code, out = self.run_cli("speech", "tutor", self.wav, "--text", "hello", "--train", "--length", "80", "--blame")
        self.assertEqual(code, 0, out)
        self.assertTrue(os.path.isfile(self.negative))
        with open(self.negative, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["format"], "radixnet-negative")

    def test_without_blame_no_negative_network_is_written(self):
        self.run_cli("speech", "tutor", self.wav, "--text", "hello", "--train", "--length", "80")
        self.assertFalse(os.path.isfile(self.negative))

    def test_a_missing_recording_is_an_error_not_a_traceback(self):
        code, out = self.run_cli("speech", "tutor", os.path.join(self.dir.name, "nope.wav"))
        self.assertEqual(code, 1)
        self.assertIn("error:", out)

    def test_the_json_mode_carries_the_lessons_and_the_report(self):
        code, out = self.run_cli("--json", "speech", "tutor", self.wav, "--text", "hello", "--train",
                                 "--length", "80", "--blame")
        self.assertEqual(code, 0, out)
        doc = json.loads(out)
        self.assertEqual(doc["modality"], "speech")
        self.assertEqual(len(doc["lessons"]), 1)
        self.assertIn("agreement", doc["lessons"][0]["grade"]["facts"])
        self.assertIn("mean_agreement", doc["report"])
        self.assertIn("taught", doc["negative"])

    def test_both_tutors_are_in_the_help(self):
        from radixnet.cli import build_parser

        text = build_parser().format_help()
        self.assertIn("speech", text)
        self.assertIn("image", text)


class TestApi(unittest.TestCase):
    """/api/speech/tutor and /api/images/tutor."""

    def setUp(self):
        from radixnet.api import ModelService

        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.svc = ModelService(
            model_path=os.path.join(self.dir.name, "m.count.json"), kind="count", backend="python", quiet=True,
        )
        self.addCleanup(lambda: self.svc.shutdown(timeout=5))
        self.text, self.taught = utterance()
        self.svc.model.train([self.text], epochs=3)

    def test_a_quiz_over_given_texts_is_marked(self):
        doc = self.svc.recall_quiz([self.text], ["a.wav"], "speech", length=80)
        self.assertEqual(doc["modality"], "speech")
        self.assertEqual(len(doc["lessons"]), 1)
        self.assertIsNone(doc["negative"])
        self.assertIn("mean_score", doc["report"])

    def test_blame_teaches_the_service_negative_network(self):
        doc = self.svc.recall_quiz([self.text], ["a.wav"], "speech", length=80, blame=True)
        self.assertIsNotNone(doc["negative"])
        self.assertIn("blamed", doc["negative"]["taught"])
        self.assertIs(self.svc.negative_model(), self.svc._parked["negative"])

    def test_nothing_to_ask_about_is_a_400(self):
        from radixnet.api import ApiError

        with self.assertRaises(ApiError) as caught:
            self.svc.recall_quiz([], [], "speech")
        self.assertEqual(caught.exception.status, 400)

    def test_the_routes_are_registered_and_documented(self):
        from radixnet.api import _ENDPOINTS, _ROUTES

        docs = {path: doc for method, path, _, doc in _ENDPOINTS if method == "POST"}
        for path in ("/api/speech/tutor", "/api/images/tutor"):
            self.assertIn("POST", _ROUTES.get(path, {}))
            self.assertIn("blame", docs[path])


class TestApiHttp(unittest.TestCase):
    """The two endpoints over real HTTP: the handlers, not just the service method."""

    def setUp(self):
        import threading

        from radixnet import api

        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.server, self.service = api.create_server(
            "127.0.0.1", 0, model_path=os.path.join(self.dir.name, "m.count.json"), kind="count",
            backend="python", quiet=True,
        )
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            thread.join(5)
            self.server.server_close()
            self.service.shutdown(timeout=10)

        self.addCleanup(stop)
        self.base = self.server.url.rstrip("/")
        self.text, _ = utterance()
        self.service.model.train([self.text], epochs=3)

    def post(self, path, body):
        import urllib.error
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with opener.open(req, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_speech_tutor_marks_given_texts(self):
        status, doc = self.post("/api/speech/tutor", {"texts": [self.text], "length": 80})
        self.assertEqual(status, 200, doc)
        self.assertEqual(doc["modality"], "speech")
        self.assertEqual(len(doc["lessons"]), 1)
        self.assertIsNone(doc["negative"])

    def test_speech_tutor_blames_what_it_forgot(self):
        status, doc = self.post("/api/speech/tutor", {"texts": [self.text], "length": 80, "blame": True})
        self.assertEqual(status, 200, doc)
        self.assertIsNotNone(doc["negative"])
        self.assertIn("reasons", doc["negative"])
        status, negative = self.post("/api/negative/judge", {"texts": [self.text]})
        self.assertEqual(status, 200, negative)

    def test_images_tutor_marks_given_texts(self):
        text, _ = picture()
        status, doc = self.post("/api/images/tutor", {"texts": [text]})
        self.assertEqual(status, 200, doc)
        self.assertEqual(doc["modality"], "image")
        self.assertEqual(len(doc["lessons"]), 1)

    def test_a_recording_is_encoded_and_quizzed_in_one_call(self):
        import base64 as b64

        wav = b64.b64encode(speech.wav_bytes(tone(), RATE)).decode("ascii")
        status, doc = self.post(
            "/api/speech/tutor",
            {"name": "a.wav", "content_base64": wav, "transcript": "hello", "rate": RATE, "length": 80},
        )
        self.assertEqual(status, 200, doc)
        self.assertEqual(len(doc["lessons"]), 1)
        self.assertIn("<speech:", doc["lessons"][0]["exercise"]["label"])

    def test_an_empty_request_is_a_400(self):
        status, doc = self.post("/api/speech/tutor", {"texts": []})
        self.assertEqual(status, 400, doc)

    def test_a_bad_option_is_a_400(self):
        status, doc = self.post("/api/speech/tutor", {"texts": [self.text], "attempts": "many"})
        self.assertEqual(status, 400, doc)

    def test_the_endpoints_are_listed_on_the_help_page(self):
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(self.base + "/", timeout=30) as resp:
            page = resp.read().decode("utf-8")
        self.assertIn("/api/speech/tutor", page)
        self.assertIn("/api/images/tutor", page)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
