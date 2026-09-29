"""The teaching loops of a model of sounds: the LLM reads English, the model learns from its own sounds.

A model of phones or syllables writes ``DH.AH0 # K.AE1.T``.  Every loop that
shows what it wrote to an LLM - the English tutor, the reviewer, the copy
editor, the chat partner and its judge, the Voice tab's Ollama - shows the
words those sounds spell instead (:func:`modelkit.llm.reader_text`),
while what the model is rewarded, punished and blamed for stays the sounds it
said, and what the LLM writes back is read as the sounds it makes.  Each test
here holds one loop to that with a scripted LLM that records every prompt.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from modelkit import blame  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from modelkit.llm import reader_text
from radixnet.encoding import SYLLABLES, Encoding, phonetok_module  # noqa: E402

try:
    phonetok_module()
    PHONETOK = True
except ValueError:  # pragma: no cover
    PHONETOK = False

CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the mat",
    "the dog ran to the park",
    "a bird sang in the tree",
]

SOUNDS = re.compile(r"\b[A-Z]{1,2}[012]\b|(?:^|\s)#(?:\s|$)|[A-Z]+\.[A-Z]")
"""A stressed vowel (``AE1``), a word boundary (``#``) or a syllable (``K.AE1``): what English never holds."""


def sounds_model(seed=1):
    net = CountRewardNet(seed=seed, encoding=Encoding(unit=SYLLABLES, n=2))
    net.train(CORPUS, epochs=2)
    return net


def _indexed(prompt):
    """``[(index, text)]`` of a prompt's ``[i] text`` lines."""
    out = []
    for line in prompt.splitlines():
        if line.startswith("[") and "]" in line:
            out.append((int(line[1 : line.index("]")]), line[line.index("]") + 2 :]))
    return out


class Teacher:
    """A scripted LLM: every request is answered from a script, and every prompt is recorded."""

    provider, url, model = "ollama", "http://scripted", "scripted"

    def __init__(self, fix=lambda text: text.rsplit(" ", 1)[0] + " mat" if " " in text else text):
        self.prompts = []
        self.fix = fix

    def generate(self, prompt, *, system=None, model=None, json_mode=False, options=None, timeout=None, **_):
        self.prompts.append(prompt)
        if "exercises now" in prompt:
            return json.dumps({"exercises": [
                {"prefix": "The cat", "answer": "The cat sat on the mat.", "focus": "past tense"},
                {"prefix": "The dog", "answer": "The dog sat on the mat.", "focus": "past tense"},
            ]})
        if "Mark these" in prompt:
            return json.dumps({"grades": [
                {"index": i, "grammar": 3, "spelling": 5, "fluency": 3, "error": "tense",
                 "correction": "The cat sits on the mat.", "comment": "Use the present tense."}
                for i, _ in _indexed(prompt)
            ]})
        if "Explain these" in prompt:
            return json.dumps({"mistakes": [
                {"index": i, "why": "the tense is wrong", "again": [{"wrong": "the dog sit", "right": "the dog sits"}]}
                for i, _ in _indexed(prompt)
            ]})
        if "Review these" in prompt:
            return json.dumps({"reviews": [{"index": i, "rating": 2, "critique": "it repeats itself"}
                                           for i, _ in _indexed(prompt)]})
        if "Correct these" in prompt:
            return json.dumps({"corrections": [
                {"index": i, "correction": self.fix(text), "reason": "vocabulary", "note": "the last word"}
                for i, text in _indexed(prompt)
            ]})
        if "The conversation:" in prompt:
            return json.dumps({"reviews": [{"index": i, "rating": 2, "critique": "it does not follow on"}
                                           for i, _ in _indexed(prompt)],
                               "overall": {"rating": 3, "critique": "short"}})
        return "the dog sat on the mat"  # a line of a conversation: the partner's, or Ollama's for the Voice tab

    def chat(self, messages, **options):
        return ""

    def models(self):
        return []

    def available(self):
        return True

    def assert_words_only(self, case):
        """No prompt the LLM was sent holds a sound."""
        case.assertTrue(self.prompts)
        for prompt in self.prompts:
            case.assertIsNone(SOUNDS.search(prompt), prompt)


@unittest.skipUnless(PHONETOK, "the phonetic tokenizer is not importable")
class TestTheTutor(unittest.TestCase):
    def test_the_teacher_marks_words_and_the_model_learns_its_sounds(self):
        from radixnet.negative import NegativeNet
        from modelkit.tutor import TutorConfig, TutorTrainer

        model = sounds_model()
        negative = NegativeNet(seed=1, encoding=model.encoding)
        teacher = Teacher()
        trainer = TutorTrainer(
            model, teacher, TutorConfig(topic="animals", exercises=2, attempts=1, variants=1), negative=negative,
        )
        record, lessons = trainer.run_round(1)
        teacher.assert_words_only(self)
        self.assertEqual(len(lessons), 2)
        for lesson in lessons:
            # the teacher's prefix as it wrote it, then the words the sounds spell
            self.assertTrue(lesson.sentence.startswith(lesson.exercise.cue), lesson.sentence)
            self.assertIsNone(SOUNDS.search(lesson.sentence), lesson.sentence)
            self.assertEqual(lesson.prefix, lesson.exercise.cue)
            # ... and what the model said, in its own units
            self.assertTrue(lesson.said.startswith("DH.AH0 # "), lesson.said)
            self.assertEqual(lesson.own, lesson.said)
            self.assertEqual(lesson.to_dict()["said"], lesson.said)
            self.assertEqual(model.encoding.spell(lesson.said).lower(), lesson.sentence.lower().strip())
        # the grades asked for the words, not the sounds
        marked = [p for p in teacher.prompts if "Mark these" in p][0]
        self.assertIn("<<The cat >>", marked)
        # a correction aligns the sounds the model said with the teacher's words
        corrections = trainer.corrections_of(lessons)
        self.assertEqual([c.wrong for c in corrections], [lesson.said for lesson in lessons])
        self.assertEqual({c.right for c in corrections}, {"The cat sits on the mat."})
        self.assertGreater(record["corrections"], 0)
        self.assertGreater(record["edits"], 0)
        # the negative network is blamed for the sounds, and learns the correction as the sounds it makes
        faults, cleared = blame.faults_from_lessons(lessons)
        self.assertEqual([f["text"] for f in faults if f["source"] == "tutor"], [lesson.said for lesson in lessons])
        self.assertTrue(all(f["correction"] == "The cat sits on the mat." for f in faults if f["source"] == "tutor"))
        self.assertIn("The cat sits on the mat.", cleared)
        self.assertGreater(record["negative_blamed"], 0)
        self.assertGreater(record["negative_edges"], 0)

    def test_a_dry_run_counts_edits_in_the_model_s_units(self):
        from radixnet import diff
        from modelkit.tutor import TutorConfig, TutorTrainer

        model = sounds_model()
        trainer = TutorTrainer(model, Teacher(), TutorConfig(topic="animals", exercises=2, attempts=1, learn=False))
        record, lessons = trainer.run_round(1)
        expected = sum(
            len(diff.summary(c.wrong, c.right, limit=0, encoding=model.encoding))
            for c in trainer.corrections_of(lessons)
        )
        self.assertEqual(record["edits"], expected)
        self.assertGreater(expected, 0)

    def test_a_model_of_letters_is_marked_as_it_writes(self):
        from modelkit.tutor import TutorConfig, TutorTrainer

        model = CountRewardNet(seed=1)
        model.train(CORPUS, epochs=2)
        trainer = TutorTrainer(model, Teacher(), TutorConfig(topic="animals", exercises=2, attempts=1))
        _, lessons = trainer.run_round(1)
        for lesson in lessons:
            self.assertEqual(lesson.said, "")
            self.assertNotIn("said", lesson.to_dict())
            self.assertEqual(lesson.sentence, lesson.exercise.cue + lesson.continuation)
            self.assertEqual(lesson.own, lesson.sentence)
            self.assertEqual(lesson.prefix, lesson.exercise.cue)


@unittest.skipUnless(PHONETOK, "the phonetic tokenizer is not importable")
class TestTheReviewerAndTheEditor(unittest.TestCase):
    def setUp(self):
        self.model = sounds_model()
        self.enc = self.model.encoding
        self.sounds = [r.text for r in self.model.generate(count=3, mode="beam", prefix="the", max_length=20)]
        self.assertTrue(all(s.startswith("DH.AH0") for s in self.sounds), self.sounds)

    def test_the_reviewer_reads_words(self):
        from modelkit.ollama import review_texts

        teacher = Teacher()
        reviews = review_texts(teacher, self.sounds, encoding=self.enc)
        teacher.assert_words_only(self)
        self.assertEqual([r["text"] for r in reviews], self.sounds)
        self.assertEqual([r["spelled"] for r in reviews], [self.enc.spell(s) for s in self.sounds])
        faults, _ = blame.faults_from_reviews(reviews)
        self.assertEqual([f["text"] for f in faults], self.sounds)  # blamed for what it said

    def test_the_editor_corrects_words_and_the_sounds_are_blamed(self):
        from radixnet.negative import NegativeNet
        from modelkit.ollama import correct_texts

        teacher = Teacher()
        corrections = correct_texts(teacher, self.sounds, encoding=self.enc)
        teacher.assert_words_only(self)
        for entry, sounds in zip(corrections, self.sounds):
            words = self.enc.spell(sounds)
            self.assertEqual((entry["text"], entry["spelled"]), (sounds, words))
            self.assertEqual(entry["correction"], teacher.fix(words))
            # the diff is the editor's, over the words it read
            self.assertEqual(entry["verdict"], "corrected" if teacher.fix(words) != words else "unchanged")
            for change in entry["changes"]:
                self.assertIsNone(SOUNDS.search(change["wrong"] + " " + change["right"]), change)
        faults, _ = blame.faults_from_corrections(corrections)
        self.assertTrue(faults)
        self.assertTrue(all(f["text"] in self.sounds for f in faults))
        negative = NegativeNet(seed=1, encoding=self.enc)
        taught = blame.teach_corrections(negative, corrections)
        self.assertGreater(taught["edges"], 0)  # the correction, read as sounds, lined up with the sounds said

    def test_an_editor_that_changes_nothing_clears_the_sounds(self):
        from modelkit.ollama import correct_texts

        corrections = correct_texts(Teacher(fix=lambda text: text), self.sounds, encoding=self.enc)
        self.assertEqual({c["verdict"] for c in corrections}, {"unchanged"})
        faults, unchanged = blame.faults_from_corrections(corrections)
        self.assertEqual((faults, unchanged), ([], self.sounds))

    def test_the_critic_reads_words(self):
        from modelkit import critic
        from radixnet.negative import NegativeNet

        for correct in (False, True):
            with self.subTest(correct=correct):
                teacher = Teacher()
                loop = critic.Critic(self.model, NegativeNet(seed=1, encoding=self.enc), teacher,
                                     critic.CriticConfig(count=3, correct=correct, prefix="the"))
                record = loop.run_round()
                teacher.assert_words_only(self)
                self.assertGreater(record["blamed"], 0)

    def test_the_adversarial_calls_read_the_model_s_samples_as_words(self):
        from modelkit.ollama import adversarial_correction, adversarial_review

        teacher = Teacher()
        review = adversarial_review(self.model, teacher, count=2, prefix="the", max_length=20)
        corrected = adversarial_correction(self.model, teacher, count=2, prefix="the", max_length=20)
        teacher.assert_words_only(self)
        self.assertTrue(all("spelled" in r for r in review["reviews"]))
        self.assertTrue(all("spelled" in c for c in corrected["corrections"]))
        # texts given by hand are read as they were given, unless an encoding is named
        given = adversarial_review(None, teacher, texts=["The cat sat."])
        self.assertNotIn("spelled", given["reviews"][0])

    def test_a_model_of_letters_is_read_as_it_writes(self):
        from modelkit.ollama import correct_texts, review_texts

        self.assertEqual(reader_text(Encoding(), "the cat"), "the cat")
        self.assertEqual(reader_text(None, "DH.AH0"), "DH.AH0")
        for entry in review_texts(Teacher(), ["the cat sat"], encoding=Encoding()):
            self.assertNotIn("spelled", entry)
        for entry in correct_texts(Teacher(), ["the cat sat"], encoding=Encoding()):
            self.assertNotIn("spelled", entry)
            self.assertEqual(entry["correction"], "the cat mat")


@unittest.skipUnless(PHONETOK, "the phonetic tokenizer is not importable")
class TestTheConversations(unittest.TestCase):
    def test_the_partner_and_the_judge_hear_words(self):
        from modelkit.chat import DEFAULT_SPEAKERS, Chat, ChatConfig
        from radixnet.negative import NegativeNet

        model = sounds_model()
        teacher = Teacher()
        exchanges = []
        loop = Chat(model, teacher, ChatConfig(turns=2, opening="the cat", guard=False),
                    negative=NegativeNet(seed=1, encoding=model.encoding))
        record = loop.run_conversation(progress=exchanges.append)
        teacher.assert_words_only(self)
        said = [line for line in record["transcript"] if line["speaker"] == DEFAULT_SPEAKERS[1]]
        self.assertTrue(said)
        for line in said:
            self.assertIsNotNone(SOUNDS.search(line["text"]), line)  # it said sounds
            self.assertEqual(line["spelled"], model.encoding.spell(line["text"]))
        for line in record["transcript"]:
            if line["speaker"] == DEFAULT_SPEAKERS[0]:
                self.assertNotIn("spelled", line)
        self.assertTrue(all(e["spelled"] == model.encoding.spell(e["reply"]) for e in exchanges if e["kind"] == "exchange"))
        # the judge marked the words, and the model is punished and blamed for its sounds
        self.assertTrue(all(r["spelled"] == model.encoding.spell(r["text"]) for r in record["reviews"]))
        self.assertGreater(record["blamed"], 0)
        self.assertIn(record["action"], ("punish", "2nrl"))

    def test_the_voice_tab_s_ollama_hears_words(self):
        from modelkit.voicechat import VoiceOptions, turn

        teacher = Teacher()
        out = turn(
            None, encoding=Encoding(unit=SYLLABLES, n=2),
            options=VoiceOptions(transcript="how are you", answer="ollama", train=False, speak=False),
            history=[["You", "hello"], ["Model", "DH.AH0 # K.AE1.T # S.AE1.T"]], ollama=teacher,
        )
        teacher.assert_words_only(self)
        self.assertIn("Model: the cat sat", teacher.prompts[-1])
        self.assertEqual(out.document["by"], "ollama")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
