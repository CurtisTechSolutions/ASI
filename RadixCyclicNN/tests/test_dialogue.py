"""Tests for radixnet.dialogue (a model conversing with itself)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.dialogue import DEFAULT_SPEAKERS, Turn, converse, normalize, tail_context, transcript  # noqa: E402
from radixnet.model import new_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "data", "sample_corpus.txt"), encoding="utf-8") as fh:
    CORPUS = [line for line in fh.read().splitlines() if line.strip()]

FAST = {"lr": 1.0, "batch_size": 1}


def picked_up(previous, context):
    """A reply's context is the tail of the previous line, or that tail minus trailing words (whole words)."""
    previous = previous.rstrip()
    if previous.endswith(context):
        return True
    words, wanted = previous.split(), context.split()
    return any(words[i:i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1))


def trained(kind="radix", epochs=2, seed=1):
    model = new_model(kind, seed=seed)
    model.train(CORPUS, epochs=epochs, **(FAST if kind == "radix" else {}))
    return model


class TestHelpers(unittest.TestCase):
    def test_tail_context_snaps_to_a_word_boundary(self):
        self.assertEqual(tail_context("the cat sat on the mat", 12), "on the mat")
        self.assertEqual(tail_context("the cat sat on the mat", 3), "mat")
        self.assertEqual(tail_context("the cat sat on the mat", 4), "mat")  # " mat" -> the word
        self.assertEqual(tail_context("supercalifragilistic", 8), "gilistic")  # no boundary inside the tail
        self.assertEqual(tail_context("short", 12), "short")
        self.assertEqual(tail_context("trailing spaces   ", 6), "spaces")
        self.assertEqual(tail_context("anything", 0), "")
        self.assertEqual(tail_context("", 12), "")
        self.assertEqual(tail_context("   ", 12), "")

    def test_normalize_and_transcript(self):
        self.assertEqual(normalize("  The   Cat "), "the cat")
        turns = [Turn(0, "A", "hello"), Turn(1, "B", "hello there")]
        self.assertEqual(transcript(turns), "A: hello\nB: hello there")
        self.assertEqual(transcript([]), "")
        d = turns[0].to_dict()
        self.assertEqual(d["speaker"], "A")
        self.assertEqual(set(d), {
            "index", "speaker", "text", "context", "reply", "cost", "probability", "reached_end", "fresh", "given",
            "repeat", "candidates", "skipped", "labels", "node_ids", "step_costs",
        })


class TestConverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = trained()

    def check_turns(self, turns, opening=None, history=()):
        said = list(history)
        for i, turn in enumerate(turns):
            self.assertEqual(turn.index, len(history) + i)
            self.assertEqual(turn.speaker, DEFAULT_SPEAKERS[turn.index % 2])
            self.assertTrue(turn.text.strip(), turn)
            if turn.given:
                self.assertEqual(turn.text, opening)
                self.assertTrue(turn.fresh)
            elif turn.fresh:
                self.assertEqual(turn.context, "")
                self.assertEqual(turn.reply, turn.text)
            else:
                # the reply picked up the tail of the previous line and carried on from it
                previous = said[-1]
                self.assertTrue(turn.context)
                self.assertTrue(picked_up(previous, turn.context), (previous, turn.context))
                self.assertEqual(turn.text, turn.context + turn.reply)
                self.assertTrue(turn.reply.strip())
            self.assertGreaterEqual(turn.candidates, 1)
            self.assertGreaterEqual(turn.cost, 0.0)
            self.assertTrue(0.0 <= turn.probability <= 1.0)
            said.append(turn.text)
        return said

    def test_beam_conversation_with_an_opening(self):
        opening = "the cat sat on the mat"
        turns = converse(self.model, opening, turns=6)
        self.assertEqual(len(turns), 7)
        self.assertTrue(turns[0].given)
        self.assertEqual((turns[0].speaker, turns[0].text, turns[0].context), ("A", opening, ""))
        self.assertGreater(turns[0].cost, 0.0)
        self.assertFalse(any(t.given for t in turns[1:]))
        said = self.check_turns(turns, opening)
        # the conversation never repeats itself and no reply merely echoes the line before
        keys = [normalize(t) for t in said]
        self.assertEqual(len(keys), len(set(keys)))
        for before, after in zip(turns, turns[1:]):
            self.assertNotIn(normalize(after.text), normalize(before.text))
        self.assertFalse(any(t.repeat for t in turns))
        # deterministic
        again = converse(self.model, opening, turns=6)
        self.assertEqual([t.text for t in again], [t.text for t in turns])
        # every utterance is made of transitions the structure knows (a reply picked up mid-text may only lack
        # the edge out of START)
        for turn in turns[1:]:
            self.assertLessEqual(self.model.score(turn.text)["unknown_transitions"], 0 if turn.fresh else 1, turn.text)

    def test_replies_pick_up_the_previous_line(self):
        turns = converse(self.model, "the quick brown fox", turns=4, context=20)
        picked = [t for t in turns[1:] if not t.fresh]
        self.assertTrue(picked, [t.text for t in turns])
        for turn in picked:
            self.assertLessEqual(len(turn.context), 20)
            self.assertTrue(turn.text.startswith(turn.context))
        # the best continuation of the context is what a prediction would give (beam, first unheard candidate)
        first = turns[1]
        if not first.fresh:
            found = self.model.predict(first.context, length=0, mode="beam", k=5, to_end=True, max_length=60)
            self.assertIn(first.text, [r.full_text for r in found.top])

    def test_without_an_opening_the_first_voice_starts_fresh(self):
        turns = converse(self.model, turns=4)
        self.assertEqual(len(turns), 4)
        self.assertTrue(turns[0].fresh)
        self.assertFalse(turns[0].given)
        self.assertEqual(turns[0].speaker, "A")
        self.assertEqual(turns[0].text, self.model.generate(mode="beam", count=1, max_length=60)[0].text)
        self.check_turns(turns)

    def test_history_continues_a_conversation(self):
        history = ["the cat sat on the mat", "on the west"]
        turns = converse(self.model, turns=3, history=history)
        self.assertEqual([t.index for t in turns], [2, 3, 4])
        self.assertEqual([t.speaker for t in turns], ["A", "B", "A"])
        self.assertFalse(any(t.given for t in turns))
        said = self.check_turns(turns, history=history)
        self.assertEqual(len({normalize(t) for t in said}), len(said))
        # an opening on top of a history is spoken next, by the voice whose turn it is
        turns = converse(self.model, "hello there", turns=1, history=history)
        self.assertEqual((turns[0].index, turns[0].speaker, turns[0].given), (2, "A", True))
        self.assertEqual((turns[1].index, turns[1].speaker), (3, "B"))

    def test_sample_mode_is_reproducible_with_a_seed(self):
        a = converse(self.model, "the cat", turns=4, mode="sample", seed=7)
        b = converse(self.model, "the cat", turns=4, mode="sample", seed=7)
        self.assertEqual([t.text for t in a], [t.text for t in b])
        self.assertEqual(len(a), 5)
        self.check_turns(a, "the cat")
        greedy = converse(self.model, turns=2, mode="sample", temperature=0.0, seed=1)
        self.assertEqual(len(greedy), 2)

    def test_speakers_partner_and_options(self):
        turns = converse(self.model, "the cat", turns=3, speakers=("cat", "dog"))
        self.assertEqual([t.speaker for t in turns], ["cat", "dog", "cat", "dog"])
        three = converse(self.model, turns=3, speakers=("x", "y", "z"))
        self.assertEqual([t.speaker for t in three], ["x", "y", "z"])
        partner = trained("count")
        turns = converse(self.model, "the cat", turns=4, partner=partner, speakers=("radix", "count"))
        self.assertEqual(len(turns), 5)
        for turn in turns[1:]:
            voice = self.model if turn.index % 2 == 0 else partner
            self.assertLessEqual(voice.score(turn.text)["unknown_transitions"], 0 if turn.fresh else 1, (turn.speaker, turn.text))
        # dijkstra is accepted as an alias of beam; the model method forwards everything
        self.assertEqual(
            [t.text for t in converse(self.model, "the cat", turns=2, mode="dijkstra")],
            [t.text for t in self.model.converse("the cat", turns=2, mode="beam")],
        )
        self.assertEqual(converse(self.model, "the cat", turns=0), [t for t in converse(self.model, "the cat", turns=0)])
        self.assertEqual(len(converse(self.model, "the cat", turns=0)), 1)
        self.assertEqual(converse(self.model, turns=0), [])
        short = converse(self.model, "the cat sat on the mat", turns=3, max_length=5)
        for turn in short[1:]:
            self.assertLessEqual(len(turn.reply), 5)

    def test_repeats_are_allowed_on_request(self):
        strict = converse(self.model, "the cat", turns=8)
        loose = converse(self.model, "the cat", turns=8, avoid_repeats=False)
        self.assertEqual(len({normalize(t.text) for t in strict}), len(strict))
        self.assertEqual(len(loose), 9)
        self.assertFalse(any(t.repeat for t in loose))  # nothing is *marked* a repeat when repeats are fine

    def test_empty_model_has_nothing_to_say(self):
        empty = new_model("radix")
        self.assertEqual(converse(empty, turns=3), [])
        turns = converse(empty, "hello", turns=3)
        self.assertEqual(len(turns), 1)
        self.assertTrue(turns[0].given)

    def test_validation(self):
        with self.assertRaises(TypeError):
            converse(self.model, None)
        with self.assertRaises(TypeError):
            converse(self.model, history=[1, 2])
        with self.assertRaises(ValueError):
            converse(self.model, mode="nope")
        for kwargs in ({"turns": -1}, {"max_length": -1}, {"context": -1}, {"k": 0}, {"beam": 0},
                       {"temperature": -1.0}, {"step_penalty": -1.0}, {"speakers": ()}, {"speakers": ("", "B")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                converse(self.model, **kwargs)


if __name__ == "__main__":
    unittest.main()
