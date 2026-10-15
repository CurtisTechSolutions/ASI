"""Auto-temperature (``../radixnet/autotemp.py``, ``../SPEC-SearchAndTraining.md`` §10): a search that finds no
answer walks again up a ladder of temperatures.

The ladder's rungs and their draws, what counts as an answer, the record a climb
leaves, and the two searches every kind has: a greedy walk that goes round and
is mended by a warmer one, a deterministic search that falls back to walks, a
generation whose silent samples are drawn again - and, off, a search that is
exactly the one it always was, draw for draw.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import autotemp  # noqa: E402
from radixnet.autotemp import (  # noqa: E402
    AUTO_DRAWS, AUTO_FLOOR, AutoTemperature, answered, check_auto_temperature, climb, first_answer,
    temperature_ladder,
)
from radixnet.encoding import Encoding  # noqa: E402
from radixnet.model import new_model  # noqa: E402
from radixnet.search import PathResult  # noqa: E402

# a strong cycle - "ab " comes round again far more often than it ends - so a greedy walk goes round for ever
LOOP = ["ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab end"] * 3 + ["ab end"]
# a text that ends as often as it goes on, so a walk from its end says nothing half the time
ENDS = ["the cat sat on the mat", "the cat sat on the mat and slept"]
MAT = "the cat sat on the mat"


class Scripted:
    """A model whose generations are scripted: every call hands back the next list of texts (at most the ``count``
    asked for), and what it was asked is kept in ``asked``."""

    def __init__(self, *calls: list[str]):
        self.calls = [list(texts) for texts in calls]
        self.asked: list[dict] = []
        self.encoding = Encoding()

    def generate(self, count=1, **options):
        self.asked.append({"count": count, **options})
        texts = self.calls.pop(0) if self.calls else []
        return [PathResult(text=text, full_text=text) for text in texts][:count]


class TestTheLadder(unittest.TestCase):
    def test_rungs_double_up_to_the_ceiling(self):
        self.assertEqual(temperature_ladder(1, 2), [1.0, 2.0])
        self.assertEqual(temperature_ladder(1, 3), [1.0, 2.0, 3.0])
        self.assertEqual(temperature_ladder(1, 1.5), [1.0, 1.5])
        self.assertEqual(temperature_ladder(0.7, 2), [0.7, 1.4, 2.0])
        self.assertEqual(temperature_ladder(0.25, 1), [0.25, 0.5, 1.0])

    def test_greedy_climbs_to_the_floor_first(self):
        self.assertEqual(AUTO_FLOOR, 0.5)
        self.assertEqual(temperature_ladder(0, 2), [0.0, 0.5, 1.0, 2.0])
        self.assertEqual(temperature_ladder(0, 0.3), [0.0, 0.3])

    def test_it_never_cools_and_off_is_no_ladder(self):
        self.assertEqual(temperature_ladder(3, 2), [3.0])
        self.assertEqual(temperature_ladder(2, 2), [2.0])
        self.assertEqual(temperature_ladder(1, 0), [])

    def test_a_ceiling_out_of_range_is_refused(self):
        check_auto_temperature(0)
        check_auto_temperature(2.5)
        for bad in (-1, float("nan"), float("inf"), "2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_auto_temperature(bad)


class TestAnAnswer(unittest.TestCase):
    def test_it_wrote_something(self):
        self.assertTrue(answered(PathResult(text=" sat", reached_end=False)))
        self.assertFalse(answered(PathResult(text="", reached_end=True)))
        self.assertFalse(answered(PathResult(text="  \n", reached_end=True)))

    def test_and_reached_the_end_when_the_end_was_asked_for(self):
        self.assertTrue(answered(PathResult(text=" sat", reached_end=True), to_end=True))
        self.assertFalse(answered(PathResult(text=" sat", reached_end=False), to_end=True))


class TestTheClimb(unittest.TestCase):
    def searches(self, answers_at=None):
        """A search that answers only at the temperatures in ``answers_at`` (as sampled walks), and its log."""
        log = []

        def search(t):
            log.append(("as asked", None) if t is None else ("sample", t))
            ok = answers_at is not None and t is not None and t in answers_at
            return PathResult(text="said" if ok else "", reached_end=True)

        return search, log

    def test_an_answer_as_asked_is_the_whole_of_it(self):
        search, log = self.searches(set())

        def asked_answers(t):
            log.append(("as asked", None) if t is None else ("sample", t))
            return PathResult(text="said", reached_end=True)

        result, record = climb(asked_answers, first_answer, temperature=1.0, ceiling=2.0, sampled=True)
        self.assertEqual(result.text, "said")
        self.assertEqual(log, [("as asked", None)])
        self.assertEqual(record.to_dict(), {"ceiling": 2.0, "climbed": False, "found": True, "temperature": 1.0,
                                            "tries": 1})

    def test_every_rung_is_three_draws_and_the_first_answer_ends_it(self):
        self.assertEqual(AUTO_DRAWS, 3)
        search, log = self.searches({2.0})
        result, record = climb(search, first_answer, temperature=1.0, ceiling=2.0, sampled=False)
        self.assertEqual(log, [("as asked", None)] + [("sample", 1.0)] * 3 + [("sample", 2.0)])
        self.assertEqual(result.text, "said")
        self.assertEqual((record.climbed, record.found, record.temperature, record.tries), (True, True, 2.0, 5))

    def test_a_greedy_rung_is_one_walk(self):
        search, log = self.searches(set())
        climb(search, first_answer, temperature=0.0, ceiling=1.0, sampled=True)
        self.assertEqual(log, [("as asked", None), ("sample", 0.0)] + [("sample", 0.5)] * 3 + [("sample", 1.0)] * 3)

    def test_nothing_found_gives_back_the_search_as_asked(self):
        first = PathResult(text="", labels=["as asked"])
        calls = []

        def search(t):
            calls.append(t)
            return first if len(calls) == 1 else PathResult(text="")

        result, record = climb(search, first_answer, temperature=1.0, ceiling=2.0, sampled=False)
        self.assertIs(result, first)
        self.assertEqual(record.to_dict(), {"ceiling": 2.0, "climbed": True, "found": False, "temperature": None,
                                            "tries": 7})

    def test_any_test_is_an_answer_test(self):
        """A caller's own test - a guard's veto - decides as well as the model's."""
        texts = iter(["vetoed", "vetoed", "clean"])
        result, record = climb(lambda t: PathResult(text=next(texts)),
                               lambda r: r if answered(r) and r.text != "vetoed" else None,
                               temperature=1.0, ceiling=2.0, sampled=True)
        self.assertEqual((result.text, record.tries, record.temperature), ("clean", 3, 1.0))


class TestPredict(unittest.TestCase):
    def test_off_is_the_search_it_always_was(self):
        model = new_model("count", seed=1)
        model.train(LOOP, epochs=2)
        before = model.graph.rng.getstate()
        plain = model.predict("ab", mode="sample", length=12)
        after = model.graph.rng.getstate()
        model.graph.rng.setstate(before)
        result, record = autotemp.predict(model, "ab", mode="sample", length=12)
        self.assertIsNone(record)
        self.assertEqual((result.text, result.node_ids), (plain.text, plain.node_ids))
        self.assertEqual(model.graph.rng.getstate(), after, "the same draws, no more")

    def test_a_greedy_walk_that_goes_round_is_mended_by_a_warmer_one(self):
        model = new_model("count", seed=1)
        model.train(LOOP, epochs=2)
        cold = model.predict("ab", mode="sample", temperature=0, length=20, max_length=30, to_end=True)
        self.assertFalse(cold.reached_end, "greedy goes round and never ends")
        result, record = autotemp.predict(model, "ab", mode="sample", temperature=0, length=20, max_length=30,
                                          to_end=True, auto_temperature=2.0)
        self.assertTrue(result.reached_end)
        self.assertTrue(result.text.strip())
        self.assertTrue(record.climbed and record.found)
        self.assertGreater(record.temperature, 0)
        self.assertGreater(record.tries, 1)

    def test_a_deterministic_search_that_finds_nothing_falls_back_to_walks(self):
        """The beam reads no temperature: when it finds nothing, the ladder is where sampling starts - and the
        walks are held to the same cap, so an end no path reaches inside it is not reached by walking either."""
        model = new_model("count", seed=1)
        model.train(LOOP, epochs=2)
        beam = model.predict("ab", mode="beam", length=2, max_length=3, to_end=True)  # " end" needs 4
        self.assertFalse(answered(beam, to_end=True))
        result, record = autotemp.predict(model, "ab", mode="beam", length=2, max_length=3, to_end=True,
                                          auto_temperature=2.0)
        self.assertEqual(record.to_dict(), {"ceiling": 2.0, "climbed": True, "found": False, "temperature": None,
                                            "tries": 1 + 2 * AUTO_DRAWS})
        self.assertEqual((result.text, result.node_ids), (beam.text, beam.node_ids), "the search as asked")

    def test_an_answer_lower_in_the_beam_is_the_beams_own(self):
        """With room to end, the beam's cheapest path still goes round to the cap - but its top K hold paths that
        end, and the first of them is the answer: found as asked, nothing walked."""
        model = new_model("count", seed=1)
        model.train(LOOP, epochs=2)
        beam = model.predict("ab", mode="beam", length=2, max_length=30, to_end=True)
        self.assertFalse(beam.reached_end, "the cheapest path is the cycle, cut at the cap")
        ending = next(p for p in beam.top if p.reached_end)
        result, record = autotemp.predict(model, "ab", mode="beam", length=2, max_length=30, to_end=True,
                                          auto_temperature=2.0)
        self.assertEqual((result.text, result.node_ids, result.reached_end), (ending.text, ending.node_ids, True))
        self.assertEqual([p.text for p in result.top], [p.text for p in beam.top], "the ranking is the search's")
        self.assertEqual((record.climbed, record.found, record.tries, record.temperature), (False, True, 1, None))

    def test_no_way_on_is_no_answer_at_any_temperature(self):
        model = new_model("count", seed=1)
        model.train(["the cat sat"], epochs=1)
        result, record = autotemp.predict(model, "the cat sat", mode="sample", length=5,
                                          auto_temperature=2.0)
        self.assertEqual(result.text, "")
        self.assertEqual(record.to_dict(), {"ceiling": 2.0, "climbed": True, "found": False, "temperature": None,
                                            "tries": 1 + 2 * AUTO_DRAWS})

    def test_every_kind_climbs(self):
        for kind in ("radix", "count", "resonant", "negative"):
            with self.subTest(kind=kind):
                model = new_model(kind, seed=1)
                model.train(LOOP, epochs=2)
                result, record = autotemp.predict(model, "ab", mode="sample", temperature=0, length=20,
                                                  max_length=30, to_end=True, auto_temperature=2.0)
                self.assertIsInstance(record, AutoTemperature)
                self.assertEqual(record.found, answered(result, to_end=True))


class TestGenerate(unittest.TestCase):
    def test_off_is_the_generation_it_always_was(self):
        model = new_model("count", seed=1)
        model.train(ENDS, epochs=2)
        plain = model.generate(count=6, mode="sample", prefix="the cat sat on the mat", seed=4)
        results, record = autotemp.generate(model, count=6, mode="sample", prefix="the cat sat on the mat", seed=4)
        self.assertIsNone(record)
        self.assertEqual([r.text for r in results], [r.text for r in plain])

    def test_samples_that_said_nothing_are_drawn_again(self):
        model = new_model("count", seed=1)
        model.train(ENDS, epochs=2)
        prefix = "the cat sat on the mat"
        plain = model.generate(count=6, mode="sample", prefix=prefix, seed=4)
        silent = sum(not autotemp.wrote(model, r, prefix) for r in plain)
        self.assertGreater(silent, 0, "some walks end where the prefix does")
        results, record = autotemp.generate(model, count=6, mode="sample", prefix=prefix, seed=4,
                                            auto_temperature=2.0)
        self.assertEqual(len(results), 6)
        self.assertTrue(all(r.text.startswith(prefix) and len(r.text) > len(prefix) for r in results))
        self.assertTrue(record.climbed and record.found)
        # the samples that said something come first, in their order
        said = [r.text for r in plain if autotemp.wrote(model, r, prefix)]
        self.assertEqual([r.text for r in results[:len(said)]], said)
        # and the whole climb is reproducible from the seed
        again, _ = autotemp.generate(model, count=6, mode="sample", prefix=prefix, seed=4, auto_temperature=2.0)
        self.assertEqual([r.text for r in again], [r.text for r in results])

    def test_too_few_at_the_ceiling_is_what_there_is(self):
        model = new_model("count", seed=1)
        model.train(["the cat sat"], epochs=1)
        results, record = autotemp.generate(model, count=3, mode="sample", prefix="the cat sat",
                                            auto_temperature=1.0)
        self.assertEqual(results, [])
        self.assertFalse(record.found)
        self.assertTrue(math.isclose(record.ceiling, 1.0))

    def test_what_a_generation_gives_at_most(self):
        self.assertEqual([autotemp.wanted(n, "dijkstra") for n in (0, 1, 3)], [0, 1, 1])
        self.assertEqual([autotemp.wanted(3, mode) for mode in ("beam", "kbest", "sample", None, "DIJKSTRA")],
                         [3, 3, 3, 3, 1])
        self.assertEqual(autotemp.generation_mode(new_model("count", seed=1)), "sample")
        self.assertEqual(autotemp.generation_mode(new_model("resonant", seed=1)), "kbest", "the k-best, exactly")
        self.assertEqual(autotemp.generation_mode(new_model("resonant", seed=1), "Beam"), "beam")

    def test_dijkstra_is_one_text_however_many_are_asked_for(self):
        """Dijkstra's generation is the single cheapest text, so that one text is the whole of its answer: it is
        walked again only when it says nothing, and then for one."""
        model = new_model("count", seed=1)
        model.train(ENDS, epochs=2)
        plain = model.generate(count=3, mode="dijkstra", max_length=40)
        self.assertEqual(len(plain), 1)
        results, record = autotemp.generate(model, count=3, mode="dijkstra", max_length=40, auto_temperature=2.0)
        self.assertEqual([r.text for r in results], [r.text for r in plain])
        self.assertEqual(record.to_dict(), {"ceiling": 2.0, "climbed": False, "found": True, "temperature": None,
                                            "tries": 1})
        # from the end of a text the cheapest is to say nothing more
        self.assertEqual([r.text for r in model.generate(count=3, mode="dijkstra", prefix=MAT, max_length=40)], [MAT])
        results, record = autotemp.generate(model, count=3, mode="dijkstra", prefix=MAT, max_length=40, seed=3,
                                            auto_temperature=2.0)
        self.assertEqual(len(results), 1)
        self.assertTrue(autotemp.wrote(model, results[0], MAT))
        self.assertTrue(record.climbed and record.found)

    def test_a_walk_keeps_to_the_generation_it_fills(self):
        """The beam's texts are distinct, so a walk that repeats one already kept is not kept again; sampled walks
        may repeat each other, as they always may."""
        script = (["ab", "ab cd", "ab ef"], ["ab cd"], ["ab gh"])  # the beam's first says nothing beyond "ab"
        beam = Scripted(*script)
        results, record = autotemp.generate(beam, count=3, mode="beam", prefix="ab", auto_temperature=2.0)
        self.assertEqual([r.text for r in results], ["ab cd", "ab ef", "ab gh"])
        self.assertEqual((record.climbed, record.found, record.temperature, record.tries), (True, True, 1.0, 3))
        self.assertEqual([(a["count"], a["mode"]) for a in beam.asked], [(3, "beam"), (1, "sample"), (1, "sample")])
        sampled = Scripted(*script)
        results, record = autotemp.generate(sampled, count=3, mode="sample", prefix="ab", auto_temperature=2.0)
        self.assertEqual([r.text for r in results], ["ab cd", "ab ef", "ab cd"])
        self.assertEqual((record.found, record.temperature, record.tries), (True, 1.0, 2))

    def test_a_beams_texts_stay_distinct(self):
        model = new_model("count", seed=1)
        model.train(ENDS, epochs=2)
        beam = [r.text for r in model.generate(count=3, mode="beam", prefix=MAT, max_length=40)]
        self.assertEqual(beam[0], MAT, "the cheapest whole text says nothing more")
        found = 0
        for seed in range(12):
            with self.subTest(seed=seed):
                results, record = autotemp.generate(model, count=3, mode="beam", prefix=MAT, max_length=40,
                                                    seed=seed, auto_temperature=2.0)
                texts = [r.text for r in results]
                self.assertEqual(len(set(texts)), len(texts))
                self.assertEqual(texts[:2], beam[1:], "the beam's own come first")
                self.assertTrue(record.climbed)
                self.assertEqual(record.found, len(texts) == 3)
                found += record.found
        self.assertGreater(found, 0, "a hotter walk finds a third")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
