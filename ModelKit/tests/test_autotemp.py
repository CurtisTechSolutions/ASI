"""Auto-temperature through the kit: the guard's veto counts as no answer, and every output path can climb.

``../RadixCyclicNN/tests/test_autotemp.py`` holds the ladder itself; this holds
what the kit does with it.  A prediction the guard leaves without an answer
walks again up the ladder (:func:`modelkit.duo.guarded_predict`), a guarded
generation draws again for the samples it vetoed, a reply with nothing to say
tries again hotter (:func:`modelkit.dialogue.reply`) - and the command line,
the HTTP API and today's format carry the option and the climb's record.
Off, every one of them is the path it always was.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

from modelkit.assistant import Ask, AskError, _Narrator, parse_anthropic, parse_openai  # noqa: E402
from modelkit.dialogue import converse, reply  # noqa: E402
from modelkit.duo import NegativeFilter, guarded_predict  # noqa: E402
from radixnet.autotemp import AUTO_DRAWS  # noqa: E402
from radixnet.model import new_model  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402

try:  # ``python -m unittest discover -s tests`` imports test modules as top-level modules
    from test_api import start_server, wait_for_job  # noqa: F401
except ImportError:  # ``python -m unittest tests.test_autotemp``
    from tests.test_api import start_server, wait_for_job  # noqa: F401

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(KIT), "RadixCyclicNN")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

CORPUS = ["the cat sat on the mat", "the dog sat on the log", "a bird flew over the hill"]
FAILURE = "the cat sat on the mat"
# a strong cycle: a greedy walk goes round "ab " for ever and never reaches the end
LOOP = ["ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab end"] * 3 + ["ab end"]


def pair(kind="count", blamed=(FAILURE,)):
    positive = new_model(kind, seed=1)
    positive.train(CORPUS, epochs=3)
    negative = NegativeNet(seed=2)
    negative.blame(list(blamed), reason="gibberish")
    return NegativeFilter(positive, negative)


def run(*args, model, expect=0):
    """``python -m radixnet --json`` from the model's checkout; the parsed document."""
    cmd = [sys.executable, "-m", "radixnet", "--backend", "python", "--json", "--model", model, *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=300)
    if proc.returncode != expect:
        raise AssertionError(f"{' '.join(cmd)}\nexit {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return json.loads(proc.stdout) if proc.stdout.strip() else {"error": proc.stderr.strip()}


class TestTheGuard(unittest.TestCase):
    def test_a_vetoed_prediction_climbs_to_one_the_guard_lets_through(self):
        duo = pair()
        found, verdicts, record = guarded_predict(duo.positive, "the cat", pair=duo, mode="beam", k=1, length=30,
                                                  to_end=True)
        self.assertIsNone(record)
        self.assertEqual(found.text, "", "off: the guard vetoed the only continuation, and nothing is said")
        self.assertEqual([v["decision"] for v in verdicts], ["reject"])
        found, verdicts, record = guarded_predict(duo.positive, "the cat", pair=duo, mode="beam", k=1, length=30,
                                                  to_end=True, auto_temperature=2.0)
        self.assertTrue(found.text.strip() and found.reached_end)
        self.assertNotEqual(found.full_text, FAILURE)
        self.assertTrue(record.climbed and record.found)
        self.assertGreater(record.tries, 1)
        self.assertEqual(len(verdicts), record.tries, "every search was judged")
        self.assertEqual(verdicts[0]["decision"], "reject")

    def test_an_answer_the_guard_lets_through_is_not_walked_past(self):
        duo = pair(blamed=["nothing like the corpus at all"])
        plain, _, _ = guarded_predict(duo.positive, "the cat", pair=duo, mode="beam", k=1, length=30, to_end=True)
        found, verdicts, record = guarded_predict(duo.positive, "the cat", pair=duo, mode="beam", k=1, length=30,
                                                  to_end=True, auto_temperature=2.0)
        self.assertEqual(found.full_text, plain.full_text)
        self.assertEqual((record.climbed, record.found, record.tries), (False, True, 1))

    def test_a_guarded_generation_draws_again_for_what_it_vetoed(self):
        # no over-sampling, so the beam offers just the three it is asked for
        beam = pair(blamed=["nothing like the corpus at all"]).generate(count=3, mode="beam", max_length=40,
                                                                         over_sample=1)
        duo = pair(blamed=beam["texts"][:2])  # the beam's own two best texts are failures now
        plain = duo.generate(count=3, mode="beam", max_length=40, seed=3, over_sample=1)
        self.assertLess(len(plain["texts"]), 3, "the beam's own texts were vetoed")
        self.assertNotIn("auto_temperature", plain)
        out = duo.generate(count=3, mode="beam", max_length=40, seed=3, over_sample=1, auto_temperature=2.0)
        record = out["auto_temperature"]
        self.assertTrue(record["climbed"])
        self.assertEqual(len(out["texts"]), len(out["results"]))
        self.assertGreaterEqual(len(out["texts"]), len(plain["texts"]))
        self.assertEqual(out["texts"][: len(plain["texts"])], plain["texts"], "the first generation's come first")
        self.assertEqual(len(set(out["texts"])), len(out["texts"]))
        self.assertGreater(out["candidates"], plain["candidates"])
        self.assertEqual(record["found"], len(out["texts"]) == 3)

    def test_a_guarded_dijkstra_generation_is_one_text(self):
        """Dijkstra's generation is one text however many are asked for: when the guard lets it through, that is
        the whole answer and nothing is walked; when it vetoes it, one walk is drawn again in its place."""
        duo = pair(blamed=["nothing like the corpus at all"])
        plain = duo.generate(count=3, mode="dijkstra", max_length=40)
        self.assertEqual(len(plain["texts"]), 1)
        out = duo.generate(count=3, mode="dijkstra", max_length=40, auto_temperature=2.0)
        self.assertEqual(out["texts"], plain["texts"])
        self.assertEqual(out["auto_temperature"], {"ceiling": 2.0, "climbed": False, "found": True,
                                                   "temperature": None, "tries": 1})
        duo = pair(blamed=plain["texts"])
        out = duo.generate(count=3, mode="dijkstra", max_length=40, seed=3, auto_temperature=2.0)
        record = out["auto_temperature"]
        self.assertTrue(record["climbed"])
        self.assertLessEqual(len(out["texts"]), 1)
        self.assertNotIn(plain["texts"][0], out["texts"])
        self.assertEqual(record["found"], len(out["texts"]) == 1)

    def test_nothing_asked_for_is_found_as_asked(self):
        """A generation of no texts has nothing missing: the ceiling is on, so the record says so, guarded or not."""
        duo = pair()
        self.assertEqual(duo.generate(count=0, auto_temperature=2.0)["auto_temperature"],
                         {"ceiling": 2.0, "climbed": False, "found": True, "temperature": 1.0, "tries": 1})

    def test_everything_vetoed_is_no_answer_at_any_temperature(self):
        duo = pair(blamed=CORPUS)
        out = duo.generate(count=2, mode="sample", max_length=40, seed=3, auto_temperature=1.0)
        record = out["auto_temperature"]
        self.assertTrue(record["climbed"])
        if not record["found"]:
            self.assertLess(len(out["texts"]), 2)
            self.assertEqual(record["tries"], 1 + 1 * AUTO_DRAWS)  # the ladder (1, 1): one rung


class TestAReply(unittest.TestCase):
    def voice(self):
        voice = new_model("count", seed=1)
        voice.train(CORPUS, epochs=3)
        return voice

    def beams_texts(self):
        """A veto of every text the beam offers from the start: a reply as asked has nothing it may say."""
        found = self.voice().predict("", length=0, mode="beam", k=3, to_end=True, max_length=60)
        return {r.full_text for r in found.top}.__contains__

    def test_a_reply_with_nothing_to_say_tries_again_hotter(self):
        """Every text the beam can offer is vetoed, but a warmer walk says something else, and that is the
        turn."""
        veto = self.beams_texts()
        self.assertIsNone(reply(self.voice(), "", mode="beam", k=3, veto=veto, rng=random.Random(1)))
        events = []
        turn = reply(self.voice(), "", mode="beam", k=3, veto=veto, rng=random.Random(1), auto_temperature=2.0,
                     trace=events.append)
        self.assertIsNotNone(turn)
        self.assertFalse(veto(turn.text))
        self.assertTrue(turn.auto_temperature["climbed"] and turn.auto_temperature["found"])
        self.assertEqual(turn.to_dict()["auto_temperature"], turn.auto_temperature)
        climbs = [e for e in events if e["kind"] == "climb"]
        self.assertEqual(climbs[0], {"kind": "climb", "temperature": 1.0, "draws": AUTO_DRAWS})
        self.assertEqual(turn.auto_temperature["temperature"], climbs[-1]["temperature"])

    def test_a_reply_that_answers_is_left_alone(self):
        turn = reply(self.voice(), "the cat", mode="beam", k=3, rng=random.Random(1), auto_temperature=2.0)
        plain = reply(self.voice(), "the cat", mode="beam", k=3, rng=random.Random(1))
        self.assertEqual(turn.text, plain.text)
        self.assertEqual(turn.auto_temperature, {"ceiling": 2.0, "climbed": False, "found": True, "temperature": None,
                                                 "tries": 1})
        self.assertNotIn("auto_temperature", plain.to_dict())

    def test_a_conversation_goes_on_where_it_would_have_stopped(self):
        veto = self.beams_texts()
        stopped = converse(self.voice(), "", 2, k=3, veto=veto, seed=4)
        self.assertEqual(stopped, [])
        going = converse(self.voice(), "", 2, k=3, veto=veto, seed=4, auto_temperature=2.0)
        self.assertGreaterEqual(len(going), 1)
        self.assertTrue(all(t.auto_temperature is not None for t in going))

    def test_the_ceiling_is_checked(self):
        with self.assertRaises(ValueError):
            reply(self.voice(), "the cat", auto_temperature=-1)


class TestTodaysFormat(unittest.TestCase):
    def test_either_dialect_carries_it(self):
        messages = [{"role": "user", "content": "the cat"}]
        for parse in (parse_anthropic, parse_openai):
            with self.subTest(parse=parse.__name__):
                ask = parse({"messages": messages, "max_tokens": 20, "auto_temperature": 2})
                self.assertEqual(ask.auto_temperature, 2.0)
                self.assertEqual(ask.dialogue_options()["auto_temperature"], 2.0)
                self.assertEqual(ask.to_dict()["auto_temperature"], 2.0)
                self.assertNotIn("auto_temperature", parse({"messages": messages}).to_dict())
                with self.assertRaises(AskError):
                    parse({"messages": messages, "auto_temperature": -1})

    def test_the_thinking_says_what_the_climb_does(self):
        lines = []
        narrator = _Narrator(lines.append)
        narrator.trace({"kind": "climb", "temperature": 1.0, "draws": 3})
        narrator.trace({"kind": "climb", "temperature": 0.0, "draws": 1})
        self.assertEqual(lines, [
            "nothing to say yet; trying again at temperature 1.00 (at most 3 replies)",
            "nothing to say yet; trying again at temperature 0.00 (at most 1 reply)",
        ])

    def test_an_ask_off_is_the_ask_it_always_was(self):
        self.assertEqual(Ask().auto_temperature, 0.0)


class TestTheCommandLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-autotemp-")
        data = os.path.join(cls.tmp.name, "loop.txt")
        with open(data, "w", encoding="utf-8") as fh:
            fh.write("\n".join(LOOP) + "\n")
        cls.model = os.path.join(cls.tmp.name, "loop.json")
        run("--kind", "count", "train", "--data", data, "--epochs", 2, model=cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_predict_climbs_out_of_a_greedy_loop(self):
        asked = ("predict", "--prefix", "ab", "--mode", "sample", "--temperature", 0, "--length", 20,
                 "--max-length", 30, "--to-end")
        plain = run(*asked, model=self.model)
        self.assertFalse(plain["reached_end"])
        self.assertNotIn("auto_temperature", plain)
        doc = run(*asked, "--auto-temperature", 2, model=self.model)
        self.assertTrue(doc["reached_end"])
        record = doc["auto_temperature"]
        self.assertTrue(record["climbed"] and record["found"])
        self.assertGreater(record["temperature"], 0)
        self.assertEqual(record["ceiling"], 2.0)

    def test_generate_reports_the_climb(self):
        doc = run("generate", "--count", 2, "--mode", "sample", "--max-length", 30, "--auto-temperature", 2,
                  model=self.model)
        self.assertEqual(len(doc["samples"]), 2)
        self.assertEqual(doc["auto_temperature"]["ceiling"], 2.0)

    def test_every_command_that_answers_takes_it(self):
        for command in ("predict", "generate", "converse", "talk"):
            with self.subTest(command=command):
                proc = subprocess.run([sys.executable, "-m", "radixnet", command, "--help"], cwd=ROOT, env=ENV,
                                      capture_output=True, text=True, timeout=120)
                self.assertIn("--auto-temperature", proc.stdout)

    def test_a_negative_ceiling_is_refused(self):
        run("predict", "--prefix", "ab", "--auto-temperature", -1, model=self.model, expect=1)


class TestTheApi(unittest.TestCase):
    def setUp(self):
        self.client, self.server, self.service = start_server(self.addCleanup)
        status, data, _ = self.client.post("/api/reset", {"kind": "count", "seed": 1})
        self.assertEqual(status, 200, data)
        with self.service.mutating() as model:
            model.train(CORPUS, epochs=3)
        self.service.negative_model().blame([FAILURE], reason="gibberish")

    def post(self, path, body, expect=200):
        status, data, _ = self.client.post(path, body)
        self.assertEqual(status, expect, data)
        return data

    def test_predict_climbs_past_the_veto(self):
        body = {"prefix": "the cat", "mode": "beam", "k": 1, "length": 30, "to_end": True}
        plain = self.post("/api/predict", body)
        self.assertEqual(plain["continuation"], "")
        self.assertNotIn("auto_temperature", plain)
        data = self.post("/api/predict", {**body, "auto_temperature": 2})
        self.assertTrue(data["continuation"].strip())
        self.assertTrue(data["auto_temperature"]["climbed"] and data["auto_temperature"]["found"])
        self.assertGreater(data["guard"]["candidates"], plain["guard"]["candidates"])

    def test_generate_and_converse_carry_it(self):
        data = self.post("/api/generate", {"count": 2, "mode": "sample", "max_length": 40, "seed": 2,
                                           "auto_temperature": 2})
        self.assertIn("auto_temperature", data)
        data = self.post("/api/converse", {"opening": "the cat", "turns": 2, "auto_temperature": 2})
        for turn in data["turns"][1:]:
            self.assertEqual(turn["auto_temperature"]["ceiling"], 2.0)

    def test_a_bad_ceiling_is_a_400(self):
        for path, body in (("/api/predict", {"prefix": "the"}), ("/api/generate", {}), ("/api/converse", {})):
            with self.subTest(path=path):
                self.post(path, {**body, "auto_temperature": -1}, expect=400)
                self.post(path, {**body, "auto_temperature": "hot"}, expect=400)

    def test_todays_format_takes_it(self):
        data = self.post("/v1/messages", {"messages": [{"role": "user", "content": "the cat"}], "max_tokens": 20,
                                          "auto_temperature": 2})
        self.assertIn("content", data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
