"""Tests for the search and training methods of ``SPEC-SearchAndTraining.md``.

The sampling filters (``top_k``, ``top_p``, ``min_p``), the diverse beam, and
the training plan - the order, the curriculum, the replay buffer and early
stopping - on every kind that learns by walking a list of texts.  The Go and
Rust ports are held to the same numbers by ``tests/test_go_parity.py`` and
``tests/test_rust_parity_methods.py``; these tests pin the rules themselves.
"""

import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.beam import diverse_pick, path_overlap  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.model import RadixNet, TrainConfig, load_model  # noqa: E402
from radixnet.resonance import ResonantNet  # noqa: E402
from radixnet.search import check_sampling, sample_walk, sampling_filter  # noqa: E402
from radixnet.training import (  # noqa: E402
    EarlyStop,
    ReplayBuffer,
    TrainingPlan,
    check_plan,
    paced,
    replay_count,
    replay_key,
    shuffle_key,
    splitmix64,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lines(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


CORPUS = _lines("sample_corpus.txt")
KINDS = {"count": CountRewardNet, "radix": RadixNet, "resonant": ResonantNet}


def _options(*costs):
    """``(child, edge, cost)`` options, child = position, as a node offers them."""
    return [(i, 100 + i, c) for i, c in enumerate(costs)]


class TestSamplingFilter(unittest.TestCase):
    def test_off_is_the_list_untouched(self):
        options = _options(0.5, 0.1, 2.0)
        self.assertIs(sampling_filter(options, 1.0), options)
        self.assertIs(sampling_filter(options, 0.0, top_k=1), options)  # greedy never filters
        one = options[:1]
        self.assertIs(sampling_filter(one, 1.0, top_k=1), one)

    def test_top_k_keeps_the_cheapest_in_the_order_offered(self):
        options = _options(0.5, 0.1, 2.0, 0.3)
        kept = sampling_filter(options, 1.0, top_k=2)
        self.assertEqual([o[0] for o in kept], [1, 3])  # the two cheapest, in node order
        self.assertIs(sampling_filter(options, 1.0, top_k=9), options)

    def test_ties_rank_by_position(self):
        options = _options(0.2, 0.2, 0.2)
        self.assertEqual([o[0] for o in sampling_filter(options, 1.0, top_k=2)], [0, 1])

    def test_min_p_is_relative_to_the_best(self):
        # weights exp(-(c - 0.1)): 1, e^-0.4 = 0.67, e^-1.9 = 0.15
        options = _options(0.1, 0.5, 2.0)
        self.assertEqual([o[0] for o in sampling_filter(options, 1.0, min_p=0.5)], [0, 1])
        self.assertEqual([o[0] for o in sampling_filter(options, 1.0, min_p=0.9)], [0])
        # a hotter temperature flattens the weights: more survive
        self.assertEqual([o[0] for o in sampling_filter(options, 10.0, min_p=0.5)], [0, 1, 2])

    def test_top_p_is_the_smallest_nucleus(self):
        options = _options(0.0, 0.0, 0.0, 0.0)  # four equal weights: a quarter each
        self.assertEqual(len(sampling_filter(options, 1.0, top_p=0.5)), 2)
        self.assertEqual(len(sampling_filter(options, 1.0, top_p=0.51)), 3)
        self.assertEqual(len(sampling_filter(options, 1.0, top_p=0.25)), 1)

    def test_the_filters_run_top_k_then_min_p_then_top_p(self):
        options = _options(0.0, 0.1, 0.2, 5.0, 6.0)
        kept = sampling_filter(options, 1.0, top_k=4, min_p=0.5, top_p=0.4)
        # top_k drops 6.0; min_p drops 5.0; top_p keeps the smallest set holding 40 % of the rest
        self.assertEqual([o[0] for o in kept], [0, 1])

    def test_the_cheapest_always_survives(self):
        for top_k, top_p, min_p in ((1, 1.0, 0.0), (0, 1e-9, 0.0), (0, 1.0, 0.999)):
            kept = sampling_filter(_options(3.0, 0.7, 1.0), 1.0, top_k, top_p, min_p)
            self.assertEqual([o[0] for o in kept], [1])

    def test_out_of_range_settings_are_refused(self):
        for bad in ({"top_k": -1}, {"top_p": 0.0}, {"top_p": 1.5}, {"min_p": 1.0}, {"min_p": -0.1},
                    {"top_p": math.nan}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_sampling(**bad)
        check_sampling(0, 1.0, 0.0)
        check_sampling(5, 0.5, 0.99)


class TestFilteredWalks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = CountRewardNet(seed=3)
        cls.model.train(CORPUS, epochs=1)

    def walk(self, rng, **options):
        return sample_walk(self.model.graph, 0, 0, max_chars=40, rng=rng, **options)

    def test_filters_off_are_the_walk_it_always_was(self):
        a = self.walk(random.Random(5), temperature=1.2)
        b = self.walk(random.Random(5), temperature=1.2, top_k=0, top_p=1.0, min_p=0.0)
        self.assertEqual(a.node_ids, b.node_ids)
        self.assertEqual(a.cost, b.cost)

    def test_top_k_one_is_greedy_yet_draws_every_step(self):
        rng = random.Random(9)
        before = rng.getstate()
        filtered = self.walk(rng, temperature=1.0, top_k=1)
        greedy = self.walk(random.Random(9), temperature=0.0)
        self.assertEqual(filtered.node_ids, greedy.node_ids)
        # one random() per step that offered a choice, however few the filters left
        self.assertNotEqual(rng.getstate(), before)

    def test_a_filter_narrows_the_texts(self):
        plain = {self.walk(random.Random(s), temperature=2.0).text for s in range(40)}
        narrow = {self.walk(random.Random(s), temperature=2.0, top_k=1).text for s in range(40)}
        self.assertEqual(len(narrow), 1)
        self.assertGreater(len(plain), 1)

    def test_generate_takes_the_filters_and_refuses_bad_ones(self):
        texts = self.model.generate(max_length=30, mode="sample", count=4, seed=1, top_p=0.5, min_p=0.1)
        self.assertEqual(len(texts), 4)
        with self.assertRaises(ValueError):
            self.model.generate(max_length=30, mode="sample", count=1, seed=1, top_p=0.0)
        with self.assertRaises(ValueError):
            self.model.generate(max_length=30, mode="beam", count=2, diversity=-1.0)


class TestDiverseBeam(unittest.TestCase):
    def test_path_overlap(self):
        self.assertEqual(path_overlap([0, 1, 2, 3], [0, 1, 2, 4]), 2 / 3)
        self.assertEqual(path_overlap([0, 1, 2], [0, 1, 2, 3, 4]), 1.0)  # the shorter one, extended
        self.assertEqual(path_overlap([0, 5, 2], [0, 1, 2]), 0.0)  # parts at the first step
        self.assertEqual(path_overlap([0], [0, 1]), 0.0)  # nothing after the start node

    def test_diverse_pick_trades_cost_for_novelty(self):
        paths = [
            (0.0, 1.0, [0, 1, 2, 3]),
            (0.0, 1.1, [0, 1, 2, 4]),  # the best one with its ending varied
            (0.0, 1.5, [0, 7, 8, 9]),  # somewhere else
        ]
        self.assertEqual(diverse_pick(paths, 3, 0.0), [0, 1, 2])
        self.assertEqual(diverse_pick(paths, 2, 1.0), [0, 2])  # 1.1 + 2/3 > 1.5 + 0
        self.assertEqual(diverse_pick(paths, 3, 1.0), [0, 2, 1])
        self.assertEqual(diverse_pick(paths, 0, 1.0), [])
        self.assertEqual(diverse_pick([], 3, 1.0), [])

    def test_punishment_ranks_before_the_diverse_cost(self):
        paths = [(0.0, 1.0, [0, 1]), (1.0, 0.5, [0, 2]), (0.0, 9.0, [0, 1, 3])]
        self.assertEqual(diverse_pick(paths, 3, 100.0), [0, 2, 1])

    def test_a_diverse_beam_keeps_the_best_first_and_spreads_the_rest(self):
        model = CountRewardNet(seed=1)
        model.train(CORPUS, epochs=2)
        plain = model.predict("", length=0, mode="beam", k=5, to_end=True, max_length=40)
        diverse = model.predict("", length=0, mode="beam", k=5, to_end=True, max_length=40, diversity=5.0)
        self.assertEqual(plain.top[0].text, diverse.top[0].text)
        self.assertEqual(len(diverse.top), 5)
        self.assertEqual(len({r.text for r in diverse.top}), 5)
        same = model.predict("", length=0, mode="beam", k=5, to_end=True, max_length=40, diversity=0.0)
        self.assertEqual([r.text for r in plain.top], [r.text for r in same.top])
        # only the top side is spread: the least likely continuations answer another question
        self.assertEqual([r.text for r in plain.bottom], [r.text for r in diverse.bottom])

        def spread(found):
            ids = [r.node_ids for r in found.top]
            return sum(path_overlap(a, b) for i, a in enumerate(ids) for b in ids[i + 1:])

        self.assertLessEqual(spread(diverse), spread(plain))


class TestKeysAndPacing(unittest.TestCase):
    def test_the_keys_are_pinned(self):
        # the Go and Rust ports assert the same numbers
        self.assertEqual(splitmix64(0), 0xE220A8397B1DCDAF)
        self.assertEqual(shuffle_key(7, 1, 0), 2915934175157713820)
        self.assertEqual(shuffle_key(-3, 2, 5), 14722852399570101988)
        self.assertEqual(replay_key(7, 0), 5594249859331236493)
        self.assertEqual(replay_key(-1, 9), 6800963518696358292)

    def test_paced(self):
        self.assertEqual([paced(10, 0.3, j, 4) for j in range(4)], [3, 6, 8, 10])
        self.assertEqual(paced(10, 0.01, 0, 4), 1)
        self.assertEqual(paced(10, 0.5, 0, 1), 10)  # one epoch: the whole list
        self.assertEqual(paced(0, 0.5, 0, 3), 0)

    def test_replay_count_rounds_half_up(self):
        self.assertEqual(replay_count(0.25, 10, 100), 3)
        self.assertEqual(replay_count(0.24, 10, 100), 2)
        self.assertEqual(replay_count(5.0, 10, 7), 7)
        self.assertEqual(replay_count(1e300, 10, 7), 7)
        self.assertEqual(replay_count(0.0, 10, 7), 0)
        self.assertEqual(replay_count(1.0, 10, 0), 0)

    def test_check_plan(self):
        check_plan()
        check_plan("shuffle", 0.5, 1.0, 0, 3, 0.1)
        for bad in (("random",), ("corpus", 0.0), ("corpus", 1.5), ("corpus", 1.0, -1.0), ("corpus", 1.0, math.inf),
                    ("corpus", 1.0, 0.0, -1), ("corpus", 1.0, 0.0, None, -1), ("corpus", 1.0, 0.0, None, 0, -0.5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_plan(*bad)
        with self.assertRaises(ValueError):
            TrainConfig(curriculum=0.0).validate()


class TestReplayBuffer(unittest.TestCase):
    def test_bottom_k_whatever_the_order_of_offers(self):
        texts = [f"text {i}" for i in range(50)]
        one = ReplayBuffer(8, seed=7)
        one.offer(texts)
        two = ReplayBuffer(8, seed=7)
        for i in range(0, 50, 3):
            two.offer(texts[i:i + 3])
        self.assertEqual(one.to_dict(), two.to_dict())
        want = sorted(range(50), key=lambda g: (replay_key(7, g), g))[:8]
        self.assertEqual(one.to_dict()["index"], want)
        self.assertEqual(one.seen, 50)
        self.assertEqual(len(one), 8)

    def test_round_trip_trusts_the_indices_not_the_order(self):
        buf = ReplayBuffer(5, seed=2)
        buf.offer([f"t{i}" for i in range(20)])
        doc = buf.to_dict()
        shuffled = {**doc, "index": doc["index"][::-1], "texts": doc["texts"][::-1]}
        self.assertEqual(ReplayBuffer.from_dict(shuffled, 2).to_dict(), doc)
        with self.assertRaises(ValueError):
            ReplayBuffer.from_dict({"size": 3, "seen": 3, "index": [0, 1], "texts": ["a"]}, 2)

    def test_resize_drops_the_largest_priorities(self):
        buf = ReplayBuffer(10, seed=3)
        buf.offer([f"t{i}" for i in range(20)])
        first = buf.texts()[:4]
        buf.resize(4)
        self.assertEqual(buf.texts(), first)
        empty = ReplayBuffer(0, seed=3)
        empty.offer(["a", "b"])
        self.assertEqual((len(empty), empty.seen), (0, 2))


class TestEarlyStop(unittest.TestCase):
    def test_patience_and_min_delta(self):
        stop = EarlyStop(patience=2, min_delta=0.1)
        self.assertFalse(stop.update(5.0))
        self.assertFalse(stop.update(4.95))  # not 0.1 better: one stale epoch
        self.assertTrue(stop.update(4.99))
        off = EarlyStop()
        self.assertFalse(any(off.update(1.0) for _ in range(10)))


class TestTrainingPlan(unittest.TestCase):
    TEXTS = ["ccc", "a", "bb", "dddd"]
    LENGTHS = [3, 1, 2, 4]

    def plan(self, **kw):
        return TrainingPlan(self.TEXTS, self.LENGTHS, seed=1, epochs=kw.pop("epochs", 3), **kw)

    def test_off_is_plain(self):
        plan = self.plan()
        self.assertTrue(plan.plain)
        self.assertEqual(plan.epoch(0, 1), ([0, 1, 2, 3], []))
        self.assertIsNone(plan.finish())

    def test_orders_and_the_curriculum(self):
        plan = self.plan(order="shortest-first", curriculum=0.5)
        self.assertFalse(plan.plain)
        self.assertEqual([plan.epoch(j, j + 1)[0] for j in range(3)], [[1, 2], [1, 2, 0], [1, 2, 0, 3]])
        self.assertEqual(self.plan(order="longest-first").epoch(0, 1)[0], [3, 0, 2, 1])

    def test_a_shuffle_moves_with_the_epoch_number(self):
        texts = [str(i) for i in range(12)]
        plan = TrainingPlan(texts, [1] * 12, seed=9, epochs=2, order="shuffle")
        a, b = plan.epoch(0, 1)[0], plan.epoch(0, 2)[0]
        self.assertEqual(sorted(a), list(range(12)))
        self.assertNotEqual(a, b)
        self.assertEqual(a, sorted(range(12), key=lambda i: (shuffle_key(9, 1, i), i)))

    def test_rehearsal_slices_walk_the_pool(self):
        buf = ReplayBuffer(3, seed=1)
        buf.offer(["x", "y", "z"])
        pool = buf.texts()
        plan = self.plan(replay=0.5, buffer=buf)
        self.assertEqual(plan.replayed(), pool)
        self.assertEqual(plan.epoch(0, 1)[1], pool[:2])
        self.assertEqual(plan.epoch(1, 2)[1], [pool[2], pool[0]])

    def test_an_epoch_inside_the_curriculum_does_not_count_toward_the_stop(self):
        plan = self.plan(curriculum=0.5, patience=1, epochs=3)
        self.assertFalse(plan.stop(0, 5.0))  # walked 2 of 4: skipped
        self.assertFalse(plan.stop(1, 5.0))  # walked 3 of 4: skipped
        self.assertFalse(plan.stop(2, 5.0))  # the first full epoch sets the best
        self.assertTrue(plan.stop(2, 5.0))


class TestPlannedTraining(unittest.TestCase):
    def test_every_setting_off_leaves_the_file_as_it_was(self):
        for kind, cls in KINDS.items():
            with self.subTest(kind=kind):
                model = cls(seed=2)
                model.train(CORPUS[:20], epochs=2)
                doc = model.to_dict()
                self.assertNotIn("replay", doc)
                self.assertFalse(any("early_stop" in r for r in doc["history"]))

    def test_the_buffer_is_saved_last_and_read_back(self):
        for kind, cls in KINDS.items():
            with self.subTest(kind=kind):
                model = cls(seed=2)
                model.train(CORPUS[:30], epochs=1, replay_size=6)
                doc = model.to_dict()
                self.assertEqual(list(doc)[-1], "replay")
                self.assertEqual((doc["replay"]["size"], doc["replay"]["seen"], len(doc["replay"]["texts"])), (6, 30, 6))
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "m.json")
                    model.save(path)
                    back = load_model(path)
                self.assertEqual(back.replay.to_dict(), model.replay.to_dict())
                self.assertEqual(json.dumps(back.to_dict()["replay"]), json.dumps(doc["replay"]))

    def test_rehearsed_texts_are_not_new(self):
        for kind, cls in KINDS.items():
            with self.subTest(kind=kind):
                model = cls(seed=2)
                model.train(CORPUS[:20], epochs=1, replay_size=10)
                texts, chars = model.meta["trained_texts"], model.meta["trained_chars"]
                records = model.train(CORPUS[20:30], epochs=2, replay=1.0)
                self.assertEqual(model.meta["trained_texts"], texts + 10)
                self.assertEqual(model.meta["trained_chars"], chars + sum(len(t) for t in CORPUS[20:30]))
                self.assertEqual(model.replay.seen, 30)
                # the rehearsal walked more transitions than the new texts alone
                plain = cls(seed=2)
                plain.train(CORPUS[:20], epochs=1)
                alone = plain.train(CORPUS[20:30], epochs=2)
                self.assertGreater(records[0]["transitions"], alone[0]["transitions"])

    def test_early_stop_marks_the_record_that_stops(self):
        for kind, cls in KINDS.items():
            with self.subTest(kind=kind):
                model = cls(seed=2)
                records = model.train(CORPUS[:20], epochs=6, patience=1, min_delta=100.0)
                self.assertEqual(len(records), 2)
                self.assertEqual([r.get("early_stop") for r in records], [None, True])
                self.assertEqual(model.meta["epochs_total"], 2)

    def test_replay_size_zero_drops_the_buffer(self):
        model = CountRewardNet(seed=2)
        model.train(CORPUS[:10], epochs=1, replay_size=4)
        self.assertIsNotNone(model.replay)
        model.train(CORPUS[10:15], epochs=1)  # no size: the buffer is kept and offered the run
        self.assertEqual(model.replay.seen, 15)
        model.train(CORPUS[15:20], epochs=1, replay_size=0)
        self.assertIsNone(model.replay)
        self.assertNotIn("replay", model.to_dict())

    def test_feedback_passes_leave_the_buffer_alone(self):
        count = CountRewardNet(seed=2)
        count.train(CORPUS[:10], epochs=1, replay_size=5)
        before = count.replay.to_dict()
        count.reward(CORPUS[10:12])
        count.punish(CORPUS[12:14])
        count.train(CORPUS[14:16], epochs=1, phase="negative")  # a phase marks a feedback pass
        self.assertEqual(count.replay.to_dict(), before)
        radix = RadixNet(seed=2)
        radix.train(CORPUS[:10], epochs=1, replay_size=5)
        before = radix.replay.to_dict()
        radix.two_nrl(CORPUS[10:12], CORPUS[12:14], neg_epochs=1, pos_epochs=1)
        radix.reward(CORPUS[14:15], epochs=1)
        radix.punish(CORPUS[15:16], epochs=1)
        self.assertEqual(radix.replay.to_dict(), before)
        resonant = ResonantNet(seed=2)
        resonant.train(CORPUS[:10], epochs=1, replay_size=5)
        before = resonant.replay.to_dict()
        resonant.reward(CORPUS[10:12])
        resonant.punish(CORPUS[12:14])
        self.assertEqual(resonant.replay.to_dict(), before)

    def test_a_curriculum_changes_the_counts_not_the_graph(self):
        plain = CountRewardNet(seed=4)
        plain.train(CORPUS, epochs=3)
        paced_model = CountRewardNet(seed=4)
        paced_model.train(CORPUS, epochs=3, order="shortest-first", curriculum=0.25)
        a, b = plain.to_dict()["graph"], paced_model.to_dict()["graph"]
        self.assertEqual(a["nodes"]["labels"], b["nodes"]["labels"])  # the structure pass saw every text
        self.assertNotEqual(a["edges"]["count"], b["edges"]["count"])  # the first epochs counted fewer


if __name__ == "__main__":
    unittest.main()


class TestApiAndCli(unittest.TestCase):
    """The settings over HTTP and on the command line: accepted in range, a 400 / usage error out of it."""

    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, kind="count", seed=1)
        status, _, _ = cls.client.post("/api/train", {"texts": CORPUS[:30], "epochs": 1, "replay_size": 8})
        assert status == 202
        cls.wait()

    @classmethod
    def wait(cls):
        from tests.test_api import wait_for_job

        return wait_for_job(cls.client)

    def test_search_settings_out_of_range_are_a_400(self):
        for route, body in (
            ("/api/predict", {"prefix": "the", "top_p": 0}),
            ("/api/predict", {"prefix": "the", "min_p": 1}),
            ("/api/predict", {"prefix": "the", "top_k": -1}),
            ("/api/generate", {"mode": "sample", "top_p": 1.5}),
            ("/api/generate", {"mode": "beam", "diversity": -0.5}),
        ):
            with self.subTest(route=route, body=body):
                status, data, _ = self.client.post(route, body)
                self.assertEqual(status, 400, data)

    def test_search_settings_in_range_are_used(self):
        status, plain, _ = self.client.post("/api/generate", {"mode": "beam", "count": 5, "max_length": 40, "guard": False})
        self.assertEqual(status, 200, plain)
        status, spread, _ = self.client.post(
            "/api/generate", {"mode": "beam", "count": 5, "max_length": 40, "diversity": 5.0, "guard": False}
        )
        self.assertEqual(status, 200, spread)
        self.assertEqual(plain["samples"][0]["text"], spread["samples"][0]["text"])
        status, data, _ = self.client.post(
            "/api/generate", {"mode": "sample", "count": 3, "seed": 4, "top_k": 2, "top_p": 0.8, "min_p": 0.1,
                              "guard": False}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual(len(data["samples"]), 3)

    def test_training_settings_out_of_range_are_a_400(self):
        for body in ({"order": "random"}, {"curriculum": 0}, {"curriculum": 2}, {"replay": -1},
                     {"replay_size": -2}, {"patience": -1}, {"min_delta": -0.1}):
            with self.subTest(body=body):
                status, data, _ = self.client.post("/api/train", {"texts": CORPUS[:3], "epochs": 1, **body})
                self.assertEqual(status, 400, data)

    def test_a_planned_run_over_http(self):
        # 20 texts, a curriculum of 0.9 over 6 epochs: 18, 19, 19, 20, 20, 20 texts
        status, _, _ = self.client.post("/api/train", {
            "texts": CORPUS[30:50], "epochs": 6, "order": "Shortest-First", "curriculum": 0.9, "replay": 0.5,
            "patience": 1, "min_delta": 100,
        })
        self.assertEqual(status, 202)
        job = self.wait()
        self.assertEqual(job["state"], "done", job)
        history = job["history"]
        # the epochs inside the curriculum never count toward the stop; the first full one sets the best
        # and the next one, no better by 100, stops the run
        self.assertEqual(len(history), 5)
        self.assertEqual([r.get("early_stop") for r in history], [None] * 4 + [True])
        self.assertEqual(self.service.model.replay.seen, 50)

    def test_the_cli_refuses_settings_out_of_range(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.json")
            data = os.path.join(ROOT, "data", "sample_corpus.txt")
            for args, flag in ((["generate", "--top-p", "0"], "--top-p"), (["generate", "--min-p", "1"], "--min-p"),
                               (["predict", "--diversity", "-1"], "--diversity"),
                               (["train", "--data", data, "--order", "random"], "--order"),
                               (["train", "--data", data, "--curriculum", "0"], "--curriculum"),
                               (["train", "--data", data, "--replay-size", "-1"], "--replay-size")):
                with self.subTest(args=args):
                    proc = subprocess.run([sys.executable, "-m", "radixnet", "--model", model, *args],
                                          cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(f"argument {flag}", proc.stderr)
                    self.assertFalse(os.path.exists(model))  # refused before anything ran
