"""Tests for radixnet.countnet (the count / reward model) and radixnet.beam (top-K / bottom-K search).

Also covers the model-kind plumbing: ``load_model`` detecting the kind, the
API's ``/api/model`` endpoints, kind-aware prediction / feedback and the
CLI's ``--kind`` option.
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import diff  # noqa: E402
from radixnet.beam import Prediction, beam_predict, default_beam, path_probability  # noqa: E402
from radixnet.countnet import COUNT_MODEL_FORMAT, CountRewardGraph, CountRewardNet  # noqa: E402
from radixnet.graph import BACK, END, FIRST, START  # noqa: E402
from radixnet.search import (  # noqa: E402
    LEAST_PUNISHED,
    PUNISH_TOLERANCE,
    PUNISHMENT,
    REWARD,
    least_punished,
    parse_traversal,
)
from radixnet.model import RadixNet, load_model, model_class, model_from_dict, model_kinds, new_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTS = [
    "the cat sat on the mat",
    "the cat ran to the door",
    "the dog sat on the log",
    "the dog ate the bone",
    "the cat sat on the mat",
]


def _lines(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


def trained(epochs=2, seed=1):
    model = CountRewardNet(seed=seed)
    model.train(TEXTS, epochs=epochs)
    return model


class TestWeightFunction(unittest.TestCase):
    """The dual frequency function: the edge's share of its node's traversals, all time and in the sliding window."""

    def test_edge_weight_formula(self):
        g = CountRewardGraph(seed=0, count_scale=0.0, global_scale=1.0, window_scale=1.0, reward_scale=2.0)
        s = g.SMOOTHING
        # 3 of the node's 4 traversals, 2 of its 2 recent ones, over 2 children, reward -1.5
        expected = math.log((3 + s) / (4 + 2 * s)) + math.log((2 + s) / (2 + 2 * s)) + 2.0 * -1.5
        self.assertAlmostEqual(g.edge_weight(3, -1.5, 4, 2, 2, 2), expected)
        g.count_scale = 1.0
        self.assertAlmostEqual(g.edge_weight(3, 0.0, 4, 2, 2, 2), math.log(4) + math.log((3 + s) / (4 + 2 * s)) + math.log((2 + s) / (2 + 2 * s)))
        # nothing seen anywhere: every child of the node gets the same weight (uniform)
        self.assertAlmostEqual(g.edge_weight(0, 0.0, 0, 3, 0, 0), g.edge_weight(0, 0.0, 0, 3, 0, 0))
        self.assertEqual(g.weight_config()["function"], "dual-frequency")
        with self.assertRaises(ValueError):
            CountRewardGraph(seed=0, window=0)

    def test_activations_are_one_so_scores_are_weights(self):
        model = trained()
        g = model.graph
        for node in g.alive_nodes():
            self.assertAlmostEqual(g.activation_of(node), 1.0)
        for p in g.alive_nodes():
            for c, score in g.child_scores(p):
                self.assertAlmostEqual(score, g.edge_w[g.children[p][c]])
        # every stored weight is the dual frequency function of the tracked numbers
        for p in g.alive_nodes():
            edges = list(g.children[p].values())
            total = sum(g.edge_count[e] for e in edges)
            recent = sum(g.window_edge_count[e] for e in edges)
            for e in edges:
                self.assertAlmostEqual(g.edge_w[e], g.edge_weight(g.edge_count[e], g.edge_reward[e], total, len(edges), g.window_edge_count[e], recent))

    def test_probabilities_follow_the_shares(self):
        model = CountRewardNet(seed=0)
        model.train(["abx", "abx", "abx", "aby"], epochs=1)
        g = model.graph
        node, _ = g.lookup("abx")
        parent = next(iter(g.parents[node]))
        probs = dict(g.child_probs(parent))
        x = probs[node]
        y = probs[g.lookup("aby")[0]]
        # shares 3/4 and 1/4 (smoothed: 3.5/5 and 1.5/5), all time and recent alike; the default scales 0.5 + 0.5
        # make the probability the (smoothed) share itself: 7 : 3
        self.assertAlmostEqual(x / y, 3.5 / 1.5)
        shares = {c: (a, r) for c, e, a, r in g.shares(parent)}
        self.assertAlmostEqual(shares[node][0], 0.75)
        self.assertAlmostEqual(shares[node][1], 0.75)
        self.assertEqual((g.total_traversals, g.window_traversals), (8, 8))
        model.train(["abx", "abx", "abx", "aby"], epochs=1)  # counts double: 6.5/9 vs 2.5/9
        probs = dict(g.child_probs(parent))
        self.assertAlmostEqual(probs[node] / probs[g.lookup("aby")[0]], 6.5 / 2.5)
        both = CountRewardNet(seed=0, global_scale=1.0, window_scale=1.0)  # both experts at full weight: the share squared
        both.train(["abx", "abx", "abx", "aby"], epochs=1)
        gb = both.graph
        pb = dict(gb.child_probs(next(iter(gb.parents[gb.lookup("abx")[0]]))))
        self.assertAlmostEqual(pb[gb.lookup("abx")[0]] / pb[gb.lookup("aby")[0]], (3.5 / 1.5) ** 2)
        self.assertEqual(g.total_traversals, 16)

    def test_sliding_window_forgets_old_traversals(self):
        model = CountRewardNet(seed=0, window=4)
        model.train(["abx"] * 6, epochs=1)  # 12 traversals: the window keeps the last 4
        g = model.graph
        self.assertEqual((g.total_traversals, g.window_traversals), (12, 4))
        model.train(["aby"] * 2, epochs=1)  # 4 more: now only aby's traversals are inside the window
        x = g.lookup("abx")[0]
        y = g.lookup("aby")[0]
        parent = next(iter(g.parents[x]))
        shares = {c: (a, r) for c, e, a, r in g.shares(parent)}
        self.assertAlmostEqual(shares[x][0], 0.75)  # all time: 6 of 8
        self.assertAlmostEqual(shares[x][1], 0.0)   # recently: none
        self.assertAlmostEqual(shares[y][1], 1.0)
        probs = dict(g.child_probs(parent))
        self.assertGreater(probs[y], probs[x])  # the recent share outweighs the all-time share here
        self.assertEqual((g.total_traversals, g.window_traversals), (16, 4))
        # the window can be resized on the fly; a wider one just keeps more of what comes next
        config = model.configure_weights(window=2)
        self.assertEqual((config["window"], g.window_traversals), (2, 2))
        model.configure_weights(window_scale=0.0)
        probs = dict(g.child_probs(parent))
        self.assertGreater(probs[x], probs[y])  # without the recent term the all-time share decides
        with self.assertRaises(ValueError):
            model.configure_weights(nope=1)
        with self.assertRaises(ValueError):
            model.configure_weights(window=0)
        with self.assertRaises(ValueError):
            model.configure_weights(global_scale=float("nan"))

    def test_reward_and_penalty_move_odds_by_e(self):
        model = CountRewardNet(seed=0)
        model.train(["abx", "aby"], epochs=1)
        g = model.graph
        x = g.lookup("abx")[0]
        y = g.lookup("aby")[0]
        parent = next(iter(g.parents[x]))
        self.assertAlmostEqual(dict(g.child_probs(parent))[x], 0.5)
        before = dict(g.child_probs(parent))
        model.reward(["abx"], strength=1.0)  # also counts a traversal: shares 2/3 vs 1/3 -> 2.5 : 1.5
        probs = dict(g.child_probs(parent))
        self.assertAlmostEqual(probs[x] / probs[y], math.e * (2.5 / 1.5))
        model.punish(["abx"], strength=2.0)  # no traversal: only the reward moves
        probs = dict(g.child_probs(parent))
        self.assertAlmostEqual(probs[x] / probs[y], math.exp(-1.0) * (2.5 / 1.5))
        self.assertEqual(g.edge_count[g.children[parent][x]], 2)
        self.assertGreater(before[x], 0)

    def test_invert_flips_rewards_only(self):
        model = trained()
        model.reward([TEXTS[0]], strength=1.5)
        g = model.graph
        counts = list(g.edge_count)
        rewards = list(g.edge_reward)
        window = list(g._window)
        model.invert()
        self.assertTrue(g.inverted)
        self.assertEqual(g.edge_count, counts)
        self.assertEqual(g.edge_reward, [-r for r in rewards])
        self.assertEqual(list(g._window), window)
        model.invert()
        self.assertFalse(g.inverted)
        self.assertEqual(g.edge_reward, rewards)

    def test_window_state_survives_a_round_trip(self):
        model = CountRewardNet(seed=1, window=7)
        model.train(TEXTS, epochs=1)
        model.reward([TEXTS[0]], strength=0.5)
        again = model_from_dict(model.to_dict())
        self.assertEqual((again.graph.window, again.graph.window_traversals, again.graph.total_traversals),
                         (7, model.graph.window_traversals, model.graph.total_traversals))
        self.assertEqual(again.weight_config(), model.weight_config())
        self.assertEqual(again.stats(), model.stats())
        self.assertEqual(again.predict("the cat", length=6, k=2).top[0].text, model.predict("the cat", length=6, k=2).top[0].text)

    def test_legacy_files_keep_the_old_function(self):
        model = CountRewardNet(seed=1)
        model.train(TEXTS[:2], epochs=1)
        d = model.to_dict()
        d["graph"]["weights"] = {"kind": "count-reward", "count_scale": 1.0, "reward_scale": 1.0}  # a file from before
        old = model_from_dict(d)
        self.assertEqual((old.graph.count_scale, old.graph.global_scale, old.graph.window_scale), (1.0, 0.0, 0.0))
        self.assertEqual(old.graph.window_traversals, 0)

    def test_invariants_and_compression(self):
        model = trained(epochs=3)
        model.graph.check_invariants(TEXTS, compressed=True)
        self.assertGreater(model.graph.compression_ratio(), 1.0)
        self.assertEqual(len(model.graph.edge_reward), len(model.graph.edge_w))
        self.assertEqual(len(model.graph.window_edge_count), len(model.graph.edge_w))


class TestTraining(unittest.TestCase):
    def test_records_and_meta(self):
        model = CountRewardNet(seed=3)
        records = model.train(TEXTS + ["hi"], epochs=3, lr=0.9, act_lr=0.9, batch_size=1)  # rates are ignored
        self.assertEqual([r["epoch"] for r in records], [1, 2, 3])
        self.assertEqual(model.history, records)
        for r in records:
            self.assertEqual(r["skipped_short"], 1)
            self.assertTrue(r["traversed"])
            self.assertEqual(r["reward"], 0.0)
            self.assertAlmostEqual(r["perplexity"], math.exp(r["loss"]), places=12)
        self.assertEqual(model.meta["trained_texts"], 5)
        self.assertEqual(model.meta["epochs_total"], 3)
        # every pass walks the same transitions and the loss never rises: counts only sharpen the data
        self.assertEqual(len({r["transitions"] for r in records}), 1)
        self.assertLessEqual(records[-1]["loss"], records[0]["loss"] + 1e-12)
        stats = model.stats()
        self.assertEqual(stats["kind"], "count")
        self.assertEqual((stats["backend"], stats["device"]), ("python", "cpu"))
        for key in ("rewards_total", "penalties_total", "feedback_passes", "edge_reward_positive", "edge_reward_negative"):
            self.assertIn(key, stats)

    def test_two_nrl_penalises_then_rewards_without_inverting(self):
        model = trained()
        # a bad text the model has seen: its path exists, so the penalty lowers a real score (an unseen bad text
        # gets its path created with a negative reward, which is still far likelier than an unknown trigram)
        bad, good = ["the dog sat on the log"], ["the dog ate the bone"]
        before_bad = model.score(bad[0])["log_prob"]
        model.punish(bad, epochs=2, strength=1.5)
        self.assertLess(model.score(bad[0])["log_prob"], before_bad)
        mid_good = model.score(good[0])["log_prob"]  # shares "the dog " with the bad text, so it dropped a little too
        model.reward(good, epochs=1, strength=1.5)
        self.assertGreater(model.score(good[0])["log_prob"], mid_good)
        result = model.two_nrl(bad, good, neg_epochs=2, pos_epochs=1, strength=1.5)
        self.assertEqual([r["phase"] for r in result["negative"]], ["negative", "negative"])
        self.assertEqual([r["phase"] for r in result["positive"]], ["positive"])
        self.assertEqual([r["reward"] for r in result["negative"] + result["positive"]], [-1.5, -1.5, 1.5])
        self.assertFalse(result["inverted"])
        self.assertEqual(model.meta["twonrl_runs"], 1)
        self.assertEqual(model.meta["feedback_passes"], 6)
        with self.assertRaises(TypeError):
            model.two_nrl(bad, good, lr=0.1)

    def test_stop_event_skips_the_positive_phase(self):
        import threading

        model = trained()
        stop = threading.Event()
        stop.set()
        result = model.two_nrl(["zzz qqq"], ["the cat sat on the mat"], neg_epochs=3, pos_epochs=3)
        self.assertEqual(len(result["negative"]), 3)
        self.assertEqual(len(result["positive"]), 3)
        result = model.two_nrl(["zzz qqq"], ["the cat sat on the mat"], neg_epochs=3, pos_epochs=3, stop_event=stop)
        self.assertEqual(len(result["negative"]), 1)
        self.assertEqual(result["positive"], [])

    def test_generate_score_and_locate_work(self):
        model = trained()
        samples = model.generate(count=3, max_length=30, seed=7)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(isinstance(s.text, str) for s in samples))
        best = model.generate(mode="dijkstra", max_length=40)[0]
        self.assertTrue(best.reached_end)
        self.assertGreater(model.score("the cat sat on the mat")["log_prob"], model.score("the cat sat on the log")["log_prob"] - 50)
        self.assertEqual(model.score("qqq")["unknown_transitions"], 2)
        self.assertEqual(model.locate("")[0], START)


class TestPrediction(unittest.TestCase):
    def test_top_and_bottom_k(self):
        model = trained()
        p = model.predict("the cat", length=8, k=3)
        self.assertIsInstance(p, Prediction)
        self.assertEqual((p.k, p.beam, p.mode), (3, default_beam(3), "beam"))
        self.assertEqual(len(p.top), 3)
        self.assertLessEqual(len(p.bottom), 3)
        self.assertGreater(len(p.bottom), 0)
        costs = [r.cost for r in p.top]
        self.assertEqual(costs, sorted(costs))
        worst = [r.cost for r in p.bottom]
        self.assertEqual(worst, sorted(worst, reverse=True))
        self.assertGreaterEqual(min(worst), max(costs) - 1e-9)
        # the prediction itself is the best path
        self.assertEqual((p.text, p.cost, p.node_ids), (p.top[0].text, p.top[0].cost, p.top[0].node_ids))
        self.assertEqual(p.full_text, "the cat" + p.text)
        self.assertTrue(all(r.full_text == "the cat" + r.text for r in p.top + p.bottom))
        texts = [tuple(r.node_ids) for r in p.top + p.bottom]
        self.assertEqual(len(texts), len(set(texts)))  # no path on both sides
        for r in p.top:
            self.assertTrue(r.reached_end or len(r.text) >= 8)
        d = p.to_dict()
        self.assertEqual(set(d) - set(p.top[0].to_dict()), {"top", "bottom", "k", "beam", "mode", "traversal"})
        self.assertEqual(d["traversal"], "reward")
        self.assertEqual(len(d["top"]), 3)

    def test_dijkstra_alias_sample_mode_and_edge_cases(self):
        model = trained()
        alias = model.predict("the cat", length=8, k=2, mode="dijkstra")
        self.assertEqual(alias.mode, "beam")
        self.assertEqual(alias.text, model.predict("the cat", length=8, k=2).text)
        walk = model.predict("the cat", length=6, mode="sample")
        self.assertEqual((walk.mode, len(walk.top), walk.bottom), ("sample", 1, []))
        self.assertTrue(walk.full_text.startswith("the cat"))
        self.assertEqual(model.predict("the cat", length=0, k=2).text, "")
        self.assertEqual(model.predict("the cat", length=5, k=0).top, [])
        capped = model.predict("the cat", length=3, k=2, max_length=4)
        self.assertTrue(all(len(r.text) <= 4 for r in capped.top + capped.bottom))
        unknown = model.predict("qqq", length=5, k=2)
        self.assertTrue(unknown.full_text.startswith("qqq"))
        to_end = model.predict("the", length=1, k=2, to_end=True)
        self.assertTrue(all(r.reached_end for r in to_end.top))
        self.assertEqual(model.predict("the", length=0, k=2, to_end=True).text, "")  # length 0 emits nothing, as RadixNet
        with self.assertRaises(ValueError):
            model.predict("the", mode="nope")
        with self.assertRaises(ValueError):
            model.predict("the", k=-1)
        with self.assertRaises(ValueError):
            model.predict("the", beam=0)
        with self.assertRaises(TypeError):
            model.predict(5)

    def test_beam_matches_dijkstra_on_a_radix_model(self):
        model = RadixNet(seed=2, backend="python")
        model.train(_lines("sample_corpus.txt")[:20], epochs=2, batch_size=8, lr=0.3)
        node, offset = model.locate("the ")
        top, bottom, expanded = beam_predict(model.graph, node, offset, min_chars=6, k=4, beam=64)
        self.assertGreater(expanded, 0)
        best = model.predict("the ", length=6)
        self.assertAlmostEqual(top[0].cost, best.cost, places=9)
        self.assertEqual(top[0].text, best.text)
        self.assertEqual(len(top), 4)
        self.assertTrue(all(0.0 < path_probability(r) <= 1.0 for r in top + bottom))
        self.assertEqual(beam_predict(model.graph, node, offset, min_chars=6, k=0), ([], [], 0))

    def test_bottom_paths_stay_comparable_with_to_end(self):
        model = trained()
        node, offset = model.locate("the ")
        top, bottom, _ = beam_predict(model.graph, node, offset, min_chars=0, k=2, to_end=True)
        longest_top = max(len(r.text) for r in top)
        self.assertTrue(all(len(r.text) <= 2 * longest_top + 40 for r in bottom))


class TestPersistenceAndKinds(unittest.TestCase):
    def test_round_trip_and_kind_detection(self):
        model = trained()
        model.reward([TEXTS[0]], strength=2.0)
        model.punish(["zzz qqq"], strength=0.5)
        d = model.to_dict()
        self.assertEqual((d["format"], d["kind"]), (COUNT_MODEL_FORMAT, "count"))
        self.assertIn("reward", d["graph"]["edges"])
        again = model_from_dict(d)
        self.assertIsInstance(again, CountRewardNet)
        self.assertEqual(again.stats(), model.stats())

        def rewards_by_labels(graph):  # to_dict compacts tombstoned edges away, so edge ids differ after a reload
            return {
                (graph.labels[p], graph.labels[c]): (graph.edge_count[e], graph.edge_reward[e])
                for p in graph.alive_nodes() for c, e in graph.children[p].items()
            }

        self.assertEqual(rewards_by_labels(again.graph), rewards_by_labels(model.graph))
        first, second = again.predict("the cat", length=8, k=3), model.predict("the cat", length=8, k=3)
        self.assertEqual([(r.text, round(r.cost, 9)) for r in first.top], [(r.text, round(r.cost, 9)) for r in second.top])
        self.assertEqual([(r.text, round(r.cost, 9)) for r in first.bottom], [(r.text, round(r.cost, 9)) for r in second.bottom])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json.gz")
            model.save(path)
            loaded = load_model(path)
            self.assertIsInstance(loaded, CountRewardNet)
            self.assertEqual(loaded.history, model.history)
            radix = RadixNet(seed=0, backend="python")
            radix.save(os.path.join(tmp, "r.json"))
            self.assertIsInstance(load_model(os.path.join(tmp, "r.json")), RadixNet)
            with self.assertRaises(ValueError):
                CountRewardNet.load(os.path.join(tmp, "r.json"))
        with self.assertRaises(ValueError):
            model_from_dict({"format": "nope"})

    def test_kind_registry(self):
        self.assertEqual([k["kind"] for k in model_kinds()], ["radix", "count", "negative", "resonant"])
        self.assertIs(model_class("count"), CountRewardNet)
        self.assertIs(model_class(None), RadixNet)
        self.assertIs(model_class(" Radix "), RadixNet)
        with self.assertRaises(ValueError):
            model_class("nope")
        self.assertIsInstance(new_model("count", seed=4), CountRewardNet)
        self.assertEqual(new_model("count", seed=4).seed, 4)

    def test_radix_reward_and_punish(self):
        model = RadixNet(seed=1, backend="python")
        model.train(TEXTS, epochs=1, batch_size=4)
        records = model.reward([TEXTS[0]], epochs=2, lr=0.2, batch_size=4, strength=3.0)  # strength is ignored
        self.assertEqual([r["phase"] for r in records], ["positive", "positive"])
        self.assertEqual(records[0]["act_lr"], 0.2 / 10)
        self.assertFalse(model.graph.inverted)
        records = model.punish(["zzz qqq"], epochs=1, lr=0.5, batch_size=4)
        self.assertEqual([r["phase"] for r in records], ["negative"])
        self.assertTrue(model.graph.inverted)


class TestCorrections(unittest.TestCase):
    """Learning from a diff: only the trigram nodes the two sentences disagree on move."""

    PAIRS = [
        ("the cat sit", "the cat sits"),
        ("he go to school", "he goes to school"),
        ("a apple a day", "an apple a day"),
        ("the cats is hungry", "the cats are hungry"),
        ("i have ate", "i have eaten"),
        ("", "hello there"),
        ("nothing to fix", "nothing to fix"),
        ("she walk to the shop yesterday", "she walked to the shop yesterday"),
        ("we was happy", "we were happy"),
        ("the mat on sat cat", "the cat sat on the mat"),
    ]

    def test_the_edits_rebuild_both_sentences(self):
        for wrong, right in self.PAIRS:
            with self.subTest(wrong=wrong):
                steps = diff.edits(wrong, right)
                self.assertEqual("".join(e.wrong for e in steps), wrong)
                self.assertEqual("".join(e.right for e in steps), right)
                for before, after in zip(steps, steps[1:]):
                    self.assertEqual((before.a1, before.b1), (after.a0, after.b0))  # end to end
                    self.assertNotEqual(before.op, after.op)  # runs of one kind are merged
                for step in steps:
                    if step.op == "equal":
                        self.assertEqual(step.wrong, step.right)

    def test_a_change_is_named_and_placed(self):
        self.assertEqual(diff.summary("nothing to fix", "nothing to fix"), [])
        self.assertEqual(
            diff.summary("the cats is hungry", "the cats are hungry"),
            [{"op": "replace", "wrong": "is", "right": "are"}],
        )
        # an insertion is an empty span on the side that lacks the text, where it belongs
        self.assertEqual(diff.changed_spans("the cat sit", "the cat sits"), ([(11, 11)], [(11, 12)]))

    def test_only_the_difference_moves(self):
        model = trained(epochs=2)
        model.train(["the cat sits on the mat"], epochs=1)
        wrong, right = "the cat sit on the mat", "the cat sits on the mat"
        before = list(model.graph.edge_reward)
        out = model.correct(wrong, right, strength=1.0, weight=1.0, reward=1.0, keep=0.0)
        self.assertEqual((out["edits"], out["penalised"], out["rewarded"], out["kept"]), (1, 1, 1, 0))
        moved = {
            e: model.graph.edge_reward[e] - (before[e] if e < len(before) else 0.0)
            for e in range(len(model.graph.edge_reward))
            if abs(model.graph.edge_reward[e] - (before[e] if e < len(before) else 0.0)) > 1e-12
        }
        self.assertEqual(len(moved), 2)  # keep=0: the step that wrote the wrong character, and the right one
        self.assertEqual(sorted(v > 0 for v in moved.values()), [False, True])
        blamed = [e for e, delta in moved.items() if delta < 0]
        self.assertEqual(blamed, [e for _prev, e in model._steps_over(model.encoder.encode(wrong), len(wrong), [(11, 11)])])

    def test_keep_spreads_a_smaller_reward_over_the_rest(self):
        model = trained(epochs=2)
        texts = model.meta["trained_texts"]
        out = model.correct("the cat sat on the log", "the cat sat on the mat", keep=0.25)
        self.assertGreater(out["kept"], 0)
        self.assertGreater(out["reward"], 0)
        self.assertGreater(out["penalty"], 0)
        self.assertEqual(model.meta["trained_texts"] - texts, 1)  # the correction is traversed once
        self.assertGreaterEqual(model.meta["feedback_passes"], 1)
        again = model.correct("the cat sat on the mat", "the cat sat on the mat")
        self.assertEqual((again["edits"], again["penalised"], again["rewarded"]), (0, 0, 0))

    def test_a_sentence_that_stopped_too_early_blames_the_end(self):
        model = trained(epochs=2)
        model.train(["the cat sat on the mat"], epochs=1)
        wrong, right = "the cat sat", "the cat sat on the mat"
        out = model.correct(wrong, right, keep=0.0)
        self.assertEqual(out["penalised"], 1)
        grams = model.encoder.encode(wrong)
        path = model.graph.node_path(grams)
        end_edge = model.graph.children[path[-2]][END]
        self.assertEqual([e for _prev, e in model._steps_over(grams, len(wrong), [(len(wrong), len(wrong))])], [end_edge])

    def test_short_and_empty_sentences_are_safe(self):
        model = trained(epochs=1)
        for wrong, right in [("", ""), ("ab", "ab"), ("", "the cat sits"), ("the cat sits", "")]:
            with self.subTest(wrong=wrong, right=right):
                model.correct(wrong, right)
        model.graph.check_invariants()

    def test_a_run_of_corrections_keeps_the_graph_sound(self):
        model = trained(epochs=2)
        for i in range(12):
            model.correct(f"the cat sit on the mat number {i}", f"the cat sits on the mat number {i}")
            model.graph.check_invariants()


class TestPathCounters(unittest.TestCase):
    """Correct / incorrect per *path*: the node that called the step decides what the step is worth."""

    def judged(self, corpus, good, bad, epochs=3):
        model = CountRewardNet(seed=1)
        model.train(corpus, epochs=epochs)
        model.reward([good])
        model.punish([bad])
        return model

    def walk(self, model, text):
        """``[(prev, parent, edge)]`` of a text: what called each step."""
        transitions = model.graph._trace(model.encoder.encode(text))[0]
        out, prev = [], START
        for i, (parent, edge) in enumerate(transitions):
            if i:
                prev = transitions[i - 1][0]
            out.append((prev, parent, edge))
        return out

    def test_training_counts_nothing_and_a_judgement_starts_the_table(self):
        model = CountRewardNet(seed=1)
        model.train(TEXTS, epochs=2)
        self.assertEqual(model.graph.path_totals()["contexts"], 0)  # a corpus is not a judgement
        model.reward(["the cat sat on the mat"])
        first = model.graph.path_totals()
        self.assertGreater(first["contexts"], 0)
        self.assertEqual(first["correct"], first["seen"])
        self.assertEqual(first["incorrect"], 0)
        model.punish(["the dog ate the bone"])
        second = model.graph.path_totals()
        self.assertGreater(second["incorrect"], 0)
        # a later training pass keeps the counters up to date without inventing contexts
        model.train(["the cat sat on the mat"], epochs=1)
        third = model.graph.path_totals()
        self.assertEqual(third["contexts"], second["contexts"])
        self.assertGreater(third["seen"], second["seen"])
        self.assertEqual((third["correct"], third["incorrect"]), (second["correct"], second["incorrect"]))

    def test_the_same_edge_is_right_after_one_word_and_wrong_after_another(self):
        # "a cat" passed and "the cat" failed, so the step into "sat" is the right move after one
        # word and the wrong one after the other - which no per-edge counter can tell apart
        good, bad = "a cat sat", "the cat sat"
        model = self.judged([good, bad, "a cat ran"], good, bad)
        graph = model.graph
        by_step = {(parent, edge): prev for prev, parent, edge in self.walk(model, good)}
        shared = [
            (edge, by_step[(parent, edge)], prev)
            for prev, parent, edge in self.walk(model, bad)
            if (parent, edge) in by_step and by_step[(parent, edge)] != prev and len(graph.children[parent]) > 1
        ]
        self.assertTrue(shared, "the two sentences should share a step, reached from different nodes, with a choice")
        edge, good_prev, bad_prev = shared[0]
        self.assertGreater(graph.path_term(good_prev, edge), 0)  # right in the sentence that passed
        self.assertLess(graph.path_term(bad_prev, edge), 0)  # wrong in the one that did not
        parent = graph.edge_parent[edge]
        plain = dict((e, cost) for _c, e, cost in graph.child_costs(parent))
        from_good = dict((e, cost) for _c, e, cost in graph.child_costs(parent, good_prev))
        from_bad = dict((e, cost) for _c, e, cost in graph.child_costs(parent, bad_prev))
        self.assertLess(from_good[edge], plain[edge])  # cheaper for the walk that was right here
        self.assertGreater(from_bad[edge], plain[edge])  # dearer for the walk that was wrong
        stats = graph.path_stats(good_prev, edge)
        self.assertEqual((stats["correct"], stats["incorrect"]), (1, 0))
        self.assertEqual(stats["correct_ratio"], 1.0)
        self.assertGreater(stats["seen_ratio"], 0.0)
        self.assertIsNone(graph.path_stats(good_prev, 10_000))

    def test_the_counters_survive_a_round_trip_and_the_predictions_with_them(self):
        good, bad = "a cat sat down", "the cat sat down"
        model = self.judged([good, bad], good, bad)
        before = model.graph.path_totals()
        again = CountRewardNet.from_dict(model.to_dict())
        self.assertEqual(again.graph.path_totals(), before)
        self.assertEqual(again.graph.weight_config()["path_scale"], 1.0)
        for prefix in ("a cat", "the cat", "a "):
            with self.subTest(prefix=prefix):
                self.assertEqual(again.predict(prefix, length=6).text, model.predict(prefix, length=6).text)
        # the same walk is priced the same after a reload
        steps = self.walk(model, good)
        reloaded = self.walk(again, good)
        self.assertEqual(
            [round(model.graph.path_term(p, e), 12) for p, _n, e in steps],
            [round(again.graph.path_term(p, e), 12) for p, _n, e in reloaded],
        )

    def test_compression_keeps_the_contexts_that_are_still_a_choice(self):
        good, bad = "a cat sat down", "the cat sat down"
        model = self.judged([good, bad], good, bad)
        before = model.graph.path_totals()
        merges = model.graph.compress()
        after = model.graph.path_totals()
        self.assertGreaterEqual(before["contexts"], after["contexts"])  # forced steps drop out
        self.assertLessEqual(after["correct"], before["correct"])
        model.graph.check_invariants()
        # and the graph still walks both sentences
        for text in (good, bad):
            self.assertIsNotNone(model.graph.node_path(model.encoder.encode(text)))
        self.assertGreater(merges, -1)

    def test_a_correction_marks_the_steps_it_moves(self):
        model = CountRewardNet(seed=1)
        model.train(["the cat sits on the mat", "the cat sat on the mat"], epochs=2)
        out = model.correct("the cat sit on the mat", "the cat sits on the mat")
        self.assertEqual((out["marked_incorrect"], out["marked_correct"]), (1, 1))
        totals = model.graph.path_totals()
        self.assertEqual((totals["correct"], totals["incorrect"]), (1, 1))
        # keep is 0 by default: a whole path is rewarded when the output was correct, not when it was corrected
        self.assertEqual(out["kept"], 0)

    def test_the_scale_can_be_turned_off(self):
        good, bad = "a cat sat", "the cat sat"
        model = self.judged([good, bad, "a cat ran"], good, bad)
        graph = model.graph
        prev, edge = next(k for k in graph.paths if len(graph.children[graph.edge_parent[k[1]]]) > 1)
        parent = graph.edge_parent[edge]
        with_paths = [cost for _c, _e, cost in graph.child_costs(parent, prev)]
        graph.configure(path_scale=0.0)
        self.assertEqual(graph.weight_config()["path_scale"], 0.0)
        without = [cost for _c, _e, cost in graph.child_costs(parent, prev)]
        self.assertEqual(without, [cost for _c, _e, cost in graph.child_costs(parent)])
        self.assertNotEqual(with_paths, without)


class TestNodeRatios(unittest.TestCase):
    """A node against the nodes around it: what share of its traffic and of its reward goes each way."""

    def branching(self):
        """A graph with a real choice in it, one arm rewarded and the other punished."""
        model = CountRewardNet(seed=1)
        model.train(["the cat sat on the mat", "a cat ran to the park", "the cat sat on the log"], epochs=2)
        model.correct("the cat ran to the mat", "the cat sat on the mat")
        return model

    def branch_node(self, model):
        """The node the two arms leave from."""
        graph = model.graph
        node = next(i for i, label in enumerate(graph.labels) if label == "at ")
        self.assertGreater(len(graph.children[node]), 1)
        return node

    def test_each_side_shares_out_the_traffic_it_carried(self):
        model = self.branching()
        row = model.graph.node_ratios(self.branch_node(model))
        for side, totals in (("from", "in_totals"), ("to", "out_totals")):
            rows = row[side]
            self.assertTrue(rows)
            self.assertAlmostEqual(sum(r["seen_ratio"] for r in rows), 1.0)
            self.assertEqual(sum(r["seen"] for r in rows), row[totals]["seen"])
            self.assertEqual(len(rows), row[totals]["edges"])
            seen = [r["seen"] for r in rows]  # most walked first
            self.assertEqual(seen, sorted(seen, reverse=True))

    def test_the_reward_share_is_signed_and_adds_up_to_the_side(self):
        model = self.branching()
        row = model.graph.node_ratios(self.branch_node(model))
        out = row["to"]
        rewarded = [r for r in out if r["reward"] > 0]
        punished = [r for r in out if r["reward"] < 0]
        self.assertTrue(rewarded and punished, "the correction should have moved one arm each way")
        self.assertTrue(all(r["reward_ratio"] > 0 for r in rewarded))
        self.assertTrue(all(r["reward_ratio"] < 0 for r in punished))
        mass = sum(abs(r["reward"]) for r in out)
        self.assertAlmostEqual(sum(abs(r["reward_ratio"]) for r in out), 1.0)
        self.assertAlmostEqual(mass, sum(abs(r["reward"]) for r in out))

    def test_the_judged_paths_land_on_the_way_they_were_walked(self):
        model = self.branching()
        row = model.graph.node_ratios(self.branch_node(model))
        right = next(r for r in row["to"] if r["label"] == "t sat")
        wrong = next(r for r in row["to"] if r["label"] == "t ran ")
        self.assertEqual((right["correct"], right["incorrect"]), (1, 0))
        self.assertEqual((wrong["correct"], wrong["incorrect"]), (0, 1))
        self.assertEqual((right["correct_ratio"], wrong["correct_ratio"]), (1.0, 0.0))
        # path_ratio is how much of the edge's traffic a judged context has been watching
        self.assertEqual(wrong["path_ratio"], wrong["path_seen"] / wrong["seen"])
        # the out-side as a whole: two of its three judged ways were walked in a correct answer
        self.assertEqual((row["out_totals"]["correct"], row["out_totals"]["incorrect"]), (2, 1))
        self.assertAlmostEqual(row["out_totals"]["correct_ratio"], 2 / 3)

    def test_an_unjudged_graph_has_the_shares_but_no_verdicts(self):
        model = CountRewardNet(seed=1)
        model.train(["the cat sat on the mat", "a cat ran to the park"], epochs=2)
        rows = model.node_ratios(limit=0)
        self.assertTrue(rows)
        for row in rows:
            for side in ("from", "to"):
                for r in row[side]:
                    self.assertEqual((r["correct"], r["incorrect"], r["path_seen"]), (0, 0, 0))
                    self.assertIsNone(r["correct_ratio"])
                    self.assertEqual(r["reward_ratio"], 0.0)
            self.assertIsNone(row["in_totals"]["correct_ratio"])

    def test_the_shares_are_of_the_side_not_of_the_visits(self):
        """A node a text starts on is entered without an in-edge, so its in-side is smaller than its visits."""
        model = CountRewardNet(seed=1)
        model.train(["the cat sat", "the cat ran"], epochs=2)
        graph = model.graph
        start = graph.children[0]  # START's children are the nodes texts begin on
        node = next(iter(start))
        row = graph.node_ratios(node)
        self.assertLessEqual(row["in_totals"]["seen"], row["visits"])
        self.assertAlmostEqual(sum(r["seen_ratio"] for r in row["from"]), 1.0)

    def test_the_table_is_most_visited_first_and_one_node_can_be_asked_for(self):
        model = self.branching()
        rows = model.node_ratios(limit=3)
        self.assertEqual(len(rows), 3)
        visits = [r["visits"] for r in rows]
        self.assertEqual(visits, sorted(visits, reverse=True))
        node = self.branch_node(model)
        one = model.node_ratios(limit=0, node=node)
        self.assertEqual([r["node"] for r in one], [node])
        self.assertIsNone(model.graph.node_ratios(len(model.graph.labels) + 5))

    def test_back_is_an_ordinary_row(self):
        """BACK needs no special case: its edge says what share of the walks leaving here have learned to go round."""
        model = CountRewardNet(seed=1)
        model.train(["the cat sat on the mat", "a cat ran to the park", "the cat sat on the log"], epochs=2)
        graph = model.graph
        node = self.branch_node(model)
        went = next(iter(graph.children[node]))
        graph.observe_back(node, went=went, amount=1.0)
        row = graph.node_ratios(node)
        back = next(r for r in row["to"] if r["node"] == BACK)
        looped = next(r for r in row["to"] if r["node"] == went)
        self.assertGreater(back["seen_ratio"], 0.0)
        self.assertGreater(back["reward_ratio"], 0.0)  # the hand-over earned it
        self.assertLess(looped["reward_ratio"], 0.0)  # the step it was about to loop through paid for it
        self.assertAlmostEqual(sum(r["seen_ratio"] for r in row["to"]), 1.0)

    def test_a_dead_node_has_no_ratios(self):
        model = self.branching()
        graph = model.graph
        dead = next((i for i, alive in enumerate(graph.alive) if not alive), None)
        if dead is None:
            self.skipTest("this corpus compressed nothing away")
        self.assertIsNone(graph.node_ratios(dead))


class TestLeastPunishedTraversal(unittest.TestCase):
    """The second traversal: a walk ranked by what went wrong on it (SPEC-LeastPunished.md)."""

    TEXTS = ["the cat sat on the mat", "the cat sat on the log"]

    def trained(self) -> CountRewardNet:
        model = CountRewardNet(seed=0)
        model.train(self.TEXTS, epochs=3)
        return model

    def test_the_names_it_answers_to(self):
        for name in ("", "reward", "Rewards", "cost"):
            self.assertEqual(parse_traversal(name), REWARD, name)
        for name in ("least-punished", "LEAST_PUNISHED", "blame"):
            self.assertEqual(parse_traversal(name), LEAST_PUNISHED, name)
        # "punishment" is the other punishment traversal (radixnet.penalty), which prices a
        # step rather than ranking the walks: it is never an alias of this one
        for name in ("punishment", "penalty"):
            self.assertEqual(parse_traversal(name), PUNISHMENT, name)
            self.assertNotEqual(parse_traversal(name), LEAST_PUNISHED, name)
        with self.assertRaises(ValueError):
            parse_traversal("sideways")

    def test_the_filter_keeps_the_cleanest_children(self):
        steps = [(4, 0, 1.0, 0.0), (5, 1, 2.0, 0.0)]
        self.assertEqual(least_punished(steps), steps, "nothing punished, nothing dropped")
        steps[0] = (4, 0, 1.0, 0.5)
        self.assertEqual([item[0] for item in least_punished(steps)], [5])
        steps[1] = (5, 1, 2.0, 0.5 + PUNISH_TOLERANCE / 2)  # the same penalty, a bit or two apart
        self.assertEqual(len(least_punished(steps)), 2)
        steps[1] = (5, 1, 2.0, math.inf)
        self.assertEqual([item[0] for item in least_punished(steps)], [4])
        self.assertEqual(least_punished([]), [])

    def test_it_is_the_old_search_until_something_is_punished(self):
        model = self.trained()
        for prefix in ("the ", "the cat", "sat on"):
            with self.subTest(prefix=prefix):
                a = model.predict(prefix, length=8, k=3, mode="beam")
                b = model.predict(prefix, length=8, k=3, mode="beam", traversal="least-punished")
                self.assertEqual(a.full_text, b.full_text)
                self.assertEqual(a.cost, b.cost)
                self.assertEqual(a.expanded, b.expanded)
                self.assertEqual([r.full_text for r in a.top], [r.full_text for r in b.top])
                self.assertEqual([r.full_text for r in a.bottom], [r.full_text for r in b.bottom])

    def test_it_leaves_a_step_that_was_judged_wrong(self):
        """Rewarded five times over, punished once - and not walked again."""
        model = self.trained()
        model.reward(["the cat sat on the mat"], epochs=1, strength=5.0)
        model.punish(["the cat sat on the mat"], epochs=1, strength=1.0)
        prefix = "the cat sat on the "
        by_reward = model.predict(prefix, length=6, k=3, mode="beam")
        by_blame = model.predict(prefix, length=6, k=3, mode="beam", traversal="least-punished")
        self.assertEqual(by_reward.full_text, "the cat sat on the mat")
        self.assertEqual(by_blame.full_text, "the cat sat on the log")
        self.assertGreater(by_blame.cost, by_reward.cost)  # dearer, and taken anyway
        self.assertEqual(by_blame.punish, 0.0)
        blamed = next(r for r in by_blame.top if r.full_text == "the cat sat on the mat")
        self.assertGreater(blamed.punish, 0.0, "the blamed walk should still be offered, marked with its blame")
        self.assertEqual(by_blame.to_dict()["traversal"], "least-punished")
        self.assertEqual(by_reward.to_dict()["traversal"], "reward")

    def test_a_reward_does_not_buy_the_blame_off(self):
        model = self.trained()
        model.punish(["the cat sat on the mat"], epochs=1, strength=1.0)
        model.reward(["the cat sat on the mat"], epochs=1, strength=50.0)
        graph = model.graph
        judged = [(prev, edge) for (prev, edge), row in graph.paths.items() if row[2] > 0]
        self.assertTrue(judged, "the punished pass recorded no context")
        for prev, edge in judged:
            self.assertEqual(graph.edge_punishment(edge), 0.0, "the edge's own reward is positive now")
            self.assertGreater(graph.step_punishment(prev, edge), 0.0, "a reward bought off the blame")

    def test_a_graph_with_no_record_of_failure_has_nothing_against_anything(self):
        model = RadixNet(seed=0)
        model.train(self.TEXTS, epochs=1)
        graph = model.graph
        self.assertEqual(graph.edge_punishment(0), 0.0)
        self.assertEqual(graph.step_punishment(START, 0), 0.0)
        self.assertTrue(all(step[3] == 0.0 for step in graph.child_steps(START, None)))
        with self.assertRaises(ValueError):  # and it says so rather than pretending
            model.predict("the", length=4, traversal="least-punished")

    def test_a_sampled_walk_follows_the_blame_too(self):
        model = self.trained()
        model.punish(["the cat sat on the mat"], epochs=2, strength=3.0)
        walks = [
            model.predict("the cat sat on the ", length=6, mode="sample", temperature=0.0,
                          traversal="least-punished").full_text
            for _ in range(5)
        ]
        self.assertEqual(set(walks), {"the cat sat on the log"}, walks)


class TestSplitsAndMergesKeepTheModel(unittest.TestCase):
    """A split or a merge must not change what the count model hangs on its edges: the recent shares, the
    verdicts and the prices - and the bottom beam of the least-punished traversal finds the most punished paths."""

    T1, T2 = "abcdefghij klm", "abcdefghij xyz"  # share "abcdefghij ", then part
    T3R, T3W = "mnop tuv", "mnop tuw"  # share nothing with the other two

    def fork(self, auto_compress, window=None):
        """The node where two texts part, after one is trained fifty times and the other once: ``(graph, node,
        {first three units of the child: (probability, traversals, windowed traversals)})``."""
        model = CountRewardNet(seed=1) if window is None else CountRewardNet(seed=1, window=window)
        model.train(["abcdefghij klmnop"], epochs=50, auto_compress=auto_compress)
        model.train(["abcdefghij qrstuv"], epochs=1, auto_compress=auto_compress)
        g = model.graph
        forks = [p for p in g.alive_nodes() if p >= FIRST and len(g.children[p]) == 2]
        self.assertEqual(len(forks), 1, forks)
        probs = dict(g.child_probs(forks[0]))
        ways = {
            g.labels[c][:3]: (probs[c], g.edge_count[e], g.window_edge_count[e])
            for c, e in g.children[forks[0]].items()
        }
        return model, forks[0], ways

    def judged_pair(self):
        model = CountRewardNet(seed=1)
        model.train([self.T1] * 5 + [self.T3R], epochs=1)
        for _ in range(3):
            model.correct(self.T2, self.T1)  # the model wrote xyz, the teacher wrote klm: blame the xyz step
        model.correct(self.T3W, self.T3R, strength=0.3)  # a light blame on the tuw step
        return model

    def halved(self, *texts):
        """A model of ``texts`` whose shared prefix the dynamic window has halved and holds apart."""
        model = CountRewardNet(seed=1)
        model.train(list(texts), epochs=2)
        model.configure_window(on=True, top=8, floor=8, size=8, auto=False)
        step = model.window_step(1)
        self.assertEqual(step["splits"], 1)
        return model

    @staticmethod
    def ranking(model, traversal, k=4):
        p = model.predict("", length=0, to_end=True, max_length=40, k=k, traversal=traversal)
        rows = lambda results: [(r.text, round(r.punish, 6), round(r.cost, 9)) for r in results]  # noqa: E731
        return rows(p.top), rows(p.bottom)

    def test_a_split_hands_the_bridge_its_window_history(self):
        _model, _p, compressed = self.fork(True)
        _model, _p, plain = self.fork(False)
        # the bridge carries the fifty windowed traversals of the node it came out of, as the chain's edge does,
        # so the compressed graph and the chain price the new branch alike
        for key in ("j k", "j q"):
            with self.subTest(key=key):
                self.assertEqual(compressed[key][1:], plain[key][1:])
                self.assertAlmostEqual(compressed[key][0], plain[key][0], places=12)
        self.assertEqual(compressed["j k"][1:], (50, 50))
        self.assertLess(compressed["j q"][0], 0.05, "one traversal in fifty-one is not a fifth of the way on")

    def test_the_rewritten_window_keeps_its_size_and_survives_the_file(self):
        model, p, ways = self.fork(True, window=60)
        g = model.graph
        self.assertEqual(g.window_traversals, 60)
        self.assertEqual(sum(g.window_edge_count), 60)  # every event in the window is counted once
        self.assertGreater(ways["j k"][2], 0, "the bridge holds part of the window")
        self.assertLess(ways["j q"][0], 0.06)
        again = CountRewardNet.from_dict(model.to_dict())
        self.assertEqual(again.graph.window_traversals, 60)
        self.assertEqual(again.predict("abcdefghij ", length=4).text, model.predict("abcdefghij ", length=4).text)
        g.check_invariants()

    def test_a_split_hands_the_bridge_no_verdict(self):
        model = self.judged_pair()
        g = model.graph
        before = {trav: self.ranking(model, trav) for trav in ("reward", "least-punished")}
        self.assertEqual([r[0] for r in before["least-punished"][0]][:2], [self.T1, self.T3R])
        totals = g.path_totals()
        self.assertEqual(g.split_window(8), 1)  # halves the shared prefix "abcdefghij "
        a = next(n for n in g.alive_nodes() if n >= FIRST and g.labels[n] == "abcdefg")
        bridge = next(iter(g.children[a].values()))
        self.assertEqual(g.edge_paths(bridge), (0, 0, 0), "a forced step carries no verdict")
        self.assertEqual(g.path_totals(), totals, "the split moved the verdicts, it did not add any")
        # the same walks, the same blame, the same prices - a split is invisible to both traversals
        after = {trav: self.ranking(model, trav) for trav in ("reward", "least-punished")}
        self.assertEqual(after, before)
        g.check_invariants()

    def test_a_merge_keeps_the_verdicts_and_the_prices(self):
        model = self.halved(self.T1, self.T2)
        g = model.graph
        model.correct(self.T2, self.T1)  # verdicts on the fork, which the first half now calls
        model.punish([self.T1], strength=0.5)
        a = next(n for n in g.alive_nodes() if n >= FIRST and g.labels[n] == "abcdefg")
        b = next(iter(g.children[a]))
        kept = {e: list(row) for (prev, e), row in g.paths.items() if prev == a and g.edge_parent[e] == b}
        self.assertEqual(len(kept), 2, "both ways out of the fork were judged")
        before = {trav: self.ranking(model, trav) for trav in ("reward", "least-punished")}
        model.configure_window(on=False)
        self.assertEqual(model.compress(), 1)
        # the fork's contexts are keyed by the merged node's caller now, counters intact
        self.assertEqual(
            {e: list(row) for (prev, e), row in g.paths.items() if prev == START and g.edge_parent[e] == a}, kept
        )
        after = {trav: self.ranking(model, trav) for trav in ("reward", "least-punished")}
        self.assertEqual(after, before, "a merge changed what the model predicts")
        g.check_invariants(compressed=True)

    def test_a_merge_carries_what_the_forced_step_alone_was_taught(self):
        model = self.halved(self.T1)
        g = model.graph
        a = next(iter(g.children[START]))  # the first half of the only text, held apart from the second
        bridge = next(iter(g.children[a].values()))
        into = g.children[START][a]
        g.add_reward([bridge], -1.0)  # blame the forced step alone
        self.assertEqual(self.ranking(model, "least-punished", k=1)[0][0][1], 1.0)
        totals = g.total_reward()
        model.configure_window(on=False)
        self.assertEqual(model.compress(), 1)
        self.assertEqual(g.edge_reward[into], -1.0, "the blame moved onto the step into the merged node")
        self.assertEqual(g.total_reward(), totals)
        self.assertEqual(self.ranking(model, "least-punished", k=1)[0][0][1], 1.0)
        # a pass that penalised every edge of the path moves nothing: the step into the node explains it
        model = self.halved(self.T1)
        g = model.graph
        model.punish([self.T1], strength=0.5)
        a = next(iter(g.children[START]))
        into = g.children[START][a]
        prices = self.ranking(model, "reward", k=1)
        model.configure_window(on=False)
        self.assertEqual(model.compress(), 1)
        self.assertEqual(g.edge_reward[into], -0.5)
        self.assertEqual(self.ranking(model, "reward", k=1), prices)

    def test_the_bottom_beam_finds_the_most_punished_paths(self):
        model = self.judged_pair()
        top, bottom = self.ranking(model, "least-punished", k=1)
        self.assertEqual(top[0][0], self.T1)
        self.assertEqual([r[0] for r in bottom], [self.T2], "the walk the corrections blamed three times over")
        self.assertGreater(bottom[0][1], 4.0)
        _top, by_cost = self.ranking(model, "reward", k=1)
        self.assertEqual([r[0] for r in by_cost], [self.T2], "by cost it is the dearest walk too")
        _top, two = self.ranking(model, "least-punished", k=2)
        self.assertEqual([r[0] for r in two], [self.T2, self.T3W], "most punished first")


if __name__ == "__main__":
    unittest.main()
