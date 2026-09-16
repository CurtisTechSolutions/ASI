"""Tests for the resonant model: the phase, the metacognitive layer and the phase-aware search.

Covers :mod:`radixnet.resonance` (``ResonantGraph`` / ``ResonantNet``),
:mod:`radixnet.metacog` (``MetaLayer``) and :mod:`radixnet.phasesearch`, plus
the kind plumbing that carries them through ``load_model``, the CLI and the
HTTP API.
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.beam import Prediction  # noqa: E402
from radixnet.encoding import Encoder  # noqa: E402
from radixnet.graph import BACK, FIRST, START  # noqa: E402
from radixnet.metacog import ABORT, ACTIONS, ESCAPE, RIDE, MetaLayer, cycle_signature  # noqa: E402
from radixnet.phasesearch import phase_dijkstra, phase_kbest  # noqa: E402
from radixnet.search import onward  # noqa: E402
from radixnet.model import RadixNet, load_model, model_class, model_from_dict, model_kinds, new_model  # noqa: E402
from radixnet.resonance import (  # noqa: E402
    RESONANT_MODEL_FORMAT,
    TAU,
    ResonantGraph,
    ResonantNet,
    trigram_phase,
)

TEXTS = [
    "the sun rises in the east",
    "the sun sets in the west",
    "the moon rises at night",
    "the earth orbits the sun",
]
GARBAGE = ["qx zz vv qq", "zzzz xxxx vvvv", "!!! ??? ;;;"]


def grams(text):
    return Encoder().encode(text)


class TestTrigramPhase(unittest.TestCase):
    def test_phase_is_stable_and_in_range(self):
        for t in ("abc", "the", " wo", "ééé"):
            value = trigram_phase(t)
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, TAU)
            self.assertEqual(value, trigram_phase(t))  # cached, and a pure function of the text

    def test_different_trigrams_get_different_phases(self):
        self.assertNotEqual(trigram_phase("abc"), trigram_phase("abd"))


class TestPhase(unittest.TestCase):
    def test_sentinels_never_advance_the_phase(self):
        # "<s>", "</s>" and "<back>" are all long enough to look like ordinary labels
        g = ResonantGraph(kick_scale=1.0)
        self.assertEqual([g.advance[n] for n in range(FIRST)], [0] * FIRST)

    def test_phase_is_the_text_s_own_and_survives_compression(self):
        for kick in (0.0, 0.5, 1.0):
            with self.subTest(kick_scale=kick):
                g = ResonantGraph(seed=0, buckets=8, kick_scale=kick)
                text = "abcdefghijklm"
                g.observe_sequence(grams(text))
                walk = sum(g.advance[n] for n in g.alive_nodes()) % g.buckets
                self.assertEqual(walk, g.text_bucket(text))
                self.assertGreater(g.compress(), 0)
                merged = sum(g.advance[n] for n in g.alive_nodes()) % g.buckets
                self.assertEqual(merged, walk, "compression must not move the phase")

    def test_split_preserves_the_phase(self):
        g = ResonantGraph(seed=0, buckets=8, kick_scale=1.0)
        g.observe_sequence(grams("abcdefgh"))
        g.compress()
        node = next(n for n in g.alive_nodes() if n >= FIRST and len(g.labels[n]) > 4)
        before = g.advance[node]
        a_id, b_id = g.split(node, 1)
        self.assertEqual((g.advance[a_id] + g.advance[b_id]) % g.buckets, before)

    def test_kick_scale_zero_is_a_pure_clock(self):
        g = ResonantGraph(buckets=8, period=8.0, kick_scale=0.0)
        self.assertEqual(g.trigram_advance("abc"), 1)      # one bucket per character
        self.assertEqual(g.trigram_advance("zzz"), 1)      # the content does not matter
        g2 = ResonantGraph(buckets=8, period=8.0, kick_scale=1.0)
        advances = {g2.trigram_advance(t) for t in ("abc", "bcd", "cde", "def", "efg")}
        self.assertGreater(len(advances), 1, "with a kick the content moves the phase")

    def test_text_bucket_is_the_sum_of_the_trigram_advances(self):
        g = ResonantGraph(buckets=8, kick_scale=1.0)
        text = "hello world"
        expected = sum(g.trigram_advance(t) for t in grams(text)) % 8
        self.assertEqual(g.text_bucket(text), expected)


class TestResonantGraph(unittest.TestCase):
    def setUp(self):
        self.g = ResonantGraph(seed=0, buckets=8, kick_scale=0.0)

    def test_activations_are_one_so_the_base_scores_are_the_amplitudes(self):
        self.g.observe_sequence(grams("hello there"))
        self.g.recompute_weights()
        for n in self.g.alive_nodes():
            self.assertAlmostEqual(self.g.activation_of(n), 1.0)
        p = self.g.trigram_index["hel"][0]
        for (c, score), (c2, _e, w) in zip(self.g.child_scores(p), [(c, e, self.g.edge_w[e]) for c, e in self.g.children[p].items()]):
            self.assertEqual(c, c2)
            self.assertAlmostEqual(score, w)

    def test_coherence_grows_with_consistent_evidence_and_is_shrunk(self):
        self.g.observe_sequence(grams("hello there"))
        e = next(iter(self.g.children[START].values()))
        self.assertEqual(self.g.edge_coherence(e), 0.0)
        self.g.record_traversal(e, 0)
        one = self.g.edge_coherence(e)
        self.assertAlmostEqual(one, 1.0 / (1.0 + self.g.concentration))
        for _ in range(19):
            self.g.record_traversal(e, 0)
        self.assertGreater(self.g.edge_coherence(e), one)
        self.assertLess(self.g.edge_coherence(e), 1.0)

    def test_an_edge_taken_at_every_phase_is_incoherent(self):
        self.g.observe_sequence(grams("hello there"))
        e = next(iter(self.g.children[START].values()))
        for b in range(self.g.buckets):
            for _ in range(5):
                self.g.record_traversal(e, b)
        self.assertLess(self.g.edge_coherence(e), 0.05)

    def test_resonance_peaks_in_phase_and_cancels_in_antiphase(self):
        self.g.observe_sequence(grams("hello there"))
        e = next(iter(self.g.children[START].values()))
        for _ in range(50):
            self.g.record_traversal(e, 0)
        self.assertGreater(self.g.resonance(e, 0), 0.8)
        self.assertLess(self.g.resonance(e, self.g.buckets // 2), -0.8)

    def test_invert_rotates_the_mean_phase_by_pi(self):
        self.g.observe_sequence(grams("hello there"))
        e = next(iter(self.g.children[START].values()))
        for _ in range(10):
            self.g.record_traversal(e, 2)
        mu = self.g.edge_mu(e)
        self.g.invert()
        self.assertAlmostEqual((self.g.edge_mu(e) - mu) % TAU, math.pi, places=6)
        self.assertTrue(self.g.inverted)

    def test_rotate_and_sharpen(self):
        self.g.observe_sequence(grams("hello there"))
        e = next(iter(self.g.children[START].values()))
        for _ in range(10):
            self.g.record_traversal(e, 1)
        self.assertGreater(self.g.edge_coherence(e), 0.0)
        self.g.sharpen([e], 0.0)
        self.assertEqual(self.g.edge_coherence(e), 0.0)  # fully decohered
        for _ in range(10):
            self.g.record_traversal(e, 1)
        self.g.sharpen([e], 100.0)
        self.assertLessEqual(self.g.edge_coherence(e), 1.0)  # saturates, never exceeds 1
        mu = self.g.edge_mu(e)
        self.g.rotate([e], 1.0)
        self.assertAlmostEqual((self.g.edge_mu(e) - mu) % TAU, math.pi, places=6)

    def test_going_round_is_learned_in_this_model_s_own_currency(self):
        """``observe_back`` must not nudge a weight by hand: the next recompute would erase it."""
        net = ResonantNet(seed=0)
        net.train(["the cat sat down", "a big cat ran away"], epochs=4)
        graph = net.graph
        node = graph.trigram_index["at "][0]
        children = list(graph.children[node])
        before = dict(graph.child_probs_at(node, 0))
        graph.observe_back(node, went=children[0], instead=children[1], amount=2.0)
        after = dict(graph.child_probs_at(node, 0))
        self.assertIn(BACK, graph.children[node])
        self.assertLess(after[children[0]], before[children[0]])   # the step it looped through
        self.assertGreater(after[children[1]], before[children[1]])  # the step it took instead
        graph.recompute_weights()
        self.assertAlmostEqual(dict(graph.child_probs_at(node, 0))[BACK], after[BACK], places=9)
        self.assertIsNotNone(graph.back_cost(node))

    def test_a_hand_over_is_counted_without_a_phase(self):
        """A voice that backed out walked outside this search and cannot say which phase it was in."""
        net = ResonantNet(seed=0)
        net.train(["the cat sat down", "a big cat ran away"], epochs=4)
        graph = net.graph
        node = graph.trigram_index["at "][0]
        edge = graph.observe_back(node, amount=2.0)
        self.assertEqual(graph.edge_coherence(edge), 0.0)
        scores = [dict((c, s) for c, _e, s in graph.child_scores_at(node, b))[BACK] for b in range(graph.buckets)]
        self.assertEqual(len({round(v, 9) for v in scores}), 1, "BACK must score the same at every phase")

    def test_the_search_hands_over_where_the_model_expects_a_loop(self):
        """``search.onward``: when BACK is the cheapest child, that branch offers nothing."""
        net = ResonantNet(seed=0)
        net.train(["the cat sat down"], epochs=4)
        before = net.predict("the ", length=12, mode="beam", k=1).text
        self.assertTrue(before)
        node = net.graph.trigram_index["at "][0]
        for _ in range(40):
            net.graph.observe_back(node, amount=2.0)  # this node goes round, over and over
        self.assertIsNotNone(net.graph.back_cost(node))
        self.assertNotEqual(net.predict("the ", length=12, mode="beam", k=1).text, before)

    def test_back_probability_and_observe_onward(self):
        net = ResonantNet(seed=0)
        net.train(TEXTS, epochs=3)
        graph = net.graph
        node = graph.trigram_index["the"][0]
        self.assertEqual(graph.back_probability(node), 0.0)
        self.assertFalse(graph.observe_onward(node), "nothing to push back on before a hand-over")
        for _ in range(4):
            graph.observe_back(node, amount=1.0)
        risen = graph.back_probability(node)
        self.assertGreater(risen, 0.0)
        self.assertTrue(graph.observe_onward(node, 2.0))
        self.assertLess(graph.back_probability(node), risen, "going round was right: the hand-over comes back down")
        self.assertFalse(graph.observe_onward(START), "a sentinel has no hand-over to teach")

    def test_metacognition_overrules_the_back_reflex_only_where_it_remembers(self):
        """``BACK`` hands the branch over; a layer that has seen these cycles and rode them overrules it."""
        model = ResonantNet(seed=0)
        model.train(["lol lol lol lol lol"], epochs=3)
        graph = model.graph
        node = graph.trigram_index["ol "][0]
        before = [r.text for r in model.predict("lol ", length=12, mode="kbest", k=3).top]
        for _ in range(8):
            graph.observe_back(node, amount=1.0)  # what a voice backing out of this node reports
        self.assertEqual(onward(graph.child_costs(node)), [], "the reflex must be vetoing for this to mean anything")
        self.assertTrue(model.metacog.rides(graph.labels[node]))
        overruled = [r.text for r in model.predict("lol ", length=12, mode="kbest", k=3).top]
        self.assertTrue(any(t.count("lol") > 1 for t in overruled), overruled)
        self.assertEqual(len(overruled), len(before))

        forgotten = ResonantNet.from_dict(model.to_dict())
        forgotten.metacog.scores.clear()
        stands = [r.text for r in forgotten.predict("lol ", length=12, mode="kbest", k=3).top]
        self.assertFalse(any(t.count("lol") > 1 for t in stands),
                         f"with no memory of these cycles the hand-over must stand, got {stands}")

    def test_teaching_back_from_the_corpus_is_off_by_default(self):
        """It is a dial, not a default: the measurement is in DESIGN section 30.3."""
        model = ResonantNet(seed=0)
        self.assertFalse(model.teach_back)
        model.train(TEXTS, epochs=3)
        self.assertEqual([n for n in model.graph.alive_nodes() if BACK in model.graph.children[n]], [])

    def test_teaching_back_from_the_corpus_stays_under_the_ceiling(self):
        model = ResonantNet(seed=0, teach_back=True, back_strength=0.25, back_ceiling=0.10)
        model.train(["lol lol lol lol", "the cat sat down", "a big cat ran away"], epochs=4)
        graph = model.graph
        taught = [n for n in graph.alive_nodes() if BACK in graph.children[n]]
        self.assertTrue(taught, "declining a cycle must reach BACK when the dial is on")
        for node in taught:
            self.assertNotEqual(onward(graph.child_costs(node)), [],
                                f"{graph.labels[node]!r} reached the veto: observation must not hand branches over")
        with self.assertRaises(ValueError):
            ResonantNet(seed=0, back_ceiling=1.5)
        with self.assertRaises(ValueError):
            ResonantNet(seed=0, back_strength=-1.0)

    def test_configure_rephases_every_node(self):
        g = ResonantGraph(seed=0, buckets=8, kick_scale=0.0)
        g.observe_sequence(grams("hello there world"))
        self.assertTrue(all(g.advance[n] == g.label_advance(g.labels[n]) for n in g.alive_nodes() if n >= FIRST))
        config = g.configure(kick_scale=1.0, buckets=16)
        self.assertEqual((config["kick_scale"], config["buckets"]), (1.0, 16))
        for n in g.alive_nodes():
            self.assertEqual(g.advance[n], 0 if n < FIRST else g.label_advance(g.labels[n]))
        with self.assertRaises(ValueError):
            g.configure(nonsense=1.0)
        with self.assertRaises(ValueError):
            g.configure(period=0.0)

    def test_probabilities_depend_on_the_phase(self):
        """A node reached from several contexts: phase-free they are one distribution, per phase they are not."""
        net = ResonantNet(seed=0, buckets=8, kick_scale=0.0, resonance_scale=2.0)
        net.train(["the cat sat down", "a big cat ran away"], epochs=6)
        graph = net.graph
        node = graph.trigram_index["at "][0]
        self.assertGreater(len(graph.children[node]), 1)
        free = dict(graph.child_probs(node))
        self.assertAlmostEqual(min(free.values()), max(free.values()), places=6)  # nothing to tell them apart
        by_phase = [dict(graph.child_probs_at(node, b)) for b in range(graph.buckets)]
        for row in by_phase:
            self.assertAlmostEqual(sum(row.values()), 1.0, places=6)
        for child in free:
            spread = {round(row[child], 6) for row in by_phase}
            self.assertGreater(len(spread), 1, "the phase must move every child's probability")
            self.assertGreater(max(spread) - min(spread), 0.2)


class TestMetaLayer(unittest.TestCase):
    def test_untrained_is_uniform(self):
        layer = MetaLayer()
        for lp in layer.log_policy("anything"):
            self.assertAlmostEqual(lp, -math.log(len(ACTIONS)))

    def test_evidence_is_add_one_smoothed_counts(self):
        layer = MetaLayer()
        sig = cycle_signature("lol", 2)
        for _ in range(9):
            layer.observe(sig, RIDE)
        # exp(logit) == 1 + score, so the softmax is (count + 1) / (total + 3)
        self.assertAlmostEqual(math.exp(layer.log_policy(sig)[0]), 10.0 / 12.0, places=6)
        self.assertEqual(layer.decide(sig), RIDE)

    def test_punish_and_invert(self):
        layer = MetaLayer()
        sig = cycle_signature("aaa", 1)
        layer.observe(sig, RIDE, 5)
        self.assertEqual(layer.decide(sig), RIDE)
        layer.punish(sig, RIDE, 10)
        self.assertNotEqual(layer.decide(sig), RIDE)
        layer2 = MetaLayer()
        layer2.observe(sig, ABORT, 5)
        layer2.invert()
        self.assertNotEqual(layer2.decide(sig), ABORT)

    def test_unseen_signature_falls_back_to_the_prior(self):
        layer = MetaLayer()
        for _ in range(10):
            layer.observe(cycle_signature("abc", 3), ESCAPE)
        self.assertEqual(layer.decide(cycle_signature("zzz", 1)), ESCAPE)

    def test_signature_clamps_long_loops_and_keys_on_the_trigram(self):
        self.assertEqual(cycle_signature("hello world", 2), "hel:2")
        self.assertEqual(cycle_signature("abc", 99), cycle_signature("abc", 1000))

    def test_round_trip(self):
        layer = MetaLayer()
        layer.observe(cycle_signature("lol", 2), RIDE, 3)
        layer.observe(cycle_signature("xyz", 1), ABORT, 2)
        back = MetaLayer.from_dict(layer.to_dict())
        self.assertEqual(len(back), len(layer))
        self.assertEqual((back.back_scale, back.prior_weight), (layer.back_scale, layer.prior_weight))
        self.assertEqual(back.observed, layer.observed)
        for sig in ("lol:2", "xyz:1", "unseen:1"):
            self.assertEqual(back.log_policy(sig), layer.log_policy(sig))
        self.assertEqual(len(MetaLayer.from_dict(None)), 0)

    def test_the_prior_carries_the_weight_it_earned_about_one_cycle(self):
        """A cycle never met must not be judged with the summed confidence of every cycle that was."""
        layer = MetaLayer(prior_weight=2.0)
        for _ in range(50):
            layer.observe(cycle_signature("abc", 2), RIDE)
        self.assertEqual(layer.decide("unseen:1"), RIDE)  # the prior's direction is kept
        seen = math.exp(layer.log_policy(cycle_signature("abc", 2))[0])
        unseen = math.exp(layer.log_policy("unseen:1")[0])
        self.assertGreater(seen, 0.9)          # 50 observations of this very cycle: confident
        self.assertLess(unseen, 0.7)           # the same direction, held far more loosely
        raw = MetaLayer(prior_weight=0.0)      # 0 keeps the prior raw, as it was before
        for _ in range(50):
            raw.observe(cycle_signature("abc", 2), RIDE)
        self.assertAlmostEqual(math.exp(raw.log_policy("unseen:1")[0]), seen, places=6)

    def test_back_is_evidence_against_riding(self):
        layer = MetaLayer()
        for _ in range(4):
            layer.observe(cycle_signature("lol", 2), RIDE)
        signature = cycle_signature("lol", 2)
        self.assertEqual(layer.decide(signature), RIDE)
        self.assertEqual(layer.decide(signature, back=0.95), RIDE, "4 observed rides outrank the node's reflex")
        thin = MetaLayer()
        thin.observe(signature, RIDE)
        self.assertEqual(thin.decide(signature), RIDE)
        self.assertNotEqual(thin.decide(signature, back=0.95), RIDE, "one observation does not")
        self.assertGreater(layer.cost(signature, RIDE, back=0.95), layer.cost(signature, RIDE))
        self.assertEqual(MetaLayer(back_scale=0.0).decide(signature, back=1.0), RIDE)  # the dial turns it off

    def test_about_and_rides_aggregate_a_node_s_cycles(self):
        layer = MetaLayer()
        layer.observe(cycle_signature("lol", 2), RIDE, 3)
        layer.observe(cycle_signature("lol", 5), RIDE, 2)
        layer.observe(cycle_signature("lol", 5), ABORT, 1)
        layer.observe(cycle_signature("zzz", 2), ESCAPE, 9)
        self.assertEqual(layer.about("lol lol"), [5.0, 0.0, 1.0])
        self.assertTrue(layer.rides("lol lol"))
        self.assertFalse(layer.rides("zzz zzz"))
        self.assertFalse(layer.rides("qqq"), "a node it knows nothing about is not a node it rides")

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ValueError):
            MetaLayer().observe("sig", "dance")


class TestResonantNet(unittest.TestCase):
    def setUp(self):
        self.model = ResonantNet(seed=0)
        self.model.train(TEXTS, epochs=3)

    def test_training_records(self):
        record = self.model.history[-1]
        for field in ("epoch", "loss", "perplexity", "nodes", "edges", "trigrams", "transitions", "cycles",
                      "signatures", "seconds", "skipped_short"):
            self.assertIn(field, record)
        self.assertEqual(self.model.meta["epochs_total"], 3)
        self.assertEqual(self.model.meta["trained_texts"], len(TEXTS))

    def test_loss_falls_as_the_phases_lock(self):
        losses = [r["loss"] for r in self.model.history]
        self.assertLess(losses[-1], losses[0])

    def test_short_texts_are_skipped(self):
        model = ResonantNet(seed=0)
        records = model.train(["ab", "hello there"], epochs=1)
        self.assertEqual(records[0]["skipped_short"], 1)

    def test_k_best_with_one_label_per_state_is_dijkstra(self):
        """``k = 1`` must settle the same states, expand the same number of times and give the same walk."""
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=3)
        graph = model.graph
        for prefix in ("the ", "a ", ""):
            with self.subTest(prefix=prefix):
                node, offset, _lead = model._prefix_start(prefix)
                bucket = graph.text_bucket(prefix) if prefix else 0
                one = phase_dijkstra(graph, node, offset, bucket, min_chars=0, max_chars=60, to_end=True)
                (best,), expanded = phase_kbest(
                    graph, None, node, offset, bucket, min_chars=0, k=1, max_chars=60, to_end=True
                )
                self.assertEqual(best.text, one.text)
                self.assertAlmostEqual(best.cost, one.cost, places=12)
                self.assertEqual(expanded, one.expanded)

    def test_k_best_is_cost_ordered_and_distinct(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=3)
        found = model.predict("the ", length=0, mode="kbest", k=4, to_end=True, max_length=60)
        self.assertGreater(len(found.top), 1)
        costs = [r.cost for r in found.top]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual(len(found.top), len({r.text for r in found.top}))
        self.assertEqual(found.bottom, [], "a k-best search has no bounded worst walk; the beam has that")

    def test_k_best_costs_less_than_a_beam_of_the_same_k(self):
        """It stops at the k-th finished walk instead of carrying a frontier to the end."""
        model = ResonantNet(seed=0)
        model.train(TEXTS * 3, epochs=3)
        kbest = model.predict("the ", length=0, mode="kbest", k=5, to_end=True, max_length=60)
        beam = model.predict("the ", length=0, mode="beam", k=5, to_end=True, max_length=60)
        self.assertLess(kbest.expanded, beam.expanded)
        self.assertLessEqual(kbest.top[0].cost, beam.top[0].cost + 1e-12)

    def test_the_layer_runs_in_the_k_best_search_but_not_in_dijkstra(self):
        """One label per state cannot say which walk reached it; k labels can, so the layer sees the cycle."""
        model = ResonantNet(seed=0)
        model.train(["lol lol lol lol lol"], epochs=3)
        self.assertIn("lol:4", model.metacog.scores)
        before = {
            "dijkstra": [r.text for r in model.predict("lol ", length=12, mode="dijkstra").top],
            "kbest": [r.text for r in model.predict("lol ", length=12, mode="kbest", k=3).top],
        }
        for signature in list(model.metacog.scores):
            model.metacog.punish(signature, RIDE, 50)
        after = {
            "dijkstra": [r.text for r in model.predict("lol ", length=12, mode="dijkstra").top],
            "kbest": [r.text for r in model.predict("lol ", length=12, mode="kbest", k=3).top],
        }
        self.assertEqual(after["dijkstra"], before["dijkstra"], "the exact single-label search is cycle-blind")
        self.assertNotEqual(after["kbest"], before["kbest"], "k labels per state make the cycle visible")

        def loopiest(texts):
            """Where in the ranking the walk that rides the cycle hardest sits."""
            return max(range(len(texts)), key=lambda i: texts[i].count("lol"))

        self.assertGreater(loopiest(after["kbest"]), loopiest(before["kbest"]),
                           "refusing to ride must push the loopiest walk down the ranking")

    def test_generation_defaults_to_the_k_best_search(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=3)
        texts = [r.text for r in model.generate(count=3, max_length=40)]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(texts[0], model.generate(count=1, max_length=40, mode="dijkstra")[0].text)
        for mode in ("beam", "sample", "dijkstra"):
            with self.subTest(mode=mode):
                self.assertTrue(model.generate(count=1, max_length=40, mode=mode))
        with self.assertRaises(ValueError):
            model.generate(count=1, mode="nope")

    def test_prediction_modes(self):
        for mode in ("kbest", "beam", "dijkstra", "sample"):
            with self.subTest(mode=mode):
                result = self.model.predict("the sun", length=10, mode=mode)
                self.assertIsInstance(result, Prediction)
                self.assertEqual(result.full_text, "the sun" + result.text)
                self.assertEqual(result.mode, mode)
        beam = self.model.predict("the ", length=8, mode="beam", k=3)
        self.assertLessEqual(len(beam.top), 3)
        self.assertTrue(all(t.cost <= b.cost for t in beam.top for b in beam.bottom))

    def test_generation_returns_whole_texts(self):
        for mode in ("beam", "dijkstra", "sample"):
            with self.subTest(mode=mode):
                for result in self.model.generate(mode=mode, count=2, max_length=40):
                    self.assertEqual(result.text, result.full_text)
        with_prefix = self.model.generate(mode="beam", count=1, max_length=40, prefix="the ")
        self.assertTrue(with_prefix[0].text.startswith("the "))

    def test_score_prefers_what_was_trained(self):
        good = self.model.score(TEXTS[0])
        bad = self.model.score("qx zz vv qq")
        self.assertGreater(good["per_char"], bad["per_char"])
        self.assertEqual(self.model.score("")["transitions"], 0)
        with self.assertRaises(TypeError):
            self.model.score(None)

    def test_the_phase_picks_the_continuation_the_context_calls_for(self):
        """Two texts sharing a middle: the phase remembers which one the walk is in, the marginal cannot."""
        model = ResonantNet(seed=0, buckets=8, kick_scale=0.0, resonance_scale=2.0)
        model.train(["the cat sat down", "a big cat ran away"], epochs=6)
        self.assertEqual(model.predict("the cat ", length=8, mode="dijkstra").text, "sat down")
        self.assertEqual(model.predict("a big cat ", length=8, mode="dijkstra").text, "ran away")
        model.configure_weights(resonance_scale=0.0)  # the same model, phase switched off
        without = {model.predict(p, length=8, mode="dijkstra").text for p in ("the cat ", "a big cat ")}
        self.assertEqual(len(without), 1, "without the phase both contexts must give the same continuation")

    def test_sampling_and_conversing_go_through_the_phase_search(self):
        """``generate(mode="sample")`` and ``converse`` reach the search through ``_search``, not the base walk."""
        model = ResonantNet(seed=0, buckets=8, kick_scale=0.0, resonance_scale=2.0)
        model.train(["the cat sat down", "a big cat ran away"], epochs=6)
        greedy = {p: model.generate(mode="sample", count=1, max_length=40, prefix=p, temperature=0)[0].text
                  for p in ("the cat ", "a big cat ")}
        self.assertEqual(greedy["the cat "], "the cat sat down")
        self.assertEqual(greedy["a big cat "], "a big cat ran away")
        model.configure_weights(resonance_scale=0.0)  # the same model, phase switched off
        without = model.generate(mode="sample", count=1, max_length=40, prefix="the cat ", temperature=0)[0].text
        self.assertNotEqual(without, greedy["the cat "])
        self.assertGreater(without.count("sat"), 2, "phase-free, the tie makes greedy sampling loop")

    def test_score_uses_the_phase(self):
        model = ResonantNet(seed=0, buckets=8, kick_scale=1.0, resonance_scale=4.0)
        model.train(TEXTS, epochs=4)
        with_phase = model.score(TEXTS[0])["log_prob"]
        model.configure_weights(resonance_scale=0.0)  # the phase-free marginal of the same model
        self.assertNotAlmostEqual(with_phase, model.score(TEXTS[0])["log_prob"], places=4)

    def test_cycles_train_the_metacognitive_layer(self):
        model = ResonantNet(seed=0)
        records = model.train(["lol lol lol lol", "abab abab abab"], epochs=2)
        self.assertGreater(records[0]["cycles"], 0)
        self.assertGreater(len(model.metacog), 0)
        self.assertEqual(model.meta["cycles_seen"], sum(r["cycles"] for r in records))
        self.assertGreater(model.metacog.totals()[RIDE], 0.0)

    def test_the_layer_changes_what_the_beam_does_with_a_cycle(self):
        model = ResonantNet(seed=0)
        model.train(["lol lol lol lol lol"], epochs=3)
        self.assertIn("lol:4", model.metacog.scores)
        ridden = [r.text for r in model.predict("lol ", length=12, mode="beam", k=2).top]
        self.assertTrue(any(t.count("lol") > 1 for t in ridden), ridden)
        for signature in list(model.metacog.scores):
            model.metacog.punish(signature, RIDE, 50)  # the layer now refuses to ride these cycles
        refused = [r.text for r in model.predict("lol ", length=12, mode="beam", k=2).top]
        self.assertNotEqual(ridden, refused)
        self.assertLess(max(t.count("lol") for t in refused), max(t.count("lol") for t in ridden))

    def test_dijkstra_is_exact_and_ignores_the_layer(self):
        model = ResonantNet(seed=0)
        model.train(["lol lol lol lol lol"], epochs=3)
        before = model.predict("lol ", length=12, mode="dijkstra").text
        for sig in list(model.metacog.scores):
            model.metacog.punish(sig, RIDE, 50)
        self.assertEqual(model.predict("lol ", length=12, mode="dijkstra").text, before)

    def test_reward_and_punish(self):
        before = self.model.score(TEXTS[0])["per_char"]
        self.model.reward([TEXTS[0]], strength=2.0)
        self.assertGreater(self.model.score(TEXTS[0])["per_char"], before)
        self.model.punish([TEXTS[0]], strength=4.0)
        self.assertLess(self.model.score(TEXTS[0])["per_char"], before)
        self.assertGreater(self.model.meta["feedback_passes"], 0)

    def test_punish_decoheres(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=4)
        before = model.coherence()["mean"]
        model.punish(TEXTS, strength=2.0)
        self.assertLess(model.coherence()["mean"], before)

    def test_two_nrl_inverts_in_between(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=2)
        result = model.two_nrl(GARBAGE, TEXTS, neg_epochs=1, pos_epochs=1)
        self.assertEqual([r["phase"] for r in result["negative"]], ["negative"])
        self.assertEqual([r["phase"] for r in result["positive"]], ["positive"])
        self.assertTrue(result["inverted"])
        self.assertEqual(model.meta["twonrl_runs"], 1)

    def test_two_nrl_weights_the_worst_failures_hardest(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=1)
        result = model.two_nrl(GARBAGE, TEXTS, neg_epochs=1, pos_epochs=1, bad_weights=[3.0, 1.0, 1.0])
        self.assertEqual(sorted({r["weight"] for r in result["negative"]}, reverse=True), [3.0, 1.0])

    def test_invert_paths_modes(self):
        for mode, expected in (("activation", "edges"), ("state", "edges")):
            with self.subTest(mode=mode):
                model = ResonantNet(seed=0)
                model.train(TEXTS, epochs=2)
                before = model.score(TEXTS[0])["per_char"]
                out = model.invert_paths([TEXTS[0]], mode=mode, amounts=[1.0])
                self.assertEqual((out["texts"], out["unit"], out["mode"]), (1, expected, mode))
                self.assertGreater(out["flipped"], 0)
                self.assertLess(model.score(TEXTS[0])["per_char"], before)
        with self.assertRaises(ValueError):
            self.model.invert_paths(TEXTS, mode="nope")
        with self.assertRaises(ValueError):
            self.model.invert_paths(TEXTS[:1], amounts=[2.0])

    def test_invert_is_reversible(self):
        before = self.model.score(TEXTS[0])["log_prob"]
        self.model.invert()
        self.assertTrue(self.model.graph.inverted)
        self.model.invert()
        self.assertAlmostEqual(self.model.score(TEXTS[0])["log_prob"], before, places=9)

    def test_stats(self):
        stats = self.model.stats()
        for field in ("kind", "nodes", "edges", "buckets", "coherence_mean", "coherence_max", "cycles_seen",
                      "meta", "weights", "total_traversals"):
            self.assertIn(field, stats)
        self.assertEqual(stats["kind"], "resonant")
        self.assertGreaterEqual(stats["coherence_mean"], 0.0)
        self.assertLessEqual(stats["coherence_max"], 1.0)

    def test_predict_argument_validation(self):
        for kwargs in ({"length": -1}, {"k": -1}, {"beam": 0}, {"max_length": -1}, {"mode": "nope"}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.model.predict("the", **kwargs)
        with self.assertRaises(TypeError):
            self.model.predict(None)


class TestPersistence(unittest.TestCase):
    def test_round_trip_through_a_file(self):
        model = ResonantNet(seed=3, buckets=12, kick_scale=0.5, resonance_scale=2.0,
                            teach_back=True, back_strength=0.5, back_ceiling=0.2)
        model.train(TEXTS, epochs=2)
        model.train(["lol lol lol lol"], epochs=1)
        model.reward([TEXTS[0]], strength=1.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            model.save(path)
            back = load_model(path)
        self.assertIsInstance(back, ResonantNet)
        self.assertEqual(back.weight_config(), model.weight_config())
        self.assertEqual(
            (back.teach_back, back.back_strength, back.back_ceiling),
            (model.teach_back, model.back_strength, model.back_ceiling),
        )
        self.assertEqual(back.metacog.back_scale, model.metacog.back_scale)
        self.assertEqual(back.metacog.prior_weight, model.metacog.prior_weight)
        self.assertEqual(back.meta["epochs_total"], model.meta["epochs_total"])
        self.assertEqual(len(back.metacog), len(model.metacog))
        self.assertEqual(len(back.history), len(model.history))
        for text in TEXTS + ["lol lol lol lol"]:
            self.assertAlmostEqual(back.score(text)["log_prob"], model.score(text)["log_prob"], places=9)
        alive = [n for n in model.graph.alive_nodes()]
        self.assertEqual([back.graph.advance[i] for i in range(len(alive))],
                         [model.graph.advance[n] for n in alive])

    def test_gzip_round_trip(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json.gz")
            model.save(path)
            back = load_model(path)
        self.assertAlmostEqual(back.score(TEXTS[0])["log_prob"], model.score(TEXTS[0])["log_prob"], places=9)

    def test_a_document_of_another_kind_is_rejected(self):
        radix = RadixNet(seed=0)
        with self.assertRaises(ValueError):
            ResonantNet.from_dict(radix.to_dict())
        self.assertIsInstance(model_from_dict(ResonantNet(seed=0).to_dict()), ResonantNet)

    def test_checkpoints_keep_the_kind(self):
        from radixnet.checkpoint import CheckpointManager

        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep=2)
            record = manager.save(model, 1, "epoch", {"loss": 0.5})
            back = load_model(record["path"])
        self.assertIsInstance(back, ResonantNet)
        self.assertEqual(back.kind, "resonant")


class TestKindRegistry(unittest.TestCase):
    def test_registered(self):
        self.assertIn("resonant", [k["kind"] for k in model_kinds()])
        self.assertIs(model_class("resonant"), ResonantNet)
        self.assertIs(model_class(" Resonant "), ResonantNet)
        self.assertIsInstance(new_model("resonant", seed=7), ResonantNet)
        self.assertEqual(new_model("resonant", seed=7).seed, 7)
        self.assertEqual(ResonantNet.format, RESONANT_MODEL_FORMAT)


class TestEvolveAndDialogue(unittest.TestCase):
    def test_evolver_runs_a_generation(self):
        from radixnet.gan import EvolveConfig, Evolver

        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=2)
        evolver = Evolver(model, TEXTS, config=EvolveConfig(samples=3, real_per_generation=2, max_length=30, seed=1))
        record = evolver.run_generation()
        for field in ("generation", "fake_score_mean", "real_score_mean", "gap", "nodes", "edges"):
            self.assertIn(field, record)

    def test_converse(self):
        model = ResonantNet(seed=0)
        model.train(TEXTS, epochs=2)
        turns = model.converse("the sun", turns=2)
        self.assertEqual(len(turns), 3)


class TestCli(unittest.TestCase):
    def test_train_predict_info_and_weights(self):
        from tests.test_cli import run_cli, run_json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            data = os.path.join(tmp, "corpus.txt")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write("\n".join(TEXTS))
            run_cli("--kind", "resonant", "train", "--data", data, "--epochs", "2", model=path)
            info = run_json("info", model=path)
            self.assertEqual(info["stats"]["kind"], "resonant")
            self.assertEqual(info["stats"]["epochs_total"], 2)
            predicted = run_json("predict", "--prefix", "the ", "--length", "8", "--mode", "beam", model=path)
            self.assertTrue(predicted["full_text"].startswith("the "))
            weights = run_json("weights", "--kick-scale", "1.0", "--buckets", "16", model=path)
            self.assertEqual((weights["weights"]["kick_scale"], weights["weights"]["buckets"]), (1.0, 16))
            run_cli("weights", "--window", "5", model=path, expect=1)  # a count-model option

    def test_the_default_model_file_follows_the_kind(self):
        from radixnet.cli import DEFAULT_RESONANT_MODEL, build_parser

        args = build_parser().parse_args(["--kind", "resonant", "info"])
        self.assertEqual(args.kind, "resonant")
        self.assertEqual(DEFAULT_RESONANT_MODEL, "model.resonant.json")


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.model_path = os.path.join(cls.tmp.name, "model.json")
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, model_path=cls.model_path)

    def wait(self):
        from tests.test_api import wait_for_job

        return wait_for_job(self.client)

    def setUp(self):
        status, data, _ = self.client.post("/api/model/select", {"kind": "resonant"})
        self.assertEqual(status, 200, data)

    def test_select_train_and_predict(self):
        self.assertEqual(self.client.get("/api/status")[1]["kind"], "resonant")
        self.assertEqual(self.client.post("/api/train", {"texts": TEXTS, "epochs": 2})[0], 202)
        self.wait()
        status, data, _ = self.client.post("/api/predict", {"prefix": "the ", "length": 10, "mode": "beam", "k": 2})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["kind"], "resonant")
        self.assertTrue(data["full_text"].startswith("the "))
        self.assertLessEqual(len(data["top"]), 2)

    def test_status_reports_the_phase_and_the_layer(self):
        self.client.post("/api/train", {"texts": TEXTS + ["lol lol lol lol"], "epochs": 2})
        self.wait()
        status = self.client.get("/api/status")[1]
        self.assertEqual(status["kind"], "resonant")
        self.assertIn("coherence_mean", status)
        self.assertIn("signatures", status["meta"])

    def test_graph_view_reports_coherence_and_the_advance(self):
        self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.wait()
        graph = self.client.get("/api/graph?limit=5")[1]
        self.assertIn("buckets", graph)
        self.assertNotIn("window", graph)
        self.assertTrue(all("advance" in n for n in graph["nodes"]))
        self.assertTrue(all({"coherence", "mu", "reward"} <= set(e) for e in graph["edges"]))

    def test_weight_endpoint(self):
        status, data, _ = self.client.post("/api/model/weights", {"kick_scale": 1.0, "resonance_scale": 2.0})
        self.assertEqual(status, 200, data)
        self.assertEqual((data["weights"]["kick_scale"], data["weights"]["resonance_scale"]), (1.0, 2.0))
        status, data, _ = self.client.post("/api/model/weights", {"window": 5})  # a count-model option
        self.assertEqual(status, 400)
        self.assertIn("window", data["error"])

    def test_reset_with_phase_options(self):
        status, data, _ = self.client.post("/api/reset", {"kind": "resonant", "buckets": 16, "kick_scale": 0.5})
        self.assertEqual(status, 200, data)
        self.assertEqual(self.client.get("/api/model")[1]["weights"]["buckets"], 16)

    def test_switching_kinds_keeps_the_model_in_memory(self):
        self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.wait()
        nodes = self.client.get("/api/status")[1]["nodes"]
        self.assertEqual(self.client.post("/api/model/select", {"kind": "radix"})[1]["kind"], "radix")
        status, data, _ = self.client.post("/api/model/select", {"kind": "resonant"})
        self.assertEqual((status, data["origin"]), (200, "memory"))
        self.assertEqual(self.client.get("/api/status")[1]["nodes"], nodes)


if __name__ == "__main__":
    unittest.main()
