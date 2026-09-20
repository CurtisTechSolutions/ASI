"""Tests for radixnet.penalty - the punishment traversal.

The option is *what* a search looks for, not how it looks: every mode of every
model kind can run either traversal, so these tests check the split the
traversal rests on (``child_evidence``), the costs it produces, that the
rewards really do leave the score, that the cheapest path is the least punished
one, and that the option reaches the CLI and the HTTP API.
"""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import cli  # noqa: E402
from radixnet.api import ModelService  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.graph import BACK  # noqa: E402
from radixnet.model import RadixNet  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.penalty import (  # noqa: E402
    DEFAULT_TRAVERSAL,
    TRAVERSALS,
    PenaltyCosts,
    PhasePenaltyCosts,
    phase_traversal_costs,
    resolve_traversal,
    traversal_costs,
)
from radixnet.resonance import ResonantNet  # noqa: E402

TEXTS = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "the cat ran up the hill",
]
PRAISED = "the cat sat on the mat"
PUNISHED = "the cat ate the rat"


def counted(epochs=4, seed=1):
    """A count / reward model with one branch praised, one punished and one left alone."""
    model = CountRewardNet(seed=seed)
    model.train(TEXTS, epochs=epochs)
    model.reward([PRAISED], strength=4.0)
    model.punish([PUNISHED], strength=4.0)
    return model


def fork(graph):
    """The node where the three texts part company (the one with the most children)."""
    return max(graph.alive_nodes(), key=lambda n: len(graph.children[n]))


def costs_of(fn, node):
    return {child: cost for child, _edge, cost in fn(node)}


class TestNames(unittest.TestCase):
    def test_the_two_traversals(self):
        self.assertEqual(TRAVERSALS, ("reward", "punishment"))
        self.assertEqual(DEFAULT_TRAVERSAL, "reward")

    def test_resolve_normalises_and_rejects(self):
        for given in (None, "", "reward", "REWARD"):
            self.assertEqual(resolve_traversal(given), "reward")
        self.assertEqual(resolve_traversal("Punishment"), "punishment")
        with self.assertRaises(ValueError):
            resolve_traversal("penalty")

    def test_the_reward_traversal_costs_nothing_extra(self):
        model = counted()
        self.assertIsNone(traversal_costs(model.graph, "reward"))
        self.assertIsNone(phase_traversal_costs(ResonantNet(seed=1).graph, "reward"))
        self.assertIsInstance(traversal_costs(model.graph, "punishment"), PenaltyCosts)

    def test_negative_scales_are_rejected(self):
        graph = counted().graph
        with self.assertRaises(ValueError):
            PenaltyCosts(graph, penalty_scale=-1.0)
        with self.assertRaises(ValueError):
            PenaltyCosts(graph, merit_scale=-0.5)


class TestEvidence(unittest.TestCase):
    """``child_evidence`` - what speaks for a step and what speaks against it, per model kind."""

    def test_the_count_model_splits_the_reward_in_two(self):
        graph = counted().graph
        node = fork(graph)
        evidence = {c: (e, merit, penalty) for c, e, merit, penalty in graph.child_evidence(node)}
        self.assertEqual(set(evidence), set(graph.children[node]))
        for child, (edge, merit, penalty) in evidence.items():
            reward = graph.edge_reward[edge]
            # the merit is the weight with the whole reward taken back out: the frequency terms alone
            self.assertAlmostEqual(merit, graph.edge_w[edge] - graph.reward_scale * reward)
            self.assertAlmostEqual(penalty, graph.reward_scale * max(0.0, -reward))
            self.assertGreaterEqual(penalty, 0.0)
        rewarded = [p for _e, _m, p in evidence.values() if _e is not None]
        self.assertTrue(any(p > 0 for p in rewarded))  # something was punished
        self.assertTrue(any(p == 0 for p in rewarded))  # and something was not

    def test_a_praised_edge_carries_no_penalty(self):
        model = counted()
        graph = model.graph
        praised = [e for e in range(len(graph.edge_reward)) if graph.edge_alive[e] and graph.edge_reward[e] > 0]
        self.assertTrue(praised)
        for node in graph.alive_nodes():
            for _c, edge, _merit, penalty in graph.child_evidence(node):
                if edge in praised:
                    self.assertEqual(penalty, 0.0)

    def test_the_sine_model_reads_its_punishments_off_the_score(self):
        model = RadixNet(seed=1)
        model.train(TEXTS, epochs=2)
        graph = model.graph
        for node in graph.alive_nodes():
            scores = dict(graph.child_scores(node))
            for child, _edge, merit, penalty in graph.child_evidence(node):
                self.assertAlmostEqual(merit - penalty, scores[child])
                self.assertGreaterEqual(min(merit, penalty), 0.0)
                self.assertEqual(min(merit, penalty), 0.0)  # a score is either for or against, never both

    def test_the_negative_network_weighs_blame_against_clearing(self):
        model = NegativeNet(seed=1)
        model.blame([PUNISHED], reason="nonsense", severity=6.0)
        model.blame([PRAISED], reason="nonsense", severity=1.0)
        graph = model.graph
        seen = 0
        for node in graph.alive_nodes():
            for _c, edge, merit, penalty in graph.child_evidence(node):
                self.assertGreaterEqual(penalty, 0.0)
                self.assertGreaterEqual(merit, 0.0)
                self.assertAlmostEqual(penalty, math.log1p(graph.evidence(edge)))
                seen += 1
        self.assertTrue(seen)

    def test_a_judged_context_splits_the_same_way(self):
        model = counted()
        graph = model.graph
        # judge one walk wrong so a path context exists and speaks against its steps
        transitions = graph.observe_sequence(model.encoder.encode(PUNISHED), count=False)
        graph.record_path(transitions, outcome=False)
        node, edge = next(
            (graph.edge_parent[e], e) for (_prev, e) in graph.paths if graph.alive[graph.edge_parent[e]]
        )
        prev = next(p for (p, e) in graph.paths if e == edge)
        plain = {e: (m, q) for _c, e, m, q in graph.child_evidence(node)}
        in_context = {e: (m, q) for _c, e, m, q in graph.child_evidence(node, prev)}
        term = graph.path_scale * graph.path_term(prev, edge)
        self.assertLess(term, 0.0)  # the context says this step was wrong here
        self.assertAlmostEqual(in_context[edge][1] - plain[edge][1], -term)  # ... so it is a penalty
        self.assertAlmostEqual(in_context[edge][0], plain[edge][0])  # ... and not merit


class TestCosts(unittest.TestCase):
    def test_the_rewards_leave_the_score(self):
        model = counted()
        graph = model.graph
        node = fork(graph)
        reward = costs_of(graph.child_costs, node)
        punishment = costs_of(PenaltyCosts(graph), node)
        by_edge = {c: e for c, e in graph.children[node].items()}
        praised = [c for c in reward if graph.edge_reward[by_edge[c]] > 0]
        neutral = [c for c in reward if graph.edge_reward[by_edge[c]] == 0]
        punished = [c for c in reward if graph.edge_reward[by_edge[c]] < 0]
        self.assertTrue(praised and neutral and punished)
        for good in praised:
            for plain in neutral:
                # the reward traversal buys the praised child a large discount over the child nobody judged ...
                bought = reward[plain] - reward[good]
                self.assertGreater(bought, 3.0)
                # ... which the punishment traversal does not honour: neither of the two was ever punished,
                # so all that separates them there is what the corpus did
                left = punishment[plain] - punishment[good]
                self.assertLess(left, bought - 3.0)
        for bad in punished:
            for plain in neutral:
                self.assertGreater(punishment[bad], punishment[plain] + 1.0)

    def test_pure_punishment_is_indifferent_between_unpunished_children(self):
        graph = counted().graph
        node = fork(graph)
        pure = costs_of(PenaltyCosts(graph, merit_scale=0.0), node)
        unpunished = [
            cost for c, cost in pure.items() if graph.edge_reward[graph.children[node][c]] >= 0
        ]
        self.assertGreater(len(unpunished), 1)
        self.assertAlmostEqual(min(unpunished), max(unpunished))

    def test_penalty_scale_sharpens_the_verdict(self):
        graph = counted().graph
        node = fork(graph)
        punished = next(c for c, e in graph.children[node].items() if graph.edge_reward[e] < 0)
        soft = costs_of(PenaltyCosts(graph, penalty_scale=0.25), node)[punished]
        hard = costs_of(PenaltyCosts(graph, penalty_scale=4.0), node)[punished]
        self.assertGreater(hard, soft)

    def test_costs_are_non_negative_log_probabilities(self):
        graph = counted().graph
        fn = PenaltyCosts(graph)
        for node in graph.alive_nodes():
            costs = fn(node)
            if not costs:
                continue
            for _c, _e, cost in costs:
                self.assertGreaterEqual(cost, -1e-12)
            total = sum(pow(2.718281828459045, -cost) for _c, _e, cost in costs)
            self.assertAlmostEqual(total, 1.0, places=9)  # still a distribution, so Dijkstra still holds

    def test_the_cache_follows_the_graph(self):
        model = counted()
        graph = model.graph
        node = fork(graph)
        fn = PenaltyCosts(graph)
        before = costs_of(fn, node)
        model.punish([PUNISHED], strength=6.0)
        after = costs_of(fn, node)
        self.assertNotEqual(before, after)

    def test_back_edges_are_priced_too(self):
        model = counted()
        graph = model.graph
        node = fork(graph)
        graph.observe_back(node, amount=1.0)
        self.assertIn(BACK, {c for c, _e, _cost in PenaltyCosts(graph)(node)})


class TestSearches(unittest.TestCase):
    def test_the_cheapest_path_is_the_least_punished_one(self):
        model = counted()
        graph = model.graph
        found = model.predict("the cat", length=16, k=3, traversal="punishment", merit_scale=0.0)
        self.assertEqual(found.traversal, "punishment")
        self.assertNotIn("ate the rat", found.text)  # the corrected branch is the dear one now
        for path in found.top:
            punishment = sum(
                max(0.0, -graph.edge_reward[graph.children[p][c]])
                for p, c in zip(path.node_ids, path.node_ids[1:])
                if c in graph.children[p]
            )
            self.assertEqual(punishment, 0.0)

    def test_the_two_traversals_disagree(self):
        model = counted()
        by_reward = model.predict("the cat", length=16, k=3)
        by_punishment = model.predict("the cat", length=16, k=3, traversal="punishment", merit_scale=0.0)
        self.assertEqual(by_reward.traversal, "reward")
        self.assertNotEqual(by_reward.text, by_punishment.text)

    def test_every_mode_takes_the_option(self):
        model = counted()
        for mode in ("beam", "sample", "dijkstra"):
            found = model.predict("the cat", length=12, k=2, mode=mode, traversal="punishment")
            self.assertEqual(found.traversal, "punishment")
            self.assertTrue(found.full_text.startswith("the cat"))

    def test_generation_takes_the_option(self):
        model = counted()
        for mode in ("beam", "sample", "dijkstra"):
            texts = model.generate(count=2, mode=mode, max_length=40, seed=3, traversal="punishment")
            self.assertTrue(texts)
            self.assertTrue(all(isinstance(r.text, str) for r in texts))

    def test_the_sine_model_takes_the_option(self):
        model = RadixNet(seed=1)
        model.train(TEXTS, epochs=2)
        for mode in ("dijkstra", "beam", "sample"):
            result = model.predict("the cat", length=12, mode=mode, traversal="punishment", penalty_scale=3.0)
            self.assertTrue(result.full_text.startswith("the cat"))

    def test_the_phase_model_takes_the_option_in_every_mode(self):
        model = ResonantNet(seed=1)
        model.train(TEXTS, epochs=3)
        model.punish([PUNISHED], strength=4.0)
        graph = model.graph
        self.assertIsInstance(phase_traversal_costs(graph, "punishment"), PhasePenaltyCosts)
        for mode in ("kbest", "dijkstra", "beam", "sample"):
            found = model.predict("the cat", length=12, k=2, mode=mode, traversal="punishment")
            self.assertEqual(found.traversal, "punishment")
        # the phase-aware split keeps the resonance as merit and only the reward's negative half as penalty
        for node in graph.alive_nodes():
            at = {c: (m, q) for c, _e, m, q in graph.child_evidence_at(node, 1)}
            for child, _edge, score in graph.child_scores_at(node, 1):
                merit, penalty = at[child]
                edge = graph.children[node][child]
                self.assertAlmostEqual(merit, score - graph.reward_scale * graph.edge_reward[edge])
                self.assertAlmostEqual(penalty, graph.reward_scale * max(0.0, -graph.edge_reward[edge]))

    def test_the_negative_network_walks_its_least_blamed_way(self):
        model = NegativeNet(seed=1)
        model.blame([PUNISHED], reason="nonsense", severity=8.0)
        model.blame([PRAISED], reason="nonsense", severity=1.0)
        likeliest = model.predict("the cat", length=14, k=2)
        least_blamed = model.predict("the cat", length=14, k=2, traversal="punishment")
        self.assertEqual(least_blamed.traversal, "punishment")
        self.assertNotEqual(likeliest.text, least_blamed.text)

    def test_an_unknown_traversal_is_refused_by_every_kind(self):
        for model in (counted(), RadixNet(seed=1), ResonantNet(seed=1), NegativeNet(seed=1)):
            with self.assertRaises(ValueError):
                model.predict("the cat", length=4, traversal="least-blame")


class TestFrontDoors(unittest.TestCase):
    """The option where a person reaches it: the CLI and the HTTP API."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "model.count.json")
        counted().save(self.path)
        self.addCleanup(self.dir.cleanup)

    def test_the_cli_carries_the_traversal(self):
        doc = _json_cli(["--model", self.path, "--json", "predict", "--prefix", "the cat", "--length", "16"])
        self.assertEqual(doc["traversal"], "reward")
        punished = _json_cli([
            "--model", self.path, "--json", "predict", "--prefix", "the cat", "--length", "16",
            "--traversal", "punishment", "--merit-scale", "0",
        ])
        self.assertEqual(punished["traversal"], "punishment")
        self.assertNotEqual(doc["continuation"], punished["continuation"])

    def test_the_cli_generates_with_it(self):
        doc = _json_cli([
            "--model", self.path, "--json", "generate", "--count", "2", "--mode", "beam",
            "--max-length", "40", "--traversal", "punishment",
        ])
        self.assertEqual(doc["traversal"], "punishment")
        self.assertTrue(doc["samples"])

    def test_the_cli_rejects_an_unknown_traversal(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["--model", self.path, "--json", "predict", "--prefix", "a", "--traversal", "nope"])
        self.assertNotEqual(code, 0)

    def test_the_api_carries_the_traversal(self):
        svc = ModelService(self.path, backend="python")
        plain = svc.predict("the cat", length=16, k=3, mode="beam", guard=False)
        punished = svc.predict(
            "the cat", length=16, k=3, mode="beam", guard=False, traversal="punishment", merit_scale=0.0,
        )
        self.assertEqual(plain["traversal"], "reward")
        self.assertEqual(punished["traversal"], "punishment")
        self.assertNotEqual(plain["continuation"], punished["continuation"])

    def test_the_api_generates_with_it(self):
        svc = ModelService(self.path, backend="python")
        out = svc.generate(count=2, mode="beam", max_length=40, guard=False, traversal="punishment")
        self.assertTrue(out["samples"])


def _json_cli(argv):
    """Run the CLI with ``--json`` and give back the document it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(argv)
    if code != 0:
        raise AssertionError(f"cli exited {code}: {buffer.getvalue()}")
    return json.loads(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
