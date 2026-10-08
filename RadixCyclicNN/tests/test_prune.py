"""Tests for radixnet.prune: auto prune, the graph letting go of the edges nothing walks and the nodes they strand.

The rule is the one of ``../SPEC-AutoPrune.md``: two thresholds over what the
graph already counts - an edge traversed fewer than ``min_count`` times, or
taking less than ``min_share`` of its node's out-traversals - and the real
nodes left with no way in or no way out go with the edges.  Pinned here: the
setting and what it refuses; a text registered but never walked pruned away
while the trained texts still walk; a busy node thinned of its rare
continuation; what was taught - a hand-over, a reward, a judged context,
blame - never pruned and the node holding it never swept; the sweep running
until nothing is stranded; a self-loop being no way in or out; the file block
and off being the old file to the bit; the automatic prune at the end of an
epoch on every kind, every ``every``-th epoch, never by hand; a prune by hand
with the setting off; the CLI and the HTTP API are the kit's ``test_prune.py``.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import Encoder  # noqa: E402
from radixnet.graph import BACK, END, FIRST, START, RadixCyclicGraph  # noqa: E402
from radixnet.model import RadixNet, model_from_dict  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.prune import DEFAULT_EVERY, DEFAULT_MIN_COUNT, DEFAULT_MIN_SHARE, AutoPrune, check_thresholds  # noqa: E402
from radixnet.resonance import ResonantNet  # noqa: E402

ENC = Encoder()
TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown cat sat on the mat",
    "a bird sang in the tree all afternoon",
]
UNWALKED = "the lazy cat sang"
KINDS = (RadixNet, CountRewardNet, NegativeNet, ResonantNet)


def teach(model, texts, **options):
    """Teach ``texts`` to a model of any kind: blamed for the negative network, trained for the rest."""
    if isinstance(model, NegativeNet):
        return model.blame(texts, reason="other", **options)
    return model.train(texts, **options)


def trained(cls, **options):
    model = cls(seed=1)
    teach(model, TEXTS, epochs=1, **options)
    return model


def register(graph, text):
    """Register ``text`` structurally - nodes and edges, nothing counted - the way scoring an unseen text does."""
    graph.observe_sequence(ENC.encode(text), count=False)


def walkable(graph, text):
    return graph.node_path(ENC.encode(text)) is not None


def strip_times(doc):
    doc = json.loads(json.dumps(doc))
    doc.pop("saved_at", None)
    doc.get("meta", {}).pop("created", None)
    for record in doc.get("history", []):
        record.pop("seconds", None)
    for entry in doc.get("log", []):
        entry.pop("at", None)
    return doc


class TestSetting(unittest.TestCase):
    def test_off_by_default_and_the_defaults_when_on(self):
        off = AutoPrune()
        self.assertFalse(off.on)
        self.assertIsNone(off.to_dict())
        self.assertEqual(str(off), "off")
        self.assertIn("off", off.describe())
        on = AutoPrune(True)
        self.assertEqual((on.min_count, on.min_share, on.every, on.auto), (DEFAULT_MIN_COUNT, DEFAULT_MIN_SHARE, DEFAULT_EVERY, True))
        self.assertEqual((DEFAULT_MIN_COUNT, DEFAULT_MIN_SHARE, DEFAULT_EVERY), (1, 0.0, 1))
        self.assertEqual(on.to_dict(), {"min_count": 1, "min_share": 0.0, "every": 1, "auto": True})
        self.assertIn("fewer than 1 time", on.rule())
        self.assertIn("every training epoch", on.describe())
        self.assertIn("by hand", AutoPrune(True, auto=False).describe())
        self.assertIn("2 training epochs", AutoPrune(True, every=2).describe())
        self.assertEqual(AutoPrune(True, 0, 0.0).rule(), "nothing (both thresholds are off)")
        self.assertIn("under 0.05 of its node", AutoPrune(True, 3, 0.05).rule())

    def test_what_the_thresholds_refuse(self):
        for min_count, min_share, every in ((-1, 0.0, 1), (1, 1.0, 1), (1, -0.1, 1), (1, 0.0, 0), (True, 0.0, 1),
                                            (1, 0.0, False), (1.5, 0.0, 1)):
            with self.subTest(min_count=min_count, min_share=min_share, every=every), self.assertRaises(ValueError):
                check_thresholds(min_count, min_share, every)
        check_thresholds(0, 0.0, 1)
        check_thresholds(5, 0.999, 7)
        with self.assertRaises(ValueError):
            AutoPrune(True, "many")
        with self.assertRaises(ValueError):
            AutoPrune(True, 1, 2.0)

    def test_when_it_is_due(self):
        self.assertFalse(AutoPrune().due(1))
        self.assertFalse(AutoPrune(True, auto=False).due(1))
        self.assertTrue(AutoPrune(True).due(1))
        every3 = AutoPrune(True, every=3)
        self.assertEqual([e for e in range(1, 10) if every3.due(e)], [3, 6, 9])

    def test_the_file_block(self):
        self.assertEqual(AutoPrune.from_dict(None), AutoPrune())
        self.assertEqual(AutoPrune.from_dict({}), AutoPrune())
        self.assertEqual(AutoPrune.from_dict({"min_count": 2, "min_share": 0.1, "every": 4, "auto": False}),
                         AutoPrune(True, 2, 0.1, 4, False))
        self.assertEqual(AutoPrune.from_dict({"min_count": 2}), AutoPrune(True, 2))  # the rest at the defaults
        for block in ("yes", {"min_count": "2"}, {"min_share": "0.1"}, {"every": 1.5}, {"auto": "no"}, {"min_share": 1.0}):
            with self.subTest(block=block), self.assertRaises(ValueError):
                AutoPrune.from_dict(block)
        for setting in (AutoPrune(True), AutoPrune(True, 3, 0.25, 2, False)):
            self.assertEqual(AutoPrune.from_dict(setting.to_dict()), setting)


class TestGraph(unittest.TestCase):
    def graph(self):
        g = RadixCyclicGraph(seed=1)
        for text in TEXTS:
            g.observe_sequence(ENC.encode(text))
        g.compress()
        return g

    def test_what_was_registered_but_never_walked_is_pruned_and_the_rest_still_walks(self):
        g = self.graph()
        register(g, UNWALKED)
        g.compress()
        self.assertTrue(walkable(g, UNWALKED))
        before = (g.num_nodes(), g.num_edges(), g.num_trigrams())
        candidates = g.prune_candidates()
        self.assertTrue(candidates)
        self.assertTrue(all(g.edge_traversals(e) == 0 for e in candidates))
        done = g.prune()
        g.compress()
        g.check_invariants(TEXTS, compressed=True)
        self.assertEqual(done["candidates"], len(candidates))
        self.assertGreaterEqual(done["edges"], len(candidates))
        self.assertGreater(done["nodes"], 0)
        self.assertLess(g.num_nodes(), before[0])
        self.assertLess(g.num_edges(), before[1])
        self.assertLess(g.num_trigrams(), before[2])
        self.assertFalse(walkable(g, UNWALKED))
        self.assertIsNone(g.lookup("y c"))  # a gram only the unwalked text had left the index
        self.assertEqual(g.prune_candidates(), [])  # nothing is left to prune
        self.assertEqual(g.prune(), {"candidates": 0, "edges": 0, "nodes": 0})
        # read again, the text is known again
        g.observe_sequence(ENC.encode(UNWALKED))
        g.compress()
        g.check_invariants(TEXTS + [UNWALKED], compressed=True)
        self.assertTrue(walkable(g, UNWALKED))

    def test_a_busy_node_is_thinned_of_its_rare_continuation(self):
        g = RadixCyclicGraph(seed=1)
        for _ in range(9):
            g.observe_sequence(ENC.encode("the cat sat"))
        g.observe_sequence(ENC.encode("the cow sat"))
        g.compress()
        self.assertEqual(g.prune_candidates(), [])  # every edge was traversed
        self.assertEqual(g.prune_candidates(min_share=0.05), [])  # the cow takes a tenth of "the c"
        rare = g.prune_candidates(min_share=0.2)
        self.assertEqual(len(rare), 1)
        p = next(node for node, ch in enumerate(g.children) if rare[0] in ch.values())
        self.assertEqual(g.edge_traversals(rare[0]), 1)
        self.assertEqual(sum(g.edge_traversals(e) for e in g.children[p].values()), 10)
        done = g.prune(min_share=0.2)
        self.assertEqual((done["candidates"], done["edges"]), (1, 2))  # the rare edge, then the stranded node's way out
        self.assertEqual(done["nodes"], 1)
        g.compress()
        g.check_invariants(["the cat sat"], compressed=True)
        self.assertFalse(walkable(g, "the cow sat"))
        self.assertTrue(walkable(g, "the cat sat"))

    def test_the_graphs_own_thresholds_are_the_setting(self):
        g = RadixCyclicGraph(seed=1)
        for _ in range(9):
            g.observe_sequence(ENC.encode("the cat sat"))
        g.observe_sequence(ENC.encode("the cow sat"))
        g.compress()
        g.auto_prune = AutoPrune(True, 0, 0.2)
        self.assertEqual(len(g.prune_candidates()), 1)
        g.auto_prune = AutoPrune(True, 0, 0.0)
        self.assertEqual(g.prune_candidates(), [])  # both rules off: nothing is a candidate
        # a threshold given overrides the setting's: the cow's two edges (into " cow sat", and out of it to END)
        self.assertEqual(len(g.prune_candidates(min_count=2)), 2)

    def test_what_was_taught_is_never_pruned_and_keeps_its_node(self):
        g = self.graph()
        p = g.lookup("qui")[0]
        back = g.observe_back(p)
        self.assertTrue(g.edge_protected(back))
        self.assertEqual(g.edge_traversals(back), 1)
        self.assertNotIn(back, g.prune_candidates(min_count=100))
        real = g.num_nodes() - FIRST
        done = g.prune(min_count=100)  # everything observed goes
        self.assertEqual(done["nodes"], real - 1)
        g.check_invariants()
        alive = [n for n in range(FIRST, len(g.labels)) if g.alive[n]]
        self.assertEqual(alive, [p])  # the node that learned to hand over stays, nameable
        self.assertEqual(list(g.children[p]), [BACK])
        self.assertEqual(g.num_edges(), 1)

    def test_the_sweep_runs_until_nothing_is_stranded(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(ENC.encode("abcdefgh"))  # uncompressed: a chain of single-gram nodes
        self.assertEqual(g.num_nodes() - FIRST, 6)
        g.check_invariants()
        # cut the first edge and the whole chain is stranded, one node after another
        first = g.children[START][g.lookup("abc")[0]]
        self.assertTrue(g.remove_edge(first))
        self.assertFalse(g.remove_edge(first))  # dead already
        self.assertIn(g.lookup("abc")[0], g.stranded_nodes())
        done = g.prune(min_count=0, min_share=0.0)  # no candidates, so no sweep: the thresholds decide
        self.assertEqual(done, {"candidates": 0, "edges": 0, "nodes": 0})
        g.observe_sequence(ENC.encode("xyz"), count=False)  # something to prune, so the sweep runs
        done = g.prune()
        g.check_invariants()
        self.assertEqual([n for n in range(FIRST, len(g.labels)) if g.alive[n]], [])
        self.assertEqual(g.num_trigrams(), 0)
        self.assertGreaterEqual(done["nodes"], 2)

    def test_a_self_loop_is_no_way_in_or_out(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(ENC.encode("aaaa"))
        node = g.lookup("aaa")[0]
        self.assertIn(node, g.children[node])  # the self-loop
        self.assertEqual(g.stranded_nodes(), [])
        g.remove_edge(g.children[START][node])
        self.assertEqual(g.stranded_nodes(), [node])
        self.assertEqual(g.remove_node(node), 2)  # the self-loop and the edge into END
        self.assertEqual((g.num_nodes(), g.num_edges(), g.num_trigrams()), (FIRST, 0, 0))
        g.check_invariants()
        with self.assertRaises(ValueError):
            g.remove_node(START)
        self.assertEqual(g.remove_node(node), 0)  # dead already

    def test_the_file_compacts_what_was_pruned(self):
        g = self.graph()
        register(g, UNWALKED)
        g.prune()
        g.compress()
        again = RadixCyclicGraph.from_dict(g.to_dict())
        again.check_invariants(TEXTS, compressed=True)
        self.assertEqual((again.num_nodes(), again.num_edges(), again.num_trigrams()), (g.num_nodes(), g.num_edges(), g.num_trigrams()))
        self.assertEqual(len(again.labels), again.num_nodes())  # no tombstones in a file
        self.assertFalse(walkable(again, UNWALKED))
        self.assertNotIn("auto_prune", g.to_dict())
        g.auto_prune = AutoPrune(True, 2, 0.1, 3, False)
        doc = g.to_dict()
        self.assertEqual(doc["auto_prune"], {"min_count": 2, "min_share": 0.1, "every": 3, "auto": False})
        self.assertEqual(RadixCyclicGraph.from_dict(doc).auto_prune, g.auto_prune)


class TestKinds(unittest.TestCase):
    def test_the_automatic_prune_at_the_end_of_an_epoch_on_every_kind(self):
        for cls in KINDS:
            with self.subTest(kind=cls.kind):
                model = trained(cls)
                graph = model.graph
                register(graph, UNWALKED)
                self.assertTrue(walkable(graph, UNWALKED))
                self.assertNotIn("auto_prune", model.stats())  # the stats are every port's; the setting has its own view
                config = model.configure_prune(on=True)
                self.assertEqual((config["on"], config["min_count"], config["every"], config["auto"]), (True, 1, 1, True))
                self.assertGreater(config["candidates"], 0)
                self.assertEqual(graph.auto_prune.to_dict(), {"min_count": 1, "min_share": 0.0, "every": 1, "auto": True})
                candidates = set(graph.prune_candidates())
                before = graph.num_edges()
                records = teach(model, [TEXTS[0]], epochs=1)
                record = records[-1]
                self.assertGreater(record["pruned_edges"], 0)
                self.assertGreater(record["pruned_nodes"], 0)
                self.assertLess(graph.num_edges(), before)
                self.assertFalse(walkable(graph, UNWALKED))
                # a blamed edge is never merged (D-046), so the negative network is not expected to be compressed
                graph.check_invariants(TEXTS, compressed=cls is not NegativeNet)
                self.assertEqual(model.prune_config()["candidates"], 0)
                for e, ok in enumerate(graph.edge_alive):  # every alive edge has a finite weight after the recompute
                    if ok:
                        self.assertTrue(graph.edge_w[e] == graph.edge_w[e] and abs(graph.edge_w[e]) < float("inf"))
                if cls is CountRewardNet:
                    self.assertFalse(candidates & set(graph._window))  # the window forgot the pruned edges
                    self.assertTrue(all(graph.window_edge_count[e] == 0 for e in candidates))
                # the model still answers, and the file carries the setting
                self.assertTrue(model.predict("the qui", length=3) is not None)
                again = model_from_dict(model.to_dict())
                self.assertEqual(again.graph.auto_prune, AutoPrune(True))
                again.graph.check_invariants(TEXTS, compressed=cls is not NegativeNet)

    def test_every_nth_epoch_and_never_by_hand(self):
        model = trained(RadixNet)
        register(model.graph, UNWALKED)
        model.configure_prune(on=True, every=2)
        records = model.train([TEXTS[0]], epochs=3)
        self.assertEqual([r["epoch"] for r in records], [2, 3, 4])  # lifetime epochs: trained() was the first
        self.assertEqual([("pruned_edges" in r) for r in records], [True, False, True])
        self.assertGreater(records[0]["pruned_edges"], 0)
        self.assertEqual(records[2]["pruned_edges"], 0)  # nothing left to prune, and the record says so
        model = trained(RadixNet)
        register(model.graph, UNWALKED)
        model.configure_prune(on=True, auto=False)
        records = model.train([TEXTS[0]], epochs=2)
        self.assertTrue(all("pruned_edges" not in r for r in records))
        self.assertTrue(walkable(model.graph, UNWALKED))  # by hand: nothing happened by itself
        done = model.prune()
        self.assertGreater(done["edges"], 0)
        self.assertFalse(walkable(model.graph, UNWALKED))

    def test_a_prune_by_hand_with_the_setting_off(self):
        model = trained(CountRewardNet)
        register(model.graph, UNWALKED)
        self.assertFalse(model.graph.auto_prune.on)
        self.assertIsNone(model.prune_config()["candidates"])
        done = model.prune()
        # a merge takes one node and one edge, after the removals
        self.assertEqual(done["nodes_before"] - done["nodes"] - done["merges"], done["nodes_after"])
        self.assertEqual(done["edges_before"] - done["edges"] - done["merges"], done["edges_after"])
        self.assertGreater(done["edges"], 0)
        self.assertGreater(done["merges"], 0)
        self.assertFalse(done["prune"]["on"])
        self.assertFalse(walkable(model.graph, UNWALKED))
        model.graph.check_invariants(TEXTS, compressed=True)
        for bad in ({"min_count": -1}, {"min_share": 1.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.prune(**bad)

    def test_what_a_judge_taught_is_kept_on_the_kinds_that_record_a_verdict(self):
        once = "the lazy dog ran"
        for cls in (CountRewardNet, ResonantNet):
            with self.subTest(kind=cls.kind):
                model = trained(cls)
                graph = model.graph
                model.train([once], epochs=1)
                model.reward([once])
                self.assertTrue(walkable(graph, once))
                done = model.prune(min_count=50)  # every observed edge is under the bar; the rewarded ones stay
                self.assertGreater(done["edges"], 0)
                graph.check_invariants(compressed=True)
                self.assertTrue(walkable(graph, once), f"{cls.kind}: the rewarded text was pruned")
                self.assertFalse(any(walkable(graph, t) for t in TEXTS))
                self.assertEqual(graph.prune_candidates(min_count=50), [])
        # the negative network's whole training is a verdict: what it blamed carries blame, and nothing goes
        model = trained(NegativeNet)
        model.blame([once], reason="other")
        self.assertEqual(model.prune(min_count=50)["edges"], 0)
        self.assertTrue(all(walkable(model.graph, t) for t in TEXTS + [once]))
        # the sine model keeps no record of a verdict - a reward moved its weights - so the thresholds alone decide
        model = trained(RadixNet)
        model.train([once], epochs=1)
        model.reward([once])
        self.assertGreater(model.prune(min_count=50)["edges"], 0)
        self.assertFalse(walkable(model.graph, once))

    def test_the_settings_and_their_refusals(self):
        model = trained(RadixNet)
        self.assertEqual(model.configure_prune()["on"], False)  # nothing given: shown, not switched
        config = model.configure_prune(min_share=0.1)
        self.assertEqual((config["on"], config["min_count"], config["min_share"]), (True, 1, 0.1))
        config = model.configure_prune(min_count=3, every=4, auto=False)
        self.assertEqual((config["min_count"], config["min_share"], config["every"], config["auto"]), (3, 0.1, 4, False))
        self.assertIn("fewer than 3 times", config["rule"])
        for bad in ({"min_count": -1}, {"min_share": 1.0}, {"every": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.configure_prune(**bad)
        self.assertEqual(model.graph.auto_prune, AutoPrune(True, 3, 0.1, 4, False))  # a refusal changes nothing
        config = model.configure_prune(on=False, min_count=9)
        self.assertEqual((config["on"], config["rule"], config["candidates"]), (False, None, None))
        config = model.configure_prune(on=True)
        self.assertEqual((config["min_count"], config["min_share"], config["every"], config["auto"]), (1, 0.0, 1, True))

    def test_off_is_the_old_file_to_the_bit(self):
        plain = trained(RadixNet)
        toggled = trained(RadixNet)
        toggled.configure_prune(on=True, min_count=5)
        self.assertIn("auto_prune", toggled.to_dict()["graph"])
        toggled.configure_prune(on=False)
        self.assertEqual(strip_times(toggled.to_dict()), strip_times(plain.to_dict()))
        self.assertNotIn("auto_prune", plain.to_dict()["graph"])
        self.assertFalse(model_from_dict(plain.to_dict()).graph.auto_prune.on)


if __name__ == "__main__":
    unittest.main()
