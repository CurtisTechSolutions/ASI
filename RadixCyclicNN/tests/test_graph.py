"""Tests for radixnet.graph (RadixCyclicGraph)."""

import json
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.activation import DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K  # noqa: E402
from radixnet.backend import CSR, NodeParams  # noqa: E402
from radixnet.encoding import END_LABEL, START_LABEL, Decoder, Encoder  # noqa: E402
from radixnet.graph import END, START, RadixCyclicGraph  # noqa: E402

ENC = Encoder()
DEC = Decoder()


def round_trip(g: RadixCyclicGraph, text: str) -> str:
    """Walk ``text`` through the trigram index by hand and decode the labels."""
    grams = ENC.encode(text)
    node, off = g.trigram_index[grams[0]]
    assert off == 0, f"first trigram of {text!r} not at offset 0"
    assert START in g.parents[node]
    path = [node]
    for gram in grams[1:]:
        n2, o2 = g.trigram_index[gram]
        if n2 == node and o2 == off + 1:
            off = o2
            continue
        assert off + 3 == len(g.labels[node]), "transition out of the middle of a node"
        assert o2 == 0, "transition into the middle of a node"
        assert n2 in g.children[node], f"missing edge {node}->{n2}"
        path.append(n2)
        node, off = n2, 0
    assert off + 3 == len(g.labels[node])
    assert END in g.children[node]
    # path holds real nodes only; one of them may legitimately be labelled "<s>"
    return DEC.decode_path([g.labels[n] for n in path], 0, include_context=True, skip_sentinels=False)


def assert_transitions_consistent(tc: unittest.TestCase, g: RadixCyclicGraph, trans, n_nodes_visited=None):
    """Every (parent, edge) must be an alive out-edge of that parent."""
    for p, e in trans:
        tc.assertTrue(g.alive[p])
        tc.assertTrue(g.edge_alive[e])
        tc.assertIn(e, g.children[p].values(), f"edge {e} is not an out-edge of {p}")
    tc.assertEqual(trans[0][0], START)
    last_p, last_e = trans[-1]
    tc.assertEqual(g.children[last_p][END], last_e)


class TestBasics(unittest.TestCase):
    def test_fresh_graph(self):
        g = RadixCyclicGraph(seed=3)
        self.assertEqual(g.labels[:2], [START_LABEL, END_LABEL])
        self.assertEqual(g.num_nodes(), 2)
        self.assertEqual(g.num_edges(), 0)
        self.assertEqual(g.num_trigrams(), 0)
        self.assertEqual(g.compression_ratio(), 0.0)
        self.assertEqual(g.alive_nodes(), [START, END])
        self.assertFalse(g.inverted)
        self.assertNotIn(START_LABEL, g.trigram_index)
        g.check_invariants()

    def test_deterministic_seed(self):
        a, b = RadixCyclicGraph(seed=11), RadixCyclicGraph(seed=11)
        for g in (a, b):
            g.observe_sequence(ENC.encode("hello world"))
        self.assertEqual(a.z, b.z)
        self.assertEqual(a.edge_w, b.edge_w)
        c = RadixCyclicGraph(seed=12)
        c.observe_sequence(ENC.encode("hello world"))
        self.assertNotEqual(a.z, c.z)

    def test_new_node_and_edge_ranges(self):
        g = RadixCyclicGraph(seed=5)
        g.observe_sequence(ENC.encode("abcdefgh"))
        for i in g.alive_nodes():
            self.assertTrue(-4.5 <= g.z[i] <= 4.5)
            self.assertEqual((g.a[i], g.b[i], g.h[i], g.k[i]), (DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K))
        for e, ok in enumerate(g.edge_alive):
            if ok:
                self.assertTrue(0.5 <= g.edge_w[e] <= 1.5)

    def test_get_or_create_and_lookup(self):
        g = RadixCyclicGraph()
        self.assertIsNone(g.lookup("abc"))
        n, o = g.get_or_create("abc")
        self.assertEqual(o, 0)
        self.assertEqual(g.labels[n], "abc")
        self.assertEqual(g.get_or_create("abc"), (n, 0))
        self.assertEqual(g.lookup("abc"), (n, 0))
        self.assertEqual(g.num_nodes(), 3)
        with self.assertRaises(ValueError):
            g.get_or_create("ab")
        g.check_invariants()

    def test_activation_of(self):
        g = RadixCyclicGraph(seed=1)
        n, _ = g.get_or_create("xyz")
        import math
        self.assertAlmostEqual(g.activation_of(n), -math.sin(g.z[n] / 3.0), places=15)
        self.assertAlmostEqual(g.activation_of(START), -math.sin(g.z[START] / 3.0), places=15)

    def test_observe_empty(self):
        g = RadixCyclicGraph()
        v = g.version
        self.assertEqual(g.observe_sequence([]), [])
        self.assertEqual(g.version, v)

    def test_observe_single_text(self):
        g = RadixCyclicGraph(seed=2)
        text = "the cat"
        trans = g.observe_sequence(ENC.encode(text))
        # START -> 5 trigram nodes -> END : 6 transitions
        self.assertEqual(len(trans), 6)
        assert_transitions_consistent(self, g, trans)
        self.assertEqual(g.num_nodes(), 2 + 5)
        self.assertEqual(g.num_edges(), 6)
        self.assertEqual(g.num_trigrams(), 5)
        self.assertEqual(g.count[START], 1)
        self.assertEqual(g.count[END], 1)
        for n in g.alive_nodes():
            self.assertEqual(g.count[n], 1)
        for p, e in trans:
            self.assertEqual(g.edge_count[e], 1)
        g.check_invariants(texts=[text])
        self.assertEqual(round_trip(g, text), text)
        self.assertEqual(g.node_path(ENC.encode(text)), [START] + [p for p, _ in trans[1:]] + [END])

    def test_observe_count_false(self):
        g = RadixCyclicGraph(seed=2)
        g.observe_sequence(ENC.encode("the cat"))
        counts = list(g.count)
        ecounts = list(g.edge_count)
        trans = g.observe_sequence(ENC.encode("the cat"), count=False)
        self.assertEqual(g.count, counts)
        self.assertEqual(g.edge_count, ecounts)
        self.assertEqual(len(trans), 6)
        trans2 = g.observe_sequence(ENC.encode("the cat"))
        self.assertEqual(trans, trans2)
        self.assertEqual(g.count[START], 2)

    def test_bad_trigrams_rejected(self):
        g = RadixCyclicGraph()
        with self.assertRaises(ValueError):
            g.observe_sequence(["abc", "xyz"])
        with self.assertRaises(ValueError):
            g.observe_sequence(["abcd"])


class TestCyclesAndRepeats(unittest.TestCase):
    def test_self_loop(self):
        g = RadixCyclicGraph(seed=4)
        trans = g.observe_sequence(ENC.encode("aaaa"))
        n, _ = g.lookup("aaa")
        self.assertIn(n, g.children[n])
        self.assertEqual(len(trans), 3)
        self.assertEqual(g.count[n], 2)
        g.check_invariants(texts=["aaaa"])
        self.assertEqual(g.compress(), 0)  # a self-loop is never merged
        g.check_invariants(texts=["aaaa"], compressed=True)
        self.assertEqual(round_trip(g, "aaaa"), "aaaa")
        self.assertEqual(round_trip(g, "aaaaaaa"), "aaaaaaa")  # the loop represents any repetition

    def test_cycle(self):
        g = RadixCyclicGraph(seed=4)
        g.observe_sequence(ENC.encode("abcabc"))
        g.check_invariants(texts=["abcabc"])
        merges = g.compress()
        self.assertGreater(merges, 0)
        g.check_invariants(texts=["abcabc"], compressed=True)
        labels = sorted(g.labels[i] for i in g.alive_nodes() if i > END)
        self.assertEqual(labels, ["abc", "bcab"])
        abc, _ = g.lookup("abc")
        bcab, _ = g.lookup("bca")
        self.assertIn(bcab, g.children[abc])
        self.assertIn(abc, g.children[bcab])
        self.assertIn(END, g.children[abc])
        self.assertEqual(round_trip(g, "abcabc"), "abcabc")
        # the cycle also represents longer repetitions structurally
        self.assertEqual(round_trip(g, "abcabcabc"), "abcabcabc")


class TestSplit(unittest.TestCase):
    def _compressed_chain(self):
        g = RadixCyclicGraph(seed=8)
        g.observe_sequence(ENC.encode("abcdefg"))
        g.compress()
        (n,) = [i for i in g.alive_nodes() if i > END]
        self.assertEqual(g.labels[n], "abcdefg")
        return g, n

    def test_split_moves_out_edges_and_index(self):
        g, n = self._compressed_chain()
        g.count[n] = 7
        e_end = g.children[n][END]
        version = g.version
        sv = g.structure_version
        a, b = g.split(n, 2)
        self.assertEqual(a, n)
        self.assertEqual(g.labels[a], "abcd")
        self.assertEqual(g.labels[b], "cdefg")
        self.assertEqual(g.children[b][END], e_end)  # edge id preserved
        self.assertEqual(g.parents[END][b], e_end)
        self.assertNotIn(a, g.parents[END])
        e_ab = g.children[a][b]
        self.assertEqual(g.edge_count[e_ab], 7)
        self.assertEqual(g.count[b], 7)
        self.assertEqual((g.z[b], g.a[b], g.b[b], g.h[b], g.k[b]), (g.z[a], g.a[a], g.b[a], g.h[a], g.k[a]))
        self.assertEqual(g.lookup("abc"), (a, 0))
        self.assertEqual(g.lookup("bcd"), (a, 1))
        self.assertEqual(g.lookup("cde"), (b, 0))
        self.assertEqual(g.lookup("efg"), (b, 2))
        self.assertGreater(g.version, version)
        self.assertGreater(g.structure_version, sv)
        g.check_invariants(texts=["abcdefg"])
        self.assertEqual(round_trip(g, "abcdefg"), "abcdefg")

    def test_split_bounds(self):
        g, n = self._compressed_chain()
        for bad in (0, 5, 6, -1):
            with self.assertRaises(ValueError):
                g.split(n, bad)
        with self.assertRaises(ValueError):
            g.split(START, 1)
        with self.assertRaises(ValueError):
            g.split(END, 1)
        g.split(n, 4)  # last legal index: A="abcdef", B="defg"
        self.assertEqual(g.labels[n], "abcdef")
        with self.assertRaises(ValueError):
            g.split(999, 1)

    def test_split_then_compress_restores(self):
        g, n = self._compressed_chain()
        before = g.to_dict()
        g.split(n, 3)
        g.split(n, 1)
        self.assertEqual(g.num_nodes(), 5)
        g.check_invariants(texts=["abcdefg"])
        self.assertEqual(g.compress(), 2)
        g.check_invariants(texts=["abcdefg"], compressed=True)
        after = g.to_dict()
        self.assertEqual(after["nodes"]["labels"], before["nodes"]["labels"])
        self.assertEqual(after["nodes"]["z"], before["nodes"]["z"])
        self.assertEqual((after["edges"]["src"], after["edges"]["dst"]), (before["edges"]["src"], before["edges"]["dst"]))

    def test_split_self_loop(self):
        g = RadixCyclicGraph(seed=9)
        g.observe_sequence(ENC.encode("ababab"))
        g.compress()
        # "abab" with a self loop (abab -> abab), plus START/END edges
        n, _ = g.lookup("aba")
        self.assertEqual(g.labels[n], "abab")
        self.assertIn(n, g.children[n])
        a, b = g.split(n, 1)
        self.assertEqual((g.labels[a], g.labels[b]), ("aba", "bab"))
        self.assertIn(a, g.children[b])  # the loop is now B -> A
        self.assertIn(b, g.children[a])
        g.check_invariants(texts=["ababab"])
        self.assertEqual(round_trip(g, "ababab"), "ababab")


class TestMerge(unittest.TestCase):
    def test_merge_child_conditions(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(ENC.encode("abcd"))
        g.observe_sequence(ENC.encode("xbcd"))
        abc, _ = g.lookup("abc")
        bcd, _ = g.lookup("bcd")
        xbc, _ = g.lookup("xbc")
        self.assertFalse(g.merge_child(START))
        self.assertFalse(g.merge_child(END))
        self.assertFalse(g.merge_child(abc))  # bcd has two parents
        self.assertFalse(g.merge_child(bcd))  # child is END
        self.assertFalse(g.merge_child(xbc))
        self.assertEqual(g.compress(), 0)
        g.check_invariants(texts=["abcd", "xbcd"], compressed=True)

    def test_merge_details(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(ENC.encode("abcde"))
        abc, _ = g.lookup("abc")
        bcd, _ = g.lookup("bcd")
        cde, _ = g.lookup("cde")
        g.count[abc], g.count[bcd] = 2, 5
        e_out = g.children[bcd][cde]
        e_in = g.children[abc][bcd]
        n_edges = g.num_edges()
        self.assertTrue(g.merge_child(abc))
        self.assertEqual(g.labels[abc], "abcd")
        self.assertFalse(g.alive[bcd])
        self.assertEqual(g.children[bcd], {})
        self.assertEqual(g.parents[bcd], {})
        self.assertFalse(g.edge_alive[e_in])
        self.assertEqual(g.children[abc][cde], e_out)
        self.assertEqual(g.parents[cde][abc], e_out)
        self.assertNotIn(bcd, g.parents[cde])
        self.assertEqual(g.count[abc], 5)
        self.assertEqual(g.lookup("bcd"), (abc, 1))
        self.assertEqual(g.num_edges(), n_edges - 1)
        g.check_invariants(texts=["abcde"])

    def test_compress_chain(self):
        g = RadixCyclicGraph(seed=1)
        text = "the quick brown fox"
        g.observe_sequence(ENC.encode(text))
        n_before = g.num_nodes()
        merges = g.compress()
        self.assertEqual(merges, n_before - 3)
        self.assertEqual(g.num_nodes(), 3)
        (n,) = [i for i in g.alive_nodes() if i > END]
        self.assertEqual(g.labels[n], text)
        self.assertEqual(g.compression_ratio(), len(text) - 2)
        self.assertEqual(g.compress(), 0)
        g.check_invariants(texts=[text], compressed=True)

    def test_merge_creating_self_loop(self):
        # "abcab": abc -> bca -> cab -> abc cycle; abc also -> END and START -> abc.
        # bca (parent abc only) / cab chain merges into "bcab" with edge back to abc.
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(ENC.encode("abcabc"))
        self.assertEqual(g.compress(), 1)
        g.check_invariants(texts=["abcabc"], compressed=True)


class TestRandomisedInvariants(unittest.TestCase):
    """Hundreds of random observe / split / compress operations with invariants after each."""

    def test_random_operations(self):
        alphabets = ["ab", "abc", "aab", "ab c", "abcd"]
        total_ops = 0
        for seed in range(6):
            rng = random.Random(1000 + seed)
            alphabet = alphabets[seed % len(alphabets)]
            g = RadixCyclicGraph(seed=seed)
            observed = []
            for _ in range(110):
                op = rng.random()
                if op < 0.6:
                    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 14)))
                    trans = g.observe_sequence(ENC.encode(text), count=rng.random() < 0.8)
                    observed.append(text)
                    assert_transitions_consistent(self, g, trans)
                    self.assertEqual(g.node_path(ENC.encode(text)), [START] + [p for p, _ in trans[1:]] + [END])
                elif op < 0.85:
                    candidates = [n for n in g.alive_nodes() if n > END and len(g.labels[n]) >= 4]
                    if candidates:
                        n = rng.choice(candidates)
                        a, b = g.split(n, rng.randint(1, len(g.labels[n]) - 3))
                        self.assertEqual(a, n)
                        self.assertTrue(g.alive[b])
                else:
                    g.compress()
                    g.check_invariants(texts=observed, compressed=True)
                    for text in observed:
                        self.assertEqual(round_trip(g, text), text)
                g.check_invariants(texts=observed)
                total_ops += 1
            g.compress()
            g.check_invariants(texts=observed, compressed=True)
            for text in observed:
                self.assertEqual(round_trip(g, text), text)
            # re-observing after compress is a structural no-op
            sv = g.structure_version
            for text in observed:
                trans = g.observe_sequence(ENC.encode(text), count=False)
                assert_transitions_consistent(self, g, trans)
            self.assertEqual(g.structure_version, sv)
            # every trigram of every observed text lives in exactly one node
            for text in observed:
                for gram in ENC.encode(text):
                    n, o = g.trigram_index[gram]
                    self.assertTrue(g.alive[n])
                    self.assertEqual(g.labels[n][o:o + 3], gram)
        self.assertGreaterEqual(total_ops, 600)

    def test_check_invariants_detects_corruption(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(ENC.encode("abcdef"))
        g.check_invariants()
        n, _ = g.lookup("bcd")
        g.trigram_index["bcd"] = (n, 5)
        with self.assertRaises(AssertionError):
            g.check_invariants()
        g.trigram_index["bcd"] = (n, 0)
        g.check_invariants()
        e = g.children[n].popitem()[1]
        with self.assertRaises(AssertionError):
            g.check_invariants()


class TestScoresAndCache(unittest.TestCase):
    def setUp(self):
        self.g = RadixCyclicGraph(seed=21)
        for t in ["the cat sat", "the dog ran", "the cat ran"]:
            self.g.observe_sequence(ENC.encode(t))
        self.g.compress()

    def test_child_scores_and_probs(self):
        g = self.g
        n, _ = g.lookup("the")
        scores = g.child_scores(n)
        self.assertGreater(len(scores), 1)
        fp = g.activation_of(n)
        for c, s in scores:
            e = g.children[n][c]
            self.assertAlmostEqual(s, g.edge_w[e] * fp * g.activation_of(c), places=12)
        probs = g.child_probs(n)
        self.assertAlmostEqual(sum(p for _, p in probs), 1.0, places=12)
        self.assertEqual([c for c, _ in probs], [c for c, _ in scores])
        self.assertEqual(g.child_probs(END), [])
        self.assertEqual(g.child_costs(END), [])

    def test_child_costs_match_probs_and_cache(self):
        import math
        g = self.g
        n, _ = g.lookup("the")
        probs = dict(g.child_probs(n))
        costs = g.child_costs(n)
        for c, e, cost in costs:
            self.assertEqual(g.children[n][c], e)
            self.assertAlmostEqual(cost, -math.log(probs[c]), places=12)
            self.assertGreaterEqual(cost, 0.0)
        self.assertIs(g.child_costs(n), costs)  # cached while the version is unchanged
        g.edge_w[costs[0][1]] += 1.0
        self.assertIs(g.child_costs(n), costs)  # no version bump -> cache still used
        g.version += 1
        fresh = g.child_costs(n)
        self.assertIsNot(fresh, costs)
        self.assertNotAlmostEqual(fresh[0][2], costs[0][2], places=6)

    def test_version_bumps(self):
        g = self.g
        v, sv = g.version, g.structure_version
        csr = g.to_csr()
        g.apply_csr_weights(csr, list(csr.weights))
        self.assertEqual(g.version, v + 1)
        self.assertEqual(g.structure_version, sv)
        g.apply_node_params(g.node_params())
        self.assertEqual(g.version, v + 2)
        g.invert()
        self.assertEqual(g.version, v + 3)
        self.assertEqual(g.structure_version, sv)
        g.observe_sequence(ENC.encode("zzz"))
        self.assertGreater(g.structure_version, sv)


class TestCSR(unittest.TestCase):
    def test_export_import(self):
        g = RadixCyclicGraph(seed=13)
        for t in ["abcabc", "abcd", "xyz", "aaaa"]:
            g.observe_sequence(ENC.encode(t))
        g.compress()
        csr = g.to_csr()
        self.assertIsInstance(csr, CSR)
        self.assertEqual(csr.n_nodes, len(g.labels))
        self.assertEqual(csr.n_edges, g.num_edges())
        for p in range(len(g.labels)):
            row = csr.row(p)
            self.assertEqual(len(row), len(g.children[p]))
            for j in row:
                c, e = csr.indices[j], csr.edge_ids[j]
                self.assertEqual(g.children[p][c], e)
                self.assertEqual(csr.weights[j], g.edge_w[e])
                self.assertEqual(csr.edge_pos[e], j)
        new_w = [w + 0.25 for w in csr.weights]
        g.apply_csr_weights(csr, new_w)
        for j, e in enumerate(csr.edge_ids):
            self.assertEqual(g.edge_w[e], new_w[j])
        with self.assertRaises(ValueError):
            g.apply_csr_weights(csr, new_w[:-1])
        g.check_invariants()

    def test_node_params_round_trip(self):
        g = RadixCyclicGraph(seed=13)
        g.observe_sequence(ENC.encode("hello"))
        params = g.node_params()
        self.assertIsInstance(params, NodeParams)
        self.assertEqual(len(params), len(g.labels))
        self.assertIsNot(params.z, g.z)
        params.z[2] += 1.0
        params.k[2] = 0.5
        g.apply_node_params(params)
        self.assertEqual(g.z[2], params.z[2])
        self.assertEqual(g.k[2], 0.5)
        with self.assertRaises(ValueError):
            g.apply_node_params(NodeParams([0.0], [0.0], [0.0], [0.0], [0.0]))


class TestSerialisation(unittest.TestCase):
    def _random_graph(self, seed):
        rng = random.Random(seed)
        g = RadixCyclicGraph(seed=seed)
        texts = ["".join(rng.choice("abc ") for _ in range(rng.randint(3, 12))) for _ in range(15)]
        for t in texts:
            g.observe_sequence(ENC.encode(t))
        g.compress()
        for t in texts[:5]:
            g.observe_sequence(ENC.encode(t))
        return g, texts

    def test_to_dict_from_dict_equality(self):
        g, texts = self._random_graph(31)
        d = g.to_dict()
        json.dumps(d)  # JSON-serialisable
        g2 = RadixCyclicGraph.from_dict(d)
        self.assertEqual(g2.to_dict(), d)
        g2.check_invariants(texts=texts)
        self.assertEqual(g2.labels[:2], [START_LABEL, END_LABEL])
        self.assertEqual(g2.num_nodes(), g.num_nodes())
        self.assertEqual(g2.num_edges(), g.num_edges())
        self.assertEqual(g2.num_trigrams(), g.num_trigrams())
        self.assertEqual(g2.inverted, g.inverted)
        self.assertEqual(g2.seed, g.seed)
        # dead nodes were compacted away
        self.assertEqual(len(g2.labels), g.num_nodes())
        self.assertTrue(all(g2.alive))
        # probabilities are preserved (compare per label)
        for p in g.alive_nodes():
            p2 = g2.lookup(g.labels[p][:3])[0] if p > END else p
            probs = {g.labels[c]: pr for c, pr in g.child_probs(p)}
            probs2 = {g2.labels[c]: pr for c, pr in g2.child_probs(p2)}
            self.assertEqual(set(probs), set(probs2))
            for label in probs:
                self.assertAlmostEqual(probs[label], probs2[label], places=12)
            self.assertEqual(g.count[p], g2.count[p2])

    def test_rng_state_restored(self):
        g, _ = self._random_graph(32)
        g2 = RadixCyclicGraph.from_dict(g.to_dict())
        t1 = g.observe_sequence(ENC.encode("qqrrss"))
        t2 = g2.observe_sequence(ENC.encode("qqrrss"))
        self.assertEqual(len(t1), len(t2))
        self.assertEqual([g.z[n] for n in g.alive_nodes()[-4:]], [g2.z[n] for n in g2.alive_nodes()[-4:]])
        self.assertEqual(sorted(g.edge_w[e] for e in g.children[START].values()),
                         sorted(g2.edge_w[e] for e in g2.children[START].values()))

    def test_from_dict_rejects_garbage(self):
        with self.assertRaises(ValueError):
            RadixCyclicGraph.from_dict({"format": "nope"})
        d = RadixCyclicGraph(seed=1).to_dict()
        d["nodes"]["labels"] = ["x", "y"]
        with self.assertRaises(ValueError):
            RadixCyclicGraph.from_dict(d)


class TestInvert(unittest.TestCase):
    def test_invert_flips_signs(self):
        g = RadixCyclicGraph(seed=17)
        for t in ["the cat sat", "the dog"]:
            g.observe_sequence(ENC.encode(t))
        w = list(g.edge_w)
        a = list(g.a)
        v = g.version
        g.invert()
        self.assertTrue(g.inverted)
        self.assertEqual(g.version, v + 1)
        self.assertEqual(g.edge_w, [-x for x in w])
        self.assertEqual(g.a, [-x for x in a])
        self.assertEqual(g.a[START], -DEFAULT_A)
        # new nodes / edges created while inverted follow the inverted convention
        n = len(g.labels)
        g.observe_sequence(ENC.encode("zzzz"))
        self.assertEqual(g.a[n], -DEFAULT_A)
        for e in g.children[n].values():
            self.assertTrue(-1.5 <= g.edge_w[e] <= -0.5)
        g.check_invariants(texts=["the cat sat", "the dog", "zzzz"])
        g.invert()
        self.assertFalse(g.inverted)
        self.assertEqual(g.edge_w[: len(w)], w)
        self.assertEqual(g.a[:n], a)
        self.assertEqual(g.a[n], DEFAULT_A)
        d = g.to_dict()
        self.assertFalse(d["inverted"])
        g.invert()
        self.assertTrue(RadixCyclicGraph.from_dict(g.to_dict()).inverted)


class TestSentinelLookalikeLabels(unittest.TestCase):
    def test_round_trip_of_texts_containing_sentinel_labels(self):
        texts = ["x<s>y", "a</s>b", "<s>", "</s>"]
        g = RadixCyclicGraph(seed=1)
        for i, t in enumerate(texts):
            g.observe_sequence(ENC.encode(t))
            g.check_invariants(texts=texts[: i + 1])
        g.compress()
        g.check_invariants(texts=texts, compressed=True)
        self.assertEqual(g.labels[START], START_LABEL)
        self.assertEqual(g.labels[END], END_LABEL)
        for trigram in ("<s>", "</s", "/s>"):
            node, off = g.lookup(trigram)
            self.assertGreater(node, END)
            self.assertEqual(g.labels[node][off : off + 3], trigram)
        self.assertIn(START_LABEL, g.labels[2:])
        self.assertIn(END_LABEL, g.labels[2:])
        clone = RadixCyclicGraph.from_dict(json.loads(json.dumps(g.to_dict())))
        clone.check_invariants(texts=texts, compressed=True)
        for t in texts:
            self.assertEqual(round_trip(clone, t), t)


if __name__ == "__main__":
    unittest.main()


class TestMergePreservesFunction(unittest.TestCase):
    """Merging a unary chain must not change any edge score / cost (network function)."""

    TEXTS = ["the quick brown fox", "the quick brown dog", "a quick brown fox", "the slow brown fox", "fox fox fox"]

    @staticmethod
    def edge_costs(g):
        """Cost of every alive edge keyed by (parent's last trigram, child's first trigram)."""
        out = {}
        for p in g.alive_nodes():
            for c, _e, cost in g.child_costs(p):
                key = (g.labels[p] if p <= END else g.labels[p][-3:], g.labels[c] if c <= END else g.labels[c][:3])
                out[key] = cost
        return out

    def test_child_costs_unchanged_by_compress(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                g = RadixCyclicGraph(seed=seed)
                for t in self.TEXTS:
                    g.observe_sequence(ENC.encode(t))
                rng = random.Random(seed)
                for i in range(len(g.labels)):  # activations of every sign and size
                    g.z[i] = rng.uniform(-6.0, 6.0)
                    g.a[i] = rng.uniform(-1.5, 1.5)
                    g.b[i] = rng.uniform(0.2, 0.8)
                    g.h[i] = rng.uniform(-1.0, 1.0)
                    g.k[i] = rng.uniform(-0.3, 0.3)
                for e in range(len(g.edge_w)):
                    g.edge_w[e] = rng.uniform(-2.0, 2.0)
                g.version += 1
                before = self.edge_costs(g)
                weights_before = max(abs(w) for w in g.edge_w)
                merges = g.compress()
                self.assertGreater(merges, 0)
                after = self.edge_costs(g)
                for key, cost in after.items():
                    self.assertIn(key, before)
                    self.assertAlmostEqual(cost, before[key], places=9, msg=key)
                for key in set(before) - set(after):  # only the (deterministic) chain edges vanished
                    self.assertEqual(before[key], 0.0, key)
                self.assertLessEqual(max(abs(w) for w in g.edge_w), weights_before + 1e-12)
                g.check_invariants(texts=self.TEXTS, compressed=True)
