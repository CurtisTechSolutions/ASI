"""Tests for radixnet.search (Dijkstra prediction and stochastic sampling)."""

import json
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.backend import PythonBackend  # noqa: E402
from radixnet.encoding import END_LABEL, START_LABEL, Encoder  # noqa: E402
from radixnet.graph import END, START, RadixCyclicGraph  # noqa: E402
from radixnet.search import PathResult, dijkstra_predict, sample_walk  # noqa: E402

ENC = Encoder()


def graph_from(texts, seed=1, compress=True):
    g = RadixCyclicGraph(seed=seed)
    for t in texts:
        g.observe_sequence(ENC.encode(t))
    if compress:
        g.compress()
    return g


def train(g, texts, steps=300, lr=0.3, act_lr=0.03):
    """A few local-rule steps so the observed transitions become the cheapest."""
    trans = []
    for t in texts:
        trans.extend(g.observe_sequence(ENC.encode(t), count=False))
    csr = g.to_csr()
    parents = [p for p, _ in trans]
    positions = [csr.edge_pos[e] for _, e in trans]
    be = PythonBackend()
    state = be.prepare(csr, g.node_params())
    loss = None
    for _ in range(steps):
        loss = be.step(state, parents, positions, lr, act_lr)
    w, params = be.finalize(state)
    g.apply_csr_weights(csr, w)
    g.apply_node_params(params)
    return loss


def path_cost(g, node_ids, step_penalty=0.0):
    total = 0.0
    for p, c in zip(node_ids, node_ids[1:]):
        total += next(cost for cc, _, cost in g.child_costs(p) if cc == c) + step_penalty
    return total


def brute_force(g, node, chars, min_chars, max_chars, step_penalty, to_end):
    """Minimum goal cost over all paths, mirroring the Dijkstra goal / expansion rules."""
    best = math.inf

    def rec(n, ch, cost):
        nonlocal best
        if n == END or (not to_end and ch >= min_chars):
            best = min(best, cost)
            return
        if ch >= max_chars:
            return
        for c, _, ec in g.child_costs(n):
            nch = ch if c == END else ch + len(g.labels[c]) - 2
            rec(c, nch, cost + ec + step_penalty)

    rec(node, chars, 0.0)
    return best


class TestPathResult(unittest.TestCase):
    def test_to_dict_json(self):
        r = PathResult(text="ab", labels=["abc"], node_ids=[2], cost=0.5, step_costs=[0.5], expanded=1, reached_end=False)
        d = r.to_dict()
        json.dumps(d)
        self.assertEqual(d["text"], "ab")
        self.assertEqual(d["full_text"], "")
        r.full_text = "xab"
        self.assertEqual(r.to_dict()["full_text"], "xab")
        with self.assertRaises(AttributeError):
            r.extra = 1  # type: ignore[attr-defined]


class TestDijkstra(unittest.TestCase):
    def test_single_text_continuation(self):
        text = "the quick brown fox"  # every trigram unique -> one compressed node
        g = graph_from([text])
        n, o = g.lookup("the")
        self.assertEqual(g.labels[n], text)
        r = dijkstra_predict(g, n, o, min_chars=5, max_chars=40)
        # the start node's own remainder already satisfies min_chars: it is the goal
        self.assertEqual(r.text, " quick brown fox")
        self.assertEqual(r.node_ids, [n])
        self.assertFalse(r.reached_end)
        self.assertEqual(r.cost, 0.0)
        r = dijkstra_predict(g, n, o, min_chars=5, max_chars=40, to_end=True)
        self.assertEqual(r.text, " quick brown fox")
        self.assertTrue(r.reached_end)
        self.assertEqual(r.node_ids, [n, END])
        self.assertEqual(r.labels[-1], END_LABEL)
        self.assertAlmostEqual(r.cost, sum(r.step_costs), places=12)
        self.assertEqual(r.step_costs, [0.0])  # END is the only child
        # matching a later trigram emits only the remainder after it
        n, o = g.lookup("row")
        r = dijkstra_predict(g, n, o, 3, 40)
        self.assertEqual(r.text, "n fox")
        # truncation to max_chars
        r = dijkstra_predict(g, n, o, 3, 4)
        self.assertEqual(r.text, "n fo")
        # a repeated trigram makes a cycle; the search still terminates and decodes
        g = graph_from(["the cat sat on the mat"])
        n, o = g.lookup("the")
        r = dijkstra_predict(g, n, o, min_chars=3, max_chars=40, to_end=True)
        self.assertTrue(r.reached_end)
        self.assertIn(r.text, (" mat", " cat sat on the mat"))

    def test_returns_trained_sequence(self):
        texts = ["the quick brown fox"] * 3 + ["the quick brown dog", "the slow brown fox"]
        g = graph_from(texts)
        train(g, texts)
        n, o = g.lookup("the")
        r = dijkstra_predict(g, n, o, min_chars=16, max_chars=30)
        self.assertEqual(r.text, " quick brown fox")
        self.assertFalse(r.reached_end)  # the 16-char goal pops one hop before END
        self.assertAlmostEqual(r.cost, path_cost(g, r.node_ids), places=12)
        r = dijkstra_predict(g, n, o, min_chars=16, max_chars=30, to_end=True)
        self.assertEqual(r.text, " quick brown fox")
        self.assertTrue(r.reached_end)
        # a smaller min_chars is satisfied earlier: the goal is the first state with enough chars
        r = dijkstra_predict(g, n, o, min_chars=10, max_chars=30)
        self.assertEqual(r.text, " quick brown ")
        self.assertFalse(r.reached_end)

    def test_to_end(self):
        texts = ["the quick fox", "the quick fox jumps"]
        g = graph_from(texts)
        n, o = g.lookup("the")
        r = dijkstra_predict(g, n, o, min_chars=1, max_chars=40, to_end=True)
        self.assertTrue(r.reached_end)
        self.assertIn(r.text, (" quick fox", " quick fox jumps"))
        self.assertAlmostEqual(r.cost, path_cost(g, r.node_ids), places=12)
        # a different min_chars does not change a to_end search
        r2 = dijkstra_predict(g, n, o, min_chars=30, max_chars=40, to_end=True)
        self.assertEqual(r2.node_ids, r.node_ids)

    def test_from_start_includes_full_first_label(self):
        text = "hello world"
        g = graph_from([text])
        r = dijkstra_predict(g, START, 0, 3, 40, to_end=True)
        self.assertEqual(r.text, text)
        self.assertEqual(r.labels[0], START_LABEL)
        self.assertTrue(r.reached_end)
        r = sample_walk(g, START, 0, 40)
        self.assertEqual(r.text, text)
        r = dijkstra_predict(g, START, 0, 3, 40, to_end=True, include_context=False)
        self.assertEqual(r.text, text[3:])

    def test_fallback_on_dead_end(self):
        g = RadixCyclicGraph(seed=1)
        n, _ = g.get_or_create("abc")  # isolated node: no path anywhere
        r = dijkstra_predict(g, n, 0, min_chars=5, max_chars=10)
        self.assertEqual(r.text, "")
        self.assertEqual(r.node_ids, [n])
        self.assertFalse(r.reached_end)
        self.assertEqual(r.cost, 0.0)
        r = dijkstra_predict(g, n, 0, 5, 10, to_end=True)
        self.assertFalse(r.reached_end)
        # a chain that stops short of min_chars returns the longest path
        g = graph_from(["abcd"])
        n, _ = g.lookup("abc")
        (m,) = [i for i in g.alive_nodes() if i > END]
        q = g._new_node("cdxyz")
        g.children[m].clear()
        g.parents[END].clear()
        g._new_edge(m, q)
        g.trigram_index.update({"cdx": (q, 0), "dxy": (q, 1), "xyz": (q, 2)})
        r = dijkstra_predict(g, m, 0, min_chars=50, max_chars=100)
        self.assertEqual(r.node_ids, [m, q])
        self.assertEqual(r.text, "dxyz")
        self.assertFalse(r.reached_end)

    def test_expansion_cap(self):
        g = graph_from(["the cat sat on the mat", "the dog sat"])
        n, o = g.lookup("the")
        r = dijkstra_predict(g, n, o, 10, 40, max_expansions=1)
        self.assertEqual(r.expanded, 1)
        self.assertEqual(r.node_ids, [n])
        self.assertIsInstance(r.text, str)

    def test_validation(self):
        g = graph_from(["abcdef"])
        n, _ = g.lookup("abc")
        with self.assertRaises(ValueError):
            dijkstra_predict(g, n, 0, 1, 5, step_penalty=-0.1)
        with self.assertRaises(ValueError):
            dijkstra_predict(g, n, 9, 1, 5)
        with self.assertRaises(ValueError):
            dijkstra_predict(g, 999, 0, 1, 5)

    def test_step_penalty_prefers_fewer_steps(self):
        # "abcd" via one compressed node vs "ab" then "bcd" via two nodes
        texts = ["xabcd", "xabyz", "wabcd"]
        g = graph_from(texts)
        n, o = g.lookup("xab")
        r0 = dijkstra_predict(g, n, o, 2, 10, step_penalty=0.0)
        r1 = dijkstra_predict(g, n, o, 2, 10, step_penalty=5.0)
        self.assertLessEqual(len(r1.node_ids), len(r0.node_ids))
        self.assertAlmostEqual(r1.cost, path_cost(g, r1.node_ids, 5.0), places=12)

    def test_optimality_against_brute_force(self):
        checked = 0
        for seed in range(25):
            rng = random.Random(500 + seed)
            alphabet = ["ab", "abc", "aab"][seed % 3]
            texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(3, 7))) for _ in range(rng.randint(2, 5))]
            g = graph_from(texts, seed=seed)
            for i in range(len(g.labels)):  # spread the costs out a little
                g.k[i] += rng.uniform(-0.5, 0.5)
            g.version += 1
            starts = [(START, 0)]
            for n in g.alive_nodes():
                if n > END:
                    starts.append((n, rng.randint(0, len(g.labels[n]) - 3)))
            for node, off in starts:
                for to_end in (False, True):
                    min_chars = rng.randint(1, 5)
                    max_chars = min_chars + rng.randint(0, 3)
                    penalty = rng.choice([0.0, 0.25])
                    start_chars = 0 if node <= END else len(g.labels[node]) - off - 3
                    expect = brute_force(g, node, start_chars, min_chars, max_chars, penalty, to_end)
                    r = dijkstra_predict(g, node, off, min_chars, max_chars, step_penalty=penalty, to_end=to_end)
                    self.assertAlmostEqual(r.cost, path_cost(g, r.node_ids, penalty), places=9)
                    if math.isinf(expect):
                        self.assertFalse(r.reached_end)
                    else:
                        self.assertAlmostEqual(r.cost, expect, places=9, msg=f"seed={seed} start={(node, off)} to_end={to_end}")
                        if to_end:
                            self.assertTrue(r.reached_end)
                        else:
                            emitted = start_chars + sum(len(g.labels[c]) - 2 for c in r.node_ids[1:] if c != END)
                            self.assertTrue(emitted >= min_chars or r.reached_end)
                    checked += 1
        self.assertGreater(checked, 100)


class TestSampleWalk(unittest.TestCase):
    def test_terminates_and_is_seeded(self):
        texts = ["the cat sat on the mat", "the dog sat on the rug", "aaaa", "abcabc"]
        g = graph_from(texts)
        n, o = g.lookup("the")
        r1 = sample_walk(g, n, o, 25, rng=random.Random(3))
        r2 = sample_walk(g, n, o, 25, rng=random.Random(3))
        self.assertEqual(r1.node_ids, r2.node_ids)
        self.assertEqual(r1.text, r2.text)
        self.assertLessEqual(len(r1.text), 25)
        self.assertAlmostEqual(r1.cost, sum(r1.step_costs), places=12)
        self.assertAlmostEqual(r1.cost, path_cost(g, r1.node_ids), places=12)
        self.assertEqual(r1.expanded, len(r1.node_ids) - 1)
        # self loops: max_chars stops the walk
        a, _ = g.lookup("aaa")
        for _ in range(20):
            r = sample_walk(g, a, 0, 10, temperature=2.0, rng=random.Random(_))
            self.assertLessEqual(len(r.text), 10)
            self.assertTrue(r.reached_end or len(r.text) >= 10 - 2)

    def test_greedy_matches_cheapest_edges(self):
        texts = ["the quick brown fox"] * 3 + ["the quick brown dog"]
        g = graph_from(texts)
        train(g, texts)
        n, o = g.lookup("the")
        r = sample_walk(g, n, o, 40, temperature=0.0)
        self.assertEqual(r.text, " quick brown fox")
        self.assertTrue(r.reached_end)
        with self.assertRaises(ValueError):
            sample_walk(g, n, o, 40, temperature=-1.0)

    def test_default_rng_is_graph_rng(self):
        g = graph_from(["the cat sat on the mat", "the dog sat on the rug"])
        n, o = g.lookup("the")
        g2 = RadixCyclicGraph.from_dict(g.to_dict())
        n2, o2 = g2.lookup("the")
        r1 = sample_walk(g, n, o, 30)
        r2 = sample_walk(g2, n2, o2, 30)
        self.assertEqual(r1.text, r2.text)

    def test_dead_end_and_stop_at_end(self):
        g = RadixCyclicGraph(seed=1)
        n, _ = g.get_or_create("abc")
        r = sample_walk(g, n, 0, 10)
        self.assertEqual((r.text, r.node_ids, r.cost), ("", [n], 0.0))
        g = graph_from(["hello"])
        n, o = g.lookup("hel")
        r = sample_walk(g, n, o, 10, stop_at_end=False)
        self.assertTrue(r.reached_end)
        self.assertEqual(r.text, "lo")


class TestSentinelLookalikes(unittest.TestCase):
    """Real nodes labelled "<s>" / "</s>" must be emitted, not mistaken for START/END."""

    @staticmethod
    def decode_by_id(g, result, include_context):
        real = [g.labels[n] for n in result.node_ids if n != START and n != END]
        if not real:
            return ""
        head = real[0] if include_context else real[0][3:]
        return head + "".join(label[2:] for label in real[1:])

    def test_results_keep_lookalike_labels(self):
        g = graph_from(["x<s>y", "a</s>b", "<s>"], compress=True)
        node, off = g.lookup("<s>")
        self.assertGreater(node, END)
        self.assertEqual((g.labels[node], off), (START_LABEL, 0))
        x_node = g.lookup("x<s")[0]
        r = dijkstra_predict(g, x_node, 0, min_chars=1, max_chars=1)
        self.assertEqual(r.labels, ["x<s", START_LABEL])
        self.assertEqual(r.text, ">")
        r = sample_walk(g, x_node, 0, max_chars=1, temperature=0.0)
        self.assertEqual(r.text, ">")
        for to_end in (False, True):
            r = dijkstra_predict(g, START, 0, min_chars=3, max_chars=10, to_end=to_end)
            self.assertEqual(r.text, self.decode_by_id(g, r, True))
            self.assertIn(r.text, ("x<s>y", "a</s>b", "<s>", "<s>y"))
        for seed in range(5):
            r = sample_walk(g, START, 0, max_chars=10, rng=random.Random(seed))
            self.assertEqual(r.text, self.decode_by_id(g, r, True))
        # the END node itself is still skipped
        end_text = graph_from(["</s>"], compress=True)
        r = dijkstra_predict(end_text, START, 0, min_chars=0, max_chars=10, to_end=True)
        self.assertEqual(r.labels, [START_LABEL, END_LABEL, END_LABEL])
        self.assertEqual(r.text, "</s>")
        self.assertTrue(r.reached_end)


if __name__ == "__main__":
    unittest.main()
