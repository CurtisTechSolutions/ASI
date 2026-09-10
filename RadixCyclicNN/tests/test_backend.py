"""Tests for radixnet.backend (PythonBackend, factory, torch parity)."""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.backend import (  # noqa: E402
    CSR,
    MIN_B,
    Backend,
    NodeParams,
    PythonBackend,
    describe_backends,
    get_backend,
    torch_available,
)
from radixnet.encoding import Encoder  # noqa: E402
from radixnet.graph import RadixCyclicGraph  # noqa: E402

ENC = Encoder()


def build_graph(texts, seed=1, compress=True, perturb=None):
    """Graph + backend inputs for a set of texts."""
    g = RadixCyclicGraph(seed=seed)
    for t in texts:
        g.observe_sequence(ENC.encode(t))
    if compress:
        g.compress()
    if perturb is not None:
        rng = random.Random(perturb)
        for i in range(len(g.labels)):
            g.a[i] += rng.uniform(-0.3, 0.3)
            g.b[i] += rng.uniform(-0.1, 0.1)
            g.h[i] += rng.uniform(-1.0, 1.0)
            g.k[i] += rng.uniform(-0.3, 0.3)
        g.version += 1
    trans = []
    for t in texts:
        trans.extend(g.observe_sequence(ENC.encode(t), count=False))
    csr = g.to_csr()
    parents = [p for p, _ in trans]
    positions = [csr.edge_pos[e] for _, e in trans]
    return g, csr, parents, positions


class TestContainers(unittest.TestCase):
    def test_csr_validation(self):
        csr = CSR([0, 1, 1], [1], [0], [0.5], {0: 0})
        self.assertEqual((csr.n_nodes, csr.n_edges), (2, 1))
        self.assertEqual(list(csr.row(0)), [0])
        self.assertEqual(list(csr.row(1)), [])
        with self.assertRaises(ValueError):
            CSR([0, 2], [1], [0], [0.5], {})
        with self.assertRaises(ValueError):
            CSR([0, 1], [1, 2], [0], [0.5], {})

    def test_node_params(self):
        p = NodeParams([1.0], [2.0], [3.0], [4.0], [5.0])
        self.assertEqual(len(p), 1)
        q = p.copy()
        q.z[0] = 9.0
        self.assertEqual(p.z[0], 1.0)
        with self.assertRaises(ValueError):
            NodeParams([1.0], [2.0], [3.0], [4.0], [])
        with self.assertRaises(AttributeError):
            p.extra = 1  # type: ignore[attr-defined]


class TestPythonBackend(unittest.TestCase):
    def test_is_backend(self):
        be = PythonBackend()
        self.assertIsInstance(be, Backend)
        self.assertEqual((be.name, be.device), ("python", "cpu"))
        with self.assertRaises(ValueError):
            PythonBackend(device="cuda")

    def test_prepare_finalize_round_trip(self):
        g, csr, parents, positions = build_graph(["abcabc", "abcd"])
        be = PythonBackend()
        state = be.prepare(csr, g.node_params())
        w, params = be.finalize(state)
        self.assertEqual(w, csr.weights)
        self.assertEqual(params.z, g.z)
        self.assertIsNot(w, state.weights)
        acts = be.node_activations(state)
        for i in range(len(g.labels)):
            self.assertAlmostEqual(acts[i], g.activation_of(i), places=15)
        with self.assertRaises(ValueError):
            be.prepare(csr, NodeParams([0.0], [0.0], [0.0], [0.0], [0.0]))

    def test_loss_decreases_on_tiny_graph(self):
        texts = ["the quick brown fox"] * 4 + ["the quick brown dog", "the slow brown fox"]
        g, csr, parents, positions = build_graph(texts)
        be = PythonBackend()
        state = be.prepare(csr, g.node_params())
        first = be.loss(state, parents, positions)
        self.assertGreater(first, 0.0)
        # entropy floor: the best any model can do with a per-parent softmax
        from collections import Counter
        per_parent = {}
        for p, pos in zip(parents, positions):
            per_parent.setdefault(p, Counter())[pos] += 1
        floor = 0.0
        for p, pos in zip(parents, positions):
            cnt = per_parent[p]
            floor -= math.log(cnt[pos] / sum(cnt.values()))
        floor /= len(parents)
        self.assertGreater(first, floor)
        losses = []
        for _ in range(80):
            losses.append(be.step(state, parents, positions, lr=0.1, act_lr=0.01))
        self.assertAlmostEqual(losses[0], first, places=12)
        final = be.loss(state, parents, positions)
        self.assertLess(final, losses[-1])
        self.assertLess(final, first)
        # at least half of the reducible loss is gone
        self.assertLess(final - floor, 0.5 * (first - floor))
        # writing back to the graph keeps it consistent and lowers its costs too
        w, params = be.finalize(state)
        g.apply_csr_weights(csr, w)
        g.apply_node_params(params)
        g.check_invariants(texts=texts)
        total = 0.0
        for p, pos in zip(parents, positions):
            e = csr.edge_ids[pos]
            total += next(c for _, ee, c in g.child_costs(p) if ee == e)
        self.assertAlmostEqual(total / len(parents), be.loss(state, parents, positions), places=9)

    def test_step_returns_loss_before_update_and_empty_batch(self):
        g, csr, parents, positions = build_graph(["abcabc", "abcd", "aaaa"], perturb=3)
        be = PythonBackend()
        state = be.prepare(csr, g.node_params())
        before = be.loss(state, parents, positions)
        got = be.step(state, parents, positions, lr=0.5, act_lr=0.05)
        self.assertAlmostEqual(got, before, places=12)
        self.assertNotAlmostEqual(be.loss(state, parents, positions), before, places=6)
        w0 = list(state.weights)
        self.assertEqual(be.step(state, [], [], lr=0.5, act_lr=0.05), 0.0)
        self.assertEqual(state.weights, w0)
        with self.assertRaises(ValueError):
            be.step(state, parents, positions[:-1], lr=0.1, act_lr=0.01)
        with self.assertRaises(ValueError):
            be.step(state, [parents[0]], [csr.n_edges - 1 if csr.indptr[parents[0] + 1] < csr.n_edges else 0], lr=0.1, act_lr=0.01)

    def test_clipping_and_b_guard(self):
        g, csr, parents, positions = build_graph(["abcabc", "abcd", "aaaa"], perturb=4)
        be = PythonBackend()
        state = be.prepare(csr, g.node_params())
        for j in range(csr.n_edges):
            state.weights[j] *= 40.0  # huge scores -> huge raw gradients
        grads = be.gradients(state, parents, positions)
        self.assertGreater(max(abs(v) for v in grads["w"].values()), 0.01)
        w0, z0 = list(state.weights), list(state.z)
        clip = 0.01
        be.step(state, parents, positions, lr=1.0, act_lr=1.0, clip=clip)
        for j in range(csr.n_edges):
            self.assertLessEqual(abs(state.weights[j] - w0[j]), clip + 1e-12)
        for i in range(len(z0)):
            self.assertLessEqual(abs(state.z[i] - z0[i]), clip + 1e-12)
        # b never drops below MIN_B
        for i in range(len(state.b)):
            state.b[i] = MIN_B + 1e-4
        be.step(state, parents, positions, lr=0.0, act_lr=1e6)
        self.assertTrue(all(b >= MIN_B for b in state.b))

    def test_gradients_are_zero_for_single_child_rows(self):
        g, csr, parents, positions = build_graph(["hello world"])
        be = PythonBackend()
        state = be.prepare(csr, g.node_params())
        grads = be.gradients(state, parents, positions)
        self.assertEqual(grads["loss"], 0.0)
        self.assertTrue(all(v == 0.0 for v in grads["w"].values()))
        self.assertEqual(be.step(state, parents, positions, 0.1, 0.1), 0.0)
        self.assertEqual(state.weights, csr.weights)


class TestFiniteDifferences(unittest.TestCase):
    """Analytic gradients of PythonBackend vs central differences (tolerance 1e-5)."""

    EPS = 1e-5
    TOL = 1e-5

    def _check(self, texts, seed, perturb):
        g, csr, parents, positions = build_graph(texts, seed=seed, perturb=perturb)
        be = PythonBackend()
        state = be.prepare(csr, g.node_params())
        grads = be.gradients(state, parents, positions)
        base = be.loss(state, parents, positions)
        self.assertAlmostEqual(grads["loss"], base, places=12)
        checked = 0

        def fd(values, idx):
            old = values[idx]
            values[idx] = old + self.EPS
            plus = be.loss(state, parents, positions)
            values[idx] = old - self.EPS
            minus = be.loss(state, parents, positions)
            values[idx] = old
            return (plus - minus) / (2 * self.EPS)

        for j in range(csr.n_edges):
            numeric = fd(state.weights, j)
            self.assertAlmostEqual(grads["w"].get(j, 0.0), numeric, delta=self.TOL, msg=f"w[{j}]")
            checked += 1
        for name, values in (("z", state.z), ("a", state.a), ("b", state.b), ("h", state.h), ("k", state.k)):
            for i in range(len(values)):
                numeric = fd(values, i)
                self.assertAlmostEqual(grads[name].get(i, 0.0), numeric, delta=self.TOL, msg=f"{name}[{i}]")
                checked += 1
        # the state must be untouched by loss / gradients
        self.assertEqual(state.weights, csr.weights)
        self.assertTrue(all(v == 0.0 for v in state.gw))
        self.assertTrue(all(v == 0.0 for v in state.gz))
        return checked

    def test_gradient_check_branching_graph(self):
        n = self._check(["the cat sat", "the dog sat", "the cat ran", "a cat"], seed=5, perturb=1)
        self.assertGreater(n, 50)

    def test_gradient_check_cycles_self_loops_and_repeated_parents(self):
        # "aaaa" -> self loop (node is parent and child of the same edge);
        # "abcabc" -> cycle; a parent appears several times in the batch.
        n = self._check(["aaaa", "abcabc", "abab", "abcd", "aaab"], seed=6, perturb=2)
        self.assertGreater(n, 30)

    def test_gradient_check_default_parameters(self):
        self._check(["abcabc", "abcd", "abxy", "aaaa"], seed=7, perturb=None)


class TestFactory(unittest.TestCase):
    def test_python(self):
        self.assertIsInstance(get_backend("python"), PythonBackend)
        self.assertIsInstance(get_backend("PYTHON"), PythonBackend)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            get_backend("numpy")

    def test_auto_returns_backend(self):
        be = get_backend("auto")
        self.assertIsInstance(be, Backend)
        self.assertIn(be.name, ("python", "torch"))
        info = describe_backends()
        if info["default"] == "python":
            self.assertIsInstance(be, PythonBackend)

    def test_describe(self):
        info = describe_backends()
        self.assertEqual(set(info), {"python", "torch", "cuda", "mps", "default"})
        self.assertTrue(info["python"])
        self.assertEqual(info["torch"], torch_available())
        self.assertIsInstance(info["cuda"], bool)
        self.assertIsInstance(info["mps"], bool)
        self.assertIn(info["default"], ("python", "torch"))
        if not info["torch"]:
            self.assertFalse(info["cuda"] or info["mps"])
            self.assertEqual(info["default"], "python")

    def test_torch_missing_gives_clear_error(self):
        try:
            be = get_backend("torch")
        except ImportError as exc:
            self.assertIn("torch backend unavailable", str(exc))
        else:
            self.assertEqual(be.name, "torch")
            self.assertIn(be.device, ("cpu", "cuda", "mps"))


class TestTorchParity(unittest.TestCase):
    def test_torch_matches_python_after_one_step(self):
        if not torch_available():
            self.skipTest("torch is not importable")
        try:
            torch_be = get_backend("torch")
        except ImportError as exc:
            self.skipTest(str(exc))
        texts = ["the cat sat on the mat", "the dog sat", "aaaa", "abcabc", "the cat ran"]
        g, csr, parents, positions = build_graph(texts, seed=9, perturb=5)
        py = PythonBackend()
        s_py = py.prepare(csr, g.node_params())
        s_t = torch_be.prepare(csr, g.node_params())
        for step in range(3):
            l_py = py.step(s_py, parents, positions, lr=0.05, act_lr=0.005)
            l_t = torch_be.step(s_t, parents, positions, lr=0.05, act_lr=0.005)
            self.assertAlmostEqual(l_py, l_t, delta=1e-6, msg=f"loss differs at step {step}")
        w_py, p_py = py.finalize(s_py)
        w_t, p_t = torch_be.finalize(s_t)
        for x, y in zip(w_py, w_t):
            self.assertAlmostEqual(x, y, delta=1e-6)
        for name in ("z", "a", "b", "h", "k"):
            for x, y in zip(getattr(p_py, name), getattr(p_t, name)):
                self.assertAlmostEqual(x, y, delta=1e-6, msg=name)
        for x, y in zip(py.node_activations(s_py), torch_be.node_activations(s_t)):
            self.assertAlmostEqual(x, y, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
