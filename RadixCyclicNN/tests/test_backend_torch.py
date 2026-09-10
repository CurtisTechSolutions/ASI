"""Tests for radixnet.backend_torch (the whole module is skipped when torch is not importable)."""

import math
import os
import random
import sys
import unittest
import warnings
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy", category=UserWarning)
        import torch
except Exception:  # noqa: BLE001 - missing or broken torch means "skip"
    torch = None
HAVE_TORCH = torch is not None

from radixnet.backend import (  # noqa: E402
    MIN_B,
    Backend,
    NodeParams,
    PythonBackend,
    describe_backends,
    get_backend,
)
from radixnet.encoding import Encoder  # noqa: E402
from radixnet.graph import END, START, RadixCyclicGraph  # noqa: E402
from radixnet.model import RadixNet  # noqa: E402

if HAVE_TORCH:
    from radixnet.backend_torch import TorchBackend, TorchState, resolve_dtype, select_device

ENC = Encoder()
CYCLIC_TEXTS = ["the cat sat on the mat", "the dog sat", "aaaa", "abcabc", "the cat ran", "abab"]
PARAM_NAMES = ("z", "a", "b", "h", "k")


def build_graph(texts, seed=1, compress=True, perturb=None):
    """Graph + backend inputs for a set of texts (same recipe as test_backend)."""
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


def entropy_floor(parents, positions):
    """Best achievable mean loss with one softmax per parent."""
    per_parent = {}
    for p, pos in zip(parents, positions):
        per_parent.setdefault(p, Counter())[pos] += 1
    floor = 0.0
    for p, pos in zip(parents, positions):
        cnt = per_parent[p]
        floor -= math.log(cnt[pos] / sum(cnt.values()))
    return floor / len(parents)


@unittest.skipUnless(HAVE_TORCH, "torch is not importable")
class TestDeviceAndDtype(unittest.TestCase):
    def test_auto_device_follows_cuda_mps_cpu_rule(self):
        be = TorchBackend()
        self.assertIsInstance(be, Backend)
        self.assertEqual(be.name, "torch")
        if torch.cuda.is_available():
            expected = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            expected = "mps"
        else:
            expected = "cpu"
        self.assertEqual(be.device, expected)
        self.assertEqual(TorchBackend(device="auto").device, expected)
        self.assertEqual(select_device(torch, None), expected)
        # the backend accepts its own device string back (Evolver does this)
        self.assertEqual(TorchBackend(device=be.device).device, be.device)

    def test_explicit_cpu_and_dtype(self):
        be = TorchBackend(device="cpu")
        self.assertEqual(be.device, "cpu")
        self.assertIs(be.dtype, torch.float64)
        self.assertEqual(TorchBackend(device=" CPU ").device, "cpu")
        self.assertIs(TorchBackend(device="cpu", dtype="float32").dtype, torch.float32)
        self.assertIs(TorchBackend(device="cpu", dtype=torch.float32).dtype, torch.float32)
        self.assertIs(TorchBackend(device="cpu", dtype="torch.float64").dtype, torch.float64)
        self.assertIs(resolve_dtype(torch, "double", "cpu"), torch.float64)
        self.assertIs(resolve_dtype(torch, None, "mps"), torch.float32)
        with self.assertRaises(ValueError):
            resolve_dtype(torch, "float64", "mps")
        for bad in ("float16", "int64", torch.int64, "bfloat16"):
            with self.assertRaises(ValueError):
                TorchBackend(device="cpu", dtype=bad)
        self.assertIn("cpu", repr(be))
        self.assertIn("float64", repr(be))

    def test_unknown_device_names(self):
        for bad in ("tpu", "cpu:1", "cuda:x", "mps:0", "gpu"):
            with self.assertRaises(ValueError, msg=bad):
                TorchBackend(device=bad)

    def test_unavailable_device_is_a_clear_error(self):
        checked = 0
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError) as ctx:
                TorchBackend(device="cuda")
            self.assertIn("CUDA", str(ctx.exception))
            checked += 1
        else:
            with self.assertRaises(RuntimeError):
                TorchBackend(device=f"cuda:{torch.cuda.device_count()}")
            checked += 1
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            with self.assertRaises(RuntimeError) as ctx:
                TorchBackend(device="mps")
            self.assertIn("MPS", str(ctx.exception))
            checked += 1
        self.assertGreaterEqual(checked, 1)

    def test_factory_wiring_and_auto_selection(self):
        be = get_backend("torch", device="cpu")
        self.assertIsInstance(be, TorchBackend)
        self.assertEqual((be.name, be.device), ("torch", "cpu"))
        self.assertIsInstance(get_backend("TORCH"), TorchBackend)
        info = describe_backends()
        self.assertTrue(info["torch"])
        accelerated = info["cuda"] or info["mps"]
        self.assertEqual(info["default"], "torch" if accelerated else "python")
        auto = get_backend("auto")
        self.assertEqual(auto.name, info["default"])
        if not accelerated:
            self.assertIsInstance(auto, PythonBackend)
            self.assertIsInstance(get_backend("auto", device="cpu"), PythonBackend)


@unittest.skipUnless(HAVE_TORCH, "torch is not importable")
class TestParity(unittest.TestCase):
    """TorchBackend must match PythonBackend (float64, cpu)."""

    def _pair(self, texts=CYCLIC_TEXTS, seed=9, perturb=5, dtype=None):
        g, csr, parents, positions = build_graph(texts, seed=seed, perturb=perturb)
        py = PythonBackend()
        tb = TorchBackend(device="cpu", dtype=dtype)
        return g, csr, parents, positions, py, py.prepare(csr, g.node_params()), tb, tb.prepare(csr, g.node_params())

    def _assert_states_match(self, py, s_py, tb, s_t, delta):
        w_py, p_py = py.finalize(s_py)
        w_t, p_t = tb.finalize(s_t)
        self.assertEqual(len(w_py), len(w_t))
        for x, y in zip(w_py, w_t):
            self.assertAlmostEqual(x, y, delta=delta)
        for name in PARAM_NAMES:
            a_py, a_t = getattr(p_py, name), getattr(p_t, name)
            self.assertEqual(len(a_py), len(a_t))
            for x, y in zip(a_py, a_t):
                self.assertAlmostEqual(x, y, delta=delta, msg=name)
        for x, y in zip(py.node_activations(s_py), tb.node_activations(s_t)):
            self.assertAlmostEqual(x, y, delta=delta)

    def test_prepare_uploads_and_finalize_downloads_plain_lists(self):
        g, csr, parents, positions, py, s_py, tb, s_t = self._pair()
        self.assertIsInstance(s_t, TorchState)
        self.assertEqual((s_t.n_nodes, s_t.n_edges), (csr.n_nodes, csr.n_edges))
        self.assertIs(s_t.weights.dtype, torch.float64)
        w, params = tb.finalize(s_t)
        self.assertIsInstance(w, list)
        self.assertIsInstance(params, NodeParams)
        self.assertTrue(all(type(x) is float for x in w))
        self.assertTrue(all(type(x) is float for x in params.b))
        self.assertEqual(w, csr.weights)
        for name in PARAM_NAMES:
            self.assertEqual(getattr(params, name), getattr(g, name))
        acts = tb.node_activations(s_t)
        self.assertEqual(len(acts), len(g.labels))
        for i in range(len(g.labels)):
            self.assertAlmostEqual(acts[i], g.activation_of(i), places=12)
        with self.assertRaises(ValueError):
            tb.prepare(csr, NodeParams([0.0], [0.0], [0.0], [0.0], [0.0]))

    def test_gradients_match_python(self):
        g, csr, parents, positions, py, s_py, tb, s_t = self._pair()
        gp = py.gradients(s_py, parents, positions)
        gt = tb.gradients(s_t, parents, positions)
        self.assertGreater(gp["loss"], 0.0)
        self.assertAlmostEqual(gp["loss"], gt["loss"], delta=1e-10)
        for name in ("w",) + PARAM_NAMES:
            self.assertEqual(set(gp[name]), set(gt[name]), name)
            for key, value in gp[name].items():
                self.assertAlmostEqual(value, gt[name][key], delta=1e-10, msg=f"{name}[{key}]")
        # the helpers must not modify the state
        self.assertEqual(tb.finalize(s_t)[0], csr.weights)
        self.assertAlmostEqual(tb.loss(s_t, parents, positions), gp["loss"], delta=1e-10)

    def test_one_step_parity(self):
        g, csr, parents, positions, py, s_py, tb, s_t = self._pair()
        l_py = py.step(s_py, parents, positions, lr=0.05, act_lr=0.005)
        l_t = tb.step(s_t, parents, positions, lr=0.05, act_lr=0.005)
        self.assertAlmostEqual(l_py, l_t, delta=1e-9)
        self._assert_states_match(py, s_py, tb, s_t, delta=1e-9)
        # the update actually changed something
        self.assertNotEqual(tb.finalize(s_t)[0], csr.weights)

    def test_many_steps_parity_with_repeated_parents_and_self_loops(self):
        # a parent appears several times per batch; "aaaa" is a self-loop
        g, csr, parents, positions, py, s_py, tb, s_t = self._pair(seed=11, perturb=7)
        batch_p, batch_q = parents * 2, positions * 2
        for step in range(40):
            l_py = py.step(s_py, batch_p, batch_q, lr=0.1, act_lr=0.01, clip=0.5)
            l_t = tb.step(s_t, batch_p, batch_q, lr=0.1, act_lr=0.01, clip=0.5)
            self.assertAlmostEqual(l_py, l_t, delta=1e-6, msg=f"loss differs at step {step}")
        self._assert_states_match(py, s_py, tb, s_t, delta=1e-6)

    def test_mixed_batch_with_single_child_rows(self):
        # "hello world" contributes single-child rows (zero loss) that still count in the mean
        texts = CYCLIC_TEXTS + ["hello world"]
        g, csr, parents, positions, py, s_py, tb, s_t = self._pair(texts=texts, seed=4, perturb=None)
        singles = [i for i, p in enumerate(parents) if csr.indptr[p + 1] - csr.indptr[p] == 1]
        self.assertGreater(len(singles), 0)
        l_py = py.step(s_py, parents, positions, lr=0.2, act_lr=0.02)
        l_t = tb.step(s_t, parents, positions, lr=0.2, act_lr=0.02)
        self.assertAlmostEqual(l_py, l_t, delta=1e-9)
        self._assert_states_match(py, s_py, tb, s_t, delta=1e-9)

    def test_float32_is_close(self):
        g, csr, parents, positions, py, s_py, tb, s_t = self._pair(dtype="float32")
        self.assertIs(s_t.weights.dtype, torch.float32)
        l_py = py.step(s_py, parents, positions, lr=0.05, act_lr=0.005)
        l_t = tb.step(s_t, parents, positions, lr=0.05, act_lr=0.005)
        self.assertAlmostEqual(l_py, l_t, delta=1e-4)
        self._assert_states_match(py, s_py, tb, s_t, delta=1e-4)


@unittest.skipUnless(HAVE_TORCH, "torch is not importable")
class TestStepSemantics(unittest.TestCase):
    def setUp(self):
        self.g, self.csr, self.parents, self.positions = build_graph(CYCLIC_TEXTS, seed=3, perturb=2)
        self.tb = TorchBackend(device="cpu")
        self.state = self.tb.prepare(self.csr, self.g.node_params())

    def _snapshot(self):
        w, p = self.tb.finalize(self.state)
        return w, [getattr(p, name) for name in PARAM_NAMES]

    def test_empty_batch_is_a_noop(self):
        before = self._snapshot()
        self.assertEqual(self.tb.step(self.state, [], [], lr=0.5, act_lr=0.05), 0.0)
        self.assertEqual(self.tb.loss(self.state, [], []), 0.0)
        self.assertEqual(self.tb.gradients(self.state, [], [])["w"], {})
        self.assertEqual(self._snapshot(), before)

    def test_validation_errors_leave_state_untouched(self):
        before = self._snapshot()
        n_edges = self.csr.n_edges
        p0 = self.parents[0]
        foreign = next(j for j in range(n_edges) if not (self.csr.indptr[p0] <= j < self.csr.indptr[p0 + 1]))
        cases = [
            ([p0], [foreign]),          # a real position, but of another row
            ([p0], [n_edges]),          # beyond the CSR
            ([p0], [-1]),
            ([len(self.g.labels)], [0]),  # parent id out of range
            ([-1], [0]),
        ]
        for parents, positions in cases:
            with self.assertRaises(ValueError, msg=(parents, positions)):
                self.tb.step(self.state, parents, positions, lr=0.1, act_lr=0.01)
            with self.assertRaises(ValueError):
                self.tb.loss(self.state, parents, positions)
        with self.assertRaises(ValueError):
            self.tb.step(self.state, self.parents, self.positions[:-1], lr=0.1, act_lr=0.01)
        with self.assertRaises(ValueError):
            self.tb.step(self.state, self.parents, self.positions, lr=0.1, act_lr=0.01, clip=-1.0)
        self.assertEqual(self._snapshot(), before)

    def test_single_child_rows_cost_nothing(self):
        g, csr, parents, positions = build_graph(["hello world"])
        state = self.tb.prepare(csr, g.node_params())
        grads = self.tb.gradients(state, parents, positions)
        self.assertEqual(grads["loss"], 0.0)
        self.assertEqual(grads["w"], {})
        self.assertEqual(self.tb.step(state, parents, positions, 0.1, 0.1), 0.0)
        self.assertEqual(self.tb.finalize(state)[0], csr.weights)

    def test_step_returns_loss_before_update(self):
        before = self.tb.loss(self.state, self.parents, self.positions)
        got = self.tb.step(self.state, self.parents, self.positions, lr=0.5, act_lr=0.05)
        self.assertAlmostEqual(got, before, places=12)
        after = self.tb.loss(self.state, self.parents, self.positions)
        self.assertNotAlmostEqual(after, before, places=6)
        self.assertLess(after, before)

    def test_clipping_and_b_guard_touch_only_batch_entries(self):
        self.state.weights.mul_(40.0)  # huge scores -> huge raw gradients
        grads = self.tb.gradients(self.state, self.parents, self.positions)
        self.assertGreater(max(abs(v) for v in grads["w"].values()), 0.01)
        w0, (z0, _, _, _, _) = self._snapshot()
        clip = 0.01
        self.tb.step(self.state, self.parents, self.positions, lr=1.0, act_lr=1.0, clip=clip)
        w1, (z1, _, _, _, _) = self._snapshot()
        for x, y in zip(w0, w1):
            self.assertLessEqual(abs(x - y), clip + 1e-12)
        for x, y in zip(z0, z1):
            self.assertLessEqual(abs(x - y), clip + 1e-12)
        # b never drops below MIN_B for nodes in the batch ...
        self.state.node[2].fill_(MIN_B + 1e-4)
        self.tb.step(self.state, self.parents, self.positions, lr=0.0, act_lr=1e6)
        self.assertTrue(all(b >= MIN_B for b in self.tb.finalize(self.state)[1].b))
        # ... while nodes outside the batch are not touched at all (sparse update):
        # on a fresh (unsaturated) state, a batch of one START transition touches
        # START and its children only, never END
        state = self.tb.prepare(self.csr, self.g.node_params())
        self.assertEqual(self.parents[0], START)
        self.assertGreater(len(self.g.children[START]), 1)
        self.assertNotIn(END, self.g.children[START])
        state.node[2, END] = MIN_B / 2
        state.node[0, END] = 1.5
        self.tb.step(state, [self.parents[0]], [self.positions[0]], lr=0.5, act_lr=0.5)
        params = self.tb.finalize(state)[1]
        self.assertEqual(params.b[END], MIN_B / 2)
        self.assertEqual(params.z[END], 1.5)
        self.assertNotEqual(params.z[START], self.g.z[START])


@unittest.skipUnless(HAVE_TORCH, "torch is not importable")
class TestLearning(unittest.TestCase):
    def test_loss_decreases_over_many_steps(self):
        texts = ["the quick brown fox"] * 4 + ["the quick brown dog", "the slow brown fox"]
        g, csr, parents, positions = build_graph(texts)
        tb = TorchBackend(device="cpu")
        state = tb.prepare(csr, g.node_params())
        first = tb.loss(state, parents, positions)
        floor = entropy_floor(parents, positions)
        self.assertGreater(first, floor)
        losses = [tb.step(state, parents, positions, lr=0.1, act_lr=0.01) for _ in range(80)]
        self.assertAlmostEqual(losses[0], first, places=12)
        final = tb.loss(state, parents, positions)
        self.assertLess(final, losses[-1])
        self.assertLess(final, first)
        self.assertLess(final - floor, 0.5 * (first - floor))
        # the downloaded state is consistent with the graph's own cost function
        w, params = tb.finalize(state)
        g.apply_csr_weights(csr, w)
        g.apply_node_params(params)
        g.check_invariants(texts=texts)
        total = 0.0
        for p, pos in zip(parents, positions):
            e = csr.edge_ids[pos]
            total += next(c for _, ee, c in g.child_costs(p) if ee == e)
        self.assertAlmostEqual(total / len(parents), final, places=9)

    def test_radixnet_training_matches_python_backend(self):
        texts = CYCLIC_TEXTS + ["the quick brown fox", "the quick brown dog", "hello world"]
        py_model = RadixNet(seed=5, backend="python")
        t_model = RadixNet(seed=5, backend="torch", device="cpu")
        self.assertEqual((t_model.backend.name, t_model.backend.device), ("torch", "cpu"))
        r_py = py_model.train(texts, epochs=4, lr=0.5, act_lr=0.05, batch_size=8)
        r_t = t_model.train(texts, epochs=4, lr=0.5, act_lr=0.05, batch_size=8)
        self.assertEqual(len(r_py), 4)
        self.assertLess(r_t[-1]["loss"], r_t[0]["loss"])
        for a, b in zip(r_py, r_t):
            self.assertAlmostEqual(a["loss"], b["loss"], delta=1e-6)
            self.assertEqual((a["nodes"], a["edges"], a["merges"]), (b["nodes"], b["edges"], b["merges"]))
        for x, y in zip(py_model.graph.edge_w, t_model.graph.edge_w):
            self.assertAlmostEqual(x, y, delta=1e-6)
        self.assertEqual(py_model.predict("the quick", length=8).text, t_model.predict("the quick", length=8).text)
        stats = t_model.stats()
        self.assertEqual((stats["backend"], stats["device"]), ("torch", "cpu"))


if __name__ == "__main__":
    unittest.main()
