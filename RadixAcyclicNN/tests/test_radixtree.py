"""RadixAcyclicNN tests.  Run: python3 -m unittest tests.test_radixtree  (from RadixAcyclicNN/)

Plain ``unittest``, standard library, seconds, deterministic.  The structural
tests mirror ``RadixCyclicNN/tests/test_graph.py`` case for case, with the
tree's answers where the graph's were cycles: ``"aaaa"`` gives no self-loop
and does not represent ``"aaaaaaa"``, and every random sequence of operations
leaves a tree.
"""

import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "PhoneticTokenizer"))

from radixtree import corpus  # noqa: E402
from radixtree.activation import DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K, MIN_B, SineActivation, sine_partials  # noqa: E402
from radixtree.check import check_gradients, check_step  # noqa: E402
from radixtree.cli import main as cli_main  # noqa: E402
from radixtree.compare import compare, load_radixnet, markdown  # noqa: E402
from radixtree.encoding import PHONES, START_LABEL, SYLLABLES, Encoding  # noqa: E402
from radixtree.model import MAX_LEGS, MAX_SLIDE, UNKNOWN_PROB, RadixTreeNet, TrainConfig, load_model  # noqa: E402
from radixtree.search import cheapest_path, sample_walk  # noqa: E402
from radixtree.tree import FIRST, ROOT, START, RadixTree  # noqa: E402

try:
    import phonetok  # noqa: E402,F401
except ImportError:  # pragma: no cover
    phonetok = None

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
ENC = Encoding()


def read_lines(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


CORPUS = read_lines("sample_corpus.txt")
GARBAGE = read_lines("sample_garbage.txt")

# The rule averages gradients over the batch, so the tests train at a rate that learns visibly
# in a few epochs; the config defaults (lr 0.05, batch 256) are the cyclic model's and need many more.
FAST = {"lr": 0.5, "act_lr": 0.05, "batch_size": 32}


def round_trip(tree: RadixTree, text: str) -> str:
    """Walk ``text`` from START and decode the real labels on its path."""
    path = tree.node_path(ENC.encode(text))
    assert path is not None, f"{text!r} is not a root path"
    assert path[0] == START and tree.is_end(path[-1])
    return tree.encoding.decode_path([tree.labels[i] for i in path if tree.is_real(i)], 0, True)


def assert_transitions_consistent(tc: unittest.TestCase, tree: RadixTree, trans):
    """Every (parent, child) must be an alive option of that parent."""
    for p, c in trans:
        tc.assertTrue(tree.alive[p] and tree.alive[c])
        tc.assertIn(c, tree.child_items(p), f"node {c} is not an option at {p}")
        tc.assertEqual(tree.parent[c], p)


def corpus_model(seed=0, epochs=6, **kw):
    model = RadixTreeNet(seed=seed, **kw)
    model.train(CORPUS, epochs=epochs, **FAST)
    return model


def random_texts(rng, alphabet, count, lo=3, hi=14):
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi))) for _ in range(count)]


# ---------------------------------------------------------------------------
# encoding and activation
# ---------------------------------------------------------------------------


class TestEncoding(unittest.TestCase):
    def test_trigrams_round_trip(self):
        self.assertEqual(ENC.encode("hello"), ["hel", "ell", "llo"])
        self.assertEqual(ENC.decode_grams(["hel", "ell", "llo"]), "hello")
        self.assertEqual(ENC.encode("hi"), [])
        self.assertEqual(ENC.decode_grams([]), "")
        self.assertEqual(ENC.grams_held("hello"), 3)
        self.assertEqual(ENC.gram_at("hello", 1), "ell")

    def test_decode_path_context_and_remainder(self):
        self.assertEqual(ENC.decode_path(["hello", "lo w"], 0, True), "hello w")
        self.assertEqual(ENC.decode_path(["hello", "lo w"], 1, False), "o w")
        self.assertEqual(ENC.decode_path(["hello", "lo w"], 2, False), " w")
        with self.assertRaises(ValueError):
            ENC.decode_path(["hello"], -1)

    def test_check_grams(self):
        ENC.check_grams(["abc", "bcd"])
        with self.assertRaises(ValueError):
            ENC.check_grams(["abc", "xyz"])
        with self.assertRaises(ValueError):
            ENC.check_grams(["abcd"])

    def test_other_windows(self):
        one = Encoding(n=1)
        self.assertEqual(one.encode("abc"), ["a", "b", "c"])
        self.assertEqual(one.decode_grams(["a", "b", "c"]), "abc")
        five = Encoding(n=5)
        self.assertEqual(five.encode("abcdef"), ["abcde", "bcdef"])
        self.assertEqual(five.decode_path(["abcdef"], 0, False), "f")
        with self.assertRaises(ValueError):
            Encoding(n=0).validate()
        self.assertEqual(Encoding.from_dict(five.to_dict()), five)
        self.assertEqual(Encoding.from_dict(None), ENC)
        with self.assertRaises(ValueError):
            Encoding.from_dict({"unit": "word", "n": 2}).validate()
        with self.assertRaises(ValueError):
            Encoding.from_dict({"unit": "char", "n": 3, "stride": 3})


class TestActivation(unittest.TestCase):
    def test_default_is_the_authors_sine(self):
        f = SineActivation()
        for x in (-3.0, -0.5, 0.0, 1.25, 4.0):
            self.assertAlmostEqual(f(x), -math.sin(x / 3.0), places=15)

    def test_partials_match_finite_differences(self):
        a, b, h, k, x = 0.7, 0.9, -0.3, 0.2, 1.1
        f, dx, da, db, dh, dk = sine_partials(x, a, b, h, k)
        eps = 1e-6

        def at(**kw):
            p = {"x": x, "a": a, "b": b, "h": h, "k": k}
            p.update(kw)
            return SineActivation(p["a"], p["b"], p["h"], p["k"])(p["x"])

        for name, val, want in (("x", x, dx), ("a", a, da), ("b", b, db), ("h", h, dh), ("k", k, dk)):
            num = (at(**{name: val + eps}) - at(**{name: val - eps})) / (2 * eps)
            self.assertAlmostEqual(num, want, places=7, msg=name)

    def test_inverted_negates_exactly(self):
        f = SineActivation(-1.0, 0.4, 0.3, 0.25)
        g = f.inverted()
        for x in (-2.0, 0.0, 1.5):
            self.assertAlmostEqual(g(x), -f(x), places=15)
        self.assertEqual(g.inverted(), f)


# ---------------------------------------------------------------------------
# the structure
# ---------------------------------------------------------------------------


class TestTreeBasics(unittest.TestCase):
    def test_fresh_tree(self):
        t = RadixTree(seed=3)
        self.assertEqual(t.labels[:FIRST], ["", START_LABEL])
        self.assertEqual((t.num_nodes(), t.num_ends(), t.num_edges(), t.num_grams()), (0, 0, 0, 0))
        self.assertEqual(t.compression_ratio(), 0.0)
        self.assertEqual(t.alive_nodes(), [ROOT, START])
        self.assertFalse(t.inverted)
        self.assertEqual(t.child_items(ROOT), [])  # START is never an option
        t.check_invariants()
        with self.assertRaises(ValueError):
            RadixTree(depth=1)

    def test_deterministic_seed(self):
        a, b = RadixTree(seed=11), RadixTree(seed=11)
        for t in (a, b):
            t.observe_sequence(ENC.encode("hello world"))
        self.assertEqual(a.z, b.z)
        self.assertEqual(a.w, b.w)
        c = RadixTree(seed=12)
        c.observe_sequence(ENC.encode("hello world"))
        self.assertNotEqual(a.z, c.z)

    def test_new_node_and_edge_ranges(self):
        t = RadixTree(seed=5)
        t.observe_sequence(ENC.encode("abcdefgh"))
        for i in t.alive_nodes():
            self.assertTrue(-4.5 <= t.z[i] <= 4.5)
            self.assertEqual((t.a[i], t.b[i], t.h[i], t.k[i]), (DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K))
            if i != ROOT:
                self.assertTrue(0.5 <= t.w[i] <= 1.5)
        self.assertEqual(t.w[ROOT], 0.0)

    def test_observe_single_text(self):
        t = RadixTree(seed=2)
        text = "the cat"
        grams = ENC.encode(text)
        trans = t.observe_sequence(grams)
        assert_transitions_consistent(self, t, trans)
        # the whole text from START, and every suffix from the root - six windows, one leaf each
        self.assertEqual(t.count[START], 1)
        self.assertEqual(t.count[ROOT], len(grams))
        self.assertEqual(t.num_grams(), 5)
        self.assertEqual(t.num_ends(), 6)
        self.assertEqual(t.num_edges(), t.num_nodes() + t.num_ends())
        self.assertEqual(t.node_path(grams), [START, trans[0][1], trans[1][1]])
        self.assertEqual(round_trip(t, text), text)
        t.check_invariants(texts=[text])
        self.assertEqual(t.compress(), 0)  # every suffix: always compressed

    def test_observe_count_false(self):
        t = RadixTree(seed=2)
        t.observe_sequence(ENC.encode("the cat"))
        counts = list(t.count)
        sv = t.structure_version
        trans = t.observe_sequence(ENC.encode("the cat"), count=False)
        self.assertEqual(t.count, counts)
        self.assertEqual(t.structure_version, sv)
        trans2 = t.observe_sequence(ENC.encode("the cat"))
        self.assertEqual(trans, trans2)
        self.assertEqual(t.count[START], 2)

    def test_bad_grams_rejected(self):
        t = RadixTree()
        with self.assertRaises(ValueError):
            t.observe_sequence(["abc", "xyz"])
        with self.assertRaises(ValueError):
            t.observe_sequence(["abcd"])
        self.assertEqual(t.observe_sequence([]), [])


class TestNoCycles(unittest.TestCase):
    """The mirror of the cyclic graph's TestCyclesAndRepeats: the same texts, and no loop anywhere."""

    def test_aaaa_makes_no_self_loop(self):
        t = RadixTree(seed=4)
        grams = ENC.encode("aaaa")
        t.observe_sequence(grams)
        for i in t.alive_nodes():
            self.assertNotIn(i, t.child_items(i))
        # three contexts hold "aaa": at the beginning (as "aaaa"), anywhere, and anywhere after "aaa"
        self.assertEqual(t.num_nodes(), 3)
        self.assertEqual(sorted(t.labels[i] for i in t.real_nodes()), ["aaa", "aaa", "aaaa"])
        self.assertEqual(t.compress(), 0)
        t.check_invariants(texts=["aaaa"], compressed=True)
        self.assertEqual(round_trip(t, "aaaa"), "aaaa")
        # the loop the graph would have made is exactly what the tree cannot say
        self.assertIsNone(t.node_path(ENC.encode("aaaaaaa")))

    def test_abcabc_is_a_tree_not_a_cycle(self):
        t = RadixTree(seed=4)
        t.observe_sequence(ENC.encode("abcabc"))
        t.check_invariants(texts=["abcabc"], compressed=True)
        self.assertEqual(t.num_nodes(), 5)  # the graph holds it in two nodes and a 2-cycle
        self.assertEqual(round_trip(t, "abcabc"), "abcabc")
        self.assertIsNone(t.node_path(ENC.encode("abcabcabc")))

    def test_repetition_is_unrolled(self):
        """The counting argument of the paper's section 3, measured: n a's cost n - 1 context nodes."""
        for n in (4, 10, 50):
            t = RadixTree(seed=1)
            t.observe_sequence(ENC.encode("a" * n))
            self.assertEqual(t.num_nodes(), n - 1)
            self.assertEqual(t.max_depth(), n - 1)

    def test_every_edge_goes_one_level_deeper(self):
        rng = random.Random(7)
        t = RadixTree(seed=7)
        texts = random_texts(rng, "ab c", 40)
        for text in texts:
            t.observe_sequence(ENC.encode(text))
        for i in t.alive_nodes():
            if i == ROOT:
                continue
            self.assertEqual(t.depth_of(i), t.depth_of(t.parent[i]) + 1)
            for c in t.child_items(i):
                self.assertNotEqual(c, i)
        t.check_invariants(texts=texts)

    def test_invariants_catch_a_cycle(self):
        t = RadixTree(seed=1)
        t.observe_sequence(ENC.encode("abcdef"))
        t.check_invariants()
        node = t.real_nodes()[0]
        keep = t.parent[node]
        t.parent[node] = node  # its own parent
        with self.assertRaises(AssertionError):
            t.check_invariants()
        t.parent[node] = keep
        t.check_invariants()


class TestSuffixProperty(unittest.TestCase):
    def test_every_substring_is_a_root_path(self):
        t = RadixTree(seed=3)
        texts = ["the cat sat on the mat", "the dog sat on the log", "a cat and a dog"]
        for text in texts:
            t.observe_sequence(ENC.encode(text))
        for text in texts:
            grams = ENC.encode(text)
            for i in range(len(grams)):
                for j in range(i + 1, len(grams) + 1):
                    self.assertIsNotNone(t.walk(ROOT, grams[i:j]), (text, i, j))
        t.check_invariants(texts=texts)

    def test_counts_are_conserved(self):
        t = RadixTree(seed=3)
        for text in CORPUS[:20]:
            t.observe_sequence(ENC.encode(text))
        for i in t.alive_nodes():
            if not t.is_end(i):
                self.assertEqual(t.count[i], sum(t.count[c] for c in t.child_items(i)), i)

    def test_locate_prefers_the_whole_history(self):
        t = RadixTree(seed=3)
        for text in ("the cat sat on the mat", "the dog sat on the log"):
            t.observe_sequence(ENC.encode(text))
        grams = ENC.encode("the cat")
        node, offset, symbols = t.locate(grams)
        self.assertEqual(symbols, len(grams) + 1)  # START and every gram
        self.assertEqual(t.depth_of(node) >= 2, True)
        # junk in front: the longest suffix the tree knows, from the root
        node2, offset2, symbols2 = t.locate(ENC.encode("zzthe cat"))
        self.assertEqual(symbols2, len(grams))
        self.assertEqual(t.labels[node2][offset2 : offset2 + 3], "cat")
        self.assertIsNone(t.locate(ENC.encode("qqq")))
        self.assertEqual(t.locate([]), (START, -1, 1))

    def test_bounded_depth_limits_the_context(self):
        t = RadixTree(seed=3, depth=4)
        t.observe_sequence(ENC.encode("the cat sat on the mat"))
        self.assertLessEqual(t.max_depth(), 4)
        node, offset, symbols = t.locate(ENC.encode("the cat sat"))
        self.assertLessEqual(symbols, 3)  # for a gram, a context holds at most depth - 1 symbols: one must follow
        self.assertTrue(t.usable(node, offset))
        node, offset, symbols = t.locate(ENC.encode("the mat"), ending=True)  # for END: START and 3 grams, then END
        self.assertEqual(symbols, 4)
        self.assertGreaterEqual(t.end_leaf[node], 0)
        self.assertLessEqual(t.locate(ENC.encode("the mat"))[2], 3)  # the same context is never asked for a gram
        t.check_invariants(texts=["the cat sat on the mat"])


class TestSplitAndMerge(unittest.TestCase):
    def _chain(self, times=1):
        t = RadixTree(seed=8)
        for _ in range(times):
            t.observe_sequence(ENC.encode("abcdefg"))
        (node,) = t.children[START].values()
        self.assertEqual(t.labels[node], "abcdefg")
        return t, node

    def test_split_moves_children_state_and_count(self):
        t, node = self._chain(times=7)
        self.assertEqual(t.count[node], 7)
        leaf = t.end_leaf[node]
        version, sv = t.version, t.structure_version
        a, b = t.split(node, 2)
        self.assertEqual(a, node)
        self.assertEqual((t.labels[a], t.labels[b]), ("abcd", "cdefg"))
        self.assertEqual(t.children[a], {"cde": b})
        self.assertEqual(t.end_leaf[a], -1)
        self.assertEqual(t.end_leaf[b], leaf)
        self.assertEqual(t.parent[leaf], b)
        self.assertEqual(t.parent[b], a)
        self.assertEqual(t.count[b], 7)
        self.assertEqual((t.z[b], t.a[b], t.b[b], t.h[b], t.k[b]), (t.z[a], t.a[a], t.b[a], t.h[a], t.k[a]))
        self.assertGreater(t.version, version)
        self.assertGreater(t.structure_version, sv)
        self.assertEqual(t.walk(START, ENC.encode("abcdefg")), (b, 2))
        self.assertEqual(round_trip(t, "abcdefg"), "abcdefg")
        t.check_invariants()

    def test_split_bounds(self):
        t, node = self._chain()
        for bad in (0, 5, 6, -1):
            with self.assertRaises(ValueError):
                t.split(node, bad)
        with self.assertRaises(ValueError):
            t.split(START, 1)
        with self.assertRaises(ValueError):
            t.split(t.end_leaf[node], 1)
        with self.assertRaises(ValueError):
            t.split(999, 1)
        t.split(node, 4)  # the last legal index
        self.assertEqual(t.labels[node], "abcdef")

    def test_split_then_compress_restores(self):
        t, node = self._chain()
        before = t.to_dict()
        t.split(node, 3)
        t.split(node, 1)
        t.check_invariants()
        self.assertEqual(t.compress(), 2)
        t.check_invariants(compressed=True)
        after = t.to_dict()
        self.assertEqual(after["nodes"]["labels"], before["nodes"]["labels"])
        self.assertEqual(after["nodes"]["parent"], before["nodes"]["parent"])
        self.assertEqual(after["nodes"]["z"], before["nodes"]["z"])

    def test_merge_preserves_every_score(self):
        """The rescale rule: after a merge no remaining probability has moved, whichever activation was kept."""
        for keep_parent in (True, False):
            t = RadixTree(seed=8)
            t.observe_sequence(ENC.encode("the cat sat"))
            t.observe_sequence(ENC.encode("the cat ran"))
            grams = ENC.encode("the cat ")
            node, offset, _ = t.locate(grams)
            self.assertTrue(t.at_end(node, offset))
            self.assertEqual(t.num_children(node), 2)
            p = t.parent[node]
            t.split(node, 2)  # node keeps two grams, the deep half keeps the branch
            (deep,) = t.children[node].values()
            t.z[node], t.z[deep] = (2.0, 0.3) if keep_parent else (0.3, 2.0)
            t.version += 1
            want_deep = dict(t.child_probs(deep))
            want_in = dict(t.child_probs(p))[node]
            self.assertTrue(t.merge_child(node))
            t.check_invariants()
            got = dict(t.child_probs(node))
            for c, prob in want_deep.items():
                self.assertAlmostEqual(got[c], prob, places=12)
            self.assertAlmostEqual(dict(t.child_probs(p))[node], want_in, places=12)

    def test_merge_conditions(self):
        t = RadixTree(seed=1)
        t.observe_sequence(ENC.encode("abcd"))
        t.observe_sequence(ENC.encode("xbcd"))
        self.assertFalse(t.merge_child(ROOT))
        self.assertFalse(t.merge_child(START))
        for i in t.real_nodes():
            self.assertFalse(t.merge_child(i))  # an END leaf is never merged away, a branch never is
        self.assertEqual(t.compress(), 0)

    def test_observation_never_leaves_a_chain(self):
        """A tree built by observing alone is compressed already, at any depth: only split leaves chains."""
        for depth in (None, 3, 5):
            t = RadixTree(seed=1, depth=depth)
            texts = CORPUS[:10]
            for text in texts:
                t.observe_sequence(ENC.encode(text))
            t.check_invariants(texts=texts, compressed=True)
            self.assertEqual(t.compress(), 0)
            node = next(i for i in t.real_nodes() if t.held(i) >= 2)
            t.split(node, 1)
            with self.assertRaises(AssertionError):
                t.check_invariants(compressed=True)
            self.assertEqual(t.compress(), 1)
            t.check_invariants(texts=texts, compressed=True)


class TestRandomisedInvariants(unittest.TestCase):
    """Hundreds of random observe / split / compress operations, with the invariants after each."""

    def test_random_operations(self):
        alphabets = ["ab", "abc", "aab", "ab c", "abcd"]
        total_ops = 0
        for seed in range(6):
            rng = random.Random(1000 + seed)
            alphabet = alphabets[seed % len(alphabets)]
            depth = None if seed % 2 == 0 else rng.choice([3, 4, 6])
            t = RadixTree(seed=seed, depth=depth)
            observed = []
            for _ in range(90):
                op = rng.random()
                if op < 0.6:
                    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 14)))
                    trans = t.observe_sequence(ENC.encode(text), count=rng.random() < 0.8)
                    observed.append(text)
                    assert_transitions_consistent(self, t, trans)
                    if depth is None:
                        self.assertEqual(round_trip(t, text), text)
                elif op < 0.85:
                    candidates = [i for i in t.real_nodes() if t.held(i) >= 2]
                    if candidates:
                        node = rng.choice(candidates)
                        a, b = t.split(node, rng.randint(1, t.held(node) - 1))
                        self.assertEqual(a, node)
                        self.assertTrue(t.alive[b])
                else:
                    t.compress()
                    t.check_invariants(texts=observed, compressed=True)
                t.check_invariants(texts=observed)
                total_ops += 1
            t.compress()
            t.check_invariants(texts=observed, compressed=True)
            if depth is None:
                for text in observed:
                    self.assertEqual(round_trip(t, text), text)
            sv = t.structure_version
            for text in observed:
                trans = t.observe_sequence(ENC.encode(text), count=False)
                assert_transitions_consistent(self, t, trans)
            self.assertEqual(t.structure_version, sv)  # re-observing is a structural no-op
        self.assertGreaterEqual(total_ops, 500)


class TestScoresAndCosts(unittest.TestCase):
    def setUp(self):
        self.t = RadixTree(seed=21)
        for text in ("the cat sat", "the cat ran", "the dog sat"):
            self.t.observe_sequence(ENC.encode(text))
        node, offset, _ = self.t.locate(ENC.encode("the cat"))
        self.branch = node

    def test_probabilities_and_costs(self):
        t = self.t
        probs = t.child_probs(self.branch)
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(sum(p for _, p in probs), 1.0, places=12)
        for (c, p), (c2, cost) in zip(probs, t.child_costs(self.branch)):
            self.assertEqual(c, c2)
            self.assertAlmostEqual(cost, -math.log(p), places=12)
            self.assertAlmostEqual(t.log_prob(self.branch, c), math.log(p), places=12)
        self.assertIsNone(t.log_prob(self.branch, START))
        for (c, s), (c2, cost) in zip(t.child_scores(self.branch), t.child_costs(self.branch, "signal")):
            self.assertEqual(cost, -s)
        with self.assertRaises(ValueError):
            t.child_costs(self.branch, "reward")

    def test_cost_cache_follows_version(self):
        t = self.t
        before = t.child_costs(self.branch)
        t.z[self.branch] += 1.0
        self.assertEqual(t.child_costs(self.branch), before)  # stale until the version moves
        t.version += 1
        self.assertNotEqual(t.child_costs(self.branch), before)


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------


class TestLearning(unittest.TestCase):
    def test_one_hop_gradients_match_finite_differences(self):
        """Read out of the code that trains, at both depths, against central differences."""
        self.assertLess(check_step(), 1e-6)
        self.assertLess(check_step(depth=4), 1e-6)
        self.assertLess(check_gradients(), 1e-6)
        self.assertLess(check_gradients(depth=4), 1e-6)

    def test_loss_falls(self):
        model = corpus_model(epochs=6)
        losses = [r["loss"] for r in model.history]
        self.assertLess(losses[-1], losses[0])
        for r in model.history:
            self.assertEqual(r["merges"], 0)  # observation leaves nothing to merge
            self.assertGreater(r["transitions"], 0)
        model.tree.check_invariants(texts=CORPUS, compressed=True)

    def test_observe_returns_transitions_that_hold(self):
        """Later texts split nodes earlier transitions named; what comes back names the structure as it stands."""
        model = RadixTreeNet(seed=1)
        sv = model.tree.structure_version
        transitions, observed = model._observe(CORPUS, count=True)
        assert_transitions_consistent(self, model.tree, transitions)
        self.assertEqual(observed, model.tree.structure_version)
        again, _ = model._observe(CORPUS, count=False)
        self.assertEqual(again, transitions)
        self.assertEqual(model.tree.structure_version, observed)

    def test_transitions_survive_later_splits(self):
        """A later text splits nodes earlier transitions named; training re-derives them instead of crashing."""
        model = RadixTreeNet(seed=1)
        model.train(CORPUS[:30], epochs=2, **FAST)
        model.train(CORPUS[30:], epochs=2, **FAST)  # splits nodes the first half made
        model.tree.check_invariants(texts=CORPUS)
        self.assertEqual(model.stats()["epochs_total"], 4)

    def test_grouping_by_parent_is_exact(self):
        """The grouped rule is the per-transition rule, summed: same loss and gradients, to the bit."""
        model = RadixTreeNet(seed=2)
        transitions, _ = model._observe(CORPUS[:12], count=True)
        batch = [t for t in transitions if model.tree.num_children(t[0]) > 1][:40]
        grouped = model.gradients_of(batch)
        # the reference: one transition at a time, gradients summed and averaged
        sums = {name: {} for name in ("w", "z", "a", "b", "h", "k")}
        loss = 0.0
        for item in batch:
            one = model.gradients_of([item])
            loss += one["loss"]
            for name in sums:
                for i, g in one[name].items():
                    sums[name][i] = sums[name].get(i, 0.0) + g
        self.assertAlmostEqual(grouped["loss"], loss / len(batch), places=10)
        for name in sums:
            for i, g in sums[name].items():
                self.assertAlmostEqual(grouped[name].get(i, 0.0), g / len(batch), places=9, msg=(name, i))

    def test_b_is_floored(self):
        model = RadixTreeNet(seed=1)
        transitions, _ = model._observe(CORPUS[:5], count=True)
        model.step(transitions, lr=0.0, act_lr=1e6, clip=1e6)
        self.assertTrue(all(b >= MIN_B for b in model.tree.b))

    def test_config_validation_and_overrides(self):
        for bad in ({"epochs": -1}, {"batch_size": 0}, {"lr": -0.1}, {"lr": float("nan")}, {"act_lr": float("inf")}, {"clip": 0.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TrainConfig(**bad).validate()
        model = RadixTreeNet()
        with self.assertRaises(TypeError):
            model.train(["hello world"], epochs=1, learning_rate=0.1)
        records = model.train(["hello world"], TrainConfig(epochs=3), epochs=2, shuffle=False)
        self.assertEqual(len(records), 2)
        self.assertEqual(model.train(["hello world"], epochs=0), [])
        with self.assertRaises(TypeError):
            model.train([1, 2])
        self.assertEqual(model.train(["hi", ""], epochs=1)[0]["skipped_short"], 2)

    def test_same_seed_same_model(self):
        a, b = corpus_model(seed=4, epochs=3), corpus_model(seed=4, epochs=3)
        self.assertEqual([r["loss"] for r in a.history], [r["loss"] for r in b.history])
        self.assertEqual(a.predict("the cat", to_end=True).text, b.predict("the cat", to_end=True).text)
        self.assertEqual(a.tree.z, b.tree.z)


# ---------------------------------------------------------------------------
# prediction and generation
# ---------------------------------------------------------------------------


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = corpus_model(seed=1, epochs=6)

    def test_recites_the_corpus(self):
        model = self.model
        for prefix, rest in (("the quick", " brown fox jumps over the lazy dog"), ("knowledge", " is power"), ("two plus", " two equals four")):
            r = model.predict(prefix, length=30, to_end=True)
            self.assertEqual(r.text, rest)
            self.assertTrue(r.reached_end)
            self.assertEqual(r.legs, 1)
            self.assertEqual(r.full_text, prefix + rest)
            self.assertEqual(r.cost, 0.0)  # every step of the way was the only option

    def test_each_node_is_visited_at_most_once(self):
        model = self.model
        r = model.predict("", length=200, to_end=True)  # from START, through the whole tree if need be
        self.assertLessEqual(r.expanded, model.tree.num_nodes() + model.tree.num_ends() + FIRST)
        self.assertEqual(len(set(r.node_ids)), len(r.node_ids))
        self.assertEqual(r.node_ids[0], START)
        for p, c in zip(r.node_ids, r.node_ids[1:]):
            self.assertEqual(model.tree.parent[c], p)

    def test_length_and_max_length(self):
        model = self.model
        r = model.predict("the cat", length=5)
        self.assertTrue(len(r.text) >= 5 or r.reached_end)
        self.assertEqual(model.predict("the cat", length=30, max_length=4).text, " sat")
        # length 0: the deterministic remainder of the located run, and nothing decided
        free = model.predict("the cat", length=0)
        node, offset, _ = model.tree.locate(ENC.encode("the cat"))
        self.assertEqual(free.text, model.tree.labels[node][offset + 3 :])
        self.assertEqual((free.cost, free.expanded, free.node_ids), (0.0, 1, [node]))
        self.assertEqual(model.predict("", length=0).text, "")
        with self.assertRaises(ValueError):
            model.predict("x", length=-1)
        with self.assertRaises(ValueError):
            model.predict("x", mode="beam")
        with self.assertRaises(TypeError):
            model.predict(3)

    def test_unknown_and_partial_prefixes(self):
        model = self.model
        r = model.predict("zzzq", length=10)
        self.assertEqual(r.node_ids[0], START)  # nothing known: a new text
        self.assertTrue(r.text)
        r = model.predict("th", length=10)
        self.assertTrue(("th" + r.text).startswith("the"))
        r = model.predict("the qu", length=10)  # ends inside a compressed run
        self.assertTrue(r.text.startswith("ick brown"))

    def test_deterministic(self):
        model = self.model
        a = model.predict("the cat", length=20)
        b = model.predict("the cat", length=20)
        self.assertEqual((a.text, a.cost, a.node_ids), (b.text, b.cost, b.node_ids))

    def test_bounded_depth_re_enters_the_tree(self):
        model = corpus_model(seed=1, epochs=3, depth=4)
        r = model.predict("the cat", length=40)
        self.assertGreaterEqual(r.legs, 2)
        self.assertTrue(len(r.text) >= 40 or r.reached_end)
        one = model.predict("the cat", length=40, max_legs=1)
        self.assertEqual(one.legs, 1)
        self.assertLess(len(one.text), 40)
        with self.assertRaises(ValueError):
            model.predict("the cat", max_legs=0)
        full = model.predict("the cat", length=40, to_end=True)
        self.assertLessEqual(full.legs, MAX_LEGS)

    def test_sampling(self):
        model = self.model
        a = model.predict("the", length=30, mode="sample", seed=3)
        b = model.predict("the", length=30, mode="sample", seed=3)
        self.assertEqual(a.text, b.text)
        greedy = model.predict("the", length=30, mode="sample", temperature=0)
        self.assertEqual(greedy.text, model.predict("the", length=30, mode="sample", temperature=0).text)
        to_end = model.predict("", length=10, mode="sample", to_end=True, seed=5, max_length=200)
        self.assertTrue(to_end.reached_end or len(to_end.text) >= 200)
        with self.assertRaises(ValueError):
            model.predict("the", mode="sample", temperature=-1)
        texts = [r.text for r in model.generate(max_length=40, count=3, seed=7)]
        self.assertEqual(len(texts), 3)
        self.assertEqual(len(model.generate(mode="dijkstra", count=3)), 1)
        with self.assertRaises(ValueError):
            model.generate(count=-1)

    def test_negative_costs_are_legal_here(self):
        """No cycles means no negative cycles: a signed cost and a negative step penalty both terminate."""
        model = self.model
        r = model.predict("the cat", length=20, costs="signal")
        self.assertTrue(r.text)
        for p, c in zip(r.node_ids, r.node_ids[1:]):
            self.assertEqual(model.tree.parent[c], p)
        bonus = model.predict("the", length=10, step_penalty=-0.5, to_end=True)
        self.assertTrue(bonus.reached_end)
        s = model.predict("the", length=10, mode="sample", costs="signal", seed=1)
        self.assertTrue(s.text)
        with self.assertRaises(ValueError):
            model.predict("the", costs="reward")

    def test_search_fallback_when_the_window_runs_out(self):
        t = RadixTree(seed=1, depth=3)
        t.observe_sequence(ENC.encode("abcdef"))
        node, offset, _ = t.locate(ENC.encode("abc"))
        r = cheapest_path(t, node, offset, min_chars=50)
        self.assertFalse(r.reached_end)  # nothing reaches 50 characters: the furthest node comes back
        self.assertTrue(r.text)
        with self.assertRaises(ValueError):
            cheapest_path(t, node, offset, min_chars=-1)
        with self.assertRaises(ValueError):
            cheapest_path(t, 999, 0, min_chars=1)
        with self.assertRaises(ValueError):
            sample_walk(t, node, offset, max_chars=5, temperature=-1)
        self.assertIsInstance(r.to_dict(), dict)


class TestSlide(unittest.TestCase):
    """Token by token: one query of the tree per token, the token fed back, the next query from the new context."""

    @classmethod
    def setUpClass(cls):
        cls.model = corpus_model(seed=1, epochs=6)

    def test_one_query_per_token(self):
        r = self.model.predict("the quick", length=40, mode="slide", to_end=True)
        self.assertEqual(r.text, " brown fox jumps over the lazy dog")
        self.assertEqual(r.expanded, len(r.text) + 1)  # every character one query, and one for END
        self.assertTrue(r.reached_end)
        self.assertEqual(r.cost, 0.0)
        self.assertEqual(len(r.node_ids), r.expanded)

    def test_a_window_forgets(self):
        model = self.model
        whole = model.predict("the cat", length=60, mode="slide", to_end=True)
        narrow = model.predict("the cat", length=60, mode="slide", to_end=True, window=1)
        self.assertNotEqual(whole.text, narrow.text)
        # a window of one gram is the first-order walk: every token is the choice at the last gram's context
        grams = ENC.encode("the cat")
        for step in range(min(8, len(narrow.text))):
            node, offset = model.context(grams, 1)
            _child, expected, _cost = model.next_token(node, offset)
            self.assertEqual(narrow.text[step], expected[-1])
            grams.append(expected)

    def test_a_window_cut_from_a_history_never_begins_a_text(self):
        model = self.model
        t = model.tree
        grams = ENC.encode("zz the quick brown")
        node, _offset = model.context(grams, 3)
        path = []
        while node != ROOT:
            path.append(node)
            node = t.parent[node]
        self.assertNotIn(START, path)
        node, _offset = model.context(ENC.encode("the quick"), None)
        self.assertEqual(t.symbols_of(node, _offset), len(ENC.encode("the quick")) + 1)  # the whole history, from START

    def test_stops_and_the_clock(self):
        model = self.model
        self.assertLessEqual(len(model.predict("the", length=5, mode="slide").text), 6)
        whole_walk = model.predict("the cat", length=30, mode="slide").text
        capped = model.predict("the cat", length=30, mode="slide", max_length=4).text
        self.assertEqual((len(capped), whole_walk.startswith(capped)), (4, True))
        r = model.predict("the", length=10, mode="slide", to_end=True, window=2)
        self.assertLessEqual(len(r.text), MAX_SLIDE)  # a window that forgets can go round; this is its clock
        a = model.predict("the", length=30, mode="slide", temperature=0.8, seed=3, window=4)
        b = model.predict("the", length=30, mode="slide", temperature=0.8, seed=3, window=4)
        self.assertEqual(a.text, b.text)
        self.assertEqual(len(model.generate(mode="slide", count=3)), 1)  # temperature 0 is deterministic: one text
        self.assertEqual(len(model.generate(mode="slide", count=3, temperature=0.8, seed=1, window=4)), 3)
        self.assertTrue(model.predict("", length=10, mode="slide").text)
        self.assertTrue(model.predict("zzzq", length=10, mode="slide").text)  # nothing known: a new text
        self.assertTrue(("th" + model.predict("th", length=10, mode="slide").text).startswith("the"))
        with self.assertRaises(ValueError):
            model.predict("the", mode="slide", window=0)

    def test_accuracy(self):
        model = self.model
        whole = model.next_token_accuracy(CORPUS)
        narrow = model.next_token_accuracy(CORPUS, window=1)
        self.assertEqual(set(whole), {"window", "tokens", "accuracy", "known"})
        self.assertEqual(whole["known"], 1.0)  # a training text is always among the options
        self.assertGreater(whole["accuracy"], narrow["accuracy"])
        self.assertEqual(whole["tokens"], sum(len(ENC.encode(t)) + 1 for t in CORPUS))


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestPhonetic(unittest.TestCase):
    """The tree over sounds: text read through the phonetic tokenizer, predictions spelled back into words."""

    def test_text_becomes_sounds(self):
        enc = Encoding(unit=PHONES)
        enc.validate()
        self.assertEqual(enc.units("The cat sat."), ["DH", "AH0", "#", "K", "AE1", "T", "#", "S", "AE1", "T", "."])
        self.assertEqual(enc.encode("the cat"), ["DH AH0 #", "AH0 # K", "# K AE1", "K AE1 T"])
        self.assertEqual(enc.decode_grams(enc.encode("the cat")), "DH AH0 # K AE1 T")
        self.assertEqual(enc.spell("DH AH0 # K AE1 T"), "the cat")
        self.assertEqual(enc.spell_tail("DH AH0 # K AE1 T # S AE1 T", "# S AE1 T"), " sat")
        self.assertEqual(enc.first_gram("DH AH0 # K AE1 T"), "DH AH0 #")
        self.assertEqual(enc.last_unit("K AE1 T"), "T")
        self.assertEqual(enc.grams_held("DH AH0 # K AE1 T"), 4)
        self.assertTrue(enc.has_unit_prefix("DH AH0 # K AE1 T # S AE1 T", "the cat"))
        self.assertEqual(str(enc), "phone:3:1")
        self.assertEqual(Encoding.from_dict(enc.to_dict()), enc)
        self.assertEqual(Encoding(unit=SYLLABLES, n=2).encode("butter cup"), ["B.AH1.T ER0", "ER0 #", "# K.AH1.P"])

    def test_a_tree_over_sounds_recites_in_words(self):
        model = RadixTreeNet(seed=1, encoding=Encoding(unit=PHONES))
        model.train(CORPUS, epochs=3, **FAST)
        model.tree.check_invariants(texts=CORPUS, compressed=True)
        self.assertEqual(model.stats()["units"], "phones")
        r = model.predict("the quick", length=20, to_end=True)
        self.assertEqual(r.spelled, " brown fox jumps over the lazy dog")
        self.assertTrue(r.text.startswith("# B R AW1 N"))
        self.assertEqual(r.full_spelled, "the quick brown fox jumps over the lazy dog")
        s = model.score("the cat sat on the mat")
        self.assertEqual((s["unknown_transitions"], s["units"]), (0, "phones"))
        self.assertEqual(model.bits_per_char(CORPUS)["unknown_transitions"], 0)
        slide = model.predict("the cat", length=12, to_end=True, mode="slide", window=4)
        self.assertEqual(slide.expanded, len(slide.text.split()) + 1)  # one query per sound, one for END
        self.assertTrue(slide.spelled.startswith(" "))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json.gz")
            model.save(path)
            loaded = load_model(path)
            self.assertEqual(loaded.encoding, model.encoding)
            self.assertEqual(loaded.predict("the quick", to_end=True).spelled, r.spelled)

    def test_syllables(self):
        model = RadixTreeNet(seed=1, encoding=Encoding(unit=SYLLABLES, n=2))
        model.train(CORPUS, epochs=3, **FAST)
        model.tree.check_invariants(texts=CORPUS, compressed=True)
        self.assertEqual(model.predict("the cat", length=10, to_end=True).spelled, " sat on the mat")
        self.assertEqual(model.stats()["units"], "syllables")

    def test_cli_speaks_in_sounds(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(["predict", "--unit", "phone", "--epochs", "2", "the quick", "--to-end"])
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().strip(), "the quick brown fox jumps over the lazy dog")
        self.assertIn("sounds DH AH0 # K W IH1 K", err.getvalue())


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_slow(model: RadixTreeNet, text: str) -> tuple[float, int]:
    """The scoring rule stated the slow way: relocate at every step to the deepest usable context."""
    t = model.tree
    enc = model.encoding
    grams = enc.encode(text)
    log_prob = 0.0
    unknown = 0
    history: list[str] = []
    for g in grams + [None]:
        if history:
            loc = t.locate(history, model.min_count, ending=g is None)
        else:
            loc = (START, -1, 1) if t.count[START] >= model.min_count else None
        node, offset = (loc[0], loc[1]) if loc is not None else (ROOT, -1)
        if g is None:
            leaf = t.end_leaf[node] if t.at_end(node, offset) else -1
            lp = t.log_prob(node, leaf) if leaf >= 0 else None
        elif not t.at_end(node, offset):
            lp = 0.0 if enc.gram_at(t.labels[node], offset + 1) == g else None
        else:
            c = t.children[node].get(g)
            lp = t.log_prob(node, c) if c is not None else None
        if lp is None:
            unknown += 1
            log_prob += math.log(UNKNOWN_PROB)
        else:
            log_prob += lp
        if g is not None:
            history.append(g)
    return log_prob, unknown


class TestScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = corpus_model(seed=1, epochs=4)

    def test_trained_texts_are_certain(self):
        model = self.model
        s = model.score("the quick brown fox jumps over the lazy dog")
        self.assertEqual(s["unknown_transitions"], 0)
        self.assertGreater(s["log_prob"], math.log(UNKNOWN_PROB))
        bad = model.score("the quick brown cat")
        self.assertGreater(bad["unknown_transitions"], 0)
        self.assertLess(bad["log_prob"], s["log_prob"])
        self.assertEqual(
            model.score("ab"),
            {"log_prob": 0.0, "per_char": 0.0, "chars": 2, "transitions": 0, "unknown_transitions": 0, "units": "chars"},
        )
        with self.assertRaises(TypeError):
            model.score(None)
        bits = model.bits_per_char(CORPUS)
        self.assertEqual(bits["unknown_transitions"], 0)
        self.assertGreater(bits["bits_per_char"], 0.0)
        self.assertEqual(set(bits), {"bits_per_char", "chars", "transitions", "unknown_transitions", "miss_rate", "units"})

    def test_the_repetition_the_graph_generalises_the_tree_does_not(self):
        model = RadixTreeNet(seed=1)
        model.train(["aaaa"], epochs=1)
        s = model.score("aaaaaaa")
        self.assertEqual(s["unknown_transitions"], 3)  # the cyclic graph scores this with none
        self.assertEqual(model.score("aaaa")["unknown_transitions"], 0)

    def test_incremental_scoring_equals_relocating_at_every_step(self):
        rng = random.Random(5)
        for depth, min_count in ((None, 1), (None, 2), (4, 1), (5, 2)):
            model = RadixTreeNet(seed=3, depth=depth, min_count=min_count)
            texts = random_texts(rng, "ab c", 30, 4, 16)
            model.train(texts, epochs=1, **FAST)
            probes = texts[:10] + random_texts(rng, "abc d", 15, 3, 18)
            for text in probes:
                fast = model.score(text)
                slow = score_slow(model, text)
                self.assertAlmostEqual(fast["log_prob"], slow[0], places=9, msg=(depth, min_count, text))
                self.assertEqual(fast["unknown_transitions"], slow[1], (depth, min_count, text))

    def test_a_trained_text_never_misses_at_any_depth(self):
        """Every context and its next gram were inserted in one window, so a text the tree holds is certain.

        The one way to break this is to ask a context at the window's limit for
        a gram it could never have seen (it stood at the end of its window), and
        it was broken that way once: a history whose last ``depth`` grams had
        ended a line elsewhere matched that line's END context and missed.
        """
        for depth in (None, 3, 4, 8):
            model = RadixTreeNet(seed=1, depth=depth)
            model.train(CORPUS, epochs=0)
            bits = model.bits_per_char(CORPUS)
            self.assertEqual(bits["unknown_transitions"], 0, depth)
            self.assertEqual(model.bits_per_char(CORPUS[:5])["miss_rate"], 0.0, depth)

    def test_min_count_trusts_shorter_contexts(self):
        strict = RadixTreeNet(seed=1, min_count=2)
        strict.train(CORPUS, epochs=2, **FAST)
        loose = RadixTreeNet(seed=1, min_count=1)
        loose.train(CORPUS, epochs=2, **FAST)
        text = "the quick brown fox jumps over the lazy dog"
        self.assertGreater(strict.score(text)["transitions"], loose.score(text)["transitions"])
        with self.assertRaises(ValueError):
            RadixTreeNet(min_count=0)


# ---------------------------------------------------------------------------
# inversion and 2NRL
# ---------------------------------------------------------------------------


class TestInversion(unittest.TestCase):
    def test_invert_twice_is_identity_and_reverses_every_ranking(self):
        model = corpus_model(seed=2, epochs=2)
        t = model.tree
        w, a, k = list(t.w), list(t.a), list(t.k)
        branches = [i for i in t.alive_nodes() if t.num_children(i) > 1][:50]
        before = {i: [c for c, _ in sorted(t.child_probs(i), key=lambda item: -item[1])] for i in branches}
        model.invert()
        self.assertTrue(t.inverted)
        for i in branches:
            after = [c for c, _ in sorted(t.child_probs(i), key=lambda item: -item[1])]
            self.assertEqual(after, list(reversed(before[i])), i)
        node = t._new_real("qqq", ROOT)  # a node born inverted carries the inverted defaults
        self.assertEqual(t.a[node], -DEFAULT_A)
        self.assertLess(t.w[node], 0.0)
        del t.children[ROOT]["qqq"]
        t.alive[node] = False
        t.labels[node] = ""
        t.parent[node] = -1
        t._n_real -= 1
        model.invert()
        self.assertFalse(t.inverted)
        self.assertEqual((t.w[:node], t.a[:node], t.k[:node]), (w, a, k))

    def test_path_inversion_is_exact_on_a_tree(self):
        """Every edge of a text's path flips sign: the two-colouring the cyclic graph can only approximate."""
        model = corpus_model(seed=2, epochs=2)
        t = model.tree
        text = "the quick brown fox jumps over the lazy dog"
        path = t.node_path(ENC.encode(text))
        before = [dict(t.child_scores(p))[c] for p, c in zip(path, path[1:])]
        out = model.invert_paths([text])
        self.assertEqual(out["unit"], "nodes")
        self.assertEqual(out["flipped"], (len(path) - 1 + 1) // 2)  # every other node after START, END leaf included
        after = [dict(t.child_scores(p))[c] for p, c in zip(path, path[1:])]
        self.assertEqual(len(after), len(path) - 1)
        for x, y in zip(before, after):
            self.assertAlmostEqual(y, -x, places=12)
        self.assertEqual(model.meta["path_inversions"], out["flipped"])
        with self.assertRaises(ValueError):
            model.invert_paths([text], amounts=[1.0, 1.0])
        with self.assertRaises(ValueError):
            model.invert_paths([text], amounts=[-1.0])
        with self.assertRaises(ValueError):
            model.invert_paths([text], mode="weights")
        half = model.invert_paths(["a text the tree has never seen before"], mode="state", amounts=[0.5])
        self.assertGreater(half["flipped"], 0)  # registered structurally first, then attenuated to neutral

    def test_two_nrl(self):
        """2NRL as feedback on a trained model, read at the branches where a garbage line leaves its twin.

        On a tree the negative phase can only teach where a branch exists: a
        garbage line the tree has never seen is a corridor, one option at every
        step and no gradient.  So the phase runs on a model that already holds
        the good texts, where the swapped word is a real branch to train toward.
        The inversion then reverses every pairwise preference exactly - the
        count after it is the complement of the count before - and the positive
        phase leaves the good continuation preferred at nearly every branch.
        """
        good = CORPUS[:30]
        pairs = [(max(CORPUS, key=lambda g: len(os.path.commonprefix([g, b]))), b) for b in GARBAGE]
        pairs = [(g, b) for g, b in pairs if g in good]
        bad = [b for _, b in pairs]

        def prefers_good(model):
            t = model.tree
            wins = total = 0
            for g, b in pairs:
                pg, pb = t.node_path(ENC.encode(g)), t.node_path(ENC.encode(b))
                for x, y, x2, y2 in zip(pg, pg[1:], pb, pb[1:]):
                    if x == x2 and y != y2:  # where the two paths part
                        probs = dict(t.child_probs(x))
                        wins += probs[y] > probs[y2]
                        total += 1
                        break
            return wins, total

        model = RadixTreeNet(seed=3)
        model.train(good, epochs=1, lr=1.0, batch_size=1)
        negative = model.train(bad, epochs=2, lr=1.0, batch_size=1, phase="negative")
        after_negative, total = prefers_good(model)
        self.assertEqual(total, len(pairs))
        self.assertLess(after_negative, total)  # the failures were learned
        model.invert()
        self.assertEqual(prefers_good(model), (total - after_negative, total))  # every preference reversed, exactly
        positive = model.train(good, epochs=2, lr=1.0, act_lr=0.1, batch_size=1, phase="positive")
        after_positive, _ = prefers_good(model)
        self.assertGreaterEqual(after_positive, 0.85 * total)
        self.assertEqual([r["phase"] for r in negative + positive], ["negative"] * 2 + ["positive"] * 2)
        # the same three steps through the method itself
        again = RadixTreeNet(seed=3)
        again.train(good, epochs=1, lr=1.0, batch_size=1)
        out = again.two_nrl(bad, good, neg_epochs=2, pos_epochs=2, neg_lr=1.0, pos_lr=1.0, batch_size=1)
        self.assertTrue(out["inverted"])
        self.assertEqual(again.meta["twonrl_runs"], 1)
        self.assertEqual(prefers_good(again), (after_positive, total))
        with self.assertRaises(TypeError):
            again.two_nrl(bad, good, epochs=1)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


class TestPersistence(unittest.TestCase):
    def test_round_trip_and_continued_training(self):
        a = RadixTreeNet(seed=5, depth=5, min_count=2)
        a.train(CORPUS, epochs=2, **FAST)
        d = a.to_dict()
        b = RadixTreeNet.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(b.to_dict()["tree"], d["tree"])
        self.assertEqual(b.stats(), a.stats())
        self.assertEqual(b.predict("the cat", length=20).text, a.predict("the cat", length=20).text)
        a.train(CORPUS, epochs=2, **FAST)
        b.train(CORPUS, epochs=2, **FAST)  # the RNG state travelled: the same shuffles, the same numbers
        self.assertEqual([r["loss"] for r in a.history], [r["loss"] for r in b.history])
        self.assertEqual(a.tree.z, b.tree.z)
        b.tree.check_invariants(texts=CORPUS)

    def test_files(self):
        model = corpus_model(seed=6, epochs=1, depth=4)
        model.compress()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("model.json", "model.json.gz"):
                path = os.path.join(tmp, name)
                model.save(path)
                loaded = load_model(path)
                self.assertEqual(loaded.to_dict()["tree"], model.to_dict()["tree"])
                self.assertEqual(loaded.meta["epochs_total"], 1)
                loaded.tree.check_invariants(texts=CORPUS)
                self.assertEqual(len(loaded.tree.labels), len(loaded.tree.alive_nodes()))  # compacted
        with self.assertRaises(ValueError):
            RadixTreeNet.from_dict({"format": "radixnet"})
        with self.assertRaises(ValueError):
            RadixTree.from_dict({"format": "radixtree-tree", "format_version": 99})


# ---------------------------------------------------------------------------
# corpus, comparison, CLI
# ---------------------------------------------------------------------------


class TestCorpusAndCompare(unittest.TestCase):
    def test_lines_are_prose(self):
        text = "# Title\n\n| a | b |\n```\ncode here that is long enough\n```\n- **bold** words in a list item long enough\n> a quoted line that is also long enough\nshort\nSee http://x for more of this line\n"
        self.assertEqual(corpus.lines_of(text), ["bold words in a list item long enough", "a quoted line that is also long enough"])
        train, test = corpus.split(list("abcdefghij"), 5)
        self.assertEqual((train, test), (list("abcdfghi"), ["e", "j"]))
        self.assertEqual(corpus.split(["a"], 0), (["a"], []))
        self.assertEqual(len(corpus.digest(["a", "b"])), 64)
        built = corpus.build(20)
        self.assertEqual(len(built), 20)
        self.assertTrue(all(corpus.MIN_LINE <= len(t) <= corpus.MAX_LINE for t in built))

    def test_compare_runs(self):
        result = compare(CORPUS[:20], CORPUS[20:24], epochs=1, verbose=False)
        self.assertEqual(set(result["models"]), {"tree", "cyclic"})
        tree = result["models"]["tree"]
        self.assertTrue(tree["available"])
        for key in ("stats", "train_bits", "test_bits", "recall", "predictions", "epoch_loss"):
            self.assertIn(key, tree)
        self.assertEqual(len(result["repetition"]), 8)
        self.assertFalse(result["generalisation"]["tree_representable"])
        self.assertIn("| real nodes |", markdown(result))
        if load_radixnet() is not None:
            self.assertTrue(result["models"]["cyclic"]["available"])
            self.assertTrue(result["generalisation"]["cyclic_representable"])
            self.assertEqual(result["repetition"][3]["cyclic_nodes"], 1)
        self.assertEqual(result["repetition"][3]["tree_nodes"], 999)


class TestCLI(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_train_predict_score_stats(self):
        sample = os.path.join(DATA, "sample_corpus.txt")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json.gz")
            code, out, _ = self.run_cli("train", "--data", sample, "--epochs", "2", "--save", path)
            self.assertEqual(code, 0)
            self.assertIn('"nodes"', out)
            code, out, err = self.run_cli("predict", "--load", path, "the quick", "--to-end")
            self.assertEqual(code, 0)
            self.assertEqual(out.strip(), "the quick brown fox jumps over the lazy dog")
            code, out, _ = self.run_cli("predict", "--load", path, "the cat", "--json", "--sample", "--sample-seed", "1")
            self.assertEqual(code, 0)
            self.assertIn("full_text", json.loads(out))
            code, out, _ = self.run_cli("score", "--load", path, "the cat sat on the mat", "xyzzy plugh")
            self.assertEqual(code, 0)
            self.assertIn("bits_per_char", out)
            code, out, _ = self.run_cli("stats", "--load", path)
            self.assertEqual(json.loads(out)["kind"], "radix-tree")
            code, out, _ = self.run_cli("generate", "--load", path, "--count", "2", "--sample-seed", "3")
            self.assertEqual(code, 0)
            self.assertEqual(len(out.strip().splitlines()), 2)
            inverted = os.path.join(tmp, "inv.json")
            code, out, _ = self.run_cli("invert", "--load", path, "--save", inverted)
            self.assertEqual(code, 0)
            self.assertTrue(load_model(inverted).tree.inverted)
        code, _, _ = self.run_cli("score", "--data", sample, "--epochs", "0")
        self.assertEqual(code, 2)

    def test_check_and_2nrl(self):
        code, out, _ = self.run_cli("check")
        self.assertEqual(code, 0)
        self.assertIn("worst error", out)
        code, out, _ = self.run_cli("2nrl", "--neg-epochs", "1", "--pos-epochs", "1")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["inverted"])


if __name__ == "__main__":
    unittest.main()
