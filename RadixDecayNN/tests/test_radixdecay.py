"""The test suite of ``radixdecay``: the tree is one tree, ``seen`` is the memory, and every verb behaves as
``DESIGN.md`` says.  Standard library ``unittest``, deterministic, seconds."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(os.path.dirname(ROOT_DIR), "PhoneticTokenizer"))

from radixdecay import (  # noqa: E402
    DECAYS, FIRST, LIFE, MAX_SLIDE, MIN_SEEN, PHONES, ROOT, START, SYLLABLES, UNKNOWN_PROB, DecayNet, DecayTree,
    Encoding, Walks, cheapest_path, load_model,
)
from radixdecay import experiment  # noqa: E402
from radixdecay.cli import main as cli_main  # noqa: E402
from radixdecay.corpus import SAMPLE, read_texts, split  # noqa: E402
from radixdecay.encoding import START_LABEL  # noqa: E402
from radixdecay.tree import KIND_END, KIND_REAL  # noqa: E402

try:
    import phonetok  # noqa: F401
except ImportError:  # pragma: no cover
    phonetok = None

TEXTS = read_texts(SAMPLE)
SMALL = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the mat",
    "a quick brown fox jumps over the lazy dog",
]
HELD = ["the cat sat on the moon", "a quick brown fox jumps over the lazy cat", "zzz qqq", "the dog sat on the floor"]


def snap(tree: DecayTree) -> tuple:
    return (list(tree.value), list(tree.stamp), tree.traversals, tree.structure_version)


def read_model(texts, **kw) -> DecayNet:
    model = DecayNet(**kw)
    model.read(texts)
    return model


def under_start(tree: DecayTree, node: int) -> bool:
    while node > ROOT:
        if node == START:
            return True
        node = tree.parent[node]
    return False


def run_cli(*argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = cli_main(list(argv))
    return code, out.getvalue()


class TestEncoding(unittest.TestCase):
    def test_trigrams_round_trip(self):
        enc = Encoding()
        self.assertEqual(enc.encode("hello"), ["hel", "ell", "llo"])
        self.assertEqual(enc.decode_grams(enc.encode("hello world")), "hello world")
        self.assertEqual(enc.encode("hi"), [])
        self.assertEqual(Encoding(n=1).encode("abc"), ["a", "b", "c"])

    def test_decode_path_context_and_remainder(self):
        enc = Encoding()
        self.assertEqual(enc.decode_path(["the cat", "at sat"], 0, True), "the cat sat")
        self.assertEqual(enc.decode_path(["the cat", "at sat"], 2, False), "at sat")


class TestTreeBasics(unittest.TestCase):
    def test_fresh_tree(self):
        tree = DecayTree()
        self.assertEqual((tree.num_nodes(), tree.num_ends(), tree.traversals), (0, 0, 0))
        self.assertEqual((tree.seen(ROOT), tree.seen(START)), (0.0, 0.0))
        self.assertEqual(tree.labels[START], START_LABEL)
        self.assertEqual(FIRST, 2)
        tree.check_invariants()
        self.assertIn("DecayTree(", repr(tree))

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            DecayTree(depth=1)
        with self.assertRaises(ValueError):
            DecayTree(decay="slowly")
        for life in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                DecayTree(life=life)
        with self.assertRaises(ValueError):
            DecayTree(min_seen=-0.1)
        for decay in DECAYS:
            DecayTree(decay=decay)
        with self.assertRaises(ValueError):
            DecayNet(decay="never")

    def test_reading_one_text_visits_every_window(self):
        tree = DecayTree(decay="none")
        made = tree.read(tree.encoding.encode("abcd"))
        # START:[abc,bcd]  ROOT:[abc,bcd]  ROOT:[bcd] - each window touches its origin, its node and its END leaf
        self.assertEqual(made, 9)
        self.assertEqual(tree.traversals, 9)
        self.assertEqual((tree.seen(START), tree.seen(ROOT)), (1.0, 2.0))
        self.assertEqual((tree.num_nodes(), tree.num_ends()), (3, 3))
        self.assertEqual(tree.grams, {"abc", "bcd"})
        self.assertAlmostEqual(tree.total_seen(), 9.0)
        tree.check_invariants(["abcd"])
        tree.read(tree.encoding.encode("abcd"))
        self.assertEqual(tree.traversals, 18)
        self.assertEqual((tree.seen(START), tree.seen(tree.children[START]["abc"])), (2.0, 2.0))
        self.assertAlmostEqual(tree.total_seen(), 18.0)

    def test_bad_grams_rejected(self):
        tree = DecayTree()
        with self.assertRaises(ValueError):
            tree.read(["ab", "abc"])
        self.assertEqual(tree.read([]), 0)
        with self.assertRaises(TypeError):
            DecayNet().read([1, 2])


class TestNoCycles(unittest.TestCase):
    def test_aaaa_makes_no_self_loop(self):
        tree = DecayTree(decay="none")
        tree.read(["aaa", "aaa"])
        tree.check_invariants(["aaaa"])
        self.assertIsNotNone(tree.walk(ROOT, ["aaa", "aaa"]))
        self.assertIsNone(tree.walk(ROOT, ["aaa"] * 3))
        self.assertIsNone(tree.node_path(["aaa"] * 5))
        for i in tree.alive_nodes():
            if i != ROOT:
                self.assertEqual(tree.depth_of(i), tree.depth_of(tree.parent[i]) + 1)

    def test_repetition_is_unrolled(self):
        for n in range(4, 11):
            tree = DecayTree()
            tree.read(tree.encoding.encode("a" * n))
            self.assertEqual(tree.num_nodes(), n - 1, n)
            tree.check_invariants(["a" * n])

    def test_every_edge_goes_one_level_deeper(self):
        tree = read_model(TEXTS).tree
        seen_once: set[int] = set()
        stack = [ROOT]
        while stack:
            p = stack.pop()
            self.assertNotIn(p, seen_once)
            seen_once.add(p)
            for c in tree.child_items(p) + ([START] if p == ROOT else []):
                self.assertEqual(tree.parent[c], p)
                self.assertEqual(tree.depth_of(c), tree.depth_of(p) + 1)
                stack.append(c)
        self.assertEqual(seen_once, set(tree.alive_nodes()))
        tree.check_invariants(TEXTS, compressed=True)

    def test_invariants_catch_a_cycle(self):
        tree = read_model(SMALL).tree
        node = next(i for i in tree.real_nodes() if tree.children[i])
        child = next(iter(tree.children[node].values()))
        tree.parent[node] = child
        with self.assertRaises(AssertionError):
            tree.check_invariants()


class TestSuffixProperty(unittest.TestCase):
    def test_every_substring_is_a_root_path(self):
        tree = read_model(SMALL).tree
        for text in SMALL:
            grams = tree.encoding.encode(text)
            for lo in range(len(grams)):
                for hi in range(lo + 1, len(grams) + 1):
                    self.assertIsNotNone(tree.walk(ROOT, grams[lo:hi]), (text, lo, hi))
        tree.check_invariants(SMALL)

    def test_locate_prefers_the_whole_history_then_backs_off(self):
        model = read_model(SMALL, decay="none")
        tree, enc = model.tree, model.encoding
        grams = enc.encode("the cat sat")
        node, offset, symbols = tree.locate(grams)
        self.assertTrue(under_start(tree, node))
        self.assertEqual(symbols, len(grams) + 1)
        grams = enc.encode("cat sat on")
        node, offset, symbols = tree.locate(grams)
        self.assertFalse(under_start(tree, node))
        self.assertEqual(symbols, len(grams))
        node, offset, symbols = tree.locate(enc.encode("the cat"), anchored=False)
        self.assertFalse(under_start(tree, node))
        self.assertIsNone(tree.locate(enc.encode("zzzz")))
        # `longest` considers no suffix longer than that, and then not the walk from START either
        node, offset, symbols = tree.locate(enc.encode("the cat sat"), longest=2)
        self.assertEqual(symbols, 2)
        self.assertFalse(under_start(tree, node))

    def test_walks_answer_what_locate_answers(self):
        """The cursors moved a gram at a time find the context a fresh walk of every suffix finds - on a tree
        that has said things and faded, where a longer context can outlive a shorter one."""
        rng = random.Random(3)
        for depth in (None, 6):
            model = read_model(TEXTS, depth=depth, life=3000)
            for _ in range(5):
                model.say("the", to_end=True, window=4)
            model.tick(2000)
            tree, enc = model.tree, model.encoding
            for window in (None, 1, 3, 5):
                for text in TEXTS[::7] + HELD + ["".join(rng.choice("the cat") for _ in range(12))]:
                    walks = Walks(tree, window=window)
                    history: list[str] = []
                    for g in enc.encode(text):
                        walks.push(g)
                        history.append(g)
                        expected = model.context(history, window)
                        got = walks.deepest()
                        self.assertEqual(None if got is None else (got[0], got[1]), expected, (depth, window, text, len(history)))
                        if got is not None:
                            cut = window is not None and len(history) > window
                            self.assertEqual(got, tree.locate(history[-window:] if cut else history, anchored=not cut))
        with self.assertRaises(ValueError):
            Walks(DecayTree(), window=0)
        self.assertIsNone(Walks(DecayTree()).deepest())

    def test_bounded_depth_limits_the_context(self):
        tree = DecayTree(depth=4)
        tree.read(tree.encoding.encode("abcdefg"))
        tree.check_invariants(["abcdefg"])
        for i in tree.real_nodes():
            self.assertLessEqual(tree.symbols_of(i, tree.held(i) - 1), 4, tree.labels[i])
        grams = tree.encoding.encode("abcdefg")
        self.assertLessEqual(tree.locate(grams)[2], 3)
        self.assertLessEqual(tree.locate(grams, ending=True)[2], 4)


class TestMemory(unittest.TestCase):
    def test_seen_reads_through_time_and_changes_nothing(self):
        tree = DecayTree(life=100, decay="half-life")
        tree.read(tree.encoding.encode("abcd"))
        node = tree.children[START]["abc"]
        self.assertEqual(tree.value[node], 1.0)
        self.assertLess(tree.seen(node), 1.0)  # the reading itself took traversals, and the clock ran
        value, stamp = tree.value[node], tree.stamp[node]
        tree.tick(100 - (tree.traversals - stamp))  # exactly one life since the visit
        self.assertAlmostEqual(tree.seen(node), 0.5)
        tree.tick(100)
        self.assertAlmostEqual(tree.seen(node), 0.25)
        self.assertEqual((tree.value[node], tree.stamp[node]), (value, stamp))  # lazy: nothing was written
        self.assertAlmostEqual(tree.settle(node), 0.25)
        self.assertEqual((tree.value[node], tree.stamp[node]), (0.25, tree.traversals))
        self.assertAlmostEqual(tree.seen(node), 0.25)

    def test_the_three_decays(self):
        for decay, expected in (("half-life", (0.5, 0.25)), ("linear", (0.0, 0.0)), ("none", (1.0, 1.0))):
            tree = DecayTree(life=100, decay=decay)
            tree.read(tree.encoding.encode("abcd"))
            node = tree.children[START]["abc"]
            tree.tick(100 - (tree.traversals - tree.stamp[node]))  # one life since the visit
            self.assertAlmostEqual(tree.seen(node), expected[0], msg=decay)
            tree.tick(100)
            self.assertAlmostEqual(tree.seen(node), expected[1], msg=decay)
        tree = DecayTree(life=100, decay="linear")
        tree.read(tree.encoding.encode("abcd"))
        node = tree.children[START]["abc"]
        tree.tick(50 - (tree.traversals - tree.stamp[node]))
        self.assertAlmostEqual(tree.seen(node), 0.5)

    def test_touch_settles_adds_one_and_ticks(self):
        tree = DecayTree(life=100, decay="half-life")
        tree.read(tree.encoding.encode("abcd"))
        node = tree.children[START]["abc"]
        tree.tick(100 - (tree.traversals - tree.stamp[node]))
        clock = tree.traversals
        self.assertAlmostEqual(tree.touch(node), 1.5)
        self.assertEqual((tree.value[node], tree.stamp[node], tree.traversals), (1.5, clock, clock + 1))
        self.assertLess(tree.touch(node, 2.5), 4.0)  # a hair faded over the one tick in between
        tree = DecayTree(decay="none")
        tree.read(tree.encoding.encode("abcd"))
        node = tree.children[START]["abc"]
        self.assertEqual(tree.touch(node, 2.5), 3.5)
        self.assertEqual(tree.touch(node, 2.5), 6.0)

    def test_tick_only_moves_the_clock(self):
        tree = read_model(SMALL).tree
        before = snap(tree)
        tree.tick(5)
        self.assertEqual(tree.traversals, before[2] + 5)
        self.assertEqual((tree.value, tree.stamp, tree.structure_version), (before[0], before[1], before[3]))
        tree.tick(0)
        with self.assertRaises(ValueError):
            tree.tick(-1)
        model = DecayNet()
        self.assertEqual(model.tick(7), 7)
        self.assertEqual(model.meta["ticks"], 7)

    def test_shares_are_proportions_of_seen(self):
        model = DecayNet(decay="none")
        model.read(["the cat sat"] * 3 + ["the dog sat"])
        tree = model.tree
        branch = experiment.first_branch(model, "the")
        shares = dict(tree.shares(branch))
        self.assertEqual(len(shares), 2)
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertEqual(sorted(shares.values()), [0.25, 0.75])
        for c, s in shares.items():
            self.assertAlmostEqual(tree.log_share(branch, c), math.log(s))
            self.assertAlmostEqual(dict(tree.child_costs(branch))[c], -math.log(s))
        self.assertIsNone(tree.log_share(branch, START))
        self.assertIsNone(tree.log_share(branch, ROOT))
        self.assertIsNone(tree.log_share(branch, branch))
        self.assertAlmostEqual(tree.option_total(branch), 4.0)

    def test_faded_options_are_not_offered(self):
        model = DecayNet(decay="linear", life=1000, min_seen=0.1)
        tree, enc = model.tree, model.encoding
        model.read("the cat sat")
        model.tick(500)
        model.read("the dog sat")
        branch = experiment.first_branch(model, "the")
        self.assertEqual(len(tree.shares(branch)), 2)
        model.tick(600)  # the cat, read more than a life ago, is gone; the dog still holds a third of a visit
        shares = tree.shares(branch)
        self.assertEqual(len(shares), 1)
        self.assertAlmostEqual(shares[0][1], 1.0)
        self.assertEqual(tree.first_gram(shares[0][0]), "e d")
        self.assertIsNotNone(tree.locate(enc.encode("the ")))
        model.tick(400)  # and now the dog
        self.assertEqual(tree.shares(branch), [])
        self.assertFalse(tree.has_options(branch))
        self.assertFalse(tree.usable(branch, tree.held(branch) - 1))
        self.assertIsNone(tree.locate(enc.encode("the ")))  # at the branch: nothing to continue with
        self.assertIsNotNone(tree.locate(enc.encode("the")))  # inside the run: the next gram is certain

    def test_min_seen_is_the_floor(self):
        for min_seen, located_after in ((MIN_SEEN, False), (0.0, True)):
            model = DecayNet(life=100, min_seen=min_seen)
            tree, enc = model.tree, model.encoding
            model.read("the cat")
            node, offset, _ = tree.locate(enc.encode("the cat"))
            model.tick(100 - (tree.traversals - tree.stamp[node]))
            self.assertTrue(tree.usable(node, offset))  # exactly half a visit is still remembered
            model.tick(1)
            self.assertEqual(tree.usable(node, offset), min_seen == 0.0)
            model.tick(tree.traversals)  # long past every visit: every context holds less than half of one
            self.assertEqual(tree.locate(enc.encode("the cat")) is not None, located_after, min_seen)

    def test_remembered_and_total_seen(self):
        model = DecayNet(decay="none")
        model.read(SMALL)
        tree = model.tree
        self.assertGreaterEqual(tree.total_seen(), tree.traversals)  # a split copies seen to both halves
        self.assertEqual(tree.remembered(), tree.num_nodes())
        model = DecayNet(life=10000)
        model.read(SMALL)
        tree = model.tree
        self.assertEqual(tree.remembered(), tree.num_nodes())
        model.tick(10000 + tree.traversals)
        self.assertLess(tree.remembered(), tree.num_nodes())
        self.assertEqual(tree.remembered(min_seen=0.0), tree.num_nodes())
        self.assertEqual(tree.remembered(min_seen=1e9), 0)

    def test_split_keeps_seen_on_both_halves_and_merge_keeps_the_larger(self):
        tree = DecayTree(decay="none")
        tree.read(tree.encoding.encode("abcdef"))
        node = tree.children[START]["abc"]
        self.assertEqual(tree.held(node), 4)
        tree.tick(3)
        node, deep = tree.split(node, 2)
        self.assertEqual((tree.labels[node], tree.labels[deep]), ("abcd", "cdef"))
        self.assertEqual((tree.value[node], tree.stamp[node]), (tree.value[deep], tree.stamp[deep]))
        self.assertEqual(tree.seen(node), tree.seen(deep))
        self.assertEqual(tree.parent[tree.end_leaf[deep]], deep)
        self.assertEqual(tree.end_leaf[node], -1)
        tree.check_invariants(["abcdef"])
        with self.assertRaises(ValueError):
            tree.split(node, 0)
        with self.assertRaises(ValueError):
            tree.split(START, 1)
        tree.touch(deep)
        self.assertEqual(tree.compress(), 1)
        self.assertEqual(tree.labels[node], "abcdef")
        self.assertEqual(tree.seen(node), 2.0)  # the larger of the two
        self.assertFalse(tree.alive[deep])
        tree.check_invariants(["abcdef"], compressed=True)

    def test_reading_never_leaves_a_chain(self):
        for depth in (None, 6):
            tree = DecayTree(depth=depth)
            for t in TEXTS:
                tree.read(tree.encoding.encode(t))
            self.assertEqual(tree.compress(), 0, depth)
            tree.check_invariants(TEXTS, compressed=True)

    def test_option_total_cache_follows_the_clock_and_the_structure(self):
        model = DecayNet(decay="none")
        model.read(SMALL)
        tree = model.tree
        branch = experiment.first_branch(model, "the")
        total = tree.option_total(branch)
        child = tree.shares(branch)[0][0]
        tree.touch(child)
        self.assertAlmostEqual(tree.option_total(branch), total + 1.0)
        tree.split(child, 1)
        self.assertAlmostEqual(tree.option_total(branch), math.fsum(tree.seen(c) for c in tree.child_items(branch)))
        model = DecayNet(life=100)
        model.read(SMALL)
        tree = model.tree
        branch = experiment.first_branch(model, "the")
        total = tree.option_total(branch)
        tree.tick(100)
        self.assertAlmostEqual(tree.option_total(branch), total / 2)

    def test_random_operations_keep_one_tree(self):
        rng = random.Random(7)
        tree = DecayTree(life=50, decay=rng.choice(DECAYS))
        enc = tree.encoding
        texts: list[str] = []
        for step in range(300):
            op = rng.random()
            if op < 0.4 or tree.num_nodes() == 0:
                text = "".join(rng.choice("abc ") for _ in range(rng.randint(3, 9)))
                texts.append(text)
                tree.read(enc.encode(text))
            elif op < 0.6:
                node = rng.choice(tree.real_nodes())
                if tree.held(node) >= 2:
                    tree.split(node, rng.randint(1, tree.held(node) - 1))
            elif op < 0.8:
                tree.touch(rng.choice(tree.alive_nodes()), rng.choice((1.0, 0.5, 3.0)))
            elif op < 0.9:
                tree.tick(rng.randint(0, 40))
            else:
                tree.compress()
            tree.check_invariants(texts)
        tree.compress()
        tree.check_invariants(texts, compressed=True)

    def test_persistence_settles_the_values(self):
        tree = DecayTree(life=100)
        tree.read(tree.encoding.encode("abcd"))
        tree.tick(100)
        d = tree.to_dict()
        self.assertEqual(d["traversals"], tree.traversals)
        self.assertAlmostEqual(d["nodes"]["seen"][START], tree.seen(START))
        self.assertLess(d["nodes"]["seen"][START], 1.0)
        back = DecayTree.from_dict(d)
        self.assertEqual(back.traversals, tree.traversals)
        self.assertTrue(all(s == back.traversals for s in back.stamp))
        for i in tree.alive_nodes():
            self.assertAlmostEqual(back.seen(i), tree.seen(i))
        tree.tick(30)
        back.tick(30)
        for i in tree.alive_nodes():
            self.assertAlmostEqual(back.seen(i), tree.seen(i))
        back.check_invariants(["abcd"])
        with self.assertRaises(ValueError):
            DecayTree.from_dict({"format": "something else"})
        with self.assertRaises(ValueError):
            DecayTree.from_dict({**d, "format_version": 99})


class TestSaying(unittest.TestCase):
    def test_recites_the_corpus_quietly(self):
        model = read_model(TEXTS)
        clock = model.clock
        rec = model.recites(TEXTS)
        self.assertEqual(rec["texts"], 60)
        self.assertGreaterEqual(rec["recited"], 55)
        self.assertEqual(model.clock, clock)
        self.assertEqual(model.meta["traversals_said"], 0)

    def test_saying_is_traversal_and_asking_quietly_is_not(self):
        model = read_model(SMALL, decay="none")
        tree = model.tree
        before = snap(tree)
        quiet = model.say("the cat", to_end=True, quiet=True)
        self.assertEqual(snap(tree), before)
        self.assertEqual(quiet.traversals, 0)
        said = model.say("the cat", to_end=True)
        self.assertEqual(said.text, quiet.text)
        self.assertTrue(said.reached_end)
        self.assertGreaterEqual(said.traversals, 3)
        self.assertEqual(tree.traversals, before[2] + said.traversals)
        changed = [i for i in range(len(before[0])) if tree.value[i] != before[0][i]]
        self.assertEqual(len(changed), said.traversals)
        for i in changed:
            self.assertEqual(tree.value[i], before[0][i] + 1.0)
        self.assertIn(tree.locate(model.encoding.encode("the cat"))[0], changed)  # the context it started from
        self.assertEqual(model.meta["traversals_said"], said.traversals)

    def test_reading_and_saying_weigh_the_same(self):
        model = DecayNet(decay="none")
        model.read(["the cat sat", "the dog sat"] * 2)
        self.assertEqual(model.say("the", to_end=True).text, " cat sat")  # a tie goes to the older option
        model.read(["the dog sat"] * 2)
        self.assertEqual(model.say("the", to_end=True).text, " dog sat")  # read 4 beats read 2 + said 1
        model.say("the", to_end=True)
        model.read(["the cat sat"] * 2)
        self.assertEqual(model.say("the", to_end=True, quiet=True).text, " dog sat")  # 4 read + 2 said beats 4 + 1
        branch = experiment.first_branch(model, "the")
        seen = {model.tree.first_gram(c): model.tree.seen(c) for c, _ in model.tree.shares(branch)}
        self.assertEqual(seen, {"e c": 5.0, "e d": 6.0})

    def test_a_habit_forms(self):
        model = read_model(TEXTS)
        branch = experiment.first_branch(model, "the")
        before = experiment.branch_shares(model, branch)
        probabilities = []
        text = None
        for _ in range(10):
            r = model.say("the", to_end=True)
            self.assertTrue(text is None or r.text == text)
            text = r.text
            probabilities.append(math.exp(-r.cost))
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertGreater(probabilities[-1], 3 * probabilities[0])
        after = experiment.branch_shares(model, branch)
        self.assertEqual(after[0]["option"], before[0]["option"])
        self.assertGreater(after[0]["share"], before[0]["share"] * 2)
        self.assertLess(experiment.entropy([a["share"] for a in after]), experiment.entropy([b["share"] for b in before]))

    def test_one_query_per_token(self):
        model = read_model(SMALL)
        r = model.say("the cat", to_end=True)
        self.assertEqual(r.expanded, len(r.text) + 1)  # one query per character, one for END
        self.assertEqual(r.full_text, "the cat" + r.text)
        self.assertEqual(r.to_dict()["expanded"], r.expanded)

    def test_a_window_forgets(self):
        model = read_model(["abcabcabcabcabc"], decay="none")
        r = model.say("abc", to_end=True, window=1, max_length=60)
        self.assertEqual(len(r.text), 60)
        self.assertFalse(r.reached_end)
        self.assertTrue(("abc" + r.text).startswith("abcabcabc"))
        r = model.say("abc", to_end=True, window=1, quiet=True)
        self.assertEqual(len(r.text), MAX_SLIDE)  # the clock stops a walk that goes round
        r = model.say("abc", to_end=True, quiet=True)  # the whole history: no window, no loop
        self.assertTrue(r.reached_end)
        self.assertEqual("abc" + r.text, "abcabcabcabcabc")

    def test_a_window_cut_from_a_history_never_begins_a_text(self):
        model = read_model(["abcd", "bcdxyz"], decay="none")
        tree = model.tree
        loc = model.context(["abc", "bcd"], window=1)
        self.assertFalse(under_start(tree, loc[0]))
        loc = model.context(["bcd"])
        self.assertTrue(under_start(tree, loc[0]))
        loc = model.context(["abc", "bcd"])
        self.assertTrue(under_start(tree, loc[0]))
        self.assertIsNone(model.context(["zzz"]))
        with self.assertRaises(ValueError):
            model.say("abc", window=0)

    def test_stops(self):
        model = read_model(SMALL)
        self.assertEqual(len(model.say("the cat", length=3).text), 3)
        r = model.say("the cat", length=100)
        self.assertTrue(r.reached_end)
        self.assertEqual(len(model.say("the cat", max_length=2).text), 2)
        self.assertEqual(len(model.say("the cat", to_end=True, max_length=5).text), 5)
        self.assertEqual(model.say("the cat", length=0).text, "")
        with self.assertRaises(ValueError):
            model.say("the cat", length=-1)
        with self.assertRaises(ValueError):
            model.say("the cat", temperature=-1)
        with self.assertRaises(TypeError):
            model.say(3)

    def test_short_and_unknown_prefixes(self):
        model = read_model(SMALL)
        r = model.say("th", to_end=True)
        self.assertTrue(r.full_text.startswith("the "))
        self.assertGreaterEqual(r.traversals, 1)
        r = model.say("zzzq", to_end=True)  # nothing known: the walk begins a text
        self.assertTrue(r.full_text.startswith("zzzq"))
        self.assertTrue(r.text.startswith("the") or r.text.startswith("a "))
        self.assertTrue(r.reached_end)
        self.assertEqual(model.say("", length=0).text, "")

    def test_a_faded_start_says_nothing(self):
        model = DecayNet(life=10)
        model.read(SMALL)
        model.tick(1000)
        r = model.say("", to_end=True)
        self.assertEqual((r.text, r.traversals, r.expanded, r.reached_end), ("", 0, 0, False))
        self.assertEqual(model.cheapest("", to_end=True).text, "")
        self.assertEqual(model.say("zzzq", to_end=True).full_text, "zzzq")
        self.assertEqual([r.text for r in model.generate(2)], ["", ""])

    def test_temperature_draws_and_seeds(self):
        model = read_model(TEXTS)
        a = model.say("the", to_end=True, temperature=1.0, seed=3, quiet=True)
        b = model.say("the", to_end=True, temperature=1.0, seed=3, quiet=True)
        self.assertEqual(a.text, b.text)
        texts = {model.say("the", to_end=True, temperature=1.0, seed=s, quiet=True).text for s in range(10)}
        self.assertGreater(len(texts), 1)
        cold = {model.say("the", to_end=True, temperature=0.0, seed=s, quiet=True).text for s in range(5)}
        self.assertEqual(len(cold), 1)
        with self.assertRaises(ValueError):
            model.generate(-1)

    def test_generate_changes_the_tree(self):
        model = read_model(SMALL, decay="none")
        clock = model.clock
        outs = model.generate(3, max_length=40)
        self.assertEqual(len(outs), 3)
        self.assertEqual(model.clock, clock + sum(r.traversals for r in outs))
        for r in outs:
            self.assertTrue(r.full_text.startswith("the ") or r.full_text.startswith("a "))
        clock = model.clock
        model.generate(2, quiet=True)
        self.assertEqual(model.clock, clock)


class TestCheapest(unittest.TestCase):
    def test_cheapest_is_the_most_probable_continuation(self):
        model = DecayNet(decay="none")
        model.read(SMALL + ["the cat sat on the mat"])
        r = model.cheapest("the cat", to_end=True, quiet=True)
        self.assertEqual(r.text, " sat on the mat")
        self.assertTrue(r.reached_end)
        self.assertAlmostEqual(r.cost, -math.log(2 / 3))
        self.assertEqual(r.full_text, "the cat sat on the mat")
        self.assertEqual(r.traversals, 0)
        r = model.cheapest("the cat", length=3, quiet=True)
        self.assertGreaterEqual(len(r.text), 3)

    def test_cheapest_touches_its_path_once_each(self):
        model = read_model(SMALL, decay="none")
        tree = model.tree
        before = list(tree.value)
        r = model.cheapest("the cat", to_end=True)
        distinct = []
        for nid in r.node_ids:
            if not distinct or distinct[-1] != nid:
                distinct.append(nid)
        self.assertEqual(r.traversals, len(distinct))
        for nid in distinct:
            self.assertEqual(tree.value[nid], before[nid] + 1.0)
        self.assertEqual(model.meta["traversals_said"], r.traversals)
        with self.assertRaises(ValueError):
            model.cheapest("the", max_legs=0)

    def test_bounded_depth_re_enters_the_tree(self):
        model = read_model(TEXTS, depth=5)
        r = model.cheapest("the quick", to_end=True, max_legs=40, quiet=True)
        self.assertGreater(len(r.text), 8)  # a leg of a five-symbol tree says one gram; the legs add up
        one = model.cheapest("the quick", to_end=True, max_legs=1, quiet=True)
        self.assertLess(len(one.text), len(r.text))

    def test_search_visits_each_node_at_most_once(self):
        tree = read_model(TEXTS).tree
        r = cheapest_path(tree, START, -1, 10 ** 6)
        self.assertLessEqual(r.expanded, len(tree.alive_nodes()))
        self.assertGreater(len(r.text), 0)
        with self.assertRaises(ValueError):
            cheapest_path(tree, START, -1, -1)


def slow_score(model: DecayNet, text: str) -> tuple[float, int]:
    """The definition the slow way: the deepest usable context of the whole history decides at every step."""
    tree, enc = model.tree, model.encoding
    grams = enc.encode(text)
    if not grams:
        return 0.0, 0
    log_prob = 0.0
    unknown = 0
    history: list[str] = []

    def located(ending: bool = False):
        """The deepest usable context of the whole history, or the empty context (the root) when there is none."""
        if history:
            loc = tree.locate(history, ending)
            return (loc[0], loc[1]) if loc is not None else (ROOT, -1)
        return (START, -1) if tree.usable(START, -1) else (ROOT, -1)

    for g in grams:
        loc = located()
        ok = False
        if loc is not None:
            node, offset = loc
            if offset < tree.held(node) - 1:
                ok = enc.gram_at(tree.labels[node], offset + 1) == g
            else:
                c = tree.children[node].get(g)
                lp = tree.log_share(node, c) if c is not None else None
                if lp is not None:
                    log_prob += lp
                    ok = True
        history.append(g)
        if not ok:
            unknown += 1
            log_prob += math.log(UNKNOWN_PROB)
    loc = located(ending=True)
    lp = None
    if loc is not None:
        node, offset = loc
        if offset == tree.held(node) - 1 and tree.end_leaf[node] >= 0:
            lp = tree.log_share(node, tree.end_leaf[node])
    if lp is None:
        unknown += 1
        log_prob += math.log(UNKNOWN_PROB)
    else:
        log_prob += lp
    return log_prob, unknown


class TestScoring(unittest.TestCase):
    def test_read_texts_are_certain(self):
        model = DecayNet(decay="none")
        model.read(SMALL + ["the cat sat on the mat"])
        for text in SMALL:
            s = model.score(text)
            self.assertEqual(s["unknown_transitions"], 0, text)
            self.assertLessEqual(s["log_prob"], 0.0)
            self.assertEqual(s["units_name"], "chars")
        self.assertGreater(model.score("the cat sat on the mat")["log_prob"], model.score("the cat sat on the floor")["log_prob"])
        bits = model.bits_per_unit(SMALL)
        self.assertEqual(bits["miss_rate"], 0.0)
        self.assertLess(bits["bits_per_unit"], 0.5)
        self.assertEqual(model.score("")["units"], 0)

    def test_unknown_costs_unknown_prob(self):
        model = read_model(SMALL)
        s = model.score("the cat sat on the moon")
        self.assertGreaterEqual(s["unknown_transitions"], 1)
        self.assertLessEqual(s["log_prob"], math.log(UNKNOWN_PROB))
        with self.assertRaises(TypeError):
            model.score(None)

    def test_measurements_are_quiet(self):
        model = read_model(SMALL)
        before = snap(model.tree)
        model.score("the cat sat on the mat")
        model.bits_per_unit(SMALL + HELD)
        model.next_token_accuracy(SMALL, window=2)
        model.recites(SMALL)
        model.stats()
        self.assertEqual(snap(model.tree), before)

    def test_the_fast_walk_equals_the_definition_on_a_tree_built_by_reading(self):
        for depth in (None, 6):
            for min_seen in (MIN_SEEN, 0.0):
                model = DecayNet(depth=depth, life=3000, min_seen=min_seen)
                model.read(TEXTS)
                for lives in (0.0, 0.5, 3.0):
                    if lives:
                        model.tick(int(lives * model.tree.life))
                    for text in TEXTS[::4] + HELD:
                        s = model.score(text)
                        lp, unknown = slow_score(model, text)
                        self.assertAlmostEqual(s["log_prob"], lp, msg=(depth, min_seen, lives, text))
                        self.assertEqual(s["unknown_transitions"], unknown, (depth, min_seen, lives, text))

    def test_a_faded_context_backs_off_to_a_fresher_shorter_one(self):
        model = DecayNet(life=100)
        text = "the cat sat on the mat"
        model.read(text)
        fresh = model.score(text)
        model.tick(150)
        model.read(["cat sat"] * 3)
        faded = model.score(text)
        self.assertLessEqual(faded["unknown_transitions"], 1)  # END may have faded from every context
        self.assertLess(faded["log_prob"], fresh["log_prob"])
        self.assertGreater(faded["log_prob"], faded["transitions"] * math.log(UNKNOWN_PROB))
        model.read(text)
        self.assertGreater(model.score(text)["log_prob"], faded["log_prob"])

    def test_accuracy(self):
        model = read_model(TEXTS)
        acc = model.next_token_accuracy(TEXTS, window=4)
        self.assertEqual(acc["window"], 4)
        self.assertGreater(acc["accuracy"], 0.85)
        self.assertEqual(acc["known"], 1.0)
        first = model.next_token_accuracy(TEXTS, window=1)
        self.assertLess(first["accuracy"], acc["accuracy"])
        whole = model.next_token_accuracy(TEXTS)
        self.assertGreaterEqual(whole["accuracy"], acc["accuracy"])
        self.assertEqual(model.next_token_accuracy([])["tokens"], 0)


class TestForgetting(unittest.TestCase):
    def test_a_memory_fades_and_reading_brings_it_back(self):
        model = read_model(TEXTS)
        nodes = model.tree.num_nodes()
        self.assertGreaterEqual(model.recites(TEXTS)["recited"], 55)
        model.tick(int(2 * LIFE))
        self.assertLessEqual(model.recites(TEXTS)["recited"], 3)
        self.assertLess(model.tree.remembered(), nodes // 5)
        self.assertGreater(model.bits_per_unit(TEXTS)["bits_per_unit"], 3.0)
        model.read(TEXTS)
        self.assertGreaterEqual(model.recites(TEXTS)["recited"], 55)
        self.assertEqual(model.tree.remembered(), nodes)
        self.assertLess(model.bits_per_unit(TEXTS)["bits_per_unit"], 0.3)

    def test_none_never_forgets(self):
        model = read_model(TEXTS, decay="none")
        bits = model.bits_per_unit(TEXTS)["bits_per_unit"]
        model.tick(10 ** 7)
        self.assertEqual(model.bits_per_unit(TEXTS)["bits_per_unit"], bits)
        self.assertGreaterEqual(model.recites(TEXTS)["recited"], 55)

    def test_linear_reaches_nothing(self):
        model = DecayNet(decay="linear", life=100)
        model.read("abcd")
        model.tick(100)
        self.assertEqual(model.tree.seen(START), 0.0)
        self.assertEqual(model.tree.shares(START), [])
        self.assertEqual(model.say("", to_end=True).text, "")

    def test_the_habit_experiment(self):
        out = experiment.habit(TEXTS, times=5, sampled=5)
        said, quiet, sampled = out["arms"]["said"], out["arms"]["quiet"], out["arms"]["sampled"]
        probs = [r["probability"] for r in said["rows"]]
        self.assertEqual(probs, sorted(probs))
        self.assertGreater(probs[-1], probs[0])
        self.assertEqual(len({r["probability"] for r in quiet["rows"]}), 1)
        self.assertEqual(quiet["before"], quiet["after"])
        self.assertGreater(said["top_share_after"], said["top_share_before"])
        self.assertLess(said["entropy_after"], said["entropy_before"])
        self.assertEqual(sampled["times"], 5)
        self.assertGreaterEqual(sampled["distinct"], 1)

    def test_the_forgetting_experiment(self):
        out = experiment.forgetting(TEXTS, lives=(0, 1, 4), decays=("half-life", "none"))
        rows = out["curves"]["half-life"]["rows"]
        self.assertEqual([r["lives"] for r in rows], [0, 1, 4])
        remembered = [r["remembered"] for r in rows]
        self.assertEqual(remembered, sorted(remembered, reverse=True))
        self.assertLess(remembered[-1], remembered[0])
        self.assertLess(rows[-1]["recited"], rows[0]["recited"])
        self.assertGreater(rows[-1]["bits_per_unit"], rows[0]["bits_per_unit"])
        flat = out["curves"]["none"]["rows"]
        self.assertEqual(len({r["remembered"] for r in flat}), 1)
        self.assertEqual(len({r["bits_per_unit"] for r in flat}), 1)

    def test_the_use_experiment(self):
        out = experiment.use(TEXTS, used=TEXTS[-1], rounds=12, gap=1000)
        arms = out["arms"]
        self.assertTrue(arms["said"]["used_recited"])
        self.assertFalse(arms["quiet"]["used_recited"])
        self.assertFalse(arms["silent"]["used_recited"])
        self.assertGreater(arms["said"]["traversals_said"], 0)
        self.assertEqual(arms["quiet"]["traversals_said"], 0)
        self.assertAlmostEqual(arms["said"]["elapsed_lives"], 1.2)

    def test_the_recovery_experiment(self):
        out = experiment.recovery(TEXTS, fade_lives=4, every=3)
        rows = out["rows"]
        self.assertEqual(len(rows), 4)
        self.assertLess(rows[1]["recited"], rows[0]["recited"])
        self.assertGreaterEqual(rows[2]["re_read_recited"], rows[2]["re_read_texts"] - 2)
        self.assertEqual(rows[2]["others_recited"], 0)
        self.assertGreaterEqual(rows[3]["recited"], 55)

    def test_run_writes_files_and_tables_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = experiment.run("habit", SMALL, out_dir=tmp, times=2, sampled=2)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "habit_results.json")))
            self.assertIn("### habit", experiment.tables(results))
            with open(results["habit"]["written"], encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["experiment"], "habit")
        with self.assertRaises(ValueError):
            experiment.run("nope", SMALL)
        results = experiment.run(["forgetting"], SMALL, lives=(0, 1), decays=("linear",))
        self.assertIn("linear", experiment.tables(results))


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        model = read_model(TEXTS, seed=5)
        model.say("the", to_end=True)
        model.tick(500)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json.gz")
            model.save(path)
            back = load_model(path)
            plain = os.path.join(tmp, "m.json")
            model.save(plain)
            self.assertEqual(DecayNet.load(plain).stats(), back.stats())
        for key, value in model.stats().items():
            self.assertEqual(back.stats()[key], value, key)
        for i in model.tree.alive_nodes():
            self.assertAlmostEqual(back.tree.seen(i), model.tree.seen(i))
        self.assertEqual(back.say("the cat", to_end=True, quiet=True).text, model.say("the cat", to_end=True, quiet=True).text)
        self.assertEqual(back.score("the cat sat on the mat"), model.score("the cat sat on the mat"))
        a = [r.text for r in model.generate(3, temperature=1.0)]
        b = [r.text for r in back.generate(3, temperature=1.0)]
        self.assertEqual(a, b)  # the RNG state travels
        self.assertEqual(back.history, model.history)

    def test_bad_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"format": "radixtree"}, fh)
            with self.assertRaises(ValueError):
                load_model(path)
        with self.assertRaises(ValueError):
            DecayNet.from_dict({"format": "radixdecay", "version": 99, "tree": {}})


@unittest.skipUnless(phonetok, "the phonetic tokenizer is not importable")
class TestPhonetic(unittest.TestCase):
    def test_reads_sounds_and_spells_back(self):
        model = DecayNet(encoding=Encoding(unit=PHONES))
        model.read(SMALL)
        self.assertEqual(model.stats()["units"], "phones")
        r = model.say("the cat", to_end=True)
        self.assertTrue(r.full_text.startswith("DH AH0 # K AE1 T"), r.full_text)
        self.assertTrue(r.full_spelled.startswith("the cat"), r.full_spelled)
        self.assertTrue(r.spelled.startswith(" sat"), r.spelled)
        s = model.score("the cat sat on the mat")
        self.assertEqual((s["unknown_transitions"], s["units_name"]), (0, "phones"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.json")
            model.save(path)
            back = load_model(path)
        self.assertEqual(back.encoding.unit, PHONES)
        self.assertEqual(back.say("the cat", to_end=True, quiet=True).full_spelled, r.full_spelled)

    def test_syllables(self):
        model = DecayNet(encoding=Encoding(unit=SYLLABLES, n=2))
        model.read(SMALL)
        self.assertGreaterEqual(model.recites(SMALL, prefix_units=3)["recited"], 3)
        self.assertEqual(model.score("the dog sat on the mat")["unknown_transitions"], 0)
        self.assertEqual(model.stats()["encoding"], "syllable:2:1")

    def test_cli_speaks_in_sounds(self):
        code, out = run_cli("say", "--unit", "phone", "the cat", "--to-end", "--quiet")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("the cat"), out)


class TestCLI(unittest.TestCase):
    def test_read_say_score_stats_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json.gz")
            code, out = run_cli("read", "--save", path)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(json.loads(out)["read"]["texts"], 60)
            code, out = run_cli("say", "--load", path, "the quick", "--to-end", "--times", "2", "--save", path)
            self.assertEqual(code, 0)
            self.assertEqual(out.count("\n"), 2)
            self.assertTrue(out.startswith("the quick"))
            code, out = run_cli("say", "--load", path, "the", "--json", "--quiet")
            self.assertEqual(json.loads(out)[0]["traversals"], 0)
            code, out = run_cli("cheapest", "--load", path, "the", "--to-end", "--json")
            self.assertTrue(json.loads(out)["reached_end"])
            code, out = run_cli("generate", "--load", path, "--count", "2", "--sample-seed", "1")
            self.assertEqual((code, out.count("\n")), (0, 2))
            code, out = run_cli("score", "--load", path, "the quick brown fox jumps over the lazy dog")
            self.assertEqual(code, 0)
            self.assertIn("bits_per_unit", out)
            self.assertEqual(run_cli("score", "--load", path)[0], 2)
            code, out = run_cli("stats", "--load", path)
            stats = json.loads(out)
            self.assertEqual(stats["kind"], "radix-decay")
            self.assertGreater(stats["traversals_said"], 0)
            code, out = run_cli("accuracy", "--load", path, "--windows", "1,4,all")
            self.assertEqual(code, 0)
            self.assertIn("all", out)
            code, out = run_cli("tick", "--load", path, "--lives", "2", "--save", path)
            tick = json.loads(out)
            self.assertLess(tick["after"]["remembered"], tick["before"]["remembered"])
            self.assertEqual(tick["lives"], 2.0)
            code, out = run_cli("read", "--load", path, "--save", path)
            self.assertEqual(json.loads(out)["stats"]["readings"], 2)
            code, out = run_cli("accuracy", "--heldout-every", "5", "--windows", "2", "--out", os.path.join(tmp, "acc.json"))
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "acc.json")))

    def test_aliases_experiment_and_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            self.assertEqual(run_cli("train", "--save", path)[0], 0)
            code, out = run_cli("predict", "--load", path, "the", "--quiet")
            self.assertEqual(code, 0)
            self.assertTrue(out.startswith("the"))
            code, out = run_cli("experiment", "--which", "habit", "--out", tmp, "--data", SAMPLE)
            self.assertEqual(code, 0)
            self.assertIn("### habit", out)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "habit_results.json")))
        code, out = run_cli("demo")
        self.assertEqual(code, 0)
        self.assertIn("recites", out)
        self.assertIn("read once more", out)

    def test_split(self):
        kept, held = split(list(range(10)), 5)
        self.assertEqual((kept, held), ([0, 1, 2, 3, 5, 6, 7, 8], [4, 9]))
        self.assertEqual(split([1, 2], 0), ([1, 2], []))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
