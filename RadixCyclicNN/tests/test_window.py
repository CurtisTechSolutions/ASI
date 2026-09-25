"""Tests for radixnet.window: the dynamic window, a ladder of node sizes halving from 32 to 4 and back up.

The window is the rule of ``../SPEC-DynamicWindow.md``: a ceiling on a node's
length that moves down a ladder of powers of two and back to the top.  Pinned
here: the ladder and what it refuses; a step halving every node longer than
the window into two halves that carry the same data, joined by a heavy
connection, and never a node of one gram; compression that stops at the
window; the cycle back up to the top regrowing what stayed unary; the
automatic step at the end of every training epoch on every kind; off is the
old file to the bit; the file block; the CLI and the HTTP API.
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

from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import parse_encoding  # noqa: E402
from radixnet.graph import FIRST, W_HEAVY, RadixCyclicGraph  # noqa: E402
from radixnet.model import RadixNet, load_model  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.resonance import ResonantNet  # noqa: E402
from radixnet.window import DEFAULT_FLOOR, DEFAULT_TOP, DynamicWindow, check_ladder, is_power_of_two, ladder  # noqa: E402

LONG = "the quick brown fox jumps over the lazy dog while the cat sat on the mat and the bird sang in the tree all afternoon"
TEXTS = [
    LONG,
    "the quick brown fox jumps over the lazy dog while the cat ran to the door",
    "a bird sang in the tree all afternoon",
]
KINDS = (RadixNet, CountRewardNet, NegativeNet, ResonantNet)


def trained(cls, encoding=None, **options):
    """A model of ``cls`` taught the texts (blamed, for the negative network): long unary chains, merged."""
    model = cls(seed=1, encoding=encoding)
    if cls is NegativeNet:
        model.blame(TEXTS, reason="other")
    else:
        model.train(TEXTS, epochs=1, **options)
    return model


def real_lengths(graph):
    return sorted(graph.label_len(n) for n in range(FIRST, len(graph.labels)) if graph.alive[n])


def strip_times(doc):
    doc = json.loads(json.dumps(doc))
    doc.pop("saved_at", None)
    doc.get("meta", {}).pop("created", None)
    for record in doc.get("history", []):
        record.pop("seconds", None)
    for entry in doc.get("log", []):
        entry.pop("at", None)
    return doc


class TestLadder(unittest.TestCase):
    def test_the_ladder_halves_from_the_top_to_the_floor(self):
        self.assertEqual(ladder(32, 4), [32, 16, 8, 4])
        self.assertEqual(ladder(8, 8), [8])
        self.assertEqual(ladder(64, 1), [64, 32, 16, 8, 4, 2, 1])
        self.assertEqual(ladder(DEFAULT_TOP, DEFAULT_FLOOR), [32, 16, 8, 4])

    def test_a_size_is_a_power_of_two(self):
        self.assertEqual([n for n in range(0, 70) if is_power_of_two(n)], [1, 2, 4, 8, 16, 32, 64])
        for top, floor, size in ((20, 4, None), (32, 3, None), (4, 32, None), (32, 4, 12), (32, 4, 64), (32, 8, 4)):
            with self.subTest(top=top, floor=floor, size=size), self.assertRaises(ValueError):
                check_ladder(top, floor, size)
        check_ladder(32, 4, 8)

    def test_the_window_stands_on_the_ladder_and_goes_round(self):
        off = DynamicWindow()
        self.assertFalse(off.on)
        self.assertEqual((off.sizes(), off.next_size(), off.to_dict(), str(off)), ([], None, None, "off"))
        self.assertIs(off.advanced(), off)
        window = DynamicWindow(32, 4)
        self.assertEqual((window.size, window.auto, window.next_size()), (32, True, 16))
        seen = []
        for _ in range(6):
            seen.append(window.size)
            window = window.advanced()
        self.assertEqual(seen, [32, 16, 8, 4, 32, 16])
        self.assertEqual(DynamicWindow(32, 4, 8, False).next_size(), 4)
        self.assertEqual(DynamicWindow(8, 8).next_size(), 8)  # a ladder of one rung stands still
        self.assertIn("32 -> 16 -> 8 -> 4", DynamicWindow(32, 4, 8).describe("chars"))
        with self.assertRaises(ValueError):
            DynamicWindow(32, 4, 12)
        with self.assertRaises(ValueError):
            DynamicWindow("wide", 4)

    def test_the_file_block(self):
        self.assertEqual(DynamicWindow(32, 4, 16, False).to_dict(), {"top": 32, "floor": 4, "size": 16, "auto": False})
        self.assertEqual(DynamicWindow.from_dict({"top": 32, "floor": 4, "size": 16, "auto": False}), DynamicWindow(32, 4, 16, False))
        self.assertEqual(DynamicWindow.from_dict({"top": 16}), DynamicWindow(16, 4, 16, True))
        self.assertEqual(DynamicWindow.from_dict(None), DynamicWindow())
        self.assertEqual(DynamicWindow.from_dict({}), DynamicWindow())
        for bad in ({"top": 12}, {"top": 32, "floor": 64}, {"top": 32, "size": 3}, {"top": "32"}, {"top": 32, "auto": "yes"}, [32]):
            with self.subTest(block=bad), self.assertRaises(ValueError):
                DynamicWindow.from_dict(bad)


class TestSplitWindow(unittest.TestCase):
    def test_a_node_longer_than_the_window_is_halved_at_its_middle_gram(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(g._encoder.encode("abcdefghij"))
        g.compress()
        node = next(n for n in range(FIRST, len(g.labels)) if g.alive[n])
        self.assertEqual(g.labels[node], "abcdefghij")  # 10 units, 8 grams
        self.assertEqual(g.split_window(8), 1)
        self.assertEqual([g.labels[n] for n in range(FIRST, len(g.labels)) if g.alive[n]], ["abcdef", "efghij"])  # 4 + 4 grams
        self.assertEqual(g.split_window(4), 2)  # 6 units > 4: 2 + 2 grams each
        self.assertEqual(sorted(g.labels[n] for n in range(FIRST, len(g.labels)) if g.alive[n]), ["abcd", "cdef", "efgh", "ghij"])
        self.assertEqual(g.split_window(4), 0)  # nothing is longer than the window now
        self.assertEqual(g.split_window(3), 4)  # 4 units > 3: every node halves into two grams of 3
        self.assertEqual(g.split_window(1), 0)  # a node of one gram cannot be halved, however small the window
        g.check_invariants(["abcdefghij"])
        with self.assertRaises(ValueError):
            g.split_window(0)

    def test_a_node_is_cut_into_its_two_halves(self):
        # ABCD under a grouping encoding is the grams AB and CD, and halves into exactly those two nodes
        grouped = RadixCyclicGraph(seed=1, encoding=parse_encoding("char:2:2"))
        grouped.observe_sequence(grouped._encoder.encode("ABCD"))
        grouped.compress()
        self.assertEqual([grouped.labels[n] for n in range(FIRST, len(grouped.labels)) if grouped.alive[n]], ["ABCD"])
        self.assertEqual(grouped.split_window(2), 1)
        self.assertEqual([grouped.labels[n] for n in range(FIRST, len(grouped.labels)) if grouped.alive[n]], ["AB", "CD"])
        grouped.check_invariants(["ABCD"])
        # under the sliding trigram it is ABC -> BCD merged, and halves into those two, sharing the pivot BC:
        # consecutive trigrams overlap by two, and a half is never shorter than one gram
        sliding = RadixCyclicGraph(seed=1)
        sliding.observe_sequence(sliding._encoder.encode("ABCD"))
        sliding.compress()
        self.assertEqual([sliding.labels[n] for n in range(FIRST, len(sliding.labels)) if sliding.alive[n]], ["ABCD"])
        self.assertEqual(sliding.split_window(2), 1)
        self.assertEqual([sliding.labels[n] for n in range(FIRST, len(sliding.labels)) if sliding.alive[n]], ["ABC", "BCD"])
        self.assertEqual(sliding.split_window(2), 0)
        sliding.check_invariants(["ABCD"])
        for spec, text, halves in (("char:4:4", "ABCDEFGH", ["ABCD", "EFGH"]), ("word:2:2", "a b c d", ["a b", "c d"])):
            with self.subTest(encoding=spec):
                g = RadixCyclicGraph(seed=1, encoding=parse_encoding(spec))
                g.observe_sequence(g._encoder.encode(text))
                g.compress()
                g.split_window(g.encoding.n)
                self.assertEqual([g.labels[n] for n in range(FIRST, len(g.labels)) if g.alive[n]], halves)

    def test_the_halves_carry_the_same_data_and_a_heavy_connection(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(g._encoder.encode("abcdefghij"))
        g.observe_sequence(g._encoder.encode("abcdefghij"))
        g.compress()
        node = next(n for n in range(FIRST, len(g.labels)) if g.alive[n])
        params = (g.z[node], g.a[node], g.b[node], g.h[node], g.k[node])
        g.z[node] = 1.5 * math.pi + 0.3  # a learned state and offset, to see them copied - at the crest of the sine
        g.h[node] = 0.3  # b * (z - h) = pi / 2: |f| = 1
        self.assertEqual(g.split_window(8), 1)
        a, b = node, len(g.labels) - 1
        self.assertEqual((g.z[b], g.a[b], g.b[b], g.h[b], g.k[b]), (1.5 * math.pi + 0.3, params[1], params[2], 0.3, params[4]))
        self.assertEqual((g.z[a], g.h[a]), (1.5 * math.pi + 0.3, 0.3))
        self.assertAlmostEqual(abs(g.activation_of(a)), 1.0)
        self.assertEqual(g.node_count(b), g.node_count(a))
        e = g.children[a][b]
        self.assertEqual(g.edge_traversals(e), g.node_count(a))  # the bridge carries every traversal of the node
        self.assertEqual(g.edge_w[e], W_HEAVY)
        self.assertEqual(g.child_probs(a), [(b, 1.0)])  # the only way on, and the heaviest
        # another child attaching to the first half competes with a fresh weight (at most 1.5) against the heavy
        # one (8) between the same activation and one no larger: the bridge stays the likeliest way on
        g.observe_sequence(g._encoder.encode("abcdefxyz"))
        probs = dict(g.child_probs(a))
        self.assertGreater(probs[b], 0.99, probs)
        g.check_invariants(["abcdefghij", "abcdefxyz"])

    def test_the_heavy_weight_follows_the_inverted_network(self):
        g = RadixCyclicGraph(seed=1)
        g.observe_sequence(g._encoder.encode("abcdefghij"))
        g.compress()
        g.invert()
        g.split_window(8)
        heavy = [g.edge_w[e] for e, ok in enumerate(g.edge_alive) if ok and abs(g.edge_w[e]) == W_HEAVY]
        self.assertEqual(heavy, [-W_HEAVY])  # negated while inverted, as every fresh weight is
        g.invert()
        self.assertEqual([g.edge_w[e] for e, ok in enumerate(g.edge_alive) if ok and abs(g.edge_w[e]) == W_HEAVY], [W_HEAVY])

    def test_the_counting_kinds_are_heavy_by_their_count(self):
        for cls in (CountRewardNet, NegativeNet, ResonantNet):
            with self.subTest(kind=cls.kind):
                model = trained(cls)
                g = model.graph
                self.assertFalse(g.learns_weights)
                first = len(g.labels)
                longest = max(range(FIRST, first), key=lambda n: g.label_len(n) if g.alive[n] else -1)
                # what passed through the node: its visits, or - in the phase model, which counts its edges and
                # not its nodes - the traversals of its out-edges; the negative network counts neither (its
                # evidence is blame, and a bridge carries none), so its bridge honestly weighs nothing
                through = sum(g.edge_traversals(e) for e in g.children[longest].values())
                expected = through if cls is ResonantNet else g.node_count(longest)
                if cls is not NegativeNet:
                    self.assertGreater(expected, 0)
                self.assertGreater(g.split_window(16), 0)
                bridges = 0
                for b in range(first, len(g.labels)):  # every second half is new, with its first half as its one parent
                    (a, e), = g.parents[b].items()
                    if g.node_count(a) != g.node_count(longest) or g.label_len(a) + g.label_len(b) > 40:
                        continue  # a half of another node
                    bridges += 1
                    self.assertEqual(g.edge_traversals(e), expected, (cls.kind, g.labels[a], g.labels[b]))
                    self.assertNotEqual(g.edge_w[e], W_HEAVY)  # a weight computed from the count, not a number set
                    self.assertEqual(g.child_probs(a), [(b, 1.0)])  # the only way on
                self.assertGreater(bridges, 0)
                g.check_invariants(TEXTS if cls is not NegativeNet else None)

    def test_compression_stops_at_the_window(self):
        for cls in KINDS:
            with self.subTest(kind=cls.kind):
                model = trained(cls)
                g = model.graph
                self.assertGreater(max(real_lengths(g)), 16)
                model.configure_window(top=16, floor=4, size=16)
                g.split_window(16)
                self.assertLessEqual(max(real_lengths(g)), 16)
                self.assertEqual(g.compress(), 0)  # the halves would merge back over 16: the window holds them apart
                g.check_invariants(TEXTS if cls is not NegativeNet else None, compressed=True)
                model.configure_window(on=False)
                self.assertGreater(g.compress(), 0)  # off: unbounded again, and the unary halves merge back
                self.assertGreater(max(real_lengths(g)), 16)
                g.check_invariants(TEXTS if cls is not NegativeNet else None, compressed=True)

    def test_a_node_of_one_gram_stays_whole_under_a_grouping_or_word_encoding(self):
        for spec in ("char:4:4", "word:3:1", "word:2:2"):
            with self.subTest(encoding=spec):
                model = CountRewardNet(seed=1, encoding=parse_encoding(spec))
                model.train(TEXTS, epochs=1)
                g = model.graph
                enc = g.encoding
                model.configure_window(top=8, floor=1, size=8)
                for size in (8, 4, 2, 1):
                    g.split_window(size)
                    for n in range(FIRST, len(g.labels)):
                        if g.alive[n]:
                            self.assertTrue(g.label_len(n) <= size or g.grams_held(n) == 1, (size, g.labels[n]))
                            self.assertEqual((g.label_len(n) - enc.n) % enc.stride, 0)
                    g.check_invariants(TEXTS)


class TestSteps(unittest.TestCase):
    def test_a_step_merges_halves_and_moves(self):
        model = trained(RadixNet)
        g = model.graph
        with self.assertRaises(ValueError):
            model.window_step()  # off
        model.configure_window(on=True)
        self.assertEqual(model.window_config()["size"], 32)
        done = model.window_step()
        self.assertEqual((done["sizes"], done["from"], done["to"], done["steps"]), ([32], 32, 16, 1))
        self.assertLessEqual(max(real_lengths(g)), 32)
        self.assertEqual(g.dynamic_window.size, 16)
        done = model.window_step(3)
        self.assertEqual((done["sizes"], done["to"]), ([16, 8, 4], 32))
        self.assertLessEqual(max(real_lengths(g)), 4)
        self.assertEqual(done["nodes_after"] - done["nodes_before"], done["splits"])
        self.assertGreater(done["splits"], 0)
        self.assertEqual(done["window"], model.window_config())
        with self.assertRaises(ValueError):
            model.window_step(0)
        for text in TEXTS:
            self.assertEqual(model.score(text)["unknown_transitions"], 0)

    def test_back_at_the_top_the_unary_halves_grow_together_again(self):
        model = trained(CountRewardNet)
        g = model.graph
        model.configure_window(on=True)
        nodes = g.num_nodes()
        model.window_step(4)  # 32, 16, 8, 4: the graph is as fine as the floor
        self.assertLessEqual(max(real_lengths(g)), 4)
        self.assertGreater(g.num_nodes(), nodes)
        done = model.window_step()  # at 32 again: what stayed unary merges back
        self.assertGreater(done["merges"], 0)
        self.assertEqual(done["splits"], 0)
        self.assertEqual(g.num_nodes(), nodes)
        self.assertEqual(sorted(real_lengths(g)), real_lengths(g))
        for text in TEXTS:
            self.assertEqual(model.score(text)["unknown_transitions"], 0)

    def test_the_settings(self):
        model = RadixNet(seed=1)
        config = model.window_config()
        self.assertEqual((config["on"], config["size"], config["sizes"], config["longer"], config["heavy"]), (False, None, [], None, W_HEAVY))
        self.assertEqual(model.configure_window()["on"], False)  # nothing asked, nothing changed
        config = model.configure_window(on=True)
        self.assertEqual((config["top"], config["floor"], config["size"], config["auto"], config["sizes"], config["next"]), (32, 4, 32, True, [32, 16, 8, 4], 16))
        config = model.configure_window(size=8, auto=False)
        self.assertEqual((config["size"], config["auto"]), (8, False))
        config = model.configure_window(top=16)  # the size stays on the ladder
        self.assertEqual((config["top"], config["size"], config["auto"]), (16, 8, False))
        config = model.configure_window(floor=16)
        self.assertEqual((config["floor"], config["size"], config["sizes"]), (16, 16, [16]))
        config = model.configure_window(top=64, floor=2, size=4)
        self.assertEqual((config["size"], config["sizes"]), (4, [64, 32, 16, 8, 4, 2]))
        for bad in (dict(top=20), dict(floor=3), dict(floor=128), dict(size=12), dict(size=1)):
            with self.subTest(**bad), self.assertRaises(ValueError):
                model.configure_window(**bad)
        config = model.configure_window(on=False, size=8)
        self.assertEqual((config["on"], config["size"], config["longer"]), (False, None, None))
        self.assertIsNone(CountRewardNet(seed=1).window_config()["heavy"])  # heavy by its count
        self.assertIsNone(model.stats()["dynamic_window"])

    def test_it_steps_by_itself_at_the_end_of_every_training_epoch(self):
        for cls in (RadixNet, CountRewardNet, ResonantNet):
            with self.subTest(kind=cls.kind):
                model = trained(cls)
                model.configure_window(on=True)
                records = model.train(TEXTS, epochs=5)
                self.assertEqual([r["window"] for r in records], [32, 16, 8, 4, 32])
                self.assertEqual([r["splits"] > 0 for r in records], [False, True, True, True, False])
                self.assertGreater(records[4]["merges"], 0)  # back at the top, the epoch's compression regrew the chains
                self.assertEqual(model.graph.dynamic_window.size, 16)
                self.assertEqual(model.stats()["dynamic_window"], 16)
                for text in TEXTS:
                    self.assertEqual(model.score(text)["unknown_transitions"], 0)
                model.configure_window(auto=False)
                more = model.train(TEXTS, epochs=2)
                self.assertNotIn("window", more[0])  # by hand: the ladder does not move
                self.assertEqual(model.graph.dynamic_window.size, 16)

    def test_the_negative_network_steps_on_its_blame_passes(self):
        model = trained(NegativeNet)
        model.configure_window(on=True)
        records = model.blame(TEXTS, reason="other", epochs=3)
        self.assertEqual([r["window"] for r in records], [32, 16, 8])
        self.assertEqual(model.graph.dynamic_window.size, 4)
        cleared = model.clear(TEXTS[:1])
        self.assertNotIn("window", cleared[0])  # a clearing pass is not a training pass
        self.assertEqual(model.graph.dynamic_window.size, 4)

    def test_a_feedback_pass_steps_too(self):
        model = trained(RadixNet)
        model.configure_window(on=True)
        records = model.reward(TEXTS[:1], epochs=2)
        self.assertEqual([r["window"] for r in records], [32, 16])


class TestFile(unittest.TestCase):
    def test_it_travels_with_the_file(self):
        for cls in KINDS:
            model = cls(seed=1, encoding=parse_encoding("word:3:1"))
            model.configure_attention(blur=0.3) if model.takes_corrections else None
            model.configure_window(top=16, floor=4, size=8, auto=False)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.json")
                model.save(path)
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                self.assertEqual(doc["graph"]["dynamic_window"], {"top": 16, "floor": 4, "size": 8, "auto": False})
                keys = list(doc["graph"])
                expected = ["format", "format_version", "encoding"] + (["attention"] if model.takes_corrections else []) + ["dynamic_window", "seed"]
                self.assertEqual(keys[: len(expected)], expected)
                again = load_model(path)
            with self.subTest(kind=cls.kind):
                self.assertEqual(again.graph.dynamic_window, DynamicWindow(16, 4, 8, False))
                self.assertEqual(again.stats()["dynamic_window"], 8)

    def test_off_is_the_old_file_to_the_bit(self):
        for cls in KINDS:
            with self.subTest(kind=cls.kind):
                model = trained(cls)
                before = json.dumps(strip_times(model.to_dict()), sort_keys=True)
                model.configure_window(on=True)
                model.configure_window(on=False)
                self.assertEqual(json.dumps(strip_times(model.to_dict()), sort_keys=True), before)
                self.assertNotIn("dynamic_window", model.to_dict()["graph"])


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-window-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, model_path=os.path.join(cls.tmp.name, "model.json")
        )

    def test_the_window_through_the_api(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.post("/api/model/select", {"kind": "count"})
        self.assertEqual(status, 200, data)
        status, data, _ = self.client.post("/api/train", {"texts": TEXTS, "epochs": 1})
        self.assertEqual(status, 202, data)
        wait_for_job(self.client)
        status, data, _ = self.client.get("/api/model/window")
        self.assertEqual((status, data["kind"], data["window"]["on"], data["window"]["longer"]), (200, "count", False, None))
        self.assertGreater(data["window"]["longest"], 16)
        status, data, _ = self.client.post("/api/model/window/step", {})
        self.assertEqual(status, 400)
        self.assertIn("off", data["error"])
        status, data, _ = self.client.post("/api/model/window", {"on": True})
        self.assertEqual((status, data["window"]["sizes"], data["window"]["size"], data["stats"]["dynamic_window"]), (200, [32, 16, 8, 4], 32, 32))
        self.assertEqual(self.client.get("/api/status")[1]["dynamic_window"], 32)
        self.assertEqual(self.client.get("/api/model")[1]["dynamic_window"]["size"], 32)
        status, data, _ = self.client.post("/api/model/window", {"size": 16})
        self.assertEqual((status, data["window"]["size"], data["window"]["next"]), (200, 16, 8))
        self.assertGreater(data["window"]["longer"], 0)
        status, data, _ = self.client.post("/api/model/window/step", {"steps": 2})
        self.assertEqual((status, data["step"]["sizes"], data["window"]["size"], data["stats"]["dynamic_window"]), (200, [16, 8], 4, 4))
        self.assertEqual(data["step"]["nodes_after"], data["stats"]["nodes"])
        self.assertGreater(data["step"]["splits"], 0)
        status, data, _ = self.client.post("/api/model/window", {"top": 16, "floor": 8, "auto": False})
        self.assertEqual((status, data["window"]["sizes"], data["window"]["size"], data["window"]["auto"]), (200, [16, 8], 8, False))
        for body in ({"top": 12}, {"size": 3}, {"floor": 64}, {"on": "yes"}, {"steps": 0}):
            with self.subTest(body=body):
                path = "/api/model/window/step" if "steps" in body else "/api/model/window"
                status, data, _ = self.client.post(path, body)
                self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/model/window", {"on": False})
        self.assertEqual((status, data["window"]["on"], data["stats"]["dynamic_window"]), (200, False, None))
        self.assertIsNone(self.client.get("/api/status")[1]["dynamic_window"])


class TestCli(unittest.TestCase):
    def test_show_set_step_and_off(self):
        from radixnet.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            trained(RadixNet).save(path)

            def run(*argv):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(["--model", path, "--json", *argv])
                return code, json.loads(out.getvalue()) if out.getvalue().strip() else None

            code, doc = run("window")
            self.assertEqual((code, doc["window"]["on"], doc["saved"], doc["step"]), (0, False, None, None))
            code, doc = run("window", "--on", "--dry-run")
            self.assertEqual((code, doc["window"]["size"], doc["saved"]), (0, 32, None))
            self.assertFalse(load_model(path).graph.dynamic_window.on)  # a dry run saves nothing
            code, doc = run("window", "--top", "16", "--floor", "8", "--manual")
            self.assertEqual((code, doc["window"]["sizes"], doc["window"]["auto"], doc["changed"]), (0, [16, 8], False, {"top": 16, "floor": 8, "size": 16, "auto": False}))
            self.assertEqual(load_model(path).graph.dynamic_window, DynamicWindow(16, 8, 16, False))
            code, doc = run("window", "--step")
            self.assertEqual((code, doc["step"]["sizes"], doc["window"]["size"]), (0, [16], 8))
            self.assertGreater(doc["step"]["splits"], 0)
            self.assertEqual(load_model(path).graph.dynamic_window.size, 8)
            code, doc = run("window", "--step", "2", "--out", os.path.join(tmp, "stepped.json"))
            self.assertEqual((code, doc["step"]["sizes"], doc["saved"]["path"]), (0, [8, 16], os.path.join(tmp, "stepped.json")))
            self.assertEqual(load_model(path).graph.dynamic_window.size, 8)  # --out left the model alone
            code, doc = run("info")
            self.assertEqual(doc["stats"]["dynamic_window"], 8)
            for argv in (("window", "--off", "--top", "16"), ("window", "--off", "--step"), ("window", "--auto", "--manual"),
                         ("window", "--top", "20"), ("window", "--size", "4")):
                with self.subTest(argv=argv):
                    code, doc = run(*argv)
                    self.assertNotEqual(code, 0)
            code, doc = run("window", "--off")
            self.assertEqual((code, doc["window"]["on"], doc["changed"]), (0, False, {"on": False}))
            with open(path, encoding="utf-8") as fh:
                self.assertNotIn("dynamic_window", json.load(fh)["graph"])
            code, doc = run("window", "--step")
            self.assertNotEqual(code, 0)  # off: nothing to step


if __name__ == "__main__":
    unittest.main()
