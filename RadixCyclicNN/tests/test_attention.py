"""Tests for radixnet.attention: the attention band, and where a correction's blame and credit land through it.

The band is the rule of ``../SPEC-AttentionBand.md``: each unit a correction
marks hands out one charge, shared among the grams that see it by how
centrally each sees it.  Pinned here: the shape of the band, the three
invariants the rule rests on (one charge per marked unit, a whole-text
judgement left alone, a unit seen by one gram charged to it in full), that
switching it off is the writer rule to the bit, and what it does to the count
model's and the negative network's corrections.
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

from radixnet import diff  # noqa: E402
from radixnet.attention import (  # noqa: E402
    DEFAULT_BLUR,
    AttentionBand,
    band_weights,
    judged_units,
    spread,
    writer_marks,
)
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.encoding import parse_encoding  # noqa: E402
from radixnet.graph import END  # noqa: E402
from radixnet.model import RadixNet, load_model  # noqa: E402
from radixnet.negative import NegativeNet  # noqa: E402
from radixnet.resonance import ResonantNet  # noqa: E402

TEXTS = [
    "the cat sat on the mat",
    "the cat ran to the door",
    "the dog sat on the log",
    "the dog ate the bone",
    "the bat sat on the mat",
]
ENCODINGS = ["char:3:1", "char:5:1", "char:4:4", "char:6:3", "char:2:1", "word:1:1", "word:2:1", "word:3:1"]


def flat(model, **options):
    """A count model trained without compression: every node is one gram, so every step is one gram's charge."""
    model.train(TEXTS, epochs=1, auto_compress=False, **options)
    return model


def moved(before, after):
    """``{edge: delta}`` of the rewards that changed."""
    out = {}
    for e, value in enumerate(after):
        delta = value - (before[e] if e < len(before) else 0.0)
        if abs(delta) > 1e-12:
            out[e] = delta
    return out


def strip_times(doc):
    """A model document without what a clock writes into it."""
    doc = json.loads(json.dumps(doc))
    doc.pop("saved_at", None)
    doc.get("meta", {}).pop("created", None)
    for record in doc.get("history", []):
        record.pop("seconds", None)
    for entry in doc.get("log", []):
        entry.pop("at", None)
    return doc


class TestBand(unittest.TestCase):
    def test_the_band_peaks_at_the_centre_and_falls_to_one_minus_blur_at_the_ends(self):
        self.assertEqual(band_weights(3, 0.5), [0.5, 1.0, 0.5])
        self.assertEqual(band_weights(5, 0.5), [0.5, 0.75, 1.0, 0.75, 0.5])
        for n in (3, 4, 5, 6, 7, 9):
            for blur in (0.0, 0.2, 0.5, 1.0):
                with self.subTest(n=n, blur=blur):
                    weights = band_weights(n, blur)
                    self.assertEqual(len(weights), n)
                    self.assertEqual(weights, weights[::-1])  # symmetric: left and right blur alike
                    self.assertAlmostEqual(weights[0], 1.0 - blur)
                    middle = weights[(n - 1) // 2 : n // 2 + 1]
                    self.assertTrue(all(w == max(weights) for w in middle))  # the centre is seen sharpest
                    if n % 2:
                        self.assertEqual(max(weights), 1.0)
                    for left, right in zip(weights[: n // 2], weights[1 : n // 2 + 1]):
                        self.assertLessEqual(left, right)  # never sharper further from the centre

    def test_a_gram_without_a_centre_is_seen_evenly(self):
        self.assertEqual(band_weights(1, 1.0), [1.0])
        self.assertEqual(band_weights(2, 0.5), [0.5, 0.5])  # both units are ends
        self.assertEqual(band_weights(5, 0.0), [1.0] * 5)  # no blur: no focus

    def test_the_blur_is_a_number_in_zero_to_one(self):
        for bad in (-0.1, 1.5, math.nan, math.inf, "sharp"):
            with self.subTest(blur=bad), self.assertRaises(ValueError):
                AttentionBand(bad)
        self.assertFalse(AttentionBand().on)
        self.assertTrue(AttentionBand(0).on)  # a flat band is a band: every gram that sees a unit shares it
        self.assertEqual(AttentionBand("0.25").blur, 0.25)

    def test_the_file_block_is_written_only_while_the_band_is_on(self):
        self.assertIsNone(AttentionBand().to_dict())
        self.assertEqual(AttentionBand(0.3).to_dict(), {"blur": 0.3})
        self.assertEqual(AttentionBand.from_dict({"blur": 0.3}), AttentionBand(0.3))
        self.assertEqual(AttentionBand.from_dict(None), AttentionBand())
        self.assertEqual(AttentionBand.from_dict({}), AttentionBand())
        with self.assertRaises(ValueError):
            AttentionBand.from_dict({"blur": 2})


class TestSpread(unittest.TestCase):
    def shares(self, spec, wrong, right, blur=0.5):
        enc = parse_encoding(spec)
        spans, _ = diff.changed_spans(wrong, right, enc)
        grams = enc.encode(wrong)
        return enc, grams, spread(enc.n, enc.stride, len(grams), enc.length(wrong), spans, band_weights(enc.n, blur))

    def test_a_changed_unit_is_charged_most_to_the_gram_that_has_it_at_its_centre(self):
        _enc, grams, out = self.shares("char:3:1", "the cat sat", "the bat sat")
        charged = {grams[g]: share for g, share in enumerate(out.shares) if share}
        self.assertEqual(charged, {"e c": 0.25, " ca": 0.5, "cat": 0.25})
        self.assertEqual([grams[g] for g, f in enumerate(out.focus) if f], [" ca"])
        # the writer rule charged the gram that only just reached the change, at its edge
        writes, _ = writer_marks(3, 1, len(grams), 11, [(4, 5)])
        self.assertEqual([grams[g] for g, w in enumerate(writes) if w], ["e c"])

    def test_every_marked_unit_hands_out_exactly_one_charge(self):
        for spec in ENCODINGS:
            enc = parse_encoding(spec)
            text = "the quick brown fox jumps over the lazy dog again and again"
            length = enc.length(text)
            grams = enc.encode(text)
            covered = enc.covered(length)
            for blur in (0.0, 0.3, 0.5, 1.0):
                for unit in range(covered):
                    with self.subTest(spec=spec, blur=blur, unit=unit):
                        out = spread(enc.n, enc.stride, len(grams), length, [(unit, unit + 1)], band_weights(enc.n, blur))
                        self.assertAlmostEqual(sum(out.shares), 1.0, places=12)
                        self.assertTrue(any(out.focus))

    def test_a_whole_text_judgement_leaves_every_gram_at_a_full_charge(self):
        # thumbs up, thumbs down and a sentence replaced outright mark every unit: no band tells the grams apart
        for spec in ENCODINGS:
            enc = parse_encoding(spec)
            text = "the quick brown fox jumps over the lazy dog"
            grams = enc.encode(text)
            length = enc.length(text)
            for blur in (0.0, 0.3, 0.5, 1.0):
                with self.subTest(spec=spec, blur=blur):
                    out = spread(enc.n, enc.stride, len(grams), length, [(0, length)], band_weights(enc.n, blur))
                    self.assertTrue(all(share >= 1.0 - 1e-12 for share in out.shares), out.shares)
                    self.assertTrue(all(out.focus))

    def test_a_unit_only_one_gram_sees_is_charged_to_it_in_full(self):
        # groups share nothing: each letter sits in one group, which takes the whole charge, as its writer did
        _enc, grams, out = self.shares("char:4:4", "abcdefghijkl", "abcdXfghijkl")
        self.assertEqual(out.shares, [0.0, 1.0, 0.0])
        # the first and the last unit of a text are seen by one gram each, however sharp the band
        for blur in (0.0, 0.5, 1.0):
            _enc, grams, first = self.shares("char:3:1", "xhe cat sat", "the cat sat", blur)
            self.assertEqual(first.shares[0], 1.0)
            self.assertEqual(sum(first.shares), 1.0)
            _enc, grams, last = self.shares("char:3:1", "the cat sax", "the cat sat", blur)
            self.assertEqual(last.shares[-1], 1.0)

    def test_blur_zero_shares_evenly_and_blur_one_leaves_the_ends_nothing(self):
        _enc, _grams, even = self.shares("char:3:1", "the cat sat", "the bat sat", blur=0.0)
        self.assertEqual([round(s, 12) for s in even.shares if s], [round(1 / 3, 12)] * 3)
        self.assertTrue(sum(even.focus) == 3)  # nobody sees it more sharply than anyone else
        _enc, _grams, sharp = self.shares("char:3:1", "the cat sat", "the bat sat", blur=1.0)
        self.assertEqual([s for s in sharp.shares if s], [1.0])
        # a unit only the ends of grams reach is split evenly between them rather than lost: here the
        # first letter, which one gram sees, at an edge the band gives nothing
        out = spread(3, 1, 3, 5, [(0, 1)], band_weights(3, 1.0))
        self.assertEqual(out.shares, [1.0, 0.0, 0.0])
        self.assertTrue(out.focus[0])
        # and a gram of two, which is all ends: each of the two grams that see a letter takes half
        out = spread(2, 1, 4, 5, [(2, 3)], band_weights(2, 1.0))
        self.assertEqual(out.shares, [0.0, 0.5, 0.5, 0.0])

    def test_the_position_after_the_last_unit_belongs_to_the_end_step(self):
        _enc, grams, out = self.shares("char:3:1", "the cat sit", "the cat sits")
        self.assertTrue(out.end)
        self.assertEqual(sum(out.shares), 0.0)  # no gram covers it
        self.assertEqual(judged_units([(11, 11)], 11), [11])
        self.assertEqual(judged_units([(2, 2), (4, 6)], 11), [2, 4, 5])

    def test_a_unit_no_gram_covers_is_charged_to_no_one(self):
        # a grouping encoding drops the tail that fills no group; a change there moves nothing, as before
        _enc, grams, out = self.shares("char:4:4", "abcdefghij", "abcdefghiX")
        self.assertEqual(out.shares, [0.0, 0.0])
        self.assertFalse(out.end)


class TestChargedSteps(unittest.TestCase):
    def test_off_is_the_writer_rule(self):
        for spec in ENCODINGS:
            model = CountRewardNet(seed=2, encoding=parse_encoding(spec))
            model.train(TEXTS, epochs=2)
            enc = model.encoding
            for wrong, right in [("the cat sat on the mat", "the bat sat on the mat"), ("the dog sat", "the dog sat on the log")]:
                grams = enc.encode(wrong)
                if not grams:
                    continue
                model._observe_grams([grams], False)
                spans, _ = diff.changed_spans(wrong, right, enc)
                with self.subTest(spec=spec, wrong=wrong):
                    old = model._steps_over(grams, enc.length(wrong), spans)
                    self.assertEqual(model._charged_steps(grams, enc.length(wrong), spans), [(p, e, 1.0, True) for p, e in old])

    def test_a_step_is_charged_what_its_grams_collected_capped_at_one(self):
        model = CountRewardNet(seed=2)
        model.train(TEXTS, epochs=2)  # compressed: a node holds a chain of grams
        model.configure_attention(blur=0.5)
        enc = model.encoding
        wrong, right = "the cat sat on the mat", "the bat sat on the mat"
        grams = enc.encode(wrong)
        spans, _ = diff.changed_spans(wrong, right, enc)
        steps = model._charged_steps(grams, enc.length(wrong), spans)
        shared = spread(3, 1, len(grams), len(wrong), spans, band_weights(3, 0.5))
        path = model.graph.node_path(grams)
        self.assertTrue(any(model.graph.label_len(n) > 3 for n in path[1:-1]))  # something really was merged
        total = sum(charge for _p, _e, charge, _f in steps)
        self.assertAlmostEqual(total, min(1.0, sum(shared.shares)))  # one unit, one charge
        self.assertTrue(all(0.0 < charge <= 1.0 for _p, _e, charge, _f in steps))
        self.assertEqual(sum(1 for step in steps if step[3]), 1)  # one focus: the node holding " ca"

    def test_a_whole_text_judgement_charges_every_step_in_full_whatever_the_band(self):
        for spec in ENCODINGS:
            model = CountRewardNet(seed=2, encoding=parse_encoding(spec))
            model.train(TEXTS, epochs=2)
            enc = model.encoding
            text = "the cat sat on the mat"
            grams = enc.encode(text)
            length = enc.length(text)
            off = model._charged_steps(grams, length, [(0, length)])
            for blur in (0.0, 0.5, 1.0):
                model.configure_attention(blur=blur)
                with self.subTest(spec=spec, blur=blur):
                    self.assertEqual(model._charged_steps(grams, length, [(0, length)]), off)
            model.configure_attention(on=False)


class TestCorrections(unittest.TestCase):
    def test_a_correction_off_the_band_moves_what_it_always_moved(self):
        # two identical models, one with the band switched on and back off: the same numbers, to the bit
        docs = []
        for toggle in (False, True):
            model = CountRewardNet(seed=4)
            model.train(TEXTS, epochs=2)
            if toggle:
                model.configure_attention(blur=0.7)
                model.configure_attention(on=False)
            model.correct("the cat sit on the mat", "the cat sits on the mat", keep=0.25)
            model.correct("a dog ate the bone", "the dog ate the bone")
            docs.append(strip_times(model.to_dict()))
        self.assertEqual(docs[0], docs[1])
        self.assertNotIn("attention", docs[0]["graph"])

    def test_a_sentence_replaced_outright_is_taught_the_same_with_the_band_on(self):
        docs = []
        for blur in (None, 0.5):
            model = CountRewardNet(seed=4)
            model.train(TEXTS, epochs=2)
            if blur is not None:
                model.configure_attention(blur=blur)
            model.correct("xyzzy plugh", "the cat sat on the mat", keep=0.25)
            doc = strip_times(model.to_dict())
            doc["graph"].pop("attention", None)
            docs.append(doc)
        self.assertEqual(docs[0], docs[1])

    def test_the_centre_takes_most_of_the_penalty_and_all_of_the_verdict(self):
        model = flat(CountRewardNet(seed=1))
        model.configure_attention(blur=0.5)
        before = list(model.graph.edge_reward)
        out = model.correct("the cat sat on the mat", "the bat sat on the mat", strength=1.0, weight=2.0, keep=0.0)
        graph = model.graph
        penalties = {graph.text_of(graph.labels[graph_child(graph, e)]): delta for e, delta in moved(before, graph.edge_reward).items() if delta < 0}
        self.assertEqual(penalties, {"e c": -0.5, " ca": -1.0, "cat": -0.5})  # weight 2 x the shares
        self.assertEqual(out["penalised"], 3)
        self.assertAlmostEqual(out["penalty"], 2.0)  # one changed letter: one charge, weighted
        self.assertEqual(out["marked_incorrect"], 1)  # the verdict is a count: the centre's
        rewards = {graph.text_of(graph.labels[graph_child(graph, e)]): delta for e, delta in moved(before, graph.edge_reward).items() if delta > 0}
        self.assertEqual(rewards, {"e b": 0.25, " ba": 0.5, "bat": 0.25})
        self.assertEqual(out["marked_correct"], 1)

    def test_keep_tops_up_a_partly_charged_step(self):
        model = flat(CountRewardNet(seed=1))
        model.configure_attention(blur=0.5)
        before = list(model.graph.edge_reward)
        model.correct("the cat sat on the mat", "the bat sat on the mat", keep=0.5)
        graph = model.graph
        rewards = {graph.text_of(graph.labels[graph_child(graph, e)]): delta for e, delta in moved(before, graph.edge_reward).items() if delta > 0}
        self.assertEqual(rewards["e b"], 0.625)  # 0.25 of the fix + 0.75 x keep
        self.assertEqual(rewards[" ba"], 0.75)
        self.assertEqual(rewards["the"], 0.5)  # untouched by the change: keep alone

    def test_a_word_model_blames_the_end_for_stopping_early(self):
        # the lengths are in words, as the spans are: a word model's sentence that stopped too early is
        # an insertion at its word count, and the step into END answers for it (as in Go and Rust)
        for blur in (None, 0.5):
            model = CountRewardNet(seed=1, encoding=parse_encoding("word:2:1"))
            model.train(TEXTS, epochs=2)
            if blur is not None:
                model.configure_attention(blur=blur)
            grams = model.encoder.encode("the cat sat")
            model._observe_grams([grams], False)
            path = model.graph.node_path(grams)
            end_edge = model.graph.children[path[-2]][END]
            before = model.graph.edge_reward[end_edge]
            out = model.correct("the cat sat", "the cat sat on the mat")
            with self.subTest(blur=blur):
                self.assertEqual(out["penalised"], 1)
                self.assertEqual(model.graph.edge_reward[end_edge] - before, -1.0)

    def test_the_negative_network_is_blamed_by_share(self):
        model = NegativeNet(seed=1)
        model.configure_attention(blur=0.5)
        out = model.correct("the cat sat on the mat", "the bat sat on the mat", reason="spelling", severity=2.0)
        graph = model.graph
        blamed = sorted(graph.edge_blame[e] for e in range(len(graph.edge_blame)) if graph.edge_blame[e] > 0)
        self.assertEqual(out["blamed"], len(blamed))
        self.assertAlmostEqual(sum(blamed), 2.0)  # one changed letter, one charge, at the severity
        self.assertAlmostEqual(model.meta["blame_total"], 2.0)
        self.assertAlmostEqual(graph.total_blame, 2.0)
        self.assertEqual(max(blamed), 1.0)  # the centre's half
        self.assertEqual(sum(graph.edge_fails), out["blamed"])  # every blamed step counts one failure

    def test_the_negative_network_off_the_band_is_unchanged(self):
        docs = []
        for toggle in (False, True):
            model = NegativeNet(seed=6)
            if toggle:
                model.configure_attention(blur=0.2)
                model.configure_attention(on=False)
            model.correct("the cat sit on the mat", "the cat sits on the mat", severity=1.5)
            model.correct("birds lays eggs", "birds lay eggs", reason="grammar")
            docs.append(strip_times(model.to_dict()))
        self.assertEqual(docs[0], docs[1])


def graph_child(graph, edge):
    """The node an edge enters."""
    for node, children in enumerate(graph.children):
        for child, e in children.items():
            if e == edge:
                return child
    raise KeyError(edge)


class TestSetting(unittest.TestCase):
    def test_on_off_and_the_blur(self):
        model = CountRewardNet(seed=1)
        self.assertFalse(model.attention_config()["on"])
        config = model.configure_attention(on=True)
        self.assertEqual((config["on"], config["blur"], config["weights"]), (True, DEFAULT_BLUR, [0.5, 1.0, 0.5]))
        self.assertEqual(model.configure_attention(blur=0.2)["blur"], 0.2)
        self.assertEqual(model.configure_attention(on=True)["blur"], 0.2)  # switching on again keeps the blur
        self.assertEqual(model.configure_attention()["blur"], 0.2)  # nothing asked, nothing changed
        config = model.configure_attention(on=False, blur=0.9)
        self.assertEqual((config["on"], config["blur"], config["weights"]), (False, None, None))
        self.assertEqual(model.stats()["attention_blur"], None)
        with self.assertRaises(ValueError):
            model.configure_attention(blur=1.2)

    def test_it_travels_with_the_file(self):
        for kind in (CountRewardNet, NegativeNet):
            model = kind(seed=1, encoding=parse_encoding("word:3:1"))
            model.configure_attention(blur=0.3)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.json")
                model.save(path)
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                self.assertEqual(doc["graph"]["attention"], {"blur": 0.3})
                keys = list(doc["graph"])
                self.assertEqual(keys[:4], ["format", "format_version", "encoding", "attention"])  # beside the encoding
                again = load_model(path)
            with self.subTest(kind=kind.kind):
                self.assertEqual(again.graph.attention, AttentionBand(0.3))
                self.assertEqual(again.stats()["attention_blur"], 0.3)
                self.assertEqual(again.attention_config()["weights"], [0.7, 1.0, 0.7])

    def test_a_kind_that_is_never_corrected_refuses_a_band(self):
        for model in (RadixNet(seed=1), ResonantNet(seed=1)):
            with self.subTest(kind=model.kind):
                self.assertFalse(model.attention_config()["applies"])
                with self.assertRaises(ValueError):
                    model.configure_attention(blur=0.5)

    def test_a_preview_shows_both_rules_and_changes_nothing(self):
        model = CountRewardNet(seed=1)
        model.train(TEXTS, epochs=1)
        before = json.dumps(strip_times(model.to_dict()), sort_keys=True)
        preview = model.attention_preview("the cat sat", "the bat sat")
        self.assertEqual(preview["blur"], DEFAULT_BLUR)  # off: the band it would be switched on at
        wrong = preview["wrong"]
        self.assertEqual(wrong["grams"][2:5], ["e c", " ca", "cat"])
        self.assertEqual(wrong["charges"][2:5], [0.25, 0.5, 0.25])
        self.assertEqual([g for g, w in zip(wrong["grams"], wrong["writer"]) if w], ["e c"])
        self.assertEqual([g for g, f in zip(wrong["grams"], wrong["focus"]) if f], [" ca"])
        self.assertFalse(preview["attention"]["on"])
        self.assertEqual(model.attention_preview("the cat sat", "the bat sat", blur=1.0)["wrong"]["charges"][2:5], [0.0, 1.0, 0.0])
        model.configure_attention(blur=0.0)
        self.assertEqual(model.attention_preview("the cat sat", "the bat sat")["blur"], 0.0)
        model.configure_attention(on=False)
        self.assertEqual(json.dumps(strip_times(model.to_dict()), sort_keys=True), before)


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-attention-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(
            cls.addClassCleanup, model_path=os.path.join(cls.tmp.name, "model.json")
        )

    def select(self, kind):
        status, data, _ = self.client.post("/api/model/select", {"kind": kind})
        self.assertEqual(status, 200, data)

    def test_the_band_through_the_api(self):
        self.select("radix")
        status, data, _ = self.client.get("/api/model/attention")
        self.assertEqual((status, data["kind"], data["attention"]["applies"]), (200, "radix", False))
        status, data, _ = self.client.post("/api/model/attention", {"blur": 0.5})
        self.assertEqual(status, 400)
        self.assertIn("never corrected", data["error"])
        self.select("count")
        status, data, _ = self.client.get("/api/model/attention")
        self.assertEqual(data["attention"], {
            "on": False, "blur": None, "weights": None, "ngram": 3, "stride": 1, "unit": "char", "units": "chars",
            "applies": True, "default_blur": DEFAULT_BLUR,
        })
        status, data, _ = self.client.post("/api/model/attention", {"on": True})
        self.assertEqual((status, data["attention"]["blur"], data["stats"]["attention_blur"]), (200, 0.5, 0.5))
        status, data, _ = self.client.post("/api/model/attention", {"blur": 0.2})
        self.assertEqual(data["attention"]["weights"], [0.8, 1.0, 0.8])
        self.assertEqual(self.client.get("/api/status")[1]["attention_blur"], 0.2)
        self.assertEqual(self.client.get("/api/model")[1]["attention"]["blur"], 0.2)
        status, data, _ = self.client.post("/api/model/attention", {"blur": 3})
        self.assertEqual(status, 400)
        self.assertIn("[0, 1]", data["error"])
        status, data, _ = self.client.post("/api/model/attention", {"on": "yes"})
        self.assertEqual(status, 400)
        status, data, _ = self.client.post(
            "/api/model/attention/preview", {"wrong": "the cat sat", "right": "the bat sat", "blur": 1.0}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["blur"], data["weights"]), (1.0, [0.0, 1.0, 0.0]))
        self.assertEqual(data["wrong"]["charges"][2:5], [0.0, 1.0, 0.0])
        self.assertEqual(data["changes"], [{"op": "replace", "wrong": "c", "right": "b"}])
        self.assertEqual(self.client.get("/api/model/attention")[1]["attention"]["blur"], 0.2)  # a preview changes nothing
        status, data, _ = self.client.post("/api/model/attention", {"on": False, "blur": 0.9})
        self.assertEqual((status, data["attention"]["on"], data["attention"]["blur"]), (200, False, None))


class TestCli(unittest.TestCase):
    def test_show_switch_and_preview(self):
        from radixnet.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.count.json")
            model = CountRewardNet(seed=1)
            model.train(TEXTS, epochs=1)
            model.save(path)

            def run(*argv):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(["--model", path, "--json", *argv])
                return code, json.loads(out.getvalue()) if out.getvalue().strip() else None

            code, doc = run("attention")
            self.assertEqual((code, doc["attention"]["on"], doc["saved"]), (0, False, None))
            code, doc = run("attention", "--blur", "0.25", "--dry-run")
            self.assertEqual((code, doc["attention"]["blur"], doc["saved"]), (0, 0.25, None))
            self.assertIsNone(load_model(path).graph.attention.blur)  # a dry run saves nothing
            code, doc = run("attention", "--on", "--wrong", "the cat sat", "--right", "the bat sat")
            self.assertEqual((code, doc["attention"]["blur"]), (0, DEFAULT_BLUR))
            self.assertEqual(doc["preview"]["wrong"]["charges"][2:5], [0.25, 0.5, 0.25])
            self.assertEqual(load_model(path).graph.attention, AttentionBand(DEFAULT_BLUR))
            code, doc = run("attention", "--off", "--blur", "0.5")
            self.assertNotEqual(code, 0)
            code, doc = run("attention", "--off")
            self.assertEqual((code, doc["attention"]["on"]), (0, False))
            with open(path, encoding="utf-8") as fh:
                self.assertNotIn("attention", json.load(fh)["graph"])


if __name__ == "__main__":
    unittest.main()
