"""Tests for radixnet.model (RadixNet, TrainConfig)."""

import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.encoding import Encoder  # noqa: E402
from radixnet.graph import END, START  # noqa: E402
from radixnet.model import UNKNOWN_PROB, RadixNet, TrainConfig  # noqa: E402
from radixnet.search import PathResult  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ENC = Encoder()


def read_lines(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


CORPUS = read_lines("sample_corpus.txt")
GARBAGE = read_lines("sample_garbage.txt")

# The learning rule averages gradients over the batch, so an edge seen ``c`` times in a
# batch of ``B`` moves by ``lr * c / B``: the tests use batch_size=1 with lr=1.0 to learn
# visibly in a few epochs (the spec defaults lr=0.05 / batch 256 need hundreds of epochs).
FAST = {"lr": 1.0, "batch_size": 1}


def make_model(seed=0):
    return RadixNet(seed=seed, backend="python")


def corpus_model(seed=0, epochs=10):
    model = make_model(seed)
    model.train(CORPUS, epochs=epochs, **FAST)
    return model


def branch_prob(model, from_tri, to_tri):
    """P(node of to_tri | node of from_tri) over the graph's learned softmax."""
    g = model.graph
    p, po = g.lookup(from_tri)
    c, co = g.lookup(to_tri)
    assert co == 0 and po + 3 == len(g.labels[p]), "not a branching transition"
    return dict(g.child_probs(p))[c]


def snapshot(model, prefixes=("the quick brown", "the cat", "knowledge", "xq", "")):
    """Deterministic prediction fingerprint (texts and costs)."""
    out = []
    for prefix in prefixes:
        r = model.predict(prefix, length=15)
        out.append((prefix, r.text, r.full_text, round(r.cost, 9)))
        r = model.predict(prefix, length=15, to_end=True)
        out.append((prefix, r.text, round(r.cost, 9), r.reached_end))
    out.append([r.text for r in model.generate(max_length=30, count=3, seed=7)])
    out.append(model.generate(mode="dijkstra", max_length=40)[0].text)
    return out


class TestTrainConfig(unittest.TestCase):
    def test_defaults_match_spec(self):
        cfg = TrainConfig()
        self.assertEqual(
            cfg.to_dict(),
            {
                "epochs": 5, "lr": 0.05, "act_lr": 0.005, "batch_size": 256, "clip": 5.0,
                "auto_compress": True, "shuffle": True, "checkpoint_every": 0, "verbose": False,
            },
        )
        cfg.validate()

    def test_validation(self):
        for bad in (
            {"epochs": -1}, {"batch_size": 0}, {"lr": -0.1}, {"lr": float("nan")},
            {"act_lr": float("inf")}, {"clip": 0.0}, {"checkpoint_every": -2},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TrainConfig(**bad).validate()

    def test_overrides(self):
        model = make_model()
        with self.assertRaises(TypeError):
            model.train(["hello world"], epochs=1, learning_rate=0.1)
        with self.assertRaises(ValueError):
            model.train(["hello world"], batch_size=0)
        records = model.train(["hello world"], TrainConfig(epochs=3), epochs=2, shuffle=False)
        self.assertEqual(len(records), 2)


class TestConstruction(unittest.TestCase):
    def test_fresh_model(self):
        model = make_model(seed=5)
        self.assertEqual(model.backend.name, "python")
        self.assertEqual(model.backend.device, "cpu")
        self.assertEqual(model.graph.seed, 5)
        self.assertEqual(model.history, [])
        for key in ("created", "seed", "epochs_total", "trained_chars", "trained_texts", "twonrl_runs"):
            self.assertIn(key, model.meta)
        stats = model.stats()
        self.assertEqual(
            set(stats),
            {
                "nodes", "edges", "trigrams", "compression_ratio", "inverted", "backend", "device",
                "epochs_total", "trained_chars", "trained_texts", "twonrl_runs", "history_len", "last_loss",
            },
        )
        self.assertEqual((stats["nodes"], stats["edges"], stats["trigrams"]), (2, 0, 0))
        self.assertIsNone(stats["last_loss"])
        json.dumps(stats)
        self.assertIn("RadixNet(", repr(model))

    def test_auto_backend_resolves(self):
        model = RadixNet(seed=0)  # backend="auto"
        self.assertIn(model.backend.name, ("python", "torch"))
        model.train("hello world", epochs=1)
        self.assertEqual(model.stats()["epochs_total"], 1)


class TestTrain(unittest.TestCase):
    def test_records_history_and_meta(self):
        model = make_model()
        records = model.train(["hello world", "hi", "", "hello there"], epochs=2, **FAST)
        self.assertEqual(len(records), 2)
        self.assertEqual(model.history, records)
        keys = {
            "epoch", "loss", "perplexity", "nodes", "edges", "trigrams", "compression_ratio",
            "merges", "transitions", "seconds", "skipped_short",
        }
        for i, rec in enumerate(records, 1):
            self.assertEqual(set(rec), keys)
            self.assertEqual(rec["epoch"], i)
            self.assertEqual(rec["skipped_short"], 2)
            self.assertAlmostEqual(rec["perplexity"], math.exp(rec["loss"]), places=12)
            self.assertGreaterEqual(rec["seconds"], 0.0)
            json.dumps(rec)
        self.assertEqual(model.meta["epochs_total"], 2)
        self.assertEqual(model.meta["trained_texts"], 2)
        self.assertEqual(model.meta["trained_chars"], len("hello world") + len("hello there"))
        self.assertEqual(model.stats()["last_loss"], records[-1]["loss"])
        # a plain string is one text; epochs=0 only registers structure
        more = model.train("goodbye world", epochs=0)
        self.assertEqual(more, [])
        self.assertIsNotNone(model.graph.lookup("goo"))
        self.assertEqual(model.meta["trained_texts"], 3)
        with self.assertRaises(TypeError):
            model.train([b"bytes"], epochs=1)

    def test_loss_and_perplexity_decrease(self):
        model = make_model()
        records = model.train(CORPUS, epochs=10, **FAST)
        losses = [r["loss"] for r in records]
        self.assertLess(losses[-1], losses[0] * 0.95)
        self.assertTrue(all(b < a for a, b in zip(losses, losses[1:])), losses)
        self.assertLess(records[-1]["perplexity"], records[0]["perplexity"])
        # every epoch trains on the compressed structure: the transition count is stable
        self.assertGreater(records[0]["merges"], 0)
        self.assertEqual(len({r["transitions"] for r in records}), 1)
        self.assertGreater(records[0]["compression_ratio"], 1.5)

    def test_learns_frequencies(self):
        texts = ["hello world"] * 3 + ["hello there"]
        for seed in range(4):
            with self.subTest(seed=seed):
                model = make_model(seed)
                model.train(texts, epochs=0)
                before = branch_prob(model, "lo ", "o w")
                model.train(texts, epochs=10, **FAST)
                after = branch_prob(model, "lo ", "o w")
                self.assertGreater(after, before)
                self.assertGreater(after, 0.6)
                self.assertEqual(model.predict("hello", length=6).text, " world")
                self.assertEqual(model.predict("hello", length=6, mode="sample", temperature=0).text, " world")

    def test_compressed_transitions_emit_nothing(self):
        model = make_model()
        (rec,) = model.train(["hello world"], epochs=1, **FAST)
        # START -> "hello world" -> END: two transitions, all in-node steps are silent
        self.assertEqual(rec["transitions"], 2)
        self.assertEqual(rec["nodes"], 3)
        self.assertEqual(rec["loss"], 0.0)

    def test_reobserves_after_structure_change(self):
        model = make_model()
        model.train(["hello world"] * 2, epochs=1, **FAST)
        seen = []

        def progress(rec):
            seen.append(rec["epoch"])
            if len(seen) == 1:
                # split the compressed node between epochs: edge ids move
                model.graph.observe_sequence(ENC.encode("hello there"))
                model.graph.compress()

        records = model.train(["hello world"] * 2, epochs=3, progress=progress, **FAST)
        self.assertEqual(seen, [2, 3, 4])
        self.assertEqual(records[0]["transitions"], 4)
        self.assertEqual(records[-1]["transitions"], 6)
        model.graph.check_invariants(texts=["hello world", "hello there"], compressed=True)

    def test_progress_stop_event_and_verbose(self):
        model = make_model()
        stop = threading.Event()
        seen = []

        def progress(rec):
            seen.append(rec["epoch"])
            if rec["epoch"] == 2:
                stop.set()

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            records = model.train(["hello world"], epochs=10, progress=progress, stop_event=stop, verbose=True)
        self.assertEqual(len(records), 2)
        self.assertEqual(seen, [1, 2])
        self.assertEqual(model.meta["epochs_total"], 2)
        self.assertEqual(err.getvalue().count("epoch"), 2)
        self.assertIn("loss=", err.getvalue())

    def test_no_compress_option(self):
        model = make_model()
        records = model.train(CORPUS, epochs=2, auto_compress=False, **FAST)
        self.assertEqual([r["merges"] for r in records], [0, 0])
        self.assertEqual(records[0]["compression_ratio"], 1.0)


class TestLocate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model()
        cls.model.train(["hello world", "hello there", "help me"], epochs=2, **FAST)

    def test_exact_and_fallbacks(self):
        g = self.model.graph
        node, off = self.model.locate("say hello wor")
        self.assertEqual(g.labels[node][off : off + 3], "wor")
        node, off = self.model.locate("hel")
        self.assertEqual((g.labels[node][off : off + 3], off), ("hel", 0))
        # unknown last trigram, two-char fallback: a trigram starting with "lo"
        node, off = self.model.locate("xxlo")
        self.assertTrue(g.labels[node][off : off + 3].startswith("lo"))
        # one-char fallback: the most-visited node holding a trigram starting with "w"
        node, off = self.model.locate("zzzw")
        self.assertTrue(g.labels[node][off : off + 3].startswith("w"))
        # short prefixes match labels; empty / hopeless prefixes fall back to START
        node, off = self.model.locate("he")
        self.assertTrue(g.labels[node].startswith("he"))
        self.assertEqual(off, 0)
        self.assertEqual(self.model.locate(""), (START, 0))
        self.assertEqual(self.model.locate("qqq"), (START, 0))
        self.assertEqual(self.model.locate("q"), (START, 0))

    def test_most_visited_wins(self):
        model = make_model()
        model.train(["abc xyz"] * 3 + ["abd xyz"], epochs=1, **FAST)
        g = model.graph
        node, off = model.locate("ab")
        self.assertTrue(g.labels[node].startswith("abc"))
        node, off = model.locate("zzab")  # two-char fallback prefers the more visited "abc"
        self.assertEqual(g.labels[node][off : off + 3], "abc")


class TestPredict(unittest.TestCase):
    def test_reproduces_training_continuation(self):
        model = make_model()
        model.train(
            ["the quick brown fox jumps over the lazy dog", "a bird in the hand is worth two in the bush", "knowledge is power"],
            epochs=10, **FAST,
        )
        r = model.predict("the quick brown", length=20)
        self.assertEqual(r.text, " fox jumps over the ")
        self.assertEqual(r.full_text, "the quick brown fox jumps over the ")
        self.assertFalse(r.reached_end)
        self.assertLessEqual(len(r.text), 40)
        r = model.predict("the quick brown", length=20, to_end=True)
        self.assertEqual(r.text, " fox jumps over the lazy dog")
        self.assertTrue(r.reached_end)
        self.assertEqual(r.labels[-1], "</s>")
        self.assertEqual(r.node_ids[-1], END)
        self.assertAlmostEqual(r.cost, sum(r.step_costs), places=12)
        self.assertIsInstance(r, PathResult)
        json.dumps(r.to_dict())

    def test_corpus_prediction_is_supported_by_the_corpus(self):
        model = corpus_model()
        r = model.predict("the quick brown", length=20)
        self.assertTrue(r.text.startswith(" f"), r.text)  # deterministic remainder of the compressed node
        self.assertTrue(len(r.text) >= 20 or r.reached_end, r.text)
        self.assertLessEqual(len(r.text), 40)
        corpus_trigrams = {t for line in CORPUS for t in ENC.encode(line)}
        self.assertTrue(all(t in corpus_trigrams for t in ENC.encode(r.full_text)), r.full_text)
        self.assertEqual(model.score(r.full_text)["unknown_transitions"], 0)
        # sampling and the temperature-0 (greedy) walk stay inside the corpus structure too
        for kwargs in ({"mode": "sample"}, {"mode": "sample", "temperature": 0.0}, {"step_penalty": 0.5}):
            with self.subTest(kwargs=kwargs):
                s = model.predict("the quick brown", length=20, **kwargs)
                self.assertTrue(s.text.startswith(" f"))
                self.assertEqual(s.full_text, "the quick brown" + s.text)
                self.assertTrue(all(t in corpus_trigrams for t in ENC.encode(s.full_text)))

    def test_lengths_and_modes(self):
        model = corpus_model(epochs=2)
        r = model.predict("the", length=5)
        self.assertGreaterEqual(len(r.text), 1)
        self.assertLessEqual(len(r.text), 10)
        r = model.predict("the", length=5, max_length=6)
        self.assertLessEqual(len(r.text), 6)
        r = model.predict("the", length=8, mode="sample")
        self.assertLessEqual(len(r.text), 8)
        r = model.predict("the", length=0)
        self.assertEqual(r.full_text, "the" + r.text)
        with self.assertRaises(ValueError):
            model.predict("the", mode="beam")
        with self.assertRaises(ValueError):
            model.predict("the", length=-1)
        with self.assertRaises(TypeError):
            model.predict(None)

    def test_partial_match_prepends_guessed_characters(self):
        model = make_model()
        model.train(["hello world"], epochs=1, **FAST)
        self.assertEqual(model.predict("xxlo", length=6).text, " world")  # "lo " matched on "lo"
        self.assertEqual(model.predict("zzzo", length=6).text, " world")  # "o w" matched on "o"
        self.assertEqual(model.predict("he", length=9).text, "llo world")  # label starts with "he"
        self.assertEqual(model.predict("he", length=9).full_text, "hello world")
        self.assertEqual(model.predict("he", length=2).text, "llo world")  # no cap: the whole cheapest path
        self.assertEqual(model.predict("he", length=2, max_length=2).text, "ll")
        self.assertEqual(model.predict("he", length=2, max_length=4).text, "llo ")
        # nothing matches: the model starts over from START (full text, prefix kept)
        r = model.predict("qqq", length=5)
        self.assertEqual(r.text, "hello world")  # not truncated without max_length
        self.assertEqual(model.predict("qqq", length=5, max_length=10).text, "hello worl")
        r = model.predict("qqq", length=5, max_length=20)
        self.assertEqual(r.text, "hello world")
        self.assertEqual(r.full_text, "qqqhello world")

    def test_continuation_never_exceeds_cap(self):
        model = make_model()
        model.train(["hello world"], epochs=1, **FAST)
        # the guessed remainder of a partially matched trigram (the "lead") counts toward the cap
        self.assertEqual(model.predict("he", length=0).text, "")
        self.assertEqual(model.predict("he", length=1, max_length=1).text, "l")
        self.assertEqual(model.predict("h", length=1, max_length=1).text, "e")
        self.assertEqual(model.predict("h", length=0, mode="sample").text, "")
        self.assertEqual(model.predict("h", length=3, max_length=1, mode="sample").text, "e")
        self.assertEqual(model.predict("h", length=3, max_length=3).text, "ell")
        self.assertEqual(model.predict("he", length=2, max_length=4).text, "llo ")
        # without max_length the shortest path is returned whole (no cap on emitted characters)
        self.assertEqual(model.predict("he", length=2).text, "llo world")
        self.assertEqual(model.predict("h", length=1).text, "ello world")
        for prefix in ("h", "he", "xxlo", "zzzo", "hello", "qqq", ""):
            for length in (0, 1, 2, 5):
                for max_length in (None, 0, 1, 3, 10):
                    for mode in ("dijkstra", "sample"):
                        with self.subTest(prefix=prefix, length=length, max_length=max_length, mode=mode):
                            r = model.predict(prefix, length=length, max_length=max_length, mode=mode)
                            if mode == "dijkstra":
                                # no cap without max_length, except that length 0 still emits nothing
                                cap = (0 if length == 0 else None) if max_length is None else max(length, max_length)
                            else:
                                cap = max_length if max_length is not None else length
                            if cap is None:
                                self.assertLessEqual(len(r.text), len("hello world"))
                            else:
                                self.assertLessEqual(len(r.text), cap)
                            self.assertEqual(r.full_text, prefix + r.text)

    def test_texts_containing_sentinel_labels(self):
        texts = ["x<s>y", "a</s>b", "<s>"]
        model = make_model()
        model.train(texts, epochs=2, **FAST)
        model.graph.check_invariants(texts=texts, compressed=True)
        # the real node labelled "<s>" must be emitted, not mistaken for START
        r = model.predict("x<s", length=1)
        self.assertEqual((r.text, r.full_text, r.labels), (">", "x<s>", ["x<s", "<s>"]))
        self.assertEqual(model.predict("x<s", length=1, mode="sample", temperature=0.0).text, ">")
        self.assertIn(model.predict("<s>", length=0, to_end=True).full_text, ("<s>", "<s>y"))
        self.assertEqual(model.predict("a</", length=3).text, "s>b")
        for t in texts:
            self.assertEqual(model.score(t)["unknown_transitions"], 0, t)
        for r in model.generate(mode="dijkstra") + model.generate(mode="sample", count=4, seed=1):
            self.assertEqual(r.full_text, r.text)
            self.assertEqual(model.score(r.text)["unknown_transitions"], 0, r.text)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sentinel.json.gz")
            model.save(path)
            loaded = RadixNet.load(path, backend="python")
        for prefix in ("x<s", "<s>", "a</", ""):
            self.assertEqual(loaded.predict(prefix, length=2).to_dict(), model.predict(prefix, length=2).to_dict())


class TestGenerate(unittest.TestCase):
    def test_sample_and_dijkstra(self):
        model = corpus_model(epochs=2)
        samples = model.generate(max_length=30, count=3, seed=11)
        self.assertEqual(len(samples), 3)
        for s in samples:
            self.assertLessEqual(len(s.text), 30)
            self.assertEqual(s.full_text, s.text)
            self.assertEqual(s.node_ids[0], START)
        again = model.generate(max_length=30, count=3, seed=11)
        self.assertEqual([s.text for s in samples], [s.text for s in again])
        (best,) = model.generate(mode="dijkstra", max_length=60, count=3)
        self.assertTrue(best.reached_end)
        self.assertEqual(model.score(best.text)["unknown_transitions"], 0)
        self.assertEqual(model.generate(count=0), [])
        with self.assertRaises(ValueError):
            model.generate(mode="beam")
        with self.assertRaises(ValueError):
            model.generate(count=-1)

    def test_unseeded_generation_consumes_model_rng(self):
        a, b = corpus_model(epochs=1), corpus_model(epochs=1)
        self.assertEqual([r.text for r in a.generate(count=4)], [r.text for r in b.generate(count=4)])
        self.assertEqual(model_state(a), model_state(b))

    def test_empty_model(self):
        model = make_model()
        self.assertEqual(model.generate(count=2)[0].text, "")
        self.assertEqual(model.predict("abc").text, "")
        self.assertEqual(model.score("abc")["unknown_transitions"], 2)


def model_state(model):
    return model.graph.rng.getstate()


class TestScore(unittest.TestCase):
    def test_trained_text_beats_garbage(self):
        model = corpus_model()
        good = model.score(CORPUS[0])
        bad = model.score(GARBAGE[0])
        self.assertEqual(set(good), {"log_prob", "per_char", "chars", "transitions", "unknown_transitions"})
        self.assertGreater(good["log_prob"], bad["log_prob"])
        self.assertGreater(good["per_char"], bad["per_char"])
        self.assertEqual(good["unknown_transitions"], 0)
        self.assertGreater(bad["unknown_transitions"], 0)
        self.assertEqual(good["chars"], len(CORPUS[0]))
        self.assertAlmostEqual(good["per_char"], good["log_prob"] / len(CORPUS[0]))
        self.assertLess(good["log_prob"], 0.0)
        # every corpus line is fully known; most garbage lines are not
        self.assertTrue(all(model.score(t)["unknown_transitions"] == 0 for t in CORPUS))
        self.assertGreater(sum(model.score(t)["unknown_transitions"] > 0 for t in GARBAGE), len(GARBAGE) // 2)

    def test_path_accounting(self):
        model = make_model()
        model.train(["hello world"], epochs=1, **FAST)
        s = model.score("hello world")
        self.assertEqual((s["transitions"], s["unknown_transitions"]), (2, 0))
        self.assertEqual(s["log_prob"], 0.0)  # single-child edges are certain
        s = model.score("hello")
        # START -> "hello world" is fine but ending inside the compressed node is impossible
        self.assertEqual((s["transitions"], s["unknown_transitions"]), (2, 1))
        self.assertAlmostEqual(s["log_prob"], math.log(UNKNOWN_PROB))
        s = model.score("ello world")  # entering the node in the middle
        self.assertEqual(s["unknown_transitions"], 1)
        s = model.score("hi")
        self.assertEqual(s, {"log_prob": 0.0, "per_char": 0.0, "chars": 2, "transitions": 0, "unknown_transitions": 0})
        s = model.score("hello zorld")
        self.assertEqual(s["unknown_transitions"], 4)  # "o z", " zo", "zor", then re-entry into "orl"
        with self.assertRaises(TypeError):
            model.score(None)

    def test_score_matches_search_costs(self):
        model = corpus_model(epochs=2)
        r = model.generate(mode="dijkstra", max_length=60)[0]
        self.assertAlmostEqual(model.score(r.text)["log_prob"], -r.cost, places=9)


class TestStructureOps(unittest.TestCase):
    def test_invert_twice_is_identity(self):
        model = corpus_model(epochs=2)
        g = model.graph
        weights, amps = list(g.edge_w), list(g.a)
        alive = [e for e, ok in enumerate(g.edge_alive) if ok]
        before = snapshot(model)
        model.invert()
        self.assertTrue(model.stats()["inverted"])
        self.assertEqual([g.edge_w[e] for e in alive], [-weights[e] for e in alive])
        self.assertEqual(g.a, [-v for v in amps])
        self.assertNotEqual(snapshot(model), before)
        model.invert()
        self.assertFalse(model.stats()["inverted"])
        self.assertEqual(g.edge_w, weights)
        self.assertEqual(g.a, amps)
        self.assertEqual(snapshot(model), before)

    def test_compress_preserves_predictions(self):
        model = make_model()
        model.train(CORPUS, epochs=2, auto_compress=False, **FAST)
        nodes = model.graph.num_nodes()
        before = snapshot(model)
        scores = [model.score(t)["log_prob"] for t in CORPUS[:10]]
        merges = model.compress()
        self.assertGreater(merges, 100)
        self.assertLess(model.graph.num_nodes(), nodes)
        self.assertEqual(model.compress(), 0)
        self.assertEqual(snapshot(model), before)
        for t, s in zip(CORPUS[:10], scores):
            self.assertAlmostEqual(model.score(t)["log_prob"], s, places=9)
        model.graph.check_invariants(texts=CORPUS, compressed=True)


class TestTwoNRL(unittest.TestCase):
    def test_phases_and_records(self):
        model = make_model()
        seen = []
        result = model.two_nrl(
            bad=GARBAGE[:5], good=CORPUS[:5], neg_epochs=2, pos_epochs=3, neg_lr=0.5, pos_lr=0.2,
            progress=seen.append, batch_size=4,
        )
        self.assertEqual(set(result), {"negative", "positive", "inverted"})
        self.assertEqual([r["phase"] for r in result["negative"]], ["negative"] * 2)
        self.assertEqual([r["phase"] for r in result["positive"]], ["positive"] * 3)
        self.assertEqual(seen, result["negative"] + result["positive"])
        self.assertEqual(model.history, seen)
        self.assertTrue(result["inverted"])
        self.assertTrue(model.graph.inverted)
        self.assertEqual(model.meta["twonrl_runs"], 1)
        self.assertEqual(model.meta["epochs_total"], 5)
        with self.assertRaises(TypeError):
            model.two_nrl(bad=[], good=[], lr=0.1)

    def test_lowers_garbage_continuation_probability(self):
        # "the quick brown fox jumps" continues with "over" (corpus) or "under" (garbage)
        model = make_model()
        model.train(CORPUS + GARBAGE, epochs=5, **FAST)
        under_before = branch_prob(model, "ps ", "s u")
        over_before = branch_prob(model, "ps ", "s o")
        gap_before = model.score(CORPUS[0])["log_prob"] - model.score(GARBAGE[0])["log_prob"]
        model.two_nrl(bad=GARBAGE, good=CORPUS, neg_epochs=8, pos_epochs=8, neg_lr=1.0, pos_lr=1.0, batch_size=1)
        under_after = branch_prob(model, "ps ", "s u")
        over_after = branch_prob(model, "ps ", "s o")
        gap_after = model.score(CORPUS[0])["log_prob"] - model.score(GARBAGE[0])["log_prob"]
        self.assertLess(under_after, under_before)
        self.assertLess(under_after, over_after)
        self.assertGreater(gap_after, gap_before)

    def test_stop_event_skips_positive_phase(self):
        model = make_model()
        stop = threading.Event()
        stop.set()
        result = model.two_nrl(bad=GARBAGE[:3], good=CORPUS[:3], neg_epochs=3, pos_epochs=3, stop_event=stop)
        self.assertEqual(len(result["negative"]), 1)
        self.assertEqual(result["positive"], [])
        self.assertTrue(result["inverted"])


class TestPersistence(unittest.TestCase):
    def test_to_dict_round_trip(self):
        model = corpus_model(epochs=2)
        d = model.to_dict()
        self.assertEqual((d["format"], d["version"], d["backend"]), ("radixnet", 1, "python"))
        json.dumps(d)
        clone = RadixNet.from_dict(json.loads(json.dumps(d)), backend="python")
        self.assertEqual(clone.stats(), model.stats())
        self.assertEqual(clone.history, model.history)
        self.assertEqual(clone.meta, model.meta)
        self.assertEqual(snapshot(clone), snapshot(model))
        with self.assertRaises(ValueError):
            RadixNet.from_dict({"format": "other"})
        with self.assertRaises(ValueError):
            RadixNet.from_dict({"format": "radixnet", "version": 99, "graph": d["graph"]})

    def test_save_load_preserves_predictions_and_training(self):
        model = corpus_model(epochs=3)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("model.json", "model.json.gz"):
                with self.subTest(name=name):
                    path = os.path.join(tmp, name)
                    model.save(path)
                    self.assertTrue(os.path.isfile(path))
                    with open(path, "rb") as fh:
                        magic = fh.read(2)
                    self.assertEqual(magic == b"\x1f\x8b", name.endswith(".gz"))
                    loaded = RadixNet.load(path, backend="python")
                    self.assertEqual(loaded.stats(), model.stats())
                    self.assertEqual(snapshot(loaded), snapshot(model))
                    self.assertEqual(model_state(loaded), model_state(model))
            # identical subsequent training (shuffle order comes from the persisted RNG state)
            path = os.path.join(tmp, "model.json.gz")
            model.save(path)
            loaded = RadixNet.load(path, backend="python")
            for texts in (CORPUS, CORPUS[:20] + ["a brand new sentence appears"]):
                a = model.train(texts, epochs=2, **FAST)
                b = loaded.train(texts, epochs=2, **FAST)
                self.assertEqual([r["loss"] for r in a], [r["loss"] for r in b])
                self.assertEqual([(r["nodes"], r["edges"], r["transitions"]) for r in a],
                                 [(r["nodes"], r["edges"], r["transitions"]) for r in b])
            self.assertEqual(snapshot(loaded), snapshot(model))
            self.assertEqual(loaded.meta["epochs_total"], model.meta["epochs_total"])
            # atomic write leaves no temporary files behind
            self.assertEqual(sorted(os.listdir(tmp)), ["model.json", "model.json.gz"])

    def test_inverted_model_round_trip(self):
        model = corpus_model(epochs=1)
        model.invert()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "inv.json")
            model.save(path)
            loaded = RadixNet.load(path, backend="python")
        self.assertTrue(loaded.graph.inverted)
        self.assertEqual(snapshot(loaded), snapshot(model))


class TestSpecDefaultsLearn(unittest.TestCase):
    def test_default_hyperparameters_lower_the_loss_monotonically(self):
        # lr=0.05 / batch 256 learn slowly (the batch-mean rule), but every epoch must still help
        model = make_model()
        records = model.train(CORPUS, epochs=4)
        losses = [r["loss"] for r in records]
        self.assertTrue(all(b < a for a, b in zip(losses, losses[1:])), losses)
        self.assertEqual(TrainConfig().lr, 0.05)
        self.assertEqual(TrainConfig().batch_size, 256)


class TestRandomisedLifecycle(unittest.TestCase):
    """Fuzz guard: tiny alphabets, pathological texts, interleaved operations, save/load fidelity."""

    ALPHABETS = ("ab", "abc", "a b", "aé😀", "\t \n")
    SPECIALS = ("", "a", "ab", "   ", "a" * 30, "ab" * 12, "abcabcabc", "😀😀😀😀", "<s>", "x<s>y", "</s>")

    def random_text(self, rng):
        if rng.random() < 0.3:
            return rng.choice(self.SPECIALS)
        alpha = rng.choice(self.ALPHABETS)
        n = rng.randint(1, 30)
        if rng.random() < 0.3:
            unit = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 3)))
            return (unit * n)[:n]
        return "".join(rng.choice(alpha) for _ in range(n))

    @staticmethod
    def fingerprint(model):
        out = []
        for prefix in ("", "a", "ab", "aba", "zzz", "😀", "<s>"):
            for to_end in (False, True):
                r = model.predict(prefix, length=6, to_end=to_end)
                out.append((prefix, to_end, r.text, r.labels, r.cost, r.step_costs, r.reached_end))
            state = model.graph.rng.getstate()  # sampling consumes the model RNG: leave it untouched
            r = model.predict(prefix, length=6, mode="sample")
            model.graph.rng.setstate(state)
            out.append((prefix, "sample", r.text, r.labels))
        out.append([model.score(t) for t in ("abab", "aaaa", "x<s>y", "")])
        out.append([r.text for r in model.generate(max_length=12, count=2, seed=5)])
        return out

    def test_random_lifecycles(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                model = make_model(seed)
                seen = []
                for _ in range(6):
                    texts = [self.random_text(rng) for _ in range(rng.randint(0, 5))]
                    op = rng.choice(("train", "train", "two_nrl", "invert", "compress"))
                    if op == "train":
                        model.train(
                            texts, epochs=rng.randint(0, 2), lr=rng.choice((0.05, 1.0)),
                            batch_size=rng.choice((1, 3, 256)), auto_compress=rng.random() < 0.7,
                        )
                    elif op == "two_nrl":
                        good = [self.random_text(rng) for _ in range(rng.randint(0, 3))]
                        model.two_nrl(texts, good, neg_epochs=1, pos_epochs=1, batch_size=2)
                        texts = texts + good
                    elif op == "invert":
                        model.invert()
                        texts = []
                    else:
                        model.compress()
                        texts = []
                    seen.extend(t for t in texts if len(t) >= 3)
                    model.graph.check_invariants(texts=seen)
                    for t in seen:
                        self.assertEqual(model.score(t)["unknown_transitions"], 0, t)
                model.compress()
                model.graph.check_invariants(texts=seen, compressed=True)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "fuzz.json.gz")
                    model.save(path)
                    loaded = RadixNet.load(path, backend="python")
                self.assertEqual(self.fingerprint(loaded), self.fingerprint(model))
                more = [self.random_text(rng) for _ in range(3)]
                a = model.train(more, epochs=2, lr=0.5, batch_size=2)
                b = loaded.train(more, epochs=2, lr=0.5, batch_size=2)
                self.assertEqual([r["loss"] for r in a], [r["loss"] for r in b])
                self.assertEqual(self.fingerprint(loaded), self.fingerprint(model))
                loaded.graph.check_invariants(texts=seen + [t for t in more if len(t) >= 3], compressed=True)


if __name__ == "__main__":
    unittest.main()
