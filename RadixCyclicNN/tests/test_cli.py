"""Subprocess smoke tests for the command-line interface (``python -m radixnet``).

Every command is exercised through a real subprocess with ``--json`` (one
document on stdout) and, for a few, in human-readable mode.  The python
backend is selected explicitly so no subprocess pays for the torch probe.
One model is trained once per module and shared by the read-only commands;
commands that write use ``--out`` with their own files.
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.cli import main  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "sample_corpus.txt")
GARBAGE = os.path.join(ROOT, "data", "sample_garbage.txt")
FAST = ["--batch-size", "8", "--lr", "0.5"]
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
COMMANDS = ("train", "predict", "generate", "score", "2nrl", "invert", "compress", "evolve", "info",
            "checkpoints", "bench", "serve")
POSIX = os.name == "posix"
HAS_BENCH = importlib.util.find_spec("radixnet.bench") is not None
HAS_API = importlib.util.find_spec("radixnet.api") is not None

TMP = None
MODEL = ""
CKPTS = ""
BASE_EPOCHS = 2


def setUpModule():
    """Train the shared model once (with checkpoints) in a module temp dir."""
    global TMP, MODEL, CKPTS
    TMP = tempfile.TemporaryDirectory(prefix="radixnet-cli-")
    MODEL = os.path.join(TMP.name, "model.json")
    CKPTS = os.path.join(TMP.name, "ckpts")
    run_json("train", "--data", CORPUS, "--epochs", BASE_EPOCHS, *FAST,
             "--checkpoint-dir", CKPTS, "--checkpoint-every", 1, "--keep", 3, model=MODEL)


def tearDownModule():
    if TMP is not None:
        TMP.cleanup()


def tmp_path(name):
    return os.path.join(TMP.name, name)


def cli_command(*args, model=None, json_mode=True):
    cmd = [sys.executable, "-m", "radixnet", "--backend", "python"]
    if json_mode:
        cmd.append("--json")
    if model is not None:
        cmd += ["--model", model]
    return cmd + [str(a) for a in args]


def run_cli(*args, model=None, json_mode=True, expect=0, timeout=180):
    """Run the CLI; assert the exit code (``expect=None`` skips the check)."""
    cmd = cli_command(*args, model=model, json_mode=json_mode)
    proc = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if expect is not None and proc.returncode != expect:
        raise AssertionError(
            f"{' '.join(cmd)}\nexit {proc.returncode}, expected {expect}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


def run_json(*args, **kwargs):
    """Run with ``--json`` and parse stdout as exactly one JSON document."""
    proc = run_cli(*args, **kwargs)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"stdout is not one JSON document ({exc}):\n{proc.stdout}\n--- stderr ---\n{proc.stderr}") from None


def _default_sigint():
    # a test runner started as a background shell job inherits SIGINT=ignored; the child must not
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def popen_cli(*args, model=None, json_mode=True):
    kwargs = {"preexec_fn": _default_sigint} if POSIX else {}
    return subprocess.Popen(
        cli_command(*args, model=model, json_mode=json_mode), cwd=ROOT, env=ENV,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", **kwargs,
    )


def wait_for_file(path, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} did not appear within {timeout}s")
        time.sleep(0.05)


def read_file(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestTrain(unittest.TestCase):
    def test_train_document_and_files(self):
        doc = run_json("info", model=MODEL)
        self.assertEqual(doc["stats"]["epochs_total"], BASE_EPOCHS)
        self.assertEqual(doc["stats"]["trained_texts"], 60)
        names = sorted(n for n in os.listdir(CKPTS) if n.startswith("ckpt-"))
        self.assertEqual(names, ["ckpt-epoch-000001.json.gz", "ckpt-epoch-000002.json.gz"])
        self.assertTrue(os.path.isfile(os.path.join(CKPTS, "latest.json")))

    def test_train_json_document(self):
        out = tmp_path("train-doc.json")
        doc = run_json("train", "--data", CORPUS, "--epochs", 2, *FAST, "--out", out, model=tmp_path("absent.json"))
        self.assertEqual(doc["model"], {"kind": "new", "path": None})
        self.assertEqual((doc["texts"], doc["chars"]), (60, sum(len(l) for l in read_file(CORPUS).split("\n") if l.strip())))
        self.assertEqual([r["epoch"] for r in doc["records"]], [1, 2])
        for record in doc["records"]:
            for key in ("loss", "perplexity", "nodes", "edges", "trigrams", "compression_ratio", "merges",
                        "transitions", "seconds", "skipped_short"):
                self.assertIn(key, record)
            self.assertGreater(record["loss"], 0.0)
            self.assertAlmostEqual(record["perplexity"], pow(2.718281828459045, record["loss"]), places=6)
        self.assertLess(doc["records"][1]["loss"], doc["records"][0]["loss"])
        self.assertEqual(doc["config"]["batch_size"], 8)
        self.assertEqual(doc["config"]["lr"], 0.5)
        self.assertFalse(doc["interrupted"])
        self.assertEqual(doc["saved"], {"path": out, "bytes": os.path.getsize(out)})
        self.assertIsNone(doc["checkpoint_dir"])
        self.assertEqual(doc["checkpoints"], [])
        self.assertEqual(doc["stats"]["epochs_total"], 2)
        self.assertEqual(doc["stats"]["backend"], "python")

    def test_train_is_incremental(self):
        out = tmp_path("incremental.json")
        doc = run_json("train", "--data", CORPUS, "--epochs", 1, *FAST, "--out", out, model=MODEL)
        self.assertEqual(doc["model"], {"kind": "model", "path": MODEL})
        self.assertEqual([r["epoch"] for r in doc["records"]], [BASE_EPOCHS + 1])
        self.assertEqual(doc["stats"]["epochs_total"], BASE_EPOCHS + 1)
        self.assertEqual(doc["stats"]["trained_texts"], 120)
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(run_json("info", model=MODEL)["stats"]["epochs_total"], BASE_EPOCHS)  # source untouched

    def test_train_writes_checkpoints_and_rotates(self):
        ckpts = tmp_path("rotating")
        out = tmp_path("rotating-model.json")
        doc = run_json("train", "--data", CORPUS, "--epochs", 4, *FAST, "--checkpoint-dir", ckpts, "--keep", 2,
                       "--out", out, model=tmp_path("absent.json"))
        self.assertEqual(doc["checkpoint_dir"], os.path.abspath(ckpts))
        self.assertEqual([c["step"] for c in doc["checkpoints"]], [3, 4])  # every epoch by default, keep 2
        self.assertEqual(doc["checkpoints"][-1]["metrics"]["epoch"], 4)
        every = run_json("train", "--data", CORPUS, "--epochs", 2, *FAST, "--checkpoint-dir", ckpts,
                         "--checkpoint-every", 2, "--keep", 5, "--out", out, model=out)
        self.assertEqual([c["step"] for c in every["checkpoints"]], [3, 4, 6])

    def test_train_resume_from_latest_checkpoint(self):
        out = tmp_path("resumed.json")
        doc = run_json("train", "--data", CORPUS, "--epochs", 1, *FAST, "--checkpoint-dir", CKPTS,
                       "--checkpoint-every", 0, "--resume", "--out", out, model=tmp_path("absent.json"))
        self.assertEqual(doc["model"]["kind"], "checkpoint")
        self.assertTrue(doc["model"]["path"].endswith("ckpt-epoch-000002.json.gz"))
        self.assertEqual([r["epoch"] for r in doc["records"]], [BASE_EPOCHS + 1])
        self.assertEqual([c["step"] for c in doc["checkpoints"]], [1, 2])  # nothing new was written
        # an empty checkpoint dir falls back to --model with a notice
        proc = run_cli("train", "--data", CORPUS, "--epochs", 1, *FAST, "--checkpoint-dir", tmp_path("empty-ckpts"),
                       "--checkpoint-every", 0, "--resume", "--out", tmp_path("resume-fallback.json"), model=MODEL)
        self.assertIn("no checkpoint to resume from", proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["model"]["kind"], "model")

    def test_train_whole_file_and_several_files(self):
        doc = run_json("train", "--data", CORPUS, GARBAGE, "--whole-file", "--epochs", 1, *FAST,
                       model=tmp_path("whole-file.json"))
        self.assertEqual(doc["texts"], 2)
        expected = len(read_file(CORPUS).rstrip("\r\n")) + len(read_file(GARBAGE).rstrip("\r\n"))
        self.assertEqual(doc["chars"], expected)
        lines = run_json("train", "--data", CORPUS, GARBAGE, "--epochs", 1, *FAST, model=tmp_path("two-files.json"))
        self.assertEqual(lines["texts"], 110)


class TestInference(unittest.TestCase):
    def test_info(self):
        doc = run_json("info", "--tail", 1, model=MODEL)
        stats = doc["stats"]
        for key in ("nodes", "edges", "trigrams", "compression_ratio", "inverted", "backend", "device",
                    "epochs_total", "trained_chars", "trained_texts", "twonrl_runs", "history_len", "last_loss"):
            self.assertIn(key, stats)
        self.assertEqual((stats["backend"], stats["device"]), ("python", "cpu"))
        self.assertGreater(stats["nodes"], 2)
        self.assertFalse(stats["inverted"])
        self.assertEqual(doc["history_len"], BASE_EPOCHS)
        self.assertEqual([r["epoch"] for r in doc["history_tail"]], [BASE_EPOCHS])
        self.assertEqual(doc["meta"]["seed"], 0)
        self.assertEqual(doc["model"], MODEL)

    def test_predict_dijkstra(self):
        prefix = "the quick brown"
        doc = run_json("predict", "--prefix", prefix, "--length", 10, model=MODEL)
        self.assertEqual(doc["prefix"], prefix)
        self.assertEqual(doc["mode"], "dijkstra")
        self.assertTrue(doc["continuation"])
        self.assertEqual(doc["full_text"], prefix + doc["continuation"])
        self.assertEqual(len(doc["path"]), len(doc["node_ids"]))
        self.assertEqual(len(doc["step_costs"]), len(doc["path"]) - 1)
        self.assertAlmostEqual(doc["cost"], sum(doc["step_costs"]), places=9)
        self.assertGreaterEqual(doc["cost"], 0.0)
        self.assertTrue(doc["reached_end"] or len(doc["continuation"]) >= 10)
        self.assertGreater(doc["expanded"], 0)
        to_end = run_json("predict", "--prefix", prefix, "--length", 5, "--max-length", 80, "--to-end",
                          "--step-penalty", 0.1, model=MODEL)
        self.assertTrue(to_end["reached_end"])
        self.assertEqual(to_end["path"][-1], "</s>")

    def test_predict_sample(self):
        prefix = "the cat"
        doc = run_json("predict", "--prefix", prefix, "--length", 12, "--mode", "sample", "--temperature", 0.5, model=MODEL)
        self.assertEqual(doc["mode"], "sample")
        self.assertEqual(doc["full_text"], prefix + doc["continuation"])
        self.assertLessEqual(len(doc["continuation"]), 12)
        greedy = run_json("predict", "--prefix", prefix, "--length", 12, "--mode", "sample", "--temperature", 0, model=MODEL)
        self.assertEqual(greedy["full_text"], prefix + greedy["continuation"])

    def test_generate(self):
        doc = run_json("--seed", 7, "generate", "--count", 3, "--max-length", 30, model=MODEL)
        self.assertEqual((doc["count"], doc["mode"], doc["max_length"]), (3, "sample", 30))
        self.assertEqual(len(doc["samples"]), 3)
        for sample in doc["samples"]:
            self.assertLessEqual(len(sample["text"]), 30)
            self.assertEqual(sample["full_text"], sample["text"])
            self.assertEqual(sample["labels"][0], "<s>")
            self.assertAlmostEqual(sample["cost"], sum(sample["step_costs"]), places=9)
        again = run_json("--seed", 7, "generate", "--count", 3, "--max-length", 30, model=MODEL)
        self.assertEqual([s["text"] for s in again["samples"]], [s["text"] for s in doc["samples"]])
        cheapest = run_json("generate", "--mode", "dijkstra", "--count", 3, "--max-length", 40, model=MODEL)
        self.assertEqual(len(cheapest["samples"]), 1)
        self.assertTrue(cheapest["samples"][0]["reached_end"])
        self.assertEqual(run_json("generate", "--count", 0, model=MODEL)["samples"], [])

    def test_score(self):
        good = run_json("score", "--text", "the cat sat on the mat", model=MODEL)
        self.assertEqual(good["count"], 1)
        result = good["results"][0]
        for key in ("text", "log_prob", "per_char", "chars", "transitions", "unknown_transitions"):
            self.assertIn(key, result)
        self.assertEqual((result["chars"], result["unknown_transitions"]), (22, 0))
        self.assertLess(result["log_prob"], 0.0)
        self.assertAlmostEqual(result["per_char"], result["log_prob"] / 22, places=9)
        bad = run_json("score", "--text", "zzqx vvkk jjpp", model=MODEL)["results"][0]
        self.assertGreater(bad["unknown_transitions"], 0)
        self.assertLess(bad["per_char"], result["per_char"])
        many = run_json("score", "--data", GARBAGE, model=MODEL)
        self.assertEqual((many["count"], len(many["results"])), (50, 50))
        self.assertAlmostEqual(many["mean_per_char"], sum(r["per_char"] for r in many["results"]) / 50, places=9)
        self.assertAlmostEqual(many["mean_log_prob"], sum(r["log_prob"] for r in many["results"]) / 50, places=9)
        corpus = run_json("score", "--data", CORPUS, model=MODEL)
        self.assertGreater(corpus["mean_per_char"], many["mean_per_char"])


class TestTransformations(unittest.TestCase):
    def test_two_nrl(self):
        out = tmp_path("twonrl.json")
        doc = run_json("2nrl", "--bad", GARBAGE, "--good", CORPUS, "--neg-epochs", 1, "--pos-epochs", 2,
                       "--batch-size", 8, "--out", out, model=MODEL)
        self.assertTrue(doc["inverted"])
        self.assertEqual((doc["bad_texts"], doc["good_texts"]), (50, 60))
        self.assertEqual([r["phase"] for r in doc["negative"]], ["negative"])
        self.assertEqual([r["phase"] for r in doc["positive"]], ["positive", "positive"])
        self.assertEqual([r["epoch"] for r in doc["negative"] + doc["positive"]], [3, 4, 5])
        self.assertEqual(doc["stats"]["twonrl_runs"], 1)
        self.assertTrue(doc["stats"]["inverted"])
        self.assertFalse(doc["interrupted"])
        self.assertEqual(doc["saved"]["path"], out)
        self.assertEqual(run_json("info", model=out)["stats"]["twonrl_runs"], 1)
        self.assertEqual(run_json("info", model=MODEL)["stats"]["twonrl_runs"], 0)  # source untouched
        fresh = run_json("2nrl", "--bad", GARBAGE, "--good", CORPUS, "--neg-epochs", 1, "--pos-epochs", 1,
                         model=tmp_path("fresh-2nrl.json"))
        self.assertEqual(fresh["model"]["kind"], "new")
        self.assertTrue(os.path.isfile(tmp_path("fresh-2nrl.json")))

    def test_invert_twice_is_identity(self):
        once, twice = tmp_path("inverted.json"), tmp_path("inverted-twice.json")
        first = run_json("invert", "--out", once, model=MODEL)
        self.assertEqual((first["was_inverted"], first["inverted"]), (False, True))
        self.assertTrue(first["stats"]["inverted"])
        second = run_json("invert", "--out", twice, model=once)
        self.assertEqual((second["was_inverted"], second["inverted"]), (True, False))
        self.assertFalse(run_json("info", model=MODEL)["stats"]["inverted"])  # source untouched
        before = run_json("predict", "--prefix", "the quick brown", model=MODEL)
        after = run_json("predict", "--prefix", "the quick brown", model=twice)
        self.assertEqual(before["continuation"], after["continuation"])
        self.assertAlmostEqual(before["cost"], after["cost"], places=9)

    def test_compress(self):
        raw = tmp_path("uncompressed.json")
        run_json("train", "--data", CORPUS, "--epochs", 1, *FAST, "--no-compress", model=raw)
        before = run_json("predict", "--prefix", "the quick brown", "--length", 8, model=raw)
        out = tmp_path("compressed.json")
        doc = run_json("compress", "--out", out, model=raw)
        self.assertGreater(doc["merges"], 0)
        self.assertEqual(doc["nodes_after"], doc["nodes_before"] - doc["merges"])
        self.assertEqual(doc["stats"]["nodes"], doc["nodes_after"])
        self.assertGreater(doc["stats"]["compression_ratio"], 1.0)
        after = run_json("predict", "--prefix", "the quick brown", "--length", 8, model=out)
        self.assertEqual(before["continuation"], after["continuation"])
        self.assertAlmostEqual(before["cost"], after["cost"], places=6)
        idempotent = run_json("compress", "--out", tmp_path("compressed-again.json"), model=out)
        self.assertEqual(idempotent["merges"], 0)

    def test_checkpoints_list_and_restore(self):
        doc = run_json("checkpoints", "--dir", CKPTS, model=MODEL)
        self.assertEqual(doc["directory"], os.path.abspath(CKPTS))
        self.assertEqual([c["name"] for c in doc["checkpoints"]], ["ckpt-epoch-000001.json.gz", "ckpt-epoch-000002.json.gz"])
        self.assertEqual([c["step"] for c in doc["checkpoints"]], [1, 2])
        self.assertEqual(doc["checkpoints"][0]["metrics"]["epoch"], 1)
        self.assertEqual(doc["latest"]["name"], "ckpt-epoch-000002.json.gz")
        out = tmp_path("restored.json")
        restored = run_json("checkpoints", "--dir", CKPTS, "--restore", "ckpt-epoch-000001", "--out", out, model=MODEL)
        self.assertEqual(restored["restored"]["name"], "ckpt-epoch-000001.json.gz")
        self.assertEqual(restored["out"], out)
        self.assertEqual(restored["stats"]["epochs_total"], 1)
        self.assertEqual(run_json("info", model=out)["stats"]["epochs_total"], 1)
        latest = run_json("checkpoints", "--dir", CKPTS, "--restore", "latest", "--out", tmp_path("latest.json"), model=MODEL)
        self.assertEqual(latest["restored"]["name"], "ckpt-epoch-000002.json.gz")
        self.assertEqual(latest["stats"]["epochs_total"], 2)
        by_path = run_json("checkpoints", "--dir", CKPTS, "--restore", doc["checkpoints"][0]["path"],
                           "--out", tmp_path("by-path.json"), model=MODEL)
        self.assertEqual(by_path["restored"]["step"], 1)
        target = tmp_path("restore-default-out.json")
        run_json("checkpoints", "--dir", CKPTS, "--restore", "latest", model=target)  # --out defaults to --model
        self.assertTrue(os.path.isfile(target))

    def test_evolve_finite_with_discriminator(self):
        out, disc, ckpts = tmp_path("evolved.json"), tmp_path("disc.json"), tmp_path("evo-ckpts")
        doc = run_json("evolve", "--data", CORPUS, "--generations", 2, "--samples", 4, "--max-length", 30,
                       "--discriminator", disc, "--checkpoint-dir", ckpts, "--out", out, model=MODEL)
        self.assertEqual([g["generation"] for g in doc["generations"]], [1, 2])
        self.assertEqual(doc["generation"], 2)
        for record in doc["generations"]:
            for key in ("fake_score_mean", "real_score_mean", "gap", "gen_loss", "nodes", "edges",
                        "compression_ratio", "sample", "seconds"):
                self.assertIn(key, record)
        self.assertEqual(doc["config"]["samples"], 4)
        self.assertFalse(doc["interrupted"])
        self.assertTrue(os.path.isfile(out))
        self.assertTrue(os.path.isfile(disc))
        self.assertEqual(doc["discriminator"]["path"], disc)
        self.assertEqual(doc["discriminator"]["stats"]["twonrl_runs"], 2)
        self.assertEqual(doc["stats"]["twonrl_runs"], 2)
        names = sorted(n for n in os.listdir(ckpts) if n.startswith("ckpt-"))
        self.assertEqual(names, ["ckpt-gen-000001.json.gz", "ckpt-gen-000002.json.gz"])
        again = run_json("evolve", "--data", CORPUS, "--generations", 1, "--samples", 2, "--discriminator", disc,
                         "--out", out, model=out)
        self.assertEqual(again["model"], {"kind": "model", "path": out})
        self.assertEqual(again["discriminator"]["stats"]["twonrl_runs"], 3)  # the saved discriminator was loaded
        self.assertEqual(again["stats"]["twonrl_runs"], 3)
        self.assertIsNone(run_json("evolve", "--data", CORPUS, "--generations", 1, "--samples", 2,
                                   "--out", tmp_path("evolved-nodisc.json"), model=MODEL)["discriminator"])


class TestOutputModes(unittest.TestCase):
    def test_human_train_prints_one_row_per_epoch(self):
        proc = run_cli("train", "--data", CORPUS, "--epochs", 2, *FAST, "--out", tmp_path("human-train.json"),
                       json_mode=False, model=MODEL)
        lines = proc.stdout.splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("epoch"))
        self.assertRegex(lines[header], r"epoch\s+loss\s+ppl\s+lr\s+act_lr\s+nodes\s+edges\s+trigrams\s+ratio\s+merges\s+transitions\s+seconds")
        self.assertRegex(lines[header + 2], r"^\s*3\s+\d+\.\d{4}\s+\d+\.\d{4}\s+\d+\.\d{4}\s+\d+\.\d{4}\s+\d+\s+\d+\s+\d+\s+")
        self.assertRegex(lines[header + 3], r"^\s*4\s+\d+\.\d{4}\s+")
        self.assertIn("saved ", proc.stdout)
        self.assertEqual(proc.stderr, "")

    def test_human_info_predict_generate_score(self):
        info = run_cli("info", json_mode=False, model=MODEL).stdout
        for word in ("nodes", "edges", "trigrams", "compression ratio", "epochs total", "last loss"):
            self.assertIn(word, info)
        predict = run_cli("predict", "--prefix", "the quick", "--length", 6, json_mode=False, model=MODEL).stdout
        for word in ("prefix", "continuation", "full text", "cost", "path:", "label"):
            self.assertIn(word, predict)
        generate = run_cli("generate", "--count", 2, json_mode=False, model=MODEL).stdout
        self.assertRegex(generate, r"#\s+cost\s+end\s+text")
        score = run_cli("score", "--text", "the cat sat on the mat", json_mode=False, model=MODEL).stdout
        self.assertRegex(score, r"log_prob\s+per_char\s+chars\s+transitions\s+unknown\s+text")

    def test_json_mode_single_document_with_progress_on_stderr(self):
        proc = run_cli("train", "--data", CORPUS, "--epochs", 2, *FAST, "--out", tmp_path("json-train.json"), model=MODEL)
        json.loads(proc.stdout)
        self.assertEqual(len(re.findall(r"^epoch \d+ loss=", proc.stderr, re.M)), 2)
        two = run_cli("2nrl", "--bad", GARBAGE, "--good", CORPUS, "--neg-epochs", 1, "--pos-epochs", 1,
                      "--out", tmp_path("json-2nrl.json"), model=MODEL)
        json.loads(two.stdout)
        self.assertRegex(two.stderr, r"epoch \d+ \[negative\] loss=")
        self.assertRegex(two.stderr, r"epoch \d+ \[positive\] loss=")

    def test_global_options_after_the_command(self):
        cmd = [sys.executable, "-m", "radixnet", "info", "--backend", "python", "--json", "--model", MODEL]
        proc = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["stats"]["backend"], "python")


class TestErrors(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cli-errors-", dir=TMP.name)
        self.missing = os.path.join(self.dir, "missing.json")

    def test_missing_model_file(self):
        for command in (["predict", "--prefix", "abc"], ["generate"], ["score", "--text", "abc"], ["info"],
                        ["invert"], ["compress"]):
            with self.subTest(command=command[0]):
                proc = run_cli(*command, model=self.missing, expect=1)
                self.assertIn("model file not found", proc.stderr)
                self.assertEqual(proc.stdout, "")

    def test_bad_arguments(self):
        cases = [
            [], ["no-such-command"], ["train"], ["score"], ["predict"], ["2nrl", "--bad", GARBAGE],
            ["train", "--data", CORPUS, "--epochs", "-1"], ["train", "--data", CORPUS, "--batch-size", "0"],
            ["train", "--data", CORPUS, "--lr", "abc"], ["evolve", "--data", CORPUS, "--generations", "x"],
            ["predict", "--prefix", "x", "--mode", "magic"], ["train", "--data", CORPUS, "--resume"],
            ["train", "--data", CORPUS, "--checkpoint-every", "2"], ["checkpoints", "--dir", CKPTS, "--out", "x.json"],
            ["info", "--unknown-option"], ["--backend", "bogus", "info"],
        ]
        for case in cases:
            with self.subTest(case=case):
                proc = run_cli(*case, model=MODEL, expect=1)
                self.assertTrue(proc.stderr.strip())
                self.assertEqual(proc.stdout, "")

    def test_library_errors_become_messages(self):
        proc = run_cli("--device", "cuda", "info", model=MODEL, expect=1)
        self.assertIn("error", proc.stderr)
        proc = run_cli("evolve", "--data", CORPUS, "--generations", 1, "--max-length", 2, model=MODEL, expect=1)
        self.assertIn("max_length", proc.stderr)

    def test_missing_or_empty_data_files(self):
        proc = run_cli("train", "--data", os.path.join(self.dir, "nope.txt"), model=self.missing, expect=1)
        self.assertIn("data file not found", proc.stderr)
        empty = os.path.join(self.dir, "empty.txt")
        with open(empty, "w", encoding="utf-8") as fh:
            fh.write("\n\n   \n")
        self.assertIn("no training texts", run_cli("train", "--data", empty, model=self.missing, expect=1).stderr)
        self.assertIn("no scorable texts", run_cli("score", "--data", empty, model=MODEL, expect=1).stderr)
        self.assertIn("no bad texts", run_cli("2nrl", "--bad", empty, "--good", CORPUS, model=MODEL, expect=1).stderr)
        missing_good = run_cli("2nrl", "--bad", GARBAGE, "--good", os.path.join(self.dir, "nope.txt"), model=MODEL, expect=1)
        self.assertIn("data file not found", missing_good.stderr)

    def test_checkpoint_errors(self):
        proc = run_cli("checkpoints", "--dir", os.path.join(self.dir, "no-dir"), expect=1)
        self.assertIn("checkpoint directory not found", proc.stderr)
        proc = run_cli("checkpoints", "--dir", CKPTS, "--restore", "ckpt-epoch-000099", "--out", self.missing, expect=1)
        self.assertIn("no checkpoint", proc.stderr)
        empty = os.path.join(self.dir, "empty-ckpts")
        os.makedirs(empty)
        self.assertEqual(run_json("checkpoints", "--dir", empty), {"directory": empty, "checkpoints": [], "latest": None})
        proc = run_cli("checkpoints", "--dir", empty, "--restore", "latest", "--out", self.missing, expect=1)
        self.assertIn("no latest checkpoint", proc.stderr)

    def test_corrupt_model_file(self):
        corrupt = os.path.join(self.dir, "corrupt.json")
        with open(corrupt, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        proc = run_cli("info", model=corrupt, expect=1)
        self.assertIn("error", proc.stderr)
        with open(corrupt, "w", encoding="utf-8") as fh:
            fh.write('{"format": "something-else"}')
        self.assertIn("not a radixnet model", run_cli("info", model=corrupt, expect=1).stderr)

    def test_help_for_every_command(self):
        top = run_cli("--help", json_mode=False, expect=0)
        self.assertIn("usage:", top.stdout)
        for command in COMMANDS:
            self.assertIn(command, top.stdout)
            with self.subTest(command=command):
                proc = run_cli(command, "--help", json_mode=False, expect=0)
                self.assertIn(f"usage: radixnet {command}", proc.stdout)
                self.assertIn("--json", proc.stdout)
                self.assertEqual(proc.stderr, "")
        version = run_cli("--version", json_mode=False, expect=0)
        self.assertIn("radixnet", version.stdout)


class TestMainInProcess(unittest.TestCase):
    """``main()`` returns exit codes instead of raising SystemExit."""

    def call(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_help_and_version_return_zero(self):
        code, out, err = self.call(["--help"])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("usage:", out)
        code, out, _ = self.call(["--version"])
        self.assertEqual(code, 0)
        self.assertIn("radixnet", out)

    def test_usage_error_returns_one(self):
        code, out, err = self.call([])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("error", err)
        code, _, err = self.call(["train", "--epochs", "-3", "--data", CORPUS])
        self.assertEqual(code, 1)
        self.assertIn("--epochs", err)

    def test_missing_model_returns_one(self):
        code, out, err = self.call(["--json", "--backend", "python", "--model", tmp_path("nothing.json"), "info"])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("model file not found", err)

    def test_success_returns_zero(self):
        code, out, _ = self.call(["--json", "--backend", "python", "--model", MODEL, "score", "--text", "the cat sat"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["count"], 1)


@unittest.skipUnless(POSIX, "SIGINT delivery to a subprocess needs POSIX")
class TestCtrlC(unittest.TestCase):
    def finish(self, proc, timeout=90):
        try:
            return proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise

    def test_evolve_forever_stops_cleanly_and_saves(self):
        ckpts, out, disc = tmp_path("ctrlc-ckpts"), tmp_path("ctrlc-evolved.json"), tmp_path("ctrlc-disc.json")
        proc = popen_cli("evolve", "--data", CORPUS, "--generations", 0, "--samples", 2, "--max-length", 20,
                         "--checkpoint-dir", ckpts, "--discriminator", disc, "--out", out, model=MODEL)
        try:
            wait_for_file(os.path.join(ckpts, "latest.json"))
            proc.send_signal(signal.SIGINT)
        finally:
            stdout, stderr = self.finish(proc)
        self.assertEqual(proc.returncode, 0, stderr)
        doc = json.loads(stdout)
        self.assertTrue(doc["interrupted"])
        self.assertGreaterEqual(len(doc["generations"]), 1)
        self.assertEqual(doc["generation"], len(doc["generations"]))
        self.assertTrue(os.path.isfile(out))
        self.assertTrue(os.path.isfile(disc))
        self.assertIn("interrupted", stderr)
        self.assertEqual(run_json("info", model=out)["stats"]["twonrl_runs"], doc["generation"])

    def test_train_streams_rows_and_stops_on_ctrl_c(self):
        out = tmp_path("ctrlc-train.json")
        proc = popen_cli("train", "--data", CORPUS, "--epochs", 100000, *FAST, "--out", out, json_mode=False, model=MODEL)
        row = re.compile(r"^\s*(\d+)\s+\d+\.\d{4}\s")
        first = None
        try:
            for line in proc.stdout:
                match = row.match(line)
                if match:
                    first = int(match.group(1))
                    break
            self.assertIsNotNone(first, "no epoch row was streamed")
            self.assertIsNone(proc.poll(), "the process had already finished when the first row arrived")
            proc.send_signal(signal.SIGINT)
        finally:
            _, stderr = self.finish(proc)
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertEqual(first, BASE_EPOCHS + 1)
        match = re.search(r"stopped after (\d+) epoch", stderr)
        self.assertIsNotNone(match, stderr)
        stopped = int(match.group(1))
        self.assertLess(stopped, 50)  # rows were flushed as epochs finished, not buffered until the end
        self.assertEqual(run_json("info", model=out)["stats"]["epochs_total"], BASE_EPOCHS + stopped)


@unittest.skipUnless(HAS_BENCH, "radixnet.bench is not available")
class TestBench(unittest.TestCase):
    def test_bench(self):
        doc = run_json("bench", "--chars", 2000, "--epochs", 1, timeout=600)
        self.assertIsInstance(doc, dict)
        self.assertTrue(doc)
        proc = run_cli("bench", "--chars", 1000, "--epochs", 1, json_mode=False, timeout=600)
        self.assertTrue(proc.stdout.strip())


@unittest.skipUnless(HAS_API, "radixnet.api is not available")
class TestServe(unittest.TestCase):
    @staticmethod
    def free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @staticmethod
    def get_json(url, timeout=30.0):
        deadline = time.monotonic() + timeout
        while True:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                if time.monotonic() > deadline:
                    raise AssertionError(f"{url} not reachable: {exc}") from None
                time.sleep(0.1)

    def test_serve_health_status_and_clean_shutdown(self):
        port = self.free_port()
        proc = popen_cli("serve", "--host", "127.0.0.1", "--port", port, "--checkpoint-dir", tmp_path("serve-ckpts"),
                         "--frontend-dir", tmp_path("no-frontend"), model=MODEL)
        try:
            health = self.get_json(f"http://127.0.0.1:{port}/api/health")
            self.assertTrue(health["ok"])
            status = self.get_json(f"http://127.0.0.1:{port}/api/status")
            self.assertEqual(status["epochs_total"], BASE_EPOCHS)  # --model was loaded at start-up
        finally:
            if POSIX:
                proc.send_signal(signal.SIGINT)
            else:
                proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
        if POSIX:
            self.assertEqual(proc.returncode, 0, stderr)
        doc, _ = json.JSONDecoder().raw_decode(stdout)  # the settings document precedes anything the server prints
        self.assertEqual((doc["host"], doc["port"], doc["model"]), ("127.0.0.1", port, MODEL))
        self.assertTrue(doc["model_exists"])
        self.assertFalse(doc["frontend_built"])


if __name__ == "__main__":
    unittest.main()
