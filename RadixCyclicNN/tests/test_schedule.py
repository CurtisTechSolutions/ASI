"""Tests for radixnet.schedule - learning-rate schedules as graph functions of the epoch.

Covers the expression evaluator (values, helpers, presets, safety), the
``TrainConfig`` integration (validation and per-epoch rates in the training
records), the HTTP API (``GET /api/schedule``, ``POST /api/schedule/preview``,
``POST /api/train`` with schedules) and the CLI (``schedule``, ``train
--lr-schedule``).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet import schedule as sched  # noqa: E402
from radixnet.model import RadixNet, TrainConfig  # noqa: E402
from radixnet.schedule import PRESETS, Schedule, ScheduleError, describe, parse_schedule, preview_points  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lines(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines() if line.strip()]


CORPUS = _lines("sample_corpus.txt")


def rates(expression, epochs, base=1.0, **extra):
    return Schedule(expression).preview(epochs, base, **extra)


def close(a, b):
    return all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(a, b)) and len(a) == len(b)


# ---------------------------------------------------------------------------
# the evaluator
# ---------------------------------------------------------------------------


class TestExpressions(unittest.TestCase):
    def test_constant_base_rate(self):
        self.assertEqual(rates("lr0", 4, 0.05), [0.05] * 4)
        self.assertEqual(rates("act_lr0", 3, 0.05, act_lr0=0.007), [0.007] * 3)

    def test_variables(self):
        self.assertEqual(rates("epoch", 4), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(rates("i", 4), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(rates("epochs", 3), [3.0, 3.0, 3.0])
        self.assertTrue(close(rates("t", 5), [0.0, 0.25, 0.5, 0.75, 1.0]))
        self.assertEqual(rates("t", 1), [0.0])  # a single epoch: t is 0, never a division by zero

    def test_arithmetic_and_functions(self):
        self.assertTrue(close(rates("lr0 * 1.25 ** i", 3, 0.08), [0.08, 0.1, 0.125]))
        self.assertTrue(close(rates("0.05 + 0.45 * t", 3), [0.05, 0.275, 0.5]))
        self.assertTrue(close(rates("sqrt(epoch)", 4), [1.0, math.sqrt(2), math.sqrt(3), 2.0]))
        self.assertTrue(close(rates("exp(i) / e ** i", 3), [1.0, 1.0, 1.0]))
        self.assertTrue(close(rates("abs(cos(pi * t))", 3), [1.0, 0.0, 1.0]))
        self.assertTrue(close(rates("clamp(epoch, 2, 3)", 4), [2.0, 2.0, 3.0, 3.0]))
        self.assertTrue(close(rates("min(epoch, 2) + max(i, 1)", 3), [2.0, 3.0, 4.0]))
        self.assertTrue(close(rates("7 // 2 + 7 % 2", 1), [4.0]))
        self.assertEqual(rates("log2(8) + log10(100) + log(e) + round(2.4) + floor(1.9) + ceil(0.1)", 1), [10.0])

    def test_conditionals(self):
        self.assertEqual(rates("0.1 if epoch < 3 else 0.5", 4), [0.1, 0.1, 0.5, 0.5])
        self.assertEqual(rates("1 if epoch >= 2 and epoch <= 3 else 2", 4), [2.0, 1.0, 1.0, 2.0])
        self.assertEqual(rates("3 if epoch == 1 or epoch == 4 else 0", 4), [3.0, 0.0, 0.0, 3.0])
        self.assertEqual(rates("5 if not epoch != 1 else 6", 2), [5.0, 6.0])

    def test_linear_helper(self):
        self.assertTrue(close(rates("linear(lr0, 4 * lr0)", 4, 0.1), [0.1, 0.2, 0.3, 0.4]))
        self.assertTrue(close(rates("linear(0.5, 0.1)", 5), [0.5, 0.4, 0.3, 0.2, 0.1]))  # decreasing is fine too
        self.assertEqual(rates("linear(2, 8)", 1), [2.0])

    def test_geometric_helper(self):
        self.assertTrue(close(rates("geometric(lr0, 8 * lr0)", 4, 0.01), [0.01, 0.02, 0.04, 0.08]))
        self.assertEqual(rates("geometric(1, 4)", 1), [1.0])

    def test_cosine_helper(self):
        values = rates("cosine(1, 3)", 5)
        self.assertTrue(close(values, [1.0, 1 + 2 * (1 - math.cos(math.pi / 4)) / 2, 2.0, 1 + 2 * (1 - math.cos(3 * math.pi / 4)) / 2, 3.0]))
        self.assertEqual(values, sorted(values))

    def test_step_helper(self):
        self.assertTrue(close(rates("step(lr0, 1.5, 2)", 6, 0.1), [0.1, 0.1, 0.15, 0.15, 0.225, 0.225]))
        self.assertTrue(close(rates("step(2, 3, 1)", 3), [2.0, 6.0, 18.0]))
        self.assertTrue(close(rates("step(2, 3, 0)", 3), [2.0, 6.0, 18.0]))  # every < 1 counts as 1

    def test_warmup_helper(self):
        self.assertTrue(close(rates("warmup(lr0 / 10, lr0, 3)", 6, 0.3), [0.03, 0.12, 0.21, 0.3, 0.3, 0.3]))
        self.assertTrue(close(rates("warmup(0, 1, 0)", 3), [1.0, 1.0, 1.0]))

    def test_activation_schedule_sees_the_learning_rate(self):
        self.assertEqual(rates("lr / 10", 2, 0.005, lr=0.5), [0.05, 0.05])
        self.assertEqual(rates("lr", 1, 0.005), [0.005])  # without an explicit lr it is the base rate

    def test_repr_and_expression_are_stripped(self):
        s = Schedule("  lr0 * 2  ")
        self.assertEqual(s.expression, "lr0 * 2")
        self.assertEqual(repr(s), "Schedule('lr0 * 2')")


class TestRejectedExpressions(unittest.TestCase):
    def assertRejected(self, expression, fragment=None, epochs=3, base=0.1):
        with self.assertRaises(ScheduleError) as ctx:
            Schedule(expression).preview(epochs, base)
        self.assertIsInstance(ctx.exception, ValueError)
        if fragment is not None:
            self.assertIn(fragment, str(ctx.exception))

    def test_empty_or_syntax_errors(self):
        self.assertRejected("", "empty")
        self.assertRejected("   ", "empty")
        self.assertRejected("lr0 *", "invalid schedule expression")
        self.assertRejected("import os", "invalid schedule expression")
        self.assertRejected("x = 1", "invalid")

    def test_no_names_outside_the_whitelist(self):
        self.assertRejected("__import__('os')", "only these functions may be called")
        self.assertRejected("open('x')", "only these functions may be called")
        self.assertRejected("eval('1')", "only these functions")
        self.assertRejected("foo", "unknown name 'foo'")
        self.assertRejected("__builtins__", "unknown name")
        self.assertRejected("globals", "unknown name")

    def test_no_attributes_subscripts_strings_or_statements(self):
        self.assertRejected("lr0.__class__", "Attribute is not allowed")
        self.assertRejected("(1).real", "Attribute")
        self.assertRejected("'abc'", "only numbers are allowed as constants")
        self.assertRejected("b'abc'", "only numbers")
        self.assertRejected("f'{lr0}'", "not allowed")
        self.assertRejected("None", "only numbers")
        self.assertRejected("True", "only numbers")
        self.assertRejected("[1, 2][0]", "not allowed")
        self.assertRejected("(1, 2)", "not allowed")
        self.assertRejected("{1: 2}", "not allowed")
        self.assertRejected("lambda: 1", "not allowed")
        self.assertRejected("[x for x in (1,)]", "not allowed")
        self.assertRejected("(y := 1)", "not allowed")
        self.assertRejected("1 & 2", "operator BitAnd is not allowed")
        self.assertRejected("1 << 2", "operator LShift is not allowed")
        self.assertRejected("~1", "operator Invert is not allowed")
        self.assertRejected("1 is 1", "comparisons are allowed")
        self.assertRejected("1 in 1", "comparisons are allowed")
        self.assertRejected("min(1, 2, key=abs)", "positional arguments only")
        self.assertRejected("max(*lr0)", "positional arguments only")
        self.assertRejected("sin.real", "Attribute")

    def test_only_helpers_may_be_called(self):
        self.assertRejected("lr0()", "only these functions may be called")
        self.assertRejected("epoch(1)", "only these functions may be called")
        self.assertRejected("sin(1)(2)", "only these functions may be called")
        self.assertTrue(close(rates("(sin)(1)", 1), [math.sin(1)]))  # grouping parentheses are fine

    def test_bad_values(self):
        self.assertRejected("lr0 - 1", "must be finite and >= 0")
        self.assertRejected("-lr0", "must be finite and >= 0")
        self.assertRejected("lr0 / 0", "failed at epoch 1")
        self.assertRejected("lr0 / i", "failed at epoch 1")  # i is 0 at the first epoch
        self.assertRejected("1e308 * 10", "finite")
        self.assertRejected("log(0)", "failed at epoch")
        self.assertRejected("sqrt(-1)", "failed at epoch")
        self.assertRejected("epoch < 3", "must give a number, got bool")
        self.assertRejected("sin", "must give a number")
        self.assertRejected("min()", "failed at epoch")
        self.assertRejected("linear(1)", "failed at epoch")

    def test_huge_powers_overflow_instead_of_hanging(self):
        # integer literals are evaluated as floats: 2 ** 100000000 raises OverflowError at once
        # instead of building a hundred-million-bit integer.
        self.assertRejected("2 ** 100000000", "failed at epoch 1")
        self.assertRejected("10 ** 400", "failed at epoch 1")
        self.assertRejected("2 ** 2 ** 2 ** 2 ** 2 ** 2", "failed at epoch 1")
        self.assertRejected("geometric(0, 1)", "failed at epoch")  # 0 ** t at t = 0 is fine, then a division by zero

    def test_failure_names_the_epoch(self):
        with self.assertRaises(ScheduleError) as ctx:
            Schedule("lr0 if epoch < 3 else lr0 - 1").preview(4, 0.1)
        self.assertIn("at epoch 3", str(ctx.exception))
        self.assertEqual(Schedule("lr0 if epoch < 3 else lr0 - 1").preview(2, 0.1), [0.1, 0.1])


class TestParseAndPreview(unittest.TestCase):
    def test_parse_schedule(self):
        self.assertIsNone(parse_schedule(None))
        self.assertIsNone(parse_schedule(""))
        self.assertIsNone(parse_schedule("   \n"))
        self.assertIsInstance(parse_schedule("lr0"), Schedule)
        with self.assertRaises(ScheduleError):
            parse_schedule(3)
        with self.assertRaises(ScheduleError):
            parse_schedule(["lr0"])

    def test_preview_points_constant_when_no_schedule(self):
        points = preview_points(None, None, 3, 0.05, 0.005)
        self.assertEqual(points, [{"epoch": e, "lr": 0.05, "act_lr": 0.005} for e in (1, 2, 3)])
        self.assertEqual(preview_points("", "  ", 2, 0.5, 0.05), [{"epoch": 1, "lr": 0.5, "act_lr": 0.05}, {"epoch": 2, "lr": 0.5, "act_lr": 0.05}])
        self.assertEqual(preview_points(None, None, 0, 0.05, 0.005), [])
        self.assertEqual(preview_points("linear(lr0, 2 * lr0)", None, 0, 0.05, 0.005), [])

    def test_preview_points_each_schedule_has_its_own_base(self):
        points = preview_points("linear(lr0, 2 * lr0)", "linear(act_lr0, 3 * act_lr0)", 3, 0.1, 0.01)
        self.assertEqual([p["epoch"] for p in points], [1, 2, 3])
        self.assertTrue(close([p["lr"] for p in points], [0.1, 0.15, 0.2]))
        self.assertTrue(close([p["act_lr"] for p in points], [0.01, 0.02, 0.03]))
        # lr0 of the activation schedule is the activation base rate; act_lr0 is visible to both
        self.assertTrue(close([p["act_lr"] for p in preview_points(None, "lr0 * 2", 2, 0.1, 0.01)], [0.02, 0.02]))
        self.assertTrue(close([p["lr"] for p in preview_points("act_lr0 * 100", None, 1, 0.1, 0.01)], [1.0]))

    def test_activation_schedule_follows_the_epoch_learning_rate(self):
        points = preview_points("linear(lr0, 5 * lr0)", "lr / 10", 5, 0.1, 0.005)
        self.assertTrue(close([p["lr"] for p in points], [0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertTrue(close([p["act_lr"] for p in points], [0.01, 0.02, 0.03, 0.04, 0.05]))
        # without a learning-rate schedule, lr is the constant base rate
        self.assertTrue(close([p["act_lr"] for p in preview_points(None, "lr / 10", 2, 0.3, 0.005)], [0.03, 0.03]))

    def test_reverse_plays_the_schedule_backwards(self):
        forward = preview_points("linear(lr0, 5 * lr0)", "lr / 10", 5, 0.1, 0.005)
        backward = preview_points("linear(lr0, 5 * lr0)", "lr / 10", 5, 0.1, 0.005, reverse=True)
        self.assertEqual([p["epoch"] for p in backward], [1, 2, 3, 4, 5])
        self.assertTrue(close([p["lr"] for p in backward], [0.5, 0.4, 0.3, 0.2, 0.1]))
        self.assertTrue(close([p["act_lr"] for p in backward], [0.05, 0.04, 0.03, 0.02, 0.01]))
        self.assertEqual([(p["lr"], p["act_lr"]) for p in backward], [(p["lr"], p["act_lr"]) for p in reversed(forward)])
        self.assertEqual(preview_points(None, None, 3, 0.1, 0.01, reverse=True), preview_points(None, None, 3, 0.1, 0.01))
        self.assertEqual(preview_points("lr0", None, 0, 0.1, 0.01, reverse=True), [])
        cfg = TrainConfig(epochs=4, lr=0.1, act_lr=0.01, lr_schedule="warmup(lr0 / 10, lr0, 3)", reverse_schedule=True)
        cfg.validate()
        self.assertTrue(close([r[0] for r in cfg.rates()], [0.1, 0.1 * 2 / 3 + 0.01 / 3, 0.1 / 3 + 0.01 * 2 / 3, 0.01]))
        self.assertIn("reverse_schedule", cfg.to_dict())

    def test_preview_points_errors(self):
        with self.assertRaises(ScheduleError):
            preview_points("lr0 - 1", None, 3, 0.05, 0.005)
        with self.assertRaises(ScheduleError):
            preview_points(None, "nope", 3, 0.05, 0.005)
        with self.assertRaises(ScheduleError):
            preview_points(None, None, -1, 0.05, 0.005)

    def test_presets_evaluate_and_start_at_the_base_rate(self):
        self.assertGreaterEqual(len(PRESETS), 5)
        names = [p["name"] for p in PRESETS]
        self.assertEqual(len(names), len(set(names)))
        for preset in PRESETS:
            with self.subTest(preset=preset["name"]):
                self.assertEqual(set(preset), {"name", "lr", "act_lr", "description"})
                for epochs in (1, 2, 6):
                    points = preview_points(preset["lr"], preset["act_lr"], epochs, 0.05, 0.005)
                    self.assertEqual(len(points), epochs)
                    self.assertTrue(all(math.isfinite(p["lr"]) and p["lr"] >= 0 for p in points))
                    self.assertTrue(all(math.isfinite(p["act_lr"]) and p["act_lr"] >= 0 for p in points))
                    # every preset is non-decreasing: "increase the learning rate per epoch"
                    lrs = [p["lr"] for p in points]
                    acts = [p["act_lr"] for p in points]
                    self.assertEqual(lrs, sorted(lrs))
                    self.assertEqual(acts, sorted(acts))
                    if preset["name"].startswith("warm-up"):
                        self.assertAlmostEqual(points[-1]["lr"], 0.05 if epochs > 3 else 0.05 * (epochs - 1) / 3 + 0.005 * (1 - (epochs - 1) / 3), places=12)
                    else:
                        self.assertGreaterEqual(points[0]["lr"], 0.05 - 1e-12)

    def test_x4_presets_reach_four_times_the_base(self):
        for name in ("linear ramp x4", "exponential ramp x4", "cosine ramp x4"):
            preset = next(p for p in PRESETS if p["name"] == name)
            points = preview_points(preset["lr"], preset["act_lr"], 7, 0.05, 0.005)
            self.assertAlmostEqual(points[0]["lr"], 0.05)
            self.assertAlmostEqual(points[-1]["lr"], 0.2)
            self.assertAlmostEqual(points[0]["act_lr"], 0.005)
            self.assertAlmostEqual(points[-1]["act_lr"], 0.02)

    def test_describe(self):
        info = describe()
        self.assertEqual(set(info), {"variables", "constants", "functions", "helpers", "presets"})
        self.assertEqual(info["variables"], list(sched.VARIABLES))
        self.assertEqual(info["constants"], ["e", "pi"])
        self.assertIn("clamp", info["functions"])
        self.assertEqual(len(info["helpers"]), len(sched.HELPERS))
        for helper, line in zip(sched.HELPERS, info["helpers"]):
            self.assertTrue(line.startswith(helper + "("), line)
        self.assertEqual(info["presets"], [dict(p) for p in PRESETS])
        info["presets"][0]["name"] = "changed"  # a copy, not the module's tuple
        self.assertEqual(PRESETS[0]["name"], "constant")


# ---------------------------------------------------------------------------
# TrainConfig and training
# ---------------------------------------------------------------------------


class TestTrainConfig(unittest.TestCase):
    def test_defaults_are_constant(self):
        cfg = TrainConfig(epochs=3, lr=0.2, act_lr=0.02)
        self.assertIsNone(cfg.lr_schedule)
        self.assertIsNone(cfg.act_lr_schedule)
        self.assertEqual(cfg.rates(), [(0.2, 0.02)] * 3)
        self.assertEqual(TrainConfig(epochs=0).rates(), [])
        self.assertIn("lr_schedule", cfg.to_dict())

    def test_rates_follow_the_schedules(self):
        cfg = TrainConfig(epochs=5, lr=0.1, act_lr=0.005, lr_schedule="linear(lr0, 5 * lr0)", act_lr_schedule="lr / 10")
        cfg.validate()
        rates = cfg.rates()
        self.assertTrue(close([r[0] for r in rates], [0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertTrue(close([r[1] for r in rates], [0.01, 0.02, 0.03, 0.04, 0.05]))
        only_act = TrainConfig(epochs=2, lr=0.1, act_lr=0.005, act_lr_schedule="act_lr0 * 3")
        self.assertTrue(close([r[0] for r in only_act.rates()], [0.1, 0.1]))
        self.assertTrue(close([r[1] for r in only_act.rates()], [0.015, 0.015]))

    def test_validate_rejects_bad_schedules(self):
        with self.assertRaises(ValueError) as ctx:
            TrainConfig(epochs=3, lr_schedule="lr0 - 1").validate()
        self.assertIsInstance(ctx.exception, ScheduleError)
        with self.assertRaises(ValueError):
            TrainConfig(epochs=3, act_lr_schedule="import os").validate()
        with self.assertRaises(ValueError) as ctx:
            TrainConfig(epochs=3, lr_schedule=5).validate()
        self.assertIn("lr_schedule must be a string expression", str(ctx.exception))
        with self.assertRaises(ValueError):
            TrainConfig(epochs=3, act_lr_schedule=["lr0"]).validate()
        TrainConfig(epochs=3, lr_schedule="lr0 if epoch < 5 else lr0 - 1").validate()  # epoch 5 is never reached
        with self.assertRaises(ValueError):
            TrainConfig(epochs=5, lr_schedule="lr0 if epoch < 5 else lr0 - 1").validate()

    def test_train_records_carry_the_epoch_rates(self):
        model = RadixNet(seed=1, backend="python")
        records = model.train(
            CORPUS[:12], epochs=5, lr=0.1, act_lr=0.005, batch_size=8,
            lr_schedule="linear(lr0, 5 * lr0)", act_lr_schedule="lr / 10",
        )
        self.assertEqual([r["epoch"] for r in records], [1, 2, 3, 4, 5])
        self.assertTrue(close([r["lr"] for r in records], [0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertTrue(close([r["act_lr"] for r in records], [0.01, 0.02, 0.03, 0.04, 0.05]))
        self.assertEqual(model.history[-5:], records)

    def test_constant_training_records_the_base_rates(self):
        model = RadixNet(seed=1, backend="python")
        records = model.train(CORPUS[:6], epochs=2, lr=0.3, act_lr=0.02, batch_size=8)
        self.assertEqual([(r["lr"], r["act_lr"]) for r in records], [(0.3, 0.02), (0.3, 0.02)])

    def test_schedule_changes_what_is_learned(self):
        # the same seed and data with a growing rate ends in a different state than the constant one
        constant = RadixNet(seed=3, backend="python")
        constant.train(CORPUS[:12], epochs=4, lr=0.1, batch_size=8)
        growing = RadixNet(seed=3, backend="python")
        growing.train(CORPUS[:12], epochs=4, lr=0.1, batch_size=8, lr_schedule="lr0 * 3 ** i")
        self.assertNotEqual(constant.score(CORPUS[0]), growing.score(CORPUS[0]))
        # ... while a schedule that evaluates to the base rate reproduces the constant run exactly
        same = RadixNet(seed=3, backend="python")
        same.train(CORPUS[:12], epochs=4, lr=0.1, batch_size=8, lr_schedule="lr0 * 1", act_lr_schedule="act_lr0")
        self.assertEqual(constant.score(CORPUS[0]), same.score(CORPUS[0]))

    def test_bad_schedule_leaves_the_model_untouched(self):
        model = RadixNet(seed=1, backend="python")
        with self.assertRaises(ValueError):
            model.train(CORPUS[:4], epochs=2, lr_schedule="nope")
        self.assertEqual(model.history, [])
        self.assertEqual(model.stats()["trained_texts"], 0)

    def test_two_nrl_accepts_schedules_in_the_positive_phase(self):
        model = RadixNet(seed=1, backend="python")
        result = model.two_nrl(
            _lines("sample_garbage.txt")[:4], CORPUS[:4], neg_epochs=1, pos_epochs=3, batch_size=8,
            lr_schedule="linear(lr0, 2 * lr0)",
        )
        positive = result["positive"]
        self.assertEqual([r["phase"] for r in positive], ["positive"] * 3)
        self.assertEqual(len(result["negative"]), 1)  # a single negative epoch: the schedule gives its base rate
        self.assertTrue(close([r["lr"] for r in positive], [positive[0]["lr"], positive[0]["lr"] * 1.5, positive[0]["lr"] * 2]))


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup)

    def test_describe_endpoint(self):
        status, data, _ = self.client.get("/api/schedule")
        self.assertEqual(status, 200)
        self.assertEqual(data, describe())

    def test_preview_endpoint(self):
        status, data, _ = self.client.post(
            "/api/schedule/preview",
            {"lr_schedule": "linear(lr0, 5 * lr0)", "act_lr_schedule": "lr / 10", "epochs": 5, "lr": 0.1, "act_lr": 0.005},
        )
        self.assertEqual(status, 200, data)
        self.assertEqual(set(data), {"lr_schedule", "act_lr_schedule", "epochs", "lr", "act_lr", "reverse_schedule", "points"})
        self.assertEqual((data["lr_schedule"], data["act_lr_schedule"], data["epochs"]), ("linear(lr0, 5 * lr0)", "lr / 10", 5))
        self.assertEqual([p["epoch"] for p in data["points"]], [1, 2, 3, 4, 5])
        self.assertTrue(close([p["lr"] for p in data["points"]], [0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertTrue(close([p["act_lr"] for p in data["points"]], [0.01, 0.02, 0.03, 0.04, 0.05]))

    def test_preview_defaults_and_blank_schedules(self):
        status, data, _ = self.client.post("/api/schedule/preview", {})
        self.assertEqual(status, 200)
        self.assertEqual((data["lr_schedule"], data["act_lr_schedule"], data["epochs"]), (None, None, TrainConfig.epochs))
        self.assertEqual(len(data["points"]), TrainConfig.epochs)
        self.assertEqual(data["points"][0], {"epoch": 1, "lr": TrainConfig.lr, "act_lr": TrainConfig.act_lr})
        status, data, _ = self.client.post("/api/schedule/preview", {"lr_schedule": "   ", "act_lr_schedule": None, "epochs": 0})
        self.assertEqual(status, 200)
        self.assertEqual((data["lr_schedule"], data["points"]), (None, []))

    def test_preview_errors(self):
        for body, fragment in (
            ({"lr_schedule": "lr0 - 1"}, "must be finite and >= 0"),
            ({"act_lr_schedule": "__import__('os')"}, "only these functions"),
            ({"lr_schedule": "lr0 *"}, "invalid schedule expression"),
            ({"lr_schedule": 5}, "'lr_schedule' must be a string"),
            ({"epochs": -1}, "'epochs' must be >= 0"),
            ({"lr": -0.1}, "'lr' must be >= 0"),
        ):
            with self.subTest(body=body):
                status, data, _ = self.client.post("/api/schedule/preview", body)
                self.assertEqual(status, 400, data)
                self.assertIn(fragment, data["error"])

    def test_train_with_schedules(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.post(
            "/api/train",
            {"texts": CORPUS[:12], "epochs": 4, "lr": 0.1, "act_lr": 0.005, "batch_size": 8,
             "lr_schedule": "geometric(lr0, 8 * lr0)", "act_lr_schedule": "lr / 10"},
        )
        self.assertEqual(status, 202, data)
        done = wait_for_job(self.client)
        self.assertEqual(done["state"], "done", done)
        self.assertTrue(close([r["lr"] for r in done["history"]], [0.1, 0.2, 0.4, 0.8]))
        self.assertTrue(close([r["act_lr"] for r in done["history"]], [0.01, 0.02, 0.04, 0.08]))
        status, data, _ = self.client.post(
            "/api/train",
            {"texts": CORPUS[:12], "epochs": 4, "lr": 0.1, "act_lr": 0.005, "batch_size": 8,
             "lr_schedule": "geometric(lr0, 8 * lr0)", "act_lr_schedule": "lr / 10", "reverse_schedule": True},
        )
        self.assertEqual(status, 202, data)
        done = wait_for_job(self.client)
        self.assertTrue(close([r["lr"] for r in done["history"]], [0.8, 0.4, 0.2, 0.1]))
        self.assertTrue(close([r["act_lr"] for r in done["history"]], [0.08, 0.04, 0.02, 0.01]))

    def test_preview_reverse(self):
        status, data, _ = self.client.post(
            "/api/schedule/preview", {"lr_schedule": "linear(lr0, 3 * lr0)", "epochs": 3, "lr": 0.1, "reverse_schedule": True},
        )
        self.assertEqual(status, 200, data)
        self.assertTrue(data["reverse_schedule"])
        self.assertTrue(close([p["lr"] for p in data["points"]], [0.3, 0.2, 0.1]))
        status, data, _ = self.client.post("/api/schedule/preview", {"lr_schedule": "lr0", "reverse_schedule": "yes"})
        self.assertEqual(status, 400)

    def test_train_rejects_a_bad_schedule_without_starting_a_job(self):
        status, before, _ = self.client.get("/api/job")
        self.assertEqual(status, 200)
        self.assertTrue(before is None or before["state"] != "running")
        status, data, _ = self.client.post("/api/train", {"texts": CORPUS[:4], "epochs": 2, "lr_schedule": "lr0 - 1"})
        self.assertEqual(status, 400, data)
        self.assertIn("must be finite and >= 0", data["error"])
        status, data, _ = self.client.post("/api/train", {"texts": CORPUS[:4], "epochs": 2, "act_lr_schedule": 7})
        self.assertEqual(status, 400, data)
        self.assertIn("'act_lr_schedule' must be a string", data["error"])
        status, after, _ = self.client.get("/api/job")
        self.assertEqual(after, before)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_cli import run_cli, run_json

        cls.run_cli = staticmethod(run_cli)
        cls.run_json = staticmethod(run_json)

    def test_schedule_lists_presets_without_expressions(self):
        doc = self.run_json("schedule")
        self.assertEqual(set(doc), {"presets", "variables", "functions", "helpers"})
        self.assertEqual(doc["presets"], [dict(p) for p in PRESETS])
        proc = self.run_cli("schedule", json_mode=False)
        self.assertIn("linear ramp x4", proc.stdout)
        self.assertIn("warmup(a, b, n)", proc.stdout)

    def test_schedule_previews_a_table_and_graph(self):
        doc = self.run_json("schedule", "--epochs", 4, "--lr", 0.1, "--act-lr", 0.005,
                            "--lr-schedule", "linear(lr0, 4 * lr0)", "--act-lr-schedule", "lr / 10")
        self.assertEqual((doc["lr_schedule"], doc["act_lr_schedule"], doc["epochs"], doc["lr"], doc["act_lr"]),
                         ("linear(lr0, 4 * lr0)", "lr / 10", 4, 0.1, 0.005))
        self.assertTrue(close([p["lr"] for p in doc["points"]], [0.1, 0.2, 0.3, 0.4]))
        self.assertTrue(close([p["act_lr"] for p in doc["points"]], [0.01, 0.02, 0.03, 0.04]))
        doc = self.run_json("schedule", "--epochs", 4, "--lr", 0.1, "--lr-schedule", "linear(lr0, 4 * lr0)", "--reverse-schedule")
        self.assertTrue(doc["reverse_schedule"])
        self.assertTrue(close([p["lr"] for p in doc["points"]], [0.4, 0.3, 0.2, 0.1]))
        proc = self.run_cli("schedule", "--epochs", 4, "--lr-schedule", "linear(lr0, 4 * lr0)", json_mode=False)
        self.assertIn("lr graph", proc.stdout)
        bars = [line.split()[2] for line in proc.stdout.splitlines() if line.strip()[:1].isdigit()]
        self.assertEqual(len(bars), 4)
        self.assertEqual(bars[-1], "#" * 30)
        self.assertTrue(all(set(b) == {"#"} for b in bars))
        self.assertEqual([len(b) for b in bars], sorted(len(b) for b in bars))

    def test_schedule_errors(self):
        proc = self.run_cli("schedule", "--lr-schedule", "lr0 - 1", expect=1)
        self.assertIn("must be finite and >= 0", proc.stderr)
        proc = self.run_cli("schedule", "--act-lr-schedule", "open('x')", expect=1)
        self.assertIn("only these functions may be called", proc.stderr)
        proc = self.run_cli("schedule", "--epochs", -1, expect=1)  # usage errors exit 1 in this CLI
        self.assertIn("usage", proc.stderr)
        self.assertIn("must be >= 0", proc.stderr)

    def test_train_with_schedules(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="radixnet-schedule-") as tmp:
            model = os.path.join(tmp, "model.json")
            doc = self.run_json(
                "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 3, "--batch-size", 8,
                "--lr", 0.1, "--act-lr", 0.005, "--lr-schedule", "lr0 * 2 ** i", "--act-lr-schedule", "lr / 10", model=model,
            )
            self.assertEqual(doc["config"]["lr_schedule"], "lr0 * 2 ** i")
            self.assertEqual(doc["config"]["act_lr_schedule"], "lr / 10")
            self.assertTrue(close([r["lr"] for r in doc["records"]], [0.1, 0.2, 0.4]))
            self.assertTrue(close([r["act_lr"] for r in doc["records"]], [0.01, 0.02, 0.04]))
            self.assertTrue(os.path.exists(model))
            doc = self.run_json(
                "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 3, "--batch-size", 8,
                "--lr", 0.1, "--lr-schedule", "lr0 * 2 ** i", "--reverse-schedule", model=os.path.join(tmp, "rev.json"),
            )
            self.assertTrue(doc["config"]["reverse_schedule"])
            self.assertTrue(close([r["lr"] for r in doc["records"]], [0.4, 0.2, 0.1]))
            proc = self.run_cli(
                "train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 2, "--batch-size", 8,
                "--lr-schedule", "linear(lr0, 3 * lr0)", model=model, json_mode=False,
            )
            self.assertIn("lr_schedule=linear(lr0, 3 * lr0)", proc.stdout)
            self.assertIn("act_lr", proc.stdout)  # the epoch table has lr / act_lr columns
            proc = self.run_cli("train", "--data", os.path.join(ROOT, "data", "sample_corpus.txt"), "--epochs", 2,
                                "--lr-schedule", "lr0 - 1", model=os.path.join(tmp, "other.json"), expect=1)
            self.assertIn("must be finite and >= 0", proc.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "other.json")))


if __name__ == "__main__":
    unittest.main()
