"""The reference implementation's tests: the matrix, the edge, the machine, credit, stimulation, time, persistence,
the experiments, the CLI, and parity with the Rust crate when it is built."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from latticefsm import (  # noqa: E402
    COEFFICIENT_LIMIT, DEFAULT_ALPHABET, DEFAULT_STATES, FEATURES, LANGUAGES, WIDTH_MAX, WIDTH_MIN, WIDTH_REST, Edge,
    Machine, Weighting, examples, language, load_machine,
)
from latticefsm import experiment  # noqa: E402
from latticefsm.lattice import Lattice  # noqa: E402

RUST = os.path.join(ROOT, "rust", "target", "release", "latticefsm")


class TestLattice(unittest.TestCase):
    def test_the_matrix_is_dense_and_three_dimensional(self):
        lat = Lattice(4, "abc")
        self.assertEqual(lat.shape, (4, 3, 4))
        self.assertEqual(len(lat), 48)
        for s in range(4):
            for a in range(3):
                for t in range(4):
                    e = lat[s, a, t]
                    self.assertEqual((e.source, e.symbol, e.target), (s, a, t))
                    self.assertIsInstance(e, Edge)
                    self.assertFalse(e.touched)
        self.assertEqual([e.target for e in lat.row(2, "b")], [0, 1, 2, 3])
        self.assertEqual(len(lat.leaving(1)), 12)
        self.assertEqual(sum(1 for _ in lat.arriving(3)), 12)
        self.assertTrue(all(e.target == 3 for e in lat.arriving(3)))

    def test_the_default_matrix_is_13_by_13_by_13(self):
        self.assertEqual((DEFAULT_STATES, len(DEFAULT_ALPHABET)), (13, 13))
        m = Machine()
        self.assertEqual(m.lattice.shape, (13, 13, 13))
        self.assertEqual(len(m.lattice), 13 ** 3)
        self.assertEqual("".join(m.alphabet), "abcdefghijklm")
        self.assertEqual(m.stats()["shape"], [13, 13, 13])
        for lang in LANGUAGES.values():           # every language is over a and b, which the default alphabet holds
            self.assertTrue(all(s < DEFAULT_STATES for s in lang.accepting))
        self.assertEqual(len(m.run("abm").transitions), 3)

    def test_every_edge_has_its_own_weighting(self):
        lat = Lattice(2, "a", prototype=Weighting(bias=0.5))
        a, b = lat[0, "a", 0], lat[0, "a", 1]
        self.assertEqual(a.weighting, b.weighting)
        self.assertIsNot(a.weighting, b.weighting)
        a.weighting.bias = 3.0
        self.assertEqual(b.weighting.bias, 0.5)
        self.assertEqual(lat.prototype.bias, 0.5)

    def test_bad_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            Lattice(0, "ab")
        with self.assertRaises(ValueError):
            Lattice(2, "aa")
        with self.assertRaises(ValueError):
            Lattice(2, "")
        lat = Lattice(2, "ab")
        with self.assertRaises(IndexError):
            lat[2, "a", 0]
        with self.assertRaises(KeyError):
            lat[0, "z", 0]


class TestEdge(unittest.TestCase):
    def test_a_fresh_edge_weighs_nothing(self):
        e = Edge(0, 0, 0)
        self.assertEqual(e.features_at(0, 1000.0), (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(e.log_weight(0, 1000.0, stimulation=3.0), 0.0)
        self.assertEqual(len(FEATURES), 5)

    def test_traversal_writes_the_history_and_the_trace(self):
        e = Edge(0, 0, 0)
        e.traverse(clock=10, life=100.0)
        self.assertEqual((e.seen, e.first_seen, e.last_seen), (1, 10, 10))
        self.assertEqual(e.recent_at(10, 100.0), 1.0)
        self.assertAlmostEqual(e.recent_at(110, 100.0), 0.5)
        self.assertAlmostEqual(e.age_at(110, 100.0), 0.5)
        e.traverse(clock=110, life=100.0)
        self.assertEqual((e.seen, e.first_seen, e.last_seen), (2, 10, 110))
        self.assertAlmostEqual(e.recent_at(110, 100.0), 1.5)
        self.assertEqual(e.features, (math.log1p(1), 0.5, 0.0, 0.5, 0.0))

    def test_credit_writes_the_verdict_and_adapts_the_weighting(self):
        e = Edge(0, 0, 0)
        e.traverse(0, 100.0)
        e.credit(+1.0, clock=5, life=100.0)
        self.assertEqual((e.rewarded, e.punished, e.last_rewarded, e.last_punished), (1.0, 0.0, 5, -1))
        self.assertAlmostEqual(e.net, 0.5)
        self.assertAlmostEqual(e.weighting.bias, 0.1)
        e.credit(-2.0, clock=6, life=100.0)
        self.assertEqual((e.rewarded, e.punished, e.last_rewarded, e.last_punished), (1.0, 2.0, 5, 6))
        self.assertAlmostEqual(e.weighting.bias, -0.1)
        e.credit(0.0, clock=7, life=100.0)
        self.assertEqual(e.last_punished, 6)

    def test_the_adaptation_follows_the_features_at_traversal(self):
        e = Edge(0, 0, 0)
        e.traverse(0, 100.0)
        e.traverse(0, 100.0)          # now features[0] = log1p(1), features[1] = recent 1.0, features[3] = age 1.0
        f = e.features
        e.credit(1.0, clock=1, life=100.0)
        self.assertAlmostEqual(e.weighting.seen, 0.1 * f[0])
        self.assertAlmostEqual(e.weighting.recent, 0.1 * f[1])
        self.assertAlmostEqual(e.weighting.age, 0.1 * f[3])

    def test_coefficients_are_clipped(self):
        w = Weighting(rate=100.0)
        w.adapt(1.0, (1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(w.to_list()[:6], [COEFFICIENT_LIMIT] * 6)
        w.adapt(-1.0, (1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(w.to_list()[:6], [-COEFFICIENT_LIMIT] * 6)

    def test_width_widens_narrows_relaxes_and_is_bounded(self):
        e = Edge(0, 0, 0)
        self.assertEqual(e.width_at(10_000, 100.0), WIDTH_REST)
        e.traverse(0, 100.0, widen=0.5)
        self.assertAlmostEqual(e.width_at(0, 100.0), 1.5)
        self.assertAlmostEqual(e.width_at(100, 100.0), 1.25)      # the excess halves in a life
        e.credit(1.0, clock=0, life=100.0, widen=1.0)
        self.assertAlmostEqual(e.width_at(0, 100.0), 3.0)
        e.credit(-1.0, clock=0, life=100.0, narrow=1.0)
        self.assertAlmostEqual(e.width_at(0, 100.0), 1.5)
        e.set_width(1e9, 0)
        self.assertEqual(e.width, WIDTH_MAX)
        e.set_width(0.0, 0)
        self.assertEqual(e.width, WIDTH_MIN)

    def test_the_stimulation_term_is_stimulation_times_log_width(self):
        wide, narrow = Edge(0, 0, 0), Edge(0, 0, 1)
        wide.set_width(4.0, 0)
        for stim in (0.0, 0.5, 1.0, 3.0):
            self.assertAlmostEqual(wide.log_weight(0, 100.0, stim) - narrow.log_weight(0, 100.0, stim),
                                   stim * math.log(4.0))

    def test_round_trip(self):
        e = Edge(1, 2, 3)
        e.traverse(4, 100.0, widen=0.1)
        e.credit(0.7, 5, 100.0, widen=0.2)
        e.credit(-0.3, 6, 100.0, narrow=0.2)
        self.assertEqual(Edge.from_list(json.loads(json.dumps(e.to_list()))), e)


class TestMachine(unittest.TestCase):
    def test_a_run_traverses_one_edge_per_symbol_and_ticks_the_clock(self):
        m = Machine(3, "ab", accepting=[0], seed=1)
        r = m.run("abba")
        self.assertEqual(len(r.transitions), 4)
        self.assertEqual(m.clock, 4)
        self.assertEqual(len(m.path), 4)
        self.assertEqual(r.states[0], 0)
        self.assertEqual(r.final, r.states[-1])
        self.assertEqual(r.accepted, r.final == 0)
        self.assertEqual(sum(e.seen for e in m.lattice), 4)
        for t, e in zip(r.transitions, m.path):
            self.assertEqual((e.source, e.target), (t.source, t.target))
            self.assertEqual(m.alphabet[e.symbol], t.symbol)
        self.assertAlmostEqual(r.log_probability, 4 * math.log(1 / 3))

    def test_a_quiet_run_moves_nothing(self):
        m = Machine(3, "ab", accepting=[0], seed=1)
        before = m.to_dict()
        r = m.run("abba", quiet=True)
        self.assertEqual(len(r.transitions), 4)
        after = m.to_dict()
        before.pop("rng"), after.pop("rng")
        self.assertEqual(before, after)
        self.assertEqual(m.credit(1.0), 0)      # nothing to land on

    def test_a_fresh_row_is_uniform_and_greedy_is_a_tie(self):
        m = Machine(4, "a")
        self.assertEqual(m.probabilities(0, "a"), [0.25] * 4)
        self.assertEqual(m.probabilities(0, "a", temperature=0), [0.25] * 4)
        m.edge(0, "a", 2).weighting.bias = 1.0
        self.assertEqual(m.probabilities(0, "a", temperature=0), [0, 0, 1, 0])
        self.assertEqual(m.transition_table()[(0, "a")], 2)

    def test_credit_is_discounted_back_from_the_end(self):
        m = Machine(3, "ab", discount=0.5, seed=1)
        m.run("aba")
        last, mid, first = m.path[2], m.path[1], m.path[0]
        self.assertEqual(m.reward(2.0), 3)
        self.assertAlmostEqual(last.rewarded, 2.0)
        self.assertAlmostEqual(mid.rewarded, 1.0)
        self.assertAlmostEqual(first.rewarded, 0.5)
        self.assertEqual(m.punish(1.0), 3)
        self.assertAlmostEqual(last.punished, 1.0)
        self.assertAlmostEqual(first.punished, 0.25)
        self.assertEqual(m.credits, 2)

    def test_the_same_edge_twice_in_a_run_is_credited_twice(self):
        m = Machine(1, "a", seed=1)
        m.run("aaa")
        e = m.edge(0, "a", 0)
        m.reward(1.0)
        self.assertAlmostEqual(e.rewarded, 1 + 0.8 + 0.64)
        self.assertEqual(e.seen, 3)

    def test_reward_moves_probability_toward_the_path_and_punishment_away(self):
        m = Machine(3, "a", seed=3)
        r = m.run("a")
        t = r.transitions[0].target
        p0 = m.probabilities(0, "a")[t]
        m.reward(1.0)
        p1 = m.probabilities(0, "a")[t]
        self.assertGreater(p1, p0)
        m.run("a")
        t2 = m.path[0].target
        before = m.probabilities(0, "a")[t2]
        m.punish(1.0)
        self.assertLess(m.probabilities(0, "a")[t2], before)

    def test_teach_is_one_traversal_and_one_credit(self):
        m = Machine(3, "ab")
        e = m.teach(1, "b", 2, amount=-0.5)
        self.assertIs(e, m.edge(1, "b", 2))
        self.assertEqual((e.seen, e.punished, m.clock), (1, 0.5, 1))
        self.assertEqual(m.state, 0)      # the machine did not move

    def test_stimulation_prefers_the_wide_channel_more_the_higher_it_is(self):
        m = Machine(3, "a")
        m.edge(0, "a", 0).set_width(4.0, 0)
        m.edge(0, "a", 2).set_width(0.25, 0)
        wide = [m.probabilities(0, "a", stimulation=s)[0] for s in (0.0, 0.5, 1.0, 2.0, 4.0)]
        self.assertAlmostEqual(wide[0], 1 / 3)
        self.assertEqual(wide, sorted(wide))
        self.assertAlmostEqual(m.probabilities(0, "a", stimulation=1.0)[0], 4 / 5.25)   # width in proportion
        self.assertGreater(wide[-1], 0.99)

    def test_stimulation_relaxes_to_the_baseline_on_the_clock(self):
        m = Machine(2, "a", life=100.0, baseline=1.0)
        self.assertEqual(m.stimulation, 1.0)
        self.assertEqual(m.stimulate(3.0), 4.0)
        m.tick(100)
        self.assertAlmostEqual(m.stimulation, 2.5)
        m.tick(100)
        self.assertAlmostEqual(m.stimulation, 1.75)
        m.stimulate(-10.0)
        self.assertEqual(m.stimulation, 0.0)
        m.tick(100)
        self.assertAlmostEqual(m.stimulation, 0.5)
        m2 = Machine(2, "a", life=100.0, calm=10.0)
        m2.stimulate(1.0)
        m2.tick(10)
        self.assertAlmostEqual(m2.stimulation, 1.5)
        with self.assertRaises(ValueError):
            m.stimulation = -1.0

    def test_a_run_at_a_stimulation_uses_it_and_records_it(self):
        m = Machine(2, "a", seed=1)
        m.edge(0, "a", 1).set_width(8.0, 0)
        r = m.run("a", stimulation=0.0)
        self.assertEqual(r.transitions[0].stimulation, 0.0)
        self.assertAlmostEqual(r.transitions[0].probability, 0.5)
        r = m.run("a", stimulation=5.0, temperature=0)
        self.assertEqual(r.transitions[0].target, 1)
        self.assertEqual(r.transitions[0].probability, 1.0)

    def test_a_custom_weight_function_replaces_the_edges_own(self):
        m = Machine(3, "a", weight_fn=lambda e, clock, life, stim: float(e.target == 1) * 10.0)
        self.assertGreater(m.probabilities(0, "a")[1], 0.99)
        m.edge(0, "a", 0).weighting.bias = 100.0
        self.assertGreater(m.probabilities(0, "a")[1], 0.99)   # the edge's own weighting is not consulted

    def test_time_fades_traces_and_widths_but_not_verdicts(self):
        m = Machine(2, "a", life=100.0)
        e = m.teach(0, "a", 1, 1.0)
        self.assertEqual(e.recent_at(e.last_seen, m.life), 1.0)
        w = e.width_at(m.clock, m.life)
        self.assertGreater(w, 1.0)
        m.tick(1000)
        self.assertLess(e.recent_at(m.clock, m.life), 0.01)
        self.assertLess(e.width_at(m.clock, m.life) - 1.0, (w - 1.0) * 0.001)
        self.assertEqual((e.rewarded, e.seen), (1.0, 1))
        self.assertAlmostEqual(e.net, 0.5)
        with self.assertRaises(ValueError):
            m.tick(-1)

    def test_the_states_count_their_visits(self):
        m = Machine(2, "a", seed=1)
        m.run("aaa")
        self.assertEqual(sum(s.visits for s in m.lattice.states), 4)    # the start, then one per step
        self.assertEqual(max(s.last_visited for s in m.lattice.states), 3)

    def test_invalid_settings_are_refused(self):
        for kw in (dict(life=0), dict(baseline=-1), dict(discount=1.5), dict(temperature=-1), dict(start=5)):
            with self.assertRaises(ValueError, msg=kw):
                Machine(2, "a", **kw)
        with self.assertRaises(IndexError):
            Machine(2, "a", accepting=[7])

    def test_stats_and_accuracy_are_quiet(self):
        m = Machine(3, "ab", accepting=[0], seed=1)
        m.run("ab")
        m.reward(1.0)
        before = m.to_dict()
        m.stats()
        m.accuracy([("ab", True), ("ba", False)])
        m.transition_table()
        m.accepts("aaa")
        after = m.to_dict()
        before.pop("rng"), after.pop("rng")
        self.assertEqual(before, after)


class TestLearning(unittest.TestCase):
    def test_a_language_is_learned_by_credit_and_not_quietly(self):
        lang = language("even-b")
        for quiet in (False, True):
            rng = random.Random(1)
            m = Machine(3, "ab", accepting=lang.accepting, seed=1)
            test = examples(lang, 200, rng)
            experiment.teach_language(m, "even-b", 1500, rng, quiet=quiet)
            acc = m.accuracy(test)
            if quiet:
                self.assertLess(acc, 0.75)
                self.assertEqual(m.stats()["touched"], 0)
            else:
                self.assertGreater(acc, 0.95)
                self.assertGreater(m.credits, 1200)     # every run but the empty strings' lands somewhere

    def test_the_learned_table_is_the_language(self):
        rng = random.Random(2)
        m = Machine(2, "ab", accepting=[0], seed=2)
        experiment.teach_language(m, "even-b", 1500, rng)
        t = m.transition_table()
        self.assertEqual((t[(0, "a")], t[(1, "a")]), (0, 1))
        self.assertEqual((t[(0, "b")], t[(1, "b")]), (1, 0))

    def test_the_experiments_run_and_say_what_the_readme_says(self):
        stim = experiment.stimulation()
        curve = [c["probabilities"][0] for c in stim["curve"]]
        self.assertAlmostEqual(curve[0], 1 / 3)
        self.assertEqual(curve, sorted(curve))
        relax = [r["stimulation"] for r in stim["relaxation"]]
        self.assertEqual(relax, sorted(relax, reverse=True))
        adapt = experiment.adaptation(uses=5, life=100.0, waits=(0.0, 1.0))
        last = adapt["stages"][-1]["edges"]
        self.assertGreater(last["rewarded"]["net"], 0)
        self.assertLess(last["punished"]["net"], 0)
        self.assertNotEqual(last["rewarded"]["weighting"], last["punished"]["weighting"])
        self.assertIn("| stage |", experiment.tables(adapt))
        learn = experiment.learning(episodes=300, sizes=(3,), seeds=(1,), tests=50)
        self.assertEqual(len(learn["rows"]), 4)
        self.assertIn("| language |", experiment.tables(learn))


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        m = Machine(3, "ab", accepting=[2], life=500.0, seed=7)
        rng = random.Random(7)
        experiment.teach_language(m, "ends-ab", 200, rng)
        m.stimulate(2.0)
        m.tick(50)
        with tempfile.TemporaryDirectory() as d:
            for name in ("m.json", "m.json.gz"):
                path = os.path.join(d, name)
                m.save(path)
                m2 = load_machine(path)
                self.assertEqual(m2.to_dict(), m.to_dict())
                self.assertEqual(m2.stats(), m.stats())
                self.assertEqual(m2.transition_table(), m.transition_table())
                self.assertEqual(m2.run("abab").states, m.run("abab").states)   # the same next draws

    def test_only_touched_edges_are_written(self):
        m = Machine(5, "abc")
        self.assertEqual(m.to_dict()["lattice"]["edges"], [])
        m.teach(0, "a", 1, 1.0)
        self.assertEqual(len(m.to_dict()["lattice"]["edges"]), 1)
        self.assertEqual(len(load_machine_from(m).lattice), 75)

    def test_a_file_from_a_foreign_generator_reseeds(self):
        m = Machine(2, "a", seed=3)
        d = m.to_dict()
        d["rng"] = {"xoshiro256": ["1", "2", "3", "4"]}
        m2 = Machine.from_dict(d)
        self.assertEqual(m2.seed, 3)
        self.assertEqual(m2.rng.random(), random.Random(3).random())


def load_machine_from(m: Machine) -> Machine:
    return Machine.from_dict(json.loads(json.dumps(m.to_dict())))


class TestCLI(unittest.TestCase):
    def run_cli(self, *args: str) -> str:
        out = subprocess.run([sys.executable, "-m", "latticefsm", *args], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_every_command(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json.gz")
            self.assertIn("greedy table", self.run_cli("train", "--episodes", "300", "--save", path))
            self.assertIn("accepted", self.run_cli("run", "aa", "--load", path, "--quiet", "--temperature-run", "0")
                          .replace("rejected", "accepted"))
            self.assertIn("credited 2 edges", self.run_cli("run", "ab", "--load", path, "--credit", "1", "--save", path))
            self.assertIn("Edge(source=1, symbol=1, target=2", self.run_cli("teach", "1", "b", "2", "--load", path, "--amount", "-1"))
            self.assertIn("even-b on", self.run_cli("accuracy", "--load", path))
            self.assertIn("greedy table", self.run_cli("table", "--load", path))
            self.assertIn("clock:", self.run_cli("stats", "--load", path))
            self.assertIn("clock", self.run_cli("tick", "--ticks", "10", "--load", path))
            self.assertIn("stimulation 3.000", self.run_cli("stimulate", "--amount", "2", "--load", path))
            self.assertIn("| stimulation |", self.run_cli("experiment", "--which", "stimulation"))
            self.assertIn("stimulation prefers the wide channel", self.run_cli("demo", "--episodes", "200"))
            self.assertIn("shape: [13, 13, 13]", self.run_cli("stats"))


@unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
class TestRustParity(unittest.TestCase):
    """The two ports agree on everything that does not draw: the stimulation and adaptation records, number for
    number, and a machine file written by one reads in the other to the same table."""

    def test_the_deterministic_experiments_agree(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([RUST, "experiment", "--which", "stimulation,adaptation", "--out", d], check=True,
                           capture_output=True)
            for name, rec in (("stimulation", experiment.stimulation()), ("adaptation", experiment.adaptation())):
                with open(os.path.join(d, f"{name}_results.json")) as f:
                    rust = json.load(f)
                self.assertEqual(strip(rust), strip(rec), name)

    def test_a_machine_file_crosses(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json.gz")
            subprocess.run([RUST, "train", "--language", "even-b", "--states", "3", "--alphabet", "ab", "--episodes", "500",
                            "--save", path], check=True, capture_output=True)
            m = load_machine(path)
            out = subprocess.run([RUST, "table", "--load", path], check=True, capture_output=True, text=True).stdout
            for (s, sym), t in m.transition_table().items():
                self.assertIn(f"{sym} -> {t}", out.splitlines()[1 + s])
            path2 = os.path.join(d, "back.json")
            m.save(path2)
            out2 = subprocess.run([RUST, "stats", "--load", path2], check=True, capture_output=True, text=True).stdout
            self.assertIn(f'"clock": {m.clock}', out2)


def strip(value):
    """Numbers rounded so a last-bit difference between two `pow` implementations does not count as disagreement."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip(v) for v in value]
    if isinstance(value, float):
        return round(value, 9)
    return value


if __name__ == "__main__":
    unittest.main()
