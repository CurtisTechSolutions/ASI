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
sys.path.insert(0, HERE)

from latticefsm import (  # noqa: E402
    COEFFICIENT_LIMIT, DEFAULT_ALPHABET, DEFAULT_STATES, FEATURES, LANGUAGES, WIDTH_MAX, WIDTH_MIN, WIDTH_REST, Edge,
    Machine, Weighting, examples, language, load_machine,
)
from latticefsm import center, center_out, compress, fidelity, load_core, shell, shell_sizes, shells  # noqa: E402
from latticefsm import FOCUS_RANGE, central_vertical, focus_index, focus_node  # noqa: E402
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


def taught(name="even-b", episodes=1500, seed=2, **kw):
    lang = language(name)
    m = Machine(accepting=lang.accepting, seed=seed, **kw)
    experiment.teach_language(m, name, episodes, random.Random(seed))
    return m


class TestGeometry(unittest.TestCase):
    def test_the_cube_has_a_centre_and_seven_shells(self):
        shape = (13, 13, 13)
        self.assertEqual(center(shape), (6, 6, 6))
        self.assertEqual(shells(shape), 7)
        self.assertEqual(shell_sizes(shape), [1, 26, 98, 218, 386, 602, 866])
        self.assertEqual(sum(shell_sizes(shape)), 13 ** 3)
        self.assertEqual(shell(shape, 6, 6, 6), 0)
        self.assertEqual(shell(shape, 0, 6, 6), 6)
        self.assertEqual(shell(shape, 7, 5, 6), 1)

    def test_center_out_visits_every_cell_once_from_the_middle_outward(self):
        shape = (13, 13, 13)
        order = center_out(shape)
        self.assertEqual(sorted(order), list(range(13 ** 3)))
        self.assertEqual(order[0], (6 * 13 + 6) * 13 + 6)
        rings = [shell(shape, c // 169, (c // 13) % 13, c % 13) for c in order]
        self.assertEqual(rings, sorted(rings))

    def test_other_shapes(self):
        self.assertEqual(center((4, 2, 4)), (2, 1, 2))
        self.assertEqual(sum(shell_sizes((4, 2, 4))), 32)
        self.assertEqual(sorted(center_out((3, 5, 3))), list(range(45)))


class TestCompression(unittest.TestCase):
    def test_exact_is_lossless_bit_for_bit(self):
        m = taught()
        core = compress(m)
        self.assertEqual(core.precision, "exact")
        self.assertEqual(core.center, (6, 6, 6))
        r = core.decompress()
        self.assertEqual(r.to_dict()["lattice"], m.to_dict()["lattice"])
        f = fidelity(m, r)
        self.assertTrue(f["lossless"])
        self.assertEqual((f["max_kl"], f["greedy_changed"], f["edges_differing"]), (0.0, 0, 0))
        self.assertEqual((r.clock, r.state, r.runs, r.credits, r.stimulation), (m.clock, m.state, m.runs, m.credits, m.stimulation))
        self.assertEqual([s.to_list() for s in r.lattice.states], [s.to_list() for s in m.lattice.states])

    def test_the_code_is_small(self):
        m = taught()
        core = compress(m)
        s = core.summary()
        touched = sum(e.touched for e in m.lattice)
        self.assertEqual(s["touched"], touched)
        self.assertEqual(core.bytes, (13 ** 3 + 7) // 8 + touched * (6 * core.int_width + 15 * 8))
        self.assertGreater(s["ratio"], 5)
        self.assertEqual(compress(Machine()).bytes, (13 ** 3 + 7) // 8)    # a fresh matrix is its bitmap

    def test_smaller_precisions_lose_a_little_and_keep_the_behaviour(self):
        m = taught()
        sizes = []
        for precision in ("exact", "float32", "float16"):
            core = compress(m, precision)
            f = fidelity(m, core.decompress())
            sizes.append(core.bytes)
            self.assertEqual(f["lossless"], precision == "exact")
            self.assertTrue(f["preserved"], (precision, f))
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreater(sizes[1], sizes[2])
        self.assertLess(fidelity(m, compress(m, "float32").decompress())["max_relative_error"], 1e-6)
        self.assertLess(fidelity(m, compress(m, "float16").decompress())["max_relative_error"], 1e-2)

    def test_a_budget_takes_the_least_lossy_precision_that_fits(self):
        m = taught()
        exact, f32, f16 = (compress(m, p).bytes for p in ("exact", "float32", "float16"))
        self.assertEqual(compress(m, budget=exact).precision, "exact")
        self.assertEqual(compress(m, budget=exact - 1).precision, "float32")
        self.assertEqual(compress(m, budget=f32 - 1).precision, "float16")
        self.assertEqual(compress(m, budget=1).precision, "float16")
        with self.assertRaises(ValueError):
            compress(m, "float8")

    def test_rebuilding_from_the_middle_outward(self):
        m = taught()
        core = compress(m)
        sizes = shell_sizes(core.shape)
        hits = [row["touched"] for row in core.shell_table()]
        self.assertEqual(sum(hits), core.n_touched)
        for k in range(len(sizes) + 1):
            part = core.decompress(shells=k)
            inside = set(center_out(core.shape)[:sum(sizes[:k])])
            for cell, (eo, ep) in enumerate(zip(m.lattice.edges, part.lattice.edges)):
                if cell in inside:
                    self.assertEqual(eo.to_list(), ep.to_list())
                else:
                    self.assertFalse(ep.touched)
            self.assertEqual(sum(e.touched for e in part.lattice), sum(hits[:k]))
        self.assertEqual(core.decompress(shells=len(sizes)).to_dict()["lattice"], m.to_dict()["lattice"])

    def test_walking_straight_from_the_code(self):
        m = taught()
        core = compress(m)
        for text in ("", "a", "abba", "bbab", "aaaa"):
            for middle in (False, True):
                states, accepted = core.run(text, from_middle=middle)
                r = m.run(text, temperature=0.0, quiet=True, from_middle=middle)
                self.assertEqual(states, r.states)
                self.assertEqual(accepted, r.accepted)
        for s in (0, 6, 12):
            self.assertEqual(core.probabilities(s, "a"), m.probabilities(s, "a"))

    def test_the_code_round_trips_through_its_file(self):
        m = taught()
        with tempfile.TemporaryDirectory() as d:
            for precision in ("exact", "float16"):
                core = compress(m, precision)
                path = os.path.join(d, f"core-{precision}.json.gz")
                core.save(path)
                back = load_core(path)
                self.assertEqual((back.ints, back.floats, back.touched), (core.ints, core.floats, core.touched))
                self.assertEqual(back.decompress().to_dict()["lattice"], core.decompress().to_dict()["lattice"])

    def test_measurements_and_compression_are_quiet(self):
        m = taught()
        before = m.to_dict()
        compress(m, "float16").decompress()
        fidelity(m, compress(m).decompress())
        after = m.to_dict()
        before.pop("rng"), after.pop("rng")
        self.assertEqual(before, after)


class TestFromTheMiddle(unittest.TestCase):
    def test_a_run_can_start_from_the_middle_state(self):
        m = Machine(seed=1)
        self.assertEqual(m.center_state, 6)
        r = m.run("ab", from_middle=True)
        self.assertEqual(r.states[0], 6)
        self.assertEqual(m.path[0].source, 6)
        q = m.run("ab", quiet=True, from_middle=True)
        self.assertEqual(q.states[0], 6)
        self.assertEqual(m.run("ab").states[0], 0)


class TestCompressEvery(unittest.TestCase):
    def test_the_matrix_is_compressed_every_n_transitions(self):
        m = Machine(seed=1, compress_every=10)
        m.run("ab" * 12)                       # 24 transitions
        self.assertEqual((m.compressions, m.since_compression, m.last_compressed), (2, 4, 20))
        self.assertIsNotNone(m.core)
        m.teach(0, "a", 1, 1.0)                # a lesson is a transition too
        self.assertEqual(m.since_compression, 5)
        m.run("a" * 5, quiet=True)             # a quiet run is not
        self.assertEqual(m.since_compression, 5)
        self.assertEqual(Machine(seed=1).compressions, 0)

    def test_an_exact_rebuild_changes_nothing_and_a_lossy_one_is_applied(self):
        plain = taught(episodes=800)
        exact = taught(episodes=800, compress_every=50, compress_rebuild=True)
        self.assertEqual(exact.to_dict()["lattice"], plain.to_dict()["lattice"])
        self.assertGreater(exact.compressions, 10)
        lossy = Machine(seed=3, compress_every=1, compress_precision="float16", compress_rebuild=True)
        lossy.teach(0, "a", 1, 0.1234567)
        e = lossy.edge(0, "a", 1)
        self.assertNotEqual(e.rewarded, 0.1234567)
        self.assertAlmostEqual(e.rewarded, 0.1234567, places=3)

    def test_a_rebuild_keeps_the_path_valid(self):
        m = Machine(seed=1, compress_every=2, compress_precision="float16", compress_rebuild=True)
        m.run("abab")
        path = list(m.path)
        self.assertTrue(all(m.lattice.edges[m.lattice.offset(e.source, e.symbol, e.target)] is e for e in path))
        self.assertEqual(m.reward(1.0), 4)
        self.assertGreater(sum(e.rewarded for e in m.lattice), 0)

    def test_the_schedule_is_saved(self):
        m = Machine(compress_every=7, compress_precision="float32", compress_rebuild=True)
        m.run("abababab")
        back = Machine.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual((back.compress_every, back.compress_precision, back.compress_rebuild), (7, "float32", True))
        self.assertEqual((back.compressions, back.since_compression), (m.compressions, m.since_compression))
        with self.assertRaises(ValueError):
            Machine(compress_every=-1)


class TestFocus(unittest.TestCase):
    def test_focus_picks_a_node_of_the_central_vertical_vector(self):
        shape = (13, 13, 13)
        self.assertEqual(central_vertical(shape), [(s, 6, 6) for s in range(13)])
        self.assertEqual(FOCUS_RANGE, (0.0, 1.0))
        cases = {None: 0, 0.0: 0, 0.07: 0, 1 / 13: 1, 0.5: 6, 0.92: 11, 12 / 13: 12, 1.0: 12}
        for f, k in cases.items():
            self.assertEqual(focus_index(13, f), k, f)
            self.assertEqual(focus_node(shape, f), (k, 6, 6))
        for bad in (-0.01, 1.01, float("nan"), float("inf"), "0.5", True):
            with self.assertRaises(ValueError, msg=bad):
                focus_index(13, bad)

    def test_runs_start_from_the_focus_node(self):
        m = Machine(seed=1)
        self.assertEqual(m.run("ab").states[0], 0)              # no focus: the start state, the top node
        m.focus = 1.0
        self.assertEqual((m.origin, m.run("ab").states[0]), (12, 12))
        self.assertEqual(m.run("ab", quiet=True).states[0], 12)
        self.assertEqual(m.run("ab", focus=0.5).states[0], 6)   # for this run only
        self.assertEqual(m.run("ab", from_middle=True).states[0], 6)
        self.assertEqual(m.stats()["focus_node"], [12, 6, 6])
        with self.assertRaises(ValueError):
            m.focus = 2.0
        with self.assertRaises(ValueError):
            Machine(focus=-1)
        back = Machine.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(back.focus, 1.0)

    def test_a_learned_focus_reads_the_input_and_learns_from_credit(self):
        m = Machine(seed=1, learn_focus=True)
        x = m.focus_learner.features(list("abba"))
        self.assertEqual(len(x), 2 + 3 * 13)
        self.assertAlmostEqual(m.focus_learner.predict(list("abba")), 1 / (1 + math.exp(3)))
        self.assertEqual(m.run("abba", quiet=True).states[0], 0)   # untrained: the top node
        before = list(m.focus_learner.weights)
        m.run("abba")
        f = m.last_focus
        self.assertIsNotNone(f)
        m.reward(1.0)
        self.assertNotEqual(m.focus_learner.weights, before)
        learner = m.focus_learner
        lo = taught("ends-ab", episodes=600, seed=4, learn_focus=True).focus_learner
        self.assertTrue(all(-8 <= w <= 8 for w in lo.weights))
        back = Machine.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(back.focus_learner.weights, learner.weights)
        self.assertTrue(back.learn_focus)

    def test_the_learner_moves_toward_a_rewarded_draw_and_away_from_a_punished_one(self):
        from latticefsm.focus import FocusLearner
        for credit, direction in ((1.0, 1), (-1.0, -1)):
            fl = FocusLearner(list("ab"), rate=0.1, explore=0.2)
            x = fl.features(list("ab"))
            mu = fl.mean(x)
            fl.learn(credit, mu + 0.3, mu, x)
            self.assertEqual((fl.mean(x) > mu) - (fl.mean(x) < mu), direction)


class TestSkip(unittest.TestCase):
    def test_a_skip_takes_the_more_efficient_two_edge_path(self):
        m = Machine(3, "ab", seed=1, skip=True, temperature=0.0)
        # the greedy step on a goes to 1, but from 1 nothing on b is good; through 2 there is a sure path to 0
        m.edge(0, "a", 1).weighting.bias = 1.0
        m.edge(0, "a", 2).weighting.bias = 0.9
        for t in range(3):
            m.edge(1, "b", t).weighting.bias = 0.0
        m.edge(2, "b", 0).weighting.bias = 6.0
        r = m.run("ab")
        self.assertEqual(len(r.transitions), 1)
        t = r.transitions[0]
        self.assertEqual((t.source, t.symbol, t.target, t.skipped), (0, "ab", 0, 2))
        self.assertEqual(r.states, [0, 0])
        self.assertEqual([(e.source, e.target) for e in m.path], [(0, 2), (2, 0)])   # both edges traversed
        self.assertEqual(m.lattice.states[2].visits, 0)                             # the node skipped is not visited
        self.assertEqual((m.skips, m.clock), (1, 2))
        plain = m.run("ab", skip=False, quiet=True)          # without skips: two moves, the greedy step first
        self.assertEqual(len(plain.transitions), 2)
        self.assertEqual(plain.transitions[0].target, 1)
        self.assertTrue(all(t.skipped is None for t in plain.transitions))
        self.assertEqual(m.reward(1.0), 2)
        m.skip_margin = 100.0
        self.assertIsNone(m.run("ab", quiet=True).transitions[0].skipped)          # not that much more efficient

    def test_without_skips_nothing_changes(self):
        a = taught(episodes=500)
        b = taught(episodes=500, skip=False)
        self.assertEqual(a.to_dict()["lattice"], b.to_dict()["lattice"])
        self.assertEqual(a.skips, 0)

    def test_skips_happen_in_training_and_are_saved(self):
        m = taught(episodes=600, skip=True)
        self.assertGreater(m.skips, 0)
        back = Machine.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual((back.skip, back.skips), (True, m.skips))
        core = compress(m)
        for text in ("abba", "aab", "bbbb"):
            states, accepted = core.run(text)
            r = m.run(text, quiet=True, temperature=0.0)
            self.assertEqual(states, r.states, text)


class TestRearrange(unittest.TestCase):
    def test_a_swap_is_a_relabelling(self):
        m = taught(episodes=800)
        before = {(s, a, t): m.lattice[s, a, t].log_weight(m.clock, m.life, 1.0)
                  for s in range(13) for a in range(13) for t in range(13)}
        accepting = {sid: m.lattice.states[sid].accepting for sid in range(13)}
        m.swap_states(0, 12)
        m.swap_symbols("a", "g")
        self.assertEqual(m.lattice.state_ids[:1] + m.lattice.state_ids[12:], [12, 0])
        self.assertEqual((m.start, m.alphabet[0], m.alphabet[6]), (12, "g", "a"))
        pos = {sid: p for p, sid in enumerate(m.lattice.state_ids)}
        old_symbols = list("abcdefghijklm")
        for (s, a, t), w in before.items():
            moved = m.lattice[pos[s], m.alphabet.index(old_symbols[a]), pos[t]]
            self.assertEqual(moved.log_weight(m.clock, m.life, 1.0), w)
            self.assertEqual((moved.source, moved.symbol, moved.target),
                             (pos[s], m.alphabet.index(old_symbols[a]), pos[t]))
        for sid, acc in accepting.items():
            self.assertEqual(m.lattice.states[pos[sid]].accepting, acc)
        self.assertEqual(m.swaps, 2)

    def test_rearranging_moves_the_busy_nodes_inward(self):
        m = taught(episodes=800)
        made = m.rearrange(full=True)
        self.assertTrue(made)
        self.assertEqual(m.rearrange(full=True), [])                 # settled
        for load in (m.lattice.state_load(), m.lattice.symbol_load()):
            c = len(load) // 2
            self.assertTrue(all(load[k] <= load[k + 1] for k in range(c)), load)
            self.assertTrue(all(load[k] >= load[k + 1] for k in range(c, len(load) - 1)), load)
        self.assertIn(m.alphabet[6], "ab")                           # the busiest symbol at the centre
        shells_hit = [row["touched"] for row in compress(m).shell_table()]
        self.assertGreater(sum(shells_hit[:5]), 0)                   # information now inside the outer shells
        self.assertEqual(compress(m).decompress().to_dict()["lattice"], m.to_dict()["lattice"])

    def test_a_rearrangement_waits_for_the_end_of_a_run(self):
        m = Machine(seed=1, rearrange_every=3, skip=True)
        for _ in range(30):
            r = m.run("abababab")
            path = list(m.path)
            self.assertTrue(all(m.lattice.edges[m.lattice.offset(e.source, e.symbol, e.target)] is e for e in path))
            m.reward(1.0) if r.accepted else m.punish(1.0)
        self.assertGreater(m.swaps, 0)
        back = Machine.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual((back.rearrange_every, back.swaps, back.lattice.state_ids), (3, m.swaps, m.lattice.state_ids))


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
            core = os.path.join(d, "core.json.gz")
            self.assertIn("bit for bit", self.run_cli("compress", "--load", path, "--core", core))
            self.assertIn("every shell", self.run_cli("expand", "--core", core))
            self.assertIn("3 shells", self.run_cli("expand", "--core", core, "--shells", "3"))
            self.assertIn("read straight from the code", self.run_cli("core-run", "ab", "--core", core, "--from-middle"))
            self.assertIn("float16", self.run_cli("compress", "--load", path, "--precision", "float16"))
            self.assertIn("6 -", self.run_cli("run", "ab", "--load", path, "--from-middle", "--quiet"))
            self.assertIn("state 12", self.run_cli("focus", "--load", path, "--level", "1", "--save", path))
            self.assertIn("12 -", self.run_cli("run", "ab", "--load", path, "--quiet"))
            self.assertIn("learned from the input", self.run_cli("focus", "--load", path, "--learn", "on", "--text", "ab"))
            self.assertIn("skips on", self.run_cli("skip", "on", "--load", path))
            self.assertIn("swaps toward the centre", self.run_cli("rearrange", "--load", path, "--full"))
            self.assertIn("swapped states 0 and 12", self.run_cli("rearrange", "--load", path, "--swap", "states", "0", "12"))


class TestNetwork(unittest.TestCase):
    """Machines connected into networks: the architecture, the wiring, the credit, the file, English in and out."""

    def test_the_architecture_string(self):
        from latticefsm.network import Layer, Network, Parallel, Sequential, parse_architecture
        net = Network("13>(13|13f)>4", alphabet="ab")
        blocks = net.root.blocks
        self.assertEqual([type(b) for b in blocks], [Layer, Parallel, Layer])
        self.assertEqual([l.machine.lattice.shape for l in net.layers()],
                         [(13, 2, 13), (13, 13, 13), (13, 13, 13), (4, 26, 4)])
        self.assertEqual([l.machine.seed for l in net.layers()], [1, 2, 3, 4])
        self.assertEqual(blocks[1].branches[1].emit, "final")
        self.assertIsInstance(parse_architecture("(3>3|3)"), Parallel)
        self.assertIsInstance(parse_architecture("3>3"), Sequential)
        for bad in ("", "13>", "(13|13", "13x", "(13|)", "0", "63"):
            with self.assertRaises(ValueError, msg=bad):
                Network(bad, alphabet="ab")
        self.assertIn("4 layers", net.summary())

    def test_one_layer_is_the_machine_it_holds(self):
        from latticefsm.network import Network, teach_network
        lang = language("mod3-a")
        m = Machine(3, "ab", accepting=lang.accepting, seed=4)
        experiment.teach_language(m, "mod3-a", 600, random.Random(4))
        net = Network("3", alphabet="ab", seed=4).compile(accepting=lang.accepting)
        teach_network(net, "mod3-a", 600, random.Random(4))
        self.assertEqual(net.layers()[0].machine.to_dict()["lattice"], m.to_dict()["lattice"])

    def test_the_wiring(self):
        from latticefsm.network import Network
        net = Network("(3|3)", alphabet="ab", merge="interleave")
        out, _ = net.forward("ab", quiet=True, temperature=0.0)
        a = [s for s, _ in zip("abc", range(3))]
        self.assertEqual(len(out), 4)
        self.assertTrue(all(o in a for o in out[0::2]) and all(o in "def" for o in out[1::2]), out)
        net = Network("(3|3)", alphabet="ab", merge="concat")
        out, _ = net.forward("ab", quiet=True, temperature=0.0)
        self.assertTrue(all(o in "abc" for o in out[:2]) and all(o in "def" for o in out[2:]), out)
        self.assertEqual(net.output_alphabet, list("abcdef"))
        net = Network("3>2f", alphabet="ab")
        self.assertEqual(len(net.forward("abab", quiet=True, temperature=0.0)[0]), 1)

    def test_votes(self):
        from latticefsm.network import Network
        for vote, want in (("majority", True), ("any", True), ("all", False)):
            net = Network("(2|2|2)", alphabet="a", vote=vote)
            for i, layer in enumerate(net.layers()):
                for st in layer.machine.lattice.states:
                    st.accepting = i < 2
            self.assertEqual(net.forward("", quiet=True)[1], want, vote)

    def test_shared_and_own_credit(self):
        from latticefsm.network import Network
        # own: the branch that accepted wrongly is punished, the one that rejected rightly rewarded
        net = Network("(2|2)", alphabet="a").compile(accepting=[0], credit="own")
        net.layers()[1].machine.lattice.states[0].accepting = False
        net.layers()[1].machine.lattice.states[1].accepting = False
        net.teach("a", False)
        a, b = net.layers()
        self.assertEqual((sum(e.rewarded for e in a.machine.lattice), sum(e.punished for e in a.machine.lattice)),
                         (0.0, 1.0) if a.last_accepted else (1.0, 0.0))
        self.assertEqual(sum(e.rewarded for e in b.machine.lattice), 1.0)
        # shared, one after another: the last block in full, the one before it layer_discount of that
        net = Network("2>2", alphabet="a").compile(accepting=[0, 1], layer_discount=0.5)
        net.teach("a", True)
        hidden, out = net.layers()
        self.assertEqual(sum(e.rewarded for e in out.machine.lattice), 1.0)
        self.assertEqual(sum(e.rewarded for e in hidden.machine.lattice), 0.5)

    def test_greedy_passes_are_deterministic_and_draw_nothing(self):
        from latticefsm.network import Network
        net = Network("13>13", alphabet="ab")
        before = [l.machine.rng.getstate() for l in net.layers()]
        first = [net.forward(s, quiet=True, temperature=0.0)[0] for s in ("ab", "bba", "")]
        again = [net.forward(s, quiet=True, temperature=0.0)[0] for s in ("ab", "bba", "")]
        self.assertEqual(first, again)
        self.assertEqual(before, [l.machine.rng.getstate() for l in net.layers()])
        self.assertEqual(first[0], ["a", "a"])  # a fresh row ties, and the tie goes to state 0

    def test_the_file_round_trips(self):
        from latticefsm.network import Network, load_network, teach_network
        net = Network("(3|3)>3", alphabet="ab", seed=3).compile(accepting=[0], credit="own", layer_discount=0.6)
        teach_network(net, "even-b", 300, random.Random(3))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "n.json.gz")
            net.save(path)
            back = load_network(path)
        self.assertEqual(back.to_dict(), net.to_dict())
        strings = ["", "a", "ab", "bab", "abba", "bbbb"]
        self.assertEqual(back.predict(strings), net.predict(strings))

    def test_english_in_and_out(self):
        from latticefsm.network import Network, PhoneticDecoder, PhoneticEncoder, load_network
        from latticefsm.phonetic import PLURALS, inventory, split
        self.assertEqual(len(inventory()), 43)
        net = Network("enc>dec")
        self.assertIsInstance(net.encoder, PhoneticEncoder)
        self.assertIsInstance(net.decoder, PhoneticDecoder)
        self.assertEqual(net.encoder.encode("cats"), ["K", "AE", "T", "S", "#", "#"])
        self.assertEqual(net.decoder.machine.lattice.shape, (43, 43, 43))
        self.assertEqual(net.translate("cat"), "")  # untaught, every row ties and the tie is the boundary
        train, test = split(PLURALS)
        history = net.fit_pairs(train, epochs=10, validation_data=test)
        self.assertGreater(history["exact"][-1], 0.6)
        self.assertEqual(net.translate("cat"), "cats")
        t = net.trace("dog")
        self.assertEqual((t["layers"][0]["english"], t["output"]), ("dog", "dogs"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            net.save(path)
            back = load_network(path)
        self.assertEqual(back.evaluate_pairs(train), net.evaluate_pairs(train))
        self.assertEqual(back.translate("hat"), net.translate("hat"))
        for bad in ("13>enc", "(enc|13)", "enc"):
            with self.assertRaises(ValueError, msg=bad):
                Network(bad)
        with self.assertRaises(ValueError):
            Network("enc>13").translate("cat")
        with self.assertRaises(ValueError):
            PhoneticDecoder(skip=True)

    def test_the_command_line(self):
        def cli(*args):
            out = subprocess.run([sys.executable, "-m", "latticefsm", "network", *args], cwd=ROOT,
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            return out.stdout
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "n.json")
            self.assertIn("2 layers", cli("new", "--arch", "3>3", "--alphabet", "ab", "--save", path))
            self.assertIn("epoch 1", cli("fit", "--load", path, "--language", "even-b", "--episodes", "200", "--save", path))
            self.assertIn("accuracy even-b on every string up to 6 symbols (127)", cli("evaluate", "--load", path))
            self.assertIn("lattice_2 read", cli("predict", "ab", "--load", path))
            t = os.path.join(d, "t.json")
            self.assertIn("held out", cli("fit", "--arch", "enc>dec", "--task", "plural", "--epochs", "6", "--save", t))
            self.assertIn("cat -> cats", cli("translate", "cat", "--load", t))
            self.assertIn("exact plural", cli("evaluate", "--load", t, "--task", "plural"))


class TestMedia(unittest.TestCase):
    """Text, audio, images and video as symbols, and networks taught from them."""

    def test_every_image_format_reads_the_same_picture(self):
        from latticefsm.media import cells, read_image
        with tempfile.TemporaryDirectory() as d:
            f = media_files(d)
            rgb = read_image(f["rgb.png"])
            self.assertEqual(rgb[:2], (13, 11))
            for other in ("rgb.bmp", "rgb.ppm", "rgba.png"):
                self.assertEqual(read_image(f[other]), rgb, other)
            w, h, px = read_image(f["gray.png"])
            self.assertEqual(px[:3], [0, 23, 46])
            self.assertEqual(read_image(f["pal.png"])[2][:4], [0, 76, 149, 255])
            self.assertEqual(len(cells(rgb, 4)), 16)
            with self.assertRaises(ValueError):
                read_image(b"GIF89a")

    def test_cells_orders_and_differences(self):
        from latticefsm.media import cells, diff_symbols, image_symbols, order, text_symbols
        self.assertEqual(order(3, "center-out")[0], 4)
        self.assertEqual(sorted(order(4, "center-out")), list(range(16)))
        self.assertEqual(order(4, "center-out")[:4], [5, 6, 9, 10])
        img = (4, 4, [0] * 8 + [255] * 8)
        self.assertEqual(cells(img, 2), [0, 0, 255, 255])
        self.assertEqual(image_symbols([0, 63, 64, 255], 4), ["L0", "L0", "L1", "L3"])
        self.assertEqual(diff_symbols([100] * 4, [107, 108, 52, 151]), ["D0", "D+1", "D-2", "D+2"])
        self.assertEqual(diff_symbols([100] * 4, [93, 92, 148, 147]), ["D0", "D-1", "D+2", "D+1"])
        self.assertEqual(text_symbols("Hi, Bob!\t2"), list("hi,_bob!_2"))

    def test_a_video_is_keyframes_differences_and_its_sound(self):
        from latticefsm.media import audio_units, video_symbols
        with tempfile.TemporaryDirectory() as d:
            f = media_files(d)
            units = audio_units(f["wav"])
            self.assertTrue(units and all(u.startswith("q") for u in units))
            v = video_symbols(f["clip"])
            self.assertEqual(v.count("|"), 8)
            self.assertEqual(v[1:65], [s for s in v[1:65] if s.startswith("L")])
            self.assertTrue(any(s.startswith("D+") for s in v) and any(s.startswith("D-") for s in v))
            self.assertEqual([s for s in v if s.startswith("q")], units)
            quiet = video_symbols(f["clip"], audio=False)
            self.assertFalse(any(s.startswith("q") for s in quiet))

    def test_networks_read_every_kind_and_keep_them_in_the_file(self):
        from latticefsm.network import ENCODERS, Network, load_network
        with tempfile.TemporaryDirectory() as d:
            f = media_files(d)
            inputs = {"text": "the cat sat", "audio": f["wav"], "image": f["rgb.png"], "diff": f["clip"],
                      "video": f["clip"], "enc": "the cat sat"}
            for kind, x in inputs.items():
                net = Network(f"{kind}>3").compile(accepting=[0])
                self.assertIsInstance(net.encoder, ENCODERS["phonetic" if kind == "enc" else kind])
                net.fit([x, x], [True, False], epochs=2)
                path = os.path.join(d, f"{kind}.json")
                net.save(path)
                back = load_network(path)
                self.assertEqual(back.to_dict(), net.to_dict(), kind)
                self.assertEqual(back.predict([x]), net.predict([x]), kind)
                self.assertEqual(back.encoder._symbols_of(x) if kind != "enc" else back._symbols(x),
                                 net.encoder._symbols_of(x) if kind != "enc" else net._symbols(x))
            img = Network("image>3")
            img.encoder.encode(f["rgb.png"])
            self.assertEqual(len(img.encoder._cached), 1)

    def test_the_command_line(self):
        def cli(*args):
            out = subprocess.run([sys.executable, "-m", "latticefsm", "network", *args], cwd=ROOT,
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            return out.stdout
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("labels.tsv", cli("make-data", "--kind", "image", "--out", d, "--count", "10"))
            out = cli("fit", "--arch", "image>3", "--data", os.path.join(d, "labels.tsv"), "--epochs", "2",
                      "--save", os.path.join(d, "n.json"))
            self.assertIn("8 examples taught x 2 epochs, 2 held out", out)
            self.assertIn("accuracy", cli("evaluate", "--load", os.path.join(d, "n.json"), "--data",
                                          os.path.join(d, "labels.tsv")))
            self.assertIn("64 symbols", cli("encode", os.path.join(d, "img000.ppm"), "--arch", "image>3"))


class TestNGram(unittest.TestCase):
    """Predicting the next thing: tokens, Kneser-Ney, the adaptive edges, the file."""

    def model(self, order=3, unit="word"):
        from latticefsm.ngram import NGram
        m = NGram(order, unit)
        m.train_lines(["the cat sat on the mat", "the cat ate the fish", "a dog sat on the log",
                       "the dog ate the bone ."])
        return m

    def test_tokens(self):
        from latticefsm.ngram import detokenize, tokenize
        self.assertEqual(tokenize("Don't stop -- it's 3.5, OK?"), ["don't", "stop", "-", "-", "it's", "3", ".", "5",
                                                                  ",", "ok", "?"])
        self.assertEqual(tokenize("Hi you", "char"), list("hi_you"))
        self.assertEqual(detokenize(["hello", ",", "world", "!"]), "hello, world!")
        self.assertEqual(detokenize(list("hi_you"), "char"), "hi you")
        self.assertEqual(tokenize("cats", "sound"), ["K", "AE", "T", "S"])
        self.assertEqual(detokenize(["K", "AE", "T", "S"], "sound"), "cats")

    def test_every_distribution_sums_to_one_and_is_kneser_ney(self):
        m = self.model()
        for text in ("", "the", "the cat", "sat on", "never seen words"):
            ctx = m.context(m.tokens(text))
            d = m.distribution(ctx)
            self.assertAlmostEqual(sum(d), 1.0, places=12)
            self.assertEqual(d[0], 0.0)
            self.assertEqual(d, [m.prob(ctx, w) for w in range(len(d))])
        self.assertEqual(m.predict("the cat", 1)[0][0], "sat")
        self.assertEqual(m.predict("sat on the", 2)[0][0], "mat")
        # an unknown context falls to the continuation counts: what follows the most different contexts
        self.assertIn(m.predict(["zzz", "zzz"], 1)[0][0], ("the", "</s>"))
        self.assertGreater(dict(m.predict(["zzz", "zzz"], 20))["the"], dict(m.predict(["zzz", "zzz"], 20))["cat"])

    def test_more_order_more_context(self):
        from latticefsm.ngram import corpus_lines, split_lines
        lines = corpus_lines(["data/corpus/sentences.txt"])[0][1]
        train, test = split_lines(lines, 5)
        ppl = {}
        for n in (1, 2, 3):
            m = self.model(n)
            m.train_lines(train)
            ppl[n] = m.evaluate(test)["perplexity"]
        self.assertLess(ppl[2], ppl[1])
        self.assertLess(ppl[3], ppl[1])

    def test_credit_widens_and_narrows_what_is_predicted(self):
        m = self.model()
        ctx = m.context(m.tokens("the cat"))
        before = m.prob(ctx, m.index["ate"])
        m.credit_text("the cat ate", 3)
        self.assertGreater(m.prob(ctx, m.index["ate"]), before)
        self.assertAlmostEqual(sum(m.distribution(ctx)), 1.0, places=12)
        m.credit_text("the cat ate", -10)
        self.assertLess(m.prob(ctx, m.index["ate"]), before)
        m.stimulation = 0.0
        self.assertEqual(m.prob(ctx, m.index["ate"]), before)

    def test_generation_and_the_file(self):
        from latticefsm.ngram import load_ngram
        m = self.model()
        self.assertEqual(m.complete("the dog", 10), "ate the fish")  # a tie goes to the token seen first
        m.credit_text("a dog sat", 1)
        with tempfile.TemporaryDirectory() as d:
            for name in ("m.json", "m.json.gz"):
                path = os.path.join(d, name)
                m.save(path)
                back = load_ngram(path)
                self.assertEqual(back.to_dict(), m.to_dict())
                self.assertEqual(back.predict("the", 5), m.predict("the", 5))

    def test_a_corpus_directory(self):
        from latticefsm.ngram import NGram, corpus_lines
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.txt"), "w") as f:
                f.write("one two three\n\nfour five\n")
            with open(os.path.join(d, "b.jsonl"), "w") as f:
                f.write(json.dumps({"text": "six seven"}) + "\n" + json.dumps("eight") + "\n")
            with open(os.path.join(d, "README.md"), "w") as f:
                f.write("not this\n")
            got = corpus_lines([d])
            self.assertEqual([lines for _, lines in got], [["one two three", "four five"], ["six seven", "eight"]])
            m = NGram(2)
            n, held = m.train_paths([d], held_out=2)
            self.assertEqual(held, [["four", "five"], ["eight"]])

    def test_predicting_every_token_while_learning_it(self):
        from latticefsm.ngram import NGram, corpus_lines
        lines = corpus_lines(["data/corpus/prose.txt"])[0][1]
        for unit, order in (("word", 3), ("word", 1), ("char", 5)):
            plain = NGram(order, unit)
            plain.train_lines(lines, predict=False)
            learner = NGram(order, unit)
            n = learner.train_lines(lines, predict=True)
            # predicting first changes nothing that is learned
            self.assertEqual(learner.to_dict(), plain.to_dict(), unit)
            online = learner.reset_online()
            self.assertEqual(online["predicted"], n)
            self.assertTrue(0 < online["accuracy"] < 1)
            self.assertAlmostEqual(online["bits"], math.log2(online["perplexity"]), places=9)
            self.assertEqual(learner.reset_online()["predicted"], 0)
            # the first guess, found without the whole distribution on the tables kept up to date, is its argmax
            # (ties to the lower id)
            for line in lines[:40]:
                toks = learner.tokens(line)
                for i in range(len(toks)):
                    ctx = learner.context(toks[:i])
                    d = learner.distribution(ctx)
                    self.assertEqual(learner.top(ctx), max(range(1, len(d)), key=lambda w: (d[w], -w)))

    def test_crediting_its_own_guesses(self):
        from latticefsm.ngram import NGram, corpus_lines
        lines = corpus_lines(["data/corpus/prose.txt"])[0][1]
        m = NGram(3)
        m.train_lines(lines, predict=True, credit=0.5)
        self.assertGreater(m.stats()["credited_edges"], 0)
        for line in lines[:20]:
            toks = m.tokens(line)
            for i in range(len(toks)):
                ctx = m.context(toks[:i])
                d = m.distribution(ctx)
                self.assertEqual(m.top(ctx), max(range(1, len(d)), key=lambda w: (d[w], -w)))


class TestChain(unittest.TestCase):
    """One input to a node, the output node predicts the next, and the walk back through the input nodes stops as
    soon as it has gathered enough confidence - fewer nodes the more confident they are."""

    def corpus(self, unit="char"):
        from latticefsm.ngram import corpus_sequences, split_lines
        seqs = [q for _, qs in corpus_sequences(["data/corpus/prose.txt"], unit) for q in qs]
        return split_lines(seqs, 10)

    def test_abc_fills_nodes_one_to_three_and_node_four_predicts(self):
        from latticefsm.chain import Chain
        m = Chain(4, "char", confidence=1.0)
        m.train_lines(["abcd", "abcd", "abce", "abcd"])
        e = m.explain("abc")
        self.assertEqual(e["inputs"], ["a", "b", "c"])
        self.assertEqual([st["node"] for st in e["walk"]], [4, 3, 2, 1])
        self.assertEqual([st["input"] for st in e["walk"]], [None, "c", "b", "a"])
        self.assertEqual(e["next"][0][0], "d")
        self.assertEqual(Chain(4).explain("abc")["walk"], [])  # nothing learned, nothing to walk

    def test_distributions_and_the_first_guess(self):
        from latticefsm.chain import Chain
        train, test = self.corpus()
        m = Chain(6)
        m.train_lines(train)
        for tau in (0.0, 0.5, 0.9, 1.0):
            for seq in test[:6]:
                ids = m.history_ids(seq)
                for i in range(1, len(ids) + 1):
                    path = m.walk(ids[:i], tau)
                    d = m._dist(path)
                    self.assertAlmostEqual(sum(d), 1.0, places=12)
                    self.assertEqual(d, [m._p(path, w) for w in range(len(d))])
                    self.assertEqual(m._guess(path), max(range(1, len(d)), key=lambda w: (d[w], -w)))
        ev0, ev1 = m.evaluate(test, 0.0), m.evaluate(test, 1.0)
        self.assertEqual(ev0["depth"], 0.0)  # no confidence needed: the output node alone
        self.assertEqual(ev0["depths"][0], ev0["tokens"])
        self.assertGreater(ev1["depth"], 3)  # never confident enough: as deep as it can
        self.assertGreater(ev1["accuracy"], ev0["accuracy"])

    def test_the_more_confident_the_shorter_the_walk(self):
        from latticefsm.chain import Chain
        train, test = self.corpus()
        m = Chain(8)
        m.train_lines(train)
        sweep = m.sweep(test, [0.3, 0.5, 0.7, 0.9, 1.0])
        depths = [r["depth"] for r in sweep]
        self.assertEqual(depths, sorted(depths))  # a lower threshold, a shorter walk
        # more training makes the nodes more confident: the same held-out text walks shorter
        before = m.evaluate(test)["depth"]
        m.train_lines(train)
        self.assertLess(m.evaluate(test)["depth"], before)
        # a reward shortens the walks of what it rewards, a punishment lengthens them
        text = "the brain is a directed cyclic graph"
        ids = m.history_ids(m.tokens(text)) + [1]
        walked = lambda: sum(len(m.walk(ids[:i])) for i in range(1, len(ids)))
        start = walked()
        self.assertGreater(m.credit_text(text, 2.0), 0)
        rewarded = walked()
        self.assertLess(rewarded, start)
        m.credit_text(text, -6.0)
        self.assertGreater(walked(), rewarded)

    def test_predicting_while_learning_changes_nothing_learned(self):
        from latticefsm.chain import Chain
        train, _ = self.corpus("word")
        plain, learner = Chain(4, "word"), Chain(4, "word")
        plain.train_lines(train, predict=False)
        n = learner.train_lines(train, predict=True)
        self.assertEqual(plain.to_dict(), learner.to_dict())
        online = learner.reset_online()
        self.assertEqual(online["predicted"], n)
        self.assertTrue(0 < online["accuracy"] < 1 and 0 < online["depth"] <= 3)
        self.assertEqual(plain.reset_online()["predicted"], 0)
        credited = Chain(4, "word")
        credited.train_lines(train, credit=0.5)
        self.assertGreater(credited.stats()["credited"], 0)

    def test_the_file(self):
        from latticefsm.chain import load_chain, Chain
        train, _ = self.corpus()
        m = Chain(5, confidence=0.8)
        m.train_lines(train[:40])
        m.credit_text("the brain", 1.5)
        with tempfile.TemporaryDirectory() as d:
            for name in ("c.json", "c.json.gz"):
                path = os.path.join(d, name)
                m.save(path)
                back = load_chain(path)
                self.assertEqual(back.to_dict(), m.to_dict())
                self.assertEqual(back.predict("the bra", 5), m.predict("the bra", 5))
                self.assertEqual(back.complete("the", 30), m.complete("the", 30))
        with self.assertRaises(ValueError):
            Chain(1)
        with self.assertRaises(ValueError):
            Chain(4, confidence=1.5)


class TestTeaching(unittest.TestCase):
    """Teaching until N of N: a reply taught until the model's first guess is right on every token of it, and a corpus
    drilled until every token is right that can be - a context continued more than one way only ever one way."""

    def test_a_reply_is_taught_until_every_token_is_right(self):
        from latticefsm.chain import Chain
        from latticefsm.ngram import NGram
        for m in (Chain(), NGram(3)):
            m.train_paths(["data/corpus"], held_out=10)
            r = m.teach("what is the brain", "a lattice of adaptive edges that predicts.")
            self.assertTrue(r["reached"], type(m).__name__)
            self.assertEqual(r["right"], r["of"])
            self.assertEqual(r["history"][-1], r["of"])
            self.assertLess(r["history"][0], r["of"])
            self.assertEqual(m.complete("what is the brain", 80).strip(" ."),
                             "a lattice of adaptive edges that predicts")

    def test_drilling_reaches_the_most_possible(self):
        from latticefsm.chain import Chain
        from latticefsm.ngram import NGram, train_corpus
        for m in (Chain(), NGram(3)):
            seqs = []
            m.train_paths(["data/corpus/sentences.txt"], held_out=10)
            train_corpus(["data/corpus/sentences.txt"], m.unit, m.lower, 10, lambda t: seqs.append(t) or 0, [])
            best, n = m.ceiling(seqs)
            self.assertLess(best, n)  # the corpus continues some contexts more than one way
            d = m.drill(seqs, 30, 30)
            self.assertTrue(d["reached"], type(m).__name__)
            self.assertEqual(d["right"], best)
            self.assertEqual(d["rounds"][-1], best)
        tiny = Chain(4)
        tiny.train_lines(["abcd", "abce"])
        self.assertEqual(tiny.ceiling([list("abcd"), list("abce")]), (9, 10))  # after abc: d or e, one of them

    def test_scores_are_the_probabilities(self):
        from latticefsm.chain import Chain
        from latticefsm.ngram import NGram
        n = NGram(3)
        n.train_lines(["the cat sat on the mat", "the cat ate the fish"])
        toks = n.tokens("the cat sat")
        scored = n.score_tokens(toks)
        self.assertEqual([t for t, _ in scored], toks + ["</s>"])
        self.assertEqual(scored[1][1], n.prob(n.context(toks[:1]), n.index["cat"]))
        c = Chain(4)
        c.train_lines(["abcd", "abcd"])
        scored = c.score_tokens(list("abc"))
        self.assertEqual([t for t, _, _ in scored], ["a", "b", "c", "</s>"])
        self.assertTrue(all(0 < p <= 1 for _, p, _ in scored))


class TestCorpus(unittest.TestCase):
    """Every kind of file the corpus reader knows, archives of any size opened member by member, and the same lines
    and bytes in both ports."""

    def test_every_kind(self):
        from corpus_fixture import make
        from latticefsm.corpus import strip_markup
        from latticefsm.ngram import corpus_lines
        with tempfile.TemporaryDirectory() as d:
            make(d)
            got = {os.path.relpath(label, d): lines for label, lines in corpus_lines([d])}
            self.assertEqual(got["a.txt"], ["plain line one", "line two", "indented three"])
            self.assertEqual(got["b.html"][:4], ["A page", "Cats & dogs", "The cat sat on the mat.",
                                                 "Café \u2014 done\u2026"])
            self.assertEqual(got["c.docx"], ["The first paragraph", "and the second."])
            self.assertEqual(got["d.epub"], ["Chapter one.", "Chapter two."])
            self.assertEqual(got["e.jsonl.gz"], ["json one", "json two", "not json"])
            self.assertEqual(got["i.zip/nested.tar.gz/nested.txt"], ["nested in a tar.gz in a zip"])
            self.assertEqual(got["i.zip/inner.zip/deeper.txt"], ["zip in a zip"])
            self.assertEqual(len(got["j.zip/big.txt"]), 2000)  # a ZIP64 member
            self.assertEqual(got["m.odt"], ["Open document.", "A heading"])
            self.assertTrue(any(k.startswith("f.tar/deep/") for k in got))  # a long name, from a pax header
            for hidden in ("i.zip/.hidden.txt", "i.zip/__MACOSX/x.txt", "f.tar/.hid.txt", "README.md", "k.unknown",
                           "i.zip/img.png"):
                self.assertNotIn(hidden, got)
        self.assertEqual(strip_markup("<p>a<br/>b</p><script>x<y</script>&lt;c&gt; &#65;&#x42; &bogus;"),
                         "\na\nb\n<c> AB &bogus;")

    def test_both_ports_read_the_same(self):
        from corpus_fixture import make
        with tempfile.TemporaryDirectory() as d:
            make(d)
            for unit in ("text", "base64"):
                outs = [subprocess.run([*cmd, "corpus", d, "--unit", unit, "--show", "3"], cwd=ROOT, check=True,
                                       capture_output=True, text=True).stdout
                        for cmd in ([sys.executable, "-m", "latticefsm"], [RUST])]
                self.assertEqual(outs[0], outs[1], unit)
                self.assertIn("documents", outs[0])

    def test_training_streams_a_zip(self):
        import zipfile
        from latticefsm.chain import Chain
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "big.zip")
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                with zf.open("a.txt", "w", force_zip64=True) as f:
                    for _ in range(300):
                        f.write(b"the cat sat on the mat\nthe dog ate the bone\n")
            m = Chain(4, "word")
            n, held = m.train_paths([path], held_out=10)
            lines = ["the cat sat on the mat", "the dog ate the bone"] * 300
            trained = [ln for i, ln in enumerate(lines) if i % 10 != 9]
            self.assertEqual(len(held), 60)
            self.assertEqual(n, sum(len(ln.split()) + 1 for ln in trained))
            self.assertEqual(m.sources, [[path, 540, n]])


class TestAutotrain(unittest.TestCase):
    """The corpus, then Ollama: sentences, continuations and a judge's credit, measured every round."""

    def test_with_a_stand_in_ollama(self):
        from fake_ollama import FakeOllama
        from latticefsm.autotrain import Ollama, autotrain, parse_judgements, parse_lines
        from latticefsm.ngram import NGram, load_ngram
        self.assertEqual(parse_lines("Here they are:\n1. One.\n- Two\n\n3) \"Three\""), ["One.", "Two", "Three"])
        self.assertEqual(parse_judgements('ok: [{"id": 2, "ok": false}, {"id": 1, "ok": true}]', 3), [True, False, None])
        self.assertEqual(parse_judgements("1: yes\n2. no", 2), [True, False])
        fake = FakeOllama()
        try:
            client = Ollama(fake.url, "fake-model")
            self.assertEqual(client.models(), ["fake-model:latest"])
            with tempfile.TemporaryDirectory() as d:
                keep, save = os.path.join(d, "ollama.txt"), os.path.join(d, "m.json")
                m = NGram(3)
                logged = []
                rec = autotrain(m, ["data/corpus/sentences.txt"], held_out=5, ollama=client, rounds=2, lines=4,
                                continuations=3, judge=4, topics=["gardens"], patience=5, keep=keep, save=save,
                                log=logged.append)
                self.assertEqual([r["round"] for r in rec], [0, 1, 2])
                self.assertEqual(rec[1]["ollama_lines"], 8)
                self.assertEqual(rec[1]["continuations"], 3)
                self.assertTrue(0 < rec[1]["judged"] <= 4)
                self.assertEqual(rec[1]["approved"], (rec[1]["judged"] + 1) // 2)
                self.assertGreater(rec[2]["tokens"], rec[0]["tokens"])
                self.assertGreater(rec[1]["credited_edges"], 0)
                self.assertIn("gardens", m.vocab)
                self.assertEqual(len(fake.prompts), 2 * (2 + 3 + 1))
                self.assertIn("English sentences about gardens", fake.prompts[0])
                with open(keep) as f:
                    self.assertEqual(len(f.read().splitlines()), 2 * (8 + 3))
                self.assertEqual(load_ngram(save).history, m.history)
                self.assertEqual(len(logged), 3)
        finally:
            fake.close()
        with self.assertRaises(Exception):
            Ollama("http://127.0.0.1:9", timeout=2).models()


class TestBase64(unittest.TestCase):
    """Anything as base64 letters, and an N-gram model that learns them."""

    def test_the_coding(self):
        import base64 as std
        from latticefsm import b64
        rng = random.Random(5)
        for n in range(0, 50):
            data = bytes(rng.randrange(256) for _ in range(n))
            self.assertEqual(b64.encode(data), std.b64encode(data).decode())
            self.assertEqual(b64.decode(b64.encode(data)), data)
            for k in range(n + 1):
                self.assertTrue(b64.encode(data).startswith(b64.encode_prefix(data[:k])))
        self.assertEqual(b64.encode_prefix(b"Hell"), "SGVsb")
        self.assertEqual(b64.decode("SGV sbG8 !\n"), b"Hello")
        self.assertEqual(b64.decode("SGVsbA"), b"Hell")

    def test_files_are_cut_at_whole_groups(self):
        from latticefsm import b64
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "x.bin"), "wb") as f:
                f.write(bytes(range(256)) * 2)
            with open(os.path.join(d, ".hidden"), "w") as f:
                f.write("no")
            (path, seqs), = b64.file_sequences([d])
            self.assertEqual([len(q) for q in seqs[:-1]], [76] * (len(seqs) - 1))
            self.assertEqual(b64.decode("".join(seqs)), bytes(range(256)) * 2)

    def test_a_model_of_any_file(self):
        from latticefsm.ngram import NGram, load_ngram
        with tempfile.TemporaryDirectory() as d:
            f = media_files(d)
            m = NGram(6, "base64")
            n, held = m.train_paths([f["clip"], f["rgb.png"], "data/corpus/sentences.txt"], held_out=10)
            self.assertLessEqual(len(m.vocab) - 3, 65)
            ev = m.evaluate(held)
            self.assertLess(ev["perplexity"], 30)
            out = m.generate_bytes(b"P5\n32 32\n255\n", max_tokens=200)
            self.assertTrue(out.startswith(b"P5\n32 32\n25"))
            self.assertGreater(len(out), 100)
            self.assertTrue(m.complete("the cat", 40).startswith("the ca"))
            path = os.path.join(d, "b.json.gz")
            m.save(path)
            self.assertEqual(load_ngram(path).predict("the", 3), m.predict("the", 3))

    def test_the_command_line(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.bin")
            with open(src, "wb") as f:
                f.write(bytes(range(200)))
            run = lambda *a: subprocess.run([sys.executable, "-m", "latticefsm", *a], cwd=ROOT, capture_output=True,
                                            text=True, check=True)
            run("base64", "encode", src, "--out", os.path.join(d, "a.b64"))
            run("base64", "decode", os.path.join(d, "a.b64"), "--out", os.path.join(d, "b.bin"))
            with open(os.path.join(d, "b.bin"), "rb") as f:
                self.assertEqual(f.read(), bytes(range(200)))
            self.assertEqual(run("base64", "encode", "Hi").stdout, "SGk=\n")
            self.assertEqual(run("base64", "decode", "SGk=").stdout, "Hi\n")


@unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
class TestRustParity(unittest.TestCase):
    """The two ports agree on everything that does not draw: the stimulation and adaptation records, number for
    number, and a machine file written by one reads in the other to the same table."""

    def test_the_deterministic_experiments_agree(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([RUST, "experiment", "--which", "stimulation,adaptation,ngram", "--out", d], check=True,
                           capture_output=True, cwd=ROOT)
            for name, rec in (("stimulation", experiment.stimulation()), ("adaptation", experiment.adaptation()),
                              ("ngram", experiment.ngram(data=(os.path.join(ROOT, "data", "corpus"),)))):
                with open(os.path.join(d, f"{name}_results.json")) as f:
                    rust = json.load(f)
                if name == "ngram":
                    rust, rec = rust["rows"], rec["rows"]
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


@unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
class TestRustNetworkParity(unittest.TestCase):
    """A network file written by either port reads in the other, which then classifies, traces and translates every
    string the same way: greedy passes draw nothing, so they do not depend on the two ports' different generators."""

    def rust(self, *args):
        return subprocess.run([RUST, "network", *args], check=True, capture_output=True, text=True).stdout

    def py(self, *args):
        return subprocess.run([sys.executable, "-m", "latticefsm", "network", *args], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout

    def test_a_language_network_crosses_both_ways(self):
        with tempfile.TemporaryDirectory() as d:
            for writer, reader in ((self.rust, self.py), (self.py, self.rust)):
                path = os.path.join(d, f"n{writer.__name__}.json")
                writer("fit", "--arch", "(3|3)>3", "--alphabet", "ab", "--language", "ends-ab", "--episodes", "600",
                       "--credit", "own", "--save", path)
                for cmd in (("evaluate", "--language", "ends-ab"), ("predict", "", "ab", "abab", "bbaab"),
                            ("summary",)):
                    self.assertEqual(writer(*cmd, "--load", path), reader(*cmd, "--load", path), cmd)

    def test_a_translator_crosses_both_ways(self):
        with tempfile.TemporaryDirectory() as d:
            for writer, reader in ((self.rust, self.py), (self.py, self.rust)):
                path = os.path.join(d, f"t{writer.__name__}.json.gz")
                writer("fit", "--arch", "enc>13>dec", "--task", "plural", "--epochs", "4", "--save", path)
                for cmd in (("evaluate", "--task", "plural"), ("translate", "cat", "glass", "drum", "The dog sat."),
                            ("predict", "boxes")):
                    self.assertEqual(writer(*cmd, "--load", path), reader(*cmd, "--load", path), cmd)


@unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
class TestRustMediaParity(unittest.TestCase):
    """Both ports turn every file into the same symbols: every image format and PNG filter, frames and their
    differences, a sound track, a video file through ffmpeg - so a network file reads the same data the same way."""

    def encode(self, cmd, arch, x):
        return subprocess.run([*cmd, "network", "encode", x, "--arch", arch], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout

    def test_the_same_symbols(self):
        py, rust = [sys.executable, "-m", "latticefsm"], [RUST]
        with tempfile.TemporaryDirectory() as d:
            f = media_files(d)
            cases = [("image>3", f[k]) for k in ("rgb.png", "gray.png", "rgba.png", "pal.png", "rgb.bmp", "rgb.ppm",
                                                  "gray.pgm")]
            cases += [("audio>3", f["wav"]), ("diff>3", f["clip"]), ("video>3", f["clip"]),
                      ("text>3", "Hello, World! It's 42.")]
            import shutil as sh
            if sh.which("ffmpeg"):
                mp4 = os.path.join(d, "clip.mp4")
                made = subprocess.run(["ffmpeg", "-loglevel", "error", "-framerate", "8", "-i",
                                       os.path.join(f["clip"], "frame%02d.ppm"), "-i", f["wav"], "-pix_fmt", "yuv420p",
                                       "-shortest", mp4], capture_output=True)
                if made.returncode == 0:
                    cases.append(("video>3", mp4))
            for arch, x in cases:
                self.assertEqual(self.encode(py, arch, x), self.encode(rust, arch, x), (arch, x))


@unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
class TestRustNGramParity(unittest.TestCase):
    """The N-gram model is the same in both ports: a model written by either predicts, writes and measures the same
    in the other, for every unit; base64 encodes alike; autotrain asks Ollama the same questions and records the same
    rounds."""

    def cli(self, port, *args, env=None):
        cmd = [sys.executable, "-m", "latticefsm"] if port == "py" else [RUST]
        return subprocess.run([*cmd, *args], cwd=ROOT, check=True, capture_output=True, text=True, env=env).stdout

    def test_models_cross_both_ways(self):
        with tempfile.TemporaryDirectory() as d:
            for unit, order in (("word", 3), ("char", 5), ("sound", 3), ("base64", 8)):
                data = "data/corpus/sentences.txt" if unit != "word" else "data/corpus"
                outs = {}
                for writer in ("py", "rs"):
                    path = os.path.join(d, f"{unit}-{writer}.json")
                    self.cli(writer, "ngram", "train", "--unit", unit, "--order", str(order), "--data", data,
                             "--save", path)
                    for reader in ("py", "rs"):
                        outs[(writer, reader)] = (
                            self.cli(reader, "ngram", "predict", "the cat", "", "--load", path, "-k", "6"),
                            self.cli(reader, "ngram", "generate", "the", "a dog", "--load", path, "--max-tokens", "40"),
                            self.cli(reader, "ngram", "evaluate", "--load", path, "--data", data))
                self.assertEqual(len(set(outs.values())), 1, unit)

    def test_predicting_while_training_measures_the_same(self):
        for unit, order in (("word", 3), ("char", 5), ("base64", 8)):
            args = ("ngram", "train", "--unit", unit, "--order", str(order), "--data", "data/corpus/sentences.txt",
                    "--self-credit", "0.3")
            out = self.cli("py", *args)
            self.assertIn("while training: predicted", out)
            self.assertEqual(out, self.cli("rs", *args), unit)

    def test_base64_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            f = media_files(d)
            self.assertEqual(self.cli("py", "base64", "encode", f["rgb.png"]), self.cli("rs", "base64", "encode", f["rgb.png"]))
            outs = set()
            for writer in ("py", "rs"):
                path = os.path.join(d, f"m-{writer}.json")
                self.cli(writer, "ngram", "train", "--unit", "base64", "--order", "6", "--data", f["clip"], "--save", path)
                for reader in ("py", "rs"):
                    out = os.path.join(d, f"{writer}{reader}.bin")
                    self.cli(reader, "ngram", "generate", "--load", path, "--out", out, "--prefix-from",
                             os.path.join(f["clip"], "frame00.ppm"), "--prefix-bytes", "15", "--max-tokens", "300")
                    with open(out, "rb") as fh:
                        outs.add(fh.read())
            self.assertEqual(len(outs), 1)

    def test_autotrain_asks_and_records_the_same(self):
        from fake_ollama import FakeOllama
        results = {}
        for port in ("py", "rs"):
            fake = FakeOllama()
            try:
                env = dict(os.environ, OLLAMA_HOST=fake.url)
                log = self.cli(port, "ngram", "autotrain", "--ollama", "--model", "fake-model", "--data",
                               "data/corpus/sentences.txt", "--held-out", "5", "--rounds", "2", "--lines", "4",
                               "--continuations", "3", "--judge", "4", "--patience", "9", env=env)
                results[port] = (log, list(fake.prompts))
            finally:
                fake.close()
        self.assertEqual(results["py"], results["rs"])
        self.assertIn("round 2:", results["py"][0])


class TestRustChainParity(unittest.TestCase):
    """The chain is the same in both ports: a chain written by either walks, predicts, writes, sweeps and measures
    the same in the other, for every unit."""

    def cli(self, port, *args):
        cmd = [sys.executable, "-m", "latticefsm"] if port == "py" else [RUST]
        return subprocess.run([*cmd, *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout

    def test_chains_cross_both_ways(self):
        with tempfile.TemporaryDirectory() as d:
            for unit, nodes, data in (("char", 8, "data/corpus"), ("word", 4, "data/corpus"),
                                      ("base64", 6, "data/corpus/sentences.txt")):
                outs = {}
                for writer in ("py", "rs"):
                    path = os.path.join(d, f"{unit}-{writer}.json")
                    trained = self.cli(writer, "chain", "train", "--unit", unit, "--nodes", str(nodes), "--data",
                                       data, "--self-credit", "0.2", "--save", path)
                    for reader in ("py", "rs"):
                        outs[(writer, reader)] = (
                            trained,
                            self.cli(reader, "chain", "predict", "the bra", "", "--load", path, "-k", "5"),
                            self.cli(reader, "chain", "generate", "the brain", "--load", path, "--max-tokens", "30"),
                            self.cli(reader, "chain", "sweep", "--load", path, "--data", data, "--thresholds",
                                     "0,0.5,0.9,1"),
                            self.cli(reader, "chain", "credit", "the brain is", "--load", path))
                self.assertEqual(len(set(outs.values())), 1, unit)
                self.assertIn("input nodes traversed", outs[("py", "py")][0])


class TestRustTeachingParity(unittest.TestCase):
    """Teaching a reply and training until N of N print the same in both ports."""

    def test_teach_and_until(self):
        with tempfile.TemporaryDirectory() as d:
            tiny = os.path.join(d, "tiny.txt")
            with open(tiny, "w") as f:
                f.write("the cat sat on the mat\nthe cat ate the fish\na dog sat on the log\n")
            for args in (("chain", "teach", "what is the brain", "a lattice of adaptive edges."),
                         ("ngram", "teach", "hello", "hi there, how are you today?"),
                         ("chain", "train", "--data", tiny, "--held-out", "0", "--until"),
                         ("ngram", "train", "--data", "data/corpus/sentences.txt", "--until"),
                         ("chain", "train", "--data", "data/corpus/sentences.txt", "--until")):
                outs = [subprocess.run([*cmd, *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
                        for cmd in ([sys.executable, "-m", "latticefsm"], [RUST])]
                self.assertEqual(outs[0], outs[1], args)
                self.assertTrue("reached" in outs[0] or "right after" in outs[0], outs[0])


class TestRustCompressionParity(unittest.TestCase):
    """The code is the same, byte for byte, from either port, and either port rebuilds the other's."""

    @unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
    def test_the_two_ports_write_the_same_code(self):
        m = taught()
        with tempfile.TemporaryDirectory() as d:
            mpath = os.path.join(d, "m.json")
            m.save(mpath)
            for precision in ("exact", "float32", "float16"):
                rpath = os.path.join(d, f"r-{precision}.json")
                subprocess.run([RUST, "compress", "--load", mpath, "--precision", precision, "--core", rpath],
                               check=True, capture_output=True)
                with open(rpath) as f:
                    rust = json.load(f)
                ours = compress(m, precision).to_dict()
                for key in ("touched", "ints", "floats", "int_width", "precision", "center"):
                    self.assertEqual(rust[key], ours[key], (precision, key))
            back = os.path.join(d, "back.json")
            compress(m).save(os.path.join(d, "p.json"))
            subprocess.run([RUST, "expand", "--core", os.path.join(d, "p.json"), "--save", back], check=True,
                           capture_output=True)
            self.assertEqual(load_machine(back).to_dict()["lattice"]["edges"], m.to_dict()["lattice"]["edges"])
            self.assertEqual(load_core(os.path.join(d, "r-exact.json")).decompress().to_dict()["lattice"],
                             m.to_dict()["lattice"])

    @unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
    def test_half_floats_round_the_same_way(self):
        # values that sit on rounding boundaries, the subnormal range, and the clamp
        m = Machine(2, "a")
        e = m.edge(0, "a", 1)
        values = [0.1, 1/3, 2049.0, 2051.0, 65504.0, 1e9, -1e9, 6e-5, 6.1e-5, 3e-8, 1.0009765625, 1.00146484375, -0.0]
        e.seen = 1
        e.recent, e.rewarded, e.punished, e.width = values[0:4]
        e.weighting = Weighting(*values[4:10], rate=0.1)
        e.features = tuple(values[8:13])
        with tempfile.TemporaryDirectory() as d:
            mpath, rpath = os.path.join(d, "m.json"), os.path.join(d, "r.json")
            m.save(mpath)
            subprocess.run([RUST, "compress", "--load", mpath, "--precision", "float16", "--core", rpath], check=True,
                           capture_output=True)
            with open(rpath) as f:
                self.assertEqual(json.load(f)["floats"], compress(m, "float16").to_dict()["floats"])


class TestRustWalkParity(unittest.TestCase):
    """Focus bands, skip decisions and rearrangement are deterministic, and the two ports agree on them."""

    @unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
    def test_focus_bands_agree(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            Machine().save(path)
            for f in (0.0, 0.07, 1 / 13, 0.3, 0.5, 0.75, 12 / 13, 1.0):
                out = subprocess.run([RUST, "run", "ab", "--load", path, "--focus", repr(f), "--quiet"], check=True,
                                     capture_output=True, text=True).stdout
                self.assertTrue(out.startswith(f"quietly: {focus_index(13, f)} -"), (f, out))

    @unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
    def test_rearrangement_and_the_code_agree(self):
        m = taught(episodes=800)
        with tempfile.TemporaryDirectory() as d:
            path, rpath = os.path.join(d, "m.json"), os.path.join(d, "r.json")
            m.save(path)
            subprocess.run([RUST, "rearrange", "--load", path, "--full", "--save", rpath], check=True, capture_output=True)
            ours = Machine.from_dict(json.loads(json.dumps(m.to_dict())))
            ours.rearrange(full=True)
            rust = load_machine(rpath)
            self.assertEqual(rust.lattice.state_ids, ours.lattice.state_ids)
            self.assertEqual(rust.alphabet, ours.alphabet)
            self.assertEqual(rust.to_dict()["lattice"]["edges"], ours.to_dict()["lattice"]["edges"])
            cpath = os.path.join(d, "c.json")
            subprocess.run([RUST, "compress", "--load", rpath, "--core", cpath], check=True, capture_output=True)
            with open(cpath) as f:
                code = json.load(f)
            self.assertEqual((code["touched"], code["floats"]), (compress(ours).to_dict()["touched"],
                                                                 compress(ours).to_dict()["floats"]))

    @unittest.skipUnless(os.path.exists(RUST), "the Rust crate is not built (make rust-build)")
    def test_quiet_skips_agree(self):
        m = taught(episodes=600, skip=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            m.save(path)
            for text in ("abba", "aabab", "bbab"):
                out = subprocess.run([RUST, "core-run", text, "--core", path], capture_output=True, text=True)
                core = compress(m)
                cpath = os.path.join(d, "c.json")
                core.save(cpath)
                out = subprocess.run([RUST, "core-run", text, "--core", cpath], check=True, capture_output=True,
                                     text=True).stdout
                states, _ = core.run(text)
                self.assertTrue(out.startswith(" -> ".join(map(str, states))), (text, out, states))


def write_png(path, width, height, rgb, ctype=2, palette=None):
    """A PNG written the long way, every row with the next of the five filters, for the readers to undo."""
    import struct as st
    import zlib as zl
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    stride = width * ch
    raw, prev = bytearray(), bytes(stride)
    for y in range(height):
        line = bytes(rgb[y * stride:(y + 1) * stride])
        f = y % 5
        out = bytearray([f])
        for i in range(stride):
            a = line[i - ch] if i >= ch else 0
            b, c = prev[i], prev[i - ch] if i >= ch else 0
            pred = [0, a, b, (a + b) // 2, None][f]
            if f == 4:
                p_ = a + b - c
                pa, pb, pc = abs(p_ - a), abs(p_ - b), abs(p_ - c)
                pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
            out.append((line[i] - pred) & 255)
        raw += out
        prev = line

    def chunk(tag, body):
        return st.pack(">I", len(body)) + tag + body + st.pack(">I", zl.crc32(tag + body) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", st.pack(">IIBBBBB", width, height, 8, ctype, 0, 0, 0))
    if palette is not None:
        data += chunk(b"PLTE", bytes(palette))
    data += chunk(b"IDAT", zl.compress(bytes(raw))) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def write_bmp(path, width, height, rgb):
    import struct as st
    stride = (width * 3 + 3) // 4 * 4
    body = bytearray()
    for y in reversed(range(height)):
        row = bytearray()
        for x in range(width):
            r, g, b = rgb[(y * width + x) * 3:(y * width + x) * 3 + 3]
            row += bytes([b, g, r])
        body += row + bytes(stride - len(row))
    header = b"BM" + st.pack("<IHHI", 54 + len(body), 0, 0, 54) + st.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0,
                                                                         len(body), 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(header + body)


def media_files(d):
    """One picture in every format the readers take, a directory of frames with a sound track, and a spoken word."""
    from latticefsm.media import make_dataset
    w, h = 13, 11
    rgb = [(x * 19 + y * 7 + c * 50) % 256 for y in range(h) for x in range(w) for c in range(3)]
    gray = [(x * 23 + y * 11) % 256 for y in range(h) for x in range(w)]
    files = {}
    write_png(os.path.join(d, "rgb.png"), w, h, rgb)
    write_png(os.path.join(d, "gray.png"), w, h, gray, ctype=0)
    write_png(os.path.join(d, "rgba.png"), w, h, [v for i in range(w * h) for v in rgb[i * 3:i * 3 + 3] + [128]], ctype=6)
    write_png(os.path.join(d, "pal.png"), w, h, [i % 4 for i in range(w * h)], ctype=3,
              palette=[0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 255])
    write_bmp(os.path.join(d, "rgb.bmp"), w, h, rgb)
    with open(os.path.join(d, "rgb.ppm"), "wb") as f:
        f.write(f"P6\n# a comment\n{w} {h}\n255\n".encode() + bytes(rgb))
    with open(os.path.join(d, "gray.pgm"), "w") as f:
        f.write(f"P2\n{w} {h}\n15\n" + " ".join(str(v * 15 // 255) for v in gray) + "\n")
    for name in ("rgb.png", "gray.png", "rgba.png", "pal.png", "rgb.bmp", "rgb.ppm", "gray.pgm"):
        files[name] = os.path.join(d, name)
    clips = make_dataset("video", os.path.join(d, "video"), count=2)
    files["clip"] = os.path.join(os.path.dirname(clips), "clip000")
    files["wav"] = os.path.join(files["clip"], "audio.wav")
    return files


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
