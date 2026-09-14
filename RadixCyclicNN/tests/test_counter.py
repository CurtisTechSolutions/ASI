"""Tests for radixnet.counter and the wrapping of every counter in the model.

A counter that only grows eventually overflows whatever holds it, so each one
is a two-digit odometer: it is set back to 0 at ``COUNTER_LIMIT`` and the reset
is counted.  What the tests below insist on is that nothing is *lost* by the
wrap - the exact totals, the weights, the shares, the rankings and a saved file
all come out the same as they would have with unbounded integers.
"""

import copy
import json
import os
import pickle
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radixnet.counter import COUNTER_LIMIT, CyclicCounter, as_float, carry, carry_series, total  # noqa: E402
from radixnet.countnet import CountRewardNet  # noqa: E402
from radixnet.graph import RadixCyclicGraph  # noqa: E402
from radixnet.model import RadixNet, meta_add, meta_counter  # noqa: E402

TEXTS = ["the cat sat on the mat", "the dog ate the bone", "the cat ate the fish"]


def near_the_limit(graph, short_by: int = 1) -> None:
    """Wind every counter of ``graph`` to ``short_by`` events before its reset."""
    step = COUNTER_LIMIT - short_by
    graph.count = [c + step for c in graph.count]
    graph.edge_count = [c + step for c in graph.edge_count]
    graph.traversals += step
    if hasattr(graph, "total_traversals"):
        graph.total_traversals += step


class TestCyclicCounter(unittest.TestCase):
    def test_carry_moves_whole_turns_into_the_resets(self):
        self.assertEqual(carry(0), (0, 0))
        self.assertEqual(carry(COUNTER_LIMIT - 1), (COUNTER_LIMIT - 1, 0))
        self.assertEqual(carry(COUNTER_LIMIT), (0, 1))
        self.assertEqual(carry(COUNTER_LIMIT + 7, 2), (7, 3))
        self.assertEqual(carry(3 * COUNTER_LIMIT + 5), (5, 3))

    def test_the_resets_wrap_too_so_the_pair_itself_cycles(self):
        self.assertEqual(carry(COUNTER_LIMIT, COUNTER_LIMIT - 1), (0, 0))
        self.assertEqual(CyclicCounter(0, COUNTER_LIMIT + 4).resets, 4)

    def test_total_and_float_are_the_events_counted(self):
        self.assertEqual(total(7, 3), 3 * COUNTER_LIMIT + 7)
        self.assertEqual(as_float(7, 3), float(3 * COUNTER_LIMIT + 7))
        self.assertEqual(as_float(7, 0), 7.0)

    def test_both_digits_survive_a_json_number_exactly(self):
        """The point of the limit: value and resets are each below 2 ** 53, so a file never rounds one."""
        for reading in (COUNTER_LIMIT - 1, COUNTER_LIMIT // 3, 1):
            counter = CyclicCounter(reading, reading)
            pair = json.loads(json.dumps(counter.to_pair()))
            self.assertEqual(pair, [reading, reading])
            self.assertEqual(CyclicCounter.from_pair(*pair), counter)
            self.assertEqual(int(float(counter.value)), counter.value)
            self.assertEqual(int(float(counter.resets)), counter.resets)

    def test_counting_past_the_limit_sets_the_value_back_to_zero(self):
        counter = CyclicCounter(COUNTER_LIMIT - 2)
        self.assertEqual((counter.value, counter.resets), (COUNTER_LIMIT - 2, 0))
        counter += 1
        self.assertEqual((counter.value, counter.resets), (COUNTER_LIMIT - 1, 0))
        counter += 1
        self.assertEqual((counter.value, counter.resets), (0, 1))
        counter += 3
        self.assertEqual((counter.value, counter.resets), (3, 1))
        self.assertEqual(counter.total, COUNTER_LIMIT + 3)

    def test_it_compares_and_hashes_as_the_number_of_events(self):
        counter = CyclicCounter(COUNTER_LIMIT - 1) + 4
        self.assertEqual(counter, COUNTER_LIMIT + 3)
        self.assertEqual(counter, CyclicCounter(3, 1))
        self.assertNotEqual(counter, 3)  # the reading alone is not the count
        self.assertGreater(counter, CyclicCounter(COUNTER_LIMIT - 1))
        self.assertLess(CyclicCounter(5), CyclicCounter(6))
        self.assertEqual(hash(CyclicCounter(5)), hash(5))
        self.assertEqual(sorted([CyclicCounter(3, 1), CyclicCounter(9)]), [CyclicCounter(9), CyclicCounter(3, 1)])
        self.assertNotEqual(counter, "3")

    def test_it_is_an_immutable_value_copies_and_pickles_included(self):
        counter = CyclicCounter(5)
        with self.assertRaises(AttributeError):
            counter.value = 6
        later = counter + 1
        self.assertEqual((counter.value, later.value), (5, 6))  # += rebinds, it does not mutate
        self.assertIs(copy.deepcopy(counter), counter)
        self.assertEqual(pickle.loads(pickle.dumps(CyclicCounter(3, 2))), CyclicCounter(3, 2))
        self.assertEqual(CyclicCounter.from_pair(*CyclicCounter(3, 2).to_pair()), CyclicCounter(3, 2))
        self.assertEqual(CyclicCounter.from_pair(4), CyclicCounter(4))  # a file without the resets field

    def test_carry_series_wraps_a_parallel_list_and_stays_sparse(self):
        values = [1, COUNTER_LIMIT + 4, 2, 2 * COUNTER_LIMIT]
        resets: dict[int, int] = {}
        self.assertEqual(carry_series(values, resets), 2)
        self.assertEqual(values, [1, 4, 2, 0])
        self.assertEqual(resets, {1: 1, 3: 2})  # only the ids that wrapped are stored
        self.assertEqual(carry_series(values, resets), 0)


class TestGraphCounters(unittest.TestCase):
    def test_visit_counters_wrap_and_keep_the_exact_totals(self):
        model = CountRewardNet(seed=3)
        model.train(TEXTS, epochs=2)
        graph = model.graph
        near_the_limit(graph)
        before = {e: graph.edge_traversals(e) for e in range(len(graph.edge_count))}
        nodes_before = {i: graph.node_count(i) for i in range(len(graph.count))}

        model.train(TEXTS[:1], epochs=1)  # training carries at the end of every epoch

        self.assertTrue(all(0 <= c < COUNTER_LIMIT for c in graph.count), "a node counter did not wrap")
        self.assertTrue(all(0 <= c < COUNTER_LIMIT for c in graph.edge_count), "an edge counter did not wrap")
        self.assertTrue(graph.count_resets and graph.edge_count_resets, "no reset was recorded")
        for e, was in before.items():
            self.assertGreaterEqual(graph.edge_traversals(e), was, "an edge lost its history")
        for i, was in nodes_before.items():
            self.assertGreaterEqual(graph.node_count(i), was, "a node lost its history")
        graph.check_invariants()

    def test_the_weights_are_what_they_would_be_without_any_wrapping(self):
        model = CountRewardNet(seed=4)
        model.train(TEXTS, epochs=2)
        graph = model.graph
        near_the_limit(graph)
        model.train(TEXTS, epochs=1)
        wrapped = list(graph.edge_w)
        wrapped_shares = [graph.shares(p) for p in range(len(graph.labels))]

        unwrapped = copy.deepcopy(graph)  # the same totals in one unbounded integer each
        unwrapped.edge_count = [graph.edge_traversals(e) for e in range(len(graph.edge_count))]
        unwrapped.edge_count_resets = {}
        unwrapped.recompute_weights()

        self.assertEqual(wrapped, unwrapped.edge_w)
        self.assertEqual(wrapped_shares, [unwrapped.shares(p) for p in range(len(unwrapped.labels))])

    def test_a_wrapped_graph_saves_and_loads_unchanged(self):
        model = CountRewardNet(seed=5)
        model.train(TEXTS, epochs=2)
        near_the_limit(model.graph)
        model.train(TEXTS, epochs=1)
        doc = json.loads(json.dumps(model.to_dict()))
        self.assertIn("count_resets", doc["graph"]["nodes"])
        self.assertIn("count_resets", doc["graph"]["edges"])
        self.assertEqual(doc["graph"]["format_version"], 2)

        loaded = CountRewardNet.from_dict(doc)
        # dead nodes are compacted away by the save, so the documents are what must match
        reloaded = loaded.to_dict()["graph"]
        blocks = (("nodes", "count"), ("nodes", "count_resets"), ("edges", "count"), ("edges", "count_resets"), ("edges", "w"))
        for block, key in blocks:
            self.assertEqual(reloaded[block][key], doc["graph"][block][key], f"{block}.{key}")
        self.assertEqual(loaded.graph.total_traversals, model.graph.total_traversals)
        self.assertEqual(loaded.graph.traversals, model.graph.traversals)
        self.assertEqual(loaded.predict("the ", length=8).text, model.predict("the ", length=8).text)

    def test_an_unwrapped_graph_writes_no_reset_fields(self):
        model = CountRewardNet(seed=6)
        model.train(TEXTS, epochs=1)
        doc = model.to_dict()["graph"]
        self.assertNotIn("count_resets", doc["nodes"])
        self.assertNotIn("count_resets", doc["edges"])
        self.assertEqual((doc["version_resets"], doc["traversals_resets"]), (0, 0))
        self.assertEqual(doc["weights"]["total_traversals_resets"], 0)

    def test_a_format_1_file_loads_with_its_counts_wrapped(self):
        """Files written before the counters were cyclic hold plain integers, over the limit or not."""
        model = CountRewardNet(seed=7)
        model.train(TEXTS, epochs=1)
        doc = model.to_dict()
        graph = doc["graph"]
        graph["format_version"] = 1
        for key in ("version_resets", "structure_version_resets", "traversals", "traversals_resets"):
            graph.pop(key)
        graph["weights"].pop("total_traversals_resets")
        graph["nodes"]["count"] = [c + COUNTER_LIMIT for c in graph["nodes"]["count"]]
        graph["edges"]["count"] = [c + COUNTER_LIMIT for c in graph["edges"]["count"]]

        loaded = CountRewardNet.from_dict(json.loads(json.dumps(doc)))
        self.assertTrue(all(0 <= c < COUNTER_LIMIT for c in loaded.graph.edge_count))
        self.assertEqual(loaded.graph.edge_traversals(0), model.graph.edge_traversals(0) + COUNTER_LIMIT)
        self.assertTrue(loaded.graph.traversals.resets, "the traversal total must bound the loaded counts")

    def test_a_reset_array_of_the_wrong_length_is_rejected(self):
        model = CountRewardNet(seed=11)
        model.train(TEXTS[:1], epochs=1)
        for block in ("nodes", "edges"):
            with self.subTest(block=block):
                doc = model.to_dict()
                doc["graph"][block]["count_resets"] = [1, 2]
                with self.assertRaises(ValueError):
                    CountRewardNet.from_dict(doc)

    def test_the_carry_sweep_is_skipped_until_a_counter_can_have_wrapped(self):
        graph = RadixCyclicGraph(seed=1)
        graph.observe_sequence(graph._encoder.encode("hello world"))
        graph.count[2] = COUNTER_LIMIT + 3  # nothing has counted that far, so nothing is swept
        self.assertEqual(graph.carry_counters(), 0)
        self.assertEqual(graph.count[2], COUNTER_LIMIT + 3)
        self.assertEqual(graph.carry_counters(force=True), 1)
        self.assertEqual((graph.count[2], graph.count_resets[2]), (3, 1))

    def test_version_stamps_stay_exact_across_a_reset(self):
        graph = RadixCyclicGraph(seed=1)
        graph.version = CyclicCounter(COUNTER_LIMIT - 1)
        stamp = graph.version
        graph.version += 1
        self.assertEqual((graph.version.value, graph.version.resets), (0, 1))
        self.assertNotEqual(graph.version, stamp, "a wrapped stamp must not look like the one before it")
        self.assertNotEqual(graph.version, CyclicCounter(0))

    def test_a_merge_keeps_the_larger_of_the_two_exact_counts(self):
        graph = RadixCyclicGraph(seed=1)
        graph.observe_sequence(graph._encoder.encode("abcdef"))
        p = graph.trigram_index["abc"][0]
        c = graph.children[p] and next(iter(graph.children[p]))
        graph.count[p], graph.count[c] = 5, 1
        graph.count_resets[c] = 1  # the child has counted a whole turn more
        graph.merge_child(p)
        self.assertEqual(graph.node_count(p), COUNTER_LIMIT + 1)

    def test_the_most_visited_node_is_found_across_a_reset(self):
        model = RadixNet(seed=2, backend="python")
        model.train(TEXTS, epochs=1)
        graph = model.graph
        node = graph.trigram_index["the"][0]
        quiet = next(i for i in range(2, len(graph.labels)) if graph.alive[i] and i != node)
        graph.count[node], graph.count[quiet] = 1, 10
        graph.count_resets[node] = 1  # a wrapped counter reads low but counted the most
        self.assertGreater(graph.node_count(node), graph.node_count(quiet))
        self.assertEqual(model._best_node_with_prefix(graph.labels[node][:3]), node)


class TestMetaCounters(unittest.TestCase):
    def test_lifetime_counters_wrap_into_their_reset_field(self):
        meta = {}
        self.assertEqual(meta_add(meta, "epochs_total", COUNTER_LIMIT - 1), COUNTER_LIMIT - 1)
        self.assertEqual(meta["epochs_total_resets"], 0)
        self.assertEqual(meta_add(meta, "epochs_total", 2), 1)
        self.assertEqual(meta["epochs_total_resets"], 1)
        self.assertEqual(meta_counter(meta, "epochs_total").total, COUNTER_LIMIT + 1)

    def test_training_reports_the_wrapped_epoch_and_its_resets(self):
        model = CountRewardNet(seed=8)
        model.meta["trained_chars"] = COUNTER_LIMIT - 5
        model.meta["epochs_total"] = COUNTER_LIMIT - 1
        records = model.train(TEXTS, epochs=2)
        self.assertEqual([r["epoch"] for r in records], [0, 1])
        stats = model.stats()
        self.assertEqual((stats["epochs_total"], stats["epochs_total_resets"]), (1, 1))
        self.assertEqual(stats["trained_chars_resets"], 1)
        self.assertLess(stats["trained_chars"], COUNTER_LIMIT)

    def test_a_saved_model_round_trips_its_wrapped_lifetime_counters(self):
        model = RadixNet(seed=9, backend="python")
        model.meta["twonrl_runs"] = COUNTER_LIMIT - 1
        meta_add(model.meta, "twonrl_runs", 3)
        loaded = RadixNet.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(meta_counter(loaded.meta, "twonrl_runs").total, COUNTER_LIMIT + 2)


if __name__ == "__main__":
    unittest.main()
