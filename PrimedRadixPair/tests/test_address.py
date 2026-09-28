"""The addressing of a primed tree, held to the brute-force insertion of every option."""

import itertools
import random
import unittest

from radixpair.address import Address, node_count
from radixpair.check import DEFAULT_SIZES, check_address, check_all


class TestBruteForceOracle(unittest.TestCase):
    def test_the_arithmetic_tree_is_the_brute_force_tree(self):
        for result in check_all(DEFAULT_SIZES):
            self.assertTrue(result["ok"])
            self.assertEqual(result["splits"], 0)
            self.assertEqual(result["unary"], 0)
            self.assertEqual(result["nodes"], node_count(result["R"], result["L"]))

    def test_one_size_reports_its_numbers(self):
        r = check_address(3, 3)
        self.assertEqual(r["nodes"], 40)


class TestAddress(unittest.TestCase):
    def test_the_worked_example(self):
        a = Address(3, 2)
        want = {(): 0, (0,): 1, (1,): 2, (2,): 3, (0, 0): 4, (0, 1): 5, (0, 2): 6, (1, 0): 7, (1, 1): 8,
                (1, 2): 9, (2, 0): 10, (2, 1): 11, (2, 2): 12}
        for seq, i in want.items():
            self.assertEqual(a.of(seq), i)
            self.assertEqual(a.seq(i), seq)
        self.assertEqual(a.N, 13)

    def test_level_by_bisection_at_every_id(self):
        a = Address(3, 4)
        for i in range(a.N):
            self.assertEqual(a.level(i), len(a.seq(i)))

    def test_the_children_block_is_contiguous(self):
        a = Address(5, 3)
        for i in range(a.bases[3]):
            l = a.level(i)
            start, stop = a.block(i, l)
            self.assertEqual(start, a.append(i, l, 0))
            self.assertEqual(stop - start, a.R)
            self.assertEqual(stop - 1, a.append(i, l, a.R - 1))

    def test_every_move_is_undone_by_its_inverse(self):
        a = Address(4, 3)
        rng = random.Random(1)
        for _ in range(500):
            l = rng.randrange(0, 3)
            seq = tuple(rng.randrange(4) for _ in range(l))
            i = a.of(seq)
            x = rng.randrange(4)
            child = a.append(i, l, x)
            self.assertEqual(a.drop_newest(child, l + 1), i)
            self.assertEqual(a.newest(child, l + 1), x)
            self.assertEqual(a.seq(a.drop_oldest(child, l + 1)), (seq + (x,))[1:])
            if l:
                self.assertEqual(a.oldest(i, l), seq[0])

    def test_substrings_against_a_naive_enumeration(self):
        a = Address(3, 3)
        rng = random.Random(2)
        for _ in range(50):
            ids = [rng.randrange(3) for _ in range(rng.randrange(0, 12))]
            naive = []
            for t in range(1, len(ids) + 1):
                for l in range(min(3, t), 0, -1):
                    naive.append((t, l, a.of(tuple(ids[t - l:t]))))
            self.assertEqual(list(a.substrings(ids)), naive)

    def test_node_count(self):
        self.assertEqual(node_count(33, 4), 1_222_981)
        self.assertEqual(node_count(92, 3), 787_245)
        self.assertEqual(node_count(50_260, 1), 50_261)
        self.assertEqual(node_count(2_004, 2), 4_018_021)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            Address(1, 3)
        with self.assertRaises(ValueError):
            Address(3, 0)
        a = Address(3, 2)
        with self.assertRaises(ValueError):
            a.of((0, 1, 2))
        with self.assertRaises(ValueError):
            a.of((3,))
        with self.assertRaises(ValueError):
            a.level(13)
        with self.assertRaises(ValueError):
            a.append(4, 2, 0)
        with self.assertRaises(ValueError):
            a.drop_oldest(0, 0)
        with self.assertRaises(ValueError):
            a.block(4, 2)


if __name__ == "__main__":
    unittest.main()
