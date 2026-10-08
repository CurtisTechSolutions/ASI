"""The brute-force oracle: prime a small tree by literal insertion and assert the arithmetic is the same tree.

The requirement says every option is inserted by brute force.  This is the one
place that does it: every sequence of length ``1..L`` over ``R`` ids, one at a
time, into a pointer trie with the standard radix insertion (a partial match
splits the edge - ``RadixTrieLLM_RNN/main.py``'s rule).  Over a complete set no
split ever fires and every edge label has length 1, which is the point:
priming leaves nothing for radix compression to do.  The walk over the result
then checks :class:`~radixpair.address.Address` pointer by pointer.
"""

from __future__ import annotations

import itertools
import time

from .address import Address, node_count

__all__ = ["DEFAULT_SIZES", "brute_force", "check_address", "check_all"]

DEFAULT_SIZES = ((2, 5), (3, 4), (5, 3), (7, 2))


class _Node:
    __slots__ = ("label", "children", "end", "seq")

    def __init__(self, label: tuple[int, ...], seq: tuple[int, ...]) -> None:
        self.label = label          # the edge label into this node
        self.children: dict[int, _Node] = {}   # keyed by the first unit of the child's label
        self.end = False
        self.seq = seq              # the whole sequence this node spells


def _common(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def insert(root: _Node, seq: tuple[int, ...]) -> int:
    """Standard radix insertion; returns the number of splits it had to make."""
    node = root
    remaining = seq
    splits = 0
    while True:
        if not remaining:
            node.end = True
            return splits
        child = node.children.get(remaining[0])
        if child is None:
            new = _Node(remaining, node.seq + remaining)
            new.end = True
            node.children[remaining[0]] = new
            return splits
        common = _common(remaining, child.label)
        if common == len(child.label):
            node = child
            remaining = remaining[common:]
            continue
        # partial match: split the child's edge
        splits += 1
        prefix, suffix = child.label[:common], child.label[common:]
        middle = _Node(prefix, node.seq + prefix)
        child.label = suffix
        middle.children[suffix[0]] = child
        node.children[prefix[0]] = middle
        node = middle
        remaining = remaining[common:]


def brute_force(R: int, L: int) -> tuple[_Node, list[_Node], int]:
    """Insert every sequence of length ``1..L``; returns the root, every node, and the splits made."""
    root = _Node((), ())
    splits = 0
    for l in range(1, L + 1):
        for seq in itertools.product(range(R), repeat=l):
            splits += insert(root, seq)
    nodes: list[_Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children.values())
    return root, nodes, splits


def check_address(R: int, L: int) -> dict:
    """Prime by brute force and compare with :class:`Address`; raises ``AssertionError`` on the first disagreement."""
    t0 = time.perf_counter()
    root, nodes, splits = brute_force(R, L)
    address = Address(R, L)
    N = node_count(R, L)
    assert len(nodes) == N, f"the brute-force tree has {len(nodes)} nodes, the arithmetic says {N}"
    assert splits == 0, f"priming a complete set should never split an edge, but split {splits} times"
    ids = set()
    unary = 0
    for node in nodes:
        seq = node.seq
        l = len(seq)
        if node is not root:
            assert len(node.label) == 1, f"edge label {node.label} longer than one unit"
        if len(node.children) == 1:
            unary += 1
        i = address.of(seq)
        assert 0 <= i < N, f"id {i} of {seq} outside the tree"
        ids.add(i)
        assert address.level(i) == l, f"level of {seq}"
        assert address.seq(i) == seq, f"seq of id {i}"
        if l < L:
            assert len(node.children) == R, f"{seq} has {len(node.children)} children, not {R}"
            for x, child in node.children.items():
                assert address.append(i, l, x) == address.of(child.seq), f"append({seq}, {x})"
        else:
            assert not node.children, f"a final node {seq} has children"
        if l >= 1:
            assert address.drop_newest(i, l) == address.of(seq[:-1]), f"drop_newest({seq})"
            assert address.drop_oldest(i, l) == address.of(seq[1:]), f"drop_oldest({seq})"
            assert address.newest(i, l) == seq[-1] and address.oldest(i, l) == seq[0], f"newest/oldest({seq})"
        start, stop = address.block(i, l) if l < L else (0, 0)
        if l < L:
            assert stop - start == R and start == address.append(i, l, 0), f"block({seq})"
    assert ids == set(range(N)), "the ids are not a bijection onto 0..N-1"
    assert unary == 0, f"{unary} nodes with exactly one child - compression would have something to do"
    return {"R": R, "L": L, "nodes": N, "splits": splits, "unary": unary, "ok": True,
            "seconds": round(time.perf_counter() - t0, 4)}


def check_all(sizes=DEFAULT_SIZES) -> list[dict]:
    return [check_address(R, L) for R, L in sizes]
