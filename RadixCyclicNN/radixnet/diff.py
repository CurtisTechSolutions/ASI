"""Unit diff between what the network wrote and what the teacher corrected.

The tutor's marking used to be a verdict on a whole sentence: the failed
attempt was garbage, the correction was gospel, and every edge of either path
moved by the same amount.  Most of a corrected sentence is, however, word for
word what the network wrote - the teacher changes a tense, an article, a
plural - and punishing the parts that were already right is a tax on the
trigrams that earned their place.

This module aligns the two sentences and says *where* they differ, so the
model can move only the nodes the mistake ran through
(:meth:`radixnet.countnet.CountRewardNet.correct`).  The alignment is a plain
longest-common-subsequence diff over *units*, after the shared prefix and
suffix have been trimmed: deterministic, dependency-free and cheap at sentence
length, and written the same way in Go
(``go/radixnet/diff.go``) so both implementations mark the same characters.

A unit is whatever the model's :class:`~radixnet.encoding.Encoding` says it is,
so a word model's diff marks whole words and its spans index the same units the
graph's labels and offsets do.  Pass ``encoding=`` to every function here; the
default is the character encoding, which is what it always was.

    >>> [ (e.op, e.wrong, e.right) for e in edits("the cat sit", "the cat sits") ]
    [('equal', 'the cat sit', 'the cat sit'), ('insert', '', 's')]
"""

from __future__ import annotations

import dataclasses

from .encoding import Encoding

__all__ = ["Edit", "MAX_CELLS", "MIN_EQUAL_RUN", "changed_spans", "edits", "summary"]

MAX_CELLS = 4_000_000
"""Above this many cells (|a| x |b| after trimming) the middle is marked changed as a whole instead of aligned."""

MIN_EQUAL_RUN = 3
"""Changes any closer than a trigram are one change: no trigram fits in the gap, so the same nodes are to blame.

This is the default encoding's rule; a diff run in another encoding uses that
encoding's ``n`` the same way."""

_CHARS = Encoding()


@dataclasses.dataclass(frozen=True)
class Edit:
    """One step of the alignment: ``op`` is ``equal``, ``replace``, ``delete`` or ``insert``.

    The spans are half-open ranges in the encoding's units - characters by
    default, words under a word encoding - ``wrong[a0:a1]`` against
    ``right[b0:b1]``; ``delete`` is text the network wrote and the teacher
    struck out, ``insert`` text the teacher added.
    """

    op: str
    a0: int
    a1: int
    b0: int
    b1: int
    wrong: str = ""
    right: str = ""

    def to_dict(self) -> dict:
        return {"op": self.op, "wrong": self.wrong, "right": self.right}


def edits(wrong: str, right: str, encoding: Encoding | None = None) -> list[Edit]:
    """Align ``wrong`` against ``right`` one unit at a time; equal runs included, in order."""
    enc = encoding or _CHARS
    min_run = enc.n
    a, b = list(enc.units(str(wrong))), list(enc.units(str(right)))
    n, m = len(a), len(b)
    head = 0
    while head < n and head < m and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < n - head and tail < m - head and a[n - 1 - tail] == b[m - 1 - tail]:
        tail += 1
    mid_a, mid_b = a[head : n - tail], b[head : m - tail]
    out: list[Edit] = []
    if head:
        out.append(_edit("equal", 0, head, 0, head, a, b, enc))
    if mid_a or mid_b:
        if not mid_a:
            out.append(_edit("insert", head, head, head, m - tail, a, b, enc))
        elif not mid_b:
            out.append(_edit("delete", head, n - tail, head, head, a, b, enc))
        elif len(mid_a) * len(mid_b) > MAX_CELLS:  # too big to align: one change covering the middle
            out.append(_edit("replace", head, n - tail, head, m - tail, a, b, enc))
        else:
            out.extend(_align(mid_a, mid_b, head, head, a, b, enc))
    if tail:
        out.append(_edit("equal", n - tail, n, m - tail, m, a, b, enc))
    return _coalesce(_merge(out, a, b, enc), a, b, enc, min_run)


def changed_spans(
    wrong: str, right: str, encoding: Encoding | None = None
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The unit ranges the two sentences disagree on: ``(spans of wrong, spans of right)``.

    An insertion is an empty span on the side that lacks the text, kept at the
    position where it belongs: the model blames the step that walked past it.
    """
    left: list[tuple[int, int]] = []
    the_right: list[tuple[int, int]] = []
    for edit in edits(wrong, right, encoding):
        if edit.op == "equal":
            continue
        left.append((edit.a0, edit.a1))
        the_right.append((edit.b0, edit.b1))
    return left, the_right


def summary(wrong: str, right: str, limit: int = 8, encoding: Encoding | None = None) -> list[dict]:
    """The changes as ``[{"op", "wrong", "right"}]`` for a lesson record (equal runs dropped)."""
    changes = [e.to_dict() for e in edits(wrong, right, encoding) if e.op != "equal"]
    return changes[:limit] if limit > 0 else changes


# -- the alignment ---------------------------------------------------------


def _edit(op: str, a0: int, a1: int, b0: int, b1: int, a: list[str], b: list[str], enc: Encoding) -> Edit:
    return Edit(op, a0, a1, b0, b1, enc.join(*a[a0:a1]), enc.join(*b[b0:b1]))


def _align(
    a: list[str], b: list[str], off_a: int, off_b: int, whole_a: list[str], whole_b: list[str], enc: Encoding
) -> list[Edit]:
    """Longest common subsequence over two trimmed middles, walked forward into edits."""
    n, m = len(a), len(b)
    # lcs[i][j] = length of the longest common subsequence of a[i:] and b[j:]
    lcs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = lcs[i], lcs[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if a[i] == b[j] else max(nxt[j], row[j + 1])
    out: list[Edit] = []
    i = j = 0
    while i < n or j < m:
        if i < n and j < m and a[i] == b[j]:
            start_i, start_j = i, j
            while i < n and j < m and a[i] == b[j]:
                i, j = i + 1, j + 1
            out.append(_edit("equal", off_a + start_i, off_a + i, off_b + start_j, off_b + j, whole_a, whole_b, enc))
            continue
        start_i, start_j = i, j
        # the same tie-break as the Go implementation: a deletion first when both are equally good
        while (i < n or j < m) and not (i < n and j < m and a[i] == b[j]):
            if i < n and (j >= m or lcs[i + 1][j] >= lcs[i][j + 1]):
                i += 1
            else:
                j += 1
        op = "replace" if i > start_i and j > start_j else ("delete" if i > start_i else "insert")
        out.append(_edit(op, off_a + start_i, off_a + i, off_b + start_j, off_b + j, whole_a, whole_b, enc))
    return out


def _merge(items: list[Edit], a: list[str], b: list[str], enc: Encoding) -> list[Edit]:
    """Join neighbouring edits of the same kind (the trimming can split a run in two)."""
    out: list[Edit] = []
    for edit in items:
        if out and _joins(out[-1], edit):
            last = out[-1]
            op = last.op if last.op == edit.op else "replace"
            out[-1] = _edit(op, last.a0, edit.a1, last.b0, edit.b1, a, b, enc)
            continue
        out.append(edit)
    return out


def _coalesce(items: list[Edit], a: list[str], b: list[str], enc: Encoding, min_run: int) -> list[Edit]:
    """Swallow equal runs shorter than a trigram between two changes ("mat" -> "park", not "m" -> "p" and "t" -> "rk")."""
    out: list[Edit] = []
    for edit in items:
        if (
            len(out) >= 2
            and edit.op != "equal"
            and out[-1].op == "equal"
            and out[-2].op != "equal"
            and out[-1].a1 - out[-1].a0 < min_run
        ):
            gap, before = out.pop(), out.pop()
            op = "replace" if (edit.a1 > before.a0) and (edit.b1 > before.b0) else before.op
            out.append(_edit(op, before.a0, edit.a1, before.b0, edit.b1, a, b, enc))
            del gap
            continue
        out.append(edit)
    return out


def _joins(last: Edit, edit: Edit) -> bool:
    if last.a1 != edit.a0 or last.b1 != edit.b0:
        return False
    if last.op == edit.op:
        return True
    return last.op != "equal" and edit.op != "equal"  # delete + insert = replace
