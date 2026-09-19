"""Layer 2: one expert - a radix tree that predicts the next character.

The tree is a **path-compressed context trie**.  Every window of up to ``D+1``
characters of the text it was given is inserted; a node's path spells a context,
its children are the characters observed after that context, and a run of nodes
with one child each is stored as one node holding the whole run.  That is the
radix rule and it is also the model's central claim: *a unary chain is a
deterministic continuation, so it is stored once and costs no decision.*

Prediction at a context walks the longest suffix of the context that the tree
knows, and then:

* **inside a segment** the continuation is the segment's next character, at
  probability 1 before smoothing - nothing to decide and nothing to learn;
* **at a branch** the next character is a soft choice among the node's children,
  scored by ``w_c * f_p * f_c`` and normalised by a softmax over the siblings.

That score, that softmax and its gradients are the one-hop local rule of
``RadixCyclicNN/radixnet/backend.py``, reproduced here for a tree instead of a
cyclic graph: the loss of a decision touches the edge weight, the two nodes'
states and the two nodes' activation parameters, and nothing further.  No
gradient crosses a second edge, so there is no product chain and no vanishing
gradient to fight (`Research/VanishingGradientIsAFeature.md` is the road not
taken here, the same way ``radixnet`` does not take it).

Everything above the character level - which tree gets which text - is
:mod:`fbradix.bank`'s business.  An expert does not know it is one of several.
"""

from __future__ import annotations

import math
import random

from .activation import DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K, MIN_B

Z_RANGE = 4.5
"""New node state ``z ~ U(-Z_RANGE, Z_RANGE)`` - ``radixnet.graph.Z_RANGE``."""

W_LOW, W_HIGH = 0.5, 1.5
"""New edge weights ``~ U(W_LOW, W_HIGH)`` - ``radixnet.graph``'s draw."""

DEPTH = 5
"""Default context depth: windows of ``DEPTH + 1`` characters are inserted."""

ALPHA = 2.0
"""Interpolation constant: a context seen ``c`` times is trusted ``c/(c+ALPHA)``.

Every prediction is a mixture of what *this* expert knows at this context and
what the shared prior knows, weighted by how often this expert has actually
been here.  Witten-Bell's rule, and it is load-bearing for the routing rather
than only for the bits: without it an expert that has never seen a context pays
the floor - eleven bits a character - while an expert that knows the context but
is unsure pays four, and the filter learns to route text to the tree that knows
*least* about it whenever the floor happens to be kinder than a diffuse guess.
Interpolating makes that impossible by construction: a prediction can never be
worse than ``(1 - own)`` of the prior's.
"""

FLOOR = 0.02
"""Smoothing floor: the model spends ``FLOOR`` of its mass on the uniform
distribution over the alphabet, so no held-out character ever costs infinity.
It is a constant, identical in every arm of the experiment, which is what makes
the arms' bits/char comparable."""

CLIP = 5.0
ACT_RATE = 0.1
"""Activation parameters move at a tenth of the weights' rate (section 8)."""


class RadixTreeNet:
    """A path-compressed context trie with a learnable sine on every node."""

    __slots__ = (
        "depth", "floor", "alphabet", "min_count", "rng",
        "seg", "kids", "par", "cnt", "w", "z", "a", "b", "h", "k",
        "chars", "decisions", "fallback", "pcache",
    )

    def __init__(
        self,
        depth: int = DEPTH,
        alphabet: int = 128,
        seed: int = 0,
        floor: float = FLOOR,
        min_count: int = 1,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if not 0.0 <= floor < 1.0:
            raise ValueError(f"floor must be in [0, 1), got {floor}")
        self.depth = int(depth)
        self.floor = float(floor)
        self.alphabet = int(alphabet) + 1  # one slot reserved for the unseen
        self.min_count = int(min_count)
        self.rng = random.Random(seed)
        self.seg: list[str] = [""]
        self.kids: list[dict[str, int]] = [{}]
        self.par: list[int] = [-1]
        self.cnt: list[int] = [0]
        self.w: list[float] = [1.0]
        self.z: list[float] = [0.0]
        self.a: list[float] = [DEFAULT_A]
        self.b: list[float] = [DEFAULT_B]
        self.h: list[float] = [DEFAULT_H]
        self.k: list[float] = [DEFAULT_K]
        self.chars = 0
        self.pcache: dict[int, list[tuple[str, int, float]]] = {}
        """Per-node softmax cache, emptied by every update (see :meth:`step`)."""
        self.decisions: list[tuple[int, int]] | None = None
        self.fallback: "RadixTreeNet | None" = None
        """Consulted when this tree knows nothing at all about a context.

        :mod:`fbradix.bank` attaches one shallow tree, built over every
        training segment, to every expert.  It is what an expert that has been
        given no text falls back on, and without it the difference between an
        empty address and a trained one is a ten-bit cliff that drowns out the
        difference the filter is trying to learn."""

    # -- structure -----------------------------------------------------------

    def _new(self, segment: str, parent: int) -> int:
        """Allocate a node holding ``segment``, with a fresh state and in-edge."""
        nid = len(self.seg)
        self.seg.append(segment)
        self.kids.append({})
        self.par.append(parent)
        self.cnt.append(0)
        self.w.append(self.rng.uniform(W_LOW, W_HIGH))
        self.z.append(self.rng.uniform(-Z_RANGE, Z_RANGE))
        self.a.append(DEFAULT_A)
        self.b.append(DEFAULT_B)
        self.h.append(DEFAULT_H)
        self.k.append(DEFAULT_K)
        return nid

    def split(self, u: int, i: int) -> int:
        """Split node ``u`` after ``i`` characters; return the new deep half.

        ``u`` keeps ``seg[:i]`` and the shorter context it now spells, and gets
        a fresh state.  The new node keeps the rest of the run together with
        ``u``'s children, count and learned state - it is the node that still
        spells what ``u`` used to spell.
        """
        s = self.seg[u]
        if not 0 < i < len(s):
            raise ValueError(f"cannot split {s!r} at {i}")
        b = self._new(s[i:], u)
        self.kids[b] = self.kids[u]
        for c in self.kids[b].values():
            self.par[c] = b
        self.cnt[b] = self.cnt[u]
        self.z[b], self.a[b] = self.z[u], self.a[u]
        self.b[b], self.h[b], self.k[b] = self.b[u], self.h[u], self.k[u]
        self.seg[u] = s[:i]
        self.kids[u] = {s[i]: b}
        self.z[u] = self.rng.uniform(-Z_RANGE, Z_RANGE)
        self.a[u], self.b[u] = DEFAULT_A, DEFAULT_B
        self.h[u], self.k[u] = DEFAULT_H, DEFAULT_K
        return b

    def insert(self, s: str) -> int:
        """Insert one window; return the node its walk ends in."""
        node = 0
        pos = 0
        self.cnt[0] += 1
        n = len(s)
        while pos < n:
            ch = s[pos]
            nxt = self.kids[node].get(ch)
            if nxt is None:
                leaf = self._new(s[pos:], node)
                self.kids[node][ch] = leaf
                self.cnt[leaf] = 1
                return leaf
            seg = self.seg[nxt]
            rem = n - pos
            i = 0
            limit = len(seg) if rem >= len(seg) else rem
            while i < limit and seg[i] == s[pos + i]:
                i += 1
            if i < len(seg):
                if i == rem:
                    self.cnt[nxt] += 1  # the window ends inside this run
                    return nxt
                self.split(nxt, i)
            self.cnt[nxt] += 1
            node = nxt
            pos += i if i < len(seg) else len(seg)
        return node

    def insert_text(self, text: str) -> None:
        """Insert every window of ``text`` (length ``depth + 1``, stride 1)."""
        w = self.depth + 1
        n = len(text)
        self.chars += n
        for i in range(n):
            self.insert(text[i : i + w])
        self.decisions = None
        self.pcache.clear()

    # -- walking -------------------------------------------------------------

    def walk(self, s: str) -> tuple[int, int] | None:
        """Locate the exact string ``s``: ``(node, offset into its segment)``.

        ``offset == len(seg[node])`` means the walk landed *on* the node (a
        branch point); a smaller offset means it stopped inside a run, where
        the continuation is fixed.  ``None`` if the tree has never seen ``s``.
        """
        node = 0
        pos = 0
        n = len(s)
        while pos < n:
            nxt = self.kids[node].get(s[pos])
            if nxt is None:
                return None
            seg = self.seg[nxt]
            rem = n - pos
            if rem < len(seg):
                return (nxt, rem) if seg[:rem] == s[pos:] else None
            if seg != s[pos : pos + len(seg)]:
                return None
            node = nxt
            pos += len(seg)
        return (node, len(self.seg[node]))

    def context(self, ctx: str) -> tuple[int, int] | None:
        """The deepest usable context: the longest suffix of ``ctx`` that the
        tree knows, has a continuation for, and has seen ``min_count`` times.

        Backing off by *suffix* is what lets a tree answer at all for a context
        it has never seen: drop the oldest character and ask again.
        """
        if self.depth < len(ctx):
            ctx = ctx[len(ctx) - self.depth :]
        for L in range(len(ctx), -1, -1):
            loc = self.walk(ctx[len(ctx) - L :] if L else "")
            if loc is None:
                continue
            node, off = loc
            if off < len(self.seg[node]):
                if self.cnt[node] >= self.min_count:
                    return loc
            elif self.kids[node] and self.cnt[node] >= self.min_count:
                return loc
        return None

    # -- the sine ------------------------------------------------------------

    def f(self, n: int) -> float:
        """``f_n(z_n)`` - the node's own activation at its own state."""
        return self.a[n] * math.sin(self.b[n] * (self.z[n] - self.h[n])) + self.k[n]

    def _partials(self, n: int) -> tuple[float, float, float, float, float]:
        """``(f, df/dz, df/da, df/db, df/dh)`` at node ``n``; ``df/dk`` is 1."""
        d = self.z[n] - self.h[n]
        u = self.b[n] * d
        su = math.sin(u)
        cu = math.cos(u)
        abc = self.a[n] * self.b[n] * cu
        return (self.a[n] * su + self.k[n], abc, su, self.a[n] * d * cu, -abc)

    def child_probs(self, p: int) -> list[tuple[str, int, float]]:
        """``(first character, child, probability)`` over ``p``'s children.

        Cached: scoring asks the same nodes over and over between updates, and
        :meth:`step` empties the cache whenever a parameter moves.
        """
        hit = self.pcache.get(p)
        if hit is not None:
            return hit
        kids = self.kids[p]
        if not kids:
            return []
        fp = self.f(p)
        items = list(kids.items())
        scores = [self.w[c] * fp * self.f(c) for _, c in items]
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        tot = sum(ex)
        out = [(ch, c, e / tot) for (ch, c), e in zip(items, ex)]
        self.pcache[p] = out
        return out

    # -- scoring -------------------------------------------------------------

    def levels(self, ctx: str):
        """Every context length this tree can answer at, shortest first.

        Yields ``(count, node, offset)`` for the root, then for each longer
        suffix of ``ctx`` that the tree knows.  Presence is monotone - the tree
        holds every window of its own text, so if a suffix is absent every
        longer one is too - which is why this can stop at the first miss
        instead of trying them all.
        """
        out = []
        top = min(self.depth, len(ctx))
        for L in range(0, top + 1):
            loc = self.walk(ctx[len(ctx) - L :] if L else "")
            if loc is None:
                break
            node, off = loc
            if self.cnt[node] < self.min_count:
                break
            out.append((self.cnt[node], node, off))
        return out

    def prob(self, ctx: str, ch: str) -> float:
        """``p(ch | ctx)``: every context length, folded longest-last.

        Each level is mixed into the answer the levels below it gave::

            p_L   = own_L * p_here(L) + (1 - own_L) * p_(L-1)
            own_L = cnt_L / (cnt_L + ALPHA)

        so a long context seen often speaks for itself, a long context seen
        once contributes a third of its opinion, and the chain rests on the
        shared prior.  Folding one level only - the deepest match - is the
        version this started as, and it is worse in both directions: the root's
        count is always enormous, so ``own`` at the root is ~1 and a
        backed-off prediction drowns out the prior it was supposed to defer to.
        The product of ``(1 - own_L)`` down the chain is what stops that, and
        it is why a deep shared prior only becomes worth anything here.
        """
        base = 1.0 / self.alphabet
        p = (
            self.fallback.prob(ctx, ch)
            if self.fallback is not None and self.fallback is not self
            else base
        )
        for cnt, node, off in self.levels(ctx):
            seg = self.seg[node]
            if off < len(seg):
                here = 1.0 if seg[off] == ch else 0.0
            elif self.kids[node]:
                here = 0.0
                for c_ch, _c, q in self.child_probs(node):
                    if c_ch == ch:
                        here = q
                        break
            else:
                continue
            own = cnt / (cnt + ALPHA)
            p = own * here + (1.0 - own) * p
        return (1.0 - self.floor) * p + self.floor * base

    def logprob(self, ctx: str, ch: str) -> float:
        """``log p(ch | ctx)`` in nats."""
        return math.log(self.prob(ctx, ch))

    def bits_per_char(self, texts) -> tuple[float, int]:
        """``(bits per character, characters scored)`` over an iterable of texts."""
        total = 0.0
        n = 0
        inv = 1.0 / math.log(2.0)
        for t in texts:
            for i, ch in enumerate(t):
                total -= self.logprob(t[max(0, i - self.depth) : i], ch) * inv
                n += 1
        return (total / n if n else 0.0, n)

    # -- training ------------------------------------------------------------

    def plan(self, texts) -> list[tuple[int, int]]:
        """The ``(branch node, chosen child)`` decisions ``texts`` ask of the tree.

        One per *level* per position, not one per position: :meth:`prob` mixes
        every context length, so every node that takes part in a prediction is
        trained on it.  Structure alone decides this list, not the weights, so
        it is computed once per built tree and reused by every epoch.  A level
        whose context lands inside a run contributes nothing - the softmax over
        one child is 1 and every gradient is exactly zero.
        """
        out: list[tuple[int, int]] = []
        for t in texts:
            for i, ch in enumerate(t):
                for _cnt, node, off in self.levels(t[max(0, i - self.depth) : i]):
                    if off < len(self.seg[node]):
                        continue
                    kids = self.kids[node]
                    if len(kids) > 1:
                        c = kids.get(ch)
                        if c is not None:
                            out.append((node, c))
        return out

    def step(self, batch: list[tuple[int, int]], lr: float, act_lr: float) -> float:
        """One mini-batch of the one-hop rule; returns the loss *before* the update.

        Gradients are summed over the batch, divided by its size, clipped to
        ``[-CLIP, CLIP]`` and applied once.  ``b`` is floored at ``MIN_B``.
        """
        n = len(batch)
        if n == 0:
            return 0.0
        self.pcache.clear()
        gw: dict[int, float] = {}
        gz: dict[int, float] = {}
        ga: dict[int, float] = {}
        gb: dict[int, float] = {}
        gh: dict[int, float] = {}
        gk: dict[int, float] = {}
        cache: dict[int, tuple[float, float, float, float, float]] = {}
        total = 0.0
        for p, target in batch:
            pp = cache.get(p)
            if pp is None:
                pp = cache[p] = self._partials(p)
            fp = pp[0]
            kids = list(self.kids[p].values())
            if len(kids) < 2:
                continue
            facts = []
            scores = []
            for c in kids:
                cc = cache.get(c)
                if cc is None:
                    cc = cache[c] = self._partials(c)
                facts.append(cc)
                scores.append(self.w[c] * fp * cc[0])
            m = max(scores)
            ex = [math.exp(s - m) for s in scores]
            ssum = sum(ex)
            ti = kids.index(target)
            total += m + math.log(ssum) - scores[ti]
            inv = 1.0 / ssum
            gfp = 0.0
            for idx, c in enumerate(kids):
                g = ex[idx] * inv
                if idx == ti:
                    g -= 1.0
                cc = facts[idx]
                fc = cc[0]
                gw[c] = gw.get(c, 0.0) + g * fp * fc
                gfp += g * self.w[c] * fc
                gfc = g * self.w[c] * fp
                gz[c] = gz.get(c, 0.0) + gfc * cc[1]
                ga[c] = ga.get(c, 0.0) + gfc * cc[2]
                gb[c] = gb.get(c, 0.0) + gfc * cc[3]
                gh[c] = gh.get(c, 0.0) + gfc * cc[4]
                gk[c] = gk.get(c, 0.0) + gfc
            gz[p] = gz.get(p, 0.0) + gfp * pp[1]
            ga[p] = ga.get(p, 0.0) + gfp * pp[2]
            gb[p] = gb.get(p, 0.0) + gfp * pp[3]
            gh[p] = gh.get(p, 0.0) + gfp * pp[4]
            gk[p] = gk.get(p, 0.0) + gfp
        scale = 1.0 / n
        for c, g in gw.items():
            self.w[c] -= lr * _clip(g * scale)
        for c, g in gz.items():
            self.z[c] -= lr * _clip(g * scale)
        for c, g in ga.items():
            self.a[c] -= act_lr * _clip(g * scale)
        for c, g in gb.items():
            nb = self.b[c] - act_lr * _clip(g * scale)
            self.b[c] = nb if nb > MIN_B else MIN_B
        for c, g in gh.items():
            self.h[c] -= act_lr * _clip(g * scale)
        for c, g in gk.items():
            self.k[c] -= act_lr * _clip(g * scale)
        return total * scale

    def train(
        self,
        texts,
        epochs: int = 3,
        lr: float = 0.5,
        act_lr: float | None = None,
        batch: int = 64,
        shuffle_seed: int = 0,
    ) -> list[float]:
        """Train on ``texts``; returns the mean decision loss per epoch (nats)."""
        if self.decisions is None:
            self.decisions = self.plan(texts)
        plan = self.decisions
        if not plan:
            return [0.0] * epochs
        if act_lr is None:
            act_lr = lr * ACT_RATE
        rng = random.Random(shuffle_seed)
        order = list(range(len(plan)))
        losses = []
        for _ in range(epochs):
            rng.shuffle(order)
            tot = 0.0
            nb = 0
            for s in range(0, len(order), batch):
                chunk = [plan[i] for i in order[s : s + batch]]
                tot += self.step(chunk, lr, act_lr)
                nb += 1
            losses.append(tot / nb if nb else 0.0)
        return losses

    # -- size ----------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Nodes, the root included."""
        return len(self.seg)

    @property
    def num_edges(self) -> int:
        """Edges - one per node bar the root, since this is a tree."""
        return len(self.seg) - 1

    @property
    def stored_chars(self) -> int:
        """Characters held in the node segments: the tree's real size."""
        return sum(len(s) for s in self.seg)

    @property
    def branches(self) -> int:
        """Nodes with more than one child - the only places a decision is made."""
        return sum(1 for d in self.kids if len(d) > 1)

    def stats(self) -> dict:
        """Sizes and shape, for the results table."""
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "stored_chars": self.stored_chars,
            "branches": self.branches,
            "chars_seen": self.chars,
            "decisions": len(self.decisions) if self.decisions is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"RadixTreeNet(depth={self.depth}, nodes={self.num_nodes}, "
            f"branches={self.branches}, chars={self.chars})"
        )


def _clip(g: float, c: float = CLIP) -> float:
    return c if g > c else (-c if g < -c else g)
