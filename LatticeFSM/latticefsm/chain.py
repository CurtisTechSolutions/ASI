"""A chain of nodes that predicts the next input, traversing only as deep as it needs to.

One input to a node: reading ``"abc"``, input node 1 holds ``a``, node 2 ``b``, node 3 ``c``, and the
**output node** after them predicts what comes next from the nodes before it.  A chain of ``nodes`` nodes has
``nodes - 1`` input nodes and one output node.  The inputs are letters by default (``char``), or any unit an
N-gram model reads (``word``, ``sound``, ``base64``).

Every input node, given what it and the nodes after it hold (its *context*: the last ``depth`` inputs), keeps
what it has seen come next and its own guess (the most frequent of those), and it is *tested* every time it is
reached while training: its guess either was the next input or it was not.  Its **confidence** is

    confidence = (right + rewarded) / (tested + rewarded + punished + 2)

- its accuracy and its rewards for, its misses and its punishments against, with two imagined misses so that a
node has to earn it: a node never tested has confidence 0.

To predict, the chain is walked from the output node back through the input nodes, nearest first (depth 0 is the
output node alone, depth 1 adds input node ``nodes - 1``, depth 2 the one before it, ...), gathering confidence
as it goes - the walk's confidence after nodes ``0..k`` is

    C_k = 1 - (1 - c_0)(1 - c_1)...(1 - c_k)

and **the walk stops as soon as it reaches the chain's ``confidence`` threshold**.  Every node's confidence
shortens the walk, continuously: the more confident the nodes near the output, the fewer are traversed, and as
training (and reward) makes them more confident the paths keep getting shorter; the less confident, the further
the walk has to go.  A node never reached, or running out of inputs, ends the walk too.  At ``confidence = 0``
the output node alone predicts; at ``confidence = 1`` the walk always goes as deep as it can (no node is ever
fully confident).

What is predicted is the distribution of the node the walk stopped at, smoothed through every node nearer the
output (Witten-Bell)::

    P_k(w) = (c_k(w) + T_k · P_{k-1}(w)) / (N_k + T_k),        P_{-1}(w) = 1 / (V - 1)

``c_k`` the counts of the node at depth ``k``, ``N_k`` their sum, ``T_k`` how many different inputs it has seen
follow, ``V`` the vocabulary with ``<s>`` (which is never predicted).  Every count is an integer and every sum is
taken in one fixed order, so the Rust crate (``chain.rs``) gives the same numbers, and reads and writes the same
files.

Training predicts every input before it learns it (as :class:`latticefsm.ngram.NGram` does), and measures it:
how often the whole chain's first guess was right, the perplexity, and how many nodes it traversed.  Reward and
punishment (:meth:`Chain.credit_text`) go to every node on the walk whose own guess was the output credited: a
rewarded node near the output is trusted sooner, so later walks stop there; a punished one sends them deeper.
"""

from __future__ import annotations

import bisect
import gzip
import json
import math
import random
from collections.abc import Iterable, Sequence

from .ngram import BOS, EOS, UNITS, UNK, detokenize, goal_of, predict_now, tokenize, train_corpus

__all__ = ["Chain", "PRIOR", "load_chain"]

_FORMAT = "latticefsm-chain"
_FORMAT_VERSION = 1
PRIOR = 2.0
"""Imagined misses every node starts with: its confidence is 0 until it is right."""
MAX_NODES = 16


class Node:
    """An input node in one context: what came next (``counts``), its guess (``best``), and its record."""

    __slots__ = ("counts", "total", "best", "tested", "right", "rewarded", "punished")

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.total = 0
        self.best = -1
        self.tested = 0
        self.right = 0
        self.rewarded = 0.0
        self.punished = 0.0

    @property
    def confidence(self) -> float:
        return (self.right + self.rewarded) / (self.tested + self.rewarded + self.punished + PRIOR)

    def add(self, w: int, n: int = 1) -> None:
        """``n`` more ``w`` after this context; the guess stays the most frequent (ties: the lower id)."""
        c = self.counts.get(w, 0) + n
        self.counts[w] = c
        self.total += n
        if self.best < 0:
            self.best = w
        else:
            b = self.counts[self.best]
            if c > b or (c == b and w < self.best):
                self.best = w


class Chain:
    """``nodes - 1`` input nodes and an output node over ``unit`` inputs, walked only as deep as confidence needs."""

    def __init__(self, nodes: int = 8, unit: str = "char", lower: bool = True, confidence: float = 0.9) -> None:
        if not (2 <= nodes <= MAX_NODES):
            raise ValueError(f"nodes is 2 to {MAX_NODES}, got {nodes}")
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence is 0 to 1, got {confidence}")
        self.nodes = int(nodes)
        self.unit = unit
        self.lower = bool(lower)
        self.confidence = float(confidence)
        self.vocab: list[str] = [BOS, EOS, UNK]
        self.index: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.tables: list[dict[tuple[int, ...], Node]] = [{} for _ in range(self.nodes)]
        """``tables[k]``: the input nodes at depth ``k`` by context (the last ``k`` inputs); ``tables[0]`` holds the
        output node alone, context ``()``."""
        self._ranks: dict[int, list[int]] = {}
        """The output node's counts as buckets (count -> ids in order), for the guess among inputs no deeper node
        has seen."""
        self.clock = 0
        self.sentences = 0
        self.sources: list[list] = []
        self.history: list[dict] = []
        self._seen = 0
        self.online = [0, 0, 0.0, 0]
        """While training with ``predict``: ``[right first guesses, inputs predicted, sum of ln P, nodes traversed]``."""

    @property
    def depth(self) -> int:
        """The most input nodes a walk can traverse."""
        return self.nodes - 1

    # ---- inputs ------------------------------------------------------------------------------------------------

    def tokens(self, text: str) -> list[str]:
        return tokenize(text, self.unit, self.lower)

    def prompt_tokens(self, text: str) -> list[str]:
        if self.unit == "base64":
            from .b64 import encode_prefix
            return list(encode_prefix(text.encode("utf-8")))
        return self.tokens(text)

    def _sequence(self, item: str | Sequence[str]) -> list[str]:
        return self.tokens(item) if isinstance(item, str) else list(item)

    def _id(self, token: str, grow: bool) -> int:
        i = self.index.get(token)
        if i is None:
            if not grow:
                return self.index[UNK]
            i = len(self.vocab)
            self.vocab.append(token)
            self.index[token] = i
        return i

    def history_ids(self, tokens: Sequence[str]) -> list[int]:
        """``<s>`` and the inputs as ids (unknown ones as ``<unk>``): what the input nodes are filled from."""
        return [0] + [self._id(t, False) for t in tokens]

    # ---- the walk ----------------------------------------------------------------------------------------------

    def walk(self, ids: Sequence[int], confidence: float | None = None) -> list[Node]:
        """The nodes a prediction after ``ids`` traverses, the output node first: it stops once the confidence gathered
        on the way (``1 - Π (1 - c)``) reaches the threshold, at a node never reached, or when the inputs or the
        input nodes run out."""
        tau = self.confidence if confidence is None else confidence
        out: list[Node] = []
        node = self.tables[0].get(())
        if node is None:
            return out
        out.append(node)
        doubt = 1.0 - node.confidence
        reach = min(self.depth, len(ids))
        k = 0
        while 1.0 - doubt < tau and k < reach:
            k += 1
            node = self.tables[k].get(tuple(ids[len(ids) - k:]))
            if node is None:
                break
            out.append(node)
            doubt *= 1.0 - node.confidence
        return out

    def _uniform(self) -> float:
        return 1.0 / (len(self.vocab) - 1)

    def _p(self, path: Sequence[Node], w: int) -> float:
        p = self._uniform() if w else 0.0
        for node in path:
            t = len(node.counts)
            p = (node.counts.get(w, 0) + t * p) / (node.total + t)
        return p

    def _dist(self, path: Sequence[Node]) -> list[float]:
        u = self._uniform()
        dist = [0.0] + [u] * (len(self.vocab) - 1)
        for node in path:
            t, n, c = len(node.counts), node.total, node.counts
            dist = [(c.get(w, 0) + t * p) / (n + t) for w, p in enumerate(dist)]
        dist[0] = 0.0
        return dist

    def _guess(self, path: Sequence[Node]) -> int:
        """The likeliest input after ``path`` (ties: the lower id; never ``<s>``) without the whole distribution: an
        input no node past the output node has seen is ranked by the output node's counts alone, so only the seen
        ones and the best of the rest are scored."""
        if not path:
            return 1
        cand: set[int] = set()
        for node in path[1:]:
            cand.update(node.counts)
        for c in sorted(self._ranks, reverse=True):
            outsider = next((w for w in self._ranks[c] if w not in cand), None)
            if outsider is not None:
                cand.add(outsider)
                break
        else:
            outsider = next((w for w in range(1, len(self.vocab)) if w not in cand and w not in path[0].counts),
                            None)
            if outsider is not None:
                cand.add(outsider)
        best, bp = 1, -1.0
        for w in sorted(cand):
            p = self._p(path, w)
            if p > bp:
                best, bp = w, p
        return best

    # ---- training ----------------------------------------------------------------------------------------------

    def train_tokens(self, tokens: Sequence[str], predict: bool = True, credit: float = 0.0,
                     reward: bool = True) -> int:
        """One sequence, ``</s>`` last.  At every position each input node already reached in its context is tested
        (its guess right or not) and then learns the input that came.  With ``predict``, the whole chain first
        predicts it - the walk, its first guess and the probability of the true input added to :attr:`online` -
        and with ``credit`` the nodes on the walk that guessed it are rewarded when the guess was right (unless not
        ``reward``) and punished when it was wrong.  Returns how many inputs were learned."""
        ids = [0] + [self._id(t, True) for t in tokens] + [1]
        for i in range(1, len(ids)):
            w = ids[i]
            if predict and predict_now(self._seen):
                path = self.walk(ids[:i])
                guess = self._guess(path)
                p = self._p(path, w)
                self.online[0] += guess == w
                self.online[1] += 1
                self.online[2] += math.log(max(p, 1e-300))
                self.online[3] += len(path) - 1 if path else 0
                if credit and path and (reward or guess != w):
                    self._credit_path(path, guess, credit if guess == w else -credit)
            if predict:
                self._seen += 1
            for k in range(min(self.depth, i) + 1):
                ctx = tuple(ids[i - k:i])
                node = self.tables[k].get(ctx)
                if node is None:
                    node = self.tables[k][ctx] = Node()
                else:
                    node.tested += 1
                    node.right += node.best == w
                if k == 0:
                    old = node.counts.get(w, 0)
                    if old:
                        bucket = self._ranks[old]
                        bucket.pop(bisect.bisect_left(bucket, w))
                        if not bucket:
                            del self._ranks[old]
                    bisect.insort(self._ranks.setdefault(old + 1, []), w)
                node.add(w)
            self.clock += 1
        self.sentences += 1
        return len(ids) - 1

    def reset_online(self) -> dict:
        """What predicting while training measured since the last reset, and a fresh start."""
        right, n, logp, depth = self.online
        self.online = [0, 0, 0.0, 0]
        self._seen = 0
        if n == 0:
            return {"predicted": 0, "accuracy": None, "perplexity": None, "bits": None, "depth": None}
        return {"predicted": n, "accuracy": right / n, "perplexity": math.exp(-logp / n),
                "bits": -logp / n / math.log(2), "depth": depth / n}

    def train_lines(self, lines: Iterable[str | Sequence[str]], source: str | None = None, predict: bool = True,
                    credit: float = 0.0, reward: bool = True) -> int:
        n = count = 0
        for line in lines:
            toks = self._sequence(line)
            if toks:
                n += self.train_tokens(toks, predict, credit, reward)
                count += 1
        if source is not None:
            self.sources.append([source, count, n])
        return n

    def _targets(self, seqs: Sequence[Sequence[str]]) -> tuple[dict, int]:
        """Every full context of ``seqs`` - all the inputs a walk can reach - with what follows it and how often, and
        how many inputs there are (each sequence's and its ``</s>``)."""
        counts: dict[tuple[int, ...], dict[int, int]] = {}
        n = 0
        for seq in seqs:
            ids = self.history_ids(seq) + [1]
            for i in range(1, len(ids)):
                row = counts.setdefault(tuple(ids[i - min(self.depth, i):i]), {})
                row[ids[i]] = row.get(ids[i], 0) + 1
                n += 1
        return counts, n

    def ceiling(self, seqs: Sequence[Sequence[str]]) -> tuple[int, int]:
        """How many inputs of ``seqs`` (each sequence and its ``</s>``) any first guess could get right, of how many:
        in every full context only the input that follows it most often."""
        counts, n = self._targets(seqs)
        return sum(max(r.values()) for r in counts.values()), n

    def drill(self, seqs: Sequence[Sequence[str]], max_rounds: int = 20, patience: int = 3,
              threshold: float = 1.0) -> dict:
        """Correct the chain on ``seqs`` (already learned) round after round until its first guess after every full
        context is the input that follows it most often - every input right that can be (:meth:`ceiling`) - or
        ``max_rounds``, or ``patience`` rounds without a gain.  A round counts the first guesses that are right, then
        after every full context whose guess is not its most frequent input punishes the nodes on the walk that
        guessed it by ``2 ^ k`` (``k``: the rounds that context has already been corrected), so the walk goes deeper,
        to the nodes that know; and when the deepest node the context reaches does not know - other text it learned
        continues the context otherwise - or knows but is outvoted by the nodes nearer the output, that node is given
        enough more of the input to make it its guess, and ``2 ^ k - 1`` more.  With ``threshold`` below 1 it stops
        once that share of the most possible is right.  Returns ``{rounds: [right, ...], right, of, best, need,
        reached}``."""
        counts, n = self._targets(seqs)
        target = {c: min(r, key=lambda w: (-r[w], w)) for c, r in counts.items()}
        best_possible = sum(max(r.values()) for r in counts.values())
        need = goal_of(threshold, best_possible)
        streak: dict[tuple[int, ...], int] = {}
        rounds: list[int] = []
        best, stale = -1, 0
        while True:
            right = 0
            for seq in seqs:
                ids = self.history_ids(seq) + [1]
                for i in range(1, len(ids)):
                    right += self._guess(self.walk(ids[:i])) == ids[i]
            rounds.append(right)
            if right > best:
                best, stale = right, 0
            else:
                stale += 1
            if right >= need or len(rounds) >= max_rounds or stale >= patience:
                break
            for c in sorted(target):
                t = target[c]
                path = self.walk(list(c))
                g = self._guess(path)
                if g != t:
                    k = streak.get(c, 0)
                    for node in path:
                        if node.best != t:
                            node.punished += 2.0 ** k
                    deep = self.tables[len(c)].get(c)
                    if deep is not None and (deep.best != t or len(path) == len(c) + 1):
                        extra = 0 if deep.best == t else deep.counts[deep.best] - deep.counts.get(t, 0) + (t > deep.best)
                        if extra + 2 ** k - 1 > 0:
                            deep.add(t, extra + 2 ** k - 1)
                    streak[c] = k + 1
        return {"rounds": rounds, "right": right, "of": n, "best": best_possible, "need": need, "reached": right >= need}

    def train_paths(self, paths: Iterable[str], held_out: int = 0, predict: bool = True,
                    credit: float = 0.0, reward: bool = True) -> tuple[int, list[list[str]]]:
        """Train on every sequence under ``paths``, streamed (:func:`latticefsm.ngram.train_corpus`), every
        ``held_out``-th of each document kept back and returned."""
        return train_corpus(paths, self.unit, self.lower, held_out,
                            lambda t: self.train_tokens(t, predict, credit, reward), self.sources)

    # ---- predicting --------------------------------------------------------------------------------------------

    def explain(self, text: str | Sequence[str] = "", k: int = 10, confidence: float | None = None) -> dict:
        """What the chain does after ``text``: the inputs in the input nodes (``inputs[j]`` in node ``j + 1``; ``<s>``
        before the first), the walk - for every node traversed, the output node first, its depth, node number,
        input, confidence, the confidence gathered up to it, guess and how often it was reached - how deep it went,
        and the ``k`` likeliest next inputs."""
        toks = self.prompt_tokens(text) if isinstance(text, str) else list(text)
        ids = self.history_ids(toks)
        shown = ids[max(0, len(ids) - self.depth):]
        inputs = [self.vocab[i] for i in shown]
        path = self.walk(ids, confidence)
        steps = []
        doubt = 1.0
        for d, node in enumerate(path):
            doubt *= 1.0 - node.confidence
            steps.append({"depth": d, "node": self.nodes - d,
                          "input": self.vocab[ids[len(ids) - d]] if d else None,
                          "confidence": node.confidence, "gathered": 1.0 - doubt, "guess": self.vocab[node.best] if node.best >= 0 else None,
                          "seen": node.total, "tested": node.tested, "right": node.right,
                          "rewarded": node.rewarded, "punished": node.punished})
        dist = self._dist(path) if path else [0.0] + [self._uniform()] * (len(self.vocab) - 1)
        best = sorted(range(1, len(dist)), key=lambda w: (-dist[w], w))[:k]
        return {"nodes": self.nodes, "inputs": [None] * (self.depth - len(inputs)) + inputs,
                "walk": steps, "depth": max(0, len(path) - 1),
                "confidence": self.confidence if confidence is None else confidence,
                "next": [(self.vocab[w], dist[w]) for w in best]}

    def predict(self, text: str | Sequence[str] = "", k: int = 10) -> list[tuple[str, float]]:
        return self.explain(text, k)["next"]

    def continue_tokens(self, toks: Sequence[str], max_tokens: int = 30, temperature: float = 0.0,
                        rng: random.Random | None = None) -> tuple[list[str], list[int]]:
        """Inputs continuing ``toks`` until ``</s>`` or ``max_tokens`` (never ``<s>`` or ``<unk>``): the likeliest each
        time at temperature 0, else drawn from ``P ^ (1 / temperature)``; and how deep each walk went."""
        ids = self.history_ids(toks)
        out: list[str] = []
        depths: list[int] = []
        rng = rng or random.Random(1)
        for _ in range(max_tokens):
            path = self.walk(ids)
            dist = self._dist(path) if path else [0.0] + [self._uniform()] * (len(self.vocab) - 1)
            dist[0] = dist[2] = 0.0
            if temperature <= 0:
                w = max(range(len(dist)), key=lambda i: (dist[i], -i))
            else:
                weights = [p ** (1.0 / temperature) if p > 0 else 0.0 for p in dist]
                r, acc, w = rng.random() * sum(weights), 0.0, 1
                for i, q in enumerate(weights):
                    acc += q
                    if r < acc:
                        w = i
                        break
            depths.append(max(0, len(path) - 1))
            if w == 1:
                break
            out.append(self.vocab[w])
            ids.append(w)
        return out, depths

    def complete(self, prompt: str = "", max_tokens: int = 30, temperature: float = 0.0,
                 rng: random.Random | None = None) -> str:
        toks, _ = self.continue_tokens(self.prompt_tokens(prompt), max_tokens, temperature, rng)
        if self.unit == "base64":
            return detokenize(self.prompt_tokens(prompt) + toks, self.unit)
        return detokenize(toks, self.unit)

    # ---- measuring ---------------------------------------------------------------------------------------------

    def evaluate(self, lines: Iterable[str | Sequence[str]], confidence: float | None = None) -> dict:
        """On held-out ``lines``: perplexity and bits over every input and ``</s>``, how often the first guess was
        right, and how many input nodes the walks traversed (on average, and how many walks stopped at each depth).
        Nothing moves."""
        logp, n, right, depth = 0.0, 0, 0, 0
        depths = [0] * self.nodes
        for line in lines:
            toks = self._sequence(line)
            if not toks:
                continue
            ids = self.history_ids(toks) + [1]
            for i in range(1, len(ids)):
                w = ids[i]
                path = self.walk(ids[:i], confidence)
                d = max(0, len(path) - 1)
                logp += math.log(max(self._p(path, w) if path else self._uniform(), 1e-300))
                right += self._guess(path) == w
                depth += d
                depths[d] += 1
                n += 1
        if n == 0:
            return {"tokens": 0, "perplexity": None, "bits": None, "accuracy": None, "depth": None, "depths": depths}
        return {"tokens": n, "perplexity": math.exp(-logp / n), "bits": -logp / n / math.log(2),
                "accuracy": right / n, "depth": depth / n, "depths": depths}

    def sweep(self, lines: Sequence[str | Sequence[str]], thresholds: Sequence[float]) -> list[dict]:
        """:meth:`evaluate` at every confidence threshold: how deep the walks go against how well they predict."""
        return [{"confidence": t, **self.evaluate(lines, t)} for t in thresholds]

    # ---- credit ------------------------------------------------------------------------------------------------

    @staticmethod
    def _credit_path(path: Sequence[Node], w: int, amount: float) -> int:
        """Every node on a walk whose own guess was ``w`` rewarded (``amount > 0``) or punished: a rewarded node near
        the output makes the next walk through it shorter, a punished one longer.  Returns how many."""
        n = 0
        for node in path:
            if node.best == w:
                if amount > 0:
                    node.rewarded += amount
                elif amount < 0:
                    node.punished -= amount
                n += 1
        return n

    def credit_text(self, text: str | Sequence[str], amount: float) -> int:
        """Every input of a text, ``</s>`` included, as a prediction: the nodes on the walk to it that guessed it
        rewarded (``amount > 0``) or punished.  Returns how many nodes were credited."""
        toks = self.tokens(text) if isinstance(text, str) else list(text)
        ids = self.history_ids(toks) + [1]
        n = 0
        for i in range(1, len(ids)):
            n += self._credit_path(self.walk(ids[:i]), ids[i], amount)
        return n

    def score_tokens(self, toks: Sequence[str]) -> list[tuple[str, float, int]]:
        """Every input of ``toks`` and the ``</s>`` after it, with the probability the chain gives it as it reads them
        and how deep the walk to it went."""
        toks = list(toks)
        ids = self.history_ids(toks) + [1]
        out = []
        for i in range(1, len(ids)):
            path = self.walk(ids[:i])
            p = self._p(path, ids[i]) if path else self._uniform()
            out.append((EOS if i == len(ids) - 1 else toks[i - 1], p, max(0, len(path) - 1)))
        return out

    def teach(self, text: str | Sequence[str], reply: str | Sequence[str], max_rounds: int = 100,
              threshold: float = 1.0) -> dict:
        """Teach the chain that ``reply`` follows ``text``, round after round, until its first guess is right on every
        input of the reply and on the ``</s>`` after it (``of`` of them) - its greedy continuation of ``text`` is then
        ``reply`` - or ``max_rounds`` rounds have passed.  A round punishes, at every wrong guess, the nodes on the walk
        that guessed it (so the next walk goes deeper, past them) - by 1 the first round a position is wrong, doubling
        every round it stays wrong, so even a node right a thousand times elsewhere gives way - and then learns
        ``text + reply`` as one sequence.  With ``threshold`` below 1 it stops once that share of them is
        right.  Returns ``{right, of, need, rounds, reached, history}``."""
        ctx = self._sequence(text)
        rep = self._sequence(reply)
        of = len(rep) + 1
        need = goal_of(threshold, of)
        history: list[int] = []
        streak = [0] * of
        rounds = 0
        while True:
            right = 0
            wrong = []
            ids = self.history_ids(ctx)
            for i, t in enumerate(rep + [EOS]):
                w = 1 if i == len(rep) else self._id(t, False)
                path = self.walk(ids)
                g = self._guess(path)
                if g == w:
                    right += 1
                    streak[i] = 0
                else:
                    wrong.append((list(ids), g, 2.0 ** streak[i]))
                    streak[i] += 1
                ids.append(w)
            history.append(right)
            if right >= need or rounds >= max_rounds:
                break
            for hist, g, amount in wrong:
                self._credit_path(self.walk(hist), g, -amount)
            self.train_tokens(ctx + rep, predict=False)
            rounds += 1
        return {"right": right, "of": of, "need": need, "rounds": rounds, "reached": right >= need, "history": history}

    # ---- describing --------------------------------------------------------------------------------------------

    def stats(self) -> dict:
        per = []
        for k, table in enumerate(self.tables):
            tested = sum(nd.tested for nd in table.values())
            right = sum(nd.right for nd in table.values())
            per.append({"depth": k, "contexts": len(table), "tested": tested,
                        "accuracy": right / tested if tested else None})
        return {"nodes": self.nodes, "unit": self.unit, "lower": self.lower, "confidence": self.confidence,
                "vocab": len(self.vocab) - 3, "contexts": sum(len(t) for t in self.tables), "tokens": self.clock,
                "sentences": self.sentences, "sources": len(self.sources),
                "credited": sum(1 for t in self.tables for nd in t.values() if nd.rewarded or nd.punished),
                "depths": per}

    # ---- the file ----------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        rows = []
        for k, table in enumerate(self.tables):
            for ctx, nd in table.items():
                counts = [x for w in sorted(nd.counts) for x in (w, nd.counts[w])]
                rows.append([k, *ctx, nd.tested, nd.right, nd.rewarded, nd.punished, *counts])
        rows.sort(key=lambda r: r[:r[0] + 1])
        return {"format": _FORMAT, "version": _FORMAT_VERSION, "nodes": self.nodes, "unit": self.unit,
                "lower": self.lower, "confidence": self.confidence, "clock": self.clock,
                "sentences": self.sentences, "vocab": self.vocab, "table": rows, "sources": self.sources,
                "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> Chain:
        if d.get("format") != _FORMAT:
            raise ValueError(f"not a {_FORMAT} file")
        m = cls(int(d["nodes"]), d.get("unit", "char"), bool(d.get("lower", True)), float(d.get("confidence", 0.9)))
        m.vocab = list(d["vocab"])
        if m.vocab[:3] != [BOS, EOS, UNK]:
            raise ValueError("the vocabulary does not start <s> </s> <unk>")
        m.index = {t: i for i, t in enumerate(m.vocab)}
        for row in d.get("table", []):
            k = int(row[0])
            nd = Node()
            ctx = tuple(int(x) for x in row[1:k + 1])
            nd.tested, nd.right = int(row[k + 1]), int(row[k + 2])
            nd.rewarded, nd.punished = float(row[k + 3]), float(row[k + 4])
            rest = row[k + 5:]
            for j in range(0, len(rest), 2):
                w, c = int(rest[j]), int(rest[j + 1])
                nd.counts[w] = c
                nd.total += c
                if nd.best < 0 or c > nd.counts[nd.best] or (c == nd.counts[nd.best] and w < nd.best):
                    nd.best = w
                if k == 0:
                    bisect.insort(m._ranks.setdefault(c, []), w)
            m.tables[k][ctx] = nd
        m.clock = int(d.get("clock", 0))
        m.sentences = int(d.get("sentences", 0))
        m.sources = list(d.get("sources", []))
        m.history = list(d.get("history", []))
        return m

    def save(self, path: str) -> None:
        text = json.dumps(self.to_dict(), separators=(",", ":"))
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as f:
            f.write(text)


def load_chain(path: str) -> Chain:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return Chain.from_dict(json.load(f))
