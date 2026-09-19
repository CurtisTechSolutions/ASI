"""The network: one activation filter in front of a bank of radix trees.

    text -> features -> [layer 1: m sine units] -> address -> [tree 2^m] -> p(next char)

Training is four steps repeated, because the thing layer 1 needs in order to
learn - *which expert would have done better* - does not exist until the experts
have been built, and the experts cannot be built until layer 1 has routed:

1. **route**   every training segment gets an address from the filter;
2. **build**   each expert's tree is rebuilt from scratch out of the segments
               addressed to it.  Rebuilt, not extended: a segment that moved
               must leave no trace behind, or every expert slowly becomes the
               corpus and the split means nothing;
3. **train**   each expert runs the one-hop rule over its own text;
4. **refilter** every live expert prices every training segment, and the filter
               is pushed - one hinge per disagreeing bit - towards the address
               that priced it cheapest.

Steps 1-3 are the hard assignment; step 4 is the only place the bank's parts
speak to each other, and what crosses is a scalar - the difference in bits
between two trees - never a gradient.

Two details decide whether any of it works:

**The shared prior.**  Every expert falls back to one shallow tree built over
all of the training text.  Without it an empty address scores about 11 bits a
character against a trained expert's 3, the counterfactual is swamped by that
constant, and the filter spends its updates avoiding a cliff rather than
learning a boundary.  The same prior is attached in every arm, the single-tree
baseline included, so it cannot flatter the bank.

**Addresses die; they are not born.**  A segment only moves to an address whose
tree already predicts it better, and an empty tree predicts nothing better than
the prior, so the live set of addresses is fixed by the first, random routing
and can only shrink.  :meth:`FilterRouter.learn` therefore only ever proposes an
address that is already occupied - comparing against a tree that does not exist
wastes the update - and the experiment reports the live count every round
rather than hiding it.
"""

from __future__ import annotations

import math
import random

from .filter import ActivationFilter, features
from .tree import FLOOR, RadixTreeNet

ADV_SCALE = 1.0
"""An advantage of this many bits/char is a full-size filter step."""

MARGIN = 0.02
"""Advantages smaller than this (bits/char) are noise and move nothing."""

PASSES = 4
"""Hinge passes per refilter step.

Pricing the segments is the expensive half and the experts do not change while
the filter is being fitted, so the prices are computed once and the cheap half
- the hinge itself - is run several times over them.  One pass is one epoch of
a perceptron, and one epoch is not a fit: measured, going from 1 to 4 is worth
more than anything else in this rule."""

BALANCE = 0.5
"""Load pressure, in bits/char per unit of load imbalance (0 disables it)."""

PRIOR_DEPTH = 1
"""The shared fallback is a unigram tree: one node per character, no more."""


# -- routers -------------------------------------------------------------------

class ConstantRouter:
    """Everything to one expert: the single-radix-tree baseline."""

    name = "constant"
    n_experts = 1
    learns = False

    def route(self, text: str, label: str | None = None) -> int:
        return 0

    def learn(self, *args, **kwargs) -> dict:
        return {}

    def prepare(self, texts: list[str]) -> None:
        """Nothing to set up (the hook every router answers)."""

    def stats(self) -> dict:
        return {"router": self.name}


class CycleRouter:
    """Round robin: a split that carries no information about the text.

    The control for "does splitting the corpus help *at all*" - if the learned
    filter cannot beat this, the routing is not what is doing the work.
    """

    name = "roundrobin"
    learns = False

    def __init__(self, n_experts: int = 8) -> None:
        self.n_experts = n_experts
        self.i = -1
        self.seen: dict[str, int] = {}

    def route(self, text: str, label: str | None = None) -> int:
        """Stable: a segment keeps the address it was first given, so this arm
        is a fixed partition rather than one that reshuffles every round."""
        r = self.seen.get(text)
        if r is None:
            self.i += 1
            r = self.seen[text] = self.i % self.n_experts
        return r

    def learn(self, *args, **kwargs) -> dict:
        return {}

    def prepare(self, texts: list[str]) -> None:
        """Nothing to set up."""

    def stats(self) -> dict:
        return {"router": self.name}


class OracleRouter:
    """Route by the true source label - the ceiling, and the only arm that
    is ever told what the labels are."""

    name = "oracle"
    learns = False

    def __init__(self, labels) -> None:
        self.labels = {l: i for i, l in enumerate(sorted(set(labels)))}
        self.n_experts = len(self.labels)

    def route(self, text: str, label: str | None = None) -> int:
        return self.labels.get(label, 0)

    def learn(self, *args, **kwargs) -> dict:
        return {}

    def prepare(self, texts: list[str]) -> None:
        """Nothing to set up."""

    def stats(self) -> dict:
        return {"router": self.name, "labels": self.labels}


class FilterRouter:
    """Layer 1 as a router: the address is the filter's sign pattern.

    ``frozen`` keeps the filter exactly as drawn - a random projection whose
    signs hash the text into buckets.  That arm is the control that says
    whether *learning* the filter is worth anything, the way ``sine-fixed`` is
    the control in `Experiments/ActivationFunctionTest/`.
    """

    learns = True

    def __init__(
        self,
        bits: int = 3,
        seed: int = 0,
        kind: str = "sine",
        frozen: bool = False,
        lr: float = 0.3,
        balance: float = BALANCE,
        margin: float = MARGIN,
        reinforce: float = 0.0,
        act_rate: float | None = None,
        passes: int = PASSES,
        code: str = "sign",
    ) -> None:
        kw = {"bits": bits, "seed": seed, "kind": kind, "code": code}
        if act_rate is not None:
            kw["act_rate"] = act_rate
        self.filter = ActivationFilter(**kw)
        self.frozen = frozen
        self.lr = lr
        self.balance = balance
        self.margin = margin
        self.reinforce = reinforce
        self.passes = int(passes)
        self.rng = random.Random(seed + 991)
        self.learns = not frozen
        self.name = ("frozen-" if frozen else "") + kind + "-" + code
        if act_rate == 0.0:
            self.name += "-fixedwave"

    @property
    def n_experts(self) -> int:
        return self.filter.n_experts

    def route(self, text: str, label: str | None = None) -> int:
        return self.filter.route(features(text))

    def prepare(self, texts: list[str]) -> None:
        """Calibrate the filter on the training features before the first round."""
        self.filter.calibrate([features(t) for t in texts])

    def learn(self, texts: list[str], routes: list[int], experts: list[RadixTreeNet]) -> dict:
        """Fit the filter to the address layer 2 would have chosen.

        Two halves.  First, **pricing**: every live expert scores every training
        segment, and the cheapest one - after a load charge - is the address
        layer 1 should have produced.  Second, **the hinge**: every bit of the
        filter that disagrees with that address is pushed across its own
        boundary, in proportion to how many bits the mistake cost.

        The experts do not change while this runs, so the prices are computed
        once and the hinge is run :attr:`passes` times over them.  One pass is
        one epoch of a perceptron and a perceptron is not fitted in one epoch;
        the second half is nearly free, so there is no reason to spend only one.

        The load charge is what stops the degenerate answer.  Without it the
        cheapest expert for almost every segment is whichever tree holds the
        most text, every segment moves there, and the bank collapses into one
        tree with a bad router in front of it.  ``balance`` prices an expert's
        share of the corpus in bits, so specialisation still wins where it is
        real and crowding does not.

        Where the filter already agrees there is nothing to fix and nothing is
        pushed: ``reinforce`` will, if turned up, spend a fraction of a step
        widening the weakest bit's margin instead, but it is off by default
        because a full-strength push on the segments that are already right is
        a constant force on a boundary with no error to correct - measured, it
        drags clean clusters together (`README.md`, "what went wrong first").
        """
        if self.frozen:
            return {"moved": 0, "considered": 0, "frozen": True}
        live = [i for i, e in enumerate(experts) if e.chars > 0]
        if len(live) < 2:
            return {"moved": 0, "considered": 0, "live": len(live)}
        n = len(texts)
        load = [0.0] * len(experts)
        for r in routes:
            load[r] += 1.0 / n
        # -- price every segment under every live expert (once) --------------
        priced = []
        for text in texts:
            costs = {j: experts[j].bits_per_char([text])[0] + self.balance * load[j] for j in live}
            best = min(costs, key=lambda j: (costs[j], j))
            priced.append((features(text), costs, best))
        # -- hinge passes over the prices (cheap) ----------------------------
        order = list(range(n))
        moved = agreed = considered = 0
        adv_sum = 0.0
        for _ in range(max(1, self.passes)):
            self.rng.shuffle(order)
            for i in order:
                x, costs, best = priced[i]
                route, resp = self.filter.address(x)
                here = costs.get(route, costs[best] + 2.0 * self.margin)
                adv = here - costs[best]
                considered += 1
                adv_sum += adv
                if best == route or adv <= self.margin:
                    agreed += 1
                    if self.reinforce > 0.0:
                        j = self.filter.weakest_bit(resp)
                        y = 1.0 if resp[j] > 0.0 else -1.0
                        if self.filter.push(x, j, y, self.lr * self.reinforce):
                            moved += 1
                    continue
                step = self.lr * min(1.0, adv / ADV_SCALE)
                if self.filter.code == "argmax":
                    # one hinge on the difference: raise the address layer 2
                    # wanted, lower the one that won
                    if self.filter.push_pair(x, best, route, step):
                        moved += 1
                    continue
                for j in range(self.filter.bits):
                    want = (best >> j) & 1
                    if ((route >> j) & 1) != want:
                        if self.filter.push(x, j, 1.0 if want else -1.0, step):
                            moved += 1
        return {
            "moved": moved,
            "agreed": agreed,
            "considered": considered,
            "passes": self.passes,
            "mean_advantage": adv_sum / considered if considered else 0.0,
            "live": len(live),
            "dead_units": self.filter.dead_units(),
            "amp_freq": [round(v, 4) for v in self.filter.amplitude_frequency()],
        }

    def stats(self) -> dict:
        return {
            "router": self.name,
            "bits": self.filter.bits,
            "kind": self.filter.kind,
            "code": self.filter.code,
            "act_rate": self.filter.act_rate,
            "passes": self.passes,
            "dead_units": self.filter.dead_units(),
            "amp_freq": [round(v, 4) for v in self.filter.amplitude_frequency()],
        }


# -- the bank ------------------------------------------------------------------

class FilteredRadixBank:
    """Layer 1 (a router) plus ``n_experts`` radix trees, fitted by rebuilding."""

    def __init__(
        self,
        router,
        depth: int = 5,
        alphabet: int = 128,
        seed: int = 0,
        floor: float = FLOOR,
        min_count: int = 1,
        prior_depth: int = PRIOR_DEPTH,
    ) -> None:
        self.router = router
        self.depth = depth
        self.alphabet = alphabet
        self.seed = seed
        self.floor = floor
        self.min_count = min_count
        self.prior_depth = prior_depth
        self.experts: list[RadixTreeNet] = []
        self.prior: RadixTreeNet | None = None
        self.routes: list[int] = []
        self.history: list[dict] = []

    # -- pieces --------------------------------------------------------------

    def _tree(self, seed_offset: int, depth: int | None = None) -> RadixTreeNet:
        return RadixTreeNet(
            depth=self.depth if depth is None else depth,
            alphabet=self.alphabet,
            seed=self.seed + seed_offset,
            floor=self.floor,
            min_count=self.min_count,
        )

    def build_prior(self, texts: list[str], epochs: int = 2, lr: float = 0.5) -> RadixTreeNet:
        """The shallow tree every expert falls back to, over all the text."""
        prior = self._tree(9_000, depth=self.prior_depth)
        for t in texts:
            prior.insert_text(t)
        prior.train(texts, epochs=epochs, lr=lr)
        return prior

    def assign(self, samples: list[dict]) -> list[int]:
        """Address every sample.  ``CycleRouter`` is stateful, so this is the
        one place routing happens and the order it happens in is fixed."""
        return [self.router.route(s["text"], s.get("source")) for s in samples]

    def build(self, samples: list[dict], routes: list[int]) -> None:
        """Rebuild every expert from the segments currently addressed to it."""
        self.experts = [self._tree(i + 1) for i in range(self.router.n_experts)]
        for s, r in zip(samples, routes):
            self.experts[r].insert_text(s["text"])
        for e in self.experts:
            e.fallback = self.prior

    def train_experts(self, samples: list[dict], routes: list[int], epochs: int, lr: float) -> float:
        """Train each expert on its own segments; returns the mean decision loss."""
        texts: list[list[str]] = [[] for _ in self.experts]
        for s, r in zip(samples, routes):
            texts[r].append(s["text"])
        tot = 0.0
        n = 0
        for e, ts in zip(self.experts, texts):
            if not ts:
                continue
            losses = e.train(ts, epochs=epochs, lr=lr, shuffle_seed=self.seed)
            tot += losses[-1] * len(ts)
            n += len(ts)
        return tot / n if n else 0.0

    # -- the loop ------------------------------------------------------------

    def fit(
        self,
        train: list[dict],
        rounds: int = 4,
        epochs: int = 3,
        lr: float = 0.5,
        test: list[dict] | None = None,
        log=None,
    ) -> list[dict]:
        """Route, build, train, refilter - ``rounds`` times.

        The filter is not updated after the last round: there would be no
        rebuild left to act on it, and the reported numbers must belong to the
        routing that produced the trees.
        """
        texts = [s["text"] for s in train]
        self.router.prepare(texts)
        self.prior = self.build_prior(texts)
        if not getattr(self.router, "learns", False):
            rounds = 1  # a router that cannot learn partitions the same way every round
        for r in range(rounds):
            routes = self.assign(train)
            self.build(train, routes)
            loss = self.train_experts(train, routes, epochs, lr)
            self.routes = routes
            rec = {
                "round": r,
                "decision_loss": loss,
                "train_bits": self.bits_per_char(train, routes)[0],
                **self.size_stats(),
                **self.load_stats(routes),
            }
            if test is not None:
                rec["test_bits"] = self.bits_per_char(test)[0]
            if r + 1 < rounds:
                rec["filter"] = self.router.learn(texts, routes, self.experts)
            self.history.append(rec)
            if log:
                log(rec)
        return self.history

    # -- scoring -------------------------------------------------------------

    def bits_per_char(self, samples: list[dict], routes: list[int] | None = None) -> tuple[float, int]:
        """Bits per character over ``samples``, each scored by its own expert."""
        if routes is None:
            routes = self.assign(samples)
        total = 0.0
        n = 0
        for s, r in zip(samples, routes):
            b, c = self.experts[r].bits_per_char([s["text"]])
            total += b * c
            n += c
        return (total / n if n else 0.0, n)

    def predict(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """The ``k`` most likely next characters after ``text``, from its expert."""
        e = self.experts[self.router.route(text, None)]
        ctx = text[-self.depth :]
        loc = e.context(ctx)
        if loc is None:
            return []
        node, off = loc
        seg = e.seg[node]
        if off < len(seg):
            return [(seg[off], 1.0 - e.floor)]
        out = [(ch, p) for ch, _c, p in e.child_probs(node)]
        out.sort(key=lambda t: -t[1])
        return out[:k]

    def generate(self, prefix: str, length: int = 60, seed: int = 0) -> str:
        """Sample ``length`` characters, re-routing after every one.

        Re-routing mid-string is the point of the demonstration: the address is
        recomputed from what has been written so far, so the bank can change
        its mind about which expert is writing.
        """
        rng = random.Random(seed)
        out = prefix
        for _ in range(length):
            cand = self.predict(out, k=64)
            if not cand:
                break
            total = sum(p for _, p in cand)
            x = rng.random() * total
            acc = 0.0
            for ch, p in cand:
                acc += p
                if x <= acc:
                    out += ch
                    break
            else:
                out += cand[0][0]
        return out

    # -- statistics ----------------------------------------------------------

    def size_stats(self) -> dict:
        """Total structure across the bank, plus the shared prior's share."""
        nodes = sum(e.num_nodes for e in self.experts)
        chars = sum(e.stored_chars for e in self.experts)
        return {
            "nodes": nodes,
            "edges": sum(e.num_edges for e in self.experts),
            "stored_chars": chars,
            "branches": sum(e.branches for e in self.experts),
            "prior_nodes": self.prior.num_nodes if self.prior else 0,
        }

    def load_stats(self, routes: list[int] | None = None) -> dict:
        """How the addresses are used: how many are live, and how evenly."""
        routes = self.routes if routes is None else routes
        k = self.router.n_experts
        counts = [0] * k
        for r in routes:
            counts[r] += 1
        n = len(routes) or 1
        live = sum(1 for c in counts if c)
        ent = 0.0
        for c in counts:
            if c:
                p = c / n
                ent -= p * math.log(p, 2)
        return {
            "live_experts": live,
            "load": counts,
            "balance": ent / math.log(k, 2) if k > 1 else 1.0,
        }

    def to_dict(self) -> dict:
        """Enough to describe the bank in a result file (not the trees)."""
        return {
            "depth": self.depth,
            "alphabet": self.alphabet,
            "floor": self.floor,
            "min_count": self.min_count,
            "experts": self.router.n_experts,
            **self.router.stats(),
            **self.size_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"FilteredRadixBank(router={self.router.name!r}, "
            f"experts={self.router.n_experts}, depth={self.depth})"
        )
